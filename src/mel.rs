//! NeMo `AudioToMelSpectrogramPreprocessor` front end, reproduced step by step.
//!
//! The window and the mel filterbank come from the checkpoint itself
//! (`preprocessor.featurizer.window` / `.fb`), so nothing is regenerated and no
//! constant is guessed. The recipe is documented in `notes/mel_spec.md` and was
//! verified against NeMo output with a maximum absolute difference of exactly 0.
//!
//! Steps, in NeMo's order:
//! pre-emphasis 0.97 -> STFT (hann 400, hop 160, n_fft 512, center, constant pad)
//! -> magnitude -> power 2.0 -> mel filterbank -> log(x + 2**-24)
//! -> per-feature normalization over the first `seq_len` frames with the last
//! frames zeroed.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};

use crate::torch::Store;

/// Sample rate the model was trained on.
pub const SAMPLE_RATE: usize = 16_000;
/// Analysis window length in samples (25 ms).
pub const WIN_LENGTH: usize = 400;
/// Frame stride in samples (10 ms).
pub const HOP_LENGTH: usize = 160;
/// FFT size; the 400-sample window is centred inside it.
pub const N_FFT: usize = 512;
/// Mel bins.
pub const N_MELS: usize = 128;
/// Pre-emphasis coefficient.
const PREEMPH: f32 = 0.97;
/// Guard added before the log, `2**-24`.
const LOG_GUARD: f32 = 5.960_464_5e-8;
/// Epsilon added to the per-feature standard deviation.
const NORM_EPS: f32 = 1e-5;

pub struct MelFrontend {
    window: Vec<f32>,
    /// (band index, weight) pairs for the non-zero entries of each mel filter.
    ///
    /// NeMo multiplies by the dense `[128, 257]` filterbank; skipping the zero
    /// weights keeps the same sum of the same non-zero terms, so the result is
    /// unchanged while the inner loop shrinks by an order of magnitude.
    sparse: Vec<Vec<(usize, f32)>>,
    bins: usize,
    left_pad: usize,
    inner_pad: usize,
    fft: Arc<dyn Fft<f32>>,
}

impl MelFrontend {
    pub fn from_store(store: &Store) -> Result<Self> {
        let window = store
            .f32("preprocessor.featurizer.window")
            .context("read mel window")?;
        if window.len() != WIN_LENGTH {
            bail!(
                "expected a {WIN_LENGTH}-sample window, found {}",
                window.len()
            );
        }
        let fb = store
            .f32("preprocessor.featurizer.fb")
            .context("read mel filterbank")?;
        let bins = N_FFT / 2 + 1;
        if fb.len() != N_MELS * bins {
            bail!(
                "expected a [{N_MELS}, {bins}] filterbank, found {} values",
                fb.len()
            );
        }
        let mut sparse = Vec::with_capacity(N_MELS);
        for bin in 0..N_MELS {
            let row = &fb[bin * bins..(bin + 1) * bins];
            let mut entries = Vec::new();
            for (band, weight) in row.iter().enumerate() {
                if *weight != 0.0 {
                    entries.push((band, *weight));
                }
            }
            sparse.push(entries);
        }

        let fft = FftPlanner::new().plan_fft_forward(N_FFT);
        Ok(Self {
            window,
            sparse,
            bins,
            // `center=True` pads n_fft/2 on both sides; the 400-sample window is
            // centred inside the 512-point frame.
            left_pad: N_FFT / 2,
            inner_pad: (N_FFT - WIN_LENGTH) / 2,
            fft,
        })
    }

    /// `get_seq_len`: the number of frames NeMo considers valid.
    pub fn seq_len(&self, samples: usize) -> usize {
        samples / HOP_LENGTH
    }

    /// Frame count of the padded STFT, one more than [`Self::seq_len`].
    pub fn frames(&self, samples: usize) -> usize {
        samples / HOP_LENGTH + 1
    }

    /// Computes `[N_MELS, frames]` log-mel features, row-major.
    pub fn compute(&self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.is_empty() {
            bail!("cannot compute mel features for empty audio");
        }
        let seq_len = self.seq_len(samples.len());
        let frames = self.frames(samples.len());

        // 1. Pre-emphasis: y[0] = x[0], y[t] = x[t] - 0.97 * x[t-1].
        let mut signal = Vec::with_capacity(samples.len());
        signal.push(samples[0]);
        for index in 1..samples.len() {
            signal.push(samples[index] - PREEMPH * samples[index - 1]);
        }

        // 2. STFT with `center=True, pad_mode="constant"`.
        let mut scratch = vec![Complex32::new(0.0, 0.0); self.fft.get_inplace_scratch_len()];
        let mut frame = vec![Complex32::new(0.0, 0.0); N_FFT];
        let mut power = vec![0.0f32; self.bins * frames];
        for index in 0..frames {
            let start = index * HOP_LENGTH;
            for (slot, cell) in frame.iter_mut().enumerate() {
                let sample_index = start as isize + slot as isize - self.left_pad as isize;
                let sample = if sample_index >= 0 && (sample_index as usize) < signal.len() {
                    signal[sample_index as usize]
                } else {
                    0.0
                };
                let window = if slot >= self.inner_pad && slot < self.inner_pad + WIN_LENGTH {
                    self.window[slot - self.inner_pad]
                } else {
                    0.0
                };
                *cell = Complex32::new(sample * window, 0.0);
            }
            self.fft.process_with_scratch(&mut frame, &mut scratch);
            for bin in 0..self.bins {
                let value = frame[bin];
                // Mag then squared, in NeMo's order.
                let magnitude = (value.re * value.re + value.im * value.im).sqrt();
                power[bin * frames + index] = magnitude * magnitude;
            }
        }

        // 3. Filterbank, log, per-feature normalization.
        let mut mel = vec![0.0f32; N_MELS * frames];
        for (bin, entries) in self.sparse.iter().enumerate() {
            for index in 0..frames {
                let mut sum = 0.0f32;
                for &(band, weight) in entries {
                    sum += weight * power[band * frames + index];
                }
                mel[bin * frames + index] = (sum + LOG_GUARD).ln();
            }
        }

        if seq_len == 0 {
            bail!("audio is shorter than one frame");
        }
        for bin in 0..N_MELS {
            let row = &mut mel[bin * frames..(bin + 1) * frames];
            let mean = row[..seq_len].iter().sum::<f32>() / seq_len as f32;
            let mut variance = 0.0f32;
            for value in &row[..seq_len] {
                let delta = value - mean;
                variance += delta * delta;
            }
            let std = (variance / (seq_len as f32 - 1.0)).sqrt() + NORM_EPS;
            for value in row.iter_mut() {
                *value = (*value - mean) / std;
            }
            for value in row[seq_len..].iter_mut() {
                *value = 0.0;
            }
        }

        Ok(mel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_geometry_matches_nemo() {
        let frontend = MelFrontend {
            window: vec![0.0; WIN_LENGTH],
            sparse: vec![Vec::new(); N_MELS],
            bins: N_FFT / 2 + 1,
            left_pad: N_FFT / 2,
            inner_pad: (N_FFT - WIN_LENGTH) / 2,
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        };
        // The reference clip: 93680 samples -> 585 valid frames, 586 STFT frames.
        assert_eq!(frontend.seq_len(93_680), 585);
        assert_eq!(frontend.frames(93_680), 586);
        assert_eq!(frontend.seq_len(16_000), 100);
        assert_eq!(frontend.frames(16_000), 101);
    }
}
