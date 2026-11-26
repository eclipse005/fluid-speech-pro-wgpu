//! NeMo `ConvSubsampling` (`dw_striding`) — the encoder's pre_encode stage.
//!
//! Structure confirmed from the checkpoint and the module repr:
//!
//! ```text
//! Conv2d(1 -> 256, k=3, s=2, p=1) + ReLU                  conv.0
//! Conv2d(256 -> 256, k=3, s=2, p=1, groups=256)           conv.2   (depthwise)
//! Conv2d(256 -> 256, k=1)                                 conv.3   (pointwise)
//! ReLU
//! Conv2d(256 -> 256, k=3, s=2, p=1, groups=256)           conv.5   (depthwise)
//! Conv2d(256 -> 256, k=1)                                 conv.6   (pointwise)
//! ReLU
//! flatten (channel-major: c * freq + f) -> Linear(4096 -> 1024)  out
//! ```
//!
//! The feature map lives in one contiguous `[time][channel][freq]` buffer. A
//! nested `Vec<Vec<Vec<f32>>>` would allocate once per (time, channel) pair —
//! hundreds of thousands of tiny allocations on a long clip, plus a pointer chase
//! per element inside the kernels. The flat layout also makes the final
//! channel-major flatten a move instead of a transpose.

use anyhow::{Context, Result, bail};

use rayon::prelude::*;

use crate::gemm::{PackW, gemm_into};
use crate::torch::Store;

/// Input mel bins.
pub const FEAT_IN: usize = 128;
/// Convolution channels.
pub const CHANNELS: usize = 256;
/// Output width.
pub const D_MODEL: usize = 1024;

struct Conv2d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    /// Kernel edge: 3 for the strided/depthwise stages, 1 for pointwise.
    kernel: usize,
    stride: usize,
    padding: usize,
    /// Set for the 1x1 stages: a 1x1 conv is a GEMM over channels, so the
    /// weights are repacked once and the per-element tap loop disappears.
    packed: Option<PackW>,
}

/// Time-major feature map `[time][channel][freq]` in one buffer.
struct Grid {
    time: usize,
    channels: usize,
    freqs: usize,
    data: Vec<f32>,
}

impl Grid {
    fn new(time: usize, channels: usize, freqs: usize) -> Self {
        Self {
            time,
            channels,
            freqs,
            data: vec![0.0f32; time * channels * freqs],
        }
    }

    #[inline]
    fn index(&self, time: usize, channel: usize, freq: usize) -> usize {
        (time * self.channels + channel) * self.freqs + freq
    }
}

pub struct PreEncode {
    conv0: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv5: Conv2d,
    conv6: Conv2d,
    out_weight: PackW,
    out_bias: Vec<f32>,
}

impl PreEncode {
    pub fn from_store(store: &Store) -> Result<Self> {
        Ok(Self {
            conv0: read_conv(store, "encoder.pre_encode.conv.0", 1, CHANNELS, 1)?,
            conv2: read_conv(
                store,
                "encoder.pre_encode.conv.2",
                CHANNELS,
                CHANNELS,
                CHANNELS,
            )?,
            conv3: read_conv(store, "encoder.pre_encode.conv.3", CHANNELS, CHANNELS, 1)?,
            conv5: read_conv(
                store,
                "encoder.pre_encode.conv.5",
                CHANNELS,
                CHANNELS,
                CHANNELS,
            )?,
            conv6: read_conv(store, "encoder.pre_encode.conv.6", CHANNELS, CHANNELS, 1)?,
            out_weight: {
                let weight = store
                    .f32("encoder.pre_encode.out.weight")
                    .context("read out.weight")?;
                let rows = store
                    .require("encoder.pre_encode.out.weight")
                    .map(|tensor| tensor.shape.first().copied().unwrap_or(0))
                    .unwrap_or(0);
                let cols = weight.len().checked_div(rows).unwrap_or(0);
                PackW::from_vec(weight, rows, cols)
            },
            out_bias: store
                .f32("encoder.pre_encode.out.bias")
                .context("read out.bias")?,
        })
    }

    /// Returns `[frames][1024]` time-major.
    ///
    /// NeMo's `MaskedConvSequential` masks time steps past the padded length
    /// before every layer and recomputes that length after each strided stage, so
    /// the tail a stride-2 conv produces beyond the real audio is dropped instead
    /// of being fed forward. `frames` counts the STFT frames (`seq_len + 1`); the
    /// last one is padding.
    /// Returns the projected features and the frame count NeMo calls valid.
    ///
    /// The two differ whenever the strided shape is padded, so callers must not
    /// infer the frame count from the buffer length.
    pub fn forward(&self, mel_time_major: &[f32], frames: usize) -> Result<(Vec<f32>, usize)> {
        if mel_time_major.len() != frames * FEAT_IN {
            bail!(
                "expected {frames} frames of {FEAT_IN} mel bins, found {} values",
                mel_time_major.len()
            );
        }
        let total = valid_steps(frames);
        let plan = plan_chunks(frames);
        // One chunk with no margin to drop is the ordinary short-utterance case.
        if plan.len() <= 1 {
            return self.forward_block(mel_time_major, frames);
        }

        // Long audio: each chunk is an ordinary forward over a slice, and the rows
        // its padding distorts are dropped at the seam. Peak memory is bounded by
        // `CHUNK_GRID_BYTES` instead of by the utterance length.
        let mut out = vec![0.0f32; total * D_MODEL];
        for chunk in &plan {
            let start = chunk.start * FEAT_IN;
            let end = (chunk.start + chunk.len) * FEAT_IN;
            let (block, valid) = self.forward_block(&mel_time_major[start..end], chunk.len)?;
            if valid < chunk.keep_from + chunk.rows {
                bail!(
                    "chunk at {} produced {valid} rows, needs {}",
                    chunk.start,
                    chunk.keep_from + chunk.rows
                );
            }
            let from = chunk.keep_from * D_MODEL;
            let take = chunk.rows * D_MODEL;
            let at = chunk.first_row * D_MODEL;
            out[at..at + take].copy_from_slice(&block[from..from + take]);
        }
        Ok((out, total))
    }

    /// One chunk: the conv stack plus the projection, with no chunking of its own.
    fn forward_block(&self, mel_time_major: &[f32], frames: usize) -> Result<(Vec<f32>, usize)> {
        check_frames(frames)?;

        // The mel buffer is already `[time][freq]`, which is this grid with one
        // channel, so the copy is a move.
        let mut grid = Grid::new(frames, 1, FEAT_IN);
        grid.data.copy_from_slice(mel_time_major);

        // `valid` follows the time axis through the strides, so the mask always
        // speaks the current grid's units.
        let conv_started = std::time::Instant::now();
        let mut valid = frames.saturating_sub(1);
        let grid = conv2d(&self.conv0, &grid);
        valid = conv_out_len(valid, &self.conv0);
        let grid = mask_tail(grid, valid);
        let grid = relu(grid);
        let grid = conv2d(&self.conv2, &grid);
        valid = conv_out_len(valid, &self.conv2);
        let grid = mask_tail(grid, valid);
        let grid = conv2d(&self.conv3, &grid);
        valid = conv_out_len(valid, &self.conv3);
        let grid = mask_tail(grid, valid);
        let grid = relu(grid);
        let grid = conv2d(&self.conv5, &grid);
        valid = conv_out_len(valid, &self.conv5);
        let grid = mask_tail(grid, valid);
        let grid = conv2d(&self.conv6, &grid);
        valid = conv_out_len(valid, &self.conv6);
        let grid = mask_tail(grid, valid);
        let grid = relu(grid);

        if std::env::var_os("FSP_PRE_TIMING").is_some() {
            eprintln!("[pre] convs {:.4}s", conv_started.elapsed().as_secs_f64());
        }
        // Set FSP_PRE_TIMING=1 to print where this stage spends its time; the
        // projection below is a large GEMM while the rest are small kernels, so the
        // split decides whether porting to the GPU is worth it.
        let timing = std::env::var_os("FSP_PRE_TIMING").is_some();
        let mark = std::time::Instant::now();

        // `[time][channel][freq]` is exactly NeMo's per-step channel-major
        // flatten, so the projection is one GEMM over all steps.
        // NeMo returns a tensor padded to the strided size but reports the shorter
        // `enc_len`, and its conformer masks attention down to that length. Feeding
        // NeMo's conformer attends only over the frames up to `enc_len`, so the
        // padded row must not reach it: an extra attention key shifts the softmax
        // denominator for every query. Measured both ways — keeping every row gives
        // 1.5e-1 against NeMo, truncating to `valid` gives ~1.5e-3 — so truncation is
        // the correct direction; the residual tail difference is tracked separately.
        let time_steps = valid;
        let flattened = grid.channels * grid.freqs;
        if self.out_weight.cols() != flattened || self.out_weight.rows() != D_MODEL {
            bail!(
                "pre_encode.out is [{}x{}], conv output is {flattened} wide",
                self.out_weight.rows(),
                self.out_weight.cols()
            );
        }
        let mut out = vec![0.0f32; time_steps * D_MODEL];
        gemm_into(
            &grid.data[..time_steps * flattened],
            &self.out_weight,
            &mut out,
            time_steps,
        );
        for step in 0..time_steps {
            let row = &mut out[step * D_MODEL..(step + 1) * D_MODEL];
            for (slot, offset) in row.iter_mut().zip(&self.out_bias) {
                *slot += offset;
            }
        }
        if timing {
            eprintln!("[pre] projection {:.4}s", mark.elapsed().as_secs_f64());
        }
        Ok((out, valid))
    }
}

fn read_conv(
    store: &Store,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
) -> Result<Conv2d> {
    let weight = store
        .f32(&format!("{prefix}.weight"))
        .with_context(|| format!("read {prefix}.weight"))?;
    let bias = store
        .f32(&format!("{prefix}.bias"))
        .with_context(|| format!("read {prefix}.bias"))?;
    let per_group = in_channels / groups;
    let unit = out_channels * per_group;
    // The pointwise stages are 1x1/stride 1; the rest are 3x3/stride 2/pad 1.
    let (kernel, stride, padding) = if weight.len() == unit {
        (1, 1, 0)
    } else if weight.len() == unit * 9 {
        (3, 2, 1)
    } else {
        bail!(
            "{prefix}.weight has {} values, expected {unit} or {}",
            weight.len(),
            unit * 9
        );
    };
    let packed = if kernel == 1 && groups == 1 {
        Some(PackW::from_rows(&weight, out_channels, in_channels))
    } else {
        None
    };
    Ok(Conv2d {
        weight,
        bias,
        in_channels,
        out_channels,
        groups,
        kernel,
        stride,
        padding,
        packed,
    })
}

/// Ceiling for one intermediate grid, in bytes.
///
/// The conv stack materialises `[time][256][64]` grids, so the allocation grows with
/// the utterance. Audio is processed in chunks of at most [`CHUNK_GRID_BYTES`], which
/// keeps this ceiling out of reach for any length; it stays as an assertion that the
/// plan is doing its job.
const MAX_GRID_BYTES: usize = 2 << 30;

/// Mel frames consumed per encoder step. Three stride-2 stages.
pub(crate) const STEP_STRIDE: usize = 8;

/// Input frames one output row depends on: `1 + 2 + 4 + 8`, the two taps each of the
/// three stride-2 kernels counted at their cumulative stride.
const RECEPTIVE_FIELD: usize = 15;

/// Rows at each end of a chunk that its own zero padding distorts.
///
/// The receptive field needs `RECEPTIVE_FIELD / STEP_STRIDE` rounded up, i.e. two
/// rows. One more covers the shift the per-layer tail mask introduces, since zeroing
/// the last row of one stage moves the next stage's last row too.
pub(crate) const CHUNK_MARGIN_ROWS: usize = RECEPTIVE_FIELD.div_ceil(STEP_STRIDE) + 1;

/// Widest intermediate a chunk may allocate.
///
/// conv0 emits `[frames/2][256][64]`, i.e. 32 KB per input frame, so 512 MB allows
/// ~16k frames (about 2.7 minutes) per chunk. Peak memory then depends on this
/// constant rather than on the utterance length.
pub(crate) const CHUNK_GRID_BYTES: usize = 512 << 20;

/// One slice of the mel input and the output rows it is responsible for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Chunk {
    /// First mel frame of the slice.
    pub start: usize,
    /// Mel frames in the slice.
    pub len: usize,
    /// First output row this chunk owns, counted from the start of the utterance.
    pub first_row: usize,
    /// Output rows this chunk owns.
    pub rows: usize,
    /// Rows to discard at the slice's start, which the padding distorted.
    pub keep_from: usize,
}

/// Splits `frames` mel frames into chunks whose peak grid stays under
/// [`CHUNK_GRID_BYTES`], with enough overlap that no kept row sees chunk padding.
///
/// A single chunk is returned when the whole utterance already fits, so short audio
/// keeps taking the one-shot path.
pub(crate) fn plan_chunks(frames: usize) -> Vec<Chunk> {
    let total = valid_steps(frames);
    if total == 0 {
        return Vec::new();
    }
    let rows_per_chunk = (CHUNK_GRID_BYTES / (CHANNELS * 64 * std::mem::size_of::<f32>()))
        / STEP_STRIDE;
    let rows_per_chunk = rows_per_chunk.max(CHUNK_MARGIN_ROWS * 2 + 1);

    let mut plan = Vec::new();
    let mut first_row = 0;
    while first_row < total {
        let end_row = (first_row + rows_per_chunk).min(total);
        let margin = if first_row == 0 { 0 } else { CHUNK_MARGIN_ROWS };
        let start = first_row * STEP_STRIDE - margin * STEP_STRIDE;
        // The final chunk must end where the audio ends: `valid_rows` is defined
        // against the true tail, so a chunk that stopped earlier would report a
        // different length.
        let end = if end_row == total {
            frames
        } else {
            ((end_row + CHUNK_MARGIN_ROWS) * STEP_STRIDE).min(frames)
        };
        plan.push(Chunk {
            start,
            len: end - start,
            first_row,
            rows: end_row - first_row,
            keep_from: first_row - start / STEP_STRIDE,
        });
        first_row = end_row;
    }
    plan
}

/// Bytes the first strided stage will materialise for `frames` mel frames.
fn peak_grid_bytes(frames: usize) -> usize {
    let time = (frames.saturating_sub(1) + 2)
        .checked_sub(3)
        .map_or(0, |span| span / 2 + 1);
    time * CHANNELS * 64 * std::mem::size_of::<f32>()
}

/// Fails early when the audio would need an unreasonably large allocation.
pub(crate) fn check_frames(frames: usize) -> anyhow::Result<()> {
    let bytes = peak_grid_bytes(frames);
    if bytes > MAX_GRID_BYTES {
        anyhow::bail!(
            "{frames} mel frames need about {} MB in a single conv buffer, above the {} MB ceiling; split the audio upstream",
            bytes / (1 << 20),
            MAX_GRID_BYTES / (1 << 20)
        );
    }
    Ok(())
}

/// Frame count this stage emits for `frames` mel frames.
///
/// Mirrors the `valid` chain in [`PreEncode::forward`] without allocating, so a caller
/// can size buffers before running anything.
pub fn valid_steps(frames: usize) -> usize {
    fn out(len: usize, kernel: usize, stride: usize, padding: usize) -> usize {
        (len + 2 * padding)
            .checked_sub(kernel)
            .map_or(0, |span| span / stride + 1)
    }
    let mut valid = frames.saturating_sub(1);
    for (kernel, stride, padding) in [(3, 2, 1), (3, 2, 1), (1, 1, 0), (3, 2, 1), (1, 1, 0)] {
        valid = out(valid, kernel, stride, padding);
    }
    valid
}

/// `calculate_conv_output_size` for one axis.
fn conv_out_len(len: usize, conv: &Conv2d) -> usize {
    // A window wider than the padded span has no valid positions.
    (len + 2 * conv.padding)
        .checked_sub(conv.kernel)
        .map_or(0, |span| span / conv.stride + 1)
}

/// Zeroes time steps at or past `valid`, which is measured on this grid.
fn mask_tail(mut grid: Grid, valid: usize) -> Grid {
    let limit = valid.min(grid.time);
    let start = limit * grid.channels * grid.freqs;
    grid.data[start..].fill(0.0);
    grid
}

fn relu(mut grid: Grid) -> Grid {
    for value in grid.data.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
    grid
}

/// Convolution over the (time, freq) grid, matching `torch.nn.functional.conv2d`.
/// Convolution over the (time, freq) grid, matching `torch.nn.functional.conv2d`.
///
/// Work splits by output time step: each task owns `channels * freq_out`
/// contiguous floats of the output, so nothing is shared and the inner loops stay
/// in cache.
fn conv2d(conv: &Conv2d, input: &Grid) -> Grid {
    // A 1x1 convolution is a channel GEMM; the tap loop below would do the same
    // arithmetic one scalar at a time.
    if let Some(packed) = &conv.packed {
        return pointwise(conv, packed, input);
    }
    let time_out = (input.time + 2 * conv.padding - conv.kernel) / conv.stride + 1;
    let freq_out = (input.freqs + 2 * conv.padding - conv.kernel) / conv.stride + 1;
    let per_group = conv.in_channels / conv.groups;
    let taps = conv.kernel * conv.kernel;

    let mut out = Grid::new(time_out, conv.out_channels, freq_out);
    let slab_len = conv.out_channels * freq_out;

    out.data
        .par_chunks_mut(slab_len)
        .enumerate()
        .for_each(|(time, slab)| {
            let base_time = time * conv.stride;
            for channel in 0..conv.out_channels {
                let group = channel * conv.groups / conv.out_channels;
                let plane = channel * freq_out;
                let bias = conv.bias[channel];
                slab[plane..plane + freq_out].fill(bias);
                for inner in 0..per_group {
                    let in_channel = group * per_group + inner;
                    let weight_base = (channel * per_group + inner) * taps;
                    for freq in 0..freq_out {
                        let base_freq = freq * conv.stride;
                        let mut sum = 0.0f32;
                        for row in 0..conv.kernel {
                            let src_time = base_time as i64 - conv.padding as i64 + row as i64;
                            if src_time < 0 || src_time >= input.time as i64 {
                                continue;
                            }
                            let row_base =
                                (src_time as usize * conv.in_channels + in_channel) * input.freqs;
                            let weights_row = &conv.weight[weight_base + row * conv.kernel
                                ..weight_base + row * conv.kernel + conv.kernel];
                            for (col, weight) in weights_row.iter().enumerate() {
                                let src_freq = base_freq as i64 - conv.padding as i64 + col as i64;
                                if src_freq < 0 || src_freq >= input.freqs as i64 {
                                    continue;
                                }
                                sum += weight * input.data[row_base + src_freq as usize];
                            }
                        }
                        slab[plane + freq] += sum;
                    }
                }
            }
        });
    out
}

/// 1x1 convolution as one GEMM.
///
/// The grid is `[time][channel][freq]`, but a channel contraction wants channels
/// contiguous, so this transposes to `[time][freq][channel]`, runs a single
/// `m = time * freq` GEMM, and writes the result back in grid order. Both
/// transposes touch the same bytes the kernel would have streamed anyway.
fn pointwise(conv: &Conv2d, packed: &PackW, input: &Grid) -> Grid {
    let (time, freq) = (input.time, input.freqs);
    let in_channels = input.channels;
    let out_channels = conv.out_channels;

    let mut a = vec![0.0f32; time * freq * in_channels];
    for t in 0..time {
        for c in 0..in_channels {
            let row = &input.data[input.index(t, c, 0)..input.index(t, c, 0) + freq];
            for (f, value) in row.iter().enumerate() {
                a[(t * freq + f) * in_channels + c] = *value;
            }
        }
    }

    let mut projected = vec![0.0f32; time * freq * out_channels];
    gemm_into(&a, packed, &mut projected, time * freq);

    let mut out = Grid::new(time, out_channels, freq);
    for t in 0..time {
        for c in 0..out_channels {
            let base = out.index(t, c, 0);
            out.data[base..base + freq].fill(conv.bias[c]);
        }
        for f in 0..freq {
            let row = &projected[(t * freq + f) * out_channels..(t * freq + f + 1) * out_channels];
            for (c, value) in row.iter().enumerate() {
                let slot = out.index(t, c, f);
                out.data[slot] += *value;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_rows_matches_the_stage_propagation() {
        for frames in [2usize, 9, 16, 17, 100, 1440, 65_536] {
            let mut valid = frames - 1;
            for _ in 0..3 {
                valid = ((valid + 2 - 3).max(0)) / 2 + 1;
            }
            assert_eq!(valid_steps(frames), valid, "frames={frames}");
        }
    }

    #[test]
    fn chunks_tile_the_output_without_gaps() {
        for frames in [16usize, 100, 1440, 8_000, 20_000, 120_000] {
            let total = valid_steps(frames);
            let plan = plan_chunks(frames);
            assert!(!plan.is_empty(), "frames={frames}");
            let mut expected = 0;
            for chunk in &plan {
                assert_eq!(chunk.first_row, expected, "frames={frames}");
                assert_eq!(chunk.start % STEP_STRIDE, 0, "frames={frames}");
                assert_eq!(
                    chunk.start / STEP_STRIDE + chunk.keep_from,
                    chunk.first_row,
                    "frames={frames}"
                );
                assert!(chunk.start + chunk.len <= frames, "frames={frames}");
                // The slice must be long enough to own its rows after the margin.
                assert!(
                    valid_steps(chunk.len) >= chunk.keep_from + chunk.rows,
                    "frames={frames} chunk={chunk:?} produces {}",
                    valid_steps(chunk.len)
                );
                expected += chunk.rows;
            }
            assert_eq!(expected, total, "frames={frames}");
        }
    }

    #[test]
    fn short_audio_stays_in_one_chunk() {
        // 1440 frames is 14.4 s, the length the golden fixtures use.
        let plan = plan_chunks(1440);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].keep_from, 0);
        assert_eq!(plan[0].rows, valid_steps(1440));
        assert_eq!(plan[0].start, 0);
    }

    #[test]
    fn long_audio_stays_under_the_grid_budget() {
        // A 25-minute utterance: 150_000 mel frames.
        let frames = 150_000;
        let plan = plan_chunks(frames);
        assert!(plan.len() > 1);
        for chunk in &plan {
            let bytes = peak_grid_bytes(chunk.len);
            assert!(
                bytes <= CHUNK_GRID_BYTES + 64 * 1024 * 1024,
                "chunk of {} frames needs {bytes} bytes",
                chunk.len
            );
        }
    }
}
