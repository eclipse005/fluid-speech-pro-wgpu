//! Helpers shared by the verification and benchmark binaries.
//!
//! Both are developer tools: they compare stages against NeMo fixtures or time
//! them. Keeping them out of `fsp` is why the shipped CLI depends only on `api`.

use std::path::Path;

use anyhow::{Context, Result, bail};

use fluid_speech_pro::api::read_wav_mono_16k;
use fluid_speech_pro::{mel, torch};

/// Loads a `.f32` time-major mel fixture or decodes a WAV through the front end.
pub(crate) fn load_features(frontend: &mel::MelFrontend, input: &Path) -> Result<(Vec<f32>, usize)> {
    if input.extension().is_some_and(|ext| ext == "f32") {
        let bytes = std::fs::read(input).with_context(|| format!("read {}", input.display()))?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let frames = values.len() / mel::N_MELS;
        return Ok((values, frames));
    }
    let samples = read_wav_mono_16k(input)?;
    let decoded = frontend.compute(&samples)?;
    let frames = frontend.frames(samples.len());
    let bins = decoded.len() / frames;
    let mut time_major = vec![0.0f32; frames * bins];
    for bin in 0..bins {
        for frame in 0..frames {
            time_major[frame * bins + bin] = decoded[bin * frames + frame];
        }
    }
    Ok((time_major, frames))
}

/// Prints the deviation between a fixture and computed values.
pub(crate) fn compare_values(expected: &[f32], actual: &[f32]) -> Result<()> {
    if expected.len() != actual.len() {
        bail!(
            "fixture holds {} values, computed {}",
            expected.len(),
            actual.len()
        );
    }
    let mut worst = 0.0f32;
    let mut total = 0.0f64;
    let mut worst_index = 0usize;
    for (index, (want, got)) in expected.iter().zip(actual).enumerate() {
        let diff = (want - got).abs();
        if diff > worst {
            worst = diff;
            worst_index = index;
        }
        total += f64::from(diff);
    }
    println!(
        "maxdiff={worst:.3e} meandiff={:.3e} at index {worst_index} (want {:.4}, got {:.4})",
        total / actual.len() as f64,
        expected[worst_index],
        actual[worst_index]
    );
    Ok(())
}

/// Compares little-endian f32 features with a fixture and prints the deviation.
pub(crate) fn compare_f32(path: &Path, features: &[f32]) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() != features.len() * 4 {
        bail!(
            "{} holds {} bytes, expected {}",
            path.display(),
            bytes.len(),
            features.len() * 4
        );
    }
    let mut worst = 0.0f32;
    let mut total = 0.0f64;
    for (chunk, actual) in bytes.as_chunks::<4>().0.iter().zip(features) {
        let expected_value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let diff = (expected_value - actual).abs();
        worst = worst.max(diff);
        total += f64::from(diff);
    }
    println!(
        "maxdiff={worst:.3e} meandiff={:.3e}",
        total / features.len() as f64
    );
    Ok(())
}

/// Reads a `f32` fixture and reports how far `got` is from it.
pub(crate) fn compare_stage(label: &str, got: &[f32], path: &str) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("read {path}"))?;
    let (words, _rest) = bytes.as_chunks::<4>();
    let expected: Vec<f32> = words.iter().map(|word| f32::from_le_bytes(*word)).collect();
    let mut worst = 0.0f32;
    let mut total = 0.0f64;
    let mut peak = 0.0f32;
    for (a, b) in expected.iter().zip(got) {
        worst = worst.max((a - b).abs());
        total += f64::from((a - b).abs());
        peak = peak.max(a.abs());
    }
    println!(
        "  {label:<8} maxdiff={worst:.3e} meandiff={:.3e} peak={peak:.1}",
        total / expected.len() as f64
    );
    Ok(())
}

/// Reads a `LayerNorm` weight/bias pair by name.
pub(crate) fn norm_pair(store: &torch::Store, prefix: &str) -> Result<(Vec<f32>, Vec<f32>)> {
    Ok((
        store.f32(&format!("{prefix}.weight"))?,
        store.f32(&format!("{prefix}.bias"))?,
    ))
}

