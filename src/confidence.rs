//! NeMo-compatible frame-, token-, and word-level confidence estimation.
//!
//! Mirrors `nemo/collections/asr/parts/utils/asr_confidence_utils.py` and the TDT
//! confidence path in `rnnt_decoding.py` / `tdt_label_looping.py`.

use anyhow::{Result, bail};

use crate::tdt::{BLANK, DURATIONS, VOCAB};
use crate::text::Tokenizer;

/// Per-frame confidence method (`ConfidenceMethodConfig.name`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConfidenceMethodKind {
    MaxProb,
    #[default]
    Entropy,
}

/// Entropy family (`ConfidenceMethodConfig.entropy_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EntropyType {
    Gibbs,
    #[default]
    Tsallis,
    Renyi,
}

/// Mapping from entropy into `[0, 1]` (`ConfidenceMethodConfig.entropy_norm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EntropyNorm {
    Lin,
    #[default]
    Exp,
}

/// Aggregation used when collapsing token/blank/duration confidences (`ConfidenceConfig.aggregation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConfidenceAggregation {
    Mean,
    #[default]
    Min,
    Max,
    Prod,
}

/// Settings for per-frame confidence calculation (`ConfidenceMethodConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct ConfidenceMethodConfig {
    pub name: ConfidenceMethodKind,
    pub entropy_type: EntropyType,
    /// Power scale `alpha > 0` (`0.33` default). When `1.0`, all entropies match Gibbs/Shannon.
    pub alpha: f32,
    pub entropy_norm: EntropyNorm,
}

impl Default for ConfidenceMethodConfig {
    fn default() -> Self {
        Self {
            name: ConfidenceMethodKind::Entropy,
            entropy_type: EntropyType::Tsallis,
            alpha: 0.33,
            entropy_norm: EntropyNorm::Exp,
        }
    }
}

/// Full confidence settings (`ConfidenceConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct ConfidenceConfig {
    pub preserve_frame_confidence: bool,
    pub preserve_token_confidence: bool,
    pub preserve_word_confidence: bool,
    /// Exclude blank-frame confidences from `token_confidence` (`true` default).
    pub exclude_blank: bool,
    /// Aggregation for word (and blank/duration) confidence (`Min` default).
    pub aggregation: ConfidenceAggregation,
    /// Attach TDT duration confidence alongside token confidence in `frame_confidence`.
    pub tdt_include_duration: bool,
    pub method_cfg: ConfidenceMethodConfig,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            preserve_frame_confidence: false,
            preserve_token_confidence: false,
            preserve_word_confidence: false,
            exclude_blank: true,
            aggregation: ConfidenceAggregation::Min,
            tdt_include_duration: false,
            method_cfg: ConfidenceMethodConfig::default(),
        }
    }
}

impl ConfidenceConfig {
    /// True if any confidence tier is requested.
    pub fn enabled(&self) -> bool {
        self.preserve_frame_confidence
            || self.preserve_token_confidence
            || self.preserve_word_confidence
    }

    pub fn wants_token(&self) -> bool {
        self.preserve_token_confidence || self.preserve_word_confidence
    }

    pub fn validate(&self) -> Result<()> {
        if self.method_cfg.alpha <= 0.0 || !self.method_cfg.alpha.is_finite() {
            bail!(
                "confidence alpha must be > 0 and finite, got {}",
                self.method_cfg.alpha
            );
        }
        Ok(())
    }
}

/// One step's confidence value inside `FrameConfidence`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameConfidenceValue {
    /// Regular token confidence (`tdt_include_duration == false`).
    Scalar(f32),
    /// `(prediction_confidence, duration_confidence)` (`tdt_include_duration == true`).
    WithDuration(f32, f32),
}

/// Per-frame confidence entries collected during decoding.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameConfidence {
    pub frame: u32,
    pub values: Vec<FrameConfidenceValue>,
}

/// Computes `log_softmax` of a slice in f64 for numerical stability, returning f64 values.
pub(crate) fn log_softmax_f64(logits: &[f32]) -> Vec<f64> {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut sum_exp = 0.0f64;
    for &v in logits {
        let vf = if v.is_nan() { 0.0 } else { v as f64 };
        sum_exp += (vf - max).exp();
    }
    let log_z = max + sum_exp.ln();
    logits
        .iter()
        .map(|&v| {
            let vf = if v.is_nan() { 0.0 } else { v as f64 };
            vf - log_z
        })
        .collect()
}

/// Computes NeMo's confidence score from raw logits using `method` and vocabulary size `vocab_size`.
///
/// Mirrors `asr_confidence_utils.py::get_confidence_measure_bank`.
pub fn score_from_logits(
    logits: &[f32],
    vocab_size: usize,
    method: &ConfidenceMethodConfig,
) -> f32 {
    let x = log_softmax_f64(logits);
    score_from_log_probs(&x, vocab_size, method)
}

/// Computes NeMo's confidence score directly from log-probabilities `x` (`sum exp(x) == 1`).
pub fn score_from_log_probs(
    x: &[f64],
    vocab_size: usize,
    method: &ConfidenceMethodConfig,
) -> f32 {
    let v = vocab_size as f64;
    let t = method.alpha as f64;
    let unit_alpha = (t - 1.0).abs() < 1e-7;

    let neg_entropy_gibbs = || -> f64 { x.iter().map(|&xi| xi.exp() * xi).sum() };
    let neg_entropy_alpha = || -> f64 { x.iter().map(|&xi| (xi * t).exp()).sum() };
    let neg_entropy_alpha_gibbs = || -> f64 { x.iter().map(|&xi| (xi * t).exp() * xi).sum() };

    let gibbs_lin_baseline = || -> f64 { 1.0 + neg_entropy_gibbs() / v.ln() };
    let gibbs_exp_baseline = || -> f64 { (neg_entropy_gibbs().exp() * v - 1.0) / (v - 1.0) };

    let value = match method.name {
        ConfidenceMethodKind::MaxProb => {
            let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if unit_alpha {
                (m.exp() * v - 1.0) / (v - 1.0)
            } else {
                let vt = v.powf(t);
                ((m * t).exp() * vt - 1.0) / (vt - 1.0)
            }
        }
        ConfidenceMethodKind::Entropy => match (method.entropy_type, method.entropy_norm) {
            (EntropyType::Gibbs, EntropyNorm::Lin) => {
                if unit_alpha {
                    gibbs_lin_baseline()
                } else {
                    1.0 + neg_entropy_alpha_gibbs() / v.ln() / v.powf(1.0 - t)
                }
            }
            (EntropyType::Gibbs, EntropyNorm::Exp) => {
                if unit_alpha {
                    gibbs_exp_baseline()
                } else {
                    let exp_neg_max_ent = v.powf(-t * v.powf(1.0 - t));
                    ((neg_entropy_alpha_gibbs() * t).exp() - exp_neg_max_ent)
                        / (1.0 - exp_neg_max_ent)
                }
            }
            (EntropyType::Tsallis, EntropyNorm::Lin) => {
                if unit_alpha {
                    gibbs_lin_baseline()
                } else {
                    1.0 + (1.0 - neg_entropy_alpha()) / (v.powf(1.0 - t) - 1.0)
                }
            }
            (EntropyType::Tsallis, EntropyNorm::Exp) => {
                if unit_alpha {
                    gibbs_exp_baseline()
                } else {
                    let exp_neg_max_ent = ((1.0 - v.powf(1.0 - t)) / (1.0 - t)).exp();
                    ((((1.0 - neg_entropy_alpha()) / (1.0 - t)).exp()) - exp_neg_max_ent)
                        / (1.0 - exp_neg_max_ent)
                }
            }
            (EntropyType::Renyi, EntropyNorm::Lin) => {
                if unit_alpha {
                    gibbs_lin_baseline()
                } else {
                    1.0 + neg_entropy_alpha().log2() / (t - 1.0) / v.log2()
                }
            }
            (EntropyType::Renyi, EntropyNorm::Exp) => {
                if unit_alpha {
                    gibbs_exp_baseline()
                } else {
                    (neg_entropy_alpha().powf(1.0 / (t - 1.0)) * v - 1.0) / (v - 1.0)
                }
            }
        },
    };
    value as f32
}

/// Computes one TDT step's confidence entry from full joint logits (`VOCAB + DURATIONS`).
///
/// Mirrors `tdt_label_looping.py::_get_frame_confidence`: both token and duration
/// distributions are scored with `num_tokens = VOCAB` (8193) as in NeMo's
/// `ConfidenceMethodMixin`.
pub fn step_confidence(logits: &[f32], cfg: &ConfidenceConfig) -> FrameConfidenceValue {
    debug_assert_eq!(logits.len(), VOCAB + DURATIONS);
    let pred_conf = score_from_logits(&logits[..VOCAB], VOCAB, &cfg.method_cfg);
    if cfg.tdt_include_duration {
        let dur_conf = score_from_logits(&logits[VOCAB..], VOCAB, &cfg.method_cfg);
        FrameConfidenceValue::WithDuration(pred_conf, dur_conf)
    } else {
        FrameConfidenceValue::Scalar(pred_conf)
    }
}

/// Aggregates a non-empty slice of confidence values according to `mode`.
pub fn aggregate(values: &[f32], mode: ConfidenceAggregation) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    match mode {
        ConfidenceAggregation::Mean => {
            let sum: f64 = values.iter().map(|&v| v as f64).sum();
            (sum / values.len() as f64) as f32
        }
        ConfidenceAggregation::Min => values.iter().copied().fold(f32::INFINITY, f32::min),
        ConfidenceAggregation::Max => values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        ConfidenceAggregation::Prod => values.iter().map(|&v| v as f64).product::<f64>() as f32,
    }
}

/// Collapses per-step frame confidences (with their corresponding emitted labels, including blanks)
/// into per-token confidences, mirroring `rnnt_decoding.py::compute_confidence` for TDT.
pub fn compute_token_confidence(
    frames: &[FrameConfidence],
    frame_labels: &[Vec<u32>],
    cfg: &ConfidenceConfig,
) -> Vec<f32> {
    let pre_agg = |val: FrameConfidenceValue| -> f32 {
        match val {
            FrameConfidenceValue::Scalar(v) => v,
            FrameConfidenceValue::WithDuration(p, d) => aggregate(&[p, d], cfg.aggregation),
        }
    };

    let mut token_confidence = Vec::new();
    let mut subsequent_blank: Vec<f32> = Vec::new();

    for (fc, labels) in frames.iter().rev().zip(frame_labels.iter().rev()) {
        for (&val, &label) in fc.values.iter().rev().zip(labels.iter().rev()) {
            let score = pre_agg(val);
            if label == BLANK {
                if !cfg.exclude_blank {
                    subsequent_blank.push(score);
                }
            } else if subsequent_blank.is_empty() {
                token_confidence.push(score);
            } else {
                let mut combined = Vec::with_capacity(1 + subsequent_blank.len());
                combined.push(score);
                combined.extend_from_slice(&subsequent_blank);
                token_confidence.push(aggregate(&combined, cfg.aggregation));
                subsequent_blank.clear();
            }
        }
    }
    token_confidence.reverse();
    token_confidence
}

/// Aggregates per-token confidences into per-word confidences for a SentencePiece tokenizer,
/// mirroring `asr_confidence_utils.py::_aggregate_token_confidence_subwords_sentencepiece`.
pub fn compute_word_confidence(
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    token_confidence: &[f32],
    mode: ConfidenceAggregation,
) -> Vec<f32> {
    if token_ids.is_empty() || token_confidence.is_empty() {
        return Vec::new();
    }
    let mut word_confidence = Vec::new();
    let mut j = 0usize;
    let mut prev_unk = false;
    let mut prev_underline = false;
    for (i, &token_id) in token_ids.iter().enumerate() {
        let raw_piece = tokenizer.raw_piece(token_id).unwrap_or("");
        let token_text = tokenizer.decode(&[token_id]);
        let is_unk = tokenizer.is_unk(token_id);
        let starts_new = (raw_piece != token_text || prev_unk) && i > j;
        if starts_new {
            if !prev_underline {
                word_confidence.push(aggregate(&token_confidence[j..i], mode));
            }
            j = i;
        }
        prev_unk = is_unk;
        prev_underline = raw_piece == "\u{2581}";
    }
    if !prev_underline && j < token_confidence.len() {
        word_confidence.push(aggregate(&token_confidence[j..token_confidence.len()], mode));
    }
    word_confidence
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaked_distribution_has_confidence_near_one() {
        let mut logits = vec![-20.0f32; 16];
        logits[3] = 20.0;
        for cfg in [
            ConfidenceMethodConfig {
                name: ConfidenceMethodKind::MaxProb,
                alpha: 1.0,
                ..Default::default()
            },
            ConfidenceMethodConfig {
                name: ConfidenceMethodKind::Entropy,
                entropy_type: EntropyType::Tsallis,
                alpha: 0.33,
                entropy_norm: EntropyNorm::Exp,
            },
            ConfidenceMethodConfig {
                name: ConfidenceMethodKind::Entropy,
                entropy_type: EntropyType::Gibbs,
                alpha: 1.0,
                entropy_norm: EntropyNorm::Lin,
            },
            ConfidenceMethodConfig {
                name: ConfidenceMethodKind::Entropy,
                entropy_type: EntropyType::Renyi,
                alpha: 0.33,
                entropy_norm: EntropyNorm::Exp,
            },
        ] {
            let s = score_from_logits(&logits, 16, &cfg);
            assert!((s - 1.0).abs() < 1e-4, "method={cfg:?} got {s}");
        }
    }

    #[test]
    fn uniform_distribution_has_confidence_zero() {
        let logits = vec![0.0f32; 16];
        for cfg in [
            ConfidenceMethodConfig {
                name: ConfidenceMethodKind::MaxProb,
                alpha: 0.5,
                ..Default::default()
            },
            ConfidenceMethodConfig {
                name: ConfidenceMethodKind::Entropy,
                entropy_type: EntropyType::Tsallis,
                alpha: 0.33,
                entropy_norm: EntropyNorm::Exp,
            },
            ConfidenceMethodConfig {
                name: ConfidenceMethodKind::Entropy,
                entropy_type: EntropyType::Tsallis,
                alpha: 0.33,
                entropy_norm: EntropyNorm::Lin,
            },
            ConfidenceMethodConfig {
                name: ConfidenceMethodKind::Entropy,
                entropy_type: EntropyType::Gibbs,
                alpha: 0.5,
                entropy_norm: EntropyNorm::Exp,
            },
            ConfidenceMethodConfig {
                name: ConfidenceMethodKind::Entropy,
                entropy_type: EntropyType::Gibbs,
                alpha: 1.0,
                entropy_norm: EntropyNorm::Lin,
            },
            ConfidenceMethodConfig {
                name: ConfidenceMethodKind::Entropy,
                entropy_type: EntropyType::Renyi,
                alpha: 0.33,
                entropy_norm: EntropyNorm::Exp,
            },
            ConfidenceMethodConfig {
                name: ConfidenceMethodKind::Entropy,
                entropy_type: EntropyType::Renyi,
                alpha: 0.33,
                entropy_norm: EntropyNorm::Lin,
            },
        ] {
            let s = score_from_logits(&logits, 16, &cfg);
            assert!(s.abs() < 1e-5, "method={cfg:?} got {s}");
        }
    }

    #[test]
    fn blank_confidence_attaches_to_preceding_token() {
        let frames = vec![
            FrameConfidence {
                frame: 0,
                values: vec![
                    FrameConfidenceValue::Scalar(0.8),
                    FrameConfidenceValue::Scalar(0.4),
                ],
            },
            FrameConfidence {
                frame: 1,
                values: vec![FrameConfidenceValue::Scalar(0.6)],
            },
        ];
        let labels = vec![vec![10, BLANK], vec![BLANK]];
        let mut cfg = ConfidenceConfig::default();
        cfg.exclude_blank = true;
        assert_eq!(compute_token_confidence(&frames, &labels, &cfg), vec![0.8]);

        cfg.exclude_blank = false;
        cfg.aggregation = ConfidenceAggregation::Min;
        assert_eq!(compute_token_confidence(&frames, &labels, &cfg), vec![0.4]);

        cfg.aggregation = ConfidenceAggregation::Mean;
        let mean = compute_token_confidence(&frames, &labels, &cfg)[0];
        assert!((mean - 0.6).abs() < 1e-6);
    }
}
