//! TDT greedy decoding: LSTM predictor, joint network, token/duration loop.
//!
//! Mirrors NeMo's `RNNTDecoder.predict`, `RNNTJoint.joint` and the TDT label
//! looping decoder. The joint projects the encoder frame and the predictor state
//! into the same 640-wide space, **adds** them, then applies
//! `ReLU -> Linear(640 -> 8198)`; the first 8193 rows are the vocabulary (8192 is
//! blank) and the last five are the TDT durations `[0, 1, 2, 3, 4]`.
//!
//! Loop rules (NeMo TDT greedy):
//! * blank token: advance by `max(duration, 1)` frames, predictor state unchanged;
//! * real token: emit at the current frame with its duration, run the predictor on
//!   it, advance by `max(duration, 1)` frames;
//! * `max_symbols_per_step` caps emissions on one frame.

use rayon::prelude::*;
use anyhow::{Result, bail};

use crate::confidence::{self, ConfidenceConfig, FrameConfidence};
use crate::fusion::{FusionModel, NON_BLANK_VOCAB};
use crate::torch::Store;

pub const VOCAB: usize = 8193;
pub const BLANK: u32 = 8192;
pub const DEC_HIDDEN: usize = 640;
pub const JOINT_HIDDEN: usize = 640;
pub const ENC_HIDDEN: usize = 1024;
/// `[0, 1, 2, 3, 4]`; the logits after `VOCAB` are durations.
pub const DURATIONS: usize = 5;
pub const JOINT_VOCAB: usize = VOCAB + DURATIONS;
pub const MAX_SYMBOLS_PER_STEP: usize = 10;
/// Frames are 80 ms in TDT (8x subsampling of 10 ms).
pub const FRAME_SECONDS: f32 = 0.08;

/// One emitted token and where it sits on the frame grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Token {
    pub id: u32,
    pub frame: u32,
    /// The TDT duration class that was predicted with this token.
    pub duration: u32,
    /// `top1 - top2` of the duration logits; small values mean the choice was a
    /// numerical near-tie rather than a structural difference.
    pub duration_margin: f32,
    /// The five raw duration logits at the decision point, for tracing.
    pub duration_logits: [f32; DURATIONS],
    /// Token score at emission: selected vocabulary logit in greedy decoding (matching
    /// NeMo `BatchedHyps.scores`), or step log-probability in beam search.
    pub score: f32,
}

/// One step recorded at a frame when `preserve_alignments` is enabled.
///
/// Matches NeMo's `Hypothesis.alignments[t][u] = (logits, label)` plus the predicted
/// TDT duration class.
#[derive(Debug, Clone)]
pub struct AlignmentEntry {
    pub label: u32,
    pub duration: u32,
    /// Full joint output (`VOCAB + DURATIONS = 8198` floats, after temperature scaling if any).
    pub logits: Vec<f32>,
}

/// All joint evaluations performed at a single encoder frame `frame`.
#[derive(Debug, Clone)]
pub struct FrameAlignment {
    pub frame: u32,
    pub steps: Vec<AlignmentEntry>,
}

/// Predictor + frame carry-over state for streaming / incremental TDT decoding.
///
/// Mirrors NeMo's `BatchedLabelLoopingState` (`tdt_label_looping.py`): carries the
/// 2-layer LSTM `(h, c)` states, the last emitted non-blank label, the pre-computed
/// predictor hidden state, the total frames decoded so far (for timestamp offsetting),
/// and `time_jump` (`time_indices - encoder_output_length` when the final duration
/// jumps past the chunk boundary).
#[derive(Debug, Clone)]
pub struct DecoderState {
    pub h0: Vec<f32>,
    pub c0: Vec<f32>,
    pub h1: Vec<f32>,
    pub c1: Vec<f32>,
    pub last_token: u32,
    pub predictor_out: Vec<f32>,
    pub decoded_frames: u32,
    pub time_jump: u32,
}

/// Configuration for greedy TDT decoding.
#[derive(Debug, Clone)]
pub struct GreedyConfig {
    /// Maximum non-blank symbols emitted on a single frame (`Some(10)` default; `None` = unlimited).
    pub max_symbols_per_step: Option<usize>,
    /// Softmax/logit temperature (`1.0` default). Values != 1.0 divide logits before scoring.
    pub temperature: f32,
    /// Record per-frame `(logits, label, duration)` alignment history.
    pub preserve_alignments: bool,
    /// Optional NeMo-compatible frame/token/word confidence configuration.
    pub confidence: ConfidenceConfig,
}

impl Default for GreedyConfig {
    fn default() -> Self {
        Self {
            max_symbols_per_step: Some(MAX_SYMBOLS_PER_STEP),
            temperature: 1.0,
            preserve_alignments: false,
            confidence: ConfidenceConfig::default(),
        }
    }
}

/// Full output of a greedy TDT decode pass.
#[derive(Debug, Clone)]
pub struct GreedyOutput {
    pub tokens: Vec<Token>,
    pub score: f32,
    pub alignments: Option<Vec<FrameAlignment>>,
    pub frame_confidence: Option<Vec<FrameConfidence>>,
    pub token_confidence: Option<Vec<f32>>,
    pub state: Option<DecoderState>,
}

struct LstmLayer {
    weight_ih: Vec<f32>,
    weight_hh: Vec<f32>,
    bias_ih: Vec<f32>,
    bias_hh: Vec<f32>,
    h: Vec<f32>,
    c: Vec<f32>,
}

impl LstmLayer {
    fn new(store: &Store, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight_ih: store.f32(&format!("{prefix}.weight_ih_l0"))?,
            weight_hh: store.f32(&format!("{prefix}.weight_hh_l0"))?,
            bias_ih: store.f32(&format!("{prefix}.bias_ih_l0"))?,
            bias_hh: store.f32(&format!("{prefix}.bias_hh_l0"))?,
            h: vec![0.0; DEC_HIDDEN],
            c: vec![0.0; DEC_HIDDEN],
        })
    }

    /// Pure single-step LSTM cell returning `(new_h, new_c)` without mutating `self`.
    fn step_pure(&self, input: &[f32], prev_h: &[f32], prev_c: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let gates = 4 * DEC_HIDDEN;
        let mut pre = vec![0.0f32; gates];
        pre.par_iter_mut().enumerate().for_each(|(row, slot)| {
            let mut sum = self.bias_ih[row] + self.bias_hh[row];
            let w_ih = &self.weight_ih[row * DEC_HIDDEN..(row + 1) * DEC_HIDDEN];
            for (weight, value) in w_ih.iter().zip(input) {
                sum += weight * value;
            }
            let w_hh = &self.weight_hh[row * DEC_HIDDEN..(row + 1) * DEC_HIDDEN];
            for (weight, value) in w_hh.iter().zip(prev_h) {
                sum += weight * value;
            }
            *slot = sum;
        });
        let mut next_h = vec![0.0f32; DEC_HIDDEN];
        let mut next_c = vec![0.0f32; DEC_HIDDEN];
        for unit in 0..DEC_HIDDEN {
            let input_gate = sigmoid(pre[unit]);
            let forget_gate = sigmoid(pre[DEC_HIDDEN + unit]);
            let cell_gate = pre[2 * DEC_HIDDEN + unit].tanh();
            let output_gate = sigmoid(pre[3 * DEC_HIDDEN + unit]);
            let c_val = forget_gate * prev_c[unit] + input_gate * cell_gate;
            next_c[unit] = c_val;
            next_h[unit] = output_gate * c_val.tanh();
        }
        (next_h, next_c)
    }

    /// One step of PyTorch's LSTM cell, gate order `i, f, g, o`.
    fn step(&mut self, input: &[f32]) -> Vec<f32> {
        let (next_h, next_c) = self.step_pure(input, &self.h, &self.c);
        self.h = next_h;
        self.c = next_c;
        self.h.clone()
    }
}

/// Immutable predictor LSTM state plus its pre-computed `joint.pred` projection,
/// used by beam search decoders for zero-copy state sharing and prefix caching.
#[derive(Clone)]
pub(crate) struct PredictorState {
    pub h0: Vec<f32>,
    pub c0: Vec<f32>,
    pub h1: Vec<f32>,
    pub c1: Vec<f32>,
    pub pred_proj: Vec<f32>,
}

pub struct TdtDecoder {
    embed: Vec<f32>,
    layers: Vec<LstmLayer>,
    pred_weight: Vec<f32>,
    pred_bias: Vec<f32>,
    enc_weight: Vec<f32>,
    enc_bias: Vec<f32>,
    out_weight: Vec<f32>,
    out_bias: Vec<f32>,
    /// When set, the vocabulary argmax runs on the device with the joint output
    /// matrix resident there; duration rows stay on the CPU.
    gpu: Option<crate::tdt_gpu::TdtGpu>,
}

impl TdtDecoder {
    pub fn from_store(store: &Store) -> Result<Self> {
        let mut layer0 = LstmLayer::new(store, "decoder.prediction.dec_rnn.lstm")?;
        let mut layer1 = LstmLayer {
            weight_ih: store.f32("decoder.prediction.dec_rnn.lstm.weight_ih_l1")?,
            weight_hh: store.f32("decoder.prediction.dec_rnn.lstm.weight_hh_l1")?,
            bias_ih: store.f32("decoder.prediction.dec_rnn.lstm.bias_ih_l1")?,
            bias_hh: store.f32("decoder.prediction.dec_rnn.lstm.bias_hh_l1")?,
            h: vec![0.0; DEC_HIDDEN],
            c: vec![0.0; DEC_HIDDEN],
        };
        layer0.h.fill(0.0);
        layer1.h.fill(0.0);

        let embed = store.f32("decoder.prediction.embed.weight")?;
        if embed.len() != VOCAB * DEC_HIDDEN {
            bail!(
                "embedding has {} values, expected {}",
                embed.len(),
                VOCAB * DEC_HIDDEN
            );
        }
        Ok(Self {
            embed,
            layers: vec![layer0, layer1],
            pred_weight: store.f32("joint.pred.weight")?,
            pred_bias: store.f32("joint.pred.bias")?,
            enc_weight: store.f32("joint.enc.weight")?,
            enc_bias: store.f32("joint.enc.bias")?,
            out_weight: store.f32("joint.joint_net.2.weight")?,
            out_bias: store.f32("joint.joint_net.2.bias")?,
            gpu: None,
        })
    }

    /// Uploads the joint output matrix and routes the vocabulary argmax through the
    /// device. Alignment is unaffected: the argmax is over the same 8193 scores, only
    /// the summation order inside each row changes.
    pub fn enable_gpu(&mut self) -> anyhow::Result<()> {
        self.enable_gpu_with(std::sync::Arc::new(crate::gpu::GpuGemm::new()?))
    }

    /// Builds the resident GPU decoder on an existing device, so it can share the
    /// queue with the encoder and consume `encoder_out` without a host readback.
    pub fn enable_gpu_with(
        &mut self,
        gpu: std::sync::Arc<crate::gpu::GpuGemm>,
    ) -> anyhow::Result<()> {
        let weights = crate::tdt_gpu::TdtWeights {
            embed: &self.embed,
            l0_w_ih: &self.layers[0].weight_ih,
            l0_w_hh: &self.layers[0].weight_hh,
            l0_b_ih: &self.layers[0].bias_ih,
            l0_b_hh: &self.layers[0].bias_hh,
            l1_w_ih: &self.layers[1].weight_ih,
            l1_w_hh: &self.layers[1].weight_hh,
            l1_b_ih: &self.layers[1].bias_ih,
            l1_b_hh: &self.layers[1].bias_hh,
            pred_weight: &self.pred_weight,
            pred_bias: &self.pred_bias,
            enc_weight: &self.enc_weight,
            enc_bias: &self.enc_bias,
            out_weight: &self.out_weight,
            out_bias: &self.out_bias,
        };
        self.gpu = Some(crate::tdt_gpu::TdtGpu::new(gpu, weights)?);
        Ok(())
    }


    /// Decodes directly from a GPU-resident `[steps][1024]` encoder output buffer.
    pub fn decode_gpu_buffer(
        &mut self,
        encoder_out: &wgpu::Buffer,
        steps: usize,
    ) -> Result<Vec<Token>> {
        let Some(gpu) = &mut self.gpu else {
            bail!("GPU TDT decoder is not enabled");
        };
        gpu.decode_device(encoder_out, steps)
    }

    /// Resident GPU greedy decode with full options (`max_symbols_per_step`, `temperature`,
    /// `preserve_alignments`, confidence, shallow `fusion` models, streaming `partial` state).
    ///
    /// Takes the zero-readback resident GPU TDT fast path whenever alignments,
    /// confidence, fusion models, streaming state, and non-unit temperature are not requested.
    pub fn decode_gpu_greedy(
        &mut self,
        encoder_out: &wgpu::Buffer,
        steps: usize,
        cfg: &GreedyConfig,
        fusion: &[(&dyn FusionModel, f32)],
        partial: Option<&DecoderState>,
        return_state: bool,
    ) -> Result<GreedyOutput> {
        let fast_path = !cfg.preserve_alignments
            && !cfg.confidence.enabled()
            && fusion.is_empty()
            && partial.is_none()
            && !return_state
            && (cfg.temperature - 1.0).abs() < 1e-7;
        if fast_path {
            let Some(gpu) = &mut self.gpu else {
                bail!("GPU TDT decoder is not enabled");
            };
            let tokens = gpu.decode_device_with(encoder_out, steps, cfg.max_symbols_per_step)?;
            let score: f32 = tokens.iter().map(|t| t.score).sum();
            return Ok(GreedyOutput {
                tokens,
                score,
                alignments: None,
                frame_confidence: None,
                token_confidence: None,
                state: None,
            });
        }
        let enc_proj = self.project_encoder_gpu(encoder_out, steps)?;
        self.decode_greedy_from_enc_proj(&enc_proj, steps, cfg, fusion, partial, return_state)
    }

    /// Projects a resident GPU encoder output `[steps][1024]` to `[steps][640]` (including `enc_bias`).
    pub fn project_encoder_gpu(
        &mut self,
        encoder_out: &wgpu::Buffer,
        steps: usize,
    ) -> Result<Vec<f32>> {
        let Some(gpu) = &mut self.gpu else {
            bail!("GPU TDT decoder is not enabled");
        };
        let mut proj = gpu.project_encoder_device(encoder_out, steps)?;
        let bias = &self.enc_bias;
        proj.par_chunks_mut(JOINT_HIDDEN).for_each(|row| {
            for (slot, &b) in row.iter_mut().zip(bias) {
                *slot += b;
            }
        });
        Ok(proj)
    }

    /// Projects host encoder output `[steps][1024]` to `[steps][640]` (including `enc_bias`).
    pub fn project_encoder_cpu(&self, encoder: &[f32], steps: usize) -> Result<Vec<f32>> {
        if encoder.len() != steps * ENC_HIDDEN {
            bail!(
                "encoder output has {} values, expected {}",
                encoder.len(),
                steps * ENC_HIDDEN
            );
        }
        let mut proj = vec![0.0f32; steps * JOINT_HIDDEN];
        proj.par_chunks_mut(JOINT_HIDDEN)
            .enumerate()
            .for_each(|(t, out_row)| {
                let enc_frame = &encoder[t * ENC_HIDDEN..(t + 1) * ENC_HIDDEN];
                for unit in 0..JOINT_HIDDEN {
                    let w = &self.enc_weight[unit * ENC_HIDDEN..(unit + 1) * ENC_HIDDEN];
                    let mut sum = self.enc_bias[unit];
                    for (weight, value) in w.iter().zip(enc_frame) {
                        sum += weight * value;
                    }
                    out_row[unit] = sum;
                }
            });
        Ok(proj)
    }

    /// Projects a predictor output `[640]` through `joint.pred` (including `pred_bias`).
    pub(crate) fn project_pred(&self, predictor: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; JOINT_HIDDEN];
        out.par_iter_mut().enumerate().for_each(|(unit, slot)| {
            let w = &self.pred_weight[unit * DEC_HIDDEN..(unit + 1) * DEC_HIDDEN];
            let mut sum = self.pred_bias[unit];
            for (weight, value) in w.iter().zip(predictor) {
                sum += weight * value;
            }
            *slot = sum;
        });
        out
    }

    /// Computes joint logits `[JOINT_VOCAB]` from already-projected encoder and predictor vectors.
    pub(crate) fn joint_from_projected(
        &self,
        enc_proj_frame: &[f32],
        pred_proj: &[f32],
        projected: &mut [f32],
        logits: &mut [f32],
    ) {
        for i in 0..JOINT_HIDDEN {
            let v = enc_proj_frame[i] + pred_proj[i];
            projected[i] = if v > 0.0 { v } else { 0.0 };
        }
        self.output_range(projected, logits, 0, JOINT_VOCAB);
    }

    /// Pure immutable predictor step from `prev` (`None` = zero initial state) on `token`.
    pub(crate) fn predict_from_state(
        &self,
        prev: Option<&PredictorState>,
        token: u32,
    ) -> PredictorState {
        let zeros = [0.0f32; DEC_HIDDEN];
        let (ph0, pc0, ph1, pc1) = match prev {
            Some(s) => (s.h0.as_slice(), s.c0.as_slice(), s.h1.as_slice(), s.c1.as_slice()),
            None => (&zeros[..], &zeros[..], &zeros[..], &zeros[..]),
        };
        let row = &self.embed[token as usize * DEC_HIDDEN..(token as usize + 1) * DEC_HIDDEN];
        let (h0, c0) = self.layers[0].step_pure(row, ph0, pc0);
        let (h1, c1) = self.layers[1].step_pure(&h0, ph1, pc1);
        let pred_proj = self.project_pred(&h1);
        PredictorState {
            h0,
            c0,
            h1,
            c1,
            pred_proj,
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.h.fill(0.0);
            layer.c.fill(0.0);
        }
    }

    fn snapshot_state(
        &self,
        last_token: u32,
        predictor_out: &[f32],
        decoded_frames: u32,
        time_jump: u32,
    ) -> DecoderState {
        DecoderState {
            h0: self.layers[0].h.clone(),
            c0: self.layers[0].c.clone(),
            h1: self.layers[1].h.clone(),
            c1: self.layers[1].c.clone(),
            last_token,
            predictor_out: predictor_out.to_vec(),
            decoded_frames,
            time_jump,
        }
    }

    fn restore_state(&mut self, state: &DecoderState) -> Result<()> {
        if state.h0.len() != DEC_HIDDEN
            || state.c0.len() != DEC_HIDDEN
            || state.h1.len() != DEC_HIDDEN
            || state.c1.len() != DEC_HIDDEN
            || state.predictor_out.len() != DEC_HIDDEN
        {
            bail!("partial DecoderState has invalid hidden dimensions");
        }
        self.layers[0].h.copy_from_slice(&state.h0);
        self.layers[0].c.copy_from_slice(&state.c0);
        self.layers[1].h.copy_from_slice(&state.h1);
        self.layers[1].c.copy_from_slice(&state.c1);
        Ok(())
    }

    /// Feeds one token through the predictor stack and returns the 640-wide state.
    fn predict(&mut self, token: u32) -> Vec<f32> {
        let row = &self.embed[token as usize * DEC_HIDDEN..(token as usize + 1) * DEC_HIDDEN];
        let mut value = row.to_vec();
        for index in 0..self.layers.len() {
            let out = self.layers[index].step(&value);
            value = out;
        }
        value
    }

    /// Output projection for rows `[from, to)` of the joint vocabulary.
    fn output_range(&self, projected: &[f32], logits: &mut [f32], from: usize, to: usize) {
        logits[from..to]
            .par_iter_mut()
            .enumerate()
            .for_each(|(offset, slot)| {
                let unit = from + offset;
                let w = &self.out_weight[unit * JOINT_HIDDEN..(unit + 1) * JOINT_HIDDEN];
                let mut sum = self.out_bias[unit];
                for (weight, value) in w.iter().zip(projected) {
                    sum += weight * value;
                }
                *slot = sum;
            });
    }

    /// Greedy TDT decode over `steps` encoder frames (`[steps][1024]`, time-major).
    pub fn decode(&mut self, encoder: &[f32], steps: usize) -> Result<Vec<Token>> {
        if encoder.len() != steps * ENC_HIDDEN {
            bail!(
                "encoder output has {} values, expected {}",
                encoder.len(),
                steps * ENC_HIDDEN
            );
        }
        if let Some(gpu) = &mut self.gpu {
            return gpu.decode_host(encoder, steps);
        }
        Ok(self
            .decode_greedy(encoder, steps, &GreedyConfig::default(), &[], None, false)?
            .tokens)
    }

    /// Configurable greedy TDT decode over host `[steps][1024]` encoder frames.
    pub fn decode_greedy(
        &mut self,
        encoder: &[f32],
        steps: usize,
        cfg: &GreedyConfig,
        fusion: &[(&dyn FusionModel, f32)],
        partial: Option<&DecoderState>,
        return_state: bool,
    ) -> Result<GreedyOutput> {
        if encoder.len() != steps * ENC_HIDDEN {
            bail!(
                "encoder output has {} values, expected {}",
                encoder.len(),
                steps * ENC_HIDDEN
            );
        }
        if cfg.temperature <= 0.0 || !cfg.temperature.is_finite() {
            bail!("temperature must be positive and finite, got {}", cfg.temperature);
        }
        let fast_gpu = !cfg.preserve_alignments
            && !cfg.confidence.enabled()
            && fusion.is_empty()
            && partial.is_none()
            && !return_state
            && (cfg.temperature - 1.0).abs() < 1e-7;
        if fast_gpu && self.gpu.is_some() {
            let tokens = self
                .gpu
                .as_mut()
                .unwrap()
                .decode_host_with(encoder, steps, cfg.max_symbols_per_step)?;
            let score: f32 = tokens.iter().map(|t| t.score).sum();
            return Ok(GreedyOutput {
                tokens,
                score,
                alignments: None,
                frame_confidence: None,
                token_confidence: None,
                state: None,
            });
        }
        let enc_proj = self.project_encoder_cpu(encoder, steps)?;
        self.decode_greedy_from_enc_proj(&enc_proj, steps, cfg, fusion, partial, return_state)
    }

    /// Greedy TDT decode directly from pre-projected encoder frames `[steps][640]` (including `enc_bias`).
    pub fn decode_greedy_from_enc_proj(
        &mut self,
        enc_proj: &[f32],
        steps: usize,
        cfg: &GreedyConfig,
        fusion: &[(&dyn FusionModel, f32)],
        partial: Option<&DecoderState>,
        return_state: bool,
    ) -> Result<GreedyOutput> {
        if enc_proj.len() != steps * JOINT_HIDDEN {
            bail!(
                "enc_proj has {} values, expected {}",
                enc_proj.len(),
                steps * JOINT_HIDDEN
            );
        }
        if cfg.temperature <= 0.0 || !cfg.temperature.is_finite() {
            bail!("temperature must be positive and finite, got {}", cfg.temperature);
        }
        if cfg.confidence.enabled() {
            cfg.confidence.validate()?;
        }

        let (mut predictor, mut last_token, frame_offset, start_frame) = match partial {
            Some(prev) => {
                self.restore_state(prev)?;
                (
                    prev.predictor_out.clone(),
                    prev.last_token,
                    prev.decoded_frames,
                    prev.time_jump as usize,
                )
            }
            None => {
                self.reset();
                (self.predict(BLANK), BLANK, 0u32, 0usize)
            }
        };
        let mut pred_proj = self.project_pred(&predictor);

        let inv_temp = 1.0 / cfg.temperature;
        let scale_temp = (cfg.temperature - 1.0).abs() >= 1e-7;
        let track_conf = cfg.confidence.enabled();
        let has_fusion = !fusion.is_empty();

        // Shallow-fusion state tracking (`tdt_label_looping.py:396-438, 481-491, 562-568`).
        let mut fusion_states: Vec<usize> = fusion.iter().map(|(m, _)| m.init_state()).collect();
        let mut fusion_scores_sum = if has_fusion {
            vec![0.0f32; NON_BLANK_VOCAB]
        } else {
            Vec::new()
        };
        let mut fusion_tmp_scores = if has_fusion {
            vec![0.0f32; NON_BLANK_VOCAB]
        } else {
            Vec::new()
        };
        let mut fusion_cands: Vec<Vec<usize>> = if has_fusion {
            vec![vec![0usize; NON_BLANK_VOCAB]; fusion.len()]
        } else {
            Vec::new()
        };
        let mut fusion_dirty = has_fusion;

        let mut tokens = Vec::new();
        let mut total_score = 0.0f32;
        let mut alignments: Option<Vec<FrameAlignment>> =
            cfg.preserve_alignments.then(Vec::new);
        let mut frame_conf: Vec<FrameConfidence> = Vec::new();
        let mut frame_labels: Vec<Vec<u32>> = Vec::new();
        let mut logits = vec![0.0f32; JOINT_VOCAB];
        let mut projected = vec![0.0f32; JOINT_HIDDEN];
        let mut frame = start_frame;

        while frame < steps {
            let mut symbols = 0usize;
            loop {
                let enc_frame = &enc_proj[frame * JOINT_HIDDEN..(frame + 1) * JOINT_HIDDEN];
                self.joint_from_projected(enc_frame, &pred_proj, &mut projected, &mut logits);
                if scale_temp {
                    for v in &mut logits {
                        *v *= inv_temp;
                    }
                }
                let (mut token, mut tok_score) = argmax(&logits[..VOCAB]);
                if has_fusion {
                    if fusion_dirty {
                        fusion_scores_sum.fill(0.0);
                        for (m_idx, &(model, alpha)) in fusion.iter().enumerate() {
                            model.advance_into(
                                fusion_states[m_idx],
                                &mut fusion_tmp_scores,
                                &mut fusion_cands[m_idx],
                            );
                            for (dst, &s) in fusion_scores_sum.iter_mut().zip(&fusion_tmp_scores) {
                                *dst += s * alpha;
                            }
                        }
                        fusion_dirty = false;
                    }
                    // Preserve blank vs non-blank category (`tdt_label_looping.py:435-437`).
                    if token as u32 != BLANK {
                        let mut best_idx = 0usize;
                        let mut best_val = f32::NEG_INFINITY;
                        for i in 0..NON_BLANK_VOCAB {
                            let v = logits[i] + fusion_scores_sum[i];
                            if v > best_val {
                                best_val = v;
                                best_idx = i;
                            }
                        }
                        token = best_idx;
                        tok_score = best_val;
                    }
                }
                let (duration_index, whole_margin) = argmax(&logits[VOCAB..]);
                let duration = duration_index as u32;
                let mut second = f32::NEG_INFINITY;
                for (index, value) in logits[VOCAB..].iter().enumerate() {
                    if index != duration_index && *value > second {
                        second = *value;
                    }
                }
                let duration_margin = whole_margin - second;
                let mut raw = [0.0f32; DURATIONS];
                raw.copy_from_slice(&logits[VOCAB..]);
                let advance = duration.max(1) as usize;
                let global_frame = frame_offset + frame as u32;

                if let Some(align_vec) = &mut alignments {
                    let entry = AlignmentEntry {
                        label: token as u32,
                        duration: if token as u32 == BLANK {
                            advance as u32
                        } else {
                            duration
                        },
                        logits: logits.clone(),
                    };
                    if align_vec.last().is_some_and(|fa| fa.frame == global_frame) {
                        align_vec.last_mut().unwrap().steps.push(entry);
                    } else {
                        align_vec.push(FrameAlignment {
                            frame: global_frame,
                            steps: vec![entry],
                        });
                    }
                }

                if track_conf {
                    let conf_val = confidence::step_confidence(&logits, &cfg.confidence);
                    if frame_conf.last().is_some_and(|fc| fc.frame == global_frame) {
                        frame_conf.last_mut().unwrap().values.push(conf_val);
                        frame_labels.last_mut().unwrap().push(token as u32);
                    } else {
                        frame_conf.push(FrameConfidence {
                            frame: global_frame,
                            values: vec![conf_val],
                        });
                        frame_labels.push(vec![token as u32]);
                    }
                }

                if token as u32 == BLANK {
                    frame += advance;
                    break;
                }
                tokens.push(Token {
                    id: token as u32,
                    frame: global_frame,
                    duration,
                    duration_margin,
                    duration_logits: raw,
                    score: tok_score,
                });
                total_score += tok_score;
                last_token = token as u32;
                if has_fusion {
                    for m_idx in 0..fusion.len() {
                        fusion_states[m_idx] = fusion_cands[m_idx][token];
                    }
                    fusion_dirty = true;
                }
                predictor = self.predict(last_token);
                pred_proj = self.project_pred(&predictor);
                symbols += 1;
                if duration == 0 {
                    if let Some(cap) = cfg.max_symbols_per_step {
                        if symbols >= cap {
                            frame += 1;
                            break;
                        }
                    }
                    continue;
                }
                frame += advance;
                break;
            }
        }

        let token_confidence = if cfg.confidence.wants_token() {
            Some(confidence::compute_token_confidence(
                &frame_conf,
                &frame_labels,
                &cfg.confidence,
            ))
        } else {
            None
        };
        let frame_confidence = if cfg.confidence.preserve_frame_confidence {
            Some(frame_conf)
        } else {
            None
        };

        let state = if return_state {
            let time_jump = frame.saturating_sub(steps) as u32;
            Some(self.snapshot_state(
                last_token,
                &predictor,
                frame_offset + steps as u32,
                time_jump,
            ))
        } else {
            None
        };

        Ok(GreedyOutput {
            tokens,
            score: total_score,
            alignments,
            frame_confidence,
            token_confidence,
            state,
        })
    }
}

fn argmax(values: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    (best, best_value)
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_prefers_the_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.2]).0, 1);
        assert_eq!(argmax(&[-1.0, -2.0]).0, 0);
    }

    #[test]
    fn frame_seconds_match_tdt_grid() {
        assert!((FRAME_SECONDS - 0.08).abs() < 1e-9);
    }
}
