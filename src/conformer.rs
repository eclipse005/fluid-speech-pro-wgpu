//! Conformer encoder: relative-position attention, convolution module, FF blocks.
//!
//! Mirrors NeMo 2.7.3 `ConformerLayer.forward`, `ConformerConvolution.forward`,
//! `RelPositionMultiHeadAttention.forward` and `RelPositionalEncoding.forward`
//! (sources dumped in `notes/nemo_conformer_src.txt`). Dropout is inactive in
//! eval, so it is omitted; `fc_factor` is the 0.5 residual weight the feed-forward
//! blocks use.
//!
//! Layouts here are time-major `[time][d_model]`, the same as `fv_insert` output;
//! NeMo returns channel-major from the encoder and transposes internally.

use anyhow::{Context, Result, bail};

use rayon::prelude::*;

use crate::gemm::{PackW, gemm_into, gemm_right_row_major_serial, gemm_rows_serial};
use crate::torch::Store;

pub const D_MODEL: usize = 1024;
pub const HEADS: usize = 8;
pub const HEAD_DIM: usize = D_MODEL / HEADS;
pub const FFN: usize = 4096;
pub const LAYERS: usize = 24;
pub const CONV_KERNEL: usize = 9;
pub const LAYER_NORM_EPS: f32 = 1e-5;
const BATCH_NORM_EPS: f32 = 1e-5;
/// Residual weight for both feed-forward blocks.
pub const FC_FACTOR: f32 = 0.5;

/// Conformer self-attention mode, mirroring `ConformerEncoder.change_attention_model`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AttentionModel {
    /// Full relative-position Transformer-XL self-attention (`rel_pos`, official default).
    #[default]
    RelPos,
    /// Local sliding-window relative-position self-attention (`rel_pos_local_attn`),
    /// attending to `[i - left, i + right]` (`O(T * (left + right + 1))` compute and O(1) score memory).
    RelPosLocalAttn { left: usize, right: usize },
}

impl AttentionModel {
    pub fn validate(self) -> Result<()> {
        if let AttentionModel::RelPosLocalAttn { left, right } = self {
            if left == 0 && right == 0 {
                bail!("rel_pos_local_attn requires max(left, right) > 0");
            }
        }
        Ok(())
    }
}

/// `y = x @ W^T + b`, with `W` repacked once so the inner loop is a blocked,
/// vectorised GEMM instead of a per-row dot product.
pub struct Linear {
    packed: PackW,
    bias: Option<Vec<f32>>,
    out_features: usize,
    in_features: usize,
}

impl Linear {
    fn from_store(store: &Store, weight: &str, bias: Option<&str>) -> Result<Self> {
        let weight_values = store
            .f32(weight)
            .with_context(|| format!("read {weight}"))?;
        let out_features = store
            .require(weight)
            .map(|tensor| tensor.shape.first().copied().unwrap_or(0))
            .unwrap_or(0);
        if out_features == 0 || weight_values.len() % out_features != 0 {
            bail!(
                "{weight} has {} values, not divisible by {out_features}",
                weight_values.len()
            );
        }
        let in_features = weight_values.len() / out_features;
        let bias = match bias {
            Some(name) => Some(store.f32(name).with_context(|| format!("read {name}"))?),
            None => None,
        };
        Ok(Self {
            packed: PackW::from_vec(weight_values, out_features, in_features),
            bias,
            out_features,
            in_features,
        })
    }

    /// `y = x @ Wᵀ + b` for `[steps][in]` → `[steps][out]`.
    fn forward(&self, input: &[f32], steps: usize, out: &mut [f32]) {
        debug_assert_eq!(input.len(), steps * self.in_features);
        debug_assert_eq!(out.len(), steps * self.out_features);
        gemm_into(input, &self.packed, out, steps);
        if let Some(bias) = &self.bias {
            for step in 0..steps {
                let row = &mut out[step * self.out_features..(step + 1) * self.out_features];
                for (slot, offset) in row.iter_mut().zip(bias) {
                    *slot += offset;
                }
            }
        }
    }
}

pub struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNorm {
    fn from_store(store: &Store, weight: &str, bias: &str) -> Result<Self> {
        Ok(Self {
            weight: store.f32(weight)?,
            bias: store.f32(bias)?,
        })
    }

    fn forward(&self, input: &[f32], steps: usize, out: &mut [f32]) {
        // Hard asserts: the parallel zip below relies on both buffers dividing
        // evenly by D_MODEL.
        assert_eq!(input.len(), steps * D_MODEL);
        assert_eq!(out.len(), steps * D_MODEL);
        // Each step is independent, so the reduction and the normalisation run in
        // one task per frame.
        out.par_chunks_mut(D_MODEL)
            .zip(input.par_chunks(D_MODEL))
            .for_each(|(target, row)| {
                let mean = row.iter().sum::<f32>() / D_MODEL as f32;
                let variance = row
                    .iter()
                    .map(|value| {
                        let delta = value - mean;
                        delta * delta
                    })
                    .sum::<f32>()
                    / D_MODEL as f32;
                let inv_std = 1.0 / (variance + LAYER_NORM_EPS).sqrt();
                for index in 0..D_MODEL {
                    target[index] =
                        (row[index] - mean) * inv_std * self.weight[index] + self.bias[index];
                }
            });
    }
}

pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl Linear {
    /// Row-major `[out_features][in_features]` weights, the layout GEMM expects.
    pub fn weight_row_major(&self) -> &[f32] {
        self.packed.row_major()
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

}

impl FeedForward {
    pub fn from_store(store: &Store, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear1: Linear::from_store(store, &format!("{prefix}.linear1.weight"), None)?,
            linear2: Linear::from_store(store, &format!("{prefix}.linear2.weight"), None)?,
        })
    }

    /// The `d_model -> ffn` projection, applied before the SiLU.
    pub fn linear1(&self) -> &Linear {
        &self.linear1
    }

    /// The `ffn -> d_model` projection after the SiLU.
    pub fn linear2(&self) -> &Linear {
        &self.linear2
    }

    fn forward(&self, input: &[f32], steps: usize) -> Vec<f32> {
        let mut hidden = vec![0.0f32; steps * FFN];
        self.linear1.forward(input, steps, &mut hidden);
        hidden
            .par_iter_mut()
            .for_each(|value| *value = silu(*value));
        let mut out = vec![0.0f32; steps * D_MODEL];
        self.linear2.forward(&hidden, steps, &mut out);
        out
    }
}

pub struct Convolution {
    pointwise1: Linear,
    depthwise: Vec<f32>,
    batch_norm_weight: Vec<f32>,
    batch_norm_bias: Vec<f32>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    pointwise2: Linear,
}

impl Convolution {
    pub fn from_store(store: &Store, prefix: &str) -> Result<Self> {
        let pointwise1 =
            Linear::from_store(store, &format!("{prefix}.pointwise_conv1.weight"), None)?;
        let pointwise2 =
            Linear::from_store(store, &format!("{prefix}.pointwise_conv2.weight"), None)?;
        Ok(Self {
            pointwise1,
            depthwise: store.f32(&format!("{prefix}.depthwise_conv.weight"))?,
            batch_norm_weight: store.f32(&format!("{prefix}.batch_norm.weight"))?,
            batch_norm_bias: store.f32(&format!("{prefix}.batch_norm.bias"))?,
            running_mean: store.f32(&format!("{prefix}.batch_norm.running_mean"))?,
            running_var: store.f32(&format!("{prefix}.batch_norm.running_var"))?,
            pointwise2,
        })
    }

    /// Folds the eval-mode BatchNorm into a per-channel `scale`/`shift` pair,
    /// which is what the GPU kernel consumes.
    pub fn folded_batch_norm(&self) -> (Vec<f32>, Vec<f32>) {
        let mut scale = vec![0.0f32; D_MODEL];
        let mut shift = vec![0.0f32; D_MODEL];
        for channel in 0..D_MODEL {
            scale[channel] = self.batch_norm_weight[channel]
                / (self.running_var[channel] + BATCH_NORM_EPS).sqrt();
            shift[channel] = self.batch_norm_bias[channel] - self.running_mean[channel] * scale[channel];
        }
        (scale, shift)
    }

    /// `Conv1d(1024 -> 2048, k=1)`, the projection the GLU splits.
    pub fn pointwise1(&self) -> &Linear {
        &self.pointwise1
    }

    pub fn pointwise2(&self) -> &Linear {
        &self.pointwise2
    }

    /// The depthwise convolution weights, `[channel][tap]`.
    pub fn depthwise_weights(&self) -> &[f32] {
        &self.depthwise
    }

    pub fn forward(&self, input: &[f32], steps: usize) -> Result<Vec<f32>> {
        // pointwise_conv1 is Conv1d(1024 -> 2048, k=1): a linear layer.
        let mut gates = vec![0.0f32; steps * 2 * D_MODEL];
        self.pointwise1.forward(input, steps, &mut gates);

        // GLU over the channel axis: a * sigmoid(b), a = first half.
        let mut activated = vec![0.0f32; steps * D_MODEL];
        activated
            .par_chunks_mut(D_MODEL)
            .enumerate()
            .for_each(|(step, target)| {
                let row = &gates[step * 2 * D_MODEL..(step + 1) * 2 * D_MODEL];
                for channel in 0..D_MODEL {
                    target[channel] = row[channel] * sigmoid(row[D_MODEL + channel]);
                }
            });

        // Depthwise Conv1d, kernel 9, symmetric padding 4 (`is_causal=False`).
        let padding = CONV_KERNEL / 2;
        let mut convolved = vec![0.0f32; steps * D_MODEL];
        convolved
            .par_chunks_mut(D_MODEL)
            .enumerate()
            .for_each(|(step, target)| {
                for channel in 0..D_MODEL {
                    let weights =
                        &self.depthwise[channel * CONV_KERNEL..(channel + 1) * CONV_KERNEL];
                    let mut sum = 0.0f32;
                    for (tap, weight) in weights.iter().enumerate() {
                        let source = step as isize + tap as isize - padding as isize;
                        if source < 0 || source >= steps as isize {
                            continue;
                        }
                        sum += weight * activated[source as usize * D_MODEL + channel];
                    }
                    target[channel] = sum;
                }
            });

        // BatchNorm1d in eval mode, then Swish.
        // The eval-mode BatchNorm is a per-channel affine map, so fold it once and
        // then run the (independent) steps in parallel.
        let (scale, shift) = self.folded_batch_norm();
        convolved.par_chunks_mut(D_MODEL).for_each(|row| {
            for channel in 0..D_MODEL {
                row[channel] = silu(row[channel] * scale[channel] + shift[channel]);
            }
        });

        let mut out = vec![0.0f32; steps * D_MODEL];
        self.pointwise2.forward(&convolved, steps, &mut out);
        Ok(out)
    }
}

/// Relative-position multi-head attention. `pub` so the wgpu path can be
/// compared against it stage by stage.
///
/// The three input projections are fused into one `[3 * D_MODEL, D_MODEL]` Linear
/// so a layer does one GEMM instead of three, and the per-head work is one Rayon
/// task per head running three serial GEMMs. NeMo's scalar formulation is
/// preserved exactly; only the summation order inside each GEMM changes.
pub struct Attention {
    /// `[Q; K; V]` stacked vertically, bias-free.
    qkv: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Vec<f32>,
    pos_bias_v: Vec<f32>,
    scale: f32,
}

impl Attention {
    pub fn from_store(store: &Store, prefix: &str) -> Result<Self> {
        // NeMo keeps `linear_q`/`linear_k`/`linear_v` separate; stacking their
        // weights vertically turns three GEMMs into one without changing the
        // result (`[x @ Wq^T | x @ Wk^T | x @ Wv^T]`).
        let mut qkv_weight = Vec::with_capacity(3 * D_MODEL * D_MODEL);
        for name in ["linear_q", "linear_k", "linear_v"] {
            qkv_weight.extend(store.f32(&format!("{prefix}.{name}.weight"))?);
        }
        Ok(Self {
            qkv: Linear {
                packed: PackW::from_vec(qkv_weight, 3 * D_MODEL, D_MODEL),
                bias: None,
                out_features: 3 * D_MODEL,
                in_features: D_MODEL,
            },
            linear_out: Linear::from_store(store, &format!("{prefix}.linear_out.weight"), None)?,
            linear_pos: Linear::from_store(store, &format!("{prefix}.linear_pos.weight"), None)?,
            pos_bias_u: store.f32(&format!("{prefix}.pos_bias_u"))?,
            pos_bias_v: store.f32(&format!("{prefix}.pos_bias_v"))?,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        })
    }

    /// The fused `[Q; K; V]` weight, `[3 * D_MODEL][D_MODEL]`, bias-free.
    pub fn qkv_weight(&self) -> &PackW {
        &self.qkv.packed
    }

    /// The `linear_pos` weight, `[D_MODEL][D_MODEL]`.
    pub fn pos_weight(&self) -> &PackW {
        &self.linear_pos.packed
    }

    /// The `linear_out` weight, `[D_MODEL][D_MODEL]`.
    pub fn out_weight(&self) -> &PackW {
        &self.linear_out.packed
    }

    pub fn pos_bias_u(&self) -> &[f32] {
        &self.pos_bias_u
    }

    pub fn pos_bias_v(&self) -> &[f32] {
        &self.pos_bias_v
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// `pos_emb` is `[2*steps-1][D_MODEL]`.
    pub fn forward(&self, input: &[f32], steps: usize, pos_emb: &[f32]) -> Vec<f32> {
        let mut qkv = vec![0.0f32; steps * 3 * D_MODEL];
        self.qkv.forward(input, steps, &mut qkv);

        let pos_len = 2 * steps - 1;
        let mut projected_pos = vec![0.0f32; pos_len * D_MODEL];
        self.linear_pos
            .forward(pos_emb, pos_len, &mut projected_pos);

        // One task per head: each owns its own scratch and writes only the head's
        // slice of the context, so no locking is needed. Queries are processed in
        // fixed-height chunks with direct relative-position indexing so long audio
        // (e.g. 25 minutes) uses O(R * T) instead of materialising three O(T^2)
        // matrices per head (~80 GB across 8 threads).
        const ATTN_CHUNK_ROWS: usize = 256;
        let chunk_cap = steps.min(ATTN_CHUNK_ROWS);
        let contexts: Vec<Vec<f32>> = (0..HEADS)
            .into_par_iter()
            .map(|head| {
                let offset = head * HEAD_DIM;
                let stride = 3 * D_MODEL;
                let q_plain = gather_strided(&qkv, steps, stride, offset);
                let k_head = gather_strided(&qkv, steps, stride, D_MODEL + offset);
                let v_head = gather_strided(&qkv, steps, stride, 2 * D_MODEL + offset);
                let p_head = gather_strided(&projected_pos, pos_len, D_MODEL, offset);

                let bias_u = &self.pos_bias_u[offset..offset + HEAD_DIM];
                let bias_v = &self.pos_bias_v[offset..offset + HEAD_DIM];

                // (q + bias_u) @ k^T  and  (q + bias_v) @ p^T
                let mut q_u = q_plain.clone();
                let mut q_v = q_plain;
                for row in 0..steps {
                    let base = row * HEAD_DIM;
                    for dim in 0..HEAD_DIM {
                        q_u[base + dim] += bias_u[dim];
                        q_v[base + dim] += bias_v[dim];
                    }
                }

                let mut content = vec![0.0f32; chunk_cap * steps];
                let mut position = vec![0.0f32; chunk_cap * pos_len];
                let mut context = vec![0.0f32; steps * HEAD_DIM];

                for row_offset in (0..steps).step_by(chunk_cap) {
                    let chunk_rows = (steps - row_offset).min(chunk_cap);
                    let q_start = row_offset * HEAD_DIM;
                    let q_end = (row_offset + chunk_rows) * HEAD_DIM;

                    let content_chunk = &mut content[..chunk_rows * steps];
                    gemm_rows_serial(
                        &q_u[q_start..q_end],
                        &k_head,
                        content_chunk,
                        chunk_rows,
                        steps,
                        HEAD_DIM,
                    );

                    let position_chunk = &mut position[..chunk_rows * pos_len];
                    gemm_rows_serial(
                        &q_v[q_start..q_end],
                        &p_head,
                        position_chunk,
                        chunk_rows,
                        pos_len,
                        HEAD_DIM,
                    );

                    // Relative-position shift equals reading column `(steps - 1 - global_row) + key`
                    // of `position`, avoiding the full `[steps][2*steps]` padded copy.
                    for local_row in 0..chunk_rows {
                        let global_row = row_offset + local_row;
                        let base = local_row * steps;
                        let pos_base = local_row * pos_len + (steps - 1 - global_row);
                        for key in 0..steps {
                            content_chunk[base + key] =
                                (content_chunk[base + key] + position_chunk[pos_base + key])
                                    * self.scale;
                        }
                        softmax(&mut content_chunk[base..base + steps]);
                    }

                    // `scores @ V`: V is already `[steps][HEAD_DIM]`, which is the
                    // row-major `[k][n]` form this entry point expects.
                    gemm_right_row_major_serial(
                        content_chunk,
                        &v_head,
                        &mut context[q_start..q_end],
                        chunk_rows,
                        HEAD_DIM,
                        steps,
                    );
                }
                context
            })
            .collect();

        let mut combined = vec![0.0f32; steps * D_MODEL];
        for (head, context) in contexts.iter().enumerate() {
            let offset = head * HEAD_DIM;
            for row in 0..steps {
                combined[row * D_MODEL + offset..row * D_MODEL + offset + HEAD_DIM]
                    .copy_from_slice(&context[row * HEAD_DIM..(row + 1) * HEAD_DIM]);
            }
        }

        let mut out = vec![0.0f32; steps * D_MODEL];
        self.linear_out.forward(&combined, steps, &mut out);
        out
    }

    /// Local sliding-window relative-position attention (`rel_pos_local_attn`).
    ///
    /// `pos_emb` is `[left + right + 1][D_MODEL]` from [`local_positional_encoding`].
    /// Mirrors NeMo's `RelPositionMultiHeadAttentionLongformer.forward` (`multi_head_attention.py:419-585`).
    pub fn forward_local(
        &self,
        input: &[f32],
        steps: usize,
        pos_emb: &[f32],
        left: usize,
        right: usize,
    ) -> Vec<f32> {
        let mut qkv = vec![0.0f32; steps * 3 * D_MODEL];
        self.qkv.forward(input, steps, &mut qkv);

        let pos_len = left + right + 1;
        let mut projected_pos = vec![0.0f32; pos_len * D_MODEL];
        self.linear_pos
            .forward(pos_emb, pos_len, &mut projected_pos);

        let w = left.max(right);
        let start_pos = w - left;
        let end_pos = w + right;
        let right_bd_start = 2 * w - right;

        let contexts: Vec<Vec<f32>> = (0..HEADS)
            .into_par_iter()
            .map(|head| {
                let offset = head * HEAD_DIM;
                let stride = 3 * D_MODEL;
                let q_plain = gather_strided(&qkv, steps, stride, offset);
                let k_head = gather_strided(&qkv, steps, stride, D_MODEL + offset);
                let v_head = gather_strided(&qkv, steps, stride, 2 * D_MODEL + offset);
                let p_head = gather_strided(&projected_pos, pos_len, D_MODEL, offset);

                let bias_u = &self.pos_bias_u[offset..offset + HEAD_DIM];
                let bias_v = &self.pos_bias_v[offset..offset + HEAD_DIM];

                let mut q_u = q_plain.clone();
                let mut q_v = q_plain;
                for row in 0..steps {
                    let base = row * HEAD_DIM;
                    for dim in 0..HEAD_DIM {
                        q_u[base + dim] += bias_u[dim];
                        q_v[base + dim] += bias_v[dim];
                    }
                }

                // `bd = (q + bias_v) @ p^T`, shape `[steps][pos_len]`.
                let mut bd = vec![0.0f32; steps * pos_len];
                gemm_rows_serial(&q_v, &p_head, &mut bd, steps, pos_len, HEAD_DIM);

                let win_len = end_pos - start_pos + 1;
                let mut scores = Vec::with_capacity(win_len);
                let mut keys = Vec::with_capacity(win_len);
                let mut context = vec![0.0f32; steps * HEAD_DIM];

                for i in 0..steps {
                    scores.clear();
                    keys.clear();
                    let qu_row = &q_u[i * HEAD_DIM..(i + 1) * HEAD_DIM];
                    let bd_row = &bd[i * pos_len..(i + 1) * pos_len];
                    for col in start_pos..=end_pos {
                        let key_i = i as isize + col as isize - w as isize;
                        if key_i < 0 || key_i >= steps as isize {
                            continue;
                        }
                        let k_u = key_i as usize;
                        let k_row = &k_head[k_u * HEAD_DIM..(k_u + 1) * HEAD_DIM];
                        let mut ac = 0.0f32;
                        for d in 0..HEAD_DIM {
                            ac += qu_row[d] * k_row[d];
                        }
                        let pos_term = if col < left {
                            bd_row[col]
                        } else if col >= right_bd_start {
                            bd_row[left + (col - right_bd_start)]
                        } else {
                            0.0
                        };
                        scores.push((ac + pos_term) * self.scale);
                        keys.push(k_u);
                    }
                    softmax(&mut scores);
                    let out_row = &mut context[i * HEAD_DIM..(i + 1) * HEAD_DIM];
                    for (&prob, &k_u) in scores.iter().zip(&keys) {
                        let v_row = &v_head[k_u * HEAD_DIM..(k_u + 1) * HEAD_DIM];
                        for d in 0..HEAD_DIM {
                            out_row[d] += prob * v_row[d];
                        }
                    }
                }
                context
            })
            .collect();

        let mut combined = vec![0.0f32; steps * D_MODEL];
        for (head, context) in contexts.iter().enumerate() {
            let offset = head * HEAD_DIM;
            for row in 0..steps {
                combined[row * D_MODEL + offset..row * D_MODEL + offset + HEAD_DIM]
                    .copy_from_slice(&context[row * HEAD_DIM..(row + 1) * HEAD_DIM]);
            }
        }

        let mut out = vec![0.0f32; steps * D_MODEL];
        self.linear_out.forward(&combined, steps, &mut out);
        out
    }
}

/// Copies `rows` slices of `HEAD_DIM` floats starting at `base + offset` in each
/// `stride`-wide source row.
fn gather_strided(source: &[f32], rows: usize, stride: usize, offset: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * HEAD_DIM];
    for row in 0..rows {
        let from = row * stride + offset;
        out[row * HEAD_DIM..(row + 1) * HEAD_DIM].copy_from_slice(&source[from..from + HEAD_DIM]);
    }
    out
}

/// NeMo pads one zero column on the left before `rel_shift`, so the `(T, 2T-1)`
/// position scores become a `(T, 2T)` block first. Kept under `cfg(test)` to verify
/// the direct relative-index formula against NeMo's padded reshape.
#[cfg(test)]
fn rel_shift_padded(matrix: &[f32], steps: usize) -> Vec<f32> {
    let pos_len = 2 * steps - 1;
    let width = 2 * steps;
    let mut padded = vec![0.0f32; steps * width];
    for row in 0..steps {
        padded[row * width + 1..(row + 1) * width]
            .copy_from_slice(&matrix[row * pos_len..(row + 1) * pos_len]);
    }
    rel_shift(&padded, steps)
}

/// NeMo's `rel_shift`: pad a zero on the left, view as `(2T, T)`, drop the first
/// row, then view as `(T, 2T-1)`. The reshape spans all queries, which is what
/// makes it a relative shift rather than a per-row roll.
#[cfg(test)]
fn rel_shift(matrix: &[f32], steps: usize) -> Vec<f32> {
    let width = 2 * steps;
    debug_assert_eq!(matrix.len(), steps * width);
    // Dropping the first row of the `(2T, T)` view skips `steps` elements.
    let dropped = &matrix[steps..];
    let mut out = vec![0.0f32; steps * (width - 1)];
    out.copy_from_slice(dropped);
    out
}

fn softmax(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in row.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    let inverse = 1.0 / sum;
    for value in row.iter_mut() {
        *value *= inverse;
    }
}

impl ConformerLayer {
    /// Writes this layer's sub-block output when `FLUID_DUMP_STAGES` is set and
    /// the layer is the first one, so a piecewise comparison is possible.
    fn dump(&self, name: &str, values: &[f32]) {
        if self.index != 0 {
            return;
        }
        let Ok(dir) = std::env::var("FLUID_DUMP_STAGES") else {
            return;
        };
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let _ = std::fs::write(format!("{dir}/rust_conformer_{name}.f32"), bytes);
    }
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

struct ConformerLayer {
    index: usize,
    norm_feed_forward1: LayerNorm,
    feed_forward1: FeedForward,
    norm_self_att: LayerNorm,
    attention: Attention,
    norm_conv: LayerNorm,
    convolution: Convolution,
    norm_feed_forward2: LayerNorm,
    feed_forward2: FeedForward,
    norm_out: LayerNorm,
}

impl ConformerLayer {
    fn from_store(store: &Store, index: usize) -> Result<Self> {
        let prefix = format!("encoder.layers.{index}");
        Ok(Self {
            index,
            norm_feed_forward1: LayerNorm::from_store(
                store,
                &format!("{prefix}.norm_feed_forward1.weight"),
                &format!("{prefix}.norm_feed_forward1.bias"),
            )?,
            feed_forward1: FeedForward::from_store(store, &format!("{prefix}.feed_forward1"))?,
            norm_self_att: LayerNorm::from_store(
                store,
                &format!("{prefix}.norm_self_att.weight"),
                &format!("{prefix}.norm_self_att.bias"),
            )?,
            attention: Attention::from_store(store, &format!("{prefix}.self_attn"))?,
            norm_conv: LayerNorm::from_store(
                store,
                &format!("{prefix}.norm_conv.weight"),
                &format!("{prefix}.norm_conv.bias"),
            )?,
            convolution: Convolution::from_store(store, &format!("{prefix}.conv"))?,
            norm_feed_forward2: LayerNorm::from_store(
                store,
                &format!("{prefix}.norm_feed_forward2.weight"),
                &format!("{prefix}.norm_feed_forward2.bias"),
            )?,
            feed_forward2: FeedForward::from_store(store, &format!("{prefix}.feed_forward2"))?,
            norm_out: LayerNorm::from_store(
                store,
                &format!("{prefix}.norm_out.weight"),
                &format!("{prefix}.norm_out.bias"),
            )?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        steps: usize,
        pos_emb: &[f32],
        attn_model: AttentionModel,
    ) -> Result<Vec<f32>> {
        let mut residual = input.to_vec();

        let mut normed = vec![0.0f32; steps * D_MODEL];
        self.norm_feed_forward1
            .forward(&residual, steps, &mut normed);
        let ff1 = self.feed_forward1.forward(&normed, steps);
        self.dump("ff1", &ff1);
        for (slot, value) in residual.iter_mut().zip(ff1) {
            *slot += value * FC_FACTOR;
        }

        self.norm_self_att.forward(&residual, steps, &mut normed);
        let attended = match attn_model {
            AttentionModel::RelPos => self.attention.forward(&normed, steps, pos_emb),
            AttentionModel::RelPosLocalAttn { left, right } => {
                self.attention.forward_local(&normed, steps, pos_emb, left, right)
            }
        };
        self.dump("attn", &attended);
        for (slot, value) in residual.iter_mut().zip(attended) {
            *slot += value;
        }

        self.norm_conv.forward(&residual, steps, &mut normed);
        let convolved = self.convolution.forward(&normed, steps)?;
        self.dump("conv", &convolved);
        for (slot, value) in residual.iter_mut().zip(convolved) {
            *slot += value;
        }

        self.norm_feed_forward2
            .forward(&residual, steps, &mut normed);
        let ff2 = self.feed_forward2.forward(&normed, steps);
        self.dump("ff2", &ff2);
        for (slot, value) in residual.iter_mut().zip(ff2) {
            *slot += value * FC_FACTOR;
        }

        let mut out = vec![0.0f32; steps * D_MODEL];
        self.norm_out.forward(&residual, steps, &mut out);
        Ok(out)
    }
}

pub struct Conformer {
    layers: Vec<ConformerLayer>,
    attention_model: AttentionModel,
}

impl Conformer {
    pub fn from_store(store: &Store) -> Result<Self> {
        let mut layers = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            layers.push(ConformerLayer::from_store(store, index)?);
        }
        Ok(Self {
            layers,
            attention_model: AttentionModel::RelPos,
        })
    }

    pub fn attention_model(&self) -> AttentionModel {
        self.attention_model
    }

    /// Switches between full `rel_pos` and sliding-window `rel_pos_local_attn` with zero weight reload.
    pub fn set_attention_model(&mut self, model: AttentionModel) -> Result<()> {
        model.validate()?;
        self.attention_model = model;
        Ok(())
    }

    /// Runs every layer, returning time-major `[steps][1024]`.
    pub fn forward(&self, input: &[f32], steps: usize) -> Result<Vec<f32>> {
        let pos_emb = match self.attention_model {
            AttentionModel::RelPos => positional_encoding(steps),
            AttentionModel::RelPosLocalAttn { left, right } => {
                local_positional_encoding(left, right)
            }
        };
        let mut current = input.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current, steps, &pos_emb, self.attention_model)?;
        }
        Ok(current)
    }

    /// Runs the first `count` layers, for stage-by-stage comparison.
    pub fn forward_upto(&self, input: &[f32], steps: usize, count: usize) -> Result<Vec<f32>> {
        let pos_emb = match self.attention_model {
            AttentionModel::RelPos => positional_encoding(steps),
            AttentionModel::RelPosLocalAttn { left, right } => {
                local_positional_encoding(left, right)
            }
        };
        let mut current = input.to_vec();
        for layer in self.layers.iter().take(count) {
            current = layer.forward(&current, steps, &pos_emb, self.attention_model)?;
        }
        Ok(current)
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

/// NeMo's `LocalAttRelPositionalEncoding`: `left + right + 1` positions from `+left` down to `-right`.
pub fn local_positional_encoding(left: usize, right: usize) -> Vec<f32> {
    let total = left + right + 1;
    let mut table = vec![0.0f32; total * D_MODEL];
    let half = D_MODEL / 2;
    let mut div_term = Vec::with_capacity(half);
    for index in 0..half {
        div_term.push((-(10_000f64.ln() / D_MODEL as f64) * (2 * index) as f64).exp());
    }
    for row in 0..total {
        let position = (left as i64) - (row as i64);
        for index in 0..half {
            let angle = position as f64 * div_term[index];
            table[row * D_MODEL + 2 * index] = angle.sin() as f32;
            table[row * D_MODEL + 2 * index + 1] = angle.cos() as f32;
        }
    }
    table
}

/// NeMo's `RelPositionalEncoding`: `2*len-1` positions centred on the sequence,
/// sinusoid with `div_term = exp(arange(0, d, 2) * -(ln(10000) / d))`.
pub fn positional_encoding(steps: usize) -> Vec<f32> {
    let total = 2 * steps - 1;
    let mut table = vec![0.0f32; total * D_MODEL];
    let half = D_MODEL / 2;
    let mut div_term = Vec::with_capacity(half);
    for index in 0..half {
        // NeMo: exp(arange(0, d_model, 2) * -(ln(10000) / d_model)) — the
        // exponent advances by 2 per frequency, not by 1.
        div_term.push((-(10_000f64.ln() / D_MODEL as f64) * (2 * index) as f64).exp());
    }
    // NeMo's `extend_pe` uses `arange(length - 1, -length, -1)`: the first row is
    // position +(steps-1) and it descends to -(steps-1).
    for row in 0..total {
        let position = (steps as i64 - 1) - row as i64;
        for index in 0..half {
            let angle = position as f64 * div_term[index];
            table[row * D_MODEL + 2 * index] = angle.sin() as f32;
            table[row * D_MODEL + 2 * index + 1] = angle.cos() as f32;
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positional_encoding_is_centred() {
        let table = positional_encoding(3);
        assert_eq!(table.len(), 5 * D_MODEL);
        // Row 2 is position 0: sin(0)=0, cos(0)=1.
        assert!((table[2 * D_MODEL] - 0.0).abs() < 1e-7);
        assert!((table[2 * D_MODEL + 1] - 1.0).abs() < 1e-7);
        // Rows 1 and 3 are +-1 and mirror each other in the sine part.
        assert!((table[D_MODEL] + table[3 * D_MODEL]).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut row = vec![0.5f32, -1.0, 2.0, 0.0];
        softmax(&mut row);
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rel_shift_matches_torch_layout() {
        // steps=2, so the matrix is 2x4 with a zero in column 0:
        //   [[0, a, b, c],
        //    [0, d, e, f]]
        // torch views it as (4, 2) = [[0,a],[b,c],[0,d],[e,f]], drops the first
        // row, and views (2, 3) = [[b,c,0],[d,e,f]].
        let matrix = [0.0f32, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0];
        let shifted = rel_shift(&matrix, 2);
        assert_eq!(shifted, vec![2.0, 3.0, 0.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn direct_pos_index_matches_rel_shift_padded() {
        for steps in [1usize, 2, 7, 17] {
            let pos_len = 2 * steps - 1;
            let position: Vec<f32> = (0..steps * pos_len).map(|i| i as f32).collect();
            let shifted = rel_shift_padded(&position, steps);
            for row in 0..steps {
                let pos_base = row * pos_len + (steps - 1 - row);
                for key in 0..steps {
                    assert_eq!(
                        position[pos_base + key],
                        shifted[row * pos_len + key],
                        "steps={steps} row={row} key={key}"
                    );
                }
            }
        }
    }

    #[test]
    fn local_positional_encoding_matches_global_slice() {
        let global = positional_encoding(10);
        let local = local_positional_encoding(4, 2);
        assert_eq!(local.len(), 7 * D_MODEL);
        // Position +4..=-2 in global(10) starts at row (9 - 4) = 5.
        let slice = &global[5 * D_MODEL..(5 + 7) * D_MODEL];
        for (a, b) in local.iter().zip(slice) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn local_attention_with_full_window_matches_global() {
        let steps = 6usize;
        let lcg = |seed: u64, len: usize| -> Vec<f32> {
            let mut s = seed;
            (0..len)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let bits = ((s >> 33) as i32) as f32 / (i32::MAX as f32);
                    bits * 0.05
                })
                .collect()
        };
        let attn = Attention {
            qkv: Linear {
                packed: PackW::from_vec(lcg(1, 3 * D_MODEL * D_MODEL), 3 * D_MODEL, D_MODEL),
                bias: None,
                out_features: 3 * D_MODEL,
                in_features: D_MODEL,
            },
            linear_out: Linear {
                packed: PackW::from_vec(lcg(2, D_MODEL * D_MODEL), D_MODEL, D_MODEL),
                bias: None,
                out_features: D_MODEL,
                in_features: D_MODEL,
            },
            linear_pos: Linear {
                packed: PackW::from_vec(lcg(3, D_MODEL * D_MODEL), D_MODEL, D_MODEL),
                bias: None,
                out_features: D_MODEL,
                in_features: D_MODEL,
            },
            pos_bias_u: lcg(4, D_MODEL),
            pos_bias_v: lcg(5, D_MODEL),
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        };
        let input = lcg(6, steps * D_MODEL);
        let global_pe = positional_encoding(steps);
        let global_out = attn.forward(&input, steps, &global_pe);

        // When left == right >= steps - 1, local window covers all keys identically.
        let w = steps + 2;
        let local_pe = local_positional_encoding(w, w);
        let local_out = attn.forward_local(&input, steps, &local_pe, w, w);

        let max_diff = global_out
            .iter()
            .zip(&local_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "local full-window attention diverged from global: max_diff={max_diff:.3e}"
        );
    }
}
