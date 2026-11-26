//! wgpu attention kernels: scores+softmax and the context mix.
//!
//! Split out from `gpu.rs` to keep that file focused on the GEMM/pointwise
//! kernels. The pipelines here are created on demand: this is the verification
//! path, which reads every intermediate back anyway, and the production shape will
//! build all pipelines once inside an encoder scratch pool (see
//! `notes/perf_progress.md`).

use anyhow::Result;

use crate::gpu::GpuGemm;

/// `AttnParams`, 32 bytes — mirrors `attn_scores_softmax.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnParams {
    total_steps: u32,
    max_steps: u32,
    width: u32,
    head_dim: u32,
    scale: f32,
    heads: u32,
    pos_width: u32,
    /// Which head this dispatch handles.
    head_offset: u32,
    /// Where this head's score window starts, in floats.
    score_offset: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// `ContextParams`, 16 bytes — mirrors `attn_context.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ContextParams {
    total_steps: u32,
    max_steps: u32,
    head_dim: u32,
    head_offset: u32,
    /// Where this head's score window starts, in floats.
    score_offset: u32,
    heads: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl GpuGemm {
    /// Scores + softmax for every (head, query).
    ///
    /// `qkv` is `[steps][3*D_MODEL]` (Q | K | V), `pos` is the projected position
    /// table `[2*steps-1][D_MODEL]`, `bias` is `pos_bias_u` followed by
    /// `pos_bias_v`. Returns `[heads][steps][steps]`.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_scores_softmax(
        &self,
        qkv: &[f32],
        pos: &[f32],
        bias: &[f32],
        steps: usize,
        width: usize,
        head_dim: usize,
        heads: usize,
        pos_width: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        let pipeline = self.attn_scores_pipeline(steps)?;
        let buffer_qkv = self.upload_plain("attn_qkv", qkv);
        let buffer_pos = self.upload_plain("attn_pos", pos);
        let buffer_bias = self.upload_plain("attn_bias", bias);

        // The kernel handles one head into a `[steps][steps]` window, so heads are
        // dispatched in turn and their results concatenated.
        let count = heads * steps * steps;
        let buffer_out = self.storage_out("attn_scores", count);
        let mut encoder = self.encoder("attn_scores");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attn_scores"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            for head in 0..heads {
                let params = self.uniform_buffer(
                    "attn_params",
                    &AttnParams {
                        total_steps: steps as u32,
                        max_steps: steps as u32,
                        width: width as u32,
                        head_dim: head_dim as u32,
                        scale,
                        heads: heads as u32,
                        pos_width: pos_width as u32,
                        head_offset: head as u32,
                    score_offset: (head * steps * steps) as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    },
                );
                let bind_group = self.bind_group(
                    "attn_scores",
                    &pipeline.layout,
                    &[
                        (0, params.as_entire_binding()),
                        (1, buffer_qkv.as_entire_binding()),
                        (2, buffer_pos.as_entire_binding()),
                        (3, buffer_bias.as_entire_binding()),
                        (4, buffer_out.as_entire_binding()),
                    ],
                );
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(steps as u32, 1, 1);
            }
        }
        self.submit_and_read(encoder, &buffer_out, count)
    }

    /// `scores @ V`, one workgroup per (head, query). Returns `[steps][D_MODEL]`.
    pub fn attn_context(
        &self,
        scores: &[f32],
        qkv: &[f32],
        steps: usize,
        width: usize,
        head_dim: usize,
        heads: usize,
    ) -> Result<Vec<f32>> {
        let pipeline = self.attn_context_pipeline()?;
        let buffer_scores = self.upload_plain("ctx_scores", scores);
        let buffer_qkv = self.upload_plain("ctx_qkv", qkv);

        let d_model = width / 3;
        let count = steps * d_model;
        let buffer_out = self.storage_out("attn_context", count);
        let mut encoder = self.encoder("attn_context");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attn_context"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            for head in 0..heads {
                let params = self.uniform_buffer(
                    "ctx_params",
                    &ContextParams {
                        total_steps: steps as u32,
                        max_steps: steps as u32,
                        head_dim: head_dim as u32,
                        head_offset: head as u32,
                        score_offset: (head * steps * steps) as u32,
                        heads: heads as u32,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    },
                );
                let bind_group = self.bind_group(
                    "attn_context",
                    &pipeline.layout,
                    &[
                        (0, params.as_entire_binding()),
                        (1, buffer_scores.as_entire_binding()),
                        (2, buffer_qkv.as_entire_binding()),
                        (3, buffer_out.as_entire_binding()),
                    ],
                );
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(steps as u32, 1, 1);
            }
        }
        self.submit_and_read(encoder, &buffer_out, count)
    }
}

