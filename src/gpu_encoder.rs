//! Encoder block execution against preallocated GPU scratch.
//!
//! The per-operator entry points in `gpu` and `gpu_attn` upload, dispatch and read
//! back one kernel at a time. That is fine for checking numerics but it measures
//! PCIe round trips, not the GPU, so it cannot answer whether the port is fast.
//! Everything here records a whole sub-block into a single `CommandEncoder` and
//! reads back only the tensor the caller asked for.

use anyhow::Result;
use wgpu::util::DeviceExt as _;

use crate::conformer::{
    Attention, AttentionModel, CONV_KERNEL, Convolution, D_MODEL, FFN, FeedForward, HEAD_DIM,
    HEADS, LAYER_NORM_EPS,
};
use crate::gpu::{CachedPipeline, GpuGemm};

/// `Dims` uniform, mirrors `gemm.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

/// `Params` uniform, mirrors `layer_norm.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerNormParams {
    steps: u32,
    width: u32,
    eps: f32,
    _pad: f32,
}

/// `GluParams` uniform, mirrors `glu.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GluParams {
    steps: u32,
    width: u32,
    _pad0: u32,
    _pad1: u32,
}

/// `ConvParams` uniform, mirrors `dw_conv_bn_silu.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvParams {
    steps: u32,
    channels: u32,
    kernel: u32,
    padding: u32,
}

/// `AttnParams` uniform, mirrors `attn_scores_softmax.wgsl`.
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

/// `ContextParams` uniform, mirrors `attn_context.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ContextParams {
    total_steps: u32,
    max_steps: u32,
    head_dim: u32,
    /// Which head this dispatch handles.
    head_offset: u32,
    /// Where this head's score window starts, in floats.
    score_offset: u32,
    heads: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Memory budget for the three chunked attention score/weight buffers combined:
/// `attn_s_ctx [R][c_pad]` + `attn_s_pos [R][p_pad]` + `attn_weights [R][k_pad]`.
///
/// 256 MiB keeps peak score memory constant on 25–45 minute utterances while still
/// supplying hundreds of GEMM workgroup waves per chunk.
const ATTN_SCORE_BUDGET: usize = 256 << 20;

/// Longest utterance allowed to record all 24 layers into a single command buffer.
/// Past this, work is submitted per layer (or per query chunk) to stay well below
/// Windows' 2 s TDR watchdog.
const SINGLE_SUBMIT_MAX_STEPS: usize = 512;

/// Maximum sequence length handled by `attn_softmax_smem.wgsl` (16 KiB workgroup tile).
/// Past this (~5.5 min), `attn_softmax.wgsl` runs in-place in `weights` with 1 KiB smem.
const ATTN_SOFTMAX_SMEM_MAX: usize = 4096;

/// Workgroup storage `attn_softmax_smem.wgsl` declares: `row: array<f32, 4096>` plus
/// `reduce: array<f32, 256>`, i.e. 16 KiB + 1 KiB. An adapter whose
/// `max_compute_workgroup_storage_size` is below this rejects the pipeline at creation,
/// so the stage is skipped there and `attn_softmax.wgsl` handles every utterance.
const ATTN_SOFTMAX_SMEM_BYTES: u32 = 17_408;

/// Padded widths the GEMM attention needs for a given frame count.
///
/// The score GEMMs write `vec4`s along `n`, so `n` must divide by four; the context
/// GEMM consumes `k` in blocks of sixteen. Deriving these from the frame count in one
/// place keeps the uniforms and the dispatch counts from disagreeing.
fn attn_pads(steps: usize) -> (usize, usize, usize) {
    let c_pad = steps.div_ceil(4) * 4;
    let p_pad = (2 * steps - 1).div_ceil(4) * 4;
    let k_pad = steps.div_ceil(16) * 16;
    (c_pad, p_pad, k_pad)
}

/// Chooses the query-row chunk size `R` for `steps` frames under `binding_limit`.
///
/// Short utterances keep `R = steps` (one chunk, zero overhead). Long utterances
/// cap the three score/weight buffers to [`ATTN_SCORE_BUDGET`] (and each individual
/// binding to `binding_limit`), rounded down to GEMM's `BM = 64` tile height.
fn attn_chunk_rows(steps: usize, binding_limit: usize) -> Result<usize> {
    if steps == 0 {
        anyhow::bail!("encoder steps must be positive");
    }
    let (c_pad, p_pad, k_pad) = attn_pads(steps);
    let f32_bytes = std::mem::size_of::<f32>();
    // `attn_s_pos` is the widest of the three buffers (`p_pad` ≈ 2 * steps).
    let max_by_binding = binding_limit / (p_pad * f32_bytes);
    if max_by_binding == 0 {
        anyhow::bail!(
            "{steps} frames need at least {} B per attention row; adapter allows {binding_limit} B per binding",
            p_pad * f32_bytes
        );
    }
    let row_bytes = (c_pad + p_pad + k_pad) * f32_bytes;
    let max_by_budget = (ATTN_SCORE_BUDGET / row_bytes).max(1);
    let raw = steps.min(max_by_binding).min(max_by_budget);
    if raw >= steps {
        return Ok(steps);
    }
    let aligned = if raw >= 128 {
        (raw / 128) * 128
    } else if raw >= 64 {
        (raw / 64) * 64
    } else {
        raw
    };
    Ok(aligned.max(1))
}

/// `attn_prepare` uniform for one head and query chunk.
fn prepare_params_for(
    steps: usize,
    head: usize,
    row_offset: usize,
    chunk_rows: usize,
    fill_kv: bool,
) -> PrepareParams {
    let (c_pad, p_pad, k_pad) = attn_pads(steps);
    PrepareParams {
        steps: steps as u32,
        head: head as u32,
        d_model: D_MODEL as u32,
        head_dim: HEAD_DIM as u32,
        pos_len: (2 * steps - 1) as u32,
        k_pad: k_pad as u32,
        c_pad: c_pad as u32,
        p_pad: p_pad as u32,
        row_offset: row_offset as u32,
        chunk_rows: chunk_rows as u32,
        fill_kv: u32::from(fill_kv),
        _pad: 0,
    }
}

/// `attn_softmax` params for a given frame count and query-chunk offset.
fn softmax_params_for(steps: usize, scale: f32, row_offset: usize) -> SoftmaxParams {
    let (c_pad, p_pad, k_pad) = attn_pads(steps);
    SoftmaxParams {
        steps: steps as u32,
        pos_len: (2 * steps - 1) as u32,
        content_n: c_pad as u32,
        pos_n: p_pad as u32,
        scale,
        k_pad: k_pad as u32,
        row_offset: row_offset as u32,
        _pad1: 0,
    }
}

/// `attn_scatter` params for one head and query chunk.
fn scatter_params_for(head: usize, row_offset: usize, chunk_rows: usize) -> ScatterParams {
    ScatterParams {
        chunk_rows: chunk_rows as u32,
        d_model: D_MODEL as u32,
        head_dim: HEAD_DIM as u32,
        head: head as u32,
        row_offset: row_offset as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    }
}

/// `Params` uniform, mirrors `attn_prepare.wgsl`. 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PrepareParams {
    steps: u32,
    head: u32,
    d_model: u32,
    head_dim: u32,
    pos_len: u32,
    k_pad: u32,
    c_pad: u32,
    p_pad: u32,
    row_offset: u32,
    chunk_rows: u32,
    fill_kv: u32,
    _pad: u32,
}

/// `Params` uniform, mirrors `attn_softmax.wgsl`. 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxParams {
    steps: u32,
    pos_len: u32,
    content_n: u32,
    pos_n: u32,
    scale: f32,
    k_pad: u32,
    row_offset: u32,
    _pad1: u32,
}

/// `Params` uniform, mirrors `attn_scatter.wgsl`. 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScatterParams {
    chunk_rows: u32,
    d_model: u32,
    head_dim: u32,
    head: u32,
    row_offset: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// `AxpyParams` uniform, mirrors `axpy.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AxpyParams {
    count: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
}

/// `LocalAttnParams` uniform, mirrors `attn_local.wgsl`. 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LocalAttnParams {
    steps: u32,
    d_model: u32,
    head_dim: u32,
    pos_len: u32,
    left_ctx: u32,
    right_ctx: u32,
    window_w: u32,
    heads: u32,
    scale: f32,
    grid_x: u32,
    _pad0: u32,
    _pad1: u32,
}

const AXPY_SHADER: &str = include_str!("axpy.wgsl");



fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A storage buffer sized for `count` floats, usable as copy source and destination.
fn scratch(device: &wgpu::Device, label: &str, count: usize) -> wgpu::Buffer {
    let size = ((count * 4).div_ceil(16) * 16) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// GPU scratch for every Conformer sub-block at a fixed capacity `steps`.
pub struct EncoderScratchGpu {
    gpu: std::sync::Arc<GpuGemm>,
    steps: usize,
    pos_len: usize,
    /// Query rows processed per attention chunk (`<= steps`).
    attn_chunk_rows: usize,
    /// True when allocated exclusively for `rel_pos_local_attn` (0 MiB global score buffers).
    local_only: bool,
    residual: wgpu::Buffer,
    normed: wgpu::Buffer,
    hidden: wgpu::Buffer,
    block: wgpu::Buffer,
    temp: wgpu::Buffer,
    gates: wgpu::Buffer,
    qkv: wgpu::Buffer,
    pos: wgpu::Buffer,
    /// `[pos_len][D_MODEL]` position table, shared by every layer.
    pos_emb: wgpu::Buffer,
    /// `[attn_chunk_rows][HEAD_DIM]`: `q + pos_bias_u` / `q + pos_bias_v` for the chunk in flight.
    attn_qu: wgpu::Buffer,
    attn_qv: wgpu::Buffer,
    /// `[c_pad][HEAD_DIM]`: K, zero-padded so `n` is a multiple of four.
    attn_km: wgpu::Buffer,
    /// `[p_pad][HEAD_DIM]`: the position table for this head, zero-padded.
    attn_pm: wgpu::Buffer,
    /// `[HEAD_DIM][k_pad]`: V transposed, zero-padded to a multiple of sixteen.
    attn_vt: wgpu::Buffer,
    /// `[attn_chunk_rows][c_pad]` and `[attn_chunk_rows][p_pad]`: the two score blocks.
    attn_s_ctx: wgpu::Buffer,
    attn_s_pos: wgpu::Buffer,
    /// `[attn_chunk_rows][k_pad]`: softmaxed weights.
    attn_weights: wgpu::Buffer,
    /// `[attn_chunk_rows][HEAD_DIM]`: one head's context for the chunk in flight.
    attn_out: wgpu::Buffer,
    axpy: wgpu::ComputePipeline,
    axpy_layout: wgpu::BindGroupLayout,
    /// GEMM-based attention: gather, two score GEMMs, softmax, context GEMM, scatter.
    attn_prepare: CachedPipeline,
    attn_softmax: CachedPipeline,
    /// Shared-memory fast path for `steps <= ATTN_SOFTMAX_SMEM_MAX` (shares
    /// `attn_softmax.layout`). `None` on adapters whose workgroup storage cannot hold
    /// [`ATTN_SOFTMAX_SMEM_BYTES`]; the global-memory kernel then runs instead.
    attn_softmax_smem: Option<wgpu::ComputePipeline>,
    attn_scatter: CachedPipeline,
    /// Fused Flash-style sliding-window relative-position local attention (`rel_pos_local_attn`).
    attn_local: CachedPipeline,
}

/// Builds one compute pipeline from source with the given bind group layout.
fn simple_pipeline(
    gpu: &GpuGemm,
    label: &'static str,
    source: &'static str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> crate::gpu::CachedPipeline {
    let device = gpu.dev();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: gpu.pipeline_cache(),
    });
    crate::gpu::CachedPipeline { pipeline, layout }
}

impl EncoderScratchGpu {
    /// Allocates scratch for `steps` frames at the model's fixed widths.
    pub fn new(steps: usize) -> Result<Self> {
        Self::with_gpu(std::sync::Arc::new(GpuGemm::new()?), steps)
    }

    /// Same, on an existing device so `pre_encode` and the encoder share one queue.
    pub fn with_gpu(gpu: std::sync::Arc<GpuGemm>, steps: usize) -> Result<Self> {
        Self::with_capacity(gpu, steps)
    }

    /// Scratch for up to `steps` frames — one utterance, the only case supported.
    pub fn with_capacity(gpu: std::sync::Arc<GpuGemm>, steps: usize) -> Result<Self> {
        let binding_limit = gpu.dev().limits().max_storage_buffer_binding_size as usize;
        let chunk_rows = attn_chunk_rows(steps, binding_limit)?;
        Self::with_capacity_and_chunk(gpu, steps, chunk_rows)
    }

    /// Scratch sized for `steps` frames under `model`: in `RelPosLocalAttn` mode, skips
    /// allocating the 256 MiB global attention score buffers completely (0 MiB score VRAM).
    pub fn with_capacity_for_model(
        gpu: std::sync::Arc<GpuGemm>,
        steps: usize,
        model: AttentionModel,
    ) -> Result<Self> {
        model.validate()?;
        match model {
            AttentionModel::RelPos => Self::with_capacity(gpu, steps),
            AttentionModel::RelPosLocalAttn { left, right } => {
                Self::build_scratch(gpu, steps, 1, Some(left + right + 1))
            }
        }
    }

    /// Same, with an explicit query-row chunk size (for verifying chunked vs unchunked).
    pub fn with_capacity_and_chunk(
        gpu: std::sync::Arc<GpuGemm>,
        steps: usize,
        chunk_rows: usize,
    ) -> Result<Self> {
        Self::build_scratch(gpu, steps, chunk_rows, None)
    }

    fn build_scratch(
        gpu: std::sync::Arc<GpuGemm>,
        steps: usize,
        chunk_rows: usize,
        local_pos_len: Option<usize>,
    ) -> Result<Self> {
        if steps == 0 {
            anyhow::bail!("encoder steps must be positive");
        }
        let local_only = local_pos_len.is_some();
        let attn_chunk_rows = chunk_rows.clamp(1, steps);
        let pos_len = local_pos_len.unwrap_or(2 * steps - 1);
        let binding_limit = gpu.dev().limits().max_storage_buffer_binding_size as usize;
        let hidden_bytes = steps * FFN * std::mem::size_of::<f32>();
        if hidden_bytes > binding_limit {
            anyhow::bail!(
                "{steps} frames need a {hidden_bytes}-byte FFN scratch buffer; this adapter allows {binding_limit} B per binding"
            );
        }
        let (c_pad, p_pad, k_pad) = attn_pads(steps);
        if !local_only {
            let pos_score_bytes = attn_chunk_rows * p_pad * std::mem::size_of::<f32>();
            if pos_score_bytes > binding_limit {
                anyhow::bail!(
                    "{steps} frames with {attn_chunk_rows}-row chunks need a {pos_score_bytes}-byte position score buffer; adapter allows {binding_limit} B per binding"
                );
            }
        }

        let (axpy_cached, attn_prepare, attn_softmax, attn_scatter, attn_local) =
            std::thread::scope(|s| {
                let t0 = s.spawn(|| {
                    simple_pipeline(
                        &gpu,
                        "axpy",
                        AXPY_SHADER,
                        &[
                            uniform_entry(0),
                            storage_entry(1, false),
                            storage_entry(2, true),
                        ],
                    )
                });
                let t1 = s.spawn(|| {
                    simple_pipeline(
                        &gpu,
                        "attn_prepare",
                        include_str!("attn_prepare.wgsl"),
                        &[
                            uniform_entry(0),
                            storage_entry(1, true),
                            storage_entry(2, true),
                            storage_entry(3, true),
                            storage_entry(4, false),
                            storage_entry(5, false),
                            storage_entry(6, false),
                            storage_entry(7, false),
                            storage_entry(8, false),
                        ],
                    )
                });
                let t2 = s.spawn(|| {
                    simple_pipeline(
                        &gpu,
                        "attn_softmax",
                        include_str!("attn_softmax.wgsl"),
                        &[
                            uniform_entry(0),
                            storage_entry(1, true),
                            storage_entry(2, true),
                            storage_entry(3, false),
                        ],
                    )
                });
                let t3 = s.spawn(|| {
                    simple_pipeline(
                        &gpu,
                        "attn_scatter",
                        include_str!("attn_scatter.wgsl"),
                        &[
                            uniform_entry(0),
                            storage_entry(1, true),
                            storage_entry(2, false),
                        ],
                    )
                });
                let t4 = s.spawn(|| {
                    simple_pipeline(
                        &gpu,
                        "attn_local",
                        include_str!("attn_local.wgsl"),
                        &[
                            uniform_entry(0),
                            storage_entry(1, true),
                            storage_entry(2, true),
                            storage_entry(3, true),
                            storage_entry(4, false),
                        ],
                    )
                });
                (
                    t0.join().expect("axpy"),
                    t1.join().expect("attn_prepare"),
                    t2.join().expect("attn_softmax"),
                    t3.join().expect("attn_scatter"),
                    t4.join().expect("attn_local"),
                )
            });
        let axpy = axpy_cached.pipeline;
        let axpy_layout = axpy_cached.layout;
        let attn_softmax_smem = if gpu.dev().limits().max_compute_workgroup_storage_size
            >= ATTN_SOFTMAX_SMEM_BYTES
        {
            let device = gpu.dev();
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("attn_softmax_smem"),
                source: wgpu::ShaderSource::Wgsl(include_str!("attn_softmax_smem.wgsl").into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("attn_softmax_smem"),
                bind_group_layouts: &[Some(&attn_softmax.layout)],
                immediate_size: 0,
            });
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("attn_softmax_smem"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: gpu.pipeline_cache(),
            }))
        } else {
            None
        };

        // Allocated before the struct literal: field initialisers run in order, so
        // `gpu` would already be moved by the time the later fields borrowed it.
        let residual = scratch(gpu.dev(), "residual", steps * D_MODEL);
        let normed = scratch(gpu.dev(), "normed", steps * D_MODEL);
        let hidden = scratch(gpu.dev(), "hidden", steps * FFN);
        let block = scratch(gpu.dev(), "block", steps * D_MODEL);
        let temp = scratch(gpu.dev(), "temp", steps * D_MODEL);
        let gates = scratch(gpu.dev(), "gates", steps * 2 * D_MODEL);
        let qkv = scratch(gpu.dev(), "qkv", steps * 3 * D_MODEL);
        let pos = scratch(gpu.dev(), "pos", pos_len * D_MODEL);
        // In `local_only` mode the global attention buffers are never dispatched, so keep
        // minimal 4-float placeholders to save the full 256+ MiB score budget.
        let (attn_qu, attn_qv, attn_km, attn_pm, attn_vt, attn_s_ctx, attn_s_pos, attn_weights, attn_out) =
            if local_only {
                (
                    scratch(gpu.dev(), "attn_qu_dummy", 4),
                    scratch(gpu.dev(), "attn_qv_dummy", 4),
                    scratch(gpu.dev(), "attn_km_dummy", 4),
                    scratch(gpu.dev(), "attn_pm_dummy", 4),
                    scratch(gpu.dev(), "attn_vt_dummy", 4),
                    scratch(gpu.dev(), "attn_s_ctx_dummy", 4),
                    scratch(gpu.dev(), "attn_s_pos_dummy", 4),
                    scratch(gpu.dev(), "attn_weights_dummy", 4),
                    scratch(gpu.dev(), "attn_out_dummy", 4),
                )
            } else {
                (
                    scratch(gpu.dev(), "attn_qu", attn_chunk_rows * HEAD_DIM),
                    scratch(gpu.dev(), "attn_qv", attn_chunk_rows * HEAD_DIM),
                    scratch(gpu.dev(), "attn_km", c_pad * HEAD_DIM),
                    scratch(gpu.dev(), "attn_pm", p_pad * HEAD_DIM),
                    scratch(gpu.dev(), "attn_vt", HEAD_DIM * k_pad),
                    scratch(gpu.dev(), "attn_s_ctx", attn_chunk_rows * c_pad),
                    scratch(gpu.dev(), "attn_s_pos", attn_chunk_rows * p_pad),
                    scratch(gpu.dev(), "attn_weights", attn_chunk_rows * k_pad),
                    scratch(gpu.dev(), "attn_out", attn_chunk_rows * HEAD_DIM),
                )
            };
        let pos_emb = scratch(gpu.dev(), "pos_emb", pos_len * D_MODEL);

        Ok(Self {
            gpu,
            steps,
            pos_len,
            attn_chunk_rows,
            local_only,
            residual,
            normed,
            hidden,
            block,
            temp,
            gates,
            qkv,
            pos,
            pos_emb,
            attn_qu,
            attn_qv,
            attn_km,
            attn_pm,
            attn_vt,
            attn_s_ctx,
            attn_s_pos,
            attn_weights,
            attn_out,
            axpy,
            axpy_layout,
            attn_prepare,
            attn_softmax,
            attn_softmax_smem,
            attn_scatter,
            attn_local,
        })
    }

    pub fn adapter(&self) -> &str {
        self.gpu.adapter()
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Seeds the residual stream, e.g. with the pre-encoder output.
    pub fn write_residual(&self, values: &[f32]) {
        self.gpu.queue_ref().write_buffer(
            &self.residual,
            0,
            bytemuck::cast_slice(&values[..self.steps * D_MODEL]),
        );
    }

    /// The live residual stream.
    pub fn residual(&self) -> Result<Vec<f32>> {
        self.gpu.read_buffer(&self.residual, self.steps * D_MODEL)
    }

    /// `ln -> linear1 -> silu -> linear2`, one submission.
    ///
    /// This model's feed-forward has no GLU — the GLU lives in the convolution
    /// module — so the activation is a plain SiLU over the `ffn` width.
    pub fn feed_forward(
        &self,
        feed_forward: &FeedForward,
        norm_weight: &[f32],
        norm_bias: &[f32],
    ) -> Result<Vec<f32>> {
        let buffer_w1 = self
            .gpu
            .upload_plain("ff_w1", feed_forward.linear1().weight_row_major());
        let buffer_w2 = self
            .gpu
            .upload_plain("ff_w2", feed_forward.linear2().weight_row_major());
        let buffer_ln_w = self.gpu.upload_plain("ff_ln_w", norm_weight);
        let buffer_ln_b = self.gpu.upload_plain("ff_ln_b", norm_bias);

        let mut encoder = self.gpu.encoder("feed_forward");
        self.record_layer_norm(
            &mut encoder,
            &self.residual,
            &buffer_ln_w,
            &buffer_ln_b,
            &self.normed,
        );
        self.record_gemm(
            &mut encoder,
            &self.normed,
            &buffer_w1,
            &self.hidden,
            self.steps,
            FFN,
            D_MODEL,
        );
        self.record_silu(&mut encoder, &self.hidden, self.steps * FFN);
        // `hidden` is `[steps][FFN]`, so this reduction runs over FFN — the width the
        // first GEMM produced, not the model width.
        self.record_gemm(
            &mut encoder,
            &self.hidden,
            &buffer_w2,
            &self.block,
            self.steps,
            D_MODEL,
            FFN,
        );
        self.gpu.queue_ref().submit(Some(encoder.finish()));

        self.gpu.read_buffer(&self.block, self.steps * D_MODEL)
    }

    /// `ln -> qkv -> pos -> scores/softmax -> context -> out`, one submission.
    ///
    /// Verification-only single-block path: builds the per-query score pipeline and
    /// quadratic score buffer on demand so the 24-layer production encoder never pays
    /// for either.
    pub fn attention(
        &self,
        attention: &Attention,
        norm_weight: &[f32],
        norm_bias: &[f32],
        pos_emb: &[f32],
    ) -> Result<Vec<f32>> {
        let scores_bytes = self.steps * self.steps * std::mem::size_of::<f32>();
        let binding_limit = self.gpu.dev().limits().max_storage_buffer_binding_size as usize;
        if scores_bytes > binding_limit {
            anyhow::bail!(
                "{} frames need a {scores_bytes}-byte score matrix for single-block attention; adapter allows {binding_limit} B per binding",
                self.steps
            );
        }
        let attn_scores = self.gpu.attn_scores_pipeline(self.steps)?;
        let attn_context = self.gpu.attn_context_pipeline()?;
        let scores = scratch(self.gpu.dev(), "scores", self.steps * self.steps);

        let buffer_ln_w = self.gpu.upload_plain("attn_ln_w", norm_weight);
        let buffer_ln_b = self.gpu.upload_plain("attn_ln_b", norm_bias);
        let buffer_qkv_w = self
            .gpu
            .upload_plain("attn_qkv_w", attention.qkv_weight().row_major());
        let buffer_pos_w = self
            .gpu
            .upload_plain("attn_pos_w", attention.pos_weight().row_major());
        let buffer_out_w = self
            .gpu
            .upload_plain("attn_out_w", attention.out_weight().row_major());
        let buffer_pos_emb = self
            .gpu
            .upload_plain("attn_pos_emb", &pos_emb[..self.pos_len * D_MODEL]);
        let mut bias = attention.pos_bias_u().to_vec();
        bias.extend_from_slice(attention.pos_bias_v());
        let buffer_bias = self.gpu.upload_plain("attn_bias", &bias);

        let mut encoder = self.gpu.encoder("attention");
        self.record_layer_norm(
            &mut encoder,
            &self.residual,
            &buffer_ln_w,
            &buffer_ln_b,
            &self.normed,
        );
        self.record_gemm(
            &mut encoder,
            &self.normed,
            &buffer_qkv_w,
            &self.qkv,
            self.steps,
            3 * D_MODEL,
            D_MODEL,
        );
        self.record_gemm(
            &mut encoder,
            &buffer_pos_emb,
            &buffer_pos_w,
            &self.pos,
            self.pos_len,
            D_MODEL,
            D_MODEL,
        );
        self.record_attn_scores(
            &mut encoder,
            &attn_scores,
            &scores,
            &buffer_bias,
            attention.scale(),
        );
        self.record_attn_context(&mut encoder, &attn_context, &scores);
        // Context goes to `temp` and the out projection to `block`: `accumulate`
        // always folds `block` into the residual, so the block's *final* output has
        // to land there. Routing the projection to `temp` would leave the bare
        // context in the residual stream.
        self.record_gemm(
            &mut encoder,
            &self.temp,
            &buffer_out_w,
            &self.block,
            self.steps,
            D_MODEL,
            D_MODEL,
        );
        self.gpu.queue_ref().submit(Some(encoder.finish()));

        self.gpu.read_buffer(&self.block, self.steps * D_MODEL)
    }

    /// `ln -> pointwise1 -> glu -> dw+bn+silu -> pointwise2`, one submission.
    pub fn convolution(
        &self,
        convolution: &Convolution,
        norm_weight: &[f32],
        norm_bias: &[f32],
    ) -> Result<Vec<f32>> {
        let buffer_ln_w = self.gpu.upload_plain("conv_ln_w", norm_weight);
        let buffer_ln_b = self.gpu.upload_plain("conv_ln_b", norm_bias);
        let buffer_pw1 = self
            .gpu
            .upload_plain("conv_pw1", convolution.pointwise1().weight_row_major());
        let buffer_pw2 = self
            .gpu
            .upload_plain("conv_pw2", convolution.pointwise2().weight_row_major());
        let buffer_dw = self
            .gpu
            .upload_plain("conv_dw", convolution.depthwise_weights());
        let (scale, shift) = convolution.folded_batch_norm();
        let buffer_scale = self.gpu.upload_plain("conv_scale", &scale);
        let buffer_shift = self.gpu.upload_plain("conv_shift", &shift);

        let mut encoder = self.gpu.encoder("convolution");
        self.record_layer_norm(
            &mut encoder,
            &self.residual,
            &buffer_ln_w,
            &buffer_ln_b,
            &self.normed,
        );
        self.record_gemm(
            &mut encoder,
            &self.normed,
            &buffer_pw1,
            &self.gates,
            self.steps,
            2 * D_MODEL,
            D_MODEL,
        );
        self.record_glu(&mut encoder, &self.gates, &self.normed);
        self.record_dw_conv(
            &mut encoder,
            &self.normed,
            &buffer_dw,
            &buffer_scale,
            &buffer_shift,
            &self.temp,
        );
        self.record_gemm(
            &mut encoder,
            &self.temp,
            &buffer_pw2,
            &self.block,
            self.steps,
            D_MODEL,
            D_MODEL,
        );
        self.gpu.queue_ref().submit(Some(encoder.finish()));

        self.gpu.read_buffer(&self.block, self.steps * D_MODEL)
    }

    /// The trailing `norm_out` applied to the residual stream.
    pub fn final_norm(&self, norm_weight: &[f32], norm_bias: &[f32]) -> Result<Vec<f32>> {
        let buffer_ln_w = self.gpu.upload_plain("out_ln_w", norm_weight);
        let buffer_ln_b = self.gpu.upload_plain("out_ln_b", norm_bias);
        let mut encoder = self.gpu.encoder("final_norm");
        self.record_layer_norm(
            &mut encoder,
            &self.residual,
            &buffer_ln_w,
            &buffer_ln_b,
            &self.block,
        );
        self.gpu.queue_ref().submit(Some(encoder.finish()));
        self.gpu.read_buffer(&self.block, self.steps * D_MODEL)
    }

    /// `residual += scale * block`, i.e. folds the sub-block output back in.
    pub fn accumulate(&self, scale: f32) -> Result<()> {
        let mut encoder = self.gpu.encoder("accumulate");
        self.record_axpy(&mut encoder, &self.residual, &self.block, scale);
        self.gpu.queue_ref().submit(Some(encoder.finish()));
        Ok(())
    }

    fn uniform<T: bytemuck::Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        self.gpu
            .dev()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(value),
                // `COPY_DST` so the frame count can be rewritten per utterance without
                // rebuilding the bind groups.
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn record_layer_norm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) {
        let params = self.uniform(
            "ln_params",
            &LayerNormParams {
                steps: self.steps as u32,
                width: D_MODEL as u32,
                eps: LAYER_NORM_EPS,
                _pad: 0.0,
            },
        );
        let (pipeline, layout) = self.gpu.layer_norm_pipeline();
        let bind_group = self.gpu.bind_group(
            "ln",
            layout,
            &[
                (0, params.as_entire_binding()),
                (1, x.as_entire_binding()),
                (2, weight.as_entire_binding()),
                (3, bias.as_entire_binding()),
                (4, out.as_entire_binding()),
            ],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ln"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // This kernel indexes by `workgroup_id`, one workgroup per step.
        pass.dispatch_workgroups(self.steps as u32, 1, 1);
    }

    /// The argument list mirrors the kernel's bindings one-to-one, same as
    /// `GpuGemm::dw_conv_bn_silu`; a struct would only add a name.
    #[allow(clippy::too_many_arguments)]
    fn record_gemm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        let dims = self.uniform(
            "gemm_dims",
            &Dims {
                m: m as u32,
                n: n as u32,
                k: k as u32,
                _pad: 0,
            },
        );
        let (_, layout) = self.gpu.gemm_pipeline();
        let (pipeline, gx, gy) = self.gpu.gemm_dispatch(m, n, k);
        let bind_group = self.gpu.bind_group(
            "gemm",
            layout,
            &[
                (0, dims.as_entire_binding()),
                (1, a.as_entire_binding()),
                (2, w.as_entire_binding()),
                (3, c.as_entire_binding()),
            ],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    fn record_silu(&self, encoder: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, count: usize) {
        let (pipeline, layout) = self.gpu.silu_pipeline();
        let bind_group = self
            .gpu
            .bind_group("silu", layout, &[(0, x.as_entire_binding())]);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("silu"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // `global_invocation_id` indexing: cover every element.
        let (gx, gy) = crate::gpu::grid_2d(count.div_ceil(256));
        pass.dispatch_workgroups(gx, gy, 1);
    }

    fn record_glu(&self, encoder: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, out: &wgpu::Buffer) {
        let params = self.uniform(
            "glu_params",
            &GluParams {
                steps: self.steps as u32,
                width: D_MODEL as u32,
                _pad0: 0,
                _pad1: 0,
            },
        );
        let (pipeline, layout) = self.gpu.glu_pipeline();
        let bind_group = self.gpu.bind_group(
            "glu",
            layout,
            &[
                (0, params.as_entire_binding()),
                (1, x.as_entire_binding()),
                (2, out.as_entire_binding()),
            ],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("glu"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (gx, gy) = crate::gpu::grid_2d((self.steps * D_MODEL).div_ceil(256));
        pass.dispatch_workgroups(gx, gy, 1);
    }

    fn record_dw_conv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        weights: &wgpu::Buffer,
        scale: &wgpu::Buffer,
        shift: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) {
        let params = self.uniform(
            "dw_params",
            &ConvParams {
                steps: self.steps as u32,
                channels: D_MODEL as u32,
                kernel: CONV_KERNEL as u32,
                padding: (CONV_KERNEL / 2) as u32,
            },
        );
        let (pipeline, layout) = self.gpu.dw_conv_pipeline();
        let bind_group = self.gpu.bind_group(
            "dw_conv",
            layout,
            &[
                (0, params.as_entire_binding()),
                (1, x.as_entire_binding()),
                (2, weights.as_entire_binding()),
                (3, scale.as_entire_binding()),
                (4, shift.as_entire_binding()),
                (5, out.as_entire_binding()),
            ],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dw_conv"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (gx, gy) = crate::gpu::grid_2d((self.steps * D_MODEL).div_ceil(256));
        pass.dispatch_workgroups(gx, gy, 1);
    }

    fn record_attn_scores(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        attn_scores: &CachedPipeline,
        scores: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        scale: f32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("attn_scores"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&attn_scores.pipeline);
        for head in 0..HEADS {
            let params = self.uniform(
                "attn_params",
                &AttnParams {
                    total_steps: self.steps as u32,
                    max_steps: self.steps as u32,
                    width: (3 * D_MODEL) as u32,
                    head_dim: HEAD_DIM as u32,
                    scale,
                    heads: HEADS as u32,
                    pos_width: D_MODEL as u32,
                    head_offset: head as u32,
                    score_offset: 0,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                },
            );
            let bind_group = self.gpu.bind_group(
                "attn_scores",
                &attn_scores.layout,
                &[
                    (0, params.as_entire_binding()),
                    (1, self.qkv.as_entire_binding()),
                    (2, self.pos.as_entire_binding()),
                    (3, bias.as_entire_binding()),
                    (4, scores.as_entire_binding()),
                ],
            );
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.steps as u32, 1, 1);
        }
    }

    fn record_attn_context(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        attn_context: &CachedPipeline,
        scores: &wgpu::Buffer,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("attn_context"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&attn_context.pipeline);
        for head in 0..HEADS {
            let params = self.uniform(
                "ctx_params",
                &ContextParams {
                    total_steps: self.steps as u32,
                    max_steps: self.steps as u32,
                    head_dim: HEAD_DIM as u32,
                    head_offset: head as u32,
                    score_offset: 0,
                    heads: HEADS as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                },
            );
            let bind_group = self.gpu.bind_group(
                "attn_context",
                &attn_context.layout,
                &[
                    (0, params.as_entire_binding()),
                    (1, scores.as_entire_binding()),
                    (2, self.qkv.as_entire_binding()),
                    (3, self.temp.as_entire_binding()),
                ],
            );
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.steps as u32, 1, 1);
        }
    }

    fn record_axpy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        scale: f32,
    ) {
        let count = self.steps * D_MODEL;
        let params = self.uniform(
            "axpy_params",
            &AxpyParams {
                count: count as u32,
                scale,
                _pad0: 0,
                _pad1: 0,
            },
        );
        let bind_group = self.gpu.bind_group(
            "axpy",
            &self.axpy_layout,
            &[
                (0, params.as_entire_binding()),
                (1, x.as_entire_binding()),
                (2, y.as_entire_binding()),
            ],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("axpy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.axpy);
        pass.set_bind_group(0, &bind_group, &[]);
        let (gx, gy) = crate::gpu::grid_2d(count.div_ceil(256));
        pass.dispatch_workgroups(gx, gy, 1);
    }
}

/// Uniforms shared by all 24 layers for a fixed capacity.
struct SharedUniforms {
    ln_params: wgpu::Buffer,
    dw_params: wgpu::Buffer,
    glu_params: wgpu::Buffer,
    axpy_half_params: wgpu::Buffer,
    axpy_one_params: wgpu::Buffer,
    attn_prepare_params: [wgpu::Buffer; HEADS],
    attn_scatter_params: [wgpu::Buffer; HEADS],
    /// Attention GEMM dimensions: content scores, position scores, context.
    dims_attn_content: wgpu::Buffer,
    dims_attn_position: wgpu::Buffer,
    dims_attn_context: wgpu::Buffer,
    attn_softmax_params: wgpu::Buffer,
    local_attn_params: wgpu::Buffer,
    dims_ff_w1: wgpu::Buffer,
    dims_ff_w2: wgpu::Buffer,
    dims_qkv: wgpu::Buffer,
    dims_pos: wgpu::Buffer,
    dims_pw1: wgpu::Buffer,
    dims_d_d: wgpu::Buffer,
}

/// Bind groups whose operands are pure scratch (reused across all 24 layers).
struct SharedBindings {
    silu: wgpu::BindGroup,
    glu: wgpu::BindGroup,
    axpy_half: wgpu::BindGroup,
    axpy_one: wgpu::BindGroup,
}

/// Layer-independent bind groups for the GEMM attention chain.
struct AttnGemm {
    gemm_content: wgpu::BindGroup,
    gemm_position: wgpu::BindGroup,
    softmax: wgpu::BindGroup,
    gemm_context: wgpu::BindGroup,
    scatter: [wgpu::BindGroup; HEADS],
}

/// Pre-built bind groups for one Conformer layer (all 16 weight-bearing dispatches).
///
/// A wgpu `BindGroup` retains its bound buffers, so the hot path performs zero buffer
/// allocations, zero `write_buffer` calls and zero bind-group creations.
struct LayerBindings {
    ln_ff1: wgpu::BindGroup,
    gemm_ff1_w1: wgpu::BindGroup,
    gemm_ff1_w2: wgpu::BindGroup,
    ln_attn: wgpu::BindGroup,
    gemm_qkv: wgpu::BindGroup,
    gemm_pos: wgpu::BindGroup,
    /// Per-layer: this layer's `pos_bias_u` / `pos_bias_v` are folded in here.
    attn_prepare: [wgpu::BindGroup; HEADS],
    /// Per-layer fused local attention (`rel_pos_local_attn`).
    attn_local: wgpu::BindGroup,
    gemm_attn_out: wgpu::BindGroup,
    ln_conv: wgpu::BindGroup,
    gemm_conv_pw1: wgpu::BindGroup,
    dw_conv: wgpu::BindGroup,
    gemm_conv_pw2: wgpu::BindGroup,
    ln_ff2: wgpu::BindGroup,
    gemm_ff2_w1: wgpu::BindGroup,
    gemm_ff2_w2: wgpu::BindGroup,
    ln_out: wgpu::BindGroup,
}

/// Which sub-blocks of a layer to record.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    All,
    FeedForward1,
    Attention,
    Convolution,
    FeedForward2,
    NormOut,
}

impl Stage {
    const ALL_BLOCKS: [Stage; 5] = [
        Stage::FeedForward1,
        Stage::Attention,
        Stage::Convolution,
        Stage::FeedForward2,
        Stage::NormOut,
    ];

    fn wants(self, block: Stage) -> bool {
        self == Stage::All || self == block
    }

    pub fn label(self) -> &'static str {
        match self {
            Stage::All => "all",
            Stage::FeedForward1 => "ff1",
            Stage::Attention => "attn",
            Stage::Convolution => "conv",
            Stage::FeedForward2 => "ff2",
            Stage::NormOut => "norm_out",
        }
    }

    pub fn all_blocks() -> &'static [Stage; 5] {
        &Self::ALL_BLOCKS
    }
}

/// Resident GPU weight buffers for one Conformer layer (~96 MiB).
///
/// Uploaded once at model load time and retained across scratch resizes so longer
/// utterances never re-upload the 2.31 GB encoder checkpoint.
struct LayerWeights {
    ff1_w1: wgpu::Buffer,
    ff1_w2: wgpu::Buffer,
    ff1_ln_w: wgpu::Buffer,
    ff1_ln_b: wgpu::Buffer,
    attn_ln_w: wgpu::Buffer,
    attn_ln_b: wgpu::Buffer,
    attn_qkv: wgpu::Buffer,
    attn_pos: wgpu::Buffer,
    attn_out: wgpu::Buffer,
    attn_bias: wgpu::Buffer,
    conv_ln_w: wgpu::Buffer,
    conv_ln_b: wgpu::Buffer,
    conv_pw1: wgpu::Buffer,
    conv_pw2: wgpu::Buffer,
    conv_dw: wgpu::Buffer,
    conv_scale: wgpu::Buffer,
    conv_shift: wgpu::Buffer,
    ff2_w1: wgpu::Buffer,
    ff2_w2: wgpu::Buffer,
    ff2_ln_w: wgpu::Buffer,
    ff2_ln_b: wgpu::Buffer,
    out_ln_w: wgpu::Buffer,
    out_ln_b: wgpu::Buffer,
}

/// Duration-sized scratch buffers, uniforms, and bind groups for [`EncoderGpu`].
struct EncoderRuntime {
    scratch: EncoderScratchGpu,
    shared: SharedBindings,
    /// Retained so the frame count can be rewritten without rebuilding bind groups.
    uniforms: SharedUniforms,
    /// `(steps_or_pos_len, attention_model)` currently uploaded into `scratch.pos_emb`.
    pos_emb_key: std::cell::Cell<(usize, AttentionModel)>,
    readback: wgpu::Buffer,
    layers: Vec<LayerBindings>,
    attn: AttnGemm,
}

/// A full encoder: permanently resident 24-layer weights plus resizable scratch.
pub struct EncoderGpu {
    gpu: std::sync::Arc<GpuGemm>,
    weights: Vec<LayerWeights>,
    attn_scale: f32,
    attention_model: AttentionModel,
    /// Wrapped in `Option` so [`Self::ensure_capacity`] drops old scratch *before*
    /// allocating a larger one, never doubling peak VRAM while keeping `weights` resident.
    runtime: Option<EncoderRuntime>,
}

/// Records one dispatch into an already-open compute pass.
///
/// wgpu tracks storage-buffer hazards between dispatches inside a pass and inserts
/// the barriers itself, so opening a pass per dispatch (576 per encode here) only
/// added pass boundaries.
fn dispatch_in(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    x: u32,
    y: u32,
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(x, y, 1);
}

/// Dispatches a GEMM using the best-fit kernel (`v4` 128x128 or 64x64) for `m x n x k`.
fn dispatch_gemm(
    pass: &mut wgpu::ComputePass<'_>,
    gpu: &GpuGemm,
    bind_group: &wgpu::BindGroup,
    m: usize,
    n: usize,
    k: usize,
) {
    let (pipeline, gx, gy) = gpu.gemm_dispatch(m, n, k);
    dispatch_in(pass, pipeline, bind_group, gx, gy);
}

impl EncoderGpu {
    /// Loads every layer's weights onto the GPU and builds every bind group once.
    pub fn new(steps: usize, store: &crate::torch::Store) -> Result<Self> {
        Self::with_gpu(std::sync::Arc::new(GpuGemm::new()?), steps, store)
    }

    /// Builds the encoder on an existing device, so a caller can share one queue
    /// between `pre_encode` and the encoder.
    pub fn with_gpu(
        gpu: std::sync::Arc<GpuGemm>,
        steps: usize,
        store: &crate::torch::Store,
    ) -> Result<Self> {
        Self::with_gpu_and_model(gpu, steps, store, AttentionModel::RelPos)
    }

    /// Builds the encoder on an existing device with an explicit [`AttentionModel`].
    pub fn with_gpu_and_model(
        gpu: std::sync::Arc<GpuGemm>,
        steps: usize,
        store: &crate::torch::Store,
        attention_model: AttentionModel,
    ) -> Result<Self> {
        attention_model.validate()?;
        let scratch = EncoderScratchGpu::with_capacity_for_model(gpu, steps, attention_model)?;
        Self::build_from_scratch(scratch, store, attention_model)
    }

    /// Builds the encoder for up to `steps` frames.
    pub fn with_capacity(
        gpu: std::sync::Arc<GpuGemm>,
        steps: usize,
        store: &crate::torch::Store,
    ) -> Result<Self> {
        Self::with_gpu_and_model(gpu, steps, store, AttentionModel::RelPos)
    }

    /// Builds the encoder with an explicit query-row chunk size (for testing chunked vs unchunked).
    pub fn with_capacity_and_chunk(
        gpu: std::sync::Arc<GpuGemm>,
        steps: usize,
        chunk_rows: usize,
        store: &crate::torch::Store,
    ) -> Result<Self> {
        let scratch = EncoderScratchGpu::with_capacity_and_chunk(gpu, steps, chunk_rows)?;
        Self::build_from_scratch(scratch, store, AttentionModel::RelPos)
    }

    fn build_from_scratch(
        scratch: EncoderScratchGpu,
        store: &crate::torch::Store,
        attention_model: AttentionModel,
    ) -> Result<Self> {
        let gpu = std::sync::Arc::clone(&scratch.gpu);
        let mut uploader = gpu.bulk_upload();
        let mut weights = Vec::with_capacity(crate::conformer::LAYERS);
        for index in 0..crate::conformer::LAYERS {
            weights.push(Self::load_layer_weights(&mut uploader, store, index)?);
        }
        uploader.finish()?;

        let attn_scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let runtime = Self::build_runtime(scratch, &weights, attn_scale, attention_model)?;
        Ok(Self {
            gpu,
            weights,
            attn_scale,
            attention_model,
            runtime: Some(runtime),
        })
    }

    fn build_runtime(
        scratch: EncoderScratchGpu,
        weights: &[LayerWeights],
        attn_scale: f32,
        attention_model: AttentionModel,
    ) -> Result<EncoderRuntime> {
        let steps = scratch.steps;
        let chunk_rows = scratch.attn_chunk_rows;
        let (pos_table, pos_key_n) = match attention_model {
            AttentionModel::RelPos => (crate::conformer::positional_encoding(steps), steps),
            AttentionModel::RelPosLocalAttn { left, right } => (
                crate::conformer::local_positional_encoding(left, right),
                left + right + 1,
            ),
        };
        scratch
            .gpu
            .queue_ref()
            .write_buffer(&scratch.pos_emb, 0, bytemuck::cast_slice(&pos_table));
        scratch.gpu.queue_ref().submit([]);
        scratch
            .gpu
            .dev()
            .poll(wgpu::PollType::wait_indefinitely())?;

        let make_dims = |m: usize, n: usize, k: usize| {
            scratch.uniform(
                "gemm_dims",
                &Dims {
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                    _pad: 0,
                },
            )
        };
        // Every frame-count-dependent uniform lives here so `rewrite_uniforms` can
        // update all of them when an utterance has a different length than the
        // allocation.
        let count = (steps * D_MODEL) as u32;
        let (c_pad, p_pad, k_pad) = attn_pads(steps);
        let uniforms = SharedUniforms {
            ln_params: scratch.uniform(
                "ln_params",
                &LayerNormParams {
                    steps: steps as u32,
                    width: D_MODEL as u32,
                    eps: LAYER_NORM_EPS,
                    _pad: 0.0,
                },
            ),
            dw_params: scratch.uniform(
                "dw_params",
                &ConvParams {
                    steps: steps as u32,
                    channels: D_MODEL as u32,
                    kernel: CONV_KERNEL as u32,
                    padding: (CONV_KERNEL / 2) as u32,
                },
            ),
            glu_params: scratch.uniform(
                "glu_params",
                &GluParams {
                    steps: steps as u32,
                    width: D_MODEL as u32,
                    _pad0: 0,
                    _pad1: 0,
                },
            ),
            axpy_half_params: scratch.uniform(
                "axpy_half",
                &AxpyParams {
                    count,
                    scale: crate::conformer::FC_FACTOR,
                    _pad0: 0,
                    _pad1: 0,
                },
            ),
            axpy_one_params: scratch.uniform(
                "axpy_one",
                &AxpyParams {
                    count,
                    scale: 1.0,
                    _pad0: 0,
                    _pad1: 0,
                },
            ),
            attn_prepare_params: std::array::from_fn(|head| {
                scratch.uniform(
                    "attn_prepare",
                    &prepare_params_for(steps, head, 0, chunk_rows, true),
                )
            }),
            attn_scatter_params: std::array::from_fn(|head| {
                scratch.uniform(
                    "attn_scatter_params",
                    &scatter_params_for(head, 0, chunk_rows),
                )
            }),
            dims_attn_content: make_dims(chunk_rows, c_pad, HEAD_DIM),
            dims_attn_position: make_dims(chunk_rows, p_pad, HEAD_DIM),
            dims_attn_context: make_dims(chunk_rows, HEAD_DIM, k_pad),
            attn_softmax_params: scratch.uniform(
                "attn_softmax",
                &softmax_params_for(steps, attn_scale, 0),
            ),
            local_attn_params: {
                let (left, right) = match attention_model {
                    AttentionModel::RelPosLocalAttn { left, right } => (left, right),
                    AttentionModel::RelPos => (1, 1),
                };
                let (gx, _) = crate::gpu::grid_2d((steps * HEADS).div_ceil(4));
                scratch.uniform(
                    "local_attn_params",
                    &LocalAttnParams {
                        steps: steps as u32,
                        d_model: D_MODEL as u32,
                        head_dim: HEAD_DIM as u32,
                        pos_len: (left + right + 1) as u32,
                        left_ctx: left as u32,
                        right_ctx: right as u32,
                        window_w: left.max(right) as u32,
                        heads: HEADS as u32,
                        scale: attn_scale,
                        grid_x: gx,
                        _pad0: 0,
                        _pad1: 0,
                    },
                )
            },
            dims_ff_w1: make_dims(steps, FFN, D_MODEL),
            dims_ff_w2: make_dims(steps, D_MODEL, FFN),
            dims_qkv: make_dims(steps, 3 * D_MODEL, D_MODEL),
            dims_pos: make_dims(scratch.pos_len, D_MODEL, D_MODEL),
            dims_pw1: make_dims(steps, 2 * D_MODEL, D_MODEL),
            dims_d_d: make_dims(steps, D_MODEL, D_MODEL),
        };

        let (_, silu_layout) = scratch.gpu.silu_pipeline();
        let (_, glu_layout) = scratch.gpu.glu_pipeline();
        let shared = SharedBindings {
            silu: scratch.gpu.bind_group(
                "silu",
                silu_layout,
                &[(0, scratch.hidden.as_entire_binding())],
            ),
            glu: scratch.gpu.bind_group(
                "glu",
                glu_layout,
                &[
                    (0, uniforms.glu_params.as_entire_binding()),
                    (1, scratch.gates.as_entire_binding()),
                    (2, scratch.normed.as_entire_binding()),
                ],
            ),
            axpy_half: scratch.gpu.bind_group(
                "axpy_half",
                &scratch.axpy_layout,
                &[
                    (0, uniforms.axpy_half_params.as_entire_binding()),
                    (1, scratch.residual.as_entire_binding()),
                    (2, scratch.block.as_entire_binding()),
                ],
            ),
            axpy_one: scratch.gpu.bind_group(
                "axpy_one",
                &scratch.axpy_layout,
                &[
                    (0, uniforms.axpy_one_params.as_entire_binding()),
                    (1, scratch.residual.as_entire_binding()),
                    (2, scratch.block.as_entire_binding()),
                ],
            ),
        };

        let layers: Vec<LayerBindings> = weights
            .iter()
            .map(|w| Self::bind_layer(&scratch, &uniforms, w))
            .collect();

        // GEMM attention: two score GEMMs (content and position use different A
        // operands), a softmax that indexes the relative offset, then the context GEMM.
        let (_, gemm_layout) = scratch.gpu.gemm_pipeline();
        let mk_gemm = |dims: &wgpu::Buffer, a: &wgpu::Buffer, w: &wgpu::Buffer, c: &wgpu::Buffer| {
            scratch.gpu.bind_group(
                "attn_gemm",
                gemm_layout,
                &[
                    (0, dims.as_entire_binding()),
                    (1, a.as_entire_binding()),
                    (2, w.as_entire_binding()),
                    (3, c.as_entire_binding()),
                ],
            )
        };
        let attn = AttnGemm {
            gemm_content: mk_gemm(
                &uniforms.dims_attn_content,
                &scratch.attn_qu,
                &scratch.attn_km,
                &scratch.attn_s_ctx,
            ),
            gemm_position: mk_gemm(
                &uniforms.dims_attn_position,
                &scratch.attn_qv,
                &scratch.attn_pm,
                &scratch.attn_s_pos,
            ),
            softmax: scratch.gpu.bind_group(
                "attn_softmax",
                &scratch.attn_softmax.layout,
                &[
                    (0, uniforms.attn_softmax_params.as_entire_binding()),
                    (1, scratch.attn_s_ctx.as_entire_binding()),
                    (2, scratch.attn_s_pos.as_entire_binding()),
                    (3, scratch.attn_weights.as_entire_binding()),
                ],
            ),
            gemm_context: mk_gemm(
                &uniforms.dims_attn_context,
                &scratch.attn_weights,
                &scratch.attn_vt,
                &scratch.attn_out,
            ),
            scatter: std::array::from_fn(|head| {
                scratch.gpu.bind_group(
                    "attn_scatter",
                    &scratch.attn_scatter.layout,
                    &[
                        (0, uniforms.attn_scatter_params[head].as_entire_binding()),
                        (1, scratch.attn_out.as_entire_binding()),
                        (2, scratch.temp.as_entire_binding()),
                    ],
                )
            }),
        };

        let readback_len = ((steps * D_MODEL * 4).div_ceil(16) * 16) as u64;
        let readback = scratch.gpu.dev().create_buffer(&wgpu::BufferDescriptor {
            label: Some("encoder_readback"),
            size: readback_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(EncoderRuntime {
            scratch,
            shared,
            uniforms,
            pos_emb_key: std::cell::Cell::new((pos_key_n, attention_model)),
            readback,
            layers,
            attn,
        })
    }

    fn rt(&self) -> &EncoderRuntime {
        self.runtime.as_ref().expect("encoder runtime initialized")
    }

    pub fn attention_model(&self) -> AttentionModel {
        self.attention_model
    }

    /// Switches between `rel_pos` and `rel_pos_local_attn` at runtime with zero weight re-upload.
    pub fn set_attention_model(&mut self, model: AttentionModel) -> Result<()> {
        model.validate()?;
        if model == self.attention_model {
            return Ok(());
        }
        self.attention_model = model;
        let cur_steps = self.rt().scratch.steps;
        let need_rebuild = match model {
            AttentionModel::RelPos => self.rt().scratch.local_only,
            AttentionModel::RelPosLocalAttn { left, right } => {
                (left + right + 1) > self.rt().scratch.pos_len || !self.rt().scratch.local_only
            }
        };
        if need_rebuild {
            self.runtime = None;
            let scratch = EncoderScratchGpu::with_capacity_for_model(
                std::sync::Arc::clone(&self.gpu),
                cur_steps,
                model,
            )?;
            self.runtime = Some(Self::build_runtime(
                scratch,
                &self.weights,
                self.attn_scale,
                model,
            )?);
        }
        Ok(())
    }

    /// Grows only scratch buffers and bind groups when `steps` exceeds current capacity.
    ///
    /// The 2.31 GB layer weights stay resident in VRAM and are never re-uploaded.
    pub fn ensure_capacity(&mut self, steps: usize) -> Result<()> {
        let need_pos = match self.attention_model {
            AttentionModel::RelPos => 2 * steps - 1,
            AttentionModel::RelPosLocalAttn { left, right } => left + right + 1,
        };
        let fits = self.runtime.as_ref().is_some_and(|rt| {
            steps <= rt.scratch.steps
                && need_pos <= rt.scratch.pos_len
                && (!rt.scratch.local_only
                    || matches!(self.attention_model, AttentionModel::RelPosLocalAttn { .. }))
        });
        if fits {
            return Ok(());
        }
        // Drop old scratch before allocating larger buffers so peak VRAM never doubles.
        self.runtime = None;
        let scratch = EncoderScratchGpu::with_capacity_for_model(
            std::sync::Arc::clone(&self.gpu),
            steps,
            self.attention_model,
        )?;
        self.runtime = Some(Self::build_runtime(
            scratch,
            &self.weights,
            self.attn_scale,
            self.attention_model,
        )?);
        Ok(())
    }

    pub fn adapter(&self) -> &str {
        self.gpu.adapter()
    }

    /// The device/queue this encoder runs on, for sharing with other stages.
    pub fn gpu(&self) -> std::sync::Arc<GpuGemm> {
        std::sync::Arc::clone(&self.gpu)
    }

    /// True when all 24 layers fit safely into a single command buffer without chunking.
    fn can_single_submit(&self, steps: usize) -> bool {
        match self.attention_model {
            AttentionModel::RelPos => {
                steps <= self.rt().scratch.attn_chunk_rows && steps <= SINGLE_SUBMIT_MAX_STEPS
            }
            AttentionModel::RelPosLocalAttn { .. } => steps <= SINGLE_SUBMIT_MAX_STEPS,
        }
    }

    /// Runs all 24 layers and leaves `[steps][1024]` resident in VRAM.
    ///
    /// The next stage (the device decoder) consumes this buffer directly, so the
    /// encoder output never returns to the host.
    pub fn encode_device(&self, input: &[f32], scale: f32) -> &wgpu::Buffer {
        let rt = self.rt();
        let s = &rt.scratch;
        s.write_residual(input);
        self.sync_scale(scale);
        if self.can_single_submit(s.steps) {
            let mut encoder = s.gpu.encoder("encode_all");
            for layer in &rt.layers {
                self.record_layer_stage(&mut encoder, layer, Stage::All, s.steps);
            }
            s.gpu.queue_ref().submit(Some(encoder.finish()));
        } else {
            self.submit_layers(s.steps, scale);
        }
        &s.residual
    }

    /// Runs the layers on a caller-supplied resident `[steps][1024]` input.
    ///
    /// `source` is typically the output of `pre_encode -> fv_insert` still in VRAM, so
    /// the whole front half of the model never touches the host.
    pub fn encode_resident(&self, source: &wgpu::Buffer, scale: f32) -> &wgpu::Buffer {
        let rt = self.rt();
        let s = &rt.scratch;
        self.sync_scale(scale);
        if self.can_single_submit(s.steps) {
            let mut encoder = s.gpu.encoder("encode_all");
            encoder.copy_buffer_to_buffer(
                source,
                0,
                &s.residual,
                0,
                (s.steps * D_MODEL * 4) as u64,
            );
            for layer in &rt.layers {
                self.record_layer_stage(&mut encoder, layer, Stage::All, s.steps);
            }
            s.gpu.queue_ref().submit(Some(encoder.finish()));
        } else {
            let mut seed = s.gpu.encoder("encode_seed");
            seed.copy_buffer_to_buffer(
                source,
                0,
                &s.residual,
                0,
                (s.steps * D_MODEL * 4) as u64,
            );
            s.gpu.queue_ref().submit(Some(seed.finish()));
            self.submit_layers(s.steps, scale);
        }
        &s.residual
    }

    /// Rewrites every frame-count-dependent uniform for `steps` frames.
    ///
    /// The scratch buffers keep their allocation; only the counts and the position
    /// table change, so an utterance shorter than the allocation runs without
    /// reallocating scratch.
    pub fn rewrite_uniforms(&self, steps: usize, scale: f32) {
        let rt = self.rt();
        let queue = rt.scratch.gpu.queue_ref();
        let u = &rt.uniforms;

        // Attention pads and score dimensions follow the run length, so they are
        // rewritten here rather than fixed at the allocation capacity.
        let init_rows = steps.min(rt.scratch.attn_chunk_rows);
        let (c_pad, p_pad, k_pad) = attn_pads(steps);
        queue.write_buffer(
            &u.dims_attn_content,
            0,
            bytemuck::bytes_of(&Dims {
                m: init_rows as u32,
                n: c_pad as u32,
                k: HEAD_DIM as u32,
                _pad: 0,
            }),
        );
        queue.write_buffer(
            &u.dims_attn_position,
            0,
            bytemuck::bytes_of(&Dims {
                m: init_rows as u32,
                n: p_pad as u32,
                k: HEAD_DIM as u32,
                _pad: 0,
            }),
        );
        queue.write_buffer(
            &u.dims_attn_context,
            0,
            bytemuck::bytes_of(&Dims {
                m: init_rows as u32,
                n: HEAD_DIM as u32,
                k: k_pad as u32,
                _pad: 0,
            }),
        );
        queue.write_buffer(
            &u.attn_softmax_params,
            0,
            bytemuck::bytes_of(&softmax_params_for(steps, scale, 0)),
        );
        for head in 0..HEADS {
            queue.write_buffer(
                &u.attn_prepare_params[head],
                0,
                bytemuck::bytes_of(&prepare_params_for(steps, head, 0, init_rows, true)),
            );
            queue.write_buffer(
                &u.attn_scatter_params[head],
                0,
                bytemuck::bytes_of(&scatter_params_for(head, 0, init_rows)),
            );
        }

        queue.write_buffer(
            &u.ln_params,
            0,
            bytemuck::bytes_of(&LayerNormParams {
                steps: steps as u32,
                width: D_MODEL as u32,
                eps: LAYER_NORM_EPS,
                _pad: 0.0,
            }),
        );
        queue.write_buffer(
            &u.dw_params,
            0,
            bytemuck::bytes_of(&ConvParams {
                steps: steps as u32,
                channels: D_MODEL as u32,
                kernel: CONV_KERNEL as u32,
                padding: (CONV_KERNEL / 2) as u32,
            }),
        );
        queue.write_buffer(
            &u.glu_params,
            0,
            bytemuck::bytes_of(&GluParams {
                steps: steps as u32,
                width: D_MODEL as u32,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        let count = (steps * D_MODEL) as u32;
        queue.write_buffer(
            &u.axpy_half_params,
            0,
            bytemuck::bytes_of(&AxpyParams {
                count,
                scale: crate::conformer::FC_FACTOR,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        queue.write_buffer(
            &u.axpy_one_params,
            0,
            bytemuck::bytes_of(&AxpyParams {
                count,
                scale: 1.0,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        let pos_len = match self.attention_model {
            AttentionModel::RelPos => 2 * steps - 1,
            AttentionModel::RelPosLocalAttn { left, right } => left + right + 1,
        };
        if let AttentionModel::RelPosLocalAttn { left, right } = self.attention_model {
            let (gx, _) = crate::gpu::grid_2d((steps * HEADS).div_ceil(4));
            queue.write_buffer(
                &u.local_attn_params,
                0,
                bytemuck::bytes_of(&LocalAttnParams {
                    steps: steps as u32,
                    d_model: D_MODEL as u32,
                    head_dim: HEAD_DIM as u32,
                    pos_len: pos_len as u32,
                    left_ctx: left as u32,
                    right_ctx: right as u32,
                    window_w: left.max(right) as u32,
                    heads: HEADS as u32,
                    scale,
                    grid_x: gx,
                    _pad0: 0,
                    _pad1: 0,
                }),
            );
        }
        let dims = |m: usize, n: usize, k: usize| Dims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            _pad: 0,
        };
        for (buffer, value) in [
            (&u.dims_ff_w1, dims(steps, FFN, D_MODEL)),
            (&u.dims_ff_w2, dims(steps, D_MODEL, FFN)),
            (&u.dims_qkv, dims(steps, 3 * D_MODEL, D_MODEL)),
            (&u.dims_pos, dims(pos_len, D_MODEL, D_MODEL)),
            (&u.dims_pw1, dims(steps, 2 * D_MODEL, D_MODEL)),
            (&u.dims_d_d, dims(steps, D_MODEL, D_MODEL)),
        ] {
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&value));
        }
        // The position table is a pure function of `(steps, attention_model)` and never
        // mutated by kernels, so skip recomputing/re-uploading it when unchanged.
        let key_n = match self.attention_model {
            AttentionModel::RelPos => steps,
            AttentionModel::RelPosLocalAttn { .. } => pos_len,
        };
        let key = (key_n, self.attention_model);
        if rt.pos_emb_key.get() != key {
            let table = match self.attention_model {
                AttentionModel::RelPos => crate::conformer::positional_encoding(steps),
                AttentionModel::RelPosLocalAttn { left, right } => {
                    crate::conformer::local_positional_encoding(left, right)
                }
            };
            queue.write_buffer(
                &rt.scratch.pos_emb,
                0,
                bytemuck::cast_slice(&table[..pos_len * D_MODEL]),
            );
            rt.pos_emb_key.set(key);
        }
    }

    /// Runs the layers on a host slice of exactly `steps` frames.
    pub fn encode_at(&self, input: &[f32], steps: usize, scale: f32) -> &wgpu::Buffer {
        let s = &self.rt().scratch;
        self.rewrite_uniforms(steps, scale);
        s.gpu
            .queue_ref()
            .write_buffer(&s.residual, 0, bytemuck::cast_slice(&input[..steps * D_MODEL]));
        self.submit_layers(steps, scale);
        &s.residual
    }

    /// Same, taking its input from a resident device buffer of `steps` frames.
    pub fn encode_resident_at(
        &self,
        source: &wgpu::Buffer,
        steps: usize,
        scale: f32,
    ) -> &wgpu::Buffer {
        let s = &self.rt().scratch;
        self.rewrite_uniforms(steps, scale);
        {
            let mut seed = s.gpu.encoder("encode_seed");
            seed.copy_buffer_to_buffer(source, 0, &s.residual, 0, (steps * D_MODEL * 4) as u64);
            s.gpu.queue_ref().submit(Some(seed.finish()));
        }
        self.submit_layers(steps, scale);
        &s.residual
    }

    /// Reads the residual stream back, for diagnostics.
    pub fn read_residual(&self, steps: usize) -> Result<Vec<f32>> {
        let s = &self.rt().scratch;
        s.gpu.read_buffer(&s.residual, steps * D_MODEL)
    }

    /// Records and submits all 24 layers in TDR-safe batches.
    ///
    /// - `steps <= 512` (~41 s): one submission for all 24 layers.
    /// - `513..=1024` (~41–82 s, e.g. 60 s VAD segments): 12 layers per submit
    ///   (~180 ms/submit, zero mid-loop blocking polls).
    /// - `1025..=attn_chunk_rows` (~82–256 s, e.g. 3m.wav): 4 layers per submit
    ///   (~275 ms/submit), double-buffered by waiting only on the *previous* submit so
    ///   the GPU pipeline never drains between batches while staying well below 2 s TDR.
    /// - `steps > attn_chunk_rows` (10m–45m): chunked attention path per layer.
    fn submit_layers(&self, steps: usize, scale: f32) {
        let rt = self.rt();
        let s = &rt.scratch;
        if matches!(self.attention_model, AttentionModel::RelPosLocalAttn { .. }) {
            let batch_size = if steps <= SINGLE_SUBMIT_MAX_STEPS {
                rt.layers.len()
            } else if steps <= 2048 {
                8
            } else if steps <= 8192 {
                3
            } else {
                1
            };
            let mut prev: Option<wgpu::SubmissionIndex> = None;
            for chunk in rt.layers.chunks(batch_size) {
                let mut encoder = s.gpu.encoder("layers_local_batch");
                for layer in chunk {
                    self.record_layer_stage(&mut encoder, layer, Stage::All, steps);
                }
                let idx = s.gpu.queue_ref().submit(Some(encoder.finish()));
                if steps > 1024 {
                    if let Some(done) = prev.replace(idx) {
                        let _ = s.gpu.dev().poll(wgpu::PollType::Wait {
                            submission_index: Some(done),
                            timeout: None,
                        });
                    }
                }
            }
            return;
        }
        if steps <= s.attn_chunk_rows {
            let batch_size = if steps <= SINGLE_SUBMIT_MAX_STEPS {
                rt.layers.len()
            } else if steps <= 1024 {
                12
            } else {
                4
            };
            let mut prev: Option<wgpu::SubmissionIndex> = None;
            for chunk in rt.layers.chunks(batch_size) {
                let mut encoder = s.gpu.encoder("layers_batch");
                for layer in chunk {
                    self.record_layer_stage(&mut encoder, layer, Stage::All, steps);
                }
                let idx = s.gpu.queue_ref().submit(Some(encoder.finish()));
                if steps > 1024 {
                    if let Some(done) = prev.replace(idx) {
                        let _ = s.gpu.dev().poll(wgpu::PollType::Wait {
                            submission_index: Some(done),
                            timeout: None,
                        });
                    }
                }
            }
            return;
        }
        for layer in &rt.layers {
            self.submit_layer_stage(layer, Stage::All, steps, scale);
        }
    }

    /// Submits one layer's selected stage(s), chunking attention queries when
    /// `steps > self.rt().scratch.attn_chunk_rows`.
    fn submit_layer_stage(
        &self,
        layer: &LayerBindings,
        stage: Stage,
        steps: usize,
        scale: f32,
    ) {
        let rt = self.rt();
        let s = &rt.scratch;
        if steps <= s.attn_chunk_rows {
            let mut encoder = s.gpu.encoder("layer");
            self.record_layer_stage(&mut encoder, layer, stage, steps);
            s.gpu.queue_ref().submit(Some(encoder.finish()));
            // Retire each long layer before queuing the next so WDDM never merges 24
            // back-to-back submits into one >2 s uninterrupted compute burst.
            if steps > SINGLE_SUBMIT_MAX_STEPS {
                let _ = s.gpu.dev().poll(wgpu::PollType::wait_indefinitely());
            }
            return;
        }

        let queue = s.gpu.queue_ref();
        let u = &rt.uniforms;
        let steps_u32 = steps as u32;
        let pos_len = 2 * steps - 1;
        let (elem_d_x, elem_d_y) = crate::gpu::grid_2d((steps * D_MODEL).div_ceil(256));
        let (elem_ffn_x, elem_ffn_y) = crate::gpu::grid_2d((steps * FFN).div_ceil(256));

        let (ln_p, _) = s.gpu.layer_norm_pipeline();
        let (silu_p, _) = s.gpu.silu_pipeline();
        let (glu_p, _) = s.gpu.glu_pipeline();
        let (dw_p, _) = s.gpu.dw_conv_pipeline();

        // Part 1: FF1 and the attention QKV/Pos projections.
        if stage.wants(Stage::FeedForward1) || stage.wants(Stage::Attention) {
            let mut encoder = s.gpu.encoder("layer_pre_attn");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("layer_pre_attn"),
                timestamp_writes: None,
            });
            if stage.wants(Stage::FeedForward1) {
                dispatch_in(&mut pass, ln_p, &layer.ln_ff1, steps_u32, 1);
                dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_ff1_w1, steps, FFN, D_MODEL);
                dispatch_in(&mut pass, silu_p, &rt.shared.silu, elem_ffn_x, elem_ffn_y);
                dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_ff1_w2, steps, D_MODEL, FFN);
                dispatch_in(&mut pass, &s.axpy, &rt.shared.axpy_half, elem_d_x, elem_d_y);
            }
            if stage.wants(Stage::Attention) {
                dispatch_in(&mut pass, ln_p, &layer.ln_attn, steps_u32, 1);
                dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_qkv, steps, 3 * D_MODEL, D_MODEL);
                dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_pos, pos_len, D_MODEL, D_MODEL);
            }
            drop(pass);
            queue.submit(Some(encoder.finish()));
        }

        // Part 2: Attention heads, chunked over query rows.
        // Each chunk rewrites the shared attention uniforms and submits on its own so
        // every dispatch sees its own `row_offset`/`chunk_rows` and each submission
        // stays far under the Windows TDR ceiling.
        if stage.wants(Stage::Attention) {
            let chunk_cap = s.attn_chunk_rows;
            let (c_pad, p_pad, k_pad) = attn_pads(steps);
            let kv_total = c_pad * HEAD_DIM + p_pad * HEAD_DIM + HEAD_DIM * k_pad;

            let mut prev_head: Option<wgpu::SubmissionIndex> = None;
            for head in 0..HEADS {
                let mut head_last: Option<wgpu::SubmissionIndex> = None;
                for row_offset in (0..steps).step_by(chunk_cap) {
                    let chunk_rows = (steps - row_offset).min(chunk_cap);
                    let fill_kv = row_offset == 0;
                    queue.write_buffer(
                        &u.attn_prepare_params[head],
                        0,
                        bytemuck::bytes_of(&prepare_params_for(
                            steps,
                            head,
                            row_offset,
                            chunk_rows,
                            fill_kv,
                        )),
                    );
                    queue.write_buffer(
                        &u.dims_attn_content,
                        0,
                        bytemuck::bytes_of(&Dims {
                            m: chunk_rows as u32,
                            n: c_pad as u32,
                            k: HEAD_DIM as u32,
                            _pad: 0,
                        }),
                    );
                    queue.write_buffer(
                        &u.dims_attn_position,
                        0,
                        bytemuck::bytes_of(&Dims {
                            m: chunk_rows as u32,
                            n: p_pad as u32,
                            k: HEAD_DIM as u32,
                            _pad: 0,
                        }),
                    );
                    queue.write_buffer(
                        &u.attn_softmax_params,
                        0,
                        bytemuck::bytes_of(&softmax_params_for(steps, scale, row_offset)),
                    );
                    queue.write_buffer(
                        &u.dims_attn_context,
                        0,
                        bytemuck::bytes_of(&Dims {
                            m: chunk_rows as u32,
                            n: HEAD_DIM as u32,
                            k: k_pad as u32,
                            _pad: 0,
                        }),
                    );
                    queue.write_buffer(
                        &u.attn_scatter_params[head],
                        0,
                        bytemuck::bytes_of(&scatter_params_for(head, row_offset, chunk_rows)),
                    );

                    let prepare_total =
                        2 * chunk_rows * HEAD_DIM + if fill_kv { kv_total } else { 0 };
                    let (prep_x, prep_y) = crate::gpu::grid_2d(prepare_total.div_ceil(256));
                    let (scat_x, scat_y) =
                        crate::gpu::grid_2d((chunk_rows * HEAD_DIM).div_ceil(256));

                    let mut encoder = s.gpu.encoder("attn_chunk");
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("attn_chunk"),
                        timestamp_writes: None,
                    });
                    dispatch_in(
                        &mut pass,
                        &s.attn_prepare.pipeline,
                        &layer.attn_prepare[head],
                        prep_x,
                        prep_y,
                    );
                    dispatch_gemm(
                        &mut pass,
                        &s.gpu,
                        &rt.attn.gemm_content,
                        chunk_rows,
                        c_pad,
                        HEAD_DIM,
                    );
                    dispatch_gemm(
                        &mut pass,
                        &s.gpu,
                        &rt.attn.gemm_position,
                        chunk_rows,
                        p_pad,
                        HEAD_DIM,
                    );
                    let softmax_p = match &s.attn_softmax_smem {
                        Some(smem) if steps <= ATTN_SOFTMAX_SMEM_MAX => smem,
                        _ => &s.attn_softmax.pipeline,
                    };
                    dispatch_in(
                        &mut pass,
                        softmax_p,
                        &rt.attn.softmax,
                        chunk_rows as u32,
                        1,
                    );
                    dispatch_gemm(
                        &mut pass,
                        &s.gpu,
                        &rt.attn.gemm_context,
                        chunk_rows,
                        HEAD_DIM,
                        k_pad,
                    );
                    dispatch_in(
                        &mut pass,
                        &s.attn_scatter.pipeline,
                        &rt.attn.scatter[head],
                        scat_x,
                        scat_y,
                    );
                    drop(pass);
                    head_last = Some(queue.submit(Some(encoder.finish())));
                }
                // Retire the previous head while the current one runs so WDDM stays well
                // inside its 2 s watchdog without draining the GPU pipeline between heads.
                if let Some(done) = prev_head.take() {
                    let _ = s.gpu.dev().poll(wgpu::PollType::Wait {
                        submission_index: Some(done),
                        timeout: None,
                    });
                }
                prev_head = head_last;
            }
            if let Some(done) = prev_head {
                let _ = s.gpu.dev().poll(wgpu::PollType::Wait {
                    submission_index: Some(done),
                    timeout: None,
                });
            }
        }

        // Part 3: Attention output projection + Conv + FF2 + NormOut.
        if stage.wants(Stage::Attention)
            || stage.wants(Stage::Convolution)
            || stage.wants(Stage::FeedForward2)
            || stage.wants(Stage::NormOut)
        {
            let mut encoder = s.gpu.encoder("layer_post_attn");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("layer_post_attn"),
                timestamp_writes: None,
            });
            if stage.wants(Stage::Attention) {
                dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_attn_out, steps, D_MODEL, D_MODEL);
                dispatch_in(&mut pass, &s.axpy, &rt.shared.axpy_one, elem_d_x, elem_d_y);
            }
            if stage.wants(Stage::Convolution) {
                dispatch_in(&mut pass, ln_p, &layer.ln_conv, steps_u32, 1);
                dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_conv_pw1, steps, 2 * D_MODEL, D_MODEL);
                dispatch_in(&mut pass, glu_p, &rt.shared.glu, elem_d_x, elem_d_y);
                dispatch_in(&mut pass, dw_p, &layer.dw_conv, elem_d_x, elem_d_y);
                dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_conv_pw2, steps, D_MODEL, D_MODEL);
                dispatch_in(&mut pass, &s.axpy, &rt.shared.axpy_one, elem_d_x, elem_d_y);
            }
            if stage.wants(Stage::FeedForward2) {
                dispatch_in(&mut pass, ln_p, &layer.ln_ff2, steps_u32, 1);
                dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_ff2_w1, steps, FFN, D_MODEL);
                dispatch_in(&mut pass, silu_p, &rt.shared.silu, elem_ffn_x, elem_ffn_y);
                dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_ff2_w2, steps, D_MODEL, FFN);
                dispatch_in(&mut pass, &s.axpy, &rt.shared.axpy_half, elem_d_x, elem_d_y);
            }
            if stage.wants(Stage::NormOut) {
                dispatch_in(&mut pass, ln_p, &layer.ln_out, steps_u32, 1);
            }
            drop(pass);
            if stage.wants(Stage::NormOut) {
                encoder.copy_buffer_to_buffer(
                    &s.block,
                    0,
                    &s.residual,
                    0,
                    (steps * D_MODEL * 4) as u64,
                );
            }
            queue.submit(Some(encoder.finish()));
            let _ = s.gpu.dev().poll(wgpu::PollType::wait_indefinitely());
        }
    }

    /// Waits for all submitted GPU work to finish; used to isolate phase timings.
    pub fn sync(&self) -> Result<()> {
        self.gpu.dev().poll(wgpu::PollType::wait_indefinitely())?;
        Ok(())
    }

    /// Updates `attn_softmax_params` only if a caller overrides the default attention scale.
    fn sync_scale(&self, scale: f32) {
        if (scale - self.attn_scale).abs() > 1e-7 {
            let rt = self.rt();
            rt.scratch.gpu.queue_ref().write_buffer(
                &rt.uniforms.attn_softmax_params,
                0,
                bytemuck::bytes_of(&softmax_params_for(rt.scratch.steps, scale, 0)),
            );
        }
    }

    /// Times one block family across all 24 layers. Profiling only.
    pub fn encode_stage(
        &self,
        input: &[f32],
        scale: f32,
        stage: Stage,
    ) -> Result<std::time::Duration> {
        let rt = self.rt();
        rt.scratch.write_residual(input);
        self.sync_scale(scale);
        let started = std::time::Instant::now();
        if self.can_single_submit(rt.scratch.steps) {
            let mut encoder = rt.scratch.gpu.encoder("encode_stage");
            for layer in &rt.layers {
                self.record_layer_stage(&mut encoder, layer, stage, rt.scratch.steps);
            }
            rt.scratch.gpu.queue_ref().submit(Some(encoder.finish()));
        } else if rt.scratch.steps <= rt.scratch.attn_chunk_rows {
            let batch_size = if rt.scratch.steps <= 1024 { 12 } else { 4 };
            let mut prev: Option<wgpu::SubmissionIndex> = None;
            for chunk in rt.layers.chunks(batch_size) {
                let mut encoder = rt.scratch.gpu.encoder("stage_batch");
                for layer in chunk {
                    self.record_layer_stage(&mut encoder, layer, stage, rt.scratch.steps);
                }
                let idx = rt.scratch.gpu.queue_ref().submit(Some(encoder.finish()));
                if rt.scratch.steps > 1024 {
                    if let Some(done) = prev.replace(idx) {
                        let _ = rt.scratch.gpu.dev().poll(wgpu::PollType::Wait {
                            submission_index: Some(done),
                            timeout: None,
                        });
                    }
                }
            }
        } else {
            for layer in &rt.layers {
                self.submit_layer_stage(layer, stage, rt.scratch.steps, scale);
            }
        }
        rt.scratch
            .gpu
            .dev()
            .poll(wgpu::PollType::wait_indefinitely())?;
        Ok(started.elapsed())
    }

    /// Runs all 24 layers plus the readback copy.
    pub fn encode(&self, input: &[f32], scale: f32) -> Result<Vec<f32>> {
        let rt = self.rt();
        let s = &rt.scratch;
        s.write_residual(input);
        self.sync_scale(scale);

        let count = s.steps * D_MODEL;
        let byte_len = ((count * 4).div_ceil(16) * 16) as u64;
        if self.can_single_submit(s.steps) {
            let mut encoder = s.gpu.encoder("encode_all");
            for layer in &rt.layers {
                self.record_layer_stage(&mut encoder, layer, Stage::All, s.steps);
            }
            encoder.copy_buffer_to_buffer(&s.residual, 0, &rt.readback, 0, byte_len);
            s.gpu.queue_ref().submit(Some(encoder.finish()));
        } else {
            self.submit_layers(s.steps, scale);
            let mut copy = s.gpu.encoder("encode_readback");
            copy.copy_buffer_to_buffer(&s.residual, 0, &rt.readback, 0, byte_len);
            s.gpu.queue_ref().submit(Some(copy.finish()));
        }

        // Retire the submitted work before mapping: NVIDIA 572.75 intermittently
        // aborts `map_async` if a copy (or cold shader compilation) is still in flight.
        s.gpu.dev().poll(wgpu::PollType::wait_indefinitely())?;
        let slice = rt.readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        s.gpu.dev().poll(wgpu::PollType::wait_indefinitely())?;
        receiver
            .recv()
            .map_err(|error| anyhow::anyhow!("readback cancelled: {error}"))?
            .map_err(|error| anyhow::anyhow!("readback failed: {error}"))?;
        let mapped = slice.get_mapped_range()?;
        let all: &[f32] = bytemuck::cast_slice(&mapped);
        let out = all[..count].to_vec();
        drop(mapped);
        rt.readback.unmap();
        Ok(out)
    }

    fn load_layer_weights(
        up: &mut crate::gpu::BulkUpload<'_>,
        store: &crate::torch::Store,
        index: usize,
    ) -> Result<LayerWeights> {
        let prefix = format!("encoder.layers.{index}");
        let label = |name: &str| format!("L{index}_{name}");

        // Stream every large weight matrix straight out of the mmap via BulkUpload —
        // zero `Vec<f32>` conversion, zero host QKV concatenation, zero `.to_vec()`
        // copy — and fold eval-mode BatchNorm directly from its four 1024-f32 vectors.
        let ff1_w1 = up.upload_tensor(&label("ff1_w1"), store, &format!("{prefix}.feed_forward1.linear1.weight"))?;
        let ff1_w2 = up.upload_tensor(&label("ff1_w2"), store, &format!("{prefix}.feed_forward1.linear2.weight"))?;
        let ff1_ln_w = up.upload_tensor(&label("ff1_ln_w"), store, &format!("{prefix}.norm_feed_forward1.weight"))?;
        let ff1_ln_b = up.upload_tensor(&label("ff1_ln_b"), store, &format!("{prefix}.norm_feed_forward1.bias"))?;

        let attn_ln_w = up.upload_tensor(&label("attn_ln_w"), store, &format!("{prefix}.norm_self_att.weight"))?;
        let attn_ln_b = up.upload_tensor(&label("attn_ln_b"), store, &format!("{prefix}.norm_self_att.bias"))?;
        let attn_qkv = up.upload_concat(
            &label("attn_qkv"),
            store,
            &[
                format!("{prefix}.self_attn.linear_q.weight"),
                format!("{prefix}.self_attn.linear_k.weight"),
                format!("{prefix}.self_attn.linear_v.weight"),
            ],
        )?;
        let attn_pos = up.upload_tensor(&label("attn_pos"), store, &format!("{prefix}.self_attn.linear_pos.weight"))?;
        let attn_out = up.upload_tensor(&label("attn_out"), store, &format!("{prefix}.self_attn.linear_out.weight"))?;
        let attn_bias = up.upload_concat(
            &label("attn_bias"),
            store,
            &[
                format!("{prefix}.self_attn.pos_bias_u"),
                format!("{prefix}.self_attn.pos_bias_v"),
            ],
        )?;

        let conv_ln_w = up.upload_tensor(&label("conv_ln_w"), store, &format!("{prefix}.norm_conv.weight"))?;
        let conv_ln_b = up.upload_tensor(&label("conv_ln_b"), store, &format!("{prefix}.norm_conv.bias"))?;
        let conv_pw1 = up.upload_tensor(&label("conv_pw1"), store, &format!("{prefix}.conv.pointwise_conv1.weight"))?;
        let conv_pw2 = up.upload_tensor(&label("conv_pw2"), store, &format!("{prefix}.conv.pointwise_conv2.weight"))?;
        let conv_dw = up.upload_tensor(&label("conv_dw"), store, &format!("{prefix}.conv.depthwise_conv.weight"))?;
        let bn_w = store.f32(&format!("{prefix}.conv.batch_norm.weight"))?;
        let bn_b = store.f32(&format!("{prefix}.conv.batch_norm.bias"))?;
        let bn_mean = store.f32(&format!("{prefix}.conv.batch_norm.running_mean"))?;
        let bn_var = store.f32(&format!("{prefix}.conv.batch_norm.running_var"))?;
        let mut scale_vec = vec![0.0f32; D_MODEL];
        let mut shift_vec = vec![0.0f32; D_MODEL];
        for ch in 0..D_MODEL {
            scale_vec[ch] = bn_w[ch] / (bn_var[ch] + 1e-5).sqrt();
            shift_vec[ch] = bn_b[ch] - bn_mean[ch] * scale_vec[ch];
        }
        let conv_scale = up.upload_bytes(&label("conv_scale"), bytemuck::cast_slice(&scale_vec))?;
        let conv_shift = up.upload_bytes(&label("conv_shift"), bytemuck::cast_slice(&shift_vec))?;

        let ff2_w1 = up.upload_tensor(&label("ff2_w1"), store, &format!("{prefix}.feed_forward2.linear1.weight"))?;
        let ff2_w2 = up.upload_tensor(&label("ff2_w2"), store, &format!("{prefix}.feed_forward2.linear2.weight"))?;
        let ff2_ln_w = up.upload_tensor(&label("ff2_ln_w"), store, &format!("{prefix}.norm_feed_forward2.weight"))?;
        let ff2_ln_b = up.upload_tensor(&label("ff2_ln_b"), store, &format!("{prefix}.norm_feed_forward2.bias"))?;
        let out_ln_w = up.upload_tensor(&label("out_ln_w"), store, &format!("{prefix}.norm_out.weight"))?;
        let out_ln_b = up.upload_tensor(&label("out_ln_b"), store, &format!("{prefix}.norm_out.bias"))?;

        Ok(LayerWeights {
            ff1_w1,
            ff1_w2,
            ff1_ln_w,
            ff1_ln_b,
            attn_ln_w,
            attn_ln_b,
            attn_qkv,
            attn_pos,
            attn_out,
            attn_bias,
            conv_ln_w,
            conv_ln_b,
            conv_pw1,
            conv_pw2,
            conv_dw,
            conv_scale,
            conv_shift,
            ff2_w1,
            ff2_w2,
            ff2_ln_w,
            ff2_ln_b,
            out_ln_w,
            out_ln_b,
        })
    }

    fn bind_layer(
        scratch: &EncoderScratchGpu,
        u: &SharedUniforms,
        w: &LayerWeights,
    ) -> LayerBindings {
        let gpu = &scratch.gpu;
        let (_, gemm_layout) = gpu.gemm_pipeline();
        let (_, ln_layout) = gpu.layer_norm_pipeline();
        let (_, dw_layout) = gpu.dw_conv_pipeline();

        let mk_ln = |lw: &wgpu::Buffer, lb: &wgpu::Buffer, out: &wgpu::Buffer| {
            gpu.bind_group(
                "ln",
                ln_layout,
                &[
                    (0, u.ln_params.as_entire_binding()),
                    (1, scratch.residual.as_entire_binding()),
                    (2, lw.as_entire_binding()),
                    (3, lb.as_entire_binding()),
                    (4, out.as_entire_binding()),
                ],
            )
        };
        let mk_gemm =
            |dims: &wgpu::Buffer, a: &wgpu::Buffer, gw: &wgpu::Buffer, c: &wgpu::Buffer| {
                gpu.bind_group(
                    "gemm",
                    gemm_layout,
                    &[
                        (0, dims.as_entire_binding()),
                        (1, a.as_entire_binding()),
                        (2, gw.as_entire_binding()),
                        (3, c.as_entire_binding()),
                    ],
                )
            };

        LayerBindings {
            ln_ff1: mk_ln(&w.ff1_ln_w, &w.ff1_ln_b, &scratch.normed),
            gemm_ff1_w1: mk_gemm(&u.dims_ff_w1, &scratch.normed, &w.ff1_w1, &scratch.hidden),
            gemm_ff1_w2: mk_gemm(&u.dims_ff_w2, &scratch.hidden, &w.ff1_w2, &scratch.block),
            ln_attn: mk_ln(&w.attn_ln_w, &w.attn_ln_b, &scratch.normed),
            gemm_qkv: mk_gemm(&u.dims_qkv, &scratch.normed, &w.attn_qkv, &scratch.qkv),
            gemm_pos: mk_gemm(&u.dims_pos, &scratch.pos_emb, &w.attn_pos, &scratch.pos),
            attn_prepare: std::array::from_fn(|head| {
                gpu.bind_group(
                    "attn_prepare",
                    &scratch.attn_prepare.layout,
                    &[
                        (0, u.attn_prepare_params[head].as_entire_binding()),
                        (1, scratch.qkv.as_entire_binding()),
                        (2, scratch.pos.as_entire_binding()),
                        (3, w.attn_bias.as_entire_binding()),
                        (4, scratch.attn_qu.as_entire_binding()),
                        (5, scratch.attn_qv.as_entire_binding()),
                        (6, scratch.attn_km.as_entire_binding()),
                        (7, scratch.attn_pm.as_entire_binding()),
                        (8, scratch.attn_vt.as_entire_binding()),
                    ],
                )
            }),
            attn_local: gpu.bind_group(
                "attn_local",
                &scratch.attn_local.layout,
                &[
                    (0, u.local_attn_params.as_entire_binding()),
                    (1, scratch.qkv.as_entire_binding()),
                    (2, scratch.pos.as_entire_binding()),
                    (3, w.attn_bias.as_entire_binding()),
                    (4, scratch.temp.as_entire_binding()),
                ],
            ),
            gemm_attn_out: mk_gemm(&u.dims_d_d, &scratch.temp, &w.attn_out, &scratch.block),
            ln_conv: mk_ln(&w.conv_ln_w, &w.conv_ln_b, &scratch.normed),
            gemm_conv_pw1: mk_gemm(&u.dims_pw1, &scratch.normed, &w.conv_pw1, &scratch.gates),
            dw_conv: gpu.bind_group(
                "dw_conv",
                dw_layout,
                &[
                    (0, u.dw_params.as_entire_binding()),
                    (1, scratch.normed.as_entire_binding()),
                    (2, w.conv_dw.as_entire_binding()),
                    (3, w.conv_scale.as_entire_binding()),
                    (4, w.conv_shift.as_entire_binding()),
                    (5, scratch.temp.as_entire_binding()),
                ],
            ),
            gemm_conv_pw2: mk_gemm(&u.dims_d_d, &scratch.temp, &w.conv_pw2, &scratch.block),
            ln_ff2: mk_ln(&w.ff2_ln_w, &w.ff2_ln_b, &scratch.normed),
            gemm_ff2_w1: mk_gemm(&u.dims_ff_w1, &scratch.normed, &w.ff2_w1, &scratch.hidden),
            gemm_ff2_w2: mk_gemm(&u.dims_ff_w2, &scratch.hidden, &w.ff2_w2, &scratch.block),
            ln_out: mk_ln(&w.out_ln_w, &w.out_ln_b, &scratch.block),
        }
    }

    fn record_layer_stage(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: &LayerBindings,
        stage: Stage,
        steps: usize,
    ) {
        let rt = self.rt();
        let s = &rt.scratch;
        let is_local = matches!(self.attention_model, AttentionModel::RelPosLocalAttn { .. });
        debug_assert!(is_local || steps <= s.attn_chunk_rows);
        let steps_u32 = steps as u32;
        let pos_len = match self.attention_model {
            AttentionModel::RelPos => 2 * steps - 1,
            AttentionModel::RelPosLocalAttn { left, right } => left + right + 1,
        };
        let (elem_d_x, elem_d_y) = crate::gpu::grid_2d((steps * D_MODEL).div_ceil(256));
        let (elem_ffn_x, elem_ffn_y) = crate::gpu::grid_2d((steps * FFN).div_ceil(256));

        let (ln_p, _) = s.gpu.layer_norm_pipeline();
        let (silu_p, _) = s.gpu.silu_pipeline();
        let (glu_p, _) = s.gpu.glu_pipeline();
        let (dw_p, _) = s.gpu.dw_conv_pipeline();

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("layer"),
            timestamp_writes: None,
        });

        if stage.wants(Stage::FeedForward1) {
            dispatch_in(&mut pass, ln_p, &layer.ln_ff1, steps_u32, 1);
            dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_ff1_w1, steps, FFN, D_MODEL);
            dispatch_in(&mut pass, silu_p, &rt.shared.silu, elem_ffn_x, elem_ffn_y);
            dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_ff1_w2, steps, D_MODEL, FFN);
            dispatch_in(&mut pass, &s.axpy, &rt.shared.axpy_half, elem_d_x, elem_d_y);
        }

        if stage.wants(Stage::Attention) {
            dispatch_in(&mut pass, ln_p, &layer.ln_attn, steps_u32, 1);
            dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_qkv, steps, 3 * D_MODEL, D_MODEL);
            dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_pos, pos_len, D_MODEL, D_MODEL);
            if is_local {
                let (lx, ly) = crate::gpu::grid_2d((steps * HEADS).div_ceil(4));
                dispatch_in(&mut pass, &s.attn_local.pipeline, &layer.attn_local, lx, ly);
            } else {
                // Scores come from the tuned GEMM, not a per-query GEMV: at long lengths the
                // GEMV form re-read K and the position table for every query, which measured
                // 3.0 s of a 4.0 s encoder. Two score GEMMs because the content term uses
                // `q + bias_u` and the position term `q + bias_v`; the softmax turns the
                // relative-position shift into an index; a third GEMM applies the weights
                // to V; the scatter places the head into the combined buffer.
                let (c_pad, p_pad, k_pad) = attn_pads(steps);
                let prepare_total = 2 * steps * HEAD_DIM
                    + c_pad * HEAD_DIM
                    + p_pad * HEAD_DIM
                    + HEAD_DIM * k_pad;
                let (prep_x, prep_y) = crate::gpu::grid_2d(prepare_total.div_ceil(256));
                let (scat_x, scat_y) = crate::gpu::grid_2d((steps * HEAD_DIM).div_ceil(256));
                let softmax_p = match &s.attn_softmax_smem {
                    Some(smem) if steps <= ATTN_SOFTMAX_SMEM_MAX => smem,
                    _ => &s.attn_softmax.pipeline,
                };
                for head in 0..HEADS {
                    dispatch_in(
                        &mut pass,
                        &s.attn_prepare.pipeline,
                        &layer.attn_prepare[head],
                        prep_x,
                        prep_y,
                    );
                    dispatch_gemm(&mut pass, &s.gpu, &rt.attn.gemm_content, steps, c_pad, HEAD_DIM);
                    dispatch_gemm(&mut pass, &s.gpu, &rt.attn.gemm_position, steps, p_pad, HEAD_DIM);
                    dispatch_in(&mut pass, softmax_p, &rt.attn.softmax, steps_u32, 1);
                    dispatch_gemm(&mut pass, &s.gpu, &rt.attn.gemm_context, steps, HEAD_DIM, k_pad);
                    dispatch_in(
                        &mut pass,
                        &s.attn_scatter.pipeline,
                        &rt.attn.scatter[head],
                        scat_x,
                        scat_y,
                    );
                }
            }
            dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_attn_out, steps, D_MODEL, D_MODEL);
            dispatch_in(&mut pass, &s.axpy, &rt.shared.axpy_one, elem_d_x, elem_d_y);
        }

        if stage.wants(Stage::Convolution) {
            dispatch_in(&mut pass, ln_p, &layer.ln_conv, steps_u32, 1);
            dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_conv_pw1, steps, 2 * D_MODEL, D_MODEL);
            dispatch_in(&mut pass, glu_p, &rt.shared.glu, elem_d_x, elem_d_y);
            dispatch_in(&mut pass, dw_p, &layer.dw_conv, elem_d_x, elem_d_y);
            dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_conv_pw2, steps, D_MODEL, D_MODEL);
            dispatch_in(&mut pass, &s.axpy, &rt.shared.axpy_one, elem_d_x, elem_d_y);
        }

        if stage.wants(Stage::FeedForward2) {
            dispatch_in(&mut pass, ln_p, &layer.ln_ff2, steps_u32, 1);
            dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_ff2_w1, steps, FFN, D_MODEL);
            dispatch_in(&mut pass, silu_p, &rt.shared.silu, elem_ffn_x, elem_ffn_y);
            dispatch_gemm(&mut pass, &s.gpu, &layer.gemm_ff2_w2, steps, D_MODEL, FFN);
            dispatch_in(&mut pass, &s.axpy, &rt.shared.axpy_half, elem_d_x, elem_d_y);
        }

        if stage.wants(Stage::NormOut) {
            dispatch_in(&mut pass, ln_p, &layer.ln_out, steps_u32, 1);
        }
        drop(pass);

        if stage.wants(Stage::NormOut) {
            encoder.copy_buffer_to_buffer(
                &s.block,
                0,
                &s.residual,
                0,
                (steps * D_MODEL * 4) as u64,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_rows_stays_unchunked_for_short_and_caps_long() {
        let limit = 2_147_483_644usize;
        assert_eq!(attn_chunk_rows(189, limit).unwrap(), 189);
        assert_eq!(attn_chunk_rows(2580, limit).unwrap(), 2580);

        // 10 min and 25 min utterances chunk cleanly under both budget and binding limit.
        for steps in [7_500usize, 19_066, 37_500] {
            let r = attn_chunk_rows(steps, limit).unwrap();
            assert!(r < steps && r >= 64 && r % 64 == 0, "steps={steps} r={r}");
            let (c_pad, p_pad, k_pad) = attn_pads(steps);
            let total = r * (c_pad + p_pad + k_pad) * 4;
            assert!(total <= ATTN_SCORE_BUDGET, "steps={steps} total={total}");
            assert!(r * p_pad * 4 <= limit);
        }

        // Tight binding limit overrides the budget.
        let tight = attn_chunk_rows(4096, 4 << 20).unwrap();
        let (_, p_pad, _) = attn_pads(4096);
        assert!(tight * p_pad * 4 <= (4 << 20));
        assert!(attn_chunk_rows(19_066, 1024).is_err());
    }

    #[test]
    fn grid_2d_respects_65535_limit_and_covers_workgroups() {
        for wg in [0usize, 1, 256, 65_535, 65_536, 76_264, 305_056, 1_000_000] {
            let (x, y) = crate::gpu::grid_2d(wg);
            assert!(x <= 65_535 && y <= 65_535, "wg={wg} -> ({x}, {y})");
            let total = x as usize * y as usize;
            assert!(total >= wg, "wg={wg} -> ({x}, {y})");
            if wg <= 65_535 {
                assert_eq!((x as usize, y), (wg, 1));
            } else {
                assert!(total - wg < y as usize, "excessive slack for wg={wg}");
            }
        }
    }
}
