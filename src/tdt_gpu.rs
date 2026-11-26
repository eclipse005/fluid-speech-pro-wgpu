//! Autonomous chunked GPU TDT greedy decoder.
//!
//! Keeps the predictor (embedding + 2-layer LSTM), the joint projections and the
//! joint output matrix resident in VRAM on a shared `Arc<GpuGemm>`. Each chunk records
//! `CHUNK` autonomous TDT steps into one `CommandEncoder` and reads back only the
//! 48-byte state; a typical 15 s utterance finishes in a single submission and the
//! encoder output never returns to the host.
//!
//! The decoder state lives on the device, so the per-step host round trips that made
//! an earlier GPU attempt slower than the CPU path are gone entirely.

use std::sync::Arc;

use anyhow::{Result, bail};
use wgpu::util::DeviceExt as _;

use crate::gpu::GpuGemm;
use crate::tdt::{
    BLANK, DEC_HIDDEN, ENC_HIDDEN, JOINT_HIDDEN, JOINT_VOCAB, MAX_SYMBOLS_PER_STEP, Token,
};

const LSTM_MATVEC_SHADER: &str = include_str!("tdt_lstm_matvec.wgsl");
const LSTM_GATE_SHADER: &str = include_str!("tdt_lstm_gate.wgsl");
const PRED_PROJ_SHADER: &str = include_str!("tdt_pred_proj.wgsl");
const JOINT_ARGMAX_SHADER: &str = include_str!("joint_argmax.wgsl");
const STEP_UPDATE_SHADER: &str = include_str!("tdt_step_update.wgsl");

/// Steps recorded per GPU submission. 80 covers a 15-18 s utterance in one submit
/// while keeping the trailing guarded no-ops negligible.
const CHUNK: usize = 80;
const GROUPS: u32 = 256;

/// Borrowed checkpoint weights needed to build the resident GPU decoder.
pub struct TdtWeights<'a> {
    pub embed: &'a [f32],
    pub l0_w_ih: &'a [f32],
    pub l0_w_hh: &'a [f32],
    pub l0_b_ih: &'a [f32],
    pub l0_b_hh: &'a [f32],
    pub l1_w_ih: &'a [f32],
    pub l1_w_hh: &'a [f32],
    pub l1_b_ih: &'a [f32],
    pub l1_b_hh: &'a [f32],
    pub pred_weight: &'a [f32],
    pub pred_bias: &'a [f32],
    pub enc_weight: &'a [f32],
    pub enc_bias: &'a [f32],
    pub out_weight: &'a [f32],
    pub out_bias: &'a [f32],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LstmMatvecParams {
    use_embed: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmDims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TdtState {
    frame: u32,
    symbols: u32,
    token_count: u32,
    last_token: u32,
    need_predict: u32,
    done: u32,
    steps: u32,
    max_tokens: u32,
    visited: u32,
    /// Per-frame symbol cap; 0 means unlimited.
    max_symbols: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuToken {
    id: u32,
    frame: u32,
    duration: u32,
    duration_margin: f32,
    dur0: f32,
    dur1: f32,
    dur2: f32,
    dur3: f32,
    dur4: f32,
    score: f32,
    _pad1: u32,
    _pad2: u32,
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

fn build_pipeline(
    gpu: &GpuGemm,
    label: &'static str,
    source: &'static str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let device = gpu.dev();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: gpu.pipeline_cache(),
    })
}

/// Records one dispatch into an already-open compute pass.
///
/// wgpu tracks storage hazards between dispatches inside a pass and inserts the
/// barriers a Vulkan backend needs anyway, so a pass per dispatch only adds
/// boundaries — 561 of them per chunk in this loop.
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

/// Buffers and bind groups sized for one maximum frame count.
struct TdtScratch {
    capacity_steps: usize,
    max_tokens: usize,
    /// Allocated lazily only by [`TdtGpu::decode_host`]; the resident device path
    /// consumes `EncoderGpu`'s output directly and never needs this buffer.
    host_enc: Option<wgpu::Buffer>,
    enc_proj: wgpu::Buffer,
    out_tokens: wgpu::Buffer,
    bg_joint: wgpu::BindGroup,
    bg_update: wgpu::BindGroup,
}

/// The resident GPU decoder.
pub struct TdtGpu {
    gpu: Arc<GpuGemm>,
    p_lstm_matvec: wgpu::ComputePipeline,
    p_lstm_gate: wgpu::ComputePipeline,
    p_pred_proj: wgpu::ComputePipeline,
    p_joint_argmax: wgpu::ComputePipeline,
    p_step_update: wgpu::ComputePipeline,

    joint_layout: wgpu::BindGroupLayout,
    update_layout: wgpu::BindGroupLayout,

    enc_weight: wgpu::Buffer,
    out_weight: wgpu::Buffer,
    out_bias: wgpu::Buffer,

    state: wgpu::Buffer,
    u_enc_dims: wgpu::Buffer,
    h0: wgpu::Buffer,
    c0: wgpu::Buffer,
    h1: wgpu::Buffer,
    c1: wgpu::Buffer,
    pred_proj: wgpu::Buffer,
    partial: wgpu::Buffer,
    dur_logits: wgpu::Buffer,

    bg_lstm0_mv: wgpu::BindGroup,
    bg_lstm0_gate: wgpu::BindGroup,
    bg_lstm1_mv: wgpu::BindGroup,
    bg_lstm1_gate: wgpu::BindGroup,
    bg_pred_proj: wgpu::BindGroup,

    scratch: Option<TdtScratch>,
}

impl TdtGpu {
    pub fn new(gpu: Arc<GpuGemm>, weights: TdtWeights<'_>) -> Result<Self> {
        if weights.out_weight.len() != JOINT_VOCAB * JOINT_HIDDEN
            || weights.out_bias.len() != JOINT_VOCAB
        {
            bail!("unexpected joint output weight/bias size");
        }
        let device = gpu.dev();

        let uniform = uniform_entry(0);
        let mv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tdt_lstm_mv"),
            entries: &[
                uniform,
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, true),
                storage_entry(7, false),
            ],
        });
        let gate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tdt_lstm_gate"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });
        let pred_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tdt_pred_proj"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
        });
        let joint_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tdt_joint_argmax"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
        });
        let update_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tdt_step_update"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });

        let (p_lstm_matvec, p_lstm_gate, p_pred_proj, p_joint_argmax, p_step_update) =
            std::thread::scope(|s| {
                let t0 = s.spawn(|| build_pipeline(&gpu, "tdt_lstm_mv", LSTM_MATVEC_SHADER, &mv_layout));
                let t1 = s.spawn(|| build_pipeline(&gpu, "tdt_lstm_gate", LSTM_GATE_SHADER, &gate_layout));
                let t2 = s.spawn(|| build_pipeline(&gpu, "tdt_pred_proj", PRED_PROJ_SHADER, &pred_layout));
                let t3 = s.spawn(|| build_pipeline(&gpu, "tdt_joint_argmax", JOINT_ARGMAX_SHADER, &joint_layout));
                let t4 = s.spawn(|| build_pipeline(&gpu, "tdt_step_update", STEP_UPDATE_SHADER, &update_layout));
                (
                    t0.join().expect("tdt_lstm_mv"),
                    t1.join().expect("tdt_lstm_gate"),
                    t2.join().expect("tdt_pred_proj"),
                    t3.join().expect("tdt_joint_argmax"),
                    t4.join().expect("tdt_step_update"),
                )
            });

        let l0_bias: Vec<f32> = weights
            .l0_b_ih
            .iter()
            .zip(weights.l0_b_hh)
            .map(|(a, b)| a + b)
            .collect();
        let l1_bias: Vec<f32> = weights
            .l1_b_ih
            .iter()
            .zip(weights.l1_b_hh)
            .map(|(a, b)| a + b)
            .collect();
        // Fold `enc_bias` into `pred_bias` so `enc_proj` stays a pure bias-free GEMM.
        let combined_pred_bias: Vec<f32> = weights
            .pred_bias
            .iter()
            .zip(weights.enc_bias)
            .map(|(a, b)| a + b)
            .collect();

        let mut up = gpu.bulk_upload();
        let embed = up.upload_bytes("tdt_embed", bytemuck::cast_slice(weights.embed))?;
        let l0_w_ih = up.upload_bytes("tdt_l0_w_ih", bytemuck::cast_slice(weights.l0_w_ih))?;
        let l0_w_hh = up.upload_bytes("tdt_l0_w_hh", bytemuck::cast_slice(weights.l0_w_hh))?;
        let l0_bias_buf = up.upload_bytes("tdt_l0_bias", bytemuck::cast_slice(&l0_bias))?;
        let l1_w_ih = up.upload_bytes("tdt_l1_w_ih", bytemuck::cast_slice(weights.l1_w_ih))?;
        let l1_w_hh = up.upload_bytes("tdt_l1_w_hh", bytemuck::cast_slice(weights.l1_w_hh))?;
        let l1_bias_buf = up.upload_bytes("tdt_l1_bias", bytemuck::cast_slice(&l1_bias))?;
        let pred_w = up.upload_bytes("tdt_pred_w", bytemuck::cast_slice(weights.pred_weight))?;
        let pred_b = up.upload_bytes("tdt_pred_b", bytemuck::cast_slice(&combined_pred_bias))?;
        let enc_weight = up.upload_bytes("tdt_enc_w", bytemuck::cast_slice(weights.enc_weight))?;
        let out_weight = up.upload_bytes("tdt_out_w", bytemuck::cast_slice(weights.out_weight))?;
        let out_bias = up.upload_bytes("tdt_out_b", bytemuck::cast_slice(weights.out_bias))?;
        up.finish()?;

        let make_rw = |label: &str, bytes: usize, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.next_multiple_of(16).max(16) as u64,
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };

        let state_bytes = std::mem::size_of::<TdtState>();
        let state = make_rw(
            "tdt_state",
            state_bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        let u_enc_dims = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tdt_enc_dims"),
            size: std::mem::size_of::<GemmDims>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let u_use_embed = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tdt_use_embed"),
            contents: bytemuck::bytes_of(&LstmMatvecParams {
                use_embed: 1,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let u_use_hidden = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tdt_use_hidden"),
            contents: bytemuck::bytes_of(&LstmMatvecParams {
                use_embed: 0,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let h0 = make_rw("tdt_h0", DEC_HIDDEN * 4, wgpu::BufferUsages::COPY_DST);
        let c0 = make_rw("tdt_c0", DEC_HIDDEN * 4, wgpu::BufferUsages::COPY_DST);
        let h1 = make_rw("tdt_h1", DEC_HIDDEN * 4, wgpu::BufferUsages::COPY_DST);
        let c1 = make_rw("tdt_c1", DEC_HIDDEN * 4, wgpu::BufferUsages::COPY_DST);
        let pre = make_rw("tdt_pre", 4 * DEC_HIDDEN * 4, wgpu::BufferUsages::empty());
        let pred_proj = make_rw(
            "tdt_pred_proj",
            JOINT_HIDDEN * 4,
            wgpu::BufferUsages::empty(),
        );
        let partial = make_rw(
            "tdt_partial",
            GROUPS as usize * 16,
            wgpu::BufferUsages::empty(),
        );
        let dur_logits = make_rw("tdt_dur_logits", 32, wgpu::BufferUsages::empty());

        let bg_lstm0_mv = gpu.bind_group(
            "tdt_l0_mv",
            &mv_layout,
            &[
                (0, u_use_embed.as_entire_binding()),
                (1, state.as_entire_binding()),
                (2, embed.as_entire_binding()),
                (3, h0.as_entire_binding()),
                (4, l0_w_ih.as_entire_binding()),
                (5, l0_w_hh.as_entire_binding()),
                (6, l0_bias_buf.as_entire_binding()),
                (7, pre.as_entire_binding()),
            ],
        );
        let bg_lstm0_gate = gpu.bind_group(
            "tdt_l0_gate",
            &gate_layout,
            &[
                (0, state.as_entire_binding()),
                (1, pre.as_entire_binding()),
                (2, c0.as_entire_binding()),
                (3, h0.as_entire_binding()),
            ],
        );
        let bg_lstm1_mv = gpu.bind_group(
            "tdt_l1_mv",
            &mv_layout,
            &[
                (0, u_use_hidden.as_entire_binding()),
                (1, state.as_entire_binding()),
                (2, h0.as_entire_binding()),
                (3, h1.as_entire_binding()),
                (4, l1_w_ih.as_entire_binding()),
                (5, l1_w_hh.as_entire_binding()),
                (6, l1_bias_buf.as_entire_binding()),
                (7, pre.as_entire_binding()),
            ],
        );
        let bg_lstm1_gate = gpu.bind_group(
            "tdt_l1_gate",
            &gate_layout,
            &[
                (0, state.as_entire_binding()),
                (1, pre.as_entire_binding()),
                (2, c1.as_entire_binding()),
                (3, h1.as_entire_binding()),
            ],
        );
        let bg_pred_proj = gpu.bind_group(
            "tdt_pred_proj",
            &pred_layout,
            &[
                (0, state.as_entire_binding()),
                (1, h1.as_entire_binding()),
                (2, pred_w.as_entire_binding()),
                (3, pred_b.as_entire_binding()),
                (4, pred_proj.as_entire_binding()),
            ],
        );

        Ok(Self {
            gpu,
            p_lstm_matvec,
            p_lstm_gate,
            p_pred_proj,
            p_joint_argmax,
            p_step_update,
            joint_layout,
            update_layout,
            enc_weight,
            out_weight,
            out_bias,
            state,
            u_enc_dims,
            h0,
            c0,
            h1,
            c1,
            pred_proj,
            partial,
            dur_logits,
            bg_lstm0_mv,
            bg_lstm0_gate,
            bg_lstm1_mv,
            bg_lstm1_gate,
            bg_pred_proj,
            scratch: None,
        })
    }

    /// Allocates scratch for `steps` frames, growing only when necessary.
    fn ensure_scratch(&mut self, steps: usize) {
        let needs_alloc = self
            .scratch
            .as_ref()
            .is_none_or(|scratch| scratch.capacity_steps < steps);
        if !needs_alloc {
            return;
        }
        let capacity = steps.max(256);
        let max_tokens = (capacity * MAX_SYMBOLS_PER_STEP).max(256);
        let device = self.gpu.dev();

        let enc_proj = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tdt_enc_proj"),
            size: (capacity * JOINT_HIDDEN * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let token_bytes = (max_tokens * std::mem::size_of::<GpuToken>()) as u64;
        let out_tokens = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tdt_out_tokens"),
            size: token_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg_joint = self.gpu.bind_group(
            "tdt_joint_argmax",
            &self.joint_layout,
            &[
                (0, self.state.as_entire_binding()),
                (1, self.out_weight.as_entire_binding()),
                (2, self.out_bias.as_entire_binding()),
                (3, enc_proj.as_entire_binding()),
                (4, self.pred_proj.as_entire_binding()),
                (5, self.partial.as_entire_binding()),
                (6, self.dur_logits.as_entire_binding()),
            ],
        );
        let bg_update = self.gpu.bind_group(
            "tdt_step_update",
            &self.update_layout,
            &[
                (0, self.state.as_entire_binding()),
                (1, self.partial.as_entire_binding()),
                (2, self.dur_logits.as_entire_binding()),
                (3, out_tokens.as_entire_binding()),
            ],
        );

        self.scratch = Some(TdtScratch {
            capacity_steps: capacity,
            max_tokens,
            host_enc: None,
            enc_proj,
            out_tokens,
            bg_joint,
            bg_update,
        });
    }

    /// Decodes from a host `[steps][1024]` slice (uploads once, then runs on the device).
    pub fn decode_host(&mut self, encoder: &[f32], steps: usize) -> Result<Vec<Token>> {
        self.decode_host_with(encoder, steps, Some(MAX_SYMBOLS_PER_STEP))
    }

    /// Decodes from a host `[steps][1024]` slice with a configurable per-frame symbol cap.
    pub fn decode_host_with(
        &mut self,
        encoder: &[f32],
        steps: usize,
        max_symbols: Option<usize>,
    ) -> Result<Vec<Token>> {
        if encoder.len() != steps * ENC_HIDDEN {
            bail!(
                "encoder output has {} values, expected {}",
                encoder.len(),
                steps * ENC_HIDDEN
            );
        }
        if steps == 0 {
            return Ok(Vec::new());
        }
        self.ensure_scratch(steps);
        {
            let device = self.gpu.dev();
            let scratch = self.scratch.as_mut().expect("tdt scratch");
            let buf = scratch.host_enc.get_or_insert_with(|| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("tdt_host_enc"),
                    size: (scratch.capacity_steps * ENC_HIDDEN * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
            self.gpu
                .queue_ref()
                .write_buffer(buf, 0, bytemuck::cast_slice(encoder));
        }
        self.run(None, steps, max_symbols)
    }

    /// Decodes from an already-resident GPU `encoder_out` buffer (`[steps][1024]`),
    /// with zero host readback of the encoder output.
    pub fn decode_device(
        &mut self,
        encoder_out: &wgpu::Buffer,
        steps: usize,
    ) -> Result<Vec<Token>> {
        self.decode_device_with(encoder_out, steps, Some(MAX_SYMBOLS_PER_STEP))
    }

    /// Resident GPU greedy decode with a configurable per-frame symbol cap (`None` = unlimited).
    pub fn decode_device_with(
        &mut self,
        encoder_out: &wgpu::Buffer,
        steps: usize,
        max_symbols: Option<usize>,
    ) -> Result<Vec<Token>> {
        if steps == 0 {
            return Ok(Vec::new());
        }
        self.ensure_scratch(steps);
        self.run(Some(encoder_out), steps, max_symbols)
    }

    /// Projects a resident GPU encoder output `[steps][1024]` through `joint.enc.weight`
    /// (`[steps][640]`, bias-free) and reads back the compact projected buffer for
    /// advanced decoding modes (alignments, confidence, beam search, LM/boosting, streaming).
    pub fn project_encoder_device(
        &mut self,
        encoder_out: &wgpu::Buffer,
        steps: usize,
    ) -> Result<Vec<f32>> {
        if steps == 0 {
            return Ok(Vec::new());
        }
        self.ensure_scratch(steps);
        self.gpu.queue_ref().write_buffer(
            &self.u_enc_dims,
            0,
            bytemuck::bytes_of(&GemmDims {
                m: steps as u32,
                n: JOINT_HIDDEN as u32,
                k: ENC_HIDDEN as u32,
                _pad: 0,
            }),
        );
        let (_, gemm_layout) = self.gpu.gemm_pipeline();
        let (gemm_pipeline, enc_gx, enc_gy) =
            self.gpu.gemm_dispatch(steps, JOINT_HIDDEN, ENC_HIDDEN);
        let enc_proj_target = self.scratch.as_ref().expect("tdt scratch").enc_proj.clone();
        let bg_enc_proj = self.gpu.bind_group(
            "tdt_enc_proj",
            gemm_layout,
            &[
                (0, self.u_enc_dims.as_entire_binding()),
                (1, encoder_out.as_entire_binding()),
                (2, self.enc_weight.as_entire_binding()),
                (3, enc_proj_target.as_entire_binding()),
            ],
        );
        let mut encoder = self.gpu.encoder("tdt_project_enc");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tdt_project_enc"),
                timestamp_writes: None,
            });
            dispatch_in(&mut pass, gemm_pipeline, &bg_enc_proj, enc_gx, enc_gy);
        }
        self.gpu
            .submit_and_read(encoder, &enc_proj_target, steps * JOINT_HIDDEN)
    }

    /// Copies the decode state back to the host and waits for it.
    ///
    /// Mapping can be rejected while the copy is still in flight on this driver, and it
    /// presents as an aborted `map_async`, so the copy is resubmitted and retried. A
    /// retry re-copies the same 48 bytes, so it cannot change the result.
    fn read_state(&self, state_bytes: u64, chunk_index: usize) -> Result<TdtState> {
        let mut last = String::new();
        for attempt in 0..3 {
            let staging = self.gpu.dev().create_buffer(&wgpu::BufferDescriptor {
                label: Some("tdt_state_staging"),
                size: state_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // A fresh encoder per attempt: `finish` consumes it.
            let mut copy_encoder = self.gpu.encoder("tdt_state_copy");
            copy_encoder.copy_buffer_to_buffer(&self.state, 0, &staging, 0, state_bytes);
            self.gpu
                .queue_ref()
                .submit(Some(copy_encoder.finish()));

            // The copy must retire before the buffer may be mapped.
            self.gpu
                .dev()
                .poll(wgpu::PollType::wait_indefinitely())?;

            let slice = staging.slice(..state_bytes);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            self.gpu
                .dev()
                .poll(wgpu::PollType::wait_indefinitely())?;
            let outcome = receiver
                .recv()
                .map_err(|error| anyhow::anyhow!("tdt state readback cancelled: {error}"))?;
            match outcome {
                Ok(()) => {
                    let mapped = slice.get_mapped_range()?;
                    let state = *bytemuck::from_bytes::<TdtState>(&mapped);
                    drop(mapped);
                    staging.unmap();
                    return Ok(state);
                }
                Err(error) => {
                    last = format!("{error} (chunk {chunk_index}, attempt {attempt})");
                    continue;
                }
            }
        }
        anyhow::bail!("tdt state readback failed after retries: {last}")
    }

    /// Reads `count` emitted tokens back with the same fresh-staging retry pattern.
    fn read_tokens(&self, out_tokens: &wgpu::Buffer, count: usize) -> Result<Vec<Token>> {
        let byte_len = (count * std::mem::size_of::<GpuToken>()) as u64;
        let mut last = String::new();
        for attempt in 0..3 {
            let staging = self.gpu.dev().create_buffer(&wgpu::BufferDescriptor {
                label: Some("tdt_tokens_staging"),
                size: byte_len,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self.gpu.encoder("tdt_read_tokens");
            encoder.copy_buffer_to_buffer(out_tokens, 0, &staging, 0, byte_len);
            self.gpu.queue_ref().submit(Some(encoder.finish()));
            self.gpu
                .dev()
                .poll(wgpu::PollType::wait_indefinitely())?;

            let slice = staging.slice(..byte_len);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            self.gpu
                .dev()
                .poll(wgpu::PollType::wait_indefinitely())?;
            match receiver
                .recv()
                .map_err(|error| anyhow::anyhow!("tdt tokens readback cancelled: {error}"))?
            {
                Ok(()) => {
                    let mapped = slice.get_mapped_range()?;
                    let raw: &[GpuToken] = bytemuck::cast_slice(&mapped);
                    let out: Vec<Token> = raw
                        .iter()
                        .map(|token| Token {
                            id: token.id,
                            frame: token.frame,
                            duration: token.duration,
                            duration_margin: token.duration_margin,
                            duration_logits: [
                                token.dur0, token.dur1, token.dur2, token.dur3, token.dur4,
                            ],
                            score: token.score,
                        })
                        .collect();
                    drop(mapped);
                    staging.unmap();
                    return Ok(out);
                }
                Err(error) => {
                    last = format!("{error} (attempt {attempt})");
                    continue;
                }
            }
        }
        anyhow::bail!("tdt tokens readback failed after retries: {last}")
    }

    fn run(
        &mut self,
        external_enc: Option<&wgpu::Buffer>,
        steps: usize,
        max_symbols: Option<usize>,
    ) -> Result<Vec<Token>> {
        let timing = std::env::var_os("FSP_TDT_TIMING").is_some();
        let started = std::time::Instant::now();
        let cap = max_symbols.unwrap_or(0) as u32;

        let init_state = TdtState {
            frame: 0,
            symbols: 0,
            token_count: 0,
            last_token: BLANK,
            need_predict: 1,
            done: 0,
            steps: steps as u32,
            max_tokens: self.scratch.as_ref().expect("tdt scratch").max_tokens as u32,
            visited: 0,
            max_symbols: cap,
            _pad1: 0,
            _pad2: 0,
        };
        self.gpu
            .queue_ref()
            .write_buffer(&self.state, 0, bytemuck::bytes_of(&init_state));
        self.gpu.queue_ref().write_buffer(
            &self.u_enc_dims,
            0,
            bytemuck::bytes_of(&GemmDims {
                m: steps as u32,
                n: JOINT_HIDDEN as u32,
                k: ENC_HIDDEN as u32,
                _pad: 0,
            }),
        );

        // The encoder projection reads either the host copy or the encoder's own
        // buffer, so its bind group depends on the caller.
        let (_, gemm_layout) = self.gpu.gemm_pipeline();
        let (gemm_pipeline, enc_gx, enc_gy) =
            self.gpu.gemm_dispatch(steps, JOINT_HIDDEN, ENC_HIDDEN);
        let enc_source = match external_enc {
            Some(buffer) => buffer.clone(),
            None => self
                .scratch
                .as_ref()
                .expect("tdt scratch")
                .host_enc
                .as_ref()
                .expect("host_enc initialized by decode_host")
                .clone(),
        };
        let enc_proj_target = self.scratch.as_ref().expect("tdt scratch").enc_proj.clone();
        let bg_enc_proj = self.gpu.bind_group(
            "tdt_enc_proj",
            gemm_layout,
            &[
                (0, self.u_enc_dims.as_entire_binding()),
                (1, enc_source.as_entire_binding()),
                (2, self.enc_weight.as_entire_binding()),
                (3, enc_proj_target.as_entire_binding()),
            ],
        );

        let state_bytes = std::mem::size_of::<TdtState>() as u64;
        // Larger chunks on long utterances cut host poll round-trips while staying
        // ~40 ms per submit (far under Windows' 2 s TDR watchdog).
        let chunk_steps = if steps > 1024 {
            512
        } else if steps > 512 {
            256
        } else {
            CHUNK
        };
        let max_chunks = (steps * MAX_SYMBOLS_PER_STEP).div_ceil(chunk_steps).max(1);
        let mut final_state = init_state;
        let mut chunks_run = 0usize;

        for chunk_index in 0..max_chunks {
            chunks_run += 1;
            let state_staging = self.gpu.dev().create_buffer(&wgpu::BufferDescriptor {
                label: Some("tdt_state_staging"),
                size: state_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self.gpu.encoder("tdt_chunk");
            if chunk_index == 0 {
                // Zero the LSTM state and project every encoder frame once.
                encoder.clear_buffer(&self.h0, 0, None);
                encoder.clear_buffer(&self.c0, 0, None);
                encoder.clear_buffer(&self.h1, 0, None);
                encoder.clear_buffer(&self.c1, 0, None);
            }

            let scratch = self.scratch.as_ref().expect("tdt scratch");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tdt_chunk"),
                timestamp_writes: None,
            });
            if chunk_index == 0 {
                dispatch_in(&mut pass, gemm_pipeline, &bg_enc_proj, enc_gx, enc_gy);
            }
            for _ in 0..chunk_steps {
                dispatch_in(
                    &mut pass,
                    &self.p_lstm_matvec,
                    &self.bg_lstm0_mv,
                    320,
                    1,
                );
                dispatch_in(
                    &mut pass,
                    &self.p_lstm_gate,
                    &self.bg_lstm0_gate,
                    3,
                    1,
                );
                dispatch_in(
                    &mut pass,
                    &self.p_lstm_matvec,
                    &self.bg_lstm1_mv,
                    320,
                    1,
                );
                dispatch_in(
                    &mut pass,
                    &self.p_lstm_gate,
                    &self.bg_lstm1_gate,
                    3,
                    1,
                );
                dispatch_in(
                    &mut pass,
                    &self.p_pred_proj,
                    &self.bg_pred_proj,
                    80,
                    1,
                );
                dispatch_in(
                    &mut pass,
                    &self.p_joint_argmax,
                    &scratch.bg_joint,
                    GROUPS,
                    1,
                );
                dispatch_in(
                    &mut pass,
                    &self.p_step_update,
                    &scratch.bg_update,
                    1,
                    1,
                );
            }

            drop(pass);
            // Fuse the 48-byte state copy into the chunk submission to avoid a second
            // submit+poll round trip; if mapping is rejected, `read_state` retries the copy.
            encoder.copy_buffer_to_buffer(&self.state, 0, &state_staging, 0, state_bytes);
            self.gpu.queue_ref().submit(Some(encoder.finish()));
            self.gpu
                .dev()
                .poll(wgpu::PollType::wait_indefinitely())?;

            let slice = state_staging.slice(..state_bytes);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            self.gpu
                .dev()
                .poll(wgpu::PollType::wait_indefinitely())?;
            final_state = match receiver
                .recv()
                .map_err(|error| anyhow::anyhow!("tdt state readback cancelled: {error}"))?
            {
                Ok(()) => {
                    let mapped = slice.get_mapped_range()?;
                    let state = *bytemuck::from_bytes::<TdtState>(&mapped);
                    drop(mapped);
                    state_staging.unmap();
                    state
                }
                Err(_) => self.read_state(state_bytes, chunk_index)?,
            };
            if final_state.done != 0 {
                break;
            }
        }

        if final_state.done == 0 {
            bail!(
                "device decoder did not finish: frame {} of {} after {chunks_run} chunks",
                final_state.frame,
                steps
            );
        }

        let count = {
            let scratch = self.scratch.as_ref().expect("tdt scratch");
            (final_state.token_count as usize).min(scratch.max_tokens)
        };
        let tokens = if count == 0 {
            Vec::new()
        } else {
            let out_tokens = self.scratch.as_ref().expect("tdt scratch").out_tokens.clone();
            self.read_tokens(&out_tokens, count)?
        };

        if timing {
            eprintln!(
                "[tdt-gpu] steps={steps} visited={} non_blank={} chunks={chunks_run} wall={:.2}ms",
                final_state.visited,
                tokens.len(),
                started.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(tokens)
    }
}
