//! On-device `fv_insert`, the block NeMo hooks onto `encoder.pre_encode`.
//!
//! `y = x + up(GELU(mix(down(LayerNorm(x)))))`
//!
//! Running it on the device is what lets `pre_encode -> fv_insert -> encoder` stay in
//! VRAM: previously the subsampled features made a 774 KB round trip to the host,
//! were processed there, and were uploaded again.
//!
//! `down` and `up` are Linear layers, so each is a GEMM *plus a bias add*; the bias is
//! added by the shared `pre_bias_act.wgsl` pass, which is also what `pre_encode_gpu`
//! uses. The middle stage is a depthwise Conv1d(15, padding 7) fused with the exact
//! (erf-based) GELU.

use anyhow::{Context, Result};
use std::sync::Arc;

use crate::gpu::GpuGemm;
use crate::torch::Store;

const DW_SHADER: &str = include_str!("dw_conv_gelu.wgsl");
const BIAS_SHADER: &str = include_str!("pre_bias_act.wgsl");
const AXPY_SHADER: &str = include_str!("axpy.wgsl");

const WIDTH: usize = 1024;
const KERNEL: usize = 15;
const PADDING: usize = KERNEL / 2;
const LAYER_NORM_EPS: f32 = 1e-5;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvParams {
    steps: u32,
    channels: u32,
    kernel: u32,
    padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerNormParams {
    steps: u32,
    width: u32,
    eps: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BiasActParams {
    rows: u32,
    cols: u32,
    relu: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AxpyParams {
    count: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
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

/// Frame-sized scratch buffers and bind groups for [`FvInsertGpu`].
///
/// Kept separate from the resident weight buffers so [`FvInsertGpu::ensure_capacity`]
/// can grow scratch for longer utterances without ever re-uploading weights.
struct FvScratch {
    capacity: usize,
    normed: wgpu::Buffer,
    _down_out: wgpu::Buffer,
    _mixed: wgpu::Buffer,
    _up_out: wgpu::Buffer,
    out: wgpu::Buffer,
    bg_down: wgpu::BindGroup,
    bg_bias_down: wgpu::BindGroup,
    bg_dw: wgpu::BindGroup,
    bg_up: wgpu::BindGroup,
    bg_bias_up: wgpu::BindGroup,
    bg_axpy: wgpu::BindGroup,
}

/// The block with its weights and pipelines resident permanently in VRAM.
pub struct FvInsertGpu {
    gpu: Arc<GpuGemm>,

    dw_pipeline: wgpu::ComputePipeline,
    dw_layout: wgpu::BindGroupLayout,
    bias_pipeline: wgpu::ComputePipeline,
    bias_layout: wgpu::BindGroupLayout,
    axpy_pipeline: wgpu::ComputePipeline,
    axpy_layout: wgpu::BindGroupLayout,

    ln_w: wgpu::Buffer,
    ln_b: wgpu::Buffer,
    down_w: wgpu::Buffer,
    down_b: wgpu::Buffer,
    mix_w: wgpu::Buffer,
    mix_b: wgpu::Buffer,
    up_w: wgpu::Buffer,
    up_b: wgpu::Buffer,

    u_ln: wgpu::Buffer,
    u_dw: wgpu::Buffer,
    u_dims: wgpu::Buffer,
    u_bias: wgpu::Buffer,
    u_axpy: wgpu::Buffer,

    scratch: Option<FvScratch>,
}

impl FvInsertGpu {
    /// Uploads the block's weights once and allocates initial scratch for `steps` frames.
    pub fn new(gpu: Arc<GpuGemm>, steps: usize, store: &Store) -> Result<Self> {
        let prefix = "encoder.fv_insert";
        let device = gpu.dev();
        let build = |label: &'static str,
                     source: &'static str,
                     entries: &[wgpu::BindGroupLayoutEntry]|
         -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
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
            (pipeline, layout)
        };

        let (dw_pipeline, dw_layout) = build(
            "dw_conv_gelu",
            DW_SHADER,
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
        );
        let (bias_pipeline, bias_layout) = build(
            "fv_bias",
            BIAS_SHADER,
            &[
                uniform_entry(0),
                storage_entry(1, false),
                storage_entry(2, true),
            ],
        );
        let (axpy_pipeline, axpy_layout) = build(
            "fv_axpy",
            AXPY_SHADER,
            &[
                uniform_entry(0),
                storage_entry(1, false),
                storage_entry(2, true),
            ],
        );

        let load = |name: &str| -> Result<wgpu::Buffer> {
            let values = store
                .f32(&format!("{prefix}.{name}"))
                .with_context(|| format!("read {prefix}.{name}"))?;
            Ok(gpu.upload_plain(&format!("fv_{name}"), &values))
        };
        let ln_w = load("ln.weight")?;
        let ln_b = load("ln.bias")?;
        let down_w = load("down.weight")?;
        let down_b = load("down.bias")?;
        let mix_w = load("mix.weight")?;
        let mix_b = load("mix.bias")?;
        let up_w = load("up.weight")?;
        let up_b = load("up.bias")?;

        let u_ln = gpu.uniform_buffer(
            "fv_ln",
            &LayerNormParams {
                steps: steps as u32,
                width: WIDTH as u32,
                eps: LAYER_NORM_EPS,
                _pad: 0.0,
            },
        );
        let u_dw = gpu.uniform_buffer(
            "fv_dw",
            &ConvParams {
                steps: steps as u32,
                channels: WIDTH as u32,
                kernel: KERNEL as u32,
                padding: PADDING as u32,
            },
        );
        let u_dims = gpu.uniform_buffer(
            "fv_dims",
            &Dims {
                m: steps as u32,
                n: WIDTH as u32,
                k: WIDTH as u32,
                _pad: 0,
            },
        );
        let u_bias = gpu.uniform_buffer(
            "fv_bias",
            &BiasActParams {
                rows: steps as u32,
                cols: WIDTH as u32,
                relu: 0,
                _pad: 0,
            },
        );
        let u_axpy = gpu.uniform_buffer(
            "fv_axpy",
            &AxpyParams {
                count: (steps * WIDTH) as u32,
                scale: 1.0,
                _pad0: 0,
                _pad1: 0,
            },
        );

        let mut this = Self {
            gpu,
            dw_pipeline,
            dw_layout,
            bias_pipeline,
            bias_layout,
            axpy_pipeline,
            axpy_layout,
            ln_w,
            ln_b,
            down_w,
            down_b,
            mix_w,
            mix_b,
            up_w,
            up_b,
            u_ln,
            u_dw,
            u_dims,
            u_bias,
            u_axpy,
            scratch: None,
        };
        this.ensure_capacity(steps.max(1));
        Ok(this)
    }

    fn sc(&self) -> &FvScratch {
        self.scratch.as_ref().expect("fv_insert scratch initialized")
    }

    /// Grows only the scratch buffers and bind groups when `steps` exceeds current
    /// capacity; resident weights stay untouched in VRAM.
    pub fn ensure_capacity(&mut self, steps: usize) {
        if self.scratch.as_ref().is_some_and(|s| steps <= s.capacity) {
            return;
        }
        // Free previous scratch before allocating larger buffers so peak VRAM never doubles.
        self.scratch = None;
        let device = self.gpu.dev();
        let make = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: ((steps * WIDTH * 4).div_ceil(16) * 16) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let normed = make("fv_normed");
        let down_out = make("fv_down");
        let mixed = make("fv_mixed");
        let up_out = make("fv_up");
        let out = make("fv_out");

        let (_, gemm_layout) = self.gpu.gemm_pipeline();
        let mk_gemm = |a: &wgpu::Buffer, w: &wgpu::Buffer, c: &wgpu::Buffer| {
            self.gpu.bind_group(
                "fv_gemm",
                gemm_layout,
                &[
                    (0, self.u_dims.as_entire_binding()),
                    (1, a.as_entire_binding()),
                    (2, w.as_entire_binding()),
                    (3, c.as_entire_binding()),
                ],
            )
        };
        let mk_bias = |data: &wgpu::Buffer, bias: &wgpu::Buffer| {
            self.gpu.bind_group(
                "fv_bias",
                &self.bias_layout,
                &[
                    (0, self.u_bias.as_entire_binding()),
                    (1, data.as_entire_binding()),
                    (2, bias.as_entire_binding()),
                ],
            )
        };

        let bg_down = mk_gemm(&normed, &self.down_w, &down_out);
        let bg_bias_down = mk_bias(&down_out, &self.down_b);
        let bg_dw = self.gpu.bind_group(
            "fv_dw",
            &self.dw_layout,
            &[
                (0, self.u_dw.as_entire_binding()),
                (1, down_out.as_entire_binding()),
                (2, self.mix_w.as_entire_binding()),
                (3, self.mix_b.as_entire_binding()),
                (4, mixed.as_entire_binding()),
            ],
        );
        let bg_up = mk_gemm(&mixed, &self.up_w, &up_out);
        let bg_bias_up = mk_bias(&up_out, &self.up_b);
        let bg_axpy = self.gpu.bind_group(
            "fv_axpy",
            &self.axpy_layout,
            &[
                (0, self.u_axpy.as_entire_binding()),
                (1, out.as_entire_binding()),
                (2, up_out.as_entire_binding()),
            ],
        );

        self.scratch = Some(FvScratch {
            capacity: steps,
            normed,
            _down_out: down_out,
            _mixed: mixed,
            _up_out: up_out,
            out,
            bg_down,
            bg_bias_down,
            bg_dw,
            bg_up,
            bg_bias_up,
            bg_axpy,
        });
    }

    /// Frames this instance can process without reallocating scratch.
    pub fn capacity(&self) -> usize {
        self.sc().capacity
    }

    /// Points every uniform at `steps` frames, which must not exceed the capacity.
    fn rewrite_uniforms(&self, steps: usize) {
        let queue = self.gpu.queue_ref();
        queue.write_buffer(
            &self.u_ln,
            0,
            bytemuck::bytes_of(&LayerNormParams {
                steps: steps as u32,
                width: WIDTH as u32,
                eps: LAYER_NORM_EPS,
                _pad: 0.0,
            }),
        );
        queue.write_buffer(
            &self.u_dw,
            0,
            bytemuck::bytes_of(&ConvParams {
                steps: steps as u32,
                channels: WIDTH as u32,
                kernel: KERNEL as u32,
                padding: PADDING as u32,
            }),
        );
        queue.write_buffer(
            &self.u_dims,
            0,
            bytemuck::bytes_of(&Dims {
                m: steps as u32,
                n: WIDTH as u32,
                k: WIDTH as u32,
                _pad: 0,
            }),
        );
        queue.write_buffer(
            &self.u_bias,
            0,
            bytemuck::bytes_of(&BiasActParams {
                rows: steps as u32,
                cols: WIDTH as u32,
                relu: 0,
                _pad: 0,
            }),
        );
        queue.write_buffer(
            &self.u_axpy,
            0,
            bytemuck::bytes_of(&AxpyParams {
                count: (steps * WIDTH) as u32,
                scale: 1.0,
                _pad0: 0,
                _pad1: 0,
            }),
        );
    }

    /// Seeds the residual from a host slice, for callers not chaining from a buffer.
    pub fn write_input(&self, values: &[f32]) {
        self.gpu
            .queue_ref()
            .write_buffer(&self.sc().out, 0, bytemuck::cast_slice(values));
    }

    /// The block's output (`x + up(...)`), `[steps][1024]`.
    pub fn output(&self) -> &wgpu::Buffer {
        &self.sc().out
    }

    /// Records the whole block into `encoder`; `x` is the caller's input buffer.
    ///
    /// One compute pass for all seven dispatches: wgpu inserts the storage barriers
    /// between them, and a pass boundary here costs ~15-20 us.
    pub fn record_apply(&self, encoder: &mut wgpu::CommandEncoder, x: &wgpu::Buffer) {
        self.record_apply_at(encoder, x, self.sc().capacity);
    }

    /// Same, for `steps` frames (must not exceed [`Self::capacity`]).
    pub fn record_apply_at(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        steps: usize,
    ) {
        let sc = self.sc();
        self.rewrite_uniforms(steps);
        let (ln_p, ln_layout) = self.gpu.layer_norm_pipeline();
        let (gemm_p, gemm_gx, gemm_gy) = self.gpu.gemm_dispatch(steps, WIDTH, WIDTH);
        let frames = steps as u32;
        let (elem_x, elem_y) = crate::gpu::grid_2d((steps * WIDTH).div_ceil(256));

        // The residual has to survive the block, so `x` is copied into `out` first and
        // the final axpy accumulates into that copy.
        if !std::ptr::eq(x as *const _, &sc.out as *const _) {
            encoder.copy_buffer_to_buffer(x, 0, &sc.out, 0, (steps * WIDTH * 4) as u64);
        }
        let bg_ln = self.gpu.bind_group(
            "fv_ln",
            ln_layout,
            &[
                (0, self.u_ln.as_entire_binding()),
                (1, x.as_entire_binding()),
                (2, self.ln_w.as_entire_binding()),
                (3, self.ln_b.as_entire_binding()),
                (4, sc.normed.as_entire_binding()),
            ],
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fv_insert"),
            timestamp_writes: None,
        });
        pass.set_pipeline(ln_p);
        pass.set_bind_group(0, &bg_ln, &[]);
        pass.dispatch_workgroups(frames, 1, 1);

        pass.set_pipeline(gemm_p);
        pass.set_bind_group(0, &sc.bg_down, &[]);
        pass.dispatch_workgroups(gemm_gx, gemm_gy, 1);

        pass.set_pipeline(&self.bias_pipeline);
        pass.set_bind_group(0, &sc.bg_bias_down, &[]);
        pass.dispatch_workgroups((WIDTH / 256) as u32, frames.div_ceil(8), 1);

        pass.set_pipeline(&self.dw_pipeline);
        pass.set_bind_group(0, &sc.bg_dw, &[]);
        pass.dispatch_workgroups(elem_x, elem_y, 1);

        pass.set_pipeline(gemm_p);
        pass.set_bind_group(0, &sc.bg_up, &[]);
        pass.dispatch_workgroups(gemm_gx, gemm_gy, 1);

        pass.set_pipeline(&self.bias_pipeline);
        pass.set_bind_group(0, &sc.bg_bias_up, &[]);
        pass.dispatch_workgroups((WIDTH / 256) as u32, frames.div_ceil(8), 1);

        pass.set_pipeline(&self.axpy_pipeline);
        pass.set_bind_group(0, &sc.bg_axpy, &[]);
        pass.dispatch_workgroups(elem_x, elem_y, 1);
    }

    /// Records and submits the block without waiting, for callers that chain the next
    /// stage onto the same queue and only synchronise once at the end.
    pub fn submit_apply(&self, x: &wgpu::Buffer) {
        self.submit_apply_at(x, self.sc().capacity);
    }

    /// Same, for `steps` frames (must not exceed [`Self::capacity`]).
    pub fn submit_apply_at(&self, x: &wgpu::Buffer, steps: usize) {
        let mut encoder = self.gpu.encoder("fv_insert");
        self.record_apply_at(&mut encoder, x, steps);
        self.gpu.queue_ref().submit(Some(encoder.finish()));
    }

    /// Convenience wrapper that submits and waits; used by the comparison command.
    pub fn apply(&self, x: &wgpu::Buffer) -> Result<()> {
        let mut encoder = self.gpu.encoder("fv_insert");
        self.record_apply(&mut encoder, x);
        self.gpu.queue_ref().submit(Some(encoder.finish()));
        self.gpu.dev().poll(wgpu::PollType::wait_indefinitely())?;
        Ok(())
    }

    /// Reads the output back for verification.
    pub fn read_output(&self) -> Result<Vec<f32>> {
        let sc = self.sc();
        self.gpu.read_buffer(&sc.out, sc.capacity * WIDTH)
    }
}
