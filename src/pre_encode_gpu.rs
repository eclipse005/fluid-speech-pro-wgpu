//! GPU implementation of NeMo `ConvSubsampling` (`dw_striding`).
//!
//! Uses NHWC `[time, freq, 256]` throughout the five conv layers so the two 1x1
//! pointwise convolutions (`conv.3`, `conv.6`) run directly on `gemm.wgsl` with no
//! transpose, and one fused 16x16 shared-memory transpose produces the
//! `[valid, 4096]` channel-major flatten before the `out` projection.
//!
//! Frame bookkeeping mirrors the CPU path exactly: `valid` starts at `frames - 1`
//! and each strided stage applies `conv_out_len`. Because rows past `valid` are zero,
//! reading only up to `in_time` in the conv kernels is equivalent to the CPU's
//! `mask_tail`, and the final projection is emitted already truncated to `valid` —
//! the boundary handling that once caused a whole-clip error on one clip.

use std::sync::Arc;

use anyhow::{Context, Result, bail};

use crate::gpu::GpuGemm;
use crate::pre_encode::{CHANNELS, D_MODEL, FEAT_IN};
use crate::torch::Store;

const CONV0_SHADER: &str = include_str!("pre_conv0.wgsl");
const DW_CONV_SHADER: &str = include_str!("pre_dw_conv.wgsl");
const BIAS_ACT_SHADER: &str = include_str!("pre_bias_act.wgsl");
const TRANSPOSE_SHADER: &str = include_str!("pre_transpose_bias_relu.wgsl");

/// Tile sizes mirrored from `gemm.wgsl`.
const BM: u32 = 64;
const BN: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Conv2dParams {
    in_time: u32,
    in_freq: u32,
    out_time: u32,
    out_freq: u32,
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
struct BiasActParams {
    rows: u32,
    cols: u32,
    relu: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TransposeParams {
    steps: u32,
    _pad0: u32,
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

/// Transposes `[256][9]` conv weights into `[9][256]` for coalesced GPU reads.
fn transpose_3x3_weights(w: &[f32]) -> Result<Vec<f32>> {
    if w.len() != CHANNELS * 9 {
        bail!(
            "expected {} weights for a 3x3 conv, got {}",
            CHANNELS * 9,
            w.len()
        );
    }
    let mut out = vec![0.0f32; CHANNELS * 9];
    for channel in 0..CHANNELS {
        for tap in 0..9 {
            out[tap * CHANNELS + channel] = w[channel * 9 + tap];
        }
    }
    Ok(out)
}

/// NeMo's `calculate_conv_output_size` for one axis.
fn conv_out_len(len: usize, kernel: usize, stride: usize, padding: usize) -> usize {
    (len + 2 * padding)
        .checked_sub(kernel)
        .map_or(0, |span| span / stride + 1)
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    label: &'static str,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    x: u32,
    y: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(x, y, 1);
}

/// Buffers and bind groups sized for one maximum frame count.
struct PreScratch {
    capacity_frames: usize,
    mel: wgpu::Buffer,
    buf_a: wgpu::Buffer,
    _buf_b: wgpu::Buffer,

    u_conv0: wgpu::Buffer,
    u_conv2: wgpu::Buffer,
    u_dims3: wgpu::Buffer,
    u_bias3: wgpu::Buffer,
    u_conv5: wgpu::Buffer,
    u_dims6: wgpu::Buffer,
    u_trans: wgpu::Buffer,
    u_dims_out: wgpu::Buffer,
    u_bias_out: wgpu::Buffer,

    bg_conv0: wgpu::BindGroup,
    bg_conv2: wgpu::BindGroup,
    bg_gemm3: wgpu::BindGroup,
    bg_bias3: wgpu::BindGroup,
    bg_conv5: wgpu::BindGroup,
    bg_gemm6: wgpu::BindGroup,
    bg_trans: wgpu::BindGroup,
    bg_gemm_out: wgpu::BindGroup,
    bg_bias_out: wgpu::BindGroup,
}

/// The five-conv subsampling stack plus the final projection, all on the device.
pub struct PreEncodeGpu {
    gpu: Arc<GpuGemm>,
    conv0_pipeline: wgpu::ComputePipeline,
    dw_conv_pipeline: wgpu::ComputePipeline,
    bias_act_pipeline: wgpu::ComputePipeline,
    transpose_pipeline: wgpu::ComputePipeline,
    conv_layout: wgpu::BindGroupLayout,
    bias_act_layout: wgpu::BindGroupLayout,
    transpose_layout: wgpu::BindGroupLayout,

    conv0_w: wgpu::Buffer,
    conv0_b: wgpu::Buffer,
    conv2_w: wgpu::Buffer,
    conv2_b: wgpu::Buffer,
    conv3_w: wgpu::Buffer,
    conv3_b: wgpu::Buffer,
    conv5_w: wgpu::Buffer,
    conv5_b: wgpu::Buffer,
    conv6_w: wgpu::Buffer,
    conv6_b: wgpu::Buffer,
    out_w: wgpu::Buffer,
    out_b: wgpu::Buffer,
    out_bias: Vec<f32>,

    scratch: Option<PreScratch>,
    /// `[steps][D_MODEL]` holding a whole utterance. Bounded by a copy per chunk
    /// rather than by the conv stack's peak grid, so its size is linear in the
    /// audio length instead of quadratic.
    dest: Option<wgpu::Buffer>,
    dest_rows: usize,
    /// Host-visible mirror of `dest`, only allocated by [`Self::forward`].
    host_read: Option<wgpu::Buffer>,
    host_read_rows: usize,
}

impl PreEncodeGpu {
    pub fn new(gpu: Arc<GpuGemm>, store: &Store) -> Result<Self> {
        let device = gpu.dev();
        let conv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pre_conv"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
        });
        let bias_act_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pre_bias_act"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, false),
                storage_entry(2, true),
            ],
        });
        let transpose_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pre_transpose"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });

        let (conv0_pipeline, dw_conv_pipeline, bias_act_pipeline, transpose_pipeline) =
            std::thread::scope(|s| {
                let t0 = s.spawn(|| build_pipeline(&gpu, "pre_conv0", CONV0_SHADER, &conv_layout));
                let t1 = s.spawn(|| build_pipeline(&gpu, "pre_dw_conv", DW_CONV_SHADER, &conv_layout));
                let t2 =
                    s.spawn(|| build_pipeline(&gpu, "pre_bias_act", BIAS_ACT_SHADER, &bias_act_layout));
                let t3 = s.spawn(|| {
                    build_pipeline(&gpu, "pre_transpose", TRANSPOSE_SHADER, &transpose_layout)
                });
                (
                    t0.join().expect("pre_conv0"),
                    t1.join().expect("pre_dw_conv"),
                    t2.join().expect("pre_bias_act"),
                    t3.join().expect("pre_transpose"),
                )
            });

        let load_3x3 = |prefix: &str| -> Result<(wgpu::Buffer, wgpu::Buffer)> {
            let raw = store
                .f32(&format!("{prefix}.weight"))
                .with_context(|| format!("read {prefix}.weight"))?;
            let transposed = transpose_3x3_weights(&raw)?;
            let bias = store
                .f32(&format!("{prefix}.bias"))
                .with_context(|| format!("read {prefix}.bias"))?;
            Ok((
                gpu.upload_plain(&format!("{prefix}.w"), &transposed),
                gpu.upload_plain(&format!("{prefix}.b"), &bias),
            ))
        };
        let load_linear = |prefix: &str| -> Result<(wgpu::Buffer, wgpu::Buffer)> {
            let weight = store
                .f32(&format!("{prefix}.weight"))
                .with_context(|| format!("read {prefix}.weight"))?;
            let bias = store
                .f32(&format!("{prefix}.bias"))
                .with_context(|| format!("read {prefix}.bias"))?;
            Ok((
                gpu.upload_plain(&format!("{prefix}.w"), &weight),
                gpu.upload_plain(&format!("{prefix}.b"), &bias),
            ))
        };

        let (conv0_w, conv0_b) = load_3x3("encoder.pre_encode.conv.0")?;
        let (conv2_w, conv2_b) = load_3x3("encoder.pre_encode.conv.2")?;
        let (conv3_w, conv3_b) = load_linear("encoder.pre_encode.conv.3")?;
        let (conv5_w, conv5_b) = load_3x3("encoder.pre_encode.conv.5")?;
        let (conv6_w, conv6_b) = load_linear("encoder.pre_encode.conv.6")?;
        let (out_w, out_b) = load_linear("encoder.pre_encode.out")?;
        let out_bias = store
            .f32("encoder.pre_encode.out.bias")
            .context("read encoder.pre_encode.out.bias")?;

        Ok(Self {
            gpu,
            conv0_pipeline,
            dw_conv_pipeline,
            bias_act_pipeline,
            transpose_pipeline,
            conv_layout,
            bias_act_layout,
            transpose_layout,
            conv0_w,
            conv0_b,
            conv2_w,
            conv2_b,
            conv3_w,
            conv3_b,
            conv5_w,
            conv5_b,
            conv6_w,
            conv6_b,
            out_w,
            out_b,
            out_bias,
            scratch: None,
            dest: None,
            dest_rows: 0,
            host_read: None,
            host_read_rows: 0,
        })
    }

    /// Allocates scratch for `frames` mel frames, growing only when necessary.
    fn ensure_scratch(&mut self, frames: usize) {
        let needs_alloc = self
            .scratch
            .as_ref()
            .is_none_or(|scratch| scratch.capacity_frames < frames);
        if !needs_alloc {
            return;
        }
        let capacity = frames.max(64);
        let t1_max = conv_out_len(capacity, 3, 2, 1).max(1);
        let t2_max = conv_out_len(t1_max, 3, 2, 1).max(1);

        let device = self.gpu.dev();
        let make_buf = |label: &str, floats: usize, usage: wgpu::BufferUsages| {
            let bytes = ((floats * 4).div_ceil(16) * 16).max(16) as u64;
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        let make_uniform = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let mel = make_buf(
            "pre_mel",
            capacity * FEAT_IN,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let buf_a = make_buf(
            "pre_buf_a",
            t1_max * 64 * CHANNELS,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let buf_b = make_buf(
            "pre_buf_b",
            t2_max * 32 * CHANNELS,
            wgpu::BufferUsages::STORAGE,
        );
        let u_conv0 = make_uniform("u_conv0");
        let u_conv2 = make_uniform("u_conv2");
        let u_dims3 = make_uniform("u_dims3");
        let u_bias3 = make_uniform("u_bias3");
        let u_conv5 = make_uniform("u_conv5");
        let u_dims6 = make_uniform("u_dims6");
        let u_trans = make_uniform("u_trans");
        let u_dims_out = make_uniform("u_dims_out");
        let u_bias_out = make_uniform("u_bias_out");

        let (_, gemm_layout) = self.gpu.gemm_pipeline();

        let bg_conv0 = self.gpu.bind_group(
            "pre_conv0",
            &self.conv_layout,
            &[
                (0, u_conv0.as_entire_binding()),
                (1, mel.as_entire_binding()),
                (2, self.conv0_w.as_entire_binding()),
                (3, self.conv0_b.as_entire_binding()),
                (4, buf_a.as_entire_binding()),
            ],
        );
        let bg_conv2 = self.gpu.bind_group(
            "pre_conv2",
            &self.conv_layout,
            &[
                (0, u_conv2.as_entire_binding()),
                (1, buf_a.as_entire_binding()),
                (2, self.conv2_w.as_entire_binding()),
                (3, self.conv2_b.as_entire_binding()),
                (4, buf_b.as_entire_binding()),
            ],
        );
        let bg_gemm3 = self.gpu.bind_group(
            "pre_gemm3",
            gemm_layout,
            &[
                (0, u_dims3.as_entire_binding()),
                (1, buf_b.as_entire_binding()),
                (2, self.conv3_w.as_entire_binding()),
                (3, buf_a.as_entire_binding()),
            ],
        );
        let bg_bias3 = self.gpu.bind_group(
            "pre_bias3",
            &self.bias_act_layout,
            &[
                (0, u_bias3.as_entire_binding()),
                (1, buf_a.as_entire_binding()),
                (2, self.conv3_b.as_entire_binding()),
            ],
        );
        let bg_conv5 = self.gpu.bind_group(
            "pre_conv5",
            &self.conv_layout,
            &[
                (0, u_conv5.as_entire_binding()),
                (1, buf_a.as_entire_binding()),
                (2, self.conv5_w.as_entire_binding()),
                (3, self.conv5_b.as_entire_binding()),
                (4, buf_b.as_entire_binding()),
            ],
        );
        let bg_gemm6 = self.gpu.bind_group(
            "pre_gemm6",
            gemm_layout,
            &[
                (0, u_dims6.as_entire_binding()),
                (1, buf_b.as_entire_binding()),
                (2, self.conv6_w.as_entire_binding()),
                (3, buf_a.as_entire_binding()),
            ],
        );
        let bg_trans = self.gpu.bind_group(
            "pre_trans",
            &self.transpose_layout,
            &[
                (0, u_trans.as_entire_binding()),
                (1, buf_a.as_entire_binding()),
                (2, self.conv6_b.as_entire_binding()),
                (3, buf_b.as_entire_binding()),
            ],
        );
        let bg_gemm_out = self.gpu.bind_group(
            "pre_gemm_out",
            gemm_layout,
            &[
                (0, u_dims_out.as_entire_binding()),
                (1, buf_b.as_entire_binding()),
                (2, self.out_w.as_entire_binding()),
                (3, buf_a.as_entire_binding()),
            ],
        );
        let bg_bias_out = self.gpu.bind_group(
            "pre_bias_out",
            &self.bias_act_layout,
            &[
                (0, u_bias_out.as_entire_binding()),
                (1, buf_a.as_entire_binding()),
                (2, self.out_b.as_entire_binding()),
            ],
        );

        self.scratch = Some(PreScratch {
            capacity_frames: capacity,
            mel,
            buf_a,
            _buf_b: buf_b,
            u_conv0,
            u_conv2,
            u_dims3,
            u_bias3,
            u_conv5,
            u_dims6,
            u_trans,
            u_dims_out,
            u_bias_out,
            bg_conv0,
            bg_conv2,
            bg_gemm3,
            bg_bias3,
            bg_conv5,
            bg_gemm6,
            bg_trans,
            bg_gemm_out,
            bg_bias_out,
        });
    }

    /// Runs the five-conv stack and the out projection in one submission.
    ///
    /// Returns `([valid][D_MODEL], valid)`, matching `PreEncode::forward`.
    /// Runs the stack and leaves the result in VRAM, returning its buffer and `valid`.
    ///
    /// The next stage (`fv_insert`, then the encoder) consumes it directly, so the
    /// 774 KB readback that `forward` performs is skipped entirely.
    pub fn forward_resident(
        &mut self,
        mel_time_major: &[f32],
        frames: usize,
    ) -> Result<(wgpu::Buffer, usize)> {
        self.record(mel_time_major, frames)
    }

    /// Same stack, with the result copied back to the host.
    pub fn forward(&mut self, mel_time_major: &[f32], frames: usize) -> Result<(Vec<f32>, usize)> {
        let (result, valid4) = self.record(mel_time_major, frames)?;
        let full = crate::pre_encode::total_steps(frames);
        let count = full * D_MODEL;
        let byte_len = (count * 4) as u64;
        self.ensure_host_read(full);
        let mut encoder = self.gpu.encoder("pre_readback");
        encoder.copy_buffer_to_buffer(
            &result,
            0,
            self.host_read.as_ref().expect("host_read initialized"),
            0,
            byte_len,
        );
        self.gpu.queue_ref().submit(Some(encoder.finish()));

        let host_read = self.host_read.as_ref().expect("host_read initialized");
        let slice = host_read.slice(..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.gpu.dev().poll(wgpu::PollType::wait_indefinitely())?;
        receiver
            .recv()
            .map_err(|error| anyhow::anyhow!("pre_encode readback cancelled: {error}"))?
            .map_err(|error| anyhow::anyhow!("pre_encode readback failed: {error}"))?;
        let mapped = slice.get_mapped_range()?;
        let values: &[f32] = bytemuck::cast_slice(&mapped);
        let out = values[..count].to_vec();
        drop(mapped);
        host_read.unmap();
        Ok((out, valid4))
    }

    /// Runs the stack over the whole utterance and submits it.
    ///
    /// Returns the buffer holding `[total_steps(frames)][D_MODEL]` and `valid`. Long audio is
    /// processed in time chunks so the conv stack's peak grid stops growing with the
    /// utterance; each chunk's kept rows are copied into one whole-utterance buffer,
    /// which is what makes a 25-minute clip fit.
    fn record(&mut self, mel_time_major: &[f32], frames: usize) -> Result<(wgpu::Buffer, usize)> {
        let valid_total = crate::pre_encode::valid_steps(frames);
        let full_total = crate::pre_encode::total_steps(frames);
        if valid_total == 0 {
            bail!("input too short: {frames} frames subsample to 0");
        }
        let plan = crate::pre_encode::plan_chunks(frames);
        self.ensure_dest(full_total);

        if full_total > valid_total {
            let dest = self.dest.as_ref().expect("dest initialized");
            for row in valid_total..full_total {
                self.gpu.queue_ref().write_buffer(
                    dest,
                    (row * D_MODEL * 4) as u64,
                    bytemuck::cast_slice(&self.out_bias),
                );
            }
        }

        for chunk in &plan {
            let start = chunk.start * FEAT_IN;
            let end = (chunk.start + chunk.len) * FEAT_IN;
            let (mut encoder, valid) =
                self.record_block(&mel_time_major[start..end], chunk.len)?;
            let src = &self.scratch.as_ref().expect("scratch initialized").buf_a;
            if valid < chunk.keep_from + chunk.rows {
                bail!(
                    "chunk at {} produced {valid} rows, needs {}",
                    chunk.start,
                    chunk.keep_from + chunk.rows
                );
            }
            // The uniforms are rewritten per chunk, so each chunk is submitted on its
            // own: one submission for all of them would leave every dispatch reading
            // the last chunk's dimensions.
            encoder.copy_buffer_to_buffer(
                src,
                (chunk.keep_from * D_MODEL * 4) as u64,
                self.dest.as_ref().expect("dest initialized"),
                (chunk.first_row * D_MODEL * 4) as u64,
                (chunk.rows * D_MODEL * 4) as u64,
            );
            self.gpu.queue_ref().submit(Some(encoder.finish()));
        }

        Ok((self.dest.as_ref().expect("dest initialized").clone(), valid_total))
    }

    /// Grows `dest` to hold `rows` encoder steps.
    fn ensure_dest(&mut self, rows: usize) {
        if self.dest_rows >= rows {
            return;
        }
        let bytes = ((rows * D_MODEL * 4).div_ceil(16) * 16).max(16) as u64;
        self.dest = Some(self.gpu.dev().create_buffer(&wgpu::BufferDescriptor {
            label: Some("pre_dest"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.dest_rows = rows;
    }

    /// Grows the host-visible mirror used by [`Self::forward`].
    fn ensure_host_read(&mut self, rows: usize) {
        if self.host_read_rows >= rows {
            return;
        }
        let bytes = ((rows * D_MODEL * 4).div_ceil(16) * 16).max(16) as u64;
        self.host_read = Some(self.gpu.dev().create_buffer(&wgpu::BufferDescriptor {
            label: Some("pre_host_read"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.host_read_rows = rows;
    }

    /// One chunk: records the whole stack for `frames` mel frames.
    fn record_block(
        &mut self,
        mel_time_major: &[f32],
        frames: usize,
    ) -> Result<(wgpu::CommandEncoder, usize)> {
        crate::pre_encode::check_frames(frames)?;
        if mel_time_major.len() != frames * FEAT_IN {
            bail!(
                "expected {frames} frames of {FEAT_IN} mel bins, found {} values",
                mel_time_major.len()
            );
        }

        let valid0 = frames.saturating_sub(1);
        let valid1 = conv_out_len(valid0, 3, 2, 1);
        let valid2 = conv_out_len(valid1, 3, 2, 1);
        let valid4 = conv_out_len(valid2, 3, 2, 1);
        if valid1 == 0 || valid2 == 0 || valid4 == 0 {
            bail!("input too short: {frames} frames subsample to {valid4}");
        }

        self.ensure_scratch(frames);
        let scratch = self.scratch.as_ref().expect("scratch initialized");
        let queue = self.gpu.queue_ref();

        queue.write_buffer(&scratch.mel, 0, bytemuck::cast_slice(mel_time_major));

        let m3 = valid2 * 32;
        let m6 = valid4 * 16;

        queue.write_buffer(
            &scratch.u_conv0,
            0,
            bytemuck::bytes_of(&Conv2dParams {
                in_time: frames as u32,
                in_freq: FEAT_IN as u32,
                out_time: valid1 as u32,
                out_freq: 64,
            }),
        );
        queue.write_buffer(
            &scratch.u_conv2,
            0,
            bytemuck::bytes_of(&Conv2dParams {
                in_time: valid1 as u32,
                in_freq: 64,
                out_time: valid2 as u32,
                out_freq: 32,
            }),
        );
        queue.write_buffer(
            &scratch.u_dims3,
            0,
            bytemuck::bytes_of(&GemmDims {
                m: m3 as u32,
                n: CHANNELS as u32,
                k: CHANNELS as u32,
                _pad: 0,
            }),
        );
        queue.write_buffer(
            &scratch.u_bias3,
            0,
            bytemuck::bytes_of(&BiasActParams {
                rows: m3 as u32,
                cols: CHANNELS as u32,
                relu: 1,
                _pad: 0,
            }),
        );
        queue.write_buffer(
            &scratch.u_conv5,
            0,
            bytemuck::bytes_of(&Conv2dParams {
                in_time: valid2 as u32,
                in_freq: 32,
                out_time: valid4 as u32,
                out_freq: 16,
            }),
        );
        queue.write_buffer(
            &scratch.u_dims6,
            0,
            bytemuck::bytes_of(&GemmDims {
                m: m6 as u32,
                n: CHANNELS as u32,
                k: CHANNELS as u32,
                _pad: 0,
            }),
        );
        queue.write_buffer(
            &scratch.u_trans,
            0,
            bytemuck::bytes_of(&TransposeParams {
                steps: valid4 as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );
        queue.write_buffer(
            &scratch.u_dims_out,
            0,
            bytemuck::bytes_of(&GemmDims {
                m: valid4 as u32,
                n: D_MODEL as u32,
                k: (CHANNELS * 16) as u32,
                _pad: 0,
            }),
        );
        queue.write_buffer(
            &scratch.u_bias_out,
            0,
            bytemuck::bytes_of(&BiasActParams {
                rows: valid4 as u32,
                cols: D_MODEL as u32,
                relu: 0,
                _pad: 0,
            }),
        );

        let (gemm_pipeline, _) = self.gpu.gemm_pipeline();
        let mut encoder = self.gpu.encoder("pre_encode");
        dispatch(
            &mut encoder,
            "pre_conv0",
            &self.conv0_pipeline,
            &scratch.bg_conv0,
            64,
            valid1 as u32,
        );
        dispatch(
            &mut encoder,
            "pre_conv2",
            &self.dw_conv_pipeline,
            &scratch.bg_conv2,
            32,
            valid2 as u32,
        );
        dispatch(
            &mut encoder,
            "pre_conv3_gemm",
            gemm_pipeline,
            &scratch.bg_gemm3,
            (m3 as u32).div_ceil(self.gpu.tile().bm),
            (CHANNELS as u32).div_ceil(self.gpu.tile().bn),
        );
        dispatch(
            &mut encoder,
            "pre_conv3_bias",
            &self.bias_act_pipeline,
            &scratch.bg_bias3,
            (CHANNELS / 256) as u32,
            (m3 as u32).div_ceil(8),
        );
        dispatch(
            &mut encoder,
            "pre_conv5",
            &self.dw_conv_pipeline,
            &scratch.bg_conv5,
            16,
            valid4 as u32,
        );
        dispatch(
            &mut encoder,
            "pre_conv6_gemm",
            gemm_pipeline,
            &scratch.bg_gemm6,
            (m6 as u32).div_ceil(self.gpu.tile().bm),
            (CHANNELS as u32).div_ceil(self.gpu.tile().bn),
        );
        dispatch(
            &mut encoder,
            "pre_conv6_trans",
            &self.transpose_pipeline,
            &scratch.bg_trans,
            (CHANNELS / 16) as u32,
            valid4 as u32,
        );
        dispatch(
            &mut encoder,
            "pre_out_gemm",
            gemm_pipeline,
            &scratch.bg_gemm_out,
            (valid4 as u32).div_ceil(BM),
            (D_MODEL as u32).div_ceil(BN),
        );
        dispatch(
            &mut encoder,
            "pre_out_bias",
            &self.bias_act_pipeline,
            &scratch.bg_bias_out,
            (D_MODEL / 256) as u32,
            (valid4 as u32).div_ceil(8),
        );

        Ok((encoder, valid4))
    }
}
