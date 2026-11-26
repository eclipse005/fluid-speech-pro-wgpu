//! Pure fp32 FMA throughput probe, so the GEMM's rate has a measured ceiling.
//!
//! Every earlier performance number in this project came from comparing one GEMM
//! against another; "1.77 TFLOP/s is near the wall" was inference. This runs a kernel
//! of nothing but register FMAs and reports the card's arithmetic rate for that mix.
//!
//! Measurement hygiene (the first version of this probe got these wrong and
//! under-reported by roughly a third):
//!
//! * two throwaway dispatches plus a `poll` first, so the driver's lazy pipeline
//!   compilation and the clock's ramp to its sustained state are outside the window;
//! * all timed dispatches go into **one** `CommandEncoder` with a single submit, so
//!   per-submit overhead is amortised and the GPU never returns to idle;
//! * a window of a second or more, so clock ramp-up is negligible;
//! * three modes, including a control with half the flops and the same instruction
//!   count, which proves the loop is real and not folded away.

use anyhow::Result;

use crate::gpu::GpuGemm;

const SHADER: &str = include_str!("flops.wgsl");

/// `Params` uniform, mirrors `flops.wgsl`. 48 bytes, no padding.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    iters: u32,
    mode: u32,
    _pad0: u32,
    _pad1: u32,
    x: [f32; 4],
    y: [f32; 4],
}

const WORKGROUPS: u32 = 512;
const THREADS: u32 = 256;
const ITERS: u32 = 1024;
const ROUNDS: u32 = 4;
/// Accumulators per round, per thread.
const ACCS: u32 = 8;
const REPS: u32 = 200;

pub fn run() -> Result<()> {
    let gpu = GpuGemm::new()?;
    println!("adapter {}", gpu.adapter());
    // What the device will accept matters for any f16/quantised path later on.
    let features = gpu.dev().features();
    let limits = gpu.dev().limits();
    println!(
        "limits: shared/workgroup={} max_buffer={} max_storage_binding={}",
        limits.max_compute_workgroup_storage_size,
        limits.max_buffer_size,
        limits.max_storage_buffer_binding_size
    );
    println!(
        "features: shader_f16={} float32_filterable={} timestamp_query={}",
        features.contains(wgpu::Features::SHADER_F16),
        features.contains(wgpu::Features::FLOAT32_FILTERABLE),
        features.contains(wgpu::Features::TIMESTAMP_QUERY),
    );
    let device = gpu.dev();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flops"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("flops"),
        entries: &[
            entry(0, wgpu::BufferBindingType::Uniform),
            entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("flops"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("flops"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buffer_out = gpu.storage_out("flops_out", 1024);
    let bind_group_for = |mode: u32| {
        let params = Params {
            iters: ITERS,
            mode,
            _pad0: 0,
            _pad1: 0,
            x: [1.0000001, 0.9999999, 1.0000002, 0.9999998],
            y: [1.0e-9, -1.0e-9, 1.0e-9, -1.0e-9],
        };
        let buffer_params = gpu.uniform_buffer("flops_params", &params);
        gpu.bind_group(
            "flops",
            &layout,
            &[
                (0, buffer_params.as_entire_binding()),
                (1, buffer_out.as_entire_binding()),
            ],
        )
    };

    let dispatch = |reps: u32, bind_group: &wgpu::BindGroup| -> Result<()> {
        let mut encoder = gpu.encoder("flops");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flops"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            for _ in 0..reps {
                pass.dispatch_workgroups(WORKGROUPS, 1, 1);
            }
        }
        gpu.queue_ref().submit(Some(encoder.finish()));
        gpu.dev().poll(wgpu::PollType::wait_indefinitely())?;
        Ok(())
    };

    // Throwaway warm-up: pipeline compilation and the clock ramp both land here.
    let warm = bind_group_for(0);
    dispatch(2, &warm)?;

    // Flops per thread per iteration, by mode.
    let vec4_ops = ROUNDS * ACCS;
    let flops_per_iter = |mode: u32| -> f64 {
        let per_op = if mode == 2 { 4.0 } else { 8.0 };
        f64::from(vec4_ops) * per_op
    };

    for (mode, label) in [(0u32, "fma(a,x,y)"), (1, "a*x+y"), (2, "a+y (control)")] {
        let bind_group = bind_group_for(mode);
        let started = std::time::Instant::now();
        dispatch(REPS, &bind_group)?;
        let elapsed = started.elapsed().as_secs_f64();

        let flops = f64::from(WORKGROUPS)
            * f64::from(THREADS)
            * f64::from(ITERS)
            * flops_per_iter(mode)
            * f64::from(REPS);
        println!(
            "mode {mode} {label:14}: {:.1} GFLOP in {:.3}s = {:.2} TFLOP/s",
            flops / 1e9,
            elapsed,
            flops / elapsed / 1e12
        );
    }
    Ok(())
}
