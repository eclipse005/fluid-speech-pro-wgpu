// LayerNorm over the last axis: one workgroup per time step.
//
// D_MODEL = 1024 and the workgroup is 256 threads, so each thread reduces four
// elements; the two reductions (mean, then mean square) run over shared memory.

struct Params {
    steps: u32,
    width: u32,
    eps: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const THREADS: u32 = 256u;

var<workgroup> reduction: array<f32, THREADS>;

/// Sum over the workgroup, leaving the total in `reduction[0]`.
fn reduce(value: f32, lane: u32) -> f32 {
    reduction[lane] = value;
    workgroupBarrier();
    var stride = THREADS / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            reduction[lane] = reduction[lane] + reduction[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return reduction[0];
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) step: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let width = params.width;
    let base = step.x * width;
    let lane = local.x;
    let per_thread = width / THREADS;

    var local_sum = 0.0f;
    for (var index: u32 = 0u; index < per_thread; index = index + 1u) {
        local_sum = local_sum + x[base + lane * per_thread + index];
    }
    let mean = reduce(local_sum, lane) / f32(width);

    var local_square = 0.0f;
    for (var index: u32 = 0u; index < per_thread; index = index + 1u) {
        let delta = x[base + lane * per_thread + index] - mean;
        local_square = local_square + delta * delta;
    }
    let variance = reduce(local_square, lane) / f32(width);
    let inv_std = 1.0f / sqrt(variance + params.eps);

    for (var index: u32 = 0u; index < per_thread; index = index + 1u) {
        let at = lane * per_thread + index;
        out[base + at] = (x[base + at] - mean) * inv_std * weight[at] + bias[at];
    }
}
