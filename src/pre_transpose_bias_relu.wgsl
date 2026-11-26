// Fused conv.6 bias + ReLU + NHWC [steps, 16, 256] -> channel-major [steps, 256, 16].
// Dispatch: (16, steps, 1) workgroups of (16, 16, 1) threads; bank-conflict-free.

struct TransposeParams {
    steps: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: TransposeParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

var<workgroup> tile: array<f32, 16u * 17u>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let c_base = group.x * 16u;
    let step = group.y;

    let c_in = c_base + local.x;
    let f_in = local.y;
    let raw = src[(step * 16u + f_in) * 256u + c_in] + bias[c_in];
    tile[f_in * 17u + local.x] = max(raw, 0.0f);
    workgroupBarrier();

    let f_out = local.x;
    let c_out = c_base + local.y;
    dst[(step * 256u + c_out) * 16u + f_out] = tile[f_out * 17u + local.y];
}
