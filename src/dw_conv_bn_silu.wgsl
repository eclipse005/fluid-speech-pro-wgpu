// Fused depthwise Conv1d(k=9, pad 4) + folded BatchNorm + SiLU.
//
// One invocation per (step, channel). The BatchNorm is already folded into a
// per-channel `scale`/`shift` pair on the host, so the whole ConformerConv module
// tail is a single dispatch with no intermediate buffers.
//

struct ConvParams {
    steps: u32,
    channels: u32,
    kernel: u32,
    padding: u32,
};

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> scale: array<f32>;
@group(0) @binding(4) var<storage, read> shift: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let total = params.steps * params.channels;
    let at = id.y * (grid.x * 256u) + id.x;
    if (at >= total) {
        return;
    }
    let row = at / params.channels;
    let channel = at % params.channels;
    let padding = i32(params.padding);

    let lo = 0;
    let hi = i32(params.steps);
    let step = i32(row);

    var sum = 0.0f;
    for (var tap: u32 = 0u; tap < params.kernel; tap = tap + 1u) {
        let source = step + i32(tap) - padding;
        if (source < lo || source >= hi) {
            continue;
        }
        sum = sum + weights[channel * params.kernel + tap] * x[u32(source) * params.channels + channel];
    }

    let value = sum * scale[channel] + shift[channel];
    out[at] = value / (1.0f + exp(-value));
}
