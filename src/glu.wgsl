// GLU over the channel axis: `out[t, c] = in[t, c] * sigmoid(in[t, c + width])`.
//
// The pointwise conv that precedes it produced a 2*width-wide buffer.

struct GluParams {
    steps: u32,
    width: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: GluParams;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let total = params.steps * params.width;
    let at = id.y * (grid.x * 256u) + id.x;
    if (at >= total) {
        return;
    }
    let step = at / params.width;
    let channel = at % params.width;
    let base = step * 2u * params.width;
    let gate = x[base + channel];
    let value = x[base + params.width + channel];
    out[at] = gate / (1.0f + exp(-value));
}
