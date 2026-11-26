// Residual accumulation: `x = x + scale * y`, elementwise.
//
// The three residual adds in a Conformer layer differ only by `scale`
// (`FC_FACTOR` for the feed-forward halves, `1` for attention and convolution), so
// one kernel covers all of them.

struct AxpyParams {
    count: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: AxpyParams;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let at = id.y * (grid.x * 256u) + id.x;
    if (at < params.count) {
        x[at] = x[at] + params.scale * y[at];
    }
}
