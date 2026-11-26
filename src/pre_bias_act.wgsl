// In-place per-column bias add with optional ReLU, 8 rows per thread.
// Used for conv.3 (cols=256, relu=1) and the out projection (cols=1024, relu=0).

struct BiasActParams {
    rows: u32,
    cols: u32,
    relu: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: BiasActParams;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let col = group.x * 256u + local.x;
    if (col >= params.cols) {
        return;
    }
    let b = bias[col];
    let r0 = group.y * 8u;
    let r1 = min(r0 + 8u, params.rows);
    for (var r: u32 = r0; r < r1; r = r + 1u) {
        let at = r * params.cols + col;
        var v = data[at] + b;
        if (params.relu != 0u && v < 0.0f) {
            v = 0.0f;
        }
        data[at] = v;
    }
}
