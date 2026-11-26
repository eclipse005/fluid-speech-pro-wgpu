// Places one head's attention output (or one query chunk) into `[steps][d_model]`.
//
// The context GEMM produces `[chunk_rows][head_dim]` per head; the conformer's
// projection expects the heads side by side starting at `row_offset`.

struct Params {
    chunk_rows: u32,
    d_model: u32,
    head_dim: u32,
    head: u32,
    row_offset: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> head_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> combined: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let total = params.chunk_rows * params.head_dim;
    let at = id.y * (grid.x * 256u) + id.x;
    if (at >= total) {
        return;
    }
    let row = at / params.head_dim;
    let col = at % params.head_dim;
    combined[(params.row_offset + row) * params.d_model + params.head * params.head_dim + col] =
        head_out[at];
}
