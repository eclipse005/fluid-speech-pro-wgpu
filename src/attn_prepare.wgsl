// Prepares one attention head (or one query-row chunk of a head) for GEMM attention.
//
// The hand-written score kernel was a GEMV per query — arithmetic intensity ~0.5
// FLOP/byte — so at T=2580 the per-head K and position tables stop fitting in L2 and it
// becomes a memory wall (measured 3.0 s of a 4.0 s encoder). Replacing it with the
// tuned GEMM needs the operands in that kernel's layout, which is what this produces.
// It also emits two A operands rather than one: the content term uses `q + pos_bias_u`
// and the position term uses `q + pos_bias_v`, so one GEMM cannot serve both.
//
// Long utterances process queries in row chunks of `chunk_rows` starting at
// `row_offset`. The first chunk of a head sets `fill_kv = 1` to populate `km`, `pm`,
// `vt`; subsequent chunks set `fill_kv = 0` and refresh only `qu` and `qv`.
//
//   qu `[chunk_rows][D]`   q + pos_bias_u   A for the content scores
//   qv `[chunk_rows][D]`   q + pos_bias_v   A for the position scores
//   km `[c_pad][D]`        K                W for the content scores
//   pm `[p_pad][D]`        P                W for the position scores
//   vt `[D][k_pad]`        V transposed     W for the context GEMM

struct Params {
    steps: u32,
    head: u32,
    d_model: u32,
    head_dim: u32,
    pos_len: u32,
    /// `align(steps, 16)`: the context GEMM consumes `k` in blocks of 16.
    k_pad: u32,
    /// `align(steps, 4)` and `align(pos_len, 4)`: the score GEMMs write `vec4`s along
    /// `n`, so both outputs must have a width divisible by four.
    c_pad: u32,
    p_pad: u32,
    /// First global query row in this chunk (`0` when unchunked).
    row_offset: u32,
    /// Number of query rows in this chunk (`steps` when unchunked).
    chunk_rows: u32,
    /// Non-zero on the first chunk of a head to also populate `km`, `pm`, `vt`.
    fill_kv: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
/// `[steps][3 * d_model]`: Q | K | V per row.
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
/// `[pos_len][d_model]`, the projected position table.
@group(0) @binding(2) var<storage, read> pos: array<f32>;
/// `[2 * d_model]`: `pos_bias_u` then `pos_bias_v`.
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> qu: array<f32>;
@group(0) @binding(5) var<storage, read_write> qv: array<f32>;
@group(0) @binding(6) var<storage, read_write> km: array<f32>;
@group(0) @binding(7) var<storage, read_write> pm: array<f32>;
@group(0) @binding(8) var<storage, read_write> vt: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let steps = params.steps;
    let d = params.head_dim;
    let dm = params.d_model;
    let width = 3u * dm;
    let pos_len = params.pos_len;
    let base = params.head * d;

    let q_n = params.chunk_rows * d;
    let k_n = params.c_pad * d;
    let p_n = params.p_pad * d;
    let kv_total = select(0u, k_n + p_n + d * params.k_pad, params.fill_kv != 0u);
    let total = q_n + q_n + kv_total;
    let at = id.y * (grid.x * 256u) + id.x;
    if (at >= total) {
        return;
    }

    // qu, then qv for rows `[row_offset, row_offset + chunk_rows)`.
    if (at < q_n) {
        let row = at / d;
        let col = at % d;
        qu[at] = qkv[(params.row_offset + row) * width + base + col] + bias[base + col];
        return;
    }
    if (at < q_n + q_n) {
        let rest = at - q_n;
        let row = rest / d;
        let col = rest % d;
        qv[rest] = qkv[(params.row_offset + row) * width + base + col] + bias[dm + base + col];
        return;
    }
    // K rows, zero-padded to `c_pad`.
    if (at < q_n + q_n + k_n) {
        let rest = at - q_n - q_n;
        let row = rest / d;
        let col = rest % d;
        if (row < steps) {
            km[rest] = qkv[row * width + dm + base + col];
        } else {
            km[rest] = 0.0f;
        }
        return;
    }

    // Position rows, zero-padded to `p_pad`.
    let rest = at - q_n - q_n - k_n;
    if (rest < p_n) {
        let row = rest / d;
        let col = rest % d;
        if (row < pos_len) {
            pm[rest] = pos[row * dm + base + col];
        } else {
            pm[rest] = 0.0f;
        }
        return;
    }

    // V transposed, `[d][k_pad]`: columns past `steps` stay zero so the context GEMM
    // can run with `k = k_pad`.
    let idx = rest - p_n;
    let ch = idx / params.k_pad;
    let frame = idx % params.k_pad;
    if (frame < steps) {
        vt[idx] = qkv[frame * width + 2u * dm + base + ch];
    } else {
        vt[idx] = 0.0f;
    }
}
