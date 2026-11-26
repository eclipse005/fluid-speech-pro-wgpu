// Tiled f32 GEMM: `C[m, n] = A[m, k] @ W[n, k]^T`.
//
// Both operands are row-major with `k` contiguous — the layout the checkpoint
// delivers and the CPU path uses — so nothing is repacked for the GPU.
//
// Tile: BM=64 rows, BN=64 columns, BK=16 per shared-memory stage.
// Workgroup: 16x16 = 256 threads (8 warps, enough to fill Pascal's four warp
// schedulers); each thread owns a 4x4 block held in four **named** `vec4`
// accumulators. Named rather than an indexed array on purpose: naga lowers an
// array behind a loop index to local memory, which costs an order of magnitude.
//
// Tile sizes were tuned by measurement, not by the usual recipe. A 4x8 block with
// BN=128 raises the FMA-per-`LDS` ratio from 8 to 10.7, yet measured *slower*
// (487 vs 695 GFLOP/s here): the encoder's `n` is 1024..4096 at BM=3 row tiles, so
// BN=128 leaves only 24 workgroups for `n=1024`, i.e. 1.5 waves on this card's 16
// SMs. BK=32 was slower again — the shared tile grows to 27 KB, only one workgroup
// fits per SM. Small `m` (189 frames) also rules out BM=128.
//
// Staging keeps the natural `[row][k4]` order and pads each row by one `vec4`, so
// the inner loop reads eight real 128-bit `LDS.128` and does 64 FMAs with them
// (8:1). Two earlier layouts were measurably worse: `[BM][BK]` without padding
// put all 16 x-threads on one bank (16-way conflict), and building a `vec4` from
// four scalar array reads compiled to four scalar loads instead of one.

struct Dims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> c: array<vec4<f32>>;

// Tile shape is substituted by the host (`GpuGemm::tile`) so it can be calibrated per
// card without editing this file.
const BM: u32 = {{BM}};
const BN: u32 = {{BN}};
const BK: u32 = {{BK}};
/// `vec4`s along `k` in one stage.
const K4: u32 = BK / 4u;
/// Row stride of a staging buffer: `K4` plus one padding slot.
const STRIDE: u32 = K4 + 1u;

const STAGE_SLOTS_A: u32 = BM * STRIDE;
const STAGE_SLOTS_W: u32 = BN * STRIDE;

// Double-buffered: stage `i+1` is fetched while stage `i` is multiplied, so the
// global-load latency is not sitting exposed in front of every stage. Shared use
// doubles to ~20 KB, which still allows multiple workgroups per SM.
var<workgroup> tile_a: array<vec4<f32>, BM * STRIDE * 2u>;
var<workgroup> tile_w: array<vec4<f32>, BN * STRIDE * 2u>;

/// Gathers component `q` of four accumulated `vec4`s into one output `vec4`.
///
/// Written as straight-line branches on `q` rather than an indexed array so naga
/// keeps the accumulators in registers.
fn pick_quad(
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    v3: vec4<f32>,
    q: u32,
) -> vec4<f32> {
    if (q == 0u) { return vec4<f32>(v0.x, v1.x, v2.x, v3.x); }
    if (q == 1u) { return vec4<f32>(v0.y, v1.y, v2.y, v3.y); }
    if (q == 2u) { return vec4<f32>(v0.z, v1.z, v2.z, v3.z); }
    return vec4<f32>(v0.w, v1.w, v2.w, v3.w);
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) block: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let m = dims.m;
    let n = dims.n;
    let k = dims.k;
    let stride_k4 = k / 4u;
    // `c` is `[m][n]`, so its row stride is `n/4`, not the operand stride.
    let stride_c4 = n / 4u;

    let row_base = block.x * BM;
    let col_base = block.y * BN;

    let lane = local.y * 16u + local.x;
    // Each of the 256 threads stages exactly one `vec4` per operand per stage.
    let stage_row = lane / K4;
    let stage_k4 = lane % K4;

    let thread_row = local.y * 4u;
    // Columns owned by this thread: `local.x + {0,16,32,48}`. Consecutive lanes then
    // read consecutive rows of the tile, so their addresses differ by one padded row
    // (20 floats, gcd(20,32)=4 -> eight distinct banks) instead of by four rows
    // (80 floats -> two distinct banks, an 8-way conflict).
    let thread_col = local.x;

    var c0 = vec4<f32>(0.0f);
    var c1 = vec4<f32>(0.0f);
    var c2 = vec4<f32>(0.0f);
    var c3 = vec4<f32>(0.0f);

    let a_row = row_base + stage_row;
    let w_row = col_base + stage_row;

    let stages = (k + BK - 1u) / BK;

    // Preload stage 0 into buffer 0.
    {
        let k4_base = 0u;
        var a_values = vec4<f32>(0.0f);
        if (a_row < m && (k4_base + stage_k4) < stride_k4) {
            a_values = a[a_row * stride_k4 + k4_base + stage_k4];
        }
        tile_a[stage_row * STRIDE + stage_k4] = a_values;
        var w_values = vec4<f32>(0.0f);
        if (w_row < n && (k4_base + stage_k4) < stride_k4) {
            w_values = w[w_row * stride_k4 + k4_base + stage_k4];
        }
        tile_w[stage_row * STRIDE + stage_k4] = w_values;
    }
    workgroupBarrier();

    for (var stage_index: u32 = 0u; stage_index < stages; stage_index = stage_index + 1u) {
        let current = (stage_index % 2u) * STAGE_SLOTS_A;
        let current_w = (stage_index % 2u) * STAGE_SLOTS_W;
        let next = ((stage_index + 1u) % 2u) * STAGE_SLOTS_A;
        let next_w = ((stage_index + 1u) % 2u) * STAGE_SLOTS_W;

        // Fetch the next stage before computing this one, so the global loads are
        // in flight while the FMAs run.
        if (stage_index + 1u < stages) {
            let k4_base = (stage_index + 1u) * K4;
            var a_values = vec4<f32>(0.0f);
            if (a_row < m && (k4_base + stage_k4) < stride_k4) {
                a_values = a[a_row * stride_k4 + k4_base + stage_k4];
            }
            tile_a[next + stage_row * STRIDE + stage_k4] = a_values;
            var w_values = vec4<f32>(0.0f);
            if (w_row < n && (k4_base + stage_k4) < stride_k4) {
                w_values = w[w_row * stride_k4 + k4_base + stage_k4];
            }
            tile_w[next_w + stage_row * STRIDE + stage_k4] = w_values;
        }

        // Eight 128-bit shared loads feed 64 FMAs.
        for (var kk: u32 = 0u; kk < K4; kk = kk + 1u) {
            let a0 = tile_a[current + (thread_row + 0u) * STRIDE + kk];
            let a1 = tile_a[current + (thread_row + 1u) * STRIDE + kk];
            let a2 = tile_a[current + (thread_row + 2u) * STRIDE + kk];
            let a3 = tile_a[current + (thread_row + 3u) * STRIDE + kk];

            let w0 = tile_w[current_w + (thread_col + 0u) * STRIDE + kk];
            let w1 = tile_w[current_w + (thread_col + 16u) * STRIDE + kk];
            let w2 = tile_w[current_w + (thread_col + 32u) * STRIDE + kk];
            let w3 = tile_w[current_w + (thread_col + 48u) * STRIDE + kk];

            c0 += vec4<f32>(dot(a0, w0), dot(a0, w1), dot(a0, w2), dot(a0, w3));
            c1 += vec4<f32>(dot(a1, w0), dot(a1, w1), dot(a1, w2), dot(a1, w3));
            c2 += vec4<f32>(dot(a2, w0), dot(a2, w1), dot(a2, w2), dot(a2, w3));
            c3 += vec4<f32>(dot(a3, w0), dot(a3, w1), dot(a3, w2), dot(a3, w3));
        }

        workgroupBarrier();
    }

    // Transpose epilogue through the shared tiles: both are dead after the final
    // `workgroupBarrier()` and are reused from offset 0, so this costs no extra
    // shared memory (32 rows * 17 = 544 slots <= 640).
    //
    // The staging layout keeps the writer's view: slot `local.x` holds the four
    // columns `local.x + {0,16,32,48}`, so a writer can store its whole `vec4`
    // untouched. The reader then regroups slots to recover contiguous output
    // columns and writes coalesced `vec4`s — which is what the 16 scalar stores
    // this replaces could not do.
    const EP_STRIDE: u32 = 17u;
    let ep_r0 = local.y * 2u;
    tile_a[(ep_r0 + 0u) * EP_STRIDE + local.x] = c0;
    tile_a[(ep_r0 + 1u) * EP_STRIDE + local.x] = c1;
    tile_w[(ep_r0 + 0u) * EP_STRIDE + local.x] = c2;
    tile_w[(ep_r0 + 1u) * EP_STRIDE + local.x] = c3;
    workgroupBarrier();

    let col4 = lane % 16u;
    let y_grp = lane / 16u;
    let quad = col4 / 4u;
    let x_base = (col4 % 4u) * 4u;

    let out_col4 = col_base / 4u + col4;
    let out_row = row_base + y_grp * 4u;

    if (out_col4 < stride_c4) {
        let r0 = y_grp * 2u;
        let b0 = (r0 + 0u) * EP_STRIDE + x_base;
        let b1 = (r0 + 1u) * EP_STRIDE + x_base;

        if (out_row + 0u < m) {
            c[(out_row + 0u) * stride_c4 + out_col4] = pick_quad(
                tile_a[b0 + 0u], tile_a[b0 + 1u], tile_a[b0 + 2u], tile_a[b0 + 3u], quad
            );
        }
        if (out_row + 1u < m) {
            c[(out_row + 1u) * stride_c4 + out_col4] = pick_quad(
                tile_a[b1 + 0u], tile_a[b1 + 1u], tile_a[b1 + 2u], tile_a[b1 + 3u], quad
            );
        }
        if (out_row + 2u < m) {
            c[(out_row + 2u) * stride_c4 + out_col4] = pick_quad(
                tile_w[b0 + 0u], tile_w[b0 + 1u], tile_w[b0 + 2u], tile_w[b0 + 3u], quad
            );
        }
        if (out_row + 3u < m) {
            c[(out_row + 3u) * stride_c4 + out_col4] = pick_quad(
                tile_w[b1 + 0u], tile_w[b1 + 1u], tile_w[b1 + 2u], tile_w[b1 + 3u], quad
            );
        }
    }
}
