// Fused ReLU(enc_proj[frame] + pred_proj) + coalesced warp-per-row joint argmax over
// 8193 vocabulary rows, plus the five duration logits computed by group 0.
//
// Dispatch: (256, 1, 1) workgroups of 256 threads = 2048 warps, each covering
// ceil(8193 / 2048) = 5 rows. Every warp reads one contiguous 512 B chunk per
// statement: fully coalesced. The previous form gave each thread one row, which put
// adjacent lanes 2560 B apart — an 8x waste of every cache line fetched.

struct TdtState {
    frame: u32,
    symbols: u32,
    token_count: u32,
    last_token: u32,
    need_predict: u32,
    done: u32,
    steps: u32,
    max_tokens: u32,
    visited: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct Partial {
    val: f32,
    idx: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> state: TdtState;
/// `[8198][160]` row-major joint output weights as `vec4`s.
@group(0) @binding(1) var<storage, read> w: array<vec4<f32>>;
/// `[8198]` joint output bias.
@group(0) @binding(2) var<storage, read> bias: array<f32>;
/// `[steps][160]` precomputed `encoder_out @ W_enc^T`.
@group(0) @binding(3) var<storage, read> enc_proj: array<vec4<f32>>;
/// `[160]` `predictor @ W_pred^T + pred_bias + enc_bias`.
@group(0) @binding(4) var<storage, read> pred_proj: array<vec4<f32>>;
/// `[256]` per-workgroup best `(val, idx)`.
@group(0) @binding(5) var<storage, read_write> partial: array<Partial>;
/// `[5]` raw duration logits for the current step.
@group(0) @binding(6) var<storage, read_write> dur_logits: array<f32>;

const VOCAB: u32 = 8193u;
const K4: u32 = 160u;
const GROUPS: u32 = 256u;
const WARPS: u32 = 8u;
const TOTAL_WARPS: u32 = 2048u;
const ITERS: u32 = 5u;

var<workgroup> shared_x: array<vec4<f32>, K4>;
var<workgroup> shared_red: array<f32, 256u>;
var<workgroup> warp_val: array<f32, WARPS>;
var<workgroup> warp_idx: array<u32, WARPS>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let tid = local.x;
    let warp = tid / 32u;
    let lane = tid % 32u;
    let is_active = (state.done == 0u);

    // Fuse `relu(enc_proj[frame] + pred_proj)` straight into shared memory.
    if (is_active && tid < K4) {
        let sum = enc_proj[state.frame * K4 + tid] + pred_proj[tid];
        shared_x[tid] = max(sum, vec4<f32>(0.0f));
    }
    workgroupBarrier();

    var best_val = -3.402823e38f;
    var best_idx = 0u;
    let base_warp = group.x * WARPS + warp;

    for (var it: u32 = 0u; it < ITERS; it = it + 1u) {
        let row = base_warp + it * TOTAL_WARPS;
        var acc = 0.0f;
        if (is_active && row < VOCAB) {
            let rb = row * K4;
            acc = dot(w[rb + lane + 0u], shared_x[lane + 0u])
                + dot(w[rb + lane + 32u], shared_x[lane + 32u])
                + dot(w[rb + lane + 64u], shared_x[lane + 64u])
                + dot(w[rb + lane + 96u], shared_x[lane + 96u])
                + dot(w[rb + lane + 128u], shared_x[lane + 128u]);
        }
        // Barriers stay in uniform control flow: WGSL forbids them inside a branch
        // that depends on a storage buffer, so an inactive step does the (free)
        // shared reduction and skips only the global read and the store.
        shared_red[tid] = acc;
        workgroupBarrier();
        if (lane < 16u) { shared_red[tid] = shared_red[tid] + shared_red[tid + 16u]; }
        workgroupBarrier();
        if (lane < 8u) { shared_red[tid] = shared_red[tid] + shared_red[tid + 8u]; }
        workgroupBarrier();
        if (lane < 4u) { shared_red[tid] = shared_red[tid] + shared_red[tid + 4u]; }
        workgroupBarrier();
        if (lane < 2u) { shared_red[tid] = shared_red[tid] + shared_red[tid + 2u]; }
        workgroupBarrier();

        if (is_active && lane == 0u && row < VOCAB) {
            let val = bias[row] + shared_red[tid] + shared_red[tid + 1u];
            if (val > best_val || (val == best_val && row < best_idx)) {
                best_val = val;
                best_idx = row;
            }
        }
        workgroupBarrier();
    }

    if (lane == 0u) {
        warp_val[warp] = best_val;
        warp_idx[warp] = best_idx;
    }
    workgroupBarrier();

    if (tid == 0u && is_active) {
        var gv = warp_val[0];
        var gi = warp_idx[0];
        for (var wi: u32 = 1u; wi < WARPS; wi = wi + 1u) {
            let cv = warp_val[wi];
            let ci = warp_idx[wi];
            if (cv > gv || (cv == gv && ci < gi)) {
                gv = cv;
                gi = ci;
            }
        }
        partial[group.x] = Partial(gv, gi, 0u, 0u);
    }

    // Group 0 piggybacks the five duration rows onto warps 0..4 while `shared_x` is hot.
    if (group.x == 0u) {
        var d_acc = 0.0f;
        if (is_active && warp < 5u) {
            let d_row = VOCAB + warp;
            let rb = d_row * K4;
            d_acc = dot(w[rb + lane + 0u], shared_x[lane + 0u])
                  + dot(w[rb + lane + 32u], shared_x[lane + 32u])
                  + dot(w[rb + lane + 64u], shared_x[lane + 64u])
                  + dot(w[rb + lane + 96u], shared_x[lane + 96u])
                  + dot(w[rb + lane + 128u], shared_x[lane + 128u]);
        }
        shared_red[tid] = d_acc;
        workgroupBarrier();
        if (lane < 16u) { shared_red[tid] = shared_red[tid] + shared_red[tid + 16u]; }
        workgroupBarrier();
        if (lane < 8u) { shared_red[tid] = shared_red[tid] + shared_red[tid + 8u]; }
        workgroupBarrier();
        if (lane < 4u) { shared_red[tid] = shared_red[tid] + shared_red[tid + 4u]; }
        workgroupBarrier();
        if (lane < 2u) { shared_red[tid] = shared_red[tid] + shared_red[tid + 2u]; }
        workgroupBarrier();

        if (is_active && lane == 0u && warp < 5u) {
            dur_logits[warp] = bias[VOCAB + warp] + shared_red[tid] + shared_red[tid + 1u];
        }
    }
}
