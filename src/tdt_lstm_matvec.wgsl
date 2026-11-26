// Warp-per-row LSTM gate pre-activation:
// `pre[r] = bias[r] + dot(W_ih[r], x) + dot(W_hh[r], h)`, r in [0, 2560).
// Dispatch: (320, 1, 1) workgroups of 256 threads (8 warps/WG -> 2560 warps).
//
// K = 640 = 160 vec4s = 32 lanes x 5 vec4s, so the 32 lanes of a warp read one
// contiguous 512 B chunk per statement: fully coalesced. The previous form gave each
// thread a whole row, which made adjacent lanes 2560 B apart.
//
// Barriers sit outside the `is_active` guard: WGSL requires uniform control flow for
// `workgroupBarrier`, so an inactive step does the (zero-cost) shared reduction and
// simply skips the global read and the store.

struct LstmMatvecParams {
    use_embed: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

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

@group(0) @binding(0) var<uniform> params: LstmMatvecParams;
@group(0) @binding(1) var<storage, read> state: TdtState;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> h: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> w_ih: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> w_hh: array<vec4<f32>>;
/// Pre-summed `bias_ih + bias_hh`, length 2560.
@group(0) @binding(6) var<storage, read> bias: array<f32>;
@group(0) @binding(7) var<storage, read_write> pre: array<f32>;

const K4: u32 = 160u;

var<workgroup> shared_red: array<f32, 256u>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let tid = local.x;
    let warp = tid / 32u;
    let lane = tid % 32u;
    let row = group.x * 8u + warp;

    let do_store = (state.done == 0u && state.need_predict != 0u && row < 2560u);
    var acc = 0.0f;
    if (do_store) {
        var x_base = 0u;
        if (params.use_embed != 0u) {
            x_base = state.last_token * K4;
        }
        let rb = row * K4;
        acc = dot(w_ih[rb + lane + 0u],   x[x_base + lane + 0u])
            + dot(w_hh[rb + lane + 0u],   h[lane + 0u])
            + dot(w_ih[rb + lane + 32u],  x[x_base + lane + 32u])
            + dot(w_hh[rb + lane + 32u],  h[lane + 32u])
            + dot(w_ih[rb + lane + 64u],  x[x_base + lane + 64u])
            + dot(w_hh[rb + lane + 64u],  h[lane + 64u])
            + dot(w_ih[rb + lane + 96u],  x[x_base + lane + 96u])
            + dot(w_hh[rb + lane + 96u],  h[lane + 96u])
            + dot(w_ih[rb + lane + 128u], x[x_base + lane + 128u])
            + dot(w_hh[rb + lane + 128u], h[lane + 128u]);
    }

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

    if (do_store && lane == 0u) {
        pre[row] = bias[row] + shared_red[tid] + shared_red[tid + 1u];
    }
}
