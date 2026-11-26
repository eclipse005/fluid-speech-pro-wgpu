// Warp-per-row predictor projection: `out[r] = combined_bias[r] + dot(W_pred[r], h1)`,
// where `combined_bias = pred_bias + enc_bias` is pre-summed once on load so the
// encoder projection can stay a plain bias-free GEMM.
// Dispatch: (80, 1, 1) workgroups of 256 threads (640 warps).

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

@group(0) @binding(0) var<storage, read> state: TdtState;
@group(0) @binding(1) var<storage, read> h1: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

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

    let do_store = (state.done == 0u && state.need_predict != 0u && row < 640u);
    var acc = 0.0f;
    if (do_store) {
        let rb = row * K4;
        acc = dot(w[rb + lane + 0u],   h1[lane + 0u])
            + dot(w[rb + lane + 32u],  h1[lane + 32u])
            + dot(w[rb + lane + 64u],  h1[lane + 64u])
            + dot(w[rb + lane + 96u],  h1[lane + 96u])
            + dot(w[rb + lane + 128u], h1[lane + 128u]);
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
        out[row] = bias[row] + shared_red[tid] + shared_red[tid + 1u];
    }
}
