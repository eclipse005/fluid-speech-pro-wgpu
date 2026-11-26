// Attention context: `out[q, head, d] = sum_k w[head, q, k] * V[k, head, d]`.
//
// One workgroup per (head, query); the key walk covers the whole sequence.
//
// One workgroup per (head, query); thread `d` owns one channel of the head, so the
// 128 threads cover `HEAD_DIM` exactly and no reduction is needed.

struct ContextParams {
    total_steps: u32,
    max_steps: u32,
    head_dim: u32,
    /// Which head this dispatch handles; see `attn_scores_softmax.wgsl`.
    head_offset: u32,
    /// Where this head's score window starts, in floats.
    score_offset: u32,
    heads: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: ContextParams;
/// `[total_steps][max_steps]` for the head in `head_offset`, already softmaxed.
@group(0) @binding(1) var<storage, read> scores: array<f32>;
/// `[total_steps][3 * D_MODEL]`: Q | K | V.
@group(0) @binding(2) var<storage, read> qkv: array<f32>;
/// `[total_steps][D_MODEL]`, heads written side by side.
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let query = group.x;
    let head = params.head_offset;
    let head_dim = params.head_dim;
    let d = local.x;
    if (d >= head_dim) {
        return;
    }

    let d_model = (head_dim * params.heads);
    // Single sequence: the key walk covers every step.
    let clip_start = 0u;
    let clip_len = params.max_steps;

    let base = params.score_offset + query * params.max_steps;
    var sum = 0.0f;
    for (var k: u32 = 0u; k < clip_len; k = k + 1u) {
        let weight = scores[base + k];
        sum = sum + weight * qkv[(clip_start + k) * (3u * d_model) + 2u * d_model + head * head_dim + d];
    }
    out[query * d_model + head * head_dim + d] = sum;
}
