// Coalesced attention scores + softmax, one workgroup per (head, query).
//
// One workgroup per (head, query); the key walk covers the whole sequence.
//
// Layout invariants: HEAD_DIM = 128 (32 vec4s), D_MODEL = 1024, width = 3*D_MODEL.
//
// Workgroup: 256 threads = 32 key slots x 8 threads per key; each thread owns four
// vec4 lanes (t + {0, 8, 16, 24}) of HEAD_DIM, giving fully coalesced 128-byte reads.
//
// Naga requires `workgroupBarrier` in uniform control flow, so the key-slot loop is
// bounded by `params.max_steps` (uniform) and only the *accesses* are guarded by the
// per-row `clip_len` (non-uniform). The three softmax loops contain no barrier and
// may use `clip_len` directly.

struct AttnParams {
    total_steps: u32,
    max_steps: u32,
    width: u32,
    head_dim: u32,
    scale: f32,
    heads: u32,
    pos_width: u32,
    /// Which head this dispatch handles. One head at a time keeps the score matrix
    /// inside a single binding's size, which is what caps a all-heads-at-once matrix
    /// on this driver.
    head_offset: u32,
    /// Where this head's `[total_steps][max_steps]` window starts, in floats.
    score_offset: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: AttnParams;
/// `[total_steps][3 * D_MODEL / 4]`: Q | K | V.
@group(0) @binding(1) var<storage, read> qkv: array<vec4<f32>>;
/// `[(2*max_steps-1)][D_MODEL / 4]`, projected position table for the whole batch.
@group(0) @binding(2) var<storage, read> pos: array<vec4<f32>>;
/// `[2 * D_MODEL / 4]`: `pos_bias_u` then `pos_bias_v`.
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
/// `[total_steps][max_steps]` for the head in `head_offset`.
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;

const THREADS: u32 = 256u;
const KEYS_PER_STEP: u32 = 32u;
/// Longest key window this pipeline supports, substituted by the host from the
/// encoder's allocation. It sizes the `row` shared array, so a value smaller than the
/// real frame count would write past the array and silently corrupt the scores.
const MAX_STEPS: u32 = {{MAX_STEPS}};
/// 32 key slots x 9 stride (padded by 1 to avoid a 4-way bank conflict on gather).
const RED_STRIDE: u32 = 9u;
const RED_LEN: u32 = KEYS_PER_STEP * RED_STRIDE;

var<workgroup> row: array<f32, MAX_STEPS>;
var<workgroup> warp_red: array<f32, RED_LEN>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let query = group.x;
    let head = params.head_offset;
    let max_steps = params.max_steps;
    let width4 = params.width / 4u;
    let head_dim4 = params.head_dim / 4u;
    let d_model4 = params.pos_width / 4u;
    let head_at4 = head * head_dim4;

    // Single sequence: every query sees the whole key range.
    let clip_start = 0u;
    let clip_len = max_steps;
    let q_local = query;

    let lane = local.x;
    let key_slot = lane / 8u;
    let t = lane % 8u;

    // Load Q and the relative biases once into registers (coalesced across 8 lanes).
    let q_base = query * width4 + head_at4;
    let bu_base = head_at4;
    let bv_base = d_model4 + head_at4;

    let q0 = qkv[q_base + t + 0u];
    let q1 = qkv[q_base + t + 8u];
    let q2 = qkv[q_base + t + 16u];
    let q3 = qkv[q_base + t + 24u];

    let qu0 = q0 + bias[bu_base + t + 0u];
    let qu1 = q1 + bias[bu_base + t + 8u];
    let qu2 = q2 + bias[bu_base + t + 16u];
    let qu3 = q3 + bias[bu_base + t + 24u];

    let qv0 = q0 + bias[bv_base + t + 0u];
    let qv1 = q1 + bias[bv_base + t + 8u];
    let qv2 = q2 + bias[bv_base + t + 16u];
    let qv3 = q3 + bias[bv_base + t + 24u];

    // 32 keys per iteration over the batch-wide key window; guards use `clip_len`.
    var k_base: u32 = 0u;
    loop {
        if (k_base >= max_steps) {
            break;
        }
        let k_local = k_base + key_slot;
        var part = 0.0f;
        if (k_local < clip_len) {
            let k = clip_start + k_local;
            let kb = k * width4 + d_model4 + head_at4;
            // Relative position depends only on `q_local - k_local`, so a table sized
            // by the batch maximum serves every clip unchanged.
            let pb = ((max_steps - 1u) + k_local - q_local) * d_model4 + head_at4;

            part = dot(qu0, qkv[kb + t + 0u]) + dot(qv0, pos[pb + t + 0u])
                 + dot(qu1, qkv[kb + t + 8u]) + dot(qv1, pos[pb + t + 8u])
                 + dot(qu2, qkv[kb + t + 16u]) + dot(qv2, pos[pb + t + 16u])
                 + dot(qu3, qkv[kb + t + 24u]) + dot(qv3, pos[pb + t + 24u]);
        }
        warp_red[key_slot * RED_STRIDE + t] = part;
        workgroupBarrier();

        if (lane < KEYS_PER_STEP) {
            let kk = k_base + lane;
            if (kk < clip_len) {
                let rb = lane * RED_STRIDE;
                let s = ((warp_red[rb + 0u] + warp_red[rb + 1u])
                       + (warp_red[rb + 2u] + warp_red[rb + 3u]))
                      + ((warp_red[rb + 4u] + warp_red[rb + 5u])
                       + (warp_red[rb + 6u] + warp_red[rb + 7u]));
                row[kk] = s * params.scale;
            }
        }
        workgroupBarrier();
        k_base = k_base + KEYS_PER_STEP;
    }

    // --- max (over this clip's keys only) ---
    var lane_max = -3.402823e38f;
    var index = lane;
    loop {
        if (index >= clip_len) {
            break;
        }
        lane_max = max(lane_max, row[index]);
        index = index + THREADS;
    }
    warp_red[lane] = lane_max;
    workgroupBarrier();

    var stride = THREADS / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            warp_red[lane] = max(warp_red[lane], warp_red[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let peak = warp_red[0];
    workgroupBarrier();

    // --- fused exp + local sum ---
    var lane_sum = 0.0f;
    index = lane;
    loop {
        if (index >= clip_len) {
            break;
        }
        let e = exp(row[index] - peak);
        row[index] = e;
        lane_sum = lane_sum + e;
        index = index + THREADS;
    }
    warp_red[lane] = lane_sum;
    workgroupBarrier();

    stride = THREADS / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            warp_red[lane] = warp_red[lane] + warp_red[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let total = warp_red[0];
    workgroupBarrier();

    // --- normalise ---
    let out_base = params.score_offset + query * max_steps;
    index = lane;
    loop {
        if (index >= clip_len) {
            break;
        }
        scores[out_base + index] = row[index] / total;
        index = index + THREADS;
    }
}
