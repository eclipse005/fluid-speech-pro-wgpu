// Fused Flash-style local relative-position attention (`rel_pos_local_attn`).
//
// Mirrors NeMo's `RelPositionMultiHeadAttentionLongformer.forward` with zero score-matrix
// VRAM: each 32-lane warp handles one `(query, head)` pair, each lane owns one `vec4<f32>`
// (`HEAD_DIM = 128 = 32 * 4`), and online softmax streams directly across the local window
// `c in [w - left_ctx, w + right_ctx]` (`O(T * (left + right + 1))` compute, O(1) memory).
//
// Dispatch: `grid_2d(ceil(steps * heads / 4))` workgroups of 128 threads (4 warps/WG).

struct LocalAttnParams {
    steps: u32,
    d_model: u32,
    head_dim: u32,
    pos_len: u32,
    left_ctx: u32,
    right_ctx: u32,
    window_w: u32,
    heads: u32,
    scale: f32,
    grid_x: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: LocalAttnParams;
/// `[steps][3 * D_MODEL / 4]` fused QKV buffer (`Q | K | V`).
@group(0) @binding(1) var<storage, read> qkv: array<vec4<f32>>;
/// `[pos_len][D_MODEL / 4]` projected relative positional encodings (`linear_pos(pos_emb)`).
@group(0) @binding(2) var<storage, read> pos_proj: array<vec4<f32>>;
/// `[2 * D_MODEL / 4]` concatenated `pos_bias_u` then `pos_bias_v`.
@group(0) @binding(3) var<storage, read> attn_bias: array<vec4<f32>>;
/// `[steps][D_MODEL / 4]` combined multi-head context output.
@group(0) @binding(4) var<storage, read_write> out_buf: array<vec4<f32>>;

const WARP_SIZE: u32 = 32u;
const WARPS_PER_WG: u32 = 4u;

var<workgroup> shared_red: array<f32, 128u>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let warp = tid / WARP_SIZE;
    let lane = tid % WARP_SIZE;
    let warp_base = warp * WARP_SIZE;

    let wg_linear = wid.y * params.grid_x + wid.x;
    let task_id = wg_linear * WARPS_PER_WG + warp;
    let total_tasks = params.steps * params.heads;
    let valid_task = task_id < total_tasks;

    let query = select(0u, task_id / params.heads, valid_task);
    let head = select(0u, task_id % params.heads, valid_task);

    let d_model4 = params.d_model / 4u;
    let qkv_stride4 = 3u * d_model4;
    let h_off4 = head * WARP_SIZE;

    var qu = vec4<f32>(0.0f);
    var qv = vec4<f32>(0.0f);
    if (valid_task) {
        let q_vec = qkv[query * qkv_stride4 + h_off4 + lane];
        qu = q_vec + attn_bias[h_off4 + lane];
        qv = q_vec + attn_bias[d_model4 + h_off4 + lane];
    }

    let start_pos = params.window_w - params.left_ctx;
    let end_pos = params.window_w + params.right_ctx;
    let right_bd_start = 2u * params.window_w - params.right_ctx;

    var m_val: f32 = -3.402823e38f;
    var l_val: f32 = 0.0f;
    var acc_o = vec4<f32>(0.0f);

    // Uniform loop bounds across all threads in the workgroup so every barrier stays uniform.
    for (var col: u32 = start_pos; col <= end_pos; col = col + 1u) {
        let key_i = i32(query) + i32(col) - i32(params.window_w);
        let in_bounds = valid_task && key_i >= 0 && u32(key_i) < params.steps;

        var dot_part: f32 = 0.0f;
        var v_vec = vec4<f32>(0.0f);
        if (in_bounds) {
            let key_u = u32(key_i);
            let k_vec = qkv[key_u * qkv_stride4 + d_model4 + h_off4 + lane];
            v_vec = qkv[key_u * qkv_stride4 + 2u * d_model4 + h_off4 + lane];
            dot_part = dot(qu, k_vec);

            // Relative-position dot term matching NeMo `multi_head_attention.py:470-475`.
            if (col < params.left_ctx) {
                let p_vec = pos_proj[col * d_model4 + h_off4 + lane];
                dot_part = dot_part + dot(qv, p_vec);
            } else if (col >= right_bd_start) {
                let p_idx = params.left_ctx + (col - right_bd_start);
                let p_vec = pos_proj[p_idx * d_model4 + h_off4 + lane];
                dot_part = dot_part + dot(qv, p_vec);
            }
        }

        shared_red[tid] = dot_part;
        workgroupBarrier();
        if (lane < 16u) { shared_red[tid] = shared_red[tid] + shared_red[tid + 16u]; }
        workgroupBarrier();
        if (lane < 8u) { shared_red[tid] = shared_red[tid] + shared_red[tid + 8u]; }
        workgroupBarrier();
        if (lane < 4u) { shared_red[tid] = shared_red[tid] + shared_red[tid + 4u]; }
        workgroupBarrier();
        if (lane < 2u) { shared_red[tid] = shared_red[tid] + shared_red[tid + 2u]; }
        workgroupBarrier();
        if (lane == 0u) { shared_red[tid] = shared_red[tid] + shared_red[tid + 1u]; }
        workgroupBarrier();

        let score = shared_red[warp_base] * params.scale;
        if (in_bounds) {
            let new_m = max(m_val, score);
            let alpha = exp(m_val - new_m);
            let pw = exp(score - new_m);
            l_val = l_val * alpha + pw;
            acc_o = acc_o * alpha + v_vec * pw;
            m_val = new_m;
        }
        // Barrier before next iteration overwrites `shared_red[tid]`.
        workgroupBarrier();
    }

    if (valid_task) {
        let inv_l = select(0.0f, 1.0f / l_val, l_val > 0.0f);
        out_buf[query * d_model4 + h_off4 + lane] = acc_o * inv_l;
    }
}
