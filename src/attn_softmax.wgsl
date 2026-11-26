// Turns the two score blocks into softmax weights for one head (or one query chunk).
//
// The score GEMMs produced `content[q_local][k] = (q + bias_u)·k` and
// `position[q_local][r] = (q + bias_v)·p[r]` — two GEMMs because the terms use
// different A operands. The score is their sum with the position read at relative
// offset `r = (T - 1) + k - (row_offset + q_local)`, so the relative-position shift
// stays an index rather than a materialised matrix.
//
// One workgroup per local query row. Intermediate scaled scores live in `weights`
// instead of workgroup storage so shared memory stays constant (1 KiB) regardless of
// utterance length — storing a full row in shared memory exceeds 48 KiB past ~12k
// frames (~16 minutes).

struct Params {
    steps: u32,
    pos_len: u32,
    content_n: u32,
    pos_n: u32,
    scale: f32,
    k_pad: u32,
    /// First global query row in this chunk (`0` when unchunked).
    row_offset: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
/// `[chunk_rows][content_n]`.
@group(0) @binding(1) var<storage, read> content: array<f32>;
/// `[chunk_rows][pos_n]`.
@group(0) @binding(2) var<storage, read> position: array<f32>;
/// `[chunk_rows][k_pad]`, the softmaxed weights (also scratch for scaled scores).
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;

const THREADS: u32 = 256u;

var<workgroup> reduce: array<f32, THREADS>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let query = group.x;
    let global_query = params.row_offset + query;
    let steps = params.steps;
    let lane = local.x;
    let content_base = query * params.content_n;
    // Relative row for key 0 is `(steps - 1) - global_query`.
    let pos_base = query * params.pos_n + (steps - 1u - global_query);
    let out_base = query * params.k_pad;

    var lane_max = -3.402823e38f;
    var index = lane;
    loop {
        if (index >= steps) {
            break;
        }
        let s = (content[content_base + index] + position[pos_base + index]) * params.scale;
        weights[out_base + index] = s;
        lane_max = max(lane_max, s);
        index = index + THREADS;
    }
    reduce[lane] = lane_max;
    workgroupBarrier();
    var stride = THREADS / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            reduce[lane] = max(reduce[lane], reduce[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let peak = reduce[0];
    workgroupBarrier();

    var lane_sum = 0.0f;
    index = lane;
    loop {
        if (index >= steps) {
            break;
        }
        let value = exp(weights[out_base + index] - peak);
        weights[out_base + index] = value;
        lane_sum = lane_sum + value;
        index = index + THREADS;
    }
    reduce[lane] = lane_sum;
    workgroupBarrier();
    stride = THREADS / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            reduce[lane] = reduce[lane] + reduce[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let total_weight = reduce[0];
    workgroupBarrier();

    // Past `steps` the tail is zero so the context GEMM can run with `k = k_pad`.
    index = lane;
    loop {
        if (index >= params.k_pad) {
            break;
        }
        if (index < steps) {
            weights[out_base + index] = weights[out_base + index] / total_weight;
        } else {
            weights[out_base + index] = 0.0f;
        }
        index = index + THREADS;
    }
}
