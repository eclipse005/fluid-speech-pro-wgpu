// Fast shared-memory variant of `attn_softmax.wgsl` for `steps <= 4096` (~5.5 min).
//
// Keeps the intermediate scaled scores and exponentials in a 16 KiB workgroup tile
// (`row[4096]`, allowing 3 workgroups/SM on 48 KiB GPUs), eliminating 4 of the 5
// global memory passes over `weights` while executing the exact same f32 arithmetic
// and reduction tree (bit-identical output). Utterances longer than 4096 frames
// dispatch `attn_softmax.wgsl` instead, which uses only 1 KiB of shared memory.

struct Params {
    steps: u32,
    pos_len: u32,
    content_n: u32,
    pos_n: u32,
    scale: f32,
    k_pad: u32,
    row_offset: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> content: array<f32>;
@group(0) @binding(2) var<storage, read> position: array<f32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;

const THREADS: u32 = 256u;
const MAX_SMEM_STEPS: u32 = 4096u;

var<workgroup> row: array<f32, MAX_SMEM_STEPS>;
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
    let pos_base = query * params.pos_n + (steps - 1u - global_query);
    let out_base = query * params.k_pad;

    var lane_max = -3.402823e38f;
    var index = lane;
    loop {
        if (index >= steps) {
            break;
        }
        let s = (content[content_base + index] + position[pos_base + index]) * params.scale;
        row[index] = s;
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
        let value = exp(row[index] - peak);
        row[index] = value;
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

    // Single global write pass: normalized weights for `0..steps`, zero tail to `k_pad`.
    index = lane;
    loop {
        if (index >= params.k_pad) {
            break;
        }
        if (index < steps) {
            weights[out_base + index] = row[index] / total_weight;
        } else {
            weights[out_base + index] = 0.0f;
        }
        index = index + THREADS;
    }
}
