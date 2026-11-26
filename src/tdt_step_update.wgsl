// Reduces the 256 partial argmax entries and performs one NeMo TDT greedy transition
// entirely on the device. Dispatch: (1, 1, 1) workgroup of 256 threads.
//
// Every tie-break uses `val > best || (val == best && idx < best_idx)`, which is
// exactly what the CPU `argmax` does with its left-to-right `>` scan, so the device
// and host decoders pick the same token on equal logits.

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
    /// Per-frame symbol cap; 0 means unlimited.
    max_symbols: u32,
    _pad1: u32,
    _pad2: u32,
};

struct Partial {
    val: f32,
    idx: u32,
    _pad0: u32,
    _pad1: u32,
};

struct GpuToken {
    id: u32,
    frame: u32,
    duration: u32,
    duration_margin: f32,
    dur0: f32,
    dur1: f32,
    dur2: f32,
    dur3: f32,
    dur4: f32,
    score: f32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> state: TdtState;
@group(0) @binding(1) var<storage, read> partial: array<Partial>;
@group(0) @binding(2) var<storage, read> dur_logits: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_tokens: array<GpuToken>;

const BLANK: u32 = 8192u;

var<workgroup> red_val: array<f32, 256u>;
var<workgroup> red_idx: array<u32, 256u>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    red_val[lane] = partial[lane].val;
    red_idx[lane] = partial[lane].idx;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            let ov = red_val[lane + stride];
            let oi = red_idx[lane + stride];
            let mv = red_val[lane];
            let mi = red_idx[lane];
            if (ov > mv || (ov == mv && oi < mi)) {
                red_val[lane] = ov;
                red_idx[lane] = oi;
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u && state.done == 0u) {
        state.visited = state.visited + 1u;
        let token = red_idx[0];

        let d0 = dur_logits[0];
        let d1 = dur_logits[1];
        let d2 = dur_logits[2];
        let d3 = dur_logits[3];
        let d4 = dur_logits[4];

        var dur: u32 = 0u;
        var best_d = d0;
        if (d1 > best_d) { best_d = d1; dur = 1u; }
        if (d2 > best_d) { best_d = d2; dur = 2u; }
        if (d3 > best_d) { best_d = d3; dur = 3u; }
        if (d4 > best_d) { best_d = d4; dur = 4u; }

        var second_d = -3.402823e38f;
        if (dur != 0u && d0 > second_d) { second_d = d0; }
        if (dur != 1u && d1 > second_d) { second_d = d1; }
        if (dur != 2u && d2 > second_d) { second_d = d2; }
        if (dur != 3u && d3 > second_d) { second_d = d3; }
        if (dur != 4u && d4 > second_d) { second_d = d4; }
        let margin = best_d - second_d;

        let advance = max(dur, 1u);
        let cur_frame = state.frame;

        if (token == BLANK) {
            let next_frame = cur_frame + advance;
            state.frame = next_frame;
            state.symbols = 0u;
            state.need_predict = 0u;
            if (next_frame >= state.steps) {
                state.done = 1u;
            }
        } else {
            let index = state.token_count;
            if (index < state.max_tokens) {
                out_tokens[index] = GpuToken(
                    token, cur_frame, dur, margin, d0, d1, d2, d3, d4, red_val[0], 0u, 0u,
                );
                state.token_count = index + 1u;
            }
            state.last_token = token;
            state.need_predict = 1u;
            let next_symbols = state.symbols + 1u;
            if (dur == 0u) {
                if (state.max_symbols != 0u && next_symbols >= state.max_symbols) {
                    let next_frame = cur_frame + 1u;
                    state.frame = next_frame;
                    state.symbols = 0u;
                    if (next_frame >= state.steps) {
                        state.done = 1u;
                    }
                } else {
                    state.symbols = next_symbols;
                }
            } else {
                let next_frame = cur_frame + advance;
                state.frame = next_frame;
                state.symbols = 0u;
                if (next_frame >= state.steps) {
                    state.done = 1u;
                }
            }
        }
    }
}
