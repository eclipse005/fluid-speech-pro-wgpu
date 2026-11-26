// Elementwise LSTM cell update for DEC_HIDDEN = 640 units.
// Gate order: i (0..640), f (640..1280), g (1280..1920), o (1920..2560).
// Dispatch: (3, 1, 1) workgroups of 256 threads.

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
@group(0) @binding(1) var<storage, read> pre: array<f32>;
@group(0) @binding(2) var<storage, read_write> c_state: array<f32>;
@group(0) @binding(3) var<storage, read_write> h_state: array<f32>;

const HIDDEN: u32 = 640u;

fn gate_sigmoid(v: f32) -> f32 {
    return 1.0f / (1.0f + exp(-v));
}

fn gate_tanh(v: f32) -> f32 {
    return tanh(clamp(v, -20.0f, 20.0f));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let unit = id.x;
    if (unit >= HIDDEN || state.done != 0u || state.need_predict == 0u) {
        return;
    }
    let i_gate = gate_sigmoid(pre[unit]);
    let f_gate = gate_sigmoid(pre[HIDDEN + unit]);
    let g_gate = gate_tanh(pre[2u * HIDDEN + unit]);
    let o_gate = gate_sigmoid(pre[3u * HIDDEN + unit]);

    let next_c = f_gate * c_state[unit] + i_gate * g_gate;
    c_state[unit] = next_c;
    h_state[unit] = o_gate * gate_tanh(next_c);
}
