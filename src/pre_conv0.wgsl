// pre_encode conv.0: Conv2d(1 -> 256, k=3, s=2, p=1) + bias + ReLU.
// Input:  [in_time, in_freq] (mel, 1 channel)
// Weight: [9, 256] (transposed once on load from [256, 9] for coalesced reads)
// Output: [out_time, out_freq, 256] (NHWC), where out_time = valid1.

struct Conv2dParams {
    in_time: u32,
    in_freq: u32,
    out_time: u32,
    out_freq: u32,
};

@group(0) @binding(0) var<uniform> params: Conv2dParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let f_out = group.x;
    let t_out = group.y;
    let c = local.x;
    if (f_out >= params.out_freq || t_out >= params.out_time) {
        return;
    }

    let base_t = i32(t_out * 2u) - 1;
    let base_f = i32(f_out * 2u) - 1;
    let in_t = i32(params.in_time);
    let in_f = i32(params.in_freq);

    var sum = 0.0f;
    for (var row: i32 = 0; row < 3; row = row + 1) {
        let st = base_t + row;
        if (st < 0 || st >= in_t) {
            continue;
        }
        let row_base = u32(st) * params.in_freq;
        for (var col: i32 = 0; col < 3; col = col + 1) {
            let sf = base_f + col;
            if (sf < 0 || sf >= in_f) {
                continue;
            }
            let tap = u32(row * 3 + col);
            sum = sum + weight[tap * 256u + c] * input[row_base + u32(sf)];
        }
    }

    let val = bias[c] + sum;
    out[(t_out * params.out_freq + f_out) * 256u + c] = max(val, 0.0f);
}
