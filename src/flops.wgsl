// Pure fp32 FMA throughput probe.
//
// Purpose: give the GEMM's measured rate a real ceiling to compare against. Every
// earlier number in this project was a GEMM-against-GEMM ratio, so "we are near the
// wall" was inference; this measures the wall.
//
// Design notes, each fixing a flaw in an earlier version of this probe:
//
// * `mode` selects the operation and is branched on **outside** the hot loop. An
//   earlier version tested it inside the loop, which added two compares and two
//   branches per unrolled round — 25% instruction overhead that under-reported the
//   rate.
// * `x`/`y` come from a uniform instead of literals, so the front end cannot fold
//   them into constants.
// * Eight accumulators give 32 independent scalar chains, far more than Pascal's
//   ~6-cycle FFMA latency needs, and the body is unrolled four times so loop
//   overhead stays small.
// * Registers: 8 accumulators (32) + x, y (8) ≈ 40 floats, under the 64-register
//   cliff, so occupancy is not the limiter being measured.
// * Three modes form a self-check:
//     mode 0 = `fma(a, x, y)`  -> 8 flops per `vec4`, one FFMA per lane
//     mode 1 = `a * x + y`     -> must match mode 0 if the driver fuses to FFMA
//     mode 2 = `a + y`         -> same instruction count, half the flops, so it must
//                                 take about the same time and report half the rate
//   Agreement across those three rules out both a multiply-add split and the loop
//   being optimised away.

struct Params {
    iters: u32,
    mode: u32,
    _pad0: u32,
    _pad1: u32,
    x: vec4<f32>,
    y: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let thread = group.x * 256u + local.x;
    let seed = f32(thread % 97u) * 1.0e-6 + 1.0;

    var a0 = vec4<f32>(seed, seed + 1.0, seed + 2.0, seed + 3.0) * 1.0e-9;
    var a1 = a0 + vec4<f32>(0.1, 0.2, 0.3, 0.4);
    var a2 = a0 + vec4<f32>(0.5, 0.6, 0.7, 0.8);
    var a3 = a0 + vec4<f32>(0.9, 1.0, 1.1, 1.2);
    var a4 = a0 + vec4<f32>(1.3, 1.4, 1.5, 1.6);
    var a5 = a0 + vec4<f32>(1.7, 1.8, 1.9, 2.0);
    var a6 = a0 + vec4<f32>(2.1, 2.2, 2.3, 2.4);
    var a7 = a0 + vec4<f32>(2.5, 2.6, 2.7, 2.8);

    let x = params.x;
    let y = params.y;

    if (params.mode == 2u) {
        for (var i: u32 = 0u; i < params.iters; i = i + 1u) {
            for (var round: u32 = 0u; round < 4u; round = round + 1u) {
                a0 = a0 + y;
                a1 = a1 + y;
                a2 = a2 + y;
                a3 = a3 + y;
                a4 = a4 + y;
                a5 = a5 + y;
                a6 = a6 + y;
                a7 = a7 + y;
            }
        }
    } else if (params.mode == 1u) {
        for (var i: u32 = 0u; i < params.iters; i = i + 1u) {
            for (var round: u32 = 0u; round < 4u; round = round + 1u) {
                a0 = a0 * x + y;
                a1 = a1 * x + y;
                a2 = a2 * x + y;
                a3 = a3 * x + y;
                a4 = a4 * x + y;
                a5 = a5 * x + y;
                a6 = a6 * x + y;
                a7 = a7 * x + y;
            }
        }
    } else {
        for (var i: u32 = 0u; i < params.iters; i = i + 1u) {
            for (var round: u32 = 0u; round < 4u; round = round + 1u) {
                a0 = fma(a0, x, y);
                a1 = fma(a1, x, y);
                a2 = fma(a2, x, y);
                a3 = fma(a3, x, y);
                a4 = fma(a4, x, y);
                a5 = fma(a5, x, y);
                a6 = fma(a6, x, y);
                a7 = fma(a7, x, y);
            }
        }
    }

    let total = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    let sum = total.x + total.y + total.z + total.w;
    // False at run time, opaque to the optimiser.
    if (sum == 1.2345e30) {
        out[thread % 1024u] = sum;
    }
}
