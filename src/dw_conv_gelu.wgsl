// `fv_insert`'s depthwise Conv1d(15, padding 7) fused with the exact GELU.
//
// The block is `y = x + up(GELU(mix(down(LayerNorm(x)))))`, so this kernel covers the
// middle: one thread per (step, channel), mixing only its own channel across time.
//
// WGSL has no `erf`, so it is spelled out with the Abramowitz & Stegun 7.1.26
// rational approximation (|error| < 1.5e-7, i.e. about one f32 ulp), which keeps the
// exact-erf GELU rather than swapping in the tanh approximation that PyTorch's
// `gelu` does *not* use by default.

struct ConvParams {
    steps: u32,
    channels: u32,
    kernel: u32,
    padding: u32,
};

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const INV_SQRT_2: f32 = 0.70710678;

/// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7.
fn erf_approx(z: f32) -> f32 {
    let a = abs(z);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592;
    let y = 1.0 - poly * t * exp(-a * a);
    return select(-y, y, z >= 0.0);
}

/// PyTorch's default GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
fn gelu_exact(v: f32) -> f32 {
    return 0.5 * v * (1.0 + erf_approx(v * INV_SQRT_2));
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let at = id.y * (grid.x * 256u) + id.x;
    let total = params.steps * params.channels;
    if (at >= total) {
        return;
    }
    let step = at / params.channels;
    let channel = at % params.channels;

    var sum = bias[channel];
    let w_base = channel * params.kernel;
    for (var tap: u32 = 0u; tap < params.kernel; tap = tap + 1u) {
        let src = i32(step) + i32(tap) - i32(params.padding);
        if (src >= 0 && src < i32(params.steps)) {
            sum = sum + weights[w_base + tap] * x[u32(src) * params.channels + channel];
        }
    }
    out[at] = gelu_exact(sum);
}
