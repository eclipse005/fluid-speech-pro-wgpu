// SiLU, elementwise in place: `x = x / (1 + exp(-x))`.

@group(0) @binding(0) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let at = id.y * (grid.x * 256u) + id.x;
    if (at < arrayLength(&x)) {
        let value = x[at];
        x[at] = value / (1.0f + exp(-value));
    }
}
