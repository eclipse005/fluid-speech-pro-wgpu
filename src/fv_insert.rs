//! The Fluid Speech `fv_insert` block, applied to the subsampled frames.
//!
//! `fluid_speech.py` attaches it as a forward hook on `encoder.pre_encode`, so it
//! runs *after* the convolution stack and *before* the first Conformer layer:
//!
//! ```text
//! y = x + up(GELU(mix(down(LayerNorm(x)))))
//! ```
//!
//! with `down`/`up` dense 1024 -> 1024, `mix` a depthwise Conv1d(15, padding 7)
//! and GELU the exact `erf` form (PyTorch's default, not the tanh approximation).
//! The hook is why a plain `pre_encode` reconstruction of this checkpoint does
//! not reproduce NeMo's output: the fixture carries the insert as well.

use anyhow::{Result, bail};

use crate::gemm::{PackW, gemm_into};
use crate::torch::Store;

const WIDTH: usize = 1024;
const KERNEL: usize = 15;
const PADDING: usize = KERNEL / 2;
const LAYER_NORM_EPS: f32 = 1e-5;
/// 1 / sqrt(2), for the erf-based GELU.
const INV_SQRT_2: f32 = 0.707_106_77;

pub struct FvInsert {
    ln_weight: Vec<f32>,
    ln_bias: Vec<f32>,
    down_weight: PackW,
    down_bias: Vec<f32>,
    mix_weight: Vec<f32>,
    mix_bias: Vec<f32>,
    up_weight: PackW,
    up_bias: Vec<f32>,
}

impl FvInsert {
    pub fn from_store(store: &Store) -> Result<Self> {
        let prefix = "encoder.fv_insert";
        let insert = Self {
            ln_weight: store.f32(&format!("{prefix}.ln.weight"))?,
            ln_bias: store.f32(&format!("{prefix}.ln.bias"))?,
            down_weight: PackW::from_vec(
                store.f32(&format!("{prefix}.down.weight"))?,
                WIDTH,
                WIDTH,
            ),
            down_bias: store.f32(&format!("{prefix}.down.bias"))?,
            mix_weight: store.f32(&format!("{prefix}.mix.weight"))?,
            mix_bias: store.f32(&format!("{prefix}.mix.bias"))?,
            up_weight: PackW::from_vec(store.f32(&format!("{prefix}.up.weight"))?, WIDTH, WIDTH),
            up_bias: store.f32(&format!("{prefix}.up.bias"))?,
        };
        insert.check()?;
        Ok(insert)
    }

    fn check(&self) -> Result<()> {
        if self.ln_weight.len() != WIDTH || self.ln_bias.len() != WIDTH {
            bail!("fv_insert layer norm is not {WIDTH}-wide");
        }
        for (name, packed) in [
            ("down.weight", &self.down_weight),
            ("up.weight", &self.up_weight),
        ] {
            if packed.rows() != WIDTH || packed.cols() != WIDTH {
                bail!(
                    "fv_insert {name} is [{}x{}], expected [{WIDTH}x{WIDTH}]",
                    packed.rows(),
                    packed.cols()
                );
            }
        }
        if self.down_bias.len() != WIDTH || self.up_bias.len() != WIDTH {
            bail!("fv_insert projections are not {WIDTH}-wide");
        }
        if self.mix_weight.len() != WIDTH * KERNEL || self.mix_bias.len() != WIDTH {
            bail!("fv_insert depthwise conv is not [{WIDTH}, 1, {KERNEL}]");
        }
        Ok(())
    }

    /// Adds the block's residual to `x`, a `[steps][1024]` time-major buffer.
    pub fn apply(&self, x: &mut [f32], steps: usize) -> Result<()> {
        if x.len() != steps * WIDTH {
            bail!(
                "expected {steps} steps of {WIDTH} values, found {}",
                x.len()
            );
        }

        // LayerNorm, then the two dense projections around the depthwise conv.
        let mut hidden = vec![0.0f32; steps * WIDTH];
        for step in 0..steps {
            let row = &x[step * WIDTH..(step + 1) * WIDTH];
            let mean = row.iter().sum::<f32>() / WIDTH as f32;
            let variance = row
                .iter()
                .map(|value| {
                    let delta = value - mean;
                    delta * delta
                })
                .sum::<f32>()
                / WIDTH as f32;
            let inv_std = 1.0 / (variance + LAYER_NORM_EPS).sqrt();
            let target = &mut hidden[step * WIDTH..(step + 1) * WIDTH];
            for index in 0..WIDTH {
                let normalized = (row[index] - mean) * inv_std;
                target[index] = normalized * self.ln_weight[index] + self.ln_bias[index];
            }
        }

        let mut projected = vec![0.0f32; steps * WIDTH];
        linear(
            &self.down_weight,
            &self.down_bias,
            &hidden,
            steps,
            &mut projected,
        );

        // Depthwise Conv1d over time: channel c mixes only its own channel.
        let mut mixed = vec![0.0f32; steps * WIDTH];
        for step in 0..steps {
            for channel in 0..WIDTH {
                let mut sum = self.mix_bias[channel];
                let weights = &self.mix_weight[channel * KERNEL..(channel + 1) * KERNEL];
                for (tap, weight) in weights.iter().enumerate() {
                    let source = step as isize + tap as isize - PADDING as isize;
                    if source < 0 || source >= steps as isize {
                        continue;
                    }
                    sum += weight * projected[source as usize * WIDTH + channel];
                }
                mixed[step * WIDTH + channel] = sum;
            }
        }

        for value in mixed.iter_mut() {
            *value = gelu(*value);
        }

        let mut restored = vec![0.0f32; steps * WIDTH];
        linear(&self.up_weight, &self.up_bias, &mixed, steps, &mut restored);
        for (slot, delta) in x.iter_mut().zip(restored) {
            *slot += delta;
        }
        Ok(())
    }
}

/// `y = x @ W^T + b` for a `[steps][WIDTH]` buffer.
fn linear(weight: &PackW, bias: &[f32], input: &[f32], steps: usize, out: &mut [f32]) {
    gemm_into(input, weight, out, steps);
    for step in 0..steps {
        let row = &mut out[step * WIDTH..(step + 1) * WIDTH];
        for (slot, offset) in row.iter_mut().zip(bias) {
            *slot += offset;
        }
    }
}

/// PyTorch's default (exact) GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + erf(value * INV_SQRT_2))
}

/// `erf` from libm: correctly rounded enough that the GELU matches PyTorch to
/// the last float32 digit or two, which a polynomial approximation does not.
fn erf(value: f32) -> f32 {
    libm::erff(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_reference_values() {
        // Reference values from a double-precision erf.
        let cases = [
            (0.0f32, 0.0f32),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (-1.0, -0.842_700_8),
            (2.0, 0.995_322_3),
        ];
        for (input, expected) in cases {
            let got = erf(input);
            assert!(
                (got - expected).abs() < 1e-6,
                "erf({input}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn gelu_matches_pytorch() {
        // torch.nn.functional.gelu for these inputs (exact erf form).
        let cases = [
            (0.0f32, 0.0f32),
            (1.0, 0.841_344_8),
            (-1.0, -0.158_655_3),
            (2.0, 1.954_499_9),
        ];
        for (input, expected) in cases {
            let got = gelu(input);
            assert!(
                (got - expected).abs() < 1e-6,
                "gelu({input}) = {got}, expected {expected}"
            );
        }
    }
}
