//! Stage-by-stage verification against NeMo fixtures.
//!
//! These commands exist to prove the port is numerically right (each stage against a
//! dumped tensor) or to emit the JSON that `compare_two.py` checks. They need the
//! crate internals, which is why they are a separate binary behind the `internals`
//! feature rather than part of `fsp`.

mod common;

use std::path::Path;

use anyhow::{Context, Result, bail};

use common::{compare_f32, compare_stage, compare_values, load_features, norm_pair};
use fluid_speech_pro::api::read_wav_mono_16k;
use fluid_speech_pro::{conformer, fv_insert, fv_insert_gpu, golden, gpu, gpu_encoder,
    mel, nemo, pre_encode, pre_encode_gpu, tdt, text};

fn mel_command(
    nemo_path: &Path,
    wav_path: &Path,
    expected: Option<&Path>,
    out: Option<&Path>,
) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let frontend = mel::MelFrontend::from_store(&bundle.store)?;

    let samples = read_wav_mono_16k(wav_path)?;

    let started = std::time::Instant::now();
    let features = frontend.compute(&samples)?;
    let elapsed = started.elapsed();
    let frames = frontend.frames(samples.len());
    println!(
        "samples={} seq_len={} frames={} mel=[{}x{}] in {:.3}s",
        samples.len(),
        frontend.seq_len(samples.len()),
        frames,
        mel::N_MELS,
        frames,
        elapsed.as_secs_f32()
    );

    if let Some(path) = out {
        let mut bytes = Vec::with_capacity(features.len() * 4);
        for value in &features {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, &bytes).with_context(|| format!("write {}", path.display()))?;
        println!("wrote {}", path.display());
    }

    if let Some(path) = expected {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        if bytes.len() != features.len() * 4 {
            bail!(
                "{} holds {} bytes, expected {}",
                path.display(),
                bytes.len(),
                features.len() * 4
            );
        }
        let mut worst = 0.0f32;
        let mut total = 0.0f64;
        for (chunk, actual) in bytes.as_chunks::<4>().0.iter().zip(&features) {
            let expected_value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let diff = (expected_value - actual).abs();
            worst = worst.max(diff);
            total += f64::from(diff);
        }
        let mean = total / features.len() as f64;
        println!("maxdiff={worst:.3e} meandiff={mean:.3e}");
    }
    Ok(())
}

/// Runs mel -> pre_encode and compares the result with a NeMo fixture.
fn pre_encode_command(
    nemo_path: &Path,
    wav_path: &Path,
    expected: Option<&Path>,
    out_path: Option<&Path>,
) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let frontend = mel::MelFrontend::from_store(&bundle.store)?;
    let pre = pre_encode::PreEncode::from_store(&bundle.store)?;

    // A `.f32` input is a time-major `[frames][128]` mel fixture, which isolates
    // this stage from the front end; a WAV goes through mel first.
    let (time_major, frames) = if wav_path.extension().is_some_and(|ext| ext == "f32") {
        let bytes =
            std::fs::read(wav_path).with_context(|| format!("read {}", wav_path.display()))?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let frames = values.len() / mel::N_MELS;
        (values, frames)
    } else {
        let samples = read_wav_mono_16k(wav_path)?;
        let decoded = frontend.compute(&samples)?;
        let frames = frontend.frames(samples.len());
        let bins = decoded.len() / frames;
        // NeMo's ConvSubsampling consumes (time, feat); the front end produces
        // (feat, time), so transpose on the way in.
        let mut time_major = vec![0.0f32; frames * bins];
        for bin in 0..bins {
            for frame in 0..frames {
                time_major[frame * bins + bin] = decoded[bin * frames + frame];
            }
        }
        (time_major, frames)
    };

    let insert = fv_insert::FvInsert::from_store(&bundle.store)?;

    let started = std::time::Instant::now();
    let (mut features, _valid) = pre.forward(&time_major, frames)?;
    let steps = features.len() / pre_encode::D_MODEL;
    // `fluid_speech.py` hooks this onto pre_encode, so NeMo's output carries it.
    // FLUID_SKIP_INSERT isolates the conv stack from the insert for debugging.
    if std::env::var_os("FLUID_SKIP_INSERT").is_none() {
        insert.apply(&mut features, steps)?;
    }
    let elapsed = started.elapsed();
    println!(
        "pre_encode=[{}x{}] in {:.3}s",
        steps,
        pre_encode::D_MODEL,
        elapsed.as_secs_f32()
    );

    if let Some(path) = out_path {
        let mut bytes = Vec::with_capacity(features.len() * 4);
        for value in &features {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, &bytes).with_context(|| format!("write {}", path.display()))?;
        println!("wrote {}", path.display());
    }

    if let Some(path) = expected {
        compare_f32(path, &features)?;
    }
    Ok(())
}

/// Runs mel -> pre_encode -> fv_insert -> 24 Conformer layers.
fn encoder_command(
    nemo_path: &Path,
    input: &Path,
    expected: Option<&Path>,
    out_path: Option<&Path>,
) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let frontend = mel::MelFrontend::from_store(&bundle.store)?;
    let pre = pre_encode::PreEncode::from_store(&bundle.store)?;
    let insert = fv_insert::FvInsert::from_store(&bundle.store)?;

    let (time_major, frames) = load_features(&frontend, input)?;

    let (mut features, _valid) = pre.forward(&time_major, frames)?;
    let steps = features.len() / pre_encode::D_MODEL;
    insert.apply(&mut features, steps)?;

    let encoder = conformer::Conformer::from_store(&bundle.store)?;
    let started = std::time::Instant::now();
    let out = encoder.forward(&features, steps)?;
    let elapsed = started.elapsed();
    println!(
        "encoder=[{}x{}] layers={} in {:.2}s",
        steps,
        conformer::D_MODEL,
        encoder.layer_count(),
        elapsed.as_secs_f32()
    );

    if let Some(path) = out_path {
        let mut bytes = Vec::with_capacity(out.len() * 4);
        for value in &out {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, &bytes).with_context(|| format!("write {}", path.display()))?;
        println!("wrote {}", path.display());
    }

    if let Some(path) = expected {
        // NeMo's encoder returns channel-major `[1024][steps]`.
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        // NeMo's encoder output is channel-major, so transpose before comparing.
        // The stride must come from the fixture itself: NeMo pads its encoder output
        // to the strided size while reporting a shorter length, so `steps` here can
        // be one smaller than the file's column count.
        let fixture_steps = values.len() / conformer::D_MODEL;
        let compared = steps.min(fixture_steps);
        if fixture_steps != steps {
            println!("comparing {compared} of {steps} steps (fixture has {fixture_steps})");
        }
        let reordered = {
            let mut transposed = vec![0.0f32; compared * conformer::D_MODEL];
            for channel in 0..conformer::D_MODEL {
                for step in 0..compared {
                    transposed[step * conformer::D_MODEL + channel] =
                        values[channel * fixture_steps + step];
                }
            }
            transposed
        };
        compare_values(&reordered, &out[..compared * conformer::D_MODEL])?;
    }

    // Per-layer comparison when the fixtures are next to each other.
    if let Some(directory) = expected.and_then(|path| path.parent()) {
        // Fixtures may be plain (`layer_1.f32`) or clip-prefixed (`de1_layer_1.f32`).
        let stem = expected
            .and_then(|path| path.file_stem())
            .and_then(|stem| stem.to_str())
            .and_then(|stem| stem.strip_suffix("_encoder_out"))
            .unwrap_or("")
            .to_string();
        for index in [0usize, 1, 2, 11, 23] {
            let candidates = [
                directory.join(format!("{stem}_layer_{index}.f32")),
                directory.join(format!("layer_{index}.f32")),
            ];
            let Some(path) = candidates.into_iter().find(|path| path.is_file()) else {
                continue;
            };
            let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            let values: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            if values.len() != steps * conformer::D_MODEL {
                continue;
            }
            let partial = encoder.forward_upto(&features, steps, index + 1)?;
            print!("layer_{index}: ");
            compare_values(&values, &partial)?;
        }
    }
    Ok(())
}

/// Dumps the tensor the Rust chain feeds into the conformer (mel -> pre_encode ->
/// fv_insert), so it can be diffed against NeMo's `encoder.pre_encode` output.
fn dump_encoder_input_command(nemo_path: &Path, mel: &Path, out_path: &Path) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let frontend = mel::MelFrontend::from_store(&bundle.store)?;
    let pre = pre_encode::PreEncode::from_store(&bundle.store)?;
    let insert = fv_insert::FvInsert::from_store(&bundle.store)?;

    let (time_major, frames) = load_features(&frontend, mel)?;
    let (mut features, _valid) = pre.forward(&time_major, frames)?;
    let steps = features.len() / pre_encode::D_MODEL;
    insert.apply(&mut features, steps)?;

    let mut bytes = Vec::with_capacity(features.len() * 4);
    for value in &features {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(out_path, &bytes).with_context(|| format!("write {}", out_path.display()))?;
    println!(
        "features [{}x{}] -> {}",
        steps,
        pre_encode::D_MODEL,
        out_path.display()
    );
    Ok(())
}

/// M0 acceptance: the wgpu GEMM must match the CPU one and hit throughput.
fn gpu_gemm_command() -> Result<()> {
    let gpu = gpu::GpuGemm::new()?;
    println!("adapter {}", gpu.adapter());
    println!(
        "{:<16} {:>7} {:>7} {:>7}  (pure GPU compute TFLOP/s, zero upload/readback)",
        "shape", "64x64", "v4", "auto"
    );
    for (m, n, k, reps) in [
        (13usize, 20usize, 32usize, 10usize),
        (189, 1024, 1024, 100),
        (189, 4096, 1024, 100),
        (189, 1024, 4096, 100),
        (377, 1024, 1024, 100),
        (2580, 128, 2592, 40),
        (2580, 1024, 1024, 40),
        (2580, 2048, 1024, 30),
        (2580, 3072, 1024, 25),
        (2580, 4096, 1024, 20),
        (2580, 1024, 4096, 20),
        (5159, 1024, 1024, 20),
        (2580, 2580, 128, 40),
        (2580, 5160, 128, 25),
    ] {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.0625)
            .collect();
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.03125)
            .collect();

        let (o64, d64) = gpu.gemm_bench_mode(&a, &w, m, n, k, reps, Some(false))?;
        let (ov4, dv4) = gpu.gemm_bench_mode(&a, &w, m, n, k, reps, Some(true))?;
        let (oa, da) = gpu.gemm_bench_mode(&a, &w, m, n, k, reps, None)?;

        let mut err = 0.0f32;
        for i in 0..o64.len() {
            err = err.max((o64[i] - ov4[i]).abs()).max((o64[i] - oa[i]).abs());
        }
        let flops = 2.0 * (m * n * k) as f64 / 1e12;
        println!(
            "{m:4}x{n:4}x{k:4}  {:7.2} {:7.2} {:7.2}  maxdiff={err:.1e}",
            flops / d64.as_secs_f64(),
            flops / dv4.as_secs_f64(),
            flops / da.as_secs_f64(),
        );
    }
    Ok(())
}

/// M1-A: the wgpu LayerNorm must match the CPU one on the same input.
fn gpu_layer_norm_command(nemo_path: &Path, input: &Path) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let weight = bundle
        .store
        .f32("encoder.layers.0.norm_feed_forward1.weight")?;
    let bias = bundle
        .store
        .f32("encoder.layers.0.norm_feed_forward1.bias")?;

    let bytes = std::fs::read(input).with_context(|| format!("read {}", input.display()))?;
    let (words, _remainder) = bytes.as_chunks::<4>();
    let x: Vec<f32> = words.iter().map(|word| f32::from_le_bytes(*word)).collect();
    let width = conformer::D_MODEL;
    let steps = x.len() / width;

    // Reference: the same arithmetic on the CPU, single threaded.
    let eps = 1e-5f32;
    let mut cpu = vec![0.0f32; x.len()];
    for step in 0..steps {
        let row = &x[step * width..(step + 1) * width];
        let mean = row.iter().sum::<f32>() / width as f32;
        let variance = row
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f32>()
            / width as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();
        for index in 0..width {
            cpu[step * width + index] = (row[index] - mean) * inv_std * weight[index] + bias[index];
        }
    }

    let gpu = gpu::GpuGemm::new()?;
    println!("adapter {}", gpu.adapter());
    let started = std::time::Instant::now();
    let out = gpu.layer_norm(&x, &weight, &bias, steps, width, eps)?;
    let elapsed = started.elapsed();

    let mut worst = 0.0f32;
    let mut total = 0.0f64;
    for (a, b) in cpu.iter().zip(&out) {
        let diff = (a - b).abs();
        worst = worst.max(diff);
        total += f64::from(diff);
    }
    println!(
        "layer_norm {steps}x{width}: maxdiff={worst:.3e} meandiff={:.3e} wall={:.3}ms",
        total / cpu.len() as f64,
        elapsed.as_secs_f64() * 1e3
    );

    // M1-A: LayerNorm -> linear1 -> SiLU -> linear2 must reproduce NeMo's ff1.
    let ff1_fixture = input.with_file_name("layer0_ff1.f32");
    if !ff1_fixture.is_file() {
        return Ok(());
    }
    let linear1 = bundle
        .store
        .f32("encoder.layers.0.feed_forward1.linear1.weight")?;
    let linear2 = bundle
        .store
        .f32("encoder.layers.0.feed_forward1.linear2.weight")?;

    let started = std::time::Instant::now();
    let mut hidden = gpu.gemm(&out, &linear1, steps, conformer::FFN, width)?;
    gpu.silu(&mut hidden)?;
    let ff1 = gpu.gemm(&hidden, &linear2, steps, width, conformer::FFN)?;
    let ff1_elapsed = started.elapsed();

    let bytes =
        std::fs::read(&ff1_fixture).with_context(|| format!("read {}", ff1_fixture.display()))?;
    let (words, _remainder) = bytes.as_chunks::<4>();
    let expected: Vec<f32> = words.iter().map(|word| f32::from_le_bytes(*word)).collect();
    let mut worst = 0.0f32;
    let mut total = 0.0f64;
    for (a, b) in expected.iter().zip(&ff1) {
        let diff = (a - b).abs();
        worst = worst.max(diff);
        total += f64::from(diff);
    }
    println!(
        "ff1 (gpu chain) {steps}x{width}: maxdiff={worst:.3e} meandiff={:.3e} wall={:.3}ms",
        total / expected.len() as f64,
        ff1_elapsed.as_secs_f64() * 1e3
    );
    Ok(())
}

/// M1-B: the whole ConformerConvolution module on the GPU vs the CPU path.
fn gpu_conv_command(nemo_path: &Path) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let convolution = conformer::Convolution::from_store(&bundle.store, "encoder.layers.0.conv")?;
    let pointwise1 = bundle
        .store
        .f32("encoder.layers.0.conv.pointwise_conv1.weight")?;
    let pointwise2 = bundle
        .store
        .f32("encoder.layers.0.conv.pointwise_conv2.weight")?;
    let depthwise = convolution.depthwise_weights().to_vec();
    let (scale, shift) = convolution.folded_batch_norm();

    let steps = 74usize;
    let width = conformer::D_MODEL;
    // Deterministic input of the same magnitude the layer actually sees.
    let input: Vec<f32> = (0..steps * width)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.25)
        .collect();

    let cpu = convolution.forward(&input, steps)?;

    let gpu = gpu::GpuGemm::new()?;
    println!("adapter {}", gpu.adapter());
    let started = std::time::Instant::now();
    let mut gates = gpu.gemm(&input, &pointwise1, steps, 2 * width, width)?;
    let mut activated = gpu.glu(&gates, steps, width)?;
    gates.clear();
    let mut convolved = gpu.dw_conv_bn_silu(
        &activated,
        &depthwise,
        &scale,
        &shift,
        steps,
        width,
        conformer::CONV_KERNEL,
        conformer::CONV_KERNEL / 2,
    )?;
    activated.clear();
    let out = gpu.gemm(&convolved, &pointwise2, steps, width, width)?;
    convolved.clear();
    let elapsed = started.elapsed();

    let mut worst = 0.0f32;
    let mut total = 0.0f64;
    let mut peak = 0.0f32;
    for (a, b) in cpu.iter().zip(&out) {
        let diff = (a - b).abs();
        worst = worst.max(diff);
        total += f64::from(diff);
        peak = peak.max(a.abs());
    }
    println!(
        "conv {steps}x{width}: maxdiff={worst:.3e} meandiff={:.3e} peak={peak:.1} wall={:.3}ms",
        total / cpu.len() as f64,
        elapsed.as_secs_f64() * 1e3
    );
    Ok(())
}

/// M1-C: the whole attention block on the GPU vs the CPU path.
fn gpu_attn_command(nemo_path: &Path) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let attention = conformer::Attention::from_store(&bundle.store, "encoder.layers.0.self_attn")?;

    // The layer feeds `norm_self_att(residual)` into attention, not the raw
    // encoder output, so the fixture must be that normalised tensor.
    let input_path = Path::new("../golden/encoder/layer0_attn_input.f32");
    let bytes =
        std::fs::read(input_path).with_context(|| format!("read {}", input_path.display()))?;
    let (words, _rest) = bytes.as_chunks::<4>();
    let input: Vec<f32> = words.iter().map(|word| f32::from_le_bytes(*word)).collect();

    let width = conformer::D_MODEL;
    let steps = input.len() / width;
    let pos_emb = conformer::positional_encoding(steps);
    let pos_len = 2 * steps - 1;

    let cpu = attention.forward(&input, steps, &pos_emb);

    let mut bias = attention.pos_bias_u().to_vec();
    bias.extend_from_slice(attention.pos_bias_v());

    let gpu = gpu::GpuGemm::new()?;
    println!("adapter {}", gpu.adapter());
    let started = std::time::Instant::now();
    let qkv = gpu.gemm(
        &input,
        attention.qkv_weight().row_major(),
        steps,
        3 * width,
        width,
    )?;
    let pos_proj = gpu.gemm(
        &pos_emb,
        attention.pos_weight().row_major(),
        pos_len,
        width,
        width,
    )?;
    let scores = gpu.attn_scores_softmax(
        &qkv,
        &pos_proj,
        &bias,
        steps,
        3 * width,
        conformer::HEAD_DIM,
        conformer::HEADS,
        width,
        attention.scale(),
    )?;
    let context = gpu.attn_context(
        &scores,
        &qkv,
        steps,
        3 * width,
        conformer::HEAD_DIM,
        conformer::HEADS,
    )?;
    let out = gpu.gemm(
        &context,
        attention.out_weight().row_major(),
        steps,
        width,
        width,
    )?;
    let elapsed = started.elapsed();

    let mut worst = 0.0f32;
    let mut total = 0.0f64;
    let mut peak = 0.0f32;
    for (a, b) in cpu.iter().zip(&out) {
        let diff = (a - b).abs();
        worst = worst.max(diff);
        total += f64::from(diff);
        peak = peak.max(a.abs());
    }
    println!(
        "attn vs cpu  {steps}x{width}: maxdiff={worst:.3e} meandiff={:.3e} peak={peak:.1} wall={:.3}ms",
        total / cpu.len() as f64,
        elapsed.as_secs_f64() * 1e3
    );

    // And against NeMo's own attention output, which also exercises the fused
    // QKV projection and the position table.
    let fixture = Path::new("../golden/encoder/layer0_attn.f32");
    if let Ok(bytes) = std::fs::read(fixture) {
        let (words, _rest) = bytes.as_chunks::<4>();
        let expected: Vec<f32> = words.iter().map(|word| f32::from_le_bytes(*word)).collect();
        let mut worst = 0.0f32;
        let mut total = 0.0f64;
        let mut peak = 0.0f32;
        for (a, b) in expected.iter().zip(&out) {
            let diff = (a - b).abs();
            worst = worst.max(diff);
            total += f64::from(diff);
            peak = peak.max(a.abs());
        }
        println!(
            "attn vs nemo {steps}x{width}: maxdiff={worst:.3e} meandiff={:.3e} peak={peak:.1}",
            total / expected.len() as f64
        );
    }
    Ok(())
}

/// M1 closed loop: the Layer-0 feed-forward half in one GPU submission.
fn gpu_ff1_command(nemo_path: &Path) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let store = &bundle.store;
    let feed_forward = conformer::FeedForward::from_store(store, "encoder.layers.0.feed_forward1")?;
    let norm_weight = store.f32("encoder.layers.0.norm_feed_forward1.weight")?;
    let norm_bias = store.f32("encoder.layers.0.norm_feed_forward1.bias")?;

    let bytes = std::fs::read("../golden/encoder/encoder_input.f32")
        .context("read golden/encoder/encoder_input.f32")?;
    let (words, _rest) = bytes.as_chunks::<4>();
    let input: Vec<f32> = words.iter().map(|word| f32::from_le_bytes(*word)).collect();
    let steps = input.len() / conformer::D_MODEL;

    let scratch = gpu_encoder::EncoderScratchGpu::new(steps)?;
    println!("adapter {}", scratch.adapter());
    scratch.write_residual(&input);

    let started = std::time::Instant::now();
    let block = scratch.feed_forward(&feed_forward, &norm_weight, &norm_bias)?;
    let elapsed = started.elapsed();

    // Reference chain through the already-verified per-operator entry points, so a
    // mismatch localises to either the recording path or the operator conventions.
    println!(
        "linear1 out={} in={} len={} | linear2 out={} in={} len={}",
        feed_forward.linear1().out_features(),
        feed_forward.linear1().in_features(),
        feed_forward.linear1().weight_row_major().len(),
        feed_forward.linear2().out_features(),
        feed_forward.linear2().in_features(),
        feed_forward.linear2().weight_row_major().len(),
    );
    let gpu = gpu::GpuGemm::new()?;
    let weight1 = feed_forward.linear1().weight_row_major();
    let weight2 = feed_forward.linear2().weight_row_major();
    let ln_ref = gpu.layer_norm(
        &input,
        &norm_weight,
        &norm_bias,
        steps,
        conformer::D_MODEL,
        conformer::LAYER_NORM_EPS,
    )?;
    let mut hidden_ref = gpu.gemm(&ln_ref, weight1, steps, conformer::FFN, conformer::D_MODEL)?;
    gpu.silu(&mut hidden_ref)?;
    let block_ref = gpu.gemm(
        &hidden_ref,
        weight2,
        steps,
        conformer::D_MODEL,
        conformer::FFN,
    )?;
    let mut chain_gap = 0.0f32;
    for (a, b) in block_ref.iter().zip(&block) {
        chain_gap = chain_gap.max((a - b).abs());
    }
    println!("chain_vs_perop maxdiff={chain_gap:.3e}");

    let fixture = std::fs::read("../golden/encoder/layer0_ff1.f32")
        .context("read golden/encoder/layer0_ff1.f32")?;
    let (fixture_words, _rest) = fixture.as_chunks::<4>();
    let expected: Vec<f32> = fixture_words
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();

    let mut worst = 0.0f32;
    let mut total = 0.0f64;
    let mut peak = 0.0f32;
    for (a, b) in expected.iter().zip(&block_ref) {
        worst = worst.max((a - b).abs());
        total += f64::from((a - b).abs());
        peak = peak.max(a.abs());
    }
    println!(
        "ff1 {steps}x1024 (single encoder): maxdiff={worst:.3e} meandiff={:.3e} peak={peak:.1} wall={:.3}ms",
        total / expected.len() as f64,
        elapsed.as_secs_f64() * 1e3
    );
    Ok(())
}

/// M1+M2 block check: a whole Conformer layer on the GPU, every stage compared.
fn gpu_block_command(nemo_path: &Path) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let store = &bundle.store;
    let prefix = "encoder.layers.0";
    let feed_forward1 =
        conformer::FeedForward::from_store(store, &format!("{prefix}.feed_forward1"))?;
    let feed_forward2 =
        conformer::FeedForward::from_store(store, &format!("{prefix}.feed_forward2"))?;
    let attention = conformer::Attention::from_store(store, &format!("{prefix}.self_attn"))?;
    let convolution = conformer::Convolution::from_store(store, &format!("{prefix}.conv"))?;

    let (ff1_w, ff1_b) = norm_pair(store, &format!("{prefix}.norm_feed_forward1"))?;
    let (attn_w, attn_b) = norm_pair(store, &format!("{prefix}.norm_self_att"))?;
    let (conv_w, conv_b) = norm_pair(store, &format!("{prefix}.norm_conv"))?;
    let (ff2_w, ff2_b) = norm_pair(store, &format!("{prefix}.norm_feed_forward2"))?;
    let (out_w, out_b) = norm_pair(store, &format!("{prefix}.norm_out"))?;

    let bytes = std::fs::read("../golden/encoder/encoder_input.f32")
        .context("read golden/encoder/encoder_input.f32")?;
    let (words, _rest) = bytes.as_chunks::<4>();
    let input: Vec<f32> = words.iter().map(|word| f32::from_le_bytes(*word)).collect();
    let steps = input.len() / conformer::D_MODEL;
    let pos_emb = conformer::positional_encoding(steps);

    let scratch = gpu_encoder::EncoderScratchGpu::new(steps)?;
    println!("adapter {}", scratch.adapter());
    scratch.write_residual(&input);

    let started = std::time::Instant::now();
    let ff1 = scratch.feed_forward(&feed_forward1, &ff1_w, &ff1_b)?;
    compare_stage("ff1", &ff1, "../golden/encoder/layer0_ff1.f32")?;
    scratch.accumulate(conformer::FC_FACTOR)?;

    let attended = scratch.attention(&attention, &attn_w, &attn_b, &pos_emb)?;
    compare_stage("attn", &attended, "../golden/encoder/layer0_attn.f32")?;
    scratch.accumulate(1.0)?;

    let convolved = scratch.convolution(&convolution, &conv_w, &conv_b)?;
    compare_stage("conv", &convolved, "../golden/encoder/layer0_conv.f32")?;

    // Isolate the convolution block the same way: per-operator reference first.
    let probe = gpu::GpuGemm::new()?;
    let residual_before = scratch.residual()?;
    let ln_ref = probe.layer_norm(
        &residual_before,
        &conv_w,
        &conv_b,
        steps,
        conformer::D_MODEL,
        conformer::LAYER_NORM_EPS,
    )?;
    let gates_ref = probe.gemm(
        &ln_ref,
        convolution.pointwise1().weight_row_major(),
        steps,
        2 * conformer::D_MODEL,
        conformer::D_MODEL,
    )?;
    let activated_ref = probe.glu(&gates_ref, steps, conformer::D_MODEL)?;
    // Scalar reference for the GLU alone, straight from the CPU definition:
    // `out[t, c] = gates[t, c] * sigmoid(gates[t, c + width])`.
    let width = conformer::D_MODEL;
    let mut glu_scalar = vec![0.0f32; steps * width];
    for step in 0..steps {
        let base = step * 2 * width;
        for channel in 0..width {
            let gate = gates_ref[base + channel];
            let value = gates_ref[base + width + channel];
            glu_scalar[step * width + channel] = gate / (1.0 + (-value).exp());
        }
    }
    let mut glu_gap = 0.0f32;
    let mut glu_peak = 0.0f32;
    for (a, b) in glu_scalar.iter().zip(&activated_ref) {
        glu_gap = glu_gap.max((a - b).abs());
        glu_peak = glu_peak.max(a.abs());
    }
    println!("  glu      maxdiff={glu_gap:.3e} peak={glu_peak:.1}");
    let conv_ref = probe.dw_conv_bn_silu(
        &activated_ref,
        convolution.depthwise_weights(),
        &convolution.folded_batch_norm().0,
        &convolution.folded_batch_norm().1,
        steps,
        conformer::D_MODEL,
        conformer::CONV_KERNEL,
        conformer::CONV_KERNEL / 2,
    )?;
    let out_ref = probe.gemm(
        &conv_ref,
        convolution.pointwise2().weight_row_major(),
        steps,
        conformer::D_MODEL,
        conformer::D_MODEL,
    )?;
    compare_stage("conv_ref", &out_ref, "../golden/encoder/layer0_conv.f32")?;

    // Fully scalar replay of the CPU definition, so a match here means the GPU
    // stages are at fault and a mismatch means the composition itself is wrong.
    let (bn_scale, bn_shift) = convolution.folded_batch_norm();
    let taps = convolution.depthwise_weights();
    let kernel = conformer::CONV_KERNEL;
    let padding = kernel / 2;
    let mut scalar_conv = vec![0.0f32; steps * conformer::D_MODEL];
    for step in 0..steps {
        for channel in 0..conformer::D_MODEL {
            let mut sum = 0.0f32;
            for tap in 0..kernel {
                let source = step as isize + tap as isize - padding as isize;
                if source < 0 || source >= steps as isize {
                    continue;
                }
                sum += taps[channel * kernel + tap]
                    * activated_ref[source as usize * conformer::D_MODEL + channel];
            }
            let value = sum * bn_scale[channel] + bn_shift[channel];
            scalar_conv[step * conformer::D_MODEL + channel] = value / (1.0 + (-value).exp());
        }
    }
    let w2 = convolution.pointwise2().weight_row_major();
    let mut scalar_out = vec![0.0f32; steps * conformer::D_MODEL];
    for step in 0..steps {
        for out_channel in 0..conformer::D_MODEL {
            let row = &w2[out_channel * conformer::D_MODEL..(out_channel + 1) * conformer::D_MODEL];
            let mut sum = 0.0f32;
            for in_channel in 0..conformer::D_MODEL {
                sum += row[in_channel] * scalar_conv[step * conformer::D_MODEL + in_channel];
            }
            scalar_out[step * conformer::D_MODEL + out_channel] = sum;
        }
    }
    compare_stage(
        "conv_scal",
        &scalar_out,
        "../golden/encoder/layer0_conv.f32",
    )?;
    let mut dw_gap = 0.0f32;
    for (a, b) in scalar_conv.iter().zip(&conv_ref) {
        dw_gap = dw_gap.max((a - b).abs());
    }
    println!("  dw_stage  scalar_vs_gpu maxdiff={dw_gap:.3e}");
    let mut gap = 0.0f32;
    for (a, b) in out_ref.iter().zip(&convolved) {
        gap = gap.max((a - b).abs());
    }
    println!("  conv chain_vs_perop maxdiff={gap:.3e}");
    scratch.accumulate(1.0)?;

    let ff2 = scratch.feed_forward(&feed_forward2, &ff2_w, &ff2_b)?;
    compare_stage("ff2", &ff2, "../golden/encoder/layer0_ff2.f32")?;
    scratch.accumulate(conformer::FC_FACTOR)?;

    // `block` now holds the ff2 projection, so `final_norm` must run before it is
    // reused; it writes the normalised residual into the same buffer by design.
    let out = scratch.final_norm(&out_w, &out_b)?;
    compare_stage("norm_out", &out, "../golden/encoder/layer0_out.f32")?;
    compare_stage("layer_0", &out, "../golden/encoder/layer_0.f32")?;
    let elapsed = started.elapsed();
    println!(
        "layer0 {steps}x{} on gpu: wall={:.2}ms",
        conformer::D_MODEL,
        elapsed.as_secs_f64() * 1e3
    );
    Ok(())
}

/// M2: all 24 encoder layers on the GPU, from the `fv_insert` output to
/// `encoder_out`. Reports weight residency separately from the steady-state encode.
///
/// Passing `chunk_rows` forces query-row attention chunking at that width and checks
/// it against both the CPU reference and the unchunked GPU path.
fn gpu_m2_command(nemo_path: &Path, chunk_rows: Option<usize>) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let store = &bundle.store;

    let load = |name: &str| -> Result<Vec<f32>> {
        let path = format!("../golden/encoder/{name}.f32");
        let bytes = std::fs::read(&path).with_context(|| format!("read {path}"))?;
        let (words, _rest) = bytes.as_chunks::<4>();
        Ok(words.iter().map(|word| f32::from_le_bytes(*word)).collect())
    };
    let input = load("de1_encoder_input")?;
    let expected = load("de1_encoder_out")?;
    let steps = input.len() / conformer::D_MODEL;
    // The attention scale is a model constant; take it from layer 0 rather than
    // hard-coding it a second time here.
    let scale = conformer::Attention::from_store(store, "encoder.layers.0.self_attn")?.scale();

    let started = std::time::Instant::now();
    let encoder = match chunk_rows {
        Some(rows) => gpu_encoder::EncoderGpu::with_capacity_and_chunk(
            std::sync::Arc::new(gpu::GpuGemm::new()?),
            steps,
            rows,
            store,
        )?,
        None => gpu_encoder::EncoderGpu::new(steps, store)?,
    };
    let build = started.elapsed();
    println!("adapter {}", encoder.adapter());

    // The first encode also pays the driver's pipeline compilation, so it is
    // reported as a cold start instead of being folded into the steady-state RTFx.
    let started = std::time::Instant::now();
    let out = encoder.encode(&input, scale)?;
    let cold = started.elapsed();

    let started = std::time::Instant::now();
    let warm = encoder.encode(&input, scale)?;
    let encode = started.elapsed();
    let mut drift = 0.0f32;
    for (a, b) in out.iter().zip(&warm) {
        drift = drift.max((a - b).abs());
    }
    println!("  warmup   maxdiff_vs_cold={drift:.3e}");

    if let Some(rows) = chunk_rows {
        let unchunked = gpu_encoder::EncoderGpu::with_gpu(encoder.gpu(), steps, store)?;
        let baseline = unchunked.encode(&input, scale)?;
        let mut chunk_gap = 0.0f32;
        for (a, b) in baseline.iter().zip(&out) {
            chunk_gap = chunk_gap.max((a - b).abs());
        }
        println!("  chunk={rows} maxdiff_vs_unchunked={chunk_gap:.3e}");
    }

    // CPU reference from the same store and the same input, so a mismatch localises
    // to the GPU chaining rather than to the fixture pairing.
    let cpu = conformer::Conformer::from_store(store)?.forward(&input, steps)?;
    compare_stage("cpu", &cpu, "../golden/encoder/de1_encoder_out.f32")?;
    compare_stage("encoder", &out, "../golden/encoder/de1_encoder_out.f32")?;
    let mut gap = 0.0f32;
    for (a, b) in cpu.iter().zip(&out) {
        gap = gap.max((a - b).abs());
    }
    println!("  gpu_vs_cpu maxdiff={gap:.3e}");

    // `FRAME_SECONDS` is the encoder's output hop, so `steps` frames cover this
    // much audio.
    let audio_seconds = steps as f64 * f64::from(tdt::FRAME_SECONDS);
    println!(
        "[build]   {:>8.3}s  weight loading + residency",
        build.as_secs_f64()
    );
    println!(
        "[cold]    {:>8.3}s  first encode (includes pipeline compilation)",
        cold.as_secs_f64()
    );
    println!(
        "[encoder] {:>8.3}s  steady state, {:.1}x over {:.2}s audio",
        encode.as_secs_f64(),
        audio_seconds / encode.as_secs_f64(),
        audio_seconds
    );
    let end_to_end = build + cold;
    println!(
        "end-to-end {:.3}s  ({:.1}x, build + first encode)",
        end_to_end.as_secs_f64(),
        audio_seconds / end_to_end.as_secs_f64()
    );
    let _ = &expected;
    Ok(())
}

/// Compares the GPU `pre_encode` against the CPU one on a wav or a mel fixture.
fn gpu_pre_encode_command(nemo_path: &Path, input: &Path) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let frontend = mel::MelFrontend::from_store(&bundle.store)?;
    let pre_cpu = pre_encode::PreEncode::from_store(&bundle.store)?;
    let (time_major, frames) = load_features(&frontend, input)?;

    let (cpu_out, cpu_valid) = pre_cpu.forward(&time_major, frames)?;
    let gpu = std::sync::Arc::new(gpu::GpuGemm::new()?);
    let mut pre_gpu = pre_encode_gpu::PreEncodeGpu::new(gpu, &bundle.store)?;
    let (cold_out, gpu_valid) = pre_gpu.forward(&time_major, frames)?;
    if cpu_valid != gpu_valid || cpu_out.len() != cold_out.len() {
        bail!(
            "shape mismatch: cpu valid={cpu_valid} len={}, gpu valid={gpu_valid} len={}",
            cpu_out.len(),
            cold_out.len()
        );
    }

    let reps = 20usize;
    let started = std::time::Instant::now();
    let mut gpu_out = Vec::new();
    for _ in 0..reps {
        gpu_out = pre_gpu.forward(&time_major, frames)?.0;
    }
    let wall = started.elapsed().as_secs_f64() / reps as f64;

    let mut worst = 0.0f32;
    let mut peak = 0.0f32;
    let mut total = 0.0f64;
    for (a, b) in cpu_out.iter().zip(&gpu_out) {
        let diff = (a - b).abs();
        worst = worst.max(diff);
        peak = peak.max(a.abs());
        total += f64::from(diff);
    }
    println!(
        "pre_encode_gpu frames={frames} valid={gpu_valid} maxdiff={worst:.3e} rel={:.3e} meandiff={:.3e} peak={peak:.2} wall={:.3}ms",
        worst / peak.max(1e-12),
        total / cpu_out.len() as f64,
        wall * 1e3
    );
    Ok(())
}

/// Diffs the resident device TDT decoder against the host decoder on one encoder
/// output, and times both.
///
/// The encoder runs on the CPU here on purpose: this isolates the decoder, so a
/// mismatch localises to the device state machine rather than the encoder chain.
fn tdt_compare_command(nemo_path: &Path, wav_path: &Path) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let frontend = mel::MelFrontend::from_store(&bundle.store)?;
    let pre = pre_encode::PreEncode::from_store(&bundle.store)?;
    let insert = fv_insert::FvInsert::from_store(&bundle.store)?;
    let encoder = conformer::Conformer::from_store(&bundle.store)?;

    let samples = read_wav_mono_16k(wav_path)?;
    let audio_seconds = samples.len() as f64 / mel::SAMPLE_RATE as f64;
    let mel_features = frontend.compute(&samples)?;
    let frames = frontend.frames(samples.len());
    let bins = mel_features.len() / frames;
    let mut time_major = vec![0.0f32; frames * bins];
    for bin in 0..bins {
        for frame in 0..frames {
            time_major[frame * bins + bin] = mel_features[bin * frames + frame];
        }
    }
    let (mut features, _valid) = pre.forward(&time_major, frames)?;
    let steps = features.len() / pre_encode::D_MODEL;
    insert.apply(&mut features, steps)?;
    let encoded = encoder.forward(&features, steps)?;

    let mut host = tdt::TdtDecoder::from_store(&bundle.store)?;
    let mut device = tdt::TdtDecoder::from_store(&bundle.store)?;
    device.enable_gpu()?;

    let host_tokens = host.decode(&encoded, steps)?;
    let device_tokens = device.decode(&encoded, steps)?;

    let same_len = host_tokens.len() == device_tokens.len();
    let mut ids_equal = same_len;
    let mut frames_equal = same_len;
    let mut durations_equal = same_len;
    let mut margin_gap = 0.0f32;
    let mut first_bad: Option<usize> = None;
    for (index, (a, b)) in host_tokens.iter().zip(&device_tokens).enumerate() {
        if a.id != b.id {
            ids_equal = false;
            first_bad.get_or_insert(index);
        }
        if a.frame != b.frame {
            frames_equal = false;
            first_bad.get_or_insert(index);
        }
        if a.duration != b.duration {
            durations_equal = false;
            first_bad.get_or_insert(index);
        }
        margin_gap = margin_gap.max((a.duration_margin - b.duration_margin).abs());
    }

    println!(
        "steps={steps} audio={audio_seconds:.2}s host_tokens={} device_tokens={}",
        host_tokens.len(),
        device_tokens.len()
    );
    println!(
        "ids_match={ids_equal} frames_match={frames_equal} durations_match={durations_equal} margin_maxgap={margin_gap:.3e}"
    );
    if let Some(index) = first_bad {
        let a = &host_tokens[index];
        let b = &device_tokens[index];
        println!(
            "first_diff[{index}] host=(id {} frame {} dur {}) device=(id {} frame {} dur {})",
            a.id, a.frame, a.duration, b.id, b.frame, b.duration
        );
    }

    let reps = 3usize;
    let started = std::time::Instant::now();
    for _ in 0..reps {
        let _ = host.decode(&encoded, steps)?;
    }
    let host_seconds = started.elapsed().as_secs_f64() / reps as f64;
    let started = std::time::Instant::now();
    for _ in 0..reps {
        let _ = device.decode(&encoded, steps)?;
    }
    let device_seconds = started.elapsed().as_secs_f64() / reps as f64;
    println!(
        "decode host={:.2}ms ({:.1}x) device={:.2}ms ({:.1}x) speedup={:.2}x",
        host_seconds * 1e3,
        audio_seconds / host_seconds,
        device_seconds * 1e3,
        audio_seconds / device_seconds,
        host_seconds / device_seconds
    );
    Ok(())
}

/// Compares the on-device `fv_insert` against the CPU one on real features.
fn gpu_fv_insert_command(nemo_path: &Path, input: &Path) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let frontend = mel::MelFrontend::from_store(&bundle.store)?;
    let pre = pre_encode::PreEncode::from_store(&bundle.store)?;
    let insert = fv_insert::FvInsert::from_store(&bundle.store)?;
    let (time_major, frames) = load_features(&frontend, input)?;
    let (features, _valid) = pre.forward(&time_major, frames)?;
    let steps = features.len() / pre_encode::D_MODEL;

    let mut cpu_features = features.clone();
    insert.apply(&mut cpu_features, steps)?;

    let gpu = std::sync::Arc::new(gpu::GpuGemm::new()?);
    let on_device = fv_insert_gpu::FvInsertGpu::new(gpu, steps, &bundle.store)?;
    on_device.write_input(&features);
    on_device.apply(on_device.output())?;
    let gpu_features = on_device.read_output()?;

    let mut worst = 0.0f32;
    let mut peak = 0.0f32;
    let mut total = 0.0f64;
    for (a, b) in cpu_features.iter().zip(&gpu_features) {
        let diff = (a - b).abs();
        worst = worst.max(diff);
        peak = peak.max(a.abs());
        total += f64::from(diff);
    }
    println!(
        "fv_insert steps={steps} maxdiff={worst:.3e} rel={:.3e} meandiff={:.3e} peak={peak:.2}",
        worst / peak.max(1e-12),
        total / cpu_features.len() as f64
    );

    // Steady-state cost of the block alone, GPU vs CPU.
    let reps = 20usize;
    let started = std::time::Instant::now();
    for _ in 0..reps {
        let mut host = features.clone();
        insert.apply(&mut host, steps)?;
    }
    let cpu_seconds = started.elapsed().as_secs_f64() / reps as f64;
    let started = std::time::Instant::now();
    for _ in 0..reps {
        on_device.apply(on_device.output())?;
    }
    let gpu_seconds = started.elapsed().as_secs_f64() / reps as f64;
    println!(
        "fv_insert cpu={:.3}ms gpu={:.3}ms speedup={:.2}x",
        cpu_seconds * 1e3,
        gpu_seconds * 1e3,
        cpu_seconds / gpu_seconds
    );
    Ok(())
}

/// Same batch pipeline as `batch`, with the encoder on the GPU.
///
/// Scratch is sized for one clip at a time, so each entry rebuilds the encoder —
/// slow (weight residency is ~2.7s) but correct. Generalising the scratch to a
/// max-steps allocation belongs with the performance work, not here: this command
/// exists to prove the GPU path reaches the same tokens as the CPU one.
fn gpu_batch_command(nemo_path: &Path, list_path: &Path, out_path: &Path) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let frontend = mel::MelFrontend::from_store(&bundle.store)?;
    let mut decoder = tdt::TdtDecoder::from_store(&bundle.store)?;
    let tokenizer = text::Tokenizer::from_model(&bundle.tokenizer_model, &bundle.vocab)?;
    let scale =
        conformer::Attention::from_store(&bundle.store, "encoder.layers.0.self_attn")?.scale();
    // One device for the whole batch: the front end and the encoder share a queue so
    // the subsampled features never round-trip through the host.
    let shared_gpu = std::sync::Arc::new(gpu::GpuGemm::new()?);
    let mut pre_gpu =
        pre_encode_gpu::PreEncodeGpu::new(std::sync::Arc::clone(&shared_gpu), &bundle.store)?;
    decoder.enable_gpu_with(std::sync::Arc::clone(&shared_gpu))?;

    let list_text = std::fs::read_to_string(list_path)
        .with_context(|| format!("read {}", list_path.display()))?;
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(&list_text).context("parse batch list")?;

    let mut results = Vec::with_capacity(entries.len());
    for entry in entries {
        let id = entry
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let wav = entry
            .get("wav")
            .and_then(|v| v.as_str())
            .context("batch entry has no wav")?;
        let (time_major, frames) = load_features(&frontend, Path::new(wav))?;
        // Same resident chain the benchmark uses: pre_encode -> fv_insert -> encoder
        // -> decoder, so the alignment check covers the production path.
        let (feat_buf, steps) = pre_gpu.forward_resident(&time_major, frames)?;

        let started = std::time::Instant::now();
        let insert_gpu =
            fv_insert_gpu::FvInsertGpu::new(std::sync::Arc::clone(&shared_gpu), steps, &bundle.store)?;
        insert_gpu.submit_apply(&feat_buf);
        let encoder = gpu_encoder::EncoderGpu::with_gpu(
            std::sync::Arc::clone(&shared_gpu),
            steps,
            &bundle.store,
        )?;
        let resident = encoder.encode_resident(insert_gpu.output(), scale);
        let tokens = decoder.decode_gpu_buffer(resident, steps)?;
        let encode_elapsed = started.elapsed();
        let ids: Vec<u32> = tokens.iter().map(|token| token.id).collect();
        let words = text::words(&tokenizer, &tokens);
        let segments = text::segments(&words);
        results.push(serde_json::json!({
            "id": id,
            "steps": steps,
            "text": tokenizer.decode(&ids),
            "token_ids": ids,
            "token_frames": tokens.iter().map(|t| t.frame).collect::<Vec<_>>(),
            "token_durations": tokens.iter().map(|t| t.duration).collect::<Vec<_>>(),
            "token_margins": tokens.iter().map(|t| t.duration_margin).collect::<Vec<_>>(),
            "words": words.iter().map(|w| serde_json::json!({
                "word": w.word,
                "start_offset": w.start_offset,
                "end_offset": w.end_offset,
                "start": w.start,
                "end": w.end,
            })).collect::<Vec<_>>(),
            "segments": segments.iter().map(|s| serde_json::json!({
                "segment": s.segment,
                "start_offset": s.start_offset,
                "end_offset": s.end_offset,
                "start": s.start,
                "end": s.end,
            })).collect::<Vec<_>>(),
        }));
        println!(
            "done {id} steps={steps} gpu_build+encode={:.3}s",
            encode_elapsed.as_secs_f64()
        );
    }

    let document = serde_json::json!({ "clips": results });
    std::fs::write(out_path, serde_json::to_vec(&document)?)
        .with_context(|| format!("write {}", out_path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("inspect-nemo") => {
            let path = args.next().context("usage: inspect-nemo <nemo>")?;
            nemo::inspect(Path::new(&path))
        }
        Some("verify-nemo") => {
            let usage = "usage: verify-nemo <nemo> <index.json>";
            let model = args.next().context(usage)?;
            let index = args.next().context(usage)?;
            nemo::verify_against_index(Path::new(&model), Path::new(&index))
        }
        Some("inspect-golden") => {
            let path = args.next().context("usage: inspect-golden <golden.json>")?;
            golden::inspect(Path::new(&path))
        }
        Some("mel") => {
            let usage = "usage: mel <nemo> <audio.wav> [expected.f32] [out.f32]";
            let model = args.next().context(usage)?;
            let wav = args.next().context(usage)?;
            let expected = args.next();
            let out = args.next();
            mel_command(
                Path::new(&model),
                Path::new(&wav),
                expected.as_deref().map(Path::new),
                out.as_deref().map(Path::new),
            )
        }
        Some("pre-encode") => {
            let usage = "usage: pre-encode <nemo> <wav|mel.f32> [expected.f32] [out.f32]";
            let model = args.next().context(usage)?;
            let input = args.next().context(usage)?;
            let expected = args.next();
            let out = args.next();
            pre_encode_command(
                Path::new(&model),
                Path::new(&input),
                expected.as_deref().map(Path::new),
                out.as_deref().map(Path::new),
            )
        }
        Some("encoder") => {
            let usage = "usage: encoder <nemo> <wav|mel.f32> [expected.f32] [out.f32]";
            let model = args.next().context(usage)?;
            let input = args.next().context(usage)?;
            let expected = args.next();
            let out = args.next();
            encoder_command(
                Path::new(&model),
                Path::new(&input),
                expected.as_deref().map(Path::new),
                out.as_deref().map(Path::new),
            )
        }
        Some("dump-encoder-input") => {
            let usage = "usage: dump-encoder-input <nemo> <mel.f32> <out.f32>";
            let model = args.next().context(usage)?;
            let mel = args.next().context(usage)?;
            let out = args.next().context(usage)?;
            dump_encoder_input_command(Path::new(&model), Path::new(&mel), Path::new(&out))
        }
        Some("gpu-gemm") => gpu_gemm_command(),
        Some("gpu-layernorm") => {
            let usage = "usage: gpu-layernorm <nemo> <input.f32>";
            let model = args.next().context(usage)?;
            let input = args.next().context(usage)?;
            gpu_layer_norm_command(Path::new(&model), Path::new(&input))
        }
        Some("gpu-conv") => {
            let model = args.next().context("usage: gpu-conv <nemo>")?;
            gpu_conv_command(Path::new(&model))
        }
        Some("gpu-attn") => {
            let model = args.next().context("usage: gpu-attn <nemo>")?;
            gpu_attn_command(Path::new(&model))
        }
        Some("gpu-ff1") => {
            let model = args.next().context("usage: gpu-ff1 <nemo>")?;
            gpu_ff1_command(Path::new(&model))
        }
        Some("gpu-block") => {
            let model = args.next().context("usage: gpu-block <nemo>")?;
            gpu_block_command(Path::new(&model))
        }
        Some("gpu-m2") => {
            let model = args.next().context("usage: gpu-m2 <nemo> [chunk_rows]")?;
            let chunk_rows = match args.next() {
                Some(raw) => Some(raw.parse().context("chunk_rows must be an integer")?),
                None => None,
            };
            gpu_m2_command(Path::new(&model), chunk_rows)
        }
        Some("gpu-pre-encode") => {
            let usage = "usage: gpu-pre-encode <nemo> <wav|mel.f32>";
            let model = args.next().context(usage)?;
            let input = args.next().context(usage)?;
            gpu_pre_encode_command(Path::new(&model), Path::new(&input))
        }
        Some("gpu-fv-insert") => {
            let usage = "usage: gpu-fv-insert <nemo> <wav|mel.f32>";
            let model = args.next().context(usage)?;
            let input = args.next().context(usage)?;
            gpu_fv_insert_command(Path::new(&model), Path::new(&input))
        }
        Some("tdt-compare") => {
            let usage = "usage: tdt-compare <nemo> <wav>";
            let model = args.next().context(usage)?;
            let wav = args.next().context(usage)?;
            tdt_compare_command(Path::new(&model), Path::new(&wav))
        }
        Some("gpu-batch") => {
            let usage = "usage: gpu-batch <nemo> <list.json> <out.json>";
            let model = args.next().context(usage)?;
            let list = args.next().context(usage)?;
            let out = args.next().context(usage)?;
            gpu_batch_command(Path::new(&model), Path::new(&list), Path::new(&out))
        }
        _ => {
            eprintln!("fsp-verify — stage-by-stage comparison against NeMo fixtures
");
            eprintln!("  inspect-nemo <nemo> | verify-nemo <nemo> <index.json> | inspect-golden <json>");
            eprintln!("  mel / pre-encode / encoder / dump-encoder-input");
            eprintln!("  gpu-gemm | gpu-layernorm | gpu-conv | gpu-attn | gpu-ff1 | gpu-block | gpu-m2");
            eprintln!("  gpu-pre-encode | gpu-fv-insert | tdt-compare | gpu-batch");
            Ok(())
        }
    }
}
