//! Performance measurement: steady-state RTFx and per-shape throughput.
//!
//! Nothing here is needed to transcribe audio; it exists to answer "where does the
//! time go" and "is the GPU actually faster", which is why it lives in its own
//! binary behind the `internals` feature.

use std::path::Path;

use anyhow::{Context, Result, bail};

use fluid_speech_pro::api::read_wav_mono_16k;
use fluid_speech_pro::{conformer, flops_probe, fv_insert, fv_insert_gpu, gpu, gpu_encoder,
    mel, nemo, pre_encode, pre_encode_gpu, tdt, text};

/// Times the pipeline against the contract in `notes/perf_contract.md`:
/// end-to-end starts before the weights are touched, and every phase prints.
fn bench_command(
    nemo_path: &Path,
    wav_path: &Path,
    process_start: std::time::Instant,
) -> Result<()> {
    let load_started = std::time::Instant::now();
    let bundle = nemo::open(nemo_path)?;
    let frontend = mel::MelFrontend::from_store(&bundle.store)?;
    let load_elapsed = load_started.elapsed();

    let samples = read_wav_mono_16k(wav_path)?;
    let audio_seconds = samples.len() as f64 / mel::SAMPLE_RATE as f64;

    let mel_started = std::time::Instant::now();
    let features = frontend.compute(&samples)?;
    let mel_elapsed = mel_started.elapsed();

    let end_to_end = process_start.elapsed();
    let infer = load_elapsed + mel_elapsed;
    println!(
        "[load] {:.3}s tensors={}",
        load_elapsed.as_secs_f64(),
        bundle.store.tensors.len()
    );
    println!(
        "[mel] {:.3}s [{}x{}]",
        mel_elapsed.as_secs_f64(),
        mel::N_MELS,
        frontend.frames(samples.len())
    );
    println!("[audio] {audio_seconds:.3}s");
    println!(
        "[end-to-end] {:.3}s RTFx={:.2}x (load+mel only; encoder/tdt pending)",
        end_to_end.as_secs_f64(),
        audio_seconds / end_to_end.as_secs_f64()
    );
    println!(
        "[infer] {:.3}s RTFx={:.2}x (load+mel only; encoder/tdt pending)",
        infer.as_secs_f64(),
        audio_seconds / infer.as_secs_f64()
    );
    println!("[mel-values] {}", features.len());
    Ok(())
}

/// Full-pipeline steady-state benchmark for one clip and one backend.
///
/// The timed window covers everything a caller pays per utterance:
/// mel -> pre_encode -> fv_insert -> encoder -> TDT decode -> detokenise.
/// `--backend gpu` builds the encoder once (weight residency is a cold-start cost)
/// and reuses it across repetitions.
fn bench_steady_command(
    nemo_path: &Path,
    wav_path: &Path,
    backend: &str,
    reps: usize,
) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let frontend = mel::MelFrontend::from_store(&bundle.store)?;
    let pre = pre_encode::PreEncode::from_store(&bundle.store)?;
    let insert = fv_insert::FvInsert::from_store(&bundle.store)?;
    let mut decoder = tdt::TdtDecoder::from_store(&bundle.store)?;
    let tokenizer = text::Tokenizer::from_model(&bundle.tokenizer_model, &bundle.vocab)?;
    let scale =
        conformer::Attention::from_store(&bundle.store, "encoder.layers.0.self_attn")?.scale();
    let samples = read_wav_mono_16k(wav_path)?;
    let audio_seconds = samples.len() as f64 / mel::SAMPLE_RATE as f64;

    // One dry run establishes the frame count, which the GPU scratch needs.
    let mel_features = frontend.compute(&samples)?;
    let frames = frontend.frames(samples.len());
    let bins = mel_features.len() / frames;
    let mut time_major = vec![0.0f32; frames * bins];
    for bin in 0..bins {
        for frame in 0..frames {
            time_major[frame * bins + bin] = mel_features[bin * frames + frame];
        }
    }
    let (mut probe, _) = pre.forward(&time_major, frames)?;
    let steps = probe.len() / pre_encode::D_MODEL;
    insert.apply(&mut probe, steps)?;

    let started = std::time::Instant::now();
    // Only build the CPU Conformer when actually benchmarking CPU; loading 2.3 GB of
    // host weights when running `--backend gpu` padded `[build]` by ~2 s for nothing.
    let encoder = if !backend.starts_with("gpu") {
        Some(conformer::Conformer::from_store(&bundle.store)?)
    } else {
        None
    };
    // `gpu` runs everything on the device (front end, encoder, decoder);
    // `gpu-cputdt` keeps the decoder on the host for A/B comparison.
    // The resident decoder is the default on GPU backends; `gpu-cputdt` keeps the host
    // one for A/B. `tdt-compare` shows both produce identical tokens.
    let tdt_on_gpu = backend.starts_with("gpu") && !backend.contains("cputdt");
    let t_gpu = std::time::Instant::now();
    let gpu_device = if backend.starts_with("gpu") {
        Some(std::sync::Arc::new(gpu::GpuGemm::new()?))
    } else {
        None
    };
    let d_gpu = t_gpu.elapsed();
    let t_enc = std::time::Instant::now();
    let gpu_encoder = match &gpu_device {
        Some(gpu) => Some(gpu_encoder::EncoderGpu::with_gpu(
            std::sync::Arc::clone(gpu),
            steps,
            &bundle.store,
        )?),
        None => None,
    };
    let d_enc = t_enc.elapsed();
    // On the GPU backends the front end runs on the device too, sharing one queue
    // with the encoder so nothing round-trips through the host in between.
    let t_pre = std::time::Instant::now();
    let mut pre_gpu = match &gpu_encoder {
        Some(gpu) => Some(pre_encode_gpu::PreEncodeGpu::new(gpu.gpu(), &bundle.store)?),
        None => None,
    };
    let d_pre = t_pre.elapsed();
    let t_fv = std::time::Instant::now();
    let fv_gpu = match &gpu_encoder {
        Some(gpu) => Some(fv_insert_gpu::FvInsertGpu::new(gpu.gpu(), steps, &bundle.store)?),
        None => None,
    };
    let d_fv = t_fv.elapsed();
    let t_tdt = std::time::Instant::now();
    if tdt_on_gpu && let Some(gpu) = &gpu_encoder {
        decoder.enable_gpu_with(gpu.gpu())?;
    }
    let d_tdt = t_tdt.elapsed();
    if let Some(gpu) = &gpu_encoder {
        gpu.gpu().save_pipeline_cache();
    }
    let build = started.elapsed();
    if gpu_encoder.is_some() {
        println!(
            "[build-detail] gpu_new={:.3}s  encoder={:.3}s  pre_gpu={:.3}s  fv_gpu={:.3}s  tdt_gpu={:.3}s",
            d_gpu.as_secs_f64(),
            d_enc.as_secs_f64(),
            d_pre.as_secs_f64(),
            d_fv.as_secs_f64(),
            d_tdt.as_secs_f64()
        );
    }

    // `pre_gpu` is passed in rather than captured: the phase-timing block below also
    // needs it mutably, and a captured borrow would conflict.
    let run_once = |decoder: &mut tdt::TdtDecoder,
                    pre_gpu: &mut Option<pre_encode_gpu::PreEncodeGpu>,
                    fv_gpu: &Option<fv_insert_gpu::FvInsertGpu>|
     -> Result<Vec<u32>> {
        let mel_features = frontend.compute(&samples)?;
        let mut time_major = vec![0.0f32; frames * bins];
        for bin in 0..bins {
            for frame in 0..frames {
                time_major[frame * bins + bin] = mel_features[bin * frames + frame];
            }
        }
        // Resident path: pre_encode -> fv_insert -> encoder -> decoder with no host
        // round trip between the stages.
        if let (Some(gpu), Some(fv)) = (&gpu_encoder, fv_gpu)
            && tdt_on_gpu
        {
            let (feat_buf, valid) = pre_gpu
                .as_mut()
                .expect("a GPU backend has a device front end")
                .forward_resident(&time_major, frames)?;
            fv.submit_apply(&feat_buf);
            let resident = gpu.encode_resident(fv.output(), scale);
            let tokens = decoder.decode_gpu_buffer(resident, valid)?;
            let ids: Vec<u32> = tokens.iter().map(|token| token.id).collect();
            let _ = tokenizer.decode(&ids);
            return Ok(ids);
        }
        let (mut features, _valid) = match pre_gpu {
            Some(gpu) => gpu.forward(&time_major, frames)?,
            None => pre.forward(&time_major, frames)?,
        };
        let steps = features.len() / pre_encode::D_MODEL;
        insert.apply(&mut features, steps)?;
        let tokens = match &gpu_encoder {
            Some(gpu) if tdt_on_gpu => {
                // The encoder output stays in VRAM: the decoder reads it directly.
                let resident = gpu.encode_device(&features, scale);
                decoder.decode_gpu_buffer(resident, steps)?
            }
            Some(gpu) => {
                let encoded = gpu.encode(&features, scale)?;
                decoder.decode(&encoded, steps)?
            }
            None => {
                let encoded = encoder
                    .as_ref()
                    .expect("CPU encoder built for CPU backend")
                    .forward(&features, steps)?;
                decoder.decode(&encoded, steps)?
            }
        };
        let ids: Vec<u32> = tokens.iter().map(|token| token.id).collect();
        // Detokenisation is part of the per-utterance cost, so it stays inside the
        // timed window.
        let _ = tokenizer.decode(&ids);
        Ok(ids)
    };

    // Two warm-ups: the first pays shader/pipeline compilation, the second
    // confirms the steady state matches it.
    let first = run_once(&mut decoder, &mut pre_gpu, &fv_gpu)?;
    let second = run_once(&mut decoder, &mut pre_gpu, &fv_gpu)?;
    if first != second {
        bail!("backend {backend} is not deterministic across runs");
    }

    // Phase split for one run: without this the optimisation target is guesswork.
    {
        let phase = std::time::Instant::now();
        let mel_features = frontend.compute(&samples)?;
        let mel_time = phase.elapsed();
        let mut tm = vec![0.0f32; frames * bins];
        for bin in 0..bins {
            for frame in 0..frames {
                tm[frame * bins + bin] = mel_features[bin * frames + frame];
            }
        }
        let (enc_time, insert_time, pre_time, phase_tokens, tdt_time);
        if let (Some(gpu), Some(fv)) = (&gpu_encoder, &fv_gpu)
            && tdt_on_gpu
        {
            let mark = std::time::Instant::now();
            let (feat_buf, valid) = pre_gpu
                .as_mut()
                .expect("a GPU backend has a device front end")
                .forward_resident(&tm, frames)?;
            gpu.sync()?;
            pre_time = mark.elapsed();

            let mark = std::time::Instant::now();
            fv.submit_apply(&feat_buf);
            gpu.sync()?;
            insert_time = mark.elapsed();

            let mark = std::time::Instant::now();
            let resident = gpu.encode_resident(fv.output(), scale);
            gpu.sync()?;
            enc_time = mark.elapsed();

            let mark = std::time::Instant::now();
            phase_tokens = decoder.decode_gpu_buffer(resident, valid)?;
            tdt_time = mark.elapsed();
        } else {
            let mark = std::time::Instant::now();
            let (mut host_features, _valid) = match &mut pre_gpu {
                Some(gpu) => gpu.forward(&tm, frames)?,
                None => pre.forward(&tm, frames)?,
            };
            pre_time = mark.elapsed();
            let steps_here = host_features.len() / pre_encode::D_MODEL;

            let mark = std::time::Instant::now();
            insert.apply(&mut host_features, steps_here)?;
            insert_time = mark.elapsed();

            let mark = std::time::Instant::now();
            let encoded_host = match &gpu_encoder {
                Some(gpu) => gpu.encode(&host_features, scale)?,
                None => encoder
                    .as_ref()
                    .expect("CPU encoder built for CPU backend")
                    .forward(&host_features, steps_here)?,
            };
            enc_time = mark.elapsed();

            let mark = std::time::Instant::now();
            phase_tokens = decoder.decode(&encoded_host, steps_here)?;
            tdt_time = mark.elapsed();
        }
        let phase_ids: Vec<u32> = phase_tokens.iter().map(|t| t.id).collect();
        let _ = tokenizer.decode(&phase_ids);
        println!(
            "[phases] mel {:.4}s  pre_encode {:.4}s  fv_insert {:.4}s  encoder {:.4}s  tdt {:.4}s",
            mel_time.as_secs_f64(),
            pre_time.as_secs_f64(),
            insert_time.as_secs_f64(),
            enc_time.as_secs_f64(),
            tdt_time.as_secs_f64()
        );
    }

    let started = std::time::Instant::now();
    for _ in 0..reps.max(1) {
        let ids = run_once(&mut decoder, &mut pre_gpu, &fv_gpu)?;
        if ids != first {
            bail!("backend {backend} diverged on a repeated run");
        }
    }
    let steady = started.elapsed().as_secs_f64() / reps.max(1) as f64;

    println!(
        "backend={backend} steps={steps} audio={audio_seconds:.2}s tokens={}",
        first.len()
    );
    println!("[build]  {:.3}s", build.as_secs_f64());
    println!(
        "[steady] {:.4}s  RTFx={:.2}x",
        steady,
        audio_seconds / steady
    );
    println!(
        "[end-to-end] {:.3}s  RTFx={:.2}x",
        build.as_secs_f64() + steady,
        audio_seconds / (build.as_secs_f64() + steady)
    );
    Ok(())
}

/// Where the 24-layer encode actually spends its time.
///
/// Report a GEMM microbenchmark at the exact shapes the encoder uses. If it
/// accounts for the whole encode time the kernel is the bottleneck; if it does
/// not, the cost is in submissions or the non-GEMM kernels.
fn gpu_profile_command(nemo_path: &Path, steps: usize) -> Result<()> {
    let bundle = nemo::open(nemo_path)?;
    let store = &bundle.store;
    let pos_len = 2 * steps - 1;
    let scale = conformer::Attention::from_store(store, "encoder.layers.0.self_attn")?.scale();
    let encoder = gpu_encoder::EncoderGpu::new(steps, store)?;
    println!("adapter {}", encoder.adapter());

    let input = vec![0.0f32; steps * conformer::D_MODEL];
    // First call pays the driver's pipeline compilation; time it separately from the
    // steady state, which is what an RTFx number has to be based on.
    let started = std::time::Instant::now();
    let _ = encoder.encode(&input, scale)?;
    let cold = started.elapsed();
    let started = std::time::Instant::now();
    let _ = encoder.encode(&input, scale)?;
    let encode = started.elapsed();
    println!(
        "encode 24 layers: cold {:.3}s  steady {:.3}s",
        cold.as_secs_f64(),
        encode.as_secs_f64()
    );

    // Per-block attribution: time each family across all 24 layers.
    for stage in gpu_encoder::Stage::all_blocks()
        .iter()
        .chain(std::iter::once(&gpu_encoder::Stage::All))
    {
        let elapsed = encoder.encode_stage(&input, scale, *stage)?;
        println!("  {:<9} {:.3}s", stage.label(), elapsed.as_secs_f64());
    }

    // (m, n, k, reps) — reps chosen so each row covers all 24 layers.
    let shapes: [(&str, usize, usize, usize, usize); 6] = [
        ("ff  w1  ", steps, conformer::FFN, conformer::D_MODEL, 48),
        ("ff  w2  ", steps, conformer::D_MODEL, conformer::FFN, 48),
        (
            "qkv     ",
            steps,
            3 * conformer::D_MODEL,
            conformer::D_MODEL,
            24,
        ),
        (
            "pos     ",
            pos_len,
            conformer::D_MODEL,
            conformer::D_MODEL,
            24,
        ),
        (
            "conv pw1",
            steps,
            2 * conformer::D_MODEL,
            conformer::D_MODEL,
            24,
        ),
        (
            "out/pw2 ",
            steps,
            conformer::D_MODEL,
            conformer::D_MODEL,
            48,
        ),
    ];

    let gpu = gpu::GpuGemm::new()?;
    let mut total = 0.0f64;
    let mut flops = 0.0f64;
    for (label, m, n, k, reps) in shapes {
        let a = vec![0.0f32; m * k];
        let w = vec![0.0f32; n * k];
        let started = std::time::Instant::now();
        let _ = gpu.gemm_reps(&a, &w, m, n, k, reps)?;
        let elapsed = started.elapsed().as_secs_f64();
        let work = 2.0 * m as f64 * n as f64 * k as f64 * reps as f64;
        total += elapsed;
        flops += work;
        println!(
            "  {label} m={m:3} n={n:4} k={k:4} reps={reps:2}: {elapsed:7.3}s  {:6.1} GFLOP/s",
            work / elapsed / 1e9
        );
    }
    println!(
        "gemm total {total:.3}s / {:.1} GFLOP -> {:.1} GFLOP/s   (encode {:.3}s)",
        flops / 1e9,
        flops / total / 1e9,
        encode.as_secs_f64()
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("bench-steady") => {
            let usage = "usage: bench-steady <nemo> <wav> <cpu|gpu> [reps]";
            let model = args.next().context(usage)?;
            let wav = args.next().context(usage)?;
            let backend = args.next().context(usage)?;
            let reps = args.next().and_then(|v| v.parse().ok()).unwrap_or(5);
            bench_steady_command(Path::new(&model), Path::new(&wav), &backend, reps)
        }
        Some("gpu-profile") => {
            let model = args.next().context("usage: gpu-profile <nemo> [steps]")?;
            let steps = match args.next() {
                Some(raw) => raw.parse().context("steps must be an integer")?,
                None => 189,
            };
            gpu_profile_command(Path::new(&model), steps)
        }
        Some("gpu-flops") => flops_probe::run(),
        Some("bench") => {
            let model = args.next().context("usage: bench <nemo> <wav>")?;
            let wav = args.next().context("usage: bench <nemo> <wav>")?;
            bench_command(Path::new(&model), Path::new(&wav), std::time::Instant::now())
        }
        _ => {
            eprintln!("fsp-bench — performance measurement
");
            eprintln!("  bench-steady <nemo> <wav> <cpu|gpu> [reps]");
            eprintln!("  gpu-profile <nemo>        per-stage encoder timing");
            eprintln!("  gpu-flops                  pure-FMA throughput probe");
            eprintln!("  bench <nemo> <wav>         phase contract from notes/perf_contract.md");
            Ok(())
        }
    }
}
