//! `fsp` — transcribe audio with a Fluid Speech Pro `.nemo` archive.
//!
//! This binary deliberately depends only on the library's public `api`. Anything that
//! needs the crate internals (stage-by-stage fixture comparison, throughput probes)
//! lives in `fsp-verify` and `fsp-bench`, which are built from this repository only.

use std::path::Path;

use anyhow::{Context, Result, bail};

use fluid_speech_pro::api::{
    AttentionModel, Backend, BlankLmScoreMode, BoostingTreeConfig, ChannelSelector,
    ConfidenceAggregation, ConfidenceMethodKind, DecodingStrategy, EntropyNorm, EntropyType,
    LoadOptions, PruningMode, TimestampType, TranscribeOptions, Transcriber,
};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut backend = Backend::Cpu;
    let mut clean_mode = false;
    let mut attention_model = AttentionModel::RelPos;
    let mut json_out = false;
    let mut n_best = false;
    let mut trans_opts = TranscribeOptions::default();
    let mut rest: Vec<String> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => {
                let value = args.next().context("--backend needs cpu or gpu")?;
                backend = parse_backend(&value)?;
            }
            "--clean-mode" => clean_mode = true,
            "--local-attn" => {
                let value = args
                    .next()
                    .context("--local-attn needs `<left>,<right>` (e.g. 128,128)")?;
                attention_model = parse_local_attn(&value)?;
            }
            "--json" => json_out = true,
            "--n-best" => n_best = true,
            "--strategy" => {
                let value = args
                    .next()
                    .context("--strategy needs greedy|greedy_batch|beam|maes|malsd_batch")?;
                trans_opts.strategy = match value.as_str() {
                    "greedy" => DecodingStrategy::Greedy,
                    "greedy_batch" => DecodingStrategy::GreedyBatch,
                    "beam" => DecodingStrategy::Beam,
                    "maes" => DecodingStrategy::Maes,
                    "malsd_batch" => DecodingStrategy::MalsdBatch,
                    other => bail!("unknown --strategy `{other}`"),
                };
            }
            "--beam-size" => {
                let value = args.next().context("--beam-size needs an integer >= 1")?;
                trans_opts.beam.beam_size = value.parse().context("invalid --beam-size")?;
            }
            "--no-score-norm" => trans_opts.beam.score_norm = false,
            "--max-symbols" => {
                let value = args.next().context("--max-symbols needs an integer or `none`")?;
                let parsed = if value == "none" || value == "0" {
                    None
                } else {
                    Some(value.parse().context("invalid --max-symbols")?)
                };
                trans_opts.max_symbols_per_step = parsed;
                trans_opts.beam.max_symbols_per_step = parsed;
            }
            "--temperature" => {
                let value = args.next().context("--temperature needs a positive float")?;
                let t: f32 = value.parse().context("invalid --temperature")?;
                trans_opts.temperature = t;
                trans_opts.beam.softmax_temperature = t;
            }
            "--maes-steps" => {
                let value = args.next().context("--maes-steps needs an integer >= 1")?;
                trans_opts.beam.maes_num_steps = value.parse().context("invalid --maes-steps")?;
            }
            "--maes-prefix-alpha" => {
                let value = args.next().context("--maes-prefix-alpha needs an integer")?;
                trans_opts.beam.maes_prefix_alpha =
                    value.parse().context("invalid --maes-prefix-alpha")?;
            }
            "--maes-gamma" => {
                let value = args.next().context("--maes-gamma needs a float")?;
                trans_opts.beam.maes_expansion_gamma =
                    value.parse().context("invalid --maes-gamma")?;
            }
            "--maes-beta" => {
                let value = args.next().context("--maes-beta needs an integer")?;
                trans_opts.beam.maes_expansion_beta =
                    value.parse().context("invalid --maes-beta")?;
            }
            "--timestamps" => {
                let value = args
                    .next()
                    .context("--timestamps needs all|word|segment|char|none")?;
                trans_opts.timestamps.timestamp_type = match value.as_str() {
                    "all" => TimestampType::All,
                    "word" => TimestampType::Word,
                    "segment" => TimestampType::Segment,
                    "char" => TimestampType::Char,
                    "none" => TimestampType::None,
                    other => bail!("unknown --timestamps `{other}`"),
                };
            }
            "--segment-gap" => {
                let value = args.next().context("--segment-gap needs a frame count")?;
                trans_opts.timestamps.segment_gap_threshold =
                    Some(value.parse().context("invalid --segment-gap")?);
            }
            "--channel" => {
                let value = args
                    .next()
                    .context("--channel needs `average`, an index, or comma-separated indices")?;
                trans_opts.channel_selector = parse_channel(&value)?;
            }
            "--alignments" => trans_opts.preserve_alignments = true,
            "--confidence" => {
                trans_opts.confidence.preserve_frame_confidence = true;
                trans_opts.confidence.preserve_token_confidence = true;
                trans_opts.confidence.preserve_word_confidence = true;
            }
            "--confidence-method" => {
                let value = args.next().context("--confidence-method needs entropy or max_prob")?;
                trans_opts.confidence.method_cfg.name = match value.as_str() {
                    "entropy" => ConfidenceMethodKind::Entropy,
                    "max_prob" => ConfidenceMethodKind::MaxProb,
                    other => bail!("unknown --confidence-method `{other}`"),
                };
            }
            "--confidence-entropy" => {
                let value = args.next().context("--confidence-entropy needs gibbs|tsallis|renyi")?;
                trans_opts.confidence.method_cfg.entropy_type = match value.as_str() {
                    "gibbs" => EntropyType::Gibbs,
                    "tsallis" => EntropyType::Tsallis,
                    "renyi" => EntropyType::Renyi,
                    other => bail!("unknown --confidence-entropy `{other}`"),
                };
            }
            "--confidence-norm" => {
                let value = args.next().context("--confidence-norm needs exp|lin")?;
                trans_opts.confidence.method_cfg.entropy_norm = match value.as_str() {
                    "exp" => EntropyNorm::Exp,
                    "lin" => EntropyNorm::Lin,
                    other => bail!("unknown --confidence-norm `{other}`"),
                };
            }
            "--confidence-alpha" => {
                let value = args.next().context("--confidence-alpha needs a float")?;
                trans_opts.confidence.method_cfg.alpha =
                    value.parse().context("invalid --confidence-alpha")?;
            }
            "--confidence-agg" => {
                let value = args.next().context("--confidence-agg needs min|max|mean|prod")?;
                trans_opts.confidence.aggregation = match value.as_str() {
                    "min" => ConfidenceAggregation::Min,
                    "max" => ConfidenceAggregation::Max,
                    "mean" => ConfidenceAggregation::Mean,
                    "prod" => ConfidenceAggregation::Prod,
                    other => bail!("unknown --confidence-agg `{other}`"),
                };
            }
            "--confidence-include-blank" => trans_opts.confidence.exclude_blank = false,
            "--confidence-include-duration" => trans_opts.confidence.tdt_include_duration = true,
            "--boost-phrase" => {
                let value = args.next().context("--boost-phrase needs a phrase string")?;
                let bt = trans_opts
                    .boosting_tree
                    .get_or_insert_with(BoostingTreeConfig::default);
                bt.key_phrases.push(value);
                if trans_opts.boosting_tree_alpha == 0.0 {
                    trans_opts.boosting_tree_alpha = 1.0;
                }
            }
            "--boost-file" => {
                let value = args.next().context("--boost-file needs a file path")?;
                let bt = trans_opts
                    .boosting_tree
                    .get_or_insert_with(BoostingTreeConfig::default);
                bt.key_phrases_file = Some(value.into());
                if trans_opts.boosting_tree_alpha == 0.0 {
                    trans_opts.boosting_tree_alpha = 1.0;
                }
            }
            "--boost-alpha" => {
                let value = args.next().context("--boost-alpha needs a float")?;
                trans_opts.boosting_tree_alpha =
                    value.parse().context("invalid --boost-alpha")?;
            }
            "--boost-score" => {
                let value = args.next().context("--boost-score needs a float")?;
                let bt = trans_opts
                    .boosting_tree
                    .get_or_insert_with(BoostingTreeConfig::default);
                bt.context_score = value.parse().context("invalid --boost-score")?;
            }
            "--boost-depth-scaling" => {
                let value = args.next().context("--boost-depth-scaling needs a float")?;
                let bt = trans_opts
                    .boosting_tree
                    .get_or_insert_with(BoostingTreeConfig::default);
                bt.depth_scaling = value.parse().context("invalid --boost-depth-scaling")?;
            }
            "--ngram-lm" => {
                let value = args.next().context("--ngram-lm needs an ARPA file path")?;
                trans_opts.ngram_lm_path = Some(value.into());
                if trans_opts.ngram_lm_alpha == 0.0 {
                    trans_opts.ngram_lm_alpha = 0.3;
                }
            }
            "--ngram-lm-alpha" => {
                let value = args.next().context("--ngram-lm-alpha needs a float")?;
                trans_opts.ngram_lm_alpha = value.parse().context("invalid --ngram-lm-alpha")?;
            }
            "--blank-lm-mode" => {
                let value = args.next().context("--blank-lm-mode needs no_score|lm_weighted_full")?;
                trans_opts.beam.blank_lm_score_mode = match value.as_str() {
                    "no_score" => BlankLmScoreMode::NoScore,
                    "lm_weighted_full" => BlankLmScoreMode::LmWeightedFull,
                    other => bail!("unknown --blank-lm-mode `{other}`"),
                };
            }
            "--pruning-mode" => {
                let value = args.next().context("--pruning-mode needs early|late")?;
                trans_opts.beam.pruning_mode = match value.as_str() {
                    "early" => PruningMode::Early,
                    "late" => PruningMode::Late,
                    other => bail!("unknown --pruning-mode `{other}`"),
                };
            }
            "--help" | "-h" => return usage(),
            other => rest.push(other.to_string()),
        }
    }

    let load_opts = LoadOptions::new(backend)
        .with_clean_mode(clean_mode)
        .with_attention_model(attention_model);

    match rest.first().map(String::as_str) {
        Some("transcribe") => {
            let usage = "usage: fsp transcribe <nemo> <audio.wav> [--backend cpu|gpu] [--clean-mode] [--strategy ...] [--json]";
            let model = rest.get(1).context(usage)?;
            let wav = rest.get(2).context(usage)?;
            transcribe(Path::new(model), Path::new(wav), load_opts, &trans_opts, json_out, n_best)
        }
        Some("batch") => {
            let usage = "usage: fsp batch <nemo> <list.json> <out.json> [--backend cpu|gpu] [--clean-mode]";
            let model = rest.get(1).context(usage)?;
            let list = rest.get(2).context(usage)?;
            let out = rest.get(3).context(usage)?;
            batch(Path::new(model), Path::new(list), Path::new(out), load_opts, &trans_opts)
        }
        Some(other) => bail!("unknown command `{other}`; try --help"),
        None => usage(),
    }
}

fn parse_backend(value: &str) -> Result<Backend> {
    match value {
        "cpu" => Ok(Backend::Cpu),
        "gpu" => Ok(Backend::Gpu),
        other => bail!("unknown backend `{other}`; expected cpu or gpu"),
    }
}

fn parse_local_attn(value: &str) -> Result<AttentionModel> {
    let (l, r) = value
        .split_once(',')
        .context("expected `<left>,<right>` for --local-attn, e.g. 128,128")?;
    let left: usize = l.trim().parse().context("invalid left context")?;
    let right: usize = r.trim().parse().context("invalid right context")?;
    let model = AttentionModel::RelPosLocalAttn { left, right };
    model.validate()?;
    Ok(model)
}

fn parse_channel(value: &str) -> Result<ChannelSelector> {
    if value.eq_ignore_ascii_case("average") || value.eq_ignore_ascii_case("avg") {
        return Ok(ChannelSelector::Average);
    }
    if value.contains(',') {
        let mut list = Vec::new();
        for part in value.split(',') {
            list.push(part.trim().parse::<usize>().context("invalid channel index")?);
        }
        return Ok(ChannelSelector::Channels(list));
    }
    Ok(ChannelSelector::Channel(
        value.trim().parse::<usize>().context("invalid channel index")?,
    ))
}

fn usage() -> Result<()> {
    println!("fsp — Fluid Speech Pro inference (CPU or wgpu)");
    println!();
    println!("  fsp transcribe <nemo> <audio.wav>        transcribe one file");
    println!("  fsp batch <nemo> <list.json> <out.json>  transcribe a list of clips");
    println!();
    println!("Model & encoder options:");
    println!("  --backend cpu|gpu              execution backend (default: cpu)");
    println!("  --clean-mode                   enable Fluid Speech Pro edited (non-verbatim) mode");
    println!("  --local-attn <L,R>             use rel_pos_local_attn sliding window (e.g. 128,128)");
    println!("  --channel average|N|N,M        multi-channel WAV selector (default: average)");
    println!();
    println!("Decoding options:");
    println!("  --strategy <name>              greedy|greedy_batch|beam|maes|malsd_batch (default: greedy_batch)");
    println!("  --beam-size <N>                beam width for beam/maes/malsd_batch (default: 2)");
    println!("  --n-best                       output all ranked N-best hypotheses");
    println!("  --no-score-norm                disable length normalization in beam ranking");
    println!("  --max-symbols <N|none>         max non-blank tokens per frame (default: 10)");
    println!("  --temperature <F>              logit/softmax temperature (default: 1.0)");
    println!("  --maes-steps <N>               mAES adaptive steps (default: 2)");
    println!("  --maes-prefix-alpha <N>        mAES prefix search depth (default: 1)");
    println!("  --maes-gamma <F>               mAES expansion pruning threshold (default: 2.3)");
    println!("  --maes-beta <N>                mAES extra expansion candidates (default: 2)");
    println!();
    println!("Timestamps, alignments & confidence:");
    println!("  --timestamps <tier>            all|word|segment|char|none (default: all)");
    println!("  --segment-gap <frames>         split segments on pause >= N 80ms frames");
    println!("  --alignments                   record per-frame alignment history");
    println!("  --confidence                   compute frame, token, and word confidences");
    println!("  --confidence-method <m>        entropy|max_prob (default: entropy)");
    println!("  --confidence-entropy <t>       tsallis|gibbs|renyi (default: tsallis)");
    println!("  --confidence-norm <n>          exp|lin (default: exp)");
    println!("  --confidence-alpha <F>         confidence power scale > 0 (default: 0.33)");
    println!("  --confidence-agg <a>           min|max|mean|prod (default: min)");
    println!("  --confidence-include-blank     fold blank frame confidence into tokens");
    println!("  --confidence-include-duration  include TDT duration confidence pairs");
    println!();
    println!("Context biasing & external LM shallow fusion:");
    println!("  --boost-phrase <text>          boost a key phrase (repeatable)");
    println!("  --boost-file <path>            file of key phrases (one per line)");
    println!("  --boost-alpha <F>              boosting tree weight (default: 1.0 when phrases given)");
    println!("  --boost-score <F>              per-token context bonus (default: 1.0)");
    println!("  --boost-depth-scaling <F>      depth scaling factor (default: 2.0)");
    println!("  --ngram-lm <arpa>              external ARPA n-gram LM file");
    println!("  --ngram-lm-alpha <F>           n-gram LM weight (default: 0.3 when LM given)");
    println!("  --blank-lm-mode <m>            no_score|lm_weighted_full (default: lm_weighted_full)");
    println!("  --pruning-mode <m>             early|late (default: late)");
    println!();
    println!("Output:");
    println!("  --json                         emit structured JSON on stdout (transcribe)");
    Ok(())
}

fn transcribe(
    nemo_path: &Path,
    wav: &Path,
    load_opts: LoadOptions,
    trans_opts: &TranscribeOptions,
    json_out: bool,
    n_best: bool,
) -> Result<()> {
    let mut asr = Transcriber::load_with(nemo_path, load_opts)?;
    if n_best {
        let hyps = asr.transcribe_file_nbest(wav, trans_opts)?;
        if json_out {
            let arr: Vec<_> = hyps.iter().map(|h| h.to_json()).collect();
            println!("{}", serde_json::to_string_pretty(&arr)?);
        } else {
            for (i, h) in hyps.iter().enumerate() {
                println!("[#{i} score={:.4}] {}", h.score, h.text);
            }
        }
        eprintln!(
            "backend={} clean_mode={} n_best={}",
            asr.backend().label(),
            asr.clean_mode(),
            hyps.len()
        );
        return Ok(());
    }
    let transcript = asr.transcribe_file_with(wav, trans_opts)?;
    if json_out {
        println!("{}", serde_json::to_string_pretty(&transcript.to_json())?);
    } else {
        println!("{}", transcript.text);
    }
    eprintln!(
        "backend={} clean_mode={} score={:.4} tokens={} words={} segments={} chars={} steps={}",
        asr.backend().label(),
        asr.clean_mode(),
        transcript.score,
        transcript.tokens.len(),
        transcript.words.len(),
        transcript.segments.len(),
        transcript.chars.len(),
        transcript.steps
    );
    Ok(())
}

fn batch(
    nemo_path: &Path,
    list_path: &Path,
    out_path: &Path,
    load_opts: LoadOptions,
    trans_opts: &TranscribeOptions,
) -> Result<()> {
    let mut asr = Transcriber::load_with(nemo_path, load_opts)?;
    let text = std::fs::read_to_string(list_path)
        .with_context(|| format!("read {}", list_path.display()))?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&text).context("parse clip list")?;

    let mut clips = Vec::with_capacity(entries.len());
    for entry in entries {
        let id = entry
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let wav = entry
            .get("wav")
            .and_then(|v| v.as_str())
            .context("entry has no wav")?;
        let transcript = asr.transcribe_file_with(Path::new(wav), trans_opts)?;
        let mut value = transcript.to_json();
        value["id"] = serde_json::Value::String(id);
        clips.push(value);
    }

    let document = serde_json::json!({ "clips": clips });
    std::fs::write(out_path, serde_json::to_vec(&document)?)
        .with_context(|| format!("write {}", out_path.display()))?;
    println!(
        "wrote {} ({} clips, backend={}, clean_mode={})",
        out_path.display(),
        document["clips"].as_array().map_or(0, |list| list.len()),
        asr.backend().label(),
        asr.clean_mode(),
    );
    Ok(())
}
