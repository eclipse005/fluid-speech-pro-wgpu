# Fluid Speech Pro wgpu

**Fluid Speech Pro (Parakeet TDT 0.63B) speech recognition in Rust with wgpu.**

[![GitHub stars](https://img.shields.io/github/stars/eclipse005/fluid-speech-pro-wgpu?style=flat-square&logo=github&color=f59e0b)](https://github.com/eclipse005/fluid-speech-pro-wgpu/stargazers) [![GitHub forks](https://img.shields.io/github/forks/eclipse005/fluid-speech-pro-wgpu?style=flat-square&logo=github)](https://github.com/eclipse005/fluid-speech-pro-wgpu/network/members) [![Build](https://img.shields.io/github/actions/workflow/status/eclipse005/fluid-speech-pro-wgpu/build.yml?branch=master&style=flat-square&label=build)](https://github.com/eclipse005/fluid-speech-pro-wgpu/actions/workflows/build.yml) [![License](https://img.shields.io/github/license/eclipse005/fluid-speech-pro-wgpu?style=flat-square&color=16a34a)](LICENSE)

[![Windows](https://img.shields.io/badge/Windows-x64-0078D6?style=flat-square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI2ZmZiI%2BPHBhdGggZD0iTTAsMEgxMS4zNzdWMTEuMzcySDBaTTEyLjYyMywwSDI0VjExLjM3MkgxMi42MjNaTTAsMTIuNjIzSDExLjM3N1YyNEgwWm0xMi42MjMsMEgyNFYyNEgxMi42MjMiLz48L3N2Zz4%3D)](#install) [![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-111111?style=flat-square&logo=apple&logoColor=white)](#install) [![Linux](https://img.shields.io/badge/Linux-x86__64-FCC624?style=flat-square&logo=linux&logoColor=111111)](#install)

**English** · [简体中文](README.zh-CN.md)

A lightweight, cross-platform Rust implementation of [Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) (NVIDIA Parakeet TDT 0.63B FastConformer + FrameCleaner), using [wgpu](https://github.com/gfx-rs/wgpu) for GPU acceleration.

The goal is simple: run Fluid Speech Pro **locally and natively** directly from the official `.nemo` archive — with full NeMo TDT feature parity, no Python, no weight conversion, and no vendor-specific GPU runtimes.

### Features

* 🦀 Pure Rust
* 🎮 GPU acceleration with wgpu (Vulkan / D3D12 / Metal)
* 🌍 Cross-platform GPU support
* 🖥️ Windows / macOS / Linux
* ⚡ Full f32 CPU backend
* 📦 Reads the official `.nemo` archive directly in memory (zero unpacked files)
* 🎙️ 25 European languages with automatic punctuation & capitalization
* ✨ Clean mode (`--clean-mode`) — edited text with filler/repetition removal via LoRA folding
* ⏱️ Four timestamp tiers (token / word / segment / char) & per-frame alignments
* 🎯 Full NeMo decoding parity: `greedy`, `greedy_batch`, `beam`, `maes`, `malsd_batch`, and N-best
* 📊 Frame / token / word confidence scoring (entropy & max-prob)
* 🔤 Context biasing (`BoostingTree`) & external ARPA N-gram LM shallow fusion (no KenLM dependency)
* 🪟 Full attention (`rel_pos`) & sliding-window local attention (`rel_pos_local_attn`) with zero-reload runtime switching
* 📡 Streaming decoder state carry-over (`partial_state`)
* 🧩 CLI + Rust library

### Install

As a Cargo dependency:

```toml
[dependencies]
fluid-speech-pro-wgpu = { git = "https://github.com/eclipse005/fluid-speech-pro-wgpu.git" }
```

Or build the CLI from source:

```bash
git clone https://github.com/eclipse005/fluid-speech-pro-wgpu.git
cd fluid-speech-pro-wgpu
cargo build --release        # target/release/fsp
```

| Feature | Description |
|---------|-------------|
| `internals` | Exposes internal encoder/decoder modules and builds developer verification/benchmark tools (`fsp-verify`, `fsp-bench`). Off by default |

### Model download

Weights are **not** included in this repository. Download the official NeMo checkpoint from Hugging Face (rights remain with the original authors, CC BY 4.0) and pass `fluid-speech-pro.nemo` directly — nothing has to be converted or extracted:

- [FluidVoice-Labs/fluid-speech-pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) (`fluid-speech-pro.nemo`)

```bash
hf download FluidVoice-Labs/fluid-speech-pro fluid-speech-pro.nemo --local-dir ./models
```

### Quick Start

```bash
fsp transcribe ./models/fluid-speech-pro.nemo ./audio.wav --backend gpu
```

By default `fsp transcribe` prints the transcript to stdout and a summary line (`backend`, `clean_mode`, `score`, token/word/segment counts, steps) to stderr. Pass `--json` to emit a structured JSON document with text, tokens, timestamps, confidence, and optional alignments.

```bash
# Structured JSON with word/segment/char timestamps
fsp transcribe ./models/fluid-speech-pro.nemo ./audio.wav --backend gpu --json

# Clean mode (edited text without fillers or repeated words)
fsp transcribe ./models/fluid-speech-pro.nemo ./audio.wav --backend gpu --clean-mode

# Leaderboard-style TDT beam search (malsd_batch, beam 4)
fsp transcribe ./models/fluid-speech-pro.nemo ./audio.wav --backend gpu --strategy malsd_batch --beam-size 4

# Sliding-window local attention for long audio with minimal VRAM
fsp transcribe ./models/fluid-speech-pro.nemo ./long.wav --backend gpu --local-attn 256,256

# Phrase boosting + confidence
fsp transcribe ./models/fluid-speech-pro.nemo ./audio.wav --backend gpu \
  --boost-phrase "Victoria Harbour" --boost-alpha 1.0 --confidence --json
```

#### CLI Options

| Option | Description |
|--------|-------------|
| `--backend cpu\|gpu` | Execution backend (default: `cpu`) |
| `--clean-mode` | Enable Fluid Speech Pro edited (non-verbatim) mode |
| `--local-attn <L,R>` | Use `rel_pos_local_attn` sliding window (e.g. `128,128` or `256,256`) |
| `--channel average\|N\|N,M` | Multi-channel WAV selector (default: `average`; auto-resampled to 16 kHz) |
| `--strategy <name>` | `greedy`, `greedy_batch` (default), `beam`, `maes`, or `malsd_batch` |
| `--beam-size <N>` | Beam width for `beam` / `maes` / `malsd_batch` (default: `2`) |
| `--n-best` | Output all ranked N-best hypotheses |
| `--no-score-norm` | Disable length normalization in beam ranking |
| `--max-symbols <N\|none>` | Max non-blank tokens emitted per frame (default: `10`) |
| `--temperature <F>` | Logit / softmax temperature (default: `1.0`) |
| `--timestamps <tier>` | `all` (default), `word`, `segment`, `char`, or `none` |
| `--segment-gap <frames>` | Split segments on pause >= N 80 ms frames |
| `--alignments` | Record per-frame alignment history |
| `--confidence` | Compute frame, token, and word confidences (`--confidence-method`, `--confidence-entropy`, `--confidence-norm`, `--confidence-alpha`, `--confidence-agg`, `--confidence-include-blank`, `--confidence-include-duration`) |
| `--boost-phrase <text>` | Boost a key phrase (repeatable) or `--boost-file <path>` (`--boost-alpha`, `--boost-score`, `--boost-depth-scaling`) |
| `--ngram-lm <arpa>` | External ARPA n-gram LM file (`--ngram-lm-alpha`, `--blank-lm-mode`, `--pruning-mode`) |
| `--json` | Emit structured JSON on stdout (`transcribe`) |

### Batch mode

`fsp batch` loads the model once into resident memory/VRAM and transcribes a list of clips in one pass, reusing weights and dynamically resizing scratch without reloading:

```bash
fsp batch ./models/fluid-speech-pro.nemo ./clips.json ./out.json --backend gpu
```

`clips.json` is an array of `{"id": "clip_001", "path": "clip_001.wav"}` entries (relative paths resolve against the list file's directory). All model, decoding, confidence, biasing, and attention flags apply across the batch.

### Library

```rust
use fluid_speech_pro::api::{Backend, Transcriber};

let mut asr = Transcriber::load("models/fluid-speech-pro.nemo".as_ref(), Backend::Gpu)?;
let out = asr.transcribe_file("audio.wav".as_ref())?;

println!("{}", out.text);
for word in &out.words {
    println!("{:6.2} {:6.2}  {}", word.start, word.end, word.word);
}
```

Weights are uploaded once at `Transcriber::load` / `Transcriber::load_with` and kept resident across calls; varying audio lengths only resize scratch buffers. Also available through `fluid_speech_pro::api`:

* **Clean mode & local attention at load or runtime** — `LoadOptions::new(Backend::Gpu).with_clean_mode(true).with_attention_model(AttentionModel::RelPosLocalAttn { left: 256, right: 256 })`, or switch dynamically with zero weight reload via `asr.change_attention_model(...)`
* **In-memory audio & batches** — `transcribe(&samples_16k)`, `transcribe_with(&samples_16k, &opts)`, `transcribe_batch(&clips, &opts)`, `transcribe_files(&paths, &opts)`
* **Beam search & N-best** — `DecodingStrategy::{Beam, Maes, MalsdBatch}` via `transcribe_nbest` / `transcribe_file_nbest`
* **Confidence & alignments** — `ConfidenceConfig` (Tsallis / Gibbs / Renyi / MaxProb) and `preserve_alignments`
* **Context biasing & external LM** — pre-compile once with `asr.build_boosting_tree(&cfg)` or `asr.load_arpa_lm(path, Some(100))` and reuse across calls
* **Streaming carry-over** — set `return_decoder_state = true` and pass `Transcript::decoder_state` into the next chunk via `TranscribeOptions::partial_state`

Voice activity detection (VAD) and content-based audio splitting are intentionally kept outside this crate so upstream pipelines stay in full control of segmentation.

See `cargo doc` for the full API.

### Why wgpu?

Instead of relying on CUDA, ROCm, or other vendor-specific runtimes, this project uses **wgpu** as a unified GPU abstraction (Vulkan, D3D12, Metal).

This makes it possible to build a single Rust-based ASR runtime that runs directly from the official `.nemo` checkpoint across platforms and GPU vendors, with a matching f32 CPU backend when no GPU is used.

### Project Status

🚧 **Active development**

Core numerical parity against the official NeMo TDT pipeline is verified across both CPU and GPU backends. Hardware coverage and performance tuning across additional GPUs are ongoing.

### Related

* [Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) — the official model release
* [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — the base architecture
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) — Qwen3-ASR in Rust with wgpu
* [qwen3-aligner-wgpu](https://github.com/eclipse005/qwen3-aligner-wgpu) — word-level forced alignment in Rust with wgpu

### License

MIT for this implementation. Model weights remain under their upstream license (CC BY 4.0 — credit **Fluid Speech Pro by FluidVoice** as specified in the model card).

This repository is an **independent Rust inference implementation** for loading and running the officially released Fluid Speech Pro weights — not an official FluidVoice or NVIDIA release, and not affiliated with the original authors.
