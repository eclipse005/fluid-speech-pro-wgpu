# Fluid Speech Pro wgpu

**基于 wgpu 的 Fluid Speech Pro（Parakeet TDT 0.63B）语音识别（Rust 实现）。**

[![GitHub stars](https://img.shields.io/github/stars/eclipse005/fluid-speech-pro-wgpu?style=flat-square&logo=github&color=f59e0b)](https://github.com/eclipse005/fluid-speech-pro-wgpu/stargazers) [![GitHub forks](https://img.shields.io/github/forks/eclipse005/fluid-speech-pro-wgpu?style=flat-square&logo=github)](https://github.com/eclipse005/fluid-speech-pro-wgpu/network/members) [![Build](https://img.shields.io/github/actions/workflow/status/eclipse005/fluid-speech-pro-wgpu/build.yml?branch=master&style=flat-square&label=build)](https://github.com/eclipse005/fluid-speech-pro-wgpu/actions/workflows/build.yml) [![License](https://img.shields.io/github/license/eclipse005/fluid-speech-pro-wgpu?style=flat-square&color=16a34a)](LICENSE)

[![Windows](https://img.shields.io/badge/Windows-x64-0078D6?style=flat-square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI2ZmZiI%2BPHBhdGggZD0iTTAsMEgxMS4zNzdWMTEuMzcySDBaTTEyLjYyMywwSDI0VjExLjM3MkgxMi42MjNaTTAsMTIuNjIzSDExLjM3N1YyNEgwWm0xMi42MjMsMEgyNFYyNEgxMi42MjMiLz48L3N2Zz4%3D)](#安装) [![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-111111?style=flat-square&logo=apple&logoColor=white)](#安装) [![Linux](https://img.shields.io/badge/Linux-x86__64-FCC624?style=flat-square&logo=linux&logoColor=111111)](#安装)

[English](README.md) · **简体中文**

[Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro)（NVIDIA Parakeet TDT 0.63B FastConformer + FrameCleaner）的轻量级跨平台 Rust 实现，使用 [wgpu](https://github.com/gfx-rs/wgpu) 进行 GPU 加速。

目标很简单：直接从官方 `.nemo` 归档**在本地原生运行** Fluid Speech Pro——完整对齐 NeMo TDT 推理特性，不依赖 Python、不转换或拆包权重，也不依赖特定厂商的 GPU 运行时。

### 特性

* 🦀 纯 Rust
* 🎮 wgpu GPU 加速（Vulkan / D3D12 / Metal）
* 🌍 跨平台 GPU 支持
* 🖥️ Windows / macOS / Linux
* ⚡ 完整 f32 CPU 后端
* 📦 内存直读官方 `.nemo` 文件（不落盘转换、不拆包）
* 🎙️ 支持 25 种欧洲语言，自动标点与大小写
* ✨ Clean Mode（`--clean-mode`）—— LoRA 原位折叠输出去口癖/去重复的编辑文本
* ⏱️ 四级时间戳（token / word / segment / char）与逐帧对齐历史
* 🎯 完整对齐 NeMo TDT 解码策略：`greedy`、`greedy_batch`、`beam`、`maes`、`malsd_batch` 与 N-best
* 📊 帧 / token / 词级置信度（熵度量与最大概率）
* 🔤 上下文短语偏置（`BoostingTree`）与外部 ARPA N-gram LM 浅融合（无需安装 KenLM）
* 🪟 支持全注意力（`rel_pos`）与滑动窗口局部注意力（`rel_pos_local_attn`），运行期零重载切换
* 📡 流式解码状态续传（`partial_state`）
* 🧩 CLI + Rust 库

### 安装

作为 Cargo 依赖：

```toml
[dependencies]
fluid-speech-pro-wgpu = { git = "https://github.com/eclipse005/fluid-speech-pro-wgpu.git" }
```

或从源码构建 CLI：

```bash
git clone https://github.com/eclipse005/fluid-speech-pro-wgpu.git
cd fluid-speech-pro-wgpu
cargo build --release        # target/release/fsp
```

| Feature | 说明 |
|---------|------|
| `internals` | 暴露内部编码器/解码器模块并构建开发者校验与基准工具（`fsp-verify`、`fsp-bench`）。默认关闭 |

### 模型下载

权重**不在**本仓库内。请从 Hugging Face 下载官方 NeMo 权重（版权归原作者，CC BY 4.0），直接将 `fluid-speech-pro.nemo` 传入即可，无需任何转换或解压：

- [FluidVoice-Labs/fluid-speech-pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro)（`fluid-speech-pro.nemo`）

```bash
hf download FluidVoice-Labs/fluid-speech-pro fluid-speech-pro.nemo --local-dir ./models
```

### 快速上手

```bash
fsp transcribe ./models/fluid-speech-pro.nemo ./audio.wav --backend gpu
```

默认情况下，`fsp transcribe` 将转写文本输出到 stdout，将摘要信息（后端、模式、分数、token/词/段数、步数）输出到 stderr。加上 `--json` 可输出包含文本、token、四级时间戳、置信度与对齐信息的完整 JSON。

```bash
# 输出带 word/segment/char 时间戳的结构化 JSON
fsp transcribe ./models/fluid-speech-pro.nemo ./audio.wav --backend gpu --json

# 启用 Clean Mode（去除口语填充词与重复词）
fsp transcribe ./models/fluid-speech-pro.nemo ./audio.wav --backend gpu --clean-mode

# 排行榜同款 TDT 束搜索（malsd_batch, beam 4）
fsp transcribe ./models/fluid-speech-pro.nemo ./audio.wav --backend gpu --strategy malsd_batch --beam-size 4

# 长音频滑动窗口局部注意力（显著降低显存占用）
fsp transcribe ./models/fluid-speech-pro.nemo ./long.wav --backend gpu --local-attn 256,256

# 热词偏置 + 置信度输出
fsp transcribe ./models/fluid-speech-pro.nemo ./audio.wav --backend gpu \
  --boost-phrase "Victoria Harbour" --boost-alpha 1.0 --confidence --json
```

#### CLI 参数

| 参数 | 说明 |
|------|------|
| `--backend cpu\|gpu` | 执行后端（默认 `cpu`） |
| `--clean-mode` | 启用 Fluid Speech Pro 编辑模式（去重复与填充词） |
| `--local-attn <L,R>` | 使用 `rel_pos_local_attn` 滑动窗口注意力（如 `128,128`、`256,256`） |
| `--channel average\|N\|N,M` | 多声道 WAV 选择策略（默认 `average`；内部自动重采样到 16 kHz） |
| `--strategy <name>` | `greedy`、`greedy_batch`（默认）、`beam`、`maes`、`malsd_batch` |
| `--beam-size <N>` | `beam` / `maes` / `malsd_batch` 的束宽（默认 `2`） |
| `--n-best` | 输出排序后的全部 N-best 候选 |
| `--no-score-norm` | 关闭束搜索排序的长度归一化 |
| `--max-symbols <N\|none>` | 每帧最多输出的非 blank token 数（默认 `10`） |
| `--temperature <F>` | logit / softmax 温度系数（默认 `1.0`） |
| `--timestamps <tier>` | 时间戳层级：`all`（默认）、`word`、`segment`、`char`、`none` |
| `--segment-gap <frames>` | 停顿超过 N 个 80 ms 帧时切分段落 |
| `--alignments` | 记录逐帧对齐历史 |
| `--confidence` | 计算帧/token/词级置信度（支持 `--confidence-method`、`--confidence-entropy`、`--confidence-norm`、`--confidence-alpha`、`--confidence-agg`、`--confidence-include-blank`、`--confidence-include-duration`） |
| `--boost-phrase <text>` | 偏置短语（可重复）或 `--boost-file <path>`（支持 `--boost-alpha`、`--boost-score`、`--boost-depth-scaling`） |
| `--ngram-lm <arpa>` | 外部 ARPA N-gram 语言模型（支持 `--ngram-lm-alpha`、`--blank-lm-mode`、`--pruning-mode`） |
| `--json` | 以结构化 JSON 输出到 stdout（`transcribe`） |

### 批量模式

`fsp batch` 一次性将权重加载常驻内存/显存，连续转写整份音频列表，跨时长仅按需调整 scratch 缓冲，绝不重载权重：

```bash
fsp batch ./models/fluid-speech-pro.nemo ./clips.json ./out.json --backend gpu
```

`clips.json` 为 `{"id": "clip_001", "path": "clip_001.wav"}` 对象数组（相对路径按列表文件所在目录解析），全部模型、解码、置信度、热词与注意力参数均对批量模式生效。

### 作为库使用

```rust
use fluid_speech_pro::api::{Backend, Transcriber};

let mut asr = Transcriber::load("models/fluid-speech-pro.nemo".as_ref(), Backend::Gpu)?;
let out = asr.transcribe_file("audio.wav".as_ref())?;

println!("{}", out.text);
for word in &out.words {
    println!("{:6.2} {:6.2}  {}", word.start, word.end, word.word);
}
```

模型权重在 `Transcriber::load` / `Transcriber::load_with` 时一次性上传并常驻，后续不同长度的音频只按需调整临时缓冲。通过 `fluid_speech_pro::api` 还可以使用：

* **Clean Mode 与局部注意力** —— 在加载时通过 `LoadOptions::new(Backend::Gpu).with_clean_mode(true).with_attention_model(AttentionModel::RelPosLocalAttn { left: 256, right: 256 })` 配置，或运行期通过 `asr.change_attention_model(...)` 零权重重载动态切换
* **内存音频与批量推理** —— `transcribe(&samples_16k)`、`transcribe_with(&samples_16k, &opts)`、`transcribe_batch(&clips, &opts)`、`transcribe_files(&paths, &opts)`
* **束搜索与 N-best** —— `DecodingStrategy::{Beam, Maes, MalsdBatch}` 配合 `transcribe_nbest` / `transcribe_file_nbest`
* **置信度与对齐** —— `ConfidenceConfig`（Tsallis / Gibbs / Renyi / MaxProb）与 `preserve_alignments`
* **热词偏置与外部 LM** —— 使用 `asr.build_boosting_tree(&cfg)` 或 `asr.load_arpa_lm(path, Some(100))` 预构建并在多次调用间复用
* **流式状态续传** —— 开启 `return_decoder_state = true`，将返回的 `Transcript::decoder_state` 经由 `TranscribeOptions::partial_state` 传入下一音频块

VAD 与按内容分段刻意保留在本 crate 之外，以便上游管线完全掌控切分策略。

完整 API 见 `cargo doc`。

### 为什么选择 wgpu？

本项目不依赖 CUDA、ROCm 或其他特定厂商的运行时，而是以 **wgpu** 作为统一的 GPU 抽象层（Vulkan / D3D12 / Metal）。

这使得同一套 Rust 语音识别运行时能够直接读取官方 `.nemo` 权重，跨平台、跨显卡厂商运行，并在无 GPU 环境下提供数值一致的 f32 CPU 后端。

### 项目状态

🚧 **积极开发中**

核心推理链路已在 CPU 与 GPU 双后端上与官方 NeMo TDT 管线完成对齐验证，跨显卡兼容性与性能调优仍在持续完善中。

### 相关项目

* [Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) —— 官方模型发布页
* [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) —— 上游基础架构
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) —— 基于 wgpu 的 Qwen3-ASR（Rust 实现）
* [qwen3-aligner-wgpu](https://github.com/eclipse005/qwen3-aligner-wgpu) —— 基于 wgpu 的词级强制对齐（Rust 实现）

### 许可证

本实现采用 MIT 许可证。模型权重遵循其上游协议（CC BY 4.0，请按模型说明署名 **Fluid Speech Pro by FluidVoice**）。

本仓库是**独立的 Rust 推理实现**，用于加载并运行官方发布的 Fluid Speech Pro 权重，并非 FluidVoice 或 NVIDIA 官方发行版，与原作者无隶属关系。
