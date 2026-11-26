//! A small, stable entry point for callers that just want transcripts.
//!
//! ```no_run
//! use fluid_speech_pro::api::{Backend, Transcriber};
//!
//! let mut asr = Transcriber::load("fluid-speech-pro.nemo".as_ref(), Backend::Gpu)?;
//! println!("{}", asr.transcribe_file("audio.wav".as_ref())?.text);
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! The pipeline is `mel -> pre_encode -> fv_insert -> conformer -> tdt -> text`, each
//! stage its own type. [`Backend`] picks where the compute-heavy middle runs: `Cpu`
//! uses the portable implementations, `Gpu` keeps the weights resident on the adapter
//! and chains the stages through one queue without host round trips.
//!
//! [`Transcript`] carries the four timestamp tiers NeMo reports: token
//! (`timestep`/`duration`), word, segment and char.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};

use crate::beam::{self, BeamHypothesis};
use crate::confidence;
use crate::conformer::Conformer;
use crate::fusion::{self, FusionModel};
use crate::fv_insert::FvInsert;
use crate::mel::{self, MelFrontend};
use crate::nemo;
use crate::pre_encode::PreEncode;
use crate::tdt::{self, GreedyConfig, GreedyOutput, TdtDecoder, Token};
use crate::text::{self, Tokenizer};
use crate::torch::Store;

pub use crate::beam::{BeamConfig, BlankLmScoreMode, PruningMode};
pub use crate::confidence::{
    ConfidenceAggregation, ConfidenceConfig, ConfidenceMethodConfig, ConfidenceMethodKind,
    EntropyNorm, EntropyType, FrameConfidence, FrameConfidenceValue,
};
pub use crate::conformer::AttentionModel;
pub use crate::fusion::{ArpaLm, BoostingTree, BoostingTreeConfig, DEFAULT_TOKEN_OFFSET};
pub use crate::gpu::DeviceSelector;
pub use crate::tdt::{AlignmentEntry, DecoderState, FrameAlignment};
pub use crate::text::{CharTimestamp, Segment, TimestampConfig, TimestampType, Word};

/// Longest utterance the device encoder handles.
///
/// Long audio used to fail here — the layers' dispatches were dropped and the encoder
/// returned its input — because all 24 layers were recorded into one command buffer,
/// which grows with the frame count. Submitting one layer at a time fixed it: a
/// 206 s clip now matches the CPU transcript exactly, three runs in a row.
///
/// The remaining bound is memory, not frames: `pre_encode::check_frames` refuses audio
/// whose conv buffers would not fit.
const GPU_MAX_VERIFIED_STEPS: usize = 100_000;

/// Initial GPU scratch capacity allocated at load time (~82 s at 80 ms/frame, ~84 MiB).
///
/// Weights are uploaded once in [`Transcriber::load`] and stay permanently resident in
/// VRAM; utterances longer than this only resize scratch buffers (~2–5 ms, zero weight
/// transfer).
const GPU_INITIAL_CAPACITY_STEPS: usize = 1024;

/// Where the encoder, subsampling and decoder run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Portable implementations on the calling thread(s).
    Cpu,
    /// wgpu: all model weights upload once at [`Transcriber::load`] and stay resident
    /// in VRAM; only scratch buffers resize on demand for longer utterances.
    Gpu,
}

impl Backend {
    /// Name used in diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Gpu => "gpu",
        }
    }
}

/// Options controlling how a `.nemo` checkpoint is loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadOptions {
    pub backend: Backend,
    /// Which adapter the GPU backend opens (`Auto` by default: the best real GPU).
    /// Ignored for [`Backend::Cpu`]. Parse one with [`DeviceSelector::parse`]
    /// (`"metal"`, `"vulkan:1"`, `"#2"`, an adapter-name substring).
    pub device_selector: DeviceSelector,
    /// Enable Fluid Speech Pro's edited (non-verbatim) clean mode, which folds the
    /// checkpoint's `clean_mode.*` LoRA and bias adapters (`alpha / rank`) into the
    /// model weights in memory. Mirrors `fluid_speech.load(..., clean_mode=True)`.
    pub clean_mode: bool,
    /// Conformer self-attention model (`RelPos` default, or `RelPosLocalAttn { left, right }`).
    /// Can also be switched at runtime via [`Transcriber::change_attention_model`] with zero
    /// weight reload.
    pub attention_model: AttentionModel,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            backend: Backend::Cpu,
            device_selector: DeviceSelector::Auto,
            clean_mode: false,
            attention_model: AttentionModel::RelPos,
        }
    }
}

impl LoadOptions {
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            device_selector: DeviceSelector::Auto,
            clean_mode: false,
            attention_model: AttentionModel::RelPos,
        }
    }

    pub fn with_device_selector(mut self, selector: DeviceSelector) -> Self {
        self.device_selector = selector;
        self
    }

    pub fn with_clean_mode(mut self, enabled: bool) -> Self {
        self.clean_mode = enabled;
        self
    }

    pub fn with_attention_model(mut self, attention_model: AttentionModel) -> Self {
        self.attention_model = attention_model;
        self
    }
}

/// How multi-channel audio is reduced to mono when reading WAV files.
///
/// Mirrors NeMo's `channel_selector` (`"average"`, `int`, or `list[int]`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ChannelSelector {
    /// Average all channels (NeMo default).
    #[default]
    Average,
    /// Select one 0-based channel index.
    Channel(usize),
    /// Average the listed 0-based channel indices.
    Channels(Vec<usize>),
}

/// TDT decoding strategy (mirrors NeMo's `RNNTDecodingConfig.strategy` for `model_type='tdt'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodingStrategy {
    /// Sample-level greedy TDT decoding.
    Greedy,
    /// Label-looping greedy TDT decoding (NeMo default).
    #[default]
    GreedyBatch,
    /// Default TDT beam search (`BeamTDTInfer::default_beam_search`).
    Beam,
    /// Modified Adaptive Expansion Search (`BeamTDTInfer::modified_adaptive_expansion_search`).
    Maes,
    /// Batched Modified Alignment-Length Synchronous Decoding (`BeamBatchedTDTInfer` / `malsd_batch`).
    MalsdBatch,
}

/// Per-call transcription options (mirrors NeMo's `RNNTDecodingConfig` + `transcribe()` kwargs).
#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    pub strategy: DecodingStrategy,
    /// Cap on non-blank tokens emitted per frame (`Some(10)` default; `None` = unlimited).
    pub max_symbols_per_step: Option<usize>,
    /// Logit temperature (`1.0` default).
    pub temperature: f32,
    /// Beam search settings (used when `strategy` is `Beam`, `Maes`, or `MalsdBatch`).
    pub beam: BeamConfig,
    /// Timestamp settings (tier selection, separators, optional gap threshold).
    pub timestamps: TimestampConfig,
    /// Record per-frame `(logits, label, duration)` alignments on `Transcript::alignments`.
    pub preserve_alignments: bool,
    /// Optional NeMo-compatible frame/token/word confidence configuration.
    pub confidence: ConfidenceConfig,
    /// Optional phrase context-biasing configuration (`BoostingTreeModelConfig`).
    /// For batch workloads, pre-compile with [`Transcriber::build_boosting_tree`] and pass
    /// via `boosting_tree_model` to avoid rebuilding per utterance.
    pub boosting_tree: Option<BoostingTreeConfig>,
    /// Pre-compiled boosting tree (takes precedence over `boosting_tree` if set).
    pub boosting_tree_model: Option<Arc<BoostingTree>>,
    /// Weight applied to boosting tree scores (`0.0` disables; e.g. `1.0` enables).
    pub boosting_tree_alpha: f32,
    /// Optional external ARPA n-gram language model file path.
    pub ngram_lm_path: Option<PathBuf>,
    /// Pre-loaded ARPA n-gram LM (takes precedence over `ngram_lm_path` if set).
    pub ngram_lm_model: Option<Arc<ArpaLm>>,
    /// Character offset used to decode ARPA BPE symbols (`Some(100)` matches NeMo default).
    pub ngram_lm_token_offset: Option<u32>,
    /// Weight applied to external n-gram LM scores (`0.0` disables; e.g. `0.3` enables).
    pub ngram_lm_alpha: f32,
    /// Multi-channel selection for [`Transcriber::transcribe_file_with`].
    pub channel_selector: ChannelSelector,
    /// Optional streaming carry-over state from a previous chunk (`partial_hypothesis`).
    pub partial_state: Option<DecoderState>,
    /// Populate `Transcript::decoder_state` so the next chunk can continue seamlessly.
    pub return_decoder_state: bool,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            strategy: DecodingStrategy::GreedyBatch,
            max_symbols_per_step: Some(tdt::MAX_SYMBOLS_PER_STEP),
            temperature: 1.0,
            beam: BeamConfig::default(),
            timestamps: TimestampConfig::default(),
            preserve_alignments: false,
            confidence: ConfidenceConfig::default(),
            boosting_tree: None,
            boosting_tree_model: None,
            boosting_tree_alpha: 0.0,
            ngram_lm_path: None,
            ngram_lm_model: None,
            ngram_lm_token_offset: Some(fusion::DEFAULT_TOKEN_OFFSET),
            ngram_lm_alpha: 0.0,
            channel_selector: ChannelSelector::Average,
            partial_state: None,
            return_decoder_state: false,
        }
    }
}

/// One utterance's output, the counterpart of NeMo's `Hypothesis`.
#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
    /// Accumulated hypothesis score (sum of selected token scores, matching NeMo's `Hypothesis.score`).
    pub score: f32,
    /// Encoder frames, i.e. 80 ms each.
    pub steps: usize,
    pub tokens: Vec<Token>,
    pub words: Vec<Word>,
    pub segments: Vec<Segment>,
    pub chars: Vec<CharTimestamp>,
    /// Per-frame alignment history when `preserve_alignments` is enabled.
    pub alignments: Option<Vec<FrameAlignment>>,
    /// Per-frame confidence scores when `confidence.preserve_frame_confidence` is enabled.
    pub frame_confidence: Option<Vec<FrameConfidence>>,
    /// Per-token confidence scores when `confidence.preserve_token_confidence` is enabled.
    pub token_confidence: Option<Vec<f32>>,
    /// Per-word confidence scores when `confidence.preserve_word_confidence` is enabled.
    pub word_confidence: Option<Vec<f32>>,
    /// Carry-over decoder state when `return_decoder_state` is enabled.
    pub decoder_state: Option<DecoderState>,
}

impl Transcript {
    /// The schema the batch command emits, so CLI and library output stay identical.
    pub fn to_json(&self) -> serde_json::Value {
        let ids: Vec<u32> = self.tokens.iter().map(|token| token.id).collect();
        let mut obj = serde_json::json!({
            "steps": self.steps,
            "text": self.text,
            "score": self.score,
            "token_ids": ids,
            "token_frames": self.tokens.iter().map(|t| t.frame).collect::<Vec<_>>(),
            "token_durations": self.tokens.iter().map(|t| t.duration).collect::<Vec<_>>(),
            "token_margins": self.tokens.iter().map(|t| t.duration_margin).collect::<Vec<_>>(),
            "words": self.words.iter().map(|w| serde_json::json!({
                "word": w.word,
                "start_offset": w.start_offset,
                "end_offset": w.end_offset,
                "start": w.start,
                "end": w.end,
            })).collect::<Vec<_>>(),
            "segments": self.segments.iter().map(|s| serde_json::json!({
                "segment": s.segment,
                "start_offset": s.start_offset,
                "end_offset": s.end_offset,
                "start": s.start,
                "end": s.end,
            })).collect::<Vec<_>>(),
            "chars": self.chars.iter().map(|c| serde_json::json!({
                "char": c.char,
                "start_offset": c.start_offset,
                "end_offset": c.end_offset,
                "start": c.start,
                "end": c.end,
            })).collect::<Vec<_>>(),
        });
        if let Some(alignments) = &self.alignments {
            obj["alignments"] = serde_json::json!(alignments.iter().map(|fa| {
                serde_json::json!({
                    "frame": fa.frame,
                    "labels": fa.steps.iter().map(|s| s.label).collect::<Vec<_>>(),
                    "durations": fa.steps.iter().map(|s| s.duration).collect::<Vec<_>>(),
                })
            }).collect::<Vec<_>>());
        }
        if let Some(fc) = &self.frame_confidence {
            obj["frame_confidence"] = serde_json::json!(fc.iter().map(|entry| {
                entry.values.iter().map(|v| match *v {
                    FrameConfidenceValue::Scalar(s) => serde_json::json!(s),
                    FrameConfidenceValue::WithDuration(p, d) => serde_json::json!([p, d]),
                }).collect::<Vec<_>>()
            }).collect::<Vec<_>>());
        }
        if let Some(tc) = &self.token_confidence {
            obj["token_confidence"] = serde_json::json!(tc);
        }
        if let Some(wc) = &self.word_confidence {
            obj["word_confidence"] = serde_json::json!(wc);
        }
        obj
    }
}

/// Portable stages.
struct CpuStages {
    pre: PreEncode,
    insert: FvInsert,
    encoder: Conformer,
    decoder: TdtDecoder,
}

/// The resident GPU pipeline: every stage's weights are uploaded once in
/// [`Transcriber::load`] and stay in VRAM; scratch buffers grow on demand with zero
/// weight re-upload.
struct Device {
    pre: crate::pre_encode_gpu::PreEncodeGpu,
    insert: crate::fv_insert_gpu::FvInsertGpu,
    encoder: crate::gpu_encoder::EncoderGpu,
    decoder: TdtDecoder,
}

enum Stages {
    Cpu(Box<CpuStages>),
    /// Fully loaded at [`Transcriber::load`]; scratch resizes automatically per clip.
    Gpu(Box<Device>),
}

/// A loaded model. Weights are read once and reused across utterances.
pub struct Transcriber {
    store: Arc<Store>,
    frontend: MelFrontend,
    tokenizer: Tokenizer,
    scale: f32,
    backend: Backend,
    clean_mode: bool,
    default_options: TranscribeOptions,
    stages: Stages,
}

impl Transcriber {
    /// Opens a `.nemo` archive and prepares the stages for `backend`.
    pub fn load(nemo_path: &Path, backend: Backend) -> Result<Self> {
        Self::load_with(nemo_path, LoadOptions::new(backend))
    }

    /// Opens a `.nemo` archive with explicit [`LoadOptions`] (such as `clean_mode`).
    pub fn load_with(nemo_path: &Path, options: LoadOptions) -> Result<Self> {
        let mut bundle = nemo::open(nemo_path)?;
        if options.clean_mode {
            let rank = bundle
                .fluid_speech
                .rank
                .context("clean_mode requested, but model_config.yaml has no fluid_speech.clean_mode.rank")?;
            let alpha = bundle
                .fluid_speech
                .alpha
                .context("clean_mode requested, but model_config.yaml has no fluid_speech.clean_mode.alpha")?;
            if rank == 0 {
                bail!("invalid clean_mode rank 0");
            }
            bundle.store.enable_clean_mode(alpha / rank as f32)?;
        }
        let frontend = MelFrontend::from_store(&bundle.store)?;
        let tokenizer = Tokenizer::from_model(&bundle.tokenizer_model, &bundle.vocab)?;
        let scale = 1.0 / (crate::conformer::HEAD_DIM as f32).sqrt();
        let store = Arc::new(bundle.store);

        options.attention_model.validate()?;
        let stages = match options.backend {
            Backend::Cpu => {
                let mut encoder = Conformer::from_store(&store)?;
                encoder.set_attention_model(options.attention_model)?;
                Stages::Cpu(Box::new(CpuStages {
                    pre: PreEncode::from_store(&store)?,
                    insert: FvInsert::from_store(&store)?,
                    encoder,
                    decoder: TdtDecoder::from_store(&store)?,
                }))
            }
            Backend::Gpu => {
                let gpu = Arc::new(crate::gpu::GpuGemm::new_on(&options.device_selector)?);
                let pre = crate::pre_encode_gpu::PreEncodeGpu::new(Arc::clone(&gpu), &store)?;
                let insert = crate::fv_insert_gpu::FvInsertGpu::new(
                    Arc::clone(&gpu),
                    GPU_INITIAL_CAPACITY_STEPS,
                    &store,
                )?;
                let encoder = crate::gpu_encoder::EncoderGpu::with_gpu_and_model(
                    Arc::clone(&gpu),
                    GPU_INITIAL_CAPACITY_STEPS,
                    &store,
                    options.attention_model,
                )?;
                let mut decoder = TdtDecoder::from_store(&store)?;
                decoder.enable_gpu_with(Arc::clone(&gpu))?;
                gpu.save_pipeline_cache();
                Stages::Gpu(Box::new(Device {
                    pre,
                    insert,
                    encoder,
                    decoder,
                }))
            }
        };

        Ok(Self {
            store,
            frontend,
            tokenizer,
            scale,
            backend: options.backend,
            clean_mode: options.clean_mode,
            default_options: TranscribeOptions::default(),
            stages,
        })
    }

    /// Which backend this instance runs on.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Whether Fluid Speech Pro's edited `clean_mode` adapter was folded at load time.
    pub fn clean_mode(&self) -> bool {
        self.clean_mode
    }

    /// What the loaded stages run on: `"cpu"`, or the adapter's one-line description
    /// (`"NVIDIA GeForce GTX 1080 (Vulkan, DiscreteGpu)"`).
    pub fn device_description(&self) -> String {
        match &self.stages {
            Stages::Cpu(_) => "cpu".to_string(),
            Stages::Gpu(device) => device.encoder.gpu().adapter_description().to_string(),
        }
    }

    /// Current Conformer self-attention model (`RelPos` or `RelPosLocalAttn`).
    pub fn attention_model(&self) -> AttentionModel {
        match &self.stages {
            Stages::Cpu(cpu) => cpu.encoder.attention_model(),
            Stages::Gpu(device) => device.encoder.attention_model(),
        }
    }

    /// Switches between full `rel_pos` and sliding-window `rel_pos_local_attn` at runtime
    /// with zero weight re-upload. Mirrors NeMo's `ConformerEncoder.change_attention_model`.
    pub fn change_attention_model(&mut self, model: AttentionModel) -> Result<()> {
        match &mut self.stages {
            Stages::Cpu(cpu) => cpu.encoder.set_attention_model(model),
            Stages::Gpu(device) => device.encoder.set_attention_model(model),
        }
    }

    /// Default options used by [`Self::transcribe`] and [`Self::transcribe_file`].
    pub fn default_options(&self) -> &TranscribeOptions {
        &self.default_options
    }

    /// Updates default options (mirrors NeMo's `change_decoding_strategy`).
    pub fn set_default_options(&mut self, options: TranscribeOptions) {
        self.default_options = options;
    }

    /// Pre-compiles a phrase [`BoostingTree`] using this model's SentencePiece tokenizer
    /// so it can be reused across utterances with zero rebuild overhead.
    pub fn build_boosting_tree(&self, cfg: &BoostingTreeConfig) -> Result<Arc<BoostingTree>> {
        Ok(Arc::new(BoostingTree::from_config(cfg, &self.tokenizer)?))
    }

    /// Loads an external ARPA n-gram language model for shallow fusion (`token_offset = Some(100)`
    /// matches NeMo's default character-offset BPE ARPA format).
    pub fn load_arpa_lm(&self, path: &Path, token_offset: Option<u32>) -> Result<Arc<ArpaLm>> {
        Ok(Arc::new(ArpaLm::from_file(
            path,
            token_offset,
            Some(&self.tokenizer),
        )?))
    }

    /// Encodes `text` with this model's SentencePiece BPE tokenizer.
    pub fn encode_text(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    fn resolve_fusion_models(
        &self,
        options: &TranscribeOptions,
    ) -> Result<(Option<Arc<ArpaLm>>, Option<Arc<BoostingTree>>)> {
        let lm = if options.ngram_lm_alpha != 0.0 {
            if let Some(m) = &options.ngram_lm_model {
                Some(Arc::clone(m))
            } else if let Some(path) = &options.ngram_lm_path {
                Some(self.load_arpa_lm(path, options.ngram_lm_token_offset)?)
            } else {
                None
            }
        } else {
            None
        };
        let bt = if options.boosting_tree_alpha != 0.0 {
            if let Some(m) = &options.boosting_tree_model {
                Some(Arc::clone(m))
            } else if let Some(cfg) = &options.boosting_tree {
                if cfg.is_empty() {
                    None
                } else {
                    Some(self.build_boosting_tree(cfg)?)
                }
            } else {
                None
            }
        } else {
            None
        };
        Ok((lm, bt))
    }

    /// Transcribes 16 kHz mono samples using [`Self::default_options`].
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<Transcript> {
        let options = self.default_options.clone();
        self.transcribe_with(samples, &options)
    }

    /// Transcribes 16 kHz mono samples with explicit [`TranscribeOptions`], returning the best hypothesis.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<Transcript> {
        let mut hyps = self.transcribe_nbest(samples, options)?;
        Ok(hyps.remove(0))
    }

    /// Transcribes 16 kHz mono samples and returns all ranked N-best hypotheses
    /// (for greedy strategies, returns a 1-element list).
    pub fn transcribe_nbest(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<Vec<Transcript>> {
        if samples.is_empty() {
            bail!("empty audio");
        }
        let mel = self.frontend.compute(samples)?;
        let frames = self.frontend.frames(samples.len());
        let bins = mel.len() / frames;
        let mut time_major = vec![0.0f32; frames * bins];
        for bin in 0..bins {
            for frame in 0..frames {
                time_major[frame * bins + bin] = mel[bin * frames + frame];
            }
        }

        // A long utterance would build device scratch this driver cannot drive
        // reliably, so it takes the CPU path for the whole utterance.
        if matches!(self.stages, Stages::Gpu(_)) && self.stages_steps(frames)? > GPU_MAX_VERIFIED_STEPS {
            let attn_model = self.attention_model();
            let mut encoder = Conformer::from_store(&self.store)?;
            encoder.set_attention_model(attn_model)?;
            self.stages = Stages::Cpu(Box::new(CpuStages {
                pre: PreEncode::from_store(&self.store)?,
                insert: FvInsert::from_store(&self.store)?,
                encoder,
                decoder: TdtDecoder::from_store(&self.store)?,
            }));
        }

        let (lm_arc, bt_arc) = self.resolve_fusion_models(options)?;
        let mut fusion_refs: Vec<(&dyn FusionModel, f32)> = Vec::new();
        if let Some(lm) = &lm_arc {
            fusion_refs.push((lm.as_ref(), options.ngram_lm_alpha));
        }
        if let Some(bt) = &bt_arc {
            fusion_refs.push((bt.as_ref(), options.boosting_tree_alpha));
        }

        let scale = self.scale;
        match options.strategy {
            DecodingStrategy::Greedy | DecodingStrategy::GreedyBatch => {
                if options.strategy == DecodingStrategy::Greedy && !fusion_refs.is_empty() {
                    bail!("strategy `greedy` does not support LM/boosting fusion; use `greedy_batch`");
                }
                let greedy_cfg = GreedyConfig {
                    max_symbols_per_step: options.max_symbols_per_step,
                    temperature: options.temperature,
                    preserve_alignments: options.preserve_alignments,
                    confidence: options.confidence.clone(),
                };
                let (out, steps): (GreedyOutput, usize) = match &mut self.stages {
                    Stages::Cpu(cpu) => {
                        let (mut features, valid) = cpu.pre.forward(&time_major, frames)?;
                        let pre_steps = features.len() / crate::pre_encode::D_MODEL;
                        cpu.insert.apply(&mut features, pre_steps)?;
                        features.truncate(valid * crate::pre_encode::D_MODEL);
                        let encoded = cpu.encoder.forward(&features, valid)?;
                        let out = cpu.decoder.decode_greedy(
                            &encoded,
                            valid,
                            &greedy_cfg,
                            &fusion_refs,
                            options.partial_state.as_ref(),
                            options.return_decoder_state,
                        )?;
                        (out, valid)
                    }
                    Stages::Gpu(device) => {
                        let (feat, valid) = device.pre.forward_resident(&time_major, frames)?;
                        let pre_steps = crate::pre_encode::total_steps(frames);
                        let feat = feat.clone();
                        device.insert.ensure_capacity(pre_steps);
                        device.encoder.ensure_capacity(valid)?;
                        device.insert.submit_apply_at(&feat, pre_steps);
                        let features = device.insert.output().clone();
                        let resident = device
                            .encoder
                            .encode_resident_at(&features, valid, scale)
                            .clone();
                        let out = device.decoder.decode_gpu_greedy(
                            &resident,
                            valid,
                            &greedy_cfg,
                            &fusion_refs,
                            options.partial_state.as_ref(),
                            options.return_decoder_state,
                        )?;
                        (out, valid)
                    }
                };
                Ok(vec![self.render(
                    out,
                    steps,
                    &options.timestamps,
                    &options.confidence,
                )])
            }
            DecodingStrategy::Beam | DecodingStrategy::Maes | DecodingStrategy::MalsdBatch => {
                if options.preserve_alignments {
                    bail!("preserve_alignments is not supported for beam decoding in NeMo TDT");
                }
                if options.confidence.enabled() {
                    bail!("confidence calculation is only supported for greedy / greedy_batch in NeMo TDT");
                }
                if options.partial_state.is_some() {
                    bail!("partial_state is not supported for beam decoding in NeMo TDT");
                }
                if options.strategy == DecodingStrategy::Beam && !fusion_refs.is_empty() {
                    bail!("strategy `beam` does not support LM/boosting fusion; use `malsd_batch`");
                }
                if options.strategy == DecodingStrategy::Maes && bt_arc.is_some() {
                    bail!("strategy `maes` does not support boosting tree; use `malsd_batch`");
                }
                let maes_lm = lm_arc
                    .as_ref()
                    .map(|lm| (lm.as_ref() as &dyn FusionModel, options.ngram_lm_alpha));
                let (hyps, steps): (Vec<BeamHypothesis>, usize) = match &mut self.stages {
                    Stages::Cpu(cpu) => {
                        let (mut features, valid) = cpu.pre.forward(&time_major, frames)?;
                        let pre_steps = features.len() / crate::pre_encode::D_MODEL;
                        cpu.insert.apply(&mut features, pre_steps)?;
                        features.truncate(valid * crate::pre_encode::D_MODEL);
                        let encoded = cpu.encoder.forward(&features, valid)?;
                        let enc_proj = cpu.decoder.project_encoder_cpu(&encoded, valid)?;
                        let hyps = run_beam_strategy(
                            &cpu.decoder,
                            &enc_proj,
                            valid,
                            options.strategy,
                            &options.beam,
                            &fusion_refs,
                            maes_lm,
                        )?;
                        (hyps, valid)
                    }
                    Stages::Gpu(device) => {
                        let (feat, valid) = device.pre.forward_resident(&time_major, frames)?;
                        let pre_steps = crate::pre_encode::total_steps(frames);
                        let feat = feat.clone();
                        device.insert.ensure_capacity(pre_steps);
                        device.encoder.ensure_capacity(valid)?;
                        device.insert.submit_apply_at(&feat, pre_steps);
                        let features = device.insert.output().clone();
                        let resident = device
                            .encoder
                            .encode_resident_at(&features, valid, scale)
                            .clone();
                        let enc_proj = device.decoder.project_encoder_gpu(&resident, valid)?;
                        let hyps = run_beam_strategy(
                            &device.decoder,
                            &enc_proj,
                            valid,
                            options.strategy,
                            &options.beam,
                            &fusion_refs,
                            maes_lm,
                        )?;
                        (hyps, valid)
                    }
                };
                Ok(hyps
                    .into_iter()
                    .map(|h| {
                        self.render(
                            GreedyOutput {
                                tokens: h.tokens,
                                score: h.score,
                                alignments: None,
                                frame_confidence: None,
                                token_confidence: None,
                                state: None,
                            },
                            steps,
                            &options.timestamps,
                            &ConfidenceConfig::default(),
                        )
                    })
                    .collect())
            }
        }
    }

    /// Frame count the device path would use, without keeping the result.
    fn stages_steps(&mut self, frames: usize) -> Result<usize> {
        match &mut self.stages {
            Stages::Cpu(_) => Ok(0),
            Stages::Gpu(_) => Ok(crate::pre_encode::valid_steps(frames)),
        }
    }

    /// Transcribes a WAV file, converting to 16 kHz mono if needed.
    pub fn transcribe_file(&mut self, wav_path: &Path) -> Result<Transcript> {
        let options = self.default_options.clone();
        self.transcribe_file_with(wav_path, &options)
    }

    /// Transcribes a WAV file with explicit [`TranscribeOptions`] (including `channel_selector`).
    pub fn transcribe_file_with(
        &mut self,
        wav_path: &Path,
        options: &TranscribeOptions,
    ) -> Result<Transcript> {
        let samples = read_wav_16k(wav_path, &options.channel_selector)?;
        self.transcribe_with(&samples, options)
    }

    /// Transcribes a WAV file and returns all ranked N-best hypotheses.
    pub fn transcribe_file_nbest(
        &mut self,
        wav_path: &Path,
        options: &TranscribeOptions,
    ) -> Result<Vec<Transcript>> {
        let samples = read_wav_16k(wav_path, &options.channel_selector)?;
        self.transcribe_nbest(&samples, options)
    }

    /// Transcribes a slice of 16 kHz mono utterances, reusing resident weights and scratch.
    pub fn transcribe_batch(
        &mut self,
        clips: &[&[f32]],
        options: &TranscribeOptions,
    ) -> Result<Vec<Transcript>> {
        let mut out = Vec::with_capacity(clips.len());
        for clip in clips {
            out.push(self.transcribe_with(clip, options)?);
        }
        Ok(out)
    }

    /// Transcribes a list of WAV files, reusing resident weights and scratch.
    pub fn transcribe_files(
        &mut self,
        paths: &[&Path],
        options: &TranscribeOptions,
    ) -> Result<Vec<Transcript>> {
        let mut out = Vec::with_capacity(paths.len());
        for path in paths {
            out.push(self.transcribe_file_with(path, options)?);
        }
        Ok(out)
    }

    fn render(
        &self,
        out: GreedyOutput,
        steps: usize,
        ts: &TimestampConfig,
        conf: &ConfidenceConfig,
    ) -> Transcript {
        let ids: Vec<u32> = out.tokens.iter().map(|token| token.id).collect();
        let text = self.tokenizer.decode(&ids);
        let need_words = matches!(
            ts.timestamp_type,
            TimestampType::Word | TimestampType::Segment | TimestampType::All
        );
        let need_segments = matches!(ts.timestamp_type, TimestampType::Segment | TimestampType::All);
        let need_chars = matches!(ts.timestamp_type, TimestampType::Char | TimestampType::All);

        let computed_words = if need_words {
            text::words(&self.tokenizer, &out.tokens)
        } else {
            Vec::new()
        };
        let segments = if need_segments {
            text::segments_with(
                &computed_words,
                &ts.segment_separators,
                ts.segment_gap_threshold,
            )
        } else {
            Vec::new()
        };
        let words = if matches!(ts.timestamp_type, TimestampType::Word | TimestampType::All) {
            computed_words
        } else {
            Vec::new()
        };
        let chars = if need_chars {
            text::chars(&self.tokenizer, &out.tokens)
        } else {
            Vec::new()
        };
        let word_confidence = if conf.preserve_word_confidence {
            out.token_confidence.as_deref().map(|tc| {
                confidence::compute_word_confidence(&self.tokenizer, &ids, tc, conf.aggregation)
            })
        } else {
            None
        };
        let token_confidence = if conf.preserve_token_confidence {
            out.token_confidence
        } else {
            None
        };
        Transcript {
            text,
            score: out.score,
            steps,
            tokens: out.tokens,
            words,
            segments,
            chars,
            alignments: out.alignments,
            frame_confidence: out.frame_confidence,
            token_confidence,
            word_confidence,
            decoder_state: out.state,
        }
    }
}

/// Reads a WAV as 16 kHz mono, averaging channels and linearly resampling.
pub fn read_wav_mono_16k(path: &Path) -> Result<Vec<f32>> {
    read_wav_16k(path, &ChannelSelector::Average)
}

fn run_beam_strategy(
    decoder: &TdtDecoder,
    enc_proj: &[f32],
    steps: usize,
    strategy: DecodingStrategy,
    cfg: &BeamConfig,
    fusion: &[(&dyn FusionModel, f32)],
    maes_lm: Option<(&dyn FusionModel, f32)>,
) -> Result<Vec<BeamHypothesis>> {
    match strategy {
        DecodingStrategy::Beam => beam::decode_tdt_beam(decoder, enc_proj, steps, cfg),
        DecodingStrategy::Maes => beam::decode_tdt_maes(decoder, enc_proj, steps, cfg, maes_lm),
        DecodingStrategy::MalsdBatch => {
            beam::decode_tdt_malsd_batch(decoder, enc_proj, steps, cfg, fusion)
        }
        DecodingStrategy::Greedy | DecodingStrategy::GreedyBatch => {
            bail!("run_beam_strategy called with greedy strategy")
        }
    }
}

/// Reads a WAV as 16 kHz mono using `channel_selector`, linearly resampling when needed.
pub fn read_wav_16k(path: &Path, channel_selector: &ChannelSelector) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 * scale))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let channels = spec.channels as usize;
    if channels == 0 {
        bail!("{} has zero channels", path.display());
    }
    let mono: Vec<f32> = match channel_selector {
        ChannelSelector::Average => {
            if channels == 1 {
                samples
            } else {
                samples
                    .chunks(channels)
                    .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
                    .collect()
            }
        }
        ChannelSelector::Channel(ch) => {
            if *ch >= channels {
                bail!(
                    "channel_selector={ch} out of range for {} ({channels} channels)",
                    path.display()
                );
            }
            if channels == 1 {
                samples
            } else {
                samples.chunks(channels).map(|frame| frame[*ch]).collect()
            }
        }
        ChannelSelector::Channels(list) => {
            if list.is_empty() {
                bail!("channel_selector list must not be empty");
            }
            for &ch in list {
                if ch >= channels {
                    bail!(
                        "channel_selector includes channel {ch}, out of range for {} ({channels} channels)",
                        path.display()
                    );
                }
            }
            let denom = list.len() as f32;
            samples
                .chunks(channels)
                .map(|frame| list.iter().map(|&ch| frame[ch]).sum::<f32>() / denom)
                .collect()
        }
    };
    let target = mel::SAMPLE_RATE;
    if spec.sample_rate as usize == target {
        return Ok(mono);
    }
    let ratio = spec.sample_rate as f64 / target as f64;
    let out_len = ((mono.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for index in 0..out_len {
        let position = index as f64 * ratio;
        let left = position.floor() as usize;
        let frac = (position - left as f64) as f32;
        let a = mono.get(left).copied().unwrap_or(0.0);
        let b = mono.get(left + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_selector_extracts_and_averages_stereo_wav() {
        let dir = std::env::temp_dir();
        let path = dir.join("fsp_test_channel_selector.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        {
            let mut writer = hound::WavWriter::create(&path, spec).unwrap();
            for &(l, r) in &[(0.2f32, 0.8f32), (-0.4, 0.4), (1.0, 0.0)] {
                writer.write_sample(l).unwrap();
                writer.write_sample(r).unwrap();
            }
            writer.finalize().unwrap();
        }
        let avg = read_wav_16k(&path, &ChannelSelector::Average).unwrap();
        let c0 = read_wav_16k(&path, &ChannelSelector::Channel(0)).unwrap();
        let c1 = read_wav_16k(&path, &ChannelSelector::Channel(1)).unwrap();
        let both = read_wav_16k(&path, &ChannelSelector::Channels(vec![0, 1])).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(c0, vec![0.2, -0.4, 1.0]);
        assert_eq!(c1, vec![0.8, 0.4, 0.0]);
        for (a, b) in avg.iter().zip([0.5f32, 0.0, 0.5]) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(avg, both);
    }

    #[test]
    fn attention_model_validation_rejects_zero_window() {
        assert!(AttentionModel::RelPos.validate().is_ok());
        assert!(AttentionModel::RelPosLocalAttn { left: 64, right: 64 }.validate().is_ok());
        assert!(AttentionModel::RelPosLocalAttn { left: 64, right: 0 }.validate().is_ok());
        assert!(AttentionModel::RelPosLocalAttn { left: 0, right: 0 }.validate().is_err());
    }
}
