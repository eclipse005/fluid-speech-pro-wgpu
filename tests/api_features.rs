//! End-to-end public-API tests exercising the crate as an external consumer (default features only).

use std::path::Path;

use fluid_speech_pro::api::{
    AttentionModel, Backend, BeamConfig, BoostingTreeConfig, ChannelSelector, ConfidenceConfig,
    DecodingStrategy, LoadOptions, TimestampConfig, TimestampType, Transcriber, TranscribeOptions,
};

const NEMO_PATH: &str = "../nemo/fluid-speech-pro.nemo";
const CLIP_EN: &str = "../test_data/clip_en.wav";

/// Verifies runtime `change_attention_model` (`RelPos -> Local -> RelPos`) without weight reload,
/// `preserve_alignments`, `TimestampType` tiers, `partial_state` streaming carry-over,
/// `transcribe_batch`, and `BoostingTree` / `ArpaLm` public builder methods on GPU.
#[test]
#[ignore = "requires local ../nemo/fluid-speech-pro.nemo and ../test_data/clip_en.wav"]
fn api_full_feature_smoke_gpu() {
    let nemo = Path::new(NEMO_PATH);
    let wav = Path::new(CLIP_EN);
    assert!(nemo.exists(), "missing {NEMO_PATH}");
    assert!(wav.exists(), "missing {CLIP_EN}");

    let mut asr = Transcriber::load_with(nemo, LoadOptions::new(Backend::Gpu)).unwrap();
    assert_eq!(asr.attention_model(), AttentionModel::RelPos);

    // 1. Baseline RelPos with alignments & confidence.
    let mut opts = TranscribeOptions::default();
    opts.preserve_alignments = true;
    opts.confidence = ConfidenceConfig {
        preserve_frame_confidence: true,
        preserve_token_confidence: true,
        preserve_word_confidence: true,
        ..Default::default()
    };
    let base = asr.transcribe_file_with(wav, &opts).unwrap();
    assert_eq!(base.tokens.len(), 54);
    assert!(base.alignments.as_ref().is_some_and(|a| !a.is_empty()));
    assert!(base.frame_confidence.as_ref().is_some_and(|c| !c.is_empty()));
    assert_eq!(base.token_confidence.as_ref().map(|v| v.len()), Some(54));
    assert_eq!(
        base.word_confidence.as_ref().map(|v| v.len()),
        Some(base.words.len())
    );

    // 2. TimestampType tiers.
    let mut ts_none = TranscribeOptions::default();
    ts_none.timestamps = TimestampConfig {
        timestamp_type: TimestampType::None,
        ..Default::default()
    };
    let r_none = asr.transcribe_file_with(wav, &ts_none).unwrap();
    assert_eq!(r_none.text, base.text);
    assert!(r_none.words.is_empty() && r_none.segments.is_empty() && r_none.chars.is_empty());

    // 3. Runtime change_attention_model: RelPos -> RelPosLocalAttn(64,64) -> RelPos (zero reload).
    asr.change_attention_model(AttentionModel::RelPosLocalAttn {
        left: 64,
        right: 64,
    })
    .unwrap();
    assert_eq!(
        asr.attention_model(),
        AttentionModel::RelPosLocalAttn {
            left: 64,
            right: 64
        }
    );
    let local_res = asr.transcribe_file(wav).unwrap();
    assert_eq!(local_res.tokens.len(), 54);
    assert!((local_res.score - (-599.9068)).abs() < 1e-2);

    asr.change_attention_model(AttentionModel::RelPos).unwrap();
    let back = asr.transcribe_file(wav).unwrap();
    assert_eq!(back.text, base.text);
    assert_eq!(
        back.tokens.iter().map(|t| t.id).collect::<Vec<_>>(),
        base.tokens.iter().map(|t| t.id).collect::<Vec<_>>()
    );

    // 4. Streaming carry-over (`return_decoder_state` -> `partial_state`).
    let samples = fluid_speech_pro::api::read_wav_16k(wav, &ChannelSelector::Average).unwrap();
    let mid = samples.len() / 2;
    let mut s1_opts = TranscribeOptions::default();
    s1_opts.return_decoder_state = true;
    let chunk1 = asr.transcribe_with(&samples[..mid], &s1_opts).unwrap();
    let state1 = chunk1
        .decoder_state
        .clone()
        .expect("decoder_state should be populated");
    assert!(state1.decoded_frames > 0);

    let mut s2_opts = TranscribeOptions::default();
    s2_opts.partial_state = Some(state1.clone());
    s2_opts.return_decoder_state = true;
    let chunk2 = asr.transcribe_with(&samples[mid..], &s2_opts).unwrap();
    assert!(!chunk2.tokens.is_empty());
    // Timestamps in chunk 2 must be offset by chunk 1's decoded_frames.
    assert!(chunk2.tokens[0].frame >= state1.decoded_frames);
    let state2 = chunk2.decoder_state.expect("chunk2 decoder_state");
    assert!(state2.decoded_frames > state1.decoded_frames);

    // 5. BoostingTree + malsd_batch via public API.
    let tree = asr
        .build_boosting_tree(&BoostingTreeConfig {
            key_phrases: vec!["Victoria Harbour".to_string()],
            ..Default::default()
        })
        .unwrap();
    let mut beam_opts = TranscribeOptions::default();
    beam_opts.strategy = DecodingStrategy::MalsdBatch;
    beam_opts.beam = BeamConfig {
        beam_size: 4,
        ..Default::default()
    };
    beam_opts.boosting_tree_model = Some(tree);
    beam_opts.boosting_tree_alpha = 1.0;
    let hyps = asr.transcribe_file_nbest(wav, &beam_opts).unwrap();
    assert!(!hyps.is_empty());
    assert!(
        hyps[0].text.contains("Victoria Harbour"),
        "expected boosted spelling, got: {}",
        hyps[0].text
    );

    // 6. transcribe_batch reuses resident weights/scratch.
    let batch = asr
        .transcribe_batch(&[&samples[..], &samples[..mid]], &TranscribeOptions::default())
        .unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].text, base.text);
}
