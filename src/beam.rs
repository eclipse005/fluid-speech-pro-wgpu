//! TDT beam search decoders: `beam` (default), `maes`, and `malsd_batch`.
//!
//! Mirrors NeMo 2.7.3's `tdt_beam_decoding.py` (`BeamTDTInfer::default_beam_search`,
//! `BeamTDTInfer::modified_adaptive_expansion_search`) and
//! `tdt_malsd_batched_computer.py` (`ModifiedALSDBatchedTDTComputer::modified_alsd_torch`)
//! + `batched_beam_decoding_utils.py` (`BatchedBeamHyps`).
//!
//! Every beam algorithm consumes pre-projected encoder frames `[steps][640]` (including
//! `enc_bias`), so both CPU and GPU encoders feed the exact same beam search engine
//! with zero code duplication.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};

use crate::fusion::{FusionModel, NON_BLANK_VOCAB};
use crate::tdt::{
    BLANK, DURATIONS, JOINT_HIDDEN, JOINT_VOCAB, MAX_SYMBOLS_PER_STEP, PredictorState, TdtDecoder,
    Token, VOCAB,
};

const TDT_DURATIONS: [u32; DURATIONS] = [0, 1, 2, 3, 4];

/// Blank scoring mode when shallow-fusing external LM / boosting models in `malsd_batch`.
///
/// Mirrors NeMo's `BlankLMScoreMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlankLmScoreMode {
    /// Blank gets no LM score adjustment (`no_score`).
    NoScore,
    /// Blank score scaled by `1 + sum(alpha)`; non-blank adjusted by `log(1 - p_blank) * sum(alpha)`.
    #[default]
    LmWeightedFull,
}

/// Pruning order when shallow-fusing external LM / boosting models in `malsd_batch`.
///
/// Mirrors NeMo's `PruningMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PruningMode {
    /// Prune top-k on ASR log-probs first, then add fusion scores (`early`).
    Early,
    /// Add fusion scores over the full vocabulary before top-k (`late`).
    #[default]
    Late,
}

/// Configuration for all NeMo TDT beam search strategies (`beam`, `maes`, `malsd_batch`).
#[derive(Debug, Clone)]
pub struct BeamConfig {
    /// Number of beams (`>= 1`; NeMo default in Fluid Speech Pro `model_config.yaml` is `2`).
    pub beam_size: usize,
    /// Normalize scores by `len(tokens) + 1` when ranking hypotheses (`true` default).
    pub score_norm: bool,
    /// When `true`, `transcribe_with` returns the top-1 hypothesis; call `transcribe_nbest`
    /// to obtain all ranked N-best hypotheses regardless of this flag.
    pub return_best_hypothesis: bool,
    /// Per-frame non-blank symbol cap for `malsd_batch` (`Some(10)` default).
    pub max_symbols_per_step: Option<usize>,
    /// Logit temperature for `beam` and `maes` (`1.0` default).
    pub softmax_temperature: f32,
    /// Number of adaptive expansion steps per frame in `maes` (`2` default).
    pub maes_num_steps: usize,
    /// Max prefix length difference for prefix search in `maes` (`1` default).
    pub maes_prefix_alpha: usize,
    /// Prune-by-value log-prob margin in `maes` (`2.3` default).
    pub maes_expansion_gamma: f32,
    /// Extra expansion candidates beyond `beam_size` in `maes` (`2` default).
    pub maes_expansion_beta: usize,
    /// Blank LM scoring mode for `malsd_batch` with fusion models (`LmWeightedFull` default).
    pub blank_lm_score_mode: BlankLmScoreMode,
    /// Pruning mode for `malsd_batch` with fusion models (`Late` default).
    pub pruning_mode: PruningMode,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam_size: 2,
            score_norm: true,
            return_best_hypothesis: true,
            max_symbols_per_step: Some(MAX_SYMBOLS_PER_STEP),
            softmax_temperature: 1.0,
            maes_num_steps: 2,
            maes_prefix_alpha: 1,
            maes_expansion_gamma: 2.3,
            maes_expansion_beta: 2,
            blank_lm_score_mode: BlankLmScoreMode::LmWeightedFull,
            pruning_mode: PruningMode::Late,
        }
    }
}

/// One decoded beam hypothesis before detokenization / timestamp rendering.
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    pub tokens: Vec<Token>,
    pub score: f32,
}

fn validate_beam_cfg(cfg: &BeamConfig) -> Result<()> {
    if cfg.beam_size < 1 {
        bail!("beam_size must be >= 1, got {}", cfg.beam_size);
    }
    if cfg.softmax_temperature <= 0.0 || !cfg.softmax_temperature.is_finite() {
        bail!(
            "softmax_temperature must be > 0 and finite, got {}",
            cfg.softmax_temperature
        );
    }
    Ok(())
}

fn log_softmax_slice(logits: &[f32], inv_temp: f32, out: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        let s = v * inv_temp;
        if s > max {
            max = s;
        }
    }
    let mut sum = 0.0f64;
    for &v in logits {
        sum += ((v * inv_temp - max) as f64).exp();
    }
    let log_z = max as f64 + sum.ln();
    for (dst, &v) in out.iter_mut().zip(logits) {
        *dst = ((v * inv_temp) as f64 - log_z) as f32;
    }
}

#[inline]
fn logaddexp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a >= b { (a as f64, b as f64) } else { (b as f64, a as f64) };
    (hi + (lo - hi).exp().ln_1p()) as f32
}

/// Returns the top-`k` `(value, index)` pairs in descending value order (tie-break: lower index first).
fn topk_desc(values: &[f32], k: usize) -> Vec<(f32, usize)> {
    let k = k.min(values.len());
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for (idx, &val) in values.iter().enumerate() {
        if top.len() < k
            || val > top.last().unwrap().0
            || (val == top.last().unwrap().0 && idx < top.last().unwrap().1)
        {
            let pos = top
                .binary_search_by(|&(v, i)| {
                    val.total_cmp(&v).then_with(|| i.cmp(&idx))
                })
                .unwrap_or_else(|e| e);
            top.insert(pos, (val, idx));
            if top.len() > k {
                top.pop();
            }
        }
    }
    top
}

fn rank_score(score: f32, token_len: usize, score_norm: bool) -> f32 {
    if score_norm {
        score / (token_len as f32 + 1.0)
    } else {
        score
    }
}

fn make_token(id: u32, frame: u32, duration: u32, step_score: f32, dur_logits: &[f32]) -> Token {
    let mut raw = [0.0f32; DURATIONS];
    raw.copy_from_slice(&dur_logits[..DURATIONS]);
    let chosen = raw[duration as usize];
    let mut second = f32::NEG_INFINITY;
    for (i, &v) in raw.iter().enumerate() {
        if i != duration as usize && v > second {
            second = v;
        }
    }
    Token {
        id,
        frame,
        duration,
        duration_margin: chosen - second,
        duration_logits: raw,
        score: step_score,
    }
}

/// Prefix-cached predictor helper shared across beam hypotheses.
struct PredCache<'a> {
    decoder: &'a TdtDecoder,
    /// Keyed by full `y_sequence` (starting with `BLANK` SOS).
    cache: HashMap<Vec<u32>, Arc<PredictorState>>,
}

impl<'a> PredCache<'a> {
    fn new(decoder: &'a TdtDecoder) -> Self {
        Self {
            decoder,
            cache: HashMap::new(),
        }
    }

    fn init_sos(&mut self) -> Arc<PredictorState> {
        let key = vec![BLANK];
        let state = Arc::new(self.decoder.predict_from_state(None, BLANK));
        self.cache.insert(key, Arc::clone(&state));
        state
    }

    fn step(&mut self, parent_seq: &[u32], parent_state: &PredictorState, token: u32) -> Arc<PredictorState> {
        let mut key = Vec::with_capacity(parent_seq.len() + 1);
        key.extend_from_slice(parent_seq);
        key.push(token);
        if let Some(cached) = self.cache.get(&key) {
            return Arc::clone(cached);
        }
        let state = Arc::new(self.decoder.predict_from_state(Some(parent_state), token));
        self.cache.insert(key, Arc::clone(&state));
        state
    }
}

// ============================================================================
// 1. Default TDT Beam Search (`BeamTDTInfer::default_beam_search`)
// ============================================================================

#[derive(Clone)]
struct DefaultBeamHyp {
    score: f32,
    y_sequence: Vec<u32>, // starts with BLANK (SOS), matching NeMo
    tokens: Vec<Token>,
    state: Arc<PredictorState>,
    last_frame: usize,
}

fn merge_default_hyps(mut hyps: Vec<DefaultBeamHyp>) -> Vec<DefaultBeamHyp> {
    hyps.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut map: HashMap<(Vec<u32>, usize), usize> = HashMap::with_capacity(hyps.len());
    let mut kept: Vec<DefaultBeamHyp> = Vec::with_capacity(hyps.len());
    for h in hyps {
        let key = (h.y_sequence.clone(), h.last_frame);
        if let Some(&idx) = map.get(&key) {
            kept[idx].score = logaddexp(kept[idx].score, h.score);
        } else {
            map.insert(key, kept.len());
            kept.push(h);
        }
    }
    kept
}

/// NeMo's `BeamTDTInfer::default_beam_search` (`strategy = "beam"`).
pub(crate) fn decode_tdt_beam(
    decoder: &TdtDecoder,
    enc_proj: &[f32],
    steps: usize,
    cfg: &BeamConfig,
) -> Result<Vec<BeamHypothesis>> {
    validate_beam_cfg(cfg)?;
    if steps == 0 {
        return Ok(vec![BeamHypothesis {
            tokens: Vec::new(),
            score: 0.0,
        }]);
    }
    let beam = cfg.beam_size.min(NON_BLANK_VOCAB);
    let beam_k = beam.min(NON_BLANK_VOCAB - 1).max(1);
    let dur_k = beam.min(DURATIONS);
    let inv_temp = 1.0 / cfg.softmax_temperature;

    let mut pred_cache = PredCache::new(decoder);
    let sos_state = pred_cache.init_sos();

    let mut kept_hyps = vec![DefaultBeamHyp {
        score: 0.0,
        y_sequence: vec![BLANK],
        tokens: Vec::new(),
        state: sos_state,
        last_frame: 0,
    }];

    let mut projected = vec![0.0f32; JOINT_HIDDEN];
    let mut logits = vec![0.0f32; JOINT_VOCAB];
    let mut logp = vec![0.0f32; VOCAB];
    let mut dur_logp = vec![0.0f32; DURATIONS];

    for time_idx in 0..steps {
        let mut hyps = Vec::new();
        let mut future = Vec::new();
        for h in kept_hyps {
            if h.last_frame == time_idx {
                hyps.push(h);
            } else if h.last_frame > time_idx {
                future.push(h);
            }
        }
        kept_hyps = future;
        let enc_frame = &enc_proj[time_idx * JOINT_HIDDEN..(time_idx + 1) * JOINT_HIDDEN];

        while !hyps.is_empty() {
            // Pop highest-scoring hypothesis in `hyps`.
            let (max_idx, _) = hyps
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.score.total_cmp(&b.score))
                .unwrap();
            let max_hyp = hyps.swap_remove(max_idx);

            decoder.joint_from_projected(enc_frame, &max_hyp.state.pred_proj, &mut projected, &mut logits);
            log_softmax_slice(&logits[..VOCAB], inv_temp, &mut logp);
            log_softmax_slice(&logits[VOCAB..], inv_temp, &mut dur_logp);

            // Top-k non-blank tokens & top-k durations, then top `beam_k` (dur, tok) pairs.
            let tok_topk = topk_desc(&logp[..NON_BLANK_VOCAB], beam_k);
            let dur_topk = topk_desc(&dur_logp, dur_k);

            let mut pairs: Vec<(f32, usize, usize)> = Vec::with_capacity(dur_topk.len() * tok_topk.len());
            for &(dl, di) in &dur_topk {
                for &(tl, ti) in &tok_topk {
                    pairs.push((dl + tl, ti, di));
                }
            }
            pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
            pairs.truncate(beam_k);

            for (total_lp, token_idx, duration_idx) in pairs {
                let dur = TDT_DURATIONS[duration_idx];
                let next_state = pred_cache.step(&max_hyp.y_sequence, &max_hyp.state, token_idx as u32);
                let mut y_seq = max_hyp.y_sequence.clone();
                y_seq.push(token_idx as u32);
                let mut toks = max_hyp.tokens.clone();
                toks.push(make_token(
                    token_idx as u32,
                    time_idx as u32,
                    dur,
                    total_lp,
                    &logits[VOCAB..],
                ));
                let new_hyp = DefaultBeamHyp {
                    score: max_hyp.score + total_lp,
                    y_sequence: y_seq,
                    tokens: toks,
                    state: next_state,
                    last_frame: max_hyp.last_frame + dur as usize,
                };
                if dur == 0 {
                    hyps.push(new_hyp);
                } else {
                    kept_hyps.push(new_hyp);
                }
            }

            // Blank expansions (non-zero durations only).
            for &(_, mut duration_idx) in &dur_topk {
                if duration_idx == 0 {
                    if dur_topk.len() == 1 {
                        duration_idx = 1; // min non-zero duration
                    } else {
                        continue;
                    }
                }
                let dur = TDT_DURATIONS[duration_idx];
                let blank_lp = logp[BLANK as usize] + dur_logp[duration_idx];
                kept_hyps.push(DefaultBeamHyp {
                    score: max_hyp.score + blank_lp,
                    y_sequence: max_hyp.y_sequence.clone(),
                    tokens: max_hyp.tokens.clone(),
                    state: Arc::clone(&max_hyp.state),
                    last_frame: max_hyp.last_frame + dur as usize,
                });
            }

            kept_hyps = merge_default_hyps(kept_hyps);

            if !hyps.is_empty() {
                let hyps_max = hyps
                    .iter()
                    .map(|h| h.score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut kept_most_prob: Vec<DefaultBeamHyp> = kept_hyps
                    .iter()
                    .filter(|h| h.score > hyps_max)
                    .cloned()
                    .collect();
                if kept_most_prob.len() >= beam {
                    kept_most_prob.sort_by(|a, b| a.score.total_cmp(&b.score));
                    kept_hyps = kept_most_prob;
                    break;
                }
            } else {
                kept_hyps.sort_by(|a, b| b.score.total_cmp(&a.score));
                kept_hyps.truncate(beam);
            }
        }
    }

    kept_hyps.sort_by(|a, b| {
        rank_score(b.score, b.tokens.len(), cfg.score_norm)
            .total_cmp(&rank_score(a.score, a.tokens.len(), cfg.score_norm))
    });
    Ok(kept_hyps
        .into_iter()
        .map(|h| BeamHypothesis {
            tokens: h.tokens,
            score: h.score,
        })
        .collect())
}

// ============================================================================
// 2. Modified Adaptive Expansion Search (`BeamTDTInfer::modified_adaptive_expansion_search`)
// ============================================================================

#[derive(Clone)]
struct MaesHyp {
    score: f32,
    y_sequence: Vec<u32>, // starts with BLANK (SOS)
    tokens: Vec<Token>,
    /// History of `pred_proj` states matching `y_sequence` positions (needed by `prefix_search`).
    states: Vec<Arc<PredictorState>>,
    last_frame: usize,
    lm_state: usize,
}

fn is_strict_prefix(x: &[u32], pref: &[u32]) -> bool {
    pref.len() < x.len() && x[..pref.len()] == *pref
}

fn merge_maes_hyps(mut hyps: Vec<MaesHyp>) -> Vec<MaesHyp> {
    hyps.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut map: HashMap<(Vec<u32>, usize), usize> = HashMap::with_capacity(hyps.len());
    let mut kept: Vec<MaesHyp> = Vec::with_capacity(hyps.len());
    for h in hyps {
        let key = (h.y_sequence.clone(), h.last_frame);
        if let Some(&idx) = map.get(&key) {
            kept[idx].score = logaddexp(kept[idx].score, h.score);
        } else {
            map.insert(key, kept.len());
            kept.push(h);
        }
    }
    kept
}

/// NeMo's `BeamTDTInfer::modified_adaptive_expansion_search` (`strategy = "maes"`).
pub(crate) fn decode_tdt_maes(
    decoder: &TdtDecoder,
    enc_proj: &[f32],
    steps: usize,
    cfg: &BeamConfig,
    ngram_lm: Option<(&dyn FusionModel, f32)>,
) -> Result<Vec<BeamHypothesis>> {
    validate_beam_cfg(cfg)?;
    if cfg.maes_num_steps < 1 {
        bail!("maes_num_steps must be >= 1, got {}", cfg.maes_num_steps);
    }
    if steps == 0 {
        return Ok(vec![BeamHypothesis {
            tokens: Vec::new(),
            score: 0.0,
        }]);
    }
    let beam = cfg.beam_size.min(NON_BLANK_VOCAB);
    let max_candidates = (beam + cfg.maes_expansion_beta).min(VOCAB);
    let inv_temp = 1.0 / cfg.softmax_temperature;

    let mut pred_cache = PredCache::new(decoder);
    let sos_state = pred_cache.init_sos();
    let init_lm = ngram_lm.map_or(0, |(lm, _)| lm.init_state());

    let mut kept_hyps = vec![MaesHyp {
        score: 0.0,
        y_sequence: vec![BLANK],
        tokens: Vec::new(),
        states: vec![sos_state],
        last_frame: 0,
        lm_state: init_lm,
    }];

    let mut projected = vec![0.0f32; JOINT_HIDDEN];
    let mut logits = vec![0.0f32; JOINT_VOCAB];
    let mut logp = vec![0.0f32; VOCAB];
    let mut dur_logp = vec![0.0f32; DURATIONS];

    for time_idx in 0..steps {
        let mut hyps = Vec::new();
        let mut future = Vec::new();
        for h in kept_hyps {
            if h.last_frame == time_idx {
                hyps.push(h);
            } else if h.last_frame > time_idx {
                future.push(h);
            }
        }
        kept_hyps = future;
        if hyps.is_empty() {
            continue;
        }
        let enc_frame = &enc_proj[time_idx * JOINT_HIDDEN..(time_idx + 1) * JOINT_HIDDEN];

        // Prefix search (NeMo `tdt_beam_decoding.py:729-788`).
        if cfg.maes_prefix_alpha > 0 && hyps.len() > 1 {
            hyps.sort_by(|a, b| b.y_sequence.len().cmp(&a.y_sequence.len()));
            for curr_idx in 0..hyps.len() - 1 {
                for pref_idx in (curr_idx + 1)..hyps.len() {
                    let curr_len = hyps[curr_idx].y_sequence.len();
                    let pref_len = hyps[pref_idx].y_sequence.len();
                    if is_strict_prefix(&hyps[curr_idx].y_sequence, &hyps[pref_idx].y_sequence)
                        && (curr_len - pref_len) <= cfg.maes_prefix_alpha
                    {
                        let pref_state = Arc::clone(hyps[pref_idx].states.last().unwrap());
                        decoder.joint_from_projected(
                            enc_frame,
                            &pref_state.pred_proj,
                            &mut projected,
                            &mut logits,
                        );
                        log_softmax_slice(&logits[..VOCAB], inv_temp, &mut logp);
                        log_softmax_slice(&logits[VOCAB..], inv_temp, &mut dur_logp);
                        let first_tok = hyps[curr_idx].y_sequence[pref_len];
                        let mut curr_score = hyps[pref_idx].score
                            + logp[first_tok as usize]
                            + dur_logp[0];
                        let mut lm_st = hyps[pref_idx].lm_state;
                        if let Some((lm, alpha)) = ngram_lm {
                            let (s, ns) = lm.advance_single(lm_st, first_tok);
                            curr_score += alpha * s;
                            lm_st = ns;
                        }
                        for k in pref_len..(curr_len - 1) {
                            let st_k = Arc::clone(&hyps[curr_idx].states[k]);
                            decoder.joint_from_projected(
                                enc_frame,
                                &st_k.pred_proj,
                                &mut projected,
                                &mut logits,
                            );
                            log_softmax_slice(&logits[..VOCAB], inv_temp, &mut logp);
                            log_softmax_slice(&logits[VOCAB..], inv_temp, &mut dur_logp);
                            let next_tok = hyps[curr_idx].y_sequence[k + 1];
                            curr_score += logp[next_tok as usize] + dur_logp[0];
                            if let Some((lm, alpha)) = ngram_lm {
                                let (s, ns) = lm.advance_single(lm_st, next_tok);
                                curr_score += alpha * s;
                                lm_st = ns;
                            }
                        }
                        hyps[curr_idx].score = logaddexp(hyps[curr_idx].score, curr_score);
                    }
                }
            }
        }

        let mut list_b: Vec<MaesHyp> = Vec::new();
        let mut list_nb: Vec<MaesHyp> = Vec::new();

        for step_n in 0..cfg.maes_num_steps {
            let mut list_exp: Vec<MaesHyp> = Vec::new();
            let mut list_nb_exp: Vec<MaesHyp> = Vec::new();

            for hyp in &hyps {
                let last_st = hyp.states.last().unwrap();
                decoder.joint_from_projected(enc_frame, &last_st.pred_proj, &mut projected, &mut logits);
                log_softmax_slice(&logits[..VOCAB], inv_temp, &mut logp);
                log_softmax_slice(&logits[VOCAB..], inv_temp, &mut dur_logp);

                let tok_topk = topk_desc(&logp, max_candidates);
                let mut total_cands: Vec<(f32, usize, usize)> =
                    Vec::with_capacity(DURATIONS * tok_topk.len());
                for (di, &dl) in dur_logp.iter().enumerate() {
                    for &(tl, ti) in &tok_topk {
                        total_cands.push((dl + tl, ti, di));
                    }
                }
                total_cands.sort_by(|a, b| b.0.total_cmp(&a.0));
                total_cands.truncate(max_candidates);

                let best_exp_score = total_cands.first().map_or(f32::NEG_INFINITY, |x| x.0);
                let threshold = best_exp_score - cfg.maes_expansion_gamma;

                for &(total_lp, k_idx, dur_idx) in &total_cands {
                    if total_lp < threshold {
                        continue;
                    }
                    let k = k_idx as u32;
                    let mut dur = TDT_DURATIONS[dur_idx];
                    if k == BLANK && dur == 0 {
                        dur = 1;
                    }
                    if k == BLANK {
                        list_b.push(MaesHyp {
                            score: hyp.score + total_lp,
                            y_sequence: hyp.y_sequence.clone(),
                            tokens: hyp.tokens.clone(),
                            states: hyp.states.clone(),
                            last_frame: hyp.last_frame + dur as usize,
                            lm_state: hyp.lm_state,
                        });
                    } else {
                        let next_st = pred_cache.step(&hyp.y_sequence, last_st, k);
                        let mut y_seq = hyp.y_sequence.clone();
                        y_seq.push(k);
                        let mut toks = hyp.tokens.clone();
                        toks.push(make_token(k, time_idx as u32, dur, total_lp, &logits[VOCAB..]));
                        let mut states = hyp.states.clone();
                        states.push(next_st);
                        let mut score = hyp.score + total_lp;
                        let mut lm_st = hyp.lm_state;
                        if let Some((lm, alpha)) = ngram_lm {
                            let (s, ns) = lm.advance_single(lm_st, k);
                            score += alpha * s;
                            lm_st = ns;
                        }
                        let new_hyp = MaesHyp {
                            score,
                            y_sequence: y_seq,
                            tokens: toks,
                            states,
                            last_frame: hyp.last_frame + dur as usize,
                            lm_state: lm_st,
                        };
                        if dur == 0 {
                            list_exp.push(new_hyp);
                        } else {
                            list_nb_exp.push(new_hyp);
                        }
                    }
                }
            }

            list_nb.extend(list_nb_exp);
            if list_exp.is_empty() {
                kept_hyps.extend(list_b.drain(..));
                kept_hyps.extend(list_nb.drain(..));
                kept_hyps = merge_maes_hyps(kept_hyps);
                kept_hyps.sort_by(|a, b| b.score.total_cmp(&a.score));
                kept_hyps.truncate(beam);
                break;
            } else if step_n + 1 < cfg.maes_num_steps {
                hyps = merge_maes_hyps(list_exp);
            } else {
                // Final mAES step: add blank transition probability to remaining zero-duration expansions.
                for hyp in &mut list_exp {
                    let last_st = hyp.states.last().unwrap();
                    decoder.joint_from_projected(
                        enc_frame,
                        &last_st.pred_proj,
                        &mut projected,
                        &mut logits,
                    );
                    log_softmax_slice(&logits[..VOCAB], inv_temp, &mut logp);
                    log_softmax_slice(&logits[VOCAB..], inv_temp, &mut dur_logp);
                    let mut best_di = 0usize;
                    let mut best_dv = f32::NEG_INFINITY;
                    for (di, &dv) in dur_logp.iter().enumerate() {
                        if dv > best_dv {
                            best_dv = dv;
                            best_di = di;
                        }
                    }
                    if best_di == 0 {
                        best_di = 1;
                    }
                    hyp.score += logp[BLANK as usize] + dur_logp[best_di];
                    hyp.last_frame += TDT_DURATIONS[best_di] as usize;
                }
                kept_hyps.extend(list_b.drain(..));
                kept_hyps.extend(list_exp);
                kept_hyps.extend(list_nb.drain(..));
                kept_hyps = merge_maes_hyps(kept_hyps);
                kept_hyps.sort_by(|a, b| b.score.total_cmp(&a.score));
                kept_hyps.truncate(beam);
            }
        }
    }

    kept_hyps.sort_by(|a, b| {
        rank_score(b.score, b.tokens.len(), cfg.score_norm)
            .total_cmp(&rank_score(a.score, a.tokens.len(), cfg.score_norm))
    });
    Ok(kept_hyps
        .into_iter()
        .map(|h| BeamHypothesis {
            tokens: h.tokens,
            score: h.score,
        })
        .collect())
}

// ============================================================================
// 3. Batched Modified ALSD (`ModifiedALSDBatchedTDTComputer::modified_alsd_torch`)
// ============================================================================

#[derive(Clone)]
struct MalsdBeam {
    score: f32,
    y_sequence: Vec<u32>, // non-blank token IDs (no SOS)
    tokens: Vec<Token>,
    state: Arc<PredictorState>,
    next_timestamp: usize,
    last_timestamp_lasts: usize,
    fusion_states: Vec<usize>,
    active: bool,
}

/// NeMo's `ModifiedALSDBatchedTDTComputer::modified_alsd_torch` (`strategy = "malsd_batch"`).
pub(crate) fn decode_tdt_malsd_batch(
    decoder: &TdtDecoder,
    enc_proj: &[f32],
    steps: usize,
    cfg: &BeamConfig,
    fusion: &[(&dyn FusionModel, f32)],
) -> Result<Vec<BeamHypothesis>> {
    validate_beam_cfg(cfg)?;
    if steps == 0 {
        return Ok(vec![BeamHypothesis {
            tokens: Vec::new(),
            score: 0.0,
        }]);
    }
    let beam_size = cfg.beam_size;
    let last_timestep = steps - 1;
    let mut pred_cache = PredCache::new(decoder);
    let sos_state = pred_cache.init_sos();
    let init_fusion: Vec<usize> = fusion.iter().map(|(m, _)| m.init_state()).collect();

    // Initialize `beam_size` slots: beam 0 active at score 0.0, others at -inf (matches `BatchedBeamHyps`).
    let mut beams: Vec<MalsdBeam> = Vec::with_capacity(beam_size);
    for b in 0..beam_size {
        beams.push(MalsdBeam {
            score: if b == 0 { 0.0 } else { f32::NEG_INFINITY },
            y_sequence: Vec::new(),
            tokens: Vec::new(),
            state: Arc::clone(&sos_state),
            next_timestamp: 0,
            last_timestamp_lasts: 0,
            fusion_states: init_fusion.clone(),
            active: true,
        });
    }

    let has_fusion = !fusion.is_empty();
    let alpha_sum: f32 = fusion.iter().map(|(_, a)| *a).sum();
    let mut f_scores_buf = if has_fusion {
        vec![0.0f32; NON_BLANK_VOCAB]
    } else {
        Vec::new()
    };
    let mut f_next_buf = if has_fusion {
        vec![0usize; NON_BLANK_VOCAB]
    } else {
        Vec::new()
    };

    let mut projected = vec![0.0f32; JOINT_HIDDEN];
    let mut logits = vec![0.0f32; JOINT_VOCAB];
    let mut logp = vec![0.0f32; VOCAB];
    let mut dur_logp = vec![0.0f32; DURATIONS];

    let eps = 1e-2f32; // Matches `tdt_malsd_batched_computer.py:633` (`eps=1e-2`).

    // Candidate struct for one `(beam_idx, candidate_slot)`.
    struct Cand {
        beam_idx: usize,
        label: i32, // -1 = inactive passthrough, 0..8191 = token, 8192 = blank
        duration: u32,
        new_score: f32,
        step_score: f32,
        dur_logits: [f32; DURATIONS],
        next_fusion_states: Vec<usize>,
    }

    while beams.iter().any(|b| b.active) {
        let mut all_cands: Vec<Cand> = Vec::with_capacity(beam_size * beam_size);

        for (b_idx, beam) in beams.iter().enumerate() {
            if !beam.active {
                // Keep inactive (finished) hypothesis at slot 0 with unchanged score; others -inf.
                all_cands.push(Cand {
                    beam_idx: b_idx,
                    label: -1,
                    duration: 0,
                    new_score: beam.score,
                    step_score: 0.0,
                    dur_logits: [0.0; DURATIONS],
                    next_fusion_states: beam.fusion_states.clone(),
                });
                continue;
            }
            if beam.score == f32::NEG_INFINITY {
                continue;
            }

            let safe_t = beam.next_timestamp.min(last_timestep);
            let enc_frame = &enc_proj[safe_t * JOINT_HIDDEN..(safe_t + 1) * JOINT_HIDDEN];
            decoder.joint_from_projected(enc_frame, &beam.state.pred_proj, &mut projected, &mut logits);
            log_softmax_slice(&logits[..VOCAB], 1.0, &mut logp);
            log_softmax_slice(&logits[VOCAB..], 1.0, &mut dur_logp);

            let mut raw_durs = [0.0f32; DURATIONS];
            raw_durs.copy_from_slice(&logits[VOCAB..]);

            // Check max_symbols_per_step forced blank (`tdt_malsd_batched_computer.py:520-536`).
            let force_blank = cfg
                .max_symbols_per_step
                .is_some_and(|cap| beam.last_timestamp_lasts >= cap);
            if force_blank {
                let max_dur_lp = dur_logp
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let step_lp = logp[BLANK as usize] + max_dur_lp;
                // NeMo selects slot 0 duration (forced >= 1).
                let best_dur_idx = dur_logp
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map_or(1, |(i, _)| i);
                let dur = if best_dur_idx == 0 { 1 } else { TDT_DURATIONS[best_dur_idx] };
                all_cands.push(Cand {
                    beam_idx: b_idx,
                    label: BLANK as i32,
                    duration: dur,
                    new_score: beam.score + step_lp,
                    step_score: step_lp,
                    dur_logits: raw_durs,
                    next_fusion_states: beam.fusion_states.clone(),
                });
                continue;
            }

            // Advance fusion models for this beam if present.
            let mut fusion_sum = Vec::new();
            let mut fusion_next_all: Vec<Vec<usize>> = Vec::new();
            if has_fusion {
                fusion_sum.resize(NON_BLANK_VOCAB, 0.0f32);
                fusion_next_all.reserve(fusion.len());
                for (m_idx, &(model, alpha)) in fusion.iter().enumerate() {
                    model.advance_into(beam.fusion_states[m_idx], &mut f_scores_buf, &mut f_next_buf);
                    for (dst, &s) in fusion_sum.iter_mut().zip(&f_scores_buf) {
                        *dst += s * alpha;
                    }
                    fusion_next_all.push(f_next_buf.clone());
                }
            }

            // Compute top-`beam_size` `(score, label, dur_idx)` candidates matching NeMo's
            // `topk_fusion_model` (`tdt_malsd_batched_computer.py:633-744`).
            let top_entries: Vec<(f32, usize, usize)> = if !has_fusion {
                // Exact top-`beam_size` over `(V + 1) x DURATIONS`: since `dur_logp` is length 5,
                // any top-`beam_size` pair comes from the top-`beam_size` vocab tokens × 5 durations!
                let tok_top = topk_desc(&logp, beam_size);
                let mut pairs = Vec::with_capacity(tok_top.len() * DURATIONS);
                for &(tl, ti) in &tok_top {
                    for (di, &dl) in dur_logp.iter().enumerate() {
                        pairs.push((tl + dl, ti, di));
                    }
                }
                pairs.sort_by(|a, b| {
                    b.0.total_cmp(&a.0)
                        .then_with(|| (a.1 * DURATIONS + a.2).cmp(&(b.1 * DURATIONS + b.2)))
                });
                pairs.truncate(beam_size);
                pairs
            } else {
                match (cfg.pruning_mode, cfg.blank_lm_score_mode) {
                    (PruningMode::Late, BlankLmScoreMode::NoScore) => {
                        for i in 0..NON_BLANK_VOCAB {
                            logp[i] += fusion_sum[i];
                        }
                        let tok_top = topk_desc(&logp, beam_size);
                        let mut pairs = Vec::with_capacity(tok_top.len() * DURATIONS);
                        for &(tl, ti) in &tok_top {
                            for (di, &dl) in dur_logp.iter().enumerate() {
                                pairs.push((tl + dl, ti, di));
                            }
                        }
                        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
                        pairs.truncate(beam_size);
                        pairs
                    }
                    (PruningMode::Late, BlankLmScoreMode::LmWeightedFull) => {
                        let blank_lp = logp[BLANK as usize];
                        let non_blank_lp = (-(blank_lp.exp().min(1.0 - eps))).ln_1p();
                        let nb_shift = non_blank_lp * alpha_sum;
                        for i in 0..NON_BLANK_VOCAB {
                            logp[i] += nb_shift + fusion_sum[i];
                        }
                        logp[BLANK as usize] *= 1.0 + alpha_sum;
                        let tok_top = topk_desc(&logp, beam_size);
                        let mut pairs = Vec::with_capacity(tok_top.len() * DURATIONS);
                        for &(tl, ti) in &tok_top {
                            for (di, &dl) in dur_logp.iter().enumerate() {
                                pairs.push((tl + dl, ti, di));
                            }
                        }
                        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
                        pairs.truncate(beam_size);
                        pairs
                    }
                    (PruningMode::Early, BlankLmScoreMode::NoScore) => {
                        let tok_top = topk_desc(&logp, beam_size);
                        let mut pairs = Vec::with_capacity(tok_top.len() * DURATIONS);
                        for &(tl, ti) in &tok_top {
                            let adj = if ti == BLANK as usize { tl } else { tl + fusion_sum[ti] };
                            for (di, &dl) in dur_logp.iter().enumerate() {
                                pairs.push((adj + dl, ti, di));
                            }
                        }
                        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
                        pairs.truncate(beam_size);
                        pairs
                    }
                    (PruningMode::Early, BlankLmScoreMode::LmWeightedFull) => {
                        let blank_lp = logp[BLANK as usize];
                        let non_blank_lp = (-(blank_lp.exp().min(1.0 - eps))).ln_1p();
                        let tok_top = topk_desc(&logp, beam_size);
                        let mut pairs = Vec::with_capacity(tok_top.len() * DURATIONS);
                        for &(tl, ti) in &tok_top {
                            let adj = if ti == BLANK as usize {
                                tl * (1.0 + alpha_sum)
                            } else {
                                tl + non_blank_lp * alpha_sum + fusion_sum[ti]
                            };
                            for (di, &dl) in dur_logp.iter().enumerate() {
                                pairs.push((adj + dl, ti, di));
                            }
                        }
                        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
                        pairs.truncate(beam_size);
                        pairs
                    }
                }
            };

            for (step_lp, label_idx, dur_idx) in top_entries {
                let mut dur = TDT_DURATIONS[dur_idx];
                if label_idx == BLANK as usize && dur == 0 {
                    dur = 1;
                }
                let next_f = if !has_fusion || label_idx == BLANK as usize {
                    beam.fusion_states.clone()
                } else {
                    fusion_next_all.iter().map(|v| v[label_idx]).collect()
                };
                all_cands.push(Cand {
                    beam_idx: b_idx,
                    label: label_idx as i32,
                    duration: dur,
                    new_score: beam.score + step_lp,
                    step_score: step_lp,
                    dur_logits: raw_durs,
                    next_fusion_states: next_f,
                });
            }
        }

        // Prune to top `beam_size` across all candidates (`step 2.4`).
        all_cands.sort_by(|a, b| b.new_score.total_cmp(&a.new_score));
        all_cands.truncate(beam_size);

        // Build next beam states (`steps 3 & 5`).
        let mut next_beams: Vec<MalsdBeam> = Vec::with_capacity(beam_size);
        for cand in all_cands {
            let parent = &beams[cand.beam_idx];
            if cand.label < 0 {
                // Finished passthrough hypothesis.
                let mut kept = parent.clone();
                kept.score = cand.new_score;
                kept.active = false;
                next_beams.push(kept);
            } else if cand.label as u32 == BLANK {
                let next_ts = parent.next_timestamp + cand.duration as usize;
                next_beams.push(MalsdBeam {
                    score: cand.new_score,
                    y_sequence: parent.y_sequence.clone(),
                    tokens: parent.tokens.clone(),
                    state: Arc::clone(&parent.state),
                    next_timestamp: next_ts,
                    last_timestamp_lasts: 0,
                    fusion_states: cand.next_fusion_states,
                    active: next_ts <= last_timestep,
                });
            } else {
                let tok_id = cand.label as u32;
                let mut sos_plus_y = Vec::with_capacity(parent.y_sequence.len() + 1);
                sos_plus_y.push(BLANK);
                sos_plus_y.extend_from_slice(&parent.y_sequence);
                let next_state = pred_cache.step(&sos_plus_y, &parent.state, tok_id);

                let mut y_seq = parent.y_sequence.clone();
                y_seq.push(tok_id);
                let mut toks = parent.tokens.clone();
                toks.push(make_token(
                    tok_id,
                    parent.next_timestamp as u32,
                    cand.duration,
                    cand.step_score,
                    &cand.dur_logits,
                ));
                let next_ts = parent.next_timestamp + cand.duration as usize;
                let lasts = if cand.duration > 0 {
                    0
                } else {
                    parent.last_timestamp_lasts + 1
                };
                next_beams.push(MalsdBeam {
                    score: cand.new_score,
                    y_sequence: y_seq,
                    tokens: toks,
                    state: next_state,
                    next_timestamp: next_ts,
                    last_timestamp_lasts: lasts,
                    fusion_states: cand.next_fusion_states,
                    active: next_ts <= last_timestep,
                });
            }
        }

        // Pad up to `beam_size` with inactive entries if fewer candidates existed.
        while next_beams.len() < beam_size {
            next_beams.push(MalsdBeam {
                score: f32::NEG_INFINITY,
                y_sequence: Vec::new(),
                tokens: Vec::new(),
                state: Arc::clone(&sos_state),
                next_timestamp: steps,
                last_timestamp_lasts: 0,
                fusion_states: init_fusion.clone(),
                active: false,
            });
        }

        // Step 4: Recombine equivalent hypotheses (`BatchedBeamHyps::recombine_hyps_`,
        // matching `(y_sequence, next_timestamp)`).
        if beam_size > 1 {
            let mut best_by_key: HashMap<(Vec<u32>, usize), usize> = HashMap::new();
            for idx in 0..next_beams.len() {
                if next_beams[idx].score == f32::NEG_INFINITY {
                    continue;
                }
                let key = (next_beams[idx].y_sequence.clone(), next_beams[idx].next_timestamp);
                if let Some(&winner) = best_by_key.get(&key) {
                    if next_beams[idx].score > next_beams[winner].score {
                        next_beams[idx].score = logaddexp(next_beams[idx].score, next_beams[winner].score);
                        next_beams[winner].score = f32::NEG_INFINITY;
                        next_beams[winner].active = false;
                        best_by_key.insert(key, idx);
                    } else {
                        next_beams[winner].score = logaddexp(next_beams[winner].score, next_beams[idx].score);
                        next_beams[idx].score = f32::NEG_INFINITY;
                        next_beams[idx].active = false;
                    }
                } else {
                    best_by_key.insert(key, idx);
                }
            }
        }

        beams = next_beams;
    }

    // Filter out inactive -inf slots and sort by normalized score (`score / (len + 1)`).
    let mut valid: Vec<MalsdBeam> = beams
        .into_iter()
        .filter(|b| b.score > f32::NEG_INFINITY)
        .collect();
    valid.sort_by(|a, b| {
        rank_score(b.score, b.tokens.len(), cfg.score_norm)
            .total_cmp(&rank_score(a.score, a.tokens.len(), cfg.score_norm))
    });
    if valid.is_empty() {
        return Ok(vec![BeamHypothesis {
            tokens: Vec::new(),
            score: 0.0,
        }]);
    }
    Ok(valid
        .into_iter()
        .map(|b| BeamHypothesis {
            tokens: b.tokens,
            score: b.score,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_desc_returns_sorted_with_lower_index_tiebreak() {
        let vals = [1.0f32, 3.0, 2.0, 3.0, 0.5];
        let got = topk_desc(&vals, 3);
        assert_eq!(got, vec![(3.0, 1), (3.0, 3), (2.0, 2)]);
    }

    #[test]
    fn logaddexp_matches_exact() {
        let v = logaddexp(-1.0, -2.0);
        let expected = ((-1.0f64).exp() + (-2.0f64).exp()).ln() as f32;
        assert!((v - expected).abs() < 1e-6);
        assert_eq!(logaddexp(f32::NEG_INFINITY, -3.5), -3.5);
    }
}
