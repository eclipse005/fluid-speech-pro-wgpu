//! Shallow-fusion models: NeMo-compatible Phrase Boosting Tree (`BoostingTree`)
//! and external ARPA N-gram Language Model (`ArpaLm`).
//!
//! Mirrors:
//! * `nemo/collections/asr/parts/context_biasing/context_graph_universal.py`
//! * `nemo/collections/asr/parts/context_biasing/boosting_graph_batched.py`
//! * `nemo/collections/asr/parts/submodules/ngram_lm/ngram_lm_batched.py`

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::tdt::VOCAB;
use crate::text::Tokenizer;

/// Number of non-blank vocabulary tokens (`VOCAB - 1 = 8192`).
pub const NON_BLANK_VOCAB: usize = VOCAB - 1;

/// Default character offset NeMo uses when encoding BPE token IDs in ARPA files.
pub const DEFAULT_TOKEN_OFFSET: u32 = 100;

/// Shallow-fusion scoring model over non-blank BPE token IDs (`0..8192`).
pub trait FusionModel: Send + Sync {
    /// Initial automaton / LM state ID (`0` for BoostingTree, `1` for ARPA BOS state).
    fn init_state(&self) -> usize;

    /// Evaluates transitions from `state` for every non-blank token `0..NON_BLANK_VOCAB`,
    /// writing natural-log scores into `scores` and destination states into `next_states`.
    fn advance_into(&self, state: usize, scores: &mut [f32], next_states: &mut [usize]);

    /// Evaluates a single token transition from `state`, returning `(log_score, next_state)`.
    fn advance_single(&self, state: usize, label: u32) -> (f32, usize);
}

// ============================================================================
// Unified SuffixTree automaton (shared by BoostingTree and ArpaLm)
// ============================================================================

#[derive(Debug, Clone)]
struct TreeState {
    arcs_start: usize,
    arcs_end: usize,
    backoff_to: usize,
    backoff_weight: f32,
    final_weight: f32,
}

#[derive(Debug, Clone, Copy)]
struct TreeArc {
    ilabel: u32,
    to: usize,
    weight: f32,
}

/// Compact CSR suffix-tree weighted acceptor supporting both `BoostingTree` and `ArpaLm`.
#[derive(Debug, Clone)]
struct SuffixTree {
    vocab_size: usize,
    bos_state: usize,
    root_unk_score: f32,
    states: Vec<TreeState>,
    /// Arcs grouped per state and sorted ascending by `ilabel`.
    arcs: Vec<TreeArc>,
}

impl SuffixTree {
    const START_STATE: usize = 0;

    fn advance_into(&self, state: usize, scores: &mut [f32], next_states: &mut [usize]) {
        debug_assert_eq!(scores.len(), self.vocab_size);
        debug_assert_eq!(next_states.len(), self.vocab_size);

        // Walk backoff chain from `state` down to `START_STATE`, recording `(st, acc_backoff)`.
        let mut chain: Vec<(usize, f32)> = Vec::with_capacity(8);
        let mut cur = state;
        let mut acc = 0.0f32;
        loop {
            chain.push((cur, acc));
            if cur == Self::START_STATE {
                break;
            }
            let st = &self.states[cur];
            acc += st.backoff_weight;
            cur = st.backoff_to;
        }

        // Fill base fallback (unknown root transitions) at the total root backoff.
        scores.fill(acc + self.root_unk_score);
        next_states.fill(Self::START_STATE);

        // Stamp arcs from root up to `state` so higher-order states override lower-order ones.
        for &(st_id, st_acc) in chain.iter().rev() {
            let st = &self.states[st_id];
            for arc in &self.arcs[st.arcs_start..st.arcs_end] {
                let idx = arc.ilabel as usize;
                if idx < self.vocab_size {
                    scores[idx] = st_acc + arc.weight;
                    next_states[idx] = arc.to;
                }
            }
        }
    }

    fn advance_single(&self, state: usize, label: u32) -> (f32, usize) {
        let mut cur = state;
        let mut acc = 0.0f32;
        loop {
            let st = &self.states[cur];
            let slice = &self.arcs[st.arcs_start..st.arcs_end];
            if let Ok(pos) = slice.binary_search_by_key(&label, |a| a.ilabel) {
                let arc = slice[pos];
                return (acc + arc.weight, arc.to);
            }
            if cur == Self::START_STATE {
                return (acc + self.root_unk_score, Self::START_STATE);
            }
            acc += st.backoff_weight;
            cur = st.backoff_to;
        }
    }
}

// ============================================================================
// Context Biasing: Phrase Boosting Tree (`BoostingTreeModelConfig` / `GPUBoostingTreeModel`)
// ============================================================================

/// Configuration for phrase context biasing, mirroring NeMo's `BoostingTreeModelConfig`.
#[derive(Debug, Clone)]
pub struct BoostingTreeConfig {
    /// Phrases to boost (`key_phrases_list`).
    pub key_phrases: Vec<String>,
    /// Optional path to a UTF-8 file with one phrase per line (`key_phrases_file`).
    pub key_phrases_file: Option<PathBuf>,
    /// Bonus per token arc transition (`1.0` default).
    pub context_score: f32,
    /// Depth scaling factor for tokens after the first (`2.0` default for TDT/RNN-T).
    pub depth_scaling: f32,
    /// Score for tokens not starting any boosted phrase (`0.0` default).
    pub unk_score: f32,
    /// EOS score at completed phrase nodes (`1.0` default).
    pub final_eos_score: f32,
    /// Use Icefall-style uniform token weights instead of depth-scaled weights (`false` default).
    pub uniform_weights: bool,
}

impl Default for BoostingTreeConfig {
    fn default() -> Self {
        Self {
            key_phrases: Vec::new(),
            key_phrases_file: None,
            context_score: 1.0,
            depth_scaling: 2.0,
            unk_score: 0.0,
            final_eos_score: 1.0,
            uniform_weights: false,
        }
    }
}

impl BoostingTreeConfig {
    pub fn is_empty(&self) -> bool {
        self.key_phrases.is_empty() && self.key_phrases_file.is_none()
    }
}

#[derive(Debug)]
struct ContextNode {
    id: usize,
    token: i32,
    token_score: f32,
    node_score: f32,
    is_end: bool,
    level: usize,
    next: BTreeMap<u32, usize>,
    fail: usize,
}

/// Compiled Aho-Corasick phrase boosting tree, mirroring NeMo's `GPUBoostingTreeModel`.
#[derive(Debug, Clone)]
pub struct BoostingTree {
    tree: SuffixTree,
}

impl BoostingTree {
    /// Builds a boosting tree from `cfg` using the archive's SentencePiece `tokenizer`.
    pub fn from_config(cfg: &BoostingTreeConfig, tokenizer: &Tokenizer) -> Result<Self> {
        let mut phrases = cfg.key_phrases.clone();
        if let Some(path) = &cfg.key_phrases_file {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("read key_phrases_file {}", path.display()))?;
            for line in content.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    phrases.push(trimmed.to_string());
                }
            }
        }
        let mut sequences: Vec<Vec<u32>> = Vec::with_capacity(phrases.len());
        for phrase in &phrases {
            let ids = tokenizer.encode(phrase);
            if !ids.is_empty() {
                sequences.push(ids);
            }
        }
        Self::from_token_sequences(
            &sequences,
            NON_BLANK_VOCAB,
            cfg.context_score,
            cfg.depth_scaling,
            cfg.unk_score,
            cfg.final_eos_score,
            cfg.uniform_weights,
        )
    }

    /// Builds a boosting tree directly from pre-tokenized BPE token ID sequences.
    pub fn from_token_sequences(
        token_ids: &[Vec<u32>],
        vocab_size: usize,
        context_score: f32,
        depth_scaling: f32,
        unk_score: f32,
        final_eos_score: f32,
        uniform_weights: bool,
    ) -> Result<Self> {
        // 1. Build Aho-Corasick trie (`context_graph_universal.py::ContextGraph::build`).
        let mut nodes = vec![ContextNode {
            id: 0,
            token: -1,
            token_score: 0.0,
            node_score: 0.0,
            is_end: false,
            level: 0,
            next: BTreeMap::new(),
            fail: 0,
        }];

        for seq in token_ids {
            if seq.is_empty() {
                continue;
            }
            let mut cur = 0usize;
            for (i, &tok) in seq.iter().enumerate() {
                if tok as usize >= vocab_size {
                    bail!("boosting phrase token {tok} >= vocab_size {vocab_size}");
                }
                let is_end = i + 1 == seq.len();
                if let Some(&child_id) = nodes[cur].next.get(&tok) {
                    let parent_node_score = nodes[cur].node_score;
                    let child = &mut nodes[child_id];
                    let ts = context_score.max(child.token_score);
                    child.token_score = ts;
                    child.node_score = parent_node_score + ts;
                    child.is_end = child.is_end || is_end;
                    cur = child_id;
                } else {
                    let token_score = if i > 0 && !uniform_weights {
                        context_score * depth_scaling + ((i + 1) as f32).ln()
                    } else {
                        context_score
                    };
                    let new_id = nodes.len();
                    let node_score = nodes[cur].node_score + token_score;
                    nodes[cur].next.insert(tok, new_id);
                    nodes.push(ContextNode {
                        id: new_id,
                        token: tok as i32,
                        token_score,
                        node_score,
                        is_end,
                        level: i + 1,
                        next: BTreeMap::new(),
                        fail: 0,
                    });
                    cur = new_id;
                }
            }
        }

        // 2. Fill fail arcs via BFS (`ContextGraph::_fill_fail_output`).
        let mut queue = VecDeque::new();
        let root_children: Vec<usize> = nodes[0].next.values().copied().collect();
        for child_id in root_children {
            nodes[child_id].fail = 0;
            queue.push_back(child_id);
        }
        while let Some(cur_id) = queue.pop_front() {
            let children: Vec<(u32, usize)> =
                nodes[cur_id].next.iter().map(|(&t, &id)| (t, id)).collect();
            let cur_fail = nodes[cur_id].fail;
            for (tok, child_id) in children {
                let mut fail = cur_fail;
                if let Some(&next_id) = nodes[fail].next.get(&tok) {
                    fail = next_id;
                } else {
                    fail = nodes[fail].fail;
                    while !nodes[fail].next.contains_key(&tok) {
                        if nodes[fail].token == -1 {
                            break;
                        }
                        fail = nodes[fail].fail;
                    }
                    if let Some(&next_id) = nodes[fail].next.get(&tok) {
                        fail = next_id;
                    }
                }
                nodes[child_id].fail = fail;
                queue.push_back(child_id);
            }
        }

        // 3. Compile into `SuffixTree` matching `BoostingTreeStorage`.
        let num_states = nodes.len();
        let mut states = Vec::with_capacity(num_states);
        let mut arcs = Vec::new();

        for node in &nodes {
            let arcs_start = arcs.len();
            for (&tok, &child_id) in &node.next {
                let child = &nodes[child_id];
                let mut weight = child.token_score;
                if uniform_weights && child.is_end && child.level > 1 {
                    weight += child.node_score;
                }
                arcs.push(TreeArc {
                    ilabel: tok,
                    to: child_id,
                    weight,
                });
            }
            let arcs_end = arcs.len();
            let (backoff_to, backoff_weight, final_weight) = if node.id == 0 {
                (0, 0.0, 0.0)
            } else {
                let bw = if node.is_end && (node.level == 1 || !uniform_weights) {
                    0.0
                } else {
                    nodes[node.fail].node_score - node.node_score
                };
                let fw = if node.is_end { final_eos_score } else { 0.0 };
                (node.fail, bw, fw)
            };
            states.push(TreeState {
                arcs_start,
                arcs_end,
                backoff_to,
                backoff_weight,
                final_weight,
            });
        }

        Ok(Self {
            tree: SuffixTree {
                vocab_size,
                bos_state: 0,
                root_unk_score: unk_score,
                states,
                arcs,
            },
        })
    }

    pub fn num_states(&self) -> usize {
        self.tree.states.len()
    }
}

impl FusionModel for BoostingTree {
    fn init_state(&self) -> usize {
        self.tree.bos_state
    }

    fn advance_into(&self, state: usize, scores: &mut [f32], next_states: &mut [usize]) {
        self.tree.advance_into(state, scores, next_states);
    }

    fn advance_single(&self, state: usize, label: u32) -> (f32, usize) {
        self.tree.advance_single(state, label)
    }
}

// ============================================================================
// External ARPA N-gram Language Model (`NGramGPULanguageModel::from_file`)
// ============================================================================

const BOS_ID: i32 = -1;
const EOS_ID: i32 = -2;
const UNK_ID: i32 = -3;
const LN_10: f64 = std::f64::consts::LN_10;

/// External ARPA N-gram Language Model for shallow fusion in `greedy_batch`, `malsd_batch`, and `maes`.
///
/// Mirrors NeMo's `NGramGPULanguageModel` (`ngram_lm_batched.py`).
#[derive(Debug, Clone)]
pub struct ArpaLm {
    tree: SuffixTree,
    max_order: usize,
}

impl ArpaLm {
    /// Loads a text ARPA file from `path`.
    ///
    /// * `token_offset`: `Some(100)` matches NeMo's default character-encoded BPE ARPA
    ///   (`chr(token_id + 100)`). Pass `None` to resolve tokens via `tokenizer` piece names
    ///   (or integer token IDs).
    pub fn from_file(
        path: &Path,
        token_offset: Option<u32>,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("read ARPA LM {}", path.display()))?;
        Self::from_arpa_str(&content, NON_BLANK_VOCAB, token_offset, tokenizer)
    }

    /// Parses an ARPA text model into a compact suffix-tree automaton matching NeMo's `SuffixTreeStorage`.
    pub fn from_arpa_str(
        content: &str,
        vocab_size: usize,
        token_offset: Option<u32>,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<Self> {
        let mut lines = content.lines();
        // 1. Parse `\data\` header.
        let mut found_data = false;
        let mut order_counts: BTreeMap<usize, usize> = BTreeMap::new();
        for raw in lines.by_ref() {
            let line = raw.trim();
            if line.is_empty() {
                if found_data && !order_counts.is_empty() {
                    break;
                }
                continue;
            }
            if !found_data {
                if line == "\\data\\" {
                    found_data = true;
                    continue;
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix("ngram ") {
                if let Some((ord_str, cnt_str)) = rest.split_once('=') {
                    let ord: usize = ord_str.trim().parse().context("parse ARPA order")?;
                    let cnt: usize = cnt_str.trim().parse().context("parse ARPA count")?;
                    order_counts.insert(ord, cnt);
                }
            }
        }
        if order_counts.is_empty() {
            bail!("ARPA file has no \\data\\ ngram header");
        }
        let max_order = *order_counts.keys().max().unwrap();

        // 2. Read all n-grams grouped by order.
        let mut by_order: Vec<Vec<(Vec<i32>, f32, f32)>> = vec![Vec::new(); max_order + 1];
        let mut cur_order = 0usize;
        for raw in lines {
            let line = raw.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                continue;
            }
            if line.starts_with("\\end\\") {
                break;
            }
            if line.starts_with('\\') && line.ends_with("-grams:") {
                let num_str = &line[1..line.len() - "-grams:".len()];
                cur_order = num_str.parse().context("parse ARPA section order")?;
                continue;
            }
            if cur_order == 0 || cur_order > max_order {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 2 {
                continue;
            }
            let log10_w: f64 = parts[0]
                .trim()
                .parse()
                .with_context(|| format!("parse ARPA weight in `{line}`"))?;
            let weight = (log10_w * LN_10) as f32;
            let backoff = if parts.len() >= 3 && !parts[2].trim().is_empty() {
                let log10_b: f64 = parts[2]
                    .trim()
                    .parse()
                    .with_context(|| format!("parse ARPA backoff in `{line}`"))?;
                (log10_b * LN_10) as f32
            } else {
                0.0
            };
            let syms = parse_arpa_symbols(parts[1], cur_order, token_offset, tokenizer)?;
            by_order[cur_order].push((syms, weight, backoff));
        }

        // 3. Build SuffixTreeStorage matching `ngram_lm_batched.py:297-489`.
        let mut states = vec![
            // State 0: START_STATE (unigram / root)
            TreeState {
                arcs_start: 0,
                arcs_end: 0,
                backoff_to: 0,
                backoff_weight: 0.0,
                final_weight: f32::NEG_INFINITY,
            },
            // State 1: BOS_STATE (`<s>`)
            TreeState {
                arcs_start: 0,
                arcs_end: 0,
                backoff_to: 0,
                backoff_weight: 0.0,
                final_weight: f32::NEG_INFINITY,
            },
        ];
        let mut state_arcs: Vec<Vec<TreeArc>> = vec![Vec::new(), Vec::new()];
        let mut state_map: HashMap<Vec<i32>, usize> = HashMap::new();
        state_map.insert(vec![BOS_ID], 1);

        // Unigrams (order 1).
        let mut unk_prob = f32::NEG_INFINITY;
        let mut num_vocab_labels = 0usize;
        by_order[1].sort_by(|a, b| a.0.cmp(&b.0));
        for (syms, weight, backoff) in &by_order[1] {
            let sym = syms[0];
            if sym == UNK_ID {
                unk_prob = *weight;
            } else if sym == BOS_ID {
                states[1].backoff_weight = *backoff;
            } else if sym == EOS_ID {
                states[0].final_weight = *weight;
            } else if sym >= 0 && (sym as usize) < vocab_size {
                let next_state = states.len();
                states.push(TreeState {
                    arcs_start: 0,
                    arcs_end: 0,
                    backoff_to: 0,
                    backoff_weight: *backoff,
                    final_weight: f32::NEG_INFINITY,
                });
                state_arcs.push(Vec::new());
                state_arcs[0].push(TreeArc {
                    ilabel: sym as u32,
                    to: next_state,
                    weight: *weight,
                });
                state_map.insert(vec![sym], next_state);
                num_vocab_labels += 1;
            }
        }
        // `normalize_unk` (`ngram_lm_batched.py:343-346`).
        let num_unk_labels = vocab_size.saturating_sub(num_vocab_labels);
        if num_unk_labels > 1 && unk_prob.is_finite() {
            unk_prob -= (num_unk_labels as f64).ln() as f32;
        }

        // Higher orders (2 ..= max_order).
        for order in 2..=max_order {
            let is_max = order == max_order;
            by_order[order].sort_by(|a, b| a.0.cmp(&b.0));
            for (syms, weight, backoff) in &by_order[order] {
                let prefix = &syms[..order - 1];
                let ilabel = syms[order - 1];
                let Some(&from_state) = state_map.get(prefix) else {
                    continue;
                };
                if ilabel == EOS_ID {
                    states[from_state].final_weight = *weight;
                    continue;
                }
                if ilabel < 0 || ilabel as usize >= vocab_size {
                    continue;
                }
                let backoff_state = state_map.get(&syms[1..]).copied().unwrap_or(0);
                if is_max {
                    state_arcs[from_state].push(TreeArc {
                        ilabel: ilabel as u32,
                        to: backoff_state,
                        weight: *weight,
                    });
                } else {
                    let next_state = states.len();
                    states.push(TreeState {
                        arcs_start: 0,
                        arcs_end: 0,
                        backoff_to: backoff_state,
                        backoff_weight: *backoff,
                        final_weight: f32::NEG_INFINITY,
                    });
                    state_arcs.push(Vec::new());
                    state_arcs[from_state].push(TreeArc {
                        ilabel: ilabel as u32,
                        to: next_state,
                        weight: *weight,
                    });
                    state_map.insert(syms.clone(), next_state);
                }
            }
        }

        // Pack per-state arcs into sorted CSR `arcs` array.
        let mut arcs = Vec::new();
        for (st_id, mut list) in state_arcs.into_iter().enumerate() {
            list.sort_by_key(|a| a.ilabel);
            let start = arcs.len();
            arcs.extend(list);
            let end = arcs.len();
            states[st_id].arcs_start = start;
            states[st_id].arcs_end = end;
        }

        Ok(Self {
            tree: SuffixTree {
                vocab_size,
                bos_state: 1,
                root_unk_score: unk_prob,
                states,
                arcs,
            },
            max_order,
        })
    }

    pub fn max_order(&self) -> usize {
        self.max_order
    }

    pub fn num_states(&self) -> usize {
        self.tree.states.len()
    }
}

impl FusionModel for ArpaLm {
    fn init_state(&self) -> usize {
        self.tree.bos_state
    }

    fn advance_into(&self, state: usize, scores: &mut [f32], next_states: &mut [usize]) {
        self.tree.advance_into(state, scores, next_states);
    }

    fn advance_single(&self, state: usize, label: u32) -> (f32, usize) {
        self.tree.advance_single(state, label)
    }
}

fn parse_arpa_symbols(
    symbols_str: &str,
    expected_len: usize,
    token_offset: Option<u32>,
    tokenizer: Option<&Tokenizer>,
) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(expected_len);
    for tok in symbols_str.split(' ') {
        if tok.is_empty() {
            continue;
        }
        let id = match tok {
            "<s>" => BOS_ID,
            "</s>" => EOS_ID,
            "<unk>" => UNK_ID,
            other => {
                if let Some(offset) = token_offset {
                    let mut chars = other.chars();
                    let ch = chars.next().context("empty ARPA symbol")?;
                    if chars.next().is_some() {
                        bail!("expected single offset-encoded char in ARPA symbol `{other}`");
                    }
                    let cp = ch as u32;
                    if cp < offset {
                        bail!("ARPA symbol `{other}` (U+{cp:04X}) < token_offset {offset}");
                    }
                    (cp - offset) as i32
                } else if let Some(tk) = tokenizer {
                    if let Some(id) = tk.piece_id(other) {
                        id as i32
                    } else if let Ok(id) = other.parse::<i32>() {
                        id
                    } else {
                        UNK_ID
                    }
                } else {
                    other
                        .parse::<i32>()
                        .with_context(|| format!("parse ARPA integer token `{other}`"))?
                }
            }
        };
        out.push(id);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boosting_tree_scores_and_backoffs_match_context_graph() {
        // Phrase [10, 20] with context_score=1.0, depth_scaling=2.0, unk_score=0.0.
        // Expected:
        //   state 0 -> 10: score = 1.0, next = 1
        //   state 1 -> 20: score = 1.0*2.0 + ln(2), next = 2 (end -> backoff_w = 0.0)
        //   state 1 -> other (e.g. 5): backoff = (0.0 - 1.0) + unk_score(0.0) = -1.0, next = 0
        //   state 2 -> other (e.g. 5): end node backoff_w = 0.0 -> score = 0.0, next = 0
        let bt = BoostingTree::from_token_sequences(&[vec![10, 20]], 32, 1.0, 2.0, 0.0, 1.0, false)
            .unwrap();
        let (s0, st1) = bt.advance_single(0, 10);
        assert!((s0 - 1.0).abs() < 1e-6);
        assert_eq!(st1, 1);

        let (s1, st2) = bt.advance_single(st1, 20);
        let expected_s1 = 2.0 + 2.0f32.ln();
        assert!((s1 - expected_s1).abs() < 1e-6);
        assert_eq!(st2, 2);

        // Partial match broken at state 1 cancels accumulated bonus (-1.0).
        let (sb1, stb1) = bt.advance_single(st1, 5);
        assert!((sb1 - (-1.0)).abs() < 1e-6);
        assert_eq!(stb1, 0);

        // Completed phrase at state 2 does not penalize transition back to root.
        let (sb2, stb2) = bt.advance_single(st2, 5);
        assert!(sb2.abs() < 1e-6);
        assert_eq!(stb2, 0);

        // Dense `advance_into` matches `advance_single` at every state and token.
        let mut scores = vec![0.0f32; 32];
        let mut nexts = vec![0usize; 32];
        for st in 0..bt.num_states() {
            bt.advance_into(st, &mut scores, &mut nexts);
            for tok in 0..32u32 {
                let (es, en) = bt.advance_single(st, tok);
                assert!((scores[tok as usize] - es).abs() < 1e-6);
                assert_eq!(nexts[tok as usize], en);
            }
        }
    }

    #[test]
    fn arpa_lm_parses_and_backs_off_accurately() {
        // Minimal 2-gram ARPA with token_offset=100: token 0='d', token 1='e'.
        let arpa = "\
\\data\\
ngram 1=4
ngram 2=2

\\1-grams:
-2.0\t<unk>
0.0\t<s>\t-0.1
-0.5\td\t-0.2
-0.7\te\t-0.3

\\2-grams:
-0.25\t<s> d
-0.15\td e

\\end\\
";
        let lm = ArpaLm::from_arpa_str(arpa, 4, Some(100), None).unwrap();
        let bos = lm.init_state();
        assert_eq!(bos, 1);

        // From BOS -> token 0 ('d'): direct 2-gram -0.25 * ln(10).
        let (s_d, st_d) = lm.advance_single(bos, 0);
        assert!((s_d - (-0.25 * LN_10 as f32)).abs() < 1e-5);

        // From 'd' -> token 1 ('e'): direct 2-gram -0.15 * ln(10).
        let (s_e, _st_e) = lm.advance_single(st_d, 1);
        assert!((s_e - (-0.15 * LN_10 as f32)).abs() < 1e-5);

        // From BOS -> token 1 ('e'): backoff (-0.1) + unigram (-0.7) = -0.8 * ln(10).
        let (s_bo, _) = lm.advance_single(bos, 1);
        assert!((s_bo - (-0.8 * LN_10 as f32)).abs() < 1e-5);

        // Dense `advance_into` matches `advance_single`.
        let mut scores = vec![0.0f32; 4];
        let mut nexts = vec![0usize; 4];
        lm.advance_into(bos, &mut scores, &mut nexts);
        for tok in 0..4u32 {
            let (es, en) = lm.advance_single(bos, tok);
            assert!((scores[tok as usize] - es).abs() < 1e-6);
            assert_eq!(nexts[tok as usize], en);
        }
    }
}
