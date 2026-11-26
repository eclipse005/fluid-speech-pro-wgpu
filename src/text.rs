//! SentencePiece detokenisation and NeMo-style timestamps.
//!
//! The vocabulary ships inside the `.nemo` archive (`*_tokenizer.vocab`), so
//! nothing is regenerated: piece `i` is line `i` of that file, and NeMo's
//! `ids_to_text` is "concatenate, turn `▁` into a space, drop the leading space".
//!
//! Timestamps follow NeMo's grid: TDT frames are 80 ms, a token occupies
//! `[frame, frame + duration)`, a word runs from its first token's start to its
//! last token's end, and a segment groups words up to and including the one that
//! ends a sentence.

use anyhow::{Result, bail};

use crate::tdt::{FRAME_SECONDS, Token};

/// Which timestamp tiers to compute — mirrors NeMo's `rnnt_timestamp_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampType {
    None,
    Char,
    Word,
    Segment,
    All,
}

/// Timestamp computation settings — mirrors NeMo's `RNNTDecodingConfig` timestamp fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimestampConfig {
    pub timestamp_type: TimestampType,
    /// Word delimiter character/string (default `" "`).
    pub word_separator: String,
    /// Sentence/segment boundary punctuation tokens (default `[".", "!", "?"]`).
    pub segment_separators: Vec<String>,
    /// Optional inter-word pause threshold in encoder frames (80 ms each) to form
    /// segments, matching NeMo's `segment_gap_threshold`.
    pub segment_gap_threshold: Option<u32>,
}

impl Default for TimestampConfig {
    fn default() -> Self {
        Self {
            timestamp_type: TimestampType::All,
            word_separator: " ".to_string(),
            segment_separators: vec![".".to_string(), "!".to_string(), "?".to_string()],
            segment_gap_threshold: None,
        }
    }
}

/// `word`, `start_offset`, `end_offset`, `start`, `end` — NeMo's field set.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub word: String,
    pub start_offset: u32,
    pub end_offset: u32,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub segment: String,
    pub start_offset: u32,
    pub end_offset: u32,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CharTimestamp {
    pub char: String,
    pub start_offset: u32,
    pub end_offset: u32,
    pub start: f32,
    pub end: f32,
}

pub struct Tokenizer {
    pieces: Vec<String>,
    /// Raw SentencePiece piece spellings (e.g. `"<unk>"` rather than `" ⁇ "`), matching
    /// NeMo's `decode_ids_to_tokens` for word-confidence boundary checks.
    raw_pieces: Vec<String>,
    /// Piece string -> `(token_id, bpe_merge_score)` for SentencePiece BPE encoding.
    encode_table: std::collections::HashMap<String, (u32, f32)>,
    /// Token ID of SentencePiece's UNKNOWN piece (if any).
    unk_id: Option<u32>,
    /// Characters NeMo treats as punctuation, from `extract_punctuation_from_vocab`.
    punctuation: std::collections::HashSet<char>,
}

/// A vocab token the punctuation scan skips: `[...]`, `<...>`, `##...`, a
/// word-boundary marker, or whitespace only.
fn is_special_vocab_token(token: &str) -> bool {
    if token.starts_with('[') && token.ends_with(']') {
        return true;
    }
    if token.starts_with('<') && token.ends_with('>') {
        return true;
    }
    if token.starts_with("##") || token.starts_with('\u{2581}') {
        return true;
    }
    token.trim().is_empty()
}

fn is_punctuation_char(ch: char) -> bool {
    use unicode_general_category::{GeneralCategory, get_general_category};
    matches!(
        get_general_category(ch),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::OtherPunctuation
    )
}

impl Tokenizer {
    /// Builds a tokenizer from the human-readable `vocab.txt` listing.
    ///
    /// Only correct when no SentencePiece model is available: the listing's piece
    /// surface forms and its unknown symbol are not what NeMo's `sp.decode` emits.
    pub fn new(vocab: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(vocab).unwrap_or_default();
        let mut pieces = Vec::new();
        let mut scores = Vec::new();
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            // `<piece>\t<score>`
            let mut parts = line.split('\t');
            let piece = parts.next().unwrap_or(line);
            let score = parts
                .next()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(-(pieces.len() as f32));
            pieces.push(piece.to_string());
            scores.push(score);
        }
        let unk_id = pieces
            .iter()
            .position(|p| p == "<unk>")
            .map(|i| i as u32);
        let normal_flags: Vec<bool> = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| unk_id != Some(i as u32) && !(p.starts_with("<|") && p.ends_with("|>")) && p != "<pad>")
            .collect();
        Self::from_pieces_with_raw(pieces.clone(), pieces, scores, normal_flags, unk_id)
    }

    /// Builds a tokenizer from the archive's SentencePiece model.
    ///
    /// This is the table NeMo decodes with, so it is the authoritative source: its
    /// pieces carry the real surface forms, and an `UNKNOWN` piece must render as
    /// SentencePiece's unknown surface rather than the spelling used in
    /// `vocab.txt`. Fixing the source here means the unknown-symbol and
    /// spacing cases need no special handling downstream.
    pub fn from_model(model: &[u8], vocab: &[u8]) -> Result<Self> {
        let parsed = parse_sentencepiece_model(model);
        if parsed.is_empty() {
            // A model-less archive still ships the listing; only fall back when the
            // model genuinely cannot be read.
            return Self::new(vocab);
        }
        let unknown = find_unk_surface(model).unwrap_or_else(|| FALLBACK_UNK_SURFACE.to_string());
        let mut unk_id = None;
        let mut raw_pieces = Vec::with_capacity(parsed.len());
        let mut pieces = Vec::with_capacity(parsed.len());
        let mut scores = Vec::with_capacity(parsed.len());
        let mut normal_flags = Vec::with_capacity(parsed.len());
        for (i, piece) in parsed.iter().enumerate() {
            raw_pieces.push(piece.text.clone());
            scores.push(piece.score);
            normal_flags.push(piece.kind == PIECE_NORMAL);
            if piece.kind == PIECE_UNKNOWN {
                unk_id = Some(i as u32);
                pieces.push(unknown.clone());
            } else {
                pieces.push(piece.text.clone());
            }
        }
        Self::from_pieces_with_raw(pieces, raw_pieces, scores, normal_flags, unk_id)
    }

    fn from_pieces_with_raw(
        pieces: Vec<String>,
        raw_pieces: Vec<String>,
        scores: Vec<f32>,
        normal_flags: Vec<bool>,
        unk_id: Option<u32>,
    ) -> Result<Self> {
        if pieces.is_empty() {
            bail!("tokenizer has no pieces");
        }
        let mut punctuation = std::collections::HashSet::new();
        for piece in &pieces {
            if is_special_vocab_token(piece) {
                continue;
            }
            for ch in piece.chars() {
                if is_punctuation_char(ch) {
                    punctuation.insert(ch);
                }
            }
        }
        let mut encode_table = std::collections::HashMap::with_capacity(raw_pieces.len());
        for (id, ((piece, &score), &is_normal)) in raw_pieces
            .iter()
            .zip(&scores)
            .zip(&normal_flags)
            .enumerate()
        {
            if is_normal {
                encode_table.insert(piece.clone(), (id as u32, score));
            }
        }
        Ok(Self {
            pieces,
            raw_pieces,
            encode_table,
            unk_id,
            punctuation,
        })
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    pub fn piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(String::as_str)
    }

    pub fn raw_piece(&self, id: u32) -> Option<&str> {
        self.raw_pieces.get(id as usize).map(String::as_str)
    }

    pub fn is_unk(&self, id: u32) -> bool {
        self.unk_id == Some(id)
    }

    pub fn piece_id(&self, piece: &str) -> Option<u32> {
        self.encode_table.get(piece).map(|&(id, _)| id)
    }

    /// Encodes `text` into SentencePiece BPE token IDs (`add_dummy_prefix = true`,
    /// `escape_whitespaces = true`, `remove_extra_whitespaces = false`).
    ///
    /// Mirrors SentencePiece's BPE merge order (`ModelProto.trainer_spec.model_type = BPE`)
    /// so phrase boosting trees tokenize identically to NeMo's `tokenizer.text_to_ids`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut normalized = String::with_capacity(text.len() + 3);
        normalized.push('\u{2581}');
        for ch in text.chars() {
            if ch == ' ' {
                normalized.push('\u{2581}');
            } else {
                normalized.push(ch);
            }
        }
        let mut symbols: Vec<String> = normalized.chars().map(|c| c.to_string()).collect();
        let mut merged = String::new();
        loop {
            let mut best_pos: Option<usize> = None;
            let mut best_score = f32::NEG_INFINITY;
            for i in 0..symbols.len().saturating_sub(1) {
                merged.clear();
                merged.push_str(&symbols[i]);
                merged.push_str(&symbols[i + 1]);
                if let Some(&(_, score)) = self.encode_table.get(&merged) {
                    if score > best_score {
                        best_score = score;
                        best_pos = Some(i);
                    }
                }
            }
            let Some(pos) = best_pos else {
                break;
            };
            let right = symbols.remove(pos + 1);
            symbols[pos].push_str(&right);
        }
        let unk = self.unk_id.unwrap_or(0);
        let mut ids = Vec::with_capacity(symbols.len());
        let mut prev_unk = false;
        for sym in symbols {
            if let Some(&(id, _)) = self.encode_table.get(&sym) {
                ids.push(id);
                prev_unk = false;
            } else if !prev_unk {
                ids.push(unk);
                prev_unk = true;
            }
        }
        ids
    }

    /// NeMo's `_refine_timestamps_tdt`: a token starting with punctuation takes
    /// the previous token's end as both its start and its end, because the model
    /// often emits punctuation after a pause.
    pub fn char_offsets(&self, tokens: &[Token]) -> Vec<(u32, u32)> {
        let mut offsets: Vec<(u32, u32)> = tokens
            .iter()
            .map(|token| (token.frame, token.frame + token.duration))
            .collect();
        for index in 0..offsets.len() {
            let Some(piece) = self.piece(tokens[index].id) else {
                continue;
            };
            // NeMo `_refine_timestamps_tdt` checks `decoded_piece in supported_punctuation`
            // (after stripping `▁`, so tokens like `"▁¿"` are recognized as single punctuation
            // marks, while multi-character tokens or `<unk>` (`" ⁇ "`) are not).
            let decoded = clean_piece(piece);
            let mut chars = decoded.chars();
            let is_single_punct = match (chars.next(), chars.next()) {
                (Some(ch), None) => ch != '⁇' && self.punctuation.contains(&ch),
                _ => false,
            };
            if index > 0 && is_single_punct {
                let anchored = offsets[index - 1].1;
                offsets[index] = (anchored, anchored);
            }
        }
        offsets
    }

    /// NeMo's `ids_to_text`: join the pieces, `▁` -> space, drop the leading space.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        for id in ids {
            if let Some(piece) = self.piece(*id) {
                text.push_str(piece);
            }
        }
        let text = text.replace('\u{2581}', " ");
        let text = text.strip_prefix(' ').unwrap_or(&text).to_string();
        // NeMo's `decode_tokens_to_str_with_strip_punctuation` drops exactly one
        // whitespace before a supported punctuation mark, and only when the model
        // declares punctuation support. Some models carry punctuation pieces with a
        // leading space, so the raw join leaves a stray gap.
        if self.punctuation.is_empty() {
            text
        } else {
            strip_one_space_before_punctuation(&text, &self.punctuation)
        }
    }
}

/// Removes a single whitespace character immediately before any punctuation mark.
fn strip_one_space_before_punctuation(
    text: &str,
    punctuation: &std::collections::HashSet<char>,
) -> String {
    // The unknown glyph is `OtherPunctuation` but is not a model punctuation mark:
    // NeMo builds its set from the model config, so stripping here would eat the
    // space SentencePiece puts on either side of an unknown piece.
    let is_punctuation = |ch: char| ch != '⁇' && punctuation.contains(&ch);
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut index = 0usize;
    while index < chars.len() {
        let ch = chars[index];
        let next_is_punctuation = chars
            .get(index + 1)
            .is_some_and(|next| is_punctuation(*next));
        if ch.is_whitespace() && next_is_punctuation {
            index += 1;
            continue;
        }
        out.push(ch);
        index += 1;
    }
    out
}

/// How NeMo renders a piece inside a word (no word-boundary marker).
fn clean_piece(piece: &str) -> String {
    piece.replace('\u{2581}', "")
}

fn seconds(offset: u32) -> f32 {
    offset as f32 * FRAME_SECONDS
}

/// Groups tokens into words: a token starting with `▁` opens a new word.
///
/// Offsets come from [`Tokenizer::char_offsets`], so the punctuation refinement
/// is already applied.
pub fn words(tokenizer: &Tokenizer, tokens: &[Token]) -> Vec<Word> {
    let offsets = tokenizer.char_offsets(tokens);
    let mut words = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        let piece = tokenizer.piece(token.id).unwrap_or("");
        let starts_word = piece.starts_with('\u{2581}');
        if starts_word && !current.is_empty() {
            words.push(build_word(tokenizer, tokens, &offsets, &current));
            current.clear();
        }
        current.push(index);
    }
    if !current.is_empty() {
        words.push(build_word(tokenizer, tokens, &offsets, &current));
    }
    words
}

fn build_word(
    tokenizer: &Tokenizer,
    tokens: &[Token],
    offsets: &[(u32, u32)],
    indices: &[usize],
) -> Word {
    let mut text = String::new();
    for &index in indices {
        text.push_str(&clean_piece(
            tokenizer.piece(tokens[index].id).unwrap_or(""),
        ));
    }
    // Callers only pass non-empty slices; fall back to a zero word rather than
    // panicking if that ever stops holding.
    let Some(&first) = indices.first() else {
        return Word {
            word: text,
            start_offset: 0,
            end_offset: 0,
            start: 0.0,
            end: 0.0,
        };
    };
    let last = *indices.last().unwrap_or(&first);
    let start_offset = offsets.get(first).map_or(0, |pair| pair.0);
    let end_offset = offsets.get(last).map_or(start_offset, |pair| pair.1);
    Word {
        word: text,
        start_offset,
        end_offset,
        start: seconds(start_offset),
        end: seconds(end_offset),
    }
}

/// Groups words into sentences using the default `.`, `!`, `?` delimiters.
pub fn segments(words: &[Word]) -> Vec<Segment> {
    segments_with(
        words,
        &[".".to_string(), "!".to_string(), "?".to_string()],
        None,
    )
}

/// Groups words into segments using NeMo's `get_segment_offsets` rules:
/// custom `segment_separators` and optional `segment_gap_threshold` (in 80 ms frames).
pub fn segments_with(
    words: &[Word],
    separators: &[String],
    gap_threshold: Option<u32>,
) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut current: Vec<&Word> = Vec::new();
    for (i, word) in words.iter().enumerate() {
        if let Some(threshold) = gap_threshold {
            if threshold > 0 && !current.is_empty() {
                let prev_end = words[i - 1].end_offset;
                let gap = word.start_offset.saturating_sub(prev_end);
                if gap >= threshold {
                    segments.push(build_segment(&current));
                    current.clear();
                    current.push(word);
                    continue;
                }
                current.push(word);
                continue;
            }
        }
        current.push(word);
        let trimmed = word.word.trim_end_matches(['"', '\'', ')', ']']);
        let ends = separators.iter().any(|sep| {
            !sep.is_empty() && (word.word == *sep || word.word.ends_with(sep.as_str()) || trimmed.ends_with(sep.as_str()))
        });
        if ends {
            segments.push(build_segment(&current));
            current.clear();
        }
    }
    if !current.is_empty() {
        segments.push(build_segment(&current));
    }
    segments
}

fn build_segment(words: &[&Word]) -> Segment {
    let Some(first) = words.first() else {
        return Segment {
            segment: String::new(),
            start_offset: 0,
            end_offset: 0,
            start: 0.0,
            end: 0.0,
        };
    };
    let last = words.last().unwrap_or(first);
    let text = words
        .iter()
        .map(|word| word.word.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    Segment {
        segment: text,
        start_offset: first.start_offset,
        end_offset: last.end_offset,
        start: first.start,
        end: last.end,
    }
}

/// One entry per token, with the `▁` marker stripped — NeMo's `timestamp['char']`.
pub fn chars(tokenizer: &Tokenizer, tokens: &[Token]) -> Vec<CharTimestamp> {
    let offsets = tokenizer.char_offsets(tokens);
    tokens
        .iter()
        .enumerate()
        .map(|(index, token)| {
            let piece = clean_piece(tokenizer.piece(token.id).unwrap_or(""));
            let (start_offset, end_offset) = offsets
                .get(index)
                .copied()
                .unwrap_or((token.frame, token.frame + token.duration));
            CharTimestamp {
                char: piece,
                start_offset,
                end_offset,
                start: seconds(start_offset),
                end: seconds(end_offset),
            }
        })
        .collect()
}

/// SentencePiece `Type` constants (`ModelProto.SentencePiece.type`).
const PIECE_NORMAL: u64 = 1;
const PIECE_UNKNOWN: u64 = 2;

/// Fallback unknown surface when the model does not carry one.
///
/// SentencePiece's `TrainerSpec.unk_surface` is authoritative and is read from the
/// model; `vocab.txt` spells the same piece `<unk>`, which is what made decoding
/// from the listing wrong in the first place.
/// SentencePiece spells an unknown piece as `⁇` with a space on each side; this is
/// what `sp.decode([sp.unk_id()])` returns for a model whose `TrainerSpec` leaves
/// `unk_surface` at its default.
const FALLBACK_UNK_SURFACE: &str = " \u{2047} ";

/// Finds `TrainerSpec.unk_surface` by walking the model's nested messages.
///
/// The field number varies across SentencePiece versions, so this looks for the
/// short string field containing the unknown glyph instead of hard-coding a tag.
fn find_unk_surface(model: &[u8]) -> Option<String> {
    find_unk_surface_at(model, 2)
}

fn find_unk_surface_at(buffer: &[u8], depth: usize) -> Option<String> {
    let mut at = 0usize;
    while at < buffer.len() {
        let key = read_varint(buffer, &mut at)?;
        let field = key >> 3;
        match key & 7 {
            0 => {
                read_varint(buffer, &mut at)?;
            }
            1 => at += 8,
            5 => at += 4,
            2 => {
                let len = read_varint(buffer, &mut at)? as usize;
                let end = at.checked_add(len)?;
                if end > buffer.len() {
                    return None;
                }
                let slice = &buffer[at..end];
                // `ModelProto` field 1 is `pieces`, which itself contains a `⁇`
                // entry, so searching it would return that piece instead of the
                // unknown surface. The surface lives in `trainer_spec` (field 2) or
                // `normalizer_spec` (field 3).
                if depth > 0 {
                    if (field == 2 || field == 3)
                        && let Some(found) = find_unk_surface_at(slice, depth - 1)
                    {
                        return Some(found);
                    }
                } else if let Ok(text) = std::str::from_utf8(slice)
                    && text.contains('\u{2047}')
                    && text.len() <= 16
                {
                    return Some(text.to_string());
                }
                at = end;
            }
            _ => return None,
        }
    }
    None
}

/// One `ModelProto.SentencePiece`.
struct SpPiece {
    text: String,
    score: f32,
    kind: u64,
}

/// Reads a base-128 varint.
fn read_varint(buffer: &[u8], at: &mut usize) -> Option<u64> {
    let mut value = 0u64;
    let mut shift = 0u32;
    while *at < buffer.len() {
        let byte = buffer[*at];
        *at += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
    None
}

/// Walks the top-level protobuf fields, collecting `ModelProto.pieces`.
///
/// Only two fields matter here (`pieces`, and inside each piece `piece`/`type`), so
/// the wire format is walked directly instead of pulling in a protobuf dependency.
fn parse_sentencepiece_model(buffer: &[u8]) -> Vec<SpPiece> {
    let mut pieces = Vec::new();
    let mut at = 0usize;
    while at < buffer.len() {
        let Some(key) = read_varint(buffer, &mut at) else {
            break;
        };
        let field = key >> 3;
        let wire = key & 7;
        match wire {
            0 => {
                if read_varint(buffer, &mut at).is_none() {
                    break;
                }
            }
            1 => at += 8,
            5 => at += 4,
            2 => {
                let Some(len) = read_varint(buffer, &mut at) else {
                    break;
                };
                let end = at.saturating_add(len as usize);
                if end > buffer.len() {
                    break;
                }
                // Kept as a single `let` rather than a nested `if`: clippy folds the
                // two together, and this reads the same.
                let piece = if field == 1 {
                    parse_piece_message(&buffer[at..end])
                } else {
                    None
                };
                if let Some(piece) = piece {
                    pieces.push(piece);
                }
                at = end;
            }
            _ => break,
        }
    }
    pieces
}

/// Reads one `SentencePiece` message: `piece` (field 1), `score` (field 2), `type` (field 3).
fn parse_piece_message(buffer: &[u8]) -> Option<SpPiece> {
    let mut text: Option<String> = None;
    let mut score = 0.0f32;
    let mut kind = PIECE_NORMAL;
    let mut at = 0usize;
    while at < buffer.len() {
        let key = read_varint(buffer, &mut at)?;
        let field = key >> 3;
        match key & 7 {
            0 => {
                let value = read_varint(buffer, &mut at)?;
                if field == 3 {
                    kind = value;
                }
            }
            1 => at += 8,
            5 => {
                if at + 4 > buffer.len() {
                    return None;
                }
                if field == 2 {
                    score = f32::from_le_bytes([
                        buffer[at],
                        buffer[at + 1],
                        buffer[at + 2],
                        buffer[at + 3],
                    ]);
                }
                at += 4;
            }
            2 => {
                let len = read_varint(buffer, &mut at)? as usize;
                let end = at.saturating_add(len);
                if end > buffer.len() {
                    return None;
                }
                if field == 1 {
                    text = Some(String::from_utf8_lossy(&buffer[at..end]).into_owned());
                }
                at = end;
            }
            _ => return None,
        }
    }
    text.map(|text| SpPiece { text, score, kind })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer(pieces: &[&str]) -> Tokenizer {
        let text = pieces
            .iter()
            .enumerate()
            .map(|(index, piece)| format!("{piece}\t-{index}"))
            .collect::<Vec<_>>()
            .join("\n");
        Tokenizer::new(text.as_bytes()).unwrap()
    }

    #[test]
    fn decode_joins_pieces_and_replaces_word_boundaries() {
        let tk = tokenizer(&["<unk>", "\u{2581}Hello", " world"]);
        assert_eq!(tk.decode(&[1, 2]), "Hello world");
    }

    #[test]
    fn words_split_on_the_boundary_marker() {
        let tk = tokenizer(&["<unk>", "\u{2581}Ü", "ber", "\u{2581}die"]);
        let tokens = [
            Token {
                id: 1,
                frame: 47,
                duration: 3,
                duration_margin: 1.0,
                duration_logits: [0.0; 5],
                score: 0.0,
            },
            Token {
                id: 2,
                frame: 50,
                duration: 3,
                duration_margin: 1.0,
                duration_logits: [0.0; 5],
                score: 0.0,
            },
            Token {
                id: 3,
                frame: 53,
                duration: 3,
                duration_margin: 1.0,
                duration_logits: [0.0; 5],
                score: 0.0,
            },
        ];
        let words = words(&tk, &tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "Über");
        assert_eq!(words[0].start_offset, 47);
        assert_eq!(words[0].end_offset, 53);
        assert!((words[0].end - 4.24).abs() < 1e-5);
        assert_eq!(words[1].word, "die");
    }

    #[test]
    fn segments_close_on_sentence_punctuation() {
        let tk = tokenizer(&["<unk>", "\u{2581}a.", "\u{2581}b"]);
        let tokens = [
            Token {
                id: 1,
                frame: 0,
                duration: 2,
                duration_margin: 1.0,
                duration_logits: [0.0; 5],
                score: 0.0,
            },
            Token {
                id: 2,
                frame: 2,
                duration: 2,
                duration_margin: 1.0,
                duration_logits: [0.0; 5],
                score: 0.0,
            },
        ];
        let words = words(&tk, &tokens);
        let segments = segments(&words);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].segment, "a.");
    }

    #[test]
    fn segments_split_on_gap_threshold() {
        let tk = tokenizer(&["<unk>", "\u{2581}hello", "\u{2581}world", "\u{2581}later"]);
        let tokens = [
            Token {
                id: 1,
                frame: 0,
                duration: 2,
                duration_margin: 1.0,
                duration_logits: [0.0; 5],
                score: 0.0,
            },
            Token {
                id: 2,
                frame: 3,
                duration: 2,
                duration_margin: 1.0,
                duration_logits: [0.0; 5],
                score: 0.0,
            },
            Token {
                id: 3,
                frame: 12,
                duration: 2,
                duration_margin: 1.0,
                duration_logits: [0.0; 5],
                score: 0.0,
            },
        ];
        let words = words(&tk, &tokens);
        let segs = segments_with(&words, &[".".into()], Some(5));
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].segment, "hello world");
        assert_eq!(segs[1].segment, "later");
    }

    #[test]
    fn char_timestamps_follow_the_token_grid() {
        let tk = tokenizer(&["<unk>", "\u{2581}ab", "cd"]);
        let tokens = [
            Token {
                id: 1,
                frame: 10,
                duration: 3,
                duration_margin: 1.0,
                duration_logits: [0.0; 5],
                score: 0.0,
            },
            Token {
                id: 2,
                frame: 13,
                duration: 2,
                duration_margin: 1.0,
                duration_logits: [0.0; 5],
                score: 0.0,
            },
        ];
        let chars = chars(&tk, &tokens);
        assert_eq!(chars[0].char, "ab");
        assert_eq!(chars[0].start_offset, 10);
        assert_eq!(chars[0].end_offset, 13);
        assert_eq!(chars[1].end_offset, 15);
    }
}
