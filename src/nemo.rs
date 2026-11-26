use std::{collections::BTreeMap, fs::File, path::Path};

use anyhow::{Context, Result, bail};

use crate::torch::{DType, Store, read_checkpoint};

/// A `.nemo` archive opened in memory: the YAML config and the parsed weights.
pub struct NemoBundle {
    pub config_yaml: String,
    pub fluid_speech: FluidSpeechConfig,
    pub store: Store,
    /// `*_tokenizer.vocab` from the same archive: one piece per line.
    ///
    /// Kept for humans and for the punctuation scan, but **not** authoritative for
    /// detokenisation: NeMo decodes through the SentencePiece model below.
    pub vocab: Vec<u8>,
    /// `*_tokenizer.model`: the SentencePiece protobuf NeMo actually decodes with.
    /// Its piece table and unknown-piece surface differ from `vocab.txt`.
    pub tokenizer_model: Vec<u8>,
}

impl NemoBundle {
    /// Tensor inventory summary: total tensors, bytes per dtype, name prefixes.
    pub fn summary(&self) -> String {
        let mut by_dtype: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        let mut by_prefix: BTreeMap<String, usize> = BTreeMap::new();
        for tensor in &self.store.tensors {
            let entry = by_dtype.entry(tensor.dtype.name()).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += tensor.numel() * dtype_size(tensor.dtype);
            *by_prefix.entry(prefix_of(&tensor.name)).or_insert(0) += 1;
        }
        let dtypes = by_dtype
            .iter()
            .map(|(name, (count, bytes))| {
                format!(
                    "{name}: {count} tensors, {:.1} MiB",
                    *bytes as f64 / 1048576.0
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        let prefixes = by_prefix
            .iter()
            .map(|(name, count)| format!("{name}: {count}"))
            .collect::<Vec<_>>()
            .join("; ");
        format!(
            "tensors={} | {dtypes} | prefixes {{ {prefixes} }}",
            self.store.tensors.len()
        )
    }
}

fn dtype_size(dtype: DType) -> usize {
    dtype.size()
}

fn prefix_of(name: &str) -> String {
    name.split('.').take(2).collect::<Vec<_>>().join(".")
}

/// A tar header field, trimmed at its first NUL.
fn tar_field(header: &[u8], start: usize, len: usize) -> String {
    let raw = &header[start..start + len];
    let end = raw.iter().position(|byte| *byte == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).into_owned()
}

/// Walks tar headers in place and returns `(offset, length)` of `target`.
///
/// A `.nemo` is an uncompressed tar, so the payload can be addressed without
/// copying it: reading the 2.6 GB checkpoint into a `Vec` first cost about a
/// second of pure memcpy. GNU long-name entries (`L`) are honoured because tar
/// stores names longer than 100 bytes that way.
fn tar_entry(map: &[u8], target: &str) -> Option<(usize, usize)> {
    let mut at = 0usize;
    let mut long_name: Option<String> = None;

    while at + 512 <= map.len() {
        let header = &map[at..at + 512];
        if header.iter().all(|byte| *byte == 0) {
            break;
        }

        let size = usize::from_str_radix(tar_field(header, 124, 12).trim(), 8).unwrap_or(0);
        let kind = header[156];
        let data = at + 512;
        let payload_end = data.saturating_add(size).min(map.len());

        if kind == b'L' {
            // The long name becomes the name of the entry that follows.
            long_name = Some(String::from_utf8_lossy(&map[data..payload_end]).into_owned());
            at = data + size.div_ceil(512) * 512;
            continue;
        }

        let raw_name = long_name
            .take()
            .unwrap_or_else(|| tar_field(header, 0, 100));
        let base = raw_name.rsplit('/').next().unwrap_or(&raw_name);
        if base == target {
            return Some((data, payload_end - data));
        }
        at = data + size.div_ceil(512) * 512;
    }
    None
}

/// Reads a `.nemo` archive without writing anything to disk or copying the
/// checkpoint body: the file is mapped and every tensor is materialised once.
pub fn open(path: &Path) -> Result<NemoBundle> {
    use memmap2::Mmap;

    let opened = std::time::Instant::now();
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: the archive is only read, never written, for the lifetime of the
    // map, and every slice below is bounds-checked against `map`.
    let map = std::sync::Arc::new(
        unsafe { Mmap::map(&file) }.with_context(|| format!("map {}", path.display()))?,
    );

    let (config_at, config_len) =
        tar_entry(&map, "model_config.yaml").context("archive has no model_config.yaml")?;
    let config_yaml = String::from_utf8_lossy(&map[config_at..config_at + config_len]).into_owned();

    let tokenizer_model = find_member(&map, "_tokenizer.model")
        .map(|(start, len)| map[start..start + len].to_vec())
        .unwrap_or_default();
    let vocab = match find_tokenizer_vocab(&map) {
        Some((at, len)) => map[at..at + len].to_vec(),
        None => Vec::new(),
    };

    if std::env::var_os("FLUID_TRACE").is_some() {
        eprintln!(
            "[load] tar scan             {:.3}s",
            opened.elapsed().as_secs_f64()
        );
    }

    let (checkpoint_at, checkpoint_len) =
        tar_entry(&map, "model_weights.ckpt").context("archive has no model_weights.ckpt")?;
    let fluid_speech = parse_fluid_speech(&config_yaml)?;
    let _ = checkpoint_len;
    let store = read_checkpoint(std::sync::Arc::clone(&map), checkpoint_at)
        .context("parse model_weights.ckpt")?;

    Ok(NemoBundle {
        config_yaml,
        fluid_speech,
        store,
        vocab,
        tokenizer_model,
    })
}

/// Locates any `*_tokenizer.vocab` member, whose name carries a hash prefix.
fn find_tokenizer_vocab(map: &[u8]) -> Option<(usize, usize)> {
    find_member(map, "_tokenizer.vocab")
}

/// Locates a tar member by name suffix, returning `(offset, length)` in the map.
fn find_member(map: &[u8], suffix: &str) -> Option<(usize, usize)> {
    let mut at = 0usize;
    while at + 512 <= map.len() {
        let header = &map[at..at + 512];
        if header.iter().all(|byte| *byte == 0) {
            break;
        }
        let size = usize::from_str_radix(tar_field(header, 124, 12).trim(), 8).unwrap_or(0);
        let name = tar_field(header, 0, 100);
        let data = at + 512;
        if name.ends_with(suffix) {
            let end = data.saturating_add(size).min(map.len());
            return Some((data, end - data));
        }
        at = data + size.div_ceil(512) * 512;
    }
    None
}

#[derive(Debug, Clone, PartialEq)]
pub struct FluidSpeechConfig {
    pub insert: Option<String>,
    pub insert_dim: Option<usize>,
    pub insert_kernel: Option<usize>,
    pub rank: Option<usize>,
    pub alpha: Option<f32>,
}

/// Cross-checks the parsed inventory against a torch-produced metadata list.
///
/// The reference holds only names, dtypes and shapes — never weights — so this
/// proves the pickle/zip reader against the framework without shipping a copy of
/// the checkpoint.
pub fn verify_against_index(path: &Path, index_path: &Path) -> Result<()> {
    let bundle = open(path)?;
    let text = std::fs::read_to_string(index_path)
        .with_context(|| format!("read {}", index_path.display()))?;
    let reference: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(&text).context("parse tensor index")?;

    let mut missing = Vec::new();
    let mut extra = Vec::new();
    let mut mismatched = Vec::new();

    for (name, expected) in &reference {
        let Some(tensor) = bundle.store.get(name) else {
            missing.push(name.clone());
            continue;
        };
        let want_dtype = expected.get("dtype").and_then(|value| value.as_str());
        let want_shape = expected
            .get("shape")
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_u64().map(|number| number as usize))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let got_dtype = torch_dtype_name(tensor.dtype);
        if want_dtype != Some(got_dtype) || want_shape != tensor.shape {
            mismatched.push(format!(
                "{name}: want {want_dtype:?} {want_shape:?}, got {got_dtype} {:?}",
                tensor.shape
            ));
        }
    }
    for tensor in &bundle.store.tensors {
        if !reference.contains_key(&tensor.name) {
            extra.push(tensor.name.clone());
        }
    }

    println!(
        "reference={} parsed={} matched={} missing={} extra={} mismatched={}",
        reference.len(),
        bundle.store.tensors.len(),
        reference.len() - missing.len() - mismatched.len(),
        missing.len(),
        extra.len(),
        mismatched.len()
    );
    for line in missing.iter().take(10) {
        println!("missing {line}");
    }
    for line in extra.iter().take(10) {
        println!("extra {line}");
    }
    for line in mismatched.iter().take(10) {
        println!("mismatch {line}");
    }
    if !missing.is_empty() || !extra.is_empty() || !mismatched.is_empty() {
        bail!("inventory does not match the reference");
    }
    Ok(())
}

fn torch_dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::I64 => "int64",
        DType::I32 => "int32",
        DType::U8 => "uint8",
        DType::Bool => "bool",
    }
}

pub fn inspect(path: &Path) -> Result<()> {
    let bundle = open(path)?;
    let parsed = &bundle.fluid_speech;
    println!(
        "fluid_speech insert={:?} insert_dim={:?} insert_kernel={:?} rank={:?} alpha={:?}",
        parsed.insert, parsed.insert_dim, parsed.insert_kernel, parsed.rank, parsed.alpha
    );
    println!("{}", bundle.summary());
    Ok(())
}

pub fn parse_fluid_speech(config: &str) -> Result<FluidSpeechConfig> {
    let mut parsed = FluidSpeechConfig {
        insert: None,
        insert_dim: None,
        insert_kernel: None,
        rank: None,
        alpha: None,
    };
    let mut inside = false;
    for raw_line in config.lines() {
        let line = raw_line.trim_end();
        let trimmed = line.trim();
        if trimmed.starts_with("fluid_speech:") {
            inside = true;
            continue;
        }
        if !inside || trimmed.is_empty() {
            continue;
        }
        if !line.starts_with(' ') && !line.starts_with('\t') {
            break;
        }
        if let Some(value) = trimmed.strip_prefix("insert:") {
            parsed.insert = Some(value.trim().to_string());
        } else if let Some(value) = trimmed.strip_prefix("insert_dim:") {
            parsed.insert_dim = Some(value.trim().parse()?);
        } else if let Some(value) = trimmed.strip_prefix("insert_kernel:") {
            parsed.insert_kernel = Some(value.trim().parse()?);
        } else if let Some(value) = trimmed.strip_prefix("rank:") {
            parsed.rank = Some(value.trim().parse()?);
        } else if let Some(value) = trimmed.strip_prefix("alpha:") {
            parsed.alpha = Some(value.trim().parse()?);
        }
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fluid_speech_block() {
        let yaml = "\
other:
  x: 1
fluid_speech:
  insert: pre
  insert_dim: 1024
  insert_kernel: 15
  clean_mode:
    rank: 32
    alpha: 64.0
next:
  y: 2
";
        let parsed = parse_fluid_speech(yaml).unwrap();
        assert_eq!(parsed.insert.as_deref(), Some("pre"));
        assert_eq!(parsed.insert_dim, Some(1024));
        assert_eq!(parsed.insert_kernel, Some(15));
        assert_eq!(parsed.rank, Some(32));
        assert_eq!(parsed.alpha, Some(64.0));
    }

    #[test]
    fn tar_scan_finds_a_member() {
        // One 512-byte header naming `./model_config.yaml` with a 4-byte payload.
        let mut archive = vec![0u8; 512 + 512 + 1024];
        archive[..22].copy_from_slice(b"./model_config.yaml\0\0\0");
        archive[124..136].copy_from_slice(b"00000000004\0");
        archive[156] = b'0';
        archive[512..516].copy_from_slice(b"abc\n");

        let found = tar_entry(&archive, "model_config.yaml").unwrap();
        assert_eq!(found, (512, 4));
        assert!(tar_entry(&archive, "model_weights.ckpt").is_none());
    }
}
