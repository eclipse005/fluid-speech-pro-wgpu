//! Minimal read-only PyTorch checkpoint reader (the `zip` layout).
//!
//! A `.nemo` archive holds `model_weights.ckpt`, which PyTorch writes as a zip:
//! `archive/data.pkl` (a pickle of the state dict) plus `archive/data/{key}` raw
//! storages. Nothing is converted or written out: the tarball is read, the
//! checkpoint is unpacked in memory, and tensors are handed back as shape + dtype
//! + a borrowed slice of the storage bytes.

use std::collections::HashMap;
use std::io::{Cursor, Read};

use anyhow::{Context, Result, bail};

/// Element type of a stored tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    F16,
    BF16,
    I64,
    I32,
    U8,
    Bool,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::F16 | DType::BF16 => 2,
            DType::U8 | DType::Bool => 1,
        }
    }

    fn from_storage_name(name: &str) -> Option<Self> {
        Some(match name {
            "FloatStorage" => DType::F32,
            "DoubleStorage" => DType::F64,
            "HalfStorage" => DType::F16,
            "BFloat16Storage" => DType::BF16,
            "LongStorage" => DType::I64,
            "IntStorage" => DType::I32,
            "ByteStorage" => DType::U8,
            "BoolStorage" => DType::Bool,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::U8 => "u8",
            DType::Bool => "bool",
        }
    }
}

/// One tensor of the checkpoint, addressed inside a shared storage.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub offset: usize,
    pub storage_key: String,
    storage: usize,
}

impl TensorEntry {
    /// Number of stored elements this tensor maps to.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Contiguous f32 copy, de-strided when the tensor is a view.
    pub fn to_f32(&self, store: &Store) -> Result<Vec<f32>> {
        let storage = store
            .storages
            .get(self.storage)
            .with_context(|| format!("tensor {} points at a missing storage", self.name))?;
        let mut out = Vec::with_capacity(self.numel());
        if self.shape.is_empty() {
            return Ok(out);
        }
        if self.stride == contiguous_stride(&self.shape) {
            let start = self.offset;
            let end = start + self.numel();
            if end > storage.len() {
                bail!(
                    "tensor {} runs past its storage ({}..{} > {})",
                    self.name,
                    start,
                    end,
                    storage.len()
                );
            }
            // Bulk path: the overwhelming majority of tensors are contiguous f32,
            // and `f32_at` re-checks the dtype for every element.
            if storage.dtype == DType::F32 && storage.little_endian {
                let span = storage.span();
                let slice = &span[start * 4..end * 4];
                let (words, _remainder) = slice.as_chunks::<4>();
                return Ok(words.iter().map(|word| f32::from_le_bytes(*word)).collect());
            }
            for index in start..end {
                out.push(storage.f32_at(index));
            }
            return Ok(out);
        }
        let mut cursor = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            let mut index = self.offset;
            for (axis, &position) in cursor.iter().enumerate() {
                index += position * self.stride[axis];
            }
            out.push(storage.f32_at(index));
            for axis in (0..cursor.len()).rev() {
                cursor[axis] += 1;
                if cursor[axis] < self.shape[axis] {
                    break;
                }
                cursor[axis] = 0;
            }
        }
        Ok(out)
    }

    /// Raw little-endian f32 bytes for a tensor, borrowing straight from the mmap
    /// when the tensor is contiguous little-endian F32 (the common case for every
    /// weight matrix in `.nemo`). Falls back to converting via [`Self::to_f32`].
    pub fn to_f32_bytes<'a>(&self, store: &'a Store) -> Result<std::borrow::Cow<'a, [u8]>> {
        let storage = store
            .storages
            .get(self.storage)
            .with_context(|| format!("tensor {} points at a missing storage", self.name))?;
        if self.shape.is_empty() {
            return Ok(std::borrow::Cow::Borrowed(&[]));
        }
        if self.stride == contiguous_stride(&self.shape)
            && storage.dtype == DType::F32
            && storage.little_endian
        {
            let start = self.offset;
            let end = start + self.numel();
            if end > storage.len() {
                bail!(
                    "tensor {} runs past its storage ({}..{} > {})",
                    self.name,
                    start,
                    end,
                    storage.len()
                );
            }
            return Ok(std::borrow::Cow::Borrowed(&storage.span()[start * 4..end * 4]));
        }
        let values = self.to_f32(store)?;
        Ok(std::borrow::Cow::Owned(
            bytemuck::cast_slice::<f32, u8>(&values).to_vec(),
        ))
    }

    /// i64 view, used by counters such as `num_batches_tracked`.
    pub fn to_i64(&self, store: &Store) -> Result<Vec<i64>> {
        let storage = store
            .storages
            .get(self.storage)
            .with_context(|| format!("tensor {} points at a missing storage", self.name))?;
        Ok((self.offset..self.offset + self.numel())
            .map(|index| storage.i64_at(index))
            .collect())
    }
}

fn contiguous_stride(shape: &[usize]) -> Vec<usize> {
    let mut stride = vec![1usize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        stride[axis] = stride[axis + 1] * shape[axis + 1];
    }
    stride
}

/// Raw bytes of one storage, kept as downloaded.
pub struct Storage {
    pub dtype: DType,
    /// The whole mapped archive; every storage is a span inside it.
    map: std::sync::Arc<memmap2::Mmap>,
    start: usize,
    byte_len: usize,
    pub little_endian: bool,
}

impl Storage {
    /// The storage's bytes, borrowed straight from the mapping.
    fn span(&self) -> &[u8] {
        &self.map[self.start..self.start + self.byte_len]
    }
}

impl Storage {
    pub fn len(&self) -> usize {
        self.span().len() / self.dtype.size().max(1)
    }

    pub fn is_empty(&self) -> bool {
        self.span().is_empty()
    }

    fn f32_at(&self, index: usize) -> f32 {
        match self.dtype {
            DType::F32 => self.read_f32(index),
            DType::F64 => self.read_f64(index) as f32,
            DType::F16 => f16_to_f32(self.read_u16(index)),
            DType::BF16 => f32::from_bits(u32::from(self.read_u16(index)) << 16),
            DType::I32 => self.read_i32(index) as f32,
            DType::I64 => self.read_i64(index) as f32,
            DType::U8 => self.span()[index] as f32,
            DType::Bool => (self.span()[index] != 0) as u8 as f32,
        }
    }

    fn i64_at(&self, index: usize) -> i64 {
        match self.dtype {
            DType::I64 => self.read_i64(index),
            DType::I32 => self.read_i32(index) as i64,
            DType::U8 => i64::from(self.span()[index]),
            _ => self.f32_at(index) as i64,
        }
    }

    fn read_u16(&self, index: usize) -> u16 {
        let at = index * 2;
        let pair = [self.span()[at], self.span()[at + 1]];
        if self.little_endian {
            u16::from_le_bytes(pair)
        } else {
            u16::from_be_bytes(pair)
        }
    }

    fn read_f32(&self, index: usize) -> f32 {
        let at = index * 4;
        let word = [
            self.span()[at],
            self.span()[at + 1],
            self.span()[at + 2],
            self.span()[at + 3],
        ];
        if self.little_endian {
            f32::from_le_bytes(word)
        } else {
            f32::from_be_bytes(word)
        }
    }

    fn read_f64(&self, index: usize) -> f64 {
        let at = index * 8;
        let mut word = [0u8; 8];
        word.copy_from_slice(&self.span()[at..at + 8]);
        if self.little_endian {
            f64::from_le_bytes(word)
        } else {
            f64::from_be_bytes(word)
        }
    }

    fn read_i32(&self, index: usize) -> i32 {
        let at = index * 4;
        let word = [
            self.span()[at],
            self.span()[at + 1],
            self.span()[at + 2],
            self.span()[at + 3],
        ];
        if self.little_endian {
            i32::from_le_bytes(word)
        } else {
            i32::from_be_bytes(word)
        }
    }

    fn read_i64(&self, index: usize) -> i64 {
        let at = index * 8;
        let mut word = [0u8; 8];
        word.copy_from_slice(&self.span()[at..at + 8]);
        if self.little_endian {
            i64::from_le_bytes(word)
        } else {
            i64::from_be_bytes(word)
        }
    }
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    let out = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut mantissa = mantissa;
            let mut shift = 0u32;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                shift += 1;
            }
            mantissa &= 0x3ff;
            let exponent = 127 - 15 - shift;
            sign | (exponent << 23) | (mantissa << 13)
        }
    } else if exponent == 0x1f {
        sign | (0xff << 23) | (mantissa << 13)
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(out)
}

/// A parsed checkpoint: tensor metadata plus the storages it points into.
pub struct Store {
    pub tensors: Vec<TensorEntry>,
    storages: Vec<Storage>,
    by_name: HashMap<String, usize>,
    /// When `Some(scale)`, [`Self::f32`] and [`Self::f32_bytes`] transparently fold
    /// Fluid Speech Pro's `clean_mode.*` LoRA / bias adapters (`scale = alpha / rank`)
    /// into the returned tensors without mutating the mmap or writing to disk.
    clean_mode_scale: Option<f32>,
}

impl Store {
    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.by_name.get(name).map(|&index| &self.tensors[index])
    }

    pub fn require(&self, name: &str) -> Result<&TensorEntry> {
        self.get(name)
            .with_context(|| format!("checkpoint has no tensor {name}"))
    }

    /// Enables Fluid Speech Pro's `clean_mode` adapter folding (`scale = alpha / rank`)
    /// in memory. Matches `fluid_speech.py::fold_clean_mode` exactly.
    pub fn enable_clean_mode(&mut self, scale: f32) -> Result<()> {
        let dec_in = self
            .require("clean_mode.decoder_input")
            .context("clean_mode requested, but checkpoint has no clean_mode.decoder_input")?;
        let joint_h = self
            .require("clean_mode.joint_hidden")
            .context("clean_mode requested, but checkpoint has no clean_mode.joint_hidden")?;
        if dec_in.shape != [640] {
            bail!("unexpected clean_mode.decoder_input shape {:?}", dec_in.shape);
        }
        if joint_h.shape != [640] {
            bail!("unexpected clean_mode.joint_hidden shape {:?}", joint_h.shape);
        }
        // Validate every `clean_mode.<name>.A`/`.B` pair has its base encoder weight.
        for tensor in &self.tensors {
            if let Some(rest) = tensor.name.strip_prefix("clean_mode.") {
                if let Some(module) = rest.strip_suffix(".A") {
                    let b_name = format!("clean_mode.{module}.B");
                    let base_name = format!("encoder.{module}.weight");
                    self.require(&b_name)
                        .with_context(|| format!("clean_mode pair missing {b_name}"))?;
                    self.require(&base_name).with_context(|| {
                        format!("clean_mode adapts {base_name}, which the model does not have")
                    })?;
                }
            }
        }
        self.clean_mode_scale = Some(scale);
        Ok(())
    }

    /// Returns `Some(scale)` when `clean_mode` folding is active.
    pub fn clean_mode_scale(&self) -> Option<f32> {
        self.clean_mode_scale
    }

    /// True if `name` is modified by `clean_mode` folding.
    fn clean_mode_encoder_module<'a>(&self, name: &'a str) -> Option<&'a str> {
        let inner = name.strip_prefix("encoder.")?.strip_suffix(".weight")?;
        let a_name = format!("clean_mode.{inner}.A");
        if self.by_name.contains_key(&a_name) {
            Some(inner)
        } else {
            None
        }
    }

    fn has_clean_mode_override(&self, name: &str) -> bool {
        if self.clean_mode_scale.is_none() {
            return false;
        }
        name == "decoder.prediction.dec_rnn.lstm.bias_ih_l0"
            || name == "joint.pred.bias"
            || self.clean_mode_encoder_module(name).is_some()
    }

    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let mut values = self.require(name)?.to_f32(self)?;
        let Some(scale) = self.clean_mode_scale else {
            return Ok(values);
        };
        if name == "decoder.prediction.dec_rnn.lstm.bias_ih_l0" {
            let weight = self
                .require("decoder.prediction.dec_rnn.lstm.weight_ih_l0")?
                .to_f32(self)?;
            let dec_in = self.require("clean_mode.decoder_input")?.to_f32(self)?;
            let cols = dec_in.len();
            if weight.len() != values.len() * cols {
                bail!(
                    "clean_mode decoder fold shape mismatch: weight {} vs {}x{}",
                    weight.len(),
                    values.len(),
                    cols
                );
            }
            for (row, bias) in values.iter_mut().enumerate() {
                let w = &weight[row * cols..(row + 1) * cols];
                let mut acc = 0.0f32;
                for (wi, xi) in w.iter().zip(&dec_in) {
                    acc += wi * xi;
                }
                *bias += acc;
            }
            return Ok(values);
        }
        if name == "joint.pred.bias" {
            let joint_h = self.require("clean_mode.joint_hidden")?.to_f32(self)?;
            if joint_h.len() != values.len() {
                bail!(
                    "clean_mode joint_hidden len {} != joint.pred.bias len {}",
                    joint_h.len(),
                    values.len()
                );
            }
            for (b, d) in values.iter_mut().zip(&joint_h) {
                *b += *d;
            }
            return Ok(values);
        }
        if let Some(inner) = self.clean_mode_encoder_module(name) {
            let a_entry = self.require(&format!("clean_mode.{inner}.A"))?;
            let b_entry = self.require(&format!("clean_mode.{inner}.B"))?;
            if a_entry.shape.len() != 2 || b_entry.shape.len() != 2 {
                bail!(
                    "clean_mode.{inner} expects 2D A/B, got {:?} and {:?}",
                    a_entry.shape,
                    b_entry.shape
                );
            }
            let rank = a_entry.shape[0];
            let in_dim = a_entry.shape[1];
            let out_dim = b_entry.shape[0];
            if b_entry.shape[1] != rank || out_dim * in_dim != values.len() {
                bail!(
                    "clean_mode.{inner} shape mismatch: B={:?} A={:?} base_len={}",
                    b_entry.shape,
                    a_entry.shape,
                    values.len()
                );
            }
            let a = a_entry.to_f32(self)?;
            let b = b_entry.to_f32(self)?;
            fold_lora_in_place(&mut values, &b, &a, out_dim, in_dim, rank, scale);
        }
        Ok(values)
    }

    /// Zero-copy little-endian f32 bytes straight out of the mapped checkpoint when
    /// the tensor is contiguous F32 (and unaffected by `clean_mode`).
    pub fn f32_bytes(&self, name: &str) -> Result<std::borrow::Cow<'_, [u8]>> {
        if self.has_clean_mode_override(name) {
            let values = self.f32(name)?;
            return Ok(std::borrow::Cow::Owned(
                bytemuck::cast_slice::<f32, u8>(&values).to_vec(),
            ));
        }
        self.require(name)?.to_f32_bytes(self)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|tensor| tensor.name.as_str())
    }
}

/// Computes `W += scale * (B @ A)` in place (`W: [out, in]`, `B: [out, rank]`, `A: [rank, in]`).
pub(crate) fn fold_lora_in_place(
    weight: &mut [f32],
    b: &[f32],
    a: &[f32],
    out_dim: usize,
    in_dim: usize,
    rank: usize,
    scale: f32,
) {
    use rayon::prelude::*;
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    debug_assert_eq!(b.len(), out_dim * rank);
    debug_assert_eq!(a.len(), rank * in_dim);
    weight
        .par_chunks_mut(in_dim)
        .enumerate()
        .for_each(|(row, w_row)| {
            let b_row = &b[row * rank..(row + 1) * rank];
            for k in 0..rank {
                let s = scale * b_row[k];
                let a_row = &a[k * in_dim..(k + 1) * in_dim];
                for (w, &ak) in w_row.iter_mut().zip(a_row) {
                    *w += s * ak;
                }
            }
        });
}

/// Reads a `model_weights.ckpt` (zip) that has already been loaded into memory.
/// Wall-clock checkpoints for `[load]`, printed when `FLUID_TRACE` is set.
fn trace(label: &str, start: std::time::Instant) {
    if std::env::var_os("FLUID_TRACE").is_some() {
        eprintln!("[load] {label:<22} {:.3}s", start.elapsed().as_secs_f64());
    }
}

pub fn read_checkpoint(map: std::sync::Arc<memmap2::Mmap>, ckpt_at: usize) -> Result<Store> {
    let bytes = &map[ckpt_at..];
    let mut phase = std::time::Instant::now();
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).context("open torch zip")?;
    let prefix = detect_prefix(&mut archive)?;

    let mut pickle = Vec::new();
    {
        let mut entry = archive
            .by_name(&format!("{prefix}data.pkl"))
            .context("checkpoint has no data.pkl")?;
        entry.read_to_end(&mut pickle)?;
    }
    let little_endian = {
        let name = format!("{prefix}byteorder");
        match archive.by_name(&name) {
            Ok(mut entry) => {
                let mut text = String::new();
                entry.read_to_string(&mut text)?;
                !text.trim().eq_ignore_ascii_case("big")
            }
            Err(_) => true,
        }
    };

    trace("pickle read", phase);
    phase = std::time::Instant::now();
    let parsed = pickle::parse(&pickle).context("parse checkpoint pickle")?;
    trace("pickle parse", phase);
    phase = std::time::Instant::now();
    let items = match parsed {
        pickle::Value::Dict(items) => items,
        other => bail!("checkpoint pickle is {}, not a dict", other.kind()),
    };

    // Storages are referenced by key; read each one only after the pickle says
    // which dtype it holds, so bytes are never reinterpreted.
    let mut storages: Vec<Storage> = Vec::new();
    let mut storage_index: HashMap<String, usize> = HashMap::new();
    let mut tensors = Vec::new();
    let mut by_name = HashMap::new();

    for (key, value) in items {
        let pickle::Value::Str(name) = key else {
            continue;
        };
        let Some(rebuilt) = value.as_rebuild() else {
            continue;
        };
        let (dtype, storage_key, numel) = match rebuilt.args.first() {
            Some(pickle::Value::PersId(parts)) => {
                let dtype = match parts.get(1) {
                    Some(pickle::Value::Global(_, class)) => DType::from_storage_name(class)
                        .with_context(|| format!("unsupported storage class {class}"))?,
                    other => bail!(
                        "storage id has no class: {}",
                        other.map_or("none", |value| value.kind())
                    ),
                };
                let storage_key = match parts.get(2) {
                    Some(pickle::Value::Str(key)) => key.clone(),
                    other => bail!(
                        "storage id has no key: {}",
                        other.map_or("none", |value| value.kind())
                    ),
                };
                let numel = match parts.get(4) {
                    Some(pickle::Value::Int(count)) => *count as usize,
                    _ => 0,
                };
                (dtype, storage_key, numel)
            }
            other => bail!(
                "tensor {name} has no storage id: {}",
                other.map_or("none", |v| v.kind())
            ),
        };

        let index = match storage_index.get(&storage_key) {
            Some(&index) => index,
            None => {
                let entry_name = format!("{prefix}data/{storage_key}");
                // Torch writes storages uncompressed, so their bytes are already
                // in the archive: copy them straight out instead of streaming
                // through the zip reader, which touches every byte individually.
                let (data_start, data_len, stored) = {
                    let entry = archive
                        .by_name(&entry_name)
                        .with_context(|| format!("checkpoint has no {entry_name}"))?;
                    (
                        entry.data_start() as usize,
                        entry.size() as usize,
                        entry.compression() == zip::CompressionMethod::Stored,
                    )
                };
                let mut owned = Vec::new();
                if !stored {
                    // Defensive path: an entry stored with compression has to be
                    // inflated before it can be addressed.
                    let mut entry = archive
                        .by_name(&entry_name)
                        .with_context(|| format!("checkpoint has no {entry_name}"))?;
                    entry.read_to_end(&mut owned)?;
                }
                let bytes = if stored {
                    &map[ckpt_at + data_start..ckpt_at + data_start + data_len]
                } else {
                    &owned[..]
                };
                if bytes.len() < numel * dtype.size() {
                    bail!(
                        "storage {storage_key} holds {} bytes, {} needs {}",
                        bytes.len(),
                        name,
                        numel * dtype.size()
                    );
                }
                storages.push(Storage {
                    dtype,
                    map: std::sync::Arc::clone(&map),
                    start: ckpt_at + data_start,
                    byte_len: data_len,
                    little_endian,
                });
                storage_index.insert(storage_key.clone(), storages.len() - 1);
                storages.len() - 1
            }
        };

        let shape = rebuilt.meta.shape();
        let stride = rebuilt.meta.stride();
        let offset = rebuilt.meta.offset.unwrap_or(0);
        by_name.insert(name.clone(), tensors.len());
        tensors.push(TensorEntry {
            name,
            dtype,
            shape,
            stride,
            offset,
            storage_key,
            storage: index,
        });
    }

    trace("storage read", phase);
    // `Store` keeps one canonical copy of the storages.
    Ok(Store {
        tensors,
        storages,
        by_name,
        clean_mode_scale: None,
    })
}

fn detect_prefix(archive: &mut zip::ZipArchive<Cursor<&[u8]>>) -> Result<String> {
    for index in 0..archive.len() {
        let entry = archive.by_index(index)?;
        let name = entry.name().to_string();
        if let Some(rest) = name.strip_suffix("data.pkl") {
            return Ok(rest.to_string());
        }
    }
    bail!("checkpoint zip has no data.pkl")
}

/// Small pickle reader covering what `torch.save` emits for state dicts.
mod pickle {
    use anyhow::{Result, bail};

    /// Pickled objects. Only the shapes a state dict produces are ever read, so
    /// the remaining payload fields exist to keep the stack faithful.
    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    pub enum Value {
        None,
        Bool(bool),
        Int(i64),
        Float(f64),
        Str(String),
        Bytes(Vec<u8>),
        Tuple(Vec<Value>),
        List(Vec<Value>),
        Dict(Vec<(Value, Value)>),
        Global(String, String),
        Reduce(Box<Value>, Vec<Value>),
        PersId(Vec<Value>),
    }

    impl Value {
        pub fn kind(&self) -> &'static str {
            match self {
                Value::None => "none",
                Value::Bool(_) => "bool",
                Value::Int(_) => "int",
                Value::Float(_) => "float",
                Value::Str(_) => "str",
                Value::Bytes(_) => "bytes",
                Value::Tuple(_) => "tuple",
                Value::List(_) => "list",
                Value::Dict(_) => "dict",
                Value::Global(..) => "global",
                Value::Reduce(..) => "reduce",
                Value::PersId(_) => "persistent id",
            }
        }

        pub fn as_rebuild(&self) -> Option<Rebuilt> {
            let Value::Reduce(callable, args) = self else {
                return None;
            };
            let Value::Global(module, name) = callable.as_ref() else {
                return None;
            };
            if !(module == "torch._utils"
                && (name == "_rebuild_tensor_v2" || name == "_rebuild_tensor"))
                && !(module == "torch._utils" && name == "_rebuild_parameter")
            {
                return None;
            }
            let meta = if module == "torch._utils" && name == "_rebuild_parameter" {
                match args.first() {
                    Some(Value::Reduce(_, inner)) => Meta::from_args(inner)?,
                    _ => return None,
                }
            } else {
                Meta::from_args(args)?
            };
            Some(Rebuilt {
                args: args.to_vec(),
                meta,
            })
        }
    }

    pub struct Rebuilt {
        pub args: Vec<Value>,
        pub meta: Meta,
    }

    pub struct Meta {
        pub offset: Option<usize>,
        pub size: Vec<i64>,
        pub stride: Vec<i64>,
    }

    impl Meta {
        fn from_args(args: &[Value]) -> Option<Self> {
            let offset = match args.get(1) {
                Some(Value::Int(value)) => Some(*value as usize),
                _ => None,
            };
            let size = match args.get(2) {
                Some(Value::Tuple(items)) => items
                    .iter()
                    .map(|item| match item {
                        Value::Int(value) => Some(*value),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()?,
                _ => return None,
            };
            let stride = match args.get(3) {
                Some(Value::Tuple(items)) => items
                    .iter()
                    .map(|item| match item {
                        Value::Int(value) => Some(*value),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()?,
                _ => Vec::new(),
            };
            Some(Self {
                offset,
                size,
                stride,
            })
        }

        pub fn shape(&self) -> Vec<usize> {
            self.size
                .iter()
                .map(|&value| value.max(0) as usize)
                .collect()
        }

        pub fn stride(&self) -> Vec<usize> {
            self.stride
                .iter()
                .map(|&value| value.max(0) as usize)
                .collect()
        }
    }

    pub fn parse(bytes: &[u8]) -> Result<Value> {
        let mut reader = Reader::new(bytes);
        let mut stack: Vec<Value> = Vec::new();
        let mut memo: Vec<Value> = Vec::new();
        let mut marks: Vec<usize> = Vec::new();

        loop {
            let opcode = reader.byte()?;
            match opcode {
                b'\x80' => {
                    reader.byte()?;
                }
                b'\x95' => {
                    reader.take(8)?;
                }
                b'.' => break,
                b'c' => {
                    let module = reader.line()?;
                    let name = reader.line()?;
                    stack.push(Value::Global(module, name));
                }
                b'\x93' => {
                    let name = pop_str(&mut stack)?;
                    let module = pop_str(&mut stack)?;
                    stack.push(Value::Global(module, name));
                }
                b')' => stack.push(Value::Tuple(Vec::new())),
                b'\x85' => {
                    let item = pop(&mut stack)?;
                    stack.push(Value::Tuple(vec![item]));
                }
                b'\x86' => {
                    let b = pop(&mut stack)?;
                    let a = pop(&mut stack)?;
                    stack.push(Value::Tuple(vec![a, b]));
                }
                b'\x87' => {
                    let c = pop(&mut stack)?;
                    let b = pop(&mut stack)?;
                    let a = pop(&mut stack)?;
                    stack.push(Value::Tuple(vec![a, b, c]));
                }
                b'(' => marks.push(stack.len()),
                b't' => {
                    let start = marks.pop().unwrap_or(0);
                    let items = stack.split_off(start);
                    stack.push(Value::Tuple(items));
                }
                b']' => stack.push(Value::List(Vec::new())),
                b'}' => stack.push(Value::Dict(Vec::new())),
                b'u' => {
                    let start = marks.pop().unwrap_or(0);
                    let items = stack.split_off(start);
                    let mut dict = take_dict(&mut stack, "SETITEMS")?;
                    for pair in items.chunks(2) {
                        if let [key, value] = pair {
                            dict.push((key.clone(), value.clone()));
                        }
                    }
                    stack.push(Value::Dict(dict));
                }
                b's' => {
                    let value = pop(&mut stack)?;
                    let key = pop(&mut stack)?;
                    let mut dict = take_dict(&mut stack, "SETITEM")?;
                    dict.push((key, value));
                    stack.push(Value::Dict(dict));
                }
                b'e' | b'a' => {
                    let appended: Vec<Value> = if opcode == b'e' {
                        let start = marks.pop().unwrap_or(0);
                        stack.split_off(start)
                    } else {
                        vec![pop(&mut stack)?]
                    };
                    let Some(Value::List(list)) = stack.pop() else {
                        bail!("APPEND without a list");
                    };
                    let mut list = list;
                    list.extend(appended);
                    stack.push(Value::List(list));
                }
                b'\x94' => {
                    // MEMOIZE stores TOS without popping it.
                    let top = stack
                        .last()
                        .ok_or_else(|| anyhow::anyhow!("MEMOIZE on an empty stack"))?;
                    memo.push(top.clone());
                }
                b'q' | b'r' | b'p' => {
                    // BINPUT / LONG_BINPUT / PUT memoize TOS and keep it.
                    let index = match opcode {
                        b'q' => reader.byte()? as usize,
                        b'r' => reader.u32()? as usize,
                        _ => reader.line()?.parse()?,
                    };
                    let top = stack
                        .last()
                        .ok_or_else(|| anyhow::anyhow!("BINPUT on an empty stack"))?;
                    let value = top.clone();
                    if memo.len() <= index {
                        memo.resize(index + 1, Value::None);
                    }
                    memo[index] = value;
                }
                b'h' | b'j' | b'g' => {
                    let index = match opcode {
                        b'h' => reader.byte()? as usize,
                        b'j' => reader.u32()? as usize,
                        _ => reader.line()?.parse()?,
                    };
                    let value = memo.get(index).unwrap_or(&Value::None);
                    stack.push(value.clone());
                }
                b'J' => stack.push(Value::Int(i64::from(reader.i32()?))),
                b'K' => stack.push(Value::Int(i64::from(reader.byte()?))),
                b'M' => stack.push(Value::Int(i64::from(reader.u16()?))),
                b'\x8a' => {
                    let len = reader.byte()? as usize;
                    stack.push(Value::Int(reader.long(len)?));
                }
                b'\x8b' => {
                    let len = reader.u32()? as usize;
                    stack.push(Value::Int(reader.long(len)?));
                }
                b'X' => {
                    let len = reader.u32()? as usize;
                    stack.push(Value::Str(reader.utf8(len)?));
                }
                b'\x8c' => {
                    let len = reader.byte()? as usize;
                    stack.push(Value::Str(reader.utf8(len)?));
                }
                b'T' => {
                    let len = reader.u32()? as usize;
                    stack.push(Value::Str(reader.utf8(len)?));
                }
                b'V' => {
                    let text = reader.line()?;
                    stack.push(Value::Str(text));
                }
                b'B' => {
                    let len = reader.u32()? as usize;
                    stack.push(Value::Bytes(reader.take(len)?.to_vec()));
                }
                b'G' => stack.push(Value::Float(reader.f64()?)),
                b'\x88' => stack.push(Value::Bool(true)),
                b'\x89' => stack.push(Value::Bool(false)),
                b'N' => stack.push(Value::None),
                b'Q' => {
                    let pid = pop(&mut stack)?;
                    match pid {
                        Value::Tuple(items) => stack.push(Value::PersId(items)),
                        other => stack.push(Value::PersId(vec![other])),
                    }
                }
                b'R' => {
                    let args = pop(&mut stack)?;
                    let callable = pop(&mut stack)?;
                    let args = match args {
                        Value::Tuple(items) => items,
                        other => vec![other],
                    };
                    stack.push(Value::Reduce(Box::new(callable), args));
                }
                b'\x81' => {
                    let args = pop(&mut stack)?;
                    let callable = pop(&mut stack)?;
                    let args = match args {
                        Value::Tuple(items) => items,
                        other => vec![other],
                    };
                    stack.push(Value::Reduce(Box::new(callable), args));
                }
                b'b' => {
                    let _state = pop(&mut stack)?;
                    // The object below the state keeps its identity; the state is
                    // only useful for `nn.Module`s, which a state dict never holds.
                }
                unknown => bail!("unsupported pickle opcode 0x{unknown:02x}"),
            }
        }

        Ok(stack.pop().unwrap_or(Value::None))
    }

    fn pop(stack: &mut Vec<Value>) -> Result<Value> {
        stack
            .pop()
            .ok_or_else(|| anyhow::anyhow!("pickle stack underflow"))
    }

    fn pop_str(stack: &mut Vec<Value>) -> Result<String> {
        match pop(stack)? {
            Value::Str(text) => Ok(text),
            other => bail!("expected a string, found {}", other.kind()),
        }
    }

    /// Torch saves a state dict as `collections.OrderedDict()`, so the object
    /// SETITEMS fills is a REDUCE result rather than a literal `}` dict.
    fn take_dict(stack: &mut Vec<Value>, what: &str) -> Result<Vec<(Value, Value)>> {
        match stack.pop() {
            Some(Value::Dict(items)) => Ok(items),
            Some(Value::Reduce(callable, _)) if is_ordered_dict(&callable) => Ok(Vec::new()),
            other => bail!(
                "{what} without a dict (found {})",
                other.map_or("none", |value| value.kind())
            ),
        }
    }

    fn is_ordered_dict(callable: &Value) -> bool {
        matches!(
            callable,
            Value::Global(module, name) if module == "collections" && name == "OrderedDict"
        )
    }

    struct Reader<'a> {
        bytes: &'a [u8],
        at: usize,
    }

    impl<'a> Reader<'a> {
        fn new(bytes: &'a [u8]) -> Self {
            Self { bytes, at: 0 }
        }

        fn take(&mut self, len: usize) -> Result<&'a [u8]> {
            let end = self.at + len;
            if end > self.bytes.len() {
                bail!("pickle read past the end");
            }
            let slice = &self.bytes[self.at..end];
            self.at = end;
            Ok(slice)
        }

        fn byte(&mut self) -> Result<u8> {
            Ok(self.take(1)?[0])
        }

        fn u16(&mut self) -> Result<u16> {
            let slice = self.take(2)?;
            Ok(u16::from_le_bytes([slice[0], slice[1]]))
        }

        fn u32(&mut self) -> Result<u32> {
            let slice = self.take(4)?;
            Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
        }

        fn i32(&mut self) -> Result<i32> {
            Ok(self.u32()? as i32)
        }

        fn f64(&mut self) -> Result<f64> {
            let slice = self.take(8)?;
            let mut word = [0u8; 8];
            word.copy_from_slice(slice);
            Ok(f64::from_be_bytes(word))
        }

        fn utf8(&mut self, len: usize) -> Result<String> {
            let slice = self.take(len)?;
            Ok(String::from_utf8_lossy(slice).into_owned())
        }

        fn line(&mut self) -> Result<String> {
            let mut end = self.at;
            while end < self.bytes.len() && self.bytes[end] != b'\n' {
                end += 1;
            }
            let slice = &self.bytes[self.at..end];
            self.at = (end + 1).min(self.bytes.len());
            Ok(String::from_utf8_lossy(slice).into_owned())
        }

        fn long(&mut self, len: usize) -> Result<i64> {
            if len > 8 {
                bail!("pickle long with {len} bytes is out of range");
            }
            let slice = self.take(len)?;
            let mut value: i64 = 0;
            for (shift, &byte) in slice.iter().enumerate() {
                value |= i64::from(byte) << (8 * shift);
            }
            if len > 0 && slice[len - 1] & 0x80 != 0 {
                for shift in len..8 {
                    value |= 0xffi64 << (8 * shift);
                }
            }
            Ok(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_lora_adds_scaled_b_times_a() {
        // W: [2, 3], B: [2, 2], A: [2, 3], scale = 2.0
        let mut w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let a = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        fold_lora_in_place(&mut w, &b, &a, 2, 3, 2, 2.0);
        // row 0: [1*0.5 + 2*2.0, 1*1.0 + 2*2.5, 1*1.5 + 2*3.0] * 2 = [9, 12, 15] -> + [1, 2, 3] = [10, 14, 18]
        // row 1: [3*0.5 + 4*2.0, 3*1.0 + 4*2.5, 3*1.5 + 4*3.0] * 2 = [19, 26, 33] -> + [4, 5, 6] = [23, 31, 39]
        assert_eq!(w, vec![10.0, 14.0, 18.0, 23.0, 31.0, 39.0]);
    }
}
