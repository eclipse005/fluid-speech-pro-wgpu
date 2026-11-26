//! Fluid Speech Pro (NVIDIA Parakeet TDT 0.63B FastConformer) inference in pure Rust,
//! with a CPU backend and a wgpu backend.
//!
//! The supported surface is [`api`]: load a `.nemo` archive once, pick a [`Backend`],
//! then transcribe samples or files and read back text, tokens and four tiers of
//! timestamps.
//!
//! ```no_run
//! use fluid_speech_pro::api::{Backend, Transcriber};
//!
//! let mut asr = Transcriber::load("fluid-speech-pro.nemo".as_ref(), Backend::Gpu)?;
//! println!("{}", asr.transcribe_file("audio.wav".as_ref())?.text);
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! Everything below `api` is implementation detail. Those modules become public only
//! with the `internals` feature, which is how this repository's verification and
//! benchmark binaries reach them; a crate that depends on this one sees only `api`.
//!
//! [`Backend`]: api::Backend

#![cfg_attr(
    not(feature = "internals"),
    allow(dead_code),
    // Helpers that only the verification/benchmark binaries call (`forward_upto`,
    // `layer_count`, the `Gpu` stage's read-back accessors) have no caller in a plain
    // build. Suppressing the lint library-wide is the price of shipping one crate with
    // optional tooling; genuine dead code is still caught by the `internals` build,
    // which enables every path.
)]

pub mod api;

/// Declares one implementation module: private by default, public for the tool binaries.
macro_rules! internal {
    ($($name:ident),* $(,)?) => {
        $(
            #[cfg(feature = "internals")]
            pub mod $name;
            #[cfg(not(feature = "internals"))]
            mod $name;
        )*
    };
}

internal!(
    beam,
    confidence,
    conformer,
    fusion,
    fv_insert,
    fv_insert_gpu,
    gemm,
    golden,
    gpu,
    gpu_attn,
    gpu_encoder,
    mel,
    nemo,
    pre_encode,
    pre_encode_gpu,
    tdt,
    tdt_gpu,
    text,
    torch,
);

/// Pure-FMA throughput probe; a benchmark tool, so it is not part of the library.
#[cfg(feature = "internals")]
pub mod flops_probe;
