//! wgpu backend, starting with the GEMM the encoder spends ~90% of its time in.
//!
//! Weights upload as plain f32 row-major `[n][k]` — the layout the checkpoint
//! already has, so nothing is repacked for the GPU either. The arithmetic stays
//! f32: on this adapter (Pascal) f16 math is 1/64 of f32 throughput, which is why
//! NeMo's own autocast measured slower than f32.

use anyhow::{Context, Result, bail};

/// One `Dims` uniform, 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

/// One `LayerNormParams` uniform, 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerNormParams {
    steps: u32,
    width: u32,
    eps: f32,
    _pad: f32,
}

const SHADER: &str = include_str!("gemm.wgsl");
const LAYER_NORM_SHADER: &str = include_str!("layer_norm.wgsl");
const SILU_SHADER: &str = include_str!("silu.wgsl");
const GLU_SHADER: &str = include_str!("glu.wgsl");
const DW_CONV_SHADER: &str = include_str!("dw_conv_bn_silu.wgsl");

/// `GluParams` / `ConvParams` are both `(a, b, c, d)` uniform blocks.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FourU32 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}
/// Must match the shader's `BM`/`BN`.
/// GEMM tile shape. Overridable so it can be calibrated per card instead of being
/// baked in: register file size, shared-memory budget and the LDS:FMA ratio differ
/// enough between architectures that one shape cannot be right everywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
}

impl Tile {
    /// Parses `BMxBNxBK`, e.g. `128x128x16`.
    fn from_env() -> Option<Self> {
        let raw = std::env::var("FSP_GEMM_TILE").ok()?;
        let parts: Vec<u32> = raw.split('x').filter_map(|p| p.parse().ok()).collect();
        if parts.len() != 3 || parts.contains(&0) {
            return None;
        }
        Some(Tile {
            bm: parts[0],
            bn: parts[1],
            bk: parts[2],
        })
    }
}

/// Default tile, calibrated for this card (see `Tile`).
const DEFAULT_TILE: Tile = Tile { bm: 64, bn: 64, bk: 16 };

/// Pure-f32 `v4` 8x8 (`BM=128, BN=128, BK=16`) GEMM tile dimensions.
const V4_BM: u32 = 128;
const V4_BN: u32 = 128;
/// Minimum 128x128 workgroups required before switching from 64x64 to v4 128x128.
const V4_MIN_WORKGROUPS: u32 = 16;

/// Generates the pure-f32 `v4` 8x8 (`BM=128, BN=128, BK=16`) SGEMM WGSL shader.
///
/// Combines `qwen3-asr-wgpu`'s k-major `vec4<f32>` shared tiles, bank-conflict-free B
/// permutation, 4x k-step unrolling, and register software pipelining with **coalesced
/// `vec4<f32>` (`LDG.128`) global loads** — 4 coalesced 128-bit loads per thread instead
/// of 16 strided scalar loads, eliminating power-of-2 L1/L2 set conflicts on `k=1024/4096`.
/// Arbitrary `m`/`n` are handled via clamped row offsets (zero branches inside the `k`-loop).
/// Shares the exact `BindGroupLayout` of `gemm.wgsl` (`dims, a, w, c`), and maps
/// `m0 = wid.y * 128`, `n0 = wid.x * 128` (caller dispatches `(n_tiles, m_tiles, 1)`).
fn gemm_v4_wgsl() -> String {
    let mut s = String::with_capacity(28_000);
    s.push_str(
        "struct Dims { m: u32, n: u32, k: u32, _pad: u32 };\n\
         @group(0) @binding(0) var<uniform> dims: Dims;\n\
         @group(0) @binding(1) var<storage, read> a: array<vec4<f32>>;\n\
         @group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;\n\
         @group(0) @binding(3) var<storage, read_write> c: array<vec4<f32>>;\n\
         const BM: u32 = 128u; const BN: u32 = 128u; const BK: u32 = 16u;\n\
         const AQ: u32 = 32u; const BQ: u32 = 32u;\n\
         var<workgroup> A4: array<vec4<f32>, 512>;\n\
         var<workgroup> B4: array<vec4<f32>, 512>;\n\
         @compute @workgroup_size(16, 16, 1)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n\
         \x20   let m = dims.m; let n = dims.n; let k = dims.k;\n\
         \x20   if (m == 0u || n == 0u || k == 0u) { return; }\n\
         \x20   let k4 = k >> 2u;\n\
         \x20   let tx = lid.x; let ty = lid.y;\n\
         \x20   let m0 = wid.y * BM; let n0 = wid.x * BN;\n\
         \x20   let tid = ty * 16u + tx;\n\
         \x20   let gk4 = tid & 3u;\n\
         \x20   let gr0 = tid >> 2u;\n\
         \x20   let lane = gr0 & 3u;\n\
         \x20   let arg0 = gr0 >> 2u;\n\
         \x20   let bpos0 = (gr0 >> 3u) + ((gr0 & 4u) << 1u);\n\
         \x20   let qoff = (gk4 << 2u) * AQ;\n",
    );
    for i in 0..8 {
        for j in 0..8 {
            s.push_str(&format!("    var c{i}{j} = 0.0;\n"));
        }
    }
    let store_vec4 = |buf: &str, pos: &str, v: &str| {
        format!(
            "        {buf}[qoff + 0u * AQ + {pos}][lane] = {v}.x;\n\
             \x20       {buf}[qoff + 1u * AQ + {pos}][lane] = {v}.y;\n\
             \x20       {buf}[qoff + 2u * AQ + {pos}][lane] = {v}.z;\n\
             \x20       {buf}[qoff + 3u * AQ + {pos}][lane] = {v}.w;\n"
        )
    };
    s.push_str(
        "    {\n\
         \x20       let a0 = a[min(m0 + gr0, m - 1u) * k4 + gk4];\n\
         \x20       let a1 = a[min(m0 + gr0 + 64u, m - 1u) * k4 + gk4];\n\
         \x20       let b0 = w[min(n0 + gr0, n - 1u) * k4 + gk4];\n\
         \x20       let b1 = w[min(n0 + gr0 + 64u, n - 1u) * k4 + gk4];\n",
    );
    s.push_str(&store_vec4("A4", "arg0", "a0"));
    s.push_str(&store_vec4("A4", "arg0 + 16u", "a1"));
    s.push_str(&store_vec4("B4", "bpos0", "b0"));
    s.push_str(&store_vec4("B4", "bpos0 + 16u", "b1"));
    s.push_str(
        "    }\n\
         \x20   workgroupBarrier();\n\
         \x20   var k0: u32 = 0u;\n\
         \x20   loop {\n\
         \x20       if (k0 >= k) { break; }\n\
         \x20       let kn = k0 + BK;\n\
         \x20       var pfa0 = vec4<f32>(0.0); var pfa1 = vec4<f32>(0.0);\n\
         \x20       var pfb0 = vec4<f32>(0.0); var pfb1 = vec4<f32>(0.0);\n\
         \x20       if (kn < k) {\n\
         \x20           let kn4 = (kn >> 2u) + gk4;\n\
         \x20           pfa0 = a[min(m0 + gr0, m - 1u) * k4 + kn4];\n\
         \x20           pfa1 = a[min(m0 + gr0 + 64u, m - 1u) * k4 + kn4];\n\
         \x20           pfb0 = w[min(n0 + gr0, n - 1u) * k4 + kn4];\n\
         \x20           pfb1 = w[min(n0 + gr0 + 64u, n - 1u) * k4 + kn4];\n\
         \x20       }\n\
         \x20       var q0: u32 = 0u;\n\
         \x20       loop {\n\
         \x20           if (q0 >= BK) { break; }\n",
    );
    const COMP: [&str; 4] = ["x", "y", "z", "w"];
    for u in 0..4 {
        s.push_str(&format!(
            "            let av_{u} = A4[(q0 + {u}u) * AQ + ty * 2u];\n\
             \x20           let aw_{u} = A4[(q0 + {u}u) * AQ + ty * 2u + 1u];\n\
             \x20           let bv_{u} = B4[(q0 + {u}u) * BQ + (tx & 7u) + 16u * (tx >> 3u)];\n\
             \x20           let bw_{u} = B4[(q0 + {u}u) * BQ + (tx & 7u) + 8u + 16u * (tx >> 3u)];\n"
        ));
        for i in 0..8 {
            let ae = if i < 4 {
                format!("av_{u}.{}", COMP[i])
            } else {
                format!("aw_{u}.{}", COMP[i - 4])
            };
            for j in 0..8 {
                let be = if j < 4 {
                    format!("bv_{u}.{}", COMP[j])
                } else {
                    format!("bw_{u}.{}", COMP[j - 4])
                };
                s.push_str(&format!("            c{i}{j} = c{i}{j} + {ae} * {be};\n"));
            }
        }
    }
    s.push_str(
        "            q0 = q0 + 4u;\n\
         \x20       }\n\
         \x20       workgroupBarrier();\n\
         \x20       if (kn < k) {\n",
    );
    s.push_str(&store_vec4("A4", "arg0", "pfa0"));
    s.push_str(&store_vec4("A4", "arg0 + 16u", "pfa1"));
    s.push_str(&store_vec4("B4", "bpos0", "pfb0"));
    s.push_str(&store_vec4("B4", "bpos0 + 16u", "pfb1"));
    s.push_str(
        "        }\n\
         \x20       workgroupBarrier();\n\
         \x20       k0 = kn;\n\
         \x20   }\n\
         \x20   let stride_c4 = n >> 2u;\n\
         \x20   let col4_0 = (n0 >> 2u) + tx * 2u;\n\
         \x20   let col4_1 = col4_0 + 1u;\n\
         \x20   let in_c0 = col4_0 < stride_c4;\n\
         \x20   let in_c1 = col4_1 < stride_c4;\n",
    );
    for i in 0..8 {
        s.push_str(&format!(
            "    let row{i} = m0 + ty * 8u + {i}u;\n\
             \x20   if (row{i} < m) {{\n\
             \x20       let off{i} = row{i} * stride_c4;\n\
             \x20       if (in_c0) {{ c[off{i} + col4_0] = vec4<f32>(c{i}0, c{i}1, c{i}2, c{i}3); }}\n\
             \x20       if (in_c1) {{ c[off{i} + col4_1] = vec4<f32>(c{i}4, c{i}5, c{i}6, c{i}7); }}\n\
             \x20   }}\n"
        ));
    }
    s.push_str("}\n");
    s
}

/// Per-adapter, per-driver Vulkan/DX12 shader pipeline cache path (driver shader binaries
/// only — never model weights).
fn pipeline_cache_path(info: &wgpu::AdapterInfo) -> Option<std::path::PathBuf> {
    let root = std::env::var_os("LOCALAPPDATA")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("XDG_CACHE_HOME").map(std::path::PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
        .or_else(|| Some(std::env::temp_dir()))?;
    let key = format!(
        "{:04x}-{:04x}-{:?}-{}",
        info.vendor,
        info.device,
        info.backend,
        info.driver_info.replace(['\\', '/', ':', ' '], "_")
    );
    Some(root.join("fluid-speech-pro").join(format!("{key}.pipeline_cache")))
}

/// Which adapter to open: `auto`, `cpu`, `vulkan[:i]`, `dx12[:i]`, `#n`, or a name substring.
///
/// Same shape as the selector the sibling `qwen3-asr-wgpu` / `ctc-forced-aligner-wgpu`
/// crates expose, so a host that embeds several of them passes one vocabulary of specs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DeviceSelector {
    #[default]
    Auto,
    Runtime {
        api: wgpu::Backend,
        index: usize,
    },
    Cpu,
    Index(usize),
    Name(String),
}

/// Runtime spellings [`DeviceSelector::parse`] accepts, with their wgpu backend.
const RUNTIMES: &[(&str, wgpu::Backend)] = &[
    ("vulkan", wgpu::Backend::Vulkan),
    ("metal", wgpu::Backend::Metal),
    ("dx12", wgpu::Backend::Dx12),
    ("d3d12", wgpu::Backend::Dx12),
    ("gl", wgpu::Backend::Gl),
];

impl DeviceSelector {
    /// Parses one spec: empty/`auto`, `cpu`, `#n` or `n` (adapter index), a runtime
    /// (`vulkan`, `metal`, `dx12`, `d3d12`, `gl`, optionally `:<index>`), or an adapter
    /// name substring (matched case-insensitively).
    pub fn parse(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        if spec.is_empty() || spec.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if spec.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if spec.eq_ignore_ascii_case("gpu") {
            bail!("`gpu` is a backend, not an adapter name — use Backend::Gpu");
        }
        if let Some(rest) = spec.strip_prefix('#') {
            return Ok(Self::Index(rest.trim().parse().context("device index")?));
        }
        if let Ok(index) = spec.parse::<usize>() {
            return Ok(Self::Index(index));
        }
        let (name, index) = match spec.split_once(':') {
            Some((name, index)) => (name, index.trim().parse().context("runtime device index")?),
            None => (spec, 0usize),
        };
        if let Some((_, api)) = RUNTIMES
            .iter()
            .find(|(runtime, _)| name.eq_ignore_ascii_case(runtime))
        {
            return Ok(Self::Runtime { api: *api, index });
        }
        Ok(Self::Name(spec.to_lowercase()))
    }
}

/// Most preferred API first. On macOS, native Metal beats MoltenVK; on Windows and
/// Linux, Vulkan leads while DX12-only Windows hosts fall through cleanly.
#[cfg(target_os = "macos")]
const API_RANK_ORDER: [wgpu::Backend; 4] = [
    wgpu::Backend::Metal,
    wgpu::Backend::Vulkan,
    wgpu::Backend::Dx12,
    wgpu::Backend::Gl,
];

#[cfg(not(target_os = "macos"))]
const API_RANK_ORDER: [wgpu::Backend; 4] = [
    wgpu::Backend::Vulkan,
    wgpu::Backend::Metal,
    wgpu::Backend::Dx12,
    wgpu::Backend::Gl,
];

/// Index in [`API_RANK_ORDER`], or 4 for a backend not listed there. Lower is better.
fn api_rank(backend: wgpu::Backend) -> u8 {
    API_RANK_ORDER
        .iter()
        .position(|b| *b == backend)
        .map_or(4, |i| i as u8)
}

/// Every native backend a desktop build can expose.
const ALL_GPU_BACKENDS: wgpu::Backends = wgpu::Backends::VULKAN
    .union(wgpu::Backends::METAL)
    .union(wgpu::Backends::DX12)
    .union(wgpu::Backends::GL);

/// Discrete, then integrated, then virtual. A CPU adapter is not a GPU.
fn device_class(kind: wgpu::DeviceType) -> u8 {
    match kind {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 3,
        _ => 4,
    }
}

/// How [`DeviceSelector::Auto`] and [`DeviceSelector::Name`] order candidates: fastest
/// device class first, then the API that talks to it most directly.
fn rank(device_type: wgpu::DeviceType, backend: wgpu::Backend) -> (u8, u8) {
    (device_class(device_type), api_rank(backend))
}

/// The device classes [`DeviceSelector::Auto`] will consider. A CPU adapter (WARP,
/// llvmpipe/lavapipe) renders in software and is never a candidate.
fn is_real_gpu(device_type: wgpu::DeviceType) -> bool {
    matches!(
        device_type,
        wgpu::DeviceType::DiscreteGpu
            | wgpu::DeviceType::IntegratedGpu
            | wgpu::DeviceType::VirtualGpu
    )
}

/// One instance restricted to `backends`.
///
/// A Vulkan-only instance enumerates in ~3 ms; `Instance::default()` loads every
/// runtime's loader first, which is most of a second on Windows.
async fn adapters_for(backends: wgpu::Backends) -> Vec<wgpu::Adapter> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends;
    wgpu::Instance::new(descriptor)
        .enumerate_adapters(backends)
        .await
}

fn list_names(adapters: &[wgpu::Adapter]) -> String {
    adapters
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let info = a.get_info();
            format!(
                "#{i} {} ({:?}, {:?})",
                info.name, info.backend, info.device_type
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn sort_by_rank(adapters: &mut [wgpu::Adapter]) {
    adapters.sort_by_key(|a| {
        let info = a.get_info();
        rank(info.device_type, info.backend)
    });
}

/// Resolves `selector` to an open adapter.
async fn pick_adapter(selector: &DeviceSelector) -> Result<wgpu::Adapter> {
    match selector {
        DeviceSelector::Cpu => bail!("DeviceSelector::Cpu cannot open a GPU adapter"),
        DeviceSelector::Auto => {
            // Lazily walk the backends in rank order and stop at the first one that
            // exposes a discrete GPU: on Windows that is Vulkan's first adapter, so a
            // discrete card costs one enumeration, while Metal-only and DX12-only hosts
            // fall through to the backend they actually have.
            let mut gpus: Vec<wgpu::Adapter> = Vec::new();
            for api in API_RANK_ORDER {
                let found = adapters_for(wgpu::Backends::from(api)).await;
                let discrete = found
                    .iter()
                    .any(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu);
                gpus.extend(
                    found
                        .into_iter()
                        .filter(|a| is_real_gpu(a.get_info().device_type)),
                );
                if discrete {
                    break;
                }
            }
            if gpus.is_empty() {
                bail!("no GPU adapter found (auto)");
            }
            sort_by_rank(&mut gpus);
            Ok(gpus.remove(0))
        }
        DeviceSelector::Runtime { api, index } => {
            let mut found = adapters_for(wgpu::Backends::from(*api)).await;
            sort_by_rank(&mut found);
            if *index >= found.len() {
                bail!(
                    "{api:?} exposes {} adapter(s), index {index} is out of range: {}",
                    found.len(),
                    list_names(&found)
                );
            }
            Ok(found.remove(*index))
        }
        DeviceSelector::Index(index) => {
            let mut found = adapters_for(ALL_GPU_BACKENDS).await;
            if *index >= found.len() {
                bail!(
                    "{index} is out of range: {} adapter(s) visible: {}",
                    found.len(),
                    list_names(&found)
                );
            }
            Ok(found.swap_remove(*index))
        }
        DeviceSelector::Name(needle) => {
            let found = adapters_for(ALL_GPU_BACKENDS).await;
            let visible = list_names(&found);
            let mut hits: Vec<wgpu::Adapter> = found
                .into_iter()
                .filter(|a| a.get_info().name.to_lowercase().contains(needle))
                .collect();
            if hits.is_empty() {
                bail!("no adapter name contains `{needle}` (visible: {visible})");
            }
            // Class before API, so a discrete GPU wins over an integrated one that
            // happens to sit on an earlier-ranked backend.
            sort_by_rank(&mut hits);
            Ok(hits.remove(0))
        }
    }
}

pub struct GpuGemm {
    tile: Tile,
    tile_override: bool,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_cache: Option<wgpu::PipelineCache>,
    pipeline_cache_path: Option<std::path::PathBuf>,
    pipeline: wgpu::ComputePipeline,
    pipeline_v4: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    layer_norm: wgpu::ComputePipeline,
    layer_norm_layout: wgpu::BindGroupLayout,
    silu: wgpu::ComputePipeline,
    silu_layout: wgpu::BindGroupLayout,
    glu: wgpu::ComputePipeline,
    glu_layout: wgpu::BindGroupLayout,
    dw_conv: wgpu::ComputePipeline,
    dw_conv_layout: wgpu::BindGroupLayout,
    adapter: String,
    adapter_description: String,
}

impl GpuGemm {
    /// Opens the adapter [`DeviceSelector::Auto`] picks: the best real GPU, discrete first.
    pub fn new() -> Result<Self> {
        Self::new_on(&DeviceSelector::Auto)
    }

    /// Opens the adapter `selector` names.
    pub fn new_on(selector: &DeviceSelector) -> Result<Self> {
        let adapter = pollster::block_on(pick_adapter(selector))?;
        Self::with_adapter(adapter)
    }

    fn with_adapter(adapter: wgpu::Adapter) -> Result<Self> {
        let info = adapter.get_info();
        let features = adapter.features();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fluid-speech-pro"),
            required_features: features & wgpu::Features::PIPELINE_CACHE,
            required_limits: limits,
            ..Default::default()
        }))
        .context("request_device")?;
        device.set_device_lost_callback(|reason, message| {
            eprintln!("wgpu device lost ({reason:?}): {message}");
        });

        let cache_supported = features.contains(wgpu::Features::PIPELINE_CACHE);
        let (pipeline_cache, pipeline_cache_path) = match pipeline_cache_path(&info) {
            Some(path) if cache_supported => {
                let seed = std::fs::read(&path).ok();
                // SAFETY: opaque driver blob keyed on adapter + driver version with
                // `fallback: true`, so stale or foreign blobs degrade to a clean cache miss.
                let cache = unsafe {
                    device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                        label: Some("fsp_pipeline_cache"),
                        data: seed.as_deref(),
                        fallback: true,
                    })
                };
                (Some(cache), Some(path))
            }
            _ => (None, None),
        };

        let tile_env = Tile::from_env();
        let tile_override = tile_env.is_some();
        let tile = tile_env.unwrap_or(DEFAULT_TILE);
        let gemm_source = SHADER
            .replace("{{BM}}", &format!("{}u", tile.bm))
            .replace("{{BN}}", &format!("{}u", tile.bn))
            .replace("{{BK}}", &format!("{}u", tile.bk));
        let v4_source = gemm_v4_wgsl();

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemm"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemm"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let layer_norm_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layer_norm"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
        });
        let layer_norm_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("layer_norm"),
                bind_group_layouts: &[Some(&layer_norm_layout)],
                immediate_size: 0,
            });
        let silu_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("silu"),
            entries: &[storage_entry(0, false)],
        });
        let silu_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("silu"),
            bind_group_layouts: &[Some(&silu_layout)],
            immediate_size: 0,
        });
        let glu_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("glu"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let glu_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("glu"),
            bind_group_layouts: &[Some(&glu_layout)],
            immediate_size: 0,
        });
        let dw_conv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dw_conv"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
            ],
        });
        let dw_conv_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dw_conv"),
                bind_group_layouts: &[Some(&dw_conv_layout)],
                immediate_size: 0,
            });

        let cache = pipeline_cache.as_ref();
        let compile = |label: &'static str, src: &str, pl: &wgpu::PipelineLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache,
            })
        };

        // Compile the six core pipelines concurrently across threads so driver SPIR-V
        // codegen scales with CPU cores even on a cold cache.
        let (pipeline, pipeline_v4, layer_norm, silu, glu, dw_conv) = std::thread::scope(|s| {
            let t0 = s.spawn(|| compile("gemm", &gemm_source, &pipeline_layout));
            let t1 = s.spawn(|| compile("gemm_v4", &v4_source, &pipeline_layout));
            let t2 = s.spawn(|| compile("layer_norm", LAYER_NORM_SHADER, &layer_norm_pipeline_layout));
            let t3 = s.spawn(|| compile("silu", SILU_SHADER, &silu_pipeline_layout));
            let t4 = s.spawn(|| compile("glu", GLU_SHADER, &glu_pipeline_layout));
            let t5 = s.spawn(|| compile("dw_conv", DW_CONV_SHADER, &dw_conv_pipeline_layout));
            (
                t0.join().expect("gemm"),
                t1.join().expect("gemm_v4"),
                t2.join().expect("layer_norm"),
                t3.join().expect("silu"),
                t4.join().expect("glu"),
                t5.join().expect("dw_conv"),
            )
        });

        Ok(Self {
            tile,
            tile_override,
            device,
            queue,
            pipeline_cache,
            pipeline_cache_path,
            pipeline,
            pipeline_v4,
            layout,
            layer_norm,
            layer_norm_layout,
            silu,
            silu_layout,
            glu,
            glu_layout,
            dw_conv,
            dw_conv_layout,
            adapter: format!("{info:?}"),
            adapter_description: format!(
                "{} ({:?}, {:?})",
                info.name, info.backend, info.device_type
            ),
        })
    }

    /// The tile shape this instance was built with, for callers that size dispatches.
    pub fn tile(&self) -> Tile {
        self.tile
    }

    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    /// One-line adapter summary, e.g. `NVIDIA GeForce GTX 1080 (Vulkan, DiscreteGpu)`.
    pub fn adapter_description(&self) -> &str {
        &self.adapter_description
    }

    /// `C[m][n] = A[m][k] @ W[n][k]^T`.
    pub fn gemm(&self, a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
        self.gemm_reps(a, w, m, n, k, 1)
    }

    /// Same product, dispatched `reps` times into the same buffers so the timed
    /// window measures GPU work rather than buffer setup and upload.
    pub fn gemm_reps(
        &self,
        a: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
        reps: usize,
    ) -> Result<Vec<f32>> {
        self.gemm_reps_mode(a, w, m, n, k, reps, None)
    }

    /// Same product, timing ONLY the GPU compute submission (`reps` dispatches in one
    /// command buffer, excluding host upload and readback) for accurate kernel A/B.
    pub fn gemm_bench_mode(
        &self,
        a: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
        reps: usize,
        force_v4: Option<bool>,
    ) -> Result<(Vec<f32>, std::time::Duration)> {
        if a.len() < m * k || w.len() < n * k {
            bail!("operands are too small for {m}x{n}x{k}");
        }
        let device = &self.device;
        let buffer_a = self.upload("a", a);
        let buffer_w = self.upload("w", w);
        let c_len = ((m * n * 4).div_ceil(16) * 16) as u64;
        let buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c"),
            size: c_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buffer_dims = self.uniform_buffer(
            "dims",
            &Dims {
                m: m as u32,
                n: n as u32,
                k: k as u32,
                _pad: 0,
            },
        );
        let bind_group = self.bind_group(
            "gemm_bench",
            &self.layout,
            &[
                (0, buffer_dims.as_entire_binding()),
                (1, buffer_a.as_entire_binding()),
                (2, buffer_w.as_entire_binding()),
                (3, buffer_c.as_entire_binding()),
            ],
        );
        let (pipeline, gx, gy) = match force_v4 {
            Some(true) => {
                if !k.is_multiple_of(16) || !n.is_multiple_of(4) {
                    bail!("gemm_v4 requires k % 16 == 0 and n % 4 == 0 (got {m}x{n}x{k})");
                }
                (
                    &self.pipeline_v4,
                    (n as u32).div_ceil(V4_BN),
                    (m as u32).div_ceil(V4_BM),
                )
            }
            Some(false) => (
                &self.pipeline,
                (m as u32).div_ceil(self.tile.bm),
                (n as u32).div_ceil(self.tile.bn),
            ),
            None => self.gemm_dispatch(m, n, k),
        };

        // Warm up + retire upload before starting the timer.
        {
            let mut enc = self.encoder("gemm_warmup");
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gemm_warmup"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            self.queue.submit(Some(enc.finish()));
            self.device.poll(wgpu::PollType::wait_indefinitely())?;
        }

        let started = std::time::Instant::now();
        {
            let mut enc = self.encoder("gemm_time");
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gemm_time"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                for _ in 0..reps.max(1) {
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
            self.queue.submit(Some(enc.finish()));
            self.device.poll(wgpu::PollType::wait_indefinitely())?;
        }
        let elapsed = started.elapsed() / reps.max(1) as u32;
        let out = self.read_buffer(&buffer_c, m * n)?;
        Ok((out, elapsed))
    }

    /// Same product with an explicit kernel selector (`Some(true)` = v4 128x128,
    /// `Some(false)` = 64x64, `None` = automatic [`Self::gemm_dispatch`]).
    pub fn gemm_reps_mode(
        &self,
        a: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
        reps: usize,
        force_v4: Option<bool>,
    ) -> Result<Vec<f32>> {
        if a.len() < m * k || w.len() < n * k {
            bail!("operands are too small for {m}x{n}x{k}");
        }
        let device = &self.device;

        let buffer_a = self.upload("a", a);
        let buffer_w = self.upload("w", w);
        let c_len = ((m * n * 4).div_ceil(16) * 16) as u64;
        let buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c"),
            size: c_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dims = Dims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            _pad: 0,
        };
        let buffer_dims = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dims"),
            size: std::mem::size_of::<Dims>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&buffer_dims, 0, bytemuck::bytes_of(&dims));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_dims.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer_c.as_entire_binding(),
                },
            ],
        });

        let (pipeline, gx, gy) = match force_v4 {
            Some(true) => {
                if !k.is_multiple_of(16) || !n.is_multiple_of(4) {
                    bail!("gemm_v4 requires k % 16 == 0 and n % 4 == 0 (got {m}x{n}x{k})");
                }
                (
                    &self.pipeline_v4,
                    (n as u32).div_ceil(V4_BN),
                    (m as u32).div_ceil(V4_BM),
                )
            }
            Some(false) => (
                &self.pipeline,
                (m as u32).div_ceil(self.tile.bm),
                (n as u32).div_ceil(self.tile.bn),
            ),
            None => self.gemm_dispatch(m, n, k),
        };

        let started = std::time::Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gemm"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            for _ in 0..reps.max(1) {
                pass.dispatch_workgroups(gx, gy, 1);
            }
        }

        let compute_elapsed = started.elapsed();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: c_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buffer_c, 0, &staging, 0, c_len);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver
            .recv()
            .map_err(|error| anyhow::anyhow!("readback cancelled: {error}"))?
            .map_err(|error| anyhow::anyhow!("readback failed: {error}"))?;

        let mapped = slice.get_mapped_range()?;
        let out: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        let _ = compute_elapsed;
        Ok(out)
    }

    fn upload(&self, label: &str, values: &[f32]) -> wgpu::Buffer {
        // The shader binds these as `array<vec4<f32>>`, so pad to a 16-byte
        // multiple; the padding lanes are never read because the kernel masks by
        // `m`/`n`/`k`.
        let mut bytes = bytemuck::cast_slice::<f32, u8>(values).to_vec();
        let padded = bytes.len().next_multiple_of(16);
        bytes.resize(padded, 0);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len().max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, &bytes);
        buffer
    }
}

/// A storage binding; `read_only` must match the shader's access exactly — wgpu
/// validates the pair and rejects a `read` shader binding against a read-write
/// layout entry.
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl GpuGemm {
    /// SiLU over a buffer, in place: `x = x / (1 + exp(-x))`.
    pub fn silu(&self, values: &mut [f32]) -> Result<()> {
        let device = &self.device;
        let mut bytes = bytemuck::cast_slice::<f32, u8>(values).to_vec();
        bytes.resize(bytes.len().next_multiple_of(16), 0);
        // `COPY_SRC` because this buffer is also the readback source.
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("silu_x"),
            size: bytes.len().max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, &bytes);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("silu"),
            layout: &self.silu_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("silu"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("silu"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.silu);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((values.len() as u32).div_ceil(256), 1, 1);
        }

        let len = ((values.len() * 4).div_ceil(16) * 16) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("silu_readback"),
            size: len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, len);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver
            .recv()
            .map_err(|error| anyhow::anyhow!("silu readback cancelled: {error}"))?
            .map_err(|error| anyhow::anyhow!("silu readback failed: {error}"))?;

        let mapped = slice.get_mapped_range()?;
        let out: &[f32] = bytemuck::cast_slice(&mapped);
        values.copy_from_slice(&out[..values.len()]);
        drop(mapped);
        staging.unmap();
        Ok(())
    }

    /// LayerNorm over the last axis of a `[steps][width]` buffer.
    pub fn layer_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        steps: usize,
        width: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        if x.len() < steps * width || weight.len() < width || bias.len() < width {
            bail!("layer_norm operands are too small for {steps}x{width}");
        }
        let device = &self.device;

        let buffer_x = self.upload("x", x);
        let buffer_w = self.upload("ln_w", weight);
        let buffer_b = self.upload("ln_b", bias);
        let out_len = ((steps * width * 4).div_ceil(16) * 16) as u64;
        let buffer_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ln_out"),
            size: out_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = LayerNormParams {
            steps: steps as u32,
            width: width as u32,
            eps,
            _pad: 0.0,
        };
        let buffer_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ln_params"),
            size: std::mem::size_of::<LayerNormParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&buffer_params, 0, bytemuck::bytes_of(&params));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_norm"),
            layout: &self.layer_norm_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffer_out.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("layer_norm"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("layer_norm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.layer_norm);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(steps as u32, 1, 1);
        }
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ln_readback"),
            size: out_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buffer_out, 0, &staging, 0, out_len);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver
            .recv()
            .map_err(|error| anyhow::anyhow!("layer_norm readback cancelled: {error}"))?
            .map_err(|error| anyhow::anyhow!("layer_norm readback failed: {error}"))?;

        let mapped = slice.get_mapped_range()?;
        let values: &[f32] = bytemuck::cast_slice(&mapped);
        let out = values[..steps * width].to_vec();
        drop(mapped);
        staging.unmap();
        Ok(out)
    }
}

/// Splits a 1D workgroup count into `(x, y)` within WebGPU's 65,535-per-axis limit.
///
/// Shaders pair this with `let at = id.y * (grid.x * 256u) + id.x;`. Counts up to
/// 65,535 stay 1D (`y == 1`), so short utterances pay nothing; long ones (e.g.
/// `silu` past 5.5 minutes or `axpy`/`dw_conv` past 21.8 minutes) keep working
/// without hitting the per-dimension dispatch ceiling.
pub(crate) fn grid_2d(workgroups: usize) -> (u32, u32) {
    const MAX_X: usize = 65_535;
    if workgroups <= MAX_X {
        (workgroups as u32, 1)
    } else {
        let y = workgroups.div_ceil(MAX_X);
        let x = workgroups.div_ceil(y);
        (x as u32, y as u32)
    }
}

impl GpuGemm {
    /// GLU over the channel axis: `out[t, c] = in[t, c] * sigmoid(in[t, c + width])`.
    pub fn glu(&self, gates: &[f32], steps: usize, width: usize) -> Result<Vec<f32>> {
        let device = &self.device;
        let buffer_x = self.upload("glu_x", gates);
        let out_len = ((steps * width * 4).div_ceil(16) * 16) as u64;
        let buffer_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glu_out"),
            size: out_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = FourU32 {
            a: steps as u32,
            b: width as u32,
            c: 0,
            d: 0,
        };
        let buffer_params = self.uniform_buffer("glu_params", &params);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glu"),
            layout: &self.glu_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_out.as_entire_binding(),
                },
            ],
        });
        self.dispatch_and_read(&self.glu, &bind_group, (steps * width).div_ceil(256), &buffer_out, out_len, steps * width)
    }

    /// Depthwise Conv1d + folded BatchNorm + SiLU in one pass.
    ///
    /// The argument list mirrors the kernel's bindings one-to-one; grouping it into
    /// a struct would only move the same fields behind one more name.
    #[allow(clippy::too_many_arguments)]
    pub fn dw_conv_bn_silu(
        &self,
        x: &[f32],
        weights: &[f32],
        scale: &[f32],
        shift: &[f32],
        steps: usize,
        channels: usize,
        kernel: usize,
        padding: usize,
    ) -> Result<Vec<f32>> {
        let device = &self.device;
        let buffer_x = self.upload("dw_x", x);
        let buffer_w = self.upload("dw_w", weights);
        let buffer_scale = self.upload("dw_scale", scale);
        let buffer_shift = self.upload("dw_shift", shift);
        let out_len = ((steps * channels * 4).div_ceil(16) * 16) as u64;
        let buffer_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dw_out"),
            size: out_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = FourU32 {
            a: steps as u32,
            b: channels as u32,
            c: kernel as u32,
            d: padding as u32,
        };
        let buffer_params = self.uniform_buffer("dw_params", &params);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dw_conv"),
            layout: &self.dw_conv_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer_scale.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffer_shift.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffer_out.as_entire_binding(),
                },
            ],
        });
        self.dispatch_and_read(
            &self.dw_conv,
            &bind_group,
            (steps * channels).div_ceil(256),
            &buffer_out,
            out_len,
            steps * channels,
        )
    }

    pub(crate) fn uniform_buffer<T: bytemuck::Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytemuck::bytes_of(value));
        buffer
    }

    fn dispatch_and_read(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: usize,
        source: &wgpu::Buffer,
        byte_len: u64,
        values: usize,
    ) -> Result<Vec<f32>> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            let (gx, gy) = grid_2d(workgroups);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_len);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver
            .recv()
            .map_err(|error| anyhow::anyhow!("readback cancelled: {error}"))?
            .map_err(|error| anyhow::anyhow!("readback failed: {error}"))?;
        let mapped = slice.get_mapped_range()?;
        let all: &[f32] = bytemuck::cast_slice(&mapped);
        let out = all[..values].to_vec();
        drop(mapped);
        staging.unmap();
        Ok(out)
    }
}

/// A compute pipeline paired with the bind group layout it was built from.
pub(crate) struct CachedPipeline {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,
}

impl GpuGemm {
    pub(crate) fn encoder(&self, label: &'static str) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    pub(crate) fn upload_plain(&self, label: &str, values: &[f32]) -> wgpu::Buffer {
        self.upload(label, values)
    }

    /// Starts a pipelined zero-copy bulk weight upload session.
    ///
    /// Keeps up to two 64 MiB DMA submissions in flight while staging a third (at most
    /// 192 MiB of staging RAM), eliminating PCIe idle bubbles between chunks while
    /// keeping every individual DMA burst well under Windows' 2 s WDDM watchdog.
    pub(crate) fn bulk_upload(&self) -> BulkUpload<'_> {
        BulkUpload {
            gpu: self,
            pending: 0,
            in_flight: [None, None],
        }
    }

    /// A storage buffer sized for `count` floats, usable as a copy source.
    pub(crate) fn storage_out(&self, label: &str, count: usize) -> wgpu::Buffer {
        let size = ((count * 4).div_ceil(16) * 16) as u64;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub(crate) fn bind_group(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        entries: &[(u32, wgpu::BindingResource<'_>)],
    ) -> wgpu::BindGroup {
        let owned: Vec<wgpu::BindGroupEntry<'_>> = entries
            .iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: resource.clone(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &owned,
        })
    }

    /// Copies `count` floats out of `source` and maps them back.
    ///
    /// This driver intermittently rejects `map_async` while the copy is still in
    /// flight, which surfaces as an aborted map rather than a wrong value. Retrying
    /// re-copies the same bytes, so the result cannot change. The caller's encoder
    /// carries the work that produced `source`; retries use a fresh encoder that only
    /// re-issues the copy, which is safe once the first submit has retired.
    pub(crate) fn submit_and_read(
        &self,
        encoder: wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f32>> {
        let byte_len = ((count * 4).div_ceil(16) * 16) as u64;
        let mut pending = Some(encoder);
        let mut last = String::new();
        for attempt in 0..3u32 {
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: byte_len,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut attempt_encoder = match pending.take() {
                Some(encoder) => encoder,
                None => self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("readback"),
                }),
            };
            attempt_encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_len);
            self.queue.submit(Some(attempt_encoder.finish()));
            self.device.poll(wgpu::PollType::wait_indefinitely())?;

            let slice = staging.slice(..byte_len);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            self.device.poll(wgpu::PollType::wait_indefinitely())?;
            match receiver
                .recv()
                .map_err(|error| anyhow::anyhow!("readback cancelled: {error}"))?
            {
                Ok(()) => {
                    let mapped = slice.get_mapped_range()?;
                    let all: &[f32] = bytemuck::cast_slice(&mapped);
                    let out = all[..count].to_vec();
                    drop(mapped);
                    staging.unmap();
                    return Ok(out);
                }
                Err(error) => {
                    last = format!("{error} (attempt {attempt})");
                }
            }
        }
        anyhow::bail!("readback failed after retries: {last}")
    }

    /// Builds the scores kernel for a key window of at most `max_steps` frames.
    ///
    /// `max_steps` sizes a shared array, so it must cover every utterance the pipeline
    /// will see; the device limit is checked here rather than discovered as silent
    /// out-of-bounds writes.
    pub(crate) fn attn_scores_pipeline(&self, max_steps: usize) -> Result<CachedPipeline> {
        let bytes = max_steps * std::mem::size_of::<f32>();
        let limit = self.device.limits().max_compute_workgroup_storage_size as usize;
        if bytes > limit {
            anyhow::bail!(
                "attention needs {bytes} B of workgroup storage for {max_steps} frames, device allows {limit} B"
            );
        }
        let source = include_str!("attn_scores_softmax.wgsl")
            .replace("{{MAX_STEPS}}", &format!("{max_steps}u"));
        self.build_attn_pipeline(
            "attn_scores",
            Box::leak(source.into_boxed_str()),
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
        )
    }

    pub(crate) fn attn_context_pipeline(&self) -> Result<CachedPipeline> {
        self.build_attn_pipeline(
            "attn_context",
            include_str!("attn_context.wgsl"),
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        )
    }

    fn build_attn_pipeline(
        &self,
        label: &'static str,
        source: &'static str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Result<CachedPipeline> {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            });
        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Ok(CachedPipeline { pipeline, layout })
    }
}

/// A uniform buffer layout entry at `binding`.
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl GpuGemm {
    pub(crate) fn dev(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue_ref(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn pipeline_cache(&self) -> Option<&wgpu::PipelineCache> {
        self.pipeline_cache.as_ref()
    }

    /// Persists the driver's compiled shader cache so future runs skip SPIR-V codegen.
    pub fn save_pipeline_cache(&self) {
        let (Some(cache), Some(path)) = (&self.pipeline_cache, &self.pipeline_cache_path) else {
            return;
        };
        let Some(data) = cache.get_data() else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(path, &data);
    }

    /// `(pipeline, layout)` for the tiled GEMM.
    pub(crate) fn gemm_pipeline(&self) -> (&wgpu::ComputePipeline, &wgpu::BindGroupLayout) {
        (&self.pipeline, &self.layout)
    }

    /// Selects the faster GEMM pipeline (`v4` 128x128 vs 64x64) and its dispatch grid.
    ///
    /// Uses `v4` whenever `m >= 256`, `n >= 128`, `k % 16 == 0`, `n % 4 == 0`, the 128x128
    /// grid supplies at least [`V4_MIN_WORKGROUPS`] workgroups, and 128x128 padding stays
    /// within 10% of 64x64 padding (avoiding small-`m` tile waste such as `m=189`).
    pub(crate) fn gemm_dispatch(
        &self,
        m: usize,
        n: usize,
        k: usize,
    ) -> (&wgpu::ComputePipeline, u32, u32) {
        if !self.tile_override && k.is_multiple_of(16) && n.is_multiple_of(4) && m >= 256 && n >= 128
        {
            let gm = (m as u32).div_ceil(V4_BM);
            let gn = (n as u32).div_ceil(V4_BN);
            let area128 = (gm as u64 * V4_BM as u64) * (gn as u64 * V4_BN as u64);
            let area64 = ((m as u64).div_ceil(self.tile.bm as u64) * self.tile.bm as u64)
                * ((n as u64).div_ceil(self.tile.bn as u64) * self.tile.bn as u64);
            if gm.saturating_mul(gn) >= V4_MIN_WORKGROUPS && area128 * 10 <= area64 * 11 {
                // `v4` maps `m0 = wid.y * 128`, `n0 = wid.x * 128`.
                return (&self.pipeline_v4, gn, gm);
            }
        }
        (
            &self.pipeline,
            (m as u32).div_ceil(self.tile.bm),
            (n as u32).div_ceil(self.tile.bn),
        )
    }

    pub(crate) fn layer_norm_pipeline(&self) -> (&wgpu::ComputePipeline, &wgpu::BindGroupLayout) {
        (&self.layer_norm, &self.layer_norm_layout)
    }

    pub(crate) fn silu_pipeline(&self) -> (&wgpu::ComputePipeline, &wgpu::BindGroupLayout) {
        (&self.silu, &self.silu_layout)
    }

    pub(crate) fn glu_pipeline(&self) -> (&wgpu::ComputePipeline, &wgpu::BindGroupLayout) {
        (&self.glu, &self.glu_layout)
    }

    pub(crate) fn dw_conv_pipeline(&self) -> (&wgpu::ComputePipeline, &wgpu::BindGroupLayout) {
        (&self.dw_conv, &self.dw_conv_layout)
    }

    pub(crate) fn read_buffer(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<f32>> {
        let encoder = self.encoder("readback");
        self.submit_and_read(encoder, buffer, count)
    }
}

/// Pipelined load-time weight uploader modeled after `qwen3-asr-wgpu`'s `BulkUpload`.
///
/// Streams raw bytes directly from the memory-mapped `.nemo` archive into device
/// storage buffers in bounded 64 MiB chunks with a 2-deep in-flight queue, keeping
/// PCIe DMA saturated back-to-back while capping staging RAM to ~192 MiB.
pub(crate) struct BulkUpload<'a> {
    gpu: &'a GpuGemm,
    pending: u64,
    in_flight: [Option<wgpu::SubmissionIndex>; 2],
}

const BULK_STAGING_BUDGET: u64 = 64 << 20;

impl<'a> BulkUpload<'a> {
    /// Allocates a 16-byte-padded storage buffer for `byte_len` bytes.
    pub(crate) fn storage(&self, label: &str, byte_len: usize) -> wgpu::Buffer {
        let size = byte_len.next_multiple_of(16).max(16) as u64;
        self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Writes `data` at `offset` inside `buf`, pumping the queue whenever staged bytes
    /// reach [`BULK_STAGING_BUDGET`].
    pub(crate) fn upload_at(&mut self, buf: &wgpu::Buffer, offset: u64, data: &[u8]) -> Result<()> {
        debug_assert_eq!(offset % 4, 0, "upload_at offset must be 4-byte aligned");
        debug_assert_eq!(data.len() % 4, 0, "upload_at length must be 4-byte aligned");
        let budget = BULK_STAGING_BUDGET as usize;
        let mut off = offset;
        for piece in data.chunks(budget) {
            if self.pending + piece.len() as u64 > BULK_STAGING_BUDGET {
                self.pump()?;
            }
            self.gpu.queue.write_buffer(buf, off, piece);
            self.pending += piece.len() as u64;
            off += piece.len() as u64;
        }
        Ok(())
    }

    /// Allocates a storage buffer and streams `data` directly into it.
    pub(crate) fn upload_bytes(&mut self, label: &str, data: &[u8]) -> Result<wgpu::Buffer> {
        let buf = self.storage(label, data.len());
        self.upload_at(&buf, 0, data)?;
        Ok(buf)
    }

    /// Streams a checkpoint tensor straight from mmap into a new GPU buffer.
    pub(crate) fn upload_tensor(
        &mut self,
        label: &str,
        store: &crate::torch::Store,
        name: &str,
    ) -> Result<wgpu::Buffer> {
        let bytes = store.f32_bytes(name)?;
        self.upload_bytes(label, &bytes)
    }

    /// Concatenates multiple tensors directly into one GPU buffer with zero host copy.
    pub(crate) fn upload_concat(
        &mut self,
        label: &str,
        store: &crate::torch::Store,
        names: &[String],
    ) -> Result<wgpu::Buffer> {
        let slices: Vec<std::borrow::Cow<'_, [u8]>> = names
            .iter()
            .map(|name| store.f32_bytes(name))
            .collect::<Result<_>>()?;
        let total: usize = slices.iter().map(|s| s.len()).sum();
        let buf = self.storage(label, total);
        let mut offset = 0u64;
        for slice in &slices {
            self.upload_at(&buf, offset, slice)?;
            offset += slice.len() as u64;
        }
        Ok(buf)
    }

    /// Submits staged writes and waits only for the submission *two* steps back,
    /// keeping PCIe DMA busy back-to-back while bounding in-flight staging to ~192 MiB.
    fn pump(&mut self) -> Result<()> {
        if self.pending == 0 {
            return Ok(());
        }
        let idx = self.gpu.queue.submit([]);
        self.pending = 0;
        let oldest = self.in_flight[0].take();
        self.in_flight[0] = self.in_flight[1].take();
        self.in_flight[1] = Some(idx);
        if let Some(prev) = oldest {
            self.gpu.device.poll(wgpu::PollType::Wait {
                submission_index: Some(prev),
                timeout: None,
            })?;
        }
        Ok(())
    }

    /// Retires all remaining staged writes before compute begins.
    pub(crate) fn finish(mut self) -> Result<()> {
        if self.pending > 0 {
            let idx = self.gpu.queue.submit([]);
            self.pending = 0;
            self.in_flight[1] = Some(idx);
        }
        if let Some(last) = self.in_flight[1].take().or_else(|| self.in_flight[0].take()) {
            self.gpu.device.poll(wgpu::PollType::Wait {
                submission_index: Some(last),
                timeout: None,
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_maps_every_documented_spec() {
        assert_eq!(DeviceSelector::parse("").unwrap(), DeviceSelector::Auto);
        assert_eq!(DeviceSelector::parse("auto").unwrap(), DeviceSelector::Auto);
        assert_eq!(DeviceSelector::parse("AUTO").unwrap(), DeviceSelector::Auto);
        assert_eq!(DeviceSelector::parse(" cpu ").unwrap(), DeviceSelector::Cpu);
        assert_eq!(DeviceSelector::parse("#1").unwrap(), DeviceSelector::Index(1));
        assert_eq!(DeviceSelector::parse("2").unwrap(), DeviceSelector::Index(2));

        for (spec, api) in [
            ("vulkan", wgpu::Backend::Vulkan),
            ("VULKAN:1", wgpu::Backend::Vulkan),
            ("metal", wgpu::Backend::Metal),
            ("metal:3", wgpu::Backend::Metal),
            ("dx12", wgpu::Backend::Dx12),
            ("d3d12:2", wgpu::Backend::Dx12),
            ("gl", wgpu::Backend::Gl),
        ] {
            let index = spec.split_once(':').map_or(0, |(_, i)| i.parse().unwrap());
            assert_eq!(
                DeviceSelector::parse(spec).unwrap(),
                DeviceSelector::Runtime { api, index },
                "{spec}"
            );
        }

        // Anything else is a name, matched case-insensitively against the adapter name.
        assert_eq!(
            DeviceSelector::parse("GeForce").unwrap(),
            DeviceSelector::Name("geforce".into())
        );
        assert_eq!(
            DeviceSelector::parse("pci:0").unwrap(),
            DeviceSelector::Name("pci:0".into())
        );
    }

    #[test]
    fn parse_rejects_backend_names_and_bad_indices() {
        // `gpu` names a backend, not an adapter.
        assert!(DeviceSelector::parse("gpu").is_err());
        assert!(DeviceSelector::parse("#abc").is_err());
        assert!(DeviceSelector::parse("vulkan:abc").is_err());
        // A bad index must not silently become a name.
        assert!(!matches!(
            DeviceSelector::parse("#abc"),
            Ok(DeviceSelector::Name(_))
        ));
    }

    #[test]
    fn auto_ranks_discrete_before_integrated_before_virtual_before_cpu() {
        use wgpu::DeviceType::{Cpu, DiscreteGpu, IntegratedGpu, VirtualGpu};
        assert!(device_class(DiscreteGpu) < device_class(IntegratedGpu));
        assert!(device_class(IntegratedGpu) < device_class(VirtualGpu));
        assert!(device_class(VirtualGpu) < device_class(Cpu));
        // A CPU adapter is not a GPU, so `Auto` never picks one.
        assert!(!is_real_gpu(Cpu));
        assert!(is_real_gpu(DiscreteGpu));
        assert!(is_real_gpu(IntegratedGpu));
        assert!(is_real_gpu(VirtualGpu));
    }

    #[test]
    fn api_rank_is_the_order_table() {
        use wgpu::Backend as B;
        // The table is the ranking: listed backends rank 0..3 in its order.
        for (i, backend) in API_RANK_ORDER.iter().enumerate() {
            assert_eq!(api_rank(*backend), i as u8, "{backend:?}");
        }
        #[cfg(target_os = "macos")]
        {
            assert!(api_rank(B::Metal) < api_rank(B::Vulkan));
            assert!(api_rank(B::Vulkan) < api_rank(B::Dx12));
        }
        #[cfg(not(target_os = "macos"))]
        {
            assert!(api_rank(B::Vulkan) < api_rank(B::Metal));
            assert!(api_rank(B::Metal) < api_rank(B::Dx12));
        }
        assert!(api_rank(B::Dx12) < api_rank(B::Gl));
        // Anything unlisted ranks last.
        assert_eq!(api_rank(B::BrowserWebGpu), 4);
        assert_eq!(api_rank(B::Noop), 4);
    }

    #[test]
    fn rank_orders_device_class_before_api() {
        use wgpu::Backend::{Dx12, Vulkan};
        use wgpu::DeviceType::{Cpu, DiscreteGpu, IntegratedGpu, VirtualGpu};
        assert!(rank(DiscreteGpu, Vulkan) < rank(IntegratedGpu, Vulkan));
        assert!(rank(IntegratedGpu, Vulkan) < rank(VirtualGpu, Vulkan));
        assert!(rank(VirtualGpu, Vulkan) < rank(Cpu, Vulkan));
        // Class wins over API: a discrete DX12 adapter beats an integrated Vulkan one.
        assert!(rank(DiscreteGpu, Dx12) < rank(IntegratedGpu, Vulkan));
    }
}
