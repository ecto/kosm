//! SU(3) lattice gauge theory in WGSL compute shaders.
//!
//! The same algorithms as phyz-qft's `su3` module (Cabibbo-Marinari heatbath
//! and overrelaxation, stout smearing, clover field strength, Wilson,
//! Polyakov and three-quark loops) in f32 on the GPU, so it runs on WebGPU in
//! a browser as well as natively. `tests` checks every kernel against
//! phyz-qft on the same gauge configuration.
//!
//! Every dispatch takes its [`Params`] from one uniform buffer at a dynamic
//! offset, so a whole Monte Carlo cycle encodes into one submission.

pub mod engine;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use eframe::egui_wgpu::wgpu;
use wgpu::util::DeviceExt;

/// Largest Wilson loop extent the kernel supports (its RCAP).
pub const RMAX_CAP: usize = 12;
/// Longest quark staircase the baryon kernel supports.
pub const PATH_CAP: usize = 32;
const SLOT: u64 = 256;
const MAX_SLOTS: u64 = 2048;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub dims: [u32; 4],
    pub n: u32,
    pub mu: u32,
    pub parity: u32,
    pub kind: u32,
    pub seed: u32,
    pub pass_id: u32,
    pub t_len: u32,
    pub count: u32,
    pub beta: f32,
    pub rho: f32,
    pub extra0: u32,
    pub extra1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Paths {
    len: [u32; 4],
    steps: [[u32; 4]; 24],
}

#[derive(Clone, Copy)]
enum Bind {
    /// The per-dispatch Params, at a dynamic offset.
    Params,
    Uniform,
    Read,
    Write,
}

struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl Kernel {
    fn new(device: &wgpu::Device, name: &str, src: &str, lattice: bool, binds: &[Bind]) -> Self {
        let mut code = String::from(include_str!("su3.wgsl"));
        if lattice {
            code.push_str(include_str!("lattice.wgsl"));
        }
        code.push_str(src);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        let entries: Vec<_> = binds
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: match b {
                    Bind::Params => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<Params>() as u64
                        ),
                    },
                    Bind::Uniform => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    Bind::Read => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    Bind::Write => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(name),
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(name),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }
}

pub struct Kernels {
    update: Kernel,
    stout: Kernel,
    fields: Kernel,
    polyakov: Kernel,
    wilson: Kernel,
    baryon: Kernel,
    correlate: Kernel,
    reduce: Kernel,
}

impl Kernels {
    pub fn new(device: &wgpu::Device) -> Self {
        use Bind::*;
        Self {
            update: Kernel::new(
                device,
                "update",
                include_str!("update.wgsl"),
                true,
                &[Params, Write, Read],
            ),
            stout: Kernel::new(
                device,
                "stout",
                include_str!("stout.wgsl"),
                true,
                &[Params, Read, Write],
            ),
            fields: Kernel::new(
                device,
                "fields",
                include_str!("fields.wgsl"),
                true,
                &[Params, Read, Write],
            ),
            polyakov: Kernel::new(
                device,
                "polyakov",
                include_str!("polyakov.wgsl"),
                true,
                &[Params, Read, Write],
            ),
            wilson: Kernel::new(
                device,
                "wilson",
                include_str!("wilson.wgsl"),
                true,
                &[Params, Read, Write],
            ),
            baryon: Kernel::new(
                device,
                "baryon",
                include_str!("baryon.wgsl"),
                true,
                &[Params, Read, Write, Uniform],
            ),
            correlate: Kernel::new(
                device,
                "correlate",
                include_str!("correlate.wgsl"),
                false,
                &[Params, Read, Read, Write],
            ),
            reduce: Kernel::new(
                device,
                "reduce",
                include_str!("reduce.wgsl"),
                false,
                &[Params, Read, Write],
            ),
        }
    }
}

/// Whether this adapter can run the compute pipeline (WebGL2 cannot).
pub fn supported(adapter: &wgpu::Adapter) -> bool {
    adapter
        .get_downlevel_capabilities()
        .flags
        .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
}

/// Which link buffer a measurement reads.
#[derive(Clone, Copy, PartialEq)]
pub enum Links {
    Rough,
    Smooth,
}

/// A lattice's buffers and bind groups.
pub struct GpuLattice {
    pub dims: [usize; 4],
    pub n: usize,
    /// Spatial sites.
    pub ns: usize,
    pub rmax: usize,
    pub tmax: usize,
    pub links: wgpu::Buffer,
    pub smooth_a: wgpu::Buffer,
    /// Ping-pong partner of `smooth_a`; only the bind groups use it.
    _smooth_b: wgpu::Buffer,
    params: wgpu::Buffer,
    pub fields: wgpu::Buffer,
    #[cfg_attr(not(test), allow(dead_code))]
    pub wilson: wgpu::Buffer,
    pub poly: wgpu::Buffer,
    #[cfg_attr(not(test), allow(dead_code))]
    pub bw: wgpu::Buffer,
    pub corr: wgpu::Buffer,
    pub partial: wgpu::Buffer,
    paths: wgpu::Buffer,
    bg: BindGroups,
}

struct BindGroups {
    update: wgpu::BindGroup,
    stout_ab: wgpu::BindGroup,
    stout_ba: wgpu::BindGroup,
    fields_rough: wgpu::BindGroup,
    fields_smooth: wgpu::BindGroup,
    poly: wgpu::BindGroup,
    wilson: wgpu::BindGroup,
    baryon: wgpu::BindGroup,
    correlate: wgpu::BindGroup,
    reduce_fields: wgpu::BindGroup,
    reduce_wilson: wgpu::BindGroup,
    reduce_bw: wgpu::BindGroup,
}

/// Groups of 256 the reduce kernel splits `n` elements into.
pub fn groups(n: usize) -> usize {
    n.div_ceil(256)
}

impl GpuLattice {
    pub fn new(device: &wgpu::Device, k: &Kernels, dims: [usize; 4]) -> Self {
        assert!(
            dims.iter().all(|&d| d >= 2 && d % 2 == 0),
            "even extents ≥ 2 needed, got {dims:?}"
        );
        let n: usize = dims.iter().product();
        let ns = n / dims[0];
        let rmax = (dims[1].min(dims[2]).min(dims[3]) / 2).min(RMAX_CAP);
        let tmax = (dims[0] / 2).clamp(1, 6);
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let buf = |label: &str, bytes: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16) as u64,
                usage: storage,
                mapped_at_creation: false,
            })
        };
        let link_bytes = 4 * n * 9 * 8;
        let links = buf("links", link_bytes);
        let smooth_a = buf("smooth a", link_bytes);
        let smooth_b = buf("smooth b", link_bytes);
        let fields = buf("fields", 6 * n * 4);
        let wilson = buf("wilson", rmax * tmax * n * 4);
        let poly = buf("polyakov", ns * 8);
        let bw = buf("baryon w", n * 4);
        let corr = buf("correlation", 2 * ns * 4);
        let partial = buf("partial", (rmax * tmax).max(6) * groups(n) * 4);
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: SLOT * MAX_SLOTS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let paths = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("paths"),
            size: std::mem::size_of::<Paths>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // checkerboard site lists: even sites, then odd
        let mut sites: Vec<u32> = Vec::with_capacity(n);
        for parity in 0..2 {
            sites.extend((0..n as u32).filter(|&s| {
                let c = coords(dims, s as usize);
                c.iter().sum::<usize>() % 2 == parity
            }));
        }
        let site_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sites"),
            contents: bytemuck::cast_slice(&sites),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let params_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &params,
            offset: 0,
            size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64),
        });
        let make = |kernel: &Kernel, label: &str, bufs: &[&wgpu::Buffer]| {
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: params_binding.clone(),
            }];
            entries.extend(bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: b.as_entire_binding(),
            }));
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &kernel.layout,
                entries: &entries,
            })
        };
        let bg = BindGroups {
            update: make(&k.update, "update", &[&links, &site_buf]),
            stout_ab: make(&k.stout, "stout a→b", &[&smooth_a, &smooth_b]),
            stout_ba: make(&k.stout, "stout b→a", &[&smooth_b, &smooth_a]),
            fields_rough: make(&k.fields, "fields rough", &[&links, &fields]),
            fields_smooth: make(&k.fields, "fields smooth", &[&smooth_a, &fields]),
            poly: make(&k.polyakov, "polyakov", &[&smooth_a, &poly]),
            wilson: make(&k.wilson, "wilson", &[&smooth_a, &wilson]),
            baryon: make(&k.baryon, "baryon", &[&smooth_a, &bw, &paths]),
            correlate: make(&k.correlate, "correlate", &[&bw, &fields, &corr]),
            reduce_fields: make(&k.reduce, "reduce fields", &[&fields, &partial]),
            reduce_wilson: make(&k.reduce, "reduce wilson", &[&wilson, &partial]),
            reduce_bw: make(&k.reduce, "reduce w", &[&bw, &partial]),
        };
        Self {
            dims,
            n,
            ns,
            rmax,
            tmax,
            links,
            smooth_a,
            _smooth_b: smooth_b,
            params,
            fields,
            wilson,
            poly,
            bw,
            corr,
            partial,
            paths,
            bg,
        }
    }

    pub fn base_params(&self) -> Params {
        Params {
            dims: self.dims.map(|d| d as u32),
            n: self.n as u32,
            extra0: self.rmax as u32,
            extra1: self.tmax as u32,
            ..Default::default()
        }
    }

    /// Upload links in phyz-qft order: `links[mu][site]` as 3×3 row-major
    /// complex f32.
    pub fn upload_links(&self, queue: &wgpu::Queue, data: &[[f32; 2]]) {
        assert_eq!(data.len(), 4 * self.n * 9);
        queue.write_buffer(&self.links, 0, bytemuck::cast_slice(data));
    }

    /// Hot start drawn on the CPU: Gram-Schmidt of Gaussian complex matrices.
    pub fn hot_links(&self, seed: u64) -> Vec<[f32; 2]> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut gauss = || {
            let (u1, u2) = (1.0 - next(), next());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        let mut out = Vec::with_capacity(4 * self.n * 9);
        for _ in 0..4 * self.n {
            let mut m = phyz_qft::su3::Su3::ZERO;
            for row in &mut m.m {
                for e in row {
                    *e = phyz_qft::su3::C64::new(gauss(), gauss());
                }
            }
            m.reunitarize();
            out.extend(m.m.iter().flatten().map(|c| [c.re as f32, c.im as f32]));
        }
        out
    }

    /// Set the three quark staircases for the baryon kernel.
    pub fn set_paths(&self, queue: &wgpu::Queue, paths: &[Vec<(usize, bool)>; 3]) {
        let mut p = Paths {
            len: [0; 4],
            steps: [[0; 4]; 24],
        };
        for (q, path) in paths.iter().enumerate() {
            assert!(path.len() <= PATH_CAP, "quark path too long");
            p.len[q] = path.len() as u32;
            for (s, &(axis, forward)) in path.iter().enumerate() {
                let k = q * PATH_CAP + s;
                p.steps[k / 4][k % 4] = axis as u32 | (u32::from(forward) << 4);
            }
        }
        queue.write_buffer(&self.paths, 0, bytemuck::bytes_of(&p));
    }
}

pub fn coords(dims: [usize; 4], site: usize) -> [usize; 4] {
    let [nt, nx, ny, _] = dims;
    [
        site % nt,
        (site / nt) % nx,
        (site / (nt * nx)) % ny,
        site / (nt * nx * ny),
    ]
}

/// One command submission: dispatches with their Params slots.
pub struct Batch<'a> {
    k: &'a Kernels,
    lat: &'a GpuLattice,
    pub enc: wgpu::CommandEncoder,
    slots: Vec<Params>,
}

impl<'a> Batch<'a> {
    pub fn new(device: &wgpu::Device, k: &'a Kernels, lat: &'a GpuLattice) -> Self {
        let enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lattice"),
        });
        Self {
            k,
            lat,
            enc,
            slots: Vec::new(),
        }
    }

    /// Room left for dispatches in this batch.
    pub fn room(&self) -> usize {
        MAX_SLOTS as usize - self.slots.len()
    }

    fn dispatch(&mut self, kernel: &Kernel, bg: &wgpu::BindGroup, p: Params, groups: (u32, u32)) {
        assert!(
            (self.slots.len() as u64) < MAX_SLOTS,
            "too many dispatches in one batch"
        );
        let offset = (self.slots.len() as u64 * SLOT) as u32;
        self.slots.push(p);
        let mut pass = self.enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, bg, &[offset]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }

    /// One compound update: a heatbath sweep then `n_or` overrelaxation sweeps.
    pub fn update(&mut self, beta: f32, seed: u32, pass_id: &mut u32, n_or: usize) {
        let (k, lat) = (self.k, self.lat);
        let half = (lat.n / 2) as u32;
        for kind in std::iter::once(0).chain(std::iter::repeat_n(1, n_or)) {
            for mu in 0..4 {
                for parity in 0..2 {
                    let p = Params {
                        mu,
                        parity,
                        kind,
                        seed,
                        pass_id: *pass_id,
                        beta,
                        ..lat.base_params()
                    };
                    *pass_id = pass_id.wrapping_add(1);
                    self.dispatch(&k.update, &lat.bg.update, p, (half.div_ceil(64), 1));
                }
            }
        }
    }

    /// Copy the rough links into the smoothing buffer.
    pub fn begin_smoothing(&mut self) {
        let size = self.lat.links.size();
        self.enc
            .copy_buffer_to_buffer(&self.lat.links, 0, &self.lat.smooth_a, 0, size);
    }

    /// Two stout steps (a→b→a), so the result is always in the smooth
    /// buffer. `spatial` smears only spatial links, with spatial staples.
    pub fn stout_pair(&mut self, rho: f32, spatial: bool) {
        let (k, lat) = (self.k, self.lat);
        let p = Params {
            rho,
            kind: spatial as u32,
            ..lat.base_params()
        };
        let g = ((4 * lat.n) as u32).div_ceil(64);
        self.dispatch(&k.stout, &lat.bg.stout_ab, p, (g, 1));
        self.dispatch(&k.stout, &lat.bg.stout_ba, p, (g, 1));
    }

    /// Plaquette, action, charge and electric densities into `fields`.
    pub fn fields(&mut self, links: Links) {
        let (k, lat) = (self.k, self.lat);
        let bg = match links {
            Links::Rough => &lat.bg.fields_rough,
            Links::Smooth => &lat.bg.fields_smooth,
        };
        self.dispatch(
            &k.fields,
            bg,
            lat.base_params(),
            ((lat.n as u32).div_ceil(64), 1),
        );
    }

    pub fn polyakov(&mut self) {
        let (k, lat) = (self.k, self.lat);
        self.dispatch(
            &k.polyakov,
            &lat.bg.poly,
            lat.base_params(),
            ((lat.ns as u32).div_ceil(64), 1),
        );
    }

    pub fn wilson(&mut self) {
        let (k, lat) = (self.k, self.lat);
        self.dispatch(
            &k.wilson,
            &lat.bg.wilson,
            lat.base_params(),
            ((lat.n as u32).div_ceil(64), 1),
        );
    }

    /// Three-quark loops with extent `t_len`, then their correlation with
    /// the action and electric densities (run `fields` on the same links first).
    pub fn baryon(&mut self, t_len: u32) {
        let (k, lat) = (self.k, self.lat);
        let p = Params {
            t_len,
            ..lat.base_params()
        };
        self.dispatch(
            &k.baryon,
            &lat.bg.baryon,
            p,
            ((lat.n as u32).div_ceil(64), 1),
        );
        self.dispatch(
            &k.correlate,
            &lat.bg.correlate,
            p,
            ((lat.ns as u32).div_ceil(64), 1),
        );
    }

    /// Partial sums of the first `comps` components of a component-major
    /// buffer into `partial` (`comps × groups(n)` values).
    pub fn reduce(&mut self, which: Reduce, comps: usize) {
        let (k, lat) = (self.k, self.lat);
        let g = groups(lat.n);
        let p = Params {
            count: g as u32,
            ..lat.base_params()
        };
        let bg = match which {
            Reduce::Fields => &lat.bg.reduce_fields,
            Reduce::Wilson => &lat.bg.reduce_wilson,
            Reduce::BaryonW => &lat.bg.reduce_bw,
        };
        self.dispatch(&k.reduce, bg, p, (g as u32, comps as u32));
    }

    /// Queue a copy of `size` bytes from `src` into the readback buffer.
    pub fn copy(
        &mut self,
        src: &wgpu::Buffer,
        src_offset: u64,
        dst: &wgpu::Buffer,
        dst_offset: u64,
        size: u64,
    ) {
        self.enc
            .copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size);
    }

    pub fn submit(self, queue: &wgpu::Queue) {
        let mut bytes = vec![0u8; self.slots.len() * SLOT as usize];
        for (i, p) in self.slots.iter().enumerate() {
            let at = i * SLOT as usize;
            bytes[at..at + std::mem::size_of::<Params>()].copy_from_slice(bytemuck::bytes_of(p));
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.lat.params, 0, &bytes);
        }
        queue.submit([self.enc.finish()]);
    }
}

#[derive(Clone, Copy)]
pub enum Reduce {
    Fields,
    Wilson,
    BaryonW,
}

/// A host-readable buffer and its mapping state; mapping completes
/// asynchronously on the web, so callers poll [`Readback::ready`].
pub struct Readback {
    pub buf: wgpu::Buffer,
    state: Arc<AtomicU8>,
}

const IDLE: u8 = 0;
const PENDING: u8 = 1;
const READY: u8 = 2;
const FAILED: u8 = 3;

impl Readback {
    pub fn new(device: &wgpu::Device, size: u64) -> Self {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size.max(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buf,
            state: Arc::new(AtomicU8::new(IDLE)),
        }
    }

    pub fn idle(&self) -> bool {
        self.state.load(Ordering::Acquire) == IDLE
    }

    /// Start mapping (after the copies into it were submitted).
    pub fn request(&self) {
        self.state.store(PENDING, Ordering::Release);
        let state = self.state.clone();
        self.buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            state.store(if r.is_ok() { READY } else { FAILED }, Ordering::Release);
        });
    }

    /// The mapped contents as f32 once ready; unmaps and returns to idle.
    pub fn take(&self) -> Option<Vec<f32>> {
        match self.state.load(Ordering::Acquire) {
            READY => {
                let data = {
                    let view = self.buf.slice(..).get_mapped_range().ok()?;
                    bytemuck::cast_slice::<u8, f32>(&view).to_vec()
                };
                self.buf.unmap();
                self.state.store(IDLE, Ordering::Release);
                Some(data)
            }
            FAILED => {
                self.state.store(IDLE, Ordering::Release);
                None
            }
            _ => None,
        }
    }
}
