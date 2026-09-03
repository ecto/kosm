//! MLS-MPM water on the GPU.
//!
//! The four kernels in `mpm.wgsl` are the CPU solver's four loops. A block
//! of substeps is one command submission; the only thing that comes back per
//! block is the body's reaction force (one 16-byte slot per substep, so the
//! fixed-point accumulators cannot overflow across a block). Particles and
//! grid mass are downloaded once a frame for surface extraction.
//!
//! The grid is block-sparse. The box around the melon is mostly air — two
//! metres of water under one of headroom — and a dense grid pays for all of
//! it, every substep, forever. So the node arrays are cut into blocks of
//! 4x4x4 nodes and only the blocks the water is in are stored.
//!
//! The bookkeeping is deliberately cheap, because there is nothing to be
//! gained by being clever about fifty thousand blocks. A dense table over the
//! box, one u32 per block, maps a block to its slot in the compact node
//! arrays or to NONE. The counting sort that bins particles into blocks
//! every substep already knows which blocks hold particles; a block is
//! active if any of its 27 neighbours does. That dilation is exactly right:
//! a particle in cell block b writes to nodes 4b-1 through 4b+5, the mass
//! blur reads one node further, and 4b-2 through 4b+6 is still inside the
//! blocks b-1, b and b+1. A single-workgroup prefix sum over the marks hands
//! out slots and writes the indirect dispatch the node kernels run under, so
//! the host never has to learn how many blocks there are. Node data lives at
//! slot*64 + local, and an inactive node reads as empty — which is what the
//! dense grid held there anyway.
//!
//! TODO: the wall-clock verdict is still open. At 2.5 cm the box is 69%
//! water, so the grid kernels save about a third of their work while every
//! particle in p2g and g2p pays an extra dependent load through the table.
//! Interleaved 6-frame runs put the sparse step anywhere from level with the
//! dense one to 1.4x slower, but the machine was carrying a load average of
//! 200 throughout and the same binary varied by 2.2x between runs, so none of
//! it is trustworthy. Re-measure on a quiet machine before optimising: if the
//! table read is really the cost, the fix is to give p2g_block and g2p their
//! block's slot once per workgroup rather than per node.
//!
//! The slot budget is fixed at startup from the fill's own footprint plus a
//! sixth for the splash (`NEWT_MAX_BLOCKS` overrides), and `step` panics with
//! the numbers if the water ever outgrows it.

pub mod caustic;
pub mod surface;
pub use caustic::{CausticCfg, GpuCaustic, Grid};
pub use surface::Candidates;

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const WG: u32 = 256;
const PARAMS_STRIDE: u64 = 256;

/// Static solver parameters.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub h: f32,
    pub dt: f32,
    pub origin: [f32; 3],
    pub n: [u32; 3],
    pub mass: f32,
    pub vol0: f32,
    pub bulk: f32,
    pub flip: f32,
    pub gravity: f32,
    /// Per-substep share of the mass-based J blended into the integrated J.
    pub j_relax: f32,
}

/// The rigid body the water couples to: an ellipsoid with a pose.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub centre: [f32; 3],
    pub axis: [f32; 3],
    pub vel: [f32; 3],
    pub semi: [f32; 3],
}

/// One substep's reaction: momentum the body took from the water (in
/// particle masses times m/s) and the water mass inside it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reaction {
    pub impulse: [f64; 3],
    pub interior_mass: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    origin_h: [f32; 4],
    n: [u32; 4],
    k: [f32; 4],
    k2: [f32; 4],
    lo: [f32; 4],
    hi: [f32; 4],
    misc: [u32; 4],
    xmax: [f32; 4],
    b_centre: [f32; 4],
    b_a: [f32; 4],
    b_b: [f32; 4],
    b_c: [f32; 4],
    b_vel: [f32; 4],
    semi: [f32; 4],
    misc2: [u32; 4],
}

/// Particle state as the GPU holds it.
pub struct Particles {
    /// xyz and J−1.
    pub x: Vec<[f32; 4]>,
    /// xyz and the particle's id, which survives the per-substep sort.
    pub v: Vec<[f32; 4]>,
    /// Three columns per particle.
    pub c: Vec<[f32; 4]>,
}

pub struct GpuMpm {
    device: wgpu::Device,
    queue: wgpu::Queue,
    params: Params,
    n: u32,
    nodes: u32,
    max_subs: u32,
    params_buf: wgpu::Buffer,
    gm: wgpu::Buffer,
    gmom: wgpu::Buffer,
    gvel: wgpu::Buffer,
    gvold: wgpu::Buffer,
    react: wgpu::Buffer,
    react_stage: wgpu::Buffer,
    /// Particle buffers ping-pong through the sort: [cur] holds the particles.
    bufs: [(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer); 2],
    cur: usize,
    binds: [wgpu::BindGroup; 2],
    nblocks: u32,
    nb: [u32; 3],
    /// Node slots the compact grid arrays are sized for; one slot is 4^3 nodes.
    max_slots: u32,
    btab: wgpu::Buffer,
    nact: wgpu::Buffer,
    indirect: wgpu::Buffer,
    small_stage: wgpu::Buffer,
    blk_mark: wgpu::ComputePipeline,
    blk_scan: wgpu::ComputePipeline,
    bind_indirect: wgpu::BindGroup,
    clear: wgpu::ComputePipeline,
    sort_zero: wgpu::ComputePipeline,
    sort_count: wgpu::ComputePipeline,
    sort_scan: wgpu::ComputePipeline,
    sort_scatter: wgpu::ComputePipeline,
    permute: wgpu::ComputePipeline,
    p2g: wgpu::ComputePipeline,
    grid: wgpu::ComputePipeline,
    blur: wgpu::ComputePipeline,
    g2p: wgpu::ComputePipeline,
    pub time: f64,
    /// Active slots after the last submitted substep.
    active: u32,
    damp: f32,
    sponge: f32,
    /// The surface-extraction pass, built on first use.
    surf: Option<surface::Surf>,
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: [f32; 3]) -> [f32; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-12);
    [a[0] / l, a[1] / l, a[2] / l]
}
fn v4(a: [f32; 3], w: f32) -> [f32; 4] {
    [a[0], a[1], a[2], w]
}

impl GpuMpm {
    /// Bring up a device and upload the particles. `max_subs` is the largest
    /// block `step` will be asked for.
    pub fn new(params: Params, particles: &Particles, max_subs: u32) -> Result<Self, String> {
        pollster::block_on(Self::new_async(params, particles, max_subs))
    }

    async fn new_async(params: Params, particles: &Particles, max_subs: u32) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
            .await
            .map_err(|e| format!("no adapter: {e}"))?;
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("newt-mpm"),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no device: {e}"))?;
        let n = particles.x.len() as u32;
        let nodes = params.n[0] * params.n[1] * params.n[2];
        let nb = [params.n[0].div_ceil(4), params.n[1].div_ceil(4), params.n[2].div_ceil(4)];
        let nblocks = nb[0] * nb[1] * nb[2];
        // How many block slots the node arrays get. The water's own footprint,
        // dilated the way the marking kernel dilates it, plus a sixth for the
        // splash: that is the whole point of the exercise, memory that scales
        // with the water and not with the box. Never more than the box holds.
        let max_slots = Self::slot_budget(&params, particles, nb, nblocks);
        let snodes = 64 * max_slots as u64;
        // Said before anything is allocated, so a fill that will not fit says
        // how big it was on the way out.
        if std::env::var_os("NEWT_PROF").is_some() {
            println!(
                "mpm    {n} particles; sparse grid {max_slots} of {nblocks} blocks, {snodes} of {nodes} nodes ({:.0}%), node buffers {:.0} MB of {:.0} MB dense, particle buffers {:.0} MB (largest {:.0} MB, limit {:.0} MB)",
                100.0 * snodes as f64 / nodes as f64,
                48.0 * snodes as f64 / 1e6,
                48.0 * nodes as f64 / 1e6,
                164.0 * n as f64 / 1e6,
                48.0 * n as f64 / 1e6,
                limits.max_buffer_size as f64 / 1e6,
            );
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mpm"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mpm.wgsl").into()),
        });
        let storage = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            })
        };
        let empty = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let x = storage("x", bytemuck::cast_slice(&particles.x));
        let v = storage("v", bytemuck::cast_slice(&particles.v));
        let c = storage("c", bytemuck::cast_slice(&particles.c));
        let x2 = empty("x2", 16 * n as u64);
        let v2 = empty("v2", 16 * n as u64);
        let c2 = empty("c2", 48 * n as u64);
        let react = empty("react", 16 * max_subs as u64);
        let nb = [params.n[0].div_ceil(4), params.n[1].div_ceil(4), params.n[2].div_ceil(4)];
        let nblocks = nb[0] * nb[1] * nb[2];
        let gm = empty("gm", 4 * snodes);
        let gmom = empty("gmom", 12 * snodes);
        let gvel = empty("gvel", 16 * snodes);
        let gvold = empty("gvold", 16 * snodes);
        let btab = empty("btab", 4 * nblocks as u64);
        let alist = empty("alist", 4 * max_slots as u64);
        let nact = empty("nact", 16);
        let indirect = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let small_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("small stage"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let counts = empty("counts", 4 * (nblocks + 1) as u64);
        let offsets = empty("offsets", 4 * (nblocks + 1) as u64);
        let fill = empty("fill", 4 * (nblocks + 1) as u64);
        let perm = empty("perm", 4 * n as u64);
        let react_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("react stage"),
            size: 16 * max_subs as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: PARAMS_STRIDE * max_subs as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let rw = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
        let mut entries = vec![entry(0, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: None })];
        for b in 1..19 {
            entries.push(entry(b, rw));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("mpm"), entries: &entries });
        let layout1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("indirect"), entries: &[entry(0, rw)] });
        let bind_indirect = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("indirect"), layout: &layout1, entries: &[wgpu::BindGroupEntry { binding: 0, resource: indirect.as_entire_binding() }] });
        let make_bind = |xin: &wgpu::Buffer, vin: &wgpu::Buffer, cin: &wgpu::Buffer, xout: &wgpu::Buffer, vout: &wgpu::Buffer, cout: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mpm"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &params_buf, offset: 0, size: Some(std::num::NonZeroU64::new(std::mem::size_of::<GpuParams>() as u64).unwrap()) }),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: xin.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: vin.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: cin.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: gm.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: gmom.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: gvel.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: gvold.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: react.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: counts.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 10, resource: offsets.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: fill.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 12, resource: perm.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 13, resource: xout.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 14, resource: vout.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 15, resource: cout.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 16, resource: btab.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 17, resource: alist.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 18, resource: nact.as_entire_binding() },
                ],
            })
        };
        let binds = [make_bind(&x, &v, &c, &x2, &v2, &c2), make_bind(&x2, &v2, &c2, &x, &v, &c)];
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("mpm"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pl_scan = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("mpm blk_scan"), bind_group_layouts: &[Some(&layout), Some(&layout1)], immediate_size: 0 });
        let pipe_with = |name: &str, l: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(l),
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipe = |name: &str| pipe_with(name, &pl);
        Ok(Self {
            clear: pipe("clear"),
            sort_zero: pipe("sort_zero"),
            sort_count: pipe("sort_count"),
            sort_scan: pipe("sort_scan"),
            sort_scatter: pipe("sort_scatter"),
            blk_mark: pipe("blk_mark"),
            blk_scan: pipe_with("blk_scan", &pl_scan),
            bind_indirect,
            permute: pipe("permute"),
            p2g: pipe("p2g_block"),
            grid: pipe("grid"),
            blur: pipe("blur"),
            g2p: pipe("g2p"),
            device,
            queue,
            params,
            n,
            nodes,
            max_subs,
            params_buf,
            bufs: [(x, v, c), (x2, v2, c2)],
            cur: 0,
            binds,
            nblocks,
            nb,
            max_slots,
            btab,
            nact,
            indirect,
            small_stage,
            gm,
            gmom,
            gvel,
            gvold,
            react,
            react_stage,
            time: 0.0,
            active: 0,
            damp: 1.0,
            sponge: 0.0,
            surf: None,
        })
    }

    /// The slot budget: mark the blocks the fill's particles sit in, dilate
    /// them by one block the way `blk_mark` does, and add a sixth for the
    /// splash. Capped at the box, since the dense grid is the worst case.
    fn slot_budget(params: &Params, particles: &Particles, nb: [u32; 3], nblocks: u32) -> u32 {
        let inv_h = 1.0 / params.h;
        let mut seen = vec![false; nblocks as usize];
        for p in &particles.x {
            let idx = |a: f32, o: f32, n: u32| (((a - o) * inv_h) as i32).clamp(0, n as i32 - 1) as u32 / 4;
            let (bx, by, bz) = (idx(p[0], params.origin[0], params.n[0]), idx(p[1], params.origin[1], params.n[1]), idx(p[2], params.origin[2], params.n[2]));
            seen[(((bz * nb[1]) + by) * nb[0] + bx) as usize] = true;
        }
        let mut n = 0u32;
        for bz in 0..nb[2] {
            for by in 0..nb[1] {
                for bx in 0..nb[0] {
                    let mut any = false;
                    for dz in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let (a, b, c) = (bx as i32 + dx, by as i32 + dy, bz as i32 + dz);
                                if a < 0 || b < 0 || c < 0 || a >= nb[0] as i32 || b >= nb[1] as i32 || c >= nb[2] as i32 {
                                    continue;
                                }
                                any |= seen[((c as u32 * nb[1] + b as u32) * nb[0] + a as u32) as usize];
                            }
                        }
                    }
                    n += any as u32;
                }
            }
        }
        let budget = std::env::var("NEWT_MAX_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(n + n / 6 + 1024);
        budget.min(nblocks).max(1)
    }

    /// Active block slots the node arrays are sized for, and the box's total.
    pub fn slots(&self) -> (u32, u32) {
        (self.max_slots, self.nblocks)
    }

    /// Width of the damping band along the side walls (0 = none).
    pub fn set_sponge(&mut self, w: f32) {
        self.sponge = w;
    }

    /// Per-substep velocity factor (1 = none); used while settling.
    pub fn set_damp(&mut self, damp: f32) {
        self.damp = damp;
    }

    pub fn count(&self) -> usize {
        self.n as usize
    }

    fn gpu_params(&self, body: &Body, slot: u32) -> GpuParams {
        let p = &self.params;
        let h = p.h;
        let n = p.n;
        let o = p.origin;
        let e = 2.0 * h; // the wall plane, see the CPU solver
        let a = norm(body.axis);
        let b = norm(cross([0.0, 0.0, 1.0], a));
        let cc = cross(a, b);
        GpuParams {
            origin_h: v4(o, h),
            n: [n[0], n[1], n[2], self.n],
            k: [p.dt, 1.0 / h, p.mass, p.vol0],
            k2: [p.bulk, p.flip, p.gravity, 4.0 / (h * h)],
            lo: [o[0] + 2.0 * h, o[1] + 2.0 * h, o[2] + 2.0 * h, e],
            hi: [o[0] + (n[0] - 3) as f32 * h, o[1] + (n[1] - 3) as f32 * h, o[2] + (n[2] - 3) as f32 * h, self.damp],
            misc: [slot, self.nb[0], self.nb[1], self.nb[2]],
            xmax: [o[0] + (n[0] - 1) as f32 * h - e, o[1] + (n[1] - 1) as f32 * h - e, o[2] + (n[2] - 1) as f32 * h - e, p.j_relax],
            b_centre: v4(body.centre, 0.0),
            b_a: v4(a, 0.0),
            b_b: v4(b, 0.0),
            b_c: v4(cc, 0.0),
            b_vel: v4(body.vel, 0.0),
            semi: v4(body.semi, self.sponge),
            misc2: [self.max_slots, 0, 0, 0],
        }
    }

    fn groups(count: u32) -> (u32, u32) {
        let g = count.div_ceil(WG);
        if g <= 65535 { (g, 1) } else { (65535, g.div_ceil(65535)) }
    }

    /// Run `subs` substeps against a fixed body pose. Returns one reaction
    /// per substep.
    pub fn step(&mut self, body: &Body, subs: u32) -> Vec<Reaction> {
        assert!(subs <= self.max_subs, "block of {subs} > {}", self.max_subs);
        let mut raw = vec![0u8; (PARAMS_STRIDE * subs as u64) as usize];
        for s in 0..subs {
            let p = self.gpu_params(body, s);
            let off = (s as u64 * PARAMS_STRIDE) as usize;
            raw[off..off + std::mem::size_of::<GpuParams>()].copy_from_slice(bytemuck::bytes_of(&p));
        }
        self.queue.write_buffer(&self.params_buf, 0, &raw);
        self.queue.write_buffer(&self.react, 0, &vec![0u8; 16 * subs as usize]);
        self.queue.write_buffer(&self.nact, 0, &[0u8; 16]);
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let (pg, pgy) = Self::groups(self.n);
            let (bg, bgy) = Self::groups(self.nblocks);
            let (wg, wgy) = if self.nblocks <= 65535 { (self.nblocks, 1) } else { (65535, self.nblocks.div_ceil(65535)) };
            for s in 0..subs {
                let off = (s as u64 * PARAMS_STRIDE) as u32;
                // sort the particles by block, from bufs[cur] into bufs[1 - cur]
                pass.set_bind_group(0, &self.binds[self.cur], &[off]);
                pass.set_pipeline(&self.sort_zero);
                pass.dispatch_workgroups(bg, bgy, 1);
                pass.set_pipeline(&self.sort_count);
                pass.dispatch_workgroups(pg, pgy, 1);
                pass.set_pipeline(&self.sort_scan);
                pass.dispatch_workgroups(1, 1, 1);
                // the block counts also say which blocks the grid needs
                pass.set_pipeline(&self.blk_mark);
                pass.dispatch_workgroups(bg, bgy, 1);
                pass.set_pipeline(&self.blk_scan);
                pass.set_bind_group(1, &self.bind_indirect, &[]);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(&self.sort_scatter);
                pass.dispatch_workgroups(pg, pgy, 1);
                pass.set_pipeline(&self.permute);
                pass.dispatch_workgroups(pg, pgy, 1);
                // the physics, on the sorted buffers
                self.cur = 1 - self.cur;
                pass.set_bind_group(0, &self.binds[self.cur], &[off]);
                // the node kernels run over the active slots only; how many
                // that is is a GPU-side number, so the dispatch is indirect
                pass.set_pipeline(&self.clear);
                pass.dispatch_workgroups_indirect(&self.indirect, 0);
                pass.set_pipeline(&self.p2g);
                pass.dispatch_workgroups(wg, wgy, 1);
                pass.set_pipeline(&self.grid);
                pass.dispatch_workgroups_indirect(&self.indirect, 0);
                pass.set_pipeline(&self.blur);
                pass.dispatch_workgroups_indirect(&self.indirect, 0);
                pass.set_pipeline(&self.g2p);
                pass.dispatch_workgroups(pg, pgy, 1);
            }
        }
        enc.copy_buffer_to_buffer(&self.react, 0, &self.react_stage, 0, 16 * subs as u64);
        enc.copy_buffer_to_buffer(&self.nact, 0, &self.small_stage, 0, 16);
        self.queue.submit([enc.finish()]);
        let words: Vec<i32> = self.read_i32(&self.react_stage, 4 * subs as usize);
        let a = self.read_i32(&self.small_stage, 4);
        self.active = a[0] as u32;
        assert!(
            a[1] as u32 <= self.max_slots,
            "block-sparse grid overflowed: {} active blocks of {} budgeted ({} in the box). Raise NEWT_MAX_BLOCKS.",
            a[1],
            self.max_slots,
            self.nblocks
        );
        self.time += subs as f64 * self.params.dt as f64;
        (0..subs as usize)
            .map(|s| Reaction {
                impulse: [words[4 * s] as f64 / 16384.0, words[4 * s + 1] as f64 / 16384.0, words[4 * s + 2] as f64 / 16384.0],
                interior_mass: words[4 * s + 3] as f64 / 65536.0,
            })
            .collect()
    }

    fn read_i32(&self, stage: &wgpu::Buffer, count: usize) -> Vec<i32> {
        let slice = stage.slice(..(4 * count) as u64);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv().expect("map").expect("map ok");
        let out: Vec<i32> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        stage.unmap();
        out
    }

    fn read_f32(&self, src: &wgpu::Buffer, bytes: u64) -> Vec<f32> {
        let stage = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("stage"), size: bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &stage, 0, bytes);
        self.queue.submit([enc.finish()]);
        let slice = stage.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv().expect("map").expect("map ok");
        let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        stage.unmap();
        out
    }

    /// Particle positions (xyz, J−1) and velocities (xyz, original id), in
    /// the GPU's current (sorted) order.
    pub fn download(&self) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
        let (x, v, _) = &self.bufs[self.cur];
        let x = self.read_f32(x, 16 * self.n as u64);
        let v = self.read_f32(v, 16 * self.n as u64);
        (bytemuck::cast_slice(&x).to_vec(), bytemuck::cast_slice(&v).to_vec())
    }

    /// Active blocks after the last submitted substep, and the nodes they hold.
    pub fn active(&self) -> (u32, u32) {
        (self.active, self.active * 64)
    }

    /// Grid mass per node after the last substep, dense over the box: the
    /// compact slots scattered back through the block table, everything else
    /// zero (which is what an inactive node holds).
    pub fn grid_mass(&self) -> Vec<f32> {
        let g = self.read_f32(&self.gvel, 16 * 64 * self.max_slots as u64);
        let tab = self.read_f32(&self.btab, 4 * self.nblocks as u64);
        let tab: &[u32] = bytemuck::cast_slice(&tab);
        let mut out = vec![0.0f32; self.nodes as usize];
        let (nx, ny, nz) = (self.params.n[0], self.params.n[1], self.params.n[2]);
        for bz in 0..self.nb[2] {
            for by in 0..self.nb[1] {
                for bx in 0..self.nb[0] {
                    let slot = tab[((bz * self.nb[1] + by) * self.nb[0] + bx) as usize];
                    if slot == u32::MAX {
                        continue;
                    }
                    for l in 0..64u32 {
                        let (i, j, k) = (4 * bx + (l & 3), 4 * by + ((l >> 2) & 3), 4 * bz + ((l >> 4) & 3));
                        if i >= nx || j >= ny || k >= nz {
                            continue;
                        }
                        out[(((k * ny) + j) * nx + i) as usize] = g[(4 * (slot * 64 + l) + 3) as usize];
                    }
                }
            }
        }
        out
    }

}
