//! MLS-MPM water on the GPU.
//!
//! The four kernels in `mpm.wgsl` are the CPU solver's four loops. A block
//! of substeps is one command submission; the only thing that comes back per
//! block is the body's reaction force (one 16-byte slot per substep, so the
//! fixed-point accumulators cannot overflow across a block). Particles and
//! grid mass are downloaded once a frame for surface extraction.

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
    damp: f32,
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
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no device: {e}"))?;
        let n = particles.x.len() as u32;
        let nodes = params.n[0] * params.n[1] * params.n[2];
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
        let gm = empty("gm", 4 * nodes as u64);
        let gmom = empty("gmom", 12 * nodes as u64);
        let gvel = empty("gvel", 16 * nodes as u64);
        let gvold = empty("gvold", 16 * nodes as u64);
        let react = empty("react", 16 * max_subs as u64);
        let nb = [params.n[0].div_ceil(4), params.n[1].div_ceil(4), params.n[2].div_ceil(4)];
        let nblocks = nb[0] * nb[1] * nb[2];
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
        for b in 1..16 {
            entries.push(entry(b, rw));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("mpm"), entries: &entries });
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
                ],
            })
        };
        let binds = [make_bind(&x, &v, &c, &x2, &v2, &c2), make_bind(&x2, &v2, &c2, &x, &v, &c)];
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("mpm"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipe = |name: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Self {
            clear: pipe("clear"),
            sort_zero: pipe("sort_zero"),
            sort_count: pipe("sort_count"),
            sort_scan: pipe("sort_scan"),
            sort_scatter: pipe("sort_scatter"),
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
            gm,
            gmom,
            gvel,
            gvold,
            react,
            react_stage,
            time: 0.0,
            damp: 1.0,
        })
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
            semi: v4(body.semi, 0.0),
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
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let (pg, pgy) = Self::groups(self.n);
            let (ng, ngy) = Self::groups(self.nodes);
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
                pass.set_pipeline(&self.sort_scatter);
                pass.dispatch_workgroups(pg, pgy, 1);
                pass.set_pipeline(&self.permute);
                pass.dispatch_workgroups(pg, pgy, 1);
                // the physics, on the sorted buffers
                self.cur = 1 - self.cur;
                pass.set_bind_group(0, &self.binds[self.cur], &[off]);
                pass.set_pipeline(&self.clear);
                pass.dispatch_workgroups(ng, ngy, 1);
                pass.set_pipeline(&self.p2g);
                pass.dispatch_workgroups(wg, wgy, 1);
                pass.set_pipeline(&self.grid);
                pass.dispatch_workgroups(ng, ngy, 1);
                pass.set_pipeline(&self.blur);
                pass.dispatch_workgroups(ng, ngy, 1);
                pass.set_pipeline(&self.g2p);
                pass.dispatch_workgroups(pg, pgy, 1);
            }
        }
        enc.copy_buffer_to_buffer(&self.react, 0, &self.react_stage, 0, 16 * subs as u64);
        self.queue.submit([enc.finish()]);
        let words: Vec<i32> = self.read_i32(&self.react_stage, 4 * subs as usize);
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

    /// Grid mass per node after the last substep.
    pub fn grid_mass(&self) -> Vec<f32> {
        let g = self.read_f32(&self.gvel, 16 * self.nodes as u64);
        g.chunks(4).map(|c| c[3]).collect()
    }

}
