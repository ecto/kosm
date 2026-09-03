//! Host side of `caustic.wgsl`: sunlight onto the tiles, on the GPU.
//!
//! This owns its own device rather than the solver's, so the caustic can be
//! traced whether the water came from the GPU solver, the CPU mirror, or the
//! ring model. All it needs is two height grids and the sun.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};

/// A height field, as the renderer's `HeightGrid` but in f32.
pub struct Grid<'a> {
    pub origin: [f32; 2],
    pub cell: f32,
    pub nx: u32,
    pub ny: u32,
    pub z: &'a [f32],
}

/// Everything about the shot that is not a height field.
#[derive(Clone, Copy, Debug)]
pub struct CausticCfg {
    /// Floor map resolution and half-extent, and the extra margin the launch
    /// lattice reaches beyond it.
    pub cell: f32,
    pub half: f32,
    pub margin: f32,
    /// Rays per floor cell per axis.
    pub sub: u32,
    pub depth: f32,
    pub n_water: f32,
    /// The direction the light travels (down from the sun).
    pub dir: [f32; 3],
    pub t: f32,
    /// The fine box's half-width, its sponge, and the blend into the far field.
    pub box_half: f32,
    pub sponge: f32,
    pub blend: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GpuCfg {
    map: [u32; 4],
    geom: [f32; 4],
    launch: [f32; 4],
    sun: [f32; 4],
    flat: [f32; 4],
    fine: [f32; 4],
    fine_n: [u32; 4],
    far: [f32; 4],
    band: [f32; 4],
}

pub struct GpuCaustic {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    clear: wgpu::ComputePipeline,
    trace: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    cfg: wgpu::Buffer,
    /// Sized to the last call's grids and map; rebuilt when they change.
    bufs: Option<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup, [u32; 3])>,
}

/// Snell through a flat surface: the reference every ray is measured against.
fn flat_refraction(d: [f32; 3], n_water: f32) -> ([f32; 3], f32) {
    let eta = 1.0 / n_water;
    let ci = -d[2];
    let k = 1.0 - eta * eta * (1.0 - ci * ci);
    let ct = k.max(0.0).sqrt();
    let dr = [d[0] * eta, d[1] * eta, d[2] * eta + (eta * ci - ct)];
    let rs = (ci - n_water * ct) / (ci + n_water * ct);
    let rp = (ct - n_water * ci) / (ct + n_water * ci);
    (dr, 1.0 - 0.5 * (rs * rs + rp * rp))
}

impl GpuCaustic {
    pub fn new() -> Result<Self, String> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
            .await
            .map_err(|e| format!("no adapter: {e}"))?;
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor { label: Some("newt-caustic"), required_limits: limits, ..Default::default() })
            .await
            .map_err(|e| format!("no device: {e}"))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("caustic"),
            source: wgpu::ShaderSource::Wgsl(include_str!("caustic.wgsl").into()),
        });
        let ent = |binding: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("caustic"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                ent(1, true),
                ent(2, true),
                ent(3, false),
                ent(4, false),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("caustic"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
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
        let cfg = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("caustic cfg"),
            size: std::mem::size_of::<GpuCfg>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self { clear: pipe("caustic_clear"), trace: pipe("caustic_trace"), resolve: pipe("caustic_resolve"), device, queue, layout, cfg, bufs: None })
    }

    /// Trace one frame. Returns `nx*ny` irradiances, 1 = what a flat surface
    /// would pass, in the same row-major order as the CPU's `Caustic::e`.
    pub fn trace(&mut self, c: &CausticCfg, fine: &Grid, far: &Grid) -> (usize, usize, Vec<f32>) {
        let nx = ((2.0 * c.half) / c.cell) as u32;
        let ny = nx;
        let lx = ((2.0 * c.half + 2.0 * c.margin) / c.cell) as u32;
        let ly = lx;
        let want = [nx * ny, fine.nx * fine.ny, far.nx * far.ny];
        let rebuild = self.bufs.as_ref().map(|b| b.5 != want).unwrap_or(true);
        if rebuild {
            let store = |label: &str, size: u64| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            };
            let fb = store("fine", 4 * want[1].max(1) as u64);
            let rb = store("far", 4 * want[2].max(1) as u64);
            let ab = store("acc", 4 * want[0] as u64);
            let ob = store("out", 4 * want[0] as u64);
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("caustic"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.cfg.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: fb.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: rb.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ab.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: ob.as_entire_binding() },
                ],
            });
            self.bufs = Some((fb, rb, ab, ob, bg, want));
        }
        let (fb, rb, _ab, ob, bg, _) = self.bufs.as_ref().expect("caustic buffers");
        self.queue.write_buffer(fb, 0, bytemuck::cast_slice(fine.z));
        self.queue.write_buffer(rb, 0, bytemuck::cast_slice(far.z));
        let (d_flat, t_flat) = flat_refraction(c.dir, c.n_water);
        let g = GpuCfg {
            map: [nx, ny, lx, ly],
            geom: [-c.half, -c.half, c.cell, c.sub as f32],
            launch: [-c.half - c.margin, -c.half - c.margin, c.depth, c.t],
            sun: [c.dir[0], c.dir[1], c.dir[2], c.n_water],
            flat: [d_flat[0], d_flat[1], d_flat[2], t_flat],
            fine: [fine.origin[0], fine.origin[1], fine.cell, 0.0],
            fine_n: [fine.nx, fine.ny, far.nx, far.ny],
            far: [far.origin[0], far.origin[1], far.cell, 1.0],
            band: [c.box_half, c.sponge, c.blend, 0.0],
        };
        self.queue.write_buffer(&self.cfg, 0, bytemuck::bytes_of(&g));
        let groups = |n: u32| {
            let w = n.div_ceil(256);
            if w <= 65535 { (w, 1) } else { (65535, w.div_ceil(65535)) }
        };
        let (cg, cgy) = groups(nx * ny);
        let (rg, rgy) = groups(lx * c.sub * ly * c.sub);
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, bg, &[]);
            pass.set_pipeline(&self.clear);
            pass.dispatch_workgroups(cg, cgy, 1);
            pass.set_pipeline(&self.trace);
            pass.dispatch_workgroups(rg, rgy, 1);
            pass.set_pipeline(&self.resolve);
            pass.dispatch_workgroups(cg, cgy, 1);
        }
        let bytes = 4 * (nx * ny) as u64;
        let stage = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("caustic stage"), size: bytes, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        enc.copy_buffer_to_buffer(ob, 0, &stage, 0, bytes);
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
        (nx as usize, ny as usize, out)
    }
}
