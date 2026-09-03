//! Host side of `surface.wgsl`: the height field and the candidate picker.
//!
//! Both live on the solver's own device, next to the grid and the particles,
//! so nothing crosses the bus but the answer: a hundred and twenty-five
//! squared floats for the surface, and a few thousand particles for the drops
//! and the foam, instead of the whole pool.

use bytemuck::{Pod, Zeroable};

use crate::GpuMpm;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct SurfParams {
    origin_h: [f32; 4],
    n: [u32; 4],
    map: [u32; 4],
    geom: [f32; 4],
    k: [f32; 4],
    band: [f32; 4],
    blocks: [u32; 4],
}

/// What the surface pass needs beyond the solver's own buffers.
pub(crate) struct Surf {
    pub nx: u32,
    pub ny: u32,
    pub cell: f32,
    pub cap: u32,
    pub has_rest: bool,
    params: wgpu::Buffer,
    rest: wgpu::Buffer,
    ha: wgpu::Buffer,
    cnt: wgpu::Buffer,
    cx: wgpu::Buffer,
    cv: wgpu::Buffer,
    /// [ping/pong] for each of the solver's two particle buffers, since the
    /// per-substep sort leaves the particles in whichever one it last wrote.
    binds: [wgpu::BindGroup; 4],
    blur: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    smooth: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    pick: wgpu::ComputePipeline,
}

/// The drops and the fast near-surface water, as the picker found them.
pub struct Candidates {
    /// xyz and how much water shares the cell (0..1).
    pub x: Vec<[f32; 4]>,
    /// velocity xyz and the particle's original id.
    pub v: Vec<[f32; 4]>,
    /// How many the GPU found, which may be more than were kept.
    pub found: u32,
}

impl GpuMpm {
    fn surf_setup(&mut self, cell: f32, half: f32, cap: u32) {
        let nx = ((2.0 * half) / cell) as u32;
        let ny = nx;
        if let Some(s) = &self.surf {
            if s.nx == nx && s.ny == ny && s.cap == cap {
                return;
            }
        }
        let dev = &self.device;
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("surface"),
            source: wgpu::ShaderSource::Wgsl(include_str!("surface.wgsl").into()),
        });
        let store = |label: &str, size: u64| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let cells = (nx * ny) as u64;
        // per active slot, like the grid itself
        let frac = store("frac", 4 * 64 * self.max_slots as u64);
        let ha = store("ha", 4 * cells);
        let hb = store("hb", 4 * cells);
        let rest = store("rest", 4 * cells);
        let cnt = store("cand count", 4);
        let cx = store("cand x", 16 * cap as u64);
        let cv = store("cand v", 16 * cap as u64);
        let params = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surf params"),
            size: std::mem::size_of::<SurfParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ent = |binding: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }];
        for (b, ro) in [(1u32, true), (2, false), (3, false), (4, false), (5, true), (6, true), (7, true), (8, false), (9, false), (10, false), (11, true), (12, true), (13, true)] {
            entries.push(ent(b, ro));
        }
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("surface"), entries: &entries });
        let mk = |a: &wgpu::Buffer, b: &wgpu::Buffer, px: &wgpu::Buffer, pv: &wgpu::Buffer| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("surface"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.gvel.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: frac.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: rest.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: px.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: pv.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: cnt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: cx.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 10, resource: cv.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: self.btab.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 12, resource: self.alist.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 13, resource: self.nact.as_entire_binding() },
                ],
            })
        };
        let binds = [
            mk(&ha, &hb, &self.bufs[0].0, &self.bufs[0].1),
            mk(&hb, &ha, &self.bufs[0].0, &self.bufs[0].1),
            mk(&ha, &hb, &self.bufs[1].0, &self.bufs[1].1),
            mk(&hb, &ha, &self.bufs[1].0, &self.bufs[1].1),
        ];
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("surface"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipe = |name: &str| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        self.surf = Some(Surf {
            nx,
            ny,
            cell,
            cap,
            has_rest: false,
            params,
            rest,
            ha,
            cnt,
            cx,
            cv,
            binds,
            blur: pipe("surf_blur"),
            scan: pipe("surf_scan"),
            smooth: pipe("surf_smooth"),
            finish: pipe("surf_finish"),
            pick: pipe("pick"),
        });
    }

    /// The rest map, captured once after the settle; subtracted from every
    /// later extraction, exactly as the CPU does it.
    pub fn set_rest(&mut self, rest: &[f32]) {
        let Some(s) = self.surf.as_ref() else { return };
        assert_eq!(rest.len(), (s.nx * s.ny) as usize, "rest map is the wrong size");
        self.queue.write_buffer(&s.rest, 0, bytemuck::cast_slice(rest));
        self.surf.as_mut().expect("surface").has_rest = true;
    }

    /// Extract the free surface from the grid mass. `half` is the box's
    /// half-width, `floor` the height a dry column reports, `level` the flat
    /// offset used before a rest map exists. Comes back as `nx*ny` heights.
    ///
    /// The `pick_*` arguments describe the particles worth keeping while the
    /// height field is still on the GPU: drops more than `pick_above` over the
    /// surface, and water faster than `pick_speed` in the band from
    /// `pick_below` under it to `pick_up` over it. `candidates` reads them.
    #[allow(clippy::too_many_arguments)]
    pub fn surface(&mut self, cell: f32, half: f32, floor: f32, level: f32, pick_above: f32, pick_speed: f32, pick_below: f32, pick_up: f32, cap: u32) -> Vec<f32> {
        self.surf_setup(cell, half, cap);
        let s = self.surf.as_ref().expect("surface");
        let p = SurfParams {
            origin_h: [self.params.origin[0], self.params.origin[1], self.params.origin[2], self.params.h],
            n: [self.params.n[0], self.params.n[1], self.params.n[2], self.n],
            map: [s.nx, s.ny, s.has_rest as u32, s.cap],
            geom: [-half, -half, cell, floor],
            k: [1000.0 * self.params.h.powi(3), level, pick_above, pick_speed],
            band: [pick_below, pick_up, 0.0, 0.0],
            blocks: [self.nb[0], self.nb[1], self.nb[2], self.max_slots],
        };
        self.queue.write_buffer(&s.params, 0, bytemuck::bytes_of(&p));
        self.queue.write_buffer(&s.cnt, 0, &[0u8; 4]);
        let cells = s.nx * s.ny;
        let (cg, cgy) = Self::groups(cells);
        let (ng, ngy) = Self::groups(64 * self.max_slots);
        let (pg, pgy) = Self::groups(self.n);
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            let bg = 2 * self.cur;
            pass.set_bind_group(0, &s.binds[bg], &[]);
            pass.set_pipeline(&s.blur);
            pass.dispatch_workgroups(ng, ngy, 1);
            pass.set_pipeline(&s.scan);
            pass.dispatch_workgroups(cg, cgy, 1);
            // four 3x3 smooths, ping-ponging so the answer lands back in ha
            pass.set_pipeline(&s.smooth);
            for i in 0..4 {
                pass.set_bind_group(0, &s.binds[bg + i % 2], &[]);
                pass.dispatch_workgroups(cg, cgy, 1);
            }
            pass.set_bind_group(0, &s.binds[bg], &[]);
            pass.set_pipeline(&s.finish);
            pass.dispatch_workgroups(cg, cgy, 1);
            pass.set_pipeline(&s.pick);
            pass.dispatch_workgroups(pg, pgy, 1);
        }
        self.queue.submit([enc.finish()]);
        self.read_f32(&s.ha, 4 * cells as u64)
    }

    /// The particles the last `surface` call picked out.
    pub fn candidates(&self) -> Candidates {
        let Some(s) = self.surf.as_ref() else { return Candidates { x: Vec::new(), v: Vec::new(), found: 0 } };
        let found = self.read_f32(&s.cnt, 4);
        let found = bytemuck::cast_slice::<f32, u32>(&found)[0];
        let keep = found.min(s.cap);
        if keep == 0 {
            return Candidates { x: Vec::new(), v: Vec::new(), found };
        }
        let x = self.read_f32(&s.cx, 16 * keep as u64);
        let v = self.read_f32(&s.cv, 16 * keep as u64);
        Candidates { x: bytemuck::cast_slice(&x).to_vec(), v: bytemuck::cast_slice(&v).to_vec(), found }
    }
}
