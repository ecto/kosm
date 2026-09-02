//! The live tier: the pool rasterized on the GPU through egui's wgpu backend.
//!
//! An offscreen pass with a depth buffer draws deck, walls, floor, the water
//! surface as a height-field mesh, the melon as an ellipsoid, and the beads
//! as instances; the panel then blits that frame. The water fragment shader
//! traces the refracted ray analytically against the floor, the walls and the
//! melon and samples the caustic map, the same construction the reference
//! tracer uses, so the two can be compared pixel for pixel.

use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu::{wgpu, CallbackResources, CallbackTrait, ScreenDescriptor};
use newt_spike::pool::{self, box_half, Caustic, Surface, DEPTH, MELON_AXES, POOL_X, POOL_Y};
use newt_spike::splash::Droplet;
use tang::Vec3 as V;

const COPING: f32 = 0.06;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    nrm: [f32; 3],
    aux: [f32; 3],
    mat: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    m: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    sun: [f32; 4],
    pool: [f32; 4],
    melon_centre: [f32; 4],
    melon_axis: [f32; 4],
    melon_semi: [f32; 4],
    caustic: [f32; 4],
}

/// Camera shared by the live and reference tiers.
#[derive(Clone, Copy)]
pub struct Camera {
    pub eye: V<f64>,
    pub target: V<f64>,
    pub vfov: f64,
}

impl Camera {
    pub fn view(&self, width: u32, height: u32) -> pool::View {
        pool::View { eye: self.eye, target: self.target, width, height, vfov: self.vfov }
    }
    fn view_proj(&self, aspect: f32) -> [[f32; 4]; 4] {
        let f = (self.target - self.eye).normalize();
        let r = f.cross(&V::new(0.0, 0.0, 1.0)).normalize();
        let u = r.cross(&f);
        let e = self.eye;
        // view: rows r, u, -f (right-handed, looking down -z in view space)
        let view = [
            [r.x, u.x, -f.x, 0.0],
            [r.y, u.y, -f.y, 0.0],
            [r.z, u.z, -f.z, 0.0],
            [-r.dot(&e), -u.dot(&e), f.dot(&e), 1.0],
        ];
        let (near, far) = (0.05f64, 60.0f64);
        let t = 1.0 / (0.5 * self.vfov).tan();
        // wgpu clip z in [0, 1]
        let proj = [
            [t / aspect as f64, 0.0, 0.0, 0.0],
            [0.0, t, 0.0, 0.0],
            [0.0, 0.0, far / (near - far), -1.0],
            [0.0, 0.0, near * far / (near - far), 0.0],
        ];
        // column-major product proj * view
        let mut out = [[0.0f32; 4]; 4];
        for c in 0..4 {
            for rr in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += proj[k][rr] * view[c][k];
                }
                out[c][rr] = s as f32;
            }
        }
        out
    }
}

/// What one frame needs from the recording.
pub struct LiveFrame {
    pub surface: Surface,
    pub caustic: Caustic,
    pub melon_centre: V<f64>,
    pub melon_axis: V<f64>,
    pub droplets: Vec<Droplet>,
}

/// GPU resources, kept in egui's callback resources.
pub struct Resources {
    scene_pipeline: wgpu::RenderPipeline,
    bead_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    scene_bind: wgpu::BindGroup,
    blit_layout: wgpu::BindGroupLayout,
    blit_bind: Option<wgpu::BindGroup>,
    sampler: wgpu::Sampler,
    caustic_tex: wgpu::Texture,
    caustic_dims: (u32, u32),
    static_vb: wgpu::Buffer,
    static_n: u32,
    water_vb: wgpu::Buffer,
    water_ib: wgpu::Buffer,
    water_n: u32,
    melon_vb: wgpu::Buffer,
    melon_ib: wgpu::Buffer,
    melon_n: u32,
    bead_vb: wgpu::Buffer,
    bead_ib: wgpu::Buffer,
    bead_n: u32,
    bead_instances: wgpu::Buffer,
    bead_count: u32,
    color: Option<(wgpu::Texture, wgpu::TextureView, wgpu::TextureView, u32, u32)>,
    /// Water grid resolution.
    wn: (usize, usize),
}

const WATER_NX: usize = 120;
const WATER_NY: usize = 80;
const SPHERE_SEG: usize = 24;

impl Resources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scene.wgsl").into()),
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let caustic_dims = ((2.0 * (box_half() + 1.2) / 0.01) as u32, (2.0 * (box_half() + 1.2) / 0.01) as u32);
        let caustic_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("caustic"),
            size: wgpu::Extent3d { width: caustic_dims.0, height: caustic_dims.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let scene_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene"),
            layout: &scene_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&caustic_tex.create_view(&Default::default())) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&nearest) },
            ],
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 36, shader_location: 3 },
            ],
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 4 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 5 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 6 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 7 },
            ],
        };
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            bind_group_layouts: &[Some(&scene_layout)],
            immediate_size: 0,
        });
        let offscreen_format = wgpu::TextureFormat::Rgba8Unorm;
        let depth = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        });
        let make = |label: &str, vs: &str, buffers: &[Option<wgpu::VertexBufferLayout>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some(vs), buffers, compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState { format: offscreen_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: depth.clone(),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let scene_pipeline = make("scene", "vs_main", &[Some(vertex_layout.clone())]);
        let bead_pipeline = make("beads", "vs_bead", &[Some(vertex_layout.clone()), Some(instance_layout)]);

        // the blit
        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("blit"), bind_group_layouts: &[Some(&blit_layout)], immediate_size: 0 });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&blit_pl),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_blit"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_blit"),
                targets: &[Some(wgpu::ColorTargetState { format: target_format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // static geometry
        let static_verts = static_geometry();
        let static_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("static"), contents: bytemuck::cast_slice(&static_verts), usage: wgpu::BufferUsages::VERTEX });
        // water grid
        // two water meshes: the far field over the pool, the fine box over the splash
        let (mut wi, n1) = grid_indices(WATER_NX, WATER_NY);
        let (wi2, _) = grid_indices(WATER_NX, WATER_NY);
        wi.extend(wi2.iter().map(|i| i + (WATER_NX * WATER_NY) as u32));
        let water_n = 2 * n1;
        let water_vb = device.create_buffer(&wgpu::BufferDescriptor { label: Some("water"), size: (2 * WATER_NX * WATER_NY * std::mem::size_of::<Vertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let water_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("water idx"), contents: bytemuck::cast_slice(&wi), usage: wgpu::BufferUsages::INDEX });
        // spheres: melon (per-frame vertices) and bead (unit, instanced)
        let (sv, si) = unit_sphere(SPHERE_SEG, 3);
        let melon_vb = device.create_buffer(&wgpu::BufferDescriptor { label: Some("melon"), size: (sv.len() * std::mem::size_of::<Vertex>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let melon_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("melon idx"), contents: bytemuck::cast_slice(&si), usage: wgpu::BufferUsages::INDEX });
        let (bv, bi) = unit_sphere(10, 4);
        let bead_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("bead"), contents: bytemuck::cast_slice(&bv), usage: wgpu::BufferUsages::VERTEX });
        let bead_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("bead idx"), contents: bytemuck::cast_slice(&bi), usage: wgpu::BufferUsages::INDEX });
        let bead_instances = device.create_buffer(&wgpu::BufferDescriptor { label: Some("bead inst"), size: (512 * std::mem::size_of::<Instance>()) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        Self {
            scene_pipeline,
            bead_pipeline,
            blit_pipeline,
            uniforms,
            scene_bind,
            blit_layout,
            blit_bind: None,
            sampler,
            caustic_tex,
            caustic_dims,
            static_vb,
            static_n: static_verts.len() as u32,
            water_vb,
            water_ib,
            water_n,
            melon_vb,
            melon_ib,
            melon_n: si.len() as u32,
            bead_vb,
            bead_ib,
            bead_n: bi.len() as u32,
            bead_instances,
            bead_count: 0,
            color: None,
            wn: (WATER_NX, WATER_NY),
        }
    }

    fn ensure_target(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if let Some((_, _, _, cw, ch)) = &self.color {
            if *cw == w && *ch == h {
                return;
            }
        }
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("live color"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("live depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let cv = color.create_view(&Default::default());
        let dv = depth.create_view(&Default::default());
        self.blit_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&cv) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        }));
        self.color = Some((color, cv, dv, w, h));
    }
}

/// The paint callback: one per painted frame.
pub struct LiveCallback {
    pub frame: Arc<LiveFrame>,
    pub camera: Camera,
    pub size: (u32, u32),
    pub time: f32,
    /// Write the offscreen frame to this path after drawing it.
    pub shot: Option<std::path::PathBuf>,
}

/// Read the live frame back and save it: the export path, and how the
/// live tier is verified against the reference without a screen capture.
pub fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, (w, h): (u32, u32)) -> image::RgbaImage {
    let row = ((4 * w + 255) / 256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    rx.recv().expect("map").expect("map ok");
    let data = slice.get_mapped_range().expect("mapped");
    let mut px = Vec::with_capacity((4 * w * h) as usize);
    for y in 0..h {
        px.extend_from_slice(&data[(y * row) as usize..(y * row + 4 * w) as usize]);
    }
    drop(data);
    buf.unmap();
    image::RgbaImage::from_raw(w, h, px).expect("image")
}

impl CallbackTrait for LiveCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // drive completed readbacks (egui screenshots) since nothing else polls
        let _ = device.poll(wgpu::PollType::Poll);
        let res: &mut Resources = resources.get_mut().expect("live resources");
        let (w, h) = (self.size.0.max(8), self.size.1.max(8));
        res.ensure_target(device, w, h);
        let f = &self.frame;

        // uniforms
        let sun = V::new(-0.35, -0.45, 0.82).normalize();
        let u = Uniforms {
            view_proj: self.camera.view_proj(w as f32 / h as f32),
            eye: [self.camera.eye.x as f32, self.camera.eye.y as f32, self.camera.eye.z as f32, self.time],
            sun: [sun.x as f32, sun.y as f32, sun.z as f32, 1.05],
            pool: [POOL_X as f32, POOL_Y as f32, DEPTH as f32, COPING],
            melon_centre: [f.melon_centre.x as f32, f.melon_centre.y as f32, f.melon_centre.z as f32, 0.0],
            melon_axis: [f.melon_axis.x as f32, f.melon_axis.y as f32, f.melon_axis.z as f32, 0.0],
            melon_semi: [MELON_AXES[0] as f32, MELON_AXES[1] as f32, MELON_AXES[2] as f32, 0.0],
            caustic: [f.caustic.origin[0] as f32, f.caustic.origin[1] as f32, f.caustic.cell as f32, 0.0],
        };
        queue.write_buffer(&res.uniforms, 0, bytemuck::bytes_of(&u));

        // caustic map
        let (cw, ch) = res.caustic_dims;
        if f.caustic.nx as u32 == cw && f.caustic.ny as u32 == ch {
            let data: Vec<f32> = f.caustic.e.iter().map(|v| *v as f32).collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &res.caustic_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                bytemuck::cast_slice(&data),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(cw * 4), rows_per_image: None },
                wgpu::Extent3d { width: cw, height: ch, depth_or_array_layers: 1 },
            );
        }

        // water surface meshes: one height per vertex, normals by central
        // differences of the sampled grid; the far field over the pool and
        // the fine box over the splash (drawn on top: same material, closer)
        let (nx, ny) = res.wn;
        let mut wv = Vec::with_capacity(2 * nx * ny);
        let bh = box_half();
        for (hx, hy, lift) in [(POOL_X, POOL_Y, 0.0), (bh, bh, 0.0005)] {
            let xs: Vec<f64> = (0..nx).map(|i| -hx + (i as f64 / (nx - 1) as f64) * 2.0 * hx).collect();
            let ys: Vec<f64> = (0..ny).map(|j| -hy + (j as f64 / (ny - 1) as f64) * 2.0 * hy).collect();
            let hs: Vec<f64> = ys.iter().flat_map(|&y| xs.iter().map(move |&x| (x, y))).map(|(x, y)| f.surface.height(x, y)).collect();
            for j in 0..ny {
                for i in 0..nx {
                    let h = |ii: usize, jj: usize| hs[jj * nx + ii];
                    let (i0, i1) = (i.saturating_sub(1), (i + 1).min(nx - 1));
                    let (j0, j1) = (j.saturating_sub(1), (j + 1).min(ny - 1));
                    let dx = (h(i1, j) - h(i0, j)) / (xs[i1] - xs[i0]);
                    let dy = (h(i, j1) - h(i, j0)) / (ys[j1] - ys[j0]);
                    let n = V::new(-dx, -dy, 1.0).normalize();
                    wv.push(Vertex { pos: [xs[i] as f32, ys[j] as f32, (h(i, j) + lift) as f32], nrm: [n.x as f32, n.y as f32, n.z as f32], aux: [0.0; 3], mat: 2 });
                }
            }
        }
        queue.write_buffer(&res.water_vb, 0, bytemuck::cast_slice(&wv));

        // melon: unit sphere placed
        let (sv, _) = unit_sphere(SPHERE_SEG, 3);
        let a = f.melon_axis;
        let b = V::new(0.0, 0.0, 1.0).cross(&a).normalize();
        let c = a.cross(&b);
        let mv: Vec<Vertex> = sv
            .iter()
            .map(|v| {
                let l = V::new(v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64);
                let p = f.melon_centre + a * (l.x * MELON_AXES[0]) + b * (l.y * MELON_AXES[1]) + c * (l.z * MELON_AXES[2]);
                let nl = V::new(l.x / MELON_AXES[0], l.y / MELON_AXES[1], l.z / MELON_AXES[2]);
                let n = (a * nl.x + b * nl.y + c * nl.z).normalize();
                Vertex { pos: [p.x as f32, p.y as f32, p.z as f32], nrm: [n.x as f32, n.y as f32, n.z as f32], aux: v.pos, mat: 3 }
            })
            .collect();
        queue.write_buffer(&res.melon_vb, 0, bytemuck::cast_slice(&mv));

        // beads
        let inst: Vec<Instance> = f
            .droplets
            .iter()
            .take(512)
            .map(|d| {
                let r = 0.006 + 0.010 * d.crowd;
                let speed = d.vel.norm();
                let axis = if speed > 0.2 { d.vel / speed } else { V::new(0.0, 0.0, 1.0) };
                let stretch = 1.0 + (speed * 0.25).min(2.0);
                let u = if axis.x.abs() < 0.9 { V::new(1.0, 0.0, 0.0) } else { V::new(0.0, 1.0, 0.0) };
                let e1 = axis.cross(&u).normalize();
                let e2 = axis.cross(&e1);
                let (ax, e1, e2) = (axis * (r * stretch), e1 * r, e2 * r);
                Instance {
                    m: [
                        [ax.x as f32, ax.y as f32, ax.z as f32, 0.0],
                        [e1.x as f32, e1.y as f32, e1.z as f32, 0.0],
                        [e2.x as f32, e2.y as f32, e2.z as f32, 0.0],
                        [d.pos.x as f32, d.pos.y as f32, d.pos.z as f32, 1.0],
                    ],
                }
            })
            .collect();
        res.bead_count = inst.len() as u32;
        if !inst.is_empty() {
            queue.write_buffer(&res.bead_instances, 0, bytemuck::cast_slice(&inst));
        }

        // the offscreen pass
        let (_, cv, dv, _, _) = res.color.as_ref().unwrap();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("live") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("live"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: cv,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.62, g: 0.75, b: 0.9, a: 1.0 }), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: dv,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&res.scene_pipeline);
            pass.set_bind_group(0, &res.scene_bind, &[]);
            pass.set_vertex_buffer(0, res.static_vb.slice(..));
            pass.draw(0..res.static_n, 0..1);
            pass.set_vertex_buffer(0, res.melon_vb.slice(..));
            pass.set_index_buffer(res.melon_ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..res.melon_n, 0, 0..1);
            pass.set_vertex_buffer(0, res.water_vb.slice(..));
            pass.set_index_buffer(res.water_ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..res.water_n, 0, 0..1);
            if res.bead_count > 0 {
                pass.set_pipeline(&res.bead_pipeline);
                pass.set_bind_group(0, &res.scene_bind, &[]);
                pass.set_vertex_buffer(0, res.bead_vb.slice(..));
                pass.set_vertex_buffer(1, res.bead_instances.slice(..));
                pass.set_index_buffer(res.bead_ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..res.bead_n, 0, 0..res.bead_count);
            }
        }
        let cmd = enc.finish();
        if let Some(path) = &self.shot {
            queue.submit([cmd]);
            let (tex, _, _, w, h) = res.color.as_ref().expect("target");
            let img = read_back(device, queue, tex, (*w, *h));
            img.save(path).expect("save shot");
            eprintln!("shot {}", path.display());
            return vec![];
        }
        vec![cmd]
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, resources: &CallbackResources) {
        let res: &Resources = resources.get().expect("live resources");
        if let Some(bind) = &res.blit_bind {
            pass.set_pipeline(&res.blit_pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..4, 0..1);
        }
    }
}

// ---- geometry ---------------------------------------------------------------

fn quad(out: &mut Vec<Vertex>, a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3], n: [f32; 3], mat: u32) {
    let v = |p: [f32; 3]| Vertex { pos: p, nrm: n, aux: [0.0; 3], mat };
    out.extend_from_slice(&[v(a), v(b), v(c), v(a), v(c), v(d)]);
}

fn static_geometry() -> Vec<Vertex> {
    let (hx, hy, dp) = (POOL_X as f32, POOL_Y as f32, DEPTH as f32);
    let big = hx + 20.0;
    let mut v = Vec::new();
    let up = [0.0, 0.0, 1.0];
    // the deck, four slabs around the water at coping height
    quad(&mut v, [-big, hy, COPING], [big, hy, COPING], [big, big, COPING], [-big, big, COPING], up, 0);
    quad(&mut v, [-big, -big, COPING], [big, -big, COPING], [big, -hy, COPING], [-big, -hy, COPING], up, 0);
    quad(&mut v, [-big, -hy, COPING], [-hx, -hy, COPING], [-hx, hy, COPING], [-big, hy, COPING], up, 0);
    quad(&mut v, [hx, -hy, COPING], [big, -hy, COPING], [big, hy, COPING], [hx, hy, COPING], up, 0);
    // inner walls, tiled, from the coping to the floor
    quad(&mut v, [-hx, -hy, -dp], [-hx, hy, -dp], [-hx, hy, COPING], [-hx, -hy, COPING], [1.0, 0.0, 0.0], 1);
    quad(&mut v, [hx, hy, -dp], [hx, -hy, -dp], [hx, -hy, COPING], [hx, hy, COPING], [-1.0, 0.0, 0.0], 1);
    quad(&mut v, [hx, -hy, -dp], [-hx, -hy, -dp], [-hx, -hy, COPING], [hx, -hy, COPING], [0.0, 1.0, 0.0], 1);
    quad(&mut v, [-hx, hy, -dp], [hx, hy, -dp], [hx, hy, COPING], [-hx, hy, COPING], [0.0, -1.0, 0.0], 1);
    // the floor
    quad(&mut v, [-hx, -hy, -dp], [hx, -hy, -dp], [hx, hy, -dp], [-hx, hy, -dp], up, 1);
    v
}

fn grid_indices(nx: usize, ny: usize) -> (Vec<u32>, u32) {
    let mut idx = Vec::with_capacity((nx - 1) * (ny - 1) * 6);
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let a = (j * nx + i) as u32;
            let b = a + 1;
            let c = a + nx as u32;
            let d = c + 1;
            idx.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
    let n = idx.len() as u32;
    (idx, n)
}

fn unit_sphere(seg: usize, mat: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut v = Vec::new();
    let rings = seg;
    for j in 0..=rings {
        let th = std::f32::consts::PI * j as f32 / rings as f32;
        for i in 0..=seg {
            let ph = std::f32::consts::TAU * i as f32 / seg as f32;
            let p = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
            v.push(Vertex { pos: p, nrm: p, aux: p, mat });
        }
    }
    let mut idx = Vec::new();
    let w = seg + 1;
    for j in 0..rings {
        for i in 0..seg {
            let a = (j * w + i) as u32;
            let b = a + 1;
            let c = a + w as u32;
            let d = c + 1;
            idx.extend_from_slice(&[a, c, d, a, d, b]);
        }
    }
    (v, idx)
}

pub use wgpu::util::DeviceExt as _;
