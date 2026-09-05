//! The ride tier: play back a recorded rollout from a `ride.json`.
//!
//! Nothing is simulated here — the file already holds every pose, so the
//! window is a film projector: play, pause, scrub, and orbit the tracked
//! actor. The renderer is deliberately plain (one vertex buffer per mesh,
//! a model matrix and a colour per instance, lambert against one sun) so
//! that any authored scene — a skatepark STL, a K1's limbs, a deck, wheels —
//! draws without the pool tier's water machinery.
//!
//! See `docs/ride-format.md`. Unknown keys are ignored: the recorder is
//! expected to grow fields the window has not learned yet.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu, CallbackResources, CallbackTrait, ScreenDescriptor};
use serde::Deserialize;
use tang::Vec3 as V;
use wgpu::util::DeviceExt as _;

use crate::live::{read_back, Camera};

// ---- the file ---------------------------------------------------------------

fn one() -> f64 {
    1.0
}
fn quat_id() -> [f64; 4] {
    [1.0, 0.0, 0.0, 0.0]
}
fn grey() -> [f32; 3] {
    [0.7, 0.7, 0.72]
}
fn z_axis() -> [f64; 3] {
    [0.0, 0.0, 1.0]
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum MeshDef {
    Stl {
        path: PathBuf,
        #[serde(default = "one")]
        scale: f64,
    },
    Box {
        half: [f64; 3],
    },
    Sphere {
        radius: f64,
    },
    Cylinder {
        radius: f64,
        half_length: f64,
        #[serde(default = "z_axis")]
        axis: [f64; 3],
    },
}

#[derive(Deserialize)]
struct Fixed {
    mesh: usize,
    #[serde(default)]
    pos: [f64; 3],
    #[serde(default = "quat_id")]
    quat: [f64; 4],
    #[serde(default = "grey")]
    colour: [f32; 3],
    #[serde(default)]
    #[allow(dead_code)]
    label: String,
}

#[derive(Deserialize)]
struct Actor {
    mesh: usize,
    #[serde(default = "grey")]
    colour: [f32; 3],
    #[serde(default)]
    label: String,
    #[serde(default)]
    offset_pos: [f64; 3],
    #[serde(default = "quat_id")]
    offset_quat: [f64; 4],
}

#[derive(Deserialize)]
struct FrameDef {
    #[serde(default)]
    t: f64,
    poses: Vec<([f64; 3], [f64; 4])>,
}

#[derive(Deserialize)]
pub struct Ride {
    #[serde(default)]
    name: String,
    #[serde(default = "one_sixtieth")]
    dt: f64,
    meshes: Vec<MeshDef>,
    #[serde(default)]
    fixed: Vec<Fixed>,
    #[serde(default)]
    actors: Vec<Actor>,
    #[serde(default)]
    track: usize,
    #[serde(default)]
    frames: Vec<FrameDef>,
}

fn one_sixtieth() -> f64 {
    1.0 / 60.0
}

impl Ride {
    /// A live ride before its header has arrived: nothing to draw yet.
    fn empty() -> Self {
        Self { name: String::new(), dt: one_sixtieth(), meshes: Vec::new(), fixed: Vec::new(), actors: Vec::new(), track: 0, frames: Vec::new() }
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let ride: Ride = serde_json::from_str(&text)?;
        if ride.frames.is_empty() {
            anyhow::bail!("{} has no frames", path.display());
        }
        Ok(ride)
    }

    fn duration(&self) -> f64 {
        self.frames.last().map(|f| f.t).unwrap_or(0.0).max(self.dt * self.frames.len().saturating_sub(1) as f64)
    }

    /// Every instance to draw at `frame`, grouped by mesh index.
    fn instances(&self, frame: usize) -> Vec<Vec<Instance>> {
        let mut out: Vec<Vec<Instance>> = vec![Vec::new(); self.meshes.len()];
        for f in &self.fixed {
            if let Some(slot) = out.get_mut(f.mesh) {
                slot.push(Instance::new(rigid(f.pos, f.quat), f.colour));
            }
        }
        for (i, a) in self.actors.iter().enumerate() {
            let Some((p, q)) = self.frames.get(frame).and_then(|f| f.poses.get(i)).copied() else { continue };
            if let Some(slot) = out.get_mut(a.mesh) {
                slot.push(Instance::new(compose(rigid(p, q), rigid(a.offset_pos, a.offset_quat)), a.colour));
            }
        }
        out
    }

    /// Where the camera looks: the tracked actor, or the origin.
    fn tracked(&self, frame: usize) -> V<f64> {
        let Some((p, _)) = self.frames.get(frame).and_then(|f| f.poses.get(self.track)).copied() else {
            return V::new(0.0, 0.0, 0.0);
        };
        V::new(p[0], p[1], p[2])
    }
}

// ---- rigid transforms -------------------------------------------------------

/// A column-major 4×4 from a position and a `[w, x, y, z]` quaternion.
fn rigid(pos: [f64; 3], q: [f64; 4]) -> [[f32; 4]; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt().max(1e-12);
    let (w, x, y, z) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    let m = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w), 0.0],
        [2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w), 0.0],
        [2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y), 0.0],
        [pos[0], pos[1], pos[2], 1.0],
    ];
    m.map(|c| c.map(|v| v as f32))
}

/// `a ∘ b`, both column-major.
fn compose(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    out
}

// ---- geometry ---------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    nrm: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    m: [[f32; 4]; 4],
    colour: [f32; 4],
}

impl Instance {
    fn new(m: [[f32; 4]; 4], colour: [f32; 3]) -> Self {
        Self { m, colour: [colour[0], colour[1], colour[2], 1.0] }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    sun: [f32; 4],
}

/// A triangle soup with one flat normal per face — flat shading is enough,
/// and STL has no shared vertices to smooth across anyway.
fn tri(out: &mut Vec<Vertex>, a: [f32; 3], b: [f32; 3], c: [f32; 3]) {
    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let n = if l > 1e-20 { [n[0] / l, n[1] / l, n[2] / l] } else { [0.0, 0.0, 1.0] };
    out.extend_from_slice(&[Vertex { pos: a, nrm: n }, Vertex { pos: b, nrm: n }, Vertex { pos: c, nrm: n }]);
}

fn tessellate(def: &MeshDef) -> anyhow::Result<Vec<Vertex>> {
    let mut v = Vec::new();
    match def {
        MeshDef::Stl { path, scale } => {
            let s = *scale as f32;
            let soup = ipse_map::stl::read_binary_stl(path)?;
            for t in soup {
                let p = |i: usize| [t[i][0] * s, t[i][1] * s, t[i][2] * s];
                tri(&mut v, p(0), p(1), p(2));
            }
        }
        MeshDef::Box { half } => {
            let (hx, hy, hz) = (half[0] as f32, half[1] as f32, half[2] as f32);
            let c = |i: usize| {
                let s = |b: usize| if i >> b & 1 == 1 { 1.0 } else { -1.0 };
                [s(0) * hx, s(1) * hy, s(2) * hz]
            };
            // the six faces, each as two triangles wound outwards
            for (a, b, cc, d) in [
                (0, 2, 3, 1), // -z … winding is irrelevant, both faces are lit
                (4, 5, 7, 6),
                (0, 1, 5, 4),
                (2, 6, 7, 3),
                (0, 4, 6, 2),
                (1, 3, 7, 5),
            ] {
                tri(&mut v, c(a), c(b), c(cc));
                tri(&mut v, c(a), c(cc), c(d));
            }
        }
        MeshDef::Sphere { radius } => {
            let r = *radius as f32;
            let (seg, rings) = (18usize, 12usize);
            let p = |i: usize, j: usize| {
                let th = std::f32::consts::PI * j as f32 / rings as f32;
                let ph = std::f32::consts::TAU * i as f32 / seg as f32;
                [r * th.sin() * ph.cos(), r * th.sin() * ph.sin(), r * th.cos()]
            };
            for j in 0..rings {
                for i in 0..seg {
                    tri(&mut v, p(i, j), p(i, j + 1), p(i + 1, j + 1));
                    tri(&mut v, p(i, j), p(i + 1, j + 1), p(i + 1, j));
                }
            }
        }
        MeshDef::Cylinder { radius, half_length, axis } => {
            let (r, hl) = (*radius as f32, *half_length as f32);
            // a frame with `axis` as its length direction
            let a = V::new(axis[0], axis[1], axis[2]);
            let a = if a.norm() > 1e-9 { a.normalize() } else { V::new(0.0, 0.0, 1.0) };
            let helper = if a.z.abs() < 0.9 { V::new(0.0, 0.0, 1.0) } else { V::new(1.0, 0.0, 0.0) };
            let e1 = a.cross(&helper).normalize();
            let e2 = a.cross(&e1);
            let seg = 20usize;
            let p = |i: usize, end: f32| {
                let t = std::f32::consts::TAU * i as f32 / seg as f32;
                let q = a * (hl * end) as f64 + e1 * (r * t.cos()) as f64 + e2 * (r * t.sin()) as f64;
                [q.x as f32, q.y as f32, q.z as f32]
            };
            let cap = |end: f32| {
                let q = a * (hl * end) as f64;
                [q.x as f32, q.y as f32, q.z as f32]
            };
            for i in 0..seg {
                tri(&mut v, p(i, -1.0), p(i + 1, -1.0), p(i + 1, 1.0));
                tri(&mut v, p(i, -1.0), p(i + 1, 1.0), p(i, 1.0));
                tri(&mut v, cap(1.0), p(i, 1.0), p(i + 1, 1.0));
                tri(&mut v, cap(-1.0), p(i + 1, -1.0), p(i, -1.0));
            }
        }
    }
    Ok(v)
}

// ---- the GPU tier -----------------------------------------------------------

struct MeshGpu {
    vb: wgpu::Buffer,
    n: u32,
    inst: wgpu::Buffer,
    cap: u32,
}

pub struct Resources {
    pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bind: wgpu::BindGroup,
    blit_layout: wgpu::BindGroupLayout,
    blit_bind: Option<wgpu::BindGroup>,
    sampler: wgpu::Sampler,
    meshes: Vec<MeshGpu>,
    color: Option<(wgpu::Texture, wgpu::TextureView, wgpu::TextureView, u32, u32)>,
}

impl Resources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat, ride: &Ride) -> anyhow::Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ride"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ride.wgsl").into()),
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ride uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ride"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ride"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
            ],
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 3 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 4 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 5 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 64, shader_location: 6 },
            ],
        };
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ride"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ride"),
            layout: Some(&pl),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[Some(vertex_layout), Some(instance_layout)], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8Unorm, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            // authored STLs are inconsistently wound: cull nothing
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ride blit"),
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
        let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ride blit"), bind_group_layouts: &[Some(&blit_layout)], immediate_size: 0 });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ride blit"),
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

        // one vertex buffer per mesh; the instance buffer is sized for the
        // most instances that mesh can ever carry (fixed plus actors)
        let mut meshes = Vec::with_capacity(ride.meshes.len());
        for (i, def) in ride.meshes.iter().enumerate() {
            let verts = tessellate(def)?;
            let cap = (ride.fixed.iter().filter(|f| f.mesh == i).count() + ride.actors.iter().filter(|a| a.mesh == i).count()).max(1) as u32;
            meshes.push(MeshGpu {
                vb: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ride mesh"),
                    contents: bytemuck::cast_slice(if verts.is_empty() { &[Vertex { pos: [0.0; 3], nrm: [0.0, 0.0, 1.0] }][..].as_ref() } else { verts.as_slice() }),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                n: verts.len() as u32,
                inst: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("ride inst"),
                    size: (cap as usize * std::mem::size_of::<Instance>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                cap,
            });
        }

        Ok(Self {
            pipeline,
            blit_pipeline,
            uniforms,
            bind,
            blit_layout,
            blit_bind: None,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            }),
            meshes,
            color: None,
        })
    }

    fn ensure_target(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if let Some((_, _, _, cw, ch)) = &self.color {
            if *cw == w && *ch == h {
                return;
            }
        }
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ride color"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ride depth"),
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
            label: Some("ride blit"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&cv) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        }));
        self.color = Some((color, cv, dv, w, h));
    }
}

/// One painted frame of the recording.
struct RideCallback {
    instances: Arc<Vec<Vec<Instance>>>,
    camera: Camera,
    size: (u32, u32),
    shot: Option<PathBuf>,
}

impl CallbackTrait for RideCallback {
    fn prepare(&self, device: &wgpu::Device, queue: &wgpu::Queue, _s: &ScreenDescriptor, _e: &mut wgpu::CommandEncoder, resources: &mut CallbackResources) -> Vec<wgpu::CommandBuffer> {
        let _ = device.poll(wgpu::PollType::Poll);
        let res: &mut Resources = resources.get_mut().expect("ride resources");
        let (w, h) = (self.size.0.max(8), self.size.1.max(8));
        res.ensure_target(device, w, h);

        let sun = V::new(-0.35, -0.45, 0.82).normalize();
        let u = Uniforms {
            view_proj: self.camera.view_proj(w as f32 / h as f32),
            sun: [sun.x as f32, sun.y as f32, sun.z as f32, 0.85],
        };
        queue.write_buffer(&res.uniforms, 0, bytemuck::bytes_of(&u));
        let mut counts = Vec::with_capacity(res.meshes.len());
        for (m, inst) in res.meshes.iter().zip(self.instances.iter()) {
            let n = (inst.len() as u32).min(m.cap);
            if n > 0 {
                queue.write_buffer(&m.inst, 0, bytemuck::cast_slice(&inst[..n as usize]));
            }
            counts.push(n);
        }

        let (_, cv, dv, _, _) = res.color.as_ref().unwrap();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ride") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ride"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: cv,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.55, g: 0.68, b: 0.85, a: 1.0 }), store: wgpu::StoreOp::Store },
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
            pass.set_pipeline(&res.pipeline);
            pass.set_bind_group(0, &res.bind, &[]);
            for (m, n) in res.meshes.iter().zip(counts) {
                if n == 0 || m.n == 0 {
                    continue;
                }
                pass.set_vertex_buffer(0, m.vb.slice(..));
                pass.set_vertex_buffer(1, m.inst.slice(..));
                pass.draw(0..m.n, 0..n);
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
        let res: &Resources = resources.get().expect("ride resources");
        if let Some(bind) = &res.blit_bind {
            pass.set_pipeline(&res.blit_pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..4, 0..1);
        }
    }
}

// ---- the live child ---------------------------------------------------------

/// Where a live ride comes from: a child process that streams the ride on
/// stdout, one JSON object per line (see "Streaming" in `docs/ride-format.md`).
pub struct LiveOpts {
    /// program and arguments, already split
    pub cmd: Vec<String>,
    /// the child's working directory — the ipse recorder resolves
    /// `objects/skateboard` relative to it, so it must run in its own tree
    pub cwd: PathBuf,
    pub shove: f64,
    pub shove_at: f64,
    pub duration: f64,
    /// flags passed through untouched (`--policy`, `--scenario`)
    pub extra: Vec<String>,
}

impl LiveOpts {
    /// The full argv for one run: the configured command, the transport's
    /// current knob values, then anything passed through.
    fn argv(&self) -> Vec<String> {
        let mut v = self.cmd.clone();
        v.extend(["--shove".to_string(), format!("{:.3}", self.shove)]);
        v.extend(["--shove-at".to_string(), format!("{:.3}", self.shove_at)]);
        v.extend(["--duration".to_string(), format!("{:.3}", self.duration)]);
        v.extend(self.extra.iter().cloned());
        v
    }
}

/// One line off the child's stdout, already parsed.
enum Msg {
    /// the header: the ride with no frames
    Header(Box<Ride>),
    Frame(Box<FrameDef>),
    /// a line that would not parse, or a read error — logged, not fatal
    Bad(String),
    /// stdout closed
    Eof,
}

/// A running child and the channel its frames arrive on.
struct Live {
    opts: LiveOpts,
    child: Option<std::process::Child>,
    rx: std::sync::mpsc::Receiver<Msg>,
    /// set once the child is gone; the inner `None` means it was killed
    ended: Option<Option<i32>>,
    error: Option<String>,
}

impl Live {
    fn spawn(opts: LiveOpts) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut live = Self { opts, child: None, rx, ended: None, error: None };
        live.start(tx);
        live
    }

    /// Kill whatever is running and start a fresh child with the current knobs.
    fn restart(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.rx = rx;
        self.kill();
        self.ended = None;
        self.error = None;
        self.start(tx);
    }

    fn start(&mut self, tx: std::sync::mpsc::Sender<Msg>) {
        let argv = self.opts.argv();
        let mut cmd = std::process::Command::new(&argv[0]);
        cmd.args(&argv[1..])
            .current_dir(&self.opts.cwd)
            .stdout(std::process::Stdio::piped())
            // the child's diagnostics are ours: inherited, never swallowed
            .stderr(std::process::Stdio::inherit());
        eprintln!("live: {} (in {})", argv.join(" "), self.opts.cwd.display());
        match cmd.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take().expect("piped stdout");
                std::thread::spawn(move || read_stream(stdout, tx));
                self.child = Some(child);
            }
            Err(e) => {
                // a missing recorder is a message in the window, not a panic
                self.error = Some(format!("could not run `{}`: {e}", argv.join(" ")));
                self.ended = Some(None);
            }
        }
    }

    fn kill(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }

    /// Reap the child, so the status line can say how it ended.
    fn poll_child(&mut self) {
        if self.ended.is_some() {
            return;
        }
        if let Some(c) = self.child.as_mut() {
            if let Ok(Some(status)) = c.try_wait() {
                self.ended = Some(status.code());
                self.child = None;
            }
        }
    }

    fn running(&self) -> bool {
        self.ended.is_none()
    }
}

impl Drop for Live {
    /// The window owns the child: closing the window ends the rollout.
    fn drop(&mut self) {
        self.kill();
    }
}

/// Parse the child's stdout: the header line first, then one frame per line.
fn read_stream(stdout: std::process::ChildStdout, tx: std::sync::mpsc::Sender<Msg>) {
    use std::io::BufRead as _;
    let mut header = false;
    for line in std::io::BufReader::new(stdout).lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.send(Msg::Bad(format!("read: {e}")));
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let msg = if header {
            match serde_json::from_str::<FrameDef>(&line) {
                Ok(f) => Msg::Frame(Box::new(f)),
                Err(e) => Msg::Bad(format!("frame: {e}")),
            }
        } else {
            match serde_json::from_str::<Ride>(&line) {
                Ok(r) => {
                    header = true;
                    Msg::Header(Box::new(r))
                }
                Err(e) => Msg::Bad(format!("header: {e}")),
            }
        };
        if tx.send(msg).is_err() {
            return;
        }
    }
    let _ = tx.send(Msg::Eof);
}

// ---- the window -------------------------------------------------------------

struct RideApp {
    ride: Ride,
    /// `None` for a ride read from a file; `Some` while a child streams one
    live: Option<Live>,
    /// keep playing out to the newest frame as frames arrive
    follow: bool,
    /// the GPU buffers are built from the ride's meshes, which in live mode
    /// only exist once the header has arrived — so they are made lazily
    ready: bool,
    /// something to show in the window instead of panicking
    error: Option<String>,
    cursor: usize,
    playing: bool,
    speed: f64,
    accum: f64,
    camera: Camera,
    orbit: (f64, f64, f64),
    /// `--shot=<path> --frame=<n>`: draw one frame offscreen and quit.
    shot: Option<PathBuf>,
    shot_frame: usize,
    shot_now: Option<PathBuf>,
    ticks: u32,
    last: std::time::Instant,
}

impl RideApp {
    fn new(cc: &eframe::CreationContext<'_>, ride: Ride, live: Option<Live>, frame: usize, shot: Option<PathBuf>) -> anyhow::Result<Self> {
        let cursor = frame.min(ride.frames.len().saturating_sub(1));
        let mut app = Self {
            ride,
            live,
            follow: true,
            ready: false,
            error: None,
            cursor,
            playing: shot.is_none(),
            speed: 1.0,
            accum: 0.0,
            camera: Camera { eye: V::new(0.0, -2.0, 1.0), target: V::new(0.0, 0.0, 0.0), vfov: 0.9 },
            // 25° up, 2.2 m back, looking along +x at the actor
            // `--orbit=az_deg,el_deg,dist_m` frames a shot without a hand on the mouse.
            orbit: std::env::args()
                .find_map(|a| a.strip_prefix("--orbit=").map(str::to_owned))
                .and_then(|v| {
                    let f: Vec<f64> = v.split(',').filter_map(|x| x.parse().ok()).collect();
                    (f.len() == 3).then(|| (f[0].to_radians(), f[1].to_radians(), f[2]))
                })
                .unwrap_or((-2.2, 25f64.to_radians(), 2.2)),
            shot,
            shot_frame: frame,
            shot_now: None,
            ticks: 0,
            last: std::time::Instant::now(),
        };
        if let Some(rs) = &cc.wgpu_render_state {
            app.build_resources(rs);
        }
        app.aim();
        Ok(app)
    }

    /// Build the mesh buffers for the ride we have. A live ride has no meshes
    /// until its header lands, so this does nothing until then and is retried.
    fn build_resources(&mut self, rs: &egui_wgpu::RenderState) {
        if self.ready || self.ride.meshes.is_empty() {
            return;
        }
        match Resources::new(&rs.device, rs.target_format, &self.ride) {
            Ok(res) => {
                rs.renderer.write().callback_resources.insert(res);
                self.ready = true;
            }
            Err(e) => self.error = Some(format!("scene: {e:#}")),
        }
    }

    /// Drain everything the reader thread has parsed since the last repaint.
    fn pump(&mut self) {
        let Some(live) = self.live.as_mut() else { return };
        live.poll_child();
        loop {
            match live.rx.try_recv() {
                // the header carries the scene; keep any frames already in hand
                Ok(Msg::Header(r)) => {
                    let frames = std::mem::take(&mut self.ride.frames);
                    self.ride = *r;
                    self.ride.frames = frames;
                }
                Ok(Msg::Frame(f)) => self.ride.frames.push(*f),
                Ok(Msg::Bad(e)) => eprintln!("live: {e}"),
                Ok(Msg::Eof) => live.poll_child(),
                Err(_) => break,
            }
        }
        if let Some(e) = self.live.as_ref().and_then(|l| l.error.clone()) {
            self.error = Some(e);
        }
    }

    /// `live: 123 frames, t = 2.05 s, child running`.
    fn live_status(&self) -> Option<String> {
        let live = self.live.as_ref()?;
        let t = self.ride.frames.last().map(|f| f.t).unwrap_or(0.0);
        let child = match live.ended {
            None => "child running".to_string(),
            Some(Some(code)) => format!("child ended ({code})"),
            Some(None) => "child ended (killed)".to_string(),
        };
        Some(format!("live: {} frames, t = {t:.2} s, {child}", self.ride.frames.len()))
    }

    fn aim(&mut self) {
        let (az, el, dist) = self.orbit;
        self.camera.target = self.ride.tracked(self.cursor);
        self.camera.eye = self.camera.target + V::new(dist * el.cos() * az.cos(), dist * el.cos() * az.sin(), dist * el.sin());
    }

    /// The knobs and the restart button: a live ride is re-run, not re-read.
    fn controls(&mut self, ui: &mut egui::Ui) {
        let Some(live) = self.live.as_mut() else { return };
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut live.opts.shove, 0.0..=40.0).text("shove peak (N·s)"));
            ui.add(egui::DragValue::new(&mut live.opts.shove_at).speed(0.05).range(0.0..=60.0).prefix("at ").suffix(" s"));
            ui.add(egui::DragValue::new(&mut live.opts.duration).speed(0.1).range(0.1..=600.0).prefix("for ").suffix(" s"));
            if ui.button("restart").clicked() {
                live.restart();
                self.ride.frames.clear();
                self.cursor = 0;
                self.accum = 0.0;
                self.follow = true;
                self.playing = true;
            }
        });
    }
}

impl eframe::App for RideApp {
    fn ui(&mut self, root: &mut egui::Ui, f: &mut eframe::Frame) {
        let ctx = root.ctx().clone();
        self.pump();
        if !self.ready {
            if let Some(rs) = f.wgpu_render_state().cloned() {
                self.build_resources(&rs);
            }
        }
        let n = self.ride.frames.len();
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last).as_secs_f64().min(0.25);
        self.last = now;
        if n > 0 && self.playing && self.shot.is_none() {
            // Playback is paced by the ride's own dt against the wall clock,
            // because the recorder runs several times faster than real time:
            // "follow live" rides the frontier of arrived frames, it does not
            // jump to it. A file ride loops; a live one holds at the newest.
            self.accum += dt * self.speed;
            let step = (self.accum / self.ride.dt).floor();
            if step >= 1.0 {
                self.accum -= step * self.ride.dt;
                let next = self.cursor + step as usize;
                self.cursor = if self.live.is_some() { next.min(n - 1) } else { next % n };
            }
        }
        self.cursor = self.cursor.min(n.saturating_sub(1));
        self.aim();

        let t = self.ride.frames.get(self.cursor).map(|f| f.t).unwrap_or(0.0);
        egui::Panel::bottom("transport").show(root, |ui| {
            ui.horizontal(|ui| {
                if ui.button(if self.playing { "⏸ pause" } else { "▶ play" }).clicked() {
                    self.playing = !self.playing;
                }
                if self.live.is_some() {
                    ui.checkbox(&mut self.follow, "follow live");
                }
                if n > 0 {
                    let mut c = self.cursor;
                    if ui.add(egui::Slider::new(&mut c, 0..=n - 1).text("frame")).changed() {
                        // scrubbing is a deliberate step off the live edge
                        self.cursor = c;
                        self.playing = false;
                        self.follow = false;
                    }
                }
                ui.separator();
                ui.label(format!("t = {:.3} s / {:.3} s", t, self.ride.duration()));
                ui.separator();
                for s in [0.25, 0.5, 1.0] {
                    ui.selectable_value(&mut self.speed, s, format!("{s}×"));
                }
                ui.separator();
                ui.label(format!("frame {} / {}", self.cursor + 1, n));
            });
            if self.live.is_some() {
                self.controls(ui);
            }
        });
        egui::Panel::top("title").show(root, |ui| {
            ui.horizontal(|ui| {
                ui.heading(if self.ride.name.is_empty() { "the ride" } else { self.ride.name.as_str() });
                ui.separator();
                let label = self.ride.actors.get(self.ride.track).map(|a| a.label.as_str()).unwrap_or("origin");
                let p = self.camera.target;
                ui.label(format!("tracking {label} at ({:+.2}, {:+.2}, {:+.2}) m — drag to orbit, scroll to zoom", p.x, p.y, p.z));
            });
            if let Some(s) = self.live_status() {
                ui.label(s);
            }
            if let Some(e) = &self.error {
                ui.colored_label(egui::Color32::from_rgb(230, 120, 90), e);
            }
        });

        egui::CentralPanel::default().show(root, |ui| {
            let size = ui.available_size();
            let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::drag());
            if resp.dragged() {
                let d = resp.drag_delta();
                self.orbit.0 -= d.x as f64 * 0.005;
                self.orbit.1 = (self.orbit.1 + d.y as f64 * 0.005).clamp(-1.4, 1.5);
                self.aim();
            }
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if resp.hovered() && scroll.abs() > 0.0 {
                self.orbit.2 = (self.orbit.2 * (1.0 - scroll as f64 * 0.002)).clamp(0.3, 40.0);
                self.aim();
            }
            if !self.ready || n == 0 {
                ui.painter().rect_filled(rect, 0.0, egui::Color32::from_gray(20));
                let msg = self.error.clone().unwrap_or_else(|| "waiting for the first frame…".into());
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, msg, egui::FontId::proportional(18.0), egui::Color32::from_gray(170));
                return;
            }
            let ppp = ctx.pixels_per_point();
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect,
                RideCallback {
                    instances: Arc::new(self.ride.instances(self.cursor)),
                    camera: self.camera,
                    size: ((size.x * ppp) as u32, (size.y * ppp) as u32),
                    shot: self.shot_now.take(),
                },
            ));
        });

        // headless verification: draw once, save, quit. A live ride waits for
        // the frame to arrive — or for the child to end without ever sending it.
        if let Some(path) = self.shot.clone() {
            let arrived = n > self.shot_frame || (n > 0 && !self.live.as_ref().map(|l| l.running()).unwrap_or(false));
            if arrived && self.ready {
                self.cursor = self.shot_frame.min(n - 1);
                self.ticks += 1;
                if self.ticks == 4 {
                    self.shot_now = Some(path);
                }
                if self.ticks == 8 {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            } else if self.error.is_some() {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}

/// Where the window gets its poses.
pub enum Source {
    /// `--ride <ride.json>`: a finished recording on disk
    File(PathBuf),
    /// `--live`: a child process streaming the ride as it runs
    Live(LiveOpts),
}

/// `kosm-view --ride <ride.json>` / `--live`: the window, playing a ride.
pub fn run(source: Source) -> anyhow::Result<()> {
    let (ride, live) = match source {
        Source::File(path) => {
            let ride = Ride::load(&path)?;
            eprintln!("{}: {} frames, {} meshes, {} actors", path.display(), ride.frames.len(), ride.meshes.len(), ride.actors.len());
            (ride, None)
        }
        Source::Live(opts) => (Ride::empty(), Some(Live::spawn(opts))),
    };
    let frame: usize = std::env::args().find_map(|a| a.strip_prefix("--frame=").and_then(|v| v.parse().ok())).unwrap_or(0);
    let shot: Option<PathBuf> = std::env::args().find_map(|a| a.strip_prefix("--shot=").map(PathBuf::from));
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 760.0]).with_title("Kosm view — the ride"),
        ..Default::default()
    };
    eframe::run_native("Kosm view", options, Box::new(move |cc| Ok(Box::new(RideApp::new(cc, ride, live, frame, shot)?))))
        .map_err(|e| anyhow::anyhow!("{e}"))
}
