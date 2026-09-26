//! The GPU side: an offscreen depth-tested pass over the field meshes and the
//! instanced quark markers, blitted into the egui panel. Same shape as
//! kosm-view's live tier.

use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor, wgpu};
use wgpu::util::DeviceExt;

use crate::field::{Mesh, Vertex, unit_sphere};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub m: [[f32; 4]; 4],
    pub tint: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    light: [f32; 4],
}

/// Vertical field of view, radians.
const FOV: f32 = 0.8;

/// Orbit camera around the origin, Z up.
#[derive(Clone, Copy)]
pub struct Orbit {
    pub azimuth: f32,
    pub elevation: f32,
    pub distance: f32,
}

impl Orbit {
    pub fn eye(&self) -> [f32; 3] {
        let (ce, se) = (self.elevation.cos(), self.elevation.sin());
        [
            self.distance * ce * self.azimuth.cos(),
            self.distance * ce * self.azimuth.sin(),
            self.distance * se,
        ]
    }

    /// Camera basis: forward (toward the origin), right, up.
    fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let e = self.eye();
        let norm = |v: [f32; 3]| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let f = norm([-e[0], -e[1], -e[2]]);
        let r = norm(cross(f, [0.0, 0.0, 1.0]));
        (f, r, cross(r, f))
    }

    /// World ray through normalized device coordinates (x right, y up, ±1).
    pub fn ray(&self, ndc: [f32; 2], aspect: f32) -> ([f32; 3], [f32; 3]) {
        let (f, r, u) = self.basis();
        let t = (0.5 * FOV).tan();
        let d: [f32; 3] =
            std::array::from_fn(|i| f[i] + ndc[0] * t * aspect * r[i] + ndc[1] * t * u[i]);
        (self.eye(), d)
    }

    /// Normalized device coordinates of a world point, if in front.
    pub fn project(&self, p: [f32; 3], aspect: f32) -> Option<[f32; 2]> {
        let m = self.view_proj(aspect);
        let c: [f32; 4] =
            std::array::from_fn(|r| m[0][r] * p[0] + m[1][r] * p[1] + m[2][r] * p[2] + m[3][r]);
        (c[3] > 1e-4).then(|| [c[0] / c[3], c[1] / c[3]])
    }

    fn view_proj(&self, aspect: f32) -> [[f32; 4]; 4] {
        let e = self.eye();
        let len = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let norm = |v: [f32; 3]| {
            let l = len(v);
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let f = norm([-e[0], -e[1], -e[2]]);
        let r = norm(cross(f, [0.0, 0.0, 1.0]));
        let up = cross(r, f);
        let view = [
            [r[0], up[0], -f[0], 0.0],
            [r[1], up[1], -f[1], 0.0],
            [r[2], up[2], -f[2], 0.0],
            [-dot(r, e), -dot(up, e), dot(f, e), 1.0],
        ];
        let (near, far) = (0.02f32, 50.0f32);
        let t = 1.0 / (0.5 * FOV).tan();
        let proj = [
            [t / aspect, 0.0, 0.0, 0.0],
            [0.0, t, 0.0, 0.0],
            [0.0, 0.0, far / (near - far), -1.0],
            [0.0, 0.0, near * far / (near - far), 0.0],
        ];
        let mut out = [[0.0f32; 4]; 4];
        for c in 0..4 {
            for rr in 0..4 {
                out[c][rr] = (0..4).map(|k| proj[k][rr] * view[c][k]).sum();
            }
        }
        out
    }
}

/// What the viewer hands the GPU each frame. `generation` changes whenever
/// the mesh does, so unchanged frames skip the upload.
pub struct Scene {
    pub generation: u64,
    pub mesh: Mesh,
    pub markers: Vec<Instance>,
}

pub struct Resources {
    pipeline: wgpu::RenderPipeline,
    inst_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bind: wgpu::BindGroup,
    blit_layout: wgpu::BindGroupLayout,
    blit_bind: Option<wgpu::BindGroup>,
    sampler: wgpu::Sampler,
    mesh: Option<(u64, wgpu::Buffer, wgpu::Buffer, u32)>,
    sphere: (wgpu::Buffer, wgpu::Buffer, u32),
    instances: wgpu::Buffer,
    n_instances: u32,
    target: Option<(
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::TextureView,
        u32,
        u32,
    )>,
}

const MAX_INSTANCES: usize = 4096;

impl Resources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qcd"),
            source: wgpu::ShaderSource::Wgsl(include_str!("qcd.wgsl").into()),
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qcd"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qcd"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 24,
                    shader_location: 2,
                },
            ],
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 4,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 5,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 6,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 7,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 64,
                    shader_location: 8,
                },
            ],
        };
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qcd"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |label: &str, vs: &str, buffers: &[Option<wgpu::VertexBufferLayout>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
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
            })
        };
        let pipeline = make("field", "vs_main", &[Some(vertex_layout.clone())]);
        let inst_pipeline = make(
            "markers",
            "vs_inst",
            &[Some(vertex_layout), Some(instance_layout)],
        );

        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit"),
            bind_group_layouts: &[Some(&blit_layout)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&blit_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_blit"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_blit"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let sph = unit_sphere(20);
        let sphere = (
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sphere"),
                contents: bytemuck::cast_slice(&sph.verts),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sphere idx"),
                contents: bytemuck::cast_slice(&sph.idx),
                usage: wgpu::BufferUsages::INDEX,
            }),
            sph.idx.len() as u32,
        );
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: (MAX_INSTANCES * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            inst_pipeline,
            blit_pipeline,
            uniforms,
            bind,
            blit_layout,
            blit_bind: None,
            sampler,
            mesh: None,
            sphere,
            instances,
            n_instances: 0,
            target: None,
        }
    }

    fn ensure_target(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if matches!(&self.target, Some((_, _, _, cw, ch)) if *cw == w && *ch == h) {
            return;
        }
        let tex = |label, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = tex(
            "qcd color",
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let depth = tex(
            "qcd depth",
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let cv = color.create_view(&Default::default());
        let dv = depth.create_view(&Default::default());
        self.blit_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&cv),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));
        self.target = Some((color, cv, dv, w, h));
    }
}

pub struct Callback {
    pub scene: Arc<Scene>,
    pub orbit: Orbit,
    pub size: (u32, u32),
    /// Save the offscreen frame here after drawing it.
    pub shot: Option<std::path::PathBuf>,
}

impl CallbackTrait for Callback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let _ = device.poll(wgpu::PollType::Poll);
        let res: &mut Resources = resources.get_mut().expect("qcd resources");
        let (w, h) = (self.size.0.max(8), self.size.1.max(8));
        res.ensure_target(device, w, h);

        let eye = self.orbit.eye();
        let u = Uniforms {
            view_proj: self.orbit.view_proj(w as f32 / h as f32),
            eye: [eye[0], eye[1], eye[2], 0.0],
            light: [0.4, -0.5, 0.9, 0.0],
        };
        queue.write_buffer(&res.uniforms, 0, bytemuck::bytes_of(&u));

        let scene = &self.scene;
        if res.mesh.as_ref().map(|m| m.0) != Some(scene.generation) {
            res.mesh = (!scene.mesh.idx.is_empty()).then(|| {
                (
                    scene.generation,
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("field"),
                        contents: bytemuck::cast_slice(&scene.mesh.verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("field idx"),
                        contents: bytemuck::cast_slice(&scene.mesh.idx),
                        usage: wgpu::BufferUsages::INDEX,
                    }),
                    scene.mesh.idx.len() as u32,
                )
            });
            let n = scene.markers.len().min(MAX_INSTANCES);
            res.n_instances = n as u32;
            if n > 0 {
                queue.write_buffer(&res.instances, 0, bytemuck::cast_slice(&scene.markers[..n]));
            }
        }

        let (_, cv, dv, _, _) = res.target.as_ref().unwrap();
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("qcd") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("qcd"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: cv,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: dv,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &res.bind, &[]);
            if let Some((_, vb, ib, n)) = &res.mesh {
                pass.set_pipeline(&res.pipeline);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*n, 0, 0..1);
            }
            if res.n_instances > 0 {
                pass.set_pipeline(&res.inst_pipeline);
                pass.set_vertex_buffer(0, res.sphere.0.slice(..));
                pass.set_vertex_buffer(1, res.instances.slice(..));
                pass.set_index_buffer(res.sphere.1.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..res.sphere.2, 0, 0..res.n_instances);
            }
        }
        let cmd = enc.finish();
        if let Some(path) = &self.shot {
            queue.submit([cmd]);
            let (tex, _, _, w, h) = res.target.as_ref().unwrap();
            read_back(device, queue, tex, (*w, *h))
                .save(path)
                .expect("save shot");
            eprintln!("shot {}", path.display());
            return vec![];
        }
        vec![cmd]
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let res: &Resources = resources.get().expect("qcd resources");
        if let Some(bind) = &res.blit_bind {
            pass.set_pipeline(&res.blit_pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..4, 0..1);
        }
    }
}

fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    (w, h): (u32, u32),
) -> image::RgbaImage {
    let row = (4 * w).div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_through_projection_hits_the_point() {
        // quark dragging projects markers to the screen and casts pointer rays
        // back; the two must invert each other
        let orbit = Orbit {
            azimuth: -1.9,
            elevation: 0.95,
            distance: 2.7,
        };
        let aspect = 1.6;
        for p in [[0.3f32, -0.2, 0.0], [-0.5, 0.4, 0.1], [0.0, 0.0, 0.0]] {
            let ndc = orbit.project(p, aspect).expect("in front");
            let (o, d) = orbit.ray(ndc, aspect);
            // closest approach of the ray to p
            let op: [f32; 3] = std::array::from_fn(|i| p[i] - o[i]);
            let dd: f32 = d.iter().map(|x| x * x).sum();
            let t = op.iter().zip(&d).map(|(a, b)| a * b).sum::<f32>() / dd;
            let miss: f32 = (0..3)
                .map(|i| (o[i] + t * d[i] - p[i]).powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(miss < 1e-4, "ray misses {p:?} by {miss}");
        }
    }
}
