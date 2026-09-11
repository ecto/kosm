//! The two passes and everything they bind.
//!
//! One shadow pass into a 2048² depth map along the sun, then one forward
//! pass over the same vertex and instance buffers. Nothing is deferred,
//! nothing is instanced across materials, and there is no G-buffer: the whole
//! tier is a hundred thousand triangles under a fragment shader that reads
//! nine coefficients out of a storage buffer, which is a rounding error on
//! any GPU that can open a window.
//!
//! The **projection must be the tracer's**, to the pixel, or the settle blend
//! shows a seam at every silhouette. [`view_proj`] is
//! `kosm_render::Camera::ray`'s rectilinear map written as a 4×4: the
//! vertical half-extent is `tan(fov/2)`, the horizontal is that times the
//! aspect, and the basis is the camera's own `(right, up, forward)` rather
//! than one rebuilt from an up-hint — which matters, because the rig's camera
//! can be rolled and a `look_at` reconstruction would quietly level it.

use std::sync::Arc;

use wgpu::util::DeviceExt as _;

use super::{CausticQuad, Instance, Scene, Vertex};
use crate::Projection;

/// The shadow map's side, in texels. One map for the level; see
/// [`super::shadow`] for why there are no cascades.
pub const SHADOW: u32 = 2048;

/// The uniform block, laid out as WGSL's `Uniforms` — every member on a
/// sixteen-byte boundary, in the order the shader declares them.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [f32; 16],
    sun_view_proj: [f32; 16],
    eye: [f32; 4],
    sun_dir: [f32; 4],
    sun_irr_a: [f32; 4],
    sun_irr_b: [f32; 4],
    probe_origin: [f32; 4],
    probe_dims: [u32; 4],
    knobs: [f32; 4],
    sea: [f32; 4],
    swell: [f32; 4],
    swell_angle: [f32; 4],
    sea_absorb: [f32; 4],
    sea_flags: [u32; 4],
    caustic_origin: [[f32; 4]; 2],
    caustic_u: [[f32; 4]; 2],
    caustic_v: [[f32; 4]; 2],
    caustic_flags: [f32; 4],
    band_to_rgb: [[f32; 4]; 6],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowUniforms {
    sun_view_proj: [f32; 16],
}

/// What one drawn frame needs beyond the scene itself.
#[derive(Clone, Debug)]
pub struct Frame {
    /// The camera, from the rig. Its `fov_deg` and its basis are used
    /// verbatim; see [`view_proj`].
    pub camera: kosm_render::Camera,
    pub size: (u32, u32),
    /// The authored exposure already multiplied by the meter's.
    pub exposure: f32,
    /// Seconds, for the swell. Simulated, not wall-clock.
    pub time: f32,
    /// Which baked sun, as a fractional index into the volume's list.
    pub sun_index: f64,
    /// Whether the caustic textures need re-uploading this frame. A photon
    /// map is retraced only when the lens moves, so most frames say no.
    pub caustics_dirty: bool,
}

impl Frame {
    pub fn new(camera: kosm_render::Camera, size: (u32, u32)) -> Self {
        Self {
            camera,
            size,
            exposure: 0.7,
            time: 0.0,
            sun_index: 0.0,
            caustics_dirty: false,
        }
    }
}

/// The rectilinear map, as the 4×4 a vertex shader can apply.
///
/// Column-major, wgpu's `z ∈ [0, 1]`. It is the same map
/// `kosm_render::Camera::ray` generates for `Projection::Rectilinear`: at
/// `sy = ±1` the ray makes an angle of `fov_deg / 2` with `forward`, and the
/// horizontal extent is that times the aspect — which is what makes the
/// raster frame and the traced frame line up pixel for pixel and the settle
/// blend a fade rather than a double exposure.
pub fn view_proj(cam: &kosm_render::Camera, aspect: f32) -> [f32; 16] {
    let (r, up, f, e) = (cam.right, cam.up, cam.forward, cam.eye);
    let view = [
        [r.x, up.x, -f.x, 0.0],
        [r.y, up.y, -f.y, 0.0],
        [r.z, up.z, -f.z, 0.0],
        [-dot(r, e), -dot(up, e), dot(f, e), 1.0],
    ];
    let (near, far) = (0.02f64, 400.0f64);
    let t = 1.0 / (0.5 * cam.fov_deg.to_radians()).tan();
    let proj = [
        [t / aspect as f64, 0.0, 0.0, 0.0],
        [0.0, t, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ];
    let mut out = [0.0f32; 16];
    for c in 0..4 {
        for rr in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += proj[k][rr] * view[c][k];
            }
            out[c * 4 + rr] = s as f32;
        }
    }
    out
}

fn dot(a: kosm_render::math::Vec3, b: kosm_render::math::Point3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

struct MeshGpu {
    vb: wgpu::Buffer,
    n: u32,
    inst: wgpu::Buffer,
    cap: u32,
}

/// The tier's GPU resources.
pub struct Raster {
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    shadow_uniforms: wgpu::Buffer,
    materials: wgpu::Buffer,
    probes: wgpu::Buffer,
    probe_inside: wgpu::Buffer,
    bind: wgpu::BindGroup,
    shadow_bind: wgpu::BindGroup,
    blit_layout: wgpu::BindGroupLayout,
    blit_bind: Option<wgpu::BindGroup>,
    blit_sampler: wgpu::Sampler,
    shadow_map: wgpu::TextureView,
    caustic_tex: wgpu::Texture,
    caustic_res: (u32, u32),
    meshes: Vec<MeshGpu>,
    colour: Option<(Arc<wgpu::Texture>, wgpu::TextureView, wgpu::TextureView, u32, u32)>,
}

impl Raster {
    /// Whether this tier can draw a camera at all.
    ///
    /// The equidistant fisheye is `r = f·θ`, which is not a projective map: no
    /// 4×4 expresses it, and a post-pass warp of a wider rectilinear render
    /// cannot reach past a 180° field or hold the corners at the same sample
    /// density the tracer gives them. So the fisheye is **tracer-only** on
    /// this tier and the caller falls back for the whole frame rather than
    /// showing a picture that is a different camera from the reference it is
    /// about to settle into.
    pub fn supports(projection: Projection) -> bool {
        matches!(projection, Projection::Rectilinear)
    }

    /// Build every resource the scene needs on `device`.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        target_format: wgpu::TextureFormat,
    ) -> anyhow::Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raster scene"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scene.wgsl").into()),
        });
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raster shadow"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shadow.wgsl").into()),
        });

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raster uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raster shadow uniforms"),
            size: std::mem::size_of::<ShadowUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let materials = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("raster materials"),
            contents: bytemuck::cast_slice(&scene.materials),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // The volume, uploaded verbatim: the bake's own hemisphere rays carry
        // the sky, so there is nothing to fold in and the shader's read is
        // `ProbeVolume::sample_sh` written in WGSL.
        let probes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("raster probes"),
            contents: bytemuck::cast_slice(&probe_words(&scene.probes)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let probe_inside = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("raster probe mask"),
            contents: bytemuck::cast_slice(&inside_words(&scene.probes)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let shadow_map = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("raster shadow map"),
                size: wgpu::Extent3d { width: SHADOW, height: SHADOW, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());

        // The caustic receivers: one array layer per rectangle, always two, so
        // a frame with no map binds the same pipeline as a frame with one.
        let caustic_res = scene
            .caustics
            .first()
            .map(|q| q.res)
            .unwrap_or((4, 4));
        let caustic_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raster caustics"),
            size: wgpu::Extent3d {
                width: caustic_res.0,
                height: caustic_res.1,
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Rgba16Float and not Rgba32Float: a linear sampler over a
            // 32-bit float texture needs `FLOAT32_FILTERABLE`, which the
            // window's own device is not asked for. Half a float is four
            // decimal digits of a density estimate, which is more than the
            // estimate has.
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raster"),
            entries: &[
                uniform_entry(0),
                storage_entry(1),
                storage_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                storage_entry(7),
            ],
        });
        let compare = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raster shadow sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raster caustic sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raster"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: materials.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: probes.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&shadow_map),
                },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&compare) },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(
                        &caustic_tex.create_view(&wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::D2Array),
                            ..Default::default()
                        }),
                    ),
                },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&linear) },
                wgpu::BindGroupEntry { binding: 7, resource: probe_inside.as_entire_binding() },
            ],
        });

        let shadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raster shadow"),
            entries: &[uniform_entry(0)],
        });
        let shadow_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raster shadow"),
            layout: &shadow_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: shadow_uniforms.as_entire_binding() }],
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
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32x4, offset: 64, shader_location: 6 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 80, shader_location: 7 },
            ],
        };

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raster"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raster"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(vertex_layout.clone()), Some(instance_layout.clone())],
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
            // Authored solids are inconsistently wound — the ride tier found
            // this and the cove's are no better — so nothing is culled and
            // the fragment stage flips a normal that faces away.
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

        let shadow_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raster shadow"),
            bind_group_layouts: &[Some(&shadow_layout)],
            immediate_size: 0,
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raster shadow"),
            layout: Some(&shadow_pl),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("vs_shadow"),
                buffers: &[Some(vertex_layout), Some(instance_layout)],
                compilation_options: Default::default(),
            },
            fragment: None,
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
            label: Some("raster blit"),
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
            label: Some("raster blit"),
            bind_group_layouts: &[Some(&blit_layout)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raster blit"),
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

        let mut meshes = Vec::with_capacity(scene.meshes.len());
        for mesh in &scene.meshes {
            let cap = mesh.instances.len().max(1) as u32;
            let filler = [Vertex::default()];
            let verts: &[Vertex] =
                if mesh.vertices.is_empty() { &filler } else { mesh.vertices.as_slice() };
            meshes.push(MeshGpu {
                vb: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("raster mesh"),
                    contents: bytemuck::cast_slice(verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                n: mesh.vertices.len() as u32,
                inst: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("raster instances"),
                    size: (cap as usize * std::mem::size_of::<Instance>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                cap,
            });
        }

        let mut me = Self {
            pipeline,
            shadow_pipeline,
            blit_pipeline,
            uniforms,
            shadow_uniforms,
            materials,
            probes,
            probe_inside,
            bind,
            shadow_bind,
            blit_layout,
            blit_bind: None,
            blit_sampler: linear,
            shadow_map,
            caustic_tex,
            caustic_res,
            meshes,
            colour: None,
        };
        me.upload_caustics(queue, &scene.caustics);
        Ok(me)
    }

    /// Re-upload the material table — the cove's override layer changes it
    /// once, at build time, and a level that repaints itself would change it
    /// again.
    pub fn upload_materials(&self, queue: &wgpu::Queue, scene: &Scene) {
        queue.write_buffer(&self.materials, 0, bytemuck::cast_slice(&scene.materials));
    }

    /// Re-upload the probe volume and its inside-solid mask. A level whose
    /// sun moves between bakes changes both.
    pub fn upload_probes(&self, queue: &wgpu::Queue, scene: &Scene) {
        queue.write_buffer(&self.probes, 0, bytemuck::cast_slice(&probe_words(&scene.probes)));
        queue.write_buffer(
            &self.probe_inside,
            0,
            bytemuck::cast_slice(&inside_words(&scene.probes)),
        );
    }

    /// Push the receiver rectangles' irradiance into the array texture.
    pub fn upload_caustics(&mut self, queue: &wgpu::Queue, quads: &[CausticQuad]) {
        let (w, h) = self.caustic_res;
        for layer in 0..2u32 {
            let mut half = vec![0u16; (w * h * 4) as usize];
            if let Some(q) = quads.get(layer as usize) {
                if q.res == (w, h) {
                    for i in 0..(w * h) as usize {
                        for c in 0..3 {
                            half[i * 4 + c] = f16(q.data.get(i * 3 + c).copied().unwrap_or(0.0));
                        }
                        half[i * 4 + 3] = f16(1.0);
                    }
                }
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.caustic_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&half),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 8),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
    }

    /// Draw one frame. The texture handed back is `Rgba8Unorm` holding
    /// already-sRGB bytes, with `Rgba8UnormSrgb` among its view formats — so
    /// it can go straight to [`crate::viewport::Image::Texture`] with nothing
    /// crossing the bus, or through [`crate::frame::read_back`] when the
    /// settle blend needs it in memory.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        frame: &Frame,
    ) -> Arc<wgpu::Texture> {
        let (w, h) = (frame.size.0.max(8), frame.size.1.max(8));
        self.ensure_target(device, w, h);
        if frame.caustics_dirty {
            self.upload_caustics(queue, &scene.caustics);
        }

        let (lo, hi) = scene.bounds;
        let fr = super::shadow::Frustum::over(scene.sun.direction, lo, hi, SHADOW);
        queue.write_buffer(
            &self.shadow_uniforms,
            0,
            bytemuck::bytes_of(&ShadowUniforms { sun_view_proj: fr.view_proj }),
        );

        let d = scene.sun.direction;
        let irr = scene.sun.irradiance;
        let po = scene.probes.origin;
        let dims = scene.probes.dims;
        let m = super::band_to_rgb();
        let sea = scene.sea.unwrap_or(super::Sea::cove(0.0, 0.0, 0.0, 1.0));
        let u = Uniforms {
            view_proj: view_proj(&frame.camera, w as f32 / h as f32),
            sun_view_proj: fr.view_proj,
            eye: [
                frame.camera.eye.x as f32,
                frame.camera.eye.y as f32,
                frame.camera.eye.z as f32,
                frame.exposure,
            ],
            sun_dir: [d[0] as f32, d[1] as f32, d[2] as f32, scene.sun.angular_radius as f32],
            sun_irr_a: [irr[0], irr[1], irr[2], irr[3]],
            sun_irr_b: [irr[4], irr[5], 0.0, 0.0],
            probe_origin: [po[0] as f32, po[1] as f32, po[2] as f32, scene.probes.spacing as f32],
            probe_dims: [dims[0], dims[1], dims[2], scene.probes.suns.len().max(1) as u32],
            knobs: [
                frame.sun_index as f32,
                // The bias, already in the shadow map's own clip depth: one
                // texel of world sideways is one texel of world *along the
                // sun* at worst, so a texel over the frustum's depth range is
                // the honest unit and the slope term widens it.
                (fr.texel_m / fr.depth_m.max(1e-6)).max(1e-6),
                SHADOW as f32,
                frame.time,
            ],
            sea: [sea.z as f32, sea.slope as f32, sea.waterline_y as f32, sea.reach as f32],
            swell: [
                sea.swell[0] as f32,
                sea.swell[1] as f32,
                sea.swell_b[0] as f32,
                sea.swell_b[1] as f32,
            ],
            swell_angle: [
                sea.swell_angle as f32,
                sea.swell_b_angle as f32,
                sea.speed as f32,
                sea.ior as f32,
            ],
            sea_absorb: [
                sea.absorption[0] as f32,
                sea.absorption[1] as f32,
                sea.absorption[2] as f32,
                sea.water_material as f32,
            ],
            sea_flags: [sea.seabed_material, u32::from(scene.sea.is_some()), 0, 0],
            caustic_origin: quad_vec4(&scene.caustics, |q| q.origin),
            caustic_u: quad_vec4(&scene.caustics, |q| q.u),
            caustic_v: quad_vec4(&scene.caustics, |q| q.v),
            caustic_flags: [scene.caustics.len().min(2) as f32, 0.02, 0.0, 0.0],
            band_to_rgb: [
                [m[0][0], m[0][1], m[0][2], 0.0],
                [m[1][0], m[1][1], m[1][2], 0.0],
                [m[2][0], m[2][1], m[2][2], 0.0],
                [m[3][0], m[3][1], m[3][2], 0.0],
                [m[4][0], m[4][1], m[4][2], 0.0],
                [m[5][0], m[5][1], m[5][2], 0.0],
            ],
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));

        // the instances, grown where a mesh has more copies than it had
        let mut counts = Vec::with_capacity(self.meshes.len());
        for (g, mesh) in self.meshes.iter_mut().zip(scene.meshes.iter()) {
            let n = mesh.instances.len() as u32;
            if n > g.cap {
                g.inst = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("raster instances"),
                    size: (n as usize * std::mem::size_of::<Instance>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                g.cap = n;
            }
            if n > 0 {
                queue.write_buffer(&g.inst, 0, bytemuck::cast_slice(&mesh.instances));
            }
            counts.push(n);
        }

        let (tex, cv, dv, _, _) = self.colour.as_ref().expect("target");
        let out = tex.clone();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("raster"),
        });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raster shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_map,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, &self.shadow_bind, &[]);
            for (g, n) in self.meshes.iter().zip(counts.iter().copied()) {
                if n == 0 || g.n == 0 {
                    continue;
                }
                pass.set_vertex_buffer(0, g.vb.slice(..));
                pass.set_vertex_buffer(1, g.inst.slice(..));
                pass.draw(0..g.n, 0..n);
            }
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raster scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: cv,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // The sky, already sRGB-encoded: the clear is what a
                        // ray that hits nothing shows, and on this tier that
                        // is the level's horizon colour rather than black.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.62,
                            g: 0.76,
                            b: 0.92,
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            for (g, n) in self.meshes.iter().zip(counts.iter().copied()) {
                if n == 0 || g.n == 0 {
                    continue;
                }
                pass.set_vertex_buffer(0, g.vb.slice(..));
                pass.set_vertex_buffer(1, g.inst.slice(..));
                pass.draw(0..g.n, 0..n);
            }
        }
        queue.submit([enc.finish()]);
        out
    }

    /// The colour target, for a caller that wants to read it back.
    pub fn texture(&self) -> Option<Arc<wgpu::Texture>> {
        self.colour.as_ref().map(|(t, ..)| t.clone())
    }

    /// Paint the last drawn frame into an existing render pass — the egui
    /// path, the same shape `ride::Resources` uses.
    pub fn blit(&self, pass: &mut wgpu::RenderPass<'static>) {
        if let Some(bind) = &self.blit_bind {
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..4, 0..1);
        }
    }

    fn ensure_target(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if let Some((_, _, _, cw, ch)) = &self.colour {
            if *cw == w && *ch == h {
                return;
            }
        }
        let colour = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raster colour"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            // the viewport blits an already-sRGB image through an sRGB view
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raster depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let cv = colour.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            ..Default::default()
        });
        let dv = depth.create_view(&Default::default());
        self.blit_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raster blit"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&cv) },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
            ],
        }));
        self.colour = Some((Arc::new(colour), cv, dv, w, h));
    }
}

fn quad_vec4(quads: &[CausticQuad], f: impl Fn(&CausticQuad) -> [f64; 3]) -> [[f32; 4]; 2] {
    let mut out = [[0.0f32; 4]; 2];
    for i in 0..2 {
        if let Some(q) = quads.get(i) {
            let v = f(q);
            out[i] = [v[0] as f32, v[1] as f32, v[2] as f32, 0.0];
        }
    }
    out
}

/// The volume's coefficients as the shader binds them: verbatim, with one
/// empty probe standing in for a volume that has none — an empty storage
/// binding is not a binding, and 216 bytes of zeros keeps the shader's loop
/// honest.
fn probe_words(v: &super::ProbeVolume) -> Vec<f32> {
    use super::probes::{BANDS, SH};
    if v.data.is_empty() { vec![0.0; SH * BANDS] } else { v.data.clone() }
}

/// The inside-solid bitmask, one bit per probe, x fastest.
fn inside_words(v: &super::ProbeVolume) -> Vec<u32> {
    if v.inside.is_empty() { vec![0u32] } else { v.inside.clone() }
}

/// IEEE binary16 from a binary32. Round-to-nearest-even on the mantissa,
/// saturating at the half's own range — a caustic's irradiance is a density
/// estimate with three decent digits in it, and a half has four.
fn f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mant = bits & 0x007f_ffff;
    if exp > 15 {
        return sign | 0x7bff; // saturate rather than go infinite
    }
    if exp < -14 {
        // subnormal or zero; the smallest half is 6e-8 and a caustic that
        // dim is not a caustic
        return sign;
    }
    let e = ((exp + 15) as u16) << 10;
    let m = (mant >> 13) as u16;
    let round = u16::from(mant & 0x1000 != 0 && (mant & 0x0fff != 0 || m & 1 != 0));
    sign | (e | m).wrapping_add(round)
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kosm_render::math::{Point3, Vec3};

    /// **The raster's projection is the tracer's.** A point placed at the
    /// direction `Camera::ray` generates for a screen coordinate must land
    /// back on that screen coordinate under the 4×4 — or the settle blend is
    /// two cameras fading into each other.
    #[test]
    fn the_matrix_is_the_tracers_rectilinear_ray() {
        let cam = kosm_render::Camera::look_at(
            Point3::new(2.0, -3.0, 1.5),
            Point3::new(0.0, 0.0, 0.5),
            Vec3::new(0.0, 0.0, 1.0),
            50.0,
        );
        let aspect = 16.0 / 9.0;
        let m = view_proj(&cam, aspect as f32);
        for (sx, sy) in [(0.0, 0.0), (0.5, 0.25), (-0.8, 0.9), (1.0, -1.0)] {
            // the tracer's own map, restated here so the test does not depend
            // on `Camera::ray` being public
            let half_h = (cam.fov_deg.to_radians() * 0.5).tan();
            let half_w = half_h * aspect;
            let dir = cam.forward + cam.right * (sx * half_w) + cam.up * (sy * half_h);
            let p = cam.eye + dir * 7.0;
            let mut clip = [0.0f64; 4];
            for r in 0..4 {
                clip[r] = m[r] as f64 * p.x
                    + m[4 + r] as f64 * p.y
                    + m[8 + r] as f64 * p.z
                    + m[12 + r] as f64;
            }
            let (nx, ny) = (clip[0] / clip[3], clip[1] / clip[3]);
            assert!(
                // the matrix is `f32`, so the agreement is to a float's own
                // precision and not to a double's
                (nx - sx).abs() < 1e-6 && (ny - sy).abs() < 1e-6,
                "({sx}, {sy}) came back ({nx}, {ny})"
            );
            assert!(clip[3] > 0.0, "a point in front of the eye has positive w");
        }
    }

    /// The uniform block is what the shader declares: every member on a
    /// sixteen-byte boundary and 528 bytes all told.
    #[test]
    fn the_uniform_block_is_the_shaders() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 528);
        assert_eq!(std::mem::size_of::<Uniforms>() % 16, 0);
        assert_eq!(std::mem::size_of::<Instance>(), 96);
        assert_eq!(std::mem::size_of::<super::super::GpuMaterial>(), 112);
        assert_eq!(std::mem::size_of::<Vertex>(), 24);
    }

    /// The half-float conversion is exact on the values a caustic carries.
    #[test]
    fn the_half_float_round_trips_the_values_a_caustic_has() {
        for x in [0.0f32, 1.0, 0.5, 2.0, 12.5, 1000.0, 0.001] {
            let h = f16(x);
            // decode
            let sign = ((h & 0x8000) as u32) << 16;
            let e = ((h >> 10) & 0x1f) as i32;
            let m = (h & 0x3ff) as u32;
            let back = if e == 0 {
                0.0
            } else {
                f32::from_bits(sign | (((e - 15 + 127) as u32) << 23) | (m << 13))
            };
            assert!(
                (back - x).abs() <= 1e-3 * x.abs().max(1e-3),
                "{x} came back {back}"
            );
        }
    }

    /// The volume goes up verbatim, and an empty one still binds.
    #[test]
    fn the_volume_uploads_as_the_baker_laid_it_out() {
        use super::super::probes::{BANDS, SH, uniform};
        let v = uniform([0.0; 3], 1.0, [2, 2, 2], 1.0);
        assert_eq!(probe_words(&v), v.data);
        assert_eq!(inside_words(&v).len(), v.inside.len());
        let mut empty = v.clone();
        empty.data.clear();
        empty.inside.clear();
        assert_eq!(probe_words(&empty).len(), SH * BANDS);
        assert_eq!(inside_words(&empty).len(), 1);
    }
}
