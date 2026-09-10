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
    inv_view_proj: [f32; 16],
    sky_a: [f32; 4],
    sky_ground: [f32; 4],
    air: [f32; 4],
    screen: [f32; 4],
    shadow_m: [f32; 4],
}

/// What the ambient-occlusion pass and its two blurs read. See
/// `shaders/ao.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AoUniforms {
    view_proj: [f32; 16],
    inv_view_proj: [f32; 16],
    size: [f32; 4],
    knobs: [f32; 4],
    eye: [f32; 4],
}

/// What the bloom's three passes and the resolve read. See
/// `shaders/post.wgsl`; [`kosm_render::post::Post`] is the same chain in Rust.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PostUniforms {
    lens: [f32; 4],
    bloom: [f32; 4],
    size: [f32; 4],
}

/// The linear HDR the scene pass writes and the post pass reads.
///
/// `Rgba16Float` and not `Rgba8Unorm`: the whole point of moving the tonemap
/// into its own pass is that the bloom threshold and the vignette act on
/// *radiance*, and a sunlit rim at forty times the exposure does not fit in a
/// byte. Half a float carries three decimal digits over sixty stops, which is
/// more than a one-sample-per-pixel reference has.
const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// How much the ambient occlusion and the bloom are shrunk before they are
/// computed. Both are low frequencies by construction — an integral over a
/// hemisphere and a wide blur — and both cost their resolution squared.
const AO_SHRINK: u32 = 2;
const BLOOM_SHRINK: u32 = 4;

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
///
/// # The pass chain
///
/// ```text
/// shadow    2048² depth along the sun, orthographic over the level
/// prepass   camera depth, geometry only — no fragment stage at all
/// ao        half-res occlusion off that depth, then two bilateral blurs
/// sky       a full-screen Preetham lookup, and the sun's disc, into HDR
/// scene     the shading, over the prepass's depth, into the same HDR
/// bright    quarter-res threshold of the exposed, vignetted HDR
/// blur ×2   a separable Gaussian over it
/// resolve   exposure, vignette, bloom, ACES, sRGB → the byte the window shows
/// ```
///
/// The prepass earns its keep twice: it is what the occlusion is computed
/// from, and it turns the scene pass's depth test into `LessEqual` with no
/// writes, so the expensive fragment shader runs once per visible pixel
/// instead of once per drawn triangle over it.
pub struct Raster {
    pipeline: wgpu::RenderPipeline,
    prepass_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    ao_pipeline: wgpu::RenderPipeline,
    ao_blur_pipeline: wgpu::RenderPipeline,
    bright_pipeline: wgpu::RenderPipeline,
    bloom_blur_pipeline: wgpu::RenderPipeline,
    resolve_pipeline: wgpu::RenderPipeline,

    uniforms: wgpu::Buffer,
    shadow_uniforms: wgpu::Buffer,
    /// The occlusion pass and its two blurs, one buffer each: the three differ
    /// only in the blur's step, and a single buffer rewritten between them
    /// would not work — every `write_buffer` of a submission lands before any
    /// of its passes run.
    ao_uniforms: [wgpu::Buffer; 3],
    /// Likewise for the bright pass, the two blurs and the resolve.
    post_uniforms: [wgpu::Buffer; 4],
    materials: wgpu::Buffer,
    probes: wgpu::Buffer,
    probe_inside: wgpu::Buffer,

    layout: wgpu::BindGroupLayout,
    bind: Option<wgpu::BindGroup>,
    shadow_bind: wgpu::BindGroup,
    ao_layout: wgpu::BindGroupLayout,
    /// Three bind groups over two ping-ponged occlusion buffers, so no pass
    /// ever has the texture it writes bound for reading.
    ao_binds: Vec<wgpu::BindGroup>,
    post_layout: wgpu::BindGroupLayout,
    post_binds: Vec<wgpu::BindGroup>,
    blit_layout: wgpu::BindGroupLayout,
    blit_bind: Option<wgpu::BindGroup>,

    compare_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,
    blit_sampler: wgpu::Sampler,
    shadow_map: wgpu::TextureView,
    caustic_tex: wgpu::Texture,
    caustic_res: (u32, u32),
    meshes: Vec<MeshGpu>,
    targets: Option<Targets>,
}

/// Everything whose size is the frame's.
struct Targets {
    size: (u32, u32),
    /// The byte the window shows, and what `read_back` and the settle blend
    /// take: `Rgba8Unorm` holding already-sRGB bytes.
    colour: Arc<wgpu::Texture>,
    colour_view: wgpu::TextureView,
    hdr: wgpu::TextureView,
    depth: wgpu::TextureView,
    ao: [wgpu::TextureView; 2],
    bloom: [wgpu::TextureView; 2],
    ao_size: (u32, u32),
    bloom_size: (u32, u32),
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
        let ao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raster ao"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ao.wgsl").into()),
        });
        let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raster post"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/post.wgsl").into()),
        });

        let uniform_buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let uniforms = uniform_buffer("raster uniforms", std::mem::size_of::<Uniforms>() as u64);
        let shadow_uniforms =
            uniform_buffer("raster shadow uniforms", std::mem::size_of::<ShadowUniforms>() as u64);
        let ao_size = std::mem::size_of::<AoUniforms>() as u64;
        let ao_uniforms = [
            uniform_buffer("raster ao", ao_size),
            uniform_buffer("raster ao blur x", ao_size),
            uniform_buffer("raster ao blur y", ao_size),
        ];
        let post_size = std::mem::size_of::<PostUniforms>() as u64;
        let post_uniforms = [
            uniform_buffer("raster bright", post_size),
            uniform_buffer("raster bloom blur x", post_size),
            uniform_buffer("raster bloom blur y", post_size),
            uniform_buffer("raster resolve", post_size),
        ];

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
        let caustic_res = scene.caustics.first().map(|q| q.res).unwrap_or((4, 4));
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
                depth_texture_entry(3),
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
                filtering_sampler_entry(6),
                storage_entry(7),
                float_texture_entry(8),
                filtering_sampler_entry(9),
            ],
        });
        let compare_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raster shadow sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raster linear sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raster depth sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
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

        let ao_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raster ao"),
            entries: &[uniform_entry(0), depth_texture_entry(1), float_texture_entry(2)],
        });
        let post_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raster post"),
            entries: &[
                uniform_entry(0),
                float_texture_entry(1),
                float_texture_entry(2),
                filtering_sampler_entry(3),
            ],
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
        let depth_state = |write: bool, compare: wgpu::CompareFunction| {
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias: Default::default(),
            })
        };
        // **The prepass and the scene pass must agree to the bit.** Same
        // vertex entry, same matrices, same swell — so `LessEqual` accepts
        // exactly the fragments the prepass kept, and nothing z-fights with
        // its own depth.
        let prepass_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raster prepass"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(vertex_layout.clone()), Some(instance_layout.clone())],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: depth_state(true, wgpu::CompareFunction::Less),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
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
                    format: HDR,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            // Authored solids are inconsistently wound — the ride tier found
            // this and the cove's are no better — so nothing is culled and
            // the fragment stage flips a normal that faces away.
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: depth_state(false, wgpu::CompareFunction::LessEqual),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raster sky"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sky"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_sky"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR,
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
            depth_stencil: depth_state(true, wgpu::CompareFunction::Less),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let full_screen_vs = |label: &str,
                              module: &wgpu::ShaderModule,
                              bgl: &wgpu::BindGroupLayout,
                              vs: &str,
                              entry: &str,
                              format: wgpu::TextureFormat| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(vs),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
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
            })
        };
        let full_screen = |label: &str,
                           module: &wgpu::ShaderModule,
                           bgl: &wgpu::BindGroupLayout,
                           entry: &str,
                           format: wgpu::TextureFormat| {
            full_screen_vs(label, module, bgl, "vs", entry, format)
        };
        let ao_pipeline = full_screen("raster ao", &ao_shader, &ao_layout, "fs_ao", AO_FORMAT);
        let ao_blur_pipeline =
            full_screen("raster ao blur", &ao_shader, &ao_layout, "fs_blur", AO_FORMAT);
        let bright_pipeline =
            full_screen("raster bright", &post_shader, &post_layout, "fs_bright", HDR);
        let bloom_blur_pipeline =
            full_screen("raster bloom blur", &post_shader, &post_layout, "fs_blur", HDR);
        let resolve_pipeline = full_screen(
            "raster resolve",
            &post_shader,
            &post_layout,
            "fs_resolve",
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raster blit"),
            entries: &[float_texture_entry(0), filtering_sampler_entry(1)],
        });
        let blit_pipeline = full_screen_vs(
            "raster blit",
            &shader,
            &blit_layout,
            "vs_blit",
            "fs_blit",
            target_format,
        );

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
            prepass_pipeline,
            sky_pipeline,
            shadow_pipeline,
            blit_pipeline,
            ao_pipeline,
            ao_blur_pipeline,
            bright_pipeline,
            bloom_blur_pipeline,
            resolve_pipeline,
            uniforms,
            shadow_uniforms,
            ao_uniforms,
            post_uniforms,
            materials,
            probes,
            probe_inside,
            layout,
            bind: None,
            shadow_bind,
            ao_layout,
            ao_binds: Vec::new(),
            post_layout,
            post_binds: Vec::new(),
            blit_layout,
            blit_bind: None,
            compare_sampler,
            linear_sampler,
            blit_sampler: nearest_sampler,
            shadow_map,
            caustic_tex,
            caustic_res,
            meshes,
            targets: None,
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

        let vp = view_proj(&frame.camera, w as f32 / h as f32);
        let inv = invert4(&vp);
        self.write_uniforms(queue, scene, frame, (w, h), &vp, &inv, &fr);

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

        let t = self.targets.as_ref().expect("target");
        let out = t.colour.clone();
        let bind = self.bind.as_ref().expect("bind group");
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("raster"),
        });

        // ── the sun's own depth ────────────────────────────────────────────
        {
            let mut pass = depth_pass(&mut enc, "raster shadow", &self.shadow_map);
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, &self.shadow_bind, &[]);
            draw_meshes(&mut pass, &self.meshes, &counts);
        }
        // ── the camera's, so the occlusion has something to read and the
        //    shading runs once a pixel ──────────────────────────────────────
        {
            let mut pass = depth_pass(&mut enc, "raster prepass", &t.depth);
            pass.set_pipeline(&self.prepass_pipeline);
            pass.set_bind_group(0, bind, &[]);
            draw_meshes(&mut pass, &self.meshes, &counts);
        }
        // ── the occlusion, and two bilateral blurs over it ─────────────────
        if scene.ao_strength > 0.0 && self.ao_binds.len() == 3 {
            for (i, (pipeline, target)) in [
                (&self.ao_pipeline, 0usize),
                (&self.ao_blur_pipeline, 1),
                (&self.ao_blur_pipeline, 0),
            ]
            .into_iter()
            .enumerate()
            {
                let mut pass = colour_pass(&mut enc, "raster ao", &t.ao[target]);
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.ao_binds[i], &[]);
                pass.draw(0..4, 0..1);
            }
        }
        // ── the sky, then the shading over it ──────────────────────────────
        {
            let mut pass = colour_pass(&mut enc, "raster sky", &t.hdr);
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..4, 0..1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raster scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.hdr,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // the sky pass already painted what a ray that hits
                        // nothing shows, so this loads rather than clears
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind, &[]);
            draw_meshes(&mut pass, &self.meshes, &counts);
        }
        // ── the film ───────────────────────────────────────────────────────
        if scene.bloom_strength > 0.0 && self.post_binds.len() == 4 {
            for (i, (pipeline, target)) in [
                (&self.bright_pipeline, 0usize),
                (&self.bloom_blur_pipeline, 1),
                (&self.bloom_blur_pipeline, 0),
            ]
            .into_iter()
            .enumerate()
            {
                let mut pass = colour_pass(&mut enc, "raster bloom", &t.bloom[target]);
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.post_binds[i], &[]);
                pass.draw(0..4, 0..1);
            }
        }
        {
            let mut pass = colour_pass(&mut enc, "raster resolve", &t.colour_view);
            pass.set_pipeline(&self.resolve_pipeline);
            pass.set_bind_group(0, &self.post_binds[3], &[]);
            pass.draw(0..4, 0..1);
        }
        queue.submit([enc.finish()]);
        out
    }

    /// Every uniform block the chain reads, written for this frame.
    #[allow(clippy::too_many_arguments)]
    fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        scene: &Scene,
        frame: &Frame,
        size: (u32, u32),
        vp: &[f32; 16],
        inv: &[f32; 16],
        fr: &super::shadow::Frustum,
    ) {
        let (w, h) = size;
        let d = scene.sun.direction;
        let irr = scene.sun.irradiance;
        let po = scene.probes.origin;
        let dims = scene.probes.dims;
        let m = super::band_to_rgb();
        let sea = scene.sea.unwrap_or(super::Sea::cove(0.0, 0.0, 0.0, 1.0));
        let half_h = (frame.camera.fov_deg.to_radians() * 0.5).tan() as f32;
        let half_w = half_h * w as f32 / h as f32;
        let sky = scene.sky;
        let u = Uniforms {
            view_proj: *vp,
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
            inv_view_proj: *inv,
            sky_a: sky.map_or([2.5, 1.0, 0.0, 0.02], |s| {
                [s.turbidity, s.scale, s.intensity, s.sun_radius]
            }),
            sky_ground: sky.map_or([0.0; 4], |s| {
                [s.ground_albedo[0], s.ground_albedo[1], s.ground_albedo[2], 1.0]
            }),
            air: [
                if sky.is_some() { scene.air.density } else { 0.0 },
                scene.air.scale_h,
                scene.ao_radius_m,
                scene.ao_strength,
            ],
            screen: [w as f32, h as f32, half_w, half_h],
            shadow_m: [
                fr.texel_m * SHADOW as f32,
                fr.depth_m,
                (scene.sun.angular_radius as f32).tan(),
                fr.texel_m,
            ],
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));

        let Some(t) = self.targets.as_ref() else { return };
        let ao = AoUniforms {
            view_proj: *vp,
            inv_view_proj: *inv,
            size: [t.ao_size.0 as f32, t.ao_size.1 as f32, w as f32, h as f32],
            knobs: [scene.ao_radius_m, 0.02, 0.0, 0.0],
            eye: [
                frame.camera.eye.x as f32,
                frame.camera.eye.y as f32,
                frame.camera.eye.z as f32,
                0.0,
            ],
        };
        for (i, step) in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]].into_iter().enumerate() {
            let mut u = ao;
            u.knobs[2] = step[0];
            u.knobs[3] = step[1];
            queue.write_buffer(&self.ao_uniforms[i], 0, bytemuck::bytes_of(&u));
        }

        let lens = [frame.exposure, scene.vignette, half_w, half_h];
        let sigma = scene.bloom_radius_px / BLOOM_SHRINK as f32;
        let full = [w as f32, h as f32, 0.0, 0.0];
        let quarter = |step: [f32; 2]| {
            [t.bloom_size.0 as f32, t.bloom_size.1 as f32, step[0], step[1]]
        };
        let blocks = [
            PostUniforms { lens, bloom: [scene.bloom_threshold, 0.0, sigma, 0.0], size: full },
            PostUniforms {
                lens,
                bloom: [scene.bloom_threshold, 0.0, sigma, 0.0],
                size: quarter([1.0, 0.0]),
            },
            PostUniforms {
                lens,
                bloom: [scene.bloom_threshold, 0.0, sigma, 0.0],
                size: quarter([0.0, 1.0]),
            },
            PostUniforms {
                lens,
                bloom: [scene.bloom_threshold, scene.bloom_strength, sigma, 0.0],
                size: full,
            },
        ];
        for (buffer, block) in self.post_uniforms.iter().zip(blocks.iter()) {
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(block));
        }
    }

    /// The colour target, for a caller that wants to read it back.
    pub fn texture(&self) -> Option<Arc<wgpu::Texture>> {
        self.targets.as_ref().map(|t| t.colour.clone())
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
        if self.targets.as_ref().map(|t| t.size) == Some((w, h)) {
            return;
        }
        let ao_size = ((w / AO_SHRINK).max(1), (h / AO_SHRINK).max(1));
        let bloom_size = ((w / BLOOM_SHRINK).max(1), (h / BLOOM_SHRINK).max(1));
        let attach = |label: &str, size: (u32, u32), format, extra| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | extra,
                view_formats: &[],
            })
        };
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
        let hdr = attach("raster hdr", (w, h), HDR, wgpu::TextureUsages::empty());
        // **The depth is stored, not discarded.** The occlusion pass reads it
        // as a texture between the prepass and the shading, which is the whole
        // reason there is a prepass.
        let depth = attach(
            "raster depth",
            (w, h),
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::empty(),
        );
        let ao_a = attach("raster ao a", ao_size, AO_FORMAT, wgpu::TextureUsages::empty());
        let ao_b = attach("raster ao b", ao_size, AO_FORMAT, wgpu::TextureUsages::empty());
        let bloom_a = attach("raster bloom a", bloom_size, HDR, wgpu::TextureUsages::empty());
        let bloom_b = attach("raster bloom b", bloom_size, HDR, wgpu::TextureUsages::empty());

        let colour_view = colour.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            ..Default::default()
        });
        let hdr_view = hdr.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let ao_views = [ao_a.create_view(&Default::default()), ao_b.create_view(&Default::default())];
        let bloom_views =
            [bloom_a.create_view(&Default::default()), bloom_b.create_view(&Default::default())];

        // The scene's own bind group carries the occlusion, so it is remade
        // whenever the frame changes size.
        self.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raster"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.materials.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.probes.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.shadow_map),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.compare_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(
                        &self.caustic_tex.create_view(&wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::D2Array),
                            ..Default::default()
                        }),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry { binding: 7, resource: self.probe_inside.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&ao_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        }));

        // **No pass reads the texture it writes.** The occlusion goes into
        // `a` while `b` is bound, the first blur into `b` while `a` is bound,
        // the second back into `a`.
        let ao_bind = |i: usize, read: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raster ao"),
                layout: &self.ao_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.ao_uniforms[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&ao_views[read]),
                    },
                ],
            })
        };
        self.ao_binds = vec![ao_bind(0, 1), ao_bind(1, 0), ao_bind(2, 1)];

        let post_bind = |i: usize, bloom: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raster post"),
                layout: &self.post_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.post_uniforms[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&hdr_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&bloom_views[bloom]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                ],
            })
        };
        self.post_binds = vec![post_bind(0, 1), post_bind(1, 0), post_bind(2, 1), post_bind(3, 0)];

        self.blit_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raster blit"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&colour_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
            ],
        }));

        self.targets = Some(Targets {
            size: (w, h),
            colour: Arc::new(colour),
            colour_view,
            hdr: hdr_view,
            depth: depth_view,
            ao: ao_views,
            bloom: bloom_views,
            ao_size,
            bloom_size,
        });
    }
}

/// The occlusion's own format. One channel, eight bits: an ambient term is a
/// number between nought and one that a bilateral blur has already smoothed,
/// and a code of it is a thousandth of a stop.
const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// A depth-only pass over a view, clearing it.
fn depth_pass<'a>(
    enc: &'a mut wgpu::CommandEncoder,
    label: &str,
    view: &'a wgpu::TextureView,
) -> wgpu::RenderPass<'a> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// A colour-only pass over a view, cleared to black. Every full-screen pass in
/// the chain writes every pixel, so the clear is a formality that keeps a
/// tile-based GPU from loading the old contents first.
fn colour_pass<'a>(
    enc: &'a mut wgpu::CommandEncoder,
    label: &str,
    view: &'a wgpu::TextureView,
) -> wgpu::RenderPass<'a> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// Every mesh with an instance, into whichever pass is open.
fn draw_meshes(pass: &mut wgpu::RenderPass<'_>, meshes: &[MeshGpu], counts: &[u32]) {
    for (g, n) in meshes.iter().zip(counts.iter().copied()) {
        if n == 0 || g.n == 0 {
            continue;
        }
        pass.set_vertex_buffer(0, g.vb.slice(..));
        pass.set_vertex_buffer(1, g.inst.slice(..));
        pass.draw(0..g.n, 0..n);
    }
}

/// The inverse of a column-major 4×4, by cofactors.
///
/// The sky pass needs it to turn a clip-space corner back into a world
/// direction, and the occlusion pass to turn a depth texel back into the point
/// behind it. Both must be the inverse of *this frame's* matrix and not of a
/// look-at rebuilt from the eye: the rig can roll, and a reconstruction would
/// quietly level the horizon under the geometry drawn over it.
pub fn invert4(m: &[f32; 16]) -> [f32; 16] {
    let a = |c: usize, r: usize| m[c * 4 + r] as f64;
    let mut inv = [0.0f64; 16];
    let s0 = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
    let s1 = a(0, 0) * a(1, 2) - a(1, 0) * a(0, 2);
    let s2 = a(0, 0) * a(1, 3) - a(1, 0) * a(0, 3);
    let s3 = a(0, 1) * a(1, 2) - a(1, 1) * a(0, 2);
    let s4 = a(0, 1) * a(1, 3) - a(1, 1) * a(0, 3);
    let s5 = a(0, 2) * a(1, 3) - a(1, 2) * a(0, 3);
    let c5 = a(2, 2) * a(3, 3) - a(3, 2) * a(2, 3);
    let c4 = a(2, 1) * a(3, 3) - a(3, 1) * a(2, 3);
    let c3 = a(2, 1) * a(3, 2) - a(3, 1) * a(2, 2);
    let c2 = a(2, 0) * a(3, 3) - a(3, 0) * a(2, 3);
    let c1 = a(2, 0) * a(3, 2) - a(3, 0) * a(2, 2);
    let c0 = a(2, 0) * a(3, 1) - a(3, 0) * a(2, 1);
    let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    if det.abs() < 1e-20 {
        return super::IDENTITY;
    }
    let d = 1.0 / det;
    inv[0] = (a(1, 1) * c5 - a(1, 2) * c4 + a(1, 3) * c3) * d;
    inv[1] = (-a(0, 1) * c5 + a(0, 2) * c4 - a(0, 3) * c3) * d;
    inv[2] = (a(3, 1) * s5 - a(3, 2) * s4 + a(3, 3) * s3) * d;
    inv[3] = (-a(2, 1) * s5 + a(2, 2) * s4 - a(2, 3) * s3) * d;
    inv[4] = (-a(1, 0) * c5 + a(1, 2) * c2 - a(1, 3) * c1) * d;
    inv[5] = (a(0, 0) * c5 - a(0, 2) * c2 + a(0, 3) * c1) * d;
    inv[6] = (-a(3, 0) * s5 + a(3, 2) * s2 - a(3, 3) * s1) * d;
    inv[7] = (a(2, 0) * s5 - a(2, 2) * s2 + a(2, 3) * s1) * d;
    inv[8] = (a(1, 0) * c4 - a(1, 1) * c2 + a(1, 3) * c0) * d;
    inv[9] = (-a(0, 0) * c4 + a(0, 1) * c2 - a(0, 3) * c0) * d;
    inv[10] = (a(3, 0) * s4 - a(3, 1) * s2 + a(3, 3) * s0) * d;
    inv[11] = (-a(2, 0) * s4 + a(2, 1) * s2 - a(2, 3) * s0) * d;
    inv[12] = (-a(1, 0) * c3 + a(1, 1) * c1 - a(1, 2) * c0) * d;
    inv[13] = (a(0, 0) * c3 - a(0, 1) * c1 + a(0, 2) * c0) * d;
    inv[14] = (-a(3, 0) * s3 + a(3, 1) * s1 - a(3, 2) * s0) * d;
    inv[15] = (a(2, 0) * s3 - a(2, 1) * s1 + a(2, 2) * s0) * d;
    // **The accessor and the layout cancel, and the result is copied
    // straight through.** `a(i, j)` reads element `(j, i)` of a column-major
    // matrix, so the cofactors above are those of `Mᵀ` and `inv` holds
    // `(Mᵀ)⁻¹ = (M⁻¹)ᵀ` in the row-major order the classic listing writes.
    // A row-major `(M⁻¹)ᵀ` and a column-major `M⁻¹` are the same sixteen
    // floats in the same order — transposing here once *more* was the bug
    // that drew the sky with a diagonal horizon.
    let mut out = [0.0f32; 16];
    for k in 0..16 {
        out[k] = inv[k] as f32;
    }
    out
}

fn depth_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn float_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn filtering_sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
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
        // 528 was the block before the sky, the air and the film's own pass;
        // the six additions are a 4×4 and five `vec4`s.
        assert_eq!(std::mem::size_of::<Uniforms>(), 528 + 64 + 5 * 16);
        assert_eq!(std::mem::size_of::<AoUniforms>() % 16, 0);
        assert_eq!(std::mem::size_of::<PostUniforms>(), 48);
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

    /// **The inverse is the inverse, in the layout the caller handed over.**
    ///
    /// The sky pass turns a clip-space corner back into a world direction with
    /// it and the occlusion pass turns a depth texel back into a point, so an
    /// inverse that came back transposed is a picture drawn under a different
    /// camera from the geometry over it — which is exactly what it looked
    /// like: a horizon at forty degrees behind a level beach.
    #[test]
    fn the_inverse_undoes_the_projection() {
        let cam = kosm_render::Camera::look_at(
            Point3::new(2.0, -3.0, 1.5),
            Point3::new(0.0, 0.0, 0.5),
            Vec3::new(0.0, 0.0, 1.0),
            50.0,
        );
        let m = view_proj(&cam, 16.0 / 9.0);
        let inv = invert4(&m);
        // M⁻¹ M is the identity, to a float's precision
        for c in 0..4 {
            for r in 0..4 {
                let mut s = 0.0f32;
                for k in 0..4 {
                    s += inv[k * 4 + r] * m[c * 4 + k];
                }
                let want = if c == r { 1.0 } else { 0.0 };
                assert!((s - want).abs() < 1e-4, "({r},{c}) is {s}, not {want}");
            }
        }
        // and the round trip a sky pixel takes: a clip corner back to a world
        // direction, and that direction forward again to the same corner
        for ndc in [[0.0f32, 0.0], [0.7, -0.4], [-1.0, 1.0]] {
            let un = |z: f32| {
                let mut o = [0.0f32; 4];
                for r in 0..4 {
                    o[r] = inv[r] * ndc[0] + inv[4 + r] * ndc[1] + inv[8 + r] * z + inv[12 + r];
                }
                [o[0] / o[3], o[1] / o[3], o[2] / o[3]]
            };
            let (a, b) = (un(0.0), un(1.0));
            let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let p = [a[0] + d[0] * 0.05, a[1] + d[1] * 0.05, a[2] + d[2] * 0.05];
            let mut clip = [0.0f32; 4];
            for r in 0..4 {
                clip[r] = m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r];
            }
            assert!(
                (clip[0] / clip[3] - ndc[0]).abs() < 1e-3
                    && (clip[1] / clip[3] - ndc[1]).abs() < 1e-3,
                "{ndc:?} came back ({}, {})",
                clip[0] / clip[3],
                clip[1] / clip[3]
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
