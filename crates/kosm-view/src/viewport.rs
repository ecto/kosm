//! The viewport: a window, a wgpu surface, and one image on it.
//!
//! Nothing is drawn but the scene's own picture — no panels, no text, no
//! widgets. The scene hands over an RGBA8 image at whatever resolution it can
//! afford and the viewport blits it across the whole surface with linear
//! filtering, so a quarter-size render still fills the window. Input is
//! forwarded as `Event`; the only key the viewport keeps for itself is Escape.

use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// An RGBA8 picture, top row first, `4 * size.0 * size.1` bytes.
pub struct Image {
    pub size: (u32, u32),
    pub rgba: Vec<u8>,
}

/// The keys the viewport passes on. Escape is not among them: it quits.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Space,
    Left,
    Right,
    Home,
}

/// What the window tells the scene about.
pub enum Event {
    /// The window's new size, in physical pixels.
    Resized((u32, u32)),
    /// A drag of the left button, in logical points.
    Drag(f64, f64),
    /// Wheel notches; positive is a scroll up.
    Zoom(f64),
    Key(Key),
}

/// What the window shows. The scene owns its own threads and clock; the
/// viewport asks it for the newest picture once per redraw.
pub trait Scene {
    fn event(&mut self, event: Event);
    /// The newest picture, or `None` to keep the one already on screen.
    fn image(&mut self) -> Option<Image>;
}

/// Open a window and show `scene` in it until it is closed or Escape is hit.
pub fn run(title: &str, size: (u32, u32), scene: impl Scene) -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = Viewport {
        scene,
        title: title.to_owned(),
        size,
        window: None,
        gpu: None,
        dragging: false,
        cursor: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

// ---- the surface ------------------------------------------------------------

/// The device, the surface, and the one pipeline there is.
struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// The scene's image on the GPU, remade whenever its size changes.
    image: Option<(wgpu::Texture, wgpu::BindGroup, (u32, u32))>,
    /// Whether the surface encodes sRGB, which decides the texture's format:
    /// the scene's bytes are already sRGB, so they must be decoded on the way
    /// in exactly when they will be re-encoded on the way out.
    srgb: bool,
}

impl Gpu {
    fn new(event_loop: &ActiveEventLoop, window: Arc<Window>) -> anyhow::Result<Self> {
        // The display handle matters on Wayland/GLES and is ignored elsewhere;
        // winit's is the one the surface will be made against.
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_with_display_handle(Box::new(event_loop.owned_display_handle())).with_env(),
        );
        let surface = instance.create_surface(window.clone())?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))?;
        let (device, queue) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor { label: Some("kosm-view"), ..Default::default() }),
        )?;

        let px = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, px.width.max(1), px.height.max(1))
            .ok_or_else(|| anyhow::anyhow!("the surface is not usable by this adapter"))?;
        let caps = surface.get_capabilities(&adapter);
        // Prefer an sRGB surface: the blit then hands the compositor the same
        // bytes the scene produced.
        if let Some(f) = caps.formats.iter().copied().find(|f| f.is_srgb()) {
            config.format = f;
        }
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);
        let srgb = config.format.is_srgb();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        Ok(Self { surface, device, queue, config, pipeline, layout, sampler, image: None, srgb })
    }

    fn resize(&mut self, px: (u32, u32)) {
        if (px.0, px.1) == (self.config.width, self.config.height) {
            return;
        }
        self.config.width = px.0;
        self.config.height = px.1;
        self.surface.configure(&self.device, &self.config);
    }

    fn upload(&mut self, image: &Image) {
        let (w, h) = image.size;
        if w == 0 || h == 0 || image.rgba.len() < (4 * w * h) as usize {
            return;
        }
        if self.image.as_ref().map(|(_, _, s)| *s) != Some(image.size) {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("scene"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: if self.srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm },
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&texture.create_view(&Default::default())) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            self.image = Some((texture, bind, image.size));
        }
        let Some((texture, _, _)) = &self.image else { return };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &image.rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * w), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }

    fn draw(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            // occluded, timed out, or refused: nothing to present into
            _ => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some((_, bind, _)) = &self.image {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
    }
}

// ---- the event loop ---------------------------------------------------------

/// How often the loop wakes to look for a newer picture. The scene's renderer
/// is far slower than this; the cost of a wake with nothing new is one blit.
const TICK: Duration = Duration::from_millis(16);

struct Viewport<S: Scene> {
    scene: S,
    title: String,
    size: (u32, u32),
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    dragging: bool,
    cursor: Option<(f64, f64)>,
}

impl<S: Scene> ApplicationHandler for Viewport<S> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(LogicalSize::new(self.size.0 as f64, self.size.1 as f64));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("view: could not open a window: {error}");
                return event_loop.exit();
            }
        };
        match Gpu::new(event_loop, window.clone()) {
            Ok(gpu) => self.gpu = Some(gpu),
            Err(error) => {
                eprintln!("view: could not bring up wgpu: {error}");
                return event_loop.exit();
            }
        }
        let px = window.inner_size();
        self.scene.event(Event::Resized((px.width, px.height)));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let scale = self.window.as_ref().map(|w| w.scale_factor()).unwrap_or(1.0);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            // A minimised window is 0×0, which nothing downstream can use.
            WindowEvent::Resized(px) if px.width > 0 && px.height > 0 => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize((px.width, px.height));
                }
                self.scene.event(Event::Resized((px.width, px.height)));
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                let key = match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => return event_loop.exit(),
                    PhysicalKey::Code(KeyCode::Space) => Key::Space,
                    PhysicalKey::Code(KeyCode::ArrowLeft) => Key::Left,
                    PhysicalKey::Code(KeyCode::ArrowRight) => Key::Right,
                    PhysicalKey::Code(KeyCode::Home | KeyCode::KeyR) => Key::Home,
                    _ => return,
                };
                self.scene.event(Event::Key(key));
            }
            WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => {
                self.dragging = state == ElementState::Pressed;
                self.cursor = None;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let now = (position.x / scale, position.y / scale);
                if let (true, Some(was)) = (self.dragging, self.cursor) {
                    self.scene.event(Event::Drag(now.0 - was.0, now.1 - was.1));
                }
                self.cursor = Some(now);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Notches, however the platform reports them: a trackpad's
                // pixels are about fifty to the notch.
                let ticks = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 50.0,
                };
                self.scene.event(Event::Zoom(ticks));
            }
            WindowEvent::RedrawRequested => {
                if let Some(gpu) = &mut self.gpu {
                    if let Some(image) = self.scene.image() {
                        gpu.upload(&image);
                    }
                    gpu.draw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + TICK));
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}
