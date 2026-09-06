//! Native space viewer. All rendering consumes the same recorded SI state.
use eframe::{
    egui,
    egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor, wgpu},
};
use kosm_spike::space::{Snapshot, SpaceScene};
use std::time::Instant;
#[path = "space_assets.rs"]
mod assets;
#[path = "space_atmosphere.rs"]
mod atmosphere;
use tang::Vec3 as V;
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    eye: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    orbit: [f32; 4],
    sun: [f32; 4],
    axis_x: [f32; 4],
    axis_y: [f32; 4],
    axis_z: [f32; 4],
    bus: [f32; 4],
    panels: [f32; 4],
    settings: [f32; 4],
}
fn v4(v: V<f64>, w: f32) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w]
}
struct Resources {
    pipeline: wgpu::RenderPipeline,
    buffer: wgpu::Buffer,
    bind: wgpu::BindGroup,
    format: wgpu::TextureFormat,
    atlas: assets::Atlas,
}
impl Resources {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        earth_assets: &std::path::Path,
    ) -> anyhow::Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("space transport"),
            source: wgpu::ShaderSource::Wgsl(include_str!("space.wgsl").into()),
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("space state"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tex = assets::texture(
            device,
            queue,
            &earth_assets.join("earth-low.png"),
            "NASA global fallback",
        )?;
        let atlas = assets::Atlas::new(device, queue, earth_assets.to_owned());
        let cloud = assets::clouds(device, queue, &earth_assets.join("clouds-8k.png"))?;
        let noise = assets::cloud_noise(device, queue, &earth_assets.join("cloud-noise.raw"))?;
        let transmission = atmosphere::texture(device, queue);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            ..Default::default()
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("space"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &tex.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&atlas.texture.create_view(
                        &wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::D2Array),
                            ..Default::default()
                        },
                    )),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: atlas.map.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(
                        &cloud.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(
                        &transmission.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(
                        &noise.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&noise_sampler),
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("space"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("space"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Ok(Self {
            pipeline,
            buffer,
            bind,
            format,
            atlas,
        })
    }
}
struct Callback {
    uniforms: Uniforms,
    shot: Option<std::path::PathBuf>,
    size: (u32, u32),
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
        let r = resources.get_mut::<Resources>().unwrap();
        r.atlas.update(queue, &self.uniforms);
        queue.write_buffer(&r.buffer, 0, bytemuck::bytes_of(&self.uniforms));
        if let Some(path) = &self.shot {
            eprintln!("Earth resident tiles: {} / 32", r.atlas.resident_count());
            let (w, h) = self.size;
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("space capture"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: r.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("space capture"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
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
                });
                pass.set_pipeline(&r.pipeline);
                pass.set_bind_group(0, &r.bind, &[]);
                pass.draw(0..3, 0..1);
            }
            queue.submit([enc.finish()]);
            let mut img = crate::live::read_back(device, queue, &tex, (w, h));
            if matches!(
                r.format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
            ) {
                for p in img.pixels_mut() {
                    p.0.swap(0, 2);
                }
            }
            match img.save(path) {
                Ok(()) => eprintln!("space capture {}", path.display()),
                Err(e) => eprintln!("capture failed: {e}"),
            }
        }
        Vec::new()
    }
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let r = resources.get::<Resources>().unwrap();
        pass.set_pipeline(&r.pipeline);
        pass.set_bind_group(0, &r.bind, &[]);
        pass.draw(0..3, 0..1);
    }
}
struct App {
    scene: SpaceScene,
    state: Snapshot,
    history: Vec<Snapshot>,
    cursor: usize,
    playing: bool,
    rate: f64,
    last: Instant,
    az: f64,
    el: f64,
    distance: f64,
    exposure: f32,
    overview: bool,
    show_ui: bool,
    shot: Option<std::path::PathBuf>,
    shot_tick: u32,
    render_started: Instant,
    save: bool,
    message: String,
    initial_energy: f64,
    trajectory: Vec<V<f64>>,
    predicted_at: f64,
}
impl App {
    fn new(cc: &eframe::CreationContext<'_>, scene: SpaceScene) -> anyhow::Result<Self> {
        let rs = cc.wgpu_render_state.as_ref().expect("space requires wgpu");
        let earth_assets = assets::root()?;
        rs.renderer
            .write()
            .callback_resources
            .insert(Resources::new(
                &rs.device,
                &rs.queue,
                rs.target_format,
                &earth_assets,
            )?);
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(6, 10, 15);
        visuals.window_fill = egui::Color32::from_rgba_unmultiplied(8, 14, 21, 235);
        visuals.override_text_color = Some(egui::Color32::from_rgb(197, 211, 222));
        visuals.selection.bg_fill = egui::Color32::from_rgb(36, 81, 104);
        cc.egui_ctx.set_visuals(visuals);
        let mut state = Snapshot::initial(&scene);
        if let Some(t) = std::env::args().find_map(|a| {
            a.strip_prefix("--space-time=")
                .and_then(|s| s.parse::<f64>().ok())
        }) {
            state.advance(&scene, t.clamp(0.0, 86400.0));
        }
        let e = state.energy(&scene);
        Ok(Self {
            scene,
            state,
            history: vec![state],
            cursor: 0,
            playing: true,
            rate: 30.0,
            last: Instant::now(),
            az: -1.04,
            el: 0.31,
            distance: 23.0,
            exposure: 1.25,
            overview: std::env::args().any(|a| a == "--space-overview"),
            show_ui: true,
            shot: std::env::args().find_map(|a| a.strip_prefix("--shot=").map(Into::into)),
            shot_tick: 0,
            render_started: Instant::now(),
            save: false,
            message: String::new(),
            initial_energy: e,
            trajectory: Vec::new(),
            predicted_at: f64::NEG_INFINITY,
        })
    }
    fn displayed(&self) -> Snapshot {
        self.history[self.cursor]
    }
    fn branch(&mut self) {
        self.state = self.displayed();
        self.history.truncate(self.cursor + 1);
        self.predicted_at = f64::NEG_INFINITY;
        self.initial_energy = self.state.energy(&self.scene);
    }
    fn burn(&mut self, sign: f64) {
        self.branch();
        self.state.impulse(self.state.v, sign * 10.0);
        self.initial_energy = self.state.energy(&self.scene);
        self.history.push(self.state);
        self.cursor = self.history.len() - 1;
        self.message = format!(
            "{} 10 m/s impulse at T+{:.1}s",
            if sign > 0.0 { "Prograde" } else { "Retrograde" },
            self.state.t
        );
    }
    fn uniforms(&self, aspect: f32) -> Uniforms {
        let s = self.displayed();
        let radial = s.r.normalize();
        let normal = s.r.cross(&s.v).normalize();
        let along = normal.cross(&radial);
        let local = along * (self.az.cos() * self.el.cos())
            + normal * (self.az.sin() * self.el.cos())
            + radial * self.el.sin();
        let eye = local * self.distance;
        let mut f = (-eye).normalize();
        let mut right = f.cross(&radial).normalize();
        let mut up = right.cross(&f);
        let mut origin = s.r / 1000.0 + eye / 1000.0;
        if self.overview {
            origin = local * (self.scene.radius / 1000.0 * 2.8);
            f = (-origin).normalize();
            right = f.cross(&V::new(0.0, 0.0, 1.0)).normalize();
            up = right.cross(&f);
        }
        let axes = s.axes();
        let sun = V::new(0.7, 0.3, 0.64).normalize();
        let body_to_world = |v: V<f64>| {
            let initial = Snapshot::initial(&self.scene);
            let z = initial.r.normalize();
            let y = initial.r.cross(&initial.v).normalize();
            let x = y.cross(&z);
            x * v.x + y * v.y + z * v.z
        };
        Uniforms {
            eye: v4(eye, aspect),
            forward: v4(f, 0.65),
            right: v4(right, self.exposure),
            up: v4(up, 0.0),
            orbit: v4(origin, (self.scene.radius / 1000.0) as f32),
            sun: v4(sun, (s.t * 7.292115e-5) as f32),
            axis_x: v4(body_to_world(axes[0]), 0.0),
            axis_y: v4(body_to_world(axes[1]), 0.0),
            axis_z: v4(body_to_world(axes[2]), 0.0),
            bus: [
                self.scene.bus[0] as f32 * 0.5,
                self.scene.bus[1] as f32 * 0.5,
                self.scene.bus[2] as f32 * 0.5,
                0.0,
            ],
            panels: [
                self.scene.panel_span as f32,
                self.scene.panel_chord as f32,
                self.scene.panel_thickness as f32,
                0.0,
            ],
            settings: [if self.overview { 1.0 } else { 0.0 }, s.t as f32, 0.0, 0.0],
        }
    }
}
impl eframe::App for App {
    fn ui(&mut self, root: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = root.ctx().clone();
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f64().min(0.1);
        self.last = now;
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.playing = !self.playing;
            if self.playing {
                self.branch();
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::H)) {
            self.show_ui = !self.show_ui;
        }
        if self.playing && self.shot.is_none() {
            if self.cursor + 1 < self.history.len() {
                self.branch();
            }
            self.state.advance(&self.scene, dt * self.rate);
            if self.state.r.norm() <= self.scene.radius + 80000.0 {
                self.playing = false;
                self.message =
                    "Reached 80 km: propagation paused at atmosphere model boundary.".into();
            }
            self.history.push(self.state);
            self.cursor = self.history.len() - 1;
            if self.history.len() > 36000 {
                self.history.drain(0..6000);
                self.cursor = self.history.len() - 1;
            }
        }
        let snapshot = self.displayed();
        if self.overview && (snapshot.t - self.predicted_at).abs() > 10.0 {
            let mut predicted = snapshot;
            let period = snapshot.elements(&self.scene).2;
            let duration = if period.is_finite() {
                period.min(18000.0)
            } else {
                6000.0
            };
            self.trajectory.clear();
            for _ in 0..=180 {
                self.trajectory.push(predicted.r);
                predicted.advance(&self.scene, duration / 180.0);
            }
            self.predicted_at = snapshot.t;
        }
        if self.show_ui {
            egui::Panel::top("space header")
                .exact_size(62.0)
                .show(root, |ui| {
                    ui.horizontal_centered(|ui| {
                        ui.add_space(19.0);
                        ui.label(
                            egui::RichText::new("k o s m")
                                .size(25.0)
                                .color(egui::Color32::WHITE),
                        );
                        ui.add_space(30.0);
                        ui.label(
                            egui::RichText::new("O R B I T A L   /   0 0 1")
                                .monospace()
                                .size(12.0),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add_space(15.0);
                            ui.label(
                                egui::RichText::new("EARTH  ·  LOW ORBIT")
                                    .monospace()
                                    .size(11.0)
                                    .color(egui::Color32::from_rgb(117, 172, 192)),
                            );
                        });
                    });
                });
            egui::Panel::bottom("space time")
                .exact_size(84.0)
                .show(root, |ui| {
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        ui.add_space(12.0);
                        if ui
                            .button(if self.playing {
                                "Ⅱ  Pause"
                            } else {
                                "▶  Resume"
                            })
                            .clicked()
                        {
                            self.playing = !self.playing;
                            if self.playing {
                                self.branch();
                            }
                        }
                        for speed in [1.0, 30.0, 100.0, 500.0] {
                            if ui
                                .selectable_label(self.rate == speed, format!("{speed:.0}×"))
                                .clicked()
                            {
                                self.rate = speed;
                            }
                        }
                        ui.add_space(20.0);
                        ui.monospace(format!(
                            "T + {:02}:{:02}:{:05.2}",
                            (snapshot.t / 3600.0) as u32,
                            (snapshot.t / 60.0) as u32 % 60,
                            snapshot.t % 60.0
                        ));
                        if ui.button("↺ Reset").clicked() {
                            self.state = Snapshot::initial(&self.scene);
                            self.history = vec![self.state];
                            self.cursor = 0;
                            self.initial_energy = self.state.energy(&self.scene);
                            self.message.clear();
                        }
                        if ui.button("Save frame").clicked() {
                            self.save = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.add_space(14.0);
                        ui.label(egui::RichText::new("RECORDING").monospace().size(10.0));
                        let end = self.history.len() - 1;
                        let slider = egui::Slider::new(&mut self.cursor, 0..=end).show_value(false);
                        if ui
                            .add_sized([ui.available_width() - 140.0, 20.0], slider)
                            .changed()
                        {
                            self.playing = false;
                        }
                        ui.monospace(format!("{} frames", self.history.len()));
                    });
                });
            egui::Panel::right("space inspector")
                .exact_size(250.0)
                .resizable(false)
                .show(root, |ui| {
                    ui.add_space(20.0);
                    ui.heading("KESTREL–01");
                    ui.label(
                        egui::RichText::new("FREE-FLYING OBSERVATORY")
                            .monospace()
                            .size(10.0)
                            .color(egui::Color32::from_rgb(115, 143, 161)),
                    );
                    ui.add_space(20.0);
                    let (peri, apo, period) = snapshot.elements(&self.scene);
                    for (label, value) in [
                        (
                            "ALTITUDE",
                            format!("{:.2} km", (snapshot.r.norm() - self.scene.radius) / 1000.0),
                        ),
                        (
                            "ORBITAL SPEED",
                            format!("{:.4} km/s", snapshot.v.norm() / 1000.0),
                        ),
                        (
                            "PERIGEE / APOGEE",
                            format!("{:.1} / {:.1} km", peri / 1000.0, apo / 1000.0),
                        ),
                        ("PERIOD", format!("{:.2} min", period / 60.0)),
                        (
                            "ANGULAR RATE",
                            format!("{:.3} °/s", snapshot.omega.norm().to_degrees()),
                        ),
                        ("TOTAL Δv", format!("{:.1} m/s", snapshot.delta_v)),
                    ] {
                        ui.label(
                            egui::RichText::new(label)
                                .monospace()
                                .size(10.0)
                                .color(egui::Color32::from_rgb(105, 139, 160)),
                        );
                        ui.label(egui::RichText::new(value).size(19.0));
                        ui.add_space(13.0);
                    }
                    ui.separator();
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("MANEUVER  /  IMPULSIVE")
                            .monospace()
                            .size(10.0),
                    );
                    ui.horizontal(|ui| {
                        if ui.button("+ Prograde").clicked() {
                            self.burn(1.0);
                        }
                        if ui.button("− Retrograde").clicked() {
                            self.burn(-1.0);
                        }
                    });
                    ui.add_space(14.0);
                    ui.selectable_value(&mut self.overview, false, "Spacecraft camera");
                    ui.selectable_value(&mut self.overview, true, "Earth overview");
                    ui.add(egui::Slider::new(&mut self.exposure, 0.3..=3.0).text("Exposure"));
                    ui.add_space(10.0);
                    ui.small("Drag to orbit · Scroll to dolly\nSpace to pause · H to hide UI");
                    ui.add_space(14.0);
                    ui.small(&self.message);
                    ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                        ui.add_space(15.0);
                        ui.label(
                            egui::RichText::new("NASA BLUE MARBLE / JULY 2004")
                                .monospace()
                                .size(9.0),
                        );
                        ui.small(
                            "J₂ gravity · f64 RK4 · ≤1 s steps\nTorque-free rigid body · SI units",
                        );
                        ui.small(format!(
                            "Relative ΔE  {:.2e}",
                            (snapshot.energy(&self.scene) - self.initial_energy)
                                / self.initial_energy.abs()
                        ));
                    });
                });
        }
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(root, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());
                if response.dragged() {
                    let d = ui.input(|i| i.pointer.delta());
                    self.az -= d.x as f64 * 0.005;
                    self.el = (self.el + d.y as f64 * 0.005).clamp(-1.4, 1.4);
                }
                if response.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    self.distance =
                        (self.distance * (-scroll as f64 * 0.002).exp()).clamp(14.0, 150.0);
                }
                self.shot_tick += 1;
                let shot = if self.shot_tick == 120 {
                    if self.shot.is_some() {
                        eprintln!(
                            "space warmup: 120 frames in {:.2}s",
                            self.render_started.elapsed().as_secs_f64()
                        );
                    }
                    self.shot.clone()
                } else if self.save {
                    self.save = false;
                    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out");
                    let _ = std::fs::create_dir_all(&dir);
                    let p = dir.join(format!("space-{:.0}.png", snapshot.t));
                    self.message = format!("Saved {}", p.display());
                    Some(p)
                } else {
                    None
                };
                let pixels = ctx.pixels_per_point();
                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    Callback {
                        uniforms: self.uniforms(rect.aspect_ratio()),
                        shot,
                        size: (
                            (rect.width() * pixels) as u32,
                            (rect.height() * pixels) as u32,
                        ),
                    },
                ));
                if self.overview {
                    let camera = self.uniforms(rect.aspect_ratio());
                    let vec = |a: [f32; 4]| V::new(a[0] as f64, a[1] as f64, a[2] as f64);
                    let eye = vec(camera.orbit) * 1000.0;
                    let project = |p: V<f64>| -> Option<egui::Pos2> {
                        let delta = p - eye;
                        let length = delta.norm();
                        let direction = delta / length;
                        let b = eye.dot(&direction);
                        let h = b * b - eye.dot(&eye) + self.scene.radius.powi(2);
                        if h > 0.0 {
                            let t = -b - h.sqrt();
                            if t > 0.0 && t < length - 100.0 {
                                return None;
                            }
                        }
                        let depth = delta.dot(&vec(camera.forward));
                        if depth <= 0.0 {
                            return None;
                        }
                        let x = delta.dot(&vec(camera.right))
                            / (depth * camera.forward[3] as f64 * rect.aspect_ratio() as f64);
                        let y = delta.dot(&vec(camera.up)) / (depth * camera.forward[3] as f64);
                        Some(
                            rect.center()
                                + egui::vec2(
                                    x as f32 * rect.width() * 0.5,
                                    -y as f32 * rect.height() * 0.5,
                                ),
                        )
                    };
                    for pair in self.trajectory.windows(2) {
                        if let (Some(a), Some(b)) = (project(pair[0]), project(pair[1])) {
                            ui.painter().line_segment(
                                [a, b],
                                egui::Stroke::new(1.2, egui::Color32::from_rgb(181, 137, 83)),
                            );
                        }
                    }
                    if let Some(p) = project(snapshot.r) {
                        ui.painter().circle_filled(p, 4.0, egui::Color32::WHITE);
                        ui.painter().text(
                            p + egui::vec2(10., -10.),
                            egui::Align2::LEFT_BOTTOM,
                            "KESTREL–01",
                            egui::FontId::monospace(10.),
                            egui::Color32::WHITE,
                        );
                    }
                }
                if self.show_ui {
                    ui.painter().text(
                        rect.left_top() + egui::vec2(26.0, 25.0),
                        egui::Align2::LEFT_TOP,
                        if self.overview {
                            "EARTH / INERTIAL VIEW"
                        } else {
                            "CAM 01 / CHASE"
                        },
                        egui::FontId::monospace(11.0),
                        egui::Color32::from_rgb(139, 162, 177),
                    );
                    ui.painter().text(
                        rect.left_bottom() + egui::vec2(26.0, -24.0),
                        egui::Align2::LEFT_BOTTOM,
                        "SOLAR ILLUMINATION  /  RAYLEIGH + MIE",
                        egui::FontId::monospace(10.0),
                        egui::Color32::from_rgb(117, 144, 165),
                    );
                }
            });
        if self.shot.is_some() && self.shot_tick > 123 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        ctx.request_repaint();
    }
}
pub fn run() -> eframe::Result<()> {
    let scene = SpaceScene::load().map_err(|e| eframe::Error::AppCreation(e.into()))?;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1500.0, 900.0])
            .with_title("Kosm / Orbital"),
        ..Default::default()
    };
    eframe::run_native(
        "Kosm / Orbital",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, scene)?))),
    )
}
