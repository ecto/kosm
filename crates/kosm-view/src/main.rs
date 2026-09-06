//! Kosm view: the window.
//!
//! The simulation runs on its own thread and hands over a snapshot per frame.
//! The window keeps every snapshot, so the timeline is a recording: play it,
//! pause it, scrub it, inspect any frame. Two renderers look at the same
//! frame: the live tier on the GPU (`live.rs`), and the reference tracer the
//! CLI uses, one button away, side by side with the live frame when paused.
//! Export is the CLI's render, as a button.

mod live;
mod space;

use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use eframe::egui;
use eframe::egui_wgpu;
use kosm_spike::pool::{self, Caustic, PoolScene, PoolSimulation, PoolSnapshot};
use tang::Vec3 as V;

/// One frame of the recording: everything a renderer needs, nothing the
/// solver needs.
#[derive(Clone)]
struct Frame {
    snapshot: PoolSnapshot,
    caustic: Caustic,
    sim_ms: u128,
}

impl Frame {
    fn take(simulation: &mut PoolSimulation, sim_ms: u128) -> Self {
        let snapshot = simulation.take_snapshot();
        let caustic = pool::caustic_for_geometry(&snapshot.surface, snapshot.geometry, 0.01);
        Self { snapshot, caustic, sim_ms }
    }

    fn melon_centre(&self) -> V<f64> {
        self.melon.centre
    }

    fn live(&self) -> Arc<live::LiveFrame> {
        Arc::new(live::LiveFrame {
            geometry: self.geometry,
            surface: self.surface.clone(),
            caustic: self.caustic.clone(),
            melon_centre: self.melon_centre(),
            melon_axis: self.melon.axis,
            melon_axes: self.melon.axes,
            droplets: self.droplets.clone(),
        })
    }
}

impl std::ops::Deref for Frame {
    type Target = PoolSnapshot;

    fn deref(&self) -> &Self::Target {
        &self.snapshot
    }
}

/// What the simulation thread is doing, for the window to show.
type Status = std::sync::Arc<std::sync::Mutex<String>>;

fn simulate(tx: mpsc::Sender<Frame>, status: Status, splash: bool, frames: usize) {
    let set = |s: String| {
        if let Ok(mut g) = status.lock() {
            *g = s;
        }
    };
    let scene = match PoolScene::reference().map(PoolScene::with_env_overrides) {
        Ok(scene) => scene,
        Err(error) => {
            set(format!("could not load the pool scene: {error}"));
            return;
        }
    };
    let mut drop = PoolSimulation::from_scene(&scene);
    if splash {
        let water = scene.water;
        let h = water.cell_size;
        let t0 = Instant::now();
        set(format!(
            "filling the water at {:.0} mm and settling it for {} s of sim time — about a minute on the GPU",
            h * 1000.0,
            water.settle_seconds
        ));
        drop = match drop.with_water_config(water) {
            Ok(drop) => drop,
            Err(error) => {
                set(format!("could not start the fine-water solver: {error}"));
                return;
            }
        };
        set(format!("settled {} particles in {:.0} s; simulating", drop.water_particle_count(), t0.elapsed().as_secs_f64()));
    }
    let steps_per_frame = (1.0 / drop.recording_fps() / drop.timestep()).round() as usize;
    for k in 0..frames {
        let t0 = Instant::now();
        for _ in 0..steps_per_frame {
            drop.step();
        }
        drop.read_water();
        let ms = t0.elapsed().as_millis();
        set(format!("frame {k} of {frames} · {ms} ms per frame · melon z {:+.2} m", drop.centre().z));
        let f = Frame::take(&mut drop, ms);
        if tx.send(f).is_err() {
            break;
        }
    }
}

struct App {
    rx: mpsc::Receiver<Frame>,
    status: Status,
    frames: Vec<Frame>,
    cursor: usize,
    playing: bool,
    follow: bool,
    annotate: bool,
    splash: bool,
    live_on: bool,
    camera: live::Camera,
    orbit: (f64, f64, f64), // azimuth, elevation, distance
    reference: Option<egui::TextureHandle>,
    reference_for: Option<usize>,
    reference_ms: u128,
    reference_size: (u32, u32),
    want_reference: bool,
    /// `--shot=<path>`: write a window capture once frames arrive, then quit.
    shot: Option<(std::path::PathBuf, u32)>,
    shot_now: Option<std::path::PathBuf>,
    live_frame: Option<(usize, Arc<live::LiveFrame>)>,
    last_tick: Instant,
    ui_fps: f32,
    export: Option<std::process::Child>,
    play_t0: f64,
    play_from: usize,
    t_start: Instant,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, rx: mpsc::Receiver<Frame>, status: Status, splash: bool) -> Self {
        if let Some(rs) = &cc.wgpu_render_state {
            let res = live::Resources::new(&rs.device, rs.target_format);
            rs.renderer.write().callback_resources.insert(res);
        }
        let mut app = Self {
            rx,
            status,
            frames: Vec::new(),
            cursor: 0,
            playing: true,
            follow: true,
            annotate: true,
            splash,
            live_on: cc.wgpu_render_state.is_some(),
            camera: live::Camera { eye: V::new(-1.42, -1.22, 0.31), target: V::new(0.02, 0.12, -0.06), vfov: 0.9 },
            orbit: (-2.43, 0.19, 1.9),
            reference: None,
            reference_for: None,
            reference_ms: 0,
            reference_size: (0, 0),
            want_reference: false,
            shot: std::env::args().find_map(|a| a.strip_prefix("--shot=").map(|v| (v.into(), 0))),
            shot_now: None,
            live_frame: None,
            last_tick: Instant::now(),
            ui_fps: 0.0,
            export: None,
            play_t0: 0.0,
            play_from: 0,
            t_start: Instant::now(),
        };
        app.apply_orbit();
        app
    }

    fn apply_orbit(&mut self) {
        let (az, el, dist) = self.orbit;
        let t = self.camera.target;
        self.camera.eye = t + V::new(dist * el.cos() * az.cos(), dist * el.cos() * az.sin(), dist * el.sin());
    }

    /// The reference tracer on the current frame, at the live camera.
    fn render_reference(&mut self, ctx: &egui::Context, size: (u32, u32)) {
        let Some(frame) = self.frames.get(self.cursor) else { return };
        let (w, h) = (size.0.min(960).max(64), ((size.0.min(960).max(64)) as f64 * size.1 as f64 / size.0.max(1) as f64).max(36.0) as u32);
        let t0 = Instant::now();
        let view = self.camera.view(w, h);
        let mut img = pool::render_snapshot(&view, frame, &frame.caustic);
        if self.annotate {
            annotate(&mut img, &view, frame);
        }
        self.reference_ms = t0.elapsed().as_millis();
        let color = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], img.as_raw());
        match &mut self.reference {
            Some(t) => t.set(color, egui::TextureOptions::LINEAR),
            None => self.reference = Some(ctx.load_texture("reference", color, egui::TextureOptions::LINEAR)),
        }
        self.reference_for = Some(self.cursor);
        self.reference_size = (w, h);
    }
}

/// Scene annotation: a marker on the melon and a stem along its velocity.
fn annotate(img: &mut image::RgbaImage, view: &pool::View, f: &Frame) {
    let fwd = (view.target - view.eye).normalize();
    let right = fwd.cross(&V::new(0.0, 0.0, 1.0)).normalize();
    let up = right.cross(&fwd);
    let fy = 0.5 * view.height as f64 / (0.5 * view.vfov).tan();
    let project = |p: V<f64>| {
        let r = p - view.eye;
        let z = r.dot(&fwd);
        if z <= 1e-6 {
            return None;
        }
        Some(((r.dot(&right) / z) * fy + view.width as f64 / 2.0, view.height as f64 / 2.0 - (r.dot(&up) / z) * fy))
    };
    let c = f.melon_centre();
    if let Some((x, y)) = project(c) {
        for k in -6..=6i64 {
            for (px, py) in [(x as i64 + k, y as i64), (x as i64, y as i64 + k)] {
                if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                    img.put_pixel(px as u32, py as u32, image::Rgba([255, 230, 60, 255]));
                }
            }
        }
        let vel = f.melon_velocity;
        if let Some((x2, y2)) = project(c + vel * 0.15) {
            let n = 40;
            for i in 0..=n {
                let t = i as f64 / n as f64;
                let (px, py) = ((x + (x2 - x) * t) as i64, (y + (y2 - y) * t) as i64);
                if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                    img.put_pixel(px as u32, py as u32, image::Rgba([255, 120, 40, 255]));
                }
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, root: &mut egui::Ui, _f: &mut eframe::Frame) {
        let ctx = root.ctx().clone();
        while let Ok(f) = self.rx.try_recv() {
            self.frames.push(f);
        }
        let n = self.frames.len();
        if self.playing && n > 0 {
            if self.follow {
                self.cursor = n - 1;
            } else {
                let t = ctx.input(|i| i.time);
                let want = ((t - self.play_t0) * 30.0) as usize + self.play_from;
                self.cursor = want.min(n - 1);
            }
        }
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32().max(1e-3);
        self.last_tick = now;
        self.ui_fps = 0.9 * self.ui_fps + 0.1 / dt;

        egui::Panel::top("bar").show(root, |ui| {
            ui.horizontal(|ui| {
                ui.heading(if self.splash { "the splash" } else { "the pool" });
                ui.separator();
                if ui.button(if self.playing { "⏸ pause" } else { "▶ play" }).clicked() {
                    self.playing = !self.playing;
                    if self.playing {
                        self.follow = self.cursor + 1 >= n;
                        self.play_t0 = ctx.input(|i| i.time);
                        self.play_from = self.cursor;
                    }
                }
                ui.checkbox(&mut self.follow, "follow live");
                ui.checkbox(&mut self.annotate, "annotate");
                ui.checkbox(&mut self.live_on, "live tier");
                ui.separator();
                if n > 0 {
                    let mut c = self.cursor;
                    if ui.add(egui::Slider::new(&mut c, 0..=n - 1).text("frame")).changed() {
                        self.cursor = c;
                        self.playing = false;
                        self.follow = false;
                    }
                }
                ui.separator();
                let running = self.export.as_mut().map(|c| c.try_wait().ok().flatten().is_none()).unwrap_or(false);
                if running {
                    ui.label("exporting…");
                } else if ui.button("export mp4").clicked() {
                    let mode = if self.splash { "--splash" } else { "--pool" };
                    self.export = std::process::Command::new(std::env::current_exe().map(|p| p.with_file_name("kosm-spike")).unwrap_or_else(|_| "kosm-spike".into()))
                        .args([mode, "150"])
                        .spawn()
                        .ok();
                }
            });
        });

        egui::Panel::right("inspector").default_size(300.0).show(root, |ui| {
            ui.heading("inspector");
            ui.label(format!("ui {:.0} fps · {}", self.ui_fps, if self.live_on { "live tier" } else { "live tier off" }));
            ui.label(format!("recorded {n} frames"));
            if let Ok(st) = self.status.lock() {
                ui.label(st.as_str());
            }
            if let Some(f) = self.frames.get(self.cursor) {
                ui.separator();
                ui.label(format!("t = {:.3} s   sim {} ms/frame", f.time, f.sim_ms));
                ui.label(format!("melon  z {:+.3} m   vz {:+.2} m/s", f.melon.centre.z, f.melon_velocity.z));
                ui.label(format!("       x {:+.3}  y {:+.3}", f.melon.centre.x, f.melon.centre.y));
                ui.label(format!("fluid force  ({:+.0}, {:+.0}, {:+.0}) N", f.fluid_force.x, f.fluid_force.y, f.fluid_force.z));
                ui.label(format!("rings {}   drops {}", f.surface.rings.len(), f.droplets.len()));
                if f.water_particles > 0 {
                    ui.label(format!("water particles {}", f.water_particles));
                }
                if let Some(g) = &f.surface.grid {
                    let (lo, hi) = g.z.iter().fold((f64::MAX, f64::MIN), |a, z| (a.0.min(*z), a.1.max(*z)));
                    ui.label(format!("surface {}×{} @ {:.0} mm   z {:+.3}…{:+.3}", g.nx, g.ny, g.cell * 1e3, lo, hi));
                }
                ui.separator();
                ui.label("camera: drag to orbit, scroll to zoom");
                ui.label(format!("eye ({:+.2}, {:+.2}, {:+.2})", self.camera.eye.x, self.camera.eye.y, self.camera.eye.z));
                ui.separator();
                ui.label("reference tier");
                if self.playing {
                    ui.label("pause to render the reference");
                } else if ui.button("render the reference beside it").clicked() {
                    self.want_reference = true;
                    self.reference_for = None;
                }
                if self.reference_for == Some(self.cursor) {
                    ui.label(format!("reference {}×{} in {} ms", self.reference_size.0, self.reference_size.1, self.reference_ms));
                }
                ui.separator();
                ui.label(format!("melon axes {:?} m", f.melon.axes));
                ui.label(format!(
                    "pool {}×{} m, {} m deep",
                    2.0 * f.geometry.half_extents[0],
                    2.0 * f.geometry.half_extents[1],
                    f.geometry.depth
                ));
            }
        });

        // the reference, when asked for and paused
        if self.want_reference && !self.playing && n > 0 && self.reference_for != Some(self.cursor) {
            let avail = ctx.content_rect().size();
            let w = ((avail.x - 320.0) * 0.5).max(240.0) as u32;
            let h = (w as f32 / (16.0 / 9.0)) as u32;
            self.render_reference(&ctx, (w, h));
        }
        if self.playing {
            self.want_reference = false;
        }

        egui::CentralPanel::default().show(root, |ui| {
            let avail = ui.available_size();
            let show_ref = !self.playing && self.reference_for == Some(self.cursor) && self.reference.is_some();
            let cols = if show_ref { 2.0 } else { 1.0 };
            let aspect = 16.0 / 9.0;
            let cell_w = (avail.x - 8.0 * (cols - 1.0)) / cols;
            let size = if cell_w / avail.y > aspect { egui::vec2(avail.y * aspect, avail.y) } else { egui::vec2(cell_w, cell_w / aspect) };
            ui.horizontal(|ui| {
                let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::drag());
                if resp.dragged() {
                    let d = resp.drag_delta();
                    self.orbit.0 -= d.x as f64 * 0.005;
                    self.orbit.1 = (self.orbit.1 + d.y as f64 * 0.005).clamp(0.02, 1.5);
                    self.apply_orbit();
                    self.reference_for = None;
                }
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if resp.hovered() && scroll.abs() > 0.0 {
                    self.orbit.2 = (self.orbit.2 * (1.0 - scroll as f64 * 0.002)).clamp(0.4, 8.0);
                    self.apply_orbit();
                    self.reference_for = None;
                }
                if self.live_on {
                    if let Some(frame) = self.frames.get(self.cursor) {
                        if self.live_frame.as_ref().map(|(i, _)| *i != self.cursor).unwrap_or(true) {
                            self.live_frame = Some((self.cursor, frame.live()));
                        }
                        let lf = self.live_frame.as_ref().unwrap().1.clone();
                        let ppp = ctx.pixels_per_point();
                        let cb = egui_wgpu::Callback::new_paint_callback(
                            rect,
                            live::LiveCallback {
                                frame: lf,
                                camera: self.camera,
                                size: ((size.x * ppp) as u32, (size.y * ppp) as u32),
                                time: self.t_start.elapsed().as_secs_f32(),
                                shot: self.shot_now.take(),
                            },
                        );
                        ui.painter().add(cb);
                    } else {
                        ui.painter().rect_filled(rect, 0.0, egui::Color32::from_gray(20));
                        let msg = self.status.lock().map(|s| s.clone()).unwrap_or_default();
                        ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, msg, egui::FontId::proportional(18.0), egui::Color32::from_gray(170));
                    }
                } else {
                    ui.painter().rect_filled(rect, 0.0, egui::Color32::from_gray(20));
                }
                if show_ref {
                    ui.add_space(8.0);
                    if let Some(t) = &self.reference {
                        ui.image((t.id(), size));
                    }
                }
            });
        });

        // headless verification: once the recording has a few frames, have
        // the live callback write its frame to disk, then quit
        if let Some((path, ticks)) = self.shot.as_mut() {
            if n >= 20 {
                *ticks += 1;
                if *ticks == 5 {
                    self.shot_now = Some(path.clone());
                }
                if *ticks == 8 {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}

struct Stderr;
impl log::Log for Stderr {
    fn enabled(&self, m: &log::Metadata) -> bool { m.level() <= log::Level::Warn }
    fn log(&self, r: &log::Record) { if self.enabled(r.metadata()) { eprintln!("{}: {}", r.level(), r.args()); } }
    fn flush(&self) {}
}

fn main() -> eframe::Result<()> {
    // Set before the simulation or UI threads exist.
    unsafe { std::env::set_var("VCAD_LOON_NO_PARAM_RECOVERY", "1") };
    let _ = log::set_logger(&Stderr).map(|()| log::set_max_level(log::LevelFilter::Warn));
    if std::env::args().any(|a| a == "--space") { return space::run(); }
    let splash = std::env::args().any(|a| a == "--splash");
    let frames: usize = std::env::args().find_map(|a| a.strip_prefix("--frames=").and_then(|v| v.parse().ok())).unwrap_or(300);
    let (tx, rx) = mpsc::channel();
    let status: Status = Default::default();
    let status_ui = status.clone();
    std::thread::spawn(move || simulate(tx, status, splash, frames));
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 800.0]).with_title("Kosm view"),
        ..Default::default()
    };
    eframe::run_native("Kosm view", options, Box::new(move |cc| Ok(Box::new(App::new(cc, rx, status_ui, splash)))))
}
