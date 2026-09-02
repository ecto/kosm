//! newt view: the window.
//!
//! The simulation runs on its own thread and hands over a snapshot per frame.
//! The window keeps every snapshot, so the timeline is a recording: play it,
//! pause it, scrub it, inspect any frame. Rendering goes through the same
//! tracer the CLI uses, at preview size while things move and at full size
//! when you stop. Export is the CLI's render, as a button.
//!
//! This is the first slice of the plan: a recording-shaped viewer, native
//! now, the same crate for the browser once the sim runs under wasm.

use std::sync::mpsc;
use std::time::Instant;

use eframe::egui;
use newt_spike::pool::{self, Drop, Surface};
use newt_spike::splash::Droplet;
use tang::Vec3 as V;

/// One frame of the recording: everything the renderer needs, nothing the
/// solver needs.
#[derive(Clone)]
struct Frame {
    t: f64,
    q: Vec<f64>,
    v: Vec<f64>,
    surface: Surface,
    droplets: Vec<Droplet>,
    fluid_force: V<f64>,
    particles: usize,
    sim_ms: u128,
}

impl Frame {
    fn take(drop: &Drop, sim_ms: u128) -> Self {
        Self {
            t: drop.state.time,
            q: drop.state.q.as_slice().to_vec(),
            v: drop.state.v.as_slice().to_vec(),
            surface: drop.surface.clone(),
            droplets: drop.droplets.clone(),
            fluid_force: drop.fluid_force,
            particles: drop.water.as_ref().map(|w| w.count()).unwrap_or(0),
            sim_ms,
        }
    }

    /// A drop the renderer can read, rebuilt from the snapshot.
    fn to_drop(&self) -> Drop {
        let mut d = Drop::new(1.3);
        for (i, x) in self.q.iter().enumerate() {
            d.state.q[i] = *x;
        }
        for (i, x) in self.v.iter().enumerate() {
            d.state.v[i] = *x;
        }
        d.state.time = self.t;
        d.surface = self.surface.clone();
        d.droplets = self.droplets.clone();
        d.fluid_force = self.fluid_force;
        d
    }
}

fn simulate(tx: mpsc::Sender<Frame>, splash: bool, frames: usize) {
    let mut drop = Drop::new(1.3);
    if splash {
        drop = drop.with_water(0.03);
    }
    let steps_per_frame = (1.0 / 30.0 / drop.model.dt).round() as usize;
    for _ in 0..frames {
        let t0 = Instant::now();
        for _ in 0..steps_per_frame {
            drop.step();
        }
        drop.read_water();
        let f = Frame::take(&drop, t0.elapsed().as_millis());
        drop.fluid_force = V::zero();
        if tx.send(f).is_err() {
            break;
        }
    }
}

struct App {
    rx: mpsc::Receiver<Frame>,
    frames: Vec<Frame>,
    cursor: usize,
    playing: bool,
    follow: bool,
    annotate: bool,
    preview: bool,
    splash: bool,
    texture: Option<egui::TextureHandle>,
    rendered_for: Option<(usize, bool)>,
    render_ms: u128,
    last_tick: Instant,
    ui_fps: f32,
    export: Option<std::process::Child>,
    play_t0: f64,
    play_from: usize,
}

impl App {
    fn new(rx: mpsc::Receiver<Frame>, splash: bool) -> Self {
        Self {
            rx,
            frames: Vec::new(),
            cursor: 0,
            playing: true,
            follow: true,
            annotate: true,
            preview: true,
            splash,
            texture: None,
            rendered_for: None,
            render_ms: 0,
            last_tick: Instant::now(),
            ui_fps: 0.0,
            export: None,
            play_t0: 0.0,
            play_from: 0,
        }
    }

    fn render(&mut self, ctx: &egui::Context) {
        let Some(frame) = self.frames.get(self.cursor) else { return };
        let full = !self.playing;
        if self.rendered_for == Some((self.cursor, full)) {
            return;
        }
        let (w, h) = if full { (960, 540) } else { (480, 270) };
        let drop = frame.to_drop();
        let t0 = Instant::now();
        let caustic = pool::caustic(&drop.surface, if full { 0.01 } else { 0.02 });
        let view = pool::View { eye: V::new(-1.42, -1.22, 0.31), target: V::new(0.02, 0.12, -0.06), width: w, height: h, vfov: 0.9 };
        let mut img = pool::render(&view, &drop, &caustic);
        if self.annotate {
            annotate(&mut img, &view, frame);
        }
        self.render_ms = t0.elapsed().as_millis();
        let color = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], img.as_raw());
        match &mut self.texture {
            Some(t) => t.set(color, egui::TextureOptions::LINEAR),
            None => self.texture = Some(ctx.load_texture("frame", color, egui::TextureOptions::LINEAR)),
        }
        self.rendered_for = Some((self.cursor, full));
    }
}

/// Scene annotation: a marker on the melon and its numbers, drawn into the frame.
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
    let c = V::new(f.q[3], f.q[4], f.q[5]);
    if let Some((x, y)) = project(c) {
        // a crosshair on the melon, and a stem to its velocity
        for k in -6..=6i64 {
            for (px, py) in [(x as i64 + k, y as i64), (x as i64, y as i64 + k)] {
                if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                    img.put_pixel(px as u32, py as u32, image::Rgba([255, 230, 60, 255]));
                }
            }
        }
        let vel = V::new(f.v[3], f.v[4], f.v[5]);
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
        // frames from the simulation
        while let Ok(f) = self.rx.try_recv() {
            self.frames.push(f);
        }
        let n = self.frames.len();
        if self.playing && n > 0 {
            if self.follow {
                self.cursor = n - 1;
            } else {
                // advance at the recording's own 30 fps against the wall clock
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
                        // resume from here, at the recording's rate; follow only if at the end
                        self.follow = self.cursor + 1 >= n;
                        self.play_t0 = ctx.input(|i| i.time);
                        self.play_from = self.cursor;
                    }
                }
                ui.checkbox(&mut self.follow, "follow live");
                ui.checkbox(&mut self.annotate, "annotate");
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
                    self.export = std::process::Command::new(std::env::current_exe().map(|p| p.with_file_name("newt-spike")).unwrap_or_else(|_| "newt-spike".into()))
                        .args([mode, "150"])
                        .spawn()
                        .ok();
                }
            });
        });

        egui::Panel::right("inspector").default_size(300.0).show(root, |ui| {
            ui.heading("inspector");
            ui.label(format!("ui {:.0} fps · render {} ms · {}", self.ui_fps, self.render_ms, if self.playing { "preview" } else { "full" }));
            ui.label(format!("recorded {n} frames"));
            if let Some(f) = self.frames.get(self.cursor) {
                ui.separator();
                ui.label(format!("t = {:.3} s   sim {} ms/frame", f.t, f.sim_ms));
                ui.label(format!("melon  z {:+.3} m   vz {:+.2} m/s", f.q[5], f.v[5]));
                ui.label(format!("       x {:+.3}  y {:+.3}", f.q[3], f.q[4]));
                ui.label(format!("fluid force  ({:+.0}, {:+.0}, {:+.0}) N", f.fluid_force.x, f.fluid_force.y, f.fluid_force.z));
                ui.label(format!("rings {}   drops {}", f.surface.rings.len(), f.droplets.len()));
                if f.particles > 0 {
                    ui.label(format!("water particles {}", f.particles));
                }
                if let Some(g) = &f.surface.grid {
                    let (lo, hi) = g.z.iter().fold((f64::MAX, f64::MIN), |a, z| (a.0.min(*z), a.1.max(*z)));
                    ui.label(format!("surface {}×{} @ {:.0} mm   z {:+.3}…{:+.3}", g.nx, g.ny, g.cell * 1e3, lo, hi));
                }
                ui.separator();
                ui.label("knobs (this frame)");
                ui.label(format!("melon axes {:?} m", pool::MELON_AXES));
                ui.label(format!("pool {}×{} m, {} m deep", 2.0 * pool::POOL_X, 2.0 * pool::POOL_Y, pool::DEPTH));
            }
        });

        egui::CentralPanel::default().show(root, |ui| {
            self.render(&ctx);
            if let Some(t) = &self.texture {
                let avail = ui.available_size();
                let aspect = 16.0 / 9.0;
                let size = if avail.x / avail.y > aspect { egui::vec2(avail.y * aspect, avail.y) } else { egui::vec2(avail.x, avail.x / aspect) };
                ui.centered_and_justified(|ui| {
                    ui.image((t.id(), size));
                });
            } else {
                ui.centered_and_justified(|ui| ui.label("settling the water…"));
            }
        });

        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}

fn main() -> eframe::Result<()> {
    let splash = std::env::args().any(|a| a == "--splash");
    let frames: usize = std::env::args().find_map(|a| a.strip_prefix("--frames=").and_then(|v| v.parse().ok())).unwrap_or(300);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || simulate(tx, splash, frames));
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 760.0]).with_title("newt view"),
        ..Default::default()
    };
    eframe::run_native("newt view", options, Box::new(move |_cc| Ok(Box::new(App::new(rx, splash)))))
}
