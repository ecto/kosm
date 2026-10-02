//! kosm-qcd: SU(3) lattice gauge theory you can poke, on the desktop or in a
//! browser.
//!
//! ```sh
//! cargo run -p kosm-qcd --release -- --live             # simulate (GPU when available)
//! cargo run -p kosm-qcd --release -- vacuum.qcdf        # view a phyz-qft file
//! cd crates/kosm-qcd && trunk serve --release           # the same, on the web
//! ```
//!
//! With a compute-capable GPU (native, or WebGPU in a browser) the Monte
//! Carlo runs in WGSL and three tabs measure real physics while you watch:
//! - **vacuum**: instanton lumps of the topological charge, played through
//!   Euclidean time, with χ^(1/4) and the instanton size distribution;
//! - **flux tube**: drag three static quarks and watch the gluon flux tube
//!   between them re-form, beside the static potential and string tension;
//! - **temperature**: shrink Euclidean time to heat the lattice through the
//!   deconfinement transition, with the Polyakov loop in the complex plane.
//!
//! Without compute shaders (WebGL2), a CPU fallback runs the vacuum tab.

mod field;
mod gpu;
mod live;
mod physics;
mod plot;
mod render;

use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui_wgpu::wgpu;
use eframe::{egui, egui_wgpu};
use field::{FieldFile, Grid3, Mesh, height_slice, isosurface, world_scale};
use gpu::engine::{Engine, Measure, Settings};
use live::{LiveParams, LiveSim};
use physics::{HBARC, reference};
use plot::{ACCENT, Axes, COOL, FAINT, INK, histogram};
use render::{Callback, Instance, Orbit, Resources, Scene};
use web_time::Duration;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Height,
    Iso,
}

#[derive(Clone, Copy, PartialEq)]
struct View {
    mode: Mode,
    field: usize,
    t: usize,
    z: f32,
    lift: f32,
    /// Isosurface level in units of the field's standard deviation.
    iso: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Tab {
    Vacuum,
    Flux,
    Temperature,
}

impl Tab {
    fn measure(self) -> Measure {
        match self {
            Tab::Vacuum => Measure::Vacuum,
            Tab::Flux => Measure::Flux,
            Tab::Temperature => Measure::Temperature,
        }
    }
}

enum Source {
    File,
    Cpu(Box<LiveSim>),
    Gpu(Box<Engine>),
}

/// Derived numbers, recomputed a couple of times a second.
#[derive(Default)]
struct Analysis {
    plaq: Option<(f64, f64)>,
    potential: Vec<Option<(f64, f64)>>,
    cornell: Option<((f64, f64, f64), f64)>,
    chi: Option<(f64, f64)>,
    rho_fm: Option<f64>,
    poly_abs: Option<f64>,
}

/// Loop extent the potential is read from: V = ln W(R, T₀) / W(R, T₀ + 1).
const T0: usize = 3;

struct App {
    file: FieldFile,
    /// Bumped whenever `file` is replaced, so meshes rebuild.
    data_gen: u64,
    stats: Vec<(f32, f32)>,
    ranges: Vec<(f32, f32)>,
    view: View,
    built: Option<(View, u64, u32)>,
    scene: Arc<Scene>,
    generation: u64,
    orbit: Orbit,
    playing: bool,
    play_fps: f32,
    play_acc: f32,
    source: Source,
    gpu_ctx: Option<(wgpu::Device, wgpu::Queue)>,
    tab: Tab,
    /// Lattice settings being edited for the vacuum/flux tabs and for the
    /// temperature tab; applied by "restart".
    vac: Settings,
    hot: Settings,
    paused: bool,
    show_flow: bool,
    /// Configuration being faded out, and fade progress 0..1.
    prev: Option<FieldFile>,
    fade: f32,
    last_swap: f64,
    shown_gen: u64,
    shown_flux: usize,
    analysis: Analysis,
    analysed_at: f64,
    dragging: Option<usize>,
    shot: Option<(PathBuf, u32)>,
    shot_now: Option<PathBuf>,
    shot_after: usize,
    /// The whole-window capture came back (capture mode waits for it).
    ui_shot_done: bool,
    gpu: bool,
}

fn empty_mesh() -> Mesh {
    Mesh {
        verts: vec![],
        idx: vec![],
    }
}

impl App {
    fn new(
        cc: &eframe::CreationContext<'_>,
        file: FieldFile,
        view: View,
        cpu: Option<LiveSim>,
        want_gpu: bool,
        tab: Tab,
        shot: Option<PathBuf>,
    ) -> Self {
        let mut gpu_ctx = None;
        if let Some(rs) = &cc.wgpu_render_state {
            rs.renderer
                .write()
                .callback_resources
                .insert(Resources::new(&rs.device, rs.target_format));
            if want_gpu && gpu::supported(&rs.adapter) {
                gpu_ctx = Some((rs.device.clone(), rs.queue.clone()));
            }
        }
        let mut vac = Settings::vacuum();
        if let Some(l) = flag::<usize>("size") {
            vac.size = l;
            vac.nt = l;
        }
        vac.beta = flag("beta").unwrap_or(vac.beta);
        let hot = Settings {
            nt: flag("nt").unwrap_or(4),
            beta: flag("beta").unwrap_or(5.8),
            smear_topo: 6,
            ..vac
        };
        let source = match (&gpu_ctx, cpu) {
            (Some((device, queue)), _) => {
                let s = if tab == Tab::Temperature { hot } else { vac };
                Source::Gpu(Box::new(Engine::new(device, queue, s, tab.measure())))
            }
            (None, Some(sim)) => Source::Cpu(Box::new(sim)),
            (None, None) => Source::File,
        };
        let live = !matches!(source, Source::File);
        let mut app = Self {
            file: FieldFile::empty(),
            data_gen: 0,
            stats: vec![],
            ranges: vec![],
            view,
            built: None,
            scene: Arc::new(Scene {
                generation: 0,
                mesh: empty_mesh(),
                markers: vec![],
            }),
            generation: 0,
            orbit: Orbit {
                azimuth: flag("azim").unwrap_or(-2.2),
                elevation: flag("elev").unwrap_or(0.55),
                distance: flag("dist").unwrap_or(3.2),
            },
            playing: live,
            play_fps: if live { 4.0 } else { 6.0 },
            play_acc: 0.0,
            source,
            gpu_ctx,
            tab,
            vac,
            hot,
            paused: false,
            show_flow: false,
            prev: None,
            fade: 1.0,
            last_swap: -1e9,
            shown_gen: 0,
            shown_flux: usize::MAX,
            analysis: Analysis::default(),
            analysed_at: -1e9,
            dragging: None,
            shot: shot.map(|p| (p, 0)),
            shot_now: None,
            shot_after: flag("shot-after").unwrap_or(if tab == Tab::Vacuum { 2 } else { 40 }),
            ui_shot_done: false,
            gpu: cc.wgpu_render_state.is_some(),
        };
        app.set_file(file);
        if matches!(app.source, Source::Gpu(_)) {
            app.enter_tab(tab);
        }
        app
    }

    fn set_file(&mut self, file: FieldFile) {
        self.stats = (0..file.names.len()).map(|f| file.stats(f)).collect();
        self.ranges = file
            .data
            .iter()
            .map(|d| {
                d.iter()
                    .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
            })
            .collect();
        self.view.t = self.view.t.min(file.dims[0] - 1);
        self.view.field = self.view.field.min(file.names.len() - 1);
        self.file = file;
        self.data_gen += 1;
    }

    /// Show `file`, morphing from the current one when shapes match.
    fn fade_to(&mut self, file: FieldFile) {
        let old = std::mem::replace(&mut self.file, FieldFile::empty());
        let same = old.dims == file.dims && old.names == file.names;
        self.set_file(file);
        if same {
            self.prev = Some(old);
            self.fade = 0.0;
        } else {
            self.prev = None;
            self.fade = 1.0;
        }
    }

    /// Switch tabs; the temperature tab runs its own short-N_t lattice.
    fn switch_tab(&mut self, tab: Tab) {
        let leaving_hot = self.tab == Tab::Temperature;
        self.tab = tab;
        let (vac, hot) = (self.vac, self.hot);
        if let (Some((device, queue)), Source::Gpu(e)) = (self.gpu_ctx.clone(), &mut self.source) {
            e.measure = tab.measure();
            if leaving_hot != (tab == Tab::Temperature) {
                e.restart(
                    &device,
                    &queue,
                    if tab == Tab::Temperature { hot } else { vac },
                );
            }
        }
        self.enter_tab(tab);
    }

    /// Reset the view and placeholder field for `tab`.
    fn enter_tab(&mut self, tab: Tab) {
        let Source::Gpu(e) = &self.source else { return };
        let s = e.settings;
        let quarks = e.quarks;
        let l = s.size;
        self.prev = None;
        self.fade = 1.0;
        self.shown_gen = 0;
        self.shown_flux = usize::MAX;
        self.last_swap = -1e9;
        self.dragging = None;
        match tab {
            Tab::Vacuum => {
                self.view = View {
                    mode: Mode::Iso,
                    field: 0,
                    t: 0,
                    z: 0.0,
                    lift: 0.5,
                    iso: 1.5,
                };
                self.orbit.elevation = 0.55;
                self.playing = true;
                let n = s.nt * l * l * l;
                self.set_file(FieldFile {
                    dims: [s.nt, l, l, l],
                    a_fm: s.a_fm(),
                    names: vec!["topo".into(), "action".into()],
                    data: vec![vec![0.0; n]; 2],
                    quarks: vec![],
                });
            }
            Tab::Flux => {
                self.view = View {
                    mode: Mode::Height,
                    field: 0,
                    t: 0,
                    z: quarks[0][2] as f32,
                    lift: 0.7,
                    iso: 2.0,
                };
                self.orbit.elevation = 0.95;
                self.set_file(flat_file(l, &["action", "electric"], &quarks));
            }
            Tab::Temperature => {
                self.view = View {
                    mode: Mode::Height,
                    field: 0,
                    t: 0,
                    z: l as f32 / 2.0,
                    lift: 0.5,
                    iso: 2.0,
                };
                self.orbit.elevation = 0.6;
                self.set_file(flat_file(l, &["Re P", "|P|"], &[]));
            }
        }
    }

    fn rebuild(&mut self) {
        let fade_q = self
            .prev
            .as_ref()
            .map_or(u32::MAX, |_| (self.fade * 24.0) as u32);
        if self.built == Some((self.view, self.data_gen, fade_q)) {
            return;
        }
        let v = self.view;
        let mut grid = self.file.time_slice(v.field, v.t);
        let (mut mean, mut std) = self.stats[v.field];
        if let Some(prev) = self.prev.as_ref().filter(|p| p.names == self.file.names) {
            let a = self.fade;
            let old = prev.time_slice(v.field, v.t);
            for (x, o) in grid.v.iter_mut().zip(&old.v) {
                *x = a * *x + (1.0 - a) * o;
            }
            let (m0, s0) = prev.stats(v.field);
            mean = a * mean + (1.0 - a) * m0;
            std = a * std + (1.0 - a) * s0;
        }
        let signed = grid.v.iter().any(|&x| x < 0.0);
        let name = self.file.names[v.field].clone();
        let mut mesh = empty_mesh();
        let mut marker_z = Vec::new();
        match v.mode {
            Mode::Height => {
                let (lo, hi) = self.ranges[v.field];
                let baryon = !self.file.quarks.is_empty();
                let polyakov = name == "Re P" || name == "|P|";
                let norm = move |x: f32| {
                    if baryon {
                        if hi > lo { (x - lo) / (hi - lo) } else { 0.9 }
                    } else if polyakov {
                        0.35 + 0.9 * x
                    } else if signed {
                        0.5 + x / (5.0 * std)
                    } else {
                        0.3 + (x - mean) / (3.0 * std)
                    }
                };
                mesh = height_slice(&grid, v.z, 6, v.lift, norm);
                let lift =
                    |q: &[f32; 3]| norm(grid.sample([q[0], q[1], v.z])) * v.lift - 0.5 * v.lift;
                marker_z = self.file.quarks.iter().map(lift).collect();
            }
            Mode::Iso => {
                let fine = grid.upsample(3);
                let level = if signed {
                    v.iso * std
                } else {
                    mean + v.iso * std
                };
                isosurface(&fine, 3, level, [0.95, 0.35, 0.12, 1.0], &mut mesh);
                if signed {
                    let neg = Grid3 {
                        n: fine.n,
                        v: fine.v.iter().map(|x| -x).collect(),
                    };
                    isosurface(&neg, 3, level, [0.15, 0.45, 0.95, 1.0], &mut mesh);
                }
            }
        }
        let markers = self.quark_markers(&grid, &marker_z);
        self.generation += 1;
        self.scene = Arc::new(Scene {
            generation: self.generation,
            mesh,
            markers,
        });
        self.built = Some((v, self.data_gen, fade_q));
    }

    fn quark_markers(&self, grid: &Grid3, surface_z: &[f32]) -> Vec<Instance> {
        let colors = [
            [0.9, 0.15, 0.1, 1.0],
            [0.2, 0.85, 0.2, 1.0],
            [0.2, 0.45, 1.0, 1.0],
        ];
        let n = grid.n;
        let s = world_scale(n);
        self.file
            .quarks
            .iter()
            .enumerate()
            .map(|(i, q)| {
                let x = (q[0] - 0.5 * n[0] as f32) * s;
                let y = (q[1] - 0.5 * n[1] as f32) * s;
                let z = match (self.view.mode, surface_z.get(i)) {
                    (Mode::Height, Some(z)) => z + 0.03,
                    _ => (q[2] - 0.5 * n[2] as f32) * s,
                };
                let r = if self.dragging == Some(i) {
                    0.06
                } else {
                    0.045
                };
                Instance {
                    m: [
                        [r, 0.0, 0.0, 0.0],
                        [0.0, r, 0.0, 0.0],
                        [0.0, 0.0, r, 0.0],
                        [x, y, z, 1.0],
                    ],
                    tint: colors[i % 3],
                }
            })
            .collect()
    }

    /// Run the simulation for this frame and feed its results to the view.
    fn tick_source(&mut self, now: f64, dt: f32) {
        if self.paused {
            return;
        }
        let gpu_ctx = self.gpu_ctx.clone();
        match &mut self.source {
            Source::File => {}
            Source::Cpu(sim) => {
                let Some((file, last)) = sim.step(Duration::from_millis(14)) else {
                    return;
                };
                if self.show_flow {
                    self.prev = None;
                    self.fade = 1.0;
                    self.set_file(file);
                } else if last {
                    self.fade_to(file);
                }
            }
            Source::Gpu(e) => {
                let (device, queue) = gpu_ctx.expect("gpu context");
                e.tick(&device, &queue, dt);
                let mut next: Option<(FieldFile, bool)> = None;
                match self.tab {
                    Tab::Vacuum => {
                        if e.vacuum_gen != self.shown_gen
                            && now - self.last_swap > 2.5
                            && let Some(v) = e.vacuum.clone()
                        {
                            self.shown_gen = e.vacuum_gen;
                            next = Some((v, true));
                        }
                    }
                    Tab::Flux => {
                        if e.flux.configs != self.shown_flux && now - self.last_swap > 0.25 {
                            self.shown_flux = e.flux.configs;
                            next = e.flux_field().map(|f| (f, false));
                        }
                    }
                    Tab::Temperature => {
                        if now - self.last_swap > 2.0 {
                            next = e.polyakov.clone().map(|p| (p, true));
                        }
                    }
                }
                if now - self.analysed_at > 0.5 {
                    self.analysed_at = now;
                    self.analysis = analyse(e);
                }
                if let Some((file, fade)) = next {
                    self.last_swap = now;
                    if fade {
                        self.fade_to(file);
                    } else {
                        self.set_file(file);
                    }
                }
            }
        }
    }

    fn tick_fade(&mut self, dt: f32) {
        if self.prev.is_none() {
            return;
        }
        self.fade += dt / 1.5;
        if self.fade >= 1.0 {
            self.fade = 1.0;
            self.prev = None;
        }
    }

    /// Drag a quark across the surface; returns whether this drag is a quark's.
    fn drag_quark(&mut self, resp: &egui::Response, rect: egui::Rect) -> bool {
        if self.tab != Tab::Flux || !matches!(self.source, Source::Gpu(_)) {
            return false;
        }
        let aspect = rect.width() / rect.height();
        let to_ndc = |p: egui::Pos2| {
            [
                2.0 * (p.x - rect.left()) / rect.width() - 1.0,
                1.0 - 2.0 * (p.y - rect.top()) / rect.height(),
            ]
        };
        if resp.drag_started()
            && let Some(pos) = resp.interact_pointer_pos()
        {
            {
                let at = to_ndc(pos);
                self.dragging = self
                    .scene
                    .markers
                    .iter()
                    .enumerate()
                    .filter_map(|(i, m)| {
                        let p = self
                            .orbit
                            .project([m.m[3][0], m.m[3][1], m.m[3][2]], aspect)?;
                        let d = ((p[0] - at[0]) * rect.width() * 0.5)
                            .hypot((p[1] - at[1]) * rect.height() * 0.5);
                        (d < 24.0).then_some((i, d))
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(i, _)| i);
            }
        }
        let Some(i) = self.dragging else { return false };
        if resp.drag_stopped() {
            self.dragging = None;
            self.built = None;
            return true;
        }
        let Some(pos) = resp.interact_pointer_pos() else {
            return true;
        };
        // intersect the pointer ray with the surface's mid plane z = 0
        let (o, d) = self.orbit.ray(to_ndc(pos), aspect);
        if d[2].abs() < 1e-4 {
            return true;
        }
        let t = -o[2] / d[2];
        let n = [self.file.dims[1], self.file.dims[2], self.file.dims[3]];
        let s = world_scale(n);
        let lx = ((o[0] + t * d[0]) / s + 0.5 * n[0] as f32).round() as i32;
        let ly = ((o[1] + t * d[1]) / s + 0.5 * n[1] as f32).round() as i32;
        let queue = self.gpu_ctx.as_ref().expect("gpu context").1.clone();
        let Source::Gpu(e) = &mut self.source else {
            return true;
        };
        let mut q = e.quarks;
        let l = e.settings.size as i32;
        let xy = [lx.clamp(1, l - 2), ly.clamp(1, l - 2)];
        if xy != [q[i][0], q[i][1]] && quarks_fit(&q, i, xy, l) {
            q[i][0] = xy[0];
            q[i][1] = xy[1];
            e.set_quarks(&queue, q);
            self.file.quarks = e.quarks.iter().map(|q| q.map(|c| c as f32)).collect();
            self.data_gen += 1;
        }
        true
    }

    // ---- panels -------------------------------------------------------------

    fn settings_panel(&mut self, ui: &mut egui::Ui) {
        let Some((device, queue)) = self.gpu_ctx.clone() else {
            return;
        };
        let temp = self.tab == Tab::Temperature;
        let mut restart = None;
        ui.collapsing("lattice", |ui| {
            let s = if temp { &mut self.hot } else { &mut self.vac };
            egui::ComboBox::from_label("size")
                .selected_text(format!("{0}^3", s.size))
                .show_ui(ui, |ui| {
                    for l in [8, 10, 12, 14, 16, 18, 20, 24] {
                        ui.selectable_value(&mut s.size, l, format!("{l}^3"));
                    }
                });
            if temp {
                ui.horizontal(|ui| {
                    ui.label("N_t");
                    for nt in [4, 6, 8] {
                        ui.selectable_value(&mut s.nt, nt, nt.to_string());
                    }
                });
            } else {
                s.nt = s.size;
            }
            ui.add(egui::Slider::new(&mut s.beta, 5.5..=6.4).text("beta"));
            ui.add(egui::Slider::new(&mut s.sep, 1..=10).text("updates between"));
            if !temp {
                let mut pairs = s.smear_topo / 2;
                if ui
                    .add(egui::Slider::new(&mut pairs, 4..=40).text("topology stout pairs"))
                    .changed()
                {
                    s.smear_topo = 2 * pairs;
                }
            }
            ui.label(format!(
                "a = {:.3} fm, box {:.2} fm",
                s.a_fm(),
                s.size as f64 * s.a_fm()
            ));
            if temp {
                ui.label(format!("T = {:.0} MeV", s.temperature_mev()));
            }
            if ui.button("restart").clicked() {
                s.seed += 1;
                restart = Some(*s);
            }
        });
        if let Some(s) = restart {
            if let Source::Gpu(e) = &mut self.source {
                e.restart(&device, &queue, s);
            }
            self.enter_tab(self.tab);
        }
    }

    fn numbers(&self, ui: &mut egui::Ui) {
        let Source::Gpu(e) = &self.source else { return };
        let a = &self.analysis;
        let s = e.settings;
        let pm = |v: Option<(f64, f64)>, digits: usize| {
            v.map_or("…".into(), |(m, e)| {
                format!("{m:.digits$} ± {e:.digits$}")
            })
        };
        egui::Grid::new("numbers")
            .striped(true)
            .num_columns(3)
            .show(ui, |ui| {
                ui.strong("");
                ui.strong("this lattice");
                ui.strong("literature");
                ui.end_row();
                ui.label("plaquette");
                ui.label(pm(a.plaq, 5));
                ui.label(if (s.beta - 6.0).abs() < 1e-9 && s.nt >= 8 {
                    format!("{} (beta 6)", reference::PLAQ_BETA6)
                } else {
                    String::new()
                });
                ui.end_row();
                if s.nt >= 8 {
                    ui.label("sqrt(sigma)");
                    ui.label(a.cornell.map_or("…".into(), |((_, _, sig), err)| {
                        let v = sig.max(0.0).sqrt() / s.a_fm() * HBARC;
                        let dv = 0.5 * err / sig.max(1e-9).sqrt() / s.a_fm() * HBARC;
                        format!("{v:.0} ± {dv:.0} MeV")
                    }));
                    ui.label(format!("~{} MeV", reference::SQRT_SIGMA));
                    ui.end_row();
                    ui.label("Coulomb term B");
                    ui.label(
                        a.cornell
                            .map_or("…".into(), |((_, b, _), _)| format!("{b:.2}")),
                    );
                    ui.label("pi/12 = 0.26 (string)");
                    ui.end_row();
                    ui.label("chi^(1/4)");
                    ui.label(
                        a.chi
                            .map_or("…".into(), |(m, e)| format!("{m:.0} ± {e:.0} MeV")),
                    );
                    ui.label(format!(
                        "{} ± {} MeV",
                        reference::CHI_QUARTER.0,
                        reference::CHI_QUARTER.1
                    ));
                    ui.end_row();
                    ui.label("instanton radius");
                    ui.label(a.rho_fm.map_or("…".into(), |r| format!("{r:.2} fm")));
                    ui.label(format!("~{} fm", reference::INSTANTON_RHO));
                    ui.end_row();
                }
                ui.label("<|P|>");
                ui.label(a.poly_abs.map_or("…".into(), |p| format!("{p:.3}")));
                ui.label(if s.nt >= 8 { "~0 (confined)" } else { "" });
                ui.end_row();
            });
    }

    fn right_panel(&self, ui: &mut egui::Ui) {
        let Source::Gpu(e) = &self.source else { return };
        ui.label(e.status());
        ui.separator();
        let a = e.settings.a_fm();
        match self.tab {
            Tab::Vacuum => {
                let q = &e.stats.q;
                ui.label(format!(
                    "topological charge Q over {} configurations",
                    q.len()
                ));
                let h = histogram(q.iter().copied(), -6.5, 6.5, 13);
                let top = h.iter().cloned().fold(1.0, f64::max);
                Axes::new(ui, 110.0, (-6.5, 6.5), (0.0, top * 1.1), "Q", "count").bars(&h, ACCENT);
                ui.add_space(6.0);
                ui.label(format!(
                    "instanton radii ({} lumps found)",
                    e.stats.radii.len()
                ));
                let h = histogram(e.stats.radii.iter().map(|r| r * a), 0.0, 0.8, 16);
                let top = h.iter().cloned().fold(1.0, f64::max);
                let ax = Axes::new(ui, 110.0, (0.0, 0.8), (0.0, top * 1.1), "rho (fm)", "count");
                ax.bars(&h, COOL);
                ax.vline(reference::INSTANTON_RHO, INK);
                ui.small("radius from each lump's peak, q(0) = 6 / (pi^2 rho^4). Smoothing fattens lumps, so read this as an upper estimate.");
            }
            Tab::Flux => {
                ui.label("static potential between a quark and an antiquark");
                let pts: Vec<(f64, f64, f64)> = self
                    .analysis
                    .potential
                    .iter()
                    .enumerate()
                    .filter_map(|(i, v)| {
                        v.map(|(m, err)| {
                            (
                                (i + 1) as f64 * a,
                                m / a * HBARC / 1000.0,
                                err / a * HBARC / 1000.0,
                            )
                        })
                    })
                    .collect();
                let (lo, hi) = pts.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
                    (lo.min(p.1 - p.2), hi.max(p.1 + p.2))
                });
                let (lo, hi) = if pts.is_empty() {
                    (0.0, 1.0)
                } else {
                    (lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
                };
                let ax = Axes::new(
                    ui,
                    170.0,
                    (0.0, (e.lat.rmax as f64 + 0.5) * a),
                    (lo, hi),
                    "r (fm)",
                    "V (GeV)",
                );
                if let Some(((c0, b, sig), _)) = self.analysis.cornell {
                    ax.curve(
                        |r| (c0 - b * a / r + sig * r / a) / a * HBARC / 1000.0,
                        ACCENT,
                    );
                }
                for (r, v, err) in pts {
                    ax.point(r, v, err, INK);
                }
                ui.small("from Wilson loops on spatially smeared links: V(r) = ln W(r, 3) / W(r, 4). Curve: Cornell fit A - B/r + sigma r. The straight-line rise is confinement.");
                ui.separator();
                ui.label(format!(
                    "flux tube: {} configurations since the quarks moved",
                    e.flux.configs
                ));
                let q = e.quarks;
                let d = |i: usize, j: usize| {
                    (((q[i][0] - q[j][0]).pow(2)
                        + (q[i][1] - q[j][1]).pow(2)
                        + (q[i][2] - q[j][2]).pow(2)) as f64)
                        .sqrt()
                        * a
                };
                ui.label(format!(
                    "quark separations {:.2}, {:.2}, {:.2} fm",
                    d(0, 1),
                    d(1, 2),
                    d(0, 2)
                ));
                ui.small("drag a quark to move it; the average restarts and sharpens as configurations stream in. Past ~0.5 fm the tube takes a Y shape.");
            }
            Tab::Temperature => {
                let s = e.settings;
                let tc = reference::TC_OVER_SQRT_SIGMA * reference::SQRT_SIGMA;
                ui.label(format!(
                    "T = {:.0} MeV (N_t = {}, beta = {:.3}), T/T_c ~ {:.2}",
                    s.temperature_mev(),
                    s.nt,
                    s.beta,
                    s.temperature_mev() / tc
                ));
                if let Some((_, bc)) = reference::BETA_C.iter().find(|(nt, _)| *nt == s.nt) {
                    let phase = if s.beta > *bc {
                        "deconfined"
                    } else {
                        "confined"
                    };
                    ui.label(format!(
                        "critical beta at N_t = {}: {bc}, so expect {phase}",
                        s.nt
                    ));
                }
                ui.label("volume-averaged Polyakov loop, one dot per configuration");
                let side = ui.available_width().min(260.0);
                let ax = Axes::new(ui, side, (-0.6, 0.6), (-0.6, 0.6), "Re P", "Im P");
                for k in 0..3 {
                    let ang = k as f64 * std::f64::consts::TAU / 3.0;
                    ax.painter.line_segment(
                        [ax.to(0.0, 0.0), ax.to(0.6 * ang.cos(), 0.6 * ang.sin())],
                        egui::Stroke::new(1.0, FAINT),
                    );
                }
                let n = e.stats.poly.len();
                for (i, &(re, im)) in e.stats.poly.iter().enumerate().skip(n.saturating_sub(1500)) {
                    let color = if i + 150 >= n {
                        ACCENT
                    } else {
                        COOL.gamma_multiply(0.6)
                    };
                    ax.point(re.clamp(-0.6, 0.6), im.clamp(-0.6, 0.6), 0.0, color);
                }
                ui.small("confined: a blob at 0, since a lone quark would cost infinite energy. Deconfined: P jumps out along one of the three Z(3) directions. Try N_t = 4 at beta 5.6, then 5.8.");
            }
        }
        ui.separator();
        self.numbers(ui);
    }
}

/// Whether quark `i` can move to `xy`: offsets from the junction must stay
/// within half the box (the correlation wraps) and fit the path buffer.
fn quarks_fit(q: &[[i32; 3]; 3], i: usize, xy: [i32; 2], l: i32) -> bool {
    let mut q = *q;
    q[i][0] = xy[0];
    q[i][1] = xy[1];
    let j: [i32; 3] =
        std::array::from_fn(|k| ((q[0][k] + q[1][k] + q[2][k]) as f64 / 3.0).round() as i32);
    q.iter().all(|p| {
        (0..3).all(|k| (p[k] - j[k]).abs() < l / 2)
            && (0..3).map(|k| (p[k] - j[k]).abs()).sum::<i32>() <= gpu::PATH_CAP as i32
    })
}

fn flat_file(l: usize, names: &[&str], quarks: &[[i32; 3]]) -> FieldFile {
    let n = l * l * l;
    let base = if names[0] == "action" { 1.0 } else { 0.0 };
    FieldFile {
        dims: [1, l, l, l],
        a_fm: 0.1,
        names: names.iter().map(|s| s.to_string()).collect(),
        data: names.iter().map(|_| vec![base; n]).collect(),
        quarks: quarks.iter().map(|q| q.map(|c| c as f32)).collect(),
    }
}

fn analyse(e: &Engine) -> Analysis {
    let s = &e.stats;
    let mean_err = |v: &[f64]| {
        physics::jackknife(v, 20, |x| {
            Some(x.iter().map(|y| **y).sum::<f64>() / x.len() as f64)
        })
    };
    Analysis {
        plaq: mean_err(&s.plaq),
        potential: s.potential(T0),
        cornell: s.string_tension(T0, 2),
        chi: if s.q.len() >= 8 {
            s.chi_quarter(e.lat.n, e.settings.a_fm())
        } else {
            None
        },
        rho_fm: (!s.radii.is_empty())
            .then(|| s.radii.iter().sum::<f64>() / s.radii.len() as f64 * e.settings.a_fm()),
        poly_abs: (!s.poly.is_empty())
            .then(|| s.poly.iter().map(|(a, b)| a.hypot(*b)).sum::<f64>() / s.poly.len() as f64),
    }
}

impl eframe::App for App {
    fn ui(&mut self, root: &mut egui::Ui, _f: &mut eframe::Frame) {
        let ctx = root.ctx().clone();
        let (now, dt) = ctx.input(|i| (i.time, i.stable_dt));
        self.tick_source(now, dt);
        self.tick_fade(dt);
        let nt = self.file.dims[0];
        if self.playing && nt > 1 {
            self.play_acc += dt * self.play_fps;
            while self.play_acc >= 1.0 {
                self.play_acc -= 1.0;
                self.view.t = (self.view.t + 1) % nt;
            }
        }

        if let Source::Gpu(_) = self.source {
            egui::Panel::top("tabs").show(root, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("QCD on a lattice");
                    ui.separator();
                    for (tab, label) in [
                        (Tab::Vacuum, "vacuum"),
                        (Tab::Flux, "flux tube"),
                        (Tab::Temperature, "temperature"),
                    ] {
                        if ui.selectable_label(self.tab == tab, label).clicked() && self.tab != tab
                        {
                            self.switch_tab(tab);
                        }
                    }
                    ui.separator();
                    ui.checkbox(&mut self.paused, "pause");
                });
            });
            egui::Panel::right("physics")
                .min_size(320.0)
                .show(root, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| self.right_panel(ui));
                });
        }

        egui::Panel::left("controls").show(root, |ui| {
            let [nt, nx, ny, nz] = self.file.dims;
            match &self.source {
                Source::File => {
                    ui.heading(if self.file.quarks.is_empty() {
                        "QCD vacuum"
                    } else {
                        "baryon flux"
                    });
                    ui.label(format!("{nt}x{nx}x{ny}x{nz}, a = {:.3} fm", self.file.a_fm));
                }
                Source::Cpu(sim) => {
                    ui.heading("QCD vacuum");
                    ui.small("CPU fallback: no WebGPU compute here, so only the vacuum runs.");
                    ui.label(sim.status());
                    ui.add(egui::ProgressBar::new(sim.progress()).desired_height(6.0));
                    ui.label(format!(
                        "plaquette {:.4}   configs {}",
                        sim.plaquette, sim.configs
                    ));
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.paused, "pause");
                        ui.checkbox(&mut self.show_flow, "show smoothing");
                    });
                }
                Source::Gpu(e) => {
                    ui.label(format!(
                        "{}x{}^3 lattice on the GPU",
                        e.settings.nt, e.settings.size
                    ));
                    ui.label(format!(
                        "a = {:.3} fm, beta = {:.2}",
                        e.settings.a_fm(),
                        e.settings.beta
                    ));
                }
            }
            self.settings_panel(ui);
            ui.separator();
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.view.mode, Mode::Height, "height");
                ui.selectable_value(&mut self.view.mode, Mode::Iso, "iso");
            });
            egui::ComboBox::from_label("field")
                .selected_text(self.file.names[self.view.field].clone())
                .show_ui(ui, |ui| {
                    for (i, name) in self.file.names.iter().enumerate() {
                        ui.selectable_value(&mut self.view.field, i, name);
                    }
                });
            if nt > 1 {
                ui.add(egui::Slider::new(&mut self.view.t, 0..=nt - 1).text("t"));
                ui.horizontal(|ui| {
                    if ui.button(if self.playing { "⏸" } else { "▶" }).clicked() {
                        self.playing = !self.playing;
                    }
                    ui.add(egui::Slider::new(&mut self.play_fps, 1.0..=30.0).text("fps"));
                });
            }
            match self.view.mode {
                Mode::Height => {
                    ui.add(
                        egui::Slider::new(&mut self.view.z, 0.0..=(nz as f32 - 1.0))
                            .text("z slice"),
                    );
                    ui.add(egui::Slider::new(&mut self.view.lift, 0.0..=1.5).text("height"));
                }
                Mode::Iso => {
                    ui.add(egui::Slider::new(&mut self.view.iso, 0.2..=6.0).text("level (sigma)"));
                }
            }
            if self.file.names[self.view.field] == "topo" {
                let total: f64 = self.file.data[self.view.field]
                    .iter()
                    .map(|&v| v as f64)
                    .sum();
                ui.label(format!("Q = {total:+.2}"));
            }
        });

        self.rebuild();

        egui::CentralPanel::default().show(root, |ui| {
            let size = ui.available_size();
            let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
            if !self.drag_quark(&resp, rect) && resp.dragged() {
                let d = resp.drag_delta();
                self.orbit.azimuth -= d.x * 0.006;
                self.orbit.elevation = (self.orbit.elevation + d.y * 0.006).clamp(-1.5, 1.5);
            }
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if resp.hovered() && scroll.abs() > 0.0 {
                self.orbit.distance =
                    (self.orbit.distance * (1.0 - scroll * 0.002)).clamp(0.6, 12.0);
            }
            if self.gpu {
                let ppp = ctx.pixels_per_point();
                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    Callback {
                        scene: self.scene.clone(),
                        orbit: self.orbit,
                        size: ((size.x * ppp) as u32, (size.y * ppp) as u32),
                        shot: self.shot_now.take(),
                    },
                ));
            } else {
                ui.painter()
                    .rect_filled(rect, 0.0, egui::Color32::from_gray(20));
            }
        });

        // headless verification: once enough is measured, capture and quit
        let ready = match &self.source {
            Source::File => true,
            Source::Cpu(s) => s.configs > 0,
            Source::Gpu(e) => match self.tab {
                Tab::Flux => e.flux.configs >= self.shot_after,
                _ => e.stats.configs >= self.shot_after && self.prev.is_none(),
            },
        };
        // the whole window too, panels and plots included, next to the 3D pass
        ctx.input(|i| {
            for ev in &i.raw.events {
                if let egui::Event::Screenshot {
                    image, user_data, ..
                } = ev
                    && let Some(path) = user_data
                        .data
                        .as_ref()
                        .and_then(|d| d.downcast_ref::<PathBuf>())
                {
                    let [w, h] = image.size;
                    let px: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_array()).collect();
                    if let Some(img) = image::RgbaImage::from_raw(w as u32, h as u32, px) {
                        let _ = img.save(path);
                        eprintln!("ui shot {}", path.display());
                        self.ui_shot_done = true;
                    }
                }
            }
        });
        if let Some((path, ticks)) = self.shot.as_mut().filter(|_| ready) {
            *ticks += 1;
            if *ticks == 4 {
                self.shot_now = Some(path.clone());
                let ui_path = path.with_extension("ui.png");
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(
                    ui_path,
                )));
            }
            if self.ui_shot_done || *ticks > 120 {
                if let Source::Gpu(e) = &self.source {
                    let a = analyse(e);
                    eprintln!(
                        "{:?} configs {} rate {:.1}/s plaq {:?} cornell {:?} chi {:?} rho_fm {:?} |P| {:?} flux configs {}",
                        e.measure,
                        e.stats.configs,
                        e.rate,
                        a.plaq,
                        a.cornell,
                        a.chi,
                        a.rho_fm,
                        a.poly_abs,
                        e.flux.configs
                    );
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        if !matches!(self.source, Source::File) && !self.paused {
            ctx.request_repaint();
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
    }
}

fn flag<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::args().find_map(|a| {
        a.strip_prefix(&format!("--{name}="))
            .and_then(|v| v.parse().ok())
    })
}

fn live_view(size: usize) -> View {
    View {
        mode: Mode::Iso,
        field: 0,
        t: 0,
        z: size as f32 / 2.0,
        lift: 0.5,
        iso: 1.5,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    let usage = "usage: kosm-qcd <file.qcdf> | --live [--cpu] [--tab=vacuum|flux|temperature] [--size=N] [--nt=N] [--beta=B] \
                 [--mode=height|iso] [--field=name] [--t=N] [--z=N] [--azim=rad] [--elev=rad] [--dist=D] [--shot=out.png] [--shot-after=N]";
    let shot = flag::<String>("shot").map(PathBuf::from);
    let tab = match flag::<String>("tab").as_deref() {
        Some("flux") => Tab::Flux,
        Some("temperature") => Tab::Temperature,
        _ => Tab::Vacuum,
    };
    let (title, file, mut view, cpu, want_gpu) = if std::env::args().any(|a| a == "--live") {
        let d = LiveParams::default();
        let params = LiveParams {
            size: flag("size").unwrap_or(12),
            beta: flag("beta").unwrap_or(d.beta),
            smear: flag("smear").unwrap_or(d.smear),
            therm: flag("therm").unwrap_or(d.therm),
            ..d
        };
        let sim = LiveSim::new(params);
        let want_gpu = !std::env::args().any(|a| a == "--cpu");
        (
            "kosm-qcd — live".to_string(),
            sim.placeholder(),
            live_view(params.size),
            Some(sim),
            want_gpu,
        )
    } else {
        let path: PathBuf = std::env::args()
            .skip(1)
            .find(|a| !a.starts_with("--"))
            .ok_or_else(|| anyhow::anyhow!(usage))?
            .into();
        let file = FieldFile::load(&path)?;
        let [_, _, _, nz] = file.dims;
        let view = View {
            mode: Mode::Height,
            field: 0,
            t: 0,
            z: file.quarks.first().map_or(nz as f32 / 2.0, |q| q[2]),
            lift: 0.5,
            iso: 2.0,
        };
        (
            format!("kosm-qcd — {}", path.display()),
            file,
            view,
            None,
            false,
        )
    };
    if let Some(m) = flag::<String>("mode") {
        view.mode = if m == "iso" { Mode::Iso } else { Mode::Height };
    }
    if let Some(i) = flag::<String>("field").and_then(|n| file.names.iter().position(|x| *x == n)) {
        view.field = i;
    }
    view.t = flag("t").unwrap_or(view.t).min(file.dims[0] - 1);
    view.z = flag("z").unwrap_or(view.z);
    view.lift = flag("lift").unwrap_or(view.lift);
    view.iso = flag("iso").unwrap_or(view.iso);
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1500.0, 860.0])
        .with_title(&title);
    if shot.is_some() {
        // capture runs shouldn't grab focus from whatever you're doing
        viewport = viewport.with_active(false).with_position([30.0, 30.0]);
    }
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        &title,
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, file, view, cpu, want_gpu, tab, shot)))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::JsCast;
    console_error_panic_hook::set_once();
    wasm_bindgen_futures::spawn_local(async {
        let canvas = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("kosm_qcd"))
            .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
            .expect("page needs <canvas id=\"kosm_qcd\">");
        let params = LiveParams::default();
        let sim = LiveSim::new(params);
        let file = sim.placeholder();
        let result = eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(move |cc| {
                    Ok(Box::new(App::new(
                        cc,
                        file,
                        live_view(params.size),
                        Some(sim),
                        true,
                        Tab::Vacuum,
                        None,
                    )))
                }),
            )
            .await;
        if let Some(loading) = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("loading"))
        {
            match result {
                Ok(()) => loading.remove(),
                Err(e) => loading.set_inner_html(&format!("could not start: {e:?}")),
            }
        }
    });
}
