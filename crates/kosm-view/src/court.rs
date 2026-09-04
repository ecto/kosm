//! The court in the window: `kosm-view --court`.
//!
//! Same shape as the pool viewer — the simulation runs on its own thread and
//! hands over one snapshot per frame, the window keeps every snapshot so the
//! timeline is a recording — but the picture is not hand-written geometry.
//! The level's roots are evaluated once by `vcad_eval` into BRep `Solid`s,
//! each gets a `vcad_kernel_raytrace::Bvh`, and the frame is a
//! `vcad_kernel_raytrace::pathtrace::Scene`: those BVHs as `Object`s with a
//! material per root name, the balls as `Object::placed` at their poses, and
//! the level's ceiling panels as `AreaLight`s. Nothing here describes a
//! shape; the CAD file does.
//!
//! ## why the CPU tier
//!
//! `vcad-kernel-raytrace` has a `gpu` feature, and it does compile in this
//! workspace — but it pins **wgpu 23**, while `eframe` 0.36 (and its
//! `egui-wgpu`) is on **wgpu 30**. Two major versions of wgpu are two
//! unrelated sets of types: the `Device`, `Queue`, `Buffer` and `Texture`
//! egui hands a paint callback cannot be passed to `RayTracePipeline`, and a
//! texture the tracer wrote cannot be sampled by egui's renderer. There is no
//! conversion; they are different crates that happen to share a name. So the
//! GPU tracer could only run on a *second*, headless wgpu-23 device with a
//! full CPU readback per frame — which is not "render into an egui texture",
//! and whose scene format (`GpuScene::from_brep`, one merged BRep with no
//! per-instance transform) would force the whole court to be rebuilt and
//! re-uploaded every time a ball moves.
//!
//! So this is the CPU path tracer, `pathtrace::render`, run small (320×180 by
//! default) on a worker thread and accumulated progressively: passes keep
//! being added to the same frame while nothing changes, and the accumulator
//! is thrown away the moment the cursor, the camera or the panel size moves.
//! It is the same integrator the reference tier uses, at fewer samples.

use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use eframe::egui;
use kosm_spike::court::{Court, CourtScene};
use kosm_spike::scene::MM;
use phyz_math::{Mat3, Vec3};
use vcad_kernel::Solid;
use vcad_kernel_math::{Point3, Transform, Vec3 as KVec3};
use vcad_kernel_raytrace::pathtrace::{self, AreaLight, Environment, Film, Object, Pbr};
use vcad_kernel_raytrace::Bvh;

/// What the simulation or the renderer is doing, for the window to show.
type Status = Arc<Mutex<String>>;

fn set(status: &Status, s: String) {
    if let Ok(mut g) = status.lock() {
        *g = s;
    }
}

// ---- the recording ----------------------------------------------------------

/// One ball at one instant, in simulation units (metres).
#[derive(Clone, Copy)]
pub struct BallPose {
    pub centre: Vec3,
    pub velocity: Vec3,
    /// The ball's body frame, for a seamed ball's texture.
    pub rot: Mat3,
}

/// One frame of the recording: the ball poses and the time, nothing else.
/// The court itself never moves, so it is built once and lives in the
/// renderer.
#[derive(Clone)]
pub struct Frame {
    pub time: f64,
    pub balls: Vec<BallPose>,
    /// When the shot's centre passed down through the rim, if it has.
    pub made_at: Option<f64>,
    /// Which ball is the shot, if there is one.
    pub shot: Option<usize>,
    pub sim_ms: u128,
}

impl Frame {
    fn take(court: &Court, sim_ms: u128) -> Self {
        Self {
            time: court.time(),
            balls: (0..court.bodies())
                .map(|k| BallPose { centre: court.centre(k), velocity: court.velocity(k), rot: court.rotation(k) })
                .collect(),
            made_at: court.made_at,
            shot: court.shot,
            sim_ms,
        }
    }
}

/// The court, stepping on its own thread.
fn simulate(tx: Sender<Frame>, status: Status, frames: usize) {
    let scene = match CourtScene::bundled() {
        Ok(scene) => scene,
        Err(error) => return set(&status, format!("could not load the court scene: {error}")),
    };
    let mut court = match Court::from_scene(&scene) {
        Ok(court) => court,
        Err(error) => return set(&status, format!("could not build the court: {error}")),
    };
    let frames = if frames > 0 { frames } else { scene.frames() };
    let steps_per_frame = (1.0 / scene.fps / scene.dt).round().max(1.0) as usize;
    let _ = tx.send(Frame::take(&court, 0));
    for k in 0..frames {
        let t0 = Instant::now();
        for _ in 0..steps_per_frame {
            court.step();
        }
        let ms = t0.elapsed().as_millis();
        set(
            &status,
            format!("frame {} of {frames} · {ms} ms per frame · t {:.2} s", k + 1, court.time()),
        );
        if tx.send(Frame::take(&court, ms)).is_err() {
            return;
        }
    }
    set(&status, format!("{frames} frames, {:.2} s simulated", court.time()));
}

// ---- the picture ------------------------------------------------------------

/// The camera, in authored millimetres.
#[derive(Clone, Copy, PartialEq)]
pub struct Camera {
    pub eye: KVec3,
    pub target: KVec3,
    pub fov_deg: f64,
    pub exposure: f32,
}

impl Camera {
    fn to_pathtrace(self) -> pathtrace::Camera {
        pathtrace::Camera::look_at(
            Point3::new(self.eye.x, self.eye.y, self.eye.z),
            Point3::new(self.target.x, self.target.y, self.target.z),
            KVec3::new(0.0, 0.0, 1.0),
            self.fov_deg,
        )
    }
}

/// A material for a root's name. The level names its roots; this is the only
/// place the viewer decides what those names look like.
fn material_for(name: &str) -> Pbr {
    match name {
        // lacquered maple: a warm dielectric under a gloss coat
        "maple" => Pbr { clearcoat: 0.7, clearcoat_roughness: 0.06, ..Pbr::plastic([0.55, 0.34, 0.16], 0.35, 0.7) },
        // the backboard. `Pbr` has no transmission, so tempered glass reads
        // here as a pale, very glossy sheet rather than a see-through one.
        "glass" => Pbr { clearcoat: 1.0, clearcoat_roughness: 0.02, ..Pbr::plastic([0.78, 0.82, 0.84], 0.06, 1.0) },
        "rim" => Pbr::plastic([0.72, 0.22, 0.05], 0.28, 0.5),
        "steel" => Pbr::metal([0.58, 0.59, 0.61], 0.35),
        "paint" => Pbr::plastic([0.85, 0.85, 0.86], 0.5, 0.2),
        "ball" | "ball-seams" | "seam" => ball_material(),
        _ => Pbr::plastic([0.55, 0.56, 0.58], 0.45, 0.1),
    }
}

fn ball_material() -> Pbr {
    Pbr::plastic([0.52, 0.20, 0.07], 0.62, 0.05)
}

/// The court, evaluated: every static root's BVH with its material, the
/// ball's own solid, the gym, and the panels. Built once.
pub struct Stage {
    /// Static geometry: a BVH, a material, and where it sits.
    statics: Vec<(Arc<Bvh>, Pbr, Transform)>,
    /// The ball's appearance: the level's own `ball` roots if it has any (each
    /// with its material), otherwise one sphere of the ball's radius.
    ball: Vec<(Arc<Bvh>, Pbr)>,
    lights: Vec<AreaLight>,
    pub camera: Camera,
    /// A moment of the recording the level suggests looking at, in seconds.
    pub still_t: f64,
    pub prims: usize,
}

impl Stage {
    /// Evaluate the level and build everything that does not move.
    pub fn build(scene: &CourtScene) -> anyhow::Result<Self> {
        let a = &scene.authored;
        let mm = |k: &str| a.parameter(k);
        let evaluated = vcad_eval::evaluate_document(&a.document, &vcad_eval::EvalOptions::default())
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        anyhow::ensure!(
            evaluated.parts.len() == a.document.roots.len(),
            "the court evaluated to {} parts for {} roots",
            evaluated.parts.len(),
            a.document.roots.len()
        );

        let mut statics = Vec::new();
        let mut ball = Vec::new();
        for part in &evaluated.parts {
            let Some(brep) = part.solid.as_ref().and_then(|s| s.as_brep()) else {
                continue;
            };
            let bvh = Arc::new(Bvh::build(brep));
            let mat = material_for(&part.material);
            match part.material.as_str() {
                // the ball's own appearance, placed at each ball's pose
                "ball" | "ball-seams" | "seam" => ball.push((bvh, mat)),
                _ => statics.push((bvh, mat, Transform::identity())),
            }
        }
        if ball.is_empty() {
            let sphere = Solid::sphere(mm("ball_r_mm")?, 48);
            let brep = sphere.as_brep().ok_or_else(|| anyhow::anyhow!("the ball sphere has no BRep"))?;
            ball.push((Arc::new(Bvh::build(brep)), ball_material()));
        }

        // the gym the level describes: walls a margin outside the slab, a
        // floor under it and a ceiling over it, all vcad boxes.
        let (wx, wy) = (0.5 * mm("court_x_mm")? + mm("gym_margin_mm")?, 0.5 * mm("court_y_mm")? + mm("gym_margin_mm")?);
        let h = mm("gym_h_mm")?;
        let t = 100.0;
        let wall = Pbr::plastic([0.44, 0.45, 0.48], 0.7, 0.0);
        let ceiling = Pbr::plastic([0.66, 0.67, 0.68], 0.85, 0.0);
        let under = Pbr::plastic([0.30, 0.31, 0.33], 0.8, 0.0);
        let mut push_box = |c: [f64; 3], half: [f64; 3], mat: Pbr| -> anyhow::Result<()> {
            let solid = Solid::cube(2.0 * half[0], 2.0 * half[1], 2.0 * half[2])
                .translate(c[0] - half[0], c[1] - half[1], c[2] - half[2]);
            let brep = solid.as_brep().ok_or_else(|| anyhow::anyhow!("a gym box has no BRep"))?;
            statics.push((Arc::new(Bvh::build(brep)), mat, Transform::identity()));
            Ok(())
        };
        push_box([wx + t, 0.0, 0.5 * h], [t, wy + 2.0 * t, 0.5 * h + t], wall)?;
        push_box([-wx - t, 0.0, 0.5 * h], [t, wy + 2.0 * t, 0.5 * h + t], wall)?;
        push_box([0.0, wy + t, 0.5 * h], [wx + 2.0 * t, t, 0.5 * h + t], wall)?;
        push_box([0.0, -wy - t, 0.5 * h], [wx + 2.0 * t, t, 0.5 * h + t], wall)?;
        push_box([0.0, 0.0, -21.0], [wx, wy, 20.0], under)?;
        push_box([0.0, 0.0, h + t], [wx, wy, t], ceiling)?;

        // the panels: the only light there is, facing down from the ceiling.
        let (rows, cols) = (a.parameter("light_rows")?.max(1.0) as usize, a.parameter("light_cols")?.max(1.0) as usize);
        let (lw, ll) = (mm("light_w_mm")?, mm("light_l_mm")?);
        let e = a.parameter_or("light_radiance", 18.0) as f32;
        let mut lights = Vec::new();
        for i in 0..cols {
            for j in 0..rows {
                let x = (i as f64 + 0.5) / cols as f64 * 2.0 * wx - wx;
                let y = (j as f64 + 0.5) / rows as f64 * 2.0 * wy - wy;
                lights.push(AreaLight {
                    center: Point3::new(x, y, h - 5.0),
                    // u × v = −z: the emitting face looks at the floor.
                    u: KVec3::new(0.5 * lw, 0.0, 0.0),
                    v: KVec3::new(0.0, -0.5 * ll, 0.0),
                    emission: [e, e, e],
                });
            }
        }

        let prims = statics.len() + ball.len();
        Ok(Self {
            statics,
            ball,
            lights,
            camera: Camera {
                eye: KVec3::new(mm("cam_x_mm")?, mm("cam_y_mm")?, mm("cam_z_mm")?),
                target: KVec3::new(mm("cam_at_x_mm")?, mm("cam_at_y_mm")?, mm("cam_at_z_mm")?),
                fov_deg: a.parameter_or("cam_vfov_deg", 42.0),
                exposure: a.parameter_or("exposure", 1.0) as f32,
            },
            still_t: a.parameter_or("still_t", 0.95),
            prims,
        })
    }

    pub fn load() -> anyhow::Result<Self> {
        Self::build(&CourtScene::bundled()?)
    }

    /// The scene at one frame: the static objects, plus the ball's solid
    /// placed at each ball's pose. Metres become millimetres here and
    /// nowhere else.
    pub fn scene(&self, balls: &[BallPose]) -> pathtrace::Scene {
        let mut objects: Vec<Object> = self
            .statics
            .iter()
            .map(|(bvh, mat, xf)| Object::placed(bvh.clone(), *mat, xf.clone()))
            .collect();
        for pose in balls {
            let c = pose.centre / MM;
            let place = Transform {
                matrix: tang::Mat4::from_rotation_translation(pose.rot, KVec3::new(c.x, c.y, c.z)),
            };
            for (bvh, mat) in &self.ball {
                objects.push(Object::placed(bvh.clone(), *mat, place.clone()));
            }
        }
        pathtrace::Scene { objects, lights: self.lights.clone(), env: Environment::default(), ground: None }
    }
}

// ---- the renderer, on its own thread ---------------------------------------

/// What the window asks for: a frame, a camera, a size, and how much to spend.
#[derive(Clone)]
pub struct Job {
    pub generation: u64,
    pub balls: Vec<BallPose>,
    pub camera: Camera,
    pub size: (u32, u32),
    pub spp: u32,
    /// Passes to accumulate before the renderer goes quiet.
    pub passes: u32,
}

/// What comes back: RGBA at the requested size, and how it was paid for.
pub struct Shot {
    pub generation: u64,
    pub size: (u32, u32),
    pub rgba: Vec<u8>,
    pub passes: u32,
    pub spp: u32,
    pub ms: u128,
}

fn options(spp: u32, seed: u64, denoise: bool) -> pathtrace::PathTraceOptions {
    pathtrace::PathTraceOptions {
        spp,
        max_depth: 5,
        rr_start: 2,
        firefly_clamp: Some(8.0),
        show_background: true,
        seed,
        denoise,
        ..Default::default()
    }
}

/// A running mean over passes, plus the guide buffers of the last one.
struct Accum {
    sum: Vec<f32>,
    alpha: Vec<f32>,
    normal: Vec<f32>,
    depth: Vec<f32>,
    albedo: Vec<f32>,
    variance: Vec<f32>,
    size: (u32, u32),
    passes: u32,
}

impl Accum {
    fn new(size: (u32, u32)) -> Self {
        let n = (size.0 * size.1) as usize;
        Self {
            sum: vec![0.0; n * 3],
            alpha: vec![0.0; n],
            normal: vec![0.0; n * 3],
            depth: vec![0.0; n],
            albedo: vec![0.0; n * 3],
            variance: vec![0.0; n],
            size,
            passes: 0,
        }
    }

    fn add(&mut self, film: Film) {
        for (s, v) in self.sum.iter_mut().zip(&film.rgb) {
            *s += v;
        }
        for (s, v) in self.alpha.iter_mut().zip(&film.alpha) {
            *s += v;
        }
        self.normal = film.normal;
        self.depth = film.depth;
        self.albedo = film.albedo;
        self.variance = film.variance;
        self.passes += 1;
    }

    /// The mean so far, denoised, as sRGB bytes.
    fn resolve(&self, exposure: f32, opts: &pathtrace::PathTraceOptions) -> Vec<u8> {
        let k = 1.0 / self.passes.max(1) as f32;
        let mut film = Film {
            width: self.size.0,
            height: self.size.1,
            rgb: self.sum.iter().map(|v| v * k).collect(),
            alpha: self.alpha.iter().map(|v| v * k).collect(),
            normal: self.normal.clone(),
            depth: self.depth.clone(),
            albedo: self.albedo.clone(),
            variance: self.variance.iter().map(|v| v * k).collect(),
        };
        if opts.denoise {
            pathtrace::denoise(&mut film, opts);
        }
        film.to_srgb8(exposure, false)
    }
}

/// The renderer: build the stage once, then keep adding passes to whatever
/// the window last asked for.
fn render_worker(jobs: Receiver<Job>, out: Sender<Shot>, status: Status) {
    set(&status, "evaluating the level".into());
    let t0 = Instant::now();
    let stage = match Stage::load() {
        Ok(stage) => stage,
        Err(error) => return set(&status, format!("could not build the court's picture: {error}")),
    };
    set(&status, format!("{} vcad solids, {} panels, in {:.1} s", stage.prims, stage.lights.len(), t0.elapsed().as_secs_f64()));

    let mut current: Option<Job> = None;
    let mut accum: Option<Accum> = None;
    loop {
        // Take the newest request; anything older is already stale.
        let mut latest = None;
        loop {
            match jobs.try_recv() {
                Ok(job) => latest = Some(job),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        let idle = latest.is_none()
            && current.as_ref().map(|j| accum.as_ref().map_or(false, |a| a.passes >= j.passes)).unwrap_or(true);
        if idle {
            match jobs.recv() {
                Ok(job) => latest = Some(job),
                Err(_) => return,
            }
        }
        if let Some(job) = latest {
            accum = None;
            current = Some(job);
        }
        let Some(job) = current.clone() else { continue };
        if job.size.0 == 0 || job.size.1 == 0 {
            continue;
        }
        let acc = accum.get_or_insert_with(|| Accum::new(job.size));
        if acc.size != job.size {
            *acc = Accum::new(job.size);
        }
        let lap = Instant::now();
        let opts = options(job.spp, 0x5eed_0000 ^ (job.generation << 20) ^ acc.passes as u64, true);
        let scene = stage.scene(&job.balls);
        let film = pathtrace::render(&scene, &job.camera.to_pathtrace(), job.size.0, job.size.1, &options(job.spp, opts.seed, false));
        acc.add(film);
        let rgba = acc.resolve(job.camera.exposure, &opts);
        let shot = Shot {
            generation: job.generation,
            size: job.size,
            rgba,
            passes: acc.passes,
            spp: job.spp,
            ms: lap.elapsed().as_millis(),
        };
        if out.send(shot).is_err() {
            return;
        }
    }
}

// ---- one still, no window ---------------------------------------------------

/// Step the court to `t` and render one frame with the CPU path tracer. The
/// frame producer the window uses, with no window: this is what makes the
/// picture testable.
pub fn still(path: &std::path::Path, t: f64, size: (u32, u32), spp: u32) -> anyhow::Result<()> {
    let scene = CourtScene::bundled()?;
    let stage = Stage::build(&scene)?;
    let t = if t < 0.0 { stage.still_t } else { t };
    let mut court = Court::from_scene(&scene)?;
    while court.time() < t {
        court.step();
    }
    let frame = Frame::take(&court, 0);
    let t0 = Instant::now();
    let picture = stage.scene(&frame.balls);
    let film = pathtrace::render(&picture, &stage.camera.to_pathtrace(), size.0, size.1, &options(spp, 0x5eed_1234, true));
    let rgba = film.to_srgb8(stage.camera.exposure, false);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    image::RgbaImage::from_raw(size.0, size.1, rgba)
        .ok_or_else(|| anyhow::anyhow!("the film is the wrong size"))?
        .save(path)?;
    println!(
        "court  t = {:.2} s, {} balls, {} vcad solids; {}×{} at {spp} spp in {:.1} s → {}",
        court.time(),
        frame.balls.len(),
        stage.prims,
        size.0,
        size.1,
        t0.elapsed().as_secs_f64(),
        path.display()
    );
    Ok(())
}

// ---- the window -------------------------------------------------------------

struct App {
    rx: Receiver<Frame>,
    shots: Receiver<Shot>,
    jobs: Sender<Job>,
    sim_status: Status,
    render_status: Status,
    frames: Vec<Frame>,
    cursor: usize,
    playing: bool,
    follow: bool,
    orbit: (f64, f64, f64),
    camera: Camera,
    /// The camera the level asks for, to go back to.
    authored_camera: Camera,
    stale: bool,
    texture: Option<egui::TextureHandle>,
    shown: Option<(u64, u32, u32)>,
    generation: u64,
    asked: Option<(usize, [i64; 7], (u32, u32))>,
    size: (u32, u32),
    spp: u32,
    passes: u32,
    last_ms: u128,
    play_t0: f64,
    play_from: usize,
}

impl App {
    fn new(
        rx: Receiver<Frame>,
        shots: Receiver<Shot>,
        jobs: Sender<Job>,
        sim_status: Status,
        render_status: Status,
        camera: Camera,
        size: (u32, u32),
        spp: u32,
    ) -> Self {
        let mut app = Self {
            rx,
            shots,
            jobs,
            sim_status,
            render_status,
            frames: Vec::new(),
            cursor: 0,
            playing: true,
            follow: true,
            orbit: (0.0, 0.0, 0.0),
            camera,
            authored_camera: camera,
            stale: false,
            texture: None,
            shown: None,
            generation: 0,
            asked: None,
            size,
            spp,
            passes: 64,
            last_ms: 0,
            play_t0: 0.0,
            play_from: 0,
        };
        app.orbit = app.orbit_from_camera();
        app
    }

    fn orbit_from_camera(&self) -> (f64, f64, f64) {
        let d = self.camera.eye - self.camera.target;
        let dist = d.norm().max(1.0);
        (d.y.atan2(d.x), (d.z / dist).asin(), dist)
    }

    fn apply_orbit(&mut self) {
        let (az, el, dist) = self.orbit;
        let t = self.camera.target;
        self.camera.eye = t + KVec3::new(dist * el.cos() * az.cos(), dist * el.cos() * az.sin(), dist * el.sin());
    }

    /// A cheap identity for "the same picture": the frame, the camera to the
    /// millimetre, and the panel size.
    fn key(&self) -> (usize, [i64; 7], (u32, u32)) {
        let c = &self.camera;
        (
            self.cursor,
            [
                c.eye.x as i64,
                c.eye.y as i64,
                c.eye.z as i64,
                c.target.x as i64,
                c.target.y as i64,
                c.target.z as i64,
                (c.fov_deg * 100.0) as i64,
            ],
            self.size,
        )
    }

    fn ask(&mut self) {
        let Some(frame) = self.frames.get(self.cursor) else { return };
        self.generation += 1;
        let job = Job {
            generation: self.generation,
            balls: frame.balls.clone(),
            camera: self.camera,
            size: self.size,
            spp: self.spp,
            passes: if self.playing { 1 } else { self.passes },
        };
        let _ = self.jobs.send(job);
        self.asked = Some(self.key());
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
                self.cursor = (((t - self.play_t0) * 30.0) as usize + self.play_from).min(n - 1);
            }
        }
        // Every shot is shown, even one the window has already moved past: a
        // pass takes longer than a UI frame, so refusing stale ones would
        // leave the panel blank for the whole of playback.
        while let Ok(shot) = self.shots.try_recv() {
            let img = egui::ColorImage::from_rgba_unmultiplied([shot.size.0 as usize, shot.size.1 as usize], &shot.rgba);
            match &mut self.texture {
                Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                None => self.texture = Some(ctx.load_texture("court", img, egui::TextureOptions::LINEAR)),
            }
            self.shown = Some((shot.generation, shot.passes, shot.spp));
            self.stale = shot.generation != self.generation;
            self.last_ms = shot.ms;
        }

        egui::Panel::top("bar").show(root, |ui| {
            ui.horizontal(|ui| {
                ui.heading("the court");
                ui.separator();
                if ui.button(if self.playing { "⏸ pause" } else { "▶ play" }).clicked() {
                    self.playing = !self.playing;
                    if self.playing {
                        self.follow = self.cursor + 1 >= n;
                        self.play_t0 = ctx.input(|i| i.time);
                        self.play_from = self.cursor;
                    }
                    self.asked = None;
                }
                ui.checkbox(&mut self.follow, "follow live");
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
                let mut spp = self.spp;
                if ui.add(egui::Slider::new(&mut spp, 1..=16).text("spp/pass")).changed() {
                    self.spp = spp;
                    self.asked = None;
                }
                let mut w = self.size.0;
                if ui.add(egui::Slider::new(&mut w, 160..=960).text("width")).changed() {
                    self.size = (w, (w * 9 / 16).max(1));
                }
            });
        });

        egui::Panel::right("inspector").default_size(320.0).show(root, |ui| {
            ui.heading("inspector");
            ui.label(format!("recorded {n} frames"));
            if let Ok(s) = self.sim_status.lock() {
                ui.label(s.as_str());
            }
            ui.separator();
            ui.label("picture: vcad, path traced on the CPU");
            if let Ok(s) = self.render_status.lock() {
                ui.label(s.as_str());
            }
            ui.label(format!("{}×{} · {} ms a pass", self.size.0, self.size.1, self.last_ms));
            match self.shown {
                Some((_, passes, spp)) => ui.label(format!(
                    "{} samples ({passes} × {spp} spp){}",
                    passes * spp,
                    if self.stale { " · catching up" } else { "" }
                )),
                None => ui.label("no frame yet"),
            };
            ui.separator();
            if let Some(f) = self.frames.get(self.cursor) {
                ui.label(format!("t = {:.3} s   sim {} ms/frame", f.time, f.sim_ms));
                for (k, b) in f.balls.iter().enumerate() {
                    let tag = if Some(k) == f.shot { "shot" } else { "ball" };
                    ui.label(format!(
                        "{tag} {k}  ({:+.2}, {:+.2}, {:.2}) m   {:.2} m/s",
                        b.centre.x,
                        b.centre.y,
                        b.centre.z,
                        b.velocity.norm()
                    ));
                }
                ui.separator();
                match (f.shot, f.made_at) {
                    (Some(_), Some(t)) => ui.colored_label(egui::Color32::from_rgb(120, 220, 130), format!("made at {t:.2} s")),
                    (Some(_), None) if f.time > 2.0 => ui.label("not through the hoop yet"),
                    (Some(_), None) => ui.label("in the air"),
                    _ => ui.label("no shot in this level"),
                };
            }
            ui.separator();
            ui.label("camera: drag to orbit, scroll to zoom");
            ui.label(format!("eye ({:.0}, {:.0}, {:.0}) mm", self.camera.eye.x, self.camera.eye.y, self.camera.eye.z));
            if ui.button("back to the level's camera").clicked() {
                self.camera = self.authored_camera;
                self.orbit = self.orbit_from_camera();
            }
        });

        egui::CentralPanel::default().show(root, |ui| {
            let avail = ui.available_size();
            let aspect = self.size.0 as f32 / self.size.1.max(1) as f32;
            let size = if avail.x / avail.y > aspect { egui::vec2(avail.y * aspect, avail.y) } else { egui::vec2(avail.x, avail.x / aspect) };
            let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::drag());
            if resp.dragged() {
                let d = resp.drag_delta();
                self.orbit.0 -= d.x as f64 * 0.005;
                self.orbit.1 = (self.orbit.1 + d.y as f64 * 0.005).clamp(-0.2, 1.4);
                self.apply_orbit();
            }
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if resp.hovered() && scroll.abs() > 0.0 {
                self.orbit.2 = (self.orbit.2 * (1.0 - scroll as f64 * 0.002)).clamp(1000.0, 40000.0);
                self.apply_orbit();
            }
            match &self.texture {
                Some(t) => {
                    ui.painter().image(t.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                }
                None => {
                    ui.painter().rect_filled(rect, 0.0, egui::Color32::from_gray(18));
                    let msg = self.render_status.lock().map(|s| s.clone()).unwrap_or_default();
                    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, msg, egui::FontId::proportional(16.0), egui::Color32::from_gray(170));
                }
            }
        });

        if n > 0 && self.asked.as_ref() != Some(&self.key()) {
            self.ask();
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(33));
    }
}

/// `kosm-view --court`: the court, live.
pub fn run(frames: usize, size: (u32, u32), spp: u32) -> eframe::Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    let (job_tx, job_rx) = std::sync::mpsc::channel();
    let (shot_tx, shot_rx) = std::sync::mpsc::channel();
    let sim_status: Status = Default::default();
    let render_status: Status = Default::default();
    set(&sim_status, "loading the court".into());
    set(&render_status, "evaluating the level".into());

    let s = sim_status.clone();
    std::thread::spawn(move || simulate(tx, s, frames));
    let s = render_status.clone();
    std::thread::spawn(move || render_worker(job_rx, shot_tx, s));

    // The level's camera, without waiting for the renderer's stage: cheap to
    // read, and the window wants it before the first frame arrives.
    let camera = CourtScene::bundled()
        .and_then(|s| {
            let a = &s.authored;
            Ok(Camera {
                eye: KVec3::new(a.parameter("cam_x_mm")?, a.parameter("cam_y_mm")?, a.parameter("cam_z_mm")?),
                target: KVec3::new(a.parameter("cam_at_x_mm")?, a.parameter("cam_at_y_mm")?, a.parameter("cam_at_z_mm")?),
                fov_deg: a.parameter_or("cam_vfov_deg", 42.0),
                exposure: a.parameter_or("exposure", 1.0) as f32,
            })
        })
        .unwrap_or(Camera {
            eye: KVec3::new(-800.0, -5600.0, 1900.0),
            target: KVec3::new(900.0, 300.0, 1900.0),
            fov_deg: 40.0,
            exposure: 1.0,
        });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 800.0]).with_title("Kosm view — the court"),
        ..Default::default()
    };
    eframe::run_native(
        "Kosm view",
        options,
        Box::new(move |_cc| Ok(Box::new(App::new(rx, shot_rx, job_tx, sim_status, render_status, camera, size, spp)))),
    )
}
