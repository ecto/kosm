//! The court in the window: `kosm-view`.
//!
//! Three threads. The simulation steps on its own and hands over one snapshot
//! per frame, so the timeline is a recording; the renderer turns the frame
//! under the cursor into a picture, pass by pass; the viewport blits whatever
//! the renderer last handed back. The picture is the court's own,
//! `kosm_spike::court::render`: the level's roots evaluated by vcad into BRep
//! solids with one BVH each, materials by root name, the balls and the net
//! placed where phyz has them, the panels as area lights. This file owns the
//! window's camera and the pace; nothing here describes a shape or a material.
//!
//! ## why the CPU tier
//!
//! `vcad-kernel-raytrace` has a `gpu` feature, and it does compile in this
//! workspace — but it pins **wgpu 23**, while the viewport's surface is on
//! **wgpu 30**. Two major versions of wgpu are two unrelated sets of types:
//! the `Device`, `Queue`, `Buffer` and `Texture` the surface hands out cannot
//! be passed to `RayTracePipeline`, and a texture the tracer wrote cannot be
//! sampled by the blit. There is no conversion; they are different crates
//! that happen to share a name. So the GPU tracer could only run on a
//! *second*, headless wgpu-23 device with a full CPU readback per frame —
//! and its scene format (`GpuScene::from_brep`, one merged BRep with no
//! per-instance transform) would force the whole court to be rebuilt and
//! re-uploaded every time a ball moves.
//!
//! So this is the CPU path tracer, `pathtrace::render`, run small on a worker
//! thread and accumulated progressively: passes keep being added to the same
//! frame while nothing changes, and the accumulator is thrown away the moment
//! the cursor, the camera or the window size moves. It is the same integrator
//! the CLI's reference tier uses, at fewer samples.

use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

use kosm_spike::court::render::{self, Snapshot};
use kosm_spike::court::{Court, CourtScene};
use vcad_kernel_math::{Point3, Vec3 as KVec3};
use vcad_kernel_raytrace::pathtrace::{self, Film};

use crate::viewport;

// ---- the recording ----------------------------------------------------------

/// One frame of the recording: where everything that moves is. The court
/// itself never moves, so it is built once and lives in the renderer.
pub type Frame = Snapshot;

/// The court, stepping on its own thread, in wall-clock time: each frame is
/// due at its own moment and the solver takes fixed `dt` steps to reach it.
/// A machine that cannot keep up runs slow — the catch-up is capped, so a
/// late frame never asks for the work of every frame it missed. It does not
/// stop: the level's `t_end` is the recording's length, not the world's, and
/// a live window keeps its world running. `frames > 0` caps it, for a test.
///
/// There is no status panel to say any of this, so what it has to say it says
/// on stderr.
fn simulate(tx: Sender<Frame>, frames: usize) {
    let scene = match CourtScene::bundled() {
        Ok(scene) => scene,
        Err(error) => return eprintln!("court: could not load the scene: {error}"),
    };
    let mut court = match Court::from_scene(&scene) {
        Ok(court) => court,
        Err(error) => return eprintln!("court: could not build the court: {error}"),
    };
    let steps_per_frame = (1.0 / scene.fps / scene.dt).round().max(1.0) as usize;
    // At most four frames of solving for one frame of wall clock.
    let cap = 4 * steps_per_frame;
    let _ = tx.send(Frame::of(&court));
    let start = Instant::now();
    let mut said = false;
    for k in 1.. {
        if frames > 0 && k > frames {
            break;
        }
        let due = k as f64 / scene.fps;
        if let Some(nap) = std::time::Duration::from_secs_f64(due).checked_sub(start.elapsed()) {
            std::thread::sleep(nap);
        }
        let mut steps = 0;
        while court.time() + 0.5 * scene.dt < due && steps < cap {
            court.step();
            steps += 1;
        }
        if tx.send(Frame::of(&court)).is_err() {
            return;
        }
        if !said && court.time() >= scene.t_end {
            said = true;
            eprintln!("court  {:.2} s simulated in {:.1} s; still running", court.time(), start.elapsed().as_secs_f64());
        }
    }
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

/// The camera the level asks for, in its `cam_*` knobs.
fn authored_camera(scene: &CourtScene) -> Camera {
    let a = &scene.authored;
    let k = |name: &str, fallback: f64| a.parameter_or(name, fallback);
    Camera {
        eye: KVec3::new(k("cam_x_mm", -800.0), k("cam_y_mm", -5600.0), k("cam_z_mm", 1900.0)),
        target: KVec3::new(k("cam_at_x_mm", 900.0), k("cam_at_y_mm", 300.0), k("cam_at_z_mm", 1900.0)),
        fov_deg: k("cam_vfov_deg", 42.0),
        exposure: k("exposure", 1.0) as f32,
    }
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

// ---- the renderer, on its own thread ---------------------------------------

/// What the window asks for: a frame, a camera, a size, and how much to spend.
#[derive(Clone)]
pub struct Job {
    pub generation: u64,
    pub frame: Frame,
    pub camera: Camera,
    pub size: (u32, u32),
    pub spp: u32,
    /// Passes to accumulate before the renderer goes quiet.
    pub passes: u32,
}

/// What comes back: RGBA at the requested size, and what the pass cost at
/// what sample count. The window sizes itself by those numbers.
pub struct Shot {
    pub size: (u32, u32),
    pub rgba: Vec<u8>,
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
fn render_worker(jobs: Receiver<Job>, out: Sender<Shot>) {
    // Evaluating the level and building its BVHs takes minutes, and the
    // window is black until it is done. Say so, or it looks broken.
    eprintln!("court  evaluating the level…");
    let t0 = Instant::now();
    let mut stage = match CourtScene::bundled().and_then(|s| render::Scene::new(&s)) {
        Ok(stage) => stage,
        Err(error) => return eprintln!("court: could not build the picture: {error}"),
    };
    eprintln!("court  {} vcad solids, {} panels, in {:.1} s", stage.static_count(), stage.light_count(), t0.elapsed().as_secs_f64());

    let mut current: Option<Job> = None;
    let mut accum: Option<Accum> = None;
    // What the last stderr line said, and when: the window is retuning itself
    // constantly and the interesting thing is the resolution it settles on.
    let mut said: Option<((u32, u32), u32)> = None;
    let mut said_at = Instant::now();
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
        let scene = stage.at_snapshot(&job.frame);
        let film = pathtrace::render(&scene, &job.camera.to_pathtrace(), job.size.0, job.size.1, &options(job.spp, opts.seed, false));
        acc.add(film);
        let rgba = acc.resolve(job.camera.exposure, &opts);
        let shot = Shot { size: job.size, rgba, spp: job.spp, ms: lap.elapsed().as_millis() };
        if said != Some((job.size, job.spp)) || said_at.elapsed().as_secs() >= 2 {
            said = Some((job.size, job.spp));
            said_at = Instant::now();
            eprintln!(
                "court  {}×{} at {} spp: {} ms a pass, {} accumulated",
                job.size.0, job.size.1, job.spp, shot.ms, acc.passes
            );
        }
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
    let mut stage = render::Scene::new(&scene)?;
    let camera = authored_camera(&scene);
    let t = if t < 0.0 { scene.authored.parameter_or("still_t", 0.95) } else { t };
    let mut court = Court::from_scene(&scene)?;
    while court.time() < t {
        court.step();
    }
    let frame = Frame::of(&court);
    let t0 = Instant::now();
    let picture = stage.at_snapshot(&frame);
    let film = pathtrace::render(&picture, &camera.to_pathtrace(), size.0, size.1, &options(spp, 0x5eed_1234, true));
    let rgba = film.to_srgb8(camera.exposure, false);
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
        stage.static_count(),
        size.0,
        size.1,
        t0.elapsed().as_secs_f64(),
        path.display()
    );
    Ok(())
}

// ---- the window -------------------------------------------------------------

/// What a pass may cost: one frame at thirty a second. The window buys that
/// with resolution — it renders the window's pixel size over an integer
/// divisor and lets the blit upscale — and with samples, one per pass while
/// anything is moving.
const TARGET_MS: f64 = 30.0;

/// The coarsest the picture is allowed to get — a thirty-second of the window
/// on each side, which on a retina window is still a couple of hundred pixels
/// wide. A machine slower than that drops frames instead of blurring further.
const MAX_SCALE: u32 = 32;

/// Passes to sit on one picture before asking for a larger one. Growing the
/// picture throws the accumulator away, so it is worth a few passes first.
const CLIMB_AFTER: u32 = 6;

/// The cost of a pass before one has been timed, in milliseconds per
/// megapixel per sample. Deliberately pessimistic: the first picture should
/// be small and quick, not right.
const GUESS: f64 = 750.0;

/// What the picture is *of*: the frame, the camera to the millimetre, and the
/// window. A change here came from the viewer or from the solver, and the
/// accumulator is worthless. How big to render it and at how many samples is
/// not part of it — that is only the window buying itself a better picture of
/// the same subject, and it must not read as motion.
#[derive(Clone, PartialEq, Eq)]
struct Subject(u64, [i64; 7], (u32, u32));

/// What was last asked for: a subject, at a size, at a sample count.
#[derive(Clone, PartialEq, Eq)]
struct Ask(Subject, (u32, u32), u32);

/// The court on screen. It owns the recording and the camera and decides what
/// to ask the render thread for; the picture itself is the render thread's.
struct App {
    rx: Receiver<Frame>,
    shots: Receiver<Shot>,
    jobs: Sender<Job>,
    frames: Vec<Frame>,
    cursor: usize,
    /// Following the simulation as it happens. The window opens this way and
    /// space comes back to it; pausing is the exception.
    live: bool,
    orbit: (f64, f64, f64),
    camera: Camera,
    /// The camera the level asks for, to go back to.
    authored: Camera,
    /// The window, in physical pixels.
    window: (u32, u32),
    /// What the window's size is divided by to get the render size.
    scale: u32,
    /// Samples a pass now, and the most it is allowed to ask for.
    samples: u32,
    spp: u32,
    /// The measured cost of a pass, in milliseconds per megapixel per sample.
    cost: f64,
    /// The subject the renderer is working on, and the passes that have
    /// landed since the window last changed what it was asking for.
    subject: Option<Subject>,
    settled: u32,
    generation: u64,
    asked: Option<Ask>,
}

impl App {
    fn new(rx: Receiver<Frame>, shots: Receiver<Shot>, jobs: Sender<Job>, camera: Camera, spp: u32) -> Self {
        let mut app = Self {
            rx,
            shots,
            jobs,
            frames: Vec::new(),
            cursor: 0,
            live: true,
            orbit: (0.0, 0.0, 0.0),
            camera,
            authored: camera,
            window: (1280, 720),
            scale: 4,
            samples: 1,
            spp: spp.max(1),
            cost: GUESS,
            subject: None,
            settled: 0,
            generation: 0,
            asked: None,
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

    /// The render size: the window over the divisor, never degenerate.
    fn size(&self) -> (u32, u32) {
        ((self.window.0 / self.scale).max(32), (self.window.1 / self.scale).max(18))
    }

    /// One pass at one sample, in milliseconds, at this divisor.
    fn pass_ms(&self, scale: u32) -> f64 {
        let (w, h) = ((self.window.0 / scale).max(32), (self.window.1 / scale).max(18));
        self.cost * w as f64 * h as f64 / 1e6
    }

    /// The largest picture whose pass is predicted to fit in a frame.
    fn affordable(&self) -> u32 {
        (1..MAX_SCALE).find(|s| self.pass_ms(*s) <= TARGET_MS).unwrap_or(MAX_SCALE)
    }

    /// Re-estimate the cost of a pixel from a pass that actually happened. A
    /// slow mean: one odd pass should not resize the picture.
    fn tune(&mut self, shot: &Shot) {
        let work = shot.size.0 as f64 * shot.size.1 as f64 / 1e6 * shot.spp.max(1) as f64;
        if work > 0.0 {
            self.cost = 0.7 * self.cost + 0.3 * (shot.ms as f64 / work);
        }
    }

    /// What the frame under the cursor looks like, to the millimetre: two
    /// frames of a world at rest are the same subject, however many arrive,
    /// so a live window whose balls have stopped still gets to accumulate.
    fn frame_key(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: i64| {
            h ^= v as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        if let Some(frame) = self.frames.get(self.cursor) {
            for (c, _) in &frame.balls {
                mix((c.x * 1e3).round() as i64);
                mix((c.y * 1e3).round() as i64);
                mix((c.z * 1e3).round() as i64);
            }
            for extra in &frame.extras {
                let m = &extra.to_world.matrix;
                mix((m[(0, 3)]).round() as i64);
                mix((m[(1, 3)]).round() as i64);
                mix((m[(2, 3)]).round() as i64);
            }
        }
        h
    }

    fn subject(&self) -> Subject {
        let c = &self.camera;
        Subject(
            self.frame_key(),
            [
                c.eye.x as i64,
                c.eye.y as i64,
                c.eye.z as i64,
                c.target.x as i64,
                c.target.y as i64,
                c.target.z as i64,
                (c.fov_deg * 100.0) as i64,
            ],
            self.window,
        )
    }

    fn ask(&mut self) {
        let Some(frame) = self.frames.get(self.cursor) else { return };
        self.generation += 1;
        let job = Job {
            generation: self.generation,
            frame: frame.clone(),
            camera: self.camera,
            size: self.size(),
            spp: self.samples,
            // Keep going on this subject until the picture has converged,
            // then go quiet rather than burn a core on nothing. A live frame
            // supersedes it long before that.
            passes: 256,
        };
        let _ = self.jobs.send(job);
        self.asked = Some(Ask(self.subject(), self.size(), self.samples));
    }
}

impl viewport::Scene for App {
    fn event(&mut self, event: viewport::Event) {
        use viewport::{Event, Key};
        match event {
            Event::Resized(px) => self.window = px,
            Event::Drag(dx, dy) => {
                self.orbit.0 -= dx * 0.005;
                self.orbit.1 = (self.orbit.1 + dy * 0.005).clamp(-0.2, 1.4);
                self.apply_orbit();
            }
            Event::Zoom(ticks) => {
                self.orbit.2 = (self.orbit.2 * (1.0 - ticks * 0.08)).clamp(1000.0, 40000.0);
                self.apply_orbit();
            }
            // Live is the resting state: unpausing rejoins the simulation
            // where it has got to, not where the cursor was left.
            Event::Key(Key::Space) => self.live = !self.live,
            Event::Key(Key::Left) => {
                if !self.live {
                    self.cursor = self.cursor.saturating_sub(1);
                }
            }
            Event::Key(Key::Right) => {
                if !self.live {
                    self.cursor = (self.cursor + 1).min(self.frames.len().saturating_sub(1));
                }
            }
            Event::Key(Key::Home) => {
                self.camera = self.authored;
                self.orbit = self.orbit_from_camera();
            }
        }
    }

    fn image(&mut self) -> Option<viewport::Image> {
        while let Ok(frame) = self.rx.try_recv() {
            self.frames.push(frame);
        }
        let n = self.frames.len();
        if self.live && n > 0 {
            self.cursor = n - 1;
        }
        // Every shot is shown, even one the window has already moved past: a
        // pass takes longer than a redraw, so refusing stale ones would leave
        // the window blank for the whole of playback.
        let mut newest = None;
        while let Ok(shot) = self.shots.try_recv() {
            self.tune(&shot);
            self.settled += 1;
            newest = Some(viewport::Image { size: shot.size, rgba: shot.rgba });
        }
        // A new subject — the solver moved the balls, or the viewer moved the
        // camera — is worth only what a frame can pay for, at one sample.
        // An old one has stopped moving, whether because it is paused or
        // because the solver has fallen behind the clock, and every few
        // passes it buys back a step of resolution and then its samples.
        let subject = self.subject();
        if self.subject.as_ref() != Some(&subject) {
            self.subject = Some(subject);
            self.settled = 0;
            self.scale = self.affordable();
            self.samples = 1;
        } else if self.settled >= CLIMB_AFTER && (self.scale > 1 || self.samples < self.spp) {
            if self.scale > 1 {
                self.scale -= 1;
            } else {
                self.samples = self.spp;
            }
            self.settled = 0;
        }
        if n > 0 && self.asked != Some(Ask(self.subject(), self.size(), self.samples)) {
            self.ask();
        }
        newest
    }
}

/// `kosm-view`: the court, live.
pub fn run(frames: usize, spp: u32) -> anyhow::Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    let (job_tx, job_rx) = std::sync::mpsc::channel();
    let (shot_tx, shot_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || simulate(tx, frames));
    std::thread::spawn(move || render_worker(job_rx, shot_tx));

    // The level's camera, without waiting for the renderer's stage: cheap to
    // read, and the window wants it before the first frame arrives.
    let camera = CourtScene::bundled().map(|s| authored_camera(&s)).unwrap_or(Camera {
        eye: KVec3::new(-800.0, -5600.0, 1900.0),
        target: KVec3::new(900.0, 300.0, 1900.0),
        fov_deg: 40.0,
        exposure: 1.0,
    });

    viewport::run("Kosm view — the court", (1280, 720), App::new(rx, shot_rx, job_tx, camera, spp))
}
