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
//! thread — the same integrator the CLI's reference tier uses, at one sample a
//! pass. What it hands back is never thrown away wholesale: [`crate::history`]
//! keeps a running mean and a sample count per pixel, reprojects them through a
//! moved camera, and discards only the pixels a moved ball, its shadow, or a
//! moved extra actually landed on. This file's job is the pace — how big to ask
//! for, at how many samples, and when the measurement was fair enough to
//! believe.

use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

use kosm_spike::court::render::{self, Snapshot};
use kosm_spike::court::{Court, CourtScene};
use vcad_kernel_math::{Point3, Vec3 as KVec3};
use vcad_kernel_raytrace::pathtrace;

use crate::history::{History, Pose, View};
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
}

/// What comes back: RGBA at the requested size, and what the pass cost at
/// what sample count. The window sizes itself by those numbers.
pub struct Shot {
    pub size: (u32, u32),
    pub rgba: Vec<u8>,
    pub spp: u32,
    pub ms: u128,
    /// The share of the screen this pass had to start over on.
    pub mask: f32,
    /// Samples behind the average pixel of the picture that came back.
    pub mean_spp: f32,
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

/// The poses of everything that moves in a snapshot, in millimetres — what
/// the history needs to know which pixels the world changed under.
fn poses(stage: &mut render::Scene, snap: &Snapshot) -> Vec<Pose> {
    let r = stage.ball_radius_mm();
    let mut out = Vec::with_capacity(snap.balls.len() + snap.extras.len());
    for (centre, rot) in &snap.balls {
        let c = *centre * render::PER_M;
        // world → body; the placement is its transpose, but for "did it turn?"
        // either reading answers the same question.
        out.push(Pose {
            centre: [c.x, c.y, c.z],
            rot: [
                rot[(0, 0)], rot[(0, 1)], rot[(0, 2)], //
                rot[(1, 0)], rot[(1, 1)], rot[(1, 2)], //
                rot[(2, 0)], rot[(2, 1)], rot[(2, 2)],
            ],
            radius: r,
        });
    }
    for extra in &snap.extras {
        let Some((c, radius)) = stage.extra_sphere(extra) else { continue };
        let m = &extra.to_world.matrix;
        out.push(Pose {
            centre: [c.x, c.y, c.z],
            rot: [
                m[(0, 0)], m[(0, 1)], m[(0, 2)], //
                m[(1, 0)], m[(1, 1)], m[(1, 2)], //
                m[(2, 0)], m[(2, 1)], m[(2, 2)],
            ],
            radius,
        });
    }
    out
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

    let lights = stage.light_centres();

    // The history is the picture. A job is only ever "this snapshot, this
    // camera, this size, one sample" — nothing here is allowed to decide that
    // the last frame was worthless. The history decides that, per pixel.
    let mut current: Option<Job> = None;
    let mut history = History::new((0, 0));
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
        // Nothing new and nothing to converge on: wait rather than spin.
        if latest.is_none() && current.is_none() {
            match jobs.recv() {
                Ok(job) => latest = Some(job),
                Err(_) => return,
            }
        }
        if let Some(job) = latest {
            current = Some(job);
        }
        let Some(job) = current.clone() else { continue };
        if job.size.0 == 0 || job.size.1 == 0 {
            continue;
        }
        if history.size() != job.size {
            history = History::new(job.size);
        }
        let lap = Instant::now();
        let cam = job.camera.to_pathtrace();
        let view = View::of(&cam, job.size.0, job.size.1);
        let poses = poses(&mut stage, &job.frame);
        let seed = 0x5eed_0000 ^ (job.generation << 20) ^ (lap.elapsed().as_nanos() as u64) ^ passes_seed(&history);
        let scene = stage.at_snapshot(&job.frame);
        let film = pathtrace::render(&scene, &cam, job.size.0, job.size.1, &options(job.spp, seed, false));
        history.merge(&film, &view, &poses, &lights);
        let opts = options(job.spp, seed, true);
        let rgba = history.resolve(job.camera.exposure, &opts);
        let shot = Shot {
            size: job.size,
            rgba,
            spp: job.spp,
            ms: lap.elapsed().as_millis(),
            mask: history.mask_fraction(),
            mean_spp: history.mean_samples(),
        };
        if said_at.elapsed().as_secs() >= 2 {
            said_at = Instant::now();
            eprintln!(
                "court  {}×{} at {} spp: {} ms a pass, {:.0}% repainted, {:.1} samples a pixel",
                job.size.0,
                job.size.1,
                job.spp,
                shot.ms,
                100.0 * shot.mask,
                shot.mean_spp
            );
        }
        if out.send(shot).is_err() {
            return;
        }
    }
}

/// A seed that moves with the history, so a converging picture keeps drawing
/// fresh samples rather than the same one over and over.
fn passes_seed(history: &History) -> u64 {
    (history.mean_samples() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
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

/// How much of the screen may be repainting and still count as quiet. Not
/// zero: a court with twenty balls in flight and a net swinging never has an
/// empty mask, and waiting for one would pin the window at its coarsest size
/// forever. What the climb is really waiting for is evidence that most of the
/// frame is being *kept* — that growing the picture is not throwing away a
/// history that was about to be discarded anyway.
const QUIET_MASK: f32 = 0.35;

/// How far past the prediction a pass has to be before it is not a render at
/// all. The net's BVH rebuilds land two orders of magnitude out; a genuinely
/// mispredicted render never does.
const OUTLIER: f64 = 4.0;

/// Passes to sit on one picture before asking for a larger one. Growing the
/// picture is the one thing that does throw the history away, so it is worth
/// a few passes first.
const CLIMB_AFTER: u32 = 6;

/// The cost of a pass before one has been timed, in milliseconds per
/// megapixel per sample. Deliberately pessimistic: the first picture should
/// be small and quick, not right.
const GUESS: f64 = 750.0;

/// What was last asked for: the frame, the camera to the millimetre, the size,
/// and the sample count. Not a *subject* any more — nothing about a change
/// here throws the picture away, because the history keeps whatever survives
/// it. This exists only so the window does not re-send an identical job.
#[derive(Clone, PartialEq, Eq)]
struct Ask(u64, [i64; 7], (u32, u32), u32);

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
    /// The window, in physical pixels, and the size the tuner last sized
    /// itself for. A resize is the one thing that starts the climb over.
    window: (u32, u32),
    sized: (u32, u32),
    /// What the window's size is divided by to get the render size.
    scale: u32,
    /// Samples a pass now, and the most it is allowed to ask for.
    samples: u32,
    spp: u32,
    /// The measured cost of a pass, in milliseconds per megapixel per sample.
    cost: f64,
    /// Consecutive passes that came back cheap and with an empty mask: the
    /// window only buys a bigger picture when the world has stopped repainting
    /// itself.
    quiet: u32,
    /// Consecutive passes that overran the budget. One is noise.
    over: u32,
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
            sized: (0, 0),
            scale: 4,
            samples: 1,
            spp: spp.max(1),
            cost: GUESS,
            quiet: 0,
            over: 0,
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

    /// Re-estimate the cost of a pixel from a pass that actually happened, and
    /// say whether the pass counts as a fair measurement at all.
    ///
    /// The court's net hands the renderer fresh solids as it deforms, and the
    /// first pass to see one pays for its BVH inside the timing. That is a
    /// build, not a render, and it is orders of magnitude out — folding it in
    /// would collapse the picture to its coarsest size and keep it there for
    /// the rest of the session. So a pass more than [`OUTLIER`] times the
    /// standing prediction is thrown away whole: it neither retunes the cost
    /// nor counts as an overrun.
    fn tune(&mut self, shot: &Shot) -> bool {
        let work = shot.size.0 as f64 * shot.size.1 as f64 / 1e6 * shot.spp.max(1) as f64;
        if work <= 0.0 {
            return false;
        }
        let measured = shot.ms as f64 / work;
        if measured > OUTLIER * self.cost {
            return false;
        }
        self.cost = 0.7 * self.cost + 0.3 * measured;
        true
    }

    /// Which frame of the recording the cursor is on. That is the whole of
    /// what the renderer needs to identify it — there is no hashing of ball
    /// positions any more, because a moved ball no longer costs the picture.
    fn frame_id(&self) -> u64 {
        self.cursor as u64
    }

    fn ask_key(&self) -> Ask {
        let c = &self.camera;
        Ask(
            self.frame_id(),
            [
                c.eye.x as i64,
                c.eye.y as i64,
                c.eye.z as i64,
                c.target.x as i64,
                c.target.y as i64,
                c.target.z as i64,
                (c.fov_deg * 100.0) as i64,
            ],
            self.size(),
            self.samples,
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
        };
        let _ = self.jobs.send(job);
        self.asked = Some(self.ask_key());
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
            let fair = self.tune(&shot);
            newest = Some(viewport::Image { size: shot.size, rgba: shot.rgba });
            if !fair {
                continue;
            }
            // A pass that blew the budget twice running costs a step of
            // resolution, or the samples that bought it. A pass that came back
            // with little of the screen repainted is one more piece of
            // evidence that the picture is worth growing.
            if shot.ms as f64 > TARGET_MS * 1.5 {
                self.over += 1;
                if self.over >= 2 {
                    if self.samples > 1 {
                        self.samples = 1;
                    } else if self.scale < MAX_SCALE {
                        self.scale += 1;
                    }
                    self.over = 0;
                }
                self.quiet = 0;
            } else if shot.mask <= QUIET_MASK {
                self.over = 0;
                self.quiet += 1;
            } else {
                self.over = 0;
                self.quiet = 0;
            }
        }
        // Only the window's own size resets the climb. Everything else — a
        // ball crossing the frame, the camera swinging round — is the
        // history's business now, and it keeps whatever it can.
        if self.sized != self.window {
            self.sized = self.window;
            self.scale = self.affordable();
            self.samples = 1;
            self.quiet = 0;
        } else if self.quiet >= CLIMB_AFTER && (self.scale > 1 || self.samples < self.spp) {
            if self.scale > 1 && self.pass_ms(self.scale - 1) <= TARGET_MS {
                self.scale -= 1;
            } else if self.scale == 1 {
                self.samples = self.spp;
            }
            self.quiet = 0;
        }
        if n > 0 && self.asked != Some(self.ask_key()) {
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
