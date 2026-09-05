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
//! ## the two tiers
//!
//! `vcad-kernel-raytrace` has a `gpu` feature, and it is on. It used to be
//! unusable here: it pinned **wgpu 23** while the viewport's surface is on
//! **wgpu 30**, and two majors of wgpu are two unrelated sets of types — the
//! `Device` the surface hands out could not be passed to `RayTracePipeline`
//! at all. vcad is on wgpu 30 now, so the tracer runs on the viewport's own
//! device (`viewport::Scene::init` hands it over) and there is one adapter in
//! the process rather than two.
//!
//! The scene format was the other half of it. `GpuScene::from_brep` packed a
//! merged BRep with no per-instance transform, so a ball moving meant
//! re-packing the whole court; `GpuScene::placed` is the answer — each solid
//! packed once, every frame saying only where its instances are. See
//! `court_gpu.rs`. The CPU integrator, `pathtrace::render`, is the fallback:
//! `--cpu` asks for it, it is what runs when there is no adapter or the court
//! will not pack, and it is the reference the GPU picture is checked against.
//!
//! Whichever traced it, a pass is **one raw sample** and neither tracer
//! accumulates. [`crate::history`] does: a running mean and a sample count per
//! pixel, a reprojection through a moved camera when the pass brought guide
//! buffers with it, and a geometric mask that throws away only the pixels a
//! moved ball, its shadow, or a moved extra actually landed on. The GPU tier
//! has no guides to give — the shader's depth and normals never leave the
//! device — so its history is mask-only, and a camera that moves costs it the
//! whole picture where it costs the CPU one a few silhouettes. Both tiers
//! answer the same `Job` with the same `Shot`, so the window's tuner does not
//! know which one it is talking to. This file's job is the pace: how big to
//! ask for, at how many samples, which rectangles, and when a measurement was
//! fair enough to believe.
//!
//! ## the pass is the mask
//!
//! [`crate::history`] answers, before a pass runs, which rectangles the world
//! moved under. The CPU tier re-traces exactly those with
//! `pathtrace::render_into`, into a `Film` it keeps between passes so the
//! pixels it did not touch are last pass's rather than black; the GPU tier
//! sets the shader's scissor to their bounding box. A masked pass is only
//! taken when it saves more than half the frame, because the pixels outside it
//! gain nothing and a picture that is always masked never converges — so the
//! bounces buy cheap passes and the quiet between them buys full ones.

use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

use kosm_spike::court::render::{self, Snapshot};
use kosm_spike::court::{Court, CourtScene};
use vcad_kernel_math::{Point3, Vec3 as KVec3};
use vcad_kernel_raytrace::pathtrace;

use crate::court_gpu;
use crate::history::{Guides, History, Pose, Plan, View};
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
    /// Which frame of the recording this is. The GPU stage re-assembles its
    /// packed scene when this changes, and not when the camera does.
    pub frame_id: u64,
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
    /// Pixels this pass actually traced. A masked pass traces its rectangles
    /// and nothing else, so this — not the frame — is what the pass cost is
    /// per, and it is what the tuner's model is fitted against.
    pub traced_px: u64,
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
    // The extras are one thing, not a hundred and sixty-eight: the net is a
    // hundred and sixty-eight cords, and a mask that took each cord's sphere
    // and each cord's ten shadow discs covered the screen every pass the net
    // so much as swayed. One bounding sphere over all of them, and its shadow,
    // is what the eye needs repainted.
    let mut spheres: Vec<(Point3, f64)> = Vec::with_capacity(snap.extras.len());
    for extra in &snap.extras {
        if let Some((c, radius)) = stage.extra_sphere(extra) {
            spheres.push((Point3::new(c.x, c.y, c.z), radius));
        }
    }
    if !spheres.is_empty() {
        let n = spheres.len() as f64;
        let cx = spheres.iter().map(|(c, _)| c.x).sum::<f64>() / n;
        let cy = spheres.iter().map(|(c, _)| c.y).sum::<f64>() / n;
        let cz = spheres.iter().map(|(c, _)| c.z).sum::<f64>() / n;
        let radius = spheres
            .iter()
            .map(|(c, r)| ((c.x - cx).powi(2) + (c.y - cy).powi(2) + (c.z - cz).powi(2)).sqrt() + r)
            .fold(0.0, f64::max);
        out.push(Pose { centre: [cx, cy, cz], rot: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], radius });
    }
    out
}

/// Which tracer a pass came from. Chosen once, at the top of the worker, and
/// named on stderr so the numbers there are attributable.
enum Tracer {
    /// vcad's compute shader, on the window's own device.
    Gpu(Box<court_gpu::Stage>),
    /// vcad's CPU integrator.
    Cpu,
}

impl Tracer {
    fn name(&self) -> &'static str {
        match self {
            Tracer::Gpu(_) => "gpu",
            Tracer::Cpu => "cpu",
        }
    }
}

/// A GPU sample, dressed as a `Film` so the history takes it exactly the way
/// it takes a CPU one. The guide buffers stay zeroed — the tracer has none to
/// give — and zero depth is vcad's background sentinel, which is the "leave
/// this pixel alone" the denoiser already knows how to read.
fn blank_film(size: (u32, u32), sample: court_gpu::Sample) -> pathtrace::Film {
    let n = (size.0 as usize) * (size.1 as usize);
    pathtrace::Film {
        width: size.0,
        height: size.1,
        rgb: sample.rgb,
        alpha: sample.alpha,
        normal: vec![0.0; n * 3],
        depth: vec![0.0; n],
        albedo: vec![0.0; n * 3],
        variance: vec![0.0; n],
    }
}

/// The renderer: build the stage once, then keep adding passes to whatever
/// the window last asked for.
fn render_worker(jobs: Receiver<Job>, out: Sender<Shot>, gpu: Option<(wgpu::Device, wgpu::Queue)>) {
    // Evaluating the level and building its BVHs takes minutes, and the
    // window is black until it is done. Say so, or it looks broken.
    eprintln!("court  evaluating the level…");
    let t0 = Instant::now();
    let mut stage = match CourtScene::bundled().and_then(|s| render::Scene::new(&s)) {
        Ok(stage) => stage,
        Err(error) => return eprintln!("court: could not build the picture: {error}"),
    };
    eprintln!("court  {} vcad solids, {} panels, in {:.1} s", stage.static_count(), stage.light_count(), t0.elapsed().as_secs_f64());

    // The GPU tier if the surface handed a device over and the court packs for
    // it; the CPU integrator otherwise, saying which and why.
    let scene = CourtScene::bundled().ok();
    let mut tracer = gpu
        .and_then(|(device, queue)| {
            let a = scene.as_ref().map(|s| &s.authored);
            let depth = a.map_or(6.0, |a| a.parameter_or("max_depth", 6.0)).max(1.0) as u32;
            let env = a.map_or(0.05, |a| a.parameter_or("env_radiance", 0.05)) as f32;
            match court_gpu::Stage::new(&stage, &device, &queue, depth, env) {
                Ok(gpu) => Some(Tracer::Gpu(Box::new(gpu))),
                Err(error) => {
                    eprintln!("court  gpu: {error}; falling back to the CPU tracer");
                    None
                }
            }
        })
        .unwrap_or(Tracer::Cpu);
    eprintln!("court  the {} path tracer", tracer.name());

    let lights = stage.light_centres();

    // The history is the picture. A job is only ever "this snapshot, this
    // camera, this size, one sample" — neither tracer is allowed to decide
    // that the last frame was worthless. The history decides that, per pixel.
    let mut current: Option<Job> = None;
    let mut history = History::new((0, 0));
    // The CPU tier's frame, kept between passes: `render_into` patches it, so
    // the pixels a masked pass did not touch are last pass's and not black.
    let mut film = pathtrace::Film::new(0, 0);
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
        if (film.width, film.height) != job.size {
            film = pathtrace::Film::new(job.size.0, job.size.1);
        }
        let lap = Instant::now();
        let cam = job.camera.to_pathtrace();
        let view = View::of(&cam, job.size.0, job.size.1);
        let poses = poses(&mut stage, &job.frame);
        let seed = 0x5eed_0000 ^ (job.generation << 20) ^ (lap.elapsed().as_nanos() as u64) ^ passes_seed(&history);

        // The mask comes *first* now. It is computed from the poses alone —
        // where every ball and the net were, where they are — so it can be
        // known before a ray is cast, and both tracers can be handed the
        // rectangles instead of the frame. An empty plan means nothing moved,
        // and that is exactly when the whole frame is worth tracing: a still
        // world is how the picture gains samples.
        let plan: Plan = history.plan(&view, &poses, &lights);
        let frame_px = (job.size.0 as u64) * (job.size.1 as u64);
        // What each tier would actually trace for this plan. The GPU gets one
        // scissored dispatch over the union — `render_with_render_state`
        // rebuilds its buffers per call, so a dispatch per rectangle would pay
        // for the whole court once per rectangle — while the CPU traces the
        // rectangles themselves.
        let patch_px: u64 = match &tracer {
            Tracer::Gpu(_) => plan.bbox().map_or(0, |r| (r[2] as u64) * (r[3] as u64)),
            Tracer::Cpu => plan.rects.iter().map(|r| (r[2] as u64) * (r[3] as u64)).sum(),
        };
        // A masked pass is only worth having when it is genuinely most of the
        // frame cheaper. It buys its rays at a price: the pixels outside get
        // *nothing*, so a picture that is always masked never converges. Half
        // the frame is where the two stop trading evenly — and it is a real
        // gate on the GPU, whose scissor is one bounding box, so four balls
        // scattered across the picture can mask a fifth of it and still make a
        // dispatch that covers four fifths.
        let full = plan.full || plan.rects.is_empty() || patch_px * 2 > frame_px;
        let traced_px = if full { frame_px } else { patch_px };

        // One raw sample, whoever traced it: the GPU hands back linear
        // radiance and nothing else, the CPU fills the guides too.
        let (guides, traced) = match &mut tracer {
            Tracer::Gpu(gpu) => {
                let scissor = if full { None } else { plan.bbox() };
                match gpu.sample(&stage, &job.frame, job.frame_id, &job.camera, job.size, scissor) {
                    Ok(sample) => {
                        film = blank_film(job.size, sample);
                        // The scissor is one rectangle, so that rectangle —
                        // not the plan's several — is what came back fresh.
                        (Guides::None, scissor.map(|r| vec![r]))
                    }
                    Err(error) => {
                        eprintln!("court  gpu: {error}; falling back to the CPU tracer");
                        tracer = Tracer::Cpu;
                        history = History::new(job.size);
                        film = pathtrace::Film::new(job.size.0, job.size.1);
                        continue;
                    }
                }
            }
            Tracer::Cpu => {
                let scene = stage.at_snapshot(&job.frame);
                let opts = options(job.spp, seed, false);
                if full {
                    film = pathtrace::render(&scene, &cam, job.size.0, job.size.1, &opts);
                    (Guides::Film, None)
                } else {
                    // `render_into` patches the film in place and never
                    // denoises, so the pixels outside the rectangles are still
                    // the previous pass's — which is what the history wants,
                    // since it is about to be told not to look at them.
                    pathtrace::render_into(&scene, &cam, &mut film, &opts, &plan.rects);
                    (Guides::Film, Some(plan.rects.clone()))
                }
            }
        };
        history.merge(&film, guides, &view, &poses, &lights, traced.as_deref());
        // Denoise the resolved buffer, blended out as the counts climb. With
        // no guides the à-trous filter passes every pixel through untouched,
        // so on the GPU tier this is a no-op rather than a blur.
        let opts = options(job.spp, seed, true);
        let rgba = history.resolve(job.camera.exposure, &opts);
        let shot = Shot {
            size: job.size,
            rgba,
            spp: job.spp,
            ms: lap.elapsed().as_millis(),
            mask: history.mask_fraction(),
            mean_spp: history.mean_samples(),
            traced_px,
        };
        if said_at.elapsed().as_secs() >= 2 {
            said_at = Instant::now();
            eprintln!(
                "court  {} {}×{} at {} spp: {} ms a {} pass, {:.0}% repainted, {:.1} samples a pixel",
                tracer.name(),
                job.size.0,
                job.size.1,
                job.spp,
                shot.ms,
                if full {
                    "full".to_string()
                } else {
                    format!("{:.0}% ", 100.0 * traced_px as f64 / (job.size.0 as f64 * job.size.1 as f64))
                },
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

/// The same still, traced on the GPU. There is no window and so no surface
/// device, so this asks `vcad-kernel-gpu` for a headless one — the same
/// adapter, just nobody's surface — and averages `passes` raw samples, which
/// is what the window does too, only without a mask to complicate it.
pub fn still_gpu(path: &std::path::Path, t: f64, size: (u32, u32), passes: u32) -> anyhow::Result<()> {
    let scene = CourtScene::bundled()?;
    let stage = render::Scene::new(&scene)?;
    let camera = authored_camera(&scene);
    let a = &scene.authored;
    let ctx = vcad_kernel_gpu::GpuContext::init_blocking().map_err(|e| anyhow::anyhow!("no GPU adapter: {e}"))?;
    let mut gpu = court_gpu::Stage::new(
        &stage,
        &ctx.device,
        &ctx.queue,
        a.parameter_or("max_depth", 6.0).max(1.0) as u32,
        a.parameter_or("env_radiance", 0.05) as f32,
    )?;

    let t = if t < 0.0 { a.parameter_or("still_t", 0.95) } else { t };
    let mut court = Court::from_scene(&scene)?;
    while court.time() < t {
        court.step();
    }
    let frame = Frame::of(&court);

    let t0 = Instant::now();
    let n = (size.0 as usize) * (size.1 as usize);
    let mut sum = vec![0.0f32; n * 3];
    let mut alpha = vec![0.0f32; n];
    let passes = passes.max(1);
    for _ in 0..passes {
        let s = gpu.sample(&stage, &frame, 0, &camera, size, None)?;
        for (acc, v) in sum.iter_mut().zip(&s.rgb) {
            *acc += v;
        }
        for (acc, v) in alpha.iter_mut().zip(&s.alpha) {
            *acc += v;
        }
    }
    let k = 1.0 / passes as f32;
    let film = pathtrace::Film {
        width: size.0,
        height: size.1,
        rgb: sum.iter().map(|v| v * k).collect(),
        alpha: alpha.iter().map(|v| v * k).collect(),
        normal: vec![0.0; n * 3],
        depth: vec![0.0; n],
        albedo: vec![0.0; n * 3],
        variance: vec![0.0; n],
    };
    let rgba = film.to_srgb8(camera.exposure, false);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    image::RgbaImage::from_raw(size.0, size.1, rgba)
        .ok_or_else(|| anyhow::anyhow!("the film is the wrong size"))?
        .save(path)?;
    println!(
        "court  t = {:.2} s, {} balls, {} vcad solids; {}×{} over {passes} gpu passes in {:.2} s → {}",
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

/// Two measurements have to be this far apart in work before they count as
/// two points and not one noisy one.
const SPREAD: f64 = 1.5;

/// What a pass costs: a fixed part and a per-pixel part.
///
/// One term was never enough. A pass has a floor that does not scale with the
/// picture at all — the GPU tier re-uploads the whole court and drags the
/// frame back across the bus whatever its size; the CPU tier rebuilds the
/// net's BVH and forks a rayon pool. Milliseconds per megapixel per sample
/// folded that floor into the slope, so the model over-charged a big picture
/// and under-charged a small one, and the tuner oscillated: grow on a
/// prediction, overrun, shrink, go quiet, grow again.
///
/// So: `ms = fixed + per * work`, where `work` is megapixels actually traced
/// times samples. Fitting a line needs two points, and the passes supply them
/// for free now that a masked pass traces a fraction of the frame — the two
/// buckets below are exponential moving averages of the cheap end and the dear
/// end of whatever work has been asked for, and the line through them is the
/// model. Until they are far enough apart to be two points, it degenerates to
/// the old one-term fit through the origin, which is what a single
/// measurement can honestly say.
#[derive(Clone, Copy)]
struct Cost {
    lo: Option<(f64, f64)>,
    hi: Option<(f64, f64)>,
}

impl Cost {
    fn new() -> Self {
        Self { lo: None, hi: None }
    }

    /// Fold one (work, milliseconds) measurement in.
    ///
    /// The *work* coordinate barely moves — a tenth of the way — while the
    /// time is a normal moving average. That asymmetry is deliberate: two
    /// buckets that chase the work they are fed collapse onto whatever size
    /// the window is currently rendering, the spread closes, and the fit falls
    /// back to one term exactly when two are needed. A measurement well
    /// outside both ends does not blend at all; it *becomes* that end, because
    /// a new extreme is news and not noise.
    fn observe(&mut self, work: f64, ms: f64) {
        let blend = |p: &mut (f64, f64)| {
            p.0 = 0.9 * p.0 + 0.1 * work;
            p.1 = 0.7 * p.1 + 0.3 * ms;
        };
        match (&mut self.lo, &mut self.hi) {
            (None, _) => self.lo = Some((work, ms)),
            (Some(lo), None) => {
                if work > lo.0 * SPREAD {
                    self.hi = Some((work, ms));
                } else if work * SPREAD < lo.0 {
                    self.hi = Some(*lo);
                    self.lo = Some((work, ms));
                } else {
                    blend(lo);
                }
            }
            (Some(lo), Some(hi)) => {
                if work * SPREAD < lo.0 {
                    *lo = (work, ms);
                } else if work > hi.0 * SPREAD {
                    *hi = (work, ms);
                } else if work <= 0.5 * (lo.0 + hi.0) {
                    blend(lo);
                } else {
                    blend(hi);
                }
            }
        }
    }

    /// `(fixed ms, ms per megapixel per sample)`.
    fn terms(&self) -> (f64, f64) {
        match (self.lo, self.hi) {
            (Some(lo), Some(hi)) if hi.0 > lo.0 * 1.001 => {
                let per = ((hi.1 - lo.1) / (hi.0 - lo.0)).max(0.0);
                ((lo.1 - per * lo.0).max(0.0), per)
            }
            (Some(lo), _) if lo.0 > 0.0 => (0.0, lo.1 / lo.0),
            _ => (0.0, GUESS),
        }
    }

    /// What a pass of `work` megapixel-samples should cost.
    fn predict(&self, work: f64) -> f64 {
        let (fixed, per) = self.terms();
        fixed + per * work
    }
}

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
    /// The finest divisor not yet known to overrun. A size that has actually
    /// overrun is remembered here and never climbed back into until the window
    /// itself changes, so a mispredicted step does not become an oscillation
    /// — grow on a prediction, overrun, shrink back, go quiet, grow again —
    /// with the history thrown away at every step. [`Cost`] is what stops the
    /// prediction being wrong in the first place, but the two are belt and
    /// braces and both are cheap.
    floor: u32,
    /// Samples a pass now, and the most it is allowed to ask for.
    samples: u32,
    spp: u32,
    /// The measured cost of a pass: a fixed part and a per-pixel part.
    cost: Cost,
    /// What a *full* pass has actually been seen to cost at each divisor, in
    /// milliseconds, or zero for one never tried. The model says what a size
    /// should cost; this remembers what it did. Shrinking is vetoed when the
    /// smaller size has been measured and was not meaningfully cheaper —
    /// which on the GPU tier is most of the time, because the pass is mostly
    /// the court going up the bus and the frame coming back down.
    seen: [f64; MAX_SCALE as usize + 2],
    /// Consecutive passes that came back cheap and with an empty mask: the
    /// window only buys a bigger picture when the world has stopped repainting
    /// itself.
    quiet: u32,
    /// Consecutive passes that overran the budget. One is noise.
    over: u32,
    /// Consecutive passes that came back at under half the budget. Enough of
    /// them and a size the tuner had sworn off is worth trying again: the
    /// overrun that condemned it may have been the net minting a solid.
    cheap: u32,
    generation: u64,
    asked: Option<Ask>,
    /// The render thread's ends of the two channels, held until the surface
    /// exists. The tracer runs on the *window's* device, and there is no
    /// device until there is a window — so the thread that would use it
    /// cannot be started before then.
    pending: Option<(Receiver<Job>, Sender<Shot>)>,
    /// Forced by `--cpu`: never hand the render thread a device.
    cpu_only: bool,
}

impl App {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rx: Receiver<Frame>,
        shots: Receiver<Shot>,
        jobs: Sender<Job>,
        camera: Camera,
        spp: u32,
        pending: (Receiver<Job>, Sender<Shot>),
        cpu_only: bool,
    ) -> Self {
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
            floor: 1,
            samples: 1,
            spp: spp.max(1),
            cost: Cost::new(),
            seen: [0.0; MAX_SCALE as usize + 2],
            quiet: 0,
            over: 0,
            cheap: 0,
            generation: 0,
            asked: None,
            pending: Some(pending),
            cpu_only,
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

    /// The work a *full* pass at this divisor is, in megapixel-samples. Full,
    /// because that is the pass whose cost decides how big the picture may be:
    /// a masked pass is cheaper by definition and never the thing that has to
    /// fit.
    fn work(&self, scale: u32) -> f64 {
        let (w, h) = ((self.window.0 / scale).max(32), (self.window.1 / scale).max(18));
        w as f64 * h as f64 / 1e6 * self.samples.max(1) as f64
    }

    /// The part of a pass at this divisor that the *size* is paying for.
    ///
    /// Only this part answers to resolution. The fixed part is paid whether
    /// the picture is 512 pixels across or 91 — on the GPU tier it is the
    /// court crossing the bus and the frame coming back, and it is most of the
    /// pass — so charging the size for it is what made the old tuner shrink
    /// the picture to nothing chasing a budget no size could meet.
    fn pixel_ms(&self, scale: u32) -> f64 {
        let (_, per) = self.cost.terms();
        per * self.work(scale)
    }

    /// What is left of the frame's budget once the unavoidable is paid. Never
    /// less than half a frame: a fixed cost that has eaten the budget whole is
    /// a reason to stop growing, not a reason to shrink to a postage stamp.
    fn budget(&self) -> f64 {
        let (fixed, _) = self.cost.terms();
        (TARGET_MS - fixed).max(0.5 * TARGET_MS)
    }

    /// The largest picture whose *per-pixel* cost fits what is left.
    fn affordable(&self) -> u32 {
        (1..MAX_SCALE).find(|s| self.pixel_ms(*s) <= self.budget()).unwrap_or(MAX_SCALE)
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
    /// A pass that traced only a patch is charged for the patch: `traced_px`,
    /// not the frame. That is what makes the two-term fit possible at all —
    /// the masked passes and the full ones are the two work levels the line is
    /// drawn through, with no probe pass and no calibration phase.
    fn tune(&mut self, shot: &Shot) -> bool {
        let work = shot.traced_px as f64 / 1e6 * shot.spp.max(1) as f64;
        if work <= 0.0 {
            return false;
        }
        let ms = shot.ms as f64;
        // A pass wildly past what the standing model says is not a render: it
        // is the net minting a solid and paying for its BVH inside the timing.
        // The floor keeps a fast machine from rejecting everything.
        if ms > OUTLIER * self.cost.predict(work).max(1.0) {
            return false;
        }
        self.cost.observe(work, ms);
        // Only a full pass says anything about what a *size* costs; a masked
        // one traced a patch whose size is the world's business, not the
        // tuner's.
        if shot.traced_px >= (shot.size.0 as u64) * (shot.size.1 as u64) {
            let slot = &mut self.seen[(self.scale as usize).min(MAX_SCALE as usize + 1)];
            *slot = if *slot > 0.0 { 0.7 * *slot + 0.3 * ms } else { ms };
        }
        true
    }

    /// Would a step coarser actually be cheaper? Unknown counts as yes — the
    /// only way to find out is to try it once.
    fn shrinking_helps(&self) -> bool {
        let (here, coarser) = (self.seen[self.scale as usize], self.seen[(self.scale + 1) as usize]);
        here <= 0.0 || coarser <= 0.0 || coarser < 0.75 * here
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
            frame_id: self.frame_id(),
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
    fn init(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some((jobs, shots)) = self.pending.take() else { return };
        let gpu = (!self.cpu_only).then(|| (device.clone(), queue.clone()));
        std::thread::spawn(move || render_worker(jobs, shots, gpu));
    }

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
            self.cheap = if (shot.ms as f64) < 0.5 * TARGET_MS { self.cheap + 1 } else { 0 };
            // An overrun is only the size's fault if the size is paying for
            // an appreciable share of the pass. When the fixed cost dominates
            // — the GPU tier, where a pass is mostly the court going up and
            // the frame coming down — a smaller picture costs the same, and
            // shrinking buys nothing but a blurrier one.
            let size_matters = self.pixel_ms(self.scale) > 0.25 * self.cost.terms().0.max(1.0)
                && self.shrinking_helps();
            if shot.ms as f64 > TARGET_MS * 1.5 && size_matters {
                self.over += 1;
                if self.over >= 2 {
                    if self.samples > 1 {
                        self.samples = 1;
                    } else if self.scale < MAX_SCALE {
                        self.scale += 1;
                        // This size overran twice: it is not affordable, and
                        // no prediction gets to say otherwise.
                        self.floor = self.scale;
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
            self.floor = 1;
            self.samples = 1;
            self.quiet = 0;
        } else if self.quiet >= CLIMB_AFTER && (self.scale > self.floor || self.samples < self.spp || self.floor > 1) {
            if self.scale > self.floor && self.pixel_ms(self.scale - 1) <= self.budget() {
                self.scale -= 1;
            } else if self.scale == self.floor && self.floor > 1 && self.cheap >= 2 * CLIMB_AFTER {
                // Passes here have run at under half the budget for a while:
                // the size above was condemned by a pass that was not a
                // render. Give it another hearing.
                self.floor -= 1;
                self.scale -= 1;
                self.samples = 1;
                self.cheap = 0;
            } else if self.scale == self.floor && self.samples < self.spp {
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
///
/// The render thread is not started here. It renders on the window's own
/// device, and the window does not exist yet — `App::init` starts it once the
/// surface has come up and handed the device over.
pub fn run(frames: usize, spp: u32, cpu_only: bool) -> anyhow::Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    let (job_tx, job_rx) = std::sync::mpsc::channel();
    let (shot_tx, shot_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || simulate(tx, frames));

    // The level's camera, without waiting for the renderer's stage: cheap to
    // read, and the window wants it before the first frame arrives.
    let camera = CourtScene::bundled().map(|s| authored_camera(&s)).unwrap_or(Camera {
        eye: KVec3::new(-800.0, -5600.0, 1900.0),
        target: KVec3::new(900.0, 300.0, 1900.0),
        fov_deg: 40.0,
        exposure: 1.0,
    });

    viewport::run(
        "Kosm view — the court",
        (1280, 720),
        App::new(rx, shot_rx, job_tx, camera, spp, (job_rx, shot_tx), cpu_only),
    )
}
