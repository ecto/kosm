//! The court in the window: `kosm-view`.
//!
//! Three threads. The simulation steps on its own and hands over one snapshot
//! per frame, so the timeline is a recording; the renderer turns the frame
//! under the cursor into a picture, pass by pass; the viewport blits whatever
//! the renderer last handed back.
//!
//! ## the world is computed, not predicted
//!
//! The frame under the cursor is not the newest frame there is. A pass costs
//! about ten times what a step of the court costs, so presenting the newest
//! frame presents a moment that is already a render old. Instead every frame
//! carries the wall-clock moment it is *for* ([`Timed`]), the window measures
//! the latency it actually has — a frame's moment against the blit of its
//! picture — and hands that back to the simulation as a head start. The
//! simulation runs that far ahead and the window asks for the frame due when
//! the picture will be on the glass.
//!
//! Nothing here interpolates and nothing here predicts. The court is
//! deterministic: the frame the renderer aims at is the frame the simulation
//! would have reached anyway, computed early. Paused, the head start is zero.
//! `KOSM_NO_LOOKAHEAD=1` puts the old pace back for comparison. The picture is the court's own,
//! `kosm::brep`: the level's roots evaluated by vcad into BRep
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
//! accumulates. [`kosm_view::history`] does: a running mean and a sample count per
//! pixel, a reprojection through a moved camera, and a geometric mask that
//! throws away only the pixels a moved ball, its shadow, or a moved extra
//! actually landed on. Both tiers bring the guide buffers the reprojection
//! and the denoiser read — vcad's `render_resident_linear` fills them on the
//! GPU exactly as `pathtrace::render` does on the CPU — and both answer the
//! same `Job` with the same `Shot`, so the window's tuner does not know which
//! one it is talking to. This file's job is the pace: how big to ask for, at
//! how many samples, which rectangles, and when a measurement was fair enough
//! to believe.
//!
//! ## the pass is the mask
//!
//! [`kosm_view::history`] answers, before a pass runs, which rectangles the world
//! moved under. The CPU tier re-traces exactly those with
//! `pathtrace::render_into`, into a `Film` it keeps between passes so the
//! pixels it did not touch are last pass's rather than black; the GPU tier
//! sets the shader's scissor to their bounding box. A masked pass is only
//! taken when it saves more than half the frame, because the pixels outside it
//! gain nothing and a picture that is always masked never converges — so the
//! bounces buy cheap passes and the quiet between them buys full ones.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use kosm::brep::{self as render, Snapshot};
use super::{Court, CourtScene};

use vcad_kernel_math::{Point3, Vec3 as KVec3};
use vcad_kernel_raytrace::pathtrace;

use super::game_gpu as court_gpu;
use kosm_view::history::{History, Plan, Pose, View};
use kosm_view::viewport;

// ---- the recording ----------------------------------------------------------

/// One frame of the recording: where everything that moves is. The court
/// itself never moves, so it is built once and lives in the renderer.
pub type Frame = Snapshot;

/// A frame and the wall-clock moment it is *for*.
///
/// The recording used to be a bare list and the window showed its last entry.
/// That is the frame the simulation has *already* reached, and by the time a
/// pass of it is on the glass it is one render behind the world. A frame that
/// knows when it is due can be asked for early instead.
#[derive(Clone)]
struct Timed {
    frame: Frame,
    /// When this frame should be on the glass.
    due: Instant,
}

/// How far ahead of the wall clock the simulation is allowed to run, in
/// microseconds, and the one number the window writes for it. The window
/// measures what a picture actually costs between a frame's moment and its
/// blit, and asks the simulation for exactly that much of a head start.
///
/// It is a head start and not a prediction. The court is deterministic and
/// costs about a millisecond of solver for thirty of render, so the frame the
/// renderer aims at is *computed* — the same state the simulation would reach
/// on its own clock, arrived at early.
type Lookahead = Arc<AtomicU64>;

/// The most the simulation will run ahead: four frames. Past that a pause or
/// a stall would have it solving a world nobody is going to see.
const MAX_LOOKAHEAD: Duration = Duration::from_millis(80);

/// How far behind the wall clock the simulation will chase before it gives up
/// chasing.
///
/// A render spike — the net minting a hundred solids and their BVHs — can
/// take the core out from under the solver for a few frames. The debt that
/// leaves is permanent if the frame numbering is nailed to a fixed origin:
/// every frame after it is due in the past, the window is handed a stale
/// moment however early it asks, and the measured latency climbs without
/// bound. It was climbing: a session that had been holding 30 ms would find
/// the simulation a second behind and stay there. So past this much slip the
/// clock is re-based — **the world runs slow rather than behind**, which is
/// what the capped catch-up already says about the solver.
const MAX_SLIP: Duration = Duration::from_millis(120);

/// The court, stepping on its own thread, in wall-clock time: each frame is
/// due at its own moment and the solver takes fixed `dt` steps to reach it.
/// A machine that cannot keep up runs slow — the catch-up is capped, so a
/// late frame never asks for the work of every frame it missed. It does not
/// stop: the level's `t_end` is the recording's length, not the world's, and
/// a live window keeps its world running. `frames > 0` caps it, for a test.
///
/// What is new is *when* it does the work. A frame due at `start + k/fps` is
/// solved `lookahead` early, so that by the time the renderer has finished a
/// pass of it the moment it was for has arrived. Nothing about the frame
/// changes — the same `dt` steps to the same sim time — only the wall clock
/// it is computed on. The simulation never waits for the renderer.
///
/// There is no status panel to say any of this, so what it has to say it says
/// on stderr.
fn simulate(tx: Sender<Timed>, frames: usize, lookahead: Lookahead) {
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
    let mut start = Instant::now();
    let _ = tx.send(Timed { frame: super::snapshot(&court), due: start });
    let mut said = false;
    let mut solved = Duration::ZERO;
    let mut snapped = Duration::ZERO;
    let mut said_at = Instant::now();
    let mut said_k = 0usize;
    let mut slipped = Duration::ZERO;
    let mut slips = 0u32;
    for k in 1.. {
        if frames > 0 && k > frames {
            break;
        }
        let due_t = k as f64 / scene.fps;
        let mut due = start + Duration::from_secs_f64(due_t);
        let now = Instant::now();
        if now > due + MAX_SLIP {
            slipped += now.duration_since(due);
            slips += 1;
            start = now - Duration::from_secs_f64(due_t);
            due = now;
        }
        let ahead = Duration::from_micros(lookahead.load(Ordering::Relaxed)).min(MAX_LOOKAHEAD);
        if let Some(nap) = due
            .checked_sub(ahead)
            .and_then(|at| at.checked_duration_since(Instant::now()))
        {
            std::thread::sleep(nap);
        }
        let lap = Instant::now();
        let mut steps = 0;
        while court.time() + 0.5 * scene.dt < due_t && steps < cap {
            court.step();
            steps += 1;
        }
        solved += lap.elapsed();
        let snap = Instant::now();
        if tx.send(Timed { frame: super::snapshot(&court), due }).is_err() {
            return;
        }
        snapped += snap.elapsed();
        if said_at.elapsed().as_secs() >= 2 {
            let f = (k - said_k) as f64;
            eprintln!(
                "court  sim: {:.2} ms solving, {:.2} ms snapshotting a frame; {:.0} frames a second of wall clock{}",
                solved.as_secs_f64() * 1e3 / f,
                snapped.as_secs_f64() * 1e3 / f,
                f / said_at.elapsed().as_secs_f64(),
                if slips > 0 {
                    format!(
                        "; slipped {slips}\u{d7} ({:.0} ms given up on)",
                        slipped.as_secs_f64() * 1e3
                    )
                } else {
                    String::new()
                },
            );
            said_at = Instant::now();
            said_k = k;
            solved = Duration::ZERO;
            snapped = Duration::ZERO;
            slipped = Duration::ZERO;
            slips = 0;
        }
        if !said && court.time() >= scene.t_end {
            said = true;
            eprintln!(
                "court  {:.2} s simulated in {:.1} s; still running",
                court.time(),
                start.elapsed().as_secs_f64(),
            );
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
        eye: KVec3::new(
            k("cam_x_mm", -800.0),
            k("cam_y_mm", -5600.0),
            k("cam_z_mm", 1900.0),
        ),
        target: KVec3::new(
            k("cam_at_x_mm", 900.0),
            k("cam_at_y_mm", 300.0),
            k("cam_at_z_mm", 1900.0),
        ),
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
    /// The wall-clock moment this frame is *for*. It rides through the
    /// renderer untouched and comes back on the [`Shot`], which is the whole
    /// of how the window measures what it costs to show a moment: the frame's
    /// due time against the instant its picture is handed to the blit.
    pub due: Instant,
}

/// What comes back: the picture at the requested size, and what the pass cost
/// at what sample count. The window sizes itself by those numbers.
///
/// The picture is bytes from the CPU tier and a *texture* from the GPU one —
/// the device history tonemaps into a storage texture on the viewport's own
/// device and nothing is read back — which is why this is a
/// [`viewport::Image`] and not a `Vec<u8>`.
pub struct Shot {
    pub size: (u32, u32),
    pub image: viewport::Image,
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
    /// Whether stepping the render size would throw the accumulated picture
    /// away. False on the CPU tier, which resamples its history across a size
    /// step; true on the GPU one, whose history is in device buffers vcad
    /// reallocates — nothing on this side can resample them without a
    /// readback, and the whole point of that tier is that nothing comes back.
    pub resize_costs_history: bool,
    /// The due time of the frame this pass was of.
    pub due: Instant,
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
                rot[(0, 0)],
                rot[(0, 1)],
                rot[(0, 2)], //
                rot[(1, 0)],
                rot[(1, 1)],
                rot[(1, 2)], //
                rot[(2, 0)],
                rot[(2, 1)],
                rot[(2, 2)],
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
        out.push(Pose {
            centre: [cx, cy, cz],
            rot: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            radius,
        });
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

/// The renderer: build the stage once, then keep adding passes to whatever
/// the window last asked for.
fn render_worker(jobs: Receiver<Job>, out: Sender<Shot>, gpu: Option<(wgpu::Device, wgpu::Queue)>) {
    // Evaluating the level and building its BVHs takes minutes, and the
    // window is black until it is done. Say so, or it looks broken.
    eprintln!("court  evaluating the level…");
    let t0 = Instant::now();
    let mut stage = match CourtScene::bundled().and_then(|s| render::Scene::new(&s.authored, s.ball_r)) {
        Ok(stage) => stage,
        Err(error) => return eprintln!("court: could not build the picture: {error}"),
    };
    eprintln!(
        "court  {} vcad solids, {} panels, in {:.1} s",
        stage.static_count(),
        stage.light_count(),
        t0.elapsed().as_secs_f64()
    );

    // The GPU tier if the surface handed a device over and the court packs for
    // it; the CPU integrator otherwise, saying which and why.
    let scene = CourtScene::bundled().ok();
    let mut tracer = gpu
        .and_then(|(device, queue)| {
            let a = scene.as_ref().map(|s| &s.authored);
            let depth = a.map_or(6.0, |a| a.parameter_or("max_depth", 6.0)).max(1.0) as u32;
            match court_gpu::Stage::new(&stage, &device, &queue, depth) {
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

    // The history is the picture — on whichever side of the bus it lives.
    // The CPU tier keeps it here: a running mean, a count, a reprojection.
    // The GPU tier keeps it on the device and keeps only [`Mask`] here, which
    // is the one question the device cannot answer — which pixels last
    // frame's mean is still true for.
    let mut current: Option<Job> = None;
    let mut history = History::new((0, 0));
    // The device's own mean history length, read back at most every two
    // seconds and only for the log and the resize freeze — a pass still reads
    // nothing back.
    let mut gpu_mean_spp = 0.0f32;
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
        // A size step is a new grid, not a new picture. The CPU tier carries
        // its whole history across — mean, count and guides, bilinearly — so
        // the tuner can step the resolution without the window going back to
        // looking like a blizzard. The GPU tier cannot: its history is in
        // device buffers vcad reallocates on a resize, so there the tuner is
        // asked not to step a converged picture at all (see `App::image`).
        history.resample(job.size);
        if (film.width, film.height) != job.size {
            film = pathtrace::Film::new(job.size.0, job.size.1);
        }
        let lap = Instant::now();
        let frame_px = (job.size.0 as u64) * (job.size.1 as u64);

        // A pass, on whichever tier. Both answer the same question — what
        // does the picture look like now, what did it cost, how much of it
        // started over — and the tuner above does not know which one it is
        // talking to.
        let resize_costs_history = matches!(tracer, Tracer::Gpu(_));
        // Whether this pass carried its history across a camera move, for the
        // log: a moved camera used to repaint the frame and now mostly does
        // not.
        let (image, mask_frac, mean_spp, traced_px, kind) = match &mut tracer {
            // On the device: one full-frame pass and a texture. Nothing comes
            // back and there is no mask on this side at all — the history is
            // carried pixel by pixel, by the previous camera and by each
            // moving instance's `prev_T \u{b7} cur_T\u{207b}\u{b9}`, and shortened where
            // this pass's own neighbourhood says it has gone stale.
            Tracer::Gpu(gpu) => {
                let pass = gpu.accumulate(
                    &stage,
                    &job.frame,
                    job.frame_id,
                    &job.camera,
                    job.size,
                    job.spp,
                );
                match pass {
                    Ok(texture) => {
                        // The one thing this tier reads back, and only for the
                        // log line and the tuner's resize freeze: the mean of
                        // the device's own per-pixel counts, at the same two
                        // seconds the log runs on.
                        if said_at.elapsed().as_secs() >= 2 {
                            if let Ok(counts) = gpu.history_counts() {
                                if !counts.is_empty() {
                                    gpu_mean_spp = counts.iter().map(|&c| c as f64).sum::<f64>()
                                        as f32
                                        / counts.len() as f32;
                                }
                            }
                        }
                        (
                            viewport::Image::Texture(texture),
                            0.0,
                            gpu_mean_spp,
                            // Every pass is the whole frame now, which is what
                            // the tuner's cost model is fitted against.
                            frame_px,
                            "temporal".to_string(),
                        )
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
            // On this side: the mask comes first, the tracer re-traces exactly
            // the rectangles it names, and the history folds the film in and
            // denoises it.
            Tracer::Cpu => {
                let cam = job.camera.to_pathtrace();
                let view = View::of(&cam, job.size.0, job.size.1);
                let poses = poses(&mut stage, &job.frame);
                let seed = 0x5eed_0000
                    ^ (job.generation << 20)
                    ^ (lap.elapsed().as_nanos() as u64)
                    ^ passes_seed(&history);
                let plan: Plan = history.plan(&view, &poses, &lights);
                let patch_px: u64 = plan
                    .rects
                    .iter()
                    .map(|r| (r[2] as u64) * (r[3] as u64))
                    .sum();
                // A masked pass is only worth having when it is genuinely
                // most of the frame cheaper. The pixels outside it get
                // *nothing*, so a picture that is always masked never
                // converges; half the frame is where the two stop trading
                // evenly.
                let full = plan.full || plan.rects.is_empty() || patch_px * 2 > frame_px;
                let scene = stage.at_snapshot(&job.frame);
                let opts = options(job.spp, seed, false);
                let traced = if full {
                    film = pathtrace::render(&scene, &cam, job.size.0, job.size.1, &opts);
                    None
                } else {
                    // `render_into` patches the film in place and never
                    // denoises, so the pixels outside the rectangles are still
                    // the previous pass's — which is what the history wants,
                    // since it is about to be told not to look at them.
                    pathtrace::render_into(&scene, &cam, &mut film, &opts, &plan.rects);
                    Some(plan.rects.clone())
                };
                let t_merge = Instant::now();
                history.merge(&film, &view, &poses, &lights, traced.as_deref());
                let merge_ms = t_merge.elapsed().as_secs_f64() * 1e3;
                let t_res = Instant::now();
                let opts = options(job.spp, seed, true);
                let rgba = history.resolve(job.camera.exposure, &opts);
                if std::env::var("KOSM_GPU_TIMING").is_ok() {
                    eprintln!(
                        "court  history: {}\u{d7}{} \u{2014} {merge_ms:.1} ms merging, {:.1} ms resolving",
                        job.size.0,
                        job.size.1,
                        t_res.elapsed().as_secs_f64() * 1e3,
                    );
                }
                let traced_px = if full { frame_px } else { patch_px };
                (
                    viewport::Image::Bytes {
                        size: job.size,
                        rgba,
                    },
                    history.mask_fraction(),
                    history.mean_samples(),
                    traced_px,
                    if full {
                        "full".to_string()
                    } else {
                        format!(
                            "{}-box ({:.0}%)",
                            plan.rects.len(),
                            100.0 * traced_px as f64 / frame_px as f64
                        )
                    },
                )
            }
        };
        let shot = Shot {
            size: job.size,
            image,
            spp: job.spp,
            ms: lap.elapsed().as_millis(),
            mask: mask_frac,
            mean_spp,
            traced_px,
            resize_costs_history,
            due: job.due,
        };
        if said_at.elapsed().as_secs() >= 2 {
            said_at = Instant::now();
            // The GPU tier has no repainted share to report — nothing on it
            // repaints a rectangle any more — so it says what the history is
            // instead: the mean number of samples behind a pixel.
            let detail = match &tracer {
                Tracer::Gpu(_) => format!("{:.1} samples a pixel", shot.mean_spp),
                Tracer::Cpu => format!(
                    "{:.0}% repainted, {:.1} samples a pixel",
                    100.0 * shot.mask,
                    shot.mean_spp
                ),
            };
            eprintln!(
                "court  {} {}×{} at {} spp: {} ms a {} pass, {}",
                tracer.name(),
                job.size.0,
                job.size.1,
                job.spp,
                shot.ms,
                kind,
                detail
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
    let mut stage = render::Scene::new(&scene.authored, scene.ball_r)?;
    let camera = authored_camera(&scene);
    let t = if t < 0.0 {
        scene.authored.parameter_or("still_t", 0.95)
    } else {
        t
    };
    let mut court = Court::from_scene(&scene)?;
    while court.time() < t {
        court.step();
    }
    let frame = super::snapshot(&court);
    let t0 = Instant::now();
    let picture = stage.at_snapshot(&frame);
    let film = pathtrace::render(
        &picture,
        &camera.to_pathtrace(),
        size.0,
        size.1,
        &options(spp, 0x5eed_1234, true),
    );
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
/// adapter, just nobody's surface — and takes `passes` samples through the
/// same device-side history the window uses. There is one readback in the
/// whole of it, at the end, for the PNG.
pub fn still_gpu(
    path: &std::path::Path,
    t: f64,
    size: (u32, u32),
    passes: u32,
) -> anyhow::Result<()> {
    let scene = CourtScene::bundled()?;
    let stage = render::Scene::new(&scene.authored, scene.ball_r)?;
    let camera = authored_camera(&scene);
    let a = &scene.authored;
    let ctx = vcad_kernel_gpu::GpuContext::init_blocking()
        .map_err(|e| anyhow::anyhow!("no GPU adapter: {e}"))?;
    let mut gpu = court_gpu::Stage::new(
        &stage,
        &ctx.device,
        &ctx.queue,
        a.parameter_or("max_depth", 6.0).max(1.0) as u32,
    )?;

    let t = if t < 0.0 {
        a.parameter_or("still_t", 0.95)
    } else {
        t
    };
    let mut court = Court::from_scene(&scene)?;
    while court.time() < t {
        court.step();
    }
    let frame = super::snapshot(&court);

    let t0 = Instant::now();
    let passes = passes.max(1);
    gpu.always_denoise();
    // The window's loop, without a window: `passes` samples folded into the
    // device-side mean and denoised there, with an all-keep mask because
    // nothing moves between them. The picture never leaves the device until
    // the last line, which reads the target texture once for the PNG.
    for _ in 0..passes {
        gpu.accumulate(&stage, &frame, 0, &camera, size, 1)?;
    }
    let rgba = gpu.read_target()?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    image::RgbaImage::from_raw(size.0, size.1, rgba)
        .ok_or_else(|| anyhow::anyhow!("the target texture is the wrong size"))?
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

/// A scripted orbit, headless: converge, move the camera, and ask the device
/// what the move cost.
///
/// The claim this checks is the one the reprojection exists for. `passes`
/// samples are folded at the level's own camera; the eye is then orbited
/// `deg` about its target and one more pass is taken, with the same keep mask
/// a *still* camera would have got and the previous pass's camera as the view
/// to reproject from. A pixel that kept its history comes back with more
/// samples than the one this pass just gave it; a pixel the move disoccluded
/// comes back at one. Before this, every pixel came back at one.
pub fn orbit_test(t: f64, size: (u32, u32), passes: u32, deg: f64) -> anyhow::Result<()> {
    let scene = CourtScene::bundled()?;
    let stage = render::Scene::new(&scene.authored, scene.ball_r)?;
    let camera = authored_camera(&scene);
    let a = &scene.authored;
    let ctx = vcad_kernel_gpu::GpuContext::init_blocking()
        .map_err(|e| anyhow::anyhow!("no GPU adapter: {e}"))?;
    let mut gpu = court_gpu::Stage::new(
        &stage,
        &ctx.device,
        &ctx.queue,
        a.parameter_or("max_depth", 6.0).max(1.0) as u32,
    )?;
    let t = if t < 0.0 {
        a.parameter_or("still_t", 0.95)
    } else {
        t
    };
    let mut court = Court::from_scene(&scene)?;
    while court.time() < t {
        court.step();
    }
    let frame = super::snapshot(&court);

    let passes = passes.max(1);
    for _ in 0..passes {
        gpu.accumulate(&stage, &frame, 0, &camera, size, 1)?;
    }
    let before = gpu.history_counts()?;
    let converged = before.iter().filter(|&&c| c > 1).count();

    // The same eye, swung `deg` about the target in the floor plane.
    let moved = {
        let d = camera.eye - camera.target;
        let a = deg.to_radians();
        let (s, c) = (a.sin(), a.cos());
        Camera {
            eye: camera.target + KVec3::new(d.x * c - d.y * s, d.x * s + d.y * c, d.z),
            ..camera
        }
    };
    // One more pass, from the moved eye. There is no mask to build: the
    // previous camera goes to the device on its own and every pixel is asked
    // whether the surface under it is the one it had.
    gpu.accumulate(&stage, &frame, 0, &moved, size, 1)?;
    anyhow::ensure!(gpu.reprojected(), "the pass should have reprojected");

    let after = gpu.history_counts()?;
    let kept = after.iter().filter(|&&c| c > 1).count();
    let n = after.len().max(1);
    println!(
        "court  orbit: {}\u{d7}{} \u{2014} {passes} passes, then {deg}\u{b0}: {:.1}% of the frame had a history, {:.1}% kept it across the move",
        size.0,
        size.1,
        100.0 * converged as f64 / n as f64,
        100.0 * kept as f64 / n as f64,
    );
    Ok(())
}

/// A short sequence of *live-tier* frames, headless, with the history running
/// through them.
///
/// `--shot` is one frame folded many times and says nothing about motion.
/// This is the other half: the court is stepped to `t` and then frame by
/// frame at the level's own frame rate, one pass a frame, through the same
/// device history the window uses — so what lands in `dir` is what the window
/// would have shown, grain, ghosting and all. It is how the rectangle's
/// absence is checked by eye.
pub fn dump_frames(dir: &std::path::Path, t: f64, size: (u32, u32), n: u32) -> anyhow::Result<()> {
    let scene = CourtScene::bundled()?;
    let stage = render::Scene::new(&scene.authored, scene.ball_r)?;
    let camera = authored_camera(&scene);
    let a = &scene.authored;
    let ctx = vcad_kernel_gpu::GpuContext::init_blocking()
        .map_err(|e| anyhow::anyhow!("no GPU adapter: {e}"))?;
    let mut gpu = court_gpu::Stage::new(
        &stage,
        &ctx.device,
        &ctx.queue,
        a.parameter_or("max_depth", 6.0).max(1.0) as u32,
    )?;
    let t = if t < 0.0 {
        a.parameter_or("still_t", 0.95)
    } else {
        t
    };
    let mut court = Court::from_scene(&scene)?;
    while court.time() < t {
        court.step();
    }
    std::fs::create_dir_all(dir)?;
    let steps_per_frame = (1.0 / scene.fps / scene.dt).round().max(1.0) as usize;
    // The window is never cold when the balls are in flight — it has been
    // converging on the frames before this one — so the sequence starts the
    // way the window would: a few passes on the first frame, and then one a
    // frame like a live tier.
    let warm = super::snapshot(&court);
    for _ in 0..8 {
        gpu.accumulate(&stage, &warm, 0, &camera, size, 1)?;
    }
    let t0 = Instant::now();
    for k in 0..n.max(1) {
        if k > 0 {
            for _ in 0..steps_per_frame {
                court.step();
            }
        }
        let frame = super::snapshot(&court);
        let lap = Instant::now();
        gpu.accumulate(&stage, &frame, k as u64 + 1, &camera, size, 1)?;
        let rgba = gpu.read_target()?;
        let path = dir.join(format!("frame_{k:02}.png"));
        image::RgbaImage::from_raw(size.0, size.1, rgba)
            .ok_or_else(|| anyhow::anyhow!("the target texture is the wrong size"))?
            .save(&path)?;
        println!(
            "court  frame {k:02}  t = {:.3} s  {} ms a temporal pass  \u{2192} {}",
            court.time(),
            lap.elapsed().as_millis(),
            path.display()
        );
    }
    println!(
        "court  {} frames at {}\u{d7}{} in {:.1} s",
        n.max(1),
        size.0,
        size.1,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Build the denoiser's v2 training set: the device's own history, sequence by
/// sequence, at the sizes the viewer runs at.
///
/// # Why this lives in the viewer
///
/// The v1 dataset was CPU films — the mean of *k* independent one-sample
/// passes and the variance of that mean — and the network that was fitted to
/// it ran on something else entirely: an exponential moving average, carried
/// across motion by a reprojection and shortened by a neighbourhood clamp. It
/// won by 16–23% on held-out CPU tiles and rendered the backboard 1.7× too
/// bright in the window. So v2 records what the neural pass is *handed*, out
/// of the same buffers it binds, after the same passes the window runs — and
/// the only thing that can drive those is the thing that owns the device.
///
/// # What a sequence is
///
/// One (camera, time) state and one *history length*. The court is driven
/// frame by frame exactly as [`dump_frames`] drives it — the simulation
/// stepping at the level's own rate, one pass a frame, the history
/// reprojecting and clamping as it goes — for as many frames as the
/// sequence's tier, and stopped there. Two thirds of the sequences also
/// **move the camera** partway through, because a viewport is not a tripod
/// and the pixels the filter matters most for are the short-history ones a
/// move leaves behind.
///
/// One length per sequence, and not a snapshot at every length along the way,
/// because the reference has to be the answer to *this* frame. The balls are
/// in flight and the net is swinging: the converged picture of the frame
/// where a pixel has one sample is a different picture from the converged
/// picture thirty frames later, and pairing the first with the second would
/// train the network to predict the future. Covering the tiers is therefore a
/// rotation across sequences.
///
/// One extra pass is taken after the tier and recorded as the successor
/// frame. Nothing references it, because the temporal consistency term
/// compares the network's two answers to each other.
///
/// The reference is taken from the same camera and the same simulated
/// instant with the history cleared, the filter off, the clamp off and the
/// cap lifted: `reference_spp` passes of the bare path tracer.
///
/// Sizes rotate through [`DATASET_SIZES`] — the viewport's own scale
/// divisors — because a 5x5 kernel covers different amounts of world at each,
/// and the net's cords are about one pixel wide at one and two at another.
pub fn dump_dataset(
    out: &std::path::Path,
    sequences: usize,
    reference_spp: u32,
    seed: u64,
) -> anyhow::Result<()> {
    use kosm::denoise::dataset as ds;

    let scene = CourtScene::bundled()?;
    let stage = render::Scene::new(&scene.authored, scene.ball_r)?;
    let base_cam = authored_camera(&scene);
    let a = &scene.authored;
    let target = KVec3::new(
        a.parameter_or("cam_at_x_mm", 900.0),
        a.parameter_or("cam_at_y_mm", 300.0),
        a.parameter_or("cam_at_z_mm", 1900.0),
    );
    let ctx = vcad_kernel_gpu::GpuContext::init_blocking()
        .map_err(|e| anyhow::anyhow!("no GPU adapter: {e}"))?;
    let mut gpu = court_gpu::Stage::new(
        &stage,
        &ctx.device,
        &ctx.queue,
        a.parameter_or("max_depth", 6.0).max(1.0) as u32,
    )?;

    // The simulation, rolled once and snapshotted, so picking an instant is a
    // lookup rather than a re-run. The window of instants is the one the
    // filter has to be good at: balls in flight and the net moving.
    let mut court = Court::from_scene(&scene)?;
    let steps_per_frame = (1.0 / scene.fps / scene.dt).round().max(1.0) as usize;
    let mut snaps: Vec<Frame> = vec![super::snapshot(&court)];
    while court.time() < T_DATASET_END {
        court.step();
        snaps.push(super::snapshot(&court));
    }
    let first = ((T_DATASET_START / scene.dt) as usize).min(snaps.len() - 1);

    let mut rng = Rng(seed);
    let mut samples = Vec::with_capacity(sequences);
    let t0 = Instant::now();
    for si in 0..sequences {
        let size = DATASET_SIZES[si % DATASET_SIZES.len()];
        let tier = ds::TIERS[si % ds::TIERS.len()];
        let cam0 = orbit(
            &base_cam,
            target,
            rng.range(-std::f64::consts::PI, std::f64::consts::PI),
            rng.range(-0.25, 0.45),
            rng.range(0.7, 1.35),
        );
        // Two sequences in three swing the eye a few degrees partway through,
        // which is what puts reprojected, clamped and freshly disoccluded
        // pixels in the training set at all. A one-frame sequence has no
        // "partway", so it moves on its own last frame or not at all.
        let moves = si % 3 != 0 && tier > 1;
        let swing = rng.range(-4.0, 4.0);
        let move_at = if tier > 1 {
            2 + (rng.next_u64() as usize % (tier as usize - 1)) as u32
        } else {
            u32::MAX
        };

        // The instant the *reference* is of is drawn from the window, and the
        // sequence is then walked backwards from it. Not forwards: a tier-32
        // sequence is about a second of simulation, and starting every one of
        // them inside the window would run them all off the end and give the
        // longest histories one instant between them.
        let last = first + (rng.unit() * (snaps.len() - 1 - first) as f64) as usize;
        let start = last.saturating_sub((tier as usize) * steps_per_frame);

        gpu.reset_sequence()?;
        let mut cur = ds::FrameState::default();
        let mut guides = None;
        let mut cam_at_tier = cam0;
        // Frame k is the k-th pass, so a pixel that kept its history through
        // all of them is at count k — which is what `tier` names.
        for k in 1..=tier + 1 {
            let idx = (start + (k as usize - 1) * steps_per_frame).min(snaps.len() - 1);
            let cam = if moves && k >= move_at {
                orbit(&cam0, target, swing.to_radians(), 0.0, 1.0)
            } else {
                cam0
            };
            gpu.accumulate(&stage, &snaps[idx], idx as u64 + 1, &cam, size, 1)?;
            if k == tier {
                let (hist, g) = gpu.read_denoise_inputs()?;
                cur = frame_state(&hist);
                guides = Some(g);
                cam_at_tier = cam;
            }
        }
        let (hist, _) = gpu.read_denoise_inputs()?;
        let next = frame_state(&hist);
        let guides = guides.ok_or_else(|| anyhow::anyhow!("no frame was recorded"))?;
        let ref_idx = (start + (tier as usize - 1) * steps_per_frame).min(snaps.len() - 1);

        // The reference: the same instant, the same camera, from a cleared
        // history with the filter and the clamp off. The history's running
        // mean over that many passes *is* the converged render, so there is
        // nothing to read but the buffer we are already reading.
        let saved = *gpu.denoise_params_mut();
        {
            let d = gpu.denoise_params_mut();
            d.iters = 0;
            d.clamp_k = 0.0;
            d.history_cap = u32::MAX;
            d.count_cutoff = u32::MAX;
            // Converging, not filtering: every sample counts, on both lobes.
            d.lobes = false;
        }
        gpu.reset_sequence()?;
        // A still camera over a still frame, so every pass is an independent
        // sample of one picture and the mean converges.
        for _ in 0..reference_spp {
            gpu.accumulate(
                &stage,
                &snaps[ref_idx],
                ref_idx as u64 + 1,
                &cam_at_tier,
                size,
                1,
            )?;
        }
        let (refh, _) = gpu.read_denoise_inputs()?;
        *gpu.denoise_params_mut() = saved;

        samples.push(ds::Sample {
            width: size.0,
            height: size.1,
            tier,
            cur,
            next,
            normal: ds::to_f16(&guides.normal),
            depth: ds::to_f16(&guides.depth),
            albedo: ds::to_f16(&guides.albedo),
            id: ds::to_f16(&guides.id),
            reference: ds::to_f16(&refh.rgb),
        });
        // Written out every so often, not just at the end. An hour of GPU is
        // long enough that something will interrupt it, and a run that has to
        // start over from nothing because it was killed eight sequences from
        // the finish is an hour nobody gets back.
        if (si + 1) % 12 == 0 && si + 1 < sequences {
            let part = ds::Dataset {
                width: DATASET_SIZES.iter().map(|s| s.0).max().unwrap_or(0),
                height: DATASET_SIZES.iter().map(|s| s.1).max().unwrap_or(0),
                samples: samples.clone(),
            };
            part.save(out)?;
        }
        let per = t0.elapsed().as_secs_f64() / (si + 1) as f64;
        println!(
            "court  dataset {:3}/{sequences}  {}\u{d7}{}  history {tier:2}  t = {:.2} s  {}  \
             {:.0} s each, {:.0} s left",
            si + 1,
            size.0,
            size.1,
            (ref_idx as f64) * scene.dt,
            if moves { "camera moves" } else { "still camera" },
            per,
            per * (sequences - si - 1) as f64
        );
    }

    let data = ds::Dataset {
        width: DATASET_SIZES.iter().map(|s| s.0).max().unwrap_or(0),
        height: DATASET_SIZES.iter().map(|s| s.1).max().unwrap_or(0),
        samples,
    };
    data.save(out)?;
    println!(
        "court  {} sequences over histories {:?}, {}-pass references \u{2192} {} ({:.0} MB) in {:.0} s",
        data.samples.len(),
        ds::TIERS,
        reference_spp,
        out.display(),
        data.bytes() as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// The device history, packed for the dataset.
fn frame_state(h: &kosm_render::gpu::History) -> kosm::denoise::dataset::FrameState {
    use kosm::denoise::dataset as ds;
    ds::FrameState {
        mean: ds::to_f16(&h.rgb),
        count: ds::to_f16(&h.count.iter().map(|&c| c as f32).collect::<Vec<_>>()),
        variance: ds::to_f16(&h.variance),
    }
}

/// The frame sizes a dataset rotates through: the viewport's own, at the scale
/// divisors it actually settles on.
pub const DATASET_SIZES: [(u32, u32); 3] = [(384, 216), (512, 288), (640, 360)];

/// The simulated window a dataset samples from, seconds. Balls in flight and
/// the net still moving — the interesting part, and the part a still `--shot`
/// never sees.
const T_DATASET_START: f64 = 0.4;
const T_DATASET_END: f64 = 1.5;

/// Orbit `cam` about `target`: `d_az` radians of azimuth, `d_el` of
/// elevation, `scale` times the distance.
///
/// The court's authored camera is the shot the level wants, and every sample
/// is a perturbation of it rather than a camera drawn from nowhere. A denoiser
/// for *this gym* should be shown this gym's framings.
fn orbit(cam: &Camera, target: KVec3, d_az: f64, d_el: f64, scale: f64) -> Camera {
    let v = cam.eye - target;
    let r = v.norm() * scale;
    let az = v.y.atan2(v.x) + d_az;
    let el = (v.z / v.norm()).asin() + d_el;
    let el = el.clamp(-1.35, 1.35);
    Camera {
        eye: target
            + KVec3::new(
                r * el.cos() * az.cos(),
                r * el.cos() * az.sin(),
                r * el.sin(),
            ),
        ..*cam
    }
}

/// splitmix64, so a dataset is reproducible from its seed with no dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// The regions the viewport check is scored over, as fractions of the frame.
///
/// Fractions and not pixels, because the check runs at whatever `--width` it
/// is given and the authored camera frames the same shot at every size. They
/// are `(name, x0, y0, x1, y1)` in [0, 1], picked off the authored shot: the
/// hoop and the net are the two the v1 network failed on and they get boxes
/// of their own, tight enough that a win there is a win on the thing that was
/// broken rather than on the wall behind it.
pub const EVAL_REGIONS: [(&str, f32, f32, f32, f32); 5] = [
    ("back wall", 0.03, 0.20, 0.30, 0.45),
    ("bleachers", 0.03, 0.50, 0.68, 0.73),
    ("floor", 0.03, 0.82, 0.65, 0.99),
    ("hoop and backboard", 0.53, 0.02, 0.68, 0.26),
    ("net", 0.53, 0.25, 0.61, 0.37),
];

/// Root-mean-square difference between two RGBA frames over one box, in 8-bit
/// codes, alpha ignored.
fn rmse_codes(a: &[u8], b: &[u8], w: u32, h: u32, box_: (f32, f32, f32, f32)) -> f64 {
    let x0 = (box_.0 * w as f32) as usize;
    let y0 = (box_.1 * h as f32) as usize;
    let x1 = ((box_.2 * w as f32) as usize).min(w as usize);
    let y1 = ((box_.3 * h as f32) as usize).min(h as usize);
    let mut acc = 0.0f64;
    let mut n = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * w as usize + x) * 4;
            for c in 0..3 {
                let d = a[i + c] as f64 - b[i + c] as f64;
                acc += d * d;
                n += 1;
            }
        }
    }
    if n == 0 { 0.0 } else { (acc / n as f64).sqrt() }
}

/// The mean absolute frame-to-frame movement of a patch, in 8-bit codes.
///
/// The flicker number. Taken on a patch of back wall, which is static
/// geometry under static light: anything that moves there is the filter
/// changing its mind, not the picture changing.
fn stability_codes(frames: &[Vec<u8>], w: u32, h: u32) -> f64 {
    // The first region is the back wall, and it is the first region for this
    // reason: static geometry under static light.
    let bx = EVAL_REGIONS[0];
    let x0 = (bx.1 * w as f32) as usize;
    let y0 = (bx.2 * h as f32) as usize;
    let x1 = (bx.3 * w as f32) as usize;
    let y1 = (bx.4 * h as f32) as usize;
    let mut acc = 0.0f64;
    let mut n = 0usize;
    for pair in frames.windows(2) {
        for y in y0..y1.min(h as usize) {
            for x in x0..x1.min(w as usize) {
                let i = (y * w as usize + x) * 4;
                for c in 0..3 {
                    acc += (pair[1][i + c] as f64 - pair[0][i + c] as f64).abs();
                    n += 1;
                }
            }
        }
    }
    if n == 0 { 0.0 } else { acc / n as f64 }
}

/// The pass/fail: run the sequence twice, once under each filter, against a
/// converged render of the same instants, and say what each costs per region.
///
/// This is the check the v1 weights failed. The held-out table said the
/// network beat the à-trous filter by a fifth at every history length; the
/// window said the backboard came out 1.7x too bright and the net smudged.
/// Both were true, because the tiles the network was scored on and the frames
/// it was run on were different distributions. So the number that decides
/// whether `--denoise neural` is the default is measured *here*, on the
/// viewport, region by region — and the two regions that broke get boxes of
/// their own.
///
/// The reference for a frame is `ref_spp` passes of the bare path tracer from
/// a cleared history with the filter and the clamp off, tonemapped through the
/// same resolve, so every number is a difference between two 8-bit pictures of
/// one instant.
pub fn denoise_eval(
    dir: Option<&std::path::Path>,
    weights: Option<&std::path::Path>,
    t: f64,
    size: (u32, u32),
    n: u32,
    ref_spp: u32,
) -> anyhow::Result<()> {
    let scene = CourtScene::bundled()?;
    let stage = render::Scene::new(&scene.authored, scene.ball_r)?;
    let camera = authored_camera(&scene);
    let a = &scene.authored;
    let ctx = vcad_kernel_gpu::GpuContext::init_blocking()
        .map_err(|e| anyhow::anyhow!("no GPU adapter: {e}"))?;
    let mut gpu = court_gpu::Stage::new(
        &stage,
        &ctx.device,
        &ctx.queue,
        a.parameter_or("max_depth", 6.0).max(1.0) as u32,
    )?;
    // A candidate fit is looked at *before* it is bundled — that is the whole
    // point of the check — so `--weights` runs a file and the default runs
    // what ships.
    let weights = match weights {
        Some(p) => kosm_render::neural::Weights::load(p)
            .map_err(|e| anyhow::anyhow!("the weights at {}: {e}", p.display()))?,
        None => court_gpu::Stage::bundled_weights()?,
    };
    let t = if t < 0.0 {
        a.parameter_or("still_t", 0.95)
    } else {
        t
    };
    let n = n.max(1);
    let steps_per_frame = (1.0 / scene.fps / scene.dt).round().max(1.0) as usize;

    // The instants, rolled once so all three runs see the same simulation.
    let mut court = Court::from_scene(&scene)?;
    while court.time() < t {
        court.step();
    }
    let mut instants = Vec::with_capacity(n as usize);
    instants.push(super::snapshot(&court));
    for _ in 1..n {
        for _ in 0..steps_per_frame {
            court.step();
        }
        instants.push(super::snapshot(&court));
    }

    // The sequence, under one filter, driven exactly as `dump_frames` drives
    // it: eight passes to warm the history and then one pass a frame.
    let mut run = |gpu: &mut court_gpu::Stage| -> anyhow::Result<Vec<Vec<u8>>> {
        gpu.reset_sequence()?;
        for _ in 0..8 {
            gpu.accumulate(&stage, &instants[0], 0, &camera, size, 1)?;
        }
        let mut out = Vec::with_capacity(n as usize);
        for (k, f) in instants.iter().enumerate() {
            gpu.accumulate(&stage, f, k as u64 + 1, &camera, size, 1)?;
            out.push(gpu.read_target()?);
        }
        Ok(out)
    };

    gpu.set_neural(None)?;
    let t0 = Instant::now();
    let atrous = run(&mut gpu)?;
    let atrous_ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;

    gpu.set_neural(Some(&weights))?;
    let t1 = Instant::now();
    let neural = run(&mut gpu)?;
    let neural_ms = t1.elapsed().as_secs_f64() * 1e3 / n as f64;
    gpu.set_neural(None)?;

    // The references: one per instant, from a cleared history, filter off,
    // clamp off, cap lifted.
    let saved = *gpu.denoise_params_mut();
    {
        let d = gpu.denoise_params_mut();
        d.iters = 0;
        d.clamp_k = 0.0;
        d.history_cap = u32::MAX;
        d.count_cutoff = u32::MAX;
        // The bare path tracer: no firefly cap, no variance box, and the
        // presented frame is the mean itself.
        d.firefly_k = 0.0;
        d.variance_gamma = 0.0;
        d.temporal_filter = 0.0;
        // The split path caps a specular history by its roughness, and a
        // reference wants every sample it was given.
        d.lobes = false;
    }
    let mut refs = Vec::with_capacity(n as usize);
    for (k, f) in instants.iter().enumerate() {
        gpu.reset_sequence()?;
        for _ in 0..ref_spp {
            gpu.accumulate(&stage, f, k as u64 + 1, &camera, size, 1)?;
        }
        refs.push(gpu.read_target()?);
    }
    *gpu.denoise_params_mut() = saved;

    if let Some(dir) = dir {
        std::fs::create_dir_all(dir)?;
        let save = |img: &[u8], name: String| -> anyhow::Result<()> {
            image::RgbaImage::from_raw(size.0, size.1, img.to_vec())
                .ok_or_else(|| anyhow::anyhow!("the target texture is the wrong size"))?
                .save(dir.join(name))?;
            Ok(())
        };
        for k in 0..n as usize {
            save(&atrous[k], format!("atrous_{k:02}.png"))?;
            save(&neural[k], format!("neural_{k:02}.png"))?;
            save(&refs[k], format!("reference_{k:02}.png"))?;
        }
        println!("court  frames in {}", dir.display());
    }

    let mean = |frames: &[Vec<u8>], bx: (f32, f32, f32, f32)| -> f64 {
        frames
            .iter()
            .zip(&refs)
            .map(|(f, r)| rmse_codes(f, r, size.0, size.1, bx))
            .sum::<f64>()
            / n as f64
    };

    println!(
        "\ncourt  denoise: {}\u{d7}{}, {n} frames from t = {t:.2} s, \
         against a {ref_spp}-pass reference. RMSE in 8-bit codes:\n",
        size.0, size.1
    );
    println!("  {:<20} {:>9} {:>9}", "region", "à-trous", "neural");
    let whole = (0.0, 0.0, 1.0, 1.0);
    let (wa, wn) = (mean(&atrous, whole), mean(&neural, whole));
    println!("  {:<20} {wa:>9.2} {wn:>9.2}", "whole frame");
    let mut worse = Vec::new();
    for (name, x0, y0, x1, y1) in EVAL_REGIONS {
        let bx = (x0, y0, x1, y1);
        let (ra, rn) = (mean(&atrous, bx), mean(&neural, bx));
        println!("  {name:<20} {ra:>9.2} {rn:>9.2}");
        if rn > ra {
            worse.push((name, ra, rn));
        }
    }
    if wn > wa {
        worse.push(("whole frame", wa, wn));
    }

    println!(
        "\n  frame-to-frame movement on a static patch: \
         à-trous {:.2}, neural {:.2} codes",
        stability_codes(&atrous, size.0, size.1),
        stability_codes(&neural, size.0, size.1),
    );
    println!(
        "  a pass costs: à-trous {atrous_ms:.1} ms, neural {neural_ms:.1} ms"
    );
    if worse.is_empty() {
        println!("\n  the neural filter wins everywhere.");
    } else {
        println!("\n  the neural filter loses on:");
        for (name, ra, rn) in worse {
            println!("    {name}: {rn:.2} against {ra:.2}");
        }
    }
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

/// How converged a picture has to be before the tuner stops stepping its
/// size on a tier that cannot carry a history across the step.
///
/// A size step is free when there is nothing to lose and ruinous when there
/// is. At five samples a pixel, growing the picture costs a frame; at six
/// hundred it costs a minute of accumulation, and the log showed exactly
/// that — the mean sample count falling from 599 to 26 on a 426→365 step.
/// The CPU tier resamples its history and pays neither price. The GPU tier's
/// history lives in device buffers vcad reallocates on a resize, and reading
/// them back to resample them is the one thing that tier is built not to do,
/// so it takes the other design: **the size is free to move while it is cheap
/// to move it, and frozen once it is dear.** The tuner's other knobs — the
/// sample count and the scissor — go on working either way, and when the
/// world starts moving again the mask restarts pixels, the mean falls back
/// under this line, and the size is the tuner's again.
const RESIZE_UNTIL: f32 = 48.0;

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
/// picture at all — the CPU tier rebuilds the net's BVH and forks a rayon
/// pool; the GPU tier used to re-upload the whole court and drag the frame
/// back across the bus whatever its size. Milliseconds per megapixel per
/// sample folded that floor into the slope, so the model over-charged a big
/// picture and under-charged a small one, and the tuner oscillated: grow on a
/// prediction, overrun, shrink, go quiet, grow again.
///
/// So: `ms = fixed + per * work`. Fitting a line needs two points, and the
/// passes supply them for free because a masked pass does less work than a
/// full one — the two buckets below are exponential moving averages of the
/// cheap end and the dear end of whatever work has been asked for, and the
/// line through them is the model. Until they are far enough apart to be two
/// points it degenerates to the old one-term fit through the origin, which is
/// what a single measurement can honestly say.
///
/// ## what `work` counts
///
/// **Megapixel-samples touched**, which is the traced patch *plus the whole
/// frame*. It used to be the traced patch alone, and that was right when the
/// tracer was the pass. It is not any more. With guides on both tiers the
/// history denoises every pass, and [`History::resolve`] costs the *frame*
/// whatever the mask let the tracer skip: measured on the GPU tier at
/// 512x288, a pass is 25 ms tracing, 5 ms merging and 420 ms resolving, and a
/// pass that traced 15% of the screen costs the same as a full one.
///
/// Charging only the patch made every one of those masked passes look like a
/// dear pass at a tenth of the work, the fit put all of it in `fixed`, and
/// `size_matters` below then said — correctly, for the model it was given —
/// that shrinking the picture would not help. The window sat at 512x288 and
/// half a second a pass with a 30 ms target. Counting the frame the history
/// walks puts that cost back on the slope where it belongs.
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
            // Two points only count as two when they are genuinely apart.
            // The buckets drift towards whatever work is being fed them, and
            // two that have drifted together give a slope of nothing over
            // nothing: at 183x102 that fitted 470 ms a megapixel-sample
            // against a real 50, and the window refused to grow past a
            // postage stamp. Below [`SPREAD`] the honest answer is the
            // one-term fit.
            (Some(lo), Some(hi)) if hi.0 > lo.0 * SPREAD => {
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
    rx: Receiver<Timed>,
    shots: Receiver<Shot>,
    jobs: Sender<Job>,
    frames: Vec<Timed>,
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
    /// How long the last fair pass took, in milliseconds: the climb rule reads
    /// the clock, not the model.
    last_ms: f64,
    /// The same clock, kept twice, because a masked pass and a full one are
    /// not the same pass. Now that a masked pass is a handful of boxes rather
    /// than the frame, it can be several times cheaper — and a climb rule
    /// that reads that cheapness buys a picture the next full pass cannot
    /// afford. So the *size* is bought with `full_ms`, which is what a grown
    /// picture will actually have to pay, and the *sample count* with
    /// `masked_ms`, which is what most passes cost while the world is moving.
    /// Zero means not yet measured, and the raw last pass stands in.
    full_ms: f64,
    masked_ms: f64,
    /// Whether the last pass came from a tier that would lose its accumulated
    /// picture if the size stepped, and how converged that picture is. See
    /// [`RESIZE_UNTIL`].
    resize_costs_history: bool,
    mean_spp: f32,
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

    // ---- the pace ----------------------------------------------------------
    /// What the window tells the simulation to run ahead by, in microseconds.
    /// Written here, read on the simulation thread.
    lookahead: Lookahead,
    /// Milliseconds between a frame's due moment and the blit of its picture,
    /// as a moving average. This is the number the whole of the lookahead is
    /// for: it is the latency a viewer sees, and the head start that cancels
    /// it is exactly its own size.
    latency_ms: f64,
    /// The worst one seen since the last pacing line.
    worst_ms: f64,
    /// Frames the simulation produced that no pass was ever taken of, and
    /// moments the window wanted a frame for that the simulation had not
    /// reached yet.
    dropped: u64,
    late: u64,
    /// One frame of the recording, in wall-clock time, learned from the first
    /// two frames rather than re-read off the level.
    gap: Duration,
    said_pace: Instant,
    /// `KOSM_NO_LOOKAHEAD=1`: present the newest frame the simulation has,
    /// the way the window used to. Kept so the two can be measured against
    /// each other in one session rather than argued about.
    no_lookahead: bool,
}

impl App {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rx: Receiver<Timed>,
        shots: Receiver<Shot>,
        jobs: Sender<Job>,
        camera: Camera,
        spp: u32,
        pending: (Receiver<Job>, Sender<Shot>),
        cpu_only: bool,
        lookahead: Lookahead,
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
            last_ms: 0.0,
            full_ms: 0.0,
            masked_ms: 0.0,
            resize_costs_history: false,
            mean_spp: 0.0,
            generation: 0,
            asked: None,
            pending: Some(pending),
            cpu_only,
            lookahead,
            latency_ms: 0.0,
            worst_ms: 0.0,
            dropped: 0,
            late: 0,
            gap: Duration::from_millis(16),
            said_pace: Instant::now(),
            no_lookahead: std::env::var("KOSM_NO_LOOKAHEAD").is_ok(),
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
        self.camera.eye = t + KVec3::new(
            dist * el.cos() * az.cos(),
            dist * el.cos() * az.sin(),
            dist * el.sin(),
        );
    }

    /// The render size: the window over the divisor, never degenerate.
    fn size(&self) -> (u32, u32) {
        (
            (self.window.0 / self.scale).max(32),
            (self.window.1 / self.scale).max(18),
        )
    }

    /// The work a *full* pass at this divisor is, in megapixel-samples. Full,
    /// because that is the pass whose cost decides how big the picture may be:
    /// a masked pass is cheaper by definition and never the thing that has to
    /// fit. The frame counts twice — once traced, once resolved — which is
    /// what [`Cost`] measures against.
    fn work(&self, scale: u32) -> f64 {
        let (w, h) = (
            (self.window.0 / scale).max(32),
            (self.window.1 / scale).max(18),
        );
        2.0 * w as f64 * h as f64 / 1e6 * self.samples.max(1) as f64
    }

    /// The part of a pass at this divisor that the *size* is paying for.
    ///
    /// Only this part answers to resolution. The fixed part is paid whether
    /// the picture is 512 pixels across or 91 — the net's BVH, a rayon pool,
    /// the dispatch and the readback — so charging the size for it is what
    /// made the old tuner shrink the picture to nothing chasing a budget no
    /// size could meet. On the GPU tier that part is now a millisecond or
    /// two: the court is resident and only the camera moves.
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
        (1..MAX_SCALE)
            .find(|s| self.pixel_ms(*s) <= self.budget())
            .unwrap_or(MAX_SCALE)
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
    /// A pass that traced only a patch is charged for the patch *and* for the
    /// frame the history then walked — see [`Cost`]. Masked and full passes
    /// are still the two work levels the line is drawn through, with no probe
    /// pass and no calibration phase; they are just no longer ten times apart
    /// when the tracer is not what the pass is made of.
    fn tune(&mut self, shot: &Shot) -> bool {
        let frame_px = (shot.size.0 as u64) * (shot.size.1 as u64);
        let work = (shot.traced_px + frame_px) as f64 / 1e6 * shot.spp.max(1) as f64;
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
        // Only a full pass says anything about what a *size* costs; a masked
        // one traced a box whose size is the world's business, not the
        // tuner's, and folding its cheapness into the model that picks the
        // resolution is how the window talks itself into a picture it cannot
        // afford the moment the boxes stop paying.
        if shot.traced_px >= (shot.size.0 as u64) * (shot.size.1 as u64) {
            self.cost.observe(work, ms);
            let slot = &mut self.seen[(self.scale as usize).min(MAX_SCALE as usize + 1)];
            *slot = if *slot > 0.0 {
                0.7 * *slot + 0.3 * ms
            } else {
                ms
            };
        }
        true
    }

    /// What a full pass costs, for the knob that decides how big the picture
    /// is. Falls back to the last pass until a full one has been timed.
    fn size_ms(&self) -> f64 {
        if self.full_ms > 0.0 {
            self.full_ms
        } else {
            self.last_ms
        }
    }

    /// What a pass costs as passes actually come, for the knob that decides
    /// how many samples one carries.
    fn samples_ms(&self) -> f64 {
        if self.masked_ms > 0.0 {
            self.masked_ms
        } else {
            self.last_ms
        }
    }

    /// Whether a size step would now cost more than it buys.
    ///
    /// Only on a tier whose history cannot survive one, and only once that
    /// history is worth keeping — see [`RESIZE_UNTIL`]. A window's own resize
    /// is not covered by this: there the old picture is of a different window
    /// and there is nothing to protect.
    fn size_is_dear(&self) -> bool {
        self.resize_costs_history && self.mean_spp >= RESIZE_UNTIL
    }

    /// Would a step coarser actually be cheaper? Unknown counts as yes — the
    /// only way to find out is to try it once.
    fn shrinking_helps(&self) -> bool {
        let (here, coarser) = (
            self.seen[self.scale as usize],
            self.seen[(self.scale + 1) as usize],
        );
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

    /// Which frame the window is *for* right now.
    ///
    /// Not the newest one the simulation has — that one is already in the
    /// past by the time a pass of it exists. The window aims at the moment
    /// the picture will actually be on the glass, which is now plus the
    /// latency it has measured, and takes the simulation's own frame for that
    /// moment. The simulation has been told to run that far ahead, so the
    /// frame is normally there; when it is not, the window falls back to the
    /// newest it has and counts the starve.
    ///
    /// Nothing here interpolates. Every frame presented is a frame the solver
    /// took `dt` steps to reach — the world is computed, not predicted.
    fn pace(&mut self, n: usize) {
        let want = if self.no_lookahead {
            n - 1
        } else {
            let deadline = Instant::now() + self.head_start();
            match self.frames.iter().rposition(|f| f.due <= deadline) {
                Some(k) => {
                    // Short by more than a frame: the simulation has not
                    // reached the moment the picture is for, and the window
                    // shows the newest real frame instead. Being short by
                    // *less* than a frame is the steady state and not news —
                    // the frame it would have wanted is the one it is holding.
                    if k == n - 1 && self.frames[k].due + self.gap < deadline {
                        self.late += 1;
                    }
                    k
                }
                None => 0,
            }
        };
        // Frames the simulation produced that no pass was ever taken of. A
        // dropped frame is not a stutter — the next frame presented is still
        // the right frame for its moment — but it is work thrown away and the
        // log says how much.
        if want > self.cursor + 1 {
            self.dropped += (want - self.cursor - 1) as u64;
        }
        self.cursor = want;
    }

    /// The head start the simulation is asked for: the latency actually
    /// measured, capped, and nothing at all while paused — a paused window
    /// renders the frame under the cursor and has no future to reach for.
    fn head_start(&self) -> Duration {
        if !self.live || self.no_lookahead {
            return Duration::ZERO;
        }
        Duration::from_micros((self.latency_ms.max(0.0) * 1e3) as u64).min(MAX_LOOKAHEAD)
    }

    fn ask(&mut self) {
        let Some(timed) = self.frames.get(self.cursor) else {
            return;
        };
        self.generation += 1;
        let job = Job {
            generation: self.generation,
            frame_id: self.frame_id(),
            frame: timed.frame.clone(),
            camera: self.camera,
            size: self.size(),
            spp: self.samples,
            due: timed.due,
        };
        let _ = self.jobs.send(job);
        self.asked = Some(self.ask_key());
    }
}

impl viewport::Scene for App {
    fn init(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some((jobs, shots)) = self.pending.take() else {
            return;
        };
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
            if let Some(prev) = self.frames.last() {
                if self.frames.len() == 1 {
                    self.gap = frame.due.saturating_duration_since(prev.due);
                }
            }
            self.frames.push(frame);
        }
        let n = self.frames.len();
        if self.live && n > 0 {
            self.pace(n);
        }
        // Every shot is shown, even one the window has already moved past: a
        // pass takes longer than a redraw, so refusing stale ones would leave
        // the window blank for the whole of playback.
        let mut newest = None;
        while let Ok(shot) = self.shots.try_recv() {
            let fair = self.tune(&shot);
            // What this picture cost the viewer: from the moment its frame is
            // *for* to the moment it goes to the blit. `image` is called from
            // the redraw, so now is that moment.
            let ms = Instant::now().saturating_duration_since(shot.due).as_secs_f64() * 1e3;
            self.latency_ms = if self.latency_ms > 0.0 {
                0.8 * self.latency_ms + 0.2 * ms
            } else {
                ms
            };
            self.worst_ms = self.worst_ms.max(ms);
            newest = Some(shot.image);
            if !fair {
                continue;
            }
            // A pass that blew the budget twice running costs a step of
            // resolution, or the samples that bought it. A pass that came back
            // with little of the screen repainted is one more piece of
            // evidence that the picture is worth growing.
            self.cheap = if (shot.ms as f64) < 0.5 * TARGET_MS {
                self.cheap + 1
            } else {
                0
            };
            self.last_ms = shot.ms as f64;
            let ema = |e: f64, ms: f64| if e > 0.0 { 0.7 * e + 0.3 * ms } else { ms };
            if shot.traced_px >= (shot.size.0 as u64) * (shot.size.1 as u64) {
                self.full_ms = ema(self.full_ms, shot.ms as f64);
            } else {
                self.masked_ms = ema(self.masked_ms, shot.ms as f64);
            }
            self.resize_costs_history = shot.resize_costs_history;
            self.mean_spp = shot.mean_spp;
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
                    } else if self.scale < MAX_SCALE && !self.size_is_dear() {
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
        } else if self.quiet >= CLIMB_AFTER {
            // The climb reads the clock, not the model. The model seeds the
            // first size after a resize and explains an overrun; whether the
            // next size up is affordable is answered by the passes that just
            // happened. A quiet run of passes with room under the budget buys
            // pixels before it buys samples — the same sample count spread
            // over a bigger picture is what the eye wants first — and a size
            // once condemned gets another hearing after a longer run of cheap
            // passes, because the pass that condemned it may not have been a
            // render at all.
            let room = self.size_ms() < 0.7 * TARGET_MS && !self.size_is_dear();
            if room && self.scale > self.floor {
                if self.samples > 1 {
                    self.samples = 1;
                }
                self.scale -= 1;
            } else if room
                && self.scale == self.floor
                && self.floor > 1
                && self.cheap >= CLIMB_AFTER
            {
                self.floor -= 1;
                self.scale -= 1;
                self.samples = 1;
                self.cheap = 0;
            } else if self.samples_ms() * 2.0 < TARGET_MS && self.samples < self.spp {
                self.samples = (self.samples * 2).min(self.spp);
            }
            self.quiet = 0;
        }
        // The one number the simulation thread reads. Paused, it is zero.
        self.lookahead
            .store(self.head_start().as_micros() as u64, Ordering::Relaxed);
        if self.said_pace.elapsed().as_secs() >= 2 {
            self.said_pace = Instant::now();
            let lead = self.frames.last().map_or(0.0, |f| {
                let now = Instant::now();
                f.due.saturating_duration_since(now).as_secs_f64() * 1e3
                    - now.saturating_duration_since(f.due).as_secs_f64() * 1e3
            });
            eprintln!(
                "court  pace: {:.0} ms presented latency (worst {:.0}), {:.0} ms head start, \
                 the sim {:.0} ms ahead; {} dropped, {} late{}",
                self.latency_ms,
                self.worst_ms,
                self.head_start().as_secs_f64() * 1e3,
                lead,
                self.dropped,
                self.late,
                if self.no_lookahead { " (lookahead off)" } else { "" },
            );
            self.dropped = 0;
            self.late = 0;
            self.worst_ms = 0.0;
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
    let lookahead: Lookahead = Arc::new(AtomicU64::new(0));
    let sim_lookahead = lookahead.clone();
    std::thread::spawn(move || simulate(tx, frames, sim_lookahead));

    // The level's camera, without waiting for the renderer's stage: cheap to
    // read, and the window wants it before the first frame arrives.
    let camera = CourtScene::bundled()
        .map(|s| authored_camera(&s))
        .unwrap_or(Camera {
            eye: KVec3::new(-800.0, -5600.0, 1900.0),
            target: KVec3::new(900.0, 300.0, 1900.0),
            fov_deg: 40.0,
            exposure: 1.0,
        });

    viewport::run(
        "Kosm view — the court",
        (1280, 720),
        App::new(
            rx,
            shot_rx,
            job_tx,
            camera,
            spp,
            (job_rx, shot_tx),
            cpu_only,
            lookahead,
        ),
    )
}
