//! The cove in the window: `kosm run rune --view`.
//!
//! The court's tier, pointed at a beach and given a player. Four threads
//! instead of the court's three, and the extra one is the puzzle:
//!
//! - [`simulate`] steps [`Cove`] at its own `dt` in wall-clock time and hands
//!   over [`Timed`] snapshots, ahead by the latency the window has measured —
//!   the court's pacing, verbatim, because the court's pacing is right.
//! - [`rune_worker`] reads the newest snapshot, scores the rune on it, and
//!   sets the gate. It is its own thread because the score is a photon trace:
//!   the simulation must never wait for one, and neither must the picture.
//! - [`render_worker`] traces one raw sample a pass and accumulates it through
//!   [`crate::history`] with reprojection and the geometric mask.
//! - the viewport blits whatever came back last.
//!
//! ## the tier is the CPU integrator, and why
//!
//! The court runs on `court_gpu::Stage`, and the cove cannot. That tier is
//! built on `GpuScene::from_brep`, which packs **analytic trimmed B-rep
//! surfaces and nothing else** — there is no triangle path in
//! `vcad_kernel_raytrace::gpu` at all. Three things in the cove are not
//! B-reps:
//!
//! - **the being.** `cove::render` tessellates the capsule on purpose (the
//!   boolean that would make one is two tangential joins, the case the kernel
//!   is worst at), and it is the level's only dielectric — a picture without
//!   it is not the game.
//! - **the sea.** A `kosm_render::HeightField` is not a solid. This one could
//!   be swapped for a flat B-rep slab at `sea_z` on that tier, since the swell
//!   is decoration.
//! - **the ground.** `cove::render::Scene` keeps `Arc<Bvh<CoveGeom>>` per
//!   part and does not retain the `vcad_kernel::Solid` a pack would need, so
//!   even the beach could not be handed to `Stage::new` as it stands.
//!
//! So the GPU route is: retain the solids in the cove's render scene, author
//! the capsule as a B-rep (or teach vcad's shader a triangle surface), and
//! substitute the sea. That is days, across two crates, and the third item is
//! in a file this change does not own. A walkable cove at low quality beats a
//! beautiful one that does not exist, so this tier is
//! `pathtrace::render_with_caustics` on the render thread at [`WIDTH`] by
//! [`HEIGHT`], one sample a pass, with the history doing what it does for the
//! court's CPU fallback. Everything else — the pacing, the mask, the
//! reprojection, the resolve — is the court's, unchanged.
//!
//! ## the camera moves every frame
//!
//! Third person, from `cove::render::camera`: three metres behind the being
//! along its facing and one and a half above, following it. So a walking
//! player moves the camera every frame and the plan is `full` every frame;
//! what keeps the picture from being a blizzard is the reprojection, which
//! carries each pixel through the moved eye. Standing still, the camera stops,
//! the mask empties, and the picture converges in a couple of seconds.
//!
//! ## the rune is not a trigger
//!
//! [`Gate`] holds the door shut until `cove::rune::score` has been over the
//! level's `open_frac` for one continuous second of *simulated* time, and then
//! `Cove::set_gate(true)` lets the hinge's spring drive it. Nothing about that
//! is a volume or a flag: move the sun in the document and the pose that opens
//! the door moves with it.
//!
//! No text is drawn. The score goes to stderr once a second while it is worth
//! seeing, and that is the whole HUD.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kosm_render::caustics::CausticMap;
use kosm_render::math::{Point3, Vec3 as RVec3};
use kosm_render::pathtrace::{self, Camera, Film, PathTraceOptions};
use super::being::{Cove, Input, Snapshot};
use super::render::{self as cove_render, PER_M, Placement};
use super::{CoveScene, bake, hint, rune};
use phyz_math::{Mat3, Vec3};

use kosm_view::budget::Budget;
use kosm_view::history::{History, Plan, Pose, View};
use kosm_view::viewport;

// ---- the knobs the window owns ---------------------------------------------

/// The nominal render size: what a still picture converges at, and the base
/// [`Budget`] measures its ladder against. `--rune-width N` moves it.
///
/// It used to be the whole story — fixed, not tuned — on the reasoning that
/// the court's thirty-millisecond tuner would walk this tier down to its floor
/// and leave it there, and that a fixed size the history is allowed to
/// converge on is the better picture. Half of that was right. A CPU pass of
/// this cove is sixty-five to a hundred and fifty milliseconds at 480×270,
/// so a *walking* player was getting seven to fifteen passes a second of a
/// picture whose every pixel was two samples old — resolution nobody could
/// see, bought with a frame rate everybody could feel.
///
/// What the old reasoning was missing is that the two states want opposite
/// things. [`Budget`] is the policy: while walking, the largest size whose
/// predicted pass fits [`TARGET_MS`], with the history carrying the pixels
/// through the moved eye so a quarter-size pass reads as motion blur rather
/// than as blocks; while standing, this size and never less — and [`PRETTY`]
/// above it when the pass fits [`CEILING_MS`], because a still frame's only
/// cost is patience.
pub const WIDTH: u32 = 480;
pub const HEIGHT: u32 = WIDTH * 9 / 16;

/// What a walking pass may cost. Twenty-five passes a second is not a frame
/// rate a path tracer will reach here, but it is the number the ladder is
/// measured against, and against a hundred-and-thirty-millisecond base pass it
/// buys the floor — a quarter size, a sixteenth of the rays.
const TARGET_MS: f64 = 40.0;

/// What a *still* pass may cost before the pretty rung is given back. A
/// quarter of a second a pass is four passes a second into a picture nobody is
/// moving, which converges perfectly well.
const CEILING_MS: f64 = 250.0;

/// The size a still picture is allowed to reach, as a multiple of [`WIDTH`].
/// Twice, which is four times the rays: 960×540 from the default 480×270.
const PRETTY: u32 = 2;

/// Radians of yaw per unit of raw mouse motion, and of tilt.
///
/// The tilt is the slower of the two on purpose: yaw is how the player looks
/// around and tilt is one of the puzzle's two knobs, clamped by the simulation
/// to `being::TILT_MAX` either side. A knob with ten degrees of travel wants a
/// finer hand than a heading with three hundred and sixty.
const YAW_PER_UNIT: f64 = 0.0025;
const TILT_PER_UNIT: f64 = 0.0010;

/// How far the being has to move, or lean, before the caustic is retraced.
///
/// A photon map is a few tens of milliseconds and a pass is a few hundred, so
/// this is not a budget — it is what stops a being standing still from
/// retracing a map that would come back the same and, through the mask, throw
/// the door's converged pixels away for it.
const CAUSTIC_MOVED_M: f64 = 0.01;
const CAUSTIC_LEANED_RAD: f64 = 0.5 * std::f64::consts::PI / 180.0;

/// How long the score has to hold above `open_frac`, in simulated seconds.
const HOLD: f64 = 1.0;

/// The most the simulation runs ahead of the wall clock, and the most it will
/// chase before the clock is re-based. Both are the court's, scaled for a pass
/// that costs ten times what the court's does: a tier at three frames a second
/// has a latency of a third of a second, and a head start capped at eighty
/// milliseconds would never cancel it.
const MAX_LOOKAHEAD: Duration = Duration::from_millis(400);
const MAX_SLIP: Duration = Duration::from_millis(300);

/// How far ahead of the wall clock the simulation may run, in microseconds.
/// Written by the window, read by the simulation.
type Lookahead = Arc<AtomicU64>;

// ---- what the player is holding ---------------------------------------------

/// The controls, as the simulation reads them.
///
/// `forward` and `strafe` are held directions in `-1..=1`; `yaw` and `tilt`
/// are *accumulated* radians the simulation drains and spreads over the steps
/// of the frame it is about to take. A held key is a held force and a mouse
/// move is a quantity of turn, which is the difference between the two halves
/// of this struct and the reason they are drained differently.
#[derive(Clone, Copy, Default)]
struct Controls {
    forward: f64,
    strafe: f64,
    yaw: f64,
    tilt: f64,
}

type Held = Arc<Mutex<Controls>>;

/// The newest snapshot, for the thread that scores it. A mutex and not a
/// channel: the rune wants the *latest* pose and has no use for the ones it
/// was too slow to see.
type Latest = Arc<Mutex<Option<Snapshot>>>;

// ---- the gate ----------------------------------------------------------------

/// The rune's answer, held for a second.
///
/// Above `open_frac` for [`HOLD`] continuous seconds of simulated time and the
/// gate opens; a moment below it and the clock starts over. Once open it stays
/// open — a door that has swung is a door that has swung.
#[derive(Clone, Copy, Debug)]
pub struct Gate {
    open_frac: f64,
    hold: f64,
    since: Option<f64>,
    open: bool,
}

impl Gate {
    pub fn new(open_frac: f64) -> Self {
        Self { open_frac, hold: HOLD, since: None, open: false }
    }

    /// Read the score at simulated time `t`, and say whether the door may
    /// swing.
    pub fn read(&mut self, t: f64, frac: f64) -> bool {
        if frac >= self.open_frac {
            let since = *self.since.get_or_insert(t);
            if t - since >= self.hold {
                self.open = true;
            }
        } else {
            self.since = None;
        }
        self.open
    }

    pub fn is_open(&self) -> bool {
        self.open
    }
}

// ---- the glint ---------------------------------------------------------------

/// The other half of the design's hint: a spark on the sand when the player
/// has stopped getting anywhere.
///
/// "After thirty seconds without progress a glint appears on the sand a short
/// step along the gradient." Progress is the *best score so far* rising — not
/// the instantaneous one, which wanders with every photon budget and every
/// footstep and would reset the clock forever. A rise resets the clock and puts
/// the glint away; `after` seconds without one brings it back.
///
/// Two rules beyond that, and both are the design's. The glint never appears
/// while the score is already over `open_frac`: a player holding the rune is
/// not stuck, and a hint pointing somewhere else while the door is unlatching
/// would be a lie. And the clock is *simulated* time, the same clock
/// [`Gate`] runs on, so a slow renderer never makes the level more helpful.
///
/// The state machine and the gradient are deliberately separate: this says
/// *whether* there is a glint, [`glint_at`] says where it goes, and only the
/// first of them is cheap enough to run at the rune thread's rate.
#[derive(Clone, Copy, Debug)]
pub struct Glint {
    after: f64,
    open_frac: f64,
    best: f64,
    since: Option<f64>,
    on: bool,
}

/// What counts as the score having risen. A photon count of fifty thousand has
/// a percent or so of noise on it, so a rise has to be bigger than that or the
/// clock never runs.
const ROSE_BY: f64 = 0.005;

impl Glint {
    pub fn new(after: f64, open_frac: f64) -> Self {
        Self { after, open_frac, best: 0.0, since: None, on: false }
    }

    /// Read the score at simulated time `t`, and say whether the sand should
    /// spark.
    pub fn read(&mut self, t: f64, frac: f64) -> bool {
        let start = self.since.get_or_insert(t);
        if frac > self.best + ROSE_BY {
            self.best = frac;
            *start = t;
            self.on = false;
        } else if t - *start >= self.after {
            self.on = true;
        }
        // holding the rune is not being stuck
        if frac >= self.open_frac {
            self.on = false;
        }
        self.on
    }

    pub fn is_on(&self) -> bool {
        self.on
    }
}

/// How long the level waits before it sparks. `KOSM_GLINT_AFTER` overrides the
/// document, which is how a headless still of the glint is taken without
/// editing the level.
fn glint_after(scene: &CoveScene) -> f64 {
    std::env::var("KOSM_GLINT_AFTER")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(scene.glint_after)
        .max(0.0)
}

/// Where the glint goes: `glint_step` along the horizontal part of the hint's
/// gradient, sitting on the sand.
///
/// The gradient is [`hint::guided_gradient`] and not [`hint::gradient`], and
/// that is the whole point. The bare score is flat over most of the beach — a
/// caustic thrown ten metres wide of the door deposits exactly nothing in the
/// keyhole, and the derivative of nothing points nowhere — so a glint that
/// followed it would sit still until the player had already solved the level.
/// The guided objective is the one the solvability sweep climbed from every
/// spawn in the cove, which is precisely the claim "following this gets you
/// there"; the glint shows the player the same direction the level was proved
/// solvable along.
///
/// It costs six lattice traces, so the caller is expected to be a thread that
/// is not the renderer and to ask rarely. Returns the centre in world metres
/// and the horizontal direction it stepped, for the line on stderr.
fn glint_at(scene: &CoveScene, pose: &rune::Pose) -> Option<(Vec3, [f64; 2])> {
    let g = hint::guided_gradient(scene, pose, hint::SWEEP_RAYS)?;
    let n = g[0].hypot(g[1]);
    if !n.is_finite() || n < 1e-12 {
        return None;
    }
    let (ux, uy) = (g[0] / n, g[1] / n);
    let (x, y) = (pose.x + ux * scene.glint_step, pose.y + uy * scene.glint_step);
    Some((Vec3::new(x, y, scene.sand_z_at(x, y) + scene.glint_r), [ux, uy]))
}

/// How often the glint's direction is recomputed while it is showing. It is
/// six lattice traces; the player is stuck, and a hint that twitches is worse
/// than one that waits.
const GLINT_EVERY: Duration = Duration::from_secs(2);

// ---- what the rune thread tells the picture ----------------------------------

/// The hint, as the renderer needs it: the newest score (which is the rim's
/// radiance) and where the glint sits, if there is one.
///
/// A mutex and not a channel, for the same reason [`Latest`] is one: the
/// picture wants the newest answer and has no use for the ones it was too slow
/// to draw.
#[derive(Clone, Copy, Default)]
struct Lit {
    score: f64,
    glint: Option<Vec3>,
}

type Glow = Arc<Mutex<Lit>>;

// ---- the being, as the picture and the rune each want it ----------------------

/// The live being read as the rune's two-knob pose.
///
/// The score's `Pose` has a lean about `+x` and no yaw, because a capsule is a
/// surface of revolution about its own axis and turning it about that axis
/// changes nothing a photon can see. The simulated being's axis is free, so
/// what is read off it here is the lean of that axis in the plane the score's
/// tilt sweeps: `tilt = atan2(−axis.y, axis.z)`, which is exact whenever the
/// being is facing the door and is the nearest two-knob pose when it is not.
fn rune_pose(snap: &Snapshot) -> rune::Pose {
    let (centre, world_to_body) = snap.being;
    let axis = world_to_body.transpose().mul_vec(Vec3::z());
    rune::Pose { x: centre.x, y: centre.y, tilt: (-axis.y).atan2(axis.z) }
}

/// The live being as the renderer's placement.
///
/// `Snapshot::being` carries phyz's world → body rotation, whose `y` column is
/// arbitrary about the capsule's own axis — the free joint has no opinion
/// about a symmetric body's heading, which is why the heading is a *number* in
/// the simulation. `Placement` wants `(right, facing, up)` as columns, so the
/// two are put back together here: `up` is the body's real axis, `facing` is
/// the commanded heading projected onto the plane across it, and `right` is
/// their cross.
fn placement_of(snap: &Snapshot) -> Placement {
    let (centre, world_to_body) = snap.being;
    let up = world_to_body.transpose().mul_vec(Vec3::z()).normalize();
    let (s, c) = snap.facing.sin_cos();
    let heading = Vec3::new(c, s, 0.0);
    let flat = heading - up * heading.dot(&up);
    // A being lying exactly along its own heading has no facing left in the
    // plane; any direction across the axis will do, and +x is one.
    let facing = if flat.norm() > 1e-6 {
        flat.normalize()
    } else {
        let alt = Vec3::x() - up * up.x;
        if alt.norm() > 1e-6 { alt.normalize() } else { Vec3::y() }
    };
    let right = facing.cross(&up);
    Placement {
        being: (centre, columns(right, facing, up)),
        door_angle: snap.door_angle,
        score: 0.0,
        glint: None,
    }
}

/// A rotation from its three axes, as columns.
fn columns(x: Vec3, y: Vec3, z: Vec3) -> Mat3 {
    Mat3::new(x.x, y.x, z.x, x.y, y.y, z.y, x.z, y.z, z.z)
}

/// The camera, rounded to the millimetre it is written in.
///
/// The camera follows the being, and the being is held up by a contact solver:
/// standing perfectly still it wanders by micrometres, which is nothing to look
/// at and everything to [`History`]. `plan` compares views for equality, and a
/// view that differs by a float is a *moved* camera — so an unquantised camera
/// would put every pass of a standing player through the reprojection, ask for
/// a full frame every time, and shed a pixel of history at every silhouette
/// while it did.
///
/// A millimetre of eye at this size and field of view is a hundredth of a
/// pixel, and the target is rounded ten metres out along the look, which is a
/// tenth of a milliradian. Both are far under what the picture can show, and
/// what they buy is that a still player is *still*: the plan empties, the mask
/// does its job, and the history converges instead of churning.
fn quantised(cam: &Camera) -> Camera {
    let q = |v: f64| v.round();
    let eye = Point3::new(q(cam.eye.x), q(cam.eye.y), q(cam.eye.z));
    let t = cam.eye + cam.forward * 10_000.0;
    let target = Point3::new(q(t.x), q(t.y), q(t.z));
    Camera::look_at(eye, target, RVec3::new(0.0, 0.0, 1.0), cam.fov_deg)
}

// ---- what the mask needs to know ---------------------------------------------

/// The poses the history masks on, in millimetres: the being, its shadow on
/// the sand, and — only on a pass whose caustic map was retraced — the door's
/// face.
///
/// The court builds these from balls and one bounding sphere over the net.
/// The cove's three are the three things a moving player changes in the
/// picture: itself, the dark shape it puts on the sand, and the bright one it
/// puts on the stone. The door's face is appended rather than always present
/// so that a *still* being leaves the list unchanged — [`Pose`]'s own
/// comparison then masks nothing, the plan comes back empty, and the pass is a
/// full one that converges. Appending on the frames the caustic moved is
/// exactly the signal the mask wants, because the list growing is what
/// `mask_rects` repaints.
fn poses(scene: &CoveScene, snap: &Snapshot, caustic_moved: bool, glint: Option<Vec3>) -> Vec<Pose> {
    let (centre, _) = snap.being;
    let mut out = Vec::with_capacity(4);
    let c = centre * PER_M;
    out.push(Pose::still([c.x, c.y, c.z], (scene.being_h / 2.0 + scene.being_r) * PER_M));
    if let Some((s, r)) = shadow(scene, centre) {
        let s = s * PER_M;
        out.push(Pose::still([s.x, s.y, s.z], r * PER_M));
    }
    if caustic_moved {
        let f = Vec3::new(scene.door_x, scene.cliff_face_y(), scene.door_sill() + scene.door_h / 2.0) * PER_M;
        out.push(Pose::still([f.x, f.y, f.z], 0.5 * scene.door_w.hypot(scene.door_h) * PER_M));
    }
    // The glint is a moving instance like the being: the list *growing* when it
    // appears, changing when it steps, and shrinking when it goes is exactly
    // what `mask_rects` needs to repaint the sand under it. A glow that
    // appeared into converged history and was never repainted would ghost.
    if let Some(g) = glint {
        let g = g * PER_M;
        out.push(Pose::still([g.x, g.y, g.z], 2.0 * scene.glint_r * PER_M));
    }
    out
}

/// Where the sun puts the being's shadow on the sand, and how wide it is.
/// Metres.
///
/// The history's own `shadow_disc` casts from a point light onto `z = 0`,
/// which is the court's floor and is not the cove's: the sand is a plane at a
/// grade. So the sun's ray is walked to *this* plane instead, and the width is
/// the being's own extent stretched by the sun's elevation — a low sun throws
/// a long shadow, and a mask that did not know it would repaint the wrong end
/// of it.
fn shadow(scene: &CoveScene, centre: Vec3) -> Option<(Vec3, f64)> {
    let d = scene.sun_dir();
    let denom = scene.beach_slope * d.y - d.z;
    if denom.abs() < 1e-9 {
        return None;
    }
    let t = (scene.sand_z_at(centre.x, centre.y) - centre.z) / denom;
    if !(t.is_finite() && t > 0.0) {
        return None;
    }
    let p = centre - d * t;
    let stretch = 1.0 / scene.sun_el.sin().max(0.15);
    Some((p, (scene.being_h / 2.0 + scene.being_r) * stretch))
}

// ---- the simulation ----------------------------------------------------------

/// A frame and the wall-clock moment it is for.
#[derive(Clone, Copy)]
struct Timed {
    frame: Snapshot,
    due: Instant,
}

/// The cove, stepping on its own thread in wall-clock time.
///
/// The court's [`crate::court`] pacing, with a player attached: each frame is
/// due at its own moment, the solver takes fixed `dt` steps to reach it, and
/// the whole frame's worth of accumulated mouse is spread evenly over those
/// steps so that a turn is a rate and not a jolt on the first millisecond.
/// Nothing here predicts: the frame the renderer aims at is the frame the
/// solver would have reached anyway, computed `lookahead` early.
fn simulate(
    tx: Sender<Timed>,
    held: Held,
    latest: Latest,
    gate: Arc<AtomicBool>,
    frames: usize,
    lookahead: Lookahead,
) {
    let scene = match CoveScene::bundled() {
        Ok(s) => s,
        Err(e) => return eprintln!("rune: could not build the cove: {e}"),
    };
    let t0 = Instant::now();
    let baked = match bake::bake(&scene, &Path::new("out").join("maps").join("cove")) {
        Ok(b) => b,
        Err(e) => return eprintln!("rune: could not bake the cove: {e}"),
    };
    eprintln!(
        "rune   the cove baked: {}×{}×{} at {:.0} mm cells in {:.1} s",
        baked.sdf.nx,
        baked.sdf.ny,
        baked.sdf.nz,
        baked.sdf.cell * 1e3,
        t0.elapsed().as_secs_f64()
    );
    let mut cove = match Cove::new(&scene, baked.sdf) {
        Ok(c) => c,
        Err(e) => return eprintln!("rune: could not build the cove: {e}"),
    };
    let dt = cove.dt();
    let fps = scene.authored.parameter_or("fps", 30.0).max(1.0);
    let steps_per_frame = (1.0 / fps / dt).round().max(1.0) as usize;
    let cap = 4 * steps_per_frame;

    let mut start = Instant::now();
    *latest.lock().unwrap_or_else(|e| e.into_inner()) = Some(cove.snapshot());
    let _ = tx.send(Timed { frame: cove.snapshot(), due: start });
    let mut solved = Duration::ZERO;
    let mut said_at = Instant::now();
    let mut said_k = 0usize;
    let mut slips = 0u32;
    let mut said_open = false;
    for k in 1.. {
        if frames > 0 && k > frames {
            break;
        }
        let due_t = k as f64 / fps;
        let mut due = start + Duration::from_secs_f64(due_t);
        let now = Instant::now();
        if now > due + MAX_SLIP {
            slips += 1;
            start = now - Duration::from_secs_f64(due_t);
            due = now;
        }
        let ahead = Duration::from_micros(lookahead.load(Ordering::Relaxed)).min(MAX_LOOKAHEAD);
        if let Some(nap) = due.checked_sub(ahead).and_then(|at| at.checked_duration_since(Instant::now())) {
            std::thread::sleep(nap);
        }

        // What the player is holding, and how much they turned since the last
        // frame. The turn is drained; the direction is not.
        let controls = {
            let mut c = held.lock().unwrap_or_else(|e| e.into_inner());
            let taken = *c;
            c.yaw = 0.0;
            c.tilt = 0.0;
            taken
        };
        let steps = (((due_t - cove.time()) / dt).round().max(0.0) as usize).min(cap);
        // The mouse is a quantity of turn over the frame, so it is divided by
        // the steps the frame is made of; the keys are a force and are not.
        let per = if steps > 0 { 1.0 / steps as f64 } else { 0.0 };
        let input = Input {
            forward: controls.forward,
            strafe: controls.strafe,
            yaw_delta: controls.yaw * per,
            tilt_delta: controls.tilt * per,
        };
        let lap = Instant::now();
        let open = gate.load(Ordering::Relaxed);
        cove.set_gate(open);
        if open && !said_open {
            said_open = true;
            eprintln!("rune   the rune holds: the door is swinging");
        }
        for _ in 0..steps {
            cove.step(&input);
        }
        solved += lap.elapsed();

        let frame = cove.snapshot();
        *latest.lock().unwrap_or_else(|e| e.into_inner()) = Some(frame);
        if tx.send(Timed { frame, due }).is_err() {
            return;
        }
        if said_at.elapsed().as_secs() >= 2 {
            let f = (k - said_k) as f64;
            eprintln!(
                "rune   sim: {:.2} ms solving a frame; {:.0} frames a second of wall clock; \
                 the being at ({:+.2}, {:+.2}) m, {:.1}° of lean, {:.2} m/s{}",
                solved.as_secs_f64() * 1e3 / f,
                f / said_at.elapsed().as_secs_f64(),
                frame.being.0.x,
                frame.being.0.y,
                cove.lean().to_degrees(),
                cove.walking_speed(),
                if slips > 0 { format!("; slipped {slips}×") } else { String::new() },
            );
            said_at = Instant::now();
            said_k = k;
            solved = Duration::ZERO;
            slips = 0;
        }
    }
}

// ---- the rune ----------------------------------------------------------------

/// The score, and the gate it drives, on a thread of their own.
///
/// It is its own thread for one reason: the score is a photon trace. Putting
/// it on the simulation thread would make every step wait for one, and putting
/// it on the render thread would tie the puzzle's clock to the frame rate. It
/// runs as fast as it can on the newest snapshot there is, skips a snapshot
/// whose simulated time it has already scored, and writes one bool.
///
/// The trace is `cove::rune::score`'s own little scene — the being, the door
/// and the sand, three pieces and no BVH over the level — which is why it can
/// afford to be the puzzle's clock at all. The renderer's caustic map is a
/// different trace of a different scene, for a different purpose: that one has
/// to look right, this one has to be a number.
fn rune_worker(latest: Latest, gate: Arc<AtomicBool>, photons: usize, glow: Glow) {
    let scene = match CoveScene::bundled() {
        Ok(s) => s,
        Err(e) => return eprintln!("rune: the score could not build the cove: {e}"),
    };
    let mut g = Gate::new(scene.open_frac);
    let after = glint_after(&scene);
    let mut glint = Glint::new(after, scene.open_frac);
    let mut placed: Option<Vec3> = None;
    // The first aim is forced by `placed` being empty, so this only has to be
    // a moment in the past that exists on every platform.
    let mut aimed = Instant::now();
    let mut scored = f64::NEG_INFINITY;
    let mut said = Instant::now();
    let mut said_cost = false;
    let mut best = 0.0f64;
    loop {
        let snap = *latest.lock().unwrap_or_else(|e| e.into_inner());
        let Some(snap) = snap else {
            std::thread::sleep(Duration::from_millis(10));
            continue;
        };
        if snap.t <= scored {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }
        scored = snap.t;
        let pose = rune_pose(&snap);
        let lap = Instant::now();
        let frac = rune::score(&scene, &pose, photons).frac;
        // What the puzzle's clock costs, once. It is the reason this is a
        // thread and not a line in the solver's loop, so it is worth a line.
        if !said_cost {
            said_cost = true;
            eprintln!("rune   the score is {photons} photons in {:.1} ms", lap.elapsed().as_secs_f64() * 1e3);
        }
        best = best.max(frac);
        let was = g.is_open();
        if g.read(snap.t, frac) != was {
            eprintln!("rune   the door is unlatched at t = {:.2} s", snap.t);
        }
        gate.store(g.is_open(), Ordering::Relaxed);

        // The hint. The state machine is a comparison and runs every pass; the
        // direction is six lattice traces and runs at most every
        // `GLINT_EVERY`, on this thread, because the renderer must never wait
        // for one.
        if glint.read(snap.t, frac) {
            if placed.is_none() || aimed.elapsed() >= GLINT_EVERY {
                aimed = Instant::now();
                match glint_at(&scene, &pose) {
                    Some((at, [ux, uy])) => {
                        if placed.is_none() {
                            eprintln!(
                                "rune   {after:.0} s without the score rising: a glint on the sand at \
                                 ({:+.2}, {:+.2}) m, {:.1} m along ({ux:+.2}, {uy:+.2})",
                                at.x, at.y, scene.glint_step
                            );
                        }
                        placed = Some(at);
                    }
                    // Nothing of the being's light reaches the door's plane at
                    // all from here, so there is no direction to point in and
                    // the level says nothing rather than something wrong.
                    None => placed = None,
                }
            }
        } else if placed.take().is_some() {
            eprintln!("rune   the score rose: the glint is gone");
        }
        *glow.lock().unwrap_or_else(|e| e.into_inner()) = Lit { score: frac, glint: placed };

        // The only thing on screen is the picture, so what the player is told
        // about the rune is told here. Below a hundredth it is silence: an
        // unlit door has nothing to say and would say it every second.
        if said.elapsed().as_secs() >= 1 && frac > 0.01 {
            eprintln!(
                "rune   score {frac:.3} of {:.2} at ({:+.2}, {:+.2}) m, {:+.1}° of lean (best {best:.3})",
                scene.open_frac,
                pose.x,
                pose.y,
                pose.tilt.to_degrees()
            );
            said = Instant::now();
        }
    }
}

// ---- the picture --------------------------------------------------------------

/// What the window asks for: a frame, and the moment it is due on the glass.
///
/// No size. The size is [`render_worker`]'s, because it is chosen from what
/// that thread measured; the window's job is to say *which* frame it wants and
/// *when*, and to blit whatever size comes back.
#[derive(Clone, Copy)]
struct Job {
    frame: Snapshot,
    due: Instant,
}

/// What comes back: the picture, at whatever size the budget chose for it, and
/// the moment it was due.
struct Shot {
    size: (u32, u32),
    rgba: Vec<u8>,
    mask: f32,
    mean_spp: f32,
    due: Instant,
}

/// The live tier's clamp on indirect radiance.
///
/// The still's is the integrator's own — twelve — and at sixty-four samples a
/// pixel a spike that survives it is one sample in a mean of sixty-four. At
/// *one* sample a pass a spike is the whole pixel, the history keeps it for
/// as long as it takes another two hundred samples to average it away, and
/// what the player sees is a white dot sitting on the cliff. So the live tier
/// clamps harder, and the number it can afford to is set by what it is *not*
/// allowed to touch: a [`CausticMap`] read is never clamped at all (the
/// integrator exempts it by construction — it is a density estimate, not a
/// Monte Carlo spike), so the rune's energy is out of reach of this knob and
/// the only thing a tighter clamp can shave is a specular-through-glass path
/// on the cliff and the sky.
///
/// It is not, however, what fixed the cove's picture, and the measurement
/// saying so is why [`History::set_firefly_cap`] exists: 8 → 3 moved the
/// cliff's Laplacian by two and a half per cent and its firefly count by
/// nothing at all, because the dots on this cliff arrive through the *first*
/// hit and the clamp only bounds depths past it. Three is kept because a
/// tighter bound on the indirect term is free and costs the sunlit sand less
/// than one level of sRGB.
const FIREFLY_CLAMP: f32 = 3.0;

/// A ceiling on one sample's luminance, as a multiple of the pixel's running
/// mean. See [`History::set_firefly_cap`] for why the integrator's own clamp
/// is not enough.
const FIREFLY_CAP: f32 = 4.0;

/// À-trous iterations on the resolve.
///
/// The integrator's own default, and left there deliberately. The filter is a
/// fixed cost — measured at 480×270 as about thirteen milliseconds an
/// iteration, sixty-five for all five, against a trace of the same order — and
/// [`DENOISE_FLOOR`] makes it a cost of *every* pass rather than only of the
/// first thirty-two. Dropping to three iterations buys twenty-six of those
/// milliseconds back and is indistinguishable at sixty passes, but it is
/// measurably worse at two (the cliff's Laplacian 2.75 against 2.23, the
/// being's fireflies half again as many) — and two passes is what a pixel the
/// mask just threw away has, which is to say it is the walking picture. The
/// milliseconds are cheap in the state where they are spent: a *standing*
/// picture is the only one whose pixels are converged, and a standing picture
/// is the one that can afford them.
const DENOISE_ITERS: u32 = 5;

/// How much of the à-trous filter a *converged* pixel of this tier keeps.
///
/// See [`History::set_denoise_floor`]. Zero is the court's behaviour and is
/// what this tier used to have.
const DENOISE_FLOOR: f32 = 0.6;

/// The integrator's settings for one raw sample of the cove.
///
/// The level's own `max_depth` — twelve, against the court's five — because
/// every path that carries the rune enters the being and leaves it before it
/// has touched anything. `denoise` is off for the trace and on for the
/// resolve: the pass is one sample and the history is what filters it.
fn options(scene: &CoveScene, seed: u64, denoise: bool) -> PathTraceOptions {
    PathTraceOptions {
        spp: 1,
        denoise_iters: scene.authored.parameter_or("denoise_iters_live", DENOISE_ITERS as f64) as u32,
        max_depth: scene.authored.parameter_or("max_depth", 12.0).max(1.0) as u32,
        rr_start: 2,
        firefly_clamp: Some(scene.authored.parameter_or("firefly_clamp_live", FIREFLY_CLAMP as f64) as f32),
        show_background: true,
        seed,
        denoise,
        ..Default::default()
    }
}

/// The cove's live photon budget, and the level's own knob for it.
fn live_photons(a: &kosm::build::Built) -> usize {
    a.parameter_or("caustic_photons_live", 50_000.0).max(0.0) as usize
}

/// One pass of the picture, and the state that carries between passes.
///
/// Split out of the worker so `--shot` runs the same code the window does. A
/// still is not a different renderer; it is this one, asked the same question
/// several times with nothing moving between.
struct Tracer {
    scene: CoveScene,
    picture: cove_render::Scene,
    history: History,
    film: Film,
    exposure: f32,
    /// The last caustic map, and the pose it was traced at.
    caustics: CausticMap,
    traced_at: Option<Placement>,
    passes: u64,
    /// The view the last pass rendered from, so "the camera moved" is a fact
    /// this side owns rather than one read back out of the history's plan. A
    /// size step is not a camera move, so the comparison is at the new size.
    last_view: Option<View>,
}

impl Tracer {
    fn new(size: (u32, u32)) -> anyhow::Result<Self> {
        let mut scene = CoveScene::bundled()?;
        for w in &scene.authored.warnings {
            eprintln!("rune   warning: {w}");
        }
        // The still's six hundred thousand photons are a minute of walking;
        // the live map is `caustic_photons_live`, and the picture's own scene
        // is the only thing that reads it.
        let photons = live_photons(&scene.authored);
        // The still's knob, overridden for the window. `Built::with` re-runs
        // the level's closure and would drop a knob the closure never asked
        // for, so the resolved list is the one place a derived value like
        // this can land.
        set_knob(&mut scene.authored, "caustic_photons", photons as f64);
        let exposure = scene.authored.parameter_or("exposure", 0.7) as f32;
        let t0 = Instant::now();
        let mut picture = cove_render::Scene::new(&scene)?;
        // The live body is achromatic. A dispersive surface draws one hero
        // wavelength per path the first time a camera ray touches it, and at
        // one sample a pixel a pass that is a saturated *coloured* sample: the
        // body's luminance converges in a second and its colour is confetti
        // for a minute. The level's dispersion is the rune's, and
        // `caustic_map` shoots its photons through the real glass whatever
        // this says, so the spectral rim on the caustic is untouched.
        picture.set_body_dispersion(scene.authored.parameter_or("body_dispersion_live", 0.0) > 0.5);
        eprintln!(
            "rune   the level evaluated: {} static solids, {} live photons, in {:.1} s",
            picture.static_count(),
            photons,
            t0.elapsed().as_secs_f64()
        );
        let mut history = History::new(size);
        // The court's fade hands a pixel back its own raw estimate once it has
        // thirty-two samples, which is right for noise that falls as
        // `1/sqrt(n)` and wrong here: the being is a rough dielectric behind
        // twelve bounces and a few paths in a thousand carry a hundred times
        // the mean, so the body is still speckled at two hundred samples and
        // the filter has been off for a hundred and seventy of them. This is
        // the floor under the fade — a converged pixel keeps this much of the
        // filtered value — and it is the only reason a still of this tier
        // looks like glass rather than like confetti.
        history.set_denoise_floor(
            scene.authored.parameter_or("denoise_floor_live", DENOISE_FLOOR as f64) as f32,
        );
        // …and the filter alone cannot reach a firefly. Its luminance
        // edge-stopping is scaled by the variance plane, and a one-sample pass
        // has almost nothing to put there, so an isolated spike reads to the
        // filter as an edge worth keeping rather than as noise: with the floor
        // at 0.6 and no cap the being still carried forty-eight bright dots
        // per ten thousand pixels at two hundred passes, and with the cap it
        // carries four. So the spikes are caught where they arrive instead.
        history.set_firefly_cap(Some(
            scene.authored.parameter_or("firefly_cap_live", FIREFLY_CAP as f64) as f32,
        ));
        Ok(Self {
            scene,
            picture,
            history,
            film: Film::new(size.0, size.1),
            exposure,
            caustics: CausticMap::empty(),
            traced_at: None,
            passes: 0,
            last_view: None,
        })
    }

    /// Whether the being has moved or leaned far enough to be worth a new map.
    fn caustic_is_stale(&self, p: &Placement) -> bool {
        let Some(was) = self.traced_at.as_ref() else {
            return true;
        };
        let (now_c, now_r) = p.being;
        let (was_c, was_r) = was.being;
        if (now_c - was_c).norm() > CAUSTIC_MOVED_M {
            return true;
        }
        // The lean, as the angle between the two axes: the third column of a
        // body → world rotation is the capsule's own axis.
        let axis = |m: &Mat3| Vec3::new(m[(0, 2)], m[(1, 2)], m[(2, 2)]);
        let d = axis(&now_r).dot(&axis(&was_r)).clamp(-1.0, 1.0);
        d.acos() > CAUSTIC_LEANED_RAD
    }

    /// One pass: trace a raw sample, fold it in, resolve.
    ///
    /// The caustic map is retraced here, on the render thread, when the being
    /// has moved — which is the CPU tier's shape of "repack the caustic". The
    /// GPU tier uploads a `CausticPack` once and re-uploads it when the map
    /// changes; the CPU integrator takes a `&CausticMap` per call, so the map
    /// only has to be *made* before the trace that reads it. A pass that
    /// retraces pays for it inside its own milliseconds, and a pass that does
    /// not shows the last map, which is what "keep showing the last map while
    /// walking" is.
    fn pass(&mut self, frame: &Snapshot, size: (u32, u32), lit: Lit) -> Passed {
        // A size step is a new grid, not a new picture: the history is
        // resampled onto it, keeping the mean, the counts and the guides. That
        // is what lets [`Budget`] change the size while the picture is
        // converging — a climb back to the base after a walk seeds itself from
        // the low-res picture instead of starting from black.
        self.history.resample(size);
        if (self.film.width, self.film.height) != size {
            self.film = Film::new(size.0, size.1);
        }
        let placement = placement_of(frame).with_score(lit.score).with_glint(lit.glint);
        let moved = self.caustic_is_stale(&placement);
        if moved {
            self.caustics = self.picture.caustic_map(&placement);
            self.traced_at = Some(placement);
        }
        let cam = quantised(&cove_render::camera(&self.scene, &placement));
        let view = View::of(&cam, size.0, size.1);
        let poses = poses(&self.scene, frame, moved, lit.glint);
        let plan: Plan = self.history.plan(&view, &poses, &[]);
        // What the budget is told. A size step re-states the stored view at
        // the new size, so the comparison is made there too and a resize is
        // not mistaken for a camera move.
        let camera_moved = self
            .last_view
            .map_or(true, |v| v.at_size(size.0, size.1) != view);
        self.last_view = Some(view);
        let repainted = plan.coverage(size);
        let frame_px = (size.0 as u64) * (size.1 as u64);
        let patch_px: u64 = plan.rects.iter().map(|r| (r[2] as u64) * (r[3] as u64)).sum();
        let full = plan.full || plan.rects.is_empty() || patch_px * 2 > frame_px;

        self.passes += 1;
        let seed = 0x5eed_c0be ^ self.passes.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let picture = self.picture.at(&placement);
        let opts = options(&self.scene, seed, false);
        let map = (!self.caustics.is_empty()).then_some(&self.caustics);
        let traced = if full {
            self.film = pathtrace::render_with_caustics(&picture, &cam, size.0, size.1, &opts, map);
            None
        } else {
            pathtrace::render_into_with_caustics(&picture, &cam, &mut self.film, &opts, &plan.rects, map);
            Some(plan.rects.clone())
        };
        self.history.merge(&self.film, &view, &poses, &[], traced.as_deref());
        let rgba = self.history.resolve(self.exposure, &options(&self.scene, seed, true));
        Passed {
            rgba,
            mask: self.history.mask_fraction(),
            mean_spp: self.history.mean_samples(),
            repainted,
            camera_moved,
        }
    }
}

/// What one pass came back with, and what the budget reads off it.
///
/// `repainted` and `camera_moved` are the two signals [`Budget::next`] takes:
/// the share of the frame the plan asked for, and whether the eye is where it
/// was. A pass with neither is a still one, and a still one is the only kind
/// allowed to grow the picture.
struct Passed {
    rgba: Vec<u8>,
    mask: f32,
    mean_spp: f32,
    repainted: f32,
    camera_moved: bool,
}

/// The renderer: build the picture once, then keep adding passes to whatever
/// the window last asked for, at whatever size [`Budget`] says it can afford.
///
/// The court's worker, with the GPU branch taken out — there is one tier here
/// — and the size restored to something that moves. The job carries the frame
/// and the deadline; the *size* is this thread's, because the size is chosen
/// from what this thread measured and nobody else has that number. A job that
/// asks for a size the budget did not choose is a job whose picture would not
/// match the history the next pass carries.
fn render_worker(jobs: Receiver<Job>, out: Sender<Shot>, glow: Glow, mut budget: Budget, ready: Arc<AtomicBool>) {
    eprintln!("rune   evaluating the level…");
    let mut size = budget.size();
    let mut tracer = match Tracer::new(size) {
        Ok(t) => t,
        Err(e) => return eprintln!("rune: could not build the picture: {e}"),
    };
    // The level is up. `--walk` waits on this: a scripted walk that started
    // during the minute the level takes to evaluate would be over before the
    // first pass, and the measurement it exists for would be of nothing.
    ready.store(true, Ordering::Release);
    eprintln!(
        "rune   the cpu path tracer, {}",
        if budget.is_on() {
            format!(
                "budgeted: {:.0} ms a walking pass, {:.0} ms a still one, {}×{} nominal",
                TARGET_MS, CEILING_MS, WIDTH, HEIGHT
            )
        } else {
            "--budget off: the size is pinned".to_owned()
        }
    );
    let mut current: Option<Job> = None;
    let mut said_at = Instant::now();
    let mut since_said = 0u32;
    loop {
        let mut latest = None;
        loop {
            match jobs.try_recv() {
                Ok(job) => latest = Some(job),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        if latest.is_none() && current.is_none() {
            match jobs.recv() {
                Ok(job) => latest = Some(job),
                Err(_) => return,
            }
        }
        if let Some(job) = latest {
            current = Some(job);
        }
        let Some(job) = current else { continue };
        if size.0 == 0 || size.1 == 0 {
            continue;
        }
        let lap = Instant::now();
        let lit = *glow.lock().unwrap_or_else(|e| e.into_inner());
        let done = tracer.pass(&job.frame, size, lit);
        let ms = lap.elapsed().as_secs_f64() * 1e3;
        let shot = Shot {
            size,
            rgba: done.rgba,
            mask: done.mask,
            mean_spp: done.mean_spp,
            due: job.due,
        };
        since_said += 1;
        // The pace line, once a second: the size and the scale the policy
        // chose, what the pass it chose them from actually cost, and how many
        // of them a second that is. A headless run is read off this.
        let elapsed = said_at.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            said_at = Instant::now();
            eprintln!(
                "rune   cpu {}×{} (×{:.2}{}) at 1 spp: {:.0} ms a pass, {:.1} passes a second, \
                 {:.0}% repainted, {:.1} samples a pixel — {}",
                shot.size.0,
                shot.size.1,
                budget.scale(),
                if budget.is_on() { "" } else { ", pinned" },
                ms,
                since_said as f64 / elapsed,
                100.0 * shot.mask,
                shot.mean_spp,
                if budget.is_still() { "still" } else { "walking" },
            );
            since_said = 0;
        }
        // Measured, then chosen: the next size is a fact about the pass that
        // just ran and about whether anything moved under it.
        size = budget.next(ms, done.repainted, done.camera_moved);
        if out.send(shot).is_err() {
            return;
        }
    }
}

// ---- one still, no window -----------------------------------------------------

/// `kosm run rune --view --shot out/rune.png`: the same tier, headless.
///
/// The being is put at the level's solved pose when it has one and at its
/// spawn when it does not, standing on the sand and facing the door, and the
/// picture is folded `passes` times through the same history the window uses.
/// It is not a different renderer: it is [`Tracer::pass`], asked the same
/// question repeatedly with nothing moving in between, which is exactly the
/// state the window converges to when the player stops walking.
///
/// `walk` is the other half of that: `walk` passes with the being stepped
/// along the beach first, driven through the same [`Budget`] the window uses,
/// and then the rest of them standing. What comes out is a picture of the
/// *climb back* — the policy's transition, in a file, without a window. `walk`
/// of zero is the old behaviour to the byte: nothing moves, the budget sees a
/// still frame from its first pass, and the size never leaves the ladder's
/// still end.
pub fn still(path: &Path, passes: u32, walk: u32, mut budget: Budget) -> anyhow::Result<()> {
    let scene = CoveScene::bundled()?;
    let solution = rune::Pose::solution(&scene);
    // `KOSM_RUNE_SPAWN=1` stands the being where the player finds it even on a
    // solved level. It is how the glint is photographed: the hint is a picture
    // of being stuck, and the solved pose is the one place in the cove nobody
    // is stuck. Paired with `KOSM_GLINT_AFTER`, that is a still of the level's
    // whole hint without a level file of its own.
    let at_spawn = std::env::var("KOSM_RUNE_SPAWN").is_ok_and(|v| v != "0");
    let (x, y, tilt) = if solution.is_solved() && !at_spawn {
        (solution.x, solution.y, solution.tilt)
    } else {
        (scene.spawn_x, scene.spawn_y, 0.0)
    };
    // The still runs the tier off a placement rather than off a simulation:
    // a `Snapshot` is the only thing `Tracer::pass` takes, so one is built
    // that says the being stands there.
    let stand = Placement::standing(&scene, x, y, tilt);
    let frame = snapshot_of(&stand, &scene);

    // The hint, exactly as the window would have it after standing here long
    // enough to earn it. The score is taken once and reused for every pass, so
    // the rim's radiance is a fact about the pose and not about the photon
    // budget's noise, and the still is reproducible.
    let pose = rune_pose(&frame);
    let frac = rune::score(&scene, &pose, live_photons(&scene.authored)).frac;
    let after = glint_after(&scene);
    let mut glint = Glint::new(after, scene.open_frac);
    glint.read(0.0, frac);
    glint.read(after + 1.0, frac);
    let stuck = glint.is_on();
    let aimed = stuck.then(|| glint_at(&scene, &pose)).flatten();
    let lit = Lit { score: frac, glint: aimed.map(|(at, _)| at) };
    match (stuck, aimed) {
        (true, Some((at, [ux, uy]))) => eprintln!(
            "rune   the still is stuck ({after:.0} s without the score rising): a glint at \
             ({:+.2}, {:+.2}) m, {:.1} m along ({ux:+.2}, {uy:+.2})",
            at.x, at.y, scene.glint_step
        ),
        (true, None) => eprintln!("rune   stuck, but nothing of the being reaches the door's plane: no glint"),
        _ => {}
    }

    // A step a pass, along the beach and back — the same order of movement a
    // walking player makes between two passes, so the budget sees the same
    // signal it would see in the window.
    let stride = 0.04;
    let walked = |k: u32| -> Snapshot {
        let d = stride * k as f64;
        snapshot_of(&Placement::standing(&scene, x + d, y, tilt), &scene)
    };

    let mut at = budget.size();
    let mut tracer = Tracer::new(at)?;
    let t0 = Instant::now();
    let mut rgba = Vec::new();
    let mut mean = 0.0;
    let mut shot = at;
    let mut smallest = at;
    for k in 0..passes.max(1) {
        let moving = k < walk;
        let held = if moving { walked(k) } else { walked(walk.saturating_sub(1)) };
        let lap = Instant::now();
        let done = tracer.pass(&held, at, lit);
        let ms = lap.elapsed().as_secs_f64() * 1e3;
        shot = at;
        if (at.0 as u64) * (at.1 as u64) < (smallest.0 as u64) * (smallest.1 as u64) {
            smallest = at;
        }
        if walk > 0 {
            eprintln!(
                "rune   pass {k}: {}×{} (×{:.2}) — {} at {:.0} ms, {:.1} samples a pixel",
                at.0,
                at.1,
                budget.scale(),
                if moving { "walking" } else { "standing" },
                ms,
                done.mean_spp
            );
        }
        rgba = done.rgba;
        mean = done.mean_spp;
        // The still runs the policy on its own measured milliseconds, which is
        // the same number the window feeds it. There is no clock to pace
        // against here, so the passes come as fast as they come — but the
        // *sizes* they come at are the window's policy exactly, which is what
        // makes this picture worth taking.
        at = budget.next(ms, done.repainted, done.camera_moved);
    }
    if walk > 0 {
        eprintln!(
            "rune   the walk took the picture down to {}×{} and standing brought it back to {}×{}",
            smallest.0, smallest.1, shot.0, shot.1
        );
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    image::RgbaImage::from_raw(shot.0, shot.1, rgba)
        .ok_or_else(|| anyhow::anyhow!("the history is the wrong size"))?
        .save(path)?;
    println!(
        "rune   the being {} at ({x:+.2}, {y:+.2}) m, {:.1}° of lean; the rune scores {frac:.3} \
         of {:.2}; {}×{} over {} passes ({:.1} samples a pixel) in {:.1} s → {}",
        match (solution.is_solved(), at_spawn) {
            (true, false) => "at the solved pose",
            (true, true) => "at its spawn (the level is solved; KOSM_RUNE_SPAWN asked)",
            _ => "at its spawn (nothing solved yet)",
        },
        tilt.to_degrees(),
        scene.open_frac,
        shot.0,
        shot.1,
        passes.max(1),
        mean,
        t0.elapsed().as_secs_f64(),
        path.display()
    );
    Ok(())
}

/// A snapshot that says the being stands where a placement puts it.
///
/// The picture takes a `Snapshot` because the window's does; a still has a
/// pose and no simulation. The rotation goes back the way `placement_of` took
/// it apart — `Placement`'s is body → world and `Snapshot`'s is phyz's world →
/// body — and the heading is read off the placement's own facing column.
fn snapshot_of(p: &Placement, scene: &CoveScene) -> Snapshot {
    let (centre, body_to_world) = p.being;
    let f = p.facing();
    Snapshot {
        t: 0.0,
        dt: 1e-3,
        being: (centre, body_to_world.transpose()),
        being_vel: (Vec3::zeros(), Vec3::zeros()),
        facing: f.y.atan2(f.x),
        tilt: 0.0,
        door_angle: p.door_angle,
        door: phyz_math::SpatialTransform::new(
            Mat3::identity(),
            Vec3::new(scene.door_x, scene.cliff_face_y(), scene.door_sill()),
        ),
        gate_open: false,
    }
}

// ---- the window ----------------------------------------------------------------

/// The cove on screen. It owns the recording and the controls; the picture is
/// the render thread's and the puzzle is the rune thread's.
struct App {
    rx: Receiver<Timed>,
    shots: Receiver<Shot>,
    jobs: Sender<Job>,
    held: Held,
    /// Which of the four walking keys are physically down. A held key is a
    /// held force, so the direction is recomputed from this on every press and
    /// release rather than accumulated.
    keys: [bool; 4],
    frames: Vec<Timed>,
    cursor: usize,
    /// The simulated time of the frame the render thread was last asked for.
    asked: Option<f64>,
    pending: Option<(Receiver<Job>, Sender<Shot>)>,
    started: bool,
    glow: Glow,
    /// The render thread's policy, handed over when that thread is spawned.
    budget: Budget,
    /// Set by the render thread once the level has evaluated: what a scripted
    /// walk waits on.
    ready: Arc<AtomicBool>,

    lookahead: Lookahead,
    latency_ms: f64,
    worst_ms: f64,
    dropped: u64,
    late: u64,
    gap: Duration,
    said_pace: Instant,
}

/// W, A, S, D.
const KEY_W: usize = 0;
const KEY_A: usize = 1;
const KEY_S: usize = 2;
const KEY_D: usize = 3;

impl App {
    fn walk(&mut self) {
        let axis = |plus: bool, minus: bool| f64::from(plus) - f64::from(minus);
        let mut c = self.held.lock().unwrap_or_else(|e| e.into_inner());
        c.forward = axis(self.keys[KEY_W], self.keys[KEY_S]);
        c.strafe = axis(self.keys[KEY_D], self.keys[KEY_A]);
    }

    fn head_start(&self) -> Duration {
        Duration::from_micros((self.latency_ms.max(0.0) * 1e3) as u64).min(MAX_LOOKAHEAD)
    }

    /// Which frame the window is for: the moment the picture will be on the
    /// glass, which is now plus the latency it has measured.
    fn pace(&mut self, n: usize) {
        let deadline = Instant::now() + self.head_start();
        let want = match self.frames.iter().rposition(|f| f.due <= deadline) {
            Some(k) => {
                if k == n - 1 && self.frames[k].due + self.gap < deadline {
                    self.late += 1;
                }
                k
            }
            None => 0,
        };
        if want > self.cursor + 1 {
            self.dropped += (want - self.cursor - 1) as u64;
        }
        self.cursor = want;
    }

    fn ask(&mut self) {
        let Some(timed) = self.frames.get(self.cursor) else {
            return;
        };
        let _ = self.jobs.send(Job { frame: timed.frame, due: timed.due });
        self.asked = Some(timed.frame.t);
    }
}

impl viewport::Scene for App {
    /// The render thread wants no device — there is no GPU tier here — but it
    /// is still started from `init`, so that the window is up and saying
    /// something before the level's minute of evaluation begins.
    fn init(&mut self, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        if self.started {
            return;
        }
        self.started = true;
        let Some((jobs, shots)) = self.pending.take() else {
            return;
        };
        let glow = self.glow.clone();
        let budget = self.budget.clone();
        let ready = self.ready.clone();
        std::thread::spawn(move || render_worker(jobs, shots, glow, budget, ready));
    }

    fn event(&mut self, event: viewport::Event) {
        use viewport::{Event, Key};
        let index = |k: Key| match k {
            Key::W => Some(KEY_W),
            Key::A => Some(KEY_A),
            Key::S => Some(KEY_S),
            Key::D => Some(KEY_D),
            _ => None,
        };
        match event {
            Event::Resized(_) => {}
            Event::Look(dx, dy) => {
                let mut c = self.held.lock().unwrap_or_else(|e| e.into_inner());
                // Mouse right turns the being to its right, which is a
                // *decreasing* yaw about +z; mouse forward leans it forward.
                c.yaw -= dx * YAW_PER_UNIT;
                c.tilt -= dy * TILT_PER_UNIT;
            }
            Event::Key(k) => {
                if let Some(i) = index(k) {
                    self.keys[i] = true;
                    self.walk();
                }
            }
            Event::KeyUp(k) => {
                if let Some(i) = index(k) {
                    self.keys[i] = false;
                    self.walk();
                }
            }
            // The cove is walked, not orbited: a drag is the same look the
            // raw motion already gave, and the wheel has nothing to zoom.
            _ => {}
        }
    }

    fn image(&mut self) -> Option<viewport::Image> {
        while let Ok(frame) = self.rx.try_recv() {
            if self.frames.len() == 1 {
                if let Some(prev) = self.frames.last() {
                    self.gap = frame.due.saturating_duration_since(prev.due);
                }
            }
            self.frames.push(frame);
        }
        let n = self.frames.len();
        if n > 0 {
            self.pace(n);
        }
        let mut newest = None;
        while let Ok(shot) = self.shots.try_recv() {
            let ms = Instant::now().saturating_duration_since(shot.due).as_secs_f64() * 1e3;
            self.latency_ms = if self.latency_ms > 0.0 { 0.8 * self.latency_ms + 0.2 * ms } else { ms };
            self.worst_ms = self.worst_ms.max(ms);
            newest = Some(viewport::Image::Bytes { size: shot.size, rgba: shot.rgba });
        }
        self.lookahead.store(self.head_start().as_micros() as u64, Ordering::Relaxed);
        if self.said_pace.elapsed().as_secs() >= 2 {
            self.said_pace = Instant::now();
            let lead = self.frames.last().map_or(0.0, |f| {
                let now = Instant::now();
                f.due.saturating_duration_since(now).as_secs_f64() * 1e3
                    - now.saturating_duration_since(f.due).as_secs_f64() * 1e3
            });
            eprintln!(
                "rune   pace: {:.0} ms presented latency (worst {:.0}), {:.0} ms head start, \
                 the sim {:.0} ms ahead; {} dropped, {} late",
                self.latency_ms,
                self.worst_ms,
                self.head_start().as_secs_f64() * 1e3,
                lead,
                self.dropped,
                self.late,
            );
            self.dropped = 0;
            self.late = 0;
            self.worst_ms = 0.0;
        }
        let key = self.frames.get(self.cursor).map(|f| f.frame.t);
        if key.is_some() && self.asked != key {
            self.ask();
        }
        newest
    }

    /// The cove is walked with the mouse, so the cursor is locked to the
    /// window and hidden. Escape lets go of it and quits.
    fn wants_cursor(&self) -> bool {
        true
    }
}

/// A knob's resolved value, set or added.
///
/// [`kosm::build::Built::with`] re-runs the level's closure, which is the
/// right way to turn a knob the level declares; this is for the derived
/// numbers the window computes for itself and hands back down the same
/// `parameter_or` path the level's own knobs come out of.
fn set_knob(built: &mut kosm::build::Built, name: &str, value: f64) {
    match built.params.iter_mut().find(|p| p.name == name) {
        Some(p) => p.value = value,
        None => built.params.push(kosm::world::Param::new(name, value)),
    }
}

/// `kosm run rune --view`: the cove, live.
///
/// `--shot PATH` takes one still through this same tier instead, `--passes N`
/// says how many passes to fold into it, and `--rune-width N` moves the
/// nominal render size. `--frames N` stops the simulation after N frames,
/// which is how the window is opened in a test.
///
/// The budget's flags, which are the same for the window and the still:
///
/// - `--budget off` pins the size at `--rune-width` — the control the
///   measurement in the commit message was taken against.
/// - `--budget-target MS` is what a walking pass may cost, `--budget-ceiling
///   MS` what a still one may before the pretty rung is given back, and
///   `--rune-pretty N` is that rung's width.
/// - `--walk N` scripts the player. In the window it holds W for `N` seconds
///   from the start, so a headless run walks and then stops without a hand on
///   the keyboard; in a `--shot` it steps the being for the first `N` passes
///   and stands for the rest, which is the picture of the climb back.
pub fn run(args: &kosm_cli::Args) -> anyhow::Result<()> {
    let num = |name: &str| args.value(name).and_then(|v| v.parse().ok());
    let ms = |name: &str| args.value(name).and_then(|v| v.parse::<f64>().ok());
    let width: u32 = num("rune-width").unwrap_or(WIDTH);
    let height = (width * 9 / 16).max(1);
    let pretty: u32 = num("rune-pretty").unwrap_or(width * PRETTY);
    let budget = if args.value("budget").as_deref() == Some("off") {
        Budget::off((width, height))
    } else {
        Budget::new(
            (width, height),
            (pretty, (pretty * 9 / 16).max(1)),
            ms("budget-target").unwrap_or(TARGET_MS),
            ms("budget-ceiling").unwrap_or(CEILING_MS),
        )
    };
    let walk: u32 = num("walk").unwrap_or(0);
    if let Some(path) = args.value("shot") {
        return still(Path::new(path), num("passes").unwrap_or(64), walk, budget);
    }
    // `--cpu` is the court's flag for "do not hand the render thread a
    // device", and this tier never does; it is accepted and says so rather
    // than being rejected.
    if args.flag("cpu") {
        eprintln!("rune   --cpu: this tier is the CPU integrator either way");
    }
    let frames: usize = args.value("frames").and_then(|v| v.parse().ok()).unwrap_or(0);
    window(frames, budget, walk)
}

/// The window itself: four threads and a viewport.
///
/// `walk` seconds of held W at the start, for a run with nobody at the
/// keyboard: it is how the pass rate while walking is measured, and it is
/// exactly the same held direction a key press sets, so the simulation cannot
/// tell the difference.
pub fn window(frames: usize, budget: Budget, walk: u32) -> anyhow::Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    let (job_tx, job_rx) = std::sync::mpsc::channel();
    let (shot_tx, shot_rx) = std::sync::mpsc::channel();
    let lookahead: Lookahead = Arc::new(AtomicU64::new(0));
    let held: Held = Arc::new(Mutex::new(Controls::default()));
    let latest: Latest = Arc::new(Mutex::new(None));
    let gate = Arc::new(AtomicBool::new(false));
    let glow: Glow = Arc::new(Mutex::new(Lit::default()));

    let photons = live_photons(&CoveScene::bundled()?.authored);

    let ready = Arc::new(AtomicBool::new(false));
    if walk > 0 {
        let (held, ready) = (held.clone(), ready.clone());
        std::thread::spawn(move || {
            while !ready.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(50));
            }
            eprintln!("rune   --walk {walk}: holding W for {walk} s, then standing");
            held.lock().unwrap_or_else(|e| e.into_inner()).forward = 1.0;
            std::thread::sleep(Duration::from_secs(walk as u64));
            held.lock().unwrap_or_else(|e| e.into_inner()).forward = 0.0;
            eprintln!("rune   --walk: let go of W");
        });
    }

    {
        let (held, latest, gate, lookahead) = (held.clone(), latest.clone(), gate.clone(), lookahead.clone());
        std::thread::spawn(move || simulate(tx, held, latest, gate, frames, lookahead));
    }
    {
        let (latest, gate, glow) = (latest.clone(), gate.clone(), glow.clone());
        std::thread::spawn(move || rune_worker(latest, gate, photons, glow));
    }

    viewport::run(
        "Kosm — the cove",
        (1280, 720),
        App {
            rx,
            shots: shot_rx,
            jobs: job_tx,
            held,
            keys: [false; 4],
            frames: Vec::new(),
            cursor: 0,
            asked: None,
            pending: Some((job_rx, shot_tx)),
            started: false,
            glow,
            budget,
            ready,
            lookahead,
            latency_ms: 0.0,
            worst_ms: 0.0,
            dropped: 0,
            late: 0,
            gap: Duration::from_millis(33),
            said_pace: Instant::now(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skatepark::Baked;

    /// The cove, baked once for every test that stands a body on it. The bake
    /// is a couple of seconds and cannot have changed between two tests in one
    /// binary.
    fn field() -> anyhow::Result<(CoveScene, &'static Baked)> {
        static FIELD: std::sync::OnceLock<Baked> = std::sync::OnceLock::new();
        static ONCE: Mutex<()> = Mutex::new(());
        let scene = CoveScene::bundled()?;
        if let Some(b) = FIELD.get() {
            return Ok((scene, b));
        }
        let held = ONCE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(b) = FIELD.get() {
            return Ok((scene, b));
        }
        let dir = std::env::temp_dir().join(format!("kosm-rune-{}", std::process::id()));
        let baked = bake::bake(&scene, &dir)?;
        let _ = std::fs::remove_dir_all(&dir);
        let baked = FIELD.get_or_init(|| baked);
        drop(held);
        Ok((scene, baked))
    }

    /// How many photons the tests score with. Well under the live budget: the
    /// checks below are "is it lit at all" and "is it dark", not a measurement.
    const TEST_PHOTONS: usize = 20_000;

    /// **Test 6, the gate's clock.** A second above the threshold opens it;
    /// nine tenths of one does not; a moment below it starts the clock over.
    ///
    /// This is the half of test 6 that does not need a solved level, and it is
    /// the half the rule actually lives in — the score is `cove::rune`'s and is
    /// checked there.
    #[test]
    fn the_gate_wants_a_whole_second_and_not_nine_tenths() {
        // The clock starts at the *first* reading over the threshold, so a
        // hold of `seconds` is that first reading and then `seconds` more.
        let step = 0.05;
        let hold = |seconds: f64| {
            let mut g = Gate::new(0.3);
            let mut t = 0.0;
            g.read(t, 0.5);
            while t < seconds - 1e-9 {
                t += step;
                g.read(t, 0.5);
            }
            g.is_open()
        };
        assert!(!hold(0.9), "nine tenths of a second opened the door");
        assert!(hold(1.0), "a whole second did not open the door");

        // A dip below the threshold restarts the clock: half a second, a
        // moment under, and another nine tenths is not a second.
        let mut g = Gate::new(0.3);
        let mut t = 0.0;
        g.read(t, 0.5);
        for _ in 0..10 {
            t += step;
            g.read(t, 0.5);
        }
        t += step;
        g.read(t, 0.1);
        for _ in 0..18 {
            t += step;
            g.read(t, 0.5);
        }
        assert!(!g.is_open(), "the clock did not restart when the score dipped");
    }

    /// **Test 6, the door.** Driven by the gate the tier drives it with, the
    /// hinge angle rises within a second of the gate opening; left shut, it
    /// does not move at all.
    #[test]
    fn the_door_swings_when_the_gate_opens_and_not_before() -> anyhow::Result<()> {
        let (scene, baked) = field()?;
        let mut cove = Cove::new(&scene, baked.sdf.clone())?;
        cove.hold_still(0.5);
        assert_eq!(cove.door_angle(), 0.0, "the door moved with the gate shut");

        // Nine tenths of a second of a passing score is not enough: the gate
        // never opens, so the door never moves.
        let mut g = Gate::new(scene.open_frac);
        for _ in 0..900 {
            let open = g.read(cove.time(), scene.open_frac + 0.1);
            cove.set_gate(open);
            cove.step(&Input::STILL);
        }
        assert!(!g.is_open(), "the gate opened before a second was up");
        assert!(cove.door_angle().abs() < 1e-9, "the door swung {:.4} rad on a nine-tenths hold", cove.door_angle());

        // The rest of the second, and then a second more to swing in.
        for _ in 0..1_100 {
            let open = g.read(cove.time(), scene.open_frac + 0.1);
            cove.set_gate(open);
            cove.step(&Input::STILL);
        }
        assert!(g.is_open(), "a whole second above the threshold did not unlatch the door");
        assert!(cove.door_angle() > 0.05, "the door only reached {:.4} rad a second after unlatching", cove.door_angle());
        Ok(())
    }

    /// **Test 6, the sun line.** A being behind the door — on the far side of
    /// the cliff face, with the sun's light travelling away from the stone —
    /// scores nothing, so the door can never open there however long the
    /// player stands.
    #[test]
    fn a_being_behind_the_door_scores_nothing() -> anyhow::Result<()> {
        let scene = CoveScene::bundled()?;
        // The sun is low off the −x, −y corner and its light travels into the
        // cliff face, so two metres *past* that face is two metres past every
        // photon that could reach the keyhole.
        let behind = rune::Pose { x: scene.door_x, y: scene.cliff_face_y() + 2.0, tilt: 0.0 };
        let s = rune::score(&scene, &behind, TEST_PHOTONS);
        assert!(s.frac < 1e-6, "a being behind the door scored {:.6}", s.frac);
        assert!(s.frac < scene.open_frac, "and it must never be able to open the door");
        Ok(())
    }

    /// **Test 6, the solved pose.** When the level has one, standing on it
    /// scores over `open_frac` and a second of that opens the door; the same
    /// pose held for nine tenths of a second does not.
    ///
    /// `scene.rs` carries the solver's own output as its `solution_*`
    /// defaults, so this normally runs; a level whose knobs are all zero has
    /// not been solved yet and prints why it is skipping rather than failing.
    #[test]
    fn the_solved_pose_opens_the_door() -> anyhow::Result<()> {
        let (scene, baked) = field()?;
        let solution = rune::Pose::solution(&scene);
        if !solution.is_solved() {
            eprintln!(
                "rune   test 6: `scene.rs` has no solved pose yet (solution_x_mm, \
                 solution_y_mm and solution_tilt_deg are all zero), so the solved half of the \
                 door check is skipped. Step 4's solve fills them in."
            );
            return Ok(());
        }
        let s = rune::score(&scene, &solution, 200_000);
        assert!(
            s.frac > scene.open_frac,
            "the solved pose scored {:.3}, under the level's own {:.2}",
            s.frac,
            scene.open_frac
        );

        // The being placed there, held, and scored through the same reading
        // the window takes — off the snapshot, not off the pose it was placed
        // from, so the sim's own settling is in the number.
        let mut cove = Cove::new(&scene, baked.sdf.clone())?;
        cove.place(solution.x, solution.y, solution.tilt);
        cove.hold_still(0.5);
        let live = rune::score(&scene, &rune_pose(&cove.snapshot()), 200_000);
        assert!(
            live.frac > scene.open_frac,
            "the being settled at the solved pose scored {:.3}, under {:.2}",
            live.frac,
            scene.open_frac
        );

        let mut nine_tenths = Gate::new(scene.open_frac);
        let mut t = 0.0;
        while t < 0.9 - 1e-9 {
            t += 0.05;
            nine_tenths.read(t, live.frac);
        }
        assert!(!nine_tenths.is_open(), "nine tenths of the solved score opened the door");

        let mut g = Gate::new(scene.open_frac);
        let start = cove.time();
        while cove.time() - start < 2.0 {
            let frac = rune::score(&scene, &rune_pose(&cove.snapshot()), TEST_PHOTONS).frac;
            let open = g.read(cove.time(), frac);
            cove.set_gate(open);
            for _ in 0..50 {
                cove.step(&Input::STILL);
            }
        }
        assert!(g.is_open(), "a second at the solved pose did not unlatch the door");
        assert!(cove.door_angle() > 0.05, "the door only reached {:.4} rad", cove.door_angle());
        Ok(())
    }

    /// The two readings of the being agree: a placement built from a pose and
    /// then read back as a pose is the pose it started as.
    ///
    /// Exactly so on the door's own line, where the being faces `+y` and the
    /// score's one-axis tilt is the being's whole lean. Off that line the
    /// facing has an `x` component the score has no room for — a capsule leaned
    /// toward a door that is off to one side is leaned partly across the plane
    /// the score sweeps — and the reading is the projection onto that plane.
    /// It is within a percent at the width of the cove's beach, which is the
    /// error the gate is run with and is stated here so it is not a surprise.
    #[test]
    fn the_being_reads_back_as_the_pose_it_was_placed_at() -> anyhow::Result<()> {
        let scene = CoveScene::bundled()?;
        for tilt in [-0.15, 0.0, 0.07] {
            let p = Placement::standing(&scene, scene.door_x, -3.0, tilt);
            let back = rune_pose(&snapshot_of(&p, &scene));
            assert!((back.x - scene.door_x).abs() < 1e-9 && (back.y + 3.0).abs() < 1e-9);
            assert!((back.tilt - tilt).abs() < 1e-9, "{tilt} read back as {}", back.tilt);

            // Off the line, the projection: same sign, and short by the cosine
            // of how far round the door is.
            let off = Placement::standing(&scene, scene.door_x + 1.5, -3.0, tilt);
            let back = rune_pose(&snapshot_of(&off, &scene));
            assert!((back.tilt - tilt).abs() < 0.01 * tilt.abs().max(1e-3), "{tilt} read back as {} off the door's line", back.tilt);
        }
        Ok(())
    }

    /// **Step 8, the rim.** The keyhole's ring is visible at a score of zero,
    /// rises with the score, and at `open_frac` is plainly brighter than the
    /// sunlit stone it sits on — which is the whole of "the rim glows in
    /// proportion to the score" as a picture rather than as a sentence.
    ///
    /// The comparison is against the door's own outgoing radiance under the
    /// level's sun, `E·cosθ·albedo/π`, because that is what the rim has to beat
    /// to read as a mark on the stone rather than as part of it.
    #[test]
    fn the_rims_radiance_rises_with_the_score_and_beats_the_sunlit_door() -> anyhow::Result<()> {
        use super::super::materials;
        let scene = CoveScene::bundled()?;
        let lum = |c: [f32; 3]| 0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64;
        let rim = |frac: f64| lum(materials::rim(scene.glow_floor, scene.glow_gain, frac).emissive);

        assert!(rim(0.0) > 0.0, "an unlit keyhole is not a puzzle, it is a wall");
        let mut last = rim(0.0);
        for k in 1..=20 {
            let now = rim(k as f64 / 20.0);
            assert!(now > last, "the rim did not rise between {} and {}", (k - 1) as f64 / 20.0, k as f64 / 20.0);
            last = now;
        }

        // the sunlit door, as the integrator will draw it
        let doc = &scene.authored.document;
        let stone = lum(materials::pbr(doc, materials::DOOR).base_color);
        let irr = scene.authored.parameter_or("sun_irradiance", 6.2);
        let cos = scene.sun_dir().dot(&scene.door_frame().normal).abs();
        let sunlit = irr * cos * stone / std::f64::consts::PI;
        assert!(
            rim(scene.open_frac) > 2.0 * sunlit,
            "at open_frac the rim is {:.3} against the sunlit door's {sunlit:.3}",
            rim(scene.open_frac)
        );
        // …and the floor is a mark, not a second sun. It has to sit *above* the
        // sunlit door or a keyhole twenty metres down the beach is a dark dot on
        // bright stone and reads as a hole rather than as a light; it has to sit
        // well under what the score buys, or holding the rune says nothing.
        assert!(rim(0.0) > sunlit, "the floor at {:.3} is darker than the door it marks", rim(0.0));
        assert!(rim(scene.open_frac) > 2.0 * rim(0.0), "holding the rune barely changes the rim");
        Ok(())
    }

    /// **Step 8, the glint's clock.** `glint_after_s` of the best score not
    /// rising brings the glint; a rise takes it away again; and it never
    /// appears while the player is already over `open_frac`.
    ///
    /// Driven on a synthetic clock and synthetic scores, because that is what
    /// the rule is made of — [`glint_at`] is the part that costs a trace and it
    /// is not this test's business.
    #[test]
    fn the_glint_waits_for_thirty_seconds_and_leaves_when_the_score_rises() {
        let mut g = Glint::new(30.0, 0.3);
        // twenty-nine seconds of standing still is not stuck yet
        assert!(!g.read(0.0, 0.05));
        assert!(!g.read(29.0, 0.05));
        assert!(g.read(30.0, 0.05), "thirty seconds without progress did not spark the sand");
        assert!(g.is_on());

        // a rise puts it away and restarts the clock
        assert!(!g.read(31.0, 0.09), "the glint stayed after the score rose");
        assert!(!g.read(60.0, 0.09), "the clock did not restart at the rise");
        assert!(g.read(61.0, 0.09), "and it did not come back thirty seconds later");

        // noise is not progress: half a percent of wander must not reset it
        let mut n = Glint::new(30.0, 0.3);
        let mut t = 0.0;
        while t < 40.0 {
            n.read(t, 0.05 + 0.002 * (t * 7.0).sin());
            t += 0.5;
        }
        assert!(n.is_on(), "photon noise kept the glint from ever appearing");

        // and holding the rune is not being stuck
        let mut h = Glint::new(30.0, 0.3);
        h.read(0.0, 0.4);
        assert!(!h.read(100.0, 0.4), "the glint appeared while the door was unlatching");
    }

    /// **Step 8, the camera at the door.** Wherever the being stands within
    /// three metres of the cliff face — and at any lean, and facing any way —
    /// the eye stays in front of that face, so the picture is never taken from
    /// inside the rock.
    ///
    /// The clearance is the being's own radius and half a metre, which is the
    /// rule the camera states; the assertion is the weaker one that matters,
    /// that the eye is in front of the *face*.
    #[test]
    fn the_camera_never_stands_inside_the_cliff() -> anyhow::Result<()> {
        let scene = CoveScene::bundled()?;
        let face = scene.cliff_face_y() * PER_M;
        let clear = (scene.cliff_face_y() - scene.being_r) * PER_M - 500.0;
        let mut worst = f64::NEG_INFINITY;
        for i in 0..13 {
            let y = scene.cliff_face_y() - 3.0 * i as f64 / 12.0 - scene.being_r;
            for j in 0..13 {
                let x = scene.door_x - 6.0 + 12.0 * j as f64 / 12.0;
                for tilt in [-0.35, -0.17, 0.0, 0.17, 0.35] {
                    let p = Placement::standing(&scene, x, y, tilt);
                    let cam = cove_render::camera(&scene, &p);
                    worst = worst.max(cam.eye.y);
                    assert!(
                        cam.eye.y <= clear + 1e-6,
                        "the eye at ({:.0}, {:.0}, {:.0}) mm is past the cliff face at {face:.0} \
                         for a being at ({x:+.2}, {y:+.2}) m leaning {:.0}°",
                        cam.eye.x,
                        cam.eye.y,
                        cam.eye.z,
                        tilt.to_degrees()
                    );
                    // and it is above the sand it is looking over
                    assert!(cam.eye.z > scene.sand_z_at(cam.eye.x * 1e-3, cam.eye.y * 1e-3) * PER_M);
                }
            }
        }
        assert!(worst > face - 2000.0, "the doorstep camera never came near the door at all");
        Ok(())
    }

    /// The doorstep framing is the one that makes the still readable: at the
    /// solved pose the eye is above the being, off to its shoulder side, and
    /// looking down at the keyhole rather than level with the being's back.
    #[test]
    fn the_solved_pose_is_seen_over_the_shoulder_and_from_above() -> anyhow::Result<()> {
        let scene = CoveScene::bundled()?;
        let solution = rune::Pose::solution(&scene);
        if !solution.is_solved() {
            return Ok(());
        }
        let p = Placement::standing(&scene, solution.x, solution.y, solution.tilt);
        let cam = cove_render::camera(&scene, &p);
        let (c, _) = p.being;
        assert!(cam.eye.z > (c.z + scene.being_h / 2.0) * PER_M, "the eye is not above the being's head");
        assert!(cam.forward.z < -0.25, "the eye is not looking down at the door");
        // the aperture is in front of the camera and not behind the being: the
        // sightline to it passes clear of the capsule's own radius
        let ap = scene.door_frame().origin * PER_M;
        let e = kosm_render::math::Vec3::new(ap.x - cam.eye.x, ap.y - cam.eye.y, ap.z - cam.eye.z);
        assert!(e.dot(&cam.forward) > 0.0, "the keyhole is behind the camera");
        let d = e.normalize();
        let b = kosm_render::math::Vec3::new(c.x * PER_M - cam.eye.x, c.y * PER_M - cam.eye.y, c.z * PER_M - cam.eye.z);
        let off = (b - d * b.dot(&d)).norm();
        assert!(
            off > scene.being_r * PER_M * 0.9,
            "the being sits on the keyhole: the sightline passes {off:.0} mm from its axis"
        );
        Ok(())
    }

    /// The shadow the mask repaints is under the being and down-sun of it: the
    /// sun is low off −x, −y, so the shadow runs to +x, +y and lands on the
    /// sand rather than in the air.
    #[test]
    fn the_shadow_lands_on_the_sand_down_sun_of_the_being() -> anyhow::Result<()> {
        let scene = CoveScene::bundled()?;
        let centre = Vec3::new(0.0, -4.0, scene.sand_z_at(0.0, -4.0) + scene.being_h / 2.0);
        let (p, r) = shadow(&scene, centre).expect("the sun casts a shadow");
        assert!((p.z - scene.sand_z_at(p.x, p.y)).abs() < 1e-9, "the shadow is off the sand plane");
        let d = scene.sun_dir();
        assert!(p.x > centre.x && p.y > centre.y, "the shadow is up-sun of the being ({d:?})");
        assert!(r > scene.being_h / 2.0, "a low sun throws a long shadow, not a short one");
        Ok(())
    }
}
