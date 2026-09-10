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
use kosm_render::pathtrace::{self, Camera, Film, PathTraceOptions, Projection};
use super::being::{Cove, Input, Player, Snapshot};
use super::render::{self as cove_render, PER_M, Placement};
use super::{CoveScene, bake, hint, materials, rune};
use phyz_math::{Mat3, Vec3};

use kosm::player::meter::Meter;
use kosm::player::rig::{Clearance, Interest, Rig, RigKnobs, Subject};
use kosm_view::budget::Budget;
use kosm_view::history::{History, Plan, Pose, View};
use kosm_view::raster;
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

/// The most presented latency the shutter is allowed to buy, in
/// milliseconds.
///
/// The shutter spends *passes* — a frame is not put on the glass until the
/// history has folded [`Rig::shutter_passes`] of them — and a pass on this
/// tier is tens of milliseconds, so an open shutter is latency as directly as
/// it is blur. A running player who has to wait a fifth of a second to see a
/// turn is a player who oversteers, and no amount of integrated motion is
/// worth that. So the fold is capped at whatever fits here, measured against
/// what a pass is actually costing right now — which is the honest place for
/// the cap, because the same four passes are 60 ms at a quarter size and 400
/// at the pretty rung.
///
/// A hundred and twenty milliseconds is about three passes of a walking-size
/// frame on this machine, and it is under the ~150 ms where a mouse turn
/// starts to feel like it is being negotiated rather than made.
const SHUTTER_LATENCY_MS: f64 = 120.0;

/// How long the score has to hold above `open_frac`, in simulated seconds.
const HOLD: f64 = 1.0;

/// How long a `--shot` of the hero lets the figure settle before it is
/// photographed, in simulated seconds.
///
/// A metre-ten figure on a six per cent grade with its boots 236 mm apart
/// across it takes the first couple of seconds to find its stance, and the
/// arm carrying the glass takes about as long to reach where it was asked to
/// go. `sims/rune/tests.rs` measures both.
const SETTLE: f64 = 3.0;

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
fn glint_at(scene: &CoveScene, snap: &Snapshot) -> Option<(Vec3, [f64; 2])> {
    let pose = rune_pose(snap);
    let g = match held_lens(snap) {
        // The hero: the two directions the glass moves when its owner walks.
        Some(lens) => hint::guided_walk(scene, &lens, hint::SWEEP_RAYS)?,
        // The capsule, whose whole body is the lens.
        None => {
            let g = hint::guided_gradient(scene, &pose, hint::SWEEP_RAYS)?;
            [g[0], g[1]]
        }
    };
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

/// The lens the snapshot is holding, as the scorer's refractor.
///
/// `Snapshot::held` is the lens's own frame — `+z` is the optical axis, which
/// is what `hero/kit.rs` cuts the glass about — so the axis is that frame's
/// third column. [`None`] is the capsule, and the capsule is scored on
/// itself.
fn held_lens(snap: &Snapshot) -> Option<rune::Held> {
    let pose = snap.held?;
    Some(rune::Held { centre: pose.pos, axis: pose.rot.mul_vec(Vec3::z()) })
}

/// What the figure is putting in its own light: its skirt, chest and head,
/// where they actually are this step.
///
/// A lens 220 mm across held beside a head 456 mm across is a lens its owner
/// can stand in front of, and a live gate that did not know it would unlatch
/// a door the player cannot see lit. `Snapshot::parts` is in the rig's own
/// link order, so the spec that names the three is the one static
/// [`super::being::HERO_RIG`] holds — no body needed, and none available on
/// this thread.
fn shadows(snap: &Snapshot) -> Vec<rune::Piece> {
    match &snap.parts {
        Some(parts) => rune::occluders(&super::being::HERO_RIG.spec, parts),
        None => Vec::new(),
    }
}

/// The rune's score for whatever body the snapshot is of.
///
/// The hero is scored on the glass in its hand with its own trunk in the way;
/// the capsule is scored on itself, which is every number the offline solve
/// and the solvability sweep were measured with. [`super::being::Cove::rune_score`]
/// is the same two cases read off a `Cove` rather than off a snapshot.
fn live_frac(scene: &CoveScene, snap: &Snapshot, photons: usize) -> f64 {
    match held_lens(snap) {
        Some(lens) => rune::score_lens_at(scene, &lens, &shadows(snap), photons).frac,
        None => rune::score(scene, &rune_pose(snap), photons).frac,
    }
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
        // The figure's solids are the render thread's and are attached in
        // `Tracer::pass`, which is the one place that has both the snapshot
        // and the scene that built them.
        hero: None,
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
    let extent = body_extent(scene, snap);
    out.push(Pose::still([c.x, c.y, c.z], extent * PER_M));
    if let Some((s, r)) = shadow_of(scene, centre, extent) {
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

/// How big a ball the body needs, metres: the radius the mask masks it with
/// and the half-extent its shadow is stretched from.
///
/// The capsule's is exactly its own bounding sphere. The hero's is **one
/// ball round the whole figure** and not a ball per link, and that is the
/// deliberate choice: the mask compares its list of poses for equality frame
/// by frame, and thirteen balls that each wander by a micrometre while the
/// figure stands is thirteen chances a second for the plan to come back
/// non-empty and throw converged pixels away. One ball 0.7 m across covers a
/// 1.11 m figure with its arm up, is still and equal to itself when the hero
/// is still, and repaints exactly what a walking hero changes.
fn body_extent(scene: &CoveScene, snap: &Snapshot) -> f64 {
    if snap.parts.is_some() { HERO_EXTENT } else { scene.being_h / 2.0 + scene.being_r }
}

/// The radius of that ball. Seven hundred millimetres: the hero's own
/// `Rig::DEFAULT` is 1113 mm tall, its capsule proxy's centre sits near the
/// middle of it, and an arm held up at 46° reaches about 640 mm off that
/// centre with the glass on the end of it.
const HERO_EXTENT: f64 = 0.7;

/// Where the sun puts the being's shadow on the sand, and how wide it is.
/// Metres.
///
/// The history's own `shadow_disc` casts from a point light onto `z = 0`,
/// which is the court's floor and is not the cove's: the sand is a plane at a
/// grade. So the sun's ray is walked to *this* plane instead, and the width is
/// the being's own extent stretched by the sun's elevation — a low sun throws
/// a long shadow, and a mask that did not know it would repaint the wrong end
/// of it.
fn shadow_of(scene: &CoveScene, centre: Vec3, extent: f64) -> Option<(Vec3, f64)> {
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
    Some((p, extent * stretch))
}

// ---- the simulation ----------------------------------------------------------

/// A frame and the wall-clock moment it is for.
#[derive(Clone)]
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
    sdf: Arc<kosm_scan::SdfGrid>,
) {
    let scene = match CoveScene::bundled() {
        Ok(s) => s,
        Err(e) => return eprintln!("rune: could not build the cove: {e}"),
    };
    let mut cove = match Cove::new(&scene, (*sdf).clone()) {
        Ok(c) => c,
        Err(e) => return eprintln!("rune: could not build the cove: {e}"),
    };
    eprintln!(
        "rune   the player is {}",
        match cove.player() {
            Player::Hero => "the hero, with the lens in its hand",
            Player::Capsule => "the capsule (KOSM_RUNE_PLAYER=capsule)",
        }
    );
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
        *latest.lock().unwrap_or_else(|e| e.into_inner()) = Some(frame.clone());
        let being = frame.being;
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
                being.0.x,
                being.0.y,
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
        let snap = latest.lock().unwrap_or_else(|e| e.into_inner()).clone();
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
        // Whichever refractor the level actually has. A hero holding the
        // glass is scored on the glass — that is the whole of step 2 — and a
        // capsule is scored on itself, which is every number the sweep and
        // the recorded solution were measured with.
        let frac = live_frac(&scene, &snap, photons);
        // What the puzzle's clock costs, once. It is the reason this is a
        // thread and not a line in the solver's loop, so it is worth a line.
        if !said_cost {
            said_cost = true;
            eprintln!(
                "rune   the score is {photons} photons through {} in {:.1} ms",
                if held_lens(&snap).is_some() { "the lens in the hero's hand" } else { "the being" },
                lap.elapsed().as_secs_f64() * 1e3
            );
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
                match glint_at(&scene, &snap) {
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
#[derive(Clone)]
struct Job {
    frame: Snapshot,
    due: Instant,
}

/// What comes back: the picture, at whatever size the budget chose for it, and
/// the moment it was due.
struct Shot {
    size: (u32, u32),
    /// The picture in memory. Empty when [`Shot::tex`] carries it instead.
    rgba: Vec<u8>,
    /// The picture on the device, when nothing had to come back across the
    /// bus: the raster tier's own colour target, already sRGB-encoded, on the
    /// viewport's own device. See [`raster_worker`].
    tex: Option<std::sync::Arc<wgpu::Texture>>,
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

/// The camera's map, as the *level* says it: `cam_projection`, zero for the
/// pinhole and anything else for the `f·θ` fisheye.
///
/// A number because every resolved knob is a number — the level's knobs are
/// [`kosm::world::Param`]s and a `Param` holds an `f64`. `--projection` is
/// the same choice written in words, because nobody types a projection as a
/// float, and it wins when it is given.
fn projection_knob(a: &kosm::build::Built) -> Projection {
    if a.parameter_or("cam_projection", 0.0) > 0.5 {
        Projection::Equidistant
    } else {
        Projection::Rectilinear
    }
}

/// `--projection rectilinear|equidistant`. `None` leaves the level's own
/// `cam_projection` alone.
fn projection_flag(args: &kosm_cli::Args) -> anyhow::Result<Option<Projection>> {
    match args.value("projection") {
        None => Ok(None),
        Some("rectilinear" | "pinhole" | "flat") => Ok(Some(Projection::Rectilinear)),
        Some("equidistant" | "fisheye") => Ok(Some(Projection::Equidistant)),
        Some(other) => {
            anyhow::bail!("--projection {other}: rectilinear or equidistant")
        }
    }
}

/// `--shutter off|on`. `None` leaves the level's own `cam_shutter` alone,
/// which is one — open — unless the level says otherwise.
fn shutter_flag(args: &kosm_cli::Args) -> anyhow::Result<Option<bool>> {
    match args.value("shutter") {
        None => Ok(None),
        Some("off" | "0" | "no") => Ok(Some(false)),
        Some("on" | "1" | "yes") => Ok(Some(true)),
        Some(other) => anyhow::bail!("--shutter {other}: on or off"),
    }
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
    /// The camera, when it is a *body*: `kosm::player::Rig`, two springs and
    /// an arm, quantised to the millimetre it is written in.
    ///
    /// [`None`] is the offline path — `--shot`, and `kosm run rune`'s own
    /// frame — which keeps [`cove_render::camera`] and is therefore still
    /// pixel-identical to what it was. The two framings are not the same
    /// camera: the rig places its eye from the *body* and pulls its aim to an
    /// [`Interest`] at the aperture, and `render::camera` blends a second eye
    /// measured from the cliff face. Re-expressing the still through the rig
    /// would move every committed frame in the level, so the still keeps the
    /// camera it was composed with and the window gets the one that breathes.
    rig: Option<Rig>,
    /// The simulated time the rig last stepped to, so its `dt` is the sim's
    /// own clock and not the wall's.
    rig_t: Option<f64>,
    /// The exposure meter: a lens over the last pass's radiance, feeding the
    /// tonemap. Built on the first pass, so the level's authored exposure is
    /// taken to be right for the picture the level opens on.
    meter: Option<Meter>,
    /// The link pivots the hero's parts are placed from. Built once.
    pivots: Vec<phyz_math::Vec3>,
    /// How this camera maps the screen. Read off the level's
    /// `cam_projection`, overridable with `--projection`.
    ///
    /// It is a property of the *tracer* and not of one pass because it is
    /// part of the identity of a [`View`]: the history reprojects through it,
    /// and a map that changed between two passes would be a camera that moved
    /// every pixel. The still and the window take it the same way, so a
    /// `--shot` is a picture of what the window would show.
    projection: Projection,
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
        let projection = projection_knob(&scene.authored);
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
            rig: None,
            rig_t: None,
            meter: None,
            pivots: cove_render::Scene::hero_pivots(),
            projection,
        })
    }

    /// The map the command line asked for, over the one the level names.
    fn with_projection(mut self, p: Option<Projection>) -> Self {
        if let Some(p) = p {
            self.projection = p;
        }
        self
    }

    /// Give this tracer the follow camera. What the window does and the still
    /// does not; see [`Tracer::rig`].
    fn with_rig(mut self, sdf: Arc<kosm_scan::SdfGrid>, which: Player) -> Self {
        let a = &self.scene.authored;
        let mut knobs = RigKnobs::from_params(|n, d| a.parameter_or(n, d))
            .in_millimetres()
            .with_water(self.scene.sea_z);
        // Two of the level's numbers were written for a capsule and a capsule
        // has no sides, so the hero re-reads them.
        if which == Player::Hero {
            // **The shoulder step goes the other way.** `cam_side_mm` steps the
            // eye to the being's *right*, which for a capsule is a free
            // choice — a surface of revolution has no sides — and for the
            // hero is the one place the camera must not be: the right hand is
            // what holds the glass, and at the door the eye ends up directly
            // behind the arm the picture is about, with the keyhole behind the
            // lens rather than under it. Stepping to the other shoulder puts
            // the figure on one side of the frame and leaves the door it is
            // aimed at clear, which is the picture: you can see what you are
            // aiming at. What it costs is the glass, which is then on the far
            // shoulder — visible walking, and taken on trust at the door.
            //
            // Widening the step instead was tried and is worse: at three and
            // a half metres out the door goes edge-on and the rim is a dot.
            knobs.side = -a.parameter_or("cam_hero_side_mm", a.parameter_or("cam_side_mm", 2000.0)) / 1000.0;
            // **And the field of view stops chasing the gait.** A settling
            // figure's speed is never exactly zero — its boots are two
            // contacts on a six per cent grade — and four degrees of field of
            // view per metre per second turns a centimetre a second of
            // shuffle into a hundredth of a degree, which is a *moved* camera
            // to `History` and a full repaint every pass for ever. A quarter
            // of a degree is half a per cent of the frame and takes six
            // centimetres a second to reach.
            knobs.fov_quantum_deg = a.parameter_or("cam_fov_quantum_deg", 0.25);
        }
        let ap = self.scene.door_frame().origin;
        self.rig = Some(
            Rig::new(knobs)
                .with_ground(Clearance::sdf(sdf))
                .with_interest(Interest {
                    point: Vec3::new(ap.x, ap.y, ap.z),
                    reach: a.parameter_or("cam_door_reach_m", 2.0).max(1e-3),
                }),
        );
        self
    }

    /// The level this tracer was built from, for a tier that needs the same
    /// numbers — the sun, the sea, the door's hinge.
    fn cove(&self) -> &CoveScene {
        &self.scene
    }

    /// The photon map as it stands, for a tier that gathers it onto a texture
    /// rather than reading it per ray.
    fn caustic_map(&self) -> &CausticMap {
        &self.caustics
    }

    /// What the light meter has multiplied the authored exposure by. One
    /// until a pass has measured something, which is the authored exposure
    /// exactly — the level's own answer for the picture it opens on.
    fn meter_gain(&self) -> f64 {
        self.meter.as_ref().map_or(1.0, |m| m.exposure())
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
    /// `present` is the shutter: `true` resolves the picture for the glass,
    /// `false` folds this pass into the history and stops there. A resolve is
    /// the à-trous filter and a tonemap over the whole frame — a fixed
    /// seventy to ninety milliseconds — so a shutter that resolved every pass
    /// it folded would be paying for pictures nobody is ever shown, and would
    /// cost more frame rate than the fold saves.
    fn pass(&mut self, frame: &Snapshot, size: (u32, u32), lit: Lit, present: bool) -> Passed {
        let ready = self.prepare(frame, lit);
        self.pass_at(frame, size, lit, present, &ready)
    }

    /// The half of a pass that is not a trace: where the body is, whether the
    /// caustic needs retracing, how much simulated time has gone by, and where
    /// the camera has sprung to.
    ///
    /// Split out because **the raster tier needs exactly this and nothing
    /// else**. Both tiers place the same figure, retrace the same photon map
    /// on the same rule, and advance the same rig by the same `dt` — so the
    /// picture the raster draws and the picture the tracer settles into are of
    /// one camera looking at one pose, which is what makes the blend a fade
    /// rather than a double exposure.
    fn prepare(&mut self, frame: &Snapshot, lit: Lit) -> Ready {
        // The figure, placed. The solids are the scene's and were built once;
        // this is one 4×4 a part, from the transforms the simulation put the
        // links at.
        let hero = frame
            .parts
            .as_ref()
            .and_then(|parts| self.picture.hero_at(parts, &self.pivots, frame.held));
        let placement =
            placement_of(frame).with_hero(hero).with_score(lit.score).with_glint(lit.glint);
        let moved = self.sync_caustics(&placement);
        // The clock both the camera and the meter run on: the *simulation's*,
        // so neither depends on how long a pass happened to take.
        let dt = self.rig_t.map_or(frame.dt, |t| (frame.t - t).max(0.0)).max(0.0);
        self.rig_t = Some(frame.t);
        let cam = self.camera(frame, &placement, dt);
        Ready { placement, caustic_moved: moved, dt, cam }
    }

    /// Retrace the photon map if the lens has moved far enough, and say
    /// whether it did.
    ///
    /// The half of [`prepare`](Self::prepare) a tracer that was *handed* its
    /// pose still has to do for itself: `pass_at` reads `self.caustics`, and
    /// the rule for when that is stale is a property of the placement, so two
    /// tracers given the same placements retrace on the same frames and end
    /// up with the same map.
    fn sync_caustics(&mut self, placement: &Placement) -> bool {
        let stale = self.caustic_is_stale(placement);
        if stale {
            self.caustics = self.picture.caustic_map(placement);
            self.traced_at = Some(placement.clone());
        }
        stale
    }

    /// The rest of it: trace a raw sample into `size`, fold it in, resolve.
    fn pass_at(
        &mut self,
        frame: &Snapshot,
        size: (u32, u32),
        lit: Lit,
        present: bool,
        ready: &Ready,
    ) -> Passed {
        // A size step is a new grid, not a new picture: the history is
        // resampled onto it, keeping the mean, the counts and the guides. That
        // is what lets [`Budget`] change the size while the picture is
        // converging — a climb back to the base after a walk seeds itself from
        // the low-res picture instead of starting from black.
        self.history.resample(size);
        if (self.film.width, self.film.height) != size {
            self.film = Film::new(size.0, size.1);
        }
        let Ready { placement, caustic_moved: moved, dt, cam } = ready;
        let (placement, moved, dt, cam) = (placement, *moved, *dt, *cam);
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
        let picture = self.picture.at(placement);
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
        // The meter reads the *film* — this pass's own linear radiance,
        // before any tonemap — and not the picture it just made. Feeding a
        // meter the frame it exposed is a loop with a gain of one: whatever
        // multiplier it chose, the frame comes back that much brighter and
        // the reading says the exposure is already right. One sample a pixel
        // is a noisy estimate of that radiance, but a log-average over a
        // hundred thousand of them is not, and the meter's own second of lag
        // is what is left to smooth.
        let meter = self.meter.get_or_insert_with(|| Meter::calibrated(&self.film.rgb).bounded(0.5, 2.0));
        // The meter runs on every pass, presented or not: it is measuring the
        // *light*, and a pass the shutter swallowed carried as much of it as
        // one the window saw.
        let e = meter.follow(&self.film.rgb, dt);
        let rgba = if present {
            self.history.resolve(self.exposure * e as f32, &options(&self.scene, seed, true))
        } else {
            Vec::new()
        };
        Passed {
            rgba,
            mask: self.history.mask_fraction(),
            mean_spp: self.history.mean_samples(),
            repainted,
            camera_moved,
            exposure: e,
            shutter: self.rig.as_ref().map_or(1, |r| r.shutter_passes()),
        }
    }

    /// Throw the accumulated picture away: the pose it is of is gone.
    ///
    /// What the raster tier does the instant the player moves. The history is
    /// remade rather than masked, because a mask carries pixels through a move
    /// and the whole point of the settle blend is that a move shows the
    /// *raster* frame, sharp, with none of the tracer's past in it.
    fn restart(&mut self, size: (u32, u32)) {
        self.history = History::new(size);
        self.history.set_denoise_floor(
            self.scene.authored.parameter_or("denoise_floor_live", DENOISE_FLOOR as f64) as f32,
        );
        self.history.set_firefly_cap(Some(
            self.scene.authored.parameter_or("firefly_cap_live", FIREFLY_CAP as f64) as f32,
        ));
        self.film = Film::new(size.0, size.1);
        self.last_view = None;
    }

    /// The camera this pass renders from.
    ///
    /// The window's is the rig: a [`Subject`] read off the body's own
    /// snapshot, advanced by the simulation's own `dt`, and quantised to the
    /// millimetre by the rig itself — which is the same rounding
    /// [`quantised`] used to do here and for the same reason, that a still
    /// player must be *still* or the history reprojects for ever.
    ///
    /// The still's is [`cove_render::camera`], unchanged.
    fn camera(&mut self, frame: &Snapshot, placement: &Placement, dt: f64) -> Camera {
        // The map is the tracer's, not the rig's: a `Rig` places an eye and
        // chooses a field of view, and how that field is drawn onto a raster
        // is the renderer's question. Applied at the one point both cameras
        // come through, so the still and the window cannot disagree about it.
        let cam = match self.rig.as_ref() {
            None => quantised(&cove_render::camera(&self.scene, placement)),
            Some(rig) => rig.follow(&subject_of(frame), dt),
        };
        cam.with_projection(self.projection)
    }
}

/// The body, as the camera needs it.
///
/// The rig frames the *proxy centre* and not the root: a figure's root is its
/// pelvis, and a camera aimed at a pelvis puts the head at the top of the
/// frame. `Snapshot::being` is the capsule proxy's centre for both bodies,
/// which sits near the middle of either of them, and it is already what every
/// other reading in this file is taken from.
fn subject_of(snap: &Snapshot) -> Subject {
    let (centre, world_to_body) = snap.being;
    let (s, c) = snap.facing.sin_cos();
    let v = snap.being_vel.0;
    let up = world_to_body.transpose().mul_vec(Vec3::z());
    Subject {
        position: centre,
        velocity: v,
        facing: Vec3::new(c, s, 0.0),
        speed: Vec3::new(v.x, v.y, 0.0).norm(),
        lean: up.z.clamp(-1.0, 1.0).acos(),
    }
}

/// Everything a pass needs that is not a trace. See [`Tracer::prepare`].
///
/// `Clone`, because on the raster tier it crosses a channel: the raster
/// thread is the one that springs the rig and places the body, and the
/// reference is traced on **its** answer rather than on one a second rig
/// arrived at separately. Two rigs fed the same snapshots agree at the fixed
/// point and disagree on the way to it, and the way to it is exactly where a
/// blend would ghost.
#[derive(Clone)]
struct Ready {
    placement: Placement,
    caustic_moved: bool,
    dt: f64,
    cam: Camera,
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
    /// What the light meter multiplied the authored exposure by.
    exposure: f64,
    /// How many passes of the history the rig's shutter is open for.
    ///
    /// [`render_worker`] spends it: it folds this many passes into the
    /// history before resolving one of them to the window, so a walking
    /// player sees fewer frames and each of them integrates the motion
    /// between. One — a still eye — is the loop this tier always ran.
    shutter: u32,
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
fn render_worker(
    jobs: Receiver<Job>,
    out: Sender<Shot>,
    glow: Glow,
    mut budget: Budget,
    ready: Arc<AtomicBool>,
    sdf: Arc<kosm_scan::SdfGrid>,
    projection: Option<Projection>,
    shutter: Option<bool>,
) {
    eprintln!("rune   evaluating the level…");
    let mut size = budget.size();
    let mut tracer = match Tracer::new(size) {
        Ok(t) => t.with_rig(sdf, Player::from_env()).with_projection(projection),
        Err(e) => return eprintln!("rune: could not build the picture: {e}"),
    };
    // The flag over the level's own `cam_shutter`, resolved here because this
    // is the first place both are in hand.
    let shutter_on =
        shutter.unwrap_or_else(|| tracer.scene.authored.parameter_or("cam_shutter", 1.0) > 0.5);
    // The level is up. `--walk` waits on this: a scripted walk that started
    // during the minute the level takes to evaluate would be over before the
    // first pass, and the measurement it exists for would be of nothing.
    ready.store(true, Ordering::Release);
    eprintln!(
        "rune   the camera is {}, the shutter is {}",
        match tracer.projection {
            Projection::Rectilinear => "rectilinear",
            Projection::Equidistant => "an equidistant fisheye",
        },
        if shutter_on {
            "open while the eye moves"
        } else {
            "--shutter off: one pass a frame"
        }
    );
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
    // The shutter, as the loop keeps it. `want` is how many passes this frame
    // is folding — decided when the frame starts and held, so a shutter that
    // closes mid-fold does not strand the passes already spent — and `folded`
    // is how many have gone in.
    //
    // The two costs are kept apart on purpose. A pass that is *not* presented
    // skips the resolve, and the resolve is the à-trous filter over the whole
    // frame: a fixed seventy to ninety milliseconds that does not scale with
    // the picture. So a folded pass and a presented one are not the same
    // measurement, and averaging them together is what a frame's latency must
    // not be estimated from. It is `(want − 1)·fold + frame`.
    let mut folded = 0u32;
    let mut want = 1u32;
    let mut fold_ms = 0.0f64;
    let mut frame_ms = 0.0f64;
    let mut capped = 0u64;
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
        let Some(job) = current.clone() else { continue };
        if size.0 == 0 || size.1 == 0 {
            continue;
        }
        // Every sub-pass of an open shutter takes the *newest* frame it can
        // see, which is what makes the fold a motion blur rather than the
        // same instant traced several times: the history integrates the
        // world moving under it, and the picture that comes out has the
        // motion of the whole exposure in it.
        let lap = Instant::now();
        let lit = *glow.lock().unwrap_or_else(|e| e.into_inner());
        let present = folded + 1 >= want;
        let done = tracer.pass(&job.frame, size, lit, present);
        let ms = lap.elapsed().as_secs_f64() * 1e3;
        let ewma = |was: f64, now: f64| if was > 0.0 { 0.7 * was + 0.3 * now } else { now };
        folded += 1;
        if !present {
            fold_ms = ewma(fold_ms, ms);
            // Mid-exposure the size is *held*. The budget is a policy over
            // what a pass costs, and a pass that skipped its resolve did not
            // cost what a pass costs — feeding it one teaches the cost model
            // a number seventy milliseconds too cheap, and the model answers
            // with a size whose presented pass takes a quarter of a second.
            // Measured, before this was true: the ladder climbed to 240×135
            // while walking and every frame took 253 ms. So the budget is
            // told about presented passes only, one measurement a frame, and
            // the size steps once a frame with it.
            since_said += 1;
            continue;
        }
        frame_ms = ewma(frame_ms, ms);
        let shot = Shot {
            size,
            rgba: done.rgba,
            tex: None,
            mask: done.mask,
            mean_spp: done.mean_spp,
            due: job.due,
        };
        let spent = folded;
        folded = 0;
        // What the *next* frame's shutter is, decided now that this one is
        // out: how far open the rig has it, capped at the number of passes
        // [`SHUTTER_LATENCY_MS`] will pay for. A frame of `n` passes is
        // `(n − 1)` folded ones and one presented, so that is what is solved
        // for — using the presented pass's own cost for both until a folded
        // one has been timed.
        let asked = if shutter_on { done.shutter.max(1) } else { 1 };
        let fold = if fold_ms > 0.0 { fold_ms } else { frame_ms };
        let affordable = if fold > 0.0 {
            (1 + ((SHUTTER_LATENCY_MS - frame_ms).max(0.0) / fold).floor() as u32).max(1)
        } else {
            asked
        };
        want = asked.min(affordable);
        if want < asked {
            capped += 1;
        }
        since_said += 1;
        // The pace line, once a second: the size and the scale the policy
        // chose, what the pass it chose them from actually cost, and how many
        // of them a second that is. A headless run is read off this.
        let elapsed = said_at.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            said_at = Instant::now();
            eprintln!(
                "rune   cpu {}×{} (×{:.2}{}) at 1 spp: {:.0} ms a pass, {:.1} passes a second, \
                 {:.0}% repainted ({:.0}% masked), {:.1} samples a pixel, ×{:.2} exposure, \
                 shutter {}/{}{} ({:.1} frames a second) — {}",
                shot.size.0,
                shot.size.1,
                budget.scale(),
                if budget.is_on() { "" } else { ", pinned" },
                ms,
                since_said as f64 / elapsed,
                100.0 * done.repainted,
                100.0 * shot.mask,
                shot.mean_spp,
                done.exposure,
                spent,
                done.shutter,
                if capped > 0 { "*" } else { "" },
                since_said as f64 / elapsed / spent.max(1) as f64,
                if budget.is_still() { "still" } else { "walking" },
            );
            if capped > 0 {
                eprintln!(
                    "rune   the shutter was capped on {capped} frame(s): {:.0} ms to fold \
                     a pass and {:.0} ms to present one, against a {:.0} ms latency ceiling",
                    // What the cap actually divided by: until a folded pass
                    // has been timed — and a shutter capped shut never folds
                    // one — that is the presented pass's own cost.
                    if fold_ms > 0.0 { fold_ms } else { frame_ms },
                    frame_ms,
                    SHUTTER_LATENCY_MS
                );
                capped = 0;
            }
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


// ---- the raster tier ---------------------------------------------------------

/// The one camera, in the raster tier's units.
///
/// **The two tiers do not measure in the same thing.** The picture the tracer
/// traces is `cove_render`'s document, and a vcad document is in
/// *millimetres* — `PER_M` is the constant, and every ray, every solid and
/// every `Pose` in this file is scaled by it. The raster tier is in
/// **metres**, because that is what the probe lattice's `origin` and
/// `spacing` are in, what `kosm::light::probes` bakes, and what the physics
/// the hero is simulated in uses.
///
/// So the rig — which is told `.in_millimetres()` and therefore hands back an
/// eye in the tracer's units, not the body's — is converted here, once, at
/// the one point a camera crosses from one tier to the other. Only the
/// lengths carry the unit: the basis is three unit vectors and the field of
/// view is an angle, and both are the same number in either.
///
/// Getting this wrong is not subtle and is not a seam: an eye a thousand
/// times too far out sees a level that subtends a milliradian, which is one
/// pixel of sand in the middle of a sky.
fn in_metres(cam: &Camera) -> Camera {
    let s = 1.0 / PER_M;
    Camera {
        eye: Point3::new(cam.eye.x * s, cam.eye.y * s, cam.eye.z * s),
        aperture: cam.aperture * s,
        focus_dist: cam.focus_dist * s,
        ortho_half_height: cam.ortho_half_height.map(|h| h * s),
        ..*cam
    }
}

/// Which tier draws the frame.
///
/// `--tier raster` is the default: the cove's static light is baked into
/// spectral SH probes (`kosm run rune --bake-light`), the geometry is
/// tessellated once, and a wgpu forward pass draws the answer sixty times a
/// second. `--tier trace` is the path tracer this file has always been —
/// unchanged, to the byte, and still the reference the raster settles into.
///
/// See `docs/plans/2026-09-10-bake-and-raster-design.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Raster,
    Trace,
}

/// `--tier raster|trace`. The default is the raster, and it falls back to the
/// tracer, loudly, when the level asks for something the raster cannot draw.
fn tier_flag(args: &kosm_cli::Args) -> anyhow::Result<Tier> {
    match args.value("tier") {
        None => Ok(Tier::Raster),
        Some("raster" | "live" | "gpu") => Ok(Tier::Raster),
        Some("trace" | "tracer" | "cpu" | "reference") => Ok(Tier::Trace),
        Some(other) => anyhow::bail!("--tier {other}: raster or trace"),
    }
}

/// Where the baked light lives, relative to a run's output.
fn probes_path(out: &Path) -> std::path::PathBuf {
    out.join("maps").join("cove").join("probes.bin")
}

/// The level's baked light, or a sky-only stand-in.
///
/// A raster tier with no bake is not an error — it is a level nobody has run
/// `--bake-light` on yet — so what it gets is a volume holding the cove's own
/// sky and no bounce at all. The picture is flatter and the shadows are
/// harder; the geometry, the materials, the sun and the caustic are exactly
/// what they will be, which is enough to work in. The line on stderr says so.
fn cove_probes(scene: &CoveScene, out: &Path) -> raster::ProbeVolume {
    let path = probes_path(out);
    match kosm::light::probes::ProbeVolume::read(&path) {
        Ok(v) => {
            eprintln!(
                "rune   the light: {}×{}×{} probes at {:.0} mm, {} sun{} — {}",
                v.dims[0],
                v.dims[1],
                v.dims[2],
                v.spacing * 1e3,
                v.suns.len(),
                if v.suns.len() == 1 { "" } else { "s" },
                path.display()
            );
            v
        }
        Err(e) => {
            eprintln!(
                "rune   no baked light at {} ({e}); the sky alone. Run `kosm run rune --bake-light` for the bounce.",
                path.display()
            );
            sky_only_probes(scene)
        }
    }
}

/// The cove's gradient sky, projected onto one probe and spread over the
/// level's volume. No occlusion and no bounce: the stand-in.
fn sky_only_probes(scene: &CoveScene) -> raster::ProbeVolume {
    use kosm_render::pathtrace::Environment;
    let (env, _) = cove_render::daylight(scene);
    let Environment::Gradient(g) = env else {
        return raster::probes::uniform([0.0; 3], 1.0, [2, 2, 2], 0.2);
    };
    let (lo, hi) = scene.volume();
    let spacing = 4.0;
    let dims = [
        (((hi.x - lo.x) / spacing).ceil() as u32 + 1).max(2),
        (((hi.y - lo.y) / spacing).ceil() as u32 + 1).max(2),
        (((hi.z - lo.z) / spacing).ceil() as u32 + 1).max(2),
    ];
    raster::probes::from_radiance([lo.x, lo.y, lo.z], spacing, dims, 512, |_, d| {
        // `GradientEnv`: the ground colour below the horizon, and horizon to
        // zenith above it — the same lookup the tracer's environment does.
        let rgb = if d[2] < 0.0 {
            g.ground
        } else {
            let t = d[2] as f32;
            [
                g.horizon[0] + (g.zenith[0] - g.horizon[0]) * t,
                g.horizon[1] + (g.zenith[1] - g.horizon[1]) * t,
                g.horizon[2] + (g.zenith[2] - g.horizon[2]) * t,
            ]
        };
        let c = [rgb[0] * g.intensity, rgb[1] * g.intensity, rgb[2] * g.intensity];
        [c[2], c[2], c[1], c[1], c[0], c[0]]
    })
}

/// The cove, as the raster tier draws it.
///
/// Built once: every solid tessellated, one material per surface name, the
/// sea's lattice, the two caustic receivers. What a frame costs after that is
/// one 4×4 per drawn part.
struct RasterTier {
    scene: raster::Scene,
    raster: raster::Raster,
    /// One entry per mesh: what it is, and its own placement inside its
    /// role's frame.
    parts: Vec<(cove_render::Role, [[f64; 4]; 4], u32)>,
    /// The link pivots the hero's placement needs.
    pivots: Vec<Vec3>,
    /// The keyhole's rim and the glint carry their brightness as a per
    /// *instance* glow rather than as a material, because the brightness is
    /// the score — a number the simulation carries — and a material rebuilt
    /// every frame would be a buffer upload every frame.
    glow_floor: f64,
    glow_gain: f64,
    glint_glow: f64,
    /// Where the caustic receivers are, so the map can be re-gathered when the
    /// lens moves.
    door_quad: ([f64; 3], [f64; 3], [f64; 3]),
    sand_quad: ([f64; 3], [f64; 3], [f64; 3]),
    caustic_res: (u32, u32),
}

/// How fine the two caustic receivers are gathered. 128² over a door a metre
/// and a half across is a centimetre a texel, which is finer than the photon
/// map's own gather radius and so is not the limit on anything.
const CAUSTIC_RES: (u32, u32) = (128, 128);

impl RasterTier {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &CoveScene,
        probes: raster::ProbeVolume,
        format: wgpu::TextureFormat,
    ) -> anyhow::Result<Self> {
        let a = &scene.authored;
        let doc = &a.document;
        let d = scene.sun_dir();
        let irr = a.parameter_or("sun_irradiance", 6.2) as f32;
        let sun = raster::Sun::from_rgb(
            [d.x, d.y, d.z],
            // the same warm low light `render::daylight` gives the tracer
            [irr, 0.77 * irr, 0.46 * irr],
            a.parameter_or("sun_angular_radius", 0.02),
        );
        let (mut rs, _by_name) = raster::Scene::new(sun, probes);
        let (lo, hi) = scene.volume();
        rs.bounds = ([lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z]);
        rs.exposure = a.parameter_or("exposure", 0.7) as f32;

        // One material per surface the level names, through
        // `materials::gpu` — which *is* `materials::pbr` laid over the
        // library, so the two tiers cannot disagree about a colour.
        let mut by_surface: std::collections::HashMap<String, u32> = Default::default();
        let mut material_of = |rs: &mut raster::Scene, name: &str| -> u32 {
            if let Some(i) = by_surface.get(name) {
                return *i;
            }
            let m = match name {
                "N-BK7" => materials::gpu_being(scene.n_d),
                other => materials::gpu(doc, other),
            };
            let i = rs.push_material(m);
            by_surface.insert(name.to_owned(), i);
            i
        };

        let meshes = cove_render::raster_meshes(scene)?;
        let mut parts = Vec::with_capacity(meshes.len() + 1);
        for m in &meshes {
            let mat = material_of(&mut rs, &m.material);
            rs.push_mesh(raster::Mesh::from_mm(&m.positions, &m.normals, &m.indices));
            parts.push((m.role, m.to_world, mat));
        }

        // The sea: the pool tier's construction at the level's own waterline,
        // with the level's own swell. It is the last mesh and carries no role
        // — nothing in the simulation moves it.
        let sand = material_of(&mut rs, "sand");
        let water = material_of(&mut rs, "water");
        let sea = raster::Sea::cove(
            scene.sea_z,
            scene.beach_slope,
            scene.waterline(),
            scene.cove + scene.seabed,
        )
        .with_swell(
            a.parameter_or("swell_a_mm", 30.0) * kosm::scene::MM,
            a.parameter_or("swell_l_mm", 7000.0) * kosm::scene::MM,
            a.parameter_or("swell_b_mm", 20.0) * kosm::scene::MM,
            a.parameter_or("swell_m_mm", 3000.0) * kosm::scene::MM,
        )
        .with_materials(sand, water);
        let (pts, idx) = sea.lattice(scene.cove, 96);
        rs.push_mesh(raster::Mesh::from_m(&pts, &[], &idx));
        parts.push((cove_render::Role::Ground, IDENTITY_ROWS, water));
        rs.sea = Some(sea);

        // The two receivers the caustic is gathered onto: the door's face, and
        // a patch of sand in front of it wide enough to hold a focus thrown
        // wide of the keyhole.
        let face = scene.cliff_face_y();
        let door_quad = (
            [
                scene.door_x - scene.door_w / 2.0,
                face - 0.01,
                scene.door_sill(),
            ],
            [scene.door_w, 0.0, 0.0],
            [0.0, 0.0, scene.door_h],
        );
        let patch = a.parameter_or("caustic_patch_m", 8.0);
        let sand_quad = (
            [
                scene.door_x - patch / 2.0,
                face - patch,
                scene.sand_z_at(scene.door_x, face - patch) + 0.005,
            ],
            [patch, 0.0, 0.0],
            // the sand is a plane at a grade, so the `v` edge rises with it
            [0.0, patch, scene.beach_slope * patch],
        );
        rs.caustics = vec![
            raster::CausticQuad::empty(CAUSTIC_RES),
            raster::CausticQuad::empty(CAUSTIC_RES),
        ];

        let raster = raster::Raster::new(device, queue, &rs, format)?;
        eprintln!(
            "rune   the raster tier: {} meshes, {} triangles, {} materials",
            rs.meshes.len(),
            rs.tris(),
            rs.materials.len()
        );
        Ok(Self {
            scene: rs,
            raster,
            parts,
            pivots: cove_render::Scene::hero_pivots(),
            glow_floor: scene.glow_floor,
            glow_gain: scene.glow_gain,
            glint_glow: scene.glint_glow,
            door_quad,
            sand_quad,
            caustic_res: CAUSTIC_RES,
        })
    }

    /// Put every drawn part where this frame says it is.
    fn place(&mut self, scene: &CoveScene, frame: &Snapshot, placement: &Placement, lit: Lit) {
        let swing = cove_render::door_swing(scene, placement.door_angle);
        let rim_glow = {
            let r = (self.glow_floor + self.glow_gain * lit.score.max(0.0)).max(0.0) as f32;
            [RIM_TINT[0] * r, RIM_TINT[1] * r, RIM_TINT[2] * r]
        };
        let glint_glow = {
            let r = self.glint_glow.max(0.0) as f32;
            [GLINT_TINT[0] * r, GLINT_TINT[1] * r, GLINT_TINT[2] * r]
        };
        let has_hero = frame.parts.as_ref().is_some_and(|p| !p.is_empty());
        for (mesh, (role, local, mat)) in self.scene.meshes.iter_mut().zip(self.parts.iter()) {
            mesh.instances.clear();
            let (frame_rows, kind, glow) = match *role {
                cove_render::Role::Ground => (IDENTITY_ROWS, raster::Kind::Solid, [0.0; 3]),
                cove_render::Role::Door => (swing, raster::Kind::Solid, [0.0; 3]),
                cove_render::Role::Rim => (swing, raster::Kind::Solid, rim_glow),
                cove_render::Role::Glint => match lit.glint {
                    Some(g) => (cove_render::at_frame(g), raster::Kind::Solid, glint_glow),
                    None => continue,
                },
                cove_render::Role::Being => {
                    if has_hero {
                        continue;
                    }
                    let (centre, rot) = placement.being;
                    (cove_render::being_frame(centre, &rot), raster::Kind::Lens, [0.0; 3])
                }
                cove_render::Role::Hero { link } => {
                    let (Some(parts), Some(pivot)) = (frame.parts.as_ref(), self.pivots.get(link))
                    else {
                        continue;
                    };
                    let Some(part) = parts.get(link) else { continue };
                    (cove_render::hero_frame(part, *pivot), raster::Kind::Solid, [0.0; 3])
                }
                cove_render::Role::Held => {
                    let Some(pose) = frame.held else { continue };
                    (cove_render::held_frame(&pose), raster::Kind::Solid, [0.0; 3])
                }
            };
            // The lens's glass is the one part of the figure the raster tier
            // cannot be honest about, so it is drawn as a Fresnel disc; the
            // brass ring beside it is an ordinary solid.
            let kind = if matches!(*role, cove_render::Role::Held) {
                raster::Kind::Lens
            } else {
                kind
            };
            let mut inst = raster::Instance::from_mm_rows(mul_rows(&frame_rows, local), *mat);
            inst = inst.with_kind(kind).with_glow(glow);
            // Neither the sea nor a disc of glass belongs in a shadow map: one
            // is a swell whose depth buffer is a field of acne, the other is a
            // surface light goes through.
            if matches!(kind, raster::Kind::Lens) {
                inst = inst.casting(false);
            }
            mesh.instances.push(inst);
        }
        // the sea, which is the last mesh and is drawn as water
        if let Some(mesh) = self.scene.meshes.last_mut() {
            if self.scene.sea.is_some() {
                mesh.instances.clear();
                let (_, _, mat) = self.parts[self.parts.len() - 1];
                mesh.instances.push(
                    raster::Instance { material: mat, ..Default::default() }
                        .with_kind(raster::Kind::Sea)
                        .casting(false),
                );
            }
        }
    }

    /// Re-gather the photon map onto the two receivers.
    fn gather(&mut self, map: &CausticMap) {
        let quads = [self.door_quad, self.sand_quad];
        self.scene.caustics = quads
            .iter()
            .map(|(o, u, v)| raster::CausticQuad::gather(map, *o, *u, *v, self.caustic_res))
            .collect();
    }

    fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        f: &raster::Frame,
    ) -> std::sync::Arc<wgpu::Texture> {
        self.raster.draw(device, queue, &self.scene, f)
    }
}

/// A row-major identity, millimetres.
const IDENTITY_ROWS: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Two row-major 4×4s, multiplied. The units are the right-hand one's.
fn mul_rows(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0f64; 4]; 4];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, o) in row.iter_mut().enumerate() {
            *o = (0..4).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    out
}

/// The rim's and the glint's tints, the same two `materials.rs` paints them.
const RIM_TINT: [f32; 3] = [1.0, 0.86, 0.58];
const GLINT_TINT: [f32; 3] = [1.0, 0.93, 0.78];


/// The raster tier's render thread.
///
/// The court's shape, one tier over: a job carries a frame and a deadline,
/// this thread draws it, and what comes back is a picture. The differences are
/// the whole design.
///
/// **The raster draws every frame.** No budget ladder and no shutter — a
/// forward pass at the window's own size is milliseconds — so the size is the
/// window's and the frame rate is the display's.
///
/// **The tracer runs only while the hero is still.** A move throws its history
/// away ([`Tracer::restart`]) and the blend drops to zero on that frame, so
/// what the player sees the instant they touch the mouse is the raster frame,
/// sharp. Standing, the tracer accumulates on the rig's exact camera at the
/// budget's size and the blend fades it up: `(spp − 4) / 24`.
///
/// **Nothing crosses the bus while the blend is zero.** A raster-only frame is
/// handed over as the texture it was drawn into. Only once the reference has
/// something to say is the frame read back — and by then nobody is moving, so
/// the readback costs a frame rate nobody is spending.
#[allow(clippy::too_many_arguments)]
fn raster_worker(
    jobs: Receiver<Job>,
    out: Sender<Shot>,
    glow: Glow,
    budget: Budget,
    ready: Arc<AtomicBool>,
    sdf: Arc<kosm_scan::SdfGrid>,
    projection: Option<Projection>,
    settle_on: bool,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: (u32, u32),
) {
    eprintln!("rune   evaluating the level…");
    let mut trace_size = budget.size();
    let player = Player::from_env();
    let field = sdf.clone();
    let mut tracer = match Tracer::new(trace_size) {
        Ok(t) => t.with_rig(sdf, player).with_projection(projection),
        Err(e) => return eprintln!("rune: could not build the picture: {e}"),
    };
    let mut tier = match RasterTier::new(
        &device,
        &queue,
        tracer.cove(),
        cove_probes(tracer.cove(), Path::new("out")),
        wgpu::TextureFormat::Rgba8Unorm,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("rune   the raster tier would not build ({e}); the tracer draws instead");
            ready.store(true, Ordering::Release);
            return trace_loop(jobs, out, glow, budget, tracer, true);
        }
    };
    ready.store(true, Ordering::Release);
    eprintln!(
        "rune   the raster tier is drawing at {}×{}; the reference settles in at {}×{} when the \
         hero is still{}",
        size.0,
        size.1,
        trace_size.0,
        trace_size.1,
        if settle_on { "" } else { " (--no-settle: it does not)" },
    );

    // **The reference runs beside the raster, not in front of it.**
    //
    // This used to be one loop: draw, then trace, then present. A trace pass
    // at 480×270 is a hundred and twenty milliseconds, so the *presented*
    // frame rate fell to seven a second exactly while the picture was
    // settling, and a walk that quantised to a still camera for one frame
    // paid a whole pass before it could show the next one. Now the pass goes
    // to [`settle_worker`] and the raster loop never waits for it: what
    // arrives back is folded into the next frame it draws.
    //
    // The rig is **not** duplicated. The raster thread springs it, and the
    // [`Ready`] it produces is what crosses the channel, so the reference is
    // traced from the raster's own camera and the blend is a fade.
    let (want_tx, want_rx) = std::sync::mpsc::channel::<Trace>();
    let (got_tx, got_rx) = std::sync::mpsc::channel::<Traced>();
    if settle_on {
        std::thread::spawn(move || settle_worker(want_rx, got_tx, budget, field, player, projection));
    }

    let mut settle = if settle_on { raster::Settle::default() } else { raster::Settle::disabled() };
    let mut said_at = Instant::now();
    let mut frames = 0u32;
    let mut raster_ms = 0.0f64;
    let mut prep_ms = 0.0f64;
    let mut last_view: Option<View> = None;
    // The pose the reference is being traced for. A move bumps it, and a
    // frame that comes back stamped with an older one is a picture of a place
    // the player has left — dropped, not blended.
    let mut generation = 0u64;
    let mut outstanding = false;
    let mut reference: Option<(Vec<u8>, (u32, u32))> = None;
    // The meter's multiplier, as the reference last measured it. One until it
    // has: the level's authored exposure is its own answer for the picture it
    // opens on, which is what a walking player is looking at.
    let mut gain = 1.0f64;
    loop {
        // **One job, and it must be a new one.** Re-preparing the frame the
        // loop already drew steps the rig by a `dt` of zero and reads as a
        // still camera however fast the player is walking, which is what used
        // to spend a trace pass in the middle of a stride.
        let mut latest = None;
        loop {
            match jobs.try_recv() {
                Ok(job) => latest = Some(job),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        let job = match latest {
            Some(job) => job,
            None => match jobs.recv() {
                Ok(job) => job,
                Err(_) => return,
            },
        };

        let lap = Instant::now();
        let lit = *glow.lock().unwrap_or_else(|e| e.into_inner());
        // The placement, the caustic and the camera: one rig, stepped once,
        // and both tiers draw the frame it hands back. That is what makes the
        // blend a fade rather than a double exposure.
        let prepared = tracer.prepare(&job.frame, lit);
        if prepared.caustic_moved {
            tier.gather(tracer.caustic_map());
        }
        tier.place(tracer.cove(), &job.frame, &prepared.placement, lit);
        // What the *pose* cost, before a triangle is drawn. On a walking
        // frame this is nearly all of it and nearly all of that is the photon
        // map: the lens moves with the hand, so `sync_caustics` retraces, and
        // fifty thousand photons is tens of milliseconds. It is the level's
        // cost and not the tier's — the trace tier pays exactly the same one
        // out of the same `prepare` — so the pace line states the two halves
        // separately rather than reporting one number that hides it.
        let prep = lap.elapsed().as_secs_f64() * 1e3;
        prep_ms = if prep_ms > 0.0 { 0.7 * prep_ms + 0.3 * prep } else { prep };

        // **Did anything move?** The camera, compared for equality. The rig
        // quantises its eye to the millimetre and its field of view to a
        // quarter of a degree, so a standing player is genuinely still and
        // this is a fact rather than a threshold.
        let view = View::of(&prepared.cam, size.0, size.1);
        let moved = last_view != Some(view);
        last_view = Some(view);
        if moved {
            settle.moved();
            generation += 1;
            reference = None;
        } else {
            settle.still(prepared.dt);
        }

        // Whatever the reference finished while this loop was drawing.
        while let Ok(done) = got_rx.try_recv() {
            outstanding = false;
            if done.generation == generation {
                settle.traced(done.spp);
                trace_size = done.size;
                gain = done.exposure;
                if !done.rgba.is_empty() {
                    reference = Some((done.rgba, done.size));
                }
            }
        }
        // And the next one, if it is worth asking for: nobody is moving, the
        // reference is not already there, and the last one has come back.
        if settle_on && !moved && !settle.is_settled() && !outstanding {
            let ask = Trace {
                frame: job.frame.clone(),
                ready: prepared.clone(),
                lit,
                generation,
            };
            outstanding = want_tx.send(ask).is_ok();
        }

        let mut f = raster::Frame::new(in_metres(&prepared.cam), size);
        f.exposure = tier.scene.exposure * gain as f32;
        f.time = job.frame.t as f32;
        f.caustics_dirty = prepared.caustic_moved;
        let drew = Instant::now();
        let texture = tier.draw(&device, &queue, &f);
        let ms = drew.elapsed().as_secs_f64() * 1e3;
        raster_ms = if raster_ms > 0.0 { 0.7 * raster_ms + 0.3 * ms } else { ms };

        let blend = settle.blend();
        let shot = if blend > 0.0 {
            // **The bus is crossed only once the reference has something to
            // say.** Below that the frame is handed over as the texture it
            // was drawn into, and nothing is read back at all — which is the
            // whole of a walking player's frame budget.
            let img = kosm_view::frame::read_back(&device, &queue, &texture, size).into_raw();
            let rgba = match &reference {
                Some((bytes, tsize)) => raster::settle::present(&img, size, bytes, *tsize, blend),
                // a blend above zero with nothing to blend cannot happen —
                // the samples come with the picture — but a raw raster frame
                // is the honest fallback and not a panic
                None => img,
            };
            Shot { size, rgba, tex: None, mask: 0.0, mean_spp: settle.spp(), due: job.due }
        } else {
            Shot {
                size,
                rgba: Vec::new(),
                tex: Some(texture),
                mask: 0.0,
                mean_spp: settle.spp(),
                due: job.due,
            }
        };

        frames += 1;
        let elapsed = said_at.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            said_at = Instant::now();
            eprintln!(
                "rune   raster {}×{}: {:.1} fps, {:.1} ms drawing + {:.1} ms posing, blend \
                 {:.2} at {:.1} spp ({}×{} traced) — {}",
                size.0,
                size.1,
                frames as f64 / elapsed,
                raster_ms,
                prep_ms,
                blend,
                settle.spp(),
                trace_size.0,
                trace_size.1,
                if moved { "moving" } else { "still" },
            );
            frames = 0;
        }
        if out.send(shot).is_err() {
            return;
        }
    }
}

/// What the raster thread asks the reference for: a pose, and which pose it
/// is.
struct Trace {
    frame: Snapshot,
    ready: Ready,
    lit: Lit,
    generation: u64,
}

/// What comes back. `rgba` is empty when the pass folded without resolving.
struct Traced {
    rgba: Vec<u8>,
    size: (u32, u32),
    spp: f32,
    /// What the light meter multiplied the authored exposure by.
    ///
    /// It rides back with the picture because **the meter lives with the
    /// tracer**: it is a lens over the radiance a pass measured, and the
    /// raster thread no longer takes any passes. Tone-mapping the raster half
    /// of the blend at the authored exposure while the traced half arrived at
    /// a measured one is a seam, and it is a seam that gets worse the more
    /// the level's light changes.
    exposure: f64,
    generation: u64,
}

/// The reference, on its own thread.
///
/// It owns a second [`Tracer`] — the level evaluated twice, a tenth of a
/// second — and it never springs a rig: the camera, the placement and the
/// `dt` all arrive in the [`Ready`] the raster thread computed, so the two
/// pictures are of one pose and the blend has no parallax in it. What it
/// *does* do for itself is [`Tracer::sync_caustics`], because `pass_at`
/// reads its own photon map and the rule for when that is stale is a
/// property of the placement it was handed.
///
/// A new generation is a new pose: the history is thrown away rather than
/// masked, for the reason [`Tracer::restart`] gives.
fn settle_worker(
    want: Receiver<Trace>,
    out: Sender<Traced>,
    mut budget: Budget,
    sdf: Arc<kosm_scan::SdfGrid>,
    player: Player,
    projection: Option<Projection>,
) {
    let mut size = budget.size();
    let mut tracer = match Tracer::new(size) {
        Ok(t) => t.with_rig(sdf, player).with_projection(projection),
        Err(e) => return eprintln!("rune: the reference would not build: {e}"),
    };
    let mut generation = u64::MAX;
    loop {
        // Only the newest ask matters: an older one is a pose the player has
        // already left, and tracing it would be spending the frame budget on
        // a picture nobody will be shown.
        let mut ask = match want.recv() {
            Ok(a) => a,
            Err(_) => return,
        };
        while let Ok(newer) = want.try_recv() {
            ask = newer;
        }
        if ask.generation != generation {
            generation = ask.generation;
            tracer.restart(size);
        }
        tracer.sync_caustics(&ask.ready.placement);
        let t0 = Instant::now();
        let done = tracer.pass_at(&ask.frame, size, ask.lit, true, &ask.ready);
        let at = size;
        size = budget.next(t0.elapsed().as_secs_f64() * 1e3, done.repainted, done.camera_moved);
        let reply = Traced {
            rgba: done.rgba,
            size: at,
            spp: done.mean_spp,
            exposure: done.exposure,
            generation: ask.generation,
        };
        if out.send(reply).is_err() {
            return;
        }
    }
}

/// The trace tier's loop with a tracer already built — the fallback path when
/// the raster tier cannot be made on this machine.
fn trace_loop(
    jobs: Receiver<Job>,
    out: Sender<Shot>,
    glow: Glow,
    mut budget: Budget,
    mut tracer: Tracer,
    shutter_on: bool,
) {
    let mut size = budget.size();
    let mut current: Option<Job> = None;
    let mut folded = 0u32;
    let mut want = 1u32;
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
        let Some(job) = current.clone() else { continue };
        let lap = Instant::now();
        let lit = *glow.lock().unwrap_or_else(|e| e.into_inner());
        let present = folded + 1 >= want;
        let done = tracer.pass(&job.frame, size, lit, present);
        folded += 1;
        if !present {
            continue;
        }
        want = if shutter_on { done.shutter.max(1) } else { 1 };
        folded = 0;
        let shot = Shot {
            size,
            rgba: done.rgba,
            tex: None,
            mask: done.mask,
            mean_spp: done.mean_spp,
            due: job.due,
        };
        size = budget.next(lap.elapsed().as_secs_f64() * 1e3, done.repainted, done.camera_moved);
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
pub fn still(
    path: &Path,
    passes: u32,
    walk: u32,
    mut budget: Budget,
    projection: Option<Projection>,
) -> anyhow::Result<()> {
    let scene = CoveScene::bundled()?;
    let solution = rune::Pose::solution(&scene);
    // `KOSM_RUNE_SPAWN=1` stands the being where the player finds it even on a
    // solved level. It is how the glint is photographed: the hint is a picture
    // of being stuck, and the solved pose is the one place in the cove nobody
    // is stuck. Paired with `KOSM_GLINT_AFTER`, that is a still of the level's
    // whole hint without a level file of its own.
    let at_spawn = std::env::var("KOSM_RUNE_SPAWN").is_ok_and(|v| v != "0");
    let (mut x, mut y, tilt) = if solution.is_solved() && !at_spawn {
        (solution.x, solution.y, solution.tilt)
    } else {
        (scene.spawn_x, scene.spawn_y, 0.0)
    };
    // How the being gets into the picture, and it is two different things.
    //
    // The **capsule** is a pose and nothing else: a `Snapshot` is the only
    // thing `Tracer::pass` takes, so one is built that says the being stands
    // there, and no simulation runs at all. That path is unchanged and is
    // what `kosm run rune`'s own frame and every committed still are.
    //
    // The **hero** cannot be posed that way, because a figure is not three
    // numbers: where its knees are, and where the glass in its hand got to,
    // are the answer to a settle. So the hero's still bakes the field, stands
    // a real [`Cove`] on it, and holds it still for [`SETTLE`] seconds — the
    // same body the window walks, asked to stop moving.
    let which = Player::from_env();
    let mut field: Option<Arc<kosm_scan::SdfGrid>> = None;
    let frame = match which {
        Player::Capsule => snapshot_of(&Placement::standing(&scene, x, y, tilt), &scene),
        Player::Hero => {
            // Where the *hero* stands, which is not where the capsule does.
            // `solution_*` is the capsule's pose — 400 mm off the cliff, with
            // a metre of glass for a body — and a figure standing on it has
            // its own hood over the keyhole it is solving. The hero's is
            // `hero::doorstep`'s: the stance solved for the lens it is
            // holding, mapped out of `hero/stage.rs`'s frame (the cove with
            // the keyhole at the origin) into the cove's by the one
            // translation that separates them.
            let step = super::hero::doorstep();
            let ap = scene.door_frame().origin;
            let (hx, hy) = (step.stance.feet.x * kosm::scene::MM + ap.x, step.stance.feet.y * kosm::scene::MM + ap.y);
            let (hx, hy) = if at_spawn { (x, y) } else { (hx, hy) };
            let yaw = if at_spawn { std::f64::consts::FRAC_PI_2 } else { std::f64::consts::FRAC_PI_2 + step.stance.yaw };
            let t0 = Instant::now();
            let baked = bake::bake(&scene, &Path::new("out").join("maps").join("cove"))?;
            field = Some(Arc::new(baked.sdf.clone()));
            let mut cove = Cove::with_player(&scene, baked.sdf, Player::Hero)?;
            cove.face(yaw);
            cove.place(hx, hy, 0.0);
            cove.hold_still(SETTLE);
            eprintln!(
                "rune   the hero settled at ({hx:+.2}, {hy:+.2}) m, facing {:.0}°, in {:.1} s: {} parts, the lens {}",
                yaw.to_degrees(),
                t0.elapsed().as_secs_f64(),
                cove.hero_parts().len(),
                match cove.held_lens() {
                    Some(p) => format!("at ({:+.2}, {:+.2}, {:+.2}) m", p.pos.x, p.pos.y, p.pos.z),
                    None => "nowhere".to_owned(),
                }
            );
            (x, y) = (hx, hy);
            cove.snapshot()
        }
    };

    // The hint, exactly as the window would have it after standing here long
    // enough to earn it. The score is taken once and reused for every pass, so
    // the rim's radiance is a fact about the pose and not about the photon
    // budget's noise, and the still is reproducible.
    let photons = live_photons(&scene.authored);
    let frac = live_frac(&scene, &frame, photons);
    let after = glint_after(&scene);
    let mut glint = Glint::new(after, scene.open_frac);
    glint.read(0.0, frac);
    glint.read(after + 1.0, frac);
    let stuck = glint.is_on();
    let aimed = stuck.then(|| glint_at(&scene, &frame)).flatten();
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
    // signal it would see in the window. The hero does not walk here: its
    // pose is a settle and a settle is not a function of `k`, so `--walk` on
    // a hero still holds the settled frame and measures the ladder's climb
    // with the picture standing.
    let stride = 0.04;
    let walked = |k: u32| -> Snapshot {
        if which == Player::Hero {
            return frame.clone();
        }
        let d = stride * k as f64;
        snapshot_of(&Placement::standing(&scene, x + d, y, tilt), &scene)
    };

    let mut at = budget.size();
    // The **capsule's** still keeps [`cove_render::camera`], and keeps it to
    // the byte: `kosm run rune`'s `frame.png` is composed with that camera and
    // is the picture every measurement in this level has been read off.
    //
    // The **hero's** still gets the rig, snapped rather than sprung — a first
    // call has no history to be late against, so `follow` with no elapsed time
    // places the eye where it wants to be and stops. That is the third of the
    // options this camera had, taken exactly where it costs nothing: there is
    // no committed hero frame for it to move, and the doorstep the hero stands
    // on is a metre and a quarter off the cliff rather than the capsule's four
    // hundred millimetres — far enough out that
    // [`cove_render::camera`]'s blend is only a third of the way round and the
    // keyhole ends up behind the hood. The rig's [`Interest`] is on the
    // aperture and its arm is on the same field the boots stand on, so it
    // frames the door from over the hero's shoulder wherever the hero is.
    let mut tracer = Tracer::new(at)?.with_projection(projection);
    if let Some(sdf) = field {
        tracer = tracer.with_rig(sdf, which);
    }
    if tracer.projection != Projection::Rectilinear {
        eprintln!("rune   the still is an equidistant fisheye");
    }
    let t0 = Instant::now();
    let mut rgba = Vec::new();
    let mut mean = 0.0;
    let mut shot = at;
    let mut smallest = at;
    for k in 0..passes.max(1) {
        let moving = k < walk;
        let held = if moving { walked(k) } else { walked(walk.saturating_sub(1)) };
        let lap = Instant::now();
        let done = tracer.pass(&held, at, lit, true);
        let ms = lap.elapsed().as_secs_f64() * 1e3;
        shot = at;
        if (at.0 as u64) * (at.1 as u64) < (smallest.0 as u64) * (smallest.1 as u64) {
            smallest = at;
        }
        if walk > 0 {
            // `repainted` is on this line and not only on the window's,
            // because it is the number that says whether the history is
            // *converging*: a picture whose plan keeps asking for the whole
            // frame is a picture starting over every pass, and under a map
            // this history had never reprojected through that is exactly the
            // failure to watch for. Headless, it is the only way to see it.
            eprintln!(
                "rune   pass {k}: {}×{} (×{:.2}) — {} at {:.0} ms, {:.1} samples a pixel, \
                 {:.0}% repainted",
                at.0,
                at.1,
                budget.scale(),
                if moving { "walking" } else { "standing" },
                ms,
                done.mean_spp,
                100.0 * done.repainted
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
        "rune   the {} {} at ({x:+.2}, {y:+.2}) m, {:.1}° of lean; the rune scores {frac:.3} \
         of {:.2}; {}×{} over {} passes ({:.1} samples a pixel) in {:.1} s → {}",
        match which {
            Player::Hero => "hero",
            Player::Capsule => "being",
        },
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
        // A still built from a placement is a pose, and a pose has no arms.
        // The hero's still is not built this way — it steps a `Cove` and
        // takes its snapshot, which is where the parts and the glass come
        // from; see [`still`].
        held: None,
        parts: None,
    }
}


/// `kosm run rune --view --shot out/rune.png --tier raster`: one raster frame,
/// headless.
///
/// The same tier the window draws, on a device this function opens for itself
/// — `kosm_render::gpu::GpuContext` is the headless adapter and it is the one
/// the crate's own GPU tests run on. The hero settles for [`SETTLE`] seconds
/// exactly as [`still`] settles it, so the two stills are of one pose and can
/// be differenced.
///
/// `--settle` folds `passes` of the reference over the raster frame at the
/// blend those samples earn, which is the picture a standing player ends up
/// looking at. Without it the file is the raster frame alone, which is what a
/// walking player sees.
fn raster_still(
    path: &Path,
    size: (u32, u32),
    budget: Budget,
    projection: Option<Projection>,
    settle_in: bool,
    passes: u32,
) -> anyhow::Result<()> {
    let size = (size.0.max(16), if size.1 == 0 { (size.0 * 9 / 16).max(9) } else { size.1 });
    let ctx = kosm_render::gpu::GpuContext::init_blocking()
        .map_err(|e| anyhow::anyhow!("--tier raster needs a device: {e}"))?;
    let (device, queue) = (&ctx.device, &ctx.queue);

    let mut budget = budget;
    let trace_size = budget.size();
    let mut tracer = Tracer::new(trace_size)?.with_projection(projection);
    if !raster::Raster::supports(tracer.projection) {
        anyhow::bail!(
            "--tier raster cannot draw an equidistant fisheye: `r = f·θ` is not a projective \
             map. Use --tier trace, or --projection rectilinear."
        );
    }
    let scene = CoveScene::bundled()?;
    let which = Player::from_env();
    // The pose: the hero settled at its doorstep, or the capsule at the level's
    // solution — the same two [`still`] photographs, so the two files differ in
    // the tier and in nothing else.
    // **The rig, always.** Not for the unit — the rig is told
    // `.in_millimetres()` and hands back an eye in the tracer's units like
    // everything else, and [`in_metres`] is what crosses it over — but
    // because the rig is the camera the *window* draws with. A still composed
    // through `cove_render::camera` and a window composed through the rig are
    // two different framings, and a parity number between the tiers would
    // then be measuring the composition. One camera, converted once.
    let (frame, sdf) = still_pose(&scene, which)?;
    tracer = tracer.with_rig(sdf, which);

    let photons = live_photons(&scene.authored);
    let frac = live_frac(&scene, &frame, photons);
    let lit = Lit { score: frac, glint: None };

    let mut tier = RasterTier::new(
        device,
        queue,
        tracer.cove(),
        cove_probes(tracer.cove(), Path::new("out")),
        wgpu::TextureFormat::Rgba8Unorm,
    )?;

    let prepared = tracer.prepare(&frame, lit);
    tier.gather(tracer.caustic_map());
    tier.place(tracer.cove(), &frame, &prepared.placement, lit);

    // **The reference first, when there is going to be one.**
    //
    // The exposure is not the level's authored number, it is that number
    // times what [`Meter`] measured off the last pass — and a meter that has
    // never seen a pass reads one. In the window that only means the raster
    // is a frame behind a gain that is already converged; in a *still* it
    // means the raster half of the blend would be drawn at the authored
    // exposure and the traced half at the measured one, which is two
    // different pictures fading into each other. Tracing first costs the
    // still nothing — the pose is prepared and neither half depends on the
    // other — and it is what makes the blend a fade.
    let mut blend = 0.0f32;
    let mut spp = 0.0f32;
    let mut traced: Option<(Passed, (u32, u32))> = None;
    if settle_in {
        let mut settle = raster::Settle::default();
        let mut at = trace_size;
        for _ in 0..passes.max(1) {
            let p = tracer.pass_at(&frame, at, lit, true, &prepared);
            settle.traced(p.mean_spp);
            let sized = at;
            at = budget.next(0.0, p.repainted, p.camera_moved);
            blend = settle.blend();
            spp = p.mean_spp;
            traced = Some((p, sized));
        }
    }

    let gain = tracer.meter_gain();
    let t0 = Instant::now();
    let mut f = raster::Frame::new(in_metres(&prepared.cam), size);
    f.exposure = tier.scene.exposure * gain as f32;
    f.time = frame.t as f32;
    f.caustics_dirty = true;
    let texture = tier.draw(device, queue, &f);
    let raster_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut rgba = kosm_view::frame::read_back(device, queue, &texture, size).into_raw();
    if let Some((p, sized)) = traced.as_ref() {
        rgba = raster::settle::present(&rgba, size, &p.rgba, *sized, blend);
    }

    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    image::RgbaImage::from_raw(size.0, size.1, rgba)
        .ok_or_else(|| anyhow::anyhow!("the raster frame is the wrong size"))?
        .save(path)?;
    println!(
        "rune   the raster tier drew {}×{} in {:.1} ms at exposure {:.3} (×{gain:.3} metered); \
         the rune scores {frac:.3} of {:.2}; blend {blend:.2} at {spp:.1} samples a pixel → {}",
        size.0,
        size.1,
        raster_ms,
        f.exposure,
        scene.open_frac,
        path.display()
    );
    Ok(())
}

/// The pose both stills photograph: the hero settled at its doorstep, or the
/// capsule at the level's solution.
///
/// Lifted out of [`still`] so the raster tier stands the same body in the same
/// place — a parity number between two tiers means nothing if they are looking
/// at two poses.
fn still_pose(
    scene: &CoveScene,
    which: Player,
) -> anyhow::Result<(Snapshot, Arc<kosm_scan::SdfGrid>)> {
    let solution = rune::Pose::solution(scene);
    let at_spawn = std::env::var("KOSM_RUNE_SPAWN").is_ok_and(|v| v != "0");
    let (x, y, tilt) = if solution.is_solved() && !at_spawn {
        (solution.x, solution.y, solution.tilt)
    } else {
        (scene.spawn_x, scene.spawn_y, 0.0)
    };
    let baked = bake::bake(scene, &Path::new("out").join("maps").join("cove"))?;
    match which {
        Player::Capsule => Ok((
            snapshot_of(&Placement::standing(scene, x, y, tilt), scene),
            Arc::new(baked.sdf),
        )),
        Player::Hero => {
            let step = super::hero::doorstep();
            let ap = scene.door_frame().origin;
            let (hx, hy) = (
                step.stance.feet.x * kosm::scene::MM + ap.x,
                step.stance.feet.y * kosm::scene::MM + ap.y,
            );
            let (hx, hy) = if at_spawn { (x, y) } else { (hx, hy) };
            let yaw = if at_spawn {
                std::f64::consts::FRAC_PI_2
            } else {
                std::f64::consts::FRAC_PI_2 + step.stance.yaw
            };
            let sdf = Arc::new(baked.sdf.clone());
            let mut cove = Cove::with_player(scene, baked.sdf, Player::Hero)?;
            cove.face(yaw);
            cove.place(hx, hy, 0.0);
            cove.hold_still(SETTLE);
            eprintln!(
                "rune   the hero settled at ({hx:+.2}, {hy:+.2}) m, facing {:.0}°: {} parts",
                yaw.to_degrees(),
                cove.hero_parts().len()
            );
            Ok((cove.snapshot(), sdf))
        }
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
    /// The baked field, handed to the render thread with everything else it
    /// is started with: the camera's arm keeps the eye out of it.
    sdf: Arc<kosm_scan::SdfGrid>,
    /// The camera's map and whether the shutter is spent, both handed to the
    /// render thread with everything else it is started with.
    projection: Option<Projection>,
    shutter: Option<bool>,
    /// Which tier draws, and whether it settles into the reference.
    tier: Tier,
    settle: bool,
    /// The window's own size in physical pixels, which is what the raster tier
    /// draws at — there is no ladder on that tier, because a forward pass at
    /// full size is milliseconds.
    window_px: (u32, u32),

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
        let _ = self.jobs.send(Job { frame: timed.frame.clone(), due: timed.due });
        self.asked = Some(timed.frame.t);
    }
}

impl viewport::Scene for App {
    /// The render thread wants no device — there is no GPU tier here — but it
    /// is still started from `init`, so that the window is up and saying
    /// something before the level's minute of evaluation begins.
    fn init(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
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
        let sdf = self.sdf.clone();
        let (projection, shutter) = (self.projection, self.shutter);
        // The raster tier draws on the *viewport's* device, so the texture it
        // draws into is the texture the blit samples and a raster-only frame
        // never crosses the bus.
        let (device, queue) = (device.clone(), queue.clone());
        let (settle, size) = (self.settle, self.window_px);
        match self.tier {
            Tier::Raster => {
                std::thread::spawn(move || {
                    raster_worker(
                        jobs, shots, glow, budget, ready, sdf, projection, settle, device, queue,
                        size,
                    )
                });
            }
            Tier::Trace => {
                std::thread::spawn(move || {
                    render_worker(jobs, shots, glow, budget, ready, sdf, projection, shutter)
                });
            }
        }
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
            Event::Resized(px) => self.window_px = px,
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
            newest = Some(match shot.tex {
                // Nothing crossed the bus: the raster tier drew into a texture
                // on this very device and the blit samples it where it is.
                Some(t) => viewport::Image::Texture(t),
                None => viewport::Image::Bytes { size: shot.size, rgba: shot.rgba },
            });
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
///
/// And the camera's own two:
///
/// - `--projection rectilinear|equidistant` is how the screen is mapped onto
///   directions, over the level's `cam_projection`. The fisheye is live now,
///   not offline-only: `kosm_view::View` carries the map and the history
///   reprojects through it, so a walking player under an `f·θ` camera
///   converges exactly as a pinhole one does. The still and the window take
///   the same flag, so a `--shot` is a picture of what the window shows.
/// - `--shutter on|off` (over the level's `cam_shutter`, on by default) folds
///   [`Rig::shutter_passes`] passes into one presented frame while the eye is
///   moving. The frame rate drops and the passes integrate the motion between
///   them, which is what a shutter is; standing still it is one pass a frame
///   and nothing changes. Capped at [`SHUTTER_LATENCY_MS`] of passes, because
///   a shutter is latency as directly as it is blur.
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
    let projection = projection_flag(args)?;
    let tier = tier_flag(args)?;
    let settle = !args.flag("no-settle");
    if let Some(path) = args.value("shot") {
        return match tier {
            Tier::Trace => {
                still(Path::new(path), num("passes").unwrap_or(64), walk, budget, projection)
            }
            Tier::Raster => raster_still(
                Path::new(path),
                (num("rune-width").unwrap_or(1280), 0),
                budget,
                projection,
                settle && args.flag("settle"),
                num("passes").unwrap_or(64),
            ),
        };
    }
    // `--cpu` is the court's flag for "do not hand the render thread a
    // device", and this tier never does; it is accepted and says so rather
    // than being rejected.
    if args.flag("cpu") {
        eprintln!("rune   --cpu: this tier is the CPU integrator either way");
    }
    let frames: usize = args.value("frames").and_then(|v| v.parse().ok()).unwrap_or(0);
    window(frames, budget, walk, projection, shutter_flag(args)?, tier, settle)
}

/// The window the cove opens at, physical pixels. The raster tier draws at
/// this size and the tracer at the budget's, which is why the settle blend
/// upscales the reference on its way over.
const WINDOW: (u32, u32) = (1280, 720);

/// The window itself: four threads and a viewport.
///
/// `walk` seconds of held W at the start, for a run with nobody at the
/// keyboard: it is how the pass rate while walking is measured, and it is
/// exactly the same held direction a key press sets, so the simulation cannot
/// tell the difference.
#[allow(clippy::too_many_arguments)]
pub fn window(
    frames: usize,
    budget: Budget,
    walk: u32,
    projection: Option<Projection>,
    shutter: Option<bool>,
    tier: Tier,
    settle: bool,
) -> anyhow::Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    let (job_tx, job_rx) = std::sync::mpsc::channel();
    let (shot_tx, shot_rx) = std::sync::mpsc::channel();
    let lookahead: Lookahead = Arc::new(AtomicU64::new(0));
    let held: Held = Arc::new(Mutex::new(Controls::default()));
    let latest: Latest = Arc::new(Mutex::new(None));
    let gate = Arc::new(AtomicBool::new(false));
    let glow: Glow = Arc::new(Mutex::new(Lit::default()));

    let scene = CoveScene::bundled()?;
    let photons = live_photons(&scene.authored);

    // The field, baked once and shared: the body stands on it and the
    // camera's arm keeps out of it. It used to be the simulation thread's
    // alone, which was fine while the camera was a formula; a rig with a
    // spring arm needs the same distances the feet do, and baking it twice
    // would be two copies of sixty megabytes and four seconds of nobody's
    // time.
    let t0 = Instant::now();
    let baked = bake::bake(&scene, &Path::new("out").join("maps").join("cove"))?;
    eprintln!(
        "rune   the cove baked: {}×{}×{} at {:.0} mm cells in {:.1} s",
        baked.sdf.nx,
        baked.sdf.ny,
        baked.sdf.nz,
        baked.sdf.cell * 1e3,
        t0.elapsed().as_secs_f64()
    );
    let sdf = Arc::new(baked.sdf);

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
        let sdf = sdf.clone();
        std::thread::spawn(move || simulate(tx, held, latest, gate, frames, lookahead, sdf));
    }
    {
        let (latest, gate, glow) = (latest.clone(), gate.clone(), glow.clone());
        std::thread::spawn(move || rune_worker(latest, gate, photons, glow));
    }

    viewport::run(
        "Kosm — the cove",
        WINDOW,
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
            sdf,
            projection,
            shutter,
            tier,
            settle,
            window_px: WINDOW,
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
    use kosm::scene::MM;

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
        let mut cove = Cove::with_player(&scene, baked.sdf.clone(), Player::Capsule)?;
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
        // The capsule, said out loud: this is the *capsule's* solved pose and
        // the capsule's own caustic, which is what `solution_*` records.
        let mut cove = Cove::with_player(&scene, baked.sdf.clone(), Player::Capsule)?;
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

    /// **The hero is drawn where the simulation put it.** Every part of the
    /// figure that the picture places comes back at its own link's transform,
    /// and the glass comes back at the pose `Body::held` says the arm got it
    /// to — not at the pose the arm was asked for, and not at the origin.
    ///
    /// The check is the one that catches the mistake this is made of: the
    /// hero-millimetre → body-metre change of frame is a rotation *and* a
    /// scale, and getting either wrong puts the whole figure somewhere
    /// plausible and wrong. So a named part with a known place — the boot,
    /// which is on the sand, and the head, which is a metre over it — is
    /// asked where it ended up in the world, and compared with where the body
    /// says its own link is.
    #[test]
    fn the_heros_parts_are_drawn_at_the_snapshots_transforms() -> anyhow::Result<()> {
        let (scene, baked) = field()?;
        let mut cove = Cove::with_player(&scene, baked.sdf.clone(), Player::Hero)?;
        cove.face(std::f64::consts::FRAC_PI_2);
        cove.place(scene.solution_x, scene.solution_y, 0.0);
        cove.hold_still(3.0);
        let snap = cove.snapshot();
        let parts = snap.parts.clone().expect("the hero has parts");

        let picture = cove_render::Scene::new(&scene)?;
        let pivots = cove_render::Scene::hero_pivots();
        let placed = picture.hero_at(&parts, &pivots, snap.held).expect("the hero is placed");
        assert!(placed.len() > 30, "the figure came back as {} solids", placed.len());

        // Where a named part's link actually is, world millimetres.
        let link = |name: &str| -> Vec3 {
            let i = parts.iter().position(|p| p.name == name).unwrap_or_else(|| panic!("no `{name}` link"));
            parts[i].pose.pos * PER_M
        };
        // …and where the solids the picture drew for it ended up, as the mean
        // of their translations.
        let drawn = |name: &str| -> Vec3 {
            let mut sum = Vec3::zeros();
            let mut n = 0.0;
            for part in placed.iter().filter(|p| p.name == name) {
                let c = part.to_world.matrix.c3;
                sum += Vec3::new(c.x, c.y, c.z);
                n += 1.0;
            }
            assert!(n > 0.0, "nothing was drawn for `{name}`");
            sum / n
        };

        // The boot rides the boot link: its solids are built about the ankle
        // and sit within a boot's radius of it.
        for (solid, joint) in [("boot_r", "boot_r"), ("boot_l", "boot_l")] {
            let (a, b) = (drawn(solid), link(joint));
            assert!(
                (a - b).norm() < 400.0,
                "`{solid}` was drawn {:.0} mm from its own ankle",
                (a - b).norm()
            );
            assert!(a.z < link("pelvis").z, "a boot was drawn above the hips");
        }
        // The hood rides the neck: over the pelvis and near the head.
        let (hood, neck) = (drawn("hood"), link("neck"));
        assert!((hood - neck).norm() < 500.0, "the hood was drawn {:.0} mm off the neck", (hood - neck).norm());
        assert!(hood.z > link("pelvis").z + 300.0, "the hood was drawn at hip height");

        // And the glass is at the held pose, to the millimetre.
        let held = snap.held.expect("the hero holds the lens");
        let lens = drawn("lens");
        let want = held.pos * PER_M;
        assert!((lens - want).norm() < 1e-6, "the lens was drawn {:.1} mm off the pose the arm got it to", (lens - want).norm());
        // …which is up, out, and clear of the hood.
        assert!(lens.z > hood.z - 400.0, "the lens is not held up: {:.0} mm against the hood's {:.0}", lens.z, hood.z);
        assert!((lens - hood).norm() > 300.0, "the lens is {:.0} mm from the hood — it is inside the head", (lens - hood).norm());
        println!(
            "the hero draws {} solids; the lens is at ({:+.0}, {:+.0}, {:+.0}) mm, {:.0} mm clear of the hood",
            placed.len(),
            lens.x,
            lens.y,
            lens.z,
            (lens - hood).norm()
        );
        Ok(())
    }

    /// **The live gate reads the glass.** With the hero holding the lens at
    /// the cove's doorstep, `rune::score_lens` puts light through the keyhole:
    /// the score is a positive number and it is *bigger* than what the same
    /// lens throws from twenty metres down the beach.
    ///
    /// It is not asserted against `open_frac`, and the reason is named rather
    /// than hidden: **the hero's solve is not wired**. `hero::doorstep` solves
    /// where a hero must stand for the lens it is holding to put the sun in
    /// the keyhole — two constraints, two unknowns, iterated — and it does it
    /// in `hero/stage.rs`'s frame, which is the cove's translated so the
    /// keyhole is the origin. Wiring it means a `--solve hero` beside
    /// `rune::solve_and_record`: map the stance into the cove, drive the body
    /// to it, and record the pose the way `solution_*` records the capsule's.
    /// TODO(rune): `--solve hero`, from `sims/rune/hero/mod.rs::doorstep`.
    /// Until then this is what the level honestly has — a lens that throws a
    /// caustic at the door, and a number that rises as it is aimed.
    #[test]
    fn the_held_lens_lights_the_keyhole_from_the_doorstep() -> anyhow::Result<()> {
        let (scene, baked) = field()?;
        let mut cove = Cove::with_player(&scene, baked.sdf.clone(), Player::Hero)?;

        // The doorstep, mapped: `hero/stage.rs` is the cove with the keyhole
        // moved to the origin, so the two frames differ by one translation.
        let step = crate::rune::hero::doorstep();
        let ap = scene.door_frame().origin;
        let (fx, fy) = (step.stance.feet.x * MM + ap.x, step.stance.feet.y * MM + ap.y);
        cove.face(std::f64::consts::FRAC_PI_2 + step.stance.yaw);
        cove.place(fx, fy, 0.0);
        cove.hold_still(SETTLE);

        let near = cove.lens().expect("the hero holds the lens");
        let here = rune::score_lens(&scene, &near, 200_000);
        println!(
            "the hero at the doorstep ({fx:+.2}, {fy:+.2}) m holds the lens at \
             ({:+.2}, {:+.2}, {:+.2}) m along ({:+.2}, {:+.2}, {:+.2}); the rune scores {:.4} \
             of {:.2} ({:.3e} deposited over {:.3e} incident)",
            near.centre.x,
            near.centre.y,
            near.centre.z,
            near.axis.x,
            near.axis.y,
            near.axis.z,
            here.frac,
            scene.open_frac,
            here.deposited[1],
            here.incident,
        );
        assert!(here.frac > 0.0, "the lens at the doorstep put nothing through the keyhole");

        // …and it is the *aim* that earns it: the same lens, the same pose,
        // twenty metres down the beach, scores nothing.
        let away = rune::Held { centre: near.centre + Vec3::new(0.0, -20.0, 0.0), ..near };
        let there = rune::score_lens(&scene, &away, 200_000);
        assert!(
            there.frac < here.frac,
            "the lens twenty metres out scored {:.4} against the doorstep's {:.4}",
            there.frac,
            here.frac
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
        let extent = scene.being_h / 2.0 + scene.being_r;
        let (p, r) = shadow_of(&scene, centre, extent).expect("the sun casts a shadow");
        assert!((p.z - scene.sand_z_at(p.x, p.y)).abs() < 1e-9, "the shadow is off the sand plane");
        let d = scene.sun_dir();
        assert!(p.x > centre.x && p.y > centre.y, "the shadow is up-sun of the being ({d:?})");
        assert!(r > scene.being_h / 2.0, "a low sun throws a long shadow, not a short one");
        Ok(())
    }

    /// **The cove, on both tiers, at the pose the level is about.**
    ///
    /// The design's number: the raster frame against the traced one at the
    /// hero's solved pose, mean absolute difference over the sand and over
    /// the door's face, and it is one camera — [`Tracer::prepare`] places the
    /// body, retraces the caustic and springs the rig once, and both tiers
    /// draw what it hands back. The raster's copy of that camera goes through
    /// [`in_metres`], which is the whole of the units story.
    ///
    /// **The two regions are not held to one tolerance, and the reason is the
    /// tier's own design.**
    ///
    /// - The **sand** is sunlit, and on a sunlit surface the direct term is
    ///   most of the answer. The raster computes it per pixel — the same
    ///   `E · max(0, n·s)` the tracer integrates, against a 2048² shadow map
    ///   — so the two agree to a fraction of a code. This is the number that
    ///   catches the mistake that matters: `sims/rune/bake.rs` bakes the
    ///   volume with `sun_direct: false` exactly so the shader is not adding
    ///   a sun the probes already carry, and if that flag ever flips back
    ///   this assert is a factor of 1.8 out.
    /// - The **door's face** is in the cliff's shadow, so every photon on it
    ///   arrived by bounce, and bounce is the half the raster tier reads out
    ///   of nine spherical harmonics on a half-metre lattice where the tracer
    ///   integrates twelve of them per pixel. Five per cent low is what L2
    ///   costs on a plane whose radiance field has a hard horizon in it;
    ///   neither a finer lattice (250 mm) nor eight times the rays moves it,
    ///   which is how we know it is the truncation and not the bake.
    ///
    /// Skips without an adapter, like the rest of the tier's GPU tests.
    #[test]
    fn the_cove_agrees_with_the_reference_on_the_sand_and_the_door() -> anyhow::Result<()> {
        let Ok(ctx) = kosm_render::gpu::GpuContext::init_blocking() else {
            eprintln!("skipping the_cove_agrees_with_the_reference: no GPU");
            return Ok(());
        };
        let (device, queue) = (&ctx.device, &ctx.queue);
        let size = (320u32, 180u32);
        let scene = CoveScene::bundled()?;
        let which = Player::from_env();

        let mut tracer = Tracer::new(size)?;
        let (frame, sdf) = still_pose(&scene, which)?;
        tracer = tracer.with_rig(sdf, which);
        let lit = Lit { score: live_frac(&scene, &frame, live_photons(&scene.authored)), glint: None };

        let probes = cove_probes(tracer.cove(), Path::new("out"));
        if probes.suns.is_empty() || probes.data.iter().all(|v| *v == 0.0) {
            eprintln!("skipping the_cove_agrees_with_the_reference: no baked light");
            return Ok(());
        }
        let mut tier = RasterTier::new(
            device,
            queue,
            tracer.cove(),
            probes,
            wgpu::TextureFormat::Rgba8Unorm,
        )?;

        let prepared = tracer.prepare(&frame, lit);
        tier.gather(tracer.caustic_map());
        tier.place(tracer.cove(), &frame, &prepared.placement, lit);

        // The reference first, so the meter has measured before the raster is
        // drawn at its exposure — the same order [`raster_still`] takes.
        let mut reference = Vec::new();
        for _ in 0..32 {
            let p = tracer.pass_at(&frame, size, lit, true, &prepared);
            if !p.rgba.is_empty() {
                reference = p.rgba;
            }
        }
        anyhow::ensure!(!reference.is_empty(), "the tracer resolved nothing");

        let cam = in_metres(&prepared.cam);
        let mut f = raster::Frame::new(cam, size);
        f.exposure = tier.scene.exposure * tracer.meter_gain() as f32;
        f.time = frame.t as f32;
        f.caustics_dirty = true;
        let texture = tier.draw(device, queue, &f);
        let raster = kosm_view::frame::read_back(device, queue, &texture, size).into_raw();

        // Where to look, in the world rather than in the frame: a pixel box
        // written down by hand would be a box about one composition, and the
        // rig springs.
        let vp = kosm_view::raster::pipeline::view_proj(&cam, size.0 as f32 / size.1 as f32);
        let project = |p: [f64; 3]| -> Option<(u32, u32)> {
            let (x, y, z, w) = (
                vp[0] * p[0] as f32 + vp[4] * p[1] as f32 + vp[8] * p[2] as f32 + vp[12],
                vp[1] * p[0] as f32 + vp[5] * p[1] as f32 + vp[9] * p[2] as f32 + vp[13],
                vp[2] * p[0] as f32 + vp[6] * p[1] as f32 + vp[10] * p[2] as f32 + vp[14],
                vp[3] * p[0] as f32 + vp[7] * p[1] as f32 + vp[11] * p[2] as f32 + vp[15],
            );
            if w <= 0.0 || z < 0.0 {
                return None;
            }
            let (sx, sy) = ((x / w * 0.5 + 0.5) * size.0 as f32, (0.5 - y / w * 0.5) * size.1 as f32);
            (sx >= 4.0 && sy >= 4.0 && sx < size.0 as f32 - 4.0 && sy < size.1 as f32 - 4.0)
                .then(|| (sx as u32, sy as u32))
        };
        let mad = |at: &[(u32, u32)]| -> Option<f64> {
            let (mut sum, mut n) = (0.0f64, 0.0f64);
            for (cx, cy) in at {
                for y in cy.saturating_sub(3)..(cy + 4).min(size.1) {
                    for x in cx.saturating_sub(3)..(cx + 4).min(size.0) {
                        let i = ((y * size.0 + x) * 4) as usize;
                        for c in 0..3 {
                            sum += (raster[i + c] as f64 - reference[i + c] as f64).abs() / 255.0;
                            n += 1.0;
                        }
                    }
                }
            }
            (n > 0.0).then(|| sum / n)
        };

        let door = scene.door_frame().origin;
        let face = scene.cliff_face_y();
        // the door's face, a little off the keyhole either way so the rim's
        // own glow and the caustic's focus are not what is being measured
        let on_door: Vec<_> = [-0.45f64, 0.45]
            .iter()
            .filter_map(|dx| {
                project([door.x + dx, face - 0.02, scene.door_sill() + scene.aperture_z])
            })
            .collect();
        // and the open sand in front of it, out of the hero's own shadow
        let on_sand: Vec<_> = [(-2.2f64, 3.0f64), (2.2, 3.5), (-1.4, 5.0), (1.4, 5.5)]
            .iter()
            .filter_map(|(dx, back)| {
                let (x, y) = (door.x + dx, face - back);
                project([x, y, scene.sand_z_at(x, y) + 0.01])
            })
            .collect();

        let sand = mad(&on_sand);
        let doorface = mad(&on_door);
        eprintln!(
            "cove parity {}×{}: sand {} over {} patches, door face {} over {} patches",
            size.0,
            size.1,
            sand.map_or("—".into(), |v| format!("{v:.4}")),
            on_sand.len(),
            doorface.map_or("—".into(), |v| format!("{v:.4}")),
            on_door.len(),
        );

        if let Some(v) = sand {
            anyhow::ensure!(
                v < 0.04,
                "the sunlit sand disagrees by {v:.4}, which is not a shading difference — \
                 check that `bake.rs` still bakes the volume with `sun_direct: false`"
            );
        }
        if let Some(v) = doorface {
            anyhow::ensure!(v < 0.10, "the door's face disagrees by {v:.4}, over the L2 bound");
        }
        anyhow::ensure!(
            sand.is_some() || doorface.is_some(),
            "neither region landed in the frame; the rig's composition has moved"
        );
        Ok(())
    }
}
