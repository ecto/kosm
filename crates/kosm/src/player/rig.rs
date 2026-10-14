//! `Rig`: the camera half of the controller. A [`Lens`] that yields a
//! [`kosm_render::Camera`] and moves like a thing with mass.
//!
//! A follow camera is not a formula for where the eye should be. It is a
//! *body* — one that is pulled toward where the eye should be and has to get
//! there, and whose lateness is the whole of the feel. Snap the eye to the
//! ideal offset every frame and a walk reads as a diagram of a walk; let it
//! trail by a seventh of a second and the same walk has weight.
//!
//! So there are two springs and a clamp:
//!
//! - **the eye**, critically damped, trailing the ideal offset by
//!   [`RigKnobs::follow_lag_s`] of travel;
//! - **the aim**, the same spring a little faster, run in the *subject's*
//!   frame so the framing never lags the body — only the look-ahead eases in
//!   when a walk starts and out when it stops;
//! - **the arm**, which shortens the offset until the eye is [`RigKnobs::clear`]
//!   of anything solid and floors it above [`RigKnobs::min_z`]. The cove's
//!   doorstep framing — the camera stepping to the shoulder as the being
//!   reaches the door — is this clamp plus an [`Interest`], not a second
//!   camera with a blend.
//!
//! And one thing that is not feel at all: the camera that comes out is
//! **quantised to the millimetre**. `sims/rune/game.rs` learned this the hard
//! way. The temporal history compares views for equality, and a view that
//! differs by a float is a *moved* camera: an unquantised spring, which
//! converges but never arrives, would put every pass of a standing player
//! through the reprojection forever. Rounded, a settled camera is exactly
//! equal to itself frame after frame, the plan empties, and the picture
//! converges. A millimetre of eye is a hundredth of a pixel; what it buys is
//! that a still player is *still*.
//!
//! ```
//! use kosm::player::rig::{Rig, Subject};
//! use phyz_math::Vec3;
//!
//! let rig = Rig::default();
//! let dt = 1.0 / 120.0;
//! let at = |y| Subject::walking(Vec3::new(0.0, y, 1.0), Vec3::new(0.0, 1.4, 0.0));
//! // the first frame snaps: a camera has no history to be late against
//! let cam = rig.follow(&at(0.0), 0.0);
//! assert!((cam.fov_deg - 55.6).abs() < 0.02);   // 50° + 4°/(m/s) at a walk
//! for i in 1..=240 {
//!     rig.follow(&at(i as f64 * 1.4 * dt), dt);
//! }
//! // two seconds in, the eye trails the offset it wants by 0.15 s of travel
//! let lag = (rig.eye() - rig.wanted_eye()).norm() / 1.4;
//! assert!((lag - 0.15).abs() < 1e-6);
//! // …and the aim leads the walk by `lookahead_s × speed`
//! assert!((rig.lead() - 0.84).abs() < 1e-3);
//! assert_eq!(rig.shutter_passes(), 3);   // moving: the shutter is open
//! ```

use std::cell::Cell;
use std::sync::Arc;

use phyz_math::{Mat3, Vec3, quat_exp};

use crate::lens::Lens;
use crate::world::World;

// ─── what the rig follows ─────────────────────────────────────────────────

/// The body the camera is about, as the camera needs it.
///
/// Deliberately not a `&Body`: a rig follows anything that has a place, a
/// velocity and a facing — the articulated hero, a capsule, a marble, a
/// scripted point in a test. `Body` produces one of these; so does
/// [`Subject::of_free_body`] for a world whose root is a free joint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Subject {
    /// Where the body is, in metres. The rig frames *this* point.
    pub position: Vec3,
    /// World-frame velocity, metres per second.
    pub velocity: Vec3,
    /// Unit facing. Only its horizontal part is used, so a leaning body
    /// still has a well-defined "behind".
    pub facing: Vec3,
    /// The gait's own speed, metres per second. Usually the horizontal
    /// velocity's norm, but a body that is being carried may differ.
    pub speed: f64,
    /// Lean from upright, radians. Read by the rig only through
    /// [`RigKnobs::lean_up`].
    pub lean: f64,
}

impl Subject {
    /// A body standing at `position`, facing +y.
    pub fn still(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::zeros(),
            facing: Vec3::new(0.0, 1.0, 0.0),
            speed: 0.0,
            lean: 0.0,
        }
    }

    /// A body at `position` moving at `velocity` and facing the way it is
    /// going. A still velocity keeps the +y facing.
    pub fn walking(position: Vec3, velocity: Vec3) -> Self {
        let flat = Vec3::new(velocity.x, velocity.y, 0.0);
        let speed = flat.norm();
        Self {
            position,
            velocity,
            facing: if speed > 1e-9 { flat / speed } else { Vec3::new(0.0, 1.0, 0.0) },
            speed,
            lean: 0.0,
        }
    }

    /// The subject a world's root free joint describes: `q[0..3]` the
    /// exponential coordinates of its rotation, `q[3..6]` its position,
    /// `v[3..6]` its linear velocity **in the body frame**, which is where
    /// phyz keeps it. The facing is the body's own +y column, flattened.
    ///
    /// This is what [`Lens::see`] uses when no subject extractor was given.
    pub fn of_free_body(world: &World) -> Self {
        let (q, v) = (world.q(), world.v());
        let at = |c: &[f64], i: usize| Vec3::new(c[i], c[i + 1], c[i + 2]);
        if q.len() < 6 || v.len() < 6 {
            return Self::still(Vec3::zeros());
        }
        let r: Mat3 = quat_exp(&at(q, 0)).to_matrix();
        let position = at(q, 3);
        let velocity = r.mul_vec(at(v, 3));
        let facing = Vec3::new(r[(0, 1)], r[(1, 1)], r[(2, 1)]);
        let up = r.mul_vec(Vec3::z());
        Self {
            position,
            velocity,
            facing,
            speed: Vec3::new(velocity.x, velocity.y, 0.0).norm(),
            lean: up.z.clamp(-1.0, 1.0).acos(),
        }
    }

    /// The horizontal facing, unit. Falls back to +y for a body that is
    /// looking straight up or down.
    fn flat_facing(&self) -> Vec3 {
        let f = Vec3::new(self.facing.x, self.facing.y, 0.0);
        if f.norm() > 1e-9 { f.normalize() } else { Vec3::new(0.0, 1.0, 0.0) }
    }

    /// The body's own right: `facing × ẑ`, horizontal whatever the lean,
    /// which is what a camera that steps sideways wants.
    fn right(&self) -> Vec3 {
        let r = self.flat_facing().cross(&Vec3::z());
        if r.norm() > 1e-9 { r.normalize() } else { Vec3::new(1.0, 0.0, 0.0) }
    }
}

// ─── the ground the arm reads ─────────────────────────────────────────────

/// Distance to the nearest solid, for the spring arm.
///
/// Named for what the camera wants and not for what the body stands on —
/// [`super::ground::Ground`] is the body's, and it answers a different
/// question (where are the contacts) with different machinery. What an arm
/// needs is one number: how much room is there here.
///
/// A closure and not a `&SdfGrid`, so the rig does not care whether a level
/// has a bake: a plane, a half-space, an analytic cliff and `kosm-scan`'s
/// grid all fit behind one signature. `None` means "no map here", and the arm
/// treats it as open air — beyond the map there is no rock, which is the same
/// reading `kosm_scan::SdfGrid::sample` gives the contact producer.
#[derive(Clone, Default)]
pub struct Clearance(Option<Arc<dyn Fn(Vec3) -> Option<f64> + Send + Sync>>);

impl std::fmt::Debug for Clearance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "Clearance(fn)" } else { "Clearance(open air)" })
    }
}

impl Clearance {
    /// No map: nothing is ever in the way.
    pub fn open_air() -> Self {
        Self(None)
    }

    /// Any distance function, metres, positive outside.
    pub fn from_fn(f: impl Fn(Vec3) -> Option<f64> + Send + Sync + 'static) -> Self {
        Self(Some(Arc::new(f)))
    }

    /// A baked signed-distance grid — the cove's, the lab's, any level with
    /// a `sdf.bin`.
    pub fn sdf(grid: Arc<kosm_scan::SdfGrid>) -> Self {
        Self::from_fn(move |p| grid.sample(p))
    }

    /// The half-space `n · p ≥ d`, as a wall. Handy in tests and for a level
    /// whose only obstacle is one cliff face.
    pub fn half_space(normal: Vec3, offset: f64) -> Self {
        let n = normal.normalize();
        Self::from_fn(move |p| Some(p.dot(&n) - offset))
    }

    /// Solid below `z` — the floor a level with no bake still has, so the arm
    /// does not swing the eye into the ground on a downhill.
    pub fn floor(z: f64) -> Self {
        Self::from_fn(move |p| Some(p.z - z))
    }

    /// Distance to the nearest solid at `p`, or `None` off the map.
    pub fn distance(&self, p: Vec3) -> Option<f64> {
        self.0.as_ref().and_then(|f| f(p))
    }
}

/// A point the aim is drawn toward as the subject gets near it — the cove's
/// aperture, a boss, whatever the level wants looked at.
///
/// The blend is the cove's: `t = clamp((reach − d) / reach, 0, 1)` on the
/// distance from the subject, and the aim uses `√t`, which spends the last
/// fifth of the approach on the composition instead of on the walk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interest {
    /// Where to look, metres.
    pub point: Vec3,
    /// How far out the pull starts, metres.
    pub reach: f64,
}

// ─── the knobs ────────────────────────────────────────────────────────────

/// Every number the rig has. Metres, seconds, degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigKnobs {
    /// How far behind the subject the eye sits, along the facing.
    pub back: f64,
    /// How far above it.
    pub up: f64,
    /// How far to the subject's right — the over-the-shoulder step that
    /// keeps the body off centre.
    pub side: f64,
    /// A constant offset of the aim ahead of the subject, along the facing.
    /// Zero aims at the body; the cove's `cam_ahead_mm` is three and a half
    /// metres.
    pub ahead: f64,
    /// How far above the subject's own point the aim sits.
    pub aim_up: f64,
    /// How much of the eye's height is bought by the subject's lean. Zero by
    /// default: a rig that rises when the body tips is a choice, not a rule.
    pub lean_up: f64,
    /// Seconds of velocity the aim slides ahead by. 0.6 s is about
    /// eight-tenths of a metre at a walk.
    pub lookahead_s: f64,
    /// How late the eye is, in seconds of travel. This is the lag a
    /// constant-velocity walk actually shows: a critically damped tracker
    /// trails a ramp by `2/ω`, so `ω = 2 / follow_lag_s`.
    pub follow_lag_s: f64,
    /// How much faster the aim's spring is than the eye's.
    pub aim_speedup: f64,
    /// Vertical field of view at a standstill, degrees.
    pub fov_base_deg: f64,
    /// Degrees of field of view bought per metre per second.
    pub fov_gain_deg: f64,
    /// How close the eye may come to anything solid, metres.
    pub clear: f64,
    /// The shortest the arm may be, as a fraction of the wanted offset — but
    /// only where taking it costs no clearance. A camera pinned to the body's
    /// own centre is worse than one a little too close to a pebble; a camera
    /// inside the cliff is worse than either.
    pub arm_min: f64,
    /// Samples along the arm. More is a finer wall, not a nearer one.
    pub arm_steps: usize,
    /// A floor under the eye, metres. Over water this is the waterline plus
    /// [`RigKnobs::sea_clear`]; a level with no sea leaves it at −∞.
    pub min_z: f64,
    /// How far over the waterline [`RigKnobs::with_water`] puts the floor.
    pub sea_clear: f64,
    /// How much of the [`Interest`] blend the aim actually takes, 0..=1.
    pub aim_bias: f64,
    /// Render units per metre. One for a level in metres, a thousand for a
    /// level assembled in millimetres — which vcad's are, and the cove is.
    pub units_per_metre: f64,
    /// The grid the eye is rounded onto, in millimetres. Zero turns the
    /// rounding off, which is only ever right for an offline still.
    pub quantum_mm: f64,
    /// …and the grid the field of view is rounded onto, in degrees. A
    /// varying field of view is a moved camera too.
    pub fov_quantum_deg: f64,
    /// The most history passes one frame may integrate while moving.
    pub shutter_max: u32,
    /// The speed at which the shutter is fully open, metres per second.
    pub shutter_speed: f64,
    /// How far a body at rest may drift before the rig follows it, metres.
    /// A standing body breathes, and a camera that followed the breath would
    /// never be still — and a camera that is never still is one the history
    /// and the settle blend can never converge under. Only at rest: a walking
    /// body is followed exactly.
    pub dead_zone: f64,
    /// Below this speed the body is at rest, metres per second: its velocity
    /// buys no look-ahead and no field of view, and [`RigKnobs::dead_zone`]
    /// holds.
    pub rest_speed: f64,
    /// How far a body at rest may lean before the rig follows it, radians.
    pub dead_lean: f64,
}

impl Default for RigKnobs {
    fn default() -> Self {
        Self {
            back: 3.0,
            up: 1.5,
            side: 0.6,
            ahead: 0.0,
            aim_up: 0.0,
            lean_up: 0.0,
            lookahead_s: 0.6,
            follow_lag_s: 0.15,
            aim_speedup: 1.5,
            fov_base_deg: 50.0,
            fov_gain_deg: 4.0,
            clear: 0.4,
            arm_min: 0.15,
            arm_steps: 24,
            min_z: f64::NEG_INFINITY,
            sea_clear: 0.4,
            aim_bias: 1.0,
            units_per_metre: 1.0,
            quantum_mm: 1.0,
            fov_quantum_deg: 0.01,
            shutter_max: 4,
            shutter_speed: 2.6,
            dead_zone: 0.02,
            rest_speed: 0.08,
            dead_lean: 0.5f64.to_radians(),
        }
    }
}

impl RigKnobs {
    /// Read the knobs out of a level's parameters, so a rig honours what the
    /// document already says. `p` is the level's own `parameter_or` — the
    /// cove's is `|n, d| scene.authored.parameter_or(n, d)`.
    ///
    /// The `cam_*_mm` names are the cove's, and they are in millimetres
    /// because vcad's documents are; everything the rig does with them is in
    /// metres. The `cam_*_s`, `cam_*_deg` and `cam_shutter_*` names are the
    /// rig's own and a level that does not set them gets the defaults.
    pub fn from_params(p: impl Fn(&str, f64) -> f64) -> Self {
        let d = Self::default();
        let mm = |v: f64| v / 1000.0;
        Self {
            back: mm(p("cam_back_mm", d.back * 1000.0)),
            up: mm(p("cam_up_mm", d.up * 1000.0)),
            side: mm(p("cam_side_mm", d.side * 1000.0)),
            ahead: mm(p("cam_ahead_mm", d.ahead * 1000.0)),
            aim_up: mm(p("cam_aim_up_mm", d.aim_up * 1000.0)),
            lean_up: mm(p("cam_lean_up_mm", d.lean_up * 1000.0)),
            lookahead_s: p("cam_lookahead_s", d.lookahead_s),
            follow_lag_s: p("cam_lag_s", d.follow_lag_s).max(1e-3),
            aim_speedup: p("cam_aim_speedup", d.aim_speedup).max(1e-3),
            fov_base_deg: p("cam_vfov_deg", d.fov_base_deg),
            fov_gain_deg: p("cam_fov_gain_deg", d.fov_gain_deg),
            clear: mm(p("cam_face_clear_mm", d.clear * 1000.0)),
            arm_min: p("cam_arm_min", d.arm_min),
            arm_steps: p("cam_arm_steps", d.arm_steps as f64).max(2.0) as usize,
            min_z: d.min_z,
            sea_clear: mm(p("cam_sea_clear_mm", d.sea_clear * 1000.0)),
            aim_bias: p("cam_aim_bias", d.aim_bias),
            units_per_metre: p("cam_units_per_m", d.units_per_metre).max(1e-9),
            quantum_mm: p("cam_quantum_mm", d.quantum_mm),
            fov_quantum_deg: p("cam_fov_quantum_deg", d.fov_quantum_deg),
            shutter_max: p("cam_shutter_max", d.shutter_max as f64).max(1.0) as u32,
            shutter_speed: p("cam_shutter_speed", d.shutter_speed).max(1e-3),
            dead_zone: p("cam_dead_zone_mm", 20.0) / 1000.0,
            rest_speed: p("cam_rest_speed", 0.08),
            dead_lean: p("cam_dead_lean_deg", 0.5).to_radians(),
        }
    }

    /// The camera comes out in millimetres, which is what a level assembled
    /// from vcad solids is drawn in.
    pub fn in_millimetres(mut self) -> Self {
        self.units_per_metre = 1000.0;
        self
    }

    /// Floor the eye [`RigKnobs::sea_clear`] over a waterline. The cove's
    /// `cam_sea_clear_mm` over its `sea_z`: wading, the being's centre drops
    /// toward the waterline and the framing's `up` drops with it, and a metre
    /// of that would put the eye under the swell looking at the inside of an
    /// opaque teal surface.
    pub fn with_water(mut self, water_z: f64) -> Self {
        self.min_z = water_z + self.sea_clear;
        self
    }

    /// The eye spring's natural frequency. Critically damped, so this is the
    /// only number it has.
    fn omega(&self) -> f64 {
        2.0 / self.follow_lag_s.max(1e-6)
    }
}

// ─── the rig ──────────────────────────────────────────────────────────────

/// What the springs remember between frames. Copy, so the whole of it lives
/// in one [`Cell`] and [`Lens::see`] can take `&self`.
#[derive(Clone, Copy, Debug)]
struct State {
    eye: Vec3,
    eye_vel: Vec3,
    /// The aim, as an offset from the subject: the spring runs in the
    /// subject's frame so the framing does not lag the body.
    aim_off: Vec3,
    aim_vel: Vec3,
    fov: f64,
    fov_vel: f64,
    want: Vec3,
    time: f64,
    started: bool,
    /// How far the eye has been knocked down, metres, and how fast it is
    /// coming back. See [`Rig::kick`].
    kick: f64,
    kick_vel: f64,
    /// A field-of-view offset, degrees, and its own return. See
    /// [`Rig::pulse_fov`].
    pulse: f64,
    pulse_vel: f64,
    /// Where the rig thinks a resting body is, and how far it thinks it
    /// leans: held until the body leaves [`RigKnobs::dead_zone`].
    anchor: Vec3,
    anchor_lean: f64,
    /// Whether the body was at rest last frame.
    resting: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            eye: Vec3::zeros(),
            eye_vel: Vec3::zeros(),
            aim_off: Vec3::zeros(),
            aim_vel: Vec3::zeros(),
            fov: 0.0,
            fov_vel: 0.0,
            want: Vec3::zeros(),
            time: 0.0,
            started: false,
            kick: 0.0,
            kick_vel: 0.0,
            pulse: 0.0,
            pulse_vel: 0.0,
            anchor: Vec3::zeros(),
            anchor_lean: 0.0,
            resting: false,
        }
    }
}

/// How fast a kick and a pulse come back, rad/s. Fast enough to be a jolt and
/// not a lurch: at 18 the eye is back inside a fifth of a second.
const KICK_OMEGA: f64 = 18.0;

/// How far a landing knocks the eye down, metres per N·s of impulse, and the
/// most it ever does. A drop from half a metre is about 100 N·s on a 30 kg
/// body, which is 30 mm.
const KICK_PER_IMPULSE: f64 = 0.30e-3;
const KICK_MAX: f64 = 0.05;

/// How [`Lens::see`] finds the subject in a world.
type SubjectOf = Arc<dyn Fn(&World) -> Subject + Send + Sync>;

/// The follow camera: knobs, a ground to keep out of, and two springs.
///
/// The springs are the rig's own memory, so a `Rig` is used through `&self`
/// and is not `Sync`. One rig, one camera, one thread that renders it.
pub struct Rig {
    /// Every number. Change them between frames and the springs chase the
    /// new ones rather than jumping.
    pub knobs: RigKnobs,
    /// What the arm keeps the eye out of.
    pub ground: Clearance,
    /// What the aim is drawn toward, when the subject is near it.
    pub interest: Option<Interest>,
    subject_of: Option<SubjectOf>,
    state: Cell<State>,
}

impl Default for Rig {
    fn default() -> Self {
        Self::new(RigKnobs::default())
    }
}

impl std::fmt::Debug for Rig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rig")
            .field("knobs", &self.knobs)
            .field("ground", &self.ground)
            .field("interest", &self.interest)
            .finish()
    }
}

impl Rig {
    /// A rig with these knobs, no ground and no point of interest.
    pub fn new(knobs: RigKnobs) -> Self {
        Self {
            knobs,
            ground: Clearance::open_air(),
            interest: None,
            subject_of: None,
            state: Cell::new(State::default()),
        }
    }

    /// What the arm keeps the eye out of.
    pub fn with_ground(mut self, ground: Clearance) -> Self {
        self.ground = ground;
        self
    }

    /// What the aim is drawn toward.
    pub fn with_interest(mut self, interest: Interest) -> Self {
        self.interest = Some(interest);
        self
    }

    /// How [`Lens::see`] finds the subject in a world. Without one it reads
    /// the root free joint ([`Subject::of_free_body`]).
    pub fn with_subject(
        mut self,
        f: impl Fn(&World) -> Subject + Send + Sync + 'static,
    ) -> Self {
        self.subject_of = Some(Arc::new(f));
        self
    }

    /// Forget the springs. The next frame snaps.
    pub fn reset(&self) {
        self.state.set(State::default());
    }

    /// Advance the springs by `dt` and hand back the camera to render from.
    ///
    /// `dt` of zero — or the first call, whose `dt` is ignored — snaps: a
    /// camera arriving for the first time has no history to be late against,
    /// and a rig that eased in from the origin would open every level with a
    /// swoop nobody asked for.
    pub fn follow(&self, subject: &Subject, dt: f64) -> kosm_render::Camera {
        let k = &self.knobs;
        let mut s = self.state.get();
        let framed = self.framed(subject, &mut s);
        let subject = &framed;

        let want = self.place(subject);
        let aim_want = self.aim_offset(subject);
        let fov_want = k.fov_base_deg + k.fov_gain_deg * subject.speed.max(0.0);

        if !s.started || !(dt > 0.0) {
            if !s.started {
                s = State {
                    kick: s.kick,
                    kick_vel: s.kick_vel,
                    pulse: s.pulse,
                    pulse_vel: s.pulse_vel,
                    eye: want,
                    eye_vel: Vec3::zeros(),
                    aim_off: aim_want,
                    aim_vel: Vec3::zeros(),
                    fov: fov_want,
                    fov_vel: 0.0,
                    want,
                    time: 0.0,
                    started: true,
                    anchor: s.anchor,
                    anchor_lean: s.anchor_lean,
                    resting: s.resting,
                };
                self.state.set(s);
            }
            return self.camera_from(&s, subject);
        }

        // The wanted eye moves with the subject, so the spring is told its
        // velocity: a damper that fought the *world* velocity of a target it
        // could not see would trail by `2/ω + dt/2` and the lag would depend
        // on the frame rate, which is the one thing a feel knob must not.
        let want_vel = (want - s.want) / dt;
        // The spring is advanced *from* the last frame's wanted eye, not to
        // this one's: the state it carries is a frame old, and starting the
        // step at the new target would hand the eye a frame of the future and
        // make the lag depend on the frame rate.
        spring(&mut s.eye, &mut s.eye_vel, s.want, want_vel, k.omega(), dt);
        s.want = want;

        // The aim's spring is in the subject's frame — its target is an
        // offset, and an offset does not run away — so nothing is told a
        // target velocity here and the steady-state look-ahead is the whole
        // `lookahead_s × v`, not what is left of it after a lag.
        spring(
            &mut s.aim_off,
            &mut s.aim_vel,
            aim_want,
            Vec3::zeros(),
            k.omega() * k.aim_speedup,
            dt,
        );

        let mut fov = Vec3::new(s.fov, 0.0, 0.0);
        let mut fov_v = Vec3::new(s.fov_vel, 0.0, 0.0);
        spring(&mut fov, &mut fov_v, Vec3::new(fov_want, 0.0, 0.0), Vec3::zeros(), k.omega(), dt);
        s.fov = fov.x;
        s.fov_vel = fov_v.x;

        // **The kick and the pulse**, both springs back to zero, so the eye is
        // knocked and recovers rather than being animated. `kick` is a landing
        // and `pulse` is the two degrees of field the apex of a jump opens up;
        // neither moves where the camera *is*, only where it is looking from
        // and how wide.
        let mut k = Vec3::new(s.kick, 0.0, 0.0);
        let mut kv = Vec3::new(s.kick_vel, 0.0, 0.0);
        spring(&mut k, &mut kv, Vec3::zeros(), Vec3::zeros(), KICK_OMEGA, dt);
        s.kick = k.x;
        s.kick_vel = kv.x;
        let mut p = Vec3::new(s.pulse, 0.0, 0.0);
        let mut pv = Vec3::new(s.pulse_vel, 0.0, 0.0);
        spring(&mut p, &mut pv, Vec3::zeros(), Vec3::zeros(), KICK_OMEGA, dt);
        s.pulse = p.x;
        s.pulse_vel = pv.x;

        // The clamp is on the *state*, not on the output: an eye that has been
        // pushed out of the rock carries on from where it actually is, the way
        // a body that has hit something does.
        s.eye = self.clamp_eye(subject, s.eye);
        s.time += dt;
        self.state.set(s);
        self.camera_from(&s, subject)
    }

    /// **Knock the eye down.** `impulse` is [`Body::landed`]'s, in N·s; the
    /// spring brings it back inside a fifth of a second.
    ///
    /// [`Body::landed`]: super::body::Body::landed
    pub fn kick(&self, impulse: f64) {
        let mut s = self.state.get();
        s.kick = (s.kick + KICK_PER_IMPULSE * impulse.max(0.0)).min(KICK_MAX);
        self.state.set(s);
    }

    /// **Open the field of view by `degrees`**, springing back. Two at the
    /// apex of a jump is enough to feel and not enough to see as a zoom.
    pub fn pulse_fov(&self, degrees: f64) {
        let mut s = self.state.get();
        s.pulse += degrees;
        self.state.set(s);
    }

    /// How far the eye is currently knocked down, metres, and how many degrees
    /// of field the pulse has added. Reported, for a test.
    pub fn kicked(&self) -> (f64, f64) {
        let s = self.state.get();
        (s.kick, s.pulse)
    }

    /// The camera as it stands, without advancing anything.
    pub fn camera(&self, subject: &Subject) -> kosm_render::Camera {
        self.camera_from(&self.state.get(), subject)
    }

    /// Where the eye actually is, metres. The lag is measured against
    /// [`Rig::wanted_eye`].
    pub fn eye(&self) -> Vec3 {
        self.state.get().eye
    }

    /// Where the eye is *wanted*, metres: the ideal offset, after the arm.
    pub fn wanted_eye(&self) -> Vec3 {
        self.state.get().want
    }

    /// Where the camera is looking, metres.
    pub fn aim(&self, subject: &Subject) -> Vec3 {
        subject.position + self.state.get().aim_off
    }

    /// How far ahead of the subject the aim is, metres, horizontally. Zero
    /// standing still, `lookahead_s × speed` at a steady walk.
    pub fn lead(&self) -> f64 {
        let k = &self.knobs;
        let off = self.state.get().aim_off - Vec3::new(0.0, 0.0, k.aim_up);
        Vec3::new(off.x, off.y, 0.0).norm()
    }

    /// The field of view the spring has reached, degrees, before rounding.
    pub fn fov_deg(&self) -> f64 {
        self.state.get().fov
    }

    /// How many passes of the temporal history one frame should integrate:
    /// the shutter, in the only unit a progressive renderer has.
    ///
    /// One when the eye is still, so a standing player's picture converges
    /// instead of being smeared across its own past; up to
    /// [`RigKnobs::shutter_max`] at [`RigKnobs::shutter_speed`], where the
    /// integrated passes read as the motion blur a moving camera should have.
    /// The render loop consumes it; nothing in `kosm-view` changes.
    ///
    /// It is read off the *eye's* speed, not the subject's — the eye is what
    /// the shutter is attached to, and an eye still catching up after the body
    /// stopped is still moving.
    pub fn shutter_passes(&self) -> u32 {
        let k = &self.knobs;
        let f = (self.state.get().eye_vel.norm() / k.shutter_speed).clamp(0.0, 1.0);
        1 + (f * (k.shutter_max.max(1) - 1) as f64).round() as u32
    }

    // ---- the pieces ------------------------------------------------------

    /// The ideal eye for this subject, before the arm: behind, above, and a
    /// step to the shoulder.
    fn ideal_eye(&self, s: &Subject) -> Vec3 {
        let k = &self.knobs;
        let (f, r) = (s.flat_facing(), s.right());
        s.position - f * k.back + r * k.side + Vec3::z() * (k.up + k.lean_up * s.lean)
    }

    /// The wanted eye: the ideal one, shortened by the arm and floored.
    /// The subject the rig frames: the body's place, not its breath.
    ///
    /// Walking, it is the body exactly. At rest — below
    /// [`RigKnobs::rest_speed`] — the rig holds the place it saw on the first
    /// frame of rest and moves it only as far as the body leaves
    /// [`RigKnobs::dead_zone`], with the lean held the same way, and a resting
    /// body's velocity buys no look-ahead. Snapping on the first frame of rest
    /// keeps where a camera settles exactly where it settled before there was
    /// a zone.
    fn framed(&self, subject: &Subject, s: &mut State) -> Subject {
        let k = &self.knobs;
        let resting = subject.speed < k.rest_speed;
        if !s.started || !resting || !s.resting {
            s.anchor = subject.position;
            s.anchor_lean = subject.lean;
        } else {
            let d = subject.position - s.anchor;
            let n = d.norm();
            if n > k.dead_zone {
                s.anchor = s.anchor + d * ((n - k.dead_zone) / n);
            }
            let dl = subject.lean - s.anchor_lean;
            if dl.abs() > k.dead_lean {
                s.anchor_lean += dl - k.dead_lean * dl.signum();
            }
        }
        s.resting = resting;
        Subject {
            position: s.anchor,
            velocity: if resting { Vec3::zeros() } else { subject.velocity },
            facing: subject.facing,
            speed: if resting { 0.0 } else { subject.speed },
            lean: s.anchor_lean,
        }
    }

    fn place(&self, s: &Subject) -> Vec3 {
        self.clamp_eye(s, self.ideal_eye(s))
    }

    /// Shorten the pivot→eye segment until every sample on it is
    /// [`RigKnobs::clear`] of a solid, then floor the result.
    ///
    /// The order matters and is deliberate: the arm runs first, so what comes
    /// out is a point on a segment from the body every sample of which is
    /// clear; the floor is applied last and only ever *raises* the eye. A
    /// level whose rock sits below its own waterline gets the floor and not
    /// the clearance, which is the honest trade — `min_z` is a rule about
    /// water, and there is no water inside a cliff.
    fn clamp_eye(&self, s: &Subject, want: Vec3) -> Vec3 {
        let k = &self.knobs;
        let pivot = s.position;
        let want = Vec3::new(want.x, want.y, want.z.max(k.min_z));
        let t = self.arm(pivot, want);
        let e = pivot + (want - pivot) * t;
        Vec3::new(e.x, e.y, e.z.max(k.min_z))
    }

    /// The largest fraction of `pivot → want` whose samples all stand
    /// [`RigKnobs::clear`] of a solid. One when the way is open.
    ///
    /// What it promises is the honest thing, not the flattering one: the eye
    /// ends up at least as clear of the rock as **the body itself is**, and
    /// [`RigKnobs::clear`] of it whenever the body has that much room. A
    /// character standing with its shoulder against a wall cannot be filmed
    /// from half a metre off that wall by any arm; the camera comes to the
    /// shoulder and no further.
    ///
    /// That is also why [`RigKnobs::arm_min`] is a floor with a condition on
    /// it. It exists so a camera does not collapse into the back of the head
    /// over a pebble — but where taking it would cost clearance, it is not
    /// taken, because a camera inside the rock is worse than a camera inside
    /// the character.
    fn arm(&self, pivot: Vec3, want: Vec3) -> f64 {
        let k = &self.knobs;
        let at = |p: Vec3| self.ground.distance(p).unwrap_or(f64::INFINITY);
        if self.ground.distance(pivot).is_none() && self.ground.distance(want).is_none() {
            // Off the map at both ends: nothing to walk into. Cheap out
            // before spending `arm_steps` samples saying so.
            return 1.0;
        }
        let n = k.arm_steps.max(2);
        let seg = want - pivot;
        let mut ok = 0.0;
        for i in 1..=n {
            let t = i as f64 / n as f64;
            if at(pivot + seg * t) < k.clear {
                break;
            }
            ok = t;
        }
        let floor = k.arm_min.clamp(0.0, 1.0);
        if ok < floor && at(pivot + seg * floor) >= k.clear.min(at(pivot)) {
            ok = floor;
        }
        ok
    }

    /// The aim, as an offset from the subject: up a little, ahead a little,
    /// and sliding along the horizontal velocity — then pulled toward the
    /// [`Interest`] by how near the subject is to it.
    fn aim_offset(&self, s: &Subject) -> Vec3 {
        let k = &self.knobs;
        let flat = Vec3::new(s.velocity.x, s.velocity.y, 0.0);
        let base =
            Vec3::z() * k.aim_up + s.flat_facing() * k.ahead + flat * k.lookahead_s;
        let Some(i) = self.interest else { return base };
        let reach = i.reach.max(1e-6);
        let d = (i.point - s.position).norm();
        let t = ((reach - d) / reach).clamp(0.0, 1.0);
        // √t, as the cove's doorstep blend does: the last fifth of the walk
        // is where the composition is won, and the aim is what wins it.
        let w = t.sqrt() * k.aim_bias.clamp(0.0, 1.0);
        let to_interest = i.point - s.position;
        base + (to_interest - base) * w
    }

    /// The camera, in render units, rounded onto the grid.
    fn camera_from(&self, s: &State, subject: &Subject) -> kosm_render::Camera {
        let k = &self.knobs;
        let u = k.units_per_metre;
        let p = |v: Vec3| kosm_render::Point3::new(v.x * u, v.y * u, v.z * u);
        // The kick is on the eye alone and not on the aim: the camera dips and
        // keeps looking at the same place, which is what a jolt is.
        let eye = p(s.eye - Vec3::new(0.0, 0.0, s.kick));
        let aim = p(subject.position + s.aim_off);
        // The look, rounded ten metres out: a millimetre there is a tenth of
        // a milliradian, far under what the picture can show.
        let d = aim - eye;
        let dir = if d.norm() > 1e-12 { d.normalize() } else { kosm_render::Vec3::new(0.0, 1.0, 0.0) };
        let far = eye + dir * (10.0 * u);
        let q = |v: kosm_render::Point3| {
            let step = k.quantum_mm * u / 1000.0;
            if step <= 0.0 {
                return v;
            }
            kosm_render::Point3::new(
                (v.x / step).round() * step,
                (v.y / step).round() * step,
                (v.z / step).round() * step,
            )
        };
        let wide = s.fov + s.pulse;
        let fov = if k.fov_quantum_deg > 0.0 {
            (wide / k.fov_quantum_deg).round() * k.fov_quantum_deg
        } else {
            wide
        };
        kosm_render::Camera::look_at(q(eye), q(far), kosm_render::Vec3::new(0.0, 0.0, 1.0), fov)
    }
}

/// A [`Lens`]: the world in, a camera out.
///
/// `dt` comes from the world's own clock, so a rig sees exactly the time the
/// simulation advanced between the two frames it was shown — not the wall
/// clock, and not a nominal frame time the sim may not have hit.
impl Lens for Rig {
    type Out = kosm_render::Camera;

    fn see(&self, world: &World) -> kosm_render::Camera {
        let subject = match &self.subject_of {
            Some(f) => f(world),
            None => Subject::of_free_body(world),
        };
        let now = world.state().time;
        let s = self.state.get();
        let dt = if s.started { (now - s.time).max(0.0) } else { 0.0 };
        let cam = self.follow(&subject, dt);
        let mut s = self.state.get();
        s.time = now;
        self.state.set(s);
        cam
    }
}

/// One step of a critically damped spring toward a target that is itself
/// moving at `target_vel`.
///
/// Exact, not integrated: `y'' + 2ω y' + ω² y = −2ω v_t` in the target's
/// frame has a closed-form solution, so the step is stable at any `dt` and
/// the steady-state lag under a constant-velocity target is exactly `2v_t/ω`
/// — which is what makes [`RigKnobs::follow_lag_s`] a promise and not a hint.
fn spring(x: &mut Vec3, v: &mut Vec3, target: Vec3, target_vel: Vec3, omega: f64, dt: f64) {
    let omega = omega.max(1e-9);
    // The particular solution: where a tracker chasing a ramp settles.
    let yp = -target_vel * (2.0 / omega);
    let h0 = (*x - target) - yp;
    let hd0 = *v - target_vel;
    let c = hd0 + h0 * omega;
    let e = (-omega * dt).exp();
    let h1 = (h0 + c * dt) * e;
    let hd1 = (hd0 - c * (omega * dt)) * e;
    *x = target + target_vel * dt + yp + h1;
    *v = target_vel + hd1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_still_subject_settles_and_stays_put() {
        let rig = Rig::default();
        let s = Subject::still(Vec3::new(1.0, 2.0, 0.5));
        let first = rig.follow(&s, 0.0);
        for _ in 0..200 {
            rig.follow(&s, 1.0 / 60.0);
        }
        let last = rig.follow(&s, 1.0 / 60.0);
        assert_eq!(first.eye, last.eye, "a standing camera drifted");
        assert_eq!(first.fov_deg, last.fov_deg);
    }

    #[test]
    fn the_arm_stops_at_a_wall() {
        // A wall at y = 0; the subject stands a metre in front of it facing
        // away, so the ideal eye is three metres *into* it.
        let k = RigKnobs { side: 0.0, up: 0.0, min_z: f64::NEG_INFINITY, ..Default::default() };
        let rig = Rig::new(k).with_ground(Clearance::half_space(Vec3::new(0.0, -1.0, 0.0), 0.0));
        let s = Subject {
            position: Vec3::new(0.0, -1.0, 0.0),
            velocity: Vec3::zeros(),
            facing: Vec3::new(0.0, -1.0, 0.0),
            speed: 0.0,
            lean: 0.0,
        };
        rig.follow(&s, 0.0);
        let eye = rig.eye();
        assert!(eye.y <= -0.4 + 1e-9, "the eye is {} from the wall", -eye.y);
    }
}
