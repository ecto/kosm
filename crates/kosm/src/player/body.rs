//! `Body`: the physics half of the controller.
//!
//! An articulated rig on one free joint. What the cove's capsule did with one
//! spring, a rig does with the same spring on the root plus PD targets on the
//! joints — so the two are one type with two [`BodySpec`]s, and
//! `sims/rune/being.rs` keeps its numbers while gaining legs.
//!
//! Four controllers, each a small function called from [`Body::step`]:
//!
//! - **upright** — a spring and a damper on the root, critically damped at
//!   [`UPRIGHT_OMEGA`] in the whole body's inertia *about the feet*, targeting
//!   world `ẑ` leaned by the commanded lean. It has to pay gravity back before
//!   it buys any stiffness of its own, because a body standing is an inverted
//!   pendulum: `k = I_p ω² + M g z_com`, `c = 2 I_p ω`. Verbatim from
//!   `being.rs::Being::upright_gains`, generalised to a composite inertia.
//! - **lean to move** — the velocity error sets a *target lean*, capped at
//!   [`LEAN_MAX`], and the upright controller drives the root to it. See
//!   [`DRIVE_ASSIST`] for what that is honestly worth and what carries the
//!   rest.
//! - **velocity curves** — a proportional controller toward `speed × input`,
//!   [`TAU_ACCEL`] into a walk and [`TAU_STOP`] out of one, [`WALK`] and
//!   [`RUN`] m/s, and a diagonal that is not faster than a straight.
//! - **gait** ([`super::gait`]) and **the tool socket** ([`super::tool`]).
//!
//! The whole thing is also a [`crate::step::Step`] over a [`World`], with the
//! [`Drive`] packed into an [`Action`] — see [`BodyStep`] — so a policy drives
//! the same body a player does.
//!
//! Metres, radians, seconds; z up. The body's own frame is `+x` forward,
//! `+y` left, `+z` up.

use std::f64::consts::PI;
use std::sync::Arc;

use phyz_contact::{ContactCache, ContactMaterial};
use phyz_math::{GRAVITY, Mat3, Quat, SpatialInertia, SpatialTransform, Vec3, quat_exp, quat_log};
use phyz_model::{GeomInstance, Geometry, Model, ModelBuilder, State};
use phyz_rigid::forward_kinematics;

use super::gait::Gait;
use super::ground::{Ground, step_on};
use super::medium::{Immersed, Medium};
use super::tool::{Pose, Tool};
use crate::step::{Action, Step};
use crate::world::{Param, World};

// ---- the numbers -----------------------------------------------------------

/// How fast the upright spring brings the body back, rad/s.
///
/// Critically damped, so the free response to a shove is
/// `θ(t) = θ₀ (1 + ω t) e^{−ω t}`: it never crosses zero, so there is nothing
/// to tune away, and it is inside a twentieth of the shove by `ω t ≈ 4.7`. At
/// 6 rad/s that is 0.8 s from 20° to under 1° — slow enough to *see*.
pub const UPRIGHT_OMEGA: f64 = 6.0;

/// The largest lean the drive will ask for, radians.
pub const LEAN_MAX: f64 = 12.0 * PI / 180.0;

/// The largest lean the *player* may add on top, radians. The cove's puzzle
/// knob: enough to walk a caustic across a door, not enough to fall over.
pub const TILT_MAX: f64 = 10.0 * PI / 180.0;

/// Into a walk, seconds.
pub const TAU_ACCEL: f64 = 0.25;

/// Out of one, seconds.
pub const TAU_STOP: f64 = 0.35;

/// Walking, m/s.
pub const WALK: f64 = 1.4;

/// Running, m/s. Shift.
pub const RUN: f64 = 2.6;

/// How much of the commanded acceleration is asked for as a *force*, on top
/// of the lean — and why it is one, not a share.
///
/// The design says input sets a lean and the ground reaction does the
/// accelerating. That is true of a body that *steps*: leaning puts the centre
/// of mass past the base of support, the body falls, and the swing leg catches
/// it. It is only partly true here, and the honest account is what the
/// measurement says.
///
/// A rigid body held upright by a torque against the inertial frame — which is
/// what a spring on a free joint is — accelerates its centre of mass only
/// while it is *turning*: the contact is the pivot, so `a_com = −α × r`, and a
/// body sitting at a steady lean has `α = 0` and therefore no horizontal force
/// at all. So the lean buys a real impulse as the body tips — the centre of
/// mass swings forward by `h sin θ` — and then nothing. It cannot hold a
/// speed. The force holds the speed.
///
/// **Measured** (`tests/player.rs::how_much_of_the_start_the_lean_is_worth`,
/// which prints it): a quarter of a second into a walk from standing, the
/// controller as it ships has the cove's being doing 0.879 m/s, and the lean
/// on its own — the same body, tilted to the cap with no drive force at all —
/// does 0.250 m/s. **The lean is 28 % of the start and the force is 72 %.**
///
/// So the lean is not a *share* of the acceleration, it is the lean a body
/// accelerating at `a` would have (`tan θ = a/g`, capped at [`LEAN_MAX`],
/// slewed in over [`TAU_LEAN`]) — and the force is the whole of what the time
/// constant asks for. That is what this constant being one means. Below one
/// the body still walks and the rise time stretches; at zero it tips, lurches
/// and stops.
pub const DRIVE_ASSIST: f64 = 1.0;

/// How fast a joint PD answers, rad/s. Limbs are light next to the trunk, so
/// this is stiff next to [`UPRIGHT_OMEGA`] and still a hundred steps of a
/// millisecond solver.
pub const JOINT_OMEGA: f64 = 25.0;

/// How many times over a joint beats the gravity that would fold it.
///
/// A leg joint sees the ground reaction as a negative spring of `M g L`. Four
/// is enough that the sag is a couple of degrees and the equilibrium is
/// unmistakably stable, and small enough that the rotor
/// ([`JOINT_ARMATURE`]) it needs is still a rotor and not most of the limb.
pub const JOINT_SUPPORT: f64 = 24.0;

/// The rotor inertia a joint's PD is given, as a multiple of `k dt²`.
///
/// MuJoCo's `armature`, and the one number in this file that is numerical
/// rather than physical. An explicit step on a spring of stiffness `k` and
/// inertia `I` is stable while `ω dt = dt sqrt(k/I) < 2`; adding `a = n k dt²`
/// puts a floor under `I` and therefore a ceiling of `1/sqrt(n)` on `ω dt`,
/// whatever the limb weighs. Fifty is `ω dt ≤ 0.14`, which leaves the joint
/// answering in milliseconds and the solver comfortable. Without it a
/// sixteen-gram-metre-squared boot asked to hold up a hundred kilograms is a
/// spring at 2000 rad/s and the figure explodes in thirty steps — measured,
/// and the reason this constant exists.
pub const JOINT_ARMATURE: f64 = 50.0;

/// The steepest ground the controller holds a body on outright, radians.
pub const SLOPE_HOLD: f64 = 35.0 * PI / 180.0;

/// The shallowest ground it holds a body on not at all, radians. Past this a
/// slope is something to slide off, which is what makes a boulder an edge.
pub const SLOPE_SLIP: f64 = 50.0 * PI / 180.0;

/// How long the lean takes to arrive, seconds. See [`Body::control`].
pub const TAU_LEAN: f64 = 0.15;

/// How far past critical a joint PD is damped.
///
/// A limb is not a thing anyone wants to watch ring. Critical is 1; a little
/// past it costs a few milliseconds of answer and buys a figure that stands
/// still instead of shivering.
pub const JOINT_ZETA: f64 = 1.5;

/// The speed, m/s, at which an *uncommanded* body is unmistakably sliding
/// rather than creeping — and so the speed at which the friction compensation
/// is worth all of `μ N`. See [`Consts::drive`].
pub const SLIDING: f64 = 0.15;

// ---- the description -------------------------------------------------------

/// A lump of stuff: a capsule between two points, or a ball when they are the
/// same point. Rest-pose, body-local metres.
#[derive(Clone, Copy, Debug)]
pub struct Lump {
    pub from: Vec3,
    pub to: Vec3,
    pub radius: f64,
    /// kg/m³, out of `kosm::material`.
    pub density: f64,
}

impl Lump {
    pub fn ball(at: Vec3, radius: f64, density: f64) -> Self {
        Self { from: at, to: at, radius, density }
    }

    pub fn bone(from: Vec3, to: Vec3, radius: f64, density: f64) -> Self {
        Self { from, to, radius, density }
    }

    /// The barrel's length: the distance between the cap centres.
    pub fn length(&self) -> f64 {
        (self.to - self.from).norm()
    }

    pub fn centre(&self) -> Vec3 {
        (self.from + self.to) * 0.5
    }

    pub fn mass(&self) -> f64 {
        let (r, l) = (self.radius, self.length());
        self.density * super::medium::capsule_volume(r, l)
    }

    /// The shape, and where it sits relative to a body frame at `origin`.
    pub fn geometry(&self, origin: Vec3) -> GeomInstance {
        let (r, l) = (self.radius, self.length());
        let geometry = if l < 1e-9 { Geometry::Sphere { radius: r } } else { Geometry::Capsule { radius: r, length: l } };
        let rot = if l < 1e-9 { Mat3::identity() } else { frame_from_z(self.to - self.from) };
        // `GeomInstance::origin.rot` is body → shape, the transpose of the
        // shape's own axes in the body frame.
        GeomInstance { name: None, origin: SpatialTransform::new(rot.transpose(), self.centre() - origin), geometry }
    }

    /// Mass and inertia about its own centre, body-local axes.
    ///
    /// The cylinder is the textbook `m(3r² + L²)/12`; a cap is a hemisphere
    /// whose own centre of mass sits `3r/8` past the cap centre, so shifting
    /// its `2/5 m r²` out to the capsule's centre through its own centre of
    /// mass leaves `m(2r²/5 + L²/4 + 3Lr/8)`. Byte for byte what
    /// `being.rs::Being::new` computes, which is why the capsule spec keeps
    /// the cove's mass and its spring.
    pub fn inertia(&self) -> (f64, Mat3) {
        let (r, l, rho) = (self.radius, self.length(), self.density);
        let (m_cyl, m_cap) = (rho * PI * r * r * l, rho * 2.0 / 3.0 * PI * r * r * r);
        let mass = m_cyl + 2.0 * m_cap;
        let i_t = m_cyl * (3.0 * r * r + l * l) / 12.0 + 2.0 * m_cap * (0.4 * r * r + l * l / 4.0 + 3.0 * l * r / 8.0);
        let i_a = 0.5 * m_cyl * r * r + 2.0 * 0.4 * m_cap * r * r;
        if l < 1e-9 {
            return (mass, Mat3::from_diagonal(&Vec3::new(i_a, i_a, i_a)));
        }
        let u = (self.to - self.from).normalize();
        let uu = outer(u, u);
        // transversely isotropic: `i_t (I − uuᵀ) + i_a uuᵀ`
        let tensor = mat_add(mat_scale(mat_sub(Mat3::identity(), uu), i_t), mat_scale(uu, i_a));
        (mass, tensor)
    }
}

/// What a joint is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PivotKind {
    /// A hinge about an axis, body-local at rest.
    Revolute(Vec3),
    /// A ball joint.
    Spherical,
}

/// One link of the rig: a joint, and the stuff that hangs off it.
#[derive(Clone, Debug)]
pub struct Link {
    pub name: String,
    /// The joint's origin, rest-pose body-local metres. Also this link's own
    /// body frame, so nothing downstream has to carry a second offset.
    pub pivot: Vec3,
    pub kind: PivotKind,
    /// The link this one hangs off, or `None` for the root.
    pub parent: Option<usize>,
    /// The stuff: what it weighs and what a picture draws.
    pub lumps: Vec<Lump>,
    /// What the contact solver sees. Empty on everything but the feet — and
    /// the root, for the capsule spec.
    pub collision: Vec<Lump>,
    /// A revolute joint's soft limits, radians.
    pub limits: Option<[f64; 2]>,
}

impl Link {
    pub fn new(name: impl Into<String>, pivot: Vec3, kind: PivotKind, parent: Option<usize>) -> Self {
        Self { name: name.into(), pivot, kind, parent, lumps: Vec::new(), collision: Vec::new(), limits: None }
    }

    pub fn with(mut self, lump: Lump) -> Self {
        self.lumps.push(lump);
        self
    }

    pub fn colliding(mut self, lump: Lump) -> Self {
        self.collision.push(lump);
        self
    }

    pub fn limited(mut self, lo: f64, hi: f64) -> Self {
        self.limits = Some([lo, hi]);
        self
    }
}

/// A foot: where the body meets the ground.
#[derive(Clone, Copy, Debug)]
pub struct Foot {
    /// Which link it is on.
    pub link: usize,
    /// Its centre, rest-pose body-local metres.
    pub centre: Vec3,
    pub radius: f64,
}

impl Foot {
    /// The lowest point of the foot at rest: the pivot the upright spring
    /// turns the whole body about.
    pub fn contact_z(&self) -> f64 {
        self.centre.z - self.radius
    }
}

/// One leg, for the gait.
#[derive(Clone, Copy, Debug)]
pub struct Leg {
    pub hip: usize,
    pub knee: usize,
    /// 0 is the body's right, 1 its left.
    pub side: usize,
}

/// One arm, for the tool socket.
#[derive(Clone, Copy, Debug)]
pub struct Arm {
    pub shoulder: usize,
    pub elbow: usize,
    /// The two link lengths, metres.
    pub upper: f64,
    pub lower: f64,
    /// Where the hand is at rest, body-local. The IK target, and where a tool
    /// is bolted to the forearm link.
    pub hand: Vec3,
    /// Which way the elbow points: out and back.
    pub hint: Vec3,
}

/// Everything a `Body` is built from.
#[derive(Clone, Debug)]
pub struct BodySpec {
    pub name: String,
    /// `links[0]` is the root and is on the free joint.
    pub links: Vec<Link>,
    pub feet: Vec<Foot>,
    pub legs: Vec<Leg>,
    pub arm: Option<Arm>,
    /// The body's own contact friction. A rolling sphere needs grip and a
    /// walking body needs to slide.
    pub friction: f64,
    pub upright_omega: f64,
    /// Whether the root's own rotation carries the facing.
    ///
    /// A capsule has no yaw to speak of, and `being.rs` keeps its heading as a
    /// number with the spring working on the *axis* alone; a figure with legs
    /// has to actually turn. `false` reproduces the capsule bit for bit.
    pub yaws: bool,
    pub walk: f64,
    pub run: f64,
    /// Hip to ankle, metres: what the gait's pendulum frequency comes from.
    pub leg_length: f64,
    /// The capsule the optics still see, `(radius, height)`, when there is
    /// one. `sims/rune`'s caustic is traced through a capsule of glass; until
    /// the refractor in the scene *is* the held lens, the score is read off
    /// this and not off the figure.
    pub capsule: Option<(f64, f64)>,
    /// How fast a joint PD answers, rad/s. Zero leaves the limbs limp.
    pub joint_omega: f64,
    pub dt: f64,
}

// ---- the two constructors --------------------------------------------------

impl BodySpec {
    /// A capsule of `radius` by `height`, filled with `density`.
    ///
    /// The cove's being, exactly: one free joint, the capsule as both the
    /// picture and the contact, the heading a number, and the spring derived
    /// from the same inertia `being.rs` derives it from. `tests/player.rs`
    /// holds it to the mass and the gains the cove reports.
    pub fn capsule(radius: f64, height: f64, density: f64) -> Self {
        let half = ((height - 2.0 * radius) / 2.0).max(1e-6);
        let barrel = Lump::bone(Vec3::new(0.0, 0.0, -half), Vec3::new(0.0, 0.0, half), radius, density);
        let root = Link::new("body", Vec3::zeros(), PivotKind::Spherical, None).with(barrel).colliding(barrel);
        Self {
            name: "capsule".into(),
            links: vec![root],
            feet: vec![Foot { link: 0, centre: Vec3::new(0.0, 0.0, -half), radius }],
            legs: Vec::new(),
            arm: None,
            friction: 0.35,
            upright_omega: UPRIGHT_OMEGA,
            yaws: false,
            walk: WALK,
            run: RUN,
            leg_length: height / 2.0,
            capsule: Some((radius, height)),
            joint_omega: JOINT_OMEGA,
            dt: 1e-3,
        }
    }

    /// The hero rig: every hinge a joint, every part a substance, the feet the
    /// only thing that touches the ground.
    ///
    /// Takes a [`Skeleton`] and not `sims/rune/hero/figure.rs`'s `Rig`,
    /// because `kosm` is under the sims and cannot see one. The `Rig` fills a
    /// `Skeleton` out of `Rig::pivots()` in one function — see
    /// `sims/rune/being.rs::hero_skeleton` — so there is still exactly one
    /// statement of where each joint is.
    pub fn hero(s: &Skeleton) -> Self {
        // The root is the pelvis: the point everything above the hips turns
        // about, and the point everything below them hangs from.
        let root_at = s.torso;
        let mut links = vec![
            Link::new("pelvis", root_at, PivotKind::Spherical, None)
                .with(Lump::ball(s.skirt, s.skirt_r, s.torso_density))
        ];
        // Leaning forward tips `+z` toward `+x`, which is a rotation about
        // `−y` — the body's own *right*. Every sagittal hinge in the figure
        // turns about one of the two signs of `ŷ` and nothing turns about `x̂`,
        // which is the mistake this line exists to not make again.
        let nod = -Vec3::y();
        let torso = push(&mut links, Link::new("torso", s.torso, PivotKind::Revolute(nod), Some(0)).limited(-0.6, 0.6).with(Lump::ball(s.chest, s.chest_r, s.torso_density)));
        push(&mut links, Link::new("neck", s.neck, PivotKind::Revolute(nod), Some(torso)).limited(-0.6, 0.6).with(Lump::ball(s.head, s.head_r, s.head_density)));

        let mut legs = Vec::new();
        let mut feet = Vec::new();
        for side in 0..2 {
            let hip = push(&mut links, Link::new(name("thigh", side), s.hip[side], PivotKind::Spherical, Some(0)).with(Lump::bone(s.hip[side], s.knee[side], s.leg_r, s.leg_density)));
            // A knee bends *backward*: about `+y`, which takes the shin's
            // `−z` toward `+x`… and therefore the ankle toward `−x`, behind
            // the body. Its rest pose is already `bend` off straight — the
            // hero's legs are short and its knees are bent — so the range
            // runs from straight (`−bend`) to folded.
            let rest = bend(s.hip[side], s.knee[side], s.ankle[side]);
            let knee = push(&mut links, Link::new(name("shin", side), s.knee[side], PivotKind::Revolute(Vec3::y()), Some(hip)).limited(-rest, 2.2 - rest).with(Lump::bone(s.knee[side], s.ankle[side], s.leg_r * 0.92, s.leg_density)));
            // The boot is a ball and a toe, and both of them touch. A figure
            // on two *points* has nothing holding it in pitch but its ankles;
            // a figure on two feet has a base of support, which is what
            // standing is. The toe is `figure.rs`'s own second sphere, sat so
            // that its underside is on the same ground the ball's is.
            let boot_at = Vec3::new(s.ankle[side].x, s.ankle[side].y, s.boot_r);
            let toe_r = s.boot_r * 0.72;
            let toe_at = Vec3::new(s.ankle[side].x + 0.9 * s.boot_r, s.ankle[side].y, toe_r);
            let ankle = push(
                &mut links,
                Link::new(name("boot", side), s.ankle[side], PivotKind::Spherical, Some(knee))
                    .with(Lump::ball(boot_at, s.boot_r, s.boot_density))
                    .with(Lump::ball(toe_at, toe_r, s.boot_density))
                    .colliding(Lump::ball(boot_at, s.boot_r, s.boot_density))
                    .colliding(Lump::ball(toe_at, toe_r, s.boot_density)),
            );
            legs.push(Leg { hip, knee, side });
            feet.push(Foot { link: ankle, centre: boot_at, radius: s.boot_r });
        }

        // The right arm carries the tool; the left is along for the ride.
        let mut arm = None;
        for side in 0..2 {
            let shoulder = push(&mut links, Link::new(name("arm", side), s.shoulder[side], PivotKind::Spherical, Some(torso)).with(Lump::bone(s.shoulder[side], s.elbow[side], s.arm_r, s.arm_density)));
            // The same for the elbow, whose hinge is what the two-link solve
            // left behind and whose rest pose is a hundred degrees off
            // straight: a range starting at 0 would forbid *reaching*, which
            // is what it did — the hand stopped 185 mm short of every target
            // because the arm was not allowed to straighten.
            let rest = bend(s.shoulder[side], s.elbow[side], s.hand[side]);
            let elbow = push(
                &mut links,
                Link::new(name("forearm", side), s.elbow[side], PivotKind::Revolute(elbow_axis(s, side)), Some(shoulder))
                    .limited(-rest, 2.6 - rest)
                    .with(Lump::bone(s.elbow[side], s.hand[side], s.arm_r * 0.9, s.arm_density))
                    .with(Lump::ball(s.hand[side], s.hand_r, s.arm_density)),
            );
            if side == 0 {
                arm = Some(Arm {
                    shoulder,
                    elbow,
                    upper: (s.elbow[0] - s.shoulder[0]).norm(),
                    lower: (s.hand[0] - s.elbow[0]).norm(),
                    hand: s.hand[0],
                    hint: Vec3::new(1.0, -0.34, -0.16),
                });
            }
        }

        Self {
            name: "hero".into(),
            links,
            feet,
            legs,
            arm,
            friction: 0.35,
            upright_omega: UPRIGHT_OMEGA,
            yaws: true,
            walk: WALK,
            run: RUN,
            leg_length: (s.hip[0].z - s.ankle[0].z).abs().max(1e-3),
            capsule: s.capsule,
            joint_omega: JOINT_OMEGA,
            dt: 1e-3,
        }
    }

    /// The capsule the optics see, when there is one.
    pub fn with_capsule(mut self, capsule: Option<(f64, f64)>) -> Self {
        self.capsule = capsule;
        self
    }

    pub fn with_friction(mut self, friction: f64) -> Self {
        self.friction = friction;
        self
    }

    pub fn with_speeds(mut self, walk: f64, run: f64) -> Self {
        self.walk = walk;
        self.run = run;
        self
    }

    pub fn with_joint_omega(mut self, omega: f64) -> Self {
        self.joint_omega = omega;
        self
    }

    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }
}

fn push(links: &mut Vec<Link>, link: Link) -> usize {
    links.push(link);
    links.len() - 1
}

fn name(part: &str, side: usize) -> String {
    format!("{part}_{}", if side == 0 { "r" } else { "l" })
}

/// The elbow's hinge axis: what a two-link solve leaves behind. `figure.rs`
/// writes it `(hand − elbow) × (elbow − shoulder)`; here the cross product is
/// the other way round, so that a **positive** angle *bends* the arm and the
/// joint's range reads `[straight, folded]` like a range should.
fn elbow_axis(s: &Skeleton, side: usize) -> Vec3 {
    let n = (s.elbow[side] - s.shoulder[side]).cross(s.hand[side] - s.elbow[side]);
    if n.norm() < 1e-9 { Vec3::y() } else { n.normalize() }
}

/// How far off straight a three-point chain sits, radians. Zero is a straight
/// limb; `π` would be folded shut.
fn bend(root: Vec3, middle: Vec3, tip: Vec3) -> f64 {
    let (u, l) = ((middle - root).try_normalize(), (tip - middle).try_normalize());
    match (u, l) {
        (Some(u), Some(l)) => u.dot(l).clamp(-1.0, 1.0).acos(),
        _ => 0.0,
    }
}

/// A rig, as plain data, in the body's own frame: `+x` forward, `+y` left,
/// `+z` up, the origin on the ground between the feet, metres.
///
/// `sims/rune/hero/figure.rs`'s `Rig::pivots()` is millimetres with `+y`
/// forward; one function in `being.rs` turns one into the other, and that is
/// the only place the two conventions meet.
#[derive(Clone, Copy, Debug)]
pub struct Skeleton {
    pub torso: Vec3,
    pub neck: Vec3,
    pub chest: Vec3,
    pub head: Vec3,
    pub skirt: Vec3,
    pub hip: [Vec3; 2],
    pub knee: [Vec3; 2],
    pub ankle: [Vec3; 2],
    pub shoulder: [Vec3; 2],
    pub elbow: [Vec3; 2],
    pub hand: [Vec3; 2],
    pub leg_r: f64,
    pub boot_r: f64,
    pub arm_r: f64,
    pub hand_r: f64,
    pub chest_r: f64,
    pub head_r: f64,
    pub skirt_r: f64,
    pub leg_density: f64,
    pub boot_density: f64,
    pub arm_density: f64,
    pub torso_density: f64,
    pub head_density: f64,
    /// The capsule the optics still see, if the level has one.
    pub capsule: Option<(f64, f64)>,
}

// ---- what the player asks for ----------------------------------------------

/// What the player — or a policy — is asking for on one step.
///
/// `forward` and `strafe` are in `-1..=1` and read as a held direction, not as
/// a speed; the deltas are this step's mouse, in radians; `aim` is the world
/// point the hand is asked for.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Drive {
    pub forward: f64,
    pub strafe: f64,
    pub run: bool,
    pub yaw_delta: f64,
    pub lean_delta: f64,
    pub aim: Option<Vec3>,
}

impl Drive {
    /// Hands off the controls.
    pub const STILL: Self = Self { forward: 0.0, strafe: 0.0, run: false, yaw_delta: 0.0, lean_delta: 0.0, aim: None };

    /// Walking along the facing and nothing else.
    pub fn walking(forward: f64) -> Self {
        Self { forward, ..Self::STILL }
    }

    /// Running along the facing.
    pub fn running(forward: f64) -> Self {
        Self { forward, run: true, ..Self::STILL }
    }

    /// As an [`Action`], so a policy drives the same body: nine numbers,
    /// `[forward, strafe, run, yaw, lean, aim_x, aim_y, aim_z, aiming]`.
    pub fn action(&self) -> Action {
        let a = self.aim.unwrap_or_else(Vec3::zeros);
        Action::new(vec![
            self.forward,
            self.strafe,
            if self.run { 1.0 } else { 0.0 },
            self.yaw_delta,
            self.lean_delta,
            a.x,
            a.y,
            a.z,
            if self.aim.is_some() { 1.0 } else { 0.0 },
        ])
    }

    /// The other way round. An empty action is [`Drive::STILL`].
    pub fn from_action(action: &Action) -> Self {
        let c = &action.ctrl;
        let at = |i: usize| c.get(i).copied().unwrap_or(0.0);
        Self {
            forward: at(0),
            strafe: at(1),
            run: at(2) > 0.5,
            yaw_delta: at(3),
            lean_delta: at(4),
            aim: (at(8) > 0.5).then(|| Vec3::new(at(5), at(6), at(7))),
        }
    }
}

// ---- what comes back out ---------------------------------------------------

/// One part of the body, placed.
#[derive(Clone, Debug)]
pub struct Part {
    pub name: String,
    /// World position of the link's own frame, and its **body → world**
    /// rotation (the renderer's way round, not phyz's).
    pub pose: Pose,
}

/// The body at one instant, as the picture and a scorer want it.
///
/// Nothing here needs the simulation to still be alive.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub t: f64,
    pub dt: f64,
    /// The root's origin, and its **world → body** rotation — phyz's
    /// convention, and `being.rs::Snapshot::being`'s, so the cove's renderer
    /// reads it unchanged.
    pub root: (Vec3, Mat3),
    /// Linear and angular velocity of the root in **world** axes.
    pub root_vel: (Vec3, Vec3),
    pub facing: f64,
    /// How far the body actually is off vertical, radians.
    pub lean: f64,
    /// The lean the player is *asking* for along the facing, radians.
    pub tilt: f64,
    pub gait_phase: f64,
    /// Every link, placed.
    pub parts: Vec<Part>,
    /// Where the held tool is, if there is one.
    pub held: Option<Pose>,
}

// ---- the derived numbers ---------------------------------------------------

/// Everything the controllers need that comes out of the spec once.
#[derive(Clone, Debug)]
pub struct Consts {
    pub mass: f64,
    /// Where the centre of mass is at rest, body-local.
    pub com: Vec3,
    /// How far the centre of mass is above the feet, metres.
    pub com_height: f64,
    /// How high the root sits above the feet at rest, metres.
    pub foot_drop: f64,
    /// The whole body's transverse inertia about the feet, kg·m².
    pub i_pivot: f64,
    /// The upright spring, N·m/rad and N·m·s/rad.
    pub k: f64,
    pub c: f64,
    /// One `(k, c)` per joint, indexed by link.
    pub joint: Vec<(f64, f64)>,
    /// Every link at or under each link, walked once so a per-step gravity
    /// compensation does not walk the tree again.
    pub subtrees: Vec<Vec<usize>>,
}

impl Consts {
    /// The stiffness and the damping the whole body's inertia asks for.
    ///
    /// Standing, a body is an inverted pendulum about its feet: inertia
    /// `I_p` about that pivot, and gravity is a *negative* spring of
    /// `M g z_com` about it, because a body tipped past vertical keeps going.
    /// So the spring pays gravity back before it buys any stiffness of its
    /// own, and a critically damped recovery at `ω` is
    /// `k = I_p ω² + M g z_com`, `c = 2 I_p ω`.
    fn of(spec: &BodySpec) -> Self {
        let pivot_z = spec.feet.iter().map(Foot::contact_z).fold(f64::INFINITY, f64::min);
        let pivot_z = if pivot_z.is_finite() { pivot_z } else { 0.0 };

        let mut mass = 0.0;
        let mut moment = Vec3::zeros();
        let mut lumps: Vec<(f64, Vec3, Mat3)> = Vec::new();
        for link in &spec.links {
            for lump in &link.lumps {
                let (m, i) = lump.inertia();
                mass += m;
                moment += lump.centre() * m;
                lumps.push((m, lump.centre(), i));
            }
        }
        let com = if mass > 0.0 { moment * (1.0 / mass) } else { Vec3::zeros() };
        // Transverse inertia about the feet: `Σ (I_xx + m d²)` where `d` is
        // the distance from the pivot in the plane the body tips in.
        let pivot = Vec3::new(com.x, com.y, pivot_z);
        let mut i_pivot = 0.0;
        for (m, c, i) in &lumps {
            let d = *c - pivot;
            i_pivot += i.get(0, 0) + m * (d.y * d.y + d.z * d.z);
        }
        let com_height = (com.z - pivot_z).max(1e-6);
        let omega = spec.upright_omega;
        let (k, c) = (i_pivot * omega * omega + mass * GRAVITY * com_height, 2.0 * i_pivot * omega);

        // Per joint, two rules and the larger of them.
        //
        // **The load.** A joint on a leg does not hold its own limb up; it
        // holds the *ground reaction*, which is the whole body's weight, on
        // whatever lever the foot below it is on. Tip that reaction by θ and
        // the joint sees `M g L sin θ` — gravity as a negative spring of
        // `M g L`, exactly as the upright spring sees it — so a joint softer
        // than `M g L` does not sag, it *folds*, and no amount of damping
        // saves it. [`JOINT_SUPPORT`] is how many times over it wins. Off a
        // leg, the load is the subtree's own weight on its own lever.
        //
        // **The answer.** However light the limb, it should still answer in
        // the time [`BodySpec::joint_omega`] asks for, so the second rule is
        // the limb's own inertia at that frequency.
        //
        // Then the damping is critical in the inertia the *solver* sees,
        // which is the limb's own plus [`JOINT_ARMATURE`]'s rotor — the one
        // number here that is numerical rather than physical, and the reason
        // a stiff PD on a 16 g·m² boot does not explode at a millisecond.
        let feet: Vec<Vec3> = spec.feet.iter().map(|f| Vec3::new(f.centre.x, f.centre.y, f.contact_z())).collect();
        let joint = (0..spec.links.len())
            .map(|i| {
                let pivot = spec.links[i].pivot;
                let mut own = 0.0f64;
                for lump in &spec.links[i].lumps {
                    let (m, tensor) = lump.inertia();
                    let d = lump.centre() - pivot;
                    own += tensor.trace() / 3.0 + m * d.norm_sq();
                }
                let own = own.max(1e-6);
                let mine = subtree(spec, i);
                let underfoot = spec.feet.iter().enumerate().find(|(_, f)| mine.contains(&f.link));
                let (load, lever) = match underfoot {
                    // On a leg: the whole body, on the lever the foot is on.
                    Some((k, _)) => (mass, (feet[k] - pivot).norm()),
                    None => {
                        let (mut m_sub, mut moment) = (0.0, Vec3::zeros());
                        for j in &mine {
                            for lump in &spec.links[*j].lumps {
                                let m = lump.mass();
                                m_sub += m;
                                moment += lump.centre() * m;
                            }
                        }
                        let com = if m_sub > 0.0 { moment * (1.0 / m_sub) } else { pivot };
                        (m_sub, (com - pivot).norm())
                    }
                };
                let w = spec.joint_omega;
                let k = (JOINT_SUPPORT * load * GRAVITY * lever).max(own * w * w).max(1e-6);
                let armature = JOINT_ARMATURE * k * spec.dt * spec.dt;
                (k, 2.0 * JOINT_ZETA * (k * (own + armature)).sqrt())
            })
            .collect();

        let subtrees = (0..spec.links.len()).map(|i| subtree(spec, i)).collect();
        Self { mass, com, com_height, foot_drop: spec.links[0].pivot.z - pivot_z, i_pivot, k, c, joint, subtrees }
    }

    /// The horizontal drive force: the velocity controller's share, plus the
    /// Coulomb friction the sand is taking out while the body slides.
    ///
    /// The feed-forward is what keeps the curves the player feels the
    /// *controller's*, rather than the sand's. A proportional controller
    /// alone stalls: on 0.35 of friction the ground asks for 3.4 m/s² before
    /// anything moves at all, so `a = e/τ` runs out of authority at 0.55 m/s
    /// and the body never reaches a walk. Cancelling exactly the friction that
    /// is there — `μ M g` against the motion, and only while there *is*
    /// motion, ramped in over [`SLIDING`] so it does not step — leaves the
    /// closed loop the one [`TAU_ACCEL`] describes. Real legs do the same
    /// thing by pushing harder.
    fn drive(&self, a_des: Vec3, velocity: Vec3, friction: f64, normal: f64) -> Vec3 {
        let speed = velocity.norm();
        let want = a_des * (DRIVE_ASSIST * self.mass);
        if speed <= 1e-9 {
            return want;
        }
        // The friction there actually is — `μ N`, and `N` is what the ground
        // is *actually* carrying, which is not `M g` the moment the body is
        // in water. Buoyancy takes weight off the feet and takes the friction
        // with it; compensating for friction that is not there is free thrust,
        // and the cove measured it: the being waded a foot deeper into a sea
        // written to stop it. Ramped in over `SLIDING` so it does not step as
        // the body breaks away.
        // **It may cancel the sand; it may not out-push it.** The compensation
        // is along the motion, so with nobody at the controls it is positive
        // feedback: a body with a millimetre a second of drift would be handed
        // `μ M g` along the drift and would walk off on its own — which is
        // what a hero standing still did, at 45 mm/s, and is why this is not
        // one line.
        //
        // Commanded, it is the whole of `μ M g`: the body is either sliding,
        // in which case that is exactly the friction there is, or stuck, in
        // which case the extra is what breaks it loose — and either way the
        // start answers in the time constant rather than in the sand's.
        //
        // Uncommanded, it ramps in quadratically over `SLIDING`: a body
        // stopping from a walk is well past that and gets all of it, so the
        // stop is the controller's time constant and not the sand's, and a
        // body creeping at a centimetre a second gets a thousandth of it.
        // And only while the body is actually *sliding*. Kinetic friction is
        // `μ N` against the motion; a body that is barely moving is held by
        // static friction and is dissipating nothing, so there is nothing to
        // cancel and cancelling it anyway is free thrust. That mattered twice:
        // hands off the controls a body with a millimetre a second of drift
        // would be handed `μ N` along the drift and walk off on its own (a
        // hero did, at 45 mm/s), and shoulder-deep in the cove's sea a being
        // stalled against the shore break would be handed the same and wade a
        // foot further than the sea was written to let it.
        let ramp = (speed / SLIDING).min(1.0);
        let feed = friction * normal * ramp * ramp;
        want + velocity * (feed / speed)
    }
}

/// Every link at or under `i`, including `i`.
fn subtree(spec: &BodySpec, i: usize) -> Vec<usize> {
    let mut out = vec![i];
    let mut k = 0;
    while k < out.len() {
        let here = out[k];
        for (j, link) in spec.links.iter().enumerate() {
            if link.parent == Some(here) {
                out.push(j);
            }
        }
        k += 1;
    }
    out
}

// ---- the body --------------------------------------------------------------

/// An articulated body, stepping.
pub struct Body {
    spec: BodySpec,
    consts: Consts,
    model: Model,
    state: State,
    material: ContactMaterial,
    cache: ContactCache,
    /// Where the body is looking: yaw about z, radians, zero along `+x`.
    facing: f64,
    /// The lean the *player* has asked for along the facing, radians.
    tilt: f64,
    /// The lean the *drive* is asking for, as a world horizontal vector whose
    /// norm is the angle. Kept for the snapshot and for the measurement.
    drive_lean: Vec3,
    gait: Gait,
    tool: Option<Tool>,
    /// Where the arm was last asked to put its hand, world.
    aim: Option<Vec3>,
    /// The mean contact normal from the last step: how steep what the body is
    /// standing on is. `None` in the air.
    support: Option<Vec3>,
}

impl Body {
    /// Build the phyz model this spec describes and stand it at the origin.
    pub fn new(spec: BodySpec) -> Self {
        let consts = Consts::of(&spec);
        let mut builder = ModelBuilder::new().gravity(Vec3::new(0.0, 0.0, -GRAVITY)).dt(spec.dt);
        for (i, link) in spec.links.iter().enumerate() {
            // Every link's own frame is its pivot, so a joint's offset from
            // its parent is the difference of the two. The *root* has no
            // parent to be offset from: its `q` is where it is in the world,
            // which is what makes `q[3..6]` the root's position and keeps
            // `being.rs`'s reading of the free joint true.
            let at = match link.parent {
                Some(p) => SpatialTransform::new(Mat3::identity(), link.pivot - spec.links[p].pivot),
                None => SpatialTransform::identity(),
            };
            let inertia = link_inertia(link);
            builder = match (i, link.kind) {
                (0, _) => builder.add_free_body(&link.name, -1, at, inertia),
                (_, PivotKind::Revolute(_)) => builder.add_revolute_body(&link.name, link.parent.unwrap() as i32, at, inertia),
                (_, PivotKind::Spherical) => builder.add_spherical_body(&link.name, link.parent.unwrap() as i32, at, inertia),
            };
        }
        let mut model = builder.build();
        let material = ContactMaterial { friction: spec.friction, restitution: 0.0, ..Default::default() };
        for (i, link) in spec.links.iter().enumerate() {
            let body = &mut model.bodies[i];
            body.visuals = link.lumps.iter().map(|l| l.geometry(link.pivot)).collect();
            if !link.collision.is_empty() {
                body.collisions = link.collision.iter().map(|l| l.geometry(link.pivot)).collect();
                body.geometry = Some(body.collisions[0].geometry.clone());
                body.material = Some(material.clone());
            }
            let joint = &mut model.joints[body.joint_idx];
            if let PivotKind::Revolute(axis) = link.kind {
                joint.axis = axis.normalize();
            }
            // The rotor: what makes a stiff PD survive an explicit step. See
            // `Consts::of`.
            if i > 0 {
                joint.armature = JOINT_ARMATURE * consts.joint[i].0 * spec.dt * spec.dt;
            }
            if let Some(limits) = link.limits {
                // The default limit stiffness is written for a unit inertia; a
                // limb that its own PD can fold walks straight through it.
                joint.limits = Some(limits);
                joint.limit_stiffness = 4.0 * consts.joint[i].0;
                joint.limit_damping = 0.5 * consts.joint[i].1;
            }
        }
        let state = model.default_state();
        let gait = Gait::new(spec.leg_length, spec.walk);
        Self {
            material: material.clone(),
            cache: ContactCache::new(material.margin.max(1e-3)),
            model,
            state,
            consts,
            gait,
            spec,
            facing: 0.0,
            tilt: 0.0,
            drive_lean: Vec3::zeros(),
            tool: None,
            aim: None,
            support: None,
        }
    }

    /// Put a tool in the hand.
    pub fn hold(&mut self, tool: Tool) {
        self.tool = Some(tool);
    }

    /// Stand the body on ground at height `z` under `(x, y)`, facing
    /// `facing`, leaning `lean` radians forward along it, at rest.
    ///
    /// The lean here is where the body *is*, which is not the same as
    /// [`Body::tilt`], where the player has asked it to be: a shove is exactly
    /// the difference between the two, and this is how a test administers one.
    pub fn place(&mut self, x: f64, y: f64, z: f64, facing: f64, lean: f64) {
        self.facing = wrap(facing);
        let rot = self.target_rotation(Vec3::new(self.facing.cos(), self.facing.sin(), 0.0) * lean);
        let w = quat_log(&mat_to_quat(&rot));
        // The root sits `foot_drop` up the body's own axis from the ground
        // under it, so a lean tips the whole body about its feet rather than
        // sliding it sideways — which is what a shove does and what
        // `being.rs::place` does.
        let centre = Vec3::new(x, y, z) + rot.mul_vec(Vec3::z()) * self.consts.foot_drop;
        for (i, v) in [w.x, w.y, w.z, centre.x, centre.y, centre.z].into_iter().enumerate() {
            self.state.q[i] = v;
        }
        for i in 0..self.model.nv.min(6) {
            self.state.v[i] = 0.0;
        }
    }

    /// One step: the controllers, then the contact solve.
    pub fn step(&mut self, drive: &Drive, ground: &dyn Ground, medium: &dyn Medium, dt: f64) {
        if (self.model.dt - dt).abs() > f64::EPSILON {
            self.model.dt = dt;
        }
        self.control(drive, medium, dt);
        self.support = step_on(&self.model, &mut self.state, ground, &self.material, &mut self.cache);
    }

    /// Step for `seconds` with the same drive held down.
    pub fn run_for(&mut self, seconds: f64, drive: &Drive, ground: &dyn Ground, medium: &dyn Medium) {
        let dt = self.model.dt;
        for _ in 0..(seconds / dt).round().max(0.0) as usize {
            self.step(drive, ground, medium, dt);
        }
    }

    /// Everything the controllers write, in one place.
    fn control(&mut self, drive: &Drive, medium: &dyn Medium, dt: f64) {
        let (xforms, vels) = forward_kinematics(&self.model, &self.state);
        self.facing = wrap(self.facing + drive.yaw_delta);
        self.tilt = (self.tilt + drive.lean_delta).clamp(-TILT_MAX, TILT_MAX);

        let r_bw = self.body_to_world();
        let omega = r_bw.mul_vec(self.angular_velocity_body());
        let velocity = r_bw.mul_vec(self.linear_velocity_body());
        // **The centre of mass is what walks.** Regulating the *root* is
        // right for a capsule, whose root is its centre, and wrong for a
        // figure: the pelvis sits below the centre of mass, so the tip into a
        // start throws it forward ahead of the body and the velocity
        // controller, seeing a speed the body has not got, backs off. The
        // hero's rise time was a third over its time constant with a visible
        // stall at 150 ms; on the centre of mass it is a tenth under, and the
        // capsule's number does not move at all, because for a capsule the
        // two are the same point.
        let com = self.com_velocity(&xforms, &vels);
        let horizontal = Vec3::new(com.x, com.y, 0.0);
        let (facing_dir, right) = (self.facing_dir(), self.right_dir());

        // ---- velocity curves, and the lean they ask for ---------------------
        let commanded = facing_dir * drive.forward + right * drive.strafe;
        let n = commanded.norm();
        // A held W and a held D are two full pushes, and unclamped they would
        // walk √2 faster along the diagonal than along either one.
        let unit = if n > 1.0 { commanded / n } else { commanded };
        let speed = if drive.run { self.spec.run } else { self.spec.walk };
        let target = unit * speed;
        let tau = if n > 1e-9 { TAU_ACCEL } else { TAU_STOP };
        let a_des = (target - horizontal) * (1.0 / tau);

        // The lean a body accelerating at `a` would *have*: `tan θ = a/g` is
        // the angle at which the line from the foot through the centre of
        // mass carries both the weight and the push. Capped at `LEAN_MAX`,
        // and pointing wherever `a` points — so a start tips forward, a stop
        // tips back into it, and a hard turn leans into the corner. What it
        // is worth is `DRIVE_ASSIST`'s business.
        let angle = (a_des.norm() / GRAVITY).atan().min(LEAN_MAX);
        let want_lean = if a_des.norm() > 1e-9 { a_des.normalize() * angle } else { Vec3::zeros() };
        // The lean *slews*. A body does not snap into a lean, and neither does
        // the number: a first-order lag at `TAU_LEAN` is what makes the tip
        // read as a body committing to a direction rather than as a step
        // input, and it is also what keeps the tip's own impulse (see
        // `DRIVE_ASSIST`) from stealing the start out from under the velocity
        // controller's time constant.
        self.drive_lean += (want_lean - self.drive_lean) * (dt / TAU_LEAN).min(1.0);
        let lean = self.drive_lean + facing_dir * self.tilt;

        // ---- the medium, first, because it says what the feet are carrying ---
        let (water_force, water_torque, submerged) = self.medium_forces(&xforms, medium, velocity);
        let normal = (self.consts.mass * GRAVITY - water_force.z).max(0.0);

        // ---- the upright spring and the drive --------------------------------
        let mut torque = self.upright_torque(&r_bw, lean, omega);
        let mut force = self.consts.drive(a_des, horizontal, self.spec.friction, normal);
        // **Standing on a hill is not sliding down it.** The friction
        // compensation above cancels the sand, and a cancelled sand cannot
        // hold a body on a slope: on the cove's six per cent beach the being
        // let go of the controls and crept downhill at a quarter of a metre a
        // second for ever, which is the equilibrium of `g sin θ = v/τ`. So
        // whatever gravity has along the ground the feet are on goes in too.
        // The lean does *not* see this term — a body standing on a slope
        // stands upright, it does not lean uphill.
        //
        // **Only on ground you could stand on.** Held against *any* slope the
        // controller is a climbing aid: the cove's reef is a line of boulders
        // and a body that does not slide off one walks over it and out of the
        // level, which is what it did. So the compensation fades out between
        // [`SLOPE_HOLD`] and [`SLOPE_SLIP`] — flat ground gets all of it, a
        // beach gets all of it, a boulder's face gets none, and there is no
        // step in between for a player to feel.
        if let Some(n) = self.support {
            let hold = ((n.z - SLOPE_SLIP.cos()) / (SLOPE_HOLD.cos() - SLOPE_SLIP.cos())).clamp(0.0, 1.0);
            let g = Vec3::new(0.0, 0.0, -GRAVITY);
            force -= (g - n * g.dot(n)) * (self.consts.mass * hold);
        }
        force += water_force;
        torque += water_torque - omega * (medium.spin_damping() * self.consts.c * submerged);

        let (torque, force) = (r_bw.transpose().mul_vec(torque), r_bw.transpose().mul_vec(force));
        for (i, v) in [torque.x, torque.y, torque.z, force.x, force.y, force.z].into_iter().enumerate() {
            self.state.ctrl[i] = v;
        }

        // ---- the joints ------------------------------------------------------
        let mut targets = vec![Vec3::zeros(); self.spec.links.len()];
        self.gait.advance(horizontal.norm(), dt);
        for leg in &self.spec.legs {
            let (hip, knee, side) = (leg.hip, leg.knee, leg.side);
            // The hip is a ball joint; a swing *forward* turns the thigh's
            // `−z` toward `+x`, which is a rotation about `−y`. The knee's own
            // hinge already points the other way, so its target is positive.
            targets[hip] = -Vec3::y() * self.gait.hip(side, horizontal.norm());
            targets[knee] = Vec3::y() * self.gait.knee(side, horizontal.norm());
        }
        self.aim = drive.aim;
        if let (Some(arm), Some(goal)) = (self.spec.arm, drive.aim) {
            let (shoulder, elbow) = self.arm_targets(&xforms, &arm, goal);
            targets[arm.shoulder] = shoulder;
            targets[arm.elbow] = self.hinge_axis(arm.elbow) * elbow;
        }
        self.joint_pd(&xforms, &targets);
    }

    /// How fast the whole body's centre of mass is going, world axes.
    ///
    /// A spatial velocity's linear part is the velocity of the body-fixed
    /// point at the frame's origin, so a lump's own centre is going
    /// `v + ω × p` in that frame and the world sees `R (v + ω × p)`.
    fn com_velocity(&self, xforms: &[SpatialTransform], vels: &[phyz_math::SpatialVec]) -> Vec3 {
        let (mut moment, mut mass) = (Vec3::zeros(), 0.0);
        for (i, link) in self.spec.links.iter().enumerate() {
            let to_world = xforms[i].rot.transpose();
            let (w, v) = (vels[i].angular, vels[i].linear);
            for lump in &link.lumps {
                let m = lump.mass();
                moment += to_world.mul_vec(v + w.cross(lump.centre() - link.pivot)) * m;
                mass += m;
            }
        }
        if mass > 0.0 { moment * (1.0 / mass) } else { Vec3::zeros() }
    }

    /// The same, from the outside. The `Body` runs forward kinematics for it,
    /// so a caller reading it every frame should read [`Body::snapshot`].
    pub fn centre_velocity(&self) -> Vec3 {
        let (xforms, vels) = forward_kinematics(&self.model, &self.state);
        self.com_velocity(&xforms, &vels)
    }

    /// A hinge's own axis, or `+y` when the link is not a hinge.
    fn hinge_axis(&self, i: usize) -> Vec3 {
        match self.spec.links[i].kind {
            PivotKind::Revolute(axis) => axis.normalize(),
            PivotKind::Spherical => Vec3::y(),
        }
    }

    /// The root's spring and damper.
    ///
    /// The error is the rotation that takes the body onto where it is asked to
    /// be, as a rotation vector, so it stays honest at 20° where a small-angle
    /// cross product does not. With `yaws` off it is the *axis* error alone —
    /// a symmetric capsule has no heading of its own — and with it on it is
    /// the whole frame, because a figure with legs has to turn.
    fn upright_torque(&self, r_bw: &Mat3, lean: Vec3, omega: Vec3) -> Vec3 {
        let want = self.target_rotation(lean);
        let error = if self.spec.yaws {
            quat_log(&mat_to_quat(&want.mul_mat(&r_bw.transpose())))
        } else {
            let axis = r_bw.mul_vec(Vec3::z());
            let target = want.mul_vec(Vec3::z());
            let cross = axis.cross(target);
            let sin = cross.norm();
            if sin > 1e-12 { cross * (sin.atan2(axis.dot(target)) / sin) } else { Vec3::zeros() }
        };
        error * self.consts.k - omega * self.consts.c
    }

    /// Where the root is asked to be: yawed to the facing, then tipped by
    /// `lean` — a world horizontal vector whose norm is the angle and whose
    /// direction is the way the body is asked to fall.
    fn target_rotation(&self, lean: Vec3) -> Mat3 {
        let yaw = Mat3::rotation_z(if self.spec.yaws { self.facing } else { 0.0 });
        let angle = lean.norm();
        if angle < 1e-12 {
            return yaw;
        }
        let axis = Vec3::z().cross(lean.normalize());
        quat_exp(&(axis * angle)).to_matrix().mul_mat(&yaw)
    }

    /// The medium's force and torque on the whole body, world axes, and how
    /// much of the body is inside it.
    ///
    /// Every part is asked separately and the answers are reduced to the root:
    /// the force sums, and the offsets become a torque. That is exact for the
    /// capsule spec, which has one part at the root, and an approximation for
    /// a figure, whose limbs are treated as rigid with the trunk rather than
    /// given the medium's force on their own joints. Buoyancy that leans a
    /// body over is the part that matters and it survives; a wave that bends a
    /// knee does not, and does not have to yet.
    fn medium_forces(&self, xforms: &[SpatialTransform], medium: &dyn Medium, velocity: Vec3) -> (Vec3, Vec3, f64) {
        let root = xforms[0].pos;
        let (mut force, mut torque, mut wet, mut mass) = (Vec3::zeros(), Vec3::zeros(), 0.0, 0.0);
        for (i, link) in self.spec.links.iter().enumerate() {
            let x = &xforms[i];
            let to_world = x.rot.transpose();
            for lump in &link.lumps {
                let inst = lump.geometry(link.pivot);
                let centre = x.pos + to_world.mul_vec(inst.origin.pos);
                let axis = to_world.mul_vec(inst.origin.rot.transpose().mul_vec(Vec3::z()));
                let part = Immersed { geometry: &inst.geometry, centre, axis, velocity };
                let got = medium.immerse(&part);
                force += got.force;
                torque += (centre - root).cross(got.force);
                let m = lump.mass();
                wet += got.submerged * m;
                mass += m;
            }
        }
        (force, torque, if mass > 0.0 { wet / mass } else { 0.0 })
    }

    /// A PD on every joint but the root, toward `targets` — a rotation vector
    /// per link, in that joint's own rest frame. Zero is the rig's rest pose,
    /// which is what a body that is standing settles to.
    fn joint_pd(&mut self, xforms: &[SpatialTransform], targets: &[Vec3]) {
        for (i, link) in self.spec.links.iter().enumerate().skip(1) {
            let joint_idx = self.model.bodies[i].joint_idx;
            let (q0, v0) = (self.model.q_offsets[joint_idx], self.model.v_offsets[joint_idx]);
            let (k, c) = self.consts.joint[i];
            // What the limb weighs, held. The PD is then a spring on the
            // *error* rather than a spring that has to out-pull gravity, which
            // is the difference between an arm that reaches a target and an
            // arm that hangs a fifth of a radian below it. Exact at rest, and
            // exact whenever the limb is not accelerating.
            let hold = xforms[i].rot.mul_vec(self.gravity_moment(xforms, i));
            match link.kind {
                PivotKind::Revolute(axis) => {
                    // A hinge's target is a rotation vector like anyone else's;
                    // what it can do with it is the part along its own axis.
                    let axis = axis.normalize();
                    let (q, qd) = (self.state.q[q0], self.state.v[v0]);
                    self.state.ctrl[v0] = k * (targets[i].dot(axis) - q) - c * qd + hold.dot(axis);
                }
                PivotKind::Spherical => {
                    let current = quat_exp(&Vec3::new(self.state.q[q0], self.state.q[q0 + 1], self.state.q[q0 + 2]));
                    let want = quat_exp(&targets[i]);
                    // In the child's own frame, which is where its `ctrl` is.
                    let error = quat_log(&current.conjugate().mul(&want));
                    let (e, h) = ([error.x, error.y, error.z], [hold.x, hold.y, hold.z]);
                    for a in 0..3 {
                        self.state.ctrl[v0 + a] = k * e[a] - c * self.state.v[v0 + a] + h[a];
                    }
                }
            }
        }
    }

    /// Minus the moment gravity puts on everything at or under link `i`, about
    /// that link's own joint, in **world** axes: what the joint has to supply
    /// to hold the limb where it is.
    fn gravity_moment(&self, xforms: &[SpatialTransform], i: usize) -> Vec3 {
        let pivot = xforms[i].pos;
        let mut moment = Vec3::zeros();
        for j in &self.consts.subtrees[i] {
            let x = &xforms[*j];
            let to_world = x.rot.transpose();
            for lump in &self.spec.links[*j].lumps {
                let centre = x.pos + to_world.mul_vec(lump.centre() - self.spec.links[*j].pivot);
                moment += (centre - pivot).cross(Vec3::new(0.0, 0.0, -lump.mass() * GRAVITY));
            }
        }
        -moment
    }

    /// The shoulder's target rotation and the elbow's target angle for a hand
    /// at `goal`, world.
    ///
    /// The two-link solve of `sims/rune/hero/figure.rs::joint`: the elbow lies
    /// on a circle about the line from shoulder to hand, and the hint picks the
    /// point on it. When the target is out of reach the chain straightens and
    /// points at it rather than failing.
    fn arm_targets(&self, xforms: &[SpatialTransform], arm: &Arm, goal: Vec3) -> (Vec3, f64) {
        let shoulder_link = &self.spec.links[arm.shoulder];
        let parent = shoulder_link.parent.unwrap_or(0);
        let parent_rot = xforms[parent].rot.transpose();
        let shoulder_world = xforms[arm.shoulder].pos;

        let hint = parent_rot.mul_vec(arm.hint);
        let elbow_world = two_link(shoulder_world, goal, arm.upper, arm.lower, hint);

        // The shoulder: the shortest rotation taking the rest upper-arm
        // direction onto the one the solve wants, read in the parent's frame.
        let rest = (self.spec.links[arm.elbow].pivot - shoulder_link.pivot).normalize();
        let want = parent_rot.transpose().mul_vec((elbow_world - shoulder_world).normalize());
        let shoulder = shortest_arc(rest, want);

        // The elbow: the interior angle, less the one the rest pose has.
        let rest_upper = (self.spec.links[arm.elbow].pivot - shoulder_link.pivot).normalize();
        let rest_lower = (arm.hand - self.spec.links[arm.elbow].pivot).normalize();
        let rest_angle = rest_upper.dot(rest_lower).clamp(-1.0, 1.0).acos();
        let upper = (elbow_world - shoulder_world).normalize();
        let lower = (goal - elbow_world).normalize();
        let angle = upper.dot(lower).clamp(-1.0, 1.0).acos();
        (shoulder, angle - rest_angle)
    }

    // ---- reading it back ---------------------------------------------------

    pub fn spec(&self) -> &BodySpec {
        &self.spec
    }

    pub fn consts(&self) -> &Consts {
        &self.consts
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    pub fn time(&self) -> f64 {
        self.state.time
    }

    pub fn dt(&self) -> f64 {
        self.model.dt
    }

    pub fn mass(&self) -> f64 {
        self.consts.mass
    }

    /// The upright spring, N·m/rad and N·m·s/rad. Reported, not tuned.
    pub fn upright_spring(&self) -> (f64, f64) {
        (self.consts.k, self.consts.c)
    }

    pub fn facing(&self) -> f64 {
        self.facing
    }

    /// The lean the player is asking for along the facing, radians.
    pub fn tilt(&self) -> f64 {
        self.tilt
    }

    /// Set it outright, rather than by a mouse delta. Clamped to
    /// [`TILT_MAX`].
    pub fn set_tilt(&mut self, tilt: f64) {
        self.tilt = tilt.clamp(-TILT_MAX, TILT_MAX);
    }

    /// The lean the *drive* is asking for, radians: what "lean into a start"
    /// is, before the body has got there.
    pub fn commanded_lean(&self) -> f64 {
        self.drive_lean.norm()
    }

    pub fn gait(&self) -> &Gait {
        &self.gait
    }

    /// The root's origin, world.
    pub fn root(&self) -> Vec3 {
        Vec3::new(self.state.q[3], self.state.q[4], self.state.q[5])
    }

    /// The body's own axis, foot to head, in world.
    pub fn axis(&self) -> Vec3 {
        self.body_to_world().mul_vec(Vec3::z())
    }

    /// How far the body is off vertical, radians. What a shove leaves behind
    /// and what the upright spring spends itself on.
    pub fn lean(&self) -> f64 {
        self.axis().dot(Vec3::z()).clamp(-1.0, 1.0).acos()
    }

    /// The root's velocity in world axes, m/s.
    pub fn velocity(&self) -> Vec3 {
        self.body_to_world().mul_vec(self.linear_velocity_body())
    }

    /// The body's horizontal speed, m/s: what `walk` caps, and what the
    /// velocity controller regulates — the **centre of mass**, not the root.
    /// For a capsule they are the same point; for a figure they are not, and
    /// the difference is a third of a time constant.
    pub fn speed(&self) -> f64 {
        let v = self.centre_velocity();
        v.x.hypot(v.y)
    }

    /// Where the body is looking, as a unit vector on the ground plane.
    pub fn facing_dir(&self) -> Vec3 {
        let (s, c) = self.facing.sin_cos();
        Vec3::new(c, s, 0.0)
    }

    /// The body's right: `facing × ẑ`, so `strafe = 1` walks to its right.
    pub fn right_dir(&self) -> Vec3 {
        self.facing_dir().cross(Vec3::z())
    }

    /// The optical proxy: where a capsule of [`BodySpec::capsule`]'s size
    /// would sit on this body, as a centre and a **body → world** rotation.
    ///
    /// `sims/rune` traces its caustic through a capsule of glass. A figure is
    /// not one, and until the refractor in the live scene *is* the held lens
    /// the score has to be read off something — so a spec may carry the
    /// capsule the optics still see, and this is where it is. For
    /// [`BodySpec::capsule`] it is the body itself.
    pub fn capsule_pose(&self) -> Option<(Vec3, Mat3)> {
        let (_, height) = self.spec.capsule?;
        let r_bw = self.body_to_world();
        Some((self.root() + r_bw.mul_vec(Vec3::z()) * (height / 2.0 - self.consts.foot_drop), r_bw))
    }

    /// Where the body's feet are on the ground, world: the root, less how far
    /// up its own axis the root sits.
    pub fn footing(&self) -> Vec3 {
        self.root() - self.axis() * self.consts.foot_drop
    }

    /// Where the hand is, world.
    pub fn hand(&self) -> Option<Pose> {
        let arm = self.spec.arm?;
        let (xforms, _) = forward_kinematics(&self.model, &self.state);
        let x = &xforms[arm.elbow];
        let to_world = x.rot.transpose();
        let local = arm.hand - self.spec.links[arm.elbow].pivot;
        Some(Pose::new(x.pos + to_world.mul_vec(local), to_world))
    }

    /// The held tool and where it actually is — not where it was asked to be,
    /// because an arm has mass. What a scorer reads.
    pub fn held(&self) -> Option<(Pose, &Tool)> {
        let tool = self.tool.as_ref()?;
        let hand = self.hand()?;
        Some((tool.placed(&hand), tool))
    }

    pub fn snapshot(&self) -> Snapshot {
        let r_bw = self.body_to_world();
        let (xforms, _) = forward_kinematics(&self.model, &self.state);
        Snapshot {
            t: self.state.time,
            dt: self.model.dt,
            root: (self.root(), r_bw.transpose()),
            root_vel: (r_bw.mul_vec(self.linear_velocity_body()), r_bw.mul_vec(self.angular_velocity_body())),
            facing: self.facing,
            lean: self.lean(),
            tilt: self.tilt,
            gait_phase: self.gait.phase,
            parts: self
                .spec
                .links
                .iter()
                .enumerate()
                .map(|(i, link)| Part { name: link.name.clone(), pose: Pose::new(xforms[i].pos, xforms[i].rot.transpose()) })
                .collect(),
            held: self.held().map(|(p, _)| p),
        }
    }

    /// The phyz view, as a [`World`]: the model and the state, plus the
    /// controller's own state as knobs.
    ///
    /// A `World` is columns and nothing that runs, and `facing`, `tilt` and
    /// the gait phase are exactly that — three numbers the controller carries
    /// between steps that phyz's `State` has no column for. Putting them in
    /// `params` is what makes [`BodyStep`] a real `Step`: pure, `World →
    /// World`, with nothing hidden in the stepper.
    pub fn world(&self) -> World {
        World::from_phyz(self.model.clone(), self.state.clone()).with_params(vec![
            Param::new("player.facing", self.facing),
            Param::new("player.tilt", self.tilt),
            Param::new("player.gait_phase", self.gait.phase),
        ])
    }

    /// The other way round: adopt a world this body's own spec produced.
    pub fn set_world(&mut self, world: &World) {
        let (model, state) = world.phyz();
        self.model = model.clone();
        self.state = state.clone();
        self.facing = world.param("player.facing").unwrap_or(self.facing);
        self.tilt = world.param("player.tilt").unwrap_or(self.tilt);
        self.gait.phase = world.param("player.gait_phase").unwrap_or(self.gait.phase);
    }

    fn body_to_world(&self) -> Mat3 {
        quat_exp(&Vec3::new(self.state.q[0], self.state.q[1], self.state.q[2])).to_matrix()
    }

    fn angular_velocity_body(&self) -> Vec3 {
        Vec3::new(self.state.v[0], self.state.v[1], self.state.v[2])
    }

    fn linear_velocity_body(&self) -> Vec3 {
        Vec3::new(self.state.v[3], self.state.v[4], self.state.v[5])
    }
}

// ---- the Step ---------------------------------------------------------------

/// The same body as a [`Step`]: `World → World`, pure, with the [`Drive`]
/// packed into the [`Action`].
///
/// It owns a `Body` and clones it per step, which is what "pure" costs when
/// the plant is a contact cache: the warm start is part of the body, and a
/// `Step` may not keep it across worlds it was not given. A player's loop
/// calls [`Body::step`] directly and pays nothing; a *policy* calls this and
/// pays one clone of the model per step, which is what
/// [`crate::step::PhyzStep`] pays too.
pub struct BodyStep {
    body: std::sync::Mutex<Body>,
    ground: Arc<dyn Ground>,
    medium: Arc<dyn Medium>,
    dt: f64,
}

impl BodyStep {
    pub fn new(body: Body, ground: Arc<dyn Ground>, medium: Arc<dyn Medium>) -> Self {
        let dt = body.dt();
        Self { body: std::sync::Mutex::new(body), ground, medium, dt }
    }
}

impl Step for BodyStep {
    fn step(&self, world: &World, action: &Action) -> World {
        let mut body = self.body.lock().expect("the player's body is not poisoned");
        body.set_world(world);
        let drive = Drive::from_action(action);
        body.step(&drive, self.ground.as_ref(), self.medium.as_ref(), self.dt);
        body.world()
    }
}

// ---- small maths -----------------------------------------------------------

/// A link's spatial inertia about its own joint frame.
fn link_inertia(link: &Link) -> SpatialInertia {
    let mut mass = 0.0;
    let mut moment = Vec3::zeros();
    let mut parts = Vec::new();
    for lump in &link.lumps {
        let (m, i) = lump.inertia();
        mass += m;
        moment += lump.centre() * m;
        parts.push((m, lump.centre(), i));
    }
    if mass <= 0.0 {
        // A link with no stuff still needs an inertia the solver can invert.
        return SpatialInertia::new(1e-3, Vec3::zeros(), Mat3::identity() * 1e-5);
    }
    let com = moment * (1.0 / mass);
    let mut tensor = Mat3::zero();
    for (m, c, i) in parts {
        let d = c - com;
        tensor = mat_add(tensor, mat_add(i, mat_scale(mat_sub(Mat3::from_diagonal(&Vec3::splat(d.norm_sq())), outer(d, d)), m)));
    }
    SpatialInertia::new(mass, com - link.pivot, tensor)
}

/// The middle joint of a two-link chain from `root` to `tip`.
pub fn two_link(root: Vec3, tip: Vec3, upper: f64, lower: f64, hint: Vec3) -> Vec3 {
    let to_tip = tip - root;
    let reach = to_tip.norm().max(1e-6);
    let d = reach.clamp(1e-6, upper + lower - 1e-6);
    let dir = to_tip * (1.0 / reach);
    let along = (upper * upper - lower * lower + d * d) / (2.0 * d);
    let off = (upper * upper - along * along).max(0.0).sqrt();
    let mut pole = hint - dir * hint.dot(dir);
    if pole.norm() < 1e-9 {
        let down = -Vec3::z();
        pole = down - dir * down.dot(dir);
    }
    root + dir * along + pole.normalize() * off
}

/// The shortest rotation taking `from` onto `to`, as a rotation vector.
fn shortest_arc(from: Vec3, to: Vec3) -> Vec3 {
    let cross = from.cross(to);
    let sin = cross.norm();
    if sin < 1e-12 {
        if from.dot(to) > 0.0 {
            return Vec3::zeros();
        }
        // antiparallel: any perpendicular will do
        let a = if from.z.abs() < 0.9 { Vec3::z() } else { Vec3::x() };
        return from.cross(a).normalize() * PI;
    }
    cross * (sin.atan2(from.dot(to)) / sin)
}

/// A rotation whose `+z` is `u`, as columns `(e1, e2, u)`.
fn frame_from_z(u: Vec3) -> Mat3 {
    let z = u.normalize();
    let a = if z.z.abs() < 0.9 { Vec3::z() } else { Vec3::x() };
    let e1 = a.cross(z).normalize();
    Mat3::from_cols(e1, z.cross(e1), z)
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

fn mat_add(a: Mat3, b: Mat3) -> Mat3 {
    Mat3::from_cols(a.col(0) + b.col(0), a.col(1) + b.col(1), a.col(2) + b.col(2))
}

fn mat_sub(a: Mat3, b: Mat3) -> Mat3 {
    Mat3::from_cols(a.col(0) - b.col(0), a.col(1) - b.col(1), a.col(2) - b.col(2))
}

fn mat_scale(a: Mat3, k: f64) -> Mat3 {
    Mat3::from_cols(a.col(0) * k, a.col(1) * k, a.col(2) * k)
}

/// A rotation matrix as a unit quaternion. Shepperd's method: pick the largest
/// of the four to divide by, so no branch is ever near zero.
fn mat_to_quat(m: &Mat3) -> Quat {
    let (m00, m11, m22) = (m.get(0, 0), m.get(1, 1), m.get(2, 2));
    let trace = m00 + m11 + m22;
    let (w, x, y, z) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (0.25 * s, (m.get(2, 1) - m.get(1, 2)) / s, (m.get(0, 2) - m.get(2, 0)) / s, (m.get(1, 0) - m.get(0, 1)) / s)
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        ((m.get(2, 1) - m.get(1, 2)) / s, 0.25 * s, (m.get(0, 1) + m.get(1, 0)) / s, (m.get(0, 2) + m.get(2, 0)) / s)
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        ((m.get(0, 2) - m.get(2, 0)) / s, (m.get(0, 1) + m.get(1, 0)) / s, 0.25 * s, (m.get(1, 2) + m.get(2, 1)) / s)
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        ((m.get(1, 0) - m.get(0, 1)) / s, (m.get(0, 2) + m.get(2, 0)) / s, (m.get(1, 2) + m.get(2, 1)) / s, 0.25 * s)
    };
    Quat { w, v: Vec3::new(x, y, z) }.normalize()
}

/// Wrap an angle into `(-π, π]`, so a facing spun a hundred times reads the
/// same as one that has not.
pub fn wrap(a: f64) -> f64 {
    let two = 2.0 * PI;
    let r = a - two * ((a + PI) / two).floor();
    if r <= -PI { r + two } else { r }
}

// ---- a rig for the crate's own tests ---------------------------------------

impl Skeleton {
    /// The hero's proportions, in the body's own frame.
    ///
    /// The numbers are `sims/rune/hero/figure.rs`'s `Rig::DEFAULT` — a 1.11 m
    /// Mii — carried over so `tests/player.rs` and the doctests have a figure
    /// with legs to stand up without kosm depending on a sim. The *level*
    /// builds its own from its own `Rig`, through `being.rs::hero_skeleton`,
    /// so this is a fixture and not a second statement of the rig.
    ///
    /// `figure.rs` is millimetres with `+y` forward; this is metres with `+x`
    /// forward, which is [`hero_local`]'s one line.
    pub fn demo_hero() -> Self {
        let d = |name: &str, fallback: f64| crate::material::named(name).map(|m| m.density).unwrap_or(fallback);
        let (leg_r, boot_r, arm_r, hand_r) = (0.056, 0.100, 0.064, 0.080);
        let hip = [hero_local(94.0, 0.0, 306.0), hero_local(-94.0, 0.0, 306.0)];
        let ankle = [hero_local(118.0, 0.0, 96.0), hero_local(-118.0, 0.0, 96.0)];
        let shoulder = [hero_local(174.0, 0.0, 588.0), hero_local(-174.0, 0.0, 588.0)];
        let hand = [hero_local(258.0, 66.0, 372.0), hero_local(-258.0, 66.0, 372.0)];
        // `Rig::LEG_SLACK` is 14 mm, and `thigh` is 0.53 of the drop plus it.
        let (thigh, shin) = (0.53 * 0.224, 0.224 - 0.53 * 0.224);
        let knee_hint = hero_local(0.0, 1.0, 0.0);
        let knee = [
            two_link(hip[0], ankle[0], thigh, shin, knee_hint),
            two_link(hip[1], ankle[1], thigh, shin, knee_hint),
        ];
        let elbow = [
            two_link(shoulder[0], hand[0], 0.208, 0.196, hero_local(1.0, -0.34, -0.16)),
            two_link(shoulder[1], hand[1], 0.208, 0.196, hero_local(-1.0, -0.34, -0.16)),
        ];
        Self {
            torso: hero_local(0.0, 0.0, 306.0),
            neck: hero_local(0.0, 0.0, 646.0),
            chest: hero_local(0.0, 0.0, 452.0),
            head: hero_local(0.0, 0.0, 885.0),
            skirt: hero_local(0.0, 0.0, 248.0),
            hip,
            knee,
            ankle,
            shoulder,
            elbow,
            hand,
            leg_r,
            boot_r,
            arm_r,
            hand_r,
            chest_r: 0.180,
            head_r: 0.228,
            skirt_r: 0.268,
            leg_density: d("cream", 1400.0),
            boot_density: d("boot", 950.0),
            arm_density: d("cloak", 1300.0),
            torso_density: d("cloak", 1300.0),
            head_density: d("skin", 1050.0),
            capsule: None,
        }
    }
}

/// Hero-local millimetres (`+y` forward, `+x` right) as body-local metres
/// (`+x` forward, `+y` left). The only place the two conventions meet.
pub fn hero_local(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3::new(y / 1000.0, -x / 1000.0, z / 1000.0)
}
