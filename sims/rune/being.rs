//! The being and the door: the two things in the cove that move.
//!
//! The being is [`kosm::player::Body`] now, and this file is what is left of
//! it that is the *cove's*: where it spawns, what the sea is made of, and the
//! door.
//!
//! What moved out, and where it went:
//!
//! - the upright spring, derived from the body's own inertia about its feet →
//!   [`kosm::player::body`];
//! - the walk, which was a bang-bang force capped at `walk_mps` and is now a
//!   velocity controller with a rise and a stop time and a lean into both →
//!   the same;
//! - buoyancy, form drag against the shore break, and the spin the water takes
//!   out of a shove → [`kosm::player::Water`], the same integral and the same
//!   numbers;
//! - the SDF contact step with the net under it → [`kosm::player::Netted`]
//!   over [`kosm::player::SdfGround`], which is `sim::step_on_sdf_over` with
//!   the count kept in the ground rather than in the level.
//!
//! What stayed: the door, a slab of stone on a revolute joint at its own
//! vertical edge, hinged into the cliff, with a soft limit at [`DOOR_LIMIT`]
//! and a spring that drives it open only while the gate is set. It is a body
//! with mass: it swings, it does not teleport. It has no contacts — it stands
//! inside a recess that is *in* the baked field, and a collidable door would
//! be born penetrating its own doorway — so it steps over
//! [`kosm::player::Nowhere`], which is the same integrator with nothing under
//! it.
//!
//! **The being has legs.** [`Player::Hero`] builds the figure of
//! `sims/rune/hero/figure.rs` out of its own `Rig::pivots()`, with the
//! costume's substances for the masses and the boots for the contacts;
//! [`Player::Capsule`] is the glass capsule the level was written against,
//! and is still what `being_r_mm` and `being_h_mm` describe. `KOSM_RUNE_PLAYER`
//! picks, and the hero is the default now.
//!
//! **And the hero holds the lens.** [`Cove::with_player`] puts
//! `hero/kit.rs`'s two-and-a-half-metre crown glass in its right hand, the
//! arm's own PD carries it to [`LENS_AIM`] — up and out, the way somebody
//! sighting through a lens holds one — and [`Cove::held_lens`] hands back
//! where it *actually got to*, which is not where it was asked to be because
//! an arm has mass. That pose is what `rune::score_lens` traces the live gate
//! through and what the picture draws the glass at, so the number and the
//! image are the same piece of glass. [`Cove::being_pose`]'s capsule proxy
//! survives for the capsule body and for the hint's gradient, which is still
//! written on the capsule's two knobs.
//!
//! Everything the dynamics needs from the player arrives as [`Input`], and
//! everything the picture needs comes back as [`Snapshot`].
//!
//! Metres, radians, seconds; z up.

use std::f64::consts::PI;
use std::sync::{Arc, LazyLock};

use kosm::player::body::TILT_MAX as PLAYER_TILT_MAX;
use kosm::player::ground::step_on;
use kosm::player::{Body, BodySpec, Drive, Netted, Nowhere, Part, Pose, SdfGround, Skeleton, Tool, Water};
use kosm_scan::SdfGrid;
use phyz_contact::{ContactCache, ContactMaterial};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{GeomInstance, Geometry, Model, ModelBuilder, State};
use phyz_rigid::forward_kinematics;

use super::CoveScene;
use super::hero::figure::Rig;
use super::sim::{Beach, GLASS};
use kosm::material::{self, Material as Substance};

/// The being is body 0 of its own model, on the free joint it starts with.
pub const BEING: usize = 0;

/// How far the player may lean the being, radians. Ten degrees is the design's
/// "a few degrees about its own horizontal axis": enough to walk the caustic
/// across the door, not enough to fall over. `kosm::player` owns the number
/// now; this is the cove's name for it.
pub const TILT_MAX: f64 = PLAYER_TILT_MAX;

/// Glass on wet sand, and the cove's own number rather than the library's.
///
/// Deliberately far below [`super::sim::SAND_FRICTION`], which is what the
/// *marble* rolls on: a rolling sphere needs grip and a walking being needs to
/// slide. Neither is `kosm::material`'s sand, because neither is a
/// measurement — they are the two ends the level wants a body to sit at.
pub const BEING_FRICTION: f64 = 0.35;

/// The sea: `sea water` out of `kosm::material`, 1025 kg/m³.
///
/// Glass ([`GLASS`]) is two and a half times as dense, so a being in the sea
/// gets lighter and keeps its feet: it never floats off the bed, and there is
/// nothing to swim.
pub static SEA: LazyLock<Substance> = LazyLock::new(|| material::named("sea water").expect("sea water is in kosm's material library"));

/// The drag coefficient of a bluff body in water. A capsule broadside is about
/// a cylinder, which is about one.
pub const WATER_DRAG_CD: f64 = 1.0;

/// How much of the upright spring's own damping the water adds when the being
/// is fully under, as a fraction. A quarter is "a little".
pub const WATER_SPIN_DAMP: f64 = 0.25;

/// How far the door swings before the recess stops it, radians.
pub const DOOR_LIMIT: f64 = 100.0 * PI / 180.0;

/// How fast the door swings, rad/s: critically damped at the limit, so the
/// angle rises monotonically and settles on it instead of slamming into it.
pub const DOOR_OMEGA: f64 = 3.0;

/// Which body the player is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Player {
    /// The glass capsule the level was written against: `being_r_mm` by
    /// `being_h_mm`, one free joint, and the whole body is the lens.
    Capsule,
    /// The adventurer of `sims/rune/hero`, on every hinge its `Rig` declares.
    Hero,
}

impl Player {
    /// What `KOSM_RUNE_PLAYER` says, or the default.
    ///
    /// Default **hero**, and that is now a statement about the optics as much
    /// as about the body. The day this comment was written the other way
    /// round has arrived: the lens is the thing in the hero's hand
    /// ([`Cove::held_lens`]), the live gate scores the caustic *that* throws
    /// ([`super::rune::score_lens`]), and the figure is drawn from its own
    /// solids at the transforms the simulation puts them at. A capsule of
    /// glass with no legs is the level's old body and its old optics, and
    /// `KOSM_RUNE_PLAYER=capsule` is still every one of `rune_tests.rs`'s
    /// numbers — the focal length, the solvable band, the recorded solution —
    /// because the offline solve and the solvability sweep are on the
    /// capsule's own `rune::Pose` and have not moved.
    pub fn from_env() -> Self {
        match std::env::var("KOSM_RUNE_PLAYER").ok().as_deref() {
            Some("capsule") => Player::Capsule,
            _ => Player::Hero,
        }
    }
}

impl Default for Player {
    fn default() -> Self {
        Player::Hero
    }
}

/// The hero's rig as [`kosm::player`]'s plain-data skeleton.
///
/// `figure.rs` is hero-local **millimetres** with `+y` forward; the player's
/// frame is metres with `+x` forward. [`kosm::player::body::hero_local`] is
/// that one line, and `Rig::pivots()` is where every joint comes from — so
/// there is still exactly one statement of where the hero's knees are.
pub fn hero_skeleton(rig: &Rig) -> Skeleton {
    let p = rig.pivots();
    let at = |v: [f64; 3]| kosm::player::body::hero_local(v[0], v[1], v[2]);
    let density = |name: &str| material::named(name).map(|m| m.density).unwrap_or(1000.0);
    Skeleton {
        torso: at(p.torso),
        neck: at(p.neck),
        chest: at([0.0, rig.pelvis_y, rig.chest_z]),
        head: at([0.0, 0.0, rig.head_z]),
        skirt: at([0.0, rig.pelvis_y, rig.skirt_z]),
        hip: [at(p.hip[0]), at(p.hip[1])],
        knee: [at(p.knee[0]), at(p.knee[1])],
        ankle: [at(p.ankle[0]), at(p.ankle[1])],
        shoulder: [at(p.shoulder[0]), at(p.shoulder[1])],
        elbow: [at(p.elbow[0]), at(p.elbow[1])],
        hand: [at(p.hand[0]), at(p.hand[1])],
        leg_r: rig.leg_r / 1000.0,
        boot_r: rig.boot_r / 1000.0,
        arm_r: rig.arm_r / 1000.0,
        hand_r: rig.hand_r / 1000.0,
        chest_r: rig.chest_r / 1000.0,
        head_r: rig.head_r / 1000.0,
        skirt_r: rig.skirt_r / 1000.0,
        leg_density: density("cream"),
        boot_density: density("boot"),
        arm_density: density("cloak"),
        torso_density: density("cloak"),
        head_density: density("skin"),
        // What the figure weighs as a body, rather than what a bag of
        // overlapping solid balls adds up to. See `Rig::mass_kg`.
        mass_kg: (rig.mass_kg > 0.0).then_some(rig.mass_kg),
        capsule: None,
    }
}

/// Where the hero's hand is asked to go, in the **body's own frame**: a
/// bearing at the shoulder, up and out to the hero's right.
///
/// `LENS_AIM_EL` is the lift and `LENS_AIM_AZ` the swing, which is negative
/// because the body's `+y` is its *left*. They are `hero/mod.rs`'s doorstep
/// numbers — 46° up and 28° out — read into the player's frame (`+x` forward,
/// `+y` left, `+z` up) rather than the figure's, and they are what puts the
/// glass clear of a 456 mm head on a 1113 mm frame.
///
/// Two angles and not a vector, because they are *knobs* now:
/// [`super::rune::HeroPose`] solves over them, and a solved elevation is what
/// decides how far off the door the hero has to stand. Where the hero has to
/// stand is [`super::rune::solve_hero`]'s question and not this constant's.
pub const LENS_AIM_EL: f64 = 46.0 * PI / 180.0;
pub const LENS_AIM_AZ: f64 = -28.0 * PI / 180.0;

/// How much of the arm's straight reach the aim asks for.
pub const LENS_REACH: f64 = 0.97;

/// The hand's target in the body's own frame, from the shoulder, for a lift
/// and a swing.
pub fn lens_aim_dir(el: f64, az: f64) -> Vec3 {
    let (se, ce) = el.sin_cos();
    let (sa, ca) = az.sin_cos();
    Vec3::new(ce * ca, ce * sa, se)
}

/// Where the glass sits relative to the fist that is holding its rim, with
/// `cant` radians of wrist on top.
///
/// A hand on a rim holds the lens a semi-diameter away in the lens's own
/// plane, and the optical axis runs across it. So the grip puts the centre of
/// the glass `LENS_D/2` up-and-outboard of the hand and turns the lens's own
/// `+z` — which is the optical axis `hero/kit.rs` cuts it about — onto the
/// body's forward. `hero/mod.rs::GRIP_HINT` is the same offset in the
/// figure's frame.
///
/// `cant` is a **turn of the wrist**: the whole grip, glass and offset
/// together, rotated about the hand's own `+z`. It swings the optical axis
/// off the forearm's line without moving the fist, which is the one thing a
/// person holding a lens by its rim can do that neither walking nor raising
/// the arm does. It costs `cos θ` of the sun the glass collects and buys
/// nothing in *where* the caustic lands — a thin lens images a parallel
/// bundle wherever its undeviated chief ray crosses the focal plane, and a
/// cant moves that ray not at all — so what it is really worth is the shape
/// of the patch, and the solve is what says how much of it to spend.
/// `hero/mod.rs::LENS_CANT_DEG` is the same turn taken about the world's
/// vertical, for the stills' framing.
pub fn lens_grip(cant: f64) -> Pose {
    let hint = Vec3::new(0.15, -0.55, 0.82).normalize();
    let radius = 0.5 * super::hero::kit::LENS_D / 1000.0;
    // the lens's own axes in the hand's frame, as columns: +z forward,
    // +x to the body's left, +y up
    let rot = Mat3::new(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    let roll = Mat3::rotation_z(cant);
    Pose::new(roll.mul_vec(hint * radius), roll.mul_mat(&rot))
}

/// The hero's rig, built once: the spec every link and lump of the figure
/// comes out of, and how far the root sits above the sand under it.
///
/// A `BodySpec` is a tree of inertias and is not free to build, and
/// [`super::rune::solve_hero`] asks for the hero's arm arithmetic tens of
/// thousands of times without a body anywhere near it. Nothing in here
/// depends on where the hero is standing, so there is one.
pub struct HeroRig {
    pub spec: BodySpec,
    /// How far up its own axis the root sits from the ground under its feet.
    pub foot_drop: f64,
}

pub static HERO_RIG: LazyLock<HeroRig> = LazyLock::new(|| {
    let spec = BodySpec::hero(&hero_skeleton(&Rig::DEFAULT));
    let foot_drop = Body::new(spec.clone()).consts().foot_drop;
    HeroRig { spec, foot_drop }
});

/// The door's substance, read off the door body the level built.
fn door_substance(scene: &CoveScene) -> anyhow::Result<Substance> {
    scene
        .authored
        .bodies
        .iter()
        .find(|b| b.name == "door")
        .ok_or_else(|| anyhow::anyhow!("the cove has no `door` body to take a substance from"))?
        .substance()
        .ok_or_else(|| anyhow::anyhow!("the door's material is not in kosm's material library"))
}

/// What the player is asking for on one step.
///
/// `forward` and `strafe` are in `-1..=1` and read as a held direction, not as
/// a speed; the deltas are this step's mouse, in radians.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Input {
    pub forward: f64,
    pub strafe: f64,
    pub yaw_delta: f64,
    pub tilt_delta: f64,
}

impl Input {
    /// Hands off the controls.
    pub const STILL: Self = Self { forward: 0.0, strafe: 0.0, yaw_delta: 0.0, tilt_delta: 0.0 };

    /// Walking along the facing and nothing else.
    pub fn walking(forward: f64) -> Self {
        Self { forward, ..Self::STILL }
    }

    /// As the controller's own drive.
    pub fn drive(&self) -> Drive {
        Drive {
            forward: self.forward,
            strafe: self.strafe,
            run: false,
            yaw_delta: self.yaw_delta,
            lean_delta: self.tilt_delta,
            aim: None,
        }
    }
}

/// The cove at one instant, as the picture wants it.
#[derive(Clone, Debug)]
pub struct Snapshot {
    /// The simulation's own clock, seconds.
    pub t: f64,
    /// The solver's step, seconds.
    pub dt: f64,
    /// The being's centre and its **world → body** rotation, phyz's convention
    /// (so `rot.transpose()` is the way out to world). This is the *capsule
    /// proxy*: what the optics still trace through.
    pub being: (Vec3, Mat3),
    /// The being's linear and angular velocity in **world** axes.
    pub being_vel: (Vec3, Vec3),
    /// Where the being is looking and how far it is leaning, radians.
    pub facing: f64,
    pub tilt: f64,
    /// The hinge angle, radians: 0 closed, [`DOOR_LIMIT`] wide open.
    pub door_angle: f64,
    /// The door's body frame — at the hinge, `rot` world → body.
    pub door: SpatialTransform,
    pub gate_open: bool,
    /// Where the held lens is, if the hero is holding one.
    pub held: Option<Pose>,
    /// Every link of the figure, placed — [`None`] for the capsule, which has
    /// only the one.
    pub parts: Option<Arc<Vec<Part>>>,
}

// `Snapshot` was `Copy` and is not any more, and the `Arc` is why it is still
// cheap. `game.rs` sends one down a channel to the render thread every frame
// and keeps the last one behind a mutex; the *parts* are what the render
// thread places the figure from, so they have to travel with it, and the only
// two ways to do that are to allocate a `Vec` per frame on the simulation
// thread or to refcount one. The body builds the `Vec` once per snapshot
// either way; this way nothing after that copies it.

/// The door's numbers and where in its own little state its one DOF lives.
#[derive(Clone, Copy, Debug)]
struct Door {
    body: usize,
    q: usize,
    v: usize,
    /// The drive toward open, N·m/rad and N·m·s/rad.
    k: f64,
    c: f64,
}

/// The cove, stepping: the being, the door, the field they stand on.
pub struct Cove {
    /// The being.
    pub body: Body,
    /// The baked field with the level's net under it.
    ground: Netted<SdfGround>,
    /// The sea.
    sea: Water,
    /// The door's own model: one fixed hinge and one slab, no contacts.
    door_model: Model,
    door_state: State,
    door_cache: ContactCache,
    door_material: ContactMaterial,
    door: Door,
    beach: Beach,
    /// The waterline, the shore break and the net's floor, kept for the
    /// readings the level's tests take.
    sea_z: f64,
    floor: f64,
    /// The capsule the optics see: radius and height, metres.
    capsule: (f64, f64),
    which: Player,
    gate: bool,
    /// Where the hand is asked to put the glass, world metres. `None` lets
    /// the arm hang, which is the capsule's whole story and is also what a
    /// hero that has put the lens away would be.
    aim: Option<Vec3>,
    /// How the hero is holding the lens: the arm's lift and swing at the
    /// shoulder, and the turn of the wrist. Three of
    /// [`super::rune::HeroPose`]'s six knobs, live.
    hold: (f64, f64, f64),
}

impl Cove {
    /// Build the cove's bodies on a baked field, with whatever body
    /// `KOSM_RUNE_PLAYER` asks for.
    pub fn new(scene: &CoveScene, sdf: SdfGrid) -> anyhow::Result<Self> {
        Self::with_player(scene, sdf, Player::from_env())
    }

    /// The same, saying which body.
    pub fn with_player(scene: &CoveScene, sdf: SdfGrid, which: Player) -> anyhow::Result<Self> {
        anyhow::ensure!(scene.being_h > 2.0 * scene.being_r, "the being's capsule has no cylinder between its caps");
        let capsule = (scene.being_r, scene.being_h);
        let spec = match which {
            Player::Capsule => BodySpec::capsule(scene.being_r, scene.being_h, GLASS.density),
            Player::Hero => BodySpec::hero(&hero_skeleton(&Rig::DEFAULT)).with_capsule(Some(capsule)),
        }
        .with_friction(BEING_FRICTION)
        .with_speeds(scene.walk_mps, 2.6)
        .with_dt(1e-3);
        let mut body = Body::new(spec);
        // The glass goes in the hand at build time, because the level is
        // about the glass: from here on `Body::held` is where the refractor
        // is and `being_pose` is only a proxy.
        if which == Player::Hero {
            body.hold(Tool::new("lens").with_grip(lens_grip(0.0)));
        }

        // ---- the door ------------------------------------------------------
        // It hangs from its +x edge, at the middle of its thickness, on the
        // sand: a positive hinge angle swings its face out of the cliff and
        // into the cove, which is the way a door opens.
        let (w, h, t) = (scene.door_w, scene.door_h, scene.door_t);
        let hinge = Vec3::new(scene.door_x + w / 2.0, scene.cliff_face_y() + t / 2.0, scene.door_sill());
        let stone = door_substance(scene)?;
        let door_mass = stone.density * w * h * t;
        let door_com = Vec3::new(-w / 2.0, 0.0, h / 2.0);
        let slab = |a: f64, b: f64| door_mass * (a * a + b * b) / 12.0;
        let door_inertia = SpatialInertia::new(door_mass, door_com, Mat3::from_diagonal(&Vec3::new(slab(t, h), slab(w, h), slab(w, t))));
        let mut door_model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
            .dt(1e-3)
            .add_fixed_body("hinge", -1, SpatialTransform::new(Mat3::identity(), hinge), SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01))
            .add_revolute_body("door", 0, SpatialTransform::identity(), door_inertia)
            .build();
        door_model.bodies[1].visuals = vec![GeomInstance::new(Geometry::Box { half_extents: Vec3::new(w / 2.0, t / 2.0, h / 2.0) }, SpatialTransform::new(Mat3::identity(), door_com))];
        let door_joint = door_model.bodies[1].joint_idx;
        // Rotating a slab about its own edge: the centre's `m(w² + t²)/12`
        // carried out to the hinge, half a width away.
        let i_hinge = slab(w, t) + door_mass * (w / 2.0).powi(2);
        {
            let joint = &mut door_model.joints[door_joint];
            joint.limits = Some([0.0, DOOR_LIMIT]);
            joint.limit_stiffness = i_hinge * 30.0 * 30.0;
            joint.limit_damping = 2.0 * i_hinge * 30.0;
        }
        let door = Door {
            body: 1,
            q: door_model.q_offsets[door_joint],
            v: door_model.v_offsets[door_joint],
            k: i_hinge * DOOR_OMEGA * DOOR_OMEGA,
            c: 2.0 * i_hinge * DOOR_OMEGA,
        };
        let door_material = ContactMaterial::default();

        let mut cove = Self {
            body,
            ground: Netted::new(SdfGround(sdf), scene.floor()),
            sea: Water { surface_z: scene.sea_z, density: SEA.fluid().density, current: Vec3::new(0.0, scene.surf, 0.0), drag_cd: WATER_DRAG_CD, spin_damp: WATER_SPIN_DAMP },
            door_state: door_model.default_state(),
            door_cache: ContactCache::new(door_material.margin.max(1e-3)),
            door_material,
            door_model,
            door,
            beach: Beach::of(scene),
            sea_z: scene.sea_z,
            floor: scene.floor(),
            capsule,
            which,
            gate: false,
            aim: None,
            hold: (LENS_AIM_EL, LENS_AIM_AZ, 0.0),
        };
        cove.hold_lens_up();
        cove.place(scene.spawn_x, scene.spawn_y, 0.0);
        Ok(cove)
    }

    /// Which body this cove is walking.
    pub fn player(&self) -> Player {
        self.which
    }

    /// Stand the being on the sand at `(x, y)`, leaning `lean` radians forward
    /// along its facing, at rest.
    ///
    /// The lean here is where the body *is*, which is not the same as
    /// [`Cove::tilt`], where the player has asked it to be: a shove is exactly
    /// the difference between the two, and this is how a test administers one.
    pub fn place(&mut self, x: f64, y: f64, lean: f64) {
        let facing = self.body.facing();
        self.body.place(x, y, self.beach.z_at(x, y), facing, lean);
        self.hold_lens_up();
    }

    /// Point the being. `tests.rs`'s compass, and the window's mouse.
    pub fn face(&mut self, facing: f64) {
        let (p, lean) = (self.body.footing(), 0.0);
        self.body.place(p.x, p.y, self.beach.z_at(p.x, p.y), facing, lean);
        self.hold_lens_up();
    }

    /// Set the lean the player is asking for, radians.
    pub fn set_tilt(&mut self, tilt: f64) {
        self.body.set_tilt(tilt);
    }

    /// One step: the being through its controllers, then the door.
    pub fn step(&mut self, input: &Input) {
        self.hold_lens_up();
        let drive = Drive { aim: self.aim, ..input.drive() };
        self.body.step(&drive, &self.ground, &self.sea, 1e-3);

        // The door: a damper always, so a door let go of stops rather than
        // coasting on a frictionless hinge, and the spring toward open only
        // while the gate is set.
        let (angle, rate) = (self.door_state.q[self.door.q], self.door_state.v[self.door.v]);
        let drive = if self.gate { self.door.k * (DOOR_LIMIT - angle) } else { 0.0 };
        self.door_state.ctrl[self.door.v] = drive - self.door.c * rate;
        step_on(&self.door_model, &mut self.door_state, &Nowhere, &self.door_material, &mut self.door_cache);
    }

    /// Step for `seconds` with nobody at the controls.
    pub fn hold_still(&mut self, seconds: f64) {
        self.run(seconds, &Input::STILL);
    }

    /// Step for `seconds` with the same input held down.
    pub fn run(&mut self, seconds: f64, input: &Input) {
        for _ in 0..(seconds / self.dt()).round().max(0.0) as usize {
            self.step(input);
        }
    }

    /// Where the hand is asked to go, world metres, or `None`.
    ///
    /// Set by [`Cove::hold_lens_up`] at every step, so it moves with the
    /// body. What comes back out of [`Cove::held_lens`] is where the arm
    /// actually got to, which trails this by however much an arm's own PD
    /// trails a target it is chasing.
    pub fn aim(&self) -> Option<Vec3> {
        self.aim
    }

    /// Hold the lens up: the aim, in the body's own frame, at the lift and
    /// swing [`Cove::set_hold`] last asked for.
    ///
    /// **Recomputed every step**, and that is not an optimisation to be
    /// undone later — it is the difference between a pose and a tug of war.
    /// [`kosm::player::Drive::aim`] is a *world* point, so an aim set once at
    /// the spawn is a point the hero walks away from: the arm reaches after
    /// it, the reach torques the body, the body leans, the lean starts a
    /// walk, and the walk moves the shoulder further from the point. The
    /// first run of this walked a hero that had let go of W across four
    /// metres of beach at a metre and a half a second and never stopped. An
    /// aim in the body's own frame is a hero holding a lens up; an aim in the
    /// world's is a hero holding on to one.
    ///
    /// [`Body::orientation`] rather than [`Body::snapshot`] because this runs
    /// at the solver's kilohertz: one quaternion, not thirteen frames of
    /// forward kinematics.
    pub fn hold_lens_up(&mut self) {
        if self.which != Player::Hero {
            return;
        }
        let Some(arm) = self.body.spec().arm else { return };
        let links = &self.body.spec().links;
        let shoulder = links[arm.shoulder].pivot;
        let out = LENS_REACH * (arm.upper + arm.lower);
        let local = shoulder + lens_aim_dir(self.hold.0, self.hold.1) * out - links[0].pivot;
        self.aim = Some(self.body.root() + self.body.orientation().mul_vec(local));
    }

    /// How the hero is holding the lens: the arm's lift and swing at the
    /// shoulder and the turn of the wrist, radians.
    pub fn hold(&self) -> (f64, f64, f64) {
        self.hold
    }

    /// Ask for a different hold. The wrist re-grips at once — the glass is
    /// bolted to the fist and turning it is not a thing an arm has to reach
    /// for — and the arm starts for the new aim on the next step.
    pub fn set_hold(&mut self, aim_el: f64, aim_az: f64, cant: f64) {
        self.hold = (aim_el, aim_az, cant);
        if self.which == Player::Hero {
            self.body.hold(Tool::new("lens").with_grip(lens_grip(cant)));
        }
        self.hold_lens_up();
    }

    /// Stand the hero at a [`super::rune::HeroPose`], holding the lens the
    /// way that pose says, at rest.
    ///
    /// The arithmetic of `rune::hero_lens` says where the glass *will* be;
    /// this is how a body is put where that arithmetic was talking about. The
    /// two are not the same to the millimetre and are not meant to be — an
    /// arm has mass and a body standing with one arm up leans a little into
    /// it — and `rune_tests` says how far apart they are.
    pub fn place_hero(&mut self, pose: &super::rune::HeroPose) {
        self.set_hold(pose.aim_el, pose.aim_az, pose.cant);
        self.body.place(pose.x, pose.y, self.beach.z_at(pose.x, pose.y), pose.yaw, 0.0);
        self.hold_lens_up();
    }

    /// Where the hero is standing and how it is holding the glass, read back
    /// off the body. The inverse of [`Cove::place_hero`], and what the live
    /// hint differentiates at.
    pub fn hero_pose(&self) -> super::rune::HeroPose {
        let p = self.body.footing();
        let (aim_el, aim_az, cant) = self.hold;
        super::rune::HeroPose { x: p.x, y: p.y, yaw: self.body.facing(), aim_el, aim_az, cant }
    }

    /// The hero's trunk, as the opaque solids the rune's little scene needs.
    ///
    /// A lens 220 mm across held beside a head 456 mm across is a lens its
    /// owner can stand in front of, and the day the caustic pass found the
    /// refractor *inside the skull* every one of its photons hit hair before
    /// it hit glass. The score has to know.
    pub fn shadows(&self) -> Vec<super::rune::Piece> {
        match self.which {
            Player::Capsule => Vec::new(),
            Player::Hero => super::rune::occluders(self.body.spec(), &self.body.snapshot().parts),
        }
    }

    /// The rune, live, on whichever body this cove is walking.
    ///
    /// The hero is scored on the glass in its hand with its own head and
    /// trunk in the way ([`super::rune::score_lens_at`]); the capsule is
    /// scored on itself, which is every number the offline solve and the
    /// solvability sweep were measured with.
    pub fn rune_score(&self, scene: &CoveScene, photons: usize) -> f64 {
        match self.lens() {
            Some(lens) => super::rune::score_lens_at(scene, &lens, &self.shadows(), photons).frac,
            None => {
                let (centre, world_to_body) = self.being_pose();
                let axis = world_to_body.transpose().mul_vec(Vec3::z());
                let pose = super::rune::Pose {
                    x: centre.x,
                    y: centre.y,
                    tilt: (-axis.y).atan2(axis.z),
                };
                super::rune::score(scene, &pose, photons).frac
            }
        }
    }

    /// Let the rune drive the door. Step 4 calls this from the score.
    pub fn set_gate(&mut self, open: bool) {
        self.gate = open;
    }

    pub fn gate_open(&self) -> bool {
        self.gate
    }

    /// The hinge angle, radians: 0 closed, [`DOOR_LIMIT`] wide open.
    pub fn door_angle(&self) -> f64 {
        self.door_state.q[self.door.q]
    }

    /// The door's body frame in the world: at the hinge, `rot` world → body.
    pub fn door_transform(&self) -> SpatialTransform {
        forward_kinematics(&self.door_model, &self.door_state).0[self.door.body]
    }

    /// The being's centre and its world → body rotation — the **capsule
    /// proxy**, which is what the rune is traced through.
    pub fn being_pose(&self) -> (Vec3, Mat3) {
        let (centre, r_bw) = self.body.capsule_pose().unwrap_or_else(|| (self.body.root(), Mat3::identity()));
        (centre, r_bw.transpose())
    }

    pub fn being_centre(&self) -> Vec3 {
        self.being_pose().0
    }

    /// The being's own axis, from foot to head, in world.
    pub fn being_axis(&self) -> Vec3 {
        self.body.axis()
    }

    /// How far the being is off the vertical, radians.
    pub fn lean(&self) -> f64 {
        self.body.lean()
    }

    /// The being's velocity in world axes, m/s.
    pub fn being_velocity(&self) -> Vec3 {
        self.body.centre_velocity()
    }

    /// Its horizontal speed, m/s: what `walk_mps` caps.
    pub fn walking_speed(&self) -> f64 {
        self.body.speed()
    }

    /// Where the being is looking, radians, zero along `+x`.
    pub fn facing(&self) -> f64 {
        self.body.facing()
    }

    /// The lean the player is asking for, radians.
    pub fn tilt(&self) -> f64 {
        self.body.tilt()
    }

    /// Where the being's centre rests standing at `(x, y)` with no lean.
    pub fn resting_centre(&self, x: f64, y: f64) -> Vec3 {
        Vec3::new(x, y, self.beach.z_at(x, y) + self.capsule.1 / 2.0)
    }

    /// How much of the being is under water, 0 on dry sand and 1 with its head
    /// gone.
    pub fn submerged(&self) -> f64 {
        let (r, h) = self.capsule;
        let half = (h - 2.0 * r) / 2.0;
        let rise = self.being_axis().z.abs() * half;
        let depth = (self.sea_z - (self.being_centre().z - rise - r)).clamp(0.0, 2.0 * (rise + r));
        kosm::player::medium::wetted(r, 2.0 * rise, depth).0 / kosm::player::medium::capsule_volume(r, 2.0 * half)
    }

    /// How deep the water is where the being is standing, metres: the
    /// waterline less the sand under its feet. Negative on dry sand.
    pub fn wading_depth(&self) -> f64 {
        let p = self.being_centre();
        self.sea_z - self.beach.z_at(p.x, p.y)
    }

    /// How many steps the net under the level has carried. Zero, on a cove
    /// that is closed.
    pub fn net_caught(&self) -> usize {
        self.ground.caught()
    }

    /// The floor the net sits at, metres.
    pub fn net_floor(&self) -> f64 {
        self.floor
    }

    pub fn time(&self) -> f64 {
        self.body.time()
    }

    pub fn dt(&self) -> f64 {
        self.body.dt()
    }

    pub fn mass(&self) -> f64 {
        self.body.mass()
    }

    /// The upright spring, N·m/rad and N·m·s/rad. Reported, not tuned.
    pub fn upright_spring(&self) -> (f64, f64) {
        self.body.upright_spring()
    }

    /// Where the being is looking, as a unit vector on the sand's plane.
    pub fn facing_dir(&self) -> Vec3 {
        self.body.facing_dir()
    }

    /// The being's right: `facing × ẑ`, so `strafe = 1` walks to its right.
    pub fn right_dir(&self) -> Vec3 {
        self.body.right_dir()
    }

    /// Where the held lens is, if the hero is holding one.
    ///
    /// Not where the arm was *asked* to put it — [`Cove::aim`] is that — but
    /// where it actually got to, because an arm has mass. This is the pose
    /// [`super::rune::score_lens`] traces the live gate through and the pose
    /// `render.rs` draws the glass at, so the number and the picture are the
    /// same piece of glass at the same moment.
    ///
    /// The lens's own frame: `+z` is the optical axis, which is what
    /// `hero/kit.rs` cuts it about.
    pub fn held_lens(&self) -> Option<Pose> {
        self.body.held().map(|(pose, _)| pose)
    }

    /// The held lens as the scorer's refractor: its centre and its optical
    /// axis, world metres.
    pub fn lens(&self) -> Option<super::rune::Held> {
        let pose = self.held_lens()?;
        Some(super::rune::Held { centre: pose.pos, axis: pose.rot.mul_vec(Vec3::z()) })
    }

    /// The hero's parts, placed, for the renderer. Empty for the capsule.
    pub fn hero_parts(&self) -> Vec<Part> {
        match self.which {
            Player::Capsule => Vec::new(),
            Player::Hero => self.body.snapshot().parts,
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        let (centre, world_to_body) = self.being_pose();
        let v = self.body.velocity();
        let snap = self.body.snapshot();
        Snapshot {
            t: self.time(),
            dt: self.dt(),
            being: (centre, world_to_body),
            being_vel: (v, snap.root_vel.1),
            facing: snap.facing,
            tilt: snap.tilt,
            door_angle: self.door_angle(),
            door: self.door_transform(),
            gate_open: self.gate,
            held: snap.held,
            parts: (self.which == Player::Hero).then(|| Arc::new(snap.parts)),
        }
    }
}
