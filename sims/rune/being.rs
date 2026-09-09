//! The being and the door: the two things in the cove that move.
//!
//! The being is a capsule of glass on one phyz free joint — the marble of
//! [`super::sim`] grown up — standing on the baked field through the same SDF
//! contact path ([`sim::step_on_sdf`]). It has no legs and no limbs. What
//! holds it up is a *spring*, not a constraint: a torque on the free joint
//! proportional to the angle between its own axis and the axis the player has
//! commanded, with damping. Shove it and it tips; let go and it comes back.
//! That is the whole difference between a weeble and a ragdoll, and it is one
//! torque.
//!
//! The sea is the third thing, and it does not move so much as it *is* moving:
//! wading out, the being meets buoyancy, form drag and a shore break running
//! shoreward, all of it in [`Cove::water`] and all of it through the same
//! `ctrl` the walk goes in by. What comes out of that is the cove's −y edge:
//! the water gets heavier with every step until, somewhere between the waist
//! and the chest, it wins — and then it walks you back out. No wall, no
//! trigger, and nowhere to drown.
//!
//! The door is a slab of stone on a revolute joint at its own vertical edge,
//! hinged into the cliff, with a soft limit at [`DOOR_LIMIT`] and a spring
//! that drives it open only while the gate is set. It is a body with mass: it
//! swings, it does not teleport. Step 4 sets the gate from the rune's score;
//! here it is set by hand.
//!
//! Everything the dynamics needs from the player arrives as [`Input`], and
//! everything the picture needs comes back as [`Snapshot`], the way the
//! court's does.
//!
//! Metres, radians, seconds; z up.

use std::f64::consts::PI;
use std::sync::LazyLock;

use kosm_scan::SdfGrid;
use phyz_contact::{ContactCache, ContactMaterial};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3, quat_exp};
use phyz_model::{GeomInstance, Geometry, Model, ModelBuilder, State};
use phyz_rigid::forward_kinematics;

use super::CoveScene;
use super::sim::{self, Beach, GLASS};
use kosm::material::{self, Material as Substance};

/// The being is body 0, on the free joint the whole model starts with.
pub const BEING: usize = 0;
/// The free joint's DOF order is angular first: `q = [w(3), pos(3)]` and
/// `v = [angular(3), linear(3)]`, both in the *body* frame for the velocity.
const ANG: usize = 0;
const POS: usize = 3;

/// How far the player may lean the being, radians. Ten degrees is the design's
/// "a few degrees about its own horizontal axis": enough to walk the caustic
/// across the door, not enough to fall over.
pub const TILT_MAX: f64 = 10.0 * PI / 180.0;

/// How fast the upright spring brings the being back, rad/s.
///
/// The being is an inverted pendulum about its feet, so the spring has to buy
/// back gravity before it buys any stiffness of its own — see
/// [`Being::upright_gains`]. Asking for a critically damped `ω_n` here, the
/// free response to a shove is `θ(t) = θ₀ (1 + ω_n t) e^{−ω_n t}`: it never
/// crosses zero (so it does not oscillate, and there is nothing to tune away)
/// and it is inside a twentieth of the shove by `ω_n t ≈ 4.7`. At 6 rad/s that
/// is 0.8 s from 20° to under 1°, which is the "about a second" the design
/// asks for and is slow enough to *see*.
pub const UPRIGHT_OMEGA: f64 = 6.0;

/// The walking force, as an acceleration: `F = WALK_GAIN · m · direction`.
///
/// It has to beat the sand before it moves the being at all — the being is
/// driven by a force at its centre of mass, not by feet that grip, so nothing
/// happens until the force passes `μ m g`. With [`BEING_FRICTION`] that floor
/// is 3.4 m/s², and what is left over is the acceleration the player feels:
/// 0.4 s from a standstill to `walk_mps`.
pub const WALK_GAIN: f64 = 6.0;

/// Glass on wet sand, and the cove's own number rather than the library's.
///
/// Deliberately far below [`sim::SAND_FRICTION`], which is what the *marble*
/// rolls on: a rolling sphere needs grip and a sliding being needs to slide.
/// Neither is `kosm::material`'s sand — `dry sand` is μ = 0.55 and its wetted
/// form 0.6875 — because neither is a measurement. They are the two ends the
/// level wants a body to sit at: 0.8 so the marble cannot slip and the roll
/// measures the field, 0.35 so the being slides at a speed a player can steer.
/// The being's own material wins for its terrain contacts — a contact against
/// the ground combines with nothing — so the two coexist in one scene.
pub const BEING_FRICTION: f64 = 0.35;

/// How long the being takes to stop once the player lets go, seconds. The
/// damping is `−m v_h / τ`; the sand takes over below what friction can hold.
pub const STOP_TAU: f64 = 0.15;

/// The sea: `sea water` out of `kosm::material`, 1025 kg/m^3.
///
/// Glass ([`sim::GLASS`]) is two and a half times as dense, so a being in the
/// sea gets lighter and keeps its feet: it never floats off the bed, and there
/// is nothing to swim. This was a `1000.0` written here; fresh water is not what
/// the cove is full of, and the library has the substance that is.
///
/// The two and a half per cent it adds is two and a half per cent more buoyancy
/// and two and a half per cent more drag, and it stops the being one centimetre
/// sooner: `tests.rs` walks it out to sea and the water wins at **0.58 m with
/// 60 % of the being under**, where fresh water's 1000 kg/m^3 gave 0.59 m and
/// 62 %. Waist to chest either way, which is the design's own tolerance.
pub static SEA: LazyLock<Substance> = LazyLock::new(|| material::named("sea water").expect("sea water is in kosm's material library"));

/// The drag coefficient of a bluff body in water. A capsule broadside is about
/// a cylinder, which is about one.
pub const WATER_DRAG_CD: f64 = 1.0;

/// How much of the upright spring's own damping the water adds when the being
/// is fully under, as a fraction. A quarter is "a little": the recovery from a
/// shove goes from critically damped to visibly sluggish without the spring
/// ever losing the argument.
pub const WATER_SPIN_DAMP: f64 = 0.25;

/// How far the door swings before the recess stops it, radians.
pub const DOOR_LIMIT: f64 = 100.0 * PI / 180.0;

/// How fast the door swings, rad/s: critically damped at the limit, so the
/// angle rises monotonically and settles on it instead of slamming into it.
/// At 3 rad/s the swing is 1.6 s of the 3 s the rune holds it open for.
pub const DOOR_OMEGA: f64 = 3.0;

/// The door's substance, read off the door body the level built.
///
/// `scene.rs` says `door.material("granite")` and that is the whole statement:
/// the density this returns is `kosm::material`'s granite, and the day the door
/// is cut out of basalt instead there is one word to change and no constant in
/// this file to keep in step with it.
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
}

/// The cove at one instant, as the picture wants it.
///
/// Shaped like `court::render::Snapshot`: poses and velocities, nothing that
/// needs the simulation to still be alive. The renderer places the being's
/// capsule and the door's slab from this and nothing else.
#[derive(Clone, Copy, Debug)]
pub struct Snapshot {
    /// The simulation's own clock, seconds.
    pub t: f64,
    /// The solver's step, seconds.
    pub dt: f64,
    /// The being's centre and its **world → body** rotation, phyz's convention
    /// (so `rot.transpose()` is the way out to world).
    pub being: (Vec3, Mat3),
    /// The being's linear and angular velocity in **world** axes, m/s and
    /// rad/s. phyz keeps the free joint's in body axes; the picture wants the
    /// world's.
    pub being_vel: (Vec3, Vec3),
    /// Where the being is looking and how far it is leaning, radians.
    pub facing: f64,
    pub tilt: f64,
    /// The hinge angle, radians: 0 closed, [`DOOR_LIMIT`] wide open.
    pub door_angle: f64,
    /// The door's body frame — at the hinge, `rot` world → body.
    pub door: SpatialTransform,
    pub gate_open: bool,
}

/// The being's numbers, resolved once from the scene.
#[derive(Clone, Copy, Debug)]
struct Being {
    r: f64,
    /// Half the distance between the cap centres: the capsule's `length / 2`.
    half: f64,
    mass: f64,
    /// Mass and inertia about the capsule's centre, which is its centre of
    /// mass: the model wants it, and so does the spring.
    inertia: SpatialInertia,
    /// The upright spring and its damping, N·m/rad and N·m·s/rad.
    k: f64,
    c: f64,
    walk_mps: f64,
}

impl Being {
    /// The capsule's mass and inertia about its centre, from the glass alone:
    /// a cylinder of length `2·half` and two hemispherical caps.
    fn new(r: f64, height: f64, walk_mps: f64) -> Self {
        let half = (height - 2.0 * r) / 2.0;
        let rho = GLASS.density;
        let (m_cyl, m_cap) = (rho * PI * r * r * 2.0 * half, rho * 2.0 / 3.0 * PI * r * r * r);
        let mass = m_cyl + 2.0 * m_cap;
        // Transverse inertia about the capsule's centre. The cylinder is the
        // textbook `m(3r² + L²)/12`; a cap is a hemisphere whose own centre of
        // mass sits `3r/8` past the cap centre, so shifting its `2/5 m r²`
        // (which is about the *sphere's* centre) out to the capsule's centre
        // through its own centre of mass leaves `m(2r²/5 + L²/4 + 3Lr/8)`.
        let l = 2.0 * half;
        let i_t = m_cyl * (3.0 * r * r + l * l) / 12.0 + 2.0 * m_cap * (0.4 * r * r + l * l / 4.0 + 3.0 * l * r / 8.0);
        // About its own axis the capsule is a cylinder plus two hemispheres
        // and nothing has to be shifted: the axis passes through both.
        let i_a = 0.5 * m_cyl * r * r + 2.0 * 0.4 * m_cap * r * r;
        let inertia = SpatialInertia::new(mass, Vec3::zeros(), Mat3::from_diagonal(&Vec3::new(i_t, i_t, i_a)));
        let (k, c) = Self::upright_gains(mass, i_t, height);
        Self { r, half, mass, inertia, k, c, walk_mps }
    }

    /// The upright spring's stiffness and damping, from the being's own
    /// numbers rather than from tuning.
    ///
    /// Standing, the being is an inverted pendulum about its feet: inertia
    /// `I_p = I_transverse + m (h/2)²` about that pivot, and gravity is a
    /// *negative* spring of `m g h/2` about it, because a body tipped past
    /// vertical keeps going. So the spring has to pay gravity back before it
    /// buys any stiffness of its own, and a critically damped recovery at
    /// [`UPRIGHT_OMEGA`] is
    ///
    /// ```text
    /// k = I_p ω_n² + m g h/2      c = 2 I_p ω_n
    /// ```
    ///
    /// For the authored being (350 mm by 1.4 m of N-BK7, 1127 kg) that is
    /// 33.6 kN·m/rad and 8.6 kN·m·s/rad. The far side of the trade is that a
    /// stiffer spring stands straighter on a slope: the ground's reaction acts
    /// under the *foot*, which on a beach of grade `s` is `r·s` uphill of the
    /// centre, so the being settles a hair downhill of vertical at
    /// `m g (L/2) s / (k − m g L/2)` — 0.4° here, and inversely proportional to
    /// `k`. Standing at that angle instead of straight up costs the centre
    /// 5 mm, which is what the standing test's centimetre of drift is spent on.
    fn upright_gains(mass: f64, i_transverse: f64, height: f64) -> (f64, f64) {
        let i_pivot = i_transverse + mass * (height / 2.0).powi(2);
        let k = i_pivot * UPRIGHT_OMEGA * UPRIGHT_OMEGA + mass * GRAVITY * height / 2.0;
        let c = 2.0 * i_pivot * UPRIGHT_OMEGA;
        (k, c)
    }
}

/// The door's numbers and where in the state its one DOF lives.
#[derive(Clone, Copy, Debug)]
struct Door {
    body: usize,
    /// The hinge angle's index into `q`, and its rate's into `v`. A revolute
    /// joint has one of each and they are the same number here, but they are
    /// two different vectors and are kept apart.
    q: usize,
    v: usize,
    /// The drive toward open, N·m/rad and N·m·s/rad.
    k: f64,
    c: f64,
}

/// The cove, stepping: the being, the door, the field they stand on.
pub struct Cove {
    pub model: Model,
    pub state: State,
    /// The baked field the being's feet read. Owned, because a `Cove` is what
    /// the window hands to its simulation thread.
    pub sdf: SdfGrid,
    /// Where the being is looking: yaw about z, radians, zero along +x.
    pub facing: f64,
    /// The lean the player is *asking* for, radians, positive forward along
    /// the facing. Where the body actually is is another question, and the
    /// answer is [`Cove::being_axis`].
    pub tilt: f64,
    beach: Beach,
    being: Being,
    door: Door,
    /// The flat waterline. The *rendered* sea has a swell on it; the physics
    /// uses the plane, because a being that bobbed with a 30 mm sine wave would
    /// be reporting the renderer's authored decoration as a force.
    sea_z: f64,
    /// How fast the shore break runs shoreward, m/s. See [`Cove::water`].
    surf: f64,
    /// The net: a plane at the bake volume's floor, under everything.
    floor: f64,
    /// How many steps the net has carried. It should stay at zero.
    net_caught: usize,
    material: ContactMaterial,
    cache: ContactCache,
    gate: bool,
}

impl Cove {
    /// Build the cove's bodies on a baked field: the being at the scene's
    /// spawn, standing on the sand, and the door closed in the cliff.
    pub fn new(scene: &CoveScene, sdf: SdfGrid) -> anyhow::Result<Self> {
        let being = Being::new(scene.being_r, scene.being_h, scene.walk_mps);
        anyhow::ensure!(being.half > 0.0, "the being's capsule has no cylinder between its caps");

        // The door hangs from its +x edge, at the middle of its thickness, on
        // the sand: that way a positive hinge angle swings its face out of the
        // cliff and into the cove, which is the way a door opens.
        let (w, h, t) = (scene.door_w, scene.door_h, scene.door_t);
        let hinge = Vec3::new(scene.door_x + w / 2.0, scene.cliff_face_y() + t / 2.0, scene.door_sill());
        // The door's substance is the door body's, not a `STONE_DENSITY` of
        // this module's: `scene.rs` authors the slab as granite and
        // `BuiltBody::substance` hands the constants back here. A two-tonne
        // slab is the only thing the mass decides — the swing is critically
        // damped in *its own* inertia (`DOOR_OMEGA`), so a denser door swings
        // the same and only pushes harder on the limit.
        let stone = door_substance(scene)?;
        let door_mass = stone.density * w * h * t;
        let door_com = Vec3::new(-w / 2.0, 0.0, h / 2.0);
        let slab = |a: f64, b: f64| door_mass * (a * a + b * b) / 12.0;
        let door_inertia = SpatialInertia::new(door_mass, door_com, Mat3::from_diagonal(&Vec3::new(slab(t, h), slab(w, h), slab(w, t))));

        let mut model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
            .dt(1e-3)
            .add_free_body("being", -1, SpatialTransform::identity(), being.inertia)
            .add_fixed_body("hinge", -1, SpatialTransform::new(Mat3::identity(), hinge), SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01))
            .add_revolute_body("door", 1, SpatialTransform::identity(), door_inertia)
            .build();

        model.bodies[BEING].geometry = Some(Geometry::Capsule { radius: being.r, length: 2.0 * being.half });
        model.bodies[BEING].material = Some(ContactMaterial { friction: BEING_FRICTION, restitution: 0.0, ..Default::default() });
        // The slab is the picture's, not the solver's. The door stands inside
        // the cliff's recess, and the cliff is *in* the baked field, so a
        // collidable door would be born penetrating the walls of its own
        // doorway and would spend the level being pushed out of them. Nothing
        // in this slice touches the door but the sun.
        model.bodies[2].visuals = vec![GeomInstance::new(Geometry::Box { half_extents: Vec3::new(w / 2.0, t / 2.0, h / 2.0) }, SpatialTransform::new(Mat3::identity(), door_com))];

        let door_joint = model.bodies[2].joint_idx;
        // Rotating a slab about its own edge: the centre's `m(w² + t²)/12`
        // carried out to the hinge, half a width away.
        let i_hinge = slab(w, t) + door_mass * (w / 2.0).powi(2);
        {
            let joint = &mut model.joints[door_joint];
            joint.limits = Some([0.0, DOOR_LIMIT]);
            // The default limit is written for a unit inertia; a two-tonne
            // slab would walk straight through it. Thirty radians a second is
            // a hard stop that is still a hundred steps of the solver's 1 ms.
            joint.limit_stiffness = i_hinge * 30.0 * 30.0;
            joint.limit_damping = 2.0 * i_hinge * 30.0;
        }
        let door = Door {
            body: 2,
            q: model.q_offsets[door_joint],
            v: model.v_offsets[door_joint],
            k: i_hinge * DOOR_OMEGA * DOOR_OMEGA,
            c: 2.0 * i_hinge * DOOR_OMEGA,
        };

        let material = ContactMaterial { friction: sim::SAND_FRICTION, restitution: 0.0, ..Default::default() };
        let mut cove = Self {
            state: model.default_state(),
            model,
            sdf,
            facing: 0.0,
            tilt: 0.0,
            beach: Beach::of(scene),
            being,
            door,
            sea_z: scene.sea_z,
            surf: scene.surf,
            floor: scene.floor(),
            net_caught: 0,
            cache: ContactCache::new(material.margin.max(1e-3)),
            material,
            gate: false,
        };
        cove.place(scene.spawn_x, scene.spawn_y, 0.0);
        Ok(cove)
    }

    /// Stand the being on the sand at `(x, y)`, leaning `lean` radians forward
    /// along its facing, at rest.
    ///
    /// The lean here is where the body *is*, which is not the same as
    /// [`Cove::tilt`], where the player has asked it to be: a shove is exactly
    /// the difference between the two, and this is how a test administers one.
    pub fn place(&mut self, x: f64, y: f64, lean: f64) {
        let (axis, w) = self.leaned(lean);
        let centre = self.beach.resting_centre(x, y, self.being.r) + axis * self.being.half;
        for (i, v) in [w.x, w.y, w.z, centre.x, centre.y, centre.z].into_iter().enumerate() {
            self.state.q[i] = v;
        }
        for i in 0..6 {
            self.state.v[i] = 0.0;
        }
    }

    /// The body axis and the free joint's exponential coordinates for a lean
    /// of `lean` radians forward along the current facing.
    ///
    /// Leaning forward turns `ẑ` toward the facing, which is a rotation about
    /// `ẑ × facing` — the being's *left*, as it must be for a nod.
    fn leaned(&self, lean: f64) -> (Vec3, Vec3) {
        let axis_of_rotation = Vec3::z().cross(self.facing_dir());
        let w = axis_of_rotation * lean;
        (quat_exp(&w).to_matrix().mul_vec(Vec3::z()), w)
    }

    /// One step of the solver: the player's intent, then the contact step.
    ///
    /// The intent is three torques and a force, written into `state.ctrl`,
    /// which a model with no actuators reads as generalized forces per DOF.
    /// The free joint's motion subspace is the identity *in the body frame*,
    /// so its six entries are a torque about the being's own axes and a force
    /// at its own centre, and everything the player asks for in world axes is
    /// rotated into the body's before it is written. Gravity is the model's.
    pub fn step(&mut self, input: &Input) {
        self.facing = wrap(self.facing + input.yaw_delta);
        self.tilt = (self.tilt + input.tilt_delta).clamp(-TILT_MAX, TILT_MAX);

        let r_bw = self.body_to_world();
        let axis = r_bw.mul_vec(Vec3::z());
        let (facing_dir, right) = (self.facing_dir(), self.right_dir());
        let omega = r_bw.mul_vec(self.angular_velocity_body());
        let velocity = r_bw.mul_vec(self.linear_velocity_body());

        // The upright spring. The error is the rotation that takes the being's
        // own axis onto the commanded one — world z leaned by `tilt` — as a
        // rotation vector, so it stays honest at 20° where a small-angle
        // cross product does not. Damping is on the whole angular velocity,
        // which also keeps the capsule from spinning about its own axis: a
        // symmetric capsule has no yaw of its own, and the being's heading is
        // `facing`, a number, not a body rotation.
        let target = (Vec3::z() * self.tilt.cos() + facing_dir * self.tilt.sin()).normalize();
        let cross = axis.cross(target);
        let sin = cross.norm();
        let error = if sin > 1e-12 { cross * (sin.atan2(axis.dot(target)) / sin) } else { Vec3::zeros() };
        let torque = error * self.being.k - omega * self.being.c;

        // Walking: a horizontal force at the centre of mass, cut off once the
        // being is already going that fast in that direction, and a mild pull
        // toward standstill when the player is asking for nothing.
        let commanded = facing_dir * input.forward + right * input.strafe;
        let n = commanded.norm();
        let force = if n > 1e-9 {
            // A held W and a held D are two full pushes, and unclamped they
            // would walk √2 faster along the diagonal than along either one.
            let unit = commanded / n;
            let along = Vec3::new(velocity.x, velocity.y, 0.0).dot(unit);
            let drive = if n > 1.0 { unit } else { commanded };
            if along < self.being.walk_mps { drive * (WALK_GAIN * self.being.mass) } else { Vec3::zeros() }
        } else {
            Vec3::new(velocity.x, velocity.y, 0.0) * (-self.being.mass / STOP_TAU)
        };

        // The sea, if the being is standing in it. Its own function, and the
        // only thing it needs from above is where the being is and how fast.
        let (water_force, water_torque) = self.water(self.being_centre(), axis, velocity, omega);
        let (torque, force) = (torque + water_torque, force + water_force);

        let (torque, force) = (r_bw.transpose().mul_vec(torque), r_bw.transpose().mul_vec(force));
        for (i, v) in [torque.x, torque.y, torque.z, force.x, force.y, force.z].into_iter().enumerate() {
            self.state.ctrl[i] = v;
        }

        // The door: a damper always, so a door let go of stops rather than
        // coasting on a frictionless hinge, and the spring toward open only
        // while the gate is set.
        let (angle, rate) = (self.state.q[self.door.q], self.state.v[self.door.v]);
        let drive = if self.gate { self.door.k * (DOOR_LIMIT - angle) } else { 0.0 };
        self.state.ctrl[self.door.v] = drive - self.door.c * rate;

        if sim::step_on_sdf_over(&self.model, &mut self.state, &self.sdf, &self.material, &mut self.cache, Some(self.floor)) {
            self.net_caught += 1;
        }
    }

    /// The sea's forces on the being, in world axes: buoyancy, drag, and the
    /// spin the water takes out of a shove. Zero everywhere the sand is dry.
    ///
    /// **Buoyancy** is Archimedes on whatever of the capsule is under
    /// [`Cove::sea_z`]: `ρ_water · V_submerged · g`, up. Glass is two and a half
    /// times sea water, so at full submersion this is forty per cent of the
    /// being's weight — enough to make it light on its feet, never enough to
    /// take them off the bed.
    ///
    /// **Drag** is the one asked for, `½ ρ C_d A |v| v`, on the submerged
    /// silhouette — with the important word being **relative**. The sea is not
    /// still. The swell the picture already rolls up the beach is water
    /// *moving*: a shore break runs shoreward at [`Cove::surf`], and what the
    /// drag law sees is the being's velocity through that water, not through
    /// the ground. Standing still with the water at your chest is then a
    /// several-kilonewton push toward the sand, and wading further out is
    /// wading up a river.
    ///
    /// That asymmetry is the whole design, and it is why the drag is written
    /// against a current instead of against still water:
    ///
    /// - Form drag in still water cannot stop this being. At C_d = 1 and a
    ///   walking pace the sea is worth about two hundred newtons against a
    ///   walk force of four and a half kilonewtons; a coefficient large enough
    ///   to hold the being still would be a wall with a number painted on it.
    /// - Neither can anything that resists *both* ways — more friction, less
    ///   purchase, a thicker medium. Buoyancy already takes the bed's grip away
    ///   faster than it takes the being's weight away, so any symmetric
    ///   resistance strong enough to stop the being walking in is also strong
    ///   enough to stop it walking back out, and the sea becomes a hole you
    ///   drown in slowly. There is no dying in this game.
    ///
    /// The current does both jobs with one force: it stalls you on the way out
    /// and it carries you on the way in. Let go of the controls at the deep end
    /// and the sea walks you back up the beach.
    ///
    /// **The spin** is [`WATER_SPIN_DAMP`] of the upright spring's own damping,
    /// in proportion to how much of the being is under: water damps a wobble.
    fn water(&self, centre: Vec3, axis: Vec3, velocity: Vec3, omega: Vec3) -> (Vec3, Vec3) {
        let (r, half) = (self.being.r, self.being.half);
        // The capsule stands within ten degrees of vertical, so its wetted
        // profile is the upright one to a part in seventy: the barrel's span
        // along z, and a cap of radius `r` at each end of it.
        let rise = axis.z.abs() * half;
        let foot = centre.z - rise - r;
        let depth = (self.sea_z - foot).clamp(0.0, 2.0 * (rise + r));
        if depth <= 0.0 {
            return (Vec3::zeros(), Vec3::zeros());
        }
        let (volume, area) = wetted(r, 2.0 * rise, depth);
        let submerged = volume / capsule_volume(r, 2.0 * half);

        // The fluid facet of the substance in [`SEA`]: both of these forces are
        // one density, and it is the sea's, not a number of this module's.
        let rho = SEA.fluid().density;

        let up = Vec3::z() * (rho * volume * GRAVITY);
        // Shoreward is +y: the waterline is the cove's -y edge.
        let through_water = velocity - Vec3::new(0.0, self.surf, 0.0);
        let drag = through_water * (-0.5 * rho * WATER_DRAG_CD * area * through_water.norm());
        let spin = omega * (-WATER_SPIN_DAMP * self.being.c * submerged);
        (up + drag, spin)
    }

    /// Step for `seconds` with nobody at the controls.
    pub fn hold_still(&mut self, seconds: f64) {
        self.run(seconds, &Input::STILL);
    }

    /// Step for `seconds` with the same input held down.
    pub fn run(&mut self, seconds: f64, input: &Input) {
        for _ in 0..(seconds / self.model.dt).round().max(0.0) as usize {
            self.step(input);
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
        self.state.q[self.door.q]
    }

    /// The door's body frame in the world: at the hinge, `rot` world → body.
    pub fn door_transform(&self) -> SpatialTransform {
        forward_kinematics(&self.model, &self.state).0[self.door.body]
    }

    /// The being's centre and its world → body rotation.
    pub fn being_pose(&self) -> (Vec3, Mat3) {
        (self.being_centre(), self.body_to_world().transpose())
    }

    pub fn being_centre(&self) -> Vec3 {
        Vec3::new(self.state.q[POS], self.state.q[POS + 1], self.state.q[POS + 2])
    }

    /// The being's own axis, from foot to head, in world.
    pub fn being_axis(&self) -> Vec3 {
        self.body_to_world().mul_vec(Vec3::z())
    }

    /// How far the being is off the vertical, radians. What a shove leaves
    /// behind and what the upright spring spends itself on.
    pub fn lean(&self) -> f64 {
        self.being_axis().dot(Vec3::z()).clamp(-1.0, 1.0).acos()
    }

    /// The being's velocity in world axes, m/s.
    pub fn being_velocity(&self) -> Vec3 {
        self.body_to_world().mul_vec(self.linear_velocity_body())
    }

    /// Its horizontal speed, m/s: what `walk_mps` caps.
    pub fn walking_speed(&self) -> f64 {
        let v = self.being_velocity();
        v.x.hypot(v.y)
    }

    /// Where the being's centre rests standing at `(x, y)` with no lean.
    pub fn resting_centre(&self, x: f64, y: f64) -> Vec3 {
        self.beach.resting_centre(x, y, self.being.r) + Vec3::z() * self.being.half
    }

    /// How much of the being is under water, 0 on dry sand and 1 with its head
    /// gone. What the water forces are proportional to, and what a test reads
    /// to say how deep the sea let the player wade.
    pub fn submerged(&self) -> f64 {
        let (r, half) = (self.being.r, self.being.half);
        let rise = self.being_axis().z.abs() * half;
        let depth = (self.sea_z - (self.being_centre().z - rise - r)).clamp(0.0, 2.0 * (rise + r));
        wetted(r, 2.0 * rise, depth).0 / capsule_volume(r, 2.0 * half)
    }

    /// How deep the water is where the being is standing, metres: the waterline
    /// less the sand under its feet. Negative on dry sand.
    pub fn wading_depth(&self) -> f64 {
        let p = self.being_centre();
        self.sea_z - self.beach.z_at(p.x, p.y)
    }

    /// How many steps the net under the level has carried. Zero, on a cove that
    /// is closed: see [`sim::step_on_sdf_over`].
    pub fn net_caught(&self) -> usize {
        self.net_caught
    }

    /// The floor the net sits at, metres.
    pub fn net_floor(&self) -> f64 {
        self.floor
    }

    pub fn time(&self) -> f64 {
        self.state.time
    }

    pub fn dt(&self) -> f64 {
        self.model.dt
    }

    pub fn mass(&self) -> f64 {
        self.being.mass
    }

    /// The upright spring, N·m/rad and N·m·s/rad. Reported, not tuned: see
    /// [`Being::upright_gains`].
    pub fn upright_spring(&self) -> (f64, f64) {
        (self.being.k, self.being.c)
    }

    /// Where the being is looking, as a unit vector on the sand's plane.
    pub fn facing_dir(&self) -> Vec3 {
        let (s, c) = self.facing.sin_cos();
        Vec3::new(c, s, 0.0)
    }

    /// The being's right: `facing × ẑ`, so `strafe = 1` walks to its right.
    pub fn right_dir(&self) -> Vec3 {
        self.facing_dir().cross(Vec3::z())
    }

    pub fn snapshot(&self) -> Snapshot {
        let r_bw = self.body_to_world();
        Snapshot {
            t: self.state.time,
            dt: self.model.dt,
            being: (self.being_centre(), r_bw.transpose()),
            being_vel: (r_bw.mul_vec(self.linear_velocity_body()), r_bw.mul_vec(self.angular_velocity_body())),
            facing: self.facing,
            tilt: self.tilt,
            door_angle: self.door_angle(),
            door: self.door_transform(),
            gate_open: self.gate,
        }
    }

    /// The being's body → world rotation, straight from the free joint's
    /// exponential coordinates. (`forward_kinematics` would say the same thing
    /// for a root body; this is the half of it that is wanted every step.)
    fn body_to_world(&self) -> Mat3 {
        quat_exp(&Vec3::new(self.state.q[ANG], self.state.q[ANG + 1], self.state.q[ANG + 2])).to_matrix()
    }

    fn angular_velocity_body(&self) -> Vec3 {
        Vec3::new(self.state.v[ANG], self.state.v[ANG + 1], self.state.v[ANG + 2])
    }

    fn linear_velocity_body(&self) -> Vec3 {
        Vec3::new(self.state.v[POS], self.state.v[POS + 1], self.state.v[POS + 2])
    }
}

/// A capsule's volume: a barrel of length `l` and a sphere of radius `r`.
fn capsule_volume(r: f64, l: f64) -> f64 {
    PI * r * r * l + 4.0 / 3.0 * PI * r * r * r
}

/// What is under water, for an upright capsule of radius `r` and barrel length
/// `l` whose foot is `depth` below the surface: the submerged volume, and the
/// submerged area of its side-on silhouette.
///
/// Both are the same integral up the capsule with a different integrand — a
/// disc of the local radius for the volume, the local width for the silhouette
/// — and both are exact rather than sampled, so buoyancy is continuous as the
/// being wades and the drag does not step. Above the barrel the capsule is its
/// own mirror image, so the far half is "everything, less what is still dry".
fn wetted(r: f64, l: f64, depth: f64) -> (f64, f64) {
    fn below(r: f64, l: f64, d: f64) -> (f64, f64) {
        if d <= 0.0 {
            (0.0, 0.0)
        } else if d <= r {
            // a spherical cap of height `d`, and the circular segment under it
            let t = d - r;
            (PI * d * d * (3.0 * r - d) / 3.0, t * (r * r - t * t).sqrt() + r * r * ((t / r).asin() + PI / 2.0))
        } else if d <= r + l {
            (2.0 / 3.0 * PI * r * r * r + PI * r * r * (d - r), PI * r * r / 2.0 + 2.0 * r * (d - r))
        } else {
            let (v, a) = below(r, l, 2.0 * r + l - d);
            (capsule_volume(r, l) - v, PI * r * r + 2.0 * r * l - a)
        }
    }
    let (whole_v, whole_a) = (capsule_volume(r, l), PI * r * r + 2.0 * r * l);
    let (v, a) = below(r, l, depth.min(2.0 * r + l));
    (v.clamp(0.0, whole_v), a.clamp(0.0, whole_a))
}

/// Wrap an angle into `(-π, π]`, so a facing that has been spun a hundred
/// times reads the same as one that has not.
fn wrap(a: f64) -> f64 {
    let two = 2.0 * PI;
    let r = a - two * ((a + PI) / two).floor();
    if r <= -PI { r + two } else { r }
}
