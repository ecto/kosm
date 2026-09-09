//! Bodies on the cove's baked field.
//!
//! [`step_on_sdf`] is the SDF contact step, lifted out of `skatepark.rs` so
//! the park, the cove's marble and — next — the being all step the same way:
//! `Simulator::step_with_contacts` with the plane swapped for the baked field,
//! normals as the field reports them. The cove's bake is exact, so the
//! gradient is the normal and there is nothing to smooth.
//!
//! [`roll_on_beach`] is the check that the field *is* the beach. A glass
//! sphere released at rest on the sand rolls without slip down a plane of
//! constant grade, so its centre reaches `v² = 10/7 · g · Δ` after a drop of
//! `Δ`, and it never leaves the fall line: a normal that leans shows up as
//! drift across the slope before it shows up in the speed.
//!
//! Metres, z up.

use ipse_map::SdfGrid;
use phyz_contact::{ContactCache, ContactMaterial, ContactSolverConfig, assemble, solve_contacts_warm};
use phyz_math::{GRAVITY, Vec3};
use phyz_model::{Model, State};
use phyz_rigid::{aba, forward_kinematics, integrate_configuration, rotate_free_joint_velocities, strip_free_joint_coriolis};

use super::CoveScene;
use crate::garage::marble_model;

/// Free-joint q is [wx, wy, wz, x, y, z]; the rolled body is joint 0.
const POS: usize = 3;

/// Wet sand under glass. The cove has no friction knob: the beach is not a
/// tuned surface, it is the one the marble check is stated on.
pub const SAND_FRICTION: f64 = 0.8;

/// N-BK7, kg/m³: the being and the test marble are the same glass.
pub const GLASS_DENSITY: f64 = 2510.0;

/// One contact step against a baked map: `Simulator::step_with_contacts` with
/// the plane swapped for the field, normals as the field reports them.
pub fn step_on_sdf(model: &Model, state: &mut State, sdf: &SdfGrid, material: &ContactMaterial, cache: &mut ContactCache) {
    let dt = model.dt;
    let (xforms, _) = forward_kinematics(model, state);
    state.body_xform = xforms;
    let contacts = ipse_map::find_terrain_contacts_model(model, state, sdf, material.margin);
    // In the frame the contacts were assembled in: a free joint's body-frame
    // turn is taken out here and put back, exactly, after the solve (phyz).
    let mut qdd = aba(model, state);
    let v_before = state.v.clone();
    strip_free_joint_coriolis(model, v_before.as_slice(), qdd.as_mut_slice());
    let free_qd = &state.v + &(&qdd * dt);
    if contacts.is_empty() {
        state.v = free_qd;
    } else {
        let materials = model.contact_materials(material);
        let config = ContactSolverConfig::simulation();
        let asm = assemble(model, state, &contacts, &materials, &free_qd, dt, &config);
        let seed = cache.warm_start(state, &contacts);
        let solution = solve_contacts_warm(&asm.problem, &config, &seed);
        cache.store(state, &contacts, &solution.impulses);
        state.v = &free_qd + &asm.velocity_delta(&solution.impulses);
    }
    rotate_free_joint_velocities(model, v_before.as_slice(), state.v.as_mut_slice(), dt);
    let v = state.v.clone();
    integrate_configuration(model, state.q.as_mut_slice(), v.as_slice(), dt);
    state.time += dt;
}

/// What the marble did on the beach.
#[derive(Clone, Debug)]
pub struct BeachRoll {
    /// Where the centre was released and where it ended.
    pub start: Vec3,
    pub end: Vec3,
    /// How far the centre fell between them.
    pub drop: f64,
    /// The speed it had got to by then, m/s.
    pub speed: f64,
    /// `sqrt(10/7 · g · Δ)`, the rolling sphere's answer.
    pub predicted: f64,
    /// Furthest the centre strayed across the fall line, m.
    pub drift: f64,
    /// Whether it reached the guard line above the waterline before `t_end`.
    pub reached_guard: bool,
    pub steps: usize,
}

impl BeachRoll {
    /// Signed error against the rolling sphere, as a fraction.
    pub fn error(&self) -> f64 {
        self.speed / self.predicted - 1.0
    }
}

/// The beach as a plane, carried away from the scene.
///
/// A body standing on the sand needs the sand, and the sand is one plane; this
/// is that plane, read out of [`CoveScene`] once so that a `Cove` stepping in
/// a window does not have to keep the document alive to know where the ground
/// is. It is not a second statement of where the beach is — the point and the
/// normal both come from the scene's own two functions.
#[derive(Clone, Copy, Debug)]
pub struct Beach {
    /// A point the sand passes through.
    pub origin: Vec3,
    /// The sand's upward unit normal.
    pub normal: Vec3,
}

impl Beach {
    pub fn of(scene: &CoveScene) -> Self {
        Self { origin: Vec3::new(0.0, 0.0, scene.sand_z_at(0.0, 0.0)), normal: scene.sand_normal() }
    }

    /// The top of the sand at `(x, y)`: the plane, solved for z.
    pub fn z_at(&self, x: f64, y: f64) -> f64 {
        self.origin.z - (self.normal.x * (x - self.origin.x) + self.normal.y * (y - self.origin.y)) / self.normal.z
    }

    /// A sphere of radius `r` at rest on the sand at `(x, y)`: its centre is
    /// one radius up the sand's normal from the surface.
    pub fn resting_centre(&self, x: f64, y: f64, r: f64) -> Vec3 {
        Vec3::new(x, y, self.z_at(x, y)) + self.normal * r
    }
}

/// A sphere of radius `r` at rest on the sand at `(x, y)`: its centre is one
/// radius up the sand's normal from the surface.
pub fn resting_centre(scene: &CoveScene, x: f64, y: f64, r: f64) -> Vec3 {
    Beach::of(scene).resting_centre(x, y, r)
}

/// Release a glass sphere at rest on the sand and roll it down the beach on
/// the baked field, stopping `guard` metres short of the waterline — past
/// there the sand runs into the sea and the plane's answer no longer holds.
pub fn roll_on_beach(scene: &CoveScene, sdf: &SdfGrid, r: f64, x0: f64, y0: f64, t_end: f64, guard: f64) -> anyhow::Result<BeachRoll> {
    let dt = 1e-3;
    let mass = GLASS_DENSITY * 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
    let mut model = marble_model(r, mass);
    model.dt = dt;
    let material = ContactMaterial { friction: SAND_FRICTION, restitution: 0.0, ..Default::default() };
    let mut state = model.default_state();
    let start = resting_centre(scene, x0, y0, r);
    state.q[POS] = start.x;
    state.q[POS + 1] = start.y;
    state.q[POS + 2] = start.z;
    let mut cache = ContactCache::new(material.margin.max(1e-3));
    let steps = (t_end / dt).round() as usize;
    let stop_y = scene.waterline() + guard;
    let (mut drift, mut end, mut speed, mut reached_guard, mut taken) = (0.0f64, start, 0.0f64, false, 0usize);
    for k in 0..steps {
        step_on_sdf(&model, &mut state, sdf, &material, &mut cache);
        let p = Vec3::new(state.q[POS], state.q[POS + 1], state.q[POS + 2]);
        anyhow::ensure!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite(), "the roll diverged at t = {:.3} s", state.time);
        drift = drift.max((p.x - x0).abs());
        end = p;
        speed = Vec3::new(state.v[POS], state.v[POS + 1], state.v[POS + 2]).norm();
        taken = k + 1;
        if p.y <= stop_y {
            reached_guard = true;
            break;
        }
    }
    let drop = start.z - end.z;
    Ok(BeachRoll {
        start,
        end,
        drop,
        speed,
        predicted: (10.0 / 7.0 * GRAVITY * drop.max(0.0)).sqrt(),
        drift,
        reached_guard,
        steps: taken,
    })
}
