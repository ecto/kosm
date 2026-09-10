//! `Ground`: what is under the feet.
//!
//! Three answers, one trait. The cove has a baked signed-distance field
//! ([`SdfGround`]); a test wants a plane and no bake ([`Plane`]); a level made
//! of authored solids wants phyz's own convex contacts against them
//! ([`Colliders`]). A [`Body`](super::body::Body) never asks which — it asks
//! for contacts and steps on them.
//!
//! [`Netted`] is the fourth, and it is not a ground: it is the floor *under*
//! the level. `kosm_scan::find_terrain_contacts_model` skips any candidate
//! outside the sampled volume, because beyond a scan there is no floor — the
//! honest reading of a scan and exactly wrong for a level, where walking off
//! the edge means falling for ever. The net is consulted only when the ground
//! has nothing to say, and it counts: a level that is closed never uses it,
//! and [`Netted::caught`] is what says so. Lifted from
//! `sims/rune/sim.rs::step_on_sdf_over`.
//!
//! [`step_on`] is the contact step itself, the same sequence phyz's
//! `Simulator::step_with_contacts` runs with the plane swapped for whatever
//! the `Ground` reports.
//!
//! Metres, seconds, z up.

use std::sync::atomic::{AtomicUsize, Ordering};

use kosm_scan::SdfGrid;
use phyz_collision::Collision;
use phyz_contact::{ContactCache, ContactMaterial, ContactSolverConfig, assemble, find_contacts, find_ground_contacts_model, solve_contacts_warm};
use phyz_math::Vec3;
use phyz_model::{Model, State};
use phyz_rigid::{aba, forward_kinematics, integrate_configuration, rotate_free_joint_velocities, strip_free_joint_coriolis};

/// What is under the feet.
pub trait Ground: Send + Sync {
    /// Every contact between the model's collision set and the ground, in the
    /// state's *current* `body_xform` (so [`step_on`] runs forward kinematics
    /// before it asks).
    fn contacts(&self, model: &Model, state: &State, margin: f64) -> Vec<Collision>;
}

/// A baked signed-distance field: the cove's own floor.
pub struct SdfGround(pub SdfGrid);

impl Ground for SdfGround {
    fn contacts(&self, model: &Model, state: &State, margin: f64) -> Vec<Collision> {
        kosm_scan::find_terrain_contacts_model(model, state, &self.0, margin)
    }
}

/// A horizontal plane at `z`. The floor a sim with no bake still has, and
/// what every test in `tests/player.rs` stands on.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub z: f64,
}

impl Plane {
    pub fn at(z: f64) -> Self {
        Self { z }
    }
}

impl Ground for Plane {
    fn contacts(&self, model: &Model, state: &State, margin: f64) -> Vec<Collision> {
        find_ground_contacts_model(model, state, self.z, margin)
    }
}

/// The world's own fixed bodies, through phyz's convex contacts.
///
/// The level is geometry rather than a half-space: whatever the model carries
/// as `collisions` collides with whatever else it carries, and the body that
/// does not move is the ground.
pub struct Colliders;

impl Ground for Colliders {
    fn contacts(&self, model: &Model, state: &State, margin: f64) -> Vec<Collision> {
        find_contacts(model, state, margin)
    }
}

/// No ground at all: a body in free fall, and what the *door* stands on —
/// a level's driven hinge is a body with no contacts, and stepping it through
/// the same path as everything else is cheaper than a second integrator.
pub struct Nowhere;

impl Ground for Nowhere {
    fn contacts(&self, _: &Model, _: &State, _: f64) -> Vec<Collision> {
        Vec::new()
    }
}

/// A ground with a plane under it, consulted only when the ground has nothing
/// to say.
///
/// Counting is the point. A closed level never falls through, so
/// [`Netted::caught`] staying at zero is a property a test can assert; the day
/// it stops being zero the level has a hole in it and the number says how
/// often the player found it.
pub struct Netted<G> {
    pub ground: G,
    pub floor_z: f64,
    caught: AtomicUsize,
}

impl<G: Ground> Netted<G> {
    pub fn new(ground: G, floor_z: f64) -> Self {
        Self { ground, floor_z, caught: AtomicUsize::new(0) }
    }

    /// How many steps the net has carried. Zero, on a level that is closed.
    pub fn caught(&self) -> usize {
        self.caught.load(Ordering::Relaxed)
    }
}

impl<G: Ground> Ground for Netted<G> {
    fn contacts(&self, model: &Model, state: &State, margin: f64) -> Vec<Collision> {
        let found = self.ground.contacts(model, state, margin);
        if !found.is_empty() {
            return found;
        }
        let net = find_ground_contacts_model(model, state, self.floor_z, margin);
        if !net.is_empty() {
            self.caught.fetch_add(1, Ordering::Relaxed);
        }
        net
    }
}

/// One contact step against a `Ground`.
///
/// `Simulator::step_with_contacts` with the plane swapped for the trait. The
/// free joint's body-frame turn is taken out before the solve and put back
/// after, exactly, which is phyz's own sequence and is the whole reason this
/// is written out rather than delegated.
///
/// Returns the **mean contact normal**, world, when the body touched anything.
/// A controller wants it — what is under the feet is how steep the ground is —
/// and the contacts are already found here, so handing it back is free where
/// asking for it again would not be.
pub fn step_on(model: &Model, state: &mut State, ground: &dyn Ground, material: &ContactMaterial, cache: &mut ContactCache) -> Option<Vec3> {
    let dt = model.dt;
    let (xforms, _) = forward_kinematics(model, state);
    state.body_xform = xforms;
    let contacts = ground.contacts(model, state, material.margin);
    let support = contacts
        .iter()
        .fold(Vec3::zeros(), |acc, c| acc + c.contact_normal)
        .try_normalize();

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
    support
}
