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
//! A ground also says what there is to *climb*:
//! [`Ground::ledge_ahead`] hands back the lip in front of the feet, if there
//! is one, and [`Body::step`](super::body::Body::step) is what puts hands on
//! it. The default is `None` — a plane has no lips — [`SdfGround`] walks a
//! ladder of heights up the face in front of the toes, and [`Terrace`] is a
//! plane with exactly one step in it, which is what `tests/feel.rs` climbs.
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
use phyz_math::{SpatialTransformExt, Vec3};
use phyz_model::{Model, State};
use phyz_rigid::{aba, forward_kinematics, integrate_configuration, rotate_free_joint_velocities, strip_free_joint_coriolis};

/// How far ahead of the feet, and between which two heights, a body looks for
/// something to climb.
///
/// The body fills it in from its own proportions — [`lo`](Self::lo) is knee
/// height, [`hi`](Self::hi) is the chest, [`reach`](Self::reach) is how far in
/// front of the toes the hands can get — so a ground never has to know how
/// tall anybody is.
#[derive(Clone, Copy, Debug)]
pub struct LedgeProbe {
    /// The lowest lip worth taking, metres above the feet.
    pub lo: f64,
    /// The highest, metres above the feet.
    pub hi: f64,
    /// How far in front of the feet to look, metres.
    pub reach: f64,
    /// How far *past* the lip has to be clear for it to be a top rather than
    /// a spike, metres.
    pub top: f64,
}

impl LedgeProbe {
    pub fn new(lo: f64, hi: f64, reach: f64, top: f64) -> Self {
        Self { lo, hi, reach, top }
    }
}

/// A lip the hands can take.
#[derive(Clone, Copy, Debug)]
pub struct LedgeInfo {
    /// The world point on the lip, on the line the body is walking.
    pub edge: Vec3,
    /// How far the top is above the feet, metres.
    pub height: f64,
    /// The face's outward normal, horizontal, pointing back at the body.
    pub normal: Vec3,
}

/// What is under the feet.
pub trait Ground: Send + Sync {
    /// Every contact between the model's collision set and the ground, in the
    /// state's *current* `body_xform` (so [`step_on`] runs forward kinematics
    /// before it asks).
    fn contacts(&self, model: &Model, state: &State, margin: f64) -> Vec<Collision>;

    /// The lip in front of `feet` along `facing`, if there is one this body
    /// could take in its hands.
    ///
    /// The default is `None`: a ground that cannot say is a ground with
    /// nothing to climb, which is right for a plane and honest for anything
    /// that has not been asked to answer.
    fn ledge_ahead(&self, _feet: Vec3, _facing: Vec3, _probe: &LedgeProbe) -> Option<LedgeInfo> {
        None
    }
}

/// A baked signed-distance field: the cove's own floor.
pub struct SdfGround(pub SdfGrid);

impl Ground for SdfGround {
    fn contacts(&self, model: &Model, state: &State, margin: f64) -> Vec<Collision> {
        kosm_scan::find_terrain_contacts_model(model, state, &self.0, margin)
    }

    /// Walk a ladder of heights up the face in front of the feet and find the
    /// last one that is still rock.
    ///
    /// Three questions per rung, and all three have to be yes for a rung to be
    /// a lip: the point `reach` ahead at `z` is *inside* (there is a face
    /// there to climb), the point a rung higher is *outside* (the face has
    /// stopped), and the point `top` further on at that height is outside too
    /// (there is somewhere to end up, rather than a spike). The field's sign
    /// is `kosm_scan`'s — positive outside the rock.
    fn ledge_ahead(&self, feet: Vec3, facing: Vec3, probe: &LedgeProbe) -> Option<LedgeInfo> {
        let dir = Vec3::new(facing.x, facing.y, 0.0).try_normalize()?;
        let ahead = feet + dir * probe.reach;
        let rung = (self.0.cell * 0.5).max(0.02);
        let solid = |p: Vec3| self.0.sample(p).map(|d| d <= 0.0).unwrap_or(false);
        let mut top = None;
        let mut z = probe.lo;
        while z <= probe.hi {
            if solid(ahead + Vec3::z() * z) {
                top = Some(z);
            }
            z += rung;
        }
        let z = top?;
        // The lip is the last rung that was rock; the one above it has to be
        // air, and so has the standing room past it.
        let lip = z + rung;
        if lip > probe.hi || solid(ahead + Vec3::z() * lip) {
            return None;
        }
        if solid(ahead + dir * probe.top + Vec3::z() * lip) {
            return None;
        }
        Some(LedgeInfo { edge: ahead + Vec3::z() * lip, height: lip, normal: -dir })
    }
}

/// A plane with one step in it: the ledge a test climbs.
///
/// `low` under everything behind `edge_x`, `high` over everything past it,
/// and a vertical riser between them that the feet cannot walk through. Two
/// calls to phyz's own plane finder, each filtered to its own side, plus a
/// wall the body's collision spheres are pushed out of — which is the whole
/// of what a step is, without a bake and without a mesh.
#[derive(Clone, Copy, Debug)]
pub struct Terrace {
    pub low: f64,
    pub high: f64,
    /// Where the riser is, along world `+x`.
    pub edge_x: f64,
}

impl Terrace {
    pub fn new(low: f64, high: f64, edge_x: f64) -> Self {
        Self { low, high, edge_x }
    }

    /// The riser's own contacts: every collision sphere below the top that
    /// has crossed the face, pushed back along `−x̂`.
    fn riser(&self, model: &Model, state: &State, margin: f64) -> Vec<Collision> {
        let mut out = Vec::new();
        for (i, body) in model.bodies.iter().enumerate() {
            let Some(x) = state.body_xform.get(i) else { continue };
            let shapes: Vec<_> = if body.collisions.is_empty() {
                body.geometry.iter().map(|g| (g.clone(), Vec3::zeros())).collect()
            } else {
                body.collisions.iter().map(|c| (c.geometry.clone(), c.origin.pos)).collect()
            };
            for (geom, at) in shapes {
                let r = match geom {
                    phyz_model::Geometry::Sphere { radius } => radius,
                    phyz_model::Geometry::Capsule { radius, .. } => radius,
                    _ => continue,
                };
                let centre = x.body_to_world_point(at);
                if centre.z - r >= self.high - 1e-9 {
                    continue;
                }
                let depth = centre.x + r - self.edge_x;
                if depth <= -margin || depth > 2.0 * r {
                    continue;
                }
                out.push(Collision {
                    body_i: i,
                    body_j: Collision::WORLD,
                    contact_point: Vec3::new(self.edge_x, centre.y, centre.z),
                    contact_normal: -Vec3::x(),
                    penetration_depth: depth,
                });
            }
        }
        out
    }
}

impl Ground for Terrace {
    fn contacts(&self, model: &Model, state: &State, margin: f64) -> Vec<Collision> {
        let mut out: Vec<Collision> = find_ground_contacts_model(model, state, self.low, margin)
            .into_iter()
            .filter(|c| c.contact_point.x < self.edge_x)
            .collect();
        out.extend(
            find_ground_contacts_model(model, state, self.high, margin)
                .into_iter()
                .filter(|c| c.contact_point.x >= self.edge_x),
        );
        out.extend(self.riser(model, state, margin));
        out
    }

    fn ledge_ahead(&self, feet: Vec3, facing: Vec3, probe: &LedgeProbe) -> Option<LedgeInfo> {
        let dir = Vec3::new(facing.x, facing.y, 0.0).try_normalize()?;
        // Only the face counts, and only from in front of it.
        if dir.x <= 0.2 || feet.x >= self.edge_x {
            return None;
        }
        let gap = self.edge_x - feet.x;
        if gap > probe.reach {
            return None;
        }
        let height = self.high - feet.z;
        if height < probe.lo || height > probe.hi {
            return None;
        }
        Some(LedgeInfo {
            edge: Vec3::new(self.edge_x, feet.y, self.high),
            height,
            normal: -Vec3::x(),
        })
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

    /// The net has nothing to climb; the ground under it might.
    fn ledge_ahead(&self, feet: Vec3, facing: Vec3, probe: &LedgeProbe) -> Option<LedgeInfo> {
        self.ground.ledge_ahead(feet, facing, probe)
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
