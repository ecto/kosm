//! Terrain contacts: the phyz ground-plane producer, with the plane
//! generalized to a signed-distance field.
//!
//! This mirrors `phyz_contact::find_ground_contacts_model` deliberately and
//! closely — same per-body candidate pool, same deepest-first ranking, same
//! `MAX_MANIFOLD_POINTS` cap, same margin band with negative-depth contacts,
//! same `body_j = usize::MAX` ("ground is not a body"), same midsurface
//! contact point — so that on a flat map it produces the same contacts and
//! the whole downstream solve is untouched. `ipse-sim/tests/terrain_parity.rs`
//! holds it to that.
//!
//! The generalization: where the plane producer measures
//! `depth = ground_height − p.z` against a fixed `+z` normal, this one
//! measures `depth = −sdf(p)` against `∇sdf(p)`. Spheres and capsules get an
//! upgrade the plane could not express: their support point drops along
//! `−∇sdf` at the centre rather than along `−ẑ`, which is exact for a curved
//! field and identical on a flat one.
//!
//! Divergence to know about: candidates whose sample point falls outside the
//! SDF volume are skipped — no floor beyond the map ([`crate::sdf`] module
//! docs) — where the infinite plane would keep reporting. Keep the robot on
//! the scan.
//!
//! When this has survived real terrain it belongs upstream in `phyz-contact`
//! next to its sibling; it lives here first so the map format and the
//! collider can iterate in one place without a phyz release per tweak.

use phyz_collision::{Collision, MAX_MANIFOLD_POINTS};
use phyz_math::{SpatialTransformExt, Vec3};
use phyz_model::{Geometry as ModelGeometry, Model, State};

use crate::sdf::SdfGrid;

/// Find contacts between the model's collision set and the terrain SDF.
///
/// Drop-in replacement for `find_ground_contacts_model(model, state,
/// GROUND_Z, margin)` — same contract, same conventions, terrain from the
/// map instead of an infinite plane.
pub fn find_terrain_contacts_model(
    model: &Model,
    state: &State,
    sdf: &SdfGrid,
    margin: f64,
) -> Vec<Collision> {
    let margin = if margin.is_finite() { margin.max(0.0) } else { 0.0 };
    let mut out = Vec::new();

    for (i, body) in model.bodies.iter().enumerate() {
        let Some(xform) = state.body_xform.get(i) else {
            continue;
        };
        let finite = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        if !finite(xform.pos)
            || !finite(xform.rot.c0)
            || !finite(xform.rot.c1)
            || !finite(xform.rot.c2)
        {
            continue;
        }

        // One candidate pool per body, exactly as the plane producer pools
        // all of a body's shapes into one manifold.
        let mut pool: Vec<Candidate> = Vec::new();
        let mut push_shape = |geom: &ModelGeometry, sx: &phyz_math::SpatialTransform| {
            collect_candidates(geom, sx, sdf, margin, &mut pool);
        };

        if body.collisions.is_empty() {
            if let Some(geom) = &body.geometry {
                push_shape(geom, xform);
            }
        } else {
            for inst in &body.collisions {
                push_shape(&inst.geometry, &shape_world_xform(xform, &inst.origin));
            }
        }

        pool.sort_by(|a, b| b.depth.total_cmp(&a.depth));
        pool.truncate(MAX_MANIFOLD_POINTS);

        for c in pool {
            out.push(Collision {
                body_i: i,
                body_j: usize::MAX, // Ground is not a body
                // Midsurface between the support point and the terrain, the
                // plane producer's convention; stays correct for a negative
                // depth, where the midpoint sits above the surface.
                contact_point: c.point + c.normal * (c.depth * 0.5),
                contact_normal: c.normal,
                penetration_depth: c.depth,
            });
        }
    }

    out
}

struct Candidate {
    depth: f64,
    /// Support point on the body, world frame.
    point: Vec3,
    /// Separating direction for the body, world frame, unit length.
    normal: Vec3,
}

/// Push a shape's terrain candidates into `pool`.
///
/// Support-point generation matches `phyz_contact::ground_candidates` shape
/// by shape (box corners in the same nested-loop order, capsule caps in the
/// same order, the same 8 cylinder rim points, every mesh vertex), so that
/// candidate order — and therefore tie-breaking in the stable deepest-first
/// sort — is preserved and a flat map reproduces the plane producer's
/// manifold bit for bit.
fn collect_candidates(
    geom: &ModelGeometry,
    xform: &phyz_math::SpatialTransform,
    sdf: &SdfGrid,
    margin: f64,
    pool: &mut Vec<Candidate>,
) {
    let pos = xform.pos;
    // A material point penetrates by `−sdf(p)`; its normal is the field
    // gradient at the point.
    let mut push_material = |p: Vec3| {
        let Some((d, g)) = sdf.sample_with_gradient(p) else {
            return;
        };
        let depth = -d;
        if depth <= -margin {
            return;
        }
        let Some(normal) = g.try_normalize() else {
            return;
        };
        pool.push(Candidate { depth, point: p, normal });
    };
    // A ball of radius `r` centred at `c` penetrates by `r − sdf(c)`; its
    // support point hangs off the centre along `−∇sdf`. On a flat field the
    // gradient is `+ẑ` and this is the plane producer's `c − r·ẑ` exactly.
    let push_ball = |c: Vec3, r: f64, pool: &mut Vec<Candidate>| {
        let Some((d, g)) = sdf.sample_with_gradient(c) else {
            return;
        };
        let depth = r - d;
        if depth <= -margin {
            return;
        }
        let Some(normal) = g.try_normalize() else {
            return;
        };
        pool.push(Candidate { depth, point: c - normal * r, normal });
    };

    match geom {
        ModelGeometry::Box { half_extents } => {
            let h = half_extents;
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    for sz in [-1.0, 1.0] {
                        push_material(
                            xform.body_to_world_point(Vec3::new(sx * h.x, sy * h.y, sz * h.z)),
                        );
                    }
                }
            }
        }
        ModelGeometry::Sphere { radius } => push_ball(pos, *radius, pool),
        ModelGeometry::Capsule { radius, length } => {
            let axis = xform.body_to_world_dir(Vec3::new(0.0, 0.0, length * 0.5));
            push_ball(pos + axis, *radius, pool);
            push_ball(pos - axis, *radius, pool);
        }
        ModelGeometry::Cylinder { radius, height } => {
            let hz = xform.body_to_world_dir(Vec3::new(0.0, 0.0, height * 0.5));
            let (ex, ey) = (
                xform.body_to_world_dir(Vec3::x()) * *radius,
                xform.body_to_world_dir(Vec3::y()) * *radius,
            );
            for k in 0..4 {
                let t = k as f64 * std::f64::consts::FRAC_PI_2;
                let r = ex * t.cos() + ey * t.sin();
                push_material(pos + hz + r);
                push_material(pos - hz + r);
            }
        }
        ModelGeometry::Mesh { vertices, .. } => {
            for v in vertices {
                push_material(xform.body_to_world_point(*v));
            }
        }
        ModelGeometry::Plane { .. } => {}
    }
}

/// World transform of a shape mounted at `origin` inside a body — the same
/// composition `phyz_contact` applies to `Body::collisions` entries.
fn shape_world_xform(
    body_xform: &phyz_math::SpatialTransform,
    origin: &phyz_math::SpatialTransform,
) -> phyz_math::SpatialTransform {
    phyz_math::SpatialTransform::new(
        origin.rot.mul_mat(&body_xform.rot),
        body_xform.body_to_world_point(origin.pos),
    )
}
