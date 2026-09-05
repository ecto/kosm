//! The marble level's colliders, told to `kosm-render`.
//!
//! The level's geometry is a vcad document, but the thing the physics
//! actually touches is the *derivation* of that document into colliders:
//! oriented boxes, spheres, cylinders (see [`crate::colliders`]). This module
//! is the seam between those and the renderer.
//!
//! It used to intersect them too. It no longer does: the box, the sphere and
//! the capped cylinder that lived here are now
//! [`kosm_render::analytic`], where the pool's ellipsoid and the court's
//! torus can share them, and where a millimetre scene's precision is
//! somebody's actual job. What is left here is the conversion — one phyz
//! [`GeomInstance`] to one [`Prim`] — which is the only part that was ever
//! about *this* level.
//!
//! So the beauty pass is still lit by the same bodies the solver collides. No
//! tessellation, no second description: a sphere's silhouette is a circle at
//! any zoom because it is intersected as a sphere, and the marble in the
//! picture is the marble in the rollout.

use kosm_render::analytic::{Analytic, Frame, Prim};
use kosm_render::{Dir3, Point3, Vec3};
use phyz_math::{Mat3, SpatialTransform, SpatialTransformExt};
use phyz_model::{GeomInstance, Geometry as PhyzGeometry};

/// A body's collider set, placed in the world by `body_to_world`.
///
/// Meshes and half-spaces are skipped: the marble level has neither, and a
/// silently tessellated collider would be a second description of the
/// geometry, which is the thing this whole level is arranged to avoid.
pub fn from_colliders(body_to_world: &SpatialTransform, colliders: &[GeomInstance]) -> Analytic {
    let mut geom = Analytic::new();
    for g in colliders {
        let center = point(body_to_world.body_to_world_point(g.origin.pos));
        // `origin.rot` is body → shape and `body_to_world.rot` is world →
        // body, so their product is world → shape. The same composition
        // `crate::frame` makes, for the same reason.
        let rot = g.origin.rot * body_to_world.rot;
        match &g.geometry {
            PhyzGeometry::Box { half_extents } => geom.push(Prim::Box {
                center,
                half: vector(*half_extents),
                rot: frame(&rot),
            }),
            PhyzGeometry::Sphere { radius } => geom.push(Prim::Sphere {
                center,
                radius: *radius,
            }),
            PhyzGeometry::Cylinder { radius, height } => geom.push(Prim::Cylinder {
                center,
                // A phyz cylinder runs along its shape frame's Z; that axis
                // in world coordinates is the third *row* of the world →
                // shape rotation.
                axis: axis_z(&rot),
                radius: *radius,
                half_height: 0.5 * height,
            }),
            _ => &mut geom,
        };
    }
    geom
}

/// A single ball, for the marble.
pub fn ball(center: phyz_math::Vec3, radius: f64) -> Analytic {
    Analytic::sphere(point(center), radius)
}

fn point(v: phyz_math::Vec3) -> Point3 {
    Point3::new(v.x, v.y, v.z)
}

fn vector(v: phyz_math::Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

/// phyz's world → shape rotation as the renderer's shape → world [`Frame`]:
/// the rows of the one are the columns of the other.
fn frame(rot: &Mat3) -> Frame {
    Frame::from_axes(row(rot, 0), row(rot, 1), row(rot, 2))
}

fn row(rot: &Mat3, i: usize) -> Vec3 {
    Vec3::new(rot[(i, 0)], rot[(i, 1)], rot[(i, 2)])
}

fn axis_z(rot: &Mat3) -> Dir3 {
    Dir3::new_normalize(row(rot, 2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kosm_render::{Bvh, Geometry, Ray};

    fn ray(from: [f64; 3], dir: [f64; 3]) -> Ray {
        Ray::new(
            Point3::new(from[0], from[1], from[2]),
            Vec3::new(dir[0], dir[1], dir[2]),
        )
    }

    #[test]
    fn a_ball_is_hit_where_the_analytic_root_says() {
        let geom = ball(phyz_math::Vec3::new(0.0, 0.0, 0.0), 0.5);
        let hit = geom
            .intersect(
                &ray([0.0, 0.0, -3.0], [0.0, 0.0, 1.0]),
                0,
                0.0,
                f64::INFINITY,
            )
            .expect("hits the ball");
        assert!((hit.t - 2.5).abs() < 1e-12, "t = {}", hit.t);
        assert!((hit.normal.z + 1.0).abs() < 1e-12, "outward normal");
    }

    #[test]
    fn an_axis_aligned_box_is_a_slab_test() {
        let geom = Analytic::from_prims(vec![Prim::Box {
            center: Point3::new(0.0, 0.0, 0.0),
            half: Vec3::new(1.0, 1.0, 0.25),
            rot: Frame::identity(),
        }]);
        let hit = geom
            .intersect(
                &ray([0.0, 0.0, 2.0], [0.0, 0.0, -1.0]),
                0,
                0.0,
                f64::INFINITY,
            )
            .expect("hits the lid");
        assert!((hit.t - 1.75).abs() < 1e-12, "t = {}", hit.t);
        assert!((hit.normal.z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_cylinder_has_a_round_side_and_flat_caps() {
        let geom = Analytic::from_prims(vec![Prim::Cylinder {
            center: Point3::new(0.0, 0.0, 0.0),
            axis: Dir3::new_normalize(Vec3::new(0.0, 0.0, 1.0)),
            radius: 0.5,
            half_height: 1.0,
        }]);
        let side = geom
            .intersect(
                &ray([-3.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                0,
                0.0,
                f64::INFINITY,
            )
            .expect("hits the side");
        assert!((side.t - 2.5).abs() < 1e-12, "t = {}", side.t);
        assert!((side.normal.x + 1.0).abs() < 1e-12);
        let cap = geom
            .intersect(
                &ray([0.0, 0.0, 3.0], [0.0, 0.0, -1.0]),
                0,
                0.0,
                f64::INFINITY,
            )
            .expect("hits the cap");
        assert!((cap.t - 2.0).abs() < 1e-12, "t = {}", cap.t);
        assert!((cap.normal.z - 1.0).abs() < 1e-12);
        // and misses beside it
        assert!(
            geom.intersect(
                &ray([0.0, 0.9, 3.0], [0.0, 0.0, -1.0]),
                0,
                0.0,
                f64::INFINITY
            )
            .is_none()
        );
    }

    #[test]
    fn a_bvh_over_the_colliders_finds_the_nearest_one() {
        let geom: Analytic = (0..8)
            .map(|k| Prim::Sphere {
                center: Point3::new(k as f64, 0.0, 0.0),
                radius: 0.25,
            })
            .collect();
        let bvh = Bvh::build(geom);
        let hit = bvh
            .trace_closest(&ray([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
            .expect("hits the first ball");
        assert_eq!(hit.prim, 0);
        assert!((hit.t - 4.75).abs() < 1e-12, "t = {}", hit.t);
    }

    /// The conversion itself: a phyz cylinder's shape Z has to come out as
    /// the renderer's axis, or every rolled part of the level is rendered
    /// lying down.
    #[test]
    fn a_phyz_cylinders_axis_survives_the_conversion() {
        let mut origin = SpatialTransform::identity();
        // Lay the cylinder down along world X. `rot` is world → shape, so its
        // *rows* are the shape's axes in world coordinates: put world X in
        // the third one and the cylinder's Z is world X.
        origin.rot = Mat3::new(
            0.0, 0.0, 1.0, //
            0.0, -1.0, 0.0, //
            1.0, 0.0, 0.0,
        );
        let inst = GeomInstance::new(
            PhyzGeometry::Cylinder {
                radius: 0.2,
                height: 2.0,
            },
            origin,
        );
        let geom = from_colliders(&SpatialTransform::identity(), &[inst]);
        let Prim::Cylinder { axis, .. } = geom.prims()[0] else {
            panic!("expected a cylinder, got {:?}", geom.prims()[0]);
        };
        assert!(axis.as_ref().x.abs() > 0.999, "axis = {:?}", axis.as_ref());

        // And it is hit on its round side from above, not on a cap.
        let hit = geom
            .intersect(
                &ray([0.0, 0.0, 3.0], [0.0, 0.0, -1.0]),
                0,
                0.0,
                f64::INFINITY,
            )
            .expect("hits the side");
        assert!((hit.t - 2.8).abs() < 1e-12, "t = {}", hit.t);
        assert!((hit.normal.z - 1.0).abs() < 1e-12);
    }
}
