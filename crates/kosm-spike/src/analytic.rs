//! The marble level's colliders, told to `kosm-render`.
//!
//! The level's geometry is a vcad document, but the thing the physics actually
//! touches is the *derivation* of that document into colliders: oriented
//! boxes, spheres, cylinders (see [`crate::colliders`]). This module is the
//! other half of the seam `kosm-render` draws — it answers the three questions
//! the tracer asks of any geometry (how many primitives, where each one is,
//! what a ray finds when it meets one) for exactly those shapes.
//!
//! So the beauty pass is lit by the same bodies the solver collides. No
//! tessellation, no second description of the level: a sphere's silhouette is
//! a circle at any zoom because it is intersected as a sphere, and the marble
//! in the picture is the marble in the rollout.
//!
//! Nothing here is generic over `tang::Scalar`. It does not need to be:
//! `kosm-render`'s integrator is an `f64` Monte Carlo estimator, and the
//! gradients this level lives on come from [`crate::frame`]'s caster, which
//! stays generic for exactly that reason. This is the pretty picture; that one
//! is the derivative.

use kosm_render::{Aabb, Dir3, Geometry, Hit, Point2, Point3, Ray, Vec3};
use phyz_math::{Mat3, SpatialTransform, SpatialTransformExt};
use phyz_model::{GeomInstance, Geometry as PhyzGeometry};

/// One collider, in world coordinates.
#[derive(Debug, Clone)]
pub enum Prim {
    /// An oriented box: half-extents in the shape frame, `rot` world → shape.
    Box {
        /// Centre, world frame.
        center: Point3,
        /// Half-extent along each shape axis.
        half: Vec3,
        /// World → shape rotation.
        rot: Mat3,
    },
    /// A ball.
    Sphere {
        /// Centre, world frame.
        center: Point3,
        /// Radius, metres.
        radius: f64,
    },
    /// A capped cylinder along the shape frame's Z axis.
    Cylinder {
        /// Centre, world frame.
        center: Point3,
        /// Radius, metres.
        radius: f64,
        /// Half the height along the shape's Z.
        half_height: f64,
        /// World → shape rotation.
        rot: Mat3,
    },
}

/// A bag of colliders a ray can be tested against: the [`Geometry`] side of
/// `kosm-render`'s seam for this level.
#[derive(Debug, Clone, Default)]
pub struct Analytic {
    prims: Vec<Prim>,
}

impl Analytic {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// The primitives, in the order the tracer indexes them.
    pub fn prims(&self) -> &[Prim] {
        &self.prims
    }

    /// Add one primitive; the index it takes is its `Hit::prim`.
    pub fn push(&mut self, prim: Prim) -> &mut Self {
        self.prims.push(prim);
        self
    }

    /// A single ball, for the marble.
    pub fn ball(center: phyz_math::Vec3, radius: f64) -> Self {
        let mut geom = Self::new();
        geom.push(Prim::Sphere {
            center: point(center),
            radius,
        });
        geom
    }

    /// A body's collider set, placed in the world by `body_to_world`.
    ///
    /// Meshes and half-spaces are skipped: the marble level has neither, and a
    /// silently tessellated collider would be a second description of the
    /// geometry, which is the thing this whole level is arranged to avoid.
    pub fn from_colliders(body_to_world: &SpatialTransform, colliders: &[GeomInstance]) -> Self {
        let mut geom = Self::new();
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
                    rot,
                }),
                PhyzGeometry::Sphere { radius } => geom.push(Prim::Sphere {
                    center,
                    radius: *radius,
                }),
                PhyzGeometry::Cylinder { radius, height } => geom.push(Prim::Cylinder {
                    center,
                    radius: *radius,
                    half_height: 0.5 * height,
                    rot,
                }),
                _ => &mut geom,
            };
        }
        geom
    }
}

fn point(v: phyz_math::Vec3) -> Point3 {
    Point3::new(v.x, v.y, v.z)
}

fn vector(v: phyz_math::Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

/// `rot · v`, with `rot` a world → shape rotation.
fn rotate(rot: &Mat3, v: Vec3) -> Vec3 {
    Vec3::new(
        rot[(0, 0)] * v.x + rot[(0, 1)] * v.y + rot[(0, 2)] * v.z,
        rot[(1, 0)] * v.x + rot[(1, 1)] * v.y + rot[(1, 2)] * v.z,
        rot[(2, 0)] * v.x + rot[(2, 1)] * v.y + rot[(2, 2)] * v.z,
    )
}

/// `rotᵀ · v`: shape → world.
fn unrotate(rot: &Mat3, v: Vec3) -> Vec3 {
    Vec3::new(
        rot[(0, 0)] * v.x + rot[(1, 0)] * v.y + rot[(2, 0)] * v.z,
        rot[(0, 1)] * v.x + rot[(1, 1)] * v.y + rot[(2, 1)] * v.z,
        rot[(0, 2)] * v.x + rot[(1, 2)] * v.y + rot[(2, 2)] * v.z,
    )
}

/// The first `t` in `(t_min, t_max)` among the two roots of a quadratic-ish
/// pair, entry first.
fn nearest(t0: f64, t1: f64, t_min: f64, t_max: f64) -> Option<f64> {
    [t0, t1]
        .into_iter()
        .filter(|t| *t > t_min && *t < t_max)
        .fold(None, |best: Option<f64>, t| {
            Some(best.map_or(t, |b: f64| b.min(t)))
        })
}

impl Prim {
    fn bounds(&self) -> Aabb {
        let mut aabb = Aabb::empty();
        match self {
            Prim::Sphere { center, radius } => {
                aabb.include_point(&Point3::new(
                    center.x - radius,
                    center.y - radius,
                    center.z - radius,
                ));
                aabb.include_point(&Point3::new(
                    center.x + radius,
                    center.y + radius,
                    center.z + radius,
                ));
            }
            // Both oriented shapes bound the same way: the eight corners of
            // the local box that contains them, carried into the world.
            Prim::Box { center, half, rot } => {
                corners(&mut aabb, *center, *half, rot);
            }
            Prim::Cylinder {
                center,
                radius,
                half_height,
                rot,
            } => {
                corners(
                    &mut aabb,
                    *center,
                    Vec3::new(*radius, *radius, *half_height),
                    rot,
                );
            }
        }
        aabb
    }

    fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit> {
        let d = *ray.direction.as_ref();
        match self {
            Prim::Sphere { center, radius } => {
                let oc = ray.origin - *center;
                let b = oc.dot(d);
                let disc = b * b - (oc.dot(oc) - radius * radius);
                if disc < 0.0 {
                    return None;
                }
                let root = disc.sqrt();
                let t = nearest(-b - root, -b + root, t_min, t_max)?;
                let p = ray.at(t);
                let n = (p - *center) / *radius;
                Some(hit(t, p, n, i))
            }
            Prim::Box { center, half, rot } => {
                let o = rotate(rot, ray.origin - *center);
                let dl = rotate(rot, d);
                let (o, dl, h) = (o.as_array(), dl.as_array(), half.as_array());
                let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
                let (mut axis0, mut axis1) = (0usize, 0usize);
                let (mut sign0, mut sign1) = (1.0f64, 1.0f64);
                for k in 0..3 {
                    let inv = 1.0 / dl[k];
                    let (mut lo, mut hi) = ((-h[k] - o[k]) * inv, (h[k] - o[k]) * inv);
                    let (mut slo, mut shi) = (-1.0, 1.0);
                    if lo > hi {
                        std::mem::swap(&mut lo, &mut hi);
                        std::mem::swap(&mut slo, &mut shi);
                    }
                    if lo > t0 {
                        t0 = lo;
                        axis0 = k;
                        sign0 = slo;
                    }
                    if hi < t1 {
                        t1 = hi;
                        axis1 = k;
                        sign1 = shi;
                    }
                    if t1 < t0 {
                        return None;
                    }
                }
                // The near face if the ray is outside, the far one if it
                // started within: a shadow ray that begins on the surface must
                // still find the box it is leaving.
                let (t, axis, sign) = if t0 > t_min && t0 < t_max {
                    (t0, axis0, sign0)
                } else if t1 > t_min && t1 < t_max {
                    (t1, axis1, sign1)
                } else {
                    return None;
                };
                let mut nl = [0.0; 3];
                nl[axis] = sign;
                let n = unrotate(rot, Vec3::new(nl[0], nl[1], nl[2]));
                Some(hit(t, ray.at(t), n, i))
            }
            Prim::Cylinder {
                center,
                radius,
                half_height,
                rot,
            } => {
                let o = rotate(rot, ray.origin - *center);
                let dl = rotate(rot, d);
                // The side: a quadratic in the xy plane, clipped to the caps.
                // The caps: two z slabs, clipped to the disc.
                let a = dl.x * dl.x + dl.y * dl.y;
                let mut best: Option<(f64, Vec3)> = None;
                let mut keep = |t: f64, n: Vec3| {
                    if t > t_min && t < t_max && best.as_ref().is_none_or(|b| t < b.0) {
                        best = Some((t, n));
                    }
                };
                if a > 1e-18 {
                    let b = o.x * dl.x + o.y * dl.y;
                    let c = o.x * o.x + o.y * o.y - radius * radius;
                    let disc = b * b - a * c;
                    if disc >= 0.0 {
                        let root = disc.sqrt();
                        for t in [(-b - root) / a, (-b + root) / a] {
                            let z = o.z + dl.z * t;
                            if z.abs() <= *half_height {
                                keep(t, Vec3::new(o.x + dl.x * t, o.y + dl.y * t, 0.0) / *radius);
                            }
                        }
                    }
                }
                if dl.z.abs() > 1e-18 {
                    for s in [-1.0, 1.0] {
                        let t = (s * half_height - o.z) / dl.z;
                        let (x, y) = (o.x + dl.x * t, o.y + dl.y * t);
                        if x * x + y * y <= radius * radius {
                            keep(t, Vec3::new(0.0, 0.0, s));
                        }
                    }
                }
                let (t, nl) = best?;
                Some(hit(t, ray.at(t), unrotate(rot, nl), i))
            }
        }
    }
}

fn corners(aabb: &mut Aabb, center: Point3, half: Vec3, rot: &Mat3) {
    for k in 0..8 {
        let local = Vec3::new(
            if k & 1 == 0 { -half.x } else { half.x },
            if k & 2 == 0 { -half.y } else { half.y },
            if k & 4 == 0 { -half.z } else { half.z },
        );
        let w = unrotate(rot, local);
        aabb.include_point(&Point3::new(
            center.x + w.x,
            center.y + w.y,
            center.z + w.z,
        ));
    }
}

fn hit(t: f64, p: Point3, n: Vec3, i: usize) -> Hit {
    Hit::new(t, p, Dir3::new_normalize(n), Point2::new(0.0, 0.0), i as u32)
}

impl Geometry for Analytic {
    fn len(&self) -> usize {
        self.prims.len()
    }

    fn bounds(&self, i: usize) -> Aabb {
        self.prims[i].bounds()
    }

    fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit> {
        self.prims[i].intersect(ray, i, t_min, t_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kosm_render::Bvh;

    fn ray(from: [f64; 3], dir: [f64; 3]) -> Ray {
        Ray::new(
            Point3::new(from[0], from[1], from[2]),
            Vec3::new(dir[0], dir[1], dir[2]),
        )
    }

    #[test]
    fn a_ball_is_hit_where_the_analytic_root_says() {
        let geom = Analytic::ball(phyz_math::Vec3::new(0.0, 0.0, 0.0), 0.5);
        let hit = geom
            .intersect(&ray([0.0, 0.0, -3.0], [0.0, 0.0, 1.0]), 0, 0.0, f64::INFINITY)
            .expect("hits the ball");
        assert!((hit.t - 2.5).abs() < 1e-12, "t = {}", hit.t);
        assert!((hit.normal.z + 1.0).abs() < 1e-12, "outward normal");
    }

    #[test]
    fn an_axis_aligned_box_is_a_slab_test() {
        let mut geom = Analytic::new();
        geom.push(Prim::Box {
            center: Point3::new(0.0, 0.0, 0.0),
            half: Vec3::new(1.0, 1.0, 0.25),
            rot: Mat3::identity(),
        });
        let hit = geom
            .intersect(&ray([0.0, 0.0, 2.0], [0.0, 0.0, -1.0]), 0, 0.0, f64::INFINITY)
            .expect("hits the lid");
        assert!((hit.t - 1.75).abs() < 1e-12, "t = {}", hit.t);
        assert!((hit.normal.z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_cylinder_has_a_round_side_and_flat_caps() {
        let mut geom = Analytic::new();
        geom.push(Prim::Cylinder {
            center: Point3::new(0.0, 0.0, 0.0),
            radius: 0.5,
            half_height: 1.0,
            rot: Mat3::identity(),
        });
        let side = geom
            .intersect(&ray([-3.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 0, 0.0, f64::INFINITY)
            .expect("hits the side");
        assert!((side.t - 2.5).abs() < 1e-12, "t = {}", side.t);
        assert!((side.normal.x + 1.0).abs() < 1e-12);
        let cap = geom
            .intersect(&ray([0.0, 0.0, 3.0], [0.0, 0.0, -1.0]), 0, 0.0, f64::INFINITY)
            .expect("hits the cap");
        assert!((cap.t - 2.0).abs() < 1e-12, "t = {}", cap.t);
        assert!((cap.normal.z - 1.0).abs() < 1e-12);
        // and misses beside it
        assert!(
            geom.intersect(&ray([0.0, 0.9, 3.0], [0.0, 0.0, -1.0]), 0, 0.0, f64::INFINITY)
                .is_none()
        );
    }

    #[test]
    fn a_bvh_over_the_colliders_finds_the_nearest_one() {
        let mut geom = Analytic::new();
        for k in 0..8 {
            geom.push(Prim::Sphere {
                center: Point3::new(k as f64, 0.0, 0.0),
                radius: 0.25,
            });
        }
        let bvh = Bvh::build(geom);
        let hit = bvh
            .trace_closest(&ray([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
            .expect("hits the first ball");
        assert_eq!(hit.prim, 0);
        assert!((hit.t - 4.75).abs() < 1e-12, "t = {}", hit.t);
    }
}
