//! The garage, as signed distance: props built from analytic primitives.
//!
//! Terrain for training does not come from a scan — it comes from here, so
//! the geometry is exact, auditable, and randomizable. Every prop below is
//! modelled from the real object's *measured* dimensions (sources in each
//! constructor's docs), because a policy trained on a plausible-looking box
//! learns the wrong step height and finds out on hardware.
//!
//! Analytic rather than meshed on purpose: an SDF built from primitives is
//! exact everywhere, has no winding or watertightness failure modes, and
//! costs nothing to re-randomize per episode. [`Prop::bake`] samples the
//! composed field into the same [`SdfGrid`] a scan produces, so the contact
//! solver cannot tell the difference.
//!
//! # Resolution and the 1-inch mat
//!
//! Interlocking gym mats are 1 inch (2.54 cm) thick. At the 2 cm default
//! cell that is a single sample — an obstacle the field cannot represent,
//! which would train against a rumour of a mat. [`GARAGE_CELL`] is
//! therefore 1 cm: the mat spans 2.5 cells, its edge is a real lip, and an
//! ankle can genuinely catch it. Everything in this module is sized against
//! that cell.

use phyz_math::Vec3;
use rayon::prelude::*;

use crate::SdfGrid;

/// Bake resolution for garage scenes, metres. See module docs: the 1-inch
/// mat sets it, not the big props.
pub const GARAGE_CELL: f64 = 0.01;

/// A signed-distance solid, in world coordinates (z-up, floor at z = 0).
#[derive(Debug, Clone)]
pub enum Solid {
    /// Ground half-space `z <= height`.
    Ground { height: f64 },
    /// Axis-aligned rounded box: `radius` rounds every edge (0 = sharp).
    Box { center: Vec3, half: Vec3, radius: f64, yaw: f64 },
    /// Z-axis cylinder.
    Cylinder { center: Vec3, radius: f64, half_height: f64 },
    /// Truncated cone about z: `r_bottom` at `center.z - half_height`.
    Frustum { center: Vec3, r_bottom: f64, r_top: f64, half_height: f64 },
    /// Capsule between two points.
    Capsule { a: Vec3, b: Vec3, radius: f64 },
    /// Union of solids (min).
    Union(Vec<Solid>),
    /// `a` minus `b` (max(a, -b)) — for hollows like a bucket's interior.
    Difference(Box<Solid>, Box<Solid>),
}

impl Solid {
    /// Signed distance at `p`: negative inside.
    pub fn distance(&self, p: Vec3) -> f64 {
        match self {
            Solid::Ground { height } => p.z - height,
            Solid::Box { center, half, radius, yaw } => {
                let d = rotate_z(p - *center, -*yaw);
                let q = Vec3::new(
                    d.x.abs() - (half.x - *radius),
                    d.y.abs() - (half.y - *radius),
                    d.z.abs() - (half.z - *radius),
                );
                let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).norm();
                let inside = q.x.max(q.y).max(q.z).min(0.0);
                outside + inside - *radius
            }
            Solid::Cylinder { center, radius, half_height } => {
                let d = p - *center;
                let radial = (d.x * d.x + d.y * d.y).sqrt() - *radius;
                let axial = d.z.abs() - *half_height;
                let outside = Vec3::new(radial.max(0.0), axial.max(0.0), 0.0).norm();
                outside + radial.max(axial).min(0.0)
            }
            Solid::Frustum { center, r_bottom, r_top, half_height } => {
                // Exact enough for contact: interpolate the radius at this
                // height, then treat as a cylinder locally. The taper on a
                // bucket is ~3 degrees; the error is well under a cell.
                let d = p - *center;
                let t = ((d.z + half_height) / (2.0 * half_height)).clamp(0.0, 1.0);
                let r = r_bottom + (r_top - r_bottom) * t;
                let radial = (d.x * d.x + d.y * d.y).sqrt() - r;
                let axial = d.z.abs() - *half_height;
                let outside = Vec3::new(radial.max(0.0), axial.max(0.0), 0.0).norm();
                outside + radial.max(axial).min(0.0)
            }
            Solid::Capsule { a, b, radius } => {
                let ab = *b - *a;
                let ap = p - *a;
                let t = (ap.dot(ab) / ab.norm_squared().max(1e-12)).clamp(0.0, 1.0);
                (ap - ab * t).norm() - *radius
            }
            Solid::Union(parts) => parts
                .iter()
                .map(|s| s.distance(p))
                .fold(f64::INFINITY, f64::min),
            Solid::Difference(a, b) => a.distance(p).max(-b.distance(p)),
        }
    }

    /// Bake into an [`SdfGrid`] over `lo..hi` at `cell`.
    pub fn bake(&self, lo: Vec3, hi: Vec3, cell: f64) -> SdfGrid {
        let extent = hi - lo;
        let nx = (extent.x / cell).ceil() as usize + 1;
        let ny = (extent.y / cell).ceil() as usize + 1;
        let nz = (extent.z / cell).ceil() as usize + 1;
        let mut data = vec![0.0f32; nx * ny * nz];
        data.par_chunks_mut(nx * ny).enumerate().for_each(|(k, slab)| {
            let z = lo.z + k as f64 * cell;
            for j in 0..ny {
                let y = lo.y + j as f64 * cell;
                for (i, out) in slab[j * nx..(j + 1) * nx].iter_mut().enumerate() {
                    *out = self.distance(Vec3::new(lo.x + i as f64 * cell, y, z)) as f32;
                }
            }
        });
        SdfGrid { origin: lo, cell, nx, ny, nz, data }
    }
}

fn rotate_z(v: Vec3, angle: f64) -> Vec3 {
    let (s, c) = angle.sin_cos();
    Vec3::new(c * v.x - s * v.y, s * v.x + c * v.y, v.z)
}

// ---------------------------------------------------------------------------
// The garage inventory
// ---------------------------------------------------------------------------

/// An interlocking gym mat: **1 inch** (25.4 mm) thick, 60 cm square by
/// default — the EVA foam tile every garage gym has, and the one obstacle
/// here that is *small enough to be invisible* to a policy trained on flat
/// ground. The ankle either clears the lip or catches it.
///
/// Modelled rigid, which is wrong in the safe direction: real foam
/// compresses, so a policy that copes with a rigid 1-inch lip copes with a
/// soft one. The 4 mm edge round is the tile's own chamfer.
pub fn mat(center: Vec3, size: f64, yaw: f64) -> Solid {
    let t = 0.0254;
    Solid::Box {
        center: Vec3::new(center.x, center.y, t * 0.5),
        half: Vec3::new(size * 0.5, size * 0.5, t * 0.5),
        radius: 0.004,
        yaw,
    }
}

/// A flat weightlifting bench, lying in the robot's way.
///
/// Dimensions from the FID/flat-bench standard: pad 122 cm long, 30 cm
/// wide, top surface at 45 cm (the IPF competition bench height is 42–45
/// cm), on two A-frame feet 8 cm off the floor. The pad's 5 cm thickness
/// and the frame tubes are what a shin actually meets.
pub fn bench(center: Vec3, yaw: f64) -> Solid {
    let pad_top = 0.45;
    let pad_t = 0.05;
    let len = 1.22;
    let wide = 0.30;
    let foot_h = 0.08;
    let pad = Solid::Box {
        center: Vec3::new(center.x, center.y, pad_top - pad_t * 0.5),
        half: Vec3::new(len * 0.5, wide * 0.5, pad_t * 0.5),
        radius: 0.02,
        yaw,
    };
    // Two feet, inset from the ends, each a low crossbar plus uprights.
    let mut parts = vec![pad];
    for s in [-1.0, 1.0] {
        let along = rotate_z(Vec3::new(s * (len * 0.5 - 0.12), 0.0, 0.0), yaw);
        let base = Vec3::new(center.x + along.x, center.y + along.y, 0.0);
        parts.push(Solid::Box {
            center: Vec3::new(base.x, base.y, foot_h * 0.5),
            half: Vec3::new(0.03, 0.28, foot_h * 0.5),
            radius: 0.01,
            yaw,
        });
        parts.push(Solid::Box {
            center: Vec3::new(base.x, base.y, (pad_top - pad_t) * 0.5),
            half: Vec3::new(0.03, 0.03, (pad_top - pad_t) * 0.5),
            radius: 0.008,
            yaw,
        });
    }
    Solid::Union(parts)
}

/// A 5-gallon bucket, upright and empty.
///
/// The US standard pail: 14.5 in (36.8 cm) tall, 11.9 in (30.2 cm) outside
/// diameter at the rim tapering to 10.3 in (26.2 cm) at the base, 2.5 mm
/// wall. Hollow, because a foot can land *in* it — the failure mode a solid
/// cylinder would never teach.
pub fn bucket(center: Vec3) -> Solid {
    let h = 0.368;
    let r_top = 0.151;
    let r_bot = 0.131;
    let wall = 0.0025;
    let outer = Solid::Frustum {
        center: Vec3::new(center.x, center.y, h * 0.5),
        r_bottom: r_bot,
        r_top,
        half_height: h * 0.5,
    };
    // Interior cavity: same taper, one wall thickness in, open at the top
    // (raised base so the floor of the bucket stays solid).
    let inner = Solid::Frustum {
        center: Vec3::new(center.x, center.y, h * 0.5 + 0.02),
        r_bottom: r_bot - wall,
        r_top: r_top - wall,
        half_height: h * 0.5,
    };
    Solid::Difference(Box::new(outer), Box::new(inner))
}

/// A plyo box — the 3-in-1 wooden box, standing on its `height` face.
///
/// Standard 20/24/30 in faces; pass 0.508, 0.610, or 0.762 m. The 2 cm
/// edge round is the real chamfer, and it matters: a sharp SDF edge makes
/// the contact normal flip discontinuously right where a foot lands on the
/// corner.
pub fn plyo_box(center: Vec3, height: f64, yaw: f64) -> Solid {
    // The other two faces of a 3-in-1, whichever they are for this height.
    let faces = [0.508, 0.610, 0.762];
    let (mut a, mut b) = (0.0, 0.0);
    for f in faces {
        if (f - height).abs() > 1e-6 {
            if a == 0.0 { a = f } else { b = f }
        }
    }
    Solid::Box {
        center: Vec3::new(center.x, center.y, height * 0.5),
        half: Vec3::new(a * 0.5, b * 0.5, height * 0.5),
        radius: 0.02,
        yaw,
    }
}

/// A step ladder lying **flat on the floor** — rails and rungs, the classic
/// ankle-breaker.
///
/// Type-1A aluminium: rails 8 cm wide spaced 43 cm apart, rungs every 30.5
/// cm (12 in), rung stock 3 cm. Lying down it is a grid of 3 cm bars at
/// ankle height — small enough to step over, exactly the wrong size to
/// stand on.
pub fn ladder_flat(center: Vec3, length: f64, yaw: f64) -> Solid {
    let rail_gap = 0.43;
    let rail_w = 0.08;
    let bar = 0.03;
    let mut parts = Vec::new();
    for s in [-1.0, 1.0] {
        let off = rotate_z(Vec3::new(0.0, s * rail_gap * 0.5, 0.0), yaw);
        parts.push(Solid::Box {
            center: Vec3::new(center.x + off.x, center.y + off.y, bar * 0.5),
            half: Vec3::new(length * 0.5, rail_w * 0.5, bar * 0.5),
            radius: 0.005,
            yaw,
        });
    }
    let n = (length / 0.305).floor() as i32;
    for k in 0..=n {
        let x = -length * 0.5 + k as f64 * 0.305;
        let off = rotate_z(Vec3::new(x, 0.0, 0.0), yaw);
        parts.push(Solid::Box {
            center: Vec3::new(center.x + off.x, center.y + off.y, bar * 0.5),
            half: Vec3::new(bar * 0.5, rail_gap * 0.5, bar * 0.5),
            radius: 0.005,
            yaw,
        });
    }
    Solid::Union(parts)
}

/// A garage scene: the floor plus whatever is scattered on it.
pub fn garage(props: Vec<Solid>) -> Solid {
    let mut parts = vec![Solid::Ground { height: 0.0 }];
    parts.extend(props);
    Solid::Union(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_is_one_inch_and_resolvable() {
        let m = garage(vec![mat(Vec3::zeros(), 0.6, 0.0)]);
        // On top of the mat: the surface is 25.4 mm up.
        assert!(m.distance(Vec3::new(0.0, 0.0, 0.0254)).abs() < 1e-9);
        // Just above it, distance is the gap.
        assert!((m.distance(Vec3::new(0.0, 0.0, 0.0354)) - 0.01).abs() < 1e-9);
        // Off the mat, the floor rules.
        assert!((m.distance(Vec3::new(1.0, 0.0, 0.01)) - 0.01).abs() < 1e-9);
        // And the lip spans more than two cells at the garage resolution.
        assert!(0.0254 / GARAGE_CELL > 2.0);
    }

    #[test]
    fn bucket_is_hollow() {
        let b = bucket(Vec3::zeros());
        // Inside the cavity, above the base: outside the solid (positive).
        assert!(b.distance(Vec3::new(0.0, 0.0, 0.25)) > 0.0);
        // In the wall: inside the solid (negative). At z = 0.25 the taper
        // puts the outer surface at r = 0.1446 and the inner at 0.1410, so
        // the wall is that 3.6 mm band — probe its middle, not its edge.
        assert!(b.distance(Vec3::new(0.143, 0.0, 0.25)) < 0.0);
        // Below the rim height, well outside: positive.
        assert!(b.distance(Vec3::new(0.5, 0.0, 0.2)) > 0.0);
    }

    #[test]
    fn props_bake_and_stand_proud_of_the_floor() {
        let scene = garage(vec![
            plyo_box(Vec3::new(0.8, 0.0, 0.0), 0.508, 0.3),
            bucket(Vec3::new(-0.7, 0.4, 0.0)),
        ]);
        let sdf = scene.bake(
            Vec3::new(-1.5, -1.5, -0.2),
            Vec3::new(1.5, 1.5, 1.0),
            0.02,
        );
        // Above the plyo box's top face (0.508 m) the field reads the gap.
        let d = sdf.sample(Vec3::new(0.8, 0.0, 0.558)).unwrap();
        assert!((d - 0.05).abs() < 0.03, "above plyo box: {d}");
        // Open floor between props is still floor.
        let d = sdf.sample(Vec3::new(0.0, -1.0, 0.1)).unwrap();
        assert!((d - 0.1).abs() < 0.01, "open floor: {d}");
    }

    #[test]
    fn ladder_rungs_are_ankle_height_bars() {
        let l = garage(vec![ladder_flat(Vec3::zeros(), 1.8, 0.0)]);
        // On a rail: solid up to 3 cm.
        assert!(l.distance(Vec3::new(0.0, 0.215, 0.015)) < 0.0);
        // Between rungs, inside the rails: floor level, nothing above it.
        assert!(l.distance(Vec3::new(0.1525, 0.0, 0.02)) > 0.0);
    }
}
