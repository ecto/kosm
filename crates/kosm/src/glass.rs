//! Glass solids the camera can see through and the lamp can shine through.
//!
//! Two shapes: a sphere, and a convex polyhedron given as half-spaces (a cube,
//! a square pyramid, anything convex). Both answer the two questions light
//! transport asks: where does a ray from outside enter, with what normal; and
//! where does a ray from inside leave, with what normal. Everything is generic
//! over `tang::Scalar`, so the same code runs on `Dual` for derivatives.
//!
//! The camera path (`shade_through`) is real: Fresnel splits the ray into a
//! reflected and a refracted share at each face, the refracted ray travels
//! inside, exits or totally internally reflects (up to a few bounces), and
//! whatever leaves the glass is shaded by the scene it lands on. The lamp's
//! highlight is not a lobe: it is the reflected ray hitting the lamp's disc.

use tang::{Scalar, Vec3};

#[derive(Clone)]
pub enum Shape<S: Scalar> {
    Sphere { centre: Vec3<S>, r: S },
    /// Convex polyhedron: outward unit normals and offsets, `n·p ≤ d` inside.
    Convex { planes: Vec<(Vec3<S>, S)>, centre: Vec3<S>, bound_r: S },
    /// The segment `a`→`b` swept by radius `r`: a cylinder between two caps.
    /// The cove's being is one of these, and a capsule is the marble grown a
    /// waist — with `a == b` it answers exactly what `Sphere` answers.
    Capsule { a: Vec3<S>, b: Vec3<S>, r: S },
}

#[derive(Clone)]
pub struct Glass<S: Scalar> {
    pub shape: Shape<S>,
    /// d-line index of refraction; dispersion shape comes from `light::index`.
    pub nd: S,
}

impl<S: Scalar> Shape<S> {
    /// A cube of edge `a` resting on the plate at `(x, y)`, turned by `yaw` about z.
    pub fn cube(x: S, y: S, a: S, yaw: S) -> Self {
        let h = a * S::HALF;
        let c = Vec3::new(x, y, h);
        let (sy, cy) = yaw.sin_cos();
        let ax = Vec3::new(cy, sy, S::ZERO);
        let ay = Vec3::new(-sy, cy, S::ZERO);
        let az = Vec3::new(S::ZERO, S::ZERO, S::ONE);
        let planes = [ax, -ax, ay, -ay, az, -az].iter().map(|n| (*n, n.dot(&c) + h)).collect();
        Shape::Convex { planes, centre: c, bound_r: h * S::from_f64(3f64.sqrt()) }
    }

    /// A square pyramid, base edge `a`, height `hgt`, resting on its base at `(x, y)`, turned by `yaw`.
    pub fn pyramid(x: S, y: S, a: S, hgt: S, yaw: S) -> Self {
        let h = a * S::HALF;
        let (sy, cy) = yaw.sin_cos();
        let ax = Vec3::new(cy, sy, S::ZERO);
        let ay = Vec3::new(-sy, cy, S::ZERO);
        let base = Vec3::new(x, y, S::ZERO);
        let apex = Vec3::new(x, y, hgt);
        let mut planes = vec![(Vec3::new(S::ZERO, S::ZERO, -S::ONE), S::ZERO)];
        // each side face contains the apex and a base edge; normal from the
        // edge midpoint direction tilted up by the face slope
        for dir in [ax, -ax, ay, -ay] {
            let n = (dir * hgt + Vec3::new(S::ZERO, S::ZERO, h)).normalize();
            planes.push((n, n.dot(&apex)));
        }
        let centre = base + Vec3::new(S::ZERO, S::ZERO, hgt * S::from_f64(0.25));
        let bound_r = (h * h * S::TWO + hgt * hgt).sqrt();
        Shape::Convex { planes, centre, bound_r }
    }

    pub fn bounds(&self) -> (Vec3<S>, S) {
        match self {
            Shape::Sphere { centre, r } => (*centre, *r),
            Shape::Convex { centre, bound_r, .. } => (*centre, *bound_r),
            Shape::Capsule { a, b, r } => ((*a + *b) * S::HALF, (*b - *a).norm() * S::HALF + *r),
        }
    }

    /// Entry from outside along `o + t d`: (t, outward normal), t > eps.
    pub fn enter(&self, o: Vec3<S>, d: Vec3<S>) -> Option<(S, Vec3<S>)> {
        match self {
            Shape::Sphere { centre, r } => {
                let oc = o - *centre;
                let b = oc.dot(&d);
                let disc = b * b - (oc.norm_sq() - *r * *r);
                if disc < S::ZERO {
                    return None;
                }
                let t = -b - disc.sqrt();
                (t > S::from_f64(1e-7)).then(|| (t, (o + d * t - *centre) / *r))
            }
            Shape::Convex { planes, .. } => {
                let (mut t_in, mut t_out) = (S::NEG_INFINITY, S::INFINITY);
                let mut n_in = Vec3::zero();
                for (n, dd) in planes {
                    let denom = n.dot(&d);
                    let num = *dd - n.dot(&o);
                    if denom.abs() < S::from_f64(1e-12) {
                        if num < S::ZERO {
                            return None;
                        }
                        continue;
                    }
                    let t = num / denom;
                    if denom < S::ZERO {
                        if t > t_in {
                            t_in = t;
                            n_in = *n;
                        }
                    } else {
                        t_out = t_out.min(t);
                    }
                }
                (t_in > S::from_f64(1e-7) && t_in < t_out).then_some((t_in, n_in))
            }
            Shape::Capsule { a, b, r } => capsule_hit(*a, *b, *r, o, d, true),
        }
    }

    /// Exit from inside along `o + t d`: (t, outward normal at the exit).
    pub fn exit(&self, o: Vec3<S>, d: Vec3<S>) -> Option<(S, Vec3<S>)> {
        match self {
            Shape::Sphere { centre, r } => {
                let oc = o - *centre;
                let b = oc.dot(&d);
                let disc = (b * b - (oc.norm_sq() - *r * *r)).max(S::ZERO);
                let t = -b + disc.sqrt();
                Some((t, (o + d * t - *centre) / *r))
            }
            Shape::Convex { planes, .. } => {
                let mut best: Option<(S, Vec3<S>)> = None;
                for (n, dd) in planes {
                    let denom = n.dot(&d);
                    if denom <= S::from_f64(1e-12) {
                        continue;
                    }
                    let t = (*dd - n.dot(&o)) / denom;
                    if best.as_ref().is_none_or(|b| t < b.0) {
                        best = Some((t, *n));
                    }
                }
                best
            }
            Shape::Capsule { a, b, r } => capsule_hit(*a, *b, *r, o, d, false),
        }
    }

    /// The same solid read in another frame: the one whose origin is
    /// `origin` and whose axes are `u`, `v`, `w`, so a point `p` becomes
    /// `(u·(p−o), v·(p−o), w·(p−o))`.
    ///
    /// The frame has to be orthonormal — this is a rigid move, not a
    /// deformation, and a plane's offset is only carried by `d − n·o`
    /// because the normal keeps its length. It is what lets the caustic
    /// trace put any receiving plane where its plate code expects one.
    pub fn to_frame(&self, origin: Vec3<S>, u: Vec3<S>, v: Vec3<S>, w: Vec3<S>) -> Self {
        let pt = |p: Vec3<S>| {
            let q = p - origin;
            Vec3::new(u.dot(&q), v.dot(&q), w.dot(&q))
        };
        let dir = |n: Vec3<S>| Vec3::new(u.dot(&n), v.dot(&n), w.dot(&n));
        match self {
            Shape::Sphere { centre, r } => Shape::Sphere { centre: pt(*centre), r: *r },
            Shape::Convex { planes, centre, bound_r } => Shape::Convex {
                planes: planes.iter().map(|(n, d)| (dir(*n), *d - n.dot(&origin))).collect(),
                centre: pt(*centre),
                bound_r: *bound_r,
            },
            Shape::Capsule { a, b, r } => Shape::Capsule { a: pt(*a), b: pt(*b), r: *r },
        }
    }
}

/// The ray `o + t d` against the capsule `a`→`b` of radius `r`: the nearest
/// entry when `near`, the farthest exit otherwise, with the outward normal.
///
/// A capsule is three pieces — the cylinder between the cap planes and the
/// two end spheres — and each piece owns the part of the surface the other
/// two do not: a cylinder root only counts while its axial parameter lies in
/// `[0, 1]`, and a cap's root only counts beyond that cap's plane. The body
/// is convex, so once the candidates are filtered the entry is the smallest
/// and the exit the largest, with no ordering left to reason about.
///
/// A segment of zero length is a sphere, and it is answered as one so that
/// the two variants cannot drift on the degenerate case.
fn capsule_hit<S: Scalar>(a: Vec3<S>, b: Vec3<S>, r: S, o: Vec3<S>, d: Vec3<S>, near: bool) -> Option<(S, Vec3<S>)> {
    let ba = b - a;
    let baba = ba.dot(&ba);
    if baba <= S::from_f64(1e-24) {
        let sphere = Shape::Sphere { centre: a, r };
        return if near { sphere.enter(o, d) } else { sphere.exit(o, d) };
    }
    let eps = S::from_f64(1e-7);
    let m = o - a;
    let bard = ba.dot(&d);
    let baoc = ba.dot(&m);
    let mut best: Option<(S, Vec3<S>)> = None;
    let mut take = |t: S, n: Vec3<S>| {
        if near && t <= eps {
            return;
        }
        let better = best.as_ref().is_none_or(|(bt, _)| if near { t < *bt } else { t > *bt });
        if better {
            best = Some((t, n));
        }
    };

    // the cylinder body, in the axial parametrisation that keeps `baba` out
    // of the square roots; `k2` is `|ba|² sin²θ` and vanishes on a ray
    // parallel to the axis, which then only ever meets the caps
    let k2 = baba - bard * bard;
    if k2 > S::from_f64(1e-18) {
        let k1 = baba * m.dot(&d) - baoc * bard;
        let k0 = baba * m.dot(&m) - baoc * baoc - r * r * baba;
        let h = k1 * k1 - k2 * k0;
        if h >= S::ZERO {
            let hs = h.sqrt();
            let t = if near { (-k1 - hs) / k2 } else { (-k1 + hs) / k2 };
            let y = baoc + t * bard;
            if y > S::ZERO && y < baba {
                take(t, (m + d * t - ba * (y / baba)) / r);
            }
        }
    }

    // the caps: a sphere's root counts only on its own side of the cap plane,
    // which is where that sphere is the capsule's surface
    for (centre, beyond_a) in [(a, true), (b, false)] {
        let oc = o - centre;
        let bq = oc.dot(&d);
        let disc = bq * bq - (oc.norm_sq() - r * r);
        if disc < S::ZERO {
            continue;
        }
        let ds = disc.sqrt();
        let t = if near { -bq - ds } else { -bq + ds };
        let y = baoc + t * bard;
        if (beyond_a && y <= S::ZERO) || (!beyond_a && y >= baba) {
            take(t, (oc + d * t) / r);
        }
    }
    best
}

// Snell, Fresnel and the mirror are `kosm-render`'s now: they are laws at an
// interface, not facts about this level's glass, and the pool's water surface
// reads the same three. Re-exported so `glass::refract` still names them.
pub use kosm_render::optics::{fresnel, reflect, refract};

/// A ray that has just entered the glass at `p` heading `d`: follow it until
/// it leaves (up to `max_bounces` internal reflections). Returns the exit
/// point, direction, and the transmitted share (Fresnel at the exit face,
/// times per-metre absorption), or `None` if it never got out.
pub fn walk_inside<S: Scalar>(
    shape: &Shape<S>,
    mut p: Vec3<S>,
    mut d: Vec3<S>,
    n_glass: S,
    absorb_per_m: S,
    max_bounces: usize,
) -> Option<(Vec3<S>, Vec3<S>, S)> {
    let mut throughput = S::ONE;
    for _ in 0..=max_bounces {
        let (t, n_out) = shape.exit(p + d * S::from_f64(1e-7), d)?;
        let q = p + d * t;
        throughput *= (-absorb_per_m * t).exp();
        // the inward-facing normal for refraction out of the glass
        let n_in = -n_out;
        match refract(d, n_in, n_glass, S::ONE) {
            Some((d_out, cos_i, cos_t)) => {
                let r = fresnel(n_glass, S::ONE, cos_i, cos_t);
                return Some((q, d_out, throughput * (S::ONE - r)));
            }
            None => {
                // total internal reflection: keep going
                d = reflect(d, n_in);
                p = q;
            }
        }
    }
    None
}

/// Lift an `f64` shape to `Dual<f64>` constants (the knob is seeded elsewhere).
pub fn to_dual(shape: &Shape<f64>) -> Shape<tang::Dual<f64>> {
    use tang::Dual;
    let v = |p: Vec3<f64>| Vec3::new(Dual::constant(p.x), Dual::constant(p.y), Dual::constant(p.z));
    match shape {
        Shape::Sphere { centre, r } => Shape::Sphere { centre: v(*centre), r: Dual::constant(*r) },
        Shape::Convex { planes, centre, bound_r } => Shape::Convex {
            planes: planes.iter().map(|(n, d)| (v(*n), Dual::constant(*d))).collect(),
            centre: v(*centre),
            bound_r: Dual::constant(*bound_r),
        },
        Shape::Capsule { a, b, r } => Shape::Capsule { a: v(*a), b: v(*b), r: Dual::constant(*r) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang::Dual;

    const A: Vec3<f64> = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    const B: Vec3<f64> = Vec3 { x: 0.0, y: 0.0, z: 1.0 };
    const R: f64 = 0.35;

    /// Distance to the capsule's surface, the independent statement of the
    /// same solid: a hit is a zero of this, and the normal is its gradient.
    fn sdf(p: Vec3<f64>, a: Vec3<f64>, b: Vec3<f64>, r: f64) -> f64 {
        let (ba, pa) = (b - a, p - a);
        let h = if ba.norm_sq() > 0.0 { (pa.dot(&ba) / ba.norm_sq()).clamp(0.0, 1.0) } else { 0.0 };
        (pa - ba * h).norm() - r
    }

    fn grad(p: Vec3<f64>, a: Vec3<f64>, b: Vec3<f64>, r: f64) -> Vec3<f64> {
        let h = 1e-6;
        let d = |e: Vec3<f64>| (sdf(p + e * h, a, b, r) - sdf(p - e * h, a, b, r)) / (2.0 * h);
        Vec3::new(d(Vec3::x()), d(Vec3::y()), d(Vec3::z())).normalize()
    }

    #[test]
    fn the_capsule_is_a_cylinder_between_two_spheres() {
        let cap = Shape::Capsule { a: A, b: B, r: R };
        let d = Vec3::new(-1.0, 0.0, 0.0);
        // through the body, through the lower cap, through the upper cap
        for (h, x) in [(0.5, R), (-0.2, (R * R - 0.04f64).sqrt()), (1.25, (R * R - 0.0625f64).sqrt())] {
            let o = Vec3::new(5.0, 0.0, h);
            let (t, n) = cap.enter(o, d).expect("the ray meets the capsule");
            assert!((t - (5.0 - x)).abs() < 1e-12, "entry at height {h}: {t} vs {}", 5.0 - x);
            let p = o + d * t;
            assert!(sdf(p, A, B, R).abs() < 1e-12, "the entry point is on the surface");
            let g = grad(p, A, B, R);
            assert!((n - g).norm() < 1e-6, "normal at height {h}: {n:?} vs {g:?}");
        }
        // and misses when it passes outside the swept radius
        assert!(cap.enter(Vec3::new(5.0, 0.0, 1.4), d).is_none());
    }

    #[test]
    fn the_capsule_exits_at_the_far_surface() {
        let cap = Shape::Capsule { a: A, b: B, r: R };
        // sideways out of the cylinder, straight up through the top cap, and
        // a slanted ray that leaves through the cap it did not start under
        let cases = [
            (Vec3::new(0.0, 0.0, 0.5), Vec3::new(-1.0, 0.0, 0.0), 0.35),
            (Vec3::new(0.0, 0.0, 0.9), Vec3::new(0.0, 0.0, 1.0), 0.45),
            (Vec3::new(0.0, 0.0, 0.2), Vec3::new(0.0, 0.0, -1.0), 0.55),
        ];
        for (o, dir, want) in cases {
            let (t, n) = cap.exit(o, dir).expect("a ray from inside leaves");
            assert!((t - want).abs() < 1e-12, "exit from {o:?} along {dir:?}: {t} vs {want}");
            let p = o + dir * t;
            assert!(sdf(p, A, B, R).abs() < 1e-12);
            assert!((n - grad(p, A, B, R)).norm() < 1e-6, "exit normal {n:?}");
        }
        // a slant with no closed form: the exit is on the surface, and just
        // past it is outside
        let (o, dir) = (Vec3::new(0.0, 0.0, 0.9), Vec3::new(1.0, 0.0, 1.0).normalize());
        let (t, n) = cap.exit(o, dir).unwrap();
        assert!(sdf(o + dir * t, A, B, R).abs() < 1e-12);
        assert!(sdf(o + dir * (t + 1e-6), A, B, R) > 0.0);
        assert!(sdf(o + dir * (t - 1e-6), A, B, R) < 0.0);
        assert!((n - grad(o + dir * t, A, B, R)).norm() < 1e-6);
    }

    #[test]
    fn a_capsule_of_no_length_is_the_sphere() {
        let c = Vec3::new(0.1, -0.2, 0.4);
        let (cap, sph) = (Shape::Capsule { a: c, b: c, r: 0.3 }, Shape::Sphere { centre: c, r: 0.3 });
        for dir in [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.3, 0.6, -0.9).normalize(), Vec3::new(0.0, 1.0, 0.0)] {
            let o = c - dir * 2.0;
            match (cap.enter(o, dir), sph.enter(o, dir)) {
                (Some((tc, nc)), Some((ts, ns))) => {
                    assert!((tc - ts).abs() < 1e-15 && (nc - ns).norm() < 1e-15);
                }
                (a, b) => panic!("entry disagrees: {:?} vs {:?}", a.is_some(), b.is_some()),
            }
            let (tc, nc) = cap.exit(c, dir).unwrap();
            let (ts, ns) = sph.exit(c, dir).unwrap();
            assert!((tc - ts).abs() < 1e-15 && (nc - ns).norm() < 1e-15);
        }
        let (bc, bs) = (cap.bounds(), sph.bounds());
        assert!((bc.1 - bs.1).abs() < 1e-15 && (bc.0 - bs.0).norm() < 1e-15);
    }

    #[test]
    fn the_capsule_entry_differentiates_in_its_radius() {
        let d = Vec3::new(-1.0, 0.0, 0.0);
        let entry = |r: f64, h: f64| {
            let cap = Shape::Capsule { a: A, b: B, r };
            cap.enter(Vec3::new(5.0, 0.0, h), d).unwrap().0
        };
        // the body, where dt/dr is −1, and the cap, where it is not
        for h in [0.5, 1.25] {
            let cap = Shape::Capsule {
                a: Vec3::new(Dual::constant(A.x), Dual::constant(A.y), Dual::constant(A.z)),
                b: Vec3::new(Dual::constant(B.x), Dual::constant(B.y), Dual::constant(B.z)),
                r: Dual::new(R, 1.0),
            };
            let o = Vec3::new(Dual::constant(5.0), Dual::constant(0.0), Dual::constant(h));
            let dd = Vec3::new(Dual::constant(-1.0), Dual::constant(0.0), Dual::constant(0.0));
            let (t, _) = cap.enter(o, dd).unwrap();
            let eps = 1e-6;
            let fd = (entry(R + eps, h) - entry(R - eps, h)) / (2.0 * eps);
            assert!((t.dual - fd).abs() < 1e-6, "at height {h}: dual {} vs fd {fd}", t.dual);
        }
    }
}
