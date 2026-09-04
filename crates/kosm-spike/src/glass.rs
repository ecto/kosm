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
        }
    }
}

/// Unpolarized Fresnel reflectance.
pub fn fresnel<S: Scalar>(n1: S, n2: S, cos_i: S, cos_t: S) -> S {
    let rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    let rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i);
    S::HALF * (rs * rs + rp * rp)
}

/// Snell refraction of unit `d` at unit normal `n` facing against `d`.
/// `None` on total internal reflection.
pub fn refract<S: Scalar>(d: Vec3<S>, n: Vec3<S>, n1: S, n2: S) -> Option<(Vec3<S>, S, S)> {
    let eta = n1 / n2;
    let cos_i = -d.dot(&n);
    let k = S::ONE - eta * eta * (S::ONE - cos_i * cos_i);
    if k < S::ZERO {
        return None;
    }
    let cos_t = k.sqrt();
    Some((d * eta + n * (eta * cos_i - cos_t), cos_i, cos_t))
}

pub fn reflect<S: Scalar>(d: Vec3<S>, n: Vec3<S>) -> Vec3<S> {
    d - n * (S::TWO * d.dot(&n))
}

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
    }
}
