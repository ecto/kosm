//! Shapes and their ray intersections.

use phyz_math::Mat3;

use super::surface::Material;
use super::V;

// ---- geometry ---------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Shape {
    /// `rot` is world → box.
    Box { c: V, half: V, rot: [[f64; 3]; 3] },
    /// `rot` is world → body, for the texture.
    Sphere { c: V, r: f64, rot: [[f64; 3]; 3] },
    /// Along world z, centred at `c`.
    Cylinder { c: V, r: f64, half_h: f64 },
    /// Axis z through `c`.
    Torus { c: V, big_r: f64, small_r: f64 },
    /// A rectangle facing `-z` (the underside of a panel): corner `c`, edges `u`, `v`.
    Panel { c: V, u: V, v: V },
}

#[derive(Clone, Debug)]
pub struct Prim {
    pub(super) shape: Shape,
    pub(super) mat: Material,
}

pub struct Hit {
    pub(super) t: f64,
    pub(super) p: V,
    pub(super) n: V,
    pub(super) prim: usize,
}

pub(super) fn mul(m: &[[f64; 3]; 3], v: V) -> V {
    V::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

pub(super) fn mul_t(m: &[[f64; 3]; 3], v: V) -> V {
    V::new(
        m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z,
        m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z,
        m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z,
    )
}

pub(super) fn arr(m: &Mat3) -> [[f64; 3]; 3] {
    let f = |i: usize, j: usize| m[(i, j)];
    [[f(0, 0), f(0, 1), f(0, 2)], [f(1, 0), f(1, 1), f(1, 2)], [f(2, 0), f(2, 1), f(2, 2)]]
}

pub(super) fn pv(v: phyz_math::Vec3) -> V {
    V::new(v.x, v.y, v.z)
}

pub(super) const EPS: f64 = 1e-6;

impl Shape {
    pub(super) fn hit(&self, o: V, d: V, t_max: f64) -> Option<(f64, V)> {
        match *self {
            Shape::Box { c, half, ref rot } => {
                let ol = mul(rot, o - c);
                let dl = mul(rot, d);
                let (mut tmin, mut tmax) = (EPS, t_max);
                let mut axis = 0;
                let mut sign = 1.0;
                let (ol_a, dl_a, h_a) = (ol.as_array(), dl.as_array(), half.as_array());
                for k in 0..3 {
                    let inv = 1.0 / dl_a[k];
                    let (mut t0, mut t1) = ((-h_a[k] - ol_a[k]) * inv, (h_a[k] - ol_a[k]) * inv);
                    let mut s = -1.0;
                    if t0 > t1 {
                        std::mem::swap(&mut t0, &mut t1);
                        s = 1.0;
                    }
                    if t0 > tmin {
                        tmin = t0;
                        axis = k;
                        sign = s;
                    }
                    tmax = tmax.min(t1);
                    if tmax < tmin {
                        return None;
                    }
                }
                if tmin <= EPS {
                    // inside: leave through the far face
                    return None;
                }
                let mut nl = [0.0; 3];
                nl[axis] = sign;
                Some((tmin, mul_t(rot, V::new(nl[0], nl[1], nl[2]))))
            }
            Shape::Sphere { c, r, .. } => {
                let oc = o - c;
                let b = oc.dot(d);
                let disc = b * b - (oc.norm_sq() - r * r);
                if disc < 0.0 {
                    return None;
                }
                let s = disc.sqrt();
                let t = if -b - s > EPS { -b - s } else { -b + s };
                (t > EPS && t < t_max).then(|| (t, (o + d * t - c).normalize()))
            }
            Shape::Cylinder { c, r, half_h } => {
                let o = o - c;
                let a = d.x * d.x + d.y * d.y;
                let mut best: Option<(f64, V)> = None;
                if a > 1e-12 {
                    let b = o.x * d.x + o.y * d.y;
                    let cc = o.x * o.x + o.y * o.y - r * r;
                    let disc = b * b - a * cc;
                    if disc >= 0.0 {
                        let s = disc.sqrt();
                        for t in [(-b - s) / a, (-b + s) / a] {
                            if t > EPS && t < t_max {
                                let p = o + d * t;
                                if p.z.abs() <= half_h {
                                    best = Some((t, V::new(p.x / r, p.y / r, 0.0)));
                                    break;
                                }
                            }
                        }
                    }
                }
                if d.z.abs() > 1e-12 {
                    for (z, n) in [(half_h, V::z()), (-half_h, -V::z())] {
                        let t = (z - o.z) / d.z;
                        if t > EPS && t < t_max && best.is_none_or(|b| t < b.0) {
                            let p = o + d * t;
                            if p.x * p.x + p.y * p.y <= r * r {
                                best = Some((t, n));
                            }
                        }
                    }
                }
                best
            }
            Shape::Torus { c, big_r, small_r } => {
                // sphere tracing on the exact distance field, inside a bounding sphere
                let oc = o - c;
                let bound = big_r + small_r;
                let b = oc.dot(d);
                let disc = b * b - (oc.norm_sq() - bound * bound);
                if disc < 0.0 {
                    return None;
                }
                let s = disc.sqrt();
                let t_in = (-b - s).max(EPS);
                let t_out = (-b + s).min(t_max);
                if t_out <= t_in {
                    return None;
                }
                let sdf = |p: V| {
                    let q = V::new(p.x.hypot(p.y) - big_r, p.z, 0.0);
                    q.norm() - small_r
                };
                let mut t = t_in;
                for _ in 0..96 {
                    let p = oc + d * t;
                    let dist = sdf(p);
                    if dist < 1e-5 {
                        let rxy = p.x.hypot(p.y).max(1e-12);
                        let q = V::new(p.x / rxy * (rxy - big_r), p.y / rxy * (rxy - big_r), p.z);
                        return Some((t, q.normalize()));
                    }
                    t += dist;
                    if t > t_out {
                        return None;
                    }
                }
                None
            }
            Shape::Panel { c, u, v } => {
                if d.z.abs() < 1e-12 {
                    return None;
                }
                let t = (c.z - o.z) / d.z;
                if t <= EPS || t >= t_max {
                    return None;
                }
                let p = o + d * t - c;
                let (lu, lv) = (u.norm_sq(), v.norm_sq());
                let (a, b) = (p.dot(u) / lu, p.dot(v) / lv);
                ((0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b)).then_some((t, -V::z()))
            }
        }
    }
}

