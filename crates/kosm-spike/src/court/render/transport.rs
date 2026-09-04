//! Light transport: sampling, next-event estimation on the panels, the path.

use std::f64::consts::PI;

use super::geometry::Shape;
use super::surface::hash;
use super::{Scene, V};

// ---- transport --------------------------------------------------------------

/// A small counter-based generator: every sample is a pure function of
/// (pixel, sample, frame).
pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Self {
        Self(hash(seed | 1))
    }
    pub(super) fn next(&mut self) -> f64 {
        self.0 = hash(self.0.wrapping_add(0x9E37_79B9_7F4A_7C15));
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn onb(n: V) -> (V, V) {
    let a = if n.x.abs() > 0.9 { V::y() } else { V::x() };
    let t = a.cross(n).normalize();
    (t, n.cross(t))
}

fn cosine_dir(n: V, u1: f64, u2: f64) -> V {
    let (t, b) = onb(n);
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    (t * (r * phi.cos()) + b * (r * phi.sin()) + n * (1.0 - u1).max(0.0).sqrt()).normalize()
}

fn reflect(d: V, n: V) -> V {
    d - n * (2.0 * d.dot(n))
}

fn schlick(f0: f64, cos: f64) -> f64 {
    f0 + (1.0 - f0) * (1.0 - cos).clamp(0.0, 1.0).powi(5)
}

/// A glossy reflection about the mirror direction: the lobe is a cosine
/// power around it, wide for a rough coat and tight for a polished one.
fn glossy_dir(d: V, n: V, roughness: f64, u1: f64, u2: f64) -> V {
    let r = reflect(d, n).normalize();
    let exponent = 2.0 / (roughness * roughness).max(1e-4) - 2.0;
    let cos_a = u1.powf(1.0 / (exponent + 1.0));
    let sin_a = (1.0 - cos_a * cos_a).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    let (t, b) = onb(r);
    let out = (t * (sin_a * phi.cos()) + b * (sin_a * phi.sin()) + r * cos_a).normalize();
    if out.dot(n) > 0.0 { out } else { r }
}

impl Scene {
    /// Radiance from the panels at `p` with normal `n`, for a Lambertian
    /// surface of albedo 1: one panel picked uniformly, one point on it.
    fn direct(&self, p: V, n: V, rng: &mut Rng) -> V {
        if self.lights.is_empty() {
            return V::zero();
        }
        let pick = (rng.next() * self.lights.len() as f64) as usize;
        let light = &self.prims[self.lights[pick.min(self.lights.len() - 1)]];
        let Shape::Panel { c, u, v } = light.shape else { return V::zero() };
        let q = c + u * rng.next() + v * rng.next();
        let to = q - p;
        let dist2 = to.norm_sq();
        let dist = dist2.sqrt();
        let l = to / dist;
        let cos_s = n.dot(l);
        let cos_l = l.z; // the panel faces down: its normal is -z, the light arrives along -l
        if cos_s <= 0.0 || cos_l <= 0.0 {
            return V::zero();
        }
        if self.occluded(p + n * 1e-4, l, dist - 1e-4) {
            return V::zero();
        }
        let area = u.cross(v).norm();
        let pdf = dist2 / (area * cos_l * self.lights.len() as f64);
        V::splat(self.light_radiance) * (cos_s / PI / pdf)
    }

    /// Radiance along a camera ray.
    pub(super) fn trace(&self, mut o: V, mut d: V, rng: &mut Rng) -> V {
        let mut radiance = V::zero();
        let mut throughput = V::splat(1.0);
        let mut specular = true;
        for bounce in 0..8 {
            let Some(hit) = self.nearest(o, d, f64::INFINITY) else {
                break;
            };
            let s = self.surface(&hit);
            if s.emission.norm_sq() > 0.0 {
                // panels are sampled directly from diffuse bounces; only a
                // camera ray or a specular chain may see them here
                if specular {
                    radiance += throughput.hadamard(s.emission);
                }
                break;
            }
            let mut n = hit.n;
            let entering = d.dot(n) < 0.0;
            if !entering {
                n = -n;
            }
            let cos = -d.dot(n);

            if s.sheet {
                let f = schlick(s.coat, cos);
                if rng.next() < f {
                    o = hit.p + n * 1e-5;
                    d = reflect(d, n);
                } else {
                    throughput = throughput.hadamard(s.tint);
                    o = hit.p - n * 1e-5;
                }
                specular = true;
                continue;
            }

            // the coat takes its Fresnel share; the rest is the diffuse base
            let f = if s.coat > 0.0 { schlick(s.coat, cos) } else { 0.0 };
            if rng.next() < f {
                o = hit.p + n * 1e-5;
                d = glossy_dir(d, n, s.roughness, rng.next(), rng.next());
                specular = s.roughness < 0.05;
                continue;
            }
            let base = throughput.hadamard(s.albedo) / (1.0 - f).max(1e-6);
            radiance += base.hadamard(self.direct(hit.p, n, rng));
            throughput = base;
            specular = false;

            if bounce >= 3 {
                let keep = throughput.max_element().clamp(0.05, 0.95);
                if rng.next() > keep {
                    break;
                }
                throughput = throughput / keep;
            }
            o = hit.p + n * 1e-5;
            d = cosine_dir(n, rng.next(), rng.next());
        }
        radiance
    }
}

/// ACES filmic curve, per channel, then sRGB.
pub(super) fn tonemap(c: V, exposure: f64) -> [u8; 3] {
    let f = |x: f64| {
        let x = x * exposure;
        let y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
        (y.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8
    };
    [f(c.x), f(c.y), f(c.z)]
}

