//! The water surface: the ring model, and the fluid's free surface when there
//! is one. Moved out of the pool level: a surface is engine, not content.

use phyz_math::GRAVITY;
use tang::Vec3 as V;

use super::splash::HeightGrid;
use super::{BLEND, SPONGE, region_inset};

// ---- the waves --------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Ring {
    pub x: f64,
    pub y: f64,
    pub t0: f64,
    pub amp: f64,
    pub wavelength: f64,
}

impl Ring {
    pub fn speed(&self) -> f64 {
        // deep-water gravity waves: c = sqrt(g λ / 2π)
        (GRAVITY * self.wavelength / std::f64::consts::TAU).sqrt()
    }
    pub fn height(&self, x: f64, y: f64, t: f64) -> f64 {
        let dt = t - self.t0;
        if dt <= 0.0 {
            return 0.0;
        }
        let r = (x - self.x).hypot(y - self.y);
        let front = self.speed() * dt;
        let k = std::f64::consts::TAU / self.wavelength;
        // a packet three wavelengths wide riding the front, spreading as 1/√r,
        // dying over a couple of seconds
        let packet = (-((r - front) / (1.5 * self.wavelength)).powi(2)).exp();
        let spread = (0.05 / r.max(0.05)).sqrt();
        let decay = (-dt / 2.5).exp();
        self.amp * packet * spread * decay * (k * (r - front)).cos()
    }
}

#[derive(Clone)]
pub struct Surface {
    pub rings: Vec<Ring>,
    pub t: f64,
    /// When the water is simulated, its free surface replaces the rings
    /// inside the box...
    pub grid: Option<HeightGrid>,
    /// ...and the far field's height carries on beyond it.
    pub far: Option<HeightGrid>,
}

impl Surface {
    pub fn height(&self, x: f64, y: f64) -> f64 {
        // ambient ripple, 1 mm, so still water is not a mirror
        let ambient = 0.0008 * ((7.0 * x + 3.0 * self.t).sin() * (5.0 * y - 2.0 * self.t).cos()) + 0.0005 * ((11.0 * x - 4.0 * y + 1.7 * self.t).sin());
        if let Some(g) = &self.grid {
            // the fluid's surface inside the box (the sub-grid rings from
            // landing drops are baked in once per frame), blending into the
            // far field over the box's last decimetre
            // the fine surface counts inside the sponge band only: at the
            // box wall it dips (the wall layer, the extraction's edge) and a
            // blend across that dip is a lens
            // ...and the blend must be wide and C1: a 10 cm linear blend
            // between two surfaces a millimetre apart is a ring of curvature,
            // and a ring of curvature is a lens (the caustic showed a frame)
            let inset = region_inset(x, y);
            let far = self.far.as_ref().map(|f| f.at(x, y)).unwrap_or(0.0) + ambient;
            if inset > SPONGE + BLEND {
                return g.at(x, y) + ambient;
            }
            if inset <= SPONGE {
                return far;
            }
            let u = (inset - SPONGE) / BLEND;
            let w = u * u * (3.0 - 2.0 * u);
            return w * (g.at(x, y) + ambient) + (1.0 - w) * far;
        }
        let mut h = 0.0;
        for r in &self.rings {
            h += r.height(x, y, self.t);
        }
        h + ambient
    }
    /// Mean height of the fluid surface over the pool's interior (0 = rest).
    pub fn mean_level(&self) -> f64 {
        let Some(g) = &self.grid else { return 0.0 };
        let mut s = 0.0;
        let mut n = 0.0f64;
        for jy in 3..g.ny.saturating_sub(3) {
            for ix in 3..g.nx.saturating_sub(3) {
                let z = g.z[jy * g.nx + ix];
                if z > -1.0 {
                    // wet cells only; the dry corners beyond the disc report the floor
                    s += z;
                    n += 1.0;
                }
            }
        }
        s / n.max(1.0)
    }
    /// The highest the surface gets this frame, plus a margin for the rings.
    pub fn top(&self) -> f64 {
        let grid = self.grid.as_ref().map(|g| g.z.iter().cloned().fold(f64::MIN, f64::max)).unwrap_or(0.0);
        grid + 0.02
    }
    pub fn normal(&self, x: f64, y: f64) -> V<f64> {
        let e = 1e-3;
        let dx = (self.height(x + e, y) - self.height(x - e, y)) / (2.0 * e);
        let dy = (self.height(x, y + e) - self.height(x, y - e)) / (2.0 * e);
        V::new(-dx, -dy, 1.0).normalize()
    }
    /// Where a ray meets the surface, by marching then bisection.
    pub fn hit(&self, o: V<f64>, d: V<f64>, t_max: f64, top_fine: f64, top_far: f64) -> Option<f64> {
        let f = |t: f64| {
            let p = o + d * t;
            p.z - self.height(p.x, p.y)
        };
        let mut t = 0.0;
        let mut prev = f(0.0);
        while t < t_max {
            // 4 mm steps up close, where the splash is; coarser with distance,
            // where the far field is smooth and the pool is fifty metres long
            let mut step = 0.004 + 0.012 * t;
            // and, above the highest water this ray could meet, stride down
            // to it: the crown's tip sets the fine bound only inside the
            // region, the far field is millimetres everywhere else
            let p = o + d * t;
            let bound = if region_inset(p.x, p.y) > -0.5 { top_fine } else { top_far };
            let clearance = p.z - bound;
            if clearance > 0.02 {
                if d.z >= 0.0 {
                    return None; // climbing away from any water
                }
                step = step.max(0.5 * clearance / -d.z);
            }
            let tn = (t + step).min(t_max);
            let cur = f(tn);
            if (prev > 0.0) != (cur > 0.0) {
                let (mut a, mut b) = (t, tn);
                for _ in 0..12 {
                    let m = 0.5 * (a + b);
                    if (f(a) > 0.0) != (f(m) > 0.0) {
                        b = m;
                    } else {
                        a = m;
                    }
                }
                return Some(0.5 * (a + b));
            }
            prev = cur;
            t = tn;
        }
        None
    }
}
