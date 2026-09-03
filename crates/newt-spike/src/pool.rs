//! A pool, and a watermelon dropped into it.
//!
//! What is physics here and what is not, said plainly:
//!
//! * **The watermelon is a phyz rigid body.** It falls under gravity, and once
//!   it meets the water it feels Archimedes' buoyancy from the submerged cap
//!   of its volume and a quadratic drag, applied as external generalized
//!   forces through phyz's own step. A watermelon's density is about 950
//!   kg/m³, so it plunges, slows, and comes back up to float with a twentieth
//!   of itself above the surface. That is why the video ends the way it does.
//! * **The water surface is a wave field, not a fluid solve.** The impact
//!   launches a ring of gravity–capillary waves with the deep-water
//!   dispersion of its wavelength, decaying as they spread and in time; the
//!   melon's bobbing sends out smaller rings; a little ambient ripple keeps
//!   the surface alive. phyz-particle could do the splash itself and one day
//!   should; today the surface is where a fluid solver's answer would go.
//! * **The light is real.** Sunlight through the surface refracts by Snell's
//!   law and is traced onto the tiles as a caustic every frame. Camera rays
//!   Fresnel-split at the surface into a sky reflection and a refracted path
//!   into water that absorbs red first, then hit tiles, walls, or the melon.
//!   The sun's highlight is the sun's reflected disc.
//!
//! Same conventions as the rest: metres, z up, the water at rest at z = 0.

use std::path::Path;
use rayon::prelude::*;

use phyz::Simulator;
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder, State};
use tang::Vec3 as V;

use crate::glass::{fresnel, refract, reflect};

// ---- the pool ---------------------------------------------------------------

// An Olympic pool: 50 m by 25 m; FINA's minimum depth is 2 m.
pub const POOL_X: f64 = 25.0; // half-lengths of the water
pub const POOL_Y: f64 = 12.5;
pub const DEPTH: f64 = 2.0;
/// The fine MPM box around the melon: half-width (NEWT_BOX overrides) and
/// depth. Beyond it the water is the far field (see `far`).
pub fn box_half() -> f64 {
    static HALF: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *HALF.get_or_init(|| std::env::var("NEWT_BOX").ok().and_then(|v| v.parse().ok()).unwrap_or(1.25))
}
pub const BOX_DEPTH: f64 = DEPTH; // the full depth: a floor the melon could fall through is no floor
/// The box's outer band where the fluid's velocity is damped so waves leave
/// instead of reflecting off a wall two metres from the splash.
pub const SPONGE: f64 = 0.25;
/// Width of the band, inside the sponge, over which the rendered surface
/// fades from the fine grid to the far field.
pub const BLEND: f64 = 0.4;
/// Frames per second of a recording (NEWT_FPS overrides).
pub fn fps() -> f64 {
    std::env::var("NEWT_FPS").ok().and_then(|v| v.parse().ok()).unwrap_or(60.0)
}
const COPING: f64 = 0.06; // deck height above the water line
const N_WATER: f64 = 1.333;
/// Absorption per metre, RGB: red goes first, which is why deep water is blue.
const ABSORB: [f64; 3] = [0.45, 0.10, 0.04];

// ---- the melon --------------------------------------------------------------

pub const MELON_AXES: [f64; 3] = [0.15, 0.105, 0.105];
fn melon_density() -> f64 {
    std::env::var("NEWT_MELON_RHO").ok().and_then(|v| v.parse().ok()).unwrap_or(950.0)
}
const WATER_DENSITY: f64 = 1000.0;

// ---- the sun ----------------------------------------------------------------

fn sun_dir() -> V<f64> {
    V::new(-0.35, -0.45, 0.82).normalize()
}
const SUN_IRRADIANCE: f64 = 1.05;
const SUN_DISC_COS: f64 = 0.99995; // an angular radius of about 0.6°

// ---- the grandstand ---------------------------------------------------------

/// A stand along the far long side: stepped rows from the deck, a seat every
/// 0.6 m, most of them taken.
pub const STAND_Y0: f64 = POOL_Y + 3.0;
pub const STAND_ROWS: usize = 14;
pub const STAND_RISE: f64 = 0.45;
pub const STAND_TREAD: f64 = 0.85;
const SEAT_PITCH: f64 = 0.6;

fn hash2(i: i64, j: i64) -> f64 {
    let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h & 0xFFFFFF) as f64 / 16777216.0
}

/// Ray against the stand's steps (a union of boxes): distance and normal.
fn stand_hit(o: V<f64>, d: V<f64>) -> Option<(f64, V<f64>)> {
    let mut best: Option<(f64, V<f64>)> = None;
    let y_back = STAND_Y0 + STAND_ROWS as f64 * STAND_TREAD;
    for r in 0..STAND_ROWS {
        let lo = V::new(-POOL_X, STAND_Y0 + r as f64 * STAND_TREAD, COPING);
        let hi = V::new(POOL_X, y_back, COPING + (r + 1) as f64 * STAND_RISE);
        // slab test
        let mut t0 = 1e-6f64;
        let mut t1 = f64::INFINITY;
        let mut n_in = V::new(0.0, 0.0, 1.0);
        for k in 0..3 {
            let (oo, dd, l, h) = match k { 0 => (o.x, d.x, lo.x, hi.x), 1 => (o.y, d.y, lo.y, hi.y), _ => (o.z, d.z, lo.z, hi.z) };
            let mut axis = V::zero();
            if dd.abs() < 1e-12 {
                if oo < l || oo > h { t0 = f64::INFINITY; }
                continue;
            }
            let (mut ta, mut tb) = ((l - oo) / dd, (h - oo) / dd);
            let mut sign = -1.0;
            if ta > tb { std::mem::swap(&mut ta, &mut tb); sign = 1.0; }
            if ta > t0 {
                t0 = ta;
                match k { 0 => axis.x = sign, 1 => axis.y = sign, _ => axis.z = sign }
                n_in = axis;
            }
            t1 = t1.min(tb);
        }
        if t0 < t1 && t0.is_finite() && best.is_none_or(|b| t0 < b.0) {
            best = Some((t0, n_in));
        }
    }
    best
}

/// A seated person at seat `i` of row `r`, if there is one: shirt colour,
/// and the signed distance from `p` (capsule torso, sphere head).
fn person_sdf(p: V<f64>, i: i64, r: i64) -> Option<(f64, [f64; 3], bool)> {
    if r < 0 || r >= STAND_ROWS as i64 {
        return None;
    }
    let h0 = hash2(i, r);
    if h0 > 0.85 {
        return None; // an empty seat
    }
    let x = -POOL_X + 0.3 + i as f64 * SEAT_PITCH + (hash2(i + 7, r) - 0.5) * 0.12;
    if x.abs() > POOL_X - 0.3 {
        return None;
    }
    let y = STAND_Y0 + r as f64 * STAND_TREAD + 0.45;
    let z = COPING + (r + 1) as f64 * STAND_RISE;
    let lean = (hash2(i + 3, r + 11) - 0.5) * 0.2;
    // torso: a capsule from the seat to the shoulders
    let a = V::new(x, y, z + 0.12);
    let b = V::new(x + lean, y - 0.05, z + 0.72);
    let ab = b - a;
    let t = ((p - a).dot(&ab) / ab.norm_sq()).clamp(0.0, 1.0);
    let d_torso = (p - (a + ab * t)).norm() - 0.19;
    let head = V::new(x + lean * 1.2, y - 0.06, z + 0.92);
    let d_head = (p - head).norm() - 0.11;
    let hue = hash2(i + 101, r + 5);
    let shirt = match (hue * 6.0) as i32 {
        0 => [0.85, 0.20, 0.15],
        1 => [0.95, 0.90, 0.85],
        2 => [0.15, 0.30, 0.75],
        3 => [0.95, 0.75, 0.15],
        4 => [0.20, 0.55, 0.30],
        _ => [0.25, 0.25, 0.30],
    };
    if d_head < d_torso { Some((d_head, [0.80, 0.60, 0.48], true)) } else { Some((d_torso, shirt, false)) }
}

fn crowd_sdf(p: V<f64>) -> (f64, [f64; 3]) {
    let i = ((p.x + POOL_X - 0.3) / SEAT_PITCH).round() as i64;
    let r = ((p.y - STAND_Y0 - 0.45) / STAND_TREAD).round() as i64;
    // the nearest seat, and the row behind (heads poke up between)
    let mut best = (1e9, [0.0; 3]);
    for (di, dr) in [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)] {
        if let Some((d, c, _)) = person_sdf(p, i + di, r + dr) {
            if d < best.0 { best = (d, c); }
        }
    }
    best
}

/// Sphere-trace the crowd inside the stand's bounding box.
fn crowd_hit(o: V<f64>, d: V<f64>, t_max: f64) -> Option<(f64, V<f64>, [f64; 3])> {
    // the box the people occupy
    let lo = V::new(-POOL_X, STAND_Y0, COPING);
    let hi = V::new(POOL_X, STAND_Y0 + STAND_ROWS as f64 * STAND_TREAD, COPING + STAND_ROWS as f64 * STAND_RISE + 1.1);
    let mut t0 = 1e-6f64;
    let mut t1 = t_max;
    for k in 0..3 {
        let (oo, dd, l, h) = match k { 0 => (o.x, d.x, lo.x, hi.x), 1 => (o.y, d.y, lo.y, hi.y), _ => (o.z, d.z, lo.z, hi.z) };
        if dd.abs() < 1e-12 {
            if oo < l || oo > h { return None; }
            continue;
        }
        let (mut ta, mut tb) = ((l - oo) / dd, (h - oo) / dd);
        if ta > tb { std::mem::swap(&mut ta, &mut tb); }
        t0 = t0.max(ta);
        t1 = t1.min(tb);
    }
    if t0 >= t1 {
        return None;
    }
    let mut t = t0;
    for _ in 0..200 {
        let p = o + d * t;
        let (dist, col) = crowd_sdf(p);
        if dist < 0.004 {
            let e = 0.003;
            let n = V::new(
                crowd_sdf(p + V::new(e, 0.0, 0.0)).0 - crowd_sdf(p - V::new(e, 0.0, 0.0)).0,
                crowd_sdf(p + V::new(0.0, e, 0.0)).0 - crowd_sdf(p - V::new(0.0, e, 0.0)).0,
                crowd_sdf(p + V::new(0.0, 0.0, e)).0 - crowd_sdf(p - V::new(0.0, 0.0, e)).0,
            )
            .normalize();
            return Some((t, n, col));
        }
        t += dist.max(0.01);
        if t > t1 {
            return None;
        }
    }
    None
}

fn shade_stand(p: V<f64>, n: V<f64>, base: [f64; 3]) -> [f64; 3] {
    let s = sun_dir();
    let lit = SUN_IRRADIANCE * n.dot(&s).max(0.0);
    let sk = sky(n);
    let mut c = [0.0; 3];
    for k in 0..3 {
        c[k] = base[k] * (0.35 * sk[k] + lit);
    }
    // haze with distance: the far end of a 50 m stand
    let dist = (p - V::new(0.0, 0.0, 0.0)).norm();
    let f = (-(dist / 120.0)).exp();
    for k in 0..3 {
        c[k] = c[k] * f + [0.66, 0.80, 0.94][k] * (1.0 - f) * 0.9;
    }
    c
}

/// The grandstand and its crowd along a ray: steps first, then people.
fn stand_and_crowd(o: V<f64>, d: V<f64>) -> Option<(f64, [f64; 3])> {
    let steps = stand_hit(o, d);
    let t_lim = steps.map(|(t, _)| t).unwrap_or(f64::INFINITY);
    if let Some((t, n, col)) = crowd_hit(o, d, t_lim.min(400.0)) {
        let p = o + d * t;
        return Some((t, shade_stand(p, n, col)));
    }
    steps.map(|(t, n)| {
        let p = o + d * t;
        // concrete steps, with a tread/riser contrast
        let base = if n.z > 0.5 { [0.62, 0.60, 0.56] } else { [0.48, 0.47, 0.45] };
        (t, shade_stand(p, n, base))
    })
}

// ---- whitewater -------------------------------------------------------------

/// A fleck of foam riding the surface: born where fast water breaks the
/// surface, drifting with the momentum it had, gone in a couple of seconds.
#[derive(Clone, Copy)]
pub struct Foam {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub age: f64,
    pub life: f64,
    pub size: f64,
}

/// Foam binned for the renderer: coverage at a point is the sum of the
/// flecks within their radius, from the bins around it.
pub struct FoamField {
    origin: [f64; 2],
    cell: f64,
    nx: usize,
    ny: usize,
    bins: Vec<Vec<Foam>>,
}

impl FoamField {
    pub fn build(foam: &[Foam], half: f64) -> Self {
        let cell = 0.1;
        let nx = (2.0 * half / cell) as usize + 1;
        let ny = nx;
        let mut bins = vec![Vec::new(); nx * ny];
        for f in foam {
            let i = ((f.x + half) / cell).floor();
            let j = ((f.y + half) / cell).floor();
            if i >= 0.0 && j >= 0.0 && (i as usize) < nx && (j as usize) < ny {
                bins[j as usize * nx + i as usize].push(*f);
            }
        }
        Self { origin: [-half, -half], cell, nx, ny, bins }
    }
    pub fn empty() -> Self {
        Self { origin: [0.0, 0.0], cell: 1.0, nx: 0, ny: 0, bins: Vec::new() }
    }
    /// Coverage in 0..1 at (x, y).
    pub fn coverage(&self, x: f64, y: f64) -> f64 {
        if self.nx == 0 {
            return 0.0;
        }
        let gi = ((x - self.origin[0]) / self.cell).floor() as i64;
        let gj = ((y - self.origin[1]) / self.cell).floor() as i64;
        let mut cov = 0.0;
        for dj in -1..=1i64 {
            for di in -1..=1i64 {
                let (i, j) = (gi + di, gj + dj);
                if i < 0 || j < 0 || i >= self.nx as i64 || j >= self.ny as i64 {
                    continue;
                }
                for f in &self.bins[j as usize * self.nx + i as usize] {
                    let r2 = (x - f.x).powi(2) + (y - f.y).powi(2);
                    let s2 = f.size * f.size;
                    if r2 < s2 {
                        let fade = 1.0 - (f.age / f.life).clamp(0.0, 1.0);
                        cov += (1.0 - r2 / s2) * fade;
                    }
                }
            }
        }
        0.85 * (1.0 - (-cov * 0.9).exp())
    }
}

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
    fn speed(&self) -> f64 {
        // deep-water gravity waves: c = sqrt(g λ / 2π)
        (GRAVITY * self.wavelength / std::f64::consts::TAU).sqrt()
    }
    fn height(&self, x: f64, y: f64, t: f64) -> f64 {
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
    pub grid: Option<crate::splash::HeightGrid>,
    /// ...and the far field's height carries on beyond it.
    pub far: Option<crate::splash::HeightGrid>,
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
            let half = box_half();
            let inset = half - x.abs().max(y.abs());
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
                s += g.z[jy * g.nx + ix];
                n += 1.0;
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
    fn hit(&self, o: V<f64>, d: V<f64>, t_max: f64) -> Option<f64> {
        let f = |t: f64| {
            let p = o + d * t;
            p.z - self.height(p.x, p.y)
        };
        let mut t = 0.0;
        let mut prev = f(0.0);
        while t < t_max {
            // 4 mm steps up close, where the splash is; coarser with distance,
            // where the far field is smooth and the pool is fifty metres long
            let step = 0.004 + 0.012 * t;
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

// ---- the melon as geometry --------------------------------------------------

#[derive(Clone, Copy)]
pub struct Melon {
    pub centre: V<f64>,
    /// Long axis heading (unit, horizontal-ish).
    pub axis: V<f64>,
}

impl Melon {
    fn frame(&self) -> (V<f64>, V<f64>, V<f64>) {
        let a = self.axis;
        let up = V::new(0.0, 0.0, 1.0);
        let b = up.cross(&a).normalize();
        let c = a.cross(&b);
        (a, b, c)
    }
    /// Ray–ellipsoid: transform into the unit sphere.
    fn hit(&self, o: V<f64>, d: V<f64>) -> Option<(f64, V<f64>)> {
        let (a, b, c) = self.frame();
        let rel = o - self.centre;
        let ol = V::new(rel.dot(&a) / MELON_AXES[0], rel.dot(&b) / MELON_AXES[1], rel.dot(&c) / MELON_AXES[2]);
        let dl = V::new(d.dot(&a) / MELON_AXES[0], d.dot(&b) / MELON_AXES[1], d.dot(&c) / MELON_AXES[2]);
        let aa = dl.norm_sq();
        let bb = ol.dot(&dl);
        let cc = ol.norm_sq() - 1.0;
        let disc = bb * bb - aa * cc;
        if disc < 0.0 {
            return None;
        }
        let t = (-bb - disc.sqrt()) / aa;
        if t < 1e-6 {
            return None;
        }
        let pl = ol + dl * t;
        // normal: gradient of the ellipsoid, back in world
        let nl = V::new(pl.x / MELON_AXES[0], pl.y / MELON_AXES[1], pl.z / MELON_AXES[2]);
        let n = (a * nl.x + b * nl.y + c * nl.z).normalize();
        Some((t, n))
    }
    /// Rind colour at a surface point: dark and light green stripes running
    /// pole to pole, wobbling, with a paler belly patch where it lay in the field.
    fn albedo(&self, p: V<f64>) -> [f64; 3] {
        let (a, b, c) = self.frame();
        let rel = p - self.centre;
        let (u, v, w) = (rel.dot(&a) / MELON_AXES[0], rel.dot(&b) / MELON_AXES[1], rel.dot(&c) / MELON_AXES[2]);
        let phi = w.atan2(v);
        let wobble = 0.35 * (6.0 * u + 2.0 * phi).sin() + 0.2 * (13.0 * u).sin();
        let stripe = (9.0 * phi + wobble).sin();
        let s = ((stripe + 0.15) * 3.0).clamp(-1.0, 1.0) * 0.5 + 0.5;
        let dark = [0.07, 0.24, 0.09];
        let light = [0.52, 0.70, 0.32];
        let mut col = [0.0; 3];
        for k in 0..3 {
            col[k] = dark[k] * s + light[k] * (1.0 - s);
        }
        // belly
        let belly = ((-w - 0.75) * 6.0).clamp(0.0, 1.0);
        for k in 0..3 {
            col[k] = col[k] * (1.0 - belly) + [0.85, 0.82, 0.55][k] * belly;
        }
        col
    }
}

// ---- physics ----------------------------------------------------------------

pub struct Drop {
    pub model: Model,
    pub state: State,
    pub sim: Simulator,
    pub surface: Surface,
    pub last_splash: f64,
    pub entered: bool,
    /// The simulated water, when the splash is on.
    pub water: Option<crate::splash::Water>,
    pub droplets: Vec<crate::splash::Droplet>,
    /// Foam on the surface.
    pub foam: Vec<Foam>,
    /// Last frame's fine surface, for its rate of rise.
    surface_prev: Option<crate::splash::HeightGrid>,
    /// Last frame's fluid force on the melon, for the log.
    pub fluid_force: V<f64>,
    /// The pool beyond the box.
    pub far: Option<crate::far::Far>,
}

impl Drop {
    pub fn new(height: f64) -> Self {
        let vol = 4.0 / 3.0 * std::f64::consts::PI * MELON_AXES[0] * MELON_AXES[1] * MELON_AXES[2];
        let m = melon_density() * vol;
        let r = (MELON_AXES[0] * MELON_AXES[1] * MELON_AXES[2]).cbrt();
        let i = 0.4 * m * r * r;
        let mut model = ModelBuilder::new()
            .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
            .dt(1e-3)
            .add_free_body("melon", -1, SpatialTransform::identity(), SpatialInertia::new(m, Vec3::zeros(), Mat3::from_diagonal(&Vec3::new(i, i, i))))
            .build();
        model.bodies[0].geometry = Some(Geometry::Sphere { radius: r });
        let mut state = model.default_state();
        state.q[3] = -0.15;
        state.q[4] = 0.05;
        state.q[5] = height;
        Self {
            model,
            state,
            sim: Simulator::new(),
            surface: Surface { rings: Vec::new(), t: 0.0, grid: None, far: None },
            last_splash: -1.0,
            entered: false,
            water: None,
            droplets: Vec::new(),
            foam: Vec::new(),
            surface_prev: None,
            fluid_force: V::zero(),
            far: None,
        }
    }

    /// Turn the water on: a dense MPM in the box around the melon, a wave
    /// field over the rest of the pool.
    pub fn with_water(mut self, h: f64) -> Self {
        // speed of sound ~ 45 m/s: at 5 m/s the impact pressure compresses the
        // water under a percent, which is what stops a melon instead of
        // letting it plough through
        let bulk = 2.0e6;
        let cs = (bulk / 1000.0f64).sqrt();
        let dt = 0.35 * h / cs;
        let mut water = crate::splash::Water::fill(h, dt, 1.0, bulk);
        // NEWT_GPU=0 keeps the solver on the CPU
        if std::env::var("NEWT_GPU").map(|v| v != "0").unwrap_or(true) {
            let subs = (self.model.dt / dt).ceil() as u32;
            match water.enable_gpu(subs.max(256)) {
                Ok(()) => println!("splash  water on the GPU"),
                Err(e) => println!("splash  water on the CPU ({e})"),
            }
        }
        // pack the fill down before anything arrives, and take the rest level
        water.settle(2.0);
        self.water = Some(water);
        self.far = Some(crate::far::Far::new(0.1));
        self
    }

    /// One 1 ms physics step when the water is simulated: the fluid pushes
    /// on the melon through the grid, phyz moves the melon.
    fn step_with_water(&mut self) {
        let melon = self.melon();
        let vel = V::new(self.state.v[3], self.state.v[4], self.state.v[5]);
        let water = self.water.as_mut().expect("water");
        let subs = (self.model.dt / water.dt).ceil() as usize;
        let body = crate::splash::Body {
            centre: Vec3::new(melon.centre.x, melon.centre.y, melon.centre.z),
            axis: Vec3::new(melon.axis.x, melon.axis.y, melon.axis.z),
            vel: Vec3::new(vel.x, vel.y, vel.z),
        };
        let f = water.step_block(&body, subs);
        let fv = V::new(f.x, f.y, f.z);
        if fv.norm() > self.fluid_force.norm() { self.fluid_force = fv; }
        // NEWT_HOLD=<z>: pin the melon at that depth and just read the force
        // the water puts on it; a hydrostatic check of the coupling
        if let Some(z) = std::env::var("NEWT_HOLD").ok().and_then(|v| v.parse::<f64>().ok()) {
            self.state.q[5] = z;
            self.state.v[3] = 0.0;
            self.state.v[4] = 0.0;
            self.state.v[5] = 0.0;
            self.state.time += self.model.dt;
            self.surface.t = self.state.time;
            return;
        }
        self.state.ctrl = phyz_math::DVec::from_slice(&[0.0, 0.0, 0.0, f.x, f.y, f.z]);
        self.state.v[0] = 0.0;
        self.state.v[1] = 0.0;
        self.state.v[2] = 0.0;
        self.sim.step_with_contacts(&self.model, &mut self.state, -DEPTH, &Default::default());
        self.state.q[0] = 0.0;
        self.state.q[1] = 0.0;
        self.state.q[2] = 0.0;
        self.surface.t = self.state.time;
    }

    /// After a frame's worth of steps: read the surface and the drops, and
    /// launch a small ring wherever a drop from last frame has landed. The
    /// grid is too coarse to show a 1 cm drop's ripple; the ring model is
    /// the sub-grid physics for it, scaled by the drop's speed.
    pub fn read_water(&mut self) {
        // the surface and the drop/foam candidates come off the GPU together;
        // the pool itself never crosses the bus (see splash::candidates)
        let picked = self.water.as_mut().map(|w| w.candidates(0.02, 0.02, 0.6, 1.5, 0.6));
        if let (Some((g, cand)), Some(w)) = (picked, &self.water) {
            let off = w.level_offset;
            // the drops: the highest 250 of the candidates over the surface
            let mut up: Vec<(f64, crate::splash::Droplet)> = cand
                .iter()
                .filter_map(|d| {
                    let s = g.at(d.pos.x, d.pos.y) + off;
                    if d.pos.z <= s + 0.02 { None } else { Some((d.pos.z - s, *d)) }
                })
                .collect();
            up.sort_by(|a, b| b.0.total_cmp(&a.0));
            up.truncate(250);
            let now: Vec<crate::splash::Droplet> = up.into_iter().map(|(_, d)| d).collect();
            let t = self.state.time;
            let mut landed = 0;
            for d in &self.droplets {
                let z_here = g.at(d.pos.x, d.pos.y) + off;
                let still_up = now.iter().any(|n| (n.pos - d.pos).norm() < 0.05 && n.pos.z > z_here + 0.02);
                if !still_up && d.pos.z < z_here + 0.12 && d.vel.z < -0.3 && landed < 10 && self.surface.rings.len() < 80 {
                    let speed = d.vel.norm();
                    self.surface.rings.push(Ring { x: d.pos.x, y: d.pos.y, t0: t, amp: (0.0015 * speed).min(0.006), wavelength: 0.06 });
                    let _ = off;
                    landed += 1;
                }
            }
            // rings that have died out
            self.surface.rings.retain(|r| t - r.t0 < 1.5);
            self.droplets = now;
            // whitewater: fast water at the surface entrains air. A fleck
            // is born where a fluid particle within a cell and a half of
            // the surface moves faster than a metre a second, more the
            // faster; it keeps that momentum, slowed, and lives a couple
            // of seconds. (Ihmsen et al.'s trapped-air potential, reduced
            // to what the picked candidates can tell us -- which is all of
            // the fast near-surface water, because that is what the picker
            // is asked for.)
            let frame_dt = 1.0 / fps();
            let mut seed = (t * 1e6) as u64 | 1;
            let mut rnd = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 11) as f64 / (1u64 << 53) as f64
            };
            let budget = 40_000usize.saturating_sub(self.foam.len());
            let mut born = 0;
            let (mut fast, mut band) = (0usize, 0usize);
            for d in &cand {
                let (p, v) = (&d.pos, &d.vel);
                if born >= budget {
                    break;
                }
                let speed = v.norm();
                if speed < 0.6 {
                    continue;
                }
                fast += 1;
                let s = g.at(p.x, p.y) + off;
                // from a cell and a half under the surface up through the crown
                if p.z < s - 1.5 * w.h || p.z > s + 0.6 {
                    continue;
                }
                band += 1;
                let rate = 3.0 * (speed - 0.6).min(4.0) * frame_dt;
                if rnd() < rate {
                    self.foam.push(Foam { x: p.x, y: p.y, vx: v.x, vy: v.y, age: 0.0, life: 1.5 + 1.5 * rnd(), size: 0.02 + 0.03 * rnd() });
                    born += 1;
                }
            }
            // ...and where the surface itself breaks: rising faster than
            // 0.4 m/s or steeper than 1 in 2, per cell of the height field
            if let Some(prev) = &self.surface_prev {
                if prev.nx == g.nx && prev.ny == g.ny {
                    let mut surf_born = 0;
                    for jy in 1..g.ny - 1 {
                        for ix in 1..g.nx - 1 {
                            if self.foam.len() >= 60_000 {
                                break;
                            }
                            let i = jy * g.nx + ix;
                            let rise = (g.z[i] - prev.z[i]) / frame_dt;
                            let sx = (g.z[i + 1] - g.z[i - 1]) / (2.0 * g.cell);
                            let sy = (g.z[i + g.nx] - g.z[i - g.nx]) / (2.0 * g.cell);
                            let slope = sx.hypot(sy);
                            // steepness is the better sign of breaking; a rising crown is mostly smooth
                            let potential = (rise.abs() - 0.8).max(0.0) / 4.0 + (slope - 0.6).max(0.0) * 0.8;
                            if potential <= 0.0 {
                                continue;
                            }
                            if rnd() < potential.min(1.0) * 0.35 {
                                let x = g.origin[0] + (ix as f64 + 0.5) * g.cell + (rnd() - 0.5) * g.cell;
                                let y = g.origin[1] + (jy as f64 + 0.5) * g.cell + (rnd() - 0.5) * g.cell;
                                // drift outward from the splash with the ring
                                let r = x.hypot(y).max(0.05);
                                let sp = 0.3 * potential.min(1.0);
                                self.foam.push(Foam { x, y, vx: sp * x / r, vy: sp * y / r, age: 0.0, life: 1.5 + 2.0 * rnd(), size: 0.012 + 0.02 * rnd() });
                                surf_born += 1;
                            }
                        }
                    }
                    born += surf_born;
                }
            }
            self.surface_prev = Some(g.clone());
            for f in self.foam.iter_mut() {
                f.age += frame_dt;
                let drag = (-2.0 * frame_dt).exp();
                f.vx *= drag;
                f.vy *= drag;
                f.x += f.vx * frame_dt;
                f.y += f.vy * frame_dt;
            }
            self.foam.retain(|f| f.age < f.life);
            if std::env::var_os("NEWT_PROF").is_some() {
                println!("foam   fast {fast}  in band {band}  born {born}  alive {}", self.foam.len());
            }
            // bake the rings into the grid for this frame
            let mut g = g;
            if !self.surface.rings.is_empty() {
                for jy in 0..g.ny {
                    for ix in 0..g.nx {
                        let x = g.origin[0] + (ix as f64 + 0.5) * g.cell;
                        let y = g.origin[1] + (jy as f64 + 0.5) * g.cell;
                        let mut h = 0.0;
                        for r in &self.surface.rings {
                            h += r.height(x, y, t);
                        }
                        g.z[jy * g.nx + ix] += h;
                    }
                }
            }
            // the box's mean level drifts by millimetres (the sponge, the
            // extraction); joined to a far field at zero that is a dish
            // four metres wide, and a dish that wide is a lens
            {
                let (mut sum, mut n) = (0.0, 0.0f64);
                for jy in 3..g.ny.saturating_sub(3) {
                    for ix in 3..g.nx.saturating_sub(3) {
                        sum += g.z[jy * g.nx + ix];
                        n += 1.0;
                    }
                }
                let mean = sum / n.max(1.0);
                for z in g.z.iter_mut() {
                    *z -= mean;
                }
            }
            if let Some(far) = self.far.as_mut() {
                // spectral, so one exact step per frame
                far.step((t - far.time).max(0.0));
                far.force(&g, box_half() - SPONGE - BLEND, box_half() - SPONGE, t);
                self.surface.far = Some(far.grid.clone());
            }
            self.surface.grid = Some(g);
        }
    }

    pub fn centre(&self) -> V<f64> {
        V::new(self.state.q[3], self.state.q[4], self.state.q[5])
    }

    pub fn melon(&self) -> Melon {
        // it lands long-axis-first-ish and settles horizontal; a slow yaw for life
        let yaw = 0.6 + 0.15 * self.state.time;
        Melon { centre: self.centre(), axis: V::new(yaw.cos(), yaw.sin(), 0.0) }
    }

    /// Submerged volume of the equivalent sphere below the local water height.
    fn submerged(&self) -> (f64, f64) {
        let c = self.centre();
        let r = (MELON_AXES[0] * MELON_AXES[1] * MELON_AXES[2]).cbrt();
        let eta = self.surface.height(c.x, c.y);
        let h = (eta - (c.z - r)).clamp(0.0, 2.0 * r); // submerged cap height
        let v = std::f64::consts::PI * h * h * (3.0 * r - h) / 3.0;
        (v, h / (2.0 * r))
    }

    pub fn step(&mut self) {
        if self.water.is_some() {
            return self.step_with_water();
        }
        self.surface.t = self.state.time;
        let (v_sub, frac) = self.submerged();
        let vel = V::new(self.state.v[3], self.state.v[4], self.state.v[5]);
        let r = (MELON_AXES[0] * MELON_AXES[1] * MELON_AXES[2]).cbrt();
        // buoyancy up, drag against motion through the submerged share of the
        // frontal area, and a little added-mass damping of the bob
        let buoy = WATER_DENSITY * GRAVITY * v_sub;
        let area = std::f64::consts::PI * r * r * frac;
        let speed = vel.norm();
        let drag = if speed > 1e-6 { vel * (-0.5 * WATER_DENSITY * 0.47 * area * speed) } else { V::zero() };
        // added-mass damping of the bob: a real melon stops bobbing in a few cycles
        let damp = vel * (-14.0 * frac);
        let f = V::new(drag.x + damp.x, drag.y + damp.y, buoy + drag.z + damp.z);
        // the free joint's generalized force is [torque; force] in the body
        // frame, applied through `ctrl` (what ipse-sim uses for a shove); the
        // body is kept unrotated so the body frame is the world frame
        self.state.ctrl = phyz_math::DVec::from_slice(&[0.0, 0.0, 0.0, f.x, f.y, f.z]);
        self.state.v[0] = 0.0;
        self.state.v[1] = 0.0;
        self.state.v[2] = 0.0;
        // the floor of the pool is the ground plane
        self.sim.step_with_contacts(&self.model, &mut self.state, -DEPTH, &Default::default());
        self.state.q[0] = 0.0;
        self.state.q[1] = 0.0;
        self.state.q[2] = 0.0;
        // splashes: the entry, then rings from the bob
        let c = self.centre();
        let t = self.state.time;
        if !self.entered && frac > 0.0 {
            self.entered = true;
            self.last_splash = t;
            let v_in = vel.z.abs();
            self.surface.rings.push(Ring { x: c.x, y: c.y, t0: t, amp: (0.012 * v_in).min(0.05), wavelength: 0.22 });
            self.surface.rings.push(Ring { x: c.x, y: c.y, t0: t + 0.08, amp: (0.006 * v_in).min(0.03), wavelength: 0.11 });
        } else if self.entered && t - self.last_splash > 0.45 && vel.z.abs() > 0.15 && self.surface.rings.len() < 8 {
            self.last_splash = t;
            self.surface.rings.push(Ring { x: c.x, y: c.y, t0: t, amp: 0.004 * vel.z.abs().min(2.0), wavelength: 0.15 });
        }
    }
}

// ---- rendering --------------------------------------------------------------

#[derive(Clone)]
pub struct Caustic {
    pub origin: [f64; 2],
    pub cell: f64,
    pub nx: usize,
    pub ny: usize,
    pub e: Vec<f64>,
}

impl Caustic {
    fn at(&self, x: f64, y: f64) -> f64 {
        let gx = (x - self.origin[0]) / self.cell - 0.5;
        let gy = (y - self.origin[1]) / self.cell - 0.5;
        if gx < 0.0 || gy < 0.0 || gx >= (self.nx - 1) as f64 || gy >= (self.ny - 1) as f64 {
            return 1.0;
        }
        let (ix, iy) = (gx.floor() as usize, gy.floor() as usize);
        let (wx, wy) = (gx - ix as f64, gy - iy as f64);
        let e = &self.e;
        e[iy * self.nx + ix] * (1.0 - wx) * (1.0 - wy)
            + e[iy * self.nx + ix + 1] * wx * (1.0 - wy)
            + e[(iy + 1) * self.nx + ix] * (1.0 - wx) * wy
            + e[(iy + 1) * self.nx + ix + 1] * wx * wy
    }
}

/// Sunlight through the surface onto the floor: irradiance relative to what a
/// flat surface would pass, so still water reads as 1 and ripples focus it.
pub fn caustic(surface: &Surface, cell: f64) -> Caustic {
    // over the box and a margin: the sun's refracted rays land a metre
    // sideways over two metres of depth, and beyond the map the floor reads 1
    let half = box_half() + 3.0;
    let nx = ((2.0 * half) / cell) as usize;
    let ny = ((2.0 * half) / cell) as usize;
    let origin = [-half, -half];
    // launch from beyond the map too: a cell near the map's edge is lit by
    // rays from both sides, or the edge shows as a frame
    let margin = 1.5;
    let (lx, ly) = (((2.0 * half + 2.0 * margin) / cell) as usize, ((2.0 * half + 2.0 * margin) / cell) as usize);
    let l_origin = [-half - margin, -half - margin];
    let mut e = vec![0.0; nx * ny];
    let s = sun_dir();
    let d = -s;
    // reference: a flat surface refracts the sun to a fixed direction with a
    // fixed transmission; each ray deposits relative to that
    let n_flat = V::new(0.0, 0.0, 1.0);
    let (d_flat, ci, ct) = refract(d, n_flat, 1.0, N_WATER).expect("sun above the horizon");
    let t_flat = 1.0 - fresnel(1.0, N_WATER, ci, ct);
    let flat_cos = -d_flat.z;
    let sub = 3; // rays per cell per axis
    let per_ray = 1.0 / (sub * sub) as f64;
    // one row of launch points per task, each with its own accumulator
    let e = (0..ly * sub)
        .into_par_iter()
        .fold(
            || vec![0.0; nx * ny],
            |mut e, iy| {
                for ix in 0..lx * sub {
                    // launch from the surface point that the flat refraction would
                    // send to this floor cell, so the reference is uniform
                    let fx = l_origin[0] + (ix as f64 + 0.5) * cell / sub as f64;
                    let fy = l_origin[1] + (iy as f64 + 0.5) * cell / sub as f64;
                    let back = DEPTH / flat_cos;
                    let sx = fx - d_flat.x * back;
                    let sy = fy - d_flat.y * back;
                    let n = surface.normal(sx, sy);
                    let Some((dr, ci, ct)) = refract(d, n, 1.0, N_WATER) else { continue };
                    let tr = 1.0 - fresnel(1.0, N_WATER, ci, ct);
                    let z0 = surface.height(sx, sy);
                    let tt = (z0 + DEPTH) / -dr.z;
                    let hx = sx + dr.x * tt;
                    let hy = sy + dr.y * tt;
                    let w = per_ray * (tr / t_flat) * (-dr.z / flat_cos);
                    let gx = (hx - origin[0]) / cell - 0.5;
                    let gy = (hy - origin[1]) / cell - 0.5;
                    let (bx, by) = (gx.floor(), gy.floor());
                    let (wx, wy) = (gx - bx, gy - by);
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (jx, jy) = (bx as i64 + dx, by as i64 + dy);
                        if jx < 0 || jy < 0 || jx >= nx as i64 || jy >= ny as i64 {
                            continue;
                        }
                        let ww = (if dx == 0 { 1.0 - wx } else { wx }) * (if dy == 0 { 1.0 - wy } else { wy });
                        e[jy as usize * nx + jx as usize] += w * ww;
                    }
                }
                e
            },
        )
        .reduce(
            || vec![0.0; nx * ny],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                a
            },
        );
    Caustic { origin, cell, nx, ny, e }
}

/// The same caustic, traced on the GPU: three million rays, the composed
/// surface evaluated in WGSL, deposited through fixed-point atomics. The CPU
/// version above stays the reference; `NEWT_GPU_RENDER=0` selects it.
pub fn caustic_gpu(gpu: &mut newt_mpm::GpuCaustic, surface: &Surface, cell: f64) -> Option<Caustic> {
    let g = surface.grid.as_ref()?;
    let half = box_half() + 3.0;
    let fz: Vec<f32> = g.z.iter().map(|z| *z as f32).collect();
    // no far field yet (the first frames, or the ring model): a flat pool
    let zero = crate::splash::HeightGrid { origin: [-POOL_X, -POOL_Y], cell: POOL_X, nx: 2, ny: 2, z: vec![0.0; 4] };
    let f = surface.far.as_ref().unwrap_or(&zero);
    let rz: Vec<f32> = f.z.iter().map(|z| *z as f32).collect();
    let d = -sun_dir();
    let cfg = newt_mpm::CausticCfg {
        cell: cell as f32,
        half: half as f32,
        margin: 1.5,
        sub: 3,
        depth: DEPTH as f32,
        n_water: N_WATER as f32,
        dir: [d.x as f32, d.y as f32, d.z as f32],
        t: surface.t as f32,
        box_half: box_half() as f32,
        sponge: SPONGE as f32,
        blend: BLEND as f32,
    };
    let fine = newt_mpm::Grid { origin: [g.origin[0] as f32, g.origin[1] as f32], cell: g.cell as f32, nx: g.nx as u32, ny: g.ny as u32, z: &fz };
    let far = newt_mpm::Grid { origin: [f.origin[0] as f32, f.origin[1] as f32], cell: f.cell as f32, nx: f.nx as u32, ny: f.ny as u32, z: &rz };
    let (nx, ny, e) = gpu.trace(&cfg, &fine, &far);
    Some(Caustic { origin: [-half, -half], cell, nx, ny, e: e.into_iter().map(|v| v as f64).collect() })
}

pub struct View {
    pub eye: V<f64>,
    pub target: V<f64>,
    pub width: u32,
    pub height: u32,
    pub vfov: f64,
}

fn sky(d: V<f64>) -> [f64; 3] {
    let t = d.z.clamp(0.0, 1.0);
    let horizon = [0.66, 0.80, 0.94];
    let zenith = [0.22, 0.44, 0.88];
    let mut c = [0.0; 3];
    for k in 0..3 {
        c[k] = horizon[k] * (1.0 - t) + zenith[k] * t;
    }
    if d.dot(&sun_dir()) > SUN_DISC_COS {
        return [12.0, 11.0, 9.0];
    }
    // a glow around the sun
    let g = ((d.dot(&sun_dir()) - 0.97) / 0.03).clamp(0.0, 1.0);
    for k in 0..3 {
        c[k] += 0.6 * g * [1.0, 0.9, 0.7][k];
    }
    c
}

fn tile(x: f64, y: f64) -> [f64; 3] {
    // lane lines: ten lanes of 2.5 m across the width, a 25 cm dark line
    // along the length of each, with the T two metres from the end walls
    let lane = ((y + POOL_Y) / 2.5).floor() * 2.5 - POOL_Y + 1.25;
    let on_line = (y - lane).abs() < 0.125;
    let on_t = (x.abs() - (POOL_X - 2.0)).abs() < 0.125 && (y - lane).abs() < 0.5;
    if on_line || on_t {
        return [0.05, 0.09, 0.18];
    }
    let f = |v: f64| ((v / 0.1).rem_euclid(1.0) - 0.5).abs();
    let grout = f(x) > 0.46 || f(y) > 0.46;
    if grout {
        [0.42, 0.52, 0.58]
    } else {
        // two blues, checkered softly
        let a = (((x / 0.1).floor() + (y / 0.1).floor()) as i64).rem_euclid(2) == 0;
        if a { [0.58, 0.78, 0.86] } else { [0.50, 0.72, 0.84] }
    }
}

fn deck() -> [f64; 3] {
    [0.80, 0.68, 0.52]
}

/// Shadow of the melon on the floor, in air-and-water terms: the sun's ray
/// through the floor point, refracted back up, blocked by the melon?
fn sun_blocked_underwater(melon: &Melon, surface: &Surface, p: V<f64>) -> bool {
    // trace the refracted sun ray backwards from p to the surface
    let d = -sun_dir();
    let n_flat = V::new(0.0, 0.0, 1.0);
    let (dr, _, _) = refract(d, n_flat, 1.0, N_WATER).expect("sun above");
    let up = -dr;
    let t_surf = (surface.height(p.x, p.y) - p.z) / up.z;
    if let Some((t, _)) = melon.hit(p + up * 1e-4, up) {
        if t < t_surf {
            return true;
        }
    }
    // and above the surface, along the sun
    let s = p + up * t_surf;
    if let Some((_t, _)) = melon.hit(s + sun_dir() * 1e-4, sun_dir()) {
        return true;
    }
    false
}

fn sun_blocked_air(melon: &Melon, p: V<f64>) -> bool {
    melon.hit(p + sun_dir() * 1e-4, sun_dir()).is_some()
}

/// Radiance along a camera ray.
pub fn radiance(view_o: V<f64>, d: V<f64>, drop: &Scene, melon: &Melon, caustic: &Caustic, depth: u32) -> [f64; 3] {
    let o = view_o;
    let s = sun_dir();
    // candidates: melon, deck/coping, water surface, pool walls above water
    let t_melon = melon.hit(o, d).map(|(t, n)| (t, n));
    // the deck: a plane at z = COPING outside the pool
    let t_deck = if d.z < 0.0 { Some((COPING - o.z) / d.z) } else { None };
    let inside_pool = |p: V<f64>| p.x.abs() < POOL_X && p.y.abs() < POOL_Y;
    // where the ray enters the pool column (z < COPING and inside), march for the surface
    let mut t_water = None;
    // the march starts where the ray drops below the highest water, which
    // is the deck line for a still pool and the crown's tip in a splash
    let top = drop.top.max(COPING);
    if d.z < 0.0 || o.z < top {
        // start marching at that plane if above it, else from the origin
        let t_start = if o.z > top { ((top - o.z) / d.z).max(0.0) } else { 0.0 };
        let p0 = o + d * t_start;
        // only if the ray is over the pool at the water line
        let t_line = if d.z != 0.0 { (0.06 - o.z) / d.z } else { 0.0 };
        let p_line = o + d * t_line.max(0.0);
        if inside_pool(p0) || inside_pool(p_line) {
            t_water = drop.surface.hit(p0, d, 80.0).map(|t| t + t_start);
        }
    }
    // pool walls seen from above the water (the tiled inside faces above z=0)
    let t_wall = {
        let mut best: Option<(f64, V<f64>)> = None;
        for (n, dd) in [(V::new(1.0, 0.0, 0.0), POOL_X), (V::new(-1.0, 0.0, 0.0), POOL_X), (V::new(0.0, 1.0, 0.0), POOL_Y), (V::new(0.0, -1.0, 0.0), POOL_Y)] {
            let denom = n.dot(&d);
            if denom.abs() < 1e-9 {
                continue;
            }
            let t = (dd - n.dot(&o)) / denom;
            let p = o + d * t;
            if t > 1e-6 && p.z <= COPING && p.z >= -0.001 && inside_pool(p + n * -1e-4) && best.is_none_or(|b| t < b.0) {
                best = Some((t, -n));
            }
        }
        best
    };
    // drops in the air: beads stretched along their velocity, bigger where
    // more water travels together, with a soft edge
    let mut t_drop: Option<(f64, V<f64>, f64)> = None;
    // secondary rays (depth 0) skip the beads: cheap, and it is how a bead
    // sees what is behind it
    for dr in drop.droplets.iter().filter(|_| depth > 0) {
        let r = 0.006 + 0.010 * dr.crowd;
        let speed = dr.vel.norm();
        let axis = if speed > 0.2 { dr.vel / speed } else { V::new(0.0, 0.0, 1.0) };
        let stretch = 1.0 + (speed * 0.25).min(2.0);
        // ellipsoid: scale space along `axis` by 1/stretch
        let oc = o - dr.pos;
        let along = oc.dot(&axis);
        let ol = oc - axis * along + axis * (along / stretch);
        let dal = d.dot(&axis);
        let dl = d - axis * dal + axis * (dal / stretch);
        let aa = dl.norm_sq();
        let bb = ol.dot(&dl);
        let cc = ol.norm_sq() - r * r;
        let disc = bb * bb - aa * cc;
        if disc > 0.0 {
            let t = (-bb - disc.sqrt()) / aa;
            if t > 1e-6 && t_drop.is_none_or(|x| t < x.0) {
                let pl = ol + dl * t;
                let nl = pl - axis * pl.dot(&axis) + axis * (pl.dot(&axis) / stretch);
                // how central the hit is, for the soft rim
                let edge = 1.0 - (disc.sqrt() / (r * aa.sqrt())).clamp(0.0, 1.0);
                t_drop = Some((t, nl.normalize(), edge));
            }
        }
    }
    // the grandstand, only for rays heading its way
    let t_stand = if d.y > 0.0 && o.y < STAND_Y0 + STAND_ROWS as f64 * STAND_TREAD { stand_and_crowd(o, d) } else { None };
    // pick the nearest of melon / water / wall / deck / drop / stand
    let mut best_t = f64::INFINITY;
    let mut what = 0; // 0 sky
    if let Some((t, _)) = t_stand {
        if t < best_t { best_t = t; what = 6; }
    }
    if let Some((t, _, _)) = t_drop {
        if t < best_t { best_t = t; what = 5; }
    }
    if let Some((t, _)) = t_melon {
        if t < best_t { best_t = t; what = 1; }
    }
    if let Some(t) = t_water {
        if t < best_t { best_t = t; what = 2; }
    }
    if let Some((t, _)) = t_wall {
        if t < best_t { best_t = t; what = 3; }
    }
    if let Some(t) = t_deck {
        let p = o + d * t;
        if t > 1e-6 && t < best_t && !inside_pool(p) { best_t = t; what = 4; }
    }
    match what {
        1 => {
            let (t, n) = t_melon.unwrap();
            let p = o + d * t;
            shade_melon(p, n, d, melon, drop, caustic, true)
        }
        2 => {
            let t = t_water.unwrap();
            let p = o + d * t;
            let n = drop.surface.normal(p.x, p.y);
            let n = if n.dot(&d) > 0.0 { -n } else { n };
            let Some((dr, ci, ct)) = refract(d, n, 1.0, N_WATER) else {
                return sky(reflect(d, n));
            };
            let r = fresnel(1.0, N_WATER, ci, ct);
            let rd = reflect(d, n);
            let reflected = if depth > 0 {
                // the reflection may show the melon or the far coping; mostly sky
                radiance_above(p + n * 1e-4, rd, drop, melon, caustic)
            } else {
                sky(rd)
            };
            let under = underwater(p - n * 1e-4, dr, drop, melon, caustic);
            let mut c = [0.0; 3];
            for k in 0..3 {
                c[k] = r * reflected[k] + (1.0 - r) * under[k];
            }
            // foam: a white scattering layer, lit by sun and sky
            let a = drop.foam.coverage(p.x, p.y);
            if a > 0.0 {
                let lit = SUN_IRRADIANCE * n.dot(&s).max(0.0) * if sun_blocked_air(melon, p) { 0.3 } else { 1.0 };
                let sk = sky(n);
                for k in 0..3 {
                    let foam = [0.93, 0.96, 1.0][k] * (0.45 * sk[k] + 0.7 * lit);
                    c[k] = c[k] * (1.0 - a) + foam * a;
                }
            }
            c
        }
        3 => {
            let (t, n) = t_wall.unwrap();
            let p = o + d * t;
            let base = tile(p.x + p.z, p.y + p.z);
            let lit = SUN_IRRADIANCE * n.dot(&s).max(0.0) * if sun_blocked_air(melon, p) { 0.0 } else { 1.0 };
            let amb = 0.35;
            let mut c = [0.0; 3];
            for k in 0..3 { c[k] = base[k] * (amb * sky(n)[k] + lit); }
            c
        }
        4 => {
            let p = o + d * best_t;
            let n = V::new(0.0, 0.0, 1.0);
            let lit = SUN_IRRADIANCE * n.dot(&s).max(0.0) * if sun_blocked_air(melon, p) { 0.15 } else { 1.0 };
            let mut c = [0.0; 3];
            for k in 0..3 { c[k] = deck()[k] * (0.35 * sky(n)[k] + lit); }
            c
        }
        5 => {
            // a drop: a bead of water. reflection of the sky, a glint of sun,
            // and a soft rim that lets what is behind it through
            let (t, n, edge) = t_drop.unwrap();
            let p = o + d * t;
            let r = 0.04 + 0.96 * (1.0 + d.dot(&n)).clamp(0.0, 1.0).powi(5);
            let sk = sky(reflect(d, n));
            let mut bead = [0.0; 3];
            for k in 0..3 {
                bead[k] = r * sk[k] + (1.0 - r) * [0.60, 0.76, 0.88][k] * 0.85;
            }
            let hl = n.dot(&(s - d).normalize()).max(0.0).powf(60.0) * 2.5;
            for v in &mut bead { *v += hl; }
            // behind the bead: the scene, without this bead
            let behind = radiance(p + d * 1e-4, d, drop, melon, caustic, 0);
            let a = (edge * 1.6).clamp(0.0, 1.0);
            let mut c = [0.0; 3];
            for k in 0..3 { c[k] = a * bead[k] + (1.0 - a) * behind[k]; }
            c
        }
        6 => t_stand.unwrap().1,
        _ => sky(d),
    }
}

/// A reflected ray from the water surface: melon, the stand, or sky.
fn radiance_above(o: V<f64>, d: V<f64>, drop: &Scene, melon: &Melon, caustic: &Caustic) -> [f64; 3] {
    if let Some((t, n)) = melon.hit(o, d) {
        let p = o + d * t;
        return shade_melon(p, n, d, melon, drop, caustic, true);
    }
    if d.y > 0.0 {
        if let Some((_, c)) = stand_and_crowd(o, d) {
            return c;
        }
    }
    sky(d)
}

/// A ray inside the water: absorb along the way, hit melon, walls, or the floor.
fn underwater(o: V<f64>, d: V<f64>, drop: &Scene, melon: &Melon, caustic: &Caustic) -> [f64; 3] {
    let inside_pool = |p: V<f64>| p.x.abs() < POOL_X + 1e-3 && p.y.abs() < POOL_Y + 1e-3;
    let mut best_t = f64::INFINITY;
    let mut what = 0;
    let mut n_hit = V::new(0.0, 0.0, 1.0);
    if let Some((t, n)) = melon.hit(o, d) {
        best_t = t; what = 1; n_hit = n;
    }
    if d.z < 0.0 {
        let t = (-DEPTH - o.z) / d.z;
        if t < best_t { best_t = t; what = 2; n_hit = V::new(0.0, 0.0, 1.0); }
    }
    for (n, dd) in [(V::new(1.0, 0.0, 0.0), POOL_X), (V::new(-1.0, 0.0, 0.0), POOL_X), (V::new(0.0, 1.0, 0.0), POOL_Y), (V::new(0.0, -1.0, 0.0), POOL_Y)] {
        let denom = n.dot(&d);
        if denom <= 1e-9 { continue; }
        let t = (dd - n.dot(&o)) / denom;
        if t > 1e-6 && t < best_t && inside_pool(o + d * t) { best_t = t; what = 3; n_hit = -n; }
    }
    if what == 0 {
        // up and out through the surface: the sky, dimmed
        return [0.3, 0.45, 0.6];
    }
    let p = o + d * best_t;
    let mut c = match what {
        1 => shade_melon(p, n_hit, d, melon, drop, caustic, false),
        2 => {
            let base = tile(p.x, p.y);
            let shadow = if sun_blocked_underwater(melon, &drop.surface, p) { 0.12 } else { 1.0 };
            let lit = SUN_IRRADIANCE * 0.92 * caustic.at(p.x, p.y) * shadow;
            let mut c = [0.0; 3];
            for k in 0..3 { c[k] = base[k] * (0.30 + lit); }
            c
        }
        _ => {
            let base = tile(p.x + p.z, p.y + p.z);
            let lit = SUN_IRRADIANCE * 0.5 * n_hit.dot(&sun_dir()).max(0.0);
            let mut c = [0.0; 3];
            for k in 0..3 { c[k] = base[k] * (0.30 + lit); }
            c
        }
    };
    // absorption along the path, and some in-scatter of blue
    for k in 0..3 {
        let a = (-ABSORB[k] * best_t).exp();
        c[k] = c[k] * a + [0.02, 0.10, 0.16][k] * (1.0 - a);
    }
    c
}

fn shade_melon(p: V<f64>, n: V<f64>, d: V<f64>, melon: &Melon, drop: &Scene, caustic: &Caustic, in_air: bool) -> [f64; 3] {
    let s = sun_dir();
    let base = melon.albedo(p);
    let cos = n.dot(&s).max(0.0);
    let submerged = p.z < drop.surface.height(p.x, p.y);
    let direct = if in_air && !submerged {
        SUN_IRRADIANCE * cos
    } else {
        // under water the sun arrives refracted and rippled
        SUN_IRRADIANCE * 0.85 * cos * caustic.at(p.x, p.y).min(2.5)
    };
    let amb = 0.35;
    let mut c = [0.0; 3];
    let sk = sky(n);
    for k in 0..3 {
        c[k] = base[k] * (amb * sk[k] + direct);
    }
    // a glossy rind: the sun's reflection, sharp
    if in_air && !submerged {
        let h = (s - d).normalize();
        let spec = n.dot(&h).max(0.0).powf(120.0) * 1.6;
        let fres = 0.04 + 0.96 * (1.0 + d.dot(&n)).clamp(0.0, 1.0).powi(5);
        let refl = sky(reflect(d, n));
        for k in 0..3 {
            c[k] += spec + fres * 0.5 * refl[k];
        }
    }
    let _ = melon;
    c
}

/// What a ray can see: the surface and the beads, borrowed from the drop so
/// rendering can run across threads without the simulator.
#[derive(Clone, Copy)]
pub struct Scene<'a> {
    pub surface: &'a Surface,
    pub droplets: &'a [crate::splash::Droplet],
    /// The surface's maximum this frame (computed once: it is a grid scan).
    pub top: f64,
    pub foam: &'a FoamField,
}

pub fn render(view: &View, drop: &Drop, caustic: &Caustic) -> image::RgbaImage {
    let melon = drop.melon();
    let foam = FoamField::build(&drop.foam, box_half() + 1.0);
    let scene = Scene { surface: &drop.surface, droplets: &drop.droplets, top: drop.surface.top(), foam: &foam };
    let drop = &scene;
    let fwd = (view.target - view.eye).normalize();
    let right = fwd.cross(&V::new(0.0, 0.0, 1.0)).normalize();
    let up = right.cross(&fwd);
    let fy = 0.5 * view.height as f64 / (0.5 * view.vfov).tan();
    let to8 = |v: f64| {
        // a filmic-ish curve so the sun's highlight rolls off instead of clipping
        let v = v / (1.0 + v * 0.35) * 1.2;
        (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8
    };
    let w = view.width as usize;
    let rows: Vec<Vec<u8>> = (0..view.height)
        .into_par_iter()
        .map(|y| {
            let mut row = Vec::with_capacity(4 * w);
            for x in 0..view.width {
                let px = (x as f64 + 0.5 - view.width as f64 / 2.0) / fy;
                let py = (view.height as f64 / 2.0 - y as f64 - 0.5) / fy;
                let d = (fwd + right * px + up * py).normalize();
                let c = radiance(view.eye, d, drop, &melon, caustic, 1);
                row.extend_from_slice(&[to8(c[0]), to8(c[1]), to8(c[2]), 255]);
            }
            row
        })
        .collect();
    let img = image::RgbaImage::from_raw(view.width, view.height, rows.concat()).expect("image");
    img
}

fn drop_h_mm() -> u32 {
    (std::env::var("NEWT_H").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.025) * 1000.0).round() as u32
}

/// The whole thing: drop the melon, render `frames` at 30 fps, encode.
pub fn run(out: &Path, frames: usize, width: u32, height: u32, splash: bool) -> anyhow::Result<()> {
    let tag = if splash { format!("splash_{}mm", (drop_h_mm())) } else { "pool".to_string() };
    let dir = out.join(&tag);
    // stale frames from another run would be swept into the encode
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let mut drop = Drop::new(1.3);
    if splash {
        drop = drop.with_water(std::env::var("NEWT_H").ok().and_then(|v| v.parse().ok()).unwrap_or(0.025));
        let w = drop.water.as_ref().unwrap();
        println!("splash {} water particles on a {}×{}×{} grid at {} cm, dt {:.2e} s ({} substeps per ms)", w.count(), w.nx, w.ny, w.nz, w.h * 100.0, w.dt, (drop.model.dt / w.dt).ceil());
    }
    // from the deck corner, low, so the far water reflects the sky and the near
    // water shows the tiles
    let view = View { eye: V::new(-3.2, -2.6, 0.9), target: V::new(0.0, 0.1, -0.1), width, height, vfov: 0.9 };
    let steps_per_frame = (1.0 / fps() / drop.model.dt).round() as usize;
    // NEWT_GPU_RENDER=0 keeps the caustic on the CPU, for comparison
    let mut cgpu = if std::env::var("NEWT_GPU_RENDER").map(|v| v != "0").unwrap_or(true) {
        match newt_mpm::GpuCaustic::new() {
            Ok(g) => {
                println!("pool   caustic on the GPU");
                Some(g)
            }
            Err(e) => {
                println!("pool   caustic on the CPU ({e})");
                None
            }
        }
    } else {
        None
    };
    let t0 = std::time::Instant::now();
    let mut lowest = f64::INFINITY;
    for k in 0..frames {
        let mut lap = std::time::Instant::now();
        let mut ms = [0u128; 5];
        let mut tick = |slot: usize, lap: &mut std::time::Instant| {
            ms[slot] = lap.elapsed().as_millis();
            *lap = std::time::Instant::now();
        };
        for _ in 0..steps_per_frame {
            drop.step();
        }
        tick(0, &mut lap);
        lowest = lowest.min(drop.centre().z);
        drop.read_water();
        tick(1, &mut lap);
        let c = cgpu.as_mut().and_then(|g| caustic_gpu(g, &drop.surface, 0.02)).unwrap_or_else(|| caustic(&drop.surface, 0.02));
        tick(2, &mut lap);
        let peak_force = drop.fluid_force;
        drop.fluid_force = V::zero();
        let img = render(&view, &drop, &c);
        tick(3, &mut lap);
        img.save(dir.join(format!("frame_{k:03}.png")))?;
        tick(4, &mut lap);
        if std::env::var_os("NEWT_PROF").is_some() {
            let w = crate::splash::take_prof().map(|ns| ns / 1_000_000);
            println!("prof   frame {k:3}  step {:5} ms (zero {} bin {} p2g {} grid {} g2p {})  read_water {:5} ms  caustic {:5} ms  render {:5} ms  save {:4} ms", ms[0], w[0], w[1], w[2], w[3], w[4], ms[1], ms[2], ms[3], ms[4]);
        }
        if k == 0 || k % 5 == 0 || k + 1 == frames || std::env::var_os("NEWT_PROF").is_some() {
            let m = drop.centre();
            // these read every particle, so they cost a full download: they
            // are diagnostics, not the frame, and only run when asked for
            if std::env::var_os("NEWT_WATER_STATS").is_some() {
                if let Some(w) = drop.water.as_mut() {
                    w.sync_from_gpu();
                }
            }
            if let (Some(w), true) = (&drop.water, std::env::var_os("NEWT_WATER_STATS").is_some()) {
                let mut zs: Vec<f64> = w.x.iter().map(|p| p.z).collect();
                zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mean = zs.iter().sum::<f64>() / zs.len() as f64;
                let mel = drop.melon();
                let body = crate::splash::Body { centre: Vec3::new(mel.centre.x, mel.centre.y, mel.centre.z), axis: Vec3::new(mel.axis.x, mel.axis.y, mel.axis.z), vel: Vec3::zeros() };
                let inside = w.x.iter().filter(|p| body.sdf(**p).0 < 0.0).count();
                println!("water  frame {k:3}  particles inside the melon {inside} ({:.2} kg)", inside as f64 * w.mass);
                if let Some(far) = &drop.surface.far {
                    let mut amp = 0.0f64;
                    for j in 0..far.ny {
                        for i in 0..far.nx {
                            let x = far.origin[0] + (i as f64 + 0.5) * far.cell;
                            let y = far.origin[1] + (j as f64 + 0.5) * far.cell;
                            let r = x.hypot(y);
                            if r > 1.5 && r < 3.0 {
                                amp = amp.max(far.z[j * far.nx + i].abs());
                            }
                        }
                    }
                    // and what the renderer sees there: slope and caustic
                    let (mut slope, mut cmin, mut cmax, mut cin_min, mut cin_max) = (0.0f64, 9.0f64, 0.0f64, 9.0f64, 0.0f64);
                    for j in 0..c.ny {
                        for i in 0..c.nx {
                            let x = c.origin[0] + (i as f64 + 0.5) * c.cell;
                            let y = c.origin[1] + (j as f64 + 0.5) * c.cell;
                            let r = x.hypot(y);
                            let e = c.e[j * c.nx + i];
                            if r > 1.5 && r < 3.0 {
                                cmin = cmin.min(e);
                                cmax = cmax.max(e);
                                let n = drop.surface.normal(x, y);
                                slope = slope.max((n.x.hypot(n.y)) / n.z);
                            } else if r < 0.5 {
                                cin_min = cin_min.min(e);
                                cin_max = cin_max.max(e);
                            }
                        }
                    }
                    println!("far    frame {k:3}  max |h| at 1.5-3 m: {:.1} mm  slope max {:.4}  caustic there {:.2}..{:.2}  in the box {:.2}..{:.2}", amp * 1000.0, slope, cmin, cmax, cin_min, cin_max);
                    if let Some(f) = &drop.far {
                        println!("far    frame {k:3}  wave energy {:.3} J  injected by the nudge so far {:.3} J", 1000.0 * f.energy(), 1000.0 * f.injected);
                    }
                }
                println!("water  frame {k:3}  particle z mean {:+.1} mm (rest {:.0})  z50 {:+.1}  z90 {:+.1}  z99 {:+.1} mm", mean * 1000.0, -BOX_DEPTH * 500.0, zs[zs.len() / 2] * 1000.0, zs[zs.len() * 9 / 10] * 1000.0, zs[zs.len() * 99 / 100] * 1000.0);
            }
            println!(
                "pool   frame {k:3}  t={:.2} s  melon z={:+.3} m  vz={:+.2} m/s  level {:+.1} mm  fluid force z={:+.0} N  water inside {:.1} kg (weight {:.0} N)  drops={} foam={}  {} ms/frame",
                drop.state.time, m.z, drop.state.v[5], drop.surface.mean_level() * 1000.0, peak_force.z, drop.water.as_ref().map(|w| w.interior_mass).unwrap_or(0.0), drop.water.as_ref().map(|w| w.interior_mass * GRAVITY).unwrap_or(0.0), drop.droplets.len(), drop.foam.len(), t0.elapsed().as_millis() / (k as u128 + 1)
            );
        }
    }
    let r = (MELON_AXES[0] * MELON_AXES[1] * MELON_AXES[2]).cbrt();
    println!("pool   the melon went {:.2} m under and floats with {:.0}% of its radius above the line", -(lowest - r), (drop.centre().z / r) * 100.0);
    let mp4 = out.join(format!("{tag}.mp4"));
    let st = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-framerate", &format!("{}", fps()), "-i"])
        .arg(dir.join("frame_%03d.png"))
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17"])
        .arg(&mp4)
        .status()?;
    if st.success() {
        println!("pool   {} frames → {}", frames, mp4.display());
    }
    Ok(())
}
