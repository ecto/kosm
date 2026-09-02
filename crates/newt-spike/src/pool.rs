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

use phyz::Simulator;
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder, State};
use tang::Vec3 as V;

use crate::glass::{fresnel, refract, reflect};

// ---- the pool ---------------------------------------------------------------

pub const POOL_X: f64 = 1.2; // half-lengths of the water
pub const POOL_Y: f64 = 0.8;
pub const DEPTH: f64 = 0.7;
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
    /// When the water is simulated, its free surface replaces the rings.
    pub grid: Option<crate::splash::HeightGrid>,
}

impl Surface {
    pub fn height(&self, x: f64, y: f64) -> f64 {
        let mut h = 0.0;
        for r in &self.rings {
            h += r.height(x, y, self.t);
        }
        if let Some(g) = &self.grid {
            // the fluid's surface, plus sub-grid rings from drops landing
            return g.at(x, y) + h;
        }
        // ambient ripple, 1 mm, so still water is not a mirror
        h + 0.0008 * ((7.0 * x + 3.0 * self.t).sin() * (5.0 * y - 2.0 * self.t).cos())
            + 0.0005 * ((11.0 * x - 4.0 * y + 1.7 * self.t).sin())
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
        let step = 0.004;
        while t < t_max {
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
    /// Last frame's fluid force on the melon, for the log.
    pub fluid_force: V<f64>,
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
            surface: Surface { rings: Vec::new(), t: 0.0, grid: None },
            last_splash: -1.0,
            entered: false,
            water: None,
            droplets: Vec::new(),
            fluid_force: V::zero(),
        }
    }

    /// Turn the water on: a dense MPM over the whole pool.
    pub fn with_water(mut self, h: f64) -> Self {
        // speed of sound ~ 45 m/s: at 5 m/s the impact pressure compresses the
        // water under a percent, which is what stops a melon instead of
        // letting it plough through
        let bulk = 2.0e6;
        let cs = (bulk / 1000.0f64).sqrt();
        let dt = 0.35 * h / cs;
        let mut water = crate::splash::Water::fill(h, dt, 0.45, bulk);
        // pack the fill down before anything arrives, and take the rest level
        water.settle(0.6);
        self.water = Some(water);
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
        let mut f = Vec3::zeros();
        for _ in 0..subs {
            f += water.step(&body);
        }
        let f = f / subs as f64;
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
        if let Some(w) = &self.water {
            let g = w.surface(0.02);
            let now = w.droplets(&g, 0.02, 400);
            let t = self.state.time;
            let mut landed = 0;
            let off = w.level_offset;
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
            self.surface.rings.retain(|r| t - r.t0 < 3.0);
            self.droplets = now;
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
    let nx = ((2.0 * POOL_X) / cell) as usize;
    let ny = ((2.0 * POOL_Y) / cell) as usize;
    let origin = [-POOL_X, -POOL_Y];
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
    for iy in 0..ny * sub {
        for ix in 0..nx * sub {
            // launch from the surface point that the flat refraction would
            // send to this floor cell, so the reference is uniform
            let fx = origin[0] + (ix as f64 + 0.5) * cell / sub as f64;
            let fy = origin[1] + (iy as f64 + 0.5) * cell / sub as f64;
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
    }
    Caustic { origin, cell, nx, ny, e }
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
pub fn radiance(view_o: V<f64>, d: V<f64>, drop: &Drop, melon: &Melon, caustic: &Caustic, depth: u32) -> [f64; 3] {
    let o = view_o;
    let s = sun_dir();
    // candidates: melon, deck/coping, water surface, pool walls above water
    let t_melon = melon.hit(o, d).map(|(t, n)| (t, n));
    // the deck: a plane at z = COPING outside the pool
    let t_deck = if d.z < 0.0 { Some((COPING - o.z) / d.z) } else { None };
    let inside_pool = |p: V<f64>| p.x.abs() < POOL_X && p.y.abs() < POOL_Y;
    // where the ray enters the pool column (z < COPING and inside), march for the surface
    let mut t_water = None;
    if d.z < 0.0 || o.z < COPING {
        // start marching at the deck plane if above it, else from the origin
        let t_start = if o.z > COPING { ((COPING - o.z) / d.z).max(0.0) } else { 0.0 };
        let p0 = o + d * t_start;
        // only if the ray is over the pool at the water line
        let t_line = if d.z != 0.0 { (0.06 - o.z) / d.z } else { 0.0 };
        let p_line = o + d * t_line.max(0.0);
        if inside_pool(p0) || inside_pool(p_line) {
            t_water = drop.surface.hit(p0, d, 6.0).map(|t| t + t_start);
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
    // pick the nearest of melon / water / wall / deck / drop
    let mut best_t = f64::INFINITY;
    let mut what = 0; // 0 sky
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
        _ => sky(d),
    }
}

/// A reflected ray from the water surface: melon or sky.
fn radiance_above(o: V<f64>, d: V<f64>, drop: &Drop, melon: &Melon, caustic: &Caustic) -> [f64; 3] {
    if let Some((t, n)) = melon.hit(o, d) {
        let p = o + d * t;
        return shade_melon(p, n, d, melon, drop, caustic, true);
    }
    sky(d)
}

/// A ray inside the water: absorb along the way, hit melon, walls, or the floor.
fn underwater(o: V<f64>, d: V<f64>, drop: &Drop, melon: &Melon, caustic: &Caustic) -> [f64; 3] {
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

fn shade_melon(p: V<f64>, n: V<f64>, d: V<f64>, melon: &Melon, drop: &Drop, caustic: &Caustic, in_air: bool) -> [f64; 3] {
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

pub fn render(view: &View, drop: &Drop, caustic: &Caustic) -> image::RgbaImage {
    let melon = drop.melon();
    let fwd = (view.target - view.eye).normalize();
    let right = fwd.cross(&V::new(0.0, 0.0, 1.0)).normalize();
    let up = right.cross(&fwd);
    let fy = 0.5 * view.height as f64 / (0.5 * view.vfov).tan();
    let mut img = image::RgbaImage::new(view.width, view.height);
    let to8 = |v: f64| {
        // a filmic-ish curve so the sun's highlight rolls off instead of clipping
        let v = v / (1.0 + v * 0.35) * 1.2;
        (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8
    };
    for y in 0..view.height {
        for x in 0..view.width {
            let px = (x as f64 + 0.5 - view.width as f64 / 2.0) / fy;
            let py = (view.height as f64 / 2.0 - y as f64 - 0.5) / fy;
            let d = (fwd + right * px + up * py).normalize();
            let c = radiance(view.eye, d, drop, &melon, caustic, 1);
            img.put_pixel(x, y, image::Rgba([to8(c[0]), to8(c[1]), to8(c[2]), 255]));
        }
    }
    img
}

/// The whole thing: drop the melon, render `frames` at 30 fps, encode.
pub fn run(out: &Path, frames: usize, width: u32, height: u32, splash: bool) -> anyhow::Result<()> {
    let dir = out.join(if splash { "splash" } else { "pool" });
    std::fs::create_dir_all(&dir)?;
    let mut drop = Drop::new(1.3);
    if splash {
        drop = drop.with_water(std::env::var("NEWT_H").ok().and_then(|v| v.parse().ok()).unwrap_or(0.025));
        let w = drop.water.as_ref().unwrap();
        println!("splash {} water particles on a {}×{}×{} grid at {} cm, dt {:.2e} s ({} substeps per ms)", w.count(), w.nx, w.ny, w.nz, w.h * 100.0, w.dt, (drop.model.dt / w.dt).ceil());
    }
    // from the deck corner, low, so the far water reflects the sky and the near
    // water shows the tiles
    let view = View { eye: V::new(-1.42, -1.22, 0.31), target: V::new(0.02, 0.12, -0.06), width, height, vfov: 0.9 };
    let steps_per_frame = (1.0 / 30.0 / drop.model.dt).round() as usize;
    let t0 = std::time::Instant::now();
    let mut lowest = f64::INFINITY;
    for k in 0..frames {
        for _ in 0..steps_per_frame {
            drop.step();
        }
        lowest = lowest.min(drop.centre().z);
        drop.read_water();
        let c = caustic(&drop.surface, 0.01);
        let peak_force = drop.fluid_force;
        drop.fluid_force = V::zero();
        let img = render(&view, &drop, &c);
        img.save(dir.join(format!("frame_{k:03}.png")))?;
        if k == 0 || k % 5 == 0 || k + 1 == frames {
            let m = drop.centre();
            println!(
                "pool   frame {k:3}  t={:.2} s  melon z={:+.3} m  vz={:+.2} m/s  fluid force z={:+.0} N  water inside {:.1} kg (weight {:.0} N)  drops={}  {} ms/frame",
                drop.state.time, m.z, drop.state.v[5], peak_force.z, drop.water.as_ref().map(|w| w.interior_mass).unwrap_or(0.0), drop.water.as_ref().map(|w| w.interior_mass * GRAVITY).unwrap_or(0.0), drop.droplets.len(), t0.elapsed().as_millis() / (k as u128 + 1)
            );
        }
    }
    let r = (MELON_AXES[0] * MELON_AXES[1] * MELON_AXES[2]).cbrt();
    println!("pool   the melon went {:.2} m under and floats with {:.0}% of its radius above the line", -(lowest - r), (drop.centre().z / r) * 100.0);
    let mp4 = out.join(if splash { "splash.mp4" } else { "pool.mp4" });
    let st = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-framerate", "30", "-i"])
        .arg(dir.join("frame_%03d.png"))
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17"])
        .arg(&mp4)
        .status()?;
    if st.success() {
        println!("pool   {} frames → {}", frames, mp4.display());
    }
    Ok(())
}
