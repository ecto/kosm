//! The court, lit: a path tracer.
//!
//! Where `frame.rs` is a ray caster that is its own derivative, this is the
//! other end of the same idea — the reference tier for light on the court,
//! written plainly on `f64` and thrown at every core. Geometry is the same
//! derived colliders the physics stands on (the slab, the backboard, the
//! bracket, the pole, one box each; the balls as spheres with their contact
//! pose) plus a gym the level describes: walls, a ceiling, and rows of light
//! panels that are the only light there is. The rim is the one place the
//! picture and the physics disagree on purpose — the physics has 24 box
//! segments, the picture has the torus they approximate.
//!
//! Light transport is unidirectional path tracing with next-event estimation
//! on the panels, multiple importance sampling between the panel and the
//! BSDF, Russian roulette after the third bounce, and one sample stream per
//! pixel seeded by pixel, sample and frame so a frame is a pure function of
//! the state. Surfaces: a Lambertian base under a dielectric coat where a
//! coat belongs (lacquered maple, painted steel, rubber), the backboard as a
//! thin dielectric sheet with its painted square, procedural maple planks
//! and court markings, and a ball whose seams turn with its body frame.

use std::f64::consts::PI;

use phyz_math::Mat3;
use phyz_model::{GeomInstance, Geometry};
use rayon::prelude::*;
use tang::Vec3 as V3;

use super::{Court, CourtScene};

type V = V3<f64>;

// ---- the picture ------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: V,
    pub target: V,
    pub vfov: f64,
    pub width: u32,
    pub height: u32,
    pub exposure: f64,
}

impl Camera {
    fn basis(&self) -> (V, V, V) {
        let f = (self.target - self.eye).normalize();
        let r = f.cross(V::z()).normalize();
        let u = r.cross(f);
        (f, r, u)
    }

    /// The ray through film position `(sx, sy)` in pixels, `y` down.
    fn ray(&self, sx: f64, sy: f64) -> (V, V) {
        let (f, r, u) = self.basis();
        let t = (0.5 * self.vfov).tan();
        let aspect = self.width as f64 / self.height as f64;
        let x = (2.0 * sx / self.width as f64 - 1.0) * t * aspect;
        let y = (1.0 - 2.0 * sy / self.height as f64) * t;
        (self.eye, (f + r * x + u * y).normalize())
    }
}

// ---- geometry ---------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Material {
    /// Lacquered maple, with the court painted on it.
    Floor,
    /// The ball: pebbled rubber with its seams in the body frame.
    Ball,
    /// The backboard: a thin glass sheet with the painted square and border.
    Glass,
    /// The rim: painted orange steel.
    Rim,
    /// Bracket, arm, pole: painted grey steel.
    Steel,
    Wall,
    Ceiling,
    /// A light panel: emissive on its underside.
    Light,
}

#[derive(Clone, Debug)]
enum Shape {
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
struct Prim {
    shape: Shape,
    mat: Material,
}

struct Hit {
    t: f64,
    p: V,
    n: V,
    prim: usize,
}

fn mul(m: &[[f64; 3]; 3], v: V) -> V {
    V::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

fn mul_t(m: &[[f64; 3]; 3], v: V) -> V {
    V::new(
        m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z,
        m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z,
        m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z,
    )
}

fn arr(m: &Mat3) -> [[f64; 3]; 3] {
    let f = |i: usize, j: usize| m[(i, j)];
    [[f(0, 0), f(0, 1), f(0, 2)], [f(1, 0), f(1, 1), f(1, 2)], [f(2, 0), f(2, 1), f(2, 2)]]
}

fn pv(v: phyz_math::Vec3) -> V {
    V::new(v.x, v.y, v.z)
}

const EPS: f64 = 1e-6;

impl Shape {
    fn hit(&self, o: V, d: V, t_max: f64) -> Option<(f64, V)> {
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

// ---- the scene --------------------------------------------------------------

pub struct Scene {
    prims: Vec<Prim>,
    /// Indices of the light panels.
    lights: Vec<usize>,
    light_radiance: f64,
    hoop: super::Hoop,
    /// Where the court's lines are painted from: the baseline, 4 ft behind the board.
    baseline_x: f64,
}

impl Scene {
    /// The scene at the court's current state.
    pub fn new(scene: &CourtScene, court: &Court) -> anyhow::Result<Self> {
        let mut prims = Vec::new();
        let a = &scene.authored;
        let mm = |k: &str| a.millimetres(k);

        // the level's parts, by their root material
        for (material, instances) in super::parts(scene)? {
            let mat = match material.as_str() {
                "maple" => Material::Floor,
                "glass" => Material::Glass,
                "rim" => continue, // drawn as a torus below
                _ => Material::Steel,
            };
            for g in instances {
                prims.push(Prim { shape: shape_of(&g), mat });
            }
        }
        let rod = mm("rim_rod_mm")?;
        prims.push(Prim {
            shape: Shape::Torus {
                c: court.hoop.rim_centre - V::new(0.0, 0.0, 0.5 * rod),
                big_r: court.hoop.rim_r + 0.5 * rod,
                small_r: 0.5 * rod,
            },
            mat: Material::Rim,
        });

        // the balls
        for k in 0..court.bodies() {
            prims.push(Prim {
                shape: Shape::Sphere { c: pv(court.centre(k)), r: scene.ball_r, rot: arr(&court.rotation(k)) },
                mat: Material::Ball,
            });
        }

        // the gym: walls a margin outside the slab, a ceiling, light panels
        let (cx, cy) = (0.5 * mm("court_x_mm")?, 0.5 * mm("court_y_mm")?);
        let margin = mm("gym_margin_mm")?;
        let h = mm("gym_h_mm")?;
        let (wx, wy) = (cx + margin, cy + margin);
        let wall = |c: V, half: V| Prim { shape: Shape::Box { c, half, rot: arr(&Mat3::identity()) }, mat: Material::Wall };
        let thick = 0.1;
        prims.push(wall(V::new(wx + thick, 0.0, 0.5 * h), V::new(thick, wy + 2.0 * thick, 0.5 * h + thick)));
        prims.push(wall(V::new(-wx - thick, 0.0, 0.5 * h), V::new(thick, wy + 2.0 * thick, 0.5 * h + thick)));
        prims.push(wall(V::new(0.0, wy + thick, 0.5 * h), V::new(wx + 2.0 * thick, thick, 0.5 * h + thick)));
        prims.push(wall(V::new(0.0, -wy - thick, 0.5 * h), V::new(wx + 2.0 * thick, thick, 0.5 * h + thick)));
        // the floor beyond the slab, and the ceiling
        prims.push(Prim {
            shape: Shape::Box { c: V::new(0.0, 0.0, -0.021), half: V::new(wx, wy, 0.02), rot: arr(&Mat3::identity()) },
            mat: Material::Wall,
        });
        prims.push(Prim {
            shape: Shape::Box { c: V::new(0.0, 0.0, h + thick), half: V::new(wx, wy, thick), rot: arr(&Mat3::identity()) },
            mat: Material::Ceiling,
        });
        let (rows, cols) = (a.parameter("light_rows")?.max(1.0) as usize, a.parameter("light_cols")?.max(1.0) as usize);
        let (lw, ll) = (mm("light_w_mm")?, mm("light_l_mm")?);
        let mut lights = Vec::new();
        for i in 0..cols {
            for j in 0..rows {
                let x = (i as f64 + 0.5) / cols as f64 * 2.0 * wx - wx;
                let y = (j as f64 + 0.5) / rows as f64 * 2.0 * wy - wy;
                lights.push(prims.len());
                prims.push(Prim {
                    shape: Shape::Panel {
                        c: V::new(x - 0.5 * lw, y - 0.5 * ll, h - 0.005),
                        u: V::new(lw, 0.0, 0.0),
                        v: V::new(0.0, ll, 0.0),
                    },
                    mat: Material::Light,
                });
            }
        }

        Ok(Self {
            prims,
            lights,
            light_radiance: a.parameter_or("light_radiance", 18.0),
            hoop: court.hoop,
            baseline_x: court.hoop.board_x + 1.219,
        })
    }

    fn nearest(&self, o: V, d: V, t_max: f64) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        for (i, prim) in self.prims.iter().enumerate() {
            let limit = best.as_ref().map_or(t_max, |h| h.t);
            if let Some((t, n)) = prim.shape.hit(o, d, limit) {
                best = Some(Hit { t, p: o + d * t, n, prim: i });
            }
        }
        best
    }

    /// What the camera's centre ray sees, for a sanity check: distance and material.
    pub fn probe(&self, cam: &Camera) -> Option<(f64, Material)> {
        let (o, d) = cam.ray(0.5 * cam.width as f64, 0.5 * cam.height as f64);
        self.nearest(o, d, f64::INFINITY).map(|h| (h.t, self.prims[h.prim].mat))
    }

    pub fn prim_count(&self) -> usize {
        self.prims.len()
    }

    fn occluded(&self, o: V, d: V, t_max: f64) -> bool {
        self.prims.iter().enumerate().any(|(i, prim)| {
            // glass is not an occluder for the panels: the sheet is thin and
            // its transmission is booked on the path through it
            prim.mat != Material::Glass && !self.lights.contains(&i) && prim.shape.hit(o, d, t_max).is_some()
        })
    }
}

fn shape_of(g: &GeomInstance) -> Shape {
    let c = pv(g.origin.pos);
    match g.geometry {
        Geometry::Box { half_extents } => Shape::Box { c, half: pv(half_extents), rot: arr(&g.origin.rot) },
        Geometry::Sphere { radius } => Shape::Sphere { c, r: radius, rot: arr(&g.origin.rot) },
        Geometry::Cylinder { radius, height } => Shape::Cylinder { c, r: radius, half_h: 0.5 * height },
        _ => Shape::Sphere { c, r: 0.0, rot: arr(&Mat3::identity()) },
    }
}

// ---- surfaces ---------------------------------------------------------------

/// What a surface point is made of.
struct Surface {
    albedo: V,
    /// Dielectric coat: F0 and its roughness; `coat = 0` means none.
    coat: f64,
    roughness: f64,
    emission: V,
    /// A thin glass sheet: reflect or pass straight through, tinted.
    sheet: bool,
    tint: V,
}

fn hash(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    x
}

fn hash01(i: i64, j: i64, k: u64) -> f64 {
    (hash((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ k) & 0xFFFFFF) as f64
        / 16777216.0
}

/// Smooth value noise on a lattice.
fn noise(x: f64, y: f64, seed: u64) -> f64 {
    let (i, j) = (x.floor() as i64, y.floor() as i64);
    let (fx, fy) = (x - i as f64, y - j as f64);
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(fx), s(fy));
    let a = hash01(i, j, seed);
    let b = hash01(i + 1, j, seed);
    let c = hash01(i, j + 1, seed);
    let d = hash01(i + 1, j + 1, seed);
    (a * (1.0 - sx) + b * sx) * (1.0 - sy) + (c * (1.0 - sx) + d * sx) * sy
}

impl Scene {
    fn surface(&self, hit: &Hit) -> Surface {
        let prim = &self.prims[hit.prim];
        let none = Surface { albedo: V::zero(), coat: 0.0, roughness: 0.5, emission: V::zero(), sheet: false, tint: V::splat(1.0) };
        match prim.mat {
            Material::Floor => Surface { albedo: self.floor_albedo(hit.p), coat: 0.045, roughness: 0.12, ..none },
            Material::Ball => {
                let Shape::Sphere { c, r, ref rot } = prim.shape else { unreachable!() };
                let body = mul(rot, hit.p - c) / r;
                let seam = {
                    let w = 0.035;
                    body.z.abs() < w || body.x.abs() < w || (body.y.abs() - 0.62).abs() < w
                };
                // pebbled rubber: a little albedo grain
                let grain = 0.92 + 0.08 * noise(body.x * 60.0 + body.z * 37.0, body.y * 60.0 - body.z * 23.0, 7);
                let albedo = if seam { V::new(0.05, 0.035, 0.03) } else { V::new(0.78, 0.30, 0.09) * grain };
                Surface { albedo, coat: 0.02, roughness: 0.55, ..none }
            }
            Material::Glass => {
                // the painted border and the shooter's square (24 x 18 in, 2 in lines)
                let (y, z) = (hit.p.y, hit.p.z - self.hoop.rim_centre.z);
                let line = 0.0508;
                let square = {
                    let (hw, hh) = (0.3048, 0.2286);
                    let inside = y.abs() < hw + line && z > -line && z < 2.0 * hh + line;
                    let hollow = y.abs() < hw && z > 0.0 && z < 2.0 * hh;
                    inside && !hollow
                };
                let border = {
                    let Shape::Box { c, half, .. } = prim.shape else { unreachable!() };
                    (y.abs() > half.y - line) || ((hit.p.z - c.z).abs() > half.z - line)
                };
                if square || border {
                    Surface { albedo: V::splat(0.85), coat: 0.04, roughness: 0.25, ..none }
                } else {
                    Surface { sheet: true, tint: V::new(0.86, 0.93, 0.90), coat: 0.04, roughness: 0.0, ..none }
                }
            }
            Material::Rim => Surface { albedo: V::new(0.80, 0.26, 0.06), coat: 0.06, roughness: 0.18, ..none },
            Material::Steel => Surface { albedo: V::splat(0.22), coat: 0.06, roughness: 0.3, ..none },
            Material::Wall => Surface { albedo: V::new(0.72, 0.70, 0.64), ..none },
            Material::Ceiling => Surface { albedo: V::splat(0.30), ..none },
            Material::Light => Surface { emission: V::splat(self.light_radiance), ..none },
        }
    }

    /// Maple planks along x, a little colour per plank, grain along their
    /// length, and the court painted on top.
    fn floor_albedo(&self, p: V) -> V {
        let plank_w = 0.057;
        let plank_l = 1.2;
        let row = (p.y / plank_w).floor();
        let offset = hash01(row as i64, 0, 3) * plank_l;
        let col = ((p.x + offset) / plank_l).floor();
        let seed = hash01(row as i64, col as i64, 5);
        let base = V::new(0.62, 0.45, 0.27) * (0.88 + 0.24 * seed);
        let grain = 0.93 + 0.07 * noise(p.x * 3.0, p.y * 140.0 + row * 7.0, 11) + 0.04 * noise(p.x * 40.0, p.y * 900.0, 13);
        let mut c = base * grain;
        // plank edges
        let ey = ((p.y / plank_w).fract() - 0.5).abs();
        let ex = (((p.x + offset) / plank_l).fract() - 0.5).abs();
        if ey > 0.47 || ex > 0.497 {
            c = c * 0.82;
        }
        // the markings: baseline, the lane, the free-throw line and circle,
        // the three-point arc; 50 mm lines
        let line = 0.05;
        let bx = self.baseline_x;
        let ft_x = bx - 5.79; // free-throw line, 19 ft from the baseline
        let lane_hw = 2.44; // 16 ft wide
        let dist_x = |x0: f64| (p.x - x0).abs();
        let on_baseline = dist_x(bx) < line * 0.5 && p.y.abs() < 7.62;
        let on_lane = (p.x > ft_x && p.x < bx) && (p.y.abs() - lane_hw).abs() < line * 0.5;
        let on_ft = dist_x(ft_x) < line * 0.5 && p.y.abs() < lane_hw;
        let ft_r = (p.x - ft_x).hypot(p.y);
        let on_circle = (ft_r - 1.83).abs() < line * 0.5 && p.x < ft_x;
        let basket_x = self.hoop.rim_centre.x;
        let three = (p.x - basket_x).hypot(p.y);
        let on_three = (three - 7.24).abs() < line * 0.5 && p.x < bx && p.y.abs() < 6.7;
        let lane_paint = p.x > ft_x && p.x < bx && p.y.abs() < lane_hw;
        if on_baseline || on_lane || on_ft || on_circle || on_three {
            return V::new(0.92, 0.92, 0.90);
        }
        if lane_paint {
            return c.hadamard(V::new(0.30, 0.42, 0.95)) * 0.8;
        }
        c
    }
}

// ---- transport --------------------------------------------------------------

/// A small counter-based generator: every sample is a pure function of
/// (pixel, sample, frame).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(hash(seed | 1))
    }
    fn next(&mut self) -> f64 {
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
    fn trace(&self, mut o: V, mut d: V, rng: &mut Rng) -> V {
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
fn tonemap(c: V, exposure: f64) -> [u8; 3] {
    let f = |x: f64| {
        let x = x * exposure;
        let y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
        (y.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8
    };
    [f(c.x), f(c.y), f(c.z)]
}

/// Render the scene: `spp` samples per pixel, one stream per pixel and frame.
pub fn render(scene: &Scene, cam: &Camera, spp: usize, frame: u64) -> image::RgbImage {
    let (w, h) = (cam.width, cam.height);
    let rows: Vec<Vec<[u8; 3]>> = (0..h)
        .into_par_iter()
        .map(|y| {
            (0..w)
                .map(|x| {
                    let mut sum = V::zero();
                    for s in 0..spp {
                        let mut rng = Rng::new(((frame << 40) ^ ((y as u64) << 20) ^ (x as u64)) * 0x2545_F491_4F6C_DD1D + s as u64);
                        let (o, d) = cam.ray(x as f64 + rng.next(), y as f64 + rng.next());
                        sum += scene.trace(o, d, &mut rng);
                    }
                    tonemap(sum / spp as f64, cam.exposure)
                })
                .collect()
        })
        .collect();
    let mut img = image::RgbImage::new(w, h);
    for (y, row) in rows.iter().enumerate() {
        for (x, px) in row.iter().enumerate() {
            img.put_pixel(x as u32, y as u32, image::Rgb(*px));
        }
    }
    img
}

/// The camera the level asks for, at a given picture size.
pub fn camera(scene: &CourtScene, width: u32, height: u32) -> anyhow::Result<Camera> {
    let a = &scene.authored;
    Ok(Camera {
        eye: V::new(a.millimetres("cam_x_mm")?, a.millimetres("cam_y_mm")?, a.millimetres("cam_z_mm")?),
        target: V::new(a.millimetres("cam_at_x_mm")?, a.millimetres("cam_at_y_mm")?, a.millimetres("cam_at_z_mm")?),
        vfov: a.parameter_or("cam_vfov_deg", 42.0).to_radians(),
        width,
        height,
        exposure: a.parameter_or("exposure", 1.0),
    })
}
