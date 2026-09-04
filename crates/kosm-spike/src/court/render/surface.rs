//! What the surfaces are made of: materials, procedural textures, and the
//! `Surface` a hit resolves to.

use super::geometry::{mul, Hit, Shape};
use super::{Scene, V};

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

// ---- surfaces ---------------------------------------------------------------

/// What a surface point is made of.
pub(super) struct Surface {
    pub(super) albedo: V,
    /// Dielectric coat: F0 and its roughness; `coat = 0` means none.
    pub(super) coat: f64,
    pub(super) roughness: f64,
    pub(super) emission: V,
    /// A thin glass sheet: reflect or pass straight through, tinted.
    pub(super) sheet: bool,
    pub(super) tint: V,
}

pub(super) fn hash(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    x
}

pub(super) fn hash01(i: i64, j: i64, k: u64) -> f64 {
    (hash((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ k) & 0xFFFFFF) as f64
        / 16777216.0
}

/// Smooth value noise on a lattice.
pub(super) fn noise(x: f64, y: f64, seed: u64) -> f64 {
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
    pub(super) fn surface(&self, hit: &Hit) -> Surface {
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

