//! The glass in the light: a goblet of red wine on linen, in low sun.
//!
//! Lathed meshes for kosm-render's path tracer, and its photon map for the
//! caustic. Three surfaces, because the tracer tracks one medium at a time
//! (`cpu/integrator.rs`: "one slot, not a stack") and a glass of wine is the
//! nested case it names:
//!
//! - **the glass**: one lathed solid, foot to rim and down the inside to the
//!   waterline, soda-lime glass;
//! - **the wet wall**: the inside of the bowl below the waterline, the
//!   boundary of the wine, with the index *ratio* n_wine / n_glass and the
//!   wine's absorption — a ray crossing it from the glass enters the wine;
//! - **the surface**: the wine's free surface, a disc (or, in phase 4, the
//!   rippled height field), red wine against air.
//!
//! Metres throughout, so `attenuation_distance` is the library's own.
//! Every lathed triangle's geometric normal is to the *right* of the profile's
//! direction of travel in the (r, z) plane; see `lathe`.

use std::sync::Arc;

use kosm::prelude::*;
use kosm_render::caustics::{self, CausticMap, CausticOptions};
use kosm_render::pathtrace::{Camera, Environment, GradientEnv, Ground, Object, PathTraceOptions, Scene, Sun, render_with_caustics};
use kosm_render::{Bvh, Point3, TriMesh, Vec3};

/// Angular segments round the axis.
const AROUND: usize = 192;

/// The goblet's solid: sizes in metres.
pub struct Shape {
    /// The library name of what is in it.
    pub wine: &'static str,
    /// The bowl's inner surface, base to rim, `(r, z)`; the shell solver's profile.
    pub inner: Vec<[f64; 2]>,
    pub wall: f64,
    pub stem_r: f64,
    pub stem_h: f64,
    pub foot_r: f64,
    pub foot_t: f64,
}

impl Shape {
    /// z of the tablecloth: the underside of the foot.
    pub fn floor(&self) -> f64 {
        -self.wall - self.stem_h - self.foot_t
    }

    /// Inner radius of the bowl at height z (linear between profile points).
    pub fn inner_r(&self, z: f64) -> f64 {
        let p = &self.inner;
        for w in p.windows(2) {
            if (w[0][1]..=w[1][1]).contains(&z) {
                let t = if w[1][1] > w[0][1] { (z - w[0][1]) / (w[1][1] - w[0][1]) } else { 0.0 };
                return w[0][0] + t * (w[1][0] - w[0][0]);
            }
        }
        p.last().map(|q| q[0]).unwrap_or(0.0)
    }

    /// The glass's closed profile, axis to axis, stopping on the inside at `level`.
    fn glass_curve(&self, level: f64) -> Vec<[f64; 2]> {
        let floor = self.floor();
        let foot_top = floor + self.foot_t;
        let mut c = vec![[0.0, floor], [self.foot_r, floor], [self.foot_r, foot_top], [self.stem_r, foot_top], [self.stem_r, -self.wall]];
        // the outside of the bowl: the inner profile pushed out along its normal
        let outer = offset(&self.inner, self.wall);
        c.extend(outer.iter().copied());
        // the inside, rim down to the waterline (or to the axis when empty)
        let rim = *self.inner.last().expect("a profile has points");
        c.push(rim);
        for q in self.inner.iter().rev().skip(1) {
            if q[1] <= level {
                break;
            }
            c.push(*q);
        }
        if level > 0.0 {
            c.push([self.inner_r(level), level]);
        } else {
            c.push([0.0, 0.0]);
        }
        c
    }

    /// The wet wall, the wine's side and bottom: axis, out along the base, up
    /// the inside to the waterline. Normals into the glass.
    fn wet_curve(&self, level: f64) -> Vec<[f64; 2]> {
        let mut c = vec![[0.0, 0.0]];
        c.extend(self.inner.iter().copied().take_while(|q| q[1] < level));
        c.push([self.inner_r(level), level]);
        c
    }
}

/// A profile pushed `d` along its right-hand normal (outward for a bowl
/// traversed base to rim).
fn offset(p: &[[f64; 2]], d: f64) -> Vec<[f64; 2]> {
    (0..p.len())
        .map(|i| {
            let a = p[i.saturating_sub(1)];
            let b = p[(i + 1).min(p.len() - 1)];
            let (dr, dz) = (b[0] - a[0], b[1] - a[1]);
            let l = (dr * dr + dz * dz).sqrt().max(1e-12);
            [p[i][0] + d * dz / l, p[i][1] - d * dr / l]
        })
        .collect()
}

/// A surface of revolution round z. Each triangle's geometric normal is to
/// the right of the curve's direction of travel in the (r, z) plane.
pub fn lathe(curve: &[[f64; 2]]) -> TriMesh {
    let mut positions = Vec::with_capacity(curve.len() * AROUND);
    for q in curve {
        for j in 0..AROUND {
            let a = std::f64::consts::TAU * j as f64 / AROUND as f64;
            positions.push(Point3::new(q[0] * a.cos(), q[0] * a.sin(), q[1]));
        }
    }
    let mut indices = vec![];
    for i in 0..curve.len() - 1 {
        for j in 0..AROUND {
            let k = (j + 1) % AROUND;
            let (a, b, c, d) = (i * AROUND + j, i * AROUND + k, (i + 1) * AROUND + k, (i + 1) * AROUND + j);
            // (a, b, c) runs round then up the curve: θ̂ × travel is the right-hand normal
            for t in [[a, c, d], [a, b, c]] {
                let [p, q, r] = t.map(|i| positions[i]);
                if (q - p).cross(&(r - p)).norm() > 1e-18 {
                    indices.extend(t.map(|i| i as u32));
                }
            }
        }
    }
    TriMesh::new(positions, Vec::new(), &indices)
}

/// The wine's free surface with ripples on it at time `t`: a polar mesh,
/// fine (a tenth of a millimetre) in the band by the wall where the ripples
/// live, coarse inside, with shading normals from the ripples' own slope so
/// the light bends by the exact gradient rather than the facets'.
pub fn rippled_surface(shape: &Shape, level: f64, rip: &kosm::fluid::Ripples, t: f64) -> TriMesh {
    let r_wall = shape.inner_r(level);
    let band = (4.0 / rip.alpha).min(0.6 * r_wall);
    let step = (rip.wavelength() / 10.0).min(1e-4);
    let mut rings = vec![];
    let mut r = r_wall;
    while r > r_wall - band {
        rings.push(r);
        r -= step;
    }
    let inner = r_wall - band;
    for i in 1..=12 {
        rings.push(inner * (1.0 - i as f64 / 12.0));
    }
    let mut positions = Vec::with_capacity(rings.len() * AROUND);
    let mut normals = Vec::with_capacity(rings.len() * AROUND);
    for &r in &rings {
        for j in 0..AROUND {
            let a = std::f64::consts::TAU * j as f64 / AROUND as f64;
            let (x, y) = (r * a.cos(), r * a.sin());
            positions.push(Point3::new(x, y, level + rip.height(r, a, t)));
            let (gx, gy) = if r > 0.0 { rip.slope(x, y, t) } else { (0.0, 0.0) };
            normals.push(Vec3::new(-gx, -gy, 1.0).normalize());
        }
    }
    // outer ring to the axis: the same winding as `lathe`, so the normal is up
    let mut indices = vec![];
    for i in 0..rings.len() - 1 {
        for j in 0..AROUND {
            let k = (j + 1) % AROUND;
            let (a, b, c, d) = (i * AROUND + j, i * AROUND + k, (i + 1) * AROUND + k, (i + 1) * AROUND + j);
            for tri in [[a, c, d], [a, b, c]] {
                let [p, q, r] = tri.map(|i| positions[i]);
                if (q - p).cross(&(r - p)).norm() > 1e-18 {
                    indices.extend(tri.map(|i| i as u32));
                }
            }
        }
    }
    TriMesh::new(positions, normals, &indices)
}

/// The light, the table and the camera.
pub struct Stage {
    pub sun_el_deg: f64,
    pub sun_az_deg: f64,
    pub width: u32,
    pub height: u32,
    pub spp: u32,
    pub photons: usize,
    pub exposure: f32,
}

pub fn scene(shape: &Shape, level: f64, surface: Option<TriMesh>, stage: &Stage) -> Scene<TriMesh> {
    let glass = material::named("soda-lime glass").expect("glass").pbr();
    let wine = material::named(shape.wine).expect("the wine is in the library").pbr();
    let mut wet = wine;
    wet.ior = wine.ior / glass.ior;
    wet.abbe = 0.0; // a ratio of two dispersive indices is not a Cauchy index
    let mut objects = vec![Object::new(Arc::new(Bvh::build(lathe(&shape.glass_curve(level)))), glass)];
    if level > 0.0 {
        objects.push(Object::new(Arc::new(Bvh::build(lathe(&shape.wet_curve(level)))), wet));
        let top = surface.unwrap_or_else(|| lathe(&[[shape.inner_r(level), level], [0.0, level]]));
        objects.push(Object::new(Arc::new(Bvh::build(top)), wine));
    }

    Scene { objects, ..scene_rest(shape, stage) }
}

fn scene_rest(shape: &Shape, stage: &Stage) -> Scene<TriMesh> {
    let (el, az) = (stage.sun_el_deg.to_radians(), stage.sun_az_deg.to_radians());
    let linen = material::named("linen").expect("linen").pbr();
    Scene {
        objects: vec![],
        lights: vec![],
        env: Environment::Gradient(GradientEnv {
            zenith: [0.30, 0.36, 0.50],
            horizon: [0.60, 0.52, 0.44],
            ground: [0.15, 0.13, 0.11],
            intensity: 0.25,
        }),
        sun: Some(Sun::new(Vec3::new(el.cos() * az.cos(), el.cos() * az.sin(), el.sin()), 0.0047, [3.2, 2.8, 2.2])),
        ground: Some(Ground { z: shape.floor(), material: linen, shadow_catcher: false }),
        splats: None,
    }
}

pub fn caustic_map(scene: &Scene<TriMesh>, stage: &Stage) -> CausticMap {
    caustics::trace(scene, &CausticOptions { photons: stage.photons, radius: Some(2.0e-3), ..CausticOptions::default() })
}

/// The frame: the glass from the side, its shadow and caustic stretched
/// across the cloth away from the sun.
pub fn frame(scene: &Scene<TriMesh>, map: &CausticMap, shape: &Shape, stage: &Stage) -> image::RgbaImage {
    let (az, el) = (stage.sun_az_deg.to_radians(), stage.sun_el_deg.to_radians());
    let floor = shape.floor();
    // where the bowl's light lands: away from the sun, about the bowl's
    // height over tan(elevation) out
    let bowl_mid = -floor + 0.5 * shape.inner.last().map(|q| q[1]).unwrap_or(0.09);
    let reach = bowl_mid / el.tan();
    let away = Vec3::new(-az.cos(), -az.sin(), 0.0);
    let side = Vec3::new(-az.sin(), az.cos(), 0.0);
    // a full bowl is a cylindrical lens: its light gathers into a bright
    // line not far behind the glass, so the frame centres between the two
    let centre = Point3::new(0.0, 0.0, floor) + away * (0.3 * reach);
    let eye = centre + side * 0.42 - away * 0.12 + Vec3::new(0.0, 0.0, 0.30);
    let target = centre + Vec3::new(0.0, 0.0, 0.03);
    let cam = Camera::look_at(eye, target, Vec3::new(0.0, 0.0, 1.0), 40.0);
    let opts = PathTraceOptions { spp: stage.spp, max_depth: 16, ..PathTraceOptions::default() };
    let film = render_with_caustics(scene, &cam, stage.width, stage.height, &opts, Some(map));
    image::RgbaImage::from_raw(stage.width, stage.height, film.to_srgb8(stage.exposure, false)).expect("the film is width × height")
}

/// A close-up of the wine's surface by the side wall (square to the sun's
/// azimuth), from where the sun's reflection in a flat surface would land in
/// the lens: a still surface shows one glint, a rippled one a band of them.
///
/// The spot and the sun have to see each other over the rim, and so do the
/// spot and the lens: a shadow ray takes glass as opaque. At the side wall,
/// 3 mm in, the open chord runs 14 mm each way, so a sun above about 50°
/// clears a 17 mm freeboard. Use a high sun for this view.
pub fn surface_closeup(scene: &Scene<TriMesh>, shape: &Shape, level: f64, stage: &Stage) -> image::RgbaImage {
    let (az, el) = (stage.sun_az_deg.to_radians(), stage.sun_el_deg.to_radians());
    let r = shape.inner_r(level) - 3e-3;
    let side = az + std::f64::consts::FRAC_PI_2;
    let p = Point3::new(r * side.cos(), r * side.sin(), level);
    let mirror = Vec3::new(-el.cos() * az.cos(), -el.cos() * az.sin(), el.sin());
    let inward = Vec3::new(-side.cos(), -side.sin(), 0.0);
    let cam = Camera::look_at(p + mirror * 0.2, p + inward * 0.004, Vec3::new(0.0, 0.0, 1.0), 12.0);
    let opts = PathTraceOptions { spp: stage.spp, max_depth: 16, ..PathTraceOptions::default() };
    let film = kosm_render::pathtrace::render(scene, &cam, stage.width, stage.height, &opts);
    image::RgbaImage::from_raw(stage.width, stage.height, film.to_srgb8(stage.exposure, false)).expect("the film is width × height")
}

#[cfg(test)]
mod tests {
    use super::*;
    use kosm_render::Geometry;

    fn goblet() -> Shape {
        let bowl = kosm::shell::Profile::goblet(0.04, 0.09, 0.005, 40);
        Shape { wine: "red wine", inner: bowl.points, wall: 1.2e-3, stem_r: 0.004, stem_h: 0.07, foot_r: 0.035, foot_t: 0.003 }
    }

    #[test]
    fn the_wine_throws_red_light() {
        let stage = Stage { sun_el_deg: 24.0, sun_az_deg: 30.0, width: 8, height: 8, spp: 1, photons: 200_000, exposure: 1.0 };
        let shape = goblet();
        let power = |level: f64| caustic_map(&scene(&shape, level, None, &stage), &stage).deposited_power();
        let (empty, full) = (power(0.0), power(0.07));
        // red wine is nearly opaque across the bowl even in the red; what gets
        // through takes the thin edges, and the glass above the waterline still
        // throws white light. So: the light that lands is measurably redder.
        let (e, f) = (empty[0] / empty[1], full[0] / full[1]);
        assert!(e < 1.3, "an empty glass throws near-white light: red/green {e:.2}");
        assert!(f > 1.2 * e, "the wine reddens it: red/green {f:.2} vs {e:.2}");
    }

    #[test]
    fn a_lathed_wall_faces_right_of_travel() {
        // up the outside of a cylinder: right of travel is outward
        let m = lathe(&[[0.05, 0.0], [0.05, 0.1]]);
        let ray = kosm_render::Ray::new(Point3::new(0.2, 0.0, 0.05), Vec3::new(-1.0, 0.0, 0.0));
        let hit = (0..m.len())
            .filter_map(|i| m.intersect(&ray, i, 1e-9, f64::INFINITY))
            .min_by(|a, b| a.t.total_cmp(&b.t))
            .expect("the ray meets the wall");
        assert!(hit.normal.x > 0.9, "normal {:?}", hit.normal);
    }
}
