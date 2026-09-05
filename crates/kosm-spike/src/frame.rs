//! The frame, as a solver.
//!
//! A ray caster written once, generic over `tang::Scalar`. Run it on `f64` and
//! it is a renderer: real shadows from a lamp of finite size, cast by the
//! derived colliders themselves, no compositing. Run it on `Dual<f64>` and the
//! same code is its own derivative: `∂pixel/∂(marble position)`, exact, from
//! one pass per seed. That is the rung-3 claim in miniature: light is another
//! physics on the same scalar as contact, and the image has gradients.
//!
//! Silhouettes are step functions and have no derivative; a finite lamp fixes
//! that where it matters. The sphere's shadow has a penumbra whose width is
//! set by the lamp radius, so the shadow's *position* is a smooth function of
//! the marble's, and "drag the shadow in the image" has a gradient. Box-on-box
//! shadows stay hard: nothing there moves.
//!
//! Objective: the mean brightness of the plate pixels inside a target disc.
//! Minimizing it puts the shadow on the target. Its gradient in the marble's
//! position comes from three dual passes over the disc's pixels only; the
//! contact adjoint carries it back to the release point.

use phyz_camera::CameraPose;
use phyz_diff::FinalStateObjective;
use phyz_math::SpatialTransformExt;
use phyz_model::{GeomInstance, Geometry};
use phyz_world::CameraIntrinsics;
use crate::analytic::Analytic;
use crate::glass::{self, Glass};
use tang::{Dual, Mat3, Scalar, Vec3};

const POS: usize = 3;

/// An oriented box in world coordinates.
#[derive(Clone)]
struct OBox<S: Scalar> {
    center: Vec3<S>,
    half: Vec3<S>,
    /// world → box rotation.
    rot: Mat3<S>,
    albedo: S,
}

#[derive(Clone)]
pub struct Scene<S: Scalar> {
    boxes: Vec<OBox<S>>,
    marble: Vec3<S>,
    marble_r: S,
    pub lamp: Vec3<S>,
    lamp_r: S,
    lamp_power: S,
    ambient: S,
    /// Glass solids the camera sees through. When non-empty the opaque
    /// marble is out of the picture.
    pub glass: Vec<Glass<S>>,
    /// Plate frame (world→plate), for the printed grid under the samples.
    pub plate: Option<phyz_math::SpatialTransform>,
    /// Caustics on the plate, looked up wherever a ray lands on it, glass or
    /// no glass in between. Light bands 650/550/450 nm feed camera bands 0/1/2.
    pub caustics: Vec<crate::light::Caustic<f64>>,
}

/// What a primary ray landed on.
#[derive(Clone, Copy)]
enum Hit<S: Scalar> {
    Solid { albedo: S, plate: bool },
    Marble,
    Glass(usize),
}

fn v3<S: Scalar>(v: phyz_math::Vec3) -> Vec3<S> {
    Vec3::new(S::from_f64(v.x), S::from_f64(v.y), S::from_f64(v.z))
}

fn m3<S: Scalar>(m: &phyz_math::Mat3) -> Mat3<S> {
    let f = |i: usize, j: usize| S::from_f64(m[(i, j)]);
    Mat3::new(f(0, 0), f(0, 1), f(0, 2), f(1, 0), f(1, 1), f(1, 2), f(2, 0), f(2, 1), f(2, 2))
}

impl<S: Scalar> Scene<S> {
    /// The static world from the track body's colliders, placed in the world.
    pub fn new(
        track: &phyz_math::SpatialTransform,
        colliders: &[GeomInstance],
        marble: Vec3<S>,
        marble_r: f64,
        lamp: phyz_math::Vec3,
        lamp_r: f64,
    ) -> Self {
        let boxes = colliders
            .iter()
            .filter_map(|g| match g.geometry {
                Geometry::Box { half_extents } => {
                    let center = track.body_to_world_point(g.origin.pos);
                    // origin.rot: body→shape; track.rot: world→body. world→shape = origin.rot · track.rot.
                    let rot = g.origin.rot * track.rot;
                    Some(OBox { center: v3(center), half: v3(half_extents), rot: m3(&rot), albedo: S::from_f64(0.78) })
                }
                _ => None,
            })
            .collect();
        Self {
            boxes,
            marble,
            marble_r: S::from_f64(marble_r),
            lamp: v3(lamp),
            lamp_r: S::from_f64(lamp_r),
            lamp_power: S::from_f64(0.09),
            ambient: S::from_f64(0.16),
            glass: Vec::new(),
            plate: None,
            caustics: Vec::new(),
        }
    }

    /// Irradiance from every caustic at a world point on the plate, per camera band.
    fn caustic_at(&self, p_world: Vec3<S>, band: usize) -> S {
        if self.caustics.is_empty() {
            return S::ZERO;
        }
        let Some(xf) = &self.plate else { return S::ZERO };
        let pl = xf.world_to_body_point(phyz_math::Vec3::new(p_world.x.to_f64(), p_world.y.to_f64(), p_world.z.to_f64()));
        let light_band = [4usize, 2, 0][band];
        let mut e = 0.0;
        for c in &self.caustics {
            e += c.at(light_band, pl.x, pl.y);
        }
        S::from_f64(e)
    }

    /// Camera bands (µm) for glass: three are enough to show colour fringes.
    const CAM_BANDS: [f64; 3] = [0.65, 0.55, 0.45];

    /// Plate albedo with a 5 mm printed grid, so refraction has something to bend.
    fn plate_albedo(&self, p_world: Vec3<S>) -> S {
        let Some(xf) = &self.plate else { return S::from_f64(0.78) };
        let pl = xf.world_to_body_point(phyz_math::Vec3::new(p_world.x.to_f64(), p_world.y.to_f64(), p_world.z.to_f64()));
        let f = |v: f64| ((v / 0.005).rem_euclid(1.0) - 0.5).abs();
        let on_line = f(pl.x) > 0.44 || f(pl.y) > 0.44;
        S::from_f64(if on_line { 0.45 } else { 0.82 })
    }

    /// Radiance of the lamp's disc if a ray hits it.
    fn lamp_seen(&self, o: Vec3<S>, d: Vec3<S>) -> Option<S> {
        let oc = self.lamp - o;
        let t = oc.dot(&d);
        if t <= S::ZERO {
            return None;
        }
        let miss = (oc - d * t).norm();
        // radiance = radiant intensity / disc area; it saturates, as a lamp does
        (miss < self.lamp_r).then(|| self.lamp_power / (S::PI * self.lamp_r * self.lamp_r))
    }

    /// Nearest hit including glass.
    fn hit_any(&self, o: Vec3<S>, d: Vec3<S>) -> Option<(S, Vec3<S>, Hit<S>)> {
        let mut best: Option<(S, Vec3<S>, Hit<S>)> = self.hit(o, d).map(|(t, n, a, plate)| (t, n, Hit::Solid { albedo: a, plate }));
        if self.glass.is_empty() {
            if let Some(t) = sphere_hit(self.marble, self.marble_r, o, d) {
                if best.as_ref().is_none_or(|h| t < h.0) {
                    best = Some((t, (o + d * t - self.marble).normalize(), Hit::Marble));
                }
            }
        }
        for (i, g) in self.glass.iter().enumerate() {
            if let Some((t, n)) = g.shape.enter(o, d) {
                if best.as_ref().is_none_or(|h| t < h.0) {
                    best = Some((t, n, Hit::Glass(i)));
                }
            }
        }
        best
    }

    /// Radiance along a ray for one camera band, following glass for up to
    /// `depth` interface events.
    pub fn radiance(&self, o: Vec3<S>, d: Vec3<S>, band: usize, depth: usize) -> S {
        if let Some(l) = self.lamp_seen(o, d) {
            // the lamp itself, if nothing is in front of it
            if self.hit_any(o, d).is_none_or(|h| h.0 > (self.lamp - o).norm()) {
                return l;
            }
        }
        let Some((t, n, what)) = self.hit_any(o, d) else {
            // the room: a soft hemisphere, brighter toward the ceiling, so
            // glass has something to reflect besides black
            return S::from_f64(0.06) + (d.z.max(S::ZERO)) * S::from_f64(0.22);
        };
        let p = o + d * t;
        match what {
            Hit::Solid { albedo, plate } => {
                let albedo = if plate { self.plate_albedo(p) } else { albedo };
                let q = p + n * S::from_f64(1e-5);
                let to = self.lamp - q;
                let r2 = to.norm_sq();
                let l = to / r2.sqrt();
                let cos = n.dot(&l).max(S::ZERO);
                let caustic = if plate { self.caustic_at(p, band) * self.lamp_power } else { S::ZERO };
                albedo * (self.ambient + self.lamp_power * cos / r2 * self.lamp_visibility(q) + caustic)
            }
            Hit::Marble => {
                let q = p + n * S::from_f64(1e-5);
                let to = self.lamp - q;
                let r2 = to.norm_sq();
                let l = to / r2.sqrt();
                let cos = n.dot(&l).max(S::ZERO);
                S::from_f64(0.92) * (self.ambient + self.lamp_power * cos / r2 * self.lamp_visibility(q))
            }
            Hit::Glass(i) => {
                if depth == 0 {
                    return S::from_f64(0.02);
                }
                let g = &self.glass[i];
                let n_glass = crate::light::index::<S>(g.nd, Self::CAM_BANDS[band]);
                let Some((d_in, cos_i, cos_t)) = glass::refract(d, n, S::ONE, n_glass) else {
                    return S::ZERO;
                };
                let r = glass::fresnel(S::ONE, n_glass, cos_i, cos_t);
                // the reflected share sees the world, the lamp included
                let d_r = glass::reflect(d, n);
                let reflected = self.radiance(p + n * S::from_f64(1e-6), d_r, band, depth - 1);
                // the refracted share walks through and out
                let transmitted = match glass::walk_inside(&g.shape, p, d_in, n_glass, S::from_f64(0.5), 4) {
                    Some((q, d_out, tr)) => tr * self.radiance(q + d_out * S::from_f64(1e-6), d_out, band, depth - 1),
                    None => S::ZERO,
                };
                r * reflected + (S::ONE - r) * transmitted
            }
        }
    }

    /// Camera RGB for a pixel: three bands through glass, one otherwise.
    pub fn rgb(&self, o: Vec3<S>, d: Vec3<S>) -> [S; 3] {
        match self.hit_any(o, d) {
            Some((_, _, Hit::Glass(_))) => [self.radiance(o, d, 0, 4), self.radiance(o, d, 1, 4), self.radiance(o, d, 2, 4)],
            _ => {
                let v = self.radiance(o, d, 1, 1);
                [v, v, v]
            }
        }
    }

    /// Nearest hit along `o + t d`, `t` in metres: (t, normal, albedo, is_plate_pixel).
    fn hit(&self, o: Vec3<S>, d: Vec3<S>) -> Option<(S, Vec3<S>, S, bool)> {
        let mut best: Option<(S, Vec3<S>, S, bool)> = None;
        for (k, b) in self.boxes.iter().enumerate() {
            if let Some((t, n)) = box_hit(b, o, d) {
                if best.as_ref().is_none_or(|h| t < h.0) {
                    best = Some((t, n, b.albedo, k == 0));
                }
            }
        }
        if let Some(t) = sphere_hit(self.marble, self.marble_r, o, d) {
            if best.as_ref().is_none_or(|h| t < h.0) {
                let p = o + d * t;
                best = Some((t, (p - self.marble).normalize(), S::from_f64(0.92), false));
            }
        }
        best
    }

    /// Visibility of the lamp from `p`: hard for boxes, a penumbra for the marble.
    pub fn lamp_visibility(&self, p: Vec3<S>) -> S {
        let to = self.lamp - p;
        let dist = to.norm();
        let d = to / dist;
        for b in &self.boxes {
            if let Some((t, _)) = box_hit(b, p, d) {
                if t < dist {
                    return S::ZERO;
                }
            }
        }
        for g in &self.glass {
            if let Some((t, _)) = g.shape.enter(p, d) {
                if t < dist {
                    // the glass blocks the direct light; its caustic puts the light back
                    return S::ZERO;
                }
            }
        }
        if !self.glass.is_empty() {
            return S::ONE;
        }
        // Closest approach of the shadow segment to the marble centre.
        let oc = self.marble - p;
        let t = oc.dot(&d).clamp(S::ZERO, dist);
        let miss = (oc - d * t).norm() - self.marble_r;
        // Penumbra half-width at the occluder: lamp radius scaled by the
        // occluder's position along the segment (similar triangles).
        let w = self.lamp_r * (t / dist).max(S::from_f64(1e-3));
        smoothstep((miss / w + S::HALF).clamp(S::ZERO, S::ONE))
    }

    /// Where a primary ray meets the plate (box 0), if it does, with the
    /// marble out of the way.
    pub fn plate_point(&self, o: Vec3<S>, d: Vec3<S>) -> Option<Vec3<S>> {
        let mut s = self.clone();
        s.marble = Vec3::new(S::ZERO, S::ZERO, S::from_f64(-10.0));
        match s.hit(o, d) {
            Some((t, _, _, true)) => Some(o + d * t),
            _ => None,
        }
    }

    /// Radiance for one primary ray.
    pub fn shade(&self, o: Vec3<S>, d: Vec3<S>) -> (S, bool) {
        let Some((t, n, albedo, plate)) = self.hit(o, d) else {
            return (S::from_f64(0.02), false);
        };
        let p = o + d * t + n * S::from_f64(1e-5);
        let to = self.lamp - p;
        let r2 = to.norm_sq();
        let l = to / r2.sqrt();
        let cos = n.dot(&l).max(S::ZERO);
        let direct = self.lamp_power * cos / r2 * self.lamp_visibility(p);
        (albedo * (self.ambient + direct), plate)
    }
}

fn smoothstep<S: Scalar>(x: S) -> S {
    x * x * (S::from_f64(3.0) - S::TWO * x)
}

fn sphere_hit<S: Scalar>(c: Vec3<S>, r: S, o: Vec3<S>, d: Vec3<S>) -> Option<S> {
    let oc = o - c;
    let b = oc.dot(&d);
    let disc = b * b - (oc.norm_sq() - r * r);
    if disc < S::ZERO {
        return None;
    }
    let t = -b - disc.sqrt();
    (t > S::from_f64(1e-6)).then_some(t)
}

/// Slab test in the box frame. Returns (t, world normal).
fn box_hit<S: Scalar>(b: &OBox<S>, o: Vec3<S>, d: Vec3<S>) -> Option<(S, Vec3<S>)> {
    let ol = b.rot * (o - b.center);
    let dl = b.rot * d;
    let (mut tmin, mut tmax) = (S::NEG_INFINITY, S::INFINITY);
    let mut axis = 0usize;
    let mut sign = S::ONE;
    let (ol_a, dl_a, h_a) = (ol.as_array(), dl.as_array(), b.half.as_array());
    for k in 0..3 {
        let inv = S::ONE / dl_a[k];
        let (mut t0, mut t1) = ((-h_a[k] - ol_a[k]) * inv, (h_a[k] - ol_a[k]) * inv);
        let mut s = -S::ONE;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
            s = S::ONE;
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
    if tmin <= S::from_f64(1e-6) {
        return None;
    }
    let mut nl = [S::ZERO; 3];
    nl[axis] = sign;
    // box → world is rotᵀ.
    let n = b.rot.transpose() * Vec3::new(nl[0], nl[1], nl[2]);
    Some((tmin, n))
}

/// Camera ray for pixel (u, v), world frame.
pub fn primary_ray(pose: &CameraPose, intr: &CameraIntrinsics, u: f64, v: f64) -> (Vec3<f64>, Vec3<f64>) {
    ray::<f64>(pose, intr, u, v)
}

fn ray<S: Scalar>(pose: &CameraPose, intr: &CameraIntrinsics, u: f64, v: f64) -> (Vec3<S>, Vec3<S>) {
    let dir_opt = phyz_math::Vec3::new((u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0);
    let dir = (pose.world_from_optical * dir_opt).normalize();
    (v3(pose.position), v3(dir))
}

/// Render the full frame on f64 to an sRGB image.
pub fn render(scene: &Scene<f64>, pose: &CameraPose, intr: &CameraIntrinsics) -> image::RgbaImage {
    let (w, h) = (intr.width, intr.height);
    let mut img = image::RgbaImage::new(w, h);
    let to8 = |v: f64| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
    for y in 0..h {
        for x in 0..w {
            let (o, d) = ray::<f64>(pose, intr, x as f64 + 0.5, y as f64 + 0.5);
            let c = scene.rgb(o, d);
            img.put_pixel(x, y, image::Rgba([to8(c[0]), to8(c[1]), to8(c[2]), 255]));
        }
    }
    img
}

// ---- the beauty pass -------------------------------------------------------
//
// The caster above is the *derivative*: generic over `tang::Scalar`, one
// bounce, a lamp of finite size, and no Monte Carlo anywhere, because a
// stochastic estimator has no useful dual. The picture the level ships is not
// that. It is `kosm-render`'s path tracer — multiple bounces, importance
// sampling, MIS against real softboxes — over the very same colliders, told to
// it through [`crate::analytic`]. Two renderers, one geometry, and each doing
// the thing it is good at.

/// `kosm-render`'s camera from phyz's pose and intrinsics.
///
/// phyz's optical frame is +z forward, +x right, +y *down*; the tracer's
/// screen basis is +y up, so the up vector is the negated optical y.
pub fn camera(pose: &CameraPose, intr: &CameraIntrinsics) -> kosm_render::Camera {
    let axis = |x: f64, y: f64, z: f64| {
        let v = pose.world_from_optical * phyz_math::Vec3::new(x, y, z);
        kosm_render::Vec3::new(v.x, v.y, v.z)
    };
    let eye = kosm_render::Point3::new(pose.position.x, pose.position.y, pose.position.z);
    let vfov = 2.0 * (0.5 * intr.height as f64 / intr.fy).atan();
    kosm_render::Camera::from_basis(
        eye,
        axis(0.0, 0.0, 1.0),
        axis(1.0, 0.0, 0.0),
        -axis(0.0, 1.0, 0.0),
        vfov.to_degrees(),
        1.0,
    )
}

/// The level as `kosm-render` sees it: the track's colliders and the marble,
/// each an analytic object with its own material, under a studio rig.
pub fn picture(
    track: &phyz_math::SpatialTransform,
    colliders: &[GeomInstance],
    marble: phyz_math::Vec3,
    marble_r: f64,
) -> kosm_render::Scene<Analytic> {
    use std::sync::Arc;
    let track_geom = Analytic::from_colliders(track, colliders);
    // The rig is sized on the track, not on the marble, or the key light ends
    // up inside the plate.
    let mut bounds = kosm_render::Aabb::empty();
    for i in 0..kosm_render::Geometry::len(&track_geom) {
        bounds.include(&kosm_render::Geometry::bounds(&track_geom, i));
    }
    let centre = bounds.center();
    let radius = 0.5
        * ((bounds.max.x - bounds.min.x).powi(2)
            + (bounds.max.y - bounds.min.y).powi(2)
            + (bounds.max.z - bounds.min.z).powi(2))
        .sqrt();
    let objects = vec![
        kosm_render::Object::new(
            Arc::new(kosm_render::Bvh::build(track_geom)),
            // the printed track: a matte, slightly warm plastic
            kosm_render::Pbr::plastic([0.42, 0.40, 0.36], 0.55, 0.0),
        ),
        kosm_render::Object::new(
            Arc::new(kosm_render::Bvh::build(Analytic::ball(marble, marble_r))),
            // the marble: a clearcoated bead, so the rig reads on it
            kosm_render::Pbr::plastic([0.80, 0.82, 0.86], 0.06, 1.0),
        ),
    ];
    kosm_render::Scene {
        objects,
        lights: kosm_render::studio_rig(centre, radius),
        env: kosm_render::Environment::default(),
        ground: None,
        sun: None,
    }
}

/// One beauty frame of the level at `state`, path-traced.
pub fn beauty(
    track: &phyz_math::SpatialTransform,
    colliders: &[GeomInstance],
    marble: phyz_math::Vec3,
    marble_r: f64,
    pose: &CameraPose,
    intr: &CameraIntrinsics,
    spp: u32,
) -> image::RgbaImage {
    let scene = picture(track, colliders, marble, marble_r);
    let cam = camera(pose, intr);
    let opts = kosm_render::PathTraceOptions { spp, ..Default::default() };
    let film = kosm_render::render(&scene, &cam, intr.width, intr.height, &opts);
    let px = film.to_srgb8(0.7, false);
    image::RgbaImage::from_raw(film.width, film.height, px).expect("film is width x height x 4")
}

/// The plate pixels whose primary hit lies within `radius` of `target`
/// (plate frame), each with a Gaussian weight (σ = radius/2) so the objective
/// has a basin as wide as the patch and not a cliff at its rim. Fixed by the
/// static geometry and the camera.
pub fn target_pixels(
    scene: &Scene<f64>,
    pose: &CameraPose,
    intr: &CameraIntrinsics,
    track: &phyz_math::SpatialTransform,
    target: [f64; 2],
    radius: f64,
) -> Vec<(f64, f64, f64)> {
    let mut px = Vec::new();
    let sigma = radius / 2.0;
    for y in 0..intr.height {
        for x in 0..intr.width {
            let (u, v) = (x as f64 + 0.5, y as f64 + 0.5);
            let (o, d) = ray::<f64>(pose, intr, u, v);
            // Plate only, and with the marble out of the way.
            let mut s = scene.clone();
            s.marble = Vec3::new(0.0, 0.0, -10.0);
            if let Some((t, _, _, true)) = s.hit(o, d) {
                let p = track.world_to_body_point(phyz_math::Vec3::new(o.x + d.x * t, o.y + d.y * t, o.z + d.z * t));
                let d = (p.x - target[0]).hypot(p.y - target[1]);
                if d <= radius {
                    px.push((u, v, (-0.5 * (d / sigma).powi(2)).exp()));
                }
            }
        }
    }
    px
}

/// The plate points under the patch pixels, world frame, with weights: the
/// shadow pass evaluates lamp visibility here directly. Unlike the beauty
/// pass it cannot be brightened by the marble's own lit body sitting in the
/// patch, which is what made "darken the patch" reward parking the marble on it.
pub fn patch_points(
    scene: &Scene<f64>,
    pose: &CameraPose,
    intr: &CameraIntrinsics,
    pixels: &[(f64, f64, f64)],
) -> Vec<(Vec3<f64>, f64)> {
    let mut s = scene.clone();
    s.marble = Vec3::new(0.0, 0.0, -10.0);
    pixels
        .iter()
        .filter_map(|&(u, v, w)| {
            let (o, d) = ray::<f64>(pose, intr, u, v);
            s.hit(o, d).map(|(t, n, _, _)| (o + d * t + n * 1e-5, w))
        })
        .collect()
}

/// Weighted mean lamp visibility over plate points: 1 lit, 0 in full shadow.
pub fn patch_visibility<S: Scalar>(scene: &Scene<S>, points: &[(Vec3<f64>, f64)]) -> S {
    let mut sum = S::ZERO;
    let mut wsum = 0.0;
    for &(p, w) in points {
        sum += scene.lamp_visibility(Vec3::new(S::from_f64(p.x), S::from_f64(p.y), S::from_f64(p.z))) * S::from_f64(w);
        wsum += w;
    }
    sum / S::from_f64(wsum.max(1e-12))
}

/// Visibility and ∂visibility/∂marble by three dual passes.
pub fn visibility_and_grad(scene: &Scene<f64>, points: &[(Vec3<f64>, f64)]) -> (f64, [f64; 3]) {
    let j = patch_visibility(scene, points);
    let mut g = [0.0; 3];
    for (k, gk) in g.iter_mut().enumerate() {
        *gk = patch_visibility(&seeded(scene, k), points).dual;
    }
    (j, g)
}

/// The shadow-pass objective: patch visibility with the marble at its final
/// position, gradient by duals, carried to the release point by the adjoint.
pub fn shadow_objective(scene: Scene<f64>, points: Vec<(Vec3<f64>, f64)>) -> FinalStateObjective<'static> {
    let scene: &'static Scene<f64> = Box::leak(Box::new(scene));
    let points: &'static Vec<(Vec3<f64>, f64)> = Box::leak(Box::new(points));
    let at = move |q: &[f64]| {
        let mut s = scene.clone();
        s.marble = Vec3::new(q[POS], q[POS + 1], q[POS + 2]);
        s
    };
    let value: &'static dyn Fn(&[f64], &[f64]) -> f64 = Box::leak(Box::new(move |q: &[f64], _: &[f64]| patch_visibility(&at(q), points)));
    type GradFn = dyn Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>);
    let gradient: &'static GradFn = Box::leak(Box::new(move |q: &[f64], v: &[f64]| {
        let (_, g) = visibility_and_grad(&at(q), points);
        let mut gq = vec![0.0; q.len()];
        gq[POS..POS + 3].copy_from_slice(&g);
        (gq, vec![0.0; v.len()])
    }));
    FinalStateObjective { value, gradient }
}

/// Mean brightness over `pixels` with the marble at `c`, generic so the same
/// evaluation yields the value (f64) or a directional derivative (Dual).
pub fn disc_brightness<S: Scalar>(scene: &Scene<S>, pose: &CameraPose, intr: &CameraIntrinsics, pixels: &[(f64, f64, f64)]) -> S {
    let mut sum = S::ZERO;
    let mut wsum = 0.0;
    for &(u, v, w) in pixels {
        let (o, d) = ray::<S>(pose, intr, u, v);
        sum += scene.shade(o, d).0 * S::from_f64(w);
        wsum += w;
    }
    sum / S::from_f64(wsum.max(1e-12))
}

/// Scene with a Dual marble position seeded along world axis `k`.
fn seeded(scene: &Scene<f64>, k: usize) -> Scene<Dual<f64>> {
    let lift = |v: Vec3<f64>| Vec3::new(Dual::constant(v.x), Dual::constant(v.y), Dual::constant(v.z));
    let liftm = |m: &Mat3<f64>| {
        let f = |i: usize, j: usize| Dual::constant(m[(i, j)]);
        Mat3::new(f(0, 0), f(0, 1), f(0, 2), f(1, 0), f(1, 1), f(1, 2), f(2, 0), f(2, 1), f(2, 2))
    };
    let mut marble = lift(scene.marble);
    let arr = [&mut marble.x, &mut marble.y, &mut marble.z];
    *arr[k] = Dual::new(scene.marble.as_array()[k], 1.0);
    Scene {
        boxes: scene
            .boxes
            .iter()
            .map(|b| OBox { center: lift(b.center), half: lift(b.half), rot: liftm(&b.rot), albedo: Dual::constant(b.albedo) })
            .collect(),
        marble,
        marble_r: Dual::constant(scene.marble_r),
        lamp: lift(scene.lamp),
        lamp_r: Dual::constant(scene.lamp_r),
        lamp_power: Dual::constant(scene.lamp_power),
        ambient: Dual::constant(scene.ambient),
        glass: scene.glass.iter().map(|g| Glass { shape: glass::to_dual(&g.shape), nd: Dual::constant(g.nd) }).collect(),
        plate: scene.plate,
        caustics: scene.caustics.clone(),
    }
}

/// J(marble) = disc brightness, and ∂J/∂marble by three dual passes.
pub fn brightness_and_grad(
    scene: &Scene<f64>,
    pose: &CameraPose,
    intr: &CameraIntrinsics,
    pixels: &[(f64, f64, f64)],
) -> (f64, [f64; 3]) {
    let j = disc_brightness(scene, pose, intr, pixels);
    let mut g = [0.0; 3];
    for (k, gk) in g.iter_mut().enumerate() {
        *gk = disc_brightness(&seeded(scene, k), pose, intr, pixels).dual;
    }
    (j, g)
}

/// The image objective as a `FinalStateObjective`: brightness of the target
/// disc with the marble at its final position, gradient by duals.
pub fn objective(
    scene: Scene<f64>,
    pose: CameraPose,
    intr: CameraIntrinsics,
    pixels: Vec<(f64, f64, f64)>,
    scale: f64,
) -> FinalStateObjective<'static> {
    let scene: &'static Scene<f64> = Box::leak(Box::new(scene));
    let pose: &'static CameraPose = Box::leak(Box::new(pose));
    let intr: &'static CameraIntrinsics = Box::leak(Box::new(intr));
    let pixels: &'static Vec<(f64, f64, f64)> = Box::leak(Box::new(pixels));
    let at = move |q: &[f64]| {
        let mut s = scene.clone();
        s.marble = Vec3::new(q[POS], q[POS + 1], q[POS + 2]);
        s
    };
    let value: &'static dyn Fn(&[f64], &[f64]) -> f64 =
        Box::leak(Box::new(move |q: &[f64], _: &[f64]| scale * disc_brightness(&at(q), pose, intr, pixels)));
    type GradFn = dyn Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>);
    let gradient: &'static GradFn = Box::leak(Box::new(move |q: &[f64], v: &[f64]| {
        let (_, g) = brightness_and_grad(&at(q), pose, intr, pixels);
        let mut gq = vec![0.0; q.len()];
        for k in 0..3 {
            gq[POS + k] = scale * g[k];
        }
        (gq, vec![0.0; v.len()])
    }));
    FinalStateObjective { value, gradient }
}

pub fn out_dir(out: &std::path::Path) -> std::path::PathBuf {
    out.join("frame")
}

/// Paint the marble (and only the marble) into an existing image: pixels whose
/// primary ray hits the sphere take the caster's shading. For a scene whose
/// backdrop is a splat rather than colliders.
pub fn composite_marble(scene: &Scene<f64>, pose: &CameraPose, intr: &CameraIntrinsics, img: &mut image::RgbaImage) -> usize {
    let mut n = 0;
    for y in 0..intr.height {
        for x in 0..intr.width {
            let (o, d) = ray::<f64>(pose, intr, x as f64 + 0.5, y as f64 + 0.5);
            if sphere_hit(scene.marble, scene.marble_r, o, d).is_some() {
                let (l, _) = scene.shade(o, d);
                let g = (l.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
                img.put_pixel(x, y, image::Rgba([g, g, g, 255]));
                n += 1;
            }
        }
    }
    n
}
