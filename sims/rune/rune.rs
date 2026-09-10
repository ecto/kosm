//! The rune: how much of the sun the being is putting through the keyhole.
//!
//! The score is not a trigger volume and it is not a dot product with a
//! marker. It is photon transport: `kosm-render`'s caustic pass fires the
//! sun's photons at the one refractive solid in the scene — the being — and
//! deposits what comes out the far side on whatever diffuse surface it
//! reaches. The rune reads the power that landed inside the aperture disc on
//! the door's face and divides by the power the sun put into the being. A
//! number in `[0, 1]`, and a physical one: move the sun in the document and
//! the solution moves with it.
//!
//! # What the tracer's numbers mean
//!
//! `caustics::trace` shoots the sun as a parallel beam off a disc that just
//! covers the refractor's bounding sphere, so [`CausticMap::emitted_power`]
//! is `irradiance · π·extent²` — the whole disc, misses included — and is
//! *not* the denominator the rune wants. The being intercepts only the part
//! of that disc it actually blocks, so the honest denominator is the sun's
//! irradiance times the capsule's own projected area toward the sun
//! ([`projected_area`]). With that denominator `frac` is the fraction of the
//! light the being caught that it put through the keyhole, it is bounded
//! above by one (Fresnel and total internal reflection keep it well under),
//! and it does not change when the tracer's aiming disc changes.
//!
//! # Geometry, and why the being is not three primitives
//!
//! [`Pieces`] is a small `Geometry` of its own rather than
//! [`kosm_render::Analytic`] because a capsule assembled out of a cylinder
//! and two spheres is *not* a capsule to a dielectric: the cylinder's end
//! discs sit inside the cap spheres, and a photon crossing one would refract
//! out of the glass and back into it at a surface that is not there. So the
//! capsule is solved as one solid, exactly as `glass::Shape::Capsule` solves
//! it — the same body, two tracers, and [`hint`](super::hint) can check one
//! against the other.
//!
//! Metres and radians, z up, as everywhere else in the cove.

use super::being::TILT_MAX;
use std::path::Path;
use std::sync::Arc;

use kosm::scene::MM;
use kosm_render::caustics::{CausticMap, CausticOptions};
use kosm_render::geometry::Geometry;
use kosm_render::math::{Aabb, Dir3, Point2, Point3, Vec3};
use kosm_render::pathtrace::{Environment, Object, Pbr, Scene, Sun};
use kosm_render::{Hit, Ray};
use kosm_render::{Bvh, Frame, Prim};

use super::CoveScene;
use kosm::player::body::{hand_for, parts_for};
use kosm::player::{BodySpec, Part, Pose as LinkPose};
use phyz_math::Mat3;

/// The real sun's angular radius, radians. A wider disc would soften the
/// terminator, which is a nicer picture and a blurrier focus; the rune is
/// scored on the focus, so it gets the real number.
pub const SUN_ANGULAR_RADIUS: f64 = 0.0047;

/// The sun's irradiance for the score. White and unit, because `frac` is a
/// ratio of two powers that both carry it and it cancels; a level that wants
/// watts scales both ends.
pub const SUN_IRRADIANCE: [f32; 3] = [1.0, 1.0, 1.0];

/// Rec. 709 luminance — `kosm-render` weighs a photon's colour this way and
/// so must anything that compares one of its powers with another.
fn luminance(c: [f32; 3]) -> f64 {
    0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64
}

/// The two knobs the player has, and the third the level solves for.
///
/// `x` and `y` are where the being stands on the sand; it always stands *on*
/// it, so there is no z. `tilt` is radians of lean about the being's right
/// vector. The being's facing is toward the door, +y, which makes its right
/// +x and a positive `tilt` a lean *backward*, away from the door and out to
/// sea. Yaw is not a puzzle knob: a capsule is a surface of revolution about
/// its own axis, so turning it about that axis changes nothing the light can
/// see.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    pub tilt: f64,
}

impl Pose {
    /// The pose the document last recorded as solved.
    pub fn solution(scene: &CoveScene) -> Self {
        Self { x: scene.solution_x, y: scene.solution_y, tilt: scene.solution_tilt }
    }

    /// Whether the document has a solution at all: zeros mean "not solved
    /// yet", which is what `scene.rs` ships with.
    pub fn is_solved(&self) -> bool {
        self.x != 0.0 || self.y != 0.0 || self.tilt != 0.0
    }
}

/// The being as a capsule in world metres: the two cap centres and the
/// radius, so a capsule of height `2·being_r` is exactly the sphere.
///
/// It stands on the sand at `(x, y)` with its centre `being_h/2` above it,
/// and its axis is world z leaned by `tilt` about +x.
pub fn being_capsule(scene: &CoveScene, pose: &Pose) -> (Vec3, Vec3, f64) {
    let r = scene.being_r;
    let half = (scene.being_h / 2.0 - r).max(0.0);
    let centre = Vec3::new(pose.x, pose.y, scene.sand_z_at(pose.x, pose.y) + scene.being_h / 2.0);
    let (s, c) = pose.tilt.sin_cos();
    // a rotation about +x sends +z to (0, −sin, cos)
    let axis = Vec3::new(0.0, -s, c);
    (centre - axis * half, centre + axis * half, r)
}

/// The capsule's cross-section as the sun sees it: the swept rectangle plus
/// one whole sphere, because the two caps are one between them.
///
/// This is the score's denominator, times the sun's irradiance.
pub fn projected_area(scene: &CoveScene, pose: &Pose) -> f64 {
    let (a, b, r) = being_capsule(scene, pose);
    let ab = b - a;
    let l = ab.norm();
    let sun = scene.sun_dir();
    let sin = if l > 1e-12 { (ab / l).cross(sun).norm() } else { 0.0 };
    2.0 * r * l * sin + std::f64::consts::PI * r * r
}

// ─── the geometry the photons meet ────────────────────────────────────────

/// One solid of the rune's little scene.
#[derive(Debug, Clone, Copy)]
pub enum Piece {
    /// A `kosm-render` primitive: the door's slab, the sand's plane.
    Prim(Prim),
    /// The being: the segment `a`→`b` swept by `r`, solved as one solid so a
    /// dielectric never meets an interface that is not on its surface.
    Capsule { a: Point3, b: Point3, r: f64 },
    /// The hero's lens: the **intersection of two spheres** of radius `r`
    /// whose centres are `a` and `b`, which is a symmetric biconvex lens
    /// whose rim lies in the plane halfway between them.
    ///
    /// The same arithmetic `sims/rune/hero/kit.rs::lens_mesh` tessellates —
    /// the top cap belongs to the sphere below the centre and the bottom cap
    /// to the one above it — so the scorer and the picture are the same
    /// glass. Solved as one solid for the same reason the capsule is: a
    /// dielectric must never meet an interface that is not on its surface,
    /// and two overlapping spheres would give it four.
    Lens { a: Point3, b: Point3, r: f64 },
}

/// A bag of [`Piece`]s. A `Scene` is generic over one `Geometry`, so the
/// being, the door and the sand are all read through this one.
#[derive(Debug, Clone, Default)]
pub struct Pieces(pub Vec<Piece>);

impl Geometry for Pieces {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn bounds(&self, i: usize) -> Aabb {
        match &self.0[i] {
            Piece::Prim(p) => p.bounds(),
            Piece::Capsule { a, b, r } => {
                let pad = Vec3::new(*r, *r, *r);
                let mut bb = Aabb::empty();
                for c in [*a, *b] {
                    bb.include_point(&(c - pad));
                    bb.include_point(&(c + pad));
                }
                bb
            }
            // The lens lies inside the convex hull of its rim disc and its
            // two poles, so a box that holds both holds it.
            Piece::Lens { a, b, r } => {
                let mid = *a + (*b - *a) * 0.5;
                let half_sep = 0.5 * (*b - *a).norm();
                let u = if half_sep > 1e-12 { (*b - *a) / (2.0 * half_sep) } else { Vec3::new(0.0, 0.0, 1.0) };
                let h = (r * r - half_sep * half_sep).max(0.0).sqrt();
                let bulge = r - half_sep;
                let at = |c: f64| h * (1.0 - c * c).max(0.0).sqrt() + bulge * c.abs();
                let pad = Vec3::new(at(u.x), at(u.y), at(u.z));
                let mut bb = Aabb::empty();
                bb.include_point(&(mid - pad));
                bb.include_point(&(mid + pad));
                bb
            }
        }
    }

    fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit> {
        match &self.0[i] {
            Piece::Prim(p) => p.intersect(ray, i as u32, t_min, t_max),
            Piece::Capsule { a, b, r } => {
                let mut best: Option<(f64, Vec3)> = None;
                capsule_hits(*a, *b, *r, ray, &mut |t, n| {
                    if t > t_min && t < t_max && best.is_none_or(|(bt, _)| t < bt) {
                        best = Some((t, n));
                    }
                });
                best.map(|(t, n)| Hit::new(t, ray.at(t), Dir3::new_normalize(n), Point2::new(0.0, 0.0), i as u32))
            }
            Piece::Lens { a, b, r } => {
                let mut best: Option<(f64, Vec3)> = None;
                lens_hits(*a, *b, *r, ray, &mut |t, n| {
                    if t > t_min && t < t_max && best.is_none_or(|(bt, _)| t < bt) {
                        best = Some((t, n));
                    }
                });
                best.map(|(t, n)| Hit::new(t, ray.at(t), Dir3::new_normalize(n), Point2::new(0.0, 0.0), i as u32))
            }
        }
    }

    fn intersect_all(&self, ray: &Ray, i: usize, out: &mut Vec<Hit>) {
        match &self.0[i] {
            // a cylinder is entered and left by the same ray, so the prims
            // answer this themselves
            Piece::Prim(p) => p.hits(ray, i as u32, 0.0, f64::INFINITY, out),
            Piece::Capsule { a, b, r } => capsule_hits(*a, *b, *r, ray, &mut |t, n| {
                out.push(Hit::new(t, ray.at(t), Dir3::new_normalize(n), Point2::new(0.0, 0.0), i as u32));
            }),
            Piece::Lens { a, b, r } => lens_hits(*a, *b, *r, ray, &mut |t, n| {
                out.push(Hit::new(t, ray.at(t), Dir3::new_normalize(n), Point2::new(0.0, 0.0), i as u32));
            }),
        }
    }
}

/// Where a ray enters and leaves the intersection of two spheres, with the
/// outward normal at each.
///
/// An intersection of convex solids is convex, and a ray meets a convex solid
/// in one interval — so the answer is the intersection of the two spheres'
/// own intervals, and the normal at each end belongs to whichever sphere is
/// the one being crossed there. Two roots or none; never four.
fn lens_hits(a: Point3, b: Point3, r: f64, ray: &Ray, out: &mut impl FnMut(f64, Vec3)) {
    let d = ray.direction.into_inner();
    let span = |centre: Point3| -> Option<(f64, f64)> {
        let oc = ray.origin - centre;
        let bq = oc.dot(d);
        let disc = bq * bq - (oc.dot(oc) - r * r);
        (disc >= 0.0).then(|| {
            let s = disc.sqrt();
            (-bq - s, -bq + s)
        })
    };
    let (Some((a0, a1)), Some((b0, b1))) = (span(a), span(b)) else { return };
    // whichever sphere is entered last, and whichever is left first
    let (enter, on_enter) = if a0 > b0 { (a0, a) } else { (b0, b) };
    let (exit, on_exit) = if a1 < b1 { (a1, a) } else { (b1, b) };
    if !(enter < exit) {
        return;
    }
    out(enter, (ray.at(enter) - on_enter) / r);
    out(exit, (ray.at(exit) - on_exit) / r);
}

/// Every root of the ray against the capsule `a`→`b` of radius `r`, with the
/// outward normal at each, in no particular order.
///
/// Three pieces, each owning the part of the surface the other two do not: a
/// cylinder root counts only while its axial parameter is inside the cap
/// planes, and a cap's root only beyond its own plane. This is
/// `glass::capsule_hit`'s partition, kept for both roots rather than the
/// nearest, because a photon needs the exit as much as the entry.
fn capsule_hits(a: Point3, b: Point3, r: f64, ray: &Ray, out: &mut impl FnMut(f64, Vec3)) {
    let d = ray.direction.into_inner();
    let ba = b - a;
    let baba = ba.dot(ba);
    let m = ray.origin - a;
    let bard = ba.dot(d);
    let baoc = ba.dot(m);

    if baba <= 1e-24 {
        // a segment of no length is the sphere, answered once and not twice
        let bq = m.dot(d);
        let disc = bq * bq - (m.dot(m) - r * r);
        if disc >= 0.0 {
            let ds = disc.sqrt();
            for t in [-bq - ds, -bq + ds] {
                out(t, (m + d * t) / r);
            }
        }
        return;
    }

    // the cylinder between the cap planes, in the axial parametrisation that
    // keeps |ba| out of the square root
    let k2 = baba - bard * bard;
    if k2 > 1e-18 {
        let k1 = baba * m.dot(d) - baoc * bard;
        let k0 = baba * m.dot(m) - baoc * baoc - r * r * baba;
        let h = k1 * k1 - k2 * k0;
        if h >= 0.0 {
            let hs = h.sqrt();
            for t in [(-k1 - hs) / k2, (-k1 + hs) / k2] {
                let y = baoc + t * bard;
                if y > 0.0 && y < baba {
                    out(t, (m + d * t - ba * (y / baba)) / r);
                }
            }
        }
    }

    // the caps, each on its own side of its cap plane
    for (centre, below) in [(a, true), (b, false)] {
        let oc = ray.origin - centre;
        let bq = oc.dot(d);
        let disc = bq * bq - (oc.dot(oc) - r * r);
        if disc < 0.0 {
            continue;
        }
        let ds = disc.sqrt();
        for t in [-bq - ds, -bq + ds] {
            let y = baoc + t * bard;
            if (below && y <= 0.0) || (!below && y >= baba) {
                out(t, (oc + d * t) / r);
            }
        }
    }
}

// ─── the scene the score is traced through ────────────────────────────────

/// The smallest scene the rune needs: the being as glass, the door as an
/// opaque face, and the sand under both.
///
/// The sand is here for the accounting, not the picture. Without it a photon
/// that misses the door flies off into the sky and is simply gone, which
/// costs nothing — but with it the map's `deposited_power` is a number that
/// can be read against `emitted_power`, and a caustic that lands a metre
/// below the keyhole shows up as sand instead of as silence.
pub fn picture(scene: &CoveScene, pose: &Pose) -> Scene<Pieces> {
    let (a, b, r) = being_capsule(scene, pose);
    picture_of(scene, Piece::Capsule { a: Point3::from_vec(a), b: Point3::from_vec(b), r }, &[])
}

/// The same scene with any one refractor in it and any number of opaque
/// bodies beside it: the being's capsule, or the lens the hero is holding and
/// the hero standing behind it.
///
/// The door and the sand do not care which body throws the caustic, and the
/// score does not either — it reads the power inside the aperture disc and
/// divides by the power the sun put into the refractor. So there is one
/// scene, and [`Piece`] is what changes.
///
/// `occluders` is the part of that which is *not* optional once the refractor
/// is small: a lens 220 mm across held beside a head 456 mm across is a lens
/// that its owner can stand in front of, and a score that did not know it
/// would happily solve for a pose whose light never reaches the glass. They
/// are plain diffuse solids — the sun's photons stop there, and nothing is
/// read off them.
pub fn picture_of(scene: &CoveScene, refractor: Piece, occluders: &[Piece]) -> Scene<Pieces> {
    let being = Pieces(vec![refractor]);

    let face = scene.cliff_face_y();
    let sill = scene.door_sill();
    let door = Pieces(vec![Piece::Prim(Prim::Box {
        center: Point3::new(scene.door_x, face + scene.door_t / 2.0, sill + scene.door_h / 2.0),
        half: Vec3::new(scene.door_w / 2.0, scene.door_t / 2.0, scene.door_h / 2.0),
        rot: Frame::identity(),
    })]);

    // A disc and not a `Prim::Plane`: an unbounded plane's bounds are
    // enormous and it is a poor citizen of a BVH. The cove is a square of
    // side `cove`, so a disc of that radius covers it and its apron.
    let n = scene.sand_normal();
    let sand = Pieces(vec![Piece::Prim(Prim::Disc {
        center: Point3::new(0.0, 0.0, scene.sand_z_at(0.0, 0.0)),
        normal: Dir3::new_normalize(Vec3::new(n.x, n.y, n.z)),
        radius: scene.cove,
    })]);

    let sun = Sun::new(scene.sun_dir(), SUN_ANGULAR_RADIUS, SUN_IRRADIANCE);
    // The being's index is the document's `n_d`, flat. A dispersion curve
    // would be more honest to look at and would put a hero wavelength on
    // every photon; over a 0.5 m focal length N-BK7 moves that focus by under
    // two millimetres across the visible, which is nothing against a 120 mm
    // keyhole, and the variance it adds is not nothing against a number a
    // search differentiates. `hint.rs` traces the five bands and says how much
    // it costs.
    let mut objects = Vec::with_capacity(4);
    objects.push(Object::new(Arc::new(Bvh::build(being)), Pbr::glass(scene.n_d as f32, 0.0)));
    objects.push(Object::new(Arc::new(Bvh::build(door)), Pbr::plastic([0.42, 0.41, 0.39], 0.9, 0.0)));
    objects.push(Object::new(Arc::new(Bvh::build(sand)), Pbr::plastic([0.76, 0.70, 0.56], 0.95, 0.0)));
    if !occluders.is_empty() {
        // One BVH for all of them, and a dull matte: what they are for is
        // stopping light, and a photon that ends on cloth ends.
        let body = Pieces(occluders.to_vec());
        objects.push(Object::new(Arc::new(Bvh::build(body)), Pbr::plastic([0.20, 0.22, 0.24], 0.95, 0.0)));
    }
    Scene { objects, lights: Vec::new(), env: Environment::default(), sun: Some(sun), ground: None, splats: None }
}

/// What the door read.
#[derive(Clone, Copy, Debug, Default)]
pub struct Score {
    /// The rune: power through the keyhole over power into the being, in
    /// `[0, 1]`.
    pub frac: f64,
    /// The power inside the aperture disc, in the tracer's own units.
    pub deposited: [f32; 3],
    /// The denominator: `irradiance · projected area`.
    pub incident: f64,
}

/// The caustic map at a pose, and the score read off it.
///
/// Split out from [`score`] because the sweep and the diagnostics want the
/// map as well as the number, and tracing it twice is the expensive half.
pub fn trace(scene: &CoveScene, pose: &Pose, photons: usize) -> (CausticMap, Score) {
    let (a, b, r) = being_capsule(scene, pose);
    read(
        scene,
        Piece::Capsule { a: Point3::from_vec(a), b: Point3::from_vec(b), r },
        &[],
        projected_area(scene, pose),
        photons,
    )
}

/// The rune's score at a pose.
pub fn score(scene: &CoveScene, pose: &Pose, photons: usize) -> Score {
    trace(scene, pose, photons).1
}

// ─── the lens in the hero's hand ───────────────────────────────────────────

/// Where a held lens is: the centre of the glass and its optical axis, world
/// metres. Everything else about it — the surface radius, the separation of
/// the two sphere centres — is `hero/kit.rs`'s and is the same for every lens
/// the hero owns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Held {
    pub centre: Vec3,
    pub axis: Vec3,
}

/// The lens's own numbers in **metres**: the surface radius, half the
/// separation of the two sphere centres, and the semi-diameter of the rim.
///
/// Straight out of [`super::hero::kit::lens_numbers`], which cuts the glass
/// for `LENS_F` by the thick lensmaker's equation. One statement of the
/// shape, two readers: the mesh the picture traces and the piece the score
/// does.
pub fn lens_numbers_m() -> (f64, f64, f64) {
    let (r, a, _) = super::hero::kit::lens_numbers();
    (r * MM, a * MM, 0.5 * super::hero::kit::LENS_D * MM)
}

/// The held lens as a [`Piece`]: two spheres whose intersection is the glass.
///
/// `lens_mesh` builds the cap on `+z` out of the sphere centred at `−a`, so
/// that is the way round the centres go here.
pub fn lens_piece(held: &Held) -> Piece {
    let (r, a, _) = lens_numbers_m();
    let u = if held.axis.norm() > 1e-12 { held.axis.normalize() } else { Vec3::new(0.0, 0.0, 1.0) };
    Piece::Lens {
        a: Point3::from_vec(held.centre - u * a),
        b: Point3::from_vec(held.centre + u * a),
        r,
    }
}

/// The lens's cross-section as the sun sees it: the rim disc foreshortened,
/// plus the knife edge it presents when it is turned edge-on.
///
/// The score's denominator for a held lens, exactly as [`projected_area`] is
/// the being's — the power the sun actually puts into the glass, so `frac`
/// stays the fraction of *caught* light that reached the keyhole and stays
/// comparable with the capsule's number.
pub fn lens_projected_area(scene: &CoveScene, held: &Held) -> f64 {
    let (r, a, h) = lens_numbers_m();
    let u = if held.axis.norm() > 1e-12 { held.axis.normalize() } else { Vec3::new(0.0, 0.0, 1.0) };
    let cos = u.dot(&scene.sun_dir()).abs();
    let thickness = 2.0 * (r - a);
    std::f64::consts::PI * h * h * cos + 2.0 * h * thickness * (1.0 - cos * cos).max(0.0).sqrt()
}

/// The caustic map the held lens throws, and the score read off it, with
/// whatever else is standing in the light.
pub fn trace_lens(scene: &CoveScene, held: &Held, occluders: &[Piece], photons: usize) -> (CausticMap, Score) {
    read(scene, lens_piece(held), occluders, lens_projected_area(scene, held), photons)
}

/// The rune's score with the lens as the refractor and nothing shadowing it.
pub fn score_lens(scene: &CoveScene, held: &Held, photons: usize) -> Score {
    trace_lens(scene, held, &[], photons).1
}

/// The rune's score with the lens as the refractor and a body behind it.
///
/// The one entry point the live gate and the hero's solve both go through:
/// give it a lens somewhere in the world and the opaque solids that can stand
/// between it and the sun, and it answers the same `frac` the capsule's score
/// answers. Where the lens *came from* — a solved [`HeroPose`], or the arm of
/// a body that is being walked about by a player — is the caller's business
/// and not the tracer's.
pub fn score_lens_at(scene: &CoveScene, held: &Held, occluders: &[Piece], photons: usize) -> Score {
    trace_lens(scene, held, occluders, photons).1
}

/// One refractor, one denominator, one reading of the keyhole. What both
/// [`trace`] and [`trace_lens`] are.
fn read(scene: &CoveScene, refractor: Piece, occluders: &[Piece], area: f64, photons: usize) -> (CausticMap, Score) {
    let picture = picture_of(scene, refractor, occluders);
    let map = kosm_render::caustics::trace(
        &picture,
        &CausticOptions { photons, radius: Some(0.02), max_bounces: 8, ..Default::default() },
    );
    let frame = scene.door_frame();
    let deposited = map.power_within(
        Point3::from_vec(frame.origin),
        Vec3::new(frame.normal.x, frame.normal.y, frame.normal.z),
        scene.aperture_r,
    );
    let incident = luminance(SUN_IRRADIANCE) * area;
    let frac = if incident > 0.0 { luminance(deposited) / incident } else { 0.0 };
    (map, Score { frac, deposited, incident })
}

/// Where the caustic actually landed on the door's face, in the frame's
/// `(right, up)` metres, and the power that landed there.
///
/// A diagnostic, not part of the score: the disc the power is read over at
/// each cell is the cell's inscribed circle, so it misses the corners — an
/// even loss over the face, which a centroid does not mind and a total does.
pub fn landing(scene: &CoveScene, map: &CausticMap, cell: f64) -> Option<([f64; 2], f64)> {
    let frame = scene.door_frame();
    let n = Vec3::new(frame.normal.x, frame.normal.y, frame.normal.z);
    let half_w = scene.door_w / 2.0;
    let (lo, hi) = (-scene.aperture_z, scene.door_h - scene.aperture_z);
    let (mut mr, mut mu, mut w) = (0.0, 0.0, 0.0);
    let nr = (2.0 * half_w / cell).ceil() as i64;
    let nu = ((hi - lo) / cell).ceil() as i64;
    for iu in 0..nu {
        for ir in 0..nr {
            let right = -half_w + (ir as f64 + 0.5) * cell;
            let up = lo + (iu as f64 + 0.5) * cell;
            let p = frame.at(right, up);
            let e = luminance(map.power_within(Point3::from_vec(p), n, cell / 2.0));
            mr += right * e;
            mu += up * e;
            w += e;
        }
    }
    (w > 0.0).then(|| ([mr / w, mu / w], w))
}

// ─── the hero, and where it holds the glass ───────────────────────────────

/// The knobs the **hero** has.
///
/// Six, and every one of them is something a person standing on a beach can
/// do: walk somewhere (`x`, `y`), turn (`yaw`), hold the lens up at some lift
/// and swing of the arm (`aim_el`, `aim_az`, measured at the shoulder in the
/// body's own frame), and turn the glass in its fist (`cant`). There is no z
/// — the hero stands *on* the sand — and there is no lean, because a lean is
/// what the walk does to a body and not a thing the puzzle is solved with.
///
/// This is the capsule's [`Pose`] grown legs. Where the capsule's whole body
/// was the lens and a tilt was the only aim it had, the hero's lens is 220 mm
/// of crown glass on the end of a 404 mm arm, and *where the arm puts it* is
/// most of the puzzle: the chief ray through the lens's centre is what
/// decides where the caustic lands, so `aim_el` moves the answer twice as far
/// as walking does. Radians and metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeroPose {
    pub x: f64,
    pub y: f64,
    /// Which way it faces, radians, zero along `+x` — [`kosm::player::Body`]'s
    /// own convention, so a pose read off a body and a pose handed to one are
    /// the same number.
    pub yaw: f64,
    /// The arm's lift and swing at the shoulder, radians, in the body's frame
    /// (`+x` forward, `+y` left, `+z` up). A negative swing is out to the
    /// hero's right, which is the hand the lens is in.
    pub aim_el: f64,
    pub aim_az: f64,
    /// The turn of the wrist: how far the glass is canted off the forearm's
    /// line. See `being::lens_grip`.
    pub cant: f64,
}

impl Default for HeroPose {
    /// Standing at the origin facing `+x`, holding the lens the way
    /// `being.rs` holds it. The knobs the level ships before it is solved.
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            yaw: 0.0,
            aim_el: super::being::LENS_AIM_EL,
            aim_az: super::being::LENS_AIM_AZ,
            cant: 0.0,
        }
    }
}

impl HeroPose {
    /// The pose the document last recorded as solved.
    pub fn solution(scene: &CoveScene) -> Self {
        Self {
            x: scene.hero_x,
            y: scene.hero_y,
            yaw: scene.hero_yaw,
            aim_el: scene.hero_aim_el,
            aim_az: scene.hero_aim_az,
            cant: scene.hero_cant,
        }
    }

    /// Whether the document has a hero solution at all: an unsolved level
    /// ships zeros for where it stands, and nobody solves for the origin.
    pub fn is_solved(&self) -> bool {
        self.x != 0.0 || self.y != 0.0
    }

    /// The same pose, turned to face a point on the sand.
    ///
    /// A hero solving the rune is looking at the door, and the yaw that does
    /// that is a function of where it is standing rather than a seventh
    /// number for a search to wander in.
    pub fn facing(self, at: Vec3) -> Self {
        Self { yaw: (at.y - self.y).atan2(at.x - self.x), ..self }
    }

    /// One knob, by index: 0 x, 1 y, 2 yaw, 3 aim_el, 4 aim_az, 5 cant.
    pub fn knob(&mut self, k: usize) -> &mut f64 {
        match k {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.yaw,
            3 => &mut self.aim_el,
            4 => &mut self.aim_az,
            _ => &mut self.cant,
        }
    }
}

/// The hero's root frame — the pelvis — standing upright on the sand.
///
/// Upright, and that is the one thing this arithmetic asserts that a stepped
/// body does not quite honour: a figure holding an arm up leans a fraction of
/// a degree into it. `rune_tests` measures the gap.
pub fn hero_root(scene: &CoveScene, pose: &HeroPose) -> LinkPose {
    let rig = &*super::being::HERO_RIG;
    let z = scene.sand_z_at(pose.x, pose.y) + rig.foot_drop;
    LinkPose::new(Vec3::new(pose.x, pose.y, z), Mat3::rotation_z(pose.yaw))
}

/// Where the gripping hand is asked to go, world metres: the same point
/// `being::Cove::hold_lens_up` asks the arm for, off the same shoulder.
pub fn hero_aim(scene: &CoveScene, pose: &HeroPose) -> Vec3 {
    let rig = &*super::being::HERO_RIG;
    let spec = &rig.spec;
    let arm = spec.arm.expect("the hero has an arm");
    let out = super::being::LENS_REACH * (arm.upper + arm.lower);
    let local = spec.links[arm.shoulder].pivot + super::being::lens_aim_dir(pose.aim_el, pose.aim_az) * out
        - spec.links[0].pivot;
    let root = hero_root(scene, pose);
    root.pos + root.rot.mul_vec(local)
}

/// Where the hand ends up, world: the two-link solve and the forward
/// kinematics under it, with no dynamics — [`kosm::player::body::hand_for`].
pub fn hero_hand(scene: &CoveScene, pose: &HeroPose) -> LinkPose {
    let rig = &*super::being::HERO_RIG;
    hand_for(&rig.spec, hero_root(scene, pose), hero_aim(scene, pose)).expect("the hero has an arm")
}

/// Every link of the figure, placed at a [`HeroPose`]: the rig at rest with
/// the lens arm solved. What the still draws the solution from.
pub fn hero_parts(scene: &CoveScene, pose: &HeroPose) -> Vec<Part> {
    let rig = &*super::being::HERO_RIG;
    parts_for(&rig.spec, hero_root(scene, pose), Some(hero_aim(scene, pose)))
}

/// The glass, where the hero is holding it: the centre and the optical axis.
///
/// The hand, then the grip. This is the whole of the map from the puzzle's
/// six knobs to the one thing the light cares about, and it is the same two
/// steps [`kosm::player::Body::held`] takes on a body that has been stepped —
/// which is what `rune_tests` holds it to.
pub fn hero_lens(scene: &CoveScene, pose: &HeroPose) -> Held {
    let placed = hero_lens_pose(scene, pose);
    Held { centre: placed.pos, axis: placed.rot.mul_vec(Vec3::z()) }
}

/// The same, as a whole frame: what the *picture* needs, because a disc of
/// glass has a rim as well as an axis. `+z` is the optical axis, which is
/// what `hero/kit.rs` cuts the lens about.
pub fn hero_lens_pose(scene: &CoveScene, pose: &HeroPose) -> LinkPose {
    super::being::lens_grip(pose.cant).then(&hero_hand(scene, pose))
}

/// Which of the hero's links can stand between the sun and the glass: the
/// skirt, the chest and the head.
///
/// The trunk and not the whole figure. An arm that is holding a lens up is
/// beside it and not behind it, the legs are half a metre below the light,
/// and every extra solid is a BVH the photon pass walks a million times. The
/// three that matter are the three that are *big*: on a 1.11 m frame the head
/// alone is 456 mm across.
const SHADES: [&str; 3] = ["pelvis", "torso", "neck"];

/// The hero's trunk as opaque [`Piece`]s, from wherever its links actually
/// are. `parts` is [`kosm::player::Snapshot::parts`], in the spec's link
/// order.
pub fn occluders(spec: &BodySpec, parts: &[Part]) -> Vec<Piece> {
    let mut out = Vec::new();
    for (i, link) in spec.links.iter().enumerate() {
        if !SHADES.contains(&link.name.as_str()) {
            continue;
        }
        let Some(part) = parts.get(i) else { continue };
        for lump in &link.lumps {
            let at = |p: Vec3| part.pose.pos + part.pose.rot.mul_vec(p - link.pivot);
            out.push(Piece::Capsule {
                a: Point3::from_vec(at(lump.from)),
                b: Point3::from_vec(at(lump.to)),
                r: lump.radius,
            });
        }
    }
    out
}

/// The same, for a hero standing at a [`HeroPose`].
pub fn hero_occluders(scene: &CoveScene, pose: &HeroPose) -> Vec<Piece> {
    let rig = &*super::being::HERO_RIG;
    occluders(&rig.spec, &hero_parts(scene, pose))
}

/// The lens's centre in the **hero's own frame**, relative to the point its
/// boots are on: what the arm and the wrist alone decide.
///
/// Read off [`hero_lens`] at the origin facing `+x`, so there is one
/// statement of the kinematics and this is a projection of it rather than a
/// second one. The sand's height cancels: it enters `hero_root` and comes
/// straight back out.
pub fn hero_lens_local(scene: &CoveScene, pose: &HeroPose) -> Vec3 {
    let at_origin = HeroPose { x: 0.0, y: 0.0, yaw: 0.0, ..*pose };
    hero_lens(scene, &at_origin).centre - Vec3::new(0.0, 0.0, scene.sand_z_at(0.0, 0.0))
}

/// Where the hero has to stand for the lens it is holding to put the sun in
/// the keyhole, and which way it has to face. The seed the solve starts from.
///
/// One equation, and it is `hero/mod.rs`'s. An ideal lens images a parallel
/// bundle at the point where its own **chief ray** — the one through the
/// centre of the glass, which is undeviated — crosses the focal plane, and
/// tilting the lens moves that point along the chief ray and nowhere else. So
/// the sun's image lands on the line that leaves the lens's centre along the
/// sunbeam, whatever the glass is doing, and staging this is two lines:
///
/// ```text
/// lens_z + k_z·(lens_y − face_y) = aim_z          k_z = −d_z/d_y, d the sunbeam
/// lens_x − k_x·(lens_y − face_y) = door_x         k_x =  d_x/d_y
/// ```
///
/// with the boots on the sand (`z = sand_z(y)`) and the lens wherever the arm
/// has put it. The first solves for `y` and the second for `x`. Facing
/// depends on position and position on facing — the arm holds the glass out
/// to the hero's right, so turning the hero moves the glass — so it is
/// iterated; six turns is far past convergence for a yaw that only ever moves
/// thirty degrees.
///
/// At the cove's 22° sun and a 1.11 m adventurer with a 404 mm arm this lands
/// the hero about **1.25 m** from the door, which is `hero/mod.rs`'s own
/// number and is not 2.5 m: at 2.5 m the glass would have to be 1.48 m over
/// the sill, 370 mm above the crown of the head holding it. The 2.5 m the
/// brief asked for is in the *lens* instead, where it belongs — `f = 2.5 m`,
/// so at this throw the cone has closed to a hundred-millimetre ellipse
/// beside a 240 mm keyhole.
pub fn hero_doorstep(scene: &CoveScene, hold: &HeroPose) -> HeroPose {
    let d = -scene.sun_dir();
    let face = scene.cliff_face_y();
    let aim_z = scene.door_sill() + scene.aperture_z;
    if d.y.abs() < 1e-9 {
        return *hold;
    }
    let (kx, kz) = (d.x / d.y, -d.z / d.y);
    let local = hero_lens_local(scene, hold);
    let (lx, ly, lz) = (local.x, local.y, local.z);
    let door = scene.door_frame().origin;
    let mut pose = HeroPose { x: scene.door_x, y: face - 1.25, ..*hold };
    for _ in 0..6 {
        let (s, c) = pose.yaw.sin_cos();
        // the lens's own y and z, as functions of where the boots are
        let y = (aim_z - scene.sea_z + scene.beach_slope * scene.waterline() - lz - kz * (lx * s + ly * c - face))
            / (scene.beach_slope + kz);
        let lens_y = y + lx * s + ly * c;
        let x = scene.door_x + kx * (lens_y - face) - (lx * c - ly * s);
        pose = HeroPose { x, y, ..pose }.facing(door);
    }
    pose
}

/// The rune's score for a hero at a pose: the caustic its lens throws, with
/// its own head and trunk in the light.
pub fn score_hero(scene: &CoveScene, pose: &HeroPose, photons: usize) -> Score {
    score_lens_at(scene, &hero_lens(scene, pose), &hero_occluders(scene, pose), photons)
}

/// The caustic map as well as the number, for the diagnostics.
pub fn trace_hero(scene: &CoveScene, pose: &HeroPose, photons: usize) -> (CausticMap, Score) {
    trace_lens(scene, &hero_lens(scene, pose), &hero_occluders(scene, pose), photons)
}

// ─── the solve ────────────────────────────────────────────────────────────

/// The beach the search is allowed to stand on: `±8 m` of the door in x, and
/// from two metres off the waterline to `margin` short of the cliff face.
fn bounds(scene: &CoveScene, margin: f64) -> ([f64; 2], [f64; 2]) {
    (
        [scene.door_x - 8.0, scene.door_x + 8.0],
        [scene.waterline() + 2.0, scene.cliff_face_y() - margin],
    )
}

/// The coarse grid: half-metre steps over the beach, two degrees of tilt,
/// on a cheap photon budget. Returns the best cell and how many cells scored
/// anything at all.
fn grid(scene: &CoveScene, photons: usize, margin: f64, log: bool) -> (Pose, Score, usize) {
    let ([x0, x1], [y0, y1]) = bounds(scene, margin);
    let (mut best, mut best_score) = (Pose::default(), Score::default());
    let mut lit = 0usize;
    let nx = ((x1 - x0) / 0.5).round() as i64;
    let ny = ((y1 - y0) / 0.5).round() as i64;
    for iy in 0..=ny {
        let y = y0 + iy as f64 * 0.5;
        let mut row = 0.0f64;
        for ix in 0..=nx {
            let x = x0 + ix as f64 * 0.5;
            // Only leans the player can hold: the grid stops at the sim's
            // clamp, or the answer is a pose the game itself refuses.
            for it in -5..=5 {
                let tilt = (it as f64 * 2.0f64).to_radians().clamp(-TILT_MAX, TILT_MAX);
                let pose = Pose { x, y, tilt };
                let s = score(scene, &pose, photons);
                if s.frac > 0.0 {
                    lit += 1;
                }
                row = row.max(s.frac);
                if s.frac > best_score.frac {
                    best = pose;
                    best_score = s;
                }
            }
        }
        if log {
            println!("rune grid  y {y:+7.2} m  best in the row {row:.5}  best so far {:.5} at ({:+.2}, {:+.2}) m, {:+.1}°", best_score.frac, best.x, best.y, best.tilt.to_degrees());
        }
    }
    (best, best_score, lit)
}

/// Nelder and Mead's simplex, in as many dimensions as the start has.
///
/// Written here because the objective is a photon count and has no
/// derivative worth the name — `hint.rs` has the derivative, of a different
/// tracer — and because a downhill simplex is thirty lines and not a
/// dependency.
fn nelder_mead(f: &mut impl FnMut(&[f64]) -> f64, start: &[f64], step: &[f64], iters: usize) -> (Vec<f64>, f64) {
    let n = start.len();
    let mut pts: Vec<Vec<f64>> = vec![start.to_vec()];
    for i in 0..n {
        let mut p = start.to_vec();
        p[i] += step[i];
        pts.push(p);
    }
    let mut vals: Vec<f64> = pts.iter().map(|p| f(p)).collect();
    let centroid = |pts: &[Vec<f64>], drop: usize| -> Vec<f64> {
        let mut c = vec![0.0; n];
        for (i, p) in pts.iter().enumerate() {
            if i == drop {
                continue;
            }
            for k in 0..n {
                c[k] += p[k] / n as f64;
            }
        }
        c
    };
    for _ in 0..iters {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|a, b| vals[*a].total_cmp(&vals[*b]));
        let (lo, hi, next) = (order[0], order[n], order[n - 1]);
        // converged when the simplex is smaller than a millimetre in every
        // coordinate it is scaled in
        let spread: f64 = (0..n)
            .map(|k| pts.iter().map(|p| p[k]).fold(f64::NEG_INFINITY, f64::max) - pts.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min))
            .fold(0.0, f64::max);
        if spread < 1e-3 {
            break;
        }
        let c = centroid(&pts, hi);
        let axis: Vec<f64> = (0..n).map(|k| c[k] - pts[hi][k]).collect();
        let at = |t: f64| -> Vec<f64> { (0..n).map(|k| c[k] + t * axis[k]).collect() };
        let (pr, vr) = { let p = at(1.0); let v = f(&p); (p, v) };
        if vr < vals[lo] {
            let pe = at(2.0);
            let ve = f(&pe);
            let (p, v) = if ve < vr { (pe, ve) } else { (pr, vr) };
            pts[hi] = p;
            vals[hi] = v;
        } else if vr < vals[next] {
            pts[hi] = pr;
            vals[hi] = vr;
        } else {
            let pc = at(-0.5);
            let vc = f(&pc);
            if vc < vals[hi] {
                pts[hi] = pc;
                vals[hi] = vc;
            } else {
                // shrink toward the best vertex
                for i in 0..=n {
                    if i == lo {
                        continue;
                    }
                    for k in 0..n {
                        pts[i][k] = pts[lo][k] + 0.5 * (pts[i][k] - pts[lo][k]);
                    }
                    vals[i] = f(&pts[i]);
                }
            }
        }
    }
    let lo = (0..=n).min_by(|a, b| vals[*a].total_cmp(&vals[*b])).unwrap();
    (pts[lo].clone(), vals[lo])
}

/// Tilt in the simplex's coordinates: degrees, so a step of one is the same
/// size of move as a step of one in metres.
const TILT_UNIT: f64 = std::f64::consts::PI / 180.0;

/// Solve for the pose whose caustic falls in the keyhole: a coarse grid over
/// the beach on a cheap budget, then a downhill simplex from the best cell on
/// the full one. Prints as it goes.
///
/// `margin` is how close to the cliff face the search may stand. The plan's
/// grid stops two metres short of it; a glass capsule of radius `r` and index
/// `n` has a line focus `n·r/(2(n−1))` from its own axis — half a metre for
/// the authored being — so [`solve_and_record`] also looks at the band the
/// plan's grid excludes, and says so.
pub fn solve_within(scene: &CoveScene, photons: usize, margin: f64) -> (Pose, Score) {
    let coarse = (photons / 5).max(20_000).min(photons);
    let (seed, seed_score, lit) = grid(scene, 20_000.min(coarse), margin, true);
    println!(
        "rune grid  {lit} cells caught light; best {:.5} at ({:+.3}, {:+.3}) m, {:+.2}°",
        seed_score.frac,
        seed.x,
        seed.y,
        seed.tilt.to_degrees()
    );
    if seed_score.frac <= 0.0 {
        return (seed, seed_score);
    }
    let ([x0, x1], [y0, y1]) = bounds(scene, margin);
    let mut evals = 0usize;
    let mut f = |v: &[f64]| -> f64 {
        evals += 1;
        let pose = Pose { x: v[0].clamp(x0, x1), y: v[1].clamp(y0, y1), tilt: (v[2] * TILT_UNIT).clamp(-TILT_MAX, TILT_MAX) };
        -score(scene, &pose, photons).frac
    };
    let (v, _) = nelder_mead(&mut f, &[seed.x, seed.y, seed.tilt / TILT_UNIT], &[0.25, 0.25, 2.0], 120);
    let pose = Pose { x: v[0].clamp(x0, x1), y: v[1].clamp(y0, y1), tilt: (v[2] * TILT_UNIT).clamp(-TILT_MAX, TILT_MAX) };
    let s = score(scene, &pose, photons);
    println!(
        "rune simplex  {evals} evaluations at {photons} photons: {:.5} at ({:+.3}, {:+.3}) m, {:+.2}°",
        s.frac,
        pose.x,
        pose.y,
        pose.tilt.to_degrees()
    );
    if s.frac >= seed_score.frac { (pose, s) } else { (seed, seed_score) }
}

/// The plan's solve: the grid it specifies, two metres off the cliff.
pub fn solve(scene: &CoveScene, photons: usize) -> (Pose, Score) {
    solve_within(scene, photons, 2.0)
}

// ─── the hero's solve ─────────────────────────────────────────────────────

/// How close to the cliff face the hero's boots may get, metres. Its skirt is
/// 268 mm across and it leans into the door: half a metre is standing at it,
/// not standing in it.
const HERO_MARGIN: f64 = 0.50;

/// Every pose the hero's search may hold, clamped: on the beach, facing the
/// door, with an arm that can only lift so far and a wrist that can only turn
/// so far.
fn hero_clamp(scene: &CoveScene, p: HeroPose) -> HeroPose {
    let door = scene.door_frame().origin;
    HeroPose {
        x: p.x.clamp(scene.door_x - 8.0, scene.door_x + 8.0),
        y: p.y.clamp(scene.waterline() + 2.0, scene.cliff_face_y() - HERO_MARGIN),
        aim_el: p.aim_el.clamp(0.0, 1.40),
        cant: p.cant.clamp(-std::f64::consts::PI, std::f64::consts::PI),
        ..p
    }
    .facing(door)
}

/// How much better than `open_frac` the solve wants the score before it will
/// spend any of it on standing further back, as a multiple.
///
/// `open_frac` is the *door's* threshold — the least light that turns the
/// lock — and a level whose authored answer sits on it is a level that fails
/// the first time a body leans two degrees into the arm it is holding up.
/// Two and a half is the door open with room to spare; past it there is
/// nothing to buy, because light already well inside a 240 mm keyhole does
/// not read as more light. Under it, the score is the only thing that counts.
const HERO_COMFORT: f64 = 2.5;

/// How far off the cliff face is far enough to stop paying for, metres.
///
/// The lens's own focal length. Past it the sun's image is behind the door
/// and the patch opens out again, so there is nothing to buy; short of it
/// there is, and the whole of the puzzle is that the hero has to find the
/// place.
const HERO_STANDOFF_ENOUGH: f64 = 2.5;

/// What the hero's solve maximises: the score up to a gate, and how far from
/// the cliff the hero is standing.
///
/// **Not the score alone**, and the reason is that the cove is a *place*
/// puzzle. The score has a broad plateau — at the doorstep the sun's image is
/// smaller than the keyhole, so once the beam is on it, walking a hand's
/// breadth either way is free — and on a plateau an optimiser goes wherever
/// its last step was pointing. The first solve went to 0.58 m off the face,
/// eighty millimetres from the clamp, which is a hero standing *in* the door
/// rather than at it: nothing to walk to, nothing to line up, and a still
/// with a figure's nose against the stone.
///
/// So the score is capped at [`HERO_COMFORT`] times `open_frac` and the rest
/// of the merit is standoff, at four points of frac a metre up to
/// [`HERO_STANDOFF_ENOUGH`]. Capping and not weighting: a weighted sum trades
/// score for distance at a fixed exchange rate all the way down, and there is
/// no rate at which a shut door is worth a longer walk. A cap says the two
/// things in the right order — under it only the score counts, so the search
/// finds the door first; over it the score is flat and only the walk counts —
/// and it is why the bonus can be small. A tenth of frac is the whole of it,
/// which is under the cap itself, so a pose that does not open the door can
/// never outrank one that does however far down the beach it stands.
///
/// What it is worth on this level: the solve without it stood **0.58 m** off
/// the face, eighty millimetres from the clamp, at frac 0.92; with it, 1.2 m
/// off at frac 0.79. The second is `hero/mod.rs`'s own chief-ray answer to
/// within a hand's breadth, which is the arithmetic and the search agreeing
/// for the first time.
fn hero_merit(scene: &CoveScene, pose: &HeroPose, s: &Score) -> f64 {
    let gate = (scene.open_frac * HERO_COMFORT).min(1.0);
    let standoff = (scene.cliff_face_y() - pose.y).clamp(0.0, HERO_STANDOFF_ENOUGH);
    s.frac.min(gate) + 0.04 * standoff
}

/// Solve for the pose whose lens puts the sun in the keyhole, as far back
/// from it as the light allows.
///
/// Three passes, and the first of them is not a search at all.
///
/// 1. **The chief ray.** [`hero_doorstep`] stages the hero analytically for
///    each way of *holding* the glass — a lens images a parallel bundle where
///    its own undeviated chief ray crosses the focal plane, so given a lift
///    and a cant there is exactly one place on the beach to stand, and it is
///    a division rather than a sweep. The coarse pass is therefore over the
///    two knobs the hero chooses, `aim_el` and `cant`, with the two that
///    follow from them solved. A blind grid over `x` and `y` would spend
///    ninety-nine cells in a hundred confirming that a beam aimed at the sea
///    lands in the sea.
/// 2. **The grid the plan asks for**, around the best of those: `x`, `y`,
///    `aim_el` and `cant` together, on a cheap photon budget. This is what
///    says whether the analytic staging left anything on the table — it is
///    exact for a *thin* lens and the hero's is 4.7 mm thick, and it knows
///    nothing about the hero's own shadow.
/// 3. **Nelder and Mead's simplex** on the full budget, in the same four.
///
/// All three rank on [`hero_merit`] and not on the score: the score is flat
/// over most of the doorstep and would leave the hero wherever the last step
/// happened to point, which on the first pass of this was eighty millimetres
/// from the clamp. What comes out instead is the pose that stands *furthest
/// back* among those that open the door with room to spare — and because the
/// only way to stand further back is to hold the glass higher, the answer is
/// the arm at its limit, which is the same place `hero/mod.rs`'s chief-ray
/// arithmetic put it.
///
/// Prints as it goes. `yaw` is not searched: a hero solving the rune is
/// looking at the door, and [`HeroPose::facing`] is that. `aim_az` is not
/// searched either — it is how a person holds a lens up and out of their own
/// way, not a thing they tune — and it is recorded so that the level can say
/// what it was.
pub fn solve_hero(scene: &CoveScene, photons: usize) -> (HeroPose, Score) {
    let coarse = (photons / 5).max(20_000).min(photons);
    let start = HeroPose { aim_el: scene.hero_aim_el, aim_az: scene.hero_aim_az, ..Default::default() };

    // 1. the chief ray, for every way of holding the glass
    let mut best = hero_clamp(scene, hero_doorstep(scene, &start));
    let mut best_score = score_hero(scene, &best, coarse);
    let mut best_merit = hero_merit(scene, &best, &best_score);
    println!(
        "rune hero  the chief ray stages it {:.3} m off the face at ({:+.3}, {:+.3}) m: {:.5}",
        scene.cliff_face_y() - best.y,
        best.x,
        best.y,
        best_score.frac
    );
    for iel in 0..13 {
        let aim_el = (20.0 + 5.0 * iel as f64).to_radians();
        let (mut row, mut row_off) = (0.0f64, 0.0f64);
        for ic in 0..13 {
            let cant = (15.0 * ic as f64).to_radians();
            let hold = HeroPose { aim_el, cant, ..start };
            let pose = hero_clamp(scene, hero_doorstep(scene, &hold));
            let s = score_hero(scene, &pose, coarse);
            let m = hero_merit(scene, &pose, &s);
            if s.frac > row {
                row = s.frac;
                row_off = scene.cliff_face_y() - pose.y;
            }
            if m > best_merit {
                best = pose;
                best_score = s;
                best_merit = m;
            }
        }
        println!(
            "rune hero  holding {:+5.1}° up: best in the row {row:.5} at {row_off:.2} m off   best so far {:.5} at ({:+.3}, {:+.3}) m — {:.2} m off — {:+.1}° up, {:+.1}° canted",
            aim_el.to_degrees(),
            best_score.frac,
            best.x,
            best.y,
            scene.cliff_face_y() - best.y,
            best.aim_el.to_degrees(),
            best.cant.to_degrees()
        );
    }

    // 2. the grid, in all four, around it
    let seed = best;
    for ix in -3..=3 {
        for iy in -3..=3 {
            for iel in -2..=2 {
                for ic in -2..=2 {
                    let pose = hero_clamp(
                        scene,
                        HeroPose {
                            x: seed.x + 0.25 * ix as f64,
                            y: seed.y + 0.25 * iy as f64,
                            aim_el: seed.aim_el + (4.0 * iel as f64).to_radians(),
                            cant: seed.cant + (12.0 * ic as f64).to_radians(),
                            ..seed
                        },
                    );
                    let s = score_hero(scene, &pose, coarse);
                    let m = hero_merit(scene, &pose, &s);
                    if m > best_merit {
                        best = pose;
                        best_score = s;
                        best_merit = m;
                    }
                }
            }
        }
    }
    println!(
        "rune hero  grid  {:.5} at ({:+.3}, {:+.3}) m — {:.2} m off the face — {:+.1}° up, {:+.1}° canted",
        best_score.frac,
        best.x,
        best.y,
        scene.cliff_face_y() - best.y,
        best.aim_el.to_degrees(),
        best.cant.to_degrees()
    );

    // 3. the simplex, on the full budget. Degrees for the two angles, so a
    // step of one is the same size of move as a step of one metre is not —
    // but is the size a *person* would call small in each.
    let unit = std::f64::consts::PI / 180.0;
    let mut evals = 0usize;
    let read = |v: &[f64]| {
        hero_clamp(scene, HeroPose { x: v[0], y: v[1], aim_el: v[2] * unit, cant: v[3] * unit, ..best })
    };
    let mut f = |v: &[f64]| -> f64 {
        evals += 1;
        let pose = read(v);
        -hero_merit(scene, &pose, &score_hero(scene, &pose, photons))
    };
    let (v, _) = nelder_mead(
        &mut f,
        &[best.x, best.y, best.aim_el / unit, best.cant / unit],
        &[0.15, 0.15, 3.0, 8.0],
        160,
    );
    let pose = read(&v);
    let s = score_hero(scene, &pose, photons);
    println!(
        "rune hero  simplex  {evals} evaluations at {photons} photons: {:.5} at ({:+.3}, {:+.3}) m — {:.2} m off the face — {:+.1}° up, {:+.1}° canted, facing {:+.1}°",
        s.frac,
        pose.x,
        pose.y,
        scene.cliff_face_y() - pose.y,
        pose.aim_el.to_degrees(),
        pose.cant.to_degrees(),
        pose.yaw.to_degrees()
    );
    if hero_merit(scene, &pose, &s) >= best_merit { (pose, s) } else { (best, best_score) }
}

/// Solve for the hero, report, and write the answer out.
///
/// [`solve_and_record`]'s twin, and the same rule: `out/solved/rune.params`
/// gets the knobs, they are printed as the `b.param` lines `scene.rs` should
/// carry, and nothing rewrites Rust behind your back.
pub fn solve_and_record_hero(scene: &CoveScene, photons: usize, out: &Path) -> anyhow::Result<(HeroPose, Score)> {
    let (pose, s) = solve_hero(scene, photons);
    let (map, _) = trace_hero(scene, &pose, photons);
    let held = hero_lens(scene, &pose);
    println!(
        "rune hero  best {:.5} at ({:+.3}, {:+.3}) m — {:.3} m off the cliff face — holding the glass at ({:+.3}, {:+.3}, {:+.3}) m, {:.3} m over its own boots; the sun put {:.4} W into the lens and {:.4} W landed in the keyhole",
        s.frac,
        pose.x,
        pose.y,
        scene.cliff_face_y() - pose.y,
        held.centre.x,
        held.centre.y,
        held.centre.z,
        held.centre.z - scene.sand_z_at(pose.x, pose.y),
        s.incident,
        luminance(s.deposited)
    );
    match landing(scene, &map, 0.05) {
        Some(([r, u], w)) => println!(
            "rune hero  the caustic's centroid on the door face is {r:+.3} m across and {u:+.3} m above the keyhole, carrying {w:.4} W"
        ),
        None => println!("rune hero  nothing reached the door's face at all"),
    }

    let dir = out.join("solved");
    std::fs::create_dir_all(&dir)?;
    let knobs = [
        ("hero_x_mm", pose.x / MM),
        ("hero_y_mm", pose.y / MM),
        ("hero_yaw_deg", pose.yaw.to_degrees()),
        ("hero_aim_el_deg", pose.aim_el.to_degrees()),
        ("hero_aim_az_deg", pose.aim_az.to_degrees()),
        ("hero_cant_deg", pose.cant.to_degrees()),
    ];
    let path = dir.join("rune.params");
    let mut text = String::from(
        "# the hero's solved pose, from `kosm run rune`.
# these are the defaults of `scene.rs`'s `hero_*` knobs.
",
    );
    for (name, value) in &knobs {
        text.push_str(&format!("{name} = {value:.4}
"));
    }
    std::fs::write(&path, &text)?;
    println!("rune hero  wrote {}", path.display());

    if s.frac >= scene.open_frac {
        println!("rune hero  the pose opens the door; `scene.rs` should read");
        for (name, value) in &knobs {
            println!("rune hero      b.param({name:?}, {value:.4});");
        }
    } else {
        println!(
            "rune hero  scene.rs keeps whatever it has: the best pose scores {:.5} and the door wants {:.3}",
            s.frac, scene.open_frac
        );
    }
    Ok((pose, s))
}

/// Solve, report, and write the answer out.
///
/// `out/solved/rune.params` always gets the solved knobs, the way the marble
/// writes its solved tilt, and they are printed as the three `b.param` lines
/// `scene.rs` would carry. `scene.rs` is not rewritten: the level is Rust
/// now, and a solver that edits its own source is a solver you cannot read a
/// diff of. Whether the pose is worth pasting across is said out loud —
/// recording a pose that does not open the door would be recording a wrong
/// answer as the right one.
pub fn solve_and_record(scene: &CoveScene, photons: usize, out: &Path) -> anyhow::Result<(Pose, Score)> {
    let (mut pose, best) = solve(scene, photons);

    // The plan's grid stops two metres short of the cliff, and the being's
    // own focal length is shorter than that. If the plan's band found
    // nothing, look at the band it excluded before calling the level
    // unsolvable — the answer is a level knob either way, but which knob
    // depends on whether the caustic can reach the face at all.
    if best.frac < scene.open_frac {
        let f = scene.n_d * scene.being_r / (2.0 * (scene.n_d - 1.0));
        println!(
            "rune  the plan's grid stops 2 m off the cliff and the being's line focus is {f:.3} m from its own axis; sweeping the band the grid excludes"
        );
        let (near_pose, near) = solve_within(scene, photons, scene.being_r + 0.05);
        if near.frac > best.frac {
            pose = near_pose;
        }
    }

    let (map, s) = trace(scene, &pose, photons);
    println!(
        "rune  best {:.5} at ({:+.3}, {:+.3}) m, {:+.2}°; the sun put {:.4} W into the being and {:.4} W landed in the keyhole",
        s.frac,
        pose.x,
        pose.y,
        pose.tilt.to_degrees(),
        s.incident,
        luminance(s.deposited)
    );
    println!(
        "rune  the pass emitted {:?} and deposited {:?}; the keyhole is at {:.2} m above the door's sill",
        map.emitted_power(),
        map.deposited_power(),
        scene.aperture_z
    );
    match landing(scene, &map, 0.05) {
        Some(([r, u], w)) => println!(
            "rune  the caustic's centroid on the door face is {r:+.3} m across and {u:+.3} m above the keyhole ({:.3} m above the sill), carrying {w:.4} W",
            scene.aperture_z + u
        ),
        None => println!("rune  nothing reached the door's face at all"),
    }

    // The level is Rust now, not a `.loon` a solver can rewrite, so the
    // answer is written where a run's outputs go and printed here rather
    // than edited back into the source. `out/solved/rune.params` is one
    // `name = value` a line in the document's own units — millimetres and
    // degrees — and `scene.rs` carries the same three numbers as the
    // defaults of its `solution_*` knobs. Paste them across to move the
    // level; nothing rewrites Rust behind your back.
    let dir = out.join("solved");
    std::fs::create_dir_all(&dir)?;
    let knobs = [
        ("solution_x_mm", pose.x / MM),
        ("solution_y_mm", pose.y / MM),
        ("solution_tilt_deg", pose.tilt.to_degrees()),
    ];
    let mut text = String::from("# the rune's solved pose, from `kosm run rune`.\n# these are the defaults of `scene.rs`'s `solution_*` knobs.\n");
    for (name, value) in &knobs {
        text.push_str(&format!("{name} = {value:.4}\n"));
    }
    let path = dir.join("rune.params");
    std::fs::write(&path, &text)?;
    println!("rune  wrote {}", path.display());

    if s.frac >= scene.open_frac {
        println!("rune  the pose opens the door; `scene.rs` should read");
        for (name, value) in &knobs {
            println!("rune      b.param({name:?}, {value:.4});");
        }
    } else {
        println!(
            "rune  scene.rs keeps whatever it has: the best pose scores {:.5} and the door wants {:.3}. That is a level bug, not a solver one — the numbers above say which knob",
            s.frac, scene.open_frac
        );
    }
    Ok((pose, s))
}
