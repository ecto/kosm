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

use kosm_render::caustics::{CausticMap, CausticOptions};
use kosm_render::geometry::Geometry;
use kosm_render::math::{Aabb, Dir3, Point2, Point3, Vec3};
use kosm_render::pathtrace::{Environment, Object, Pbr, Scene, Sun};
use kosm_render::{Hit, Ray};
use kosm_render::{Bvh, Frame, Prim};

use super::CoveScene;

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
    /// yet", which is what `levels/cove.loon` ships with.
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
        }
    }
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
    let being = Pieces(vec![Piece::Capsule { a: Point3::from_vec(a), b: Point3::from_vec(b), r }]);

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
    Scene {
        objects: vec![
            // The being's index is the document's `n_d`, flat. A dispersion
            // curve would be more honest to look at and would put a hero
            // wavelength on every photon; over a 0.5 m focal length N-BK7
            // moves that focus by under two millimetres across the visible,
            // which is nothing against a 120 mm keyhole, and the variance it
            // adds is not nothing against a number a search differentiates.
            // `hint.rs` traces the five bands and says how much it costs.
            Object::new(Arc::new(Bvh::build(being)), Pbr::glass(scene.n_d as f32, 0.0)),
            Object::new(Arc::new(Bvh::build(door)), Pbr::plastic([0.42, 0.41, 0.39], 0.9, 0.0)),
            Object::new(Arc::new(Bvh::build(sand)), Pbr::plastic([0.76, 0.70, 0.56], 0.95, 0.0)),
        ],
        lights: Vec::new(),
        env: Environment::default(),
        sun: Some(sun),
        ground: None,
        splats: None,
    }
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
    let picture = picture(scene, pose);
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
    let incident = luminance(SUN_IRRADIANCE) * projected_area(scene, pose);
    let frac = if incident > 0.0 { luminance(deposited) / incident } else { 0.0 };
    (map, Score { frac, deposited, incident })
}

/// The rune's score at a pose.
pub fn score(scene: &CoveScene, pose: &Pose, photons: usize) -> Score {
    trace(scene, pose, photons).1
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

/// Solve, report, and write the answer back into the document.
///
/// `out/solved/cove.loon` always gets the solved knobs, the way the marble
/// writes its solved tilt. `levels/cove.loon` itself only gets them when the
/// solve actually reached `open_frac`: the solution is a solved parameter,
/// and recording a pose that does not open the door would be recording a
/// wrong answer as the right one.
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

    let dir = out.join("solved");
    std::fs::create_dir_all(&dir)?;
    let knobs = [
        ("solution_x_mm", pose.x / crate::scene::MM),
        ("solution_y_mm", pose.y / crate::scene::MM),
        ("solution_tilt_deg", pose.tilt.to_degrees()),
    ];
    let text = scene.authored.with_parameters(&knobs);
    std::fs::write(dir.join("cove.loon"), &text)?;
    println!("rune  wrote {}", dir.join("cove.loon").display());

    if s.frac >= scene.open_frac {
        std::fs::write(scene.authored.path(), &text)?;
        println!("rune  wrote the solution back into {}", scene.authored.path().display());
    } else {
        println!(
            "rune  {} keeps its zeros: the best pose scores {:.5} and the door wants {:.3}. That is a level bug, not a solver one — the numbers above say which knob",
            scene.authored.path().display(),
            s.frac,
            scene.open_frac
        );
    }
    Ok((pose, s))
}
