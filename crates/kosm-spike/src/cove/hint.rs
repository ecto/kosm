//! The hint: the same rune, differentiated.
//!
//! [`rune`](super::rune) scores the being by firing photons at it and asking
//! the door what landed. That is the right answer and it has no derivative:
//! a photon count is a random variable and `power_within` is a hard disc.
//! This module scores the *same* body with `light.rs`'s deterministic lattice
//! tracer — five spectral bands from a lamp, through the one solid, onto a
//! receiving plane — which is written once over `tang::Scalar`. Run it on
//! `f64` and it is the score again, to within the two tracers' different
//! discretisations. Run it on `Dual` and the score's derivative in the
//! being's `x`, `y` and tilt falls out of the same pass, one seed at a time.
//!
//! That gradient is the level's hint (the aperture's rim glows, a glint
//! appears a step along it) and the author's solvability check, and it is the
//! same object in both roles.
//!
//! # The conventions this has to get right
//!
//! `light::trace_onto` moves the lamp and the glass into the receiver's frame
//! and then *is* the plate case, so the receiver has to be orthonormal and
//! right-handed with `normal = u × v`, and the light has to arrive travelling
//! against the normal. The door's frame is `u = +x`, `v = +z`,
//! `normal = -y`, and `x̂ × ẑ = -ŷ`: right-handed, and the sun's light
//! travels into the face. The grid the trace returns is indexed in the
//! receiver's `(u, v)` with the aperture's centre at `(0, 0)`, which is what
//! makes "within `aperture_r` of the origin" the keyhole and not an offset
//! from it.
//!
//! # The lamp does not move
//!
//! The sun is a point lamp a kilometre out along `sun_dir`, anchored on the
//! *door*, not on the being. A lamp placed relative to the being would move
//! when the being does, and then `∂score/∂x` would be the derivative of
//! walking the sun across the sky along with the player — not a hint, and not
//! what the photon pass computes either.
//!
//! # What the two tracers do not share
//!
//! `light.rs` writes irradiance into its grid with the receiving plane's
//! obliquity already folded in (`cos_plate`), because that is what a plate
//! wants to be shaded with. A *power* through a keyhole does not want it, so
//! it is divided back out here at the nominal angle between the sun and the
//! door's normal — every ray arrives within a few degrees of that, the sun
//! being a point a kilometre away. [`crate::cove::rune_tests`] checks the two
//! frac numbers against each other and prints both.

use tang::{Dual, Scalar, Vec3};

use super::CoveScene;
use super::rune::Pose;
use crate::glass::{self, Shape};
use crate::light::{self, BANDS_LEN, Caustic, Receiver};

/// The lamp's distance: far enough that the cone reaching the being is
/// parallel to a part in ten thousand, near enough that `1/D²` is a number
/// and not a limit.
pub const LAMP_DISTANCE: f64 = 1_000.0;

/// The cell budget's extent on the door's face, metres, and the cells across
/// it: together they set the grid's *cell*, five millimetres, which is what
/// they are really for. The grid itself grows past this to hold light that
/// lands wide, keeping the cell size, and `light.rs` caps it at 720 cells a
/// side.
///
/// The plan asked for a two-metre window in two hundred cells. Five
/// millimetres and not ten because the grid is not anchored on the keyhole:
/// `light.rs` centres it on where the light lands, so it slides sub-cell as
/// the being moves and the deposit re-quantises under it. That re-quantising
/// is the whole of the difference between this score's derivative and its
/// central differences, and it falls as the square of the cell.
pub const WINDOW: f64 = 1.0;

/// Cells across [`WINDOW`]; see there.
pub const CELLS: usize = 200;

/// Rays per band on the lattice the gradient is taken on.
pub const RAYS: usize = 400 * 400;

/// Rays per band for the solvability sweep, which runs the objective tens of
/// thousands of times and only needs to know which way is up.
pub const SWEEP_RAYS: usize = 64 * 64;

/// How many cells wide the keyhole's rim is softened over; see [`through`].
const RIM_CELLS: f64 = 12.0;

/// The being's pose with a scalar that may carry a derivative.
#[derive(Clone, Copy, Debug)]
pub struct Knobs<S: Scalar> {
    pub x: S,
    pub y: S,
    pub tilt: S,
}

impl Knobs<f64> {
    pub fn of(pose: &Pose) -> Self {
        Self { x: pose.x, y: pose.y, tilt: pose.tilt }
    }

    /// The same pose with one knob seeded: 0 is x, 1 is y, 2 is tilt.
    pub fn seed(&self, k: usize) -> Knobs<Dual<f64>> {
        let d = |i: usize, v: f64| if i == k { Dual::new(v, 1.0) } else { Dual::constant(v) };
        Knobs { x: d(0, self.x), y: d(1, self.y), tilt: d(2, self.tilt) }
    }
}

/// The sun as a lamp: a point a kilometre out along `sun_dir`, anchored on
/// the door so it stays put while the being walks.
pub fn lamp(scene: &CoveScene) -> Vec3<f64> {
    scene.door_frame().origin + scene.sun_dir() * LAMP_DISTANCE
}

/// The door's face as a receiver: the aperture's centre, the normal out into
/// the cove, and the face's own axes. Right-handed, `normal = u × v`.
pub fn receiver(scene: &CoveScene) -> Receiver {
    let f = scene.door_frame();
    Receiver { origin: f.origin, normal: f.normal, u: f.right, v: f.up }
}

/// The being as `glass.rs` sees it, on whatever scalar the caller is using.
/// The same solid `rune::being_capsule` hands the photon tracer: the cap
/// centres and the radius, standing on the sand at `(x, y)`, leaned by `tilt`
/// about +x.
pub fn capsule<S: Scalar>(scene: &CoveScene, k: Knobs<S>) -> Shape<S> {
    let r = S::from_f64(scene.being_r);
    let half = S::from_f64((scene.being_h / 2.0 - scene.being_r).max(0.0));
    let z = S::from_f64(scene.sea_z)
        + S::from_f64(scene.beach_slope) * (k.y - S::from_f64(scene.waterline()))
        + S::from_f64(scene.being_h / 2.0);
    let centre = Vec3::new(k.x, k.y, z);
    let (s, c) = k.tilt.sin_cos();
    let axis = Vec3::new(S::ZERO, -s, c);
    Shape::Capsule { a: centre - axis * half, b: centre + axis * half, r }
}

/// The power the lamp puts into the being: irradiance `1/D²` at the being
/// times the capsule's cross-section toward the lamp.
///
/// The same denominator [`rune::score`](super::rune::score) uses, in the
/// point lamp's units, and generic because `∂frac/∂tilt` has a term in it:
/// leaning the being changes what it catches as well as where it throws it.
fn incident<S: Scalar>(scene: &CoveScene, k: Knobs<S>) -> S {
    let (a, b, r) = match capsule(scene, k) {
        Shape::Capsule { a, b, r } => (a, b, r),
        _ => unreachable!("the being is a capsule"),
    };
    let l0 = lamp(scene);
    let lamp_s = Vec3::new(S::from_f64(l0.x), S::from_f64(l0.y), S::from_f64(l0.z));
    let centre = (a + b) * S::HALF;
    let to = lamp_s - centre;
    let d2 = to.norm_sq();
    let w = to / d2.sqrt();
    let ab = b - a;
    let len = ab.norm();
    let sin = if len.to_f64() > 1e-12 { (ab / len).cross(&w).norm() } else { S::ZERO };
    (r * len * S::TWO * sin + r * r * S::from_f64(std::f64::consts::PI)) / d2
}

/// How obliquely the sun meets the door. `light.rs`'s grid carries this
/// factor and a power does not; see the module docs.
fn obliquity(scene: &CoveScene) -> f64 {
    let f = scene.door_frame();
    scene.sun_dir().dot(&f.normal).abs().max(1e-6)
}

/// Trace the being's caustic onto the door's face.
pub fn caustic<S: Scalar>(scene: &CoveScene, k: Knobs<S>, rays_per_band: usize) -> Caustic<S> {
    light::trace_onto(
        lamp(scene),
        &capsule(scene, k),
        S::from_f64(scene.n_d),
        receiver(scene),
        WINDOW,
        CELLS,
        rays_per_band,
    )
}

/// The power in the caustic's cells whose centres lie within `aperture_r` of
/// the receiver's origin — the keyhole — over the power the lamp put into the
/// being. The rune's score, on whatever scalar the caller brought.
pub fn score_of<S: Scalar>(scene: &CoveScene, k: Knobs<S>, rays_per_band: usize) -> S {
    let c = caustic(scene, k, rays_per_band);
    through(&c, scene.aperture_r) / (incident(scene, k) * S::from_f64(obliquity(scene) * BANDS_LEN as f64))
}

/// The power the grid holds inside a disc of radius `r` about its origin,
/// summed over the bands. Each band carries the whole lamp, so the caller
/// divides by the band count.
///
/// The rim is soft over [`RIM_CELLS`] cells, and that is not a cosmetic
/// choice. `light.rs` centres its grid on where the light *lands*, so the
/// grid slides as the being moves; a hard `d < r` test on cell centres then
/// admits and drops whole rings of cells as the phase changes, and the score
/// becomes a staircase whose steps are as large as the derivative being
/// measured. A window that is smooth over several cells makes this a
/// midpoint rule with an error in `cell²` instead, and the score a smooth
/// function of the pose — which is the difference between a dual that agrees
/// with central differences and one that does not. The window is centred on
/// `r`, so it passes the same power a hard disc would through a caustic that
/// is flat across the rim.
///
/// Only the cells the window can reach are visited. Far out on the beach the
/// grid is the full 720 cells a side and the keyhole is a few dozen of them;
/// walking all half-million would make the solvability sweep an hour.
fn through<S: Scalar>(c: &Caustic<S>, r: f64) -> S {
    let reach = r + 0.5 * RIM_CELLS * c.cell;
    let span = |o: f64| {
        let lo = ((-reach - o) / c.cell - 0.5).floor().max(0.0) as usize;
        let hi = (((reach - o) / c.cell - 0.5).ceil() + 1.0).max(0.0) as usize;
        lo..hi.min(c.n)
    };
    let area = S::from_f64(c.cell * c.cell);
    let mut sum = S::ZERO;
    for iy in span(c.origin[1]) {
        let v = c.origin[1] + (iy as f64 + 0.5) * c.cell;
        for ix in span(c.origin[0]) {
            let u = c.origin[0] + (ix as f64 + 0.5) * c.cell;
            let w = ((r - (u * u + v * v).sqrt()) / (RIM_CELLS * c.cell) + 0.5).clamp(0.0, 1.0);
            let w = w * w * (3.0 - 2.0 * w);
            if w <= 0.0 {
                continue;
            }
            let w = S::from_f64(w);
            for b in 0..BANDS_LEN {
                sum += c.e[b][iy * c.n + ix] * w;
            }
        }
    }
    sum * area
}

/// The caustic's power-weighted centroid on the door's face, in the
/// receiver's `(u, v)` metres, and the power it carries.
///
/// `None` when the trace caught nothing at all — which happens when the being
/// is not between the sun and the door.
pub fn centroid(c: &Caustic<f64>) -> Option<([f64; 2], f64)> {
    let (mut mu, mut mv, mut w) = (0.0, 0.0, 0.0);
    // band by band and straight down the row, because the grid is mostly
    // empty and this is the one loop the sweep runs a hundred thousand times
    for b in 0..BANDS_LEN {
        for (i, &e) in c.e[b].iter().enumerate() {
            if e == 0.0 {
                continue;
            }
            mu += (c.origin[0] + (i % c.n) as f64 * c.cell + 0.5 * c.cell) * e;
            mv += (c.origin[1] + (i / c.n) as f64 * c.cell + 0.5 * c.cell) * e;
            w += e;
        }
    }
    (w > 0.0).then(|| ([mu / w, mv / w], w * c.cell * c.cell / BANDS_LEN as f64))
}

/// The rune's score at a pose, on `f64`, through the lattice tracer.
pub fn score(scene: &CoveScene, pose: &Pose) -> f64 {
    score_of(scene, Knobs::of(pose), RAYS)
}

/// The score with one knob seeded: its value and its derivative in that knob.
pub fn score_dual(scene: &CoveScene, k: Knobs<Dual<f64>>) -> Dual<f64> {
    score_of(scene, k, RAYS)
}

/// `∂frac/∂(x, y, tilt)`, three dual passes. Metres and radians.
pub fn gradient(scene: &CoveScene, pose: &Pose) -> [f64; 3] {
    let k = Knobs::of(pose);
    [0, 1, 2].map(|i| score_dual(scene, k.seed(i)).dual)
}

// ─── the ascent, and the glint ────────────────────────────────────────────

/// Where the being's beam lands on the door's *plane*, in the receiver's
/// `(u, v)` metres — the median of a coarse refracted fan.
///
/// The caustic grid cannot answer this. `light.rs` centres its grid on where
/// the light lands and caps it at 720 cells, so a being twenty metres down
/// the beach throws its light a hundred metres along the cliff and the grid
/// holds a three-metre window of it, often an empty one. The glint needs a
/// number that is defined from anywhere on the sand, and this is the cheapest
/// honest one: the same cone of rays, refracted once through the same solid,
/// and the *median* landing point, because the rays that graze the capsule's
/// silhouette fly off at any angle at all and a mean is theirs and not the
/// beam's.
///
/// `None` when nothing got through — the being edge-on to nothing, or the
/// beam turned away from the plane.
pub fn beam_centre(scene: &CoveScene, pose: &Pose) -> Option<[f64; 2]> {
    const SIDE: usize = 48;
    let recv = receiver(scene);
    let shape = capsule::<f64>(scene, Knobs::of(pose))
        .to_frame(recv.origin, recv.u, recv.v, recv.normal);
    let l = lamp(scene);
    let lamp_p = Vec3::new(
        recv.u.dot(&(l - recv.origin)),
        recv.v.dot(&(l - recv.origin)),
        recv.normal.dot(&(l - recv.origin)),
    );
    let (centre, bound) = shape.bounds();
    let to = centre - lamp_p;
    let dist = to.norm();
    let axis = to / dist;
    let e1 = axis.cross(&if axis.x.abs() < 0.9 { Vec3::x() } else { Vec3::y() }).normalize();
    let e2 = axis.cross(&e1);
    let ang = (bound / dist).min(0.999).asin();
    let n_glass = light::index::<f64>(scene.n_d, light::D_LINE_UM);
    let (mut us, mut vs) = (Vec::new(), Vec::new());
    for i in 0..SIDE {
        for j in 0..SIDE {
            let a = (i as f64 + 0.5) / SIDE as f64 * 2.0 - 1.0;
            let c = (j as f64 + 0.5) / SIDE as f64 * 2.0 - 1.0;
            let rr = a * a + c * c;
            if rr > 1.0 {
                continue;
            }
            let theta = ang * rr.sqrt();
            let phi = c.atan2(a);
            let d = axis * theta.cos() + (e1 * phi.cos() + e2 * phi.sin()) * theta.sin();
            let Some((t1, n1)) = shape.enter(lamp_p, d) else { continue };
            let p1 = lamp_p + d * t1;
            let Some((d_in, _, _)) = glass::refract(d, n1, 1.0, n_glass) else { continue };
            let Some((p2, d_out, _)) = glass::walk_inside(&shape, p1, d_in, n_glass, 0.0, 8) else { continue };
            if d_out.z >= 0.0 {
                continue;
            }
            let hit = p2 + d_out * (-p2.z / d_out.z);
            us.push(hit.x);
            vs.push(hit.y);
        }
    }
    if us.len() < 8 {
        return None;
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    Some([median(us), median(vs)])
}

/// Past this, the keyhole is outside the caustic grid `light.rs` would build
/// and the score is exactly zero — so the sweep does not pay for the trace.
const BEAM_REACH: f64 = 3.0;

/// What the ascent climbs: the score, plus a small pull toward the keyhole.
///
/// The score alone is flat over most of the beach — a caustic thrown ten
/// metres wide of the door deposits exactly nothing in the aperture, and a
/// gradient of nothing points nowhere. The design's hint is a glint on the
/// sand a step along the gradient, so the objective the glint follows is the
/// one that is defined everywhere: the score, less a thousandth of the
/// distance from where the beam lands on the door's plane to where the
/// keyhole is. Far away that term is the whole objective and it says "bring
/// your light to the door"; close in it is worth a millimetre of score and
/// the focus takes over. The glint follows *this*, not the bare score, and
/// the sweep says which spawns needed it.
///
/// `None` when nothing at all reached the receiving plane.
pub fn guided(scene: &CoveScene, pose: &Pose, rays_per_band: usize) -> Option<f64> {
    let [u, v] = beam_centre(scene, pose)?;
    let miss = (u * u + v * v).sqrt();
    let frac = if miss > BEAM_REACH { 0.0 } else { score_of(scene, Knobs::of(pose), rays_per_band) };
    Some(frac - 1e-3 * miss)
}

/// The guided objective's gradient by central differences, in the same
/// `(x, y, tilt)` the score's is in.
///
/// Differences and not duals: the beam's median landing point is a rank
/// statistic and has no dual worth the name, and a step of two centimetres is
/// the scale the being walks in anyway. The *score's* own gradient is
/// [`gradient`], and that one is exact.
pub fn guided_gradient(scene: &CoveScene, pose: &Pose, rays_per_band: usize) -> Option<[f64; 3]> {
    let h = [0.02, 0.02, 0.2f64.to_radians()];
    let mut g = [0.0; 3];
    for i in 0..3 {
        let (mut lo, mut hi) = (*pose, *pose);
        for (p, sgn) in [(&mut lo, -1.0), (&mut hi, 1.0)] {
            match i {
                0 => p.x += sgn * h[i],
                1 => p.y += sgn * h[i],
                _ => p.tilt += sgn * h[i],
            }
        }
        g[i] = (guided(scene, &hi, rays_per_band)? - guided(scene, &lo, rays_per_band)?) / (2.0 * h[i]);
    }
    Some(g)
}

/// One spawn's climb.
#[derive(Clone, Copy, Debug)]
pub struct Climb {
    pub from: Pose,
    pub to: Pose,
    /// The best `frac` the climb reached.
    pub best: f64,
    /// Steps taken, and metres walked.
    pub steps: usize,
    pub path: f64,
    /// Did it reach `open_frac`?
    pub solved: bool,
}

/// Gradient ascent on the guided objective from one spawn: at most 0.2 m a
/// step in the horizontal plane and half a degree of tilt, until the score
/// passes `open_frac` or the steps run out.
///
/// The three knobs are put in one space where a unit is 0.2 m of walking or
/// half a degree of lean, the direction is the normalised gradient there, and
/// the step is backtracked until it improves. Without the backtracking the
/// climb arrives at the door and then paces across the answer: the focus is
/// tighter than the step, and a fixed stride cannot stand still. Without the
/// common space the tilt is a sign and nothing else, and it walks itself into
/// its own limit while the position is still converging.
pub fn ascend(scene: &CoveScene, from: Pose, steps: usize, rays_per_band: usize) -> Climb {
    const STEP_M: f64 = 0.2;
    let step_rad = 0.5f64.to_radians();
    let x_lo = scene.door_x - 12.0;
    let x_hi = scene.door_x + 12.0;
    let y_lo = scene.waterline() + 1.0;
    let y_hi = scene.cliff_face_y() - scene.being_r - 0.02;
    let at = |p: Pose| Pose { x: p.x.clamp(x_lo, x_hi), y: p.y.clamp(y_lo, y_hi), tilt: p.tilt.clamp(-0.35, 0.35) };

    let mut pose = at(from);
    let mut best = score_of(scene, Knobs::of(&pose), rays_per_band);
    let mut here = match guided(scene, &pose, rays_per_band) {
        Some(v) => v,
        None => return Climb { from, to: pose, best, steps: 0, path: 0.0, solved: best >= scene.open_frac },
    };
    let mut path = 0.0;
    let mut taken = 0;
    for _ in 0..steps {
        if best >= scene.open_frac {
            break;
        }
        let Some(g) = guided_gradient(scene, &pose, rays_per_band) else { break };
        // one unit is a step of either kind, so the direction is the gradient
        // read in that unit
        let d = [g[0] * STEP_M, g[1] * STEP_M, g[2] * step_rad];
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if !n.is_finite() || n < 1e-14 {
            break;
        }
        let mut moved = None;
        for scale in [1.0, 0.5, 0.25, 0.1] {
            let next = at(Pose {
                x: pose.x + scale * STEP_M * d[0] / n,
                y: pose.y + scale * STEP_M * d[1] / n,
                tilt: pose.tilt + scale * step_rad * d[2] / n,
            });
            let Some(v) = guided(scene, &next, rays_per_band) else { continue };
            if v > here {
                moved = Some((next, v));
                break;
            }
        }
        let Some((next, v)) = moved else { break };
        path += ((next.x - pose.x).powi(2) + (next.y - pose.y).powi(2)).sqrt();
        pose = next;
        here = v;
        taken += 1;
        best = best.max(score_of(scene, Knobs::of(&pose), rays_per_band));
    }
    Climb { from, to: pose, best, steps: taken, path, solved: best >= scene.open_frac }
}

/// The design's solvability check: a grid of spawns over the beach, gradient
/// ascent from each, and a line per spawn written to `report`.
///
/// A cove that fails this does not ship — and the fix is a level knob, not a
/// looser test.
pub fn solvable(scene: &CoveScene, side: usize, steps: usize, report: &std::path::Path) -> anyhow::Result<Vec<Climb>> {
    let (x0, x1) = (scene.door_x - 8.0, scene.door_x + 8.0);
    let (y0, y1) = (scene.waterline() + 2.0, scene.cliff_face_y() - 2.0);
    let mut climbs = Vec::new();
    let mut lines = vec![format!(
        "the cove's solvability sweep: {side}×{side} spawns, {steps} steps of 0.2 m, open_frac {:.3}",
        scene.open_frac
    )];
    for iy in 0..side {
        for ix in 0..side {
            let from = Pose {
                x: x0 + (x1 - x0) * (ix as f64 + 0.5) / side as f64,
                y: y0 + (y1 - y0) * (iy as f64 + 0.5) / side as f64,
                tilt: 0.0,
            };
            let c = ascend(scene, from, steps, SWEEP_RAYS);
            lines.push(format!(
                "from ({:+7.2}, {:+7.2}) → ({:+7.2}, {:+7.2}) at {:+5.1}°: {:3} steps, {:6.2} m walked, best frac {:.5} — {}",
                c.from.x,
                c.from.y,
                c.to.x,
                c.to.y,
                c.to.tilt.to_degrees(),
                c.steps,
                c.path,
                c.best,
                if c.solved { "open" } else { "shut" }
            ));
            println!("{}", lines.last().unwrap());
            climbs.push(c);
        }
    }
    let solved = climbs.iter().filter(|c| c.solved).count();
    lines.push(format!("{solved} of {} spawns reached open_frac", climbs.len()));
    if let Some(dir) = report.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(report, lines.join("\n") + "\n")?;
    Ok(climbs)
}
