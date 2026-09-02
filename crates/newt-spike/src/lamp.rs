//! The marble under a lamp.
//!
//! A point lamp above the table casts the marble's shadow onto the plate. The
//! objective is no longer "the marble is in the cup" but "the marble's shadow
//! lands here". Light and motion share one backward pass: the light part is
//! the analytic derivative of a ray-plane projection, the motion part is the
//! convex-contact adjoint, and the chain rule joins them at the marble's
//! final position. `∂shadow/∂release` is exact, through rolling, through the
//! cup rim, through the lamp.
//!
//! Two knobs solve the same target: move the release point (gradient descent
//! on the chained adjoint) or move the lamp (closed form, because the shadow
//! is affine in the lamp position at fixed height).
//!
//! phyz-camera draws no shadows, so the frame gets the shadow composited in:
//! the sphere's silhouette is projected along each point's own lamp ray onto
//! the plate, then through the camera. Same geometry the objective uses.

use std::path::Path;

use phyz_camera::CameraPose;
use phyz_diff::{ConvexContactRollout, FinalStateObjective, convex_adjoint_gradient, convex_rollout_objective};
use phyz_math::{DVec, Mat3, SpatialTransform, SpatialTransformExt, Vec3};
use phyz_world::CameraIntrinsics;

const POS: usize = 3;

/// A lamp in the world and a target on the plate.
#[derive(Clone, Copy)]
pub struct Lamp {
    /// Lamp position, world frame, metres.
    pub pos: Vec3,
    /// Where the shadow should land, plate frame, metres (z = 0).
    pub target: [f64; 2],
}

/// Shadow of the sphere centre `c` (world) on the plate plane, in plate
/// coordinates, plus its Jacobian with respect to `c` (world).
pub fn shadow(lamp: &Lamp, xf: &SpatialTransform, c: Vec3) -> ([f64; 2], [[f64; 3]; 2]) {
    // Work in the plate frame: the plate top is z = 0 there.
    let l = xf.world_to_body_point(lamp.pos);
    let cl = xf.world_to_body_point(c);
    let d = cl - l;
    let t = l.z / (l.z - cl.z);
    let s = l + d * t;
    // ∂s_xy/∂cl = t·[I 0] + d_xy ⊗ ∂t/∂cl_z,  ∂t/∂cl_z = l.z/(l.z − cl.z)²
    let dt = l.z / ((l.z - cl.z) * (l.z - cl.z));
    let j_local = [[t, 0.0, d.x * dt], [0.0, t, d.y * dt]];
    // cl = R (c − pos), so ∂cl/∂c = R (world→body).
    let r: &Mat3 = &xf.rot;
    let mut j = [[0.0; 3]; 2];
    for (row, jl) in j_local.iter().enumerate() {
        for k in 0..3 {
            j[row][k] = jl[0] * r[(0, k)] + jl[1] * r[(1, k)] + jl[2] * r[(2, k)];
        }
    }
    ([s.x, s.y], j)
}

/// J = |shadow(marble) − target|², with its gradient in the marble's q.
pub fn objective(lamp: Lamp, xf: SpatialTransform) -> FinalStateObjective<'static> {
    let lamp: &'static Lamp = Box::leak(Box::new(lamp));
    let xf: &'static SpatialTransform = Box::leak(Box::new(xf));
    let value: &'static dyn Fn(&[f64], &[f64]) -> f64 = Box::leak(Box::new(move |q: &[f64], _: &[f64]| {
        let (s, _) = shadow(lamp, xf, Vec3::new(q[POS], q[POS + 1], q[POS + 2]));
        let (dx, dy) = (s[0] - lamp.target[0], s[1] - lamp.target[1]);
        dx * dx + dy * dy
    }));
    type GradFn = dyn Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>);
    let gradient: &'static GradFn = Box::leak(Box::new(move |q: &[f64], v: &[f64]| {
        let (s, j) = shadow(lamp, xf, Vec3::new(q[POS], q[POS + 1], q[POS + 2]));
        let e = [2.0 * (s[0] - lamp.target[0]), 2.0 * (s[1] - lamp.target[1])];
        let mut gq = vec![0.0; q.len()];
        for k in 0..3 {
            gq[POS + k] = e[0] * j[0][k] + e[1] * j[1][k];
        }
        (gq, vec![0.0; v.len()])
    }));
    FinalStateObjective { value, gradient }
}

/// The lamp position (world, at the same height) that puts the shadow of a
/// marble at `c` exactly on the target. Closed form: at fixed lamp height the
/// shadow is affine in the lamp's xy.
pub fn lamp_for(lamp: &Lamp, xf: &SpatialTransform, c: Vec3) -> Vec3 {
    let l = xf.world_to_body_point(lamp.pos);
    let cl = xf.world_to_body_point(c);
    let t = l.z / (l.z - cl.z);
    // s = l + t (cl − l)  ⇒  l_xy = (s − t·cl_xy) / (1 − t)
    let lx = (lamp.target[0] - t * cl.x) / (1.0 - t);
    let ly = (lamp.target[1] - t * cl.y) / (1.0 - t);
    xf.body_to_world_point(Vec3::new(lx, ly, l.z))
}

/// Composite the marble's shadow, the target ring and the lamp into a frame.
pub fn draw(
    img: &mut image::RgbaImage,
    pose: &CameraPose,
    intr: &CameraIntrinsics,
    lamp: &Lamp,
    xf: &SpatialTransform,
    c: Vec3,
    r: f64,
) {
    // Silhouette: the great circle perpendicular to the lamp ray through the
    // centre; each point casts along its own ray onto the plate.
    let to_lamp = (lamp.pos - c).normalize();
    let u = if to_lamp.x.abs() < 0.9 { Vec3::x() } else { Vec3::y() };
    let e1 = to_lamp.cross(&u).normalize();
    let e2 = to_lamp.cross(&e1);
    let n = 64;
    let mut outline = Vec::with_capacity(n);
    for k in 0..n {
        let a = std::f64::consts::TAU * k as f64 / n as f64;
        let p = c + e1 * (r * a.cos()) + e2 * (r * a.sin());
        let (s, _) = shadow(lamp, xf, p);
        if let Some(px) = pose.project(intr, xf.body_to_world_point(Vec3::new(s[0], s[1], 0.0))) {
            outline.push(px);
        }
    }
    fill_polygon(img, &outline, [20, 20, 28, 170]);

    // Target: a ring on the plate.
    let mut ring = Vec::new();
    for k in 0..n {
        let a = std::f64::consts::TAU * k as f64 / n as f64;
        let p = Vec3::new(lamp.target[0] + 0.012 * a.cos(), lamp.target[1] + 0.012 * a.sin(), 0.0005);
        if let Some(px) = pose.project(intr, xf.body_to_world_point(p)) {
            ring.push(px);
        }
    }
    stroke_polygon(img, &ring, [230, 160, 40, 255]);

    // The lamp, if it is in frame.
    if let Some((x, y)) = pose.project(intr, lamp.pos) {
        disc(img, x, y, 6.0, [255, 220, 120, 255]);
    }
}

fn fill_polygon(img: &mut image::RgbaImage, poly: &[(f64, f64)], rgba: [u8; 4]) {
    if poly.len() < 3 {
        return;
    }
    let (w, h) = (img.width() as i64, img.height() as i64);
    let x0 = poly.iter().map(|p| p.0).fold(f64::MAX, f64::min).floor().max(0.0) as i64;
    let x1 = poly.iter().map(|p| p.0).fold(f64::MIN, f64::max).ceil().min((w - 1) as f64) as i64;
    let y0 = poly.iter().map(|p| p.1).fold(f64::MAX, f64::min).floor().max(0.0) as i64;
    let y1 = poly.iter().map(|p| p.1).fold(f64::MIN, f64::max).ceil().min((h - 1) as f64) as i64;
    for y in y0..=y1 {
        for x in x0..=x1 {
            if inside(poly, x as f64 + 0.5, y as f64 + 0.5) {
                blend(img, x as u32, y as u32, rgba);
            }
        }
    }
}

fn stroke_polygon(img: &mut image::RgbaImage, poly: &[(f64, f64)], rgba: [u8; 4]) {
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        let steps = ((b.0 - a.0).hypot(b.1 - a.1).ceil() as usize).max(1);
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let (x, y) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
            if x >= 0.0 && y >= 0.0 && (x as u32) < img.width() && (y as u32) < img.height() {
                blend(img, x as u32, y as u32, rgba);
            }
        }
    }
}

fn disc(img: &mut image::RgbaImage, cx: f64, cy: f64, r: f64, rgba: [u8; 4]) {
    for y in (cy - r).floor() as i64..=(cy + r).ceil() as i64 {
        for x in (cx - r).floor() as i64..=(cx + r).ceil() as i64 {
            if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
                if (x as f64 + 0.5 - cx).hypot(y as f64 + 0.5 - cy) <= r {
                    blend(img, x as u32, y as u32, rgba);
                }
            }
        }
    }
}

fn inside(poly: &[(f64, f64)], x: f64, y: f64) -> bool {
    let mut c = false;
    let n = poly.len();
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[(i + n - 1) % n];
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            c = !c;
        }
    }
    c
}

fn blend(img: &mut image::RgbaImage, x: u32, y: u32, rgba: [u8; 4]) {
    let p = img.get_pixel_mut(x, y);
    let a = rgba[3] as f64 / 255.0;
    for k in 0..3 {
        p.0[k] = (p.0[k] as f64 * (1.0 - a) + rgba[k] as f64 * a).round() as u8;
    }
    p.0[3] = 255;
}

/// Gradient descent on the release point against the shadow objective, with
/// backtracking. Returns the release point and the final shadow miss.
#[allow(clippy::too_many_arguments)]
pub fn solve_release<'a>(
    model: &phyz_model::Model,
    xf: &SpatialTransform,
    obj: &FinalStateObjective<'static>,
    q0_for: &dyn Fn([f64; 2]) -> DVec,
    clamp: &dyn Fn([f64; 2]) -> [f64; 2],
    rollout: &dyn Fn(DVec) -> ConvexContactRollout<'a>,
    start: [f64; 2],
    log: &dyn Fn(String),
) -> ([f64; 2], f64) {
    let _ = model;
    // The cup is a trap: once caught, the marble's position is pinned and the
    // shadow cannot move. Look before descending: a coarse grid over release
    // points, then descend from the best.
    let mut xy = start;
    let mut j = convex_rollout_objective(&rollout(q0_for(xy)), obj);
    for gx in [-0.12, -0.10, -0.08] {
        for gy in [-0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075] {
            let cand = clamp([gx, gy]);
            let jc = convex_rollout_objective(&rollout(q0_for(cand)), obj);
            if jc < j {
                xy = cand;
                j = jc;
            }
        }
    }
    log(format!("lamp   grid: best of 21 releases is ({:+.3}, {:+.3}), shadow miss {:.4} m", xy[0], xy[1], j.sqrt()));
    let mut step = 0.01;
    for it in 0..14 {
        log(format!("lamp   it {it:2}  release ({:+.3}, {:+.3})  shadow miss {:.4} m", xy[0], xy[1], j.sqrt()));
        if j.sqrt() < 0.002 {
            break;
        }
        let gr = match convex_adjoint_gradient(&rollout(q0_for(xy)), obj) {
            Ok(g) => g,
            Err(e) => {
                log(format!("lamp   adjoint refused: {e}"));
                break;
            }
        };
        let gl = xf.rot * Vec3::new(gr.d_q0[POS], gr.d_q0[POS + 1], gr.d_q0[POS + 2]);
        let gn = (gl.x * gl.x + gl.y * gl.y).sqrt().max(1e-12);
        let dir = [-gl.x / gn, -gl.y / gn];
        let mut accepted = false;
        for _ in 0..6 {
            let cand = clamp([xy[0] + step * dir[0], xy[1] + step * dir[1]]);
            let jc = convex_rollout_objective(&rollout(q0_for(cand)), obj);
            if jc < j {
                xy = cand;
                j = jc;
                accepted = true;
                step = (step * 1.5).min(0.02);
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            log(format!("lamp   no descent direction within {:.1} mm; stopping", step * 2e3));
            break;
        }
    }
    (xy, j.sqrt())
}

pub fn out_dir(out: &Path) -> std::path::PathBuf {
    out.join("lamp")
}
