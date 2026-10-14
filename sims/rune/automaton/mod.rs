//! The player, as a made thing: a porcelain automaton holding a lens.
//!
//! ```text
//! kosm run rune/automaton --out out/
//! ```
//!
//! The cove is an island of made things — a cut door, a dressed keyhole, a
//! being that is a machined body of glass — so the player is a made thing too:
//! a doll of porcelain, brass, lacquered wood and glass, `height_mm` tall, and
//! *every part of it is a primitive*. There is not a triangle in here that the
//! tracer did not make itself: spheres, cones, tori, capped cylinders and one
//! intersection of two spheres. That is the whole aesthetic argument. A path
//! tracer on analytic surfaces gives a perfect silhouette at any distance, an
//! exact normal at every shading point, real contact shadows and a real
//! caustic; a doll is a shape a person would actually build out of those
//! surfaces. The style the cove is written to — flat saturated albedo, clean
//! silhouettes, soft sky light, no textures — is what falls out.
//!
//! ## The assembly
//!
//! One named body per part, so the day this walks it is a phyz rig and not a
//! remodel. The joint centres are written down here and used by
//! [`Layout`]; a revolute or ball joint goes at each of them, parent → child:
//!
//! | joint | pivot (body frame, mm) | axis |
//! |---|---|---|
//! | neck | `(0, 0, neck_z1)` | ball, tipped `head_tilt_deg` back here |
//! | shoulder | `(±shoulder_x, 0, shoulder_z)` | ball |
//! | elbow | [`Layout::elbow`] | hinge about the arm plane's normal |
//! | wrist | the grip point on the lens rim | ball |
//! | hip | `(±hip_x, 0, hip_z)` | ball |
//! | knee | [`Layout::knee`] | hinge about ±x |
//! | ankle | `(±ankle_x, 0, ankle_z)` | ball |
//!
//! The body frame is the doll's own: origin between the soles, `+y` its
//! facing, `+x` its right, `+z` up, millimetres and degrees like every other
//! vcad document. It is placed in the cove by one rigid transform
//! ([`Pose`]) and never authored in world coordinates, so the same document
//! is the turntable model and the doorstep hero.
//!
//! ## The lens, and why it is thin
//!
//! The being of the current slice is a body that happens to focus. The
//! automaton's is a body's worth of glass *figured into a tool*: a biconvex
//! lens `lens_d_mm` across, the intersection of two spheres of equal radius,
//! held overhead in both hands with its axis on the sun.
//!
//! Its focal length is not chosen, it is solved. The doorstep asks one thing
//! of the lens: that the sun through it lands in the keyhole. Put the keyhole
//! at `K`, the sun's travel direction at `d` (down-sun, unit), the lens centre
//! at `L`; then `L = K − f·d`, and the automaton's soles have to be on the
//! sand under it. With the sand a plane of grade `s` through the waterline,
//! that is one equation in one unknown — see [`Pose::solve`] — and it gives
//! `f ≈ 2.45 m` for a doll 1.2 m tall holding the lens 1.44 m up.
//!
//! Then the radii. For a symmetric biconvex lens of radius `R` on both faces
//! (so `R₁ = +R`, `R₂ = −R`) and centre thickness `t`, the thick-lens
//! (lensmaker's) equation is
//!
//! ```text
//! 1/f = (n − 1) [ 2/R − (n − 1) t / (n R²) ]
//! ```
//!
//! and the geometry of the intersection ties `t` to `R`: two spheres of radius
//! `R` whose centres are `2c` apart meet on a circle of radius
//! `h = √(R² − c²)`, which is the lens's semi-aperture, and the lens's
//! vertices sit at `±(R − c)`, so `t = 2(R − c) = 2(R − √(R² − h²))`. One
//! fixed-point pass over those two lines converges in three iterations from
//! the thin-lens guess `R₀ = 2(n − 1)f`.
//!
//! At `f = 2.45 m` and `h = 125 mm` that is `R ≈ 2.53 m` and `t ≈ 6.2 mm`:
//! **the lens is a disc**. An f/10 objective a quarter of a metre across is
//! thin, and pretending otherwise would be pretending about the optics, so the
//! model does not — it puts a brass bezel round the rim instead, which is what
//! a real burning glass that size has and what makes it read as an instrument
//! rather than as a pane. The thickness term above moves `f` by four parts in
//! ten thousand; it is carried anyway, because the arithmetic is the point.
//!
//! The glass is [`materials::being`]'s — N-BK7 with the datasheet's Sellmeier
//! pair — so the automaton's rune disperses by exactly the curve the being's
//! does, and `caustics::trace` finds the lens with nothing told to it: the
//! pass aims at whatever transmits ([`render::Scene::push_part`]).
//!
//! ## The stills
//!
//! Under `out/characters/automaton/`: `door.png` at the doorstep with the
//! caustic on the door face, `portrait.png` close on the head and the lens,
//! and `turntable.png`, three yaws on plain sand. All three are the cove's own
//! sun, sky and materials.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use kosm::brep::instances::{self, Prims};
use kosm::build::{Built, Builder, Params, Shape, build};
use kosm::scene::MM;
use kosm_render::math::{Point3, Transform, Vec3};
use kosm_render::pathtrace::{self, Camera, Ground, Object, Pbr};
use kosm_render::{Bvh, TriMesh};
use vcad_kernel_math::Transform as VTransform;

use super::render::{self, CoveGeom, CoveObject, PER_M};
use super::{materials, scene, CoveScene};

// ---- the layout ------------------------------------------------------------

/// Every length the doll is laid out from, millimetres and degrees.
///
/// Read once from a knob source and used twice — inside the build closure with
/// the [`Builder`]'s knobs, outside it with the [`Built`]'s — so the document
/// and the picture can never disagree about where the lens is. Everything but
/// the lens is a fraction of `h`, so `height_mm` scales the whole doll and the
/// [`Pose`] solve follows it.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    /// Crown of the head above the soles.
    pub h: f64,

    // the legs
    pub foot_r: f64,
    pub foot_l: f64,
    pub foot_y: f64,
    pub ankle_x: f64,
    pub ankle_z: f64,
    pub ankle_r: f64,
    pub hip_x: f64,
    pub hip_z: f64,
    pub hip_r: f64,
    pub thigh: f64,
    pub thigh_r: f64,
    pub shin: f64,
    pub shin_r: f64,
    pub knee_r: f64,

    // the trunk
    pub pelvis_r: f64,
    pub pelvis_z: f64,
    pub waist_major: f64,
    pub waist_minor: f64,
    pub waist_z: f64,
    pub chest_z0: f64,
    pub chest_z1: f64,
    pub chest_r0: f64,
    pub chest_r1: f64,
    pub yoke_major: f64,
    pub yoke_minor: f64,
    pub yoke_z: f64,
    pub stud_r: f64,
    pub stud_z: f64,
    pub stud_proud: f64,
    pub studs: u32,
    pub key_major: f64,
    pub key_minor: f64,
    pub key_z: f64,

    // the arms
    pub shoulder_x: f64,
    pub shoulder_z: f64,
    pub shoulder_r: f64,
    pub upper: f64,
    pub upper_r: f64,
    pub fore: f64,
    pub fore_r: f64,
    pub elbow_r: f64,
    pub wrist_r: f64,
    pub hand_r: f64,

    // the head
    pub neck_r: f64,
    pub neck_z0: f64,
    pub neck_z1: f64,
    pub collar_major: f64,
    pub collar_minor: f64,
    pub collar_z: f64,
    pub head_r: f64,
    pub head_z: f64,
    pub head_tilt: f64,
    pub brow_z: f64,
    pub brow_minor: f64,
    pub eye_d: f64,
    pub eye_yaw: f64,
    pub eye_pitch: f64,

    // the lens
    /// Clear diameter of the glass, before the bezel takes its outer ring.
    pub lens_d: f64,
    pub lens_y: f64,
    pub lens_z: f64,
    /// Elevation of the lens axis above horizontal: the sun's, when the cove
    /// sets it.
    pub lens_pitch: f64,
    pub bezel_minor: f64,
    /// The glass's index at the d line. N-BK7's, like the being's.
    pub n_d: f64,
}

/// The doll the knob defaults are millimetres of. Everything scales off it.
const REFERENCE_H: f64 = 1200.0;

/// The layout, from any source of knobs.
///
/// `layout(&|_, default| default)` is the doll at its authored proportions
/// with nothing registered on the document — which is what a *level* that
/// wants the figure as a prop asks for, since the automaton's forty knobs are
/// the automaton's business and not the cove's.
pub fn layout(knob: &dyn Fn(&str, f64) -> f64) -> Layout {
    let h = knob("height_mm", 1200.0);
    // A length knob: its default is millimetres on the reference 1.2 m doll,
    // and whatever it resolves to is rescaled to *this* doll's height. So
    // overriding `height_mm` alone scales the whole assembly, and overriding
    // one length alone still means what it says.
    let f = |name: &str, mm: f64| knob(name, mm) * h / REFERENCE_H;
    let head_r = f("head_r_mm", 112.0);
    let head_z = h - head_r;
    Layout {
        h,

        foot_r: f("foot_r_mm", 40.0),
        foot_l: f("foot_l_mm", 110.0),
        foot_y: f("foot_y_mm", 30.0),
        ankle_x: f("ankle_x_mm", 92.0),
        ankle_z: f("ankle_z_mm", 78.0),
        ankle_r: f("ankle_r_mm", 26.0),
        hip_x: f("hip_x_mm", 108.0),
        hip_z: f("hip_z_mm", 600.0),
        hip_r: f("hip_r_mm", 40.0),
        thigh: f("thigh_mm", 272.0),
        thigh_r: f("thigh_r_mm", 34.0),
        shin: f("shin_mm", 262.0),
        shin_r: f("shin_r_mm", 31.0),
        knee_r: f("knee_r_mm", 34.0),

        pelvis_r: f("pelvis_r_mm", 95.0),
        pelvis_z: f("pelvis_z_mm", 630.0),
        waist_major: f("waist_major_mm", 76.0),
        waist_minor: f("waist_minor_mm", 18.0),
        waist_z: f("waist_z_mm", 700.0),
        chest_z0: f("chest_z0_mm", 700.0),
        chest_z1: f("chest_z1_mm", 880.0),
        chest_r0: f("chest_r0_mm", 82.0),
        chest_r1: f("chest_r1_mm", 100.0),
        yoke_major: f("yoke_major_mm", 128.0),
        yoke_minor: f("yoke_minor_mm", 34.0),
        yoke_z: f("yoke_z_mm", 878.0),
        stud_r: f("stud_r_mm", 11.0),
        stud_z: f("stud_z_mm", 800.0),
        stud_proud: f("stud_proud_mm", 6.0),
        studs: knob("studs", 8.0).max(3.0) as u32,
        key_major: f("key_major_mm", 44.0),
        key_minor: f("key_minor_mm", 9.0),
        key_z: f("key_z_mm", 780.0),

        shoulder_x: f("shoulder_x_mm", 150.0),
        shoulder_z: f("shoulder_z_mm", 890.0),
        shoulder_r: f("shoulder_r_mm", 46.0),
        upper: f("upper_arm_mm", 300.0),
        upper_r: f("upper_arm_r_mm", 30.0),
        fore: f("forearm_mm", 288.0),
        fore_r: f("forearm_r_mm", 26.0),
        elbow_r: f("elbow_r_mm", 36.0),
        wrist_r: f("wrist_r_mm", 27.0),
        hand_r: f("hand_r_mm", 40.0),

        neck_r: f("neck_r_mm", 38.0),
        neck_z0: f("neck_z0_mm", 860.0),
        neck_z1: f("neck_z1_mm", 1010.0),
        collar_major: f("collar_major_mm", 52.0),
        collar_minor: f("collar_minor_mm", 13.0),
        collar_z: f("collar_z_mm", 928.0),
        head_r,
        head_z,
        head_tilt: knob("head_tilt_deg", 16.0),
        brow_z: f("brow_z_mm", 56.0),
        brow_minor: f("brow_minor_mm", 7.0),
        eye_d: f("eye_d_mm", 40.0),
        eye_yaw: knob("eye_yaw_deg", 22.0),
        eye_pitch: knob("eye_pitch_deg", 8.0),

        lens_d: knob("lens_d_mm", 250.0),
        lens_y: f("lens_y_mm", 140.0),
        lens_z: f("lens_z_mm", 1440.0),
        lens_pitch: knob("lens_pitch_deg", 22.0),
        bezel_minor: knob("bezel_minor_mm", 12.0),
        n_d: knob("n_d", 1.5168),
    }
}

impl Layout {
    /// The lens's clear semi-aperture.
    pub fn lens_h(&self) -> f64 {
        0.5 * self.lens_d
    }

    /// The lens's centre in the body frame.
    pub fn lens_centre(&self) -> [f64; 3] {
        [0.0, self.lens_y, self.lens_z]
    }

    /// The lens's optical axis in the body frame: forward and up by
    /// `lens_pitch`, which is the sun's elevation when the cove sets it.
    pub fn lens_axis(&self) -> [f64; 3] {
        let (s, c) = self.lens_pitch.to_radians().sin_cos();
        [0.0, c, s]
    }

    /// Where a hand grips the rim: the two ends of the rim's horizontal
    /// diameter, which for an axis in the body's own y–z plane is `±x`.
    pub fn grip(&self, side: f64) -> [f64; 3] {
        let c = self.lens_centre();
        [c[0] + side * self.lens_h(), c[1], c[2]]
    }

    /// The shoulder's ball centre.
    pub fn shoulder(&self, side: f64) -> [f64; 3] {
        [side * self.shoulder_x, 0.0, self.shoulder_z]
    }

    /// The elbow's ball centre: two-link IK from the shoulder to the grip,
    /// with the elbow carried out and back so the arms make a lyre rather than
    /// a pair of struts.
    pub fn elbow(&self, side: f64) -> [f64; 3] {
        two_link(self.shoulder(side), self.grip(side), self.upper, self.fore, [0.8 * side, -1.0, 0.0])
    }

    /// The hip's ball centre.
    pub fn hip(&self, side: f64) -> [f64; 3] {
        [side * self.hip_x, 0.0, self.hip_z]
    }

    /// The ankle's ball centre.
    pub fn ankle(&self, side: f64) -> [f64; 3] {
        [side * self.ankle_x, 0.0, self.ankle_z]
    }

    /// The knee's ball centre: the same IK, bending forward.
    pub fn knee(&self, side: f64) -> [f64; 3] {
        two_link(self.hip(side), self.ankle(side), self.thigh, self.shin, [0.0, 1.0, 0.0])
    }

    /// The chest cone's radius at a height, for hanging filigree on it.
    fn chest_r(&self, z: f64) -> f64 {
        let t = ((z - self.chest_z0) / (self.chest_z1 - self.chest_z0)).clamp(0.0, 1.0);
        self.chest_r0 + t * (self.chest_r1 - self.chest_r0)
    }

    /// One face's radius of curvature, and the lens's centre thickness, for a
    /// focal length in millimetres. See the module note for the equations.
    ///
    /// Fixed point on the thick-lens equation from the thin-lens guess. The
    /// thickness term is worth four parts in ten thousand at f/10 and the
    /// whole of the difference at f/1, so it is carried.
    pub fn lens_radius(&self, f: f64) -> (f64, f64) {
        let (n, h) = (self.n_d, self.lens_h());
        let mut r = 2.0 * (n - 1.0) * f;
        for _ in 0..8 {
            // t from the geometry of the intersection at this R
            let t = 2.0 * (r - (r * r - h * h).max(0.0).sqrt());
            // and R from 1/f = (n−1)[2/R − (n−1)t/(nR²)], solved for R by one
            // Newton step in 1/R: the equation is a quadratic in u = 1/R,
            //   (n−1)(n−1)t/n · u² − 2(n−1) u + 1/f = 0
            let a = (n - 1.0) * (n - 1.0) * t / n;
            let b = -2.0 * (n - 1.0);
            let c = 1.0 / f;
            let u = if a.abs() < 1e-18 {
                -c / b
            } else {
                // the root that goes to the thin-lens one as t → 0
                let disc = (b * b - 4.0 * a * c).max(0.0);
                (-b - disc.sqrt()) / (2.0 * a)
            };
            let next = 1.0 / u;
            if (next - r).abs() < 1e-9 {
                r = next;
                break;
            }
            r = next;
        }
        let t = 2.0 * (r - (r * r - h * h).max(0.0).sqrt());
        (r, t)
    }

    /// The focal length the radii above were solved for, back out of the same
    /// equation. A round trip, for the report and for the test.
    pub fn focal_of(&self, r: f64, t: f64) -> f64 {
        let n = self.n_d;
        1.0 / ((n - 1.0) * (2.0 / r - (n - 1.0) * t / (n * r * r)))
    }
}

/// Two links of length `a` then `b` from `s` to `e`, with the joint carried
/// toward `hint`. Returns the joint. Straightens when the span is too long.
fn two_link(s: [f64; 3], e: [f64; 3], a: f64, b: f64, hint: [f64; 3]) -> [f64; 3] {
    let d = sub(e, s);
    let l = norm(d);
    if l < 1e-9 {
        return s;
    }
    let u = scale(d, 1.0 / l);
    let along = ((a * a - b * b + l * l) / (2.0 * l)).clamp(-a, a);
    let off = (a * a - along * along).max(0.0).sqrt();
    let mut v = sub(hint, scale(u, dot(hint, u)));
    let vn = norm(v);
    if vn < 1e-9 {
        // the hint was parallel to the span: any perpendicular will do
        v = if u[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        v = sub(v, scale(u, dot(v, u)));
    }
    let v = scale(v, 1.0 / norm(v));
    add(add(s, scale(u, along)), scale(v, off))
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn mid(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    scale(add(a, b), 0.5)
}

// ---- the document ----------------------------------------------------------

/// A shape authored along `+z` at the origin, turned to point along `dir` and
/// put at `at`.
///
/// vcad's rotations are right-handed Euler XYZ, so `rotate_y(θ)` takes `+z` to
/// `(sin θ, 0, cos θ)` and `rotate_z(φ)` swings that round to
/// `(sin θ cos φ, sin θ sin φ, cos θ)` — which is `dir` for
/// `θ = acos(d_z)`, `φ = atan2(d_y, d_x)`.
fn aim(shape: Shape, dir: [f64; 3], at: [f64; 3]) -> Shape {
    let n = norm(dir);
    if n < 1e-12 {
        return shape.at(at[0], at[1], at[2]);
    }
    let theta = (dir[2] / n).clamp(-1.0, 1.0).acos().to_degrees();
    let phi = dir[1].atan2(dir[0]).to_degrees();
    shape.rotate_y(theta).rotate_z(phi).at(at[0], at[1], at[2])
}

/// A capsule-ended rod between two joint centres: a cylinder of radius `r`
/// spanning them, with a sphere at each end so the joint is never a hard
/// annulus. The joint balls are separate brass bodies, so this is the wood.
fn rod(b: &Builder, from: [f64; 3], to: [f64; 3], r: f64) -> Shape {
    let d = sub(to, from);
    let l = norm(d);
    aim(b.rod_z(r, l.max(1e-6)), d, mid(from, to))
}

/// The biconvex lens: two spheres of radius `r` whose centres are `2c` apart,
/// intersected. Authored about the origin with its axis along `+z`, so
/// [`aim`] puts it on the sun.
fn biconvex(b: &Builder, r: f64, h: f64) -> Shape {
    let c = (r * r - h * h).max(0.0).sqrt();
    b.sphere(r).at(0.0, 0.0, c).intersection(b.sphere(r).at(0.0, 0.0, -c))
}

/// The same lens, as triangles with exact normals.
///
/// **Why this exists.** vcad's kernel evaluates the intersection of two spheres
/// to a solid with a B-rep, and `kosm run rune/automaton`'s own test shows that
/// B-rep's spherical faces come back *untrimmed* to the ray tracer: a ray a
/// metre off the lens's axis, where there is no lens at all, still reports a
/// hit on the 2.5 m sphere the face was cut from
/// (`the_kernel_leaves_a_booleans_faces_untrimmed`). Traced that way the lens
/// is not a lens, it is two enormous glass balls with the doll inside them.
///
/// So the document keeps the honest CAD — the intersection is what a machinist
/// would grind — and the *tracer* is handed this instead: the two spherical
/// caps the intersection is, tessellated, with the analytic normal on every
/// vertex. Shading and refraction are then exact and only the silhouette is a
/// polygon, which at 256 segments is a chord error of nine microns on a
/// 250 mm rim. It is the trade [`render`]'s own `capsule_mesh` makes for the
/// being, for the same reason.
///
/// Built about the origin with the axis along `+z`, exactly as [`biconvex`] is,
/// so the instance walk's own placement carries it.
fn lens_mesh(r: f64, h: f64, segments: usize, rings: usize) -> TriMesh {
    let phi_max = (h / r).clamp(-1.0, 1.0).asin();
    let c = (r * r - h * h).max(0.0).sqrt();
    // South to north, like the capsule: the lower cap from its apex out to the
    // rim, then the upper cap from the rim in to its apex. The rim ring is in
    // the list twice, with the two caps' different normals — which is right,
    // because the rim is a crease and not a smooth pole.
    let mut rows: Vec<(f64, f64)> = Vec::new();
    for v in 0..=rings {
        rows.push((phi_max * v as f64 / rings as f64, -1.0));
    }
    for v in 0..=rings {
        rows.push((phi_max * (1.0 - v as f64 / rings as f64), 1.0));
    }
    let mut positions = Vec::with_capacity(rows.len() * segments);
    let mut normals = Vec::with_capacity(rows.len() * segments);
    for &(phi, s) in &rows {
        let (sp, cp) = phi.sin_cos();
        for i in 0..segments {
            let th = std::f64::consts::TAU * i as f64 / segments as f64;
            let (st, ct) = th.sin_cos();
            // the cap on the +z side belongs to the sphere centred at −c, so
            // its outward normal is (sin φ cos θ, sin φ sin θ, +cos φ)
            let n = Vec3::new(sp * ct, sp * st, s * cp);
            positions.push(Point3::new(r * sp * ct, r * sp * st, s * (r * cp - c)));
            normals.push(n);
        }
    }
    let mut indices: Vec<u32> = Vec::with_capacity(6 * segments * (rows.len() - 1));
    for band in 0..rows.len() - 1 {
        let (a0, b0) = ((band * segments) as u32, ((band + 1) * segments) as u32);
        for i in 0..segments as u32 {
            let j = (i + 1) % segments as u32;
            indices.extend_from_slice(&[a0 + i, a0 + j, b0 + j]);
            indices.extend_from_slice(&[a0 + i, b0 + j, b0 + i]);
        }
    }
    TriMesh::new(positions, normals, &indices)
}

/// The doll, as a vcad document: one named body per part, every one of them
/// decorative (nothing here is stepped yet) and every one of them a union of
/// primitives, so the instance walk hands the tracer analytic B-reps.
pub fn doll(params: &Params) -> anyhow::Result<Built> {
    build(params, |b| {
        let l = layout(&|name, default| b.param(name, default));
        let f = b.param("lens_f_mm", 2450.0);
        assemble(b, &l, f, true);
    })
}

/// The doll's bodies into a builder that is already open, at the doll's own
/// origin — the soles on `z = 0`, `+y` its facing.
///
/// [`doll`] is this with a document of its own; a level that wants the figure
/// standing in it calls this and then places what
/// [`kosm::build::Builder::bodies_since`] hands back. Nothing in here is
/// authored in world coordinates, which is the whole reason the same closure
/// serves the turntable and the doorstep.
///
/// `glass` is the one thing a *level* turns off. The automaton's lens and its
/// two eyes are intersections of spheres two and a half metres across — the
/// kernel's worst case, which is why [`parts`] substitutes [`lens_mesh`] for
/// them — and, worse, a transmissive body in the cove would join the caustic
/// pass's aim ([`kosm_render::caustics::is_caustic_refractor`] looks for
/// exactly that) and spend the rune's photons on a bystander. So the cove
/// takes the doll without its glass: what is left in the hands is the brass
/// bezel, which is the ring the instrument is *missing* its lens from.
pub fn assemble(b: &Builder, l: &Layout, lens_f_mm: f64, glass: bool) {
    {
        // ---- the legs ------------------------------------------------------
        // Ball at the hip, ball at the knee, ball at the ankle; lacquered rods
        // between them; a porcelain capsule for the foot, laid along the
        // facing so the toe is forward of the ankle it hangs from.
        for (name, side) in [("l", -1.0), ("r", 1.0)] {
            let (hip, knee, ankle) = (l.hip(side), l.knee(side), l.ankle(side));
            b.body(&format!("thigh_{name}")).material("lacquer").decorative().add(rod(b, hip, knee, l.thigh_r));
            b.body(&format!("shin_{name}")).material("lacquer").decorative().add(rod(b, knee, ankle, l.shin_r));
            b.body(&format!("hip_ball_{name}")).material("brass").decorative().add(b.sphere(l.hip_r).at(hip[0], hip[1], hip[2]));
            b.body(&format!("knee_ball_{name}")).material("brass").decorative().add(b.sphere(l.knee_r).at(knee[0], knee[1], knee[2]));
            b.body(&format!("ankle_ball_{name}")).material("brass").decorative().add(b.sphere(l.ankle_r).at(ankle[0], ankle[1], ankle[2]));
            b.body(&format!("foot_{name}"))
                .material("porcelain")
                .decorative()
                .add(
                    b.rod_y(l.foot_r, l.foot_l)
                        .union(b.sphere(l.foot_r).at(0.0, 0.5 * l.foot_l, 0.0))
                        .union(b.sphere(l.foot_r).at(0.0, -0.5 * l.foot_l, 0.0))
                        .at(side * l.ankle_x, l.foot_y, l.foot_r),
                );
        }

        // ---- the trunk -----------------------------------------------------
        // A lacquered pelvis ball, a brass waist ring over the join, a cone
        // opening from the waist to the shoulders, and a torus yoke round the
        // cone's top rim. The yoke is what a sphere cap cannot be: at this
        // height a cap big enough to carry the shoulders swallows the neck.
        b.body("pelvis")
            .material("lacquer")
            .decorative()
            .add(b.sphere(l.pelvis_r).at(0.0, 0.0, l.pelvis_z));
        b.body("waist")
            .material("brass")
            .decorative()
            .add(b.torus(l.waist_major, l.waist_minor).at(0.0, 0.0, l.waist_z));
        // A barrel, not a funnel: the chest opens only a little from the waist
        // to the collarbones, and what carries the shoulders is the yoke above
        // it. A cone flared straight out to shoulder width reads as a lampshade
        // with a head sitting in it, which is what the first pass was.
        b.body("chest")
            .material("lacquer")
            .decorative()
            .add(b.cone(l.chest_r0, l.chest_r1, l.chest_z1 - l.chest_z0).at(0.0, 0.0, l.chest_z0));
        // The yoke: a brass ring round the top of the chest, wide enough that
        // the shoulder balls sit in its outer edge, so the arms hang off the
        // *shoulders* rather than out of the torso's rim.
        b.body("yoke")
            .material("brass")
            .decorative()
            .add(b.torus(l.yoke_major, l.yoke_minor).at(0.0, 0.0, l.yoke_z));
        // Filigree where filigree is cheap: one circular pattern of brass beads
        // round the chest, seated proud of the cone so they read as rivets and
        // not as a change of colour.
        b.body("studs").material("brass").decorative().add(
            b.sphere(l.stud_r)
                .at(l.chest_r(l.stud_z) + l.stud_proud, 0.0, l.stud_z)
                .circular_pattern([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], l.studs, 360.0),
        );
        // The winding key, in its back. Nobody winds it in this still; it is
        // the one part that says what the thing is.
        {
            let back = -l.chest_r(l.key_z);              // the lacquer it comes out of
            let stem = 1.6 * l.key_major;
            let ring = back - stem;                      // where the bow sits
            b.body("key").material("brass").decorative().add(
                // `rotate_x(90)` lays the torus in the x-z plane, so the bow
                // stands up off the back rather than lying flat on it
                b.torus(l.key_major, l.key_minor)
                    .rotate_x(90.0)
                    .at(0.0, ring, l.key_z)
                    .union(b.rod_y(l.key_minor, stem).at(0.0, back - 0.5 * stem, l.key_z)),
            );
        }

        // ---- the arms ------------------------------------------------------
        // Raised, both of them, gripping the lens's rim at the ends of its
        // horizontal diameter. The elbows are solved, not authored: see
        // `Layout::elbow`.
        for (name, side) in [("l", -1.0), ("r", 1.0)] {
            let (sh, el, gr) = (l.shoulder(side), l.elbow(side), l.grip(side));
            b.body(&format!("upper_arm_{name}")).material("lacquer").decorative().add(rod(b, sh, el, l.upper_r));
            b.body(&format!("forearm_{name}")).material("lacquer").decorative().add(rod(b, el, gr, l.fore_r));
            b.body(&format!("shoulder_ball_{name}")).material("brass").decorative().add(b.sphere(l.shoulder_r).at(sh[0], sh[1], sh[2]));
            b.body(&format!("elbow_ball_{name}")).material("brass").decorative().add(b.sphere(l.elbow_r).at(el[0], el[1], el[2]));
            // The wrist ball sits back along the forearm from the grip, so the
            // porcelain hand is the thing actually touching the glass.
            let wrist = add(gr, scale(sub(el, gr), l.hand_r / norm(sub(el, gr))));
            b.body(&format!("wrist_ball_{name}")).material("brass").decorative().add(b.sphere(l.wrist_r).at(wrist[0], wrist[1], wrist[2]));
            b.body(&format!("hand_{name}")).material("porcelain").decorative().add(b.sphere(l.hand_r).at(gr[0], gr[1], gr[2]));
        }

        // ---- the head ------------------------------------------------------
        // A neck of porcelain and a brass collar, both upright; then the head,
        // authored about its own pivot at the top of the neck and tipped back
        // `head_tilt_deg` so it is looking up at what it is holding. A perfect
        // sphere, because a perfect sphere is what this renderer is best at
        // and what a moulded head is: the shape is carried by the brass brow
        // band and the two glass eyes and not by a modelled face.
        b.body("neck")
            .material("porcelain")
            .decorative()
            .add(b.cylinder(l.neck_r, l.neck_z1 - l.neck_z0).at(0.0, 0.0, l.neck_z0));
        b.body("collar")
            .material("brass")
            .decorative()
            .add(b.torus(l.collar_major, l.collar_minor).at(0.0, 0.0, l.collar_z));

        // head-local: the pivot is the origin, the ball centre is above it
        let pivot = l.neck_z1;
        let hc = l.head_z - pivot;
        b.body("head")
            .material("porcelain")
            .decorative()
            .add(b.sphere(l.head_r).at(0.0, 0.0, hc))
            .rotate_x(l.head_tilt)
            .at(0.0, 0.0, pivot);
        // The brow band: a torus in the plane the head sphere cuts at
        // `brow_z`, with the major radius the sphere has there, so it is half
        // sunk into the porcelain all the way round.
        let brow_r = (l.head_r * l.head_r - l.brow_z * l.brow_z).max(0.0).sqrt();
        b.body("brow")
            .material("brass")
            .decorative()
            .add(b.torus(brow_r, l.brow_minor).at(0.0, 0.0, hc + l.brow_z))
            .rotate_x(l.head_tilt)
            .at(0.0, 0.0, pivot);

        // The eyes: two little biconvex lenses of the same glass as the big
        // one, set into the porcelain on the head's own normal, each in a
        // brass bezel. They are lenses and not painted discs because that is
        // the whole conceit — the thing sees by refraction.
        {
            // A fat little lens: the radius comes from the sagitta wanted and
            // not from a focal length, because an eye is a jewel and nothing
            // is read through it. 0.866 d gives a lens a third as thick as it
            // is wide, which is as convex as a sphere of that radius allows
            // before the rim goes to a knife edge.
            let r_eye = 0.866 * l.eye_d;
            let eyes = glass.then(|| {
                let eyes = b.body("eyes");
                eyes.material("glass").decorative();
                eyes
            });
            let bezels = b.body("eye_bezels");
            bezels.material("brass").decorative();
            for side in [-1.0f64, 1.0] {
                let (sy, cy) = (side * l.eye_yaw).to_radians().sin_cos();
                let (sp, cp) = l.eye_pitch.to_radians().sin_cos();
                let dir = [sy * cp, cy * cp, sp];
                let seat = scale(dir, l.head_r - 0.10 * l.eye_d);
                let at = [seat[0], seat[1], hc + seat[2]];
                if let Some(eyes) = eyes.as_ref() {
                    eyes.add(aim(biconvex(b, r_eye, 0.5 * l.eye_d), dir, at));
                }
                bezels.add(aim(b.torus(0.5 * l.eye_d, 0.10 * l.eye_d), dir, at));
            }
            if let Some(eyes) = eyes.as_ref() {
                eyes.rotate_x(l.head_tilt).at(0.0, 0.0, pivot);
            }
            bezels.rotate_x(l.head_tilt).at(0.0, 0.0, pivot);
        }

        // ---- the lens ------------------------------------------------------
        // The focal length is a knob here and is *solved* by `Pose::solve`
        // before this document is built, so the same closure serves the
        // turntable (where nothing constrains it) and the doorstep (where the
        // keyhole does).
        let (r_lens, _t) = l.lens_radius(lens_f_mm);
        if glass {
            b.body("lens")
                .material("glass")
                .decorative()
                .add(aim(biconvex(b, r_lens, l.lens_h()), l.lens_axis(), l.lens_centre()));
        }
        b.body("bezel")
            .material("brass")
            .decorative()
            .add(aim(b.torus(l.lens_h(), l.bezel_minor), l.lens_axis(), l.lens_centre()));
    }
}

// ---- the pose --------------------------------------------------------------

/// Where the automaton stands, and what that asks of its lens.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    /// The focal length the doorstep asks for, metres.
    pub f: f64,
    /// The soles' world point, metres.
    pub feet: [f64; 3],
    /// The body's facing, as a world azimuth about `+z` from `+x`. Radians.
    pub yaw: f64,
    /// The lens's centre in world metres, the solve's own answer.
    pub lens: [f64; 3],
}

impl Pose {
    /// Stand the automaton so its lens throws the sun into the keyhole.
    ///
    /// The body faces the sun square on, which is the pose the puzzle is: an
    /// offering. Then the only freedom left is where the feet go, and the
    /// keyhole fixes it. With `K` the keyhole, `d` the sun's travel direction,
    /// `m` the lens's offset from the soles once the body is turned, and the
    /// sand a plane of grade `s` through `sand(K_y)`,
    ///
    /// ```text
    /// P = K − f·d − m          the soles
    /// P_z = sand(K_y) + s·(P_y − K_y)
    /// ⇒ f · (s·d_y − d_z) = sand(K_y) + m_z − s·m_y − K_z
    /// ```
    ///
    /// one equation, one unknown, and `f` falls out. `s·d_y − d_z` is positive
    /// for any sun above the horizon travelling shoreward, so it never
    /// divides by zero on a level that can be solved at all.
    pub fn solve(cove: &CoveScene, l: &Layout) -> anyhow::Result<Self> {
        let k = cove.door_frame().origin;
        let dir = cove.sun_dir();
        let d = [-dir.x, -dir.y, -dir.z]; // down-sun: the way the light goes
        let yaw = cove.sun_az; // square to the sun
        // the lens's offset from the soles, turned into the world
        let c = l.lens_centre();
        let m = [
            (c[1] * yaw.cos() + c[0] * yaw.sin()) * MM,
            (c[1] * yaw.sin() - c[0] * yaw.cos()) * MM,
            c[2] * MM,
        ];
        let s = cove.beach_slope;
        let denom = s * d[1] - d[2];
        anyhow::ensure!(denom > 1e-6, "the sun does not travel shoreward and down: nothing to solve");
        let f = (cove.sand_z_at(k.x, k.y) + m[2] - s * m[1] - k.z) / denom;
        anyhow::ensure!(f > 0.2, "the keyhole is under the lens: focal length {f:.3} m");
        let lens = [k.x - f * d[0], k.y - f * d[1], k.z - f * d[2]];
        Ok(Self { f, feet: [lens[0] - m[0], lens[1] - m[1], lens[2] - m[2]], yaw, lens })
    }

    /// The automaton dropped on flat ground at the origin, turned by `yaw`.
    /// The turntable's pose: nothing to solve, so the lens keeps whatever
    /// focal length it was built with.
    pub fn on_flat(yaw: f64, f: f64) -> Self {
        Self { f, feet: [0.0, 0.0, 0.0], yaw, lens: [0.0, 0.0, 0.0] }
    }

    /// Body → world, in millimetres: a turn about `+z` that takes the body's
    /// `+y` to the facing azimuth, then the soles into place.
    ///
    /// A turn by `β` takes `+y` to `(−sin β, cos β)`, so the facing azimuth
    /// `ψ` wants `β = ψ − 90°`.
    pub fn to_world(&self) -> VTransform {
        let beta = self.yaw - std::f64::consts::FRAC_PI_2;
        let t = VTransform::translation(self.feet[0] * PER_M, self.feet[1] * PER_M, self.feet[2] * PER_M);
        VTransform { matrix: t.matrix * VTransform::rotation_z(beta).matrix }
    }
}

// ---- the parts, as things a tracer can place -------------------------------

/// One traceable piece of the automaton: a BVH, where it sits in the body
/// frame (millimetres), and what it is made of.
pub struct Part {
    bvh: Arc<Bvh<CoveGeom>>,
    to_body: VTransform,
    pbr: Pbr,
}

/// The doll's bodies, walked to placed primitives and given their materials.
///
/// The same walk [`render::Scene::new`] takes over the cove's roots, with one
/// substitution: a body whose material is `glass` is traced as [`lens_mesh`]'s
/// tessellation rather than as the boolean's untrimmed B-rep (see there). The
/// placement is still the document's own, so the two cannot drift apart.
///
/// The glass itself is [`materials::being`]'s — N-BK7 with the Sellmeier pair —
/// resolved here rather than in the cove's table because it is a function of
/// the index and not of a name.
fn parts(doll: &Built) -> anyhow::Result<Vec<Part>> {
    let doc = &doll.document;
    let l = layout(&|n, v| doll.parameter_or(n, v));
    let f = doll.parameter_or("lens_f_mm", 2450.0);
    let mut prims = Prims::default();
    let mut out = Vec::new();
    for (i, root) in doc.roots.iter().enumerate() {
        // `Built::bodies` and `Document::roots` are the same list in the same
        // order, which is what lets a root be asked which part it is.
        let name = doll.bodies.get(i).map(|b| b.name.as_str()).unwrap_or("");
        let glass = root.material == "glass";
        let pbr = if glass { materials::being(l.n_d) } else { materials::pbr(doc, &root.material) };
        // the cap geometry a glass body is traced as: the big lens's radius is
        // the solve's, an eye's is its own
        let cap = match name {
            "lens" => Some((l.lens_radius(f).0, l.lens_h(), 256, 8)),
            "eyes" => Some((0.866 * l.eye_d, 0.5 * l.eye_d, 96, 12)),
            _ => None,
        };
        for inst in instances::instances(doc, root.root, &mut prims)? {
            let bvh = match cap {
                Some((r, h, seg, rings)) => Arc::new(Bvh::build(render::mesh_geometry(lens_mesh(r, h, seg, rings)))),
                None => Arc::new(Bvh::build(render::geometry_of(&inst.solid))),
            };
            out.push(Part { bvh, to_body: inst.to_world, pbr });
        }
    }
    anyhow::ensure!(!out.is_empty(), "the automaton evaluated to no solid");
    anyhow::ensure!(
        out.iter().any(|p| p.pbr.transmission > 0.0),
        "the automaton has no glass, so there is nothing to throw a rune with"
    );
    Ok(out)
}

/// `world · local`, both object → world in millimetres.
fn compose(world: &VTransform, local: &VTransform) -> VTransform {
    VTransform { matrix: world.matrix * local.matrix }
}

/// The same parts, as objects for a picture of my own (the turntable).
fn objects(parts: &[Part], world: &VTransform) -> Vec<CoveObject> {
    parts
        .iter()
        .map(|p| Object::placed(p.bvh.clone(), p.pbr, Transform { matrix: compose(world, &p.to_body).matrix }))
        .collect()
}

// ---- the stills ------------------------------------------------------------

/// A camera at `dist` metres from `at`, on the azimuth `az`, its eye `up`
/// metres above the ground under it. Metres in, millimetres out.
fn eye_at(at: [f64; 3], az: f64, dist: f64, up: f64, target: [f64; 3], vfov: f64) -> Camera {
    let e = [at[0] + dist * az.cos(), at[1] + dist * az.sin(), at[2] + up];
    Camera::look_at(
        Point3::new(e[0] * PER_M, e[1] * PER_M, e[2] * PER_M),
        Point3::new(target[0] * PER_M, target[1] * PER_M, target[2] * PER_M),
        Vec3::new(0.0, 0.0, 1.0),
        vfov,
    )
}

/// `kosm run rune/automaton`.
pub fn run(args: &kosm_cli::Args) -> anyhow::Result<()> {
    let dir = args.out().join("characters").join("automaton");
    fs::create_dir_all(&dir)?;

    let cove = CoveScene::of(scene::scene(&Params::default())?)?;
    let spp: usize = args
        .value("spp")
        .and_then(|v| v.parse().ok())
        .or_else(|| std::env::var("KOSM_SPP").ok().and_then(|v| v.parse().ok()))
        .unwrap_or(128);
    let (w, h) = (960u32, 540u32);

    // The pose is solved before the doll is built, because the solve is what
    // tells the lens what focal length to be. Two passes over the same
    // layout: the first with the authored `lens_f_mm`, which the solve does
    // not read, and the second with the answer.
    let mut knobs = Params::new();
    knobs.set("lens_pitch_deg", cove.sun_el.to_degrees());
    if let Some(v) = args.value("height").and_then(|v| v.parse::<f64>().ok()) {
        knobs.set("height_mm", v);
    }
    let probe = doll(&knobs)?;
    let l = layout(&|name, default| probe.parameter_or(name, default));
    let pose = Pose::solve(&cove, &l)?;
    knobs.set("lens_f_mm", pose.f * PER_M);
    let doll = doll(&knobs)?;
    let l = layout(&|name, default| doll.parameter_or(name, default));
    for warning in &doll.warnings {
        eprintln!("automaton warning: {warning}");
    }

    let (r_lens, t_lens) = l.lens_radius(pose.f * PER_M);
    let parts = parts(&doll)?;
    let glass = parts.iter().filter(|p| p.pbr.transmission > 0.0).count();
    println!(
        "automaton: {} bodies, {} placed primitives ({glass} of them glass, traced as caps), {:.0} mm tall",
        doll.document.roots.len(),
        parts.len(),
        l.h
    );
    println!(
        "automaton lens: {:.0} mm clear, R = {:.1} mm on both faces, t = {:.2} mm at the centre, n_d = {:.4} -> f = {:.3} m (asked {:.3} m)",
        l.lens_d,
        r_lens,
        t_lens,
        l.n_d,
        l.focal_of(r_lens, t_lens) * MM,
        pose.f
    );
    println!(
        "automaton stands at ({:+.3}, {:+.3}) m, {:.2} m out from the cliff face, facing {:.0}deg; the lens is at ({:+.3}, {:+.3}, {:+.3}) m and the keyhole {:.3} m down-sun of it",
        pose.feet[0],
        pose.feet[1],
        cove.cliff_face_y() - pose.feet[1],
        pose.yaw.to_degrees(),
        pose.lens[0],
        pose.lens[1],
        pose.lens[2],
        pose.f
    );

    // ---- the two cove stills ----------------------------------------------
    let world = pose.to_world();
    let mut picture = render::Scene::new(&cove)?;
    picture.set_being_visible(false);
    // A figured lens focuses two orders tighter than a body does: the sun's
    // half-degree through f/10 is a spot about 25 mm across, so the cove's
    // gather — half a keyhole — would average the rune away.
    picture.set_photons(args.value("photons").and_then(|v| v.parse().ok()).unwrap_or(1_200_000));
    picture.set_gather(args.value("gather").and_then(|v| v.parse().ok()).unwrap_or(28.0));
    for p in &parts {
        picture.push_object(p.bvh.clone(), p.pbr, Transform { matrix: compose(&world, &p.to_body).matrix });
    }

    // The placement is the cove's, and with the being hidden it carries only
    // the door's angle, the rim's score and the glint: none of them ours.
    let placement = render::Placement::standing(&cove, pose.feet[0], pose.feet[1], 0.0);
    let t0 = std::time::Instant::now();
    let rune = picture.caustic_map(&placement);
    println!("automaton rune: photon map in {:.1} s", t0.elapsed().as_secs_f64());
    let at = picture.at(&placement);
    let opts = render::options(&cove, spp, SEED);
    let exposure = cove.authored.parameter_or("exposure", 0.7);

    // (1) the doorstep. Three metres out on an azimuth `look_off` round from
    // the sun, which is the whole composition: on the sun's own line the
    // camera, the lens and the keyhole are collinear and the doll stands in
    // front of the thing it is solving.
    let look_off = args.value("look_off").and_then(|v| v.parse::<f64>().ok()).unwrap_or(45.0);
    let k = cove.door_frame().origin;
    let ground = [pose.feet[0], pose.feet[1], pose.feet[2]];
    let target = [
        0.65 * pose.feet[0] + 0.35 * k.x,
        0.65 * pose.feet[1] + 0.35 * k.y,
        pose.feet[2] + 1.05,
    ];
    let cam = eye_at(ground, pose.yaw + look_off.to_radians(), 2.8, 0.55, target, 40.0);
    let t0 = std::time::Instant::now();
    let film = pathtrace::render_with_caustics(&at, &cam, w, h, &opts, Some(&rune));
    let path = dir.join("door.png");
    render::to_image(&film, exposure).save(&path)?;
    println!("automaton door: {w}x{h} at {spp} spp -> {} in {:.1} s", path.display(), t0.elapsed().as_secs_f64());

    // (2) the portrait: a metre and a half out, on the head and the lens.
    let head = [pose.feet[0], pose.feet[1], pose.feet[2] + l.head_z * MM];
    let subject = [head[0], head[1], pose.feet[2] + 0.5 * (l.head_z + l.lens_z) * MM];
    let cam = eye_at(
        [head[0], head[1], pose.feet[2]],
        pose.yaw + (look_off + 15.0).to_radians(),
        1.6,
        (0.5 * (l.head_z + l.lens_z) * MM) - 0.10,
        subject,
        32.0,
    );
    let t0 = std::time::Instant::now();
    let film = pathtrace::render_with_caustics(&at, &cam, w, h, &opts, Some(&rune));
    let path = dir.join("portrait.png");
    render::to_image(&film, exposure).save(&path)?;
    println!("automaton portrait: {w}x{h} at {spp} spp -> {} in {:.1} s", path.display(), t0.elapsed().as_secs_f64());

    // (3) the turntable, on plain sand.
    turntable(&cove, &doll, &l, &parts, &dir, spp)?;
    Ok(())
}

/// Three yaws on a plane of sand, in the cove's own light, side by side.
///
/// Its own picture rather than the cove's: a model sheet wants nothing in it
/// but the model, its shadow and the ground it stands on. The sun and the sky
/// are still [`render::daylight`]'s, so the palette is the one the level
/// judges it by.
fn turntable(
    cove: &CoveScene,
    doll: &Built,
    l: &Layout,
    parts: &[Part],
    dir: &Path,
    spp: usize,
) -> anyhow::Result<()> {
    let (pw, ph) = (320u32, 540u32);
    let (env, sun) = render::daylight(cove);
    let opts = render::options(cove, spp, SEED ^ 0x77);
    let exposure = cove.authored.parameter_or("exposure", 0.7);
    let f = doll.parameter_or("lens_f_mm", 2450.0) * MM;
    let mut panels = Vec::new();
    let t0 = std::time::Instant::now();
    for k in 0..3 {
        let yaw = cove.sun_az + (k as f64) * std::f64::consts::TAU / 3.0;
        let pose = Pose::on_flat(yaw, f);
        let picture = pathtrace::Scene {
            objects: objects(parts, &pose.to_world()),
            lights: Vec::new(),
            env: env.clone(),
            sun: Some(sun),
            ground: Some(Ground { z: 0.0, material: materials::pbr(&doll.document, "sand"), shadow_catcher: false }),
            splats: None,
        };
        let subject = [0.0, 0.0, 0.5 * l.lens_z * MM];
        let cam = eye_at([0.0, 0.0, 0.0], cove.sun_az + 40f64.to_radians(), 3.2, 0.9, subject, 34.0);
        let film = pathtrace::render(&picture, &cam, pw, ph, &opts);
        panels.push(render::to_image(&film, exposure));
    }
    let mut strip = image::RgbaImage::new(pw * 3, ph);
    for (i, panel) in panels.iter().enumerate() {
        for y in 0..ph {
            for x in 0..pw {
                strip.put_pixel(i as u32 * pw + x, y, *panel.get_pixel(x, y));
            }
        }
    }
    let path = dir.join("turntable.png");
    strip.save(&path)?;
    println!(
        "automaton turntable: 3 x {pw}x{ph} at {spp} spp -> {} in {:.1} s",
        path.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// The stills' seed. Fixed, so two runs differ only where the doll does.
const SEED: u64 = 0xa0_70_11a_11;

#[cfg(test)]
mod tests {
    use super::*;

    fn bundled() -> (CoveScene, Built, Layout) {
        let cove = CoveScene::bundled().expect("the bundled cove");
        let mut knobs = Params::new();
        knobs.set("lens_pitch_deg", cove.sun_el.to_degrees());
        let d = doll(&knobs).expect("the automaton builds");
        let l = layout(&|n, v| d.parameter_or(n, v));
        (cove, d, l)
    }

    /// The lens is the one the arithmetic in the module note describes: the
    /// radii solved from a focal length give that focal length back, and the
    /// thickness is the intersection's own.
    #[test]
    fn the_lens_radii_give_the_focal_length_back() {
        let (_, _, l) = bundled();
        for f in [1000.0, 2450.0, 5000.0] {
            let (r, t) = l.lens_radius(f);
            assert!((l.focal_of(r, t) - f).abs() < 1e-6 * f, "f = {f}: R = {r}, t = {t} gives {}", l.focal_of(r, t));
            // t is exactly what two spheres of that radius cut
            let c = (r * r - l.lens_h() * l.lens_h()).sqrt();
            assert!((t - 2.0 * (r - c)).abs() < 1e-9);
            // and the thin-lens guess is within a per cent of it at f/10
            assert!((r - 2.0 * (l.n_d - 1.0) * f).abs() < 0.01 * r);
        }
    }

    /// The solve puts the focus in the keyhole: march the sun's own ray from
    /// the lens and it arrives where the rune is read.
    #[test]
    fn the_solved_pose_aims_the_focus_at_the_keyhole() {
        let (cove, _, l) = bundled();
        let pose = Pose::solve(&cove, &l).expect("a solvable doorstep");
        let d = cove.sun_dir();
        let hit = [
            pose.lens[0] - pose.f * d.x,
            pose.lens[1] - pose.f * d.y,
            pose.lens[2] - pose.f * d.z,
        ];
        let k = cove.door_frame().origin;
        assert!((hit[0] - k.x).abs() < 1e-9 && (hit[1] - k.y).abs() < 1e-9 && (hit[2] - k.z).abs() < 1e-9, "{hit:?} is not the keyhole");
        // …and the automaton is standing on the sand, not in it or over it
        assert!((pose.feet[2] - cove.sand_z_at(pose.feet[0], pose.feet[1])).abs() < 1e-9);
        // …a couple of metres off the door, which is the design's doorstep
        let out = cove.cliff_face_y() - pose.feet[1];
        assert!((1.5..3.5).contains(&out), "the doorstep is {out:.2} m out");
        assert!((2.0..3.0).contains(&pose.f), "the focal length is {:.2} m", pose.f);
    }

    /// The arms actually reach the rim: the two-link solve never has to
    /// straighten, and never has slack it cannot take up.
    #[test]
    fn the_hands_are_on_the_lens_rim_and_the_arms_can_reach() {
        let (_, _, l) = bundled();
        for side in [-1.0f64, 1.0] {
            let (sh, el, gr) = (l.shoulder(side), l.elbow(side), l.grip(side));
            assert!((norm(sub(el, sh)) - l.upper).abs() < 1e-6, "the upper arm is not its own length");
            assert!((norm(sub(gr, el)) - l.fore).abs() < 1e-6, "the forearm is not its own length");
            let span = norm(sub(gr, sh));
            assert!(span < l.upper + l.fore, "the arm is straight: {span:.1} of {:.1}", l.upper + l.fore);
            assert!(span > 0.9 * (l.upper + l.fore), "the arm is folded: {span:.1}");
            // the grip is on the rim, at the lens's own centre height
            let c = l.lens_centre();
            assert!((norm(sub(gr, c)) - l.lens_h()).abs() < 1e-9);
        }
        // and the legs are nearly straight but not locked
        for side in [-1.0f64, 1.0] {
            let (hip, knee, ankle) = (l.hip(side), l.knee(side), l.ankle(side));
            assert!((norm(sub(knee, hip)) - l.thigh).abs() < 1e-6);
            assert!((norm(sub(ankle, knee)) - l.shin).abs() < 1e-6);
            assert!(knee[1] > 0.0, "the knee bends forward");
        }
    }

    /// Every opaque part is a primitive with a B-rep the tracer traces
    /// analytically, and the glass is exactly three pieces: the lens and two
    /// eyes, and nothing else in the doll refracts.
    #[test]
    fn the_doll_is_primitives_and_refracts_in_three_places() {
        let (_, d, _) = bundled();
        let parts = parts(&d).expect("the automaton walks to instances");
        assert!(parts.len() >= 30, "only {} primitives", parts.len());
        let glass = parts.iter().filter(|p| p.pbr.transmission > 0.0).count();
        assert_eq!(glass, 3, "the automaton should refract in exactly three places, not {glass}");
        // and every solid the walk produced is a B-rep, glass included: the
        // document is honest CAD even where the tracer is handed a mesh
        let doc = &d.document;
        let mut prims = Prims::default();
        for root in &doc.roots {
            for inst in instances::instances(doc, root.root, &mut prims).unwrap() {
                assert!(inst.solid.as_brep().is_some(), "a part came back with no B-rep at all");
            }
        }
    }

    /// Why the glass is meshed. vcad evaluates the intersection of two spheres
    /// to a B-rep whose spherical faces the ray tracer does *not* trim: a ray a
    /// metre off the lens's axis, where the lens does not exist, still reports
    /// a hit on the 2.5 m sphere the cap was cut from. [`lens_mesh`] is the way
    /// round it, and this is the test that says so — if a later vcad fixes the
    /// trimming, this fails and the mesh can go.
    #[test]
    fn the_kernel_leaves_a_booleans_faces_untrimmed() {
        use kosm_render::Ray;
        let (_, d, l) = bundled();
        let doc = &d.document;
        let mut prims = Prims::default();
        let i = d.bodies.iter().position(|b| b.name == "lens").expect("a lens body");
        let inst = instances::instances(doc, doc.roots[i].root, &mut prims).unwrap().remove(0);
        // A ray in the lens's own frame, crossing a metre above it. The lens
        // is six millimetres thick about z = 0 and 250 mm across, so there is
        // nothing whatever up there — but the sphere the top cap was cut from
        // reaches 2.5 m, and that is what gets hit.
        let ray = Ray::new(Point3::new(0.0, -4000.0, 1000.0), Vec3::new(0.0, 1.0, 0.0));
        let raw = Bvh::build(render::geometry_of(&inst.solid));
        assert!(raw.trace_closest(&ray).is_some(), "vcad trims booleans now: drop lens_mesh");
        // …and the tessellation this module traces instead does not
        let (r, _) = l.lens_radius(d.parameter_or("lens_f_mm", 2450.0));
        let capped = Bvh::build(render::mesh_geometry(lens_mesh(r, l.lens_h(), 256, 8)));
        assert!(capped.trace_closest(&ray).is_none(), "the meshed lens is not a lens either");
        // the mesh is the lens: its rim is the semi-aperture and it is as thick
        // at the centre as the intersection is
        let (_, t) = l.lens_radius(d.parameter_or("lens_f_mm", 2450.0));
        let b = capped.bounds().expect("the mesh has bounds");
        assert!((b.max.x - l.lens_h()).abs() < 0.02, "the rim is at {:.3}, not {:.3}", b.max.x, l.lens_h());
        assert!((b.max.z - b.min.z - t).abs() < 1e-6, "the mesh is {:.4} thick, not {t:.4}", b.max.z - b.min.z);
    }

    /// The doll stands on its soles and its crown is where `height_mm` says.
    #[test]
    fn the_doll_is_as_tall_as_it_says() {
        let (_, _, l) = bundled();
        assert!((l.head_z + l.head_r - l.h).abs() < 1e-9, "the crown is not at height_mm");
        assert!(l.lens_z > l.h, "the lens is not raised over the head");
        assert!(l.ankle_z - l.foot_r < l.foot_r, "the ankle is not over the foot");
    }
}

