//! Light as physics.
//!
//! The marble is glass. Light from the lamp enters it, bends by Snell's law,
//! loses a Fresnel share at each face, bends again on the way out, and lands
//! on the plate as a caustic: a bright focus with a coloured rim, because the
//! index of refraction depends on wavelength and the bands fan out. Nothing
//! here is a shading trick. It is forward light transport, spectral, energy
//! conserving, written once over `tang::Scalar` like the rest of the frame.
//!
//! Which means it differentiates. Seed the index of refraction with a `Dual`
//! and the caustic's derivative with respect to it comes out of the same
//! pass; compare with finite differences; then run it backwards: given the
//! caustic a marble makes, recover what it is made of. ∂image/∂material, with
//! the material being a physical constant and not an artist's slider.
//!
//! Dispersion is Sellmeier, with N-BK7's coefficients as the *shape* and the
//! d-line index `n_d` as the knob; the generic Sellmeier is checked against
//! `vcad-kernel-optics`' own N-BK7 at three wavelengths so the two crates
//! cannot drift.

use std::path::Path;

use phyz_math::SpatialTransformExt;
use tang::{Dual, Scalar, Vec3};

/// Wavelength bands (µm) and how each contributes to display RGB. A five-band
/// spectrum is enough to show a rim that is red on one side and blue on the
/// other; more bands make the fan smoother, not different.
const BANDS: [(f64, [f64; 3]); 5] = [
    (0.450, [0.05, 0.05, 1.00]),
    (0.500, [0.00, 0.60, 0.60]),
    (0.550, [0.20, 1.00, 0.10]),
    (0.600, [0.80, 0.55, 0.00]),
    (0.650, [1.00, 0.05, 0.05]),
];

/// How many bands the spectrum is split into.
pub const BANDS_LEN: usize = 5;

// Sellmeier's dispersion, and the d-line knob on top of it, are
// `kosm-render`'s: the shape of a glass's index is optics and not this
// level's. `light::index` and `light::sellmeier` still name them.
pub use kosm_render::optics::{D_LINE_UM, index, sellmeier};

/// The caustic a glass sphere throws on the plate: irradiance per band, on a
/// grid in plate coordinates, plus a shadow-free reference (what the plate
/// would receive from the lamp through the same solid angle with no marble).
#[derive(Clone)]
pub struct Caustic<S: Scalar> {
    /// Grid origin (plate frame, metres) and cell size.
    pub origin: [f64; 2],
    pub cell: f64,
    pub n: usize,
    /// `bands × n × n` irradiance, W/m² per unit lamp radiant intensity.
    pub e: Vec<Vec<S>>,
    /// Rays traced, rays lost to total internal reflection, rays that left
    /// the glass but landed outside the window.
    pub traced: usize,
    pub tir: usize,
    pub outside: usize,
    /// How much of the lamp the solid actually caught: the cone's solid angle
    /// per ray times the aperture weight of every ray that entered the glass,
    /// summed over the bands.
    ///
    /// The denominator a *ratio* wants, and it is here rather than left to
    /// the caller's own arithmetic because the caller cannot state it. The
    /// rays are weighted by [`crate::glass::Shape::rim_weight`], which
    /// feathers the outer few per cent of an aperture so the sum has a
    /// derivative at all; an analytic projected area is the *hard* disc, and
    /// dividing the one by the other would be measuring two different pieces
    /// of glass. Read both ends off the lattice and the feather is in both,
    /// so it cancels and what is left is the transport — which is what the
    /// derivative was ever about. `sims/rune`'s `hint::lens_incident` is the
    /// analytic area kept beside it, and the ratio of the two is what the
    /// soft rim costs.
    pub caught: S,
}

/// The plane the light is caught on: its origin, the normal it is lit from,
/// and the two in-plane axes the caustic's grid is indexed by.
///
/// The plate is the identity case. The cove's door is not: its face is a
/// vertical wall, and the rune is the caustic read in *its* coordinates, so
/// the receiver has to be a frame and not an assumption about z.
#[derive(Clone, Copy, Debug)]
pub struct Receiver {
    pub origin: Vec3<f64>,
    pub normal: Vec3<f64>,
    pub u: Vec3<f64>,
    pub v: Vec3<f64>,
}

impl Receiver {
    /// The marble's plate: the world's own axes, receiving at z = 0.
    pub fn plate() -> Self {
        Self { origin: Vec3::zero(), normal: Vec3::z(), u: Vec3::x(), v: Vec3::y() }
    }

    /// A world point in the receiver's coordinates.
    fn point(&self, p: Vec3<f64>) -> Vec3<f64> {
        let q = p - self.origin;
        Vec3::new(self.u.dot(&q), self.v.dot(&q), self.normal.dot(&q))
    }
}

/// Trace `rays_per_band` light rays per band from the lamp through a glass
/// `shape` to the plate. Everything in the plate frame (plate top is z = 0).
pub fn trace<S: Scalar>(
    lamp: Vec3<f64>,
    shape: &crate::glass::Shape<S>,
    nd: S,
    window: f64,
    cells: usize,
    rays_per_band: usize,
) -> Caustic<S> {
    trace_onto(lamp, shape, nd, Receiver::plate(), window, cells, rays_per_band)
}

/// The same trace onto any receiving plane. The lamp and the glass are moved
/// into the receiver's frame — one rigid change of coordinates, which the
/// duals carry through untouched — and from there this *is* the plate case,
/// so there is one ray walk and not two to keep in agreement.
///
/// The returned `Caustic`'s grid is in the receiver's `(u, v)`.
pub fn trace_onto<S: Scalar>(
    lamp: Vec3<f64>,
    shape: &crate::glass::Shape<S>,
    nd: S,
    receiver: Receiver,
    window: f64,
    cells: usize,
    rays_per_band: usize,
) -> Caustic<S> {
    let lift = |p: Vec3<f64>| Vec3::new(S::from_f64(p.x), S::from_f64(p.y), S::from_f64(p.z));
    let shape =
        &shape.to_frame(lift(receiver.origin), lift(receiver.u), lift(receiver.v), lift(receiver.normal));
    let lamp = receiver.point(lamp);
    let (centre_s, bound_s) = shape.bounds();
    // the window: `window` is the cell budget's extent; the grid is centred on
    // where the light actually lands (a first pass on f64 at low density),
    // and grown to hold it, keeping the cell size. A pyramid throws its light
    // well past its own footprint and a fixed window silently dropped it.
    let cell = window / cells as f64;
    let (origin, cells) = {
        let (lo, hi) = landing_box(lamp, shape, nd.to_f64(), 4_000);
        let ext = ((hi[0] - lo[0]).max(hi[1] - lo[1]) + 4.0 * cell).max(window);
        let n = ((ext / cell).ceil() as usize).min(720);
        let ext = n as f64 * cell;
        let c = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
        // **Snapped to the cell lattice the receiver's own origin sits on.**
        //
        // Where the grid is placed comes out of a first pass on `f64`, so a
        // `Dual` sees it as a constant — and if it were free to slide, that
        // constant would be a lie: `gx = (hit − origin)/cell` would be
        // missing its `∂origin/∂knob` term, and the deposit would
        // re-quantise under a keyhole that is *not* moving with it. That is
        // a bias in the derivative and not a noise: it does not fall when
        // rays are added, because it is not a sampling error.
        //
        // Snapping makes the lie true. The origin can then only move a whole
        // cell at a time, and a whole cell is a *relabelling* — every cell
        // centre lands where some other cell's centre was, the reader
        // ([`crate::hint::through`] and its kin) tests those same centres
        // against the same disc, and the score does not move at all. So the
        // derivative of the grid's placement really is zero almost
        // everywhere, which is exactly what the dual carries.
        let snap = |v: f64| (v / cell).round() * cell;
        ([snap(c[0] - ext * 0.5), snap(c[1] - ext * 0.5)], n)
    };
    let mut e = vec![vec![S::ZERO; cells * cells]; BANDS.len()];
    let (mut traced, mut tir, mut outside) = (0usize, 0usize, 0usize);

    // Rays: a jittered lattice over the disc of directions that covers the
    // object's bounding sphere.
    //
    // **Aimed on `S` and not on real parts**, and that is the difference
    // between a dual that agrees with a central difference and one that does
    // not. The cone is a change of variables — the integrand is zero outside
    // the silhouette and the cone is drawn wider than that, so the domain's
    // boundary contributes nothing and the estimator is exact under it. Aim
    // it on real parts instead and a seeded knob slides the *shape* through a
    // *fixed* lattice: rays that used to hit now miss, the estimate loses
    // energy that the physics never lost, and the dual reports that loss as a
    // derivative. How badly depends on how narrow the cone is, which is how
    // small the solid is against how far away the lamp is — for the marble,
    // a few per cent; for a 220 mm lens with the lamp a kilometre out, the
    // cone is a ten-thousandth of a radian and a two-centimetre step is a
    // fifth of it, which is the whole answer.
    //
    // The solid angle carries the dual with it, because a cone that follows
    // the shape is also a cone that changes size when the shape does.
    //
    // **Reciprocals and not divisions**, everywhere the cone is built. A
    // `Dual`'s division is a multiply by the reciprocal of the divisor and an
    // `f64`'s is a real divide, so `to / dist` on the two scalars disagrees
    // in the last bit — and an ulp on the cone's axis is the initial
    // condition of a walk that reflects up to eight times inside the glass,
    // which amplifies it until a grazing ray picks the other side of total
    // internal reflection. Multiplying by a reciprocal on both scalars is the
    // same arithmetic on both, and it is what keeps the dual's real part the
    // `f64` answer to the bit.
    let lamp_s = Vec3::new(S::from_f64(lamp.x), S::from_f64(lamp.y), S::from_f64(lamp.z));
    let to = centre_s - lamp_s;
    let inv_dist = to.norm().recip();
    let axis = to * inv_dist;
    let u = if axis.x.to_f64().abs() < 0.9 { Vec3::x() } else { Vec3::y() };
    let across = axis.cross(&u);
    let e1 = across * across.norm().recip();
    let e2 = axis.cross(&e1);
    let ang = (bound_s * inv_dist).min(S::from_f64(0.999)).asin();
    let side = (rays_per_band as f64).sqrt().ceil() as usize;
    let cone_sr = S::TAU * (S::ONE - ang.cos());
    let in_disc = (side * side) as f64 * std::f64::consts::FRAC_PI_4;
    let cell_area = cell * cell;
    // the cone's solid angle per lattice ray, and the same divided by a
    // cell's area, which is what turns a ray's power into an irradiance
    let per_ray = cone_sr * S::from_f64(in_disc.recip());
    let d_omega = per_ray * S::from_f64(cell_area.recip());
    let mut caught = S::ZERO;

    for (b, (lambda, _)) in BANDS.iter().enumerate() {
        let n_glass = index::<S>(nd, *lambda);
        for i in 0..side {
            for j in 0..side {
                // a deterministic jitter per lattice cell trades the lattice's
                // moiré for fine noise; the same jitter on every call keeps
                // duals and finite differences on identical rays
                let h = (i.wrapping_mul(73856093) ^ j.wrapping_mul(19349663) ^ b.wrapping_mul(83492791)) as u32;
                let (jx, jy) = (((h & 0xffff) as f64 / 65535.0) - 0.5, (((h >> 16) & 0xffff) as f64 / 65535.0) - 0.5);
                let a = (i as f64 + 0.5 + jx) / side as f64 * 2.0 - 1.0;
                let c = (j as f64 + 0.5 + jy) / side as f64 * 2.0 - 1.0;
                if a * a + c * c > 1.0 {
                    continue;
                }
                let theta = ang * S::from_f64((a * a + c * c).sqrt());
                let phi = c.atan2(a);
                // `cos()` and `sin()` and not `sin_cos()`: on `f64` the pair
                // is one libm call and on `Dual` it is two, and an ulp
                // between them flips a grazing ray in or out of the glass.
                // The dual's real part is the `f64` answer, and it is that
                // exactly.
                let d = axis * theta.cos() + (e1 * S::from_f64(phi.cos()) + e2 * S::from_f64(phi.sin())) * theta.sin();
                let Some((t1, n1)) = shape.enter(lamp_s, d) else { continue };
                let p1 = lamp_s + d * t1;
                // the feathered aperture: one over nearly all of the glass,
                // smoothly to zero at the rim, so the sum is C¹ in whatever
                // moves the silhouette. See `glass::Shape::rim_weight`.
                let rim = shape.rim_weight(p1);
                if rim.to_f64() <= 0.0 {
                    continue;
                }
                traced += 1;
                caught += per_ray * rim;
                let Some((d_in, cos_i, cos_t)) = crate::glass::refract(d, n1, S::ONE, n_glass) else {
                    tir += 1;
                    continue;
                };
                let t_in = S::ONE - crate::glass::fresnel(S::ONE, n_glass, cos_i, cos_t);
                let Some((p2, d_out, t_out)) = crate::glass::walk_inside(shape, p1, d_in, n_glass, S::ZERO, 8) else {
                    tir += 1;
                    continue;
                };
                if d_out.z >= S::ZERO {
                    continue;
                }
                let tp = -p2.z / d_out.z;
                let hit = p2 + d_out * tp;
                let cos_plate = -d_out.z;
                let power = t_in * t_out * cos_plate * d_omega * rim;
                let gx = (hit.x - S::from_f64(origin[0])) / S::from_f64(cell) - S::HALF;
                let gy = (hit.y - S::from_f64(origin[1])) / S::from_f64(cell) - S::HALF;
                let (fx, fy) = (gx.to_f64().floor(), gy.to_f64().floor());
                if !(fx > -2.0 && fy > -2.0 && fx < cells as f64 + 1.0 && fy < cells as f64 + 1.0) {
                    outside += 1;
                    continue;
                }
                let (wx, wy) = (gx - S::from_f64(fx), gy - S::from_f64(fy));
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (ix, iy) = (fx as i64 + dx, fy as i64 + dy);
                    if ix < 0 || iy < 0 || ix >= cells as i64 || iy >= cells as i64 {
                        continue;
                    }
                    let w = (if dx == 0 { S::ONE - wx } else { wx }) * (if dy == 0 { S::ONE - wy } else { wy });
                    e[b][iy as usize * cells + ix as usize] += power * w;
                }
            }
        }
    }
    Caustic { origin, cell, n: cells, e, traced, tir, outside, caught }
}

/// Where the transmitted light lands on the plate: the 2nd..98th percentile
/// box of hit points from a low-density f64 trace at the d line.
fn landing_box(lamp: Vec3<f64>, shape: &crate::glass::Shape<impl Scalar>, nd: f64, rays: usize) -> ([f64; 2], [f64; 2]) {
    let shape = shape_f64(shape);
    let (centre, bound) = shape.bounds();
    let to = centre - lamp;
    let dist = to.norm();
    let axis = to / dist;
    let u = if axis.x.abs() < 0.9 { Vec3::x() } else { Vec3::y() };
    let e1 = axis.cross(&u).normalize();
    let e2 = axis.cross(&e1);
    let ang = (bound / dist).min(0.999).asin();
    let side = (rays as f64).sqrt().ceil() as usize;
    let n_glass = index::<f64>(nd, D_LINE_UM);
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for i in 0..side {
        for j in 0..side {
            let a = (i as f64 + 0.5) / side as f64 * 2.0 - 1.0;
            let c = (j as f64 + 0.5) / side as f64 * 2.0 - 1.0;
            if a * a + c * c > 1.0 {
                continue;
            }
            let theta = ang * (a * a + c * c).sqrt();
            let phi = c.atan2(a);
            let d = axis * theta.cos() + (e1 * phi.cos() + e2 * phi.sin()) * theta.sin();
            let Some((t1, n1)) = shape.enter(lamp, d) else { continue };
            let p1 = lamp + d * t1;
            let Some((d_in, _, _)) = crate::glass::refract(d, n1, 1.0, n_glass) else { continue };
            let Some((p2, d_out, _)) = crate::glass::walk_inside(&shape, p1, d_in, n_glass, 0.0, 8) else { continue };
            if d_out.z >= 0.0 {
                continue;
            }
            let hit = p2 + d_out * (-p2.z / d_out.z);
            xs.push(hit.x);
            ys.push(hit.y);
        }
    }
    if xs.len() < 8 {
        let hit0 = lamp + to * (-lamp.z / to.z);
        return ([hit0.x - 0.01, hit0.y - 0.01], [hit0.x + 0.01, hit0.y + 0.01]);
    }
    xs.sort_by(|a, b| a.total_cmp(b));
    ys.sort_by(|a, b| a.total_cmp(b));
    let q = |v: &[f64], f: f64| v[((v.len() - 1) as f64 * f) as usize];
    ([q(&xs, 0.02), q(&ys, 0.02)], [q(&xs, 0.98), q(&ys, 0.98)])
}

/// Any-scalar shape to f64 (reads the real parts).
fn shape_f64<S: Scalar>(shape: &crate::glass::Shape<S>) -> crate::glass::Shape<f64> {
    use crate::glass::Shape;
    let v = |p: Vec3<S>| Vec3::new(p.x.to_f64(), p.y.to_f64(), p.z.to_f64());
    match shape {
        Shape::Sphere { centre, r } => Shape::Sphere { centre: v(*centre), r: r.to_f64() },
        Shape::Convex { planes, centre, bound_r } => Shape::Convex {
            planes: planes.iter().map(|(n, d)| (v(*n), d.to_f64())).collect(),
            centre: v(*centre),
            bound_r: bound_r.to_f64(),
        },
        Shape::Capsule { a, b, r } => Shape::Capsule { a: v(*a), b: v(*b), r: r.to_f64() },
        Shape::Lens { c1, c2, r1, r2 } => Shape::Lens { c1: v(*c1), c2: v(*c2), r1: r1.to_f64(), r2: r2.to_f64() },
    }
}

impl Caustic<f64> {
    /// Bilinear irradiance at a plate point for one light band, 0 outside the window.
    pub fn at(&self, band: usize, x: f64, y: f64) -> f64 {
        let gx = (x - self.origin[0]) / self.cell - 0.5;
        let gy = (y - self.origin[1]) / self.cell - 0.5;
        if gx < 0.0 || gy < 0.0 || gx >= (self.n - 1) as f64 || gy >= (self.n - 1) as f64 {
            return 0.0;
        }
        let (ix, iy) = (gx.floor() as usize, gy.floor() as usize);
        let (wx, wy) = (gx - ix as f64, gy - iy as f64);
        let e = &self.e[band];
        e[iy * self.n + ix] * (1.0 - wx) * (1.0 - wy)
            + e[iy * self.n + ix + 1] * wx * (1.0 - wy)
            + e[(iy + 1) * self.n + ix] * (1.0 - wx) * wy
            + e[(iy + 1) * self.n + ix + 1] * wx * wy
    }
}

impl<S: Scalar> Caustic<S> {
    /// Total deposited energy per band, a scalar summary.
    pub fn total(&self, band: usize) -> S {
        self.e[band].iter().fold(S::ZERO, |a, &v| a + v)
    }
    /// A scalar that moves with the caustic's *shape*: the second moment of
    /// irradiance about the window centre. Focus tightens it, dispersion
    /// widens the rim.
    pub fn spread(&self, band: usize) -> S {
        let mut m = S::ZERO;
        let mut w = S::ZERO;
        let half = self.n as f64 / 2.0 - 0.5;
        for iy in 0..self.n {
            for ix in 0..self.n {
                let v = self.e[band][iy * self.n + ix];
                let d2 = ((ix as f64 - half) * self.cell).powi(2) + ((iy as f64 - half) * self.cell).powi(2);
                m += v * S::from_f64(d2);
                w += v;
            }
        }
        m / (w + S::from_f64(1e-12))
    }
}

/// Sum over bands and cells of (E − E_target)², the photometric loss.
pub fn loss<S: Scalar>(c: &Caustic<S>, target: &Caustic<f64>) -> S {
    let mut l = S::ZERO;
    for b in 0..BANDS.len() {
        for (v, t) in c.e[b].iter().zip(&target.e[b]) {
            let d = *v - S::from_f64(*t);
            l += d * d;
        }
    }
    l / S::from_f64((BANDS.len() * c.n * c.n) as f64)
}

/// ∂loss/∂nd by one dual pass.
pub fn loss_grad(lamp: Vec3<f64>, shape: &crate::glass::Shape<f64>, nd: f64, target: &Caustic<f64>, rays: usize) -> (f64, f64) {
    let c = trace::<Dual<f64>>(lamp, &crate::glass::to_dual(shape), Dual::new(nd, 1.0), target.cell * target.n as f64, target.n, rays);
    let l = loss(&c, target);
    (l.real, l.dual)
}

/// The caustic as an sRGB image over the window, exposure `gain` on the
/// direct-light irradiance scale (1 = the plate's own direct light).
pub fn image(c: &Caustic<f64>, gain: f64) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(c.n as u32, c.n as u32);
    for iy in 0..c.n {
        for ix in 0..c.n {
            let mut rgb = [0.0f64; 3];
            for (b, (_, w)) in BANDS.iter().enumerate() {
                let v = c.e[b][iy * c.n + ix] * gain / BANDS.len() as f64;
                for k in 0..3 {
                    rgb[k] += v * w[k];
                }
            }
            let px = |v: f64| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            // image y down; plate y up
            img.put_pixel(ix as u32, (c.n - 1 - iy) as u32, image::Rgba([px(rgb[0]), px(rgb[1]), px(rgb[2]), 255]));
        }
    }
    img
}

/// Composite the caustic into a rendered frame: plate pixels whose plate
/// point lies in the window get the caustic's RGB added on top of the
/// (grey) direct shading, with the same exposure as the direct term.
pub fn composite(
    img: &mut image::RgbaImage,
    c: &Caustic<f64>,
    pose: &phyz_camera::CameraPose,
    intr: &phyz_world::CameraIntrinsics,
    xf: &phyz_math::SpatialTransform,
    lamp_power: f64,
    scene: &crate::frame::Scene<f64>,
) -> usize {
    let mut n = 0;
    for y in 0..intr.height {
        for x in 0..intr.width {
            let (o, d) = crate::frame::primary_ray(pose, intr, x as f64 + 0.5, y as f64 + 0.5);
            let Some(p) = scene.plate_point(o, d) else { continue };
            let pl = xf.world_to_body_point(phyz_math::Vec3::new(p.x, p.y, p.z));
            let gx = (pl.x - c.origin[0]) / c.cell - 0.5;
            let gy = (pl.y - c.origin[1]) / c.cell - 0.5;
            if gx < 0.0 || gy < 0.0 || gx >= (c.n - 1) as f64 || gy >= (c.n - 1) as f64 {
                continue;
            }
            let (ix, iy) = (gx.floor() as usize, gy.floor() as usize);
            let (wx, wy) = (gx - ix as f64, gy - iy as f64);
            let mut rgb = [0.0f64; 3];
            for (b, (_, w)) in BANDS.iter().enumerate() {
                let e = &c.e[b];
                let v = e[iy * c.n + ix] * (1.0 - wx) * (1.0 - wy)
                    + e[iy * c.n + ix + 1] * wx * (1.0 - wy)
                    + e[(iy + 1) * c.n + ix] * (1.0 - wx) * wy
                    + e[(iy + 1) * c.n + ix + 1] * wx * wy;
                // irradiance × albedo × lamp power, in the frame's units
                let lin = v * lamp_power * 0.78 / BANDS.len() as f64;
                for k in 0..3 {
                    rgb[k] += lin * w[k];
                }
            }
            if rgb.iter().all(|v| *v < 1e-4) {
                continue;
            }
            let px = img.get_pixel_mut(x, y);
            for k in 0..3 {
                let base = (px.0[k] as f64 / 255.0).powf(2.2);
                px.0[k] = ((base + rgb[k]).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            }
            n += 1;
        }
    }
    n
}

pub fn out_dir(out: &Path) -> std::path::PathBuf {
    out.join("light")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glass::Shape;

    /// Plate coordinates as world coordinates for a receiver that is the
    /// wall `x = 0` facing `+x`: `(X, Y, Z)` on the plate is `(Z, X, Y)` in
    /// the world, which is the plate stood on end.
    fn on_end(p: Vec3<f64>) -> Vec3<f64> {
        Vec3::new(p.z, p.x, p.y)
    }

    fn wall() -> Receiver {
        Receiver { origin: Vec3::zero(), normal: Vec3::x(), u: Vec3::y(), v: Vec3::z() }
    }

    const ND: f64 = 1.5168;
    const WINDOW: f64 = 0.04;
    const CELLS: usize = 48;
    const RAYS: usize = 4_000;

    #[test]
    fn the_wall_receives_what_the_plate_receives() {
        let (lamp, centre, r) = (Vec3::new(0.0, 0.0, 0.4), Vec3::new(0.0, 0.0, 0.05), 0.0125);
        let flat = trace::<f64>(lamp, &Shape::Sphere { centre, r }, ND, WINDOW, CELLS, RAYS);
        let stood = trace_onto::<f64>(
            on_end(lamp),
            &Shape::Sphere { centre: on_end(centre), r },
            ND,
            wall(),
            WINDOW,
            CELLS,
            RAYS,
        );
        assert!(flat.traced > 1_000, "the plate case traced {} rays", flat.traced);
        assert!(flat.total(2) > 0.0, "the plate case caught nothing");
        assert_eq!((flat.n, flat.traced, flat.tir, flat.outside), (stood.n, stood.traced, stood.tir, stood.outside));
        assert!((flat.origin[0] - stood.origin[0]).abs() < 1e-12 && (flat.origin[1] - stood.origin[1]).abs() < 1e-12);
        for b in 0..BANDS_LEN {
            for (f, s) in flat.e[b].iter().zip(&stood.e[b]) {
                assert!((f - s).abs() <= 1e-12 * f.abs().max(1.0), "band {b}: {f} vs {s}");
            }
        }
    }

    /// The wall is not a second code path, so the dual it carries is the same
    /// dual the plate carries: `∂E/∂n_d` against central differences.
    #[test]
    fn the_wall_still_differentiates_in_the_index() {
        let (lamp, centre, r) = (Vec3::new(0.0, 0.0, 0.4), Vec3::new(0.0, 0.0, 0.05), 0.0125);
        let (lamp, centre) = (on_end(lamp), on_end(centre));
        let total = |nd: f64| {
            let c = trace_onto::<f64>(lamp, &Shape::Sphere { centre, r }, nd, wall(), WINDOW, CELLS, RAYS);
            (0..BANDS_LEN).map(|b| c.total(b)).sum::<f64>()
        };
        let shape = crate::glass::to_dual(&Shape::Sphere { centre, r });
        let c = trace_onto::<Dual<f64>>(lamp, &shape, Dual::new(ND, 1.0), wall(), WINDOW, CELLS, RAYS);
        let d: Dual<f64> = (0..BANDS_LEN).map(|b| c.total(b)).fold(Dual::constant(0.0), |a, v| a + v);
        let h = 1e-5;
        let fd = (total(ND + h) - total(ND - h)) / (2.0 * h);
        // To the bit, even though the cone is aimed on `S` now: see the
        // reciprocals `trace_onto` builds its frame out of.
        let want = total(ND);
        assert!(d.real == want, "the dual's real part is the f64 answer: {} vs {want}", d.real);
        assert!((d.dual - fd).abs() < 1e-4 * fd.abs().max(1.0), "dual {} vs central differences {fd}", d.dual);
    }
}
