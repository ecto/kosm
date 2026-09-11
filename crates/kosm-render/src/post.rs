//! What happens between the last ray and the pixel: haze, exposure, the
//! lens's own falloff, bloom, and the tonemap.
//!
//! # Why this is a module and not four lines in `Film::to_srgb8`
//!
//! There are two tiers. The path tracer resolves a frame and the rasterizer
//! draws one, and the whole settle blend rests on the two agreeing — so every
//! operation between the radiance and the byte has to be *the same operation*
//! on both, stated once. This module is the statement;
//! `kosm-view/src/raster/shaders/post.wgsl` is its port, and the constants
//! live here.
//!
//! # The four terms
//!
//! - **Aerial perspective.** A participating medium, as a closed form rather
//!   than as transport: the ray is attenuated toward the sky's own colour
//!   with an optical depth that falls off exponentially with height. The
//!   reference tracer does **not** trace this — there is no medium in the
//!   scene — so it is applied here, to the tracer's primary-hit depth, by
//!   exactly the formula the raster shader applies per fragment. It is an
//!   approximation on both tiers and it is the same approximation, which is
//!   what parity needs; a level that wanted real single scattering would put
//!   a medium in the scene and turn this off.
//! - **Exposure.** The meter's gain times the level's number, unchanged.
//! - **Vignette.** `cos⁴` of the angle off the optical axis — the physical
//!   falloff of an ideal lens, not a painted-on corner darkening. It is a
//!   function of the rig's field of view and of nothing else, so both tiers
//!   compute it from the same `tan(fov/2)`.
//! - **Bloom.** The pixels over a threshold, blurred and added back at a low
//!   weight, so a rim light and a sun glint bleed a little the way they do
//!   in a real lens.
//!
//! Order matters and is physical: the haze is in the world, the exposure is
//! the shutter, the vignette is the lens, the bloom is the glass and the
//! sensor, and the tonemap is the film.

use crate::cpu::film::{linear_to_srgb, tonemap_aces};
use crate::env::SkyEnv;
use crate::pathtrace::{Camera, Film, Projection};

/// A distance-and-height in-scatter term: the closed form both tiers apply.
///
/// The medium's density is `density · exp(-z / scale_h)`, so the optical
/// depth along a segment from `z0` to `z1` of length `d` has an exact
/// integral and there is nothing to sample:
///
/// ```text
/// τ = density · d · e^{-z₀/H} · (1 − e^{-u}) / u,   u = (z₁ − z₀) / H
/// ```
///
/// written that way rather than as the algebraically equal
/// `ρ·H·(d/Δz)·(e^{-z₀/H} − e^{-z₁/H})` because that form subtracts two
/// nearly equal exponentials and divides by a nearly zero `Δz`: in `f32`, a
/// near-horizontal view ray came out two per cent wrong, which is a visible
/// band across a beach. `(1 − e^{-u})/u` has a series at `u = 0` and no
/// cancellation anywhere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aerial {
    /// Extinction at `z = 0`, per metre. Zero switches the term off.
    pub density: f32,
    /// The height the density falls by `e` over, metres.
    pub scale_h: f32,
}

impl Default for Aerial {
    fn default() -> Self {
        // Off. A level that wants haze says so; a datasheet ball does not.
        Self { density: 0.0, scale_h: 40.0 }
    }
}

impl Aerial {
    /// Optical depth along a segment. `d` is its length in metres.
    #[inline]
    pub fn optical_depth(&self, z0: f32, z1: f32, d: f32) -> f32 {
        if self.density <= 0.0 || d <= 0.0 {
            return 0.0;
        }
        let h = self.scale_h.max(1e-3);
        let u = (z1 - z0) / h;
        // `(1 − e^{-u}) / u`, by its series where the quotient would cancel
        let f = if u.abs() < 1e-3 { 1.0 - 0.5 * u + u * u / 6.0 } else { (1.0 - (-u).exp()) / u };
        self.density * d * (-z0 / h).exp() * f
    }

    /// How much of the pixel is the medium's own glow: `1 − e^{−τ}`.
    #[inline]
    pub fn haze(&self, z0: f32, z1: f32, d: f32) -> f32 {
        1.0 - (-self.optical_depth(z0, z1, d)).exp()
    }
}

/// The whole film-to-byte chain, as data.
///
/// [`Default`] is the identity chain that `Film::to_srgb8` has always been:
/// exposure one, no haze, no vignette, no bloom. Everything else is opt-in
/// per level, which is what keeps the datasheet's parity numbers pinned while
/// the cove gets a look.
#[derive(Debug, Clone, Copy)]
pub struct Post {
    /// Linear multiplier before the tonemap: the meter's gain times the
    /// level's own number.
    pub exposure: f32,
    /// How much of the `cos⁴` falloff to apply, 0 to 1.
    pub vignette: f32,
    /// Exposed linear luminance a pixel has to pass to bleed.
    pub bloom_threshold: f32,
    /// What fraction of the blurred overshoot is added back. Zero is off.
    pub bloom_strength: f32,
    /// The blur's standard deviation, in pixels of the **full-size** frame.
    pub bloom_radius_px: f32,
    /// The haze.
    pub aerial: Aerial,
    /// The sky the haze scatters toward, if there is one. Without it the
    /// aerial term is off, because a haze with no colour is a grey wash.
    pub sky: Option<SkyEnv>,
    /// **How many of the film's world units make a metre.**
    ///
    /// [`Aerial`] is stated per metre and the film is not: a level authored in
    /// vcad's millimetres traces a scene whose `Film::depth` is millimetres
    /// and whose camera sits a thousand units off the sand. The raster tier
    /// works in metres and leaves this at one; `sims/rune` sets it to
    /// `PER_M`, and getting it wrong is a thousand-fold error in the haze
    /// rather than a subtle one — which is why it is a field and not an
    /// assumption.
    pub units_per_metre: f32,
}

impl Default for Post {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            vignette: 0.0,
            bloom_threshold: 1.0,
            bloom_strength: 0.0,
            bloom_radius_px: 4.0,
            aerial: Aerial::default(),
            sky: None,
            units_per_metre: 1.0,
        }
    }
}

/// How much light a lens at `tan(fov/2)` puts in a pixel at screen `(sx, sy)`
/// relative to one on the axis: `cos⁴θ`, lerped by `amount`.
///
/// `sx` and `sy` are the tracer's own screen coordinates — `±1` at the edges
/// — so this is the one function both tiers can call with the numbers they
/// already have.
#[inline]
pub fn vignette(amount: f32, sx: f32, sy: f32, half_w: f32, half_h: f32) -> f32 {
    if amount <= 0.0 {
        return 1.0;
    }
    let (u, v) = (sx * half_w, sy * half_h);
    // the ray is `forward + right·u + up·v` with an orthonormal basis, so
    // `cos θ = 1 / |ray|` and `cos⁴θ` is one over the square of `1 + u² + v²`
    let q = 1.0 + u * u + v * v;
    let cos4 = 1.0 / (q * q);
    1.0 + (cos4 - 1.0) * amount.clamp(0.0, 1.0)
}

/// The Gaussian weights of a nine-tap separable blur at `sigma` taps.
///
/// Nine taps and not thirteen: at the quarter resolution the bloom runs at,
/// a σ of two taps is eight full-size pixels, and past three σ the weight is
/// under a thousandth of the centre's — which is under a code once the
/// strength has divided it by ten.
pub fn gaussian9(sigma: f32) -> [f32; 9] {
    let s = sigma.max(1e-3);
    let mut w = [0.0f32; 9];
    let mut sum = 0.0f32;
    for (i, v) in w.iter_mut().enumerate() {
        let x = i as f32 - 4.0;
        *v = (-0.5 * x * x / (s * s)).exp();
        sum += *v;
    }
    for v in &mut w {
        *v /= sum;
    }
    w
}

impl Post {
    /// The chain, applied to a film, into 8-bit sRGB RGBA.
    ///
    /// `camera` is the rig the film was traced with: the vignette needs its
    /// field of view and the haze needs its eye, and both are wrong if they
    /// come from anywhere else. A camera under a projection this chain has no
    /// screen map for — the fisheye — skips the vignette and the haze rather
    /// than applying a rectilinear one to it.
    pub fn apply(&self, film: &Film, camera: &Camera, transparent: bool) -> Vec<u8> {
        let (w, h) = (film.width as usize, film.height as usize);
        let n = w * h;
        let rect = matches!(camera.projection, Projection::Rectilinear);
        let half_h = (camera.fov_deg.to_radians() * 0.5).tan() as f32;
        let half_w = half_h * w as f32 / h.max(1) as f32;

        // ── the world: haze toward the sky, then the shutter ──
        let mut lin = vec![0.0f32; n * 3];
        for j in 0..h {
            let sy = 1.0 - 2.0 * (j as f32 + 0.5) / h as f32;
            for i in 0..w {
                let sx = 2.0 * (i as f32 + 0.5) / w as f32 - 1.0;
                let k = i + j * w;
                let mut c = [film.rgb[k * 3], film.rgb[k * 3 + 1], film.rgb[k * 3 + 2]];
                if let Some(sky) =
                    self.sky.filter(|_| rect && self.aerial.density > 0.0)
                {
                        let per_m = self.units_per_metre.max(1e-6);
                        let d = film.depth[k] / per_m;
                        if d > 0.0 {
                            let dir = (camera.forward
                                + camera.right * (sx * half_w) as f64
                                + camera.up * (sy * half_h) as f64)
                                .normalize();
                            let z0 = camera.eye.z as f32 / per_m;
                            let z1 = z0 + (dir.z as f32) * d;
                            let a = self.aerial.haze(z0, z1, d);
                            let s = sky.radiance(dir);
                            for c3 in 0..3 {
                                c[c3] += (s[c3] - c[c3]) * a;
                            }
                        }
                }
                let e = self.exposure * if rect { vignette(self.vignette, sx, sy, half_w, half_h) } else { 1.0 };
                for c3 in 0..3 {
                    lin[k * 3 + c3] = c[c3] * e;
                }
            }
        }

        // ── the lens: what is over the threshold, blurred, added back ──
        if self.bloom_strength > 0.0 {
            add_bloom(&mut lin, w, h, self.bloom_threshold, self.bloom_strength, self.bloom_radius_px);
        }

        // ── the film ──
        let mut out = vec![0u8; n * 4];
        for k in 0..n {
            for c in 0..3 {
                let v = tonemap_aces(lin[k * 3 + c]);
                out[k * 4 + c] = (linear_to_srgb(v) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
            out[k * 4 + 3] = if transparent {
                (film.alpha[k] * 255.0 + 0.5).clamp(0.0, 255.0) as u8
            } else {
                255
            };
        }
        out
    }
}

/// The bloom, in place: threshold at quarter resolution, two separable
/// Gaussian passes, added back at `strength`.
///
/// Quarter resolution because a bloom is a *low* frequency by construction —
/// blurring at full size costs sixteen times as much for a picture that is
/// the same to a code — and because that is the resolution the raster tier's
/// own bloom chain runs at, so the two agree.
pub fn add_bloom(lin: &mut [f32], w: usize, h: usize, threshold: f32, strength: f32, radius_px: f32) {
    let (bw, bh) = ((w / 4).max(1), (h / 4).max(1));
    let mut bright = vec![0.0f32; bw * bh * 3];
    // the box downsample: every quarter-res texel is the mean of its sixteen
    for j in 0..bh {
        for i in 0..bw {
            let mut acc = [0.0f32; 3];
            let mut count = 0.0f32;
            for dy in 0..4 {
                let y = j * 4 + dy;
                if y >= h {
                    continue;
                }
                for dx in 0..4 {
                    let x = i * 4 + dx;
                    if x >= w {
                        continue;
                    }
                    let k = y * w + x;
                    let c = [lin[k * 3], lin[k * 3 + 1], lin[k * 3 + 2]];
                    let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                    // A soft knee would be prettier and would also make the
                    // WGSL port a second formula to keep in step; the hard
                    // threshold on luminance, scaling the colour, is one line
                    // in both.
                    let over = (l - threshold).max(0.0);
                    let s = if l > 1e-6 { over / l } else { 0.0 };
                    for c3 in 0..3 {
                        acc[c3] += c[c3] * s;
                    }
                    count += 1.0;
                }
            }
            for c3 in 0..3 {
                bright[(j * bw + i) * 3 + c3] = acc[c3] / count.max(1.0);
            }
        }
    }
    let kernel = gaussian9(radius_px / 4.0);
    let mut tmp = vec![0.0f32; bw * bh * 3];
    for j in 0..bh {
        for i in 0..bw {
            let mut acc = [0.0f32; 3];
            for (t, weight) in kernel.iter().enumerate() {
                let x = (i as isize + t as isize - 4).clamp(0, bw as isize - 1) as usize;
                for c3 in 0..3 {
                    acc[c3] += bright[(j * bw + x) * 3 + c3] * weight;
                }
            }
            tmp[(j * bw + i) * 3..(j * bw + i) * 3 + 3].copy_from_slice(&acc);
        }
    }
    for j in 0..bh {
        for i in 0..bw {
            let mut acc = [0.0f32; 3];
            for (t, weight) in kernel.iter().enumerate() {
                let y = (j as isize + t as isize - 4).clamp(0, bh as isize - 1) as usize;
                for c3 in 0..3 {
                    acc[c3] += tmp[(y * bw + i) * 3 + c3] * weight;
                }
            }
            bright[(j * bw + i) * 3..(j * bw + i) * 3 + 3].copy_from_slice(&acc);
        }
    }
    // and back up, bilinear on texel centres — the sampler's own upscale
    for j in 0..h {
        let sy = ((j as f32 + 0.5) / 4.0 - 0.5).clamp(0.0, bh as f32 - 1.0);
        let (y0, fy) = (sy.floor() as usize, sy - sy.floor());
        let y1 = (y0 + 1).min(bh - 1);
        for i in 0..w {
            let sx = ((i as f32 + 0.5) / 4.0 - 0.5).clamp(0.0, bw as f32 - 1.0);
            let (x0, fx) = (sx.floor() as usize, sx - sx.floor());
            let x1 = (x0 + 1).min(bw - 1);
            for c3 in 0..3 {
                let at = |x: usize, y: usize| bright[(y * bw + x) * 3 + c3];
                let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
                let bot = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
                lin[(j * w + i) * 3 + c3] += (top * (1.0 - fy) + bot * fy) * strength;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Point3, Vec3};

    fn flat_film(w: u32, h: u32, value: f32, depth: f32) -> Film {
        let mut f = Film::new(w, h);
        for v in f.rgb.iter_mut() {
            *v = value;
        }
        for v in f.depth.iter_mut() {
            *v = depth;
        }
        f
    }

    fn cam() -> Camera {
        Camera::look_at(
            Point3::new(0.0, -4.0, 1.6),
            Point3::new(0.0, 0.0, 1.6),
            Vec3::new(0.0, 0.0, 1.0),
            50.0,
        )
    }

    /// **The default chain is the one `to_srgb8` always was.** Exposure, ACES,
    /// sRGB, and nothing else — which is what keeps every pinned image and
    /// every parity number where it was until a level asks for more.
    #[test]
    fn the_default_post_is_the_old_tonemap() {
        let film = flat_film(8, 6, 0.4, 3.0);
        let post = Post { exposure: 0.7, ..Post::default() };
        assert_eq!(post.apply(&film, &cam(), false), film.to_srgb8(0.7, false));
    }

    /// The vignette is `cos⁴` and it is one on the axis: a corner of a 50°
    /// frame loses about a tenth of its light and the centre loses none.
    #[test]
    fn the_vignette_is_cos_to_the_fourth() {
        let half_h = (50.0f32.to_radians() * 0.5).tan();
        let half_w = half_h * 16.0 / 9.0;
        assert_eq!(vignette(1.0, 0.0, 0.0, half_w, half_h), 1.0);
        let corner = vignette(1.0, 1.0, 1.0, half_w, half_h);
        let theta = ((half_w * half_w + half_h * half_h).sqrt()).atan();
        assert!(
            (corner - theta.cos().powi(4)).abs() < 1e-6,
            "the corner is {corner}, not cos⁴ of {theta}"
        );
        // `cos⁴` is a strong falloff — the corner of a 16:9 50° frame is
        // forty-four degrees off the axis and keeps a quarter of its light —
        // which is why `amount` exists and why a level uses a third of it.
        assert!((0.25..0.30).contains(&corner), "a 50° frame keeps {corner} in the corner");
        // and `amount` fades it out
        assert_eq!(vignette(0.0, 1.0, 1.0, half_w, half_h), 1.0);
    }

    /// **The haze is stated per metre and the film may not be.** A level
    /// traced in millimetres has a depth buffer a thousand times bigger, and
    /// without `units_per_metre` its far headland would be opaque white.
    #[test]
    fn the_units_reach_the_haze() {
        let cam = cam();
        let mm = Camera { eye: Point3::new(0.0, -4000.0, 1600.0), ..cam };
        let m_film = flat_film(8, 6, 0.5, 30.0);
        let mm_film = flat_film(8, 6, 0.5, 30_000.0);
        let sky = SkyEnv::new(Vec3::new(0.0, -0.5, 0.5), 2.5, [0.4; 3], 0.2, 0.02);
        let base = Post {
            exposure: 0.7,
            aerial: Aerial { density: 0.01, scale_h: 40.0 },
            sky: Some(sky),
            ..Post::default()
        };
        let metres = base.apply(&m_film, &cam, false);
        let millimetres = Post { units_per_metre: 1000.0, ..base }.apply(&mm_film, &mm, false);
        for (a, b) in metres.iter().zip(millimetres.iter()) {
            assert!(a.abs_diff(*b) <= 1, "{a} against {b}");
        }
        // and taking the units away is a wholly different picture
        let wrong = Post { units_per_metre: 1.0, ..base }.apply(&mm_film, &mm, false);
        assert!(
            wrong.iter().zip(metres.iter()).any(|(a, b)| a.abs_diff(*b) > 40),
            "a thousandfold unit error did not change the frame"
        );
    }

    /// **The haze is a closed form and it is monotone.** Twice the distance
    /// is more haze, and a segment high in the atmosphere is less hazy than
    /// the same segment at sea level.
    #[test]
    fn the_haze_grows_with_distance_and_falls_with_height() {
        let a = Aerial { density: 0.02, scale_h: 30.0 };
        assert!(a.haze(0.0, 0.0, 40.0) > a.haze(0.0, 0.0, 20.0));
        assert!(a.haze(0.0, 0.0, 20.0) > a.haze(60.0, 60.0, 20.0));
        assert_eq!(Aerial::default().haze(0.0, 0.0, 1000.0), 0.0, "the default is off");
        // **The sloped integral is continuous at a flat ray.** The naive
        // difference-of-exponentials form was two per cent out here in `f32`,
        // which is a band across the beach where a view ray crosses level.
        let flat = a.optical_depth(5.0, 5.0, 10.0);
        for dz in [1e-5f32, 1e-4, 1e-3, 1e-2] {
            let almost = a.optical_depth(5.0, 5.0 + dz, 10.0);
            assert!(
                (flat - almost).abs() < 2e-4 + 0.02 * dz,
                "a slope of {dz} m gave {almost} against the flat {flat}"
            );
        }
    }

    /// Bloom adds light and it adds it *around* the bright pixel: a single
    /// hot texel in a black frame lights its neighbours.
    #[test]
    fn bloom_spreads_a_bright_pixel() {
        let (w, h) = (32usize, 32usize);
        let mut lin = vec![0.0f32; w * h * 3];
        for c in 0..3 {
            lin[((16 * w) + 16) * 3 + c] = 200.0;
        }
        let before = lin.clone();
        add_bloom(&mut lin, w, h, 1.0, 0.2, 4.0);
        let at = |v: &[f32], x: usize, y: usize| v[(y * w + x) * 3];
        assert!(at(&lin, 20, 16) > at(&before, 20, 16), "the neighbour did not brighten");
        assert!(at(&lin, 2, 2) < at(&lin, 20, 16), "the bloom is not local");
    }

    /// The Gaussian sums to one, or bloom would change the picture's total
    /// energy with its radius.
    #[test]
    fn the_blur_kernel_is_normalised() {
        for sigma in [0.5f32, 1.0, 2.0, 6.0] {
            let sum: f32 = gaussian9(sigma).iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "σ {sigma} sums to {sum}");
        }
    }
}
