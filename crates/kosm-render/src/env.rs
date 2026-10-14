//! Environments the renderer can build for itself: synthesised studio HDRIs,
//! and a lat-long Radiance `.hdr` reader.
//!
//! # Why the built-ins are generated rather than shipped
//!
//! The obvious move is to vendor a couple of Poly Haven CC0 HDRIs. These are
//! synthesised instead, for three reasons: the crate carries no binary blobs
//! and no third-party licence to track, the maps are exactly as
//! high-frequency as the sampler needs to be exercised (crisp softbox discs,
//! a hot rim), and a studio environment is a handful of soft rectangles in a
//! dim room — precisely the thing that is cheaper to describe than to store.
//! Real-world HDRIs are fully supported by [`parse_hdr`], which is the path
//! any Poly Haven download takes.
//!
//! # Why the `.hdr` reader is hand-written
//!
//! `image` would do it, but this crate's dependency rule is tang, rayon,
//! wgpu, bytemuck and pollster — nothing else. Radiance RGBE is a header, a
//! resolution line and either flat or run-length-encoded scanlines; the whole
//! decoder is under two hundred lines and builds for wasm without a thought.

use crate::math::Vec3;
use crate::pathtrace::EnvMap;

/// Resolution of the generated built-in maps. Enough to keep the softbox
/// edges clean without making CDF construction show up in a profile.
const BUILTIN_W: usize = 256;
const BUILTIN_H: usize = 128;

/// Sin-weighted mean luminance the built-ins are normalised to, so switching
/// between them changes the *look* and not the exposure.
pub const BUILTIN_MEAN: f32 = 0.30;

/// The generated studio environments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinEnv {
    /// Neutral three-light studio: broad key, cool fill, hot rim.
    Studio,
    /// Two crisp softboxes against near-black — a product-shot look with
    /// strong specular shapes.
    Softbox,
    /// Bright, even overcast dome. Low contrast, flattering to complex parts.
    Overcast,
}

impl BuiltinEnv {
    /// Lower-case spelling, as a CLI would take it.
    pub fn name(self) -> &'static str {
        match self {
            BuiltinEnv::Studio => "studio",
            BuiltinEnv::Softbox => "softbox",
            BuiltinEnv::Overcast => "overcast",
        }
    }

    /// Every built-in, for help text and tests.
    pub fn all() -> [BuiltinEnv; 3] {
        [
            BuiltinEnv::Studio,
            BuiltinEnv::Softbox,
            BuiltinEnv::Overcast,
        ]
    }

    /// Parse a lower-case built-in name.
    pub fn parse(s: &str) -> Option<Self> {
        BuiltinEnv::all().into_iter().find(|b| b.name() == s)
    }
}

// ─── generated studio maps ────────────────────────────────────────────────

/// A soft-edged disc of light centred on `dir`, of angular radius `radius`
/// radians.
fn disc(d: Vec3, dir: Vec3, radius: f64, radiance: [f32; 3]) -> [f32; 3] {
    let a = d.dot(dir.normalize()).clamp(-1.0, 1.0).acos();
    if a >= radius {
        return [0.0; 3];
    }
    let t = (1.0 - a / radius) as f32;
    let k = t * t * (3.0 - 2.0 * t);
    [radiance[0] * k, radiance[1] * k, radiance[2] * k]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Radiance of a built-in environment in direction `d`, before normalisation.
fn builtin_radiance(kind: BuiltinEnv, d: Vec3) -> [f32; 3] {
    let z = d.z as f32;
    match kind {
        BuiltinEnv::Studio => {
            // Dim room: slightly cool above, warm floor bounce below.
            let base = if z >= 0.0 {
                let k = z.powf(0.7);
                [0.14 + 0.05 * k, 0.15 + 0.07 * k, 0.17 + 0.11 * k]
            } else {
                let k = (-z).powf(0.5);
                [0.14 - 0.08 * k, 0.135 - 0.08 * k, 0.13 - 0.08 * k]
            };
            let key = disc(d, Vec3::new(-0.8, -1.0, 1.1), 0.34, [26.0, 25.0, 23.5]);
            let fill = disc(d, Vec3::new(1.3, -0.6, 0.25), 0.50, [2.6, 2.9, 3.5]);
            let rim = disc(d, Vec3::new(0.35, 1.25, 0.8), 0.13, [60.0, 59.0, 57.0]);
            add(add(base, key), add(fill, rim))
        }
        BuiltinEnv::Softbox => {
            // Near-black surround so the specular shapes read hard.
            let base = if z >= 0.0 {
                [0.020, 0.021, 0.024]
            } else {
                [0.012, 0.012, 0.013]
            };
            let a = disc(d, Vec3::new(-0.7, -1.0, 0.5), 0.30, [55.0, 54.0, 52.0]);
            let b = disc(d, Vec3::new(0.9, -0.9, 0.35), 0.20, [22.0, 23.0, 26.0]);
            let c = disc(d, Vec3::new(0.1, 1.1, 0.55), 0.10, [70.0, 69.0, 68.0]);
            add(add(base, a), add(b, c))
        }
        BuiltinEnv::Overcast => {
            let base = if z >= 0.0 {
                let k = z.powf(0.6);
                [0.55 + 0.45 * k, 0.57 + 0.47 * k, 0.60 + 0.50 * k]
            } else {
                let k = (-z).powf(0.5);
                [0.30 - 0.18 * k, 0.29 - 0.17 * k, 0.28 - 0.16 * k]
            };
            // A brighter break in the cloud, so there is something for the
            // CDF to find and for gloss to catch.
            let sun = disc(d, Vec3::new(-0.4, -0.7, 1.0), 0.45, [3.2, 3.2, 3.1]);
            add(base, sun)
        }
    }
}

/// Generate a built-in map, normalised to a common mean radiance.
pub fn generate(kind: BuiltinEnv) -> EnvMap {
    let (w, h) = (BUILTIN_W, BUILTIN_H);
    let mut pixels = vec![[0.0f32; 3]; w * h];
    // Solid-angle-weighted mean, so normalisation is over the sphere and not
    // over image area (which would over-count the poles).
    let mut weighted = 0.0f64;
    let mut weight = 0.0f64;

    for j in 0..h {
        let theta = core::f64::consts::PI * (j as f64 + 0.5) / h as f64;
        let (st, ct) = theta.sin_cos();
        for i in 0..w {
            let phi = core::f64::consts::TAU * (i as f64 + 0.5) / w as f64;
            let d = Vec3::new(st * phi.cos(), st * phi.sin(), ct);
            let c = builtin_radiance(kind, d);
            pixels[j * w + i] = c;
            let lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
            weighted += lum as f64 * st;
            weight += st;
        }
    }

    let mean = if weight > 0.0 {
        (weighted / weight) as f32
    } else {
        0.0
    };
    if mean > 0.0 {
        let k = BUILTIN_MEAN / mean;
        for p in &mut pixels {
            p[0] *= k;
            p[1] *= k;
            p[2] *= k;
        }
    }

    EnvMap::new(w, h, pixels).expect("built-in environment dimensions are valid")
}

// ─── Radiance .hdr ────────────────────────────────────────────────────────

/// One RGBE quadruple to linear RGB.
///
/// The `e == 0` case is exact zero, not `2^-128`: Radiance writes an all-zero
/// pixel that way and the naive formula would turn it into a denormal.
#[inline]
fn rgbe_to_rgb(p: [u8; 4]) -> [f32; 3] {
    if p[3] == 0 {
        return [0.0; 3];
    }
    let f = libm_exp2(p[3] as i32 - (128 + 8));
    [
        (p[0] as f32 + 0.5) * f,
        (p[1] as f32 + 0.5) * f,
        (p[2] as f32 + 0.5) * f,
    ]
}

/// `2^n` for a small signed `n`, without `powi` on a wasm target's libm.
#[inline]
fn libm_exp2(n: i32) -> f32 {
    let n = n.clamp(-126, 127);
    f32::from_bits((((n + 127) as u32) & 0xFF) << 23)
}

/// Decode a lat-long Radiance `.hdr` (RGBE) image from its bytes.
///
/// The image is taken as equirectangular with row 0 at the zenith, which is
/// the convention every HDRI library ships. Both scanline encodings are
/// handled: flat RGBE, and the "new" adaptive run-length encoding.
pub fn parse_hdr(bytes: &[u8]) -> Result<EnvMap, String> {
    // ── header ──
    let mut i = 0usize;
    let mut line = Vec::new();
    let mut saw_magic = false;
    let resolution: (usize, usize);

    loop {
        line.clear();
        while i < bytes.len() && bytes[i] != b'\n' {
            line.push(bytes[i]);
            i += 1;
        }
        if i >= bytes.len() {
            return Err("truncated Radiance header".into());
        }
        i += 1; // the newline
        let s = String::from_utf8_lossy(&line).trim().to_string();
        if line.is_empty() || s.is_empty() {
            // Blank line ends the header; the resolution line is next.
            line.clear();
            while i < bytes.len() && bytes[i] != b'\n' {
                line.push(bytes[i]);
                i += 1;
            }
            if i >= bytes.len() {
                return Err("truncated Radiance header".into());
            }
            i += 1;
            let s = String::from_utf8_lossy(&line).trim().to_string();
            let parts: Vec<&str> = s.split_whitespace().collect();
            // Only the standard `-Y h +X w` orientation is accepted: every
            // other flip would need the scanlines reordered, and no HDRI
            // library writes one.
            if parts.len() != 4 || parts[0] != "-Y" || parts[2] != "+X" {
                return Err(format!("unsupported Radiance resolution line: {s:?}"));
            }
            let h: usize = parts[1]
                .parse()
                .map_err(|_| format!("bad height in {s:?}"))?;
            let w: usize = parts[3]
                .parse()
                .map_err(|_| format!("bad width in {s:?}"))?;
            resolution = (w, h);
            break;
        }
        if s.starts_with("#?") {
            saw_magic = true;
        }
        if let Some(fmt) = s.strip_prefix("FORMAT=") {
            if fmt.trim() != "32-bit_rle_rgbe" {
                return Err(format!("unsupported Radiance FORMAT={fmt}"));
            }
        }
    }
    if !saw_magic {
        return Err("not a Radiance file (no #? magic)".into());
    }
    let (w, h) = resolution;
    if w < 2 || h < 1 {
        return Err("environment map too small".into());
    }

    // ── scanlines ──
    let mut pixels = vec![[0.0f32; 3]; w * h];
    let mut row = vec![[0u8; 4]; w];

    for y in 0..h {
        read_scanline(bytes, &mut i, &mut row)?;
        for (x, p) in row.iter().enumerate() {
            pixels[y * w + x] = rgbe_to_rgb(*p);
        }
    }

    EnvMap::new(w, h, pixels)
}

/// One scanline, either encoding, into `row`.
fn read_scanline(bytes: &[u8], i: &mut usize, row: &mut [[u8; 4]]) -> Result<(), String> {
    let w = row.len();
    let need = |i: usize, n: usize| -> Result<(), String> {
        if i + n > bytes.len() {
            Err("truncated Radiance scanline".into())
        } else {
            Ok(())
        }
    };
    need(*i, 4)?;
    let head = [bytes[*i], bytes[*i + 1], bytes[*i + 2], bytes[*i + 3]];
    let rle_len = ((head[2] as usize) << 8) | head[3] as usize;

    if !(head[0] == 2
        && head[1] == 2
        && head[2] & 0x80 == 0
        && rle_len == w
        && (4..=0x7fff).contains(&w))
    {
        // Flat RGBE, one quadruple per pixel. (Old-style RLE — a 1,1,1,n
        // repeat marker — is not produced by any modern writer and is not
        // handled; a file using it fails the pixel count rather than
        // decoding to garbage.)
        need(*i, 4 * w)?;
        for p in row.iter_mut() {
            *p = [bytes[*i], bytes[*i + 1], bytes[*i + 2], bytes[*i + 3]];
            *i += 4;
        }
        return Ok(());
    }
    *i += 4;

    // New-style adaptive RLE: the four components are stored in separate
    // runs, each either a literal block (count <= 128) or a repeat.
    for c in 0..4 {
        let mut x = 0usize;
        while x < w {
            need(*i, 1)?;
            let n = bytes[*i] as usize;
            *i += 1;
            if n > 128 {
                need(*i, 1)?;
                let v = bytes[*i];
                *i += 1;
                let run = n - 128;
                if x + run > w {
                    return Err("Radiance run overruns the scanline".into());
                }
                for _ in 0..run {
                    row[x][c] = v;
                    x += 1;
                }
            } else {
                if n == 0 {
                    return Err("zero-length Radiance run".into());
                }
                need(*i, n)?;
                if x + n > w {
                    return Err("Radiance run overruns the scanline".into());
                }
                for _ in 0..n {
                    row[x][c] = bytes[*i];
                    *i += 1;
                    x += 1;
                }
            }
        }
    }
    Ok(())
}

/// Read a lat-long Radiance `.hdr` file from disk. Native only.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_hdr(path: &std::path::Path) -> Result<EnvMap, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_hdr(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

// ─── the analytic clear sky ───────────────────────────────────────────────

/// A Preetham clear-sky model: warm horizon, blue zenith, a glow around the
/// sun, and a ground half below.
///
/// # Why this and not a gradient
///
/// [`GradientEnv`](crate::pathtrace::GradientEnv) is two colours and a lerp.
/// It cannot know where the sun is, so it cannot put the bright, desaturated
/// aureole around it that a real sky has, and it cannot warm the horizon in
/// the sun's own azimuth while leaving the opposite one cold. Those two
/// asymmetries are most of what makes a late-afternoon beach read as a
/// *place* rather than as a studio backdrop, and they cost eight polynomial
/// evaluations per lookup.
///
/// Preetham rather than Hosek–Wilkie on purpose: Hosek is nine coefficients
/// per channel out of a fitted table, which is a data blob this crate's
/// dependency rule would have to carry. Preetham is five Perez coefficients
/// per channel, each an affine function of turbidity, and the whole model is
/// forty lines that port to WGSL character for character — which is what the
/// raster tier needs, because the two tiers have to agree about the sky or
/// the settle blend has a seam at the horizon.
///
/// # The sun disc is **not** in here
///
/// [`Sun`](crate::pathtrace::Sun) owns the disc, on both tiers, and this
/// model excludes it: the Perez angle `γ` is clamped to
/// [`SkyEnv::sun_radius`] before it is evaluated, which caps the circumsolar
/// term at the value it has on the disc's own rim. So the glow is here and
/// the disc is the sun's, exactly once, and a camera ray that lands in the
/// cone gets `sky + Sun::radiance_in` with no double count.
///
/// # Units
///
/// Preetham's zenith luminance is in kcd/m², which is nobody's render
/// units. The model is normalised at construction so that the **mean
/// radiance over the upper hemisphere is one**, and [`SkyEnv::intensity`] is
/// then that mean in the tracer's own units — the same number
/// `sky_intensity` always was. Turbidity therefore changes the sky's *shape*
/// and *colour* without changing the exposure, which is what makes it a knob
/// a level author can turn.
///
/// ```
/// use kosm_render::env::SkyEnv;
/// use kosm_render::math::Vec3;
/// let sky = SkyEnv::new(Vec3::new(-0.35, -0.45, 0.42), 2.5, [0.42, 0.36, 0.24], 0.42, 0.02);
/// let zenith = sky.radiance(Vec3::new(0.0, 0.0, 1.0));
/// let horizon = sky.radiance(Vec3::new(1.0, 0.0, 0.02));
/// // the zenith is blue and the horizon is not
/// assert!(zenith[2] / zenith[0] > horizon[2] / horizon[0]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SkyEnv {
    /// Unit vector **toward** the sun.
    pub sun_dir: Vec3,
    /// Preetham's turbidity: 2 is an exceptionally clear day, 3 a normal
    /// clear one, 6 hazy, 10 the kind of murk that turns the sky white.
    pub turbidity: f32,
    /// Linear-RGB albedo of the ground half — what a downward ray finds.
    pub ground_albedo: [f32; 3],
    /// Mean radiance over the upper hemisphere, in the tracer's units.
    pub intensity: f32,
    /// The sun's angular radius, radians. The circumsolar term is clamped at
    /// it so the disc belongs to [`Sun`](crate::pathtrace::Sun) alone.
    pub sun_radius: f32,
    /// The normaliser that makes the mean upper-hemisphere luminance one.
    /// Derived by [`SkyEnv::new`]; the raster tier uploads it so the two
    /// tiers scale the same model by the same number.
    pub scale: f32,
}

/// The five Perez coefficients of one channel, as affine functions of
/// turbidity — the model's whole parameterisation.
#[inline]
fn perez_coeffs(t: f32) -> ([f32; 5], [f32; 5], [f32; 5]) {
    (
        // Y (luminance)
        [
            0.1787 * t - 1.4630,
            -0.3554 * t + 0.4275,
            -0.0227 * t + 5.3251,
            0.1206 * t - 2.5771,
            -0.0670 * t + 0.3703,
        ],
        // x
        [
            -0.0193 * t - 0.2592,
            -0.0665 * t + 0.0008,
            -0.0004 * t + 0.2125,
            -0.0641 * t - 0.8989,
            -0.0033 * t + 0.0452,
        ],
        // y
        [
            -0.0167 * t - 0.2608,
            -0.0950 * t + 0.0092,
            -0.0079 * t + 0.2102,
            -0.0441 * t - 1.6537,
            -0.0109 * t + 0.0529,
        ],
    )
}

/// Perez's five-parameter sky function, `F(θ, γ)`.
///
/// `cos_theta` is the cosine of the angle from the zenith and `gamma` the
/// angle from the sun. The `1/cosθ` in the first factor is why a direction on
/// the horizon needs a floor under it: at `cosθ = 0` the exponential is
/// either zero or infinite depending on the sign of `B`, and the model is not
/// defined there.
#[inline]
fn perez(c: &[f32; 5], cos_theta: f32, gamma: f32) -> f32 {
    let ct = cos_theta.max(0.01);
    let cg = gamma.cos();
    (1.0 + c[0] * (c[1] / ct).exp()) * (1.0 + c[2] * (c[3] * gamma).exp() + c[4] * cg * cg)
}

/// Preetham's zenith chromaticity and luminance at a solar zenith angle.
fn zenith(t: f32, theta_s: f32) -> (f32, f32, f32) {
    let (t2, ts) = (t * t, theta_s);
    let (ts2, ts3) = (ts * ts, ts * ts * ts);
    let x = t2 * (0.00166 * ts3 - 0.00375 * ts2 + 0.00209 * ts)
        + t * (-0.02903 * ts3 + 0.06377 * ts2 - 0.03202 * ts + 0.00394)
        + (0.11693 * ts3 - 0.21196 * ts2 + 0.06052 * ts + 0.25886);
    let y = t2 * (0.00275 * ts3 - 0.00610 * ts2 + 0.00317 * ts)
        + t * (-0.04214 * ts3 + 0.08970 * ts2 - 0.04153 * ts + 0.00516)
        + (0.15346 * ts3 - 0.26756 * ts2 + 0.06670 * ts + 0.26688);
    let chi = (4.0 / 9.0 - t / 120.0) * (core::f32::consts::PI - 2.0 * ts);
    let lum = (4.0453 * t - 4.9710) * chi.tan() - 0.2155 * t + 2.4192;
    (x, y, lum.max(0.05))
}

/// CIE xyY to linear sRGB, with the luminance carried through unchanged.
///
/// The matrix is written to its published eight figures rather than to the
/// seven an `f32` can hold, because `shaders/scene.wgsl` has the same nine
/// literals and the two have to be read as the same matrix by a person.
#[allow(clippy::excessive_precision)]
#[inline]
fn xyy_to_rgb(x: f32, y: f32, big_y: f32) -> [f32; 3] {
    let y = y.max(1e-4);
    let (xx, zz) = (x / y * big_y, (1.0 - x - y) / y * big_y);
    [
        3.2404542 * xx - 1.5371385 * big_y - 0.4985314 * zz,
        -0.9692660 * xx + 1.8760108 * big_y + 0.0415560 * zz,
        0.0556434 * xx - 0.2040259 * big_y + 1.0572252 * zz,
    ]
}

impl SkyEnv {
    /// A sky under a sun at `sun_dir`, normalised to `intensity`.
    pub fn new(
        sun_dir: Vec3,
        turbidity: f32,
        ground_albedo: [f32; 3],
        intensity: f32,
        sun_radius: f32,
    ) -> Self {
        let mut me = Self {
            sun_dir: sun_dir.normalize(),
            turbidity: turbidity.clamp(1.7, 10.0),
            ground_albedo,
            intensity,
            sun_radius: sun_radius.clamp(1e-4, 0.5),
            scale: 1.0,
        };
        me.scale = 1.0 / me.mean_upper().max(1e-6);
        me
    }

    /// Solid-angle-weighted mean of the raw (unscaled) model over the upper
    /// hemisphere. Cheap enough to do at construction — a 32×128 quadrature
    /// is four thousand Perez evaluations, once.
    fn mean_upper(&self) -> f32 {
        let mut bare = *self;
        bare.scale = 1.0;
        bare.intensity = 1.0;
        let (mut sum, mut weight) = (0.0f64, 0.0f64);
        let (nj, ni) = (32usize, 128usize);
        for j in 0..nj {
            let theta = 0.5 * core::f64::consts::PI * (j as f64 + 0.5) / nj as f64;
            let (st, ct) = theta.sin_cos();
            for i in 0..ni {
                let phi = core::f64::consts::TAU * (i as f64 + 0.5) / ni as f64;
                let d = Vec3::new(st * phi.cos(), st * phi.sin(), ct);
                let c = bare.upper(d);
                sum += (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) as f64 * st;
                weight += st;
            }
        }
        if weight > 0.0 { (sum / weight) as f32 } else { 1.0 }
    }

    /// The model above the horizon, before [`Self::scale`] and
    /// [`Self::intensity`].
    fn upper(&self, d: Vec3) -> [f32; 3] {
        let (cy, cx, cyy) = perez_coeffs(self.turbidity);
        let cos_theta = d.z.max(0.0) as f32;
        let cos_theta_s = self.sun_dir.z.clamp(-1.0, 1.0) as f32;
        let theta_s = cos_theta_s.max(0.0).acos();
        // The disc belongs to `Sun`: clamping γ here caps the circumsolar
        // term at the value it takes on the disc's rim, so what is left is
        // the aureole and nothing else.
        let gamma = (d.normalize().dot(self.sun_dir).clamp(-1.0, 1.0) as f32)
            .acos()
            .max(self.sun_radius);
        let (xz, yz, lz) = zenith(self.turbidity, theta_s);
        let f0 = |c: &[f32; 5]| perez(c, 1.0, theta_s);
        let big_y = lz * perez(&cy, cos_theta, gamma) / f0(&cy).max(1e-4);
        let x = xz * perez(&cx, cos_theta, gamma) / f0(&cx).max(1e-4);
        let y = yz * perez(&cyy, cos_theta, gamma) / f0(&cyy).max(1e-4);
        let c = xyy_to_rgb(x, y, big_y.max(0.0));
        [c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)]
    }

    /// Radiance from `d`, in the tracer's units. Below the horizon this is
    /// the ground: the sky at the horizon in that azimuth, fading into the
    /// albedo's own colour at the nadir.
    pub fn radiance(&self, d: Vec3) -> [f32; 3] {
        let k = self.scale * self.intensity;
        if d.z >= 0.0 {
            let c = self.upper(d);
            return [c[0] * k, c[1] * k, c[2] * k];
        }
        // The horizon in the same azimuth, so the ground and the sky meet
        // rather than step.
        let flat = Vec3::new(d.x, d.y, 0.02).normalize();
        let h = self.upper(flat);
        let t = ((-d.z as f32).sqrt()).clamp(0.0, 1.0);
        let s = t * t * (3.0 - 2.0 * t);
        // Half the horizon's radiance is what a Lambertian ground of albedo
        // one returns under a sky of that radiance and the sun behind it —
        // near enough for a term nothing in the frame looks at directly.
        let g = [
            self.ground_albedo[0] * h[0],
            self.ground_albedo[1] * h[1],
            self.ground_albedo[2] * h[2],
        ];
        [
            (h[0] + (g[0] - h[0]) * s) * k,
            (h[1] + (g[1] - h[1]) * s) * k,
            (h[2] + (g[2] - h[2]) * s) * k,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every built-in must be sampleable and normalised to the same exposure,
    /// or switching environments would silently change image brightness.
    #[test]
    fn builtins_share_an_exposure() {
        for kind in BuiltinEnv::all() {
            let map = generate(kind);
            let mut weighted = 0.0f64;
            let mut weight = 0.0f64;
            let n = 200;
            for j in 0..n {
                let theta = core::f64::consts::PI * (j as f64 + 0.5) / n as f64;
                let (st, ct) = theta.sin_cos();
                for i in 0..n {
                    let phi = core::f64::consts::TAU * (i as f64 + 0.5) / n as f64;
                    let d = Vec3::new(st * phi.cos(), st * phi.sin(), ct);
                    let c = map.radiance(d);
                    let lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                    weighted += lum as f64 * st;
                    weight += st;
                }
            }
            let mean = weighted / weight;
            assert!(
                (mean - BUILTIN_MEAN as f64).abs() < 0.05,
                "{} has mean radiance {mean}, expected ~{BUILTIN_MEAN}",
                kind.name()
            );
        }
    }

    /// The built-ins exist to exercise the importance sampler, so they must
    /// actually be high-frequency: far brighter somewhere than on average.
    #[test]
    fn builtins_are_high_frequency() {
        for kind in BuiltinEnv::all() {
            let map = generate(kind);
            let mut peak = 0.0f32;
            let n = 240;
            for j in 0..n {
                let theta = core::f64::consts::PI * (j as f64 + 0.5) / n as f64;
                let (st, ct) = theta.sin_cos();
                for i in 0..n {
                    let phi = core::f64::consts::TAU * (i as f64 + 0.5) / n as f64;
                    let d = Vec3::new(st * phi.cos(), st * phi.sin(), ct);
                    peak = peak.max(map.radiance(d)[0]);
                }
            }
            assert!(
                peak > 4.0 * BUILTIN_MEAN,
                "{} peak radiance {peak} is too flat to need a CDF",
                kind.name()
            );
        }
    }



    /// The sky's shape, in the four statements that make it a sky and not a
    /// wash: the zenith is bluer than the horizon, the sun's own side of the
    /// sky is brighter than the far side, the aureole is brighter still, and
    /// nothing anywhere is negative.
    #[test]
    fn the_sky_is_blue_above_and_warm_toward_the_sun() {
        use super::SkyEnv;
        let d = Vec3::new(-0.35, -0.45, 0.42).normalize();
        let sky = SkyEnv::new(d, 2.5, [0.42, 0.36, 0.24], 0.42, 0.02);
        let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let zenith = sky.radiance(Vec3::new(0.0, 0.0, 1.0));
        let near = sky.radiance(Vec3::new(d.x, d.y, 0.05).normalize());
        let far = sky.radiance(Vec3::new(-d.x, -d.y, 0.05).normalize());
        let aureole = sky.radiance((d + Vec3::new(0.05, 0.0, 0.0)).normalize());
        assert!(zenith[2] / zenith[0] > near[2] / near[0], "the zenith is not the blue end");
        assert!(lum(near) > lum(far), "the sun's own horizon is not the bright one");
        assert!(lum(aureole) > lum(zenith), "there is no glow around the sun");
        for v in [zenith, near, far, aureole, sky.radiance(Vec3::new(0.0, 0.0, -1.0))] {
            assert!(v.iter().all(|c| *c >= 0.0 && c.is_finite()), "{v:?}");
        }
    }

    /// **Turbidity changes the look and not the exposure.** The model is
    /// normalised to a mean radiance over the upper hemisphere, so a level
    /// author can turn the haze up without the frame going dark — which is
    /// the only reason it is a knob rather than a constant.
    #[test]
    fn turbidity_holds_the_exposure() {
        use super::SkyEnv;
        let d = Vec3::new(-0.35, -0.45, 0.42).normalize();
        for t in [2.0f32, 2.5, 4.0, 7.0] {
            let sky = SkyEnv::new(d, t, [0.4; 3], 0.42, 0.02);
            let (mut sum, mut weight) = (0.0f64, 0.0f64);
            let n = 120;
            for j in 0..n {
                let theta = 0.5 * core::f64::consts::PI * (j as f64 + 0.5) / n as f64;
                let (st, ct) = theta.sin_cos();
                for i in 0..n {
                    let phi = core::f64::consts::TAU * (i as f64 + 0.5) / n as f64;
                    let c = sky.radiance(Vec3::new(st * phi.cos(), st * phi.sin(), ct));
                    sum += (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) as f64 * st;
                    weight += st;
                }
            }
            let mean = sum / weight;
            assert!(
                (mean - 0.42).abs() < 0.02,
                "turbidity {t} gives mean radiance {mean:.4}, not the 0.42 it was asked for"
            );
        }
    }

    /// **The disc is the sun's, not the sky's.** The model's circumsolar term
    /// is clamped at the sun's own angular radius, so looking straight at the
    /// sun through the sky alone gives the rim's value and not a spike — and
    /// `Sun::radiance_in` is free to add the disc exactly once.
    #[test]
    fn the_sky_leaves_the_disc_to_the_sun() {
        use super::SkyEnv;
        let d = Vec3::new(-0.3, -0.4, 0.5).normalize();
        let sky = SkyEnv::new(d, 2.5, [0.4; 3], 0.42, 0.02);
        let centre = sky.radiance(d);
        // a direction on the disc's own rim: the sun tilted by its radius
        let side = d.cross(Vec3::new(0.0, 0.0, 1.0)).normalize();
        let rim = sky.radiance((d + side * 0.02).normalize());
        for c in 0..3 {
            assert!(
                (centre[c] - rim[c]).abs() < 0.02 * centre[c].max(1e-3),
                "the sky has its own disc: centre {centre:?} against rim {rim:?}"
            );
        }
    }

    #[test]
    fn names_round_trip() {
        for kind in BuiltinEnv::all() {
            assert_eq!(BuiltinEnv::parse(kind.name()), Some(kind));
        }
        assert_eq!(BuiltinEnv::parse("nope"), None);
    }

    /// A flat-RGBE file, written by hand, decodes to the values that went in.
    #[test]
    fn decodes_a_flat_hdr() {
        let (w, h) = (4usize, 2usize);
        let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n".to_vec();
        for _ in 0..(w * h) {
            // (128,128,128) at exponent 128+1 => (128.5/256)*2 ≈ 1.00390625.
            bytes.extend_from_slice(&[128, 128, 128, 129]);
        }
        let map = parse_hdr(&bytes).expect("decodes");
        let c = map.radiance(Vec3::new(0.0, 0.0, 1.0));
        assert!((c[0] - 1.00390625).abs() < 1e-5, "got {c:?}");
    }

    /// The adaptive RLE path: a repeat run per channel across a wide row.
    #[test]
    fn decodes_new_style_rle() {
        let (w, h) = (8usize, 2usize);
        let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 8\n".to_vec();
        for _ in 0..h {
            bytes.extend_from_slice(&[2, 2, (w >> 8) as u8, (w & 0xFF) as u8]);
            for v in [64u8, 128, 192, 129] {
                bytes.push(128 + w as u8); // a repeat of `w`
                bytes.push(v);
            }
        }
        let map = parse_hdr(&bytes).expect("decodes");
        let c = map.radiance(Vec3::new(0.0, 0.0, 1.0));
        assert!((c[0] - (64.5 / 256.0 * 2.0)).abs() < 1e-5, "got {c:?}");
        assert!((c[2] - (192.5 / 256.0 * 2.0)).abs() < 1e-5, "got {c:?}");
    }

    /// A literal-run scanline, and a mixed one, must decode too.
    #[test]
    fn decodes_literal_runs() {
        let w = 4usize;
        let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 4\n".to_vec();
        bytes.extend_from_slice(&[2, 2, 0, 4]);
        for base in [10u8, 20, 30, 128] {
            bytes.push(w as u8); // literal block of 4
            for k in 0..w {
                bytes.push(base.wrapping_add(k as u8));
            }
        }
        let map = parse_hdr(&bytes).expect("decodes");
        assert_eq!(map.width(), 4);
        assert_eq!(map.height(), 1);
    }

    #[test]
    fn rejects_a_non_radiance_file() {
        let err = parse_hdr(b"PNG\n\n-Y 1 +X 2\n").expect_err("must fail");
        assert!(err.contains("Radiance"), "unhelpful: {err}");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_missing_file_is_a_clean_error() {
        let err = load_hdr(std::path::Path::new("/nope/missing.hdr")).expect_err("must fail");
        assert!(err.contains("missing.hdr"), "unhelpful: {err}");
    }
}
