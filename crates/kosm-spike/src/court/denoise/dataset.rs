//! The training set: the court, rendered noisy and rendered right.
//!
//! # What one sample is
//!
//! A sample is one (camera, time) state of the court, rendered at one size.
//! It carries three things:
//!
//! * **The noisy tiers.** Five running means over 1, 2, 4, 8 and 16
//!   independent one-sample-per-pixel passes, each with the Monte Carlo
//!   estimator's own variance. These are exactly what
//!   `kosm_render::gpu::history`'s `mean` and `stats` buffers hold after that
//!   many accumulated passes — a running mean of unweighted samples and the
//!   variance of that mean — so a network fitted here sees at training time
//!   what it will be handed at inference time. The prefix structure is not an
//!   economy: the 16 passes *are* the 8, plus 8 more, which is what the
//!   device's history is.
//! * **The guides.** Normal, distance and denoise albedo at each pixel's
//!   first hit, the same three planes the à-trous filter is steered by.
//!   Taken off the 16-pass film, where they are least noisy; they are
//!   primary-hit quantities and barely move between tiers.
//! * **The reference.** A 1024-spp render with the filter off. This is the
//!   whole point: no denoiser trained on a general corpus has ever seen this
//!   gym, and we can make as much of its exact answer as we are willing to
//!   wait for.
//!
//! # The file
//!
//! One header, then one block per sample, every plane f16 and every plane
//! whole-frame. Crops are taken at load time rather than baked in, so the
//! same file trains a 64x64 tile network and evaluates a full frame, and a
//! change of tile size is not a regeneration.
//!
//! f16 is not a compromise here. Radiance in this gym runs from about 1e-3 to
//! a few hundred, well inside f16's range, and its 11-bit significand is
//! finer than the noise on a 1024-spp estimate of any of it.

use std::io::{Read, Write};
use std::path::Path;

use kosm_render::pathtrace::{Camera, Film, PathTraceOptions};
use vcad_kernel_raytrace::pathtrace::Scene as PtScene;
use vcad_kernel_math::{Point3, Vec3};

use crate::court::render::{self, Snapshot};
use crate::court::{Court, CourtScene};

/// The accumulated pass counts each sample is rendered at.
///
/// Powers of two up to 16 because that is the interesting part of the curve:
/// a pixel's history is short exactly when the filter is doing work, and
/// `GpuDenoiseParams::count_cutoff` fades the filter out by 32 anyway.
pub const TIERS: [u32; 5] = [1, 2, 4, 8, 16];

/// Planes per tier: mean radiance (3) and the variance of that mean (1).
const TIER_PLANES: usize = 4;
/// Guide planes: normal (3), depth (1), albedo (3).
const GUIDE_PLANES: usize = 7;
/// Reference planes: converged radiance (3).
const REF_PLANES: usize = 3;

/// Every f16 plane one sample carries.
pub const SAMPLE_PLANES: usize = TIERS.len() * TIER_PLANES + GUIDE_PLANES + REF_PLANES;

const MAGIC: &[u8; 8] = b"KOSMDN01";

/// One sampled state of the court, all planes f16 and row-major.
#[derive(Clone)]
pub struct Sample {
    pub width: u32,
    pub height: u32,
    /// `TIERS.len()` running means, 3 f16 per pixel each.
    pub mean: Vec<Vec<u16>>,
    /// The variance of each of those means, 1 f16 per pixel.
    pub variance: Vec<Vec<u16>>,
    /// World normal at the first hit, 3 f16 per pixel.
    pub normal: Vec<u16>,
    /// Distance to the first hit; zero is the background sentinel.
    pub depth: Vec<u16>,
    /// Denoise albedo, 3 f16 per pixel.
    pub albedo: Vec<u16>,
    /// The 1024-spp answer, 3 f16 per pixel.
    pub reference: Vec<u16>,
}

impl Sample {
    fn pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

/// A whole dataset, in memory.
pub struct Dataset {
    pub width: u32,
    pub height: u32,
    pub samples: Vec<Sample>,
}

impl Dataset {
    /// Bytes on disk, as [`Dataset::save`] writes them.
    pub fn bytes(&self) -> usize {
        32 + self.samples.len() * (self.width as usize) * (self.height as usize) * SAMPLE_PLANES * 2
    }

    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(MAGIC)?;
        f.write_all(&(self.width).to_le_bytes())?;
        f.write_all(&(self.height).to_le_bytes())?;
        f.write_all(&(self.samples.len() as u32).to_le_bytes())?;
        f.write_all(&(TIERS.len() as u32).to_le_bytes())?;
        f.write_all(&[0u8; 8])?;
        for s in &self.samples {
            for k in 0..TIERS.len() {
                write_u16(&mut f, &s.mean[k])?;
                write_u16(&mut f, &s.variance[k])?;
            }
            write_u16(&mut f, &s.normal)?;
            write_u16(&mut f, &s.depth)?;
            write_u16(&mut f, &s.albedo)?;
            write_u16(&mut f, &s.reference)?;
        }
        f.flush()
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut head = [0u8; 32];
        f.read_exact(&mut head)?;
        if &head[0..8] != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a kosm denoise dataset",
            ));
        }
        let width = u32::from_le_bytes(head[8..12].try_into().unwrap());
        let height = u32::from_le_bytes(head[12..16].try_into().unwrap());
        let n = u32::from_le_bytes(head[16..20].try_into().unwrap()) as usize;
        let tiers = u32::from_le_bytes(head[20..24].try_into().unwrap()) as usize;
        if tiers != TIERS.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "tier count does not match this build",
            ));
        }
        let px = (width as usize) * (height as usize);
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let mut mean = Vec::with_capacity(tiers);
            let mut variance = Vec::with_capacity(tiers);
            for _ in 0..tiers {
                mean.push(read_u16(&mut f, px * 3)?);
                variance.push(read_u16(&mut f, px)?);
            }
            samples.push(Sample {
                width,
                height,
                mean,
                variance,
                normal: read_u16(&mut f, px * 3)?,
                depth: read_u16(&mut f, px)?,
                albedo: read_u16(&mut f, px * 3)?,
                reference: read_u16(&mut f, px * 3)?,
            });
        }
        Ok(Self {
            width,
            height,
            samples,
        })
    }
}

fn write_u16<W: Write>(w: &mut W, v: &[u16]) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(v.len() * 2);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&buf)
}

fn read_u16<R: Read>(r: &mut R, n: usize) -> std::io::Result<Vec<u16>> {
    let mut buf = vec![0u8; n * 2];
    r.read_exact(&mut buf)?;
    Ok(buf
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

// ─── f16, by hand ─────────────────────────────────────────────────────────
//
// Half a dozen lines against a dependency, and the rounding is
// round-to-nearest-even like everything else.

/// f32 to IEEE binary16, round to nearest even, saturating to infinity.
pub fn f16_from_f32(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        // inf or NaN; a NaN keeps a non-zero mantissa so it stays a NaN
        return sign | 0x7c00 | if man != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // subnormal: shift the implicit one back in and round
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rounded = m + half - 1 + ((m >> shift) & 1);
        return sign | (rounded >> shift) as u16;
    }
    let rounded = man + 0x0fff + ((man >> 13) & 1);
    // the round can carry into the exponent, which is exactly what we want
    sign | (((e as u32) << 10) + (rounded >> 13)) as u16
}

/// IEEE binary16 to f32.
pub fn f32_from_f16(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = ((h as u32) & 0x03ff) << 13;
    if exp == 0 {
        if man == 0 {
            return f32::from_bits(sign);
        }
        // subnormal: renormalise
        let mut e = 127 - 15 + 1;
        let mut m = man;
        while m & 0x0080_0000 == 0 {
            m <<= 1;
            e -= 1;
        }
        return f32::from_bits(sign | ((e as u32) << 23) | (m & 0x007f_ffff));
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | man);
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | man)
}

fn to_f16(v: &[f32]) -> Vec<u16> {
    v.iter().copied().map(f16_from_f32).collect()
}

// ─── generation ───────────────────────────────────────────────────────────

/// What [`generate`] is asked for.
pub struct Config {
    pub width: u32,
    pub height: u32,
    /// How many (camera, time) states to sample.
    pub samples: usize,
    /// Samples per pixel in the reference render.
    pub reference_spp: u32,
    /// The latest simulated instant a sample may be taken at, seconds.
    pub t_end: f64,
    /// Seed for the camera orbit and the render seeds.
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            width: 320,
            height: 180,
            samples: 50,
            reference_spp: 1024,
            t_end: 4.0,
            seed: 0x5EED_C0FF_EE12_3456,
        }
    }
}

/// A small deterministic PRNG, so a dataset is reproducible from its seed
/// without a dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// Orbit `cam` around `target` by `d_az` radians of azimuth and `d_el`
/// radians of elevation, at `scale` times its distance.
///
/// The court's authored camera is the shot the level wants; every sample is a
/// perturbation of it rather than a camera drawn from nowhere. A denoiser for
/// *this gym* should see this gym's framings — the floor at this grazing
/// angle, the panels at this distance — and not waste its capacity on views
/// the game will never take.
pub fn orbit(cam: &Camera, target: Point3, d_az: f64, d_el: f64, scale: f64) -> Camera {
    let v = cam.eye - target;
    let r = v.norm() * scale;
    let az = v.y.atan2(v.x) + d_az;
    let el = (v.z / v.norm()).asin() + d_el;
    let el = el.clamp(-1.35, 1.35);
    let eye = target
        + Vec3::new(
            r * el.cos() * az.cos(),
            r * el.cos() * az.sin(),
            r * el.sin(),
        );
    let mut out = Camera::look_at(eye, target, Vec3::z(), cam.fov_deg);
    out.aperture = cam.aperture;
    out.focus_dist = if cam.aperture > 0.0 { r } else { cam.focus_dist };
    out
}

/// Render the court at every tier and at the reference, for one state.
///
/// The tiers are prefix means of the *same* pass sequence: pass `j` is
/// rendered once, folded into the running sum, and the sum snapshotted
/// whenever `j + 1` is a tier. Sixteen one-spp renders, not thirty-one.
fn render_state(
    picture: &PtScene,
    cam: &Camera,
    cfg: &Config,
    base_seed: u64,
) -> (Vec<Film>, Film) {
    let scene_opts = |spp: u32, seed: u64| PathTraceOptions {
        spp,
        max_depth: 6,
        show_background: true,
        seed,
        denoise: false,
        ..Default::default()
    };

    let n = (cfg.width as usize) * (cfg.height as usize);
    let mut sum_rgb = vec![0.0f32; n * 3];
    let mut sum_a = vec![0.0f32; n];
    // Welford is not needed: the estimator's variance we want is the variance
    // *of the mean*, and each pass contributes one independent sample, so the
    // running sum of squares over passes is enough.
    let mut sum_l = vec![0.0f32; n];
    let mut sum_l2 = vec![0.0f32; n];
    let mut tiers: Vec<Film> = Vec::with_capacity(TIERS.len());
    let mut last: Option<Film> = None;

    let top = *TIERS.last().unwrap();
    for j in 0..top {
        let f = render::render(
            picture,
            cam,
            cfg.width,
            cfg.height,
            &scene_opts(1, base_seed.wrapping_add(j as u64 * 0x9E37_79B9)),
        );
        for i in 0..n {
            let l = 0.2126 * f.rgb[i * 3] + 0.7152 * f.rgb[i * 3 + 1] + 0.0722 * f.rgb[i * 3 + 2];
            sum_l[i] += l;
            sum_l2[i] += l * l;
            sum_a[i] += f.alpha[i];
            for c in 0..3 {
                sum_rgb[i * 3 + c] += f.rgb[i * 3 + c];
            }
        }
        let k = j + 1;
        if TIERS.contains(&k) {
            let kf = k as f32;
            let mut snap = Film {
                width: f.width,
                height: f.height,
                rgb: f.rgb.clone(),
                alpha: f.alpha.clone(),
                normal: f.normal.clone(),
                depth: f.depth.clone(),
                albedo: f.albedo.clone(),
                variance: f.variance.clone(),
            };
            for i in 0..n {
                for c in 0..3 {
                    snap.rgb[i * 3 + c] = sum_rgb[i * 3 + c] / kf;
                }
                snap.alpha[i] = sum_a[i] / kf;
                // The variance of the mean of `k` independent passes: the
                // sample variance over passes, divided by k. One pass has no
                // sample variance, so it falls back on the integrator's own
                // per-pass estimate, which is what the device's history does
                // on a pixel's first sample.
                snap.variance[i] = if k >= 2 {
                    let mu = sum_l[i] / kf;
                    let s2 = (sum_l2[i] / kf - mu * mu).max(0.0) * kf / (kf - 1.0);
                    s2 / kf
                } else {
                    f.variance[i]
                };
            }
            tiers.push(snap);
        }
        last = Some(f);
    }
    // The guides come off the last pass, which is the least noisy set of
    // primary-hit quantities we rendered.
    let guides = last.expect("at least one tier");
    for t in tiers.iter_mut() {
        t.normal.clone_from(&guides.normal);
        t.depth.clone_from(&guides.depth);
        t.albedo.clone_from(&guides.albedo);
    }

    let reference = render::render(
        picture,
        cam,
        cfg.width,
        cfg.height,
        &scene_opts(cfg.reference_spp, base_seed ^ 0xD1CE_D1CE_D1CE_D1CE),
    );
    (tiers, reference)
}

/// Sample the court and render every sample noisy and converged.
///
/// `progress` is called with `(index, total)` before each sample; a reference
/// render is minutes of CPU and a caller wants to know it is alive.
pub fn generate(
    scene: &CourtScene,
    cfg: &Config,
    mut progress: impl FnMut(usize, usize),
) -> anyhow::Result<Dataset> {
    let mut court = Court::from_scene(scene)?;
    let mut picture = render::Scene::new(scene)?;
    let base_cam = render::camera(scene)?;
    // The level states its aim point directly, so every orbit turns about the
    // point the authored shot is about rather than about something recovered
    // from the forward ray.
    let a = &scene.authored;
    let target = Point3::new(
        a.parameter("cam_at_x_mm")?,
        a.parameter("cam_at_y_mm")?,
        a.parameter("cam_at_z_mm")?,
    );

    // Roll the simulation once and keep a snapshot per step, so a random time
    // is a lookup rather than a re-run.
    let mut snaps: Vec<Snapshot> = vec![Snapshot::of(&court)];
    while court.time() < cfg.t_end {
        court.step();
        snaps.push(Snapshot::of(&court));
    }

    let mut rng = Rng(cfg.seed);
    let mut samples = Vec::with_capacity(cfg.samples);
    for s in 0..cfg.samples {
        progress(s, cfg.samples);
        let snap = &snaps[(rng.unit() * (snaps.len() - 1) as f64) as usize];
        let cam = orbit(
            &base_cam,
            target,
            rng.range(-std::f64::consts::PI, std::f64::consts::PI),
            rng.range(-0.25, 0.45),
            rng.range(0.7, 1.35),
        );
        let pt = picture.at_snapshot(snap);
        let (tiers, reference) = render_state(&pt, &cam, cfg, rng.next_u64());
        samples.push(Sample {
            width: cfg.width,
            height: cfg.height,
            mean: tiers.iter().map(|f| to_f16(&f.rgb)).collect(),
            variance: tiers.iter().map(|f| to_f16(&f.variance)).collect(),
            normal: to_f16(&tiers[0].normal),
            depth: to_f16(&tiers[0].depth),
            albedo: to_f16(&tiers[0].albedo),
            reference: to_f16(&reference.rgb),
        });
    }
    progress(cfg.samples, cfg.samples);
    Ok(Dataset {
        width: cfg.width,
        height: cfg.height,
        samples,
    })
}

/// One 64x64 (or whatever) crop of one sample at one tier, unpacked to f32
/// and laid out as the network's planes.
pub struct Tile {
    pub size: usize,
    /// Accumulated passes behind `mean`.
    pub count: u32,
    /// Running mean radiance, 3 planes.
    pub mean: Vec<f32>,
    /// Variance of the mean, 1 plane.
    pub variance: Vec<f32>,
    pub normal: Vec<f32>,
    pub depth: Vec<f32>,
    pub albedo: Vec<f32>,
    pub reference: Vec<f32>,
}

/// Cut every non-overlapping `size`-square crop out of `sample` at tier
/// index `tier`.
pub fn tiles(sample: &Sample, tier: usize, size: usize) -> Vec<Tile> {
    let w = sample.width as usize;
    let h = sample.height as usize;
    let _ = sample.pixels();
    let mut out = Vec::new();
    let plane = |src: &[u16], ox: usize, oy: usize, c: usize| -> Vec<f32> {
        let mut v = vec![0.0f32; size * size * c];
        for y in 0..size {
            for x in 0..size {
                let s = ((oy + y) * w + ox + x) * c;
                let d = (y * size + x) * c;
                for k in 0..c {
                    v[d + k] = f32_from_f16(src[s + k]);
                }
            }
        }
        v
    };
    for oy in (0..h.saturating_sub(size - 1)).step_by(size) {
        for ox in (0..w.saturating_sub(size - 1)).step_by(size) {
            out.push(Tile {
                size,
                count: TIERS[tier],
                mean: plane(&sample.mean[tier], ox, oy, 3),
                variance: plane(&sample.variance[tier], ox, oy, 1),
                normal: plane(&sample.normal, ox, oy, 3),
                depth: plane(&sample.depth, ox, oy, 1),
                albedo: plane(&sample.albedo, ox, oy, 3),
                reference: plane(&sample.reference, ox, oy, 3),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trips_the_values_a_render_produces() {
        for &x in &[0.0f32, 1.0, 0.5, 1e-3, 123.5, -2.25, 65504.0] {
            let back = f32_from_f16(f16_from_f32(x));
            assert!(
                (back - x).abs() <= x.abs() * 1e-3 + 1e-7,
                "{x} came back as {back}"
            );
        }
        assert_eq!(f32_from_f16(f16_from_f32(0.0)), 0.0);
        assert!(f32_from_f16(f16_from_f32(1e30)).is_infinite());
    }
}
