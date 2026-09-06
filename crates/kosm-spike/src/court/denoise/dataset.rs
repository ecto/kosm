//! The training set: the court's *own device history*, and the court rendered
//! right.
//!
//! # The lesson this format exists to encode
//!
//! The first dataset was fifty CPU renders at 320x180, and each of its noisy
//! tiers was the mean of *k* independent one-sample passes with the variance
//! of that mean. That is a perfectly good description of a Monte Carlo
//! estimator and it is not what the viewer hands the denoiser. The viewer's
//! history is an exponential moving average with a cap, carried across camera
//! and object motion by a reprojection, and shortened wherever a
//! neighbourhood clamp decides a pixel has gone stale. Its `count` and its
//! `variance` are different quantities with the same names, and two of the
//! network's input planes are exactly those two. The v1 weights won on
//! held-out CPU tiles and rendered the backboard 1.7x too bright in the
//! window, and that gap is the whole of the reason.
//!
//! So: **train on the distribution you run on.** Every plane here is read
//! back off the device, out of the same buffers `neural.wgsl` binds, after
//! the same passes the window runs — trace, reproject, accumulate, clamp.
//! The dataset builder lives in `kosm-view` because that is where the GPU
//! stage is; this module is the format the two ends agree on and the tiles
//! the trainer cuts.
//!
//! # What one sample is
//!
//! A sample is one *sequence*: a (camera, time) state of the court, driven
//! frame by frame exactly as `--dump-frames` drives it, with the simulation
//! stepping and — for some sequences — the camera moving mid-flight. Along
//! the way it records:
//!
//! * **The history, at the sequence's own length.** Mean radiance, per-pixel
//!   sample count and per-pixel variance of the mean, straight out of the
//!   device's `History` after that many passes. The
//!   counts are what the reprojection and the clamp left behind, so a frame
//!   here has short-history pixels sitting beside long-history ones, which is
//!   what a frame in the window looks like and what a frame in v1 never did.
//! * **The guides.** Normal, depth, albedo and the biased hit id, read out of
//!   the resident scene's guide planes — bindings 1 and 2 of the neural pass.
//!   The id is new: it is what tells the network that the backboard's glass
//!   and the wall behind it are different surfaces when their normals and
//!   depths agree.
//! * **The reference.** A [`Config::reference_spp`]-pass render of the *same*
//!   instant from the *same* camera, with the history at rest, the denoiser
//!   off and no clamp. Exact ground truth for the frame, not for a frame near
//!   it.
//!
//! # One sequence, one instant, one reference
//!
//! A sample stores **one** history length, named by [`Sample::tier`], and the
//! reference is of *that frame's* instant. It cannot be otherwise: the balls
//! are in flight and the net is swinging, so the converged answer for the
//! frame where a pixel has one sample is a different picture from the
//! converged answer thirty frames later. Covering [`TIERS`] therefore means
//! several sequences, each run for as many frames as its tier and stopped
//! there — not one sequence snapshotted along the way.
//!
//! The one thing stored twice is the *successor* frame, [`Sample::next`]: one
//! more pass of the same sequence, whose reference nothing needs because the
//! temporal consistency term in [`super::train`] compares the network's two
//! answers to each other and not to the truth.
//!
//! # The file
//!
//! One header, then one block per sample, every plane f16 and whole-frame.
//! Crops are taken at load time, so tile size is not baked in. f16 is not a
//! compromise: radiance in this gym runs from about 1e-3 to a few hundred,
//! well inside its range, and counts never exceed the history cap.

use std::io::{Read, Write};
use std::path::Path;

/// The history lengths the network is trained and scored at.
pub const TIERS: [u32; 6] = [1, 2, 4, 8, 16, 32];

/// Planes per recorded frame: mean radiance (3), sample count (1), variance
/// of the mean (1). Two frames are stored: the tier, and its successor.
const FRAME_PLANES: usize = 5;
/// Guide planes: normal (3), depth (1), albedo (3), id (1).
const GUIDE_PLANES: usize = 8;
/// Reference planes: converged radiance (3).
const REF_PLANES: usize = 3;

/// Every f16 plane one sample carries.
pub const SAMPLE_PLANES: usize = 2 * FRAME_PLANES + GUIDE_PLANES + REF_PLANES;

const MAGIC: &[u8; 8] = b"KOSMDN02";

/// One frame of a sequence, as the device history held it.
#[derive(Clone, Default)]
pub struct FrameState {
    /// Running mean radiance, 3 f16 per pixel.
    pub mean: Vec<u16>,
    /// How many samples each pixel's mean is over, 1 f16 per pixel.
    pub count: Vec<u16>,
    /// Variance of that mean, 1 f16 per pixel.
    pub variance: Vec<u16>,
}

/// One sampled sequence of the court, all planes f16 and row-major.
#[derive(Clone)]
pub struct Sample {
    pub width: u32,
    pub height: u32,
    /// The nominal history length this sequence was stopped at — the frame
    /// index, and therefore the count a pixel that kept its history holds.
    /// One of [`TIERS`]; carried so an evaluation can group by it.
    pub tier: u32,
    /// The history at frame `tier`. What the reference is the answer to.
    pub cur: FrameState,
    /// The history one pass later, for the temporal term. Same guides: on the
    /// pixels that term is masked to, nothing under them moved.
    pub next: FrameState,
    /// World normal at the first hit, 3 f16 per pixel.
    pub normal: Vec<u16>,
    /// Distance to the first hit; zero is the background sentinel.
    pub depth: Vec<u16>,
    /// Denoise albedo, 3 f16 per pixel.
    pub albedo: Vec<u16>,
    /// Biased hit id, 1 f16 per pixel; zero on background.
    pub id: Vec<u16>,
    /// The converged answer, 3 f16 per pixel.
    pub reference: Vec<u16>,
}

/// A whole dataset, in memory.
///
/// Unlike v1, samples may be **different sizes**: the viewer runs at whatever
/// the window and its scale divisor make, and the point of this dataset is
/// that the network sees the sizes it will be run at. `width`/`height` are
/// the largest, kept only so a caller can size a scratch buffer.
pub struct Dataset {
    pub width: u32,
    pub height: u32,
    pub samples: Vec<Sample>,
}

impl Sample {
    fn pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    fn bytes(&self) -> usize {
        self.pixels() * SAMPLE_PLANES * 2
    }
}

impl Dataset {
    /// Bytes on disk, as [`Dataset::save`] writes them.
    pub fn bytes(&self) -> usize {
        16 + self.samples.iter().map(|s| 16 + s.bytes()).sum::<usize>()
    }

    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(MAGIC)?;
        f.write_all(&(self.samples.len() as u32).to_le_bytes())?;
        f.write_all(&(TIERS.len() as u32).to_le_bytes())?;
        for s in &self.samples {
            f.write_all(&s.width.to_le_bytes())?;
            f.write_all(&s.height.to_le_bytes())?;
            f.write_all(&s.tier.to_le_bytes())?;
            f.write_all(&0u32.to_le_bytes())?;
            for fr in [&s.cur, &s.next] {
                write_u16(&mut f, &fr.mean)?;
                write_u16(&mut f, &fr.count)?;
                write_u16(&mut f, &fr.variance)?;
            }
            write_u16(&mut f, &s.normal)?;
            write_u16(&mut f, &s.depth)?;
            write_u16(&mut f, &s.albedo)?;
            write_u16(&mut f, &s.id)?;
            write_u16(&mut f, &s.reference)?;
        }
        f.flush()
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut head = [0u8; 16];
        f.read_exact(&mut head)?;
        if &head[0..8] != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a kosm denoise dataset (v2)",
            ));
        }
        let n = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        let nt = u32::from_le_bytes(head[12..16].try_into().unwrap()) as usize;
        if nt != TIERS.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "tier count does not match this build",
            ));
        }
        let mut samples = Vec::with_capacity(n);
        let (mut mw, mut mh) = (0u32, 0u32);
        for _ in 0..n {
            let mut wh = [0u8; 16];
            f.read_exact(&mut wh)?;
            let width = u32::from_le_bytes(wh[0..4].try_into().unwrap());
            let height = u32::from_le_bytes(wh[4..8].try_into().unwrap());
            let tier = u32::from_le_bytes(wh[8..12].try_into().unwrap());
            mw = mw.max(width);
            mh = mh.max(height);
            let px = (width as usize) * (height as usize);
            let mut read_frame = |f: &mut std::io::BufReader<std::fs::File>| {
                Ok::<_, std::io::Error>(FrameState {
                    mean: read_u16(f, px * 3)?,
                    count: read_u16(f, px)?,
                    variance: read_u16(f, px)?,
                })
            };
            let cur = read_frame(&mut f)?;
            let next = read_frame(&mut f)?;
            samples.push(Sample {
                width,
                height,
                tier,
                cur,
                next,
                normal: read_u16(&mut f, px * 3)?,
                depth: read_u16(&mut f, px)?,
                albedo: read_u16(&mut f, px * 3)?,
                id: read_u16(&mut f, px)?,
                reference: read_u16(&mut f, px * 3)?,
            });
        }
        Ok(Self {
            width: mw,
            height: mh,
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


/// Pack an f32 plane for storage.
pub fn to_f16(v: &[f32]) -> Vec<u16> {
    v.iter().copied().map(f16_from_f32).collect()
}

/// Unpack an f16 plane.
pub fn from_f16(v: &[u16]) -> Vec<f32> {
    v.iter().copied().map(f32_from_f16).collect()
}

// --- tiles ---------------------------------------------------------------

/// One crop of one sample at one tier, unpacked to f32 and laid out as the
/// network's planes.
///
/// `count` is a plane and not a scalar, which is the other half of the v1
/// lesson: on the device it always was one, and a reprojected frame has
/// pixels at one sample beside pixels at the cap.
pub struct Tile {
    pub size: usize,
    /// The nominal history length this tier is, for reporting.
    pub tier: u32,
    /// Running mean radiance, interleaved.
    pub mean: Vec<f32>,
    /// Variance of the mean.
    pub variance: Vec<f32>,
    /// Per-pixel sample count.
    pub count: Vec<f32>,
    pub normal: Vec<f32>,
    pub depth: Vec<f32>,
    pub albedo: Vec<f32>,
    pub id: Vec<f32>,
    pub reference: Vec<f32>,
    /// The *next* frame of the same sequence, when the file has one: its
    /// mean, count and variance over the same guides. The temporal term
    /// compares the network's answer here with its answer there.
    pub next: Option<Box<TileNext>>,
}

/// The successor frame's varying planes; the guides are shared with [`Tile`].
pub struct TileNext {
    pub mean: Vec<f32>,
    pub variance: Vec<f32>,
    pub count: Vec<f32>,
}

fn crop(src: &[u16], w: usize, ox: usize, oy: usize, size: usize, c: usize) -> Vec<f32> {
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
}

/// Cut every non-overlapping `size`-square crop out of `sample`.
pub fn tiles(sample: &Sample, size: usize) -> Vec<Tile> {
    let w = sample.width as usize;
    let h = sample.height as usize;
    let (cur, nxt) = (&sample.cur, &sample.next);
    let mut out = Vec::new();
    for oy in (0..h.saturating_sub(size - 1)).step_by(size) {
        for ox in (0..w.saturating_sub(size - 1)).step_by(size) {
            out.push(Tile {
                size,
                tier: sample.tier,
                mean: crop(&cur.mean, w, ox, oy, size, 3),
                variance: crop(&cur.variance, w, ox, oy, size, 1),
                count: crop(&cur.count, w, ox, oy, size, 1),
                normal: crop(&sample.normal, w, ox, oy, size, 3),
                depth: crop(&sample.depth, w, ox, oy, size, 1),
                albedo: crop(&sample.albedo, w, ox, oy, size, 3),
                id: crop(&sample.id, w, ox, oy, size, 1),
                reference: crop(&sample.reference, w, ox, oy, size, 3),
                next: Some(Box::new(TileNext {
                    mean: crop(&nxt.mean, w, ox, oy, size, 3),
                    variance: crop(&nxt.variance, w, ox, oy, size, 1),
                    count: crop(&nxt.count, w, ox, oy, size, 1),
                })),
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

    #[test]
    fn a_dataset_round_trips_through_a_file() {
        let px = 6 * 4;
        let mk = |v: f32, n: usize| to_f16(&vec![v; n]);
        let s = Sample {
            width: 6,
            height: 4,
            tier: 4,
            cur: FrameState {
                mean: mk(0.1, px * 3),
                count: mk(4.0, px),
                variance: mk(0.02, px),
            },
            next: FrameState {
                mean: mk(0.2, px * 3),
                count: mk(5.0, px),
                variance: mk(0.01, px),
            },
            normal: mk(0.5, px * 3),
            depth: mk(1200.0, px),
            albedo: mk(0.3, px * 3),
            id: mk(7.0, px),
            reference: mk(0.25, px * 3),
        };
        let d = Dataset {
            width: 6,
            height: 4,
            samples: vec![s],
        };
        let path = std::env::temp_dir().join("kosm-denoise-v2-roundtrip.bin");
        d.save(&path).unwrap();
        let back = Dataset::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(back.samples.len(), 1);
        assert_eq!(back.samples[0].width, 6);
        assert_eq!(back.samples[0].tier, 4);
        assert_eq!(f32_from_f16(back.samples[0].id[0]), 7.0);
        let t = &tiles(&back.samples[0], 4)[0];
        assert_eq!(t.tier, 4);
        assert_eq!(t.count[0], 4.0);
        assert_eq!(t.next.as_ref().unwrap().count[0], 5.0);
    }
}
