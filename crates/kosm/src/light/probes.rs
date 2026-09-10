//! Spectral SH probes: a room's static light, solved once.
//!
//! A cove's lighting does not change. The sun is fixed, the sand and the
//! cliff are static, the sky is a gradient; what moves is the hero and one
//! caustic. A live tier that re-solves that illumination sixty times a second
//! is doing the same integral over and over and getting a noisier answer than
//! the path tracer got offline — so the path tracer does it once, into this,
//! and the raster tier reads it.
//!
//! **What a probe holds.** At each grid point, the incoming radiance field
//! projected onto nine real spherical harmonics ([`SH`], L2) in six bands
//! ([`BANDS`], the same bands [`crate::material::Spectrum`] carries), per sun
//! position. [`ProbeVolume::sample`] convolves that with the clamped-cosine
//! lobe — Ramamoorthi and Hanrahan's `Â = [π, 2π/3, π/4]` — and hands back
//! **irradiance** per band at a normal, trilinear between probes and linear
//! between sun positions. A shader multiplies it by the material's per-band
//! albedo and projects to RGB through
//! [`band_to_rgb`](crate::material::band_to_rgb); the tracer's `pbr()` sees
//! the same numbers through the same projection, which is what keeps the two
//! tiers honest about the same material.
//!
//! **The sun is not sampled.** A sun disc is a millionth of the sphere: at
//! sixty-four rays a probe would find it once in fifteen probes and the
//! answer would be confetti. So the hemisphere sampling carries the sky and
//! every bounce, the sun's *direct* term is added analytically — its
//! irradiance times a visibility fraction times the SH basis at its direction
//! — and the directions inside the disc are skipped by the sampled term so
//! nothing is counted twice. That is why a bake at 64 rays has a readable
//! sun-shadow and a noisy ambient rather than the other way round.
//!
//! **Bands are the tracer's spectral resolution, in the file's shape.** The
//! CPU integrator carries RGB with a hero wavelength where a material
//! disperses, so a band pair holds the channel it belongs to. The file is
//! band-shaped because the *contract* is spectral: the day the tracer resolves
//! six bands, nothing downstream changes.
//!
//! ```no_run
//! use kosm::light::probes::{self, ProbeVolume, VolumeSpec};
//!
//! # fn main() -> anyhow::Result<()> {
//! let v = ProbeVolume::read("out/maps/cove/probes.bin")?;
//! let e = v.sample(0.0, [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]);   // irradiance per band
//! assert_eq!(e.len(), probes::BANDS);
//! # Ok(()) }
//! ```

use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use kosm_render::geometry::Geometry as RenderGeometry;
use kosm_render::math::{Point3, Vec3 as RVec3};
use kosm_render::pathtrace::{
    Environment, Object, PathTraceOptions, Scene, Sun, Tracer, studio_rig,
};
use kosm_render::{Bvh, Ray};

use crate::lens::Lens;
use crate::world::World;

/// How many bands a probe carries: [`crate::material::BANDS`].
pub const BANDS: usize = crate::material::BANDS;

/// How many spherical harmonics: L2, real, in the order `(0,0)`, `(1,−1)`,
/// `(1,0)`, `(1,1)`, `(2,−2)`, `(2,−1)`, `(2,0)`, `(2,1)`, `(2,2)`.
pub const SH: usize = 9;

/// The clamped-cosine lobe's SH coefficients, per band index `l`.
const A_L: [f64; 3] = [
    std::f64::consts::PI,
    2.0 * std::f64::consts::PI / 3.0,
    std::f64::consts::PI / 4.0,
];

/// The projection [`crate::material::band_to_rgb`] is, re-exported here
/// because a reader of a probe volume needs it and should not have to know
/// it lives with the materials.
pub use crate::material::{band_to_rgb, bands_to_rgb};

/// The real L2 SH basis at a unit direction, in [`SH`]'s order.
pub fn sh_basis(d: [f64; 3]) -> [f64; SH] {
    let (x, y, z) = (d[0], d[1], d[2]);
    [
        0.282_094_791_773_878_14,
        0.488_602_511_902_919_9 * y,
        0.488_602_511_902_919_9 * z,
        0.488_602_511_902_919_9 * x,
        1.092_548_430_592_079_2 * x * y,
        1.092_548_430_592_079_2 * y * z,
        0.315_391_565_252_520_02 * (3.0 * z * z - 1.0),
        1.092_548_430_592_079_2 * x * z,
        0.546_274_215_296_039_6 * (x * x - y * y),
    ]
}

/// Which `Â_l` a basis index belongs to.
#[inline]
fn a_of(i: usize) -> f64 {
    match i {
        0 => A_L[0],
        1..=3 => A_L[1],
        _ => A_L[2],
    }
}

/// An RGB radiance as bands, in the layout [`crate::material::Spectrum::rgb`]
/// uses: each primary spread over the two bands it owns.
#[inline]
fn bands_of(rgb: [f32; 3]) -> [f32; BANDS] {
    [rgb[2], rgb[2], rgb[1], rgb[1], rgb[0], rgb[0]]
}

// ── the volume ────────────────────────────────────────────────────────────

/// Where the probes go: a corner, a cubic spacing, and a count per axis.
///
/// Metres, because the rest of kosm is metres. `scene_per_metre` is the one
/// place a scene in other units crosses — the cove's picture is assembled in
/// millimetres, so its bake passes `1000.0` and the file it writes is still
/// in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeSpec {
    /// The corner of cell `(0, 0, 0)`, metres.
    pub origin: [f64; 3],
    /// Cubic cell size, metres.
    pub spacing: f64,
    /// Probes per axis. One per axis is legal and makes a plane or a line.
    pub dims: [u32; 3],
    /// Scene units per metre: `1.0` for a scene in metres, `1000.0` for one
    /// in millimetres.
    pub scene_per_metre: f64,
}

impl VolumeSpec {
    /// A volume covering `min..max` at `spacing`, inclusive of both ends.
    pub fn over(min: [f64; 3], max: [f64; 3], spacing: f64) -> Self {
        let spacing = spacing.max(1e-6);
        let mut dims = [1u32; 3];
        for k in 0..3 {
            let span = (max[k] - min[k]).max(0.0);
            dims[k] = ((span / spacing).floor() as u32).saturating_add(1);
        }
        Self { origin: min, spacing, dims, scene_per_metre: 1.0 }
    }

    /// The same in a scene whose units are not metres.
    pub fn in_scene_units(mut self, per_metre: f64) -> Self {
        self.scene_per_metre = per_metre;
        self
    }

    /// How many probes that is.
    pub fn count(&self) -> usize {
        self.dims.iter().map(|d| *d as usize).product()
    }

    /// The world point of probe `(ix, iy, iz)`, metres.
    pub fn point(&self, ix: u32, iy: u32, iz: u32) -> [f64; 3] {
        [
            self.origin[0] + self.spacing * ix as f64,
            self.origin[1] + self.spacing * iy as f64,
            self.origin[2] + self.spacing * iz as f64,
        ]
    }
}

/// A baked volume: SH radiance per probe per sun, plus the sky on its own.
///
/// `data` is `suns × nz × ny × nx × SH × BANDS`, x fastest between probes and
/// band fastest inside one, which is what [`ProbeVolume::index`] returns the
/// start of.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeVolume {
    /// Metres, corner of cell `(0, 0, 0)`.
    pub origin: [f64; 3],
    /// Metres, cubic cells.
    pub spacing: f64,
    /// `nx, ny, nz`.
    pub dims: [u32; 3],
    /// Unit vectors *toward* the sun, one per baked sun position; at least one.
    pub suns: Vec<[f64; 3]>,
    /// `suns × nz × ny × nx × SH × BANDS`, x fastest inside a probe block.
    pub data: Vec<f32>,
    /// The sky alone — the environment with the sun below the horizon and no
    /// geometry in the way — for the sun-independent term.
    pub sky: [[f32; BANDS]; SH],
    /// One bit per probe, x fastest: whether the probe sits inside a solid.
    /// Those probes carry no light and are skipped by [`ProbeVolume::sample`].
    pub inside: Vec<u32>,
}

impl ProbeVolume {
    /// An empty volume of the given shape: every probe dark, none inside.
    pub fn zeros(spec: &VolumeSpec, suns: Vec<[f64; 3]>) -> Self {
        let n = spec.count();
        let suns = if suns.is_empty() { vec![[0.0, 0.0, 1.0]] } else { suns };
        Self {
            origin: spec.origin,
            spacing: spec.spacing,
            dims: spec.dims,
            data: vec![0.0; n * suns.len() * SH * BANDS],
            suns,
            sky: [[0.0; BANDS]; SH],
            inside: vec![0; n.div_ceil(32)],
        }
    }

    /// How many probes the volume holds.
    pub fn count(&self) -> usize {
        self.dims.iter().map(|d| *d as usize).product()
    }

    /// The start of one probe's `SH × BANDS` block in [`Self::data`].
    pub fn index(&self, sun: usize, ix: u32, iy: u32, iz: u32) -> usize {
        let (nx, ny, nz) = (self.dims[0] as usize, self.dims[1] as usize, self.dims[2] as usize);
        let (ix, iy, iz) = (ix as usize, iy as usize, iz as usize);
        (((sun * nz + iz) * ny + iy) * nx + ix) * SH * BANDS
    }

    /// The flat probe index, ignoring the sun.
    #[inline]
    fn probe_index(&self, ix: u32, iy: u32, iz: u32) -> usize {
        let (nx, ny) = (self.dims[0] as usize, self.dims[1] as usize);
        (iz as usize * ny + iy as usize) * nx + ix as usize
    }

    /// Whether that probe sits inside a solid.
    pub fn is_inside(&self, ix: u32, iy: u32, iz: u32) -> bool {
        let i = self.probe_index(ix, iy, iz);
        self.inside.get(i / 32).is_some_and(|w| w >> (i % 32) & 1 == 1)
    }

    /// Mark a probe as inside a solid.
    pub fn set_inside(&mut self, ix: u32, iy: u32, iz: u32, inside: bool) {
        let i = self.probe_index(ix, iy, iz);
        if let Some(w) = self.inside.get_mut(i / 32) {
            if inside {
                *w |= 1 << (i % 32);
            } else {
                *w &= !(1 << (i % 32));
            }
        }
    }

    /// One probe's SH block, `SH × BANDS`.
    pub fn probe(&self, sun: usize, ix: u32, iy: u32, iz: u32) -> &[f32] {
        let i = self.index(sun, ix, iy, iz);
        &self.data[i..i + SH * BANDS]
    }

    /// The world point of probe `(ix, iy, iz)`.
    pub fn point(&self, ix: u32, iy: u32, iz: u32) -> [f64; 3] {
        [
            self.origin[0] + self.spacing * ix as f64,
            self.origin[1] + self.spacing * iy as f64,
            self.origin[2] + self.spacing * iz as f64,
        ]
    }

    /// Irradiance per band at `p` on a surface facing `n`, for a fractional
    /// sun index.
    ///
    /// Trilinear over the eight probes around `p`, linear over the two sun
    /// slices `sun` falls between, then the SH convolved with the clamped
    /// cosine lobe at `n`. Probes inside solids drop out of the trilinear
    /// weights — a probe buried in the cliff would otherwise drag the sand in
    /// front of it to black — and when all eight are inside, the nearest
    /// probe that is not stands in for them. A point outside the volume reads
    /// the edge probes, which is the right answer for a volume that covers
    /// the level and a defensible one for a point just off it.
    pub fn sample(&self, sun: f64, p: [f64; 3], n: [f64; 3]) -> [f32; BANDS] {
        let sh = self.sample_sh(sun, p);
        let nn = normalize(n);
        let y = sh_basis(nn);
        let mut out = [0.0f32; BANDS];
        for i in 0..SH {
            let k = (a_of(i) * y[i]) as f32;
            for (b, o) in out.iter_mut().enumerate() {
                *o += k * sh[i * BANDS + b];
            }
        }
        for o in out.iter_mut() {
            // L2 ringing can undershoot; irradiance cannot be negative.
            *o = o.max(0.0);
        }
        out
    }

    /// The interpolated SH block at `p`, `SH × BANDS`. What [`Self::sample`]
    /// convolves, exposed for a tier that wants to convolve it itself (a
    /// shader hands it a normal per pixel, not per probe).
    pub fn sample_sh(&self, sun: f64, p: [f64; 3]) -> [f32; SH * BANDS] {
        let mut out = [0.0f32; SH * BANDS];
        if self.suns.is_empty() || self.data.is_empty() {
            return out;
        }
        let ns = self.suns.len();
        let s = sun.clamp(0.0, (ns - 1) as f64);
        let s0 = s.floor() as usize;
        let s1 = (s0 + 1).min(ns - 1);
        let ts = s - s0 as f64;

        // The cell and the fractional position inside it.
        let mut base = [0u32; 3];
        let mut frac = [0.0f64; 3];
        for k in 0..3 {
            let d = self.dims[k].max(1);
            let g = ((p[k] - self.origin[k]) / self.spacing).clamp(0.0, (d - 1) as f64);
            let i = (g.floor() as u32).min(d.saturating_sub(2));
            base[k] = i;
            frac[k] = (g - i as f64).clamp(0.0, 1.0);
        }

        let mut total = 0.0f64;
        let mut acc = [0.0f64; SH * BANDS];
        for corner in 0..8u32 {
            let mut ijk = [0u32; 3];
            let mut w = 1.0f64;
            for k in 0..3 {
                let hi = corner >> k & 1 == 1;
                let last = self.dims[k].saturating_sub(1);
                ijk[k] = if hi { (base[k] + 1).min(last) } else { base[k] };
                // A single-probe axis has no far corner to weigh.
                let f = if self.dims[k] <= 1 { 0.0 } else { frac[k] };
                w *= if hi { f } else { 1.0 - f };
            }
            if w <= 0.0 || self.is_inside(ijk[0], ijk[1], ijk[2]) {
                continue;
            }
            self.add_probe(&mut acc, s0, s1, ts, ijk, w);
            total += w;
        }
        if total <= 1e-9 {
            // Every corner is buried. Walk outwards for one that is not.
            let mut nearest = None;
            'search: for r in 1..=4i64 {
                for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                                continue;
                            }
                            let mut ijk = [0u32; 3];
                            let mut ok = true;
                            for (k, d) in [dx, dy, dz].into_iter().enumerate() {
                                let i = base[k] as i64 + d;
                                if i < 0 || i >= self.dims[k] as i64 {
                                    ok = false;
                                    break;
                                }
                                ijk[k] = i as u32;
                            }
                            if ok && !self.is_inside(ijk[0], ijk[1], ijk[2]) {
                                nearest = Some(ijk);
                                break 'search;
                            }
                        }
                    }
                }
            }
            match nearest {
                Some(ijk) => {
                    self.add_probe(&mut acc, s0, s1, ts, ijk, 1.0);
                    total = 1.0;
                }
                None => return out,
            }
        }
        for (o, a) in out.iter_mut().zip(acc) {
            *o = (a / total) as f32;
        }
        out
    }

    fn add_probe(
        &self,
        acc: &mut [f64; SH * BANDS],
        s0: usize,
        s1: usize,
        ts: f64,
        ijk: [u32; 3],
        w: f64,
    ) {
        let a = self.index(s0, ijk[0], ijk[1], ijk[2]);
        let b = self.index(s1, ijk[0], ijk[1], ijk[2]);
        for k in 0..SH * BANDS {
            let v = self.data[a + k] as f64 * (1.0 - ts) + self.data[b + k] as f64 * ts;
            acc[k] += w * v;
        }
    }

    // ── the file ──────────────────────────────────────────────────────────

    /// Write the volume: a header, then the data as little-endian `f32`.
    ///
    /// `KPRB`, version 1, the shape, the sun list, the sky term, the
    /// inside-solid bits, the data. Every number is little-endian and nothing
    /// is derived at read time, so [`Self::read`] round-trips bit for bit.
    pub fn write(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let path = path.as_ref();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(b"KPRB")?;
        f.write_all(&1u32.to_le_bytes())?;
        for v in self.origin {
            f.write_all(&v.to_le_bytes())?;
        }
        f.write_all(&self.spacing.to_le_bytes())?;
        for d in self.dims {
            f.write_all(&d.to_le_bytes())?;
        }
        f.write_all(&(self.suns.len() as u32).to_le_bytes())?;
        f.write_all(&(SH as u32).to_le_bytes())?;
        f.write_all(&(BANDS as u32).to_le_bytes())?;
        for s in &self.suns {
            for v in s {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        for row in &self.sky {
            for v in row {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        f.write_all(&(self.inside.len() as u32).to_le_bytes())?;
        for w in &self.inside {
            f.write_all(&w.to_le_bytes())?;
        }
        f.write_all(&(self.data.len() as u64).to_le_bytes())?;
        let mut bytes = Vec::with_capacity(self.data.len() * 4);
        for v in &self.data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&bytes)?;
        f.flush()?;
        Ok(())
    }

    /// Read one back. A file of a different version or a different `SH ×
    /// BANDS` shape is refused rather than read into the wrong columns.
    pub fn read(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        anyhow::ensure!(&magic == b"KPRB", "{} is not a probe volume", path.display());
        let u32s = |f: &mut std::io::BufReader<std::fs::File>| -> anyhow::Result<u32> {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };
        let version = u32s(&mut f)?;
        anyhow::ensure!(version == 1, "probe volume version {version}, not 1");
        let f64s = |f: &mut std::io::BufReader<std::fs::File>| -> anyhow::Result<f64> {
            let mut b = [0u8; 8];
            f.read_exact(&mut b)?;
            Ok(f64::from_le_bytes(b))
        };
        let mut origin = [0.0f64; 3];
        for v in &mut origin {
            *v = f64s(&mut f)?;
        }
        let spacing = f64s(&mut f)?;
        let mut dims = [0u32; 3];
        for d in &mut dims {
            *d = u32s(&mut f)?;
        }
        let n_suns = u32s(&mut f)? as usize;
        let sh = u32s(&mut f)? as usize;
        let bands = u32s(&mut f)? as usize;
        anyhow::ensure!(
            sh == SH && bands == BANDS,
            "probe volume is {sh}×{bands} per probe, this build is {SH}×{BANDS}"
        );
        let mut suns = Vec::with_capacity(n_suns);
        for _ in 0..n_suns {
            let mut s = [0.0f64; 3];
            for v in &mut s {
                *v = f64s(&mut f)?;
            }
            suns.push(s);
        }
        let f32s = |f: &mut std::io::BufReader<std::fs::File>| -> anyhow::Result<f32> {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            Ok(f32::from_le_bytes(b))
        };
        let mut sky = [[0.0f32; BANDS]; SH];
        for row in &mut sky {
            for v in row {
                *v = f32s(&mut f)?;
            }
        }
        let n_inside = u32s(&mut f)? as usize;
        let mut inside = Vec::with_capacity(n_inside);
        for _ in 0..n_inside {
            inside.push(u32s(&mut f)?);
        }
        let mut len = [0u8; 8];
        f.read_exact(&mut len)?;
        let len = u64::from_le_bytes(len) as usize;
        let mut bytes = vec![0u8; len * 4];
        f.read_exact(&mut bytes)?;
        let data = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Self { origin, spacing, dims, suns, data, sky, inside })
    }
}

// ── the bake ──────────────────────────────────────────────────────────────

/// Everything the bake does that is not the scene.
pub struct BakeSpec<'a> {
    /// Where the probes go.
    pub volume: VolumeSpec,
    /// Unit vectors toward the sun, one volume per entry. Empty bakes one
    /// slice with the scene's own sun.
    pub suns: Vec<[f64; 3]>,
    /// Switch the sun off in every slice: the sky and the bounces alone.
    pub sky_only: bool,
    /// Hemisphere rays per probe. The ambient term's noise falls as
    /// `1/sqrt(rays)`; the sun's does not depend on it at all.
    pub rays: usize,
    /// The seed. Every probe's stream is derived from it and its own index,
    /// so the bake does not depend on how rayon scheduled it.
    pub seed: u64,
    /// Path length. Three is a probe's sensible budget: the sky, what it
    /// lights, and one bounce off that.
    pub max_depth: u32,
    /// The largest radiance one ray may bring back, in the scene's own
    /// units. `None` derives it: eight times the scene's irradiance scale,
    /// which is the sun plus the sky.
    ///
    /// A probe samples the sphere uniformly, and uniform sampling finds the
    /// sun's *specular image* — the glitter off water, the highlight on wet
    /// rock — once in ten thousand rays with a radiance of five thousand.
    /// That is real light and a real estimator, and at sixty-four rays it is
    /// also a probe that is four times its neighbours. The cap is the usual
    /// bias: the glitter's mean is kept, its spikes are not.
    /// `Some(f32::INFINITY)` is the unbiased bake.
    pub clamp: Option<f32>,
    /// Shadow rays into the sun's disc, for the analytic direct term.
    ///
    /// The sun's *penumbra* is what this buys, and it is the one place a
    /// probe volume is quantised rather than noisy: at four samples a probe
    /// half in shadow can only report a visibility of 0, ¼, ½, ¾ or 1, and
    /// the cove's penumbra — 0.02 rad at ten metres — is as wide as the probe
    /// spacing. Sixteen costs six per cent of a 256-ray bake.
    pub sun_samples: usize,
    /// "Is this point inside a solid?", when the caller has something better
    /// than six rays — the cove has a baked signed distance field. Metres.
    pub inside: Option<&'a (dyn Fn([f64; 3]) -> bool + Sync)>,
    /// Called with `(done, total)` as probes land.
    pub progress: Option<&'a (dyn Fn(usize, usize) + Sync)>,
}

impl Default for BakeSpec<'_> {
    fn default() -> Self {
        Self {
            volume: VolumeSpec::over([0.0; 3], [1.0; 3], 0.5),
            suns: Vec::new(),
            sky_only: false,
            rays: 64,
            seed: 0x5eed_b00c,
            max_depth: 3,
            clamp: None,
            sun_samples: 16,
            inside: None,
            progress: None,
        }
    }
}

/// Bake a probe volume over `scene`.
///
/// The stated form of [`bake_with`]: uniform directions, the scene's own sun
/// aimed at each of `suns`, and the six-ray inside test.
pub fn bake<G: RenderGeometry + Send + Sync>(
    scene: &Scene<G>,
    volume: &VolumeSpec,
    suns: &[[f64; 3]],
    sky_only: bool,
    rays: usize,
    seed: u64,
) -> ProbeVolume {
    bake_with(
        scene,
        &BakeSpec {
            volume: *volume,
            suns: suns.to_vec(),
            sky_only,
            rays,
            seed,
            ..BakeSpec::default()
        },
    )
}

/// The whole bake.
///
/// One [`Tracer`] per sun position — the sun is a field of the scene, so a
/// second sun is a second scene over the same geometry — and rayon over the
/// probes inside each. Every probe:
///
/// 1. **inside?** `spec.inside` if the caller has one, else [`inside_solid`]'s
///    crossing count. Geometry does not move between sun positions, so this
///    is done once for the whole bake. An inside probe carries nothing and is
///    skipped by [`ProbeVolume::sample`].
/// 2. **the sky and the bounces.** `rays` directions on a jittered spherical
///    spiral, one path each, projected onto SH with weight `4π/rays`, each
///    capped at [`BakeSpec::clamp`]. Directions inside the sun's disc are
///    dropped; the next step has them.
/// 3. **the sun.** `sun_samples` shadow rays into the disc give a visibility
///    fraction `V`; the direct term is `irradiance · V · Y(s)`, which is the
///    SH of a delta light and is exactly what the clamped-cosine convolution
///    turns back into `E · max(0, n·s)`.
pub fn bake_with<G: RenderGeometry + Send + Sync>(
    scene: &Scene<G>,
    spec: &BakeSpec<'_>,
) -> ProbeVolume {
    let base = scene.sun.unwrap_or_default();
    let suns: Vec<[f64; 3]> = if spec.suns.is_empty() {
        vec![[base.direction.x, base.direction.y, base.direction.z]]
    } else {
        spec.suns.clone()
    };
    let sky_only = spec.sky_only || scene.sun.is_none();
    let mut out = ProbeVolume::zeros(&spec.volume, suns.clone());
    let opts = PathTraceOptions {
        spp: 1,
        max_depth: spec.max_depth.max(1),
        show_background: true,
        adaptive: false,
        denoise: false,
        seed: spec.seed,
        ..PathTraceOptions::default()
    };

    // The sky on its own: the same environment with nothing in the way.
    let empty: Scene<G> = Scene {
        objects: Vec::new(),
        lights: Vec::new(),
        env: scene.env.clone(),
        sun: None,
        ground: None,
        splats: None,
    };
    let sky_tracer = Tracer::new(&empty, opts);
    out.sky = sky_sh(&sky_tracer, spec.rays.max(64), spec.seed ^ 0x5b1);

    // The scene's own scale, for the firefly cap: the irradiance an upward
    // face out in the open receives, sun and sky together.
    let sky_e = std::f64::consts::PI * out.sky[0].iter().fold(0.0f32, |a, b| a.max(*b)) as f64 * 0.282_094_79;
    let sun_e = if sky_only { 0.0 } else { base.irradiance.iter().fold(0.0f32, |a, b| a.max(*b)) as f64 };
    let cap = spec.clamp.unwrap_or((8.0 * (sky_e + sun_e).max(1e-6)) as f32);

    let (nx, ny) = (spec.volume.dims[0], spec.volume.dims[1]);
    let n_probes = spec.volume.count();
    let index = |i: usize| {
        (
            (i % nx as usize) as u32,
            ((i / nx as usize) % ny as usize) as u32,
            (i / (nx as usize * ny as usize)) as u32,
        )
    };

    // Which probes are buried. The geometry is the same at every sun, so this
    // is one pass however many suns there are.
    {
        let tracer = Tracer::new(scene, opts);
        let bits: Vec<bool> = (0..n_probes)
            .into_par_iter()
            .map(|i| {
                let (ix, iy, iz) = index(i);
                let p = spec.volume.point(ix, iy, iz);
                match spec.inside {
                    Some(f) => f(p),
                    None => inside_solid(&tracer, scale(p, spec.volume.scene_per_metre)),
                }
            })
            .collect();
        for (i, inside) in bits.iter().enumerate() {
            if *inside {
                let (ix, iy, iz) = index(i);
                out.set_inside(ix, iy, iz, true);
            }
        }
    }

    let done = AtomicUsize::new(0);
    let total = n_probes * suns.len();
    let block = SH * BANDS;
    for (si, s) in suns.iter().enumerate() {
        let sun = (!sky_only)
            .then(|| Sun::new(RVec3::new(s[0], s[1], s[2]), base.angular_radius, base.irradiance));
        let lit = with_sun(scene, sun);
        let tracer = Tracer::new(&lit, opts);
        let mut slice: Vec<f32> = vec![0.0; n_probes * block];
        slice
            .par_chunks_mut(block)
            .enumerate()
            .for_each(|(i, probe)| {
                let (ix, iy, iz) = index(i);
                if !out.is_inside(ix, iy, iz) {
                    let p = scale(spec.volume.point(ix, iy, iz), spec.volume.scene_per_metre);
                    // The probe's own stream, from the bake's seed and where
                    // it sits, so rayon's schedule cannot reach the numbers.
                    let seed = spec
                        .seed
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((si as u64) << 40)
                        .wrapping_add(i as u64);
                    let sh = probe_sh(&tracer, p, spec.rays, seed, sun.as_ref(), spec.sun_samples, cap);
                    probe.copy_from_slice(&sh);
                }
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if let Some(p) = spec.progress
                    && (n.is_multiple_of(4096) || n == total)
                {
                    p(n, total);
                }
            });
        out.data[si * n_probes * block..(si + 1) * n_probes * block].copy_from_slice(&slice);
    }
    out
}

/// The same scene with a different sun. Every BVH is shared; only the top
/// level structure is rebuilt, by the [`Tracer`] that gets it.
fn with_sun<G>(scene: &Scene<G>, sun: Option<Sun>) -> Scene<G> {
    Scene {
        objects: scene
            .objects
            .iter()
            .map(|o| Object::placed(Arc::clone(&o.bvh), o.material, o.transform.clone()))
            .collect(),
        lights: scene.lights.clone(),
        env: scene.env.clone(),
        sun,
        ground: scene.ground,
        splats: scene.splats.clone(),
    }
}

/// Inside a solid, by counting crossings along three rays.
///
/// The obvious test — "does the first face I meet point away from me?" — is
/// wrong here, and it is wrong for a reason worth writing down: the cove's
/// ground is *authored* as one union but arrives at the renderer as the
/// primitives that union was made of, so a ray inside the cliff meets the
/// inner face of the headland that overlaps it and reads it as an outside
/// face. Overlapping closed solids need the count instead: along a ray to
/// infinity, every solid the point is inside contributes one more exit than
/// entry, so `back − front > 0` is "inside", however many primitives overlap.
///
/// Three axes and a majority, because an open surface — the sea is a height
/// field, not a solid — can hand one axis a spurious exit and cannot hand
/// two.
fn inside_solid<G: RenderGeometry>(tracer: &Tracer<'_, G>, p: [f64; 3]) -> bool {
    const DIRS: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    const MAX_HITS: usize = 64;
    let mut votes = 0;
    for d in DIRS {
        let mut net = 0i32;
        let mut from = p;
        for _ in 0..MAX_HITS {
            let ray = Ray::new(Point3::new(from[0], from[1], from[2]), RVec3::new(d[0], d[1], d[2]));
            let Some(h) = tracer.first_hit(&ray) else { break };
            net += if h.front { -1 } else { 1 };
            // Step past the face we just crossed. The scene's own scale sets
            // the step: a millionth of the distance travelled, never zero.
            let eps = (h.distance * 1e-6).max(1e-6);
            from = [
                h.point[0] + d[0] * eps,
                h.point[1] + d[1] * eps,
                h.point[2] + d[2] * eps,
            ];
        }
        if net > 0 {
            votes += 1;
        }
    }
    votes >= 2
}

/// One probe: the sampled sphere, then the analytic sun.
#[allow(clippy::too_many_arguments)]
fn probe_sh<G: RenderGeometry>(
    tracer: &Tracer<'_, G>,
    p: [f64; 3],
    rays: usize,
    seed: u64,
    sun: Option<&Sun>,
    sun_samples: usize,
    cap: f32,
) -> [f32; SH * BANDS] {
    let mut sh = [0.0f32; SH * BANDS];
    let rays = rays.max(1);
    let w = (4.0 * std::f64::consts::PI / rays as f64) as f32;
    let origin = Point3::new(p[0], p[1], p[2]);
    // Two offsets out of the probe's own stream, so the spiral is stratified
    // and every probe's is its own.
    let (u0, v0) = (hash01(seed ^ 0x1), hash01(seed ^ 0x2));
    let cos_disc = sun.map(|s| s.cos_radius()).unwrap_or(2.0);
    for i in 0..rays {
        let z = 1.0 - 2.0 * (i as f64 + u0) / rays as f64;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = std::f64::consts::TAU * ((i as f64 * 0.618_033_988_749_894_9 + v0) % 1.0);
        let d = [r * phi.cos(), r * phi.sin(), z];
        if let Some(s) = sun {
            let sd = s.direction;
            if d[0] * sd.x + d[1] * sd.y + d[2] * sd.z >= cos_disc {
                continue; // the disc is the analytic term's, below
            }
        }
        let ray = Ray::new(origin, RVec3::new(d[0], d[1], d[2]));
        let rgb = tracer.radiance(ray, seed.wrapping_add(i as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
        let l = bands_of([rgb[0].min(cap), rgb[1].min(cap), rgb[2].min(cap)]);
        let y = sh_basis(d);
        for k in 0..SH {
            let yk = (y[k] as f32) * w;
            for b in 0..BANDS {
                sh[k * BANDS + b] += yk * l[b];
            }
        }
    }
    if let Some(s) = sun {
        let n = sun_samples.max(1);
        let mut visible = 0.0f64;
        for i in 0..n {
            let (d, _, _) = s.sample(
                (i as f64 + hash01(seed ^ 0x3)) / n as f64,
                hash01(seed.wrapping_add(i as u64) ^ 0x4),
            );
            if !tracer.occluded([p[0], p[1], p[2]], [d.x, d.y, d.z], f64::INFINITY) {
                visible += 1.0;
            }
        }
        let v = (visible / n as f64) as f32;
        if v > 0.0 {
            let sd = s.direction;
            let y = sh_basis([sd.x, sd.y, sd.z]);
            let e = bands_of(s.irradiance);
            for k in 0..SH {
                let yk = (y[k] as f32) * v;
                for b in 0..BANDS {
                    sh[k * BANDS + b] += yk * e[b];
                }
            }
        }
    }
    sh
}

/// The environment on its own, projected onto SH: no geometry, no sun.
fn sky_sh<G: RenderGeometry>(tracer: &Tracer<'_, G>, rays: usize, seed: u64) -> [[f32; BANDS]; SH] {
    let flat = probe_sh(tracer, [0.0; 3], rays, seed, None, 0, f32::INFINITY);
    let mut out = [[0.0f32; BANDS]; SH];
    for k in 0..SH {
        out[k].copy_from_slice(&flat[k * BANDS..(k + 1) * BANDS]);
    }
    out
}

#[inline]
fn scale(p: [f64; 3], k: f64) -> [f64; 3] {
    [p[0] * k, p[1] * k, p[2] * k]
}

#[inline]
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-12 { [v[0] / n, v[1] / n, v[2] / n] } else { [0.0, 0.0, 1.0] }
}

/// A `[0, 1)` draw from a seed, for the two offsets a probe needs.
#[inline]
fn hash01(seed: u64) -> f64 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

// ── as a lens ─────────────────────────────────────────────────────────────

/// The light in a world, as a probe volume. `World → ProbeVolume`.
///
/// The scene is the world's own columns, exactly as [`crate::lens::Camera`]
/// reads them: every body's colliders as analytic geometry at its
/// forward-kinematics pose, every sphere body as glass, and a studio rig
/// sized on the bounds. What differs is [`Self::subject_visible`]: a volume
/// baked to *light* a subject leaves the subject out, or every probe around
/// it reads the shadow of the thing it is about to shade.
pub struct Probes {
    /// Where the probes go.
    pub volume: VolumeSpec,
    /// Hemisphere rays per probe.
    pub rays: usize,
    /// The seed.
    pub seed: u64,
    /// Whether the sphere bodies — the subject — are in the scene.
    pub subject_visible: bool,
}

impl Probes {
    /// A volume over `min..max` at `spacing`, metres.
    pub fn over(min: [f64; 3], max: [f64; 3], spacing: f64) -> Self {
        Self {
            volume: VolumeSpec::over(min, max, spacing),
            rays: 128,
            seed: 0x5eed_b00c,
            subject_visible: true,
        }
    }

    /// The same, without the subject in it.
    pub fn without_subject(mut self) -> Self {
        self.subject_visible = false;
        self
    }

    /// How many rays each probe spends.
    pub fn with_rays(mut self, rays: usize) -> Self {
        self.rays = rays;
        self
    }
}

impl Lens for Probes {
    type Out = ProbeVolume;

    fn see(&self, world: &World) -> ProbeVolume {
        use phyz_math::{SpatialTransformExt, Vec3};
        use phyz_model::Geometry;
        use phyz_rigid::forward_kinematics;

        let (model, state) = world.phyz();
        let xforms = forward_kinematics(model, state).0;
        let mut objects = Vec::new();
        let mut bounds = kosm_render::Aabb::empty();
        let mut take = |geom: kosm_render::Analytic, pbr: kosm_render::Pbr| {
            for i in 0..kosm_render::Geometry::len(&geom) {
                bounds.include(&kosm_render::Geometry::bounds(&geom, i));
            }
            objects.push(Object::new(Arc::new(Bvh::build(geom)), pbr));
        };
        for (i, body) in model.bodies.iter().enumerate() {
            if !body.collisions.is_empty() {
                take(
                    crate::analytic::from_colliders(&xforms[i], &body.collisions),
                    kosm_render::Pbr::plastic([0.42, 0.40, 0.36], 0.55, 0.0),
                );
            }
            if self.subject_visible
                && let Some(Geometry::Sphere { radius }) = body.geometry
            {
                let c = xforms[i].body_to_world_point(Vec3::zeros());
                take(crate::analytic::ball(c, radius), kosm_render::Pbr::glass(1.5168, 0.0));
            }
        }
        let centre = bounds.center();
        let radius = 0.5
            * ((bounds.max.x - bounds.min.x).powi(2)
                + (bounds.max.y - bounds.min.y).powi(2)
                + (bounds.max.z - bounds.min.z).powi(2))
            .sqrt();
        let scene = Scene {
            objects,
            lights: studio_rig(centre, radius.max(1e-3)),
            env: Environment::default(),
            ground: None,
            sun: None,
            splats: None,
        };
        bake_with(
            &scene,
            &BakeSpec {
                volume: self.volume,
                rays: self.rays,
                seed: self.seed,
                sky_only: true,
                ..BakeSpec::default()
            },
        )
    }
}

/// The studio rig's own probe volume, around the material ball's stage.
///
/// The datasheet's parity check needs the two tiers lit by the same light:
/// the tracer solves the rig directly, the raster tier reads this. Baked
/// **without the ball** — the subject is what the volume is for, not part of
/// it — over a 120 mm cube around the plate at 20 mm, and cached at
/// `<out>/materials/studio_probes.bin` because it is the same volume for
/// every substance in the library.
pub fn studio_probes(out: &Path) -> anyhow::Result<ProbeVolume> {
    let path = out.join("materials").join("studio_probes.bin");
    if let Ok(v) = ProbeVolume::read(&path) {
        return Ok(v);
    }
    let world = crate::material::ball_world()?;
    let volume = Probes::over([-0.06, -0.06, -0.01], [0.06, 0.06, 0.07], 0.02)
        .without_subject()
        .with_rays(256);
    let baked = volume.see(&world);
    baked.write(&path)?;
    Ok(baked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_basis_is_orthonormal_on_the_sphere() {
        // 4π/N Σ Y_i Y_j is the identity, which is the projection this module
        // does, run against itself.
        let n = 20_000;
        let mut m = [[0.0f64; SH]; SH];
        for k in 0..n {
            let z = 1.0 - 2.0 * (k as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = std::f64::consts::TAU * ((k as f64 * 0.618_033_988_749_894_9) % 1.0);
            let y = sh_basis([r * phi.cos(), r * phi.sin(), z]);
            for i in 0..SH {
                for j in 0..SH {
                    m[i][j] += y[i] * y[j] * 4.0 * std::f64::consts::PI / n as f64;
                }
            }
        }
        for i in 0..SH {
            for j in 0..SH {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((m[i][j] - want).abs() < 2e-2, "Y{i}·Y{j} = {}", m[i][j]);
            }
        }
    }

    #[test]
    fn a_constant_sky_convolves_to_pi() {
        // L = 1 everywhere is L_00 = sqrt(4π), and π is what a surface under
        // it receives whichever way it faces.
        let spec = VolumeSpec::over([0.0; 3], [1.0; 3], 1.0);
        let mut v = ProbeVolume::zeros(&spec, vec![[0.0, 0.0, 1.0]]);
        // L_00 = ∫ 1 · Y_00 dω = 4π · 0.28209 = √(4π).
        let l00 = (4.0 * std::f64::consts::PI).sqrt() as f32;
        for i in (0..v.data.len()).step_by(SH * BANDS) {
            for b in 0..BANDS {
                v.data[i + b] = l00;
            }
        }
        for n in [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]] {
            let e = v.sample(0.0, [0.5, 0.5, 0.5], n);
            for b in 0..BANDS {
                assert!(
                    (e[b] as f64 - std::f64::consts::PI).abs() < 1e-4,
                    "{:?}: band {b} is {}",
                    n,
                    e[b]
                );
            }
        }
    }
}
