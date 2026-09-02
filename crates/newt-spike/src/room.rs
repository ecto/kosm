//! The room the marble is in.
//!
//! `audio.rs` renders the track's modes: the sound of the plate, dry, as if the
//! tray hung in a vacuum with a microphone glued to it. That is "ok if it were
//! in space". This module puts the tray on a table in a shoebox room and puts
//! a head 60 cm from it, and it does that the plain way: a **room impulse
//! response** per source position, convolved with the dry signal.
//!
//! Three parts to each RIR, and one of them per ear:
//!
//! 1. **Direct path.** 1/r, a fractional delay, and the air.
//! 2. **Early reflections**, by the **image-source method** (Allen & Berkley
//!    1979). A shoebox has an exact image lattice: mirror the source in each
//!    pair of walls, `order` deep. Each image is a tap at `distance/c` with
//!    `1/distance` spreading, `√(1−α)` per bounce off each wall it passed
//!    through, and a distance-dependent high-frequency loss for the air.
//! 3. **A late tail.** After the last order-`N` image the lattice is too sparse
//!    to be a room, so from there on the response is exponentially decaying
//!    filtered noise — the standard diffuse-field model, and the honest choice
//!    here over a feedback delay network: an FDN's decay is a tuning exercise
//!    in delay-line lengths, whereas a noise tail's envelope *is* `RT60`
//!    directly, and since we are convolving anyway it costs nothing extra. It
//!    is seeded, so it is the same tail every run, and independently seeded per
//!    ear, so the tail is decorrelated and the room sounds wide.
//!
//! `RT60` comes from **Eyring** on the room's own volume and area-weighted
//! absorption, `0.161 V / (−S ln(1−ᾱ))` — Eyring rather than Sabine because a
//! carpeted room is absorptive enough (ᾱ ≈ 0.15) that Sabine already runs long.
//!
//! **Two ears.** Two receivers `ear_spacing` apart on the room's x axis, so the
//! ITD falls out of the image-source geometry for free — every tap is simply
//! computed twice, from two points. The ILD is a head shadow: a tap arriving at
//! an ear from the far side of the head is low-passed and slightly attenuated,
//! by how far round the head it had to come. No HRTF database, no pinna, no
//! elevation — just the two cues a sphere with two microphones would give you.
//!
//! Coordinates: room corner at the origin, metres, x/y on the floor and z up.
//! The level's world origin (the plate top) sits at `table`, axes aligned.

/// Speed of sound in air at 20 °C (m/s).
pub const C: f64 = 343.0;

/// The room, the table in it, and the head listening.
#[derive(Clone, Copy, Debug)]
pub struct RoomSpec {
    /// Interior dimensions (m): x, y, z.
    pub dims: [f64; 3],
    /// Where the level's world origin — the plate top — sits (m, room frame).
    pub table: [f64; 3],
    /// Centre of the head (m, room frame).
    pub ear: [f64; 3],
    /// Absorption coefficients: floor, ceiling, walls (all four).
    pub absorb: [f64; 3],
    /// Maximum image-source reflection order.
    pub order: usize,
    /// Ear-to-ear spacing (m).
    pub ear_spacing: f64,
}

impl Default for RoomSpec {
    /// A small carpeted room with a table in the middle of it and someone
    /// leaning over the tray.
    ///
    /// 4.0 × 5.0 × 2.7 m; a 750 mm table; the head 600 mm back from the tray
    /// on the room's −y side and 450 mm above it. Absorption is mid-band
    /// (500 Hz–1 kHz) textbook: **0.30 floor** (cut-pile carpet on a concrete
    /// slab), **0.10 ceiling** (painted plaster), **0.08 walls** (painted
    /// plasterboard). That is ᾱ ≈ 0.14 over the whole envelope — a living room,
    /// not a studio and not a stairwell.
    fn default() -> Self {
        Self {
            dims: [4.0, 5.0, 2.7],
            table: [2.0, 2.0, 0.75],
            ear: [2.0, 1.4, 1.2],
            absorb: [0.30, 0.10, 0.08],
            order: 6,
            ear_spacing: 0.17,
        }
    }
}

impl RoomSpec {
    /// Enclosed volume (m³).
    pub fn volume(&self) -> f64 {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Total interior surface area (m²).
    pub fn surface(&self) -> f64 {
        let [x, y, z] = self.dims;
        2.0 * (x * y + x * z + y * z)
    }

    /// Area-weighted mean absorption ᾱ over floor, ceiling and the four walls.
    pub fn mean_absorption(&self) -> f64 {
        let [x, y, z] = self.dims;
        let (floor, walls) = (x * y, 2.0 * (x * z + y * z));
        (floor * self.absorb[0] + floor * self.absorb[1] + walls * self.absorb[2]) / self.surface()
    }

    /// `RT60` by Eyring: `0.161 V / (−S ln(1−ᾱ))`. Sabine is the ᾱ→0 limit of
    /// this and overstates the tail in an absorptive room.
    pub fn rt60(&self) -> f64 {
        let a = self.mean_absorption().clamp(1e-4, 0.999);
        0.161 * self.volume() / (-self.surface() * (1.0 - a).ln())
    }

    /// The two ears: left (−x) and right (+x) of the head centre.
    pub fn ears(&self) -> [[f64; 3]; 2] {
        let h = 0.5 * self.ear_spacing;
        [
            [self.ear[0] - h, self.ear[1], self.ear[2]],
            [self.ear[0] + h, self.ear[1], self.ear[2]],
        ]
    }

    /// A point in the level's frame (metres, origin on the plate top) placed in
    /// the room. The two frames are axis-aligned; only the origin moves.
    pub fn place(&self, level: [f64; 3]) -> [f64; 3] {
        [self.table[0] + level[0], self.table[1] + level[1], self.table[2] + level[2]]
    }

    /// Reflection coefficient of each of the six surfaces, in the order
    /// x=0, x=L, y=0, y=L, z=0 (floor), z=L (ceiling): `β = √(1−α)`.
    fn beta(&self) -> [f64; 6] {
        let w = (1.0 - self.absorb[2]).max(0.0).sqrt();
        let f = (1.0 - self.absorb[0]).max(0.0).sqrt();
        let c = (1.0 - self.absorb[1]).max(0.0).sqrt();
        [w, w, w, w, f, c]
    }
}

// ---- one-pole filters --------------------------------------------------------

/// A one-pole low-pass, `y = a·y + (1−a)·x`, applied in place.
fn lowpass(buf: &mut [f64], a: f64) {
    let mut y = 0.0;
    for x in buf.iter_mut() {
        y = a * y + (1.0 - a) * *x;
        *x = y;
    }
}

/// The pole `a` whose one-pole low-pass has magnitude `target` at `hz`.
/// Bisected — closed forms for this exist but this is once per bucket.
fn pole_for(hz: f64, sr: f64, target: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * hz / sr;
    let mag = |a: f64| {
        let (re, im) = (1.0 - a * w.cos(), a * w.sin());
        (1.0 - a) / (re * re + im * im).sqrt()
    };
    let (mut lo, mut hi) = (0.0, 0.999_999);
    if mag(lo) <= target {
        return lo;
    }
    for _ in 0..60 {
        let m = 0.5 * (lo + hi);
        if mag(m) > target { lo = m } else { hi = m }
    }
    0.5 * (lo + hi)
}

// ---- the impulse response ----------------------------------------------------

/// What one RIR turned out to be, for the summary line and the tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct RirStats {
    /// How many image sources were summed (both ears see the same lattice).
    pub images: usize,
    /// Energy in the direct path (the first arrival, ±1 ms), summed over ears.
    pub direct_energy: f64,
    /// Energy in everything after it.
    pub reverb_energy: f64,
    /// Sample index of the direct arrival at the left ear.
    pub direct_sample: usize,
}

impl RirStats {
    /// Direct-to-reverberant ratio (dB).
    pub fn drr_db(&self) -> f64 {
        10.0 * (self.direct_energy / self.reverb_energy.max(1e-30)).log10()
    }
}

/// Air absorption, as a fraction of the pressure left at 8 kHz after `d` metres.
/// α_air ≈ 0.02 Np/m at 8 kHz, 20 °C, 50 % RH (ISO 9613-1 order of magnitude) —
/// a gentle high-shelf loss that grows with path length, which is what makes a
/// far wall sound far rather than merely quiet.
fn air_hf_gain(d: f64) -> f64 {
    (-0.02 * d).exp()
}

/// Number of distance buckets the images are sorted into before the air filter
/// is run over each. Filtering every tap separately would be exact and slow;
/// six buckets over the whole lattice is inaudible against the real thing.
const AIR_BUCKETS: usize = 6;
/// Shadow buckets, same trick for the head.
const SHADOW_BUCKETS: usize = 5;

/// The stereo room impulse response for one source position (room frame).
///
/// Returns `[left, right]` and what the lattice did.
pub fn rir(room: &RoomSpec, src: [f64; 3], sr: f64) -> ([Vec<f64>; 2], RirStats) {
    let rt60 = room.rt60();
    let [lx, ly, lz] = room.dims;
    let beta = room.beta();
    let ears = room.ears();

    // Every image, once: mirror flags p and lattice translations n per axis.
    // Allen & Berkley: x = (1−2p)·sx + 2n·L, with |n−p| bounces off the low
    // wall and |n| off the high one; the order is those six counts summed.
    let nmax = room.order as i32;
    let mut images: Vec<([f64; 3], f64)> = Vec::new();
    for px in 0..2 {
        for py in 0..2 {
            for pz in 0..2 {
                for nx in -nmax..=nmax {
                    for ny in -nmax..=nmax {
                        for nz in -nmax..=nmax {
                            let cnt = [
                                ((nx - px).abs(), nx.abs()),
                                ((ny - py).abs(), ny.abs()),
                                ((nz - pz).abs(), nz.abs()),
                            ];
                            let order: i32 = cnt.iter().map(|(a, b)| a + b).sum();
                            if order > nmax {
                                continue;
                            }
                            let g = beta[0].powi(cnt[0].0) * beta[1].powi(cnt[0].1)
                                * beta[2].powi(cnt[1].0)
                                * beta[3].powi(cnt[1].1)
                                * beta[4].powi(cnt[2].0)
                                * beta[5].powi(cnt[2].1);
                            let pos = [
                                (1 - 2 * px) as f64 * src[0] + 2.0 * nx as f64 * lx,
                                (1 - 2 * py) as f64 * src[1] + 2.0 * ny as f64 * ly,
                                (1 - 2 * pz) as f64 * src[2] + 2.0 * nz as f64 * lz,
                            ];
                            images.push((pos, g));
                        }
                    }
                }
            }
        }
    }

    let mut ds: Vec<f64> =
        images.iter().map(|(p, _)| dist(*p, ears[0]).max(dist(*p, ears[1]))).collect();
    ds.sort_by(f64::total_cmp);
    let max_d = *ds.last().unwrap_or(&1.0);
    // The **mixing time**: where the image lattice stops being a set of
    // distinguishable reflections and starts being a diffuse field. The corner
    // of an order-N lattice is a lonely outlier — matching the tail's level
    // there would match it to almost nothing — so take the 60th percentile of
    // the arrivals, which is where the taps are densest, and hand over there.
    let t_mix = ds[ds.len() * 3 / 5] / C;
    let n = ((max_d / C + rt60 * 1.2) * sr) as usize + 64;

    let mut out = [vec![0.0; n], vec![0.0; n]];
    let mut stats = RirStats { images: images.len(), ..Default::default() };
    let mut first = usize::MAX;

    for (e, ear) in ears.iter().enumerate() {
        // buckets[shadow][air]: taps that will share a filter pass.
        let mut buckets = vec![vec![0.0; n]; AIR_BUCKETS * SHADOW_BUCKETS];
        // Which way this ear faces: away from the head centre, along ±x.
        let facing = if e == 0 { -1.0 } else { 1.0 };
        for (pos, g) in &images {
            let d = dist(*pos, *ear).max(1e-3);
            // How far round the head the wave had to bend: 0 for anything on
            // this ear's side of the interaural plane — a sound in front of you
            // is not shadowed by your own head — rising to 1 for one directly
            // opposite, which is the only direction that is all the way round.
            let cos = (pos[0] - room.ear[0]) / d.max(1e-6) * facing;
            let s = (-cos).clamp(0.0, 1.0);
            let a_i =
                ((d / max_d.max(1e-6)) * (AIR_BUCKETS - 1) as f64).round() as usize % AIR_BUCKETS;
            let s_i = (s * (SHADOW_BUCKETS - 1) as f64).round() as usize;
            let amp = g / d * (1.0 - 0.35 * s);
            // Fractional delay by linear interpolation between the two
            // neighbouring samples — a marble is not a click track, and the
            // ITD we care about is tens of samples wide.
            let tau = d / C * sr;
            let k = tau.floor() as usize;
            let frac = tau - k as f64;
            let b = &mut buckets[s_i * AIR_BUCKETS + a_i];
            if k + 1 < n {
                b[k] += amp * (1.0 - frac);
                b[k + 1] += amp * frac;
            }
            if e == 0 {
                first = first.min(k);
            }
        }
        for (i, b) in buckets.iter_mut().enumerate() {
            let (s_i, a_i) = (i / AIR_BUCKETS, i % AIR_BUCKETS);
            let d = (a_i as f64 / (AIR_BUCKETS - 1) as f64) * max_d;
            lowpass(b, pole_for(8000.0, sr, air_hf_gain(d)));
            let s = s_i as f64 / (SHADOW_BUCKETS - 1) as f64;
            if s > 0.0 {
                // A head is ~1 kHz worth of obstacle: full shadow rolls off
                // from there, no shadow leaves the tap alone.
                lowpass(b, pole_for(1000.0, sr, 1.0 - 0.85 * s));
            }
            for (o, &v) in out[e].iter_mut().zip(b.iter()) {
                *o += v;
            }
        }

        // Hand over: the images fade out over the same 20 ms the noise fades
        // in, so the late field is described once, not twice. Past the mixing
        // time the lattice is only pretending to be a room anyway.
        let k_mix = (t_mix * sr) as usize;
        let fade = (0.020 * sr) as usize;
        for (i, o) in out[e].iter_mut().enumerate().skip(k_mix) {
            let w = ((i - k_mix) as f64 / fade as f64).min(1.0);
            *o *= 0.5 + 0.5 * (std::f64::consts::PI * w).cos();
        }

        // The tail: decorrelated seeded noise under the Eyring envelope,
        // low-passed (absorption eats the top first), spliced in at t_mix and
        // level-matched to the last 10 ms of the image sum so the decay is
        // continuous rather than a step.
        let mut tail = vec![0.0; n];
        let mut rng = Rng(0x51ed_2701 ^ (e as u64 + 1) * 0x9e37_79b9_7f4a_7c15);
        // Faded in over 20 ms rather than switched on, because the handover is
        // a blur and not an event: on either side of it the same diffuse field
        // is being described twice, once as taps and once as noise.
        for (i, t) in tail.iter_mut().enumerate().skip(k_mix) {
            let sec = i as f64 / sr;
            let w = ((i - k_mix) as f64 / fade as f64).min(1.0);
            let w = 0.5 - 0.5 * (std::f64::consts::PI * w).cos();
            *t = w * rng.next() * (-6.907 * sec / rt60).exp();
        }
        lowpass(&mut tail, pole_for(4000.0, sr, 0.5));
        let win = (0.020 * sr) as usize;
        let energy = |b: &[f64], a: usize, c: usize| -> f64 {
            b[a.min(b.len())..c.min(b.len())].iter().map(|v| v * v).sum()
        };
        let early = energy(&out[e], k_mix.saturating_sub(win), k_mix);
        let raw = energy(&tail, k_mix + fade, k_mix + fade + win);
        let scale = if raw > 0.0 { (early / raw).sqrt() } else { 0.0 };
        for (o, t) in out[e].iter_mut().zip(tail.iter()) {
            *o += scale * t;
        }
    }

    stats.direct_sample = first;
    let lo = first.saturating_sub((0.001 * sr) as usize);
    let hi = (first + (0.001 * sr) as usize).min(n);
    for e in 0..2 {
        stats.direct_energy += out[e][lo..hi].iter().map(|v| v * v).sum::<f64>();
        stats.reverb_energy += out[e][hi..].iter().map(|v| v * v).sum::<f64>();
    }
    (out, stats)
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// The same xorshift as `audio.rs` — the room is as deterministic as the modes.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

// ---- convolution -------------------------------------------------------------

/// In-place iterative radix-2 FFT (`inv` for the inverse, unnormalized).
fn fft(re: &mut [f64], im: &mut [f64], inv: bool) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = 2.0 * std::f64::consts::PI / len as f64 * if inv { 1.0 } else { -1.0 };
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0_f64, 0.0_f64);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let (vr, vi) = (
                    re[i + k + len / 2] * cr - im[i + k + len / 2] * ci,
                    re[i + k + len / 2] * ci + im[i + k + len / 2] * cr,
                );
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// `a ∗ b`, by FFT. The RIRs are tens of thousands of taps long and the dry
/// signal is a couple of seconds; direct convolution is minutes, this is a
/// blink.
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let out_len = a.len() + b.len() - 1;
    let n = out_len.next_power_of_two();
    let (mut ar, mut ai) = (vec![0.0; n], vec![0.0; n]);
    let (mut br, mut bi) = (vec![0.0; n], vec![0.0; n]);
    ar[..a.len()].copy_from_slice(a);
    br[..b.len()].copy_from_slice(b);
    fft(&mut ar, &mut ai, false);
    fft(&mut br, &mut bi, false);
    for i in 0..n {
        let (r, im) = (ar[i] * br[i] - ai[i] * bi[i], ar[i] * bi[i] + ai[i] * br[i]);
        ar[i] = r;
        ai[i] = im;
    }
    fft(&mut ar, &mut ai, true);
    ar.truncate(out_len);
    ar.iter().map(|v| v / n as f64).collect()
}

// ---- the mix -----------------------------------------------------------------

/// The one line the run prints about the room.
#[derive(Clone, Copy, Debug)]
pub struct Summary {
    /// Eyring RT60 (s).
    pub rt60: f64,
    /// Image sources per RIR.
    pub images: usize,
    /// Distance from the head centre to the reference source (m).
    pub ear_distance: f64,
    /// Direct-to-reverberant ratio at the reference source (dB).
    pub drr_db: f64,
}

/// Put the dry buses in the room: convolve each with the RIR of its own source
/// position and sum, interleaved stereo, peak-normalized to −1 dBFS.
///
/// The marble moves, so one RIR would be a lie; but impacts are few and the
/// track is small, so instead of crossfading a continuous position we render
/// the dry sound onto a handful of **anchor** positions (release, cup, end
/// wall) and give each anchor its own RIR. Every contact is dry-rendered onto
/// its nearest anchor, so the pan and the reflection pattern move with the
/// marble without any crossfade to smear the transients.
///
/// `buses` is `(level-frame position (m), dry mono)`. `reference` picks which
/// bus the summary's DRR and distance describe.
pub fn mix(
    room: &RoomSpec,
    buses: &[([f64; 3], Vec<f64>)],
    reference: usize,
    sr: f64,
    duration: f64,
) -> (Vec<f32>, Summary) {
    let n = (duration * sr) as usize;
    let mut acc = [vec![0.0; n], vec![0.0; n]];
    let mut summary =
        Summary { rt60: room.rt60(), images: 0, ear_distance: 0.0, drr_db: 0.0 };
    for (i, (pos, dry)) in buses.iter().enumerate() {
        let src = room.place(*pos);
        let (h, stats) = rir(room, src, sr);
        if i == reference {
            summary.images = stats.images;
            summary.ear_distance = dist(src, room.ear);
            summary.drr_db = stats.drr_db();
        }
        if dry.iter().all(|&v| v == 0.0) {
            continue;
        }
        for e in 0..2 {
            let wet = convolve(dry, &h[e]);
            for (a, w) in acc[e].iter_mut().zip(wet.iter()) {
                *a += w;
            }
        }
    }
    let peak = acc.iter().flatten().fold(1e-12_f64, |p, &v| p.max(v.abs()));
    let norm = 0.891 / peak;
    let mut out = Vec::with_capacity(2 * n);
    for i in 0..n {
        out.push((acc[0][i] * norm) as f32);
        out.push((acc[1][i] * norm) as f32);
    }
    (out, summary)
}

/// Write interleaved stereo samples as a 16-bit PCM WAV.
pub fn write_wav_stereo(path: &std::path::Path, samples: &[f32], sr: f64) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: sr as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    w.finalize()?;
    Ok(())
}

// ---- tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44_100.0;

    fn small() -> RoomSpec {
        RoomSpec { order: 4, ..Default::default() }
    }

    #[test]
    fn rt60_falls_as_the_room_gets_more_absorbent() {
        let mut r = small();
        let mut prev = f64::INFINITY;
        for a in [0.05, 0.1, 0.2, 0.4, 0.8] {
            r.absorb = [a, a, a];
            let t = r.rt60();
            assert!(t.is_finite() && t > 0.0);
            assert!(t < prev, "RT60 {t} at α={a} is not below {prev}");
            prev = t;
        }
        // A carpeted living room, not a cathedral and not an anechoic chamber.
        assert!((0.2..1.5).contains(&small().rt60()), "{}", small().rt60());
    }

    #[test]
    fn the_itd_flips_sign_across_the_midline() {
        let room = small();
        let onset = |b: &[f64]| b.iter().position(|v| v.abs() > 1e-6).unwrap();
        // A source 0.5 m to the room's −x side of the head, then +x.
        let mut prev = None;
        for side in [-0.5, 0.5] {
            let src = [room.ear[0] + side, room.ear[1] + 0.6, room.ear[2]];
            let (h, _) = rir(&room, src, SR);
            let d = onset(&h[0]) as i64 - onset(&h[1]) as i64;
            assert!(d != 0, "no ITD at all for offset {side}");
            // 17 cm of head is at most ~22 samples at 44.1 kHz.
            assert!(d.abs() < 30, "ITD {d} samples is impossibly large");
            if let Some(p) = prev {
                assert!(p * d < 0, "ITD did not flip across the midline: {p} then {d}");
            }
            prev = Some(d);
        }
    }

    #[test]
    fn the_response_decays_monotonically_after_the_direct_sound() {
        let room = small();
        let (h, stats) = rir(&room, room.place([0.09, 0.0, 0.01]), SR);
        let win = (0.050 * SR) as usize;
        let mut k = stats.direct_sample;
        let mut prev = f64::INFINITY;
        let mut windows = 0;
        while k + win < h[0].len() {
            let e: f64 = (0..2).map(|e| h[e][k..k + win].iter().map(|v| v * v).sum::<f64>()).sum();
            assert!(e < prev, "50 ms window {windows} has energy {e} ≥ {prev}");
            prev = e;
            k += win;
            windows += 1;
        }
        assert!(windows >= 4, "only {windows} windows to check");
    }

    #[test]
    fn the_image_lattice_grows_with_order_and_shrinks_with_absorption() {
        let mut r = small();
        let (_, a) = rir(&r, r.place([0.0, 0.0, 0.0]), SR);
        r.order = 6;
        let (_, b) = rir(&r, r.place([0.0, 0.0, 0.0]), SR);
        assert!(b.images > a.images);
        // More absorption means less energy behind the direct sound.
        let quiet = RoomSpec { absorb: [0.7, 0.7, 0.7], ..small() };
        let (_, c) = rir(&quiet, quiet.place([0.0, 0.0, 0.0]), SR);
        assert!(c.drr_db() > a.drr_db(), "{} vs {}", c.drr_db(), a.drr_db());
    }

    #[test]
    fn convolution_matches_the_direct_sum() {
        let a: Vec<f64> = (0..40).map(|i| (i as f64 * 0.3).sin()).collect();
        let b: Vec<f64> = (0..17).map(|i| (i as f64 * 0.7).cos()).collect();
        let got = convolve(&a, &b);
        for (k, g) in got.iter().enumerate() {
            let want: f64 = (0..b.len())
                .filter(|j| k >= *j && k - j < a.len())
                .map(|j| a[k - j] * b[j])
                .sum();
            assert!((g - want).abs() < 1e-9, "tap {k}: {g} vs {want}");
        }
    }
}


