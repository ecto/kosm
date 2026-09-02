//! The marble, heard.
//!
//! No samples. Every sound in `out/marble.wav` is a sum of damped sinusoids
//! whose frequencies come from the level's own geometry and materials, excited
//! by the contact events of the same rollout that draws the frames.
//!
//! **Where the modes come from.** `vcad-kernel-acoustics` is the workspace's
//! acoustics kernel, and its structural half (`strike`) is a *free-free
//! Euler–Bernoulli bar*: `BarSpec` in, `fem_hz` (hole-aware Hermite-beam FEM)
//! or `closed_form_hz` out, plus `free_free_beta_l` / `mode_shape` for the
//! strike gains. There is no plate, shell or solid eigensolver in it — the rest
//! of the crate is air-side (Helmholtz cavities, ports, baffled pistons), which
//! is not what a struck track needs. So: **the bar solver is what we use**, run
//! on each part of the track along the axis that governs it, and for the plate
//! along *both* in-plane axes, which is a 1-D stand-in for the 2-D plate
//! spectrum and is the honest limit of what the crate offers today. The
//! marble is a sphere and no bar; its modes come from Lamb's closed-form
//! radial (breathing) frequency equation, implemented here (see
//! [`sphere_radial_hz`]).
//!
//! **The marble is inaudible, and that is the physics.** A 10 mm glass sphere's
//! lowest radial mode is ~450 kHz. It is computed, printed and then filtered
//! out by the audible-band gate, because that is where it actually lies. The
//! glassy edge you hear is the *track* rung by a very short Hertzian contact:
//! a hard, light, stiff marble is a bright hammer, not a bell.
//!
//! Units: SI everywhere inside this module (metres, kg, Pa, seconds).

use vcad_kernel_acoustics::strike::{BarSpec, fem_hz, free_free_beta_l, mode_shape};

/// Render sample rate (Hz).
pub const SR: f64 = 44_100.0;

// ---- materials ---------------------------------------------------------------

/// An isotropic elastic material: what a part is made of, acoustically.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// Density (kg/m³).
    pub rho: f64,
    /// Young's modulus (Pa).
    pub e: f64,
    /// Poisson's ratio.
    pub nu: f64,
    /// Structural loss factor η; Q = 1/η.
    pub loss: f64,
}

impl Material {
    /// Quality factor of every mode (frequency-independent hysteretic damping).
    fn q(self) -> f64 {
        1.0 / self.loss
    }
    /// Dilatational (longitudinal) wave speed of the bulk solid.
    fn c_long(self) -> f64 {
        (self.e * (1.0 - self.nu) / (self.rho * (1.0 + self.nu) * (1.0 - 2.0 * self.nu))).sqrt()
    }
    /// Shear wave speed.
    fn c_shear(self) -> f64 {
        (self.e / (2.0 * self.rho * (1.0 + self.nu))).sqrt()
    }
    /// Plane-strain contact modulus E/(1−ν²), the Hertz half of a pair.
    fn e_contact(self) -> f64 {
        self.e / (1.0 - self.nu * self.nu)
    }
}

/// Printed PLA — the track. Defaults; the level can override them with
/// `track_density`, `track_e`, `track_nu`, `track_loss` `defparam`s.
pub const PLA: Material = Material { rho: 1240.0, e: 3.5e9, nu: 0.36, loss: 0.03 };

/// Soda-lime glass — the marble. Overridable as `marble_density`, `marble_e`,
/// `marble_nu`, `marble_loss`.
pub const GLASS: Material = Material { rho: 2500.0, e: 70e9, nu: 0.23, loss: 0.001 };

/// Audible band; modes outside it are computed and reported but not rendered.
const AUDIBLE: (f64, f64) = (20.0, 18_000.0);

// ---- modes -------------------------------------------------------------------

/// One mode of one part: where it sits, how fast it dies, and how strongly a
/// strike at a given point along the part's axis excites it.
#[derive(Clone, Copy, Debug)]
pub struct Mode {
    /// Frequency (Hz).
    pub hz: f64,
    /// Amplitude decay rate (1/s): `exp(−decay·t)`.
    pub decay: f64,
    /// Free-free eigenvalue βL, for the strike-position mode shape.
    pub beta_l: f64,
    /// Which of the part's two surface coordinates this mode runs along.
    pub axis: usize,
}

impl Mode {
    /// |φₙ| at a strike `frac ∈ [0,1]` along the mode's axis.
    fn shape(&self, uv: [f64; 2]) -> f64 {
        mode_shape(self.beta_l, uv[self.axis].clamp(0.0, 1.0)).abs()
    }
}

/// The modal bank of one named part.
#[derive(Clone, Debug)]
pub struct Bank {
    /// Part name, for the summary line.
    pub name: &'static str,
    /// Its modes, strongest (lowest) first.
    pub modes: Vec<Mode>,
    /// Radiating efficiency of the part relative to the plate.
    pub level: f64,
}

impl Bank {
    /// The `n` lowest frequencies, for printing.
    pub fn top(&self, n: usize) -> Vec<f64> {
        self.modes.iter().take(n).map(|m| m.hz).collect()
    }
    /// Only the modes that will actually be rendered.
    fn audible(&self) -> impl Iterator<Item = &Mode> {
        self.modes.iter().filter(|m| m.hz > AUDIBLE.0 && m.hz < AUDIBLE.1 && m.hz.is_finite())
    }
}

/// Free-free bending modes of a bar of `len` × `thk` (m), from
/// `vcad-kernel-acoustics`' Hermite-beam FEM, tagged with `axis`.
pub fn bar_modes(len: f64, wid: f64, thk: f64, mat: Material, count: usize, axis: usize) -> Vec<Mode> {
    let bar = BarSpec {
        length_mm: len * 1e3,
        width_mm: wid * 1e3,
        thickness_mm: thk * 1e3,
        holes_mm: Vec::new(),
        hole_dia_mm: 0.0,
        modulus_gpa: mat.e / 1e9,
        density_kg_m3: mat.rho,
    };
    let hz = fem_hz(&bar, count);
    let betas = free_free_beta_l(hz.len());
    let q = mat.q();
    hz.iter()
        .zip(betas)
        .map(|(&hz, beta_l)| Mode { hz, decay: std::f64::consts::PI * hz / q, beta_l, axis })
        .collect()
}

/// Lamb's radial (breathing) modes of a free homogeneous elastic sphere.
///
/// The stress-free surface condition on the purely radial spheroidal family
/// gives `tan x = x / (1 − x²/(4κ²))` with `x = ωR/c_L` and `κ = c_L/c_T`
/// (Love, *Elasticity* §194). Roots bracketed one per branch of `tan` and
/// bisected — the marble is the one part of the level that is a solid, not a
/// bar, and the acoustics crate has no solid eigensolver.
pub fn sphere_radial_hz(radius: f64, mat: Material, count: usize) -> Vec<f64> {
    let kappa = mat.c_long() / mat.c_shear();
    // sin x·(1 − x²/4κ²) − x·cos x = 0 — the pole-free form of tan x = x/(1 − x²/4κ²).
    let root_eq = |x: f64| x.sin() * (1.0 - x * x / (4.0 * kappa * kappa)) - x * x.cos();
    let mut out = Vec::with_capacity(count);
    let step = 1e-3;
    let mut prev = root_eq(step);
    let mut x = 2.0 * step;
    while out.len() < count && x < 200.0 {
        let cur = root_eq(x);
        if prev * cur < 0.0 {
            let (mut a, mut b) = (x - step, x);
            for _ in 0..80 {
                let m = 0.5 * (a + b);
                if root_eq(a) * root_eq(m) <= 0.0 { b = m } else { a = m }
            }
            out.push(0.5 * (a + b) * mat.c_long() / (2.0 * std::f64::consts::PI * radius));
        }
        prev = cur;
        x += step;
    }
    out
}

/// The marble's own bank. Physically ultrasonic; kept so the summary can say so.
pub fn marble_bank(radius: f64, mat: Material) -> Bank {
    let q = mat.q();
    let modes = sphere_radial_hz(radius, mat, 6)
        .into_iter()
        .map(|hz| Mode { hz, decay: std::f64::consts::PI * hz / q, beta_l: 4.730, axis: 0 })
        .collect();
    Bank { name: "marble (glass sphere, Lamb radial)", modes, level: 1.0 }
}

// ---- the track's parts -------------------------------------------------------

/// The geometry the sound needs, in metres, straight off the level's knobs.
#[derive(Clone, Copy, Debug)]
pub struct TrackSpec {
    /// Plate extent along x and y, and its thickness.
    pub plate: [f64; 3],
    /// Side/end wall height and thickness.
    pub wall: [f64; 2],
    /// Cup centre x, inside radius, wall thickness, height.
    pub cup: [f64; 4],
    /// Marble radius and mass.
    pub marble: (f64, f64),
    /// Track material.
    pub track_mat: Material,
    /// Marble material.
    pub marble_mat: Material,
}

/// Which part of the track a contact struck. Each is its own voice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// The plate top.
    Plate,
    /// A side or end wall.
    Wall,
    /// The cup ring.
    Cup,
}

impl TrackSpec {
    /// The plate: the bar solver run along x and along y, because a plate is
    /// two families of bending waves and the crate only knows about one at a
    /// time. Merged and sorted, this is a coarse but real 2-D spectrum.
    pub fn plate_bank(&self) -> Bank {
        let [lx, ly, t] = self.plate;
        let mut modes = bar_modes(lx, ly, t, self.track_mat, 6, 0);
        modes.extend(bar_modes(ly, lx, t, self.track_mat, 6, 1));
        modes.sort_by(|a, b| a.hz.total_cmp(&b.hz));
        Bank { name: "plate (PLA, free-free bar on both axes)", modes, level: 1.0 }
    }

    /// A side wall: a tall thin bar standing on edge, so its bending stiffness
    /// is set by the wall thickness and its length by the plate.
    pub fn wall_bank(&self) -> Bank {
        let [h, t] = self.wall;
        let modes = bar_modes(self.plate[0] + 2.0 * t, h, t, self.track_mat, 6, 0);
        Bank { name: "wall (PLA)", modes, level: 0.7 }
    }

    /// The cup: the kept arc of the ring, unrolled into a bar of that arc
    /// length, `cup_wall` thick and `cup_h` wide. Short and stiff — the high,
    /// rattly voice.
    pub fn cup_bank(&self) -> Bank {
        let [_, r, w, h] = self.cup;
        let arc = 2.0 * std::f64::consts::PI * (r + 0.5 * w) * (11.0 / 16.0);
        let modes = bar_modes(arc, h, w, self.track_mat, 6, 0);
        Bank { name: "cup (PLA arc)", modes, level: 1.3 }
    }

    /// Hertzian contact duration (s) for the marble striking the track at
    /// closing speed `v`: `t_c = 2.87 (m²/(R E*² v))^{1/5}` (Hertz; see
    /// Johnson, *Contact Mechanics* §11.2). It sets the excitation bandwidth —
    /// a short contact is a bright hammer.
    pub fn contact_time(&self, v: f64) -> f64 {
        let (r, m) = self.marble;
        let e_star = 1.0 / (1.0 / self.marble_mat.e_contact() + 1.0 / self.track_mat.e_contact());
        let v = v.abs().max(1e-3);
        2.87 * (m * m / (r * e_star * e_star * v)).powf(0.2)
    }
}

// ---- excitation --------------------------------------------------------------

/// One impact, harvested from the rollout.
#[derive(Clone, Copy, Debug)]
pub struct Impact {
    /// Time in the rollout (s).
    pub t: f64,
    /// Normal impulse magnitude (N·s) = m·|Δv|.
    pub impulse: f64,
    /// Closing speed at the impact (m/s), for the Hertz contact time.
    pub speed: f64,
    /// Which part was struck.
    pub part: Part,
    /// Where, as two normalized surface coordinates of that part.
    pub uv: [f64; 2],
}

/// One step of rolling contact: broadband excitation, not an event.
#[derive(Clone, Copy, Debug)]
pub struct Roll {
    /// Time (s).
    pub t: f64,
    /// Tangential speed at the contact (m/s).
    pub speed: f64,
    /// Where on the plate.
    pub uv: [f64; 2],
}

/// Everything the rollout heard.
#[derive(Clone, Debug, Default)]
pub struct Contacts {
    /// Discrete impacts.
    pub impacts: Vec<Impact>,
    /// Per-step rolling contact.
    pub rolls: Vec<Roll>,
}

// ---- synthesis ---------------------------------------------------------------

/// A tiny xorshift, so the rolling noise is reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// One-pole low-pass gain of the Hertz contact pulse at frequency `hz`.
fn contact_gain(hz: f64, tc: f64) -> f64 {
    let fc = 1.0 / tc;
    1.0 / (1.0 + (hz / fc).powi(2)).sqrt()
}

/// Add a struck bank's ring-down into `out`, starting at sample `start`.
fn strike_into(out: &mut [f64], start_in: usize, bank: &Bank, imp: &Impact, tc: f64, gain: f64) {
    let start = start_in.min(out.len());
    for m in bank.audible() {
        let a0 = gain * bank.level * imp.impulse * m.shape(imp.uv) * contact_gain(m.hz, tc) / m.hz.sqrt();
        if a0.abs() < 1e-12 {
            continue;
        }
        let w = 2.0 * std::f64::consts::PI * m.hz / SR;
        let d = (-m.decay / SR).exp();
        let mut amp = a0;
        for (k, o) in out[start..].iter_mut().enumerate() {
            *o += amp * (w * k as f64).sin();
            amp *= d;
            if amp.abs() < 1e-9 * a0.abs().max(1e-12) {
                break;
            }
        }
    }
}

/// Rolling: speed-scaled noise poured through the plate's modes as
/// two-pole resonators. Continuous excitation, not an event.
fn roll_into(out: &mut [f64], bank: &Bank, rolls: &[Roll], gain: f64) {
    if rolls.is_empty() {
        return;
    }
    let n = out.len();
    let mut noise = vec![0.0; n];
    let mut rng = Rng(0x9e3779b97f4a7c15);
    for r in rolls {
        let k = (r.t * SR) as usize;
        if k < n {
            noise[k] = rng.next() * r.speed;
        }
    }
    let mut drive = vec![0.0; n];
    for m in bank.audible() {
        // The contact point moves, so each mode is driven through its own
        // shape at wherever the marble is at that instant.
        drive.iter_mut().for_each(|d| *d = 0.0);
        for r in rolls {
            let k = (r.t * SR) as usize;
            if k < n {
                drive[k] = noise[k] * m.shape(r.uv);
            }
        }
        let r = (-m.decay / SR).exp();
        let w = 2.0 * std::f64::consts::PI * m.hz / SR;
        let (a1, a2) = (2.0 * r * w.cos(), -r * r);
        let g = gain * (1.0 - r) / m.hz.sqrt();
        let (mut y1, mut y2) = (0.0, 0.0);
        for (o, &x) in out.iter_mut().zip(drive.iter()) {
            let y = g * x + a1 * y1 + a2 * y2;
            y2 = y1;
            y1 = y;
            *o += y;
        }
    }
}

/// The whole soundtrack: impacts on their parts, plus the roll, peak-normalized
/// to −1 dBFS. `duration` includes the ring-out tail.
pub fn render(spec: &TrackSpec, banks: &[(Part, Bank)], contacts: &Contacts, duration: f64) -> Vec<f32> {
    let n = (duration * SR) as usize;
    let mut buf = vec![0.0_f64; n];
    let plate = banks.iter().find(|(p, _)| *p == Part::Plate).map(|(_, b)| b);
    for imp in &contacts.impacts {
        let start = (imp.t * SR) as usize;
        if start >= n {
            continue;
        }
        let tc = spec.contact_time(imp.speed);
        if let Some((_, bank)) = banks.iter().find(|(p, _)| *p == imp.part) {
            strike_into(&mut buf, start, bank, imp, tc, 1.0);
        }
        // Every strike also shakes the plate the part is printed onto.
        if let Some(b) = plate.filter(|_| imp.part != Part::Plate) {
            strike_into(&mut buf, start, b, imp, tc, 0.35);
        }
    }
    if let Some(b) = plate {
        roll_into(&mut buf, b, &contacts.rolls, 0.9);
    }
    let peak = buf.iter().fold(1e-12_f64, |p, &v| p.max(v.abs()));
    let norm = 0.891 / peak;
    buf.iter().map(|&v| (v * norm) as f32).collect()
}

/// Write mono f32 samples as a 16-bit PCM WAV.
pub fn write_wav(path: &std::path::Path, samples: &[f32]) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SR as u32,
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

    fn spec() -> TrackSpec {
        TrackSpec {
            plate: [0.300, 0.200, 0.010],
            wall: [0.030, 0.008],
            cup: [0.090, 0.022, 0.003, 0.014],
            marble: (0.010, 0.012),
            track_mat: PLA,
            marble_mat: GLASS,
        }
    }

    #[test]
    fn plate_modes_are_audible_and_finite() {
        let s = spec();
        for bank in [s.plate_bank(), s.wall_bank(), s.cup_bank()] {
            assert!(!bank.modes.is_empty(), "{} has no modes", bank.name);
            for m in &bank.modes {
                assert!(m.hz.is_finite() && m.decay.is_finite());
            }
            let audible = bank.audible().count();
            assert!(audible > 0, "{} has nothing in the audible band", bank.name);
        }
    }

    #[test]
    fn bending_frequency_scales_as_thickness_over_length_squared() {
        // Euler–Bernoulli: f ∝ t/L². The FEM should reproduce that.
        let f = |l: f64, t: f64| bar_modes(l, 0.2, t, PLA, 1, 0)[0].hz;
        let base = f(0.300, 0.010);
        assert!((f(0.300, 0.020) / base - 2.0).abs() < 1e-3, "thickness doubling should double f");
        assert!((f(0.600, 0.010) / base - 0.25).abs() < 1e-3, "length doubling should quarter f");
    }

    #[test]
    fn a_single_impact_decays() {
        let s = spec();
        let banks = vec![(Part::Plate, s.plate_bank())];
        let contacts = Contacts {
            impacts: vec![Impact {
                t: 0.0,
                impulse: 0.01,
                speed: 1.0,
                part: Part::Plate,
                uv: [0.3, 0.4],
            }],
            rolls: Vec::new(),
        };
        let buf = render(&s, &banks, &contacts, 1.0);
        let energy = |w: &[f32]| w.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        let half = buf.len() / 2;
        let (a, b) = (energy(&buf[..half]), energy(&buf[half..]));
        assert!(a > 0.0, "silence");
        assert!(b < 0.05 * a, "impact did not decay: {a} then {b}");
    }

    #[test]
    fn the_marble_rings_ultrasonically() {
        // A 10 mm glass sphere's lowest breathing mode is ~200 kHz, not a bell.
        let hz = sphere_radial_hz(0.010, GLASS, 3);
        assert!(hz[0] > 300e3 && hz[0] < 600e3, "sphere fundamental {} Hz", hz[0]);
        assert!(hz.windows(2).all(|w| w[1] > w[0]));
    }
}
