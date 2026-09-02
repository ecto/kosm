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

use vcad_kernel_acoustics::radiation::bessel_j1;
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
    /// Radiation efficiency σ of this mode: how much of its surface velocity
    /// actually becomes air. See [`radiation_efficiency`].
    pub rad: f64,
}

impl Mode {
    /// |φₙ| at a strike `frac ∈ [0,1]` along the mode's axis.
    fn shape(&self, uv: [f64; 2]) -> f64 {
        mode_shape(self.beta_l, uv[self.axis].clamp(0.0, 1.0)).abs()
    }
    /// Amplitude weight from the radiation efficiency: pressure goes as √σ.
    fn radiates(&self) -> f64 {
        self.rad.sqrt()
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
        .map(|(&hz, beta_l)| Mode { hz, decay: std::f64::consts::PI * hz / q, beta_l, axis, rad: 1.0 })
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
        .map(|hz| Mode { hz, decay: std::f64::consts::PI * hz / q, beta_l: 4.730, axis: 0, rad: 1.0 })
        .collect();
    Bank { name: "marble (glass sphere, Lamb radial)", modes, level: 1.0 }
}


// ---- radiation ---------------------------------------------------------------

/// Speed of sound in air (m/s), for the coincidence frequency.
const C_AIR: f64 = 343.0;

/// How much of a mode's surface velocity actually becomes sound.
///
/// A vibrating plate is a poor loudspeaker, and it is poor in a very specific,
/// frequency-dependent way — which is why a dry modal sum sounds buzzy: it
/// radiates every mode as if it were a piston. Two effects, multiplied:
///
/// 1. **The plate is small compared to the wavelength.** At low frequency the
///    two sides of the plate short-circuit each other around the edge. The
///    baffled-piston radiation resistance of an equivalent-area disc,
///    `σ = 1 − J₁(2ka)/(ka)` with `a = √(A/π)`, is the textbook version of
///    that; it is ∝ (ka)²/2 at low `ka` and → 1 above it. `J₁` here is
///    `vcad-kernel-acoustics`' own `radiation::bessel_j1` — that module is the
///    crate's air-side radiator (baffled piston, Rayleigh integral,
///    directivity) and has no *plate* radiation-efficiency function, so the
///    coincidence half below is ours, but the Bessel function is theirs.
/// 2. **Bending waves are slower than sound below coincidence.** Under the
///    critical frequency `f_c = c²/(1.8·c_L·t)` (with the plate longitudinal
///    speed `c_L = √(E/ρ(1−ν²))`) the bending wavelength is shorter than the
///    acoustic one and the plate only radiates from its edges; above it the
///    plate and the air are phase-matched and σ → 1. `min(1, f/f_c)` is the
///    crude, monotone stand-in for Maidanik's edge/corner-mode formulae.
///
/// The product tames exactly what needed taming: the big slow low modes.
pub fn radiation_efficiency(hz: f64, area: f64, thickness: f64, mat: Material) -> f64 {
    if hz <= 0.0 || area <= 0.0 {
        return 0.0;
    }
    let a = (area / std::f64::consts::PI).sqrt();
    let ka = 2.0 * std::f64::consts::PI * hz / C_AIR * a;
    let piston = if ka < 1e-6 { 0.0 } else { (1.0 - bessel_j1(2.0 * ka) / ka).clamp(0.0, 1.0) };
    let c_l = (mat.e / (mat.rho * (1.0 - mat.nu * mat.nu))).sqrt();
    let f_c = C_AIR * C_AIR / (1.8 * c_l * thickness);
    (piston * (hz / f_c).min(1.0)).clamp(1e-4, 1.0)
}

impl Bank {
    /// Weight every mode by the radiating panel it belongs to.
    fn radiating(mut self, area: f64, thickness: f64, mat: Material) -> Self {
        for m in &mut self.modes {
            m.rad = radiation_efficiency(m.hz, area, thickness, mat);
        }
        self
    }
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
            .radiating(lx * ly, t, self.track_mat)
    }

    /// A side wall: a tall thin bar standing on edge, so its bending stiffness
    /// is set by the wall thickness and its length by the plate.
    pub fn wall_bank(&self) -> Bank {
        let [h, t] = self.wall;
        let modes = bar_modes(self.plate[0] + 2.0 * t, h, t, self.track_mat, 6, 0);
        Bank { name: "wall (PLA)", modes, level: 0.7 }.radiating(
            (self.plate[0] + 2.0 * t) * h,
            t,
            self.track_mat,
        )
    }

    /// The cup: the kept arc of the ring, unrolled into a bar of that arc
    /// length, `cup_wall` thick and `cup_h` wide. Short and stiff — the high,
    /// rattly voice.
    pub fn cup_bank(&self) -> Bank {
        let [_, r, w, h] = self.cup;
        let arc = 2.0 * std::f64::consts::PI * (r + 0.5 * w) * (11.0 / 16.0);
        let modes = bar_modes(arc, h, w, self.track_mat, 6, 0);
        Bank { name: "cup (PLA arc)", modes, level: 1.3 }.radiating(arc * h, w, self.track_mat)
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
    /// Where in the level's frame (m), so the room knows where to put it.
    pub pos: [f64; 3],
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
    /// Where in the level's frame (m).
    pub pos: [f64; 3],
    /// Normal force pressing the marble into the plate (N).
    pub normal: f64,
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
        let a0 = gain
            * bank.level
            * imp.impulse
            * m.shape(imp.uv)
            * m.radiates()
            * contact_gain(m.hz, tc)
            / m.hz.sqrt();
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

/// Rolling, as surface roughness.
///
/// The old version poured one noise sample per physics step through the modes,
/// which is a train of impulses at 1 kHz wearing a noise costume — it sounded
/// synthetic because it *was* a synthesizer. A marble rolling on a printed
/// tray is doing something simpler and more specific: it is tracing the layer
/// lines. The nozzle laid them down every `asperity` metres (0.2 mm is a
/// normal layer/line spacing), so a marble at `v` m/s crosses them at `v/λ`
/// per second, and that — not the sample rate, not the step rate — is the
/// corner frequency of the excitation. Faster marble, brighter roll; that is
/// the whole trick, and it is the Stronge / Othman rolling-noise picture in its
/// simplest form.
///
/// So: white noise, band-limited by a one-pole at `f_c = v/λ`, scaled by the
/// normal force and the speed, is a **continuous force** on the plate; the
/// modes are two-pole resonators driven by it at whatever point the marble has
/// reached. The roughness itself is fixed and seeded, so the same run makes
/// the same noise; the level and the colour of it come from the physics.
fn roll_into(out: &mut [f64], bank: &Bank, rolls: &[Roll], asperity: f64, gain: f64) {
    if rolls.is_empty() {
        return;
    }
    let n = out.len();
    // The rollout samples contact at the physics step; the render wants it at
    // the sample rate. Hold-and-interpolate speed, load and position between
    // steps, so nothing changes discontinuously under the filter.
    let mut speed = vec![0.0; n];
    let mut load = vec![0.0; n];
    let mut uv = vec![[0.0_f64; 2]; n];
    for w in rolls.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (ka, kb) = ((a.t * SR) as usize, (b.t * SR) as usize);
        if ka >= n {
            break;
        }
        let kb = kb.min(n);
        for k in ka..kb {
            let f = if kb > ka { (k - ka) as f64 / (kb - ka) as f64 } else { 0.0 };
            speed[k] = a.speed + f * (b.speed - a.speed);
            load[k] = a.normal + f * (b.normal - a.normal);
            uv[k] = [a.uv[0] + f * (b.uv[0] - a.uv[0]), a.uv[1] + f * (b.uv[1] - a.uv[1])];
        }
    }
    // The roughness force: unit-variance noise through a speed-tracking
    // one-pole, times the load and the speed. `√(1−a²)` keeps the filter's
    // output variance at one whatever the corner does, so the level is the
    // physics and not the filter.
    let mut force = vec![0.0; n];
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let mut lp = 0.0;
    for k in 0..n {
        let e = rng.next();
        let fc = (speed[k] / asperity).clamp(20.0, 0.45 * SR);
        let a = (-2.0 * std::f64::consts::PI * fc / SR).exp();
        lp = a * lp + (1.0 - a * a).sqrt() * e;
        force[k] = gain * load[k] * speed[k] * lp;
    }
    for m in bank.audible() {
        let r = (-m.decay / SR).exp();
        let w = 2.0 * std::f64::consts::PI * m.hz / SR;
        let (a1, a2) = (2.0 * r * w.cos(), -r * r);
        let g = (1.0 - r) * m.radiates() / m.hz.sqrt();
        let (mut y1, mut y2) = (0.0, 0.0);
        for k in 0..n {
            let x = force[k] * m.shape(uv[k]);
            let y = g * x + a1 * y1 + a2 * y2;
            y2 = y1;
            y1 = y;
            out[k] += y;
        }
    }
}

/// The soundtrack, dry, split onto one bus per source **anchor**.
///
/// The marble moves, and a room impulse response is per position, so the dry
/// sound is rendered onto a handful of fixed positions instead of one: each
/// impact goes to whichever anchor it happened nearest, and the roll — which
/// is continuous and would not survive being cut up — goes whole to the anchor
/// nearest the middle of its own path. `room::mix` gives each bus its own RIR.
///
/// Anchors are level-frame metres. `duration` includes the ring-out.
pub fn render_dry(
    spec: &TrackSpec,
    banks: &[(Part, Bank)],
    contacts: &Contacts,
    duration: f64,
    anchors: &[[f64; 3]],
    roughness: f64,
) -> Vec<Vec<f64>> {
    let n = (duration * SR) as usize;
    let mut buses = vec![vec![0.0_f64; n]; anchors.len().max(1)];
    let nearest = |p: [f64; 3]| -> usize {
        anchors
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let d = |q: &[f64; 3]| {
                    (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) + (q[2] - p[2]).powi(2)
                };
                d(a).total_cmp(&d(b))
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    let plate = banks.iter().find(|(p, _)| *p == Part::Plate).map(|(_, b)| b);
    for imp in &contacts.impacts {
        let start = (imp.t * SR) as usize;
        if start >= n {
            continue;
        }
        let tc = spec.contact_time(imp.speed);
        let bus = &mut buses[nearest(imp.pos)];
        if let Some((_, bank)) = banks.iter().find(|(p, _)| *p == imp.part) {
            strike_into(bus, start, bank, imp, tc, 1.0);
        }
        // Every strike also shakes the plate the part is printed onto.
        if let Some(b) = plate.filter(|_| imp.part != Part::Plate) {
            strike_into(bus, start, b, imp, tc, 0.35);
        }
    }
    if let Some(b) = plate {
        let mut mid = [0.0; 3];
        for r in &contacts.rolls {
            for k in 0..3 {
                mid[k] += r.pos[k] / contacts.rolls.len() as f64;
            }
        }
        let i = nearest(mid);
        let bus = &mut buses[i];
        roll_into(bus, b, &contacts.rolls, roughness, 0.9);
    }
    buses
}

/// Layer-line spacing of a printed tray (m): the default asperity length.
pub const ROUGHNESS: f64 = 0.2e-3;

/// The dry mix, mono and peak-normalized — what `out/marble.wav` was before
/// there was a room. Kept because it is the thing the room is applied *to*,
/// and because the modal tests want a signal without a tail on it.
pub fn render(spec: &TrackSpec, banks: &[(Part, Bank)], contacts: &Contacts, duration: f64) -> Vec<f32> {
    let buses =
        render_dry(spec, banks, contacts, duration, &[[0.0, 0.0, 0.0]], ROUGHNESS);
    let peak = buses[0].iter().fold(1e-12_f64, |p, &v| p.max(v.abs()));
    let norm = 0.891 / peak;
    buses[0].iter().map(|&v| (v * norm) as f32).collect()
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
                pos: [0.0, 0.0, 0.0],
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

    #[test]
    fn radiation_tames_the_low_modes() {
        let s = spec();
        let plate = s.plate_bank();
        let lo = plate.modes.first().unwrap();
        let hi = plate.modes.iter().rev().find(|m| m.hz < 18_000.0).unwrap();
        assert!(lo.rad < 0.2 * hi.rad, "low mode σ={} vs high σ={}", lo.rad, hi.rad);
        // σ is a fraction of a piston, and monotone in frequency for one panel.
        let sig = |hz| radiation_efficiency(hz, 0.06, 0.010, PLA);
        assert!((0.0..=1.0).contains(&sig(100.0)) && (0.0..=1.0).contains(&sig(10_000.0)));
        assert!(sig(100.0) < sig(1_000.0) && sig(1_000.0) < sig(10_000.0));
    }

    #[test]
    fn rolling_noise_level_scales_with_speed() {
        let s = spec();
        let banks = vec![(Part::Plate, s.plate_bank())];
        let rms = |v: f64| {
            let rolls: Vec<Roll> = (0..1000)
                .map(|k| Roll {
                    t: k as f64 * 1e-3,
                    speed: v,
                    uv: [0.5, 0.5],
                    pos: [0.0, 0.0, 0.0],
                    normal: 0.12,
                })
                .collect();
            let contacts = Contacts { impacts: Vec::new(), rolls };
            let buses =
                render_dry(&s, &banks, &contacts, 1.0, &[[0.0, 0.0, 0.0]], ROUGHNESS);
            let n = buses[0].len() as f64;
            (buses[0].iter().map(|x| x * x).sum::<f64>() / n).sqrt()
        };
        let (a, b, c) = (rms(0.1), rms(0.4), rms(1.6));
        assert!(a > 0.0, "silence");
        assert!(b > 2.0 * a, "0.4 m/s ({b}) is not louder than 0.1 m/s ({a})");
        assert!(c > 2.0 * b, "1.6 m/s ({c}) is not louder than 0.4 m/s ({b})");
    }
}
