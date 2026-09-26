//! The wine glass: a goblet of red wine, tapped, filled and tuned.
//!
//! `docs/plans/2026-09-25-wine-glass-design.md` is the whole sim. The bowl is
//! a thin shell of revolution (`kosm::shell`), the substances are the
//! library's soda-lime glass (its modal facet) and red wine (its fluid facet),
//! and a fingernail tap on the rim becomes sound: every mode a damped
//! sinusoid, the (2, 0) oval the note.
//!
//! `kosm run glass --out out/` writes
//! - `glass.wav`: one tap at the `fill` knob's level,
//! - `fill_sweep.wav`: the glass tapped from empty to nearly full,
//! - `metrics.json`: the modes, the fill law against French's, and the fill
//!   that plays `target_hz` (A4 by default), found by Newton on the
//!   eigenproblem's adjoint and checked against central differences.
//!
//! Knobs: `rim_r`, `bowl_h`, `base_r`, `wall_t`, `fill` (millimetres),
//! `contact_ms`, `seconds`, `target_hz`.

use kosm::prelude::*;
use kosm::shell::{self, Liquid, Profile, ShellMode};

const SR: f64 = 44_100.0;
const SEGMENTS: usize = 40;

fn params(args: &kosm_cli::Args) -> Vec<Param> {
    let knob = |name: &str, default: f64| Param::new(name, args.value(name).and_then(|v| v.parse().ok()).unwrap_or(default));
    vec![
        knob("rim_r", 40.0),
        knob("bowl_h", 90.0),
        knob("base_r", 5.0),
        knob("wall_t", 1.2),
        knob("fill", 0.0),
        knob("contact_ms", 0.15),
        knob("seconds", 4.0),
        knob("target_hz", 440.0),
    ]
}

fn get(params: &[Param], name: &str) -> f64 {
    params.iter().find(|p| p.name == name).map(|p| p.value).expect("every knob has a default")
}

/// The glass as the solver sees it.
pub struct Goblet {
    pub bowl: Profile,
    pub wall: f64,
    pub glass: kosm::audio::Material,
    pub wine: f64,
    pub rim_r: f64,
    pub height: f64,
}

impl Goblet {
    pub fn from_params(params: &[Param]) -> Self {
        let mm = |name| get(params, name) * 1e-3;
        let glass = material::named("soda-lime glass").expect("soda-lime glass is in the library");
        let wine = material::named("red wine").expect("red wine is in the library");
        Self {
            bowl: Profile::goblet(mm("rim_r"), mm("bowl_h"), mm("base_r"), SEGMENTS),
            wall: mm("wall_t"),
            glass: glass.modal(),
            wine: wine.fluid().density,
            rim_r: mm("rim_r"),
            height: mm("bowl_h"),
        }
    }

    fn liquid(&self, level: f64) -> Option<Liquid> {
        (level > 0.0).then_some(Liquid { density: self.wine, level })
    }

    /// Every mode worth hearing at a fill level (metres), lowest first.
    pub fn bank(&self, level: f64) -> Vec<ShellMode> {
        let mut bank: Vec<ShellMode> = (1..=7).flat_map(|n| shell::modes_filled(&self.bowl, self.wall, self.glass, n, 2, self.liquid(level))).collect();
        bank.sort_by(|a, b| a.hz.total_cmp(&b.hz));
        bank
    }

    /// The note: the (2, 0) mode.
    pub fn note(&self, level: f64) -> ShellMode {
        shell::modes_filled(&self.bowl, self.wall, self.glass, 2, 1, self.liquid(level)).remove(0)
    }

    /// d note / d level, from the adjoint.
    pub fn slope(&self, level: f64) -> f64 {
        shell::d_hz_d_level(&self.bowl, &self.note(level), Liquid { density: self.wine, level })
    }

    fn tap(&self, level: f64, contact_s: f64, seconds: f64) -> Vec<f32> {
        let area = self.bowl.area();
        let (wall, glass) = (self.wall, self.glass);
        // a fingernail on the rim, just below its edge
        shell::strike(&self.bank(level), self.bowl.nodes() - 2, 1e-3, contact_s, seconds, SR, |hz| {
            kosm::audio::radiation_efficiency(hz, area, wall, glass)
        })
    }

    /// Newton on the fill: the level whose note is `target` Hz, or `None` when
    /// the glass cannot reach it. Each step reads the slope off the adjoint.
    pub fn tune(&self, target: f64) -> Option<(f64, Vec<(f64, f64)>)> {
        let top = self.height * 0.97;
        if !(self.note(top).hz..=self.note(0.0).hz).contains(&target) {
            return None;
        }
        let mut level = 0.6 * self.height;
        let mut path = vec![];
        for _ in 0..30 {
            let hz = self.note(level).hz;
            path.push((level, hz));
            if (hz - target).abs() < 0.01 {
                return Some((level, path));
            }
            let slope = self.slope(level);
            // the flat bottom of the fill law has almost no slope: bisect-ish there
            let step = if slope.abs() > 1.0 { (target - hz) / slope } else { 0.1 * self.height };
            level = (level + step.clamp(-0.2 * self.height, 0.2 * self.height)).clamp(1e-4, top);
        }
        None
    }
}

pub fn run(args: &kosm_cli::Args) -> anyhow::Result<()> {
    let params = params(args);
    let goblet = Goblet::from_params(&params);
    let (contact, seconds) = (get(&params, "contact_ms") * 1e-3, get(&params, "seconds"));
    let fill = get(&params, "fill") * 1e-3;
    let mut rec = Recorder::new(args.out(), "glass", &params, 0)?;

    // one tap at the knob's fill
    let bank = goblet.bank(fill);
    let note = bank.iter().find(|m| m.n == 2 && m.m == 0).expect("the bowl has an oval mode").clone();
    println!("glass: fill {:.0} mm, (2,0) {:.1} Hz, rings {:.1} s to -60 dB", fill * 1e3, note.hz, 6.9 / note.decay);
    for m in bank.iter().take(8) {
        println!("  ({},{})  {:8.1} Hz  {:5.2}× the note", m.n, m.m, m.hz, m.hz / note.hz);
    }
    write_wav(&rec.path("glass.wav")?, &goblet.tap(fill, contact, seconds))?;
    rec.metric("ring_hz", note.hz)?;
    rec.metric("ring_t60_s", 6.9 / note.decay)?;
    for m in bank.iter().filter(|m| m.m == 0 && m.n <= 5) {
        rec.metric(&format!("mode_{}_0_hz", m.n), m.hz)?;
    }

    // the sweep: empty to nearly full, one tap a step
    let empty = goblet.note(0.0).hz;
    let steps = 10;
    let mut sweep = vec![];
    let mut law = vec![];
    for i in 0..=steps {
        let level = goblet.height * 0.9 * i as f64 / steps as f64;
        let hz = goblet.note(level).hz;
        let french = shell::french_ratio(goblet.rim_r, goblet.height, goblet.wall, goblet.glass.rho, goblet.wine, level);
        println!("  fill {:4.1} mm  {:6.1} Hz  ratio {:.3}  French {:.3}", level * 1e3, hz, hz / empty, french);
        law.push([level * 1e3, hz, hz / empty, french]); // fill mm, Hz, ratio, French
        sweep.extend(goblet.tap(level, contact, 0.8));
    }
    write_wav(&rec.path("fill_sweep.wav")?, &sweep)?;
    rec.metric("fill_law", law)?;

    // fill it to play the target
    let target = get(&params, "target_hz");
    match goblet.tune(target) {
        Some((level, path)) => {
            let h = 1e-5;
            let fd = (goblet.note(level + h).hz - goblet.note(level - h).hz) / (2.0 * h);
            let adj = goblet.slope(level);
            println!("tune: {target} Hz at fill {:.2} mm in {} steps; slope {adj:.1} Hz/m (fd {fd:.1})", level * 1e3, path.len());
            rec.metric("tune_fill_mm", level * 1e3)?;
            rec.metric("tune_steps", path.len())?;
            rec.metric("tune_slope_adjoint_hz_per_m", adj)?;
            rec.metric("tune_slope_fd_hz_per_m", fd)?;
            write_wav(&rec.path("tuned.wav")?, &goblet.tap(level, contact, seconds))?;
        }
        None => println!("tune: {target} Hz is out of this glass's range"),
    }
    rec.finish()?;
    Ok(())
}

fn write_wav(path: &std::path::Path, mono: &[f32]) -> anyhow::Result<()> {
    let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();
    kosm::room::write_wav_stereo(path, &stereo, SR)
}
