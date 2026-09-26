//! The wine glass, phase 1: the empty goblet, tapped.
//!
//! `docs/plans/2026-09-25-wine-glass-design.md` is the whole sim; this is its
//! first link. The bowl is a thin shell of revolution (`kosm::shell`), the
//! substance is the library's soda-lime glass through its modal facet, and a
//! fingernail tap on the rim becomes `glass.wav`: every mode a damped
//! sinusoid, the (2, 0) oval the note.
//!
//! `kosm run glass --out out/` — a second or two. Knobs: `rim_r`, `bowl_h`,
//! `base_r`, `wall_t` (millimetres), `contact_ms`, `seconds`.

use kosm::prelude::*;
use kosm::shell::{self, Profile, ShellMode};

const SR: f64 = 44_100.0;
const SEGMENTS: usize = 60;

fn params(args: &kosm_cli::Args) -> Vec<Param> {
    let knob = |name: &str, default: f64| Param::new(name, args.value(name).and_then(|v| v.parse().ok()).unwrap_or(default));
    vec![
        knob("rim_r", 40.0),
        knob("bowl_h", 90.0),
        knob("base_r", 5.0),
        knob("wall_t", 1.2),
        knob("contact_ms", 0.15),
        knob("seconds", 4.0),
    ]
}

fn get(params: &[Param], name: &str) -> f64 {
    params.iter().find(|p| p.name == name).map(|p| p.value).expect("every knob has a default")
}

pub fn run(args: &kosm_cli::Args) -> anyhow::Result<()> {
    let params = params(args);
    let mm = |name| get(&params, name) * 1e-3;
    let glass = material::named("soda-lime glass").expect("soda-lime glass is in the library");
    let modal = glass.modal();

    let bowl = Profile::goblet(mm("rim_r"), mm("bowl_h"), mm("base_r"), SEGMENTS);
    let wall = mm("wall_t");
    let mut bank: Vec<ShellMode> = (1..=8).flat_map(|n| shell::modes(&bowl, wall, modal, n, 3)).collect();
    bank.sort_by(|a, b| a.hz.total_cmp(&b.hz));

    let note = bank.iter().find(|m| m.n == 2 && m.m == 0).expect("the bowl has an oval mode");
    println!("glass: (2,0) {:.1} Hz, rings {:.1} s to -60 dB", note.hz, 6.9 / note.decay);
    for m in bank.iter().take(10) {
        println!("  ({},{})  {:8.1} Hz  {:5.2}× the note", m.n, m.m, m.hz, m.hz / note.hz);
    }

    // a fingernail on the rim, just below its edge
    let rim = bowl.nodes() - 2;
    let area = bowl.area();
    let audio = shell::strike(&bank, rim, 1e-3, mm("contact_ms"), get(&params, "seconds"), SR, |hz| {
        kosm::audio::radiation_efficiency(hz, area, wall, modal)
    });
    let stereo: Vec<f32> = audio.iter().flat_map(|&s| [s, s]).collect();

    let mut rec = Recorder::new(args.out(), "glass", &params, 0)?;
    kosm::room::write_wav_stereo(&rec.path("glass.wav")?, &stereo, SR)?;
    rec.metric("ring_hz", note.hz)?;
    rec.metric("ring_t60_s", 6.9 / note.decay)?;
    rec.metric("rayleigh_ring_hz", shell::ring_hz(mm("rim_r"), wall, modal, 2))?;
    for m in bank.iter().filter(|m| m.m == 0 && m.n <= 5) {
        rec.metric(&format!("mode_{}_0_hz", m.n), m.hz)?;
    }
    rec.finish()?;
    Ok(())
}
