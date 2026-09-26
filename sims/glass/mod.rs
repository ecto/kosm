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
//! - `frame.png`: the goblet filled to the tuned level, on linen in low sun,
//!   with its caustic (`render.rs`); skipped by `--no-frame`.
//!
//! - `ripple.mp4` (with `--ripple_frames N`): the surface ripples the ring
//!   drives (`kosm::fluid::ripple`), seen as the sun's glint on the wine by
//!   the far wall, strobed at 1 + 1/48 of the period so it moves at a
//!   watchable pace. `--drive` scales the tap. (On the tablecloth the same
//!   ripples move the caustic by about 2 mm: below the photon gather radius,
//!   so the table view is the wrong place to look for them.)
//!
//! Knobs: `rim_r`, `bowl_h`, `base_r`, `wall_t`, `fill` (millimetres),
//! `contact_ms`, `seconds`, `target_hz`, `sun_el`, `sun_az` (degrees),
//! `width`, `height`, `spp`, `photons`, `exposure`, `drive`, `ripple_frames`, `white` (1 pours white wine).

mod render;

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
        knob("sun_el", 24.0),
        knob("sun_az", 30.0),
        knob("width", 640.0),
        knob("height", 400.0),
        knob("spp", 64.0),
        knob("photons", 2_000_000.0),
        knob("exposure", 0.7),
        knob("drive", 1.0),
        knob("white", 0.0),
        knob("ripple_frames", 0.0),
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
    pub wine_name: &'static str,
    pub rim_r: f64,
    pub height: f64,
}

impl Goblet {
    pub fn from_params(params: &[Param]) -> Self {
        let mm = |name| get(params, name) * 1e-3;
        let glass = material::named("soda-lime glass").expect("soda-lime glass is in the library");
        let wine_name = if get(params, "white") > 0.5 { "white wine" } else { "red wine" };
        let wine = material::named(wine_name).expect("the wine is in the library");
        Self {
            bowl: Profile::goblet(mm("rim_r"), mm("bowl_h"), mm("base_r"), SEGMENTS),
            wall: mm("wall_t"),
            glass: glass.modal(),
            wine: wine.fluid().density,
            wine_name,
            rim_r: mm("rim_r"),
            height: mm("bowl_h"),
        }
    }

    fn liquid(&self, level: f64) -> Option<Liquid> {
        (level > 0.0).then_some(Liquid { density: self.wine, level })
    }

    /// The whole goblet, stem and foot, for the renderer.
    fn shape(&self) -> render::Shape {
        render::Shape { wine: self.wine_name, inner: self.bowl.points.clone(), wall: self.wall, stem_r: 0.004, stem_h: 0.07, foot_r: 0.035, foot_t: 0.003 }
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

    /// The ripples a rim tap of `impulse` N·s leaves on wine at `level`:
    /// the (2, 0) mode's displacement at the waterline sets their amplitude.
    ///
    /// The solver's mass matrix leaves out the ∫cos² nθ dθ = π of the
    /// circumference, so a physically mass-normalized shape is φ/√π, and an
    /// impulse P at the rim (θ = 0) leaves the waterline moving with
    /// amplitude P φ(rim) φ(waterline) / (π ω).
    pub fn ripples(&self, level: f64, impulse: f64) -> kosm::fluid::Ripples {
        let note = self.note(level);
        let omega = std::f64::consts::TAU * note.hz;
        let rim = self.bowl.nodes() - 2;
        let at_level = self.bowl.points.iter().position(|q| q[1] >= level).unwrap_or(rim);
        let amp = impulse * (note.radial(rim) * note.radial(at_level)).abs() / (std::f64::consts::PI * omega);
        let r = self.shape().inner_r(level);
        let wine = material::named(self.wine_name).expect("the wine is in the library").fluid();
        kosm::fluid::Ripples::from_wall(note.hz, 2, amp, note.decay, r, &wine)
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
    let tuned = goblet.tune(target);
    match &tuned {
        Some((level, path)) => {
            let level = *level;
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

    // the picture: filled to the note if there is one, else to the knob
    if !args.flag("no-frame") {
        let level = tuned.as_ref().map(|t| t.0).unwrap_or(fill);
        let shape = goblet.shape();
        let stage = render::Stage {
            sun_el_deg: get(&params, "sun_el"),
            sun_az_deg: get(&params, "sun_az"),
            width: get(&params, "width") as u32,
            height: get(&params, "height") as u32,
            spp: get(&params, "spp") as u32,
            photons: get(&params, "photons") as usize,
            exposure: get(&params, "exposure") as f32,
        };
        let t0 = std::time::Instant::now();
        let scene = render::scene(&shape, level, None, &stage);
        let map = render::caustic_map(&scene, &stage);
        let t_map = t0.elapsed().as_secs_f64();
        println!("frame: caustic map {:.1} s, {} photons deposited", t_map, map.len());
        rec.png("frame.png", &render::frame(&scene, &map, &shape, &stage))?;
        println!("frame: total {:.1} s", t0.elapsed().as_secs_f64());
        rec.metric("photons_deposited", map.len())?;

        // the ripples: the tap's waterline motion, strobed a little slower
        // than the note so the light moves slowly, as under a stroboscope
        let impulse = 1e-3 * get(&params, "drive");
        let rip = goblet.ripples(level, impulse);
        println!(
            "ripples: λ {:.2} mm, reach {:.1} mm, wall amplitude {:.2} µm, slope {:.4}",
            rip.wavelength() * 1e3,
            1e3 / rip.alpha,
            rip.amp * 1e6,
            rip.amp * rip.k
        );
        rec.metric("ripple_wavelength_mm", rip.wavelength() * 1e3)?;
        rec.metric("ripple_reach_mm", 1e3 / rip.alpha)?;
        rec.metric("ripple_amp_um", rip.amp * 1e6)?;
        rec.metric("ripple_slope", rip.amp * rip.k)?;
        let frames = get(&params, "ripple_frames") as usize;
        if frames > 0 {
            let dir = rec.path("ripple")?;
            std::fs::create_dir_all(&dir)?;
            let period = std::f64::consts::TAU / rip.omega;
            let t0 = std::time::Instant::now();
            for f in 0..frames {
                let t = f as f64 * period * (1.0 + 1.0 / 48.0);
                let top = render::rippled_surface(&shape, level, &rip, t);
                let scene = render::scene(&shape, level, Some(top), &stage);
                render::surface_closeup(&scene, &shape, level, &stage).save(dir.join(format!("frame_{f:03}.png")))?;
            }
            println!("ripple: {frames} frames in {:.1} s", t0.elapsed().as_secs_f64());
            let mp4 = rec.path("ripple.mp4")?;
            let st = std::process::Command::new("ffmpeg")
                .args(["-y", "-loglevel", "error", "-framerate", "24", "-i"])
                .arg(dir.join("frame_%03d.png"))
                .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17"])
                .arg(&mp4)
                .status();
            match st {
                Ok(s) if s.success() => println!("ripple: → {}", mp4.display()),
                _ => println!("ripple: no ffmpeg; frames are in {}", dir.display()),
            }
        }
    }
    rec.finish()?;
    Ok(())
}

fn write_wav(path: &std::path::Path, mono: &[f32]) -> anyhow::Result<()> {
    let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();
    kosm::room::write_wav_stereo(path, &stereo, SR)
}
