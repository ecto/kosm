//! The glass, live: `kosm run glass --view` (build with `--features view`).
//!
//! The same scene the stills use — lathed goblet, wine, linen, a low sun, the
//! photon-mapped caustic — path traced on the CPU on a thread of its own and
//! refined while nothing moves. Every knob is the physics's:
//!
//! - **drag** orbits, **wheel** dollies (get close to the rim for the ripples);
//! - **W / S** pours and drinks, 2 mm a press: the note is re-solved and
//!   printed, and the caustic is re-traced;
//! - **Space** taps the rim: `glass.wav`'s sound through the speakers, and the
//!   ripples it drives on the wine;
//! - **Shift** (held) rubs the rim: the glass harmonica, looping for as long
//!   as it is held, and ripples that stay while you rub;
//! - **Left / Right** swing the sun round, **A / D** lower and raise it;
//! - **Home** pours to the tuned note (A4) and resets the view.
//!
//! The ripples' phase is shown 200× slowed: at 440 Hz they would alias to
//! noise at any display rate. Their amplitude and decay are real time.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kosm::shell;
use kosm_render::pathtrace::{Camera, PathTraceOptions, render_with_caustics};
use kosm_render::{Point3, Vec3};
use kosm_view::{Event, Image, Key, Scene, Viewer};

use super::{Goblet, SR, render};

/// Ripple phase is shown this many times slower than it runs.
const SLOW: f64 = 200.0;

#[derive(Clone, Copy, PartialEq)]
struct Look {
    yaw: f64,
    pitch: f64,
    dist: f64,
}

#[derive(Clone)]
struct State {
    look: Look,
    level: f64,
    sun_az: f64,
    sun_el: f64,
    /// When the last tap landed, and how hard.
    tap: Option<(Instant, f64)>,
    /// The waterline amplitude a rub holds, while it is held.
    rub: Option<f64>,
    size: (u32, u32),
}

struct Shared {
    state: Mutex<State>,
    image: Mutex<Option<Image>>,
    quit: AtomicBool,
}

pub fn run(goblet: Goblet, level: f64, sun_el: f64, sun_az: f64) -> anyhow::Result<()> {
    let shared = Arc::new(Shared {
        state: Mutex::new(State {
            look: Look { yaw: (sun_az + 110.0).to_radians(), pitch: 0.42, dist: 0.5 },
            level,
            sun_az,
            sun_el,
            tap: None,
            rub: None,
            size: (1280, 800),
        }),
        image: Mutex::new(None),
        quit: AtomicBool::new(false),
    });
    let goblet = Arc::new(goblet);
    let tracer = {
        let (shared, goblet) = (shared.clone(), goblet.clone());
        std::thread::spawn(move || trace(&shared, &goblet))
    };
    let audio = rodio::OutputStream::try_default().ok();
    if audio.is_none() {
        eprintln!("glass: no audio output; the window is silent");
    }
    println!("glass: drag orbit · wheel dolly · W/S pour/drink · Space tap · Shift rub · ←/→ A/D sun · Home reset");
    announce(&goblet, level);
    let tuned = goblet.tune(440.0).map(|t| t.0).unwrap_or(level);
    let window = Window { shared: shared.clone(), goblet, audio, rubbing: None, drag: None, tuned };
    let result = Viewer::run("kosm · glass", (1280, 800), window);
    shared.quit.store(true, Ordering::Relaxed);
    let _ = tracer.join();
    result
}

fn announce(goblet: &Goblet, level: f64) {
    let note = goblet.note(level);
    println!("glass: {:4.1} mm of {} → {:6.1} Hz ({})", level * 1e3, goblet.wine_name, note.hz, pitch_name(note.hz));
}

/// The nearest equal-tempered note and how far off it is, in cents.
fn pitch_name(hz: f64) -> String {
    const NAMES: [&str; 12] = ["A", "A♯", "B", "C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯"];
    let semis = 12.0 * (hz / 440.0).log2();
    let n = semis.round();
    let octave = 4 + ((n as i64 + 9).div_euclid(12));
    format!("{}{} {:+.0}¢", NAMES[(n as i64).rem_euclid(12) as usize], octave, (semis - n) * 100.0)
}

struct Window {
    shared: Arc<Shared>,
    goblet: Arc<Goblet>,
    audio: Option<(rodio::OutputStream, rodio::OutputStreamHandle)>,
    rubbing: Option<rodio::Sink>,
    drag: Option<()>,
    tuned: f64,
}

impl Window {
    fn play(&self, mono: Vec<f32>) -> Option<rodio::Sink> {
        let (_, handle) = self.audio.as_ref()?;
        let sink = rodio::Sink::try_new(handle).ok()?;
        sink.append(rodio::buffer::SamplesBuffer::new(1, SR as u32, mono));
        Some(sink)
    }

    /// The rub's attack once, then its steady tail round and round until the
    /// sink is stopped. The loop runs from one upward zero crossing to the one
    /// nearest a second later, so the seam lands on the same phase of the
    /// note and does not click.
    fn play_looped(&self, mono: Vec<f32>) -> Option<rodio::Sink> {
        use rodio::Source;
        let (_, handle) = self.audio.as_ref()?;
        let sink = rodio::Sink::try_new(handle).ok()?;
        let up = |from: usize| (from..mono.len() - 1).find(|&i| mono[i] <= 0.0 && mono[i + 1] > 0.0);
        let start = up((1.5 * SR) as usize)?;
        let end = up(start + SR as usize)?;
        let attack = mono[..start].to_vec();
        let tail = mono[start..end].to_vec();
        sink.append(rodio::buffer::SamplesBuffer::new(1, SR as u32, attack));
        sink.append(rodio::buffer::SamplesBuffer::new(1, SR as u32, tail).repeat_infinite());
        Some(sink)
    }

    fn with<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        f(&mut self.shared.state.lock().expect("the state lock"))
    }
}

impl Scene for Window {
    fn event(&mut self, event: Event) {
        let _ = self.drag;
        match event {
            Event::Resized(size) => self.with(|s| s.size = size),
            Event::Drag(dx, dy) => self.with(|s| {
                s.look.yaw -= dx * 0.006;
                s.look.pitch = (s.look.pitch + dy * 0.004).clamp(0.05, 1.45);
            }),
            Event::Zoom(z) => self.with(|s| s.look.dist = (s.look.dist * (-z * 0.08).exp()).clamp(0.06, 1.5)),
            Event::Key(Key::W) | Event::Key(Key::S) => {
                let up = matches!(event, Event::Key(Key::W));
                let top = self.goblet.height * 0.97;
                let level = self.with(|s| {
                    s.level = (s.level + if up { 2e-3 } else { -2e-3 }).clamp(0.0, top);
                    s.level
                });
                announce(&self.goblet, level);
            }
            Event::Key(Key::Space) => {
                let level = self.with(|s| s.level);
                let sound = self.goblet.tap(level, 0.15e-3, 3.0);
                if let Some(sink) = self.play(sound) {
                    sink.detach();
                }
                let amp = self.goblet.ripples(level, 1e-3).amp;
                self.with(|s| s.tap = Some((Instant::now(), amp)));
            }
            Event::Key(Key::Shift) => {
                let level = self.with(|s| s.level);
                let g = &self.goblet;
                let bank = g.bank(level);
                let rim = g.bowl.nodes() - 2;
                let waterline = g.bowl.points.iter().position(|q| q[1] >= level).unwrap_or(rim);
                let area = g.bowl.area();
                let rubbed = shell::rub(&bank, rim, g.rim_r, waterline, shell::Rub::default(), 3.0, SR, |hz| {
                    kosm::audio::radiation_efficiency(hz, area, g.wall, g.glass)
                });
                let steady = *rubbed.amplitude.last().unwrap_or(&0.0);
                self.rubbing = self.play_looped(rubbed.audio);
                self.with(|s| s.rub = Some(steady));
            }
            Event::KeyUp(Key::Shift) => {
                if let Some(sink) = self.rubbing.take() {
                    sink.stop();
                }
                self.with(|s| s.rub = None);
            }
            Event::Key(Key::Left) => self.with(|s| s.sun_az -= 10.0),
            Event::Key(Key::Right) => self.with(|s| s.sun_az += 10.0),
            Event::Key(Key::A) => self.with(|s| s.sun_el = (s.sun_el - 4.0).max(6.0)),
            Event::Key(Key::D) => self.with(|s| s.sun_el = (s.sun_el + 4.0).min(80.0)),
            Event::Key(Key::Home) => {
                let tuned = self.tuned;
                self.with(|s| {
                    s.level = tuned;
                    s.look = Look { yaw: (s.sun_az + 110.0).to_radians(), pitch: 0.42, dist: 0.5 };
                });
                announce(&self.goblet, tuned);
            }
            _ => {}
        }
    }

    fn image(&mut self) -> Option<Image> {
        self.shared.image.lock().expect("the image lock").take()
    }
}

/// The tracer's loop: rebuild what changed, draw a pass, fold it in.
fn trace(shared: &Shared, goblet: &Goblet) {
    let shape = goblet.shape();
    // what the caustic map was traced for: fill, sun, and ripples at a phase
    let mut built: Option<(f64, f64, f64, Option<f64>)> = None;
    let mut scene = None;
    let mut map = None;
    let mut accum: Vec<f32> = vec![];
    let mut passes = 0u32;
    let mut last: Option<(Look, (u32, u32))> = None;

    while !shared.quit.load(Ordering::Relaxed) {
        let s = shared.state.lock().expect("the state lock").clone();

        // the ripples on the wine right now, if any: amplitude in real time,
        // phase slowed so the eye can follow it
        let now = Instant::now();
        let ripple = match (s.rub, s.tap) {
            (Some(a), _) => Some((a, 0.0, now)),
            (None, Some((at, a))) => {
                let rip = goblet.ripples(s.level, 1e-3);
                let t = now.duration_since(at).as_secs_f64();
                let amp = a * (-rip.decay * t).exp();
                (amp > 0.05e-6).then_some((a, t, at))
            }
            _ => None,
        };
        let phase = ripple.map(|(_, _, at)| now.duration_since(at).as_secs_f64() / SLOW);
        let key = (s.level, s.sun_az, s.sun_el, phase);
        if built != Some(key) || scene.is_none() {
            let stage = stage(&s, 0);
            let top = ripple.map(|(amp, t_real, at)| {
                let mut rip = goblet.ripples(s.level, 1e-3);
                rip.amp = amp;
                if s.rub.is_some() {
                    rip.decay = 0.0;
                }
                let _ = t_real;
                render::rippled_surface(&shape, s.level, &rip, now.duration_since(at).as_secs_f64() / SLOW)
            });
            let sc = render::scene(&shape, s.level, top, &stage);
            // the caustic is re-traced when the glass or the sun changes; a
            // ripple only moves it by less than the gather radius
            if built.is_none_or(|b| (b.0, b.1, b.2) != (s.level, s.sun_az, s.sun_el)) {
                map = Some(render::caustic_map(&sc, &render::Stage { photons: 600_000, ..stage }));
            }
            scene = Some(sc);
            built = Some(key);
            passes = 0;
        }
        if last != Some((s.look, s.size)) {
            passes = 0;
            last = Some((s.look, s.size));
        }

        // still: full-ish resolution, accumulating; moving: a quick small pass
        let moving = passes == 0 || phase.is_some();
        let div = if moving { 3 } else { 2 };
        let (w, h) = ((s.size.0 / div).max(64), (s.size.1 / div).max(40));
        let cam = camera(&s, &shape);
        let opts = PathTraceOptions { spp: if moving { 2 } else { 4 }, max_depth: 16, seed: passes as u64, ..PathTraceOptions::default() };
        let (Some(sc), Some(m)) = (&scene, &map) else { continue };
        let film = render_with_caustics(sc, &cam, w, h, &opts, Some(m));
        let n = (w * h * 3) as usize;
        if passes == 0 || accum.len() != n {
            accum = film.rgb.clone();
            passes = 1;
        } else {
            passes += 1;
            let k = 1.0 / passes as f32;
            for (a, x) in accum.iter_mut().zip(&film.rgb) {
                *a += (x - *a) * k;
            }
        }
        let mut shown = film;
        shown.rgb.copy_from_slice(&accum);
        let rgba = shown.to_srgb8(0.8, false);
        *shared.image.lock().expect("the image lock") = Some(Image::Bytes { size: (w, h), rgba });
        if passes > 64 {
            std::thread::sleep(Duration::from_millis(30)); // converged: idle until something moves
        }
    }
}

fn stage(s: &State, spp: u32) -> render::Stage {
    render::Stage { sun_el_deg: s.sun_el, sun_az_deg: s.sun_az, width: s.size.0, height: s.size.1, spp, photons: 600_000, exposure: 0.8 }
}

/// An orbit round the bowl, looking a little below its middle.
fn camera(s: &State, shape: &render::Shape) -> Camera {
    let bowl_top = shape.inner.last().map(|q| q[1]).unwrap_or(0.09);
    let target = Point3::new(0.0, 0.0, 0.45 * bowl_top);
    let (cy, sy, cp, sp) = (s.look.yaw.cos(), s.look.yaw.sin(), s.look.pitch.cos(), s.look.pitch.sin());
    let eye = target + Vec3::new(cy * cp, sy * cp, sp) * s.look.dist;
    Camera::look_at(eye, target, Vec3::new(0.0, 0.0, 1.0), 40.0)
}
