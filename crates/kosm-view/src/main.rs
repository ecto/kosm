//! Kosm view: the window, and nothing but the window.
//!
//! A winit window, a wgpu surface, and one image blitted across it. There are
//! no panels, no text and no widgets: anything on screen is the scene's own
//! picture. `viewport.rs` owns the window and the blit; `court.rs` owns the
//! court — the simulation on one thread, a path tracer on another (vcad's
//! compute shader on the window's own device, or its CPU integrator with
//! `--cpu`) — and answers the viewport with the newest picture it has.
//!
//! ```text
//! kosm-view                     the court, in a window
//! kosm-view --cpu               the CPU integrator, not the GPU tracer
//! kosm-view --shot out/x.png    one still, no window
//! kosm-view --orbit-test        what a camera move costs the GPU history
//! kosm-view --dump-frames 6     six live frames, with the history, to out/view_seq
//! kosm-view --denoise neural    the court's own trained filter, not the à-trous one
//! ```
//!
//! `--denoise` chooses the filter the device runs over the history.
//! `atrous` is the default and is the hand-tuned wavelet in `history.wgsl`;
//! `neural` is a small kernel-predicting network fitted to 1024-spp reference
//! renders of this court, embedded in the binary. `neural=weights.bin` runs a
//! different fit, which is how `kosm-spike`'s `denoise_dataset` example's
//! output gets looked at without a rebuild.
//!
//! It opens live: the simulation runs in wall-clock time and the window shows
//! the frame it is on. Controls are keyboard and mouse: drag to orbit, wheel
//! to zoom, space to pause and to rejoin the simulation, left and right to
//! step a frame while paused, Home or R to go back to the level's camera,
//! Escape to quit.

mod court;
mod court_gpu;
mod history;
mod viewport;
// The skatepark's ride viewer from main is an egui app; it rides along behind
// the `ride` feature so the default binary stays the bare viewport.
#[cfg(feature = "ride")]
mod live;
#[cfg(feature = "ride")]
mod ride;

/// The ipse tree the live recorder lives in — it resolves its assets against
/// its own cwd, so the child is run from there.
#[cfg(feature = "ride")]
const IPSE: &str = "/Users/cam/Developer/ipse";

/// Warnings from wgpu and vcad, on stderr; anything quieter is noise.
struct Stderr;

impl log::Log for Stderr {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Warn
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            eprintln!("{}: {}", r.level(), r.args());
        }
    }
    fn flush(&self) {}
}

/// `--key=value` or `--key value`, whichever the caller wrote.
fn arg(key: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().enumerate().find_map(|(i, a)| {
        a.strip_prefix(&format!("--{key}="))
            .map(str::to_owned)
            .or_else(|| {
                (a == &format!("--{key}"))
                    .then(|| args.get(i + 1).cloned())
                    .flatten()
            })
    })
}

fn parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    arg(key).and_then(|v| v.parse().ok())
}

fn main() -> anyhow::Result<()> {
    // Set before the simulation or render threads exist.
    unsafe { std::env::set_var("VCAD_LOON_NO_PARAM_RECOVERY", "1") };
    let _ = log::set_logger(&Stderr).map(|()| log::set_max_level(log::LevelFilter::Warn));

    // `--ride <ride.json>` plays a recorded rollout, `--live` streams one from
    // a child simulator: the skatepark's viewer, when built with `--features ride`.
    #[cfg(feature = "ride")]
    {
        let args: Vec<String> = std::env::args().collect();
        let after = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
        let source = if let Some(i) = args.iter().position(|a| a == "--ride") {
            let Some(path) = args.get(i + 1) else {
                eprintln!("--ride wants a path to a ride.json");
                std::process::exit(2);
            };
            Some(ride::Source::File(path.into()))
        } else if args.iter().any(|a| a == "--live") {
            let (cmd, cwd) = match after("--live-cmd") {
                Some(c) => (c.split_whitespace().map(str::to_string).collect(), std::env::current_dir().unwrap_or_else(|_| ".".into())),
                None => (vec![format!("{IPSE}/target/release/examples/k1_skatepark_ride"), "--stream".into()], std::path::PathBuf::from(IPSE)),
            };
            let num = |flag: &str, d: f64| after(flag).and_then(|v| v.parse().ok()).unwrap_or(d);
            let mut extra = Vec::new();
            for flag in ["--policy", "--scenario"] {
                if let Some(v) = after(flag) {
                    extra.push(flag.to_string());
                    extra.push(v);
                }
            }
            Some(ride::Source::Live(ride::LiveOpts { cmd, cwd, shove: num("--shove", 8.0), shove_at: num("--shove-at", 0.5), duration: num("--duration", 6.0), extra }))
        } else {
            None
        };
        if let Some(source) = source {
            return ride::run(source);
        }
    }

    // `--shot <path>` renders one frame with the same producer the window
    // uses and writes it, no window: the picture, testable.
    if let Some(path) = arg("shot") {
        let width: u32 = parse("width").unwrap_or(960);
        let size = (width, (width * 9 / 16).max(1));
        // the level's own `still_t` unless asked otherwise
        let t: f64 = parse("at").unwrap_or(-1.0);
        // The GPU tracer unless asked otherwise: `--cpu` takes the CPU
        // integrator, which is the reference the GPU picture is checked
        // against.
        let path = std::path::Path::new(&path);
        let spp = parse("spp").unwrap_or(32);
        if std::env::args().any(|a| a == "--cpu") {
            return court::still(path, t, size, spp);
        }
        return match court::still_gpu(path, t, size, spp) {
            Ok(()) => Ok(()),
            Err(error) => {
                eprintln!("court  gpu: {error}; falling back to the CPU tracer");
                court::still(path, t, size, spp)
            }
        };
    }
    // `--dump-frames N --at T` writes N consecutive live-tier frames, with the
    // history running through them, so the temporal accumulation can be
    // looked at rather than argued about.
    if let Some(n) = parse::<u32>("dump-frames") {
        let width: u32 = parse("width").unwrap_or(480);
        let size = (width, (width * 9 / 16).max(1));
        let t: f64 = parse("at").unwrap_or(-1.0);
        let dir = arg("out").unwrap_or_else(|| "out/view_seq".into());
        return court::dump_frames(std::path::Path::new(&dir), t, size, n);
    }
    // `--orbit-test` is the reprojection, scripted: converge headlessly, swing
    // the camera a few degrees, take one more pass, and say how much of the
    // frame kept its history across the move.
    if std::env::args().any(|a| a == "--orbit-test" || a.starts_with("--orbit-test=")) {
        let width: u32 = parse("width").unwrap_or(480);
        let size = (width, (width * 9 / 16).max(1));
        let t: f64 = parse("at").unwrap_or(-1.0);
        let deg: f64 = arg("orbit-test")
            .and_then(|v| v.parse().ok())
            .unwrap_or(3.0);
        return court::orbit_test(t, size, parse("spp").unwrap_or(8), deg);
    }
    // `--cpu` pins the CPU integrator; without it the window uses the GPU
    // tracer when the adapter and the court allow it.
    let cpu_only = std::env::args().any(|a| a == "--cpu");
    court::run(
        parse("frames").unwrap_or(0),
        parse("spp").unwrap_or(4),
        cpu_only,
    )
}
