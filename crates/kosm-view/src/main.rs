//! Kosm view: the window, and nothing but the window.
//!
//! A winit window, a wgpu surface, and one image blitted across it. There are
//! no panels, no text and no widgets: anything on screen is the scene's own
//! picture. `viewport.rs` owns the window and the blit; `court.rs` owns the
//! court — the simulation on one thread, vcad's CPU path tracer on another —
//! and answers the viewport with the newest picture it has.
//!
//! ```text
//! kosm-view                     the court, in a window
//! kosm-view --shot out/x.png    one still, no window
//! ```
//!
//! It opens live: the simulation runs in wall-clock time and the window shows
//! the frame it is on. Controls are keyboard and mouse: drag to orbit, wheel
//! to zoom, space to pause and to rejoin the simulation, left and right to
//! step a frame while paused, Home or R to go back to the level's camera,
//! Escape to quit.

mod court;
mod viewport;

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
            .or_else(|| (a == &format!("--{key}")).then(|| args.get(i + 1).cloned()).flatten())
    })
}

fn parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    arg(key).and_then(|v| v.parse().ok())
}

fn main() -> anyhow::Result<()> {
    // Set before the simulation or render threads exist.
    unsafe { std::env::set_var("VCAD_LOON_NO_PARAM_RECOVERY", "1") };
    let _ = log::set_logger(&Stderr).map(|()| log::set_max_level(log::LevelFilter::Warn));

    // `--shot <path>` renders one frame with the same producer the window
    // uses and writes it, no window: the picture, testable.
    if let Some(path) = arg("shot") {
        let width: u32 = parse("width").unwrap_or(960);
        let size = (width, (width * 9 / 16).max(1));
        // the level's own `still_t` unless asked otherwise
        let t: f64 = parse("at").unwrap_or(-1.0);
        return court::still(std::path::Path::new(&path), t, size, parse("spp").unwrap_or(32));
    }
    court::run(parse("frames").unwrap_or(0), parse("spp").unwrap_or(4))
}
