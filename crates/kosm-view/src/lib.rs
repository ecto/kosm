//! The viewer, as a library a sim can drive.
//!
//! A winit window, a wgpu surface, and one image blitted across it: that is
//! [`viewport`], and a sim that wants a window implements [`viewport::Scene`]
//! and hands it to [`Viewer::run`]. [`temporal`] is the temporal history the
//! tiers accumulate into, behind one trait; [`history`] is the CPU tier's
//! implementation of it — reprojection, a geometric mask, an à-trous filter,
//! a firefly cap — for a sim with no GPU tracer to hand a scene to;
//! [`budget`] chooses how big a picture the next pass may be, from what the
//! last one cost and whether anything moved; [`frame`] the camera and the
//! readback both the window and a still share, and [`ride`] the
//! recorded-rollout player.
//!
//! Nothing here names a sim. `sims/court/game.rs` and `sims/pool/game.rs` are
//! the per-sim viewer modes and live with their sims.

pub mod budget;
pub mod frame;
pub mod history;
pub mod temporal;
pub mod viewport;

#[cfg(feature = "ride")]
pub mod ride;

pub use budget::Budget;
pub use frame::{Camera, read_back};
pub use temporal::{Pose, TemporalHistory, View};
pub use viewport::{Event, Image, Key, Scene};

/// The window, driven by a sim's [`Scene`].
pub struct Viewer;

impl Viewer {
    /// Open a window of `size` titled `title` and let `scene` answer it.
    pub fn run(title: &str, size: (u32, u32), scene: impl Scene) -> anyhow::Result<()> {
        viewport::run(title, size, scene)
    }
}

/// Warnings from wgpu and vcad, on stderr; anything quieter is noise.
pub struct Stderr;

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

/// Warnings on stderr, and vcad's parameter recovery off. Idempotent; call it
/// before any simulation or render thread exists.
pub fn init() {
    unsafe { std::env::set_var("VCAD_LOON_NO_PARAM_RECOVERY", "1") };
    let _ = log::set_logger(&Stderr).map(|()| log::set_max_level(log::LevelFilter::Warn));
}

/// `--key=value` or `--key value`, whichever the caller wrote.
pub fn arg(key: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().enumerate().find_map(|(i, a)| {
        a.strip_prefix(&format!("--{key}="))
            .map(str::to_owned)
            .or_else(|| (a == &format!("--{key}")).then(|| args.get(i + 1).cloned()).flatten())
    })
}

pub fn parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    arg(key).and_then(|v| v.parse().ok())
}
