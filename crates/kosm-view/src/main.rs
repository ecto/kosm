//! Kosm view: the window, and nothing but the window.
//!
//! A winit window, a wgpu surface, and one image blitted across it. There are
//! no panels, no text and no widgets: anything on screen is the scene's own
//! picture. What the window shows is a `Trajectory`, and the binary knows two
//! sources for one:
//!
//! ```text
//! kosm-view --ride out/ride.json   a recorded rollout, played back
//! kosm-view --live                 one streamed from a child simulator
//! ```
//!
//! A sim's own viewer mode is the sim's: `kosm run court --view` and
//! `kosm run pool --view` drive [`kosm_view::Viewer`] from `sims/*/game.rs`.
//! Nothing here names a sim.

/// The ipse tree the live recorder lives in — it resolves its assets against
/// its own cwd, so the child is run from there.
#[cfg(feature = "ride")]
const IPSE: &str = "/Users/cam/Developer/ipse";

fn main() -> anyhow::Result<()> {
    kosm_view::init();

    // `--ride <ride.json>` plays a recorded rollout, `--live` streams one from
    // a child simulator.
    #[cfg(feature = "ride")]
    {
        use kosm_view::ride;
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
                Some(c) => (
                    c.split_whitespace().map(str::to_string).collect(),
                    std::env::current_dir().unwrap_or_else(|_| ".".into()),
                ),
                None => (
                    vec![format!("{IPSE}/target/release/examples/k1_skatepark_ride"), "--stream".into()],
                    std::path::PathBuf::from(IPSE),
                ),
            };
            let num = |flag: &str, d: f64| after(flag).and_then(|v| v.parse().ok()).unwrap_or(d);
            let mut extra = Vec::new();
            for flag in ["--policy", "--scenario"] {
                if let Some(v) = after(flag) {
                    extra.push(flag.to_string());
                    extra.push(v);
                }
            }
            Some(ride::Source::Live(ride::LiveOpts {
                cmd,
                cwd,
                shove: num("--shove", 8.0),
                shove_at: num("--shove-at", 0.5),
                duration: num("--duration", 6.0),
                extra,
            }))
        } else {
            None
        };
        if let Some(source) = source {
            return ride::run(source);
        }
    }

    eprintln!("kosm-view plays a trajectory: --ride <ride.json>, or --live.");
    eprintln!("a sim's own viewer is the sim's: `kosm run <sim> --view`.");
    Ok(())
}
