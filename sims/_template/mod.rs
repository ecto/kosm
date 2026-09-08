//! The smallest sim there is: copy this directory to start one.
//!
//! Four stages, in order, and the fourth is optional:
//!
//!   1. **build**   — a `World` out of a phyz rig and some named `Param`s.
//!   2. **run**     — a `rollout` through a `Step`, giving a `Trajectory`.
//!   3. **observe** — `Lens`es over that trajectory, written by a `Recorder`
//!                    under `out/<run id>/`.
//!   4. **optimise**— optional. Nothing here: the marble sim is the worked
//!                    example (adjoint gradients, tilt search, an n_d fit).
//!
//! The whole point is the headless loop: `kosm run _template --out out/`
//! writes a png and a `metrics.json` an agent can read, in a couple of
//! seconds, with a render budget of four samples per pixel.
//!
//! A directory under `sims/` with a `mod.rs` in it *is* a sim — `build.rs`
//! walks the tree, so there is no list to add yourself to.

use kosm::prelude::*;
use phyz_math::Vec3;

/// The knobs. A `Param` is a name and a value; it is part of the run hash,
/// so changing one changes where the outputs land.
fn params(args: &kosm_cli::Args) -> Vec<Param> {
    let height = args.value("height").and_then(|v| v.parse().ok()).unwrap_or(0.30);
    vec![Param::new("height", height), Param::new("steps", 400.0)]
}

/// **Stage 1, build.** A world is phyz's model and state plus kosm's own
/// columns. `demo_marble` is a bead over a fixed plate; a real sim builds its
/// geometry through `kosm::build` / `AuthoredScene` and its colliders through
/// `colliders_from_document`.
fn build(params: &[Param]) -> World {
    let (model, mut state) = kosm::world::demo_marble();
    let height = params.iter().find(|p| p.name == "height").map(|p| p.value).unwrap_or(0.3);
    state.q[5] = height; // free-joint q is [wx wy wz x y z]
    World::from_phyz(model, state).with_params(params.to_vec())
}

/// `kosm run _template --out DIR`.
pub fn run(args: &kosm_cli::Args) -> anyhow::Result<()> {
    let params = params(args);
    let world = build(&params);

    // Stage 2, run: a rollout is a trajectory, and a step is pure.
    let steps = params.iter().find(|p| p.name == "steps").map(|p| p.value as usize).unwrap_or(400);
    let traj = rollout(&world, &PhyzStep::new(1e-3), &Zero, steps);
    let last = traj.last().expect("a rollout keeps the world it started from");

    // Stage 3, observe. A probe is a lens on a column; a reward is a closure
    // as a lens; a camera is a lens through kosm-render. None of them holds
    // the world.
    let height = Probe::q("bead height", 5);
    let dropped = reward("dropped", |w: &World| world.q()[5] - w.q()[5]);
    let camera = Camera::look_at(
        Vec3::new(0.25, -0.35, 0.30),
        Vec3::new(0.0, 0.0, 0.10),
        320,
        240,
        4, // a tiny budget: this must finish in seconds
    );

    // The recorder puts everything under `<out>/<run id>/` and prints the id.
    let mut rec = Recorder::new(args.out(), "_template", &params, 0)?;
    rec.png("frame.png", &camera.see(last).image)?;
    rec.metric("height_start_m", height.see(&world))?;
    rec.metric("height_end_m", height.see(last))?;
    rec.metric("dropped_m", dropped.see(last))?;
    rec.metric("steps", steps)?;
    // worlds diff: how far the rollout moved every column
    rec.metric("max_column_change", diff(&world, last).max_abs())?;
    rec.finish()?;
    Ok(())
}
