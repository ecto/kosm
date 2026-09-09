//! The cove's first check: the beach is a plane, and the field is the beach.
//!
//! Nothing here knows about the being or the rune yet. What it establishes is
//! the ground under both: that a 100 mm bake of a tilted slab is the slab, and
//! that a body rolled on it by the SDF contact path obeys the mechanics of a
//! sphere on an incline rather than the mechanics of a staircase.

use std::path::PathBuf;

use super::{CoveScene, bake, sim};

/// Bake into a directory of our own so the test does not depend on, or
/// disturb, whatever `--cove` last wrote into `out/`.
fn baked() -> anyhow::Result<(CoveScene, crate::skatepark::Baked)> {
    let scene = CoveScene::bundled()?;
    let dir: PathBuf = std::env::temp_dir().join(format!("kosm-cove-{}", std::process::id()));
    let baked = bake::bake(&scene, &dir)?;
    Ok((scene, baked))
}

#[test]
fn the_marble_rolls_down_the_cove_at_the_speed_a_sphere_rolls() -> anyhow::Result<()> {
    let (scene, baked) = baked()?;

    // The field is the sand: on a plane, trilinear interpolation of an exact
    // signed distance is exact, so this is well under half a cell.
    let worst = bake::plane_error(&scene, &baked.sdf, scene.spawn_x, 20)?;
    assert!(worst < scene.cell / 2.0, "the field is {:.1} mm off the beach plane, more than half a {:.0} mm cell", worst * 1e3, scene.cell * 1e3);

    // A glass sphere released at rest on the slope, rolled to a stop short of
    // the waterline: rolling without slip, v² = 10/7 · g · Δ.
    let r = sim::roll_on_beach(&scene, &baked.sdf, bake::MARBLE_R, scene.spawn_x, scene.spawn_y, bake::ROLL_T, bake::ROLL_GUARD)?;
    assert!(r.drop > 0.2, "the marble only fell {:.3} m; it never got rolling", r.drop);
    assert!(r.end.y > scene.waterline(), "the marble rolled into the sea, so the check ran past the beach");
    assert!(
        r.error().abs() < 0.05,
        "the marble reached {:.3} m/s after falling {:.3} m, against {:.3} m/s for a rolling sphere ({:+.1} %)",
        r.speed,
        r.drop,
        r.predicted,
        r.error() * 100.0
    );
    // A normal that leans shows up across the fall line before it shows up in
    // the speed, so this is the tighter half of the check.
    assert!(r.drift < 0.01, "the marble drifted {:.1} mm across the fall line", r.drift * 1e3);

    let _ = std::fs::remove_dir_all(&baked.dir);
    Ok(())
}
