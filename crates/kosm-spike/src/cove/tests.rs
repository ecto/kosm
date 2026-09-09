//! The cove's first check: the beach is a plane, and the field is the beach.
//!
//! Nothing here knows about the being or the rune yet. What it establishes is
//! the ground under both: that a 100 mm bake of a tilted slab is the slab, and
//! that a body rolled on it by the SDF contact path obeys the mechanics of a
//! sphere on an incline rather than the mechanics of a staircase.

use std::path::PathBuf;

use super::{CoveScene, bake, being, sim};

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

// ---------------------------------------------------------------------------
// The being and the door. Everything below stands a body on the field the
// marble rolled down: it is the same bake, the same contact step and the same
// glass, with a spring where the being's legs would be.

/// The cove, baked once for every check that needs a field.
///
/// 16 M cells is a few seconds; baking it once per test would be that again
/// for an answer that cannot have changed. The directory is only the road the
/// field travels in on — the grid is in memory the moment [`bake::bake`]
/// returns — so it is removed straight away and nothing is left in `/tmp`.
fn field() -> anyhow::Result<(CoveScene, &'static crate::skatepark::Baked)> {
    static FIELD: std::sync::OnceLock<crate::skatepark::Baked> = std::sync::OnceLock::new();
    // `OnceLock::get_or_init` cannot fail, and a bake can, so the one bake is
    // kept behind a lock of its own: the second test through waits for the
    // first rather than baking into — and deleting — the same directory.
    static ONCE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let scene = CoveScene::bundled()?;
    if let Some(baked) = FIELD.get() {
        return Ok((scene, baked));
    }
    let held = ONCE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(baked) = FIELD.get() {
        return Ok((scene, baked));
    }
    let dir: PathBuf = std::env::temp_dir().join(format!("kosm-cove-being-{}", std::process::id()));
    let baked = bake::bake(&scene, &dir)?;
    let _ = std::fs::remove_dir_all(&dir);
    let baked = FIELD.get_or_init(|| baked);
    drop(held);
    Ok((scene, baked))
}

/// A cove with the being at the spawn, standing.
fn standing() -> anyhow::Result<(CoveScene, being::Cove)> {
    let (scene, baked) = field()?;
    let cove = being::Cove::new(&scene, baked.sdf.clone())?;
    Ok((scene, cove))
}

const DEG: f64 = std::f64::consts::PI / 180.0;

#[test]
fn the_being_stands_still_on_the_slope() -> anyhow::Result<()> {
    let (scene, mut cove) = standing()?;
    let start = cove.being_centre();
    let rest = cove.resting_centre(scene.spawn_x, scene.spawn_y);

    let (mut drift, mut lean, mut sink) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..10_000 {
        cove.step(&being::Input::STILL);
        let p = cove.being_centre();
        drift = drift.max((p.x - start.x).hypot(p.y - start.y));
        lean = lean.max(cove.lean());
        sink = sink.max((rest.z - p.z).abs());
    }

    // Ten seconds of nobody at the controls. The being is held up by a spring,
    // so what this says is that the spring's equilibrium *is* standing: it
    // does not creep down the 6 % grade, it does not lie over, and it does not
    // settle into the sand.
    assert!(drift < 0.01, "the being wandered {:.1} mm in ten seconds of standing still", drift * 1e3);
    assert!(lean < DEG, "the being leaned {:.2}° off vertical while standing", lean / DEG);
    assert!(sink < 0.02, "the being's centre is {:.0} mm off the height it rests at", sink * 1e3);
    Ok(())
}

#[test]
fn the_being_comes_back_up_from_a_shove() -> anyhow::Result<()> {
    let (scene, mut cove) = standing()?;
    cove.place(scene.spawn_x, scene.spawn_y, 20.0 * DEG);
    let shove = cove.lean();
    assert!((shove - 20.0 * DEG).abs() < 1e-6, "the shove put the being at {:.1}°, not 20°", shove / DEG);

    // Two seconds to stand back up.
    let mut worst_after = 0.0f64;
    for k in 0..5_000 {
        cove.step(&being::Input::STILL);
        if k >= 2_000 {
            worst_after = worst_after.max(cove.lean());
        }
    }
    assert!(cove.lean() < DEG, "two seconds after a 20° shove the being is still {:.2}° over", cove.lean() / DEG);
    // Critically damped means it arrives and stays: no overshoot to ring out.
    assert!(worst_after < 2.0 * DEG, "the being oscillated to {:.2}° after recovering", worst_after / DEG);
    Ok(())
}

#[test]
fn the_being_walks_at_the_walking_speed() -> anyhow::Result<()> {
    let (scene, mut cove) = standing()?;
    let walk = being::Input::walking(1.0);
    let mut fastest = 0.0f64;
    for _ in 0..5_000 {
        cove.step(&walk);
        fastest = fastest.max(cove.walking_speed());
    }
    let cruise = cove.walking_speed();
    assert!(
        (cruise - scene.walk_mps).abs() < 0.1 * scene.walk_mps,
        "five seconds of walking got the being to {:.3} m/s, not the {:.2} m/s it is capped at",
        cruise,
        scene.walk_mps
    );
    assert!(fastest < 1.05 * scene.walk_mps, "the being reached {:.3} m/s, past the {:.2} m/s cap", fastest, scene.walk_mps);

    // Let go and it stops: the sand takes what the damping does not.
    cove.hold_still(1.0);
    assert!(cove.walking_speed() < 0.05, "a second after letting go the being is still doing {:.3} m/s", cove.walking_speed());
    Ok(())
}

#[test]
fn the_door_swings_when_the_gate_opens() -> anyhow::Result<()> {
    let (_, mut cove) = standing()?;

    // Closed, it is a wall: nothing drives a hinge whose axis is vertical.
    cove.hold_still(1.0);
    assert!(cove.door_angle().abs() < 1e-9, "the door drifted to {:.3}° with the gate shut", cove.door_angle() / DEG);

    cove.set_gate(true);
    let (mut previous, mut worst_backslide) = (0.0f64, 0.0f64);
    for _ in 0..3_000 {
        cove.step(&being::Input::STILL);
        let a = cove.door_angle();
        worst_backslide = worst_backslide.max(previous - a);
        previous = a;
    }
    assert!(worst_backslide < 1e-9, "the door swung back by {:.4}° on its way open", worst_backslide / DEG);
    assert!(
        (being::DOOR_LIMIT - previous).abs() < DEG,
        "three seconds open and the door is at {:.1}°, not the {:.0}° the recess stops it at",
        previous / DEG,
        being::DOOR_LIMIT / DEG
    );
    Ok(())
}
