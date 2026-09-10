//! The cove's first check: the beach is a plane, and the field is the beach.
//!
//! Nothing here knows about the being or the rune yet. What it establishes is
//! the ground under both: that a 100 mm bake of a tilted slab is the slab, and
//! that a body rolled on it by the SDF contact path obeys the mechanics of a
//! sphere on an incline rather than the mechanics of a staircase.

use std::path::PathBuf;

use super::{CoveScene, bake, being, sim};

/// Bake into a directory of our own so the test does not depend on, or
/// disturb, whatever `kosm run rune` last wrote into `out/`.
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

/// A cove with the **hero** at the spawn, holding the lens. What
/// `rune_tests.rs` reads the live gate off, sharing the one bake.
pub(super) fn hero_cove() -> anyhow::Result<(CoveScene, being::Cove)> {
    let (scene, baked) = field()?;
    let cove = being::Cove::with_player(&scene, baked.sdf.clone(), being::Player::Hero)?;
    Ok((scene, cove))
}

/// A cove with the **capsule** at the spawn, standing.
///
/// Said explicitly and not left to [`being::Player::from_env`], whose default
/// is the hero now. Everything below this line is a measurement of the
/// capsule — its mass, its spring, the speed `walk_mps` caps, how deep it
/// wades — and a figure has none of those numbers.
fn standing() -> anyhow::Result<(CoveScene, being::Cove)> {
    let (scene, baked) = field()?;
    let cove = being::Cove::with_player(&scene, baked.sdf.clone(), being::Player::Capsule)?;
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

// ---------------------------------------------------------------------------
// The edges of the cove. Everything below is about what happens when the player
// walks *away* from the puzzle: past the baked field there is no floor, so a
// level whose sand simply stops is a level you fall out of. The cove is closed
// instead — headlands on ±x, the cliff on +y, and along −y a seabed that runs
// out under the water to a reef — and these are the checks that it is.

/// The eight points of the compass, as a facing in radians.
fn compass(k: usize) -> f64 {
    std::f64::consts::TAU * k as f64 / 8.0
}

/// **Nowhere to fall.** From the spawn, hold W for three quarters of a minute
/// in each of eight directions. Whatever the being walks into — the cliff, a
/// headland, the sea, the reef — it stays on the ground the level put there:
/// its centre never drops more than 200 mm below where it would rest on the
/// sand under it, it never leaves the volume the field is baked over, and the
/// net under the level ([`sim::step_on_sdf_over`]) is never once asked to catch
/// it.
#[test]
fn there_is_nowhere_in_the_cove_to_fall_off() -> anyhow::Result<()> {
    let (scene, mut cove) = standing()?;
    let (lo, hi) = scene.volume();
    let r = scene.being_r;

    let (mut worst_drop, mut worst_drop_at) = (f64::NEG_INFINITY, 0usize);
    for k in 0..8 {
        cove.face(compass(k));
        cove.set_tilt(0.0);
        cove.place(scene.spawn_x, scene.spawn_y, 0.0);
        let walk = being::Input::walking(1.0);
        for _ in 0..45_000 {
            cove.step(&walk);
            let p = cove.being_centre();
            assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite(), "the being diverged walking {:.0}°", compass(k).to_degrees());
            // inside the field, by the radius the contact producer samples with
            assert!(
                p.x > lo.x + r && p.x < hi.x - r && p.y > lo.y + r && p.y < hi.y && p.z > lo.z,
                "walking {:.0}° took the being to ({:+.2}, {:+.2}, {:+.2}) m, outside the baked volume",
                compass(k).to_degrees(),
                p.x,
                p.y,
                p.z
            );
            let drop = cove.resting_centre(p.x, p.y).z - p.z;
            if drop > worst_drop {
                worst_drop = drop;
                worst_drop_at = k;
            }
        }
        assert_eq!(cove.net_caught(), 0, "the net caught the being walking {:.0}°", compass(k).to_degrees());
    }
    println!("eight directions, forty-five seconds each: the worst the being's centre got below the sand under it is {:.0} mm, walking {:.0}°", worst_drop * 1e3, compass(worst_drop_at).to_degrees());
    assert!(
        worst_drop < 0.2,
        "walking {:.0}° the being's centre got {:.0} mm below the sand under it",
        compass(worst_drop_at).to_degrees(),
        worst_drop * 1e3
    );
    Ok(())
}

/// **The sea stops you.** Walk straight out to sea and the water gets heavier
/// with every step: the shore break the being is pushing into rises with the
/// area it puts under, and somewhere between the waist and the chest it wins.
/// Nothing here is a wall — the seabed carries on for another few metres and
/// the reef is past that — and nothing here is a trigger.
#[test]
fn the_sea_stops_the_being_before_its_head_goes_under() -> anyhow::Result<()> {
    let (scene, mut cove) = standing()?;
    // Start on the dry sand a few metres up from the waterline, facing -y: the
    // spawn is twenty-odd metres away and this test is about the last five.
    cove.face(-std::f64::consts::FRAC_PI_2);
    cove.place(scene.spawn_x, scene.waterline() + 4.0, 0.0);

    let walk = being::Input::walking(1.0);
    // The fastest the being was seen going at each 50 mm of water, from the
    // moment it is up to speed: what "heavier the deeper you go" means as a
    // number the test can read.
    let mut by_depth: Vec<f64> = vec![0.0; 32];
    let (mut deepest, mut under) = (0.0f64, 0.0f64);
    for k in 0..30_000 {
        cove.step(&walk);
        let p = cove.being_centre();
        assert!(p.z > scene.sea_z - 0.2, "the being's centre sank to {:+.3} m, below the waterline's own guard", p.z);
        deepest = deepest.max(cove.wading_depth());
        under = under.max(cove.submerged());
        if k > 3_000 && cove.wading_depth() > 0.0 {
            let bin = (cove.wading_depth() / 0.05) as usize;
            if let Some(v) = by_depth.get_mut(bin) {
                *v = v.max(cove.walking_speed());
            }
        }
    }

    let depth = cove.wading_depth();
    let submerged = cove.submerged();
    println!(
        "the sea stopped the being in {:.2} m of water with {:.0} % of it under, its centre {:+.3} m off the waterline, at {:.4} m/s",
        depth,
        submerged * 100.0,
        cove.being_centre().z - scene.sea_z,
        cove.walking_speed()
    );
    assert!(cove.walking_speed() < 0.05, "thirty seconds of walking out to sea and the being is still doing {:.3} m/s", cove.walking_speed());
    assert!(under < 1.0, "the being went under: {:.0} % of it was submerged", under * 100.0);
    // waist to chest, and no deeper
    assert!(
        (0.4..=0.8).contains(&submerged),
        "the sea stopped the being with {:.0} % of it under, which is neither waist nor chest",
        submerged * 100.0
    );
    // and it got there by getting slower, not by hitting something: every 50 mm
    // of water it walked in was slower than the 50 mm before it
    let seen: Vec<(usize, f64)> = by_depth.iter().copied().enumerate().filter(|(_, v)| *v > 0.0).collect();
    assert!(seen.len() > 4, "the being never waded far enough to measure: {seen:?}");
    for w in seen.windows(2) {
        assert!(
            w[1].1 <= w[0].1 + 1e-3,
            "the being was doing {:.3} m/s in {:.2} m of water and {:.3} m/s in {:.2} m",
            w[0].1,
            w[0].0 as f64 * 0.05,
            w[1].1,
            w[1].0 as f64 * 0.05
        );
    }
    assert!(deepest > 0.3, "the being never really got into the sea: {deepest:.2} m");

    // …and the sea is not a hole. Turn round and the same shore break that
    // stopped you carries you back up the beach: there is no dying in this
    // game, so there is nowhere in it you can walk to and not walk out of.
    cove.face(std::f64::consts::FRAC_PI_2);
    cove.run(20.0, &walk);
    assert!(
        cove.wading_depth() < 0.0,
        "twenty seconds of walking back and the being is still in {:.2} m of water",
        cove.wading_depth()
    );
    Ok(())
}

/// **The hero stands too.** The same cove with `KOSM_RUNE_PLAYER=hero`: the
/// figure of `sims/rune/hero` on every hinge its `Rig` declares, on the same
/// baked field, held up by the same spring — and its boots, not a capsule, are
/// what touch the sand.
///
/// It is not the default, and [`being::Player::from_env`] says why: the cove's
/// rune is a caustic through a lens of glass and the figure is not one. What
/// this test is for is that the *body* works — that the migration to
/// [`kosm::player::Body`] gave the level a second body it can switch to on the
/// day the lens is the thing in the hero's hand, and that the day it does the
/// figure will already be standing.
#[test]
fn the_hero_stands_on_the_sand_and_walks_over_it() -> anyhow::Result<()> {
    let (scene, baked) = field()?;
    let mut cove = being::Cove::with_player(&scene, baked.sdf.clone(), being::Player::Hero)?;
    assert_eq!(cove.player(), being::Player::Hero);
    let start = cove.being_centre();

    cove.hold_still(5.0);
    let at = |c: &being::Cove| {
        let p = c.being_centre();
        (p.x - start.x).hypot(p.y - start.y)
    };
    let settled = at(&cove);
    cove.hold_still(5.0);
    let drift = at(&cove) - settled;
    println!("hero settle {:.0} mm in the first five seconds, {:.1} mm in the next five", settled * 1e3, drift * 1e3);
    println!(
        "the hero weighs {:.1} kg, stands {:.1} mm off the spawn after five seconds, {:.2}° off vertical, on {} parts",
        cove.mass(),
        drift * 1e3,
        cove.lean() / DEG,
        cove.hero_parts().len()
    );
    // The first five seconds are the figure *settling*: the beach is a six
    // per cent grade and the boots are 236 mm apart across it, so one lands
    // before the other and the legs find a stance. What the second five
    // seconds say is that the stance holds — a body whose joints were softer
    // than `player::body::JOINT_SUPPORT` crept downhill at 25 mm/s for ever,
    // and this is the number that caught it.
    assert!(settled < 0.10, "the hero took {:.0} mm to settle on the sand", settled * 1e3);
    assert!(drift < 0.005, "the hero crept {:.1} mm in the second five seconds of standing still", drift * 1e3);
    assert!(cove.lean() < 5.0 * DEG, "the hero stood {:.2}° off vertical", cove.lean() / DEG);
    assert!(cove.hero_parts().len() >= 13, "the hero has {} parts to draw", cove.hero_parts().len());
    assert_eq!(cove.net_caught(), 0, "the net caught the hero standing still");

    // and it walks, up the beach, at the speed the level asks for
    cove.face(std::f64::consts::FRAC_PI_2);
    cove.run(4.0, &being::Input::walking(1.0));
    println!("four seconds of walking and the hero is doing {:.3} m/s (the level says {:.2})", cove.walking_speed(), scene.walk_mps);
    assert!(
        (cove.walking_speed() - scene.walk_mps).abs() < 0.2 * scene.walk_mps,
        "the hero walks at {:.3} m/s, not the {:.2} the level asks for",
        cove.walking_speed(),
        scene.walk_mps
    );
    assert_eq!(cove.net_caught(), 0, "the net caught the hero walking");
    Ok(())
}

/// **The reef holds.** The −y edge of the cove is a line of boulders, and a
/// line of boulders is only an edge if there is no gap in it the player fits
/// through. Read along it at the height the being's foot sits at: a boulder
/// blocks the being wherever the field is within a radius of it, and no run of
/// clear water between two boulders is as wide as the being is.
#[test]
fn the_reef_has_no_gap_the_being_fits_through() -> anyhow::Result<()> {
    let (scene, baked) = field()?;
    let r = scene.being_r;
    // The band the reef stands in, and the height of the being's lower cap
    // centre standing on the seabed under it.
    let (y0, y1) = (scene.seabed_y(), scene.seabed_y() + 4.0);
    let mut worst_gap = 0.0f64;
    let mut gap = 0.0f64;
    let mut worst_at = 0.0f64;
    let step = 0.05;
    let mut x = -scene.headland_x;
    while x <= scene.headland_x {
        // the closest the ground comes to the being's foot anywhere across the
        // reef's band at this x
        let mut nearest = f64::INFINITY;
        let mut y = y0;
        while y <= y1 {
            let p = phyz_math::Vec3::new(x, y, scene.sand_z_at(x, y) + r);
            if let Some(d) = baked.sdf.sample(p) {
                nearest = nearest.min(d);
            }
            y += step;
        }
        if nearest > r {
            gap += step;
            if gap > worst_gap {
                worst_gap = gap;
                worst_at = x;
            }
        } else {
            gap = 0.0;
        }
        x += step;
    }
    println!("the reef's widest gap is {:.2} m, at x = {:+.1} m; the being is {:.2} m across", worst_gap, worst_at, 2.0 * r);
    assert!(worst_gap < 2.0 * r, "the reef has a {:.2} m gap at x = {:+.1} m, and the being is {:.2} m across", worst_gap, worst_at, 2.0 * r);
    Ok(())
}

