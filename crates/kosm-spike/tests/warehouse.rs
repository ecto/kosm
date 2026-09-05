//! The warehouse, baked and ridden on: three pieces of it, checked against
//! the physics they were drawn for rather than against a golden mesh.
//!
//! One bake, three questions. A quarter pipe is a quarter pipe if a sphere
//! released on it reaches the flat at `10/7 · g · Δ`. A kicker is a kicker if
//! a wheel rolled at it leaves the ground above its lip. A rail is Ø 50 mm if
//! the field a wheel would feel reads its radius on its axis.

use kosm_spike::skatepark::{self, SkateparkScene};
use phyz_math::Vec3;

#[test]
fn the_quarter_pipe_the_kicker_and_the_rail_are_what_the_level_says() {
    let scene = SkateparkScene::load(kosm_spike::scene::AuthoredScene::bundled_path("warehouse.loon")).expect("level loads");
    let dir = std::env::temp_dir().join("kosm-warehouse-test");
    let baked = skatepark::bake(&scene, &dir).expect("bakes");
    let sdf = &baked.sdf;
    println!("{} roots, {} collision tris, {} MB of field", baked.parts, baked.tris, (sdf.data.len() * 4) / 1_000_000);

    // The half pipe the level names in `check_x`: the field is its arc, and
    // the wheel obeys the same 10/7 as the mini ramp's.
    let worst = skatepark::arc_error(&scene, sdf, 64).expect("arc inside the volume");
    println!("half pipe arc error {:.2} mm at {:.0} mm cells", worst * 1e3, scene.cell * 1e3);
    assert!(worst < scene.cell * 0.5, "field is {worst} m off the ideal arc");

    // The +x quarter pipe, beside the door. It has no far wall and no flat of
    // its own, so the check moves to it: `check_x` is where its transition
    // meets the floor and `flat` is the run of clear floor in front of it.
    let mut qp = SkateparkScene::load(kosm_spike::scene::AuthoredScene::bundled_path("warehouse.loon")).expect("level loads");
    qp.check_x = qp.authored.millimetres("qp_x_mm").expect("qp_x_mm");
    qp.check_y = -(qp.authored.millimetres("door_w_mm").expect("door_w_mm") / 2.0
        + qp.authored.millimetres("qp_gap_mm").expect("qp_gap_mm")
        + qp.authored.millimetres("qp_w_mm").expect("qp_w_mm") / 2.0);
    qp.check_z = 0.0;
    qp.flat = 3.0;
    let r = skatepark::roll(&qp, sdf).expect("rolls");
    let rel = r.flat_speed / r.predicted_speed - 1.0;
    println!("quarter pipe: {:.3} m/s vs {:.3} m/s ({:+.1} %); drift {:.1} mm", r.flat_speed, r.predicted_speed, rel * 100.0, r.drift * 1e3);
    assert!(rel.abs() < 0.03, "quarter pipe flat speed off by {:.1} %", rel * 100.0);

    // The kicker: a wheel rolled at 4 m/s up the −x wedge has to leave it,
    // and be higher than the lip it left once past it.
    let lip_x = -scene.authored.millimetres("kick_gap_mm").expect("kick_gap_mm") / 2.0;
    let rise = scene.authored.millimetres("kick_rise_mm").expect("kick_rise_mm");
    let start = Vec3::new(lip_x - 1.25, 0.0, scene.wheel_r);
    let launch = skatepark::roll_from(&scene, sdf, start, Vec3::new(4.0, 0.0, 0.0), 1.0).expect("rolls");
    let apex = launch.path.iter().filter(|p| p.x > lip_x).map(|p| p.z).fold(f64::MIN, f64::max);
    println!("kicker: lip at x = {lip_x:.2} m, {:.0} mm tall; apex {:.3} m", rise * 1e3, apex);
    assert!(apex > rise + scene.wheel_r, "the wheel only reached {apex} m over a {rise} m lip");

    // The rail: the field's surface sits one radius above its axis. Not the
    // value *on* the axis, which a Ø 50 mm rod on 25 mm cells cannot carry —
    // no grid node lands within 21 mm of the axis, so trilinear interpolation
    // never sees the −25 mm the analytic field has there. What survives
    // coarse sampling is where the zero is, and that is what a wheel feels.
    // Sampled 400 mm along the +x leg, clear of the posts and of the bend.
    let p = scene.authored.parameter("rail_bend_deg").expect("rail_bend_deg").to_radians();
    let rail_r = scene.authored.millimetres("rail_r_mm").expect("rail_r_mm");
    let d = 0.4;
    let axis = Vec3::new(
        scene.authored.millimetres("rail_x_mm").expect("rail_x_mm") + d * p.cos(),
        scene.authored.millimetres("rail_y_mm").expect("rail_y_mm") + d * p.sin(),
        scene.authored.millimetres("rail_z_mm").expect("rail_z_mm"),
    );
    let top = sdf.sample(axis + Vec3::new(0.0, 0.0, rail_r)).expect("the rail is inside the baked volume");
    println!("rail top: the field reads {:+.1} mm one {:.0} mm radius above the axis", top * 1e3, rail_r * 1e3);
    assert!(top.abs() < scene.cell, "the surface is {top} m from where a {rail_r} m rail puts it");
}
