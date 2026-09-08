//! The skatepark, baked and rolled on: the field is the arc, and a rolling
//! sphere on that field obeys 10/7 · g · Δ.

use kosm::skatepark::{self, SkateparkScene};

#[test]
fn the_field_is_the_arc_and_the_wheel_rolls_as_predicted() {
    let scene = SkateparkScene::bundled().expect("level loads");
    let dir = std::env::temp_dir().join("kosm-skatepark-test");
    let baked = skatepark::bake(&scene, &dir).expect("bakes");
    let worst = skatepark::arc_error(&scene, &baked.sdf, 64).expect("arc inside the volume");
    println!("arc error {:.2} mm over {} tris", worst * 1e3, baked.tris);
    assert!(worst < scene.cell * 0.5, "field is {worst} m off the ideal arc");

    let r = skatepark::roll(&scene, &baked.sdf).expect("rolls");
    let rel = r.flat_speed / r.predicted_speed - 1.0;
    println!("flat {:.3} m/s vs {:.3} m/s ({:+.1} %); far apex {:?} vs {:.3}; drift {:.1} mm", r.flat_speed, r.predicted_speed, rel * 100.0, r.far_apex, r.release_height, r.drift * 1e3);
    assert!(rel.abs() < 0.03, "flat speed off by {:.1} %", rel * 100.0);
    let apex = r.far_apex.expect("reached the far wall");
    assert!(apex > 0.85 * r.release_height, "climbed only to {apex} of {}", r.release_height);
    assert!(r.drift < 0.01, "drifted {} m sideways", r.drift);
}
