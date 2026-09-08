//! The rulebook's inflation test, in the simulator: a ball dropped from 1.8 m
//! with restitution e comes back to e² of that, and keeps doing so.
use kosm::court::{Court, CourtScene};

#[test]
fn a_dropped_ball_bounces_to_e_squared() {
    let mut scene = CourtScene::bundled().unwrap();
    scene.n_balls = 1;
    let mut court = Court::from_scene(&scene).unwrap();
    while court.time() < 3.5 {
        court.step();
    }
    let apexes = court.apexes_of(0);
    println!("apexes {:?}", apexes);
    assert!(apexes.len() >= 3, "expected at least three bounces, got {}", apexes.len());
    let e2 = scene.restitution * scene.restitution;
    let first = apexes[0] / scene.drop;
    assert!((first / e2 - 1.0).abs() < 0.05, "first apex ratio {first:.4}, expected e² = {e2:.4}");
    for w in apexes.windows(2).take(3) {
        let ratio = w[1] / w[0];
        assert!((ratio / e2 - 1.0).abs() < 0.10, "apex ratio {ratio:.4}, expected e² = {e2:.4}");
    }
}
