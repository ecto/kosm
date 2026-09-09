//! The shot's gradient, and the solve that uses it.
//!
//! Two things are worth a gate. First, that the adjoint is the derivative: on
//! a free-flight horizon — before the ball touches the rim, where the rollout
//! is smooth and there is no active set to switch — one backward pass must
//! agree with central differences on both channels, `dJ/d(release point)` and
//! `dJ/d(release velocity)`. Second, that the solve does its job: a short free
//! throw, a miss in the production simulator, comes back a make.
use crate::court::{Court, CourtScene, Shot, aim};

#[test]
fn the_adjoint_is_the_derivative_in_free_flight() {
    let scene = CourtScene::bundled().unwrap();
    let horizon = aim::steps(&scene).unwrap();
    let free = aim::free_flight_steps(&scene, horizon).unwrap();
    println!("horizon {horizon} steps, free flight for {free}");
    assert!(free > 100, "expected a long free flight before the first touch, got {free} steps");

    let check = aim::check(&scene, free).unwrap();
    for line in check.lines() {
        println!("{line}");
    }
    let (lane, gap) = check.worst();
    assert!(gap < 1e-3, "adjoint and central differences differ by {gap:.2e} in d/d {lane}");
}

#[test]
fn a_short_shot_is_solved_back_into_the_hoop() {
    let mut scene = CourtScene::bundled().unwrap();
    let base = scene.shot.expect("the court level has a shot");
    scene.shot = Some(Shot { speed: 6.8, ..base });

    // it misses to start with, or there is nothing to solve
    let mut court = Court::from_scene(&scene).unwrap();
    while court.time() < scene.t_end && court.made_at.is_none() {
        court.step();
    }
    assert!(court.made_at.is_none(), "6.8 m/s was supposed to be short, but it went in");

    let steps = aim::steps(&scene).unwrap();
    let mut log = Vec::new();
    let solved = aim::solve(&mut scene, steps, &mut log).unwrap();
    for line in &log {
        println!("{line}");
    }
    println!("{solved:?}");
    assert!(
        solved.made_at.is_some(),
        "the solve left the shot missing: {:.3} m/s at {:.2}°, miss {:.1} mm",
        solved.speed,
        solved.elevation.to_degrees(),
        solved.miss * 1e3
    );
    assert!(solved.miss < 0.15, "the miss at the horizon is still {:.1} mm", solved.miss * 1e3);
}
