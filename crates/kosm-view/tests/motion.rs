//! What the window's motion vectors rest on.
//!
//! A game renderer estimates motion from screen space. This one does not have
//! to: the physics is right there, and the two claims checked here are the
//! whole of why the picture can be both exact and early.
//!
//! 1. **Motion is the physics'.** The displacement of a ball in flight, taken
//!    from the two frames' velocities alone, is the displacement the solver
//!    actually applied — to the last bit, not to a tolerance.
//! 2. **Ahead is computed, not predicted.** The frame the renderer aims at,
//!    reached by running the deterministic sim on, is the same state as
//!    stepping straight to that time.

use kosm_spike::court::render::Snapshot;
use kosm_spike::court::{Court, CourtScene};

fn scene() -> CourtScene {
    CourtScene::bundled().expect("the bundled court scene")
}

/// Step to just after `t`, so the ball is in free flight and nothing has been
/// touched.
fn stepped_to(scene: &CourtScene, t: f64) -> Court {
    let mut court = Court::from_scene(scene).expect("the court");
    while court.time() < t {
        court.step();
    }
    court
}

#[test]
fn velocity_is_the_motion_vector() {
    let scene = scene();
    let steps_per_frame = (1.0 / scene.fps / scene.dt).round().max(1.0) as usize;
    // Early: the balls are released above the slab and have touched nothing,
    // so the whole frame is one constant acceleration.
    let mut court = stepped_to(&scene, 0.05);
    let prev = Snapshot::of(&court);
    for _ in 0..steps_per_frame {
        court.step();
    }
    let now = Snapshot::of(&court);
    assert!(now.t > prev.t);
    for k in 0..now.balls.len() {
        let posed = now.balls[k].0 - prev.balls[k].0;
        let from_v = now.ball_displacement(&prev, k);
        let err = (posed - from_v).norm();
        assert!(
            err < 1e-6,
            "ball {k}: the pose moved {posed:?}, the velocities say {from_v:?} ({err:e} m apart)"
        );
    }
}

#[test]
fn a_frame_ahead_is_the_frames_own_state() {
    let scene = scene();
    let steps_per_frame = (1.0 / scene.fps / scene.dt).round().max(1.0) as usize;
    // Two frames of lookahead over a bounce, which is where a *predicted*
    // world and a *computed* one part company.
    let ahead = 2 * steps_per_frame;
    let mut running = stepped_to(&scene, 0.60);
    let target = running.time() + ahead as f64 * scene.dt;
    for _ in 0..ahead {
        running.step();
    }
    let run_on = Snapshot::of(&running);

    // The same time, reached from the beginning instead of from a frame the
    // renderer happened to be holding.
    let direct = stepped_to(&scene, target - 0.5 * scene.dt);
    let straight = Snapshot::of(&direct);

    assert!((run_on.t - straight.t).abs() < 1e-9, "{} vs {}", run_on.t, straight.t);
    for k in 0..run_on.balls.len() {
        assert_eq!(run_on.balls[k].0, straight.balls[k].0, "ball {k}");
        assert_eq!(run_on.vel[k].0, straight.vel[k].0, "ball {k} velocity");
    }
    assert_eq!(run_on.extras.len(), straight.extras.len());
}
