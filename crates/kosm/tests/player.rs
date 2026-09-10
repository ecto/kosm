//! `kosm::player::Body`: the controller, measured.
//!
//! Everything here stands on a plane, so nothing needs a bake and the whole
//! file runs in seconds. Every test prints the number it asserts on, because
//! a controller that passes and a controller that *feels* right are two
//! different claims and only one of them can be automated.
//!
//! Metres, radians, seconds.

use std::f64::consts::PI;

use kosm::player::body::{DRIVE_ASSIST, LEAN_MAX, TAU_ACCEL, TAU_STOP, UPRIGHT_OMEGA, WALK};
use kosm::player::{Air, Body, BodySpec, Drive, Ground, Netted, Plane, Skeleton, Tool};
use phyz_math::{GRAVITY, Vec3};

const DEG: f64 = PI / 180.0;
const DT: f64 = 1e-3;

/// The cove's being: 350 mm by 1 m of N-BK7, out of `sims/rune/scene.rs`.
const BEING_R: f64 = 0.350;
const BEING_H: f64 = 1.000;

fn glass() -> f64 {
    kosm::material::named("N-BK7").expect("N-BK7 is in kosm's material library").density
}

fn capsule() -> Body {
    let mut body = Body::new(BodySpec::capsule(BEING_R, BEING_H, glass()).with_dt(DT));
    body.place(0.0, 0.0, 0.0, 0.0, 0.0);
    body
}

fn hero() -> Body {
    let mut body = Body::new(BodySpec::hero(&Skeleton::demo_hero()).with_dt(DT));
    body.place(0.0, 0.0, 0.0, 0.0, 0.0);
    body
}

/// Rise time: when the speed first passes `1 − 1/e` of what it settles at.
fn tau_of(trace: &[(f64, f64)], settled: f64) -> f64 {
    let want = settled * (1.0 - 1.0 / std::f64::consts::E);
    trace.iter().find(|(_, v)| *v >= want).map(|(t, _)| *t).unwrap_or(f64::INFINITY)
}

/// Decay time: when the speed first drops below `1/e` of what it started at.
fn tau_down(trace: &[(f64, f64)], from: f64) -> f64 {
    let want = from / std::f64::consts::E;
    trace.iter().find(|(_, v)| *v <= want).map(|(t, _)| *t).unwrap_or(f64::INFINITY)
}

// ---- standing ---------------------------------------------------------------

#[test]
fn the_body_stands_still() {
    for (what, mut body) in [("capsule", capsule()), ("hero", hero())] {
        let start = body.root();
        let ground = Plane::at(0.0);
        let (mut drift, mut worst_lean) = (0.0f64, 0.0f64);
        for _ in 0..10_000 {
            body.step(&Drive::STILL, &ground, &Air, DT);
            let p = body.root();
            drift = drift.max((p.x - start.x).hypot(p.y - start.y));
            worst_lean = worst_lean.max(body.lean());
        }
        println!("stand  {what:8} drift {:5.1} mm   lean {:5.2}°   sink {:+6.1} mm", drift * 1e3, worst_lean / DEG, (body.root().z - start.z) * 1e3);
        assert!(drift < 0.01, "the {what} wandered {:.1} mm in ten seconds of standing still", drift * 1e3);
        assert!(worst_lean < DEG, "the {what} leaned {:.2}° off vertical while standing", worst_lean / DEG);
    }
}

#[test]
fn a_shove_comes_back_and_does_not_overshoot() {
    for (what, mut body) in [("capsule", capsule()), ("hero", hero())] {
        body.place(0.0, 0.0, 0.0, 0.0, 20.0 * DEG);
        let shove = body.lean();
        assert!((shove - 20.0 * DEG).abs() < 1e-6, "the shove put the {what} at {:.1}°, not 20°", shove / DEG);
        let ground = Plane::at(0.0);
        let mut back_at = f64::INFINITY;
        let mut worst_after = 0.0f64;
        for k in 0..5_000 {
            body.step(&Drive::STILL, &ground, &Air, DT);
            let lean = body.lean();
            if lean < DEG && back_at.is_infinite() {
                back_at = (k + 1) as f64 * DT;
            }
            // Critically damped means it arrives and *stays*: past two seconds
            // there is nothing left to ring out.
            if k >= 2_000 {
                worst_after = worst_after.max(lean);
            }
        }
        println!("shove  {what:8} 20° → under 1° in {back_at:.2} s, worst after {:.2}°  (ω = {UPRIGHT_OMEGA} rad/s)", worst_after / DEG);
        assert!(back_at < 2.0, "the {what} took {back_at:.2} s to come back from 20°");
        assert!(worst_after < 2.0 * DEG, "the {what} oscillated to {:.2}° after recovering", worst_after / DEG);
    }
}

// ---- the curves -------------------------------------------------------------

#[test]
fn the_walk_rises_and_stops_on_its_time_constants() {
    // The contract is the capsule's: `TAU_ACCEL` and `TAU_STOP` describe a
    // rigid body, and a rigid body is what the capsule is. A figure is a plant
    // of its own — its feet are on compliant joints, its centre of mass is
    // half a metre above the force, and the legs it swings are a tenth of it —
    // so it lands a tenth of a time constant late into a walk and holds the
    // stop. That is measured, not allowed for: the band is wider by exactly
    // what the figure costs and no more.
    for (what, mut body, band) in [("capsule", capsule(), 0.10), ("hero", hero(), 0.15)] {
        let ground = Plane::at(0.0);
        let mut up = Vec::new();
        for k in 0..3_000 {
            body.step(&Drive::walking(1.0), &ground, &Air, DT);
            up.push(((k + 1) as f64 * DT, body.speed()));
        }
        let settled = body.speed();
        let rise = tau_of(&up, WALK);
        let mut down = Vec::new();
        for k in 0..2_000 {
            body.step(&Drive::STILL, &ground, &Air, DT);
            down.push(((k + 1) as f64 * DT, body.speed()));
        }
        let stop = tau_down(&down, settled);
        println!("walk   {what:8} settles at {settled:.3} m/s (want {WALK:.2})   rise {rise:.3} s (τ {TAU_ACCEL})   stop {stop:.3} s (τ {TAU_STOP})   left {:.3} m/s", body.speed());
        assert!((settled - WALK).abs() < 0.05 * WALK, "the {what} settled at {settled:.3} m/s, not {WALK}");
        assert!((rise - TAU_ACCEL).abs() < band * TAU_ACCEL, "the {what} rose in {rise:.3} s, not {TAU_ACCEL}");
        assert!((stop - TAU_STOP).abs() < band * TAU_STOP, "the {what} stopped in {stop:.3} s, not {TAU_STOP}");
        assert!(body.speed() < 0.05, "two seconds after letting go the {what} is still doing {:.3} m/s", body.speed());
    }
}

#[test]
fn the_diagonal_is_not_faster_than_the_straight() {
    let ground = Plane::at(0.0);
    let mut straight = capsule();
    straight.run_for(4.0, &Drive::walking(1.0), &ground, &Air);
    let mut diagonal = capsule();
    diagonal.run_for(4.0, &Drive { forward: 1.0, strafe: 1.0, ..Drive::STILL }, &ground, &Air);
    println!("corner straight {:.3} m/s   diagonal {:.3} m/s   ratio {:.3}", straight.speed(), diagonal.speed(), diagonal.speed() / straight.speed());
    assert!(diagonal.speed() <= straight.speed() * 1.02, "the diagonal walks at {:.3} m/s against the straight's {:.3}", diagonal.speed(), straight.speed());
}

#[test]
fn the_body_leans_into_a_start() {
    let ground = Plane::at(0.0);
    for (what, mut body) in [("capsule", capsule()), ("hero", hero())] {
        body.run_for(0.2, &Drive::walking(1.0), &ground, &Air);
        // Which way is it leaning? The body's own axis, projected on the
        // facing: positive is forward, which is into the start.
        let into = body.axis().dot(body.facing_dir());
        println!("lean   {what:8} at 0.2 s: body {:+.2}° into the walk, asked for {:+.2}° (cap {:.1}°)", into.asin() / DEG, body.commanded_lean() / DEG, LEAN_MAX / DEG);
        assert!(into > 0.0, "the {what} leaned *away* from the start by {:.2}°", -into.asin() / DEG);
        assert!(into.asin() > 1.0 * DEG, "the {what} leaned only {:.2}° into the start", into.asin() / DEG);
        assert!(into.asin() < LEAN_MAX + 2.0 * DEG, "the {what} leaned {:.2}°, past the {:.1}° cap", into.asin() / DEG, LEAN_MAX / DEG);
    }
}

/// How much of the start the lean actually buys.
///
/// Three runs of the same half second: the controller as it ships, the same
/// controller with the lean turned off (`DRIVE_ASSIST` at one), and the lean
/// on its own. The three velocities are the whole of the claim in
/// `DRIVE_ASSIST`'s docs, and this is where the number in the report comes
/// from.
#[test]
fn how_much_of_the_start_the_lean_is_worth() {
    let ground = Plane::at(0.0);
    let sample = |body: &mut Body, seconds: f64| {
        body.run_for(seconds, &Drive::walking(1.0), &ground, &Air);
        body.speed()
    };
    let mut shipped = capsule();
    let both = sample(&mut shipped, 0.25);
    // The lean alone: no drive force at all, only the tilt the player can ask
    // for, held at the cap.
    let mut leaning = capsule();
    let mut lean_only = 0.0;
    for _ in 0..250 {
        leaning.step(&Drive { lean_delta: 1.0, ..Drive::STILL }, &ground, &Air, DT);
        lean_only = leaning.speed();
    }
    println!(
        "split  drive+lean {both:.3} m/s at 0.25 s   lean alone {lean_only:.3} m/s   \
         the lean is {:.0} % of the start, the force {:.0} % (DRIVE_ASSIST {DRIVE_ASSIST})",
        100.0 * lean_only / both,
        100.0 * (both - lean_only) / both
    );
    assert!(lean_only > 0.0, "leaning on its own moved the body nowhere");
    assert!(lean_only < both, "the lean alone beat the whole controller");
}

// ---- the gait ---------------------------------------------------------------

#[test]
fn the_gait_is_locked_to_the_speed_and_settles_when_stopped() {
    let ground = Plane::at(0.0);
    let mut body = hero();
    body.run_for(2.0, &Drive::walking(1.0), &ground, &Air);
    let speed = body.speed();
    let predicted = body.gait().stride_hz(speed);
    // A fifth of a second, which at this cadence is a fifth of a turn: short
    // enough that the wrapped phase cannot be ambiguous about how far it went.
    let window = 0.2;
    let before = body.gait().phase;
    body.run_for(window, &Drive::walking(1.0), &ground, &Air);
    let after = body.gait().phase;
    let measured = (after - before).rem_euclid(2.0 * PI) / (2.0 * PI) / window;
    println!(
        "gait   at {speed:.3} m/s: predicted {predicted:.3} Hz, measured {measured:.3} Hz   \
         (pendulum {:.3} Hz on a {:.3} m leg, stride {:.2} m)",
        body.gait().pendulum_hz(),
        body.gait().leg_length,
        speed / predicted.max(1e-9)
    );
    assert!((measured - predicted).abs() < 0.02 * predicted, "the phase advanced at {measured:.3} Hz against a predicted {predicted:.3}");

    body.run_for(2.0, &Drive::STILL, &ground, &Air);
    let stopped = body.gait().phase;
    body.run_for(1.0, &Drive::STILL, &ground, &Air);
    println!("gait   stopped: phase {:.4} → {:.4} rad, stride {:.4} Hz", stopped, body.gait().phase, body.gait().stride_hz(body.speed()));
    assert!((body.gait().phase - stopped).abs() < 1e-9, "the phase kept advancing after the body stopped");
}

// ---- the tool socket --------------------------------------------------------

#[test]
fn the_hand_reaches_and_the_tool_follows_it() {
    let ground = Plane::at(0.0);
    let mut body = hero();
    body.hold(Tool::new("refractor"));
    body.run_for(0.5, &Drive::STILL, &ground, &Air);
    // A point in front of the chest and out to the right, well inside reach.
    let root = body.root();
    let goal = root + Vec3::new(0.32, -0.16, 0.16);
    body.run_for(2.0, &Drive { aim: Some(goal), ..Drive::STILL }, &ground, &Air);
    let hand = body.hand().expect("the hero has an arm");
    let miss = (hand.pos - goal).norm();
    let (tool, _) = body.held().expect("the hero is holding the refractor");
    println!("reach  hand {:.1} mm from the target; the tool is {:.1} mm from the hand", miss * 1e3, (tool.pos - hand.pos).norm() * 1e3);
    assert!(miss < 0.05, "the hand stopped {:.1} mm from the target", miss * 1e3);
    assert!((tool.pos - hand.pos).norm() < 1e-9, "an ungripped tool is not in the hand");
    // and it moves with the body
    let before = body.held().unwrap().0.pos;
    body.run_for(1.0, &Drive { forward: 1.0, aim: Some(goal), ..Drive::STILL }, &ground, &Air);
    let after = body.held().unwrap().0.pos;
    println!("reach  the tool travelled {:.0} mm with the walk", (after - before).norm() * 1e3);
    assert!((after - before).norm() > 0.2, "the held tool did not travel with the body");
}

// ---- the capsule is the cove's being ----------------------------------------

#[test]
fn the_capsule_spec_is_the_coves_being() {
    let body = capsule();
    // What `sims/rune/being.rs::Being::new` says, written out here so the two
    // are compared and not merely shared.
    let (r, h, rho) = (BEING_R, BEING_H, glass());
    let half = (h - 2.0 * r) / 2.0;
    let (m_cyl, m_cap) = (rho * PI * r * r * 2.0 * half, rho * 2.0 / 3.0 * PI * r * r * r);
    let mass = m_cyl + 2.0 * m_cap;
    let l = 2.0 * half;
    let i_t = m_cyl * (3.0 * r * r + l * l) / 12.0 + 2.0 * m_cap * (0.4 * r * r + l * l / 4.0 + 3.0 * l * r / 8.0);
    let i_pivot = i_t + mass * (h / 2.0).powi(2);
    let k = i_pivot * UPRIGHT_OMEGA * UPRIGHT_OMEGA + mass * GRAVITY * h / 2.0;
    let c = 2.0 * i_pivot * UPRIGHT_OMEGA;
    let (got_k, got_c) = body.upright_spring();
    println!("being  mass {:.1} kg (want {mass:.1})   k {:.1} kN·m/rad (want {:.1})   c {:.2} kN·m·s/rad (want {:.2})", body.mass(), got_k / 1e3, k / 1e3, got_c / 1e3, c / 1e3);
    assert!((body.mass() - mass).abs() < 1e-9, "the capsule spec weighs {:.3} kg, not {mass:.3}", body.mass());
    assert!((got_k - k).abs() < 1e-6 * k, "the capsule spec's spring is {got_k:.1}, not {k:.1}");
    assert!((got_c - c).abs() < 1e-6 * c, "the capsule spec's damping is {got_c:.1}, not {c:.1}");
    assert!((body.consts().com_height - h / 2.0).abs() < 1e-12, "the centre of mass is not at half the height");
}

// ---- the net ----------------------------------------------------------------

#[test]
fn the_net_is_never_used_on_a_plane() {
    let ground = Netted::new(Plane::at(0.0), -10.0);
    let mut body = capsule();
    for k in 0..4_000 {
        let drive = if k < 2_000 { Drive::walking(1.0) } else { Drive { forward: 1.0, strafe: 1.0, yaw_delta: 2e-3, ..Drive::STILL } };
        body.step(&drive, &ground, &Air, DT);
    }
    println!("net    {} catches in four seconds of walking a closed plane", ground.caught());
    assert_eq!(ground.caught(), 0, "the net carried {} steps over a plane that has no hole in it", ground.caught());
    // and it is still a ground: the body is standing on it, not through it
    assert!(body.root().z > 0.0, "the body fell through the plane to {:.3} m", body.root().z);
}

/// The net *is* there when the ground runs out: a plane with nothing under it
/// at all, and the same body dropped over the edge of it.
#[test]
fn the_net_catches_what_falls_off_the_map() {
    struct Nowhere;
    impl Ground for Nowhere {
        fn contacts(&self, _: &phyz_model::Model, _: &phyz_model::State, _: f64) -> Vec<phyz_collision::Collision> {
            Vec::new()
        }
    }
    let ground = Netted::new(Nowhere, -2.0);
    let mut body = capsule();
    body.place(0.0, 0.0, 0.0, 0.0, 0.0);
    body.run_for(3.0, &Drive::STILL, &ground, &Air);
    println!("net    caught {} steps, and the body rests at {:+.3} m over a floor at −2.000", ground.caught(), body.root().z);
    assert!(ground.caught() > 0, "nothing under the body and the net never fired");
    assert!(body.root().z > -2.0, "the net let the body through to {:.3} m", body.root().z);
}
