//! `kosm::player::Rig`, measured.
//!
//! Every number the design asked for is checked here against a scripted walk
//! rather than asserted in prose: the eye's lag, the aim's lead and how fast
//! it comes back, the field of view's gain, the arm's clearance, and — the
//! one that is not about feel — that a settled camera is the *same* camera,
//! to the millimetre, twice.

use kosm::player::meter::Meter;
use kosm::player::rig::{Clearance, Interest, Rig, RigKnobs, Subject};
use phyz_math::Vec3;

/// 120 Hz, which is the rate the springs are measured at. They are exact
/// solutions, so the numbers below do not move when this does — the last
/// test in this file says so.
const DT: f64 = 1.0 / 120.0;

/// A walk: `walk_s` at `speed` along +y, then `stop_s` standing still.
/// Yields `(t, subject)` per frame, from the first frame after the snap.
fn walk(speed: f64, walk_s: f64, stop_s: f64) -> Vec<(f64, Subject)> {
    let n = ((walk_s + stop_s) / DT).round() as usize;
    let mut out = Vec::with_capacity(n);
    let mut y = 0.0;
    for i in 0..=n {
        let t = i as f64 * DT;
        let moving = t < walk_s;
        let v = if moving { Vec3::new(0.0, speed, 0.0) } else { Vec3::zeros() };
        out.push((
            t,
            Subject {
                position: Vec3::new(0.0, y, 1.0),
                velocity: v,
                facing: Vec3::new(0.0, 1.0, 0.0),
                speed: if moving { speed } else { 0.0 },
                lean: 0.0,
            },
        ));
        if moving {
            y += speed * DT;
        }
    }
    out
}

/// A critically damped tracker trails a constant-velocity target by `2/ω` of
/// travel, and `follow_lag_s` *is* `2/ω`. So the eye's distance behind the
/// point it wants, divided by the speed, is the lag in seconds — and it
/// should be a seventh of a second, not a frame more or less.
#[test]
fn the_eye_lags_by_the_time_it_says_it_does() {
    let rig = Rig::default();
    let script = walk(1.4, 3.0, 2.0);
    rig.follow(&script[0].1, 0.0);

    let mut steady = Vec::new();
    let mut settle = None;
    for (t, s) in &script[1..] {
        rig.follow(s, DT);
        let err = (rig.eye() - rig.wanted_eye()).norm();
        // the last second of the walk, by which the spring has long settled
        if (2.0..3.0).contains(t) {
            steady.push(err / s.speed.max(1e-9));
        }
        if *t > 3.0 && settle.is_none() && err < 1e-3 {
            settle = Some(t - 3.0);
        }
    }
    let lag = steady.iter().sum::<f64>() / steady.len() as f64;
    let spread = steady.iter().fold(0.0f64, |m, v| m.max((v - lag).abs()));
    println!("lag {lag:.6} s (spread {spread:.2e}), settled to 1 mm in {:.3} s", settle.unwrap());
    assert!((lag - 0.15).abs() < 0.15 * 0.02, "the eye lagged by {lag:.4} s, wanted 0.15");
    assert!(spread < 1e-6, "the lag was not steady: {spread:.2e}");
    let settle = settle.expect("the eye never settled after the walk stopped");
    assert!(settle < 0.8, "it took {settle:.3} s to settle within a millimetre");
}

/// The aim slides `lookahead_s × velocity` along the ground, which at a walk
/// is about eight and a half tenths of a metre — and lets go of it when the
/// walking stops.
#[test]
fn the_aim_leads_the_walk_and_lets_go_of_it() {
    let rig = Rig::default();
    let want = 0.6 * 1.4;
    let script = walk(1.4, 3.0, 2.0);
    rig.follow(&script[0].1, 0.0);

    let mut steady = 0.0;
    let mut back = None;
    for (t, s) in &script[1..] {
        rig.follow(s, DT);
        if (2.0..3.0).contains(t) {
            steady = rig.lead();
            // the aim is ahead of the body, not behind it
            let ahead = (rig.aim(s) - s.position).dot(&Vec3::new(0.0, 1.0, 0.0));
            assert!(ahead > 0.0, "the aim fell behind the walk at t = {t:.2}");
        }
        if *t > 3.0 && back.is_none() && rig.lead() < 0.05 * want {
            back = Some(t - 3.0);
        }
    }
    println!("lead {steady:.4} m (wanted {want:.4}), back to 5% in {:.3} s", back.unwrap());
    assert!((steady - want).abs() < 0.01 * want, "the aim led by {steady:.4} m");
    let back = back.expect("the aim never came back");
    assert!(back < 0.5, "the aim took {back:.3} s to come back");
}

/// `fov = base + gain × speed`, smoothed. Standing still it is the base
/// exactly — which the quantisation guarantees, and which is why a still
/// frame's view compares equal to itself.
#[test]
fn the_field_of_view_widens_with_the_walk() {
    let rig = Rig::default();
    let k = RigKnobs::default();
    let script = walk(1.4, 3.0, 2.0);
    rig.follow(&script[0].1, 0.0);

    let mut walking = 0.0;
    for (t, s) in &script[1..] {
        let cam = rig.follow(s, DT);
        if (2.9..3.0).contains(t) {
            walking = cam.fov_deg;
        }
    }
    let standing = rig.follow(&script.last().unwrap().1, DT).fov_deg;
    let want = k.fov_base_deg + k.fov_gain_deg * 1.4;
    println!("fov walking {walking:.3}°, standing {standing:.3}°, wanted {want:.3}°");
    assert!((walking - want).abs() < 0.02, "walking at {walking}°, wanted {want}°");
    assert!((standing - k.fov_base_deg).abs() < 1e-12, "standing at {standing}°");
}

/// The arm and the floor, over a sweep of poses: a wall at `y = 0` with the
/// subject walking at it from every angle, and a waterline under everything.
/// Not one pose in the sweep may put the eye inside the clearance or under
/// the sea.
#[test]
fn the_eye_is_never_in_the_rock_nor_under_the_sea() {
    let k = RigKnobs { min_z: 0.6, ..Default::default() };
    let (clear, min_z) = (k.clear, k.min_z);
    // solid where y > 0
    let wall = Clearance::half_space(Vec3::new(0.0, -1.0, 0.0), 0.0);
    let rig = Rig::new(k).with_ground(wall);

    let mut worst = f64::INFINITY;
    let mut lowest = f64::INFINITY;
    // …measured only where the body itself has the room: no arm can film a
    // character from half a metre off a wall it is leaning on.
    let mut slack = f64::INFINITY;
    for turn in 0..24 {
        let a = turn as f64 / 24.0 * std::f64::consts::TAU;
        let facing = Vec3::new(a.sin(), a.cos(), 0.0);
        rig.reset();
        // walk in from four metres out to right up against the face
        for step in 0..=120 {
            let d = 4.0 - 3.9 * step as f64 / 120.0;
            let s = Subject {
                position: Vec3::new(0.0, -d, 0.5 + 0.5 * (step as f64 * 0.05).sin()),
                velocity: facing * 1.4,
                facing,
                speed: 1.4,
                lean: 0.0,
            };
            rig.follow(&s, DT);
            let eye = rig.eye();
            lowest = lowest.min(eye.z);
            // the eye is never nearer the face than the clearance, nor —
            // when the body is inside it already — than the body is.
            let room = (-s.position.y).min(clear);
            slack = slack.min(-eye.y - room);
            if -s.position.y >= clear {
                worst = worst.min(-eye.y);
            }
        }
    }
    println!(
        "closest the eye came to the face: {worst:.4} m (clearance {clear}); \
         least slack over what the body itself had: {slack:.4} m; lowest: {lowest:.4} m"
    );
    assert!(worst >= clear - 1e-9, "the eye came {worst:.4} m from the face");
    assert!(slack >= -1e-9, "the eye was {slack:.4} m nearer the face than the body");
    assert!(lowest >= min_z - 1e-9, "the eye dipped to {lowest:.4} m");
}

/// The one that is not about feel. A settled camera must be *exactly* the
/// same camera frame after frame, or `kosm_view::History` treats every pass
/// of a standing player as a moved eye and the picture never converges. Two
/// independent runs of the same script must also agree to the last bit.
#[test]
fn a_settled_camera_is_the_same_millimetre_every_time() {
    let script = walk(1.4, 2.0, 2.0);
    let run = || {
        let rig = Rig::default();
        rig.follow(&script[0].1, 0.0);
        let mut cams = Vec::new();
        for (_, s) in &script[1..] {
            let c = rig.follow(s, DT);
            cams.push((c.eye, c.forward, c.fov_deg));
        }
        cams
    };
    let (a, b) = (run(), run());
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.0, y.0, "frame {i}: the eye differed between two runs");
        assert_eq!(x.1, y.1, "frame {i}: the look differed between two runs");
        assert_eq!(x.2, y.2, "frame {i}: the field of view differed");
    }
    // and the tail is one repeated camera, not a converging sequence
    let tail = &a[a.len() - 60..];
    for (i, c) in tail.iter().enumerate() {
        assert_eq!(c.0, tail[0].0, "the settled eye moved at tail frame {i}");
        assert_eq!(c.1, tail[0].1, "the settled look moved at tail frame {i}");
        assert_eq!(c.2, tail[0].2, "the settled field of view moved");
    }
    println!("settled eye {:?}, fov {}", tail[0].0, tail[0].2);
}

/// The springs are closed-form, so the frame rate is not one of their knobs:
/// the same walk at 30, 60 and 240 Hz settles on the same lag.
#[test]
fn the_lag_does_not_depend_on_the_frame_rate() {
    let lag_at = |hz: f64| {
        let dt = 1.0 / hz;
        let rig = Rig::default();
        let mut y = 0.0;
        let sub = |y: f64| Subject {
            position: Vec3::new(0.0, y, 1.0),
            velocity: Vec3::new(0.0, 1.4, 0.0),
            facing: Vec3::new(0.0, 1.0, 0.0),
            speed: 1.4,
            lean: 0.0,
        };
        rig.follow(&sub(y), 0.0);
        for _ in 0..(3.0 * hz) as usize {
            y += 1.4 * dt;
            rig.follow(&sub(y), dt);
        }
        (rig.eye() - rig.wanted_eye()).norm() / 1.4
    };
    let (a, b, c) = (lag_at(30.0), lag_at(60.0), lag_at(240.0));
    println!("lag at 30 Hz {a:.6}, 60 Hz {b:.6}, 240 Hz {c:.6}");
    for l in [a, b, c] {
        assert!((l - 0.15).abs() < 1e-6, "{l:.6} s");
    }
}

/// The shutter: one pass standing still, more while moving, and back to one
/// when the eye has caught up.
#[test]
fn the_shutter_opens_with_the_eye() {
    let rig = Rig::default();
    let script = walk(2.6, 2.0, 2.0);
    rig.follow(&script[0].1, 0.0);
    assert_eq!(rig.shutter_passes(), 1, "a camera that has not moved yet");
    let mut open = 0;
    for (t, s) in &script[1..] {
        rig.follow(s, DT);
        if (1.5..2.0).contains(t) {
            open = open.max(rig.shutter_passes());
        }
    }
    println!("shutter while running: {open} passes; standing: {}", rig.shutter_passes());
    assert!(open >= 3, "the shutter stayed at {open} passes at a run");
    assert_eq!(rig.shutter_passes(), 1, "the shutter never closed again");
}

/// An [`Interest`] pulls the aim as the subject nears it — the cove's
/// doorstep framing, which used to be a second camera and a blend.
#[test]
fn the_aim_turns_to_what_the_level_wants_looked_at() {
    let aperture = Vec3::new(3.0, 6.0, 1.6);
    let rig = Rig::default().with_interest(Interest { point: aperture, reach: 2.0 });
    let far = Subject::walking(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.4, 0.0));
    rig.follow(&far, 0.0);
    let off_axis = (rig.aim(&far) - aperture).norm();

    let near = Subject::walking(Vec3::new(3.0, 5.5, 1.0), Vec3::new(0.0, 0.4, 0.0));
    rig.reset();
    rig.follow(&near, 0.0);
    for _ in 0..120 {
        rig.follow(&near, DT);
    }
    let on = (rig.aim(&near) - aperture).norm();
    println!("aim misses the aperture by {off_axis:.2} m out on the sand, {on:.2} m at the door");
    assert!(on < 0.2 * off_axis, "the aim never turned: {on:.2} m vs {off_axis:.2} m");
}

/// The exposure meter, from a frame twice as bright as the one it was keyed
/// to. A first-order lag with a one-second time constant covers `1 − 1/e` of
/// the way in a second and is within five per cent after three; both are
/// checked, because "settles in about a second" is only true of the first and
/// only useful because of the second.
#[test]
fn the_meter_settles_from_a_brighter_frame() {
    let dim = vec![0.25f32; 3 * 256];
    let bright = vec![0.5f32; 3 * 256];
    let meter = Meter::calibrated(&dim);
    assert!((meter.exposure() - 1.0).abs() < 1e-12);

    let dt = 1.0 / 60.0;
    let mut at_one = 0.0;
    let mut at_three = 0.0;
    for i in 1..=180 {
        let e = meter.follow(&bright, dt);
        if i == 60 {
            at_one = e;
        }
        at_three = e;
    }
    // In the log domain the remaining error after one τ is 1/e of the stop.
    let stop = 2f64.ln();
    let remaining = (at_one / 0.5).ln() / stop;
    println!(
        "meter: {at_one:.4} after 1 s ({:.1}% of the stop left), {at_three:.4} after 3 s",
        remaining * 100.0
    );
    assert!(
        (remaining - (-1f64).exp()).abs() < 0.05 * (-1f64).exp(),
        "the time constant is not a second: {remaining:.4} of the stop left after 1 s"
    );
    assert!(
        (at_three - 0.5).abs() < 0.05 * 0.5,
        "after three time constants the meter was at {at_three:.4}, wanted 0.5"
    );
}

/// The projection is additive: a reference scene rendered under the default
/// camera and under an explicitly rectilinear one is the same bytes, and the
/// fisheye is a different picture rather than the same one warped.
#[test]
fn the_default_projection_renders_the_same_bytes() {
    use std::sync::Arc;
    use kosm_render::{
        Bvh, Camera, Environment, Object, PathTraceOptions, Pbr, Point3, Projection, Scene, Vec3 as RVec3,
        pathtrace, studio_rig,
    };

    let ball = kosm::analytic::ball(phyz_math::Vec3::new(0.0, 0.0, 0.1), 0.08);
    let scene = Scene {
        objects: vec![Object::new(Arc::new(Bvh::build(ball)), Pbr::plastic([0.6, 0.4, 0.3], 0.4, 0.0))],
        lights: studio_rig(Point3::new(0.0, 0.0, 0.1), 0.4),
        env: Environment::default(),
        ground: None,
        sun: None,
        splats: None,
    };
    let cam = Camera::look_at(
        Point3::new(0.35, -0.5, 0.4),
        Point3::new(0.0, 0.0, 0.1),
        RVec3::new(0.0, 0.0, 1.0),
        50.0,
    );
    assert_eq!(cam.projection, Projection::Rectilinear);
    let opts = PathTraceOptions { spp: 4, seed: 0x11, ..Default::default() };
    let a = pathtrace::render(&scene, &cam, 64, 48, &opts);
    let b = pathtrace::render(&scene, &cam.with_projection(Projection::Rectilinear), 64, 48, &opts);
    assert_eq!(a.to_srgb8(0.7, false), b.to_srgb8(0.7, false), "an explicit Rectilinear moved a pixel");
    let f = pathtrace::render(&scene, &cam.with_projection(Projection::Equidistant), 64, 48, &opts);
    assert_ne!(a.to_srgb8(0.7, false), f.to_srgb8(0.7, false), "the fisheye traced rectilinear rays");
}

/// The cove's `cam_*` knobs, read straight off a level's parameters — the
/// door's framing is knobs on the same rig, not a second camera.
#[test]
fn the_coves_knobs_are_the_rigs_knobs() {
    let params = |n: &str, d: f64| match n {
        "cam_back_mm" => 3000.0,
        "cam_up_mm" => 1500.0,
        "cam_side_mm" => 2000.0,
        "cam_ahead_mm" => 3500.0,
        "cam_vfov_deg" => 50.0,
        "cam_face_clear_mm" => 500.0,
        "cam_sea_clear_mm" => 400.0,
        _ => d,
    };
    let k = RigKnobs::from_params(params).in_millimetres().with_water(0.0);
    assert_eq!((k.back, k.up, k.side, k.ahead), (3.0, 1.5, 2.0, 3.5));
    assert_eq!(k.clear, 0.5);
    assert_eq!(k.min_z, 0.4);
    assert_eq!(k.fov_base_deg, 50.0);

    // …and a camera out of it is in millimetres, on the millimetre.
    let rig = Rig::new(k);
    let s = Subject::still(Vec3::new(0.0, 0.0, 1.0));
    let cam = rig.follow(&s, 0.0);
    for v in [cam.eye.x, cam.eye.y, cam.eye.z] {
        assert_eq!(v, v.round(), "the eye is at {v} mm, not on a millimetre");
    }
    assert!(cam.eye.z > 2000.0, "the eye should be a metre and a half up: {}", cam.eye.z);
}

/// The arm against a real level: the cove's own baked field, swept over
/// every open pose in it.
///
/// This is the design's "the eye never behind rock in a sweep of poses",
/// asked of a bake rather than of a half-space. Poses are drawn from the map
/// itself — a coarse grid over the cove, keeping the ones the field says have
/// a body's worth of room — and each is filmed from eight facings. The claim
/// is the arm's own: the eye ends up at least as clear of the rock as the
/// body is, and `clear` of it wherever the body has that much room.
///
/// Ignored because it wants a baked map: `cargo run -p kosm-cli -- run rune`
/// writes `out/maps/cove/sdf.bin`, and this reads it. Run it with
/// `cargo test -p kosm --release --test rig -- --ignored --nocapture`.
#[test]
#[ignore = "wants out/maps/cove/sdf.bin from a `kosm run rune`"]
fn the_arm_sweeps_the_cove_without_entering_it() {
    use std::sync::Arc;
    let path = std::path::Path::new("../../out/maps/cove/sdf.bin");
    let path = if path.exists() { path } else { std::path::Path::new("out/maps/cove/sdf.bin") };
    let grid = Arc::new(
        kosm_scan::SdfGrid::load(path)
            .unwrap_or_else(|e| panic!("no cove bake at {}: {e}", path.display())),
    );
    let lo = grid.origin;
    let hi = lo
        + Vec3::new(
            (grid.nx - 1) as f64 * grid.cell,
            (grid.ny - 1) as f64 * grid.cell,
            (grid.nz - 1) as f64 * grid.cell,
        );

    let k = RigKnobs::default();
    let clear = k.clear;
    let rig = Rig::new(k).with_ground(Clearance::sdf(grid.clone()));

    let (mut poses, mut shortened, mut worst) = (0usize, 0usize, f64::INFINITY);
    let mut tightest = 3.5f64;
    let n = 24;
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..6 {
                let p = Vec3::new(
                    lo.x + (hi.x - lo.x) * ix as f64 / (n - 1) as f64,
                    lo.y + (hi.y - lo.y) * iy as f64 / (n - 1) as f64,
                    lo.z + (hi.z - lo.z) * iz as f64 / 5.0,
                );
                // only poses a body could actually stand in
                let Some(room) = grid.sample(p) else { continue };
                if room < 1.0 {
                    continue;
                }
                for turn in 0..8 {
                    let a = turn as f64 / 8.0 * std::f64::consts::TAU;
                    let facing = Vec3::new(a.sin(), a.cos(), 0.0);
                    rig.reset();
                    let s = Subject {
                        position: p,
                        velocity: facing * 1.2,
                        facing,
                        speed: 1.2,
                        lean: 0.0,
                    };
                    rig.follow(&s, 0.0);
                    poses += 1;
                    let eye = rig.eye();
                    let d = grid.sample(eye).unwrap_or(f64::INFINITY);
                    assert!(
                        d >= clear.min(room) - 1e-6,
                        "at {p:?} facing {facing:?} the eye is {d:.3} m from the rock \
                         (the body had {room:.3} m)"
                    );
                    worst = worst.min(d);
                    let arm = (eye - p).norm();
                    if arm < 3.4 {
                        shortened += 1;
                        tightest = tightest.min(arm);
                    }
                }
            }
        }
    }
    println!(
        "cove: {poses} poses swept, the arm shortened on {shortened} of them \
         (tightest {tightest:.2} m of a wanted 3.41), closest the eye ever came \
         to the rock {worst:.3} m against a clearance of {clear}"
    );
    assert!(poses > 100, "only {poses} open poses in the cove — is the bake right?");
    assert!(shortened > 0, "the arm never shortened anywhere in the cove");
}

/// **A breathing body does not move the camera.** A body at rest breathes: its
/// centre rises and falls five millimetres at a quarter of a hertz and sways
/// three, its lean wanders a fifth of a degree, and the velocity that carries
/// is millimetres a second. None of it may reach the camera, or the history
/// under it never sees a still frame and the settle blend never rises. Stop
/// after a walk and breathe for eight seconds: once the eye's spring has
/// settled, every camera is the same camera, and it is the camera an
/// unbreathing stop settles on, to the bit.
#[test]
fn a_breathing_body_does_not_move_the_camera() {
    let stop = 2.0;
    let breathe = |script: &mut Vec<(f64, Subject)>| {
        let w = std::f64::consts::TAU * 0.25;
        for (t, s) in script.iter_mut() {
            let tb = *t - stop;
            if tb <= 0.0 {
                continue;
            }
            let (sn, cs) = (w * tb).sin_cos();
            s.position = s.position + Vec3::new(0.0, 0.003 * sn, 0.005 * sn);
            s.velocity = Vec3::new(0.0, 0.003 * w * cs, 0.005 * w * cs);
            s.speed = s.velocity.y.abs();
            s.lean = 0.2f64.to_radians() * sn;
        }
    };
    let run = |script: &[(f64, Subject)]| {
        let rig = Rig::default();
        rig.follow(&script[0].1, 0.0);
        script[1..]
            .iter()
            .map(|(t, s)| {
                let c = rig.follow(s, DT);
                (*t, c.eye, c.forward, c.fov_deg)
            })
            .collect::<Vec<_>>()
    };
    let calm = run(&walk(1.4, stop, 8.0));
    let mut script = walk(1.4, stop, 8.0);
    breathe(&mut script);
    let breathing = run(&script);
    let tail: Vec<_> = breathing.iter().filter(|c| c.0 > stop + 1.5).collect();
    assert!(tail.len() > 700, "{} frames of breathing", tail.len());
    for c in &tail {
        assert_eq!(c.1, tail[0].1, "the eye moved with the breath at t = {:.2} s", c.0);
        assert_eq!(c.2, tail[0].2, "the look moved with the breath at t = {:.2} s", c.0);
        assert_eq!(c.3, tail[0].3, "the field of view moved with the breath at t = {:.2} s", c.0);
    }
    let settled = calm.last().expect("a calm run");
    assert_eq!(tail[0].1, settled.1, "the breathing stop settled somewhere the calm one did not");
    assert_eq!(tail[0].2, settled.2);
    assert_eq!(tail[0].3, settled.3);
    println!("{} breathing frames, one camera: eye {:?}", tail.len(), tail[0].1);
}
