//! The rune and its hint, checked.
//!
//! Two tracers see the same being here. [`rune`](super::rune) fires photons
//! at it through `kosm-render` and asks the door what landed;
//! [`hint`](super::hint) walks a deterministic lattice through it on
//! `tang::Scalar` and differentiates. The first three tests say they are
//! looking at one solid and that its score is the fraction of the sun it
//! claims to be; the rest say the derivative is the score's.
//!
//! Where a test runs on a cove that is not `scene.rs`, it says so and
//! says which knob it moved. The authored cove cannot be solved — see
//! [`the_rune_scores_at_the_authored_solution`] — and that is a fact about
//! the level, recorded here rather than hidden by a looser assertion.

use kosm_render::math::{Point3, Vec3};
use kosm_render::{Geometry, Ray};
use tang::Dual;

use super::CoveScene;
use super::hint::{self, Knobs};
use super::being;
use super::rune::{self, Piece, Pieces, Pose};
use kosm::glass::Shape;

/// The photon budget the tests score on. Enough that `frac` is stable in its
/// third digit, cheap enough that the file runs in seconds.
const PHOTONS: usize = 200_000;

/// A cove whose keyhole this being can actually light.
///
/// Three knobs move, and the solve's report says why each one has to:
///
/// - `sun_az_deg` 200 → 250. A sphere of glass of index `n` focuses at
///   `n·r/(2(n−1))` from its own centre — 1.47 r, half a metre for the
///   authored being — and a body standing against a wall is `r/cos θ` from
///   it, where θ is the angle between the light and the wall's normal. At
///   200° the sun meets the door at 71°, so the wall is 3.15 r away and the
///   focus lands a metre and a half short of it, inside the glass's own
///   shadow. The condition is `cos θ ≥ 2(n−1)/n`, which is θ ≤ 47°, and 250°
///   gives 20°. It is still a low sun out over the sea, which is what the
///   design asks of it.
/// - `aperture_z_mm` 1500 → 400. The sun is *above* the horizon, so the light
///   it throws through the being travels downward and can never land higher
///   than the being's own head. The keyhole has to be inside the band the
///   caustic can reach, which is roughly the being's centre height above the
///   sill.
/// - `being_h_mm` 1400 → 1000. The design's own listed risk: a capsule's
///   caustic is a *line*, and a keyhole is a disc. Shortening the being
///   toward a sphere gathers the line into a spot.
fn reachable() -> CoveScene {
    let mut s = CoveScene::bundled().expect("the cove builds");
    s.sun_az = 250.0f64.to_radians();
    s.aperture_z = 0.40;
    s.door_x = 0.10;
    s.being_h = 1.0;
    s
}

/// The workspace's `out/`, not the crate's: a test runs with its own manifest
/// directory as the working directory, and the level's solved knobs belong
/// beside the other levels' and not under `crates/`.
fn out_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out")
}

/// The solved pose of [`reachable`], found by the grid and a downhill
/// simplex on 400 k photons. Recorded rather than re-solved because the
/// search is a minute of grid and this file is a test.
const REACHED: Pose = Pose { x: -0.0200, y: 15.6490, tilt: -0.025 };

// ─── the solid ────────────────────────────────────────────────────────────

/// The photon tracer's capsule and the lattice tracer's are one body.
///
/// This is the reason [`Pieces`] exists at all: a capsule assembled from a
/// cylinder and two spheres would answer these rays with interfaces that are
/// not on its surface.
#[test]
fn the_being_is_one_solid_to_both_tracers() {
    let scene = CoveScene::bundled().unwrap();
    let pose = Pose { x: -1.0, y: 12.0, tilt: 0.12 };
    let (a, b, r) = rune::being_capsule(&scene, &pose);
    let solid = Shape::Capsule { a, b, r };
    let pieces = Pieces(vec![Piece::Capsule { a: Point3::from_vec(a), b: Point3::from_vec(b), r }]);
    let centre = (a + b) * 0.5;

    let dirs = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.6, -0.5, 0.62).normalize(),
        Vec3::new(-0.3, 0.9, -0.31).normalize(),
        Vec3::new(0.2, 0.1, 0.97).normalize(),
    ];
    for d in dirs {
        // from outside, aimed at the body and at a miss beside it
        for offset in [Vec3::zero(), Vec3::new(0.0, 0.0, 1.0).cross(&d).normalize() * (r * 0.8), Vec3::new(0.0, 0.0, 1.0).cross(&d).normalize() * (r * 3.0)] {
            let o = centre + offset - d * 8.0;
            let ray = Ray::new(Point3::from_vec(o), d);
            let mine = Geometry::intersect(&pieces, &ray, 0, 1e-7, f64::INFINITY);
            match (solid.enter(o, d), mine) {
                (Some((t, n)), Some(hit)) => {
                    assert!((t - hit.t).abs() < 1e-9, "entry {t} vs {}", hit.t);
                    assert!((n - hit.normal.into_inner()).norm() < 1e-9, "entry normal {n:?} vs {:?}", hit.normal);
                }
                (None, None) => {}
                (x, y) => panic!("the two tracers disagree on a hit: {} vs {}", x.is_some(), y.is_some()),
            }
        }
        // and from inside: the exit is the farthest of every root
        let o = centre;
        let ray = Ray::new(Point3::from_vec(o), d);
        let mut all = Vec::new();
        Geometry::intersect_all(&pieces, &ray, 0, &mut all);
        let far = all.iter().map(|h| h.t).fold(f64::NEG_INFINITY, f64::max);
        let (t, n) = solid.exit(o, d).expect("a ray from inside leaves");
        assert!((t - far).abs() < 1e-9, "exit {t} vs {far}");
        let at = all.iter().find(|h| (h.t - far).abs() < 1e-12).unwrap();
        assert!((n - at.normal.into_inner()).norm() < 1e-9);
    }
}

// ─── the score ────────────────────────────────────────────────────────────

/// The score is what it says it is: the fraction of the sun the being caught
/// that came out through the keyhole.
///
/// The tracer's own `emitted_power` is *not* that denominator — it is the sun
/// off a disc that covers the being's bounding sphere, misses included — and
/// this pins the difference, so a change in how `caustics::trace` aims cannot
/// quietly rescale the rune.
#[test]
fn the_score_is_the_fraction_of_the_sun_the_being_caught() {
    let scene = reachable();
    let (map, score) = rune::trace(&scene, &REACHED, PHOTONS);

    // the aiming disc: the radius of the refractor's bounding sphere, which
    // for one capsule is half the diagonal of its box
    let (a, b, r) = rune::being_capsule(&scene, &REACHED);
    let d = Vec3::new(
        (b.x - a.x).abs() + 2.0 * r,
        (b.y - a.y).abs() + 2.0 * r,
        (b.z - a.z).abs() + 2.0 * r,
    );
    let extent = 0.5 * d.norm();
    let emitted = 0.2126 * map.emitted_power()[0] as f64
        + 0.7152 * map.emitted_power()[1] as f64
        + 0.0722 * map.emitted_power()[2] as f64;
    let disc = std::f64::consts::PI * extent * extent;
    assert!(
        (emitted - disc).abs() < 0.01 * disc,
        "the pass emitted {emitted:.4} W where the aiming disc is {disc:.4} m²"
    );
    assert!(
        score.incident < emitted,
        "the being catches less than the whole aiming disc: {:.4} vs {emitted:.4}",
        score.incident
    );
    assert!((0.0..=1.0).contains(&score.frac), "frac {} is not a fraction", score.frac);
}

/// **Test 2, as the plan states it.** At the authored solution the score
/// clears `open_frac`.
///
/// Ignored, and it is the level that is wrong, not the test. The authored
/// cove cannot reach `open_frac` from anywhere: a 350 mm glass capsule of
/// index 1.5168 focuses 0.51 m from its own axis, the sun at 200° azimuth
/// puts the door's face 3.15 radii away along the light, and the keyhole at
/// 1500 mm above the sill is a metre above the highest a downward-travelling
/// beam can land. The best pose anywhere on the beach scores 0.0003 against
/// an `open_frac` of 0.3. [`reachable`] carries the three knob changes that
/// fix it and [`the_rune_scores_at_a_keyhole_the_light_can_reach`] is this
/// same test on that cove.
#[test]
fn the_rune_scores_at_the_authored_solution() {
    let scene = CoveScene::bundled().unwrap();
    let pose = Pose::solution(&scene);
    let pose = if pose.is_solved() { pose } else { rune::solve(&scene, 100_000).0 };
    let here = rune::score(&scene, &pose, PHOTONS).frac;
    let there = rune::score(&scene, &Pose { x: pose.x + 10.0, ..pose }, PHOTONS).frac;
    println!("at the solution {here:.5}, ten metres along {there:.5}, open_frac {:.3}", scene.open_frac);
    assert!(here > scene.open_frac);
    assert!(there < 0.02);
}

/// **Test 2**, on the cove of [`reachable`]: at the solved pose the score
/// clears `open_frac`, and ten metres along the beach it is nothing.
#[test]
fn the_rune_scores_at_a_keyhole_the_light_can_reach() {
    let scene = reachable();
    let here = rune::score(&scene, &REACHED, PHOTONS).frac;
    let there = rune::score(&scene, &Pose { x: REACHED.x + 10.0, ..REACHED }, PHOTONS).frac;
    println!(
        "the rune at ({:+.3}, {:+.3}) m, {:+.2}°: {here:.5}; ten metres along the beach: {there:.5}; open_frac {:.3}",
        REACHED.x,
        REACHED.y,
        REACHED.tilt.to_degrees(),
        scene.open_frac
    );
    assert!(here > scene.open_frac, "the solved pose scores {here:.5}, under open_frac {:.3}", scene.open_frac);
    assert!(there < 0.02, "ten metres along the beach still scores {there:.5}");
}

// ─── the hint ─────────────────────────────────────────────────────────────

/// The lattice tracer and the photon tracer agree on the same number.
///
/// Not to the third digit, and the reasons are physical and stated: the
/// photon pass runs a flat `n_d` while the lattice runs five Sellmeier bands,
/// so the lattice's focus is the *dispersed* one and reads low exactly where
/// the focus is tight; the lattice's keyhole has a rim softened over a few
/// cells where the photon map's is hard; and `light.rs` folds the receiving
/// plane's obliquity into its grid, which is divided back out at the nominal
/// angle rather than each ray's own.
#[test]
fn the_hint_agrees_with_the_photons() {
    let scene = reachable();
    for pose in [REACHED, Pose { y: 15.50, ..REACHED }, Pose { tilt: 0.06, ..REACHED }] {
        let photons = rune::score(&scene, &pose, 400_000).frac;
        let lattice = hint::score(&scene, &pose);
        println!(
            "at ({:+.3}, {:+.3}) m, {:+.2}°: photons {photons:.5}, lattice {lattice:.5}, ratio {:.3}",
            pose.x,
            pose.y,
            pose.tilt.to_degrees(),
            lattice / photons
        );
        assert!(
            (lattice - photons).abs() < 0.3 * photons,
            "the two tracers are {:.0}% apart",
            100.0 * (lattice / photons - 1.0).abs()
        );
    }
}

/// **Test 3.** `∂frac/∂(x, y, tilt)` on `Dual` against central differences.
///
/// Not to four digits, and the reason is one line of `light.rs`: the lattice
/// it fires is a cone aimed at the shape's bounding sphere, and the aiming is
/// computed on `f64` real parts. So a dual holds the lattice still while the
/// being moves through it, and a central difference carries the lattice along
/// — the same physical integral, sampled two ways. The disagreement is
/// therefore a sampling error and falls as `1/side`: at 200², 400² and 800²
/// rays a band the `x` knob agrees to 2.7 %, 1.0 % and 0.6 %. The two knobs
/// that do not move the capsule's midpoint — `y` barely, `tilt` not at all,
/// since a lean is about the centre — agree to a part in a thousand, which is
/// what the whole check would read if the lattice were pinned.
///
/// The other half of the story is the accumulation grid, and that one *is*
/// fixed here: `light.rs` centres its grid on where the light lands, so it
/// slides under the keyhole as the being moves. Summing over a hard disc of
/// cell centres made the score a staircase and the agreement worse than one
/// digit; `hint::through`'s soft rim and five-millimetre cell are what make
/// central differences converge as `h` shrinks at all.
///
/// What the three poses below actually read, at `h` of 2 cm and 0.2°: `x`
/// within 5 %, 1.5 % and 1.4 %; `y` within 0.4 %, 1.5 % and 0.6 %; `tilt`
/// within 0.02 %, 0.05 % and 0.9 % — four significant digits on the one knob
/// that leaves the lattice alone.
#[test]
fn the_hint_matches_central_differences() {
    let scene = reachable();
    let h = [0.02, 0.02, 0.2f64.to_radians()];
    // strictly inside the beach: at `y = 15.65` the being's surface is on the
    // door and a difference step of two centimetres pushes it through
    for pose in [Pose { y: 15.50, ..REACHED }, Pose { x: 0.0, y: 15.40, tilt: 0.05 }, Pose { x: 0.10, y: 15.30, tilt: -0.04 }] {
        let dual = hint::gradient(&scene, &pose);
        let mut fd = [0.0; 3];
        for i in 0..3 {
            let (mut lo, mut hi) = (pose, pose);
            for (p, sign) in [(&mut lo, -1.0), (&mut hi, 1.0)] {
                match i {
                    0 => p.x += sign * h[i],
                    1 => p.y += sign * h[i],
                    _ => p.tilt += sign * h[i],
                }
            }
            fd[i] = (hint::score(&scene, &hi) - hint::score(&scene, &lo)) / (2.0 * h[i]);
        }
        let scale = fd.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        println!(
            "at ({:+.3}, {:+.3}) m, {:+.2}°: dual [{:+.5}, {:+.5}, {:+.5}]  differences [{:+.5}, {:+.5}, {:+.5}]",
            pose.x,
            pose.y,
            pose.tilt.to_degrees(),
            dual[0],
            dual[1],
            dual[2],
            fd[0],
            fd[1],
            fd[2]
        );
        for i in 0..3 {
            assert!(
                (dual[i] - fd[i]).abs() < 0.06 * fd[i].abs() + 0.01 * scale,
                "knob {i}: dual {} vs central differences {}",
                dual[i],
                fd[i]
            );
        }
    }
}

/// The dual's real part is the `f64` answer, which is the cheap half of
/// "it is the same code".
///
/// **To the bit**, on both bodies. That is a stronger claim than it looks:
/// `light.rs` aims its cone of rays on whatever scalar the caller brought,
/// so the `Dual` pass builds the frame with `Dual` arithmetic — and a
/// `Dual`'s division is a multiply by the divisor's reciprocal where an
/// `f64`'s is a real divide. An ulp on the cone's axis is the initial
/// condition of a walk that reflects up to eight times inside the glass,
/// and that walk amplifies it until a grazing ray picks the other side of
/// total internal reflection: sixty-odd rays in half a million changed their
/// minds, and the score moved in its seventh digit. `trace_onto` shares one
/// reciprocal instead, which is the same arithmetic on both scalars.
///
/// **And not only in the cone.** Every division on `S` between the lamp and
/// the keyhole is a reciprocal now — the landing `t`, the grid cell, Snell's
/// ratio and Fresnel's two in `kosm_render::optics`, the shapes' normals and
/// roots in `glass.rs`, and the score's own ratio in `hint.rs`. Before that
/// the claim held only where the rounding happened to agree: when the arm's
/// solve was made exact and the staged lens moved, 590 of the grid's cells a
/// band came out an ulp apart and the score's real part landed one ulp off
/// (0.8038051637735457 against …456) at the staged doorstep, at the solution
/// and at the solved hold's doorstep alike. After it, not one cell differs.
#[test]
fn the_dual_carries_the_score_it_differentiates() {
    let scene = reachable();
    let k = Knobs::of(&REACHED);
    let plain = hint::score_of::<f64>(&scene, k, hint::RAYS);
    for i in 0..3 {
        let d: Dual<f64> = hint::score_dual(&scene, k.seed(i));
        assert!(d.real == plain, "the capsule, seed {i}: {} vs {plain}", d.real);
    }

    // and the hero's lens, whose six knobs are the ones the hint is really
    // taken in
    let cove = CoveScene::bundled().unwrap();
    let pose = rune::hero_doorstep(&cove, &rune::HeroPose::default());
    let lens = hint::LensKnobs::of(&rune::hero_lens(&cove, &pose));
    let plain = hint::score_lens_of::<f64>(&cove, lens, hint::RAYS);
    println!("the lattice scores the held lens {plain:.6} at the staged doorstep");
    for i in 0..6 {
        let d = hint::score_hero_dual(&cove, lens.seed(i), hint::RAYS);
        assert!(d.real == plain, "the lens, seed {i}: {} vs {plain}", d.real);
    }
}

/// The glint's objective is defined everywhere the score is not: a being far
/// down the beach still knows which way to walk, because the beam it throws
/// on the door's plane has a place even when that place is a hundred metres
/// wide of the keyhole.
#[test]
fn the_glint_points_home_from_the_far_end_of_the_beach() {
    let scene = reachable();
    let far = Pose { x: -6.0, y: -10.0, tilt: 0.0 };
    assert!(hint::score(&scene, &far) == 0.0, "the bare score is flat out there, which is the problem");
    let g = hint::guided_gradient(&scene, &far, hint::SWEEP_RAYS).expect("the beam lands somewhere");
    println!("at ({:+.1}, {:+.1}) the glint points ({:+.4}, {:+.4}, {:+.4})", far.x, far.y, g[0], g[1], g[2]);
    assert!(g[1] > 0.0, "up the beach, toward the door: {g:?}");
}

// ─── the long ones ────────────────────────────────────────────────────────

/// **Test 4, solvability.** Gradient ascent from a 6×6 grid of spawns, and
/// `out/cove/solvable.txt`.
///
/// Ignored because it is two minutes of tracing and because, on
/// `scene.rs`, no spawn can reach `open_frac` — no *pose* can. Run it
/// with `--ignored` to write the report; the numbers in it are the level's
/// verdict, not the sweep's.
#[test]
#[ignore = "two minutes of tracing; `kosm run rune` runs the same sweep and reports it"]
fn the_cove_is_solvable() {
    let scene = CoveScene::bundled().unwrap();
    let climbs = hint::solvable(&scene, 6, 200, &out_dir().join("cove/solvable.txt")).unwrap();
    let solved = climbs.iter().filter(|c| c.solved).count();
    assert_eq!(solved, climbs.len(), "{} of {} spawns reached open_frac", solved, climbs.len());
}

/// **Test 4** on the cove of [`reachable`]: from every spawn in the same
/// 6×6 grid, the glint's ascent walks the being to a pose over `open_frac`.
///
/// Ignored for its minute of tracing, not for its verdict. It is the check
/// the design says a cove has to pass before it ships, and it passes here —
/// which is what makes the three knobs in [`reachable`] a fix and not a
/// guess.
#[test]
#[ignore = "a minute of tracing; writes out/cove/solvable-reachable.txt"]
fn the_reachable_cove_is_solvable() {
    let scene = reachable();
    let climbs = hint::solvable(&scene, 6, 200, &out_dir().join("cove/solvable-reachable.txt")).unwrap();
    let solved = climbs.iter().filter(|c| c.solved).count();
    assert_eq!(solved, climbs.len(), "{} of {} spawns reached open_frac", solved, climbs.len());
}

/// The solve, written back into the document. Ignored: it is the author's
/// pass, not a check, and it takes minutes.
#[test]
#[ignore = "the author's solve: minutes of tracing, and it writes out/solved/rune.params"]
fn the_cove_solves() {
    let scene = CoveScene::bundled().unwrap();
    let (pose, score) = rune::solve_and_record(&scene, 100_000, &out_dir()).unwrap();
    println!("solved ({:+.3}, {:+.3}) m at {:+.2}°: {:.5}", pose.x, pose.y, pose.tilt.to_degrees(), score.frac);
}


// ─── the hero ─────────────────────────────────────────────────────────────

/// The hero's lens, as arithmetic and as a body.
///
/// [`rune::hero_lens`] is a two-link solve and a grip and takes microseconds;
/// [`kosm::player::Body::held`] is where an arm with mass actually got to
/// after a second of PD. The solve is written on the first and the game is
/// played with the second, so they had better be the same lens — and this is
/// where the claim is cashed, in centimetres and degrees rather than in a
/// comment.
///
/// A plane and not the cove's baked field: the hero stands still at one
/// point, and the sand there *is* a plane at that height. What is being
/// measured is an arm, not a bake.
#[test]
fn the_heros_arithmetic_is_where_the_body_puts_the_lens() {
    use kosm::player::{Air, Body, Drive, Plane, Tool};
    let scene = CoveScene::bundled().unwrap();
    let rig = &*being::HERO_RIG;
    let cases = [
        rune::hero_doorstep(&scene, &rune::HeroPose::default()),
        rune::HeroPose { x: -2.0, y: 12.0, yaw: 1.1, aim_el: 0.55, aim_az: -0.62, cant: 0.9, },
        rune::HeroPose { x: 3.0, y: 8.0, yaw: -0.4, aim_el: 0.95, aim_az: -0.35, cant: -0.5 },
    ];
    for pose in cases {
        let want = rune::hero_lens_pose(&scene, &pose);
        let mut body = Body::new(rig.spec.clone().with_dt(1e-3));
        body.hold(Tool::new("lens").with_grip(being::lens_grip(pose.cant)));
        let ground = Plane::at(scene.sand_z_at(pose.x, pose.y));
        body.place(pose.x, pose.y, scene.sand_z_at(pose.x, pose.y), pose.yaw, 0.0);
        let aim = rune::hero_aim(&scene, &pose);
        body.run_for(3.0, &Drive { aim: Some(aim), ..Drive::STILL }, &ground, &Air);
        let (got, _) = body.held().expect("the hero is holding the lens");

        let miss = (got.pos - want.pos).norm();
        // the angle between the two optical axes
        let (ga, wa) = (got.rot.mul_vec(Vec3::new(0.0, 0.0, 1.0)), want.rot.mul_vec(Vec3::new(0.0, 0.0, 1.0)));
        let turn = ga.dot(&wa).clamp(-1.0, 1.0).acos();
        println!(
            "hero at ({:+.2}, {:+.2}) holding {:+.1}° up, {:+.1}° canted: the arithmetic puts the glass at ({:+.3}, {:+.3}, {:+.3}) and the body puts it {:.1} mm away, {:.2}° off axis (the body leaned {:.2}°)",
            pose.x, pose.y, pose.aim_el.to_degrees(), pose.cant.to_degrees(),
            want.pos.x, want.pos.y, want.pos.z, miss * 1e3, turn.to_degrees(), body.lean().to_degrees(),
        );
        // Five millimetres and half a degree. Both hands land *on* the aim —
        // the two-link solve is exact and the joints hold the arm's weight as
        // feed-forward — so the gap is the PD's tracking and the few
        // millimetres the settled pelvis breathes and leans off `hero_root`.
        // It was 11.6 mm and 1.54° while the solve left the shoulder's twist
        // to chance and so passed every millimetre of the pelvis to the hand.
        assert!(miss < 0.005, "the glass is {:.1} mm from where the arithmetic said", miss * 1e3);
        assert!(turn.to_degrees() < 0.5, "the optical axis is {:.2}° off", turn.to_degrees());
    }
}

/// The lens is one solid to both tracers, exactly as the capsule is.
///
/// `rune::Piece::Lens` answers the photon pass and `glass::Shape::Lens`
/// answers the lattice and its duals. Two statements of the intersection of
/// two spheres, and this is the test that says they are one piece of glass.
#[test]
fn the_lens_is_one_solid_to_both_tracers() {
    let scene = CoveScene::bundled().unwrap();
    let pose = rune::hero_doorstep(&scene, &rune::HeroPose::default());
    let held = rune::hero_lens(&scene, &pose);
    let (r, a, h) = rune::lens_numbers_m();
    let u = held.axis.normalize();
    let piece = Pieces(vec![rune::lens_piece(&held)]);
    let solid: Shape<f64> = hint::lens_shape(
        tang::Vec3::new(held.centre.x, held.centre.y, held.centre.z),
        tang::Vec3::new(held.axis.x, held.axis.y, held.axis.z),
    );
    let tv = |v: Vec3| tang::Vec3::new(v.x, v.y, v.z);
    // the two sphere centres agree
    match &solid {
        Shape::Lens { c1, c2, r1, r2 } => {
            let want = (held.centre - u * a, held.centre + u * a);
            assert!((c1 - &tv(want.0)).norm() < 1e-12 && (c2 - &tv(want.1)).norm() < 1e-12);
            assert!((r1 - r).abs() < 1e-12 && (r2 - r).abs() < 1e-12);
        }
        _ => panic!("lens_shape did not make a lens"),
    }
    // and every ray sees the same surface: down the axis, across the rim, and
    // one that misses beside it
    let side = u.cross(&Vec3::new(0.0, 0.0, 1.0)).normalize();
    for d in [u, -u, side, (u + side * 0.6).normalize(), Vec3::new(0.31, 0.87, -0.38).normalize()] {
        for off in [0.0, 0.5 * h, 0.92 * h, 1.4 * h] {
            let o = held.centre + side.cross(&d).normalize() * off - d * 3.0;
            let ray = Ray::new(Point3::from_vec(o), d);
            let mine = Geometry::intersect(&piece, &ray, 0, 1e-7, f64::INFINITY);
            match (solid.enter(tv(o), tv(d)), mine) {
                (Some((t, n)), Some(hit)) => {
                    assert!((t - hit.t).abs() < 1e-9, "entry {t} vs {}", hit.t);
                    let hn = hit.normal.into_inner();
                    assert!((n - tv(hn)).norm() < 1e-9, "entry normal {n:?} vs {hn:?}");
                }
                (None, None) => {}
                (x, y) => panic!("the two tracers disagree at offset {off}: {} vs {}", x.is_some(), y.is_some()),
            }
        }
        // from inside, the exit is the nearest of the two spheres' far roots
        let o = held.centre;
        let ray = Ray::new(Point3::from_vec(o), d);
        let mut all = Vec::new();
        Geometry::intersect_all(&piece, &ray, 0, &mut all);
        let far = all.iter().map(|hit| hit.t).fold(f64::NEG_INFINITY, f64::max);
        let (t, n) = solid.exit(tv(o), tv(d)).expect("a ray from inside leaves");
        assert!((t - far).abs() < 1e-9, "exit {t} vs {far}");
        // The normal belongs to one of the two caps, and which one is not
        // always a question with an answer. A ray leaving the centre *across*
        // the axis exits on the **rim**, where the caps meet: the two
        // spheres' far roots are the same number to the bit, and an edge does
        // not have a normal to be right about. So what is checked is that
        // both tracers are on the *surface* — each one's normal is the
        // outward normal of one of the two spheres at the point they agree
        // on — and not that they picked the same cap out of a tie.
        let at = held.centre + d * far;
        let caps = [(at - (held.centre - u * a)) / r, (at - (held.centre + u * a)) / r];
        let on_surface = |v: Vec3| caps.iter().any(|c| (v - c).norm() < 1e-9);
        assert!(on_surface(Vec3::new(n.x, n.y, n.z)), "the lattice's exit normal {n:?} is on neither cap");
        let seen = all.iter().find(|hit| (hit.t - far).abs() < 1e-9).expect("the photon tracer found no face");
        assert!(on_surface(seen.normal.into_inner()), "the photon tracer's exit normal is on neither cap");
    }
    // the thing is the right size: the rim is `h` across and the glass is
    // `2(r − a)` thick, which is `hero/kit.rs`'s own arithmetic
    let up = u.cross(&side).normalize();
    for (dir, want) in [(side, h), (up, h), (u, r - a)] {
        let (t, _) = solid.exit(tv(held.centre), tv(dir)).unwrap();
        assert!((t - want).abs() < 1e-9, "the glass is {t} along {dir:?}, not {want}");
    }
}

/// **The hero's hint.** `∂frac/∂(x, y, yaw, aim_el, aim_az, cant)` against
/// central differences of the same lattice score.
///
/// [`hint::gradient_hero`] is six exact dual passes in the *lens's* six knobs
/// chained onto twelve differences of the arm's arithmetic; this differences
/// the whole composition instead. What is being checked is therefore the
/// chain rule and the duals, against a sampling of the same integral.
///
/// **It is tight, and the capsule's is not**, and the difference is the
/// aperture. A lens is a thin disc that spends its life being turned, so its
/// silhouette sweeps across the lattice and rays step in and out of it a
/// whole ray at a time — a step a `Dual` cannot see, a step a difference
/// sees all of, and a step that does not get smaller when rays are added
/// because there are proportionally more of them. Two things fixed it, and
/// both are in `crates/kosm`:
///
/// * `glass::Shape::rim_weight` feathers the outer
///   `glass::RIM_FEATHER` of the aperture with a smoothstep, so the
///   integrand reaches the silhouette at zero with a zero slope and the sum
///   is C¹ in every knob. The hint therefore scores a lens whose outer 5 %
///   is a graded filter and `rune.rs`'s photon pass scores the hard one;
///   [`the_hint_agrees_with_the_photons`] is where that gap is measured.
/// * `light::trace_onto` snaps its accumulation grid to the cell lattice the
///   keyhole itself sits on, so the grid can only ever move by a whole cell
///   — a relabelling, which changes no score — instead of sliding under a
///   keyhole that is not sliding with it.
///
/// Staged at the doorstep for the way the **solved** pose holds the glass
/// (its lift, swing and cant, with the chief ray putting the boots down).
/// The beam is smaller than the keyhole there, so walking a centimetre
/// either way is free and the score is flat in `x` and `y` by design; the
/// four that turn the glass carry the check. Measured at 200² rays a band and
/// a step of 2 cm: `x` −0.0003/−0.0003, `y` +0.0004/+0.0004, `yaw`
/// −0.0723/−0.0724, `aim_el` +0.0419/+0.0418, `aim_az` +0.0049/+0.0049,
/// `cant` +0.0670/+0.0671 (dual/difference).
///
/// Why not the default hold's doorstep any more: when the two-link solve was
/// made exact the default hold's lens moved 18 cm and turned 14°, and there
/// the beam straddles the keyhole's edge in `x`, where the score is curved on
/// the scale of the step. The 2 cm difference then reads −0.4798 against a
/// dual of −0.4518 (5.9 %), and halving the step walks it in — −0.4591,
/// −0.4540, −0.4526, −0.4523 at 1 cm to 1.25 mm — so the dual is right and
/// the step's truncation is what failed. Same bound, same step; the pose it
/// is taken at is the solution's.
#[test]
fn the_heros_hint_matches_central_differences() {
    let scene = CoveScene::bundled().unwrap();
    let seed = rune::hero_doorstep(&scene, &rune::HeroPose::solution(&scene));
    let rays = 200 * 200;
    let h = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02];
    let dual = hint::gradient_hero(&scene, &seed, rays);
    let mut fd = [0.0; 6];
    let at = |p: &rune::HeroPose| hint::score_lens_of(&scene, hint::LensKnobs::of(&rune::hero_lens(&scene, p)), rays);
    for k in 0..6 {
        let (mut lo, mut hi) = (seed, seed);
        *lo.knob(k) -= h[k];
        *hi.knob(k) += h[k];
        fd[k] = (at(&hi) - at(&lo)) / (2.0 * h[k]);
    }
    let scale = fd.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    for k in 0..6 {
        println!(
            "  knob {k}: dual {:+10.5}   differences {:+10.5}   {:+6.2}%",
            dual[k],
            fd[k],
            100.0 * (dual[k] - fd[k]) / fd[k].abs().max(1e-12)
        );
    }
    for k in 0..6 {
        assert!(
            (dual[k] - fd[k]).abs() < 0.02 * fd[k].abs() + 2e-4 * scale,
            "knob {k}: dual {} vs central differences {}",
            dual[k],
            fd[k]
        );
    }
    // and the two knobs the design says should dominate do
    assert!(scale > 0.0, "the score does not move at all");
    assert!(dual[3].abs() > dual[4].abs(), "the lift moves the score less than the swing does");
}

/// The lens's beam centre is analytic, and it is where the light lands.
///
/// A thin lens images a parallel bundle wherever its own undeviated chief ray
/// crosses the plane, so [`hint::beam_centre_hero`] is a division rather than
/// a fan of rays. This says the lattice tracer agrees: at the staged pose the
/// chief ray is on the keyhole and so is the light.
#[test]
fn the_heros_beam_lands_where_the_chief_ray_says() {
    let scene = CoveScene::bundled().unwrap();
    let seed = rune::hero_doorstep(&scene, &rune::HeroPose::default());
    let [u, v] = hint::beam_centre_hero(&scene, &seed).expect("the beam reaches the face");
    let c = hint::caustic_of_lens(&scene, &rune::hero_lens(&scene, &seed), hint::RAYS);
    let ([cu, cv], w) = hint::centroid(&c).expect("the lattice caught something");
    println!(
        "the chief ray lands ({u:+.4}, {v:+.4}) m from the keyhole; the lattice's centroid is ({cu:+.4}, {cv:+.4}) m, carrying {w:.4} W"
    );
    assert!(u.hypot(v) < 0.01, "the staged chief ray misses the keyhole by {:.1} mm", u.hypot(v) * 1e3);
    assert!(
        (cu - u).hypot(cv - v) < 0.05,
        "the chief ray and the light disagree by {:.0} mm",
        (cu - u).hypot(cv - v) * 1e3
    );
}

/// **What the soft rim costs.** The lattice's feathered aperture against the
/// hard disc [`rune::lens_projected_area`] states and the photon pass shines
/// on.
///
/// `glass::Shape::rim_weight` grades the outer `glass::RIM_FEATHER` of the
/// lens so the hint's sum is C¹ in the knobs that turn it, and the price is
/// that the hint scores slightly less glass than there is. Two things are in
/// the gap and they pull the same way:
///
/// * the feather itself. The graded band is `1 − 0.95² ≈ 9.8 %` of the
///   aperture's area and a smoothstep passes half of it on average, so about
///   **4.9 %**. This is a *constant* — the feather is in normalised radius,
///   so the fraction does not move when the glass is turned — which is why it
///   cancels out of [`hint::score_lens_of`]'s ratio and shows up only here.
/// * the analytic area's own approximation: `π h² cos θ` plus a rectangular
///   knife edge is a flat disc's silhouette, and a biconvex lens is not flat.
///
/// Measured here: **9.1 %** of the hard aperture at the staged doorstep and
/// **6.0 %** with the wrist turned a further 45° — 7.2 % and 5.0 % before the
/// arm's two-link solve was made exact and moved the staged lens 18 cm and
/// 14°, which is the formula's share moving and not the feather's, as it
/// should. Same bounds; the new lens reads inside them. The 4.9 % is the feather
/// and does not move; the rest is the formula, and it does — which is the
/// whole argument for [`hint::score_lens_of`] dividing one lattice sum by
/// another rather than by an analytic area. An analytic denominator would
/// have put that two per cent of cant-dependence straight into `∂frac/∂cant`
/// as a bias no amount of rays could wash out.
#[test]
fn the_feathered_rim_costs_a_twentieth_of_the_glass() {
    let scene = CoveScene::bundled().unwrap();
    let pose = rune::hero_doorstep(&scene, &rune::HeroPose::default());
    let held = rune::hero_lens(&scene, &pose);
    let k = hint::LensKnobs::of(&held);
    let hard = hint::lens_incident::<f64>(&scene, k) * kosm::light::BANDS_LEN as f64;
    let soft = hint::caustic_of_lens(&scene, &held, hint::RAYS).caught;
    let kept = soft / hard;
    println!(
        "the hard aperture subtends {hard:.6e} sr of lamp, the feathered lattice {soft:.6e}: the rim and the formula spend {:.2}% of the glass",
        100.0 * (1.0 - kept)
    );
    assert!((0.90..=0.96).contains(&kept), "the feathered lattice keeps {kept:.4} of the hard aperture, not about 0.93");

    // the feather is a constant of the glass; the formula is not. Turn the
    // wrist and the gap closes, which is the two per cent of cant-dependence
    // an analytic denominator would have handed the gradient as a bias.
    let turned = rune::HeroPose { cant: pose.cant + std::f64::consts::FRAC_PI_4, ..pose };
    let held = rune::hero_lens(&scene, &turned);
    let k = hint::LensKnobs::of(&held);
    let also = hint::caustic_of_lens(&scene, &held, hint::RAYS).caught / (hint::lens_incident::<f64>(&scene, k) * kosm::light::BANDS_LEN as f64);
    println!("canted a further 45°, the feathered lattice keeps {also:.4} against {kept:.4}");
    assert!((0.93..=0.97).contains(&also), "canted, the lattice keeps {also:.4}, not the feather's own 0.95");
    assert!(also > kept, "the projected-area formula is supposed to be worst edge-on");
}

/// **The live gate.** At the solved hero pose the score the *game* reads —
/// the glass in a stepped body's hand, with the body's own head and trunk in
/// the light — clears `open_frac`.
///
/// Not the arithmetic's score: [`being::Cove::rune_score`] is what the rune
/// thread calls every frame, off a body that has been placed and let settle.
/// The offline solve is only worth anything if the thing the player walks
/// scores the same, and this is where the two meet.
#[test]
fn the_hero_at_the_solved_pose_opens_the_door() -> anyhow::Result<()> {
    let (scene, cove) = super::tests::hero_cove()?;
    let mut cove = cove;
    let solved = rune::HeroPose::solution(&scene);
    assert!(solved.is_solved(), "scene.rs has no hero solution; run `kosm run rune`");
    cove.place_hero(&solved);
    cove.hold_still(2.0);

    let read = cove.hero_pose();
    let arithmetic = rune::score_hero(&scene, &solved, PHOTONS).frac;
    let live = cove.rune_score(&scene, PHOTONS);
    println!(
        "solved ({:+.3}, {:+.3}) at {:+.1}°, holding {:+.1}° up and {:+.1}° canted; the body settled at ({:+.3}, {:+.3}) at {:+.1}° leaning {:.2}°\n\
         the arithmetic scores {arithmetic:.5}, the live body {live:.5}, open_frac {:.3}",
        solved.x, solved.y, solved.yaw.to_degrees(), solved.aim_el.to_degrees(), solved.cant.to_degrees(),
        read.x, read.y, read.yaw.to_degrees(), cove.lean().to_degrees(), scene.open_frac,
    );
    assert!(live >= scene.open_frac, "the live score is {live:.5}, under open_frac {:.3}", scene.open_frac);
    Ok(())
}

/// **The glint.** From the spawn, the hint points up the beach at the door.
///
/// The bare score out there is exactly zero — the beam lands a hundred metres
/// along the cliff — so what is being read is the guided objective, which is
/// the one the solvability sweep climbed from every spawn.
#[test]
fn the_heros_glint_points_at_the_door() {
    let scene = CoveScene::bundled().unwrap();
    let door = scene.door_frame().origin;
    let from = rune::HeroPose { x: scene.spawn_x, y: scene.spawn_y, ..Default::default() }.facing(door);
    assert!(
        hint::score_lens_of(&scene, hint::LensKnobs::of(&rune::hero_lens(&scene, &from)), hint::SWEEP_RAYS) == 0.0,
        "the bare score is flat at the spawn, which is the problem the glint solves"
    );
    let g = hint::guided_gradient_hero(&scene, &from, hint::SWEEP_RAYS).expect("the beam lands somewhere");
    let n = g[0].hypot(g[1]);
    let (ux, uy) = (g[0] / n, g[1] / n);
    let (dx, dy) = (door.x - from.x, door.y - from.y);
    let toward = (ux * dx + uy * dy) / dx.hypot(dy);
    println!(
        "at the spawn ({:+.1}, {:+.1}) the glint steps ({ux:+.3}, {uy:+.3}); the door is ({:+.3}, {:+.3}) away, {:.0}% of the step is toward it",
        from.x, from.y, dx / dx.hypot(dy), dy / dx.hypot(dy), 100.0 * toward
    );
    assert!(uy > 0.0, "the glint points away from the beach: {g:?}");
    assert!(toward > 0.5, "only {:.0}% of the glint's step is toward the door", 100.0 * toward);

    // and the game's own version of it agrees. `game.rs` has a *body* and not
    // a pose, so it differentiates the objective in the two directions the
    // glass moves when its owner walks ([`hint::guided_walk`]) rather than in
    // the hero's knobs. Walking translates the lens rigidly, so the two are
    // the same arrow.
    let lens = rune::hero_lens(&scene, &from);
    let w = hint::guided_walk(&scene, &lens, hint::SWEEP_RAYS).expect("the beam lands somewhere");
    let wn = w[0].hypot(w[1]);
    let agree = (w[0] * ux + w[1] * uy) / wn;
    println!("walking the lens instead steps ({:+.3}, {:+.3}): {:.0}% the same arrow", w[0] / wn, w[1] / wn, 100.0 * agree);
    assert!(agree > 0.9, "the lens's own glint and the hero's disagree by {:.0}°", agree.acos().to_degrees());
}

/// **Test 4 for the hero**, and the one the design says a cove has to pass
/// before it ships: from every spawn in a 6×6 grid, the guided ascent walks
/// the hero to a pose over `open_frac`.
#[test]
#[ignore = "a minute of tracing; `kosm run rune` runs the same sweep and reports it"]
fn the_cove_is_solvable_by_the_hero() {
    let scene = CoveScene::bundled().unwrap();
    let climbs = hint::solvable_hero(&scene, 6, 200, &out_dir().join("cove/solvable-hero.txt")).unwrap();
    let solved = climbs.iter().filter(|c| c.solved).count();
    assert_eq!(solved, climbs.len(), "{} of {} spawns reached open_frac", solved, climbs.len());
}

/// The **lens's own** six knobs, dual against differences, at three lattice
/// densities and three step sizes.
///
/// A probe and not a gate: [`the_heros_hint_matches_central_differences`] is
/// the gate, and it reads the composition. This one reads the expensive half
/// on its own, which is where an aperture that stopped being C¹ would show up
/// first — as a disagreement that does *not* fall when rays are added, and
/// that gets worse as the step shrinks. It reads under a tenth of a per cent
/// in all fifty-four cells; before `glass::Shape::rim_weight` and
/// `light.rs`'s snapped grid it read up to 26 %.
#[test]
#[ignore = "a probe"]
fn probe_lens_gradient() {
    let scene = CoveScene::bundled().unwrap();
    let seed = rune::hero_doorstep(&scene, &rune::HeroPose::default());
    let k = hint::LensKnobs::of(&rune::hero_lens(&scene, &seed));
    for side in [200usize, 400, 800] {
        let rays = side * side;
        println!("=== {side}² rays a band ===");
        for i in 0..6 {
            let d = hint::score_hero_dual(&scene, k.seed(i), rays).dual;
            for h in [1e-4, 1e-3, 1e-2] {
                let (mut lo, mut hi) = (k, k);
                let bump = |kk: &mut hint::LensKnobs<f64>, s: f64| match i {
                    0 => kk.centre.x += s,
                    1 => kk.centre.y += s,
                    2 => kk.centre.z += s,
                    3 => kk.axis.x += s,
                    4 => kk.axis.y += s,
                    _ => kk.axis.z += s,
                };
                bump(&mut lo, -h);
                bump(&mut hi, h);
                let fd = (hint::score_lens_of(&scene, hi, rays) - hint::score_lens_of(&scene, lo, rays)) / (2.0 * h);
                println!(
                    "  lens knob {i}  h {h:.0e}:  dual {d:+10.5}   fd {fd:+10.5}   {:+6.2}%",
                    100.0 * (d - fd) / fd.abs().max(1e-12)
                );
            }
        }
    }
}

