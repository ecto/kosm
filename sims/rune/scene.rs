//! The cove's level, in Rust.
//!
//! This was `levels/cove.loon`. A beach in a cove: sand rising out of the sea
//! to a cliff, a stone door set into the cliff, and a small round aperture in
//! the door that reads the caustic a being of glass throws through itself.
//! Z-up, mm, vcad's conventions (cube corner at the origin, cylinder base at
//! `z = 0` along z, [`Builder::boxed`](kosm::build::Builder::boxed) centred).
//!
//! The level is a `cove_mm` square centred on x = 0. The sea is along −y: the
//! waterline is `y = −cove_mm/2` at `z = sea_z_mm`, and the sand rises toward
//! +y at `beach_slope`, so it is 0.06 · 40 m = 2.4 m up by the time it reaches
//! the cliff. The sea itself is not geometry — it is a rendered height field at
//! `sea_z_mm` — and neither is the aperture: the door's face is solid stone,
//! and the keyhole is the disc on it the score is read over. [`super`] reads
//! this function's knobs and nothing else.
//!
//! Both roots are [`decorative`](kosm::build::Body::decorative): nothing in the
//! cove ever stands on a convex decomposition. The ground is baked into a
//! signed distance field ([`super::bake`]) that the marble and the being step
//! against, and the door is a hinged rig `being.rs` assembles itself — so
//! deriving colliders for a forty-metre boolean would be minutes of work
//! nobody reads.

use kosm::build::{Built, Params, build};

pub fn scene(params: &Params) -> anyhow::Result<Built> {
    build(params, |b| {
        // ---- the cove ------------------------------------------------------
        let cove = b.param("cove_mm", 40000.0); // the level is a cove_mm square, sea along −y
        let beach_slope = b.param("beach_slope", 0.06); // the sand rises from the waterline at this grade
        let sea_z = b.param("sea_z_mm", 0.0); // the waterline, at y = −cove_mm/2
        let cliff_h = b.param("cliff_h_mm", 6000.0); // the back wall, above the sand at its foot
        let cliff_t = b.param("cliff_t_mm", 4000.0); // how deep the cliff is along y; its −y face carries the door

        // ---- the door ------------------------------------------------------
        // A slab of stone standing on the sand in a recess in the cliff face,
        // its face toward −y. Its own root, because it is the one part of the
        // level that moves.
        let door_w = b.param("door_w_mm", 1800.0);
        let door_h = b.param("door_h_mm", 2600.0);
        let door_t = b.param("door_t_mm", 200.0);
        let door_x = b.param("door_x_mm", 100.0); // where along the cliff the door sits (100 centres the solved spot)
        b.param("aperture_r_mm", 120.0); // the rune's keyhole: a disc on the door face, not geometry
        b.param("aperture_z_mm", 400.0); // keyhole height above the sand at the door: sunlight travels
        // downward, so it can never land above the being's head
        b.param("open_frac", 0.3); // the score that opens the door

        // ---- the hint, as light ---------------------------------------------
        // No text ever appears, so what the level tells the player it tells
        // with light. The aperture's rim is a thin emissive ring on the door
        // face: it glows glow_floor + glow_gain · score, so the keyhole is a
        // faint mark from across the beach and unmistakable when the rune is
        // nearly held. The glint is the second half: stand for glint_after_s
        // without the score rising and a spark appears on the sand
        // glint_step_m along the gradient's horizontal part.
        b.param("glow_floor", 0.8); // the rim at a score of zero — findable, not loud
        b.param("glow_gain", 4.0); // at open_frac the rim is several times the sunlit door
        b.param("rim_w_mm", 30.0); // the ring, outside aperture_r_mm
        b.param("glint_after_s", 30.0); // seconds of no progress before the sand sparks
        b.param("glint_step_m", 1.5); // how far along the gradient the spark sits
        b.param("glint_r_mm", 60.0); // the spark itself
        b.param("glint_glow", 6.0); // and its radiance: a spark, not a second keyhole

        // ---- the camera ------------------------------------------------------
        // Over the shoulder out on the sand, and something else at the door:
        // the solved pose stands the being 400 mm off the cliff, and three
        // metres behind it is inside the stone with the being's own body over
        // the keyhole it is solving. So the last two metres blend into a second
        // framing measured from the door's face rather than from the being —
        // round to the shoulder side, above, and aimed at the aperture — and
        // the eye is kept a body's radius and cam_face_clear_mm in front of the
        // face whatever the pose.
        b.param("cam_side_mm", 2000.0); // the shoulder step, out on the sand
        b.param("cam_door_back_mm", 1500.0); // out from the cliff face, not from the being
        b.param("cam_door_up_mm", 1500.0); // above the sand under the eye
        b.param("cam_door_side_mm", 2400.0); // round to the being's right, so the keyhole clears it
        b.param("cam_door_reach_m", 2.0); // the last two metres are where the framing turns
        b.param("cam_face_clear_mm", 500.0); // and the eye never gets nearer the face than this

        // ---- the sun ---------------------------------------------------------
        // A low afternoon sun out over the sea, so what it throws through the
        // being lands on the cliff face. A glass body focuses 1.47 radii from
        // its axis, so the sun must be within 47° of the door's normal (−y is
        // 270°) or the focus falls inside the being's own shadow.
        b.param("sun_az_deg", 250.0); // toward the sun, about z, from +x toward +y
        b.param("sun_el_deg", 22.0); // above the horizon

        // ---- the being --------------------------------------------------------
        // A capsule of glass on one free joint: the marble grown up, and the lens.
        b.param("being_r_mm", 350.0);
        b.param("being_h_mm", 1000.0); // nearer a sphere: a long capsule smears its focus into a line
        b.param("n_d", 1.5168); // the d-line index of N-BK7
        b.param("walk_mps", 1.4); // the speed the walking force is capped at
        b.param("spawn_x_mm", -9000.0); // where you start
        b.param("spawn_y_mm", -6000.0);

        // ---- the solution ------------------------------------------------------
        // Solved, not authored: the pose whose caustic falls in the aperture.
        // These are the solver's own output — `kosm run rune` re-runs it, prints
        // the knobs and writes them to `out/solved/rune.params`; paste them back
        // here to move the default. Zeros mean "not solved yet".
        b.param("solution_x_mm", -47.2224);
        b.param("solution_y_mm", 15600.0);
        b.param("solution_tilt_deg", -10.0);

        // ---- the bake ----------------------------------------------------------
        b.param("sdf_cell_mm", 100.0); // the being's feet are 350 mm across
        b.param("sdf_pad_mm", 300.0); // volume beyond the cove; past it there is no floor

        let half_cove = 0.5 * cove;
        // the beach's tilt about x: `rotate_x(a)` sends +z to (0, −sin a, cos a),
        // so a positive angle lifts the +y end, which is the way the sand runs.
        let beach_deg = beach_slope.atan().to_degrees();
        // the top of the sand at y, which is the plane the being walks on
        let sand_z = |y: f64| sea_z + beach_slope * (y + half_cove);

        // ---- the beach ---------------------------------------------------------
        // A slab modelled flat with its top face through the origin, tilted
        // about x, then lifted so the sand meets the sea at the waterline.
        // Tilting shortens its y footprint by cos a, so it is authored 1/cos a
        // long and ends up exactly the cove square.
        let beach_t = b.param("beach_t_mm", 2000.0);
        let beach_l = cove * (1.0 + beach_slope * beach_slope).sqrt();
        let beach = b
            .boxed(cove, beach_l, beach_t)
            .at(0.0, 0.0, -0.5 * beach_t)
            .rotate_x(beach_deg)
            .at(0.0, 0.0, sand_z(0.0));

        // ---- the cliff ----------------------------------------------------------
        // The +y edge of the cove, from below the sand to cliff_h_mm above the
        // sand at its foot, with the door's recess cut out of its face.
        let cliff_face_y = half_cove - cliff_t; // the face the door is set into
        let door_sill = sand_z(cliff_face_y); // the sand at the door
        let cliff_lo = sea_z - 4000.0;
        let cliff_hi = sand_z(half_cove) + cliff_h;
        let recess = b
            .boxed(door_w, door_t + 20.0, door_h)
            .at(door_x, cliff_face_y + 0.5 * door_t, door_sill + 0.5 * door_h);
        let cliff = b
            .boxed(cove, cliff_t, cliff_hi - cliff_lo)
            .at(0.0, 0.5 * (cliff_face_y + half_cove), 0.5 * (cliff_lo + cliff_hi))
            .difference(recess);

        // ---- the rocks -----------------------------------------------------------
        // Boulders half buried in the sand and one sea stack, unioned into the
        // same solid so the bake has one inside. They are away from the line the
        // marble test rolls down, which is the beach and nothing else.
        let rock = |x: f64, y: f64, r: f64| b.sphere(r).at(x, y, sand_z(y) - 0.35 * r);
        let stack = b.cylinder(700.0, 2600.0).at(-13000.0, 6000.0, sand_z(6000.0) - 400.0);
        let rocks = rock(5000.0, -3000.0, 900.0)
            .union(rock(-3000.0, 2000.0, 600.0))
            .union(rock(11000.0, -11000.0, 1500.0))
            .union(stack);

        // ---- the ground ----------------------------------------------------------
        // One solid, so the baked signed distance has one inside.
        let ground = b.body("ground");
        ground.material("sand").decorative();
        ground.add(beach.union(cliff).union(rocks));

        // ---- the door -------------------------------------------------------------
        let door = b.body("door");
        door.material("stone").decorative();
        door.add(
            b.boxed(door_w, door_t, door_h)
                .at(door_x, cliff_face_y + 0.5 * door_t, door_sill + 0.5 * door_h),
        );
    })
}
