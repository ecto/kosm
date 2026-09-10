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
//! **The cove is closed, and it is closed with rock.** Past the baked field
//! there is no floor ([`kosm_scan::SdfGrid::sample`] returns nothing outside
//! its volume and the contact producer skips the candidate), so a level whose
//! sand simply stops is a level you can walk out of the bottom of. Rune's rule
//! is that nothing is a trigger and nothing is an invisible wall, so the four
//! edges are four pieces of geometry you can see: the cliff along +y, a
//! stepped **headland** on each of ±x, and along −y the sand carries on under
//! the water as a **seabed** and ends in a **reef** of rock that breaks the
//! surface. Everything is unioned into the one `ground` solid, because the
//! bake wants one inside.
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

        // ---- what closes the cove -------------------------------------------
        // The three edges that are not the cliff. Every one of them is rock the
        // player can see, and every one of them is far enough out that the
        // solvability sweep (x within ±8 m of the door) and the spawn never
        // touch it.
        let seabed = b.param("seabed_mm", 14000.0); // the sand carries on past the waterline this far, at the same grade
        let headland_x = b.param("headland_x_mm", 15000.0); // the inner face of each headland: |x| beyond this is rock
        let headland_step = b.param("headland_step_mm", 1600.0); // how far each step sets back toward the outside
        let headland_rise = b.param("headland_rise_mm", 1600.0); // and how high it stands over the one below it
        let headland_steps = b.param("headland_steps", 3.0).max(1.0) as usize; // three steps is 4.8 m of rock, higher than the being can reach
        let reef_r = b.param("reef_r_mm", 1800.0); // the reef's rocks, before the varied radius
        let reef_gap = b.param("reef_gap_mm", 2000.0); // and how far apart their centres sit along x

        // The sea is water, and water moves. The swell the picture rolls up the
        // beach is a shore break running shoreward at `surf_mps`, and that is
        // the number that decides how deep you can wade: see
        // [`being::Cove::water`](super::being::Cove). Everything else about the
        // water — buoyancy, the drag coefficient — comes from the being's own
        // size and from the density of sea water, and is not a knob.
        b.param("surf_mps", 4.0);

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

        // ---- the window ------------------------------------------------------
        // How fast the level is *played*, which is not how fast it is solved.
        // The simulation steps at 1 ms and hands a snapshot over `fps` times a
        // second, and the window can only present a frame it has been handed:
        // at thirty this knob was the frame rate, whatever the tier drew at.
        // Sixty costs nothing — the solver is four tenths of a millisecond a
        // frame — and it is what the raster tier is capable of.
        b.param("fps", 60.0);
        // How often the live photon map may be retraced while the lens is
        // moving, milliseconds. The map is fifty thousand photons and tens of
        // milliseconds, and it is wanted by two things that can both wait: the
        // caustic on the door and the sand, and the gate's own score (which
        // holds for a second before it opens anything). So it is traced off
        // the frame path, at most this often while the lens is moving, and
        // once more when it stops. See `game.rs::Retrace`.
        b.param("caustic_live_ms", 100.0);

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
        b.param("cam_sea_clear_mm", 400.0); // nor lower than this over the waterline: the picture never dips under

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

        // And the hero's, which is the one the default body is walking. Six
        // knobs rather than three because the lens is on the end of an arm:
        // where it stands, which way it faces, the arm's lift and swing at
        // the shoulder, and the turn of the wrist that cants the glass. Same
        // rule as above — solved, printed, written to
        // `out/solved/rune.params`, and pasted back here by hand.
        //
        // The lift is at its clamp and that is the answer rather than an
        // accident: `rune::hero_merit` spends score on *standoff* once the
        // door is comfortably open, and the only way to stand further back
        // is to hold the glass higher, so the arm's limit is what decides
        // where the hero stands. It lands 1.14 m off the face, which is
        // `hero/mod.rs`'s own chief-ray answer for a lens held as high as a
        // 1.11 m adventurer's arm reaches.
        b.param("hero_x_mm", -610.4487);
        b.param("hero_y_mm", 14858.2961); // 1.14 m off the cliff face: a place, not a doorframe
        b.param("hero_yaw_deg", 58.1072);
        b.param("hero_aim_el_deg", 80.2141); // the arm at its limit, which is what buys the standoff
        b.param("hero_aim_az_deg", -28.0000);
        b.param("hero_cant_deg", 66.5040);
        b.param("hero_mass_kg", 30.0); // a 1.11 m figure of cloth and leather, not a bollard

        // ---- the bake ----------------------------------------------------------
        b.param("sdf_cell_mm", 100.0); // the being's feet are 350 mm across
        b.param("sdf_pad_mm", 300.0); // volume beyond the cove; past it there is no floor

        let half_cove = 0.5 * cove;
        // the beach's tilt about x: `rotate_x(a)` sends +z to (0, −sin a, cos a),
        // so a positive angle lifts the +y end, which is the way the sand runs.
        let beach_deg = beach_slope.atan().to_degrees();
        // the top of the sand at y, which is the plane the being walks on
        let sand_z = |y: f64| sea_z + beach_slope * (y + half_cove);

        // ---- the beach and the seabed ------------------------------------------
        // One slab, modelled flat with its top face through the origin, tilted
        // about x, then lifted so the sand passes through the waterline. It runs
        // from the cliff at +cove/2 out to `seabed_mm` past the waterline, so
        // the same plane is dry sand above `sea_z_mm` and seabed below it — the
        // sea has a floor, which is what lets the being wade instead of fall.
        // Tilting shortens the y footprint by cos a, so it is authored 1/cos a
        // long and ends up exactly that span.
        let beach_t = b.param("beach_t_mm", 2000.0);
        let sand_y0 = -half_cove - seabed; // the seaward edge of the seabed
        let sand_span = half_cove - sand_y0;
        let sand_mid = 0.5 * (sand_y0 + half_cove);
        let beach_l = sand_span * (1.0 + beach_slope * beach_slope).sqrt();
        // `rotate_x` turns about the world x axis through the origin, so a slab
        // authored centred on y = 0 comes out of the turn with its top face
        // through the origin at the beach's grade; translating it to
        // `(0, ym, sand_z(ym))` then puts that face on the sand plane exactly.
        let on_sand = |shape: kosm::build::Shape, x: f64, t: f64, lift: f64| {
            shape.at(0.0, 0.0, -0.5 * t).rotate_x(beach_deg).at(x, sand_mid, sand_z(sand_mid) + lift)
        };
        let beach = on_sand(b.boxed(cove, beach_l, beach_t), 0.0, beach_t, 0.0);

        // ---- the headlands -------------------------------------------------------
        // Rock on both ±x edges, running the whole length of the sand — out of
        // the water at the seaward end, up under the cliff at the other. Each is
        // a stair of slabs cut like the beach, so every tread follows the grade
        // and every riser is a vertical face `headland_rise_mm` tall: at 1.6 m
        // that is higher than the being (1 m) can put a foot, and there are
        // three of them. They overlap the beach slab, so the union has one
        // inside, and their inner face is 15 m out — clear of the spawn at
        // −9 m and of the sweep's ±8 m band about the door.
        let headland_t = 4000.0; // deep enough that the bottom step is buried in the seabed
        let mut headlands: Option<kosm::build::Shape> = None;
        for step in 0..headland_steps {
            let inner = headland_x + step as f64 * headland_step;
            // out past the cove square, so the sampled volume never ends on a
            // face of the rock
            let outer = half_cove + 2000.0;
            let lift = (step as f64 + 1.0) * headland_rise;
            for side in [-1.0, 1.0] {
                let slab = on_sand(
                    b.boxed(outer - inner, beach_l, headland_t),
                    side * 0.5 * (inner + outer),
                    headland_t,
                    lift,
                );
                headlands = Some(match headlands {
                    Some(h) => h.union(slab),
                    None => slab,
                });
            }
        }
        let headlands = headlands.expect("headland_steps is at least one");

        // ---- the reef --------------------------------------------------------------
        // The −y edge, where the seabed ends. Not a wall: a line of boulders of
        // varied size, half buried in the seabed and overlapping each other, so
        // there is no gap a being 700 mm across can walk through and the tops
        // break the surface. The jitter is a fixed function of the index rather
        // than a draw, so the reef is the same reef in every run and in the
        // baked field's hash.
        let reef_y = sand_y0 + 2200.0; // in from the edge, so no boulder hangs off the slab
        let reef_n = ((2.0 * headland_x / reef_gap).round() as usize).max(2);
        let mut reef: Option<kosm::build::Shape> = None;
        for k in 0..=reef_n {
            let t = k as f64;
            let x = -headland_x + 2.0 * headland_x * t / reef_n as f64;
            let r = reef_r * (1.0 + 0.22 * (t * 2.399).sin());
            let y = reef_y + 700.0 * (t * 1.7).sin();
            // A boulder's centre sits on the seabed or above it, never under
            // it, so the shortest of them still stands `reef_r_mm` less its
            // variation — 1.4 m — out of ground the sea is 0.8 m deep over.
            // The being is a metre tall: there is nothing here to step onto.
            let z = sand_z(y) + 0.15 * r * (1.0 + (t * 0.9).sin());
            let rock = b.sphere(r).at(x, y, z);
            reef = Some(match reef {
                Some(s) => s.union(rock),
                None => rock,
            });
        }
        let reef = reef.expect("the reef has at least three boulders");

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
        // `sand` is the library's `dry sand`; the one thing the cove overrides
        // about it is the friction the marble check is stated on, and that
        // override lives in `sim::SAND_FRICTION` and says why.
        let ground = b.body("ground");
        ground.material("sand").decorative();
        ground.add(beach.union(cliff).union(rocks).union(headlands).union(reef));

        // ---- the door -------------------------------------------------------------
        // Granite, and the library's: `BuiltBody::substance` resolves the name
        // and `being.rs` weighs the slab out of what it hands back, so the
        // door's density is stated once, in `kosm::material`, and nowhere here.
        let door = b.body("door");
        door.material(super::materials::DOOR).decorative();
        door.add(
            b.boxed(door_w, door_t, door_h)
                .at(door_x, cliff_face_y + 0.5 * door_t, door_sill + 0.5 * door_h),
        );
    })
}
