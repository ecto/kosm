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
//! surface.
//!
//! Every root here is [`decorative`](kosm::build::Body::decorative): nothing in
//! the cove ever stands on a convex decomposition. The ground is baked into a
//! signed distance field ([`super::bake`]) that the marble and the being step
//! against, and the door is a hinged rig `being.rs` assembles itself — so
//! deriving colliders for a forty-metre boolean would be minutes of work
//! nobody reads.
//!
//! # The ground is not one solid, and does not have to be
//!
//! It was, once: beach ∪ cliff ∪ rocks ∪ headlands ∪ reef, "because the bake
//! wants one inside". That line cost ten minutes. A mesh union re-classifies
//! every triangle of both operands against every triangle of the other, the
//! accumulated solid grows with each one, and a cornice tessellated along forty
//! metres or a basin cut out of a three-metre sphere is tens of thousands of
//! triangles to classify — so `vcad_eval` sat single-threaded in
//! `mesh::csg::classify` and `kosm run rune --view` never opened a window.
//!
//! The bake does not want one inside. It wants the right **sign**, and the sign
//! it uses is the angle-weighted pseudonormal of the nearest triangle
//! ([`kosm_scan::TriMesh::signed_distance`], Bærentzen & Aanæs), not ray
//! parity. Take any point *outside* the union of closed parts. The nearest
//! surface point over the whole soup cannot be buried: a soup point strictly
//! inside some part B is beaten by the point where the segment from p to it
//! crosses B's boundary, and that crossing is itself either on the union's
//! outer boundary or inside a further part, where the same argument repeats and
//! terminates. So the nearest point is on the union's outer boundary, its
//! triangle's outward normal *is* the union's outward normal there, and both
//! the distance and the sign come out exactly the union's — with no boolean.
//! Which is why [`super::CoveScene::parts`] hands the bake a list of roots and
//! [`crate::skatepark::collision_mesh`] pours them into one soup.
//!
//! Only a point *inside* the ground can read a buried face and take its sign,
//! and the only samples inside the ground are a foot's few millimetres of
//! penetration. So the rule this file keeps is: **a part that sits on another
//! sinks well into it** — two metres of beach under every headland step, four
//! under the cliff, a boulder's whole lower half, 400 mm of sand under the sea
//! stack's base. Nothing that stands on the sand has a face within a foot of
//! the sand's own surface.
//!
//! Two things cannot be sunk away, and both are named where they happen. The
//! first is the crease where two parts *emerge* from one another — a boulder's
//! waterline, the inner edge of a bed's ledge: there a buried face meets the
//! outer surface by construction, and the field inside is soft within about its
//! own distance from that crease. The second is the cliff, whose recess is only
//! `strata_relief` (220 mm) deep, so nothing can hide behind it further than
//! that; the bed loop below says what was measured when it was tried. Neither
//! reaches the cell that resolves the face, which is the cell contact is solved
//! in — 3.6 % of the cliff's face columns read air somewhere in the first
//! 100 mm behind it when the whole cove was one boolean, and 9.4 % when it is
//! fifty roots, all of it inside rock nothing samples, and none of it moving
//! the outermost crossing. `tests.rs` walks into the cliff from eight
//! directions and stands the hero at its foot.
//!
//! What that buys, besides the minutes: which of the two rocks a solid is, is
//! now the root's own declared material rather than a footprint measured back
//! off a merged solid in `render.rs`. The parity rule is unchanged and still
//! lives on one lattice — it is simply stated where the bed is laid.

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
        let headland_step = b.param("headland_step_mm", 1200.0); // how far each step sets back toward the outside
        // How high each step stands over the one below it, and how many. The
        // rise is `strata_h_mm` on purpose: the headlands are cut out of the
        // same beds as the cliff, so a tread lands on a bedding plane and
        // `bed_rock`'s parity rule alternates the two rocks up the stair
        // exactly as it does up the cliff face. Five steps
        // of 1.1 m is 5.5 m of rock, and a metre-tall being cannot put a foot
        // on a riser taller than it is.
        let headland_rise = b.param("headland_rise_mm", 1100.0);
        let headland_steps = b.param("headland_steps", 5.0).max(1.0) as usize;
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
        let aperture_r = b.param("aperture_r_mm", 120.0); // the rune's keyhole: a disc on the door face…
        let aperture_z = b.param("aperture_z_mm", 400.0); // keyhole height above the sand at the door: sunlight travels
        // downward, so it can never land above the being's head
        b.param("open_frac", 0.3); // the score that opens the door
        // …and the disc is now a *bore*, so the rim sits on an edge. The score
        // is read off `rune.rs`'s own analytic door — a box, no hole — so this
        // is geometry the picture has and the puzzle does not: a hundred
        // millimetres into two hundred of stone, which is deep enough that the
        // throat goes black and shallow enough that it never breaks through.
        let bore_d = b.param("keyhole_bore_mm", 100.0);
        // The glyph: three straight grooves in a Y, cut into the door's face
        // above the keyhole. Its lowest point is `glyph_z0_mm` up, which is
        // clear of the rim's outer edge at `aperture_z + aperture_r + rim_w`;
        // nothing about the caustic knows it is there.
        let glyph_w = b.param("glyph_w_mm", 74.0);
        let glyph_d = b.param("glyph_d_mm", 15.0);
        let glyph_z0 = b.param("glyph_z0_mm", 940.0); // above the sill
        let glyph_stem = b.param("glyph_stem_mm", 700.0);
        let glyph_arm = b.param("glyph_arm_mm", 700.0);
        let glyph_arm_deg = b.param("glyph_arm_deg", 40.0);

        // ---- the door's frame -------------------------------------------------
        // A made thing in a natural one: a lintel and two jambs of a darker
        // stone standing proud of the cliff's face, and a threshold laid flat
        // on the sand in front of them.
        //
        // **The frame is decorative and the threshold is flush.** The hero
        // solves the rune standing 1.14 m off the face; a sill that raised the
        // sand there by even a few centimetres would move the lens and the
        // level would have to be re-solved. So the frame is its own root, left
        // out of the collision set by `CoveScene::parts`, and the threshold's
        // top rides the sand plane 60 mm proud — enough not to fight the sand
        // for the same depth, not enough to be a step anywhere the field would
        // have to carry.
        let jamb_w = b.param("jamb_w_mm", 300.0);
        let frame_proud = b.param("frame_proud_mm", 130.0);
        let lintel_h = b.param("lintel_h_mm", 340.0);
        let threshold_out = b.param("threshold_out_mm", 700.0);
        let threshold_proud = b.param("threshold_proud_mm", 60.0);

        // ---- the beds ----------------------------------------------------------
        // The cliff is not a box. It is a stack of beds of two rocks, each one
        // `strata_h_mm` thick, laid on one lattice measured up from the sand at
        // the door; the harder bed stands `strata_relief_mm` proud of the
        // softer, and the whole face leans back `batter_deg` as it rises. The
        // headlands are cut out of the same lattice (see `headland_rise_mm`),
        // which is what makes the cove read as one geology rather than as two
        // pieces of scenery.
        //
        // Which of the two rocks a solid is, is *not* a table: it is the parity
        // of the bed its top face lands in, and `bed_rock` below is the one
        // place that says so.
        let strata_h = b.param("strata_h_mm", 1100.0);
        let strata_relief = b.param("strata_relief_mm", 220.0);
        let batter_deg = b.param("batter_deg", 2.5);
        let cornice_r = b.param("cornice_r_mm", 460.0);

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
        // `rune::hero_merit` spends score on *standoff* once the door is
        // comfortably open, so the answer is the furthest back the hero can
        // stand with the glass still in the sun's line to the keyhole: 1.09 m
        // off the face at frac 0.750. Re-solved when the arm's two-link solve
        // was made exact — the old one left the hand 182 mm off every aim, so
        // the previous answer (80.2° up, 66.5° canted, 1.14 m off) was a lens
        // no arm could hold where the arithmetic said. The wrist is now at its
        // clamp, the grip turned right over, and the lift is not.
        b.param("hero_x_mm", -647.4135);
        b.param("hero_y_mm", 14906.1456); // 1.09 m off the cliff face: a place, not a doorframe
        b.param("hero_yaw_deg", 55.6558);
        b.param("hero_aim_el_deg", 72.9176);
        b.param("hero_aim_az_deg", -28.0000);
        b.param("hero_cant_deg", 180.0000); // the wrist at its clamp: the grip turned over
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
        let cliff_face_y = half_cove - cliff_t; // the face the door is set into
        let door_sill = sand_z(cliff_face_y); // the sand at the door

        // **Which of the two rocks a solid is, is not a table.** It is the
        // parity of the bed its top face lands in, on the one lattice
        // `door_sill + k · strata_h` the cliff's strata and the headland's
        // treads are laid out on: the top is the surface you see, so a
        // stratum's tread, a headland's step and the crown of a boulder all
        // take the colour of the bed they reach into. The harder rock is the
        // odd bed, which is also the one standing `strata_relief` proud — which
        // is what a bedded cliff looks like and why the rule is worth having.
        //
        // Read half a bed *below* the top face, which for a bed of the cliff is
        // its own middle: sampling the top face itself would put the sample on
        // a lattice line, where `(k+1)·h / h` is `k+1` or `k+1 − ε` depending
        // on the last bit, and a cliff whose colours flip on a rounding mode is
        // not a rule but a bug waiting for a different machine.
        let bed_rock = |top_z: f64| {
            let k = ((top_z - 0.5 * strata_h - door_sill) / strata_h).floor() as i64;
            if k.rem_euclid(2) == 0 { "rock" } else { "limestone" }
        };
        // One piece of the cove's geology: its own root, painted by name,
        // decorative, and in the bake's collision set because of the prefix.
        // See [`super::is_ground`] — and the module doc for why there is no
        // union holding these together.
        let ground = |name: &str, rock: &str, shape: kosm::build::Shape| {
            let body = b.body(&format!("{}{name}", super::GROUND));
            body.material(rock).decorative();
            body.add(shape);
        };

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
        // three of them. Each slab overlaps the beach by five metres of x and
        // the step below it by a whole tread, and their inner face is 15 m out — clear of the spawn at
        // −9 m and of the sweep's ±8 m band about the door.
        // Deep enough that every step's underside is buried in the seabed: the
        // stair grew taller when the treads were put on the beds' lattice, so
        // the thickness follows the lift rather than being a constant that
        // used to be big enough.
        // Every slab's underside lands on the same plane two metres below the
        // beach's own, so the only buried face a step has is the part of its
        // tread the step above stands on — and that meets the open tread at the
        // riser, which is a crease and not a seam.
        for step in 0..headland_steps {
            let inner = headland_x + step as f64 * headland_step;
            // out past the cove square, so the sampled volume never ends on a
            // face of the rock
            let outer = half_cove + 2000.0;
            let lift = (step as f64 + 1.0) * headland_rise;
            let headland_t = 4000.0 + lift;
            // A tread is a tilted plane, so the corner that decides which bed
            // the step is cut from is its highest, which is the one at the cliff.
            let rock = bed_rock(sand_z(half_cove) + lift);
            for side in [-1.0, 1.0] {
                let slab = on_sand(
                    b.boxed(outer - inner, beach_l, headland_t),
                    side * 0.5 * (inner + outer),
                    headland_t,
                    lift,
                );
                ground(&format!("headland_{}{step}", if side < 0.0 { "l" } else { "r" }), rock, slab);
            }
        }

        // ---- the reef --------------------------------------------------------------
        // The −y edge, where the seabed ends. Not a wall: a line of boulders of
        // varied size, half buried in the seabed and overlapping each other, so
        // there is no gap a being 700 mm across can walk through and the tops
        // break the surface. The jitter is a fixed function of the index rather
        // than a draw, so the reef is the same reef in every run and in the
        // baked field's hash.
        let reef_y = sand_y0 + 2200.0; // in from the edge, so no boulder hangs off the slab
        let reef_n = ((2.0 * headland_x / reef_gap).round() as usize).max(2);
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
            // A sphere has no faces to bury, only the crease where it comes out
            // of the seabed, so a boulder needs no sinking beyond the half of
            // it that is already under.
            ground(&format!("reef_{k:02}"), bed_rock(z + r), b.sphere(r).at(x, y, z));
        }

        // ---- the cliff, as strata --------------------------------------------------
        // The +y edge of the cove, from below the sand to cliff_h_mm above the
        // sand at its foot. Not one box: a stack of beds on the lattice
        // `door_sill + k·strata_h`, each one the full width of the cove and
        // `cliff_t` deep, the odd beds standing `strata_relief` proud of the
        // even ones, the whole face leaning back at `batter_deg`, and a rounded
        // cornice over the top of it.
        //
        // **The door's recess is kept exactly.** Every bed and the buttress are
        // cut by one opening — the door's width, the door's height, from
        // `strata_relief` *in front of* the nominal face back to `door_t + 10`
        // behind it. In front of the face there is nothing but air and the
        // proud beds, so cutting from there costs nothing and buys a clean
        // reveal; behind it, the cut ends exactly where it always did, so the
        // stone the door hangs on and the sill it stands on are the numbers
        // `render.rs`'s hinge and `mod.rs`'s `door_sill()` already read.
        let cliff_lo = sea_z - 4000.0;
        let cliff_hi = sand_z(half_cove) + cliff_h;
        let batter = batter_deg.to_radians().tan();
        let opening = b.box_at(
            [door_x - 0.5 * door_w, door_x + 0.5 * door_w],
            [cliff_face_y - strata_relief - 20.0, cliff_face_y + door_t + 10.0],
            [door_sill, door_sill + door_h],
        );
        // The bed a height sits in, and the face that bed presents. A bed is
        // proud when its index is odd, which is the same parity `bed_rock`
        // paints it by, so the harder rock is both the darker and the one
        // standing out — which is what a bedded cliff looks like and why the
        // rule is worth having.
        let bed_of = |z: f64| ((z - door_sill) / strata_h).floor();
        let k_lo = bed_of(cliff_lo) as i64;
        let k_hi = bed_of(cliff_hi - 1e-9) as i64;
        let mut top_face = cliff_face_y;
        for k in k_lo..=k_hi {
            let z0 = (door_sill + k as f64 * strata_h).max(cliff_lo);
            let z1 = (door_sill + (k + 1) as f64 * strata_h).min(cliff_hi);
            if z1 - z0 < 1.0 {
                continue;
            }
            let proud = if k.rem_euclid(2) == 1 { strata_relief } else { 0.0 };
            let face = cliff_face_y - proud + batter * (z1 - door_sill).max(0.0);
            if k == k_hi {
                top_face = face;
            }
            // **The beds are stacked, not interlocked, and that was measured.**
            // The obvious move is to sink each bed past its own bedding planes
            // so that no two of them share a plane; the cove will not take it.
            // A recessed bed is the only one that can grow without moving a
            // silhouette, and it grows *behind* the proud bed either side of
            // it — which puts its face, a buried face, `strata_relief` less the
            // batter (172 mm) in from the rock's outer surface, nearer than the
            // ledge it was meant to tidy. Scanning the whole face at 20 mm and
            // reading 300 mm in: with the beds as authored, 9.4 % of the
            // columns read air somewhere in the first 100 mm behind the face;
            // sunk 300 mm, 33.1 %. So the beds stay exactly as the dressing
            // pass laid them, the shared bedding planes stay, and what is left
            // is the crease case the module doc names — bounded by the bed
            // lattice, never nearer the face than the relief, and never in the
            // cell that resolves the face itself.
            // Out to the cove's back edge whatever the face does, so a leaning
            // bed grows into the hill rather than hanging off it. Only a bed
            // the doorway actually crosses is cut — the rest stay primitives,
            // which is a boolean the kernel does not have to do and a solid the
            // instance walk can hand over whole.
            let slab = b.box_at([-0.5 * cove, 0.5 * cove], [face, half_cove], [z0, z1]);
            let slab = if z1 > door_sill && z0 < door_sill + door_h {
                slab.difference(opening.clone())
            } else {
                slab
            };
            ground(&format!("bed_{:02}", k - k_lo), bed_rock(z1), slab);
        }
        // The buttress: plain stone at the nominal face, the width of the
        // doorway and its frame, from under the sand to three beds up — which
        // is just over the lintel, and which lands it on an even bed so that
        // the stone immediately round the door is the same granite the cliff's
        // dark beds are. It is what guarantees a flat plane around the opening
        // whatever the beds in front of it are doing, and it is what the darker
        // lintel and jambs are bedded against.
        //
        // Its face is the nominal one, so it stands a little proud of the
        // recessed beds (the batter, tens of millimetres) and sits behind the
        // proud ones by what is left of `strata_relief` — the same 172 mm the
        // beds hide behind each other by, and for the same unavoidable reason:
        // the relief *is* how deep the recess goes.
        let buttress_w = door_w + 2.0 * jamb_w + 2.0 * 240.0;
        ground(
            "buttress",
            bed_rock(door_sill + 3.0 * strata_h),
            b.box_at(
                [door_x - 0.5 * buttress_w, door_x + 0.5 * buttress_w],
                [cliff_face_y, half_cove],
                [cliff_lo, door_sill + 3.0 * strata_h],
            )
            .difference(opening.clone()),
        );
        // The cornice: a rod along the cliff's top front edge, `cornice_r` from
        // the topmost bed's face and the same from the skyline, so it is
        // tangent to both.
        //
        // **It is the one piece of the geology the field does not get**, and
        // being tangent is exactly why. Inside the top bed it is inscribed in
        // that bed's own corner, which in a soup of parts is a buried surface
        // with *zero* clearance from the outer boundary — the one arrangement
        // the sign rule cannot take, because a sample a millimetre inside the
        // cliff's face up there is a millimetre outside the rod and reads the
        // rod's outward normal. What it adds to the cove's shape is the 90 mm
        // of fillet that bulges out of the recess below the top bed, five
        // metres over the sand and forty metres from anywhere a being can
        // stand. So it is a drawn root and not a `ground_` one: both tiers
        // paint it, nothing walks on it.
        let cornice = b.body("cornice");
        cornice.material(bed_rock(cliff_hi)).decorative();
        cornice.add(b.rod_x(cornice_r, cove).at(0.0, top_face + cornice_r, cliff_hi - cornice_r));

        // ---- the rocks -----------------------------------------------------------
        // Boulders sunk a third of their radius into the sand and one sea
        // stack standing 400 mm into it. They are away from the line the
        // marble test rolls down, which is the beach and nothing else, and the
        // three at the cliff's foot are outside the door's apron — see
        // `dressing::APRON`, which is the same three metres.
        let rock = |name: &str, x: f64, y: f64, r: f64| {
            ground(name, bed_rock(sand_z(y) + 0.65 * r), b.sphere(r).at(x, y, sand_z(y) - 0.35 * r))
        };
        rock("rock_0", 5000.0, -3000.0, 900.0);
        rock("rock_1", -3000.0, 2000.0, 600.0);
        rock("rock_2", 11000.0, -11000.0, 1500.0);
        // The sea stack's base is a disc, and a disc is a face: 400 mm under
        // the sand is more than the 300 the module doc asks of a buried one.
        ground(
            "stack",
            bed_rock(sand_z(6000.0) + 2200.0),
            b.cylinder(700.0, 2600.0).at(-13000.0, 6000.0, sand_z(6000.0) - 400.0),
        );
        // …and the fall of rock at the cliff's own foot, which is where the
        // beds above it went. Big enough to read from the spawn, far enough
        // out along the face that neither is in the apron.
        rock("rock_3", -5600.0, 14300.0, 1500.0);
        rock("rock_4", -7100.0, 15100.0, 1050.0);
        rock("rock_5", 5900.0, 14600.0, 1800.0);

        // ---- the tide pools --------------------------------------------------------
        // Two basins in the sand just above the waterline, each cut with one
        // big sphere so its floor is a dish rather than a bucket, and each
        // ringed with a stone lip. The basins are cut out of the beach itself —
        // they are ground, and a pool the field did not know about would be a
        // pool you walked over — and `render.rs` lays the sea's own material in
        // them as a disc, because the sea's height field stops at the
        // waterline.
        //
        // Two spheres out of a box, one after the other rather than as a union
        // of the two: a box has twelve triangles and the basins do not touch,
        // so this is the cheapest boolean in the file and it is measured in
        // `kosm run rune`'s evaluation line with everything else.
        let pools = super::dressing::pools(&|name: &str, default: f64| b.param(name, default));
        let mut sand = beach;
        for (i, p) in pools.iter().enumerate() {
            sand = sand.difference(b.sphere(p.tool_r).at(p.x, p.y, sand_z(p.y) - p.depth + p.tool_r));
            // The lip: a ring of stone round the rim, its axis on the sand, so
            // half the tube is under and the crease is the waterline of the
            // ring. Its own root, like everything else here.
            let lip_t = 0.16 * p.lip_r;
            ground(&format!("lip_{i}"), bed_rock(sand_z(p.y) + lip_t), b.torus(p.lip_r, lip_t).at(p.x, p.y, sand_z(p.y)));
        }

        // ---- the beach ----------------------------------------------------------
        // The last root, because the pools are cut out of it. `sand` is the
        // library's `dry sand`; the one thing the cove overrides about it is
        // the friction the marble check is stated on, and that override lives
        // in `sim::SAND_FRICTION` and says why.
        ground("beach", "sand", sand);

        // ---- the door -------------------------------------------------------------
        // Granite, and the library's: `BuiltBody::substance` resolves the name
        // and `being.rs` weighs the slab out of what it hands back, so the
        // door's density is stated once, in `kosm::material`, and nowhere here.
        //
        // A made thing, so it is cut: the keyhole is a real bore and there is a
        // rune cut into the face above it. Neither is in `rune.rs`'s optical
        // scene — the score is read off an analytic slab with no hole in it —
        // so this is the picture agreeing with the fiction and changing no
        // number.
        let door = b.body("door");
        door.material(super::materials::DOOR).decorative();
        let bore = b
            .cylinder(aperture_r, bore_d + 20.0)
            // `rotate_x(-90)` sends +z to +y: the bore goes into the face.
            .rotate_x(-90.0)
            .at(door_x, cliff_face_y - 10.0, door_sill + aperture_z);
        // The Y: a stem up the door's centre line and two arms off its top.
        let groove_y = cliff_face_y - 5.0;
        let stem = b.box_at(
            [door_x - 0.5 * glyph_w, door_x + 0.5 * glyph_w],
            [groove_y, groove_y + glyph_d + 5.0],
            [door_sill + glyph_z0, door_sill + glyph_z0 + glyph_stem],
        );
        let arm_top = door_sill + glyph_z0 + glyph_stem;
        let (sa, ca) = glyph_arm_deg.to_radians().sin_cos();
        let mut glyph = stem;
        for side in [-1.0f64, 1.0] {
            let arm = b
                .boxed(glyph_w, glyph_d + 5.0, glyph_arm)
                .rotate_y(side * glyph_arm_deg)
                .at(
                    door_x + side * 0.5 * glyph_arm * sa,
                    groove_y + 0.5 * (glyph_d + 5.0),
                    arm_top + 0.5 * glyph_arm * ca,
                );
            glyph = glyph.union(arm);
        }
        door.add(
            b.boxed(door_w, door_t, door_h)
                .at(door_x, cliff_face_y + 0.5 * door_t, door_sill + 0.5 * door_h)
                .difference(bore)
                .difference(glyph),
        );

        // ---- the door's frame -------------------------------------------------------
        // The dressed stone round the opening: two jambs, a lintel over them,
        // and a threshold laid on the sand. Its own root and its own darker
        // substance, so `render.rs` paints it by name rather than by the
        // `ground` root's footprint rule; decorative, so the field is exactly
        // what it was.
        let frame = b.body("door_frame");
        frame.material("basalt").decorative();
        let frame_y = [cliff_face_y - frame_proud, cliff_face_y];
        let jamb_z = [door_sill, door_sill + door_h + lintel_h];
        for side in [-1.0f64, 1.0] {
            let inner = door_x + side * 0.5 * door_w;
            let outer = inner + side * jamb_w;
            frame.add(b.box_at([inner.min(outer), inner.max(outer)], frame_y, jamb_z));
        }
        frame.add(b.box_at(
            [door_x - 0.5 * door_w - jamb_w, door_x + 0.5 * door_w + jamb_w],
            frame_y,
            [door_sill + door_h, door_sill + door_h + lintel_h],
        ));
        // The threshold, tilted with the beach so it lies on the sand, its top
        // `threshold_proud` above it. Authored the way the beach is: flat, with
        // its top face through the origin, turned, then set down on the sand.
        let threshold_t = 260.0;
        let mid_y = cliff_face_y - 0.5 * threshold_out;
        frame.add(
            b.boxed(door_w + 2.0 * jamb_w, threshold_out * (1.0 + beach_slope * beach_slope).sqrt(), threshold_t)
                .at(0.0, 0.0, -0.5 * threshold_t)
                .rotate_x(beach_deg)
                .at(door_x, mid_y, sand_z(mid_y) + threshold_proud),
        );

        // ---- what the tide left ------------------------------------------------------
        // Pebbles, shells, driftwood, kelp and marram, and the cove's other two
        // inhabitants. All of it decorative, all of it instanced, none of it in
        // the light path — see `super::dressing`.
        let site = super::dressing::Site {
            half: half_cove,
            sea_z,
            slope: beach_slope,
            face: cliff_face_y,
            door_x,
            door_w,
            headland_x,
            headland_step,
            headland_rise,
            headland_steps,
            sand_y0,
            reef_y,
            spawn_x: b.param("spawn_x_mm", -9000.0),
            seed: b.param("scatter_seed", 7.0).max(0.0) as u64,
        };
        super::dressing::scatter(b, &site);
        super::dressing::inhabitants(b, &site, &pools);
    })
}
