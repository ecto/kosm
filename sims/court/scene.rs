//! The court's level, in Rust.
//!
//! This was `court.loon`. A hardwood slab, a regulation hoop at the +x end,
//! basketballs, and the gym they are in. The slab, backboard, rim, bracket
//! and pole are vcad geometry (Z-up, mm, cube corner at origin, cylinder base
//! at `z = 0`), and the balls, the drop, the shot and the recording are
//! knobs. `mod.rs` reads what this builds and nothing else.
//!
//! A size-7 ball is 749–780 mm around (r ≈ 120 mm) and 567–650 g. The NBA's
//! inflation rule is a bounce test: dropped from 6 ft (1.83 m, bottom of the
//! ball), the *top* of the ball must come back to 49–54 in (1.245–1.37 m), so
//! the bottom reaches 1.005–1.13 m: a coefficient of restitution of 0.75–0.79
//! on hardwood. The first apex the run prints is that test, in the sim.

use kosm::build::{Built, Params, build};

pub fn scene(params: &Params) -> anyhow::Result<Built> {
    build(params, |b| {
        // ---- the balls ----------------------------------------------------
        b.param("n_balls", 3.0); // dropped balls, in a line along y at drop_x
        let ball_r = b.param("ball_r_mm", 120.0);
        b.param("ball_g", 620.0);
        b.param("restitution", 0.77);
        b.param("friction", 0.6);
        b.param("drop_mm", 1830.0); // bottom of the ball: the rulebook's 6 ft
        b.param("drop_x_mm", -1500.0);
        b.param("spacing_mm", 500.0);
        b.param("stagger_mm", 120.0);

        // ---- the shot -----------------------------------------------------
        // a free throw: released 4.57 m (15 ft) from the backboard face, from
        // a hand 2.1 m up. speed 0 means no shot.
        b.param("shot_x_mm", -2830.0); // board_x − 4572
        b.param("shot_y_mm", 0.0);
        b.param("shot_z_mm", 2100.0);
        b.param("shot_speed", 7.4);
        b.param("shot_elev_deg", 52.0);
        b.param("shot_azimuth_deg", 0.0);
        b.param("shot_backspin_rps", 2.0);

        // ---- the aim ------------------------------------------------------
        b.param("aim", 1.0);
        b.param("aim_t", -1.0); // the horizon, s; <= 0 means the ballistic time

        // ---- the recording ------------------------------------------------
        b.param("t_end", 4.0);
        b.param("fps", 60.0);
        b.param("dt_ms", 1.0);

        // ---- the court ----------------------------------------------------
        let court_x = b.param("court_x_mm", 14000.0);
        let court_y = b.param("court_y_mm", 5000.0);
        let court_t = b.param("court_t_mm", 40.0); // a slab: its top is z = 0

        // ---- the bake (a map for the K1 in ../ipse; see court/bake.rs) -----
        b.param("sdf_cell_mm", 10.0);
        b.param("sdf_pad_mm", 300.0);
        b.param("sdf_top_mm", 3400.0);
        b.param("spawn_x_mm", 0.0);
        b.param("spawn_y_mm", 0.0);
        b.param("spawn_yaw_deg", 0.0);

        // ---- the hoop (regulation) ----------------------------------------
        let board_x = b.param("board_x_mm", 1742.0); // the backboard's front face
        let board_w = b.param("board_w_mm", 1829.0); // 72 in
        let board_h = b.param("board_h_mm", 1067.0); // 42 in
        let board_t = b.param("board_t_mm", 12.0);
        let board_bottom = b.param("board_bottom_mm", 2896.0); // 9.5 ft
        let rim_z = b.param("rim_z_mm", 3048.0); // 10 ft to the top of the rim
        let rim_r = b.param("rim_r_mm", 228.6); // 18 in inside diameter
        let rim_rod = b.param("rim_rod_mm", 16.0); // 5/8 in steel
        let rim_offset = b.param("rim_offset_mm", 381.0); // rim centre from the face
        let rim_n = b.param("rim_n", 24.0); // segments the ring is built from
        let pole_r = b.param("pole_r_mm", 60.0);
        let pole_x = b.param("pole_x_mm", 2900.0);

        b.body("slab")
            .material("maple")
            .add(b.boxed(court_x, court_y, court_t).at(0.0, 0.0, -0.5 * court_t));

        // the backboard: its front face at board_x, glass, bottom edge at
        // board_bottom.
        b.body("board").material("glass").add(b.boxed(board_t, board_w, board_h).at(
            board_x + 0.5 * board_t,
            0.0,
            board_bottom + 0.5 * board_h,
        ));

        // the rim: a ring of rim_n box segments of rod cross-section, centred
        // on the rod's centreline (radius rim_r + rod/2), top at rim_z.
        let rim_cx = board_x - rim_offset;
        let rod_mid = rim_r + 0.5 * rim_rod;
        let seg_len = 1.02 * 2.0 * rod_mid * 0.130526; // 2 R sin(7.5°)
        b.body("rim").material("rim").add(
            b.boxed(rim_rod, seg_len, rim_rod)
                .at(rod_mid, 0.0, 0.0)
                .circular_pattern([0.0; 3], [0.0, 0.0, 1.0], rim_n as u32, 360.0)
                .at(rim_cx, 0.0, rim_z - 0.5 * rim_rod),
        );

        // the bracket: a flat bar from the back of the ring to the board face
        b.body("bracket").material("steel").add(b.boxed(rim_offset - rim_r, 120.0, 20.0).at(
            board_x - 0.5 * (rim_offset - rim_r),
            0.0,
            rim_z - (rim_rod + 10.0),
        ));

        // the pole and the arm that carries the board
        b.body("pole")
            .material("steel")
            .add(b.cylinder(pole_r, board_bottom + board_h).at(pole_x, 0.0, 0.0));
        b.body("arm").material("steel").add(b.boxed(pole_x - board_x, 100.0, 100.0).at(
            0.5 * (pole_x + board_x),
            0.0,
            board_bottom + 0.5 * board_h,
        ));

        // ---- the net -------------------------------------------------------
        // A mass-spring net hung from the rim's rod: net_rows rings of
        // net_strands nodes below the rim, each node tied to the two nodes
        // half a strand round on the ring below — the classic diamond mesh.
        // A regulation net is 15–18 in long and narrows towards the bottom.
        // It is simulated, not authored geometry, so it has no root here.
        b.param("net_strands", 12.0); // 0 turns the net off
        b.param("net_rows", 7.0);
        b.param("net_length_mm", 400.0);
        b.param("net_bottom_r_mm", 150.0);
        b.param("net_cord_mm", 5.0);
        b.param("net_stiffness", 2000.0); // N/m per cord: nylon, not elastic
        b.param("net_damping", 0.9);
        b.param("net_mass_g", 110.0);
        b.param("net_drag", 2.0); // air on the cords, 1/s

        // ---- the gym -------------------------------------------------------
        // The room is geometry, not a background: four walls, a ceiling, and
        // the floor the slab is let into. There are no textures in the
        // picture, so anything you can see is a body with a material.
        //
        // The camera lives at cam_y_mm = −5.6 m, so the margin has to keep the
        // south wall behind it: gym_margin_mm ≥ 6 m puts it at y = −8.5 m.
        let gym_h = b.param("gym_h_mm", 9000.0);
        let gym_margin = b.param("gym_margin_mm", 6000.0);
        let wall_t = b.param("wall_t_mm", 200.0);
        b.param("light_rows", 2.0);
        b.param("light_cols", 5.0);
        b.param("light_w_mm", 1200.0);
        b.param("light_l_mm", 600.0);
        b.param("light_radiance", 18.0);
        b.param("env_radiance", 0.05);

        // ---- daylight -------------------------------------------------------
        // `sky 0` is a constant grey of env_radiance from infinity and no sun.
        // `sky 1` puts a sky gradient and a sun disc outside the room, and the
        // only way either reaches the floor is the clerestory band — the walls
        // are closed solids, so what you get is a row of sun patches thrown
        // from the window openings onto the floor and the bleachers. The band
        // is glazed; kosm-render carries a shadow ray *through* a thin
        // dielectric, attenuated by `1 − F(cos)`, so NEE sees the sun through
        // the glass and the band converges in a handful of passes.
        b.param("sky", 1.0);
        b.param("sky_zenith", 1.1);
        b.param("sky_horizon", 1.6);
        b.param("sun_elevation_deg", 35.0);
        b.param("sun_azimuth_deg", 250.0);
        // Irradiance, not radiance, so widening the disc softens the shadow
        // without changing the exposure. 6 is about six times what the ceiling
        // panels put on the floor.
        b.param("sun_irradiance", 6.0);
        b.param("sun_angular_radius_deg", 0.27);
        b.param("exposure", 1.0);
        b.param("max_depth", 6.0);
        b.param("denoise", 1.0);

        let court_hx = 0.5 * court_x;
        let court_hy = 0.5 * court_y;
        let gym_hx = court_hx + gym_margin; // inner face of the ±x walls
        let gym_hy = court_hy + gym_margin; // inner face of the ±y walls
        let wall_hz = 0.5 * gym_h;
        let wall_off_x = gym_hx + 0.5 * wall_t;
        let wall_off_y = gym_hy + 0.5 * wall_t;
        let room_x = 2.0 * gym_hx + 2.0 * wall_t;
        let room_y = 2.0 * gym_hy + 2.0 * wall_t;

        // a window band high on the two long walls, above everything a ball
        // can reach. The wall itself is a gap — a course below the sill, a
        // course above the head, and a pier between each pair of openings —
        // and the openings are filled with glass.
        let window_n = b.param("window_n", 8.0);
        let window_w = b.param("window_w_mm", 1800.0);
        let window_h = b.param("window_h_mm", 1400.0);
        let window_sill = b.param("window_sill_mm", 6200.0);
        let window_pitch = 2.0 * gym_hx / window_n;
        let window_x0 = -gym_hx + 0.5 * window_pitch;
        let window_head = window_sill + window_h;
        let pier_w = window_pitch - window_w;

        // the four walls — a ball can hit them, so they are colliders
        let wall_side = || b.boxed(wall_t, room_y, gym_h);
        let long_wall = |y: f64| {
            b.boxed(2.0 * gym_hx, wall_t, window_sill)
                .at(0.0, y, 0.5 * window_sill)
                .union(
                    b.boxed(2.0 * gym_hx, wall_t, gym_h - window_head).at(
                        0.0,
                        y,
                        0.5 * (gym_h + window_head),
                    ),
                )
                // one pier before each opening and one after the last: the end
                // piers run a little past the corner, into the ±x walls, which
                // is where a jamb belongs.
                .union(
                    b.boxed(pier_w, wall_t, window_h)
                        .at(window_x0 - 0.5 * window_pitch, y, window_sill + 0.5 * window_h)
                        .linear_pattern([1.0, 0.0, 0.0], window_n as u32 + 1, window_pitch),
                )
        };
        b.body("walls")
            .material("wall")
            .add(wall_side().at(wall_off_x, 0.0, wall_hz))
            .add(wall_side().at(-wall_off_x, 0.0, wall_hz))
            .add(long_wall(wall_off_y))
            .add(long_wall(-wall_off_y));

        // the glazing: one pane per opening, on the wall's centreline. 20 mm
        // boxes, which is thicker than the glass they stand for — the `window`
        // material is thin-walled, so the daylight arrives undisplaced rather
        // than refracted twice through a slab it is not really 20 mm of.
        let pane_t = b.param("pane_t_mm", 20.0);
        let glazing_wall = |y: f64| {
            b.boxed(window_w, pane_t, window_h)
                .at(window_x0, y, window_sill + 0.5 * window_h)
                .linear_pattern([1.0, 0.0, 0.0], window_n as u32, window_pitch)
        };
        b.body("panes")
            .material("window")
            .decorative()
            .add(glazing_wall(wall_off_y))
            .add(glazing_wall(-wall_off_y));

        b.body("ceiling").material("ceiling").add(b.boxed(room_x, room_y, wall_t).at(
            0.0,
            0.0,
            gym_h + 0.5 * wall_t,
        ));

        // the floor outside the slab: four strips of the slab's own thickness,
        // so the top is exactly z = 0 and a ball rolling off the maple does
        // not step.
        let floor_z = -0.5 * court_t;
        let floor_margin_x = court_hx + 0.5 * gym_margin;
        let floor_margin_y = court_hy + 0.5 * gym_margin;
        let strip_x = || b.boxed(gym_margin, 2.0 * gym_hy, court_t);
        let strip_y = || b.boxed(2.0 * court_hx, gym_margin, court_t);
        b.body("floor")
            .material("floor")
            .add(strip_x().at(floor_margin_x, 0.0, floor_z))
            .add(strip_x().at(-floor_margin_x, 0.0, floor_z))
            .add(strip_y().at(0.0, floor_margin_y, floor_z))
            .add(strip_y().at(0.0, -floor_margin_y, floor_z));

        // the baseline wall, padded: a row of thick pads on the inside of the
        // +x wall, behind the hoop, where a driving player arrives.
        let pad_n = b.param("pad_n", 8.0);
        let pad_w = b.param("pad_w_mm", 1800.0);
        let pad_h = b.param("pad_h_mm", 1800.0);
        let pad_t = b.param("pad_t_mm", 100.0);
        let pad_gap = b.param("pad_gap_mm", 40.0);
        let pad_pitch = pad_w + pad_gap;
        b.body("pads").material("pad").add(
            b.boxed(pad_t, pad_w, pad_h)
                .at(gym_hx - 0.5 * pad_t, -0.5 * pad_pitch * (pad_n - 1.0), 0.5 * pad_h)
                .linear_pattern([0.0, 1.0, 0.0], pad_n as u32, pad_pitch),
        );

        // a small bleacher along the +y side: rows stepping up towards the
        // wall. Real benches — they collide, you can stand on them.
        let rows = b.param("bleacher_rows", 5.0);
        let rise = b.param("bleacher_rise_mm", 400.0);
        let run = b.param("bleacher_run_mm", 800.0);
        let bleacher_len = b.param("bleacher_len_mm", 14000.0);
        let bleacher_pitch = (run * run + rise * rise).sqrt();
        let bleacher_y0 = gym_hy - rows * run + 0.5 * run;
        b.body("bleachers").material("oak").add(
            b.boxed(bleacher_len, run, rise)
                .at(0.0, bleacher_y0, 0.5 * rise)
                .linear_pattern([0.0, run, rise], rows as u32, bleacher_pitch),
        );

        // ---- the markings ---------------------------------------------------
        // Painted lines are thin solids standing on the slab (z = 0 to
        // paint_t_mm), not a texture. Everything is measured off the hoop
        // knobs, so moving the board moves the whole court with it and it
        // stays regulation.
        //
        // Arcs are patterns of short bars, the way the rim is a pattern of rod
        // segments. A bar is cut to the *arc* length of its step rather than
        // the chord, which is long by step²/24 (0.16% at 11°) and is taken up
        // by the 3% overlap the bars already carry.
        let line_w = b.param("line_w_mm", 50.0);
        let paint_t = b.param("paint_t_mm", 1.0);
        let key_t = b.param("key_t_mm", 0.5);
        let baseline_off = b.param("baseline_off_mm", 1219.0); // 4 ft
        let lane_w = b.param("lane_w_mm", 4877.0); // 16 ft
        let lane_len = b.param("lane_len_mm", 5791.0); // 19 ft
        let ft_circle_r = b.param("ft_circle_r_mm", 1829.0); // 6 ft
        let ft_circle_n = b.param("ft_circle_n", 32.0);
        let three_r = b.param("three_r_mm", 7239.0); // 23 ft 9 in
        let three_n = b.param("three_n", 24.0);
        // The corner straights begin at |y| = 6706 mm (22 ft) from the basket,
        // which is off this 5 m slab entirely, so the arc is swept only as far
        // as the slab is wide.
        let three_sweep = b.param("three_sweep_deg", 38.0);

        let baseline_x = board_x + baseline_off;
        let ft_x = baseline_x - lane_len;
        let lane_mid_x = 0.5 * (ft_x + baseline_x);
        let paint_z = 0.5 * paint_t;
        let bar = |x: f64, y: f64, lx: f64, ly: f64| b.boxed(lx, ly, paint_t).at(x, y, paint_z);
        let arc_chord = |r: f64, step_deg: f64| 1.03 * r * step_deg.to_radians();

        let ft_step = 360.0 / ft_circle_n;
        let ft_circle = bar(ft_circle_r, 0.0, line_w, arc_chord(ft_circle_r, ft_step))
            .circular_pattern([0.0; 3], [0.0, 0.0, 1.0], ft_circle_n as u32, 360.0)
            .at(ft_x, 0.0, 0.0);

        let three_step = three_sweep / three_n;
        let three_arc = bar(-three_r, 0.0, line_w, arc_chord(three_r, three_step))
            .rotate_z(-0.5 * three_step * (three_n - 1.0))
            .circular_pattern([0.0; 3], [0.0, 0.0, 1.0], three_n as u32, three_sweep)
            .at(rim_cx, 0.0, 0.0);

        b.body("markings")
            .material("paint")
            .decorative()
            .add(bar(baseline_x, 0.0, line_w, court_y))
            .add(bar(lane_mid_x, 0.5 * lane_w, lane_len, line_w))
            .add(bar(lane_mid_x, -0.5 * lane_w, lane_len, line_w))
            .add(bar(ft_x, 0.0, line_w, lane_w))
            .add(ft_circle)
            .add(three_arc);

        b.body("key-fill")
            .material("key")
            .decorative()
            .add(b.boxed(lane_len, lane_w, key_t).at(lane_mid_x, 0.0, 0.5 * key_t));

        // ---- the ball --------------------------------------------------------
        // The ball's appearance, at the origin: the renderer places one copy at
        // each ball's pose. Two bodies, because a body carries one material —
        // the rubber sphere, and the seams as four thin rings straddling its
        // surface. Rings rather than cut channels: a `difference` on a sphere
        // is a convex decomposition every time the colliders are derived, and
        // four unioned tori cost nothing.
        //
        // The classic eight-panel pattern: one ring round the equator, one
        // round a meridian, and two parallel to that meridian at ±0.62 r —
        // whose radius on the sphere is r·√(1 − 0.62²).
        let seam = b.param("ball_seam_mm", 6.0);
        let seam_off_frac = b.param("ball_seam_off", 0.62);
        let seam_off = ball_r * seam_off_frac;
        let seam_r = ball_r * (1.0 - seam_off_frac * seam_off_frac).sqrt();
        let seam_ring = || b.torus(seam_r, seam).rotate_x(90.0);
        b.body("ball").material("ball").decorative().add(b.sphere(ball_r));
        b.body("seams")
            .material("ball-seams")
            .decorative()
            .add(b.torus(ball_r, seam))
            .add(b.torus(ball_r, seam).rotate_x(90.0))
            .add(seam_ring().at(0.0, seam_off, 0.0))
            .add(seam_ring().at(0.0, -seam_off, 0.0));

        // ---- the recording's picture ------------------------------------------
        b.param("render_w", 960.0);
        b.param("render_h", 540.0);
        b.param("render_spp", 64.0);
        b.param("still_w", 1920.0);
        b.param("still_h", 1080.0);
        b.param("still_spp", 512.0);
        b.param("still_t", 0.95);
        b.param("cam_x_mm", -800.0);
        b.param("cam_y_mm", -5600.0);
        b.param("cam_z_mm", 1900.0);
        b.param("cam_at_x_mm", 900.0);
        b.param("cam_at_y_mm", 300.0);
        b.param("cam_at_z_mm", 1900.0);
        b.param("cam_vfov_deg", 40.0);
        b.param("cam_aperture_mm", 0.0); // iris radius; 0 is a pinhole
        b.param("cam_focus_mm", 0.0); // the sharp plane; 0 focuses on cam_at
        b.param("shutter", 0.0); // exposure as a fraction of a frame
        b.param("shutter_steps", 4.0);
    })
}
