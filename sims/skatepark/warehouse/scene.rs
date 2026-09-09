//! The warehouse's level, in Rust: THPS1's first level, scaled to the K1,
//! in a shed.
//!
//! This was `warehouse.loon`. Same conventions as `skatepark/scene.rs`: Z-up,
//! mm, vcad's cube has a corner at the origin and its cylinder a base at
//! `z = 0` along z.
//!
//! One body per piece, each with a material name. Bodies whose material is
//! `no-collide` (roof, trusses, skylight, glazing, door frame, lamps, the
//! walls above the skirt) are drawn but never baked; everything else is the
//! collision set, and `skatepark.rs` bakes exactly that into `mesh.stl` and
//! the SDF.
//!
//! Two rules the bake imposes on the collision set, both from the same fact —
//! the field's sign comes from the *nearest* triangle's pseudonormal:
//!
//!   * within a body, overlapping solids must be unioned (CSG merges them
//!     into one shell), or left disjoint;
//!   * between bodies, no two shells may share a face. A point just inside
//!     one of a coincident pair is equally near a triangle that calls it
//!     outside. So the floor's top sits `seat_mm` *below* the datum and
//!     everything that stands on it bases at `z = 0`: a hairline slot no
//!     25 mm cell can see.

use kosm::build::{Built, Params, build};

use crate::skatepark::scene::{qp_x, qp_y, transition_run};

pub fn scene(params: &Params) -> anyhow::Result<Built> {
    build(params, |b| {
        // ---- the room ---------------------------------------------------
        let room_x = b.param("room_x_mm", 16000.0); // inner length, x
        let room_y = b.param("room_y_mm", 9000.0); // inner width, y
        let wall_h = b.param("wall_h_mm", 4500.0); // eaves
        let wall_t = b.param("wall_t_mm", 200.0);
        let skirt_h = b.param("skirt_h_mm", 1000.0); // the bottom metre that collides
        let ridge = b.param("ridge_mm", 5500.0);
        let slab_t = b.param("slab_t_mm", 200.0);
        let seat = b.param("seat_mm", 1.0); // the floor's top, below the datum
        let door_w = b.param("door_w_mm", 3000.0);
        let door_h = b.param("door_h_mm", 3000.0);

        // ---- the half pipe (the mini ramp of skatepark/scene.rs) ---------
        let hp_x = b.param("hp_x_mm", -5000.0); // middle of its flat
        let tr_r = b.param("tr_r_mm", 1200.0);
        let lip = b.param("lip_mm", 600.0);
        let width = b.param("width_mm", 3000.0);
        let flat = b.param("flat_mm", 3000.0);
        let deck = b.param("deck_mm", 400.0);
        let coping_r = b.param("coping_r_mm", 30.0);
        let hp_slab = b.param("hp_slab_mm", 40.0);

        // ---- the mezzanine ----------------------------------------------
        let mez_x = b.param("mez_x_mm", 3000.0); // depth from the −x wall
        let mez_y = b.param("mez_y_mm", 3000.0);
        let mez_z = b.param("mez_z_mm", 2600.0); // floor top
        let mez_t = b.param("mez_t_mm", 200.0);
        let mez_rail = b.param("mez_rail_mm", 900.0);
        let mez_rod_r = b.param("mez_rod_r_mm", 30.0);

        // ---- the +x quarter pipes, either side of the door ---------------
        let qp_xx = b.param("qp_x_mm", 6000.0); // where their flat ends
        let qp_w = b.param("qp_w_mm", 2500.0);
        let qp_gap = b.param("qp_gap_mm", 100.0); // clear of the door opening

        // ---- the platform on the +y wall, with a quarter pipe up to it ---
        let plat_x0 = b.param("plat_x0_mm", 2000.0);
        let plat_x1 = b.param("plat_x1_mm", 5000.0);
        let plat_y0 = b.param("plat_y0_mm", 2000.0);
        let plat_h = b.param("plat_h_mm", 800.0);
        let plat_qp_r = b.param("plat_qp_r_mm", 1200.0);
        let top_qp_r = b.param("top_qp_r_mm", 600.0);
        let top_qp_lip = b.param("top_qp_lip_mm", 300.0);

        // ---- the rail ----------------------------------------------------
        let rail_x = b.param("rail_x_mm", 3500.0); // the bend
        let rail_y = b.param("rail_y_mm", 0.0);
        let rail_z = b.param("rail_z_mm", 300.0); // its axis
        let rail_r = b.param("rail_r_mm", 25.0); // Ø 50 mm
        let rail_half = b.param("rail_half_mm", 1000.0); // each half, so 2 m of rail
        let rail_bend = b.param("rail_bend_deg", 7.5); // half of the 15° bend, each way

        // ---- the kickers -------------------------------------------------
        let kick_run = b.param("kick_run_mm", 900.0);
        let kick_rise = b.param("kick_rise_mm", 300.0);
        let kick_w = b.param("kick_w_mm", 1200.0);
        let kick_gap = b.param("kick_gap_mm", 1500.0); // lip to lip
        let kick_t = b.param("kick_t_mm", 400.0);
        let kick_deg = b.param("kick_deg", 18.4349); // atan(rise/run)

        // ---- the ledge and the box piles ---------------------------------
        let ledge_l = b.param("ledge_l_mm", 3000.0);
        let ledge_d = b.param("ledge_d_mm", 400.0);
        let ledge_h = b.param("ledge_h_mm", 350.0);
        let ledge_x = b.param("ledge_x_mm", 0.0);
        let box_mm = b.param("box_mm", 500.0);

        // ---- the roof ----------------------------------------------------
        let roof_t = b.param("roof_t_mm", 100.0);
        let roof_over = b.param("roof_over_mm", 100.0);
        let roof_deg = b.param("roof_deg", 12.0125);
        let sky_half = b.param("sky_half_mm", 750.0);
        let sky_t = b.param("sky_t_mm", 30.0);
        let truss_r = b.param("truss_r_mm", 60.0);

        // ---- the windows and the piers -----------------------------------
        // the Warehouse's shell: bay after bay of floor-to-ceiling steel sash
        // down both long walls, with a brick pier between them. Five bays a
        // wall at a 2.8 m pitch is 14.4 m of the 16 m, so the end piers sit
        // 0.8 m in.
        let win_w = b.param("win_w_mm", 2400.0);
        let win_sill = b.param("win_sill_mm", 1000.0); // the skirt's top
        let win_head = b.param("win_head_mm", 4300.0); // just under the eave
        let win_pitch = b.param("win_pitch_mm", 2800.0); // win_w + pier_w
        let glass_t = b.param("glass_t_mm", 20.0);
        let muntin = b.param("muntin_mm", 40.0); // the sash bars
        let muntin_t = b.param("muntin_t_mm", 60.0); // proud of the glass
        let pier_w = b.param("pier_w_mm", 400.0);
        let pier_proud = b.param("pier_proud_mm", 300.0);
        let pier_foot = b.param("pier_foot_mm", 1000.0); // the bottom metre
        let lamp_r = b.param("lamp_r_mm", 250.0);
        let lamp_h = b.param("lamp_h_mm", 120.0);
        let lamp_z = b.param("lamp_z_mm", 4180.0);

        // ---- the bake ----------------------------------------------------
        // 20 mm cells over 17 × 10 × 4.4 m is 372 MB of f32; 25 mm is 191 MB,
        // and the K1's feet and a 27 mm wheel both still find the transition.
        b.param("sdf_cell_mm", 25.0);
        b.param("sdf_pad_mm", 300.0);

        // ---- the check ---------------------------------------------------
        b.param("check_x_mm", -3500.0);
        b.param("check_y_mm", 0.0);
        b.param("check_z_mm", 40.0); // the half pipe stands on its own slab
        b.param("wheel_r_mm", 27.0);
        b.param("wheel_g", 60.0);
        b.param("drop_mm", 450.0);
        b.param("friction", 0.8);
        b.param("t_end", 4.0);
        b.param("dt_ms", 1.0);

        // ---- the K1 ------------------------------------------------------
        b.param("spawn_x_mm", 0.0); // the middle of the kickers' gap
        b.param("spawn_y_mm", 0.0);
        b.param("spawn_yaw_deg", 0.0); // +x, toward the rail
        b.param("shove_at", 1.0);
        b.param("shove_ns", 8.0);

        // ---- the shell's coordinates -------------------------------------
        let half_x = 0.5 * room_x;
        let half_y = 0.5 * room_y;
        let out_x = half_x + wall_t;
        let out_y = half_y + wall_t;
        let in_x = half_x - seat; // where a collision body may reach a wall
        let in_y = half_y - seat;
        // cutting boxes stay near the thing they cut: vcad meshes a 40 m tool
        // as if it mattered.
        let big = 2000.0;

        // ---- the floor ---------------------------------------------------
        // its top is seat_mm below the datum; everything else bases at z = 0.
        b.body("floor")
            .material("concrete")
            .add(b.box_at([-out_x, out_x], [-out_y, out_y], [-seat - slab_t, -seat]));

        // ---- the half pipe -----------------------------------------------
        let hp_run = transition_run(tr_r, lip);
        let hp_half = 0.5 * flat;
        let hp_side = |s: f64| {
            qp_x(b, s, hp_x + s * hp_half, 0.0, 0.0, width, tr_r, lip, hp_run)
                .union(b.box_at(
                    sorted(hp_x + s * (hp_half + hp_run), hp_x + s * (hp_half + hp_run + deck)),
                    [-0.5 * width, 0.5 * width],
                    [0.0, lip],
                ))
                .union(b.rod_y(coping_r, width).at(hp_x + s * (hp_half + hp_run), 0.0, lip))
        };
        let hp_span = hp_half + hp_run + deck;
        b.body("halfpipe")
            .material("plywood")
            .add(b.box_at([hp_x - hp_span, hp_x + hp_span], [-0.5 * width, 0.5 * width], [0.0, hp_slab]))
            .add(hp_side(1.0).at(0.0, 0.0, hp_slab))
            .add(hp_side(-1.0).at(0.0, 0.0, hp_slab));

        // ---- the mezzanine -----------------------------------------------
        // the secret room over the half pipe, hung off the −x wall — a post
        // to the floor would land in the middle of the ramp's flat.
        let mez_x1 = -in_x + mez_x;
        let mez_rail_x = mez_x1 - 2.0 * mez_rod_r;
        b.body("mezzanine")
            .material("plywood")
            .add(b.box_at([-in_x, mez_x1], [-0.5 * mez_y, 0.5 * mez_y], [mez_z - mez_t, mez_z]))
            .add(b.rod_z(mez_rod_r, mez_rail).at(mez_rail_x, 0.5 * mez_y - mez_rod_r, mez_z + 0.5 * mez_rail))
            .add(b.rod_z(mez_rod_r, mez_rail).at(mez_rail_x, mez_rod_r - 0.5 * mez_y, mez_z + 0.5 * mez_rail))
            .add(b.rod_y(mez_rod_r, mez_y).at(mez_rail_x, 0.0, mez_z + mez_rail));

        // ---- the +x quarter pipes ----------------------------------------
        let qp_yc = 0.5 * door_w + qp_gap + 0.5 * qp_w;
        let qp_lip_x = qp_xx + hp_run;
        let qp = |yc: f64| {
            qp_x(b, 1.0, qp_xx, 0.0, yc, qp_w, tr_r, lip, hp_run)
                .union(b.box_at([qp_lip_x, in_x], [yc - 0.5 * qp_w, yc + 0.5 * qp_w], [0.0, lip]))
                .union(b.rod_y(coping_r, qp_w).at(qp_lip_x, yc, lip))
        };
        b.body("qp_left").material("masonite").add(qp(qp_yc));
        b.body("qp_right").material("masonite").add(qp(-qp_yc));

        // ---- the platform -------------------------------------------------
        let plat_w = plat_x1 - plat_x0;
        let plat_xc = 0.5 * (plat_x0 + plat_x1);
        let plat_run = transition_run(plat_qp_r, plat_h);
        let top_run = transition_run(top_qp_r, top_qp_lip);
        b.body("platform")
            .material("masonite")
            .add(b.box_at([plat_x0, plat_x1], [plat_y0, in_y], [0.0, plat_h]))
            .add(qp_y(b, 1.0, plat_y0 - plat_run, 0.0, plat_xc, plat_w, plat_qp_r, plat_h, plat_run))
            .add(qp_y(b, 1.0, in_y - top_run, plat_h, plat_xc, plat_w, top_qp_r, top_qp_lip, top_run));

        // ---- the rail ------------------------------------------------------
        // two 1 m halves meeting at a 15° bend, on three posts.
        let rail_leg = |s: f64| {
            b.rod_x(rail_r, rail_half)
                .at(s * 0.5 * rail_half, 0.0, 0.0)
                .rotate_z(s * rail_bend)
                .at(rail_x, rail_y, rail_z)
        };
        let rail_post =
            |s: f64| b.rod_z(rail_r, rail_z).at(rail_x + s * rail_half * 0.98, rail_y, 0.5 * rail_z);
        b.body("rail")
            .material("steel")
            .add(rail_leg(1.0))
            .add(rail_leg(-1.0))
            .add(b.rod_z(rail_r, rail_z).at(rail_x, rail_y, 0.5 * rail_z))
            .add(rail_post(1.0))
            .add(rail_post(-1.0));

        // ---- the kickers ---------------------------------------------------
        // a box rotated about y and sunk into the slab: the cut is what makes
        // the wedge, and it leaves the lip a clean vertical face.
        let kick_box_l = (kick_run * kick_run + kick_rise * kick_rise).sqrt();
        let kick_wedge = || {
            b.boxed(kick_box_l, kick_w, kick_t)
                .at(0.5 * kick_box_l, 0.0, -0.5 * kick_t)
                .rotate_y(-kick_deg)
                .difference(b.boxed(3.0 * kick_run, 2.0 * kick_w, big).at(0.0, 0.0, -0.5 * big))
                .at(-kick_run, 0.0, 0.0)
        };
        b.body("kickers")
            .material("plywood")
            .add(kick_wedge().at(-0.5 * kick_gap, 0.0, 0.0))
            .add(kick_wedge().rotate_z(180.0).at(0.5 * kick_gap, 0.0, 0.0));

        // ---- the ledge -------------------------------------------------------
        let ledge_y1 = -(in_y - ledge_d);
        b.body("ledge")
            .material("concrete")
            .add(b.box_at(
                [ledge_x - 0.5 * ledge_l, ledge_x + 0.5 * ledge_l],
                [-in_y, ledge_y1],
                [0.0, ledge_h],
            ))
            .add(b.rod_x(coping_r, ledge_l).at(ledge_x, ledge_y1, ledge_h));

        // ---- the box piles ---------------------------------------------------
        // stacked cubes overlap by a millimetre so each pile unions into one
        // shell.
        let cube_at = |x: f64, y: f64, z: f64| b.boxed(box_mm, box_mm, box_mm).at(x, y, z);
        let pile2 = |x: f64, y: f64| {
            cube_at(x, y, 0.5 * box_mm).union(cube_at(x, y, 1.5 * box_mm - 1.0))
        };
        let pile3 = |x: f64, y: f64| pile2(x, y).union(cube_at(x, y, 2.5 * box_mm - 2.0));
        b.body("boxes")
            .material("cardboard")
            .add(pile3(-6500.0, 3700.0))
            .add(pile2(-2500.0, 3900.0))
            .add(pile2(600.0, 3900.0))
            .add(pile2(-6500.0, -3900.0))
            .add(pile3(5000.0, -3900.0));

        // ---- the walls -------------------------------------------------------
        // a ring: the outer box minus the inner one, cut in two at the skirt's
        // top.
        let ring = |z0: f64, z1: f64| {
            b.box_at([-out_x, out_x], [-out_y, out_y], [z0, z1])
                .difference(b.box_at([-half_x, half_x], [-half_y, half_y], [z0 - 500.0, z1 + 500.0]))
        };
        let doorway =
            || b.box_at([half_x - 1.0, out_x + 1.0], [-0.5 * door_w, 0.5 * door_w], [-100.0, door_h]);
        b.body("wall_skirt").material("brick").add(ring(0.0, skirt_h).difference(doorway()));

        // the window bays, five a wall, cut through both long walls (±y). The
        // bay centres are ±2 and ±1 pitches out from the middle and the middle
        // itself.
        let win_x = |i: f64| i * win_pitch;
        let opening = |y0: f64, y1: f64, i: f64| {
            b.box_at(
                [win_x(i) - 0.5 * win_w, win_x(i) + 0.5 * win_w],
                sorted(y0, y1),
                [win_sill, win_head],
            )
        };
        let wall_openings = |y0: f64, y1: f64| {
            let mut s = opening(y0, y1, -2.0);
            for i in [-1.0, 0.0, 1.0, 2.0] {
                s = s.union(opening(y0, y1, i));
            }
            s
        };
        let openings = || {
            wall_openings(half_y - 1.0, out_y + 1.0).union(wall_openings(-(out_y + 1.0), 1.0 - half_y))
        };
        b.body("walls_upper")
            .material("no-collide brick")
            .add(ring(skirt_h, wall_h).difference(doorway()).difference(openings()));

        // the glazing sits in the middle of the wall's thickness, `yc` =
        // ±(half_y + wall_t/2), so a pane never shares a face with the reveal
        // it fills.
        let pane = |yc: f64, i: f64| {
            b.box_at(
                [win_x(i) - 0.5 * win_w, win_x(i) + 0.5 * win_w],
                [yc - 0.5 * glass_t, yc + 0.5 * glass_t],
                [win_sill, win_head],
            )
        };
        let wall_glass = |yc: f64| {
            let mut s = pane(yc, -2.0);
            for i in [-1.0, 0.0, 1.0, 2.0] {
                s = s.union(pane(yc, i));
            }
            s
        };
        let glass_yc = half_y + 0.5 * wall_t;
        b.body("glass")
            .material("no-collide glass")
            .add(wall_glass(glass_yc))
            .add(wall_glass(-glass_yc));

        // the sash bars: six columns of 400 mm and eight rows per bay, so five
        // uprights and seven rails, all one body a wall — 120 boxes rather
        // than 120 roots the window would have to draw one at a time.
        let win_h = win_head - win_sill;
        let mun_half = 0.5 * muntin;
        let bar_v = |yc: f64, i: f64, j: f64| {
            let x = win_x(i) + j * 400.0;
            b.box_at(
                [x - mun_half, x + mun_half],
                [yc - 0.5 * muntin_t, yc + 0.5 * muntin_t],
                [win_sill, win_head],
            )
        };
        let bar_h = |yc: f64, i: f64, j: f64| {
            let z = win_sill + j * win_h / 8.0;
            b.box_at(
                [win_x(i) - 0.5 * win_w, win_x(i) + 0.5 * win_w],
                [yc - 0.5 * muntin_t, yc + 0.5 * muntin_t],
                [z - mun_half, z + mun_half],
            )
        };
        let sash = |yc: f64, i: f64| {
            let mut s = bar_v(yc, i, -2.0);
            for j in [-1.0, 0.0, 1.0, 2.0] {
                s = s.union(bar_v(yc, i, j));
            }
            for j in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] {
                s = s.union(bar_h(yc, i, j));
            }
            s
        };
        let wall_sash = |yc: f64| {
            let mut s = sash(yc, -2.0);
            for i in [-1.0, 0.0, 1.0, 2.0] {
                s = s.union(sash(yc, i));
            }
            s
        };
        b.body("muntins")
            .material("no-collide steel")
            .add(wall_sash(glass_yc))
            .add(wall_sash(-glass_yc));

        // the piers between the bays, on the inside face of both long walls:
        // six a wall, at the bay boundaries and the two ends.
        let pier = |x: f64, y0: f64, y1: f64, z0: f64, z1: f64| {
            b.box_at([x - 0.5 * pier_w, x + 0.5 * pier_w], sorted(y0, y1), [z0, z1])
        };
        let pier_x = |i: f64| i * win_pitch + 0.5 * (win_w + pier_w);
        let wall_piers = |y0: f64, y1: f64, z0: f64, z1: f64| {
            let mut s = pier(pier_x(-3.0), y0, y1, z0, z1);
            for i in [-2.0, -1.0, 0.0, 1.0, 2.0] {
                s = s.union(pier(pier_x(i), y0, y1, z0, z1));
            }
            s
        };
        let pier_y0 = in_y - pier_proud;
        b.body("piers")
            .material("no-collide brick")
            .add(wall_piers(pier_y0, in_y, 0.0, wall_h))
            .add(wall_piers(-in_y, -pier_y0, 0.0, wall_h));

        // their feet, the bottom metre, which does collide. Not every pier
        // gets one: the ledge runs along the −y wall through x = ±1400 and the
        // platform stands against the +y wall at x = 4200, and two collision
        // shells that share a volume make the field's sign ambiguous there. So
        // those three are left to the drawn body alone. The feet are inset
        // `seat_mm` inside the pier they sit in, so no drawn face is doubled
        // and none reaches the skirt at half_y.
        let (py0, py1) = (pier_y0 + seat, in_y - seat);
        let (ny0, ny1) = (-in_y + seat, -pier_y0 - seat);
        let feet = b.body("pier_feet");
        feet.material("brick");
        for i in [-3.0, -2.0, -1.0, 0.0, 2.0] {
            feet.add(pier(pier_x(i), py0, py1, 0.0, pier_foot));
        }
        for i in [-3.0, -2.0, 1.0, 2.0] {
            feet.add(pier(pier_x(i), ny0, ny1, 0.0, pier_foot));
        }

        b.body("door_frame").material("no-collide steel").add(
            b.box_at(
                [half_x - 1.0, out_x + 1.0],
                [-(0.5 * door_w + 150.0), 0.5 * door_w + 150.0],
                [-200.0, door_h + 150.0],
            )
            .difference(b.box_at(
                [half_x - 2.0, out_x + 2.0],
                [-0.5 * door_w, 0.5 * door_w],
                [-300.0, door_h],
            )),
        );

        // ---- the roof --------------------------------------------------------
        let roof_rise = ridge - wall_h;
        let roof_hyp = (out_y * out_y + roof_rise * roof_rise).sqrt();
        let roof_cos = out_y / roof_hyp;
        let roof_slope = roof_rise / out_y; // dz per unit of plan y
        let roof_len = 2.0 * out_x + 2.0 * roof_over;
        let slab_at = |s: f64, yc: f64, l: f64, t: f64| {
            b.boxed(roof_len, l, t).rotate_x(-s * roof_deg).at(0.0, s * yc, ridge - yc * roof_slope)
        };
        let panel_yc = 0.5 * (sky_half + out_y);
        let panel_l = (out_y - sky_half) / roof_cos;
        b.body("roof")
            .material("no-collide galvanized")
            .add(slab_at(1.0, panel_yc, panel_l, roof_t))
            .add(slab_at(-1.0, panel_yc, panel_l, roof_t));

        let sky_yc = 0.5 * sky_half;
        let sky_l = sky_half / roof_cos;
        b.body("skylight")
            .material("no-collide glass")
            .add(slab_at(1.0, sky_yc, sky_l, sky_t))
            .add(slab_at(-1.0, sky_yc, sky_l, sky_t));

        // ---- the trusses -----------------------------------------------------
        // bar joists, not tubes: a truss is twenty members, and unioning twenty
        // cylinders costs the B-rep kernel far more than unioning twenty boxes.
        let chord = |s: f64| {
            b.boxed(2.0 * truss_r, roof_hyp, 2.0 * truss_r)
                .rotate_x(-s * roof_deg)
                .at(0.0, s * 0.5 * out_y, 0.5 * (wall_h + ridge))
        };
        let truss = |x: f64| {
            b.boxed(2.0 * truss_r, 2.0 * out_y, 2.0 * truss_r)
                .at(0.0, 0.0, wall_h)
                .union(b.boxed(2.0 * truss_r, 2.0 * truss_r, roof_rise).at(
                    0.0,
                    0.0,
                    0.5 * (wall_h + ridge),
                ))
                .union(chord(1.0))
                .union(chord(-1.0))
                .at(x, 0.0, 0.0)
        };
        let trusses = b.body("trusses");
        trusses.material("no-collide steel");
        for x in [-6000.0, -3000.0, 0.0, 3000.0, 6000.0] {
            trusses.add(truss(x));
        }

        // ---- the lamps -------------------------------------------------------
        let lamp = |x: f64| {
            b.rod_z(lamp_r, lamp_h).at(x, 0.0, lamp_z + 0.5 * lamp_h).union(
                b.rod_z(20.0, wall_h - (lamp_z + lamp_h)).at(
                    x,
                    0.0,
                    0.5 * (lamp_z + lamp_h + wall_h),
                ),
            )
        };
        let lamps = b.body("lamps");
        lamps.material("no-collide lamp");
        for x in [-6000.0, -3600.0, -1200.0, 1200.0, 3600.0, 6000.0] {
            lamps.add(lamp(x));
        }
    })
}

/// A span, low end first: a level that puts a piece on the −x side hands its
/// bounds in the order it thinks of them.
fn sorted(a: f64, b: f64) -> [f64; 2] {
    if a <= b { [a, b] } else { [b, a] }
}
