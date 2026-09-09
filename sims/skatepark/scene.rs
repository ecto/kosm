//! The skatepark's level, in Rust.
//!
//! This was `skatepark.loon`. A mini ramp: a flat bottom with a quarterpipe
//! at each end, a deck behind each lip and a coping rod on the edge. Z-up,
//! mm, vcad's conventions (cube corner at the origin, cylinder base at
//! `z = 0` along z). The flat's top is `z = 0` and `x = 0` is the middle of
//! the flat.
//!
//! Sized for a Booster K1 (1.2 m tall), not a person: a 600 mm lip on a
//! 1200 mm radius is a 60° transition, well under vert. Everything is one
//! body so the baked signed distance has one inside.

use kosm::build::{Builder, Built, Params, Shape, build};

pub fn scene(params: &Params) -> anyhow::Result<Built> {
    build(params, |b| {
        let tr_r = b.param("tr_r_mm", 1200.0); // transition radius
        let lip = b.param("lip_mm", 600.0); // lip height above the flat (< tr_r)
        let width = b.param("width_mm", 2400.0); // ramp width, along y
        let flat = b.param("flat_mm", 3000.0); // flat bottom, along x
        let deck_mm = b.param("deck_mm", 600.0); // platform behind each lip
        let coping_r = b.param("coping_r_mm", 30.0);
        let slab_t = b.param("slab_t_mm", 40.0); // slab under everything, top at z = 0
        let second_side = b.param("second_side", 1.0) > 0.5;

        // the bake. the wheels are 27 mm; 20 mm cells would flatten the coping
        b.param("sdf_cell_mm", 10.0);
        b.param("sdf_pad_mm", 300.0);

        // the check: a wheel-sized sphere set on the +x transition, released,
        // rolled on the baked map through the same SDF contact path the K1's
        // feet use. check_x/y/z name the flat end of that transition, so a
        // level with its ramp somewhere other than the origin checks the same.
        b.param("check_x_mm", 1500.0);
        b.param("check_y_mm", 0.0);
        b.param("check_z_mm", 0.0);
        b.param("wheel_r_mm", 27.0);
        b.param("wheel_g", 60.0);
        b.param("drop_mm", 450.0);
        b.param("friction", 0.8);
        b.param("t_end", 4.0);
        b.param("dt_ms", 1.0);

        // the ipse scenario: where the K1 stands, which way it faces, and the
        // shove that sends it at the +x transition.
        b.param("spawn_x_mm", 0.0);
        b.param("spawn_y_mm", 0.0);
        b.param("spawn_yaw_deg", 0.0);
        b.param("shove_at", 1.0);
        b.param("shove_ns", 8.0);

        let half_flat = 0.5 * flat;
        // the lip sits tr_r·sin(θ) past the end of the flat, cos(θ) = 1 − lip/tr_r
        let lip_cos = 1.0 - lip / tr_r;
        let tr_x = tr_r * (1.0 - lip_cos * lip_cos).sqrt();
        let cut_l = 1.2 * width;

        // one quarterpipe on side s (±1): a block from the end of the flat to
        // the lip, minus the cylinder whose axis runs along y one radius above
        // the flat's end.
        let transition = |s: f64| {
            b.boxed(tr_x, width, lip)
                .at(s * (half_flat + 0.5 * tr_x), 0.0, 0.5 * lip)
                .difference(b.rod_y(tr_r, cut_l).at(s * half_flat, 0.0, tr_r))
        };
        let deck = |s: f64| {
            b.boxed(deck_mm, width, lip).at(s * (half_flat + tr_x + 0.5 * deck_mm), 0.0, 0.5 * lip)
        };
        let coping = |s: f64| b.rod_y(coping_r, width).at(s * (half_flat + tr_x), 0.0, lip);
        let side = |s: f64| transition(s).union(deck(s)).union(coping(s));

        let span = 2.0 * (half_flat + tr_x + deck_mm);
        let park = b.body("park");
        park.material("concrete");
        park.add(b.boxed(span, width, slab_t).at(0.0, 0.0, -0.5 * slab_t));
        park.add(side(1.0));
        if second_side {
            park.add(side(-1.0));
        }
    })
}

/// A quarter pipe rising along x from its flat end at `x0`: a block from
/// there to its lip, minus the cylinder whose axis runs along y one radius
/// above that flat end. `s` is the direction it rises in (±1) and `run` is
/// [`transition_run`].
#[allow(clippy::too_many_arguments)]
pub fn qp_x(b: &Builder, s: f64, x0: f64, z0: f64, yc: f64, w: f64, r: f64, lip: f64, run: f64) -> Shape {
    b.box_at(
        [x0 - (1.0 - s) * 0.5 * run, x0 + (1.0 + s) * 0.5 * run],
        [yc - 0.5 * w, yc + 0.5 * w],
        [z0, z0 + lip],
    )
    .difference(b.rod_y(r, 1.4 * w).at(x0, yc, z0 + r))
}

/// The same thing rising along y instead.
#[allow(clippy::too_many_arguments)]
pub fn qp_y(b: &Builder, s: f64, y0: f64, z0: f64, xc: f64, w: f64, r: f64, lip: f64, run: f64) -> Shape {
    b.box_at(
        [xc - 0.5 * w, xc + 0.5 * w],
        [y0 - (1.0 - s) * 0.5 * run, y0 + (1.0 + s) * 0.5 * run],
        [z0, z0 + lip],
    )
    .difference(b.rod_x(r, 1.4 * w).at(xc, y0, z0 + r))
}

/// The x a transition of radius `r` reaches by the time it is `lip` high.
pub fn transition_run(r: f64, lip: f64) -> f64 {
    let c = 1.0 - lip / r;
    r * (1.0 - c * c).sqrt()
}
