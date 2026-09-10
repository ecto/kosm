//! The marble's level, in Rust.
//!
//! This was `marble.loon` and `marble-cup.loon`. It is the same geometry —
//! vcad units, Z-up, millimetres, degrees, cube corner at the origin,
//! cylinder base at `z = 0` — written through [`kosm::build`], which emits
//! the same `vcad_ir::Document` the loon evaluator used to. The knobs that
//! were `defparam`s are [`kosm::build::Builder::param`] calls: they resolve
//! to `f64` here and are registered on the world, so `built.with(&[..])`
//! rebuilds the whole level around a turned knob.
//!
//! Two cups, as there were two files:
//!
//! - [`scene`] builds the cup as a ring of box segments. A union of convex
//!   primitives is a union of convex colliders, one for one.
//! - [`scene_hollow`] builds the same cup the way a person would model it: a
//!   cylinder with a bore taken out and a slot cut in the uphill side. That
//!   is two `Difference`s, and a difference has no convex collider — so
//!   `kosm::colliders` decomposes it into a ring of convex wedges and checks
//!   that none of them reaches into what was cut away.

use kosm::build::{Builder, Built, Params, Shape, build};

/// The knobs and the track, with the cup as a ring of box segments.
pub fn scene(params: &Params) -> anyhow::Result<Built> {
    build(params, |b| {
        let cup = cup_ring(b, knobs(b));
        track(b, cup);
    })
}

/// The same level, with the cup as a real hollow: a bore and a slot.
pub fn scene_hollow(params: &Params) -> anyhow::Result<Built> {
    build(params, |b| {
        let cup = cup_hollow(b, knobs(b));
        track(b, cup);
    })
}

/// Everything the level declares. Reading them all every build keeps the
/// world's `Param` list the same whichever cup is used, and keeps the ones
/// only the audio, the room or the optics read from vanishing.
struct Knobs {
    plate: [f64; 3],
    wall: [f64; 2],
    cup: [f64; 4],
}

fn knobs(b: &Builder) -> Knobs {
    // the game
    b.param("pitch_deg", 5.0);
    b.param("roll_deg", 0.0);
    b.param("start_x", -100.0);
    b.param("start_y", 25.0);
    b.param("marble_r", 10.0);
    b.param("marble_g", 12.0);
    b.param("marble_nd", 1.5168);
    b.param("t_end", 1.2);

    // the glass samples on the tray, and how they are turned
    b.param("cube_mm", 20.0);
    b.param("pyramid_mm", 25.0);
    b.param("pyramid_h_mm", 20.0);
    b.param("sample_yaw_deg", 25.0);

    // the lamp, and where its shadow should land
    b.param("lamp_x", 60.0);
    b.param("lamp_y", -250.0);
    b.param("lamp_z", 250.0);
    b.param("lamp_r", 25.0);
    b.param("shadow_x", 149.0);
    b.param("shadow_y", -20.0);

    // what things are made of — the sound is these
    b.param("track_density", 1240.0);
    b.param("track_e", 3.5e9);
    b.param("track_nu", 0.36);
    b.param("track_loss", 0.03);
    b.param("marble_density", 2500.0);
    b.param("marble_e", 70.0e9);
    b.param("marble_nu", 0.23);
    b.param("marble_loss", 0.001);

    // the room the tray is in — the sound is these too
    b.param("room_x", 4000.0);
    b.param("room_y", 5000.0);
    b.param("room_z", 2700.0);
    b.param("table_x", 2000.0);
    b.param("table_y", 2000.0);
    b.param("table_z", 750.0);
    b.param("ear_x", 2000.0);
    b.param("ear_y", 1400.0);
    b.param("ear_z", 1200.0);
    b.param("ear_spacing", 170.0);
    b.param("room_absorb_floor", 0.30);
    b.param("room_absorb_ceiling", 0.10);
    b.param("room_absorb_walls", 0.08);
    b.param("room_order", 6.0);
    b.param("track_roughness_mm", 0.2);

    Knobs {
        plate: [b.param("plate_x", 300.0), b.param("plate_y", 200.0), b.param("plate_t", 10.0)],
        wall: [b.param("wall_h", 30.0), b.param("wall_t", 8.0)],
        cup: [
            b.param("cup_x", 90.0),
            b.param("cup_r", 22.0),
            b.param("cup_wall", 3.0),
            b.param("cup_h", 14.0),
        ],
    }
}

/// The plate, its four walls, and whichever cup was handed in. One body:
/// the printed track. The plate's top is `z = 0`.
fn track(b: &Builder, cup: Shape) {
    let k = knobs(b);
    let [px, py, pt] = k.plate;
    let [wh, wt] = k.wall;

    let plate = b.boxed(px, py, pt).at(0.0, 0.0, -0.5 * pt).named("plate");
    let side = |sign: f64| b.boxed(px + 2.0 * wt, wt, wh).at(0.0, sign * 0.5 * (py + wt), 0.5 * wh);
    let end = |sign: f64| b.boxed(wt, py, wh).at(sign * 0.5 * (px + wt), 0.0, 0.5 * wh);

    b.body("track")
        .material("pla")
        .add(plate)
        .add(side(1.0))
        .add(side(-1.0))
        .add(end(1.0))
        .add(end(-1.0))
        .add(cup);
}

/// The cup as a ring of box segments, mouth facing uphill (−x), so it
/// catches. Sixteen segments of 22.5°; eleven are kept, starting at 247.5°
/// and sweeping through +x round to 112.5°. One segment at angle 0,
/// patterned about the cup axis, then the whole ring turned to its start.
fn cup_ring(b: &Builder, k: Knobs) -> Shape {
    let [cup_x, cup_r, cup_wall, cup_h] = k.cup;
    let (seg_n, seg_kept) = (16u32, 11u32);
    let seg_deg = 360.0 / seg_n as f64;
    let seg_len = 1.02 * 2.0 * (cup_r + cup_wall) * 0.19509; // 2 R sin(11.25°)
    let r_mid = cup_r + 0.5 * cup_wall;

    b.boxed(seg_len, cup_wall, cup_h)
        .rotate_z(90.0)
        .at(r_mid, 0.0, 0.5 * cup_h)
        .circular_pattern([0.0; 3], [0.0, 0.0, 1.0], seg_kept, seg_kept as f64 * seg_deg)
        .rotate_z(247.5)
        .at(cup_x, 0.0, 0.0)
        .named("cup")
}

/// The cup as a subtraction. Outer radius `cup_r + cup_wall`, bored to
/// `cup_r`; the bore is 10% taller than the wall so it goes right through and
/// leaves no floor (the plate is the floor). Then the mouth: a box cut off
/// the −x side, leaving the same 112.5° gap the segment ring leaves. The
/// chord at ±56.25° sits at `x = −cos(56.25°)·R = −0.5556 R` and reaches
/// `y = ±sin(56.25°)·R = ±0.8315 R`.
fn cup_hollow(b: &Builder, k: Knobs) -> Shape {
    let [cup_x, cup_r, cup_wall, cup_h] = k.cup;
    let cup_big = cup_r + cup_wall;


    // Mouth first, then the bore. The two cuts commute — same solid either way
    // — but vcad 0.10's Difference does not: with the bore taken first, the
    // outer difference against the mouth box silently does nothing (the tube's
    // −x wall survives to x = 65 mm, and the collider decomposition then fills
    // the mouth and falls back to a solid hull). Cutting the mouth off the
    // plain cylinder first avoids the difference-of-a-difference that trips it.
    let mouth = b
        .boxed(1.4444 * cup_big, 1.6630 * cup_big, 1.2 * cup_h)
        .at(cup_x - 1.2778 * cup_big, 0.0, 0.5 * cup_h);
    b.cylinder(cup_big, cup_h)
        .at(cup_x, 0.0, 0.0)
        .difference(mouth)
        .difference(b.cylinder(cup_r, 1.1 * cup_h).at(cup_x, 0.0, 0.0))
        .named("cup")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ring_cup_is_all_convex_primitives() -> anyhow::Result<()> {
        let built = scene(&Params::default())?;
        let d = &built.bodies[0].colliders;
        assert!(d.warnings.is_empty(), "no decomposition should be needed: {:?}", d.warnings);
        // plate + four walls + eleven cup segments
        assert_eq!(d.colliders.len(), 5 + 11);
        assert_eq!(built.param("cup_r"), Some(22.0));
        Ok(())
    }

    #[test]
    fn the_hollow_cup_is_hollow() -> anyhow::Result<()> {
        let built = scene_hollow(&Params::default())?;
        let d = &built.bodies[0].colliders;
        assert!(d.warnings.is_empty(), "the cup should not have fallen back: {:?}", d.warnings);
        assert!(kosm::colliders::verify_no_intrusion(d) < 0.5e-3);
        Ok(())
    }

    #[test]
    fn a_turned_knob_moves_the_geometry() -> anyhow::Result<()> {
        let built = scene(&Params::default())?;
        let wide = built.with(&[("plate_x", 400.0)])?;
        assert_eq!(wide.param("plate_x"), Some(400.0));
        let half_x = |b: &Built| match &b.bodies[0].colliders.colliders[0].geometry {
            phyz_model::Geometry::Box { half_extents } => half_extents.x,
            other => panic!("the plate should be a box, got {other:?}"),
        };
        assert!((half_x(&built) - 0.150).abs() < 1e-12);
        assert!((half_x(&wide) - 0.200).abs() < 1e-12);
        Ok(())
    }
}
