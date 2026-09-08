//! The court, baked: the field is zero on the hardwood, negative inside the
//! slab and the stanchion, and a rim's radius from the rod under the hoop.
//!
//! The full map is 1.2 GB and ten minutes of exact distances, which is the
//! CLI's job (`kosm --court-bake`). The test bakes the same level at the
//! same cell against the same collision mesh, over a small box around each
//! probe — `bake_sdf_within` is the function the bake calls — and writes one
//! small map through `bake_parts` to check the files and the manifest.

use crate::court::bake::{self, CourtBake};
use crate::skatepark::{self, BakeOpts};
use phyz_math::Vec3;

/// A half-metre box of field around `p`, at the level's own cell.
fn field_around(court: &CourtBake, mesh: &ipse_map::TriMesh, p: Vec3) -> impl Fn(Vec3) -> f64 {
    let half = Vec3::splat(0.25);
    let sdf = skatepark::bake_sdf_within(mesh, court.cell, p - half, p + half);
    move |q: Vec3| sdf.sample(q).unwrap_or_else(|| panic!("{q:?} is outside the sampled box"))
}

#[test]
fn the_field_reads_the_floor_the_rim_and_the_stanchion() {
    let court = CourtBake::bundled().expect("level loads");
    let parts = court.parts().expect("evaluates");
    let (tris, mesh) = skatepark::collision_mesh(&parts).expect("has collision geometry");
    let cell = court.cell;
    let colliding: Vec<&str> = parts.iter().filter(|p| p.collides()).map(|p| p.name.as_str()).collect();
    println!("{} collision tris from {colliding:?}", tris.len());
    assert!(colliding.contains(&"slab") && colliding.contains(&"rim") && colliding.contains(&"pole") && colliding.contains(&"bleachers"));
    assert!(!parts.iter().any(|p| p.material == "ball" || p.material == "paint" && p.collides()));

    // the floor: zero on it, a fifth of a metre up is a fifth of a metre, and
    // the slab's middle is half its thickness inside
    let at = field_around(&court, &mesh, Vec3::zero());
    let floor = at(Vec3::zero());
    assert!(floor.abs() < 0.5 * cell, "centre court reads {:.1} mm", floor * 1e3);
    let up = at(Vec3::new(0.0, 0.0, 0.2));
    assert!((up - 0.2).abs() < cell, "0.2 m over centre court reads {:.3} m", up);
    let inside = at(Vec3::new(0.0, 0.0, -court.court_t / 2.0));
    assert!((inside + court.court_t / 2.0).abs() < 0.5 * cell, "the slab's middle reads {:.1} mm", inside * 1e3);

    // the rim: in its plane at its centre, the rod is rim_r away all round
    let eye = court.rim_eye();
    let at = field_around(&court, &mesh, eye);
    let d_eye = at(eye);
    assert!((d_eye - court.hoop.rim_r).abs() < cell, "the rim's eye reads {:.1} mm for a {:.1} mm radius", d_eye * 1e3, court.hoop.rim_r * 1e3);
    // and a metre under it, the nearest thing is still the rim, not the floor
    let under = eye - Vec3::new(0.0, 0.0, 1.0);
    let d_under = field_around(&court, &mesh, under)(under);
    assert!(d_under > 0.9 && d_under < 1.1, "a metre under the rim reads {:.3} m", d_under);

    // the stanchion: its axis is a pole radius inside
    let axis = Vec3::new(court.pole_x, 0.0, 1.0);
    let at = field_around(&court, &mesh, axis);
    let d_pole = at(axis);
    assert!((d_pole + court.pole_r).abs() < cell, "the pole's axis reads {:.1} mm for a {:.1} mm radius", d_pole * 1e3, court.pole_r * 1e3);
    let d_beside = at(axis + Vec3::new(0.0, 0.2, 0.0));
    assert!((d_beside - (0.2 - court.pole_r)).abs() < cell, "0.2 m beside the pole reads {:.3} m", d_beside);

    println!(
        "floor {:+.2} mm, slab {:+.1} mm, rim eye {:.1} mm, under {:.3} m, pole {:+.1} mm, beside {:.3} m",
        floor * 1e3, inside * 1e3, d_eye * 1e3, d_under, d_pole * 1e3, d_beside
    );
}

#[test]
fn a_small_map_bakes_with_the_courts_extent() {
    let court = CourtBake::bundled().expect("level loads");
    let parts = court.parts().expect("evaluates");
    let dir = std::env::temp_dir().join("kosm-court-bake-test");
    // the court's options, on a metre of centre court: the files and the
    // manifest are the same, the field is a thousandth the size
    let opts = BakeOpts { volume: Some((Vec3::new(-0.5, -0.5, -0.1), Vec3::new(0.5, 0.5, 0.4))), ..court.opts() };
    let baked = skatepark::bake_parts(&court.authored, &parts, opts, &dir).expect("bakes");
    let map = ipse_map::Map::load(&dir).expect("loads back");
    let sdf = map.standable().expect("has a floor");
    assert!(sdf.sample(Vec3::zero()).unwrap().abs() < 0.5 * court.cell);
    assert!(sdf.sample(Vec3::new(0.0, 0.0, 1.0)).is_none(), "the volume is the box, not the gym");
    let extent = map.manifest.extent.expect("the bake writes the court's extent");
    assert_eq!(extent.hi, [court.court_x / 2.0, court.court_y / 2.0, 0.0]);
    assert_eq!(extent.lo, [-court.court_x / 2.0, -court.court_y / 2.0, -court.court_t]);
    assert!(dir.join("parts.json").exists() && dir.join("parts/rim.stl").exists());
    let scenario = bake::scenario_toml(&court, &dir);
    assert!(scenario.contains("at = [0, 0, 0]"), "{scenario}");
    println!("{} tris, {} parts, {}×{}×{}", baked.tris, baked.parts, baked.sdf.nx, baked.sdf.ny, baked.sdf.nz);
}
