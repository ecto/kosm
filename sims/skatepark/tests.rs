//! The skatepark, baked and rolled on: the field is the arc, and a rolling
//! sphere on that field obeys 10/7 · g · Δ.

use crate::skatepark::{self, SkateparkScene};

#[test]
fn the_field_is_the_arc_and_the_wheel_rolls_as_predicted() {
    let scene = SkateparkScene::bundled().expect("level loads");
    let dir = std::env::temp_dir().join("kosm-skatepark-test");
    let baked = skatepark::bake(&scene, &dir).expect("bakes");
    let worst = skatepark::arc_error(&scene, &baked.sdf, 64).expect("arc inside the volume");
    println!("arc error {:.2} mm over {} tris", worst * 1e3, baked.tris);
    assert!(worst < scene.cell * 0.5, "field is {worst} m off the ideal arc");

    let r = skatepark::roll(&scene, &baked.sdf).expect("rolls");
    let rel = r.flat_speed / r.predicted_speed - 1.0;
    println!("flat {:.3} m/s vs {:.3} m/s ({:+.1} %); far apex {:?} vs {:.3}; drift {:.1} mm", r.flat_speed, r.predicted_speed, rel * 100.0, r.far_apex, r.release_height, r.drift * 1e3);
    assert!(rel.abs() < 0.03, "flat speed off by {:.1} %", rel * 100.0);
    let apex = r.far_apex.expect("reached the far wall");
    assert!(apex > 0.85 * r.release_height, "climbed only to {apex} of {}", r.release_height);
    assert!(r.drift < 0.01, "drifted {} m sideways", r.drift);
}

/// **The bake welds each part, and only each part.** Two unit cubes touching
/// at one corner: each is welded to its eight corners (a `-0.0` in one of its
/// triangles included), and the corner they share stays two vertices, one per
/// part, so neither shell's pseudonormals see the other's faces. Half a metre
/// past an edge in a face's plane then reads half a metre, outside.
#[test]
fn the_collision_mesh_welds_each_part_and_not_across_them() {
    use phyz_math::Vec3;
    let cube = |o: Vec3| -> Vec<[Vec3; 3]> {
        let c = |x: f64, y: f64, z: f64| o + Vec3::new(x, y, z);
        let corners = [c(0.0, 0.0, 0.0), c(1.0, 0.0, 0.0), c(1.0, 1.0, 0.0), c(0.0, 1.0, 0.0), c(0.0, 0.0, 1.0), c(1.0, 0.0, 1.0), c(1.0, 1.0, 1.0), c(0.0, 1.0, 1.0)];
        let quads = [[4, 5, 6, 7], [1, 0, 3, 2], [5, 1, 2, 6], [0, 4, 7, 3], [6, 2, 3, 7], [0, 1, 5, 4]];
        quads
            .iter()
            .flat_map(|q| [[corners[q[0]], corners[q[1]], corners[q[2]]], [corners[q[0]], corners[q[2]], corners[q[3]]]])
            .collect()
    };
    let mut a = cube(Vec3::zeros());
    for v in a[3].iter_mut() {
        if v.x == 0.0 {
            v.x = -0.0;
        }
    }
    let part = |name: &str, tris| skatepark::Part { name: name.into(), material: "concrete".into(), collide: true, tris };
    let parts = vec![part("a", a), part("b", cube(Vec3::new(1.0, 1.0, 1.0)))];
    let (tris, mesh) = skatepark::collision_mesh(&parts).expect("has collision geometry");
    assert_eq!(tris.len(), 24);
    assert_eq!(mesh.triangles.len(), 24);
    assert_eq!(mesh.vertices.len(), 16, "eight corners a cube, the shared corner once per part");
    let d = mesh.signed_distance(Vec3::new(0.5, -0.5, 1.0));
    assert!((d - 0.5).abs() < 1e-12, "past cube a's top-front edge, in its top's plane: {d}");
}
