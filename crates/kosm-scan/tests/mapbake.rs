//! End-to-end: a synthetic y-up "phone scan" goes through the `mapbake`
//! binary and comes out as a loadable map in the z-up, floor-at-zero frame.

use std::process::Command;

use kosm_scan::{Map, stl};
use phyz_math::Vec3;

/// A slab the way a phone would hand it to us: y-up, with the walkable face
/// at y = 1.3 (the phone started at eye height; the scan frame's origin is
/// wherever the session began, never the floor).
fn y_up_slab() -> Vec<[Vec3; 3]> {
    let lo = Vec3::new(-1.0, 0.8, -1.0);
    let hi = Vec3::new(1.0, 1.3, 1.0);
    let v = |x: f64, y: f64, z: f64| Vec3::new(x, y, z);
    let corners = [
        v(lo.x, lo.y, lo.z),
        v(hi.x, lo.y, lo.z),
        v(hi.x, hi.y, lo.z),
        v(lo.x, hi.y, lo.z),
        v(lo.x, lo.y, hi.z),
        v(hi.x, lo.y, hi.z),
        v(hi.x, hi.y, hi.z),
        v(lo.x, hi.y, hi.z),
    ];
    // Outward winding for a y-up viewer: +y is up.
    let quads = [
        [3usize, 7, 6, 2], // +y (top)
        [0, 1, 5, 4],      // -y
        [1, 2, 6, 5],      // +x
        [0, 4, 7, 3],      // -x
        [4, 5, 6, 7],      // +z
        [0, 3, 2, 1],      // -z
    ];
    let mut tris = Vec::new();
    for q in quads {
        tris.push([corners[q[0]], corners[q[1]], corners[q[2]]]);
        tris.push([corners[q[0]], corners[q[2]], corners[q[3]]]);
    }
    tris
}

#[test]
fn phone_scan_to_loadable_map() {
    let dir = std::env::temp_dir().join(format!("ipse-mapbake-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let scan = dir.join("scan.stl");
    stl::write_binary_stl(&scan, &y_up_slab()).unwrap();

    let out = dir.join("map");
    let status = Command::new(env!("CARGO_BIN_EXE_mapbake"))
        .args([
            scan.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--cell",
            "0.05",
            "--pad",
            "0.3",
            "--up",
            "y",
            "--set-floor",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "mapbake failed");

    let map = Map::load(&out).expect("baked map must load");

    // The floor shift undid the phone's eye-height origin exactly.
    let align = map.manifest.align.as_ref().expect("align recorded");
    assert_eq!(align.rotate, "y-up-to-z-up");
    assert!(
        (align.translate[2] - (-1.3)).abs() < 1e-6,
        "floor shift {} should be -1.3",
        align.translate[2]
    );

    // The map frame contract holds: above the walkable face, distance = z.
    let d = map.standable().unwrap().sample(Vec3::new(0.0, 0.0, 0.05)).expect("inside grid");
    assert!((d - 0.05).abs() < 1e-4, "sdf above floor = {d}, want 0.05");
    // And below it is inside the slab.
    let d = map.standable().unwrap().sample(Vec3::new(0.0, 0.0, -0.05)).expect("inside grid");
    assert!(d < 0.0, "below the floor should be inside the slab, got {d}");

    let _ = std::fs::remove_dir_all(&dir);
}
