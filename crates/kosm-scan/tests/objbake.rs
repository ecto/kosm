//! End-to-end: a y-up "phone scan" of a thing on a table goes through the
//! `objbake` binary and comes out as a loadable object standing on its own
//! base at the origin, with the table gone.

use std::process::Command;

use kosm_scan::{Object, stl};
use phyz_math::Vec3;

/// A box, as triangles, in whatever frame the caller is working in.
fn box_tris(lo: Vec3, hi: Vec3) -> Vec<[Vec3; 3]> {
    let v = |i: usize| {
        Vec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        )
    };
    // Six quads, each wound outward.
    let quads = [
        ([0, 2, 3, 1], false), // -z
        ([4, 5, 7, 6], false), // +z
        ([0, 1, 5, 4], false), // -y
        ([2, 6, 7, 3], false), // +y
        ([0, 4, 6, 2], false), // -x
        ([1, 3, 7, 5], false), // +x
    ];
    let mut tris = Vec::new();
    for (q, _) in quads {
        tris.push([v(q[0]), v(q[1]), v(q[2])]);
        tris.push([v(q[0]), v(q[2]), v(q[3])]);
    }
    tris
}

/// The scan a phone hands over: y is up, the table top is at y = 0.9 (the
/// session started at the scanner's own height, not the floor), the thing is
/// a 20 cm cube standing on the table a metre off to one side, and the table
/// itself is in the scan because it always is.
fn scan_with_a_table() -> Vec<[Vec3; 3]> {
    let mut tris = box_tris(
        Vec3::new(-1.5, 0.85, -1.5),
        Vec3::new(1.5, 0.90, 1.5),
    );
    tris.extend(box_tris(
        Vec3::new(0.90, 0.90, -0.10),
        Vec3::new(1.10, 1.10, 0.10),
    ));
    tris
}

/// A splat of the same scene: a handful of gaussians on the cube and a
/// scattering across the table, in the same y-up frame.
fn write_scan_splat(path: &std::path::Path) {
    let mut points: Vec<[f32; 3]> = Vec::new();
    for i in 0..40 {
        let t = i as f32 / 39.0;
        points.push([0.9 + 0.2 * t, 0.9 + 0.2 * t, -0.1 + 0.2 * t]);
    }
    for i in 0..200 {
        let t = i as f32 / 199.0;
        points.push([-1.5 + 3.0 * t, 0.9, -1.5 + 3.0 * t]);
    }
    let mut bytes = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n\
         property float x\nproperty float y\nproperty float z\n\
         property float f_dc_0\nend_header\n",
        points.len()
    )
    .into_bytes();
    for p in &points {
        for c in [p[0], p[1], p[2], 0.5] {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
    }
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn a_scan_of_a_thing_on_a_table_becomes_an_object_standing_at_the_origin() {
    let dir = std::env::temp_dir().join(format!("ipse-objbake-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let scan = dir.join("scan.stl");
    stl::write_binary_stl(&scan, &scan_with_a_table()).unwrap();
    let splat = dir.join("scan.ply");
    write_scan_splat(&splat);

    let out = dir.join("cube");
    // The crop box is in the up-rotated (z-up) frame, which is what
    // `--inspect` prints: y-up (x, y, z) -> (x, -z, y), so the cube at
    // x 0.9..1.1, y 0.9..1.1, z -0.1..0.1 lands at x 0.9..1.1, y -0.1..0.1,
    // z 0.9..1.1.
    let status = Command::new(env!("CARGO_BIN_EXE_objbake"))
        .args([
            scan.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--up",
            "y",
            "--crop",
            "0.85,-0.15,0.88,1.15,0.15,1.15",
            "--splat",
            splat.to_str().unwrap(),
            "--mass",
            "1.6",
            "--pieces",
            "4",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "objbake failed");

    let obj = Object::load(&out).expect("a baked object must load");
    let p = obj.simulatable().unwrap();

    // ── the frame contract ──
    // 20 cm cube: base at z = 0, centred on its own footprint, 0.2 tall.
    let e = obj.manifest.extent.as_ref().expect("extent recorded");
    assert!(e.lo[2].abs() < 1e-6, "base is not at z = 0: {}", e.lo[2]);
    assert!((e.hi[2] - 0.2).abs() < 1e-6, "height {}", e.hi[2]);
    for axis in 0..2 {
        assert!(
            (e.lo[axis] + e.hi[axis]).abs() < 1e-6,
            "axis {axis} is not centred: {} .. {}",
            e.lo[axis],
            e.hi[axis]
        );
    }
    // And the transform that got it there is recorded for the renderer.
    let align = obj.manifest.align.as_ref().expect("align recorded");
    assert_eq!(align.rotate, "y-up-to-z-up");
    assert!((align.translate[0] - -1.0).abs() < 1e-6, "{:?}", align.translate);
    assert!((align.translate[2] - -0.9).abs() < 1e-6, "{:?}", align.translate);

    // ── the physics ──
    // The table was cropped away, so the volume is the cube's alone.
    assert!((p.volume - 0.008).abs() < 1e-6, "volume {}", p.volume);
    assert_eq!(p.mass_source, "measured");
    assert!((p.mass - 1.6).abs() < 1e-9);
    // Centre of mass halfway up, which is the whole reason the frame puts
    // the base at zero rather than the centroid.
    assert!((p.com[2] - 0.1).abs() < 1e-6, "com {:?}", p.com);
    // A cube is convex, so one piece is exact and the decomposition should
    // not have wasted effort splitting it.
    assert_eq!(p.pieces, 1, "a convex thing came out in {} pieces", p.pieces);
    assert!(p.hull_error < 1e-6, "hull error {}", p.hull_error);
    // m/12 (a² + b²) about every axis, for a cube.
    let want = 1.6 / 12.0 * (0.04 + 0.04);
    for i in 0..3 {
        assert!((p.inertia[i] - want).abs() < 1e-6, "inertia {i} = {}", p.inertia[i]);
    }

    // ── the appearance ──
    let kept = stl::read_splat_positions(&obj.splat_path().unwrap()).unwrap();
    assert_eq!(kept.len(), 40, "the table's gaussians survived the crop");

    // ── and it is ready to be a rigid body ──
    let si = obj.spatial_inertia().unwrap();
    assert!((si.mass - 1.6).abs() < 1e-9);
    assert!((si.com.z - 0.1).abs() < 1e-6);
    assert_eq!(obj.collisions().len(), 1);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_scan_with_no_mass_is_refused() {
    let dir = std::env::temp_dir().join(format!("ipse-objbake-nomass-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let scan = dir.join("scan.stl");
    stl::write_binary_stl(
        &scan,
        &box_tris(Vec3::zeros(), Vec3::new(0.1, 0.1, 0.1)),
    )
    .unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_objbake"))
        .args([scan.to_str().unwrap(), "--out", dir.join("x").to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--mass"), "{err}");
    assert!(
        !dir.join("x").exists(),
        "a refused bake must not leave a directory behind"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_concave_thing_comes_out_in_pieces() {
    let dir = std::env::temp_dir().join(format!("ipse-objbake-l-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // An L: a long low arm and an upright at one end. One hull over this is
    // a slab with a hole nobody can see.
    let mut tris = box_tris(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.40, 0.10, 0.10));
    tris.extend(box_tris(
        Vec3::new(0.30, 0.0, 0.10),
        Vec3::new(0.40, 0.10, 0.40),
    ));
    let scan = dir.join("l.stl");
    stl::write_binary_stl(&scan, &tris).unwrap();

    let out = dir.join("ell");
    let status = Command::new(env!("CARGO_BIN_EXE_objbake"))
        .args([
            scan.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--density",
            "700",
            "--pieces",
            "6",
            "--tolerance",
            "0.05",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let obj = Object::load(&out).unwrap();
    let p = obj.simulatable().unwrap();
    assert!(p.pieces > 1, "the L came out as one lump");
    // Within a few percent of the true 0.0043 m³ — a single hull would be
    // 0.016, nearly four times as much.
    let truth = 0.40 * 0.10 * 0.10 + 0.10 * 0.10 * 0.30;
    assert!(
        (p.volume - truth).abs() / truth < 0.10,
        "volume {} vs {truth}",
        p.volume
    );
    assert_eq!(p.mass_source, "density");
    assert!((p.mass - 700.0 * p.volume).abs() < 1e-9);

    let _ = std::fs::remove_dir_all(&dir);
}
