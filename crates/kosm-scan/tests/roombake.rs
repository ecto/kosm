//! End-to-end: a synthetic `<journal>.derived` directory — VGGT-shaped
//! `.npy` files in a y-down unit frame with a metric scale in `report.json`
//! — goes through the `roombake` binary and comes out as a loadable,
//! standable map with the floor at `z = 0`, and a one-view "person" gone.

use std::path::Path;
use std::process::Command;

use kosm_scan::npy::write_npy_f32;
use kosm_scan::{FloorFrame, Map};
use phyz_math::{Mat3, Vec3};

const W: usize = 80;
const H: usize = 60;
const FX: f64 = 70.0;
const SCALE: f64 = 2.5;

/// Map→camera rotation for a camera at `yaw` about z, pitched `pitch` down.
fn camera_rot(yaw: f64, pitch: f64) -> Mat3 {
    let fwd = Vec3::new(yaw.cos() * pitch.cos(), yaw.sin() * pitch.cos(), -pitch.sin());
    let right = fwd.cross(Vec3::new(0.0, 0.0, 1.0)).normalize();
    let down = fwd.cross(right).normalize();
    Mat3::new(
        right.x, right.y, right.z, down.x, down.y, down.z, fwd.x, fwd.y, fwd.z,
    )
}

/// z-depth to the floor `z = 0` (or a 0.4 m ball at `mover`), from a camera
/// pose in the map frame; 0 where the ray never lands.
fn render(rot: Mat3, pos: Vec3, mover: Option<Vec3>) -> Vec<f32> {
    let cam_to_map = rot.transpose();
    let mut out = vec![0f32; W * H];
    for v in 0..H {
        for u in 0..W {
            let ray = Vec3::new((u as f64 + 0.5 - W as f64 / 2.0) / FX, (v as f64 + 0.5 - H as f64 / 2.0) / FX, 1.0);
            let dir = cam_to_map * ray;
            let mut t = if dir.z < -1e-6 { -pos.z / dir.z } else { f64::INFINITY };
            if let Some(c) = mover {
                let oc = pos - c;
                let b = oc.dot(dir);
                let disc = b * b - dir.dot(dir) * (oc.dot(oc) - 0.16);
                if disc > 0.0 {
                    let ts = (-b - disc.sqrt()) / dir.dot(dir);
                    if ts > 0.0 && ts < t {
                        t = ts;
                    }
                }
            }
            if t.is_finite() && t < 6.0 {
                out[v * W + u] = t as f32;
            }
        }
    }
    out
}

fn write_derived(dir: &Path) {
    // The unit frame: VGGT-like, floor normal ≈ −y, floor 0.7 m "below" the
    // unit origin, first-camera forward +z.
    let floor = FloorFrame::new(Vec3::new(0.01, -1.0, 0.02), 0.7, Vec3::new(0.0, 0.0, 1.0));
    let o = Vec3::new(0.0, 0.0, floor.d);
    let mut extri = Vec::new();
    let mut intri = Vec::new();
    let mut depth = Vec::new();
    let mut n = 0;
    for k in 0..16 {
        let yaw = k as f64 * std::f64::consts::TAU / 16.0;
        let rot_c = camera_rot(yaw, 0.6);
        let pos_c = Vec3::new(0.0, 0.0, 1.0);
        // One view sees a ball at 1.2 m in front of it, hovering 0.5 m up.
        let mover = (k == 3).then(|| pos_c + Vec3::new(1.2 * yaw.cos(), 1.2 * yaw.sin(), -0.5));
        let d = render(rot_c, pos_c, mover);
        // Express as VGGT would: unit-frame extrinsics, unit depth.
        let rot = rot_c * floor.rot;
        let t = rot_c * (o - pos_c) * (1.0 / SCALE);
        for r in 0..3 {
            let row = rot.row(r);
            extri.extend_from_slice(&[row.x as f32, row.y as f32, row.z as f32, [t.x, t.y, t.z][r] as f32]);
        }
        intri.extend_from_slice(&[FX as f32, 0.0, W as f32 / 2.0, 0.0, FX as f32, H as f32 / 2.0, 0.0, 0.0, 1.0]);
        depth.extend(d.iter().map(|z| z / SCALE as f32));
        n += 1;
    }
    std::fs::create_dir_all(dir).unwrap();
    write_npy_f32(&dir.join("extri.npy"), &[n, 3, 4], &extri).unwrap();
    write_npy_f32(&dir.join("intri.npy"), &[n, 3, 3], &intri).unwrap();
    write_npy_f32(&dir.join("depth.npy"), &[n, H, W], &depth).unwrap();
    write_npy_f32(
        &dir.join("floor.npy"),
        &[4],
        &[floor.normal.x as f32, floor.normal.y as f32, floor.normal.z as f32, floor.d as f32],
    )
    .unwrap();
    std::fs::write(dir.join("report.json"), format!("{{\"scale_m_per_unit\": {SCALE}}}")).unwrap();
}

#[test]
fn derived_directory_to_standable_map() {
    let root = std::env::temp_dir().join(format!("ipse-roombake-{}", std::process::id()));
    let derived = root.join("scan.derived");
    let out = root.join("map");
    write_derived(&derived);

    let status = Command::new(env!("CARGO_BIN_EXE_roombake"))
        .arg(&derived)
        .arg("--out")
        .arg(&out)
        .args(["--cell", "0.04", "--min-views", "2", "--min-depth", "0.2"])
        .status()
        .expect("run roombake");
    assert!(status.success(), "roombake failed");

    let map = Map::load(&out).expect("map loads");
    let sdf = map.standable().expect("has a physics layer");
    assert_eq!(map.manifest.align.as_ref().unwrap().rotate, "floor-frame");

    // The floor is at z = 0 in the map: just above it reads a small positive
    // distance, just below reads negative, and the mover's volume is free.
    let above = sdf.sample(Vec3::new(1.4, 0.0, 0.05)).unwrap();
    assert!(above > 0.0 && above < 0.1, "above floor: {above}");
    let below = sdf.sample(Vec3::new(1.4, 0.0, -0.04)).unwrap();
    assert!(below < 0.0, "below floor: {below}");
    let yaw = 3.0 * std::f64::consts::TAU / 16.0;
    let mover = Vec3::new(1.2 * yaw.cos(), 1.2 * yaw.sin(), 0.5);
    let at_mover = sdf.sample(mover).unwrap();
    assert!(at_mover > 0.04, "the one-view ball should be voted away, sdf {at_mover}");

    // The mesh sits on the floor.
    let tris = kosm_scan::stl::read_binary_stl(&map.mesh_path().unwrap()).unwrap();
    assert!(!tris.is_empty());
    let mut zs: Vec<f32> = tris.iter().flat_map(|t| t.iter().map(|v| v[2])).collect();
    zs.sort_by(|a, b| a.total_cmp(b));
    let median = zs[zs.len() / 2];
    assert!(median.abs() < 0.03, "median vertex z {median}");
    // Nothing near the mover.
    let near_mover = tris
        .iter()
        .filter(|t| {
            let p = Vec3::new(t[0][0] as f64, t[0][1] as f64, t[0][2] as f64);
            (p - mover).norm() < 0.35
        })
        .count();
    assert_eq!(near_mover, 0);

    std::fs::remove_dir_all(&root).ok();
}
