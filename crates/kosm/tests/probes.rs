//! The probe volume, end to end: a furnace, an overhang, a cell boundary and
//! a file.
//!
//! `cargo test -p kosm --release --test probes`. Release, because two of
//! these bake a small volume with a path tracer behind every ray.
//!
//! The furnace is the calibration: a sky of unit radiance and nothing else
//! puts π on every surface whichever way it faces, so a bake that comes back
//! with anything else has its projection, its weights or its convolution
//! wrong. The overhang is the *shadow* claim — a probe under a slab sees less
//! sun than one in the open — and the other two are the file and the
//! interpolation, which need no light at all.

use std::sync::Arc;

use kosm::light::probes::{self, BakeSpec, ProbeVolume, VolumeSpec};
use kosm::material::{self, band_to_rgb};
use kosm_render::pathtrace::{Environment, Object, Pbr, Scene, Sun};
use kosm_render::{Bvh, TriMesh};
use kosm_render::math::{Point3, Vec3};

/// A closed axis-aligned box as triangles, so the inside test has a solid to
/// find and the sun has something to be blocked by.
fn slab(min: [f64; 3], max: [f64; 3]) -> TriMesh {
    let p = |i: usize| {
        Point3::new(
            if i & 1 == 0 { min[0] } else { max[0] },
            if i & 2 == 0 { min[1] } else { max[1] },
            if i & 4 == 0 { min[2] } else { max[2] },
        )
    };
    let positions: Vec<Point3> = (0..8).map(p).collect();
    // Every face wound so its normal points out of the box.
    let idx: Vec<u32> = vec![
        0, 2, 3, 0, 3, 1, // z min, seen from below
        4, 5, 7, 4, 7, 6, // z max
        0, 1, 5, 0, 5, 4, // y min
        2, 6, 7, 2, 7, 3, // y max
        0, 4, 6, 0, 6, 2, // x min
        1, 3, 7, 1, 7, 5, // x max
    ];
    TriMesh::new(positions, Vec::new(), &idx)
}

fn object(mesh: TriMesh, pbr: Pbr) -> Object<TriMesh> {
    Object::new(Arc::new(Bvh::build(mesh)), pbr)
}

/// A sky of unit radiance, no sun, no geometry.
fn furnace() -> Scene<TriMesh> {
    Scene {
        objects: Vec::new(),
        lights: Vec::new(),
        env: Environment::constant([1.0, 1.0, 1.0]),
        sun: None,
        ground: None,
        splats: None,
    }
}

#[test]
fn the_white_furnace_puts_pi_on_every_probe() {
    let volume = VolumeSpec::over([-1.0, -1.0, 0.0], [1.0, 1.0, 1.0], 1.0);
    let v = probes::bake(&furnace(), &volume, &[[0.0, 0.0, 1.0]], true, 256, 7);
    assert_eq!(v.dims, [3, 3, 2]);
    assert_eq!(v.count(), 18);
    let pi = std::f64::consts::PI;
    let mut worst = 0.0f64;
    for iz in 0..v.dims[2] {
        for iy in 0..v.dims[1] {
            for ix in 0..v.dims[0] {
                assert!(!v.is_inside(ix, iy, iz), "nothing to be inside of");
                let p = v.point(ix, iy, iz);
                for n in [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.6, -0.8, 0.0]] {
                    let e = v.sample(0.0, p, n);
                    for b in 0..probes::BANDS {
                        worst = worst.max((e[b] as f64 / pi - 1.0).abs());
                    }
                }
            }
        }
    }
    assert!(worst < 0.03, "the furnace is off by {:.2} %, not the 3 % a 256-ray bake is allowed", worst * 100.0);
    println!("furnace: worst probe is {:.2} % from pi at 256 rays", worst * 100.0);

    // The sky term is the same number by the same projection: unit radiance
    // over the whole sphere.
    let sky = &v.sky;
    let e0 = std::f64::consts::PI * sky[0][0] as f64 * 0.282_094_79;
    assert!((e0 / pi - 1.0).abs() < 0.03, "the sky term is {e0}");
}

#[test]
fn a_probe_under_an_overhang_sees_less_sun() {
    // A ground plane, a slab a metre up over the +x half, and a sun straight
    // overhead. Two probes at the same height: one under the slab, one out
    // in the open.
    let ground = object(slab([-8.0, -8.0, -0.2], [8.0, 8.0, 0.0]), Pbr::plastic([0.5, 0.5, 0.5], 0.6, 0.0));
    let roof = object(slab([0.5, -4.0, 1.0], [6.0, 4.0, 1.2]), Pbr::plastic([0.5, 0.5, 0.5], 0.6, 0.0));
    let scene = Scene {
        objects: vec![ground, roof],
        lights: Vec::new(),
        env: Environment::constant([0.05, 0.05, 0.05]),
        sun: Some(Sun::new(Vec3::new(0.0, 0.0, 1.0), 0.02, [10.0, 10.0, 10.0])),
        ground: None,
        splats: None,
    };
    // Probes at x = -2 (open) and x = +2 (shaded), z = 0.5.
    let volume = VolumeSpec { origin: [-2.0, 0.0, 0.5], spacing: 4.0, dims: [2, 1, 1], scene_per_metre: 1.0 };
    let v = probes::bake(&scene, &volume, &[[0.0, 0.0, 1.0]], false, 64, 11);
    let up = [0.0, 0.0, 1.0];
    let open = v.sample(0.0, [-2.0, 0.0, 0.5], up);
    let shade = v.sample(0.0, [2.0, 0.0, 0.5], up);
    let (o, s) = (open[4] as f64, shade[4] as f64);
    assert!(o > 9.0, "the open probe should have the sun's 10 on it, not {o}");
    assert!(s < 0.25 * o, "the shaded probe sees {s} against the open probe's {o}");
    println!("overhang: {o:.3} in the open, {s:.3} under the slab");
}

/// A volume with a smooth but non-constant field in it: enough variation that
/// a seam at a cell boundary would show, little enough that a millimetre of
/// travel cannot move the answer by 1e-3.
fn wavy(dims: [u32; 3], spacing: f64) -> ProbeVolume {
    let spec = VolumeSpec { origin: [0.0; 3], spacing, dims, scene_per_metre: 1.0 };
    let mut v = ProbeVolume::zeros(&spec, vec![[0.0, 0.0, 1.0]]);
    for iz in 0..dims[2] {
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                let i = v.index(0, ix, iy, iz);
                for k in 0..probes::SH {
                    for b in 0..probes::BANDS {
                        let f = (ix as f64 * 1.7 + iy as f64 * 2.3 + iz as f64 * 0.9 + k as f64).sin();
                        v.data[i + k * probes::BANDS + b] = (0.05 * f * (1.0 + 0.1 * b as f64)) as f32;
                    }
                }
            }
        }
    }
    v
}

#[test]
fn sample_is_continuous_across_a_cell_boundary() {
    let v = wavy([4, 4, 4], 0.5);
    let n = [0.3, -0.5, 0.81];
    let d = 5e-4; // half a millimetre either side of the boundary
    let mut worst = 0.0f32;
    for face in 0..3 {
        for step in 1..4 {
            let mut lo = [0.31, 0.42, 0.23];
            let mut hi = lo;
            lo[face] = step as f64 * 0.5 - d;
            hi[face] = step as f64 * 0.5 + d;
            let (a, b) = (v.sample(0.0, lo, n), v.sample(0.0, hi, n));
            for k in 0..probes::BANDS {
                worst = worst.max((a[k] - b[k]).abs());
            }
        }
    }
    assert!(worst < 1e-3, "a 1 mm step across a cell boundary moved the answer by {worst}");
}

#[test]
fn a_volume_round_trips_through_its_file() {
    let mut v = wavy([3, 2, 4], 0.25);
    v.suns = vec![[0.0, 0.0, 1.0]];
    v.sky = [[0.125; probes::BANDS]; probes::SH];
    v.set_inside(1, 1, 1, true);
    let dir = std::env::temp_dir().join("kosm-probe-round-trip");
    let path = dir.join("probes.bin");
    v.write(&path).expect("write");
    let back = ProbeVolume::read(&path).expect("read");
    assert_eq!(back, v, "the file is the volume, bit for bit");
    assert!(back.is_inside(1, 1, 1));
    assert!(!back.is_inside(0, 0, 0));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_inside_probe_falls_back_to_its_nearest_outside_neighbour() {
    let mut v = wavy([3, 3, 3], 0.5);
    // Bury the whole cell around (0.25, 0.25, 0.25) but leave the rest.
    for iz in 0..2 {
        for iy in 0..2 {
            for ix in 0..2 {
                v.set_inside(ix, iy, iz, true);
            }
        }
    }
    let n = [0.0, 0.0, 1.0];
    let e = v.sample(0.0, [0.25, 0.25, 0.25], n);
    let near = v.sample(0.0, [1.0, 1.0, 1.0], n);
    assert_eq!(e, near, "a buried cell reads its nearest outside probe");
}

#[test]
fn the_gpu_albedo_and_the_bands_meet_at_the_same_matrix() {
    // The contract the raster tier is built against, stated once more from
    // outside the crate: albedo through `band_to_rgb` is `pbr`'s base colour.
    let m = band_to_rgb();
    for name in material::names() {
        let s = material::named(name).expect("a listed substance");
        let g = s.gpu();
        let p = s.pbr();
        let mut rgb = [0.0f32; 3];
        for b in 0..material::BANDS {
            for (c, out) in rgb.iter_mut().enumerate() {
                *out += g.albedo[b] * m[b][c];
            }
        }
        for c in 0..3 {
            assert!((rgb[c] - p.base_color[c]).abs() < 1e-5, "{name} channel {c}: {rgb:?} vs {:?}", p.base_color);
        }
    }
    let (buf, index) = material::library_gpu();
    for name in material::names() {
        assert!(index.contains_key(*name), "{name} is not in the library buffer");
    }
    assert_eq!(buf.len(), index.len());
}

#[test]
fn the_studio_rig_bakes_a_volume_around_the_ball() {
    // The parity test's other half: the datasheet's ball, lit by the same rig
    // the tracer solves directly. Baked once and cached, because it is the
    // same stage for every substance in the library.
    let out = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../out");
    let v = probes::studio_probes(&out).expect("the studio bake");
    assert_eq!(v.dims, [7, 7, 5]);
    assert_eq!(v.suns.len(), 1);
    // The rig is above and around the ball, so an upward face at the ball's
    // own place carries several times what a downward one does.
    let up = v.sample(0.0, [0.0, 0.0, material::BALL_RADIUS], [0.0, 0.0, 1.0]);
    let down = v.sample(0.0, [0.0, 0.0, material::BALL_RADIUS], [0.0, 0.0, -1.0]);
    println!(
        "studio: {:.3} up, {:.3} down at the ball, {} probes at {:.0} mm",
        up[4],
        down[4],
        v.count(),
        v.spacing * 1e3
    );
    assert!(up[4] > 0.05, "the rig lights the stage: {}", up[4]);
    assert!(up[4] > down[4], "and it lights it from above");
    // Cached: the second call reads the file rather than tracing again.
    let t = std::time::Instant::now();
    let again = probes::studio_probes(&out).expect("the cached studio bake");
    assert_eq!(again, v);
    assert!(t.elapsed().as_secs_f64() < 1.0, "the cache is a file, not a bake");
}

#[test]
fn a_bake_is_the_same_bake_twice() {
    let volume = VolumeSpec::over([-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], 1.0);
    let spec = |seed| BakeSpec { volume, rays: 32, seed, sky_only: true, ..BakeSpec::default() };
    let a = probes::bake_with(&furnace(), &spec(3));
    let b = probes::bake_with(&furnace(), &spec(3));
    assert_eq!(a, b, "same seed, same bytes");
    let c = probes::bake_with(&furnace(), &spec(4));
    assert_ne!(a.data, c.data, "a different seed draws different rays");
}
