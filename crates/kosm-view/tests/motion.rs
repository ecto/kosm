//! What the viewer's history does when things move.
//!
//! The viewer talks to one trait, [`TemporalHistory`], and
//! `kosm_render::gpu::History` is the implementation a tier without a device
//! gets: a per-pixel running mean with no reprojection and no filter. These
//! tests pin what that means at the seams — a still camera accumulates, a
//! moved camera or a moved ball does not, and a size step is a new picture.
//!
//! The device tier's own history is pinned in kosm-render's `gpu_temporal`
//! tests; the one here that needs an adapter only checks that the type those
//! tests read back is the type this trait drives, and skips cleanly when
//! there is no adapter, the same way they do.

use kosm_render::gpu::GpuContext;
use kosm_view::temporal::{self, Pose, TemporalHistory, View};
use vcad_kernel_math::{Point3, Vec3};
use vcad_kernel_raytrace::pathtrace::{Camera, Film};

const W: u32 = 8;
const H: u32 = 8;

fn view_from(eye: [f64; 3]) -> View {
    let cam = Camera::look_at(
        Point3::new(eye[0], eye[1], eye[2]),
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        45.0,
    );
    View::of(&cam, W, H)
}

/// A flat film of one brightness, so a mean is easy to read.
fn film(value: f32) -> Film {
    let mut f = Film::new(W, H);
    f.rgb.fill(value);
    f.alpha.fill(1.0);
    f
}

fn ball(x: f64) -> Vec<Pose> {
    vec![Pose::still([x, 0.0, 0.0], 100.0)]
}

#[test]
fn a_still_camera_accumulates() {
    let mut h = temporal::empty((W, H));
    let view = view_from([1000.0, 0.0, 200.0]);
    for value in [0.0f32, 1.0, 2.0, 3.0] {
        assert!(h.reproject(&view, &ball(0.0)) >= 0.0);
        h.accumulate(&film(value));
    }
    assert_eq!(h.mean_samples(), 4.0, "four passes, four samples a pixel");
    // The running mean of 0, 1, 2, 3.
    assert!((h.rgb[0] - 1.5).abs() < 1e-5, "mean was {}", h.rgb[0]);
}

#[test]
fn a_moved_camera_starts_the_picture_over() {
    let mut h = temporal::empty((W, H));
    let view = view_from([1000.0, 0.0, 200.0]);
    h.reproject(&view, &ball(0.0));
    h.accumulate(&film(1.0));
    h.accumulate(&film(1.0));
    assert_eq!(h.mean_samples(), 2.0);

    let kept = h.reproject(&view_from([1000.0, 400.0, 200.0]), &ball(0.0));
    assert_eq!(kept, 0.0, "no pixel survives a camera move on this tier");
    assert_eq!(h.mean_samples(), 0.0);
    h.accumulate(&film(4.0));
    assert!((h.rgb[0] - 4.0).abs() < 1e-5, "the new pass is the picture");
}

#[test]
fn a_moved_ball_starts_the_picture_over_and_a_still_one_does_not() {
    let mut h = temporal::empty((W, H));
    let view = view_from([1000.0, 0.0, 200.0]);
    h.reproject(&view, &ball(0.0));
    h.accumulate(&film(1.0));

    assert_eq!(h.reproject(&view, &ball(0.0001)), 1.0, "a micron is not a move");
    h.accumulate(&film(1.0));
    assert_eq!(h.mean_samples(), 2.0);

    assert_eq!(h.reproject(&view, &ball(50.0)), 0.0, "50 mm is");
    assert_eq!(h.mean_samples(), 0.0);
}

#[test]
fn a_size_step_is_a_new_picture() {
    let mut h = temporal::empty((W, H));
    h.accumulate(&film(1.0));
    assert_eq!(h.mean_samples(), 1.0);
    h.begin((W, H));
    assert_eq!(h.mean_samples(), 1.0, "the same size is the same picture");
    h.begin((W * 2, H));
    assert_eq!((h.width, h.height), (W * 2, H));
    assert_eq!(h.mean_samples(), 0.0);
}

#[test]
fn resolve_is_srgb_bytes_of_the_mean() {
    let mut h = temporal::empty((W, H));
    h.accumulate(&film(0.0));
    let dark = h.resolve(1.0);
    assert_eq!(dark.len() as u32, W * H * 4);
    h.reset();
    h.accumulate(&film(1.0));
    let bright = h.resolve(1.0);
    assert!(bright[0] > dark[0], "a brighter mean resolves brighter");
}

/// The type the device hands back is the type the trait drives.
///
/// `read_history` returns exactly this struct, so a tier that reads its
/// history off the device and a tier that keeps one on the host speak the same
/// planes. Skips when there is no adapter, like kosm-render's `gpu_*` tests.
#[test]
fn gpu_history_is_the_same_history_the_trait_drives() {
    let Ok(_ctx) = GpuContext::init_blocking() else {
        eprintln!("skipping gpu_history_is_the_same_history_the_trait_drives: no GPU");
        return;
    };
    let mut h: kosm_render::gpu::History = temporal::empty((W, H));
    h.accumulate(&film(2.0));
    assert_eq!(h.count[0], 1);
    assert_eq!((h.width, h.height), (W, H));
    assert_eq!(h.rgb.len(), (W * H * 3) as usize);
}
