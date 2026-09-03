//! Archimedes, measured.
//!
//! Hold the melon fully submerged in settled water and read the force the
//! fluid puts on it. There is one right answer — rho g V, 68.0 N for a melon
//! of 6.93 litres — and the coupling either reproduces it or it does not.
//! Everything about the way the melon floats follows from this number, since
//! the net buoyancy at 950 kg/m^3 is only 5% of it.
use newt_spike::pool::MELON_AXES;
use newt_spike::splash::{Body, Water};
use phyz_math::{GRAVITY, Vec3};

/// Mean fluid force on a melon held at `z`, in newtons, and the ratio to
/// Archimedes.
fn hold(h: f64, gpu: bool, z: f64) -> (f64, f64) {
    let bulk = 2.0e6;
    let dt = 0.35 * h / (bulk / 1000.0f64).sqrt();
    let mut w = Water::fill(h, dt, 1.0, bulk);
    if gpu {
        w.enable_gpu(256).expect("gpu");
    }
    w.settle(2.0);
    let vol = 4.0 / 3.0 * std::f64::consts::PI * MELON_AXES[0] * MELON_AXES[1] * MELON_AXES[2];
    let arch = 1000.0 * GRAVITY * vol;
    let at = |z: f64, vz: f64| Body { centre: Vec3::new(0.0, 0.0, z), axis: Vec3::new(1.0, 0.0, 0.0), vel: Vec3::new(0.0, 0.0, vz) };
    // lower the melon in from above rather than teleporting it into place:
    // dropped into settled water it traps the litres it should have
    // displaced, and the trapped water's weight is booked as force
    let (z0, speed) = (0.25, 0.25);
    let mut zc = z0;
    while zc > z {
        let n = 64usize;
        let step = speed * dt * n as f64;
        zc = (zc - step).max(z);
        w.step_block(&at(zc, -speed), n);
    }
    let body = at(z, 0.0);
    // let the water close around the body before believing the number
    let warm = (0.6 / dt) as usize;
    let mut left = warm;
    while left > 0 {
        let n = left.min(256);
        w.step_block(&body, n);
        left -= n;
    }
    let meas = (0.5 / dt) as usize;
    let (mut acc, mut n) = (0.0, 0usize);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut left = meas;
    while left > 0 {
        let k = left.min(256);
        let b = w.step_block(&body, k).z;
        lo = lo.min(b);
        hi = hi.max(b);
        acc += b * k as f64;
        n += k;
        left -= k;
    }
    let f = acc / n as f64;
    println!("  blocks of 256 substeps ranged {lo:.1} .. {hi:.1} N");
    w.sync_from_gpu();
    let inside = w.x.iter().filter(|p| body.sdf(**p).0 < 0.0).count();
    println!("hydrostatic h={:.3} {} z={z:+.2}: force {f:8.2} N  archimedes {arch:.2} N  ratio {:.3}  particles inside {inside}", h, if gpu { "gpu" } else { "cpu" }, f / arch);
    (f, f / arch)
}

#[test]
fn archimedes() {
    let z: f64 = std::env::var("NEWT_TEST_Z").ok().and_then(|v| v.parse().ok()).unwrap_or(-0.35);
    let hs: Vec<f64> = std::env::var("NEWT_TEST_H")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect())
        .unwrap_or_else(|| vec![0.05, 0.025]);
    for &h in &hs {
        hold(h, true, z);
        if std::env::var_os("NEWT_TEST_CPU").is_some() {
            hold(h, false, z);
        }
    }
}
