//! The surface and the caustic, on the GPU against the CPU reference.

use kosm::fluid::{self as pool, Surface};
use kosm::fluid::splash::{HeightGrid, Water};

#[test]
fn surface_matches_the_cpu() {
    let h: f64 = std::env::var("KOSM_TEST_H").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);
    let bulk = 2.0e6;
    let dt = 0.35 * h / (bulk / 1000.0f64).sqrt();
    let mut w = Water::fill(h, dt, 0.5, bulk);
    w.enable_gpu(256).expect("gpu");
    w.settle(1.0);
    // the CPU reference reads the mirrors, so refresh them once
    w.sync_from_gpu();
    let cpu = w.surface(0.02);
    let gpu = w.surface_gpu(0.02, 0.02, 0.6, 1.5, 0.6).expect("gpu surface");
    assert_eq!((cpu.nx, cpu.ny), (gpu.nx, gpu.ny));
    let (mut sum, mut worst) = (0.0f64, 0.0f64);
    for (a, b) in cpu.z.iter().zip(&gpu.z) {
        let d = (a - b).abs();
        sum += d * d;
        worst = worst.max(d);
    }
    let rms = (sum / cpu.z.len() as f64).sqrt();
    println!("surface  {}x{} cells   rms {rms:.2e} m   max {worst:.2e} m", cpu.nx, cpu.ny);
    assert!(worst < 1e-4, "GPU surface differs by {worst} m");

    // and the picker: every candidate must satisfy the filter the CPU uses
    let cand = w.gpu_candidates();
    let off = w.level_offset;
    let mut bad = 0;
    for d in &cand {
        let s = gpu.at(d.pos.x, d.pos.y) + off;
        let drop = d.pos.z > s + 0.02;
        let foam = d.vel.norm() >= 0.6 && d.pos.z >= s - 1.5 * h && d.pos.z <= s + 0.6;
        if !drop && !foam {
            bad += 1;
        }
    }
    println!("picker   {} candidates of {} particles, {bad} outside the filter", cand.len(), w.count());
    assert_eq!(bad, 0);
}

#[test]
fn caustic_matches_the_cpu() {
    // a test frame: a ripple in the box, a swell outside it
    let cell = 0.02;
    let half = pool::box_half();
    let n = ((2.0 * half) / cell) as usize;
    let mut fine = HeightGrid { origin: [-half, -half], cell, nx: n, ny: n, z: vec![0.0; n * n] };
    for j in 0..n {
        for i in 0..n {
            let x = -half + (i as f64 + 0.5) * cell;
            let y = -half + (j as f64 + 0.5) * cell;
            fine.z[j * n + i] = 0.01 * ((6.0 * x).sin() * (5.0 * y).cos()) * (-(x * x + y * y) / 0.6).exp();
        }
    }
    let mut far = HeightGrid { origin: [-pool::POOL_X, -pool::POOL_Y], cell: 0.1, nx: 500, ny: 250, z: vec![0.0; 500 * 250] };
    for j in 0..far.ny {
        for i in 0..far.nx {
            let x = far.origin[0] + (i as f64 + 0.5) * far.cell;
            far.z[j * far.nx + i] = 0.002 * (1.5 * x).sin();
        }
    }
    let surface = Surface { rings: Vec::new(), t: 0.37, grid: Some(fine), far: Some(far) };
    let cpu = pool::caustic(&surface, cell);
    let mut gpu_dev = kosm_mpm::GpuCaustic::new().expect("gpu");
    let gpu = pool::caustic_gpu(&mut gpu_dev, &surface, cell).expect("gpu caustic");
    assert_eq!((cpu.nx, cpu.ny), (gpu.nx, gpu.ny));
    // the mean over a flat stretch of floor is the sanity check: every ray
    // that a flat surface would have sent here still arrives
    let (mut mc, mut mg, mut cnt) = (0.0, 0.0, 0.0f64);
    let (mut sum, mut worst) = (0.0f64, 0.0f64);
    for j in 2..cpu.ny - 2 {
        for i in 2..cpu.nx - 2 {
            let (a, b) = (cpu.e[j * cpu.nx + i], gpu.e[j * cpu.nx + i]);
            let d = (a - b).abs();
            sum += d * d;
            worst = worst.max(d);
            let x = cpu.origin[0] + (i as f64 + 0.5) * cell;
            let y = cpu.origin[1] + (j as f64 + 0.5) * cell;
            if x.hypot(y) > 2.5 {
                mc += a;
                mg += b;
                cnt += 1.0;
            }
        }
    }
    let rms = (sum / ((cpu.nx - 4) * (cpu.ny - 4)) as f64).sqrt();
    println!("caustic  {}x{}   flat mean cpu {:.4} gpu {:.4}   rms {rms:.2e}  max {worst:.2e}", cpu.nx, cpu.ny, mc / cnt, mg / cnt);
    assert!((mc / cnt - 1.0).abs() < 0.02, "cpu flat mean {}", mc / cnt);
    assert!((mg / cnt - 1.0).abs() < 0.02, "gpu flat mean {}", mg / cnt);
    assert!(rms < 0.03, "caustic rms {rms}");
}

#[test]
fn caustic_flux_probe() {
    let cell = 0.02;
    let half = pool::box_half();
    let n = ((2.0 * half) / cell) as usize;
    let fine = HeightGrid { origin: [-half, -half], cell, nx: n, ny: n, z: vec![0.0; n * n] };
    let far = HeightGrid { origin: [-pool::POOL_X, -pool::POOL_Y], cell: 0.1, nx: 500, ny: 250, z: vec![0.0; 500 * 250] };
    let surface = Surface { rings: Vec::new(), t: 0.0, grid: Some(fine), far: Some(far) };
    let cpu = pool::caustic(&surface, cell);
    let mut dev = kosm_mpm::GpuCaustic::new().expect("gpu");
    let gpu = pool::caustic_gpu(&mut dev, &surface, cell).expect("gpu");
    println!("flux cpu {:.1} gpu {:.1} cells {}", cpu.e.iter().sum::<f64>(), gpu.e.iter().sum::<f64>(), cpu.e.len());
}
