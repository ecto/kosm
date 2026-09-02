//! The GPU solver against the CPU solver from the same state.
use newt_spike::splash::{Body, Water};
use phyz_math::Vec3;

fn rms(a: &[Vec3], b: &[Vec3]) -> (f64, f64) {
    let mut s = 0.0;
    let mut m: f64 = 0.0;
    for (x, y) in a.iter().zip(b) {
        let d = (*x - *y).norm();
        s += d * d;
        m = m.max(d);
    }
    ((s / a.len() as f64).sqrt(), m)
}

#[test]
fn one_and_many_substeps() {
    let h: f64 = std::env::var("NEWT_TEST_H").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);
    let air: f64 = std::env::var("NEWT_TEST_AIR").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
    let bulk = 2.0e6;
    let dt = 0.35 * h / (bulk / 1000.0f64).sqrt();
    let mut cpu = Water::fill(h, dt, air, bulk);
    let mut gpu = Water::fill(h, dt, air, bulk);
    gpu.enable_gpu(1000).expect("gpu");
    let far = Body { centre: Vec3::new(0.0, 0.0, 50.0), axis: Vec3::new(1.0, 0.0, 0.0), vel: Vec3::zeros() };
    for &n in &[1usize, 10, 100, 1000] {
        for _ in 0..n {
            cpu.step(&far);
        }
        gpu.step_block(&far, n);
        gpu.sync_from_gpu();
        let (vr, vm) = rms(&cpu.v, &gpu.v);
        let (xr, xm) = rms(&cpu.x, &gpu.x);
        let vscale = (cpu.v.iter().map(|v| v.norm_squared()).sum::<f64>() / cpu.v.len() as f64).sqrt();
        println!("after +{n:4} steps: |v| rms {vscale:.4}  dv rms {vr:.2e} max {vm:.2e}   dx rms {xr:.2e} max {xm:.2e}");
        let gm: f64 = gpu.g_mass.iter().sum();
        let cm: f64 = cpu.g_mass.iter().sum();
        println!("                 grid mass cpu {cm:.5} gpu {gm:.5}");
    }
}

#[test]
fn settle_and_droplets() {
    let h: f64 = std::env::var("NEWT_TEST_H").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);
    let air: f64 = std::env::var("NEWT_TEST_AIR").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
    let bulk = 2.0e6;
    let dt = 0.35 * h / (bulk / 1000.0f64).sqrt();
    let mut cpu = Water::fill(h, dt, air, bulk);
    let mut gpu = Water::fill(h, dt, air, bulk);
    gpu.enable_gpu(256).expect("gpu");
    cpu.settle(0.6);
    gpu.settle(0.6);
    let (xr, xm) = rms(&cpu.x, &gpu.x);
    println!("after settle: dx rms {xr:.2e} max {xm:.2e}  level cpu {:.4} gpu {:.4}", cpu.level_offset, gpu.level_offset);
    let (vr, _) = rms(&cpu.v, &gpu.v);
    println!("v rms diff {vr:.2e} (cpu |v| {:.2e})", (cpu.v.iter().map(|v| v.norm_squared()).sum::<f64>() / cpu.v.len() as f64).sqrt());
    let gc = cpu.surface(0.02);
    let gg = gpu.surface(0.02);
    let (zr, zm) = rms(&gc.z.iter().map(|z| Vec3::new(*z, 0.0, 0.0)).collect::<Vec<_>>(), &gg.z.iter().map(|z| Vec3::new(*z, 0.0, 0.0)).collect::<Vec<_>>());
    println!("surface z rms diff {zr:.2e} max {zm:.2e}");
    println!("droplets cpu {} gpu {}", cpu.droplets(&gc, 0.02, 250).len(), gpu.droplets(&gg, 0.02, 250).len());
    let gm: f64 = gpu.g_mass.iter().sum();
    let cm: f64 = cpu.g_mass.iter().sum();
    println!("grid mass cpu {cm:.5} gpu {gm:.5}");
}
