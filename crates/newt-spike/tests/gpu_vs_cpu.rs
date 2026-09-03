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
        // where the worst particle is: radius from the region centre, depth
        let (mut worst, mut wi) = (0.0, 0usize);
        for (i, (a, b)) in cpu.v.iter().zip(&gpu.v).enumerate() {
            let d = (*a - *b).norm();
            if d > worst { worst = d; wi = i; }
        }
        let p = cpu.x[wi];
        let (mut far_cnt, mut far_sum) = (0usize, 0.0);
        for (a, b, x) in cpu.v.iter().zip(&gpu.v).zip(&cpu.x).map(|((a, b), x)| (a, b, x)) {
            if x.x.hypot(x.y) > newt_spike::pool::box_half() - 0.15 { far_cnt += 1; far_sum += (*a - *b).norm_squared(); }
        }
        println!("      worst particle at r={:.3} z={:.3}; rms dv within 15 cm of the edge {:.2e} ({far_cnt} particles)", p.x.hypot(p.y), p.z, (far_sum / far_cnt.max(1) as f64).sqrt());
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
    cpu.settle(1.0);
    gpu.settle(1.0);
    let (xr, xm) = rms(&cpu.x, &gpu.x);
    println!("after settle: dx rms {xr:.2e} max {xm:.2e}  level cpu {:.4} gpu {:.4}", cpu.level_offset, gpu.level_offset);
    let (vr, _) = rms(&cpu.v, &gpu.v);
    println!("v rms diff {vr:.2e} (cpu |v| {:.2e})", (cpu.v.iter().map(|v| v.norm_squared()).sum::<f64>() / cpu.v.len() as f64).sqrt());
    let gc = cpu.surface(0.02);
    let gg = gpu.surface(0.02);
    let (zr, zm) = rms(&gc.z.iter().map(|z| Vec3::new(*z, 0.0, 0.0)).collect::<Vec<_>>(), &gg.z.iter().map(|z| Vec3::new(*z, 0.0, 0.0)).collect::<Vec<_>>());
    println!("surface z rms diff {zr:.2e} max {zm:.2e}");
    println!("droplets cpu {} gpu {}", cpu.droplets(&gc, 0.02, 250).len(), gpu.droplets(&gg, 0.02, 250).len());
    for (name, w) in [("cpu", &cpu), ("gpu", &gpu)] {
        let jm = w.j.iter().sum::<f64>() / w.j.len() as f64;
        let jmin = w.j.iter().cloned().fold(f64::MAX, f64::min);
        let zmax = w.x.iter().map(|p| p.z).fold(f64::MIN, f64::max);
        let zmean = w.x.iter().map(|p| p.z).sum::<f64>() / w.x.len() as f64;
        // the top of the bulk: 99th percentile of z
        let mut zs: Vec<f64> = w.x.iter().map(|p| p.z).collect();
        zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let z99 = zs[(zs.len() as f64 * 0.99) as usize];
        println!("{name}: J mean {jm:.4} min {jmin:.4}   z mean {zmean:.4} (rest -0.35)  z99 {z99:.4}  zmax {zmax:.4}  level_offset {:.4}", w.level_offset);
    }
    let gm: f64 = gpu.g_mass.iter().sum();
    let cm: f64 = cpu.g_mass.iter().sum();
    println!("grid mass cpu {cm:.5} gpu {gm:.5}");
}

#[test]
fn density_estimate_at_fill() {
    for ppa in [1usize, 2, 3] {
        unsafe { std::env::set_var("NEWT_PPC", ppa.to_string()) };
        let h = 0.05;
        let mut w = Water::fill(h, 1e-4, 0.5, 2.0e6);
        let far = Body { centre: Vec3::new(0.0, 0.0, 50.0), axis: Vec3::new(1.0, 0.0, 0.0), vel: Vec3::zeros() };
        w.step(&far); // one step, so the grid mass is filled; positions move ~0
        let (mut r, mut b, mut n) = (0.0, 0.0, 0);
        for p in &w.x {
            if p.z > -0.55 && p.z < -0.15 && p.x.abs() < 0.9 && p.y.abs() < 0.5 {
                let (raw, blur) = w.density_at(*p);
                r += raw;
                b += blur;
                n += 1;
            }
        }
        println!("ppc {}^3: bulk density estimate / rest: raw {:.4}  blurred {:.4}  ({} particles)", ppa, r / n as f64, b / n as f64, n);
    }
}

#[test]
fn settled_density_gpu() {
    let h: f64 = std::env::var("NEWT_TEST_H").ok().and_then(|v| v.parse().ok()).unwrap_or(0.025);
    let bulk = 2.0e6;
    let dt = 0.35 * h / (bulk / 1000.0f64).sqrt();
    let mut w = Water::fill(h, dt, 1.0, bulk);
    w.enable_gpu(256).expect("gpu");
    w.settle(2.0);
    let (mut r, mut b, mut j, mut n) = (0.0, 0.0, 0.0, 0);
    for (p, jp) in w.x.iter().zip(&w.j) {
        if p.z > -0.55 && p.z < -0.2 && p.x.abs() < 0.9 && p.y.abs() < 0.5 {
            let (raw, blur) = w.density_at(*p);
            r += raw;
            b += blur;
            j += jp;
            n += 1;
        }
    }
    let mut zs: Vec<f64> = w.x.iter().map(|p| p.z).collect();
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("settled bulk: density/rest raw {:.4} blurred {:.4}   J mean {:.4}   z mean {:.3} z99 {:.3}  ({} particles)", r / n as f64, b / n as f64, j / n as f64, zs.iter().sum::<f64>() / zs.len() as f64, zs[zs.len() * 99 / 100], n);
    let gm: f64 = w.g_mass.iter().sum();
    println!("grid mass {:.2} of {:.2}", gm, w.mass * w.x.len() as f64);
    let dec: Vec<String> = (0..=10).map(|d| format!("{:+.3}", zs[((zs.len() - 1) * d) / 10])).collect();
    println!("z deciles: {}", dec.join(" "));
    // the grid's view: mass per k-row in the central column region
    let full = 1000.0 * h * h * h;
    let (nx, ny, nz) = (w.nx, w.ny, w.nz);
    let mut rows = Vec::new();
    for k in 0..nz {
        let mut m = 0.0;
        let mut c = 0;
        for j in ny / 3..2 * ny / 3 {
            for i in nx / 3..2 * nx / 3 {
                m += w.g_mass[(k * ny + j) * nx + i];
                c += 1;
            }
        }
        rows.push(m / (c as f64 * full));
    }
    let z0 = w.origin.z;
    let profile: Vec<String> = rows.iter().enumerate().filter(|(_, m)| **m > 0.01).map(|(k, m)| format!("{:+.2}:{:.2}", z0 + k as f64 * h, m)).collect();
    println!("grid column (z:mass/full): {}", profile.join(" "));
    println!("particle count {}  grid nodes with mass {}  rest cells {}", w.x.len(), w.g_mass.iter().filter(|m| **m > 0.0).count(), w.x.len() / 8);
    // where the mass is, by distance to the nearest side wall (in cells)
    let mut by_wall = vec![0.0; 8];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let d = i.min(nx - 1 - i).min(j).min(ny - 1 - j).min(7);
                by_wall[d] += w.g_mass[(k * ny + j) * nx + i];
            }
        }
    }
    let total: f64 = by_wall.iter().sum();
    let shells: Vec<String> = by_wall.iter().enumerate().map(|(d, m)| format!("{d}:{:.1}%", 100.0 * m / total)).collect();
    println!("mass by side-wall distance (node shells): {}", shells.join(" "));
    // and the same for the particles, by |x|/POOL_X
    let mut px = vec![0usize; 10];
    for p in &w.x {
        let f = (p.x.abs() / newt_spike::pool::POOL_X).min(0.999);
        px[(f * 10.0) as usize] += 1;
    }
    println!("particles by |x|/POOL_X decile: {:?}", px.iter().map(|c| c * 100 / w.x.len()).collect::<Vec<_>>());
    let mut pz = vec![0usize; 7];
    for p in &w.x {
        let f = ((p.z + 0.7) / 0.1).clamp(0.0, 6.99);
        pz[f as usize] += 1;
    }
    println!("particles by 10 cm z band from the floor: {:?}", pz.iter().map(|c| c * 100 / w.x.len()).collect::<Vec<_>>());
    let floor: usize = w.x.iter().filter(|p| p.z < -0.7 + h).count();
    println!("particles within a cell of the floor: {:.1}% (rest {:.1}%)", 100.0 * floor as f64 / w.x.len() as f64, 100.0 * h / 0.7);
}
