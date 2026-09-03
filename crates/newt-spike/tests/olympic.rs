use newt_spike::pool::{self, Drop};
#[test]
fn surface_across_the_box_edge() {
    let mut d = Drop::new(1.3).with_water(0.05);
    for _ in 0..17 {
        d.step();
    }
    d.read_water();
    let w = d.water.as_ref().unwrap();
    println!("level_offset {:.4}  box {}", w.level_offset, pool::box_half());
    for x in [0.0, 0.5, 0.85, 0.9, 0.95, 0.99, 1.01, 1.2, 2.0, 10.0] {
        println!("x={x:5.2}  height {:+.4}  fine {:+.4}  far {:+.4}", d.surface.height(x, 0.0), d.surface.grid.as_ref().map(|g| g.at(x, 0.0)).unwrap_or(f64::NAN), d.surface.far.as_ref().map(|g| g.at(x, 0.0)).unwrap_or(f64::NAN));
    }
    println!("top {:+.4}", d.surface.top());
}

#[test]
fn caustic_of_flat_water_is_one() {
    use newt_spike::pool::{caustic, Surface};
    use newt_spike::splash::HeightGrid;
    let flat = |cell: f64, half: f64| HeightGrid { origin: [-half, -half], cell, nx: (2.0 * half / cell) as usize, ny: (2.0 * half / cell) as usize, z: vec![0.0; ((2.0 * half / cell) as usize).pow(2)] };
    for (name, s) in [
        ("flat grid + flat far, t=0", Surface { rings: Vec::new(), t: 0.0, grid: Some(flat(0.02, 1.0)), far: Some(flat(0.1, 25.0)) }),
        ("rings only (ambient ripple), t=0", Surface { rings: Vec::new(), t: 0.0, grid: None, far: None }),
    ] {
        let c = caustic(&s, 0.02);
        let (mut lo, mut hi, mut sum, mut n) = (9.0f64, 0.0f64, 0.0, 0);
        for j in 0..c.ny {
            for i in 0..c.nx {
                let x = c.origin[0] + (i as f64 + 0.5) * c.cell;
                let y = c.origin[1] + (j as f64 + 0.5) * c.cell;
                if x.hypot(y) < 3.0 {
                    let e = c.e[j * c.nx + i];
                    lo = lo.min(e);
                    hi = hi.max(e);
                    sum += e;
                    n += 1;
                }
            }
        }
        println!("{name}: caustic within 3 m: min {lo:.3} max {hi:.3} mean {:.3}", sum / n as f64);
    }
}

#[test]
fn seam_profile_at_impact() {
    let h: f64 = std::env::var("NEWT_TEST_H").ok().and_then(|v| v.parse().ok()).unwrap_or(0.025);
    let frame: usize = std::env::var("NEWT_TEST_FRAME").ok().and_then(|v| v.parse().ok()).unwrap_or(46);
    let mut d = Drop::new(1.3).with_water(h);
    let steps = (1.0 / pool::fps() / d.model.dt).round() as usize;
    for _ in 0..frame {
        for _ in 0..steps {
            d.step();
        }
        d.read_water();
    }
    let half = pool::box_half();
    println!("box half {half}  sponge {}  blend {}", pool::SPONGE, pool::BLEND);
    let mut x = 0.0;
    while x < half + 0.6 {
        let y = if std::env::var_os("NEWT_TEST_DIAG").is_some() { x } else { 0.0 };
        let hgt = d.surface.height(x, y);
        let fine = d.surface.grid.as_ref().map(|g| g.at(x, y)).unwrap_or(f64::NAN);
        let far = d.surface.far.as_ref().map(|g| g.at(x, y)).unwrap_or(f64::NAN);
        let n = d.surface.normal(x, y);
        println!("x={x:5.2}  h {:+7.1} mm  fine {:+7.1}  far {:+7.1}  slope {:.3}", hgt * 1000.0, fine * 1000.0, far * 1000.0, n.x.hypot(n.y) / n.z);
        x += 0.05;
    }
}

#[test]
fn far_field_conserves_energy() {
    use newt_spike::far::Far;
    let mut far = Far::new(0.1);
    // a Gaussian bump, 2 cm high, 30 cm wide, released from rest
    let (nx, cell) = (far.grid.nx, far.grid.cell);
    for j in 0..far.grid.ny {
        for i in 0..nx {
            let x = far.grid.origin[0] + (i as f64 + 0.5) * cell;
            let y = far.grid.origin[1] + (j as f64 + 0.5) * cell;
            far.grid.z[j * nx + i] = 0.02 * (-(x * x + y * y) / (2.0 * 0.3 * 0.3)).exp();
        }
    }
    // zero mean: the k=0 mode is not a wave, is not damped, and would sit in H as a constant
    let mean = far.grid.z.iter().sum::<f64>() / far.grid.z.len() as f64;
    for z in far.grid.z.iter_mut() {
        *z -= mean;
    }
    let e0 = far.energy();
    let t0 = std::time::Instant::now();
    let mut worst: f64 = 0.0;
    for k in 0..600 {
        far.step(1.0 / 60.0);
        let e = far.energy();
        // the step has a 0.05/s damping; undo it for the comparison
        let expect = e0 * (-0.1 * far.time).exp();
        worst = worst.max((e - expect).abs() / e0);
        if k % 120 == 0 {
            println!("t={:5.2} s  H={:.6e}  expected {:.6e}  rel err {:.2e}", far.time, e, expect, (e - expect).abs() / e0);
        }
    }
    println!("worst relative energy error over 10 s: {worst:.2e}   ({:.1} ms per step incl. energy)", t0.elapsed().as_millis() as f64 / 600.0);
    assert!(worst < 1e-6, "linear spectral step must conserve H to roundoff");
}

#[test]
fn far_fft_speed() {
    use newt_spike::far::Far;
    let mut far = Far::new(0.1);
    far.grid.z[1000] = 0.01;
    let t = std::time::Instant::now();
    for _ in 0..20 {
        far.step(1.0 / 60.0);
    }
    println!("step only: {:.1} ms", t.elapsed().as_millis() as f64 / 20.0);
    let t = std::time::Instant::now();
    for _ in 0..20 {
        let _ = far.energy();
    }
    println!("energy only: {:.1} ms", t.elapsed().as_millis() as f64 / 20.0);
}
