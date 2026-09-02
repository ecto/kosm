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
