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
