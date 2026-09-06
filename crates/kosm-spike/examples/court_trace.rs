use kosm_spike::court::{Court, CourtScene};
fn main() {
    let mut scene = CourtScene::bundled().unwrap();
    scene.n_balls = 1;
    let mut court = Court::from_scene(&scene).unwrap();
    let r = scene.ball_r;
    let mut prev_bottom = f64::INFINITY;
    while court.time() < 2.5 {
        court.step();
        let bottom = court.centre(0).z - r;
        let vz = court.velocity(0).z;
        // print only near the floor
        if bottom < 0.02 || (prev_bottom < 0.02) {
            println!("t {:.4}  bottom {:+.5}  vz {:+.4}", court.time(), bottom, vz);
        }
        prev_bottom = bottom;
    }
}
