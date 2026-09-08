//! What the court's whole picture costs to assemble: every root walked to its
//! placed primitives, a BVH per distinct solid, the gym and the lights.
use kosm::court::{render, CourtScene};

fn main() {
    let scene = CourtScene::bundled().unwrap();
    let t = std::time::Instant::now();
    let picture = render::Scene::new(&scene.authored, scene.ball_r).unwrap();
    println!("Scene::new {:.3} s  {} statics", t.elapsed().as_secs_f64(), picture.static_count());
}
