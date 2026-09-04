use kosm_spike::court::{render, Court, CourtScene};
fn main() {
    let scene = CourtScene::bundled().unwrap();
    let court = Court::from_scene(&scene).unwrap();
    let picture = render::Scene::new(&scene, &court).unwrap();
    let cam = render::camera(&scene, 96, 54).unwrap();
    println!("prims {}  camera {:?}", picture.prim_count(), cam);
    println!("centre ray sees {:?}", picture.probe(&cam));
    let t = std::time::Instant::now();
    let img = render::render(&picture, &cam, 4, 0);
    let bright = img.pixels().filter(|p| p.0.iter().any(|&c| c > 0)).count();
    println!("{} of {} pixels lit in {:?}", bright, img.width() * img.height(), t.elapsed());
}
