//! What the court's picture is made of, and how long a small one takes.
use kosm::court::{render, Court, CourtScene};
fn main() {
    let scene = CourtScene::bundled().unwrap();
    let court = Court::from_scene(&scene).unwrap();
    let mut picture = render::Scene::new(&scene).unwrap();
    let cam = render::camera(&scene).unwrap();
    println!(
        "static {}  panels {}  camera eye {:?} fov {:.0}° aperture {:.1} mm focus {:.0} mm",
        picture.static_count(),
        picture.light_count(),
        cam.eye,
        cam.fov_deg,
        cam.aperture,
        cam.focus_dist
    );
    let at = picture.at(&court);
    println!("objects {} (statics + {} balls + {} extras)", at.objects.len(), court.bodies(), court.extras.len());
    let t = std::time::Instant::now();
    let film = render::render(&at, &cam, 96, 54, &render::options(&scene, 4, 0));
    let img = render::to_image(&film, 1.0, 1);
    let bright = img.pixels().filter(|p| p.0[..3].iter().any(|&c| c > 0)).count();
    println!("{} of {} pixels lit in {:?}", bright, img.width() * img.height(), t.elapsed());
}
