//! Which of the court's roots cost what to evaluate to a solid.
use kosm_spike::court::CourtScene;
fn main() {
    let scene = CourtScene::bundled().unwrap();
    let doc = &scene.authored.document;
    let mut cache = std::collections::HashMap::new();
    for root in &doc.roots {
        let t = std::time::Instant::now();
        let solid = vcad_eval::evaluate_node(root.root, &doc.nodes, &mut cache).unwrap();
        let faces = solid.as_ref().and_then(|s| s.as_brep()).map(|b| b.topology.faces.len()).unwrap_or(0);
        println!("{:<12} {:>8.2} s  {:>5} faces", root.material, t.elapsed().as_secs_f64(), faces);
    }
}
