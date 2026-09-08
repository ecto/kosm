//! What each of the court's roots costs to turn into geometry, both ways:
//! evaluated to one solid with vcad's booleans, and walked to the placed
//! primitives it is a union of.
use kosm::brep::instances::{instances, Prims};
use kosm::court::CourtScene;

fn main() {
    let scene = CourtScene::bundled().unwrap();
    let doc = &scene.authored.document;

    println!("{:<12} {:>9}  {:>9}  {:>7}  {:>6}", "root", "eval", "walk", "solids", "faces");
    let mut cache = std::collections::HashMap::new();
    let mut prims = Prims::default();
    let (mut eval_total, mut walk_total) = (0.0, 0.0);
    for root in &doc.roots {
        let t = std::time::Instant::now();
        let solid = vcad_eval::evaluate_node(root.root, &doc.nodes, &mut cache).unwrap();
        let eval = t.elapsed().as_secs_f64();
        let faces = solid.as_ref().and_then(|s| s.as_brep()).map(|b| b.topology.faces.len()).unwrap_or(0);

        let t = std::time::Instant::now();
        let placed = instances(doc, root.root, &mut prims).unwrap();
        let walk = t.elapsed().as_secs_f64();

        eval_total += eval;
        walk_total += walk;
        println!("{:<12} {eval:>8.3} s  {walk:>8.3} s  {:>7}  {faces:>6}", root.material, placed.len());
    }
    println!("{:<12} {eval_total:>8.3} s  {walk_total:>8.3} s  ({} distinct primitives)", "total", prims.distinct());
}
