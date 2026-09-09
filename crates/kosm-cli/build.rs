//! The registry, from the tree under `sims/`. The walk is `kosm-registry`'s,
//! so a consumer of kosm (ipse's `sims/k1/...`) gets the same one.

fn main() {
    let sims = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../sims");
    kosm_registry::generate(sims, kosm_registry::out_path());
}
