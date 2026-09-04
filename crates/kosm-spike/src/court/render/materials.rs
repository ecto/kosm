//! A root's material name, resolved to a vcad PBR.
//!
//! The document's own `[material ...]` definition wins, then the court's own
//! table for the names this level uses (`maple`, `rim`, `ball`, the gym), then
//! vcad-render's built-in library (`steel`, `glass`, `rubber`, …), then vcad's
//! default clay grey. One table, so the physics and the picture agree on what
//! a root is made of by reading the same string.

use vcad_ir::Document;
use vcad_kernel_raytrace::pathtrace::Pbr;

/// The PBR a root's material name means, in this document.
pub fn pbr(doc: &Document, name: &str) -> Pbr {
    if let Some(def) = doc.materials.get(name) {
        return Pbr::from_material_def(Some(def), None);
    }
    court(name).unwrap_or_else(|| Pbr::from_material_def(vcad_render::materials::builtin(name).as_ref(), None))
}

/// The names a basketball court uses that no general material library has.
fn court(name: &str) -> Option<Pbr> {
    let p = |base: [f32; 3], roughness: f32, clearcoat: f32| Pbr {
        base_color: base,
        roughness,
        clearcoat,
        clearcoat_roughness: 0.06,
        ..Default::default()
    };
    Some(match name {
        // lacquered hard maple: pale, warm, and under a gloss coat
        "maple" => p([0.52, 0.36, 0.19], 0.30, 0.85),
        // the backboard: a bright dielectric sheet, near-mirror
        "glass" => p([0.72, 0.78, 0.80], 0.05, 0.0),
        // painted steel ring: the one orange everyone knows
        "rim" => p([0.72, 0.20, 0.04], 0.32, 0.55),
        // the lines and the painted key: flat latex on wood
        "paint" => p([0.78, 0.78, 0.75], 0.40, 0.25),
        // a ball is rubber, pebbled, barely glossy
        "ball" => p([0.58, 0.24, 0.08], 0.62, 0.10),
        // a net: nylon cord
        "net" => p([0.86, 0.86, 0.83], 0.65, 0.0),
        // the room
        "wall" => p([0.56, 0.55, 0.51], 0.85, 0.0),
        "ceiling" => p([0.20, 0.20, 0.21], 0.90, 0.0),
        "window" => p([0.72, 0.78, 0.80], 0.05, 0.0),
        "decor" => p([0.45, 0.45, 0.47], 0.60, 0.0),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_courts_names_and_the_library_both_resolve() {
        let doc = Document::new();
        // the court's own table
        assert_eq!(pbr(&doc, "maple").clearcoat, 0.85);
        assert!(pbr(&doc, "rim").base_color[0] > pbr(&doc, "rim").base_color[2]);
        // vcad-render's library, for the names it already knows
        assert_eq!(pbr(&doc, "steel").metallic, 1.0);
        // an unknown name is vcad's default clay, not a panic
        assert_eq!(pbr(&doc, "unobtainium").metallic, 0.0);
    }
}
