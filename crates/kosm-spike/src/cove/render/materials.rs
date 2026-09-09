//! A surface's name, resolved to the cove's PBR.
//!
//! The court's table is a gym: lacquer over maple, moulded rubber, nylon
//! fuzz, every one of them a material whose *look* is a fact about how light
//! gets a millimetre under its skin. The cove is the opposite claim. Its look
//! is the new Switch Sports — flat saturated albedo, clean silhouettes, soft
//! sky light, no textures — and that look is what a path tracer gives you
//! when you hand it clean solids, a sky, and materials with nothing in them
//! but a colour and a roughness. So there is no `subsurface` here, no
//! `sheen`, no clearcoat on anything that is not literally wet: sand, rock
//! and stone are one colour each at roughness 0.9, and the whole of the
//! picture's interest comes from the shapes, the sun and the glass.
//!
//! Two exceptions, and both earn it. The sea is smooth, because a specular
//! sky reflection is the only thing that makes water read as water in a flat
//! palette; it is *opaque* teal rather than a transmissive volume, because
//! the only refractor in this level is the being — anything else with a
//! transmission lobe would join the caustic pass's aim
//! ([`kosm_render::caustics::is_caustic_refractor`] looks for exactly that)
//! and the rune's photons would be spent on the ocean. And the being is
//! N-BK7 with its Sellmeier pair, so the caustic it throws disperses by the
//! same curve `light.rs` traces it with.
//!
//! Colours are linear, not sRGB. The document's own `[material ...]`
//! definition still wins over everything here, so a level can repaint itself
//! without touching this file.

use kosm_render::pathtrace::Pbr;
use vcad_ir::Document;

/// The PBR a name means in this document: the document's own definition
/// first, then the cove's table, then vcad-render's library, then clay.
pub fn pbr(doc: &Document, name: &str) -> Pbr {
    if let Some(def) = doc.materials.get(name) {
        return vcad_kernel_raytrace::pathtrace::from_material_def(Some(def), None);
    }
    cove(name).unwrap_or_else(|| {
        vcad_kernel_raytrace::pathtrace::from_material_def(vcad_render::materials::builtin(name).as_ref(), None)
    })
}

/// The names the cove uses. One colour and one roughness each; that is the
/// whole style.
fn cove(name: &str) -> Option<Pbr> {
    // Flat: a diffuse lobe, a weak dielectric highlight, nothing layered.
    let flat = |base: [f32; 3], roughness: f32| Pbr {
        base_color: base,
        roughness,
        specular: 0.25,
        ..Default::default()
    };
    Some(match name {
        // Dry sand in low sun: warm, pale, and the brightest thing in the
        // picture that is not the sky, so the being reads dark against it.
        "sand" => flat([0.85, 0.54, 0.22], 0.9),
        // The cliff and the boulders. Grey-blue, so the one big vertical mass
        // in the frame sits back from the sand instead of competing with it,
        // and so the sky's colour has something cool to land on.
        "rock" => flat([0.13, 0.19, 0.33], 0.9),
        // The door: the same stone, cut and dressed. Darker and warmer than
        // the cliff it is set into — that difference is the whole reason the
        // door reads as a door from across the beach, before the aperture is
        // visible at all.
        "stone" => flat([0.24, 0.20, 0.16], 0.85),
        // The sea. Opaque on purpose (see the module note): a saturated teal
        // body colour under a smooth dielectric surface, so the swell is
        // legible as a field of sky reflections and the water is still teal
        // where it is not reflecting anything.
        "water" => Pbr {
            base_color: [0.03, 0.30, 0.30],
            roughness: 0.10,
            specular: 0.5,
            ior: 1.333,
            ..Default::default()
        },
        _ => return None,
    })
}

/// The being: a body of N-BK7, at the index the level authored.
///
/// `transmission: 1.0` with no `thin_walled` is precisely what the caustic
/// pass looks for, so declaring the being's glass *is* declaring what the
/// rune is thrown by. The Sellmeier pair is the datasheet's, not an Abbe
/// approximation, because it is the same curve `light.rs` traces the score
/// with — one glass, one dispersion, whichever code is asking.
pub fn being(n_d: f64) -> Pbr {
    Pbr {
        base_color: [1.0, 1.0, 1.0],
        roughness: 0.02,
        transmission: 1.0,
        ior: n_d as f32,
        specular: 1.0,
        sellmeier: Some(kosm_render::spectrum::BK7_SELLMEIER),
        // The iron in the melt: a metre of it transmits about this, the same
        // Beer-Lambert pair the court's backboard carries. On a twelve
        // millimetre board it is invisible and only the polished edge is
        // bottle-green; on a body seven hundred millimetres thick it is the
        // difference between a being you can see and a hole in the picture
        // where the sand shows through unchanged.
        attenuation_color: [0.78, 0.92, 0.83],
        attenuation_distance: 1000.0,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_coves_names_resolve_flat_and_the_being_is_a_refractor() {
        let doc = Document::new();
        // sand is warm: more red than blue; rock is cool: more blue than red
        assert!(pbr(&doc, "sand").base_color[0] > pbr(&doc, "sand").base_color[2]);
        assert!(pbr(&doc, "rock").base_color[2] > pbr(&doc, "rock").base_color[0]);
        // flat means no layering, at any of them
        for name in ["sand", "rock", "stone"] {
            let p = pbr(&doc, name);
            assert_eq!(p.clearcoat, 0.0, "{name} is flat");
            assert_eq!(p.subsurface, 0.0, "{name} is flat");
            assert!(p.roughness >= 0.8, "{name} is rough");
        }
        // the sea is smooth and opaque — opaque so the caustic pass aims at
        // the being and not at the ocean
        assert!(pbr(&doc, "water").roughness < 0.2);
        assert_eq!(pbr(&doc, "water").transmission, 0.0);
        // and the being is the one thing the caustic pass should find
        let g = being(1.5168);
        assert!(g.transmission > 0.0 && !g.thin_walled);
        assert!(g.sellmeier.is_some());
    }
}
