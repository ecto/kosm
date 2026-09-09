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

/// The being with its dispersion curve taken off, for a tier that cannot
/// afford the spectral variance.
///
/// One hero wavelength is drawn per path the moment a camera ray meets a
/// dispersive surface, and every one of the being's paths meets it: at one
/// sample a pixel a pass that is a *coloured* sample, not a grey one, and the
/// live tier's body is confetti long after its luminance has converged. The
/// dispersion that matters in this level is the caustic's — the rune is a
/// spectral rim on a bright ellipse, and [`Scene::caustic_map`] traces its
/// photons through the real [`being`] either way — so the live picture drops
/// it on the body alone and keeps it where it is the point.
///
/// [`Scene::caustic_map`]: super::Scene::caustic_map
pub fn being_achromatic(n_d: f64) -> Pbr {
    achromatic(being(n_d))
}

/// The keyhole's rim, glowing in proportion to the score.
///
/// The design's hint is light and nothing else: "the aperture's rim glows in
/// proportion to the score". So the rim is an emissive surface — a `Pbr` with
/// `emissive`, not an `AreaLight` — which is the cheap and correct way to put
/// a small self-lit thing in this integrator: a path that lands on it adds the
/// radiance and stops, and no next-event estimator spends a shadow ray on a
/// ring a hundred and fifty millimetres across. It has no transmission, so
/// [`kosm_render::caustics::is_caustic_refractor`] passes over it and the
/// rune's photons are still spent entirely on the being.
///
/// `floor` is what the rim shows at a score of zero, and it is not decoration:
/// a keyhole nobody can see from across the beach is not a puzzle. `gain` is
/// what the score buys on top of it, and it is set so that at `open_frac` the
/// rim is several times the radiance of the sunlit door it sits on.
///
/// Warm white-gold, because the level's one warm light is the low sun and a
/// cold rim would read as a different world's UI.
pub fn rim(floor: f64, gain: f64, score: f64) -> Pbr {
    let r = (floor + gain * score.max(0.0)).max(0.0) as f32;
    Pbr {
        // Dark under its own light: what the rim shows is what it emits, so a
        // bright albedo would only add a second, duller ring of bounced sun.
        base_color: [0.06, 0.05, 0.04],
        roughness: 0.6,
        specular: 0.2,
        emissive: [RIM_TINT[0] * r, RIM_TINT[1] * r, RIM_TINT[2] * r],
        ..Default::default()
    }
}

/// The glint: the same light, on the sand, a step along the gradient.
///
/// Modest on purpose. The rim says "here is the lock"; the glint says "this
/// way", and a glint as bright as the rim would read as a second keyhole.
pub fn glint(radiance: f64) -> Pbr {
    let r = radiance.max(0.0) as f32;
    Pbr {
        base_color: [0.06, 0.05, 0.04],
        roughness: 0.6,
        specular: 0.2,
        emissive: [GLINT_TINT[0] * r, GLINT_TINT[1] * r, GLINT_TINT[2] * r],
        ..Default::default()
    }
}

/// White-gold: the sun's own warmth, a little further toward gold.
const RIM_TINT: [f32; 3] = [1.0, 0.86, 0.58];

/// The glint is whiter than the rim — it is a spark on wet sand, not the lock.
const GLINT_TINT: [f32; 3] = [1.0, 0.93, 0.78];

/// The same material with its index made flat across the spectrum.
///
/// Taken off the caller's own `Pbr` rather than rebuilt from an index, so a
/// document that repainted the being keeps everything about it but the
/// dispersion.
pub fn achromatic(pbr: Pbr) -> Pbr {
    Pbr { sellmeier: None, abbe: 0.0, ..pbr }
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
        // …and the live tier's body is the same glass with the curve off, so
        // the two differ in nothing but their dispersion.
        let a = being_achromatic(1.5168);
        assert!(a.sellmeier.is_none() && !a.is_dispersive());
        assert_eq!((a.ior, a.transmission, a.roughness), (g.ior, g.transmission, g.roughness));
        assert_eq!(a.attenuation_color, g.attenuation_color);
    }

    /// The rim is always visible and always rises with the score, and it is
    /// neither a refractor nor a light the caustic pass would aim at.
    #[test]
    fn the_rim_glows_from_a_floor_and_rises_with_the_score() {
        let lum = |p: Pbr| 0.2126 * p.emissive[0] + 0.7152 * p.emissive[1] + 0.0722 * p.emissive[2];
        let dark = rim(0.8, 4.0, 0.0);
        assert!(lum(dark) > 0.0, "an unlit keyhole is not findable");
        let mut last = lum(dark);
        for k in 1..=10 {
            let now = lum(rim(0.8, 4.0, k as f64 / 10.0));
            assert!(now > last, "the rim did not rise at score {}", k as f64 / 10.0);
            last = now;
        }
        // …and nothing about it joins the rune's photon budget
        for p in [rim(0.8, 4.0, 0.5), glint(6.0)] {
            assert_eq!(p.transmission, 0.0);
        }
    }
}
