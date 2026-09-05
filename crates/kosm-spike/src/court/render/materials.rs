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
        return vcad_kernel_raytrace::pathtrace::from_material_def(Some(def), None);
    }
    court(name).unwrap_or_else(|| vcad_kernel_raytrace::pathtrace::from_material_def(vcad_render::materials::builtin(name).as_ref(), None))
}

/// The names a basketball court uses that no general material library has.
///
/// Written in the principled model's own terms rather than in
/// metallic-roughness alone, because a gym is mostly the kinds of surface that
/// metallic-roughness is worst at: a specularly smooth lacquer over a
/// diffusely rough substrate, a rubber ball whose colour comes from a
/// millimetre under its skin, and nylon cord that is fuzz all the way down.
/// `diffuse_roughness` separates the wood's grain from the varnish over it;
/// `subsurface` is what stops the ball reading as painted plastic; `sheen` is
/// the net and the ball's pebbling.
fn court(name: &str) -> Option<Pbr> {
    let p = |base: [f32; 3], roughness: f32, clearcoat: f32| Pbr {
        base_color: base,
        roughness,
        clearcoat,
        clearcoat_roughness: 0.06,
        ..Default::default()
    };
    Some(match name {
        // Lacquered hard maple. Two roughnesses doing two jobs: the coat is
        // near-mirror (that is the long reflection of the lights down the
        // floor) and the wood beneath it is diffusely rough, which is the
        // grain reading flat rather than plasticky.
        "maple" => Pbr {
            diffuse_roughness: 0.55,
            clearcoat_roughness: 0.045,
            ..p([0.52, 0.36, 0.19], 0.30, 0.85)
        },
        // The backboard: real soda-lime glass. A full transmission lobe at
        // 1.52, so the ring's shadow and the wall behind carry through it the
        // way they do in a gym, and the faint green of iron in float glass —
        // a metre of it transmits about (0.78, 0.92, 0.83) — put in as
        // Beer-Lambert absorption rather than as a tint on the surface, so a
        // 12 mm board is barely green and its polished edge is bottle-green,
        // which is exactly the tell that a backboard is glass and not acrylic.
        "glass" => Pbr {
            transmission: 1.0,
            ior: 1.52,
            attenuation_color: [0.78, 0.92, 0.83],
            attenuation_distance: 1000.0,
            specular: 1.0,
            ..p([1.0, 1.0, 1.0], 0.02, 0.0)
        },
        // Painted steel ring: baked enamel over metal. The paint is a
        // dielectric, so the ring is not `metallic` — the coat is what makes
        // it read as painted steel rather than as orange plastic.
        "rim" => Pbr {
            diffuse_roughness: 0.35,
            clearcoat_roughness: 0.09,
            ..p([0.72, 0.20, 0.04], 0.32, 0.55)
        },
        // The lines and the painted key: flat latex on wood. Latex is the
        // canonical high-`diffuse_roughness` surface — chalky, nearly
        // shadowless at the terminator — and it barely reflects at all.
        "paint" => Pbr {
            diffuse_roughness: 0.85,
            specular: 0.3,
            ..p([0.78, 0.78, 0.75], 0.45, 0.15)
        },
        // A ball is rubber: pebbled, barely glossy, and lit a millimetre under
        // its own skin. A touch of `subsurface` is what keeps the terminator
        // soft; the sheen is the pebbling catching the light at the silhouette.
        "ball" => Pbr {
            diffuse_roughness: 0.7,
            subsurface: 0.25,
            specular: 0.35,
            sheen: 0.25,
            sheen_color: [1.0, 0.85, 0.72],
            sheen_roughness: 0.55,
            ..p([0.58, 0.24, 0.08], 0.62, 0.06)
        },
        // The seams are the same rubber, moulded into a channel and darker for
        // it — same model, no colour left to reflect.
        "ball-seams" | "seam" => Pbr {
            diffuse_roughness: 0.7,
            specular: 0.35,
            ..p([0.03, 0.025, 0.02], 0.70, 0.0)
        },
        // The lane, painted a team colour under the same lacquer as the maple.
        "key" => Pbr {
            diffuse_roughness: 0.8,
            clearcoat_roughness: 0.05,
            ..p([0.10, 0.18, 0.45], 0.40, 0.60)
        },
        // Wall padding: vinyl over foam. Vinyl is a soft sheet with a low,
        // broad sheen off its texture, and it is the one thing in the gym with
        // a visible grazing bloom.
        "pad" => Pbr {
            diffuse_roughness: 0.6,
            sheen: 0.2,
            sheen_roughness: 0.7,
            specular: 0.45,
            ..p([0.10, 0.20, 0.52], 0.55, 0.10)
        },
        // The floor beyond the maple: sealed concrete. Rough both ways, with
        // just enough coat to be sealed rather than raw.
        "floor" => Pbr {
            diffuse_roughness: 0.9,
            specular: 0.3,
            ..p([0.46, 0.46, 0.45], 0.80, 0.05)
        },
        // A net is nylon cord, which is fuzz: fibres standing off the strand
        // in every direction. Sheen is the whole material — it is why a net
        // reads as bright rope against a dark gym and not as white tubing.
        "net" => Pbr {
            diffuse_roughness: 0.8,
            sheen: 0.9,
            sheen_roughness: 0.3,
            specular: 0.25,
            ..p([0.86, 0.86, 0.83], 0.65, 0.0)
        },
        // The room.
        "wall" => Pbr {
            diffuse_roughness: 0.85,
            specular: 0.3,
            ..p([0.56, 0.55, 0.51], 0.85, 0.0)
        },
        "ceiling" => Pbr {
            diffuse_roughness: 0.9,
            specular: 0.2,
            ..p([0.20, 0.20, 0.21], 0.90, 0.0)
        },
        // The clerestory panes. Modelled as 20 mm boxes, which is thicker
        // than the pane they stand for, so they are `thin_walled`: the
        // daylight behind them arrives undisplaced instead of being refracted
        // twice through a slab it is not really twenty millimetres of.
        "window" => Pbr {
            transmission: 1.0,
            thin_walled: true,
            ior: 1.52,
            specular: 1.0,
            ..p([1.0, 1.0, 1.0], 0.02, 0.0)
        },
        "decor" => Pbr {
            diffuse_roughness: 0.5,
            ..p([0.45, 0.45, 0.47], 0.60, 0.0)
        },
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
