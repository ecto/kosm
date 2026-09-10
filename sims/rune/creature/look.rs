//! What the creature is made of, and the small stage it stands on.
//!
//! The cove's own palette is in `sims/rune/materials.rs` and this borrows its
//! sand and its stone verbatim, along with the sky and the sun from
//! `render.rs`'s `daylight`. What it adds is the one thing the cove has no
//! word for: a body that light gets *into*. The being of the puzzle is
//! N-BK7, a solid of glass, and it works because glass is a single surface
//! event; a creature is the other case entirely, and reads as alive exactly
//! because light goes a centimetre under its skin and comes back out warmer
//! and softer than it went in.
//!
//! So the bell, the limbs and the core are subsurface materials in the sense
//! `kosm_render`'s CPU tier means it — a real random walk through a medium,
//! not a flattened diffuse lobe. The mean free paths below are the whole
//! tuning: at this scale (the animal is 1.1 m and its limbs are 50 mm thick)
//! a path of 60 mm is about a limb's width, which is exactly the regime where
//! a part reads as translucent — thick parts stay solid, thin parts light up.

use kosm_render::pathtrace::{Environment, GradientEnv, Pbr, Sun};
use kosm_render::math::Vec3;

/// The palette, keyed by the body names `super::body` writes.
///
/// `None` for a name this file has no opinion about, so the caller can fall
/// through to the cove's own table.
pub fn creature(name: &str) -> Option<Pbr> {
    // One medium, three thicknesses. The colour is a pale aquamarine that is
    // nearly white at the surface and saturates with depth, which is what a
    // per-channel mean free path buys: the blue-green channels travel
    // furthest, so the deep parts of the animal go green and the thin parts
    // go almost white.
    let jelly = |color: [f32; 3], radius: [f64; 3], roughness: f32| Pbr {
        // The surface colour under the specular; the walk carries the rest.
        base_color: color,
        roughness,
        // Wet. A creature just out of the sea has a sharp sky highlight on it
        // and that highlight is half of why it reads as flesh and not wax.
        specular: 0.7,
        ior: 1.38,
        subsurface: 1.0,
        subsurface_color: color,
        subsurface_radius: radius,
        // Forward scattering, as nearly every biological medium is. It makes
        // the thin parts noticeably more transmissive at the same mean free
        // path, which is the effect the limbs are here for.
        subsurface_anisotropy: 0.45,
        ..Default::default()
    };

    Some(match name {
        // The bell. The longest paths in the animal, so it is the part that
        // glows: 78 mm in blue against 44 in red over a 90 mm crown means the
        // sun through it comes out green-blue and the rim goes pale.
        // Saturated rather than pale: `subsurface_color` *is* the surface
        // reflectance the walk is solved to reproduce, so a value near 0.86
        // comes back very nearly white — which is what the second pass at
        // these images did. 0.36/0.72/0.68 is the aquamarine actually wanted.
        "bell" => Pbr {
            // A little transmission, so the bell is also a weak lens in the
            // renderer's eyes and not only in the CAD — enough for a real
            // refracted pool on the door, not so much that the scattering
            // stops carrying the look. The two lobes split by this weight.
            transmission: 0.35,
            ..jelly([0.36, 0.72, 0.68], [44.0, 70.0, 78.0], 0.10)
        },
        // The limbs, at half the bell's path length so that a 50 mm limb is
        // about one mean free path thick and lights up right through.
        "limbs" => jelly([0.44, 0.76, 0.71], [34.0, 52.0, 58.0], 0.14),
        // The gills, shorter again and more saturated: they are the one part
        // that should read as a colour rather than as a glow.
        "gills" => jelly([0.30, 0.66, 0.72], [22.0, 36.0, 42.0], 0.18),
        // The core is dense: 16 mm inside a 300 mm egg is opaque everywhere
        // but the last few millimetres of its edge, which is what gives the
        // animal a solid centre to read the translucency against.
        "core" => jelly([0.66, 0.74, 0.71], [12.0, 16.0, 18.0], 0.30),
        // Near-black and very glossy: the eyes are two specular highlights
        // with a shape behind them, and nothing else.
        "eyes" => Pbr {
            base_color: [0.020, 0.026, 0.032],
            roughness: 0.06,
            specular: 1.0,
            ..Default::default()
        },
        // The heart. Warm against the whole animal's cool, so the glow under
        // the bell is a different colour from the sky that lights it.
        "heart" => Pbr {
            base_color: [0.0; 3],
            emissive: [9.0, 4.2, 1.6],
            ..Default::default()
        },
        // The stage.
        "sand" => flat([0.85, 0.54, 0.22], 0.9),
        "stone" => flat([0.24, 0.20, 0.16], 0.85),
        // The keyhole's throat: it must go black, so that the pool of light
        // the bell throws has something to be bright against.
        "keyhole" => flat([0.012, 0.010, 0.009], 0.95),
        _ => return None,
    })
}

/// The cove's flat style: one colour, one roughness, a weak highlight.
fn flat(base: [f32; 3], roughness: f32) -> Pbr {
    Pbr {
        base_color: base,
        roughness,
        specular: 0.25,
        ..Default::default()
    }
}

/// The same material with its subsurface weight switched off.
///
/// The A/B still is one scene rendered twice, and this is the only difference
/// between the halves — the parameters stay, the transport stops. That is
/// deliberately *not* the same as deleting the fields: what it isolates is
/// the walk, and a surface whose diffuse lobe has taken its share back is the
/// honest thing to compare against.
pub fn without_subsurface(mut pbr: Pbr) -> Pbr {
    pbr.subsurface = 0.0;
    pbr
}

/// The cove's own sky and sun, at the values `sims/rune/render.rs::daylight`
/// uses, so a creature lit here and a creature lit in the cove match.
///
/// The direction is the one free parameter: the cove's sun comes from the
/// scene's own knob, and these stills each want it somewhere different.
pub fn daylight(sun_towards: Vec3) -> (Environment, Sun) {
    let env = Environment::Gradient(GradientEnv {
        zenith: [0.14, 0.30, 0.62],
        horizon: [0.38, 0.58, 0.85],
        ground: [0.42, 0.36, 0.24],
        intensity: 0.42,
    });
    let irr = 6.2f32;
    // Low afternoon light: warm, and warmer the lower it is.
    let sun = Sun::new(sun_towards, 0.02, [irr, 0.77 * irr, 0.46 * irr]);
    (env, sun)
}
