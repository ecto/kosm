//! The material, as the fragment shader reads it.
//!
//! [`GpuMaterial`] and [`library_gpu`] are `kosm::material`'s own — the
//! table both tiers read, so a parity number between them is a statement
//! about the *shading* and not about two different brasses. What is here is
//! the raster tier's small additions:
//!
//! - [`from_rgb`], for a surface that was never a substance: the cove's
//!   opaque sea, the keyhole's dark stone. It spreads a linear RGB triple
//!   over the six bands the way `Spectrum::rgb` does, which
//!   [`band_to_rgb`]'s projection exactly inverts.
//! - [`overrides`], which applies a level's art direction to a copy of the
//!   library rather than to the library itself.
//!
//! ```
//! use kosm_view::raster::material::{library_gpu, albedo_rgb};
//! let (mats, by_name) = library_gpu();
//! let brass = mats[by_name["brass"] as usize];
//! assert_eq!(brass.metallic, 1.0, "brass is a conductor");
//! // and the six bands project to the RGB the tracer's own `Pbr` carries
//! let want = kosm::material::named("brass").unwrap().pbr().base_color;
//! assert!((albedo_rgb(&brass)[0] - want[0]).abs() < 1e-5);
//! ```

pub use kosm::material::{BANDS, GpuMaterial, band_to_rgb, bands_to_rgb, library_gpu};

/// Clay: a flat mid-grey dielectric. What a surface whose name nothing knows
/// is drawn as, and what an index past the end of the table would have been.
pub fn clay() -> GpuMaterial {
    from_rgb([0.5, 0.5, 0.5], 0.8, 0.25)
}

/// An authored linear RGB colour as a material.
///
/// The six bands are `Spectrum::rgb`'s spread — each primary over the two
/// bands it owns — so [`band_to_rgb`]'s projection gives the triple back
/// unchanged. It is how a `Pbr` that was never a substance reaches the same
/// shader path as one that was.
pub fn from_rgb(rgb: [f32; 3], roughness: f32, specular: f32) -> GpuMaterial {
    GpuMaterial {
        albedo: [rgb[2], rgb[2], rgb[1], rgb[1], rgb[0], rgb[0], 0.0, 0.0],
        emission: [0.0; 8],
        roughness,
        metallic: 0.0,
        specular,
        transmission: 0.0,
        ior: 1.5,
        film_nm: 0.0,
        film_ior: 1.3,
        sss_weight: 0.0,
        sss_radius_m: [0.0; 3],
        sss_aniso: 0.0,
    }
}

/// The same material with an emitted radiance, linear RGB.
pub fn emitting(mut m: GpuMaterial, rgb: [f32; 3]) -> GpuMaterial {
    m.emission = [rgb[2], rgb[2], rgb[1], rgb[1], rgb[0], rgb[0], 0.0, 0.0];
    m
}

/// The same material repainted: a new albedo, a new roughness, a new
/// dielectric highlight, and **everything else left alone**.
///
/// This is the shape a level's override layer takes. `sims/rune/materials.rs`
/// does exactly this to `Pbr` — "the substance is the library's and the look
/// is the level's" — and a raster tier that repainted by building a fresh
/// material would quietly drop the index, the film and the subsurface walk
/// with it.
pub fn repaint(m: GpuMaterial, rgb: [f32; 3], roughness: f32, specular: f32) -> GpuMaterial {
    GpuMaterial {
        albedo: [rgb[2], rgb[2], rgb[1], rgb[1], rgb[0], rgb[0], 0.0, 0.0],
        roughness,
        specular,
        ..m
    }
}

/// The albedo projected to linear RGB, which is `Pbr::base_color`.
pub fn albedo_rgb(m: &GpuMaterial) -> [f32; 3] {
    bands_to_rgb(&m.bands())
}

/// A level's override layer applied to a copy of the library.
///
/// `f` is asked for every canonical name in turn and hands back the material
/// that name should draw as; `None` leaves the library's own. The point of
/// the indirection is that the list of overrides lives in **one** function in
/// the level, next to the `Pbr` overrides the tracer reads, so the two cannot
/// drift — see `sims/rune/materials.rs::gpu_cove`.
pub fn overrides(
    mut mats: Vec<GpuMaterial>,
    by_name: &std::collections::HashMap<String, u32>,
    f: impl Fn(&str, GpuMaterial) -> Option<GpuMaterial>,
) -> Vec<GpuMaterial> {
    for (name, i) in by_name {
        if let Some(m) = f(name, mats[*i as usize]) {
            mats[*i as usize] = m;
        }
    }
    mats
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The two facets agree.** Every entry in the library projects to the
    /// same `base_color`, `roughness`, `metallic` and `ior` the tracer's own
    /// `pbr()` carries.
    #[test]
    fn the_gpu_facet_projects_to_the_pbr_one() {
        let (mats, by_name) = library_gpu();
        assert_eq!(by_name.len(), kosm::material::names().len(), "every name is covered");
        for name in kosm::material::names() {
            let m = kosm::material::named(name).expect("a listed name is a material");
            let p = m.pbr();
            let g = mats[by_name[*name] as usize];
            let rgb = albedo_rgb(&g);
            for c in 0..3 {
                assert!(
                    (rgb[c] - p.base_color[c]).abs() < 1e-5,
                    "{name}: the gpu albedo projects to {rgb:?}, the pbr says {:?}",
                    p.base_color
                );
            }
            assert!((g.roughness - p.roughness).abs() < 1e-6, "{name} roughness");
            assert!((g.metallic - p.metallic).abs() < 1e-6, "{name} metallic");
            assert!((g.transmission - p.transmission).abs() < 1e-6, "{name} transmission");
            assert!((g.film_nm - p.thin_film_thickness).abs() < 1e-6, "{name} film");
        }
    }

    /// `from_rgb` and the projection are exact inverses, which is what lets an
    /// authored colour that was never a substance reach the same shader path.
    #[test]
    fn an_authored_colour_round_trips_through_the_bands() {
        for rgb in [[0.85f32, 0.54, 0.22], [0.13, 0.19, 0.33], [1.0, 1.0, 1.0]] {
            let got = albedo_rgb(&from_rgb(rgb, 0.9, 0.25));
            for c in 0..3 {
                assert!((got[c] - rgb[c]).abs() < 1e-6, "{rgb:?} came back {got:?}");
            }
        }
    }

    /// A repaint changes the look and keeps the substance: the index, the
    /// film and the subsurface walk survive it.
    #[test]
    fn a_repaint_keeps_everything_it_did_not_name() {
        let (mats, by_name) = library_gpu();
        let skin = mats[by_name["skin"] as usize];
        assert!(skin.sss_weight > 0.0, "skin scatters");
        let flat = repaint(skin, [0.9, 0.7, 0.6], 0.4, 0.25);
        assert_eq!(flat.sss_weight, skin.sss_weight);
        assert_eq!(flat.ior, skin.ior);
        assert!((albedo_rgb(&flat)[0] - 0.9).abs() < 1e-6);
        assert!((flat.roughness - 0.4).abs() < 1e-6);
    }

    /// The override layer reaches every name it is asked about and nothing
    /// else.
    #[test]
    fn the_override_layer_touches_only_what_it_names() {
        let (mats, by_name) = library_gpu();
        let out = overrides(mats.clone(), &by_name, |name, m| {
            (name == "granite").then(|| repaint(m, [0.24, 0.20, 0.16], 0.85, 0.25))
        });
        let g = by_name["granite"] as usize;
        assert!((albedo_rgb(&out[g])[0] - 0.24).abs() < 1e-6);
        for (i, (a, b)) in mats.iter().zip(out.iter()).enumerate() {
            if i != g {
                assert_eq!(a, b, "the override touched something it did not name");
            }
        }
    }

    /// The layout the shader binds: 112 bytes, with the `vec3` on its own
    /// sixteen-byte boundary, which is where `#[repr(C)]` puts it too.
    #[test]
    fn the_layout_is_what_the_shader_declares() {
        assert_eq!(std::mem::size_of::<GpuMaterial>(), 112);
        assert_eq!(std::mem::offset_of!(GpuMaterial, sss_radius_m), 96);
    }
}
