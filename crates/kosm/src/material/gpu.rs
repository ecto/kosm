//! The shader's facet: one substance as the bytes a raster tier binds.
//!
//! [`Material::pbr`](super::Material::pbr) is what the path tracer reads;
//! this is what the live tier reads, and the two come out of the same
//! constants. Every scalar here is taken *from* `pbr()` rather than authored
//! beside it — the one place the two facets could drift is the one place
//! there is no second table — and the only field that is not is the albedo,
//! which stays spectral because the probe volume the shader multiplies it by
//! is spectral. [`band_to_rgb`] is the projection that closes the loop:
//! `gpu().albedo` through it is exactly `pbr().base_color`.
//!
//! ```
//! use kosm::material::{self, band_to_rgb};
//!
//! let brass = material::named("brass").unwrap();
//! let g = brass.gpu();
//! let m = band_to_rgb();
//! let mut rgb = [0.0f32; 3];
//! for b in 0..material::BANDS {
//!     for c in 0..3 {
//!         rgb[c] += g.albedo[b] * m[b][c];
//!     }
//! }
//! assert!((rgb[0] - brass.pbr().base_color[0]).abs() < 1e-5);
//! ```

use std::collections::HashMap;

use super::{BANDS, Material, Spectrum};

/// One substance as a shader reads it: `#[repr(C)]`, `Pod`, and the same
/// numbers [`Material::pbr`](super::Material::pbr) hands the tracer.
///
/// The two `[f32; 8]` spectra are six bands and two zeros, so the struct
/// lands on 16-byte boundaries in a storage buffer without a hand-written
/// pad, and a `vec4`-shaped read of `albedo` picks up bands 0..4 exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
    /// Reflectance per band at [`BANDS_NM`](super::BANDS_NM), then two
    /// zeros. `F0` under [`Optics::Conductor`], as everywhere else.
    pub albedo: [f32; 8],
    /// Emitted radiance per band, then two zeros. All zeros if the substance
    /// does not emit.
    pub emission: [f32; 8],
    /// Perceptual roughness, 0..1.
    pub roughness: f32,
    /// 0 for a dielectric, 1 for a metal.
    pub metallic: f32,
    /// Disney's normalised incident specular; 0.5 means `F0 = 0.04`.
    pub specular: f32,
    /// Weight of the transmission lobe, 0..1.
    pub transmission: f32,
    /// Index at the d line.
    pub ior: f32,
    /// Thin-film thickness, nanometres. Zero is no film.
    pub film_nm: f32,
    /// The film's own index.
    pub film_ior: f32,
    /// Weight of the subsurface term, 0..1. Zero switches it off.
    pub sss_weight: f32,
    /// Mean free path per RGB channel, metres.
    pub sss_radius_m: [f32; 3],
    /// Henyey–Greenstein `g` of the medium.
    pub sss_aniso: f32,
}

impl Default for GpuMaterial {
    fn default() -> Self {
        Material::default().gpu()
    }
}

/// How a [`Spectrum`]'s bands project to linear RGB, as the matrix a shader
/// multiplies by: `rgb[c] = Σ_b bands[b] · band_to_rgb()[b][c]`.
///
/// This *is* [`Spectrum::to_rgb`] — the mean of each primary's two bands —
/// written as a matrix so the raster tier and the tracer cannot drift. The
/// probe volume's irradiance is per band, the material's albedo is per band,
/// and the product goes through here on its way to the film.
pub fn band_to_rgb() -> [[f32; 3]; BANDS] {
    let mut m = [[0.0f32; 3]; BANDS];
    for b in 0..BANDS {
        let mut bands = [0.0f64; BANDS];
        bands[b] = 1.0;
        let rgb = Spectrum::new(bands).to_rgb();
        m[b] = [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32];
    }
    m
}

/// Project a band vector to linear RGB through [`band_to_rgb`].
pub fn bands_to_rgb(bands: &[f32; BANDS]) -> [f32; 3] {
    let m = band_to_rgb();
    let mut rgb = [0.0f32; 3];
    for b in 0..BANDS {
        for (c, out) in rgb.iter_mut().enumerate() {
            *out += bands[b] * m[b][c];
        }
    }
    rgb
}

impl Material {
    /// The shader's facet. See [`GpuMaterial`].
    ///
    /// Everything but the two spectra is read off [`Material::pbr`], so a
    /// field that changes there changes here in the same edit.
    pub fn gpu(&self) -> GpuMaterial {
        let p = self.pbr();
        let spread = |s: &Spectrum| {
            let mut out = [0.0f32; 8];
            for i in 0..BANDS {
                out[i] = s.bands[i] as f32;
            }
            out
        };
        GpuMaterial {
            albedo: spread(&self.albedo),
            emission: self.emission.as_ref().map(spread).unwrap_or([0.0; 8]),
            roughness: p.roughness,
            metallic: p.metallic,
            specular: p.specular,
            transmission: p.transmission,
            ior: p.ior,
            film_nm: p.thin_film_thickness,
            film_ior: p.thin_film_ior,
            sss_weight: p.subsurface,
            sss_radius_m: [
                p.subsurface_radius[0] as f32,
                p.subsurface_radius[1] as f32,
                p.subsurface_radius[2] as f32,
            ],
            sss_aniso: p.subsurface_anisotropy,
        }
    }
}

/// The whole library as a shader buffer, and the index of every canonical
/// name in it.
///
/// The order is [`names`](super::names)' order, so the buffer is stable
/// between runs and a baked scene may hold indices rather than strings. The
/// map is keyed by the canonical name exactly as the library spells it;
/// aliases resolve through [`named`](super::named) and are not listed.
pub fn library_gpu() -> (Vec<GpuMaterial>, HashMap<String, u32>) {
    let mut buf = Vec::with_capacity(super::names().len());
    let mut index = HashMap::with_capacity(super::names().len());
    for name in super::names() {
        let Some(m) = super::named(name) else { continue };
        index.insert((*name).to_string(), buf.len() as u32);
        buf.push(m.gpu());
    }
    (buf, index)
}

/// Whether a substance transmits, for a tier that sorts its draws.
impl GpuMaterial {
    /// The albedo's six bands, without the alignment zeros.
    pub fn bands(&self) -> [f32; BANDS] {
        let mut out = [0.0f32; BANDS];
        out.copy_from_slice(&self.albedo[..BANDS]);
        out
    }

    /// The albedo as linear RGB — what [`Material::pbr`]'s `base_color` is.
    pub fn base_color(&self) -> [f32; 3] {
        bands_to_rgb(&self.bands())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gpu_albedo_projects_to_the_tracers_base_colour() {
        for name in super::super::names() {
            let m = super::super::named(name).expect("a listed name is in the library");
            let (g, p) = (m.gpu(), m.pbr());
            let rgb = g.base_color();
            for c in 0..3 {
                assert!(
                    (rgb[c] - p.base_color[c]).abs() < 1e-5,
                    "{name}: gpu albedo projects to {rgb:?}, pbr says {:?}",
                    p.base_color
                );
            }
            assert_eq!(g.albedo[6], 0.0);
            assert_eq!(g.albedo[7], 0.0);
            assert!(matches!(m.optics, super::super::Optics::Conductor) == (g.metallic == 1.0));
        }
    }

    #[test]
    fn the_library_is_every_canonical_name() {
        let (buf, index) = library_gpu();
        assert_eq!(buf.len(), super::super::names().len());
        for name in super::super::names() {
            let i = *index.get(*name).unwrap_or_else(|| panic!("{name} is in the map"));
            let m = super::super::named(name).unwrap();
            assert_eq!(buf[i as usize], m.gpu(), "{name} is at its own index");
        }
    }

    #[test]
    fn the_struct_is_plain_old_data() {
        let g = Material::default().gpu();
        let bytes = bytemuck::bytes_of(&g);
        assert_eq!(bytes.len(), 28 * 4, "no padding anywhere in it");
        let back: GpuMaterial = *bytemuck::from_bytes(bytes);
        assert_eq!(back, g);
    }
}
