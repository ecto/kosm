//! The raster tier: the baked answer, drawn stably, sixty times a second.
//!
//! `docs/plans/2026-09-10-bake-and-raster-design.md` is the why. The claim in
//! one sentence: a level whose sun is fixed and whose geometry is static does
//! not need its illumination re-solved by Monte Carlo every frame — the path
//! tracer should solve it once into [`probes::ProbeVolume`], and a rasterizer
//! should draw that answer without the noise.
//!
//! What is here is a plain forward renderer with one unusual thing in its
//! fragment shader: the indirect term is a **spectral** SH probe read, six
//! bands wide, multiplied by the material's six-band albedo and only then
//! projected to RGB. That is what makes the two tiers comparable — the
//! tracer's own transport is spectral, so a raster tier that shaded in RGB
//! would disagree with it on every saturated surface and the parity number
//! would be measuring the colour space rather than the shading.
//!
//! ```text
//! shadow pass   2048² depth along the sun, orthographic over the level
//! depth prepass the camera's own depth, geometry only
//! ao pass       half-res hemisphere occlusion off it, two bilateral blurs
//! sky pass      Preetham's clear sky, and the sun's disc, into linear HDR
//! scene pass    per fragment:
//!                 direct   = sun irradiance × (Lambert + GGX) × PCSS shadow
//!                          + the caustic quads' irradiance where they land
//!                 indirect = probes.sample(sun, p, n) × ao       [6 bands]
//!                 colour   = band_to_rgb(albedo × (direct + indirect))
//!                          + emission + the instance's own glow
//!                 air      = mix(colour, sky(view), 1 − e^{−τ})
//! post pass     exposure, cos⁴ vignette, bloom, ACES, sRGB, out
//! ```
//!
//! ## the light is the level's, and both tiers read the same struct
//!
//! [`Scene::sky`] is `kosm_render::env::SkyEnv` itself, [`Scene::air`] is
//! `kosm_render::post::Aerial`, and the vignette, the bloom and the tonemap
//! are `kosm_render::post::Post` — the same values the path tracer resolves
//! its own frame through. That is not tidiness: the settle blend fades one
//! frame into the other in code space, so a sky, a haze or a tonemap that
//! differed by a per cent would show as the picture shifting under a standing
//! player. The WGSL ports live in `shaders/scene.wgsl` and `shaders/post.wgsl`
//! and a GPU test holds each to its Rust original.
//!
//! Every one of those is **off** in [`Scene::new`]. A level asks for a sky, a
//! haze, an occlusion, a vignette and a bloom or it gets the picture this tier
//! drew before any of them existed — which is what keeps the datasheet ball's
//! parity numbers measuring the shading and not the art direction.
//!
//! **The volume it is handed must not carry the sun's direct term.** The
//! shader computes `E · max(0, n·s)` itself, per pixel, against the shadow
//! map — which resolves the hero's own shadow and the jamb of a door, neither
//! of which a lattice half a metre across can — so a bake that also put the
//! direct term in the SH would be counted twice. On the cove's sunlit sand
//! that is a factor of one and eight tenths, and it is a whole tenth of the
//! parity number. `kosm::light::probes::BakeSpec::sun_direct` is the switch,
//! `sims/rune/bake.rs` is where it is set false, and the sky and every
//! *bounce* of the sun stay in the volume either way. A volume with no sun at
//! all — `studio_probes`, the sky-only fallback — is drawn with
//! [`Sun::irradiance`] zero and the same shader is exact.
//!
//! ## what it does not do
//!
//! **It does not refract.** A transmissive material is drawn as its Fresnel
//! rim over the sky and the probes behind it; the being's glass and the
//! hero's lens are *shaped* right and are not the reference's picture of
//! them. That is the whole reason [`Settle`] exists: stand still and the path
//! tracer's frame is blended over this one, and what you end up looking at is
//! the reference.
//!
//! **The fisheye is tracer-only.** [`View::projection`](crate::Projection)'s
//! equidistant map is `r = f·θ`, which is not a projective transform: no
//! 4×4 puts it in a vertex shader, and a post-pass warp of a wider
//! rectilinear render cannot reach past 180° or hold the corners at the same
//! sample density. So [`Frame::projection`] takes `Rectilinear` only, and a
//! level asking for the fisheye falls back to the tracer for the whole frame
//! — [`Raster::supports`] is the predicate, and `sims/rune/game.rs` prints
//! the reason.
//!
//! ## the modules
//!
//! - [`probes`] — the spectral SH volume, and the cosine convolution that is
//!   the whole lighting model. Also the raster tier's copy of the bake
//!   agent's interface, until that lands.
//! - [`material`] — [`material::GpuMaterial`] and `library_gpu()`, ditto.
//! - [`shadow`] — the sun's orthographic frustum over a bounding box.
//! - [`water`] — the sea's knobs, as the shader's analytic surface.
//! - [`pipeline`] — the wgpu resources and the eight passes.
//! - [`settle`] — the blend that turns this picture into the reference's when
//!   nothing is moving, in bytes.
//! - [`blend`] — the same mix on the device, so a standing frame crosses the
//!   bus no more often than a walking one does.

pub mod blend;
pub mod material;
pub mod pipeline;
pub mod probes;
pub mod settle;
pub mod shadow;
pub mod water;

pub use blend::Blend;
pub use material::{GpuMaterial, band_to_rgb, library_gpu};
pub use pipeline::{Frame, Raster};
pub use probes::ProbeVolume;
pub use settle::{Settle, SettleReport};
pub use water::Sea;

use probes::BANDS;

/// One vertex: a position and a normal, both in world metres.
///
/// There is no per-vertex colour and no texture coordinate. A surface's
/// appearance is entirely its instance's material index, which is the whole
/// point of the material library being the contract between the tiers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
}

/// One drawn copy of a mesh: where it is, what it is made of, and how it is
/// shaded.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    /// Object → world, column-major, metres.
    pub model: [f32; 16],
    /// Index into the material storage buffer.
    pub material: u32,
    /// Which shading branch: see [`Kind`].
    pub kind: u32,
    /// Whether the shadow pass draws this instance. The sea does not (a
    /// shadow map of a swell is a field of acne) and neither does the lens
    /// disc.
    pub casts: u32,
    pub _pad: u32,
    /// Radiance added straight to the pixel, linear RGB. What the keyhole's
    /// rim and the sand's glint are: their brightness is the *score*, which
    /// is a number the simulation carries and not a property of stone.
    pub glow: [f32; 4],
}

impl Default for Instance {
    fn default() -> Self {
        Self {
            model: IDENTITY,
            material: 0,
            kind: Kind::Solid as u32,
            casts: 1,
            _pad: 0,
            glow: [0.0; 4],
        }
    }
}

/// A column-major 4×4 identity.
pub const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

impl Instance {
    /// An instance at a rigid placement: a rotation as three world columns
    /// and a translation, both in metres.
    pub fn rigid(rot: [[f64; 3]; 3], t: [f64; 3], material: u32) -> Self {
        // `rot[c]` is the world direction of the object's own axis `c`, which
        // is exactly a column of the object → world matrix, and column-major
        // storage puts a column in four contiguous floats.
        let mut m = IDENTITY;
        for c in 0..3 {
            for r in 0..3 {
                m[c * 4 + r] = rot[c][r] as f32;
            }
        }
        m[12] = t[0] as f32;
        m[13] = t[1] as f32;
        m[14] = t[2] as f32;
        Self { model: m, material, ..Self::default() }
    }

    /// An instance from a row-major 4×4 in **millimetres** — which is the
    /// form every vcad placement is in — transposed into the column-major
    /// **metres** the shader wants.
    pub fn from_mm_rows(rows: [[f64; 4]; 4], material: u32) -> Self {
        let mut m = [0.0f32; 16];
        for c in 0..4 {
            for r in 0..4 {
                m[c * 4 + r] = rows[r][c] as f32;
            }
        }
        // only the translation carries the unit; the linear part is a ratio
        for k in [12usize, 13, 14] {
            m[k] *= 1e-3;
        }
        Self { model: m, material, ..Self::default() }
    }

    pub fn with_kind(mut self, kind: Kind) -> Self {
        self.kind = kind as u32;
        self
    }

    pub fn with_glow(mut self, rgb: [f32; 3]) -> Self {
        self.glow = [rgb[0], rgb[1], rgb[2], 0.0];
        self
    }

    pub fn casting(mut self, on: bool) -> Self {
        self.casts = u32::from(on);
        self
    }

    /// The instance's world-space translation, metres.
    pub fn origin(&self) -> [f32; 3] {
        [self.model[12], self.model[13], self.model[14]]
    }
}

/// Which branch of the fragment shader a surface takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Kind {
    /// The ordinary path: sun, GGX, probes, caustics.
    Solid = 0,
    /// The sea: an analytic swell normal, a Fresnel split between the sky and
    /// the refracted seabed, and a horizon fade. See [`water`].
    Sea = 1,
    /// The lens: a thin disc with a view-dependent Fresnel highlight and the
    /// sky tinted through it. True refraction is the reference tier's.
    Lens = 2,
}

/// One mesh, tessellated once.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    /// Every copy of it drawn this frame.
    pub instances: Vec<Instance>,
}

impl Mesh {
    /// A mesh from positions, normals and indices, in **millimetres** — the
    /// units vcad authors in and `sims/rune/render.rs` assembles in.
    pub fn from_mm(positions: &[[f64; 3]], normals: &[[f64; 3]], indices: &[u32]) -> Self {
        Self::from_units(positions, normals, indices, 1e-3)
    }

    /// The same in metres.
    pub fn from_m(positions: &[[f64; 3]], normals: &[[f64; 3]], indices: &[u32]) -> Self {
        Self::from_units(positions, normals, indices, 1.0)
    }

    fn from_units(
        positions: &[[f64; 3]],
        normals: &[[f64; 3]],
        indices: &[u32],
        scale: f64,
    ) -> Self {
        let smooth;
        let normals = if normals.len() == positions.len() {
            normals
        } else {
            smooth = smooth_normals(positions, indices);
            &smooth
        };
        let mut vertices = Vec::with_capacity(indices.len());
        for i in indices {
            let Some(p) = positions.get(*i as usize) else { continue };
            let n = normals.get(*i as usize).copied().unwrap_or([0.0, 0.0, 1.0]);
            vertices.push(Vertex {
                pos: [(p[0] * scale) as f32, (p[1] * scale) as f32, (p[2] * scale) as f32],
                nrm: [n[0] as f32, n[1] as f32, n[2] as f32],
            });
        }
        Self { vertices, instances: Vec::new() }
    }

    /// How many triangles it draws.
    pub fn tris(&self) -> usize {
        self.vertices.len() / 3
    }
}

/// Area-weighted vertex normals, welded at a millimetre.
///
/// The same construction `sims/rune/hero/stage.rs::smooth_normals` uses, kept
/// here so a mesh handed over without normals still shades smoothly — a
/// tessellated cliff with facet normals reads as a low-poly mountain, which
/// is not what the tracer draws.
pub fn smooth_normals(positions: &[[f64; 3]], indices: &[u32]) -> Vec<[f64; 3]> {
    use std::collections::HashMap;
    let key = |p: &[f64; 3]| {
        let q = |v: f64| (v * 1e3).round() as i64;
        (q(p[0]), q(p[1]), q(p[2]))
    };
    let mut sum: HashMap<(i64, i64, i64), [f64; 3]> = HashMap::new();
    for t in indices.chunks_exact(3) {
        let (Some(a), Some(b), Some(c)) = (
            positions.get(t[0] as usize),
            positions.get(t[1] as usize),
            positions.get(t[2] as usize),
        ) else {
            continue;
        };
        let (u, v) = (sub(*b, *a), sub(*c, *a));
        // the *unnormalised* cross is area-weighted, which is the weighting a
        // tessellation with wildly different triangle sizes wants
        let n = cross(u, v);
        for i in t {
            if let Some(p) = positions.get(*i as usize) {
                let e = sum.entry(key(p)).or_insert([0.0; 3]);
                for k in 0..3 {
                    e[k] += n[k];
                }
            }
        }
    }
    positions
        .iter()
        .map(|p| {
            let n = sum.get(&key(p)).copied().unwrap_or([0.0, 0.0, 1.0]);
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if l > 1e-12 { [n[0] / l, n[1] / l, n[2] / l] } else { [0.0, 0.0, 1.0] }
        })
        .collect()
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The one direct light: the level's sun, in the tracer's own units.
#[derive(Clone, Copy, Debug)]
pub struct Sun {
    /// Unit vector **toward** the sun.
    pub direction: [f64; 3],
    /// Irradiance per band, the same units the probes are baked in.
    pub irradiance: [f32; BANDS],
    /// The disc's angular radius, radians. It softens the terminator and
    /// widens the GGX highlight; it is the level's `sun_angular_radius`.
    pub angular_radius: f64,
}

impl Sun {
    /// A sun from a direction and a linear RGB irradiance — the form
    /// `sims/rune/render.rs::daylight` states it in.
    pub fn from_rgb(direction: [f64; 3], rgb: [f32; 3], angular_radius: f64) -> Self {
        Self {
            direction,
            // `Spectrum::rgb`'s spread: each primary over the two bands it owns
            irradiance: [rgb[2], rgb[2], rgb[1], rgb[1], rgb[0], rgb[0]],
            angular_radius,
        }
    }
}

/// A rectangle of a receiver plane carrying baked irradiance: the caustic,
/// where it lands.
///
/// Two of them are what the cove needs — the door's face and a patch of sand
/// around the focus — and the shader tests every fragment against both, which
/// is three dot products and is cheaper than deciding per instance which one
/// a surface belongs to.
#[derive(Clone, Debug)]
pub struct CausticQuad {
    /// The rectangle's corner, world metres.
    pub origin: [f64; 3],
    /// The `u` edge: direction times the rectangle's width in metres.
    pub u: [f64; 3],
    /// The `v` edge, likewise.
    pub v: [f64; 3],
    /// `res.0 × res.1` irradiance samples, row-major along `u`.
    pub res: (u32, u32),
    /// Linear RGB irradiance per texel, `res.0 * res.1 * 3` long.
    pub data: Vec<f32>,
}

impl CausticQuad {
    /// An empty quad of the given resolution: no caustic, but the same
    /// binding, so a frame with no map does not need a different pipeline.
    pub fn empty(res: (u32, u32)) -> Self {
        Self {
            origin: [0.0; 3],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            res,
            data: vec![0.0; (res.0 * res.1) as usize * 3],
        }
    }

    /// Gather a [`kosm_render::caustics::CausticMap`] onto this rectangle.
    ///
    /// This is `CausticMap::to_texture` from the design's interface list,
    /// written against the map's own public `irradiance(p, n)` so the raster
    /// tier does not have to wait for the bake agent to land it. When that
    /// method exists this becomes a call to it; the numbers are the same
    /// either way, because it is the same density estimate read at the same
    /// points.
    pub fn gather(
        map: &kosm_render::caustics::CausticMap,
        origin: [f64; 3],
        u: [f64; 3],
        v: [f64; 3],
        res: (u32, u32),
    ) -> Self {
        use rayon::prelude::*;
        let n = unit(cross(u, v));
        let mut data = vec![0.0f32; (res.0 * res.1) as usize * 3];
        // **A row a task.** A 128² receiver is sixteen thousand density
        // estimates and there are two of them, which a walking player pays
        // every frame the lens moves far enough to be worth retracing. Done
        // in one thread it was thirty milliseconds and it was the raster
        // tier's whole frame budget; the estimates do not talk to each other,
        // so it is a `par_chunks_mut` and nothing else.
        if !map.is_empty() {
            data.par_chunks_mut(res.0 as usize * 3).enumerate().for_each(|(j, row)| {
                let tv = (j as f64 + 0.5) / res.1 as f64;
                for i in 0..res.0 as usize {
                    let tu = (i as f64 + 0.5) / res.0 as f64;
                    let p = kosm_render::math::Point3::new(
                        origin[0] + u[0] * tu + v[0] * tv,
                        origin[1] + u[1] * tu + v[1] * tv,
                        origin[2] + u[2] * tu + v[2] * tv,
                    );
                    let e = map.irradiance(p, kosm_render::math::Vec3::new(n[0], n[1], n[2]));
                    row[i * 3..i * 3 + 3].copy_from_slice(&e);
                }
            });
        }
        Self { origin, u, v, res, data }
    }

    /// The plane's unit normal.
    pub fn normal(&self) -> [f64; 3] {
        unit(cross(self.u, self.v))
    }
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-12 { [v[0] / n, v[1] / n, v[2] / n] } else { [0.0, 0.0, 1.0] }
}

/// The world box a probe lattice covers: the corner of cell `(0,0,0)` and the
/// corner of the last one. The shadow map's frustum defaults to it, because a
/// bake that covered the playable volume is the honest statement of what can
/// cast into the frame.
pub fn volume_bounds(v: &ProbeVolume) -> ([f64; 3], [f64; 3]) {
    let hi = [
        v.origin[0] + v.spacing * v.dims[0].saturating_sub(1) as f64,
        v.origin[1] + v.spacing * v.dims[1].saturating_sub(1) as f64,
        v.origin[2] + v.spacing * v.dims[2].saturating_sub(1) as f64,
    ];
    (v.origin, hi)
}

/// Everything the tier draws, and the light it draws it in.
///
/// Built once from the level and then mutated per frame: the meshes and the
/// material buffer do not change, the instances do.
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<GpuMaterial>,
    pub sun: Sun,
    pub probes: ProbeVolume,
    /// Which of the volume's baked suns this level is under, as a fractional
    /// index. A fixed sun is zero.
    pub sun_index: f64,
    pub sea: Option<Sea>,
    /// The two receiver rectangles the caustic is gathered onto.
    pub caustics: Vec<CausticQuad>,
    /// The volume the shadow map's orthographic frustum covers, world metres.
    pub bounds: ([f64; 3], [f64; 3]),
    /// The authored exposure, before the meter's multiplier.
    pub exposure: f32,
    /// The analytic sky the level hangs under, if it has one.
    ///
    /// [`None`] is what this tier did before there was a sky model: every
    /// direction lookup falls back to the probe volume's own low-frequency
    /// read and the background is the horizon's colour. The datasheet ball is
    /// still drawn that way, which is what keeps its parity numbers pinned to
    /// the studio rig they were measured under.
    ///
    /// It is [`kosm_render::env::SkyEnv`] itself and not a copy of its
    /// fields: the tracer holds the same value, so the two tiers cannot drift
    /// apart about what the sky is.
    pub sky: Option<kosm_render::env::SkyEnv>,
    /// The haze. `density` zero — the default — is no aerial perspective.
    pub air: kosm_render::post::Aerial,
    /// The world radius the screen-space occlusion looks for geometry in,
    /// metres, and how much of what it finds is applied.
    pub ao_radius_m: f32,
    pub ao_strength: f32,
    /// How much of the lens's `cos⁴` falloff the post pass applies, 0 to 1.
    pub vignette: f32,
    /// The bloom: the exposed luminance a pixel has to pass to bleed, what
    /// fraction of the blur comes back, and its σ in full-size pixels.
    pub bloom_threshold: f32,
    pub bloom_strength: f32,
    pub bloom_radius_px: f32,
}

impl Scene {
    /// A scene with the library's materials, one sun, and nothing in it.
    ///
    /// The map back is the library's own index: `by_name["dry sand"]` is the
    /// slot the beach draws out of, and it is the same slot the bake wrote
    /// and the tracer reads. A level that repaints a name does it with
    /// [`material::overrides`] on the returned [`Scene::materials`], not by
    /// pushing a second entry — see `sims/rune/materials.rs::gpu_cove`.
    pub fn new(sun: Sun, probes: ProbeVolume) -> (Self, std::collections::HashMap<String, u32>) {
        let (materials, by_name) = library_gpu();
        let bounds = volume_bounds(&probes);
        (
            Self {
                meshes: Vec::new(),
                materials,
                sun,
                probes,
                sun_index: 0.0,
                sea: None,
                caustics: Vec::new(),
                bounds,
                exposure: 0.7,
                // Everything a level has to ask for is off here, so a scene
                // built without a word about light draws exactly what this
                // tier drew before any of it existed.
                sky: None,
                air: kosm_render::post::Aerial::default(),
                ao_radius_m: 0.3,
                ao_strength: 0.0,
                vignette: 0.0,
                bloom_threshold: 1.0,
                bloom_strength: 0.0,
                bloom_radius_px: 6.0,
            },
            by_name,
        )
    }

    /// Add a material the library does not have a name for — the cove's
    /// opaque sea, the rim's dark stone — and get its index.
    pub fn push_material(&mut self, m: GpuMaterial) -> u32 {
        self.materials.push(m);
        self.materials.len() as u32 - 1
    }

    /// Add a mesh and get its index.
    pub fn push_mesh(&mut self, mesh: Mesh) -> usize {
        self.meshes.push(mesh);
        self.meshes.len() - 1
    }

    /// How many triangles the whole scene is.
    pub fn tris(&self) -> usize {
        self.meshes.iter().map(Mesh::tris).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rigid instance's columns are the object's axes in the world, which
    /// is what a column-major model matrix means — and what the vertex shader
    /// multiplies a normal by.
    #[test]
    fn a_rigid_instance_puts_the_axes_in_the_columns() {
        // a quarter turn about +z: the object's +x is the world's +y
        let i = Instance::rigid([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [1.0, 2.0, 3.0], 7);
        assert_eq!(&i.model[0..3], &[0.0, 1.0, 0.0]);
        assert_eq!(&i.model[4..7], &[-1.0, 0.0, 0.0]);
        assert_eq!(i.origin(), [1.0, 2.0, 3.0]);
        assert_eq!(i.material, 7);
    }

    /// A vcad placement is row-major millimetres; the shader wants
    /// column-major metres, and only the translation carries the unit.
    #[test]
    fn a_vcad_placement_transposes_and_converts_only_its_translation() {
        let rows = [
            [0.0, -1.0, 0.0, 1000.0],
            [1.0, 0.0, 0.0, 2000.0],
            [0.0, 0.0, 1.0, 3000.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let i = Instance::from_mm_rows(rows, 3);
        // millimetres over a thousand, in `f32`: three metres is 3.0000002
        for (got, want) in i.origin().iter().zip([1.0f32, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-6, "{:?} is not {want}", i.origin());
        }
        // column 0 is the object's +x in the world: (0, 1, 0), still unit
        assert_eq!(&i.model[0..3], &[0.0, 1.0, 0.0]);
    }

    /// Welded, area-weighted normals: a flat square's are all +z, and the
    /// shared edge of two triangles gets one normal and not two.
    #[test]
    fn smooth_normals_weld_a_shared_edge() {
        let p = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let n = smooth_normals(&p, &[0, 1, 2, 0, 2, 3]);
        for v in &n {
            assert!((v[2] - 1.0).abs() < 1e-9, "a flat square points up, not {v:?}");
        }
    }

    /// A mesh with no normals gets smoothed ones, and the unit conversion is
    /// applied to the positions and to nothing else.
    #[test]
    fn a_mesh_in_millimetres_arrives_in_metres() {
        let p = [[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [1000.0, 1000.0, 0.0]];
        let m = Mesh::from_mm(&p, &[], &[0, 1, 2]);
        assert_eq!(m.tris(), 1);
        assert_eq!(m.vertices[1].pos, [1.0, 0.0, 0.0]);
        assert!((m.vertices[0].nrm[2] - 1.0).abs() < 1e-6);
    }
}
