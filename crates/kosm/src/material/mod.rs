//! `Material`: one substance, and every facet derived from it.
//!
//! The claim is one noun. A [`Material`] is a *substance* — brass, wet sand,
//! N-BK7 — held as physical constants, and everything a solver wants is
//! **derived** from those constants rather than authored beside them. There
//! is no second table where the renderer's brass and the contact solver's
//! brass drift apart, because there is only one brass:
//!
//! | facet | what asks for it |
//! |---|---|
//! | [`Material::pbr`] | `kosm-render`'s path tracer |
//! | [`Material::contact`] | phyz's contact solve |
//! | [`Material::modal`] | [`crate::audio`]'s modal bank |
//! | [`Material::fluid`] | anything that needs a density and a viscosity |
//! | [`Material::colour`] | the window's Lambert tier, and `parts.json` |
//!
//! **Every constant is a [`Param`].** [`Material::params`] flattens the
//! substance into named knobs (`"N-BK7.n_d"`, `"wet sand.friction"`), so a
//! run hash includes its materials and a fit can move them;
//! [`Material::with_params`] is the way back, and [`Material::fit`] descends
//! a loss over named constants by central differences — the zeroth-order
//! path, per `docs/architecture.md` rule 4.
//!
//! **Provenance, or it is a guess.** Every constant group carries a [`Cite`]:
//! a source string and [`Source::Measured`] or [`Source::Estimated`]. A
//! number nobody can point at is marked as estimated, and the datasheet
//! prints the mark.
//!
//! This is *not* [`crate::world::Material`], which is a render column on a
//! [`World`](crate::world::World) — an albedo and a roughness, indexed however
//! the sim indexes its surfaces. The prelude spells this one `Substance` for
//! exactly that reason.
//!
//! ```
//! use kosm::material::{self, Optics};
//!
//! let brass = material::named("brass").expect("brass is in the library");
//! assert_eq!(brass.density, 8500.0);
//! assert!(matches!(brass.optics, Optics::Conductor));
//!
//! // one substance, four facets
//! assert_eq!(brass.pbr().metallic, 1.0);            // a metal, to the tracer
//! assert_eq!(brass.contact().friction, brass.friction);
//! assert_eq!(brass.modal().rho, brass.density);
//! assert_eq!(brass.fluid().density, brass.density);
//! assert!(brass.fluid().viscosity.is_none(), "a solid does not flow");
//! assert_eq!(kosm::materials::colour("brass"), brass.colour());   // the window's tier
//!
//! // and every constant is a knob
//! let params = brass.params();
//! assert!(params.iter().any(|p| p.name == "brass.density"));
//! let denser = brass.with_params(&[kosm::world::Param::new("brass.density", 8600.0)]);
//! assert_eq!(denser.density, 8600.0);
//! assert_eq!(brass.density, 8500.0, "`with_params` does not mutate");
//! ```

use kosm_render::pathtrace::Pbr;
use phyz_contact::ContactMaterial;
use serde::{Deserialize, Serialize};

use crate::audio;
use crate::world::Param;

mod datasheet;
pub mod gpu;
mod library;

pub use datasheet::{
    BALL_RADIUS, Ball, Bounce, Datasheet, ball_world, bounce_curve, datasheet, ring_spectrum,
    settle,
};
pub use gpu::{GpuMaterial, band_to_rgb, bands_to_rgb, library_gpu};
pub use library::{named, names};

// ── spectra ───────────────────────────────────────────────────────────────

/// How many bands a [`Spectrum`] carries.
pub const BANDS: usize = 6;

/// The band centres, nanometres. Two per RGB channel, ascending.
pub const BANDS_NM: [f64; BANDS] = [420.0, 470.0, 520.0, 570.0, 620.0, 670.0];

/// A quantity that varies with wavelength: a reflectance, an F0, a radiance.
///
/// Six bands, two per RGB primary, because the film is RGB and a reflectance
/// curve resolved much finer than that is a curve nothing downstream can
/// spend. [`Spectrum::rgb`] and [`Spectrum::to_rgb`] are exact inverses — the
/// projection is the mean of each primary's two bands — so a material whose
/// colour was authored as RGB reports that RGB back unchanged, and a material
/// whose reflectance was measured band by band still projects to something
/// the tracer can use.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Spectrum {
    /// The value at each of [`BANDS_NM`].
    pub bands: [f64; BANDS],
}

impl Spectrum {
    pub const fn new(bands: [f64; BANDS]) -> Self {
        Self { bands }
    }

    /// The same value in every band.
    pub const fn flat(v: f64) -> Self {
        Self { bands: [v; BANDS] }
    }

    /// An RGB triple, spread over the two bands each primary owns.
    pub const fn rgb(c: [f64; 3]) -> Self {
        Self { bands: [c[2], c[2], c[1], c[1], c[0], c[0]] }
    }

    /// The projection at the film: the mean of each primary's two bands.
    /// Exactly inverts [`Spectrum::rgb`].
    pub fn to_rgb(&self) -> [f64; 3] {
        let b = &self.bands;
        [0.5 * (b[4] + b[5]), 0.5 * (b[2] + b[3]), 0.5 * (b[0] + b[1])]
    }

    /// The same, as the `f32` triple every `Pbr` field wants.
    pub fn to_rgb32(&self) -> [f32; 3] {
        let c = self.to_rgb();
        [c[0] as f32, c[1] as f32, c[2] as f32]
    }

    /// The value at a wavelength, linearly interpolated and clamped at both
    /// ends of [`BANDS_NM`].
    pub fn at(&self, lambda_nm: f64) -> f64 {
        if lambda_nm <= BANDS_NM[0] {
            return self.bands[0];
        }
        for i in 1..BANDS {
            if lambda_nm <= BANDS_NM[i] {
                let t = (lambda_nm - BANDS_NM[i - 1]) / (BANDS_NM[i] - BANDS_NM[i - 1]);
                return self.bands[i - 1] + t * (self.bands[i] - self.bands[i - 1]);
            }
        }
        self.bands[BANDS - 1]
    }

    /// Every band multiplied by `k`.
    pub fn scale(&self, k: f64) -> Self {
        let mut out = *self;
        for b in &mut out.bands {
            *b *= k;
        }
        out
    }
}

// ── optics ────────────────────────────────────────────────────────────────

/// What light does at the surface.
///
/// The three cases a path tracer actually distinguishes. [`Material::albedo`]
/// is read as a diffuse reflectance under [`Optics::Opaque`] and as `F0`, the
/// normal-incidence reflectance, under [`Optics::Conductor`] — the same
/// convention `Pbr::base_color` uses, which is why there is no separate `f0`
/// field to keep in agreement with it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Optics {
    /// A dielectric surface that does not transmit: stone, wood, paint.
    Opaque,
    /// A dielectric that does: glass, water, jelly.
    Dielectric(Dielectric),
    /// A metal. [`Material::albedo`] is `F0`.
    Conductor,
}

/// A transmitting dielectric's index and what its interior takes out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dielectric {
    /// Index at the sodium d line, 587.6 nm.
    pub n_d: f64,
    /// Abbe number `V_d = (n_d − 1)/(n_F − n_C)`. Smaller is more dispersive.
    pub abbe: f64,
    /// The datasheet's Sellmeier `(B, C)` pair, when there is one. It
    /// overrides [`Self::abbe`] in the tracer, and it is the same curve
    /// [`crate::light`] traces a caustic with.
    pub sellmeier: Option<([f64; 3], [f64; 3])>,
    /// Beer–Lambert absorption, when the interior takes anything out.
    pub absorption: Option<Absorption>,
}

impl Dielectric {
    /// A clear medium: an index, a dispersion, and nothing absorbed.
    pub const fn clear(n_d: f64, abbe: f64) -> Self {
        Self { n_d, abbe, sellmeier: None, absorption: None }
    }

    pub fn with_sellmeier(mut self, coeffs: ([f64; 3], [f64; 3])) -> Self {
        self.sellmeier = Some(coeffs);
        self
    }

    pub fn with_absorption(mut self, transmittance: Spectrum, distance_m: f64) -> Self {
        self.absorption = Some(Absorption { transmittance, distance_m });
        self
    }
}

/// What one [`Self::distance_m`] of the interior transmits.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Absorption {
    pub transmittance: Spectrum,
    pub distance_m: f64,
}

/// A random walk inside the substance: skin, wax, marble, sea foam.
///
/// [`Self::radius_m`] is the mean free path per RGB channel in **metres**,
/// which is the unit `Pbr::subsurface_radius` is in (scene units) and the
/// unit the physics is in; the library authors it in millimetres and converts
/// once.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Subsurface {
    /// Weight of the walk in 0..1. `0` switches it off.
    pub weight: f64,
    /// Mean free path per channel, metres.
    pub radius_m: [f64; 3],
    /// Henyey–Greenstein asymmetry `g` in −1..1. `0` is isotropic.
    pub anisotropy: f64,
}

impl Subsurface {
    /// A weight and one mean free path, in millimetres, isotropic.
    pub fn mm(weight: f64, mfp_mm: f64) -> Self {
        let r = mfp_mm * 1e-3;
        Self { weight, radius_m: [r, r, r], anisotropy: 0.0 }
    }

    /// The same with the red channel reaching furthest, which is what most
    /// organic media do and why a hand held to a light goes red at the edges.
    pub fn organic_mm(weight: f64, mfp_mm: f64, anisotropy: f64) -> Self {
        let r = mfp_mm * 1e-3;
        Self { weight, radius_m: [r, 0.55 * r, 0.35 * r], anisotropy }
    }
}

/// A thin film over the surface: a soap bubble, nacre, an oxide.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Film {
    /// Thickness, nanometres. A few hundred is where the interference lands
    /// in the visible band.
    pub thickness_nm: f64,
    /// The film's own index, between air and the substrate.
    pub ior: f64,
}

// ── provenance ────────────────────────────────────────────────────────────

/// Whether a constant came from a measurement or from a plausible guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Source {
    Measured,
    Estimated,
}

/// A source string for one group of constants, and how it was arrived at.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cite {
    pub source: String,
    pub kind: Source,
}

impl Cite {
    pub fn measured(source: impl Into<String>) -> Self {
        Self { source: source.into(), kind: Source::Measured }
    }

    pub fn estimated(source: impl Into<String>) -> Self {
        Self { source: source.into(), kind: Source::Estimated }
    }

    pub fn is_measured(&self) -> bool {
        self.kind == Source::Measured
    }
}

/// Where each group of a substance's constants came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Density, Young's modulus, Poisson's ratio, loss factor.
    pub mechanics: Cite,
    /// Index, dispersion, absorption, F0.
    pub optics: Cite,
    /// Albedo, roughness, subsurface, thin film, emission.
    pub appearance: Cite,
}

impl Provenance {
    /// The same citation for all three groups.
    pub fn all(cite: Cite) -> Self {
        Self { mechanics: cite.clone(), optics: cite.clone(), appearance: cite }
    }

    /// Every group, for the datasheet.
    pub fn groups(&self) -> [(&'static str, &Cite); 3] {
        [
            ("mechanics", &self.mechanics),
            ("optics", &self.optics),
            ("appearance", &self.appearance),
        ]
    }
}

impl Default for Provenance {
    fn default() -> Self {
        Self::all(Cite::estimated("unsourced"))
    }
}

// ── the fluid facet ───────────────────────────────────────────────────────

/// What a fluid solver wants: a density, and a viscosity if it flows.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fluid {
    /// kg/m³.
    pub density: f64,
    /// Dynamic viscosity, Pa·s. `None` is a solid — a solid's viscosity is
    /// not zero, it is unbounded, and `None` says so without writing an
    /// infinity into a JSON file that cannot hold one.
    pub viscosity: Option<f64>,
}

impl Fluid {
    pub fn is_liquid(&self) -> bool {
        self.viscosity.is_some()
    }

    /// Kinematic viscosity, m²/s.
    pub fn kinematic(&self) -> Option<f64> {
        self.viscosity.map(|mu| mu / self.density)
    }
}

// ── the substance ─────────────────────────────────────────────────────────

/// One substance. Physical constants as columns; every facet derived.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Material {
    /// What it is called. Also the prefix of every one of its [`Param`]s.
    pub name: String,
    /// kg/m³.
    pub density: f64,
    /// Young's modulus, Pa. Zero for something with no elastic modulus worth
    /// naming (a flame); for a liquid it holds the bulk modulus instead, with
    /// [`Self::poisson`] at 0.5 — see the entries in [`library`].
    pub young: f64,
    /// Poisson's ratio.
    pub poisson: f64,
    /// Structural loss factor η, dimensionless; `Q = 1/η`. Small rings.
    pub loss: f64,
    /// Coulomb μ against a generic partner. Which partner is in the
    /// [`Provenance`]; against itself unless the citation says otherwise.
    pub friction: f64,
    /// Coefficient of restitution, 0..1, against a plate of itself.
    pub restitution: f64,
    /// Contact stiffness, N/m, when the substance is worth pinning one for.
    /// `None` takes phyz's own default.
    pub stiffness: Option<f64>,
    /// Dynamic viscosity, Pa·s. `Some` makes it a liquid.
    pub viscosity: Option<f64>,
    /// What light does at the surface.
    pub optics: Optics,
    /// Reflectance vs wavelength — `F0` under [`Optics::Conductor`].
    pub albedo: Spectrum,
    /// Perceptual roughness, 0..1.
    pub roughness: f64,
    /// A random walk inside, when there is one.
    pub sss: Option<Subsurface>,
    /// Emitted radiance, for an emitter.
    pub emission: Option<Spectrum>,
    /// A thin film over the surface, for iridescence.
    pub thin_film: Option<Film>,
    /// A source per constant group, and whether each was measured.
    pub provenance: Provenance,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: String::new(),
            density: 1000.0,
            young: 1e9,
            poisson: 0.3,
            loss: 0.02,
            friction: 0.5,
            restitution: 0.2,
            stiffness: None,
            viscosity: None,
            optics: Optics::Opaque,
            albedo: Spectrum::rgb([0.60, 0.60, 0.62]),
            roughness: 0.6,
            sss: None,
            emission: None,
            thin_film: None,
            provenance: Provenance::default(),
        }
    }
}

impl Material {
    /// The library lookup, spelled as an associated function.
    /// See [`named`].
    pub fn named(name: &str) -> Option<Material> {
        named(name)
    }

    /// Index at the d line, when this substance has one.
    pub fn n_d(&self) -> Option<f64> {
        match &self.optics {
            Optics::Dielectric(d) => Some(d.n_d),
            _ => None,
        }
    }

    /// Whether every constant group was measured rather than estimated.
    pub fn is_measured(&self) -> bool {
        self.provenance.groups().iter().all(|(_, c)| c.is_measured())
    }

    // ── facets ────────────────────────────────────────────────────────────

    /// The renderer's facet: a `Pbr` for `kosm-render`'s path tracer.
    ///
    /// - `base_color` ← [`Self::albedo`] projected to RGB (an albedo for a
    ///   dielectric, `F0` for a metal, which is `Pbr`'s own convention);
    /// - `metallic` ← 1 under [`Optics::Conductor`], 0 otherwise;
    /// - `roughness` ← [`Self::roughness`];
    /// - `transmission`, `ior`, `abbe`, `sellmeier`, `attenuation_*` ←
    ///   [`Optics::Dielectric`];
    /// - `subsurface*` ← [`Self::sss`], through [`sss_onto_pbr`];
    /// - `emissive` ← [`Self::emission`];
    /// - `thin_film_*` ← [`Self::thin_film`].
    pub fn pbr(&self) -> Pbr {
        let mut p = Pbr {
            base_color: self.albedo.to_rgb32(),
            roughness: self.roughness as f32,
            ..Pbr::default()
        };
        match &self.optics {
            Optics::Opaque => {}
            Optics::Conductor => p.metallic = 1.0,
            Optics::Dielectric(d) => {
                // A full dielectric transmission lobe: the diffuse and opaque
                // specular lobes switch off and one rough dielectric both
                // reflects and refracts, split by the exact Fresnel equations.
                p.transmission = 1.0;
                p.specular = 1.0;
                p.ior = d.n_d as f32;
                p.abbe = d.abbe as f32;
                p.sellmeier = d.sellmeier;
                if let Some(a) = &d.absorption {
                    p.attenuation_color = a.transmittance.to_rgb32();
                    p.attenuation_distance = a.distance_m as f32;
                }
            }
        }
        if let Some(s) = &self.sss {
            sss_onto_pbr(&mut p, s, self.albedo);
        }
        if let Some(e) = &self.emission {
            p.emissive = e.to_rgb32();
        }
        if let Some(f) = &self.thin_film {
            p.thin_film_thickness = f.thickness_nm as f32;
            p.thin_film_ior = f.ior as f32;
        }
        p
    }

    /// The solver's facet: a `ContactMaterial` for phyz.
    ///
    /// - `friction` ← [`Self::friction`], `restitution` ← [`Self::restitution`];
    /// - `stiffness` ← [`Self::stiffness`], or phyz's default;
    /// - `damping` ← phyz's default, scaled with the stiffness so a substance
    ///   that pins a stiffness keeps the solver's own `k/c` ratio;
    /// - everything else — `solref`, `solimp`, `margin`, the soft terms —
    ///   phyz's defaults.
    ///
    /// The **loss factor does not appear here**, and that is deliberate. η is
    /// internal friction inside the solid: it decides how long a struck bell
    /// rings, which is [`Self::modal`]'s question. What a bounce keeps is
    /// [`Self::restitution`], and phyz's convex solve reads that (plus
    /// `solref`/`solimp`) rather than the `stiffness`/`damping` pair, which
    /// only the deprecated penalty law ever touched.
    pub fn contact(&self) -> ContactMaterial {
        let d = ContactMaterial::default();
        let stiffness = self.stiffness.unwrap_or(d.stiffness);
        ContactMaterial {
            stiffness,
            damping: d.damping * (stiffness / d.stiffness),
            friction: self.friction,
            restitution: self.restitution,
            ..d
        }
    }

    /// The bell's facet: the four constants [`crate::audio`]'s modal bank runs on.
    pub fn modal(&self) -> audio::Material {
        audio::Material {
            rho: self.density,
            e: self.young,
            nu: self.poisson,
            loss: self.loss,
        }
    }

    /// The fluid facet: a density, and a viscosity if it flows.
    pub fn fluid(&self) -> Fluid {
        Fluid { density: self.density, viscosity: self.viscosity }
    }

    /// Linear RGB for the window's Lambert tier — the old
    /// [`crate::materials`] table's job, now derived from the albedo.
    pub fn colour(&self) -> [f64; 3] {
        self.albedo.to_rgb()
    }

    // ── state and layers ──────────────────────────────────────────────────

    /// The same substance, wet.
    ///
    /// For a granular or porous solid — sand, stone, wood — water in the pores
    /// does three things, and this does exactly those three and no more:
    ///
    /// 1. **Darker**, albedo × 0.6. Water in the pore space raises the index
    ///    the grain boundaries see from 1.0 to 1.33, which drops the scattering
    ///    contrast and lets more of each bounce be absorbed before it comes
    ///    back out. 0.6 is the middle of the 0.5–0.7 that sand, brick and
    ///    concrete measure; it is an estimate, and the returned material says so.
    /// 2. **Grippier**, friction × 1.25. Capillary bridges between grains, up
    ///    to the point of saturation. Beyond a slurry this is wrong, and this
    ///    function does not model a slurry.
    /// 3. **Heavier**, density + `pore_fraction × 998`. The pore fraction is
    ///    estimated at 0.35 for a granular material, which is close-packed
    ///    spheres' 0.36 and typical of a beach.
    ///
    /// Smoother, too: the film fills the microrelief, so roughness × 0.75.
    /// Nothing else moves — the modulus, the loss and the index of the grains
    /// themselves are the grains', not the water's.
    ///
    /// ```
    /// # use kosm::material;
    /// let dry = material::named("dry sand").unwrap();
    /// let wet = dry.wet();
    /// assert_eq!(wet.name, "wet sand");
    /// assert!(wet.colour()[0] < dry.colour()[0], "wet sand is darker");
    /// assert!(wet.friction > dry.friction, "and grippier");
    /// assert!(wet.density > dry.density, "and heavier");
    /// // the library's own entry is this derivation, not a second table
    /// assert_eq!(material::named("wet sand").unwrap(), wet);
    /// ```
    pub fn wet(&self) -> Material {
        const PORE: f64 = 0.35;
        const WATER: f64 = 998.0;
        let mut out = self.clone();
        out.name = match self.name.strip_prefix("dry ") {
            Some(rest) => format!("wet {rest}"),
            None => format!("wet {}", self.name),
        };
        out.density = self.density + PORE * WATER;
        out.albedo = self.albedo.scale(0.6);
        out.friction = self.friction * 1.25;
        out.roughness = (self.roughness * 0.75).clamp(0.0, 1.0);
        out.provenance.appearance = Cite::estimated(format!(
            "{} wetted: albedo x0.6, roughness x0.75 (kosm `Material::wet`)",
            self.provenance.appearance.source
        ));
        out.provenance.mechanics = Cite::estimated(format!(
            "{} wetted: +0.35 pore fraction of water, friction x1.25 (kosm `Material::wet`)",
            self.provenance.mechanics.source
        ));
        out
    }

    /// The same substance under a clear coat: lacquer over wood.
    ///
    /// The substrate keeps its mechanics — a lacquered oak plank weighs and
    /// rings like oak, because the coat is tens of microns of a few hundred
    /// kilograms per cubic metre over centimetres of wood. What changes is the
    /// surface: the coat's roughness and its index are what the first bounce
    /// meets, and the substrate's albedo is what shows *under* it. That is a
    /// clearcoat lobe, so `pbr()` gets one.
    ///
    /// `thickness_m` is recorded as the coat's thin film only when it is thin
    /// enough to interfere — under a micron. A 30 µm lacquer does not
    /// iridesce and does not get a film.
    pub fn coated(&self, coat: &Material, thickness_m: f64) -> Material {
        let mut out = self.clone();
        out.name = format!("{} under {}", self.name, coat.name);
        out.roughness = coat.roughness;
        out.friction = coat.friction;
        // The substrate's albedo, seen through the coat: two passes of the
        // coat's own transmittance, if it has any.
        if let Optics::Dielectric(d) = &coat.optics {
            if let Some(a) = &d.absorption {
                let t = a.transmittance.at(550.0).powf(2.0 * thickness_m / a.distance_m);
                out.albedo = out.albedo.scale(t.clamp(0.0, 1.0));
            }
        }
        out.thin_film = (thickness_m < 1e-6).then(|| Film {
            thickness_nm: thickness_m * 1e9,
            ior: coat.n_d().unwrap_or(1.5),
        });
        out.provenance.appearance = Cite::estimated(format!(
            "{} under {:.0} um of {} (kosm `Material::coated`)",
            self.provenance.appearance.source,
            thickness_m * 1e6,
            coat.name
        ));
        out.provenance.optics = coat.provenance.optics.clone();
        out
    }

    // ── knobs ─────────────────────────────────────────────────────────────

    /// Every constant, as a named [`Param`]: `"<material>.<constant>"`.
    ///
    /// Put these on a [`World`](crate::world::World) and the run hash includes
    /// the materials, so a run with a different brass lands somewhere else.
    pub fn params(&self) -> Vec<Param> {
        let mut out = Vec::new();
        self.each(&mut |key, value| out.push(Param::new(format!("{}.{key}", self.name), value)));
        out
    }

    /// A copy with those knobs set.
    ///
    /// A [`Param`] matches either fully qualified (`"brass.density"`) or bare
    /// (`"density"`); anything that names no constant of this substance is
    /// ignored, so a world's whole param list can be handed over.
    pub fn with_params(&self, params: &[Param]) -> Material {
        let mut out = self.clone();
        let prefix = format!("{}.", self.name);
        for p in params {
            let key = p.name.strip_prefix(&prefix).unwrap_or(&p.name);
            out.set(key, p.value);
        }
        out
    }

    /// One constant's value, by key (`"n_d"`, `"density"`).
    pub fn constant(&self, key: &str) -> Option<f64> {
        let mut found = None;
        self.each(&mut |k, v| {
            if k == key {
                found = Some(v);
            }
        });
        found
    }

    /// Descend `loss` over the named constants by central differences.
    ///
    /// Gradients are one estimator, not the estimator
    /// (`docs/architecture.md` rule 4): a caustic loss is a ray budget with a
    /// grid under it, and a finite difference over it costs two traces and no
    /// tape. `which` names constants either bare (`"n_d"`) or fully qualified
    /// (`"N-BK7.n_d"`).
    ///
    /// The descent is in **relative** coordinates — each constant divided by
    /// its own starting magnitude — so an index near 1.5 and a density near
    /// 2510 take the same step, and the step is halved until the loss actually
    /// falls. Monotone by construction: the returned material is never worse
    /// than the one that went in.
    ///
    /// ```
    /// # use kosm::material;
    /// let start = material::named("N-BK7").unwrap();
    /// // any loss over the substance; here: pull the index toward 1.6
    /// let fitted = start.fit(|m| (m.n_d().unwrap() - 1.6).powi(2), &["n_d"], 40);
    /// assert!((fitted.n_d().unwrap() - 1.6).abs() < 1e-3, "{:?}", fitted.n_d());
    /// assert_eq!(fitted.density, start.density, "nothing else moved");
    /// ```
    ///
    /// `crates/kosm/tests/material.rs` does it against a real caustic: N-BK7's
    /// index, started at 1.42, recovered from the light a sphere of it throws.
    pub fn fit<F: Fn(&Material) -> f64>(&self, loss: F, which: &[&str], steps: usize) -> Material {
        let prefix = format!("{}.", self.name);
        let keys: Vec<String> = which
            .iter()
            .map(|w| w.strip_prefix(&prefix).unwrap_or(w).to_string())
            .filter(|k| self.constant(k).is_some())
            .collect();
        if keys.is_empty() {
            return self.clone();
        }
        // Relative coordinates: the scale of each constant is its own start.
        let scale: Vec<f64> = keys
            .iter()
            .map(|k| self.constant(k).unwrap_or(0.0).abs().max(1e-9))
            .collect();

        let mut cur = self.clone();
        let mut best = loss(&cur);
        let mut step = 0.05;
        let h = 1e-3;
        for _ in 0..steps {
            // ∂loss/∂u by a central difference in relative units.
            let mut grad = vec![0.0; keys.len()];
            for (i, key) in keys.iter().enumerate() {
                let v = cur.constant(key).unwrap_or(0.0);
                let d = h * scale[i];
                let mut lo = cur.clone();
                lo.set(key, v - d);
                let mut hi = cur.clone();
                hi.set(key, v + d);
                grad[i] = (loss(&hi) - loss(&lo)) / (2.0 * h);
            }
            let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if !norm.is_finite() || norm < 1e-14 {
                break;
            }
            let mut moved = false;
            for _ in 0..8 {
                let mut cand = cur.clone();
                for (i, key) in keys.iter().enumerate() {
                    let v = cur.constant(key).unwrap_or(0.0);
                    cand.set(key, v - step * (grad[i] / norm) * scale[i]);
                }
                let l = loss(&cand);
                if l < best {
                    cur = cand;
                    best = l;
                    moved = true;
                    break;
                }
                step *= 0.5;
            }
            if !moved {
                break;
            }
        }
        cur
    }

    /// Visit every constant as `(key, value)`. The mirror of [`Self::set`];
    /// they are next to each other so a new field cannot land in one alone.
    fn each(&self, f: &mut dyn FnMut(&str, f64)) {
        f("density", self.density);
        f("young", self.young);
        f("poisson", self.poisson);
        f("loss", self.loss);
        f("friction", self.friction);
        f("restitution", self.restitution);
        f("roughness", self.roughness);
        if let Some(k) = self.stiffness {
            f("stiffness", k);
        }
        if let Some(mu) = self.viscosity {
            f("viscosity", mu);
        }
        if let Optics::Dielectric(d) = &self.optics {
            f("n_d", d.n_d);
            f("abbe", d.abbe);
        }
        for (i, nm) in BANDS_NM.iter().enumerate() {
            f(&format!("albedo.{nm:.0}"), self.albedo.bands[i]);
        }
        if let Some(s) = &self.sss {
            f("sss.weight", s.weight);
            f("sss.radius.r", s.radius_m[0]);
            f("sss.radius.g", s.radius_m[1]);
            f("sss.radius.b", s.radius_m[2]);
            f("sss.anisotropy", s.anisotropy);
        }
        if let Some(e) = &self.emission {
            for (i, nm) in BANDS_NM.iter().enumerate() {
                f(&format!("emission.{nm:.0}"), e.bands[i]);
            }
        }
        if let Some(t) = &self.thin_film {
            f("film.thickness_nm", t.thickness_nm);
            f("film.ior", t.ior);
        }
    }

    /// Set one constant by key. Returns whether the key named anything.
    fn set(&mut self, key: &str, value: f64) -> bool {
        match key {
            "density" => self.density = value,
            "young" => self.young = value,
            "poisson" => self.poisson = value,
            "loss" => self.loss = value,
            "friction" => self.friction = value,
            "restitution" => self.restitution = value,
            "roughness" => self.roughness = value,
            "stiffness" => self.stiffness = Some(value),
            "viscosity" => self.viscosity = Some(value),
            "n_d" => match &mut self.optics {
                Optics::Dielectric(d) => d.n_d = value,
                _ => return false,
            },
            "abbe" => match &mut self.optics {
                Optics::Dielectric(d) => d.abbe = value,
                _ => return false,
            },
            "sss.weight" => match &mut self.sss {
                Some(s) => s.weight = value,
                None => return false,
            },
            "sss.radius.r" | "sss.radius.g" | "sss.radius.b" => match &mut self.sss {
                Some(s) => {
                    let i = match key.as_bytes()[key.len() - 1] {
                        b'r' => 0,
                        b'g' => 1,
                        _ => 2,
                    };
                    s.radius_m[i] = value;
                }
                None => return false,
            },
            "sss.anisotropy" => match &mut self.sss {
                Some(s) => s.anisotropy = value,
                None => return false,
            },
            "film.thickness_nm" => match &mut self.thin_film {
                Some(t) => t.thickness_nm = value,
                None => return false,
            },
            "film.ior" => match &mut self.thin_film {
                Some(t) => t.ior = value,
                None => return false,
            },
            _ => {
                if let Some(rest) = key.strip_prefix("albedo.") {
                    return match band_index(rest) {
                        Some(i) => {
                            self.albedo.bands[i] = value;
                            true
                        }
                        None => false,
                    };
                }
                if let Some(rest) = key.strip_prefix("emission.") {
                    return match (band_index(rest), &mut self.emission) {
                        (Some(i), Some(e)) => {
                            e.bands[i] = value;
                            true
                        }
                        _ => false,
                    };
                }
                return false;
            }
        }
        true
    }
}

/// Which band a `"420"`-style suffix names.
fn band_index(nm: &str) -> Option<usize> {
    let want: f64 = nm.parse().ok()?;
    BANDS_NM.iter().position(|b| (b - want).abs() < 0.5)
}

/// The one place a [`Subsurface`] becomes `Pbr`'s subsurface fields.
///
/// Kept as a free function with one caller so the mapping is one edit when
/// `Pbr` grows a field. `subsurface_color` is the *surface* albedo the walk
/// is asked to produce — which is what [`Material::albedo`] already is —
/// `subsurface_radius` is already in metres, the scene's own unit, and
/// `subsurface_anisotropy` is the medium's Henyey–Greenstein `g` under
/// similarity theory, which is what [`Subsurface::anisotropy`] holds.
///
/// The GPU tier ignores every one of these; its walk is a shorter one.
pub fn sss_onto_pbr(p: &mut Pbr, s: &Subsurface, albedo: Spectrum) {
    p.subsurface = s.weight as f32;
    p.subsurface_color = albedo.to_rgb32();
    p.subsurface_radius = s.radius_m;
    p.subsurface_anisotropy = s.anisotropy as f32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spectrum_round_trips_through_rgb_exactly() {
        let c = [0.55, 0.72, 0.80];
        assert_eq!(Spectrum::rgb(c).to_rgb(), c);
        // and a measured curve projects to the mean of each primary's bands
        let s = Spectrum::new([0.20, 0.24, 0.46, 0.60, 0.70, 0.74]);
        let rgb = s.to_rgb();
        assert!((rgb[0] - 0.72).abs() < 1e-12);
        assert!((rgb[1] - 0.53).abs() < 1e-12);
        assert!((rgb[2] - 0.22).abs() < 1e-12);
        // interpolation is clamped at both ends
        assert_eq!(s.at(300.0), 0.20);
        assert_eq!(s.at(900.0), 0.74);
        assert!((s.at(445.0) - 0.22).abs() < 1e-12);
    }

    #[test]
    fn the_four_facets_come_out_of_the_same_constants() {
        let m = named("N-BK7").expect("N-BK7");
        let p = m.pbr();
        assert_eq!(p.transmission, 1.0, "glass transmits");
        assert!((p.ior - 1.5168).abs() < 1e-6);
        assert!(p.sellmeier.is_some(), "the datasheet's curve, not an Abbe fit");
        let c = m.contact();
        assert_eq!(c.friction, m.friction);
        assert_eq!(c.restitution, m.restitution);
        let a = m.modal();
        assert_eq!((a.rho, a.e, a.nu, a.loss), (m.density, m.young, m.poisson, m.loss));
        assert_eq!(m.fluid().density, m.density);
        assert!(m.fluid().viscosity.is_none());
    }

    #[test]
    fn a_conductor_is_metallic_and_its_albedo_is_f0() {
        let brass = named("brass").expect("brass");
        assert!(matches!(brass.optics, Optics::Conductor));
        let p = brass.pbr();
        assert_eq!(p.metallic, 1.0);
        assert_eq!(p.transmission, 0.0);
        assert_eq!(p.base_color, brass.albedo.to_rgb32());
        // brass is warm: F0 rises toward the red end
        assert!(brass.albedo.bands[5] > brass.albedo.bands[0]);
    }

    #[test]
    fn every_constant_is_a_param_and_comes_back() {
        let m = named("granite").expect("granite");
        let params = m.params();
        let names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"granite.density"));
        assert!(names.contains(&"granite.friction"));
        assert!(names.contains(&"granite.albedo.420"));
        assert!(!names.contains(&"granite.n_d"), "granite does not transmit");
        // round trip: the params rebuild the material they came from
        assert_eq!(Material { density: 0.0, ..m.clone() }.with_params(&params), m);
        // a bare key works too, and an unknown one is ignored
        let d = m.with_params(&[Param::new("density", 3000.0), Param::new("nonsense", 1.0)]);
        assert_eq!(d.density, 3000.0);
        assert_eq!(m.density, 2650.0, "`with_params` does not mutate");
    }

    #[test]
    fn wet_sand_is_darker_grippier_and_heavier() {
        let dry = named("dry sand").expect("dry sand");
        let wet = dry.wet();
        assert_eq!(wet.name, "wet sand");
        assert!(wet.colour()[0] < dry.colour()[0]);
        assert!((wet.colour()[0] - 0.6 * dry.colour()[0]).abs() < 1e-12);
        assert!(wet.friction > dry.friction);
        assert!((wet.density - (dry.density + 0.35 * 998.0)).abs() < 1e-9);
        assert_eq!(wet.young, dry.young, "the grains are the grains");
        assert!(!wet.provenance.mechanics.is_measured(), "wetting is an estimate");
        // and the library's own `wet sand` is that same derivation
        assert_eq!(named("wet sand").expect("wet sand"), wet);
    }

    #[test]
    fn a_clear_coat_keeps_the_substrate_and_takes_the_surface() {
        let oak = named("oak").expect("oak");
        let lacquer = named("lacquer").expect("lacquer");
        let coated = oak.coated(&lacquer, 30e-6);
        assert_eq!(coated.density, oak.density, "the coat weighs nothing");
        assert_eq!(coated.young, oak.young);
        assert_eq!(coated.roughness, lacquer.roughness, "the coat is the surface");
        assert!(coated.thin_film.is_none(), "30 um does not iridesce");
        // a film thin enough to interfere does get one
        let thin = oak.coated(&lacquer, 400e-9);
        assert_eq!(thin.thin_film.expect("a film").thickness_nm, 400.0);
    }

    #[test]
    fn a_fit_moves_the_constant_it_is_told_to_and_nothing_else() {
        let m = named("N-BK7").expect("N-BK7");
        let start = m.with_params(&[Param::new("N-BK7.n_d", 1.42)]);
        let fitted = start.fit(|m| (m.n_d().unwrap() - 1.5168).powi(2), &["n_d"], 40);
        assert!((fitted.n_d().unwrap() - 1.5168).abs() < 1e-4, "got {:?}", fitted.n_d());
        assert_eq!(fitted.density, start.density, "only `n_d` was named");
        // an unknown constant is a no-op, not a panic
        assert_eq!(start.fit(|_| 0.0, &["unobtainium"], 5), start);
    }

    #[test]
    fn subsurface_reaches_the_pbr() {
        let wax = named("beeswax").expect("beeswax");
        let p = wax.pbr();
        let sss = wax.sss.expect("beeswax scatters");
        assert!(p.subsurface > 0.0);
        assert_eq!(p.subsurface_color, wax.albedo.to_rgb32());
        assert_eq!(p.subsurface_radius, sss.radius_m);
        assert_eq!(p.subsurface_anisotropy, sss.anisotropy as f32);
        // red reaches furthest, which is why a candle's edge goes red
        assert!(sss.radius_m[0] > sss.radius_m[2]);
        // and a substance with no walk switches the lobe off outright
        assert_eq!(named("granite").expect("granite").pbr().subsurface, 0.0);
    }

    #[test]
    fn an_emitter_emits_and_a_film_interferes() {
        let flame = named("candle flame").expect("candle flame");
        let p = flame.pbr();
        assert!(p.emissive[0] > p.emissive[2], "a flame is warm");
        let nacre = named("shell (nacre)").expect("nacre");
        let p = nacre.pbr();
        assert!(p.thin_film_thickness > 100.0 && p.thin_film_thickness < 1000.0);
        assert!(p.thin_film_ior > 1.0);
    }

    #[test]
    fn a_liquid_flows_and_a_solid_does_not() {
        let water = named("water").expect("water");
        let f = water.fluid();
        assert_eq!(f.viscosity, Some(1.0e-3));
        assert!((f.kinematic().unwrap() - 1.0e-3 / 998.0).abs() < 1e-15);
        assert!(named("granite").unwrap().fluid().viscosity.is_none());
        // sea water is denser and slower than fresh
        let sea = named("sea water").expect("sea water");
        assert!(sea.density > water.density);
        assert!(sea.n_d().unwrap() > water.n_d().unwrap());
    }
}
