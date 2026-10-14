//! The library: a fixed list of substances, each constant with a source.
//!
//! [`named`] is the lookup, case-insensitive. Every entry cites a source per
//! constant group and marks it [`Source::Measured`] or [`Source::Estimated`];
//! there is no third state, so a number nobody can point at is an estimate and
//! says so on the datasheet.
//!
//! **The level names are here too.** `plywood`, `rim`, `key`, `pad` and the
//! rest of the old [`crate::materials`] colour table are entries in their own
//! right, carrying the exact linear RGB that table gave them. Those colours
//! are appearance, not reflectance — they were chosen to read as the material
//! under the window's Lambert tier — so their appearance citation says so and
//! is marked as an estimate, while the mechanics beside them are the real
//! substance's. That is what lets the court, the skatepark and the cove render
//! the same colours after this change and still ask a `rim` for its density.
//! The cove's costume — `cloak`, `cream`, `skin`, `blush`, `ink`, `boot` — is
//! there on the same terms, so `sims/rune/hero` can build a figure whose every
//! part answers `substance()` and a later `kosm::player` can hang phyz joints
//! on it without writing a second table of densities.
//!
//! ```
//! use kosm::material;
//! // case-insensitive, and the aliases resolve
//! assert_eq!(material::named("N-BK7"), material::named("n-bk7"));
//! assert_eq!(material::named("gold").unwrap().name, "gold leaf");
//! assert!(material::named("unobtainium").is_none());
//! ```

use super::{Cite, Dielectric, Film, Material, Optics, Provenance, Spectrum, Subsurface};

/// The substance a name means, or `None`.
///
/// Case-insensitive, and surrounding whitespace is ignored. Aliases resolve to
/// the canonical entry, whose [`Material::name`] is the canonical name — so
/// `named("gold")` is the material called `"gold leaf"`, and its params are
/// `"gold leaf.density"` and friends.
pub fn named(name: &str) -> Option<Material> {
    let key = name.trim().to_ascii_lowercase();
    build(&key)
}

/// Every canonical name in the library, in the order this module lists them.
pub fn names() -> &'static [&'static str] {
    NAMES
}

/// The canonical names. Aliases are not listed; [`named`] resolves those.
const NAMES: &[&str] = &[
    // glass and transparent things
    "N-BK7",
    "soda-lime glass",
    "glass",
    "fused silica",
    "lead crystal",
    "water",
    "sea water",
    "soap film",
    "jelly",
    "lacquer",
    // ground
    "dry sand",
    "wet sand",
    "sandstone",
    "limestone",
    "granite",
    "basalt",
    "slate",
    "clay",
    "terracotta",
    "chalk",
    "concrete",
    "brick",
    // metals
    "brass",
    "bell bronze",
    "copper",
    "iron",
    "steel",
    "gold leaf",
    "silver",
    "pewter",
    "galvanized",
    "rim",
    // ceramics and organics
    "porcelain",
    "bone china",
    "oak",
    "maple",
    "teak",
    "driftwood",
    "bamboo",
    "rope (hemp)",
    "leather",
    "wool felt",
    "linen",
    "paper",
    "parchment",
    "beeswax",
    "cork",
    "rubber",
    "plywood",
    "masonite",
    "cardboard",
    "pla",
    // the sea
    "coral",
    "shell (nacre)",
    "kelp",
    "marram grass",
    "sea foam",
    // emitters
    "candle flame",
    "lamp",
    // the court's painted surfaces
    "wall",
    "ceiling",
    "floor",
    "pad",
    "window",
    "paint",
    "key",
    // the cove's costume
    "cloak",
    "cream",
    "skin",
    "blush",
    "ink",
    "boot",
];

// ── sources ───────────────────────────────────────────────────────────────

const SCHOTT: &str = "Schott N-BK7 optical glass datasheet (2023)";
const CRC: &str = "CRC Handbook of Chemistry and Physics, 97th ed.";
const ASHBY: &str = "Ashby, Materials Selection in Mechanical Design, 4th ed., Appendix A";
const RAO: &str = "Rao, Mechanical Vibrations 6th ed., material damping tables";
const ETB: &str = "Engineering ToolBox, coefficients of friction, dry";
const RII: &str = "refractiveindex.info, room-temperature dispersion data";
const POPE_FRY: &str = "Pope & Fry, Appl. Opt. 36(33) 8710 (1997), pure-water absorption; \
     Smith & Baker, Appl. Opt. 20(2) 177 (1981) agrees to a few per cent over these bands";
const OPTICS4: &str = "Bass et al., Handbook of Optics vol. 4, tabulated n and k";
const WOOD: &str = "USDA Wood Handbook, FPL-GTR-190, ch. 5";
const LOOK: &str = "kosm's window tier: the linear RGB the old `materials` colour table gave this name";

fn measured(s: &str) -> Cite {
    Cite::measured(s)
}

fn estimated(s: &str) -> Cite {
    Cite::estimated(s)
}

// ── a small builder, so an entry is one readable paragraph ────────────────

struct B(Material);

impl B {
    fn new(name: &str) -> Self {
        Self(Material { name: name.to_owned(), ..Material::default() })
    }

    /// Density kg/m³, Young's modulus Pa, Poisson's ratio, loss factor η.
    fn mech(mut self, rho: f64, e: f64, nu: f64, loss: f64) -> Self {
        self.0.density = rho;
        self.0.young = e;
        self.0.poisson = nu;
        self.0.loss = loss;
        self
    }

    /// Coulomb μ and coefficient of restitution.
    fn rub(mut self, friction: f64, restitution: f64) -> Self {
        self.0.friction = friction;
        self.0.restitution = restitution;
        self
    }

    /// Linear-RGB albedo and perceptual roughness.
    fn look(mut self, rgb: [f64; 3], roughness: f64) -> Self {
        self.0.albedo = Spectrum::rgb(rgb);
        self.0.roughness = roughness;
        self
    }

    /// A metal: the six bands are `F0`, and the surface is a conductor.
    fn metal(mut self, f0: [f64; 6], roughness: f64) -> Self {
        self.0.optics = Optics::Conductor;
        self.0.albedo = Spectrum::new(f0);
        self.0.roughness = roughness;
        self
    }

    /// The same for a metal whose `F0` is flat within each primary — a
    /// near-achromatic one, and the shape a name from the old colour table
    /// has to keep so `colour` reproduces it to the bit.
    fn metal_flat(mut self, f0: [f64; 3], roughness: f64) -> Self {
        self.0.optics = Optics::Conductor;
        self.0.albedo = Spectrum::rgb(f0);
        self.0.roughness = roughness;
        self
    }

    /// A transmitting dielectric.
    fn clear(mut self, d: Dielectric) -> Self {
        self.0.optics = Optics::Dielectric(d);
        self
    }

    fn sss(mut self, s: Subsurface) -> Self {
        self.0.sss = Some(s);
        self
    }

    fn emit(mut self, rgb: [f64; 3]) -> Self {
        self.0.emission = Some(Spectrum::rgb(rgb));
        self
    }

    fn film(mut self, thickness_nm: f64, ior: f64) -> Self {
        self.0.thin_film = Some(Film { thickness_nm, ior });
        self
    }

    /// Dynamic viscosity, Pa·s — what makes it a liquid.
    fn flows(mut self, viscosity: f64) -> Self {
        self.0.viscosity = Some(viscosity);
        self
    }

    fn stiffness(mut self, k: f64) -> Self {
        self.0.stiffness = Some(k);
        self
    }

    fn cite(mut self, mechanics: Cite, optics: Cite, appearance: Cite) -> Self {
        self.0.provenance = Provenance { mechanics, optics, appearance };
        self
    }

    fn done(self) -> Option<Material> {
        Some(self.0)
    }
}

/// The table. One arm per name; aliases share an arm.
fn build(key: &str) -> Option<Material> {
    match key {
        // ── glass and transparent things ──────────────────────────────────
        //
        // The reference glass of this engine: it is what the being is, what
        // the marble's caustic was fitted against, and the one entry whose
        // Sellmeier pair — not an Abbe approximation — reaches the tracer, so
        // `light.rs` and `pathtrace` disperse the same photon the same way.
        "n-bk7" | "bk7" => B::new("N-BK7")
            .mech(2510.0, 82.0e9, 0.206, 1.0e-3)
            .rub(0.5, 0.65)
            .look([1.0, 1.0, 1.0], 0.02)
            .clear(
                Dielectric::clear(1.5168, 64.17)
                    .with_sellmeier(kosm_render::spectrum::BK7_SELLMEIER),
            )
            .cite(
                measured(SCHOTT),
                measured(SCHOTT),
                estimated("clear: albedo white, polished to roughness 0.02"),
            )
            .done(),
        "soda-lime glass" | "soda-lime" | "float glass" => B::new("soda-lime glass")
            .mech(2500.0, 70.0e9, 0.23, 1.0e-3)
            .rub(0.5, 0.6)
            .look([1.0, 1.0, 1.0], 0.02)
            .clear(Dielectric::clear(1.5200, 59.0).with_absorption(
                // The iron in the melt: green on the polished edge, invisible
                // through a window.
                Spectrum::new([0.72, 0.80, 0.93, 0.90, 0.78, 0.74]),
                1.0,
            ))
            .cite(
                measured(&format!("{CRC} (soda-lime); {RAO} for the loss factor")),
                measured(RII),
                estimated("clear, with the float-glass iron tint on a metre of path"),
            )
            .done(),
        // The level name. Soda-lime's constants under the pale blue the
        // window's Lambert tier has always drawn `glass` with.
        "glass" => B::new("glass")
            .mech(2500.0, 70.0e9, 0.23, 1.0e-3)
            .rub(0.5, 0.6)
            .look([0.55, 0.72, 0.80], 0.05)
            .clear(Dielectric::clear(1.5200, 59.0))
            .cite(measured(CRC), measured(RII), estimated(LOOK))
            .done(),
        "fused silica" | "quartz glass" => B::new("fused silica")
            .mech(2200.0, 73.0e9, 0.17, 1.0e-4)
            .rub(0.5, 0.7)
            .look([1.0, 1.0, 1.0], 0.02)
            .clear(Dielectric::clear(1.4585, 67.8))
            .cite(
                measured(&format!("Corning 7980 datasheet; {RAO} for η")),
                measured(RII),
                estimated("clear and polished"),
            )
            .done(),
        // Lead makes it heavy, soft and *dispersive*: an Abbe near 33 is half
        // N-BK7's, so a rune thrown through this splits about twice as wide.
        "lead crystal" | "flint glass" => B::new("lead crystal")
            .mech(3100.0, 60.0e9, 0.22, 2.0e-3)
            .rub(0.5, 0.6)
            .look([1.0, 1.0, 1.0], 0.02)
            .clear(Dielectric::clear(1.6000, 33.0))
            .cite(
                measured(&format!("{CRC} (lead glass, 24% PbO)")),
                measured(RII),
                estimated("clear and polished"),
            )
            .done(),
        // A liquid has no Young's modulus; `young` carries the bulk modulus
        // and `poisson` is 0.5, which is what incompressible means. The modal
        // facet declines to ring it — see `Datasheet`.
        "water" | "fresh water" => B::new("water")
            .mech(998.0, 2.2e9, 0.5, 0.0)
            .rub(0.0, 0.0)
            .look([1.0, 1.0, 1.0], 0.02)
            .clear(Dielectric::clear(1.3330, 55.7))
            .flows(1.0e-3)
            .cite(
                measured(&format!("{CRC} (20 °C: ρ, bulk modulus, dynamic viscosity)")),
                measured(RII),
                estimated("clear; a still surface is smooth"),
            )
            .done(),
        "sea water" | "seawater" => B::new("sea water")
            .mech(1025.0, 2.3e9, 0.5, 0.0)
            .rub(0.0, 0.0)
            .look([1.0, 1.0, 1.0], 0.02)
            .clear(Dielectric::clear(1.3390, 55.0).with_absorption(
                // **Why the sea is teal.** Pope & Fry's pure-water absorption
                // coefficients, `a(λ)` in reciprocal metres, at this band
                // set: 0.0455, 0.0179, 0.0476, 0.0799, 0.2755, 0.4390.
                // Written here as what one metre transmits, `e^{−a}`, which
                // is what `Absorption` is. Red goes first — a metre of clear
                // water keeps under two thirds of it — and blue travels
                // furthest, which is the whole of the colour of water and
                // the reason a body of it darkens toward cyan with depth
                // rather than toward grey.
                //
                // Salt does not change this: at 35 PSU the difference from
                // pure water is under a per cent over the visible, which is
                // below the resolution of six bands. What *does* change a
                // real sea is what is suspended in it, and that is
                // scattering rather than absorption — a level that wants
                // turbid water adds it on top of these (the cove does; see
                // `kosm_view::raster::Sea::scatter`).
                Spectrum::new([0.9555, 0.9823, 0.9535, 0.9232, 0.7592, 0.6447]),
                1.0,
            ))
            .flows(1.07e-3)
            .cite(
                measured(&format!("{CRC} (35 PSU, 20 °C)")),
                measured(&format!("{RII}; {POPE_FRY}")),
                estimated("clear; a still surface is smooth"),
            )
            .done(),
        // A bubble: a film thin enough that the two surfaces interfere, which
        // is the entire reason to have a thin-film term at all.
        "soap film" | "bubble" => B::new("soap film")
            .mech(1000.0, 1.0e5, 0.5, 0.5)
            .rub(0.0, 0.0)
            .look([1.0, 1.0, 1.0], 0.02)
            .clear(Dielectric::clear(1.3300, 55.0))
            .flows(2.0e-3)
            .film(500.0, 1.33)
            .cite(
                estimated("soap solution, near water"),
                measured(&format!("{RII} (soap solution ~1.33)")),
                estimated("500 nm: the middle of the 100-1000 nm a bubble runs before it drains"),
            )
            .done(),
        "jelly" | "gel" => B::new("jelly")
            .mech(1000.0, 1.0e4, 0.49, 0.3)
            .rub(0.4, 0.1)
            .look([0.86, 0.28, 0.30], 0.15)
            .clear(Dielectric::clear(1.3400, 50.0))
            .sss(Subsurface::organic_mm(1.0, 40.0, 0.2))
            .stiffness(500.0)
            .cite(
                estimated("gelatin gel, 5-10% w/w: kilopascal-scale modulus, near-water density"),
                estimated("near water"),
                estimated("40 mm mean free path: a spoonful glows through"),
            )
            .done(),
        // A clear coat, and the coat `Material::coated` is written for.
        "lacquer" | "varnish" | "clearcoat" => B::new("lacquer")
            .mech(1200.0, 2.0e9, 0.35, 0.05)
            .rub(0.4, 0.4)
            .look([1.0, 1.0, 1.0], 0.10)
            .clear(Dielectric::clear(1.5000, 45.0))
            .cite(
                estimated("nitrocellulose lacquer: a stiff amorphous polymer"),
                estimated("a generic oil or lacquer sits near 1.5"),
                estimated("clear, and smooth because that is the point of lacquering"),
            )
            .done(),

        // ── ground ────────────────────────────────────────────────────────
        //
        // The beach. Internal friction near 33 degrees is tan 0.65, and a
        // grain landing on grains keeps almost nothing.
        "dry sand" | "sand" => B::new("dry sand")
            .mech(1600.0, 4.0e7, 0.30, 0.5)
            .rub(0.55, 0.05)
            .look([0.80, 0.71, 0.50], 0.95)
            .stiffness(2.0e4)
            .cite(
                measured("Das, Principles of Geotechnical Engineering: loose quartz sand, phi ~30-33 deg"),
                estimated("opaque; a grain is quartz but a beach is not"),
                estimated(LOOK),
            )
            .done(),
        // Also a named entry, and it is exactly the derivation: whatever
        // `dry sand` says, wetted. One rule, not two tables.
        "wet sand" => named("dry sand").map(|m| m.wet()),
        "sandstone" => B::new("sandstone")
            .mech(2200.0, 15.0e9, 0.25, 0.02)
            .rub(0.6, 0.2)
            .look([0.66, 0.52, 0.36], 0.9)
            .cite(measured(ASHBY), estimated("opaque"), estimated("weathered buff sandstone"))
            .done(),
        "limestone" | "stone" => B::new("limestone")
            .mech(2600.0, 50.0e9, 0.25, 0.01)
            .rub(0.6, 0.25)
            .look([0.52, 0.50, 0.46], 0.85)
            .cite(measured(ASHBY), estimated("opaque"), estimated(LOOK))
            .done(),
        "granite" | "rock" => B::new("granite")
            .mech(2650.0, 60.0e9, 0.25, 5.0e-3)
            .rub(0.65, 0.3)
            .look([0.45, 0.43, 0.40], 0.85)
            .cite(measured(ASHBY), estimated("opaque"), estimated(LOOK))
            .done(),
        "basalt" => B::new("basalt")
            .mech(2900.0, 70.0e9, 0.25, 5.0e-3)
            .rub(0.7, 0.3)
            .look([0.20, 0.19, 0.18], 0.85)
            .cite(measured(ASHBY), estimated("opaque"), estimated("dark grey volcanic rock"))
            .done(),
        "slate" => B::new("slate")
            .mech(2750.0, 60.0e9, 0.25, 0.01)
            .rub(0.55, 0.25)
            .look([0.22, 0.23, 0.25], 0.6)
            .cite(measured(ASHBY), estimated("opaque"), estimated("split slate, faintly blue"))
            .done(),
        // The default a body gets when nothing else was said, so its colour is
        // the clay grey `materials::colour` has always fallen back to.
        "clay" => B::new("clay")
            .mech(1750.0, 2.0e7, 0.35, 0.2)
            .rub(0.5, 0.05)
            .look([0.60, 0.60, 0.62], 0.9)
            .cite(
                estimated("wet modelling clay: a soft cohesive solid"),
                estimated("opaque"),
                estimated(LOOK),
            )
            .done(),
        "terracotta" | "fired clay" => B::new("terracotta")
            .mech(2000.0, 14.0e9, 0.22, 0.02)
            .rub(0.6, 0.3)
            .look([0.60, 0.28, 0.17], 0.8)
            .cite(measured(ASHBY), estimated("opaque"), estimated("unglazed fired earthenware"))
            .done(),
        "chalk" => B::new("chalk")
            .mech(2100.0, 9.0e9, 0.25, 0.03)
            .rub(0.7, 0.1)
            .look([0.90, 0.89, 0.85], 0.95)
            .cite(measured(ASHBY), estimated("opaque"), estimated("the brightest ground there is"))
            .done(),
        "concrete" => B::new("concrete")
            .mech(2400.0, 30.0e9, 0.20, 0.02)
            .rub(0.7, 0.2)
            .look([0.60, 0.60, 0.58], 0.9)
            .cite(measured(ASHBY), estimated("opaque"), estimated(LOOK))
            .done(),
        "brick" => B::new("brick")
            .mech(1900.0, 16.0e9, 0.15, 0.03)
            .rub(0.7, 0.2)
            .look([0.58, 0.32, 0.25], 0.9)
            .cite(measured(ASHBY), estimated("opaque"), estimated(LOOK))
            .done(),

        // ── metals ────────────────────────────────────────────────────────
        //
        // The automaton's joints. A metal's `albedo` is F0, so the six bands
        // are the reflectance curve and its rise toward the red is the whole
        // reason brass looks like brass.
        "brass" => B::new("brass")
            .mech(8500.0, 100.0e9, 0.34, 1.0e-3)
            .rub(0.35, 0.4)
            .metal([0.20, 0.24, 0.46, 0.60, 0.70, 0.74], 0.30)
            .cite(
                measured(&format!("{ASHBY} (Cu-Zn 70/30); {ETB}: brass on steel")),
                measured(OPTICS4),
                estimated("cast and lightly polished: rough enough to smear the highlight"),
            )
            .done(),
        // A bell is a loss factor. η at 1e-4 is Q = 10 000, which is why a
        // struck bell is still audible ten seconds later and a struck brass
        // fitting is not.
        "bell bronze" | "bell metal" => B::new("bell bronze")
            .mech(8700.0, 105.0e9, 0.33, 1.0e-4)
            .rub(0.35, 0.5)
            .metal([0.22, 0.26, 0.41, 0.53, 0.64, 0.68], 0.25)
            .cite(
                measured("Rossing, Science of Percussion Instruments: Cu-Sn 78/22 bell metal"),
                measured(OPTICS4),
                estimated("cast and polished"),
            )
            .done(),
        "copper" => B::new("copper")
            .mech(8960.0, 117.0e9, 0.34, 2.0e-3)
            .rub(0.5, 0.4)
            .metal([0.500, 0.576, 0.560, 0.716, 0.930, 0.980], 0.20)
            .cite(measured(CRC), measured(OPTICS4), estimated("polished, unoxidised"))
            .done(),
        "iron" | "wrought iron" => B::new("iron")
            .mech(7870.0, 211.0e9, 0.29, 1.0e-3)
            .rub(0.6, 0.5)
            .metal([0.575, 0.585, 0.565, 0.575, 0.555, 0.565], 0.45)
            .cite(measured(CRC), measured(OPTICS4), estimated("dark and slightly rough"))
            .done(),
        "steel" => B::new("steel")
            .mech(7850.0, 200.0e9, 0.30, 5.0e-4)
            .rub(0.6, 0.6)
            .metal_flat([0.62, 0.64, 0.68], 0.35)
            .cite(
                measured(&format!("{ASHBY} (low-carbon steel); {ETB}: steel on steel")),
                measured(OPTICS4),
                estimated(LOOK),
            )
            .done(),
        "gold leaf" | "gold" => B::new("gold leaf")
            .mech(19300.0, 79.0e9, 0.44, 3.0e-4)
            .rub(0.5, 0.2)
            .metal([0.300, 0.372, 0.600, 0.932, 1.000, 1.000], 0.10)
            .cite(measured(CRC), measured(OPTICS4), estimated("beaten leaf: nearly a mirror"))
            .done(),
        "silver" | "mirror" => B::new("silver")
            .mech(10490.0, 83.0e9, 0.37, 2.0e-4)
            .rub(0.5, 0.3)
            .metal([0.900, 0.930, 0.955, 0.965, 0.970, 0.974], 0.02)
            .cite(measured(CRC), measured(OPTICS4), estimated("a fresh silvered surface"))
            .done(),
        "pewter" => B::new("pewter")
            .mech(7300.0, 45.0e9, 0.33, 5.0e-3)
            .rub(0.4, 0.2)
            .metal([0.60, 0.62, 0.62, 0.64, 0.63, 0.65], 0.40)
            .cite(
                estimated("Sn-Sb-Cu pewter: tin's density and a soft modulus"),
                estimated("near tin, achromatic"),
                estimated("satin, as a cast pewter piece is"),
            )
            .done(),
        // Zinc over steel: steel's mechanics, zinc's spangle.
        "galvanized" | "galvanised" => B::new("galvanized")
            .mech(7850.0, 200.0e9, 0.30, 5.0e-4)
            .rub(0.55, 0.5)
            .metal_flat([0.70, 0.72, 0.74], 0.45)
            .cite(measured(ASHBY), measured(OPTICS4), estimated(LOOK))
            .done(),
        // The court's hoop: steel under paint, so it is not a conductor at
        // the surface any more.
        "rim" => B::new("rim")
            .mech(7850.0, 200.0e9, 0.30, 5.0e-4)
            .rub(0.6, 0.6)
            .look([0.85, 0.30, 0.16], 0.35)
            .cite(measured(ASHBY), estimated("painted: an opaque dielectric over steel"), estimated(LOOK))
            .done(),

        // ── ceramics and organics ─────────────────────────────────────────
        "porcelain" => B::new("porcelain")
            .mech(2400.0, 70.0e9, 0.22, 3.0e-3)
            .rub(0.4, 0.5)
            .look([0.88, 0.845, 0.79], 0.15)
            .cite(
                measured(ASHBY),
                estimated("glazed: a smooth dielectric over an opaque body"),
                estimated("the warmth a glaze has, smooth enough to hold a soft sheen"),
            )
            .done(),
        // Thin enough to be translucent, which is the whole point of it.
        "bone china" => B::new("bone china")
            .mech(2450.0, 80.0e9, 0.23, 2.0e-3)
            .rub(0.4, 0.5)
            .look([0.92, 0.90, 0.86], 0.12)
            .sss(Subsurface::mm(0.35, 3.0))
            .cite(
                measured(ASHBY),
                estimated("glazed"),
                estimated("3 mm mean free path: a held-up cup glows at the rim"),
            )
            .done(),
        "oak" => B::new("oak")
            .mech(750.0, 11.0e9, 0.35, 0.010)
            .rub(0.5, 0.4)
            .look([0.62, 0.45, 0.28], 0.6)
            .cite(measured(WOOD), estimated("opaque"), estimated(LOOK))
            .done(),
        "maple" => B::new("maple")
            .mech(705.0, 12.6e9, 0.35, 0.010)
            .rub(0.5, 0.4)
            .look([0.78, 0.60, 0.36], 0.5)
            .cite(measured(WOOD), estimated("opaque"), estimated(LOOK))
            .done(),
        "teak" => B::new("teak")
            .mech(660.0, 13.0e9, 0.35, 0.012)
            .rub(0.4, 0.4)
            .look([0.48, 0.33, 0.19], 0.55)
            .cite(measured(WOOD), estimated("opaque"), estimated("oiled teak, dark and warm"))
            .done(),
        "driftwood" => B::new("driftwood")
            .mech(500.0, 8.0e9, 0.35, 0.030)
            .rub(0.5, 0.3)
            .look([0.62, 0.60, 0.55], 0.85)
            .cite(
                estimated("waterlogged and dried softwood: lighter and lossier than the timber"),
                estimated("opaque"),
                estimated("sun-bleached grey"),
            )
            .done(),
        "bamboo" => B::new("bamboo")
            .mech(700.0, 20.0e9, 0.30, 0.008)
            .rub(0.4, 0.5)
            .look([0.72, 0.62, 0.38], 0.4)
            .cite(
                measured("Janssen, Mechanical Properties of Bamboo (1991)"),
                estimated("opaque"),
                estimated("the culm's own polish"),
            )
            .done(),
        // A rope is not a solid; these are the effective constants of the laid
        // rope, not of a hemp fibre, and the citation says which.
        "rope (hemp)" | "rope" | "hemp" => B::new("rope (hemp)")
            .mech(700.0, 2.0e9, 0.30, 0.10)
            .rub(0.6, 0.1)
            .look([0.62, 0.55, 0.38], 0.95)
            .stiffness(3.0e3)
            .cite(
                estimated("laid hemp rope, effective: a bundle of fibres with air in it"),
                estimated("opaque"),
                estimated("hairy and pale"),
            )
            .done(),
        "leather" => B::new("leather")
            .mech(900.0, 0.5e9, 0.40, 0.10)
            .rub(0.6, 0.3)
            .look([0.34, 0.22, 0.14], 0.55)
            .stiffness(5.0e3)
            .cite(
                measured("Ashby, Materials and Design: vegetable-tanned leather"),
                estimated("opaque"),
                estimated("waxed and darkened"),
            )
            .done(),
        // The quiet end of the loss scale: nothing struck against felt rings.
        "wool felt" | "felt" => B::new("wool felt")
            .mech(300.0, 1.0e6, 0.30, 0.50)
            .rub(0.7, 0.05)
            .look([0.72, 0.70, 0.66], 1.0)
            .stiffness(1.0e3)
            .cite(
                measured("Beranek, Noise and Vibration Control: pressed wool felt"),
                estimated("opaque"),
                estimated("undyed wool, and as rough as a surface gets"),
            )
            .done(),
        "linen" => B::new("linen")
            .mech(400.0, 3.0e9, 0.30, 0.08)
            .rub(0.5, 0.1)
            .look([0.82, 0.79, 0.70], 0.9)
            .stiffness(2.0e3)
            .cite(
                estimated("woven flax, effective: fibre modulus over a cloth's own density"),
                estimated("opaque"),
                estimated("unbleached"),
            )
            .done(),
        "paper" => B::new("paper")
            .mech(800.0, 4.0e9, 0.20, 0.06)
            .rub(0.4, 0.1)
            .look([0.88, 0.87, 0.84], 0.85)
            .cite(
                measured("Niskanen, Paper Physics: machine-made bond paper"),
                estimated("opaque at this thickness"),
                estimated("bright, uncoated"),
            )
            .done(),
        "parchment" | "vellum" => B::new("parchment")
            .mech(1000.0, 2.0e9, 0.35, 0.10)
            .rub(0.4, 0.15)
            .look([0.86, 0.80, 0.66], 0.7)
            .sss(Subsurface::organic_mm(0.4, 1.5, 0.3))
            .cite(
                estimated("prepared animal skin: collagen, between leather and paper"),
                estimated("translucent"),
                estimated("1.5 mm mean free path: a sheet held to a candle lights through"),
            )
            .done(),
        "beeswax" | "wax" => B::new("beeswax")
            .mech(960.0, 0.3e9, 0.40, 0.20)
            .rub(0.4, 0.1)
            .look([0.90, 0.80, 0.50], 0.35)
            .sss(Subsurface::organic_mm(0.7, 6.0, 0.4))
            .stiffness(4.0e3)
            .cite(
                measured(&format!("{CRC} (beeswax density); modulus from Ashby's wax range")),
                estimated("translucent"),
                estimated("6 mm mean free path, which is why a candle glows and does not just shine"),
            )
            .done(),
        // Poisson's ratio of about zero is real and is cork's whole trick:
        // squeeze it and it does not bulge, which is why a cork goes back in.
        "cork" => B::new("cork")
            .mech(200.0, 0.02e9, 0.0, 0.15)
            .rub(0.6, 0.3)
            .look([0.72, 0.56, 0.36], 0.9)
            .stiffness(2.0e3)
            .cite(
                measured("Gibson & Ashby, Cellular Solids, ch. 9: Poisson's ratio ~0"),
                estimated("opaque"),
                estimated("natural cork"),
            )
            .done(),
        "rubber" => B::new("rubber")
            .mech(1100.0, 0.01e9, 0.49, 0.20)
            .rub(0.9, 0.8)
            .look([0.16, 0.16, 0.17], 0.7)
            .stiffness(5.0e3)
            .cite(
                measured(&format!("{ASHBY} (vulcanised natural rubber); {ETB}: rubber on concrete")),
                estimated("opaque"),
                estimated(LOOK),
            )
            .done(),
        "plywood" => B::new("plywood")
            .mech(600.0, 9.0e9, 0.30, 0.02)
            .rub(0.5, 0.35)
            .look([0.72, 0.55, 0.33], 0.7)
            .cite(measured(WOOD), estimated("opaque"), estimated(LOOK))
            .done(),
        "masonite" | "hardboard" => B::new("masonite")
            .mech(900.0, 4.0e9, 0.30, 0.03)
            .rub(0.45, 0.3)
            .look([0.42, 0.30, 0.22], 0.6)
            .cite(measured(WOOD), estimated("opaque"), estimated(LOOK))
            .done(),
        "cardboard" => B::new("cardboard")
            .mech(350.0, 2.0e9, 0.25, 0.08)
            .rub(0.5, 0.1)
            .look([0.76, 0.62, 0.42], 0.9)
            .cite(
                measured("Niskanen, Paper Physics: corrugated board, effective"),
                estimated("opaque"),
                estimated(LOOK),
            )
            .done(),
        // The printed track. These are `audio::PLA`'s four constants, and this
        // entry is where they are now stated once.
        "pla" => B::new("pla")
            .mech(1240.0, 3.5e9, 0.36, 0.03)
            .rub(0.45, 0.5)
            .look([0.60, 0.60, 0.62], 0.55)
            .cite(
                measured("Farah et al., Adv. Drug Deliv. Rev. 107 (2016): PLA physical properties"),
                estimated("opaque"),
                estimated(LOOK),
            )
            .done(),

        // ── the sea ───────────────────────────────────────────────────────
        "coral" => B::new("coral")
            .mech(2700.0, 60.0e9, 0.25, 0.02)
            .rub(0.7, 0.2)
            .look([0.86, 0.62, 0.55], 0.85)
            .sss(Subsurface::organic_mm(0.3, 2.0, 0.2))
            .cite(
                measured(&format!("{CRC} (aragonite); skeleton modulus from Ashby's ceramic range")),
                estimated("opaque skeleton with a translucent surface layer"),
                estimated("bleached pink"),
            )
            .done(),
        // Nacre is a stack of aragonite platelets a few hundred nanometres
        // thick, and the interference between them *is* the iridescence.
        "shell (nacre)" | "nacre" | "mother of pearl" => B::new("shell (nacre)")
            .mech(2700.0, 70.0e9, 0.28, 5.0e-3)
            .rub(0.4, 0.5)
            .look([0.86, 0.85, 0.82], 0.08)
            .film(380.0, 1.56)
            .cite(
                measured("Currey, Proc. R. Soc. B 196 (1977): nacre modulus and density"),
                measured(&format!("{RII} (aragonite, n ~1.53-1.68)")),
                measured("Jackson et al.: aragonite platelets 200-500 nm thick"),
            )
            .done(),
        "kelp" | "seaweed" => B::new("kelp")
            .mech(1050.0, 0.02e9, 0.45, 0.30)
            .rub(0.2, 0.05)
            .look([0.18, 0.20, 0.09], 0.4)
            .sss(Subsurface::organic_mm(0.5, 3.0, 0.5))
            .stiffness(1.0e3)
            .cite(
                estimated("hydrated kelp blade: near-seawater density, a rubbery modulus"),
                estimated("translucent"),
                estimated("dark olive, wet and glossy"),
            )
            .done(),
        // Ammophila arenaria, the grass that holds a dune together. A blade is
        // a rolled tube of thick-walled fibre with a waxy cuticle — that roll
        // is the plant's answer to salt wind — so it is stiffer, drier and
        // paler than a lawn grass, and it stands up instead of lying over.
        // The one plant the cove needs, and the look note in the design says
        // why: "the one thing that style needs and we do not have is grass".
        "marram grass" | "marram" => B::new("marram grass")
            .mech(600.0, 2.0e9, 0.35, 0.10)
            .rub(0.35, 0.15)
            .look([0.40, 0.44, 0.21], 0.65)
            .sss(Subsurface::organic_mm(0.4, 1.2, 0.6))
            .stiffness(3.0e3)
            .cite(
                estimated("a dry grass blade: half the density of water, a fibre modulus a thousandth of wood's along the leaf"),
                estimated("thin and translucent: a blade a third of a millimetre thick lights up against the sun"),
                estimated("grey-green, dusty, drier than any lawn"),
            )
            .done(),
        "sea foam" | "foam" => B::new("sea foam")
            .mech(100.0, 1.0e4, 0.30, 0.60)
            .rub(0.2, 0.02)
            .look([0.94, 0.95, 0.95], 0.9)
            .sss(Subsurface::mm(0.9, 8.0))
            .stiffness(200.0)
            .cite(
                estimated("a wet foam at roughly a tenth the density of water"),
                estimated("a scattering medium, not a surface"),
                estimated("as white as anything in the cove gets"),
            )
            .done(),

        // ── emitters ──────────────────────────────────────────────────────
        //
        // The sun is not a material. These are, and their radiance is in the
        // renderer's own units, where `studio_rig`'s key light sits at 4.2.
        "candle flame" | "flame" | "candle" => B::new("candle flame")
            .mech(0.3, 0.0, 0.0, 1.0)
            .rub(0.0, 0.0)
            .look([0.0, 0.0, 0.0], 1.0)
            .emit([18.0, 8.0, 2.2])
            .cite(
                estimated("hot air: a flame has no modulus and should not be asked for one"),
                estimated("emissive only"),
                estimated("about 1850 K, scaled to the renderer's units"),
            )
            .done(),
        "lamp" => B::new("lamp")
            .mech(2500.0, 70.0e9, 0.23, 1.0e-3)
            .rub(0.4, 0.4)
            .look([0.98, 0.94, 0.80], 0.2)
            .emit([12.0, 11.3, 9.2])
            .cite(
                measured(&format!("{CRC} (the glass envelope)")),
                estimated("emissive; the envelope is soda-lime"),
                estimated(LOOK),
            )
            .done(),

        // ── the court's painted surfaces ──────────────────────────────────
        //
        // Names that describe a *place*, not a substance. Each carries the
        // mechanics of what it is actually made of and the colour the window
        // has always drawn it with.
        "wall" => B::new("wall")
            .mech(1600.0, 5.0e9, 0.25, 0.03)
            .rub(0.6, 0.2)
            .look([0.80, 0.78, 0.72], 0.85)
            .cite(estimated("painted plaster on block"), estimated("opaque"), estimated(LOOK))
            .done(),
        "ceiling" => B::new("ceiling")
            .mech(1600.0, 5.0e9, 0.25, 0.03)
            .rub(0.6, 0.2)
            .look([0.86, 0.86, 0.84], 0.9)
            .cite(estimated("painted plaster"), estimated("opaque"), estimated(LOOK))
            .done(),
        "floor" => B::new("floor")
            .mech(705.0, 12.6e9, 0.35, 0.010)
            .rub(0.55, 0.45)
            .look([0.70, 0.55, 0.34], 0.25)
            .cite(measured(WOOD), estimated("lacquered maple: a clear coat over wood"), estimated(LOOK))
            .done(),
        "pad" => B::new("pad")
            .mech(80.0, 2.0e5, 0.30, 0.40)
            .rub(0.8, 0.1)
            .look([0.20, 0.30, 0.60], 0.8)
            .stiffness(2.0e3)
            .cite(
                estimated("vinyl over polyurethane foam: a wall pad"),
                estimated("opaque"),
                estimated(LOOK),
            )
            .done(),
        "window" => B::new("window")
            .mech(2500.0, 70.0e9, 0.23, 1.0e-3)
            .rub(0.5, 0.6)
            .look([0.70, 0.82, 0.90], 0.05)
            .cite(measured(CRC), estimated("glazing, drawn opaque by the window tier"), estimated(LOOK))
            .done(),
        "paint" => B::new("paint")
            .mech(1400.0, 2.0e9, 0.35, 0.05)
            .rub(0.5, 0.3)
            .look([0.95, 0.95, 0.95], 0.8)
            .cite(estimated("a latex film"), estimated("opaque"), estimated(LOOK))
            .done(),
        "key" => B::new("key")
            .mech(705.0, 12.6e9, 0.35, 0.010)
            .rub(0.55, 0.45)
            .look([0.55, 0.20, 0.18], 0.25)
            .cite(measured(WOOD), estimated("paint on a lacquered maple floor"), estimated(LOOK))
            .done(),

        // ── the cove's costume ────────────────────────────────────────────
        //
        // Rune's adventurer, part by part. Same rule as the court's names
        // above: each is a *place on the figure*, and each carries the
        // mechanics of the substance it is actually cut from — so `cloak` is
        // wool felt with a dye on it, `cream` is linen, `boot` is leather
        // that has been blacked and waxed — while the colour is the one
        // `sims/rune/hero/stage.rs` paints it, chosen against the sand, the
        // cliff and the door.
        //
        // The reason they are here rather than in the sim is
        // `Body::substance`: a hero whose parts resolve to substances is a
        // hero `kosm::player` can hang phyz joints on without a second
        // table — the cloak's 300 kg/m³ and the boot's μ = 0.6 come from
        // this file and nowhere else.
        "cloak" => B::new("cloak")
            .mech(300.0, 1.0e6, 0.30, 0.50)
            .rub(0.7, 0.05)
            .look([0.055, 0.42, 0.47], 0.75)
            .stiffness(1.0e3)
            .cite(
                measured("Beranek, Noise and Vibration Control: pressed wool felt"),
                estimated("opaque"),
                estimated("dyed a saturated cyan-teal: the one hue the cove does not already own"),
            )
            .done(),
        "cream" => B::new("cream")
            .mech(400.0, 3.0e9, 0.30, 0.08)
            .rub(0.5, 0.1)
            .look([0.93, 0.90, 0.78], 0.7)
            .stiffness(2.0e3)
            .cite(
                estimated("woven flax, effective: fibre modulus over a cloth's own density"),
                estimated("opaque"),
                estimated("bleached linen: the brightest thing on the figure, so the collar reads at 20 m"),
            )
            .done(),
        // Soft tissue, not a solid: the modulus is the dermis's, which is six
        // orders under bone and is why a finger deforms and a knuckle does
        // not. The scattering pair is the reason skin is not just a colour.
        "skin" => B::new("skin")
            .mech(1050.0, 0.5e6, 0.45, 0.30)
            .rub(0.6, 0.2)
            .look([0.88, 0.63, 0.47], 0.55)
            .sss(Subsurface::organic_mm(0.5, 2.6, 0.8))
            .stiffness(1.5e3)
            .cite(
                measured(&format!("{CRC} (soft tissue density); dermis modulus from Ashby's elastomer range")),
                measured("Jensen et al., A Practical Model for Subsurface Light Transport (2001), table 1: skin"),
                estimated("a warm mid tan, held twice as bright as the cove's stone and half as saturated as its sand"),
            )
            .done(),
        // The cheek tint. Skin under a rose wash, so the mechanics are skin's
        // to the digit and only the albedo moves.
        "blush" => B::new("blush")
            .mech(1050.0, 0.5e6, 0.45, 0.30)
            .rub(0.6, 0.2)
            .look([0.92, 0.46, 0.42], 0.55)
            .sss(Subsurface::organic_mm(0.5, 2.6, 0.8))
            .stiffness(1.5e3)
            .cite(
                measured(&format!("{CRC} (soft tissue density); dermis modulus from Ashby's elastomer range")),
                measured("Jensen et al. (2001), table 1: skin"),
                estimated("a rose wash over the same skin, flat enough to stay a Mii's cheek and not a shadow"),
            )
            .done(),
        // Lamp black bound in a film: the darkest thing the figure has, and
        // the only one whose whole job is to be a hole.
        "ink" => B::new("ink")
            .mech(1800.0, 2.0e9, 0.35, 0.05)
            .rub(0.4, 0.1)
            .look([0.015, 0.015, 0.02], 0.5)
            .cite(
                measured(&format!("{CRC} (carbon black, bound)")),
                estimated("opaque: a 1.5 % albedo is what a lamp-black film measures"),
                estimated("a dot eye, and nothing in the picture darker"),
            )
            .done(),
        "boot" | "waxed leather" => B::new("boot")
            .mech(900.0, 0.5e9, 0.40, 0.10)
            .rub(0.6, 0.3)
            .look([0.055, 0.06, 0.075], 0.55)
            .stiffness(5.0e3)
            .cite(
                measured("Ashby, Materials and Design: vegetable-tanned leather"),
                estimated("opaque"),
                estimated("blacked and waxed, and cool rather than warm so the feet plant the figure"),
            )
            .done(),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_canonical_name_resolves_to_itself() {
        for name in names() {
            let m = named(name).unwrap_or_else(|| panic!("`{name}` is listed but does not resolve"));
            assert_eq!(&m.name, name, "`{name}` resolved to a different canonical name");
        }
        assert!(named("unobtainium").is_none());
    }

    #[test]
    fn lookup_is_case_and_whitespace_insensitive() {
        assert_eq!(named("N-BK7"), named("n-bk7"));
        assert_eq!(named("BRASS"), named("  brass "));
        // an alias resolves to the canonical entry, and says so
        assert_eq!(named("gold").expect("gold").name, "gold leaf");
        assert_eq!(named("mirror").expect("mirror").name, "silver");
        assert_eq!(named("rock").expect("rock").name, "granite");
    }

    #[test]
    fn every_entry_carries_a_source_for_every_group() {
        for name in names() {
            let m = named(name).expect(name);
            for (group, cite) in m.provenance.groups() {
                assert!(
                    !cite.source.trim().is_empty(),
                    "`{name}` has no source for its {group}"
                );
                assert_ne!(cite.source, "unsourced", "`{name}`'s {group} is unsourced");
            }
        }
    }

    #[test]
    fn the_constants_are_physically_ordered() {
        // Density: the noble metals are the heavy end, foam the light one.
        let rho = |n: &str| named(n).expect(n).density;
        assert!(rho("gold leaf") > rho("silver"));
        assert!(rho("silver") > rho("steel"));
        assert!(rho("steel") > rho("granite"));
        assert!(rho("granite") > rho("water"));
        assert!(rho("water") > rho("oak"));
        assert!(rho("oak") > rho("sea foam"));

        // Loss: a bell rings, felt does not.
        let loss = |n: &str| named(n).expect(n).loss;
        assert!(loss("bell bronze") < loss("brass"));
        assert!(loss("brass") < loss("oak"));
        assert!(loss("oak") < loss("rubber"));
        assert!(loss("rubber") < loss("wool felt"));

        // Dispersion: lead crystal splits about twice as wide as N-BK7.
        let abbe = |n: &str| match named(n).expect(n).optics {
            Optics::Dielectric(d) => d.abbe,
            _ => panic!("`{n}` does not transmit"),
        };
        let index = |n: &str| named(n).expect(n).n_d().expect(n);
        assert!(abbe("lead crystal") < abbe("N-BK7"));
        assert!(index("lead crystal") > index("N-BK7"));
        assert!(index("N-BK7") > index("fused silica"));
    }

    #[test]
    fn the_metals_are_conductors_and_nothing_else_is() {
        let metals =
            ["brass", "bell bronze", "copper", "iron", "steel", "gold leaf", "silver", "pewter", "galvanized"];
        for name in metals {
            assert!(
                matches!(named(name).expect(name).optics, Optics::Conductor),
                "`{name}` should be a conductor"
            );
        }
        for name in ["oak", "granite", "rim", "porcelain"] {
            assert!(
                matches!(named(name).expect(name).optics, Optics::Opaque),
                "`{name}` should be opaque"
            );
        }
    }

    #[test]
    fn the_transmitters_transmit_and_the_liquids_flow() {
        for name in ["N-BK7", "soda-lime glass", "fused silica", "lead crystal", "water", "jelly"] {
            let m = named(name).expect(name);
            assert!(m.n_d().is_some(), "`{name}` should have an index");
            assert_eq!(m.pbr().transmission, 1.0, "`{name}` should transmit");
        }
        for name in ["water", "sea water", "soap film"] {
            assert!(named(name).expect(name).fluid().is_liquid(), "`{name}` should flow");
        }
        for name in ["granite", "steel", "oak"] {
            assert!(!named(name).expect(name).fluid().is_liquid(), "`{name}` should not flow");
        }
    }

    /// A `Material` survives the round trip through `datasheet.json`.
    ///
    /// To a ULP, not to the bit: `serde_json`'s parser takes a fast path that
    /// can land one unit in the last place away without the `float_roundtrip`
    /// feature, so this asserts what the format actually promises. Every
    /// constant, its name and its provenance come back.
    #[test]
    fn every_entry_serialises() {
        for name in names() {
            let m = named(name).expect(name);
            let json = serde_json::to_string(&m).expect(name);
            let back: Material = serde_json::from_str(&json).expect(name);
            assert_eq!(back.name, m.name);
            assert_eq!(back.provenance, m.provenance, "`{name}` lost its provenance");
            assert_eq!(back.optics.clone(), m.optics.clone(), "`{name}` lost its optics");
            let (was, now) = (m.params(), back.params());
            assert_eq!(was.len(), now.len(), "`{name}` changed shape");
            for (a, b) in was.iter().zip(&now) {
                assert_eq!(a.name, b.name);
                let d = (a.value - b.value).abs();
                assert!(
                    d <= 1e-12 * a.value.abs().max(1.0),
                    "`{name}`.{} went {} -> {}",
                    a.name,
                    a.value,
                    b.value
                );
            }
        }
    }
}
