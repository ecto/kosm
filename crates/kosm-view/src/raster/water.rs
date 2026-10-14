//! The sea, as the shader has it.
//!
//! `sims/pool/scene.wgsl` is the construction and this is the cove's version
//! of it: a surface at a fixed height whose normal comes from an analytic
//! swell, a Fresnel split between the sky reflected off it and the seabed
//! refracted through it, Beer–Lambert extinction over the path inside, a foam
//! band where it runs out of depth, and a horizon fade so the lattice's far
//! edge is not an edge.
//!
//! Two things differ from the pool's. The cove's seabed is a **plane at a
//! grade** rather than a flat floor — the sand runs on under the water at
//! `beach_slope` — so the refracted ray is intersected with that plane and
//! shaded with the *sand's* own material, which is the same entry the beach
//! above the waterline is drawn with. And the swell is the level's:
//! `sims/rune/render.rs::sea_field` sums two sine waves crossing at shallow
//! angles, and the same two are evaluated here in closed form rather than
//! sampled off a height lattice, so the raster's water and the tracer's are
//! the same surface written twice.
//!
//! ## What makes it teal
//!
//! Water is not glass, and the bug this module was rewritten to fix was that
//! it looked like glass: the refracted seabed came back at nearly full
//! brightness through eight hundred millimetres of ocean, so the raster tier
//! read clear until the settle blended the reference's teal in and the sea
//! changed colour under the player's feet.
//!
//! Two coefficients now stand between the seabed and the eye, both per metre
//! and both per linear-RGB channel:
//!
//! - [`Sea::absorption`] is what the water **takes**. It is `sea water`'s own
//!   `Dielectric::absorption` out of `kosm::material` — Pope & Fry's measured
//!   `a(λ)`, projected onto the three primaries by [`absorption_per_m`] — and
//!   it is small: 0.36 / 0.06 / 0.03 per metre. Clear water really is nearly
//!   clear over a metre, which is why absorption alone cannot be the answer.
//! - [`Sea::scatter`] is what it **gives back**. A real cove's water carries
//!   sand and plankton, and that is what turns the transmitted seabed into a
//!   field of the water's own colour within a metre. It is not a measurement
//!   — it is one knob, [`Sea::with_scatter`], spectrally flat, because sand
//!   grains are far larger than a wavelength and scatter grey (Mie, not
//!   Rayleigh); the colour is the absorption's — and it is what the reference tracer's
//!   deliberately *opaque* teal (`sims/rune/materials.rs`) is the limit of.
//!
//! The transmitted radiance is `e^{−(a+b)·d}` of the seabed's, and what is
//! not transmitted comes back as `ω·(1−e^{−(a+b)·d})` of the water's own
//! colour under the sky, where `ω = b/(a+b)` is the single-scattering albedo.
//! Deep water therefore tends toward the sea's colour and not toward black,
//! and red — which `ω` is lowest in — is the channel that goes first.
//!
//! ## The shore
//!
//! Two decorations that the reference tracer does **not** model, and that are
//! marked as raster-only wherever a parity number is taken:
//!
//! - [`Sea::foam`] is a band where the swell runs out of water: full inside
//!   [`Sea::foam_depth`] of the seabed, plus the crests of the swell while it
//!   is still shallow, with an edge that breathes with the swell's own phase
//!   so it is not a contour line drawn on the sea.
//! - [`Sea::wet`] is how wet the sand at a point is: one within the water and
//!   fading to nothing [`Sea::wet_band`] above the swell's current top. The
//!   shader spends it on `kosm::material::Material::wet`'s rule — albedo
//!   ×0.6, roughness ×0.75, and the dielectric highlight a wet grain has and
//!   a dry one does not.
//!
//! ```
//! use kosm_view::raster::Sea;
//! // the cove's authored swell: 30 mm at 7 m, 20 mm at 3 m
//! let sea = Sea::cove(0.0, 0.06, -20.0, 40.0).with_swell(0.03, 7.0, 0.02, 3.0);
//! // a surface, not a plane: the height wanders by the two amplitudes
//! let h = sea.height(3.0, -4.0, 0.0);
//! assert!(h.abs() <= 0.05 + 1e-9);
//! // and the water darkens what is under it: a metre of it keeps most of the
//! // blue and little of the red
//! let t = sea.transmittance(1.0);
//! assert!(t[0] < t[2] && t[2] < 1.0);
//! ```

use kosm::material::{self, Material, Optics};

/// The sea's knobs, in world metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sea {
    /// The flat waterline height.
    pub z: f64,
    /// The beach's grade, `dz/dy`, which the seabed carries on at.
    pub slope: f64,
    /// The `y` at which the sand meets the water.
    pub waterline_y: f64,
    /// How far out to sea the surface is drawn before the horizon fade has
    /// finished, metres.
    pub reach: f64,
    /// The first swell: amplitude and wavelength, metres, and the angle its
    /// crests run at either side of the shore normal, radians.
    pub swell: [f64; 2],
    pub swell_angle: f64,
    /// The second.
    pub swell_b: [f64; 2],
    pub swell_b_angle: f64,
    /// How fast the crests walk in, radians a second per wave.
    pub speed: f64,
    /// The index of the water.
    pub ior: f64,
    /// What a metre of it **takes out**, per linear RGB channel: the
    /// substance's own measured absorption, through [`absorption_per_m`].
    pub absorption: [f64; 3],
    /// What a metre of it **scatters**, per linear RGB channel: the level's
    /// one turbidity knob, through [`Sea::with_scatter`]. Extinction is the
    /// sum of the two and the single-scattering albedo is this over that.
    pub scatter: [f64; 3],
    /// The foam's albedo, linear RGB — `sea foam` out of the library.
    pub foam_albedo: [f64; 3],
    /// How deep the water may be before the shore's foam has gone, metres.
    pub foam_depth: f64,
    /// How much foam there is at all, 0..1. Zero switches the band off.
    pub foam_strength: f64,
    /// How far above the swell's current top the sand still reads wet,
    /// metres.
    pub wet_band: f64,
    /// The material the seabed is shaded with, and the one the surface's own
    /// body colour comes from.
    pub seabed_material: u32,
    pub water_material: u32,
}

/// Beer–Lambert absorption per metre, per linear RGB channel, out of a
/// substance's own interior.
///
/// `kosm::material` authors an [`Optics::Dielectric`]'s absorption as what one
/// `distance_m` of it *transmits*, six bands of it; a shader wants a
/// coefficient and has three channels. So: `a_b = −ln(T_b)/d`, then the two
/// bands each primary owns are averaged — the same pairing
/// `sims/rune/materials.rs::over` spreads an RGB albedo back over, so a
/// colour that goes one way and comes back is the colour it started as.
///
/// [`None`] when the substance does not transmit or takes nothing out.
///
/// ```
/// let sea = kosm::material::named("sea water").expect("sea water");
/// let a = kosm_view::raster::absorption_per_m(&sea).expect("sea water absorbs");
/// // Pope & Fry: red first, blue furthest
/// assert!(a[0] > a[1] && a[1] > a[2]);
/// assert!((a[0] - 0.357).abs() < 0.01);
/// ```
pub fn absorption_per_m(m: &Material) -> Option<[f64; 3]> {
    let Optics::Dielectric(d) = &m.optics else { return None };
    let abs = d.absorption.as_ref()?;
    let dist = abs.distance_m.max(1e-9);
    let a = |b: usize| -(abs.transmittance.bands[b].clamp(1e-9, 1.0)).ln() / dist;
    let out = [(a(4) + a(5)) * 0.5, (a(2) + a(3)) * 0.5, (a(0) + a(1)) * 0.5];
    out.iter().any(|v| *v > 0.0).then_some(out)
}

/// The cove's turbidity: how much a metre of its water scatters at 550 nm.
///
/// Not a measurement. It is the one number that decides how far you can see
/// into the sea, and it is set where the raster tier's water reaches the
/// reference tracer's opaque teal within about a metre of path — which is
/// what `sims/rune/materials.rs` authored the reference to be and why this
/// number is the level's and not the library's.
pub const COVE_SCATTER: f64 = 2.2;

/// The primaries' centres, nanometres — the mean of the two bands each one
/// owns in [`kosm::material::BANDS_NM`]. Kept for a level whose turbidity
/// is fine silt or plankton, which does tilt toward the blue; the cove's is
/// sand, and sand scatters grey.
#[allow(dead_code)]
const PRIMARY_NM: [f64; 3] = [645.0, 545.0, 445.0];

impl Sea {
    /// The cove's sea: a waterline, the beach's grade, where they meet, and
    /// how far out the surface reaches.
    ///
    /// The optics come out of `kosm::material`: `sea water`'s index and its
    /// absorption, `sea foam`'s albedo. Only the turbidity and the shore's
    /// two widths are the level's own, and each of those has a setter.
    pub fn cove(z: f64, slope: f64, waterline_y: f64, reach: f64) -> Self {
        let water = material::named("sea water");
        let ior = water.as_ref().and_then(|m| m.n_d()).unwrap_or(1.339);
        // The fallback is the entry's own numbers, so a library that lost the
        // line still draws the sea it drew before rather than a pane of glass.
        let absorption = water
            .as_ref()
            .and_then(absorption_per_m)
            .unwrap_or([0.3573, 0.0638, 0.0317]);
        let foam_albedo =
            material::named("sea foam").map_or([0.94, 0.95, 0.95], |m| m.colour());
        Self {
            z,
            slope,
            waterline_y,
            reach,
            // the level's own defaults, in metres — `swell_a_mm` and friends
            swell: [0.03, 7.0],
            swell_angle: 0.20,
            swell_b: [0.02, 3.0],
            swell_b_angle: -0.55,
            speed: 0.6,
            ior,
            absorption,
            scatter: [COVE_SCATTER; 3],
            foam_albedo,
            foam_depth: 0.06,
            foam_strength: 1.0,
            wet_band: 0.6,
            seabed_material: 0,
            water_material: 0,
        }
    }

    /// The two authored swell components, metres.
    pub fn with_swell(mut self, a1: f64, l1: f64, a2: f64, l2: f64) -> Self {
        self.swell = [a1, l1.max(1e-3)];
        self.swell_b = [a2, l2.max(1e-3)];
        self
    }

    pub fn with_materials(mut self, seabed: u32, water: u32) -> Self {
        self.seabed_material = seabed;
        self.water_material = water;
        self
    }

    /// The turbidity, as one scattering coefficient, the same in every
    /// primary: a suspension of grains much larger than the wavelength
    /// scatters grey, and the water's colour is then the absorption's alone,
    /// red first. (`tilted` is the `1/λ` law for a silt or plankton haze.)
    pub fn with_scatter(mut self, per_m: f64) -> Self {
        self.scatter = [per_m; 3];
        self
    }

    /// The shore's foam: how deep the water may be before it has gone, and
    /// how much of it there is.
    pub fn with_foam(mut self, depth_m: f64, strength: f64) -> Self {
        self.foam_depth = depth_m.max(1e-4);
        self.foam_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// How far above the water the sand still reads wet, metres.
    pub fn with_wet_band(mut self, band_m: f64) -> Self {
        self.wet_band = band_m.max(0.0);
        self
    }

    /// Extinction per metre, per channel: what is absorbed plus what is
    /// scattered out of the path.
    pub fn extinction(&self) -> [f64; 3] {
        [
            self.absorption[0] + self.scatter[0],
            self.absorption[1] + self.scatter[1],
            self.absorption[2] + self.scatter[2],
        ]
    }

    /// The single-scattering albedo `b/(a+b)`: how much of what the water
    /// stops it gives back rather than eats. This is what keeps deep water
    /// the sea's own colour instead of black.
    pub fn scattering_albedo(&self) -> [f64; 3] {
        let e = self.extinction();
        [
            self.scatter[0] / e[0].max(1e-9),
            self.scatter[1] / e[1].max(1e-9),
            self.scatter[2] / e[2].max(1e-9),
        ]
    }

    /// What a path of `d` metres inside the water transmits, per channel.
    pub fn transmittance(&self, d: f64) -> [f64; 3] {
        let e = self.extinction();
        [(-e[0] * d).exp(), (-e[1] * d).exp(), (-e[2] * d).exp()]
    }

    /// The surface's height above [`Self::z`] at a point and a time.
    ///
    /// The same sum `sea_field` fills its lattice with, evaluated rather than
    /// sampled. The shader has this function too; this one is what the tests
    /// hold it to, and what the CPU side uses to place the sea's own mesh.
    pub fn height(&self, x: f64, y: f64, t: f64) -> f64 {
        let (c1, s1) = self.swell_angle.sin_cos();
        let (c2, s2) = self.swell_b_angle.sin_cos();
        let k1 = std::f64::consts::TAU / self.swell[1];
        let k2 = std::f64::consts::TAU / self.swell_b[1];
        self.swell[0] * (k1 * (x * c1 + y * s1) - self.speed * t).sin()
            + self.swell_b[0] * (k2 * (x * c2 + y * s2) + 1.7 - self.speed * t).sin()
    }

    /// The surface's unit normal, from the analytic gradient of [`Self::height`].
    pub fn normal(&self, x: f64, y: f64, t: f64) -> [f64; 3] {
        let (c1, s1) = self.swell_angle.sin_cos();
        let (c2, s2) = self.swell_b_angle.sin_cos();
        let k1 = std::f64::consts::TAU / self.swell[1];
        let k2 = std::f64::consts::TAU / self.swell_b[1];
        let p1 = k1 * (x * c1 + y * s1) - self.speed * t;
        let p2 = k2 * (x * c2 + y * s2) + 1.7 - self.speed * t;
        let (a1, a2) = (self.swell[0] * k1 * p1.cos(), self.swell_b[0] * k2 * p2.cos());
        let dx = a1 * c1 + a2 * c2;
        let dy = a1 * s1 + a2 * s2;
        let n = [-dx, -dy, 1.0];
        let l = (n[0] * n[0] + n[1] * n[1] + 1.0).sqrt();
        [n[0] / l, n[1] / l, n[2] / l]
    }

    /// The seabed's height at a point: the beach's own plane, carried on
    /// under the water.
    pub fn seabed_z(&self, y: f64) -> f64 {
        self.z + self.slope * (y - self.waterline_y)
    }

    /// How much water there is over the seabed under a point of the surface,
    /// metres. Never negative: where the swell's trough is under the sand the
    /// answer is nothing, and that is where the foam is.
    pub fn depth(&self, x: f64, y: f64, t: f64) -> f64 {
        (self.z + self.height(x, y, t) - self.seabed_z(y)).max(0.0)
    }

    /// The foam at a point of the surface, 0..1.
    ///
    /// **A raster-only decoration.** The reference tracer draws the sea as one
    /// smooth opaque body and has no foam in it at all, so wherever the two
    /// tiers are differenced the foam's band is excluded from the region — see
    /// `sims/rune/game.rs`'s cove parity test, which measures the sea beyond
    /// two metres of the waterline for exactly this reason.
    ///
    /// Two terms. The shore's band is full where the water has run out and
    /// gone by [`Self::foam_depth`], and its edge is displaced by the swell's
    /// own phase so the band breathes in and out with the sets rather than
    /// sitting on the beach as a contour. The crests' term is the top of the
    /// swell while the water is still shallow, which is a wave breaking.
    ///
    /// The shader has this function too, in `shaders/scene.wgsl`; the test
    /// below is what holds the two together.
    pub fn foam(&self, x: f64, y: f64, t: f64) -> f64 {
        if self.foam_strength <= 0.0 {
            return 0.0;
        }
        let e = self.foam_depth.max(1e-4);
        let amp = (self.swell[0] + self.swell_b[0]).max(1e-4);
        let h = self.height(x, y, t);
        let d = self.depth(x, y, t);
        let edge = (e * (1.0 + 0.6 * h / amp)).max(1e-4);
        let band = 1.0 - smoothstep(0.0, edge, d);
        let crest =
            smoothstep(0.45 * amp, 0.95 * amp, h) * (1.0 - smoothstep(8.0 * e, 30.0 * e, d));
        (self.foam_strength * band.max(crest)).clamp(0.0, 1.0)
    }

    /// How wet the ground at a point is, 0..1: one at and under the water,
    /// nothing [`Self::wet_band`] above the swell's current top.
    ///
    /// Raster-only, for the same reason [`Self::foam`] is: the tracer paints
    /// the whole beach with one `dry sand`, so the wet band is a divergence
    /// and the parity region stays clear of it.
    pub fn wet(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        if self.wet_band <= 0.0 {
            return f64::from(z <= self.z + self.height(x, y, t));
        }
        let level = self.z + self.height(x, y, t);
        1.0 - smoothstep(0.0, self.wet_band, z - level)
    }

    /// A flat lattice for the surface, `n × n` quads over the sea's own
    /// footprint. The swell is applied in the *vertex* shader from
    /// [`Self::height`], so this carries no heights of its own — which is why
    /// a moving swell costs one uniform and not a re-upload.
    pub fn lattice(&self, half_x: f64, n: u32) -> (Vec<[f64; 3]>, Vec<u32>) {
        let n = n.max(1);
        let (y0, y1) = (self.waterline_y - self.reach, self.waterline_y);
        let mut p = Vec::with_capacity(((n + 1) * (n + 1)) as usize);
        for j in 0..=n {
            let y = y0 + (y1 - y0) * j as f64 / n as f64;
            for i in 0..=n {
                let x = -half_x + 2.0 * half_x * i as f64 / n as f64;
                p.push([x, y, self.z]);
            }
        }
        let mut idx = Vec::with_capacity((n * n * 6) as usize);
        for j in 0..n {
            for i in 0..n {
                let a = j * (n + 1) + i;
                idx.extend_from_slice(&[a, a + 1, a + n + 2]);
                idx.extend_from_slice(&[a, a + n + 2, a + n + 1]);
            }
        }
        (p, idx)
    }
}

/// One scattering coefficient at 550 nm, spread over the three primaries with
/// the `1/λ` tilt a particulate suspension has.
#[allow(dead_code)]
fn tilted(per_m_550: f64) -> [f64; 3] {
    let b = per_m_550.max(0.0);
    [
        b * 550.0 / PRIMARY_NM[0],
        b * 550.0 / PRIMARY_NM[1],
        b * 550.0 / PRIMARY_NM[2],
    ]
}

/// WGSL's `smoothstep`, so the two copies of [`Sea::foam`] cannot drift.
fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a).max(1e-12)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The normal is the analytic gradient of the height — checked against a
    /// central difference, because a swell whose normal does not match its
    /// surface is a sea with the light coming from the wrong place.
    #[test]
    fn the_normal_is_the_gradient_of_the_height() {
        let sea = Sea::cove(0.0, 0.06, -20.0, 40.0);
        let h = 1e-5;
        for (x, y, t) in [(0.0, -5.0, 0.0), (3.7, -12.3, 1.4), (-8.1, -21.0, 3.0)] {
            let dx = (sea.height(x + h, y, t) - sea.height(x - h, y, t)) / (2.0 * h);
            let dy = (sea.height(x, y + h, t) - sea.height(x, y - h, t)) / (2.0 * h);
            let want = {
                let n = [-dx, -dy, 1.0];
                let l = (n[0] * n[0] + n[1] * n[1] + 1.0).sqrt();
                [n[0] / l, n[1] / l, n[2] / l]
            };
            let got = sea.normal(x, y, t);
            for a in 0..3 {
                assert!((got[a] - want[a]).abs() < 1e-5, "at ({x}, {y}): {got:?} vs {want:?}");
            }
        }
    }

    /// The seabed is the beach's own plane: at the waterline they are the
    /// same height, and it drops going out to sea.
    #[test]
    fn the_seabed_is_the_beach_carried_on() {
        let sea = Sea::cove(1.5, 0.06, -20.0, 40.0);
        assert!((sea.seabed_z(-20.0) - 1.5).abs() < 1e-12);
        assert!(sea.seabed_z(-30.0) < sea.seabed_z(-20.0));
    }

    /// The lattice is closed and covers the footprint exactly once.
    #[test]
    fn the_lattice_closes_over_its_footprint() {
        let sea = Sea::cove(0.0, 0.06, -20.0, 40.0);
        let (p, idx) = sea.lattice(20.0, 4);
        assert_eq!(p.len(), 25);
        assert_eq!(idx.len(), 4 * 4 * 6);
        assert!(idx.iter().all(|&i| (i as usize) < p.len()));
        assert_eq!(p[0], [-20.0, -60.0, 0.0]);
        assert_eq!(p[24], [20.0, -20.0, 0.0]);
    }

    /// **The bug this module was rewritten for.** Eight hundred millimetres of
    /// water is not a pane of glass: it keeps well under half of what the
    /// seabed sends up, and it keeps far more blue than red. Five centimetres
    /// of it, at the waterline, is a film — most of the sand still comes
    /// through.
    #[test]
    fn a_metre_of_the_cove_is_not_a_pane_of_glass() {
        let sea = Sea::cove(0.0, 0.06, -20.0, 40.0);
        let reef = sea.transmittance(0.8);
        assert!(reef[0] < 0.25, "the reef's red comes through at {:.3}", reef[0]);
        assert!(reef[2] < 0.30 && reef[2] > reef[0], "blue {:.3}, red {:.3}", reef[2], reef[0]);
        let film = sea.transmittance(0.05);
        assert!(film[1] > 0.80, "five centimetres should read as wet sand: {:.3}", film[1]);
        // and what it stops it mostly gives back, so the deep water is teal
        // and not a hole in the frame
        let w = sea.scattering_albedo();
        assert!(w[2] > 0.95 && w[0] > 0.75 && w[0] < w[1], "scattering albedo {w:?}");
    }

    /// The absorption is the library's, not a number typed in here: change
    /// `sea water`'s spectrum and the cove's sea changes with it.
    #[test]
    fn the_absorption_comes_out_of_the_library() {
        let m = material::named("sea water").expect("sea water");
        let a = absorption_per_m(&m).expect("sea water absorbs");
        assert_eq!(Sea::cove(0.0, 0.0, 0.0, 1.0).absorption, a);
        // Pope & Fry's shape: red is an order of magnitude over blue
        assert!(a[0] > 5.0 * a[2], "{a:?}");
        // and the index is the library's too
        assert!((Sea::cove(0.0, 0.0, 0.0, 1.0).ior - 1.339).abs() < 1e-9);
    }

    /// The foam is where the water runs out and nowhere else: full at the
    /// waterline, gone by a knee-depth, and it moves with the swell.
    #[test]
    fn the_foam_is_a_band_at_the_waterline() {
        let sea = Sea::cove(0.0, 0.06, -20.0, 40.0).with_foam(0.06, 1.0);
        // the band peaks where the water runs out, which the swell moves a
        // few centimetres either side of the authored waterline
        let peak = (0..=24).map(|i| sea.foam(0.0, -20.6 + i as f64 * 0.05, 0.0)).fold(0.0, f64::max);
        assert!(peak > 0.9, "the foam's peak near the waterline is {peak}");
        // ten metres out is six hundred millimetres of water: no foam
        assert!(sea.foam(0.0, -30.0, 0.0) < 1e-3, "{}", sea.foam(0.0, -30.0, 0.0));
        // and the band is not a contour: it moves between two times
        let a: f64 = (0..40).map(|i| sea.foam(i as f64 * 0.25, -21.0, 0.0)).sum();
        let b: f64 = (0..40).map(|i| sea.foam(i as f64 * 0.25, -21.0, 2.3)).sum();
        assert!((a - b).abs() > 1e-3, "the foam does not move: {a:.4} against {b:.4}");
        // switched off, it is off
        assert_eq!(sea.with_foam(0.06, 0.0).foam(0.0, -20.0, 0.0), 0.0);
    }

    /// The wet band is the sand the sea reaches: one under the water, nothing
    /// a band above it, monotone in between.
    #[test]
    fn the_wet_band_fades_up_the_beach() {
        let sea = Sea::cove(0.0, 0.06, -20.0, 40.0).with_wet_band(0.6);
        let level = sea.height(0.0, -20.0, 0.0);
        assert!((sea.wet(0.0, -20.0, level - 0.1, 0.0) - 1.0).abs() < 1e-12);
        assert_eq!(sea.wet(0.0, -20.0, level + 0.7, 0.0), 0.0);
        let mut last = 1.0;
        for i in 0..=12 {
            let w = sea.wet(0.0, -20.0, level + i as f64 * 0.05, 0.0);
            assert!(w <= last + 1e-12, "the band is not monotone at {i}");
            last = w;
        }
    }
}
