//! The physics kit: a lens, a prism and a mirror.
//!
//! Rune's puzzles are optics, so the adventurer's inventory is optics. Three
//! tools, each a real solid of a real glass or metal, each cut to a number
//! that is written down here and checked by a test — because the whole claim
//! of this game is that the light in the picture is the light the puzzle is
//! solved with, and a lens whose focal length is a guess breaks that claim on
//! the first frame.
//!
//! # The lens
//!
//! A symmetric biconvex lens, 220 mm across, cut for **f = 2.5 m** — the
//! distance across the cove's doorstep, so that a hero standing back from the
//! door and holding it up to the sun puts the sun's image on the stone rather
//! than a bright smear.
//!
//! Two spheres of radius `R` whose centres are `2a` apart make a biconvex
//! lens with
//!
//! ```text
//! a = sqrt(R² − h²)          h = 110 mm, the semi-diameter
//! d = 2(R − a)               the centre thickness
//! ```
//!
//! and its focal length is the *thick* lensmaker's equation, with `R1 = +R`
//! and `R2 = −R`:
//!
//! ```text
//! 1/f = (n − 1) [ 2/R − (n − 1) d / (n R²) ]
//! ```
//!
//! `f` rises monotonically with `R`, so [`radius_for`] bisects it. At
//! n = 1.5168 (N-BK7's d line) the answer is
//!
//! ```text
//! R ≈ 2584 mm      a ≈ 2581.7 mm      d ≈ 4.7 mm      f = 2500 mm
//! ```
//!
//! — a wafer, which is what a 220 mm lens at f/11 is. That is why it is
//! carried in a brass ring: the ring is what a hand can hold and what an eye
//! can see, and it swallows the knife edge where the two caps meet.
//!
//! # The prism
//!
//! Equilateral, 150 mm on a side, 200 mm long, and **lead crystal, not
//! N-BK7**. At apex `A = 60°` the minimum deviation is
//!
//! ```text
//! δ = 2 asin(n sin(A/2)) − A
//! ```
//!
//! and the same formula at the ends of the visible band is the spectrum's
//! whole width. The glass is the only lever on it that matters:
//!
//! | glass | n_d | V | δ at n_d | 400–700 nm spread |
//! |-------|-----|---|----------|-------------------|
//! | N-BK7 | 1.5168 | 64 | 38.6° | 1.6° |
//! | lead crystal | 1.600 | 33 | 46.3° | 3.9° |
//!
//! A spectrum through a slit of width `w` thrown `D` is `w + D·Δ` long and
//! `w/(w + D·Δ)` as bright as bare sun, and both of those are fixed by `Δ`
//! alone. At N-BK7's 1.6° a **300 mm** band needs eleven metres of beach; at
//! lead crystal's 3.9° it needs four, and four metres fits in a frame. So the
//! adventurer's prism is flint and the lens is crown — which is what an
//! optician would do anyway, because a prism is for splitting and a lens is
//! for focusing and low dispersion is a *virtue* in the second job.
//!
//! [`aim`] is what turns it: the exit of a prism at minimum deviation lies on
//! a cone about the incoming sunbeam whatever the prism's own orientation is,
//! so "throw the spectrum at that stone" is a one-dimensional search.
//!
//! # The mirror
//!
//! A polished disc 250 mm across on a short handle, roughness 0.02. It makes
//! no caustic — [`kosm_render::caustics`] deposits a photon only if a
//! *transmissive solid* bent it, so a metal is invisible to the photon pass —
//! and it does not need to: a sun patch off a near-mirror is a
//! diffuse-then-specular path the integrator finds by BSDF sampling, which is
//! the one specular case an ordinary path tracer is good at.

use kosm::build::{Built, Params, build};
use kosm_render::TriMesh;
use kosm_render::math::{Point3, Transform, Vec3};

use super::stage::{self, N_D};

/// The lens, across.
pub const LENS_D: f64 = 220.0;
/// What it is cut for.
pub const LENS_F: f64 = 2500.0;
/// The prism: an equilateral triangle this far on a side, this long.
pub const PRISM_SIDE: f64 = 150.0;
pub const PRISM_LEN: f64 = 200.0;
/// What it is cut from. `kosm::material`'s entry, and the name
/// [`super::stage::palette`] answers with [`super::stage::flint`].
pub const PRISM_GLASS: &str = "lead crystal";
/// How wide the slit in the prism's stop is. See [`hardware`].
pub const PRISM_SLIT: f64 = 60.0;
/// The mirror, across, and how thick the disc is.
pub const MIRROR_D: f64 = 250.0;
pub const MIRROR_T: f64 = 14.0;

// ---- the lens's arithmetic -------------------------------------------------

/// The centre thickness of a symmetric biconvex lens of surface radius `r`
/// and semi-diameter `h`: twice the sagitta of one cap.
pub fn centre_thickness(r: f64, h: f64) -> f64 {
    2.0 * (r - (r * r - h * h).max(0.0).sqrt())
}

/// The focal length of that lens at index `n`, by the thick lensmaker's
/// equation. Same units in and out.
pub fn focal(r: f64, h: f64, n: f64) -> f64 {
    let d = centre_thickness(r, h);
    1.0 / ((n - 1.0) * (2.0 / r - (n - 1.0) * d / (n * r * r)))
}

/// The surface radius that gives focal length `f`. Bisected, because `focal`
/// is monotone in `r` and a closed form would be a quartic nobody would
/// check.
pub fn radius_for(f: f64, h: f64, n: f64) -> f64 {
    let (mut lo, mut hi) = (h * 1.0001, 400.0 * f);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if focal(mid, h, n) < f { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

/// The lens's own numbers: surface radius, half the centre separation, and
/// the centre thickness. Millimetres.
pub fn lens_numbers() -> (f64, f64, f64) {
    let h = 0.5 * LENS_D;
    let r = radius_for(LENS_F, h, N_D);
    let a = (r * r - h * h).max(0.0).sqrt();
    (r, a, centre_thickness(r, h))
}

/// How wide the converging cone still is when it has only travelled
/// `throw` of its focal length: what the bright patch on a surface at that
/// distance actually measures, and therefore how many times brighter than
/// bare sun it is (`(D/patch)²`).
pub fn patch_at(throw: f64) -> f64 {
    (LENS_D * (1.0 - throw / LENS_F)).abs()
}

// ---- the prism's arithmetic ------------------------------------------------

/// The minimum deviation of an equilateral prism at index `n`, radians.
pub fn min_deviation(n: f64) -> f64 {
    let apex = std::f64::consts::FRAC_PI_3;
    2.0 * (n * (apex / 2.0).sin()).asin() - apex
}

/// How far apart the ends of the visible band come out, radians: the whole
/// of what makes a spectrum a spectrum.
pub fn dispersion(n_blue: f64, n_red: f64) -> f64 {
    min_deviation(n_blue) - min_deviation(n_red)
}

/// N-BK7 at a wavelength, from the Sellmeier pair the glass carries. The same
/// curve `kosm-render` refracts with, so the arithmetic in this file and the
/// photons in the picture cannot disagree.
pub fn index_at(lambda_nm: f64) -> f64 {
    let (b, c) = kosm_render::spectrum::BK7_SELLMEIER;
    let l2 = (lambda_nm / 1000.0) * (lambda_nm / 1000.0);
    (1.0 + (0..3).map(|i| b[i] * l2 / (l2 - c[i])).sum::<f64>()).sqrt()
}

/// The prism's glass at a wavelength.
///
/// Lead crystal has an Abbe number in the library and no Sellmeier pair, so
/// this is [`kosm_render::spectrum::cauchy_index`] — which is *exactly* what
/// the renderer falls back to for a dispersive material without coefficients
/// (`cpu::material::Material::index_at`). Same function, same two numbers:
/// the band this file predicts and the band the photons draw are one band.
pub fn prism_index_at(lambda_nm: f64) -> f64 {
    let (n_d, abbe) = prism_glass();
    kosm_render::spectrum::cauchy_index(n_d, abbe, lambda_nm / 1000.0)
}

/// The prism glass's `(n_d, V)`, from the library, so this file has no
/// optical constant of its own to drift.
pub fn prism_glass() -> (f64, f64) {
    let m = kosm::material::named(PRISM_GLASS).expect("the prism's glass is in the library");
    let p = m.pbr();
    (p.ior as f64, p.abbe as f64)
}

/// The prism's deviation at the d line, and how wide it spreads the visible
/// band. Radians, and the two numbers the whole staging of `tools.png` turns
/// on.
pub fn prism_deviation() -> (f64, f64) {
    let (n_d, _) = prism_glass();
    let (blue, red) = (prism_index_at(400.0), prism_index_at(700.0));
    (min_deviation(n_d), min_deviation(blue) - min_deviation(red))
}

/// How long the band is on a screen `throw` away whose normal makes an angle
/// with the beam of `cos_incidence`, from a slit of width `slit`.
///
/// `slit + throw · (Δ + sun)` square to the beam, stretched by the screen's
/// tilt. The sun's own half-degree is in there because at four metres it is
/// 37 mm and the slit is 60: leaving it out would predict a band a fifth
/// crisper than the one the tracer draws.
pub fn band_length(slit: f64, throw: f64, cos_incidence: f64) -> f64 {
    let (_, spread) = prism_deviation();
    (slit + throw * (spread + SUN_ANGULAR_DIAMETER)) / cos_incidence.abs().max(1e-3)
}

/// How bright that band is against bare sun: a slit passes a slit's worth of
/// light and the spread only ever dilutes it.
pub fn band_brightness(slit: f64, throw: f64) -> f64 {
    let (_, spread) = prism_deviation();
    slit / (slit + throw * (spread + SUN_ANGULAR_DIAMETER))
}

/// The sun's angular diameter, radians. The floor under every caustic in
/// this file: no slit image and no lens focus is ever sharper than this.
pub const SUN_ANGULAR_DIAMETER: f64 = 0.0093;

// ---- aiming ----------------------------------------------------------------

/// Where a ray leaving `from` along `dir` meets the beach.
///
/// The sand is the plane `z = SLOPE · y` (see [`super::stage`]), so this is
/// one line of algebra and not a trace — which matters, because it is called
/// inside a bisection.
pub fn land_on_sand(from: Point3, dir: Vec3) -> Option<Point3> {
    let denom = dir.z - stage::SLOPE * dir.y;
    if denom.abs() < 1e-9 {
        return None;
    }
    let t = (stage::SLOPE * from.y - from.z) / denom;
    (t > 0.0).then(|| from + dir * t)
}

/// The same for any plane: where a ray leaving `from` along `dir` meets the
/// plane through `on` with normal `n`.
///
/// `tools.png` throws all three tools at one standing stone rather than at
/// the sand, so this is the function [`aim_at_plane`] bisects on.
pub fn land_on_plane(from: Point3, dir: Vec3, on: Point3, n: Vec3) -> Option<Point3> {
    let denom = dir.dot(n);
    if denom.abs() < 1e-9 {
        return None;
    }
    let t = (on - from).dot(n) / denom;
    (t > 0.0).then(|| from + dir * t)
}

/// A direction at `half_angle` to `axis`, spun by `spin` about it.
///
/// The exit of a prism at minimum deviation lies on exactly this cone about
/// the incoming sunbeam, whatever the prism's own orientation is — which is
/// what makes "turn the prism until the spectrum grazes the sand" a
/// one-dimensional search instead of a three-dimensional one.
pub fn on_cone(axis: Vec3, half_angle: f64, spin: f64) -> Vec3 {
    let a = axis.normalize();
    let up = Vec3::new(0.0, 0.0, 1.0);
    let u = (up - a * up.dot(a)).normalize();
    let v = a.cross(u);
    (a * half_angle.cos() + (u * spin.cos() + v * spin.sin()) * half_angle.sin()).normalize()
}

/// The spin that throws the prism's exit beam `throw` millimetres down the
/// sand from `from`.
///
/// At a spin of zero the exit is the highest point of the cone — for the
/// cove's sun and an equilateral prism that is 16° *above* the horizon, and
/// the spectrum never comes down at all. Turning the prism walks the exit
/// down the cone and the throw shortens all the way to the far side, where
/// the beam is 60° below the horizon and lands at the prism's own foot. So
/// the reach falls monotonically from infinity and a bisection finds the
/// turn that lands it where it was asked. Returns the spin and the exit it
/// picks.
pub fn aim(from: Point3, axis: Vec3, half_angle: f64, throw: f64) -> (f64, Vec3) {
    let reach = |spin: f64| {
        let d = on_cone(axis, half_angle, spin);
        match land_on_sand(from, d) {
            Some(p) => (p - from).norm(),
            None => f64::INFINITY, // over the horizon: too far, by definition
        }
    };
    let (mut lo, mut hi) = (0.0, std::f64::consts::PI);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if reach(mid) > throw { lo = mid } else { hi = mid }
    }
    let spin = 0.5 * (lo + hi);
    (spin, on_cone(axis, half_angle, spin))
}

/// A direction leaving `from` at `azimuth` that comes down on the sand
/// exactly `throw` away.
///
/// Closed form, not a search: the beach is a plane, so travelling `t` along
/// `(cos a cos β, sin a cos β, −sin β)` drops `t sin β` while the sand under
/// it falls by `slope · t cos β sin a`. Setting the two equal at `t = throw`
/// leaves `sin β + k cos β = h/throw` with `k = slope sin a`, which is one
/// harmonic addition away from `β`. The mirror is aimed with it.
pub fn descend(from: Point3, azimuth: f64, throw: f64) -> Option<Vec3> {
    let (sa, ca) = azimuth.sin_cos();
    let k = stage::SLOPE * sa;
    let h = from.z - stage::sand_z(from.y);
    let c = h / throw.max(1e-6) / (1.0 + k * k).sqrt();
    if !(-1.0..=1.0).contains(&c) {
        return None; // no elevation puts it there
    }
    let beta = c.asin() - k.atan();
    let (sb, cb) = beta.sin_cos();
    Some(Vec3::new(ca * cb, sa * cb, -sb).normalize())
}

/// Where a tool has to stand for its beam to land on `target`.
///
/// The inverse of [`aim`], and the one that stages a picture. `aim` asks
/// "given where the prism is, which way can it throw?" — a bisection, because
/// the beach is a plane and the cone is not. This asks "given where the light
/// has to land, where does the prism go?", which is not a search at all: pick
/// a spin, take the exit direction off the cone, and step back along it.
///
/// So a still that wants three marks of light on one stone chooses the three
/// marks first and lets the tools fall where they must. That is the whole
/// difference between `tools.png` as it was — three tools placed on the sand
/// and their light wherever it went — and as it is.
pub fn place_for(target: Point3, axis: Vec3, half_angle: f64, spin: f64, throw: f64) -> (Point3, Vec3) {
    let out = on_cone(axis, half_angle, spin);
    (target - out * throw, out)
}

/// The prism's placement, from the beam that goes in and the beam that comes
/// out./// The prism's placement, from the beam that goes in and the beam that comes
/// out.
///
/// At minimum deviation the ray *inside* the glass runs square to the apex
/// bisector — it enters one of the two faces that meet at the apex and leaves
/// by the other, crossing the glass sideways — and the two outside rays are
/// symmetric about that bisector, each tilted `δ/2` toward the apex. So
///
/// ```text
/// in  = cos(δ/2) travel + sin(δ/2) apex
/// out = cos(δ/2) travel − sin(δ/2) apex
/// ```
///
/// and the frame is forced by inverting those two: the way the light works
/// its way across is `in + out`, the apex is `in − out`, and the refracting
/// edge is what is left. [`prism_mesh`] is built with its apex down its own
/// −y and its edge along +z, so `y` here is the direction *from* the apex
/// into the glass, which is `out − in`.
///
/// Getting these two the wrong way round — putting the apex along the travel
/// — builds a prism the beam enters through a face and leaves through the
/// *base*, at a deviation nothing predicted; it is a mistake that costs
/// nothing at build time and shows up only as a spectrum that is not where
/// the arithmetic said, which is exactly what [`super::tools`] prints the
/// irradiance at three points to catch.
pub fn prism_frame(centre: Point3, into: Vec3, out_of: Vec3) -> Transform {
    let travel = (into.normalize() + out_of.normalize()).normalize();
    let into_glass = (out_of.normalize() - into.normalize()).normalize();
    let edge = travel.cross(into_glass).normalize();
    frame(centre, travel, into_glass, edge)
}

/// A flat mirror's placement: the disc is built facing local +z, so its frame
/// is the one whose +z is the normal that turns `into` into `out_of`.
pub fn mirror_frame(centre: Point3, into: Vec3, out_of: Vec3) -> Transform {
    let mut n = (into.normalize() - out_of.normalize()).normalize();
    if into.dot(n) > 0.0 {
        n = -n; // the light has to arrive on the front of it
    }
    let up = if n.z.abs() > 0.99 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(0.0, 0.0, 1.0) };
    let right = up.cross(n).normalize();
    frame(centre, right, n.cross(right), n)
}

/// A rigid placement from three axes and an origin, columns as given.
pub fn frame(o: Point3, x: Vec3, y: Vec3, z: Vec3) -> Transform {
    Transform {
        matrix: tang::Mat4::new(
            x.x, y.x, z.x, o.x, //
            x.y, y.y, z.y, o.y, //
            x.z, y.z, z.z, o.z, //
            0.0, 0.0, 0.0, 1.0,
        ),
    }
}

// ---- the glass ------------------------------------------------------------

/// The lens, as triangles with exact spherical normals.
///
/// Two caps of a sphere of radius `r`, their centres `2a` apart, meeting at
/// the rim in the plane `z = 0`: local +z is the optical axis, the origin is
/// the centre of the glass.
///
/// It is a mesh and not a boolean because the boolean is two spheres two and
/// a half **metres** across intersecting in a disc a hundred and ten
/// millimetres wide — a ratio no CAD kernel should be asked to trim, for a
/// surface whose analytic normal is one line of trigonometry. The same trade
/// `sims/rune/render.rs` makes for the being's capsule, and for the same
/// reason: the shape is exact, only the silhouette is tessellated.
pub fn lens_mesh(segments: usize, rings: usize) -> TriMesh {
    let (r, a, _) = lens_numbers();
    let h = 0.5 * LENS_D;
    let theta_max = (h / r).asin();
    let segments = segments.max(8);
    let rings = rings.max(2);

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    // Two caps, each a fan of rings from its pole out to the shared rim. The
    // top cap belongs to the sphere centred a below the origin, and the
    // bottom cap to the one a above it — which is what makes them bulge apart.
    for &up in &[1.0f64, -1.0] {
        for i in 0..=rings {
            let theta = theta_max * i as f64 / rings as f64;
            let (st, ct) = theta.sin_cos();
            for j in 0..segments {
                let phi = std::f64::consts::TAU * j as f64 / segments as f64;
                let (sp, cp) = phi.sin_cos();
                let n = Vec3::new(st * cp, st * sp, up * ct);
                positions.push(Point3::new(r * n.x, r * n.y, r * n.z - up * a));
                normals.push(n);
            }
        }
    }
    let band = (rings + 1) * segments;
    let mut indices: Vec<u32> = Vec::new();
    for (cap, flip) in [(0usize, false), (band, true)] {
        for i in 0..rings {
            let (a0, b0) = ((cap + i * segments) as u32, (cap + (i + 1) * segments) as u32);
            for j in 0..segments as u32 {
                let k = (j + 1) % segments as u32;
                let quad = [[a0 + j, a0 + k, b0 + k], [a0 + j, b0 + k, b0 + j]];
                for t in quad {
                    if flip {
                        indices.extend_from_slice(&[t[0], t[2], t[1]]);
                    } else {
                        indices.extend_from_slice(&t);
                    }
                }
            }
        }
    }
    // Close the knife edge: the two rims are coincident circles, so stitching
    // them costs nothing and leaves a watertight solid rather than two shells.
    let (top, bottom) = ((rings * segments) as u32, (band + rings * segments) as u32);
    for j in 0..segments as u32 {
        let k = (j + 1) % segments as u32;
        indices.extend_from_slice(&[top + j, bottom + j, bottom + k]);
        indices.extend_from_slice(&[top + j, bottom + k, top + k]);
    }
    TriMesh::new(positions, normals, &indices)
}

/// The prism: an equilateral triangle of side `PRISM_SIDE` in the local
/// xy-plane, its apex down −y and its base across +y, centred on its own
/// centroid and extruded `PRISM_LEN` along local z.
///
/// Flat faces, so no normals are supplied and the tracer uses the geometric
/// ones — which for a plane are the exact ones. Eight triangles is the whole
/// solid.
pub fn prism_mesh() -> TriMesh {
    let s = PRISM_SIDE;
    let hz = 0.5 * PRISM_LEN;
    let back = s * (std::f64::consts::FRAC_PI_3).sin(); // √3/2 · s
    // the apex down −y and the base up +y, about the centroid, so the point
    // the picture places is the middle of the glass and not one of its edges
    let mid = back / 3.0;
    let tri = [[0.0, -2.0 * mid], [-0.5 * s, mid], [0.5 * s, mid]];
    let mut positions = Vec::new();
    for &z in &[-hz, hz] {
        for v in tri {
            positions.push(Point3::new(v[0], v[1], z));
        }
    }
    // 0,1,2 is the −z cap and 3,4,5 the +z cap. **Winding matters here and
    // nowhere else in this file.** No normals are supplied — a plane's
    // geometric normal is its exact one — so the winding *is* the normal, and
    // the integrator reads the sign of it to decide whether a ray is entering
    // the glass or leaving it. Wound inside out, a prism refracts as though
    // every ray were already inside it: the photons come off it in
    // directions nothing predicts and the spectrum simply is not there.
    let indices: Vec<u32> = vec![
        0, 1, 2, // −z cap, outward
        3, 5, 4, // +z cap
        0, 4, 1, 0, 3, 4, // the entry face
        1, 5, 2, 1, 4, 5, // the base
        2, 3, 0, 2, 5, 3, // the exit face
    ];
    TriMesh::new(positions, Vec::new(), &indices)
}

// ---- the hardware ----------------------------------------------------------

/// Everything in the kit that is not glass: the lens's brass ring, and the
/// mirror with its back and its handle.
///
/// Each in its own local frame, centred and axis-aligned, so the picture
/// places whole tools by one transform each. Every root is a primitive or a
/// union of primitives — no boolean, so every one of them keeps its analytic
/// BRep.
pub fn hardware(params: &Params) -> anyhow::Result<Built> {
    build(params, |b| {
        let lens_h = b.param("lens_d_mm", LENS_D) * 0.5;
        let ring = b.param("lens_ring_mm", 30.0); // how far the ring stands proud of the glass
        let mirror_r = b.param("mirror_d_mm", MIRROR_D) * 0.5;
        let mirror_t = b.param("mirror_t_mm", MIRROR_T);

        // The lens's ring: a torus straddling the rim, so the knife edge
        // where the two caps meet is inside metal and never seen.
        b.body("lens_ring")
            .material("brass")
            .add(b.torus(lens_h + 0.15 * ring, 0.5 * ring));

        // The stop: two blocks of stone in front of the prism's entry face,
        // leaving a slit between them. It is not decoration and it is not a
        // cheat — it is the reason the spectrum exists.
        //
        // A beam as wide as the prism's own face has to travel
        // `150 mm / tan 3.9° ≈ 2.2 m` before red has walked clear of violet,
        // and what lands short of that is a white patch with coloured edges —
        // which is what a prism on a table actually looks like and is not
        // what anybody means by a spectrum. Narrow the beam to a slit and the
        // same 3.9° separates it in proportion. Every optics bench in the
        // world has this stone on it.
        //
        // **How wide the slit is, is the whole trade.** A band is
        // `slit + throw·Δ` long and `slit/(slit + throw·Δ)` as bright as bare
        // sun — so a narrow slit buys resolution and pays for it in light, at
        // exactly one for one, and there is no third option. Sixty
        // millimetres at a four-metre throw is a 380 mm band at 0.16 of full
        // sun, which against a stone in shade is about twice its surround.
        // Twenty-six, the last pass's number, would be 0.07 and invisible.
        // It is *turned to face the beam*, which is not a nicety either: the
        // light crosses the prism's own frame at δ/2 = 19° to it, so a slit
        // cut square through 90 mm of stone is a tunnel 31 mm off-axis and a
        // 26 mm slit through it passes precisely nothing. Square to the beam,
        // the same stone passes the slit's full width.
        let slit = b.param("prism_slit_mm", PRISM_SLIT);
        let half = 0.5 * prism_deviation().0.to_degrees();
        let (sh, ch) = (-half).to_radians().sin_cos();
        // The stop is the cliff's own rock rather than the door's stone: a
        // brown block on a brown slab in front of a brown door is a prism
        // nobody can find in the frame, and the cove's rock is a cool
        // blue-grey that reads against every one of them.
        let stop = b.body("stop");
        stop.material("rock");
        for side in [-1.0, 1.0] {
            // along the beam, and across it
            let off = side * (0.5 * slit + 78.0);
            let (ux, uy) = (ch, sh);
            let (vx, vy) = (-sh, ch);
            stop.add(
                b.boxed(80.0, 156.0, 1.25 * PRISM_LEN)
                    .rotate_z(-half)
                    .at(-1.05 * PRISM_SIDE * ux + off * vx, -1.05 * PRISM_SIDE * uy + off * vy, 0.0),
            );
            // …and a cap over each end of the slit, so the slit is *shorter*
            // than the prism it feeds. Without these the top and bottom 25 mm
            // of the slit see past the glass entirely and lay a bar of
            // undeviated sun three metres from the spectrum — invisible at a
            // 250 mm throw, which is why the last pass never saw it, and the
            // brightest thing in the frame at four.
            stop.add(
                b.boxed(80.0, slit + 40.0, 90.0)
                    .rotate_z(-half)
                    .at(
                        -1.05 * PRISM_SIDE * ux,
                        -1.05 * PRISM_SIDE * uy,
                        side * (0.47 * PRISM_LEN + 45.0),
                    ),
            );
        }

        // The mirror: a polished face at z = 0 looking along +z, a dark back,
        // and a handle running out along −y from the rim.
        b.body("mirror").material("silver").add(b.cylinder(mirror_r, mirror_t).at(0.0, 0.0, -mirror_t));
        b.body("mirror_back")
            .material("leather")
            .add(b.cylinder(mirror_r + 8.0, 12.0).at(0.0, 0.0, -mirror_t - 11.0));
        b.body("mirror_handle").material("brass").add(
            b.cylinder(18.0, 210.0)
                .rotate_x(90.0)
                .at(0.0, -mirror_r + 10.0, -mirror_t - 4.0)
                .union(b.sphere(26.0).at(0.0, -mirror_r - 190.0, -mirror_t - 4.0)),
        );
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lens is cut for two and a half metres, and the arithmetic that
    /// says so is the arithmetic the mesh is built from.
    #[test]
    fn the_lens_is_cut_for_two_and_a_half_metres() {
        let h = 0.5 * LENS_D;
        let (r, a, d) = lens_numbers();
        assert!((focal(r, h, N_D) - LENS_F).abs() < 0.5, "f came out {}", focal(r, h, N_D));
        // the numbers quoted in the module header, to the digit quoted
        assert!((r - 2584.0).abs() < 2.0, "R = {r}");
        assert!((a - (r * r - h * h).sqrt()).abs() < 1e-9);
        assert!((d - 4.7).abs() < 0.2, "d = {d}");
        // a thin-lens sanity check: R/(2(n−1)) is within a millimetre of it
        assert!((r / (2.0 * (N_D - 1.0)) - LENS_F).abs() < 2.0);
        // and the mesh really is that shape: h across, d thick, closed
        let m = lens_mesh(64, 8);
        let (mut rad, mut top, mut bot) = (0.0f64, f64::MIN, f64::MAX);
        for p in m.positions() {
            rad = rad.max((p.x * p.x + p.y * p.y).sqrt());
            top = top.max(p.z);
            bot = bot.min(p.z);
        }
        assert!((rad - h).abs() < 1e-6, "the rim is at {rad}");
        assert!((top - bot - d).abs() < 1e-6, "the glass is {} thick", top - bot);
        for n in m.normals() {
            assert!((n.norm() - 1.0).abs() < 1e-12);
        }
    }

    /// The patch on a surface short of the focus is the cone, not a point,
    /// and knowing which is the difference between staging a bright spot and
    /// wondering where it went.
    #[test]
    fn the_converging_cone_is_the_size_the_throw_says() {
        assert!((patch_at(0.0) - LENS_D).abs() < 1e-9);
        assert!(patch_at(LENS_F) < 1e-9, "at the focus it is a point");
        // the doorstep's own throw: about a metre and a half, so about ninety
        // millimetres of caustic — legible beside a 240 mm keyhole
        let p = patch_at(1500.0);
        assert!((80.0..100.0).contains(&p), "the doorstep patch is {p} mm");
    }

    /// The prism is flint, it deviates by forty-seven degrees, and it spreads
    /// the band nearly four — two and a half times what the lens's crown
    /// would, which is the only reason a 300 mm rainbow fits on this beach.
    #[test]
    fn the_prism_is_flint_and_splits_two_and_a_half_times_as_wide_as_the_lens() {
        let (n_d, abbe) = prism_glass();
        assert!((n_d - 1.60).abs() < 1e-6, "the prism's glass is n_d = {n_d}");
        assert!((abbe - 33.0).abs() < 1e-6, "and V = {abbe}");
        let (delta, spread) = prism_deviation();
        assert!((delta.to_degrees() - 46.3).abs() < 0.3, "δ = {}°", delta.to_degrees());
        let (blue, red) = (prism_index_at(400.0), prism_index_at(700.0));
        assert!(blue > red, "glass is more bending to the blue: {blue} vs {red}");
        assert!((prism_index_at(587.56) - n_d).abs() < 1e-9, "the d line is n_d");
        assert!((3.4..4.4).contains(&spread.to_degrees()), "the spectrum spreads {}°", spread.to_degrees());
        // and it really is wider than the lens's crown, which is the claim
        let crown = dispersion(index_at(400.0), index_at(700.0));
        assert!(spread > 2.2 * crown, "flint {spread} vs crown {crown}");

        // The staging trade, in the two functions that state it: a band gets
        // longer and dimmer together and there is no third option.
        let short = band_length(PRISM_SLIT, 1000.0, 1.0);
        let long = band_length(PRISM_SLIT, 4000.0, 1.0);
        assert!(long > 2.5 * short, "{short} → {long}");
        assert!(band_brightness(PRISM_SLIT, 4000.0) < band_brightness(PRISM_SLIT, 1000.0));
        assert!((band_length(PRISM_SLIT, 4000.0, 1.0) * band_brightness(PRISM_SLIT, 4000.0) - PRISM_SLIT).abs() < 1e-9);
        // the number `tools.png` is staged on: a 300–450 mm band at a four
        // metre throw, at a sixth of bare sun
        assert!((300.0..460.0).contains(&long), "the band is {long} mm");
        let bright = band_brightness(PRISM_SLIT, 4000.0);
        assert!((0.12..0.22).contains(&bright), "the band is {bright} of bare sun");

        // eight triangles, none degenerate, and every one of them wound so
        // that its normal points *out* of the glass
        let m = prism_mesh();
        assert_eq!(m.triangles().len(), 8);
        for t in m.triangles() {
            let (a, b, c) = (
                m.positions()[t[0] as usize],
                m.positions()[t[1] as usize],
                m.positions()[t[2] as usize],
            );
            let n = (b - a).cross(c - a).normalize();
            // the centroid is the origin, so "outward" is "away from it"
            let mid = (a.coords() + b.coords() + c.coords()) / 3.0;
            assert!(n.dot(mid) > 0.0, "a face of the prism is wound inside out");
        }
    }

    /// Aiming is a one-dimensional search on the cone about the sunbeam, and
    /// it lands where it was asked to.
    #[test]
    fn the_aim_lands_the_spectrum_where_it_was_told_to() {
        let from = Point3::new(0.0, -3000.0, stage::sand_z(-3000.0) + 140.0);
        let axis = stage::sun_ray();
        let (_, out) = aim(from, axis, min_deviation(N_D), 1400.0);
        let hit = land_on_sand(from, out).expect("the beam comes down somewhere");
        assert!(((hit - from).norm() - 1400.0).abs() < 1.0, "it landed {} away", (hit - from).norm());
        // …and on the sand, not through it
        assert!((hit.z - stage::sand_z(hit.y)).abs() < 1e-6);
    }

    /// And staging is the *inverse* of aiming: pick where the light lands,
    /// and the tool's place falls out with no search at all.
    #[test]
    fn a_tool_stands_where_the_mark_it_has_to_make_puts_it() {
        let target = Point3::new(1200.0, -900.0, 700.0);
        let axis = stage::sun_ray();
        let (delta, _) = prism_deviation();
        let (prism, out) = place_for(target, axis, delta, 1.35, 4000.0);
        assert!(((target - prism).norm() - 4000.0).abs() < 1e-9, "the throw is not the throw");
        assert!(((prism + out * 4000.0) - target).norm() < 1e-9, "the beam misses its own target");
        // the exit is on the cone: that is what makes this a *prism's* beam
        // and not a wish
        assert!((axis.normalize().dot(out).acos() - delta).abs() < 1e-9);
        // and a plane through the target catches it where it was told to
        let n = Vec3::new(-axis.x, -axis.y, 0.0).normalize();
        let hit = land_on_plane(prism, out, target, n).expect("it meets the stone");
        assert!((hit - target).norm() < 1e-6, "it landed {:.3} mm off", (hit - target).norm());
    }

    /// The prism's frame really does turn the beam it was built for, and the
    /// mirror's really does reflect it: the two placements are the only
    /// geometry in the kit that could be silently wrong.
    #[test]
    fn the_frames_point_the_tools_at_what_they_were_aimed_at() {
        let o = Point3::new(0.0, 0.0, 0.0);
        let into = stage::sun_ray();
        let out = on_cone(into, min_deviation(N_D), 1.1);
        let t = prism_frame(o, into, out);
        // the refracting edge is local z, and both beams are square to it
        let edge = t.apply_vec(&Vec3::new(0.0, 0.0, 1.0));
        assert!(into.normalize().dot(edge).abs() < 1e-9);
        assert!(out.dot(edge).abs() < 1e-9);
        // local x is the way the light works across the glass, and the two
        // beams lean off it by the same δ/2 — that is what minimum deviation
        // *is*
        let travel = t.apply_vec(&Vec3::new(1.0, 0.0, 0.0));
        let (a, b) = (into.normalize().dot(travel), out.dot(travel));
        assert!((a - b).abs() < 1e-9, "the prism is not symmetric about its own faces");
        assert!(a > 0.9, "the light should be crossing the glass, not grazing it: {a}");
        // and local y runs from the apex into the glass, so the beam is bent
        // *away* from the apex — the sign that decides which two faces the
        // light uses
        let into_glass = t.apply_vec(&Vec3::new(0.0, 1.0, 0.0));
        assert!(into.normalize().dot(into_glass) < 0.0);
        assert!(out.dot(into_glass) > 0.0);
        // the mesh is centred on its own centroid, so `centre` is the glass
        let m = prism_mesh();
        let mean = m.positions().iter().fold(Vec3::new(0.0, 0.0, 0.0), |a, p| a + p.coords())
            / m.positions().len() as f64;
        assert!(mean.norm() < 1e-9, "the prism is not centred: {mean:?}");

        let want = Vec3::new(0.7, 0.2, -0.2).normalize();
        let m = mirror_frame(o, into, want);
        let n = m.apply_vec(&Vec3::new(0.0, 0.0, 1.0));
        let got = into.normalize() - n * (2.0 * into.normalize().dot(n));
        assert!((got - want).norm() < 1e-9, "the mirror sent it to {got:?}");
        assert!(into.dot(n) < 0.0, "the light arrives on the back of the mirror");
    }
}
