//! The creature, as CAD.
//!
//! Millimetres and degrees, vcad's units, z up, and the origin is the patch
//! of sand it is standing on. Every part is a named body so that the render
//! can give each one its own material and so that a later articulation pass
//! has something to hang a joint on; the pivots it will want are written down
//! beside the parts that would turn about them.
//!
//! Nothing here is a mesh and nothing here is a magic number: the whole
//! animal is `sphere`, `cylinder`, `cone` and `torus` through `intersection`,
//! `difference` and `circular_pattern`, driven by the [`Creature`] table. The
//! bell in particular is a *lens*, and the arithmetic that makes it one is in
//! [`Creature::focal_length`] rather than in a comment over a hard-coded
//! radius, so that changing the bell changes the optics and the docs at once.

use kosm::build::{Builder, Shape};

/// How many capsules a limb is made of.
///
/// Six rather than four: the joints between them are visible as steps where
/// the taper changes, and six short steps read as a smooth taper where four
/// long ones read as a machined rod.
const LIMB_SEGMENTS: u32 = 6;

/// Every dimension of the animal, in millimetres unless it says otherwise.
///
/// The defaults are one creature about 1.1 m tall; the type exists so that a
/// second one is a struct literal and not a fork of this file.
#[derive(Debug, Clone, Copy)]
pub struct Creature {
    /// Crown of the bell above the sand.
    pub height: f64,

    // ─── the bell, which is a lens ───────────────────────────────────────
    /// Radius of curvature of the bell's upper (outer) surface.
    pub bell_r_outer: f64,
    /// Radius of curvature of the bell's lower (inner) surface — the shallow
    /// dish that sits over the core. Larger than `bell_r_outer`, which is
    /// what makes the meniscus *converge* rather than diverge.
    pub bell_r_inner: f64,
    /// Thickness of the bell on its axis, crown to dish.
    pub bell_axis_thickness: f64,
    /// Refractive index of the bell's medium.
    pub bell_ior: f64,

    // ─── the core ────────────────────────────────────────────────────────
    /// Radius of the sphere that gives the core its round underside.
    pub core_r: f64,
    /// Radius of the second sphere, which caps the core with a gentler dome.
    pub core_cap_r: f64,
    /// How far below the core's centre that second sphere sits.
    pub core_cap_drop: f64,
    /// Height of the core's centre above the sand.
    pub core_z: f64,

    // ─── the face ────────────────────────────────────────────────────────
    /// Radius of an eye.
    pub eye_r: f64,
    /// Half the distance between the eyes.
    pub eye_spread: f64,
    /// Height of the eyes above the sand.
    pub eye_z: f64,

    /// Radius of the emissive heart.
    pub heart_r: f64,

    // ─── the limbs ───────────────────────────────────────────────────────
    /// How many trailing limbs.
    pub limbs: u32,
    /// Radius of a limb at the shoulder; it tapers to a quarter of this.
    pub limb_r: f64,

    // ─── the gills ───────────────────────────────────────────────────────
    /// How many gill filaments in the crown.
    pub gills: u32,
    /// Length of one filament.
    pub gill_len: f64,
    /// Radius of a filament at its root.
    pub gill_r: f64,
}

impl Default for Creature {
    fn default() -> Self {
        Self {
            height: 1100.0,
            // 255 outer against 430 inner: see `focal_length`. The pair is
            // chosen together and neither means anything alone.
            bell_r_outer: 255.0,
            bell_r_inner: 430.0,
            bell_axis_thickness: 110.0,
            bell_ior: 1.38,
            // Deliberately narrow against the bell's 492 mm span. The bell is
            // the animal's silhouette and the core is the thing inside it; a
            // core as wide as the bell reads as a snowman, which is what the
            // first pass at these numbers looked like.
            core_r: 108.0,
            core_cap_r: 150.0,
            core_cap_drop: 68.0,
            core_z: 818.0,
            eye_r: 40.0,
            eye_spread: 56.0,
            eye_z: 850.0,
            heart_r: 34.0,
            limbs: 8,
            limb_r: 34.0,
            gills: 20,
            gill_len: 130.0,
            gill_r: 19.0,
        }
    }
}

impl Creature {
    /// Centre of the bell's outer sphere, on the axis.
    ///
    /// The crown of that sphere is the top of the animal, so the centre sits
    /// one outer radius below [`Self::height`].
    pub fn bell_centre_z(&self) -> f64 {
        self.height - self.bell_r_outer
    }

    /// Centre of the sphere that hollows the bell's underside, on the axis.
    ///
    /// Its crown is one axis thickness below the bell's, so everything under
    /// that is scooped out and the bell is left as a dish over the core.
    pub fn bell_dish_centre_z(&self) -> f64 {
        self.bell_centre_z() + self.bell_r_outer - self.bell_axis_thickness - self.bell_r_inner
    }

    /// Crown of the dish that hollows the bell — the top of the cavity under
    /// the dome, and where the heart hangs.
    pub fn bell_dish_apex_z(&self) -> f64 {
        self.bell_dish_centre_z() + self.bell_r_inner
    }

    /// Where the two spheres cross, relative to the bell's own centre — the
    /// knife edge where the bell's rim runs out of thickness, and therefore
    /// where the cap has to be cut.
    ///
    /// Solving `R1² − z² = R2² − (z − c)²` for the crossing height `z`, with
    /// `c` the dish centre in the same frame, gives one linear equation:
    /// `z = (R1² − R2² + c²) / (2c)`.
    fn rim_z_local(&self) -> f64 {
        let (r1, r2) = (self.bell_r_outer, self.bell_r_inner);
        let c = self.bell_dish_centre_z() - self.bell_centre_z();
        (r1 * r1 - r2 * r2 + c * c) / (2.0 * c)
    }

    /// Radius of the bell at its rim.
    pub fn bell_rim_radius(&self) -> f64 {
        let z = self.rim_z_local();
        (self.bell_r_outer * self.bell_r_outer - z * z).max(0.0).sqrt()
    }

    /// Height of the bell's rim above the sand.
    pub fn bell_rim_z(&self) -> f64 {
        self.bell_centre_z() + self.rim_z_local()
    }

    /// The focal length of the bell, by the thick-lens maker's equation.
    ///
    /// Light arrives at the crown, crosses a convex surface of radius
    /// `bell_r_outer`, travels `bell_axis_thickness` through the medium and
    /// leaves through a concave surface of radius `bell_r_inner`. Both
    /// centres of curvature lie *beyond* their vertex in the direction the
    /// light is going, so both radii are positive in the usual convention and
    ///
    /// ```text
    ///     1/f = (n − 1) · [ 1/R1 − 1/R2 + (n − 1)·d / (n·R1·R2) ]
    /// ```
    ///
    /// At the defaults — `R1 = 255`, `R2 = 430`, `d = 110`, `n = 1.38` — the
    /// bracket is `0.003922 − 0.002326 + 0.000276 = 0.001872` and
    /// `f ≈ 1406 mm`. That is the whole reason the inner radius is the larger
    /// of the two: a shell of even thickness has `R1 ≈ R2`, the bracket
    /// collapses, and a bell that looks exactly like this one has no focus at
    /// all. Make the dish *tighter* than the crown and the bracket goes
    /// negative and the bell diverges.
    ///
    /// A focus at 1.41 m in front of an animal standing 2 m from the door
    /// means the cone crosses over and is opening again by the time it lands:
    /// from a 246 mm rim it closes to nothing at 1.41 m and reopens to about
    /// 104 mm by 2 m. The keyhole is 120 mm, so the pool very nearly fills
    /// the aperture — which is the picture `door.png` is supposed to show,
    /// and it is soft-edged because a fat little lens like this one has
    /// spherical aberration measured in whole centimetres.
    pub fn focal_length(&self) -> f64 {
        let (r1, r2, d, n) = (
            self.bell_r_outer,
            self.bell_r_inner,
            self.bell_axis_thickness,
            self.bell_ior,
        );
        let bracket = 1.0 / r1 - 1.0 / r2 + (n - 1.0) * d / (n * r1 * r2);
        1.0 / ((n - 1.0) * bracket)
    }

    /// Radius of the bell's beam where it lands `distance` in front of the
    /// bell, by similar triangles about the focus.
    ///
    /// Before the focus the cone is closing and after it is opening; the
    /// absolute value is the same arithmetic either side.
    pub fn beam_radius_at(&self, distance: f64) -> f64 {
        let f = self.focal_length();
        self.bell_rim_radius() * (1.0 - distance / f).abs()
    }
}

/// Build the whole animal into `b`, at the origin, facing −y.
///
/// Returns nothing: the bodies are in the document, and the caller finds them
/// by the names below. Those names are also the material names — see
/// `super::look`.
pub fn assemble(b: &Builder, c: &Creature) {
    bell(b, c);
    core(b, c);
    face(b, c);
    stalk(b, c);
    limbs(b, c);
    gills(b, c);
}

/// The bell: a converging meniscus, cut back to its knife edge.
///
/// Joint pivot for a later articulation pass: the bell pulses about its own
/// axis, so the "joint" is a scale along z about `(0, 0, bell_rim_z)` rather
/// than a rotation — a jellyfish swims by squeezing, and the rim is what
/// travels.
fn bell(b: &Builder, c: &Creature) {
    let cz = c.bell_centre_z();
    let dish = b.sphere(c.bell_r_inner).at(0.0, 0.0, c.bell_dish_centre_z());
    // A slab from the rim upwards, wide enough that only its underside cuts.
    let keep = b
        .boxed(4.0 * c.bell_r_outer, 4.0 * c.bell_r_outer, 4.0 * c.bell_r_outer)
        .at(0.0, 0.0, c.bell_rim_z() + 2.0 * c.bell_r_outer);
    b.body("bell").material("bell").add(
        b.sphere(c.bell_r_outer)
            .at(0.0, 0.0, cz)
            .difference(dish)
            .intersection(keep),
    );
}

/// The core: an egg, from two spheres that disagree about where the top is.
///
/// The lower sphere gives the round underside, the upper one caps it with a
/// gentler dome, and the intersection of the two is an egg wider than it is
/// tall — which is what stops the animal reading as a stack of balls.
///
/// Joint pivot: the neck, at `(0, 0, core_z + core_cap_drop)`, where the core
/// would nod under the bell.
fn core(b: &Builder, c: &Creature) {
    let cap = b
        .sphere(c.core_cap_r)
        .at(0.0, 0.0, c.core_z - c.core_cap_drop);
    b.body("core").material("core")
        .add(b.sphere(c.core_r).at(0.0, 0.0, c.core_z).intersection(cap));
}

/// Two eyes and a heart.
///
/// The eyes sit proud of the core on the −y side, which is the side the
/// camera is on in all four stills. The heart is not inside the core — it is
/// in the cavity the bell's dish leaves above it, so that its light has only
/// the bell's translucent medium to cross and the glow reads as coming from
/// under the dome rather than from behind an opaque body.
///
/// Joint pivots: none. Eyes on this animal do not move; it turns its whole
/// bell instead, which is the point of it.
fn face(b: &Builder, c: &Creature) {
    // Far enough out that most of the sphere clears the core's surface.
    let y = -(c.core_r * 0.88);
    let eye = |x: f64| b.sphere(c.eye_r).at(x, y, c.eye_z);
    b.body("eyes").material("eyes").add(eye(-c.eye_spread)).add(eye(c.eye_spread));

    // Inside the cavity the bell's dish leaves, a little under halfway from
    // the rim plane to the dish's crown.
    let z = c.bell_rim_z() + 0.45 * (c.bell_dish_apex_z() - c.bell_rim_z());
    b.body("heart").material("heart").add(b.sphere(c.heart_r).at(0.0, 0.0, z));
}

/// A capsule from `p` to `q`: a cylinder with a sphere on each end.
///
/// The builder's `rod_z` is a cylinder centred on the origin along z, so the
/// two rotations below are the ones that carry `+z` onto `q − p`: pitch away
/// from the axis first, then yaw about it. `Shape`'s operations wrap, so
/// written left to right they apply in that order.
fn capsule(b: &Builder, p: [f64; 3], q: [f64; 3], r: f64) -> Shape {
    let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let pitch = (d[2] / len).clamp(-1.0, 1.0).acos().to_degrees();
    let yaw = d[1].atan2(d[0]).to_degrees();
    let mid = [
        0.5 * (p[0] + q[0]),
        0.5 * (p[1] + q[1]),
        0.5 * (p[2] + q[2]),
    ];
    b.rod_z(r, len)
        .rotate_y(pitch)
        .rotate_z(yaw)
        .at(mid[0], mid[1], mid[2])
        .union(b.sphere(r).at(p[0], p[1], p[2]))
        .union(b.sphere(r).at(q[0], q[1], q[2]))
}

/// The limbs: one chain of four tapering capsules, patterned about the axis.
///
/// The waypoints are a gentle outward-and-down curve rather than a straight
/// leg, so the silhouette has some sweep in it and the sun has something to
/// come through edge-on. They are written as fractions of the core radius and
/// the standing height so that a taller creature's limbs are still its own.
///
/// Joint pivots: the four waypoints, in order, are the shoulder, elbow, wrist
/// and tip. A later pass rotates each segment about the waypoint that starts
/// it; the chain is already built that way.
fn limbs(b: &Builder, c: &Creature) {
    // The curve, as a function rather than as a list of waypoints: the limb
    // leaves the core almost straight down and swings outward late, which is
    // what a trailing tentacle does and what a tripod leg does not. `t^1.75`
    // is the delay, `1 − t^1.3` the drop.
    let shoulder = c.core_r * 0.78;
    let z0 = c.core_z - c.core_r * 0.62;
    let reach = 310.0;
    let n = LIMB_SEGMENTS;
    let point = |t: f64| {
        [
            shoulder + reach * t.powf(1.75),
            0.0,
            (z0 * (1.0 - t.powf(1.3))).max(c.limb_r * 0.22),
        ]
    };

    let body = b.body("limbs");
    body.material("limbs");
    let mut chain: Option<Shape> = None;
    for i in 0..n {
        let (t0, t1) = (i as f64 / n as f64, (i + 1) as f64 / n as f64);
        // Taper to a quarter of the shoulder radius over the whole chain.
        let r = c.limb_r * (1.0 - 0.75 * t0);
        let seg = capsule(b, point(t0), point(t1), r);
        chain = Some(match chain {
            Some(s) => s.union(seg),
            None => seg,
        });
    }
    if let Some(chain) = chain {
        body.add(chain.circular_pattern([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], c.limbs, 360.0));
    }
}

/// The stalk: what holds the bell over the core.
///
/// Without it the bell floats, and the animal reads as a lampshade on a
/// bulb — which is exactly what the second pass at these images looked like.
/// A short tapered column from the core's crown up into the bell's dish.
///
/// Joint pivot: its base, at `(0, 0, core top)`. This is the joint the bell
/// nods and swivels on, and therefore the one that aims the lens.
fn stalk(b: &Builder, c: &Creature) {
    let base = c.core_z - c.core_cap_drop + c.core_cap_r;
    let top = c.bell_dish_apex_z();
    let body = b.body("stalk");
    body.material("core");
    body.add(
        b.cone(c.core_r * 0.62, c.core_r * 0.40, (top - base).max(1.0))
            .at(0.0, 0.0, base),
    );
}

/// The gills: a crown of tapered filaments where the core meets the bell.
///
/// An axolotl's external gills, which is the half of the animal that is not
/// the jellyfish. A cone with a small top rather than a point, so the tips
/// catch a specular highlight instead of aliasing to nothing.
///
/// Joint pivot: the root of each filament, on the ring of radius
/// `core_r × 0.95` at `gill_z` — they would trail and flick about it.
fn gills(b: &Builder, c: &Creature) {
    // On the core's own equator, which is its widest circle, so every
    // filament starts outside the body instead of inside it.
    let ring = c.core_r * 0.98;
    let z = c.core_z;
    // Out and *past* horizontal at 105° from vertical, so the filaments hang
    // rather than bristle: at 130 mm that puts a tip 234 mm from the axis and
    // 34 mm below its root — just inside the bell's 246 mm rim, which turns
    // them from the whiskers of the second pass into a frill under the dome.
    let filament = b
        .cone(c.gill_r, c.gill_r * 0.34, c.gill_len)
        .rotate_y(105.0)
        .at(ring, 0.0, z);
    b.body("gills").material("gills").add(filament.circular_pattern(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        c.gills,
        360.0,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bell has to actually be a converging lens, or the whole reason it
    /// is a meniscus and not a shell is gone.
    #[test]
    fn the_bell_converges() {
        let c = Creature::default();
        let f = c.focal_length();
        assert!(f > 0.0, "the bell diverges: f = {f}");
        assert!(
            (1250.0..1600.0).contains(&f),
            "f = {f}, not the ~1406 mm the docs claim"
        );
    }

    /// And it stops converging the moment the dish is tighter than the crown,
    /// which is the trap the module note warns about.
    #[test]
    fn a_shell_of_even_thickness_has_no_focus() {
        let c = Creature {
            bell_r_inner: 200.0,
            ..Creature::default()
        };
        assert!(
            c.focal_length() < 0.0,
            "a tighter dish should diverge, not converge"
        );
    }

    /// The rim is where the two spheres cross, so the bell must have real
    /// width there and must sit below the crown.
    #[test]
    fn the_bell_has_a_rim_under_its_crown() {
        let c = Creature::default();
        assert!(c.bell_rim_radius() > 200.0, "{}", c.bell_rim_radius());
        // The bell has to be the silhouette, not the core.
        assert!(c.bell_rim_radius() > 1.8 * c.core_r, "the core is too fat");
        assert!(c.bell_rim_z() < c.height);
        assert!(c.bell_rim_z() > c.core_z);
    }

    /// The beam is a few hundred millimetres across at the door, not a point:
    /// the picture the still is meant to show.
    #[test]
    fn the_beam_is_soft_by_the_time_it_lands() {
        let c = Creature::default();
        let r = c.beam_radius_at(2000.0);
        assert!(
            (60.0..200.0).contains(&r),
            "beam radius at the door is {r} mm"
        );
    }
}
