//! `Medium`: what the body moves *through*.
//!
//! Air is nothing and is written down anyway, because "no medium" and "a
//! medium that does nothing" being the same object is what lets a sim swap
//! one for the other without a branch. [`Water`] is the cove's sea, ported
//! whole from `sims/rune/being.rs::Cove::water`: buoyancy on the submerged
//! volume, quadratic form drag against a **current**, and a share of the
//! body's own upright damping taken out of a wobble.
//!
//! The current is the design and not a detail. Form drag in still water
//! cannot stop a walking body, and anything that resists both ways turns the
//! sea into a hole you drown in slowly — buoyancy takes the bed's grip away
//! faster than it takes the weight away, so a symmetric resistance strong
//! enough to stop you wading in also stops you wading out. A shore break does
//! both jobs with one force: it stalls you on the way out and carries you on
//! the way in.
//!
//! Metres, seconds, kilograms; z up.

use std::f64::consts::PI;

use phyz_math::{GRAVITY, Vec3};
use phyz_model::Geometry;

/// One part of a body, as the medium sees it.
#[derive(Clone, Copy, Debug)]
pub struct Immersed<'a> {
    /// The part's shape, in its own frame.
    pub geometry: &'a Geometry,
    /// Where the shape's centre is, in world metres.
    pub centre: Vec3,
    /// The shape's own +z in world: a capsule's or a cylinder's axis.
    pub axis: Vec3,
    /// How fast that centre is going through the world, m/s.
    pub velocity: Vec3,
}

/// What the medium does to one part.
#[derive(Clone, Copy, Debug, Default)]
pub struct Immersion {
    /// The force at the part's centre, world axes, newtons.
    pub force: Vec3,
    /// How much of the part is inside the medium, `0..=1`. The body scales
    /// its own angular damping by this rather than the medium guessing at a
    /// torque it has no inertia to size.
    pub submerged: f64,
}

/// What the body moves through at a point.
pub trait Medium: Send + Sync {
    fn immerse(&self, part: &Immersed) -> Immersion;

    /// How much of the body's own upright damping the medium adds when a part
    /// is fully inside it, as a fraction of that damping.
    fn spin_damping(&self) -> f64 {
        0.0
    }
}

/// Nothing, written down.
#[derive(Clone, Copy, Debug, Default)]
pub struct Air;

impl Medium for Air {
    fn immerse(&self, _: &Immersed) -> Immersion {
        Immersion::default()
    }
}

/// The sea: a flat waterline, a density, a current, a drag coefficient.
///
/// The waterline is *flat* on purpose. A rendered sea has a swell on it; a
/// body that bobbed with a 30 mm sine wave would be reporting the renderer's
/// authored decoration as a force.
#[derive(Clone, Copy, Debug)]
pub struct Water {
    /// Where the surface is, world z.
    pub surface_z: f64,
    /// kg/m³. Sea water is 1025; fresh is 1000, and the cove is not full of
    /// fresh water.
    pub density: f64,
    /// How fast the water itself is moving, m/s, world axes. The shore break.
    pub current: Vec3,
    /// Form-drag coefficient. A bluff body broadside is about a cylinder,
    /// which is about one.
    pub drag_cd: f64,
    /// How much of the body's upright damping full submersion adds. A quarter
    /// is "a little": a recovery from a shove goes from critically damped to
    /// visibly sluggish without the spring ever losing the argument.
    pub spin_damp: f64,
}

impl Water {
    /// The sea at `surface_z`, with a shoreward current of `surf` m/s along
    /// `+y` and the cove's own coefficients.
    pub fn sea(surface_z: f64, density: f64, surf: f64) -> Self {
        Self { surface_z, density, current: Vec3::new(0.0, surf, 0.0), drag_cd: 1.0, spin_damp: 0.25 }
    }
}

impl Medium for Water {
    fn immerse(&self, part: &Immersed) -> Immersion {
        let (r, l) = match part.geometry {
            Geometry::Sphere { radius } => (*radius, 0.0),
            Geometry::Capsule { radius, length } => (*radius, *length),
            Geometry::Cylinder { radius, height } => (*radius, (height - 2.0 * radius).max(0.0)),
            // A box, a mesh or a plane is not a shape this integral knows; the
            // sphere that contains it is close enough for a costume part and
            // is at least continuous as it wades.
            Geometry::Box { half_extents } => (half_extents.norm(), 0.0),
            _ => return Immersion::default(),
        };
        // A part within ten degrees of upright has the upright wetted profile
        // to a part in seventy: the barrel's span along z, and a cap of radius
        // `r` at each end of it.
        let rise = part.axis.z.abs() * l / 2.0;
        let foot = part.centre.z - rise - r;
        let depth = (self.surface_z - foot).clamp(0.0, 2.0 * (rise + r));
        if depth <= 0.0 {
            return Immersion::default();
        }
        let (volume, area) = wetted(r, 2.0 * rise, depth);
        let whole = capsule_volume(r, l);
        let up = Vec3::z() * (self.density * volume * GRAVITY);
        let through_water = part.velocity - self.current;
        let drag = through_water * (-0.5 * self.density * self.drag_cd * area * through_water.norm());
        Immersion { force: up + drag, submerged: if whole > 0.0 { volume / whole } else { 0.0 } }
    }

    fn spin_damping(&self) -> f64 {
        self.spin_damp
    }
}

/// A capsule's volume: a barrel of length `l` and a sphere of radius `r`.
pub fn capsule_volume(r: f64, l: f64) -> f64 {
    PI * r * r * l + 4.0 / 3.0 * PI * r * r * r
}

/// What is under water, for an upright capsule of radius `r` and barrel length
/// `l` whose foot is `depth` below the surface: the submerged volume, and the
/// submerged area of its side-on silhouette.
///
/// Both are the same integral up the capsule with a different integrand — a
/// disc of the local radius for the volume, the local width for the silhouette
/// — and both are exact rather than sampled, so buoyancy is continuous as the
/// body wades and the drag does not step. Above the barrel the capsule is its
/// own mirror image, so the far half is "everything, less what is still dry".
pub fn wetted(r: f64, l: f64, depth: f64) -> (f64, f64) {
    fn below(r: f64, l: f64, d: f64) -> (f64, f64) {
        if d <= 0.0 {
            (0.0, 0.0)
        } else if d <= r {
            let t = d - r;
            (PI * d * d * (3.0 * r - d) / 3.0, t * (r * r - t * t).sqrt() + r * r * ((t / r).asin() + PI / 2.0))
        } else if d <= r + l {
            (2.0 / 3.0 * PI * r * r * r + PI * r * r * (d - r), PI * r * r / 2.0 + 2.0 * r * (d - r))
        } else {
            let (v, a) = below(r, l, 2.0 * r + l - d);
            (capsule_volume(r, l) - v, PI * r * r + 2.0 * r * l - a)
        }
    }
    let (whole_v, whole_a) = (capsule_volume(r, l), PI * r * r + 2.0 * r * l);
    let (v, a) = below(r, l, depth.min(2.0 * r + l));
    (v.clamp(0.0, whole_v), a.clamp(0.0, whole_a))
}
