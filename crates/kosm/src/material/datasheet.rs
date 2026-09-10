//! The datasheet: what a substance actually does, measured rather than claimed.
//!
//! [`datasheet`] renders the material ball, drops it, rings it, settles a
//! grain in it, and writes every number plus the constants and their
//! provenance to `datasheet.json`. The numbers are deterministic — a phyz
//! rollout and two closed forms — so [`crate::snapshot::assert_close`] pins
//! them and a changed number means changed code.
//!
//! The picture is a [`Lens`]. [`Ball`] is `World → RgbaImage` over the one
//! canonical world [`ball_world`] builds — a 30 mm sphere resting on a plate
//! — so the substance is the only thing that varies between two balls, which
//! is the entire point of a material ball.

use std::path::{Path, PathBuf};

use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, SpatialTransformExt, Vec3};
use phyz_model::{GeomInstance, Geometry, ModelBuilder};
use phyz_rigid::forward_kinematics;
use serde::{Deserialize, Serialize};

use super::Material;
use crate::build::{Params, build};
use crate::lens::Lens;
use crate::step::{PhyzStep, Zero, rollout};
use crate::world::World;

/// The ball's radius, metres. 30 mm across, which is also the sphere the
/// ring spectrum is taken from — one object, two measurements.
pub const BALL_RADIUS: f64 = 0.015;

/// The timestep every measurement here runs at.
const DT: f64 = 1e-3;

/// The heights the bounce curve is measured from, metres.
const DROPS: [f64; 3] = [0.05, 0.10, 0.20];

/// A quartz grain, for the settling number: 100 µm across, ρ 2650.
const GRAIN_RADIUS: f64 = 50e-6;
const GRAIN_DENSITY: f64 = 2650.0;

// ── the picture ───────────────────────────────────────────────────────────

/// The canonical world a material ball is posed in: a sphere of
/// [`BALL_RADIUS`] resting at the centre of a 300 mm plate.
///
/// Authored in millimetres like every other level, and it is a level — the
/// same [`build`] path, so the ball's geometry and the plate's colliders come
/// out of a vcad document rather than out of the renderer.
pub fn ball_world() -> anyhow::Result<World> {
    let r = BALL_RADIUS * 1e3;
    let built = build(&Params::default(), move |b| {
        b.body("plate").boxed(300.0, 300.0, 20.0).at(0.0, 0.0, -10.0);
        b.body("ball").sphere(r).dynamic(0.05).at(0.0, 0.0, r);
    })?;
    Ok(built.world)
}

/// The material ball: this substance, on a neutral plate, under `studio_rig`.
///
/// A [`Lens`] — it reads the world's bodies and their poses and gives back
/// pixels, holding nothing and changing nothing. Every body that carries a
/// sphere geometry is the substance; everything else is the plate, in a fixed
/// neutral grey, so two datasheets differ in the ball and in nothing else.
pub struct Ball {
    pub material: Material,
    pub width: u32,
    pub height: u32,
    /// Samples per pixel. Sixty-four is the datasheet's, and it is set by the
    /// *glass*: a transmissive ball at sixteen is still carrying enough path
    /// noise to read as frosted rather than as clear, which is the one thing a
    /// glass datasheet must not do. An opaque ball converges long before that.
    /// The crate's own tests pass a much smaller one.
    pub spp: u32,
}

impl Ball {
    pub fn new(material: &Material) -> Self {
        Self { material: material.clone(), width: 320, height: 240, spp: 64 }
    }

    pub fn with_spp(mut self, spp: u32) -> Self {
        self.spp = spp;
        self
    }
}

impl Lens for Ball {
    type Out = image::RgbaImage;

    fn see(&self, world: &World) -> image::RgbaImage {
        use std::sync::Arc;

        let (model, state) = world.phyz();
        let xforms = forward_kinematics(model, state).0;
        let plate = kosm_render::Pbr::plastic([0.30, 0.30, 0.31], 0.55, 0.0);
        let subject = self.material.pbr();
        let mut objects = Vec::new();
        let mut bounds = kosm_render::Aabb::empty();
        let mut take = |geom: kosm_render::Analytic, pbr: kosm_render::Pbr| {
            for i in 0..kosm_render::Geometry::len(&geom) {
                bounds.include(&kosm_render::Geometry::bounds(&geom, i));
            }
            objects.push(kosm_render::Object::new(Arc::new(kosm_render::Bvh::build(geom)), pbr));
        };
        for (i, body) in model.bodies.iter().enumerate() {
            if !body.collisions.is_empty() {
                take(crate::analytic::from_colliders(&xforms[i], &body.collisions), plate);
            }
            if let Some(Geometry::Sphere { radius }) = body.geometry {
                let c = xforms[i].body_to_world_point(Vec3::zeros());
                take(crate::analytic::ball(c, radius), subject);
            }
        }
        let centre = bounds.center();
        let radius = 0.5
            * ((bounds.max.x - bounds.min.x).powi(2)
                + (bounds.max.y - bounds.min.y).powi(2)
                + (bounds.max.z - bounds.min.z).powi(2))
            .sqrt();
        let scene = kosm_render::Scene {
            objects,
            lights: kosm_render::studio_rig(centre, radius.max(1e-3)),
            env: kosm_render::Environment::default(),
            ground: None,
            sun: None,
            splats: None,
        };
        // Close and slightly above, so the ball fills the frame and the plate
        // is a floor rather than a backdrop.
        let eye = Vec3::new(0.075, -0.105, 0.055);
        let target = Vec3::new(0.0, 0.0, BALL_RADIUS);
        let pose = phyz_camera::CameraPose::look_at(eye, target, Vec3::z());
        let intr = phyz_world::CameraIntrinsics::from_vfov(self.width, self.height, 0.6, 0.005, 10.0);
        let cam = crate::frame::camera(&pose, &intr);
        let opts = kosm_render::PathTraceOptions { spp: self.spp, ..Default::default() };
        let film = kosm_render::pathtrace::render(&scene, &cam, self.width, self.height, &opts);
        image::RgbaImage::from_raw(film.width, film.height, film.to_srgb8(0.7, false))
            .expect("film is width x height x 4")
    }
}

// ── the numbers ───────────────────────────────────────────────────────────

/// One drop: how high it was let go, how high it came back.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bounce {
    /// Gap between the ball's underside and the plate at release, metres.
    pub drop_m: f64,
    /// The same gap at the top of the first rebound, metres.
    pub rebound_m: f64,
    /// `rebound / drop`. For a rigid bounce this is `e²`.
    pub ratio: f64,
}

/// Everything [`datasheet`] measured, plus the constants it measured it from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Datasheet {
    pub name: String,
    /// The substance itself: every constant, and every citation.
    pub material: Material,
    /// A sphere of it dropped on a plate of it, at [`DROPS`].
    pub bounce: Vec<Bounce>,
    /// The five lowest breathing modes of a 30 mm sphere of it, Hz. Empty
    /// when the substance has no elastic modulus to ring — a liquid, a flame.
    pub ring_hz: Vec<f64>,
    /// How fast a 100 µm quartz grain settles through it, m/s. `None` for a
    /// solid, which nothing settles through.
    pub settle_mps: Option<f64>,
    /// Where the material ball was written, relative to the run's own output.
    pub ball: Option<PathBuf>,
}

impl Datasheet {
    /// Every number, flattened, for [`crate::snapshot::assert_close`].
    ///
    /// A fixed length whatever the substance is — 20 values, with a zero
    /// standing in for a facet the substance does not have — so the snapshot's
    /// shape does not depend on which material it is of, and a shape change
    /// reads as a shape change rather than as a new material.
    pub fn numbers(&self) -> Vec<f64> {
        let m = &self.material;
        let mut v = vec![
            m.density,
            m.young,
            m.poisson,
            m.loss,
            m.friction,
            m.restitution,
            m.roughness,
        ];
        v.extend_from_slice(&m.colour());
        v.push(m.n_d().unwrap_or(0.0));
        for b in &self.bounce {
            v.push(b.rebound_m);
        }
        v.resize(11 + DROPS.len(), 0.0);
        for k in 0..5 {
            v.push(self.ring_hz.get(k).copied().unwrap_or(0.0));
        }
        v.push(self.settle_mps.unwrap_or(0.0));
        v
    }

    /// The datasheet as the JSON it is written as.
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Measure a substance and write `datasheet.json` (and `ball.png`) into
/// `out_dir`.
///
/// Three measurements, each one a rollout or a closed form and none of them a
/// claim:
///
/// - **bounce** — a sphere of it dropped on a plate of it from 50, 100 and
///   200 mm, through [`PhyzStep`] with the substance's own
///   [`Material::contact`] on both sides. Rebound is the top of the first
///   arc after the first touch.
/// - **ring** — the five lowest Lamb radial modes of a 30 mm sphere, through
///   [`crate::audio::sphere_radial_hz`] on the substance's own
///   [`Material::modal`]. Empty when there is no modulus to ring.
/// - **settle** — Stokes' terminal velocity of a 100 µm quartz grain, for a
///   liquid only. At this size the Reynolds number is below one in water, so
///   Stokes is the right law rather than a convenient one.
pub fn datasheet(m: &Material, out_dir: impl AsRef<Path>) -> anyhow::Result<Datasheet> {
    let dir = out_dir.as_ref();
    std::fs::create_dir_all(dir)?;
    let ball_path = dir.join("ball.png");
    Ball::new(m).see(&ball_world()?).save(&ball_path)?;

    let sheet = Datasheet {
        name: m.name.clone(),
        material: m.clone(),
        bounce: bounce_curve(m),
        ring_hz: ring_spectrum(m),
        settle_mps: settle(m),
        ball: Some(PathBuf::from("ball.png")),
    };
    std::fs::write(dir.join("datasheet.json"), sheet.to_json()?)?;
    Ok(sheet)
}

/// The bounce curve, without touching the disk.
pub fn bounce_curve(m: &Material) -> Vec<Bounce> {
    DROPS.iter().map(|&h| one_drop(m, h)).collect()
}

/// The ring spectrum, without touching the disk.
///
/// A substance with no modulus, or an incompressible one, has no dilatational
/// wave speed and therefore no breathing mode — the closed form divides by
/// `1 − 2ν`. That is a fact about the substance, not a failure, so it comes
/// back empty rather than as a NaN.
pub fn ring_spectrum(m: &Material) -> Vec<f64> {
    if m.young <= 0.0 || m.density <= 0.0 || m.poisson >= 0.4999 {
        return Vec::new();
    }
    crate::audio::sphere_radial_hz(BALL_RADIUS, m.modal(), 5)
        .into_iter()
        .filter(|hz| hz.is_finite())
        .collect()
}

/// Stokes settling of a 100 µm quartz grain, for a liquid.
pub fn settle(m: &Material) -> Option<f64> {
    let mu = m.viscosity?;
    let dr = GRAIN_DENSITY - m.density;
    Some(2.0 / 9.0 * dr * GRAVITY * GRAIN_RADIUS * GRAIN_RADIUS / mu)
}

/// One drop of a sphere of `m` onto a plate of `m`.
///
/// The plate is phyz's ground plane at `z = 0` carrying the substance's own
/// contact material, which is the same thing as a plate of it: phyz combines
/// a material with itself to itself exactly.
fn one_drop(m: &Material, height: f64) -> Bounce {
    let world = drop_world(m, height);
    let step = PhyzStep::new(DT).with_ground(0.0).with_material(m.contact());
    // Long enough to fall from the highest drop and come back: 0.2 s down,
    // and no rebound of a physical restitution takes longer to come back up.
    let traj = rollout(&world, &step, &Zero, 900);
    let z: Vec<f64> = traj.iter().map(|w| w.q()[5]).collect();
    let vz: Vec<f64> = traj.iter().map(|w| w.v()[5]).collect();
    // The impact is where the ball stops falling, not where it reaches some
    // threshold height. phyz's contact is *soft* and carries a margin, so a
    // bouncing sphere never comes within a hair of the plane on the way
    // through — a height test finds the moment it finally settles, tens of
    // bounces later, and reports no rebound at all.
    let Some(impact) = (1..vz.len()).find(|&i| vz[i] > 0.0 && vz[i - 1] <= 0.0) else {
        return Bounce { drop_m: height, rebound_m: 0.0, ratio: 0.0 };
    };
    // The highest point after it. Every later apex is lower than the first —
    // restitution is not a gain — so the maximum over the rest of the rollout
    // *is* the first rebound.
    let peak = z[impact..].iter().copied().fold(BALL_RADIUS, f64::max);
    let rebound = (peak - BALL_RADIUS).max(0.0);
    Bounce { drop_m: height, rebound_m: rebound, ratio: rebound / height }
}

/// A free sphere of `m`, `height` metres above a plane at `z = 0`.
fn drop_world(m: &Material, height: f64) -> World {
    let r = BALL_RADIUS;
    let mass = m.density.max(1e-6) * 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
    let i = 0.4 * mass * r * r;
    let inertia = SpatialInertia::new(mass, Vec3::zeros(), Mat3::from_diagonal(&Vec3::new(i, i, i)));
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(DT)
        .add_free_body("ball", -1, SpatialTransform::identity(), inertia)
        .build();
    let sphere = GeomInstance {
        name: Some("ball".into()),
        origin: SpatialTransform::identity(),
        geometry: Geometry::Sphere { radius: r },
    };
    // Ground detection walks `collisions`, and *only* `collisions`: a body
    // that also carries the same sphere as its centred `geometry` is found
    // twice and lands on two contact rows instead of one, which stiffens the
    // bounce by a factor nobody asked for. One sphere, one row.
    model.bodies[0].collisions = vec![sphere.clone()];
    model.bodies[0].visuals = vec![sphere];
    let mut state = model.default_state();
    state.q[5] = r + height;
    World::from_phyz(model, state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::named;

    #[test]
    fn a_bouncier_substance_bounces_higher() {
        let rubber = bounce_curve(&named("rubber").expect("rubber"));
        let felt = bounce_curve(&named("wool felt").expect("felt"));
        let sand = bounce_curve(&named("dry sand").expect("sand"));
        for (r, f) in rubber.iter().zip(&felt) {
            assert!(r.rebound_m > f.rebound_m, "rubber {r:?} vs felt {f:?}");
        }
        for b in &sand {
            assert!(b.ratio < 0.2, "sand should barely come back: {b:?}");
        }
        // and the curve is monotone in the drop height
        for w in rubber.windows(2) {
            assert!(w[1].rebound_m > w[0].rebound_m, "{w:?}");
        }
        // deterministic: the same substance twice is the same numbers
        assert_eq!(bounce_curve(&named("rubber").expect("rubber")), rubber);
    }

    #[test]
    fn a_bell_rings_where_a_liquid_does_not() {
        let bronze = ring_spectrum(&named("bell bronze").expect("bronze"));
        assert_eq!(bronze.len(), 5);
        assert!(bronze[0] > 0.0 && bronze.windows(2).all(|w| w[1] > w[0]), "{bronze:?}");
        // stiffer per unit mass rings higher, and that is a wave speed, not a
        // modulus: lead crystal is softer than bell bronze and three times
        // lighter, so it rings *above* it.
        let lead = ring_spectrum(&named("lead crystal").expect("lead crystal"));
        assert!(lead[0] > bronze[0], "bronze {} vs lead crystal {}", bronze[0], lead[0]);
        let pewter = ring_spectrum(&named("pewter").expect("pewter"));
        assert!(bronze[0] > pewter[0], "bronze {} vs pewter {}", bronze[0], pewter[0]);
        // Rubber's breathing mode is an order of magnitude below the bell's —
        // not zero, because a nearly incompressible solid still has a fast
        // dilatational wave; what rubber lacks is the Q to hold the note, and
        // that is the loss factor, checked in the library's own test.
        let rubber = ring_spectrum(&named("rubber").expect("rubber"));
        assert!(rubber[0] * 5.0 < bronze[0], "rubber {} vs bronze {}", rubber[0], bronze[0]);
        // an incompressible liquid has no breathing mode, and says so
        assert!(ring_spectrum(&named("water").expect("water")).is_empty());
        assert!(ring_spectrum(&named("candle flame").expect("flame")).is_empty());
    }

    #[test]
    fn a_grain_settles_through_a_liquid_and_not_through_a_solid() {
        let water = settle(&named("water").expect("water")).expect("water is a liquid");
        assert!(water > 1e-3 && water < 0.1, "settling {water} m/s is not a fine grain in water");
        // sea water is denser and more viscous, so the grain is slower
        let sea = settle(&named("sea water").expect("sea water")).expect("a liquid");
        assert!(sea < water);
        assert!(settle(&named("granite").expect("granite")).is_none());
    }

    #[test]
    fn the_numbers_are_a_fixed_shape() {
        let glass = Datasheet {
            name: "N-BK7".into(),
            material: named("N-BK7").expect("N-BK7"),
            bounce: bounce_curve(&named("N-BK7").expect("N-BK7")),
            ring_hz: ring_spectrum(&named("N-BK7").expect("N-BK7")),
            settle_mps: None,
            ball: None,
        };
        let water = Datasheet {
            name: "water".into(),
            material: named("water").expect("water"),
            bounce: Vec::new(),
            ring_hz: Vec::new(),
            settle_mps: settle(&named("water").expect("water")),
            ball: None,
        };
        assert_eq!(glass.numbers().len(), water.numbers().len());
        assert_eq!(glass.numbers().len(), 20);
    }

    #[test]
    fn a_material_ball_is_a_lens_over_one_canonical_world() -> anyhow::Result<()> {
        let world = ball_world()?;
        let ball = Ball { material: named("brass").expect("brass"), width: 24, height: 18, spp: 1 };
        let image = ball.see(&world);
        assert_eq!(image.dimensions(), (24, 18));
        // the lens read the world and changed nothing about it
        assert_eq!(world.q()[5], BALL_RADIUS);
        Ok(())
    }
}
