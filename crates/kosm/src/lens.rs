//! `Lens`: `World → observation`.
//!
//! The third noun. A lens reads a world; it never holds one, and it never
//! changes one. A camera is a lens, a reward is a lens, a probe on a column
//! is a lens. The renderer is not a subsystem a sim addresses — it is the
//! crate [`Camera`] is implemented over.
//!
//! ```
//! use kosm::prelude::*;
//! # fn main() -> anyhow::Result<()> {
//! let (model, state) = kosm::world::demo_marble();
//! let world = World::from_phyz(model, state);
//!
//! // a probe reads one scalar out of one column
//! let height = Probe::q("marble z", 5);
//! assert_eq!(height.see(&world), 0.2);
//!
//! // a reward is a closure adapted into a lens
//! let low = reward("low is good", |w: &World| -w.q()[5]);
//! assert_eq!(low.see(&world), -0.2);
//! # Ok(()) }
//! ```

use phyz_camera::CameraPose;
use phyz_math::{SpatialTransformExt, Vec3};
use phyz_model::Geometry;
use phyz_rigid::forward_kinematics;
use phyz_world::CameraIntrinsics;

use crate::world::World;

/// `World → observation`.
pub trait Lens {
    type Out;
    fn see(&self, world: &World) -> Self::Out;
}

/// Which of a world's state columns a [`Probe`] reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Column {
    Q,
    V,
    Ctrl,
}

/// A named scalar readout of one column entry — a body's height, a joint's
/// rate, one actuator's command. The smallest lens there is.
#[derive(Clone, Debug)]
pub struct Probe {
    pub name: String,
    pub column: Column,
    pub index: usize,
}

impl Probe {
    pub fn new(name: impl Into<String>, column: Column, index: usize) -> Self {
        Self { name: name.into(), column, index }
    }

    pub fn q(name: impl Into<String>, index: usize) -> Self {
        Self::new(name, Column::Q, index)
    }

    pub fn v(name: impl Into<String>, index: usize) -> Self {
        Self::new(name, Column::V, index)
    }
}

impl Lens for Probe {
    type Out = f64;

    fn see(&self, world: &World) -> f64 {
        let col = match self.column {
            Column::Q => world.q(),
            Column::V => world.v(),
            Column::Ctrl => world.state().ctrl.as_slice(),
        };
        col.get(self.index).copied().unwrap_or(f64::NAN)
    }
}

/// A closure, as a lens. `reward(name, f)` is the adapter.
pub struct Reward<F> {
    pub name: String,
    f: F,
}

impl<F> Reward<F> {
    pub fn new(name: impl Into<String>, f: F) -> Self {
        Self { name: name.into(), f }
    }
}

impl<F: Fn(&World) -> f64> Lens for Reward<F> {
    type Out = f64;

    fn see(&self, world: &World) -> f64 {
        (self.f)(world)
    }
}

/// Adapt a closure into a lens.
pub fn reward<F: Fn(&World) -> f64>(name: impl Into<String>, f: F) -> Reward<F> {
    Reward::new(name, f)
}

/// What a [`Camera`] sees: an image, and optionally the depth it was traced
/// at. Depth is metres from the eye; zero means the ray escaped.
pub struct Frame {
    pub image: image::RgbaImage,
    pub depth: Option<Vec<f32>>,
}

impl Frame {
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        self.image.save(path.as_ref())?;
        Ok(())
    }
}

/// A camera, through `kosm-render`.
///
/// The scene is the world's own columns: every body's colliders as analytic
/// geometry at its forward-kinematics pose, every body with a sphere
/// `Geometry` as a glass bead, and a studio rig sized on the bounds. `spp`
/// is the budget — a sim that only needs a thumbnail passes a small one, and
/// `_template` passes a very small one.
pub struct Camera {
    pub pose: CameraPose,
    pub intr: CameraIntrinsics,
    pub spp: u32,
    /// Whether to keep the depth AOV. Cheap: the tracer writes it anyway.
    pub depth: bool,
}

impl Camera {
    pub fn new(pose: CameraPose, intr: CameraIntrinsics, spp: u32) -> Self {
        Self { pose, intr, spp, depth: false }
    }

    /// Look at `target` from `eye`, at a 0.75 rad vertical field of view.
    pub fn look_at(eye: Vec3, target: Vec3, width: u32, height: u32, spp: u32) -> Self {
        Self::new(
            CameraPose::look_at(eye, target, Vec3::z()),
            CameraIntrinsics::from_vfov(width, height, 0.75, 0.01, 100.0),
            spp,
        )
    }

    pub fn with_depth(mut self, depth: bool) -> Self {
        self.depth = depth;
        self
    }
}

impl Lens for Camera {
    type Out = Frame;

    fn see(&self, world: &World) -> Frame {
        use std::sync::Arc;

        let (model, state) = world.phyz();
        let xforms = forward_kinematics(model, state).0;
        let mut objects = Vec::new();
        let mut bounds = kosm_render::Aabb::empty();
        let mut take = |geom: kosm_render::Analytic, bounds: &mut kosm_render::Aabb| {
            for i in 0..kosm_render::Geometry::len(&geom) {
                bounds.include(&kosm_render::Geometry::bounds(&geom, i));
            }
            geom
        };
        for (i, body) in model.bodies.iter().enumerate() {
            if !body.collisions.is_empty() {
                let geom = take(crate::analytic::from_colliders(&xforms[i], &body.collisions), &mut bounds);
                objects.push(kosm_render::Object::new(
                    Arc::new(kosm_render::Bvh::build(geom)),
                    kosm_render::Pbr::plastic([0.42, 0.40, 0.36], 0.55, 0.0),
                ));
            }
            if let Some(Geometry::Sphere { radius }) = body.geometry {
                let c = xforms[i].body_to_world_point(Vec3::zeros());
                let geom = take(crate::analytic::ball(c, radius), &mut bounds);
                objects.push(kosm_render::Object::new(
                    Arc::new(kosm_render::Bvh::build(geom)),
                    kosm_render::Pbr::glass(1.5168, 0.0)
                        .with_sellmeier(kosm_render::spectrum::BK7_SELLMEIER),
                ));
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
        let cam = crate::frame::camera(&self.pose, &self.intr);
        let opts = kosm_render::PathTraceOptions { spp: self.spp, ..Default::default() };
        let film =
            kosm_render::pathtrace::render(&scene, &cam, self.intr.width, self.intr.height, &opts);
        let depth = self.depth.then(|| film.depth.clone());
        let px = film.to_srgb8(0.7, false);
        let image = image::RgbaImage::from_raw(film.width, film.height, px)
            .expect("film is width x height x 4");
        Frame { image, depth }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{World, demo_marble};

    fn world() -> World {
        let (m, s) = demo_marble();
        World::from_phyz(m, s)
    }

    #[test]
    fn a_probe_reads_a_named_column_entry() {
        let w = world();
        assert_eq!(Probe::q("z", 5).see(&w), 0.2);
        assert_eq!(Probe::v("vz", 5).see(&w), 0.0);
        assert!(Probe::q("off the end", 99).see(&w).is_nan());
    }

    #[test]
    fn a_reward_is_a_closure() {
        let w = world();
        let r = reward("height", |w: &World| w.q()[5] * 10.0);
        assert_eq!(r.see(&w), 2.0);
        assert_eq!(r.name, "height");
    }

    #[test]
    fn a_camera_sees_pixels_and_depth() {
        let w = world();
        let cam = Camera::look_at(Vec3::new(0.3, -0.4, 0.35), Vec3::new(0.0, 0.0, 0.2), 32, 24, 1)
            .with_depth(true);
        let frame = cam.see(&w);
        assert_eq!(frame.image.dimensions(), (32, 24));
        assert_eq!(frame.depth.as_ref().unwrap().len(), 32 * 24);
    }
}
