//! `World`: columns, and nothing that runs.
//!
//! The first of the three nouns. A world is state as data — phyz's `Model`
//! and `State` plus kosm's own columns (materials, lights, params) — and it
//! has no method that integrates, renders, or scores. [`crate::step`] does
//! the first, [`crate::lens`] the other two.
//!
//! The phyz view is lossless both ways: [`World::from_phyz`] takes a model
//! and a state, [`World::phyz`] hands both back by reference, and
//! [`World::into_phyz`] hands them back by value. Nothing is dropped in
//! between, so a sim that already has a phyz rig can adopt `World` without
//! giving anything up.
//!
//! **A batch is a `Vec<World>`.** [`World::repeat`] makes one. That is the
//! whole batching story: no leading-axis type, no `Batch<World>` wrapper —
//! the doc's "a world with a leading axis" is a slice of worlds, and
//! [`crate::step::step_batch`] walks it with rayon. When the columns
//! themselves grow a leading axis this is the type that changes.
//!
//! ```
//! use kosm::prelude::*;
//! # fn main() -> anyhow::Result<()> {
//! let (model, state) = kosm::world::demo_marble();
//! let world = World::from_phyz(model, state).with_params(vec![Param::new("tilt", 0.05)]);
//! assert_eq!(world.param("tilt"), Some(0.05));
//! let tilted = world.with(&[("tilt", 0.09)]);
//! assert_eq!(tilted.param("tilt"), Some(0.09));
//! assert_eq!(world.param("tilt"), Some(0.05), "`with` does not mutate");
//! assert_eq!(tilted.repeat(4).len(), 4);
//! # Ok(()) }
//! ```

use std::sync::Arc;

use phyz_model::{Model, State};

/// One named knob, with a value. The gradient is taken with respect to these.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Param {
    pub name: String,
    pub value: f64,
}

impl Param {
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self { name: name.into(), value }
    }
}

/// A surface column. kosm's own, not phyz's: phyz has inertias and colliders,
/// and nothing about how a thing looks.
#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub name: String,
    pub albedo: [f64; 3],
    pub roughness: f64,
    pub metallic: f64,
}

impl Default for Material {
    fn default() -> Self {
        Self { name: String::new(), albedo: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0.0 }
    }
}

/// A point light column.
#[derive(Clone, Debug, PartialEq)]
pub struct Light {
    pub name: String,
    pub pos: [f64; 3],
    /// Watts.
    pub power: f64,
}

/// The columns. No methods that run anything.
///
/// `model` is shared (`Arc`) because a rollout clones a world once per step
/// and the model is the big, static half; the state and kosm's columns are
/// the part that actually moves.
#[derive(Clone)]
pub struct World {
    model: Arc<Model>,
    state: State,
    /// Surfaces, indexed however the sim indexes them.
    pub materials: Vec<Material>,
    /// Lights.
    pub lights: Vec<Light>,
    /// Knobs.
    pub params: Vec<Param>,
}

impl World {
    /// A world over an existing phyz rig. The view back is [`World::phyz`].
    pub fn from_phyz(model: Model, state: State) -> Self {
        Self::from_shared(Arc::new(model), state)
    }

    /// The same, when the model is already shared — a rollout's inner loop.
    pub fn from_shared(model: Arc<Model>, state: State) -> Self {
        Self { model, state, materials: Vec::new(), lights: Vec::new(), params: Vec::new() }
    }

    /// The phyz view, by reference. Lossless: this is the same model and the
    /// same state that went in.
    pub fn phyz(&self) -> (&Model, &State) {
        (&self.model, &self.state)
    }

    /// The phyz view, by value.
    pub fn into_phyz(self) -> (Model, State) {
        let model = Arc::try_unwrap(self.model).unwrap_or_else(|shared| (*shared).clone());
        (model, self.state)
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The shared handle, for a caller building a sibling world cheaply.
    pub fn model_arc(&self) -> &Arc<Model> {
        &self.model
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    /// Replace the shared model, keeping every other column. `pub(crate)`:
    /// [`crate::step::PhyzStep`] uses it to carry a retimed model forward,
    /// and a sim that wants different physics builds a new world instead.
    pub(crate) fn set_model(&mut self, model: Arc<Model>) {
        self.model = model;
    }

    pub fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    /// Generalised positions.
    pub fn q(&self) -> &[f64] {
        self.state.q.as_slice()
    }

    /// Generalised velocities.
    pub fn v(&self) -> &[f64] {
        self.state.v.as_slice()
    }

    /// Simulation time, seconds.
    pub fn time(&self) -> f64 {
        self.state.time
    }

    pub fn with_params(mut self, params: Vec<Param>) -> Self {
        self.params = params;
        self
    }

    pub fn with_materials(mut self, materials: Vec<Material>) -> Self {
        self.materials = materials;
        self
    }

    pub fn with_lights(mut self, lights: Vec<Light>) -> Self {
        self.lights = lights;
        self
    }

    /// The value of a knob.
    pub fn param(&self, name: &str) -> Option<f64> {
        self.params.iter().find(|p| p.name == name).map(|p| p.value)
    }

    /// A copy with those knobs set. Unknown names are appended, so `with` is
    /// also how a knob is introduced. Nothing else changes: rebuilding the
    /// geometry from the new values is the sim's job, because only the sim
    /// knows what a knob means.
    pub fn with(&self, updates: &[(&str, f64)]) -> World {
        let mut out = self.clone();
        for (name, value) in updates {
            match out.params.iter_mut().find(|p| p.name == *name) {
                Some(p) => p.value = *value,
                None => out.params.push(Param::new(*name, *value)),
            }
        }
        out
    }

    /// A batch: `n` copies of this world. See the module docs — a batch is a
    /// `Vec<World>`, and [`crate::step::step_batch`] is what walks it.
    pub fn repeat(&self, n: usize) -> Vec<World> {
        vec![self.clone(); n]
    }

    /// Every column, flattened, for [`crate::diff`]. The names are stable:
    /// two worlds of the same shape produce the same names in the same order.
    pub fn columns(&self) -> Vec<(String, Vec<f64>)> {
        let mut out = vec![
            ("q".to_string(), self.q().to_vec()),
            ("v".to_string(), self.v().to_vec()),
            ("ctrl".to_string(), self.state.ctrl.as_slice().to_vec()),
            ("time".to_string(), vec![self.state.time]),
        ];
        if !self.materials.is_empty() {
            let mut albedo = Vec::new();
            let mut rough = Vec::new();
            for m in &self.materials {
                albedo.extend_from_slice(&m.albedo);
                rough.push(m.roughness);
                rough.push(m.metallic);
            }
            out.push(("materials.albedo".into(), albedo));
            out.push(("materials.roughness_metallic".into(), rough));
        }
        if !self.lights.is_empty() {
            let mut pos = Vec::new();
            let mut power = Vec::new();
            for l in &self.lights {
                pos.extend_from_slice(&l.pos);
                power.push(l.power);
            }
            out.push(("lights.pos".into(), pos));
            out.push(("lights.power".into(), power));
        }
        out
    }
}

impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("World")
            .field("nq", &self.model.nq)
            .field("nv", &self.model.nv)
            .field("bodies", &self.model.bodies.len())
            .field("time", &self.state.time)
            .field("params", &self.params.len())
            .finish()
    }
}

/// A two-body rig — a ball above a fixed plate — for the doctests and the
/// crate's own tests. Not a sim: the sims are under `sims/`.
pub fn demo_marble() -> (Model, State) {
    use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
    use phyz_model::{Geometry, ModelBuilder};

    let r = 0.01;
    let m = 0.005;
    let i = 0.4 * m * r * r;
    let ball = SpatialInertia::new(m, Vec3::zeros(), Mat3::from_diagonal(&Vec3::new(i, i, i)));
    let fixed = SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01);
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_free_body("marble", -1, SpatialTransform::identity(), ball)
        .add_fixed_body("plate", -1, SpatialTransform::identity(), fixed)
        .build();
    model.bodies[0].geometry = Some(Geometry::Sphere { radius: r });
    // the plate is a slab the bead can actually land on, so a rollout over
    // this rig has a contact in it and the camera lens has something to see
    model.bodies[1].collisions = vec![phyz_model::GeomInstance {
        name: Some("plate".into()),
        origin: SpatialTransform::identity(),
        geometry: Geometry::Box { half_extents: Vec3::new(0.1, 0.1, 0.005) },
    }];
    model.bodies[1].visuals = model.bodies[1].collisions.clone();
    let mut state = model.default_state();
    // free-joint q is [wx, wy, wz, x, y, z]
    state.q[5] = 0.2;
    (model, state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_phyz_view_is_lossless_both_ways() {
        let (model, state) = demo_marble();
        let (nq, q5) = (model.nq, state.q[5]);
        let world = World::from_phyz(model, state);
        assert_eq!(world.phyz().0.nq, nq);
        let (model, state) = world.into_phyz();
        assert_eq!(model.nq, nq);
        assert_eq!(state.q[5], q5);
    }

    #[test]
    fn with_is_a_copy_and_introduces_unknown_knobs() {
        let (m, s) = demo_marble();
        let a = World::from_phyz(m, s).with_params(vec![Param::new("tilt", 1.0)]);
        let b = a.with(&[("tilt", 2.0), ("roll", 3.0)]);
        assert_eq!(a.param("tilt"), Some(1.0));
        assert_eq!(b.param("tilt"), Some(2.0));
        assert_eq!(b.param("roll"), Some(3.0));
        assert_eq!(a.param("roll"), None);
    }

    #[test]
    fn columns_are_named_and_flat() {
        let (m, s) = demo_marble();
        let w = World::from_phyz(m, s)
            .with_lights(vec![Light { name: "key".into(), pos: [0.0, 0.0, 1.0], power: 12.0 }]);
        let cols = w.columns();
        let names: Vec<&str> = cols.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["q", "v", "ctrl", "time", "lights.pos", "lights.power"]);
        assert_eq!(cols[0].1.len(), w.model().nq);
    }
}
