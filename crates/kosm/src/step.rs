//! `Step`: `World → World`, pure.
//!
//! The second noun. An implementation owns the integrator's settings — the
//! timestep, the ground plane, the contact material — and nothing that
//! changes step to step; everything that does is in the world or the action.
//!
//! [`PhyzStep`] is the one implementation kosm ships: phyz's contact
//! simulator at a fixed `dt`. The real robot will be another one (see
//! `docs/architecture.md`, "ipse, the first consumer"), which is the whole
//! reason this is a trait and not a function.
//!
//! ```
//! use kosm::prelude::*;
//! # fn main() -> anyhow::Result<()> {
//! let (model, state) = kosm::world::demo_marble();
//! let world = World::from_phyz(model, state);
//! let step = PhyzStep::new(1e-3);
//!
//! // one step
//! let next = step.step(&world, &Action::none());
//! assert!(next.q()[5] < world.q()[5], "the marble falls");
//!
//! // a rollout is a trajectory
//! let traj = rollout(&world, &step, &Zero, 50);
//! assert_eq!(traj.len(), 51, "the initial world is index 0");
//! assert!(traj.last().unwrap().time() > 0.0);
//!
//! // a batch is a Vec<World>
//! let batch = world.repeat(8);
//! let stepped = step_batch(&step, &batch, &[]);
//! assert_eq!(stepped.len(), 8);
//!
//! // and a gradient of a rollout, zeroth order: two more rollouts
//! let score = |w: &World| -rollout(w, &step, &Zero, 100).last().unwrap().q()[5];
//! let h = 1e-4;
//! let mut lo = world.clone(); lo.state_mut().q[5] -= h;
//! let mut hi = world.clone(); hi.state_mut().q[5] += h;
//! let d_score_d_height = (score(&hi) - score(&lo)) / (2.0 * h);
//! assert!(d_score_d_height.is_finite());
//! # Ok(()) }
//! ```

use std::sync::Arc;

use phyz::Simulator;
use phyz_contact::ContactMaterial;
use rayon::prelude::*;

use crate::world::World;

/// What a policy hands a step. Generalised forces or actuator controls,
/// whichever the model has; empty means "no control this step".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Action {
    pub ctrl: Vec<f64>,
}

impl Action {
    /// No control at all.
    pub fn none() -> Self {
        Self::default()
    }

    pub fn new(ctrl: Vec<f64>) -> Self {
        Self { ctrl }
    }

    pub fn is_empty(&self) -> bool {
        self.ctrl.is_empty()
    }
}

/// `World → World`, pure.
pub trait Step: Sync {
    fn step(&self, world: &World, action: &Action) -> World;
}

/// phyz's contact simulator at a fixed timestep.
///
/// The `dt` here is the authority. If a world's model disagrees, the first
/// step retimes a copy of the model and every world downstream carries the
/// retimed one, so the clone is paid once and not per step.
#[derive(Clone, Debug)]
pub struct PhyzStep {
    pub dt: f64,
    /// The ground plane's height. The default is far below any level: in
    /// kosm a level is geometry, not a half-space.
    pub ground_height: f64,
    pub material: ContactMaterial,
}

impl PhyzStep {
    pub fn new(dt: f64) -> Self {
        Self { dt, ground_height: -10.0, material: ContactMaterial::default() }
    }

    pub fn with_ground(mut self, height: f64) -> Self {
        self.ground_height = height;
        self
    }

    pub fn with_material(mut self, material: ContactMaterial) -> Self {
        self.material = material;
        self
    }
}

impl Step for PhyzStep {
    fn step(&self, world: &World, action: &Action) -> World {
        let mut state = world.state().clone();
        if !action.ctrl.is_empty() {
            let n = state.ctrl.len().min(action.ctrl.len());
            state.ctrl.as_mut_slice()[..n].copy_from_slice(&action.ctrl[..n]);
        }
        let model = if (world.model().dt - self.dt).abs() > f64::EPSILON {
            let mut retimed = world.model().clone();
            retimed.dt = self.dt;
            Arc::new(retimed)
        } else {
            Arc::clone(world.model_arc())
        };
        Simulator::new().step_with_contacts(&model, &mut state, self.ground_height, &self.material);
        let mut next = world.clone();
        *next.state_mut() = state;
        // the retimed model, if there was one, travels with the world
        next.set_model(model);
        next
    }
}

/// A world indexed by time.
#[derive(Clone, Debug, Default)]
pub struct Trajectory {
    worlds: Vec<World>,
}

impl Trajectory {
    pub fn new(worlds: Vec<World>) -> Self {
        Self { worlds }
    }

    pub fn len(&self) -> usize {
        self.worlds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.worlds.is_empty()
    }

    /// The world at time index `t`.
    pub fn get(&self, t: usize) -> Option<&World> {
        self.worlds.get(t)
    }

    pub fn first(&self) -> Option<&World> {
        self.worlds.first()
    }

    pub fn last(&self) -> Option<&World> {
        self.worlds.last()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, World> {
        self.worlds.iter()
    }

    pub fn into_worlds(self) -> Vec<World> {
        self.worlds
    }
}

/// What acts. A closure `Fn(&World, usize) -> Action` is one.
pub trait Policy: Sync {
    fn act(&self, world: &World, t: usize) -> Action;
}

impl<F: Fn(&World, usize) -> Action + Sync> Policy for F {
    fn act(&self, world: &World, t: usize) -> Action {
        self(world, t)
    }
}

/// The policy that does nothing. Most of kosm's own levels are passive.
pub struct Zero;

impl Policy for Zero {
    fn act(&self, _: &World, _: usize) -> Action {
        Action::none()
    }
}

/// `n` steps, keeping every world. Index 0 is the world that went in, so a
/// trajectory of `n` steps has `n + 1` entries.
pub fn rollout<S: Step + ?Sized, P: Policy + ?Sized>(world: &World, step: &S, policy: &P, n: usize) -> Trajectory {
    let mut worlds = Vec::with_capacity(n + 1);
    worlds.push(world.clone());
    let mut current = world.clone();
    for t in 0..n {
        let action = policy.act(&current, t);
        current = step.step(&current, &action);
        worlds.push(current.clone());
    }
    Trajectory::new(worlds)
}

/// One step of a batch, in parallel. `actions` is either empty (every world
/// gets [`Action::none`]) or one action per world; anything else is a
/// mismatch and panics, because silently recycling actions is how a batch
/// quietly stops being the thing you meant.
pub fn step_batch<S: Step + ?Sized>(step: &S, worlds: &[World], actions: &[Action]) -> Vec<World> {
    assert!(
        actions.is_empty() || actions.len() == worlds.len(),
        "step_batch: {} worlds but {} actions",
        worlds.len(),
        actions.len()
    );
    let none = Action::none();
    worlds
        .par_iter()
        .enumerate()
        .map(|(i, w)| step.step(w, actions.get(i).unwrap_or(&none)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::demo_marble;

    fn world() -> World {
        let (m, s) = demo_marble();
        World::from_phyz(m, s)
    }

    #[test]
    fn a_step_is_pure() {
        let w = world();
        let before = w.q()[5];
        let s = PhyzStep::new(1e-3);
        let next = s.step(&w, &Action::none());
        assert_eq!(w.q()[5], before, "the input world did not move");
        assert!(next.q()[5] < before);
    }

    #[test]
    fn the_step_owns_the_timestep() {
        let w = world();
        let slow = PhyzStep::new(2e-3).step(&w, &Action::none());
        assert!((slow.time() - 2e-3).abs() < 1e-12);
        assert!((slow.model().dt - 2e-3).abs() < 1e-12);
    }

    #[test]
    fn a_rollout_is_a_trajectory() {
        let traj = rollout(&world(), &PhyzStep::new(1e-3), &Zero, 20);
        assert_eq!(traj.len(), 21);
        assert_eq!(traj.get(0).unwrap().time(), 0.0);
        assert!(traj.last().unwrap().q()[5] < traj.first().unwrap().q()[5]);
        assert!(traj.get(99).is_none());
    }

    #[test]
    fn a_batch_steps_like_the_singles_it_is_made_of() {
        let w = world();
        let s = PhyzStep::new(1e-3);
        let one = s.step(&w, &Action::none());
        let many = step_batch(&s, &w.repeat(4), &[]);
        assert_eq!(many.len(), 4);
        for m in &many {
            assert_eq!(m.q()[5], one.q()[5]);
        }
    }
}
