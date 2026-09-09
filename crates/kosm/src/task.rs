//! `Task`: the seam a level crosses to be trainable and gateable.
//!
//! Modelled on ipse-dojo's `Task`, with its types moved onto the three
//! nouns: the rig is a [`World`], the episode is a [`Trajectory`], and the
//! score is one number over that trajectory rather than a per-step verdict
//! carried in a task-owned scoring state. What is kept is the shape that
//! mattered there — spawn-as-data (the domain-randomisation axis, the
//! curriculum axis and the replay key at once), **one** scoring rule shared
//! by every consumer, a frozen held-out set, and a task that states its own
//! invariants.
//!
//! ```
//! use kosm::prelude::*;
//!
//! struct Drop;
//! impl Task for Drop {
//!     type Spawn = f64; // release height
//!     fn name(&self) -> &str { "drop" }
//!     fn spawn(&self, seed: u64) -> f64 { 0.1 + (seed % 5) as f64 * 0.05 }
//!     fn build(&self, h: &f64) -> World {
//!         let (model, mut state) = kosm::world::demo_marble();
//!         state.q[5] = *h;
//!         World::from_phyz(model, state)
//!     }
//!     fn horizon(&self) -> usize { 20 }
//!     fn score(&self, traj: &Trajectory, _: &f64) -> f64 {
//!         -traj.last().unwrap().q()[5]
//!     }
//!     fn held_out(&self) -> Vec<(String, f64)> {
//!         vec![("low".into(), 0.1), ("high".into(), 0.3)]
//!     }
//!     fn invariants(&self) -> Vec<Invariant> {
//!         vec![Invariant::new("falling scores above hovering", || Ok(()))]
//!     }
//! }
//!
//! assert!(check_invariants(&Drop).is_empty());
//! let spawn = Drop.spawn(2);
//! let traj = Drop.rollout(&Drop.build(&spawn), &Zero);
//! assert!(Drop.score(&traj, &spawn) < 0.0);
//! ```

use crate::step::{PhyzStep, Policy, Trajectory, rollout};
use crate::world::World;

/// A named invariant a task asserts about its own reward.
///
/// The mechanism that makes "a wish cannot be granted wrong twice"
/// enforceable rather than aspirational. A task that registers none is a
/// task whose author has not yet been surprised.
pub struct Invariant {
    pub name: &'static str,
    /// `Ok(())`, or the reason the reward is exploitable.
    pub check: Box<dyn Fn() -> Result<(), String> + Send + Sync>,
}

impl Invariant {
    pub fn new(
        name: &'static str,
        check: impl Fn() -> Result<(), String> + Send + Sync + 'static,
    ) -> Self {
        Self { name, check: Box::new(check) }
    }
}

/// A task: spawn, build, score, held out, invariants.
pub trait Task: Sync {
    /// The domain-randomisation / curriculum / replay axis.
    type Spawn: Clone + Send + Sync;

    /// A stable identity, for the ledger and the gate.
    fn name(&self) -> &str;

    /// Draw a spawn. Deterministic in the seed, or a held-out number is not
    /// reproducible.
    fn spawn(&self, seed: u64) -> Self::Spawn;

    /// The world for a spawn. Deterministic in the spawn, for the same
    /// reason.
    fn build(&self, spawn: &Self::Spawn) -> World;

    /// How many steps an episode is.
    fn horizon(&self) -> usize;

    /// One scoring rule, shared by the trainer, the gate and the renderer.
    /// Higher is better.
    fn score(&self, trajectory: &Trajectory, spawn: &Self::Spawn) -> f64;

    /// The frozen evaluation set. These must not move between runs — they
    /// are the only reason two numbers from different days are comparable.
    fn held_out(&self) -> Vec<(String, Self::Spawn)>;

    /// What the task asserts about its own reward, checked before any
    /// compute is spent. The default is empty, and that is a smell.
    fn invariants(&self) -> Vec<Invariant> {
        Vec::new()
    }

    /// How an episode is run. The default is phyz at the model's own
    /// timestep for [`Task::horizon`] steps; a task with a different plant
    /// (the real robot, say) overrides it.
    fn rollout(&self, world: &World, policy: &dyn Policy) -> Trajectory {
        rollout(world, &PhyzStep::new(world.model().dt), policy, self.horizon())
    }
}

/// Run every invariant a task declares, returning the failures.
///
/// Called before a run starts. A reward that can be gamed should fail here,
/// cheaply, rather than after an overnight run.
pub fn check_invariants<T: Task + ?Sized>(task: &T) -> Vec<(&'static str, String)> {
    task.invariants()
        .into_iter()
        .filter_map(|inv| (inv.check)().err().map(|why| (inv.name, why)))
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::step::Zero;

    pub(crate) struct Drop;

    impl Task for Drop {
        type Spawn = f64;
        fn name(&self) -> &str {
            "drop"
        }
        fn spawn(&self, seed: u64) -> f64 {
            0.1 + (seed % 5) as f64 * 0.05
        }
        fn build(&self, h: &f64) -> World {
            let (model, mut state) = crate::world::demo_marble();
            state.q[5] = *h;
            World::from_phyz(model, state)
        }
        fn horizon(&self) -> usize {
            10
        }
        fn score(&self, traj: &Trajectory, _: &f64) -> f64 {
            -traj.last().unwrap().q()[5]
        }
        fn held_out(&self) -> Vec<(String, f64)> {
            vec![("low".into(), 0.1), ("high".into(), 0.3)]
        }
        fn invariants(&self) -> Vec<Invariant> {
            vec![
                Invariant::new("a spawn is deterministic", || Ok(())),
                Invariant::new("no step dominates", || Err("a landing paid 800".into())),
            ]
        }
    }

    #[test]
    fn a_failing_invariant_is_reported_before_compute() {
        let failures = check_invariants(&Drop);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, "no step dominates");
        assert!(failures[0].1.contains("800"));
    }

    #[test]
    fn a_spawn_is_deterministic_and_so_is_the_world_it_builds() {
        assert_eq!(Drop.spawn(7), Drop.spawn(7));
        let w = Drop.build(&Drop.spawn(7));
        assert_eq!(w.q()[5], Drop.build(&Drop.spawn(7)).q()[5]);
    }

    #[test]
    fn the_default_rollout_runs_the_horizon() {
        let traj = Drop.rollout(&Drop.build(&0.2), &Zero);
        assert_eq!(traj.len(), 11);
        assert!(Drop.score(&traj, &0.2) < -0.19);
    }
}
