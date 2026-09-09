//! The flat interface a trainer wants, and the adapter from a core task.
//!
//! A [`Task`] is the contract: a spawn distribution, a world for each spawn, a
//! horizon, and one score over the whole trajectory. A trainer needs something
//! narrower and denser — a vector in, a vector out, one number of reward *per
//! control step* — because that is what a policy gradient is computed from.
//! [`Env`] is that narrower thing, and [`TaskEnv`] is the adapter, so the task
//! stays the only place the episode is defined.

use kosm::step::{Action, Step};
use kosm::task::Task;
use kosm::world::World;

/// What one control step returned.
pub struct StepOut {
    pub obs: Vec<f64>,
    pub reward: f64,
    /// The episode ended here. `true` means *terminated* — the world reached
    /// an absorbing state and the critic bootstraps zero. An episode that ran
    /// out of horizon is truncated, not terminated; see
    /// [`Env::truncated`].
    pub done: bool,
}

/// A world a trainer can drive: flat observations in, flat actions out.
///
/// Dimensions are runtime values, not consts. That is the whole difference
/// between this and the `OBS_DIM`/`ACT_DIM` version it was lifted from: one
/// binary can train two rigs, and a schema change is a number rather than a
/// recompile and a pile of artifacts that silently read the wrong columns.
pub trait Env {
    fn obs_dim(&self) -> usize;
    fn act_dim(&self) -> usize;

    /// Draw an episode and return the first observation. Deterministic in the
    /// seed, or nothing downstream is reproducible.
    fn reset(&mut self, seed: u64) -> Vec<f64>;

    /// One control step.
    fn step(&mut self, action: &[f64]) -> StepOut;

    /// Extra state the *critic* may see and the actor may not — the
    /// asymmetric-critic trick. Privileged input is legitimate in simulation
    /// precisely because the critic is thrown away at deploy time; the actor
    /// only ever reads [`Env::step`]'s observation.
    ///
    /// Must be the same width on every call, or return `None` always.
    fn privileged(&self) -> Option<Vec<f64>> {
        None
    }

    /// The last episode ended by running out of horizon rather than by
    /// reaching an absorbing state. Read only when `done` is true; the
    /// default `false` makes every `done` a termination, which is right for
    /// an env with no clock.
    fn truncated(&self) -> bool {
        false
    }
}

/// A [`Task`] driven as an [`Env`].
///
/// The task supplies the episode (spawn, world, horizon); the two lenses
/// supply the trainer's dense signal. `control_every` is how many `Step`
/// substeps one action is held for — 20 substeps of a 1 ms plant is 50 Hz
/// control, which is the rate a real bus runs at.
///
/// The reward lens is read on **every substep** and the sum is divided by
/// `control_every`, so a control step's reward stays O(1) and changing the
/// control rate does not silently rescale the return.
pub struct TaskEnv<T, S, O, R> {
    task: T,
    step: S,
    observe: O,
    reward: R,
    privileged: Option<Box<dyn Fn(&World) -> Vec<f64> + Send + Sync>>,
    control_every: usize,
    act_dim: usize,
    obs_dim: usize,
    world: Option<World>,
    substep: usize,
    truncated: bool,
}

impl<T, S, O, R> TaskEnv<T, S, O, R>
where
    T: Task,
    S: Step,
    O: Fn(&World) -> Vec<f64>,
    R: Fn(&World) -> f64,
{
    /// `act_dim` is how many leading `ctrl` entries the policy writes; the
    /// observation width is measured once, off the task's seed-0 spawn.
    pub fn new(task: T, step: S, observe: O, reward: R, act_dim: usize, control_every: usize) -> Self {
        assert!(control_every >= 1, "control_every must be at least 1");
        let obs_dim = observe(&task.build(&task.spawn(0))).len();
        Self {
            task,
            step,
            observe,
            reward,
            privileged: None,
            control_every,
            act_dim,
            obs_dim,
            world: None,
            substep: 0,
            truncated: false,
        }
    }

    /// Give the critic more than the actor sees. See [`Env::privileged`].
    pub fn with_privileged(
        mut self,
        lens: impl Fn(&World) -> Vec<f64> + Send + Sync + 'static,
    ) -> Self {
        self.privileged = Some(Box::new(lens));
        self
    }

    /// The world as it stands, for a lens the trainer does not know about —
    /// a camera, a snapshot, the task's own score at the end of an episode.
    pub fn world(&self) -> Option<&World> {
        self.world.as_ref()
    }

    pub fn task(&self) -> &T {
        &self.task
    }
}

impl<T, S, O, R> Env for TaskEnv<T, S, O, R>
where
    T: Task,
    S: Step,
    O: Fn(&World) -> Vec<f64>,
    R: Fn(&World) -> f64,
{
    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    fn act_dim(&self) -> usize {
        self.act_dim
    }

    fn reset(&mut self, seed: u64) -> Vec<f64> {
        let world = self.task.build(&self.task.spawn(seed));
        let obs = (self.observe)(&world);
        self.world = Some(world);
        self.substep = 0;
        self.truncated = false;
        obs
    }

    fn step(&mut self, action: &[f64]) -> StepOut {
        let mut world = self.world.take().expect("TaskEnv::step before reset");
        let act = Action::new(action.to_vec());
        let mut reward = 0.0;
        for _ in 0..self.control_every {
            if self.substep >= self.task.horizon() {
                break;
            }
            world = self.step.step(&world, &act);
            self.substep += 1;
            reward += (self.reward)(&world);
        }
        let obs = (self.observe)(&world);
        self.world = Some(world);
        // Out of horizon is truncation, not termination: the value at the
        // last state is still worth bootstrapping.
        self.truncated = self.substep >= self.task.horizon();
        StepOut { obs, reward: reward / self.control_every as f64, done: self.truncated }
    }

    fn privileged(&self) -> Option<Vec<f64>> {
        let lens = self.privileged.as_ref()?;
        Some(lens(self.world.as_ref()?))
    }

    fn truncated(&self) -> bool {
        self.truncated
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use kosm::step::PhyzStep;

    /// A one-dimensional world with no physics in it: the state is a
    /// position and a velocity, the action is an acceleration, and the
    /// reward is how close to the target the position got. Small enough that
    /// a PPO test finishes in seconds and honest enough that learning it is
    /// evidence the update works.
    pub(crate) struct Reach {
        pub target: f64,
        x: f64,
        v: f64,
        t: usize,
        pub horizon: usize,
    }

    impl Reach {
        pub(crate) fn new(target: f64, horizon: usize) -> Self {
            Self { target, x: 0.0, v: 0.0, t: 0, horizon }
        }
    }

    impl Env for Reach {
        fn obs_dim(&self) -> usize {
            2
        }
        fn act_dim(&self) -> usize {
            1
        }
        fn reset(&mut self, seed: u64) -> Vec<f64> {
            // A spread of starts, so the policy learns a controller and not
            // one trajectory.
            self.x = ((seed % 11) as f64 - 5.0) * 0.2;
            self.v = 0.0;
            self.t = 0;
            vec![self.x - self.target, self.v]
        }
        fn step(&mut self, action: &[f64]) -> StepOut {
            let a = action[0].clamp(-1.0, 1.0);
            self.v = (self.v + 0.1 * a) * 0.95;
            self.x += 0.1 * self.v;
            self.t += 1;
            let err = (self.x - self.target).abs();
            StepOut {
                obs: vec![self.x - self.target, self.v],
                reward: -err,
                done: self.t >= self.horizon,
            }
        }
        fn truncated(&self) -> bool {
            self.t >= self.horizon
        }
    }

    #[test]
    fn reach_is_a_well_formed_env() {
        let mut e = Reach::new(0.5, 8);
        let obs = e.reset(3);
        assert_eq!(obs.len(), e.obs_dim());
        let mut last = None;
        for _ in 0..8 {
            last = Some(e.step(&[1.0]));
        }
        assert!(last.unwrap().done, "the horizon ends the episode");
    }


    /// A real core [`Task`] driven through [`TaskEnv`]: kosm's demo marble,
    /// dropped from a spawn-dependent height onto a plate, observed as
    /// (height, vertical velocity) and paid for staying high.
    ///
    /// This is the seam the crate exists for, so it is tested against the
    /// actual `Task` trait rather than a mock of it.
    struct Drop;

    impl Task for Drop {
        type Spawn = f64;
        fn name(&self) -> &str {
            "drop"
        }
        fn spawn(&self, seed: u64) -> f64 {
            0.15 + (seed % 5) as f64 * 0.02
        }
        fn build(&self, h: &f64) -> World {
            let (model, mut state) = kosm::world::demo_marble();
            let mut world = World::from_phyz(model, state.clone());
            state.q[5] = *h;
            *world.state_mut() = state;
            world
        }
        fn horizon(&self) -> usize {
            40
        }
        fn score(&self, traj: &kosm::step::Trajectory, _: &f64) -> f64 {
            traj.last().map_or(0.0, |w| w.q()[5])
        }
        fn held_out(&self) -> Vec<(String, f64)> {
            vec![("high".into(), 0.2)]
        }
    }

    #[test]
    fn a_task_becomes_an_env() {
        let mut env = TaskEnv::new(
            Drop,
            PhyzStep::new(1e-3),
            |w: &World| vec![w.q()[5], w.state().v[5]],
            |w: &World| w.q()[5],
            0,
            10,
        );
        assert_eq!(env.obs_dim(), 2);
        assert_eq!(env.act_dim(), 0);

        let obs = env.reset(0);
        assert!((obs[0] - 0.15).abs() < 1e-12, "spawn 0 starts at 0.15, got {}", obs[0]);

        // The horizon is 40 substeps and one control step is 10 of them, so
        // the fourth control step is the last.
        let mut done_at = None;
        for k in 1..=8 {
            let out = env.step(&[]);
            assert!(out.reward.is_finite());
            if out.done && done_at.is_none() {
                done_at = Some(k);
                break;
            }
        }
        assert_eq!(done_at, Some(4), "the task's horizon ends the episode");
        assert!(env.truncated(), "running out of horizon is truncation");

        // Gravity is on: the marble is lower than it spawned.
        let end = env.world().unwrap().q()[5];
        assert!(end < 0.15, "the marble did not fall: {end}");
    }

    #[test]
    fn a_privileged_lens_widens_the_critic_only() {
        let mut env = TaskEnv::new(
            Drop,
            PhyzStep::new(1e-3),
            |w: &World| vec![w.q()[5]],
            |w: &World| w.q()[5],
            0,
            5,
        )
        .with_privileged(|w: &World| vec![w.state().v[5], w.state().time]);
        assert_eq!(env.privileged(), None, "no world before reset");
        env.reset(1);
        assert_eq!(env.privileged().map(|p| p.len()), Some(2));
        assert_eq!(env.obs_dim(), 1, "the actor still sees one channel");
    }

    #[test]
    fn reset_is_deterministic_in_the_seed() {
        let mut a = Reach::new(0.5, 4);
        let mut b = Reach::new(0.5, 4);
        assert_eq!(a.reset(9), b.reset(9));
    }
}
