//! PPO, generic over [`Env`].
//!
//! Lifted from `ipse-sim`'s `rl.rs` and made to forget the robot. What is
//! verbatim: [`Mlp`]'s flat parameter order (a documented fact of the
//! artifact format) and its seeded initialisation, the GAE and
//! clipped-surrogate update in [`process_iteration`], the KL brake, the Huber
//! value clip, and every default in [`PpoConfig`]. What changed:
//!
//! - **Runtime dims.** `OBS_DIM`, `ACT_DIM`, `CMD_DIM` and `CRITIC_DIM` were
//!   consts and the arrays were `[f64; N]`. They are `Vec<f64>` and numbers
//!   carried on the [`Actor`] now, so one binary trains two rigs and a schema
//!   change cannot silently re-point a saved network's columns.
//! - **No command block.** The three `cmd` slots were the K1's heading error,
//!   speed and trick clock. A command is an observation; an [`Env`] that has
//!   one puts it in [`Env::step`]'s observation vector.
//! - **No `LEG_ACT_DIMS` / `UPPER_CLAMP_SCALE` / `UPPER_STD_SCALE`.** Those
//!   split the action vector into legs and arms. The clamp and the initial
//!   std are uniform here; a rig that wants them per-dimension sets
//!   [`Actor::log_std`] itself after [`set_init_std`].
//! - **No `widen_input`, no `OBS_FAMILY=legacy55`.** Those existed to read
//!   artifacts from before an observation-schema change. The artifact format
//!   here is versioned in its header from the first file written, so the
//!   ambiguity they resolved cannot arise.
//! - **No `SkateCondition`.** [`collect`] takes environments.
//! - **`control_every` moved** onto [`kosm_train::env::TaskEnv`](crate::env::TaskEnv),
//!   which is what actually holds an action across substeps; and `noise_rho`
//!   is gone, because its own doc comment in ipse says to leave it at zero —
//!   AR(1) exploration makes the behaviour policy differ from the target
//!   policy and every importance ratio is then wrong.
//! - **The gradient is derived, not written.** ipse's update carried three
//!   hand-derived derivatives — `∂logp/∂μ = z/σ`, `∂logp/∂log σ = z² − 1`,
//!   and the clipped value gradient `2·clamp(err, ±δ)/b` — each pushed back
//!   through `tang_train`'s layer-by-layer backward. Now the whole minibatch
//!   objective is one [`kosm::ir`] graph ([`loss_graph`]): both networks'
//!   forwards, the Gaussian log-prob, the ratio, the clipped surrogate with
//!   its active-branch tolerance, the entropy bonus and the Huber value loss,
//!   and [`TensorGraph::backward`] produces every gradient, `log_std`'s
//!   included. The derivatives above are what it produces; the test
//!   `the_loss_gradient_is_the_one_ipse_derived` holds it to them and to
//!   finite differences.
//! - **f32.** The graph is compiled to C and run natively in `f32`
//!   ([`kosm::ir::Program`]); the weights are kept in `f64` between steps
//!   and in every artifact. The optimiser is [`kosm::ir::Adam`], the same
//!   update `tang_train::ModuleAdam` made (β 0.9 / 0.999, ε 1e-8, no weight
//!   decay, no clipping).
//!
//! The critic is asymmetric: it reads the observation *plus* whatever
//! [`Env::privileged`] returns, which the actor never sees.

use kosm::ir::{Adam, ExprId, Monoid, Program, TensorGraph, TensorId, id, lit, pick};
use rayon::prelude::*;
use tang::Scalar;

use crate::env::Env;
pub use crate::net::Mlp;
use crate::net::{MlpVars, compile_zeroed, mlp};
use crate::search::XorShift;

/// The actor: an MLP mean plus a global learned log-std vector.
pub struct Actor {
    pub net: Mlp,
    /// Log of the exploration std, per action dimension. A parameter of the
    /// update like any weight: [`loss_graph`] differentiates it.
    pub log_std: Vec<f64>,
    /// Applied-action clamp, in the action's own units.
    pub act_clamp: f64,
    obs_dim: usize,
    act_dim: usize,
}

impl Actor {
    pub fn new(obs_dim: usize, act_dim: usize, hidden: usize, seed: u64) -> Self {
        // std ≈ 0.05. Measured, not taste: at std 0.2 the exploration
        // itself shook the K1 down in 0.6 s, every episode, so PPO never
        // saw the standing regime its own mean already occupied.
        // Exploration around a working controller has to be gentle
        // enough to leave it working.
        let log_std = vec![-3.0; act_dim];
        let mut net = Mlp::new(obs_dim, hidden, act_dim, seed);
        net.scale_output(0.01);
        Self { net, log_std, act_clamp: 0.3, obs_dim, act_dim }
    }

    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    pub fn act_dim(&self) -> usize {
        self.act_dim
    }

    /// Clamp a whole action vector.
    pub fn clamp_action(&self, act: &mut [f64]) {
        for v in act.iter_mut() {
            *v = v.clamp(-self.act_clamp, self.act_clamp);
        }
    }

    /// Deterministic action: the mean, clamped. Shorter observations are
    /// zero-padded and longer ones truncated, so a mismatch is a quiet wrong
    /// answer unless the caller checks [`Actor::obs_dim`]; [`train_from`]
    /// asserts the widths agree up front.
    pub fn mean_action(&self, obs: &[f64]) -> Vec<f64> {
        let mut a = self.net.forward(obs);
        a.truncate(self.act_dim);
        self.clamp_action(&mut a);
        a
    }
}

/// PPO hyperparameters. Defaults are ipse's measured ones; the config exists
/// so a sweep is a loop, not an edit.
#[derive(Debug, Clone, Copy)]
pub struct PpoConfig {
    pub gamma: f64,
    pub lam: f64,
    pub clip: f64,
    pub lr: f64,
    pub epochs: usize,
    pub minibatch: usize,
    pub entropy_coef: f64,
    pub hidden: usize,
    /// Episodes collected per iteration, spread across the environments.
    pub episodes_per_iter: usize,
    pub seed: u64,
    /// Initial exploration width. Default 0.05 — the value that let a
    /// standing policy learn without shaking itself down. A trick needs less:
    /// measured on the 63 cm ollie frame, deterministic returns 199 and PPO's
    /// iteration ZERO returns 94, so the noise destroyed more than half the
    /// trick before learning started.
    pub init_std: f64,
    /// Zero-mean Gaussian noise added to every observation entry the policy
    /// reads. The stored sample is the noisy one — the policy trains on what
    /// it saw. Sensors are not clean on hardware.
    pub obs_noise: f64,
    /// Control steps of action delay: a freshly sampled action takes effect
    /// this many control periods later. `1` at 50 Hz is the one-tick bus
    /// latency the deploy stack measures; `0` is the clean sim.
    pub latency_steps: usize,
    /// Stop an iteration's epoch loop once the approximate KL between old and
    /// new policy exceeds this. Measured need: without it a run reached the
    /// full episode clock at iteration 27 and collapsed back to a quarter of
    /// it by 35 — the update was walking off the policy it had just learned.
    pub target_kl: f64,
    /// Applied-action clamp. 0.3 matches ipse's linear policies; 0.45 gives
    /// the net room to own its posture instead of living inside a hand-tuned
    /// stance.
    pub act_clamp: f64,
    /// Hard cap on control steps in one episode, so an [`Env`] that never
    /// reports `done` fails loudly rather than filling memory.
    pub max_steps: usize,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lam: 0.95,
            clip: 0.2,
            lr: 3e-4,
            epochs: 4,
            minibatch: 256,
            entropy_coef: 1e-3,
            hidden: 64,
            episodes_per_iter: 12,
            seed: 7,
            init_std: 0.05,
            obs_noise: 0.0,
            latency_steps: 0,
            target_kl: 0.02,
            act_clamp: 0.3,
            max_steps: 10_000,
        }
    }
}

/// One collected control step.
pub(crate) struct Sample {
    pub(crate) obs: Vec<f64>,
    pub(crate) priv_obs: Vec<f64>,
    pub(crate) act: Vec<f64>,
    pub(crate) logp: f64,
    pub(crate) reward: f64,
    /// Value target scaffolding, filled by GAE.
    pub(crate) adv: f64,
    pub(crate) ret: f64,
}

/// A whole episode, in order.
pub struct Episode {
    pub(crate) samples: Vec<Sample>,
    /// True if it ended in an absorbing state (bootstrap value 0), false if
    /// it ran out of horizon.
    pub(crate) terminated: bool,
    /// Critic input at the final state, for bootstrapping truncated episodes.
    pub(crate) last_critic_in: Vec<f64>,
}

impl Episode {
    /// Control steps recorded before the episode ended.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Sum of the per-sample rewards — the return the update sees.
    pub fn total_reward(&self) -> f64 {
        self.samples.iter().map(|s| s.reward).sum()
    }

    /// Ended in an absorbing state rather than by the clock.
    pub fn terminated(&self) -> bool {
        self.terminated
    }

    /// Per-control-step observation trace — the cross-engine bisection hook.
    pub fn obs_trace(&self) -> Vec<Vec<f64>> {
        self.samples.iter().map(|s| s.obs.clone()).collect()
    }
}


/// Roll one stochastic episode on one environment.
///
/// The mean is [`Mlp::forward`]: one compiled program per worker thread,
/// reused across steps and iterations, with the weights rewritten only when
/// the actor has changed since that thread last ran it.
fn rollout<E: Env + ?Sized>(
    env: &mut E,
    actor: &Actor,
    cfg: &PpoConfig,
    rng: &mut XorShift,
    seed: u64,
    priv_dim: usize,
) -> Episode {
    let act_dim = actor.act_dim;
    let mut obs = env.reset(seed);
    let std: Vec<f64> = actor.log_std.iter().map(|l| l.exp()).collect();
    let mut samples = Vec::new();
    let mut delay_queue: std::collections::VecDeque<Vec<f64>> = std::collections::VecDeque::new();
    let mut applied = vec![0.0; act_dim];
    let terminated;
    let mut steps = 0usize;

    loop {
        if cfg.obs_noise > 0.0 {
            for v in obs.iter_mut() {
                *v += cfg.obs_noise * rng.normal();
            }
        }
        let priv_obs = privileged_or_zeros(env, priv_dim);
        let mean = actor.net.forward(&obs);
        let mut act = vec![0.0; act_dim];
        let mut logp = 0.0;
        for i in 0..act_dim {
            let m = mean[i];
            let z = rng.normal();
            act[i] = m + std[i] * z;
            logp += -0.5 * z * z - std[i].ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
        }
        // Latency: the sampled action reaches the plant `latency_steps`
        // control periods from now; until then the previous command stands,
        // exactly as a delayed bus would have it.
        let mut sampled = act.clone();
        actor.clamp_action(&mut sampled);
        delay_queue.push_back(sampled);
        if delay_queue.len() > cfg.latency_steps {
            if let Some(a) = delay_queue.pop_front() {
                applied = a;
            }
        }

        let out = env.step(&applied);
        samples.push(Sample {
            obs: obs.clone(),
            priv_obs,
            act,
            logp,
            reward: out.reward,
            adv: 0.0,
            ret: 0.0,
        });
        obs = out.obs;
        steps += 1;
        if out.done {
            terminated = !env.truncated();
            break;
        }
        assert!(
            steps < cfg.max_steps,
            "env ran {steps} control steps without reporting done — see PpoConfig::max_steps"
        );
    }

    let mut last_critic_in = obs.clone();
    last_critic_in.resize(actor.obs_dim, 0.0);
    last_critic_in.extend(privileged_or_zeros(env, priv_dim));

    Episode { samples, terminated, last_critic_in }
}

fn privileged_or_zeros<E: Env + ?Sized>(env: &E, priv_dim: usize) -> Vec<f64> {
    if priv_dim == 0 {
        return Vec::new();
    }
    let mut p = env.privileged().unwrap_or_default();
    p.resize(priv_dim, 0.0);
    p
}


/// One iteration's episodes, collected across the environments in parallel.
///
/// Episode seeds are drawn up front on the calling thread so worker
/// scheduling cannot reorder the PRNG stream: the same `cfg.seed` gives the
/// same batch on any machine. Environments are used round-robin, each one
/// running its share sequentially (an `Env` is `&mut self`, so it cannot be
/// two episodes at once) while rayon runs the environments side by side.
/// The actor is shared by reference: its forward is `&self`, and each worker
/// thread runs it through its own compiled program.
pub fn collect<E: Env + Send>(
    envs: &mut [E],
    actor: &Actor,
    cfg: &PpoConfig,
    rng: &mut XorShift,
) -> Vec<Episode> {
    assert!(!envs.is_empty(), "collect needs at least one environment");
    let priv_dim = priv_dim_of(&envs[0]);
    let n_envs = envs.len();
    let seeds: Vec<u64> = (0..cfg.episodes_per_iter).map(|_| rng.next_u64()).collect();

    let mut out: Vec<(usize, Episode)> = envs
        .par_iter_mut()
        .enumerate()
        .flat_map_iter(|(e, env)| {
            let mine: Vec<(usize, u64)> = seeds
                .iter()
                .enumerate()
                .filter(|(i, _)| i % n_envs == e)
                .map(|(i, &s)| (i, s))
                .collect();
            mine.into_iter()
                .map(|(i, s)| {
                    let mut r = XorShift::new(s);
                    (i, rollout(env, actor, cfg, &mut r, s, priv_dim))
                })
                .collect::<Vec<_>>()
                .into_iter()
        })
        .collect();
    // Back into the seed order the stream drew them in, so the update sees a
    // batch that does not depend on how rayon scheduled it.
    out.sort_by_key(|(i, _)| *i);
    out.into_iter().map(|(_, ep)| ep).collect()
}

/// The privileged width an env reports, measured once. `0` means the critic
/// is symmetric.
fn priv_dim_of<E: Env + ?Sized>(env: &E) -> usize {
    env.privileged().map_or(0, |p| p.len())
}

/// One PPO iteration's outcome, for the training log.
#[derive(Debug, Clone, Copy)]
pub struct PpoIter {
    pub mean_return: f64,
    pub mean_len: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
    /// Approximate KL between the old and new policy over the iteration's
    /// minibatches. The epoch loop stops early when this exceeds
    /// [`PpoConfig::target_kl`].
    pub kl: f64,
    /// Optimizer steps actually taken this iteration — `epochs × ceil(n /
    /// minibatch)` unless the KL brake stopped early. Logged because the
    /// brake is the difference between "the policy did not learn" and "the
    /// policy was never allowed to move".
    pub grad_steps: usize,
}

/// Huber delta for the value loss, as a multiple of the batch's own return
/// spread. Past it the loss stays quadratic in the log but the GRADIENT
/// saturates, so one absurd return cannot drag the critic.
const VALUE_HUBER: f64 = 10.0;


// Graph inputs of [`loss_graph`], in declaration order. Per-row tensors are
// `[B, …]` for the fixed minibatch `B`; a ragged last minibatch is padded
// with copies of its first row at weight zero, so one compiled program
// serves every minibatch of every iteration.
const IN_XA: usize = 0; //       [B, obs]      actor input
const IN_XC: usize = 1; //       [B, critic]   critic input (obs ++ privileged)
const IN_ACT: usize = 2; //      [B, act]      the sampled action
const IN_LOGP: usize = 3; //     [B]           its log-prob under the behaviour policy
const IN_ADV: usize = 4; //      [B]           normalised GAE advantage
const IN_RET: usize = 5; //      [B]           value target
const IN_WT: usize = 6; //       [B]           1/b on real rows, 0 on padding
const IN_CLIP: usize = 7; //     [1]           surrogate clip ε
const IN_ENT: usize = 8; //      [1]           entropy coefficient
const IN_VDELTA: usize = 9; //   [1]           Huber δ, in return units
const IN_ACTOR: usize = 10; //   6 tensors     actor mean MLP, flat order
const IN_LOG_STD: usize = 16; // [act]
const IN_CRITIC: usize = 17; //  6 tensors     critic MLP, flat order

// Outputs of [`loss_graph`].
#[cfg_attr(not(test), allow(dead_code))]
const OUT_LOSS: usize = 0; //  [1] the objective `backward` differentiates
const OUT_PLOSS: usize = 1; // [1] the clipped surrogate alone (logged)
const OUT_VLOSS: usize = 2; // [1] plain MSE of the value (logged; the gradient is Huber's)
const OUT_K3: usize = 3; //    [1] Schulman's k3, `mean(r − 1 − ln r)` — the KL brake
const OUT_GRAD: usize = 4; //  13 tensors: actor ×6, log_std, critic ×6

/// The whole PPO minibatch objective as one graph, and its gradient.
///
/// For each row (weight `w`, `Σw = 1`):
///
/// - `μ = actor(x_a)`, `z = (a − μ)·e^{−log σ}`,
///   `logp = Σ_d −½z² − log σ_d − ½ln 2π` — the same expression the rollout
///   evaluated when it sampled `a`;
/// - `r = e^{logp − logp_old}`, `c = clamp(r, 1 ± ε)`, and the surrogate
///   `−min(r·A, c·A)` where the unclipped branch counts as the minimum
///   within `1e-12` — ipse's tie rule, so a ratio inside the clip interval
///   always carries its gradient and one outside it carries none;
/// - `e = critic(x_c) − ret` and the Huber loss `e²` inside `|e| ≤ δ`,
///   `2δ|e| − δ²` outside, whose gradient is ipse's `2·clamp(e, ±δ)`.
///
/// The objective is `Σ w·surrogate − c_H·Σ_d log σ_d + Σ w·huber`: the
/// entropy of a diagonal Gaussian is `Σ log σ` plus a constant. The actor
/// and critic terms share no parameter, so one `backward` gives each
/// network the gradient of its own loss.
pub(crate) fn loss_graph(
    b: usize,
    actor: (usize, usize, usize),
    critic: (usize, usize, usize),
) -> (TensorGraph, Vec<TensorId>) {
    let (od, _, ad) = actor;
    let cd = critic.0;
    assert_eq!(critic.2, 1, "the critic has one output");
    let mut g = TensorGraph::new();
    let xa = g.input(&[b, od]);
    let xc = g.input(&[b, cd]);
    let act = g.input(&[b, ad]);
    let logp_old = g.input(&[b]);
    let adv = g.input(&[b]);
    let ret = g.input(&[b]);
    let wt = g.input(&[b]);
    let clip = g.input(&[1usize]);
    let ent_coef = g.input(&[1usize]);
    let vdelta = g.input(&[1usize]);
    let pa = MlpVars::inputs(&mut g, actor);
    let log_std = g.input(&[ad]);
    let pc = MlpVars::inputs(&mut g, critic);

    let mean = mlp(&mut g, xa, pa, b, actor);
    let half_ln_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    let logp = g
        .comp(
            &[b],
            &[ad],
            vec![pick(act, 2, &[Some(0), Some(1)]), pick(mean, 2, &[Some(0), Some(1)]), pick(log_std, 2, &[Some(1)])],
            Some(Monoid::sum()),
            |v| {
                let z = (v[0] - v[1]) * (-v[2]).exp();
                vec![lit(-0.5) * z * z - v[2] - lit(half_ln_2pi)]
            },
        )
        .expect("logp");

    // A row tensor `[B]` or a scalar `[1]`, read in the `[1] ⊕ [B]` space of
    // a batch sum: rows ride the reduce coordinate, scalars the output one.
    let row = |t: TensorId| pick(t, 2, &[Some(1)]);
    let one = |t: TensorId| pick(t, 2, &[Some(0)]);

    let ploss = g
        .comp(&[1usize], &[b], vec![row(logp), row(logp_old), row(adv), row(wt), one(clip)], Some(Monoid::sum()), |v| {
            let r = (v[0] - v[1]).exp();
            let c = r.clamp(lit(1.0) - v[4], lit(1.0) + v[4]);
            let (u, k) = (r * v[2], c * v[2]);
            // `select(p, a, b)` is `a` where `p > 0`: the clipped branch
            // only once it is below the unclipped one by more than 1e-12.
            let m = <ExprId as Scalar>::select(u - k - lit(1e-12), k, u);
            vec![-(v[3] * m)]
        })
        .expect("surrogate");
    let k3 = g
        .comp(&[1usize], &[b], vec![row(logp), row(logp_old), row(wt)], Some(Monoid::sum()), |v| {
            let lr = v[0] - v[1];
            vec![v[2] * (lr.exp() - lit(1.0) - lr)]
        })
        .expect("k3");
    let entropy = g
        .comp(&[1usize], &[ad], vec![pick(log_std, 2, &[Some(1)])], Some(Monoid::sum()), |v| vec![v[0]])
        .expect("entropy");

    let value = mlp(&mut g, xc, pc, b, critic); // [B, 1]
    // `value[row, 0]`: its unit column rides the output coordinate.
    let val = pick(value, 2, &[Some(1), Some(0)]);
    let vloss = g
        .comp(&[1usize], &[b], vec![val.clone(), row(ret), row(wt)], Some(Monoid::sum()), |v| {
            let e = v[0] - v[1];
            vec![v[2] * e * e]
        })
        .expect("mse");
    let huber = g
        .comp(&[1usize], &[b], vec![val, row(ret), row(wt), one(vdelta)], Some(Monoid::sum()), |v| {
            let e = v[0] - v[1];
            let (a, d) = (e.abs(), v[3]);
            let h = <ExprId as Scalar>::select(a - d, lit(2.0) * d * a - d * d, e * e);
            vec![v[2] * h]
        })
        .expect("huber");

    let loss = g
        .comp(
            &[1usize],
            &[] as &[usize],
            vec![id(ploss, 1, 1), id(entropy, 1, 1), id(huber, 1, 1), id(ent_coef, 1, 1)],
            None,
            |v| vec![v[0] - v[3] * v[1] + v[2]],
        )
        .expect("loss");

    let mut wrt: Vec<TensorId> = pa.0.to_vec();
    wrt.push(log_std);
    wrt.extend(pc.0);
    let grads = g.backward(loss, &wrt).expect("backward");
    let mut outs = vec![loss, ploss, vloss, k3];
    outs.extend(grads);
    kosm::ir::optimize(&g, &outs)
}

/// The update's compiled loss and its two optimisers — the state
/// [`process_iteration`] carries from one iteration to the next.
///
/// Compiling the loss is the expensive part (a C compile, a fraction of a
/// second); it happens once, in [`PpoUpdate::new`], and every minibatch after
/// that is a write of its rows and the current weights, one run, and two
/// [`Adam`] steps.
pub struct PpoUpdate {
    program: Program,
    batch: usize,
    obs_dim: usize,
    act_dim: usize,
    critic_dim: usize,
    opt_actor: Adam,
    opt_critic: Adam,
}

impl PpoUpdate {
    pub fn new(actor: &Actor, critic: &Mlp, cfg: &PpoConfig) -> Self {
        let batch = cfg.minibatch.max(1);
        let (g, outs) = loss_graph(batch, actor.net.dims, critic.dims);
        let n_actor = actor.net.params().len() + actor.act_dim;
        let n_critic = critic.params().len();
        Self {
            program: compile_zeroed(&g, &outs),
            batch,
            obs_dim: actor.obs_dim,
            act_dim: actor.act_dim,
            critic_dim: critic.dims.0,
            opt_actor: Adam::new(n_actor, cfg.lr),
            opt_critic: Adam::new(n_critic, cfg.lr),
        }
    }

    /// One minibatch: evaluate the loss and its gradient at the current
    /// weights, then step both networks. Returns the pre-step surrogate,
    /// value MSE and k3, each a mean over the real rows.
    fn step(
        &mut self,
        rows: &[&Sample],
        actor: &mut Actor,
        critic: &mut Mlp,
        cfg: &PpoConfig,
        vdelta: f64,
    ) -> (f64, f64, f64) {
        let (b, od, ad, cd) = (self.batch, self.obs_dim, self.act_dim, self.critic_dim);
        let [xa, xc, act, logp, adv, ret, wt] = pack_rows(rows, b, od, ad, cd);
        let p = &mut self.program;
        for (k, v) in [(IN_XA, &xa), (IN_XC, &xc), (IN_ACT, &act), (IN_LOGP, &logp), (IN_ADV, &adv), (IN_RET, &ret), (IN_WT, &wt)] {
            p.write_f64(k, v);
        }
        p.write_f64(IN_CLIP, &[cfg.clip]);
        p.write_f64(IN_ENT, &[cfg.entropy_coef]);
        p.write_f64(IN_VDELTA, &[vdelta]);
        MlpVars::write(p, IN_ACTOR, &actor.net);
        p.write_f64(IN_LOG_STD, &actor.log_std);
        MlpVars::write(p, IN_CRITIC, critic);
        p.run();

        let gather = |p: &Program, from: usize, count: usize| -> Vec<f64> {
            (from..from + count).flat_map(|i| p.read_f64(i)).collect()
        };
        let mut ga = gather(p, OUT_GRAD, 7);
        let gc = gather(p, OUT_GRAD + 7, 6);
        let mut wa = actor.net.to_flat();
        let na = wa.len();
        wa.extend_from_slice(&actor.log_std);
        ga.truncate(wa.len());
        self.opt_actor.step(&mut wa, &ga);
        actor.net.set_flat(&wa[..na]);
        actor.log_std.copy_from_slice(&wa[na..]);
        let mut wc = critic.to_flat();
        self.opt_critic.step(&mut wc, &gc);
        critic.set_flat(&wc);

        let scalar = |i: usize| p.read_f64(i)[0];
        (scalar(OUT_PLOSS), scalar(OUT_VLOSS), scalar(OUT_K3))
    }
}

/// A minibatch as [`loss_graph`]'s seven per-row inputs, `IN_XA..=IN_WT`.
/// Rows past `rows.len()` repeat the first row at weight zero: finite, so
/// `0 · term` is 0, where a zeroed row's ratio could overflow `f32`.
fn pack_rows(rows: &[&Sample], b: usize, od: usize, ad: usize, cd: usize) -> [Vec<f64>; 7] {
    let n = rows.len();
    assert!(n >= 1 && n <= b, "minibatch of {n} rows for a program of {b}");
    let mut xa = vec![0.0; b * od];
    let mut xc = vec![0.0; b * cd];
    let mut act = vec![0.0; b * ad];
    let (mut logp, mut adv, mut ret, mut wt) = (vec![0.0; b], vec![0.0; b], vec![0.0; b], vec![0.0; b]);
    for r in 0..b {
        let s = rows[if r < n { r } else { 0 }];
        let k = s.obs.len().min(od);
        xa[r * od..r * od + k].copy_from_slice(&s.obs[..k]);
        xc[r * cd..r * cd + k].copy_from_slice(&s.obs[..k]);
        let p = s.priv_obs.len().min(cd - od);
        xc[r * cd + od..r * cd + od + p].copy_from_slice(&s.priv_obs[..p]);
        let k = s.act.len().min(ad);
        act[r * ad..r * ad + k].copy_from_slice(&s.act[..k]);
        (logp[r], adv[r], ret[r]) = (s.logp, s.adv, s.ret);
        wt[r] = if r < n { 1.0 / n as f64 } else { 0.0 };
    }
    [xa, xc, act, logp, adv, ret, wt]
}

/// GAE + the clipped update over one iteration's episodes.
pub fn process_iteration(
    episodes: Vec<Episode>,
    actor: &mut Actor,
    critic: &mut Mlp,
    update: &mut PpoUpdate,
    cfg: &PpoConfig,
    rng: &mut XorShift,
) -> PpoIter {
    let obs_dim = actor.obs_dim;
    let critic_dim = critic.dims.0;
    let n_eps = episodes.len().max(1);

    // ── GAE ── (a scan, so plain Rust; the values are one batched forward)
    let mut all: Vec<Sample> = Vec::new();
    let mut total_reward = 0.0;
    let mut total_len = 0usize;
    for mut ep in episodes {
        let n = ep.samples.len();
        total_len += n;
        if n == 0 {
            continue;
        }
        let mut buf = vec![0.0; (n + 1) * critic_dim];
        for (i, s) in ep.samples.iter().enumerate() {
            let row = &mut buf[i * critic_dim..(i + 1) * critic_dim];
            let k = s.obs.len().min(obs_dim);
            row[..k].copy_from_slice(&s.obs[..k]);
            let p = s.priv_obs.len().min(critic_dim - obs_dim);
            row[obs_dim..obs_dim + p].copy_from_slice(&s.priv_obs[..p]);
        }
        let k = ep.last_critic_in.len().min(critic_dim);
        buf[n * critic_dim..n * critic_dim + k].copy_from_slice(&ep.last_critic_in[..k]);
        let v = critic.forward_rows(&buf);

        let last_v = if ep.terminated { 0.0 } else { v[n] };
        let mut gae = 0.0;
        for i in (0..n).rev() {
            let next_v = if i == n - 1 { last_v } else { v[i + 1] };
            let delta = ep.samples[i].reward + cfg.gamma * next_v - v[i];
            gae = delta + cfg.gamma * cfg.lam * gae;
            ep.samples[i].adv = gae;
            ep.samples[i].ret = gae + v[i];
            total_reward += ep.samples[i].reward;
        }
        all.extend(ep.samples);
    }
    let empty = |mean_return: f64| PpoIter {
        mean_return,
        mean_len: 0.0,
        policy_loss: 0.0,
        value_loss: 0.0,
        kl: 0.0,
        grad_steps: 0,
    };
    if all.is_empty() {
        return empty(0.0);
    }

    // A non-finite reward means the plant or the value target has diverged.
    // One ipse GPU run reached NaN at iteration 150 and spent 700 more
    // producing it; the loop reports and stops instead now.
    if let Some(bad) = all.iter().position(|s| !s.reward.is_finite() || !s.logp.is_finite()) {
        eprintln!(
            "non-finite sample {bad} (reward {:.3e}, logp {:.3e}) — stopping",
            all[bad].reward, all[bad].logp
        );
        return empty(f64::NAN);
    }

    // Advantage normalization, batch-wide.
    let mean_adv = all.iter().map(|s| s.adv).sum::<f64>() / all.len() as f64;
    let var_adv =
        all.iter().map(|s| (s.adv - mean_adv).powi(2)).sum::<f64>() / all.len() as f64;
    let std_adv = var_adv.sqrt().max(1e-8);
    for s in &mut all {
        s.adv = (s.adv - mean_adv) / std_adv;
    }

    // ── update ──
    let mut policy_loss_acc = 0.0;
    let mut value_loss_acc = 0.0;
    let mut loss_batches = 0usize;
    let mut brake_acc = 0.0;
    let mut brake_batches = 0usize;
    // Huber delta in return units, from this batch's own spread.
    let mean_ret = all.iter().map(|s| s.ret).sum::<f64>() / all.len() as f64;
    let std_ret =
        (all.iter().map(|s| (s.ret - mean_ret).powi(2)).sum::<f64>() / all.len() as f64).sqrt();
    let vdelta = (VALUE_HUBER * std_ret).max(1e-6);
    let mut stop = false;
    let n_all = all.len();
    for _ in 0..cfg.epochs {
        if stop {
            break;
        }
        // Fisher–Yates over an index vec, from the same PRNG.
        let mut order: Vec<usize> = (0..n_all).collect();
        for i in (1..n_all).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        for chunk in order.chunks(cfg.minibatch) {
            let rows: Vec<&Sample> = chunk.iter().map(|&i| &all[i]).collect();
            let (ploss, vloss, k3) = update.step(&rows, actor, critic, cfg, vdelta);
            policy_loss_acc += ploss;
            value_loss_acc += vloss;
            loss_batches += 1;
            // Early stop: one more minibatch past this and the policy is no
            // longer the one the batch was collected under.
            brake_acc += k3;
            brake_batches += 1;
            if k3 > cfg.target_kl {
                stop = true;
                break;
            }
        }
    }

    PpoIter {
        mean_return: total_reward / n_eps as f64,
        mean_len: total_len as f64 / n_eps as f64,
        policy_loss: policy_loss_acc / loss_batches.max(1) as f64,
        value_loss: value_loss_acc / loss_batches.max(1) as f64,
        kl: brake_acc / brake_batches.max(1) as f64,
        grad_steps: loss_batches,
    }
}

/// Set an actor's exploration width, uniformly across the action vector.
///
/// ipse scaled the upper-body dimensions to a quarter of this, because arm
/// and head noise knocked the rider over faster than the legs learned. That
/// is a fact about *that* rig's action layout, not about PPO, so it belongs
/// to the caller: set [`Actor::log_std`] directly after calling this.
pub fn set_init_std(actor: &mut Actor, std: f64) {
    for v in actor.log_std.iter_mut() {
        *v = std.max(1e-6).ln();
    }
}

/// Train PPO on a set of environments. Returns the actor, the critic, and the
/// per-iteration history. Deterministic per seed: the PRNG stream is drawn on
/// the calling thread and episodes are assigned to environments round-robin.
pub fn train<E: Env + Send>(
    envs: Vec<E>,
    cfg: PpoConfig,
    iterations: usize,
    progress: impl FnMut(usize, &PpoIter, &Actor),
) -> (Actor, Mlp, Vec<PpoIter>) {
    train_from(envs, cfg, iterations, None, progress)
}

/// [`train`] continued from a saved actor — the curriculum handoff. A policy
/// that learned the easy rung starts the next one inside the regime it
/// already reaches instead of two seconds from a fall.
///
/// A warm actor comes off disk with `log_std` at zeros (exploration width
/// 1.0, twenty times what any trainer runs); this calls [`set_init_std`]
/// before collecting, and a caller that hand-rolls a warm start must too.
pub fn train_from<E: Env + Send>(
    mut envs: Vec<E>,
    cfg: PpoConfig,
    iterations: usize,
    warm: Option<Actor>,
    mut progress: impl FnMut(usize, &PpoIter, &Actor),
) -> (Actor, Mlp, Vec<PpoIter>) {
    assert!(!envs.is_empty(), "train needs at least one environment");
    let obs_dim = envs[0].obs_dim();
    let act_dim = envs[0].act_dim();
    // Reset once so an env that only reports its privileged width after a
    // spawn (TaskEnv does) reports it here.
    let _ = envs[0].reset(cfg.seed);
    let priv_dim = priv_dim_of(&envs[0]);

    let mut actor = warm.unwrap_or_else(|| Actor::new(obs_dim, act_dim, cfg.hidden, cfg.seed));
    assert_eq!(
        (actor.obs_dim, actor.act_dim),
        (obs_dim, act_dim),
        "warm actor was trained against a different observation/action schema"
    );
    set_init_std(&mut actor, cfg.init_std);
    actor.act_clamp = cfg.act_clamp;

    let mut critic = Mlp::new(obs_dim + priv_dim, cfg.hidden, 1, cfg.seed ^ 0xC217);
    let mut update = PpoUpdate::new(&actor, &critic, &cfg);
    let mut rng = XorShift::new(cfg.seed);
    let mut history = Vec::with_capacity(iterations);

    for it in 0..iterations {
        let episodes = collect(&mut envs, &actor, &cfg, &mut rng);
        let iter = process_iteration(episodes, &mut actor, &mut critic, &mut update, &cfg, &mut rng);
        progress(it, &iter, &actor);
        history.push(iter);
    }
    (actor, critic, history)
}

/// The artifact format's version line. Bumped only when the meaning of a
/// field changes; a file that does not carry it is not one of ours.
const ACTOR_MAGIC: &str = "# kosm-train actor v1";

/// Save an actor as a text artifact: a versioned header naming the
/// architecture and the schema, then the flat parameters.
///
/// The header is what makes [`load_actor`] able to refuse a file rather than
/// read the wrong columns out of it. ipse's format had no version and no
/// declared observation width for its first year, and the cost was an
/// `OBS_FAMILY=legacy55` environment variable and a paragraph of prose about
/// which 58-wide files meant which schema.
pub fn save_actor(actor: &Actor, path: &str, note: &str) -> std::io::Result<()> {
    use std::io::Write;
    let (i, h, o) = actor.net.dims;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "{ACTOR_MAGIC}")?;
    writeln!(f, "# arch mlp {i} {h} {o} tanh")?;
    writeln!(f, "# obs_dim {}", actor.obs_dim)?;
    writeln!(f, "# act_dim {}", actor.act_dim)?;
    writeln!(f, "# act_clamp {}", actor.act_clamp)?;
    writeln!(f, "# {note}")?;
    for v in actor.net.to_flat() {
        writeln!(f, "{v:.17e}")?;
    }
    Ok(())
}

/// Load an actor saved by [`save_actor`].
///
/// `log_std` comes back as ZEROS — exploration std 1.0 on every dimension,
/// not the 0.05 any trainer runs — because the file carries the mean network
/// and nothing else. [`train_from`] calls [`set_init_std`]; a caller that
/// collects with a warm actor by hand must too.
pub fn load_actor(path: &str) -> Option<Actor> {
    let text = std::fs::read_to_string(path).ok()?;
    if !text.lines().next().is_some_and(|l| l.trim() == ACTOR_MAGIC) {
        eprintln!("{path}: not a `{ACTOR_MAGIC}` artifact");
        return None;
    }
    let mut arch = None;
    let mut clamp = 0.3;
    let mut obs_dim = None;
    let mut act_dim = None;
    for line in text.lines().filter(|l| l.starts_with('#')) {
        let words: Vec<&str> = line.trim_start_matches('#').split_whitespace().collect();
        match words.as_slice() {
            ["arch", "mlp", i, h, o, "tanh"] => {
                arch = Some((i.parse().ok()?, h.parse().ok()?, o.parse().ok()?));
            }
            ["obs_dim", d] => obs_dim = d.parse().ok(),
            ["act_dim", d] => act_dim = d.parse().ok(),
            ["act_clamp", c] => clamp = c.parse().ok()?,
            _ => {}
        }
    }
    let (i, h, o) = arch?;
    let obs_dim = obs_dim?;
    let act_dim = act_dim?;
    if (i, o) != (obs_dim, act_dim) {
        eprintln!("{path}: header says {obs_dim}->{act_dim} but the net is {i}->{o}");
        return None;
    }
    let flat: Vec<f64> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    Some(Actor {
        net: Mlp::from_flat(i, h, o, &flat)?,
        log_std: vec![0.0; act_dim],
        act_clamp: clamp,
        obs_dim,
        act_dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::tests::Reach;

    #[test]
    fn an_actor_round_trips_through_a_file() {
        let mut a = Actor::new(4, 2, 8, 3);
        a.act_clamp = 0.45;
        let path = std::env::temp_dir().join(format!("kosm-train-actor-{}", std::process::id()));
        let path = path.to_str().unwrap();
        save_actor(&a, path, "test").unwrap();
        let b = load_actor(path).expect("loads");
        assert_eq!(a.net.to_flat(), b.net.to_flat());
        assert_eq!((b.obs_dim(), b.act_dim()), (4, 2));
        assert_eq!(b.act_clamp, 0.45);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn an_unversioned_file_is_refused() {
        let path = std::env::temp_dir().join(format!("kosm-train-bad-{}", std::process::id()));
        std::fs::write(&path, "# arch mlp 4 8 2 tanh\n0.0\n").unwrap();
        assert!(load_actor(path.to_str().unwrap()).is_none());
        let _ = std::fs::remove_file(&path);
    }

    /// PPO on the 1-D reach: the policy must learn to drive the position to
    /// a target it starts away from. The assertion is on the *improvement*
    /// over the first iteration, not an absolute return, because the return
    /// scale is the reward lens's business — but the margin is wide enough
    /// that a broken update cannot pass it.
    #[test]
    fn ppo_learns_to_reach_a_target() {
        let envs: Vec<Reach> = (0..4).map(|_| Reach::new(0.5, 40)).collect();
        let cfg = PpoConfig {
            episodes_per_iter: 16,
            init_std: 0.3,
            act_clamp: 1.0,
            lr: 3e-3,
            entropy_coef: 0.0,
            minibatch: 128,
            seed: 11,
            ..PpoConfig::default()
        };
        let (_actor, _critic, history) = train(envs, cfg, 150, |_, _, _| {});
        let first = history[0].mean_return;
        let last: f64 = history[history.len() - 10..]
            .iter()
            .map(|h| h.mean_return)
            .sum::<f64>()
            / 10.0;
        eprintln!("reach: first {first:.3} -> last-10 mean {last:.3}");
        assert!(
            last > first + 3.0,
            "PPO did not learn the reach: {first:.2} -> {last:.2}"
        );
    }

    /// `Actor::new` at the K1's shape is the network tang_train drew,
    /// including the 0.01 output scale: a hash of the bits, printed from
    /// 45e4509.
    #[test]
    fn a_k1_shaped_actor_inits_as_it_always_did() {
        let f = Actor::new(48, 12, 64, 7).net.to_flat();
        let h = f.iter().fold(0u64, |h, v| h.rotate_left(5) ^ v.to_bits());
        assert_eq!(h, 0x80bb4b1333cf5646);
    }

    /// A tiny minibatch for the loss-graph tests: four real rows chosen so
    /// every branch is live — a ratio inside the clip interval, one clipped
    /// above with A > 0 (no gradient), one clipped below with A < 0 (no
    /// gradient), one below with A > 0 (unclipped is the minimum, gradient
    /// flows) — and returns far enough from the critic that two rows sit on
    /// the Huber's linear arm. Six slots, so two are padding.
    struct Fixture {
        actor: Actor,
        critic: Mlp,
        samples: Vec<Sample>,
        b: usize,
        clip: f64,
        ent: f64,
        vdelta: f64,
    }

    /// `net` over `rows` (row-major), in f64 through the interpreter.
    fn forward64(net: &Mlp, rows: &[f64]) -> Vec<f64> {
        let n = rows.len() / net.dims.0;
        let (g, outs) = crate::net::forward_graph(net.dims, n);
        let mut inputs = vec![rows.to_vec()];
        for (r, _) in Mlp::blocks(net.dims) {
            inputs.push(net.params()[r].to_vec());
        }
        let refs: Vec<&[f64]> = inputs.iter().map(|v| v.as_slice()).collect();
        let mut p = Program::interpreted(&g, &outs, &refs);
        p.run();
        p.read_f64(0)
    }

    fn logp_of(act: &[f64], mean: &[f64], log_std: &[f64]) -> f64 {
        (0..act.len())
            .map(|d| {
                let z = (act[d] - mean[d]) / log_std[d].exp();
                -0.5 * z * z - log_std[d] - 0.5 * (2.0 * std::f64::consts::PI).ln()
            })
            .sum()
    }

    fn fixture() -> Fixture {
        let mut actor = Actor::new(2, 2, 3, 5);
        actor.net.scale_output(50.0); // undo most of the 0.01, so μ is not ≈ 0
        actor.log_std = vec![-0.3, 0.2];
        let critic = Mlp::new(3, 3, 1, 9);
        // (obs, priv, act, target ratio, adv, ret)
        let rows: [([f64; 2], f64, [f64; 2], f64, f64, f64); 4] = [
            ([0.3, -0.5], 0.7, [0.2, -0.1], 1.05, 0.8, 0.05),
            ([-0.6, 0.1], -0.2, [-0.4, 0.3], 1.6, 1.1, 2.0),
            ([0.9, 0.4], 0.1, [0.5, 0.6], 0.5, -0.7, -1.5),
            ([-0.1, -0.9], 0.5, [0.0, -0.5], 0.5, 0.6, 0.1),
        ];
        let samples = rows
            .iter()
            .map(|&(obs, p, act, r, adv, ret)| {
                let mean = forward64(&actor.net, &obs);
                Sample {
                    obs: obs.to_vec(),
                    priv_obs: vec![p],
                    act: act.to_vec(),
                    logp: logp_of(&act, &mean, &actor.log_std) - r.ln(),
                    reward: 0.0,
                    adv,
                    ret,
                }
            })
            .collect();
        Fixture { actor, critic, samples, b: 6, clip: 0.2, ent: 0.01, vdelta: 0.3 }
    }

    impl Fixture {
        /// Every graph input, f64, with the parameters taken from `theta`
        /// (actor flat ++ log_std ++ critic flat).
        fn inputs(&self, theta: &[f64]) -> Vec<Vec<f64>> {
            let rows: Vec<&Sample> = self.samples.iter().collect();
            let mut v: Vec<Vec<f64>> = pack_rows(&rows, self.b, 2, 2, 3).into_iter().collect();
            v.extend([vec![self.clip], vec![self.ent], vec![self.vdelta]]);
            let na = self.actor.net.params().len();
            for (r, _) in Mlp::blocks(self.actor.net.dims) {
                v.push(theta[r].to_vec());
            }
            v.push(theta[na..na + 2].to_vec());
            for (r, _) in Mlp::blocks(self.critic.dims) {
                v.push(theta[na + 2 + r.start..na + 2 + r.end].to_vec());
            }
            v
        }

        fn theta(&self) -> Vec<f64> {
            let mut t = self.actor.net.to_flat();
            t.extend(&self.actor.log_std);
            t.extend(self.critic.params());
            t
        }

        fn graph(&self) -> (TensorGraph, Vec<TensorId>) {
            loss_graph(self.b, self.actor.net.dims, self.critic.dims)
        }
    }

    fn run_interp(g: &TensorGraph, outs: &[TensorId], inputs: &[Vec<f64>]) -> Program {
        let refs: Vec<&[f64]> = inputs.iter().map(|v| v.as_slice()).collect();
        let mut p = Program::interpreted(g, outs, &refs);
        p.run();
        p
    }

    /// The gradient `backward` derives, on every parameter including
    /// `log_std`, against central differences of the same graph's loss in
    /// f64 — and against the derivatives ipse wrote by hand, so the switch
    /// from writing them to deriving them is checked to be a no-op.
    #[test]
    fn the_loss_gradient_is_the_one_ipse_derived() {
        let fx = fixture();
        let (g, outs) = fx.graph();
        let theta = fx.theta();
        let p = run_interp(&g, &outs, &fx.inputs(&theta));
        let grad: Vec<f64> = (OUT_GRAD..OUT_GRAD + 13).flat_map(|i| p.read_f64(i)).collect();
        assert_eq!(grad.len(), theta.len());

        let eps = 1e-6;
        for k in 0..theta.len() {
            let f = |d: f64| {
                let mut t = theta.clone();
                t[k] += d;
                run_interp(&g, &outs[OUT_LOSS..=OUT_LOSS], &fx.inputs(&t)).read_f64(0)[0]
            };
            let fd = (f(eps) - f(-eps)) / (2.0 * eps);
            assert!((fd - grad[k]).abs() < 1e-6 * (1.0 + fd.abs()), "θ[{k}]: fd {fd:.9} vs ir {:.9}", grad[k]);
        }

        // ipse's hand-derived terms, from f64 forwards of the same nets.
        let n = fx.samples.len() as f64;
        let (mut ploss, mut vloss, mut k3) = (0.0, 0.0, 0.0);
        let mut dlogstd = vec![-fx.ent; 2];
        let mut db3 = [0.0; 2];
        let mut dcb3 = 0.0;
        let (mut active, mut clipped) = (0, 0);
        for s in &fx.samples {
            let mean = forward64(&fx.actor.net, &s.obs);
            let std: Vec<f64> = fx.actor.log_std.iter().map(|l| l.exp()).collect();
            let lr = logp_of(&s.act, &mean, &fx.actor.log_std) - s.logp;
            let r = lr.exp();
            k3 += (r - 1.0 - lr) / n;
            let c = r.clamp(1.0 - fx.clip, 1.0 + fx.clip);
            ploss += -(r * s.adv).min(c * s.adv) / n;
            if r * s.adv <= c * s.adv + 1e-12 {
                active += 1;
                let coef = -s.adv * r / n;
                for d in 0..2 {
                    let z = (s.act[d] - mean[d]) / std[d];
                    db3[d] += coef * z / std[d];
                    dlogstd[d] += coef * (z * z - 1.0);
                }
            } else {
                clipped += 1;
            }
            let mut xc = s.obs.clone();
            xc.extend(&s.priv_obs);
            let err = forward64(&fx.critic, &xc)[0] - s.ret;
            vloss += err * err / n;
            dcb3 += 2.0 * err.clamp(-fx.vdelta, fx.vdelta) / n;
        }
        assert_eq!((active, clipped), (2, 2), "the fixture lost a branch");
        let close = |a: f64, b: f64, what: &str| assert!((a - b).abs() < 1e-10, "{what}: {a} vs {b}");
        close(p.read_f64(OUT_PLOSS)[0], ploss, "surrogate");
        close(p.read_f64(OUT_VLOSS)[0], vloss, "value mse");
        close(p.read_f64(OUT_K3)[0], k3, "k3");
        let ga_b3 = p.read_f64(OUT_GRAD + 5);
        let g_ls = p.read_f64(OUT_GRAD + 6);
        let gc_b3 = p.read_f64(OUT_GRAD + 12);
        for d in 0..2 {
            close(ga_b3[d], db3[d], "∂/∂μ through l3.b");
            close(g_ls[d], dlogstd[d], "∂/∂log σ");
        }
        close(gc_b3[0], dcb3, "Huber value gradient through l3.b");
    }

    /// The compiled f32 program and the f64 interpreter run the same graph.
    #[test]
    fn the_native_loss_matches_the_interpreter() {
        let fx = fixture();
        let (g, outs) = fx.graph();
        let inputs = fx.inputs(&fx.theta());
        let want = run_interp(&g, &outs, &inputs);
        let mut native = compile_zeroed(&g, &outs);
        assert!(native.is_native(), "no C compiler: this test needs one");
        for (k, v) in inputs.iter().enumerate() {
            native.write_f64(k, v);
        }
        native.run();
        for i in 0..outs.len() {
            for (a, b) in native.read_f64(i).iter().zip(want.read_f64(i)) {
                assert!((a - b).abs() < 2e-5 * (1.0 + b.abs()), "output {i}: native {a} vs interp {b}");
            }
        }
    }

    /// `PpoUpdate::step` moves both nets and `log_std`, and a ragged
    /// minibatch (4 rows into a program of 6) reports the means of its real
    /// rows only.
    #[test]
    fn an_update_step_moves_every_parameter() {
        let fx = fixture();
        let (mut actor, mut critic) = (fx.actor, fx.critic);
        let cfg = PpoConfig { minibatch: fx.b, clip: fx.clip, entropy_coef: fx.ent, ..PpoConfig::default() };
        let mut up = PpoUpdate::new(&actor, &critic, &cfg);
        let rows: Vec<&Sample> = fx.samples.iter().collect();
        let before = (actor.net.to_flat(), actor.log_std.clone(), critic.to_flat());
        let (_, vloss, _) = up.step(&rows, &mut actor, &mut critic, &cfg, fx.vdelta);
        let mean_err2: f64 = fx
            .samples
            .iter()
            .map(|s| {
                let mut xc = s.obs.clone();
                xc.extend(&s.priv_obs);
                (forward64(&Mlp::from_flat(3, 3, 1, &before.2).unwrap(), &xc)[0] - s.ret).powi(2)
            })
            .sum::<f64>()
            / 4.0;
        assert!((vloss - mean_err2).abs() < 1e-5 * (1.0 + mean_err2), "{vloss} vs {mean_err2}");
        assert_ne!(actor.net.to_flat(), before.0);
        assert!(actor.log_std.iter().zip(&before.1).all(|(a, b)| a != b));
        assert_ne!(critic.to_flat(), before.2);
    }
}


#[cfg(test)]
mod bench {
    use super::*;
    use crate::env::StepOut;

    /// A K1-shaped env with no physics: 48 observations, 12 actions, a
    /// fixed linear drift, 200 control steps. For timing the loop, not for
    /// learning anything.
    pub(crate) struct Wide {
        x: Vec<f64>,
        t: usize,
    }
    impl Env for Wide {
        fn obs_dim(&self) -> usize { 48 }
        fn act_dim(&self) -> usize { 12 }
        fn reset(&mut self, seed: u64) -> Vec<f64> {
            let mut r = XorShift::new(seed);
            self.x = (0..48).map(|_| 0.1 * r.normal()).collect();
            self.t = 0;
            self.x.clone()
        }
        fn step(&mut self, a: &[f64]) -> StepOut {
            for (i, v) in self.x.iter_mut().enumerate() {
                *v = 0.98 * *v + 0.05 * a[i % 12];
            }
            self.t += 1;
            let reward = -self.x.iter().map(|v| v * v).sum::<f64>();
            StepOut { obs: self.x.clone(), reward, done: self.t >= 200 }
        }
        fn truncated(&self) -> bool { self.t >= 200 }
    }

    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn time_ppo_iterations() {
        let envs: Vec<Wide> = (0..4).map(|_| Wide { x: vec![], t: 0 }).collect();
        let cfg = PpoConfig { episodes_per_iter: 12, ..PpoConfig::default() };
        let t0 = std::time::Instant::now();
        let mut marks = Vec::new();
        let (_a, _c, h) = train(envs, cfg, 6, |_, _, _| marks.push(t0.elapsed().as_secs_f64()));
        let per: Vec<f64> = marks.windows(2).map(|w| w[1] - w[0]).collect();
        eprintln!("first iteration {:.3} s; later iterations {:?} s; grad steps {}", marks[0], per, h[1].grad_steps);
    }
}
