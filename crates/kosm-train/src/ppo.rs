//! PPO, generic over [`Env`].
//!
//! Lifted from `ipse-sim`'s `rl.rs` and made to forget the robot. What is
//! verbatim: [`Mlp`] (including its flat parameter order, which is a
//! documented fact of the artifact format), the GAE and clipped-surrogate
//! update in [`process_iteration`], the KL brake, the Huber value clip, and
//! every default in [`PpoConfig`]. What changed:
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
//!
//! The critic is asymmetric: it reads the observation *plus* whatever
//! [`Env::privileged`] returns, which the actor never sees.

use rayon::prelude::*;
use tang_tensor::{Shape, Tensor};
use tang_train::{Linear, Module, ModuleAdam, Optimizer, Parameter, Tanh};

use crate::env::Env;
use crate::search::XorShift;

/// A two-hidden-layer tanh MLP, `f64`, with flat import/export.
///
/// Hand-rolled rather than `Sequential` so the parameter order is a stable,
/// documented fact of the artifact format: `l1.w, l1.b, l2.w, l2.b, l3.w,
/// l3.b`, row-major.
pub struct Mlp {
    l1: Linear<f64>,
    a1: Tanh<f64>,
    l2: Linear<f64>,
    a2: Tanh<f64>,
    l3: Linear<f64>,
    pub dims: (usize, usize, usize),
}

impl Mlp {
    pub fn new(input: usize, hidden: usize, output: usize, seed: u64) -> Self {
        Self {
            l1: Linear::new(input, hidden, seed ^ 0x11),
            a1: Tanh::new(),
            l2: Linear::new(hidden, hidden, seed ^ 0x22),
            a2: Tanh::new(),
            l3: Linear::new(hidden, output, seed ^ 0x33),
            dims: (input, hidden, output),
        }
    }

    pub fn forward(&mut self, x: &Tensor<f64>) -> Tensor<f64> {
        let x = self.l1.forward(x);
        let x = self.a1.forward(&x);
        let x = self.l2.forward(&x);
        let x = self.a2.forward(&x);
        self.l3.forward(&x)
    }

    pub fn backward(&mut self, grad: &Tensor<f64>) {
        let g = self.l3.backward(grad);
        let g = self.a2.backward(&g);
        let g = self.l2.backward(&g);
        let g = self.a1.backward(&g);
        let _ = self.l1.backward(&g);
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<f64>> {
        let mut p = self.l1.parameters_mut();
        p.extend(self.l2.parameters_mut());
        p.extend(self.l3.parameters_mut());
        p
    }

    pub fn zero_grad(&mut self) {
        for p in self.parameters_mut() {
            p.zero_grad();
        }
    }

    /// A fresh net with the same weights and no cached activations — for
    /// forwarding slices of one batch on several threads. `forward` is
    /// `&mut self` because tang caches for `backward`, so one net cannot be
    /// shared across workers; a copy per worker can.
    pub fn clone_weights(&self) -> Self {
        Self::from_flat(self.dims.0, self.dims.1, self.dims.2, &self.to_flat())
            .expect("clone_weights: from_flat of to_flat")
    }

    pub fn to_flat(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for t in [
            &self.l1.weight.data,
            &self.l1.bias.data,
            &self.l2.weight.data,
            &self.l2.bias.data,
            &self.l3.weight.data,
            &self.l3.bias.data,
        ] {
            out.extend_from_slice(t.data());
        }
        out
    }

    pub fn from_flat(input: usize, hidden: usize, output: usize, flat: &[f64]) -> Option<Self> {
        let mut m = Self::new(input, hidden, output, 0);
        let mut at = 0;
        for t in [
            &mut m.l1.weight.data,
            &mut m.l1.bias.data,
            &mut m.l2.weight.data,
            &mut m.l2.bias.data,
            &mut m.l3.weight.data,
            &mut m.l3.bias.data,
        ] {
            let n = t.data().len();
            if at + n > flat.len() {
                return None;
            }
            t.data_mut().copy_from_slice(&flat[at..at + n]);
            at += n;
        }
        (at == flat.len()).then_some(m)
    }

    pub fn param_count(input: usize, hidden: usize, output: usize) -> usize {
        input * hidden + hidden + hidden * hidden + hidden + hidden * output + output
    }

    /// Scale the output layer's initial weights. The initial policy is then
    /// near-zero mean — plain PD at whatever pose the env spawns in, a
    /// controller that already works — and exploration is the log-std rather
    /// than Xavier noise saturating the clamp. Measured in ipse: without
    /// this, 120 iterations sat at 0.56 s episodes because the policy was
    /// fighting its own initialization.
    pub fn scale_output(&mut self, k: f64) {
        for v in self.l3.weight.data.data_mut() {
            *v *= k;
        }
    }
}

/// The actor: an MLP mean plus a global learned log-std vector.
pub struct Actor {
    pub net: Mlp,
    pub log_std: Parameter<f64>,
    /// Applied-action clamp, in the action's own units.
    pub act_clamp: f64,
    obs_dim: usize,
    act_dim: usize,
}

impl Actor {
    pub fn new(obs_dim: usize, act_dim: usize, hidden: usize, seed: u64) -> Self {
        let mut log_std = Parameter::new(Tensor::zeros(Shape::from_slice(&[act_dim])));
        for v in log_std.data.data_mut() {
            // std ≈ 0.05. Measured, not taste: at std 0.2 the exploration
            // itself shook the K1 down in 0.6 s, every episode, so PPO never
            // saw the standing regime its own mean already occupied.
            // Exploration around a working controller has to be gentle
            // enough to leave it working.
            *v = -3.0;
        }
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

    /// Deterministic action: the mean, clamped.
    pub fn mean_action(&mut self, obs: &[f64]) -> Vec<f64> {
        let x = actor_input(obs, self.obs_dim);
        let m = self.net.forward(&x);
        let mut a: Vec<f64> = m.data()[..self.act_dim].to_vec();
        self.clamp_action(&mut a);
        a
    }

    /// The weights and exploration widths only — enough to rebuild an actor
    /// on a worker thread. `forward` caches for `backward`, so one actor
    /// cannot be shared across threads, but a snapshot can.
    fn snapshot(&self) -> ActorSnapshot {
        ActorSnapshot {
            dims: self.net.dims,
            flat: self.net.to_flat(),
            log_std: self.log_std.data.data().to_vec(),
            act_clamp: self.act_clamp,
            obs_dim: self.obs_dim,
            act_dim: self.act_dim,
        }
    }
}

#[derive(Clone)]
struct ActorSnapshot {
    dims: (usize, usize, usize),
    flat: Vec<f64>,
    log_std: Vec<f64>,
    act_clamp: f64,
    obs_dim: usize,
    act_dim: usize,
}

impl ActorSnapshot {
    fn rebuild(&self) -> Actor {
        let mut log_std = Parameter::new(Tensor::zeros(Shape::from_slice(&[self.act_dim])));
        log_std.data.data_mut().copy_from_slice(&self.log_std);
        Actor {
            net: Mlp::from_flat(self.dims.0, self.dims.1, self.dims.2, &self.flat)
                .expect("actor snapshot round-trips"),
            log_std,
            act_clamp: self.act_clamp,
            obs_dim: self.obs_dim,
            act_dim: self.act_dim,
        }
    }
}

/// One observation as a row tensor of the network's input width. Shorter
/// observations are zero-padded and longer ones truncated, so a mismatch is a
/// quiet wrong answer rather than a panic only if the caller ignores
/// [`Actor::obs_dim`]; [`train_from`] asserts the widths agree up front.
pub(crate) fn actor_input(obs: &[f64], input_dim: usize) -> Tensor<f64> {
    let mut t = Tensor::zeros(Shape::from_slice(&[input_dim]));
    let n = obs.len().min(input_dim);
    t.data_mut()[..n].copy_from_slice(&obs[..n]);
    t
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
fn rollout<E: Env + ?Sized>(
    env: &mut E,
    actor: &mut Actor,
    cfg: &PpoConfig,
    rng: &mut XorShift,
    seed: u64,
    priv_dim: usize,
) -> Episode {
    let act_dim = actor.act_dim;
    let mut obs = env.reset(seed);
    let std: Vec<f64> = actor.log_std.data.data().iter().map(|l| l.exp()).collect();
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
        let x = actor_input(&obs, actor.obs_dim);
        let mean = actor.net.forward(&x);
        let mut act = vec![0.0; act_dim];
        let mut logp = 0.0;
        for i in 0..act_dim {
            let m = mean.data()[i];
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
pub fn collect<E: Env + Send>(
    envs: &mut [E],
    actor: &Actor,
    cfg: &PpoConfig,
    rng: &mut XorShift,
) -> Vec<Episode> {
    assert!(!envs.is_empty(), "collect needs at least one environment");
    let priv_dim = priv_dim_of(&envs[0]);
    let snap = actor.snapshot();
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
            let mut a = snap.rebuild();
            mine.into_iter()
                .map(|(i, s)| {
                    let mut r = XorShift::new(s);
                    (i, rollout(env, &mut a, cfg, &mut r, s, priv_dim))
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

/// GAE + the clipped update over one iteration's episodes.
pub fn process_iteration(
    episodes: Vec<Episode>,
    actor: &mut Actor,
    critic: &mut Mlp,
    opt_actor: &mut ModuleAdam,
    opt_critic: &mut ModuleAdam,
    cfg: &PpoConfig,
    rng: &mut XorShift,
) -> PpoIter {
    let obs_dim = actor.obs_dim;
    let act_dim = actor.act_dim;
    let critic_dim = critic.dims.0;
    let n_eps = episodes.len().max(1);

    // ── GAE ──
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
        let mut x = Tensor::zeros(Shape::from_slice(&[n + 1, critic_dim]));
        x.data_mut().copy_from_slice(&buf);
        let values = critic.forward(&x);
        let v = values.data();

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
            let b = chunk.len();
            let mut xa = Tensor::zeros(Shape::from_slice(&[b, obs_dim]));
            let mut xc = Tensor::zeros(Shape::from_slice(&[b, critic_dim]));
            for (row, &i) in chunk.iter().enumerate() {
                let s = &all[i];
                let k = s.obs.len().min(obs_dim);
                xa.data_mut()[row * obs_dim..row * obs_dim + k].copy_from_slice(&s.obs[..k]);
                xc.data_mut()[row * critic_dim..row * critic_dim + k].copy_from_slice(&s.obs[..k]);
                let p = s.priv_obs.len().min(critic_dim - obs_dim);
                xc.data_mut()[row * critic_dim + obs_dim..row * critic_dim + obs_dim + p]
                    .copy_from_slice(&s.priv_obs[..p]);
            }

            actor.net.zero_grad();
            if let Some(g) = actor.log_std.grad.as_mut() {
                for v in g.data_mut() {
                    *v = 0.0;
                }
            }
            let mean = actor.net.forward(&xa);
            let std: Vec<f64> = actor.log_std.data.data().iter().map(|l| l.exp()).collect();

            // New log-probs and the clipped-surrogate gradient wrt mean.
            let mut dmean = Tensor::zeros(Shape::from_slice(&[b, act_dim]));
            let mut dlogstd = vec![0.0; act_dim];
            let mut ploss = 0.0;
            // Schulman k3: mean(r − 1 − ln r) over THIS minibatch.
            let mut mb_k3 = 0.0;
            for (row, &i) in chunk.iter().enumerate() {
                let s = &all[i];
                let mut logp = 0.0;
                for d in 0..act_dim {
                    let m = mean.data()[row * act_dim + d];
                    let z = (s.act[d] - m) / std[d];
                    logp +=
                        -0.5 * z * z - std[d].ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
                }
                let log_ratio = logp - s.logp;
                let ratio = log_ratio.exp();
                mb_k3 += ratio - 1.0 - log_ratio;
                let clipped = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip);
                ploss += -(ratio * s.adv).min(clipped * s.adv);
                // Gradient flows only through the unclipped branch when it is
                // the active one.
                let active = (ratio * s.adv) <= (clipped * s.adv) + 1e-12;
                if active {
                    let coef = -s.adv * ratio / b as f64;
                    for d in 0..act_dim {
                        let m = mean.data()[row * act_dim + d];
                        let z = (s.act[d] - m) / std[d];
                        // d logp / d mean = z / std; d logp / d logstd = z² − 1.
                        dmean.data_mut()[row * act_dim + d] = coef * (z / std[d]);
                        dlogstd[d] += coef * (z * z - 1.0);
                    }
                }
            }
            // Entropy bonus: dH/dlogstd = 1 per dim.
            for d in 0..act_dim {
                dlogstd[d] -= cfg.entropy_coef;
            }
            actor.net.backward(&dmean);
            {
                let g = actor
                    .log_std
                    .grad
                    .get_or_insert_with(|| Tensor::zeros(Shape::from_slice(&[act_dim])));
                for (gd, d) in g.data_mut().iter_mut().zip(&dlogstd) {
                    *gd += *d;
                }
            }
            let mut actor_params = actor.net.parameters_mut();
            actor_params.push(&mut actor.log_std);
            opt_actor.step(&mut actor_params);

            // Critic: MSE to the GAE returns.
            critic.zero_grad();
            let values = critic.forward(&xc);
            let mut dv = Tensor::zeros(Shape::from_slice(&[b, 1]));
            let mut vloss = 0.0;
            for (row, &i) in chunk.iter().enumerate() {
                let err = values.data()[row] - all[i].ret;
                vloss += err * err;
                let d = err.clamp(-vdelta, vdelta);
                dv.data_mut()[row] = 2.0 * d / b as f64;
            }
            critic.backward(&dv);
            let mut critic_params = critic.parameters_mut();
            opt_critic.step(&mut critic_params);

            policy_loss_acc += ploss / b as f64;
            value_loss_acc += vloss / b as f64;
            loss_batches += 1;
            // Early stop: one more minibatch past this and the policy is no
            // longer the one the batch was collected under.
            let brake = mb_k3 / b as f64;
            brake_acc += brake;
            brake_batches += 1;
            if brake > cfg.target_kl {
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
    for v in actor.log_std.data.data_mut() {
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
    let mut opt_actor = ModuleAdam::new(cfg.lr);
    let mut opt_critic = ModuleAdam::new(cfg.lr);
    let mut rng = XorShift::new(cfg.seed);
    let mut history = Vec::with_capacity(iterations);

    for it in 0..iterations {
        let episodes = collect(&mut envs, &actor, &cfg, &mut rng);
        let iter = process_iteration(
            episodes,
            &mut actor,
            &mut critic,
            &mut opt_actor,
            &mut opt_critic,
            &cfg,
            &mut rng,
        );
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
        log_std: Parameter::new(Tensor::zeros(Shape::from_slice(&[act_dim]))),
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
        assert!(
            last > first + 3.0,
            "PPO did not learn the reach: {first:.2} -> {last:.2}"
        );
    }
}
