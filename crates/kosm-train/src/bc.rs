//! Behaviour cloning: fit an actor's mean network to demonstrated actions.
//!
//! Lifted from `ipse-sim`'s `bc.rs` — [`fit_actor`] and the standardisation
//! it depends on, and nothing else from that file (the demonstrator rollouts,
//! the DAPG auxiliary term and the ollie scripts are ipse's, not a loop kosm
//! owns). The arrays are `Vec<f64>` here and the observation is written
//! straight into the network's input width, since there is no blind mask and
//! no command block to pack around.

use tang_tensor::{Shape, Tensor};
use tang_train::{ModuleAdam, Optimizer};

use crate::ppo::{Actor, Mlp};
use crate::search::XorShift;

/// One demonstrated control step: what the policy would have seen, and what
/// the demonstrator did about it.
#[derive(Clone, Debug)]
pub struct BcSample {
    pub obs: Vec<f64>,
    /// The demonstrated action — already clamped, so a regression fit cannot
    /// be asked to reach past the clamp it will be applied through.
    pub act: Vec<f64>,
}

/// Supervised-fit knobs. Defaults are the ones ipse's paired stand runs used.
#[derive(Clone, Copy, Debug)]
pub struct BcFit {
    pub epochs: usize,
    pub lr: f64,
    pub minibatch: usize,
    pub seed: u64,
}

impl Default for BcFit {
    fn default() -> Self {
        Self { epochs: 40, lr: 1e-3, minibatch: 512, seed: 0xBC0F }
    }
}

/// Per-dimension mean and standard deviation of the demonstrated actions.
///
/// This is not a nicety, it is the difference between the fit working and the
/// fit being a null. **The demonstrated action is a whisper**: measured on
/// ipse's 32 frozen draws, a projected hold commands about 5 mrad RMS against
/// an action clamp of 450 mrad, because the offset that reproduces a torque
/// is `tau / kp` and the leg gains are large.
///
/// Regressing that directly with a Xavier-initialised head asks the optimiser
/// to resolve a signal three orders of magnitude below the scale its weights
/// are drawn at, and it does not: the first attempt reached 6.8e-4 rad RMS
/// error — 13 % of the signal — and the cloned actor scored 0/32 on the
/// frozen gate while the controller it copied scored 22/32.
///
/// So the fit runs in standardised space and the affine is folded back into
/// the output layer afterwards, which is exact because that layer is linear.
struct ActStats {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl ActStats {
    fn of(samples: &[BcSample], act_dim: usize) -> Self {
        let n = samples.len().max(1) as f64;
        let mut mean = vec![0.0; act_dim];
        let mut std = vec![0.0; act_dim];
        for s in samples {
            for d in 0..act_dim.min(s.act.len()) {
                mean[d] += s.act[d] / n;
            }
        }
        for s in samples {
            for d in 0..act_dim.min(s.act.len()) {
                std[d] += (s.act[d] - mean[d]).powi(2) / n;
            }
        }
        for v in std.iter_mut() {
            // A dimension the demonstrator never moves has std 0. Folding a
            // zero scale back zeroes that output row, which is exactly right
            // — the demonstrator's action there IS the constant mean — but
            // the standardised target must not be a division by zero.
            *v = v.sqrt();
            if *v < 1e-12 {
                *v = 0.0;
            }
        }
        Self { mean, std }
    }

    fn standardize(&self, act: &[f64], d: usize) -> f64 {
        if self.std[d] == 0.0 { 0.0 } else { (act[d] - self.mean[d]) / self.std[d] }
    }

    /// Fold `y -> mean + std * y` into the network's linear output layer, so
    /// the returned network predicts raw actions. Exact: `l3` is `W·h + b`,
    /// so scaling its row and shifting its bias scales and shifts the output
    /// and nothing else.
    ///
    /// The flat parameter order is a documented, stable fact of the artifact
    /// format ([`Mlp`]'s own doc comment): l1.w, l1.b, l2.w, l2.b, l3.w,
    /// l3.b, row-major.
    fn fold_into(&self, net: &mut Mlp) {
        let (i, h, o) = net.dims;
        let mut flat = net.to_flat();
        let l3_w = i * h + h + h * h + h;
        let l3_b = l3_w + h * o;
        for d in 0..o {
            for c in 0..h {
                flat[l3_w + d * h + c] *= self.std[d];
            }
            flat[l3_b + d] = flat[l3_b + d] * self.std[d] + self.mean[d];
        }
        *net = Mlp::from_flat(i, h, o, &flat).expect("fold: from_flat of to_flat");
    }
}

/// Fit an actor's MLP to `samples` by mean-squared regression on the action,
/// returning the per-epoch losses **in the raw action space's own squared
/// units**, so a number here is comparable across runs and against the
/// action's own scale.
///
/// The regression runs on standardised targets — see `ActStats` — and the
/// scale is folded back into the linear output layer at the end.
///
/// Only the mean network is touched. `log_std` is PPO's business, and
/// [`crate::ppo::train_from`] overwrites it with
/// [`set_init_std`](crate::ppo::set_init_std) regardless.
pub fn fit_actor(actor: &mut Actor, samples: &[BcSample], fit: BcFit) -> Vec<f64> {
    if samples.is_empty() {
        return Vec::new();
    }
    let a_in = actor.net.dims.0;
    let act_dim = actor.act_dim();
    let stats = ActStats::of(samples, act_dim);
    // Variance per dimension, to convert a standardised loss back to the
    // action's own squared units.
    let var: Vec<f64> = stats.std.iter().map(|s| s * s).collect();
    let mut opt = ModuleAdam::new(fit.lr);
    let mut rng = XorShift::new(fit.seed);
    let n = samples.len();
    let mut losses = Vec::with_capacity(fit.epochs);
    for _ in 0..fit.epochs {
        // Fisher-Yates from the same PRNG, as the PPO update does.
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let mut acc = 0.0;
        let mut batches = 0usize;
        for chunk in order.chunks(fit.minibatch.max(1)) {
            let b = chunk.len();
            let mut x = Tensor::zeros(Shape::from_slice(&[b, a_in]));
            for (row, &i) in chunk.iter().enumerate() {
                let s = &samples[i];
                let k = s.obs.len().min(a_in);
                x.data_mut()[row * a_in..row * a_in + k].copy_from_slice(&s.obs[..k]);
            }
            actor.net.zero_grad();
            let out = actor.net.forward(&x);
            let mut grad = Tensor::zeros(Shape::from_slice(&[b, act_dim]));
            let mut loss = 0.0;
            for (row, &i) in chunk.iter().enumerate() {
                for d in 0..act_dim {
                    let err =
                        out.data()[row * act_dim + d] - stats.standardize(&samples[i].act, d);
                    // Reported in the action's units: a standardised residual
                    // on dimension `d` is worth `var[d]` of real variance.
                    loss += err * err * var[d];
                    grad.data_mut()[row * act_dim + d] = 2.0 * err / (b * act_dim) as f64;
                }
            }
            actor.net.backward(&grad);
            let mut params = actor.net.parameters_mut();
            opt.step(&mut params);
            acc += loss / (b * act_dim) as f64;
            batches += 1;
        }
        losses.push(acc / batches.max(1) as f64);
    }
    stats.fold_into(&mut actor.net);
    losses
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The demonstrator is a linear map, so the fit has something exactly
    /// representable to reach and "did the loop learn" is not a judgement
    /// call. The scale is deliberately a whisper — see `ActStats`.
    #[test]
    fn it_fits_a_linear_map() {
        let mut rng = XorShift::new(4);
        let samples: Vec<BcSample> = (0..1024)
            .map(|_| {
                let o = vec![rng.normal(), rng.normal(), rng.normal()];
                let act = vec![
                    0.004 * o[0] - 0.002 * o[1],
                    0.001 * o[1] + 0.003 * o[2],
                ];
                BcSample { obs: o, act }
            })
            .collect();

        let mut actor = Actor::new(3, 2, 32, 7);
        let losses = fit_actor(
            &mut actor,
            &samples,
            BcFit { epochs: 300, lr: 5e-3, minibatch: 256, seed: 1 },
        );
        assert!(
            losses[losses.len() - 1] < losses[0] * 0.05,
            "loss barely moved: {:.3e} -> {:.3e}",
            losses[0],
            losses[losses.len() - 1]
        );

        // And the folded network predicts raw actions, not standardised ones.
        let mut err = 0.0;
        for s in samples.iter().take(64) {
            let p = actor.net.forward(&crate::ppo::actor_input(&s.obs, 3));
            for d in 0..2 {
                err += (p.data()[d] - s.act[d]).powi(2);
            }
        }
        let rms = (err / (64.0 * 2.0)).sqrt();
        assert!(rms < 3e-4, "cloned actor is off by {rms:.3e}");
    }
}
