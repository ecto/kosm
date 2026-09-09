//! The frozen gate: N fixed spawns, one deterministic number.
//!
//! A gate is `gate.toml` beside a task — a seed, a count, a tolerance — and
//! [`check`] turns it into a [`GateReport`]. Same task, same policy, same
//! spec, same number: the spawns come from the seed and the count and
//! nothing else, so a number from today is comparable with a number from a
//! month ago.
//!
//! ```
//! use kosm::prelude::*;
//! let spec: GateSpec = toml::from_str("seed = 7\ncount = 4\ntolerance = 0.01").unwrap();
//! assert_eq!(spec.count, 4);
//! ```

use crate::step::Policy;
use crate::task::Task;

/// What a `gate.toml` says.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GateSpec {
    /// The first seed. Spawn `i` is drawn at `seed + i`.
    pub seed: u64,
    /// How many spawns. Frozen: changing it changes the number.
    pub count: usize,
    /// How far a score may drift from a baseline before the gate is a
    /// regression. Not used by [`check`], which only measures; it is what
    /// [`GateReport::regressed_from`] compares against.
    pub tolerance: f64,
}

impl Default for GateSpec {
    fn default() -> Self {
        Self { seed: 0, count: 32, tolerance: 0.01 }
    }
}

impl GateSpec {
    pub fn load(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        Ok(toml::from_str(&text)?)
    }

    /// The spawns, in order. Deterministic in the spec.
    pub fn spawns<T: Task>(&self, task: &T) -> Vec<(u64, T::Spawn)> {
        (0..self.count as u64).map(|i| (self.seed + i, task.spawn(self.seed + i))).collect()
    }
}

/// One gate run: every draw, and the one number.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GateReport {
    pub task: String,
    pub seed: u64,
    pub count: usize,
    pub tolerance: f64,
    /// `(label, score)` per draw. The label is the seed for a sampled draw
    /// and the task's own name for a held-out one.
    pub scores: Vec<(String, f64)>,
    /// The mean score. This is *the* number.
    pub score: f64,
}

impl GateReport {
    /// Whether this report is worse than `baseline` by more than the
    /// tolerance. Higher scores are better.
    pub fn regressed_from(&self, baseline: f64) -> bool {
        self.score < baseline - self.tolerance
    }
}

/// Run a task's frozen gate under a policy.
pub fn check<T: Task, P: Policy>(task: &T, policy: &P, spec: &GateSpec) -> GateReport {
    let mut scores = Vec::with_capacity(spec.count);
    for (seed, spawn) in spec.spawns(task) {
        let traj = task.rollout(&task.build(&spawn), policy);
        scores.push((format!("seed:{seed}"), task.score(&traj, &spawn)));
    }
    for (label, spawn) in task.held_out() {
        let traj = task.rollout(&task.build(&spawn), policy);
        scores.push((format!("held-out:{label}"), task.score(&traj, &spawn)));
    }
    let score = if scores.is_empty() {
        0.0
    } else {
        scores.iter().map(|(_, s)| s).sum::<f64>() / scores.len() as f64
    };
    GateReport {
        task: task.name().to_owned(),
        seed: spec.seed,
        count: spec.count,
        tolerance: spec.tolerance,
        scores,
        score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::Zero;
    use crate::task::tests::Drop;

    #[test]
    fn a_gate_is_the_same_number_twice() {
        let spec = GateSpec { seed: 7, count: 3, tolerance: 0.01 };
        let a = check(&Drop, &Zero, &spec);
        let b = check(&Drop, &Zero, &spec);
        assert_eq!(a.score, b.score);
        assert_eq!(a.scores.len(), 3 + Drop.held_out().len());
        assert_eq!(a.scores[0].0, "seed:7");
        assert_eq!(a.scores[3].0, "held-out:low");
    }

    #[test]
    fn a_gate_regresses_only_past_its_tolerance() {
        let report = check(&Drop, &Zero, &GateSpec { seed: 0, count: 2, tolerance: 0.05 });
        assert!(!report.regressed_from(report.score + 0.04));
        assert!(report.regressed_from(report.score + 0.06));
    }

    #[test]
    fn a_spec_is_read_from_toml() {
        let spec: GateSpec = toml::from_str("seed = 3\ncount = 32\ntolerance = 0.02").unwrap();
        assert_eq!(spec, GateSpec { seed: 3, count: 32, tolerance: 0.02 });
        assert_eq!(spec.spawns(&Drop).len(), 32);
    }
}
