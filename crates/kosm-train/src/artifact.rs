//! The policy ledger: what a trained thing has to carry to be usable by
//! anyone but its author.
//!
//! Lifted from `ipse-dojo`'s `artifact.rs` unchanged. Note the name
//! collision, and that it is not an accident of naming but two different
//! books: [`Ledger`] here is the **policy ledger** — one entry per trained
//! artifact, its role map, its plant, its measured score — while
//! `kosm::ledger::Ledger` is the **run ledger**, one append-only line per
//! gate evaluation. A policy is written into this one when it is trained;
//! the score it earns is written into that one when it is gated.
//!
//! This is not bookkeeping. It is the conclusion of a measurement: the K1's
//! skateboard policy rides a Unitree G1 for 6.3 s at the leg gains it was
//! trained against (omega 40) and falls in 0.7 s at omega 60 — the gains
//! bare-ground PD actually prefers on that body. **The PD plant is part of
//! the policy.** A file of weights alone is not a policy anyone else can
//! run; it is a policy for one robot at one bandwidth, and nothing on disk
//! says which.
//!
//! So an artifact carries three things beyond its weights:
//!
//! - the **role map** it was trained against (slot k means "left hip pitch",
//!   whatever that joint is called on this robot),
//! - the **normalization** that made its numbers mean something, and
//! - the **plant**: gains, armature, control rate.
//!
//! Plus provenance, so a number can be traced to the run that produced it.

/// Which engine produced a score.
///
/// Carried explicitly and never inferred, for the same reason the robot's
/// mode is: it changes what every other number means. Policies do not
/// transfer between the impulse solver and the penalty-contact GPU path in
/// either direction — measured, both ways — so a held-out score is only
/// comparable to another score from the same engine. The app draws a number
/// as *the* score only when this says `Cpu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// The impulse-solver CPU physics. The referee.
    Cpu,
    /// The batched penalty-contact GPU physics. A sampling engine.
    Gpu,
}

impl Engine {
    pub fn label(&self) -> &'static str {
        match self {
            Engine::Cpu => "cpu",
            Engine::Gpu => "gpu",
        }
    }

    /// Whether a score from this engine may be reported as the run's score.
    pub fn is_referee(&self) -> bool {
        matches!(self, Engine::Cpu)
    }
}

/// One actuated slot in the canonical role space.
///
/// Role space rather than index space is what let a policy trained on one
/// humanoid run on another at all: `left_hip_pitch_joint` and
/// `Left_Hip_Pitch` are the same slot, and lowercase substring matching
/// resolved 10/10 leg roles on a Unitree G1 on the first attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct Role {
    /// Canonical role name, e.g. `left_hip_pitch`.
    pub role: String,
    /// What this robot calls the joint.
    pub joint: String,
    /// Its `v` index in this robot's model.
    pub v: usize,
}

/// The per-robot half of an artifact: everything that must travel with the
/// weights for them to mean the same thing on another machine or another
/// body.
#[derive(Debug, Clone)]
pub struct BodyPack {
    /// What robot this was trained on.
    pub robot: String,
    /// The canonical slots, in policy order.
    pub roles: Vec<Role>,
    /// Closed-loop bandwidth of the leg servos, rad/s. The measurement that
    /// made this field exist: the same weights ride at 40 and fall at 60.
    pub leg_omega: f64,
    pub leg_zeta: f64,
    /// Rotor inertia applied to actuated joints. URDF cannot express it, and
    /// without a default the mass-scaled gains explode in 0.08 s — so an
    /// artifact that does not say what it assumed is not reproducible.
    pub armature: f64,
    /// Control rate the policy was trained at, Hz. Replaying a 50 Hz policy
    /// at the 1 kHz physics rate is a different controller — measured, it
    /// turned a full-clock stand into a 2 s fall.
    pub control_hz: f64,
    /// Applied-action clamp, rad.
    pub act_clamp: f64,
}

impl BodyPack {
    /// Whether this artifact can be run against a robot presenting `roles`.
    ///
    /// Missing roles are the honest failure: a body without an ankle roll
    /// cannot run a policy whose slot 4 is an ankle roll, and finding that
    /// out at load time is better than a silent index mismatch. (That exact
    /// class already bit once: PD targets indexed by model DOF instead of
    /// registration order stood up fine in the upright pose and drove the
    /// robot into a pose nobody asked for in the loose one.)
    pub fn missing_roles(&self, available: &[String]) -> Vec<String> {
        self.roles
            .iter()
            .filter(|r| !available.iter().any(|a| a == &r.role))
            .map(|r| r.role.clone())
            .collect()
    }
}

/// A score, with the engine that produced it attached.
#[derive(Debug, Clone)]
pub struct Score {
    pub condition: String,
    pub value: f64,
    pub survived: bool,
    pub engine: Engine,
}

/// Everything about a trained artifact except the weights themselves.
#[derive(Debug, Clone)]
pub struct Provenance {
    /// The task specification's identity — what was being asked for.
    pub task: String,
    /// The **policy slot** this artifact fills: `walk`, `stand`, `step_up`.
    ///
    /// The role is the half of the ledger key that is morphology-agnostic;
    /// [`BodyPack::robot`] is the other half. A mode graph names roles and
    /// never names artifacts, so the same graph runs on a K1 and a G1 and
    /// only the resolution differs — and a role with no artifact for this
    /// body is a *training todo*, which is a different thing from a broken
    /// graph and must read as such.
    pub role: String,
    /// The body half of the contract.
    pub body: BodyPack,
    /// Held-out results. Only the `Cpu`-stamped ones are the run's score.
    pub held_out: Vec<Score>,
    /// The artifact this one warm-started from, if any. Lineage is how a
    /// week of runs becomes a browsable graph instead of a directory of
    /// filenames.
    pub parent: Option<String>,
    /// Source revision, so a number can be traced to the code that made it.
    pub rev: String,
}

impl Provenance {
    /// The run's headline number: the mean over CPU-refereed conditions.
    /// Returns `None` when nothing was refereed, which is a different thing
    /// from a score of zero and must be shown as such.
    pub fn refereed_mean(&self) -> Option<f64> {
        let refereed: Vec<&Score> =
            self.held_out.iter().filter(|s| s.engine.is_referee()).collect();
        if refereed.is_empty() {
            return None;
        }
        Some(refereed.iter().map(|s| s.value).sum::<f64>() / refereed.len() as f64)
    }

    /// How many refereed conditions the artifact survived to the clock.
    pub fn refereed_survivals(&self) -> usize {
        self.held_out
            .iter()
            .filter(|s| s.engine.is_referee() && s.survived)
            .count()
    }
}

/// A closed interval measured off a rollout: `[min, max]` of some scalar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Band {
    pub min: f64,
    pub max: f64,
}

impl Band {
    /// The empty band, which widens to the first sample handed to it.
    pub fn empty() -> Self {
        Band {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    pub fn new(min: f64, max: f64) -> Self {
        Band { min, max }
    }

    /// Whether any sample has been seen. An unwidened band is not a band of
    /// zero width around zero, and must never be read as one.
    pub fn is_empty(&self) -> bool {
        self.min > self.max
    }

    pub fn widen(&mut self, v: f64) {
        self.min = self.min.min(v);
        self.max = self.max.max(v);
    }

    pub fn width(&self) -> f64 {
        if self.is_empty() { 0.0 } else { self.max - self.min }
    }

    /// Whether `v` lies inside, with an absolute slack on each end.
    ///
    /// `tol` exists because a re-measurement is a *different rollout* — a
    /// shorter one, another engine build — and asking a chaotic contact
    /// simulation to reproduce an interval to the last bit is asking for a
    /// test that fails on a compiler upgrade.
    pub fn contains(&self, v: f64, tol: f64) -> bool {
        !self.is_empty() && v >= self.min - tol && v <= self.max + tol
    }
}

/// One role's posture interval over the measured window, radians.
#[derive(Debug, Clone, PartialEq)]
pub struct JointBand {
    pub role: String,
    pub band: Band,
}

/// **The measured steady-state envelope of a mode's policy.**
///
/// TRIX's decision, written here because this is the type that enforces it:
/// *entry sets are measured, not declared.* An edge (`step_up`, `mount`)
/// succeeds when it delivers the body into the next mode's entry set, so if
/// that set were a hand-written band the edge would be trained against a
/// wish. Declared bands are still allowed to exist — as **invariants the
/// measurement must fall inside** — but the number an edge is scored on
/// comes off a rollout of the artifact that actually owns the mode.
///
/// Two consequences worth stating, because both are load-bearing:
///
/// - Every field is a `[min, max]` over an explicit window of an explicit
///   rollout ([`Self::measured_over`]), not a mean. An edge that arrives at
///   the mean of a limit cycle is not inside anything.
/// - The engine is carried, for the same reason [`Score`] carries it: a band
///   measured on the GPU sampler is not the band the referee would draw.
#[derive(Debug, Clone)]
pub struct MeasuredEntry {
    /// Which steady state this is — a mode name (`stand`) or a named regime
    /// of one (`gait`). A mode may bank more than one.
    pub mode: String,
    /// Prose: the rollout, the config, and the window. What a reader needs to
    /// re-run the measurement rather than trust it.
    pub measured_over: String,
    /// The rollout window, seconds of simulated time.
    pub window: Band,
    /// Control samples inside the window.
    pub samples: usize,
    pub engine: Engine,
    /// Posture band per role, radians, in the artifact's role order.
    pub posture: Vec<JointBand>,
    /// Base linear velocity band, m/s, **heading-relative** (`vx, vy, vz`) —
    /// the frame `Intent.twist` is in, so an intent and an entry set are
    /// comparable without a rotation nobody wrote down.
    pub lin: [Band; 3],
    /// Base angular velocity band, rad/s, body frame (`wx, wy, wz`).
    pub ang: [Band; 3],
}

impl MeasuredEntry {
    pub fn posture_band(&self, role: &str) -> Option<Band> {
        self.posture.iter().find(|j| j.role == role).map(|j| j.band)
    }

    /// Which of the given `(role, angle)` postures fall outside the band.
    ///
    /// Returns names, not a bool, for the same reason
    /// [`BodyPack::missing_roles`] does: "the entry set was not met" is not
    /// actionable and "the left knee was 0.08 rad below it" is. A role this
    /// entry set never measured is reported as such rather than passed.
    pub fn posture_violations(&self, posture: &[(String, f64)], tol: f64) -> Vec<String> {
        let mut out = Vec::new();
        for (role, q) in posture {
            match self.posture_band(role) {
                Some(b) if b.contains(*q, tol) => {}
                Some(b) => out.push(format!(
                    "{role} {q:.4} outside [{:.4}, {:.4}]",
                    b.min, b.max
                )),
                None => out.push(format!("{role} not measured")),
            }
        }
        out
    }

    /// Whether a heading-relative base twist lies in the velocity band.
    pub fn admits_twist(&self, lin: [f64; 3], ang: [f64; 3], tol: f64) -> bool {
        (0..3).all(|i| self.lin[i].contains(lin[i], tol))
            && (0..3).all(|i| self.ang[i].contains(ang[i], tol))
    }
}

/// One row of the ledger: an artifact, by the key everything resolves it on.
#[derive(Debug, Clone)]
pub struct LedgerEntry {
    /// Stable identity, e.g. `k1/walk`.
    pub id: String,
    /// Where the payload lives, relative to the ledger root. For the walk
    /// this is the manifest itself — the "weights" are a `WalkConfig` and
    /// there is no second file — and saying so beats inventing an empty one.
    pub path: String,
    pub provenance: Provenance,
    /// The measured entry sets of the mode this artifact drives.
    pub entries: Vec<MeasuredEntry>,
}

impl LedgerEntry {
    pub fn role(&self) -> &str {
        &self.provenance.role
    }

    pub fn robot(&self) -> &str {
        &self.provenance.body.robot
    }

    pub fn entry_set(&self, mode: &str) -> Option<&MeasuredEntry> {
        self.entries.iter().find(|e| e.mode == mode)
    }
}

/// `(role, robot) -> artifact`, and nothing else.
///
/// Deliberately a list and a linear scan: the ledger has one entry today and
/// the interesting property is not lookup speed but that **there is exactly
/// one place a role is resolved**. No `builtin:walk` escape hatch — the walk
/// is a controller with parameters rather than weights, and it still resolves
/// through here, because the first artifact that skips the ledger is the one
/// that later has no `BodyPack` when a second body arrives.
#[derive(Debug, Clone, Default)]
pub struct Ledger {
    entries: Vec<LedgerEntry>,
}

impl Ledger {
    pub fn new(entries: Vec<LedgerEntry>) -> Self {
        Ledger { entries }
    }

    pub fn push(&mut self, entry: LedgerEntry) {
        self.entries.push(entry);
    }

    pub fn entries(&self) -> &[LedgerEntry] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The artifact for this slot on this body, or `None`.
    ///
    /// `None` is the "learning" colour on an edge — this body has no artifact
    /// for that role *yet* — and is never a fallback to another robot's
    /// weights. That is exactly the substitution the `BodyPack` exists to
    /// prevent.
    pub fn resolve(&self, role: &str, robot: &str) -> Option<&LedgerEntry> {
        self.entries
            .iter()
            .find(|e| e.role() == role && e.robot() == robot)
    }

    /// Roles this body has an artifact for.
    pub fn roles_for(&self, robot: &str) -> Vec<String> {
        self.entries
            .iter()
            .filter(|e| e.robot() == robot)
            .map(|e| e.role().to_string())
            .collect()
    }

    /// Bodies the ledger knows anything about.
    pub fn robots(&self) -> Vec<String> {
        let mut r: Vec<String> = self.entries.iter().map(|e| e.robot().to_string()).collect();
        r.sort();
        r.dedup();
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack() -> BodyPack {
        BodyPack {
            robot: "k1".into(),
            roles: ["left_hip_pitch", "left_knee", "right_ankle_roll"]
                .iter()
                .enumerate()
                .map(|(i, r)| Role {
                    role: (*r).into(),
                    joint: format!("J{i}"),
                    v: i,
                })
                .collect(),
            leg_omega: 40.0,
            leg_zeta: 1.0,
            armature: 0.05,
            control_hz: 50.0,
            act_clamp: 0.7,
        }
    }

    /// A body that cannot present a role is a load-time failure, not a
    /// silently misindexed rollout.
    #[test]
    fn a_missing_role_is_named_not_ignored() {
        let quadruped = vec!["left_hip_pitch".to_string(), "left_knee".to_string()];
        let missing = pack().missing_roles(&quadruped);
        assert_eq!(missing, vec!["right_ankle_roll".to_string()]);
        let humanoid: Vec<String> =
            pack().roles.iter().map(|r| r.role.clone()).collect();
        assert!(pack().missing_roles(&humanoid).is_empty());
    }

    /// A GPU number must never become the run's score.
    #[test]
    fn only_the_referee_scores() {
        let p = Provenance {
            task: "skate-stand".into(),
            role: "stand".into(),
            body: pack(),
            held_out: vec![
                Score { condition: "chocked".into(), value: 9000.0, survived: true, engine: Engine::Gpu },
                Score { condition: "free".into(), value: 1000.0, survived: true, engine: Engine::Cpu },
            ],
            parent: None,
            rev: "abc123".into(),
        };
        assert_eq!(p.refereed_mean(), Some(1000.0), "a GPU score reached the headline");
        assert_eq!(p.refereed_survivals(), 1);
    }

    /// Nothing refereed is not a zero.
    #[test]
    fn an_unrefereed_run_has_no_score() {
        let p = Provenance {
            task: "skate-stand".into(),
            role: "stand".into(),
            body: pack(),
            held_out: vec![Score {
                condition: "chocked".into(),
                value: 9000.0,
                survived: true,
                engine: Engine::Gpu,
            }],
            parent: None,
            rev: "abc123".into(),
        };
        assert!(p.refereed_mean().is_none(), "a GPU-only run reported a score");
    }

    fn walk_entry() -> LedgerEntry {
        LedgerEntry {
            id: "k1/walk".into(),
            path: "k1/walk.toml".into(),
            provenance: Provenance {
                task: "walk".into(),
                role: "walk".into(),
                body: pack(),
                held_out: Vec::new(),
                parent: None,
                rev: "abc123".into(),
            },
            entries: vec![MeasuredEntry {
                mode: "stand".into(),
                measured_over: "default walk, flat ground, [0.5, 1.0) s".into(),
                window: Band::new(0.5, 1.0),
                samples: 500,
                engine: Engine::Cpu,
                posture: vec![JointBand {
                    role: "left_knee".into(),
                    band: Band::new(0.10, 0.12),
                }],
                lin: [Band::new(-0.01, 0.01); 3],
                ang: [Band::new(-0.02, 0.02); 3],
            }],
        }
    }

    /// The one lookup every later rung goes through.
    #[test]
    fn the_ledger_resolves_a_role_on_a_body() {
        let ledger = Ledger::new(vec![walk_entry()]);
        let e = ledger.resolve("walk", "k1").expect("walk/k1 did not resolve");
        assert_eq!(e.id, "k1/walk");
        assert_eq!(e.robot(), "k1");
        assert!(e.entry_set("stand").is_some(), "no measured stand entry set");
        assert_eq!(ledger.roles_for("k1"), vec!["walk".to_string()]);
    }

    /// A body with no artifact for a role resolves to nothing — never to
    /// another robot's weights.
    #[test]
    fn an_unbuilt_body_resolves_to_none() {
        let ledger = Ledger::new(vec![walk_entry()]);
        assert!(ledger.resolve("walk", "g1").is_none(), "g1 borrowed k1's walk");
        assert!(ledger.resolve("step_up", "k1").is_none(), "an untrained edge resolved");
        assert!(ledger.roles_for("g1").is_empty());
        assert_eq!(ledger.robots(), vec!["k1".to_string()]);
    }

    /// Outside the measured band is a named joint, not a bool.
    #[test]
    fn the_entry_set_names_what_left_it() {
        let e = walk_entry();
        let entry = e.entry_set("stand").unwrap();
        assert!(entry.posture_violations(&[("left_knee".into(), 0.11)], 0.0).is_empty());
        let v = entry.posture_violations(&[("left_knee".into(), 0.4)], 0.0);
        assert_eq!(v.len(), 1);
        assert!(v[0].starts_with("left_knee"), "{v:?}");
        let unmeasured = entry.posture_violations(&[("neck_yaw".into(), 0.0)], 0.0);
        assert_eq!(unmeasured, vec!["neck_yaw not measured".to_string()]);
        assert!(entry.admits_twist([0.0; 3], [0.0; 3], 0.0));
        assert!(!entry.admits_twist([0.5, 0.0, 0.0], [0.0; 3], 0.0));
    }
}
