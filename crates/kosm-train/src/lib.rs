//! Loops that produce policies.
//!
//! The seam this crate is on the far side of is drawn in
//! `docs/architecture.md`, "ipse, the first consumer": **core `kosm` holds the
//! contracts, `kosm-train` holds the loops.**
//!
//! - A [`kosm::task::Task`] is a contract — spawn, build, horizon, score,
//!   held out, invariants — and it lives in core, next to `diff` and the run
//!   hash, because the gate and the renderer read it too.
//! - `kosm::gate::check` and `kosm::ledger::Ledger` are assertions over that
//!   contract, and they live in core for the same reason: they are how a
//!   number from Tuesday is comparable to a number from Friday.
//! - Everything here *spends compute to move weights*. PPO ([`ppo`]),
//!   behaviour cloning ([`bc`]), CEM and MAP-Elites ([`search`]). None of it
//!   is a contract; all of it is a loop, and a loop can be replaced without
//!   invalidating a single stored score.
//!
//! The bridge between the two halves is [`env::TaskEnv`]: it wraps a core
//! `Task` plus a `Step` plus two lenses (what the policy sees, what it is
//! paid) into the flat [`env::Env`] a trainer wants — `Vec<f64>` in,
//! `Vec<f64>` out, one number of reward. Everything in this crate is generic
//! over `Env` and knows no robot, no marble, and no task by name.
//!
//! ```no_run
//! use kosm_train::{env::Env, ppo::{self, PpoConfig}};
//! # fn demo<E: Env + Send>(envs: Vec<E>) {
//! let cfg = PpoConfig { episodes_per_iter: 16, ..PpoConfig::default() };
//! let (actor, _critic, history) = ppo::train(envs, cfg, 200, |it, iter, _| {
//!     if it % 20 == 0 {
//!         println!("{it:4}  return {:8.2}", iter.mean_return);
//!     }
//! });
//! ppo::save_actor(&actor, "out/policy.actor", "200 iterations").unwrap();
//! # }
//! ```
//!
//! [`artifact`] is the policy ledger — what a trained thing must carry to be
//! runnable by someone other than its author. It is not `kosm`'s run ledger;
//! see that module's docs.

pub mod artifact;
pub mod bc;
pub mod env;
pub mod ppo;
pub mod search;

pub use env::{Env, StepOut, TaskEnv};
pub use search::XorShift;
