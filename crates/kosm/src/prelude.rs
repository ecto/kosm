//! One import.
//!
//! ```
//! use kosm::prelude::*;
//! # fn main() -> anyhow::Result<()> {
//! // build: a world out of a phyz rig and some knobs
//! let (model, state) = kosm::world::demo_marble();
//! let world = World::from_phyz(model, state).with_params(vec![Param::new("tilt", 0.05)]);
//!
//! // run: a pure step, and a rollout that is a trajectory
//! let traj = rollout(&world, &PhyzStep::new(1e-3), &Zero, 40);
//!
//! // observe: lenses read the world, they do not hold it
//! let drop = reward("drop", |w: &World| world_z(w) - 0.2);
//! fn world_z(w: &World) -> f64 { w.q()[5] }
//! assert!(drop.see(traj.last().unwrap()) < 0.0);
//!
//! // and worlds diff
//! assert!(!diff(traj.first().unwrap(), traj.last().unwrap()).is_within(1e-9));
//! # Ok(()) }
//! ```

pub use crate::diff::{ColumnDiff, Diff, diff};
pub use crate::gate::{self, GateReport, GateSpec};
pub use crate::ledger::{Entry as LedgerEntry, Ledger};
pub use crate::lens::{Camera, Column, Frame, Lens, Probe, Reward, reward};
pub use crate::run::{Manifest, Recorder, RunId};
pub use crate::snapshot::{assert_close, assert_image_close};
pub use crate::step::{Action, PhyzStep, Policy, Step, Trajectory, Zero, rollout, step_batch};
pub use crate::task::{Invariant, Task, check_invariants};
pub use crate::world::{Light, Material, Param, World};
// substances: one `Material` per stuff, every facet derived from its
// constants. Spelled `Substance` here because `Material` above is the render
// column on a `World`, and the two are different nouns.
pub use crate::material::{self, Fluid, Material as Substance, Optics, Spectrum};

// the player: a body on the ground, through a medium, and what drives it.
// `player::Snapshot` and `player::Pose` keep their module in a name because a
// sim has snapshots and poses of its own.
pub use crate::player::{self, Body, BodySpec, Drive, Ground, Medium, Skeleton, Tool};

// the build side: geometry in rust, and the colliders derived from it
pub use crate::build::{Authored, Built, Params, build};
pub use crate::colliders::{Derived as DerivedColliders, colliders_from_document};
pub use crate::scene::MM;
