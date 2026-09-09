//! kosm: the engine.
//!
//! The public vocabulary is three words, and nothing else is a first-class
//! thing: [`World`](world::World) is columns, [`Step`](step::Step) is
//! `World → World` and pure, [`Lens`](lens::Lens) is `World → observation`.
//! Around them: [`diff`] between two worlds, [`snapshot`] assertions,
//! [`run`]'s hash and recorder, and [`task`], [`gate`] and [`ledger`], which
//! are contracts over the three nouns with no loop of their own.
//!
//! One import: [`prelude`]. The rest of the crate is what those three are
//! built out of — colliders, [`material`]'s substances, audio, light, the BRep path the
//! picture is traced over, the denoiser and the fluid. The levels that used
//! to live here are `sims/` now.
pub mod analytic;
pub mod audio;
pub mod brep;
pub mod build;
pub mod colliders;
pub mod denoise;
pub mod diff;
pub mod far;
pub mod fluid;
pub mod frame;
pub mod garage;
pub mod gate;
pub mod glass;
pub mod lamp;
pub mod ledger;
pub mod lens;
pub mod light;
pub mod material;
pub mod materials;
pub mod prelude;
pub mod room;
pub mod run;
pub mod scene;
pub mod snapshot;
pub mod step;
pub mod task;
pub mod world;
/// The MPM splash lives in [`fluid`] now; this keeps its old path.
pub use fluid::splash;
