//! kosm: the engine. Worlds, colliders, materials, audio, light, the BRep
//! path the picture is traced over, the denoiser and the fluid. The levels
//! that used to live here are `sims/` now.
pub mod analytic;
pub mod audio;
pub mod brep;
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
