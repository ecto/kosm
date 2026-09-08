//! kosm: the engine. Worlds, colliders, materials, audio, light, the BRep
//! path the picture is traced over, the denoiser and the fluid. The levels
//! that used to live here are `sims/` now.
pub mod analytic;
pub mod audio;
pub mod brep;
pub mod colliders;
pub mod denoise;
pub mod far;
pub mod fluid;
pub mod frame;
pub mod garage;
pub mod glass;
pub mod lamp;
pub mod light;
pub mod materials;
pub mod room;
pub mod scene;
/// The MPM splash lives in [`fluid`] now; this keeps its old path.
pub use fluid::splash;
