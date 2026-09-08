//! kosm as a library: the marble, the lamp, the frame, the glass, the
//! pool and the splash, so the viewer and the CLI share one implementation.
pub mod analytic;
pub mod audio;
pub mod brep;
pub mod colliders;
pub mod court;
pub mod denoise;
pub mod far;
pub mod fluid;
pub mod frame;
pub mod garage;
pub mod glass;
pub mod lamp;
pub mod light;
pub mod materials;
pub mod pool;
pub mod room;
pub mod scene;
pub mod skatepark;

/// The MPM splash lives in [`fluid`] now; this keeps its old path.
pub use fluid::splash;
