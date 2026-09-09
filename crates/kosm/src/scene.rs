//! Units.
//!
//! This was the boundary between loon authorship and computation. Geometry is
//! Rust now — [`crate::build`] is where a level is written — and what is left
//! of the boundary is the one constant both sides have to agree on.

/// Millimetres, the native length unit of authored vcad geometry, to metres.
///
/// Authoring is millimetres and degrees because vcad is a CAD kernel;
/// simulation is metres and radians because phyz is a physics engine. Every
/// length crosses here and nowhere else: [`crate::colliders`] applies it on
/// the way into a model, [`crate::build::Built::millimetres`] on the way out
/// of a knob, and [`crate::brep::PER_M`] is its reciprocal for the picture.
pub const MM: f64 = 1e-3;
