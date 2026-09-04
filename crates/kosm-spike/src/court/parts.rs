//! Solids the picture draws that the physics does not own.
//!
//! The court's static parts come from the level's roots and the balls from
//! phyz; everything else that moves — a net, a strand, a decal a system
//! animates — is handed to the renderer as a placed vcad solid. Millimetres,
//! like every vcad solid; the placement is object → world.

use std::sync::Arc;

use vcad_kernel::Solid;
use vcad_kernel_math::Transform;

/// A vcad solid, where it is, and what it is made of (a root material name,
/// resolved the same way the level's roots are).
#[derive(Clone)]
pub struct PlacedSolid {
    pub solid: Arc<Solid>,
    /// Object → world, in millimetres.
    pub to_world: Transform,
    pub material: String,
}

/// Root materials that are appearance only: the physics leaves them out of
/// the court body, the picture draws them. `ball` is the ball's own solid,
/// placed at each ball's pose; `paint` is the court's markings; `decor` is
/// anything else that should not be stood on.
pub fn collides(material: &str) -> bool {
    !matches!(material, "ball" | "paint" | "decor" | "window")
}
