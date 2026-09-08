//! The pool's extents, as the engine sees them.

use super::{DEPTH, POOL_X, POOL_Y};

/// Authored pool knobs consumed by the pool's scene-specific computations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoolGeometry {
    pub half_extents: [f64; 2],
    pub depth: f64,
}

impl PoolGeometry {
    pub const fn reference() -> Self {
        Self {
            half_extents: [POOL_X, POOL_Y],
            depth: DEPTH,
        }
    }

    pub fn stand_y0(self) -> f64 {
        self.half_extents[1] + 3.0
    }
}

impl Default for PoolGeometry {
    fn default() -> Self {
        Self::reference()
    }
}

