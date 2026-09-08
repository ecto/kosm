//! Fluid: the water surface, its caustic, and the MPM splash.
//!
//! This is the engine half of what used to be the pool level — everything a
//! renderer or a viewer needs from water, with nothing about a watermelon or
//! a grandstand in it.

use tang::Vec3 as V;

pub mod caustic;
pub mod splash;
pub mod surface;

mod geometry;

pub use caustic::{Caustic, caustic, caustic_for_geometry, caustic_gpu, caustic_gpu_for_geometry};
pub use geometry::PoolGeometry;
pub use splash::Droplet;
pub use surface::{Ring, Surface};

// Reference dimensions for the fine-water solver. Coarse pool computations and
// renderers carry `PoolGeometry` from the authored scene instead.
pub const POOL_X: f64 = 25.0; // half-lengths of the water
pub const POOL_Y: f64 = 12.5;
pub const DEPTH: f64 = 2.0;
/// The fine MPM region: a disc of this radius (KOSM_BOX overrides) around
/// the region's centre, which is the pool's centre for now. Beyond it the
/// water is the far field (see `far`). The GPU grid spans the whole pool;
/// only the blocks the particles touch are allocated, so the region is a set
/// of particles, not a box — the first step toward water that appears where
/// something happens and leaves when it is over.
pub fn box_half() -> f64 {
    static HALF: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *HALF.get_or_init(|| std::env::var("KOSM_BOX").ok().and_then(|v| v.parse().ok()).unwrap_or(1.25))
}
/// Distance from the region's edge, positive inside: the same measure the
/// sponge, the render blend and the far field's nudge all use.
pub fn region_inset(x: f64, y: f64) -> f64 {
    box_half() - x.hypot(y)
}
pub const BOX_DEPTH: f64 = DEPTH; // the full depth: a floor the melon could fall through is no floor
/// The box's outer band where the fluid's velocity is damped so waves leave
/// instead of reflecting off a wall two metres from the splash.
pub const SPONGE: f64 = 0.25;
/// Width of the band, inside the sponge, over which the rendered surface
/// fades from the fine grid to the far field.
pub const BLEND: f64 = 0.4;
/// Frames per second of a recording (KOSM_FPS overrides).
pub fn fps() -> f64 {
    std::env::var("KOSM_FPS").ok().and_then(|v| v.parse().ok()).unwrap_or(60.0)
}
pub const COPING: f64 = 0.06; // deck height above the water line
pub const N_WATER: f64 = 1.333;
/// Absorption per metre, RGB: red goes first, which is why deep water is blue.
pub const ABSORB: [f64; 3] = [0.45, 0.10, 0.04];

// ---- the sun ----------------------------------------------------------------

pub fn sun_dir() -> V<f64> {
    V::new(-0.35, -0.45, 0.82).normalize()
}
pub const SUN_IRRADIANCE: f64 = 1.05;

// ---- the grandstand ---------------------------------------------------------

/// A stand along the far long side: stepped rows from the deck, a seat every
/// 0.6 m, most of them taken.
pub const STAND_ROWS: usize = 14;
pub const STAND_RISE: f64 = 0.45;
pub const STAND_TREAD: f64 = 0.85;
