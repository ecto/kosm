//! The camera: where the picture is taken from, and the ray for a film point.

use super::V;

// ---- the picture ------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: V,
    pub target: V,
    pub vfov: f64,
    pub width: u32,
    pub height: u32,
    pub exposure: f64,
}

impl Camera {
    pub(super) fn basis(&self) -> (V, V, V) {
        let f = (self.target - self.eye).normalize();
        let r = f.cross(V::z()).normalize();
        let u = r.cross(f);
        (f, r, u)
    }

    /// The ray through film position `(sx, sy)` in pixels, `y` down.
    pub(super) fn ray(&self, sx: f64, sy: f64) -> (V, V) {
        let (f, r, u) = self.basis();
        let t = (0.5 * self.vfov).tan();
        let aspect = self.width as f64 / self.height as f64;
        let x = (2.0 * sx / self.width as f64 - 1.0) * t * aspect;
        let y = (1.0 - 2.0 * sy / self.height as f64) * t;
        (self.eye, (f + r * x + u * y).normalize())
    }
}

