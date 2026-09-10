//! The exposure meter: a light meter with a time constant, over the last
//! frame the renderer produced.
//!
//! An eye walking out of a cave does not see the beach correctly for a second
//! or so, and neither does a camera. Every engine ships this as
//! "auto-exposure" and implements it as a post-process guess; it is simpler
//! than that, and it is a lens: read the **log-average luminance** of the
//! frame that just went out, follow it with a first-order lag, and hand back
//! the multiplier the next frame's film should be scaled by.
//!
//! Log-average and not mean, because brightness is perceived in stops and a
//! mean is dominated by the sun: one blown highlight in a hundredth of the
//! frame moves an arithmetic mean more than the whole of the sand does. The
//! log-average is the geometric mean, which is the middle of the *stops*.
//!
//! # Where the multiplier goes
//!
//! The meter does not touch the renderer. What it produces is a number the
//! sim multiplies its authored exposure by, at the one place a film becomes
//! an image — the `exposure` argument of `sims/rune/render.rs`'s `to_image`,
//! which is `Film::to_srgb8`'s first argument:
//!
//! ```text
//! let e = meter.follow(&film.rgb, dt);
//! let image = render::to_image(&film, scene.authored.parameter_or("exposure", 0.7) * e);
//! ```
//!
//! Nothing about the trace changes: the film holds radiance, the meter reads
//! it, and the tonemap is where the aperture actually is.
//!
//! ```
//! use kosm::player::meter::Meter;
//!
//! // A meter calibrated on a frame is at unity on that frame.
//! let dim = vec![0.2f32; 3 * 64];
//! let meter = Meter::calibrated(&dim);
//! assert!((meter.exposure() - 1.0).abs() < 1e-12);
//!
//! // Twice as bright: it walks toward a half-stop-down multiplier, and it
//! // takes about a second to cover the first `1 − 1/e` of the way.
//! let bright = vec![0.4f32; 3 * 64];
//! for _ in 0..300 {
//!     meter.follow(&bright, 3.0 / 300.0);
//! }
//! assert!((meter.exposure() - 0.5).abs() < 0.05 * 0.5);
//! ```

use std::cell::Cell;

use crate::lens::Lens;
use crate::world::World;

/// Rec. 709 luminance of a linear RGB triple.
#[inline]
fn luma(rgb: &[f32]) -> f64 {
    0.2126 * rgb[0] as f64 + 0.7152 * rgb[1] as f64 + 0.0722 * rgb[2] as f64
}

/// Below this, a pixel is dark enough that its log is a measurement of the
/// renderer's noise floor rather than of the scene.
const FLOOR: f64 = 1e-4;

/// A light meter over the previous frame.
///
/// The state is one number — the log-average luminance it has settled on —
/// and it lives behind a [`Cell`] so the meter reads as a lens does, through
/// `&self`.
#[derive(Debug)]
pub struct Meter {
    /// The log-average luminance the authored exposure was chosen for. A
    /// frame this bright gets a multiplier of exactly one.
    pub key: f64,
    /// The e-folding time, seconds. One second: a frame twice as bright is
    /// 63% of the way to its new exposure after a second, and within five per
    /// cent of it after three.
    pub tau: f64,
    /// The multiplier is clamped to `[min, max]` around the authored
    /// exposure, so a meter cannot decide the level is a different level.
    pub min: f64,
    /// The upper bound of the same clamp.
    pub max: f64,
    log_avg: Cell<f64>,
}

impl Default for Meter {
    fn default() -> Self {
        Self::new(0.18)
    }
}

impl Meter {
    /// A meter keyed to a reference log-average luminance — the brightness
    /// the level's authored exposure is right for. Middle grey, 0.18, is the
    /// photographer's default and a reasonable one.
    pub fn new(key: f64) -> Self {
        let key = key.max(FLOOR);
        Self { key, tau: 1.0, min: 0.25, max: 4.0, log_avg: Cell::new(key.ln()) }
    }

    /// A meter keyed to a frame, so it starts at a multiplier of one: the
    /// level's authored exposure is taken to be right for the picture the
    /// level opens on.
    pub fn calibrated(rgb: &[f32]) -> Self {
        Self::new(log_average(rgb))
    }

    /// The time constant, seconds.
    pub fn with_tau(mut self, tau: f64) -> Self {
        self.tau = tau.max(1e-6);
        self
    }

    /// The bounds on the multiplier.
    pub fn bounded(mut self, min: f64, max: f64) -> Self {
        self.min = min.min(max);
        self.max = max.max(min);
        self
    }

    /// Forget the frame history and sit at unity again.
    pub fn reset(&self) {
        self.log_avg.set(self.key.ln());
    }

    /// Fold one frame in over `dt` seconds and hand back the multiplier.
    ///
    /// `rgb` is linear radiance, three floats a pixel — a
    /// `kosm_render::Film`'s `rgb`, or any slice shaped like one. The lag is
    /// exact rather than integrated (`1 − e^{−dt/τ}`), so the meter does not
    /// change its mind about how fast it is when the frame rate does.
    pub fn follow(&self, rgb: &[f32], dt: f64) -> f64 {
        let target = log_average(rgb).max(FLOOR).ln();
        let a = if dt > 0.0 { 1.0 - (-dt / self.tau.max(1e-9)).exp() } else { 0.0 };
        let l = self.log_avg.get();
        self.log_avg.set(l + (target - l) * a);
        self.exposure()
    }

    /// The multiplier as it stands, without reading a frame.
    pub fn exposure(&self) -> f64 {
        (self.key / self.log_avg.get().exp()).clamp(self.min, self.max)
    }

    /// The log-average luminance the meter has settled on. Its own reading,
    /// in the units it measures.
    pub fn luminance(&self) -> f64 {
        self.log_avg.get().exp()
    }
}

/// The geometric mean luminance of a linear RGB frame.
///
/// Pixels under [`FLOOR`] are floored rather than dropped: a frame that is
/// genuinely half black *is* dark, and throwing its black away would tell the
/// meter the picture is the bright half only.
pub fn log_average(rgb: &[f32]) -> f64 {
    let n = rgb.len() / 3;
    if n == 0 {
        return FLOOR;
    }
    let mut sum = 0.0;
    for i in 0..n {
        sum += luma(&rgb[i * 3..i * 3 + 3]).max(FLOOR).ln();
    }
    (sum / n as f64).exp()
}

/// The meter is a lens over a world only in the trivial sense: it reads no
/// world at all, because what it measures is the *picture*, which is what the
/// last lens produced. `see` hands back the standing multiplier, so a meter
/// can sit in a list of lenses beside the ones that do read the world.
impl Lens for Meter {
    type Out = f64;

    fn see(&self, _world: &World) -> f64 {
        self.exposure()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_calibrated_meter_starts_at_unity() {
        let f = vec![0.3f32; 3 * 100];
        let m = Meter::calibrated(&f);
        assert!((m.exposure() - 1.0).abs() < 1e-12);
        // and stays there while the picture does not change
        for _ in 0..100 {
            m.follow(&f, 0.01);
        }
        assert!((m.exposure() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn the_multiplier_is_bounded() {
        let dim = vec![0.2f32; 3 * 16];
        let m = Meter::calibrated(&dim).bounded(0.25, 4.0);
        let blinding = vec![100.0f32; 3 * 16];
        for _ in 0..1000 {
            m.follow(&blinding, 0.1);
        }
        assert!((m.exposure() - 0.25).abs() < 1e-12, "{}", m.exposure());
        let black = vec![0.0f32; 3 * 16];
        for _ in 0..1000 {
            m.follow(&black, 0.1);
        }
        assert!((m.exposure() - 4.0).abs() < 1e-12);
    }
}
