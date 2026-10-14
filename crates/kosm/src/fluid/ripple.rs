//! Ripples a ringing wall makes on a liquid: capillary–gravity waves.
//!
//! A wine glass ringing in its (n, 0) mode pushes the liquid at the
//! waterline in `cos nθ` lobes, hundreds of times a second. At that
//! frequency the restoring force is mostly surface tension, so the waves are
//! capillary ripples a millimetre long, and viscosity kills them within a
//! centimetre of the wall: a band of standing lobes round the rim, which is
//! what you see on a real glass.
//!
//! **Model.** Linear, deep water (the depth is many wavelengths). The wall is
//! a wavemaker: the surface at the wall moves with the wall's own normal
//! displacement, and a wave runs inward with the wavenumber the dispersion
//! relation `ω² = g k + (σ/ρ) k³` gives at the ring's ω. Since `k R` is in the
//! hundreds, the inward wave is the Bessel function's asymptote:
//!
//! `η(r, θ, t) = a e^{−β t} cos nθ √(R/r) e^{−α (R−r)} cos(k (R−r) − ω t)`
//!
//! with the ring's own decay β and the viscous spatial decay
//! `α = 2 ν k² / c_g` (Lamb's temporal rate `2 ν k²` carried at the group
//! velocity). One-way: the ripples take no energy back from the wall.
//! Faraday's parametric instability (waves at ω/2 above a threshold drive)
//! is out of scope; a tap is far below it.
//!
//! SI units.

use phyz_math::GRAVITY;

use crate::material::Fluid;

/// The capillary–gravity wavenumber at angular frequency `omega`, by Newton
/// on `ω² = g k + (σ/ρ) k³`.
pub fn wavenumber(omega: f64, density: f64, surface_tension: f64) -> f64 {
    let s = surface_tension / density;
    // start from whichever limit dominates
    let mut k = (omega * omega / GRAVITY).min((omega * omega / s).cbrt());
    for _ in 0..50 {
        let f = GRAVITY * k + s * k.powi(3) - omega * omega;
        let df = GRAVITY + 3.0 * s * k * k;
        let step = f / df;
        k -= step;
        if step.abs() < 1e-12 * k {
            break;
        }
    }
    k
}

/// Group velocity `dω/dk` at wavenumber `k`.
pub fn group_velocity(k: f64, density: f64, surface_tension: f64) -> f64 {
    let s = surface_tension / density;
    let omega = (GRAVITY * k + s * k.powi(3)).sqrt();
    (GRAVITY + 3.0 * s * k * k) / (2.0 * omega)
}

/// The ripple field inside a circular waterline of radius `radius`.
#[derive(Clone, Copy, Debug)]
pub struct Ripples {
    /// Waterline radius, m.
    pub radius: f64,
    /// Lobes round the rim: the wall mode's n.
    pub n: u32,
    /// Surface amplitude at the wall, m: the wall's own normal displacement there.
    pub amp: f64,
    /// Drive, rad/s: the wall mode's.
    pub omega: f64,
    /// Temporal decay of the drive, 1/s: the ring's.
    pub decay: f64,
    /// Wavenumber, 1/m.
    pub k: f64,
    /// Spatial decay inward from the wall, 1/m.
    pub alpha: f64,
}

impl Ripples {
    /// Ripples driven by a wall mode of `hz`, `n` lobes, displacing the
    /// waterline by `amp` metres and dying at `decay` per second, in `liquid`.
    ///
    /// # Panics
    /// If the liquid has no viscosity or no surface tension.
    pub fn from_wall(hz: f64, n: u32, amp: f64, decay: f64, radius: f64, liquid: &Fluid) -> Self {
        let sigma = liquid.surface_tension.expect("ripples need a surface tension");
        let nu = liquid.kinematic().expect("ripples need a liquid");
        let omega = std::f64::consts::TAU * hz;
        let k = wavenumber(omega, liquid.density, sigma);
        let alpha = 2.0 * nu * k * k / group_velocity(k, liquid.density, sigma);
        Self { radius, n, amp, omega, decay, k, alpha }
    }

    /// Wavelength, m.
    pub fn wavelength(&self) -> f64 {
        std::f64::consts::TAU / self.k
    }

    /// Radial envelope and phase at `r`: (amplitude factor, phase).
    fn radial(&self, r: f64) -> (f64, f64) {
        let d = (self.radius - r).max(0.0);
        let env = (self.radius / r.max(1e-6)).sqrt() * (-self.alpha * d).exp();
        (env, self.k * d)
    }

    /// Surface elevation at polar `(r, θ)`, time `t`.
    pub fn height(&self, r: f64, theta: f64, t: f64) -> f64 {
        let (env, phase) = self.radial(r);
        self.amp * (-self.decay * t).exp() * (self.n as f64 * theta).cos() * env * (phase - self.omega * t).cos()
    }

    /// `(∂η/∂x, ∂η/∂y)` at cartesian `(x, y)`, time `t`: the surface slope,
    /// which is what bends the light.
    pub fn slope(&self, x: f64, y: f64, t: f64) -> (f64, f64) {
        let r = (x * x + y * y).sqrt().max(1e-9);
        let theta = y.atan2(x);
        let nf = self.n as f64;
        let (env, phase) = self.radial(r);
        let a = self.amp * (-self.decay * t).exp();
        let wave = (phase - self.omega * t).cos();
        let dwave_dr = (phase - self.omega * t).sin() * self.k; // d/dr of cos(k(R−r) − ωt)
        // d env / dr = env (−1/(2r) + α) inside the wall
        let denv_dr = env * (-0.5 / r + if r < self.radius { self.alpha } else { 0.0 });
        let ang = (nf * theta).cos();
        let d_dr = a * ang * (denv_dr * wave + env * dwave_dr);
        let d_dth = a * -nf * (nf * theta).sin() * env * wave;
        let (c, s) = (x / r, y / r);
        (d_dr * c - d_dth * s / r, d_dr * s + d_dth * c / r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::named;

    fn wine() -> Fluid {
        named("red wine").expect("red wine").fluid()
    }

    #[test]
    fn the_wavenumber_satisfies_the_dispersion_relation() {
        let f = wine();
        let sigma = f.surface_tension.unwrap();
        for hz in [1.0, 20.0, 440.0, 2000.0] {
            let w = std::f64::consts::TAU * hz;
            let k = wavenumber(w, f.density, sigma);
            let back = (GRAVITY * k + sigma / f.density * k.powi(3)).sqrt();
            assert!((back - w).abs() / w < 1e-9, "{hz} Hz: k {k}");
        }
        // at 440 Hz a wine ripple is about a millimetre long
        let k = wavenumber(std::f64::consts::TAU * 440.0, f.density, sigma);
        let lambda = std::f64::consts::TAU / k;
        assert!((0.8e-3..1.5e-3).contains(&lambda), "λ = {:.3} mm", lambda * 1e3);
    }

    #[test]
    fn the_sampled_field_has_the_dispersion_wavelength() {
        let rip = Ripples::from_wall(440.0, 2, 1e-6, 0.0, 0.035, &wine());
        // zero crossings along θ = 0 inward from the wall, at t = 0
        let dr = 1e-6;
        let mut crossings = vec![];
        let mut prev = rip.height(rip.radius, 0.0, 0.0);
        let mut r = rip.radius;
        while r > rip.radius - 8e-3 {
            r -= dr;
            let h = rip.height(r, 0.0, 0.0);
            if h.signum() != prev.signum() {
                crossings.push(r);
            }
            prev = h;
        }
        let spacing = (crossings[0] - crossings[crossings.len() - 1]) / (crossings.len() - 1) as f64;
        let measured = 2.0 * spacing;
        assert!((measured - rip.wavelength()).abs() / rip.wavelength() < 0.05, "measured {measured:e} vs {:e}", rip.wavelength());
    }

    #[test]
    fn viscosity_confines_the_ripples_to_the_rim() {
        let rip = Ripples::from_wall(440.0, 2, 1e-6, 0.0, 0.035, &wine());
        let reach = 1.0 / rip.alpha;
        assert!((1e-3..3e-2).contains(&reach), "decay length {:.1} mm", reach * 1e3);
        let at = |d: f64| (0..64).map(|i| rip.height(rip.radius - d, 0.0, i as f64 / 64.0 * std::f64::consts::TAU / rip.omega).abs()).fold(0.0, f64::max);
        assert!(at(3.0 * reach) < 0.1 * at(0.0));
    }

    #[test]
    fn the_slope_is_the_heights_gradient() {
        let rip = Ripples::from_wall(440.0, 2, 1e-6, 1.5, 0.035, &wine());
        let t = 3.3e-4;
        for (x, y) in [(0.033, 0.004), (-0.02, 0.028), (0.001, -0.0345)] {
            let h = 1e-8;
            let at = |x: f64, y: f64| rip.height((x * x + y * y).sqrt(), y.atan2(x), t);
            let fd = ((at(x + h, y) - at(x - h, y)) / (2.0 * h), (at(x, y + h) - at(x, y - h)) / (2.0 * h));
            let (gx, gy) = rip.slope(x, y, t);
            let scale = rip.amp * rip.k;
            assert!((gx - fd.0).abs() < 1e-4 * scale && (gy - fd.1).abs() < 1e-4 * scale, "({x}, {y}): {:?} vs {:?}", (gx, gy), fd);
        }
    }
}
