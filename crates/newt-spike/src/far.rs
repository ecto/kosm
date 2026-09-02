//! The far field: the rest of an Olympic pool as a linear wave field.
//!
//! The fine MPM box around the melon is a couple of metres across; the pool
//! is fifty. Outside the box the water is a linear wave field solved
//! spectrally with the real dispersion of water waves, ω² = g k tanh(k D),
//! which is exact for small waves of any wavelength: a 40 cm ring travels
//! at 0.8 m/s, a 10 m swell at 4 m/s. (A shallow-water solver sent the ring
//! out at 4.4 m/s, and the seam with the box showed the difference.) The
//! field is nudged toward the box's surface where they overlap; the box's
//! edge is a sponge that absorbs what comes back, so the coupling is
//! one-way and honest about it.

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

use crate::pool::{DEPTH, POOL_X, POOL_Y};
use crate::splash::HeightGrid;
use phyz_math::GRAVITY;

pub struct Far {
    pub grid: HeightGrid,
    /// dh/dt per cell.
    v: Vec<f64>,
    /// Heights the box nudged toward last frame, for the forced velocity.
    forced_prev: Vec<(usize, f64)>,
    forced_t: f64,
    pub time: f64,
    // spectral machinery: padded power-of-two transforms
    px: usize,
    py: usize,
    fwd_x: Arc<dyn Fft<f64>>,
    inv_x: Arc<dyn Fft<f64>>,
    fwd_y: Arc<dyn Fft<f64>>,
    inv_y: Arc<dyn Fft<f64>>,
    /// ω per mode.
    omega: Vec<f64>,
}

impl Far {
    pub fn new(cell: f64) -> Self {
        let nx = ((2.0 * POOL_X) / cell) as usize;
        let ny = ((2.0 * POOL_Y) / cell) as usize;
        let px = nx.next_power_of_two();
        let py = ny.next_power_of_two();
        let mut planner = FftPlanner::new();
        let mut omega = vec![0.0; px * py];
        for j in 0..py {
            for i in 0..px {
                let kx = std::f64::consts::TAU * (if i <= px / 2 { i as f64 } else { i as f64 - px as f64 }) / (px as f64 * cell);
                let ky = std::f64::consts::TAU * (if j <= py / 2 { j as f64 } else { j as f64 - py as f64 }) / (py as f64 * cell);
                let k = kx.hypot(ky);
                omega[j * px + i] = (GRAVITY * k * (k * DEPTH).tanh()).sqrt();
            }
        }
        Self {
            grid: HeightGrid { origin: [-POOL_X, -POOL_Y], cell, nx, ny, z: vec![0.0; nx * ny] },
            v: vec![0.0; nx * ny],
            forced_prev: Vec::new(),
            forced_t: 0.0,
            time: 0.0,
            px,
            py,
            fwd_x: planner.plan_fft_forward(px),
            inv_x: planner.plan_fft_inverse(px),
            fwd_y: planner.plan_fft_forward(py),
            inv_y: planner.plan_fft_inverse(py),
            omega,
        }
    }

    fn fft2(&self, a: &mut [Complex<f64>], inverse: bool) {
        let (px, py) = (self.px, self.py);
        let (fx, fy) = if inverse { (&self.inv_x, &self.inv_y) } else { (&self.fwd_x, &self.fwd_y) };
        for row in a.chunks_mut(px) {
            fx.process(row);
        }
        let mut col = vec![Complex::new(0.0, 0.0); py];
        for i in 0..px {
            for j in 0..py {
                col[j] = a[j * px + i];
            }
            fy.process(&mut col);
            for j in 0..py {
                a[j * px + i] = col[j];
            }
        }
        if inverse {
            let s = 1.0 / (px * py) as f64;
            for v in a.iter_mut() {
                *v *= s;
            }
        }
    }

    /// Advance by `dt`: every mode exactly, h_k(t+dt) = h_k cos ωdt + (v_k/ω) sin ωdt.
    pub fn step(&mut self, dt: f64) {
        let (nx, ny, px, py) = (self.grid.nx, self.grid.ny, self.px, self.py);
        let mut hk = vec![Complex::new(0.0, 0.0); px * py];
        let mut vk = vec![Complex::new(0.0, 0.0); px * py];
        for j in 0..ny {
            for i in 0..nx {
                hk[j * px + i] = Complex::new(self.grid.z[j * nx + i], 0.0);
                vk[j * px + i] = Complex::new(self.v[j * nx + i], 0.0);
            }
        }
        self.fft2(&mut hk, false);
        self.fft2(&mut vk, false);
        let damping = (-0.05 * dt).exp(); // per second: viscosity, the lane ropes
        for m in 0..px * py {
            let w = self.omega[m];
            let (c, s) = ((w * dt).cos(), (w * dt).sin());
            let (h, v) = (hk[m], vk[m]);
            if w > 1e-9 {
                hk[m] = (h * c + v * (s / w)) * damping;
                vk[m] = (v * c - h * (w * s)) * damping;
            }
        }
        self.fft2(&mut hk, true);
        self.fft2(&mut vk, true);
        for j in 0..ny {
            for i in 0..nx {
                self.grid.z[j * nx + i] = hk[j * px + i].re;
                self.v[j * nx + i] = vk[j * px + i].re;
            }
        }
        self.time += dt;
    }

    /// Nudge the far field toward the fine surface: fully inside `inner`,
    /// ramping to nothing at `outer` with the same smoothstep the renderer
    /// blends with, so the two surfaces agree by construction.
    pub fn force(&mut self, fine: &HeightGrid, inner: f64, outer: f64, t: f64) {
        let dt = (t - self.forced_t).max(1e-3);
        let (nx, ny, cell) = (self.grid.nx, self.grid.ny, self.grid.cell);
        let mut now = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let x = self.grid.origin[0] + (i as f64 + 0.5) * cell;
                let y = self.grid.origin[1] + (j as f64 + 0.5) * cell;
                let m = x.abs().max(y.abs());
                if m >= outer {
                    continue;
                }
                let u = ((outer - m) / (outer - inner)).clamp(0.0, 1.0);
                let a = u * u * (3.0 - 2.0 * u);
                now.push((j * nx + i, fine.at(x, y), a));
            }
        }
        for (k, (g, h, a)) in now.iter().enumerate() {
            let prev = self.forced_prev.get(k).filter(|(pg, _)| pg == g).map(|(_, ph)| *ph).unwrap_or(*h);
            let v_fine = (h - prev) / dt;
            self.v[*g] += a * (v_fine - self.v[*g]);
            self.grid.z[*g] += a * (h - self.grid.z[*g]);
        }
        self.forced_prev = now.into_iter().map(|(g, h, _)| (g, h)).collect();
        self.forced_t = t;
    }
}
