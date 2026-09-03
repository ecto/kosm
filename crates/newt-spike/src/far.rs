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
//!
//! This is a Hamiltonian system in Zakharov's variables, the elevation η and
//! the surface potential ψ (we carry v = ∂η/∂t = G[ψ] in physical space and
//! convert; G = k tanh kD is the Dirichlet-to-Neumann operator):
//!
//!   H = ½ ∫ (g η² + ψ G[ψ]) dA,   η̇ = δH/δψ,   ψ̇ = −δH/δη.
//!
//! The linear step is exact per mode, so H is conserved to roundoff; the
//! second-order nonlinear terms (NEWT_FAR_NL=1) add the steepening a real
//! ring has. Every joule the nudge puts in or takes out is booked in
//! `injected`, which is the port's ledger: when the coupling to the fine
//! solver is made two-way, that number has to match what the fine side lost.

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
    /// G = k tanh kD per mode: the Dirichlet-to-Neumann operator.
    gk: Vec<f64>,
    /// k per mode, for the nonlinear terms.
    kx: Vec<f64>,
    ky: Vec<f64>,
    /// Energy the nudge has put in (J), summed since the start.
    pub injected: f64,
    nonlinear: bool,
}

impl Far {
    pub fn new(cell: f64) -> Self {
        let nx = ((2.0 * POOL_X) / cell) as usize;
        let ny = ((2.0 * POOL_Y) / cell) as usize;
        // exact sizes: zero padding would let waves leak into the pad and be
        // truncated every step, which is neither periodic nor conserving
        let px = nx;
        let py = ny;
        let mut planner = FftPlanner::new();
        let mut omega = vec![0.0; px * py];
        let mut gk = vec![0.0; px * py];
        let mut kxs = vec![0.0; px * py];
        let mut kys = vec![0.0; px * py];
        for j in 0..py {
            for i in 0..px {
                let kx = std::f64::consts::TAU * (if i <= px / 2 { i as f64 } else { i as f64 - px as f64 }) / (px as f64 * cell);
                let ky = std::f64::consts::TAU * (if j <= py / 2 { j as f64 } else { j as f64 - py as f64 }) / (py as f64 * cell);
                let k = kx.hypot(ky);
                let g = k * (k * DEPTH).tanh();
                gk[j * px + i] = g;
                omega[j * px + i] = (GRAVITY * g).sqrt();
                kxs[j * px + i] = kx;
                kys[j * px + i] = ky;
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
            gk,
            kx: kxs,
            ky: kys,
            injected: 0.0,
            nonlinear: std::env::var("NEWT_FAR_NL").map(|v| v == "1").unwrap_or(false),
        }
    }

    fn to_spectral(&self, field: &[f64]) -> Vec<Complex<f64>> {
        let (nx, ny, px, py) = (self.grid.nx, self.grid.ny, self.px, self.py);
        let mut a = vec![Complex::new(0.0, 0.0); px * py];
        for j in 0..ny {
            for i in 0..nx {
                a[j * px + i] = Complex::new(field[j * nx + i], 0.0);
            }
        }
        self.fft2(&mut a, false);
        a
    }

    fn to_physical(&self, mut a: Vec<Complex<f64>>) -> Vec<f64> {
        let (nx, ny, px) = (self.grid.nx, self.grid.ny, self.px);
        self.fft2(&mut a, true);
        let mut f = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                f[j * nx + i] = a[j * px + i].re;
            }
        }
        f
    }

    /// The wave energy H = ½ ∫ (g η² + ψ G[ψ]) dA, in joules per unit
    /// density (multiply by 1000 kg/m³ for water). ψ G[ψ] is evaluated in
    /// spectral space where G is diagonal; v = G ψ so ψ G ψ = v²/G.
    pub fn energy(&self) -> f64 {
        let (px, py, cell) = (self.px, self.py, self.grid.cell);
        let hk = self.to_spectral(&self.grid.z);
        let vk = self.to_spectral(&self.v);
        let norm = (px * py) as f64;
        let mut e = 0.0;
        for m in 0..px * py {
            let g = self.gk[m];
            e += GRAVITY * hk[m].norm_sqr();
            if g > 1e-9 {
                e += vk[m].norm_sqr() / g;
            }
        }
        // Parseval: Σ|f_k|² = N Σ|f|², and each cell is cell² of area
        0.5 * e / norm * cell * cell
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
        if self.nonlinear {
            // second-order Zakharov terms, one explicit Euler step of them
            // before the exact linear step (a Strang split would be tidier):
            //   η̇ += −∇·(η ∇ψ) − G[η G[ψ]]
            //   ψ̇ += −½|∇ψ|² + ½ (G[ψ])²
            // with ψ_k = v_k / G_k and G[ψ] = v.
            let (px, py, nx, ny) = (self.px, self.py, self.grid.nx, self.grid.ny);
            let mut psik = vec![Complex::new(0.0, 0.0); px * py];
            let mut dpx = vec![Complex::new(0.0, 0.0); px * py];
            let mut dpy = vec![Complex::new(0.0, 0.0); px * py];
            for m in 0..px * py {
                let g = self.gk[m];
                if g > 1e-9 {
                    psik[m] = vk[m] / g;
                }
                let i = Complex::new(0.0, 1.0);
                dpx[m] = i * self.kx[m] * psik[m];
                dpy[m] = i * self.ky[m] * psik[m];
            }
            let psi_x = self.to_physical(dpx);
            let psi_y = self.to_physical(dpy);
            let eta = &self.grid.z;
            let gpsi = &self.v;
            // fluxes η ∇ψ and the product η G[ψ], back to spectral for their derivatives
            let fx: Vec<f64> = (0..nx * ny).map(|q| eta[q] * psi_x[q]).collect();
            let fy: Vec<f64> = (0..nx * ny).map(|q| eta[q] * psi_y[q]).collect();
            let eg: Vec<f64> = (0..nx * ny).map(|q| eta[q] * gpsi[q]).collect();
            let dpsi: Vec<f64> = (0..nx * ny).map(|q| -0.5 * (psi_x[q] * psi_x[q] + psi_y[q] * psi_y[q]) + 0.5 * gpsi[q] * gpsi[q]).collect();
            let fxk = self.to_spectral(&fx);
            let fyk = self.to_spectral(&fy);
            let egk = self.to_spectral(&eg);
            let dpsik = self.to_spectral(&dpsi);
            let i = Complex::new(0.0, 1.0);
            for m in 0..px * py {
                let div = i * self.kx[m] * fxk[m] + i * self.ky[m] * fyk[m];
                let deta = -div - self.gk[m] * egk[m];
                // η += dt·deta; ψ += dt·dpsi, and v = G ψ
                hk[m] += deta * dt;
                vk[m] += self.gk[m] * dpsik[m] * dt;
            }
        }
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
        let before = self.energy();
        self.force_inner(fine, inner, outer, t);
        self.injected += self.energy() - before;
    }

    fn force_inner(&mut self, fine: &HeightGrid, inner: f64, outer: f64, t: f64) {
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
