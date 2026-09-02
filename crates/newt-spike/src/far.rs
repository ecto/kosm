//! The far field: the rest of an Olympic pool as a height field.
//!
//! The fine MPM box around the melon is two metres across; the pool is
//! fifty. Outside the box the water is a linear shallow-water wave field on
//! a 10 cm grid, forced by the box's surface where the two overlap and
//! radiating whatever the splash sends outward. The box's edge is a sponge
//! that absorbs what comes back, so the coupling is one-way and honest
//! about it: far waves do not re-enter the box (they are millimetres by
//! the time they would).

use rayon::prelude::*;

use crate::pool::{DEPTH, POOL_X, POOL_Y};
use crate::splash::HeightGrid;
use phyz_math::GRAVITY;

pub struct Far {
    pub grid: HeightGrid,
    /// dh/dt per cell.
    v: Vec<f64>,
    /// Heights the box forced last frame, to give the forced cells a velocity.
    forced_prev: Vec<(usize, f64)>,
    forced_t: f64,
    pub time: f64,
}

impl Far {
    pub fn new(cell: f64) -> Self {
        let nx = ((2.0 * POOL_X) / cell) as usize;
        let ny = ((2.0 * POOL_Y) / cell) as usize;
        Self {
            grid: HeightGrid { origin: [-POOL_X, -POOL_Y], cell, nx, ny, z: vec![0.0; nx * ny] },
            v: vec![0.0; nx * ny],
            forced_prev: Vec::new(),
            forced_t: 0.0,
            time: 0.0,
        }
    }

    /// Advance by `dt`, in substeps that respect the wave CFL.
    pub fn step(&mut self, dt: f64) {
        let c2 = GRAVITY * DEPTH;
        let cell = self.grid.cell;
        let max_dt = 0.5 * cell / (c2.sqrt() * std::f64::consts::SQRT_2);
        let n = (dt / max_dt).ceil().max(1.0) as usize;
        let sub = dt / n as f64;
        let (nx, ny) = (self.grid.nx, self.grid.ny);
        for _ in 0..n {
            let z = &self.grid.z;
            let acc: Vec<f64> = (0..nx * ny)
                .into_par_iter()
                .map(|g| {
                    let (i, j) = (g % nx, g / nx);
                    // Neumann at the pool walls: the wall reflects
                    let l = z[j * nx + i.saturating_sub(1)];
                    let r = z[j * nx + (i + 1).min(nx - 1)];
                    let d = z[j.saturating_sub(1) * nx + i];
                    let u = z[(j + 1).min(ny - 1) * nx + i];
                    c2 * (l + r + d + u - 4.0 * z[g]) / (cell * cell)
                })
                .collect();
            let damping = 0.08; // per second: viscosity, and the lane ropes
            self.v.par_iter_mut().zip(&acc).for_each(|(v, a)| *v = (*v + a * sub) * (1.0 - damping * sub));
            self.grid.z.par_iter_mut().zip(&self.v).for_each(|(z, v)| *z += v * sub);
            self.time += sub;
        }
    }

    /// Nudge the far field toward the fine surface: fully inside `inner`,
    /// ramping to nothing at `outer` with the same smoothstep the renderer
    /// blends with, so the two surfaces agree by construction. Overwriting
    /// a square of cells left a step at its edge wherever the fine surface
    /// had a dent or a ring the far field had not been told about yet.
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
