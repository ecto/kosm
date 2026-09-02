//! The splash: water as particles, the melon as a body, coupled.
//!
//! A dense-grid MPM for weakly compressible water (APIC transfer, quadratic
//! B-splines, an equation of state with a modest speed of sound), with one
//! thing phyz-particle's reference solver does not have: a **rigid collider
//! that pushes back**. Grid nodes inside the melon take the melon's velocity
//! along the surface normal; the momentum that costs the fluid is booked and
//! handed to phyz as the force on the melon. Buoyancy is no longer a formula
//! here. It is what the water does.
//!
//! This is the organ phyz-particle should grow. It lives here first so the
//! coupling and the pool can iterate together; the interface it wants from
//! phyz-coupling is a `Solver` whose external input is a rigid body's pose
//! and velocity and whose output is a wrench.
//!
//! Units: metres, z up, water at rest at z = 0.

use phyz_math::{GRAVITY, Mat3, Vec3};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Nanoseconds spent per phase of `Water::step` since the last `take_prof`:
/// zero, bin, p2g, grid, g2p.
static PROF: [AtomicU64; 5] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
fn prof(slot: usize, t: &mut std::time::Instant) {
    PROF[slot].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    *t = std::time::Instant::now();
}
pub fn take_prof() -> [u64; 5] {
    std::array::from_fn(|i| PROF[i].swap(0, Ordering::Relaxed))
}

use crate::pool::{DEPTH, MELON_AXES, POOL_X, POOL_Y};

pub struct Water {
    pub h: f64,
    pub dt: f64,
    /// Grid origin (node 0,0,0) and node counts.
    pub origin: Vec3,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    // particles
    pub x: Vec<Vec3>,
    pub v: Vec<Vec3>,
    pub c: Vec<Mat3>,
    /// Volume ratio (1 at rest).
    pub j: Vec<f64>,
    pub mass: f64,
    pub vol0: f64,
    // grid
    g_mass: Vec<f64>,
    g_mom: Vec<Vec3>,
    /// Grid velocity before the update, for the FLIP part of the transfer.
    g_vel_old: Vec<Vec3>,
    /// FLIP share of the particle velocity update (0 = PIC, 1 = FLIP).
    pub flip: f64,
    /// Bulk modulus (Pa): sets the speed of sound, and so the timestep.
    pub bulk: f64,
    pub time: f64,
    /// Diagnostics from the last step: water mass inside the body, and the
    /// part of the reaction that is just that mass' weight.
    pub interior_mass: f64,
    /// The extracted surface's rest level, so still water reads as z = 0.
    pub level_offset: f64,
    /// Particle order by scatter group from the last step (see `compact`).
    order: Vec<u32>,
}

/// The collider the water feels: an ellipsoid with a velocity.
#[derive(Clone, Copy)]
pub struct Body {
    pub centre: Vec3,
    pub axis: Vec3,
    pub vel: Vec3,
}

impl Body {
    fn frame(&self) -> (Vec3, Vec3, Vec3) {
        let a = self.axis;
        let up = Vec3::new(0.0, 0.0, 1.0);
        let b = up.cross(&a).normalize();
        let c = a.cross(&b);
        (a, b, c)
    }
    /// Approximate signed distance (negative inside) and outward normal.
    fn sdf(&self, p: Vec3) -> (f64, Vec3) {
        let (a, b, c) = self.frame();
        let rel = p - self.centre;
        let l = Vec3::new(rel.dot(&a) / MELON_AXES[0], rel.dot(&b) / MELON_AXES[1], rel.dot(&c) / MELON_AXES[2]);
        let k0 = l.norm();
        let l2 = Vec3::new(l.x / MELON_AXES[0], l.y / MELON_AXES[1], l.z / MELON_AXES[2]);
        let k1 = l2.norm().max(1e-9);
        let d = k0 * (k0 - 1.0) / k1;
        let nl = Vec3::new(l.x / MELON_AXES[0], l.y / MELON_AXES[1], l.z / MELON_AXES[2]);
        let n = (a * nl.x + b * nl.y + c * nl.z).normalize();
        (d, n)
    }
}

impl Water {
    /// Fill the pool: one particle per cell, jittered, `depth` deep.
    pub fn fill(h: f64, dt: f64, air_above: f64, bulk: f64) -> Self {
        let origin = Vec3::new(-POOL_X - 2.0 * h, -POOL_Y - 2.0 * h, -DEPTH - 2.0 * h);
        let nx = ((2.0 * POOL_X + 4.0 * h) / h).ceil() as usize + 1;
        let ny = ((2.0 * POOL_Y + 4.0 * h) / h).ceil() as usize + 1;
        let nz = ((DEPTH + air_above + 4.0 * h) / h).ceil() as usize + 1;
        let mut x = Vec::new();
        let mut seed = 12345u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f64 / u32::MAX as f64) - 0.5
        };
        let mut px = -POOL_X + h * 0.5;
        while px < POOL_X - h * 0.25 {
            let mut py = -POOL_Y + h * 0.5;
            while py < POOL_Y - h * 0.25 {
                let mut pz = -DEPTH + h * 0.5;
                while pz < -h * 0.25 {
                    x.push(Vec3::new(px + rnd() * h * 0.6, py + rnd() * h * 0.6, pz + rnd() * h * 0.6));
                    pz += h;
                }
                py += h;
            }
            px += h;
        }
        let n = x.len();
        let vol0 = h * h * h;
        Self {
            h,
            dt,
            origin,
            nx,
            ny,
            nz,
            x,
            v: vec![Vec3::zeros(); n],
            c: vec![zero3(); n],
            j: vec![1.0; n],
            mass: 1000.0 * vol0,
            vol0,
            g_mass: vec![0.0; nx * ny * nz],
            g_mom: vec![Vec3::zeros(); nx * ny * nz],
            g_vel_old: vec![Vec3::zeros(); nx * ny * nz],
            flip: 0.9,
            bulk,
            time: 0.0,
            interior_mass: 0.0,
            level_offset: 0.0,
            order: Vec::new(),
        }
    }

    pub fn count(&self) -> usize {
        self.x.len()
    }

    #[inline]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * self.nx + i
    }

    #[inline]
    fn node_pos(&self, i: usize, j: usize, k: usize) -> Vec3 {
        self.origin + Vec3::new(i as f64, j as f64, k as f64) * self.h
    }

    /// Reorder the particles by the scatter group they were in last step, so
    /// the parallel scatter reads them sequentially. Once a frame is plenty;
    /// a particle moves well under a cell per substep.
    pub fn compact(&mut self) {
        if self.order.len() != self.x.len() {
            return;
        }
        let o = &self.order;
        let pick = |v: &Vec<Vec3>| o.par_iter().map(|&p| v[p as usize]).collect::<Vec<_>>();
        self.x = pick(&self.x);
        self.v = pick(&self.v);
        self.c = o.par_iter().map(|&p| self.c[p as usize]).collect();
        self.j = o.par_iter().map(|&p| self.j[p as usize]).collect();
        self.order.clear();
    }

    /// One substep. Returns the wrench the water put on the body (force only).
    pub fn step(&mut self, body: &Body) -> Vec3 {
        let h = self.h;
        let inv_h = 1.0 / h;
        let dt = self.dt;
        let n = self.x.len();
        let mut pt = std::time::Instant::now();
        self.g_mass.par_iter_mut().for_each(|m| *m = 0.0);
        self.g_mom.par_iter_mut().for_each(|m| *m = Vec3::zeros());
        prof(0, &mut pt);
        // ---- P2G (MLS-MPM: stress folded into the affine momentum) ----
        // Scatter in parallel by colouring the grid in 3x3 columns of cells:
        // a particle's base (i, j) puts it in group (i/3, j/3), whose 3x3x3
        // stencil touches i in [3gx, 3gx+5) and j in [3gy, 3gy+5), so groups
        // whose gx and gy both share parity never share a node. Four passes,
        // each parallel over its ~200 groups. (Colouring in k was tried first;
        // the water only fills the bottom rows, so few groups had work.)
        let d_inv = 4.0 * inv_h * inv_h; // quadratic B-spline
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let origin = self.origin;
        let (mass, vol0, bulk) = (self.mass, self.vol0, self.bulk);
        let slab = 3usize;
        let (ngx, ngy) = (nx / slab + 1, ny / slab + 1);
        let ngroups = ngx * ngy;
        let group = |xp: &Vec3| {
            let b = ((*xp - origin) * inv_h - Vec3::new(0.5, 0.5, 0.5)).map_floor();
            let gx = ((b.x.max(0.0) as usize) / slab).min(ngx - 1);
            let gy = ((b.y.max(0.0) as usize) / slab).min(ngy - 1);
            (gy * ngx + gx) as u32
        };
        let group_of: Vec<u32> = self.x.par_iter().map(group).collect();
        let mut counts = group_of
            .par_chunks(8192)
            .fold(
                || vec![0usize; ngroups + 1],
                |mut c, chunk| {
                    for &g in chunk {
                        c[g as usize + 1] += 1;
                    }
                    c
                },
            )
            .reduce(|| vec![0usize; ngroups + 1], |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                a
            });
        for g in 0..ngroups {
            counts[g + 1] += counts[g];
        }
        let mut fill = counts.clone();
        let mut order = vec![0u32; n];
        for (p, &g) in group_of.iter().enumerate() {
            order[fill[g as usize]] = p as u32;
            fill[g as usize] += 1;
        }
        self.order = order;
        prof(1, &mut pt);
        {
            let Water { x, v, c, j, g_mass, g_mom, order, .. } = &mut *self;
            let (x, v, c, j, order) = (&*x, &*v, &*c, &*j, &*order);
            #[derive(Clone, Copy)]
            struct Grid(*mut f64, *mut Vec3);
            unsafe impl Sync for Grid {}
            unsafe impl Send for Grid {}
            let grid = Grid(g_mass.as_mut_ptr(), g_mom.as_mut_ptr());
            let idx = |i: usize, jj: usize, k: usize| (k * ny + jj) * nx + i;
            for colour in 0..4 {
                let (cx, cy) = (colour % 2, colour / 2);
                (0..ngroups).into_par_iter().filter(|g| (g % ngx) % 2 == cx && (g / ngx) % 2 == cy).for_each(|g| {
                    let grid = grid;
                    for &p in &order[counts[g]..counts[g + 1]] {
                        let p = p as usize;
                        let xp = x[p];
                        let base = ((xp - origin) * inv_h - Vec3::new(0.5, 0.5, 0.5)).map_floor();
                        let fx = (xp - origin) * inv_h - base;
                        let w = weights(fx);
                        // pressure from the equation of state: p = K (1/J − 1), clamped so
                        // stretched (splashing) water does not pull
                        let jp = j[p];
                        let pressure = (bulk * (1.0 / jp - 1.0)).max(0.0);
                        let stress = Mat3::identity() * (-pressure);
                        let affine = stress * (-dt * vol0 * jp * d_inv) + c[p] * mass;
                        let mv = v[p] * mass;
                        let bi = base.x as i64;
                        let bj = base.y as i64;
                        let bk = base.z as i64;
                        for di in 0..3 {
                            for dj in 0..3 {
                                for dk in 0..3 {
                                    let (i, jj, k) = (bi + di as i64, bj + dj as i64, bk + dk as i64);
                                    if i < 0 || jj < 0 || k < 0 || i >= nx as i64 || jj >= ny as i64 || k >= nz as i64 {
                                        continue;
                                    }
                                    let dpos = (Vec3::new(di as f64, dj as f64, dk as f64) - fx) * h;
                                    let wt = w[0][di] * w[1][dj] * w[2][dk];
                                    let gi = idx(i as usize, jj as usize, k as usize);
                                    // SAFETY: groups of one colour touch disjoint (i, j) columns (see above),
                                    // and gi is in bounds by the check just made.
                                    unsafe {
                                        *grid.0.add(gi) += wt * mass;
                                        *grid.1.add(gi) += (mv + affine * dpos) * wt;
                                    }
                                }
                            }
                        }
                    }
                });
            }
        }
        prof(2, &mut pt);
        // ---- grid: gravity, walls, the body ----  (parallel over rows)
        let (lo, hi) = (origin + Vec3::new(2.0, 2.0, 2.0) * h, origin + Vec3::new((nx - 3) as f64, (ny - 3) as f64, (nz - 3) as f64) * h);
        let (reaction, interior) = self
            .g_mom
            .par_chunks_mut(nx)
            .zip(self.g_vel_old.par_chunks_mut(nx))
            .zip(self.g_mass.par_chunks(nx))
            .enumerate()
            .map(|(row, ((mom, vold), gm))| {
                let (k, jj) = (row / ny, row % ny);
                let mut reaction = Vec3::zeros();
                let mut interior = 0.0;
                {
                    for i in 0..nx {
                        let g = i;
                        let m = gm[g];
                        if m <= 0.0 {
                            continue;
                        }
                        let mut vel = mom[g] / m;
                        vold[g] = vel;
                        vel.z -= GRAVITY * dt;
                        let xi = origin + Vec3::new(i as f64, jj as f64, k as f64) * h;
                        // the pool: floor and walls, free-slip
                        if xi.x < lo.x && vel.x < 0.0 { vel.x = 0.0; }
                        if xi.x > hi.x && vel.x > 0.0 { vel.x = 0.0; }
                        if xi.y < lo.y && vel.y < 0.0 { vel.y = 0.0; }
                        if xi.y > hi.y && vel.y > 0.0 { vel.y = 0.0; }
                        if xi.z < lo.z && vel.z < 0.0 { vel.z = 0.0; }
                        if xi.z > hi.z && vel.z > 0.0 { vel.z = 0.0; }
                        // the melon: nodes within half a cell of its surface or inside
                        // take its normal velocity; what that costs is booked
                        let (d, nrm) = body.sdf(xi);
                        if d < 0.0 {
                            // inside the melon: no relative motion through the
                            // surface in either direction, so pressure from below
                            // *and* above reaches the body; the tangential part is
                            // left alone, since a sticky interior turned out to be
                            // a brake that held the melon at neutral depth
                            interior += m;
                            let rel = vel - body.vel;
                            let new = vel - nrm * rel.dot(&nrm);
                            reaction += (new - vel) * m;
                            vel = new;
                        } else if d < 0.5 * h {
                            // the shell: no approach through the surface, free slip along it
                            let rel = vel - body.vel;
                            let vn = rel.dot(&nrm);
                            if vn < 0.0 {
                                let new = vel - nrm * vn;
                                reaction += (new - vel) * m;
                                vel = new;
                            }
                        }
                        mom[g] = vel * m;
                    }
                }
                (reaction, interior)
            })
            .reduce(|| (Vec3::zeros(), 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        prof(3, &mut pt);
        // ---- G2P (APIC) ----  (parallel over particles, grid read-only)
        let flip = self.flip;
        let e = 1.5 * h;
        let (xmax, ymax, zmax) = (origin.x + (nx - 1) as f64 * h - e, origin.y + (ny - 1) as f64 * h - e, origin.z + (nz - 1) as f64 * h - e);
        {
            let Water { x, v, c, j, g_mass, g_mom, g_vel_old, .. } = &mut *self;
            let (g_mass, g_mom, g_vel_old) = (&*g_mass, &*g_mom, &*g_vel_old);
            x.par_iter_mut().zip(v.par_iter_mut()).zip(c.par_iter_mut()).zip(j.par_iter_mut()).for_each(|(((xp, vp), cp), jp)| {
                let base = ((*xp - origin) * inv_h - Vec3::new(0.5, 0.5, 0.5)).map_floor();
                let fx = (*xp - origin) * inv_h - base;
                let w = weights(fx);
                let mut vnew = Vec3::zeros();
                let mut dv = Vec3::zeros();
                let mut b = zero3();
                let bi = base.x as i64;
                let bj = base.y as i64;
                let bk = base.z as i64;
                for di in 0..3 {
                    for dj in 0..3 {
                        for dk in 0..3 {
                            let (i, jj, k) = (bi + di as i64, bj + dj as i64, bk + dk as i64);
                            if i < 0 || jj < 0 || k < 0 || i >= nx as i64 || jj >= ny as i64 || k >= nz as i64 {
                                continue;
                            }
                            let g = (k as usize * ny + jj as usize) * nx + i as usize;
                            let m = g_mass[g];
                            if m <= 0.0 {
                                continue;
                            }
                            let dpos = (Vec3::new(di as f64, dj as f64, dk as f64) - fx) * h;
                            let wt = w[0][di] * w[1][dj] * w[2][dk];
                            let gv = g_mom[g] / m;
                            vnew += gv * wt;
                            dv += (gv - g_vel_old[g]) * wt;
                            b = b + outer(gv * wt, dpos);
                        }
                    }
                }
                // FLIP keeps the particle's own velocity and adds the grid's change;
                // PIC takes the grid's velocity. The blend is the usual trade of
                // dissipation for noise.
                *vp = (*vp + dv) * flip + vnew * (1.0 - flip);
                *cp = b * d_inv;
                // volume from the trace of the velocity gradient
                *jp *= 1.0 + dt * (cp[(0, 0)] + cp[(1, 1)] + cp[(2, 2)]);
                *jp = jp.clamp(0.5, 2.0);
                *xp += vnew * dt;
                // keep particles in the box
                xp.x = xp.x.clamp(origin.x + e, xmax);
                xp.y = xp.y.clamp(origin.y + e, ymax);
                xp.z = xp.z.clamp(origin.z + e, zmax);
            });
        }
        prof(4, &mut pt);
        self.time += dt;
        self.interior_mass = interior;
        // the force on the body is minus what the fluid gained, per unit time
        -reaction / dt
    }

    /// Let the fill pack down under gravity with nothing in the pool, then
    /// take the extracted surface's mean as the rest level.
    pub fn settle(&mut self, seconds: f64) {
        let far = Body { centre: Vec3::new(0.0, 0.0, 50.0), axis: Vec3::new(1.0, 0.0, 0.0), vel: Vec3::zeros() };
        let n = (seconds / self.dt) as usize;
        for _ in 0..n {
            self.step(&far);
        }
        // damp the settling out
        for v in self.v.iter_mut() {
            *v = Vec3::zeros();
        }
        for c in self.c.iter_mut() {
            *c = zero3();
        }
        self.time = 0.0;
        self.level_offset = 0.0;
        let g = self.surface(0.02);
        let inner: Vec<f64> = (0..g.ny)
            .flat_map(|j| (0..g.nx).map(move |i| (i, j)))
            .filter(|(i, j)| *i > 2 && *j > 2 && *i + 3 < g.nx && *j + 3 < g.ny)
            .map(|(i, j)| g.z[j * g.nx + i])
            .collect();
        self.level_offset = inner.iter().sum::<f64>() / inner.len().max(1) as f64;
    }

    /// The free surface as a height field at `cell` resolution over the pool,
    /// read from the grid's mass: the first node from the top where the
    /// mass fraction crosses one half, interpolated. Columns with no water
    /// (a cavity) report the floor.
    pub fn surface(&self, cell: f64) -> HeightGrid {
        let nx = ((2.0 * POOL_X) / cell) as usize;
        let ny = ((2.0 * POOL_Y) / cell) as usize;
        let full = 1000.0 * self.h * self.h * self.h;
        // a 3×3×3 box blur of the mass fraction first: the grid is coarse and
        // a raw threshold makes cliffs
        let frac: Vec<f64> = self.g_mass.iter().map(|m| m / full).collect();
        let mut blurred = frac.clone();
        for k in 1..self.nz - 1 {
            for j in 1..self.ny - 1 {
                for i in 1..self.nx - 1 {
                    let mut acc = 0.0;
                    for dk in 0..3 {
                        for dj in 0..3 {
                            for di in 0..3 {
                                acc += frac[self.idx(i + di - 1, j + dj - 1, k + dk - 1)];
                            }
                        }
                    }
                    blurred[self.idx(i, j, k)] = acc / 27.0;
                }
            }
        }
        let mut z = vec![-DEPTH; nx * ny];
        for jy in 0..ny {
            for ix in 0..nx {
                let x = -POOL_X + (ix as f64 + 0.5) * cell;
                let y = -POOL_Y + (jy as f64 + 0.5) * cell;
                // nearest grid column
                let gi = (((x - self.origin.x) / self.h).round() as i64).clamp(0, self.nx as i64 - 1) as usize;
                let gj = (((y - self.origin.y) / self.h).round() as i64).clamp(0, self.ny as i64 - 1) as usize;
                let mut prev = 0.0;
                for k in (0..self.nz).rev() {
                    let f = blurred[self.idx(gi, gj, k)];
                    if f >= 0.5 {
                        // interpolate between this node and the one above
                        let t = if prev < 0.5 && k + 1 < self.nz { (0.5 - prev) / (f - prev).max(1e-9) } else { 0.0 };
                        z[jy * nx + ix] = self.node_pos(gi, gj, k).z + (1.0 - t) * self.h - self.level_offset;
                        break;
                    }
                    prev = f;
                }
            }
        }
        // a 3×3 smooth, twice
        let mut out = HeightGrid { origin: [-POOL_X, -POOL_Y], cell, nx, ny, z };
        for _ in 0..2 {
            let mut s = out.z.clone();
            for jy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let mut acc = 0.0;
                    for dy in 0..3 {
                        for dx in 0..3 {
                            acc += out.z[(jy + dy - 1) * nx + ix + dx - 1];
                        }
                    }
                    s[jy * nx + ix] = acc / 9.0;
                }
            }
            out.z = s;
        }
        out
    }

    /// Particles flying above the local surface: the drops, with their
    /// velocity and how much water is around them (0..1, from the grid mass
    /// at their node), which sets how big a bead to draw.
    pub fn droplets(&self, surface: &HeightGrid, above: f64, max: usize) -> Vec<Droplet> {
        let full = 1000.0 * self.h * self.h * self.h;
        let mut d: Vec<(f64, Droplet)> = self
            .x
            .iter()
            .zip(&self.v)
            .filter_map(|(p, v)| {
                // the surface grid is level-corrected for rendering; particle
                // positions are raw, so compare in raw terms
                let s = surface.at(p.x, p.y) + self.level_offset;
                if p.z <= s + above {
                    return None;
                }
                let gi = (((p.x - self.origin.x) / self.h).round() as i64).clamp(0, self.nx as i64 - 1) as usize;
                let gj = (((p.y - self.origin.y) / self.h).round() as i64).clamp(0, self.ny as i64 - 1) as usize;
                let gk = (((p.z - self.origin.z) / self.h).round() as i64).clamp(0, self.nz as i64 - 1) as usize;
                let crowd = (self.g_mass[self.idx(gi, gj, gk)] / full).min(1.0);
                Some((p.z - s, Droplet { pos: *p, vel: *v, crowd }))
            })
            .collect();
        d.sort_by(|a, b| b.0.total_cmp(&a.0));
        d.truncate(max);
        d.into_iter().map(|(_, p)| p).collect()
    }
}

#[derive(Clone, Copy)]
pub struct Droplet {
    pub pos: Vec3,
    pub vel: Vec3,
    /// How much water shares the drop's cell, 0..1.
    pub crowd: f64,
}

#[derive(Clone)]
pub struct HeightGrid {
    pub origin: [f64; 2],
    pub cell: f64,
    pub nx: usize,
    pub ny: usize,
    pub z: Vec<f64>,
}

impl HeightGrid {
    pub fn at(&self, x: f64, y: f64) -> f64 {
        let gx = ((x - self.origin[0]) / self.cell - 0.5).clamp(0.0, (self.nx - 1) as f64 - 1e-6);
        let gy = ((y - self.origin[1]) / self.cell - 0.5).clamp(0.0, (self.ny - 1) as f64 - 1e-6);
        let (ix, iy) = (gx.floor() as usize, gy.floor() as usize);
        let (wx, wy) = (gx - ix as f64, gy - iy as f64);
        let z = &self.z;
        z[iy * self.nx + ix] * (1.0 - wx) * (1.0 - wy)
            + z[iy * self.nx + ix + 1] * wx * (1.0 - wy)
            + z[(iy + 1) * self.nx + ix] * (1.0 - wx) * wy
            + z[(iy + 1) * self.nx + ix + 1] * wx * wy
    }
}

fn weights(fx: Vec3) -> [[f64; 3]; 3] {
    let w1 = |f: f64| [0.5 * (1.5 - f).powi(2), 0.75 - (f - 1.0).powi(2), 0.5 * (f - 0.5).powi(2)];
    [w1(fx.x), w1(fx.y), w1(fx.z)]
}

fn zero3() -> Mat3 {
    Mat3::identity() * 0.0
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::new(a.x * b.x, a.x * b.y, a.x * b.z, a.y * b.x, a.y * b.y, a.y * b.z, a.z * b.x, a.z * b.y, a.z * b.z)
}

trait MapFloor {
    fn map_floor(self) -> Vec3;
}
impl MapFloor for Vec3 {
    fn map_floor(self) -> Vec3 {
        Vec3::new(self.x.floor(), self.y.floor(), self.z.floor())
    }
}
