//! Sunlight refracted through the surface onto the floor.

use rayon::prelude::*;
use tang::Vec3 as V;

use kosm_render::optics::{fresnel, refract};

use super::splash::HeightGrid;
use super::surface::Surface;
use super::{BLEND, N_WATER, POOL_X, POOL_Y, PoolGeometry, SPONGE, box_half, sun_dir};

// ---- rendering --------------------------------------------------------------

#[derive(Clone)]
pub struct Caustic {
    pub origin: [f64; 2],
    pub cell: f64,
    pub nx: usize,
    pub ny: usize,
    pub e: Vec<f64>,
}

impl Caustic {
    pub fn at(&self, x: f64, y: f64) -> f64 {
        let gx = (x - self.origin[0]) / self.cell - 0.5;
        let gy = (y - self.origin[1]) / self.cell - 0.5;
        if gx < 0.0 || gy < 0.0 || gx >= (self.nx - 1) as f64 || gy >= (self.ny - 1) as f64 {
            return 1.0;
        }
        let (ix, iy) = (gx.floor() as usize, gy.floor() as usize);
        let (wx, wy) = (gx - ix as f64, gy - iy as f64);
        let e = &self.e;
        e[iy * self.nx + ix] * (1.0 - wx) * (1.0 - wy)
            + e[iy * self.nx + ix + 1] * wx * (1.0 - wy)
            + e[(iy + 1) * self.nx + ix] * (1.0 - wx) * wy
            + e[(iy + 1) * self.nx + ix + 1] * wx * wy
    }
}

/// Sunlight through the surface onto the floor: irradiance relative to what a
/// flat surface would pass, so still water reads as 1 and ripples focus it.
pub fn caustic(surface: &Surface, cell: f64) -> Caustic {
    caustic_for_geometry(surface, PoolGeometry::reference(), cell)
}

pub fn caustic_for_geometry(
    surface: &Surface,
    geometry: PoolGeometry,
    cell: f64,
) -> Caustic {
    // over the box and a margin: the sun's refracted rays land a metre
    // sideways over two metres of depth, and beyond the map the floor reads 1
    let half = box_half() + 3.0;
    let nx = ((2.0 * half) / cell) as usize;
    let ny = ((2.0 * half) / cell) as usize;
    let origin = [-half, -half];
    // launch from beyond the map too: a cell near the map's edge is lit by
    // rays from both sides, or the edge shows as a frame
    let margin = 1.5;
    let (lx, ly) = (((2.0 * half + 2.0 * margin) / cell) as usize, ((2.0 * half + 2.0 * margin) / cell) as usize);
    let l_origin = [-half - margin, -half - margin];
    let mut e = vec![0.0; nx * ny];
    let s = sun_dir();
    let d = -s;
    // reference: a flat surface refracts the sun to a fixed direction with a
    // fixed transmission; each ray deposits relative to that
    let n_flat = V::new(0.0, 0.0, 1.0);
    let (d_flat, ci, ct) = refract(d, n_flat, 1.0, N_WATER).expect("sun above the horizon");
    let t_flat = 1.0 - fresnel(1.0, N_WATER, ci, ct);
    let flat_cos = -d_flat.z;
    let sub = 3; // rays per cell per axis
    let per_ray = 1.0 / (sub * sub) as f64;
    // one row of launch points per task, each with its own accumulator
    let e = (0..ly * sub)
        .into_par_iter()
        .fold(
            || vec![0.0; nx * ny],
            |mut e, iy| {
                for ix in 0..lx * sub {
                    // launch from the surface point that the flat refraction would
                    // send to this floor cell, so the reference is uniform
                    let fx = l_origin[0] + (ix as f64 + 0.5) * cell / sub as f64;
                    let fy = l_origin[1] + (iy as f64 + 0.5) * cell / sub as f64;
                    let back = geometry.depth / flat_cos;
                    let sx = fx - d_flat.x * back;
                    let sy = fy - d_flat.y * back;
                    let n = surface.normal(sx, sy);
                    let Some((dr, ci, ct)) = refract(d, n, 1.0, N_WATER) else { continue };
                    let tr = 1.0 - fresnel(1.0, N_WATER, ci, ct);
                    let z0 = surface.height(sx, sy);
                    let tt = (z0 + geometry.depth) / -dr.z;
                    let hx = sx + dr.x * tt;
                    let hy = sy + dr.y * tt;
                    let w = per_ray * (tr / t_flat) * (-dr.z / flat_cos);
                    let gx = (hx - origin[0]) / cell - 0.5;
                    let gy = (hy - origin[1]) / cell - 0.5;
                    let (bx, by) = (gx.floor(), gy.floor());
                    let (wx, wy) = (gx - bx, gy - by);
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (jx, jy) = (bx as i64 + dx, by as i64 + dy);
                        if jx < 0 || jy < 0 || jx >= nx as i64 || jy >= ny as i64 {
                            continue;
                        }
                        let ww = (if dx == 0 { 1.0 - wx } else { wx }) * (if dy == 0 { 1.0 - wy } else { wy });
                        e[jy as usize * nx + jx as usize] += w * ww;
                    }
                }
                e
            },
        )
        .reduce(
            || vec![0.0; nx * ny],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                a
            },
        );
    Caustic { origin, cell, nx, ny, e }
}

/// The same caustic, traced on the GPU: three million rays, the composed
/// surface evaluated in WGSL, deposited through fixed-point atomics. The CPU
/// version above stays the reference; `KOSM_GPU_RENDER=0` selects it.
pub fn caustic_gpu(gpu: &mut kosm_mpm::GpuCaustic, surface: &Surface, cell: f64) -> Option<Caustic> {
    caustic_gpu_for_geometry(gpu, surface, PoolGeometry::reference(), cell)
}

pub fn caustic_gpu_for_geometry(
    gpu: &mut kosm_mpm::GpuCaustic,
    surface: &Surface,
    geometry: PoolGeometry,
    cell: f64,
) -> Option<Caustic> {
    let g = surface.grid.as_ref()?;
    let half = box_half() + 3.0;
    let fz: Vec<f32> = g.z.iter().map(|z| *z as f32).collect();
    // no far field yet (the first frames, or the ring model): a flat pool
    let zero = HeightGrid { origin: [-POOL_X, -POOL_Y], cell: POOL_X, nx: 2, ny: 2, z: vec![0.0; 4] };
    let f = surface.far.as_ref().unwrap_or(&zero);
    let rz: Vec<f32> = f.z.iter().map(|z| *z as f32).collect();
    let d = -sun_dir();
    let cfg = kosm_mpm::CausticCfg {
        cell: cell as f32,
        half: half as f32,
        margin: 1.5,
        sub: 3,
        depth: geometry.depth as f32,
        n_water: N_WATER as f32,
        dir: [d.x as f32, d.y as f32, d.z as f32],
        t: surface.t as f32,
        box_half: box_half() as f32,
        sponge: SPONGE as f32,
        blend: BLEND as f32,
    };
    let fine = kosm_mpm::Grid { origin: [g.origin[0] as f32, g.origin[1] as f32], cell: g.cell as f32, nx: g.nx as u32, ny: g.ny as u32, z: &fz };
    let far = kosm_mpm::Grid { origin: [f.origin[0] as f32, f.origin[1] as f32], cell: f.cell as f32, nx: f.nx as u32, ny: f.ny as u32, z: &rz };
    let (nx, ny, e) = gpu.trace(&cfg, &fine, &far);
    Some(Caustic { origin: [-half, -half], cell, nx, ny, e: e.into_iter().map(|v| v as f64).collect() })
}

