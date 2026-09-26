//! `.qcdf` field files from phyz-qft, and the meshes the viewer builds from
//! them: a height-field slice and ± isosurfaces.
//!
//! Lattice fields are coarse (a 16³ slice is 16 samples a side), so both
//! meshes resample the periodic lattice with Catmull-Rom cubics first. That is
//! what turns blocky voxels into the smooth lumps the lattice papers show.

use anyhow::{Context, bail};
use std::path::Path;

#[derive(Clone)]
pub struct FieldFile {
    /// `[nt, nx, ny, nz]`.
    pub dims: [usize; 4],
    pub a_fm: f64,
    pub names: Vec<String>,
    /// One array per name, site order `t + nt (x + nx (y + ny z))`.
    pub data: Vec<Vec<f32>>,
    /// Static quark positions (lattice units, x y z).
    pub quarks: Vec<[f32; 3]>,
}

impl FieldFile {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let nl = bytes
            .iter()
            .position(|&b| b == b'\n')
            .context("no header line")?;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes[..nl]).context("header json")?;
        let dims: Vec<usize> = serde_json::from_value(header["dims"].clone()).context("dims")?;
        let dims: [usize; 4] = dims.try_into().ok().context("dims must have 4 entries")?;
        let names: Vec<String> =
            serde_json::from_value(header["fields"].clone()).context("fields")?;
        let quarks: Vec<[f32; 3]> =
            serde_json::from_value(header["quarks"].clone()).unwrap_or_default();
        let a_fm = header["a_fm"].as_f64().unwrap_or(0.1);

        let n: usize = dims.iter().product();
        let body = &bytes[nl + 1..];
        if body.len() != names.len() * n * 4 {
            bail!(
                "expected {} bytes of field data, found {}",
                names.len() * n * 4,
                body.len()
            );
        }
        let data = body
            .chunks_exact(n * 4)
            .map(|c| {
                c.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect()
            })
            .collect();
        Ok(Self {
            dims,
            a_fm,
            names,
            data,
            quarks,
        })
    }

    /// A single-site zero field, before any data arrives.
    pub fn empty() -> Self {
        Self {
            dims: [1; 4],
            a_fm: 0.1,
            names: vec!["-".into()],
            data: vec![vec![0.0]],
            quarks: vec![],
        }
    }

    /// The 3D spatial field at Euclidean time `t`, indexed `x + nx (y + ny z)`.
    pub fn time_slice(&self, field: usize, t: usize) -> Grid3 {
        let [nt, nx, ny, nz] = self.dims;
        let d = &self.data[field];
        let mut v = Vec::with_capacity(nx * ny * nz);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    v.push(d[t + nt * (x + nx * (y + ny * z))]);
                }
            }
        }
        Grid3 { n: [nx, ny, nz], v }
    }

    /// Mean and standard deviation of a whole field, for normalizing.
    pub fn stats(&self, field: usize) -> (f32, f32) {
        let d = &self.data[field];
        let n = d.len() as f64;
        let mean = d.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = d.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        (mean as f32, var.sqrt().max(1e-12) as f32)
    }
}

/// A periodic 3D scalar grid.
pub struct Grid3 {
    pub n: [usize; 3],
    pub v: Vec<f32>,
}

fn catmull_rom(p: [f32; 4], t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p[1])
        + (-p[0] + p[2]) * t
        + (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3]) * t2
        + (-p[0] + 3.0 * p[1] - 3.0 * p[2] + p[3]) * t3)
}

impl Grid3 {
    #[inline]
    pub fn at(&self, x: isize, y: isize, z: isize) -> f32 {
        let w = |i: isize, n: usize| i.rem_euclid(n as isize) as usize;
        let [nx, ny, nz] = self.n;
        self.v[w(x, nx) + nx * (w(y, ny) + ny * w(z, nz))]
    }

    /// Periodic tricubic (Catmull-Rom) sample at lattice coordinates.
    pub fn sample(&self, p: [f32; 3]) -> f32 {
        let i = p.map(|c| c.floor() as isize);
        let f = [p[0] - i[0] as f32, p[1] - i[1] as f32, p[2] - i[2] as f32];
        let mut zs = [0.0; 4];
        for (dz, zv) in zs.iter_mut().enumerate() {
            let mut ys = [0.0; 4];
            for (dy, yv) in ys.iter_mut().enumerate() {
                let xs = std::array::from_fn(|dx| {
                    self.at(
                        i[0] + dx as isize - 1,
                        i[1] + dy as isize - 1,
                        i[2] + dz as isize - 1,
                    )
                });
                *yv = catmull_rom(xs, f[0]);
            }
            *zv = catmull_rom(ys, f[1]);
        }
        catmull_rom(zs, f[2])
    }

    /// Resample `k` points per lattice spacing (one period, endpoints shared).
    pub fn upsample(&self, k: usize) -> Grid3 {
        let n = self.n.map(|c| c * k + 1);
        let mut v = Vec::with_capacity(n[0] * n[1] * n[2]);
        let s = 1.0 / k as f32;
        for z in 0..n[2] {
            for y in 0..n[1] {
                for x in 0..n[0] {
                    v.push(self.sample([x as f32 * s, y as f32 * s, z as f32 * s]));
                }
            }
        }
        Grid3 { n, v }
    }

    #[inline]
    fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        self.v[x + self.n[0] * (y + self.n[1] * z)]
    }

    fn grad(&self, x: usize, y: usize, z: usize) -> [f32; 3] {
        let gx = {
            let (lo, hi) = (x.saturating_sub(1), (x + 1).min(self.n[0] - 1));
            (self.get(hi, y, z) - self.get(lo, y, z)) / (hi - lo).max(1) as f32
        };
        let gy = {
            let (lo, hi) = (y.saturating_sub(1), (y + 1).min(self.n[1] - 1));
            (self.get(x, hi, z) - self.get(x, lo, z)) / (hi - lo).max(1) as f32
        };
        let gz = {
            let (lo, hi) = (z.saturating_sub(1), (z + 1).min(self.n[2] - 1));
            (self.get(x, y, hi) - self.get(x, y, lo)) / (hi - lo).max(1) as f32
        };
        [gx, gy, gz]
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    pub col: [f32; 4],
}

pub struct Mesh {
    pub verts: Vec<Vertex>,
    pub idx: Vec<u32>,
}

/// Map lattice coordinates (0..n) to world space: the box spans [−1, 1] along
/// its longest spatial side, centred at the origin.
pub fn world_scale(n: [usize; 3]) -> f32 {
    2.0 / *n.iter().max().unwrap() as f32
}

/// Turbo colormap (Mikhailov 2019, polynomial fit), `t` in [0, 1].
#[allow(clippy::excessive_precision)]
pub fn turbo(t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let r = 0.13572138
        + t * (4.61539260
            + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
    let g = 0.09140261
        + t * (2.19418839
            + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
    let b = 0.10667330
        + t * (12.64194608
            + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
    [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
}

/// Height field over the (x, y) plane at lattice height `z` (fractional ok).
/// `norm` maps a field value to [0, 1] for colour and to height via `lift`.
pub fn height_slice(g: &Grid3, z: f32, k: usize, lift: f32, norm: impl Fn(f32) -> f32) -> Mesh {
    let [nx, ny, _] = g.n;
    let s = world_scale(g.n);
    let (mx, my) = (nx * k + 1, ny * k + 1);
    let step = 1.0 / k as f32;
    let vals: Vec<f32> = (0..my)
        .flat_map(|j| (0..mx).map(move |i| (i, j)))
        .map(|(i, j)| norm(g.sample([i as f32 * step, j as f32 * step, z])))
        .collect();
    let h = |i: usize, j: usize| vals[j * mx + i] * lift;
    let ox = -0.5 * nx as f32 * s;
    let oy = -0.5 * ny as f32 * s;
    let mut verts = Vec::with_capacity(mx * my);
    for j in 0..my {
        for i in 0..mx {
            let (i0, i1) = (i.saturating_sub(1), (i + 1).min(mx - 1));
            let (j0, j1) = (j.saturating_sub(1), (j + 1).min(my - 1));
            let dx = (h(i1, j) - h(i0, j)) / ((i1 - i0) as f32 * step * s);
            let dy = (h(i, j1) - h(i, j0)) / ((j1 - j0) as f32 * step * s);
            let nl = (dx * dx + dy * dy + 1.0).sqrt();
            let c = turbo(0.08 + 0.87 * vals[j * mx + i]);
            verts.push(Vertex {
                pos: [
                    ox + i as f32 * step * s,
                    oy + j as f32 * step * s,
                    h(i, j) - 0.5 * lift,
                ],
                nrm: [-dx / nl, -dy / nl, 1.0 / nl],
                col: [c[0], c[1], c[2], 1.0],
            });
        }
    }
    let mut idx = Vec::with_capacity((mx - 1) * (my - 1) * 6);
    for j in 0..my - 1 {
        for i in 0..mx - 1 {
            let a = (j * mx + i) as u32;
            let b = a + 1;
            let c = a + mx as u32;
            let d = c + 1;
            idx.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
    Mesh { verts, idx }
}

/// Cube corners and the six tetrahedra sharing the 0–6 diagonal.
const CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];
const TETS: [[usize; 4]; 6] = [
    [0, 5, 1, 6],
    [0, 1, 2, 6],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 7, 4, 6],
    [0, 4, 5, 6],
];

/// Marching tetrahedra: the `iso` level set of an (already upsampled) grid,
/// where `k` is its samples per lattice spacing. Normals point down the
/// gradient, out of the region where the field exceeds `iso`.
pub fn isosurface(g: &Grid3, k: usize, iso: f32, color: [f32; 4], out: &mut Mesh) {
    let [mx, my, mz] = g.n;
    // world transform: fine index → lattice units → world
    let lat = g.n.map(|c| (c - 1) / k);
    let s = world_scale(lat) / k as f32;
    let o = lat.map(|c| -0.5 * c as f32 * world_scale(lat));
    let world = |p: [f32; 3]| [o[0] + p[0] * s, o[1] + p[1] * s, o[2] + p[2] * s];

    for z in 0..mz - 1 {
        for y in 0..my - 1 {
            for x in 0..mx - 1 {
                let c: [[usize; 3]; 8] = CORNERS.map(|d| [x + d[0], y + d[1], z + d[2]]);
                let v: [f32; 8] = c.map(|p| g.get(p[0], p[1], p[2]));
                if v.iter().all(|&a| a < iso) || v.iter().all(|&a| a >= iso) {
                    continue;
                }
                for tet in TETS {
                    let inside: Vec<usize> = tet.iter().copied().filter(|&i| v[i] >= iso).collect();
                    let outside: Vec<usize> = tet.iter().copied().filter(|&i| v[i] < iso).collect();
                    let mut edge = |a: usize, b: usize| -> u32 {
                        let t = ((iso - v[a]) / (v[b] - v[a])).clamp(0.0, 1.0);
                        let p = std::array::from_fn(|i| {
                            c[a][i] as f32 + t * (c[b][i] as f32 - c[a][i] as f32)
                        });
                        let (ga, gb) = (
                            g.grad(c[a][0], c[a][1], c[a][2]),
                            g.grad(c[b][0], c[b][1], c[b][2]),
                        );
                        let gr: [f32; 3] = std::array::from_fn(|i| ga[i] + t * (gb[i] - ga[i]));
                        let l = (gr[0] * gr[0] + gr[1] * gr[1] + gr[2] * gr[2])
                            .sqrt()
                            .max(1e-12);
                        out.verts.push(Vertex {
                            pos: world(p),
                            nrm: gr.map(|q| -q / l),
                            col: color,
                        });
                        (out.verts.len() - 1) as u32
                    };
                    match (inside.len(), outside.len()) {
                        (1, 3) | (3, 1) => {
                            let (lone, rest) = if inside.len() == 1 {
                                (inside[0], &outside)
                            } else {
                                (outside[0], &inside)
                            };
                            let a = edge(lone, rest[0]);
                            let b = edge(lone, rest[1]);
                            let cc = edge(lone, rest[2]);
                            out.idx.extend_from_slice(&[a, b, cc]);
                        }
                        (2, 2) => {
                            let a = edge(inside[0], outside[0]);
                            let b = edge(inside[0], outside[1]);
                            let cc = edge(inside[1], outside[1]);
                            let d = edge(inside[1], outside[0]);
                            out.idx.extend_from_slice(&[a, b, cc, a, cc, d]);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

/// Unit sphere, for quark markers.
pub fn unit_sphere(seg: usize) -> Mesh {
    let mut verts = Vec::new();
    let mut idx = Vec::new();
    for j in 0..=seg {
        let th = std::f32::consts::PI * j as f32 / seg as f32;
        for i in 0..=2 * seg {
            let ph = std::f32::consts::PI * i as f32 / seg as f32;
            let n = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
            verts.push(Vertex {
                pos: n,
                nrm: n,
                col: [1.0; 4],
            });
        }
    }
    let row = 2 * seg as u32 + 1;
    for j in 0..seg as u32 {
        for i in 0..2 * seg as u32 {
            let a = j * row + i;
            idx.extend_from_slice(&[a, a + row, a + 1, a + 1, a + row, a + row + 1]);
        }
    }
    Mesh { verts, idx }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsample_interpolates_lattice_points() {
        let g = Grid3 {
            n: [4, 4, 4],
            v: (0..64).map(|i| (i as f32 * 0.37).sin()).collect(),
        };
        let u = g.upsample(3);
        for (x, y, z) in [(0, 0, 0), (1, 2, 3), (3, 3, 3)] {
            assert!(
                (u.get(3 * x, 3 * y, 3 * z) - g.at(x as isize, y as isize, z as isize)).abs()
                    < 1e-5
            );
        }
    }

    #[test]
    fn sphere_isosurface_is_closed_and_round() {
        // f = r² about the centre of an 8³ box; iso = 4 is a radius-2 sphere.
        let n = 8;
        let v = (0..n * n * n)
            .map(|i| {
                let (x, y, z) = (
                    (i % n) as f32 - 4.0,
                    ((i / n) % n) as f32 - 4.0,
                    (i / (n * n)) as f32 - 4.0,
                );
                x * x + y * y + z * z
            })
            .collect();
        let g = Grid3 { n: [n, n, n], v };
        let fine = g.upsample(2);
        let mut m = Mesh {
            verts: vec![],
            idx: vec![],
        };
        isosurface(&fine, 2, 4.0, [1.0; 4], &mut m);
        assert!(!m.idx.is_empty());
        // every vertex sits at radius 2 lattice units = 2 · world_scale
        let r_world = 2.0 * world_scale([n, n, n]);
        let c = -0.5 * n as f32 * world_scale([n, n, n]) + 4.0 * world_scale([n, n, n]);
        for vtx in &m.verts {
            let r = vtx
                .pos
                .iter()
                .map(|p| (p - c) * (p - c))
                .sum::<f32>()
                .sqrt();
            assert!(
                (r - r_world).abs() < 0.08 * r_world,
                "r = {r}, want {r_world}"
            );
        }
    }

    #[test]
    fn turbo_endpoints() {
        let lo = turbo(0.1);
        let hi = turbo(1.0);
        assert!(lo[2] > lo[0], "low end is blue");
        assert!(hi[0] > hi[2], "high end is red");
    }
}
