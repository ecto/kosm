//! The baked signed-distance grid — the physics layer's runtime form.
//!
//! Dense f32 samples on a regular node-centred grid, baked once from the
//! collision mesh by exact closest-triangle queries and read millions of
//! times per rollout by trilinear interpolation. Dense rather than
//! narrow-band because a lab at 2 cm cells is tens of megabytes, and "the
//! whole grid is always resident and every lookup is three multiplies and a
//! fetch" is worth more here than the memory back.
//!
//! # `sdf.bin` — ISDF version 1, little-endian
//!
//! ```text
//! offset  size  field
//! 0       4     magic "ISDF"
//! 4       4     u32 version = 1
//! 8       12    u32 nx, ny, nz         sample counts per axis (nodes)
//! 20      24    f64 origin x, y, z     world position of sample (0,0,0)
//! 44      8     f64 cell               sample spacing, metres
//! 52      4*n   f32 data               x-fastest: i = x + nx*(y + ny*z)
//! ```
//!
//! Sample `(i,j,k)` sits at `origin + (i,j,k)*cell`. Values are metres,
//! positive outside the scanned surface.
//!
//! # Outside the grid
//!
//! [`SdfGrid::sample`] returns `None` outside the sampled volume and the
//! contact producer skips the candidate — beyond the map there is no floor.
//! Deliberately *not* clamped-edge extension: extending border distances
//! flat would fabricate an infinite zero-gradient apron whose normals are
//! garbage. If the robot walks off the scan, it falls, which is the honest
//! reading of "the map ends here".

use std::io::{Read, Write};
use std::path::Path;

use phyz_math::Vec3;
use rayon::prelude::*;

use crate::MapError;
use crate::mesh::TriMesh;

/// A dense signed-distance grid in the map frame (z-up, metres).
#[derive(Clone)]
pub struct SdfGrid {
    /// World position of sample `(0,0,0)`.
    pub origin: Vec3,
    /// Sample spacing, metres.
    pub cell: f64,
    /// Sample counts per axis.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// x-fastest: `data[x + nx*(y + ny*z)]`.
    pub data: Vec<f32>,
}

impl std::fmt::Debug for SdfGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SdfGrid")
            .field("origin", &self.origin)
            .field("cell", &self.cell)
            .field("dims", &(self.nx, self.ny, self.nz))
            .finish()
    }
}

const MAGIC: &[u8; 4] = b"ISDF";
const VERSION: u32 = 1;

impl SdfGrid {
    /// Bake a grid from a mesh: the mesh AABB padded by `pad` on every side,
    /// sampled every `cell` metres, exact signed distance at every node.
    ///
    /// Rayon-parallel over z-slabs; a 10 m lab at 3 cm cells bakes in
    /// seconds, not minutes, because every query goes through the mesh BVH.
    pub fn bake(mesh: &TriMesh, cell: f64, pad: f64) -> SdfGrid {
        assert!(cell > 0.0 && cell.is_finite(), "cell must be positive");
        let (lo, hi) = mesh.aabb();
        let origin = lo - Vec3::splat(pad);
        let extent = hi + Vec3::splat(pad) - origin;
        let nx = (extent.x / cell).ceil() as usize + 1;
        let ny = (extent.y / cell).ceil() as usize + 1;
        let nz = (extent.z / cell).ceil() as usize + 1;

        let mut data = vec![0.0f32; nx * ny * nz];
        data.par_chunks_mut(nx * ny).enumerate().for_each(|(k, slab)| {
            let z = origin.z + k as f64 * cell;
            for j in 0..ny {
                let y = origin.y + j as f64 * cell;
                for (i, out) in slab[j * nx..(j + 1) * nx].iter_mut().enumerate() {
                    let p = Vec3::new(origin.x + i as f64 * cell, y, z);
                    *out = mesh.signed_distance(p) as f32;
                }
            }
        });

        SdfGrid { origin, cell, nx, ny, nz, data }
    }

    /// Signed distance at `p` by trilinear interpolation, or `None` outside
    /// the sampled volume.
    #[inline]
    pub fn sample(&self, p: Vec3) -> Option<f64> {
        let (c, f) = self.locate(p)?;
        Some(self.trilinear(c, f))
    }

    /// Signed distance and its gradient at `p`, or `None` outside the volume.
    ///
    /// The gradient is the analytic derivative of the same trilinear
    /// interpolant `sample` reads — not a finite difference of neighbouring
    /// samples — so the pair is exactly consistent: contact depth and contact
    /// normal always describe the same local surface.
    #[inline]
    pub fn sample_with_gradient(&self, p: Vec3) -> Option<(f64, Vec3)> {
        let (c, f) = self.locate(p)?;
        let v = self.corners(c);
        let (fx, fy, fz) = (f.x, f.y, f.z);

        // Interpolate along x for the four y-z corner pairs.
        let c00 = v[0] + (v[1] - v[0]) * fx;
        let c10 = v[2] + (v[3] - v[2]) * fx;
        let c01 = v[4] + (v[5] - v[4]) * fx;
        let c11 = v[6] + (v[7] - v[6]) * fx;
        let c0 = c00 + (c10 - c00) * fy;
        let c1 = c01 + (c11 - c01) * fy;
        let value = c0 + (c1 - c0) * fz;

        let inv = 1.0 / self.cell;
        let dx = ((v[1] - v[0]) * (1.0 - fy) + (v[3] - v[2]) * fy) * (1.0 - fz) * inv
            + ((v[5] - v[4]) * (1.0 - fy) + (v[7] - v[6]) * fy) * fz * inv;
        let dy = (c10 - c00) * (1.0 - fz) * inv + (c11 - c01) * fz * inv;
        let dz = (c1 - c0) * inv;

        Some((value, Vec3::new(dx, dy, dz)))
    }

    /// Cell index and fractional position of `p`, or `None` when the eight
    /// surrounding samples are not all inside the grid.
    #[inline]
    fn locate(&self, p: Vec3) -> Option<([usize; 3], Vec3)> {
        let g = (p - self.origin) * (1.0 / self.cell);
        if !(g.x.is_finite() && g.y.is_finite() && g.z.is_finite()) {
            return None;
        }
        if g.x < 0.0 || g.y < 0.0 || g.z < 0.0 {
            return None;
        }
        let (ix, iy, iz) = (g.x.floor() as usize, g.y.floor() as usize, g.z.floor() as usize);
        if ix + 1 >= self.nx || iy + 1 >= self.ny || iz + 1 >= self.nz {
            return None;
        }
        Some(([ix, iy, iz], Vec3::new(g.x - ix as f64, g.y - iy as f64, g.z - iz as f64)))
    }

    /// The eight corner samples of cell `c`, in x-fastest order:
    /// `[000, 100, 010, 110, 001, 101, 011, 111]`.
    #[inline]
    fn corners(&self, c: [usize; 3]) -> [f64; 8] {
        let at = |dx: usize, dy: usize, dz: usize| {
            self.data[(c[0] + dx) + self.nx * ((c[1] + dy) + self.ny * (c[2] + dz))] as f64
        };
        [
            at(0, 0, 0),
            at(1, 0, 0),
            at(0, 1, 0),
            at(1, 1, 0),
            at(0, 0, 1),
            at(1, 0, 1),
            at(0, 1, 1),
            at(1, 1, 1),
        ]
    }

    #[inline]
    fn trilinear(&self, c: [usize; 3], f: Vec3) -> f64 {
        let v = self.corners(c);
        let c00 = v[0] + (v[1] - v[0]) * f.x;
        let c10 = v[2] + (v[3] - v[2]) * f.x;
        let c01 = v[4] + (v[5] - v[4]) * f.x;
        let c11 = v[6] + (v[7] - v[6]) * f.x;
        let c0 = c00 + (c10 - c00) * f.y;
        let c1 = c01 + (c11 - c01) * f.y;
        c0 + (c1 - c0) * f.z
    }

    /// Write ISDF v1.
    pub fn save(&self, path: &Path) -> Result<(), MapError> {
        let io = |e| MapError::Io(path.to_path_buf(), e);
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(io)?);
        f.write_all(MAGIC).map_err(io)?;
        f.write_all(&VERSION.to_le_bytes()).map_err(io)?;
        for n in [self.nx, self.ny, self.nz] {
            f.write_all(&(n as u32).to_le_bytes()).map_err(io)?;
        }
        for v in [self.origin.x, self.origin.y, self.origin.z, self.cell] {
            f.write_all(&v.to_le_bytes()).map_err(io)?;
        }
        for v in &self.data {
            f.write_all(&v.to_le_bytes()).map_err(io)?;
        }
        f.flush().map_err(io)
    }

    /// Read ISDF v1.
    pub fn load(path: &Path) -> Result<SdfGrid, MapError> {
        let io = |e| MapError::Io(path.to_path_buf(), e);
        let bad = |msg: String| MapError::Format(path.to_path_buf(), msg);
        let mut f = std::io::BufReader::new(std::fs::File::open(path).map_err(io)?);

        let mut head = [0u8; 52];
        f.read_exact(&mut head).map_err(io)?;
        if &head[0..4] != MAGIC {
            return Err(bad("not an ISDF file".into()));
        }
        let version = u32::from_le_bytes(head[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(bad(format!("ISDF version {version}, expected {VERSION}")));
        }
        let nx = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        let ny = u32::from_le_bytes(head[12..16].try_into().unwrap()) as usize;
        let nz = u32::from_le_bytes(head[16..20].try_into().unwrap()) as usize;
        let origin = Vec3::new(
            f64::from_le_bytes(head[20..28].try_into().unwrap()),
            f64::from_le_bytes(head[28..36].try_into().unwrap()),
            f64::from_le_bytes(head[36..44].try_into().unwrap()),
        );
        let cell = f64::from_le_bytes(head[44..52].try_into().unwrap());
        if !(cell > 0.0 && cell.is_finite()) {
            return Err(bad(format!("cell {cell} is not a positive length")));
        }
        let count = nx
            .checked_mul(ny)
            .and_then(|v| v.checked_mul(nz))
            .ok_or_else(|| bad(format!("dims {nx}x{ny}x{nz} overflow")))?;

        let mut bytes = vec![0u8; count * 4];
        f.read_exact(&mut bytes).map_err(io)?;
        let data = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok(SdfGrid { origin, cell, nx, ny, nz, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slab() -> TriMesh {
        // A 4 x 4 x 0.5 m slab, top face at z = 0.
        crate::mesh::tests::box_mesh(Vec3::new(-2.0, -2.0, -0.5), Vec3::new(2.0, 2.0, 0.0))
    }

    #[test]
    fn bake_matches_analytic_above_slab() {
        let sdf = SdfGrid::bake(&slab(), 0.05, 0.3);
        // Above the top-face interior the field is exactly z, and z is linear,
        // so trilinear interpolation reproduces it to f32 rounding.
        for (x, y, z) in [(0.0, 0.0, 0.1), (0.4, -0.7, 0.02), (-1.0, 1.0, 0.25), (0.13, 0.6, 0.005)]
        {
            let d = sdf.sample(Vec3::new(x, y, z)).unwrap();
            assert!((d - z).abs() < 1e-6, "sdf({x},{y},{z}) = {d}, want {z}");
        }
        // Inside the slab: negative.
        let d = sdf.sample(Vec3::new(0.0, 0.0, -0.1)).unwrap();
        assert!((d - (-0.1)).abs() < 1e-6, "inside: {d}");
    }

    #[test]
    fn gradient_is_up_above_slab() {
        let sdf = SdfGrid::bake(&slab(), 0.05, 0.3);
        let (d, g) = sdf.sample_with_gradient(Vec3::new(0.3, -0.4, 0.08)).unwrap();
        assert!((d - 0.08).abs() < 1e-6);
        let n = g.normalize();
        assert!((n.z - 1.0).abs() < 1e-6, "normal {n:?}");
        assert!(n.x.abs() < 1e-6 && n.y.abs() < 1e-6);
    }

    #[test]
    fn outside_is_none() {
        let sdf = SdfGrid::bake(&slab(), 0.05, 0.1);
        assert!(sdf.sample(Vec3::new(50.0, 0.0, 0.0)).is_none());
        assert!(sdf.sample(Vec3::new(0.0, 0.0, 10.0)).is_none());
        assert!(sdf.sample(Vec3::new(f64::NAN, 0.0, 0.0)).is_none());
    }

    #[test]
    fn roundtrip() {
        let sdf = SdfGrid::bake(&slab(), 0.25, 0.25);
        let dir = std::env::temp_dir().join("ipse-map-sdf-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sdf.bin");
        sdf.save(&path).unwrap();
        let back = SdfGrid::load(&path).unwrap();
        assert_eq!(back.data, sdf.data);
        assert_eq!((back.nx, back.ny, back.nz), (sdf.nx, sdf.ny, sdf.nz));
        assert_eq!(back.cell, sdf.cell);
        assert_eq!(back.origin.z, sdf.origin.z);
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let sdf = SdfGrid::bake(&slab(), 0.05, 0.3);
        // Near the slab edge the field is genuinely 3D; check the analytic
        // gradient against central differences of `sample` inside one cell.
        let p = Vec3::new(1.97, 0.31, 0.06);
        let (_, g) = sdf.sample_with_gradient(p).unwrap();
        let h = 1e-4;
        for (axis, ga) in [(Vec3::x(), g.x), (Vec3::y(), g.y), (Vec3::z(), g.z)] {
            let plus = sdf.sample(p + axis * h).unwrap();
            let minus = sdf.sample(p - axis * h).unwrap();
            let fd = (plus - minus) / (2.0 * h);
            assert!((fd - ga).abs() < 1e-3, "axis {axis:?}: fd {fd} vs analytic {ga}");
        }
    }
}
