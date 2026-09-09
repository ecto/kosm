//! TSDF fusion: depth frames become the map, no mesh detour.
//!
//! The robot's native map output is the physics layer itself. Each depth
//! frame, with a camera pose, is integrated straight into a truncated
//! signed-distance grid — the same representation the contact solver reads
//! ([`crate::SdfGrid`]) — so "mapping" and "building the thing the robot can
//! stand on" are one operation. The mesh ([`FusedGrid::extract_mesh`]) is a
//! byproduct for the viewer and for submap ICP, not the pipeline's spine.
//!
//! Fusion is the standard projective TSDF (KinectFusion lineage): every
//! voxel in the camera frustum projects into the depth image, the measured
//! surface distance along the ray is truncated to a band, and voxels
//! accumulate a running weighted average. Two honest properties of that
//! choice:
//!
//! * **Projective distance ≠ true distance.** Off the optical axis the
//!   projective TSDF overestimates distance to slanted surfaces (by 1/cosθ
//!   at grazing angles). Averaged over viewpoints it converges toward the
//!   true field, which is why [`FusedGrid::into_sdf`] is honest after a
//!   sweep and biased after a single frame. The fusion sim test bounds this.
//! * **Unknown is a first-class state.** Weight 0 means "never observed",
//!   and [`FusedGrid::into_sdf`] maps unknown to +truncation — no contact,
//!   the same "no floor beyond the map" semantics as out-of-grid sampling.
//!   The free/unknown boundary is exactly the exploration frontier
//!   ([`FusedGrid::frontiers`]) — where a mapping policy's information
//!   lives.
//!
//! Camera convention matches the rest of the workspace (`see.rs`,
//! `phyz-camera`): OpenCV optical frame, `+z` forward, `+x` right, `+y`
//! down; depth is metric z-depth (distance along the optical axis, not ray
//! length). The pose is the phyz body-transform convention: `rot` is
//! world→camera, `pos` is the camera origin in world.

use phyz_math::{SpatialTransformExt, Vec3};
use rayon::prelude::*;

use crate::SdfGrid;

/// One depth frame with its pinhole intrinsics, ready to fuse.
///
/// Depth in metres; non-finite or non-positive samples are holes and fuse
/// nothing. A deliberately minimal type so this crate stays wgpu-free —
/// `ipse-sim` converts from `phyz-camera` frames, the robot path converts
/// from the vendor's mono16 stream after the units are calibrated.
pub struct DepthFrame<'a> {
    pub width: usize,
    pub height: usize,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// Row-major, `width * height`, metres.
    pub depth: &'a [f32],
}

/// A TSDF grid under construction: distances plus evidence.
///
/// Same layout conventions as [`SdfGrid`] (node-centred, x-fastest, z-up
/// map frame) so `into_sdf` is a copy, not a resample.
#[derive(Clone)]
pub struct FusedGrid {
    pub origin: Vec3,
    pub cell: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Truncation band, metres. Measured distances are clamped to ±this.
    pub truncation: f64,
    /// Truncated signed distance, metres. Meaningless where `weight == 0`.
    pub tsdf: Vec<f32>,
    /// Accumulated evidence per voxel, capped so the map can still change
    /// its mind about a region observed long ago.
    pub weight: Vec<f32>,
}

/// Weight cap: after this much agreeing evidence a voxel updates at a fixed
/// floor rate instead of freezing entirely.
const MAX_WEIGHT: f32 = 64.0;

impl FusedGrid {
    /// An empty (all-unknown) grid covering `lo..hi`, `cell` spacing, with a
    /// truncation band of 3 cells — wide enough that trilinear interpolation
    /// across the band stays smooth, narrow enough that opposing surfaces a
    /// hand-width apart don't fight.
    pub fn empty(lo: Vec3, hi: Vec3, cell: f64) -> FusedGrid {
        assert!(cell > 0.0 && cell.is_finite(), "cell must be positive");
        let extent = hi - lo;
        let nx = (extent.x / cell).ceil() as usize + 1;
        let ny = (extent.y / cell).ceil() as usize + 1;
        let nz = (extent.z / cell).ceil() as usize + 1;
        let truncation = 3.0 * cell;
        FusedGrid {
            origin: lo,
            cell,
            nx,
            ny,
            nz,
            truncation,
            tsdf: vec![truncation as f32; nx * ny * nz],
            weight: vec![0.0; nx * ny * nz],
        }
    }

    /// Integrate one depth frame taken from `pose` (world→camera rotation,
    /// camera origin in world; OpenCV optical axes).
    ///
    /// Every voxel projects into the image; voxels behind the camera,
    /// outside the image, behind the measured surface by more than the
    /// truncation band, or under a depth hole are untouched.
    pub fn integrate(&mut self, frame: &DepthFrame<'_>, pose: &phyz_math::SpatialTransform) {
        self.integrate_impl(frame, None, pose);
    }

    /// [`FusedGrid::integrate`] with the projective bias corrected.
    ///
    /// The plain projective TSDF stores `d − z` along the optical axis, which
    /// is the true distance only where the surface faces the optical axis
    /// squarely; on a floor seen obliquely it over-reads by `1/cos θ`, and a
    /// foot's contact then fires late and shallow. With a per-pixel surface
    /// normal (camera frame, see [`depth_normals`]) the stored value becomes
    /// the point-to-plane distance `(d − z)·(r·n)`, `r` the pixel ray with
    /// unit optical-axis component — exact for a locally
    /// planar surface, and what the contact solver actually needs. The
    /// truncation band is still applied on the projective distance so a
    /// grazing normal cannot smear a surface deep behind itself.
    pub fn integrate_oriented(
        &mut self,
        frame: &DepthFrame<'_>,
        normals: &[[f32; 3]],
        pose: &phyz_math::SpatialTransform,
    ) {
        assert_eq!(normals.len(), frame.width * frame.height, "one normal per pixel");
        self.integrate_impl(frame, Some(normals), pose);
    }

    fn integrate_impl(
        &mut self,
        frame: &DepthFrame<'_>,
        normals: Option<&[[f32; 3]]>,
        pose: &phyz_math::SpatialTransform,
    ) {
        let trunc = self.truncation;
        let (nx, ny) = (self.nx, self.ny);
        let origin = self.origin;
        let cell = self.cell;

        self.tsdf
            .par_chunks_mut(nx * ny)
            .zip(self.weight.par_chunks_mut(nx * ny))
            .enumerate()
            .for_each(|(k, (tsdf_slab, weight_slab))| {
                let z = origin.z + k as f64 * cell;
                for j in 0..ny {
                    let y = origin.y + j as f64 * cell;
                    for i in 0..nx {
                        let p = Vec3::new(origin.x + i as f64 * cell, y, z);
                        let cam = pose.world_to_body_point(p);
                        if cam.z <= 0.0 {
                            continue; // behind the camera
                        }
                        let u = frame.fx * cam.x / cam.z + frame.cx;
                        let v = frame.fy * cam.y / cam.z + frame.cy;
                        if !(u >= 0.0 && v >= 0.0) {
                            continue;
                        }
                        let (ui, vi) = (u as usize, v as usize);
                        if ui >= frame.width || vi >= frame.height {
                            continue;
                        }
                        let pix = vi * frame.width + ui;
                        let d = frame.depth[pix] as f64;
                        if !(d.is_finite() && d > 0.0) {
                            continue; // hole
                        }
                        // Signed distance along the ray, projective: positive
                        // in front of the measured surface.
                        let sdf = d - cam.z;
                        if sdf < -trunc {
                            continue; // occluded, deep behind the surface
                        }
                        // Point-to-plane: scale by the cosine between the
                        // pixel ray and the surface normal, if we have one.
                        let factor = match normals {
                            Some(nm) => {
                                let n = nm[pix];
                                let r = Vec3::new(cam.x / cam.z, cam.y / cam.z, 1.0);
                                // (d − z)·(r·n) with r un-normalised (r_z = 1):
                                // exact point-to-plane distance for a locally
                                // planar surface; > 1 on rays steeper than the
                                // optical axis, < 1 on ones grazing the surface.
                                let c = (r.x * n[0] as f64 + r.y * n[1] as f64 + n[2] as f64).abs();
                                if c.is_finite() { c.clamp(0.1, 4.0) } else { 1.0 }
                            }
                            None => 1.0,
                        };
                        let sdf = (sdf * factor).min(trunc) as f32;
                        let idx = i + nx * j;
                        let w = weight_slab[idx];
                        let w_new = (w + 1.0).min(MAX_WEIGHT);
                        tsdf_slab[idx] = (tsdf_slab[idx] * w + sdf) / (w + 1.0);
                        weight_slab[idx] = w_new;
                    }
                }
            });
    }

    /// Freeze into the contact solver's [`SdfGrid`]. Unknown voxels become
    /// +truncation: far from any surface, therefore no contact — beyond the
    /// observed world there is no floor, matching out-of-grid semantics.
    pub fn into_sdf(&self) -> SdfGrid {
        let data = self
            .tsdf
            .iter()
            .zip(&self.weight)
            .map(|(&d, &w)| if w > 0.0 { d } else { self.truncation as f32 })
            .collect();
        SdfGrid {
            origin: self.origin,
            cell: self.cell,
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            data,
        }
    }

    /// Persistence vote: forget every voxel observed fewer than `min_weight`
    /// times. Below the bar a voxel becomes unknown again — it extracts no
    /// surface and reads `+truncation` in [`FusedGrid::into_sdf`].
    ///
    /// Weight counts integrations (one per view, capped at [`MAX_WEIGHT`]),
    /// so `min_weight = 3` is "seen from at least three views": what a wall
    /// passes and a person walking through one frame does not
    /// ([`crate::room`]).
    pub fn prune(&mut self, min_weight: f32) {
        for w in &mut self.weight {
            if *w < min_weight {
                *w = 0.0;
            }
        }
    }

    /// Fraction of voxels observed at least once. The coverage metric an
    /// exploration policy is scored on.
    pub fn observed_fraction(&self) -> f64 {
        let seen = self.weight.iter().filter(|&&w| w > 0.0).count();
        seen as f64 / self.weight.len().max(1) as f64
    }

    /// Frontier voxels: observed free space adjacent to unknown.
    ///
    /// "Free" is a voxel seen at least `min_weight` times with a TSDF value
    /// safely positive (more than one cell from any surface). Returned as
    /// world positions of voxel centres, unclustered — clustering and
    /// scoring belong to the policy, this is just where the map's knowledge
    /// ends.
    pub fn frontiers(&self, min_weight: f32) -> Vec<Vec3> {
        let free_band = self.cell as f32;
        let idx = |i: usize, j: usize, k: usize| i + self.nx * (j + self.ny * k);
        let mut out = Vec::new();
        for k in 1..self.nz.saturating_sub(1) {
            for j in 1..self.ny.saturating_sub(1) {
                for i in 1..self.nx.saturating_sub(1) {
                    let c = idx(i, j, k);
                    if self.weight[c] < min_weight || self.tsdf[c] <= free_band {
                        continue;
                    }
                    let neighbours = [
                        idx(i - 1, j, k),
                        idx(i + 1, j, k),
                        idx(i, j - 1, k),
                        idx(i, j + 1, k),
                        idx(i, j, k - 1),
                        idx(i, j, k + 1),
                    ];
                    if neighbours.iter().any(|&n| self.weight[n] == 0.0) {
                        out.push(Vec3::new(
                            self.origin.x + i as f64 * self.cell,
                            self.origin.y + j as f64 * self.cell,
                            self.origin.z + k as f64 * self.cell,
                        ));
                    }
                }
            }
        }
        out
    }

    /// Extract the zero surface as triangles (world frame), surface-nets
    /// style: one vertex per sign-change cell at the mean of its edge zero
    /// crossings, two triangles per grid face whose adjacent samples change
    /// sign, wound so the normal points toward positive TSDF (outward).
    ///
    /// Surface nets rather than marching cubes on purpose: a tenth of the
    /// code, no case tables, and the consumers — the viewer, submap ICP —
    /// need a faithful surface, not a topologically exhaustive one. Cells
    /// touching unknown voxels extract nothing: no surface is invented at
    /// the observation boundary.
    /// Wrap a baked SDF as a fully-observed grid, so an analytic scene can be
    /// meshed by the same surface-nets pass fusion uses.
    ///
    /// The one place a map can come from something other than a sensor:
    /// [`crate::props`] builds a garage out of solids and bakes it straight to
    /// an [`SdfGrid`], with no mesh anywhere in the chain — and a map
    /// directory owes the viewer a `mesh.stl`. Rather than a second surface
    /// extractor with its own winding bugs, the analytic field becomes a grid
    /// where every sample is *known*, which is exactly what it is.
    ///
    /// `weight = 1` everywhere is the whole difference from a real fusion
    /// grid, and it is load-bearing: a fused grid extracts nothing at the
    /// observation boundary, which is right for a scan and wrong for a scene
    /// that was never observed at all.
    pub fn from_sdf(sdf: &crate::SdfGrid) -> FusedGrid {
        FusedGrid {
            origin: sdf.origin,
            cell: sdf.cell,
            nx: sdf.nx,
            ny: sdf.ny,
            nz: sdf.nz,
            truncation: sdf.cell * 3.0,
            tsdf: sdf.data.clone(),
            weight: vec![1.0; sdf.data.len()],
        }
    }

    pub fn extract_mesh(&self) -> Vec<[Vec3; 3]> {
        let idx = |i: usize, j: usize, k: usize| i + self.nx * (j + self.ny * k);
        let known = |i: usize, j: usize, k: usize| self.weight[idx(i, j, k)] > 0.0;
        let val = |i: usize, j: usize, k: usize| self.tsdf[idx(i, j, k)] as f64;
        let pos = |i: usize, j: usize, k: usize| {
            self.origin + Vec3::new(i as f64, j as f64, k as f64) * self.cell
        };

        // Pass 1: a vertex for every cell whose 8 corners are known and not
        // all one sign.
        let ncx = self.nx.saturating_sub(1);
        let ncy = self.ny.saturating_sub(1);
        let ncz = self.nz.saturating_sub(1);
        let cell_id = |i: usize, j: usize, k: usize| i + ncx * (j + ncy * k);
        let mut vertex: Vec<Option<Vec3>> = vec![None; ncx * ncy * ncz];

        const EDGES: [([usize; 3], [usize; 3]); 12] = [
            ([0, 0, 0], [1, 0, 0]),
            ([0, 1, 0], [1, 1, 0]),
            ([0, 0, 1], [1, 0, 1]),
            ([0, 1, 1], [1, 1, 1]),
            ([0, 0, 0], [0, 1, 0]),
            ([1, 0, 0], [1, 1, 0]),
            ([0, 0, 1], [0, 1, 1]),
            ([1, 0, 1], [1, 1, 1]),
            ([0, 0, 0], [0, 0, 1]),
            ([1, 0, 0], [1, 0, 1]),
            ([0, 1, 0], [0, 1, 1]),
            ([1, 1, 0], [1, 1, 1]),
        ];

        for k in 0..ncz {
            for j in 0..ncy {
                for i in 0..ncx {
                    let mut all_known = true;
                    let mut any_neg = false;
                    let mut any_pos = false;
                    for dz in 0..2 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                if !known(i + dx, j + dy, k + dz) {
                                    all_known = false;
                                } else if val(i + dx, j + dy, k + dz) < 0.0 {
                                    any_neg = true;
                                } else {
                                    any_pos = true;
                                }
                            }
                        }
                    }
                    if !(all_known && any_neg && any_pos) {
                        continue;
                    }
                    let mut acc = Vec3::zeros();
                    let mut n = 0.0;
                    for (a, b) in EDGES {
                        let va = val(i + a[0], j + a[1], k + a[2]);
                        let vb = val(i + b[0], j + b[1], k + b[2]);
                        if (va < 0.0) != (vb < 0.0) {
                            let t = va / (va - vb);
                            let pa = pos(i + a[0], j + a[1], k + a[2]);
                            let pb = pos(i + b[0], j + b[1], k + b[2]);
                            acc = acc + pa + (pb - pa) * t;
                            n += 1.0;
                        }
                    }
                    if n > 0.0 {
                        vertex[cell_id(i, j, k)] = Some(acc * (1.0 / n));
                    }
                }
            }
        }

        // Pass 2: for each interior grid edge whose two samples change sign,
        // connect the four cells sharing that edge. Winding from the sign of
        // the edge direction: normal must face the positive (outside) end.
        let mut tris = Vec::new();
        let mut quad = |cells: [usize; 4], flip: bool| {
            let [a, b, c, d] = cells;
            if let (Some(v0), Some(v1), Some(v2), Some(v3)) =
                (vertex[a], vertex[b], vertex[c], vertex[d])
            {
                if flip {
                    tris.push([v0, v2, v1]);
                    tris.push([v0, v3, v2]);
                } else {
                    tris.push([v0, v1, v2]);
                    tris.push([v0, v2, v3]);
                }
            }
        };

        for k in 1..ncz {
            for j in 1..ncy {
                for i in 1..ncx {
                    if !known(i, j, k) {
                        continue;
                    }
                    let here = val(i, j, k) < 0.0;
                    // Edge along +x from sample (i,j,k).
                    if i + 1 < self.nx && known(i + 1, j, k) && ((val(i + 1, j, k) < 0.0) != here)
                    {
                        quad(
                            [
                                cell_id(i, j - 1, k - 1),
                                cell_id(i, j, k - 1),
                                cell_id(i, j, k),
                                cell_id(i, j - 1, k),
                            ],
                            !here,
                        );
                    }
                    // Edge along +y.
                    if j + 1 < self.ny && known(i, j + 1, k) && ((val(i, j + 1, k) < 0.0) != here)
                    {
                        quad(
                            [
                                cell_id(i - 1, j, k - 1),
                                cell_id(i - 1, j, k),
                                cell_id(i, j, k),
                                cell_id(i, j, k - 1),
                            ],
                            !here,
                        );
                    }
                    // Edge along +z.
                    if k + 1 < self.nz && known(i, j, k + 1) && ((val(i, j, k + 1) < 0.0) != here)
                    {
                        quad(
                            [
                                cell_id(i - 1, j - 1, k),
                                cell_id(i, j - 1, k),
                                cell_id(i, j, k),
                                cell_id(i - 1, j, k),
                            ],
                            !here,
                        );
                    }
                }
            }
        }
        tris
    }
}

/// Per-pixel surface normals of a depth frame, in the camera frame, unit
/// length, facing the camera (`n_z < 0` for a surface seen head-on).
///
/// Central differences of the back-projected points over `step` pixels;
/// pixels next to a hole or the image border get `[0, 0, -1]` (no
/// correction). Feed to [`FusedGrid::integrate_oriented`].
pub fn depth_normals(frame: &DepthFrame<'_>, step: usize) -> Vec<[f32; 3]> {
    let (w, h) = (frame.width, frame.height);
    let step = step.max(1);
    let point = |u: usize, v: usize| -> Option<Vec3> {
        let d = frame.depth[v * w + u] as f64;
        if !(d.is_finite() && d > 0.0) {
            return None;
        }
        Some(Vec3::new(
            (u as f64 + 0.5 - frame.cx) / frame.fx * d,
            (v as f64 + 0.5 - frame.cy) / frame.fy * d,
            d,
        ))
    };
    let mut out = vec![[0.0, 0.0, -1.0]; w * h];
    for v in step..h.saturating_sub(step) {
        for u in step..w.saturating_sub(step) {
            let (Some(px0), Some(px1), Some(py0), Some(py1)) =
                (point(u - step, v), point(u + step, v), point(u, v - step), point(u, v + step))
            else {
                continue;
            };
            let Some(n) = (px1 - px0).cross(py1 - py0).try_normalize() else { continue };
            // Face the camera: the ray from the origin to the point should
            // run against the normal.
            let n = if n.z > 0.0 { -n } else { n };
            out[v * w + u] = [n.x as f32, n.y as f32, n.z as f32];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use phyz_math::{Mat3, SpatialTransform};

    /// A camera at height `h` looking straight down: optical `+z` maps to
    /// world `−z`, optical `+x` to world `+x` (so optical `+y` → world `−y`).
    /// `rot` is world→camera.
    fn downward_camera(h: f64) -> SpatialTransform {
        let rot = Mat3::from_cols(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        );
        SpatialTransform::new(rot, Vec3::new(0.0, 0.0, h))
    }

    /// Looking straight down at the floor `z = 0`, metric z-depth is the
    /// camera height at *every* pixel — depth is distance along the optical
    /// axis, not ray length.
    fn flat_depth(w: usize, hgt: usize, d: f32) -> Vec<f32> {
        vec![d; w * hgt]
    }

    fn frame(depth: &[f32]) -> DepthFrame<'_> {
        DepthFrame {
            width: 64,
            height: 64,
            fx: 64.0,
            fy: 64.0,
            cx: 32.0,
            cy: 32.0,
            depth,
        }
    }

    #[test]
    fn fused_floor_matches_analytic() {
        let mut g = FusedGrid::empty(
            Vec3::new(-0.6, -0.6, -0.2),
            Vec3::new(0.6, 0.6, 0.6),
            0.02,
        );
        let depth = flat_depth(64, 64, 1.0);
        g.integrate(&frame(&depth), &downward_camera(1.0));

        let sdf = g.into_sdf();
        // Under the camera the fused field near the floor is the true field.
        for (p, want) in [
            (Vec3::new(0.0, 0.0, 0.03), 0.03),
            (Vec3::new(0.1, -0.1, 0.05), 0.05),
            (Vec3::new(-0.2, 0.15, -0.03), -0.03),
        ] {
            let d = sdf.sample(p).unwrap();
            assert!(
                (d - want).abs() < 0.5 * g.cell,
                "fused sdf at {p:?} = {d:.4}, want {want:.4}"
            );
        }
        assert!(g.observed_fraction() > 0.05);
    }

    #[test]
    fn unknown_means_no_contact_and_frontiers_exist() {
        let mut g = FusedGrid::empty(
            Vec3::new(-2.0, -2.0, -0.2),
            Vec3::new(2.0, 2.0, 0.6),
            0.04,
        );
        let depth = flat_depth(64, 64, 1.0);
        g.integrate(&frame(&depth), &downward_camera(1.0));

        let sdf = g.into_sdf();
        // Far outside the frustum: unobserved, mapped to +truncation → no
        // contact for any candidate there.
        let far = sdf.sample(Vec3::new(1.8, 1.8, 0.0)).unwrap();
        assert!(
            (far - g.truncation).abs() < 1e-6,
            "unknown should read +truncation, got {far}"
        );

        // The frustum boundary is a frontier: free space next to unknown.
        let f = g.frontiers(1.0);
        assert!(!f.is_empty(), "a partial observation must have frontiers");
        // Frontier points sit in observed space, above the floor.
        assert!(f.iter().all(|p| p.z > 0.0));
    }

    #[test]
    fn oriented_fusion_matches_projective_on_a_square_on_floor_and_corrects_a_slant() {
        // Looking straight down, the floor faces the optical axis and the
        // normal map is uniformly (0, 0, −1): oriented == projective.
        let depth = flat_depth(64, 64, 1.0);
        let f = frame(&depth);
        let normals = depth_normals(&f, 2);
        assert!(normals.iter().all(|n| (n[2] + 1.0).abs() < 1e-6));
        let lo = Vec3::new(-0.6, -0.6, -0.2);
        let hi = Vec3::new(0.6, 0.6, 0.6);
        let mut a = FusedGrid::empty(lo, hi, 0.02);
        let mut b = FusedGrid::empty(lo, hi, 0.02);
        a.integrate(&f, &downward_camera(1.0));
        b.integrate_oriented(&f, &normals, &downward_camera(1.0));
        assert_eq!(a.tsdf, b.tsdf);

        // A camera 1 m up pitched 45° down at a floor: projective over-reads
        // the height of a point above the floor, oriented reads it true.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        // rows = camera axes in world: x right, y down-forward, z forward-down.
        let rot = Mat3::new(1.0, 0.0, 0.0, 0.0, -s, -s, 0.0, s, -s);
        let pose = SpatialTransform::new(rot, Vec3::new(0.0, 0.0, 1.0));
        let (w, h, fx) = (96usize, 96usize, 96.0);
        let mut d = vec![0f32; w * h];
        for v in 0..h {
            for u in 0..w {
                let ray = Vec3::new((u as f64 + 0.5 - 48.0) / fx, (v as f64 + 0.5 - 48.0) / fx, 1.0);
                let dir = rot.transpose() * ray;
                d[v * w + u] = (-1.0 / dir.z) as f32; // z-depth to the plane z = 0
            }
        }
        let f = DepthFrame { width: w, height: h, fx, fy: fx, cx: 48.0, cy: 48.0, depth: &d };
        let normals = depth_normals(&f, 2);
        let lo = Vec3::new(-0.5, 0.4, -0.15);
        let hi = Vec3::new(0.5, 1.6, 0.4);
        let mut a = FusedGrid::empty(lo, hi, 0.02);
        let mut b = FusedGrid::empty(lo, hi, 0.02);
        a.integrate(&f, &pose);
        b.integrate_oriented(&f, &normals, &pose);
        let p = Vec3::new(0.0, 1.0, 0.04);
        let pa = a.into_sdf().sample(p).unwrap();
        let pb = b.into_sdf().sample(p).unwrap();
        assert!(pa > 0.05, "projective should over-read 0.04 by ~√2, got {pa}");
        assert!((pb - 0.04).abs() < 0.01, "oriented should read ~0.04, got {pb}");
    }

    #[test]
    fn extracted_mesh_lies_on_the_floor() {
        let mut g = FusedGrid::empty(
            Vec3::new(-0.6, -0.6, -0.2),
            Vec3::new(0.6, 0.6, 0.6),
            0.02,
        );
        let depth = flat_depth(64, 64, 1.0);
        // Two frames so the weight field is non-trivial.
        g.integrate(&frame(&depth), &downward_camera(1.0));
        g.integrate(&frame(&depth), &downward_camera(1.0));

        let tris = g.extract_mesh();
        assert!(!tris.is_empty(), "a floor was observed; a surface must come out");
        for t in &tris {
            for v in t {
                assert!(
                    v.z.abs() < g.cell,
                    "surface vertex at z = {} should sit on the floor",
                    v.z
                );
            }
            // Outward (toward +TSDF, i.e. up) winding.
            let n = (t[1] - t[0]).cross(t[2] - t[0]);
            assert!(n.z > 0.0, "floor normal must point up, got {n:?}");
        }
    }
}
