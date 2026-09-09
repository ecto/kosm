//! Room journals → maps: TSDF persistence over VGGT's per-view depth.
//!
//! `scripts/room_fuse.py` runs one joint VGGT inference over a room journal
//! and writes, next to the journal in `<journal>.derived/`, the per-view
//! geometry it solved: `extri.npy` (N×3×4 world→camera, OpenCV, in VGGT's
//! *unit* scale), `intri.npy` (N×3×3 at the inference resolution),
//! `depth.npy` (N×H×W z-depth, unit scale), `floor.npy` (`[nx, ny, nz, d]`,
//! the floor plane in unit-frame direction and *metric* offset), and
//! `report.json` (metric scale, `scale_m_per_unit`). This module is the
//! crate-native consumer of that directory: it never runs inference, it
//! turns solved views into the map the contact solver stands on.
//!
//! # Persistence
//!
//! Every view fuses into one [`FusedGrid`]. A voxel's weight is the number
//! of views that observed it (capped), so after all views are in, pruning
//! voxels below `min_views` ([`FusedGrid::prune`]) is exactly the persistence
//! vote the throwaway `voxel_ply.py` took — but on a signed field, so what
//! survives is not a point cloud but a surface with an inside. Two things
//! fail the vote and average away in the field even before pruning:
//!
//! * **Movers.** A person standing in one view puts a surface in that view's
//!   band; every other view that looks through the same space carves it as
//!   free (`+truncation`, full weight). The running mean goes positive and
//!   the surface never appears.
//! * **VGGT depth noise.** Per-view depth is a few centimetres of jitter on
//!   top of a consistent room; the weighted mean over views converges to
//!   the consistent part.
//!
//! # Frame
//!
//! The map frame is the crate's contract: z-up, metres, floor at `z ≈ 0`.
//! [`FloorFrame`] builds it from `floor.npy` and the scale: `z_map` is the
//! metric height above the fitted floor plane; `x_map` is VGGT's `+z`
//! (roughly the first camera's forward) projected onto the floor. The
//! rotation is recorded in the manifest as `floor-frame` and the shift as
//! `[0, 0, d]`.

use std::path::Path;

use phyz_math::{Mat3, SpatialTransform, Vec3};

use crate::fuse::{depth_normals, DepthFrame, FusedGrid};
use crate::npy::read_npy;
use crate::MapError;

/// The rigid transform from VGGT's unit-scale world into the map frame.
///
/// `p_map = rot * (scale * p_unit) + [0, 0, d]`. `rot`'s rows are the map
/// axes expressed in the unit frame; the third row is the floor normal.
#[derive(Debug, Clone, Copy)]
pub struct FloorFrame {
    /// Floor normal in the unit frame, pointing up (toward the cameras).
    pub normal: Vec3,
    /// Metric offset: `height_m = normal · (scale * p_unit) + d`.
    pub d: f64,
    /// Unit-frame → map-frame rotation (rows: map x, y, z in unit coords).
    pub rot: Mat3,
}

impl FloorFrame {
    /// Build the frame from a floor plane and a forward hint. The hint's
    /// projection onto the floor becomes map `+x`; if the hint is (nearly)
    /// parallel to the normal, unit `+x` is used instead.
    pub fn new(normal: Vec3, d: f64, forward_hint: Vec3) -> FloorFrame {
        let n = normal.normalize();
        let project = |v: Vec3| v - n * v.dot(n);
        let hint = project(forward_hint);
        let x = if hint.norm() > 1e-3 {
            hint.normalize()
        } else {
            project(Vec3::new(1.0, 0.0, 0.0)).normalize()
        };
        let y = n.cross(x);
        // Rows are the map axes: `rot * v` gives v's map coordinates.
        let rot = Mat3::new(x.x, x.y, x.z, y.x, y.y, y.z, n.x, n.y, n.z);
        FloorFrame { normal: n, d, rot }
    }

    /// A unit-frame point into the map frame.
    #[inline]
    pub fn to_map(&self, p_unit: Vec3, scale: f64) -> Vec3 {
        self.rot * (p_unit * scale) + Vec3::new(0.0, 0.0, self.d)
    }
}

/// One solved view: pinhole intrinsics at the depth resolution, the
/// unit-frame world→camera extrinsics, and unit-scale z-depth.
#[derive(Debug, Clone)]
pub struct RoomView {
    pub width: usize,
    pub height: usize,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// World(unit)→camera rotation.
    pub rot: Mat3,
    /// World(unit)→camera translation: `p_cam = rot * p_unit + t`.
    pub t: Vec3,
    /// Row-major z-depth in unit scale; `<= 0` or non-finite is a hole.
    pub depth: Vec<f32>,
}

impl RoomView {
    /// Camera centre in the unit frame: `-Rᵀ t`.
    pub fn centre_unit(&self) -> Vec3 {
        -(self.rot.transpose() * self.t)
    }
}

/// A loaded `.derived` directory: every view plus the frame that makes it
/// metric and floor-aligned.
#[derive(Debug, Clone)]
pub struct RoomDerived {
    pub views: Vec<RoomView>,
    /// Metres per VGGT unit — the pose scale, from the upstream anchor.
    pub scale: f64,
    /// Extra multiplier on depth only, `1.0` until
    /// [`RoomDerived::calibrate_depth`] measures otherwise. VGGT's depth head
    /// and camera head carry independent scales; the poses were anchored
    /// upstream, the depth is anchored here.
    pub depth_scale: f64,
    /// Per-view *measured* camera height above the real floor, metres, when
    /// the journal's `head_pose` is available ([`RoomDerived::attach_journal`]).
    pub camera_heights: Vec<Option<f64>>,
    pub floor: FloorFrame,
}

impl RoomDerived {
    /// Load `dir/{extri,intri,depth,floor}.npy` and the scale from
    /// `dir/report.json` (or `scale_override`).
    pub fn load(dir: &Path, scale_override: Option<f64>) -> Result<RoomDerived, MapError> {
        let extri = read_npy(&dir.join("extri.npy"))?;
        let intri = read_npy(&dir.join("intri.npy"))?;
        let depth = read_npy(&dir.join("depth.npy"))?;
        let floor = read_npy(&dir.join("floor.npy"))?;
        let bad = |what: &str, msg: String| MapError::Format(dir.join(what), msg);

        if extri.shape.len() != 3 || extri.shape[1] != 3 || extri.shape[2] != 4 {
            return Err(bad("extri.npy", format!("want N×3×4, got {:?}", extri.shape)));
        }
        let n = extri.shape[0];
        if intri.shape != vec![n, 3, 3] {
            return Err(bad("intri.npy", format!("want {n}×3×3, got {:?}", intri.shape)));
        }
        if depth.shape.len() != 3 || depth.shape[0] != n {
            return Err(bad("depth.npy", format!("want {n}×H×W, got {:?}", depth.shape)));
        }
        if floor.shape != vec![4] {
            return Err(bad("floor.npy", format!("want [nx, ny, nz, d], got {:?}", floor.shape)));
        }
        let (h, w) = (depth.shape[1], depth.shape[2]);

        let scale = match scale_override {
            Some(s) => s,
            None => read_scale(&dir.join("report.json"))?,
        };
        if !(scale > 0.0 && scale.is_finite()) {
            return Err(MapError::Manifest(format!("scale {scale} is not a positive number")));
        }

        let mut views = Vec::with_capacity(n);
        for i in 0..n {
            let e = &extri.data[i * 12..(i + 1) * 12];
            let k = &intri.data[i * 9..(i + 1) * 9];
            let rot = Mat3::new(
                e[0] as f64, e[1] as f64, e[2] as f64,
                e[4] as f64, e[5] as f64, e[6] as f64,
                e[8] as f64, e[9] as f64, e[10] as f64,
            );
            let t = Vec3::new(e[3] as f64, e[7] as f64, e[11] as f64);
            views.push(RoomView {
                width: w,
                height: h,
                fx: k[0] as f64,
                fy: k[4] as f64,
                cx: k[2] as f64,
                cy: k[5] as f64,
                rot,
                t,
                depth: depth.data[i * h * w..(i + 1) * h * w].to_vec(),
            });
        }

        let normal = Vec3::new(floor.data[0] as f64, floor.data[1] as f64, floor.data[2] as f64);
        // VGGT's world is (close to) the first camera's frame: +z forward.
        let floor = FloorFrame::new(normal, floor.data[3] as f64, Vec3::new(0.0, 0.0, 1.0));
        Ok(RoomDerived { views, scale, depth_scale: 1.0, camera_heights: vec![None; n], floor })
    }

    /// The map-frame pose of view `i`, in the phyz body-transform
    /// convention [`FusedGrid::integrate`] expects: `rot` map→camera, `pos`
    /// camera origin in the map.
    ///
    /// Derivation: `p_cam = R_e p_u + t_e` and `p_map = R_f s p_u + o`, so
    /// `p_cam = (R_e R_fᵀ)(p_map − o) + s t_e`… which in metric camera units
    /// (`s p_cam`) is `(R_e R_fᵀ)(p_map − pos)` with `pos = R_f s C_u + o`.
    pub fn camera_pose(&self, i: usize) -> SpatialTransform {
        let v = &self.views[i];
        let rot = v.rot * self.floor.rot.transpose();
        let pos = self.floor.to_map(v.centre_unit(), self.scale);
        SpatialTransform::new(rot, pos)
    }

    /// View `i`'s depth in metres, with everything outside
    /// `min_depth_m..=max_depth_m` (or non-positive/non-finite) turned into
    /// a hole. VGGT's far depth is its least trustworthy and its near depth
    /// is the robot itself; a hole fuses nothing.
    pub fn depth_metric(&self, i: usize, min_depth_m: f64, max_depth_m: f64) -> Vec<f32> {
        let s = (self.scale * self.depth_scale) as f32;
        let (min, max) = (min_depth_m as f32, max_depth_m as f32);
        self.views[i]
            .depth
            .iter()
            .map(|&d| {
                let m = d * s;
                if m.is_finite() && m >= min && m <= max { m } else { 0.0 }
            })
            .collect()
    }

    /// Back-project a subsample of every view (every `stride`-th pixel in
    /// both axes) into the map frame. The cheap cloud bounds and floor
    /// checks are computed from.
    pub fn sample_points(&self, stride: usize, params: &RoomBakeParams) -> Vec<Vec3> {
        let stride = stride.max(1);
        let mut out = Vec::new();
        for (i, v) in self.views.iter().enumerate() {
            let pose = self.camera_pose(i);
            let depth = self.depth_metric(i, params.min_depth_m, params.max_depth_m);
            let cam_to_map = pose.rot.transpose();
            for py in (0..v.height).step_by(stride) {
                for px in (0..v.width).step_by(stride) {
                    let z = depth[py * v.width + px] as f64;
                    if z <= 0.0 {
                        continue;
                    }
                    let x = (px as f64 + 0.5 - v.cx) / v.fx * z;
                    let y = (py as f64 + 0.5 - v.cy) / v.fy * z;
                    out.push(pose.pos + cam_to_map * Vec3::new(x, y, z));
                }
            }
        }
        out
    }

    /// Fusion volume: per-axis percentiles of the sampled cloud, padded,
    /// with the floor guaranteed inside (`lo.z ≤ −pad`).
    pub fn bounds(&self, params: &RoomBakeParams) -> (Vec3, Vec3) {
        let pts = self.sample_points(8, params);
        assert!(!pts.is_empty(), "no depth to bound");
        let axis = |f: fn(&Vec3) -> f64| {
            let mut v: Vec<f64> = pts.iter().map(f).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            let at = |q: f64| v[(((v.len() - 1) as f64) * q).round() as usize];
            (at(params.bounds_pct), at(1.0 - params.bounds_pct))
        };
        let (x0, x1) = axis(|p| p.x);
        let (y0, y1) = axis(|p| p.y);
        let (z0, z1) = axis(|p| p.z);
        let pad = params.pad;
        let lo = Vec3::new(x0 - pad, y0 - pad, (z0 - pad).min(-pad));
        let hi = Vec3::new(x1 + pad, y1 + pad, z1 + pad);
        (lo, hi)
    }

    /// Where the depth-head cloud puts the floor, in the current map frame:
    /// the height of the lowest dense slab within ±0.4 m of `z = 0`.
    ///
    /// `floor.npy` is fit upstream on VGGT's *world-point* head; the depth
    /// head this module fuses is a different prediction and lands the floor
    /// somewhere else (+9 cm on the first garage scan). The map's contract is
    /// "floor at `z ≈ 0`" for the geometry the robot stands on, so the
    /// offset is measured on that geometry: 1 cm height histogram, the
    /// *lowest* local peak that holds at least a quarter of the tallest
    /// bin's count, then the mean height within ±2 cm of it. Mats and low
    /// objects sit above the slab and do not move it, however much of the
    /// view they fill; the ceiling and walls are outside the band.
    pub fn floor_offset(&self, params: &RoomBakeParams) -> Option<f64> {
        let pts = self.sample_points(6, params);
        const LO: f64 = -0.4;
        const HI: f64 = 0.4;
        const BIN: f64 = 0.01;
        let nbins = ((HI - LO) / BIN) as usize;
        let mut hist = vec![0usize; nbins];
        let mut band: Vec<f64> = Vec::new();
        for p in &pts {
            if p.z > LO && p.z < HI {
                hist[((p.z - LO) / BIN) as usize] += 1;
                band.push(p.z);
            }
        }
        let peak = *hist.iter().max()?;
        if peak == 0 {
            return None;
        }
        // Lowest bin that is a local maximum (over a 5-bin window) and at
        // least a quarter as tall as the tallest.
        let b = (0..nbins).find(|&b| {
            let c = hist[b];
            c * 4 >= peak
                && (b.saturating_sub(2)..(b + 3).min(nbins)).all(|k| hist[k] <= c)
        })?;
        let centre = LO + (b as f64 + 0.5) * BIN;
        let near: Vec<f64> = band.iter().copied().filter(|z| (z - centre).abs() <= 0.02).collect();
        if near.is_empty() {
            return None;
        }
        Some(near.iter().sum::<f64>() / near.len() as f64)
    }

    /// Shift the frame so [`RoomDerived::floor_offset`] reads zero. Returns
    /// the shift applied (positive: the depth floor was above `z = 0`).
    pub fn refit_floor(&mut self, params: &RoomBakeParams) -> Option<f64> {
        let off = self.floor_offset(params)?;
        self.floor.d -= off;
        Some(off)
    }

    /// Read per-view measured camera heights from the journal the `.derived`
    /// directory came from, matching views to vantages through
    /// `poses.json` (`stop`, `vantage`). Views without a `head_pose` stay
    /// `None`. Returns how many were matched.
    pub fn attach_journal(&mut self, journal_dir: &Path, derived_dir: &Path) -> Result<usize, MapError> {
        let poses_path = derived_dir.join("poses.json");
        let text = std::fs::read_to_string(&poses_path)
            .map_err(|e| MapError::Io(poses_path.clone(), e))?;
        let poses: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| MapError::Format(poses_path.clone(), format!("json: {e}")))?;
        let views = poses
            .get("views")
            .and_then(|v| v.as_array())
            .ok_or_else(|| MapError::Format(poses_path.clone(), "no views".into()))?;
        if views.len() != self.views.len() {
            return Err(MapError::Format(
                poses_path,
                format!("{} views in poses.json, {} in depth.npy", views.len(), self.views.len()),
            ));
        }
        let journal = crate::journal::RoomJournal::open(journal_dir)
            .map_err(|e| MapError::Io(journal_dir.to_path_buf(), e))?;
        let entries = journal
            .entries()
            .map_err(|e| MapError::Io(journal_dir.join("journal.jsonl"), e))?;
        let mut heights = std::collections::HashMap::new();
        for e in &entries {
            for v in &e.vantages {
                if let Some(hp) = &v.head_pose {
                    heights.insert((e.stop.clone(), v.name.clone()), hp.pos[2]);
                }
            }
        }
        let mut matched = 0;
        for (i, v) in views.iter().enumerate() {
            let stop = v.get("stop").and_then(|s| s.as_str()).unwrap_or("");
            let vant = v.get("vantage").and_then(|s| s.as_str()).unwrap_or("");
            self.camera_heights[i] = heights.get(&(stop.to_string(), vant.to_string())).copied();
            matched += self.camera_heights[i].is_some() as usize;
        }
        Ok(matched)
    }

    /// Anchor the depth head to kinematics: scale depth so the cameras sit
    /// as far above the depth-head floor as the robot says they are.
    ///
    /// The camera-to-floor distance in a view depends on that view's depth
    /// alone (not on where the camera head put the camera), and the head's
    /// height above the ground is measured by the robot's own kinematics —
    /// so their ratio is a clean measurement of the depth head's scale
    /// against the pose scale. On the first garage scan it was 1.12: the
    /// depth head read 12 % short. Applies the mean ratio over views with a
    /// measured height and re-zeroes the floor; returns the ratio, or `None`
    /// when no view has a height (or no floor is found).
    pub fn calibrate_depth(&mut self, params: &RoomBakeParams) -> Option<f64> {
        let floor_z = self.floor_offset(params)?;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..self.views.len() {
            let Some(h) = self.camera_heights[i] else { continue };
            let cam_z = self.camera_pose(i).pos.z;
            let to_floor = cam_z - floor_z;
            if to_floor > 0.2 && h > 0.2 {
                num += h;
                den += to_floor;
            }
        }
        if den <= 0.0 {
            return None;
        }
        let ratio = num / den;
        self.depth_scale *= ratio;
        self.refit_floor(params);
        Some(ratio)
    }

    /// Fuse every view into a persistence-pruned TSDF.
    ///
    /// The returned grid has had [`FusedGrid::prune`] applied: voxels seen
    /// by fewer than `min_views` views are unknown, and so extract nothing
    /// and read `+truncation` in the SDF.
    pub fn bake(&self, params: &RoomBakeParams) -> FusedGrid {
        let (lo, hi) = self.bounds(params);
        self.bake_in(lo, hi, params)
    }

    /// [`RoomDerived::bake`] over an explicit volume.
    pub fn bake_in(&self, lo: Vec3, hi: Vec3, params: &RoomBakeParams) -> FusedGrid {
        let mut g = FusedGrid::empty(lo, hi, params.cell);
        g.truncation = params.trunc_cells * params.cell;
        for i in 0..self.views.len() {
            let v = &self.views[i];
            let depth = self.depth_metric(i, params.min_depth_m, params.max_depth_m);
            let frame = DepthFrame {
                width: v.width,
                height: v.height,
                fx: v.fx,
                fy: v.fy,
                cx: v.cx,
                cy: v.cy,
                depth: &depth,
            };
            // Point-to-plane, not projective: the floor is seen obliquely
            // from a standing camera, and its distance must read true or
            // the feet float (`FusedGrid::integrate_oriented`).
            let normals = depth_normals(&frame, params.normal_step);
            g.integrate_oriented(&frame, &normals, &self.camera_pose(i));
        }
        g.prune(params.min_views as f32);
        g
    }
}

/// Knobs for [`RoomDerived::bake`].
#[derive(Debug, Clone, Copy)]
pub struct RoomBakeParams {
    /// Voxel/SDF spacing, metres.
    pub cell: f64,
    /// Truncation band in cells.
    pub trunc_cells: f64,
    /// A voxel must be observed by at least this many views to survive.
    pub min_views: u32,
    /// Depth beyond this (metres) fuses nothing.
    pub max_depth_m: f64,
    /// Depth nearer than this (metres) fuses nothing. A head camera 0.8 m
    /// up cannot see the room closer than this; what it does see that
    /// close is its own torso (or a textureless mat VGGT pulled forward),
    /// and neither belongs in the map.
    pub min_depth_m: f64,
    /// Volume padding around the cloud's percentile bounds, metres.
    pub pad: f64,
    /// Percentile trimmed off each end of each axis when bounding.
    pub bounds_pct: f64,
    /// Pixel stride for the depth-normal finite differences; larger is
    /// smoother against per-pixel depth jitter.
    pub normal_step: usize,
}

impl Default for RoomBakeParams {
    fn default() -> Self {
        RoomBakeParams {
            cell: 0.02,
            trunc_cells: 3.0,
            min_views: 3,
            max_depth_m: 8.0,
            min_depth_m: 0.5,
            pad: 0.3,
            bounds_pct: 0.005,
            normal_step: 4,
        }
    }
}

fn read_scale(path: &Path) -> Result<f64, MapError> {
    let text = std::fs::read_to_string(path).map_err(|e| MapError::Io(path.to_path_buf(), e))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| MapError::Format(path.to_path_buf(), format!("json: {e}")))?;
    v.get("scale_m_per_unit")
        .and_then(|s| s.as_f64())
        .ok_or_else(|| MapError::Format(path.to_path_buf(), "no scale_m_per_unit".into()))
}

/// Where the extracted surface says the floor and ceiling are.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceLevels {
    /// Area-weighted median height of upward-facing triangles near `z = 0`.
    pub floor_z: Option<f64>,
    /// Area-weighted median height of downward-facing triangles above 1.5 m.
    pub ceiling_z: Option<f64>,
    /// Total upward-facing area within ±0.5 m of the floor, m².
    pub floor_area: f64,
}

/// Measure the floor and ceiling of a triangle soup in the map frame.
///
/// The check `roombake` prints and the acceptance test asserts: after
/// fusion, the surface's floor must land where `floor.npy` said the floor
/// is (`z ≈ 0`) — otherwise the frame or the depth scale is wrong.
pub fn surface_levels(tris: &[[Vec3; 3]]) -> SurfaceLevels {
    let mut up: Vec<(f64, f64)> = Vec::new();
    let mut down: Vec<(f64, f64)> = Vec::new();
    for t in tris {
        let cross = (t[1] - t[0]).cross(t[2] - t[0]);
        let area = 0.5 * cross.norm();
        let Some(n) = cross.try_normalize() else { continue };
        let z = (t[0].z + t[1].z + t[2].z) / 3.0;
        if n.z > 0.8 && z.abs() < 0.5 {
            up.push((z, area));
        } else if n.z < -0.8 && z > 1.5 {
            down.push((z, area));
        }
    }
    let median = |v: &mut Vec<(f64, f64)>| {
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        let total: f64 = v.iter().map(|(_, a)| a).sum();
        let mut acc = 0.0;
        for (z, a) in v.iter() {
            acc += a;
            if acc >= 0.5 * total {
                return Some(*z);
            }
        }
        v.last().map(|(z, _)| *z)
    };
    let floor_area = up.iter().map(|(_, a)| a).sum();
    SurfaceLevels { floor_z: median(&mut up), ceiling_z: median(&mut down), floor_area }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npy::write_npy_f32;
    use phyz_math::SpatialTransformExt;

    /// A synthetic room in the map frame: floor z=0, ceiling z=2.5, walls
    /// at x=±3, y=±2. Analytic z-depth for a pinhole camera.
    struct Room;
    impl Room {
        const CEIL: f64 = 2.5;
        fn hit(o: Vec3, dir: Vec3) -> f64 {
            // Smallest positive t over the six planes.
            let mut best = f64::INFINITY;
            let planes = [
                (Vec3::new(0.0, 0.0, 1.0), 0.0),
                (Vec3::new(0.0, 0.0, -1.0), Self::CEIL),
                (Vec3::new(1.0, 0.0, 0.0), -3.0),
                (Vec3::new(-1.0, 0.0, 0.0), -3.0),
                (Vec3::new(0.0, 1.0, 0.0), -2.0),
                (Vec3::new(0.0, -1.0, 0.0), -2.0),
            ];
            for (n, c) in planes {
                let denom = n.dot(dir);
                if denom.abs() < 1e-9 {
                    continue;
                }
                let t = -(n.dot(o) + c) / denom;
                if t > 1e-6 && t < best {
                    best = t;
                }
            }
            best
        }
    }

    /// Camera at `pos`, yawed by `yaw` about z, pitched down by `pitch`
    /// (positive = look down). Returns map→camera rotation (OpenCV axes).
    fn camera_rot(yaw: f64, pitch: f64) -> Mat3 {
        let fwd = Vec3::new(yaw.cos() * pitch.cos(), yaw.sin() * pitch.cos(), -pitch.sin());
        let up = Vec3::new(0.0, 0.0, 1.0);
        let right = fwd.cross(up).normalize(); // camera +x
        let down = fwd.cross(right).normalize(); // camera +y (down)
        // rows = camera axes in world.
        Mat3::new(
            right.x, right.y, right.z, down.x, down.y, down.z, fwd.x, fwd.y, fwd.z,
        )
    }

    struct Synth {
        w: usize,
        h: usize,
        fx: f64,
        cx: f64,
        cy: f64,
        scale: f64,
        floor: FloorFrame,
    }

    impl Synth {
        fn new() -> Synth {
            // Unit frame with "up" = -y (VGGT-like), d = 0.9 m of the floor
            // below the unit origin, forward hint +z.
            let floor = FloorFrame::new(Vec3::new(0.02, -1.0, 0.01), 0.9, Vec3::new(0.0, 0.0, 1.0));
            Synth { w: 96, h: 72, fx: 80.0, cx: 48.0, cy: 36.0, scale: 2.5, floor }
        }

        /// Render a view from a map-frame pose, and express it as VGGT would
        /// have: unit-frame extrinsics and unit-scale depth.
        fn view(&self, rot_c: Mat3, pos_c: Vec3, mover: Option<(Vec3, f64)>) -> RoomView {
            let cam_to_map = rot_c.transpose();
            let mut depth = vec![0f32; self.w * self.h];
            for py in 0..self.h {
                for px in 0..self.w {
                    let d = Vec3::new(
                        (px as f64 + 0.5 - self.cx) / self.fx,
                        (py as f64 + 0.5 - self.cy) / self.fx,
                        1.0,
                    );
                    let dir_map = cam_to_map * d;
                    let mut t = Room::hit(pos_c, dir_map);
                    if let Some((c, r)) = mover {
                        // A sphere occluder.
                        let oc = pos_c - c;
                        let b = oc.dot(dir_map);
                        let cc = oc.dot(oc) - r * r;
                        let disc = b * b - dir_map.dot(dir_map) * cc;
                        if disc > 0.0 {
                            let ts = (-b - disc.sqrt()) / dir_map.dot(dir_map);
                            if ts > 0.0 && ts < t {
                                t = ts;
                            }
                        }
                    }
                    // z-depth = t since d.z = 1.
                    depth[py * self.w + px] = (t / self.scale) as f32;
                }
            }
            let o = Vec3::new(0.0, 0.0, self.floor.d);
            let rot = rot_c * self.floor.rot;
            let t = rot_c * (o - pos_c) * (1.0 / self.scale);
            RoomView {
                width: self.w,
                height: self.h,
                fx: self.fx,
                fy: self.fx,
                cx: self.cx,
                cy: self.cy,
                rot,
                t,
                depth,
            }
        }

        fn sweep(&self, mover_in_view: Option<usize>) -> RoomDerived {
            let mut views = Vec::new();
            let mut k = 0;
            for &pitch in &[0.35f64, 0.0, -0.35] {
                for i in 0..12 {
                    let yaw = i as f64 * std::f64::consts::TAU / 12.0;
                    let mover = (mover_in_view == Some(k)).then(|| (Vec3::new(1.5, 0.0, 0.8), 0.4));
                    views.push(self.view(camera_rot(yaw, pitch), Vec3::new(0.0, 0.0, 1.0), mover));
                    k += 1;
                }
            }
            RoomDerived { views, scale: self.scale, depth_scale: 1.0, camera_heights: vec![None; 36], floor: self.floor }
        }
    }

    #[test]
    fn floor_frame_is_a_rotation_that_puts_the_normal_up() {
        let f = FloorFrame::new(Vec3::new(0.02, -1.0, 0.01), 0.9, Vec3::new(0.0, 0.0, 1.0));
        let r = f.rot;
        let rt = r * r.transpose();
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((rt.row(i).dot(Vec3::new(
                    if j == 0 { 1.0 } else { 0.0 },
                    if j == 1 { 1.0 } else { 0.0 },
                    if j == 2 { 1.0 } else { 0.0 }
                )) - want).abs() < 1e-12);
            }
        }
        // Right-handed.
        assert!((r.row(0).cross(r.row(1)).dot(r.row(2)) - 1.0).abs() < 1e-12);
        // Normal maps to +z; a point on the plane lands at z = 0.
        let up = r * f.normal;
        assert!((up - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-12);
        let s = 2.0;
        let on_plane = f.normal * (-f.d / s); // n·(s p) + d = 0
        assert!(f.to_map(on_plane, s).z.abs() < 1e-12);
    }

    #[test]
    fn camera_pose_roundtrips_the_unit_extrinsics() {
        let s = Synth::new();
        let rot_c = camera_rot(0.7, 0.2);
        let pos_c = Vec3::new(0.3, -0.4, 1.1);
        let v = s.view(rot_c, pos_c, None);
        let d = RoomDerived { views: vec![v], scale: s.scale, depth_scale: 1.0, camera_heights: vec![None], floor: s.floor };
        let pose = d.camera_pose(0);
        assert!((pose.pos - pos_c).norm() < 1e-9, "pos {:?} vs {:?}", pose.pos, pos_c);
        // A map point in front of the camera lands at metric depth = its
        // camera z, and its pixel matches the render.
        let p = pos_c + rot_c.transpose() * Vec3::new(0.1, -0.05, 1.7);
        let c = pose.world_to_body_point(p);
        assert!((c - Vec3::new(0.1, -0.05, 1.7)).norm() < 1e-9, "{c:?}");
    }

    #[test]
    fn synthetic_room_bakes_floor_and_ceiling_and_stands() {
        let s = Synth::new();
        let d = s.sweep(None);
        let params = RoomBakeParams { cell: 0.05, pad: 0.2, min_views: 2, ..Default::default() };
        let g = d.bake(&params);
        let tris = g.extract_mesh();
        assert!(!tris.is_empty());
        let lv = surface_levels(&tris);
        let floor = lv.floor_z.expect("a floor");
        let ceil = lv.ceiling_z.expect("a ceiling");
        assert!(floor.abs() < 0.03, "floor at {floor}");
        assert!((ceil - Room::CEIL).abs() < 0.05, "ceiling at {ceil}");

        // The SDF stands: distance above the floor reads as height, and a
        // sphere resting on it makes a contact with an up normal.
        let sdf = g.into_sdf();
        // Point-to-plane fusion: the floor's distance reads true even though
        // every view sees it obliquely.
        let h = sdf.sample(Vec3::new(1.5, 0.3, 0.06)).unwrap();
        assert!((h - 0.06).abs() < 0.02, "sdf above floor {h}");
        let below = sdf.sample(Vec3::new(1.5, 0.3, -0.04)).unwrap();
        assert!(below < 0.0, "below the floor should be inside: {below}");

        use phyz_model::{Geometry, Joint, ModelBuilder};
        use phyz_math::SpatialInertia;
        let mut model = ModelBuilder::new()
            .add_body(
                "ball",
                -1,
                Joint::free(SpatialTransform::identity()),
                SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity()),
            )
            .build();
        model.bodies[0].geometry = Some(Geometry::Sphere { radius: 0.1 });
        let mut state = model.default_state();
        state.body_xform[0] = SpatialTransform::new(Mat3::identity(), Vec3::new(1.5, 0.3, 0.09));
        let contacts = crate::find_terrain_contacts_model(&model, &state, &sdf, 0.0);
        assert!(!contacts.is_empty(), "a ball 1 cm into the floor must touch it");
        let c = &contacts[0];
        assert!(c.contact_normal.z > 0.9, "normal {:?}", c.contact_normal);
        assert!(c.penetration_depth > 0.0 && c.penetration_depth < 0.05, "depth {}", c.penetration_depth);
    }

    #[test]
    fn floor_offset_finds_a_misplaced_floor_and_refit_zeroes_it() {
        let s = Synth::new();
        let mut d = s.sweep(None);
        let params = RoomBakeParams::default();
        let off = d.floor_offset(&params).expect("a floor");
        assert!(off.abs() < 0.01, "clean floor offset {off}");
        // Pretend the upstream plane was 12 cm too low (floor reads +0.12).
        d.floor.d += 0.12;
        let off = d.floor_offset(&params).unwrap();
        assert!((off - 0.12).abs() < 0.015, "offset {off}");
        let applied = d.refit_floor(&params).unwrap();
        assert!((applied - 0.12).abs() < 0.015);
        assert!(d.floor_offset(&params).unwrap().abs() < 0.01);
    }

    #[test]
    fn depth_is_anchored_to_measured_camera_height() {
        let s = Synth::new();
        let mut d = s.sweep(None);
        // The depth head reads 10 % short; the cameras really are 1 m up.
        for v in &mut d.views {
            for z in &mut v.depth {
                *z *= 0.9;
            }
        }
        d.camera_heights = vec![Some(1.0); d.views.len()];
        let params = RoomBakeParams { cell: 0.05, pad: 0.2, min_views: 2, ..Default::default() };
        let before = d.floor_offset(&params).unwrap();
        assert!(before > 0.08, "short depth should lift the floor: {before}");
        let ratio = d.calibrate_depth(&params).expect("calibrated");
        assert!((ratio - 1.0 / 0.9).abs() < 0.02, "ratio {ratio}");
        assert!(d.floor_offset(&params).unwrap().abs() < 0.01);
        let lv = surface_levels(&d.bake(&params).extract_mesh());
        assert!(lv.floor_z.unwrap().abs() < 0.03, "floor {:?}", lv.floor_z);
        assert!((lv.ceiling_z.unwrap() - Room::CEIL).abs() < 0.06, "ceiling {:?}", lv.ceiling_z);
    }

    #[test]
    fn a_mover_seen_once_is_voted_away() {
        let s = Synth::new();
        // Mover in a view that looks along +x at pitch 0 (i = 0 of the
        // middle band → k = 12).
        let with_mover = s.sweep(Some(12));
        let clean = s.sweep(None);
        let params = RoomBakeParams { cell: 0.05, pad: 0.2, min_views: 2, ..Default::default() };
        let (lo, hi) = clean.bounds(&params);
        let g_m = with_mover.bake_in(lo, hi, &params);
        let g_c = clean.bake_in(lo, hi, &params);
        // Sanity: the mover really is in the depth of that view.
        let dm = &with_mover.views[12].depth;
        let dc = &clean.views[12].depth;
        assert!(dm.iter().zip(dc).any(|(a, b)| (a - b).abs() > 0.05));

        // Nothing solid at the mover's centre in either bake, and the two
        // fields agree there to well within a cell.
        let c = Vec3::new(1.5, 0.0, 0.8);
        let sm = g_m.into_sdf();
        let sc = g_c.into_sdf();
        for p in [c, c + Vec3::new(0.0, 0.0, 0.3), c + Vec3::new(-0.3, 0.0, 0.0)] {
            let a = sm.sample(p).unwrap();
            let b = sc.sample(p).unwrap();
            assert!(a > 0.5 * params.cell, "mover left a surface at {p:?}: {a}");
            assert!((a - b).abs() < params.cell, "field differs at {p:?}: {a} vs {b}");
        }
        // The mesh has no triangles inside the mover's ball.
        let inside = g_m
            .extract_mesh()
            .iter()
            .filter(|t| (t[0] - c).norm() < 0.35)
            .count();
        assert_eq!(inside, 0, "{inside} triangles inside the mover");
    }

    #[test]
    fn derived_directory_roundtrips_through_npy() {
        let s = Synth::new();
        let d = s.sweep(None);
        let dir = std::env::temp_dir().join(format!("ipse-map-room-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = d.views.len();
        let mut extri = Vec::new();
        let mut intri = Vec::new();
        let mut depth = Vec::new();
        for v in &d.views {
            for r in 0..3 {
                let row = v.rot.row(r);
                extri.extend_from_slice(&[row.x as f32, row.y as f32, row.z as f32, [v.t.x, v.t.y, v.t.z][r] as f32]);
            }
            intri.extend_from_slice(&[
                v.fx as f32, 0.0, v.cx as f32, 0.0, v.fy as f32, v.cy as f32, 0.0, 0.0, 1.0,
            ]);
            depth.extend_from_slice(&v.depth);
        }
        write_npy_f32(&dir.join("extri.npy"), &[n, 3, 4], &extri).unwrap();
        write_npy_f32(&dir.join("intri.npy"), &[n, 3, 3], &intri).unwrap();
        write_npy_f32(&dir.join("depth.npy"), &[n, s.h, s.w], &depth).unwrap();
        let fl = s.floor;
        write_npy_f32(
            &dir.join("floor.npy"),
            &[4],
            &[fl.normal.x as f32, fl.normal.y as f32, fl.normal.z as f32, fl.d as f32],
        )
        .unwrap();
        std::fs::write(dir.join("report.json"), format!("{{\"scale_m_per_unit\": {}}}", s.scale)).unwrap();

        let loaded = RoomDerived::load(&dir, None).unwrap();
        assert_eq!(loaded.views.len(), n);
        assert!((loaded.scale - s.scale).abs() < 1e-12);
        for i in [0, 7, n - 1] {
            let a = loaded.camera_pose(i);
            let b = d.camera_pose(i);
            assert!((a.pos - b.pos).norm() < 1e-4, "view {i} pos");
            assert!((a.rot.row(2) - b.rot.row(2)).norm() < 1e-4, "view {i} fwd");
        }
        assert!(RoomDerived::load(&dir, Some(-1.0)).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
