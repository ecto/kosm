//! What the last frame already knew.
//!
//! The path tracer is asked for one sample a pixel a pass, which on its own is
//! a blizzard. What makes that watchable is not spending more: it is refusing
//! to throw the previous frame away. A game engine keeps the samples it has,
//! carries them into the new camera, and only discards the pixels that are
//! genuinely different. That is all this module is — a per-pixel running mean
//! with a count, a reprojection, and a mask saying where the world moved.
//!
//! ## what vcad's guide buffers mean
//!
//! Read off `pathtrace::radiance` and `pathtrace::render`, because the whole
//! reprojection rests on it:
//!
//! - **`Film::depth[i]`** is `(hit - eye).norm()` for the *first* sample's
//!   primary ray: the **distance along the ray from the eye to the first hit**,
//!   in world units — millimetres here — **not** a view-space `z` and not
//!   normalised. **Zero means the primary ray escaped**; it is the background
//!   sentinel, and the denoiser treats it as inviolable.
//! - **`Film::normal[i]`** is the **world-space** unit normal at that hit,
//!   already face-forwarded towards the eye. Zero for background.
//! - **`Film::albedo[i]`** is the surface's `denoise_albedo` at the hit — the
//!   colour demodulated out before filtering.
//! - The guides come from sample 0 only, jittered inside the pixel, so they are
//!   accurate to about half a pixel. Everything below is toleranced for that.
//!
//! Because depth is a ray distance, unprojecting is exact and cheap:
//! `p = eye + normalize(dir(px, py)) * depth`. No matrix inverse anywhere.
//!
//! ## both tiers bring them
//!
//! They did not always. The GPU tier used to hand over colour alone — the
//! compute shader wrote depth and normals into a buffer it allocated itself
//! and never returned — so its history was mask-only: a camera that moved
//! threw the whole picture away rather than most of it, and the a-trous
//! filter, which passes a pixel through untouched wherever `depth` is zero,
//! was a no-op on it.
//!
//! vcad's `render_resident_linear` returns the guides in exactly the
//! conventions above, so [`History::merge`] no longer needs to ask which
//! tracer it is being fed. Reprojection and denoising are the same code on
//! either tier.
//!
//! ## the mask buys rays now
//!
//! vcad grew `pathtrace::render_into(scene, cam, &mut Film, opts, rects)` — a
//! bit-identical masked re-trace of a list of rectangles — and
//! `GpuRenderState::set_scissor`, which sizes the compute dispatch to one
//! rectangle. So the mask is no longer only a filter on which samples survive:
//! it is the work. [`History::plan`] answers *before* a pass which rectangles
//! the world moved under, the renderer traces only those, and
//! [`History::merge`] is told which rectangles were actually traced so it
//! leaves every other pixel — its mean *and* its count — exactly as it was.
//! A pass now costs in proportion to what moved.
//!
//! When nothing moved the plan is empty, and an empty plan means a *full*
//! pass: that is the only way the picture converges, and it is what the quiet
//! stretches between bounces are for.

use vcad_kernel_math::{Point3, Vec3};
use vcad_kernel_raytrace::pathtrace::{self, Film, PathTraceOptions};

/// A pixel's history is kept if the reprojected distance agrees to this,
/// relative.
const DEPTH_TOL: f32 = 0.02;

/// …and if the normals agree to this, as a dot product.
const NORMAL_TOL: f32 = 0.9;

/// Pixels of slack around every projected rectangle: the guide buffers are a
/// jittered sample, silhouettes are soft, and a shadow's penumbra is wider
/// than its umbra.
const DILATE: i32 = 3;

/// A pose that moved by less than this, in millimetres, did not move.
const MOVED_MM: f64 = 1.0;

/// …nor did one whose rotation matrix changed by less than this per element.
const TURNED: f64 = 1e-4;

/// Where the reprojection puts a background pixel: far enough that the
/// direction is all that matters, near enough not to lose float precision.
const FAR_MM: f64 = 1.0e7;

/// A pixel with this many samples is converged enough that the denoiser can
/// only soften it.
const DENOISE_UNTIL: u32 = 32;

// ---- the camera, as the reprojection needs it -------------------------------

/// A pinhole view: where the eye is, the screen basis, and the tangent
/// half-extents of the frustum. Built from `pathtrace::Camera` and a size,
/// which is exactly what `Camera::ray` uses, so pixel centres agree to the
/// float.
///
/// The aperture is ignored. A defocused camera's primary rays leave from the
/// lens, not the eye, so the unprojection is off by at most the aperture
/// radius; the depth and normal tests below reject anything that matters.
#[derive(Clone, Copy, PartialEq)]
pub struct View {
    pub eye: Point3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub half_w: f64,
    pub half_h: f64,
    pub width: u32,
    pub height: u32,
}

impl View {
    pub fn of(cam: &pathtrace::Camera, width: u32, height: u32) -> Self {
        let half_h = (cam.fov_deg.to_radians() * 0.5).tan();
        Self {
            eye: cam.eye,
            forward: cam.forward,
            right: cam.right,
            up: cam.up,
            half_w: half_h * (width as f64 / height as f64),
            half_h,
            width,
            height,
        }
    }

    /// The same eye and the same frustum, sampled at a different raster size.
    ///
    /// What a resampled history was taken under: the camera did not move, so
    /// carrying the picture across a size step must not read as a camera move
    /// — [`History::merge`] compares views, and a view that differs only in
    /// its raster would send the whole picture through the reprojection with
    /// old-size pixel coordinates into new-size buffers.
    pub fn at_size(&self, width: u32, height: u32) -> Self {
        Self {
            half_w: self.half_h * (width as f64 / height as f64),
            width,
            height,
            ..*self
        }
    }

    /// The unit direction through a pixel's centre.
    pub fn ray_dir(&self, px: u32, py: u32) -> Vec3 {
        let sx = 2.0 * ((px as f64 + 0.5) / self.width as f64) - 1.0;
        let sy = 1.0 - 2.0 * ((py as f64 + 0.5) / self.height as f64);
        (self.forward + self.right * (sx * self.half_w) + self.up * (sy * self.half_h)).normalize()
    }

    /// Where a world point lands, in pixel coordinates (a pixel centre is at
    /// the integer). `None` if it is at or behind the eye plane.
    pub fn project(&self, p: Point3) -> Option<(f64, f64)> {
        let v = p - self.eye;
        let z = v.dot(&self.forward);
        if z <= 1e-6 {
            return None;
        }
        let sx = v.dot(&self.right) / (z * self.half_w);
        let sy = v.dot(&self.up) / (z * self.half_h);
        Some((
            (sx + 1.0) * 0.5 * self.width as f64 - 0.5,
            (1.0 - sy) * 0.5 * self.height as f64 - 0.5,
        ))
    }

    /// The screen rectangle a world sphere covers, dilated. `None` when it
    /// falls off the screen entirely; `Some` covering everything when the
    /// sphere contains or straddles the eye, where there is no rectangle.
    fn sphere_rect(&self, centre: Point3, radius: f64) -> Option<Rect> {
        let v = centre - self.eye;
        let z = v.dot(&self.forward);
        // Behind the camera altogether: nothing on screen. A ball that has
        // rolled into the corner behind the viewer used to come back as the
        // whole screen from here, and with it every shadow it threw.
        if z < -radius {
            return None;
        }
        // Straddling the camera plane: there is no rectangle, so everything.
        if z <= radius + 1e-6 {
            return Some(Rect::everything(self.width, self.height));
        }
        let (cx, cy) = self.project(centre)?;
        // Angular half-size at the near-most point of the sphere, which is
        // conservative for the whole of it.
        let r_px = radius / ((z - radius) * self.half_h) * 0.5 * self.height as f64;
        Rect::around(cx, cy, r_px, self.width, self.height)
    }
}

/// A half-open pixel rectangle.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

impl Rect {
    fn everything(w: u32, h: u32) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: w,
            y1: h,
        }
    }

    /// `[x, y, w, h]`, which is what both tracers' masked entry points take.
    fn to_xywh(self) -> [u32; 4] {
        [self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0]
    }

    fn around(cx: f64, cy: f64, r: f64, w: u32, h: u32) -> Option<Self> {
        let r = r + DILATE as f64;
        let x0 = (cx - r).floor();
        let x1 = (cx + r).ceil() + 1.0;
        let y0 = (cy - r).floor();
        let y1 = (cy + r).ceil() + 1.0;
        if x1 <= 0.0 || y1 <= 0.0 || x0 >= w as f64 || y0 >= h as f64 {
            return None;
        }
        Some(Self {
            x0: x0.max(0.0) as u32,
            y0: y0.max(0.0) as u32,
            x1: (x1.min(w as f64) as u32).max(1),
            y1: (y1.min(h as f64) as u32).max(1),
        })
    }
}

// ---- what moved -------------------------------------------------------------

/// One thing that can move, in millimetres: where it is, how it is turned, and
/// a sphere that contains it. The renderer builds these from a `Snapshot` —
/// the balls, then the extras — and the order is the identity, so a court that
/// gains or loses a body invalidates everything for one frame.
#[derive(Clone, Copy, PartialEq)]
pub struct Pose {
    pub centre: [f64; 3],
    pub rot: [f64; 9],
    pub radius: f64,
}

impl Pose {
    #[cfg(test)]
    pub fn still(centre: [f64; 3], radius: f64) -> Self {
        Self {
            centre,
            rot: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            radius,
        }
    }

    fn point(&self) -> Point3 {
        Point3::new(self.centre[0], self.centre[1], self.centre[2])
    }

    /// Did this pose change enough to be worth a repaint? A millimetre of
    /// travel, or any turn a seam would show.
    fn differs(&self, other: &Pose) -> bool {
        let d = [
            self.centre[0] - other.centre[0],
            self.centre[1] - other.centre[1],
            self.centre[2] - other.centre[2],
        ];
        if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() > MOVED_MM {
            return true;
        }
        self.rot
            .iter()
            .zip(&other.rot)
            .any(|(a, b)| (a - b).abs() > TURNED)
    }
}

/// What a pass should trace, decided before it runs.
///
/// `full` means the whole frame: either there is no history to keep (the first
/// pass, a resize) or the camera moved, which the mask cannot describe. An
/// empty `rects` with `full` false means *nothing moved*, and the caller
/// should also render the whole frame — that is the only way a picture with a
/// still world gains samples.
pub struct Plan {
    pub rects: Vec<[u32; 4]>,
    pub full: bool,
}

impl Plan {
    /// The pixels the plan asks for, as a share of the screen. Rectangles may
    /// overlap, so this is an upper bound — which is the safe side for a
    /// caller deciding whether a patch is still cheaper than the frame.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn coverage(&self, size: (u32, u32)) -> f32 {
        let n = (size.0 as f32) * (size.1 as f32);
        if self.full || n <= 0.0 {
            return 1.0;
        }
        let px: f32 = self
            .rects
            .iter()
            .map(|r| (r[2] as f32) * (r[3] as f32))
            .sum();
        (px / n).min(1.0)
    }

    /// One rectangle covering every rectangle in the plan — what a single
    /// scissored dispatch can do.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn bbox(&self) -> Option<[u32; 4]> {
        let mut it = self.rects.iter();
        let first = *it.next()?;
        let (mut x0, mut y0) = (first[0], first[1]);
        let (mut x1, mut y1) = (first[0] + first[2], first[1] + first[3]);
        for r in it {
            x0 = x0.min(r[0]);
            y0 = y0.min(r[1]);
            x1 = x1.max(r[0] + r[2]);
            y1 = y1.max(r[1] + r[3]);
        }
        Some([x0, y0, x1 - x0, y1 - y0])
    }
}

// ---- the history ------------------------------------------------------------

/// The running mean of every pixel, how many samples went into it, and the
/// guide buffers of the last pass.
pub struct History {
    size: (u32, u32),
    mean: Vec<f32>,
    alpha: Vec<f32>,
    /// Samples behind each pixel's mean. Zero is "nothing here yet".
    count: Vec<u32>,
    normal: Vec<f32>,
    depth: Vec<f32>,
    albedo: Vec<f32>,
    variance: Vec<f32>,
    view: Option<View>,
    poses: Vec<Pose>,
    /// The share of the screen the last merge threw away.
    mask_fraction: f32,
}

impl History {
    pub fn new(size: (u32, u32)) -> Self {
        let n = (size.0 as usize) * (size.1 as usize);
        Self {
            size,
            mean: vec![0.0; n * 3],
            alpha: vec![0.0; n],
            count: vec![0; n],
            normal: vec![0.0; n * 3],
            depth: vec![0.0; n],
            albedo: vec![0.0; n * 3],
            variance: vec![0.0; n],
            view: None,
            poses: Vec::new(),
            mask_fraction: 1.0,
        }
    }

    pub fn mask_fraction(&self) -> f32 {
        self.mask_fraction
    }

    /// Carry the whole history into a new raster size.
    ///
    /// The tuner steps the render size — that is how the window buys a pass
    /// that fits in thirty milliseconds — and every step used to throw the
    /// accumulated picture away: a 426→365 step took the mean sample count
    /// from 599 to 26 and the window went back to looking like a blizzard for
    /// several seconds. Nothing about a resolution step says the *picture*
    /// changed, though. It is the same camera looking at the same world; only
    /// the grid it is sampled on moved.
    ///
    /// So every plane is bilinearly resampled — the mean, the alpha, the
    /// variance and the three guide planes — and the counts with them, rounded
    /// rather than interpolated because a count is a number of samples and
    /// there is no such thing as 3.4 of one. Costing a couple of passes over a
    /// few hundred kilobytes, it is far cheaper than re-converging.
    ///
    /// The stored view is re-stated at the new size rather than left alone: it
    /// records the camera the history was taken under, and the camera did not
    /// move.
    pub fn resample(&mut self, size: (u32, u32)) {
        if size == self.size {
            return;
        }
        let (nw, nh) = size;
        let n = (nw as usize) * (nh as usize);
        if n == 0 || self.size.0 == 0 || self.size.1 == 0 {
            *self = History::new(size);
            return;
        }
        let (ow, oh) = self.size;
        // Where each new pixel's centre falls on the old grid, and the four
        // taps around it. Clamped at the edges, which is the right reading of
        // a picture that has no samples beyond its border.
        let taps: Vec<(usize, usize, usize, usize, f32, f32)> = (0..n)
            .map(|i| {
                let px = (i as u32) % nw;
                let py = (i as u32) / nw;
                let fx = ((px as f32 + 0.5) * ow as f32 / nw as f32 - 0.5).max(0.0);
                let fy = ((py as f32 + 0.5) * oh as f32 / nh as f32 - 0.5).max(0.0);
                let x0 = (fx.floor() as u32).min(ow - 1);
                let y0 = (fy.floor() as u32).min(oh - 1);
                let x1 = (x0 + 1).min(ow - 1);
                let y1 = (y0 + 1).min(oh - 1);
                let tx = fx - x0 as f32;
                let ty = fy - y0 as f32;
                (
                    (y0 * ow + x0) as usize,
                    (y0 * ow + x1) as usize,
                    (y1 * ow + x0) as usize,
                    (y1 * ow + x1) as usize,
                    tx.clamp(0.0, 1.0),
                    ty.clamp(0.0, 1.0),
                )
            })
            .collect();
        let lerp = |src: &[f32], lanes: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; n * lanes];
            for (i, &(a, b, c, d, tx, ty)) in taps.iter().enumerate() {
                for l in 0..lanes {
                    let top = src[a * lanes + l] + (src[b * lanes + l] - src[a * lanes + l]) * tx;
                    let bot = src[c * lanes + l] + (src[d * lanes + l] - src[c * lanes + l]) * tx;
                    out[i * lanes + l] = top + (bot - top) * ty;
                }
            }
            out
        };
        self.mean = lerp(&self.mean, 3);
        self.alpha = lerp(&self.alpha, 1);
        self.normal = lerp(&self.normal, 3);
        self.depth = lerp(&self.depth, 1);
        self.albedo = lerp(&self.albedo, 3);
        self.variance = lerp(&self.variance, 1);
        let counts: Vec<f32> = self.count.iter().map(|&c| c as f32).collect();
        self.count = lerp(&counts, 1)
            .into_iter()
            .map(|c| c.round().max(0.0) as u32)
            .collect();
        self.size = size;
        self.view = self.view.map(|v| v.at_size(nw, nh));
    }

    /// Samples per pixel, averaged over the screen: the number that says
    /// whether the picture is actually converging.
    pub fn mean_samples(&self) -> f32 {
        if self.count.is_empty() {
            return 0.0;
        }
        self.count.iter().map(|&c| c as f64).sum::<f64>() as f32 / self.count.len() as f32
    }

    /// The samples behind one pixel. For tests, mostly.
    #[cfg(test)]
    pub fn samples_at(&self, px: u32, py: u32) -> u32 {
        self.count[(py * self.size.0 + px) as usize]
    }

    /// Fold one pass in.
    ///
    /// The order matters and is the whole design: reproject first, so a moved
    /// camera carries its samples across, *then* mask, so a moved ball takes
    /// out the pixels it is in — at both its old and its new pose — whether or
    /// not the camera also moved.
    pub fn merge(
        &mut self,
        film: &Film,
        view: &View,
        poses: &[Pose],
        lights: &[Point3],
        traced: Option<&[[u32; 4]]>,
    ) {
        // A size step is a new grid, not a new picture: carry it across.
        self.resample((film.width, film.height));
        let n = (self.size.0 as usize) * (self.size.1 as usize);

        // Which pixels this pass actually re-traced. `None` is the whole
        // frame; anything else and every pixel outside keeps its mean *and*
        // its count, because no new sample was drawn there and a count that
        // climbed without one would weight a stale mean against the next
        // real sample.
        let fresh: Option<Vec<bool>> = traced.map(|rects| {
            let mut f = vec![false; n];
            for r in rects {
                let x0 = r[0].min(self.size.0);
                let y0 = r[1].min(self.size.1);
                let x1 = r[0].saturating_add(r[2]).min(self.size.0);
                let y1 = r[1].saturating_add(r[3]).min(self.size.1);
                for py in y0..y1 {
                    for px in x0..x1 {
                        f[(py * self.size.0 + px) as usize] = true;
                    }
                }
            }
            f
        });

        // A pixel is live if it has history that survived both tests.
        let mut live: Vec<bool> = self.count.iter().map(|&c| c > 0).collect();
        match self.view {
            // A moved camera, with depth to unproject through: carry what
            // reprojects. Without it, nothing can be carried at all.
            Some(old) if old != *view => self.reproject(&old, view, film, &mut live),
            None => live.iter_mut().for_each(|l| *l = false),
            _ => {}
        }

        let masked = self.paint_mask(view, poses, lights, &mut live);
        self.mask_fraction = masked as f32 / n.max(1) as f32;

        for i in 0..n {
            if fresh.as_ref().is_some_and(|f| !f[i]) {
                continue;
            }
            if live[i] {
                let c = self.count[i] + 1;
                self.count[i] = c;
                let k = 1.0 / c as f32;
                for c3 in 0..3 {
                    let m = self.mean[i * 3 + c3];
                    self.mean[i * 3 + c3] = m + (film.rgb[i * 3 + c3] - m) * k;
                }
                self.alpha[i] += (film.alpha[i] - self.alpha[i]) * k;
                self.variance[i] += (film.variance[i] - self.variance[i]) * k;
            } else {
                self.count[i] = 1;
                self.mean[i * 3..i * 3 + 3].copy_from_slice(&film.rgb[i * 3..i * 3 + 3]);
                self.alpha[i] = film.alpha[i];
                self.variance[i] = film.variance[i];
            }
        }
        // The guides are always the newest pass's: they describe the geometry
        // the picture is of *now*, and the next reprojection reads them. Both
        // tiers keep their film between passes — the CPU patches it in place
        // with `render_into`, the GPU reads back device buffers a scissored
        // pass only partly rewrote — so outside a masked rectangle the guides
        // are still last pass's, which is the same picture. They are copied
        // whole either way.
        self.normal.copy_from_slice(&film.normal);
        self.depth.copy_from_slice(&film.depth);
        self.albedo.copy_from_slice(&film.albedo);
        self.view = Some(*view);
        self.poses = poses.to_vec();
    }

    /// Carry the history into a moved camera.
    ///
    /// Backwards, which is why it needs the new pass in hand: every new pixel
    /// unprojects its own hit through the *new* view, projects that world
    /// point back through the *old* one, and takes the sample it finds there —
    /// nearest tap, since a bilinear one would smear a silhouette across the
    /// pixels either side of it and the counts with it. A background pixel has
    /// no hit, so it unprojects along its direction alone and may only match
    /// another background pixel.
    fn reproject(&mut self, old: &View, new: &View, film: &Film, live: &mut [bool]) {
        let (w, h) = self.size;
        let n = (w as usize) * (h as usize);
        let mut mean = vec![0.0f32; n * 3];
        let mut alpha = vec![0.0f32; n];
        let mut count = vec![0u32; n];
        let mut variance = vec![0.0f32; n];

        for py in 0..h {
            for px in 0..w {
                let i = (py * w + px) as usize;
                let d = film.depth[i] as f64;
                let dir = new.ray_dir(px, py);
                let p = new.eye + dir * if d > 0.0 { d } else { FAR_MM };
                let Some((u, v)) = old.project(p) else {
                    live[i] = false;
                    continue;
                };
                let (sx, sy) = (u.round(), v.round());
                if sx < 0.0 || sy < 0.0 || sx >= w as f64 || sy >= h as f64 {
                    live[i] = false;
                    continue;
                }
                let j = (sy as u32 * w + sx as u32) as usize;
                if self.count[j] == 0 {
                    live[i] = false;
                    continue;
                }
                let was_hit = self.depth[j] > 0.0;
                if (d > 0.0) != was_hit {
                    live[i] = false;
                    continue;
                }
                if d > 0.0 {
                    let want = (p - old.eye).norm() as f32;
                    if (want - self.depth[j]).abs() > DEPTH_TOL * want.max(1e-6) {
                        live[i] = false;
                        continue;
                    }
                    let dot = (0..3)
                        .map(|c| film.normal[i * 3 + c] * self.normal[j * 3 + c])
                        .sum::<f32>();
                    if dot < NORMAL_TOL {
                        live[i] = false;
                        continue;
                    }
                }
                live[i] = true;
                mean[i * 3..i * 3 + 3].copy_from_slice(&self.mean[j * 3..j * 3 + 3]);
                alpha[i] = self.alpha[j];
                variance[i] = self.variance[j];
                count[i] = self.count[j];
            }
        }
        self.mean = mean;
        self.alpha = alpha;
        self.count = count;
        self.variance = variance;
    }

    /// The rectangles the world moved under.
    ///
    /// For each pose that changed: the bounding sphere at the old pose and at
    /// the new one, and — because a ball's shadow is as visibly wrong as the
    /// ball — the disc that sphere casts on the floor from each panel. The
    /// shadow is a cone; the disc where it meets `z = 0` is approximated by
    /// its axis (the light-through-centre line, met with the floor) and a
    /// radius scaled by how much further the floor is than the ball.
    ///
    /// Called *before* the pass now, so the tracer can be handed the
    /// rectangles rather than the frame. It reads only what the history
    /// already holds (`self.poses`), which is why it is `&self`.
    fn mask_rects(&self, view: &View, poses: &[Pose], lights: &[Point3]) -> Vec<Rect> {
        mask_rects(self.size, view, poses, &self.poses, lights)
    }
}

/// The rectangles the world moved under, for a size, a view, and the poses of
/// two consecutive frames.
///
/// Free of any history because both tiers need it and only one of them keeps
/// a history now: the CPU tier's [`History`] calls it to plan and to paint its
/// own mask, and the GPU tier's [`Mask`] calls it to build the keep mask the
/// device-side accumulator takes. One geometry, one answer, both tiers.
fn mask_rects(
    size: (u32, u32),
    view: &View,
    poses: &[Pose],
    prev: &[Pose],
    lights: &[Point3],
) -> Vec<Rect> {
    {
        let (w, h) = size;
        let screen = (w as usize) * (h as usize);
        let _ = h;
        let mut rects: Vec<Rect> = Vec::new();
        let mut push = |r: Option<Rect>| {
            if let Some(r) = r {
                rects.push(r);
            }
        };
        let paint = |p: &Pose, push: &mut dyn FnMut(Option<Rect>)| {
            let body = view.sphere_rect(p.point(), p.radius);
            if let Some(r) = &body {
                let area = ((r.x1 - r.x0) as usize) * ((r.y1 - r.y0) as usize);
                if area * 5 > screen * 3 && std::env::var_os("KOSM_MASK_DEBUG").is_some() {
                    eprintln!(
                        "mask   a pose covers {}% of the screen: centre {:?} radius {:.0} mm",
                        100 * area / screen.max(1),
                        p.centre,
                        p.radius
                    );
                }
            }
            push(body);
            for light in lights {
                if let Some((c, r)) = shadow_disc(*light, p.point(), p.radius) {
                    let disc = view.sphere_rect(c, r);
                    if let Some(rc) = &disc {
                        let area = ((rc.x1 - rc.x0) as usize) * ((rc.y1 - rc.y0) as usize);
                        if area * 5 > screen * 3 && std::env::var_os("KOSM_MASK_DEBUG").is_some() {
                            eprintln!(
                                "mask   a shadow covers {}%: light {:?} pose {:?} r {:.0} → disc r {:.0} mm",
                                100 * area / screen.max(1),
                                light,
                                p.centre,
                                p.radius,
                                r
                            );
                        }
                    }
                    push(disc);
                }
            }
        };
        // Bodies are appended — the court drops its balls in over several
        // seconds — so a changed count is not a reason to repaint the whole
        // screen. The shared prefix is compared pairwise; anything past the
        // end of either list appeared or left and is masked on its own.
        let shared = poses.len().min(prev.len());
        for k in 0..shared {
            if !poses[k].differs(&prev[k]) {
                continue;
            }
            paint(&poses[k], &mut push);
            paint(&prev[k], &mut push);
        }
        for p in poses.iter().skip(shared).chain(prev.iter().skip(shared)) {
            paint(p, &mut push);
        }
        rects
    }
}

impl History {
    /// What the next pass should trace, given where everything now is.
    ///
    /// This is the whole of the masked-pass restructuring: the mask used to be
    /// a consequence of a pass and is now its brief. A view that does not
    /// match the one the history was built under — a moved camera, a resize —
    /// or a history with nothing in it yet asks for the whole frame, because
    /// no rectangle describes what changed there.
    pub fn plan(&self, view: &View, poses: &[Pose], lights: &[Point3]) -> Plan {
        if self.view != Some(*view) || self.count.iter().all(|&c| c == 0) {
            return Plan {
                rects: Vec::new(),
                full: true,
            };
        }
        let (w, h) = self.size;
        let raw = self.mask_rects(view, poses, lights);
        if raw.is_empty() {
            return Plan {
                rects: Vec::new(),
                full: false,
            };
        }
        // Four balls with ten shadow discs each, at their old pose and their
        // new, is eighty-odd rectangles that overlap heavily — and
        // `render_into` traces an overlap once per rectangle it is in, so the
        // raw list measured passes at seventeen times the work of the frame.
        // Overlapping rectangles are therefore merged into their union until
        // none of them meet: what comes out is a handful of disjoint boxes.
        //
        // A disjoint *cover* is not the only thing wanted here. Cutting the
        // union into exact row-runs is disjoint too, and it is much tighter —
        // and it was five times slower, because `render_into` sets up a rayon
        // traversal of the whole film per rectangle, and a circular mask cut
        // into rows is one rectangle per row. Fewer, fatter boxes win.
        // The same clustering the GPU tier's scissor uses. `render_into`
        // sets up a rayon traversal of the film per rectangle, so a handful
        // of fat boxes beats a long list of thin ones here for the same
        // reason it beats one dispatch per rectangle there.
        let boxes = cluster(&raw, MAX_BOXES);
        let covered: usize = boxes
            .iter()
            .map(|r| ((r.x1 - r.x0) as usize) * ((r.y1 - r.y0) as usize))
            .sum();
        // Past this much of the screen the boxes cost more than the rays they
        // save, and the whole frame is the better pass: it is one rectangle,
        // and every pixel outside the mask gains a sample from it.
        if covered * 10 > (w as usize) * (h as usize) * 6 {
            return Plan {
                rects: Vec::new(),
                full: true,
            };
        }
        let _ = h;
        Plan {
            rects: boxes.into_iter().map(Rect::to_xywh).collect(),
            full: false,
        }
    }

    /// Kill the pixels the world moved under, and say how many.
    fn paint_mask(
        &self,
        view: &View,
        poses: &[Pose],
        lights: &[Point3],
        live: &mut [bool],
    ) -> usize {
        let (w, _) = self.size;
        // The union, counted once — overlapping rectangles are one mask.
        let mut masked = vec![false; live.len()];
        for r in self.mask_rects(view, poses, lights) {
            for py in r.y0..r.y1 {
                for px in r.x0..r.x1 {
                    masked[(py * w + px) as usize] = true;
                }
            }
        }
        for (l, &m) in live.iter_mut().zip(&masked) {
            if m {
                *l = false;
            }
        }
        masked.iter().filter(|&&m| m).count()
    }

    /// The picture so far: the mean, denoised where it is still noisy, as sRGB
    /// bytes.
    ///
    /// The à-trous filter is a fixed global pass, so the per-pixel sample count
    /// cannot be handed to it — instead the denoised result is blended back
    /// towards the raw mean as a pixel's count climbs, and a pixel that has
    /// seen [`DENOISE_UNTIL`] samples keeps its own estimate untouched. A wall
    /// that has been accumulating for a minute should not be smeared by a
    /// filter tuned for one sample — and once *every* pixel has that many, the
    /// filter is not run at all. It is a fixed cost, 70-90 ms whether the
    /// picture is 170x96 or 512x288, and a still window reaches
    /// [`DENOISE_UNTIL`] everywhere in a couple of seconds and would then pay
    /// it forever for an answer the blend throws away.
    pub fn resolve(&self, exposure: f32, opts: &PathTraceOptions) -> Vec<u8> {
        let n = (self.size.0 as usize) * (self.size.1 as usize);
        let mut film = Film {
            width: self.size.0,
            height: self.size.1,
            rgb: self.mean.clone(),
            alpha: self.alpha.clone(),
            normal: self.normal.clone(),
            depth: self.depth.clone(),
            albedo: self.albedo.clone(),
            variance: self.variance.clone(),
        };
        if opts.denoise && self.count.iter().any(|&c| c.max(1) < DENOISE_UNTIL) {
            let mut filtered = Film {
                width: film.width,
                height: film.height,
                rgb: film.rgb.clone(),
                alpha: film.alpha.clone(),
                normal: film.normal.clone(),
                depth: film.depth.clone(),
                albedo: film.albedo.clone(),
                variance: film.variance.clone(),
            };
            pathtrace::denoise(&mut filtered, opts);
            for i in 0..n {
                let c = self.count[i].max(1);
                if c >= DENOISE_UNTIL {
                    continue;
                }
                let k = 1.0 - (c - 1) as f32 / (DENOISE_UNTIL - 1) as f32;
                for c3 in 0..3 {
                    let raw = film.rgb[i * 3 + c3];
                    film.rgb[i * 3 + c3] = raw + (filtered.rgb[i * 3 + c3] - raw) * k;
                }
            }
        }
        film.to_srgb8(exposure, false)
    }
}

/// Rectangles merged until none of them overlap.
///
/// Any two that meet are replaced by the box around both, which over-covers a
/// little and is exactly the right trade: the caller traces every pixel of
/// every box it is handed, so what matters is that no pixel is in two boxes
/// (it would be traced twice) and that there are few of them (each costs a
/// traversal of the film). A pixel inside a box but outside the true mask is
/// simply a pixel that got a fresh sample it did not need, and the history
/// accumulates it like any other.
fn merged(mut rects: Vec<Rect>) -> Vec<Rect> {
    let meet = |a: &Rect, b: &Rect| a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
    let mut again = true;
    while again {
        again = false;
        let mut out: Vec<Rect> = Vec::with_capacity(rects.len());
        for r in rects {
            match out.iter().position(|o| meet(o, &r)) {
                Some(k) => {
                    out[k] = Rect {
                        x0: out[k].x0.min(r.x0),
                        y0: out[k].y0.min(r.y0),
                        x1: out[k].x1.max(r.x1),
                        y1: out[k].y1.max(r.y1),
                    };
                    again = true;
                }
                None => out.push(r),
            }
        }
        rects = out;
    }
    rects
}

/// The most boxes a masked pass will be cut into.
///
/// Each box is a dispatch of its own, and a dispatch is not free. vcad's
/// `accumulate_and_denoise_resident` scissors the *trace*, but the reproject,
/// accumulate, demodulate, a-trous and resolve passes behind it are still
/// dispatched over the whole frame, and the keep mask is re-uploaded with
/// them. Past a handful of boxes that fixed part outgrows the rays a tighter
/// cover saves.
pub const MAX_BOXES: usize = 4;

/// Cluster disjoint rectangles into at most `k` disjoint boxes.
///
/// Greedy, and the greed is over *wasted* area: the pair whose union adds the
/// least to what the two already cover is merged first, so boxes that nearly
/// touch collapse long before boxes at opposite corners of the court do.
/// Merging down to `k` is not optional — the dispatch budget is what it is —
/// but a pair whose union adds nothing is merged even when the count already
/// fits, because two boxes that tile a rectangle are strictly worse than the
/// one box they tile.
///
/// A union can overlap a box neither of its parents met, so the cover is
/// re-merged after every step: what comes out is always disjoint, which is
/// what lets both tiers add the areas up and trust the total.
fn cluster(rects: &[Rect], k: usize) -> Vec<Rect> {
    let mut boxes = merged(rects.to_vec());
    if k == 0 {
        return boxes;
    }
    let area = |r: &Rect| ((r.x1 - r.x0) as u64) * ((r.y1 - r.y0) as u64);
    let union = |a: &Rect, b: &Rect| Rect {
        x0: a.x0.min(b.x0),
        y0: a.y0.min(b.y0),
        x1: a.x1.max(b.x1),
        y1: a.y1.max(b.y1),
    };
    while boxes.len() > 1 {
        // The cheapest pair to merge, by the area their union adds to them.
        let mut best: Option<(usize, usize, u64)> = None;
        for i in 0..boxes.len() {
            for j in (i + 1)..boxes.len() {
                let cost = area(&union(&boxes[i], &boxes[j]))
                    .saturating_sub(area(&boxes[i]) + area(&boxes[j]));
                if best.is_none_or(|(_, _, b)| cost < b) {
                    best = Some((i, j, cost));
                }
            }
        }
        let (i, j, cost) = best.expect("two or more boxes have a pair");
        // Over the budget a merge happens whatever it costs; under it, only a
        // merge that adds no area is worth losing a box for.
        if boxes.len() <= k && cost > 0 {
            break;
        }
        let m = union(&boxes[i], &boxes[j]);
        boxes.swap_remove(j);
        boxes[i] = m;
        boxes = merged(boxes);
    }
    boxes
}

/// Where a sphere's shadow lands on `z = 0`, from a light above it, and how
/// big it is.
///
/// The centre is where the light-to-centre line meets the floor; the radius is
/// the sphere's, grown by the ratio of the two distances — the similar
/// triangles of a cone through a sphere, near enough for a mask.
fn shadow_disc(light: Point3, centre: Point3, radius: f64) -> Option<(Point3, f64)> {
    let d = centre - light;
    if d.z >= -1e-6 || light.z <= 0.0 {
        return None;
    }
    let t = -light.z / d.z;
    if t <= 1.0 {
        return None;
    }
    let floor = light + d * t;
    let to_ball = d.norm().max(1e-6);
    let to_floor = (floor - light).norm();
    Some((floor, radius * to_floor / to_ball))
}

// ---- the mask, without a history behind it ---------------------------------

/// What the GPU tier keeps of its device-side history.
///
/// The device holds the running mean and the count now — vcad's
/// [`accumulate_and_denoise_resident`] folds each raw sample in on the GPU and
/// never sends one back — so the only thing left on this side is the question
/// the device cannot answer: *which pixels is last frame's mean still true
/// for?* That is the same geometry [`History`] masks with, so it is the same
/// [`mask_rects`], and this struct is only the two frames of state that
/// function needs: the view and the poses the last pass was taken under.
///
/// The answer is a **keep mask**: one byte a pixel, 1 to go on accumulating
/// and 0 to start that pixel over at this pass's sample.
pub struct Mask {
    size: (u32, u32),
    view: Option<View>,
    poses: Vec<Pose>,
    /// Samples behind each pixel, mirroring what the device is doing to its
    /// own counts — the only reason to keep it is to be able to say how far
    /// the picture has converged without reading the device back.
    counts: Vec<u32>,
    fraction: f32,
    /// Whether the consumer reprojects its history on the device. With it on
    /// a camera move is no longer a restart: the mask names only the
    /// rectangles the *world* moved under, exactly as on a still-camera pass,
    /// and vcad's reprojection pass decides which pixels survive the move.
    reproject: bool,
}

impl Mask {
    pub fn new(size: (u32, u32)) -> Self {
        Self {
            size,
            view: None,
            poses: Vec::new(),
            counts: vec![0; (size.0 as usize) * (size.1 as usize)],
            fraction: 1.0,
            reproject: false,
        }
    }

    /// The same mask, for a consumer that reprojects on the device.
    pub fn reprojecting(size: (u32, u32)) -> Self {
        Self {
            reproject: true,
            ..Self::new(size)
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// The share of the frame the last mask restarted.
    pub fn fraction(&self) -> f32 {
        self.fraction
    }

    /// Samples behind the average pixel.
    pub fn mean_samples(&self) -> f32 {
        if self.counts.is_empty() {
            return 0.0;
        }
        self.counts.iter().map(|&c| c as f64).sum::<f64>() as f32 / self.counts.len() as f32
    }

    /// The keep mask for a pass of `samples` samples, and a note that it was
    /// taken.
    ///
    /// A camera that did not move restarts only the rectangles the world
    /// moved under, which is the whole point — the walls keep accumulating
    /// while the balls bounce through them.
    ///
    /// A camera that *moved* depends on who is asking. Without device
    /// reprojection nothing downstream knows where last frame's pixel went,
    /// so the move restarts everything. With it (see [`Mask::reprojecting`])
    /// the move is masked exactly like a still-camera pass and vcad's
    /// reprojection decides what survives — but the pass goes over the whole
    /// frame, since every pixel is looking somewhere new, so no boxes are
    /// offered on a move. The pass after it is boxed again like any other.
    pub fn keep(&mut self, view: &View, poses: &[Pose], lights: &[Point3], samples: u32) -> Keep {
        let n = (self.size.0 as usize) * (self.size.1 as usize);
        let moved = self.view.is_some() && self.view != Some(*view);
        let empty = self.view.is_none() || self.counts.iter().all(|&c| c == 0);
        let restart_all = empty || (moved && !self.reproject);
        let mut keep = vec![u8::from(!restart_all); n];
        let mut boxes: Vec<[u32; 4]> = Vec::new();
        if !restart_all {
            let rects = merged(mask_rects(self.size, view, poses, &self.poses, lights));
            paint_zero(&mut keep, self.size, rects.iter().map(|r| r.to_xywh()));
            // One rectangle over everything that restarted was what a single
            // scissored dispatch could do, and with four balls spread across
            // the court that rectangle is most of the frame — the bound never
            // paid and every pass went full. So the change rects are
            // clustered into at most [`MAX_BOXES`] boxes instead, one
            // dispatch each, and vcad's accumulate honours each box in turn:
            // every pixel outside them keeps its mean, its count and its
            // variance untouched, so nothing stale is folded in as fresh.
            // Still only worth taking when the boxes together save more than
            // half the frame — outside them no pixel gains a sample, and a
            // picture that is always masked never converges. A moved camera
            // takes none: the reprojection needs this pass's depth
            // everywhere.
            if !moved {
                let clustered = cluster(&rects, MAX_BOXES);
                let area: usize = clustered
                    .iter()
                    .map(|r| ((r.x1 - r.x0) as usize) * ((r.y1 - r.y0) as usize))
                    .sum();
                if area * 2 < n {
                    boxes = clustered.into_iter().map(Rect::to_xywh).collect();
                }
            }
        }
        let restarted = keep.iter().filter(|&&k| k == 0).count();
        self.fraction = restarted as f32 / n.max(1) as f32;
        let samples = samples.max(1);
        // A scissored pass touches nothing outside its rectangle, so nothing
        // outside it gains a sample either — the mirror of the counts has to
        // say the same thing the device's own do.
        for (i, (c, &k)) in self.counts.iter_mut().zip(&keep).enumerate() {
            let inside = boxes.is_empty() || {
                let (px, py) = ((i as u32) % self.size.0, (i as u32) / self.size.0);
                boxes
                    .iter()
                    .any(|s| px >= s[0] && px < s[0] + s[2] && py >= s[1] && py < s[1] + s[3])
            };
            if !inside {
                continue;
            }
            *c = if k == 0 { samples } else { *c + samples };
        }
        self.view = Some(*view);
        self.poses = poses.to_vec();
        Keep {
            keep,
            boxes,
            reproject: moved && self.reproject && !empty,
        }
    }
}

/// A keep mask and, when they pay for themselves, the few boxes the pass need
/// not step outside of.
pub struct Keep {
    /// One byte a pixel: 1 to go on accumulating, 0 to start over.
    pub keep: Vec<u8>,
    /// The boxes this pass need not step outside of — `[x, y, w, h]` each,
    /// disjoint, at most [`MAX_BOXES`] of them. Empty for a pass that covers
    /// the whole frame.
    pub boxes: Vec<[u32; 4]>,
    /// The camera moved and the consumer reprojects: this pass should carry
    /// its history across the move rather than restart it. Note the counts
    /// this mask mirrors are optimistic on such a pass — the device restarts
    /// the pixels the reprojection could not match and this side cannot know
    /// which those were.
    pub reproject: bool,
}

/// One rectangle covering all of them.
fn bounding(rects: &[Rect]) -> Option<[u32; 4]> {
    let mut it = rects.iter();
    let first = it.next()?;
    let (mut x0, mut y0, mut x1, mut y1) = (first.x0, first.y0, first.x1, first.y1);
    for r in it {
        x0 = x0.min(r.x0);
        y0 = y0.min(r.y0);
        x1 = x1.max(r.x1);
        y1 = y1.max(r.y1);
    }
    (x1 > x0 && y1 > y0).then_some([x0, y0, x1 - x0, y1 - y0])
}

/// Zero every pixel inside `rects`, clipped to `size`. The keep mask's one
/// piece of raster work, shared so a caller cannot get the clipping subtly
/// different from [`History::merge`]'s.
fn paint_zero(keep: &mut [u8], size: (u32, u32), rects: impl Iterator<Item = [u32; 4]>) {
    for r in rects {
        let x0 = r[0].min(size.0);
        let y0 = r[1].min(size.1);
        let x1 = r[0].saturating_add(r[2]).min(size.0);
        let y1 = r[1].saturating_add(r[3]).min(size.1);
        for py in y0..y1 {
            let row = (py * size.0) as usize;
            keep[row + x0 as usize..row + x1 as usize].fill(0);
        }
    }
}

// ---- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 32;
    const H: u32 = 24;

    fn camera(eye: Point3) -> pathtrace::Camera {
        pathtrace::Camera::look_at(
            eye,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            45.0,
        )
    }

    /// A film whose every pixel hits the plane `y = 0`, facing the camera —
    /// the depth is whatever that plane is at, which is what a reprojection
    /// has to reproduce.
    fn plane_film(view: &View, value: f32) -> Film {
        plane_film_sized(view, value, W, H)
    }

    fn plane_film_sized(view: &View, value: f32, w: u32, h: u32) -> Film {
        let n = (w * h) as usize;
        let mut film = Film {
            width: w,
            height: h,
            rgb: vec![value; n * 3],
            alpha: vec![1.0; n],
            normal: vec![0.0; n * 3],
            depth: vec![0.0; n],
            albedo: vec![0.5; n * 3],
            variance: vec![0.0; n],
        };
        for py in 0..h {
            for px in 0..w {
                let i = (py * w + px) as usize;
                let dir = view.ray_dir(px, py);
                // Plane y = 0, normal -y (towards a camera at negative y).
                let t = -view.eye.y / dir.y;
                film.depth[i] = if t > 0.0 { t as f32 } else { 0.0 };
                film.normal[i * 3 + 1] = -1.0;
            }
        }
        film
    }

    #[test]
    fn a_still_scene_accumulates() {
        let cam = camera(Point3::new(0.0, -3000.0, 0.0));
        let view = View::of(&cam, W, H);
        let film = plane_film(&view, 1.0);
        let poses = [Pose::still([0.0, 0.0, 500.0], 120.0)];
        let mut h = History::new((W, H));
        for _ in 0..16 {
            h.merge(&film, &view, &poses, &[], None);
        }
        for py in 0..H {
            for px in 0..W {
                assert_eq!(h.samples_at(px, py), 16, "pixel {px},{py}");
            }
        }
        assert_eq!(h.mask_fraction(), 0.0);
        // The mean of sixteen identical passes is that pass.
        assert!((h.mean[0] - 1.0).abs() < 1e-6);
    }

    /// The tuner steps the render size and the picture survives it: the
    /// counts come across, the mean comes across, and the pass after the step
    /// goes on accumulating rather than starting from one.
    #[test]
    fn a_size_step_keeps_the_picture() {
        let cam = camera(Point3::new(0.0, -3000.0, 0.0));
        let view = View::of(&cam, W, H);
        let film = plane_film(&view, 1.0);
        let poses = [Pose::still([0.0, 0.0, 500.0], 120.0)];
        let mut h = History::new((W, H));
        for _ in 0..40 {
            h.merge(&film, &view, &poses, &[], None);
        }
        assert!((h.mean_samples() - 40.0).abs() < 1e-3);

        // A step to five sixths of the size, as the tuner takes it.
        let (w2, h2) = (W * 5 / 6, H * 5 / 6);
        let view2 = View::of(&cam, w2, h2);
        let film2 = plane_film_sized(&view2, 1.0, w2, h2);
        h.merge(&film2, &view2, &poses, &[], None);
        assert_eq!(h.size, (w2, h2));
        // Forty-one, not one: nothing was thrown away and this pass counted.
        assert!(
            h.mean_samples() > 40.0,
            "mean spp collapsed to {}",
            h.mean_samples()
        );
        assert!((h.mean[0] - 1.0).abs() < 1e-4);
        assert_eq!(h.mask_fraction(), 0.0);
    }

    #[test]
    fn a_moved_ball_invalidates_only_its_mask() {
        let cam = camera(Point3::new(0.0, -3000.0, 0.0));
        let view = View::of(&cam, W, H);
        let film = plane_film(&view, 1.0);
        let mut h = History::new((W, H));
        let at = |z: f64| [Pose::still([0.0, 0.0, z], 100.0)];
        for _ in 0..8 {
            h.merge(&film, &view, &at(0.0), &[], None);
        }
        let before: Vec<u32> = h.count.clone();
        assert!(before.iter().all(|&c| c == 8));

        // Move it half a metre: a real move, but a small one on screen.
        h.merge(&film, &view, &at(500.0), &[], None);
        let f = h.mask_fraction();
        assert!(
            f > 0.0 && f < 0.9,
            "the mask should be a patch, not the screen: {f}"
        );
        let inside = h.count.iter().filter(|&&c| c == 1).count();
        let outside = h.count.iter().filter(|&&c| c == 9).count();
        assert_eq!(inside + outside, (W * H) as usize);
        assert!(inside > 0 && outside > 0);
    }

    /// The GPU tier's mask, across a camera move. Without device
    /// reprojection a move restarts the frame; with it the move is masked
    /// like a still-camera pass — only the world's own rectangles — and asks
    /// for the reprojection instead. Neither offers a scissor on the move:
    /// every pixel is looking somewhere new.
    #[test]
    fn a_moved_camera_restarts_or_reprojects() {
        let a = camera(Point3::new(0.0, -3000.0, 0.0));
        let va = View::of(&a, W, H);
        let b = camera(Point3::new(200.0, -3000.0, 0.0));
        let vb = View::of(&b, W, H);
        let poses = [Pose::still([0.0, 0.0, 500.0], 100.0)];
        let n = (W * H) as usize;

        let mut plain = Mask::new((W, H));
        let _ = plain.keep(&va, &poses, &[], 1);
        let moved = plain.keep(&vb, &poses, &[], 1);
        assert_eq!(
            moved.keep.iter().filter(|&&k| k == 0).count(),
            n,
            "a move is a restart"
        );
        assert!(!moved.reproject);
        assert!(moved.boxes.is_empty());

        let mut device = Mask::reprojecting((W, H));
        let _ = device.keep(&va, &poses, &[], 1);
        let moved = device.keep(&vb, &poses, &[], 1);
        assert!(
            moved.reproject,
            "a moved camera should ask to be reprojected"
        );
        assert!(
            moved.boxes.is_empty(),
            "a reprojected pass needs this pass's depth everywhere"
        );
        let restarted = moved.keep.iter().filter(|&&k| k == 0).count();
        assert!(
            restarted < n / 2,
            "the mask should be the world's patch, not the frame: {restarted}"
        );

        // The first pass is still a restart, reprojection or not: there is no
        // history behind it to carry.
        let mut fresh = Mask::reprojecting((W, H));
        let first = fresh.keep(&va, &poses, &[], 1);
        assert_eq!(first.keep.iter().filter(|&&k| k == 0).count(), n);
        assert!(!first.reproject);
    }

    #[test]
    fn a_moved_camera_reprojects_a_plane() {
        // Slide the eye sideways: every pixel still sees the same plane, so
        // almost all of the history should survive.
        let a = camera(Point3::new(0.0, -3000.0, 0.0));
        let va = View::of(&a, W, H);
        let mut h = History::new((W, H));
        let poses = [Pose::still([0.0, 0.0, 500.0], 100.0)];
        for _ in 0..8 {
            h.merge(&plane_film(&va, 1.0), &va, &poses, &[], None);
        }
        let b = pathtrace::Camera::look_at(
            Point3::new(200.0, -3000.0, 0.0),
            Point3::new(200.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            45.0,
        );
        let vb = View::of(&b, W, H);
        h.merge(&plane_film(&vb, 1.0), &vb, &poses, &[], None);

        // The strip that slid in from the edge is new; the rest is carried.
        let kept = h.count.iter().filter(|&&c| c == 9).count();
        assert!(
            kept as f64 > 0.75 * (W * H) as f64,
            "a pure pan should keep most of the frame, kept {kept} of {}",
            W * H
        );
    }

    #[test]
    fn a_still_world_plans_nothing_and_a_moved_one_plans_a_patch() {
        let cam = camera(Point3::new(0.0, -3000.0, 0.0));
        let view = View::of(&cam, W, H);
        let film = plane_film(&view, 1.0);
        let at = |z: f64| [Pose::still([0.0, 0.0, z], 100.0)];
        let mut h = History::new((W, H));
        // Nothing in the history yet: the whole frame, because no rectangle
        // describes a picture that does not exist.
        assert!(h.plan(&view, &at(0.0), &[]).full);
        h.merge(&film, &view, &at(0.0), &[], None);

        // A world that did not move asks for nothing — and an empty plan is
        // the caller's cue to render the whole frame and converge.
        let quiet = h.plan(&view, &at(0.0), &[]);
        assert!(!quiet.full && quiet.rects.is_empty());

        // A ball that moved asks for a patch, not the screen.
        let moved = h.plan(&view, &at(500.0), &[]);
        assert!(!moved.full && !moved.rects.is_empty());
        let share = moved.coverage((W, H));
        assert!(
            share > 0.0 && share < 0.9,
            "a patch, not the screen: {share}"
        );
        let bbox = moved.bbox().unwrap();
        for r in &moved.rects {
            assert!(
                r[0] >= bbox[0] && r[0] + r[2] <= bbox[0] + bbox[2],
                "{r:?} outside {bbox:?}"
            );
            assert!(
                r[1] >= bbox[1] && r[1] + r[3] <= bbox[1] + bbox[3],
                "{r:?} outside {bbox:?}"
            );
        }
    }

    /// The point of a masked pass: a pixel nobody re-traced must not have its
    /// count incremented, or the next real sample there is weighed against a
    /// mean that never earned its weight.
    #[test]
    fn a_masked_pass_leaves_the_rest_of_the_screen_alone() {
        let cam = camera(Point3::new(0.0, -3000.0, 0.0));
        let view = View::of(&cam, W, H);
        let mut h = History::new((W, H));
        let at = |z: f64| [Pose::still([0.0, 0.0, z], 100.0)];
        for _ in 0..8 {
            h.merge(&plane_film(&view, 1.0), &view, &at(0.0), &[], None);
        }
        assert!(h.count.iter().all(|&c| c == 8));

        let plan = h.plan(&view, &at(500.0), &[]);
        assert!(!plan.rects.is_empty());
        // A film whose fresh patch is a different colour; outside it is what
        // the previous pass left, exactly as `render_into` would leave it.
        let mut film = plane_film(&view, 1.0);
        for r in &plan.rects {
            for py in r[1]..(r[1] + r[3]).min(H) {
                for px in r[0]..(r[0] + r[2]).min(W) {
                    let i = (py * W + px) as usize;
                    film.rgb[i * 3..i * 3 + 3].copy_from_slice(&[0.25, 0.25, 0.25]);
                }
            }
        }
        h.merge(&film, &view, &at(500.0), &[], Some(&plan.rects));

        let mut inside = 0usize;
        for py in 0..H {
            for px in 0..W {
                let i = (py * W + px) as usize;
                let in_plan = plan
                    .rects
                    .iter()
                    .any(|r| px >= r[0] && px < r[0] + r[2] && py >= r[1] && py < r[1] + r[3]);
                if in_plan {
                    inside += 1;
                    // Re-traced and masked: one sample, the fresh one.
                    assert_eq!(h.count[i], 1, "pixel {px},{py}");
                    assert!((h.mean[i * 3] - 0.25).abs() < 1e-6);
                } else {
                    // Untouched: the same eight samples and the same mean.
                    assert_eq!(h.count[i], 8, "pixel {px},{py}");
                    assert!((h.mean[i * 3] - 1.0).abs() < 1e-6);
                }
            }
        }
        assert!(inside > 0 && inside < (W * H) as usize);
    }

    /// The plan must tile the mask, not merely cover it: a pixel in two
    /// rectangles is a pixel traced twice, and the CPU tier once measured
    /// passes at seventeen times the work of the frame that way.
    #[test]
    fn the_plan_is_a_disjoint_cover() {
        let cam = camera(Point3::new(0.0, -3000.0, 0.0));
        let view = View::of(&cam, W, H);
        let mut h = History::new((W, H));
        // Three balls, all moving, with shadows: plenty of overlap in the raw
        // rectangles the mask makes.
        let at = |k: f64| {
            [
                Pose::still([0.0, 0.0, k], 200.0),
                Pose::still([120.0, 0.0, k + 100.0], 200.0),
                Pose::still([-120.0, 60.0, k + 50.0], 200.0),
            ]
        };
        let lights = [
            Point3::new(0.0, 0.0, 6000.0),
            Point3::new(400.0, 200.0, 6000.0),
        ];
        h.merge(&plane_film(&view, 1.0), &view, &at(300.0), &lights, None);
        let plan = h.plan(&view, &at(600.0), &lights);
        assert!(!plan.rects.is_empty());

        let mut hits = vec![0u32; (W * H) as usize];
        for r in &plan.rects {
            for py in r[1]..r[1] + r[3] {
                for px in r[0]..r[0] + r[2] {
                    assert!(px < W && py < H, "{r:?} leaves the film");
                    hits[(py * W + px) as usize] += 1;
                }
            }
        }
        assert!(hits.iter().all(|&n| n <= 1), "a pixel is in two rectangles");
        assert!(
            plan.rects.len() < 12,
            "{} boxes is too many to trace",
            plan.rects.len()
        );
        let traced: u32 = plan.rects.iter().map(|r| r[2] * r[3]).sum();
        assert_eq!(traced as usize, hits.iter().filter(|&&n| n == 1).count());
        assert!(traced <= W * H, "the plan traces more than the frame");
    }

    #[test]
    fn a_shadow_grows_with_the_throw() {
        let light = Point3::new(0.0, 0.0, 6000.0);
        let (c, r) = shadow_disc(light, Point3::new(0.0, 0.0, 3000.0), 120.0).unwrap();
        assert!(c.z.abs() < 1e-9);
        // Twice as far as the ball, so twice the radius.
        assert!((r - 240.0).abs() < 1e-6, "{r}");
    }

    // ---- clustering the change rects into boxes ------------------------------

    fn r(x0: u32, y0: u32, x1: u32, y1: u32) -> Rect {
        Rect { x0, y0, x1, y1 }
    }

    fn area(r: &Rect) -> u64 {
        ((r.x1 - r.x0) as u64) * ((r.y1 - r.y0) as u64)
    }

    /// Nine scattered rectangles are still only ever [`MAX_BOXES`] dispatches.
    #[test]
    fn clustering_holds_the_dispatch_budget() {
        let mut rects = Vec::new();
        for i in 0..3u32 {
            for j in 0..3u32 {
                rects.push(r(i * 70, j * 70, i * 70 + 20, j * 70 + 20));
            }
        }
        let out = cluster(&rects, MAX_BOXES);
        assert!(out.len() <= MAX_BOXES, "{} boxes", out.len());
        assert!(!out.is_empty());
    }

    /// Clustering only ever merges, so the cover can never grow past the one
    /// box the old code used — the bound the boxes have to beat to be worth
    /// their dispatches.
    #[test]
    fn clustering_never_costs_more_than_the_bounding_box() {
        let rects = vec![
            r(0, 0, 20, 20),
            r(180, 0, 200, 20),
            r(0, 180, 20, 200),
            r(180, 180, 200, 200),
            r(90, 90, 110, 110),
        ];
        let bbox = bounding(&merged(rects.clone())).expect("a bound");
        let bbox_area = (bbox[2] as u64) * (bbox[3] as u64);
        for k in 1..=6 {
            let out = cluster(&rects, k);
            let covered: u64 = out.iter().map(area).sum();
            assert!(
                covered <= bbox_area,
                "k={k}: {covered} covered vs {bbox_area} for the bound",
            );
        }
        // One box is the bounding box, exactly.
        assert_eq!(cluster(&rects, 1).len(), 1);
    }

    /// Boxes far enough apart that merging any pair would cost more than it
    /// saves are handed back untouched when they already fit the budget.
    #[test]
    fn disjoint_boxes_that_fit_are_left_alone() {
        let rects = vec![
            r(0, 0, 20, 20),
            r(180, 0, 200, 20),
            r(0, 180, 20, 200),
            r(180, 180, 200, 200),
        ];
        let out = cluster(&rects, MAX_BOXES);
        assert_eq!(out.len(), 4);
        let covered: u64 = out.iter().map(area).sum();
        assert_eq!(covered, 4 * 400, "no area was added");
    }

    /// Two boxes that tile a rectangle are strictly worse than the one box
    /// they tile, so they are merged even with dispatches to spare.
    #[test]
    fn boxes_that_tile_a_rectangle_merge_under_the_budget() {
        let rects = vec![r(0, 0, 50, 100), r(50, 0, 100, 100)];
        let out = cluster(&rects, MAX_BOXES);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], r(0, 0, 100, 100));
    }

    /// Whatever comes out is disjoint: both tiers add the areas up and trust
    /// the total, and a union can overlap a box neither parent met.
    #[test]
    fn clustering_leaves_the_cover_disjoint() {
        let rects = vec![
            r(0, 0, 40, 40),
            r(60, 0, 100, 40),
            r(30, 30, 70, 70),
            r(0, 60, 40, 100),
            r(60, 60, 100, 100),
            r(120, 120, 160, 160),
        ];
        for k in 1..=6 {
            let out = cluster(&rects, k);
            for i in 0..out.len() {
                for j in (i + 1)..out.len() {
                    let (a, b) = (&out[i], &out[j]);
                    let meet = a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
                    assert!(!meet, "k={k}: {a:?} meets {b:?}");
                }
            }
        }
    }
}
