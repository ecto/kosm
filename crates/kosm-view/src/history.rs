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
//! ## and because it is a ray distance, the projection is a choice
//!
//! Every step of the reprojection below goes through exactly two functions —
//! [`View::ray_dir`], which turns a pixel into a direction, and
//! [`View::project`], which turns a world point back into a pixel — and
//! nothing else in this module knows how a camera maps a screen. So a second
//! map is a second arm on each of those and no change at all to the
//! reprojection, the mask, the plan or the resolve. `kosm_render`'s
//! [`Projection::Equidistant`] is that second map, and it is why the fisheye
//! is no longer offline-only: an `f·θ` frame reprojects through an `f·θ`
//! inverse and lands on the pixel it came from. A depth that was a view-space
//! `z`, or a projection that was a matrix, would both have had to be
//! rewritten instead.
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

use crate::temporal::TemporalHistory;
use kosm_render::pathtrace::Projection;
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

/// Where the reprojection puts a background pixel: far enough that the
/// direction is all that matters, near enough not to lose float precision.
const FAR_MM: f64 = 1.0e7;

/// A pixel with this many samples is converged enough that the denoiser can
/// only soften it.
const DENOISE_UNTIL: u32 = 32;

/// Samples a pixel must already hold before [`History::set_firefly_cap`]
/// engages. Below it the running mean is not a measurement of anything and a
/// cap read off it would clamp the picture to its first sample.
const FIREFLY_WARMUP: u32 = 8;

/// Below this luminance a pixel is dark enough that the cap would be
/// clamping noise against noise, so it is left alone.
const FIREFLY_FLOOR: f32 = 1e-3;

// ---- the camera, as the reprojection needs it -------------------------------

/// The view and the pose the reprojection needs are [`crate::temporal`]'s:
/// one `View` and one `Pose` for every tier, so a sim can hand the same two
/// to this history and to the device's. What this module adds to them is the
/// screen-space arithmetic only a host-side reprojection does — a ray through
/// a pixel, a point projected back, the rectangle a moving sphere covers.
pub use crate::temporal::{Pose, View};

impl View {
    /// The unit direction through a pixel's centre.
    ///
    /// The screen coordinates are the tracer's own — `2·(px + 0.5)/w − 1` and
    /// `1 − 2·(py + 0.5)/h`, which is `kosm_render`'s integrator with the
    /// jitter at the pixel centre — so the two generators can be compared ray
    /// for ray. See [`View::dir`] for the maps themselves.
    pub fn ray_dir(&self, px: u32, py: u32) -> Vec3 {
        let sx = 2.0 * ((px as f64 + 0.5) / self.width as f64) - 1.0;
        let sy = 1.0 - 2.0 * ((py as f64 + 0.5) / self.height as f64);
        self.dir(sx, sy)
    }

    /// The unit direction through normalised screen coordinates in `[-1, 1]`,
    /// under this view's own map. The inverse of [`View::project`].
    ///
    /// This is `kosm_render::Camera::ray` with the aperture taken out (a
    /// history reprojects from a pinhole whatever the lens is doing) and the
    /// `half_fov` folded into [`View`]'s half-extents rather than multiplied
    /// at the end.
    pub fn dir(&self, sx: f64, sy: f64) -> Vec3 {
        match self.projection {
            // Unmoved: the expression this function was before there was a
            // second map, so every rectilinear picture is the one it was.
            Projection::Rectilinear => {
                (self.forward + self.right * (sx * self.half_w) + self.up * (sy * self.half_h))
                    .normalize()
            }
            // `r = f·θ`. The half-extents are radians here, so `(u, v)` *is*
            // the angle off the axis, resolved into its magnitude and its
            // azimuth around the optical axis.
            Projection::Equidistant => {
                let (u, v) = (sx * self.half_w, sy * self.half_h);
                let theta = (u * u + v * v).sqrt();
                if !(theta > 0.0) {
                    return self.forward;
                }
                // Past half a turn the map folds back on itself; the tracer
                // clamps and so does this, so the two agree even there.
                let (st, ct) = theta.min(std::f64::consts::PI).sin_cos();
                (self.forward * ct + (self.right * (u / theta) + self.up * (v / theta)) * st)
                    .normalize()
            }
        }
    }

    /// Where a world point lands, in pixel coordinates (a pixel centre is at
    /// the integer).
    ///
    /// `None` when the point has no image: at the eye, or — under the pinhole
    /// only — at or behind the eye plane. The fisheye has no such plane, which
    /// is the whole of why it is a different map: a point ninety degrees off
    /// the axis is a point this camera can see, and the answer is a screen
    /// radius, not a division by a depth that has gone to zero. A point past
    /// the edge of the frame still comes back, with `|sx|` or `|sy|` over one;
    /// every caller here bounds-checks the pixel it gets.
    pub fn project(&self, p: Point3) -> Option<(f64, f64)> {
        let v = p - self.eye;
        let z = v.dot(&self.forward);
        let (sx, sy) = match self.projection {
            Projection::Rectilinear => {
                if z <= 1e-6 {
                    return None;
                }
                (
                    v.dot(&self.right) / (z * self.half_w),
                    v.dot(&self.up) / (z * self.half_h),
                )
            }
            Projection::Equidistant => {
                let (x, y) = (v.dot(&self.right), v.dot(&self.up));
                let r = x.hypot(y);
                // `atan2` takes the angle past ninety degrees without a
                // singularity, which is the half of the sphere a pinhole
                // cannot describe at all.
                let theta = r.atan2(z);
                if r <= 0.0 {
                    // On the axis, or exactly behind: the first is the centre
                    // of the frame and the second has no azimuth to give.
                    if z <= 0.0 {
                        return None;
                    }
                    (0.0, 0.0)
                } else {
                    (
                        theta * (x / r) / self.half_w,
                        theta * (y / r) / self.half_h,
                    )
                }
            }
        };
        Some((
            (sx + 1.0) * 0.5 * self.width as f64 - 0.5,
            (1.0 - sy) * 0.5 * self.height as f64 - 0.5,
        ))
    }

    /// The screen rectangle a world sphere covers, dilated. `None` when it
    /// falls off the screen entirely; `Some` covering everything when the
    /// sphere contains or straddles the eye, where there is no rectangle.
    ///
    /// The two early outs read the forward depth rather than the map, and
    /// they are right for both of these: a sphere entirely behind the eye
    /// plane is off a frame of any field of view under 180°, and one
    /// straddling it has no rectangle under either map. A fisheye wider than
    /// that would need the first of them rewritten as an angle.
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
        // conservative for the whole of it — expressed, either way, as the
        // screen radius the map gives that angle. The pinhole's `half_h` is a
        // tangent and the tangent is already in hand; the fisheye's is
        // radians, so the angle itself is what is divided by it.
        let r_px = match self.projection {
            Projection::Rectilinear => radius / ((z - radius) * self.half_h),
            Projection::Equidistant => (radius / (z - radius)).atan() / self.half_h,
        } * 0.5
            * self.height as f64;
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

/// The pose the mask follows is [`crate::temporal::Pose`]. The court's balls
/// carry a rotation because a seam shows it; the cove's being, its shadow and
/// the door's face are all followed by where they are and how big they are,
/// and nothing else.
impl Pose {
    fn point(&self) -> Point3 {
        Point3::new(self.centre[0], self.centre[1], self.centre[2])
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
    ///
    /// It is also what [`crate::budget::Budget`] is told a pass repainted, so
    /// it is no longer a test-only convenience.
    pub fn coverage(&self, size: (u32, u32)) -> f32 {
        let n = (size.0 as f32) * (size.1 as f32);
        if self.full || n <= 0.0 {
            return 1.0;
        }
        // `fold` from a positive zero rather than `sum`, whose identity for a
        // float is *negative* zero — which every pace line in the workspace
        // has been faithfully printing as "-0% repainted" on a still frame.
        // `clamp` does not fix it either: `-0.0 < 0.0` is false.
        let px: f32 = self
            .rects
            .iter()
            .map(|r| (r[2] as f32) * (r[3] as f32))
            .fold(0.0, |a, b| a + b);
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
    /// The least the à-trous filter may contribute to a *converged* pixel.
    /// Zero — the default, and the court's — is [`History::resolve`]'s
    /// original behaviour, to the float. See [`History::set_denoise_floor`].
    denoise_floor: f32,
    /// A ceiling on one sample's luminance, as a multiple of what the pixel
    /// has already measured. `None` — the default, and the court's — folds
    /// every sample in whole. See [`History::set_firefly_cap`].
    firefly_cap: Option<f32>,
    /// What [`TemporalHistory::reproject`] was last told, waiting for the film
    /// [`TemporalHistory::accumulate`] will merge under it. Unused by a caller
    /// that drives [`History::merge`] itself.
    pending: Option<(View, Vec<Pose>)>,
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
            denoise_floor: 0.0,
            firefly_cap: None,
            pending: None,
        }
    }

    /// Cap one sample's luminance at `cap` times the pixel's running mean.
    ///
    /// The integrator's own `firefly_clamp` is an absolute number against a
    /// quantity whose scale is the scene's, and it exempts the direct term at
    /// depth zero and every read of a caustic map — so a spike that arrives
    /// through the first hit gets past it whatever it is set to. Measured on
    /// the cove: taking it from 8 to 2 moved the dots on the cliff by nothing
    /// at all.
    ///
    /// What catches those is a cap read off the pixel itself. A pixel that has
    /// already seen [`FIREFLY_WARMUP`] samples knows roughly how bright it is;
    /// a sample `cap` times brighter than that is a path the estimator will
    /// spend hundreds more samples apologising for, and scaling it back to the
    /// ceiling — all three channels by the same factor, so its hue survives —
    /// costs a little energy in exchange for a picture that stops flashing. It
    /// is scale-free, so a pixel inside the caustic has a high mean and keeps
    /// its energy while one on the shaded cliff does not, which is the trade
    /// this level wants.
    ///
    /// `None` is the default and the court's, and folds every sample in whole.
    pub fn set_firefly_cap(&mut self, cap: Option<f32>) {
        self.firefly_cap = cap.filter(|c| c.is_finite() && *c > 0.0);
    }

    /// Keep the filter engaged on pixels that have passed [`DENOISE_UNTIL`].
    ///
    /// The blend in [`Self::resolve`] fades the à-trous filter out as a pixel
    /// accumulates, on the reasoning that a wall which has been averaging for
    /// a minute should not be smeared by a filter tuned for one sample. That
    /// is right whenever the residual noise falls as `1/sqrt(n)` — the
    /// court's does — and wrong for a tier whose noise does not. The cove's
    /// glass is a rough dielectric behind twelve bounces: a handful of paths
    /// out of a thousand carry a hundred times the mean, so a pixel is still
    /// visibly speckled at two hundred samples and the fade has long since
    /// handed it back its own raw estimate.
    ///
    /// This is the floor under that fade: `0.0` (the default) is the court,
    /// unchanged; `k` leaves a converged pixel at `k` of the filtered value
    /// and `1 − k` of its own. Only the rune tier sets it, and it costs the
    /// filter's fixed pass on every resolve rather than only while the picture
    /// is young — the short-circuit that skips the filter once *every* pixel
    /// is converged cannot fire while a floor is asking for it.
    pub fn set_denoise_floor(&mut self, k: f32) {
        self.denoise_floor = k.clamp(0.0, 1.0);
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
                let held = self.count[i];
                let c = held + 1;
                self.count[i] = c;
                let k = 1.0 / c as f32;
                // The sample, scaled back to the pixel's own ceiling if it is
                // a firefly. See `set_firefly_cap`.
                let mut s = [
                    film.rgb[i * 3],
                    film.rgb[i * 3 + 1],
                    film.rgb[i * 3 + 2],
                ];
                if let Some(cap) = self.firefly_cap
                    && held >= FIREFLY_WARMUP
                {
                    let was = luminance(&self.mean[i * 3..i * 3 + 3]);
                    let now = luminance(&s);
                    let ceiling = cap * was.max(FIREFLY_FLOOR);
                    if now > ceiling && now > 0.0 {
                        let scale = ceiling / now;
                        s.iter_mut().for_each(|v| *v *= scale);
                    }
                }
                for c3 in 0..3 {
                    let m = self.mean[i * 3 + c3];
                    self.mean[i * 3 + c3] = m + (s[c3] - m) * k;
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
    ///
    /// [`Self::set_denoise_floor`] is the opt-out, for a tier whose noise does
    /// not fall the way that reasoning assumes.
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
        let floor = self.denoise_floor.clamp(0.0, 1.0);
        if opts.denoise && (floor > 0.0 || self.count.iter().any(|&c| c.max(1) < DENOISE_UNTIL)) {
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
                let k = if c >= DENOISE_UNTIL {
                    floor
                } else {
                    (1.0 - (c - 1) as f32 / (DENOISE_UNTIL - 1) as f32).max(floor)
                };
                if k <= 0.0 {
                    continue;
                }
                for c3 in 0..3 {
                    let raw = film.rgb[i * 3 + c3];
                    film.rgb[i * 3 + c3] = raw + (filtered.rgb[i * 3 + c3] - raw) * k;
                }
            }
        }
        film.to_srgb8(exposure, false)
    }
}

/// Rec. 709 luminance, the scalar a firefly is measured on.
fn luminance(rgb: &[f32]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
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
///
/// **The GPU tier no longer asks it.** That tier decides per pixel on the
/// device, from the previous camera and each instance's own motion, and a
/// rectangle around a ball is exactly the thing it exists not to draw. What
/// is left here is the CPU tier's own geometry — [`mask_rects`], which
/// [`History::plan`] still calls — and this wrapper, which the tests below
/// keep honest against it.
#[allow(dead_code)]
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

#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

// ---- the viewer's trait -------------------------------------------------------

/// The CPU tier's [`TemporalHistory`], and the reason the trait exists.
///
/// The viewer's contract is four calls in order — [`begin`] at this pass's
/// size, [`reproject`] with the camera it is about to render from and what
/// moved under it, [`accumulate`] the film that came back, [`resolve`] for
/// the glass — and this history does all four, so it fits without the trait
/// moving. Two notes on how the richer API underneath is folded into it:
///
/// - **the reprojection is in [`History::merge`]**, not in a pass of its own,
///   because a host-side reproject has to read *this* pass's guide buffers to
///   decide which pixels survive. So [`reproject`] records the view and the
///   poses the next [`accumulate`] will merge under, and answers with the
///   share of the screen the plan expects to keep. Every tier's `reproject`
///   is already "tell me where things are before you hand me the film"; this
///   one just does the work a step later.
/// - **the whole frame is traced.** The trait has no way to say "trace only
///   these rectangles", and a sim that wants the masked pass calls
///   [`History::plan`] and [`History::merge`] directly, which is what
///   `sims/rune/game.rs` does.
///
/// [`begin`]: TemporalHistory::begin
/// [`reproject`]: TemporalHistory::reproject
/// [`accumulate`]: TemporalHistory::accumulate
/// [`resolve`]: TemporalHistory::resolve
impl TemporalHistory for History {
    fn begin(&mut self, size: (u32, u32)) {
        self.resample(size);
    }

    fn reproject(&mut self, view: &View, poses: &[Pose]) -> f32 {
        let plan = self.plan(view, poses, &[]);
        self.pending = Some((*view, poses.to_vec()));
        1.0 - plan.coverage(self.size)
    }

    fn accumulate(&mut self, film: &Film) {
        let (view, poses) = match self.pending.take() {
            Some(p) => p,
            // No `reproject` this pass: the camera and the world are wherever
            // the last merge left them, which is the honest reading of "the
            // viewer did not say anything moved". With nothing merged yet
            // there is no history to carry either way, so any view will do
            // and the next `reproject` sees a camera move and starts over.
            None => (
                self.view.unwrap_or(View {
                    eye: Point3::new(0.0, 0.0, 0.0),
                    forward: Vec3::new(0.0, 0.0, -1.0),
                    right: Vec3::new(1.0, 0.0, 0.0),
                    up: Vec3::new(0.0, 1.0, 0.0),
                    half_w: 1.0,
                    half_h: 1.0,
                    width: film.width,
                    height: film.height,
                    projection: Projection::Rectilinear,
                }),
                self.poses.clone(),
            ),
        };
        self.merge(film, &view, &poses, &[], None);
    }

    fn resolve(&self, exposure: f32) -> Vec<u8> {
        History::resolve(self, exposure, &PathTraceOptions::default())
    }

    fn reset(&mut self) {
        *self = History::new(self.size);
    }

    fn mean_samples(&self) -> f32 {
        History::mean_samples(self)
    }
}

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

    // ---- the two maps, against the tracer's own rays --------------------

    /// A quad of half-size `half` on the plane through `at` with normal `n`,
    /// as a [`TriMesh`]. Big enough that every ray in the cone lands on it.
    fn quad(at: Point3, n: Vec3, half: f64) -> kosm_render::TriMesh {
        let n = n.normalize();
        // Any basis of the plane will do; a quad has no parameterisation
        // anything here reads.
        let hint = if n.z.abs() < 0.9 {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let t = n.cross(hint).normalize();
        let b = n.cross(t).normalize();
        let p = |a: f64, c: f64| at + t * (a * half) + b * (c * half);
        kosm_render::TriMesh::new(
            vec![p(-1.0, -1.0), p(1.0, -1.0), p(1.0, 1.0), p(-1.0, 1.0)],
            Vec::new(),
            &[0, 1, 2, 0, 2, 3],
        )
    }

    /// One plane, rendered: the distance from the eye along each pixel's
    /// primary ray. Zero where the ray missed.
    fn plane_depth(
        cam: &pathtrace::Camera,
        at: Point3,
        n: Vec3,
        w: u32,
        h: u32,
        seed: u64,
    ) -> Vec<f32> {
        let scene = kosm_render::pathtrace::Scene::<kosm_render::TriMesh> {
            objects: vec![kosm_render::pathtrace::Object::new(
                std::sync::Arc::new(kosm_render::Bvh::build(quad(at, n, 4.0e5))),
                kosm_render::pathtrace::Pbr::default(),
            )],
            lights: Vec::new(),
            env: kosm_render::pathtrace::Environment::default(),
            sun: None,
            ground: None,
            splats: None,
        };
        let opts = PathTraceOptions {
            spp: 1,
            seed,
            denoise: false,
            ..Default::default()
        };
        kosm_render::pathtrace::render(&scene, cam, w, h, &opts).depth
    }

    /// The direction the tracer actually fired through every pixel, solved
    /// out of three renders.
    ///
    /// `Film::depth` is the distance along the primary ray to the first hit,
    /// so a plane through `p0` with normal `n` gives one linear equation in
    /// the unknown direction: `d·n = (p0 − eye)·n / depth`. Three planes
    /// through the same point, with independent normals, give three — and
    /// the solve is the tracer's own ray, jitter and all. The jitter is the
    /// *same* in all three, because a pixel's sampler is seeded from the seed
    /// and the pixel and knows nothing about the camera, so the three
    /// equations describe one ray rather than three neighbouring ones.
    ///
    /// This is the only handle on `Camera::ray` from outside the crate — it
    /// is `pub(crate)` — and it is a better test than calling it would be:
    /// what has to agree is not the function but the picture it makes.
    fn traced_dirs(cam: &pathtrace::Camera, p0: Point3, w: u32, h: u32) -> Vec<Option<Vec3>> {
        // Three normals that span, all facing back towards a camera looking
        // along +y, and none of them so edge-on that a depth loses its
        // precision at the corner of the frame.
        let ns = [
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(-0.5, -1.0, 0.0),
            Vec3::new(0.0, -1.0, -0.5),
        ];
        let depths: Vec<Vec<f32>> = ns
            .iter()
            .map(|n| plane_depth(cam, p0, *n, w, h, 0x51de_face))
            .collect();
        let ns: Vec<Vec3> = ns.iter().map(|n| n.normalize()).collect();
        let cs: Vec<f64> = ns.iter().map(|n| (p0 - cam.eye).dot(n)).collect();
        let det3 = |m: [[f64; 3]; 3]| {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        };
        let rows = [
            [ns[0].x, ns[0].y, ns[0].z],
            [ns[1].x, ns[1].y, ns[1].z],
            [ns[2].x, ns[2].y, ns[2].z],
        ];
        let d = det3(rows);
        assert!(d.abs() > 1e-6, "the three planes have to span");
        (0..(w as usize) * (h as usize))
            .map(|i| {
                let b: Vec<f64> = (0..3)
                    .map(|k| cs[k] / depths[k][i].max(f32::MIN_POSITIVE) as f64)
                    .collect();
                if depths.iter().any(|dep| dep[i] <= 0.0) {
                    return None;
                }
                // Cramer, which is exact enough for a 3×3 whose rows are unit
                // vectors and whose determinant has just been checked.
                let sub = |col: usize| {
                    let mut m = rows;
                    for r in 0..3 {
                        m[r][col] = b[r];
                    }
                    det3(m) / d
                };
                Some(Vec3::new(sub(0), sub(1), sub(2)).normalize())
            })
            .collect()
    }

    /// Both maps, ray for ray, against what the tracer fired.
    ///
    /// The agreement that matters is not that `View` and `Camera` compute the
    /// same expression — they do not, one folds the half field of view into
    /// its extents — but that a pixel means the same direction to both. A
    /// history whose `ray_dir` disagreed with the generator by a pixel would
    /// smear its own past by a pixel every pass.
    ///
    /// The tolerance is a pixel and a half of angle, because the tracer's ray
    /// is jittered inside its pixel and this one is through its centre. The
    /// second half of the test is what gives the first half teeth: the *other*
    /// map is out by far more than that.
    #[test]
    fn both_projections_agree_with_the_tracers_own_primary_rays() {
        let (w, h) = (96u32, 64u32);
        let eye = Point3::new(0.0, 0.0, 0.0);
        let cam = pathtrace::Camera::look_at(
            eye,
            Point3::new(0.0, 1000.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            100.0,
        );
        let p0 = Point3::new(0.0, 1000.0, 0.0);

        for projection in [Projection::Rectilinear, Projection::Equidistant] {
            let cam = cam.with_projection(projection);
            let view = View::of(&cam, w, h);
            let other = View {
                projection: match projection {
                    Projection::Rectilinear => Projection::Equidistant,
                    Projection::Equidistant => Projection::Rectilinear,
                },
                ..View::of(
                    &cam.with_projection(match projection {
                        Projection::Rectilinear => Projection::Equidistant,
                        Projection::Equidistant => Projection::Rectilinear,
                    }),
                    w,
                    h,
                )
            };
            let fired = traced_dirs(&cam, p0, w, h);

            // A pixel of angle, vertically, under this map. The half-extent is
            // a tangent or an angle depending on the map, so it is measured
            // rather than read off.
            let px_rad = {
                let a = view.ray_dir(w / 2, h / 2);
                let b = view.ray_dir(w / 2, h / 2 + 1);
                a.dot(&b).clamp(-1.0, 1.0).acos()
            };
            let tol = 1.5 * px_rad;

            let mut worst = 0.0f64;
            let mut worst_other = 0.0f64;
            let mut tested = 0;
            for py in 0..h {
                for px in 0..w {
                    let Some(d) = fired[(py * w + px) as usize] else {
                        continue;
                    };
                    tested += 1;
                    let off = |v: &View| {
                        v.ray_dir(px, py).dot(&d).clamp(-1.0, 1.0).acos()
                    };
                    worst = worst.max(off(&view));
                    worst_other = worst_other.max(off(&other));
                }
            }
            assert!(
                tested > (w * h) as usize / 2,
                "{projection:?}: only {tested} pixels landed on all three planes"
            );
            assert!(
                worst < tol,
                "{projection:?}: worst disagreement {:.5} rad, over {:.5} rad ({:.2} pixels)",
                worst,
                tol,
                worst / px_rad
            );
            // The other map is out by five times the tolerance the right one
            // passes at — a quarter of a radian, fourteen degrees of picture.
            // Without this the first assertion could be passed by a `ray_dir`
            // that was merely *a* plausible map.
            assert!(
                worst_other > 5.0 * tol,
                "{projection:?}: the wrong map was only {:.5} rad out against {:.5} — \
                 this test has no teeth",
                worst_other,
                worst
            );
        }
    }

    /// A view records the map it was built under, and carries it across a
    /// size step. Cheap, and it is the thing every arm below rests on.
    #[test]
    fn a_view_carries_its_projection() {
        let cam = camera(Point3::new(0.0, -3000.0, 0.0));
        let flat = View::of(&cam, W, H);
        let fish = View::of(&cam.with_projection(Projection::Equidistant), W, H);
        assert_eq!(flat.projection, Projection::Rectilinear);
        assert_eq!(fish.projection, Projection::Equidistant);
        assert_ne!(flat, fish, "two maps are two views, whatever else matches");
        assert_eq!(fish.at_size(W * 2, H * 2).projection, Projection::Equidistant);
        // The pinhole keeps its tangent and the fisheye takes radians.
        assert!((flat.half_h - 22.5f64.to_radians().tan()).abs() < 1e-15);
        assert!((fish.half_h - 22.5f64.to_radians()).abs() < 1e-15);
    }

    /// The fisheye reprojects to the right pixels.
    ///
    /// A real scene, really rendered: a converged picture under an `f·θ`
    /// camera, a fifteen-degree turn, and one pass at the new camera. What
    /// the history hands back afterwards has to look like a picture *taken*
    /// at the new camera — so it is compared against one, converged from
    /// black over the same number of passes.
    ///
    /// The wall is a checkerboard on purpose. A smooth scene would pass this
    /// test with the reprojection deleted: two low-contrast pictures of the
    /// same wall from cameras fifteen degrees apart are already nearly the
    /// same bytes, and the comparison would be measuring nothing. High
    /// spatial frequency is what makes "the right pixel" and "a pixel nearby"
    /// two different answers.
    ///
    /// Only the pixels the history says it *carried* are compared, which is
    /// the claim under test; the strip that swung in from the edge holds one
    /// pass of raw samples and is noise, not evidence. And the comparison is
    /// against the same pixels of the picture it came *from*: what says the
    /// samples went to the right place is that they moved most of the way to
    /// the new camera's picture. Under a fisheye a pure yaw moves a pixel by
    /// an amount that depends on where in the frame it is, so a pinhole
    /// inverse — the map this `View` had until it learned there were two —
    /// fails exactly here.
    #[test]
    fn a_fisheye_reprojects_a_turn_onto_the_right_pixels() {
        let (w, h) = (96u32, 72u32);
        let eye = Point3::new(0.0, -2500.0, 400.0);
        let look = |yaw_deg: f64| {
            let (s, c) = yaw_deg.to_radians().sin_cos();
            pathtrace::Camera::look_at(
                eye,
                eye + Vec3::new(s * 2500.0, c * 2500.0, -400.0),
                Vec3::new(0.0, 0.0, 1.0),
                100.0,
            )
            .with_projection(Projection::Equidistant)
        };
        // A checkerboard wall, and one panel of it pulled forward so the
        // frame has depth in it as well as contrast.
        let cell = 900.0;
        let mut objects = Vec::new();
        for i in -4i32..=4 {
            for j in -3i32..=3 {
                let near = i == 1 && j == 0;
                let at = Point3::new(
                    i as f64 * 2.0 * cell,
                    if near { -600.0 } else { 2400.0 },
                    400.0 + j as f64 * 2.0 * cell,
                );
                let albedo = if (i + j).rem_euclid(2) == 0 {
                    [0.92, 0.90, 0.85]
                } else {
                    [0.04, 0.05, 0.07]
                };
                objects.push(kosm_render::pathtrace::Object::new(
                    std::sync::Arc::new(kosm_render::Bvh::build(quad(
                        at,
                        Vec3::new(0.0, -1.0, 0.0),
                        if near { cell * 0.5 } else { cell },
                    ))),
                    kosm_render::pathtrace::Pbr::plastic(albedo, 0.6, 0.0),
                ));
            }
        }
        let scene = kosm_render::pathtrace::Scene::<kosm_render::TriMesh> {
            objects,
            lights: Vec::new(),
            env: kosm_render::pathtrace::Environment::default(),
            sun: None,
            ground: None,
            splats: None,
        };
        // Low spp, as a live pass is: four samples and sixteen passes, which
        // is a picture that has converged enough to compare and nowhere near
        // enough to hide a misplaced sample.
        let pass = |cam: &pathtrace::Camera, k: u64| {
            let opts = PathTraceOptions {
                spp: 4,
                seed: 0xf1_5eed ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                denoise: false,
                ..Default::default()
            };
            kosm_render::pathtrace::render(&scene, cam, w, h, &opts)
        };
        // The resolve is the raw accumulated mean: the filter is a different
        // question and would only blur the one being asked here.
        let plain = PathTraceOptions { denoise: false, ..Default::default() };
        let converge = |cam: &pathtrace::Camera, n: u64| {
            let view = View::of(cam, w, h);
            let mut hist = History::new((w, h));
            for k in 0..n {
                hist.merge(&pass(cam, k), &view, &[], &[], None);
            }
            hist
        };

        let (a, b) = (look(0.0), look(15.0));
        let mut hist = converge(&a, 16);
        let stale = hist.resolve(1.0, &plain);
        let held: Vec<u32> = hist.count.clone();

        let vb = View::of(&b, w, h);
        hist.merge(&pass(&b, 99), &vb, &[], &[], None);
        let kept: Vec<bool> = hist
            .count
            .iter()
            .zip(&held)
            .map(|(&now, &was)| now > was && was > 0)
            .collect();
        let carried = kept.iter().filter(|&&k| k).count();
        assert!(
            carried as f64 > 0.5 * (w * h) as f64,
            "a fifteen-degree turn should carry most of the frame, carried {carried} of {}",
            w * h
        );

        let want = converge(&b, 16).resolve(1.0, &plain);
        let got = hist.resolve(1.0, &plain);
        // Mean channel difference over the carried pixels alone.
        let over = |x: &[u8], y: &[u8]| {
            let mut sum = 0.0;
            let mut n = 0.0;
            for (i, _) in kept.iter().enumerate().filter(|&(_, &k)| k) {
                for c in 0..3 {
                    sum += (x[i * 4 + c] as f64 - y[i * 4 + c] as f64).abs();
                    n += 1.0;
                }
            }
            sum / n
        };
        let reprojected = over(&got, &want);
        let unmoved = over(&stale, &want);
        assert!(
            reprojected < 0.35 * unmoved,
            "the reprojection should land on the new camera's picture: \
             {reprojected:.2} of 255 against {unmoved:.2} for the picture it came from"
        );
        assert!(
            reprojected < 16.0,
            "reprojected pixels are {reprojected:.2} of 255 from a picture taken there"
        );
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

    /// The two opt-in fields are off by default, which is the whole promise
    /// the court is owed: a `History` nobody has spoken to behaves exactly as
    /// it did before either existed.
    #[test]
    fn the_new_knobs_are_off_until_they_are_asked_for() {
        let cam = camera(Point3::new(0.0, -3000.0, 0.0));
        let view = View::of(&cam, W, H);
        let poses = [Pose::still([0.0, 0.0, 500.0], 120.0)];
        let dim = plane_film(&view, 1.0);
        let mut spike = plane_film(&view, 1.0);
        spike.rgb[..3].copy_from_slice(&[500.0; 3]);

        // Default: the spike is folded in whole, and a converged pixel is left
        // alone by the filter.
        let mut plain = History::new((W, H));
        // …and the same history with both knobs set.
        let mut tuned = History::new((W, H));
        tuned.set_firefly_cap(Some(4.0));
        tuned.set_denoise_floor(1.0);
        for k in 0..(DENOISE_UNTIL + 8) {
            let film = if k == DENOISE_UNTIL + 4 { &spike } else { &dim };
            plain.merge(film, &view, &poses, &[], None);
            tuned.merge(film, &view, &poses, &[], None);
        }
        // The uncapped mean carries the spike; the capped one is near the four
        // times its own mean it was allowed.
        assert!(plain.mean[0] > 10.0, "the spike went in whole: {}", plain.mean[0]);
        assert!(tuned.mean[0] < 1.2, "the cap held it back: {}", tuned.mean[0]);

        let opts = PathTraceOptions { denoise: true, ..Default::default() };
        let plain_px = plain.resolve(1.0, &opts);
        // With no floor a converged pixel is its own mean, filter or no filter.
        let raw = History { denoise_floor: 0.0, ..clone_of(&plain) }
            .resolve(1.0, &PathTraceOptions { denoise: false, ..opts.clone() });
        assert_eq!(plain_px, raw, "the default resolve leaves a converged pixel alone");
    }

    /// A copy of a history, so a resolve can be compared against another
    /// resolve of the same samples under different knobs.
    fn clone_of(h: &History) -> History {
        History {
            size: h.size,
            mean: h.mean.clone(),
            alpha: h.alpha.clone(),
            count: h.count.clone(),
            normal: h.normal.clone(),
            depth: h.depth.clone(),
            albedo: h.albedo.clone(),
            variance: h.variance.clone(),
            view: h.view,
            poses: h.poses.clone(),
            mask_fraction: h.mask_fraction,
            denoise_floor: h.denoise_floor,
            firefly_cap: h.firefly_cap,
            pending: None,
        }
    }

    /// A floor keeps the filter engaged on a picture every pixel of which has
    /// long since passed [`DENOISE_UNTIL`] — which is the state a still window
    /// reaches in a couple of seconds and the state the cove's glass is still
    /// speckled in.
    #[test]
    fn a_denoise_floor_keeps_the_filter_on_past_the_fade() {
        let cam = camera(Point3::new(0.0, -3000.0, 0.0));
        let view = View::of(&cam, W, H);
        let poses = [Pose::still([0.0, 0.0, 500.0], 120.0)];
        let flat = plane_film(&view, 0.2);
        let mut h = History::new((W, H));
        for _ in 0..(DENOISE_UNTIL + 8) {
            h.merge(&flat, &view, &poses, &[], None);
        }
        // One pixel that disagrees with a converged, otherwise uniform field,
        // kept well under the tonemap's shoulder so the bytes still move.
        let hot = (H / 2 * W + W / 2) as usize;
        h.mean[hot * 3..hot * 3 + 3].copy_from_slice(&[0.6; 3]);
        // The filter's luminance weight is scaled by the variance plane, and
        // a film of zeros tells it every pixel is exact — which is a real
        // failure mode of this tier and not the one under test here.
        h.variance.iter_mut().for_each(|v| *v = 0.05);
        let opts = PathTraceOptions { denoise: true, ..Default::default() };

        let mut with_floor = clone_of(&h);
        with_floor.set_denoise_floor(1.0);
        let a = h.resolve(1.0, &opts);
        let b = with_floor.resolve(1.0, &opts);
        assert!(
            b[hot * 4] < a[hot * 4],
            "the floor pulled the outlier down: {} against {}",
            b[hot * 4],
            a[hot * 4]
        );
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
