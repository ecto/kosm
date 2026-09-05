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
//! ## a pass without guides
//!
//! The GPU tier has no guide buffers to hand over: the compute shader writes
//! depth and normals into a buffer it allocates itself, does not mark it
//! `COPY_SRC` and does not return it. So [`History::merge`] takes a
//! [`Guides`] flag, and a pass that says [`Guides::None`] is merged on the
//! strength of the mask alone — which still works, because the mask is
//! *geometric*: it is computed from where the balls and the net are and where
//! their shadows fall, not from anything the tracer measured. What is lost is
//! reprojection. A pixel cannot be carried into a moved camera without knowing
//! how far away it was, so a camera that moves throws the whole picture away
//! rather than most of it, and the à-trous filter — which passes a pixel
//! through untouched wherever `depth` is zero — becomes a no-op. The GPU tier
//! is therefore sharp and unfiltered at one sample, and starts over on an
//! orbit; the CPU tier is neither.
//!
//! ## what is not here
//!
//! vcad's `pathtrace::render` renders a whole frame: it splits `rgb` into
//! scanline chunks with rayon and walks every pixel of every row. There is no
//! sub-rectangle entry point, no pixel-list entry point, and no per-pixel
//! public function — `radiance` is private, and `cpu.rs` is the *other*
//! renderer (a studio rasteriser), not a tap into this one. So the mask cannot
//! yet buy fewer rays; it only buys which samples survive. The next multiplier
//! would be a `render_into(&mut Film, &[Rect])` beside `render` in
//! `crates/vcad-kernel-raytrace/src/pathtrace.rs`, taking the same
//! `par_chunks_mut` loop and skipping pixels outside the rects — an hour's
//! work there, and a five-to-ten times cut in rays here whenever the mask is
//! the ten per cent of the screen a bouncing ball actually is. Not this
//! change: vcad is not ours to edit today.

use vcad_kernel_math::{Point3, Vec3};
use vcad_kernel_raytrace::pathtrace::{self, Film, PathTraceOptions};

/// Whether the pass being merged brought depth, normals and albedo with it.
///
/// The CPU integrator fills all three; the GPU tracer returns colour alone.
/// Without them there is nothing to reproject through and nothing for the
/// denoiser to stop on, so a moved camera invalidates everything.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Guides {
    /// `film.depth`, `film.normal` and `film.albedo` are the tracer's.
    Film,
    /// Colour only.
    None,
}

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
        Self { x0: 0, y0: 0, x1: w, y1: h }
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
        Self { centre, rot: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], radius }
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
        self.rot.iter().zip(&other.rot).any(|(a, b)| (a - b).abs() > TURNED)
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

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    pub fn mask_fraction(&self) -> f32 {
        self.mask_fraction
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
    pub fn merge(&mut self, film: &Film, guides: Guides, view: &View, poses: &[Pose], lights: &[Point3]) {
        if self.size != (film.width, film.height) {
            *self = History::new((film.width, film.height));
        }
        let n = (self.size.0 as usize) * (self.size.1 as usize);

        // A pixel is live if it has history that survived both tests.
        let mut live: Vec<bool> = self.count.iter().map(|&c| c > 0).collect();
        match self.view {
            // A moved camera, with depth to unproject through: carry what
            // reprojects. Without it, nothing can be carried at all.
            Some(old) if old != *view => match guides {
                Guides::Film => self.reproject(&old, view, film, &mut live),
                Guides::None => live.iter_mut().for_each(|l| *l = false),
            },
            None => live.iter_mut().for_each(|l| *l = false),
            _ => {}
        }

        let masked = self.paint_mask(view, poses, lights, &mut live);
        self.mask_fraction = masked as f32 / n.max(1) as f32;

        for i in 0..n {
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
        // the picture is of *now*, and the next reprojection reads them. A
        // pass with none leaves them zeroed, which is the background sentinel
        // — the denoiser passes such a pixel through, so it becomes a no-op
        // rather than a blur with nothing to stop it.
        if guides == Guides::Film {
            self.normal.copy_from_slice(&film.normal);
            self.depth.copy_from_slice(&film.depth);
            self.albedo.copy_from_slice(&film.albedo);
        }
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

    /// Kill the pixels the world moved under, and say how many.
    ///
    /// For each pose that changed: the bounding sphere at the old pose and at
    /// the new one, and — because a ball's shadow is as visibly wrong as the
    /// ball — the disc that sphere casts on the floor from each panel. The
    /// shadow is a cone; the disc where it meets `z = 0` is approximated by
    /// its axis (the light-through-centre line, met with the floor) and a
    /// radius scaled by how much further the floor is than the ball.
    fn paint_mask(&self, view: &View, poses: &[Pose], lights: &[Point3], live: &mut [bool]) -> usize {
        let (w, h) = self.size;
        let mut rects: Vec<Rect> = Vec::new();
        let mut push = |r: Option<Rect>| {
            if let Some(r) = r {
                rects.push(r);
            }
        };
        // Bodies are appended — the court drops its balls in over several
        // seconds — so a changed count is not a reason to repaint the whole
        // screen. The shared prefix is compared pairwise; anything past the
        // end of either list appeared or left and is masked on its own.
        let screen = (w as usize) * (h as usize);
        let paint = |p: &Pose, push: &mut dyn FnMut(Option<Rect>)| {
            let body = view.sphere_rect(p.point(), p.radius);
            if let Some(r) = &body {
                let area = ((r.x1 - r.x0) as usize) * ((r.y1 - r.y0) as usize);
                if area * 5 > screen * 3 && std::env::var_os("KOSM_MASK_DEBUG").is_some() {
                    eprintln!("mask   a pose covers {}% of the screen: centre {:?} radius {:.0} mm", 100 * area / screen.max(1), p.centre, p.radius);
                }
            }
            push(body);
            for light in lights {
                if let Some((c, r)) = shadow_disc(*light, p.point(), p.radius) {
                    let disc = view.sphere_rect(c, r);
                    if let Some(rc) = &disc {
                        let area = ((rc.x1 - rc.x0) as usize) * ((rc.y1 - rc.y0) as usize);
                        if area * 5 > screen * 3 && std::env::var_os("KOSM_MASK_DEBUG").is_some() {
                            eprintln!("mask   a shadow covers {}%: light {:?} pose {:?} r {:.0} → disc r {:.0} mm", 100 * area / screen.max(1), light, p.centre, p.radius, r);
                        }
                    }
                    push(disc);
                }
            }
        };
        let shared = poses.len().min(self.poses.len());
        for k in 0..shared {
            if !poses[k].differs(&self.poses[k]) {
                continue;
            }
            paint(&poses[k], &mut push);
            paint(&self.poses[k], &mut push);
        }
        for p in poses.iter().skip(shared).chain(self.poses.iter().skip(shared)) {
            paint(p, &mut push);
        }

        // The union, counted once — overlapping rectangles are one mask.
        let mut masked = vec![false; (w as usize) * (h as usize)];
        for r in rects {
            for py in r.y0..r.y1 {
                for px in r.x0..r.x1 {
                    masked[(py * w + px) as usize] = true;
                }
            }
        }
        let _ = h;
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
    /// filter tuned for one sample.
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
        if opts.denoise {
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

// ---- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 32;
    const H: u32 = 24;

    fn camera(eye: Point3) -> pathtrace::Camera {
        pathtrace::Camera::look_at(eye, Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 45.0)
    }

    /// A film whose every pixel hits the plane `y = 0`, facing the camera —
    /// the depth is whatever that plane is at, which is what a reprojection
    /// has to reproduce.
    fn plane_film(view: &View, value: f32) -> Film {
        let n = (W * H) as usize;
        let mut film = Film {
            width: W,
            height: H,
            rgb: vec![value; n * 3],
            alpha: vec![1.0; n],
            normal: vec![0.0; n * 3],
            depth: vec![0.0; n],
            albedo: vec![0.5; n * 3],
            variance: vec![0.0; n],
        };
        for py in 0..H {
            for px in 0..W {
                let i = (py * W + px) as usize;
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
            h.merge(&film, Guides::Film, &view, &poses, &[]);
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

    #[test]
    fn a_moved_ball_invalidates_only_its_mask() {
        let cam = camera(Point3::new(0.0, -3000.0, 0.0));
        let view = View::of(&cam, W, H);
        let film = plane_film(&view, 1.0);
        let mut h = History::new((W, H));
        let at = |z: f64| [Pose::still([0.0, 0.0, z], 100.0)];
        for _ in 0..8 {
            h.merge(&film, Guides::Film, &view, &at(0.0), &[]);
        }
        let before: Vec<u32> = h.count.clone();
        assert!(before.iter().all(|&c| c == 8));

        // Move it half a metre: a real move, but a small one on screen.
        h.merge(&film, Guides::Film, &view, &at(500.0), &[]);
        let f = h.mask_fraction();
        assert!(f > 0.0 && f < 0.9, "the mask should be a patch, not the screen: {f}");
        let inside = h.count.iter().filter(|&&c| c == 1).count();
        let outside = h.count.iter().filter(|&&c| c == 9).count();
        assert_eq!(inside + outside, (W * H) as usize);
        assert!(inside > 0 && outside > 0);
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
            h.merge(&plane_film(&va, 1.0), Guides::Film, &va, &poses, &[]);
        }
        let b = pathtrace::Camera::look_at(
            Point3::new(200.0, -3000.0, 0.0),
            Point3::new(200.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            45.0,
        );
        let vb = View::of(&b, W, H);
        h.merge(&plane_film(&vb, 1.0), Guides::Film, &vb, &poses, &[]);

        // The strip that slid in from the edge is new; the rest is carried.
        let kept = h.count.iter().filter(|&&c| c == 9).count();
        assert!(
            kept as f64 > 0.75 * (W * H) as f64,
            "a pure pan should keep most of the frame, kept {kept} of {}",
            W * H
        );
    }

    #[test]
    fn a_shadow_grows_with_the_throw() {
        let light = Point3::new(0.0, 0.0, 6000.0);
        let (c, r) = shadow_disc(light, Point3::new(0.0, 0.0, 3000.0), 120.0).unwrap();
        assert!(c.z.abs() < 1e-9);
        // Twice as far as the ball, so twice the radius.
        assert!((r - 240.0).abs() < 1e-6, "{r}");
    }
}
