//! What the last frame already knew, behind one trait.
//!
//! The path tracer is asked for a sample a pixel a pass, which on its own is a
//! blizzard. What makes that watchable is not spending more: it is refusing to
//! throw the previous frame away. Every tier does that somewhere — the GPU
//! tier in device buffers, through `kosm_render::gpu`'s history and denoise
//! passes; a CPU tier, when one comes back, in host memory — and the viewer
//! should not care which.
//!
//! So the viewer talks to [`TemporalHistory`]: begin a frame, carry the
//! history across whatever moved, fold this pass in, resolve for the glass,
//! reset. The CPU temporal history that used to live in `kosm_view::history`
//! is gone; it predated the render port and was a second copy of what
//! `kosm_render::gpu` already does properly, on the device, with reprojection
//! and an à-trous filter this side never had.
//!
//! [`kosm_render::gpu::History`] — the running mean, count and variance the
//! device hands back — implements the trait, and that implementation is the
//! whole of the CPU tier's accumulation now: a per-pixel running mean, reset
//! by any camera or world move, and no filter. It is what a tier with no
//! adapter gets. A real CPU history can grow behind the same trait later
//! without the viewer changing a line.

use kosm_render::gpu::History;
use vcad_kernel_raytrace::pathtrace::{self, Film};

/// How a camera maps the screen onto directions, re-exported so a caller can
/// name [`View`]'s own field without reaching past this crate for it. It is
/// `kosm_render`'s: there is one answer to what a pixel means, and the
/// renderer owns it.
pub use kosm_render::pathtrace::Projection;

/// A pose that moved by less than this, in millimetres, did not move.
const MOVED_MM: f64 = 1.0;

/// …nor did one whose rotation matrix changed by less than this per element.
const TURNED: f64 = 1e-4;

/// A view: where the eye is, the screen basis, the half-extents of the frame
/// and the map those half-extents are measured in. Built from
/// `pathtrace::Camera` and a size, which is exactly what `Camera::ray` uses,
/// so pixel centres agree to the float.
///
/// ## the half-extents are in the projection's own units
///
/// This used to be a pinhole frustum and nothing else: `half_h` was
/// `tan(fov/2)`, `half_w` that times the aspect, and
/// [`crate::history`]'s `project` divided by the forward depth. That is the
/// `r = f·tan θ` map written out, and it is the reason
/// [`Projection::Equidistant`] was offline-only — a fisheye frame handed to a
/// history that reprojects through a pinhole smears its own past across the
/// frame.
///
/// So the half-extents are now stated in whatever unit the map measures
/// screen radius in — **a tangent for the pinhole, radians for the fisheye** —
/// and both maps read them the same way. Screen coordinates `(sx, sy)` in
/// `[-1, 1]`:
///
/// - [`Projection::Rectilinear`]: the direction is
///   `forward + right·(sx·half_w) + up·(sy·half_h)`, with `half_h =
///   tan(fov/2)`. Unmoved, to the bit.
/// - [`Projection::Equidistant`]: `(u, v) = (sx·half_w, sy·half_h)` are the
///   *angle* offsets, `θ = |(u, v)|` is the angle from the forward axis and
///   `(u, v)/θ` its azimuth, with `half_h = fov/2` in radians. That is
///   `r = f·θ` with `f = 1`, which is `kosm_render`'s generator with the
///   `half_fov` folded into the extents rather than multiplied at the end.
///
/// Both agree at the vertical edge of the frame, because
/// `kosm_render::Projection` says they must: `sy = 1` is `fov/2` off the axis
/// under either. What differs is everything between.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct View {
    pub eye: vcad_kernel_math::Point3,
    pub forward: vcad_kernel_math::Vec3,
    pub right: vcad_kernel_math::Vec3,
    pub up: vcad_kernel_math::Vec3,
    pub half_w: f64,
    pub half_h: f64,
    pub width: u32,
    pub height: u32,
    /// How `(sx, sy)` becomes a direction — and how a direction becomes
    /// `(sx, sy)` again. Part of the identity of a view: a camera that
    /// changed its map moved every pixel, whatever else stayed still.
    pub projection: Projection,
}

impl View {
    pub fn of(cam: &pathtrace::Camera, width: u32, height: u32) -> Self {
        let half = cam.fov_deg.to_radians() * 0.5;
        let half_h = match cam.projection {
            // `tan(fov/2)`: the expression this was before the map was a
            // choice, so every rectilinear view is the one it always was.
            Projection::Rectilinear => half.tan(),
            // Radians. `tan` is not merely wrong here, it is unbounded — a
            // 180° fisheye is a perfectly good camera and `tan(90°)` is not a
            // number a history can compare.
            Projection::Equidistant => half,
        };
        Self {
            eye: cam.eye,
            forward: cam.forward,
            right: cam.right,
            up: cam.up,
            half_w: half_h * (width as f64 / height as f64),
            half_h,
            width,
            height,
            projection: cam.projection,
        }
    }

    /// The same eye and the same frame, sampled at a different raster size.
    ///
    /// The vertical extent is the map's and does not depend on the raster;
    /// only the horizontal one, which is the vertical times the aspect. True
    /// of both maps, which is what lets this stay one line.
    pub fn at_size(&self, width: u32, height: u32) -> Self {
        Self {
            half_w: self.half_h * (width as f64 / height as f64),
            width,
            height,
            ..*self
        }
    }
}

/// Where one thing that moves is, in millimetres.
///
/// The motion a history is told about: a sphere is enough, because what a
/// history wants to know is not the shape but whether the pixels it covers
/// still mean what they meant.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Pose {
    pub centre: [f64; 3],
    pub rot: [f64; 9],
    pub radius: f64,
}

impl Pose {
    /// A pose that is not turning. For tests and for callers with no rotation.
    pub fn still(centre: [f64; 3], radius: f64) -> Self {
        Self {
            centre,
            rot: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            radius,
        }
    }

    /// Did this pose change enough to be worth a repaint? A millimetre of
    /// travel, or any turn a seam would show.
    pub fn differs(&self, other: &Pose) -> bool {
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

/// A temporal history, whichever side of the bus it lives on.
///
/// The order the viewer calls these in is the design: [`begin`] at this pass's
/// size, [`reproject`] with the camera it is about to render from and what
/// moved under it, [`accumulate`] the film that came back, [`resolve`] for the
/// glass. [`reset`] throws the picture away.
///
/// [`begin`]: TemporalHistory::begin
/// [`reproject`]: TemporalHistory::reproject
/// [`accumulate`]: TemporalHistory::accumulate
/// [`resolve`]: TemporalHistory::resolve
/// [`reset`]: TemporalHistory::reset
pub trait TemporalHistory {
    /// Start a pass at `size`. A size step is a new grid; an implementation
    /// that cannot carry its picture across one starts over here.
    fn begin(&mut self, size: (u32, u32));

    /// Carry the history into `view`, given where everything that moves now
    /// is. Returns the share of the screen that survived, in `0..=1`.
    fn reproject(&mut self, view: &View, poses: &[Pose]) -> f32;

    /// Fold one pass's film into the running mean.
    fn accumulate(&mut self, film: &Film);

    /// The picture so far, as sRGB bytes, `width * height * 4` of them.
    fn resolve(&self, exposure: f32) -> Vec<u8>;

    /// Forget every pixel.
    fn reset(&mut self);

    /// Samples per pixel, averaged over the screen: the number that says
    /// whether the picture is converging.
    fn mean_samples(&self) -> f32;
}

/// The device's own history struct, accumulated on the host.
///
/// [`kosm_render::gpu::History`] is the plane layout both tiers speak — mean
/// radiance, coverage, per-pixel count, variance of the mean — so a tier with
/// no adapter keeps its picture in exactly the shape a tier with one reads
/// back. There is no reprojection and no filter here: a camera or world move
/// starts the picture over, which is the honest fallback, and the à-trous and
/// neural filters stay where they belong, in `kosm_render::gpu`.
impl TemporalHistory for History {
    fn begin(&mut self, (w, h): (u32, u32)) {
        if (self.width, self.height) == (w, h) {
            return;
        }
        let n = (w as usize) * (h as usize);
        self.width = w;
        self.height = h;
        self.rgb = vec![0.0; n * 3];
        self.alpha = vec![0.0; n];
        self.count = vec![0; n];
        self.variance = vec![0.0; n];
    }

    fn reproject(&mut self, view: &View, poses: &[Pose]) -> f32 {
        let still = LAST.with(|last| {
            let mut last = last.borrow_mut();
            let same = last
                .as_ref()
                .is_some_and(|(v, p)| v == view && p.len() == poses.len() && !p
                    .iter()
                    .zip(poses)
                    .any(|(a, b)| a.differs(b)));
            *last = Some((*view, poses.to_vec()));
            same
        });
        if still {
            return 1.0;
        }
        self.reset();
        0.0
    }

    fn accumulate(&mut self, film: &Film) {
        if (film.width, film.height) != (self.width, self.height) {
            self.begin((film.width, film.height));
        }
        for i in 0..(self.count.len()) {
            let n = self.count[i] as f32 + 1.0;
            for c in 0..3 {
                let mean = self.rgb[i * 3 + c];
                self.rgb[i * 3 + c] = mean + (film.rgb[i * 3 + c] - mean) / n;
            }
            let a = self.alpha[i];
            self.alpha[i] = a + (film.alpha[i] - a) / n;
            let v = self.variance[i];
            self.variance[i] = v + (film.variance[i] - v) / n;
            self.count[i] += 1;
        }
    }

    fn resolve(&self, exposure: f32) -> Vec<u8> {
        let n = (self.width as usize) * (self.height as usize);
        let film = Film {
            width: self.width,
            height: self.height,
            rgb: self.rgb.clone(),
            alpha: self.alpha.clone(),
            normal: vec![0.0; n * 3],
            depth: vec![0.0; n],
            albedo: vec![0.0; n * 3],
            variance: self.variance.clone(),
        };
        film.to_srgb8(exposure, false)
    }

    fn reset(&mut self) {
        self.rgb.fill(0.0);
        self.alpha.fill(0.0);
        self.count.fill(0);
        self.variance.fill(0.0);
    }

    fn mean_samples(&self) -> f32 {
        if self.count.is_empty() {
            return 0.0;
        }
        self.count.iter().map(|&c| c as f64).sum::<f64>() as f32 / self.count.len() as f32
    }
}

thread_local! {
    /// The view and the poses the last [`TemporalHistory::reproject`] on this
    /// thread was told about.
    ///
    /// [`History`] is a plain data struct — it is what comes off the device —
    /// so the one bit of state this implementation needs beyond its planes
    /// lives beside it rather than in it. A viewer runs its history on one
    /// render thread, which is what makes that sound.
    static LAST: std::cell::RefCell<Option<(View, Vec<Pose>)>> =
        const { std::cell::RefCell::new(None) };
}

/// A history of `size` with nothing in it: what a tier starts a picture from.
pub fn empty(size: (u32, u32)) -> History {
    let mut h = History {
        width: 0,
        height: 0,
        rgb: Vec::new(),
        alpha: Vec::new(),
        count: Vec::new(),
        variance: Vec::new(),
    };
    h.begin(size);
    h
}
