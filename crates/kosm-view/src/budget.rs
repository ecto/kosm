//! What a pass may cost, and what to spend it on: pretty when we can,
//! otherwise fast.
//!
//! A live CPU path tracer is asked for one raw sample a pixel a pass, and what
//! that costs is a fact about the machine, not about the picture. On this one
//! the cove is sixty-five to a hundred and fifty milliseconds at 480×270 — so
//! a player walking gets seven to fifteen passes a second, and a player
//! standing still gets a picture that converges. Those are two different
//! wants, and one fixed size cannot serve both:
//!
//! - **walking**, the frame is a moving window on a moving world and every
//!   pixel is two samples old. Resolution buys nothing there; *rate* does.
//!   The temporal history carries each pixel through the moved eye, so a
//!   quarter-size pass at four times the rate does not read as a blocky
//!   picture — it reads as motion blur, which is what a moving camera should
//!   look like.
//! - **standing still**, nothing is being thrown away: the history is
//!   accumulating and every extra pixel is a pixel that will converge. A still
//!   frame can afford a size a moving one cannot, and it can afford one *above*
//!   the nominal size — the ceiling here is patience, not smoothness.
//!
//! So [`Budget`] is a small pure function of two measurements: what the last
//! pass cost, and whether anything moved under it. It owns no picture, no
//! clock and no threads; a render worker calls [`Budget::next`] once a pass
//! and renders at whatever comes back.
//!
//! ```
//! use kosm_view::budget::Budget;
//! // 480×270 nominal, 960×540 when a still frame can afford it, 40 ms while
//! // moving, 250 ms for a still.
//! let mut b = Budget::new((480, 270), (960, 540), 40.0, 250.0);
//! assert_eq!(b.size(), (480, 270));
//! // A hundred and twenty milliseconds with the whole frame repainting: the
//! // policy steps down rather than hold a size the machine cannot pace.
//! let next = b.next(120.0, 1.0, true);
//! assert!(next.0 < 480, "a slow walking pass should have stepped down");
//! ```
//!
//! ## the ladder, and why it is quantised
//!
//! Every size step resamples the temporal history onto a new grid — cheap
//! (`History::resample` is a couple of bilinear passes) but not free, and a
//! continuously-varying size would pay it every pass forever. So the sizes are
//! a short ladder: [`SCALES`] of the base size, plus one rung above it for the
//! pretty size. The floor is a quarter, which is a sixteenth of the rays and
//! is where a low-res frame stops reading as blur and starts reading as
//! blocks.
//!
//! ## the cost model
//!
//! `ms = fixed + per · pixels`. One term is not enough: a pass has a floor
//! that does not scale with the picture at all — the à-trous resolve, the
//! caustic map, forking a rayon pool — and folding that into the slope
//! over-charges a big picture, under-charges a small one, and makes the
//! policy oscillate. Two exponential moving averages, one at the cheap end of
//! whatever has been asked for and one at the dear end, are the two points the
//! line goes through; until they are [`SPREAD`] apart in pixels the model
//! degenerates to a one-term fit through the origin, which is what a single
//! measurement can honestly say.
//!
//! The averaging is also the whole of the noise robustness. A single slow
//! pass — a caustic retrace, the OS scheduling somebody else — moves the model
//! by a third of the way, not all of it, so the policy does not flap on it.
//!
//! ## hysteresis
//!
//! Down is immediate and up is earned. A size whose *predicted* cost is over
//! the target is abandoned on the next pass, because the alternative is a
//! window that stutters; a larger size is taken only when it is predicted
//! under [`UP_MARGIN`] of the target, and only after [`UP_HOLD`] consecutive
//! passes have said so. The two conditions are a rung apart in pixels — a rung
//! is about 1.8× the pixels of the one below it — so there is no timing, noisy
//! or otherwise, that satisfies both.
//!
//! The pretty rung has the same deadband against the *ceiling* and it needs
//! it more, because there is no rung between it and the base for the two
//! conditions to straddle. It is taken at [`UP_MARGIN`] of the ceiling and
//! given back only over the ceiling itself, so a machine whose still pass
//! lands between the two keeps whichever rung it is already on. Measured
//! before that margin existed: a cove whose still pass cost 216 to 292 ms
//! against a 250 ms ceiling changed size on nine of seventy passes, which is
//! the picture visibly blinking between two resolutions while nothing moves.
//!
//! While still the policy never steps *down* the ladder: a converged picture
//! is the one thing worth keeping, and the cost of a still pass is nobody's
//! problem. Giving the pretty rung back is not a step down that ladder — it
//! is handing back a rung that was never nominal.

/// The ladder, as fractions of the base size. Largest first.
///
/// Five rungs over a factor of four. Below a quarter the picture stops reading
/// as motion blur and starts reading as blocks, and the blit's linear upscale
/// cannot hide the difference.
pub const SCALES: [f64; 5] = [1.0, 0.75, 0.5, 0.375, 0.25];

/// The rung [`SCALES`] calls full size — the index the pretty rung sits above
/// and the moving rungs sit below.
const BASE: usize = 1;

/// How much of the frame may be repainting and still count as still.
///
/// Not zero: a level with anything moving in it — a swell on the sea, a door
/// on its hinge — never has an empty mask, and waiting for one would pin the
/// picture at its base size forever. What "still" really means is that almost
/// nothing is being thrown away.
pub const QUIET: f32 = 0.05;

/// How far under the target a larger rung has to be predicted before it is
/// worth taking.
const UP_MARGIN: f64 = 0.8;

/// Consecutive passes that have to agree before the picture grows.
const UP_HOLD: u32 = 3;

/// How much of a new measurement each bucket takes.
const EWMA: f64 = 0.35;

/// Two measurements have to be this far apart in pixels before they count as
/// two points and not one noisy one.
const SPREAD: f64 = 1.5;

/// The cost of a pass before one has been timed, in milliseconds per
/// megapixel. Deliberately pessimistic: the first picture should be small and
/// quick, not right.
const GUESS: f64 = 900.0;

/// The smallest picture any rung may be, on each axis.
const FLOOR: u32 = 32;

/// What a pass costs: a fixed part and a per-megapixel part, fitted from the
/// cheap end and the dear end of what has actually been asked for.
#[derive(Clone, Copy, Debug, Default)]
struct Cost {
    /// `(megapixels, milliseconds)`, exponentially averaged.
    lo: Option<(f64, f64)>,
    hi: Option<(f64, f64)>,
}

impl Cost {
    /// Fold one measurement in, into whichever bucket it belongs to.
    fn observe(&mut self, mpx: f64, ms: f64) {
        if !(mpx.is_finite() && ms.is_finite()) || mpx <= 0.0 || ms < 0.0 {
            return;
        }
        let near = |b: &(f64, f64)| (mpx / b.0).max(b.0 / mpx) <= SPREAD;
        if let Some(lo) = &mut self.lo {
            if near(lo) {
                lo.0 = mpx;
                lo.1 += EWMA * (ms - lo.1);
                return;
            }
        }
        if let Some(hi) = &mut self.hi {
            if near(hi) {
                hi.0 = mpx;
                hi.1 += EWMA * (ms - hi.1);
                return;
            }
        }
        // A size neither bucket owns: it is a new end of the range, or it is
        // between them, in which case it replaces whichever end it is nearer
        // to and leaves the pair as far apart as it can.
        match (self.lo, self.hi) {
            (None, _) => self.lo = Some((mpx, ms)),
            (Some(lo), None) => {
                if mpx < lo.0 {
                    self.hi = Some(lo);
                    self.lo = Some((mpx, ms));
                } else {
                    self.hi = Some((mpx, ms));
                }
            }
            (Some(lo), Some(hi)) => {
                if mpx < lo.0 || mpx / lo.0 < hi.0 / mpx {
                    self.lo = Some((mpx, ms));
                } else {
                    self.hi = Some((mpx, ms));
                }
            }
        }
    }

    /// What a pass of `mpx` megapixels would cost.
    fn predict(&self, mpx: f64) -> f64 {
        match (self.lo, self.hi) {
            (Some(lo), Some(hi)) if hi.0 / lo.0 >= SPREAD => {
                let per = (hi.1 - lo.1) / (hi.0 - lo.0);
                // A dear point that measured *cheaper* than a cheap one is
                // noise, not a slope: fall back to the through-origin fit
                // through the dearer of the two, which is the safe side.
                if per > 0.0 {
                    (lo.1 - per * lo.0).max(0.0) + per * mpx
                } else {
                    hi.1 / hi.0 * mpx
                }
            }
            (Some(lo), None) => lo.1 / lo.0 * mpx,
            (Some(lo), Some(hi)) => hi.1.max(lo.1) / hi.0.max(lo.0) * mpx,
            _ => GUESS * mpx,
        }
    }

    /// Whether anything has been measured yet.
    fn measured(&self) -> bool {
        self.lo.is_some()
    }
}

/// The render size, chosen each pass from what the last one cost.
///
/// Pure: it holds two exponential averages and an index into a ladder of
/// sizes, and nothing else. See the module docs for the policy.
#[derive(Clone, Debug)]
pub struct Budget {
    /// The ladder, largest first. Index 0 is the pretty rung when there is
    /// one; [`BASE`] is always the base size.
    rungs: Vec<(u32, u32)>,
    base: (u32, u32),
    /// Whether index 0 is a genuine pretty rung above the base size.
    pretty: bool,
    target_ms: f64,
    ceiling_ms: f64,
    cost: Cost,
    /// The rung the last [`Budget::next`] handed out.
    at: usize,
    /// Consecutive passes that have asked for a larger picture.
    up_for: u32,
    /// Whether the last pass counted as still.
    still: bool,
    /// `false` pins the size at the base: the `--budget off` measurement.
    on: bool,
}

impl Budget {
    /// A policy over `base`, allowed up to `pretty` when a still pass fits
    /// inside `ceiling_ms`, and holding a moving pass under `target_ms`.
    ///
    /// A `pretty` no larger than `base` simply has no pretty rung.
    pub fn new(base: (u32, u32), pretty: (u32, u32), target_ms: f64, ceiling_ms: f64) -> Self {
        let base = (base.0.max(FLOOR), base.1.max(FLOOR));
        let mut rungs = Vec::with_capacity(SCALES.len() + 1);
        // Index 0 is the pretty rung. When there is not one it is the base
        // size again, so `BASE` is still the base and the "climb to pretty"
        // step is a no-op rather than a special case.
        let has_pretty = (pretty.0 as u64) * (pretty.1 as u64) > (base.0 as u64) * (base.1 as u64);
        rungs.push(if has_pretty {
            (pretty.0.max(FLOOR), pretty.1.max(FLOOR))
        } else {
            base
        });
        for s in SCALES {
            rungs.push((
                ((base.0 as f64 * s).round() as u32).max(FLOOR),
                ((base.1 as f64 * s).round() as u32).max(FLOOR),
            ));
        }
        Self {
            rungs,
            base,
            pretty: has_pretty,
            target_ms: target_ms.max(1.0),
            ceiling_ms: ceiling_ms.max(1.0),
            cost: Cost::default(),
            at: BASE,
            up_for: 0,
            still: false,
            on: true,
        }
    }

    /// The policy this viewer's live tiers want: the nominal size, twice it
    /// when a still frame can afford it, forty milliseconds moving and a
    /// quarter of a second for a still.
    pub fn around(base: (u32, u32)) -> Self {
        Self::new(base, (base.0 * 2, base.1 * 2), 40.0, 250.0)
    }

    /// A policy that always answers `base`: the control the measurement needs,
    /// and what `--budget off` builds. It still records what it is told, so
    /// the report line reads the same either way.
    pub fn off(base: (u32, u32)) -> Self {
        let mut b = Self::new(base, base, 40.0, 250.0);
        b.on = false;
        b
    }

    /// Whether the policy is choosing sizes or pinned at the base.
    pub fn is_on(&self) -> bool {
        self.on
    }

    /// The size the next pass should render at.
    pub fn size(&self) -> (u32, u32) {
        if self.on { self.rungs[self.at] } else { self.base }
    }

    /// That size as a multiple of the base, on each axis. `1.0` is nominal and
    /// the pretty rung is above it.
    pub fn scale(&self) -> f64 {
        self.size().0 as f64 / self.base.0 as f64
    }

    /// Whether the last pass counted as still — the half of the policy a
    /// report line wants to name.
    pub fn is_still(&self) -> bool {
        self.still
    }

    /// Whether the current size is the pretty rung.
    pub fn is_pretty(&self) -> bool {
        self.on && self.pretty && self.at == 0
    }

    /// What a pass of the current size is predicted to cost, in milliseconds.
    /// `None` until something has been measured.
    pub fn predicted_ms(&self) -> Option<f64> {
        self.cost
            .measured()
            .then(|| self.cost.predict(mpx(self.size())))
    }

    /// Record what the pass just rendered at [`Budget::size`] cost, and choose
    /// the size of the next one.
    ///
    /// `repainted_fraction` is the share of the frame the temporal history
    /// asked to repaint — a mask coverage, or `1.0` for a full pass — and
    /// `camera_moved` says whether the eye is where it was. Together they are
    /// the whole of "is this a still picture".
    pub fn next(&mut self, measured_ms: f64, repainted_fraction: f32, camera_moved: bool) -> (u32, u32) {
        self.cost.observe(mpx(self.size()), measured_ms);
        let still = !camera_moved && repainted_fraction <= QUIET;
        self.still = still;
        if !self.on {
            return self.base;
        }
        if still {
            self.up_for = 0;
            self.stand();
        } else {
            self.walk();
        }
        self.size()
    }

    /// The still policy: never smaller, one rung larger a pass, and the pretty
    /// rung when a pass of it is predicted inside the ceiling.
    ///
    /// One rung a pass and not a jump to the top, because every step resamples
    /// the history and a picture that sharpens over three passes reads better
    /// than one that blinks. The climb from the floor to the base is four
    /// passes, which at a still tier's rate is under a second.
    fn stand(&mut self) {
        if self.at > BASE {
            self.at -= 1;
            return;
        }
        let ms = self.cost.predict(mpx(self.rungs[0]));
        if self.at == BASE && self.pretty && ms <= self.ceiling_ms * UP_MARGIN {
            // At the base with comfortable room above it.
            self.at = 0;
        } else if self.at == 0 && ms > self.ceiling_ms {
            // On the pretty rung and no longer affording it. Between the two
            // thresholds nothing happens, which is what stops a machine
            // sitting on the ceiling from blinking.
            self.at = BASE;
        }
    }

    /// The moving policy: the largest rung at or below the base whose
    /// predicted cost fits the target, down at once and up on evidence.
    fn walk(&mut self) {
        // The pretty rung is a still frame's alone.
        self.at = self.at.max(BASE);
        let fits = |b: &Self, i: usize| b.cost.predict(mpx(b.rungs[i])) <= b.target_ms;
        let want = (BASE..self.rungs.len())
            .find(|&i| fits(self, i))
            .unwrap_or(self.rungs.len() - 1);
        if want > self.at {
            // Too dear: give the rung up now rather than stutter for three
            // passes proving it.
            self.at = want;
            self.up_for = 0;
            return;
        }
        if self.at == BASE {
            self.up_for = 0;
            return;
        }
        // Growing is the one move that has to be earned: comfortably under
        // the target, and said so for several passes running.
        if self.cost.predict(mpx(self.rungs[self.at - 1])) <= self.target_ms * UP_MARGIN {
            self.up_for += 1;
            if self.up_for >= UP_HOLD {
                self.at -= 1;
                self.up_for = 0;
            }
        } else {
            self.up_for = 0;
        }
    }
}

/// A size in megapixels.
fn mpx((w, h): (u32, u32)) -> f64 {
    (w as f64) * (h as f64) / 1.0e6
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE_SIZE: (u32, u32) = (480, 270);
    const PRETTY: (u32, u32) = (960, 540);

    fn budget() -> Budget {
        Budget::new(BASE_SIZE, PRETTY, 40.0, 250.0)
    }

    /// A machine that costs `fixed` milliseconds a pass plus `per`
    /// milliseconds a megapixel. The synthetic sequences below are all this
    /// against the policy.
    fn machine(fixed: f64, per: f64) -> impl Fn((u32, u32)) -> f64 {
        move |size| fixed + per * mpx(size)
    }

    /// A deterministic ±`amp` wobble, so "noisy" means the same thing on every
    /// run.
    fn jitter(k: u32, amp: f64) -> f64 {
        let mut z = (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
        z = (z ^ (z >> 31)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z ^= z >> 29;
        1.0 + amp * (2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0)
    }

    #[test]
    fn a_walk_drops_the_scale_within_two_passes() {
        // The cove on this machine: a hundred and thirty milliseconds at the
        // base size, most of it in the pixels.
        let cost = machine(20.0, 850.0);
        let mut b = budget();
        assert_eq!(b.size(), BASE_SIZE);
        let mut size = b.size();
        for _ in 0..2 {
            size = b.next(cost(size), 1.0, true);
        }
        assert!(
            b.scale() <= 0.5 + 1e-9,
            "two walking passes left the scale at {:.3}",
            b.scale()
        );
        assert!(b.scale() >= 0.25, "and it must not go under the floor");
        assert!(!b.is_still());
    }

    #[test]
    fn a_walk_settles_where_the_pass_fits_the_target() {
        let cost = machine(8.0, 600.0);
        let mut b = budget();
        let mut size = b.size();
        for _ in 0..30 {
            size = b.next(cost(size), 1.0, true);
        }
        let ms = cost(b.size());
        assert!(ms <= b.target_ms, "it settled at {ms:.1} ms, over the 40 ms target");
        // …and not needlessly small: the rung above it must genuinely not fit.
        let above = b.rungs[b.at.saturating_sub(1).max(BASE)];
        if above != b.size() {
            assert!(
                cost(above) > b.target_ms * UP_MARGIN,
                "it stopped at {:?} when {above:?} would have fitted",
                b.size()
            );
        }
    }

    #[test]
    fn the_floor_is_a_quarter_however_slow_the_machine() {
        let cost = machine(500.0, 20_000.0);
        let mut b = budget();
        let mut size = b.size();
        for _ in 0..20 {
            size = b.next(cost(size), 1.0, true);
        }
        assert_eq!(b.size(), b.rungs[SCALES.len()], "a hopeless machine left the floor");
        assert!((b.scale() - 0.25).abs() < 1e-9, "the floor is a quarter, not {:.3}", b.scale());
    }

    #[test]
    fn stopping_climbs_back_within_a_few_passes() {
        let cost = machine(20.0, 850.0);
        let mut b = budget();
        let mut size = b.size();
        for _ in 0..8 {
            size = b.next(cost(size), 1.0, true);
        }
        let walking = b.scale();
        assert!(walking < 1.0, "the walk never stepped down");

        // The player stops: an empty plan and a camera that did not move.
        let mut climbed = None;
        for k in 0..8 {
            size = b.next(cost(size), 0.0, false);
            if b.size() == BASE_SIZE && climbed.is_none() {
                climbed = Some(k + 1);
            }
        }
        assert!(b.is_still());
        let passes = climbed.expect("standing still never reached the base size");
        assert!(passes <= 5, "it took {passes} passes to climb back");
    }

    #[test]
    fn a_still_picture_never_drops_its_resolution() {
        // Dear enough that every still pass overruns the moving target by a
        // wide margin. It must not matter.
        let cost = machine(50.0, 2_000.0);
        let mut b = budget();
        let mut size = b.size();
        let mut lowest = 1.0f64;
        for _ in 0..40 {
            size = b.next(cost(size), 0.0, false);
            lowest = lowest.min(b.scale());
        }
        assert!(lowest >= 1.0, "a still picture fell to {lowest:.3} of the base size");
    }

    #[test]
    fn a_still_frame_that_is_fast_enough_goes_pretty() {
        // Twelve milliseconds at the base, forty-odd at the pretty size:
        // inside the quarter-second ceiling with room to spare.
        let cost = machine(4.0, 60.0);
        let mut b = budget();
        let mut size = b.size();
        for _ in 0..6 {
            size = b.next(cost(size), 0.0, false);
        }
        assert_eq!(b.size(), PRETTY, "a cheap still frame did not reach the pretty size");
        assert!(b.is_pretty());
        assert!(b.scale() > 1.0);

        // …and it gives the rung back the moment the player walks, because
        // pretty is a still frame's alone.
        let after = b.next(cost(b.size()), 1.0, true);
        assert!(after.0 <= BASE_SIZE.0, "the pretty rung survived a camera move");
    }

    #[test]
    fn a_still_frame_sitting_on_the_ceiling_does_not_blink() {
        // The cove, measured: 65 ms at the base and 240-odd at the pretty
        // rung against a 250 ms ceiling, with a tenth of jitter on every
        // pass. Without the deadband this changed size on one pass in eight.
        let cost = machine(10.0, 450.0);
        let mut b = budget();
        let mut size = b.size();
        for k in 0..10 {
            size = b.next(cost(size) * jitter(k, 0.1), 0.0, false);
        }
        let mut changes = 0;
        for k in 10..80 {
            let was = b.size();
            size = b.next(cost(size) * jitter(k, 0.1), 0.0, false);
            if b.size() != was {
                changes += 1;
            }
        }
        assert_eq!(changes, 0, "the still picture changed size {changes} times on the ceiling");
    }

    #[test]
    fn a_still_frame_that_is_too_dear_stays_at_the_base() {
        // Half a second at the pretty size, against a 250 ms ceiling.
        let cost = machine(10.0, 900.0);
        let mut b = budget();
        let mut size = b.size();
        for _ in 0..10 {
            size = b.next(cost(size), 0.0, false);
        }
        assert_eq!(b.size(), BASE_SIZE, "it took a pretty rung it could not pay for");
    }

    #[test]
    fn noisy_timings_do_not_make_it_flap() {
        // A machine sitting right on the target — the worst case for a
        // policy with a threshold in it — with a quarter of jitter on every
        // measurement.
        let cost = machine(6.0, 260.0);
        let mut b = budget();
        let mut size = b.size();
        for k in 0..12 {
            size = b.next(cost(size) * jitter(k, 0.25), 1.0, true);
        }
        let settled = b.size();
        let mut changes = 0;
        for k in 12..80 {
            let was = b.size();
            size = b.next(cost(size) * jitter(k, 0.25), 1.0, true);
            if b.size() != was {
                changes += 1;
            }
        }
        assert!(
            changes <= 1,
            "the size changed {changes} times in sixty-eight noisy passes (settled at {settled:?}, ended at {:?})",
            b.size()
        );
    }

    #[test]
    fn a_walk_after_a_still_gives_the_pretty_rung_straight_back() {
        let cost = machine(4.0, 60.0);
        let mut b = budget();
        let mut size = b.size();
        for _ in 0..6 {
            size = b.next(cost(size), 0.0, false);
        }
        assert!(b.is_pretty());
        // Ten passes of walking on a machine this quick: the picture is
        // allowed to sit at the base, but never above it.
        for _ in 0..10 {
            size = b.next(cost(size), 1.0, true);
            assert!(b.scale() <= 1.0, "a walking pass was handed the pretty rung");
        }
        assert_eq!(size, BASE_SIZE);
    }

    #[test]
    fn a_small_mask_still_counts_as_standing_still() {
        let cost = machine(20.0, 850.0);
        let mut b = budget();
        let mut size = b.size();
        for _ in 0..8 {
            size = b.next(cost(size), 1.0, true);
        }
        let walking = b.scale();
        // A swell on the sea and a door on its hinge: a percent of the frame.
        for _ in 0..6 {
            size = b.next(cost(size), 0.01, false);
        }
        assert!(b.is_still(), "a one-percent mask read as motion");
        assert!(b.scale() > walking, "a one-percent mask stopped the climb");
    }

    #[test]
    fn off_pins_the_size_and_still_measures() {
        let cost = machine(20.0, 850.0);
        let mut b = Budget::off(BASE_SIZE);
        assert!(!b.is_on());
        let mut size = b.size();
        for _ in 0..20 {
            size = b.next(cost(size), 1.0, true);
            assert_eq!(size, BASE_SIZE);
        }
        assert!((b.scale() - 1.0).abs() < 1e-9);
        let p = b.predicted_ms().expect("`off` still measures what it was told");
        assert!((p - cost(BASE_SIZE)).abs() < 30.0, "the model came back {p:.0} ms");
    }

    #[test]
    fn a_base_with_no_pretty_never_grows_past_it() {
        let cost = machine(1.0, 10.0);
        let mut b = Budget::new(BASE_SIZE, BASE_SIZE, 40.0, 250.0);
        let mut size = b.size();
        for _ in 0..20 {
            size = b.next(cost(size), 0.0, false);
            assert_eq!(size, BASE_SIZE);
        }
        assert!(!b.is_pretty());
    }

    #[test]
    fn the_ladder_is_the_scales_and_the_pretty_rung() {
        let b = budget();
        assert_eq!(b.rungs.len(), SCALES.len() + 1);
        assert_eq!(b.rungs[0], PRETTY);
        assert_eq!(b.rungs[BASE], BASE_SIZE);
        // Strictly descending: a rung that repeated one would make the climb
        // stutter in place.
        for w in b.rungs.windows(2) {
            assert!(mpx(w[0]) > mpx(w[1]), "the ladder is not strictly descending: {w:?}");
        }
    }

    #[test]
    fn the_cost_model_separates_the_fixed_part() {
        // Eighty milliseconds of fixed cost and four hundred a megapixel: a
        // one-term fit through the origin would over-charge the small sizes
        // by a factor of two.
        let cost = machine(80.0, 400.0);
        let mut c = Cost::default();
        for _ in 0..10 {
            c.observe(mpx(BASE_SIZE), cost(BASE_SIZE));
            c.observe(mpx((240, 135)), cost((240, 135)));
        }
        let want = cost((360, 202));
        let got = c.predict(mpx((360, 202)));
        assert!(
            (got - want).abs() < 0.05 * want,
            "the model said {got:.1} ms where the machine costs {want:.1}"
        );
    }
}
