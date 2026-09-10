//! `Gait`: legs as pendulums, driven by the root's speed.
//!
//! No keyframes and no policy. A leg of length `L` swinging under gravity has
//! a natural frequency `sqrt(g/L)`, and a walk is two of them out of phase; so
//! the stride frequency is that number scaled by how fast the root is going,
//! the phase is one number that advances at it, and the hip and knee targets
//! are a sine and a raised cosine of the phase. A trained gait from
//! `kosm-train` replaces this behind the same joint targets later.
//!
//! **Be honest about the amplitude.** The hero rig
//! (`sims/rune/hero/figure.rs`) has `LEG_SLACK = 14 mm` — the two leg bones
//! are 14 mm longer than the drop from hip to ankle — which is the whole of
//! how far its feet can move. At 210 mm of leg that is about 74 mm of travel,
//! so a hip swing of ten degrees is the *most* this figure can be asked for,
//! and what comes out is a bob and a swing rather than a step. The phase is
//! honest, the frequency is honest, the stride length is a Mii's.
//!
//! Radians, seconds, metres.

use std::f64::consts::{PI, TAU};

use phyz_math::GRAVITY;

/// How much of the leg's own pendulum frequency a walk runs at.
///
/// One: a walking leg swings at about its natural frequency, which is what
/// makes walking cheap. Anything else here would be a number with no physics
/// behind it.
pub const CADENCE: f64 = 1.0;

/// The largest hip swing, radians. Ten degrees, and see the module docs: at
/// the hero's proportions that is already the whole of `LEG_SLACK`.
pub const SWING_MAX: f64 = 10.0 * PI / 180.0;

/// How much the knee bends over a stride, as a fraction of the hip's swing.
/// A knee that did not bend would drag a straight leg through the sand.
pub const KNEE_RATIO: f64 = 1.6;

/// Below this speed there is no walk to have a frequency, m/s. The phase
/// stops advancing and the amplitude goes with it, so the legs settle to the
/// rig's rest pose rather than freezing mid-stride.
pub const STANDING: f64 = 0.05;

/// One phase, and the leg length it is derived from.
#[derive(Clone, Copy, Debug)]
pub struct Gait {
    /// The stride phase, radians, wrapped into `0..2π`.
    pub phase: f64,
    /// Hip to ankle, metres.
    pub leg_length: f64,
    /// The speed the stride frequency is quoted at, m/s.
    pub reference_speed: f64,
}

impl Gait {
    pub fn new(leg_length: f64, reference_speed: f64) -> Self {
        Self { phase: 0.0, leg_length: leg_length.max(1e-3), reference_speed: reference_speed.max(1e-3) }
    }

    /// The pendulum frequency of this leg, Hz: `sqrt(g/L) / 2π`.
    pub fn pendulum_hz(&self) -> f64 {
        CADENCE * (GRAVITY / self.leg_length).sqrt() / TAU
    }

    /// The stride frequency at `speed`, Hz. Zero standing still, the
    /// pendulum's own at the reference speed, and proportional in between.
    pub fn stride_hz(&self, speed: f64) -> f64 {
        if speed < STANDING {
            return 0.0;
        }
        self.pendulum_hz() * (speed / self.reference_speed).min(2.0)
    }

    /// Advance the phase for one step at `speed`.
    pub fn advance(&mut self, speed: f64, dt: f64) {
        self.phase = (self.phase + TAU * self.stride_hz(speed) * dt).rem_euclid(TAU);
    }

    /// How far the legs swing at `speed`, radians. Proportional to speed and
    /// capped at [`SWING_MAX`], so a body at rest stands at the rig's own
    /// pose and nothing has to be faded out by hand.
    pub fn amplitude(&self, speed: f64) -> f64 {
        SWING_MAX * (speed / self.reference_speed).clamp(0.0, 1.0)
    }

    /// This leg's own phase: the right leg leads, the left is half a stride
    /// behind it. `side` is 0 for the right and 1 for the left.
    pub fn leg_phase(&self, side: usize) -> f64 {
        self.phase + if side == 0 { 0.0 } else { PI }
    }

    /// The hip's target angle for a side, radians: positive swings the thigh
    /// forward.
    pub fn hip(&self, side: usize, speed: f64) -> f64 {
        self.amplitude(speed) * self.leg_phase(side).sin()
    }

    /// The knee's, radians: a half sine, so the knee bends through the
    /// *swing* — the half of the stride the foot is coming forward — and is
    /// straight through the stance, which is what carries the foot over the
    /// ground instead of scuffing it along it. Positive bends the shin
    /// backward, which is the way a knee bends.
    ///
    /// Scuffing is not cosmetic: a swinging foot that drags eats the walk's
    /// momentum, and with the knee on a raised cosine (bent under the body,
    /// straight at the front) the hero's rise time was a third over its time
    /// constant. On a half sine it is a tenth.
    pub fn knee(&self, side: usize, speed: f64) -> f64 {
        let a = self.amplitude(speed) * KNEE_RATIO;
        a * self.leg_phase(side).sin().max(0.0)
    }

    /// How far off the ground this foot is asked to be over the stride,
    /// `0..=1`. The picture's, not the solver's: the joints are what move.
    pub fn lift(&self, side: usize, speed: f64) -> f64 {
        if speed < STANDING {
            return 0.0;
        }
        (self.leg_phase(side).sin().max(0.0)) * (speed / self.reference_speed).clamp(0.0, 1.0)
    }
}
