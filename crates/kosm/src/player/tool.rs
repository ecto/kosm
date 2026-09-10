//! `Tool`: what the hand holds.
//!
//! A pose relative to the hand, and a name. The socket is the hero's two-link
//! IK: the player aims with the mouse, [`super::body::Drive::aim`] is the
//! world point the hand is asked for, the arm solves for it, and a PD at the
//! shoulder and the elbow follows. What comes back out of
//! [`super::body::Body::held`] is where the tool actually *is* — which is not
//! where it was asked to be, because an arm has mass — and that is what a
//! scorer reads.
//!
//! Metres, z up.

use phyz_math::{Mat3, SpatialTransform, Vec3};

/// A place and an orientation, body → world.
///
/// `rot`'s columns are the pose's own axes in world, which is the way round a
/// renderer wants them; phyz's `SpatialTransform::rot` is the transpose of
/// this, and [`Pose::from_xform`] is the one place the two meet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub pos: Vec3,
    pub rot: Mat3,
}

impl Pose {
    pub fn new(pos: Vec3, rot: Mat3) -> Self {
        Self { pos, rot }
    }

    pub fn identity() -> Self {
        Self { pos: Vec3::zeros(), rot: Mat3::identity() }
    }

    /// From phyz's body frame, whose `rot` is world → body.
    pub fn from_xform(x: &SpatialTransform) -> Self {
        Self { pos: x.pos, rot: x.rot.transpose() }
    }

    /// As phyz's body frame.
    pub fn to_xform(self) -> SpatialTransform {
        SpatialTransform::new(self.rot.transpose(), self.pos)
    }

    /// `self` composed onto `parent`: this pose read in the parent's frame.
    pub fn then(self, parent: &Pose) -> Pose {
        Pose { pos: parent.pos + parent.rot.mul_vec(self.pos), rot: parent.rot.mul_mat(&self.rot) }
    }
}

/// Something held in a hand.
#[derive(Clone, Debug)]
pub struct Tool {
    pub name: String,
    /// Where the tool sits relative to the hand, in the hand's frame.
    pub grip: Pose,
}

impl Tool {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), grip: Pose::identity() }
    }

    pub fn with_grip(mut self, grip: Pose) -> Self {
        self.grip = grip;
        self
    }

    /// Where this tool is, given where the hand is.
    pub fn placed(&self, hand: &Pose) -> Pose {
        self.grip.then(hand)
    }
}
