//! `kosm::player`: a third-person controller any sim can drop in.
//!
//! Three small types on the three nouns, and nothing new. [`body::Body`] is
//! the physics — an articulated rig on one free joint, an upright spring, a
//! lean that starts a walk, a pendulum gait and a socket for what the hand
//! holds. [`rig::Rig`] is the camera, a `Lens` that yields a frame and moves
//! like a thing with mass. [`body::Drive`] is what turns a window's input, or
//! a policy's action, into either.
//!
//! Under them: [`ground::Ground`], what is under the feet (a baked field, a
//! plane, the world's own colliders, and a net under all three), and
//! [`medium::Medium`], what the body moves through (air, or the cove's sea).
//!
//! Metres, radians, seconds; z up. The body's own frame is `+x` forward,
//! `+y` left, `+z` up.
//!
//! ```
//! use kosm::player::{Body, BodySpec, Drive, Plane, medium::Air};
//! let mut body = Body::new(BodySpec::capsule(0.175, 1.4, 2510.0));
//! body.place(0.0, 0.0, 0.0, 0.0, 0.0);
//! body.run_for(1.0, &Drive::walking(1.0), &Plane::at(0.0), &Air);
//! assert!(body.speed() > 0.5);
//! ```
pub mod body;
pub mod gait;
pub mod ground;
pub mod medium;
pub mod meter;
pub mod rig;
pub mod tool;

pub use body::{
    Arm, Body, BodySpec, BodyStep, Consts, Dangle, Drive, Foot, Forgiveness, Leg, Link, Lump, Part,
    PivotKind, Skeleton, Snapshot,
};
pub use gait::Gait;
pub use ground::{Colliders, Ground, LedgeInfo, LedgeProbe, Netted, Nowhere, Plane, SdfGround, Terrace};
pub use medium::{Air, Immersed, Immersion, Medium, Water};
pub use tool::{Pose, Tool};
