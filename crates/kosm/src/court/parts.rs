//! Which of the court's root materials the physics owns.

pub use crate::brep::PlacedSolid;

/// Root materials that are appearance only: the physics leaves them out of
/// the court body, the picture draws them. `ball` is the ball's own solid and
/// `ball-seams` (`seam`) its four rings, both placed at each ball's pose;
/// `paint` is the court's markings and `key` the lane's colour under them;
/// `decor` is anything else that should not be stood on.
///
/// Everything else collides, including names this list has never heard of —
/// the gym's `wall`, `ceiling`, `floor`, `pad` and `oak` are here by saying
/// nothing. Appearance is the exception a root has to ask for.
pub fn collides(material: &str) -> bool {
    !matches!(
        material,
        "ball" | "ball-seams" | "seam" | "paint" | "key" | "decor" | "window"
    )
}
