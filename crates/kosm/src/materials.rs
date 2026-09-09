//! Material names to linear-RGB colours, for the drawn parts of a level.
//!
//! A level's roots each carry a material name and nothing else — vcad's
//! `SceneEntry` has no colour — so the one place a name becomes a colour is
//! here, and `parts.json` carries the answer to the window rather than making
//! every reader keep its own table. Linear RGB in 0..1, chosen to read as the
//! material under the window's Lambert tier rather than to match a swatch.

/// Linear RGB for a material name; an unknown name is clay grey.
pub fn colour(material: &str) -> [f64; 3] {
    match material {
        "plywood" => [0.72, 0.55, 0.33],
        "masonite" => [0.42, 0.30, 0.22],
        "steel" => [0.62, 0.64, 0.68],
        "concrete" => [0.60, 0.60, 0.58],
        "brick" => [0.58, 0.32, 0.25],
        "galvanized" => [0.70, 0.72, 0.74],
        "glass" => [0.55, 0.72, 0.80],
        "lamp" => [0.98, 0.94, 0.80],
        "cardboard" => [0.76, 0.62, 0.42],
        "rubber" => [0.16, 0.16, 0.17],
        // the court
        "maple" => [0.78, 0.60, 0.36],
        "oak" => [0.62, 0.45, 0.28],
        "rim" => [0.85, 0.30, 0.16],
        "wall" => [0.80, 0.78, 0.72],
        "ceiling" => [0.86, 0.86, 0.84],
        "floor" => [0.70, 0.55, 0.34],
        "pad" => [0.20, 0.30, 0.60],
        "window" => [0.70, 0.82, 0.90],
        "paint" => [0.95, 0.95, 0.95],
        "key" => [0.55, 0.20, 0.18],
        // the cove
        "sand" => [0.80, 0.71, 0.50],
        "rock" => [0.45, 0.43, 0.40],
        "stone" => [0.52, 0.50, 0.46],
        _ => [0.60, 0.60, 0.62],
    }
}

/// Roots whose material starts with this are drawn but never baked.
pub const NO_COLLIDE: &str = "no-collide";

/// Split a root's material into "does it collide" and "what colour is it".
///
/// vcad gives a root one string, and the level needs to say two things about
/// its roof: that the SDF must not see it, and that it is galvanized steel.
/// So the marker may carry the real material after it — `"no-collide
/// galvanized"` — and a bare `"no-collide"` falls through to clay grey.
pub fn split(material: &str) -> (bool, &str) {
    match material.strip_prefix(NO_COLLIDE) {
        Some(rest) => (false, rest.trim()),
        None => (true, material),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_material_is_clay_grey() {
        assert_eq!(colour("unobtainium"), colour(NO_COLLIDE));
        assert_ne!(colour("brick"), colour("steel"));
    }
}
