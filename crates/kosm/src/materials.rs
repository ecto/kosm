//! Material names to linear-RGB colours, for the drawn parts of a level.
//!
//! A level's roots each carry a material name and nothing else — vcad's
//! `SceneEntry` has no colour — so the one place a name becomes a colour is
//! here, and `parts.json` carries the answer to the window rather than making
//! every reader keep its own table. Linear RGB in 0..1, chosen to read as the
//! material under the window's Lambert tier rather than to match a swatch.
//!
//! **The table is [`crate::material`] now.** This module used to hold a
//! `match` of two dozen names; [`colour`] resolves the name through the
//! substance library and asks the substance for its albedo, so a level that
//! says `steel` and a solver that asks `steel` for its density are looking at
//! the same entry. Every name the old table had is an entry in the library
//! carrying exactly the RGB it had here, so nothing renders differently.
//!
//! A name the library does not know is still clay grey — nothing breaks — but
//! it is no longer *silent*: [`colour`] warns once per unknown name, naming
//! the two ways to fix it.

/// Linear RGB for a material name; an unknown name is clay grey.
///
/// The lookup is [`crate::material::named`], so it is case-insensitive and
/// resolves aliases. An unknown name warns once — per name, per process — and
/// falls back to clay grey, which is what it has always done.
pub fn colour(material: &str) -> [f64; 3] {
    match crate::material::named(material) {
        Some(m) => m.colour(),
        None => {
            // A bare `no-collide` marker carries no material after it, and
            // that is not a mistake worth a line of output.
            if !material.trim().is_empty() {
                warn_unknown(material);
            }
            CLAY
        }
    }
}

/// What an unnamed surface is drawn as.
const CLAY: [f64; 3] = [0.60, 0.60, 0.62];

/// One line per unknown name, however many bodies carry it.
fn warn_unknown(name: &str) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let first = match seen.lock() {
        Ok(mut s) => s.insert(name.to_owned()),
        Err(_) => true,
    };
    if first {
        eprintln!(
            "warn   material `{name}` is not in kosm's library; drawing it clay grey. \
             Add an entry to `kosm::material::library`, or author the body with \
             `Body::substance(&material)` and hand it the constants."
        );
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
        assert_eq!(colour("unobtainium"), CLAY);
        assert_ne!(colour("brick"), colour("steel"));
    }

    /// The names every existing level writes, with the linear RGB the old
    /// table gave them. The court, the skatepark and the cove must render the
    /// same after the table moved into [`crate::material`], so this is the
    /// list, spelled out, rather than a property.
    #[test]
    fn every_name_the_old_table_had_is_unchanged() {
        let old: &[(&str, [f64; 3])] = &[
            ("plywood", [0.72, 0.55, 0.33]),
            ("masonite", [0.42, 0.30, 0.22]),
            ("steel", [0.62, 0.64, 0.68]),
            ("concrete", [0.60, 0.60, 0.58]),
            ("brick", [0.58, 0.32, 0.25]),
            ("galvanized", [0.70, 0.72, 0.74]),
            ("glass", [0.55, 0.72, 0.80]),
            ("lamp", [0.98, 0.94, 0.80]),
            ("cardboard", [0.76, 0.62, 0.42]),
            ("rubber", [0.16, 0.16, 0.17]),
            // the court
            ("maple", [0.78, 0.60, 0.36]),
            ("oak", [0.62, 0.45, 0.28]),
            ("rim", [0.85, 0.30, 0.16]),
            ("wall", [0.80, 0.78, 0.72]),
            ("ceiling", [0.86, 0.86, 0.84]),
            ("floor", [0.70, 0.55, 0.34]),
            ("pad", [0.20, 0.30, 0.60]),
            ("window", [0.70, 0.82, 0.90]),
            ("paint", [0.95, 0.95, 0.95]),
            ("key", [0.55, 0.20, 0.18]),
            // the cove
            ("sand", [0.80, 0.71, 0.50]),
            ("rock", [0.45, 0.43, 0.40]),
            ("stone", [0.52, 0.50, 0.46]),
            // and the default a body gets when nothing was said
            ("clay", CLAY),
            ("pla", CLAY),
        ];
        for (name, want) in old {
            assert_eq!(colour(name), *want, "`{name}` changed colour");
        }
        // the `no-collide` prefix still carries the material after it
        assert_eq!(colour(split("no-collide galvanized").1), [0.70, 0.72, 0.74]);
        assert!(!split("no-collide galvanized").0);
    }

    /// Every name resolves to a substance that can also be asked for its
    /// physics — which is the point of moving the table.
    #[test]
    fn a_level_name_is_a_substance() {
        let steel = crate::material::named("steel").expect("steel");
        assert_eq!(colour("steel"), steel.colour());
        assert!(steel.density > 7000.0, "and it knows what it weighs");
        assert_eq!(steel.contact().friction, steel.friction);
    }
}
