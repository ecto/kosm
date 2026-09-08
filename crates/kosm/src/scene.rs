//! Authored scene source and resolved parameters.
//!
//! This is the common boundary between Loon/vcad authorship and computation.
//! It deliberately knows nothing about rigid bodies, fluids, rendering, or
//! objectives; those remain scene-specific computations.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use vcad_ir::Document;

/// Millimetres, the native length unit of authored vcad scenes, to metres.
pub const MM: f64 = 1e-3;

/// A Loon-authored scene after evaluation and parameter resolution.
pub struct AuthoredScene {
    path: PathBuf,
    source: String,
    pub document: Document,
    pub parameters: HashMap<String, f64>,
    pub warnings: Vec<String>,
}

impl AuthoredScene {
    /// Resolve a `.loon` that ships beside its sim, without depending on the
    /// caller's working directory. The sims tree is the search path: a file
    /// named here is looked for anywhere under `sims/`, so `court.loon` is
    /// found at `sims/court/court.loon` and `warehouse.loon` at
    /// `sims/skatepark/warehouse/warehouse.loon`.
    pub fn bundled_path(name: impl AsRef<Path>) -> PathBuf {
        let name = name.as_ref();
        let sims = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../sims");
        let leaf = name.file_name().unwrap_or(name.as_os_str());
        find(&sims, leaf).unwrap_or_else(|| sims.join(name))
    }

    pub fn load_bundled(name: impl AsRef<Path>) -> anyhow::Result<Self> {
        Self::load(Self::bundled_path(name))
    }

    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)?;
        let (document, warnings) = vcad_loon::eval_vcad_parametric(&source, path.parent(), None)
            .map_err(|error| anyhow::anyhow!("{error}"))?;
        let parameters = vcad_ir::resolve_parameters(&document.parameters)
            .map_err(|error| anyhow::anyhow!("{error:?}"))?;

        Ok(Self {
            path: path.to_owned(),
            source,
            document,
            parameters,
            warnings: warnings
                .into_iter()
                .map(|warning| warning.to_string())
                .collect(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The Loon source, as authored. Root names live only here — vcad's
    /// `SceneEntry` carries a material and nothing else — so a caller that
    /// wants to name the parts it exported reads them back off the text.
    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn parameter(&self, name: &str) -> anyhow::Result<f64> {
        self.parameters
            .get(name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("scene {} has no `{name}`", self.path.display()))
    }

    pub fn parameter_or(&self, name: &str, default: f64) -> f64 {
        self.parameters.get(name).copied().unwrap_or(default)
    }

    pub fn millimetres(&self, name: &str) -> anyhow::Result<f64> {
        Ok(self.parameter(name)? * MM)
    }

    /// Return the source with selected `defparam` values replaced.
    pub fn with_parameters(&self, updates: &[(&str, f64)]) -> String {
        let mut out = String::with_capacity(self.source.len());
        for line in self.source.lines() {
            let mut replaced = None;
            for (name, value) in updates {
                let head = format!("[defparam {name} ");
                if let Some(rest) = line.strip_prefix(&head) {
                    let tail = rest.find(']').map(|index| &rest[index..]).unwrap_or("]");
                    replaced = Some(format!("{head}{value:.4}{tail}"));
                }
            }
            out.push_str(&replaced.unwrap_or_else(|| line.to_owned()));
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_updates_preserve_the_rest_of_the_source() {
        let scene = AuthoredScene {
            path: "test.loon".into(),
            source: "[defparam start_x 10]\n[cube [1 2 3]]\n".into(),
            document: Document::new(),
            parameters: HashMap::new(),
            warnings: Vec::new(),
        };

        assert_eq!(
            scene.with_parameters(&[("start_x", -2.5)]),
            "[defparam start_x -2.5000]\n[cube [1 2 3]]\n"
        );
    }

    #[test]
    fn authored_lengths_cross_the_unit_boundary_once() {
        let mut scene = AuthoredScene {
            path: "test.loon".into(),
            source: String::new(),
            document: Document::new(),
            parameters: HashMap::new(),
            warnings: Vec::new(),
        };
        scene.parameters.insert("radius".into(), 125.0);

        assert_eq!(scene.millimetres("radius").unwrap(), 0.125);
    }
}

/// The first file named `leaf` anywhere under `dir`, breadth-first-ish.
fn find(dir: &Path, leaf: &std::ffi::OsStr) -> Option<PathBuf> {
    let mut dirs = Vec::new();
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            dirs.push(path);
        } else if path.file_name() == Some(leaf) {
            return Some(path);
        }
    }
    dirs.sort();
    dirs.iter().find_map(|d| find(d, leaf))
}
