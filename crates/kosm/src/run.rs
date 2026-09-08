//! A run is `hash(sim path, params, seed, kosm rev)`.
//!
//! Its lens outputs live at that hash, immutably. Two runs with the same
//! hash are the same run, so the gate, the parity check and the snapshot
//! test are all "does this hash's lens output equal that one's".
//!
//! ```
//! use kosm::prelude::*;
//! let a = RunId::of("marble", &[Param::new("tilt", 0.05)], 7);
//! let b = RunId::of("marble", &[Param::new("tilt", 0.05)], 7);
//! let c = RunId::of("marble", &[Param::new("tilt", 0.06)], 7);
//! assert_eq!(a, b);
//! assert_ne!(a, c);
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::world::Param;

/// The identity of a run: sim path, params, seed, and kosm's git rev.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RunId(String);

impl RunId {
    /// The hash. Sixteen hex characters — enough that a collision is not a
    /// thing that happens, short enough to be a directory name an agent can
    /// read out of a log line.
    pub fn of(sim: &str, params: &[Param], seed: u64) -> Self {
        let mut h = Sha256::new();
        h.update(sim.as_bytes());
        h.update([0]);
        // sorted, so a caller's declaration order is not part of the identity
        let sorted: BTreeMap<&str, f64> =
            params.iter().map(|p| (p.name.as_str(), p.value)).collect();
        for (name, value) in sorted {
            h.update(name.as_bytes());
            h.update(value.to_le_bytes());
        }
        h.update(seed.to_le_bytes());
        h.update(git_rev().as_bytes());
        Self(hex(&h.finalize()[..8]))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// kosm's git rev, short. `"unknown"` outside a checkout — a run made there
/// is still reproducible against itself, just not against a commit.
pub fn git_rev() -> String {
    static REV: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    REV.get_or_init(|| {
        std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".into())
    })
    .clone()
}

/// The version of `kosm-render` this run's camera lens came from.
pub fn kosm_render_version() -> &'static str {
    // kosm-render is a path dependency in this workspace, so its version is
    // the workspace version this binary was built from.
    env!("CARGO_PKG_VERSION")
}

/// What a run was.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub run: String,
    pub sim: String,
    pub params: Vec<Param>,
    pub seed: u64,
    pub rev: String,
    pub kosm_render: String,
}

/// Writes a run's lens outputs under `out_dir/<run_id>/`.
///
/// pngs through `image`, numbers into `metrics.json`, and what the run *was*
/// into `run.json`. The manifest and the metrics are written by
/// [`Recorder::finish`], so a sim that fails half way leaves no manifest
/// claiming it succeeded.
pub struct Recorder {
    dir: PathBuf,
    manifest: Manifest,
    metrics: serde_json::Map<String, serde_json::Value>,
}

impl Recorder {
    pub fn new(out_dir: impl AsRef<Path>, sim: &str, params: &[Param], seed: u64) -> anyhow::Result<Self> {
        let id = RunId::of(sim, params, seed);
        let dir = out_dir.as_ref().join(id.as_str());
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            manifest: Manifest {
                run: id.to_string(),
                sim: sim.to_owned(),
                params: params.to_vec(),
                seed,
                rev: git_rev(),
                kosm_render: kosm_render_version().to_owned(),
            },
            metrics: serde_json::Map::new(),
        })
    }

    /// `out_dir/<run_id>/`. Everything this run writes goes here.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn id(&self) -> &str {
        &self.manifest.run
    }

    /// A path inside the run directory, with any parent directories made.
    pub fn path(&self, name: &str) -> anyhow::Result<PathBuf> {
        let path = self.dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(path)
    }

    /// One image lens output.
    pub fn png(&self, name: &str, image: &image::RgbaImage) -> anyhow::Result<PathBuf> {
        let path = self.path(name)?;
        image.save(&path)?;
        Ok(path)
    }

    /// One number (or anything serde can write) into `metrics.json`.
    pub fn metric(&mut self, key: &str, value: impl serde::Serialize) -> anyhow::Result<()> {
        self.metrics.insert(key.to_owned(), serde_json::to_value(value)?);
        Ok(())
    }

    /// Write `metrics.json` and `run.json`, and say where.
    pub fn finish(self) -> anyhow::Result<PathBuf> {
        std::fs::write(
            self.dir.join("metrics.json"),
            serde_json::to_vec_pretty(&serde_json::Value::Object(self.metrics))?,
        )?;
        std::fs::write(self.dir.join("run.json"), serde_json::to_vec_pretty(&self.manifest)?)?;
        println!("run    {} → {}", self.manifest.run, self.dir.display());
        Ok(self.dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_id_ignores_the_order_the_knobs_were_declared_in() {
        let a = RunId::of("marble", &[Param::new("a", 1.0), Param::new("b", 2.0)], 0);
        let b = RunId::of("marble", &[Param::new("b", 2.0), Param::new("a", 1.0)], 0);
        assert_eq!(a, b);
        assert_eq!(a.as_str().len(), 16);
    }

    #[test]
    fn the_seed_and_the_sim_are_part_of_the_identity() {
        let a = RunId::of("marble", &[], 0);
        assert_ne!(a, RunId::of("marble", &[], 1));
        assert_ne!(a, RunId::of("pool", &[], 0));
    }

    #[test]
    fn a_recorder_writes_metrics_and_a_manifest_under_the_hash() {
        let tmp = std::env::temp_dir().join(format!("kosm-run-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let params = [Param::new("tilt", 0.05)];
        let mut rec = Recorder::new(&tmp, "marble", &params, 3).unwrap();
        let id = rec.id().to_owned();
        rec.metric("miss_m", 0.012).unwrap();
        rec.png("frame.png", &image::RgbaImage::new(2, 2)).unwrap();
        let dir = rec.finish().unwrap();
        assert_eq!(dir, tmp.join(&id));
        let metrics: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("metrics.json")).unwrap()).unwrap();
        assert_eq!(metrics["miss_m"], 0.012);
        let manifest: Manifest =
            serde_json::from_slice(&std::fs::read(dir.join("run.json")).unwrap()).unwrap();
        assert_eq!(manifest.sim, "marble");
        assert_eq!(manifest.seed, 3);
        assert_eq!(manifest.params[0].name, "tilt");
        assert!(dir.join("frame.png").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
