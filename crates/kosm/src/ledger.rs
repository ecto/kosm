//! The artifact ledger: append-only `ledger.jsonl`, keyed by [`RunId`].
//!
//! One line per gate run: the number, and enough provenance to say what
//! produced it. Append-only on purpose — a ledger you can edit is a ledger
//! that cannot tell you a score moved.
//!
//! [`RunId`]: crate::run::RunId

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use crate::gate::GateReport;
use crate::run::{RunId, git_rev, kosm_render_version};

/// One line of the ledger.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Entry {
    /// The run hash this score belongs to.
    pub run: String,
    pub task: String,
    pub score: f64,
    pub seed: u64,
    pub count: usize,
    pub tolerance: f64,
    pub rev: String,
    pub kosm_render: String,
}

impl Entry {
    /// The entry a gate report makes under a run id.
    pub fn from_gate(run: &RunId, report: &GateReport) -> Self {
        Self {
            run: run.to_string(),
            task: report.task.clone(),
            score: report.score,
            seed: report.seed,
            count: report.count,
            tolerance: report.tolerance,
            rev: git_rev(),
            kosm_render: kosm_render_version().to_owned(),
        }
    }
}

/// `ledger.jsonl` under an out directory.
pub struct Ledger {
    path: PathBuf,
}

impl Ledger {
    /// Open (creating the directory if need be) the ledger under `dir`.
    pub fn open(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        Ok(Self { path: dir.join("ledger.jsonl") })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one entry. Never rewrites what is already there.
    pub fn append(&self, entry: &Entry) -> anyhow::Result<()> {
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        writeln!(file, "{}", serde_json::to_string(entry)?)?;
        Ok(())
    }

    /// Every entry, oldest first.
    pub fn entries(&self) -> anyhow::Result<Vec<Entry>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let file = std::io::BufReader::new(std::fs::File::open(&self.path)?);
        let mut out = Vec::new();
        for line in file.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                out.push(serde_json::from_str(&line)?);
            }
        }
        Ok(out)
    }

    /// The most recent score for a task, for a gate to compare against.
    pub fn last_score(&self, task: &str) -> anyhow::Result<Option<f64>> {
        Ok(self.entries()?.into_iter().rev().find(|e| e.task == task).map(|e| e.score))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::{GateSpec, check};
    use crate::step::Zero;
    use crate::task::tests::Drop;
    use crate::world::Param;

    #[test]
    fn the_ledger_appends_and_reads_back() {
        let tmp = std::env::temp_dir().join(format!("kosm-ledger-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let ledger = Ledger::open(&tmp).unwrap();
        assert!(ledger.entries().unwrap().is_empty());
        let report = check(&Drop, &Zero, &GateSpec { seed: 1, count: 2, tolerance: 0.01 });
        let id = RunId::of("drop", &[Param::new("h", 0.2)], 1);
        ledger.append(&Entry::from_gate(&id, &report)).unwrap();
        ledger.append(&Entry::from_gate(&id, &report)).unwrap();
        let entries = ledger.entries().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].run, id.to_string());
        assert_eq!(entries[0].task, "drop");
        // json round-trips a float to within a bit; the ledger is provenance,
        // not the gate's own arithmetic
        assert!((ledger.last_score("drop").unwrap().unwrap() - report.score).abs() < 1e-12);
        assert_eq!(ledger.last_score("ollie").unwrap(), None);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
