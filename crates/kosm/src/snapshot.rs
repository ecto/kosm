//! Snapshot tests are diffs.
//!
//! `assert_close("marble/distance", &values, tol)` compares a lens output
//! against `sims/marble/snapshots/distance.json` (or `.png` for an image).
//! When the file is missing it is written and the assertion passes — the
//! first run records, every run after that checks. `KOSM_UPDATE_SNAPSHOTS=1`
//! overwrites an existing one.
//!
//! The name is `<sim path>/<snapshot>`; the sim path may be nested
//! (`k1/skate/ollie/landing`), and everything but the last segment is the
//! directory under `sims/`.

use std::path::{Path, PathBuf};

/// `sims/`, found from this crate rather than from the caller's cwd.
pub fn sims_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../sims")
}

fn split(name: &str) -> anyhow::Result<(PathBuf, String)> {
    let (sim, leaf) = name
        .rsplit_once('/')
        .ok_or_else(|| anyhow::anyhow!("snapshot `{name}` needs a `<sim>/<name>` path"))?;
    Ok((sims_dir().join(sim).join("snapshots"), leaf.to_owned()))
}

fn updating() -> bool {
    std::env::var("KOSM_UPDATE_SNAPSHOTS").is_ok_and(|v| v != "0" && !v.is_empty())
}

/// Compare a vector of numbers against the stored snapshot.
///
/// Fails with the index, the two values, and how far apart they were — an
/// error that says what to change, as `docs/architecture.md` asks.
pub fn assert_close(name: &str, values: &[f64], tol: f64) -> anyhow::Result<()> {
    let (dir, leaf) = split(name)?;
    let path = dir.join(format!("{leaf}.json"));
    if !path.exists() || updating() {
        std::fs::create_dir_all(&dir)?;
        std::fs::write(&path, serde_json::to_vec_pretty(&values.to_vec())?)?;
        println!("snap   wrote {}", path.display());
        return Ok(());
    }
    let stored: Vec<f64> = serde_json::from_slice(&std::fs::read(&path)?)?;
    anyhow::ensure!(
        stored.len() == values.len(),
        "snapshot {name}: {} values now, {} stored. If the lens changed shape on \
         purpose, re-record with KOSM_UPDATE_SNAPSHOTS=1",
        values.len(),
        stored.len()
    );
    for (i, (got, want)) in values.iter().zip(&stored).enumerate() {
        let d = (got - want).abs();
        anyhow::ensure!(
            d <= tol,
            "snapshot {name}[{i}]: {got} vs stored {want} ({d:.3e} > {tol:.3e}). \
             A changed number means changed code; if the change is wanted, \
             re-record with KOSM_UPDATE_SNAPSHOTS=1"
        );
    }
    Ok(())
}

/// The same, for an image. `tol` is the mean absolute channel difference in
/// 0..255, so a lens that dithers passes and a lens that moved does not.
pub fn assert_image_close(name: &str, image: &image::RgbaImage, tol: f64) -> anyhow::Result<()> {
    let (dir, leaf) = split(name)?;
    let path = dir.join(format!("{leaf}.png"));
    if !path.exists() || updating() {
        std::fs::create_dir_all(&dir)?;
        image.save(&path)?;
        println!("snap   wrote {}", path.display());
        return Ok(());
    }
    let stored = image::open(&path)?.to_rgba8();
    anyhow::ensure!(
        stored.dimensions() == image.dimensions(),
        "snapshot {name}: {:?} now, {:?} stored. Re-record with \
         KOSM_UPDATE_SNAPSHOTS=1 if the camera changed on purpose",
        image.dimensions(),
        stored.dimensions()
    );
    let n = (image.as_raw().len() as f64).max(1.0);
    let sum: f64 = image
        .as_raw()
        .iter()
        .zip(stored.as_raw())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum();
    let mean = sum / n;
    anyhow::ensure!(
        mean <= tol,
        "snapshot {name}: mean channel difference {mean:.3} > {tol:.3} against {}. \
         Re-record with KOSM_UPDATE_SNAPSHOTS=1 if the frame changed on purpose",
        path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_snapshot_name_is_a_sim_path_and_a_leaf() {
        let (dir, leaf) = split("k1/skate/ollie/landing").unwrap();
        assert!(dir.ends_with("k1/skate/ollie/snapshots"));
        assert_eq!(leaf, "landing");
        assert!(split("nameless").is_err());
    }
}
