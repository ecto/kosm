//! The court, baked into a map the K1 can stand on.
//!
//! The skatepark's bake path, pointed at [`levels/court.loon`]: every root
//! [`parts::collides`] admits — the slab, the gym's floor, walls and ceiling,
//! the hoop's board, rim, bracket, pole and arm, the pads, the bleachers —
//! goes into `mesh.stl` and decides the sign of the field; the ball, the net
//! and the paint do not. The drawn set is everything but the ball, which is a
//! body the physics places and not a fixture of the room.
//!
//! The gym is 26 by 17 by 9 metres and the park's 10 mm cell would make a
//! dense field of it eighteen gigabytes, so the field is sampled over the
//! court alone: the slab's footprint padded by `sdf_pad_mm`, from under the
//! slab to `sdf_top_mm` — above the rim, so a point under the hoop reads the
//! rim and not the edge of the volume. Past the volume there is no floor,
//! which is the same rule the park has past its padding.
//!
//! Metres, z up, the slab's top at z = 0, the origin at centre court.

use std::fs;
use std::path::Path;

use phyz_math::Vec3;

use super::{Hoop, parts};
use crate::scene::{AuthoredScene, MM};
use crate::skatepark::{self, BakeOpts, Baked, Part};

pub const DEFAULT_COURT_LEVEL: &str = "levels/court.loon";

/// The court's bake knobs, resolved to metres.
pub struct CourtBake {
    pub authored: AuthoredScene,
    /// The slab: its x and y extent and thickness; its top is z = 0.
    pub court_x: f64,
    pub court_y: f64,
    pub court_t: f64,
    pub hoop: Hoop,
    pub rim_rod: f64,
    pub pole_x: f64,
    pub pole_r: f64,
    pub cell: f64,
    pub pad: f64,
    /// Top of the sampled volume, above the slab.
    pub top: f64,
    /// Where the K1 stands in the scenario, and which way it faces.
    pub spawn_x: f64,
    pub spawn_y: f64,
    pub spawn_yaw_deg: f64,
}

impl CourtBake {
    pub fn bundled() -> anyhow::Result<Self> {
        Self::load(AuthoredScene::bundled_path(super::DEFAULT_COURT_SCENE))
    }

    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let a = AuthoredScene::load(path)?;
        let s = Self {
            court_x: a.millimetres("court_x_mm")?,
            court_y: a.millimetres("court_y_mm")?,
            court_t: a.millimetres("court_t_mm")?,
            hoop: Hoop::from_scene(&a)?,
            rim_rod: a.millimetres("rim_rod_mm")?,
            pole_x: a.millimetres("pole_x_mm")?,
            pole_r: a.millimetres("pole_r_mm")?,
            cell: a.millimetres("sdf_cell_mm")?,
            pad: a.millimetres("sdf_pad_mm")?,
            top: a.millimetres("sdf_top_mm")?,
            spawn_x: a.parameter_or("spawn_x_mm", 0.0) * MM,
            spawn_y: a.parameter_or("spawn_y_mm", 0.0) * MM,
            spawn_yaw_deg: a.parameter_or("spawn_yaw_deg", 0.0),
            authored: a,
        };
        anyhow::ensure!(s.cell > 0.0 && s.pad >= 0.0, "court needs a positive sdf cell and a non-negative pad");
        anyhow::ensure!(s.top > s.hoop.rim_centre.z, "sdf_top_mm must clear the rim, or the field ends under the hoop");
        Ok(s)
    }

    /// The playable floor: the slab's footprint, its top at z = 0.
    pub fn extent(&self) -> (Vec3, Vec3) {
        (Vec3::new(-self.court_x / 2.0, -self.court_y / 2.0, -self.court_t), Vec3::new(self.court_x / 2.0, self.court_y / 2.0, 0.0))
    }

    /// The sampled volume: the footprint padded, from under the slab to `top`.
    pub fn volume(&self) -> (Vec3, Vec3) {
        let (lo, hi) = self.extent();
        (lo - Vec3::new(self.pad, self.pad, self.pad), Vec3::new(hi.x + self.pad, hi.y + self.pad, self.top))
    }

    /// The level's roots in metres; the ball is not one of them here.
    pub fn parts(&self) -> anyhow::Result<Vec<Part>> {
        let mut parts = skatepark::parts_of(&self.authored, &parts::collides)?;
        parts.retain(|p| !matches!(p.material.as_str(), "ball" | "ball-seams" | "seam"));
        Ok(parts)
    }

    pub fn opts(&self) -> BakeOpts {
        BakeOpts { cell: self.cell, pad: self.pad, volume: Some(self.volume()), extent: Some(self.extent()) }
    }

    /// A point in the rim's plane at its centre: `rim_r` from the rod's inner
    /// face in every direction, the sample the SDF test reads.
    pub fn rim_eye(&self) -> Vec3 {
        Vec3::new(self.hoop.rim_centre.x, self.hoop.rim_centre.y, self.hoop.rim_centre.z - self.rim_rod / 2.0)
    }
}

/// Bake the court into `dir`.
pub fn bake(court: &CourtBake, dir: &Path) -> anyhow::Result<Baked> {
    skatepark::bake_parts(&court.authored, &court.parts()?, court.opts(), dir)
}

/// A scenario ipse's runner can take as is: the K1 at centre court, standing.
pub fn scenario_toml(court: &CourtBake, map_dir: &Path) -> String {
    format!(
        "# written by kosm-spike --court-bake; run from the ipse checkout.\n\
         name = \"court — centre court\"\n\
         notes = \"Stand at centre court on the baked hardwood for six seconds. No board, no shove: the floor is the test.\"\n\
         \n\
         [[condition]]\n\
         label = \"centre court, quiet\"\n\
         map = {map:?}\n\
         at = [{sx}, {sy}, {yaw}]\n\
         duration = 6.0\n",
        map = map_dir.display().to_string(),
        sx = court.spawn_x,
        sy = court.spawn_y,
        yaw = court.spawn_yaw_deg,
    )
}

/// `kosm-spike --court-bake [level]`: bake, sample, report.
pub fn run(level: &Path, out: &Path) -> anyhow::Result<()> {
    let court = CourtBake::load(level)?;
    for w in &court.authored.warnings {
        eprintln!("court warning: {w}");
    }
    let stem = level.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "court".into());
    let dir = out.join("maps").join(&stem);
    let t0 = std::time::Instant::now();
    let baked = bake(&court, &dir)?;
    let s = &baked.sdf;
    let bytes: u64 = ["sdf.bin", "mesh.stl"].iter().filter_map(|f| fs::metadata(dir.join(f)).ok()).map(|m| m.len()).sum();
    println!(
        "court {}: {} roots, {} collision tris → {}  sdf {}×{}×{} at {:.0} mm cells ({:.0} MB, {:.0} MB on disk with the mesh), baked in {:.1} s",
        level.display(),
        baked.parts,
        baked.tris,
        dir.display(),
        s.nx,
        s.ny,
        s.nz,
        s.cell * 1e3,
        (s.data.len() * 4) as f64 / 1e6,
        bytes as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    let at = |p: Vec3| s.sample(p).map_or("outside".to_string(), |d| format!("{:+.1} mm", d * 1e3));
    println!(
        "court field: centre court {}, 20 mm into the slab {}, rim height under the hoop {}, inside the stanchion {}",
        at(Vec3::zero()),
        at(Vec3::new(0.0, 0.0, -0.02)),
        at(court.rim_eye()),
        at(Vec3::new(court.pole_x, 0.0, 1.0)),
    );
    let scenario = dir.join("scenario.toml");
    fs::write(&scenario, scenario_toml(&court, &fs::canonicalize(&dir)?))?;
    println!("court scenario: {}  (cd ../ipse && cargo run --release -p ipse-sim --example k1_court_stand -- --scenario {})", scenario.display(), scenario.display());
    Ok(())
}
