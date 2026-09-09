//! The cove baked into a map a body can stand on.
//!
//! The skatepark's bake path, pointed at [`super::scene`]: every root but
//! the door — the beach, the cliff and the rocks are one solid, so that is the
//! `ground` root — goes into `mesh.stl` and decides the sign of the field. The
//! door is left out because it is the one part of the level that moves; until
//! it is a hinged body it is a door-shaped hole in the cliff, which is what a
//! doorway is.
//!
//! The field is sampled over the cove square, from a metre under the waterline
//! to the top of the cliff. At the document's 100 mm cell that is 401 × 401 ×
//! 98 ≈ 16 M cells and 63 MB, which is fine for one cove and is not fine for
//! an island; the sparse field is the follow-up. The cell can be that coarse
//! because the beach is a *plane*: exact signed distance to a plane is linear,
//! and trilinear interpolation of a linear function is exact, so the sand a
//! body walks on costs nothing at any spacing. [`plane_error`] is the claim,
//! and it is the cove's version of the park's `arc_error`.
//!
//! Metres, z up.

use std::path::Path;

use kosm_scan::SdfGrid;
use phyz_math::Vec3;

use super::{CoveScene, sim};
use crate::skatepark::{self, Baked};

/// Bake the cove into `dir`: `mesh.stl`, `sdf.bin`, `map.toml`, the SVG, and
/// `parts/<root>.stl` for everything the level draws, the door included.
pub fn bake(scene: &CoveScene, dir: &Path) -> anyhow::Result<Baked> {
    skatepark::bake_parts(&scene.authored, &scene.parts()?, scene.opts(), dir)
}

/// How far the baked field is from zero on the sand, sampled up the fall line
/// the marble rolls down: the worst absolute distance, metres, over `n`
/// points. On a plane this is interpolation error and nothing else.
pub fn plane_error(scene: &CoveScene, sdf: &SdfGrid, x: f64, n: usize) -> anyhow::Result<f64> {
    anyhow::ensure!(n > 1, "the plane check needs at least two points");
    let (lo, hi) = (scene.waterline() + 1.0, scene.cliff_face_y() - 1.0);
    let mut worst: f64 = 0.0;
    for k in 0..n {
        let y = lo + (hi - lo) * k as f64 / (n - 1) as f64;
        let p = Vec3::new(x, y, scene.sand_z_at(x, y));
        let d = sdf.sample(p).ok_or_else(|| anyhow::anyhow!("the sand at {p:?} is outside the baked volume"))?;
        worst = worst.max(d.abs());
    }
    Ok(worst)
}

/// The marble the beach is checked with: a glass sphere the size of a fist.
pub const MARBLE_R: f64 = 0.1;
/// How long it is rolled, and how far short of the waterline it is stopped.
pub const ROLL_T: f64 = 6.0;
pub const ROLL_GUARD: f64 = 1.0;

/// Bake the cove and roll the marble down it, printing both.
pub fn run(scene: &CoveScene, out: &Path) -> anyhow::Result<()> {
    let dir = out.join("maps").join("cove");
    let t0 = std::time::Instant::now();
    let baked = bake(scene, &dir)?;
    let s = &baked.sdf;
    println!(
        "cove bake: {} roots, {} collision tris → {}  sdf {}×{}×{} at {:.0} mm cells ({} cells, {:.0} MB), baked in {:.1} s",
        baked.parts,
        baked.tris,
        dir.display(),
        s.nx,
        s.ny,
        s.nz,
        s.cell * 1e3,
        s.nx * s.ny * s.nz,
        (s.data.len() * 4) as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    let worst = plane_error(scene, s, scene.spawn_x, 20)?;
    println!("cove sand: the field is within {:.2} mm of the beach plane over 20 points up the fall line (half a cell is {:.0} mm)", worst * 1e3, scene.cell * 5e2);
    let t1 = std::time::Instant::now();
    let r = sim::roll_on_beach(scene, s, MARBLE_R, scene.spawn_x, scene.spawn_y, ROLL_T, ROLL_GUARD)?;
    println!(
        "cove marble: released at rest at ({:+.1}, {:+.1}) m, {:.2} m down the beach in {:.1} s it reached {:.3} m/s against {:.3} m/s for a rolling sphere ({:+.1} %); drift {:.1} mm; {} ms of physics",
        r.start.x,
        r.start.y,
        r.drop,
        ROLL_T,
        r.speed,
        r.predicted,
        r.error() * 100.0,
        r.drift * 1e3,
        t1.elapsed().as_millis()
    );
    Ok(())
}
