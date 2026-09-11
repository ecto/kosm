//! The cove baked into a map a body can stand on.
//!
//! The skatepark's bake path, pointed at [`super::scene`]: every
//! [`ground`](super::is_ground) root — the beach, the cliff's beds, the
//! headlands, the reef, the boulders and the pools' lips — goes into
//! `mesh.stl` as triangles and together they decide the sign of the field.
//! They are separate solids and no union holds them together; `scene.rs`'s
//! module doc is the argument for why the field does not care. The door is
//! left out because it is the one part of the level that moves; until it is a
//! hinged body it is a door-shaped hole in the cliff, which is what a doorway
//! is.
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
///
/// It is handed [`CoveScene::solids`] and not the whole document, for the
/// reason [`super::is_solid_root`] gives: `bake_parts` draws its `park.svg`
/// from whatever it is given, and a drawing of the whole level is a drawing of
/// six hundred blades of marram. The geology and the door are what a map's
/// companion drawing is *of*, and it is the same view `kosm run rune` writes
/// beside the STL — minutes, and now a third of a second.
pub fn bake(scene: &CoveScene, dir: &Path) -> anyhow::Result<Baked> {
    skatepark::bake_parts(&scene.solids(), &scene.parts()?, scene.opts(), dir)
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
    // The two halves, separately: evaluating the level's solid roots to
    // triangles, and turning those triangles into a field. The first is the
    // one the dressing pass made minutes long, so it is printed rather than
    // folded into the second.
    let t_eval = std::time::Instant::now();
    let parts = scene.parts()?;
    let eval_s = t_eval.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let baked = skatepark::bake_parts(&scene.solids(), &parts, scene.opts(), &dir)?;
    let s = &baked.sdf;
    println!(
        "cove bake: {} roots evaluated in {eval_s:.1} s, {} collision tris → {}  sdf {}×{}×{} at {:.0} mm cells ({} cells, {:.0} MB), baked in {:.1} s",
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

// ── the light ─────────────────────────────────────────────────────────────

/// The cove's static light, baked into spectral SH probes.
///
/// `kosm run rune --bake-light` — see `docs/plans/2026-09-10-bake-and-raster-design.md`.
/// The cove's lighting does not move: the sun is a knob, the sky is a
/// gradient, and the sand, the cliff, the reef and the door are geometry that
/// stands still. So the path tracer solves it once over
/// [`CoveScene::volume`] — the same volume the signed distance field is
/// sampled over — and [`kosm::light::probes`] writes the answer to
/// `out/maps/cove/probes.bin` for a raster tier to read.
///
/// Four knobs, and only the first is authored: `probe_spacing_mm` (500, and
/// the level does not need to declare it, though `--probe-spacing-mm`
/// overrides it), `--bake-rays` (64), `--bake-suns` (1) and `--bake-depth`
/// (3, which is the sky, what it lights, and one bounce off that). One sun is the level's own; more is the day's arc, from
/// 5° to 60° of elevation along the authored azimuth, which is what a level
/// whose sun moves would read with a fractional index.
///
/// **The being is not in it.** The hero walks; a probe volume that had baked
/// its shadow into the sand would carry it wherever the hero went.
pub fn run_light(scene: &CoveScene, args: &kosm_cli::Args, out: &Path) -> anyhow::Result<()> {
    use kosm::light::probes::{self, BakeSpec, VolumeSpec};
    use kosm::scene::MM;

    let a = &scene.authored;
    let spacing = args
        .value("probe-spacing-mm")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| a.parameter_or("probe_spacing_mm", 500.0))
        * MM;
    let rays: usize = args.value("bake-rays").and_then(|v| v.parse().ok()).unwrap_or(RAYS);
    let n_suns: usize = args.value("bake-suns").and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let depth: u32 = args.value("bake-depth").and_then(|v| v.parse().ok()).unwrap_or(3);
    let sun_samples: usize = args
        .value("bake-sun-samples")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| kosm::light::probes::BakeSpec::default().sun_samples);
    let seed: u64 = args.value("bake-seed").and_then(|v| v.parse().ok()).unwrap_or(LIGHT_SEED);

    let (lo, hi) = scene.volume();
    let volume = VolumeSpec::over([lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z], spacing)
        .in_scene_units(super::render::PER_M);

    // One sun is the level's; more is its day. The elevations are the arc the
    // design asks for and the azimuth is the authored one, because the cove's
    // whole optical argument — a focus 1.47 radii off the axis has to land on
    // the door — is an argument about *that* azimuth.
    let suns: Vec<[f64; 3]> = if n_suns == 1 {
        let d = scene.sun_dir();
        vec![[d.x, d.y, d.z]]
    } else {
        (0..n_suns)
            .map(|i| {
                let el = (5.0 + (60.0 - 5.0) * i as f64 / (n_suns - 1) as f64).to_radians();
                let (sa, ca) = scene.sun_az.sin_cos();
                let (se, ce) = el.sin_cos();
                [ce * ca, ce * sa, se]
            })
            .collect()
    };

    let mut picture = super::render::Scene::new(scene)?;
    picture.set_being_visible(false);
    let placement = super::render::Placement::standing(scene, scene.spawn_x, scene.spawn_y, 0.0);
    let lit = picture.at(&placement);

    let suns_n = suns.len();
    let dir = out.join("maps").join("cove");
    let path = dir.join("probes.bin");
    let n = volume.count();
    println!(
        "cove light: {}×{}×{} probes at {:.0} mm over ({:+.1}, {:+.1}, {:+.1}) … ({:+.1}, {:+.1}, {:+.1}) m, {} sun{} at {rays} rays and depth {depth} — {} paths",
        volume.dims[0],
        volume.dims[1],
        volume.dims[2],
        spacing * 1e3,
        lo.x, lo.y, lo.z, hi.x, hi.y, hi.z,
        suns.len(),
        if suns.len() == 1 { "" } else { "s" },
        n * suns.len() * rays,
    );
    let t0 = std::time::Instant::now();
    // Every twentieth of the way, and at the end. `bake_with` calls this
    // every few thousand probes, so the step is what turns that into a line.
    let step = (n * suns_n / 20).max(1);
    let tick = |done: usize, total: usize| {
        if done == total || done / step != done.saturating_sub(4096) / step {
            println!("  {:3.0} %  {:.0} s", 100.0 * done as f64 / total as f64, t0.elapsed().as_secs_f64());
        }
    };
    let baked = probes::bake_with(
        &lit,
        &BakeSpec {
            volume,
            suns,
            rays,
            seed,
            max_depth: depth,
            sun_samples,
            // **The sun's direct term is the raster tier's, not the bake's.**
            // The volume this writes is read by a rasterizer that computes
            // `E · max(0, n·s)` itself, per pixel, against a 2048² shadow
            // map — which resolves the hero's own shadow and the door's jamb,
            // neither of which a lattice half a metre across ever could. What
            // is left in the SH is the sky and every bounce, *including* the
            // sun's: the sun still lights the sand the probes look at, so the
            // warm throw onto the cliff face is in here. Baking the direct
            // term as well and letting the shader add its own is a factor of
            // 1.8 on the cove's sunlit sand, which is what this line is for.
            sun_direct: false,
            progress: Some(&tick),
            ..BakeSpec::default()
        },
    );
    let secs = t0.elapsed().as_secs_f64();
    baked.write(&path)?;
    let buried = (0..baked.dims[2])
        .flat_map(|z| (0..baked.dims[1]).flat_map(move |y| (0..baked.dims[0]).map(move |x| (x, y, z))))
        .filter(|(x, y, z)| baked.is_inside(*x, *y, *z))
        .count();
    println!(
        "cove light: {n} probes ({buried} inside the rock, {:.0} % traced), {:.1} MB → {} in {:.1} s",
        100.0 * (n - buried) as f64 / n as f64,
        (baked.data.len() * 4) as f64 / 1e6,
        path.display(),
        secs
    );
    // What the bake is worth, as a number rather than as a picture: the sand
    // in front of the door, up-facing, in the middle band.
    let door = scene.door_frame().origin;
    let p = [door.x, door.y - 2.0, scene.sand_z_at(door.x, door.y - 2.0) + 0.05];
    let e = baked.sample(0.0, p, [0.0, 0.0, 1.0]);
    println!(
        "cove light: the sand two metres off the door reads {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} per band ({:.2} {:.2} {:.2} as RGB)",
        e[0], e[1], e[2], e[3], e[4], e[5],
        probes::bands_to_rgb(&e)[0], probes::bands_to_rgb(&e)[1], probes::bands_to_rgb(&e)[2],
    );
    Ok(())
}

/// The light bake's seed. Fixed, so two bakes differ only where the level
/// does; `--bake-seed` moves it, which is how the noise below was measured.
pub const LIGHT_SEED: u64 = 0x6c19_47_5eed;

/// Hemisphere rays per probe.
///
/// The design's estimate was sixty-four, on a budget of ten minutes. The
/// whole cove at 0.5 m is two seconds at that count on this machine — the
/// probes are trivially parallel and the sun costs nothing because it is not
/// sampled — so the budget buys four times the rays and half the noise
/// instead, and the bake is still under ten seconds.
pub const RAYS: usize = 256;
