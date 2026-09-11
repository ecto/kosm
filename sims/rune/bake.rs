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

use std::path::{Path, PathBuf};

use kosm_scan::{SdfGrid, TriMesh};
use phyz_math::Vec3;

use super::{CoveScene, sim};
use crate::skatepark::{self, Baked, Part};

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

/// `out` the way the sim registry reads `sims/`: a relative path is the
/// workspace root's, not the working directory's.
///
/// `cargo test` runs in `crates/kosm-cli`, a window can be opened from
/// anywhere, and `--bake-light` is usually run from the root; a probe volume
/// written by one and looked for by another under three different `out/`s is
/// a raster tier that quietly draws the sky alone. An absolute `--out` is
/// left as it is.
pub fn workspace_out(out: &Path) -> PathBuf {
    if out.is_absolute() {
        return out.to_owned();
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.ancestors().nth(2).unwrap_or(manifest).join(out)
}

/// Where the cove's baked light lives under a run's `out`, resolved by
/// [`workspace_out`]. The bake writes here and the window reads here.
pub fn probes_path(out: &Path) -> PathBuf {
    workspace_out(out).join("maps").join("cove").join("probes.bin")
}

/// The cove's signed distance field, in memory: the mesh and the volume
/// [`bake`] writes to `sdf.bin`, and nothing written.
pub fn field(scene: &CoveScene) -> anyhow::Result<SdfGrid> {
    let (_, mesh) = skatepark::collision_mesh(&scene.parts()?)?;
    let (lo, hi) = scene.volume();
    Ok(skatepark::bake_sdf_within(&mesh, scene.cell, lo, hi))
}

/// Which points of the cove are rock, for the light bake. Metres.
///
/// **Not a ray count.** [`kosm::light::probes`]'s fallback counts crossings
/// along three rays, which is an inside test for one closed shell and not for
/// the cove's ground, which is fifty of them in one soup with no union
/// (`scene.rs`'s module doc). Where two shells are coplanar — the buttress and
/// a recessed bed share the nominal face, and every bed shares its bedding
/// planes with the next — a ray meets both faces at one distance, the step
/// past the first skips the second, and a crossing goes uncounted. That read
/// the sand half a metre under the door as air.
///
/// **The field first.** Its sign is the nearest triangle's pseudonormal, and
/// outside the union that is exactly the union's sign, so a point the field
/// calls rock is rock and a point in the air is never called rock.
///
/// **Then the parts, for what the field calls air.** Inside the union the
/// field's sign is not the union's: two metres into the cliff the nearest
/// triangles are a bed's top and the next bed's bottom, coplanar and facing
/// opposite ways, and their tie is a coin — the column over the door reads air
/// for most of its height, rock and all. The distance is still right; the sign
/// is not an inside test there, and nothing a contact solver asks of it ever
/// is. So a point the field calls air is asked of each solid root whose
/// bounds hold it, and one closed, welded shell's own signed distance *is* an
/// inside test, inside and out. The door is one of those roots — the field is
/// baked without it, because it moves, so the field has a door-shaped recess
/// where the picture has 200 mm of granite, and a probe traced from inside
/// that slab would drag the door's face toward black.
pub struct Rock<'a> {
    sdf: &'a SdfGrid,
    /// Every [`super::is_solid_root`], welded on its own, with its bounds.
    parts: Vec<((Vec3, Vec3), TriMesh)>,
}

impl<'a> Rock<'a> {
    /// The cove's rock: `sdf` (the level's field, see [`field`]) and the
    /// level's solid roots, the door included.
    pub fn new(scene: &CoveScene, sdf: &'a SdfGrid) -> anyhow::Result<Self> {
        let parts = scene
            .parts()?
            .into_iter()
            .map(|p| {
                let (_, mesh) = skatepark::collision_mesh(&[Part { collide: true, ..p }])?;
                Ok((mesh.aabb(), mesh))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self { sdf, parts })
    }

    /// Whether `p` is inside the rock.
    pub fn contains(&self, p: [f64; 3]) -> bool {
        let q = Vec3::new(p[0], p[1], p[2]);
        if self.sdf.sample(q).is_some_and(|d| d < 0.0) {
            return true;
        }
        self.parts.iter().any(|((lo, hi), mesh)| {
            (lo.x..=hi.x).contains(&q.x)
                && (lo.y..=hi.y).contains(&q.y)
                && (lo.z..=hi.z).contains(&q.z)
                && mesh.signed_distance(q) < 0.0
        })
    }
}

/// How the cove's light is baked: everything [`bake_light`] takes that is not
/// the level or its field.
#[derive(Clone, Debug)]
pub struct Light {
    /// Probe spacing, metres.
    pub spacing: f64,
    /// Hemisphere rays per probe.
    pub rays: usize,
    /// Unit vectors toward the sun, one slice each.
    pub suns: Vec<[f64; 3]>,
    /// Path length.
    pub depth: u32,
    /// Shadow rays into the sun's disc. Unused while `sun_direct` is false,
    /// and carried so a bake that turns it on has the level's number.
    pub sun_samples: usize,
    /// The seed.
    pub seed: u64,
}

impl Light {
    /// The level's own bake: its authored spacing, [`RAYS`], one sun, depth
    /// three, [`LIGHT_SEED`]. What `kosm run rune --bake-light` does with no
    /// flags, and what a test that needs the cove's light bakes.
    pub fn level(scene: &CoveScene) -> Self {
        let d = scene.sun_dir();
        Self {
            spacing: scene.authored.parameter_or("probe_spacing_mm", 500.0) * kosm::scene::MM,
            rays: RAYS,
            suns: vec![[d.x, d.y, d.z]],
            depth: 3,
            sun_samples: kosm::light::probes::BakeSpec::default().sun_samples,
            seed: LIGHT_SEED,
        }
    }

    /// The level's bake with `--probe-spacing-mm`, `--bake-rays`,
    /// `--bake-suns`, `--bake-depth`, `--bake-sun-samples` and `--bake-seed`
    /// over it.
    ///
    /// One sun is the level's; more is its day. The elevations are the arc the
    /// design asks for and the azimuth is the authored one, because the cove's
    /// whole optical argument — a focus 1.47 radii off the axis has to land on
    /// the door — is an argument about *that* azimuth.
    pub fn from_args(scene: &CoveScene, args: &kosm_cli::Args) -> Self {
        let mut l = Self::level(scene);
        let num = |name: &str| args.value(name).and_then(|v| v.parse::<f64>().ok());
        if let Some(mm) = num("probe-spacing-mm") {
            l.spacing = mm * kosm::scene::MM;
        }
        if let Some(v) = args.value("bake-rays").and_then(|v| v.parse().ok()) {
            l.rays = v;
        }
        if let Some(v) = args.value("bake-depth").and_then(|v| v.parse().ok()) {
            l.depth = v;
        }
        if let Some(v) = args.value("bake-sun-samples").and_then(|v| v.parse().ok()) {
            l.sun_samples = v;
        }
        if let Some(v) = args.value("bake-seed").and_then(|v| v.parse().ok()) {
            l.seed = v;
        }
        let n_suns: usize = args.value("bake-suns").and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
        if n_suns > 1 {
            l.suns = (0..n_suns)
                .map(|i| {
                    let el = (5.0 + (60.0 - 5.0) * i as f64 / (n_suns - 1) as f64).to_radians();
                    let (sa, ca) = scene.sun_az.sin_cos();
                    let (se, ce) = el.sin_cos();
                    [ce * ca, ce * sa, se]
                })
                .collect();
        }
        l
    }

    /// The probe lattice: [`CoveScene::volume`] at [`Self::spacing`].
    pub fn volume(&self, scene: &CoveScene) -> kosm::light::probes::VolumeSpec {
        let (lo, hi) = scene.volume();
        kosm::light::probes::VolumeSpec::over([lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z], self.spacing)
            .in_scene_units(super::render::PER_M)
    }
}

/// Bake the cove's light: the picture without the being, over
/// [`Light::volume`], with [`Rock`] deciding which probes are rock.
///
/// The one bake both `kosm run rune --bake-light` and a test that compares the
/// raster tier with the tracer go through, so the volume a window reads and
/// the volume a parity number was measured against cannot be two different
/// bakes.
pub fn bake_light(
    scene: &CoveScene,
    light: &Light,
    sdf: &SdfGrid,
    progress: Option<&(dyn Fn(usize, usize) + Sync)>,
) -> anyhow::Result<kosm::light::probes::ProbeVolume> {
    use kosm::light::probes::{self, BakeSpec};

    let mut picture = super::render::Scene::new(scene)?;
    picture.set_being_visible(false);
    let placement = super::render::Placement::standing(scene, scene.spawn_x, scene.spawn_y, 0.0);
    let lit = picture.at(&placement);
    let rock = Rock::new(scene, sdf)?;
    let inside = |p: [f64; 3]| rock.contains(p);
    Ok(probes::bake_with(
        &lit,
        &BakeSpec {
            volume: light.volume(scene),
            suns: light.suns.clone(),
            rays: light.rays,
            seed: light.seed,
            max_depth: light.depth,
            sun_samples: light.sun_samples,
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
            inside: Some(&inside),
            progress,
            ..BakeSpec::default()
        },
    ))
}

/// The cove's static light, baked into spectral SH probes.
///
/// `kosm run rune --bake-light` — see `docs/plans/2026-09-10-bake-and-raster-design.md`.
/// The cove's lighting does not move: the sun is a knob, the sky is a
/// gradient, and the sand, the cliff, the reef and the door are geometry that
/// stands still. So the path tracer solves it once over
/// [`CoveScene::volume`] — the same volume the signed distance field is
/// sampled over — and [`kosm::light::probes`] writes the answer to
/// [`probes_path`] for a raster tier to read.
///
/// Four knobs, and only the first is authored: `probe_spacing_mm` (500, and
/// the level does not need to declare it, though `--probe-spacing-mm`
/// overrides it), `--bake-rays` (256), `--bake-suns` (1) and `--bake-depth`
/// (3, which is the sky, what it lights, and one bounce off that). One sun is
/// the level's own; more is the day's arc, from 5° to 60° of elevation along
/// the authored azimuth, which is what a level whose sun moves would read with
/// a fractional index. [`Light::from_args`] reads them.
///
/// **The being is not in it.** The hero walks; a probe volume that had baked
/// its shadow into the sand would carry it wherever the hero went.
///
/// **Which probes are rock is the field's call**, so the field is baked first
/// ([`field`], in memory, the same one `kosm run rune` writes) and [`Rock`]
/// reads its sign, and asks the level's closed parts where the sign cannot
/// say.
pub fn run_light(scene: &CoveScene, args: &kosm_cli::Args, out: &Path) -> anyhow::Result<()> {
    use kosm::light::probes;

    let light = Light::from_args(scene, args);
    let volume = light.volume(scene);
    let (lo, hi) = scene.volume();
    let path = probes_path(out);
    let n = volume.count();
    let suns_n = light.suns.len();

    let t_sdf = std::time::Instant::now();
    let sdf = field(scene)?;
    println!(
        "cove light: the field that decides what is rock, {}×{}×{} at {:.0} mm, in {:.1} s",
        sdf.nx,
        sdf.ny,
        sdf.nz,
        sdf.cell * 1e3,
        t_sdf.elapsed().as_secs_f64()
    );
    println!(
        "cove light: {}×{}×{} probes at {:.0} mm over ({:+.1}, {:+.1}, {:+.1}) … ({:+.1}, {:+.1}, {:+.1}) m, {} sun{} at {} rays and depth {} — {} paths",
        volume.dims[0],
        volume.dims[1],
        volume.dims[2],
        light.spacing * 1e3,
        lo.x, lo.y, lo.z, hi.x, hi.y, hi.z,
        suns_n,
        if suns_n == 1 { "" } else { "s" },
        light.rays,
        light.depth,
        n * suns_n * light.rays,
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
    let baked = bake_light(scene, &light, &sdf, Some(&tick))?;
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
