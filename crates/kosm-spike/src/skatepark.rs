//! The skatepark: a mini ramp, baked into a map the K1 can stand on.
//!
//! ipse already has the skateboard and a K1 that rides it on a flat floor;
//! its terrain is an `ipse-map` directory (a collision mesh and a signed
//! distance grid baked from it) that a scenario file points at. This module
//! is the park: [`levels/skatepark.loon`] authored as vcad geometry, its
//! mesh baked with ipse-map's own baker into `out/maps/skatepark/`, and the
//! physics of *that map* checked before a robot sees it.
//!
//! The check is the court's e² test for a ramp. A wheel-sized solid sphere is
//! set on the +x transition with its contact point `drop_mm` above the flat
//! and released, and rolled with the same SDF contact path the K1's feet use
//! (`ipse_map::find_terrain_contacts_model`, raw normals — this is an exact
//! bake, not a fused scan, so the gradient is the normal). Rolling without
//! slip, the centre reaches the flat at `v² = 10/7 · g · Δ`, where `Δ` is the
//! centre's drop, and climbs the far wall back to the same height minus what
//! the solver loses. Lateral drift should be zero; a leaning normal shows up
//! there first.
//!
//! Metres, z up, the flat's top at z = 0, x = 0 the middle of the flat.

use std::fs;
use std::path::{Path, PathBuf};

use ipse_map::manifest::{CollisionLayer, MapManifest, Provenance};
use ipse_map::{Map, SdfGrid, TriMesh, stl};
use phyz_contact::{ContactCache, ContactMaterial, ContactSolverConfig, assemble, solve_contacts_warm};
use phyz_math::{GRAVITY, Vec3};
use phyz_model::{Model, State};
use phyz_rigid::{aba, forward_kinematics, integrate_configuration, rotate_free_joint_velocities, strip_free_joint_coriolis};

use crate::garage::marble_model;
use crate::scene::{AuthoredScene, MM};

pub const DEFAULT_SKATEPARK_SCENE: &str = "skatepark.loon";

/// Free-joint q is [wx, wy, wz, x, y, z]; the wheel is joint 0.
const POS: usize = 3;

/// The park's knobs, resolved to simulation units.
pub struct SkateparkScene {
    pub authored: AuthoredScene,
    pub tr_r: f64,
    pub lip: f64,
    pub width: f64,
    pub flat: f64,
    pub deck: f64,
    pub coping_r: f64,
    pub second_side: bool,
    pub cell: f64,
    pub pad: f64,
    pub wheel_r: f64,
    pub wheel_mass: f64,
    /// Height of the wheel's contact point above the flat at release.
    pub drop: f64,
    pub friction: f64,
    pub t_end: f64,
    pub dt: f64,
    pub shove_at: f64,
    pub shove_ns: f64,
}

impl SkateparkScene {
    pub fn bundled() -> anyhow::Result<Self> {
        Self::load(AuthoredScene::bundled_path(DEFAULT_SKATEPARK_SCENE))
    }

    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let a = AuthoredScene::load(path)?;
        let s = Self {
            tr_r: a.millimetres("tr_r_mm")?,
            lip: a.millimetres("lip_mm")?,
            width: a.millimetres("width_mm")?,
            flat: a.millimetres("flat_mm")?,
            deck: a.millimetres("deck_mm")?,
            coping_r: a.millimetres("coping_r_mm")?,
            second_side: a.parameter("second_side")? > 0.5,
            cell: a.millimetres("sdf_cell_mm")?,
            pad: a.millimetres("sdf_pad_mm")?,
            wheel_r: a.millimetres("wheel_r_mm")?,
            wheel_mass: a.parameter("wheel_g")? * 1e-3,
            drop: a.millimetres("drop_mm")?,
            friction: a.parameter("friction")?,
            t_end: a.parameter("t_end")?,
            dt: a.parameter("dt_ms")? * 1e-3,
            shove_at: a.parameter("shove_at")?,
            shove_ns: a.parameter("shove_ns")?,
            authored: a,
        };
        anyhow::ensure!(s.lip < s.tr_r, "lip_mm must be below tr_r_mm (under vert)");
        anyhow::ensure!(s.drop < s.lip, "drop_mm must be below the lip");
        anyhow::ensure!(s.cell > 0.0 && s.dt > 0.0 && s.t_end > 0.0, "skatepark needs a positive cell, dt and t_end");
        Ok(s)
    }

    /// Where the flat ends, ±x.
    pub fn half_flat(&self) -> f64 {
        self.flat / 2.0
    }

    /// Angle of the lip around the transition, from the bottom.
    pub fn lip_angle(&self) -> f64 {
        (1.0 - self.lip / self.tr_r).acos()
    }

    /// A point on the +x transition's surface, `theta` around from the bottom.
    pub fn arc_point(&self, side: f64, theta: f64) -> Vec3 {
        Vec3::new(side * (self.half_flat() + self.tr_r * theta.sin()), 0.0, self.tr_r * (1.0 - theta.cos()))
    }

    /// The wheel's release: its centre, sitting on the +x arc with the contact
    /// point `drop` above the flat.
    pub fn release(&self) -> Vec3 {
        let theta = (1.0 - self.drop / self.tr_r).acos();
        let rho = self.tr_r - self.wheel_r;
        Vec3::new(self.half_flat() + rho * theta.sin(), 0.0, self.tr_r - rho * theta.cos())
    }

    /// How far the wheel's centre falls from release to the flat.
    pub fn centre_drop(&self) -> f64 {
        self.release().z - self.wheel_r
    }

    /// Speed of a solid sphere rolling without slip after that drop.
    pub fn predicted_flat_speed(&self) -> f64 {
        (10.0 / 7.0 * GRAVITY * self.centre_drop()).sqrt()
    }

    pub fn material(&self) -> ContactMaterial {
        ContactMaterial { friction: self.friction, restitution: 0.0, ..Default::default() }
    }

    /// The park's mesh in metres: every evaluated part's triangles.
    pub fn triangles(&self) -> anyhow::Result<Vec<[Vec3; 3]>> {
        let opts = vcad_eval::EvalOptions { skip_clash_detection: true, ..Default::default() };
        let scene = vcad_eval::evaluate_document(&self.authored.document, &opts).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let mut tris = Vec::new();
        for part in &scene.parts {
            let p = &part.mesh.positions;
            let at = |i: u32| {
                let i = i as usize * 3;
                Vec3::new(p[i] as f64, p[i + 1] as f64, p[i + 2] as f64) * MM
            };
            for t in part.mesh.indices.chunks_exact(3) {
                tris.push([at(t[0]), at(t[1]), at(t[2])]);
            }
        }
        anyhow::ensure!(!tris.is_empty(), "the park evaluated to no triangles");
        Ok(tris)
    }
}

/// A baked map directory.
pub struct Baked {
    pub dir: PathBuf,
    pub tris: usize,
    pub sdf: SdfGrid,
}

/// Bake the park into `dir`: `mesh.stl`, `sdf.bin`, `map.toml`, `park.svg`.
pub fn bake(scene: &SkateparkScene, dir: &Path) -> anyhow::Result<Baked> {
    fs::create_dir_all(dir)?;
    let tris = scene.triangles()?;
    let mesh = {
        let mut vertices = Vec::with_capacity(tris.len() * 3);
        let mut triangles = Vec::with_capacity(tris.len());
        for t in &tris {
            let i = vertices.len() as u32;
            vertices.extend_from_slice(t);
            triangles.push([i, i + 1, i + 2]);
        }
        TriMesh::new(vertices, triangles)
    };
    let sdf = SdfGrid::bake(&mesh, scene.cell, scene.pad);
    let err = |e: ipse_map::MapError| anyhow::anyhow!("{e:?}");
    stl::write_binary_stl(&dir.join("mesh.stl"), &tris).map_err(err)?;
    sdf.save(&dir.join("sdf.bin")).map_err(err)?;
    let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "skatepark".into());
    MapManifest {
        name,
        splat: None,
        collision: Some(CollisionLayer { mesh: "mesh.stl".into(), sdf: "sdf.bin".into(), cell: Some(scene.cell), colour: None }),
        align: None,
        provenance: Some(Provenance {
            captured: None,
            device: Some("kosm".into()),
            notes: Some(format!("baked from {}", scene.authored.path().display())),
        }),
        extent: None,
    }
    .save(dir)
    .map_err(err)?;
    let svg = vcad_render::render_svg_str(&scene.authored.document.to_json()?, 2.0).map_err(|e| anyhow::anyhow!(e))?;
    fs::write(dir.join("park.svg"), svg)?;
    Ok(Baked { dir: dir.to_owned(), tris: tris.len(), sdf })
}

/// How far the baked field is from zero along the ideal arc of each
/// transition, sampled from the bottom to just under the lip: the worst
/// absolute distance, metres, over `n` points per side.
pub fn arc_error(scene: &SkateparkScene, sdf: &SdfGrid, n: usize) -> anyhow::Result<f64> {
    let sides: &[f64] = if scene.second_side { &[1.0, -1.0] } else { &[1.0] };
    let mut worst: f64 = 0.0;
    for &side in sides {
        for k in 0..n {
            let theta = scene.lip_angle() * 0.95 * k as f64 / (n - 1) as f64;
            let p = scene.arc_point(side, theta);
            let d = sdf.sample(p).ok_or_else(|| anyhow::anyhow!("arc point {p:?} is outside the baked volume"))?;
            worst = worst.max(d.abs());
        }
    }
    Ok(worst)
}

/// One contact step against the map: `Simulator::step_with_contacts` with the
/// plane swapped for the field, normals as the field reports them.
fn step(model: &Model, state: &mut State, sdf: &SdfGrid, material: &ContactMaterial, cache: &mut ContactCache) {
    let dt = model.dt;
    let (xforms, _) = forward_kinematics(model, state);
    state.body_xform = xforms;
    let contacts = ipse_map::find_terrain_contacts_model(model, state, sdf, material.margin);
    // In the frame the contacts were assembled in: a free joint's body-frame
    // turn is taken out here and put back, exactly, after the solve (phyz).
    let mut qdd = aba(model, state);
    let v_before = state.v.clone();
    strip_free_joint_coriolis(model, v_before.as_slice(), qdd.as_mut_slice());
    let free_qd = &state.v + &(&qdd * dt);
    if contacts.is_empty() {
        state.v = free_qd;
    } else {
        let materials = model.contact_materials(material);
        let config = ContactSolverConfig::simulation();
        let asm = assemble(model, state, &contacts, &materials, &free_qd, dt, &config);
        let seed = cache.warm_start(state, &contacts);
        let solution = solve_contacts_warm(&asm.problem, &config, &seed);
        cache.store(state, &contacts, &solution.impulses);
        state.v = &free_qd + &asm.velocity_delta(&solution.impulses);
    }
    rotate_free_joint_velocities(model, v_before.as_slice(), state.v.as_mut_slice(), dt);
    let v = state.v.clone();
    integrate_configuration(model, state.q.as_mut_slice(), v.as_slice(), dt);
    state.time += dt;
}

/// What the rolled wheel did.
#[derive(Clone, Debug)]
pub struct RollReport {
    /// Fastest the centre moved while over the flat, m/s.
    pub flat_speed: f64,
    /// `sqrt(10/7 · g · Δ)`.
    pub predicted_speed: f64,
    /// Highest the centre got on the far (−x) side, above the flat, or `None`
    /// if it never got there.
    pub far_apex: Option<f64>,
    /// Where the centre was released, above the flat.
    pub release_height: f64,
    /// Furthest the centre strayed from y = 0, m.
    pub drift: f64,
    /// Centre per step.
    pub path: Vec<Vec3>,
}

/// Release the wheel on the +x transition and roll it on the baked map.
pub fn roll(scene: &SkateparkScene, sdf: &SdfGrid) -> anyhow::Result<RollReport> {
    let mut model = marble_model(scene.wheel_r, scene.wheel_mass);
    model.dt = scene.dt;
    let material = scene.material();
    let mut state = model.default_state();
    let c0 = scene.release();
    state.q[POS] = c0.x;
    state.q[POS + 1] = c0.y;
    state.q[POS + 2] = c0.z;
    let mut cache = ContactCache::new(material.margin.max(1e-3));
    let steps = (scene.t_end / scene.dt).round() as usize;
    let mut path = Vec::with_capacity(steps);
    let (mut flat_speed, mut drift) = (0.0f64, 0.0f64);
    let mut far_apex: Option<f64> = None;
    let on_flat = scene.half_flat() - scene.wheel_r;
    let trace = std::env::var_os("KOSM_TRACE").is_some();
    for k in 0..steps {
        step(&model, &mut state, sdf, &material, &mut cache);
        let p = Vec3::new(state.q[POS], state.q[POS + 1], state.q[POS + 2]);
        if trace && k % 100 == 0 {
            eprintln!(
                "skatepark t={:.3} p=({:+.4},{:+.4},{:+.4}) |v|={:.3} w=({:+.2},{:+.2},{:+.2}) sdf(centre)={:?}",
                state.time, p.x, p.y, p.z, Vec3::new(state.v[3], state.v[4], state.v[5]).norm(), state.v[0], state.v[1], state.v[2], sdf.sample(p)
            );
        }
        anyhow::ensure!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite(), "the roll diverged at t = {:.3} s", state.time);
        let v = Vec3::new(state.v[3], state.v[4], state.v[5]).norm();
        if p.x.abs() < on_flat {
            flat_speed = flat_speed.max(v);
        }
        if p.x < -on_flat {
            far_apex = Some(far_apex.map_or(p.z, |a: f64| a.max(p.z)));
        }
        drift = drift.max(p.y.abs());
        path.push(p);
    }
    Ok(RollReport {
        flat_speed,
        predicted_speed: scene.predicted_flat_speed(),
        far_apex,
        release_height: c0.z,
        drift,
        path,
    })
}

/// A scenario ipse's runner can take as is: the K1 on its board on the flat,
/// shoved toward the +x transition.
pub fn scenario_toml(scene: &SkateparkScene, map_dir: &Path) -> String {
    format!(
        "# written by kosm-spike --skatepark; run from the ipse checkout.\n\
         # `objects/skateboard` is the board rigbake writes.\n\
         name = \"skatepark — mini ramp\"\n\
         notes = \"Stand on the board on the flat, take a push toward the +x transition, ride it.\"\n\
         \n\
         [[condition]]\n\
         label = \"flat, shoved toward the transition\"\n\
         map = {map:?}\n\
         at = [0.0, 0.0, 0.0]\n\
         duration = 6.0\n\
         \x20 [[condition.thing]]\n\
         \x20 object = \"objects/skateboard\"\n\
         \x20 at = [0.0, 0.0, 0.0]\n\
         \x20 under_robot = true\n\
         \x20 [[condition.shove]]\n\
         \x20 axis = \"sagittal\"\n\
         \x20 at = {at}\n\
         \x20 peak = {peak}\n\
         \n\
         [search]\n\
         method = \"cem\"\n\
         population = 48\n\
         iterations = 12\n\
         seed = 1\n\
         out = \"models/skatepark.policy\"\n",
        map = map_dir.display().to_string(),
        at = scene.shove_at,
        peak = scene.shove_ns,
    )
}

/// `kosm-spike --skatepark [level]`: bake, check, report.
pub fn run(level: &Path, out: &Path) -> anyhow::Result<()> {
    let scene = SkateparkScene::load(level)?;
    for w in &scene.authored.warnings {
        eprintln!("skatepark warning: {w}");
    }
    let dir = out.join("maps").join("skatepark");
    let t0 = std::time::Instant::now();
    let baked = bake(&scene, &dir)?;
    let s = &baked.sdf;
    println!(
        "skatepark {}: {} tris → {}  sdf {}×{}×{} at {:.0} mm cells ({:.0} MB), baked in {:.1} s",
        level.display(),
        baked.tris,
        dir.display(),
        s.nx,
        s.ny,
        s.nz,
        s.cell * 1e3,
        (s.data.len() * 4) as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    let worst = arc_error(&scene, s, 64)?;
    println!(
        "skatepark transition: r = {:.2} m, lip {:.2} m at {:.0}°; the field is within {:.2} mm of the ideal arc",
        scene.tr_r,
        scene.lip,
        scene.lip_angle().to_degrees(),
        worst * 1e3
    );
    let map = Map::load(&dir).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let sdf = map.standable().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let t1 = std::time::Instant::now();
    let r = roll(&scene, sdf)?;
    println!(
        "skatepark wheel: released with its centre {:.3} m up; on the flat it reached {:.3} m/s against {:.3} m/s for a rolling sphere ({:+.1} %); {} ms of physics",
        r.release_height,
        r.flat_speed,
        r.predicted_speed,
        (r.flat_speed / r.predicted_speed - 1.0) * 100.0,
        t1.elapsed().as_millis()
    );
    match r.far_apex {
        Some(a) => println!(
            "skatepark far wall: the centre climbed to {:.3} m ({:.1} % of the release height); drift {:.1} mm",
            a,
            a / r.release_height * 100.0,
            r.drift * 1e3
        ),
        None => println!("skatepark far wall: the wheel never reached it; drift {:.1} mm", r.drift * 1e3),
    }
    let scenario = dir.join("scenario.toml");
    fs::write(&scenario, scenario_toml(&scene, &fs::canonicalize(&dir)?))?;
    println!("skatepark scenario: {}  (cd ../ipse && cargo run -p ipse-sim --bin train -- {})", scenario.display(), scenario.display());
    Ok(())
}
