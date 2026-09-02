//! The marble in the garage.
//!
//! Rung 4 in miniature: a *captured* place instead of an authored one. ipse's
//! `garage-perim` map is a phone-free capture of a real garage: a Gaussian
//! splat for what it looks like, and a signed-distance grid baked from the
//! fused depth for what it is to stand on. Here the marble is dropped onto the
//! captured floor and rolls wherever the real floor sends it. Nothing here is
//! modelled; the floor's slope is a measurement, and the marble's drift is how
//! we read it.
//!
//! The step is phyz's own contact pipeline with the ground plane swapped for
//! the SDF (`ipse_map::find_terrain_contacts_model`), exactly as ipse-sim does
//! it for the robot. The frame is the splat, rendered by tang-3dgs on the GPU,
//! with the ray-cast marble composited where its primary rays hit.

use std::path::Path;

use ipse_map::Map;
use phyz_contact::{ContactCache, ContactMaterial, ContactSolverConfig, assemble, find_contacts, solve_contacts_warm};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder, State};
use phyz_rigid::{aba, forward_kinematics, integrate_configuration};

const POS: usize = 3;

pub struct Garage {
    pub map: Map,
    /// Where the floor is under the drop point, metres.
    pub floor_z: f64,
}

impl Garage {
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let map = Map::load(dir).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let sdf = map.standable().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        // Floor under the grid's centre: the first zero crossing walking down.
        let cx = sdf.origin.x + sdf.cell * sdf.nx as f64 * 0.5;
        let cy = sdf.origin.y + sdf.cell * sdf.ny as f64 * 0.5;
        let floor_z = floor_under(sdf, cx, cy).ok_or_else(|| anyhow::anyhow!("no floor under the map centre"))?;
        Ok(Self { map, floor_z })
    }

    pub fn extents(&self) -> (Vec3, Vec3) {
        let s = self.map.sdf.as_ref().expect("standable");
        let lo = s.origin;
        (lo, lo + Vec3::new(s.cell * s.nx as f64, s.cell * s.ny as f64, s.cell * s.nz as f64))
    }

    pub fn centre_xy(&self) -> (f64, f64) {
        let (lo, hi) = self.extents();
        ((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5)
    }

    /// Floor height at (x, y), if the SDF has one there.
    pub fn floor_at(&self, x: f64, y: f64) -> Option<f64> {
        floor_under(self.map.sdf.as_ref()?, x, y)
    }
}

fn floor_under(sdf: &ipse_map::SdfGrid, x: f64, y: f64) -> Option<f64> {
    floor_under_from(sdf, x, y, 1.0)
}

fn floor_under_from(sdf: &ipse_map::SdfGrid, x: f64, y: f64, z0: f64) -> Option<f64> {
    let mut z = z0;
    let mut prev = sdf.sample(Vec3::new(x, y, z))?;
    while z > -0.5 {
        let zn = z - 0.005;
        let d = sdf.sample(Vec3::new(x, y, zn))?;
        if prev > 0.0 && d <= 0.0 {
            // linear zero crossing
            return Some(zn + 0.005 * d / (d - prev));
        }
        prev = d;
        z = zn;
    }
    None
}

/// Outward normal of the *zero level set* around `p`, from the floor heights
/// at ±h: a plane through where the surface actually is.
///
/// Not `∇sdf`. On this capture the field's gradient is tilted a consistent
/// 2–3° from the level set it bounds (|∇| ≈ 0.7: a truncated, fused field is
/// not a distance), and a normal force along a tilted normal has a lateral
/// component. Measured: with the gradient normal a frictionless marble slid
/// 2.2 m in 2.2 s across a floor that is flat to 1 mm; a 5 cm ball with
/// friction stayed put only because static friction can hide a 3° lie.
fn smoothed_normal(sdf: &ipse_map::SdfGrid, p: Vec3, h: f64) -> Option<Vec3> {
    let z0 = p.z + 0.05;
    let zx1 = floor_under_from(sdf, p.x + h, p.y, z0)?;
    let zx0 = floor_under_from(sdf, p.x - h, p.y, z0)?;
    let zy1 = floor_under_from(sdf, p.x, p.y + h, z0)?;
    let zy0 = floor_under_from(sdf, p.x, p.y - h, z0)?;
    let g = Vec3::new(-(zx1 - zx0) / (2.0 * h), -(zy1 - zy0) / (2.0 * h), 1.0);
    Some(g / g.norm())
}

pub fn marble_model(r: f64, m: f64) -> Model {
    let i = 0.4 * m * r * r;
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_free_body("marble", -1, SpatialTransform::identity(), SpatialInertia::new(m, Vec3::zeros(), Mat3::from_diagonal(&Vec3::new(i, i, i))))
        .build();
    model.bodies[0].geometry = Some(Geometry::Sphere { radius: r });
    model
}

/// One phyz contact step against the captured floor: the pipeline of
/// `Simulator::step_with_contacts` with the plane replaced by the SDF.
pub fn step(model: &Model, state: &mut State, garage: &Garage, material: &ContactMaterial, cache: &mut ContactCache) -> usize {
    let dt = model.dt;
    let sdf = garage.map.sdf.as_ref().expect("standable");
    let (xforms, _) = forward_kinematics(model, state);
    state.body_xform = xforms;
    let mut contacts = ipse_map::find_terrain_contacts_model(model, state, sdf, material.margin);
    // The fused field's gradient wobbles 1–10° cell to cell on a floor that is
    // flat to ±5 mm, and a 1 cm marble reads that as a slope. Keep the depth
    // the field reports; take the normal from a stencil the size of the ball.
    let trace = std::env::var_os("NEWT_TRACE").is_some() && (state.time * 1000.0).round() as i64 % 500 == 1;
    for c in &mut contacts {
        let raw = c.contact_normal;
        if let Some(n) = smoothed_normal(sdf, c.contact_point, 0.03) {
            c.contact_normal = n;
        }
        if trace {
            let centre = Vec3::new(state.q[POS], state.q[POS + 1], state.q[POS + 2]);
            let off = c.contact_point - centre;
            eprintln!(
                "garage contact t={:.3}: raw n=({:+.3},{:+.3},{:+.3}) used n=({:+.3},{:+.3},{:+.3}) depth={:+.5} point−centre=({:+.4},{:+.4},{:+.4}) body_j={}",
                state.time, raw.x, raw.y, raw.z, c.contact_normal.x, c.contact_normal.y, c.contact_normal.z, c.penetration_depth, off.x, off.y, off.z, c.body_j
            );
        }
    }
    contacts.extend(find_contacts(model, state, material.margin));
    let n = contacts.len();
    let qdd = aba(model, state);
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
    let v = state.v.clone();
    integrate_configuration(model, state.q.as_mut_slice(), v.as_slice(), dt);
    state.time += dt;
    n
}

/// Drop the marble at (x, y) just above the captured floor and let it roll.
/// Returns the path (world) and the final state.
pub fn roll(model: &Model, garage: &Garage, at: (f64, f64), seconds: f64, material: &ContactMaterial) -> anyhow::Result<(Vec<Vec3>, State)> {
    let r = match model.bodies[0].geometry {
        Some(Geometry::Sphere { radius }) => radius,
        _ => anyhow::bail!("marble is not a sphere"),
    };
    let floor = garage.floor_at(at.0, at.1).ok_or_else(|| anyhow::anyhow!("no floor at ({}, {})", at.0, at.1))?;
    let mut state = model.default_state();
    state.q[POS] = at.0;
    state.q[POS + 1] = at.1;
    state.q[POS + 2] = floor + r + 0.002;
    let mut cache = ContactCache::new(material.margin.max(1e-3));
    let steps = (seconds / model.dt).round() as usize;
    let mut path = Vec::with_capacity(steps);
    let trace = std::env::var_os("NEWT_TRACE").is_some();
    for k in 0..steps {
        let n = step(model, &mut state, garage, material, &mut cache);
        let p = Vec3::new(state.q[POS], state.q[POS + 1], state.q[POS + 2]);
        if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) || state.v.as_slice()[3..6].iter().any(|v| v.abs() > 5.0) {
            // the marble met something in the scan at speed and the solve came
            // apart. keep the last good state; report where it stopped.
            eprintln!("garage the contact solve diverged at t = {:.3} s; keeping the last finite state", state.time);
            let last = *path.last().unwrap_or(&p);
            state.q[POS] = last.x;
            state.q[POS + 1] = last.y;
            state.q[POS + 2] = last.z;
            break;
        }
        path.push(p);
        if trace && (k < 5 || k % 100 == 0) {
            let p = path[path.len() - 1];
            let sdf = garage.map.sdf.as_ref().unwrap().sample(p);
            eprintln!("garage t={:.3} p=({:+.4},{:+.4},{:+.4}) |v|={:.3} contacts={n} sdf(centre)={:?}", state.time, p.x, p.y, p.z, Vec3::new(state.v[3], state.v[4], state.v[5]).norm(), sdf);
        }
    }
    Ok((path, state))
}

/// Render the splat from `eye` looking at `target`, RGB f32 row-major.
pub fn render_splat(
    garage: &Garage,
    eye: Vec3,
    target: Vec3,
    fx: f64,
    width: u32,
    height: u32,
) -> anyhow::Result<(Vec<f32>, usize)> {
    let path = garage.map.splat_path().ok_or_else(|| anyhow::anyhow!("map has no splat"))?;
    let cloud = tang_3dgs::load_ply(&path).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let camera = tang_3dgs::Camera::look_at(
        [eye.x as f32, eye.y as f32, eye.z as f32],
        [target.x as f32, target.y as f32, target.z as f32],
        [0.0, 0.0, 1.0],
        tang_3dgs::Intrinsics { fx: fx as f32, fy: fx as f32, cx: width as f32 / 2.0, cy: height as f32 / 2.0 },
        width,
        height,
        0.05,
        50.0,
    );
    let config = tang_3dgs::RasterConfig { width, height, bg_color: [0.03, 0.03, 0.04], ..Default::default() };
    let rasterizer = tang_3dgs::Rasterizer::new(config);
    let out = rasterizer.forward(&cloud, &camera);
    Ok((out.image, cloud.count))
}
