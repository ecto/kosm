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
//! the SDF (`kosm_scan::find_terrain_contacts_model`), exactly as ipse-sim does
//! it for the robot. The frame is the splat, rendered by tang-3dgs on the GPU,
//! with the ray-cast marble composited where its primary rays hit.

use std::path::Path;

use kosm_scan::Map;
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

fn floor_under(sdf: &kosm_scan::SdfGrid, x: f64, y: f64) -> Option<f64> {
    floor_under_from(sdf, x, y, 1.0)
}

fn floor_under_from(sdf: &kosm_scan::SdfGrid, x: f64, y: f64, z0: f64) -> Option<f64> {
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
fn smoothed_normal(sdf: &kosm_scan::SdfGrid, p: Vec3, h: f64) -> Option<Vec3> {
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
    let mut contacts = kosm_scan::find_terrain_contacts_model(model, state, sdf, material.margin);
    // The fused field's gradient wobbles 1–10° cell to cell on a floor that is
    // flat to ±5 mm, and a 1 cm marble reads that as a slope. Keep the depth
    // the field reports; take the normal from a stencil the size of the ball.
    let trace = std::env::var_os("KOSM_TRACE").is_some() && (state.time * 1000.0).round() as i64 % 500 == 1;
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
    let trace = std::env::var_os("KOSM_TRACE").is_some();
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

/// The map's Gaussian cloud, with the load-time pruning knobs applied.
///
/// KOSM_SPLAT picks another .ply in the map directory (the map ships several
/// trainings: splat7k is the dense uncleaned one, splat_clean0.12 the most
/// pruned). A sparse-view training grows needles (extreme anisotropy),
/// floaters (huge scale) and dust (near-zero opacity); all three are cosmetic
/// to remove with KOSM_SPLAT_MAX_ANISO / _MAX_SCALE / _MIN_OPACITY, and none
/// of them are what the SDF stands on.
pub fn load_cloud(garage: &Garage) -> anyhow::Result<tang_3dgs::GaussianCloud> {
    let path = match std::env::var("KOSM_SPLAT") {
        Ok(name) => garage.map.dir.join(name),
        Err(_) => garage.map.splat_path().ok_or_else(|| anyhow::anyhow!("map has no splat"))?,
    };
    let mut cloud = tang_3dgs::load_ply(&path).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let env = |k: &str, default: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(default);
    let (max_scale, max_aniso, min_opacity) = (env("KOSM_SPLAT_MAX_SCALE", f32::INFINITY), env("KOSM_SPLAT_MAX_ANISO", f32::INFINITY), env("KOSM_SPLAT_MIN_OPACITY", 0.0));
    if max_scale.is_finite() || max_aniso.is_finite() || min_opacity > 0.0 {
        let (mut big, mut needle, mut dust) = (0, 0, 0);
        for i in 0..cloud.count {
            let s = cloud.scales[i].map(f32::exp);
            let (lo, hi) = (s.iter().cloned().fold(f32::MAX, f32::min), s.iter().cloned().fold(f32::MIN, f32::max));
            let opacity = 1.0 / (1.0 + (-cloud.opacities[i]).exp());
            let drop = if hi > max_scale { big += 1; true } else if hi / lo.max(1e-6) > max_aniso { needle += 1; true } else if opacity < min_opacity { dust += 1; true } else { false };
            if drop {
                cloud.opacities[i] = -20.0;
            }
        }
        eprintln!("garage splat: pruned {big} wider than {max_scale} m, {needle} with anisotropy over {max_aniso}, {dust} under opacity {min_opacity}");
    }
    Ok(cloud)
}

/// The same cloud behind kosm-render's [`Splats`] seam.
///
/// The only work is undoing the trainer's activations — scales are stored as
/// logs and opacities as logits, and `Splats::from_parts` wants neither — and
/// reinterpreting `sh_coeffs`, which is already coefficient-major RGB, as the
/// `[f32; 3]` triples that constructor reads. Nothing is resampled: the
/// rasteriser and the tracer are handed the *same* Gaussians, so any
/// difference in the two pictures is a difference in compositing math and
/// not in the data.
pub fn splats_of(cloud: &tang_3dgs::GaussianCloud) -> kosm_render::Splats {
    let scales: Vec<[f32; 3]> = cloud.scales.iter().map(|s| s.map(f32::exp)).collect();
    let opacities: Vec<f32> = cloud.opacities.iter().map(|o| 1.0 / (1.0 + (-o).exp())).collect();
    let sh: Vec<[f32; 3]> = cloud.sh_coeffs.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    kosm_render::Splats::from_parts(&cloud.positions, &scales, &cloud.rotations, &opacities, &sh)
}

/// The garage as kosm-render sees it: the captured cloud as the environment,
/// the marble as analytic glass inside it, path traced.
///
/// This is the same frame `render_splat` rasterises, through the other
/// renderer. The splat backdrop is composited per ray at each Gaussian's
/// maximum-response point instead of by projecting it to a 2D footprint, and
/// the marble is *in* the cloud rather than pasted over it: the room's own
/// radiance is what reflects off it, refracts through it, and lights the
/// floor patch under it, because to the integrator the splat field is an
/// environment that happens to have depth.
///
/// Returns the 8-bit RGBA frame and how many Gaussians it traced. Tonemapping
/// is deliberately *not* applied: a captured cloud's colours are already
/// display-referred, so a straight clamp is what makes this comparable to the
/// rasteriser's output pixel for pixel.
#[allow(clippy::too_many_arguments)]
pub fn render_kosm(
    cloud: &tang_3dgs::GaussianCloud,
    marble: Vec3,
    radius: f64,
    eye: Vec3,
    target: Vec3,
    vfov: f64,
    width: u32,
    height: u32,
) -> anyhow::Result<(image::RgbaImage, usize)> {
    use std::sync::Arc;

    let splats = splats_of(cloud);
    let count = splats.count();
    let splats = Arc::new(kosm_render::Bvh::build(splats));

    // A glass marble: it shows the room twice over, once reflected off the
    // front and once refracted through the middle, and both of those images
    // can only come from the capture.
    let ball = Arc::new(kosm_render::Bvh::build(kosm_render::Analytic::sphere(
        kosm_render::Point3::new(marble.x, marble.y, marble.z),
        radius,
    )));
    let scene = kosm_render::Scene {
        objects: vec![kosm_render::Object::new(ball, kosm_render::Pbr::glass(1.52, 0.02))],
        lights: Vec::new(),
        // Black at infinity: every photon in this frame came from the
        // capture. Anything the marble shows that is not black is the garage.
        env: kosm_render::Environment::constant([0.0; 3]),
        sun: None,
        ground: None,
        splats: Some(splats),
    };
    let camera = kosm_render::Camera::look_at(
        kosm_render::Point3::new(eye.x, eye.y, eye.z),
        kosm_render::Point3::new(target.x, target.y, target.z),
        Vec3::z(),
        vfov.to_degrees(),
    );
    let spp = std::env::var("KOSM_GARAGE_SPP").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
    let opts = kosm_render::PathTraceOptions {
        spp,
        max_depth: 6,
        // The cloud is the only illuminant and it is not importance sampled,
        // so the estimator is BSDF-only; the denoiser earns its keep here.
        denoise: true,
        ..Default::default()
    };
    let film = kosm_render::render(&scene, &camera, width, height, &opts);
    let mut img = image::RgbaImage::new(width, height);
    for (i, px) in img.pixels_mut().enumerate() {
        let c = |k: usize| (film.rgb[i * 3 + k].clamp(0.0, 1.0) * 255.0) as u8;
        *px = image::Rgba([c(0), c(1), c(2), 255]);
    }
    Ok((img, count))
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
    let cloud = load_cloud(garage)?;
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

/// Standalone: render any splat `.ply` from six axis directions around the
/// cloud's core, so an unfamiliar frame (COLMAP scenes have their own up)
/// can be read off by eye. Writes `<out>/<stem>_{px,nx,py,ny,pz,nz}.png`.
pub fn survey_splat(ply: &Path, out: &Path, width: u32, height: u32) -> anyhow::Result<()> {
    let mut cloud = tang_3dgs::load_ply(ply).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    // the core: the middle 80% of positions per axis, so background floaters
    // do not decide the framing
    let mut xs: Vec<f32> = cloud.positions.iter().map(|p| p[0]).collect();
    let mut ys: Vec<f32> = cloud.positions.iter().map(|p| p[1]).collect();
    let mut zs: Vec<f32> = cloud.positions.iter().map(|p| p[2]).collect();
    for v in [&mut xs, &mut ys, &mut zs] {
        v.sort_by(|a, b| a.total_cmp(b));
    }
    let q = |v: &[f32], f: f64| v[((v.len() - 1) as f64 * f) as usize] as f64;
    let lo = Vec3::new(q(&xs, 0.1), q(&ys, 0.1), q(&zs, 0.1));
    let hi = Vec3::new(q(&xs, 0.9), q(&ys, 0.9), q(&zs, 0.9));
    let centre = (lo + hi) * 0.5;
    let radius = (hi - lo).norm() * 0.5;
    println!(
        "splat  {}: {} gaussians, sh degree {}; core {:.2}×{:.2}×{:.2} around ({:+.2}, {:+.2}, {:+.2}), median ({:+.2}, {:+.2}, {:+.2})",
        ply.display(), cloud.count, cloud.sh_degree, hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, centre.x, centre.y, centre.z,
        q(&xs, 0.5), q(&ys, 0.5), q(&zs, 0.5)
    );
    // optional pruning, same knobs as the garage
    let env = |k: &str, default: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(default);
    let (max_scale, max_aniso, min_opacity) = (env("KOSM_SPLAT_MAX_SCALE", f32::INFINITY), env("KOSM_SPLAT_MAX_ANISO", f32::INFINITY), env("KOSM_SPLAT_MIN_OPACITY", 0.0));
    if max_scale.is_finite() || max_aniso.is_finite() || min_opacity > 0.0 {
        for i in 0..cloud.count {
            let sc = cloud.scales[i].map(f32::exp);
            let (lo_s, hi_s) = (sc.iter().cloned().fold(f32::MAX, f32::min), sc.iter().cloned().fold(f32::MIN, f32::max));
            let opacity = 1.0 / (1.0 + (-cloud.opacities[i]).exp());
            if hi_s > max_scale || hi_s / lo_s.max(1e-6) > max_aniso || opacity < min_opacity {
                cloud.opacities[i] = -20.0;
            }
        }
    }
    // the object of interest: the median position, which sits on the thing
    // the cameras circled rather than in the middle of the background
    let median = Vec3::new(q(&xs, 0.5), q(&ys, 0.5), q(&zs, 0.5));
    let stem = ply.file_stem().and_then(|s| s.to_str()).unwrap_or("splat");
    std::fs::create_dir_all(out)?;
    let fx = std::env::var("KOSM_SPLAT_FX").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.9 * height as f64);
    let config = tang_3dgs::RasterConfig { width, height, bg_color: [0.03, 0.03, 0.04], ..Default::default() };
    let rasterizer = tang_3dgs::Rasterizer::new(config);
    // KOSM_SPLAT_VIEW="ex,ey,ez;tx,ty,tz;ux,uy,uz" renders one free view
    // instead of the survey; KOSM_SPLAT_UP="-y" picks the survey's up.
    let parse3 = |t: &str| -> Option<Vec3> {
        let v: Vec<f64> = t.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (v.len() == 3).then(|| Vec3::new(v[0], v[1], v[2]))
    };
    let up_axis = match std::env::var("KOSM_SPLAT_UP").as_deref() {
        Ok("-y") => -Vec3::y(), Ok("y") => Vec3::y(), Ok("-z") => -Vec3::z(), _ => Vec3::z(),
    };
    let side = if up_axis.y.abs() > 0.5 { Vec3::z() } else { Vec3::y() };
    let mut views: Vec<(String, Vec3, Vec3, Vec3)> = Vec::new();
    if let Ok(spec) = std::env::var("KOSM_SPLAT_VIEW") {
        let parts: Vec<&str> = spec.split(';').collect();
        if let (Some(e), Some(t), Some(u)) = (parts.first().and_then(|p| parse3(p)), parts.get(1).and_then(|p| parse3(p)), parts.get(2).and_then(|p| parse3(p))) {
            views.push(("view".into(), e, t, u));
        }
    } else {
        for (name, dir, up) in [
            ("px", Vec3::x(), up_axis), ("nx", -Vec3::x(), up_axis),
            ("py", Vec3::y(), if up_axis.y.abs() > 0.5 { Vec3::z() } else { up_axis }), ("ny", -Vec3::y(), if up_axis.y.abs() > 0.5 { Vec3::z() } else { up_axis }),
            ("pz", Vec3::z(), if up_axis.z.abs() > 0.5 { Vec3::y() } else { up_axis }), ("nz", -Vec3::z(), if up_axis.z.abs() > 0.5 { Vec3::y() } else { up_axis }),
        ] {
            views.push((name.into(), centre + dir * (radius * 1.6), centre, up));
        }
        // and a close-up: from the side, a little above, at the median
        let eye = median + side * (radius * 0.35) - up_axis * (radius * 0.12);
        views.push(("close".into(), eye, median, up_axis));
    }
    for (name, eye, target, up) in views {
        let camera = tang_3dgs::Camera::look_at(
            [eye.x as f32, eye.y as f32, eye.z as f32],
            [target.x as f32, target.y as f32, target.z as f32],
            [up.x as f32, up.y as f32, up.z as f32],
            tang_3dgs::Intrinsics { fx: fx as f32, fy: fx as f32, cx: width as f32 / 2.0, cy: height as f32 / 2.0 },
            width, height, 0.01, 100.0,
        );
        let t0 = std::time::Instant::now();
        let o = rasterizer.forward(&cloud, &camera);
        let mut img = image::RgbaImage::new(width, height);
        for (i, px) in img.pixels_mut().enumerate() {
            let c = |k: usize| (o.image[i * 3 + k].clamp(0.0, 1.0) * 255.0) as u8;
            *px = image::Rgba([c(0), c(1), c(2), 255]);
        }
        let path = out.join(format!("{stem}_{name}.png"));
        img.save(&path)?;
        println!("splat  view {name}: {} ms → {}", t0.elapsed().as_millis(), path.display());
    }
    Ok(())
}
