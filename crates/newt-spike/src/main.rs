//! The marble.
//!
//! One level that is, at the same time, a vcad document (printable), a phyz
//! rollout (the same solver the robots use), an adjoint gradient (the hint
//! button), and a rendered frame.
//!
//! The level is `levels/marble.loon`. Everything below reads that file and
//! nothing else: the geometry is the vcad document it evaluates to, the
//! colliders and visuals are a derivation of that document (`colliders.rs`,
//! checked against the tessellation before the first step), and the knobs
//! (tilt, release point, marble, horizon) are its `defparam`s. A solved level
//! is the same file with the knobs rewritten.
//!
//! Units: vcad millimetres in the level, phyz metres in the physics.
//!
//! The game: a tilted plate with side walls, a marble released from rest near
//! the top, and a cup partway down with its mouth facing uphill. After
//! `t_end` seconds the marble should be in the cup. Two knobs, both solved:
//!   - **placement**: hold the tilt, move the release point. Exact gradient by
//!     the convex-contact adjoint, checked against central differences.
//!   - **tilt**: hold the release point, rotate the plate. Two scalars, central
//!     differences of the same rollout (tilt is not an adjoint channel yet).

mod audio;
mod colliders;

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use phyz::Simulator;
use phyz_camera::{CameraPose, RenderScene, RgbdCamera, SceneOptions};
use phyz_contact::{ContactMaterial, ContactSolverConfig, find_contacts};
use phyz_diff::{
    ConvexContactRollout, FinalStateObjective, convex_adjoint_gradient, convex_rollout_objective,
};
use phyz_math::{DVec, GRAVITY, Mat3, SpatialInertia, SpatialTransform, SpatialTransformExt, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder, State};
use phyz_rigid::forward_kinematics;
use phyz_world::{CameraIntrinsics, Scene, SensorContext};
use vcad_ir::{CsgOp, Document, Node};

const DT: f64 = 1e-3;
const MARBLE: usize = 0;
const TRACK: usize = 1;
/// Free-joint q is [wx, wy, wz, x, y, z]; the marble is joint 0.
const POS: usize = 3;
const MM: f64 = 1e-3;

// ---- the level --------------------------------------------------------------

/// A `.loon` level: its document plus the knobs it declared.
struct Level {
    source: String,
    doc: Document,
    params: HashMap<String, f64>,
}

impl Level {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let source = fs::read_to_string(path)?;
        // Provenance recovery re-evaluates the program 2n+2 times to learn which
        // geometry each knob drives; half these knobs are game state, not
        // geometry, so skip it.
        // SAFETY: single-threaded at this point; nothing else reads the environment.
        unsafe { std::env::set_var("VCAD_LOON_NO_PARAM_RECOVERY", "1") };
        let (doc, warnings) = vcad_loon::eval_vcad_parametric(&source, path.parent(), None)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        for w in warnings {
            println!("level  {w}");
        }
        let params = vcad_ir::resolve_parameters(&doc.parameters).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(Self { source, doc, params })
    }

    fn p(&self, name: &str) -> anyhow::Result<f64> {
        self.params.get(name).copied().ok_or_else(|| anyhow::anyhow!("level has no `{name}`"))
    }

    fn tilt(&self) -> anyhow::Result<Tilt> {
        Ok(Tilt { pitch: self.p("pitch_deg")?.to_radians(), roll: self.p("roll_deg")?.to_radians() })
    }

    /// Release point in the plate frame, metres.
    fn start(&self) -> anyhow::Result<[f64; 2]> {
        Ok([self.p("start_x")? * MM, self.p("start_y")? * MM])
    }

    fn marble_r(&self) -> anyhow::Result<f64> {
        Ok(self.p("marble_r")? * MM)
    }

    fn steps(&self) -> anyhow::Result<usize> {
        Ok((self.p("t_end")? / DT).round() as usize)
    }

    /// The same file with some `defparam`s given new values.
    fn with_params(&self, updates: &[(&str, f64)]) -> String {
        let mut out = String::with_capacity(self.source.len());
        for line in self.source.lines() {
            let mut replaced = None;
            for (name, value) in updates {
                let head = format!("[defparam {name} ");
                if let Some(rest) = line.strip_prefix(&head) {
                    let tail = rest.find(']').map(|i| &rest[i..]).unwrap_or("]");
                    replaced = Some(format!("{head}{value:.4}{tail}"));
                }
            }
            out.push_str(&replaced.unwrap_or_else(|| line.to_string()));
            out.push('\n');
        }
        out
    }
}

#[derive(Clone, Copy, Debug)]
struct Tilt {
    /// About the plate's y axis (radians). Positive lowers the +x end.
    pitch: f64,
    /// About the plate's x axis (radians). Positive lowers the −y end.
    roll: f64,
}

impl Tilt {
    /// Plate-frame → world rotation.
    fn rotation(self) -> Mat3 {
        Mat3::rotation_y(self.pitch) * Mat3::rotation_x(self.roll)
    }
    /// The track body's pose. `SpatialTransform::rot` is world→body.
    fn pose(self) -> SpatialTransform {
        SpatialTransform::new(self.rotation().transpose(), Vec3::new(0.0, 0.0, 0.25))
    }
}

// ---- CAD --------------------------------------------------------------------

/// The document with every root wrapped in a `Rotate`, for the tilted view.
fn tilted(doc: &Document, t: Tilt) -> Document {
    let mut d = doc.clone();
    let mut next = d.nodes.keys().copied().max().unwrap_or(0) + 1;
    for root in &mut d.roots {
        d.nodes.insert(
            next,
            Node {
                id: next,
                name: Some("tilt".into()),
                op: CsgOp::Rotate {
                    child: root.root,
                    angles: vcad_ir::Vec3::new(t.roll.to_degrees(), t.pitch.to_degrees(), 0.0),
                },
            },
        );
        root.root = next;
        next += 1;
    }
    d
}

/// `track.stl` (flat, printable) and `track.svg` (at the tilt), from the document.
fn export_track(doc: &Document, t: Tilt, out: &Path) -> anyhow::Result<()> {
    fs::create_dir_all(out)?;
    let scene = vcad_eval::evaluate_document(doc, &vcad_eval::EvalOptions::default())
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let (mut f32s, mut u32s, mut meshes) = (Vec::new(), Vec::new(), Vec::new());
    for part in &scene.parts {
        meshes.push(vcad_kernel_export::StlMeshSpec {
            positions: [f32s.len(), part.mesh.positions.len()],
            indices: [u32s.len(), part.mesh.indices.len()],
        });
        f32s.extend_from_slice(&part.mesh.positions);
        u32s.extend_from_slice(&part.mesh.indices);
    }
    let spec = vcad_kernel_export::StlSpec { name: "newt marble track".into(), meshes };
    let stl = vcad_kernel_export::build_stl(&spec, &f32s, &u32s).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    fs::write(out.join("track.stl"), stl)?;
    let svg = vcad_render::render_svg_str(&tilted(doc, t).to_json()?, 2.0).map_err(|e| anyhow::anyhow!(e))?;
    fs::write(out.join("track.svg"), svg)?;
    println!(
        "cad    {}/track.stl  {} tris, printed flat; track.svg at pitch {:+.2}° roll {:+.2}°",
        out.display(),
        u32s.len() / 3,
        t.pitch.to_degrees(),
        t.roll.to_degrees()
    );
    Ok(())
}

// ---- physics ----------------------------------------------------------------

/// The marble plus the track body, whose colliders and visuals are the
/// document's derivation.
fn build_model(level: &Level, t: Tilt) -> anyhow::Result<Model> {
    let r = level.marble_r()?;
    let m = level.p("marble_g")? * 1e-3;
    let i = 0.4 * m * r * r;
    let marble_inertia = SpatialInertia::new(m, Vec3::zeros(), Mat3::from_diagonal(&Vec3::new(i, i, i)));
    let static_inertia = SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01);

    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(DT)
        .add_free_body("marble", -1, SpatialTransform::identity(), marble_inertia)
        .add_fixed_body("track", -1, t.pose(), static_inertia)
        .build();
    model.bodies[MARBLE].geometry = Some(Geometry::Sphere { radius: r });
    let derived = colliders::colliders_from_document(&level.doc)?;
    model.bodies[TRACK].collisions = derived.colliders.clone();
    model.bodies[TRACK].visuals = derived.colliders;
    Ok(model)
}

fn track_xform(model: &Model) -> SpatialTransform {
    forward_kinematics(model, &model.default_state()).0[TRACK]
}

/// Track-local point (m) → world.
fn local_to_world(model: &Model, p: Vec3) -> Vec3 {
    track_xform(model).body_to_world_point(p)
}

fn q0_for(model: &Model, level: &Level, xy: [f64; 2]) -> DVec {
    let r = level.marble_r().unwrap_or(0.0);
    let p = local_to_world(model, Vec3::new(xy[0], xy[1], r + 0.001));
    let mut q = DVec::zeros(model.nq);
    q[POS] = p.x;
    q[POS + 1] = p.y;
    q[POS + 2] = p.z;
    q
}

fn goal(model: &Model, level: &Level) -> anyhow::Result<Vec3> {
    Ok(local_to_world(model, Vec3::new(level.p("cup_x")? * MM, 0.0, level.marble_r()?)))
}

fn material() -> ContactMaterial {
    ContactMaterial { friction: 0.4, ..Default::default() }
}

fn rollout<'a>(model: &'a Model, q0: DVec, steps: usize, ctrl: &'a dyn Fn(usize) -> DVec) -> ConvexContactRollout<'a> {
    ConvexContactRollout {
        model,
        ground_height: -10.0, // the level is the track; the ground is out of play
        material: material(),
        config: ContactSolverConfig::gradients(),
        q0,
        v0: DVec::zeros(model.nv),
        steps,
        ctrl,
    }
}

/// J = |marble − cup|² at the final step.
fn objective(g: Vec3) -> FinalStateObjective<'static> {
    let g: &'static Vec3 = Box::leak(Box::new(g));
    let value: &'static dyn Fn(&[f64], &[f64]) -> f64 = Box::leak(Box::new(move |q: &[f64], _: &[f64]| {
        let d = Vec3::new(q[POS] - g.x, q[POS + 1] - g.y, q[POS + 2] - g.z);
        d.dot(&d)
    }));
    type GradFn = dyn Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>);
    let gradient: &'static GradFn = Box::leak(Box::new(move |q: &[f64], v: &[f64]| {
        let mut gq = vec![0.0; q.len()];
        gq[POS] = 2.0 * (q[POS] - g.x);
        gq[POS + 1] = 2.0 * (q[POS + 1] - g.y);
        gq[POS + 2] = 2.0 * (q[POS + 2] - g.z);
        (gq, vec![0.0; v.len()])
    }));
    FinalStateObjective { value, gradient }
}

/// Forward rollout with the production simulator.
fn simulate(model: &Model, q0: &DVec, steps: usize) -> (Vec<Vec3>, State) {
    let sim = Simulator::new();
    let mut state = model.default_state();
    state.q = q0.clone();
    let mat = material();
    let mut traj = Vec::with_capacity(steps);
    for _ in 0..steps {
        sim.step_with_contacts(model, &mut state, -10.0, &mat);
        traj.push(Vec3::new(state.q[POS], state.q[POS + 1], state.q[POS + 2]));
    }
    (traj, state)
}

fn in_cup(model: &Model, level: &Level, p: Vec3) -> anyhow::Result<bool> {
    let local = track_xform(model).world_to_body_point(p);
    let d = Vec3::new(local.x - level.p("cup_x")? * MM, local.y, 0.0).norm();
    Ok(d < level.p("cup_r")? * MM && local.z < level.p("cup_h")? * MM)
}

fn verdict(model: &Model, level: &Level, traj: &[Vec3]) -> anyhow::Result<&'static str> {
    Ok(if in_cup(model, level, traj[traj.len() - 1])? { "in the cup" } else { "not in the cup" })
}


// ---- audio ------------------------------------------------------------------

/// The level's acoustic description: geometry in metres, materials from the
/// level's `defparam`s where it declares them and from `audio`'s documented
/// defaults where it does not.
fn track_spec(level: &Level) -> anyhow::Result<audio::TrackSpec> {
    let m = |name: &str, fallback: f64| level.params.get(name).copied().unwrap_or(fallback);
    let track_mat = audio::Material {
        rho: m("track_density", audio::PLA.rho),
        e: m("track_e", audio::PLA.e),
        nu: m("track_nu", audio::PLA.nu),
        loss: m("track_loss", audio::PLA.loss),
    };
    let marble_mat = audio::Material {
        rho: m("marble_density", audio::GLASS.rho),
        e: m("marble_e", audio::GLASS.e),
        nu: m("marble_nu", audio::GLASS.nu),
        loss: m("marble_loss", audio::GLASS.loss),
    };
    Ok(audio::TrackSpec {
        plate: [level.p("plate_x")? * MM, level.p("plate_y")? * MM, level.p("plate_t")? * MM],
        wall: [level.p("wall_h")? * MM, level.p("wall_t")? * MM],
        cup: [
            level.p("cup_x")? * MM,
            level.p("cup_r")? * MM,
            level.p("cup_wall")? * MM,
            level.p("cup_h")? * MM,
        ],
        marble: (level.marble_r()?, level.p("marble_g")? * 1e-3),
        track_mat,
        marble_mat,
    })
}

/// Which part a contact point (track-local, metres) belongs to, and where on it.
fn classify(level: &Level, local: Vec3) -> anyhow::Result<(audio::Part, [f64; 2])> {
    let (px, py) = (level.p("plate_x")? * MM, level.p("plate_y")? * MM);
    let (cx, cr, cw, ch) = (
        level.p("cup_x")? * MM,
        level.p("cup_r")? * MM,
        level.p("cup_wall")? * MM,
        level.p("cup_h")? * MM,
    );
    let d = ((local.x - cx).powi(2) + local.y * local.y).sqrt();
    if d < cr + 3.0 * cw && local.z < ch + 0.02 {
        let a = local.y.atan2(local.x - cx);
        let u = (a / (2.0 * std::f64::consts::PI) + 0.5).clamp(0.0, 1.0);
        return Ok((audio::Part::Cup, [u, (local.z / ch).clamp(0.0, 1.0)]));
    }
    let wall = local.x.abs() > px / 2.0 - 0.005 || local.y.abs() > py / 2.0 - 0.005;
    let u = ((local.x + px / 2.0) / px).clamp(0.0, 1.0);
    let v = ((local.y + py / 2.0) / py).clamp(0.0, 1.0);
    if wall { Ok((audio::Part::Wall, [u, v])) } else { Ok((audio::Part::Plate, [u, v])) }
}

/// The same forward rollout, listening. An impact is a step where the marble's
/// velocity jumps by more than gravity could have done while a contact is open;
/// everything else in contact is rolling — a continuous excitation whose level
/// is the tangential speed.
fn simulate_listening(
    model: &Model,
    level: &Level,
    q0: &DVec,
    steps: usize,
) -> anyhow::Result<(Vec<Vec3>, State, audio::Contacts)> {
    let sim = Simulator::new();
    let mut state = model.default_state();
    state.q = q0.clone();
    let mat = material();
    let mass = level.p("marble_g")? * 1e-3;
    let xf = track_xform(model);
    let mut traj: Vec<Vec3> = Vec::with_capacity(steps);
    let mut out = audio::Contacts::default();
    let mut prev_v = Vec3::zeros();
    let mut p_prev = Vec3::new(q0[POS], q0[POS + 1], q0[POS + 2]);
    // A ball that has just been struck stays in contact for a few steps; one
    // impact per contact episode, not one per step.
    let mut cooldown = 0usize;
    for k in 0..steps {
        sim.step_with_contacts(model, &mut state, -10.0, &mat);
        let p = Vec3::new(state.q[POS], state.q[POS + 1], state.q[POS + 2]);
        let v = (p - p_prev) / DT;
        let dv = v - prev_v + Vec3::new(0.0, 0.0, GRAVITY * DT);
        let contacts = find_contacts(model, &state, 1e-3);
        let t = k as f64 * DT;
        // Of an open manifold, the contact that took the blow is the one whose
        // normal the velocity jump ran along.
        let hit = contacts.iter().max_by(|a, b| {
            dv.dot(&a.contact_normal).abs().total_cmp(&dv.dot(&b.contact_normal).abs())
        });
        if let Some(c) = hit {
            let local = xf.world_to_body_point(c.contact_point);
            let (part, uv) = classify(level, local)?;
            let n = c.contact_normal;
            if dv.dot(&n).abs() > 0.02 && cooldown == 0 {
                let jump = dv.dot(&n).abs();
                out.impacts.push(audio::Impact { t, impulse: mass * jump, speed: 0.5 * jump, part, uv });
                cooldown = 5;
            }
            let tangential = (v - n * v.dot(&n)).norm();
            if tangential > 1e-3 {
                out.rolls.push(audio::Roll { t, speed: tangential, uv });
            }
        }
        cooldown = cooldown.saturating_sub(1);
        prev_v = v;
        p_prev = p;
        traj.push(p);
    }
    Ok((traj, state, out))
}

// ---- frame ------------------------------------------------------------------

fn render(model: &Model, state: &State, path: &Path) -> anyhow::Result<()> {
    let intr = CameraIntrinsics::from_vfov(800, 600, 0.75, 0.05, 5.0);
    let mut cam = RgbdCamera::new(intr)?;
    let scene = Scene::empty();
    let ctx = SensorContext::free_flight(model, state, &scene);
    let rs = RenderScene::from_context(&ctx, &SceneOptions::new());
    let target = Vec3::new(0.0, 0.0, 0.25);
    let pose = CameraPose::look_at(target + Vec3::new(-0.05, -0.42, 0.28), target, Vec3::z());
    let frame = cam.render(&rs, &pose)?;
    let rgba = frame.color_cpu().ok_or_else(|| anyhow::anyhow!("no cpu colour buffer"))?;
    let img = image::RgbaImage::from_raw(frame.width(), frame.height(), rgba.to_vec())
        .ok_or_else(|| anyhow::anyhow!("frame size mismatch"))?;
    img.save(path)?;
    Ok(())
}

// ---- main -------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    let level_path = std::env::args().nth(1).unwrap_or_else(|| "levels/marble.loon".into());
    let out = Path::new("out");
    let ctrl = |_: usize| DVec::zeros(6);

    // 1. the level: one loon file → document → STL, SVG, colliders
    let level = Level::load(Path::new(&level_path))?;
    let tilt = level.tilt()?;
    let start = level.start()?;
    let steps = level.steps()?;
    println!(
        "level  {} → {} IR nodes, {} knobs; pitch {:+.2}° roll {:+.2}°, release ({:+.3}, {:+.3}), t_end {:.2} s",
        level_path,
        level.doc.nodes.len(),
        level.params.len(),
        tilt.pitch.to_degrees(),
        tilt.roll.to_degrees(),
        start[0],
        start[1],
        steps as f64 * DT
    );
    export_track(&level.doc, tilt, out)?;
    let derived = colliders::colliders_from_document(&level.doc)?;
    for w in &derived.warnings {
        println!("warn   {w}");
    }
    let worst = colliders::verify_against_mesh(&level.doc, &derived)?;
    println!(
        "derive {} colliders; support functions agree with the tessellation to {:.3} mm",
        derived.colliders.len(),
        worst / MM
    );
    anyhow::ensure!(worst < 0.5 * MM, "colliders disagree with the CAD");

    // 2. the level as physics
    let model = build_model(&level, tilt)?;
    let g = goal(&model, &level)?;
    let obj = objective(g);
    let q0 = q0_for(&model, &level, start);
    let (traj, final_state) = simulate(&model, &q0, steps);
    let end = traj[traj.len() - 1];
    println!(
        "sim    {} steps at {} ms; marble ends {:.3} m from the cup centre, {}",
        steps,
        DT * 1e3,
        (end - g).norm(),
        verdict(&model, &level, &traj)?
    );
    render(&model, &final_state, &out.join("frame_before.png"))?;

    // 3. the hint, exactly: adjoint on the release point, checked against FD
    let xf = track_xform(&model);
    match convex_adjoint_gradient(&rollout(&model, q0.clone(), steps, &ctrl), &obj) {
        Ok(gr) => {
            let h = 1e-5;
            let mut fd = [0.0; 3];
            for (k, slot) in fd.iter_mut().enumerate() {
                let (mut qp, mut qm) = (q0.clone(), q0.clone());
                qp[POS + k] += h;
                qm[POS + k] -= h;
                let fp = convex_rollout_objective(&rollout(&model, qp, steps, &ctrl), &obj);
                let fm = convex_rollout_objective(&rollout(&model, qm, steps, &ctrl), &obj);
                *slot = (fp - fm) / (2.0 * h);
            }
            println!(
                "adjoint dJ/d(release) = [{:+.4e} {:+.4e} {:+.4e}]\n fd     dJ/d(release) = [{:+.4e} {:+.4e} {:+.4e}]",
                gr.d_q0[POS], gr.d_q0[POS + 1], gr.d_q0[POS + 2], fd[0], fd[1], fd[2]
            );
        }
        Err(e) => println!("adjoint refused: {e}"),
    }

    // Gradient descent with backtracking: the landscape has a cliff at the cup
    // mouth (caught vs deflected), so a step is only accepted if it helps.
    let (px, py, r) = (level.p("plate_x")? * MM, level.p("plate_y")? * MM, level.marble_r()?);
    let clamp = |xy: [f64; 2]| [xy[0].clamp(-px / 2.0 + r, px / 2.0 - r), xy[1].clamp(-py / 2.0 + r, py / 2.0 - r)];
    let j_release = |xy: [f64; 2]| convex_rollout_objective(&rollout(&model, q0_for(&model, &level, xy), steps, &ctrl), &obj);
    let mut xy = start;
    let mut j = j_release(xy);
    let mut step = 0.02;
    for it in 0..12 {
        println!("hint   it {it:2}  release ({:+.3}, {:+.3})  miss {:.4} m", xy[0], xy[1], j.sqrt());
        let (t, _) = simulate(&model, &q0_for(&model, &level, xy), steps);
        if j.sqrt() < 0.002 || in_cup(&model, &level, t[t.len() - 1])? {
            break;
        }
        let gr = match convex_adjoint_gradient(&rollout(&model, q0_for(&model, &level, xy), steps, &ctrl), &obj) {
            Ok(g) => g,
            Err(e) => {
                println!("hint   it {it:2}  adjoint refused: {e}");
                break;
            }
        };
        // world = pos + rotᵀ·local  ⇒  dJ/dlocal = rot · dJ/dworld
        let gl = xf.rot * Vec3::new(gr.d_q0[POS], gr.d_q0[POS + 1], gr.d_q0[POS + 2]);
        let gn = (gl.x * gl.x + gl.y * gl.y).sqrt().max(1e-12);
        let dir = [-gl.x / gn, -gl.y / gn];
        let mut accepted = false;
        for _ in 0..5 {
            let cand = clamp([xy[0] + step * dir[0], xy[1] + step * dir[1]]);
            let jc = j_release(cand);
            if jc < j {
                xy = cand;
                j = jc;
                accepted = true;
                step = (step * 1.5).min(0.02);
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            println!("hint   no descent direction within {:.1} mm; stopping", step * 2e3);
            break;
        }
    }
    let (traj, hinted, heard) = simulate_listening(&model, &level, &q0_for(&model, &level, xy), steps)?;
    println!("hint   final: release ({:+.3}, {:+.3}), {}", xy[0], xy[1], verdict(&model, &level, &traj)?);
    render(&model, &hinted, &out.join("frame_hint.png"))?;
    fs::create_dir_all(out.join("hinted"))?;
    fs::write(
        out.join("hinted/marble.loon"),
        level.with_params(&[("start_x", xy[0] / MM), ("start_y", xy[1] / MM)]),
    )?;

    // 5. the hinted run, heard: the same contacts, as modal synthesis
    let spec = track_spec(&level)?;
    let banks = vec![
        (audio::Part::Plate, spec.plate_bank()),
        (audio::Part::Wall, spec.wall_bank()),
        (audio::Part::Cup, spec.cup_bank()),
    ];
    let marble = audio::marble_bank(spec.marble.0, spec.marble_mat);
    let hz = |v: Vec<f64>| v.iter().map(|f| format!("{f:.0}")).collect::<Vec<_>>().join(" ");
    let t0 = std::time::Instant::now();
    let samples = audio::render(&spec, &banks, &heard, steps as f64 * DT + 0.5);
    let wav = out.join("marble.wav");
    audio::write_wav(&wav, &samples)?;
    for (_, b) in &banks {
        println!("audio  {:<44} {} Hz", b.name, hz(b.top(5)));
    }
    println!(
        "audio  {:<44} {} kHz (ultrasonic — computed, not rendered)",
        marble.name,
        hz(marble.top(5).iter().map(|f| f / 1e3).collect())
    );
    println!(
        "audio  {} impacts, {} rolling steps → {} ({:.2} s, {:.0} ms to render)",
        heard.impacts.len(),
        heard.rolls.len(),
        wav.display(),
        samples.len() as f64 / audio::SR,
        t0.elapsed().as_secs_f64() * 1e3
    );

    // 4. the other knob: tilt, by central differences of the same rollout
    let j_at = |t: Tilt| -> anyhow::Result<f64> {
        let m = build_model(&level, t)?;
        let g = goal(&m, &level)?;
        Ok(convex_rollout_objective(&rollout(&m, q0_for(&m, &level, start), steps, &ctrl), &objective(g)))
    };
    // The landscape is cliffy (caught vs deflected vs stuck on the rim), so
    // look before descending: a coarse grid of ±2° around the level's tilt,
    // then gradient descent from the best cell.
    let mut t = tilt;
    let mut j = j_at(t)?;
    let deg = 1f64.to_radians();
    for dp in -2..=2 {
        for dr in -2..=2 {
            let cand = Tilt { pitch: tilt.pitch + dp as f64 * deg, roll: tilt.roll + dr as f64 * deg };
            let jc = j_at(cand)?;
            if jc < j {
                t = cand;
                j = jc;
            }
        }
    }
    println!("tilt   grid: best of 25 cells is pitch {:+.2}° roll {:+.2}°, miss {:.4} m", t.pitch.to_degrees(), t.roll.to_degrees(), j.sqrt());
    let mut step = 0.5 * deg;
    for it in 0..10 {
        let e = 1e-5;
        let dp = (j_at(Tilt { pitch: t.pitch + e, ..t })? - j_at(Tilt { pitch: t.pitch - e, ..t })?) / (2.0 * e);
        let dr = (j_at(Tilt { roll: t.roll + e, ..t })? - j_at(Tilt { roll: t.roll - e, ..t })?) / (2.0 * e);
        println!(
            "tilt   it {it:2}  pitch {:+.2}°  roll {:+.2}°  miss {:.4} m  dJ/d(pitch,roll) = ({:+.3e}, {:+.3e})",
            t.pitch.to_degrees(),
            t.roll.to_degrees(),
            j.sqrt(),
            dp,
            dr
        );
        if j.sqrt() < 0.002 {
            break;
        }
        let gn = (dp * dp + dr * dr).sqrt().max(1e-12);
        let mut accepted = false;
        for _ in 0..5 {
            let cand = Tilt { pitch: t.pitch - step * dp / gn, roll: t.roll - step * dr / gn };
            let jc = j_at(cand)?;
            if jc < j {
                t = cand;
                j = jc;
                accepted = true;
                step = (step * 1.5).min(1f64.to_radians());
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            println!("tilt   no descent direction within {:.2}°; stopping", (step * 2.0).to_degrees());
            break;
        }
    }
    let solved = build_model(&level, t)?;
    let (traj, solved_state) = simulate(&solved, &q0_for(&solved, &level, start), steps);
    println!("tilt   final: {}", verdict(&solved, &level, &traj)?);
    render(&solved, &solved_state, &out.join("frame_tilted.png"))?;
    export_track(&level.doc, t, &out.join("solved"))?;
    fs::write(
        out.join("solved/marble.loon"),
        level.with_params(&[("pitch_deg", t.pitch.to_degrees()), ("roll_deg", t.roll.to_degrees())]),
    )?;
    Ok(())
}
