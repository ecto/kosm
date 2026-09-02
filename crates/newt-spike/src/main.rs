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
mod frame;
mod garage;
mod lamp;
mod room;

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
use phyz_model::{GeomInstance, Geometry, Model, ModelBuilder, State};
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
///
/// The colliders are passed in rather than re-derived: only the track's *pose*
/// changes as the tilt solver hunts, and decomposing a `Difference` is far too
/// expensive to redo fifty times for an answer that cannot have changed.
fn build_model(level: &Level, t: Tilt, colliders: &[GeomInstance]) -> anyhow::Result<Model> {
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
    model.bodies[TRACK].collisions = colliders.to_vec();
    model.bodies[TRACK].visuals = colliders.to_vec();
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

/// The room the tray is in, from the level's `defparam`s where it declares
/// them and from `room::RoomSpec`'s documented defaults where it does not.
/// Everything but the absorption coefficients is millimetres in the level.
fn room_spec(level: &Level) -> room::RoomSpec {
    let d = room::RoomSpec::default();
    let mm = |name: &str, fallback: f64| level.params.get(name).copied().map(|v| v * MM).unwrap_or(fallback);
    let raw = |name: &str, fallback: f64| level.params.get(name).copied().unwrap_or(fallback);
    room::RoomSpec {
        dims: [mm("room_x", d.dims[0]), mm("room_y", d.dims[1]), mm("room_z", d.dims[2])],
        table: [mm("table_x", d.table[0]), mm("table_y", d.table[1]), mm("table_z", d.table[2])],
        ear: [mm("ear_x", d.ear[0]), mm("ear_y", d.ear[1]), mm("ear_z", d.ear[2])],
        absorb: [
            raw("room_absorb_floor", d.absorb[0]),
            raw("room_absorb_ceiling", d.absorb[1]),
            raw("room_absorb_walls", d.absorb[2]),
        ],
        order: raw("room_order", d.order as f64).round().max(0.0) as usize,
        ear_spacing: mm("ear_spacing", d.ear_spacing),
    }
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
                out.impacts.push(audio::Impact {
                    t,
                    impulse: mass * jump,
                    speed: 0.5 * jump,
                    part,
                    uv,
                    pos: [local.x, local.y, local.z],
                });
                cooldown = 5;
            }
            let tangential = (v - n * v.dot(&n)).norm();
            if tangential > 1e-3 {
                // The load on the contact is the marble's weight resolved onto
                // the plate; phyz does not hand back the solved normal force.
                let normal = mass * -GRAVITY * n.z.abs();
                out.rolls.push(audio::Roll {
                    t,
                    speed: tangential,
                    uv,
                    pos: [local.x, local.y, local.z],
                    normal,
                });
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
    render_with(model, state, path, None)
}

fn render_with(model: &Model, state: &State, path: &Path, lamp: Option<(&lamp::Lamp, f64)>) -> anyhow::Result<()> {
    let intr = CameraIntrinsics::from_vfov(800, 600, 0.75, 0.05, 5.0);
    let mut cam = RgbdCamera::new(intr)?;
    let scene = Scene::empty();
    let ctx = SensorContext::free_flight(model, state, &scene);
    let rs = RenderScene::from_context(&ctx, &SceneOptions::new());
    let target = Vec3::new(0.0, 0.0, 0.25);
    let pose = CameraPose::look_at(target + Vec3::new(-0.05, -0.42, 0.28), target, Vec3::z());
    let frame = cam.render(&rs, &pose)?;
    let rgba = frame.color_cpu().ok_or_else(|| anyhow::anyhow!("no cpu colour buffer"))?;
    let mut img = image::RgbaImage::from_raw(frame.width(), frame.height(), rgba.to_vec())
        .ok_or_else(|| anyhow::anyhow!("frame size mismatch"))?;
    if let Some((l, r)) = lamp {
        let c = Vec3::new(state.q[POS], state.q[POS + 1], state.q[POS + 2]);
        lamp::draw(&mut img, &pose, &intr, l, &track_xform(model), c, r);
    }
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
    for n in &derived.notes {
        println!("derive {n}");
    }
    let worst = colliders::verify_against_mesh(&level.doc, &derived)?;
    println!(
        "derive {} colliders; support functions agree with the tessellation to {:.3} mm",
        derived.colliders.len(),
        worst / MM
    );
    anyhow::ensure!(worst < 0.5 * MM, "colliders disagree with the CAD");
    // The support test only ever sees the outside of the level, where a cup and
    // a puck are the same shape. Cut geometry gets the second check.
    if !derived.removed.is_empty() {
        let intrusion = colliders::verify_no_intrusion(&derived);
        println!(
            "derive {} cut volume(s); no collider reaches more than {:.3} mm into what was removed",
            derived.removed.len(),
            intrusion / MM
        );
        anyhow::ensure!(intrusion < 0.5 * MM, "colliders fill a hole the CAD cut out");
    }

    // 2. the level as physics
    let model = build_model(&level, tilt, &derived.colliders)?;
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
    // The dry sound is rendered onto three fixed source positions — where the
    // marble is let go, the cup, and the end wall — and each of those gets its
    // own room impulse response, so the reflections and the pan move with the
    // marble without a crossfade smearing the impacts.
    let anchors = [
        [start[0], start[1], spec.marble.0],
        [level.p("cup_x")? * MM, 0.0, 0.02],
        [level.p("plate_x")? * MM * 0.5, 0.0, 0.02],
    ];
    let air = room_spec(&level);
    let rt60 = air.rt60();
    let roughness = level.params.get("track_roughness_mm").copied().unwrap_or(0.2) * MM;
    let dry = audio::render_dry(
        &spec,
        &banks,
        &heard,
        steps as f64 * DT + 0.5,
        &anchors,
        roughness,
    );
    let buses: Vec<_> = anchors.iter().copied().zip(dry).collect();
    let (samples, room_summary) =
        room::mix(&air, &buses, 1, audio::SR, steps as f64 * DT + rt60);
    let wav = out.join("marble.wav");
    room::write_wav_stereo(&wav, &samples, audio::SR)?;
    for (_, b) in &banks {
        println!("audio  {:<44} {} Hz", b.name, hz(b.top(5)));
    }
    println!(
        "audio  {:<44} {} kHz (ultrasonic — computed, not rendered)",
        marble.name,
        hz(marble.top(5).iter().map(|f| f / 1e3).collect())
    );
    println!(
        "room   {:.0}×{:.0}×{:.0} cm, RT60 {:.2} s (Eyring, ᾱ={:.2}), {} image sources, ears {:.0} cm apart at {:.2} m, DRR at the cup {:+.1} dB",
        air.dims[0] * 100.0,
        air.dims[1] * 100.0,
        air.dims[2] * 100.0,
        room_summary.rt60,
        air.mean_absorption(),
        room_summary.images,
        air.ear_spacing * 100.0,
        room_summary.ear_distance,
        room_summary.drr_db,
    );
    println!(
        "audio  {} impacts, {} rolling steps → {} (stereo, {:.2} s, {:.0} ms to render)",
        heard.impacts.len(),
        heard.rolls.len(),
        wav.display(),
        samples.len() as f64 / 2.0 / audio::SR,
        t0.elapsed().as_secs_f64() * 1e3
    );

    // 4. the other knob: tilt, by central differences of the same rollout
    let j_at = |t: Tilt| -> anyhow::Result<f64> {
        let m = build_model(&level, t, &derived.colliders)?;
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
    let solved = build_model(&level, t, &derived.colliders)?;
    let (traj, solved_state) = simulate(&solved, &q0_for(&solved, &level, start), steps);
    println!("tilt   final: {}", verdict(&solved, &level, &traj)?);
    render(&solved, &solved_state, &out.join("frame_tilted.png"))?;
    export_track(&level.doc, t, &out.join("solved"))?;
    fs::write(
        out.join("solved/marble.loon"),
        level.with_params(&[("pitch_deg", t.pitch.to_degrees()), ("roll_deg", t.roll.to_degrees())]),
    )?;

    // 5. the lamp: the objective is the marble's *shadow*. light is analytic,
    //    motion is the adjoint, one chain rule joins them.
    if level.params.contains_key("lamp_x") {
        let lamp0 = lamp::Lamp {
            pos: Vec3::new(level.p("lamp_x")? * MM, level.p("lamp_y")? * MM, 0.25 + level.p("lamp_z")? * MM),
            target: [level.p("shadow_x")? * MM, level.p("shadow_y")? * MM],
        };
        let r = level.marble_r()?;
        let sobj = lamp::objective(lamp0, xf);
        let q0_fn = |xy: [f64; 2]| q0_for(&model, &level, xy);
        let roll_fn = |q: DVec| rollout(&model, q, steps, &ctrl);
        let (s0, _) = lamp::shadow(&lamp0, &xf, traj_end(&model, &q0_fn(start), steps));
        println!(
            "lamp   at ({:+.2}, {:+.2}, {:+.2}) m; shadow of the release run lands at ({:+.3}, {:+.3}), target ({:+.3}, {:+.3})",
            lamp0.pos.x, lamp0.pos.y, lamp0.pos.z, s0[0], s0[1], lamp0.target[0], lamp0.target[1]
        );
        // adjoint check on the chained objective, at a horizon before the cup
        // (smooth rolling, where the gradient is well defined) and at t_end.
        for (label, n) in [("0.5 s, rolling", steps.min(500)), ("t_end, after the cup", steps)] {
            let roll_n = |q: DVec| rollout(&model, q, n, &ctrl);
            if let Ok(gr) = convex_adjoint_gradient(&roll_n(q0_fn(start)), &sobj) {
                let h = 1e-5;
                let mut fd = [0.0; 2];
                for (k, slot) in fd.iter_mut().enumerate() {
                    let (mut qp, mut qm) = (q0_fn(start), q0_fn(start));
                    qp[POS + k] += h;
                    qm[POS + k] -= h;
                    *slot = (convex_rollout_objective(&roll_n(qp), &sobj) - convex_rollout_objective(&roll_n(qm), &sobj)) / (2.0 * h);
                }
                println!(
                    "lamp   ∂shadow²/∂release at {label}: adjoint [{:+.4e} {:+.4e}]  fd [{:+.4e} {:+.4e}]",
                    gr.d_q0[POS], gr.d_q0[POS + 1], fd[0], fd[1]
                );
            }
        }
        // knob 1: the release point, by the chained adjoint
        let (xy, miss) = lamp::solve_release(&model, &xf, &sobj, &q0_fn, &clamp, &roll_fn, start, &|s| println!("{s}"));
        let (traj, st) = simulate(&model, &q0_fn(xy), steps);
        println!("lamp   release solved to ({:+.3}, {:+.3}): shadow miss {:.4} m, marble {}", xy[0], xy[1], miss, verdict(&model, &level, &traj)?);
        let ld = lamp::out_dir(out);
        fs::create_dir_all(&ld)?;
        render_with(&model, &st, &ld.join("frame_release.png"), Some((&lamp0, r)))?;
        fs::write(ld.join("marble.loon"), level.with_params(&[("start_x", xy[0] / MM), ("start_y", xy[1] / MM)]))?;
        // knob 2: the lamp, in closed form, for the original release
        let c_end = traj_end(&model, &q0_fn(start), steps);
        let moved = lamp::Lamp { pos: lamp::lamp_for(&lamp0, &xf, c_end), ..lamp0 };
        let (s1, _) = lamp::shadow(&moved, &xf, c_end);
        println!(
            "lamp   or move the lamp to ({:+.3}, {:+.3}, {:+.3}) m: shadow lands at ({:+.3}, {:+.3})",
            moved.pos.x, moved.pos.y, moved.pos.z, s1[0], s1[1]
        );
        let (_, st0) = simulate(&model, &q0_fn(start), steps);
        render_with(&model, &st0, &ld.join("frame_lamp.png"), Some((&moved, r)))?;
        fs::write(
            ld.join("marble-lamp.loon"),
            level.with_params(&[("lamp_x", moved.pos.x / MM), ("lamp_y", moved.pos.y / MM)]),
        )?;

        // 6. the frame as a solver: a ray caster generic over tang::Scalar.
        //    f64 renders the image with real shadows; Dual differentiates it.
        //    the objective is the brightness of the target disc *in the image*.
        let intr = CameraIntrinsics::from_vfov(800, 600, 0.75, 0.05, 5.0);
        let target = Vec3::new(0.0, 0.0, 0.25);
        let pose = CameraPose::look_at(target + Vec3::new(-0.05, -0.42, 0.28), target, Vec3::z());
        let lamp_r = level.params.get("lamp_r").copied().unwrap_or(25.0) * MM;
        let colliders = model.bodies[TRACK].collisions.clone();
        let scene_at = |c: Vec3| frame::Scene::<f64>::new(&xf, &colliders, tang::Vec3::new(c.x, c.y, c.z), r, lamp0.pos, lamp_r);
        let fd = frame::out_dir(out);
        fs::create_dir_all(&fd)?;
        let c_end = traj_end(&model, &q0_fn(start), steps);
        let t0 = std::time::Instant::now();
        frame::render(&scene_at(c_end), &pose, &intr).save(fd.join("frame_release.png"))?;
        println!("frame  ray cast {}×{} with penumbra shadows in {:.0} ms → {}/frame_release.png", intr.width, intr.height, t0.elapsed().as_secs_f64() * 1e3, fd.display());
        let pixels = frame::target_pixels(&scene_at(c_end), &pose, &intr, &xf, lamp0.target, 0.05);
        let points = frame::patch_points(&scene_at(c_end), &pose, &intr, &pixels);
        let (j_far, _) = frame::visibility_and_grad(&scene_at(c_end), &points);
        // the gradient check where the shadow overlaps the patch: the lamp-stage release
        let c_near = traj_end(&model, &q0_fn(xy), steps);
        let (j, g) = frame::visibility_and_grad(&scene_at(c_near), &points);
        // duals vs central differences of the shadow pass; the penumbra is ~1 mm wide, so h is small
        let h = 1e-6;
        let mut fdg = [0.0; 3];
        for k in 0..3 {
            let mut cp = c_near;
            let mut cm = c_near;
            match k { 0 => { cp.x += h; cm.x -= h; } 1 => { cp.y += h; cm.y -= h; } _ => { cp.z += h; cm.z -= h; } }
            fdg[k] = (frame::patch_visibility(&scene_at(cp), &points) - frame::patch_visibility(&scene_at(cm), &points)) / (2.0 * h);
        }
        println!(
            "frame  shadow pass over {} plate points: lamp visibility {:.4} with the shadow near, {:.4} far away\nframe  ∂visibility/∂marble  dual [{:+.4e} {:+.4e} {:+.4e}]  fd [{:+.4e} {:+.4e} {:+.4e}]",
            points.len(), j, j_far, g[0], g[1], g[2], fdg[0], fdg[1], fdg[2]
        );
        for gy in [-0.075, -0.05, -0.025, 0.0, 0.025] {
            let c = traj_end(&model, &q0_fn([-0.12, gy]), steps);
            let cl = xf.world_to_body_point(c);
            println!("frame  grid probe: release (-0.120, {gy:+.3}) → marble ends ({:+.3}, {:+.3}), visibility {:.4}", cl.x, cl.y, frame::patch_visibility(&scene_at(c), &points));
        }
        // image → marble → contact → release, in one backward pass
        let iobj = frame::shadow_objective(scene_at(c_end), points.clone());
        let (xy_img, jimg) = lamp::solve_release(&model, &xf, &iobj, &q0_fn, &clamp, &roll_fn, start, &|s| println!("{}", s.replace("lamp   ", "frame  ").replace("shadow miss", "patch visibility")));
        let c_img = traj_end(&model, &q0_fn(xy_img), steps);
        frame::render(&scene_at(c_img), &pose, &intr).save(fd.join("frame_solved.png"))?;
        println!("frame  release solved on the shadow pass to ({:+.3}, {:+.3}): patch visibility {:.4} (lit: {:.4})", xy_img[0], xy_img[1], jimg * jimg, j_far);
        fs::write(fd.join("marble.loon"), level.with_params(&[("start_x", xy_img[0] / MM), ("start_y", xy_img[1] / MM)]))?;
    }

    // 7. a captured place: the marble on the real garage floor, and the
    //    splat as the frame's backdrop. rung 4 in miniature.
    let map_dir = std::env::var("NEWT_MAP").unwrap_or_else(|_| "/Users/cam/Developer/ipse/maps/garage-perim".into());
    if Path::new(&map_dir).join("map.toml").exists() {
        garage_stage(&level, Path::new(&map_dir), out)?;
    }
    Ok(())
}

fn garage_stage(level: &Level, map_dir: &Path, out: &Path) -> anyhow::Result<()> {
    let g = garage::Garage::load(map_dir)?;
    let (lo, hi) = g.extents();
    let (cx, cy) = g.centre_xy();
    println!(
        "garage {}: sdf {:.1}×{:.1}×{:.1} m at {:.2} m cells; floor under the centre at z = {:+.3} m",
        map_dir.display(), hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, g.map.sdf.as_ref().map(|s| s.cell).unwrap_or(0.0), g.floor_z
    );
    let r = level.marble_r()?;
    let model = garage::marble_model(r, level.p("marble_g")? * 1e-3);
    let mat = material();
    let t0 = std::time::Instant::now();
    let (path, state) = garage::roll(&model, &g, (cx, cy), 2.5, &mat)?;
    let (a, b) = (path[0], path[path.len() - 1]);
    let drift = Vec3::new(b.x - a.x, b.y - a.y, 0.0);
    let secs = path.len() as f64 * model.dt;
    println!(
        "garage marble dropped at ({:+.2}, {:+.2}); after {:.2} s it is at ({:+.3}, {:+.3}, {:+.3}), {:.1} cm away, heading ({:+.2}, {:+.2}). {} ms of physics",
        cx, cy, secs, b.x, b.y, b.z, drift.norm() * 100.0, drift.x / drift.norm().max(1e-9), drift.y / drift.norm().max(1e-9), t0.elapsed().as_millis()
    );
    if std::env::var_os("NEWT_TRACE").is_some() {
        // what moves the marble on a flat scan?
        for (label, rr, mu) in [("r = 1 cm, μ = 0.4", r, 0.4), ("r = 1 cm, μ = 0", r, 0.0), ("r = 5 cm, μ = 0.4", 0.05, 0.4), ("r = 5 cm, μ = 0", 0.05, 0.0)] {
            let m = garage::marble_model(rr, level.p("marble_g")? * 1e-3 * (rr / r).powi(3));
            let mat = ContactMaterial { friction: mu, ..Default::default() };
            let (pth, _) = garage::roll(&m, &g, (cx, cy), 2.5, &mat)?;
            let d = pth[pth.len() - 1] - pth[0];
            eprintln!("garage experiment {label}: rolled {:.1} cm in {:.2} s", d.x.hypot(d.y) * 100.0, pth.len() as f64 * m.dt);
        }
        let sdf = g.map.sdf.as_ref().unwrap();
        for k in 0..=8 {
            let p = a + (b - a) * (k as f64 / 8.0);
            let fz = g.floor_at(p.x, p.y);
            let grad = fz.and_then(|z| sdf.sample_with_gradient(Vec3::new(p.x, p.y, z + 0.01)));
            let (gx, gy, gz, gn) = grad.map(|(_, n)| (n.x, n.y, n.z, n.norm())).unwrap_or((0.0, 0.0, 0.0, 0.0));
            eprintln!("garage path {k}/8 at ({:+.3},{:+.3}): floor z {:+.4} m; ∇sdf 1 cm above = ({:+.3},{:+.3},{:+.3}) |∇|={:.3}, tilt {:.2}°",
                p.x, p.y, fz.unwrap_or(f64::NAN), gx, gy, gz, gn, (gx.hypot(gy)/gz.abs().max(1e-9)).atan().to_degrees());
        }
    }
    // the floor the marble actually crossed, read back from the level set
    let profile: Vec<(f64, f64)> = (0..=16)
        .map(|k| {
            let p = a + (b - a) * (k as f64 / 16.0);
            (drift.norm() * k as f64 / 16.0, g.floor_at(p.x, p.y).unwrap_or(f64::NAN))
        })
        .collect();
    let z0 = profile[0].1;
    let (deepest_s, deepest) = profile.iter().copied().fold((0.0, z0), |acc, (s, z)| if z < acc.1 { (s, z) } else { acc });
    let grade_in = (z0 - deepest) / deepest_s.max(1e-9);
    println!(
        "garage floor along the path: {:.1} mm lower {:.0} cm in, back to {:+.1} mm at the end. that first stretch is a {:.1}% grade; rolling on it predicts {:.2} m/s², the marble averaged {:.2} m/s²",
        (z0 - deepest) * 1e3, deepest_s * 100.0, (profile[16].1 - z0) * 1e3, grade_in * 100.0,
        5.0 / 7.0 * GRAVITY * grade_in.atan().sin(), 2.0 * drift.norm() / (secs * secs)
    );
    // the frame: splat backdrop, ray-cast marble on top
    let (w, h) = (800u32, 600u32);
    let intr = CameraIntrinsics::from_vfov(w, h, 0.75, 0.05, 50.0);
    let target = Vec3::new(b.x, b.y, b.z);
    // Up close the splat is mush (ipse-map's own warning: the SDF exists for a
    // reason), so the frame is the room view with the marble's path drawn on.
    let eye = match std::env::var("NEWT_VIEW").ok().as_deref() {
        Some("top") => Vec3::new(cx, cy, 3.0),
        Some("close") => target + Vec3::new(-0.35, -0.6, 0.45),
        _ => Vec3::new(cx + 2.4, cy - 2.4, 1.5),
    };
    let target = Vec3::new((a.x + b.x) * 0.5, (a.y + b.y) * 0.5, 0.0);
    let t1 = std::time::Instant::now();
    let (rgb, count) = garage::render_splat(&g, eye, target, intr.fx, w, h)?;
    let mut img = image::RgbaImage::new(w, h);
    for (i, px) in img.pixels_mut().enumerate() {
        let c = |k: usize| (rgb[i * 3 + k].clamp(0.0, 1.0) * 255.0) as u8;
        *px = image::Rgba([c(0), c(1), c(2), 255]);
    }
    // the marble, ray cast with the same generic caster, composited where it hits
    let pose = CameraPose::look_at(eye, target, Vec3::z());
    let lamp = Vec3::new(target.x, target.y, target.z + 2.0);
    let scene = frame::Scene::<f64>::new(&SpatialTransform::identity(), &[], tang::Vec3::new(b.x, b.y, b.z), r, lamp, 0.05);
    let hits = frame::composite_marble(&scene, &pose, &intr, &mut img);
    // the path, as amber dots on the floor
    for p in path.iter().step_by(10) {
        if let Some((u, v)) = pose.project(&intr, *p) {
            if u >= 0.0 && v >= 0.0 && (u as u32) < w && (v as u32) < h {
                img.put_pixel(u as u32, v as u32, image::Rgba([235, 170, 50, 255]));
            }
        }
    }
    let gd = out.join("garage");
    fs::create_dir_all(&gd)?;
    img.save(gd.join("frame.png"))?;
    println!("garage frame: {count} gaussians rendered by tang-3dgs in {} ms, marble composited over {hits} pixels → {}/frame.png", t1.elapsed().as_millis(), gd.display());
    let _ = state;
    Ok(())
}

fn traj_end(model: &Model, q0: &DVec, steps: usize) -> Vec3 {
    let (t, _) = simulate(model, q0, steps);
    t[t.len() - 1]
}
