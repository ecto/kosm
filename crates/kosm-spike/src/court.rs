//! The court: basketballs dropped on a hardwood slab.
//!
//! The first slice of a basketball level. The slab is the authored scene's
//! vcad geometry, derived into phyz colliders the same way the marble track
//! is; the balls are phyz free bodies; the bounce is phyz's own restitution,
//! entering the convex contact solve as a target normal velocity (not a
//! post-solve velocity flip), with the low-speed ramp that lets a ball come to
//! rest instead of micro-bouncing forever.
//!
//! What is checked, and printed: each ball's apexes. With coefficient of
//! restitution `e`, the bottom of the ball after bounce `k` should reach
//! `e^{2k}` of the drop height, and successive apexes should be in the ratio
//! `e²`. That is the rulebook's inflation test, run in the simulator.
//!
//! Metres, z up, the slab's top at z = 0.

use std::path::Path;

use phyz::Simulator;
use phyz_camera::{CameraPose, RenderScene, RgbdCamera, SceneOptions};
use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder, State};
use phyz_world::{CameraIntrinsics, Scene, SensorContext};

use crate::colliders;
use crate::scene::{AuthoredScene, MM};

pub const DEFAULT_COURT_SCENE: &str = "court.loon";

/// The court scene's knobs, resolved to simulation units.
pub struct CourtScene {
    pub authored: AuthoredScene,
    pub n_balls: usize,
    pub ball_r: f64,
    pub ball_mass: f64,
    pub restitution: f64,
    pub friction: f64,
    /// Bottom of the ball above the floor at release.
    pub drop: f64,
    pub spacing: f64,
    pub stagger: f64,
    pub t_end: f64,
    pub fps: f64,
    pub dt: f64,
}

impl CourtScene {
    pub fn bundled() -> anyhow::Result<Self> {
        Self::load(AuthoredScene::bundled_path(DEFAULT_COURT_SCENE))
    }

    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let authored = AuthoredScene::load(path)?;
        let scene = Self {
            n_balls: authored.parameter("n_balls")?.round().max(1.0) as usize,
            ball_r: authored.millimetres("ball_r_mm")?,
            ball_mass: authored.parameter("ball_g")? * 1e-3,
            restitution: authored.parameter("restitution")?,
            friction: authored.parameter("friction")?,
            drop: authored.millimetres("drop_mm")?,
            spacing: authored.millimetres("spacing_mm")?,
            stagger: authored.millimetres("stagger_mm")?,
            t_end: authored.parameter("t_end")?,
            fps: authored.parameter("fps")?,
            dt: authored.parameter("dt_ms")? * 1e-3,
            authored,
        };
        anyhow::ensure!(scene.dt > 0.0 && scene.fps > 0.0, "court needs a positive dt and fps");
        Ok(scene)
    }

    pub fn frames(&self) -> usize {
        (self.t_end * self.fps).round() as usize
    }

    /// Where ball `i` is released: a line along x, centred, each ball a little
    /// higher than the last so the bounces do not all land on the same beat.
    pub fn release(&self, i: usize) -> Vec3 {
        let x = (i as f64 - (self.n_balls as f64 - 1.0) / 2.0) * self.spacing;
        Vec3::new(x, 0.0, self.drop + self.ball_r + i as f64 * self.stagger)
    }

    pub fn material(&self) -> ContactMaterial {
        ContactMaterial {
            friction: self.friction,
            restitution: self.restitution,
            ..Default::default()
        }
    }
}

/// An apex: a moment a ball stopped rising.
#[derive(Clone, Copy, Debug)]
pub struct Apex {
    pub ball: usize,
    pub t: f64,
    /// Bottom of the ball above the floor.
    pub height: f64,
}

/// The balls and the slab, stepping.
pub struct Court {
    pub model: Model,
    pub state: State,
    pub apexes: Vec<Apex>,
    material: ContactMaterial,
    sim: Simulator,
    ball_r: f64,
    n_balls: usize,
    q_pos: Vec<usize>,
    v_lin: Vec<usize>,
    prev_vz: Vec<f64>,
}

impl Court {
    pub fn from_scene(scene: &CourtScene) -> anyhow::Result<Self> {
        let derived = colliders::colliders_from_document(&scene.authored.document)?;
        anyhow::ensure!(!derived.colliders.is_empty(), "the court scene has no geometry to stand on");
        let worst = colliders::verify_against_mesh(&scene.authored.document, &derived)?;
        anyhow::ensure!(worst < 0.5 * MM, "court colliders disagree with the CAD by {:.3} mm", worst / MM);

        let (r, m) = (scene.ball_r, scene.ball_mass);
        // a hollow sphere: I = 2/3 m r²
        let i = 2.0 / 3.0 * m * r * r;
        let ball_inertia = SpatialInertia::new(m, Vec3::zeros(), Mat3::from_diagonal(&Vec3::new(i, i, i)));
        let static_inertia = SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01);

        let mut builder = ModelBuilder::new().gravity(Vec3::new(0.0, 0.0, -GRAVITY)).dt(scene.dt);
        for k in 0..scene.n_balls {
            builder = builder.add_free_body(&format!("ball{k}"), -1, SpatialTransform::identity(), ball_inertia);
        }
        let mut model = builder
            .add_fixed_body("court", -1, SpatialTransform::identity(), static_inertia)
            .build();
        for k in 0..scene.n_balls {
            model.bodies[k].geometry = Some(Geometry::Sphere { radius: r });
        }
        let court = scene.n_balls;
        model.bodies[court].collisions = derived.colliders.clone();
        model.bodies[court].visuals = derived.colliders;

        let q_pos: Vec<usize> = (0..scene.n_balls).map(|k| model.q_offsets[model.bodies[k].joint_idx] + 3).collect();
        let v_lin: Vec<usize> = (0..scene.n_balls).map(|k| model.v_offsets[model.bodies[k].joint_idx] + 3).collect();

        let mut state = model.default_state();
        for k in 0..scene.n_balls {
            let p = scene.release(k);
            state.q[q_pos[k]] = p.x;
            state.q[q_pos[k] + 1] = p.y;
            state.q[q_pos[k] + 2] = p.z;
        }

        let sim = Simulator::new().with_contact_config(ContactSolverConfig::simulation());

        Ok(Self {
            model,
            state,
            apexes: Vec::new(),
            material: scene.material(),
            sim,
            ball_r: r,
            n_balls: scene.n_balls,
            q_pos,
            v_lin,
            prev_vz: vec![0.0; scene.n_balls],
        })
    }

    pub fn time(&self) -> f64 {
        self.state.time
    }

    pub fn centre(&self, ball: usize) -> Vec3 {
        let q = self.q_pos[ball];
        Vec3::new(self.state.q[q], self.state.q[q + 1], self.state.q[q + 2])
    }

    pub fn velocity(&self, ball: usize) -> Vec3 {
        let v = self.v_lin[ball];
        Vec3::new(self.state.v[v], self.state.v[v + 1], self.state.v[v + 2])
    }

    pub fn centres(&self) -> Vec<Vec3> {
        (0..self.n_balls).map(|k| self.centre(k)).collect()
    }

    /// One step. The ground plane is far below the slab: the level is the court.
    pub fn step(&mut self) {
        self.sim.step_with_contacts(&self.model, &mut self.state, -10.0, &self.material);
        for k in 0..self.n_balls {
            let vz = self.velocity(k).z;
            if self.prev_vz[k] > 0.0 && vz <= 0.0 {
                self.apexes.push(Apex { ball: k, t: self.state.time, height: self.centre(k).z - self.ball_r });
            }
            self.prev_vz[k] = vz;
        }
    }

    /// Apex heights of one ball, in order.
    pub fn apexes_of(&self, ball: usize) -> Vec<f64> {
        self.apexes.iter().filter(|a| a.ball == ball).map(|a| a.height).collect()
    }
}

/// Rendered with phyz-camera: the model's own geometry, balls and slab.
pub fn render(court: &Court, width: u32, height: u32) -> anyhow::Result<image::RgbaImage> {
    let intr = CameraIntrinsics::from_vfov(width, height, 0.8, 0.05, 30.0);
    let mut cam = RgbdCamera::new(intr)?;
    let scene = Scene::empty();
    let ctx = SensorContext::free_flight(&court.model, &court.state, &scene);
    let opts = SceneOptions { body_albedo: [0.80, 0.42, 0.18], ..SceneOptions::new() };
    let rs = RenderScene::from_context(&ctx, &opts);
    let pose = CameraPose::look_at(Vec3::new(0.6, -4.6, 1.4), Vec3::new(0.0, 0.0, 0.9), Vec3::z());
    let frame = cam.render(&rs, &pose)?;
    let rgba = frame.color_cpu().ok_or_else(|| anyhow::anyhow!("no cpu colour buffer"))?;
    image::RgbaImage::from_raw(frame.width(), frame.height(), rgba.to_vec())
        .ok_or_else(|| anyhow::anyhow!("frame size mismatch"))
}

/// Drop the balls, record, report the bounces against `e²`, encode.
pub fn run(out: &Path, frames: Option<usize>, width: u32, height: u32) -> anyhow::Result<()> {
    let scene = CourtScene::bundled()?;
    for warning in &scene.authored.warnings {
        println!("scene  {warning}");
    }
    let frames = frames.unwrap_or_else(|| scene.frames());
    let dir = out.join("court");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;

    let mut court = Court::from_scene(&scene)?;
    println!(
        "court  {} balls, r {:.0} mm, {:.0} g, e {:.2}, μ {:.2}; dropped from {:.2} m; dt {:.1} ms, {} frames at {:.0} fps",
        scene.n_balls,
        scene.ball_r / MM,
        scene.ball_mass * 1e3,
        scene.restitution,
        scene.friction,
        scene.drop,
        scene.dt * 1e3,
        frames,
        scene.fps
    );
    let steps_per_frame = (1.0 / scene.fps / scene.dt).round().max(1.0) as usize;
    let t0 = std::time::Instant::now();
    let mut sim_time = std::time::Duration::ZERO;
    for k in 0..frames {
        let lap = std::time::Instant::now();
        for _ in 0..steps_per_frame {
            court.step();
        }
        sim_time += lap.elapsed();
        render(&court, width, height)?.save(dir.join(format!("frame_{k:03}.png")))?;
    }
    println!(
        "court  {:.2} s simulated in {} ms; {} frames rendered in {:.1} s",
        court.time(),
        sim_time.as_millis(),
        frames,
        t0.elapsed().as_secs_f64()
    );
    report(&scene, &court);

    let mp4 = out.join("court.mp4");
    let st = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-framerate", &format!("{}", scene.fps), "-i"])
        .arg(dir.join("frame_%03d.png"))
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17"])
        .arg(&mp4)
        .status();
    match st {
        Ok(s) if s.success() => println!("court  {} frames → {}", frames, mp4.display()),
        Ok(s) => println!("court  ffmpeg exited with {s}; frames are in {}", dir.display()),
        Err(e) => println!("court  no ffmpeg ({e}); frames are in {}", dir.display()),
    }
    Ok(())
}

/// Each ball's apexes against the `e²` law.
pub fn report(scene: &CourtScene, court: &Court) {
    let e2 = scene.restitution * scene.restitution;
    for k in 0..scene.n_balls {
        let drop = scene.drop + k as f64 * scene.stagger;
        let apexes = court.apexes_of(k);
        let mut line = format!("bounce ball {k}: from {:.3} m →", drop);
        let mut prev = drop;
        for (j, h) in apexes.iter().enumerate().take(6) {
            let predicted = prev * e2;
            line.push_str(&format!(" {:.3} m ({:+.1}%)", h, (h / predicted - 1.0) * 100.0));
            prev = *h;
            if j == 0 {
                line.push_str(&format!(" [rulebook {:.3}]", drop * e2));
            }
        }
        if apexes.len() > 6 {
            line.push_str(&format!(" … {} bounces", apexes.len()));
        }
        println!("{line}");
    }
}
