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
use phyz_contact::{ContactMaterial, ContactSolverConfig};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Model, ModelBuilder, State};
use phyz_rigid::forward_kinematics;

use crate::colliders;
use crate::scene::{AuthoredScene, MM};

pub mod parts;
pub mod render;

pub const DEFAULT_COURT_SCENE: &str = "court.loon";

/// The court scene's knobs, resolved to simulation units.
pub struct CourtScene {
    pub authored: AuthoredScene,
    pub n_balls: usize,
    pub drop_x: f64,
    pub shot: Option<Shot>,
    pub hoop: Hoop,
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
            n_balls: authored.parameter("n_balls")?.round().max(0.0) as usize,
            drop_x: authored.millimetres("drop_x_mm")?,
            shot: Shot::from_scene(&authored)?,
            hoop: Hoop::from_scene(&authored)?,
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
        anyhow::ensure!(scene.n_balls + scene.shot.is_some() as usize > 0, "court has no balls");
        Ok(scene)
    }

    /// Bodies in the model: the dropped balls, then the shot if there is one.
    pub fn bodies(&self) -> usize {
        self.n_balls + self.shot.is_some() as usize
    }

    pub fn frames(&self) -> usize {
        (self.t_end * self.fps).round() as usize
    }

    /// Where ball `i` is released: a line along x, centred, each ball a little
    /// higher than the last so the bounces do not all land on the same beat.
    pub fn release(&self, i: usize) -> Vec3 {
        let y = (i as f64 - (self.n_balls as f64 - 1.0) / 2.0) * self.spacing;
        Vec3::new(self.drop_x, y, self.drop + self.ball_r + i as f64 * self.stagger)
    }

    pub fn material(&self) -> ContactMaterial {
        ContactMaterial {
            friction: self.friction,
            restitution: self.restitution,
            ..Default::default()
        }
    }
}

/// The hoop, in metres: the rim's centre and inside radius, the board's face.
#[derive(Clone, Copy, Debug)]
pub struct Hoop {
    pub rim_centre: Vec3,
    pub rim_r: f64,
    pub board_x: f64,
}

impl Hoop {
    fn from_scene(s: &AuthoredScene) -> anyhow::Result<Self> {
        let board_x = s.millimetres("board_x_mm")?;
        Ok(Self {
            rim_centre: Vec3::new(board_x - s.millimetres("rim_offset_mm")?, 0.0, s.millimetres("rim_z_mm")?),
            rim_r: s.millimetres("rim_r_mm")?,
            board_x,
        })
    }
}

/// A shot: where the ball leaves the hand, and how.
#[derive(Clone, Copy, Debug)]
pub struct Shot {
    pub release: Vec3,
    pub speed: f64,
    pub elevation: f64,
    pub azimuth: f64,
    /// Backspin, rad/s, about the horizontal axis to the left of the shot.
    pub backspin: f64,
}

impl Shot {
    fn from_scene(s: &AuthoredScene) -> anyhow::Result<Option<Self>> {
        let speed = s.parameter_or("shot_speed", 0.0);
        if speed <= 0.0 {
            return Ok(None);
        }
        Ok(Some(Self {
            release: Vec3::new(s.millimetres("shot_x_mm")?, s.millimetres("shot_y_mm")?, s.millimetres("shot_z_mm")?),
            speed,
            elevation: s.parameter("shot_elev_deg")?.to_radians(),
            azimuth: s.parameter_or("shot_azimuth_deg", 0.0).to_radians(),
            backspin: s.parameter_or("shot_backspin_rps", 0.0) * 2.0 * std::f64::consts::PI,
        }))
    }

    pub fn velocity(&self) -> Vec3 {
        let (ce, se) = (self.elevation.cos(), self.elevation.sin());
        Vec3::new(self.speed * ce * self.azimuth.cos(), self.speed * ce * self.azimuth.sin(), self.speed * se)
    }

    /// Angular velocity for the backspin: about the horizontal axis to the
    /// left of the shot, so the top of the ball moves backwards.
    pub fn angular_velocity(&self) -> Vec3 {
        let left = Vec3::new(-self.azimuth.sin(), self.azimuth.cos(), 0.0);
        left * -self.backspin
    }

    /// The speed that puts the ball's centre through `target` at this
    /// elevation, gravity only: `v² = g d² / (2 cos²θ (d tanθ − Δh))`.
    pub fn ballistic_speed(&self, target: Vec3) -> Option<f64> {
        let d = ((target.x - self.release.x).powi(2) + (target.y - self.release.y).powi(2)).sqrt();
        let dh = target.z - self.release.z;
        let (ce, te) = (self.elevation.cos(), self.elevation.tan());
        let denom = 2.0 * ce * ce * (d * te - dh);
        (denom > 0.0).then(|| (GRAVITY * d * d / denom).sqrt())
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
    /// When the shot's centre passed down through the rim, if it did.
    pub made_at: Option<f64>,
    pub hoop: Hoop,
    pub shot: Option<usize>,
    /// Solids the picture draws that the physics does not own — a net, a
    /// strand, a decal — placed in millimetres. Empty unless something fills it.
    pub extras: Vec<parts::PlacedSolid>,
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
        // only the roots that are meant to be stood on; see `parts::collides`
        let mut doc = scene.authored.document.clone();
        doc.roots.retain(|root| parts::collides(&root.material));
        let derived = colliders::colliders_from_document(&doc)?;
        anyhow::ensure!(!derived.colliders.is_empty(), "the court scene has no geometry to stand on");
        let worst = colliders::verify_against_mesh(&doc, &derived)?;
        anyhow::ensure!(worst < 0.5 * MM, "court colliders disagree with the CAD by {:.3} mm", worst / MM);

        let (r, m) = (scene.ball_r, scene.ball_mass);
        // a hollow sphere: I = 2/3 m r²
        let i = 2.0 / 3.0 * m * r * r;
        let ball_inertia = SpatialInertia::new(m, Vec3::zeros(), Mat3::from_diagonal(&Vec3::new(i, i, i)));
        let static_inertia = SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01);

        let n = scene.bodies();
        let mut builder = ModelBuilder::new().gravity(Vec3::new(0.0, 0.0, -GRAVITY)).dt(scene.dt);
        for k in 0..n {
            builder = builder.add_free_body(&format!("ball{k}"), -1, SpatialTransform::identity(), ball_inertia);
        }
        let mut model = builder
            .add_fixed_body("court", -1, SpatialTransform::identity(), static_inertia)
            .build();
        for k in 0..n {
            model.bodies[k].geometry = Some(Geometry::Sphere { radius: r });
        }
        model.bodies[n].collisions = derived.colliders.clone();
        model.bodies[n].visuals = derived.colliders;

        let q_pos: Vec<usize> = (0..n).map(|k| model.q_offsets[model.bodies[k].joint_idx] + 3).collect();
        let v_lin: Vec<usize> = (0..n).map(|k| model.v_offsets[model.bodies[k].joint_idx] + 3).collect();

        let mut state = model.default_state();
        for k in 0..scene.n_balls {
            let p = scene.release(k);
            state.q[q_pos[k]] = p.x;
            state.q[q_pos[k] + 1] = p.y;
            state.q[q_pos[k] + 2] = p.z;
        }
        let shot = scene.shot.map(|shot| {
            let k = scene.n_balls;
            let (p, v, w) = (shot.release, shot.velocity(), shot.angular_velocity());
            let (p, v, w) = (p.as_array(), v.as_array(), w.as_array());
            for i in 0..3 {
                state.q[q_pos[k] + i] = p[i];
                state.v[v_lin[k] + i] = v[i];
                state.v[v_lin[k] - 3 + i] = w[i];
            }
            k
        });

        let sim = Simulator::new().with_contact_config(ContactSolverConfig::simulation());

        Ok(Self {
            model,
            state,
            apexes: Vec::new(),
            made_at: None,
            hoop: scene.hoop,
            shot,
            extras: Vec::new(),
            material: scene.material(),
            sim,
            ball_r: r,
            n_balls: n,
            q_pos,
            v_lin,
            prev_vz: vec![0.0; n],
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

    /// Every ball, the shot included.
    pub fn bodies(&self) -> usize {
        self.n_balls
    }

    /// World→body rotation of a ball, from the free joint's exponential coordinates.
    pub fn rotation(&self, ball: usize) -> Mat3 {
        forward_kinematics(&self.model, &self.state).0[ball].rot
    }

    /// Where the shot's centre is relative to the rim's centre, if there is a shot.
    pub fn shot_offset(&self) -> Option<Vec3> {
        self.shot.map(|k| self.centre(k) - self.hoop.rim_centre)
    }

    /// One step. The ground plane is far below the slab: the level is the court.
    pub fn step(&mut self) {
        let before = self.shot_offset();
        self.sim.step_with_contacts(&self.model, &mut self.state, -10.0, &self.material);
        if let (Some(a), Some(b), None) = (before, self.shot_offset(), self.made_at) {
            // through the hoop: the centre crossed the rim plane downward,
            // inside the ring, on this step
            if a.z >= 0.0 && b.z < 0.0 && b.x.hypot(b.y) < self.hoop.rim_r {
                self.made_at = Some(self.state.time);
            }
        }
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
    if let Some(shot) = scene.shot {
        println!(
            "shot   from ({:+.2}, {:+.2}, {:.2}) at {:.2} m/s, {:.1}° up, {:.1} rps backspin; the rim centre wants {:.2} m/s at this elevation",
            shot.release.x,
            shot.release.y,
            shot.release.z,
            shot.speed,
            shot.elevation.to_degrees(),
            shot.backspin / (2.0 * std::f64::consts::PI),
            shot.ballistic_speed(scene.hoop.rim_centre).unwrap_or(f64::NAN)
        );
    }
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
    let a = &scene.authored;
    let (w, h) = if width > 0 { (width, height) } else { (a.parameter_or("render_w", 960.0) as u32, a.parameter_or("render_h", 540.0) as u32) };
    let spp = std::env::var("KOSM_SPP").ok().and_then(|v| v.parse().ok()).unwrap_or(a.parameter_or("render_spp", 8.0) as usize);
    let cam = render::camera(&scene)?;
    let exposure = a.parameter_or("exposure", 1.0);
    // the shutter: what fraction of a frame the film is exposed for, and how
    // many sub-frames that exposure is sampled at. 0 is an instant.
    let shutter = a.parameter_or("shutter", 0.0).clamp(0.0, 1.0);
    let subs = if shutter > 0.0 { a.parameter_or("shutter_steps", 4.0).max(1.0) as usize } else { 1 };
    let open = ((shutter * steps_per_frame as f64).round() as usize).min(steps_per_frame);
    let closed = steps_per_frame - open;
    // the sample budget is the frame's, split across the sub-frames
    let sub_spp = spp.div_ceil(subs);
    let still_t = a.parameter_or("still_t", -1.0);
    let mut still_done = still_t < 0.0;
    let mut picture = render::Scene::new(&scene)?;
    println!(
        "render {w}×{h} at {spp} spp per frame ({subs} × {sub_spp}, shutter {:.2} frame); {} static objects, {} panels; still {}×{} at {} spp at t = {still_t:.2} s",
        shutter,
        picture.static_count(),
        picture.light_count(),
        a.parameter_or("still_w", 1920.0) as u32,
        a.parameter_or("still_h", 1080.0) as u32,
        a.parameter_or("still_spp", 128.0) as usize
    );
    let t0 = std::time::Instant::now();
    let mut sim_time = std::time::Duration::ZERO;
    let mut render_time = std::time::Duration::ZERO;
    for k in 0..frames {
        // the shutter opens `open` steps before the end of the frame's span;
        // each sub-frame is a render at one point inside it, and the frame is
        // their average
        let mut stepped = 0usize;
        let mut film: Option<render::Film> = None;
        for j in 0..subs {
            let lap = std::time::Instant::now();
            let target = closed + (j + 1) * open / subs;
            while stepped < target {
                court.step();
                stepped += 1;
            }
            sim_time += lap.elapsed();
            let lap = std::time::Instant::now();
            let at = picture.at(&court);
            let opts = render::options(&scene, sub_spp, (k as u64) << 8 | j as u64);
            let f = render::render(&at, &cam, w, h, &opts);
            match &mut film {
                Some(acc) => render::accumulate(acc, &f),
                None => film = Some(f),
            }
            render_time += lap.elapsed();
        }
        let lap = std::time::Instant::now();
        let film = film.expect("at least one sub-frame");
        render::to_image(&film, exposure, subs).save(dir.join(format!("frame_{k:03}.png")))?;
        render_time += lap.elapsed();
        if !still_done && court.time() >= still_t {
            still_done = true;
            let (sw, sh) = (a.parameter_or("still_w", 1920.0) as u32, a.parameter_or("still_h", 1080.0) as u32);
            let sspp = a.parameter_or("still_spp", 128.0) as usize;
            let lap = std::time::Instant::now();
            let still = out.join("court_still.png");
            let at = picture.at(&court);
            let opts = render::options(&scene, sspp, 1 << 32);
            render::to_image(&render::render(&at, &cam, sw, sh, &opts), exposure, 1).save(&still)?;
            println!("render {} at t = {:.2} s, {sw}×{sh} × {sspp} spp in {:.1} s", still.display(), court.time(), lap.elapsed().as_secs_f64());
        }
        if k % 30 == 0 {
            println!("render frame {k} of {frames}: {:.1} s per frame", render_time.as_secs_f64() / (k + 1) as f64);
        }
    }
    println!(
        "court  {:.2} s simulated in {} ms; {} frames rendered in {:.1} s ({:.1} s total)",
        court.time(),
        sim_time.as_millis(),
        frames,
        render_time.as_secs_f64(),
        t0.elapsed().as_secs_f64()
    );
    report(&scene, &court);
    if let Some(k) = court.shot {
        let c = court.centre(k);
        match court.made_at {
            Some(t) => println!("shot   through the hoop at {t:.2} s; the ball ends at ({:+.2}, {:+.2}, {:.2})", c.x, c.y, c.z),
            None => println!("shot   missed; the ball ends at ({:+.2}, {:+.2}, {:.2})", c.x, c.y, c.z),
        }
    }

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
