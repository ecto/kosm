//! The K1 shoved sideways, held by PD (left) and by gradient MPC (right).
//!
//! The physics is not here: phyz-diff's `mpc_record` example runs both
//! controllers through the traced K1 contact step (MPC plans 16 steps with
//! exact gradients from the step's tang-expr adjoint, replanning every 10)
//! and writes each link's world pose every 10 ms. This sim is the picture:
//! every link is a box fitted to its STL, both robots stand on one floor,
//! and the camera lens path-traces the pair frame by frame; ffmpeg writes
//! `k1_push.mp4` (unlabelled: titles are the edit's job).
//!
//! `kosm run k1_push --rec k1_push.txt [--spp 16] [--width 1280] [--frames N]`

use kosm::prelude::*;
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{GeomInstance, Geometry, ModelBuilder};

struct Link {
    body: usize,
    centre: Vec3,
    half: Vec3,
}

struct Recording {
    z0: f64,
    shove: f64,
    solve_ms: f64,
    links: Vec<Link>,
    /// `[controller][frame][body]`: position and world→body rotation.
    poses: [Vec<Vec<(Vec3, Mat3)>>; 2],
}

fn read(path: &str) -> anyhow::Result<Recording> {
    let text = std::fs::read_to_string(path)?;
    let mut rec = Recording { z0: 0.0, shove: 0.0, solve_ms: 0.0, links: Vec::new(), poses: [Vec::new(), Vec::new()] };
    for line in text.lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        let f = |i: usize| -> f64 { w[i].parse().unwrap() };
        match w.first() {
            Some(&"meta") => (rec.z0, rec.shove, rec.solve_ms) = (f(1), f(2), f(3)),
            Some(&"link") => {
                let (lo, hi) = (Vec3::new(f(2), f(3), f(4)), Vec3::new(f(5), f(6), f(7)));
                rec.links.push(Link { body: w[1].parse()?, centre: (lo + hi) * 0.5, half: (hi - lo) * 0.5 });
            }
            Some(&"pose") => {
                let c = usize::from(w[1] == "mpc");
                let (frame, body): (usize, usize) = (w[2].parse()?, w[3].parse()?);
                let frames = &mut rec.poses[c];
                if frames.len() <= frame {
                    frames.resize(frame + 1, Vec::new());
                }
                if frames[frame].len() <= body {
                    frames[frame].resize(body + 1, (Vec3::zeros(), Mat3::identity()));
                }
                let r = Mat3::new(f(7), f(8), f(9), f(10), f(11), f(12), f(13), f(14), f(15));
                frames[frame][body] = (Vec3::new(f(4), f(5), f(6)), r);
            }
            _ => {}
        }
    }
    Ok(rec)
}

/// One frame as a world: a fixed stage body carrying the floor and both
/// robots' link boxes at their recorded poses, PD shifted to -y, MPC to +y.
fn stage(rec: &Recording, frame: usize, gap: f64) -> World {
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(1e-3)
        .add_fixed_body("stage", -1, SpatialTransform::identity(), SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01))
        .build();
    let mut shapes = vec![GeomInstance {
        name: Some("floor".into()),
        origin: SpatialTransform::new(Mat3::identity(), Vec3::new(0.0, 0.0, -0.01)),
        geometry: Geometry::Box { half_extents: Vec3::new(1.6, 1.6, 0.01) },
    }];
    for (c, frames) in rec.poses.iter().enumerate() {
        let poses = &frames[frame.min(frames.len() - 1)];
        let shift = Vec3::new(0.0, if c == 0 { -gap } else { gap }, 0.0);
        for l in &rec.links {
            let Some(&(p, r)) = poses.get(l.body) else { continue };
            // world→shape is the body's rotation (the box is axis-aligned in
            // the link frame); its centre is p + rᵀ·centre
            shapes.push(GeomInstance {
                name: None,
                origin: SpatialTransform::new(r, p + r.transpose() * l.centre + shift),
                geometry: Geometry::Box { half_extents: l.half },
            });
        }
    }
    model.bodies[0].collisions = shapes;
    let state = model.default_state();
    World::from_phyz(model, state)
}

pub fn run(args: &kosm_cli::Args) -> anyhow::Result<()> {
    let path = args.value("rec").ok_or_else(|| anyhow::anyhow!("--rec RECORDING (from phyz-diff's mpc_record)"))?;
    let rec = read(path)?;
    let spp: u32 = args.value("spp").and_then(|v| v.parse().ok()).unwrap_or(16);
    let width: u32 = args.value("width").and_then(|v| v.parse().ok()).unwrap_or(1280);
    let height = width * 9 / 16;
    let n = rec.poses[0].len().max(rec.poses[1].len());
    let frames: usize = args.value("frames").and_then(|v| v.parse().ok()).unwrap_or(n).min(n);
    let gap = 0.62;
    let params = vec![Param::new("shove", rec.shove), Param::new("spp", spp as f64), Param::new("width", width as f64), Param::new("frames", frames as f64)];
    let mut out = Recorder::new(args.out(), "k1_push", &params, 0)?;
    let camera = Camera::look_at(Vec3::new(2.0, -1.15, 0.85), Vec3::new(0.0, 0.05, 0.36), width, height, spp);
    let dir = out.dir().join("frames");
    std::fs::create_dir_all(&dir)?;
    let start = std::time::Instant::now();
    let lowest = |c: usize, upto: usize| {
        rec.poses[c][..upto.min(rec.poses[c].len())].iter().map(|b| b[1].0.z).fold(f64::INFINITY, f64::min)
    };
    for f in 0..frames {
        let img = camera.see(&stage(&rec, f, gap)).image;
        img.save(dir.join(format!("frame_{f:04}.png")))?;
        if f % 25 == 0 {
            eprintln!("k1_push frame {f}/{frames} ({:.1} s each)", start.elapsed().as_secs_f64() / (f + 1) as f64);
        }
    }
    let (pd_drop, mpc_drop) = (rec.z0 - lowest(0, frames), rec.z0 - lowest(1, frames));
    out.metric("trunk_drop_pd_mm", pd_drop * 1e3)?;
    out.metric("trunk_drop_mpc_mm", mpc_drop * 1e3)?;
    out.metric("mpc_solve_ms", rec.solve_ms)?;
    out.metric("frames", frames)?;
    out.metric("render_s_per_frame", start.elapsed().as_secs_f64() / frames as f64)?;

    // 10 ms of simulated time per frame, played at 25 fps: 2.5x slow motion
    let mp4 = out.dir().join("k1_push.mp4");
    let st = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-framerate", "25", "-i"])
        .arg(dir.join("frame_%04d.png"))
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17"])
        .arg(&mp4)
        .status();
    match st {
        Ok(s) if s.success() => println!("k1_push {frames} frames → {}", mp4.display()),
        _ => println!("k1_push frames in {} (ffmpeg failed; no mp4)", dir.display()),
    }
    out.finish()?;
    Ok(())
}
