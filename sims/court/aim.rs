//! The shot, hinted: a gradient on the free throw and a two-knob solve.
//!
//! The marble's hint is an adjoint of "distance to the cup" at a fixed step
//! count with respect to the release point. This is the same thing pointed at
//! the hoop. The horizon `aim_t` is the moment the ball's centre falls back
//! through the rim plane — ballistic, from `Shot::velocity()` and gravity —
//! rounded to a step, and
//!
//! ```text
//! J = |centre(T) − rim centre|²
//! ```
//!
//! is the miss at that step, squared. The xy part is the one that decides a
//! make; the z part is what stops the problem being a line of equivalent
//! (speed, elevation) pairs, because at a fixed horizon the ball has to be at
//! the rim's *height* as well as over its middle.
//!
//! The adjoint problem is the shot alone on the court
//! (`Court::from_scene_shot_only`): one free body against the same derived
//! colliders, so the backward pass is not carrying three dropped balls it does
//! not care about.
//!
//! `phyz_diff::ConvexAdjointGradients` gives `d_q0` (the release point) *and*
//! `d_v0` (the release velocity) from one backward pass, so both are checked
//! against central differences here. The solve turns `d_v0` into the two knobs
//! the level actually spells — `shot_speed` and `shot_elev_deg` — through
//!
//! ```text
//! ∂v₀/∂speed = (cosθ cosψ, cosθ sinψ, sinθ)
//! ∂v₀/∂θ     = speed · (−sinθ cosψ, −sinθ sinψ, cosθ)
//! ```
//!
//! and descends with backtracking, as the marble's tilt solve does. Elevation
//! is stepped as `θ · speed` rather than `θ`, because those two columns are
//! orthogonal and of equal length in that scaling: the descent direction then
//! points more or less straight at the answer instead of down a valley.

use phyz_contact::{ContactMaterial, ContactSolverConfig, find_contacts};
use phyz_diff::{
    ConvexContactRollout, FinalStateObjective, convex_adjoint_gradient, convex_rollout_objective,
};
use phyz_math::{DVec, GRAVITY, Vec3};
use phyz_model::Model;
use phyz_rigid::forward_kinematics;

use super::{Court, CourtScene, Shot};

/// Central-difference step for the printed gradient check.
const FD_H: f64 = 1e-5;

/// Is the aim wanted at all? `aim 0` turns it off for the render-only path.
pub fn enabled(scene: &CourtScene) -> bool {
    scene.authored.parameter_or("aim", 1.0) > 0.0
}

/// The ballistic time for the shot's centre to fall back through the rim
/// plane: gravity only, the later root of `z₀ + v_z t − ½ g t² = rim_z`.
pub fn ballistic_horizon(scene: &CourtScene) -> Option<f64> {
    let shot = scene.shot?;
    let vz = shot.velocity().z;
    let dz = scene.hoop.rim_centre.z - shot.release.z;
    let disc = vz * vz - 2.0 * GRAVITY * dz;
    (disc >= 0.0 && GRAVITY > 0.0).then(|| (vz + disc.sqrt()) / GRAVITY)
}

/// The horizon the level asks for: `aim_t` if it is positive, else ballistic.
pub fn horizon(scene: &CourtScene) -> anyhow::Result<f64> {
    let knob = scene.authored.parameter_or("aim_t", -1.0);
    if knob > 0.0 {
        return Ok(knob);
    }
    ballistic_horizon(scene)
        .filter(|t| *t > 0.0)
        .ok_or_else(|| anyhow::anyhow!("this shot never reaches the rim plane; set `aim_t`"))
}

/// The horizon, rounded to a whole step.
pub fn steps(scene: &CourtScene) -> anyhow::Result<usize> {
    Ok((horizon(scene)? / scene.dt).round().max(1.0) as usize)
}

/// How many steps the shot flies before it comes within the solver's margin
/// of anything — the horizon on which the rollout is smooth and the adjoint
/// has no active set to switch.
pub fn free_flight_steps(scene: &CourtScene, max: usize) -> anyhow::Result<usize> {
    let mut court = Court::from_scene_shot_only(scene)?;
    let k = court.shot.ok_or_else(|| anyhow::anyhow!("no shot"))?;
    let margin = scene.material().margin;
    for s in 0..max {
        // `find_contacts` reads `state.body_xform`, which a step writes; at
        // step zero it is still the default pose, so run the kinematics first
        court.state.body_xform = forward_kinematics(&court.model, &court.state).0;
        // the court body's own colliders touch each other (the rim's segments,
        // the bracket on the board); only the ball's pairs are contact here
        let touching = find_contacts(&court.model, &court.state, margin)
            .iter()
            .any(|c| c.body_i != c.body_j && (c.body_i == k || c.body_j == k));
        if touching {
            return Ok(s);
        }
        court.step();
    }
    Ok(max)
}

/// J = |centre − target|² at the last step, and its gradient in `q`.
fn objective(target: Vec3, q_pos: usize) -> FinalStateObjective<'static> {
    let g: &'static Vec3 = Box::leak(Box::new(target));
    let value: &'static dyn Fn(&[f64], &[f64]) -> f64 =
        Box::leak(Box::new(move |q: &[f64], _: &[f64]| {
            let d = Vec3::new(q[q_pos] - g.x, q[q_pos + 1] - g.y, q[q_pos + 2] - g.z);
            d.dot(&d)
        }));
    type GradFn = dyn Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>);
    let gradient: &'static GradFn = Box::leak(Box::new(move |q: &[f64], v: &[f64]| {
        let mut gq = vec![0.0; q.len()];
        gq[q_pos] = 2.0 * (q[q_pos] - g.x);
        gq[q_pos + 1] = 2.0 * (q[q_pos + 1] - g.y);
        gq[q_pos + 2] = 2.0 * (q[q_pos + 2] - g.z);
        (gq, vec![0.0; v.len()])
    }));
    FinalStateObjective { value, gradient }
}

fn rollout<'a>(
    model: &'a Model,
    material: &ContactMaterial,
    q0: DVec,
    v0: DVec,
    steps: usize,
    ctrl: &'a dyn Fn(usize) -> DVec,
) -> ConvexContactRollout<'a> {
    ConvexContactRollout {
        model,
        ground_height: -10.0, // the level is the court; the ground is out of play
        material: material.clone(),
        config: ContactSolverConfig::gradients(),
        q0,
        v0,
        steps,
        ctrl,
    }
}

/// The shot's initial velocity vector, for a candidate speed and elevation.
fn v_for(model: &Model, base: Shot, v_lin: usize, speed: f64, elevation: f64) -> DVec {
    let shot = Shot { speed, elevation, ..base };
    let mut v = DVec::zeros(model.nv);
    let (lin, ang) = (shot.velocity().as_array(), shot.angular_velocity().as_array());
    for i in 0..3 {
        v[v_lin + i] = lin[i];
        v[v_lin - 3 + i] = ang[i];
    }
    v
}

/// One backward pass against central differences, for both channels.
pub struct Check {
    pub steps: usize,
    /// `dJ/d(release point)`, adjoint and central differences.
    pub d_q0: [f64; 3],
    pub fd_q0: [f64; 3],
    /// `dJ/d(release velocity)`, adjoint and central differences.
    pub d_v0: [f64; 3],
    pub fd_v0: [f64; 3],
}

impl Check {
    /// Worst relative disagreement over the six numbers, and which one.
    pub fn worst(&self) -> (&'static str, f64) {
        const LANES: [&str; 6] =
            ["release x", "release y", "release z", "release vx", "release vy", "release vz"];
        let pairs = self.d_q0.iter().zip(&self.fd_q0).chain(self.d_v0.iter().zip(&self.fd_v0));
        pairs
            .zip(LANES)
            .map(|((a, f), lane)| (lane, (a - f).abs() / a.abs().max(f.abs()).max(1e-12)))
            .fold(("none", 0.0), |best, x| if x.1 > best.1 { x } else { best })
    }

    /// Worst relative disagreement over the six numbers.
    pub fn worst_relative(&self) -> f64 {
        self.worst().1
    }

    pub fn lines(&self) -> Vec<String> {
        let row = |name: &str, g: &[f64; 3]| {
            format!("{name} = [{:+.4e} {:+.4e} {:+.4e}]", g[0], g[1], g[2])
        };
        vec![
            row("adjoint dJ/d(release)   ", &self.d_q0),
            row(" fd     dJ/d(release)   ", &self.fd_q0),
            row("adjoint dJ/d(release v) ", &self.d_v0),
            row(" fd     dJ/d(release v) ", &self.fd_v0),
        ]
    }
}

/// The adjoint at `steps`, and the same six numbers by central differences.
///
/// Returns the adjoint's own refusal verbatim when it will not produce a
/// gradient — that is the contract of `phyz_diff`, and it is worth printing.
pub fn check(scene: &CourtScene, steps: usize) -> anyhow::Result<Check> {
    let court = Court::from_scene_shot_only(scene)?;
    let k = court.shot.ok_or_else(|| anyhow::anyhow!("no shot"))?;
    let (q_pos, v_lin) = (court.q_pos(k), court.v_lin(k));
    let (q0, v0) = (court.state.q.clone(), court.state.v.clone());
    let obj = objective(scene.hoop.rim_centre, q_pos);
    let material = scene.material();
    let ctrl = |_: usize| DVec::zeros(court.model.nv);

    let gr = convex_adjoint_gradient(
        &rollout(&court.model, &material, q0.clone(), v0.clone(), steps, &ctrl),
        &obj,
    )
    .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut fd_q0 = [0.0; 3];
    let mut fd_v0 = [0.0; 3];
    for (slot, i) in fd_q0.iter_mut().zip(0..3) {
        let (mut a, mut b) = (q0.clone(), q0.clone());
        a[q_pos + i] += FD_H;
        b[q_pos + i] -= FD_H;
        let fa = convex_rollout_objective(&rollout(&court.model, &material, a, v0.clone(), steps, &ctrl), &obj);
        let fb = convex_rollout_objective(&rollout(&court.model, &material, b, v0.clone(), steps, &ctrl), &obj);
        *slot = (fa - fb) / (2.0 * FD_H);
    }
    for (slot, i) in fd_v0.iter_mut().zip(0..3) {
        let (mut a, mut b) = (v0.clone(), v0.clone());
        a[v_lin + i] += FD_H;
        b[v_lin + i] -= FD_H;
        let fa = convex_rollout_objective(&rollout(&court.model, &material, q0.clone(), a, steps, &ctrl), &obj);
        let fb = convex_rollout_objective(&rollout(&court.model, &material, q0.clone(), b, steps, &ctrl), &obj);
        *slot = (fa - fb) / (2.0 * FD_H);
    }

    Ok(Check {
        steps,
        d_q0: [gr.d_q0[q_pos], gr.d_q0[q_pos + 1], gr.d_q0[q_pos + 2]],
        fd_q0,
        d_v0: [gr.d_v0[v_lin], gr.d_v0[v_lin + 1], gr.d_v0[v_lin + 2]],
        fd_v0,
    })
}

/// What the solve landed on.
#[derive(Clone, Debug)]
pub struct Solved {
    pub speed: f64,
    pub elevation: f64,
    /// `|centre(T) − rim centre|` at the horizon, metres.
    pub miss: f64,
    pub iterations: usize,
    /// When the production simulator called it, if it did.
    pub made_at: Option<f64>,
    /// Did the adjoint carry the gradient, or did it refuse and leave it to
    /// central differences?
    pub adjoint: bool,
    /// The adjoint's refusal, if it refused.
    pub refusal: Option<String>,
}

/// Does the production simulator put this shot through the hoop?
///
/// `Court::from_scene` — the whole level, dropped balls and all, the
/// simulation contact config — stepped until the call or the horizon.
fn made(scene: &mut CourtScene, speed: f64, elevation: f64) -> anyhow::Result<Option<f64>> {
    let base = scene.shot.ok_or_else(|| anyhow::anyhow!("no shot"))?;
    scene.shot = Some(Shot { speed, elevation, ..base });
    let out = (|| -> anyhow::Result<Option<f64>> {
        let mut court = Court::from_scene(scene)?;
        let k = court.shot.ok_or_else(|| anyhow::anyhow!("no shot"))?;
        let floor = scene.hoop.rim_centre.z - 1.0;
        while court.time() < scene.t_end && court.made_at.is_none() {
            court.step();
            // a metre below the rim and still falling: whatever happens next,
            // it is not this shot going in
            if court.centre(k).z < floor && court.velocity(k).z < 0.0 {
                break;
            }
        }
        Ok(court.made_at)
    })();
    scene.shot = Some(base);
    out
}

/// Descend on (speed, elevation) from the level's values until the shot goes
/// in, or the miss at the horizon is under a centimetre.
pub fn solve(scene: &mut CourtScene, steps: usize, log: &mut Vec<String>) -> anyhow::Result<Solved> {
    let base = scene.shot.ok_or_else(|| anyhow::anyhow!("this court has no shot"))?;
    let court = Court::from_scene_shot_only(scene)?;
    let k = court.shot.ok_or_else(|| anyhow::anyhow!("no shot"))?;
    let (q_pos, v_lin) = (court.q_pos(k), court.v_lin(k));
    let q0 = court.state.q.clone();
    let model = &court.model;
    let material = scene.material();
    let obj = objective(scene.hoop.rim_centre, q_pos);
    let ctrl = |_: usize| DVec::zeros(model.nv);
    let horizon = steps as f64 * scene.dt;

    let j = |speed: f64, elevation: f64| {
        let v0 = v_for(model, base, v_lin, speed, elevation);
        convex_rollout_objective(&rollout(model, &material, q0.clone(), v0, steps, &ctrl), &obj)
    };

    // The two knobs move the ball along ∂v₀/∂speed and ∂v₀/∂θ, which are
    // orthogonal; scaling θ by the release speed makes them the same length,
    // so a normalised descent direction points at the answer.
    let scale = base.speed.max(1.0);
    let mut speed = base.speed;
    let mut elevation = base.elevation;
    let mut value = j(speed, elevation);
    let mut refusal = None;
    let mut used_adjoint = false;
    let mut made_at = None;
    let mut iterations = 0;

    for it in 0..24 {
        iterations = it;
        let miss = value.sqrt();
        log.push(format!(
            "it {it:2}  speed {speed:.3} m/s  elev {:.2}°  miss {miss:.4} m",
            elevation.to_degrees()
        ));
        // The production check is the whole level — four bodies, the colliders
        // re-derived — so it is only asked once the miss is inside a ball's
        // radius, where a call is plausible at all, and only after the descent
        // has taken a step: the level's own numbers may already fall through
        // the hoop, and stopping on that before doing anything says nothing.
        if (miss < 0.01 || (it > 0 && miss < scene.ball_r))
            && let Some(t) = made(scene, speed, elevation)?
        {
            made_at = Some(t);
            break;
        }
        if miss < 0.01 {
            break;
        }

        // dJ/d(speed, θ): the adjoint's dJ/dv₀ through the chain rule when it
        // will give one, central differences on the two knobs when it will not.
        let (ce, se) = (elevation.cos(), elevation.sin());
        let (ca, sa) = (base.azimuth.cos(), base.azimuth.sin());
        let d_speed = Vec3::new(ce * ca, ce * sa, se);
        let d_elev = Vec3::new(-se * ca, -se * sa, ce) * speed;
        let v0 = v_for(model, base, v_lin, speed, elevation);
        let grad = match convex_adjoint_gradient(
            &rollout(model, &material, q0.clone(), v0, steps, &ctrl),
            &obj,
        ) {
            Ok(gr) => {
                used_adjoint = true;
                let g = Vec3::new(gr.d_v0[v_lin], gr.d_v0[v_lin + 1], gr.d_v0[v_lin + 2]);
                [g.dot(&d_speed), g.dot(&d_elev)]
            }
            Err(e) => {
                if refusal.is_none() {
                    let text = format!("{e:?}");
                    log.push(format!("it {it:2}  adjoint refused: {text}; central differences"));
                    refusal = Some(text);
                }
                let h = 1e-4;
                [
                    (j(speed + h, elevation) - j(speed - h, elevation)) / (2.0 * h),
                    (j(speed, elevation + h) - j(speed, elevation - h)) / (2.0 * h),
                ]
            }
        };

        let gs = grad[0];
        let gb = grad[1] / scale;
        let norm = (gs * gs + gb * gb).sqrt();
        if !norm.is_finite() || norm < 1e-14 {
            log.push(format!("it {it:2}  the gradient vanished; stopping"));
            break;
        }
        let dir = [-gs / norm, -gb / norm];

        // `miss / horizon` is the step that closes the miss exactly if the map
        // from knobs to the ball's place at T were affine — which it is right
        // up until the ball starts catching the rim instead of clearing it.
        // Around that cliff a halving search stalls on the wrong side, so the
        // whole ladder is tried and the best improvement taken.
        let base_step = (miss / horizon).clamp(0.005, 3.0);
        let mut best: Option<(f64, f64, f64)> = None;
        for factor in [2.0, 1.4, 1.0, 0.7, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015] {
            let step = base_step * factor;
            let cs = (speed + step * dir[0]).clamp(1.0, 25.0);
            let cb = (elevation + step * dir[1] / scale)
                .clamp(5.0f64.to_radians(), 80.0f64.to_radians());
            let cj = j(cs, cb);
            if cj < value && best.is_none_or(|(bj, _, _)| cj < bj) {
                best = Some((cj, cs, cb));
            }
        }
        let Some((cj, cs, cb)) = best else {
            log.push(format!("it {it:2}  no descent along the gradient; stopping"));
            break;
        };
        value = cj;
        speed = cs;
        elevation = cb;
    }

    if made_at.is_none() {
        made_at = made(scene, speed, elevation)?;
    }

    Ok(Solved {
        speed,
        elevation,
        miss: value.sqrt(),
        iterations,
        made_at,
        adjoint: used_adjoint,
        refusal,
    })
}

/// The whole hint, as lines for the caller to print.
///
/// Reloads the level so it has a scene of its own to move the shot around in.
pub fn hint(_scene: &CourtScene) -> anyhow::Result<Vec<String>> {
    let mut out = Vec::new();
    let mut work = CourtScene::bundled()?;
    let Some(base) = work.shot else {
        out.push("this court has no shot; nothing to aim".into());
        return Ok(out);
    };
    let n = steps(&work)?;
    let horizon = n as f64 * work.dt;
    let free = free_flight_steps(&work, n)?;
    out.push(format!(
        "horizon {horizon:.3} s ({n} steps at {:.1} ms): the centre falls back through z = {:.3} m; free flight for {free} of them",
        work.dt * 1e3,
        work.hoop.rim_centre.z
    ));

    // 1. the adjoint against central differences, in free flight, where the
    //    rollout is smooth and both channels should agree to the last digits.
    match check(&work, free) {
        Ok(c) => {
            let (lane, gap) = c.worst();
            out.push(format!("free flight, {} steps: worst relative gap {gap:.1e} (d/d {lane})", c.steps));
            out.extend(c.lines());
        }
        Err(e) => out.push(format!("free flight, {free} steps: adjoint refused: {e}")),
    }
    // 2. and at the aim horizon, which for the level's default shot is past
    //    the ball's first touch on the rim and glass.
    match check(&work, n) {
        Ok(c) => {
            let (lane, gap) = c.worst();
            out.push(format!(
                "at the horizon, {} steps: worst relative gap {gap:.1e} (d/d {lane}) — past the ball's first touch on the rim, where a micron of sideways nudge changes which rod segment it catches and the difference quotient stops being a derivative",
                c.steps
            ));
            out.extend(c.lines());
        }
        Err(e) => out.push(format!("at the horizon, {n} steps: adjoint refused: {e}")),
    }

    // 3. the solve, from the level's own knobs
    let t0 = std::time::Instant::now();
    let mut log = Vec::new();
    let solved = solve(&mut work, n, &mut log)?;
    out.extend(log);
    out.push(report_line("level", base, &solved, t0.elapsed().as_secs_f64()));

    let path = std::path::Path::new("out/hinted");
    std::fs::create_dir_all(path)?;
    let file = path.join("court.json");
    let aimed = work.authored.with(&[
        ("shot_speed", solved.speed),
        ("shot_elev_deg", solved.elevation.to_degrees()),
    ])?;
    std::fs::write(&file, aimed.document.to_json()?)?;
    out.push(format!("wrote {} ({:.4} m/s, {:.3}°)", file.display(), solved.speed, solved.elevation.to_degrees()));

    // 4. an off target: a short shot, brought back
    let short = 6.8;
    work.shot = Some(Shot { speed: short, ..base });
    let n_short = steps(&work)?;
    let before = made(&mut work, short, base.elevation)?;
    out.push(format!(
        "off target: shot_speed {short:.2} m/s at {:.1}° is {}; horizon {:.3} s",
        base.elevation.to_degrees(),
        match before {
            Some(t) => format!("in at {t:.2} s"),
            None => "a miss".into(),
        },
        n_short as f64 * work.dt
    ));
    let t0 = std::time::Instant::now();
    let mut log = Vec::new();
    let back = solve(&mut work, n_short, &mut log)?;
    out.extend(log);
    out.push(report_line("off target", Shot { speed: short, ..base }, &back, t0.elapsed().as_secs_f64()));

    Ok(out)
}

fn report_line(what: &str, from: Shot, s: &Solved, secs: f64) -> String {
    format!(
        "{what}: {:.2} m/s at {:.1}° → {:.3} m/s at {:.2}° in {} iterations ({}), miss {:.1} mm, {}; {secs:.1} s",
        from.speed,
        from.elevation.to_degrees(),
        s.speed,
        s.elevation.to_degrees(),
        s.iterations + 1,
        if s.adjoint { "adjoint" } else { "central differences" },
        s.miss * 1e3,
        match s.made_at {
            Some(t) => format!("in at {t:.2} s"),
            None => "still missing".into(),
        }
    )
}
