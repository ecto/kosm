# Architecture

*Written 2026-09-08. This is the shape the crates are moving toward, and the
reasons. synthesis.md says what the engine is for; research.md is the physics
literature. This document is about code layout and API, and it leans on a
different literature: the papers on how simulators and game engines are
actually built. Its first consumer is ipse, and the last section is about
that.*

## Three nouns

The public vocabulary is three words. Nothing else is a first-class thing.

- **World.** Columns: bodies, colliders, materials, lights, fluid, params.
  Built by composing sims (below). No methods that run anything.
- **Step.** `World → World`, pure. Batched by construction: a batch is a
  world with a leading axis. A `Trajectory` is a world indexed by time.
- **Lens.** `World → observation`. A camera is a lens. A microphone is a
  lens. A reward is a lens. A depth sensor, a gate score, a probe, a
  contact-force readout: lenses. The renderer is not a subsystem the user
  addresses; it is the crate that implements the camera lens.

Every arrow is differentiable, and its adjoint runs the other way. The
adjoint of a lens is the question "which columns did this observation come
from": which pixel came from which material, which joint limit is holding the
ollie back, which parameter caused the failure. synthesis.md rule 4 (the
question chooses the level) is that adjoint. It is the first thing a user
learns, not the last.

Because worlds are columns, `diff(a, b)` exists, and it is three things at
once: a snapshot test when `b` is the stored frame, a sim2real gap when `a` is
fitted and `b` is the CAD, and a curriculum step when `b` is the next rung.

A run is `hash(sim path, params, seed, kosm rev)`. Its lens outputs live at
that hash, immutably. Two runs with the same hash are the same run, so the
gate, the parity check, and the snapshot test are all "does this hash's lens
output equal that one's".

## The rules

Four rules, each with a paper behind it and a consequence for the tree.

1. **State is data; systems are functions over it.** Brax (2106.13281) and
   Isaac Gym (2108.10470) run `step(state, params, action) -> state` over flat
   arrays with no objects carrying methods, and that is why they vectorise
   across thousands of environments and differentiate for free. Archetype
   ECS (2606.14919) is the game side arriving at the same place: components
   are columns, entities rows, systems queries, and the layout *is* the
   semantics. Hence World is columnar and Step is pure.

2. **A small core and leaf subsystems.** Subsystem-dependency recovery on
   real engines (SyDRA 2406.05487, and 2303.02429, 2309.06329 before it)
   finds one core that everything depends on and render, physics, audio,
   scripting as leaves that do not depend on each other. Leaf-to-leaf
   coupling is the failure mode they measure. Hence every lens and every Step
   implementation depends on the core and on nothing else in this repo.

3. **The gradient path is designed in.** DiffTaichi (1910.00935) is
   megakernels plus a tape over the time loop; Dojo (2203.00806) and Nimble
   (2103.16021) differentiate the contact solve implicitly rather than
   unrolling it; Dr.Jit (2202.01284) traces the render and takes its adjoint.
   tang is our tape and scalar, phyz's adjoint is our implicit contact
   gradient, kosm-render's Film is the traced render. The core exposes them.

4. **Gradients are one estimator, not the estimator.** Through long
   contact-rich horizons the Jacobian spectrum blows up and first-order
   gradients go chaotic (2111.05803); zeroth-order estimates then win
   (2202.00817). The API offers both: `grad` of a rollout, and batched
   rollouts for finite differences, CEM, MAP-Elites, sampling-based control.
   Rule 1 is what makes the batched path cheap.

## The tree

```
crates/
  kosm/            core: World, Step, Lens, Trajectory, build, Param,
                   diff, run hash, Task, gate, ledger, colliders,
                   materials, audio, brep→bvh, denoise, urdf import
  kosm-train/      loops that produce policies: PPO, BC, CEM, MAP-Elites
                   (from ipse-dojo and ipse-sim)
  kosm-scan/       leaf: room scans, hulls, object bake, SDF grids
                   (from ipse-map). depends on phyz only; core reads it.
  kosm-registry/   build-dependency: walks a sims/ tree into a registry
  kosm-render/     lens: camera, depth. cpu and gpu tiers. depends on tang only.
  kosm-mpm/        step: material point fluids on wgpu. depends on wgpu only.
  kosm-view/       plays any Trajectory. generic over World.
sims/              this repo's own sims (marble, court, pool, skatepark)
docs/
```

Dependency edges, and only these:

```
kosm-view, kosm-train ──▶ kosm ──▶ kosm-render
                            │  ──▶ kosm-mpm
                            │  ──▶ kosm-scan
                            │  ──▶ phyz, vcad, tang   (git revs)
sims/*                ──▶ kosm (+ kosm-train, kosm-view as needed)
```

`kosm-render`, `kosm-mpm`, and `kosm-scan` never see `kosm`; render and mpm
never see phyz or vcad either. `kosm-view`
never sees a sim by name. vcad's dependency on `kosm-render` is a temporary
back-edge (the `[patch]` in Cargo.toml); it goes away when kosm-render is
tagged and vcad tracks the tag.

## Sims

The filesystem is the registry, and nesting is composition. This is the
Next.js app router applied to simulation: a directory under `sims/` is a sim,
a few conventional filenames carry the roles, and each level of nesting
composes the world of its parent the way a nested layout wraps a page.

```
sims/
  k1/
    scene.rs             body from urdf, actuators, sensors
    policy.rs            observe(), act_dofs, the PD plant
    skate/
      scene.rs           + board, terrain heightfield
      reward.rs          the reward lens
      ollie/
        task.rs          spawn distribution, score, held_out, invariants
        gate.toml        32 frozen draws, one number
        policy.rs        keyframe schedule
      balance/
        task.rs
        gate.toml
  marble/
    scene.rs
    game.rs
```

- **Convention files.** `scene.rs`, `task.rs`, `policy.rs`, `reward.rs`,
  `game.rs`, `gate.toml`. Only the files present matter; `balance/` needs a
  task and a gate and inherits everything else.
- **Composition is function composition.** `ollie = k1 ∘ skate ∘ ollie`, each
  `scene.rs` a function of its parent's World. Gradients cross the
  composition: the ollie score can be differentiated with respect to a
  wheelbase declared two directories up. Board design against trick score.
- **A build script walks the tree** and generates the registry.
  `kosm run k1/skate/ollie --out` resolves by path. No `mod` lists, no
  `[[example]]` tables.
- **Geometry is Rust** over vcad-kernel through `kosm::build`, and knobs are
  `Param` values with a name and a gradient. No loon: one language, one API,
  nothing for an agent to learn beside Rust. The README's "the level is a CAD
  file" becomes "the level is a Rust function that emits CAD".
- **Every run writes `out/<hash>/`** with every lens output: png, mp4, wav,
  `metrics.json`. CI does it per branch and posts the frame. That is the
  preview deployment.
- **Tier is a declaration**, `tier = "gpu"` in the sim's manifest, not a
  second file.

## Agent DX

Most kosm users will be agents, or people working through one. What an agent
needs is a loop it can close by itself:

- **Headless, files out.** It cannot see a window. It runs the sim, reads
  the png, reads `metrics.json`, edits, runs again. `--view` opens
  kosm-view; it is never the default.
- **Deterministic.** Same hash, same bytes. A changed frame means changed
  code.
- **One import, one doc.** `kosm::prelude`; a repo `CLAUDE.md` with five
  recipes (build a world, add a body, batch a step, take a gradient, add a
  lens); doctests on everything the prelude exports.
- **Errors that say what to change.** `KosmError` carries a `help` line:
  "collider has no convex decomposition: wrap it in `hull()` or split the
  difference".
- **Snapshot tests are diffs.** `assert_lens!("k1/skate/ollie", camera,
  tol)` diffs against the stored run. The gate is the same assertion on the
  score lens.

## The core API

The shape, not the signatures; the signatures are decided in the move.

```rust
// geometry in rust, knobs as params. a columnar World out.
let tilt = Param::new("tilt", 0.05);
let world = build(|b| {
    b.body("plate").cube(400.0, 400.0, 10.0).rotate_x(tilt);
    b.body("marble").sphere(8.0).glass().at(0.0, 0.0, 30.0);
})?;

// a step, pure. a batch is a world with a leading axis.
let next = step(&world, &action);
let next = step(&world.repeat(n), &actions);

// lenses read a world; they do not hold it.
let film  = Camera::new(pose, budget).see(&world);
let mix   = Microphone::at(listener).see(&world);
let score = reward.see(&world);

// a rollout is a trajectory; a gradient is a lens through a rollout.
let traj = rollout(&world, &policy, steps);
let (score, grad) = tang::grad(|p| reward.see(&rollout(&world.with(p), &policy, steps).last()));

// worlds diff.
let gap = diff(&fitted, &cad);
```

A lens that needs a different layout builds its own resident copy
(kosm-render's `gpu::resident` already does) and invalidates it itself.

## What moves

The current `kosm-spike` is the core and the levels in one crate, and
`kosm-view` reaches into the levels. Sorting it:

**Into `kosm` (engine, from the levels):**

- `court/render`: evaluates vcad solids into BRep, builds the BVH, and
  produces the viewer's `Snapshot`. Nothing court-specific but the name.
  Becomes `kosm::brep`.
- `court/denoise` (`dataset`, `kpn`, `train`): the neural denoiser and its
  training loop. Becomes `kosm::denoise`.
- `pool::{Surface, Caustic, PoolGeometry}` and `splash::Droplet`: the bridge
  from the mpm surface to render, and the particle type. Become
  `kosm::fluid`.
- `colliders`, `scene`, `materials`, `audio`, `light`, `lamp`, `frame`,
  `glass`, `far`, `analytic`: already engine code, just misnamed.

**Into `sims/` (content):**

- `court/{aim, net, bake, parts}`, `pool/scene`, `skatepark`, `garage`,
  `room`, `warehouse`, and the marble spike from `main.rs`, each rewritten
  against `kosm::build` in place of its `.loon`.
- `kosm-view/src/court.rs` and `court_gpu.rs`: per-level viewer modes. They
  become `sims/court/game.rs`. `live.rs`'s pool half likewise.

**Deleted:**

- `kosm-view/src/history.rs`. The CPU temporal history that predates the
  port; `kosm_render::gpu::History` is the same design and the one that is
  maintained. The viewer uses it behind a trait.

**Split, not moved:**

- `kosm-render/src/pathtrace.rs` (7.4k lines) into `cpu/{film, camera,
  integrator, material, denoise}.rs`, mirroring `gpu/` so the tiers are
  visibly parallel.

## ipse, the first consumer

kosm knows no robot by name. ipse keeps everything that is *this* robot: the
K1 body, the mind (ipse-world, temperament, probe, split, experience), the
ledger, the deploy bus, and the ethics. Everything in ipse that is not about
the K1 specifically is training infrastructure that grew there because
nowhere else existed. Two crates cross the seam:

- **ipse-dojo splits two ways.** `Task` (spawn, score, held out,
  invariants), the frozen gate, and the artifact ledger are contracts and
  assertions over the three nouns with no loop of their own; they go into
  core `kosm` beside `diff` and the run hash. CEM and MAP-Elites are loops
  that produce policies; they go into `kosm-train`, and so do PPO (`rl.rs`)
  and behaviour cloning (`bc.rs`) from ipse-sim, which are generic over a
  `Task` once the observation layout is the task's. `Ladder` stays in ipse:
  escrow and the reluctance rule are the project's ethical commitments, and
  kosm must not be able to violate them by default.
- **ipse-map → kosm-scan.** Scans, hulls, object bake, fusion into a world.
  Already what kosm's skatepark and court consume, through hardcoded
  worktree paths today.

What the survey of ipse-sim found, in one sentence that appears ten times in
its comments: "so the two cannot drift apart". CPU rollouts vs GPU rollouts,
impulse vs penalty contacts, the model that simulates vs the copy that
renders, the gate's draw vs the trainer's draw, demo vs actor rollouts. Each
is a duplicated pipeline held together by a comment and a parity test. Each
is the cost of not having one pure Step and one World that every lens reads.

**The real robot is a Step.** `ipse-deploy` implements `Step` for the K1
over DDS, and its sensors are lenses. Then `capture` produces a `Trajectory`
that kosm-view plays like any other; sim2real is `diff(real.see(x),
sim.see(x))` per lens per timestep; the body-schema experiment, its own crate
today, is the gradient of that diff with respect to the inertial columns;
and the split contract's on-board column is a World, a Step, and the lenses
the self needs, with off-board as extra lenses. Losing the link loses lenses,
not the world. That is the "recognizably itself" test written as a type.

**Migration, three phases:**

1. **Render only.** ipse adds kosm for camera and ride lenses over its
   existing phyz rigs, through a `World::from_phyz` view. `see.rs`, the
   eight `*_video` examples, and the two hardcoded worktree paths go. No
   physics changes.
2. **The sims tree.** `sims/k1/` in ipse: `scene.rs` wrapping the URDF load,
   `skate/` and `ollie/` composing on top, `gate.toml` beside each task.
   Trainers call the batched Step; `rl_gpu.rs` and the six `gpu_*_parity`
   examples are deleted. `Task`, gate, and ledger move into core; the
   trainers move into `kosm-train`; ipse implements `Task` and keeps its
   policies, observation layout, and `Ladder`.
3. **Reality as a Step.** deploy implements `Step` and the sensor lenses,
   `capture` records a `Trajectory`, the body schema becomes a gradient of a
   diff.

**Decided before phase 1:** `World` wraps a phyz `Model` with a lossless view
both ways; whether it ever replaces it is a phyz question. The K1 URDF loader
stays in ipse, on a generic `kosm::build::urdf`.

## Order of work

1. Rename and split: `kosm-spike` → `kosm` + `sims/`. Promote the engine
   pieces above. Move the per-level viewer modes into their sims. Pure moves
   and `use` rewrites; the build is the check. Then the three nouns as
   traits, `_template`, the run hash and `out/<hash>/`, `--out`, and a
   `CLAUDE.md`, so the agent loop closes before any level loses its loon.
2. Externals to git revs. Replace the phyz worktree paths and vcad branch
   with revs; local-checkout overrides go in `.cargo/config.toml`, so a clean
   clone builds.
3. The demo. One file under fifty lines: the K1 on a board, an ollie, the
   gradient of the landing score with respect to wheelbase and deck concave,
   step the geometry, re-render, before and after. If it runs, every claim
   in this document is true. This needs ipse phase 1 and 2.
4. Delete `kosm-view/history.rs` behind a history trait.
5. Split `pathtrace.rs` into `cpu/`.
6. Tag `kosm-render`, point vcad at the tag, drop the `[patch]` back-edge.

1 and 2 are structure only and go first. 3 is the foundation's proof and goes
before anything is called done. 4 and 5 wait for the next time those files
are touched. 6 waits for the denoiser to stop changing daily.

## Papers

- Brax, 2106.13281. Isaac Gym, 2108.10470. Isaac Lab, 2511.04831.
- The Essence of Entity Component System, 2606.14919.
- SyDRA, 2406.05487. Game engine architecture recovery, 2303.02429.
  Visualising subsystem coupling, 2309.06329.
- DiffTaichi, 1910.00935. Dojo, 2203.00806. Nimble, 2103.16021.
- Gradients are Not All You Need, 2111.05803. Do Differentiable Simulators
  Give Better Policy Gradients?, 2202.00817.
- Dr.Jit, 2202.01284. Differentiable Rendering: A Survey, 2006.12057.
