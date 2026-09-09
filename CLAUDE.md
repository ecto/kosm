# kosm, for an agent

Three nouns and one loop. `docs/architecture.md` is the why; this is the how.

- **World** — columns (phyz's `Model` + `State`, plus materials, lights,
  params). No method on it runs anything.
- **Step** — `World → World`, pure. A batch is a `Vec<World>`.
- **Lens** — `World → observation`. A camera is a lens, a reward is a lens,
  a probe on a column is a lens.

One import: `use kosm::prelude::*;`. Everything it exports has a doctest —
`cargo test -p kosm --doc` runs them, and the snippets below are copied from
those doctests and the crate's own tests, so they compile.

## The headless loop

You cannot see a window. Run the sim, read the files, edit, run again.

```
cargo run -p kosm-cli -- list                       # the tree under sims/ is the registry
cargo run -p kosm-cli -- run _template --out out/   # a few seconds, tiny render budget
# then look at out/<id>/frame.png and out/<id>/metrics.json
```

`<id>` is `hash(sim path, params, seed, kosm git rev)`; the run prints it.
Same id, same bytes: a changed frame means changed code. `--view` opens
kosm-view and is never what you want here.

## Six recipes

### 1. Build a world

The level is a Rust function that emits CAD. `build` runs it, walks the
document it produced into phyz colliders, and hands back a `Built`:

```rust
use kosm::prelude::*;
let built = build(&Params::default(), |b| {
    let tilt = b.param("tilt", 5.0);            // a knob: degrees, mm, vcad's units
    b.body("plate").boxed(400.0, 400.0, 10.0).material("pla").rotate_y(tilt);
    b.body("marble").sphere(8.0).glass().dynamic(0.012).at(0.0, 0.0, 30.0);
})?;
assert_eq!(built.world.param("tilt"), Some(5.0));
let steeper = built.with(&[("tilt", 9.0)])?;    // re-runs the closure; never mutates
assert_eq!(steeper.world.param("tilt"), Some(9.0));
# Ok::<(), anyhow::Error>(())
```

Authoring is **millimetres and degrees** (vcad's), simulation is metres and
radians (phyz's); `kosm::scene::MM` is the one place they cross. Origins are
vcad's too: `cube` has a corner at the origin, `cylinder`'s base is at
`z = 0`, `sphere` and `boxed` are centred. `Shape` carries `translate` /
`at`, `rotate_x/y/z`, `scale`, `union`, `difference`, `intersection`,
`linear_pattern` and `circular_pattern`; a `difference` goes through
`colliders.rs`'s convex decomposition, so a cup stays hollow.

`Built` gives you `document` (for the STL, the SVG, `brep::Scene`), `params`,
`bodies[i].colliders`, and `world`. `World::from_phyz` is lossless both ways
— `world.phyz()` hands the model and state back, `world.into_phyz()` by
value — so a sim whose physics needs more than `build`'s default rig takes
`built.document`, derives what it wants and builds its own `Model`; see
`sims/marble/scene.rs` and `sims/marble/mod.rs`.

### 2. Add a body

Bodies are phyz's, because the physics is phyz's. Build the model, then wrap
it. From `kosm::world::demo_marble`:

```rust
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{GeomInstance, Geometry, ModelBuilder};
let (r, m) = (0.01, 0.005);
let i = 0.4 * m * r * r;
let mut model = ModelBuilder::new()
    .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
    .dt(1e-3)
    .add_free_body("marble", -1, SpatialTransform::identity(),
        SpatialInertia::new(m, Vec3::zeros(), Mat3::from_diagonal(&Vec3::new(i, i, i))))
    .add_fixed_body("plate", -1, SpatialTransform::identity(),
        SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01))
    .build();
model.bodies[0].geometry = Some(Geometry::Sphere { radius: r });   // the camera lens draws this as glass
model.bodies[1].collisions = vec![GeomInstance {
    name: Some("plate".into()),
    origin: SpatialTransform::identity(),
    geometry: Geometry::Box { half_extents: Vec3::new(0.1, 0.1, 0.005) },
}];
```

### 3. Batch a step

A batch is a `Vec<World>`; `step_batch` walks it with rayon.

```rust
use kosm::prelude::*;
let (model, state) = kosm::world::demo_marble();
let world = World::from_phyz(model, state);
let step = PhyzStep::new(1e-3);

let next = step.step(&world, &Action::none());     // one step, pure
let traj = rollout(&world, &step, &Zero, 50);      // index 0 is the world that went in
assert_eq!(traj.len(), 51);

let stepped = step_batch(&step, &world.repeat(8), &[]);   // eight at once
assert_eq!(stepped.len(), 8);
```

`step_batch` takes either no actions (every world gets `Action::none()`) or
exactly one per world; anything else panics rather than quietly recycling.

### 4. Take a gradient

Gradients are one estimator, not the estimator (architecture.md rule 4).
Batched rollouts are the zeroth-order path and cost nothing to reach:

```rust
use kosm::prelude::*;
let (model, state) = kosm::world::demo_marble();
let world = World::from_phyz(model, state);
let step = PhyzStep::new(1e-3);
let score = |w: &World| -rollout(w, &step, &Zero, 100).last().unwrap().q()[5];

let h = 1e-4;
let mut lo = world.clone(); lo.state_mut().q[5] -= h;
let mut hi = world.clone(); hi.state_mut().q[5] += h;
let d_score_d_height = (score(&hi) - score(&lo)) / (2.0 * h);
```

The first-order path is phyz's convex-contact adjoint through
`phyz_diff::convex_adjoint_gradient`, and tang duals through
`kosm::frame` / `kosm::light` for the render and the optics.
`sims/marble/mod.rs` does all three and checks each against central
differences — read it before writing a fourth.

### 5. Add a lens

```rust
use kosm::prelude::*;
use phyz_math::Vec3;
let (model, state) = kosm::world::demo_marble();
let world = World::from_phyz(model, state);

let height = Probe::q("bead height", 5);            // one column entry
assert_eq!(height.see(&world), 0.2);

let low = reward("low is good", |w: &World| -w.q()[5]);   // a closure
assert_eq!(low.see(&world), -0.2);

let cam = Camera::look_at(Vec3::new(0.25, -0.35, 0.30), Vec3::new(0.0, 0.0, 0.10), 320, 240, 4);
let frame = cam.see(&world);                        // kosm-render; `.with_depth(true)` for depth
```

A lens of your own is one `impl`:

```rust
struct MissDistance { goal: phyz_math::Vec3 }
impl kosm::lens::Lens for MissDistance {
    type Out = f64;
    fn see(&self, w: &kosm::world::World) -> f64 {
        (phyz_math::Vec3::new(w.q()[3], w.q()[4], w.q()[5]) - self.goal).norm()
    }
}
```

### 6. Train a policy

Loops live in `kosm-train`, contracts in `kosm`. A `Task` says what the
episode is; `TaskEnv` turns it into the flat thing a trainer wants — an
observation lens, a per-step reward lens, and `control_every` substeps per
action — and `ppo::train` runs the loop over a batch of them.

```rust,ignore
use kosm::prelude::*;
use kosm::world::World;
use kosm_train::{env::TaskEnv, ppo::{self, PpoConfig}};

// One env per worker; rayon runs them side by side, seeds are drawn on this
// thread so the batch does not depend on how they were scheduled.
let envs: Vec<_> = (0..8)
    .map(|_| TaskEnv::new(
        Cup,                                        // any `impl Task`
        PhyzStep::new(1e-3),
        |w: &World| vec![w.q()[3], w.q()[4], w.q()[5]],   // what the policy sees
        |w: &World| -w.q()[5],                            // what it is paid, per step
        3,                                                // action width
        20,                                               // 20 substeps = 50 Hz control
    ).with_privileged(|w: &World| vec![w.state().time]))  // critic-only, optional
    .collect();

let cfg = PpoConfig { episodes_per_iter: 32, ..PpoConfig::default() };
let (actor, _critic, history) = ppo::train(envs, cfg, 300, |it, iter, _| {
    println!("{it:4}  return {:8.2}  len {:5.1}  kl {:.4}", iter.mean_return, iter.mean_len, iter.kl);
});
ppo::save_actor(&actor, "out/cup.actor", "300 iterations")?;
```

`load_actor` reads it back; the header is versioned, so a file from a
different observation schema is refused rather than read into the wrong
columns. `log_std` is not saved — a warm start must call
`ppo::set_init_std` before it collects, and `train_from` does.

The other loops are `kosm_train::search` (CEM and MAP-Elites over a
parameter vector, no gradient) and `kosm_train::bc::fit_actor` (regress an
actor onto demonstrated actions before PPO ever runs). `kosm_train::artifact`
is the **policy** ledger — role map, plant, measured score — as opposed to
`kosm::ledger`, which is the run ledger.

## Writing outputs

```rust
use kosm::prelude::*;
// inside a sim's `pub fn run(args: &kosm_cli::Args) -> anyhow::Result<()>`
let params = vec![Param::new("height", 0.3)];
let mut rec = Recorder::new(args.out(), "_template", &params, 0)?;   // out/<run id>/
rec.metric("height_end_m", 0.02)?;
rec.png("frame.png", &image::RgbaImage::new(4, 4))?;
rec.finish()?;   // writes metrics.json and run.json, prints the id
```

`sims/_template/mod.rs` is under 80 lines and does exactly this, through the
four stages (build → run → observe → optimise-optional). Copy it.

## Snapshots, gates, the ledger

- `kosm::snapshot::assert_close("marble/released_300", &values, 1e-6)` diffs
  against `sims/marble/snapshots/released_300.json`, writing it when it does
  not exist. `KOSM_UPDATE_SNAPSHOTS=1` re-records. `assert_image_close` is
  the same for a frame, with a mean-channel-difference tolerance.
- `diff(&a, &b)` is per column plus per param, with `.is_within(tol)`. A
  shape change reads as infinity, not as a small number.
- A `Task` is spawn / build / horizon / score / held_out / invariants.
  `gate::check(&task, &policy, &spec)` runs a `gate.toml`'s frozen draws and
  returns one number; `Ledger::open(dir).append(&Entry::from_gate(..))`
  writes it to an append-only `ledger.jsonl`. `sims/marble` has both:
  `Cup` implements `Task` and `sims/marble/gate.toml` is its frozen gate.
- `check_invariants(&task)` runs before compute is spent. A task that
  declares none is a task whose author has not yet been surprised.

## Caveats

- **Disk.** This machine runs near full. `target/` is tens of gigabytes.
  Check `df -h /System/Volumes/Data` before a build; do not `cargo clean`
  (the rebuild costs more than it frees) and do not build `--release` unless
  a number depends on it.
- **GPU.** The default build is headless and CPU. The `view` feature pulls
  in eframe, egui and wgpu and is the heaviest thing in the graph — leave it
  off. `kosm-render`'s GPU tier and `kosm-mpm` need a real device and are
  not available in a headless test.
- **Render budget.** `Camera`'s `spp` is the cost. Four is a thumbnail,
  ninety-six is the marble's beauty frame and takes seconds per frame.
  A test or a template stays small.
- **Externals.** phyz, vcad and tang are git revs in the workspace
  `Cargo.toml` — a clean clone builds with no sibling checkouts. `.cargo/config.toml`
  patches those same sources back to the local checkouts on this machine, so
  edits next door still land in a kosm build; it is committed, it overrides the
  manifest's patches, and CI must run without it. `tang` is unified by a
  `[patch.crates-io]` so `tang::Scalar` is one trait across the graph. Changing
  any of it is a full rebuild.
- **One phyz.** There used to be a gap here: ipse-map pinned phyz twelve commits
  behind the rev kosm pins, cargo will not let a manifest `[patch]` pull a git URL
  onto itself, and a clean clone got two `Model` types. `crates/kosm-scan` is
  ipse-map lifted into this workspace, so ipse is no longer a dependency and the
  problem is gone — every phyz in the graph is the one rev the manifest pins, on a
  clean clone as much as here.
