# Rune cove: implementation plan

Date: 2026-09-09. Design: `2026-09-09-rune-cove-design.md`. Status: steps 1–7
built and committed the same day; test 5 (GPU vs CPU) not written because the
GPU tracer cannot trace the being's mesh, so the window runs the CPU tier.

Eight steps. Each names the files it touches, what it builds on, and the
check that says it is done. Steps 3, 4+5 and 6 are independent once 1 and 2
are in; 7 needs 3 and 6; 8 is last. Everything runs under
`cargo test -p kosm-spike --release -- cove` except the window itself.

## 1. The level and its scene

**Files:** `levels/cove.loon`, `crates/kosm-spike/src/cove/mod.rs`,
`crates/kosm-spike/src/cli.rs`, `lib.rs`.

- Write the document with the design's knob table. Roots: `beach` (tilted
  slab), `cliff` (box), `rocks` (spheres and cylinders), all unioned into
  `ground`; `door` as its own root; `aperture` as a `defparam` disc, not
  geometry (the door face is solid stone, the aperture is where the score
  is read).
- `CoveScene::load` resolves knobs to metres through `AuthoredScene`
  (`parameter`, `millimetres`), the way `SkateparkScene::load` does.
- `kosm-spike --cove [level]` in `cli.rs`, defaulting to the bundled file.
  For now it evaluates the document and writes `out/cove/cove.stl` and an
  isometric SVG through the skatepark's `write_parts`.

**Done when** the document evaluates, the STL opens, and the CLI test in
`cli.rs` covers the new flag.

## 2. The bake and the first test

**Files:** `cove/bake.rs`, `cove/tests.rs`.

- Reuse `skatepark::parts_of` with a `collides` that admits every root but
  `door`, then `skatepark::bake_parts` with `BakeOpts { cell: 0.1, pad: 0.3,
  volume: Some(cove bounds) }` into `out/maps/cove/`. At 100 mm over a 40 m
  by 40 m by 10 m cove that is 16 M cells; note the size in the report.
- **Test 1, the marble on the beach.** Adapt `skatepark::roll_from` (the
  `step` in `skatepark.rs` is the SDF contact step; lift it to
  `cove/sim.rs` so both use it). Release a glass sphere on the slope at
  rest, roll it to the waterline, and check `v² = 10/7·g·Δ` on a plane
  slope to within the skatepark's tolerance, with lateral drift under
  `1 cm`. A leaning normal shows up in the drift first.

**Done when** test 1 passes and `--cove` writes the map directory.

## 3. The being and the door

**Files:** `cove/sim.rs`.

- `Cove` holds the phyz `Model`, `State`, the `SdfGrid`, the contact
  cache and the door's gate. The being is `add_free_body` with
  `Geometry::Capsule { radius, length }` (phyz has it; ipse-map's contact
  path handles it for the K1's limbs). Mass from glass density and the
  capsule's volume.
- **Upright.** Each step, before `aba`, add a torque on the free joint
  proportional to the tilt of the body z-axis from world z, with damping,
  and a bias toward the player's commanded tilt. This is a spring, not a
  constraint: a shove tips it and it recovers.
- **Walking.** `Input { forward, strafe, yaw_delta, tilt_delta }`. A
  horizontal force at the centre of mass along the facing, zero once the
  horizontal speed passes `walk_mps` (1.4). Yaw is integrated into the
  being's facing and fed to the spring as the target heading.
- **The door.** `add_revolute_body` on a fixed hinge body at the door's
  edge, axis z, with a swing limit of 100°. A spring torque toward open is
  applied only while `gate.open` is set; step 4 sets it.
- `Cove::step(&Input)` is one `dt`; `Cove::snapshot()` returns the being's
  and the door's transforms for the renderer, the way `court::Snapshot`
  does.

**Done when** the being stands on the slope for ten seconds with drift
under 1 cm, walks at the capped speed, and the door swings when the gate is
forced open in a test.

## 4. The rune score

**Files:** `cove/rune.rs`, `cove/render/mod.rs` (the geometry half only).

- Build a `kosm_render::pathtrace::Scene` with the being as a dielectric
  (`Pbr` with `transmission: 1.0`, `ior: n_d`; that is what
  `is_caustic_refractor` looks for), the door as an opaque face, and the
  design's `Sun`. `caustics::trace` already splits photons between area
  lights and the sun and aims them at the refractor's bounds, so the
  being's caustic falls out of it with no new tracing code.
- `score(map: &CausticMap, door: &DoorFrame) -> f64`: sum the power of
  photons within `aperture_r` of the aperture centre on the door face over
  the sun's power on the being (`irradiance` times the capsule's
  projected area toward the sun). Expose the photon list for this through a
  `CausticMap::power_within(centre, normal, r)` method in kosm-render.
- **Solve the authored solution.** A coarse grid over the beach and over
  tilt on the f64 score, then Nelder–Mead from the best cell, and write
  `solution_x_mm`, `solution_y_mm`, `solution_tilt_deg` back into the
  document with `AuthoredScene::with_parameters`, as the marble writes its
  solved knobs. The solution is a solved parameter, not a designer's guess.
- **Test 2.** At the solved pose the score exceeds `open_frac`; ten metres
  along the beach it is below 0.02.

**Done when** test 2 passes and `--cove` prints the solved pose and score.

## 5. The hint

**Files:** `crates/kosm-spike/src/glass.rs`, `light.rs`, `cove/hint.rs`.

- `glass::Shape::Capsule { a, b, r }` with `enter`/`exit` generic over
  `Scalar`: two sphere caps and a cylinder, nearest entry and farthest exit.
  Add it to `to_dual`.
- `light::trace` receives on the plate at `z = 0`. Add a receiver frame:
  the door's origin and basis, so the door face is the plate. The sun is a
  lamp 1 km out along the sun direction; the existing cone sampling covers
  the being's bounding sphere from there.
- `hint::gradient(cove, pose) -> (dx, dy, dtilt)`: the score on `Dual` with
  the being's `x`, `y` and tilt as the dual parts, one at a time.
- **Test 3.** The gradient matches central differences to four digits at
  three poses, as `light.rs` already checks for `n_d`.
- **Test 4, solvability.** From a 6 by 6 grid of spawns over the beach,
  gradient ascent with a step of 0.2 m reaches `open_frac` within 200
  steps from every one. Write `out/cove/solvable.txt` with the path lengths.
  If a spawn fails, that is a level bug, and the fix is `being_r_mm` or
  `being_h_mm`, not the test.

**Done when** tests 3 and 4 pass.

## 6. The picture, offline

**Files:** `cove/render/mod.rs`, `cove/render/materials.rs`.

- Mirror `court::render::Scene`: static parts from the `ground` root,
  the being and the door as placed solids from the snapshot, `Sun`, a
  `GradientEnv` sky, and `caustic_map()` from step 4's trace. Materials
  flat: sand, rock, stone, one colour each, roughness 0.9. The sea is a
  `HeightField` with an authored swell at `sea_z`.
- `--cove` writes `out/cove/frame.png` through the CPU integrator at the
  authored camera, being at the solved pose, so the caustic is on the
  door in the picture.

**Done when** the frame shows the being, its shadow, and its caustic on the
door, and a reviewer can read the design's look in it.

## 7. The window: `kosm-view --rune`

**Files:** `crates/kosm-view/src/rune.rs`, `viewport.rs`, `main.rs`.

- Input. `Key` gains `W A S D`, `Event` gains `Look(dx, dy)` from
  `DeviceEvent::MouseMotion`, and the viewport reports key releases so
  held keys read as a held force. The court's bindings are untouched.
- The loop is `court::run`'s three threads: `simulate` steps `Cove` with
  the latest `Input` and hands `Timed` snapshots ahead by the measured
  latency; `render_worker` traces one raw sample a pass over
  `court_gpu::Stage` built on the cove's render scene; the viewport blits.
  Lift the thread plumbing out of `court.rs` into a shared module only if
  the copy is larger than the abstraction; the court did not need it yet.
- Placed instances: ground and door packed once, being and door moving.
  The caustic map is retraced (on the simulation thread, at
  `photons: 50_000`) whenever the being has moved more than 1 cm or tilted
  more than 0.5°, and repacked to the `CausticPack`.
- The history mask: extend `Pose` so the mask covers the being, its shadow
  and the door's face when the caustic changed, so stillness converges and
  walking stays live.
- Camera: third person, 3 m behind and 1.5 m above the being along its
  facing, `fov 50°`.
- **Test 5.** `--shot` on the rune tier at the solved pose against the CPU
  frame of step 6, compared on the door's face at the court's tolerance.
- **Test 6.** Headless: drive the sim to the solved pose, hold one second,
  assert the door's hinge angle rises; hold at 0.9 s, assert it does not;
  place the being behind the sun line, assert the score is zero.

**Done when** the cove runs in the window at walking pace and tests 5 and 6
pass.

## 8. The glow and the glint

**Files:** `cove/render/mod.rs`, `rune.rs`.

- The aperture rim is a thin emissive ring whose radiance is `score ·
  glow`, updated per snapshot.
- After 30 s without the score rising, a small emissive sphere is placed on
  the sand 0.5 m from the being along the gradient's horizontal part, and
  removed when the score rises.

**Done when** both show in the window and no text has been added.

## Risks and fallbacks

- **The capsule's caustic is a line.** If step 4's grid search finds no
  pose over `open_frac`, shorten `being_h_mm` toward a sphere before
  touching the score. The sweep in step 5 reports it.
- **Photon count versus pace.** 50 k photons is a few milliseconds on the
  CPU for one refractor; if it is not, trace only when the being stops
  moving and show the last map while walking.
- **The SDF at 100 mm.** 64 MB is fine; if the bake is slow, `volume`
  restricts the sampled box to the beach and the door's apron.
- **Input plumbing.** `viewport.rs` has no key-up today. If key-up is
  awkward under winit's event model, the fallback is toggling keys, which
  is worse to play but unblocks 7.
