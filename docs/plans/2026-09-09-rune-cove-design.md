# Rune: one cove, a glass being, one rune door

Date: 2026-09-09. Status: design, nothing built.

## The claim

Myst was legendary because its world looked more real than any real-time
engine could manage and because its puzzles were mechanisms in a consistent
world. Every one of those mechanisms was a state machine painted to look like
physics. Rune's are not painted. A lever moves a mirror and a ray is traced;
a bell rings and its geometry rings; a rune is light through glass. The
puzzles are the engine's subsystems with a player attached, which is why the
engine wants this game.

The look is the new Switch Sports: flat saturated albedo, clean silhouettes,
soft sky light, no textures to speak of. That is what a path tracer produces
when it is handed clean CAD solids and a sky, so the style and the stack
agree. The one thing that style needs and we do not have is grass. The first
slice is a cove of sand, rock and water so that grass is nobody's problem
yet.

## The slice

You are a being made of glass, standing on a beach in a cove. The sun is
low. Across the sand a stone door is set into the cliff, and in the door is
a small round aperture, the keyhole of a rune. Your body is a lens. Stand in
the right place, at the right tilt, and the sun through you focuses on the
aperture; hold it there and the door swings open. That is the whole game.

What the slice proves:

- a body walks on authored terrain through real contact (the K1's path);
- the world is path traced live around a moving dielectric;
- a rune is a caustic, scored physically, not a trigger volume;
- the hint is a gradient: the same trace on `Dual` says which way to move.

Not in the slice: grass, a second puzzle, the tide as a clock, sound, an
agent that writes the level, the cel pass. Each is a follow-up with its own
design.

## The cove: `levels/cove.loon`

Z-up, millimetres, vcad's conventions, one document. Knobs:

| knob | default | meaning |
|---|---|---|
| `cove_mm` | 40000 | the level is a `cove_mm` square, sea along -y |
| `beach_slope` | 0.06 | sand rises from the waterline at this grade |
| `cliff_h_mm` | 6000 | the back wall of the cove |
| `door_w_mm`, `door_h_mm` | 1800, 2600 | the stone door, hinged on one edge |
| `door_x_mm` | 0 | where along the cliff the door sits |
| `aperture_r_mm` | 120 | the rune's keyhole, a disc on the door face |
| `aperture_z_mm` | 1500 | keyhole height above the sand at the door |
| `sun_az_deg`, `sun_el_deg` | 200, 22 | a low afternoon sun |
| `being_r_mm`, `being_h_mm` | 350, 1400 | the being: a capsule of glass |
| `n_d` | 1.5168 | the being's d-line index (N-BK7's) |
| `spawn_x_mm`, `spawn_y_mm` | -9000, -6000 | where you start |
| `sdf_cell_mm` | 100 | bake spacing; the being's feet are 350 mm |
| `sea_z_mm` | 0 | the waterline |

The terrain is a height field written as vcad geometry: the beach is a
tilted slab, the cliff a box, a few rocks are spheres and cylinders unioned
in, and the whole solid is one `Union` so the bake has one inside. The door
is a separate root so it can move. The sea is not geometry; it is
`kosm_render::HeightField` at `sea_z_mm` with a small authored swell, purely
a surface to render and to stop the being at (a wall collider at the
waterline, this slice).

The document is the only description of the geometry, as in every other
level. `cove.rs` reads it and nothing else.

## The being

A capsule of glass, `being_r_mm` by `being_h_mm`, on one phyz free joint.
No limbs. It is the first spike's marble grown up, and its curvature is the
lens.

- **Standing.** The capsule's feet are the SDF contact path the K1's feet
  use (`ipse_map::find_terrain_contacts_model` on the baked cove), with the
  skatepark's solver settings. An orientation spring on the free joint holds
  it upright with a low centre of mass, so it is a weeble, not a ragdoll.
- **Walking.** WASD is a horizontal force at the centre of mass, capped at a
  walking speed; mouse X yaws the being; mouse Y tilts it a few degrees
  about its own horizontal axis. The tilt is the second knob of the puzzle:
  a capsule's focus moves with its lean.
- **Camera.** Third person over the shoulder, as in the screenshot that set
  the look: behind and above, looking down the being's facing.

Two knobs the player controls, position and tilt. `n_d` is authored in this
slice; "drink from the tide pool to change your index" is the obvious second
puzzle and is not built here.

## The rune

The rune is the caustic the sun throws through the being, and the door
reads it.

- **The trace.** `kosm_render::caustics::trace` already shoots photons from
  a light through dielectrics and deposits them in a `CausticMap`; the
  court's backboard is the refractor there. Here the light is the sun, the
  refractor is the being, and the receiver is the door face. The map is
  retraced whenever the being moves and packed to the GPU as a
  `CausticPack` so the picture shows the focus crawling across the door as
  you walk.
- **The score.** `deposited_power` inside the aperture disc, over the power
  incident on the being. It is a number in [0, 1] and it is physical: it is
  how much of the sun you are putting through the keyhole.
- **The door.** A hinge joint in phyz with a spring that is disengaged until
  the score has been above `open_frac` (0.3) for one second, then drives it
  open. It is a body with mass; it swings, it does not teleport.

Nothing here is a trigger. Move the sun in the document and the solution
moves with it.

## The hint

`light.rs` traces a spectral caustic generic over `tang::Scalar`. On `Dual`
the same trace gives the score's derivative with respect to the being's
position and tilt, the way `frame.rs` gives "put the shadow here" its
gradient. The hint is that gradient, shown as light: the aperture's rim
glows in proportion to the score, and after thirty seconds without progress
a glint appears on the sand a short step along the gradient. The player
follows light to make light. Text never appears.

The same gradient is the author's solvability check: from every spawn in a
grid over the beach, gradient ascent must reach `open_frac`. A cove that
fails that test does not ship.

## Rendering

`kosm-view` gets a `--rune` tier beside `--court` and `--ride`, and it is
the court's loop: three threads, the simulation stepping ahead by the
measured latency, one raw sample a pass, `history.rs` accumulating with
reprojection and the geometric mask. What the mask throws away here is what
the being, its shadow and its caustic touched, so standing still converges
and walking stays live.

- Geometry is `GpuScene::placed`: cove and door packed once, the being and
  the door as instances that move.
- The being is the rough dielectric in `bsdf.wgsl` with spectral IOR from
  `n_d`; the sun is `Sun`, the sky the builtin gradient environment.
- Materials are flat: sand, rock, stone, one colour each, roughness high.
  That is the look. A cel quantise on demodulated illumination is one extra
  pass in `history.wgsl` and is a follow-up, not this slice.
- The caustic pack is sampled in the integrator on the door's face as the
  court samples it on the floor.

The CPU integrator stays the reference, as everywhere else.

## Code

- `levels/cove.loon`: the level.
- `crates/kosm-spike/src/cove.rs`: `CoveScene` (knobs resolved to metres),
  the bake to `out/maps/cove/`, the being's model, the door's model, the
  rune score, the `Dual` hint, and the solvability sweep.
  `kosm-spike --cove` runs the bake and the sweep and writes one frame.
- `crates/kosm-view/src/rune.rs`: the tier. Input, camera, the loop.

No new crate. The game is a level, a scene module and a window tier, which
is what the court and the skatepark are.

## Tests

1. **The marble on the beach.** A glass sphere released on the slope rolls
   to the waterline at `v² = 10/7·g·Δ`, the skatepark's check on this bake.
2. **The rune scores.** With the being placed at the authored solution the
   score exceeds `open_frac`; ten metres away it is below 0.02.
3. **The hint is right.** `∂score/∂(x, y, tilt)` on `Dual` matches central
   differences to four digits, as `light.rs` already does for `n_d`.
4. **The cove is solvable.** Gradient ascent from a grid of spawns reaches
   `open_frac` from every one.
5. **The picture agrees.** The GPU frame matches the CPU reference on the
   door face within the court's tolerance, caustic included.
6. **The door is a door.** Above threshold for one second it opens; below,
   it does not, and it never opens while the being is behind the sun line.

## Open questions

- The bake at 100 mm over a 40 m cove is 16 M cells; fine for one cove, not
  for an island. Sparse or multi-resolution SDF is the follow-up when the
  world grows.
- A capsule's caustic is a line, not a point, at most tilts. If the score is
  too easy or too hard to hit we change `being_r_mm`/`being_h_mm`, and the
  solvability sweep says which.
- The sea is a rendered plane with a wall. When the tide becomes the clock,
  the Zakharov far field and the MPM residual replace it, and the design
  for that is the pool's.
