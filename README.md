# Kosm

A game engine where the level is a CAD file, the physics is the robot's
simulator, and every knob has a gradient. Built on
[vcad](https://github.com/ecto/vcad) (geometry), [phyz](https://github.com/ecto/phyz)
(differentiable multi-physics), [tang](https://github.com/ecto/tang) (one scalar,
one IR) and loon (scripting).

`AuthoredScene` is the narrow shared boundary: it evaluates a Loon/vcad source,
resolves its parameters, converts authored millimetres at the computation edge,
and can write solved parameters back. Marble and pool then attach different
computations to that source; they do not share an artificial simulation loop.

## the marble (`crates/kosm-spike`)

The first spike. The level is [`levels/marble.loon`](levels/marble.loon):
geometry in vcad's loon vocabulary, knobs as `defparam`s (tilt, release point,
marble, horizon). Everything else is derived from that one file:

- a vcad document, written as a printable STL and an isometric SVG
- a phyz rollout of a glass marble on a tilted plate
- an exact adjoint gradient of "distance to the cup" with respect to the release
  point, checked against central differences
- a two-knob tilt solve by finite differences of the same rollout
- three RGBD frames from phyz-camera
- `out/marble.wav`: the hinted run, heard — modal synthesis in a room, no samples

```bash
cargo run --release -p kosm-spike            # or: kosm-spike levels/other.loon
open out/track.svg out/frame_before.png out/frame_hint.png out/frame_tilted.png
afplay out/marble.wav
diff levels/marble.loon out/solved/marble.loon   # the solved knobs, written back
```

The document is the only description of the geometry. `colliders.rs` walks
the vcad IR (`Union`, `Translate`, `Rotate`, `LinearPattern`,
`CircularPattern` over `Cube` / `Cylinder` / `Sphere`) into phyz colliders on one
fixed track body, and checks every collider's support function against the
tessellation before the first step. Plate, walls and cup are all real colliders;
the cup is a ring of box segments with a mouth facing uphill so it catches.

### a cup that is actually hollow ([`levels/marble-cup.loon`](levels/marble-cup.loon))

A ring of box segments is a union of convex primitives, which is what a phyz
collider is. Modelled the way a person would actually model it — a cylinder with
a bore taken out and a slot cut in the uphill side — the cup is a `Difference`,
and a difference has no convex collider. Hulling it turns the cup into a puck
and the marble bounces off the lid.

So `colliders.rs` decomposes a `Difference` instead. It evaluates the subtree to
a mesh, clips that mesh against a family of convex sectors — a grid on the cut's
bounding box, or angular wedges about an axis through it — and hulls each
sector's share, including the points where the mesh's edges cross the sector
planes, so the pieces tile the solid rather than merely sample it. Candidates
are tried fewest-pieces-first and each is scored before it is accepted:
**coverage** (points all over the result's surface must each land inside some
piece, so the union is the solid and not a sieve) and **intrusion** (no point of
any piece may lie more than 0.5 mm inside anything that was cut away). The
first candidate that passes both wins; if none does, it is still the old single
hull and the old warning.

`verify_no_intrusion` is the second half of the pre-flight check, next to the
support test. The support test only ever sees the outside of the level, where a
cup and a puck are the same shape — it is exactly blind to this bug.

```bash
cargo run --release -p kosm-spike levels/marble-cup.loon
```

The cup comes out as 24 wedges reaching 0.376 mm into the bore, and the marble
is caught after the same hint and tilt solves as `marble.loon`. Both levels run;
the pattern-based cup is still there.

### the frame

`frame.rs` is a ray caster written once, generic over `tang::Scalar`. On `f64`
it renders the level with real penumbra shadows from a lamp of finite radius,
cast by the derived colliders themselves (800×600 in ~50 ms on the CPU). On
`Dual<f64>` the same code is its own derivative: the shadow pass's
`∂visibility/∂marble` matches central differences to five digits. Chained to
the contact adjoint, "put the shadow here in the image" solves for the release
point. Only box colliders are rendered so far; the decomposed cup's hull
pieces are not.

### light as physics

`light.rs` makes the marble glass. Light from the lamp refracts in and out
by Snell's law with Fresnel losses at each face, five wavelength bands with
Sellmeier dispersion (N-BK7's shape, the d-line index as the knob), and lands
on the plate as a spectral caustic: forward light transport, energy
conserving, generic over `tang::Scalar`. On `Dual` the same trace gives
`∂caustic/∂n_d`, which matches finite differences to four digits, and a
gradient fit recovers the marble's index of refraction from its caustic
(1.5169 against a true 1.5168) in nine steps. The generic Sellmeier is checked
against `vcad-kernel-optics` N-BK7 to machine precision.

`glass.rs` makes the camera see glass too: Fresnel splits each ray into a
reflected share (which can land on the lamp's disc, the highlight being the
lamp's image) and a refracted share that walks through the solid with up to
four internal bounces before it leaves and is shaded by whatever it lands on.
Three samples sit on a plate with a printed 5 mm grid: the marble, a cube and
a square pyramid, all convex solids the same code handles for the camera and
for the caustic tracer. Sizes are level knobs (`cube_mm`, `pyramid_mm`,
`pyramid_h_mm`, `sample_yaw_deg`) so they can be set to real samples.

### the pool

`kosm-spike --pool [frames]` drops a watermelon into the authored scene
[`levels/pool.loon`](levels/pool.loon) (`pool.rs`). Drop height, melon geometry
and density, water parameters, and recording cadence come from that scene.
Typed pool geometry is carried through rigid dynamics, snapshots, and both the
reference and live renderers. The fine-water/far-field solver still requires the
reference 50 m × 25 m × 2 m basin until its grids are scene-sized. The
melon is a phyz rigid body with Archimedes buoyancy and quadratic drag applied
as generalized forces, so it plunges, slows and floats. The water surface is a
wave field driven by the impact (deep-water dispersion, spreading and decaying
rings), not a fluid solve; the light is real: sun caustics traced through the
surface onto the tiles each frame, Fresnel sky reflection, refraction into
water that absorbs red first, and the melon seen through it. 1280×720 at
~0.3–1 s a frame on the CPU, encoded with ffmpeg to `out/pool.mp4`.

### the splash

`kosm-spike --splash [frames]` runs the same drop with the water simulated
(`splash.rs`): a dense-grid MLS-MPM for weakly compressible water over the
whole pool, APIC transfer with a FLIP blend, and the thing phyz-particle's
reference solver lacks, a rigid collider that pushes back. Grid nodes inside
the melon lose their normal relative velocity and the momentum that costs is
booked as the force phyz applies to the melon. The impact (740–860 N peak),
the crater, the Worthington jet and the drops are the fluid's. What the
coupling does *not* yet deliver is hydrostatics: a melon held submerged reads
8–25 N of support instead of the 68 N of displaced water, so a 5 %-buoyant
melon hovers at half depth instead of floating. That is the finding for
phyz-particle: at 2.5–3 cm cells a weakly compressible EOS can't resolve the
pressure integral over an 11 cm body; it wants a pressure projection or a
much finer grid.

### the garage

`garage.rs` drops the marble onto ipse's captured garage (`ipse-map`: a splat
for appearance, a fused-depth SDF to stand on) using phyz's own contact
pipeline with the plane swapped for the field, exactly as ipse-sim does for the
robot. The marble rolls into a 9 mm dip in the real floor at the acceleration
a 3 % grade predicts. Contact normals come from the level set, not `∇sdf`: the
fused field is truncated (|∇| ≈ 0.7) and its gradient leans a consistent few
degrees off the surface it bounds, which a frictionless marble reads as a
push (2.2 m in 2.2 s on a flat floor) and a robot's foot never notices. The
frame is the splat, rendered by tang-3dgs, with the path drawn on the floor.
Set `KOSM_MAP` to another map directory.

### the lamp

`lamp.rs` puts a point lamp in the room and changes the objective to "the
marble's shadow lands here". The light is an analytic ray-plane projection
with its Jacobian; the motion is the convex-contact adjoint; the chain rule
joins them at the marble's final position. Mid-roll, `∂shadow²/∂release`
matches central differences to four digits through the lamp and through
rolling contact in one backward pass. After the cup collision it is one-sided,
as documented for the contact adjoint. Two knobs solve the same target: the
release point (grid, then the chained adjoint) and the lamp position (closed
form: at fixed height the shadow is affine in the lamp's xy). phyz-camera
draws no shadows, so the frame composites the projected silhouette.

Two phyz bugs surfaced and were fixed in the phyz worktree this depends on:
sphere-on-box contact points were taken from the box's degenerate face
support (the marble fell through the plate), and the body-body adjoint froze
a sphere's contact point in the sphere's frame and let the sphere own the
contact normal (gradients off by 100×). Tests: `phyz/tests/sphere_on_fixed_box.rs`,
`phyz-diff/tests/sphere_body_body_adjoint.rs`.

### the skatepark

`kosm-spike --skatepark [level]` is a training level for the Booster K1 in
`../ipse`, which already has the skateboard rig and a K1 that rides it on a
flat floor. Its terrain is an `ipse-map` directory — a collision mesh and a
signed-distance grid baked from it — that a scenario file points at, so the
park is authored here as vcad geometry
([`levels/skatepark.loon`](levels/skatepark.loon): a mini ramp, one solid, the
transition radius, lip, width, flat, deck and coping as `defparam`s) and
baked with ipse-map's own baker into `out/maps/skatepark/` (`mesh.stl`,
`sdf.bin`, `map.toml`, `park.svg`, and a `scenario.toml` that stands the K1
on its board on the flat and shoves it at the transition). The field is
sampled along the ideal arc and reported against the analytic circle: 2.5 mm
at 10 mm cells, which is vcad's tessellation of the cylinder, not the bake.

The check is the court's e² test for a ramp. A wheel-sized solid sphere is
set on one transition and released, and rolled on the baked map through the
same SDF contact path the K1's feet use. Rolling without slip, its centre
reaches the flat at `v² = 10/7 · g · Δ` and climbs the far wall back to the
height it left. Measured: 2.480 m/s against 2.483 predicted, the far wall to
99.7 % of the release height, zero lateral drift. `tests/skatepark.rs` pins
all three.

Two engine bugs surfaced and were fixed in the stack this depends on. In
ipse-map, a point far past the SDF's volume saturated the `as usize` cast and
`ix + 1` wrapped to 0 in release, passing the bounds check and indexing off
the end of the grid. In phyz, a free joint's linear velocity is stored in the
body frame, and its Coriolis term `−ω × v` — the frame turning, not a force —
was integrated by explicit Euler, which turns *and* stretches: `|v|` grows by
`(ω dt)² / 2` a step, invisible on a trunk and a runaway on a 27 mm wheel at
74 rad/s (the wheel reached 45 m/s on the flat). The fix strips that term
from `aba`'s acceleration before the contact solve, which was assembled in
the start-of-step frame, and turns the solved velocity into the end-of-step
frame exactly afterwards; the adjoint carries the turn's tangent. Tests:
`phyz/tests/spinning_free_body.rs`, `ipse-map` `sdf::tests::outside_is_none`.

```bash
cargo run --release -p kosm-spike -- --skatepark
open out/maps/skatepark/park.svg
cd ../ipse && cargo run -p ipse-sim --bin train -- ../kosm/out/maps/skatepark/scenario.toml
```

## the sound (`audio.rs`)

Nothing is sampled. `audio.rs` asks `vcad-kernel-acoustics` for the level's
modes — its `strike` module is a free-free Euler–Bernoulli bar (`BarSpec` →
`fem_hz`, plus `free_free_beta_l` / `mode_shape` for the strike gains), which
is the only structural eigensolver the crate has; the rest of it is air-side.
So the plate is run through that solver on both of its in-plane axes, and the
walls and the cup arc each get their own bar. Materials are the level's
`track_*` and `marble_*` `defparam`s. The rollout's contacts are the
excitation: a velocity jump along a contact normal is an impact, scaled by a
Hertzian contact time (a hard light marble is a bright hammer), and everything
else in contact is rolling — speed-scaled noise poured through the plate's
modes.

A dry modal sum sounds like it is in space, so `room.rs` puts the tray on a
table in a shoebox room (`room_*`, `table_*`, `ear_*`, `room_absorb_*`) and a
head 60 cm from it. The room impulse response is the direct path plus every
image source up to order 6 (Allen–Berkley, `1/r`, `√(1−α)` per bounce, air
absorption as a distance-dependent HF loss, fractional delay), handed over at
the mixing time to an exponentially decaying seeded-noise tail whose `RT60`
is Eyring on the room's own volume and absorption. Two receivers 17 cm apart
give the ITD for free and a head-shadow low-pass gives the ILD; the tail is
decorrelated per ear. The marble moves, so the dry sound is rendered onto
three anchors — release, cup, end wall — and each gets its own RIR. Two more
things stop it sounding synthetic: every mode is weighted by its **radiation
efficiency** (a baffled piston of the same area, times a coincidence factor
for the plate), which tames the big slow low modes; and rolling is **surface
roughness** — noise low-passed at `v / 0.2 mm`, the speed at which the marble
crosses the printed layer lines, as a continuous force on the plate rather
than a stack of impulses.

The marble is a sphere and no bar, so its modes come from Lamb's radial
frequency equation, solved in `sphere_radial_hz`. It answers ~450 kHz: a 10 mm
glass sphere is not a bell, it is an ultrasonic click. Those modes are printed
and then gated out of the render, and the glassy edge you hear is the track
rung through a 100 µs contact.

## the window

`cargo run --release -p kosm-view [-- --splash] [--frames=N]` opens the pool
(or the splash) in a window. The simulation runs on its own thread and hands
over a snapshot per frame; the window keeps every snapshot, so the timeline
is a recording: play, pause, scrub, follow live, and an inspector for the
frame under the cursor (melon state, fluid force, rings, drops, particles,
surface extents), with the melon annotated in the frame. Rendering is the
same CPU tracer the CLI uses, preview size while playing and full size when
paused; "export mp4" runs the CLI's render. First slice of the viewer plan:
recording-shaped now, wgpu live rendering and wasm next.

`cargo run --release -p kosm-view -- --ride <ride.json>` plays a recorded
rollout instead of simulating one — a Booster K1 riding a skatepark, say.
The file (`docs/ride-format.md`) carries its own meshes (binary STL, box,
sphere, cylinder), the fixed scenery, one actor per moving body and a pose
per actor per frame, so playback touches no solver: play, pause, scrub the
frame slider, pick a playback speed, and the camera orbits whatever actor
`track` names (drag to orbit, scroll to zoom, z up). Rendering is a plain
instanced rasterizer (`ride.rs`, `ride.wgsl`) — flat normals, one sun, a sky
ambient, no culling, because authored STLs are wound however they were wound.
`--shot=out.png --frame=N` draws one frame offscreen and quits, which is how
the mode is checked without a screen.

## building

`vcad` depends on a sibling `../tang` checkout and `phyz` on crates.io `tang`;
the workspace `[patch.crates-io]` unifies them on the checkout so `tang::Scalar`
is one trait across the graph. `vcad-kernel` is built with `no-builtin-font`
so it does not need vcad's `node_modules`.
