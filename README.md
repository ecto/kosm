# newt

A game engine where the level is a CAD file, the physics is the robot's
simulator, and every knob has a gradient. Built on
[vcad](https://github.com/ecto/vcad) (geometry), [phyz](https://github.com/ecto/phyz)
(differentiable multi-physics), [tang](https://github.com/ecto/tang) (one scalar,
one IR) and loon (scripting).

## the marble (`crates/newt-spike`)

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
cargo run --release -p newt-spike            # or: newt-spike levels/other.loon
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
cargo run --release -p newt-spike levels/marble-cup.loon
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

## building

`vcad` depends on a sibling `../tang` checkout and `phyz` on crates.io `tang`;
the workspace `[patch.crates-io]` unifies them on the checkout so `tang::Scalar`
is one trait across the graph. `vcad-kernel` is built with `no-builtin-font`
so it does not need vcad's `node_modules`.
