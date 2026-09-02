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

```bash
cargo run --release -p newt-spike            # or: newt-spike levels/other.loon
open out/track.svg out/frame_before.png out/frame_hint.png out/frame_tilted.png
diff levels/marble.loon out/solved/marble.loon   # the solved knobs, written back
```

The document is the only description of the geometry. `colliders.rs` walks
the vcad IR (`Union`, `Translate`, `Rotate`, `LinearPattern`,
`CircularPattern` over `Cube` / `Cylinder` / `Sphere`) into phyz colliders on one fixed track body, falls back to a convex
hull for anything else with a warning, and checks every collider's support
function against the tessellation before the first step. Plate, walls and cup
are all real colliders; the cup is a ring of box segments with a mouth facing
uphill so it catches.

Two phyz bugs surfaced and were fixed in the phyz worktree this depends on:
sphere-on-box contact points were taken from the box's degenerate face
support (the marble fell through the plate), and the body-body adjoint froze
a sphere's contact point in the sphere's frame and let the sphere own the
contact normal (gradients off by 100×). Tests: `phyz/tests/sphere_on_fixed_box.rs`,
`phyz-diff/tests/sphere_body_body_adjoint.rs`.

## building

`vcad` depends on a sibling `../tang` checkout and `phyz` on crates.io `tang`;
the workspace `[patch.crates-io]` unifies them on the checkout so `tang::Scalar`
is one trait across the graph. `vcad-kernel` is built with `no-builtin-font`
so it does not need vcad's `node_modules`.
