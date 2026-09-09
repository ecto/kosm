# kosm::player: a third-person controller any sim can drop in

Date: 2026-09-09. Status: design, agreed in conversation; the hero rig
(`sims/rune/hero/figure.rs`) and the render budget (`kosm_view::budget`)
are being built alongside it. Follows `2026-09-09-rune-cove-design.md`.

## The claim

Game feel is four things: the body accelerates the way a body does, the
camera breathes, the picture answers the hand inside a tenth of a second,
and the world reacts to being moved through. Engines fake all four with
curves and post-processing. kosm has physics for the first, a lens for the
second, a budget for the third, and materials for the fourth, so the
controller is three small types on the three nouns and nothing new.

What Bodycam does to a finished image, this does in the world: the fisheye
is a projection in the ray generator, the chromatic fringe is dispersion in
a lens element, the vignette is cos⁴, auto-exposure is a light meter with a
time constant, motion blur is the shutter integrating the trajectory the
history already holds, and the bob is a body on a spring behind the player.

## Placement

`crates/kosm/src/player/`: `Body` is a Step wrapper, `Rig` is a Lens,
`Drive` is what turns a window's input into an Action. `Budget` lives in
kosm-view because it is about passes and pixels, not the world. Rune's
`sims/rune/{being,game}.rs` become the first user and shrink to what is the
cove's own: the door, the gate, the glint.

## Body

An articulated rig on one free joint, from the hero's named pivots (hips,
knees, ankles, shoulders, elbows, neck, torso), each a phyz joint. What the
old capsule did with one spring, the rig does with the same spring on the
root plus PD targets on the joints.

- **Standing** is the upright spring on the root, critically damped from
  the body's own inertia, exactly as `being.rs` derives it. Feet are the
  capsule's contacts today; the boots become the contacts when the SDF
  contact path is asked for two spheres instead of one.
- **Lean to move.** Input sets a target lean, not a force. The upright
  controller drives the root to that lean and the ground reaction does the
  accelerating, so the figure tips into a start and back into a stop. The
  puzzle's tilt is an offset on the same lean.
- **Velocity, not bang-bang.** A proportional controller toward
  `speed × input` with τ_accel 0.25 s and τ_stop 0.35 s; Shift runs at
  2.6 m/s; a diagonal is not faster than a straight.
- **Gait.** Legs are pendulums driven by the root's velocity: a stride
  frequency from the leg length, a phase per leg, PD to the swing angle.
  No keyframes and no policy in this slice; a trained gait from kosm-train
  replaces the pendulums behind the same joint targets later.
- **The tool socket.** The hero's two-link IK gives the hand a target from
  a `Tool` pose the player aims with the mouse; a PD on shoulder and elbow
  follows it. The rune scorer reads the held refractor, not the body.
- **Ground.** A trait with three impls: an SDF map (the cove's bake), a
  plane, and phyz colliders, so a sim with no bake still has a floor.
- **Water.** Buoyancy, drag and the shore current from `being.rs`, as a
  `Medium` the Body asks the world for at its position.

## Rig

A Lens that yields a `Camera` per frame, and moves like a thing with mass.

- **Follow** with a spring-damper on the eye, about 0.15 s of lag, and a
  look-ahead that slides the aim a metre along the velocity.
- **FOV** widens a few degrees with speed.
- **Spring arm** from the SDF distance, so the eye never enters rock and
  never dips under the sea; the cove's doorstep framing is a special case
  of the same clamp.
- **The physical lens**, in order: projection (rectilinear or equidistant
  fisheye) in the ray generator; an exposure meter, a lens over the last
  frame's luminance with a time constant of a second, feeding the film;
  shutter time, which is how many passes of the history a frame integrates
  while moving. Dispersion in a lens element is a second pass and needs a
  glass element in front of the sensor.

## Drive

The input mapping `game.rs` already gets right: held directions are forces
and are not divided by the frame's steps; turns are quantities and are.
Yaw and lean sensitivities, run and interact keys, cursor capture. It emits
an `Action` so a policy can drive the same Body.

## Budget (kosm-view)

Pretty when we can, otherwise fast: the resolution scale each pass from the
measured pass time and whether anything moved. Still means full size, or a
prettier size when the pass fits under a ceiling; moving means the largest
scale under a 40 ms target with hysteresis; the history resamples across
the change and keeps what it had.

## Tests

- Body: standing drift, shove recovery, walk speed and its curves (rise
  and stop times within 10% of τ), the diagonal cap, lean into a start,
  gait phase locked to speed, the arm reaching a target within 5 cm.
- Rig: the eye never behind rock in a sweep of poses, lag and look-ahead
  measured on a scripted walk, exposure settling in a second when the sun
  is walked out of.
- Budget: synthetic timings, and a headless measurement moving vs still.
- Rune: the door test and the solvability sweep unchanged with the hero
  holding the lens.

## Order

1. Budget (in flight), 2. Body on the hero rig with the pendulum gait,
3. Rig with the follow camera and spring arm, 4. Drive and the rune
migration, 5. the physical lens: projection, meter, shutter.
