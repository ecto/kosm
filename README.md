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

### the court

`kosm-spike --court [frames]` is a gym: the authored scene
[`levels/court.loon`](levels/court.loon) (`court/`) has a maple slab, a
regulation hoop at one end — backboard, a rim of 24 rod segments, bracket, arm
and pole, all vcad geometry derived into colliders the same way the marble
track is — and basketballs that are phyz free bodies. Three are dropped from
6 ft; one is a free throw, released 15 ft from the board with the speed,
elevation and backspin the level says. The bounce is phyz's own restitution.
Each dropped ball's apexes are printed against Newton's `e²` law — with
`e = 0.77` the first apex is the NBA's inflation test (the top of the ball
must come back to 49–54 in), and every bounce lands within 0.7% of `e²`. The
shot is called: the step its centre passes down through the rim. The level's
default goes in at 1.02 s, off the glass.

The rim wears a net (`court/net.rs`): 12 strands over 7 rings in the diamond
mesh a real net is knotted into, hung from the rod as masses on damped
springs, substepped under the court's `dt` with a length pass so the cord
cannot stretch, and handed to the picture as one thin vcad cylinder per
segment. The balls push the net and the net does not push back yet; it costs
about 17 µs a step and `tests/net.rs` checks that it hangs to its cut length
and that the free throw still goes through it.
The room is authored too, because the picture has no textures: everything you
can see is a root with a material. The level draws the gym — four walls, a
ceiling, the floor outside the slab flush with the maple so a ball rolling off
does not step, a row of pads on the baseline wall, a band of windows high on
the long walls and a small stepped bleacher — and paints the court as thin
solids standing 1 mm proud of the slab: baseline, lane and key, the free-throw
line and its circle, and the three-point arc, the last two as patterns of
short bars the way the rim is a pattern of rod segments. Every line is
measured off the hoop knobs, so moving the backboard moves the court with it
and it stays regulation. The ball is two more roots at the origin, placed at
each ball's pose: a sphere, and its eight-panel seams as four thin rings.
Which of them the physics stands on is one list, `court::parts::collides` —
`ball`, `paint`, `key`, `window` and friends are appearance, and a root that
says nothing collides.
The shot has a gradient too. `court/aim.rs` is the marble's release-point hint
pointed at the hoop: the horizon `aim_t` is the moment the ball's centre falls
back through the rim plane — ballistic, 0.995 s, 995 steps — and
`J = |centre(T) − rim centre|²` is the miss there, squared. The adjoint problem
is the shot alone on the court (`Court::from_scene_shot_only`), one free body
against the same derived colliders. One backward pass gives `dJ/d(release
point)` and `dJ/d(release velocity)` together; on the 927 steps of free flight
before the ball first touches the rim, both agree with central differences to
1.3e-6 relative. Carried all the way to the horizon they do not: the ball is
rattling on the rod by then, and while `x` and `z` still match to every digit
printed, the `y` lane has the adjoint at −1.3e-9 (zero, by symmetry) against a
difference quotient of −14.2, because a micron of sideways nudge changes which
of the 24 rod segments the ball catches. The adjoint is right and the
difference quotient is not a derivative there — which is the whole reason to
have one.

The solve turns `dJ/dv₀` into the two knobs the level spells, `shot_speed` and
`shot_elev_deg`, and descends with backtracking. Elevation is stepped as
`θ · speed`, because those two columns of `∂v₀/∂(knobs)` are orthogonal and
that scaling makes them the same length, so the direction points at the answer
instead of down a valley. The level's own 7.40 m/s at 52° already goes in, but
its centre is 85 mm off the rim's middle at the horizon; two iterations take it
to 7.319 m/s at 51.77° and 23.5 mm, through at 1.00 s, and the knobs go back to
`out/hinted/court.loon`. Off target, `shot_speed 6.8` is a plain miss; the same
solve returns 7.128 m/s at 47.94° in two iterations and it drops at 0.86 s. The
whole thing is about 15 s, and `aim 0` in the level turns it off for the
render-only path. `tests/aim.rs` gates both halves.

The picture is `court/render.rs`, a path tracer: the same derived colliders
the physics stands on (the rim drawn as the torus its segments approximate),
the balls with their contact pose so the seams turn with the backspin, and a
gym the level describes — walls, a ceiling, rows of light panels that are the
only light there is. Next-event estimation on the panels, Russian roulette,
one sample stream per pixel and frame. Lacquered maple planks with the court
painted on them, a thin dielectric backboard with its square, painted steel,
pebbled rubber. 960×540 at 64 spp is about 4 s a frame on the CPU; the still
`out/court_still.png` is 1080p at 512 spp, taken at `still_t`. Every knob —
the balls, the shot, the hoop, the gym, the lights, the camera, the sample
counts — is a `defparam` in the level. `tests/court.rs` is the rulebook test
in the simulator.

The picture is vcad's. `court/render/` holds no renderer — it assembles a
scene for `vcad-kernel-raytrace`'s `pathtrace`, the same integrator
`vcad-render --photoreal` uses. The document's roots are walked to the placed
primitives they are unions of, not evaluated to one solid each: a union of
disjoint solids needs no boolean, so the court's markings are sixty instances
of a few cubes and the rim twenty-four of one bar, each primitive built once,
given one BVH over its *untessellated* BRep, and shared by every instance.
Only a genuinely boolean subtree still goes through `vcad-eval`. That took the
level's scene from 3.9 s of booleans to nothing measurable, and the markings —
which used to evaluate to a BRep-less mesh — are on the GPU tier at last. The
rim's silhouette is still the ring the CAD says it is at any resolution, not an
approximation the picture keeps separately from the one the physics stands
on. A root's material name resolves to a PBR through one table: the
document's own `[material ...]` first, then the court's names (`maple`,
`rim`, `ball`, `wall`), then vcad-render's built-in library. A root named
`ball` is not part of the court — it is the ball's own solid, placed at each
ball's pose from phyz, and without one the ball is `Solid::sphere` at the
level's radius. `Court::extras` is drawn the same way, for solids that move
and that the physics does not own. vcad is in millimetres and phyz is in
metres; the whole picture is built in millimetres and every phyz quantity
crosses that boundary once, in `Scene::at`.

The gym is still code: four walls, a ceiling, a floor beyond the slab, and the
rows of light panels that are the only light there is — but only until the
level grows roots with material `wall` or `ceiling`, after which the room is
the level's and only the panels stay. The camera is the level's `cam_*` knobs
with a real iris (`cam_aperture_mm`, `cam_focus_mm`), and `shutter` opens the
film for a fraction of a frame: `shutter_steps` sub-frames are rendered across
that span of physics steps and averaged, so a ball at 8 m/s smears the way it
does on film. 960×540 at 64 spp is about 23 s a frame on the CPU — five times
the hand-written tracer it replaces, which is what real BRep intersection and
next-event estimation on ten panels cost. The still `out/court_still.png` is
1080p at 512 spp, taken at `still_t`. Every knob — the balls, the shot, the
hoop, the gym, the lights, the camera, the sample counts, the denoiser — is a
`defparam` in the level. `tests/court.rs` is the rulebook test in the
simulator.

```bash
cargo run --release -p kosm-spike -- --court          # out/court.mp4, out/court_still.png
KOSM_SPP=4 cargo run --release -p kosm-spike -- --court 30   # a quick look
cargo run --release -p kosm-spike --example court_trace      # one ball, height and speed through each impact
diff levels/court.loon out/hinted/court.loon                 # the aimed shot, written back
```

The balls did not bounce until phyz did. Its soft contact is a resting-contact
model — an impedance that delivers the rebound target scaled by `d`, a margin
band that tapers `d` to nothing, and a stabilization push that adds `erp` to
the effective `e` — and its own benchmarks recorded restitution 8–19% short of
Newton with no rebound at all from 5 cm. A basketball at 6 m/s came back with
77% of its nominal `e`, and the third bounce, detected half a millimetre above
the floor, was swallowed whole. Fixed in the phyz worktree this depends on: an
impacting contact row is rigid (to a part in a thousand, so a box landing on
four corners stays conditioned) and bias-free, on the same smoothstep the
restitution ramp already uses, and restitution reads the approach speed at the
start of the step rather than off the free velocity with `g·dt` already in it
(that was `m·g·dt·|v|` of energy gained per bounce). The §6.2 drop-height
benchmark in `phyz/tests/contact_physics_benchmarks.rs` is now a Newton gate
rather than a guard on the measured shortfall.

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

`cargo run --release -p kosm-view` opens the court in a bare 3D viewport: a
winit window, a wgpu surface, and one RGBA8 image blitted across it by a
twenty-line shader. There is no UI — no panels, no text, no timeline widget,
no inspector. Anything on screen is the scene's own picture. Controls are the
keyboard and the mouse:

| | |
|---|---|
| drag | orbit |
| wheel | zoom |
| space | pause, and rejoin the simulation |
| ← → | step a frame while paused |
| Home, R | back to the level's camera |
| Escape | quit |

It opens live and stays live: the simulation steps on its own thread in
wall-clock time — fixed `dt` steps up to each frame's moment, capped so a late
frame never asks for the work of every frame it missed — and the window shows
the frame it is on. Space pauses; space again rejoins the simulation where it
has got to.

The picture is not written by the viewer. The level's roots are evaluated once
by `vcad-eval` into BRep solids, and every frame is that geometry with the
balls and the net where phyz has them, a material per root name, and the
ceiling panels as the only lights. The renderer is asked for what it can finish
in about 30 ms: the window's size over an integer divisor, one sample a pass,
the divisor retuned from the times measured.

Two tracers answer that. `vcad-kernel-raytrace`'s `gpu` feature used to be
unreachable here — it pinned wgpu 23 while the surface is on wgpu 30 — but vcad
is on wgpu 30 now (a worktree of it, `claude/wgpu-30`), so the compute tracer
runs on the window's own device. The court is *resident* on it: one
`ResidentScene`, packed once, where a frame rewrites only the placements that
moved and a pass rewrites only the camera and the render state. A pass used to
re-upload the whole court and rebuild every buffer and bind group; the 960×540
still went from 4.40 s for thirty-two passes to 1.39 s, and in the window,
where the surface and the tracer share a device and a queue, from two to four
*seconds* a pass to twenty-five milliseconds. `--cpu` picks the CPU
integrator, which is the reference and the fallback.

Neither tracer accumulates, and each tier now has its own accumulator. A pass
is one raw sample; what makes one sample a pixel watchable is refusing to throw
the last frame away. Every pixel keeps a running mean and a count, and when
something moves the renderer knows which something — the bounding sphere of
each ball and extra whose pose changed, at its old pose and its new, plus the
disc its shadow throws from each panel. That mask is geometric, so it reads the
same on both tiers and the walls accumulate for the whole run while the balls
bounce through them. It is computed before a ray is cast, from the poses alone.

On the CPU tier the accumulator is `history.rs`: `History::plan` hands the
renderer disjoint rectangles, `pathtrace::render_into` re-traces exactly those
into a film kept between passes, `History::merge` leaves every other pixel's
mean *and* its count alone, and a reprojection carries the picture through a
moved camera. A masked pass is taken only when it saves more than half the
frame — the pixels outside it get nothing, and a picture that is always masked
never converges.

A tuner step is not a new picture, and no longer costs one. The window buys a
pass that fits in thirty milliseconds with resolution, so the render size moves
under the accumulator's feet, and every step used to throw the whole history
away: the log showed the mean sample count fall from 599 to 26 on a 426→365
step, and the window went back to a blizzard for seconds at a time. The two
tiers answer that differently, because they have to. On the CPU tier
`History::resample` bilinearly carries every plane — the mean, the alpha, the
variance, the three guide planes — and the per-pixel counts with them, rounded,
because there is no such thing as 3.4 samples; the stored view is re-stated at
the new raster so the step does not read as a camera move. It costs a couple of
passes over a few hundred kilobytes and the picture goes straight on
converging. On the GPU tier there is nothing to resample on this side: the mean
and the count are in device buffers vcad reallocates on a resize, and reading
them back to resample them is the one thing that tier exists not to do. So it
takes the other design — **the size is free to move while it is cheap to move
it, and frozen once it is dear.** Under `RESIZE_UNTIL` samples a pixel the
tuner steps the size as it always did; past it the size is left alone and the
tuner spends its other knobs, the sample count and the scissor. A window's own
resize is exempt, because there the old picture is of a different window. And
it un-freezes itself: when the world starts moving the mask restarts pixels,
the mean falls back under the line, and the size is the tuner's again.

**The GPU tier does none of that on this side, and reads nothing back.** vcad's
`accumulate_and_denoise_resident` traces the sample, folds it into a mean and
count that live in device buffers, runs the à-trous filter against the resident
guide planes and tonemaps into a storage texture — on the viewport's own
device, which the blit samples directly (`viewport::Image` is bytes *or* a
texture now; the CPU tier still hands over bytes). What is left on this side is
the **keep mask**: one byte a pixel, 1 to go on accumulating and 0 to start
over, built by `history::Mask` from the same `mask_rects` the CPU tier plans
with, so the two tiers mask on one piece of geometry. Reprojection stays CPU-side
only, so a camera move uploads an all-restart mask and the GPU picture loses
its whole history where the CPU one keeps most of it.

The scissor is back on that tier. `set_scissor` used to size the *trace* alone
while vcad's accumulate pass walked every pixel of the frame, so outside the
rectangle it would have folded a stale raw sample in as a fresh one; the
accumulate pass honours the same rectangle now, leaving every pixel outside it
with the mean, the count and the variance it already had, and the resolve pass
still covers the frame so the target texture stays whole. So the GPU tier takes
the same bargain the CPU one does: when the keep mask's bounding box is worth
less than half the frame, the pass is that box and nothing else. It is armed
and it is rarely taken on *this* level: a ten-minute session repainted 17–33%
of the screen on the busy frames, but four balls spread across the court put
one box around all of them and that box is more than half the frame every time,
so every pass in the session was a full one. The rule is the same because the
reason is: outside the box no pixel gains a sample, and a picture
that is always scissored never converges.

That the GPU tier once had no cheap pass at all is why `Cost::terms` exists in
its present form. It refuses to call two buckets
two points until they are `SPREAD` apart: drifted together, they were fitting
470 ms a megapixel-sample against a real fifty and the window sat at 183×102
refusing to grow. And a pass now really is `spp` samples: vcad folds one sample
a call, so a pass of four is four calls, and the tuner's sample knob means
something on this tier for the first time.

The wait is the other honest thing. With nothing read back, nothing
synchronises the two sides, so the worker queued passes faster than the device
retired them and the tuner timed `queue.submit`. A pass ends on
`device.poll(wait)` — a fence, not a readback.

What that bought, at a 1280×720 window on a retina display (so 2560×1440
physical): the GPU tier used to settle at 320×180 with a pass of about 450 ms,
of which 420 ms was the CPU à-trous filter. It now climbs 256×144 → 284×160 →
365×205 in the first second, finds 365×205 costs 61 ms against a 30 ms budget,
steps back to 320×180 at 20–50 ms a pass — and then *stays* there for the rest
of the session, accumulating past 1700 samples a pixel without one collapse.
Before the resample and the freeze it lost the lot at every step. It does
**not** reach the full window: a
sample measured on this machine costs about 9 ms at 320×180 and 31 ms at
960×540, so 30 ms buys roughly half a megapixel and no more. The CPU tier is
unchanged, masked passes and all.

It is incomplete elsewhere too. The painted markings have no BRep to pack and
are CPU-only — the GPU court has no lines on its floor. The ball's seams *are*
on the GPU again: the shader used to trace a torus wide enough to engulf the
ball it was drawn on, and vcad's torus intersection is fixed.

The environment is no longer a difference between them. The GPU tier used to
scale the shader's analytic studio gradient by the level's `env_radiance`,
which is a different colour in every direction where the CPU's is that constant
flat; `GpuRenderState::set_gradient_env` now sends the very
`Environment::constant(env_radiance)` the CPU tier builds. It moved no pixel of
the 960×540 still, and neither does raising it a hundredfold: the gym is
closed, no ray reaches the environment on either tier, and at
`env_radiance = 0.05` under ten panels at 18 the environment is not what either
picture is made of.

The two pictures agree on brightness now, and the answer was one flag. The GPU
tier used to read 71 over a patch of the +y wall to the CPU's 105, and it was
neither the environment nor the path budget: `GpuRenderState` starts with
camera-visible lights *off*, and with them off the shader would not shade a
light the camera can see. The walls had always agreed to within a twentieth of
a per cent; the ten ceiling panels the whole gym is lit by came back black.
`set_camera_visible_lights(true)`, and the same for two other fields
`GpuRenderState::new` re-derives every time it is called — `max_depth`, which
it sets from the frame index, and `ground_enabled`, which the level does not
want because it authors its own floor — and over the 960×540 still at 32 passes
the two pictures read:

| patch | CPU | GPU |
|---|---|---|
| +y wall | luma 71.4 | 71.7 |
| ceiling panels | luma 110.6 | 110.9 |
| floor | luma 69.3 | 69.5 |
| whole frame | luma 85.17 | 84.99 |

— a fifth of a per cent apart over the frame, with a mean absolute per-pixel
difference of 1.8 of 255. The dark disc that used to sit on the +y wall at
x = 265, y = 270 — the camera's own retro-reflection point — is gone too; both
pictures now read 67.7 there, which vcad's own retro-incidence fix answered and
this worktree did not touch.

The one readback left in the GPU path is `--shot`'s: N passes through the
device history and one copy of the target texture out for the PNG. Nothing in
the window reads back at all. The deforming net still pays for a BVH build
inside every CPU pass. Evaluating the level takes **a minute or two** on this
machine — single-threaded, almost all of it in `propagate_boolean` sorting face
names under `circular_pattern` — and the window is black until it is done; it
says so on stderr while it works, and a `timeout 90` never gets past it.

`kosm-view --shot out/view_court.png` runs the same frame producer with no
window, which is how the picture is checked; it uses the GPU tracer unless
`--cpu` says otherwise. `--pool` and `--splash` are gone for now: they were
egui, and the pool's live tier went with it.

## building

`vcad` depends on a sibling `../tang` checkout and `phyz` on crates.io `tang`;
the workspace `[patch.crates-io]` unifies them on the checkout so `tang::Scalar`
is one trait across the graph. `vcad-kernel` is built with `no-builtin-font`
so it does not need vcad's `node_modules`.
