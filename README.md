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

The *beauty* pass is no longer that caster, and no longer a rasteriser either.
`out/frame_before.png`, `frame_hint.png` and `frame_tilted.png` are
`kosm-render`'s path tracer — multiple bounces, MIS against three softboxes, a
real camera — pointed at the level through `analytic.rs`, which implements
`kosm_render::Geometry` over the derived colliders themselves: oriented boxes,
spheres and the cup's cylinders, intersected analytically, so the marble is a
sphere at any zoom and its shadow is traced rather than composited. Two
renderers, one geometry, each doing what it is good at: a Monte Carlo estimator
has no useful dual, and the caster that does stays exactly where it was.

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
does not step, a row of pads on the baseline wall, a band of clerestory openings high on
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
painted on them, a real glass backboard — full transmission at IOR 1.52, with
float glass's faint iron green as Beer–Lambert absorption, so the ring's shadow
and the wall behind read through it — painted steel, pebbled rubber. 960×540 at 64 spp is about 4 s a frame on the CPU; the still
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

The gym has daylight. `sky 1` in the level swaps the constant grey outside the
room for a sky gradient (`sky_zenith`, `sky_horizon`) and hangs a sun disc at
`sun_elevation_deg` / `sun_azimuth_deg` with `sun_irradiance` on a surface
square-on and `sun_angular_radius_deg` across — 35°, 250°, 6 and 0.27° by
default, which is about six times what the ceiling panels put on the floor:
bright enough to be the light in the room, low enough that the patches hold
their colour instead of clipping. The room is closed solids, so
the only way any of it gets in is the clerestory band on the long walls, and
what the picture shows is eight slabs of afternoon sun thrown across the floor
and up the far wall, with the panels' own even light underneath. `sky 0` is
the old still exactly: `Environment::constant(env_radiance)` and no sun. Both
tiers read the same two values off `render::Scene` — the CPU integrator takes
the `Environment` and the `Sun` directly, the GPU tier uploads them through
`set_gradient_env` and `set_sun` — so neither can be lit differently from the
other.

The band is a real opening and not a pane of glass, which is a renderer's
limit showing through the level. Next-event estimation is an any-hit shadow
ray: anything it touches occludes, a thin dielectric included. With glass in
the window band the sun can only be found by a BSDF-sampled path that refracts
through a pane and then wanders into a 0.27° cone, which is roughly one ray in
a hundred thousand — 32 passes of that is salt and pepper, not daylight. With
the band open, NEE sees the sun directly and the patches are clean in a
handful of passes. Real gyms have glass; *NEE through thin dielectrics* is the
kosm-render item that would let the panes come back, and the `window` material
is still in the table waiting for it.

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

One box was not enough. The change rects used to be reduced to a single
bounding box, and with four balls spread across the court that box is most of
the frame: it never cleared the half-frame bar, so every pass went full. They
are clustered into at most **four** boxes now — greedily, merging the pair
whose union adds the least area until the count fits, and merging a pair whose
union adds nothing even when it already does. The cover is re-merged after
every step, so what comes out is always disjoint and both tiers can add the
areas up. Each box is its own scissored dispatch on the GPU tier and its own
rectangle in `render_into` on the CPU tier, and the log names them: `14 ms a
2-box pass (17%)` against `32 ms a full pass` at the same size, or `19 ms a
1-box pass (1%)` against `87 ms` at four samples. A camera move still takes a
full pass — the reprojection needs this pass's depth everywhere — and the pass
after it is boxed again.

The budget is four because a dispatch is not free. vcad scissors the *trace*,
but the reproject, accumulate, demodulate, à-trous and resolve passes behind it
are still dispatched over the whole frame, and the keep mask is re-uploaded
with them, so past a handful of boxes that fixed part outgrows the rays a
tighter cover saves. Splitting the fold from the filter — an
`accumulate_resident(scissor)` per box and one `denoise_and_resolve` for the
frame — is what would make many boxes cheap, and it is kosm-render's to give:
`HistoryPipeline`'s five passes are private and only the fused
`accumulate_and_denoise_resident` is public.

And the tuner keeps two clocks. A boxed pass can be four times cheaper than a
full one, and a climb rule reading that cheapness buys a picture the next full
pass cannot afford — so the *size* is decided from the full-pass time, and only
full passes feed the cost model the size is chosen from, while the *sample
count* is decided from the masked one.

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
with, so the two tiers mask on one piece of geometry.

Reprojection is no longer CPU-side only, and a camera move no longer costs the
GPU picture its history. vcad grew
`accumulate_and_denoise_resident_reprojected`, which takes the *previous*
pass's camera: each pixel is unprojected through this pass's depth, projected
back into that view, and keeps the mean and count it finds where the surfaces
agree. So a moved camera now uploads the mask a *still* camera would have got —
only the rectangles the world moved under — and lets the device settle the
rest; only disocclusions restart. `--orbit-test` is that claim, scripted:
converge headlessly, swing the eye three degrees about its target, take one
more pass and ask the device's own counts what survived. At 320×180 after eight
passes, **88.8% of the frame kept its history across the move**, where before it
was none of it. The previous camera is offered only when it is worth offering:
a still-camera pass passes `None` (a view reprojected onto itself is two
dispatches for nothing) and so does the first pass at a new size, since vcad
reallocates the history on a resize and there is no previous depth plane to
test against. A reprojected pass takes no scissor either — the reprojection
needs this pass's depth everywhere — and the window's log names it (`a full
reprojected pass`).

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

## the light (`crates/kosm-render`)

The renderer moved. It used to live in vcad, as `vcad-kernel-raytrace`: a ray,
a BVH over B-rep faces, a TLAS over placed instances, and a path tracer on
top. All of that was written against `BRepSolid`, which meant the only thing
in this workspace that could be *lit* was a CAD document.

But an engine owns its renderer. Kosm has three kinds of geometry already — a
vcad solid's analytic faces, a phyz collider, and (soon) a splat — and they
all want the same light. So the acceleration structures and the integrator are
now `crates/kosm-render`, generic over one trait:

```rust
fn len(&self) -> usize;
fn bounds(&self, i: usize) -> Aabb;
fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit>;
```

How many primitives you have, where each one is, and what a ray finds when it
meets one. That is the whole seam. A `Hit` carries the primitive index and
nothing that names it, so the geometry — not the renderer — says whether index
7 is a `FaceId`, a triangle or a gaussian.

The integrator went with them. `Scene`, `Object`, `Pbr`, the softboxes, the
environment, the film, the à-trous denoiser and `render` never knew what a
B-rep was — the one line that did, reading a `vcad_ir::MaterialDef` into a
`Pbr`, stayed in vcad as `pathtrace::from_material_def`.

The geometry stayed with the geometry too. `intersect/` (plane, cylinder,
sphere, cone, torus, bilinear, B-spline) and `trim.rs` are still vcad's,
because knowing that a ray-sphere hit at *(u, v)* falls outside a trimmed
face's boundary loop is a B-rep fact, not a lighting one.
Kosm's own tracers are clients too. `kosm-spike/src/analytic.rs` implements the
trait over the marble level's phyz colliders — box, sphere, cylinder — so the
marble's beauty pass is the same integrator the court's is. And Snell, Fresnel
and Sellmeier moved *into* the renderer as `kosm_render::optics`, generic over
`tang::Scalar` (they are laws at an interface, not facts about one level's
glass): `glass.rs`, `light.rs` and the pool's water surface all read the one
copy, and on `Dual` it is still its own derivative.

`vcad-kernel-raytrace` implements the trait over its faces, and `Bvh`, `Tlas`,
`Instance`, `Object` and `Scene` are aliases for the generic ones with vcad's
geometry filled in — so `vcad-render --photoreal`, the GPU upload and the
window all draw the same picture through the same code. Its own suite (92
tests), `vcad-render`'s (114) and the 42 that came with the renderer all
pass, and the 960×540 court still is the same one, to within the low bit of
float noise two builds of the same source already differ by.

### the material

The BSDF is Disney's "principled" parameterisation (Burley 2012) with the
corrections the field settled on afterwards, composed the way OpenPBR 1.0
composes them. One `Pbr` struct, one `bsdf_eval`, one `bsdf_sample`, in Rust
for the CPU renderer and in WGSL for the GPU one, checked against each other on
real hardware.

Four lobes, layered coat → sheen → (specular + diffuse):

- **Diffuse — EON.** Not Lambert. d'Eon, Portsmouth, Hill and Fascione's
  energy-preserving Oren-Nayar (JCGT 14(1), 2025), which is the Fujii
  Oren-Nayar single-scattering term plus a closed-form multiple-scattering
  compensation. Oren-Nayar is what makes chalk, latex paint and unfinished
  concrete read as those things rather than as shiny plastic dimmed; the
  compensation is what makes it safe to leave on, since plain Oren-Nayar loses
  up to a fifth of the light it is given. `diffuse_roughness` (OpenPBR's
  `base_diffuse_roughness`) drives it and is separate from the specular
  roughness, because a surface's slope statistics and its subsurface scattering
  length are unrelated facts. `subsurface` blends towards Disney's
  Hanrahan-Krueger lobe for the short-mean-free-path look of rubber and skin.
- **Specular — compensated anisotropic GGX.** VNDF sampling as before, with
  Turquin's (2019) multiple-scattering compensation on top: scale the
  single-scattering lobe by `1 + F0·(1 - E)/E`, where `E(mu, alpha)` is the
  lobe's own `F = 1` directional albedo, baked into a 32×32 table in
  `tables.rs`. That table is not a textbook GGX — it is a stratified VNDF
  estimate of `E[G2/G1]` for exactly the `d_ggx` and `v_smith` in this file, so
  the compensation compensates *this* renderer.
- **Sheen — LTC.** Zeltner, Burley and Chiang's "Practical Multiple-Scattering
  Sheen Using Linearly Transformed Cosines" (SIGGRAPH 2022), their published
  `approx` fit transcribed from the authors' pbrt-v3 reference. It is a real
  fit to multiple scattering in a layer of normally oriented fibres, not
  Disney's 2012 Schlick-weighted tint, and being an LTC it evaluates,
  integrates and importance-samples in closed form. OpenPBR calls this layer
  *fuzz*; the parameters here are `sheen`, `sheen_color` and `sheen_roughness`,
  which are OpenPBR's `fuzz_weight` / `fuzz_color` / `fuzz_roughness` under
  more familiar names.
- **Coat — GTR1.** Disney's clearcoat distribution (γ = 1, longer-tailed than
  GGX) at a fixed IOR of 1.5, layering onto everything beneath with `1 - F`
  attenuation.

Two parameters describe the same number, so the precedence is stated rather
than left to chance: `ior` wins whenever it is not the default 1.5, and
otherwise `specular` drives `F0` through Disney's `F0 = 0.08 · specular`. The
two agree exactly at the defaults (both 0.04), so the rule has no seam where it
switches. `clearcoat_roughness` is Disney's `clearcoatGloss` respelled the way
the rest of the struct spells roughness: Disney maps gloss onto GTR1's alpha as
`mix(0.1, 0.001, gloss)`, so their satin end is `clearcoat_roughness ≈ 0.32`
and their gloss end `≈ 0.032`.

Every parameter added on top of the old metallic-roughness set defaults to the
value that reduces the model to what it was — `diffuse_roughness = 0` *is*
Lambert, `sheen = 0` and `subsurface = 0` switch their lobes off outright — so
a material written before any of them exists renders as it did.

What the tests pin:

| test | what it holds |
|---|---|
| `the_eon_diffuse_lobe_passes_a_white_furnace` | a white diffuse surface reflects ≥ 0.98 and ≤ 1 at every roughness and every incidence angle |
| `the_eon_diffuse_lobe_is_reciprocal` | `f(wo, wi) == f(wi, wo)` to 1e-6 |
| `a_rough_metal_furnace_closes_to_one_percent` | `F0 = 1` GGX returns 1 ± 1% at α = 0.2, 0.5, 1.0 — uncompensated it keeps 0.947, 0.687 and 0.307 |
| `a_dielectric_specular_lobe_stays_under_one` | compensation is a correction, not a licence to make light |
| `the_sheen_lobe_is_bounded_and_brightens_at_grazing` | sheen albedo in [0, 1], and grazing at least 1.2× normal |
| `the_layered_bsdf_is_reciprocal` | the whole stack, coat attenuation and sheen albedo-scaling included, to 1% |
| `sampling_every_lobe_recovers_the_evaluated_albedo` | `E[f/pdf]` lands on the evaluator's own quadrature for every lobe |
| `tests/gpu_bsdf.rs` | the WGSL port agrees with the Rust across a 1728-case parameter sweep (worst relative disagreement 6e-7), the device's sample PDF equals its eval PDF, and the furnace closes on the GPU's own table interpolation |

#### transmission and dispersion

Glass is a fifth lobe: a **rough dielectric** (Walter et al. 2007) sharing the
specular lobe's alpha and its VNDF sampling, with the reflect/transmit split
taken from the *exact* unpolarised Fresnel equations rather than from Schlick.
That choice is not fussiness — at a glass/air interface Schlick's error near
grazing is the difference between a rim that lights up and one that does not,
and total internal reflection has to be what the same formula returns past the
critical angle rather than a branch bolted on beside it.

The parameters, all additive and all defaulting to the old behaviour:

| parameter | meaning |
|---|---|
| `transmission` | 0..1, OpenPBR's `transmission_weight`. Takes the diffuse and opaque-specular lobes away as it goes, so 1 is pure glass |
| `ior` | already there; now it is a real index and not only an `F0` |
| `abbe` | `V_d = (n_d − 1)/(n_F − n_C)`. 0 is no dispersion. Reconstructs a one-term Cauchy `n(λ) = A + B/λ²` — OpenPBR's `specular_ior_dispersion` |
| `sellmeier` | `Option<([f64;3], [f64;3])>`, a real glass's datasheet. Overrides `abbe`. `spectrum::BK7_SELLMEIER` is the pair the marble's own caustic tracer uses |
| `attenuation_color` / `attenuation_distance` | Beer–Lambert inside the medium: `σ = ln(1/color)/distance`, applied over each interior segment |
| `thin_walled` | a sheet with no interior — refract in and straight back out, roughness kept, no lateral offset and no absorption. What a window pane modelled as a 20 mm box actually wants |

**Conventions, because they are the part that goes silently wrong.** `eta` is
always `n_transmitted / n_incident`, and the integrator computes it from which
side of the *geometric* normal the ray arrived on, read before the face-forward
destroys the distinction — so a non-nested solid needs no medium stack at all,
just one slot recording which material the path is currently inside. Radiance
carries the camera-path `1/η²` scaling (pbrt's "radiance mode"), which cancels
Walter's `η_t²` out of the expression entirely; the consequence to hold onto is
that a *single* interface therefore does not have unit throughput, and it is
the round trip in and out that is the no-op. Both halves are tested separately.

**Dispersion is one wavelength per path.** A path that meets a dispersive
material draws a hero wavelength uniformly over 380–780 nm and multiplies its
RGB throughput, once, by the linear-sRGB colour-matching response normalised so
that `∫ r̄(λ) dλ = (1,1,1)` — Wyman/Sloan/Shirley's multi-lobe Gaussian fit to
CIE 1931, twenty lines and no tables, so it ports to WGSL verbatim. Because of
that normalisation the spectral estimator has the same mean as the RGB one it
replaced: dispersion only shows up where the geometry downstream genuinely
depends on λ. A path that meets its first dispersive surface at bounce three
adopts λ there. **A path that never meets one stays RGB, draws no extra random
number, and is bit-identical to what it was** — which is what keeps every
existing scene where it was. The cost is noise: one hero wavelength, not four
with spectral MIS, and the sRGB response has negative lobes, so a single sample
can land negative in a channel. The mean is right; the variance is the price.

**NEE does not go through glass.** `occluded()` treats a transmissive surface
as an opaque blocker, which is the standard choice — a shadow ray has no way to
find the bent path a refraction would have taken, and pretending otherwise adds
bias, not caustics. So caustics here come only from BSDF-sampled paths that
happen to land on an emitter, and they converge slowly. The MIS weights stay
consistent because both strategies agree that the light was not reachable: the
NEE sample returns zero and the BSDF path carries the full contribution.

What the tests pin, on top of the table above:

| test | what it holds |
|---|---|
| `dielectric_fresnel_at_normal_incidence_is_the_textbook_number` | `((n−1)/(n+1))²` to 1e-6 at n = 1.33, 1.5, 1.52, 1.9, 2.42 |
| `brewsters_angle_halves_the_unpolarised_reflectance` | at `θ_B = atan(n)` the p-polarised term vanishes, so the average is exactly `R_s/2` |
| `past_the_critical_angle_everything_reflects` | TIR is the formula's own answer, not a guard |
| `a_smooth_slab_refracts_at_snells_angle` | sampled through the whole lobe — VNDF facet, Fresnel branch, Walter half-vector — `sin θ_t` matches `sin θ_i / n` to 5e-3 at 10°, 30°, 50°, 70° |
| `a_rough_glass_sphere_closes_the_furnace` | reflection + transmission in [0.97, 1] at roughness 0.05–0.3 and three incidences, with the η² transport scaling taken back out |
| `a_slab_is_a_round_trip_no_op` | in and back out returns unit throughput — the other half of the η² convention |
| `absorption_is_beer_lambert_to_the_letter` | `exp(−σt) == color^(t/d)` to 1e-6, and one attenuation distance reproduces the colour that named it |
| `a_bk7_prism_spreads_f_to_c_by_the_analytic_angle` | a 60° N-BK7 prism at 45° incidence fans the F (486 nm) and C (656 nm) lines by 0.75°, and the traced spread matches the two-surface analytic `i₁ + i₂ − A` to 2e-3 rad |
| `the_hero_weights_run_red_to_violet` | 650 nm reads red-dominant, 540 green, 450 blue — the fan comes out in the right order |
| `hero_weight_integrates_to_white` | the normalisation constants, re-derived by quadrature, to 2e-3 |
| `the_dielectric_sample_pdf_matches_its_eval_pdf` | the MIS invariant on *both* sides of the surface, thin-walled included |
| `tests/gpu_bsdf.rs` | the sweep now carries five transmissive materials and sweeps `wi` below the surface too; `gpu_dispersion_matches_the_cpu_reference` pins the device's Sellmeier, Cauchy and CIE fit against the CPU's |

What is *still* not here: **iridescence** (thin films), **BSSRDF** —
`subsurface` is Disney's local approximation, which will not bleed light into a
shadow or through a thin part — **nested dielectrics** (one medium slot, so a
bubble inside glass gets the outer medium wrong), and **spectral MIS** (the
hero wavelength has no companion wavelengths).

### the GPU tier

The compute pipeline moved too, behind `--features gpu`. Same split, one level
down: `kosm-render` owns the *light* — the device, the BSDF, the environment,
the camera and render state, the accumulator, the per-pixel history and the
denoiser — and a client owns the *geometry*, in two halves.

The WGSL half. A geometry module is a string of WGSL, composed between the
renderer's prelude and its integrator, defining five functions:

```wgsl
fn trace_scene(origin: vec3<f32>, dir: vec3<f32>) -> RayHit
fn hit_normal(hit: RayHit) -> vec3<f32>
fn hit_tangent(hit: RayHit) -> vec3<f32>
fn hit_material_index(hit: RayHit) -> u32
fn hit_orientation(hit: RayHit) -> u32
```

`RayHit`, `MAX_T`, `EPSILON`, the two `face_idx` sentinels, the scale-aware
`ray_eps`, `intersect_aabb` and `shading_frame` come from the prelude; `PI`,
`GpuMaterial` and `onb` from the BSDF. `trace_scene` returns `FACE_IDX_MISS` on
a miss and must never return `FACE_IDX_GROUND`, which the integrator keeps for
its implicit ground plane. `hit_tangent` may return the zero vector where the
parameterisation is degenerate — the anisotropy just goes round.

The binding half, which is where the interesting constraint is. A browser
guarantees only **ten** storage buffers per compute stage. The renderer takes
five and leaves five:

| binding | owner | what |
|---|---|---|
| 0 | renderer | camera uniform |
| **1..=5** | **client** | geometry slabs, read-only storage |
| 6 | renderer | output storage texture |
| 7 | renderer | render-state uniform |
| 8 | renderer | accumulation buffer |
| 9 | renderer | materials |
| 10 | renderer | depth/normal + denoise guide planes |
| 11 | renderer | area lights |
| 12 | renderer | feature ids (analytic crease detection) |
| 13, 14 | renderer | environment *textures* |

The environment is a pair of textures rather than buffers for exactly this
reason: there were no storage slots left. Native Metal allows far more, so a
sixth client buffer passes every test on this machine and dies in Chrome with
an invalid bind-group layout, a valid pipeline and a blank viewport. That is
why `RayTracePipeline::new` rejects a module with more than five bindings and
why `the_renderer_leaves_five_storage_bindings_for_geometry` asserts the sum,
not either half.

The Rust half is two types: a `GeometryModule` (the WGSL plus the layout
entries, fixed for the pipeline's life) and a `GpuGeometry` implementation that
hands over the packed bytes as `GeometrySlab`s per scene. `SceneRef` carries
those alongside the renderer's own materials, lights and environment; a client
writes `impl From<&MyScene> for SceneRef` and passes `&my_scene` everywhere.

`kosm-render` ships one worked example, `AnalyticGeometry` — spheres and
planes, one storage buffer, ~140 lines of WGSL — so the renderer's own GPU
tests trace something without a client crate:

```bash
cargo test -p kosm-render --features gpu -- --ignored --test-threads=1
```

`vcad-kernel-raytrace::gpu` is now the B-rep adapter over that seam: it packs
faces, surfaces, the BVH, trim loops and inner-loop descriptors into the five
slabs, supplies `brep.wgsl`, and re-exports everything else so
`vcad-kernel-wasm`, `vcad-ffi` and `kosm-view` see one API. The device moved
with the renderer, and `vcad-kernel-gpu` re-exports it, so there is still
exactly one `GpuContext` in the graph. The 960×540 court still is byte-identical
across the move.

### the sky, and the sun

`kosm_render::env` builds environments the renderer needs no asset for. Three
studio HDRIs — `studio`, `softbox`, `overcast` — are *synthesised* rather than
vendored: no binary blobs in the repo, no third-party licence to track, and
maps that are exactly as high-frequency as the importance sampler needs to be
exercised. Real HDRIs load through `env::parse_hdr`, a hand-written Radiance
RGBE decoder (both scanline encodings), because `image` is not on the
dependency list and the format is two hundred lines. Both came out of
`vcad-render::envmap`, which keeps only the CLI shape of `--env` — parsing
`gradient`, a name or a path is a fact about that binary and nothing else's.

`Scene::sun` is a directional light of finite angular size. An `AreaLight`
cannot express one (its solid angle falls off with distance) and the analytic
gradient has no disc in it at all, so daylight through a window had nothing to
come from. It is stated as **irradiance**, not radiance, so widening
`angular_radius` softens the shadow terminator without changing the exposure.
It joins MIS as a strategy of its own on both tiers: NEE samples the cone
uniformly, a BSDF ray that escapes into it picks the same radiance up under
the balance heuristic. A 0.5° disc is found by BSDF sampling roughly once in
fifty thousand rays, which is the whole reason it needs one.
`tests/gpu_sun.rs` holds both tiers to `E·cos(theta)` on a Lambertian plane —
measured 0.00–0.02% off the analytic answer at 0°, 30° and 60°.

### the denoiser, and what it now refuses to spend

The à-trous filter is Dammertz, SVGF-flavoured: 5x5 B3-spline taps at a
doubling stride, edge-stopped on normal, relative depth and *demodulated
luminance*, with the luminance tolerance scaled by the estimator's own
variance — 3x3-prefiltered, because a noisy error bar makes the weight jitter
between "trust" and "reject" pixel to pixel. Radiance is divided by albedo
going in and multiplied back coming out, so a part's colour is never blurred
into its neighbour's. The device port in `history.wgsl` is the same filter
weight for weight, and `tests/gpu_denoise.rs` pins it over `AnalyticGeometry`:
with a one-sample history the two frames come out **byte-identical**.

What is new is that the filter's *cost* falls as a pixel converges, not just
its strength. `resolve` always faded the filtered result out against the
temporal mean, but a faded filter costs exactly what a full one does, and the
widest iteration — stride 16, a 65-pixel footprint — is 25 scattered taps.
Now the per-pixel iteration *budget* falls too (`gpu::atrous_iters_for`, and
its mirror in the shader), full at a single sample so parity survives, none
at `count_cutoff`. RMSE against a 512-sample reference, in 8-bit codes:

| samples | raw | denoised | iterations/pixel |
|---|---|---|---|
| 1 | 46.25 | 7.62 | 5 |
| 4 | 19.14 | 5.66 | 5 |
| 16 | 8.43 | 5.10 | 3 |

At 16 samples that is 3 iterations where it used to run 5 — 40% of the filter
gone — for 0.09 of one 8-bit code. The 1- and 4-sample rows are unchanged to
the digit, because at those counts the budget is still full.

### the pixel filter

Primary rays were jittered uniformly inside the pixel. That is a box filter —
the worst reconstruction filter there is — and it is why a thin bright feature
against a dark background stairsteps however many samples it gets: the rim's
ellipse, the net's cords.

The usual fix carries a per-pixel weight sum, which is a second buffer and a
different accumulation rule on both tiers. It does not have to. Importance-
sample the *kernel* — draw the sample position from the filter — and the plain
mean of the samples already is the filtered estimate. Nothing about the
accumulation, the device history's running mean, or the variance estimator
changes; only where in the pixel a ray is aimed.

`PathTraceOptions::filter` takes `PixelFilter::Box` (the default, and
bit-identical to the old jitter — its warp is `u - 0.5` and the sample
position was `u`), `Gaussian` (sigma 0.4 px) or `BlackmanHarris`, both over a
1.5-pixel support. Both invert their own CDF by bisection; the Gaussian's goes
through `erf` and Blackman-Harris integrates to a sum of sines, so there is no
table to keep in sync.

The GPU needs no shader change at all. Its sub-pixel jitter is a Halton pair
computed on the *host*, so `GpuRenderState::set_pixel_filter` warps two
numbers with the same `PixelFilter::warp` the CPU integrator calls before they
are uploaded. There is no filter code in WGSL and no way for the two tiers to
drift — `tests/pixel_filter.rs` pins them to the same function, and pins a
converged edge to the kernel's analytic footprint.

`kosm-view --shot out/view_court_gpu.png --filter blackman` shows it on the
court: the rim's ellipse stops breaking up, the net's cords read as lines
rather than dots, and the ball's seams lose their jaggies. Without the flag
the shot is byte-identical to the one that was there before.

### two more geometries: water, and a scan

The seam earns its keep when something implements it that the renderer was
never designed around. Two now do, and neither needed a line of the
integrator.

`HeightField` is a regular `nx × ny` lattice of heights over the `xy` plane —
the pool's free surface, or terrain. Its primitive is a **cell**, and a cell
is a bilinear patch rather than two triangles. The patch is the right call
twice over: the ray–patch equation is a plain quadratic in `t` (surface and
ray are both linear in `x` and `y`, so their difference is degree two), so a
hit point lands on the interpolated surface to machine precision instead of to
a triangulation's chord error; and `kosm-spike`'s pool already marches
`p.z - surface.height(p.x, p.y)` against a *bilinear* sample of that same
grid, so triangles would have had the renderer and the simulator looking at
two different sheets of water.

Water moves, though, and a hierarchy over 65 025 cells is not something to
rebuild sixty times a second. It does not have to be. The tree's *topology* —
which cell sits in which leaf — is a partition of indices over a lattice that
never changes; only the boxes move, and only in `z`. So `Bvh::refit` walks the
existing tree bottom-up recomputing bounds, no sort, no SAH, no allocation,
and the frame loop becomes `update_heights` then `refit`. On a 256×256 grid
that is **0.79 ms against 81.8 ms for a rebuild — 104×**. There is also a fast
path that skips the tree entirely: `intersect_march` runs a 2D DDA over the
lattice, testing patches in increasing `t`, which is what a camera ray coming
down at the water wants.

`Splats` is a cloud of anisotropic Gaussians — a 3DGS scan, the shape
`tang-3dgs` trains and `ipse-map` will carry. A Gaussian is not a surface, so
"the hit" has to be defined rather than found: `intersect` reports the
**maximum-response point**, the `t` where the density along the ray peaks,
which is closed form for a quadratic exponent (`t* = -dᵀAΔ / dᵀAd`). The
response there, `α·exp(-½d²)`, is exactly the alpha `tang-3dgs`'s rasteriser
computes per pixel, only evaluated in 3D against the ray; it rides back in the
`Hit`'s payload word alongside the splat index, so a compositor does not
recompute it. Bounds are the 3σ box and the intersector culls at the same 3σ,
so the tree and the test agree on what exists. Colour stays a call —
`Splats::colour(i, dir)`, spherical harmonics to degree 3 — because it needs
the view direction, which a `Hit` does not carry. 100k random splats build in
148 ms and trace in **4.3 µs/ray**.

#### what a client still cannot do

**Composite a splat cloud.** The geometry is there and `Bvh::trace` hands back
every splat along a ray sorted by `t`, which is precisely the front-to-back
order compositing needs — but the integrator has no path that consumes it. The
missing piece is the walk `C += T·α·c; T *= (1 - α)`, breaking when `T` falls
under ~1e-4, with the result treated as an **emissive backdrop** rather than a
BSDF: a scanned cloud has its lighting baked in already, and shading it again
double-counts. Mixing with analytic geometry is that same walk with one bound
— find the nearest opaque analytic hit first, composite splats only in front
of it, then add `T · L_analytic`. Until that lands, a `Splats` in a `Scene`
traces correctly and shades wrongly, so put one in a scene only to measure it.

The rule that keeps it honest: **`kosm-render` depends on `tang`, `rayon`,
`wgpu`, `bytemuck` and `pollster` — never on `vcad-*`, `phyz-*`, or any other
Kosm crate.** It is a leaf. And it must compile for the browser, GPU tier
included:

```bash
cargo check -p kosm-render --target wasm32-unknown-unknown --features gpu
```

which is why `rayon` and `pollster` are `cfg(not(target_arch = "wasm32"))`
dependencies rather than hard ones, and why the wasm buffer-mapping path uses
`js-sys` and `wasm-bindgen-futures` directly instead of logging through
`web-sys`. A renderer that cannot run where the picture is looked at is half a
renderer.

### using kosm-render from another crate

It carries its own package metadata (`0.1.0`, MIT, a README, keywords) and
`cargo package -p kosm-render` is clean, so it is a crates.io crate that has
not been published yet. Until it is, a consumer depends on it by path — but
declares the version it will want anyway:

```toml
[workspace.dependencies]
kosm-render = { path = "/path/to/kosm/crates/kosm-render", version = "0.1.0" }
```

Cargo uses the path locally and embeds the version when the consumer is
packaged, so publishing kosm-render means deleting one `path =` in one place.
That is how vcad does it: the workspace holds the path, and
`vcad-kernel-raytrace` and `vcad-kernel-gpu` say `kosm-render.workspace = true`.

Two rules come with the dependency. It is a leaf — `tang`, `rayon`, `wgpu`,
`bytemuck`, `pollster` and nothing else — so a consumer never expects it to
reach back for a `vcad-*` or `phyz-*` type; the geometry goes *in*, through
`Geometry`. And it builds for `wasm32-unknown-unknown` with `--features gpu`,
so anything a consumer adds here must too.

## building

`vcad` depends on a sibling `../tang` checkout and `phyz` on crates.io `tang`;
the workspace `[patch.crates-io]` unifies them on the checkout so `tang::Scalar`
is one trait across the graph. `vcad-kernel` is built with `no-builtin-font`
so it does not need vcad's `node_modules`.
