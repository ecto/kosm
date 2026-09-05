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

`--pool-render kosm` runs the same simulation through `kosm-render`'s
general path tracer instead (`pool/render.rs`), writing `out/pool_kosm/` and
`out/pool_kosm.mp4`; `legacy` is the default. The pool ports over cleanly as
geometry: the basin, deck, coping and grandstand are boxes, the tiles a
two-material checker, the melon a unit-sphere `TriMesh` under a non-uniform
`Transform` (the TLAS renormalises the local ray and maps the normal by the
inverse transpose, so a scaled sphere is an honest ellipsoid), and the water
five `HeightField`s — one fine over the splash, four coarse tiling the rest
of the pool around it — sampled from `Surface::height`, so the fine grid, the
far field and the blend between them arrive already resolved. The per-frame
cost is what the height field was built for: `update_heights` plus
`Bvh::refit` is **0.3–0.8 ms** for the whole surface, against a 100–180 ms
rebuild. Water is `transmission: 1`, `ior: 1.333`, `roughness: 0.02`, with
`attenuation_color` the reference's per-metre `ABSORB` as `exp(-a)` over a
metre; the sky is a `GradientEnv` of the reference's own two blues and the
sun is a real 0.6° disc. Tonemapping is `kosm-render`'s ACES rather than the
pool's filmic curve, which needs about a stop of headroom (exposure 0.35) to
keep the deck off the clip point.

**The caustic did not survive the port, and then it did — by a second pass,
not by more samples.** The reference tracer's `Caustic` grid is a *forward*
transport step
— sun rays pushed through the surface by Snell's law and binned where they
land — bolted onto a backward tracer, and it exists precisely because a
unidirectional tracer cannot find the sun through moving water. Two things
in `kosm-render` stop it. First, `Scene::occluded` is a material-blind
any-hit test, so the water is an opaque blocker: next-event estimation to the
sun is rejected at every point on the tiles, and the only surviving path is a
BSDF bounce off the floor that refracts back up and lands inside a disc of
3.4e-4 sr — about one cosine-sampled ray in 10^4. Second, the sun's radiance
is its irradiance over that disc, roughly 9000 in these units, and the
default `firefly_clamp` is 12: the rare paths that *do* find the sun are
scaled down 750x before averaging, so the caustic is not merely noisy but
clamped to nothing. Measured at 640x360: at 64 spp the floor is perfectly
smooth and perfectly caustic-free; at 1024 spp with the clamp disabled
(`KOSM_CLAMP=0 KOSM_DENOISE=0`, 304 s for the one frame) the caustic energy
appears as isolated single-sample specks scattered evenly over the whole
floor with no ring structure at all — 0.1 expected sun hits per pixel. A
readable caustic would need 10^4-10^5 spp, hours to a day per frame. It is
unusable, and no sample count fixes it.

**The verdict, updated.** The diagnosis above is unchanged and so is the
conclusion drawn from it: no sample count fixes this. What fixed it is
`kosm_render::caustics` — a photon-mapped caustic pass, which is the cheap and
honest answer for this scene and, not coincidentally, what the reference's
`Caustic` grid already was. Same physics, general machinery: sun photons
aimed at the water, pushed through the surface by the BSDF's own refraction,
deposited into a world-space grid, read back as direct light. NEE still treats
the water as opaque, so the two do not double count.

Measured at 1280×720, 64 spp, frame 0 (ambient ripple, no impact yet), over a
patch of tile floor: mean sRGB **59.6 → 78.8** and standard deviation **9.5 →
15.9**. The floor is no longer "perfectly smooth and perfectly caustic-free" —
it is brighter, because the sun now reaches it at all, and it is dappled,
because the ripple focuses. After the melon lands (frame 36) the expanding
rings throw a matching ring of light onto the tiles, the same structure the
legacy tracer's grid draws in the same frame, softer at the edges because a
6 cm gather radius is a blur where a binned grid is not. The pass costs
**1.4 s of a 30.4 s frame** — about 5% — for 2M photons over the whole pool.
`KOSM_PHOTONS=0` restores the caustic-free render this section used to
describe.

The legacy renderer stays the default: foam, the crowd and its filmic curve
are still its own. Foam is
skipped as well (it is a coverage field that whitens the surface shade, and
`Pbr` has no per-point channel a client can drive without textures), as is
the crowd (raymarched SDFs in the reference); droplets do port, as small
placed spheres of the same water.

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

The same frame is now also rendered by kosm-render (`out/garage/frame_kosm.png`,
and a close-up `frame_kosm_close.png`), from the *same* 274 499 Gaussians: the
cloud goes in as `Scene::splats` and the marble as an analytic glass sphere
inside it, with a black environment so that every photon in the picture came
from the capture. The two backdrops agree — the same red car, the same open
floor, the same needles where the training grew them — which they should,
because it is one set of Gaussians composited two ways: the rasteriser
projects each one to a 2D footprint and sorts by depth, we take each one's
alpha at the ray's maximum-response point in 3D and sort by `t`. Where they
differ is exposure: our walk reaches every Gaussian within 3σ of the ray, and
the sum of those alphas runs a little hot, so the bright floor blooms where
the rasteriser's tiled 2D footprints stay inside the clamp. And where the
rasterised frame pastes the marble over the picture, this one has the marble
*in* it — the glass takes its reflection and its refraction from the room's
own radiance, which is what "an environment with depth" buys. At 3.4 m a 1 cm
marble is ten pixels, so `frame_kosm_close.png` is a 6 cm ball seen from 80 cm
instead: a dark sphere with a bright rim where the lit wall is, its darkness
the floor it is refracting. That frame also shows the two honest limits of
this. A capture has no hole in it where you put something, so the segment
*inside* the ball composites the floor Gaussians the ball is standing among —
the cloud is a volume and the marble shares space with it. And at 80 cm the
capture is mush, exactly as `ipse-map`'s own warning says: the SDF exists
because the splat stops being a picture of anything close up. 24 spp at
800×600 takes about 50 s on the CPU tier (`KOSM_GARAGE_SPP` to change it).

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

`levels/warehouse.loon` is the same machinery at level scale: THPS1's first
level scaled to the K1 inside a 16 × 9 m shed — half pipe, mezzanine, two
quarter pipes either side of an open door, a platform, a bent rail, kickers,
box piles, a ledge, and the building itself. One root per piece, each with a
material name; roots whose material begins `no-collide` (roof, trusses,
skylight, glazing, piers, door frame, the wall above the brick skirt) are drawn
but never baked, so the SDF stops at the metre of wall the K1 can hit
instead of following the roof to the ridge. The bake writes
`parts/<root>.stl` and a `parts.json` of names, materials and colours, and
the ride recorder draws the level from those rather than from one grey mesh.
20 mm cells over that volume is 372 MB of f32, so the level asks for 25 mm
(191 MB, 18 s). `tests/warehouse.rs` pins the quarter pipe against
10/7 · g · Δ, the kicker against the height a wheel rolled at it reaches,
and the rail against the radius the field puts around its axis.

```bash
cargo run --release -p kosm-spike -- --skatepark
cargo run --release -p kosm-spike -- --skatepark levels/warehouse.loon
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

### the world is computed, not predicted

A game renderer that wants to hide its own latency guesses: it estimates
motion from screen space and interpolates frames it never rendered. This one
owns the physics, so it does neither. The court costs about **3 ms of solver
for a frame that costs 30 ms of render**, so the cheap half is told to run
early: the window measures the time between a frame's own moment and the blit
of its picture, hands that number to the simulation as a head start, and then
asks for the frame due at *the moment the picture will be on the glass* rather
than the newest frame there is. The simulation is deterministic, so that frame
is not a prediction — it is the frame, taken in `dt` steps to its own time,
just computed early. `--at`-style stepping and the lookahead reach the same
state bit for bit, and a test says so. Measured on this machine, over the same
minute of the same level: **59 ms of presented latency before, 31 ms after**,
with the pass time unchanged at ~30 ms. Paused, the head start is zero — the
window renders the frame under the cursor and has no future to reach for.

Motion vectors come from the same place. The frames are real frames, the
reprojection differences the pose the *previous pass* was folded at against
this one's, and `Snapshot` now carries each ball's world-frame linear and
angular velocity and each net cord's alongside the poses — so a displacement
can be read straight out of the physics. It is exact, not an estimate: the
solver advances a position by the velocity at the end of each step, so the
trapezoid of two frames' velocities plus half a step of their difference is
the displacement the solver actually applied, and for a ball in flight the two
agree to 1e-16 m. That is what `Snapshot::ball_displacement` is, and what
`velocity_is_the_motion_vector` checks.

Nothing blocks: the simulation never waits for the renderer, the window
presents at the display's refresh with the newest completed picture, and the
frames that go by unrendered are counted rather than hidden — `court pace:`
says the latency, the head start, how far ahead the simulation is, and how
many frames were dropped and how many moments arrived late, every two seconds.
A pass that overruns a refresh still leaves the picture smooth in *time* —
every frame presented is a real frame at its right moment — and only coarser
in pixels, which is the trade the tuner was already making. If a render spike
takes the core out from under the solver, the simulation gives up the debt
rather than chasing it: past 120 ms of slip the clock is re-based, so the
world runs slow rather than permanently behind. (Chasing it was worth a
second of latency in a session that had been holding thirty milliseconds.)
`KOSM_NO_LOOKAHEAD=1` puts the old behaviour back, so the two can be measured
against each other in one sitting.

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
disc its shadow throws from each panel. That mask is geometric and it is computed before a ray is cast, from the
poses alone. It is the **CPU tier's** now: the GPU tier stopped drawing
rectangles and decides per pixel, on the device (see below).

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
texture now; the CPU tier still hands over bytes). What was left on this side
was a **keep mask** — one byte a pixel, 1 to go on accumulating and 0 to start
over. It is gone; what replaced it is next.

### the rectangle is not the answer

That mask is a rectangle, and a rectangle is visible. Every pixel inside the
box around a ball's old and new pose restarts from one sample — the ball
itself included, whose shading barely changed — so what travels with the ball
is a box of grain with a straight edge on it. Real-time tracers do not do
this; they decide per *pixel* whether last frame's estimate is still about the
same thing.

kosm-render can do that now, and the whole of it is on the device.

**The guides carry an identity.** The integrator already wrote a normal, a
depth and an albedo per pixel for the denoiser. It writes the hit's own
primitive id too, packed into the albedo plane's spare lane — no new buffer, no
new binding, and the storage-buffer budget untouched.

**The reprojection follows objects, not just the camera.** `InstanceMotion` is
one frame's answer to "what moved": which instance each primitive belongs to,
and each instance's `prev_T · cur_T⁻¹`. The reproject pass unprojects a pixel
through this pass's depth, carries the world point back through *its own
instance's* transform, projects it through the previous camera, and takes that
pixel's history where the id, the depth (tangent-plane, as before) and the
normal all agree. A ball in flight keeps its own shading; the floor uncovered
behind it has a different id and restarts. Measured over twelve frames of a
sphere sliding across a plane: the sphere's pixels average ten frames of
history when the motion is declared and 2.4 when it is not, the plane never
restarts, and the uncovered trail is the only thing that does.

**The fold is bounded.** A four-hundred-sample running mean cannot be moved by
what the pixel is seeing now. The history is capped — 64 by default — which
makes it an exponential moving average with a floor under its weight. Against
a 640-pass reference a capped 128-pass render is as close as an uncapped one
(0.0141 against 0.0130), so the still tier loses nothing.

**And what no geometric test can see gets clamped.** A ball leaves its shadow
behind on a floor that did not move, at the same depth with the same normal
and the same id. The only witness is this pass's own sample: compare the 3x3
mean of the *history* against the 3x3 mean of the *raw sample*, and where they
differ by more than the error bar on nine samples, shorten the history rather
than trust it. Shortened, not overwritten — snapping the colour to the
neighbourhood is the usual TAA move and it is biased, and a hundred still
passes of a biased nudge is a tint. Moving the light rig under a settled
frame: 27% of the stale lighting gone in four frames, 40% in eight.

**Nothing that restarts shows grain.** A pixel with fewer than four samples has
no error bar of its own — two moments over one sample are not statistics — so
it is given SVGF's spatial estimate instead, a 7x7 luminance variance over
neighbours that pass the same depth and normal gates the filter itself uses.
The à-trous pass is then wide and correct on the frame the pixel appears. A
one-sample frame measures *smoother* than a 64-sample one: 10/255 at the 95th
percentile of the deviation from a pixel's own 3x3 mean, against 11.

`tests/gpu_temporal.rs` is all five of those claims. The CPU-parity test —
the device à-trous against `pathtrace::denoise`, weight for weight — still
holds, with the spatial variance switched off: the CPU filter has no such
estimator, so the two tiers cannot agree byte for byte with it on, and the
test says so.

The viewer's half has landed too. `vcad-kernel-raytrace` points at *this*
`kosm-render` now, so `court_gpu.rs` hands `InstanceMotion` over instead of a
mask. Every frame it assembles the merged scene it already assembled, and
notes for each placed instance — every ball part, every net segment — which
range of *faces* it owns, because the face index is exactly the primitive id
the integrator writes into the guide plane. Differencing this frame's poses
against **the pass the history is in** (not the frame: two passes of one frame
moved nothing) gives one `prev_T · cur_T⁻¹` per instance that actually moved,
and the id table points that instance's faces at it. The court itself — a
hundred and forty-eight solids, eight hundred and eighty-five faces — stays
`InstanceMotion::STATIC` and costs a lane apiece.

There is no keep mask, no box and no scissor left on that tier: every pass is
the whole frame, and every pixel decides for itself whether last pass's mean
is still about the same thing. A camera move goes through the same call — the
previous camera as `prev_view`, the same motion table — rather than a path of
its own. The reprojection is offered only when something moved, eye or object:
a still camera over a still frame, which is what `--shot` is pass after pass,
would otherwise put every pixel through a depth and normal gate it can only
lose by, and the many-pass still moved on 37 000 pixels when it did. Skipped,
`--shot` is the picture it was — max 2 of 255 on five pixels of 480×270, where
two runs of the *same* binary differ by 4 on twenty-nine.

`--history-cap N`, `--clamp-k K`, `--clamp-reset N` and
`--no-spatial-variance` are the knobs, over kosm-render's own defaults (64,
4.0, 2, on).

`--restir` turns on ReSTIR DI for the ceiling panels — `--restir=32` names the
candidate count, `--restir-spatial N` and `--restir-radius R` the reuse — and
without it nothing changes at all. Sixty live frames at 640x360 from t = 0.4 s,
with and without: the same picture, the walls' even speckle down by a third
(mean deviation from a pixel's own 3x3, 6.5/255 without, 4.3/255 with) and the
mean luminance within 1%. No boiling, no ghosting, no trail behind the ball in
flight — the reservoirs are gated on the same depth and normal tests the
history is, and a pixel whose surface changed simply starts over. What ReSTIR
does *not* quiet here is most of it: this court is lit through its clerestory
by the sun, which keeps its own sample, and by three bounces of indirect,
which are nobody's reservoir. The pass goes from 53 ms to 85 ms. `Mask` and `Keep` stay in `history.rs` for the geometry the CPU
tier plans with and for the tests that keep it honest; nothing on the GPU tier
calls them.

`--dump-frames N --at T` is how the absence of the rectangle is checked by
eye: N consecutive live-tier frames at the level's own frame rate, one pass a
frame, through the same device history the window uses, written to
`out/view_seq/`. At 480×270 from t = 0.6 s — three balls rolling, one in
flight, the net about to be hit — there is no box around anything, no straight
edge travelling with a ball, and no patch of first-sample grain: the airborne
ball is round, carries its seams and its shading, and sits where a 32-pass
still of the same instant puts it. What the live frames do carry is an even
speckle over the walls, which is one sample a frame filtered, and a slightly
darker fringe on the ball's shadowed rim. No trail: the balls that look
smeared into a line are three real balls rolling in a line, and the static
reference shows the same three.

Reprojection is no longer CPU-side only, and a camera move no longer costs the
GPU picture its history. vcad grew
`accumulate_and_denoise_resident_reprojected`, which takes the *previous*
pass's camera: each pixel is unprojected through this pass's depth, projected
back into that view, and keeps the mean and count it finds where the surfaces
agree. So a moved camera uploads no mask at all now and lets the device settle it;
only disocclusions restart. `--orbit-test` is that claim, scripted:
converge headlessly, swing the eye three degrees about its target, take one
more pass and ask the device's own counts what survived. At 320×180 after eight
passes, **87.0% of the frame kept its history across the move**, where before it
was none of it. The previous camera is offered only when it is worth offering:
a still-camera pass passes `None` (a view reprojected onto itself is two
dispatches for nothing) and so does the first pass at a new size, since vcad
reallocates the history on a resize and there is no previous depth plane to
test against. A reprojected pass takes no scissor either — the reprojection
needs this pass's depth everywhere — and the window's log names it (`a full
reprojected pass`).

The scissor had a life on that tier and it is over; what follows is what it
was. `set_scissor` used to size the *trace* alone
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
that is always scissored never converges. With the mask retired there is
nothing left to scissor *by*, and the rule is moot on that tier.

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
of which 420 ms was the CPU à-trous filter. It then climbed 256×144 → 284×160 →
365×205 in the first second, found 365×205 cost 61 ms against a 30 ms budget,
stepped back to 320×180 at 20–50 ms a pass — and *stayed* there for the rest
of the session, accumulating past 1700 samples a pixel without one collapse.
With the mask gone and every pass full-frame it settles higher: a ninety-second
session opens at 512×288, finds it dear at 48 ms, and holds **426×240 at
27–30 ms a pass**, the mean history swinging between 9 and 28 samples a pixel
as the balls move through it — short, because a moving world is what a
bounded, clamped history is *for*.
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

The readbacks left in the GPU path are two, and neither is a pixel of the
picture. `--shot` copies the target texture out once at the end for the PNG.
And the window, at the same two seconds its log line runs on, asks the device
for its per-pixel counts so the line can say how long the history is and the
tuner can tell a converged picture from a churning one — the mask used to
mirror those counts on this side, and with the mask gone there is nothing here
that knows. A *pass* still reads nothing back. The deforming net still pays for a BVH build
inside every CPU pass. Evaluating the level takes **a minute or two** on this
machine — single-threaded, almost all of it in `propagate_boolean` sorting face
names under `circular_pattern` — and the window is black until it is done; it
says so on stderr while it works, and a `timeout 90` never gets past it.

### where the samples go

A game spends its samples uniformly. One ray per pixel per frame, whether the
pixel is a wall that has looked the same for four hundred frames or the ball
that crossed it on this one. The wall's ray buys nothing — its mean is already
inside a display bit — and the ball's one ray is not nearly enough.

The renderer knows which is which, and it knows *before* it traces anything:

* **the physics.** Every instance's transform this frame is already on the
  device, packed for the reprojection as `InstanceMotion`. With the previous
  frame's depth plane that is the exact screen-space displacement of the
  surface under each pixel — a motion vector, not a guess.
* **the history.** Every pixel carries its own sample count and the variance of
  its own mean. sigma/sqrt(n) is the error bar another sample would buy down.
* **the image.** The last raw sample against the running mean — the same
  disagreement the neighbourhood clamp measures, and the only thing that sees
  lighting going stale under a surface that did not move.

`gpu/budget.rs` and `gpu/shaders/budget.wgsl` turn those into a per-pixel
**sample budget** `b(p)` in five compute passes at the head of the frame:
one for the three drives, two to dilate the motion drive over the à-trous
filter's own footprint — a moved ball drags its shadow, its bounce and
everything the filter will reach for — one to normalise so the frame's total is
exactly the tuner's `rays_per_frame`, and one to turn `b(p)` into the set of
trace rounds each pixel takes. Directing the samples never spends more of
them. It moves them from the wall to the ball.

The formula the numbers below came out of, per pixel:

```
error   = min(sigma/(mu*sqrt(n)) / 0.15, 3)          the error bar, bounded
        + 6 * max(0, (8 - n)/8)                      ...and how little is behind it
change  = max(0, |L(raw) - L(mean)| / (2*sigma_1) - 1)   in the sample's own sigmas
motion  = dilate(displacement in pixels / 0.5)       what the physics says
w       = mix(1, 0.03 + error + 3*motion + 3*change, bias)
b(p)    = w * rays_per_frame / sum(w),   clamped to the pass's round count
```

Three things in there were each worth more than the rest of the design put
together, and all three were wrong in the first version:

* **the short-history term is not a small correction.** A pixel the clamp has
  just shortened — a floor the ball's shadow has moved off — is holding two
  samples and is the worst pixel in the frame. Capping its claim at one share,
  as the error-bar term is capped, told the budget it was no more deserving
  than a converged neighbour, and the whole feature measured as a rounding
  error. Six shares, and it measures.
* **the error-bar term has to be bounded, and its target has to be honest.**
  Every pixel of a live path-traced frame is over a 2% error bar for a long
  time, so a term written against 2% pins every pixel to its ceiling and is a
  flat field wearing a disguise.
* **disagreement has to be measured in the sample's own sigmas, not as a
  fraction of the mean.** A 1 spp sample sits half its own magnitude from the
  truth on a good day, so a relative test reads the Monte Carlo noise on every
  pixel far louder than the shadow that actually moved, and the budget follows
  the noise.

**Deterministic or stochastic.** The natural thing is for `accumulate` to loop
`b(p)` samples at pixel `p`, and that needs the *trace* to loop per pixel. The
integrator takes one sample per invocation per dispatch, so the pass is split
into rounds instead, and on round `r` a pixel takes its sample or does not.

Not by a coin, though — that was the second thing measured and rejected.
A coin at probability `b/rounds` is unbiased in the *mean*, which is all an
unbiasedness argument covers; but a pixel's error goes as `1/sqrt(count)`, and
`1/sqrt` is convex, so a count that scatters around `b` is worse than a count
that *is* `b`. On the fixture it gave up a third of what the feature was
buying. So a pixel takes exactly `floor(b)` rounds and one more with
probability `frac(b)` — the expectation is still exactly `b`, and the count
never strays by more than one. Nothing is reweighted by `1/p`: the selection
never looks at the sample's value, so the mean over what was folded is unbiased
on its own, and a reweighting would only add variance.

**And the rounds a pixel takes are chosen so the trace can skip them.** This is
where the feature stopped being a tax. The selection used to be made inside
`accumulate`, one round at a time, *after* the trace: every round traced the
whole frame and the fold threw away the samples the budget had not asked for.
Four rounds of rays to place one frame's worth of samples — the placement was
directed and the cost was not.

It is decided in a fifth pass now, `budget_select`, once, for every round at
once, and written where the trace can read it: guide plane 3 of the
depth/normal buffer, one word a pixel, bit `r` set when the pixel folds round
`r`. It rides in that buffer rather than in a binding of its own because the
integrator already binds all ten storage buffers a browser guarantees — the
same reason ReSTIR's reservoirs live there. `integrator.wgsl` reads that word
at the top of its entry point and returns before casting a ray; `accumulate`
reads the *same* word rather than re-deriving it, so the two can never
disagree about which rounds a pixel took.

Two details decide whether that saves any time at all, and neither is about
rays:

* **which** rounds. A GPU traces a workgroup, not a pixel. The certain
  `floor(b)` rounds are therefore the *low* ones — a busy pixel takes round 0
  and up, never a round drawn at random — so the rounds a tile is busy on are
  a prefix, and the tail of the frame is the handful of tiles with real work
  left in them.
* **whose** dither. The fractional round's coin is the **8x8 tile's**, not the
  pixel's, and so is the starvation floor's phase. A per-pixel coin scatters
  the selected pixels evenly over the frame, which puts one in every workgroup
  and makes every workgroup run: the skip saves the rays and none of the time.
  Keyed on the tile, a converged patch of wall decides together. Measured, that
  one change was the difference between a 15% saving and a 45% one.

What is given up is the fine-grained decorrelation of neighbouring pixels'
round sets. Each pixel still takes `floor(b)` or `ceil(b)` rounds, so the
counts and the budget's accounting are untouched; the equal-samples ratio moved
from 0.80 to 0.83.

Round 0 is the exception to the skip. The history pass reprojects *every*
pixel through this frame's guide planes, folded or not, so a skipped pixel
cannot be left holding last frame's primary hit. On round 0 the pixels the mask
skips run **guides-only**: the ray, the guide planes, and none of the shading,
which is where a path sample's cost is. It measures at about 4 ms of the pass.

A pixel that skips every round still has its history *committed* — a pixel that
skipped a frame the camera moved on would otherwise silently lose the history
the reprojection carried onto it.

**Nothing starves.** `budget_floor_k` guarantees every pixel a sample once
every k frames, on its tile's phase so the cost is spread rather than periodic
— and the guarantee is taken deterministically, because a guarantee that holds
two frames in three is not a guarantee. The phase is the tile's for the same
reason the dither is: a floor scattered pixel by pixel puts one floored pixel
in every workgroup of the frame, and a workgroup with one pixel in it costs
what a full one does.

`tests/gpu_sample_budget.rs` pins six claims on a sphere over a plane at
64x64, and these are its numbers:

* with the scene static and no motion table at all, the frame's budget totals
  **4094 samples against a target of 4096** — 0.05% — and the noisiest quarter
  of the frame by its own relative error bar gets **1.71 samples/pixel against
  0.66** for the calmest;
* with one ball moving, the ball's own pixels get **2.72 samples/pixel**, the
  neighbourhood the filter will drag with it **2.69**, and the rest of the
  frame **0.32** — a factor of eight;
* from a converged frame, sixteen frames of a ball crossing it at **equal
  samples folded per frame**: RMSE against a 512-sample reference of **0.0658
  uniform against 0.0547 directed, a ratio of 0.83**. Split, that is the ball
  0.121 → 0.102 and everything else 0.053 → 0.044 — the background improves
  too, because what it was short of was not rays but rays *where the clamp had
  just thrown a history away*;
* over one floor period, **0 of 4096 pixels** go untouched;
* a budgeted frame skips **2318 of 4096 pixels entirely**, and every pixel the
  ball has just moved onto still reads the ball's distance in the guide plane
  to within 1.2% of a CPU sphere intersection — skipped or not;
* and the skip folds what the fold used to fold, bit for bit.

`kosm-view --budget` turns it on in the viewer; `--budget=0.4` names the bias,
where 0 is the uniform spend the window always had and 1 is entirely where the
budget says. `--budget-rounds` (4), `--budget-radius` (32 pixels),
`--budget-floor` (16 frames) and `--rays-per-frame` (one a pixel) are the rest
of it.

`--dump-frames 60 --at 0.4 --width 640`, with and without: at frame 8, where
the history is still short, the directed frames are better everywhere at once —
the walls and the bleachers smoother, the net and the rim legible, the balls
sharp where the uniform run has them furred. By frame 59 the trade has settled
into what it is for: the moving ball, the net and the backboard are markedly
cleaner, the shadows on the floor crisper, and the far wall carries a little
more grain than the uniform run left it, because that is where the samples came
from.

The cost. A directed pass at 640x360 used to take **3.4 uniform passes** to
place one sample a pixel — 155 ms against 39 on a cold machine, four full-frame
trace rounds for 230,400 folded samples. With the skip it takes **2.1**, and
what it buys with the difference is samples: a directed pass placing **three**
samples a pixel now costs the same 3.4 passes the one-sample pass used to. The
frame that bought one directed sample buys three.

At equal *time*, though — three directed samples against the four flat ones the
same time buys — the two are a wash on the fixture (a ratio of 1.01): a fourth
flat sample is worth about what directing three is. The win is at equal
samples, which is the 0.83 above, and in no longer paying four rounds of rays
for one round of samples. `--rays-per-frame N` names the samples a pixel, so
both halves of that can be measured from the command line.

The one thing the skip changes that the fold did not: a skipped pixel's raw
sample and its guide planes go stale by a round. The neighbourhood clamp reads
its neighbours' raw samples, so a clamped pixel now sees each neighbour's most
recent sample rather than this round's — an equally good independent sample of
the same pixel, and not the same float.
`the_skip_folds_the_same_samples` pins both ends of that: bit for bit over one
budgeted frame with the clamp off, and 0.028 RMSE over eight frames with it on,
against the 0.066 either arm carries against the converged reference.

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
  length are unrelated facts. `subsurface` takes its share of this lobe away
  and hands it to a random walk inside the object — see below.
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

#### thin-film iridescence

A soap bubble, an oil slick on wet tarmac and the straw-to-blue run of colours
on tempered steel are one phenomenon: a film thin enough that the wave
reflected off its top and the wave that went through it, bounced off the
substrate and came back out are still coherent. They interfere — constructively
at the wavelengths where the optical path difference is a whole number of them,
destructively in between — so the reflectance stops being a smooth Fresnel
curve in λ and becomes a comb, and the comb slides as the angle changes because
the path difference does.

Belcour and Barla's "A Practical Extension to Microfacet Theory for the
Modeling of Varying Iridescence" (SIGGRAPH 2017) is the model. The Airy
summation over the film's infinitely many internal bounces is exact and cheap;
what is expensive is integrating the resulting comb against the three CIE
curves. Their observation is that you never have to: the summation's terms are
pure cosines in the path difference, so the integral wants the *Fourier
transform* of the colour matching functions at that frequency — and CIE 1931
fitted as a handful of Gaussians has one in closed form. That is
`optics::sensitivity`: six constants, no tables, and it ports to WGSL verbatim
like the rest of the spectral code.

| parameter | meaning |
|---|---|
| `thin_film_thickness` | OpenPBR's, in nanometres. `0` is no film and the default. A soap film runs 100–1000 nm, an oxide on steel 20–80, an AR coating a quarter of a wavelength |
| `thin_film_ior` | OpenPBR's `thin_film_ior`, default 1.5. The film sits between the outside and the substrate, and its contrast against both is what sets the strength |

The film modulates the **specular lobe's Fresnel**, and that is one line in one
place, so it colours the dielectric `F0` path and the metal path alike — a
metal's `f0` *is* its base colour, and an oxide over steel composes exactly
that way. It is a function of the half-vector cosine, so it is exactly as
reciprocal as the lobe it sits on, which `an_iridescent_bsdf_is_reciprocal`
holds it to by measuring the *same material without the film* as its bar.

**A path that already carries a hero wavelength does not want the colour
integral.** It wants the reflectance at its own λ, so it gets it: the series is
geometric and sums in closed form, which is both cheaper and exact rather than
truncated at two harmonics. `thin_film_thickness = 0` returns the plain Schlick
`fresnel` itself and not a limit of the film model that happens to be close, so
a material without a film is bit-for-bit what it was — and the court's CPU shot
is byte-identical across this change.

What the tests pin:

| test | what it holds |
|---|---|
| `a_three_hundred_nanometre_film_matches_the_analytic_airy` | a 300 nm n = 1.34 film on glass, at 450/550/650 nm, within 1% of the textbook Airy formula written from *amplitude* coefficients — a different derivation, not the same series rearranged |
| `a_zero_thickness_film_is_the_plain_fresnel_bit_for_bit` | no film is `assert_eq!`, not "close" |
| `a_film_never_reflects_more_than_it_receives` | every channel in [0, 1] over thickness × film index × angle × substrate `F0` |
| `an_iridescent_bsdf_is_reciprocal` | the film adds no asymmetry the compensated GGX beneath it did not already have |
| `an_iridescent_lobe_stays_under_one` | hemispherical albedo ≤ 1 for a white metal at three thicknesses and three roughnesses, RGB and hero paths both |
| `a_film_in_the_visible_band_is_chromatic` | a 320 nm film is not grey — otherwise none of this bought anything |
| `tests/gpu_bsdf.rs` | the sweep grew films over dielectrics and metals and a hero-wavelength axis: 4680 cases, worst relative disagreement 9.4e-6 |

#### subsurface, as transport rather than as a look

Disney's 2012 `subsurface` was a *blend*: a Hanrahan-Krueger-flavoured lobe
that flattened the diffuse falloff and brightened grazing angles the way a
short mean free path does. It looked like scattering and transported nothing.
Light never entered the object, so it never came out anywhere else, and the
effect vanished the moment you asked it for the thing that actually separates
skin, marble and rubber from paint of the same colour: light going *in* here
and coming *out* over there.

This is the transport — Chiang, Kutz and Burley's "Practical and Controllable
Subsurface Scattering for Production Path Tracing" (SIGGRAPH 2016). On a
subsurface entry the path stops being a surface event: it crosses into the
object, samples a distance against the medium's extinction, and either the
boundary comes first — in which case the path leaves *there*, from a new point
with a new normal — or it does not, in which case it scatters isotropically and
goes again. The exit point is found with the same `Geometry`/TLAS trace
everything else uses, with the ray inside the solid.

| parameter | meaning |
|---|---|
| `subsurface` | OpenPBR's `subsurface_weight`. Takes that fraction of the diffuse lobe away and replaces it with the walk. `0` is off and the default |
| `subsurface_color` | OpenPBR's, the **surface** albedo — what the material looks like, not what the medium is |
| `subsurface_radius` | OpenPBR's, the mean free path per channel in scene units. Per channel because red travels furthest through most organic media, which is why a hand held up to a light goes red at the edges |

**The inversion is the whole usability of it.** Nobody can pick a
single-scattering albedo: a medium at 0.9 reads very nearly white at the
surface, and the map between the two is a transcendental function of the
transport. Chiang's cubic fit inverts it, so the knob is the surface colour and
the renderer solves for the medium that produces it.

**The walk is not a BSDF and is not pretended to be one.** It has no density at
the point it entered — it leaves from somewhere else — so it never appears in
the sum `bsdf_eval` returns; it appears only in the lobe-selection weights and
in `bsdf_sample`, which reports "into the object" instead of a direction. The
exit continues as a specular chain, because no NEE strategy found that
direction and an emitter downstream must therefore take full MIS weight.

**The GPU walks too, and it walks shorter.** The CPU takes up to 1024
scattering events; the shader takes 12. A GPU path loop is a
uniform-control-flow budget shared by every lane in the workgroup, and a
thousand-step inner loop makes the whole wave wait on the one pixel that landed
in a bright medium. Twelve is plenty for the short mean free paths these scenes
use — the basketball's 2 mm rubber exits in three or four — and a medium bright
enough to need more comes back darker on the device than on the CPU. That is
the one place the two tiers are knowingly different; `subsurface = 0` is
identical on both, and the BSDF halves agree to 3e-6 in the parity harness.

What the tests pin:

| test | what it holds |
|---|---|
| `a_semi_infinite_slab_returns_its_own_colour` | a half-space of the material reflects `subsurface_color` back to within 3% at 60k walks — the fit, the distance sampling and the per-channel MIS weights all at once |
| `a_very_bright_medium_runs_a_little_dark` | and where the cubic gives out: 0.9 comes back as 0.87, stated rather than hidden behind a wider tolerance |
| `a_thin_slab_transmits` | a slab 0.4 mean free paths thick passes more than 20% out the far side and still reflects some, and the two together stay under 1 |
| `the_walk_never_returns_more_than_it_took` | energy ≤ 1 for white and for strongly coloured media |
| `subsurface_zero_leaves_the_other_lobes_alone` | `assert_eq!` on the whole evaluator, and no subsurface lobe to pick |
| `tests/gpu_bsdf.rs` | all six lobe-selection probabilities are now compared outright — a divergence there is invisible in any single evaluation and shows up only as noise |

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

**NEE through glass has its own section now** — see *light through glass* and
*caustics* below. The short version: a shadow ray goes *through* a thin pane,
attenuated; a refracting solid still blocks it, and a photon pass carries that
light instead.

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

### light through glass

A shadow ray used to stop at anything it touched. `Scene::occluded` was a
material-blind any-hit test, so a window pane was a brick wall: next-event
estimation found a blocker, returned black, and the only way light got into a
glazed room was a BSDF path that happened to refract through the pane and then
wander into the sun's 0.27° cone. That is one ray in tens of thousands, which
over the passes a frame gets is salt-and-pepper, not daylight — and it is why
`levels/court.loon` had its clerestory band cut open as a *hole* for a while,
with a comment saying so.

A thin sheet is not a blocker, it is a filter. `occluded` is now
`shadow_transmittance`, which returns *how much* survives the trip rather than
whether anything is in the way:

* an opaque surface, or a refracting **solid**, blocks as before — `None`;
* a `thin_walled` transmissive sheet is crossed, multiplying the ray's
  throughput by `transmission · (1 − F(cos θ))`;
* at most **four** sheets per shadow ray. A window is one pane, a double
  glazing two, a display case four; past that the light is not meaningfully
  getting through, and the cap is also what bounds the traversal.

The Fresnel factor is deliberately `1 − F` and not `(1 − F)²`. This renderer's
thin-walled lobe is a *single* interface with `R + T = 1` — refract in and
straight back out, no interior, no lateral offset — and
`dielectric_eval`'s thin-walled branch applies exactly that factor to a
BSDF-sampled path. A shadow ray that disagreed by a second `(1 − F)` would make
the two strategies estimate different integrals, and MIS would then double count
the refracted path in one direction and lose energy in the other. Agreement is
the requirement; the second interface is an approximation this renderer's sheet
does not make anywhere else, so it is not made here either.

A **frosted** pane still blocks. A rough sheet scatters, and the straight-line
shadow ray is only the right answer in the smooth limit, so the transmittance
tapers to zero as the specular lobe opens up (`1 − alpha`). Clear glass at
`roughness 0.02` loses 0.04% to that taper; a fully rough pane loses all of it
and behaves exactly as it did before.

Both tiers. `gpu/shaders/integrator.wgsl` has the same function with the same
cap and the same factor, and `tests/gpu_pane.rs` holds the device to it.

| test | what it holds |
|---|---|
| `next_event_passes_through_a_thin_pane` | a panel light behind a pane lights a Lambertian point at `(1 − F)` of the open irradiance, within 2% |
| `a_converged_render_through_a_pane_matches_the_single_strategy` | the *combined* NEE + BSDF estimator lands on the same factor within 3% — which is the MIS consistency check: if the two strategies disagreed, the combination would not |
| `a_shadow_ray_gives_up_past_the_sheet_cap` | five stacked panes is opaque |
| `a_frosted_pane_still_blocks_the_shadow_ray` | the smooth-limit approximation refuses to be used outside it |
| `an_opaque_scene_occludes_exactly_as_the_any_hit_test_did` | 4000 random rays against an opaque scene, hit for hit — the bit-identity guarantee for every render that predates panes |
| `tests/gpu_pane.rs` | the device lights a floor through a pane at `(1 − F)`, within 2% |

The court's clerestory is glazed again, and the `sky 1` still reads the same
brightness it did with bare openings, four percent down.

### caustics

The other half of the problem is the half a shadow ray cannot be talked into.
A *thin* pane does not bend the line to the light measurably, which is why the
attenuation trick above is legitimate. A glass ball, or a metre of moving
water, bends it completely: the light that arrives came in along a refracted
path, and there is no straight line to attenuate. `### the pool` spent a while
establishing that this is not a sample-count problem — at 1024 spp with the
clamp off, the pool floor's caustic was isolated single-sample specks with no
ring structure at all.

So `kosm_render::caustics` goes the other way. Forward light transport, the
direction photons actually travel:

1. **Emit at the glass.** Photons come off each light aimed into the cone that
   subtends the refractive geometry's bounds — a cosine-weighted point and a
   cone direction for an area light, a disc covering the bounds for the sun.
   Aiming is importance sampling, not a cheat: the photon's power carries the
   cone's solid angle, so the estimator is the one a full-hemisphere emission
   would give and simply spends none of its budget on photons that were never
   going to reach the glass. A photon whose first hit is not transmissive is
   dropped.
2. **Follow it with the camera path's own BSDF.** Refraction, reflection,
   Beer–Lambert inside the medium, and dispersion — a photon that meets a
   dispersive material draws a hero wavelength and deposits RGB through
   `spectrum::hero_weight`, the same fan the backward tracer uses. No
   correction factor is needed for importance transport, because
   `dielectric_eval` already cancels Walter's `η_t²` against the `1/η²`
   radiance compression, so what it returns is the symmetric quantity.
3. **Deposit at the first diffuse surface**, into a world-space hash grid of
   splats. Not a per-object texel grid: the grid does not care what shape the
   receiver is, needs no parameterisation and no projection onto a dominant
   plane, and a pool floor with a drain and a step in it is exactly where a
   projected grid goes wrong. The kernel is constant over the gather disc —
   `Φ/(π r²)` — which conserves energy exactly, so the energy test below tests
   the pass and not a kernel's normalisation.
4. **Read it as direct light.** At a diffuse hit the integrator adds
   `albedo/π × E(x)` from the map.

**Nothing is counted twice, and the rule is one line.** A photon is written
into the map only if its history included a transmissive event on geometry that
is **not** thin-walled. Every unit of light is claimed by exactly one
estimator: unobstructed light and light through a pane belong to NEE (which now
sees through panes); light through a refracting solid or a water surface
belongs to the map, and a shadow ray still treats those as opaque, so NEE
contributes nothing along those directions.

**Energy.** A flat glass slab at normal incidence has an analytic
transmittance, `(1 − F)² = 0.9216` at n = 1.5, and the pass reproduces it to
within 5% — which exercises the emission disc's normalisation and area, the aim
at the bounds, the first-hit filter, the walk through two interfaces and the
deposit, all at once. A glass ball under a straight-down sun of irradiance 1
takes in π (its silhouette) and deposits **2.656, or 84.5%** — the Fresnel loss
over two interfaces plus total internal reflection at the rim — concentrating
it into a spot of **102** at the paraxial focus, a hundred times the sun's own
irradiance. Floor out of the ball's reach renders pixel-for-pixel as it did
without the map.

**CPU only this round.** The GPU integrator ignores the map: there is no
buffer upload and no device-side gather, so `--features gpu` renders the same
scene without its caustic. That is a real gap and not a rounding of one.

**Where it shows up.** The pool's tiles (see `### the pool` — mean sRGB 59.6 →
78.8, standard deviation 9.5 → 15.9 over a patch of floor, at 5% of the frame
time) and the marble's beauty frame, where the bead now throws a bright spot
onto the plate under it instead of a plain shadow — over the 16x12 px patch
directly beneath the bead, mean sRGB **134.9 -> 154.8** and peak **166 ->
198**. That spot is the one
`light.rs` has traced since the beginning — 472k rays × 5 spectral bands, peak
368× direct, 13.3 mm rms spread — except that `light.rs` computes a number for
the optimiser and this puts it in the picture, through the same N-BK7
Sellmeier curve.

Cost is `photons`, a gather `radius` and `max_bounces`. Photons are shot in
parallel with one seeded stream each, so the pass is deterministic however
rayon schedules it — the same property the pixel loop has, for the same reason.

### fireflies, and what the clamp is allowed to eat

The default `firefly_clamp` is an absolute cap in radiance units: any direct
estimate past depth 0 is truncated to 12. Cheap, effective, and biased in a way
that does not matter for a studio render — but it is a fixed number against a
quantity whose scale is the scene's, so a bright scene has its highlights
shaved and a dim one keeps its fireflies. In the pool it was worse than that:
the sun's radiance there is about 9000, so every path that *did* find the sun
through the water was scaled down 750× before averaging. The caustic was not
merely noisy, it was clamped to nothing.

Two changes:

* **The caustic map is never clamped.** It is a density estimate over many
  photons, not a Monte Carlo spike; it has no long tail to cut, and clamping a
  focused spot is indistinguishable from deleting it. The map's contribution is
  added outside the clamp.
* **`firefly_clamp_relative`** clamps against the pixel's own running mean
  instead of a fixed number: `8.0` lets through any sample within eight times
  the brightness the pixel has settled on, and cuts the rest. Scale-free, so
  the same value works on a sunlit court and a dim pool, and it adapts to the
  pixel rather than to the scene — a pixel inside a caustic has a high running
  mean and keeps its energy, a pixel in shadow does not. It does not engage
  until `firefly_clamp_warmup` (16) samples have landed, because the first few
  have no mean to speak of.

`firefly_clamp_relative` defaults to `None`, which leaves the absolute clamp in
charge and every render that predates the field bit-identical.

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

### ReSTIR, for the panels

Bitterli, Wyman, Pharr, Shirley, Lefohn and Jarosz 2020 — spatiotemporal
reservoir resampling. The direct term at the primary hit stops being one
next-event sample and becomes a *resampled* one. Sixteen candidate points are
drawn on the panels per pixel per frame and reduced to a single survivor by
weighted reservoir sampling against an unshadowed target

    p̂(y) = luminance( f(wo, wi)·cosθ · Le(y) ) · G(x, y)

which costs no rays at all — and only the survivor is shadow-tested. The
reservoir is then reused: from this pixel last frame, reprojected through the
previous camera and gated on the same 10%-depth / 25°-normal similarity the
history uses, and from a few neighbours this frame. So the pixel still spends
exactly one shadow ray on the panels, and spends it on the light that
hundreds of samples agreed was worth testing.

`GpuRenderState::set_restir(candidates, spatial_passes, spatial_radius)` is
the whole interface, and **zero candidates — the default — is the path the
shader has always taken**. Every existing test and every `--shot` is
byte-for-byte what it was; nothing about ReSTIR runs, and nothing about it is
allocated.

**What it covers, and what it deliberately does not.** The area panels, at
the primary hit. Indirect bounces keep their one-light NEE — a reservoir is a
per-pixel object and there is no pixel behind a second bounce — and so does
the environment. So does *the sun*, and that one was learned the hard way: the
sun went into the reservoir first, as one strategy among the panels, which in
a court lit through clerestory openings by the sun meant ten pixels in eleven
got no sun sample at all that frame. Sixty live frames came out visibly dark
and three times specklier than the same frames without ReSTIR. A 0.6° disc is
one light with a good importance sampler already; resampling is for choosing
among many.

**Biased, and by how much.** The temporal and spatial combinations are the
paper's biased ones — Algorithm 4 without MIS weights, with the previous M
clamped to 20x this frame's — because the unbiased variant needs a visibility
ray per reused neighbour and the whole point is that reuse costs none. The
spatial pass does normalise by *Z*, the candidates that could actually have
produced the survivor, rather than by every candidate looked at; without that
the frame comes out 10.4% dark. What is left biased is the visibility the
pass does not test: a sample visible at the neighbour and occluded here still
lights this pixel. On a closed room of twenty-four ceiling panels at 128x96,
three bounces, against a 160-frame plain-NEE reference of the same scene:

| | mean radiance | vs plain NEE |
|---|---|---|
| plain NEE | 2.46279 | — |
| ReSTIR, temporal only | 2.45945 | −0.14% |
| ReSTIR, one spatial pass | 2.45967 | −0.13% |

Two things are *not* folded into the reservoir, and both were measured
brightening it by several percent before they were taken out. Visibility is
one: the reservoir never learns whether its survivor was occluded — the
shadow ray is spent at shading time — because a chain that keeps the samples
that were visible and drops the ones that were not is conditioned on
visibility, and read +4.5%. The other is the sample's own existence: a
reservoir whose survivor names a point the panel has since moved out from
under is dropped *whole*, M and all, rather than kept at zero weight.

**Noise.** One sample a frame, after eight frames, root-mean-square from a
192-frame reference:

| | plain NEE | ReSTIR (M=16, 1 spatial) | |
|---|---|---|---|
| direct term only | 1.534 | 0.526 | **2.91x quieter** |
| the whole 3-bounce path | 2.333 | 1.835 | 1.27x quieter |

The second row is the honest one for a viewport: ReSTIR does nothing about
indirect noise, and indirect is most of what is left. Wide reuse measures
*worse* than narrow at every candidate count — 2.91x at a 4 px radius, 2.68x
at 16 px — because a neighbour four pixels away is looking at very nearly the
same integral and one thirty pixels away is not, so the default radius is 4
and one pass.

**A light that moves does not linger.** Nothing else would notice: a panel
keeps its index and its emission when it is translated, so a stale point in
empty air resolves to a perfectly plausible direction, distance and geometry
term. Re-evaluating p̂ every frame only helps if p̂ can *tell*, so
`restir_resolve` asks whether the stored point is still on the panel it names.
Without that check, translating the test room's rig under settled reservoirs
left the picture 26% bright and it stayed there. With it, the very next frame
is within 2% of a render that never saw the old rig, and stays there.

**Cost.** 512x288, M=16, against the same scene's plain pass: 33 ms with one
spatial pass and 42 ms with two, over 10–14 ms plain. Four submissions a
frame rather than one — the stages differ only in a uniform, and a queue
write applies to every command buffer in the submission it precedes. The
reservoirs ride in spare planes of `depth_normal_buffer`, 192 bytes a pixel,
because the shader already binds all ten storage buffers a browser
guarantees and five of those belong to the geometry module. Four slots: two
ping-pong across frames and carry the temporal chain, two ping-pong across
this frame's spatial passes. Keeping them apart is not tidiness — feeding
spatial output back into the temporal chain compounds its bias, and a 30 px
radius measured *worse than plain NEE* by frame eight while measuring 2.4x
better on frame one, which is what a feedback loop looks like.

`tests/gpu_restir.rs` is all four claims.

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

### a denoiser trained on our own reference renders

The à-trous filter above is *general*: five constants that have to be right for
a kitchen, a skatepark and a gym at once. There is another option, and it is
one only a renderer that owns its scenes can take. The court is ours, the path
tracer is ours, and a 1024-spp render of the court is exact ground truth we can
make as much of as we are willing to wait for. So: sample the level, render
each sample noisy *and* converged, and fit a filter to the difference.

**The dataset** (`kosm-spike`, `examples/denoise_dataset.rs`). Fifty (camera,
time) states — the authored camera orbited through a full circle of azimuth,
±0.25/0.45 rad of elevation and 0.7–1.35× its distance, at a random instant of
the first four seconds of the shot. Each rendered at 320×180 sixteen times at
one sample per pixel, snapshotted as running means at 1, 2, 4, 8 and 16 passes
with the variance of each mean, plus the normal, depth and albedo guides — and
then once more at **1024 spp with the filter off**. Every plane f16, whole
frame, crops taken at load time: **172.8 MB, 39 minutes** on sixteen cores
(8.2 hours of CPU). Cut into 64×64 tiles that is 2000 training tiles and 500
held out.

**The network** (`kosm_render::neural`, trained by `court::denoise`). A
kernel-predicting network in the sense of Bako et al., not a U-Net: three 3×3
convolutions over ten per-pixel feature planes, 32 hidden channels, ReLU, and a
softmax over 25 outputs that are then used as the weights of a 5×5 average over
the frame's *own* demodulated illumination. **19,385 parameters, 77 KB.** It
predicts a filter and not a picture, so it cannot invent light — the same
promise the à-trous filter makes, with the weights learned rather than
stipulated. A U-Net's downsamples are exactly what would let it hallucinate a
shadow, and a hallucinated shadow that flickers is worse than the grain it
replaced.

The convolutions are hand-written `f32` loops with hand-written backward
passes. `tang_train::Conv2d` exists and is gradient-checked, but its forward is
`Tensor::from_fn` over a multi-dimensional `get`; this network is 80 MMAC per
tile forward and that difference is a training run against a weekend. What
`tang_train` does supply is what it is good at: `Parameter` holding the weights
and their gradients, and `ModuleAdam` stepping them. The backward pass is
checked against a directional finite difference through every layer.

The loss is L1 on `x/(1+x)`-compressed radiance plus a tenth-weight
gradient-domain term — L1 rather than L2 because L2's optimum under uncertainty
is the mean and the mean of "this edge is here or one pixel over" is a blurred
edge; plus gradients because L1 alone is indifferent between the right values
and the right values in the wrong arrangement.

Thirty epochs, batch 16, Adam at 2e-3: **18 minutes**, loss 0.0137 → 0.0076.
Held-out RMSE on the 500 tiles it never saw, through the same tone curve:

| accumulated passes | raw | à-trous | neural |
|---|---|---|---|
| 1 | 0.0764 | 0.0201 | **0.0155** |
| 2 | 0.0607 | 0.0164 | **0.0130** |
| 4 | 0.0472 | 0.0128 | **0.0107** |
| 8 | 0.0357 | 0.0111 | **0.0091** |
| 16 | 0.0268 | 0.0103 | **0.0080** |

16–23% better than the filter it replaces, at every history length.

**Inference** (`gpu/neural.rs`, `neural.wgsl`, `--denoise neural`). Three
compute dispatches, one per convolution, with the hidden activations in storage
buffers rather than workgroup memory — a fused pass would need 39 KB of the
16 KB budget and would recompute every pixel outside its own 8×8 tile twice.
The pass reads the history's running mean, its per-pixel statistics and the
scene's guide planes, and writes into the same `(illumination, variance)`
scratch buffer the wavelet iterations write, so `resolve` remodulates and
tonemaps without knowing which filter ran. `tests/gpu_neural.rs` pins the WGSL
against the Rust reference forward to 2e-3 relative, and a test in `kosm-spike`
pins the *trainer's* forward against that same reference — three
implementations of one network, and the weight file means the same thing to all
three.

At 512×288 on Metal: **13.6 ms** against the à-trous chain's 0.7–1.0 ms on a
converged frame. It is not a cheaper filter. `9·(10h + h² + 25h)` is 19,300
multiply-accumulates a pixel and there is no getting around it at 32 channels.

**And then we looked at it.** `--dump-frames` at 512×288, à-trous and neural,
against a 256-pass reference of the same instant. RMSE in 8-bit codes:

| region | à-trous | neural |
|---|---|---|
| whole frame | 9.31 | 17.84 |
| back wall | 6.45 | 7.55 |
| bleachers | 7.08 | 7.75 |
| floor | 10.07 | **9.81** |
| hoop and backboard | 12.77 | **42.68** |

The held-out table says the network wins and the viewer says it loses, and both
are true. On the flat, noisy, low-frequency majority of the frame the two are
within a code of each other. The whole of the difference is the hoop: the glass
backboard comes out **1.7× too bright** (159 against the reference's 94) and
the net's cords blur into a white smudge where the à-trous filter keeps the
lattice.

What it does *not* do is flicker. Frame to frame on a patch of static wall
across the sequence the neural filter moves 3.8, 4.8, 5.6 codes against the
à-trous filter's 3.8, 5.2, 6.5 — slightly steadier, which is what a
kernel-predicting network with no downsampling should be. The failure is
spatial and it is stationary; it is the same wrong backboard every frame.

Two things are different between the tiles it was scored on and the frame it
was run on, and both are ours:

* **The history does not mean what the dataset said it means.** A dataset tier
  is the mean of *k* independent one-spp passes and the variance of that mean.
  The viewer's history is an exponential moving average with a `history_cap`,
  carried across camera and object motion by the reprojection and shortened by
  the neighbourhood clamp. Its `count` and `variance` are not the dataset's
  `count` and `variance`, and two of the network's ten input planes are exactly
  those. The task's better option — driving the device history headlessly to
  build the dataset — is the fix, and this is the bill for not taking it.
* **The tiles are 320×180 and the frame is 512×288.** A 5×5 kernel covers
  different amounts of world in the two, and the net's cords are about a pixel
  wide in one and two in the other.

So: the method works, the plumbing is right end to end, and the weights that
ship are not yet good enough to make `--denoise neural` the default. That is
the honest state of it, and the next move is a dataset built from the device's
own history at the viewport's own resolution rather than a bigger network.

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

#### what a client can do now: composite a splat cloud

A `Scene` may carry `splats: Option<Arc<Bvh<Splats>>>`, and the integrator
consumes it. On **every ray segment** — the camera ray and each bounce ray
alike — `splats::composite` gathers the cloud front to back up to whatever the
segment ran into and walks `C += T·α·c(dir); T *= (1 - α)`, stopping when `T`
falls under 1e-4. The radiance is added as emission (`L += throughput·C`) and
the rest of the path is scaled by what got through (`throughput *= T`), so
splats in front of a wall veil it and splats behind it are never reached.
Shadow rays multiply their transmittance by the same accumulation, which is
why a captured wall stops a light and reconstruction dust does not.

The model, stated once: **a splat cloud is emissive and absorbing.** Its
colours already contain the lighting of the room it was captured in, so it is
never shaded, never receives light, and spawns no secondary rays — shading it
again would double-count. What that makes it, for the integrator, is **an
environment with depth**: like a lat-long `EnvMap` it supplies the radiance
for a ray that finds no analytic surface, so a marble dropped inside one picks
up reflections and diffuse bounce from the real room; unlike one it also
occupies space, so it can stand in front of things as well as behind them.

The limitation that comes with it: **the splat field is not importance
sampled.** There is no `Environment::sample` aimed at it and no MIS strategy
for its bright spots, so its indirect light arrives only on BSDF-sampled
bounce rays. A smooth or specular surface under a captured room is clean at
low sample counts; a rough diffuse one under a small bright window is noisy —
the same trade, for the same reason, as the analytic `GradientEnv`. The gather
also pays a sorted `Bvh::trace` per segment rather than an any-hit test, which
is what makes the CPU tier the tier this ships on. And a cloud is a volume,
not a shell: an analytic object placed inside one shares space with whatever
Gaussians are already there, so the segments *through* a glass sphere pick up
the cloud the sphere is standing in. There is no carving, and that is a
statement about the capture rather than about the integrator.

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

`cargo run --release -p kosm-view -- --live` runs the rollout instead of
reading one: the ipse recorder
(`target/release/examples/k1_skatepark_ride --stream`, run from
`/Users/cam/Developer/ipse` because it resolves `objects/skateboard` against
the cwd) streams the ride on stdout — the header first, then one frame per
line — and a reader thread appends frames as they land while its stderr goes
straight to ours. `--live-cmd "<program and args>"` swaps in another streamer,
split on whitespace, run from the current directory. The transport gains
"follow live" (on by default), a shove peak (N·s), a shove time and a
duration, and a "restart" button that kills the child and re-runs it with
those values on an empty timeline; the status line reads
`live: 123 frames, t = 2.05 s, child running`. Playback stays paced by the
ride's own `dt` even though the recorder runs several times faster than real
time, so following live rides the frontier of arrived frames rather than
jumping to it, and pausing or scrubbing back works while frames keep
arriving. Closing the window kills the child. `--shot`/`--frame` work here
too: the window waits for that frame (or for the child to end) before it
draws and quits.

## building

`vcad` depends on a sibling `../tang` checkout and `phyz` on crates.io `tang`;
the workspace `[patch.crates-io]` unifies them on the checkout so `tang::Scalar`
is one trait across the graph. `vcad-kernel` is built with `no-builtin-font`
so it does not need vcad's `node_modules`.
