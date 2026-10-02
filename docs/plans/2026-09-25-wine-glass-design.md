# The wine glass: one substance table, heard, seen and rippling

Date: 2026-09-25. Status: built 2026-09-26 (phases 1–5); see "As built" at the end. A new sim, `sims/glass/`, and the
engine pieces it forces into `kosm`.

## The claim

A stemmed glass of red wine in low sun, struck with a fingernail or rubbed
round the rim with a wet finger. It rings. The wine level sets the pitch.
While it rings, the wine's surface shivers into a ring of standing ripples at
the wall, and those ripples scatter the spectral caustic the glass throws on
the tablecloth. The sound shows up in the light.

Nothing is keyframed or sampled. The ring comes from soda-lime glass's
modal facet, the tuning from red wine's fluid facet, the ripples from the same
fluid's surface tension, the caustic from both substances' `pbr()` and
dispersion. One `Material` per substance, four facets, one world. This is the
thing no other engine does, and the sim is its demonstration.

The inverse problem is the gradient story: **"fill it to play an A."** Wine
height is a `Param`; `ring_hz` is a lens; its derivative with respect to fill
comes from duals, checked against central differences, exactly as
`sims/marble` does for the caustic.

## The scene

- A goblet authored as a surface of revolution in the `build` closure: bowl
  radius `bowl_r` (40 mm), height `bowl_h` (90 mm), wall `wall_t` (1.2 mm),
  stem and foot. A `difference` keeps it hollow (colliders.rs's convex
  decomposition already does this for the marble's cup).
- Wine to height `fill` (0–80 mm). Substance `"red wine"`, added to the
  library (see below).
- A linen tablecloth, one low sun (elevation ~15°) so the caustic stretches
  long and the rosette opens up, a dim warm sky.
- Excitation: `Strike { at, impulse }` (a fingernail tap, a Hertzian contact
  like the marble's) or `Rub { speed, normal_force }` (stick-slip at the rim,
  the glass harmonica). Strike first; Rub is phase 4.

## What exists, what is new

| piece | exists | new |
|---|---|---|
| glass substance, dispersion | `material::named("soda-lime glass")`, `pbr()` with `n_d`, Abbe | nothing |
| spectral caustic | `fluid::caustic`, `light`, `glass.rs` (sphere, lens, convex) | glass.rs has no surface of revolution; the goblet needs a thin-shell SoR shape (enter/exit/normal, generic over `Scalar`) |
| modal audio | `audio.rs`: bars (vcad FEM) and Lamb's sphere; `render` sums damped sinusoids from contacts | **no shell solver.** A goblet's voice is the (n, m) wall modes of a thin shell of revolution; n = 2 is the note |
| liquid loading | none | added mass of the wine on the wall modes: the pitch falls as the glass fills |
| free surface | `fluid::surface` (gravity-wave rings, heightfield) and `HeightGrid` | capillary–gravity dispersion and parametric (Faraday-type) forcing from the wall's motion |
| wine | water is in the library | `"red wine"`: ρ ≈ 990 kg/m³, σ ≈ 0.047 N/m, ν ≈ 1.4e-6 m²/s, n_d ≈ 1.343, and an **absorption spectrum** per band (anthocyanins eat the green, ~520 nm). Marked `Estimated` until each constant has a source |
| volumetric absorption | the tracer does thin-film and SSS | Beer–Lambert through the wine along the refracted path, per band. Without it the caustic is not red |

## The chain, and the equations each link uses

1. **Shell modes.** Axisymmetric thin shell of revolution, Rayleigh–Ritz or a
   1-D FEM along the meridian with a circumferential wavenumber n. Output: a
   `Bank` of (n, m) modes with shapes along the meridian. Check first against
   the closed form for a cylindrical cup and against published wine-glass
   measurements (the empty (2,0) mode of a typical goblet sits in the high
   hundreds of Hz). Lives in `kosm::audio` next to the bar and sphere, or in
   vcad-kernel-acoustics if it belongs there; decide when writing it.
2. **Liquid loading.** Wet wall area adds mass to each mode in proportion to
   how much of that mode's shape is below the surface. First pass: the
   empirical fill law from the wine-glass literature, f(fill)/f(0) =
   [1 + α (ρ_l R / ρ_g t) (fill/H)^4]^-½, fitted α. Second pass: potential
   flow added mass from the mode shape, which should reproduce the law without
   the fit. The fourth power is why the first half-glass barely moves the note
   and the last centimetres move it a lot, which is itself nice to hear.
3. **Ring.** `audio::render` with strike contacts, radiation efficiency from
   the bowl's area. Output `glass.wav`.
4. **Ripples.** The wall's n = 2 motion pumps the free surface at the rim.
   Capillary–gravity dispersion ω² = g k + (σ/ρ) k³ gives the wavelength at the
   forcing frequency (millimetres at several hundred Hz). Above a threshold
   amplitude the surface goes parametrically unstable at half the forcing
   frequency; below it, a driven, damped ring of ripples near the wall that
   follows the n = 2 pattern (four lobes). A `HeightGrid` over the wine's
   disc, stepped with the linearised equation. The damping comes from ν, so
   it decays with the ring.
5. **Light.** The sun refracts through the glass wall, the wine surface
   (flat or rippled), the wine volume (absorbing), and out through the bowl
   onto the cloth. The caustic is `fluid::caustic` generalised to take the
   rippled heightfield as one of its interfaces. The rosette shimmers at the
   ripple frequency, strobed by the camera's shutter.

Every link is a `Lens`: `RingHz`, `Decay`, `RippleAmp`, `CausticContrast`.
`metrics.json` carries all four.

## Knobs

`fill`, `bowl_r`, `bowl_h`, `wall_t`, `strike_z`, `strike_impulse`,
`sun_el`, `sun_az`, plus every substance constant (already `Param`s, so they
are in the run hash).

## Checks

- **Shell solver:** a cylinder matches its closed form to 1%; a goblet's
  empty (2,0) mode lands within the range of published measurements.
- **Fill law:** f(fill) is monotone decreasing, flat to ~40% fill, and the
  full-glass ratio sits within the published spread.
- **Ripple wavelength:** the heightfield's measured wavelength matches the
  dispersion relation at the drive frequency within 5%.
- **Gradient:** d ring_hz / d fill from duals agrees with central differences
  within 1%; a descent on `fill` reaches A4 (440 Hz) from an empty glass for
  a goblet whose empty note is above it.
- **Snapshots:** `glass/ring` (the first 8 modes), `glass/caustic` (image,
  small spp) under `sims/glass/snapshots/`.
- **Invariants:** energy in the ring never increases after the strike;
  wine volume is conserved across the ripple step.

## Order

1. Shell-of-revolution modes + `glass.wav` for the empty goblet. Audible
   progress on day one, and the solver everything else hangs off.
2. Wine substance + liquid loading + the fill sweep (`fill_sweep.wav`: the
   glass played from empty to full) + the gradient to A4.
3. Goblet shape in `glass.rs` + absorbing wine in the tracer: the still
   caustic, `frame.png`.
4. Ripples coupled to the ring; the caustic reads the heightfield;
   `ripple.mp4` at a small spp.
5. Rub excitation (stick-slip), the glass harmonica. Optional.

## Not in scope

- Breaking the glass (fracture at resonance). Tempting, separate project.
- Full coupled FSI of wine and glass: one-way coupling (wall drives surface;
  the surface's back-reaction on the wall is the added mass of link 2) is the
  honest simplification.
- The GPU raster tier. This is an offline sim.

## Risks

- The shell solver is the real work. If it balloons, fall back to the
  empirical (2,0) law for pitch and do shapes later; the demo still works.
- Millimetre ripples need a fine grid over a 80 mm disc (~0.2 mm cells, a
  400² grid). Fine for CPU, but the caustic trace through it needs care to
  stay small in the tests.
- The wine constants are the least certain numbers. Each one gets a source or
  an `Estimated` mark, and `fit` can recover them from a recording later
  (record a real glass, fit σ and the loading α): that is the natural sequel.

## As built (2026-09-26)

`cargo run --release -p kosm-cli -- run glass --out out/` (add
`--ripple_frames 96 --sun_el 58` for the video). Engine pieces:
`kosm::shell` (modes, fill, adjoint, strike, rub), `kosm::fluid::ripple`,
`red wine` / `white wine` and a `surface_tension` constant in the library.

| check | result |
|---|---|
| free cylinder vs Rayleigh's ring | within 2%, n = 2..4 |
| empty goblet (40 × 90 mm, 1.2 mm wall) | (2,0) 619 Hz; partials 2.65×, 4.95× |
| fill law vs French 1983 | within 0.08 at every centimetre |
| d hz / d fill: adjoint vs central differences | equal to 0.1 Hz/m |
| fill for A4 | 72.8 mm, 4 Newton steps on the adjoint |
| ripples at A4 in red wine | λ 1.16 mm, reach 8 mm; a tap moves the waterline 12.5 µm |
| rub (0.08 m/s, 1 N) | sings on (2,0) at 440 Hz, 41.6 µm at the rim; ripple slope 0.18 |

Where it departed from the plan:

- **The gradient is the eigenproblem's adjoint**, dλ = −λ φᵀ dM φ, not duals
  through the solver. Exact, one solve, checked against central differences.
- **Added mass** is local potential flow, ρ r / n per wetted strip, faded by
  tanh(n δ / r) toward the free surface. No fitted α.
- **The goblet is lathed meshes for kosm-render**, not a `glass.rs` shape. The
  tracer tracks one medium, so the wet wall is its own glass-to-wine
  interface surface (index ratio, wine absorption).
- **A full bowl is a cylindrical lens.** Its red light gathers into a line
  beside the stem's shadow, and red wine absorbs most of what crosses the
  bowl.
- **The ripples do not show on the tablecloth.** They shift the caustic
  ~2 mm, under the photon gather radius. `ripple.mp4` shows them where they
  do show: the sun's glint on the band at the side wall, strobed at
  1 + 1/48 of the period. Sun and lens must both clear the rim, so that
  needs a sun above ~50°.
- **n = 1 is dropped.** With the bowl clamped at the stem, the sway modes
  take their frequency from the clamp. The rub excited them hardest until
  they were removed.
- **The rub mostly slips.** A rigid fingertip never catches a rim moving
  microns, so the negative friction slope sustains the note, and a pad
  damping (∝ n²) keeps it on the oval. Skin compliance, and so true stick,
  is not modelled. Faraday's parametric ripples (at ω/2, above a threshold)
  are out of scope, and the rub's slope 0.18 may be past that threshold.
