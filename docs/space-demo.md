# Orbital: the space scene

Run from this Kosm checkout:

```sh
python3 scripts/fetch-earth-assets.py
cargo run -p kosm-view -- --space
```

The setup command downloads roughly 35 MB of attributed source imagery, verifies fixed SHA-256 digests, then builds a local, ignored `assets/earth` cache of the prepared tiles. It requires Python with Pillow and NumPy. Set `KOSM_EARTH_ASSETS` to use an existing cache elsewhere. The executable never makes network requests.

The scene is `levels/space.loon`. It resolves through `AuthoredScene`, then feeds a dedicated `kosm_spike::space` computation and the native viewer’s `space.wgsl` renderer. Existing pool and splash entry points retain their behavior.

## Controls

- Drag the scene to orbit the camera; scroll to dolly.
- Space pauses/resumes; H hides/shows instruments.
- Select 1×, 30×, 100× or 500× time. The solver still takes steps no larger than one simulated second.
- Scrub the recording to inspect an earlier state. Resuming or maneuvering from that state truncates its old future and starts a new branch.
- Prograde/retrograde apply instantaneous ±10 m/s along the current velocity. Δv, osculating apsides, and the overview trajectory update from the changed state.
- Earth overview shows a numerically propagated orbit, with hidden segments occluded by Earth. Predictions refresh every ten simulated seconds or on a new branch.
- Save frame writes the GPU scene to `out/space-<seconds>.png` (scene only, without UI overlays).

## Physics and rendering boundary

Translation is double-precision ECI SI state, fourth-order Runge–Kutta, central Earth gravity plus the unnormalized J2 coefficient. Earth’s rotation affects the texture, not the inertial orbit. Osculating apsides and period use the two-body elements of the instantaneous state; J2 means they change over an orbit. A near-circular 420 km, 51.6° initial orbit is specified by the Loon parameters.

Attitude integrates the coupled torque-free Euler equations and quaternion kinematics with RK4 and quaternion normalization. The inertia approximation is a uniform spacecraft bus; the rendered appendages have no separate mass or flexibility. This is a demonstrator, not an ephemeris or flight-dynamics product: no atmospheric drag, higher harmonics, third-body gravity, gravity-gradient torque, thrust duration, propellant depletion, solar pressure, or control system. Propagation pauses if altitude reaches 80 km rather than continuing through a missing reentry model. The Sun direction is fixed for the demonstration, not tied to a calendar epoch.

The GPU traces camera-relative spacecraft geometry in metres and Earth in kilometres. NASA surface imagery uses a 20480 × 10240 tiled raster (~1.96 km at the equator), about five times the linear resolution of the original 4K map. A 32-layer GPU cache (~172 MiB including mips), a bounded decoding worker, and a global fallback keep the entire raster off the GPU. Mips average RGB in linear light and preserve linear water coverage. Tile requests follow a sampled camera frustum; rapid motion may temporarily show the lower-resolution fallback.

Clouds use separate 8K coverage, a spherical 1.2–11 km density volume, lower decks and taller convection, periodic 3D erosion, view-ray integration, and Sun-aligned surface shadows. Height and density structure are procedural interpretations of a frozen image, not measured weather. Internal cloud multiple scattering uses an inexpensive approximation; clouds and clear air are composed at an opacity-weighted cloud depth rather than fully coupled transport.

The atmosphere uses a 512 × 128 solar-transmission LUT integrated in double precision with exponential Rayleigh/Mie profiles and a triangular ozone layer. The same LUT illuminates ground, clouds, and view-ray scattering. View transport integrates segment attenuation; diffuse skylight is a low-order closure, not a validated Bruneton or Hillaire multiple-scattering implementation. Numerical tests compare solar transmission against an analytical vertical column and higher-resolution quadrature.

Ocean reflections use a separate water mask, Fresnel reflectance, and a microfacet slope distribution at a fixed 5 m/s reference wind. Spacecraft materials use GGX, self-shadowing, and approximate Earth reflection. Radiance/exposure remains artist-scaled; there is no radiometric calibration, HDR history buffer, temporal antialiasing, terrain displacement, live weather, or general-purpose path tracing. Stars are a dim procedural backdrop, not a catalog.

The authored bus dimensions and array envelope feed both CAD inspection and rendering. Telescope, dish, radiator and thrusters are renderer details, not additional authored CAD roots.

## Verification

```sh
cargo test -p kosm-spike --lib space::tests
cargo test -p kosm-view space::
cargo run -p kosm-view -- --space --shot=/tmp/space.png
cargo run -p kosm-view -- --space --space-time=2800 --shot=/tmp/space-night.png
cargo run -p kosm-view -- --space --space-overview --shot=/tmp/space-earth.png
```

The tests check circular orbit closure (<1 cm after one period), force versus potential gradient, J2 energy and axial angular momentum conservation, torque-free rotational invariants, and the expected apogee increase from a prograde impulse. `--shot` warms the renderer for 120 frames to allow tile loading, captures the frozen initial frame (or the propagated `--space-time` state), then exits. The GPU validates the shader and binding layout at launch.

Imagery sources, attribution, and cache layout: `docs/earth-assets.md`.
