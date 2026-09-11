# Bake and raster: a live tier that the reference tracer keeps honest

Date: 2026-09-10. Status: design, agreed; two agents build it in parallel
against the interfaces below. Follows `2026-09-09-player-controller-design.md`.

## The claim

The cove's lighting does not change. The sun is fixed, the sand, cliff, door
and reef are static, the sky is a gradient; only the hero, the door once, and
the caustic move. The live tier re-solves that static illumination sixty
times a second with one Monte Carlo sample per pixel, which is why it is
choppy. The path tracer should solve it once, offline, and a rasterizer
should draw the answer stably. The tracer stays as the reference, the baker,
and the thing the picture becomes when the player stands still.

Materials are the contract between the two tiers: one `Material`, one
`gpu()` facet read by the shader and one `pbr()` facet read by the tracer,
from the same fields, and the datasheet ball rendered on both and compared.

## Interfaces (both agents build against these exactly)

### Spectral SH probes: `kosm::light::probes`

```rust
pub const BANDS: usize = kosm::material::BANDS;          // 6, at BANDS_NM
pub const SH: usize = 9;                                 // L2 real SH, ordering: (0,0),(1,-1),(1,0),(1,1),(2,-2),(2,-1),(2,0),(2,1),(2,2)
pub struct ProbeVolume {
    pub origin: [f64; 3],        // metres, corner of cell (0,0,0)
    pub spacing: f64,            // metres, cubic cells
    pub dims: [u32; 3],          // nx, ny, nz
    pub suns: Vec<[f64; 3]>,     // unit vectors toward the sun, one per baked sun position; ≥1
    pub data: Vec<f32>,          // suns × nz × ny × nx × SH × BANDS, x fastest inside a probe block
    pub sky: [[f32; BANDS]; SH], // the sky alone (sun below the horizon), for the sun-independent term
}
impl ProbeVolume {
    pub fn index(&self, sun: usize, ix: u32, iy: u32, iz: u32) -> usize;          // start of that probe's SH×BANDS block
    pub fn sample(&self, sun: f64 /*fractional index*/, p: [f64;3], n: [f64;3]) -> [f32; BANDS]; // irradiance per band, trilinear over probes, linear over suns, SH convolved with the cosine lobe at n
    pub fn write(&self, path) / pub fn read(path)          // out/maps/<sim>/probes.bin: a small header (magic "KPRB", version 1, the fields) then data as little-endian f32
}
```
Units: radiance in the tracer's units (the cove's `Sun.irradiance` and sky are the same units the tracer uses), per band; the shader multiplies by the material's albedo per band and projects to RGB with `Spectrum::to_rgb`'s matrix, which the bake agent exports as `pub fn band_to_rgb() -> [[f32; 3]; BANDS]`.

### GPU material: `kosm::material::GpuMaterial`

```rust
#[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
    pub albedo: [f32; 8],        // BANDS values then two zeros (16-byte alignment)
    pub emission: [f32; 8],      // radiance per band, zeros if none
    pub roughness: f32, pub metallic: f32, pub specular: f32, pub transmission: f32,
    pub ior: f32, pub film_nm: f32, pub film_ior: f32, pub sss_weight: f32,
    pub sss_radius_m: [f32; 3], pub sss_aniso: f32,
}
impl Material { pub fn gpu(&self) -> GpuMaterial }   // from the same fields pbr() reads; a test asserts gpu().albedo projects to pbr().base_color
pub fn library_gpu() -> (Vec<GpuMaterial>, HashMap<String, u32>)   // every named entry, index by canonical name
```

### The caustic map as a texture

`kosm_render::caustics::CausticMap::to_texture(plane_origin, u, v, w_m, h_m, res) -> Vec<[f32; 3]>`: irradiance gathered on a rectangle of a receiver plane, for the door face and a sand patch around the focus.

## The bake (crates/kosm/src/light/probes.rs + sims/rune)

`Probes` is a Lens over a `World` plus a render scene: for each probe, for each
sun position, trace `rays` cosine-weighted hemisphere paths on the CPU
integrator with the sun and sky, accumulate the six-band radiance into SH.
Probes inside solids are marked and filled from their nearest outside
neighbour. `kosm run rune --bake-light` writes `out/maps/cove/probes.bin` at
0.5 m spacing over the playable volume with suns at the level's azimuth and
elevation plus a `bake_suns` knob (default 1; the day is 7 elevations from 5°
to 60° along the authored azimuth's arc). Determinism: seeded; a snapshot
pins a few probes.

## The raster tier (crates/kosm-view/src/raster/ + sims/rune/game.rs)

A wgpu pipeline beside the ride tier: meshes from the cove's solids
(tessellated once, with smoothed normals), per-instance model matrix and
material index, the material buffer from `library_gpu()`, a 2048² sun shadow
map with PCF, probes sampled per pixel with the pixel's normal, direct sun
with Lambert plus a GGX highlight from roughness/specular/metallic, thin-film
Fresnel tint when `film_nm > 0`, wrap lighting when `sss_weight > 0`, the
pool tier's water shader for the sea at `sea_z` with the swell, the caustic
texture on the door and the sand patch, emissives for the rim and glint, the
lens drawn as its brass ring plus a Fresnel highlight and the caustic. The
projection honours the rig's `Projection`. Tone map with the meter's exposure,
sRGB out.

**Settle into the reference.** The path tracer keeps running on the rig's
camera in a background thread at the budget's size while the hero is still;
`blend = clamp((spp − 4) / 24, 0, 1)` mixes the resolved traced frame over the
raster frame; any camera or world move resets it. The pace line prints the
blend.

## Tests

- `gpu()` round trip; `library_gpu` covers every name.
- Probe volume: a white furnace (unit sky, no sun) gives irradiance π per
  band at every probe; a probe under an overhang sees less sun than one in
  the open; `sample` is continuous across cells; read/write round-trip.
- Parity: every material's datasheet ball rendered on the raster tier under
  a probe volume baked for the studio rig versus the tracer's ball, mean
  absolute difference under a per-material tolerance (report the numbers;
  glass and metals get a looser one and say why).
- The cove: the raster frame at the solved pose versus the traced frame, mean
  difference on the sand and the door face under a tolerance; the settle
  blend reaches 1 within 3 s standing and drops to 0 on a walk.
- Pace: headless `--walk`: raster ≥ 60 fps at 1280×720, presented latency
  < 30 ms.

## Order

Bake and `gpu()` first (the raster agent starts on the pipeline with a
synthetic volume and the library, then reads the real bake when it exists).
