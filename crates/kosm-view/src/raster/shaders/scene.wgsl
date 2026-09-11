// The raster tier's one shading pass.
//
// Six bands in, RGB out. The material's albedo, the sun's irradiance and the
// probes' irradiance are all spectral, they are multiplied band by band, and
// only the product is projected to the film — which is what makes this tier
// comparable with the spectral path tracer that baked the probes.
//
//   direct   = E_sun · pcss(p, n) · (albedo/π · wrap(n·l) + GGX)  [6 bands]
//            + caustic irradiance where a receiver quad lands
//   indirect = probes(sun, p, n) · ao                             [6 bands]
//   colour   = M · (albedo ⊙ (direct + indirect)) + emission + glow
//   out      = mix(colour, sky(view), 1 − e^{−τ})                 linear HDR
//
// **and stops there.** The exposure, the lens's `cos⁴`, the bloom and the
// tonemap are `shaders/post.wgsl`, which is `kosm_render::post::Post` in
// WGSL — the same chain the reference tracer puts its own film through, so
// the settle blend has no seam.

const PI: f32 = 3.14159265359;
const BANDS: u32 = 6u;
const SH: u32 = 9u;

// ---- what a frame carries ---------------------------------------------------

struct Uniforms {
    view_proj: mat4x4<f32>,
    sun_view_proj: mat4x4<f32>,
    // xyz eye, w exposure (the authored one already times the meter's)
    eye: vec4<f32>,
    // xyz toward the sun, w its angular radius in radians
    sun_dir: vec4<f32>,
    // the sun's irradiance, bands 0..3 and 4..5 (zw of the second is spare)
    sun_irr_a: vec4<f32>,
    sun_irr_b: vec4<f32>,
    // xyz the probe lattice's origin in metres, w its spacing
    probe_origin: vec4<f32>,
    // nx, ny, nz, how many baked suns
    probe_dims: vec4<u32>,
    // x the fractional sun index, y the shadow texel in metres, z the shadow
    // map's side, w the time in seconds
    knobs: vec4<f32>,
    // the sea: z, slope, waterline y, reach
    sea: vec4<f32>,
    // the swell: a1, l1, a2, l2, all metres
    swell: vec4<f32>,
    // its angles and speed: angle1, angle2, speed, the water's index
    swell_angle: vec4<f32>,
    // the sea's absorption per RGB metre, w the *water's* material index
    sea_absorb: vec4<f32>,
    // x the *seabed's* material index, y whether there is a sea at all, zw spare
    sea_flags: vec4<u32>,
    // two receiver rectangles: origin.xyz + w unused, then the u and v edges
    caustic_origin: array<vec4<f32>, 2>,
    caustic_u: array<vec4<f32>, 2>,
    caustic_v: array<vec4<f32>, 2>,
    // x how many quads are live, y their plane tolerance in metres
    caustic_flags: vec4<f32>,
    // the six-band → linear RGB projection, one row per band
    band_to_rgb: array<vec4<f32>, 6>,
    // clip → world, for the sky pass's view ray and the AO pass's position
    inv_view_proj: mat4x4<f32>,
    // the analytic sky: turbidity, its normaliser, its mean radiance, and the
    // sun's angular radius (the model is clamped at it; the disc is the sun's)
    sky_a: vec4<f32>,
    // the ground half's linear-RGB albedo, w > 0.5 when there is a sky model
    // at all — without one every sky lookup falls back to the probe read
    sky_ground: vec4<f32>,
    // the haze and the occlusion: density per metre, its scale height, the
    // AO radius in metres, the AO strength
    air: vec4<f32>,
    // the frame: width, height, and the rig's own tan(fov/2) across and up
    screen: vec4<f32>,
    // the sun's shadow frustum in world metres: its width, its depth range,
    // the tangent of the sun's angular radius, and one texel of its width
    shadow_m: vec4<f32>,
    // the sea's scattering per RGB metre, w the foam's depth in metres. What
    // the water gives back rather than eats; `water.rs` is the argument.
    sea_scatter: vec4<f32>,
    // x the foam's strength, y the wet band's width in metres, zw spare
    sea_shore: vec4<f32>,
    // the foam's linear-RGB albedo, from the library's `sea foam`
    foam_albedo: vec4<f32>,
};

struct GpuMaterial {
    albedo: array<vec4<f32>, 2>,     // six bands then two zeros
    emission: array<vec4<f32>, 2>,
    roughness: f32,
    metallic: f32,
    specular: f32,
    transmission: f32,
    ior: f32,
    film_nm: f32,
    film_ior: f32,
    sss_weight: f32,
    sss_radius_m: vec3<f32>,
    sss_aniso: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> materials: array<GpuMaterial>;
// suns × nz × ny × nx × SH × BANDS, x fastest inside a probe block — the
// baker's own layout, uploaded verbatim. The sky is already in it: the bake's
// hemisphere rays carry the environment, and `ProbeVolume::sky` is a separate
// unoccluded reference term that `sample` does not add either.
@group(0) @binding(2) var<storage, read> probes: array<f32>;
// One bit per probe, x fastest: whether it sits inside a solid. A probe buried
// in the cliff carries no light and would drag the sand in front of it to
// black, so it drops out of the trilinear weights and the rest are
// renormalised — which is exactly what `ProbeVolume::sample_sh` does.
@group(0) @binding(7) var<storage, read> probe_inside: array<u32>;
@group(0) @binding(3) var shadow_tex: texture_depth_2d;
@group(0) @binding(4) var shadow_smp: sampler_comparison;
@group(0) @binding(5) var caustic_tex: texture_2d_array<f32>;
@group(0) @binding(6) var caustic_smp: sampler;
// The half-resolution ambient occlusion, blurred. One channel, one at every
// pixel a bake had nothing to say about; see `shaders/ao.wgsl`.
@group(0) @binding(8) var ao_tex: texture_2d<f32>;
@group(0) @binding(9) var ao_smp: sampler;

fn band(a: array<vec4<f32>, 2>, i: u32) -> f32 {
    if i < 4u { return a[0][i]; }
    return a[1][i - 4u];
}

fn sun_band(i: u32) -> f32 {
    if i < 4u { return u.sun_irr_a[i]; }
    return u.sun_irr_b[i - 4u];
}

// ---- the vertex stage -------------------------------------------------------

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) wpos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) @interpolate(flat) mat: u32,
    @location(3) @interpolate(flat) kind: u32,
    @location(4) @interpolate(flat) glow: vec4<f32>,
};

// The swell, in closed form. `water.rs::Sea::height` is this function in
// Rust, and the test that holds them together is the one that says the
// normal is the gradient of the height.
fn swell_height(x: f32, y: f32) -> f32 {
    let t = u.knobs.w * u.swell_angle.z;
    let c1 = cos(u.swell_angle.x);
    let s1 = sin(u.swell_angle.x);
    let c2 = cos(u.swell_angle.y);
    let s2 = sin(u.swell_angle.y);
    let k1 = 6.28318530718 / max(u.swell.y, 1e-3);
    let k2 = 6.28318530718 / max(u.swell.w, 1e-3);
    return u.swell.x * sin(k1 * (x * c1 + y * s1) - t)
         + u.swell.z * sin(k2 * (x * c2 + y * s2) + 1.7 - t);
}

fn swell_normal(x: f32, y: f32) -> vec3<f32> {
    let t = u.knobs.w * u.swell_angle.z;
    let c1 = cos(u.swell_angle.x);
    let s1 = sin(u.swell_angle.x);
    let c2 = cos(u.swell_angle.y);
    let s2 = sin(u.swell_angle.y);
    let k1 = 6.28318530718 / max(u.swell.y, 1e-3);
    let k2 = 6.28318530718 / max(u.swell.w, 1e-3);
    let a1 = u.swell.x * k1 * cos(k1 * (x * c1 + y * s1) - t);
    let a2 = u.swell.z * k2 * cos(k2 * (x * c2 + y * s2) + 1.7 - t);
    return normalize(vec3<f32>(-(a1 * c1 + a2 * c2), -(a1 * s1 + a2 * s2), 1.0));
}

// ---- the shore -------------------------------------------------------------
//
// `water.rs::Sea::depth`, `::foam` and `::wet` are these three functions in
// Rust and the tests there are what holds the two copies together. The foam
// and the wet band are **raster-only decorations**: the reference tracer
// draws one smooth opaque sea and one dry beach, so wherever the tiers are
// differenced these two regions are excluded and the parity test says so.

// How much water stands over the seabed under a point of the surface. Never
// negative: where the trough is under the sand there is none, and that is
// where the foam is.
fn sea_depth(x: f32, y: f32) -> f32 {
    let bed = u.sea.x + u.sea.y * (y - u.sea.z);
    return max(u.sea.x + swell_height(x, y) - bed, 0.0);
}

// The foam, 0..1. A band where the water has run out, whose edge is displaced
// by the swell's own phase so it breathes with the sets instead of lying on
// the beach as a contour line, plus the crests of the swell while it is still
// shallow — which is a wave breaking.
fn foam_at(x: f32, y: f32) -> f32 {
    let strength = u.sea_shore.x;
    if strength <= 0.0 {
        return 0.0;
    }
    let e = max(u.sea_scatter.w, 1e-4);
    let amp = max(u.swell.x + u.swell.z, 1e-4);
    let h = swell_height(x, y);
    let d = sea_depth(x, y);
    let edge = max(e * (1.0 + 0.6 * h / amp), 1e-4);
    let band = 1.0 - smoothstep(0.0, edge, d);
    let crest = smoothstep(0.45 * amp, 0.95 * amp, h)
        * (1.0 - smoothstep(8.0 * e, 30.0 * e, d));
    return clamp(strength * max(band, crest), 0.0, 1.0);
}

// How wet the ground at a point is: one at and under the water, nothing a
// band above the swell's current top.
fn wet_at(p: vec3<f32>) -> f32 {
    if u.sea_flags.y == 0u {
        return 0.0;
    }
    let level = u.sea.x + swell_height(p.x, p.y);
    if u.sea_shore.y <= 0.0 {
        return select(0.0, 1.0, p.z <= level);
    }
    return 1.0 - smoothstep(0.0, u.sea_shore.y, p.z - level);
}

// `kosm::material::Material::wet`'s rule, as a factor: albedo x0.6, roughness
// x0.75, and the dielectric highlight a wet grain has and a dry one does not.
// One entry in the library, one rule, applied here to the pixels the sea
// actually reaches rather than to a second material.
fn wetted(m: GpuMaterial, w: f32) -> GpuMaterial {
    var o = m;
    if w <= 0.0 {
        return o;
    }
    let k = 1.0 - 0.4 * w;
    o.albedo[0] = m.albedo[0] * k;
    o.albedo[1] = m.albedo[1] * k;
    o.roughness = m.roughness * (1.0 - 0.25 * w);
    o.specular = mix(m.specular, max(m.specular, 0.5), w);
    return o;
}

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) m0: vec4<f32>,
    @location(3) m1: vec4<f32>,
    @location(4) m2: vec4<f32>,
    @location(5) m3: vec4<f32>,
    @location(6) ids: vec4<u32>,
    @location(7) glow: vec4<f32>,
) -> VsOut {
    let m = mat4x4<f32>(m0, m1, m2, m3);
    var world = (m * vec4<f32>(pos, 1.0)).xyz;
    var n = normalize(mat3x3<f32>(m0.xyz, m1.xyz, m2.xyz) * nrm);
    // The sea's lattice is flat on the way up and gets its swell here, so a
    // moving surface costs one uniform rather than a re-upload every frame.
    if ids.y == 1u {
        world.z = world.z + swell_height(world.x, world.y);
        n = swell_normal(world.x, world.y);
    }
    var o: VsOut;
    o.clip = u.view_proj * vec4<f32>(world, 1.0);
    o.wpos = world;
    o.nrm = n;
    o.mat = ids.x;
    o.kind = ids.y;
    o.glow = glow;
    return o;
}

// ---- the light --------------------------------------------------------------

fn sh_basis(n: vec3<f32>) -> array<f32, 9> {
    return array<f32, 9>(
        0.2820950,
        0.4886030 * n.y,
        0.4886030 * n.z,
        0.4886030 * n.x,
        1.0925480 * n.x * n.y,
        1.0925480 * n.y * n.z,
        0.3153920 * (3.0 * n.z * n.z - 1.0),
        1.0925480 * n.x * n.z,
        0.5462740 * (n.x * n.x - n.y * n.y),
    );
}

// The basis already scaled by the cosine lobe's per-band weights: π, 2π/3,
// π/4. `probes.rs::sh_cosine` is the same nine numbers in Rust, and the test
// `the_cosine_convolution_is_the_integral` holds them to ∫L(ω)max(0,ω·n)dω.
fn sh_cosine(n: vec3<f32>) -> array<f32, 9> {
    var y = sh_basis(n);
    let a0 = PI;
    let a1 = 2.0 * PI / 3.0;
    let a2 = PI / 4.0;
    return array<f32, 9>(
        y[0] * a0,
        y[1] * a1, y[2] * a1, y[3] * a1,
        y[4] * a2, y[5] * a2, y[6] * a2, y[7] * a2, y[8] * a2,
    );
}

// The probe read, in two halves — exactly as `ProbeVolume` splits it.
//
// `probe_sh` interpolates the nine coefficients at a point: trilinear over the
// lattice, linear over the sun's fractional index, probes inside solids
// dropped and the rest renormalised. `convolve` turns that block into
// irradiance at a normal. The shading does not take this path any more — see
// `probe_irradiance2`, which reads the lattice once for both of a fragment's
// lobes and never builds the block — and the two fallbacks that want a sky
// out of the probes (no sky model bound) still do.
fn probe_sh(p: vec3<f32>) -> array<f32, 54> {
    var out: array<f32, 54>;
    for (var i = 0u; i < 54u; i = i + 1u) { out[i] = 0.0; }
    let nx = u.probe_dims.x;
    let ny = u.probe_dims.y;
    let nz = u.probe_dims.z;
    let ns = u.probe_dims.w;
    if nx == 0u || ny == 0u || nz == 0u || ns == 0u {
        return out;
    }
    let sp = max(u.probe_origin.w, 1e-6);
    let t = (p - u.probe_origin.xyz) / sp;
    let last = vec3<f32>(f32(nx - 1u), f32(ny - 1u), f32(nz - 1u));
    let cl = clamp(floor(t), vec3<f32>(0.0), max(last - vec3<f32>(1.0), vec3<f32>(0.0)));
    let f = clamp(t - cl, vec3<f32>(0.0), vec3<f32>(1.0));
    let base = vec3<u32>(u32(cl.x), u32(cl.y), u32(cl.z));

    let si = clamp(u.knobs.x, 0.0, f32(ns - 1u));
    let s0 = u32(floor(si));
    let s1 = min(s0 + 1u, ns - 1u);
    let ts = si - floor(si);
    let per_probe = SH * BANDS;
    let probes_per_sun = nx * ny * nz;

    var total = 0.0;
    for (var c = 0u; c < 8u; c = c + 1u) {
        let dx = c & 1u;
        let dy = (c >> 1u) & 1u;
        let dz = (c >> 2u) & 1u;
        var w = 1.0;
        if nx <= 1u { if dx == 1u { w = 0.0; } } else if dx == 0u { w = w * (1.0 - f.x); } else { w = w * f.x; }
        if ny <= 1u { if dy == 1u { w = 0.0; } } else if dy == 0u { w = w * (1.0 - f.y); } else { w = w * f.y; }
        if nz <= 1u { if dz == 1u { w = 0.0; } } else if dz == 0u { w = w * (1.0 - f.z); } else { w = w * f.z; }
        if w <= 0.0 { continue; }
        let ix = min(base.x + dx, nx - 1u);
        let iy = min(base.y + dy, ny - 1u);
        let iz = min(base.z + dz, nz - 1u);
        let flat = (iz * ny + iy) * nx + ix;
        if ((probe_inside[flat / 32u] >> (flat % 32u)) & 1u) == 1u { continue; }
        total = total + w;
        let a = (s0 * probes_per_sun + flat) * per_probe;
        if ts <= 0.0 {
            for (var k = 0u; k < 54u; k = k + 1u) {
                out[k] = out[k] + w * probes[a + k];
            }
        } else {
            let b = (s1 * probes_per_sun + flat) * per_probe;
            for (var k = 0u; k < 54u; k = k + 1u) {
                out[k] = out[k] + w * mix(probes[a + k], probes[b + k], ts);
            }
        }
    }
    if total <= 1e-9 {
        return out;
    }
    for (var k = 0u; k < 54u; k = k + 1u) {
        out[k] = out[k] / total;
    }
    return out;
}

// One interpolated block, convolved with the cosine lobe at `n`. Six bands of
// irradiance out, clamped at zero — L2 ringing can undershoot and irradiance
// cannot be negative, which is where `ProbeVolume::sample` clamps too.
fn convolve(sh: array<f32, 54>, n: vec3<f32>) -> array<f32, 6> {
    var block = sh;
    let k = sh_cosine(n);
    var out = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var q = 0u; q < SH; q = q + 1u) {
        for (var b = 0u; b < BANDS; b = b + 1u) {
            out[b] = out[b] + k[q] * block[q * BANDS + b];
        }
    }
    for (var b = 0u; b < BANDS; b = b + 1u) {
        out[b] = max(out[b], 0.0);
    }
    return out;
}

fn probe_irradiance(p: vec3<f32>, n: vec3<f32>) -> array<f32, 6> {
    return convolve(probe_sh(p), n);
}

// Six bands of irradiance at two directions, as two halves of three.
struct Irradiance2 {
    n_lo: vec3<f32>,
    n_hi: vec3<f32>,
    r_lo: vec3<f32>,
    r_hi: vec3<f32>,
};

// **The same two numbers as `convolve(probe_sh(p), n)` and
// `convolve(probe_sh(p), r)`, summed in the other order.**
//
// The cosine lobe is linear in the coefficients, so
//
//   Σ_q k_q(n) · (Σ_c w_c L_cq) / Σw  =  (Σ_c w_c Σ_q k_q(n) L_cq) / Σw,
//
// and the right-hand side never builds the interpolated block. That block is
// fifty-four floats written through a loop the compiler cannot unroll, which
// a GPU keeps in thread memory rather than in registers — and reading it, and
// copying it into `convolve` twice, was the most expensive thing in the
// frame: taking the probe read out altogether took a third off the shading
// pass. This keeps twelve accumulators and the two lobes' nine weights each,
// and reads every coefficient exactly once. `probe_sh` stays for the two
// fallbacks that want the block itself.
fn probe_irradiance2(p: vec3<f32>, n: vec3<f32>, r: vec3<f32>) -> Irradiance2 {
    var o: Irradiance2;
    o.n_lo = vec3<f32>(0.0);
    o.n_hi = vec3<f32>(0.0);
    o.r_lo = vec3<f32>(0.0);
    o.r_hi = vec3<f32>(0.0);
    let nx = u.probe_dims.x;
    let ny = u.probe_dims.y;
    let nz = u.probe_dims.z;
    let ns = u.probe_dims.w;
    if nx == 0u || ny == 0u || nz == 0u || ns == 0u {
        return o;
    }
    let sp = max(u.probe_origin.w, 1e-6);
    let t = (p - u.probe_origin.xyz) / sp;
    let last = vec3<f32>(f32(nx - 1u), f32(ny - 1u), f32(nz - 1u));
    let cl = clamp(floor(t), vec3<f32>(0.0), max(last - vec3<f32>(1.0), vec3<f32>(0.0)));
    let f = clamp(t - cl, vec3<f32>(0.0), vec3<f32>(1.0));
    let base = vec3<u32>(u32(cl.x), u32(cl.y), u32(cl.z));

    let si = clamp(u.knobs.x, 0.0, f32(ns - 1u));
    let s0 = u32(floor(si));
    let s1 = min(s0 + 1u, ns - 1u);
    let ts = si - floor(si);
    let per_probe = SH * BANDS;
    let probes_per_sun = nx * ny * nz;
    let kn = sh_cosine(n);
    let kr = sh_cosine(r);

    var total = 0.0;
    for (var c = 0u; c < 8u; c = c + 1u) {
        let dx = c & 1u;
        let dy = (c >> 1u) & 1u;
        let dz = (c >> 2u) & 1u;
        var w = 1.0;
        if nx <= 1u { if dx == 1u { w = 0.0; } } else if dx == 0u { w = w * (1.0 - f.x); } else { w = w * f.x; }
        if ny <= 1u { if dy == 1u { w = 0.0; } } else if dy == 0u { w = w * (1.0 - f.y); } else { w = w * f.y; }
        if nz <= 1u { if dz == 1u { w = 0.0; } } else if dz == 0u { w = w * (1.0 - f.z); } else { w = w * f.z; }
        if w <= 0.0 { continue; }
        let ix = min(base.x + dx, nx - 1u);
        let iy = min(base.y + dy, ny - 1u);
        let iz = min(base.z + dz, nz - 1u);
        let flat = (iz * ny + iy) * nx + ix;
        if ((probe_inside[flat / 32u] >> (flat % 32u)) & 1u) == 1u { continue; }
        total = total + w;
        let a = (s0 * probes_per_sun + flat) * per_probe;
        let b = (s1 * probes_per_sun + flat) * per_probe;
        for (var q = 0u; q < SH; q = q + 1u) {
            let i = a + q * BANDS;
            var lo = vec3<f32>(probes[i], probes[i + 1u], probes[i + 2u]);
            var hi = vec3<f32>(probes[i + 3u], probes[i + 4u], probes[i + 5u]);
            if ts > 0.0 {
                let j = b + q * BANDS;
                lo = mix(lo, vec3<f32>(probes[j], probes[j + 1u], probes[j + 2u]), ts);
                hi = mix(hi, vec3<f32>(probes[j + 3u], probes[j + 4u], probes[j + 5u]), ts);
            }
            let wn = w * kn[q];
            let wr = w * kr[q];
            o.n_lo = o.n_lo + wn * lo;
            o.n_hi = o.n_hi + wn * hi;
            o.r_lo = o.r_lo + wr * lo;
            o.r_hi = o.r_hi + wr * hi;
        }
    }
    if total > 1e-9 {
        let inv = 1.0 / total;
        o.n_lo = o.n_lo * inv;
        o.n_hi = o.n_hi * inv;
        o.r_lo = o.r_lo * inv;
        o.r_hi = o.r_hi * inv;
    }
    // L2 ringing can undershoot and irradiance cannot be negative — the same
    // clamp `convolve` applies, per band, after the whole sum
    o.n_lo = max(o.n_lo, vec3<f32>(0.0));
    o.n_hi = max(o.n_hi, vec3<f32>(0.0));
    o.r_lo = max(o.r_lo, vec3<f32>(0.0));
    o.r_hi = max(o.r_hi, vec3<f32>(0.0));
    return o;
}

// A sixteen-point Poisson-ish disc, used twice: once to look for blockers and
// once to filter across the penumbra they imply. Spiralled rather than random
// so the pattern has no clumps and needs no per-pixel rotation to hide them —
// a rotation would be temporal noise on a tier whose whole claim is that it
// does not flicker.
fn disc16(i: u32) -> vec2<f32> {
    let f = (f32(i) + 0.5) / 16.0;
    let r = sqrt(f);
    // the golden angle: successive taps land two fifths of a turn apart
    let a = f32(i) * 2.39996323;
    return vec2<f32>(r * cos(a), r * sin(a));
}

// **PCSS: the sun has a size, so its shadows have an edge that widens.**
//
// The 3×3 PCF this replaced filtered over one texel whatever the geometry,
// which draws every shadow with the same hard edge — a hero's shadow at its
// own feet and the cliff's thrown forty metres both. The reference tracer
// samples the sun's *disc*, so its penumbra grows with the distance from the
// blocker, and at the door the two tiers plainly disagreed.
//
// The construction is the standard one, stated in world metres because that
// is where the sun's angular radius lives, and `shadow_m` carries the
// frustum's own metres so nothing has to be recovered from a bias:
//
//   1. average the depth of the blockers within a search radius,
//   2. `penumbra = (receiver − blocker) · tan α`, α the disc's radius,
//   3. filter over that, in the map's own uv.
fn sun_shadow(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let c = u.sun_view_proj * vec4<f32>(p, 1.0);
    if c.w <= 0.0 {
        return 1.0;
    }
    let ndc = c.xyz / c.w;
    if abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let cos_l = clamp(dot(n, u.sun_dir.xyz), 0.0, 1.0);
    let tan_l = sqrt(max(1.0 - cos_l * cos_l, 0.0)) / max(cos_l, 0.05);
    let side = max(u.knobs.z, 1.0);
    let texel = 1.0 / side;
    // one texel of world, widened by the slope's rise across it, already in
    // the map's own clip depth
    let bias = u.knobs.y * (1.0 + min(tan_l, 8.0));
    let extent = max(u.shadow_m.x, 1e-6);
    let depth_m = max(u.shadow_m.y, 1e-6);
    let tan_a = u.shadow_m.z;
    // world metres of penumbra, as a uv radius on the map
    let per_m = 1.0 / extent;

    // ---- 1. how deep are the blockers -------------------------------------
    // The search radius is the widest penumbra worth looking for — a blocker
    // the whole frustum away — capped at eight texels, because past that the
    // taps scatter into geometry that has nothing to do with this pixel.
    let search = clamp(tan_a * depth_m * per_m, texel, 8.0 * texel);
    // **The blocker search needs the bias of its own radius, not of a texel.**
    // A lit plane that leans away from the sun rises across the search disc by
    // the slope times its *width*; tested against a one-texel bias, half the
    // taps on a flat sunlit cliff come back as blockers, the penumbra estimate
    // follows the sixteen-tap spiral, and the wall wears a set of faint
    // diagonal stripes. Scaling the same slope-scaled bias by how many texels
    // wide the search is is the whole fix.
    let search_bias = bias * max(search / texel, 1.0);
    var blocker = 0.0;
    var found = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        // `textureLoad` and not a second sampler: PCSS wants a blocker's
        // *depth*, which a comparison sampler will not give, and a nearest
        // read of a depth map is exactly what a blocker search wants anyway.
        let q = clamp(
            vec2<i32>((uv + disc16(i) * search) * side),
            vec2<i32>(0, 0),
            vec2<i32>(i32(side) - 1, i32(side) - 1),
        );
        let d = textureLoad(shadow_tex, q, 0);
        if d < ndc.z - search_bias {
            blocker = blocker + d;
            found = found + 1.0;
        }
    }
    if found < 0.5 {
        return 1.0;                     // nothing between here and the sun
    }
    blocker = blocker / found;

    // ---- 2. how wide is the penumbra --------------------------------------
    let gap_m = max((ndc.z - blocker) * depth_m, 0.0);
    // never narrower than a texel: a contact shadow filtered over nothing is
    // the aliased staircase a PCF exists to hide
    let radius = clamp(tan_a * gap_m * per_m, texel, 12.0 * texel);

    // ---- 3. filter over it ------------------------------------------------
    var sum = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        sum = sum + textureSampleCompareLevel(
            shadow_tex, shadow_smp, uv + disc16(i) * radius, ndc.z - bias);
    }
    return sum / 16.0;
}

// The caustic's extra irradiance at a point on a receiver plane, linear RGB.
// Every fragment tests both quads: three dot products each, against the cost
// of deciding per instance which surface belongs to which rectangle.
fn caustic_at(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var out = vec3<f32>(0.0);
    let live = u32(u.caustic_flags.x);
    for (var q = 0u; q < 2u; q = q + 1u) {
        if q >= live { break; }
        let o = u.caustic_origin[q].xyz;
        let uu = u.caustic_u[q].xyz;
        let vv = u.caustic_v[q].xyz;
        let pn = normalize(cross(uu, vv));
        let rel = p - o;
        // only surfaces lying in the rectangle's own plane, facing the same
        // way: a caustic is deposited on the receiver, not on what is in
        // front of it
        if abs(dot(rel, pn)) > u.caustic_flags.y { continue; }
        if dot(n, pn) < 0.2 { continue; }
        let a = dot(rel, uu) / max(dot(uu, uu), 1e-9);
        let b = dot(rel, vv) / max(dot(vv, vv), 1e-9);
        if a < 0.0 || a > 1.0 || b < 0.0 || b > 1.0 { continue; }
        out = out + textureSampleLevel(caustic_tex, caustic_smp, vec2<f32>(a, b), i32(q), 0.0).rgb;
    }
    return out;
}

// ---- the surface ------------------------------------------------------------

fn f0_of(m: GpuMaterial, albedo_rgb: vec3<f32>) -> vec3<f32> {
    if m.metallic > 0.5 {
        return albedo_rgb;
    }
    // a dielectric's normal-incidence reflectance from its index, scaled by
    // the material's own `specular` knob — the same 0.08·specular convention
    // the tracer's `Pbr` uses at `ior = 1.5`
    let r = (m.ior - 1.0) / (m.ior + 1.0);
    return vec3<f32>(clamp(r * r * 4.0 * m.specular, 0.0, 1.0));
}

fn fresnel(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// ---- the thin film, ported from kosm-render's bsdf.wgsl ---------------------

fn tf_f0_of(n1: f32, n2: f32) -> f32 {
    let r = (n1 - n2) / (n1 + n2);
    return r * r;
}

fn tf_ior_of_f0(f0: f32) -> f32 {
    let s = sqrt(clamp(f0, 0.0, 0.9999));
    return (1.0 + s) / (1.0 - s);
}

fn tf_schlick(f0: f32, cos_theta: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn xyz_to_linear_srgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        3.2404542 * c.x - 1.5371385 * c.y - 0.4985314 * c.z,
        -0.9692660 * c.x + 1.8760108 * c.y + 0.0415560 * c.z,
        0.0556434 * c.x - 0.2040259 * c.y + 1.0572252 * c.z,
    );
}

fn tf_sensitivity(opd: f32, shift: vec3<f32>) -> vec3<f32> {
    let two_pi = 2.0 * PI;
    let phase = two_pi * opd * 1.0e-9;
    let val = vec3<f32>(5.4856e-13, 4.4201e-13, 5.2481e-13);
    let pos = vec3<f32>(1.6810e+06, 1.7953e+06, 2.2084e+06);
    let vr = vec3<f32>(4.3278e+09, 9.3046e+09, 6.6121e+09);
    var xyz = val * sqrt(two_pi * vr) * cos(pos * phase + shift) * exp(-vr * phase * phase);
    xyz.x += 9.7470e-14
        * sqrt(two_pi * 4.5282e+09)
        * cos(2.2399e+06 * phase + shift.x)
        * exp(-4.5282e+09 * phase * phase);
    xyz = xyz / 1.0685e-7;
    return xyz_to_linear_srgb(xyz);
}

fn thin_film_fresnel(thickness_nm: f32, film_ior: f32, cos_theta1: f32, base_f0: vec3<f32>) -> vec3<f32> {
    let outside = 1.0;
    let t = clamp(thickness_nm / 0.03, 0.0, 1.0);
    let n1 = outside + (film_ior - outside) * (t * t * (3.0 - 2.0 * t));
    let ratio = outside / n1;
    let cos2sq = 1.0 - ratio * ratio * (1.0 - cos_theta1 * cos_theta1);
    if cos2sq < 0.0 {
        return vec3<f32>(1.0);
    }
    let cos_theta2 = sqrt(cos2sq);
    let r12 = tf_schlick(tf_f0_of(n1, outside), cos_theta1);
    let t121 = 1.0 - r12;
    var phi12 = 0.0;
    if n1 < outside { phi12 = PI; }
    let phi21 = PI - phi12;
    var r23 = vec3<f32>(0.0);
    var phi = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c = c + 1u) {
        let n3 = tf_ior_of_f0(base_f0[c]);
        r23[c] = tf_schlick(tf_f0_of(n3, n1), cos_theta2);
        var phi23 = 0.0;
        if n3 < n1 { phi23 = PI; }
        phi[c] = phi21 + phi23;
    }
    let opd = 2.0 * n1 * thickness_nm * cos_theta2;
    let r123 = clamp(r12 * r23, vec3<f32>(1e-5), vec3<f32>(0.9999));
    let r = sqrt(r123);
    let rs = (t121 * t121) * r23 / (vec3<f32>(1.0) - r123);
    var out = vec3<f32>(r12) + rs;
    var cm = rs - vec3<f32>(t121);
    for (var m = 1u; m <= 2u; m = m + 1u) {
        let s = tf_sensitivity(f32(m) * opd, f32(m) * phi);
        cm = cm * r;
        out = out + cm * 2.0 * s;
    }
    return clamp(out, vec3<f32>(0.0), vec3<f32>(1.0));
}

// GGX, the Smith height-correlated form, with the sun's disc widening the
// lobe: a highlight from a light of angular radius `α` cannot be sharper than
// `α`, and a shader that ignored that puts a one-pixel star on every polished
// surface at every distance.
fn ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let h = normalize(v + l);
    let a = max(roughness * roughness, 1e-3);
    let a_widened = clamp(a + u.sun_dir.w, a, 1.0);
    let a2 = a_widened * a_widened;
    let ndh = max(dot(n, h), 0.0);
    let ndv = max(dot(n, v), 1e-4);
    let ndl = max(dot(n, l), 1e-4);
    let d = ndh * ndh * (a2 - 1.0) + 1.0;
    let dist = a2 / max(PI * d * d, 1e-8);
    let k = a_widened * 0.5;
    let gv = ndv / (ndv * (1.0 - k) + k);
    let gl = ndl / (ndl * (1.0 - k) + k);
    return dist * gv * gl / max(4.0 * ndv * ndl, 1e-6);
}

// ---- Preetham's clear sky, the port of `kosm_render::env::SkyEnv` ---------
//
// Five Perez coefficients a channel, each affine in turbidity; a zenith
// chromaticity and luminance from turbidity and the solar zenith angle; and
// the ratio `F(θ,γ)/F(0,θs)`. The Rust is `crates/kosm-render/src/env.rs` and
// this is it line for line, because the settle blend fades one into the other
// and a sky that disagreed would show as a seam across the whole frame.
//
// **The sun's disc is not in here.** `γ` is clamped at `u.sky_a.w`, the sun's
// own angular radius, which caps the circumsolar term at the value it takes
// on the disc's rim. The disc itself is drawn once, by `fs_sky`, out of the
// same `Sun` the shading uses.

fn perez_c(t: f32, ch: u32) -> array<f32, 5> {
    if ch == 0u {
        return array<f32, 5>(
            0.1787 * t - 1.4630, -0.3554 * t + 0.4275, -0.0227 * t + 5.3251,
            0.1206 * t - 2.5771, -0.0670 * t + 0.3703,
        );
    }
    if ch == 1u {
        return array<f32, 5>(
            -0.0193 * t - 0.2592, -0.0665 * t + 0.0008, -0.0004 * t + 0.2125,
            -0.0641 * t - 0.8989, -0.0033 * t + 0.0452,
        );
    }
    return array<f32, 5>(
        -0.0167 * t - 0.2608, -0.0950 * t + 0.0092, -0.0079 * t + 0.2102,
        -0.0441 * t - 1.6537, -0.0109 * t + 0.0529,
    );
}

fn perez_f(c: array<f32, 5>, cos_theta: f32, gamma: f32) -> f32 {
    var k = c;
    let ct = max(cos_theta, 0.01);
    let cg = cos(gamma);
    return (1.0 + k[0] * exp(k[1] / ct)) * (1.0 + k[2] * exp(k[3] * gamma) + k[4] * cg * cg);
}

// (x, y, Y) at the zenith, Y in kcd/m² — a unit the normaliser divides out.
fn sky_zenith(t: f32, ts: f32) -> vec3<f32> {
    let t2 = t * t;
    let ts2 = ts * ts;
    let ts3 = ts2 * ts;
    let x = t2 * (0.00166 * ts3 - 0.00375 * ts2 + 0.00209 * ts)
        + t * (-0.02903 * ts3 + 0.06377 * ts2 - 0.03202 * ts + 0.00394)
        + (0.11693 * ts3 - 0.21196 * ts2 + 0.06052 * ts + 0.25886);
    let y = t2 * (0.00275 * ts3 - 0.00610 * ts2 + 0.00317 * ts)
        + t * (-0.04214 * ts3 + 0.08970 * ts2 - 0.04153 * ts + 0.00516)
        + (0.15346 * ts3 - 0.26756 * ts2 + 0.06670 * ts + 0.26688);
    let chi = (4.0 / 9.0 - t / 120.0) * (PI - 2.0 * ts);
    let lum = (4.0453 * t - 4.9710) * tan(chi) - 0.2155 * t + 2.4192;
    return vec3<f32>(x, y, max(lum, 0.05));
}

fn xyy_to_rgb(x: f32, y: f32, big_y: f32) -> vec3<f32> {
    let yy = max(y, 1e-4);
    let xx = x / yy * big_y;
    let zz = (1.0 - x - yy) / yy * big_y;
    return vec3<f32>(
        3.2404542 * xx - 1.5371385 * big_y - 0.4985314 * zz,
        -0.9692660 * xx + 1.8760108 * big_y + 0.0415560 * zz,
        0.0556434 * xx - 0.2040259 * big_y + 1.0572252 * zz,
    );
}

// The model above the horizon, before the normaliser and the intensity.
fn sky_upper(d: vec3<f32>) -> vec3<f32> {
    let t = u.sky_a.x;
    let cy = perez_c(t, 0u);
    let cx = perez_c(t, 1u);
    let cyy = perez_c(t, 2u);
    let cos_theta = max(d.z, 0.0);
    let theta_s = acos(max(clamp(u.sun_dir.z, -1.0, 1.0), 0.0));
    let gamma = max(acos(clamp(dot(normalize(d), u.sun_dir.xyz), -1.0, 1.0)), u.sky_a.w);
    let z = sky_zenith(t, theta_s);
    let big_y = z.z * perez_f(cy, cos_theta, gamma) / max(perez_f(cy, 1.0, theta_s), 1e-4);
    let x = z.x * perez_f(cx, cos_theta, gamma) / max(perez_f(cx, 1.0, theta_s), 1e-4);
    let y = z.y * perez_f(cyy, cos_theta, gamma) / max(perez_f(cyy, 1.0, theta_s), 1e-4);
    return max(xyy_to_rgb(x, y, max(big_y, 0.0)), vec3<f32>(0.0));
}

fn sky_model(d: vec3<f32>) -> vec3<f32> {
    let k = u.sky_a.y * u.sky_a.z;
    if d.z >= 0.0 {
        return sky_upper(d) * k;
    }
    let h = sky_upper(normalize(vec3<f32>(d.x, d.y, 0.02)));
    let t = clamp(sqrt(-d.z), 0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    return mix(h, h * u.sky_ground.rgb, s) * k;
}

// The gradient sky the level hangs under, as a direction lookup. The probes
// carry the sky's *irradiance*; this is its radiance, which is what a mirror
// and the sea's surface need.
fn sky_from(sh: array<f32, 54>, d: vec3<f32>) -> vec3<f32> {
    var e = convolve(sh, normalize(d));
    var c = vec3<f32>(0.0);
    for (var b = 0u; b < BANDS; b = b + 1u) {
        c = c + e[b] * u.band_to_rgb[b].xyz;
    }
    // irradiance over the cosine lobe's own solid angle is the mean radiance
    // in that lobe, which is what a broad sky reflection actually shows
    return c / PI;
}

// The sky a mirror, the sea and the haze see. With a model bound it is the
// model; without one it is the probe volume's own low-frequency read, which
// is what this tier had before there was a sky and what the datasheet ball is
// still drawn under.
fn sky_rgb(d: vec3<f32>) -> vec3<f32> {
    if u.sky_ground.w > 0.5 {
        return sky_model(normalize(d));
    }
    return sky_from(probe_sh(u.eye.xyz), d);
}

fn to_rgb(bands: array<f32, 6>) -> vec3<f32> {
    var b = bands;
    var c = vec3<f32>(0.0);
    for (var i = 0u; i < BANDS; i = i + 1u) {
        c = c + b[i] * u.band_to_rgb[i].xyz;
    }
    return c;
}

// The shared shading of an opaque surface: everything but the sea and the
// lens goes through here.
//
// `ao` is the ambient occlusion at this pixel: it multiplies the **indirect**
// half and nothing else. The direct sun already has the shadow map, which
// resolves what a hemisphere of screen-space taps cannot and would be
// double-counted if the two were multiplied together — a rock in its own
// shadow would go twice as dark as the reference has it.
fn shade_solid(m: GpuMaterial, p: vec3<f32>, n: vec3<f32>, v: vec3<f32>, ao: f32) -> vec3<f32> {
    let l = u.sun_dir.xyz;
    let shadow = sun_shadow(p, n);

    // Wrap lighting for a substance light walks *into*: the terminator moves
    // round past 90° by how far a photon travels before it comes back out.
    // `w` is the mean free path against the body's own scale, capped — a
    // material with a millimetre of walk barely wraps, skin wraps a lot.
    var wrap = 0.0;
    if m.sss_weight > 0.0 {
        let mfp = (m.sss_radius_m.x + m.sss_radius_m.y + m.sss_radius_m.z) / 3.0;
        wrap = clamp(m.sss_weight * mfp * 200.0, 0.0, 0.9);
    }
    let ndl = dot(n, l);
    let lambert = max(0.0, (ndl + wrap) / (1.0 + wrap));

    let albedo_rgb = to_rgb(array<f32, 6>(
        band(m.albedo, 0u), band(m.albedo, 1u), band(m.albedo, 2u),
        band(m.albedo, 3u), band(m.albedo, 4u), band(m.albedo, 5u),
    ));
    var f0 = f0_of(m, albedo_rgb);
    let ndv = max(dot(n, v), 1e-4);

    // The specular lobe's Fresnel, or the film's Airy reflectance over it.
    let h = normalize(v + l);
    var fspec = fresnel(f0, max(dot(v, h), 0.0));
    if m.film_nm > 0.0 {
        fspec = thin_film_fresnel(m.film_nm, m.film_ior, max(dot(v, h), 0.0), f0);
    }
    let spec_weight = ggx(n, v, l, m.roughness) * max(ndl, 0.0);

    // ---- direct, per band -------------------------------------------------
    let caustic = caustic_at(p, n) * shadow;
    var direct = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var b = 0u; b < BANDS; b = b + 1u) {
        direct[b] = sun_band(b) * shadow * lambert;
    }
    // The probes, read once for both lobes: the diffuse one at the normal and
    // the sky's reflection at the mirror direction.
    let refl = reflect(-v, n);
    let e = probe_irradiance2(p, n, normalize(refl));
    var indirect = array<f32, 6>(
        e.n_lo.x * ao, e.n_lo.y * ao, e.n_lo.z * ao,
        e.n_hi.x * ao, e.n_hi.y * ao, e.n_hi.z * ao,
    );

    // ---- the diffuse half: spectral, then projected ----------------------
    var lit = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let kd = 1.0 - m.metallic;
    for (var b = 0u; b < BANDS; b = b + 1u) {
        lit[b] = band(m.albedo, b) * (direct[b] + indirect[b]) / PI;
    }
    var colour = to_rgb(lit) * kd;
    // the caustic is already a linear-RGB irradiance: a density estimate, not
    // a spectrum, because that is what `CausticMap` hands back
    colour = colour + albedo_rgb * kd * caustic / PI;

    // ---- the specular half: RGB, because Fresnel is ----------------------
    var sun_rgb = to_rgb(array<f32, 6>(
        sun_band(0u), sun_band(1u), sun_band(2u),
        sun_band(3u), sun_band(4u), sun_band(5u),
    ));
    colour = colour + fspec * spec_weight * sun_rgb * shadow;
    // and the sky's own reflection, at the mirror direction: `sky_from`, off
    // the irradiance the fused read already took there. The sky's reflection
    // is ambient too, so the occlusion applies to it — a rock in a crevice
    // does not mirror a sky it cannot see.
    let sky_refl = to_rgb(array<f32, 6>(
        e.r_lo.x, e.r_lo.y, e.r_lo.z, e.r_hi.x, e.r_hi.y, e.r_hi.z,
    )) / PI;
    colour = colour + fresnel(f0, ndv) * sky_refl * (1.0 - m.roughness * 0.85) * ao;

    // ---- what it emits ---------------------------------------------------
    var em = array<f32, 6>(
        band(m.emission, 0u), band(m.emission, 1u), band(m.emission, 2u),
        band(m.emission, 3u), band(m.emission, 4u), band(m.emission, 5u),
    );
    return colour + to_rgb(em);
}

// The sea: an analytic swell, a Fresnel split, the seabed under it, and a
// horizon fade. `sims/pool/scene.wgsl` is the construction.
fn shade_sea(p: vec3<f32>, n_in: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    var n = n_in;
    if dot(n, v) < 0.0 { n = -n; }
    let d = -v;
    let eta = 1.0 / max(u.swell_angle.w, 1.0);
    let cos_i = clamp(-dot(d, n), 0.0, 1.0);
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    let reflected = sky_rgb(reflect(d, n));
    if k < 0.0 {
        return reflected;
    }
    let cos_t = sqrt(k);
    let dr = eta * d + (eta * cos_i - cos_t) * n;
    // Schlick against water, which is within a per cent of the exact Fresnel
    // everywhere but the last degree of grazing
    let f0 = 0.02;
    let fr = f0 + (1.0 - f0) * pow(1.0 - cos_i, 5.0);

    // the horizon fade, first, because at 1 nothing under the water is seen
    // and at 0 the horizon's own sky is not
    let far = clamp((u.sea.z - p.y) / max(u.sea.w, 1e-3), 0.0, 1.0);
    let fade = smoothstep(0.55, 1.0, far);
    let horizon = sky_rgb(normalize(vec3<f32>(d.x, d.y, 0.02)));
    if fade >= 1.0 {
        return horizon;
    }

    // the seabed: the beach's own plane, carried on under the water
    let slope = u.sea.y;
    let denom = dr.z - slope * dr.y;
    var under = vec3<f32>(0.02, 0.10, 0.16);
    var dist = 8.0;
    if denom < -1e-6 {
        let t = (u.sea.x + slope * (p.y - u.sea.z) - p.z) / denom;
        if t > 0.0 {
            let q = p + dr * t;
            let sand = materials[u.sea_flags.x];
            let sn = normalize(vec3<f32>(0.0, -slope, 1.0));
            under = shade_solid(sand, q, sn, -dr, 1.0);
            dist = t;
        }
    }
    let a = exp(-u.sea_absorb.xyz * dist);
    let body = materials[u32(u.sea_absorb.w)];
    let tint = to_rgb(array<f32, 6>(
        band(body.albedo, 0u), band(body.albedo, 1u), band(body.albedo, 2u),
        band(body.albedo, 3u), band(body.albedo, 4u), band(body.albedo, 5u),
    ));
    let sky_here = sky_rgb(vec3<f32>(0.0, 0.0, 1.0));
    under = under * a + tint * sky_here * (1.0 - a);
    var c = fr * reflected + (1.0 - fr) * under;

    // the horizon: past the lattice's reach the sea becomes the sky it
    // reflects, so the far edge of the field is not an edge
    return mix(c, horizon, fade);
}

// The lens: a thin disc with a view-dependent Fresnel highlight and the sky
// tinted through it. True refraction is the reference tier's — stand still
// and the settle blend puts it back.
fn shade_lens(m: GpuMaterial, p: vec3<f32>, n_in: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    var n = n_in;
    if dot(n, v) < 0.0 { n = -n; }
    let ndv = clamp(dot(n, v), 0.0, 1.0);
    let r = (m.ior - 1.0) / (m.ior + 1.0);
    let f0 = vec3<f32>(r * r);
    let fr = fresnel(f0, ndv);
    // what comes through: the sky along the view, tinted by the glass's own
    // six bands — a lens with a bottle-green melt reads green, which is the
    // one thing about the glass this tier can be honest about
    let through = sky_rgb(-v) * to_rgb(array<f32, 6>(
        band(m.albedo, 0u), band(m.albedo, 1u), band(m.albedo, 2u),
        band(m.albedo, 3u), band(m.albedo, 4u), band(m.albedo, 5u),
    ));
    let l = u.sun_dir.xyz;
    let spec = ggx(n, v, l, 0.02) * max(dot(n, l), 0.0) * sun_shadow(p, n);
    var sun_rgb = to_rgb(array<f32, 6>(
        sun_band(0u), sun_band(1u), sun_band(2u),
        sun_band(3u), sun_band(4u), sun_band(5u),
    ));
    return fr * sky_rgb(reflect(-v, n)) + (vec3<f32>(1.0) - fr) * through + fr * spec * sun_rgb;
}

// ---- the air ----------------------------------------------------------------

// **Aerial perspective, as a closed form.** `kosm_render::post::Aerial` is
// this function in Rust and the tracer applies it to `Film::depth`, so the two
// tiers haze by the same amount at the same distance. It is a single-scatter
// approximation with an exponential density and the sky's own radiance as the
// in-scattered colour — the reference does *not* trace a medium, and that is
// stated in `post.rs` rather than hidden here.
//
//   τ = ρ · d · e^{−z₀/H} · (1 − e^{−u}) / u,   u = (z₁ − z₀) / H
//
// written that way and not as a difference of exponentials over `Δz`, which
// cancels catastrophically in `f32` on a near-level view ray and banded the
// beach when it was tried.
fn haze(z0: f32, z1: f32, d: f32) -> f32 {
    if u.air.x <= 0.0 || d <= 0.0 {
        return 0.0;
    }
    let h = max(u.air.y, 1e-3);
    let uu = (z1 - z0) / h;
    var f = 1.0 - 0.5 * uu + uu * uu / 6.0;
    if abs(uu) >= 1e-3 {
        f = (1.0 - exp(-uu)) / uu;
    }
    let tau = u.air.x * d * exp(-z0 / h) * f;
    return 1.0 - exp(-tau);
}

fn with_air(c: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    if u.air.x <= 0.0 || u.sky_ground.w <= 0.5 {
        return c;
    }
    let rel = p - u.eye.xyz;
    let d = length(rel);
    let a = haze(u.eye.z, p.z, d);
    return mix(c, sky_model(rel / max(d, 1e-6)), a);
}

// ---- the film ---------------------------------------------------------------
//
// **This pass writes linear HDR, not bytes.** The exposure, the vignette, the
// bloom and the tonemap are one pass later, in `shaders/post.wgsl`, which is
// `kosm_render::post::Post` written in WGSL — so the two tiers apply the same
// chain to the same radiance and the settle blend still has no seam.

// The ambient occlusion at this pixel, from the half-resolution buffer.
fn ao_at(frag: vec2<f32>) -> f32 {
    if u.air.w <= 0.0 {
        return 1.0;
    }
    let uv = frag / max(u.screen.xy, vec2<f32>(1.0));
    let a = textureSampleLevel(ao_tex, ao_smp, uv, 0.0).r;
    return clamp(1.0 - (1.0 - a) * u.air.w, 0.0, 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let v = normalize(u.eye.xyz - in.wpos);
    var n = normalize(in.nrm);
    // Authored solids are inconsistently wound, exactly as the ride tier
    // found: a normal facing away from the eye is a winding, not a backface.
    if dot(n, v) < 0.0 && in.kind == 0u {
        n = -n;
    }
    let m = materials[in.mat];
    var c: vec3<f32>;
    if in.kind == 1u {
        c = shade_sea(in.wpos, n, v);
    } else if in.kind == 2u {
        c = shade_lens(m, in.wpos, n, v);
    } else {
        c = shade_solid(m, in.wpos, n, v, ao_at(in.clip.xy));
    }
    c = c + in.glow.rgb;
    return vec4<f32>(with_air(c, in.wpos), 1.0);
}

// ---- the sky, as a pass -----------------------------------------------------
//
// A full-screen quad at the far plane, before the geometry: what a ray that
// hits nothing shows. It replaces a constant clear colour, which is what this
// tier used to draw the sky as and which is why the cove's horizon was a flat
// grey band whatever the sun was doing.
//
// **The disc is drawn here and only here.** `SkyEnv` clamps its circumsolar
// term at the sun's own angular radius, so the model has no disc in it; this
// adds `Sun`'s, once, at the radiance the shading uses — irradiance over the
// disc's solid angle. Its edge is smoothed over a twentieth of the radius,
// because a hard step on a cone half a degree across is a staircase at any
// resolution a window runs at.

struct SkyOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) i: u32) -> SkyOut {
    var o: SkyOut;
    let x = f32(i & 1u);
    let y = f32(i >> 1u);
    let ndc = vec2<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0);
    o.clip = vec4<f32>(ndc, 0.0, 1.0);
    o.ndc = ndc;
    return o;
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    // the view ray, out of the same 4×4 the geometry is drawn with, so the
    // sky and the silhouettes in front of it cannot be two cameras
    let near = u.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let far = u.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let d = normalize(far.xyz / far.w - near.xyz / near.w);
    var c: vec3<f32>;
    if u.sky_ground.w > 0.5 {
        c = sky_model(d);
    } else {
        c = sky_from(probe_sh(u.eye.xyz), d);
    }
    let cos_a = dot(d, u.sun_dir.xyz);
    let cos_r = cos(u.sun_dir.w);
    let soft = 0.05 * (1.0 - cos_r);
    if cos_a > cos_r - soft {
        var sun_rgb = to_rgb(array<f32, 6>(
            sun_band(0u), sun_band(1u), sun_band(2u),
            sun_band(3u), sun_band(4u), sun_band(5u),
        ));
        // radiance is irradiance over the disc's solid angle, which is what
        // `Sun::radiance` is on the tracer
        let omega = max(6.28318530718 * (1.0 - cos_r), 1e-12);
        let edge = smoothstep(cos_r - soft, cos_r + soft, cos_a);
        c = c + sun_rgb / omega * edge;
    }
    return vec4<f32>(c, 1.0);
}

// ---- the blit of the offscreen colour onto whatever is showing --------------

struct BlitOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var blit_tex: texture_2d<f32>;
@group(0) @binding(1) var blit_smp: sampler;

@vertex
fn vs_blit(@builtin(vertex_index) i: u32) -> BlitOut {
    var o: BlitOut;
    let x = f32(i & 1u);
    let y = f32(i >> 1u);
    o.clip = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}

@fragment
fn fs_blit(in: BlitOut) -> @location(0) vec4<f32> {
    return textureSample(blit_tex, blit_smp, in.uv);
}
