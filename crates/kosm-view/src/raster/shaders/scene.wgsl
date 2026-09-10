// The raster tier's one shading pass.
//
// Six bands in, RGB out. The material's albedo, the sun's irradiance and the
// probes' irradiance are all spectral, they are multiplied band by band, and
// only the product is projected to the film — which is what makes this tier
// comparable with the spectral path tracer that baked the probes.
//
//   direct   = E_sun · shadow · (albedo/π · wrap(n·l) + GGX)      [6 bands]
//            + caustic irradiance where a receiver quad lands
//   indirect = probes(sun, p, n)                                  [6 bands]
//   colour   = M · (albedo ⊙ (direct + indirect)) + emission + glow
//
// then ACES and sRGB, the same two curves `Film::to_srgb8` applies, so the
// settle blend has no seam.

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
    // the sea's absorption per RGB metre, w the seabed's material index
    sea_absorb: vec4<f32>,
    // x the water's material index, y whether there is a sea at all, zw spare
    sea_flags: vec4<u32>,
    // two receiver rectangles: origin.xyz + w unused, then the u and v edges
    caustic_origin: array<vec4<f32>, 2>,
    caustic_u: array<vec4<f32>, 2>,
    caustic_v: array<vec4<f32>, 2>,
    // x how many quads are live, y their plane tolerance in metres
    caustic_flags: vec4<f32>,
    // the six-band → linear RGB projection, one row per band
    band_to_rgb: array<vec4<f32>, 6>,
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
// irradiance at a normal. They are separate because **a fragment reads the
// same block twice** — once at the shading normal and once at the mirror
// direction for the sky term — and interpolating 54 floats twice is the
// difference between forty frames a second and sixty.
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

// 3×3 PCF with a slope-scaled bias. The bias is stated in *world metres* —
// one shadow texel, widened by how steeply the surface leans away from the
// sun — and converted to the map's own depth by the frustum's range, which is
// the only way a bias survives a sun that moves.
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
    // one texel across, plus the slope's rise over that texel; the frustum's
    // depth is baked into knobs.y so this arrives already in clip units
    let bias = u.knobs.y * (1.0 + min(tan_l, 8.0));
    let side = max(u.knobs.z, 1.0);
    var sum = 0.0;
    for (var j = -1; j <= 1; j = j + 1) {
        for (var i = -1; i <= 1; i = i + 1) {
            let o = vec2<f32>(f32(i), f32(j)) / side;
            sum = sum + textureSampleCompareLevel(shadow_tex, shadow_smp, uv + o, ndc.z - bias);
        }
    }
    return sum / 9.0;
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

fn sky_rgb(d: vec3<f32>) -> vec3<f32> {
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
fn shade_solid(m: GpuMaterial, p: vec3<f32>, n: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
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
    let sh = probe_sh(p);
    var indirect = convolve(sh, n);

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
    // and the sky's own reflection, at the mirror direction
    let refl = reflect(-v, n);
    colour = colour + fresnel(f0, ndv) * sky_from(sh, refl) * (1.0 - m.roughness * 0.85);

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
            under = shade_solid(sand, q, sn, -dr);
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
    let far = clamp((u.sea.z - p.y) / max(u.sea.w, 1e-3), 0.0, 1.0);
    let fade = smoothstep(0.55, 1.0, far);
    return mix(c, sky_rgb(normalize(vec3<f32>(d.x, d.y, 0.02))), fade);
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

// ---- the film ---------------------------------------------------------------

// `kosm_render::cpu::film::tonemap_aces`, character for character.
fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb1(x: f32) -> f32 {
    if x <= 0.0031308 {
        return 12.92 * x;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(linear_to_srgb1(c.x), linear_to_srgb1(c.y), linear_to_srgb1(c.z));
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
        c = shade_solid(m, in.wpos, n, v);
    }
    c = c + in.glow.rgb;
    return vec4<f32>(linear_to_srgb(tonemap_aces(c * u.eye.w)), 1.0);
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
