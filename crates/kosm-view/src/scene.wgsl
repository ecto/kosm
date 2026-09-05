// the live tier: the pool, its water, the melon and the drops, shaded the
// way the reference tracer shades them, on the GPU.

struct Uniforms {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,          // xyz, w = time
    sun: vec4<f32>,          // xyz dir, w = irradiance
    pool: vec4<f32>,         // half x, half y, depth, coping
    melon_centre: vec4<f32>, // xyz, w unused
    melon_axis: vec4<f32>,   // xyz long axis, w unused
    melon_semi: vec4<f32>,   // semi-axes, w unused
    caustic: vec4<f32>,      // origin x, origin y, cell, unused
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var caustic_tex: texture_2d<f32>;
@group(0) @binding(2) var caustic_smp: sampler;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) aux: vec3<f32>,   // per material: melon local coords, etc
    @location(3) mat: u32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) wpos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) aux: vec3<f32>,
    @location(3) @interpolate(flat) mat: u32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var o: VsOut;
    o.clip = u.view_proj * vec4<f32>(in.pos, 1.0);
    o.wpos = in.pos;
    o.nrm = in.nrm;
    o.aux = in.aux;
    o.mat = in.mat;
    return o;
}

// instanced beads: a unit sphere scaled and turned per instance
struct BeadIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
};

@vertex
fn vs_bead(in: BeadIn) -> VsOut {
    let m = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let wp = m * vec4<f32>(in.pos, 1.0);
    var o: VsOut;
    o.clip = u.view_proj * wp;
    o.wpos = wp.xyz;
    // normal through the inverse transpose is overkill for a near-sphere
    o.nrm = normalize((m * vec4<f32>(in.nrm, 0.0)).xyz);
    o.aux = vec3<f32>(0.0);
    o.mat = 4u;
    return o;
}

// ---- the world, as the tracer has it ----------------------------------------

fn sky(d: vec3<f32>) -> vec3<f32> {
    let t = clamp(d.z, 0.0, 1.0);
    let horizon = vec3<f32>(0.66, 0.80, 0.94);
    let zenith = vec3<f32>(0.22, 0.44, 0.88);
    var c = mix(horizon, zenith, t);
    let cs = dot(d, u.sun.xyz);
    if cs > 0.99995 {
        return vec3<f32>(12.0, 11.0, 9.0);
    }
    let g = clamp((cs - 0.97) / 0.03, 0.0, 1.0);
    c += 0.6 * g * vec3<f32>(1.0, 0.9, 0.7);
    return c;
}

fn tile(x: f32, y: f32) -> vec3<f32> {
    let fx = abs(fract(x / 0.1) - 0.5);
    let fy = abs(fract(y / 0.1) - 0.5);
    if fx > 0.46 || fy > 0.46 {
        return vec3<f32>(0.42, 0.52, 0.58);
    }
    let a = (i32(floor(x / 0.1)) + i32(floor(y / 0.1))) % 2 == 0;
    if a {
        return vec3<f32>(0.58, 0.78, 0.86);
    }
    return vec3<f32>(0.50, 0.72, 0.84);
}

fn deck_colour() -> vec3<f32> {
    return vec3<f32>(0.80, 0.68, 0.52);
}

// ray–ellipsoid for the melon; returns t or -1
fn melon_hit(o: vec3<f32>, d: vec3<f32>) -> f32 {
    let a = u.melon_axis.xyz;
    let b = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), a));
    let c = cross(a, b);
    let s = u.melon_semi.xyz;
    let rel = o - u.melon_centre.xyz;
    let ol = vec3<f32>(dot(rel, a) / s.x, dot(rel, b) / s.y, dot(rel, c) / s.z);
    let dl = vec3<f32>(dot(d, a) / s.x, dot(d, b) / s.y, dot(d, c) / s.z);
    let aa = dot(dl, dl);
    let bb = dot(ol, dl);
    let cc = dot(ol, ol) - 1.0;
    let disc = bb * bb - aa * cc;
    if disc < 0.0 {
        return -1.0;
    }
    let t = (-bb - sqrt(disc)) / aa;
    if t < 1e-5 {
        return -1.0;
    }
    return t;
}

fn fresnel(n1: f32, n2: f32, cos_i: f32, cos_t: f32) -> f32 {
    let rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    let rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i);
    return 0.5 * (rs * rs + rp * rp);
}

fn caustic_at(x: f32, y: f32) -> f32 {
    let dims = vec2<f32>(textureDimensions(caustic_tex));
    let uv = (vec2<f32>(x, y) - u.caustic.xy) / (u.caustic.z * dims);
    if uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 {
        return 1.0;
    }
    return textureSampleLevel(caustic_tex, caustic_smp, uv, 0.0).r;
}

// the sun's refracted ray, back from a floor point: is the melon in the way?
fn sun_blocked_under(p: vec3<f32>) -> bool {
    let d = -u.sun.xyz;
    let eta = 1.0 / 1.333;
    let cos_i = -d.z;
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    let dr = eta * d + (eta * cos_i - sqrt(k)) * vec3<f32>(0.0, 0.0, 1.0);
    let up = -dr;
    let t_surf = (0.0 - p.z) / up.z;
    let t = melon_hit(p + up * 1e-3, up);
    if t > 0.0 && t < t_surf {
        return true;
    }
    let s = p + up * t_surf;
    return melon_hit(s + u.sun.xyz * 1e-3, u.sun.xyz) > 0.0;
}

fn melon_albedo(l: vec3<f32>) -> vec3<f32> {
    // l: melon-local unit coords (u along the long axis)
    let phi = atan2(l.z, l.y);
    let wobble = 0.35 * sin(6.0 * l.x + 2.0 * phi) + 0.2 * sin(13.0 * l.x);
    let stripe = sin(9.0 * phi + wobble);
    let s = clamp((stripe + 0.15) * 3.0, -1.0, 1.0) * 0.5 + 0.5;
    let dark = vec3<f32>(0.07, 0.24, 0.09);
    let light = vec3<f32>(0.52, 0.70, 0.32);
    var col = mix(light, dark, s);
    let belly = clamp((-l.z - 0.75) * 6.0, 0.0, 1.0);
    col = mix(col, vec3<f32>(0.85, 0.82, 0.55), belly);
    return col;
}

fn shade_melon(p: vec3<f32>, n: vec3<f32>, d: vec3<f32>, l: vec3<f32>, in_air: bool) -> vec3<f32> {
    let s = u.sun.xyz;
    let base = melon_albedo(l);
    let cosl = max(dot(n, s), 0.0);
    var direct = u.sun.w * cosl;
    if !in_air {
        direct = u.sun.w * 0.85 * cosl * min(caustic_at(p.x, p.y), 2.5);
    }
    let sk = sky(n);
    var c = base * (0.35 * sk + direct);
    if in_air {
        let h = normalize(s - d);
        let spec = pow(max(dot(n, h), 0.0), 120.0) * 1.6;
        let fres = 0.04 + 0.96 * pow(clamp(1.0 + dot(d, n), 0.0, 1.0), 5.0);
        c += spec + fres * 0.5 * sky(reflect(d, n));
    }
    return c;
}

// a ray inside the water: melon, floor or wall, absorbed by distance
fn underwater(o: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    let hx = u.pool.x;
    let hy = u.pool.y;
    let depth = u.pool.z;
    var best = 1e9;
    var what = 0;
    var n_hit = vec3<f32>(0.0, 0.0, 1.0);
    let tm = melon_hit(o, d);
    if tm > 0.0 { best = tm; what = 1; }
    if d.z < 0.0 {
        let t = (-depth - o.z) / d.z;
        if t < best { best = t; what = 2; }
    }
    // walls
    if d.x > 1e-6 { let t = (hx - o.x) / d.x; if t > 0.0 && t < best { best = t; what = 3; n_hit = vec3<f32>(-1.0, 0.0, 0.0); } }
    if d.x < -1e-6 { let t = (-hx - o.x) / d.x; if t > 0.0 && t < best { best = t; what = 3; n_hit = vec3<f32>(1.0, 0.0, 0.0); } }
    if d.y > 1e-6 { let t = (hy - o.y) / d.y; if t > 0.0 && t < best { best = t; what = 3; n_hit = vec3<f32>(0.0, -1.0, 0.0); } }
    if d.y < -1e-6 { let t = (-hy - o.y) / d.y; if t > 0.0 && t < best { best = t; what = 3; n_hit = vec3<f32>(0.0, 1.0, 0.0); } }
    if what == 0 {
        return vec3<f32>(0.3, 0.45, 0.6);
    }
    let p = o + d * best;
    var c: vec3<f32>;
    if what == 1 {
        // melon under water: normal from the ellipsoid gradient
        let a = u.melon_axis.xyz;
        let b = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), a));
        let cc = cross(a, b);
        let s = u.melon_semi.xyz;
        let rel = p - u.melon_centre.xyz;
        let l = vec3<f32>(dot(rel, a) / s.x, dot(rel, b) / s.y, dot(rel, cc) / s.z);
        let nl = vec3<f32>(l.x / s.x, l.y / s.y, l.z / s.z);
        let n = normalize(a * nl.x + b * nl.y + cc * nl.z);
        c = shade_melon(p, n, d, l, false);
    } else if what == 2 {
        let base = tile(p.x, p.y);
        var shadow = 1.0;
        if sun_blocked_under(p) { shadow = 0.12; }
        let lit = u.sun.w * 0.92 * caustic_at(p.x, p.y) * shadow;
        c = base * (0.30 + lit);
    } else {
        let base = tile(p.x + p.z, p.y + p.z);
        let lit = u.sun.w * 0.5 * max(dot(n_hit, u.sun.xyz), 0.0);
        c = base * (0.30 + lit);
    }
    let absorb = vec3<f32>(0.45, 0.10, 0.04);
    let a = exp(-absorb * best);
    return c * a + vec3<f32>(0.02, 0.10, 0.16) * (1.0 - a);
}

fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let v = c / (1.0 + c * 0.35) * 1.2;
    return pow(clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = normalize(in.wpos - u.eye.xyz);
    let n = normalize(in.nrm);
    let s = u.sun.xyz;
    var c = vec3<f32>(0.0);
    switch in.mat {
        case 0u: { // deck
            var lit = u.sun.w * max(n.z, 0.0);
            if melon_hit(in.wpos + s * 1e-3, s) > 0.0 { lit *= 0.15; }
            c = deck_colour() * (0.35 * sky(n) + lit);
        }
        case 1u: { // pool walls above the water line, tiled
            let base = tile(in.wpos.x + in.wpos.z, in.wpos.y + in.wpos.z);
            var lit = u.sun.w * max(dot(n, s), 0.0);
            if melon_hit(in.wpos + s * 1e-3, s) > 0.0 { lit = 0.0; }
            c = base * (0.35 * sky(n) + lit);
        }
        case 2u: { // the water surface
            var nn = n;
            if dot(nn, d) > 0.0 { nn = -nn; }
            let eta = 1.0 / 1.333;
            let cos_i = -dot(d, nn);
            let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
            let rd = reflect(d, nn);
            var reflected = sky(rd);
            let tm = melon_hit(in.wpos + nn * 1e-4, rd);
            if tm > 0.0 {
                let p = in.wpos + rd * tm;
                // a rough normal for the reflected melon
                let nm = normalize(p - u.melon_centre.xyz);
                reflected = shade_melon(p, nm, rd, vec3<f32>(0.0, 0.0, 0.0), true);
            }
            if k < 0.0 {
                c = reflected;
            } else {
                let cos_t = sqrt(k);
                let dr = eta * d + (eta * cos_i - cos_t) * nn;
                let r = fresnel(1.0, 1.333, cos_i, cos_t);
                let under = underwater(in.wpos - nn * 1e-4, dr);
                c = r * reflected + (1.0 - r) * under;
            }
        }
        case 3u: { // the melon above water
            c = shade_melon(in.wpos, n, d, in.aux, true);
        }
        default: { // a bead
            let r = 0.04 + 0.96 * pow(clamp(1.0 + dot(d, n), 0.0, 1.0), 5.0);
            let sk = sky(reflect(d, n));
            c = r * sk + (1.0 - r) * vec3<f32>(0.60, 0.76, 0.88) * 0.85;
            c += pow(max(dot(n, normalize(s - d)), 0.0), 60.0) * 2.5;
        }
    }
    return vec4<f32>(tonemap(c), 1.0);
}

// ---- the blit of the offscreen frame into the panel -------------------------

@group(0) @binding(0) var blit_tex: texture_2d<f32>;
@group(0) @binding(1) var blit_smp: sampler;

struct BlitOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_blit(@builtin(vertex_index) i: u32) -> BlitOut {
    var o: BlitOut;
    let x = f32(i & 1u) * 2.0;
    let y = f32(i >> 1u) * 2.0;
    o.clip = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}

@fragment
fn fs_blit(in: BlitOut) -> @location(0) vec4<f32> {
    return textureSample(blit_tex, blit_smp, in.uv);
}
