// MLS-MPM, weakly compressible water, quadratic B-splines, APIC/FLIP blend.
// Mirrors newt_spike::splash::Water::step; f32 on the GPU, f64 on the CPU.
//
// Grid accumulation goes through fixed-point atomics (WGSL has no float
// atomicAdd): mass in units of particle-mass / 2^20, momentum in units of
// particle-mass / 2^16 (so a value is a velocity times 2^16), the body
// reaction in particle-mass / 2^14 per substep slot, interior mass in
// particle-mass / 2^16.

struct Params {
    origin_h: vec4<f32>,   // origin.xyz, h
    n: vec4<u32>,          // nx, ny, nz, particle count
    k: vec4<f32>,          // dt, inv_h, mass, vol0
    k2: vec4<f32>,         // bulk, flip, gravity, d_inv
    lo: vec4<f32>,         // wall lo.xyz, e (clamp margin)
    hi: vec4<f32>,         // wall hi.xyz, 0
    misc: vec4<u32>,       // slot, 0, 0, 0
    xmax: vec4<f32>,       // clamp max xyz
    b_centre: vec4<f32>,
    b_a: vec4<f32>,
    b_b: vec4<f32>,
    b_c: vec4<f32>,
    b_vel: vec4<f32>,
    semi: vec4<f32>,       // ellipsoid semi-axes
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> x: array<vec4<f32>>;      // xyz, J-1 (J itself loses the per-step increment in f32)
@group(0) @binding(2) var<storage, read_write> v: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> c: array<vec4<f32>>;      // 3 columns per particle
@group(0) @binding(4) var<storage, read_write> gm: array<atomic<i32>>;   // grid mass (fixed point)
@group(0) @binding(5) var<storage, read_write> gmom: array<atomic<i32>>; // grid momentum ×3 (fixed point)
@group(0) @binding(6) var<storage, read_write> gvel: array<vec4<f32>>;   // vel.xyz, mass
@group(0) @binding(7) var<storage, read_write> gvold: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> react: array<atomic<i32>>; // 4 per slot: fx fy fz interior

const MASS_SCALE: f32 = 1048576.0; // 2^20
const MOM_SCALE: f32 = 65536.0;    // 2^16
const REACT_SCALE: f32 = 16384.0;  // 2^14
const INT_SCALE: f32 = 65536.0;    // 2^16
const WG: u32 = 256u;

fn linear_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * WG;
}

fn node_index(i: i32, j: i32, k: i32) -> i32 {
    return (k * i32(P.n.y) + j) * i32(P.n.x) + i;
}

fn w1(f: f32) -> vec3<f32> {
    return vec3<f32>(0.5 * (1.5 - f) * (1.5 - f), 0.75 - (f - 1.0) * (f - 1.0), 0.5 * (f - 0.5) * (f - 0.5));
}

@compute @workgroup_size(256)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = linear_id(gid, nwg);
    let nn = P.n.x * P.n.y * P.n.z;
    if (g >= nn) { return; }
    atomicStore(&gm[g], 0);
    atomicStore(&gmom[3u * g], 0);
    atomicStore(&gmom[3u * g + 1u], 0);
    atomicStore(&gmom[3u * g + 2u], 0);
}

@compute @workgroup_size(256)
fn p2g(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let p = linear_id(gid, nwg);
    if (p >= P.n.w) { return; }
    let h = P.origin_h.w;
    let inv_h = P.k.y;
    let dt = P.k.x;
    let mass = P.k.z;
    let vol0 = P.k.w;
    let bulk = P.k2.x;
    let d_inv = P.k2.w;
    let xp = x[p];
    let rel = (xp.xyz - P.origin_h.xyz) * inv_h;
    let base = floor(rel - vec3<f32>(0.5));
    let fx = rel - base;
    let wx = w1(fx.x);
    let wy = w1(fx.y);
    let wz = w1(fx.z);
    let dj = xp.w;
    let jp = 1.0 + dj;
    // pressure from the equation of state: p = K (1/J − 1) = -K dJ/J, clamped so
    // stretched (splashing) water does not pull. The stress is folded into
    // the affine momentum: (-p I) * (-dt vol0 J d_inv) = s I
    let pressure = max(-bulk * dj / jp, 0.0);
    let s = pressure * dt * vol0 * jp * d_inv / mass; // per unit particle mass
    let c0 = c[3u * p].xyz;
    let c1 = c[3u * p + 1u].xyz;
    let c2 = c[3u * p + 2u].xyz;
    let vp = v[p].xyz;
    let bi = i32(base.x);
    let bj = i32(base.y);
    let bk = i32(base.z);
    for (var di = 0; di < 3; di++) {
        for (var dj = 0; dj < 3; dj++) {
            for (var dk = 0; dk < 3; dk++) {
                let i = bi + di;
                let j = bj + dj;
                let k = bk + dk;
                if (i < 0 || j < 0 || k < 0 || i >= i32(P.n.x) || j >= i32(P.n.y) || k >= i32(P.n.z)) { continue; }
                let dpos = (vec3<f32>(f32(di), f32(dj), f32(dk)) - fx) * h;
                let wt = wx[di] * wy[dj] * wz[dk];
                let g = u32(node_index(i, j, k));
                // momentum per unit particle mass: v + (s I + C) dpos
                let mom = (vp + s * dpos + c0 * dpos.x + c1 * dpos.y + c2 * dpos.z) * wt;
                atomicAdd(&gm[g], i32(round(wt * MASS_SCALE)));
                atomicAdd(&gmom[3u * g], i32(round(mom.x * MOM_SCALE)));
                atomicAdd(&gmom[3u * g + 1u], i32(round(mom.y * MOM_SCALE)));
                atomicAdd(&gmom[3u * g + 2u], i32(round(mom.z * MOM_SCALE)));
            }
        }
    }
}

// Approximate signed distance to the ellipsoid (negative inside) and its
// outward normal, the same construction as Body::sdf on the CPU.
fn body_sdf(p: vec3<f32>) -> vec4<f32> {
    let a = P.b_a.xyz;
    let b = P.b_b.xyz;
    let cc = P.b_c.xyz;
    let rel = p - P.b_centre.xyz;
    let semi = P.semi.xyz;
    let l = vec3<f32>(dot(rel, a), dot(rel, b), dot(rel, cc)) / semi;
    let k0 = length(l);
    let l2 = l / semi;
    let k1 = max(length(l2), 1e-9);
    let d = k0 * (k0 - 1.0) / k1;
    let nl = l / semi;
    let n = normalize(a * nl.x + b * nl.y + cc * nl.z);
    return vec4<f32>(n, d);
}

@compute @workgroup_size(256)
fn grid(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = linear_id(gid, nwg);
    let nn = P.n.x * P.n.y * P.n.z;
    if (g >= nn) { return; }
    let mass = P.k.z;
    let h = P.origin_h.w;
    let dt = P.k.x;
    let mi = atomicLoad(&gm[g]);
    if (mi <= 0) {
        gvel[g] = vec4<f32>(0.0);
        gvold[g] = vec4<f32>(0.0);
        return;
    }
    let m_units = f32(mi) / MASS_SCALE;          // in particle masses
    let m = m_units * mass;
    let mom = vec3<f32>(f32(atomicLoad(&gmom[3u * g])), f32(atomicLoad(&gmom[3u * g + 1u])), f32(atomicLoad(&gmom[3u * g + 2u]))) / MOM_SCALE;
    var vel = mom / m_units;
    gvold[g] = vec4<f32>(vel, m);
    vel.z -= P.k2.z * dt;
    let i = i32(g % P.n.x);
    let j = i32((g / P.n.x) % P.n.y);
    let k = i32(g / (P.n.x * P.n.y));
    let xi = P.origin_h.xyz + vec3<f32>(f32(i), f32(j), f32(k)) * h;
    // the pool: floor and walls, free-slip. By index, not position: the CPU
    // tests x_i < origin + 2h, which is i < 2 exactly, and an f32 sum can
    // land an ulp on the wrong side of that.
    if (i < 2 && vel.x < 0.0) { vel.x = 0.0; }
    if (i > i32(P.n.x) - 3 && vel.x > 0.0) { vel.x = 0.0; }
    if (j < 2 && vel.y < 0.0) { vel.y = 0.0; }
    if (j > i32(P.n.y) - 3 && vel.y > 0.0) { vel.y = 0.0; }
    if (k < 2 && vel.z < 0.0) { vel.z = 0.0; }
    if (k > i32(P.n.z) - 3 && vel.z > 0.0) { vel.z = 0.0; }
    // the melon: nodes within half a cell of its surface or inside take its
    // normal velocity; what that costs is booked as the reaction
    let sd = body_sdf(xi);
    let d = sd.w;
    let nrm = sd.xyz;
    var dv = vec3<f32>(0.0);
    var inside = 0.0;
    if (d < 0.0) {
        inside = m_units;
        let rel = vel - P.b_vel.xyz;
        let vn2 = vel - nrm * dot(rel, nrm);
        dv = vn2 - vel;
        vel = vn2;
    } else if (d < 0.5 * h) {
        let rel = vel - P.b_vel.xyz;
        let vn = dot(rel, nrm);
        if (vn < 0.0) {
            let vn2 = vel - nrm * vn;
            dv = vn2 - vel;
            vel = vn2;
        }
    }
    if (inside > 0.0 || dot(dv, dv) > 0.0) {
        let slot = 4u * P.misc.x;
        let r = dv * m_units * REACT_SCALE;
        atomicAdd(&react[slot], i32(round(r.x)));
        atomicAdd(&react[slot + 1u], i32(round(r.y)));
        atomicAdd(&react[slot + 2u], i32(round(r.z)));
        atomicAdd(&react[slot + 3u], i32(round(inside * INT_SCALE)));
    }
    gvel[g] = vec4<f32>(vel, m);
}

@compute @workgroup_size(256)
fn g2p(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let p = linear_id(gid, nwg);
    if (p >= P.n.w) { return; }
    let h = P.origin_h.w;
    let inv_h = P.k.y;
    let dt = P.k.x;
    let flip = P.k2.y;
    let d_inv = P.k2.w;
    var xp = x[p];
    let rel = (xp.xyz - P.origin_h.xyz) * inv_h;
    let base = floor(rel - vec3<f32>(0.5));
    let fx = rel - base;
    let wx = w1(fx.x);
    let wy = w1(fx.y);
    let wz = w1(fx.z);
    var vnew = vec3<f32>(0.0);
    var dv = vec3<f32>(0.0);
    var b0 = vec3<f32>(0.0);
    var b1 = vec3<f32>(0.0);
    var b2 = vec3<f32>(0.0);
    let bi = i32(base.x);
    let bj = i32(base.y);
    let bk = i32(base.z);
    for (var di = 0; di < 3; di++) {
        for (var dj = 0; dj < 3; dj++) {
            for (var dk = 0; dk < 3; dk++) {
                let i = bi + di;
                let j = bj + dj;
                let k = bk + dk;
                if (i < 0 || j < 0 || k < 0 || i >= i32(P.n.x) || j >= i32(P.n.y) || k >= i32(P.n.z)) { continue; }
                let g = u32(node_index(i, j, k));
                let gv4 = gvel[g];
                if (gv4.w <= 0.0) { continue; }
                let dpos = (vec3<f32>(f32(di), f32(dj), f32(dk)) - fx) * h;
                let wt = wx[di] * wy[dj] * wz[dk];
                let gv = gv4.xyz;
                vnew += gv * wt;
                dv += (gv - gvold[g].xyz) * wt;
                // outer(gv*wt, dpos): column c is (gv*wt) * dpos[c]
                let a = gv * wt;
                b0 += a * dpos.x;
                b1 += a * dpos.y;
                b2 += a * dpos.z;
            }
        }
    }
    // FLIP keeps the particle's own velocity and adds the grid's change;
    // PIC takes the grid's velocity. The blend is the usual trade of
    // dissipation for noise.
    let vp = (v[p].xyz + dv) * flip + vnew * (1.0 - flip);
    v[p] = vec4<f32>(vp, 0.0);
    let c0 = b0 * d_inv;
    let c1 = b1 * d_inv;
    let c2 = b2 * d_inv;
    c[3u * p] = vec4<f32>(c0, 0.0);
    c[3u * p + 1u] = vec4<f32>(c1, 0.0);
    c[3u * p + 2u] = vec4<f32>(c2, 0.0);
    // volume from the trace of the velocity gradient
    var dj = xp.w + (1.0 + xp.w) * dt * (c0.x + c1.y + c2.z);
    dj = clamp(dj, -0.5, 1.0);
    var pos = xp.xyz + vnew * dt;
    // keep particles in the box
    let e = P.lo.w;
    pos = clamp(pos, P.origin_h.xyz + vec3<f32>(e), P.xmax.xyz);
    x[p] = vec4<f32>(pos, dj);
}
