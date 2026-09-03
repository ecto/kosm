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
    lo: vec4<f32>,         // region centre.xy, 0, e (clamp margin)
    hi: vec4<f32>,         // region radius, 0, 0, velocity damping per substep
    misc: vec4<u32>,       // slot, nbx, nby, nbz (blocks per axis)
    xmax: vec4<f32>,       // clamp max xyz, J relax share
    b_centre: vec4<f32>,
    b_a: vec4<f32>,
    b_b: vec4<f32>,
    b_c: vec4<f32>,
    b_vel: vec4<f32>,
    semi: vec4<f32>,       // ellipsoid semi-axes, sponge width
    misc2: vec4<u32>,      // max active slots, 0, 0, 0
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> x: array<vec4<f32>>;      // xyz, J-1 (J itself loses the per-step increment in f32)
@group(0) @binding(2) var<storage, read_write> v: array<vec4<f32>>;      // xyz, original particle id (the sort permutes)
@group(0) @binding(3) var<storage, read_write> c: array<vec4<f32>>;      // 3 columns per particle
@group(0) @binding(4) var<storage, read_write> gm: array<atomic<i32>>;   // grid mass (fixed point)
@group(0) @binding(5) var<storage, read_write> gmom: array<atomic<i32>>; // grid momentum ×3 (fixed point)
@group(0) @binding(6) var<storage, read_write> gvel: array<vec4<f32>>;   // vel.xyz, mass
@group(0) @binding(7) var<storage, read_write> gvold: array<vec4<f32>>;  // vel.xyz, blurred mass
@group(0) @binding(8) var<storage, read_write> react: array<atomic<i32>>; // 4 per slot: fx fy fz interior
// the sort: particles binned into 4x4x4-cell blocks each substep
@group(0) @binding(9) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(11) var<storage, read_write> fill: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> perm: array<u32>;
@group(0) @binding(13) var<storage, read_write> xo: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read_write> vo: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read_write> co: array<vec4<f32>>;
// the block-sparse grid: a dense table over the box mapping a 4^3-node block
// to its slot in the compact node arrays, the slot->block list, the counter,
// and the indirect dispatch args the node kernels run under
@group(0) @binding(16) var<storage, read_write> btab: array<u32>;
@group(0) @binding(17) var<storage, read_write> alist: array<u32>;
@group(0) @binding(18) var<storage, read_write> nact: array<atomic<u32>>;
// its own group: a buffer cannot be a storage binding and the source of an
// indirect dispatch in the same dispatch's usage scope, so only blk_scan,
// which writes it, has it bound
@group(1) @binding(0) var<storage, read_write> indirect: array<u32>;

const BLK: i32 = 4;      // cells per block per axis
const TN: i32 = 7;       // nodes a block's particles touch per axis: [4b-1, 4b+5]
const TILE: u32 = 343u;  // TN^3
var<workgroup> tile_m: array<atomic<i32>, 343>;
var<workgroup> tile_p: array<atomic<i32>, 1029>;
var<workgroup> partial: array<u32, 256>;

const MASS_SCALE: f32 = 1048576.0; // 2^20
const MOM_SCALE: f32 = 65536.0;    // 2^16
const REACT_SCALE: f32 = 16384.0;  // 2^14
const INT_SCALE: f32 = 65536.0;    // 2^16
const WG: u32 = 256u;

fn linear_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * WG;
}

const NONE: u32 = 0xffffffffu;

// A node's place in the compact arrays: slot*64 + local, or -1 when the
// node's block is not active this substep. An inactive node reads as empty
// (zero mass, zero velocity), which is exactly what the dense grid held
// there: a block is active whenever any particle's 3x3x3 stencil, dilated by
// the blur's own 3x3x3, can reach into it.
fn node_index(i: i32, j: i32, k: i32) -> i32 {
    let b = ((k / BLK) * i32(P.misc.z) + (j / BLK)) * i32(P.misc.y) + (i / BLK);
    let s = btab[u32(b)];
    if (s == NONE) { return -1; }
    return i32(s) * 64 + (((k & 3) * 4 + (j & 3)) * 4 + (i & 3));
}

// The node (i,j,k) a compact index belongs to. Slots run 0..nact and every
// slot names a block through alist.
fn node_of(g: u32) -> vec3<i32> {
    let slot = g / 64u;
    let l = i32(g % 64u);
    let b = i32(alist[slot]);
    let nbx = i32(P.misc.y);
    let nby = i32(P.misc.z);
    let blk = vec3<i32>(b % nbx, (b / nbx) % nby, b / (nbx * nby));
    return blk * BLK + vec3<i32>(l & 3, (l >> 2) & 3, (l >> 4) & 3);
}

fn active_nodes() -> u32 {
    return atomicLoad(&nact[0]) * 64u;
}

fn w1(f: f32) -> vec3<f32> {
    return vec3<f32>(0.5 * (1.5 - f) * (1.5 - f), 0.75 - (f - 1.0) * (f - 1.0), 0.5 * (f - 0.5) * (f - 0.5));
}

@compute @workgroup_size(256)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = linear_id(gid, nwg);
    if (g >= active_nodes()) { return; }
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
                let gs = node_index(i, j, k);
                if (gs < 0) { continue; }
                let dpos = (vec3<f32>(f32(di), f32(dj), f32(dk)) - fx) * h;
                let wt = wx[di] * wy[dj] * wz[dk];
                let g = u32(gs);
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
    if (g >= active_nodes()) { return; }
    let ijk = node_of(g);
    // a block on the far edge of the box can hang off the end of the grid
    if (ijk.x >= i32(P.n.x) || ijk.y >= i32(P.n.y) || ijk.z >= i32(P.n.z)) { return; }
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
    let i = ijk.x;
    let j = ijk.y;
    let k = ijk.z;
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
    // the region's edge: beyond the radius the water is the far field's, at
    // rest, so no outflow crosses it (a free-slip cylinder, the wall the box
    // had); inside, the sponge band damps the motion (see the CPU solver)
    let sw = P.semi.w;
    if (sw > 0.0) {
        let rv = xi.xy - P.lo.xy;
        let dist = length(rv);
        let inset = P.hi.x - dist; // the region is a disc
        if (outside_region(i, j) && dist > 1e-6) {
            let nrm2 = rv / dist;
            let out = dot(vel.xy, nrm2);
            if (out > 0.0) { vel = vec3<f32>(vel.xy - nrm2 * out, vel.z); }
        }
        if (inset < sw) {
            let r = 1.0 - max(inset / sw, 0.0);
            vel = vel * (1.0 - 0.03 * r * r);
        }
    }
    // the melon: nodes within half a cell of its surface or inside take its
    // normal velocity; what that costs is booked as the reaction
    let sd = body_sdf(xi);
    let d = sd.w;
    let nrm = sd.xyz;
    var dv = vec3<f32>(0.0);
    var inside = 0.0;
    if (d < 0.0) {
        inside = m_units;
        // fully the body's velocity inside (see the CPU solver)
        dv = P.b_vel.xyz - vel;
        vel = P.b_vel.xyz;
    } else if (d < h) {  // a full cell of seal, not half of one (see the CPU solver)
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
    var rho = 0.0;
    let bi = i32(base.x);
    let bj = i32(base.y);
    let bk = i32(base.z);
    // Away from the two-node wall band the mirrored node is the node itself,
    // so one table read does for both. (Caching the eight blocks the 27 nodes
    // can span is tempting and is a loss: a dynamically indexed local array
    // lands in thread-private memory, which is slower than re-reading a table
    // the neighbouring threads are all hitting anyway.)
    let ilo = 2;
    let ihi = vec3<i32>(i32(P.n.x), i32(P.n.y), i32(P.n.z)) - vec3<i32>(3);
    for (var di = 0; di < 3; di++) {
        for (var dj = 0; dj < 3; dj++) {
            for (var dk = 0; dk < 3; dk++) {
                let i = bi + di;
                let j = bj + dj;
                let k = bk + dk;
                if (i < 0 || j < 0 || k < 0 || i >= i32(P.n.x) || j >= i32(P.n.y) || k >= i32(P.n.z)) { continue; }
                let gs = node_index(i, j, k);
                let wt = wx[di] * wy[dj] * wz[dk];
                // the density the pressure sees, mirrored at the walls
                var gr = gs;
                if (i < ilo || j < ilo || k < ilo || i > ihi.x || j > ihi.y || k > ihi.z) {
                    gr = node_index(clamp(i, ilo, ihi.x), clamp(j, ilo, ihi.y), clamp(k, ilo, ihi.z));
                }
                if (gr >= 0) { rho += wt * gvold[u32(gr)].w; }
                if (gs < 0) { continue; }
                let g = u32(gs);
                let gv4 = gvel[g];
                if (gv4.w <= 0.0) { continue; }
                let dpos = (vec3<f32>(f32(di), f32(dj), f32(dk)) - fx) * h;
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
    var vp = ((v[p].xyz + dv) * flip + vnew * (1.0 - flip)) * P.hi.w;
    let c0 = b0 * d_inv;
    let c1 = b1 * d_inv;
    let c2 = b2 * d_inv;
    c[3u * p] = vec4<f32>(c0, 0.0);
    c[3u * p + 1u] = vec4<f32>(c1, 0.0);
    c[3u * p + 2u] = vec4<f32>(c2, 0.0);
    // volume from the mass around the particle (see the CPU solver)
    // rest node mass is rho0 h^3 = particle mass * h^3 / vol0
    let full = P.k.z * h * h * h / P.k.w;
    // J from the (blurred) mass around the particle: see the CPU solver
    // integrated J relaxed toward the mass density: see the CPU solver
    let dj_int = xp.w + (1.0 + xp.w) * dt * (c0.x + c1.y + c2.z);
    let dj_mass = full / max(rho, 1e-12) - 1.0;
    let dj = clamp(dj_int + (dj_mass - dj_int) * P.xmax.w, -0.5, 1.0);
    var pos = xp.xyz + vnew * dt;
    // keep particles in the box
    let e = P.lo.w;
    pos = clamp(pos, P.origin_h.xyz + vec3<f32>(e), P.xmax.xyz);
    // and out of the body (see the CPU solver)
    let sd = body_sdf(pos);
    if (sd.w < 0.0) {
        pos -= sd.xyz * (sd.w - 0.25 * h);
        let vn = dot(vp - P.b_vel.xyz, sd.xyz);
        if (vn < 0.0) {
            vp -= sd.xyz * vn;
            // booked into the same ledger as the grid constraint: this is the
            // body pushing on the water too (see the CPU solver)
            let slot = 4u * P.misc.x;
            let r = -sd.xyz * vn * REACT_SCALE;
            atomicAdd(&react[slot], i32(round(r.x)));
            atomicAdd(&react[slot + 1u], i32(round(r.y)));
            atomicAdd(&react[slot + 2u], i32(round(r.z)));
        }
    }
    v[p] = vec4<f32>(vp, v[p].w); // w carries the particle's original id
    x[p] = vec4<f32>(pos, dj);
}

fn block_of(xp: vec3<f32>) -> u32 {
    let rel = (xp - P.origin_h.xyz) * P.k.y;
    let cell = clamp(vec3<i32>(floor(rel)), vec3<i32>(0), vec3<i32>(P.n.xyz) - vec3<i32>(1));
    let b = cell / BLK;
    return u32((b.z * i32(P.misc.z) + b.y) * i32(P.misc.y) + b.x);
}

// The region's circle on node indices (see the CPU solver): centre index
// from the centre coordinate, radius² in cells.
fn outside_region(i: i32, j: i32) -> bool {
    let inv_h = P.k.y;
    let ic = i32(round((P.lo.x - P.origin_h.x) * inv_h));
    let jc = i32(round((P.lo.y - P.origin_h.y) * inv_h));
    let di = i - ic;
    let dj = j - jc;
    let r = P.hi.x * inv_h;
    return f32(di * di + dj * dj) > r * r;
}

fn nblocks() -> u32 {
    return P.misc.y * P.misc.z * P.misc.w;
}

@compute @workgroup_size(256)
fn sort_zero(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let b = linear_id(gid, nwg);
    if (b >= nblocks()) { return; }
    atomicStore(&counts[b], 0u);
    atomicStore(&fill[b], 0u);
    btab[b] = 0u; // the marks start clean each substep
}

@compute @workgroup_size(256)
fn sort_count(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let p = linear_id(gid, nwg);
    if (p >= P.n.w) { return; }
    atomicAdd(&counts[block_of(x[p].xyz)], 1u);
}

// exclusive prefix sum of the block counts, one workgroup
@compute @workgroup_size(256)
fn sort_scan(@builtin(local_invocation_index) t: u32) {
    let nb = nblocks();
    let chunk = (nb + 255u) / 256u;
    let lo = t * chunk;
    let hi = min(lo + chunk, nb);
    var sum = 0u;
    for (var b = lo; b < hi; b++) {
        sum += atomicLoad(&counts[b]);
    }
    partial[t] = sum;
    workgroupBarrier();
    if (t == 0u) {
        var acc = 0u;
        for (var i = 0u; i < 256u; i++) {
            let c = partial[i];
            partial[i] = acc;
            acc += c;
        }
        offsets[nb] = acc;
    }
    workgroupBarrier();
    var run = partial[t];
    for (var b = lo; b < hi; b++) {
        offsets[b] = run;
        run += atomicLoad(&counts[b]);
    }
}

// The block table, rebuilt every substep from the sort's block counts.
//
// A particle in a 4^3-cell block writes to nodes 4b-1 .. 4b+5, which is the
// node blocks b-1, b and b+1; the mass blur then reads one node further, and
// 4b-2 .. 4b+6 still falls inside those same three blocks. So the set of
// blocks anything touches is exactly the blocks holding particles dilated by
// one block in each direction, and that is what these two kernels build:
// mark (a gather over the 27 neighbours, so no races), then a single-
// workgroup prefix sum that hands out compact slots.
@compute @workgroup_size(256)
fn blk_mark(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    // a scatter from the blocks that hold particles to their 27 neighbours:
    // over a table that spans the pool almost every block is empty, and a
    // gather of 27 counts per block was the cost that made that table dear
    let b = linear_id(gid, nwg);
    if (b >= nblocks()) { return; }
    if (atomicLoad(&counts[b]) == 0u) { return; }
    let nbx = i32(P.misc.y);
    let nby = i32(P.misc.z);
    let nbz = i32(P.misc.w);
    let bi = i32(b);
    let bx = bi % nbx;
    let by = (bi / nbx) % nby;
    let bz = bi / (nbx * nby);
    for (var dz = -1; dz <= 1; dz++) {
        let z = bz + dz;
        if (z < 0 || z >= nbz) { continue; }
        for (var dy = -1; dy <= 1; dy++) {
            let y = by + dy;
            if (y < 0 || y >= nby) { continue; }
            for (var dx = -1; dx <= 1; dx++) {
                let xx = bx + dx;
                if (xx < 0 || xx >= nbx) { continue; }
                btab[u32((z * nby + y) * nbx + xx)] = 1u; // a flag; blk_scan makes it a slot
            }
        }
    }
}

// Exclusive prefix sum over the marks: block -> slot, slot -> block, and the
// node kernels' indirect dispatch. Blocks past the slot budget get NONE and
// the true total is left in nact[1] for the host to shout about.
@compute @workgroup_size(256)
fn blk_scan(@builtin(local_invocation_index) t: u32) {
    let nb = nblocks();
    let chunk = (nb + 255u) / 256u;
    let lo = t * chunk;
    let hi = min(lo + chunk, nb);
    var sum = 0u;
    for (var b = lo; b < hi; b++) {
        sum += btab[b];
    }
    partial[t] = sum;
    workgroupBarrier();
    if (t == 0u) {
        var acc = 0u;
        for (var i = 0u; i < 256u; i++) {
            let c = partial[i];
            partial[i] = acc;
            acc += c;
        }
        let used = min(acc, P.misc2.x);
        atomicStore(&nact[0], used);
        atomicMax(&nact[1], acc);
        // ceil(used*64 / 256) workgroups, folded into 2D past the 65535 limit
        let g = (used * 64u + 255u) / 256u;
        indirect[0] = min(g, 65535u);
        indirect[1] = (g + 65534u) / 65535u;
        indirect[2] = 1u;
        // and one workgroup per active block for the scatter (offset 16 bytes)
        indirect[4] = min(used, 65535u);
        indirect[5] = (used + 65534u) / 65535u;
        indirect[6] = 1u;
    }
    workgroupBarrier();
    var run = partial[t];
    for (var b = lo; b < hi; b++) {
        if (btab[b] == 0u) {
            btab[b] = NONE;
        } else if (run < P.misc2.x) {
            btab[b] = run;
            alist[run] = b;
            run += 1u;
        } else {
            btab[b] = NONE;
            run += 1u;
        }
    }
}

@compute @workgroup_size(256)
fn sort_scatter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let p = linear_id(gid, nwg);
    if (p >= P.n.w) { return; }
    let b = block_of(x[p].xyz);
    let i = atomicAdd(&fill[b], 1u);
    perm[offsets[b] + i] = p;
}

@compute @workgroup_size(256)
fn permute(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linear_id(gid, nwg);
    if (i >= P.n.w) { return; }
    let p = perm[i];
    xo[i] = x[p];
    vo[i] = v[p];
    co[3u * i] = c[3u * p];
    co[3u * i + 1u] = c[3u * p + 1u];
    co[3u * i + 2u] = c[3u * p + 2u];
}

// P2G by block: one workgroup per 4x4x4-cell block, its particles contiguous
// after the sort; contributions accumulate in a 7^3 tile of workgroup
// atomics and are flushed to the grid once
@compute @workgroup_size(256)
fn p2g_block(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    // one workgroup per ACTIVE block (dispatched indirectly from blk_scan):
    // over a table that spans the pool, launching a workgroup per block to
    // have it return was most of the substep
    let slot = wg.x + wg.y * nwg.x;
    if (slot >= atomicLoad(&nact[0])) { return; }
    let b = alist[slot];
    // an empty block has nothing to scatter, and skipping it here saves
    // zeroing and scanning the 7^3 tile for most of the box
    if (offsets[b] == offsets[b + 1u]) { return; }
    for (var i = t; i < TILE; i += 256u) {
        atomicStore(&tile_m[i], 0);
        atomicStore(&tile_p[3u * i], 0);
        atomicStore(&tile_p[3u * i + 1u], 0);
        atomicStore(&tile_p[3u * i + 2u], 0);
    }
    workgroupBarrier();
    let nbx = i32(P.misc.y);
    let nby = i32(P.misc.z);
    let bi = i32(b);
    let bx = bi % nbx;
    let by = (bi / nbx) % nby;
    let bz = bi / (nbx * nby);
    let node0 = vec3<i32>(bx, by, bz) * BLK - vec3<i32>(1);
    let start = offsets[b];
    let end = offsets[b + 1u];
    let h = P.origin_h.w;
    let inv_h = P.k.y;
    let dt = P.k.x;
    let mass = P.k.z;
    let vol0 = P.k.w;
    let bulk = P.k2.x;
    let d_inv = P.k2.w;
    for (var p = start + t; p < end; p += 256u) {
        let xp = x[p];
        let rel = (xp.xyz - P.origin_h.xyz) * inv_h;
        let base = floor(rel - vec3<f32>(0.5));
        let fx = rel - base;
        let wx = w1(fx.x);
        let wy = w1(fx.y);
        let wz = w1(fx.z);
        let dj = xp.w;
        let jp = 1.0 + dj;
        let pressure = max(-bulk * dj / jp, 0.0);
        let s = pressure * dt * vol0 * jp * d_inv / mass;
        let c0 = c[3u * p].xyz;
        let c1 = c[3u * p + 1u].xyz;
        let c2 = c[3u * p + 2u].xyz;
        let vp = v[p].xyz;
        let lb = vec3<i32>(base) - node0; // tile-local base, in [0, 4]
        for (var di = 0; di < 3; di++) {
            for (var dj2 = 0; dj2 < 3; dj2++) {
                for (var dk = 0; dk < 3; dk++) {
                    let l = lb + vec3<i32>(di, dj2, dk);
                    let dpos = (vec3<f32>(f32(di), f32(dj2), f32(dk)) - fx) * h;
                    let wt = wx[di] * wy[dj2] * wz[dk];
                    let ti = u32((l.z * TN + l.y) * TN + l.x);
                    let mom = (vp + s * dpos + c0 * dpos.x + c1 * dpos.y + c2 * dpos.z) * wt;
                    atomicAdd(&tile_m[ti], i32(round(wt * MASS_SCALE)));
                    atomicAdd(&tile_p[3u * ti], i32(round(mom.x * MOM_SCALE)));
                    atomicAdd(&tile_p[3u * ti + 1u], i32(round(mom.y * MOM_SCALE)));
                    atomicAdd(&tile_p[3u * ti + 2u], i32(round(mom.z * MOM_SCALE)));
                }
            }
        }
    }
    workgroupBarrier();
    for (var i = t; i < TILE; i += 256u) {
        let m = atomicLoad(&tile_m[i]);
        if (m == 0) { continue; }
        let ii = i32(i);
        let l = vec3<i32>(ii % TN, (ii / TN) % TN, ii / (TN * TN));
        let g = node0 + l;
        if (g.x < 0 || g.y < 0 || g.z < 0 || g.x >= i32(P.n.x) || g.y >= i32(P.n.y) || g.z >= i32(P.n.z)) { continue; }
        let gs = node_index(g.x, g.y, g.z);
        if (gs < 0) { continue; }
        let gi = u32(gs);
        atomicAdd(&gm[gi], m);
        atomicAdd(&gmom[3u * gi], atomicLoad(&tile_p[3u * i]));
        atomicAdd(&gmom[3u * gi + 1u], atomicLoad(&tile_p[3u * i + 1u]));
        atomicAdd(&gmom[3u * gi + 2u], atomicLoad(&tile_p[3u * i + 2u]));
    }
}

// 3x3x3 box blur of the node mass into gvold.w, between the grid pass and G2P
@compute @workgroup_size(256)
fn blur(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = linear_id(gid, nwg);
    if (g >= active_nodes()) { return; }
    let ijk = node_of(g);
    if (ijk.x >= i32(P.n.x) || ijk.y >= i32(P.n.y) || ijk.z >= i32(P.n.z)) { return; }
    let i = ijk.x;
    let j = ijk.y;
    let k = ijk.z;
    let hh = P.origin_h.w;
    let full_node = P.k.z * hh * hh * hh / P.k.w; // a node's mass at rest density
    var acc = 0.0;
    for (var dk = -1; dk <= 1; dk++) {
        for (var dj = -1; dj <= 1; dj++) {
            for (var di = -1; di <= 1; di++) {
                // mirrored at the walls: see the CPU solver's g_blur
                let a = clamp(i + di, 2, i32(P.n.x) - 3);
                let b = clamp(j + dj, 2, i32(P.n.y) - 3);
                let cc = clamp(k + dk, 2, i32(P.n.z) - 3);
                // inside the body: the nearest fluid node across the
                // surface, the same Neumann extension the walls get from the
                // clamp above (see the CPU solver)
                let xn = P.origin_h.xyz + vec3<f32>(f32(a), f32(b), f32(cc)) * P.origin_h.w;
                let sd = body_sdf(xn);
                var aa = a; var bb = b; var ccc = cc;
                if (sd.w < 0.0) {
                    let q = (xn + sd.xyz * (hh - sd.w) - P.origin_h.xyz) * P.k.y;
                    aa = clamp(i32(round(q.x)), 2, i32(P.n.x) - 3);
                    bb = clamp(i32(round(q.y)), 2, i32(P.n.y) - 3);
                    ccc = clamp(i32(round(q.z)), 2, i32(P.n.z) - 3);
                }
                // an inactive neighbour is empty; the dilation guarantees every
                // node with mass, and every mirror target of one, is in an active block
                // beyond the region: the far field's water, at rest (see the CPU solver)
                if (outside_region(aa, bb)) {
                    if (xn.z < 0.0) { acc += full_node; }
                } else {
                    let gn = node_index(aa, bb, ccc);
                    if (gn >= 0) { acc += gvel[u32(gn)].w; }
                }
            }
        }
    }
    let old = gvold[g];
    gvold[g] = vec4<f32>(old.xyz, acc / 27.0);
}
