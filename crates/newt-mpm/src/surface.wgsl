// The free surface, extracted on the GPU from the grid mass, and the
// particles worth downloading.
//
// This is `newt_spike::splash::Water::surface` and the first half of
// `droplets`/the foam scan, moved off the CPU. The semantics are copied
// exactly, step for step, because the CPU version stays the reference:
// a 3x3x3 box blur of the mass fraction, a top-down scan per column for the
// node where the blurred fraction crosses one half (interpolated against the
// node above), four passes of a 3x3 smooth, and a per-cell rest map
// subtracted at the end. What comes back is 125x125 floats instead of two
// hundred megabytes of particles.
//
// The picker is the other half of the saving: instead of downloading every
// particle so the CPU can find the two hundred and fifty highest drops and
// the fast water near the surface, a compaction kernel keeps only the
// candidates and an atomic counter says how many there are.

struct Surf {
    origin_h: vec4<f32>,  // grid origin.xyz, h
    n: vec4<u32>,         // grid nx, ny, nz, particle count
    map: vec4<u32>,       // height map nx, ny, has_rest, candidate cap
    geom: vec4<f32>,      // map origin.xy, cell, floor z (-BOX_DEPTH)
    k: vec4<f32>,         // full node mass, level offset, drop height, foam speed
    band: vec4<f32>,      // foam band below the surface, above it, unused, unused
    blocks: vec4<u32>,    // blocks per axis nbx, nby, nbz, and the slot budget
}

@group(0) @binding(0) var<uniform> S: Surf;
@group(0) @binding(1) var<storage, read> gvel: array<vec4<f32>>;   // vel.xyz, mass
@group(0) @binding(2) var<storage, read_write> frac: array<f32>;   // blurred mass fraction
@group(0) @binding(3) var<storage, read_write> ha: array<f32>;     // height, ping
@group(0) @binding(4) var<storage, read_write> hb: array<f32>;     // height, pong
@group(0) @binding(5) var<storage, read> rest: array<f32>;         // the rest map
@group(0) @binding(6) var<storage, read> px: array<vec4<f32>>;     // particle xyz, J-1
@group(0) @binding(7) var<storage, read> pv: array<vec4<f32>>;     // particle vel.xyz, id
@group(0) @binding(8) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> cx: array<vec4<f32>>;  // candidate xyz, crowd
@group(0) @binding(10) var<storage, read_write> cv: array<vec4<f32>>; // candidate vel.xyz, id
// the block-sparse grid's table: block -> slot (or NONE), slot -> block, active count
@group(0) @binding(11) var<storage, read> btab: array<u32>;
@group(0) @binding(12) var<storage, read> alist: array<u32>;
@group(0) @binding(13) var<storage, read> nact: array<u32>;

const BLK: i32 = 4;
const NONE: u32 = 0xffffffffu;

const WG: u32 = 256u;

fn linear_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * WG;
}

// Compact node index through the block table, -1 for an inactive block
// (which holds no mass). The same layout as the physics kernels.
fn node_index(i: i32, j: i32, k: i32) -> i32 {
    if (i < 0 || j < 0 || k < 0 || u32(i) >= S.n.x || u32(j) >= S.n.y || u32(k) >= S.n.z) { return -1; }
    let b = ((k / BLK) * i32(S.blocks.y) + (j / BLK)) * i32(S.blocks.x) + (i / BLK);
    let sl = btab[u32(b)];
    if (sl == NONE) { return -1; }
    return i32(sl) * 64 + (((k & 3) * 4 + (j & 3)) * 4 + (i & 3));
}

fn mass_at(i: i32, j: i32, k: i32) -> f32 {
    let g = node_index(i, j, k);
    if (g < 0) { return 0.0; }
    return gvel[u32(g)].w;
}

fn frac_at(i: i32, j: i32, k: i32) -> f32 {
    let g = node_index(i, j, k);
    if (g < 0) { return 0.0; }
    return frac[u32(g)];
}

// The node (i,j,k) a compact index belongs to.
fn node_of(g: u32) -> vec3<i32> {
    let slot = g / 64u;
    let l = i32(g % 64u);
    let b = i32(alist[slot]);
    let nbx = i32(S.blocks.x);
    let nby = i32(S.blocks.y);
    let bx = b % nbx;
    let by = (b / nbx) % nby;
    let bz = b / (nbx * nby);
    return vec3<i32>(bx * BLK + (l & 3), by * BLK + ((l >> 2) & 3), bz * BLK + (l >> 4));
}

// The mass fraction, 3x3x3-blurred over the interior; the boundary shell keeps
// its raw value, as on the CPU.
@compute @workgroup_size(256)
fn surf_blur(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    // over the active nodes only; an inactive neighbour holds no mass
    let g = linear_id(gid, nwg);
    if (g >= nact[0] * 64u) { return; }
    let ijk = node_of(g);
    let i = ijk.x;
    let j = ijk.y;
    let k = ijk.z;
    let full = S.k.x;
    let interior = i > 0 && j > 0 && k > 0 && u32(i) + 1u < S.n.x && u32(j) + 1u < S.n.y && u32(k) + 1u < S.n.z;
    if (!interior) {
        frac[g] = gvel[g].w / full;
        return;
    }
    var acc = 0.0;
    for (var dk = -1; dk <= 1; dk = dk + 1) {
        for (var dj = -1; dj <= 1; dj = dj + 1) {
            for (var di = -1; di <= 1; di = di + 1) {
                acc = acc + mass_at(i + di, j + dj, k + dk);
            }
        }
    }
    frac[g] = acc / (27.0 * full);
}

// One thread per height-map cell: the top-down scan for the half crossing.
@compute @workgroup_size(256)
fn surf_scan(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = linear_id(gid, nwg);
    if (g >= S.map.x * S.map.y) { return; }
    let ix = g % S.map.x;
    let jy = g / S.map.x;
    let cell = S.geom.z;
    let x = S.geom.x + (f32(ix) + 0.5) * cell;
    let y = S.geom.y + (f32(jy) + 0.5) * cell;
    let h = S.origin_h.w;
    let gi = clamp(i32(round((x - S.origin_h.x) / h)), 0, i32(S.n.x) - 1);
    let gj = clamp(i32(round((y - S.origin_h.y) / h)), 0, i32(S.n.y) - 1);
    var z = S.geom.w;
    var prev = 0.0;
    for (var k = i32(S.n.z) - 1; k >= 0; k = k - 1) {
        let f = frac_at(gi, gj, k);
        if (f >= 0.5) {
            var t = 0.0;
            if (prev < 0.5 && u32(k) + 1u < S.n.z) {
                t = (0.5 - prev) / max(f - prev, 1e-9);
            }
            z = S.origin_h.z + f32(k) * h + (1.0 - t) * h;
            break;
        }
        prev = f;
    }
    ha[g] = z;
}

// A 3x3 smooth of the interior, ha -> hb; the caller runs it four times,
// swapping the bind group so the answer lands back in ha.
@compute @workgroup_size(256)
fn surf_smooth(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = linear_id(gid, nwg);
    let nx = S.map.x;
    let ny = S.map.y;
    if (g >= nx * ny) { return; }
    let ix = g % nx;
    let jy = g / nx;
    if (ix == 0u || jy == 0u || ix + 1u >= nx || jy + 1u >= ny) {
        hb[g] = ha[g];
        return;
    }
    var acc = 0.0;
    for (var dy = 0u; dy < 3u; dy = dy + 1u) {
        for (var dx = 0u; dx < 3u; dx = dx + 1u) {
            acc = acc + ha[(jy + dy - 1u) * nx + ix + dx - 1u];
        }
    }
    hb[g] = acc / 9.0;
}

// The rest map (or, before it is captured, a flat level) comes off last.
@compute @workgroup_size(256)
fn surf_finish(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = linear_id(gid, nwg);
    if (g >= S.map.x * S.map.y) { return; }
    // the rest map cancels the lattice's static extraction noise, but only
    // where it recorded water: a column that was empty at rest (the floor,
    // beyond the disc) and gains a few particles later would otherwise read
    // two metres high, start every ray's march up there and spawn foam
    let floor = S.geom.w;
    if (S.map.z == 1u && rest[g] > floor + 0.05) {
        ha[g] = ha[g] - rest[g];
    } else {
        ha[g] = ha[g] - S.k.y;
    }
}

// Bilinear read of the finished height map, in the same clamped form as
// `HeightGrid::at`.
fn height_at(x: f32, y: f32) -> f32 {
    let nx = S.map.x;
    let ny = S.map.y;
    let gx = clamp((x - S.geom.x) / S.geom.z - 0.5, 0.0, f32(nx - 1u) - 1e-6);
    let gy = clamp((y - S.geom.y) / S.geom.z - 0.5, 0.0, f32(ny - 1u) - 1e-6);
    let ix = u32(floor(gx));
    let iy = u32(floor(gy));
    let wx = gx - floor(gx);
    let wy = gy - floor(gy);
    return ha[iy * nx + ix] * (1.0 - wx) * (1.0 - wy)
        + ha[iy * nx + ix + 1u] * wx * (1.0 - wy)
        + ha[(iy + 1u) * nx + ix] * (1.0 - wx) * wy
        + ha[(iy + 1u) * nx + ix + 1u] * wx * wy;
}

// The particles the CPU still wants: the drops flying above the surface, and
// the fast water in the band around it that seeds foam. Everything else --
// the ninety-nine per cent of the pool that is just sitting there -- never
// leaves the GPU.
@compute @workgroup_size(256)
fn pick(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let p = linear_id(gid, nwg);
    if (p >= S.n.w) { return; }
    let xp = px[p];
    let vp = pv[p];
    // the height map is level-corrected for rendering; particle positions are
    // raw, so compare in raw terms
    let s = height_at(xp.x, xp.y) + S.k.y;
    let drop = xp.z > s + S.k.z;
    let speed = length(vp.xyz);
    let foam = speed >= S.k.w && xp.z >= s - S.band.x && xp.z <= s + S.band.y;
    if (!drop && !foam) { return; }
    let slot = atomicAdd(&cnt[0], 1u);
    if (slot >= S.map.w) { return; }
    // how much water shares the drop's cell, for the bead's size
    let h = S.origin_h.w;
    let gi = clamp(i32(round((xp.x - S.origin_h.x) / h)), 0, i32(S.n.x) - 1);
    let gj = clamp(i32(round((xp.y - S.origin_h.y) / h)), 0, i32(S.n.y) - 1);
    let gk = clamp(i32(round((xp.z - S.origin_h.z) / h)), 0, i32(S.n.z) - 1);
    let crowd = min(mass_at(gi, gj, gk) / S.k.x, 1.0);
    cx[slot] = vec4<f32>(xp.xyz, crowd);
    cv[slot] = vec4<f32>(vp.xyz, vp.w);
}
