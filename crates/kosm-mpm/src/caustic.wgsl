// Sunlight through the water onto the tiles, on the GPU.
//
// This is `kosm_spike::pool::caustic`, kernel for kernel: one thread per
// launch ray, the composed surface (the fine MPM height field inside the box,
// the spectral far field outside it, a smoothstep blend across the seam, and
// the millimetre of ambient ripple over both) evaluated in the shader, Snell
// through the normal, Fresnel for the weight, and a bilinear deposit onto the
// floor map through fixed-point atomics -- the same trick the P2G kernel uses,
// because WGSL still has no atomic add on floats.
//
// The CPU version stays the reference; `KOSM_GPU_RENDER=0` selects it.

struct Caus {
    map: vec4<u32>,       // map nx, ny, launch nx, launch ny
    geom: vec4<f32>,      // map origin.xy, cell, rays per cell per axis
    launch: vec4<f32>,    // launch origin.xy, depth, time
    sun: vec4<f32>,       // the direction light travels, xyz; n_water in w
    flat: vec4<f32>,      // flat refraction dir.xyz, flat transmission
    fine: vec4<f32>,      // fine grid origin.xy, cell, (nx | ny << 16)
    fine_n: vec4<u32>,    // fine nx, ny, far nx, far ny
    far: vec4<f32>,       // far grid origin.xy, cell, has_fine
    band: vec4<f32>,      // box half, sponge, blend, unused
}

@group(0) @binding(0) var<uniform> C: Caus;
@group(0) @binding(1) var<storage, read> fine: array<f32>;
@group(0) @binding(2) var<storage, read> far: array<f32>;
@group(0) @binding(3) var<storage, read_write> acc: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const WG: u32 = 256u;
const SCALE: f32 = 1048576.0; // 2^20, as in the P2G accumulators

fn linear_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * WG;
}

// `HeightGrid::at`, on either grid.
fn grid_at(base: u32, nx: u32, ny: u32, ox: f32, oy: f32, cell: f32, x: f32, y: f32) -> f32 {
    let gx = clamp((x - ox) / cell - 0.5, 0.0, f32(nx - 1u) - 1e-6);
    let gy = clamp((y - oy) / cell - 0.5, 0.0, f32(ny - 1u) - 1e-6);
    let ix = u32(floor(gx));
    let iy = u32(floor(gy));
    let wx = gx - floor(gx);
    let wy = gy - floor(gy);
    if (base == 0u) {
        return fine[iy * nx + ix] * (1.0 - wx) * (1.0 - wy)
            + fine[iy * nx + ix + 1u] * wx * (1.0 - wy)
            + fine[(iy + 1u) * nx + ix] * (1.0 - wx) * wy
            + fine[(iy + 1u) * nx + ix + 1u] * wx * wy;
    }
    return far[iy * nx + ix] * (1.0 - wx) * (1.0 - wy)
        + far[iy * nx + ix + 1u] * wx * (1.0 - wy)
        + far[(iy + 1u) * nx + ix] * (1.0 - wx) * wy
        + far[(iy + 1u) * nx + ix + 1u] * wx * wy;
}

fn height(x: f32, y: f32) -> f32 {
    let t = C.launch.w;
    let ambient = 0.0008 * (sin(7.0 * x + 3.0 * t) * cos(5.0 * y - 2.0 * t)) + 0.0005 * sin(11.0 * x - 4.0 * y + 1.7 * t);
    if (C.far.w == 0.0) { return ambient; }
    let inset = C.band.x - length(vec2<f32>(x, y)); // the region is a disc
    let f = grid_at(1u, C.fine_n.z, C.fine_n.w, C.far.x, C.far.y, C.far.z, x, y) + ambient;
    if (inset <= C.band.y) { return f; }
    let g = grid_at(0u, C.fine_n.x, C.fine_n.y, C.fine.x, C.fine.y, C.fine.z, x, y) + ambient;
    if (inset > C.band.y + C.band.z) { return g; }
    let u = (inset - C.band.y) / C.band.z;
    let w = u * u * (3.0 - 2.0 * u);
    return w * g + (1.0 - w) * f;
}

fn normal(x: f32, y: f32) -> vec3<f32> {
    let e = 1e-3;
    let dx = (height(x + e, y) - height(x - e, y)) / (2.0 * e);
    let dy = (height(x, y + e) - height(x, y - e)) / (2.0 * e);
    return normalize(vec3<f32>(-dx, -dy, 1.0));
}

fn fresnel(n1: f32, n2: f32, ci: f32, ct: f32) -> f32 {
    let rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
    let rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
    return 0.5 * (rs * rs + rp * rp);
}

@compute @workgroup_size(256)
fn caustic_clear(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = linear_id(gid, nwg);
    if (g >= C.map.x * C.map.y) { return; }
    atomicStore(&acc[g], 0);
}

@compute @workgroup_size(256)
fn caustic_trace(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = linear_id(gid, nwg);
    let sub = u32(C.geom.w);
    let lw = C.map.z * sub;
    let lh = C.map.w * sub;
    if (g >= lw * lh) { return; }
    let ix = g % lw;
    let iy = g / lw;
    let cell = C.geom.z;
    let nw = C.sun.w;
    let d = C.sun.xyz;
    let d_flat = C.flat.xyz;
    let t_flat = C.flat.w;
    let flat_cos = -d_flat.z;
    // launch from the surface point the flat refraction would send to this
    // floor cell, so the reference is uniform
    let fx = C.launch.x + (f32(ix) + 0.5) * cell / C.geom.w;
    let fy = C.launch.y + (f32(iy) + 0.5) * cell / C.geom.w;
    let back = C.launch.z / flat_cos;
    let sx = fx - d_flat.x * back;
    let sy = fy - d_flat.y * back;
    let n = normal(sx, sy);
    let eta = 1.0 / nw;
    let ci = -dot(d, n);
    let kk = 1.0 - eta * eta * (1.0 - ci * ci);
    if (kk < 0.0) { return; }
    let ct = sqrt(kk);
    let dr = d * eta + n * (eta * ci - ct);
    let tr = 1.0 - fresnel(1.0, nw, ci, ct);
    let z0 = height(sx, sy);
    let tt = (z0 + C.launch.z) / -dr.z;
    let hx = sx + dr.x * tt;
    let hy = sy + dr.y * tt;
    let w = (1.0 / (C.geom.w * C.geom.w)) * (tr / t_flat) * (-dr.z / flat_cos);
    let gx = (hx - C.geom.x) / cell - 0.5;
    let gy = (hy - C.geom.y) / cell - 0.5;
    let bx = floor(gx);
    let by = floor(gy);
    let wx = gx - bx;
    let wy = gy - by;
    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            let jx = i32(bx) + dx;
            let jy = i32(by) + dy;
            if (jx < 0 || jy < 0 || jx >= i32(C.map.x) || jy >= i32(C.map.y)) { continue; }
            var ww = wx;
            if (dx == 0) { ww = 1.0 - wx; }
            var wv = wy;
            if (dy == 0) { wv = 1.0 - wy; }
            atomicAdd(&acc[u32(jy) * C.map.x + u32(jx)], i32(w * ww * wv * SCALE));
        }
    }
}

@compute @workgroup_size(256)
fn caustic_resolve(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = linear_id(gid, nwg);
    if (g >= C.map.x * C.map.y) { return; }
    out[g] = f32(atomicLoad(&acc[g])) / SCALE;
}
