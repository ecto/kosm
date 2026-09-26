// C_k(r) numerators: Σ over junction sites x0 of W3Q(x0) · S_k(x0 + r) on the
// time slice t0 + T/2, one thread per spatial offset r. Densities from the
// fields kernel: k = 0 action (component 1), k = 1 chromo-electric (3 + 4 + 5).

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> dens: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nt = p.dims.x;
    let nx = p.dims.y;
    let ny = p.dims.z;
    let nz = p.dims.w;
    let ns = nx * ny * nz;
    if (gid.x >= ns) {
        return;
    }
    let rx = gid.x % nx;
    let ry = (gid.x / nx) % ny;
    let rz = gid.x / (nx * ny);
    let tm = p.t_len / 2u;
    var a = 0.0;
    var e = 0.0;
    for (var z = 0u; z < nz; z++) {
        let zz = (z + rz) % nz;
        for (var y = 0u; y < ny; y++) {
            let yy = (y + ry) % ny;
            for (var x = 0u; x < nx; x++) {
                let xx = (x + rx) % nx;
                for (var t = 0u; t < nt; t++) {
                    let wv = w[t + nt * (x + nx * (y + ny * z))];
                    let s = (t + tm) % nt + nt * (xx + nx * (yy + ny * zz));
                    a += wv * dens[p.n + s];
                    e += wv * (dens[3u * p.n + s] + dens[4u * p.n + s] + dens[5u * p.n + s]);
                }
            }
        }
    }
    out[gid.x] = a;
    out[ns + gid.x] = e;
}
