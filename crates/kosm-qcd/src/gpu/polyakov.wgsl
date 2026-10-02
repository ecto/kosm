// Polyakov loop Tr Π_t U_0(t, x) / 3 at every spatial site x + nx (y + ny z).

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> links: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ns = p.dims.y * p.dims.z * p.dims.w;
    if (gid.x >= ns) {
        return;
    }
    let x = gid.x % p.dims.y;
    let y = (gid.x / p.dims.y) % p.dims.z;
    let z = gid.x / (p.dims.y * p.dims.z);
    var site = index(p, vec4<u32>(0u, x, y, z));
    var m = m_id();
    for (var t = 0u; t < p.dims.x; t++) {
        m = m_mul(m, load(0u, site));
        site = fwd(p, site, 0u);
    }
    out[gid.x] = m_tr(m) / 3.0;
}
