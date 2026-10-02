// Per-site field densities, component-major out[c * n + site]:
// 0 mean plaquette Re Tr / 3, 1 action density −Σ Tr F², 2 topological
// charge density, 3..5 chromo-electric −Tr F_0i² for i = x, y, z.

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> links: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if (site >= p.n) {
        return;
    }
    var plaq = 0.0;
    var f: array<M3, 6>;
    var k = 0u;
    for (var mu = 0u; mu < 4u; mu++) {
        for (var nu = mu + 1u; nu < 4u; nu++) {
            plaq += m_re_tr(plaquette(site, mu, nu));
            f[k] = m_scale(m_ta(clover(site, mu, nu)), 0.25);
            k++;
        }
    }
    var action = 0.0;
    for (var a = 0u; a < 6u; a++) {
        action -= m_re_tr(m_mul(f[a], f[a]));
    }
    // ε expands to 8 (F01 F23 − F02 F13 + F03 F12), planes in (01 02 03 12 13 23) order
    let topo = (8.0 / (32.0 * 9.8696044)) * (m_re_tr(m_mul(f[0], f[5])) - m_re_tr(m_mul(f[1], f[4])) + m_re_tr(m_mul(f[2], f[3])));
    out[site] = plaq / 18.0;
    out[p.n + site] = action;
    out[2u * p.n + site] = topo;
    out[3u * p.n + site] = -m_re_tr(m_mul(f[0], f[0]));
    out[4u * p.n + site] = -m_re_tr(m_mul(f[1], f[1]));
    out[5u * p.n + site] = -m_re_tr(m_mul(f[2], f[2]));
}
