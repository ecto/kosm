// One stout smearing step, all links at once: U' = exp(−ρ TA(U A)) U.
// kind 1 smears only spatial links with spatial staples, leaving time links
// untouched (the transfer matrix stays that of the Wilson action).

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> links: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x + gid.y * 65535u * 64u;
    if (id >= 4u * p.n) {
        return;
    }
    let mu = id / p.n;
    let site = id % p.n;
    let u = load(mu, site);
    var v = u;
    if (p.kind == 0u || mu != 0u) {
        let x = m_ta(m_mul(u, staple_from(site, mu, p.kind)));
        v = m_mul(m_exp(m_scale(x, -p.rho)), u);
    }
    let b = (mu * p.n + site) * 9u;
    for (var k = 0u; k < 9u; k++) {
        out[b + k] = v.e[k];
    }
}
