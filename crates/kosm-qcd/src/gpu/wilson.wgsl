// On-axis planar Wilson loops W(R, T) = Re Tr / 3, averaged over the three
// spatial directions, for R ≤ RMAX (extra0) and T ≤ TMAX (extra1).
// Component-major out[((R − 1) TMAX + T − 1) n + site].

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> links: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

const RCAP: u32 = 12u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if (site >= p.n) {
        return;
    }
    let rmax = min(p.extra0, RCAP);
    let tmax = p.extra1;
    for (var k = 0u; k < rmax * tmax; k++) {
        out[k * p.n + site] = 0.0;
    }
    for (var i = 1u; i < 4u; i++) {
        // spatial lines from the base site at t0, and base sites along them
        var s0: array<M3, RCAP>;
        var base: array<u32, RCAP>;
        var line = m_id();
        var s = site;
        for (var r = 0u; r < rmax; r++) {
            line = m_mul(line, load(i, s));
            s = fwd(p, s, i);
            s0[r] = line;
            base[r] = s;
        }
        // temporal lines up from the base site (index 0) and from each x + R î
        var tl: array<M3, RCAP>;
        var top: array<u32, RCAP>;
        var t0 = m_id();
        var top0 = site;
        for (var r = 0u; r < rmax; r++) {
            tl[r] = m_id();
            top[r] = base[r];
        }
        for (var t = 0u; t < tmax; t++) {
            t0 = m_mul(t0, load(0u, top0));
            top0 = fwd(p, top0, 0u);
            for (var r = 0u; r < rmax; r++) {
                tl[r] = m_mul(tl[r], load(0u, top[r]));
                top[r] = fwd(p, top[r], 0u);
            }
            // spatial line along the top, closed back down the base line
            var up = m_id();
            var su = top0;
            for (var r = 0u; r < rmax; r++) {
                up = m_mul(up, load(i, su));
                su = fwd(p, su, i);
                let w = m_re_tr(m_mul_dag(m_mul_dag(m_mul(s0[r], tl[r]), up), t0)) / 3.0;
                out[(r * tmax + t) * p.n + site] += w / 3.0;
            }
        }
    }
}
