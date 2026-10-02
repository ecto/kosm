// Static three-quark loop W3Q = ε ε M1 M2 M3 / 6 with the junction at each
// base site. Quark q's staircase is `paths.steps` from q * 32, `paths.len[q]`
// long; each step packs its axis (1..3) in bits 0..3 and forward in bit 4.

struct Paths {
    len: vec4<u32>,
    steps: array<vec4<u32>, 24>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> links: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> paths: Paths;

fn step_at(q: u32, s: u32) -> u32 {
    let k = q * 32u + s;
    return paths.steps[k / 4u][k % 4u];
}

struct Walk {
    m: M3,
    end: u32,
};

fn walk(start: u32, q: u32) -> Walk {
    var m = m_id();
    var site = start;
    for (var s = 0u; s < paths.len[q]; s++) {
        let st = step_at(q, s);
        let mu = st & 15u;
        if ((st & 16u) != 0u) {
            m = m_mul(m, load(mu, site));
            site = fwd(p, site, mu);
        } else {
            site = bwd(p, site, mu);
            m = m_mul_dag(m, load(mu, site));
        }
    }
    return Walk(m, site);
}

fn time_line(start: u32) -> Walk {
    var m = m_id();
    var site = start;
    for (var t = 0u; t < p.t_len; t++) {
        m = m_mul(m, load(0u, site));
        site = fwd(p, site, 0u);
    }
    return Walk(m, site);
}

// ε_abc ε_a'b'c' A_aa' B_bb' C_cc'
fn eps3(a: M3, b: M3, c: M3) -> vec2<f32> {
    var perm = array<vec3<u32>, 6>(
        vec3<u32>(0u, 1u, 2u), vec3<u32>(1u, 2u, 0u), vec3<u32>(2u, 0u, 1u),
        vec3<u32>(0u, 2u, 1u), vec3<u32>(2u, 1u, 0u), vec3<u32>(1u, 0u, 2u),
    );
    var sum = vec2<f32>(0.0);
    for (var x = 0u; x < 6u; x++) {
        for (var y = 0u; y < 6u; y++) {
            let px = perm[x];
            let py = perm[y];
            var s = 1.0;
            if ((x < 3u) != (y < 3u)) { s = -1.0; }
            let t = cmul(cmul(a.e[3u * px.x + py.x], b.e[3u * px.y + py.y]), c.e[3u * px.z + py.z]);
            sum += s * t;
        }
    }
    return sum;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if (site >= p.n) {
        return;
    }
    let top = time_line(site).end;
    var m: array<M3, 3>;
    for (var q = 0u; q < 3u; q++) {
        let lower = walk(site, q);
        let line = time_line(lower.end);
        let upper = walk(top, q);
        m[q] = m_mul_dag(m_mul(lower.m, line.m), upper.m);
    }
    out[site] = eps3(m[0], m[1], m[2]).x / 6.0;
}
