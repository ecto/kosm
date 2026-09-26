// SU(3) lattice gauge theory on the GPU: the shared prelude.
//
// Mirrors phyz-qft's `su3` module in f32 so results can be checked against
// it: same site order (t + nt (x + nx (y + ny z))), same staple and clover
// orientation, same Cabibbo-Marinari subgroup order.
//
// Links live in one storage array of complex numbers, nine per link, at
// ((mu * n + site) * 9 + 3 i + j) for row i, column j.

struct Params {
    dims: vec4<u32>,      // nt, nx, ny, nz
    n: u32,               // sites
    mu: u32,
    parity: u32,
    kind: u32,            // update: 0 heatbath, 1 overrelax
    seed: u32,
    pass_id: u32,
    t_len: u32,           // baryon loop time extent
    count: u32,           // generic element count
    beta: f32,
    rho: f32,
    extra0: u32,
    extra1: u32,
};

struct M3 {
    e: array<vec2<f32>, 9>,
};

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn conj2(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

fn m_zero() -> M3 {
    var m: M3;
    for (var k = 0u; k < 9u; k++) {
        m.e[k] = vec2<f32>(0.0);
    }
    return m;
}

fn m_id() -> M3 {
    var m = m_zero();
    m.e[0] = vec2<f32>(1.0, 0.0);
    m.e[4] = vec2<f32>(1.0, 0.0);
    m.e[8] = vec2<f32>(1.0, 0.0);
    return m;
}

fn m_mul(a: M3, b: M3) -> M3 {
    var r: M3;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            var s = vec2<f32>(0.0);
            for (var k = 0u; k < 3u; k++) {
                s += cmul(a.e[3u * i + k], b.e[3u * k + j]);
            }
            r.e[3u * i + j] = s;
        }
    }
    return r;
}

// a · b†
fn m_mul_dag(a: M3, b: M3) -> M3 {
    var r: M3;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            var s = vec2<f32>(0.0);
            for (var k = 0u; k < 3u; k++) {
                s += cmul(a.e[3u * i + k], conj2(b.e[3u * j + k]));
            }
            r.e[3u * i + j] = s;
        }
    }
    return r;
}

// a† · b
fn m_dag_mul(a: M3, b: M3) -> M3 {
    var r: M3;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            var s = vec2<f32>(0.0);
            for (var k = 0u; k < 3u; k++) {
                s += cmul(conj2(a.e[3u * k + i]), b.e[3u * k + j]);
            }
            r.e[3u * i + j] = s;
        }
    }
    return r;
}

fn m_dag(a: M3) -> M3 {
    var r: M3;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            r.e[3u * i + j] = conj2(a.e[3u * j + i]);
        }
    }
    return r;
}

fn m_add(a: M3, b: M3) -> M3 {
    var r: M3;
    for (var k = 0u; k < 9u; k++) {
        r.e[k] = a.e[k] + b.e[k];
    }
    return r;
}

fn m_scale(a: M3, s: f32) -> M3 {
    var r: M3;
    for (var k = 0u; k < 9u; k++) {
        r.e[k] = a.e[k] * s;
    }
    return r;
}

fn m_re_tr(a: M3) -> f32 {
    return a.e[0].x + a.e[4].x + a.e[8].x;
}

fn m_tr(a: M3) -> vec2<f32> {
    return a.e[0] + a.e[4] + a.e[8];
}

// Gram-Schmidt the first two rows, third row = conj(row0 × row1): det = 1.
fn m_reunit(a: M3) -> M3 {
    var r0 = array<vec2<f32>, 3>(a.e[0], a.e[1], a.e[2]);
    var r1 = array<vec2<f32>, 3>(a.e[3], a.e[4], a.e[5]);
    let n0 = inverseSqrt(dot(r0[0], r0[0]) + dot(r0[1], r0[1]) + dot(r0[2], r0[2]));
    for (var k = 0u; k < 3u; k++) {
        r0[k] *= n0;
    }
    var d = vec2<f32>(0.0);
    for (var k = 0u; k < 3u; k++) {
        d += cmul(conj2(r0[k]), r1[k]);
    }
    for (var k = 0u; k < 3u; k++) {
        r1[k] -= cmul(d, r0[k]);
    }
    let n1 = inverseSqrt(dot(r1[0], r1[0]) + dot(r1[1], r1[1]) + dot(r1[2], r1[2]));
    for (var k = 0u; k < 3u; k++) {
        r1[k] *= n1;
    }
    var r: M3;
    r.e[0] = r0[0]; r.e[1] = r0[1]; r.e[2] = r0[2];
    r.e[3] = r1[0]; r.e[4] = r1[1]; r.e[5] = r1[2];
    r.e[6] = conj2(cmul(r0[1], r1[2]) - cmul(r0[2], r1[1]));
    r.e[7] = conj2(cmul(r0[2], r1[0]) - cmul(r0[0], r1[2]));
    r.e[8] = conj2(cmul(r0[0], r1[1]) - cmul(r0[1], r1[0]));
    return r;
}

// Traceless anti-Hermitian part: (M − M†)/2 − Tr(M − M†)/6.
fn m_ta(a: M3) -> M3 {
    var r: M3;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            r.e[3u * i + j] = 0.5 * (a.e[3u * i + j] - conj2(a.e[3u * j + i]));
        }
    }
    let t = m_tr(r) / 3.0;
    r.e[0] -= t;
    r.e[4] -= t;
    r.e[8] -= t;
    return r;
}

// exp of a small anti-Hermitian Q: Taylor series, then back onto SU(3).
fn m_exp(q: M3) -> M3 {
    var result = m_id();
    var term = m_id();
    for (var k = 1u; k <= 8u; k++) {
        term = m_scale(m_mul(term, q), 1.0 / f32(k));
        result = m_add(result, term);
    }
    return m_reunit(result);
}

// ---- lattice indexing ------------------------------------------------------

fn coords(pp: Params, site: u32) -> vec4<u32> {
    let nt = pp.dims.x;
    let nx = pp.dims.y;
    let ny = pp.dims.z;
    return vec4<u32>(site % nt, (site / nt) % nx, (site / (nt * nx)) % ny, site / (nt * nx * ny));
}

fn index(pp: Params, c: vec4<u32>) -> u32 {
    return c.x + pp.dims.x * (c.y + pp.dims.y * (c.z + pp.dims.z * c.w));
}

fn fwd(pp: Params, site: u32, mu: u32) -> u32 {
    var c = coords(pp, site);
    c[mu] = (c[mu] + 1u) % pp.dims[mu];
    return index(pp, c);
}

fn bwd(pp: Params, site: u32, mu: u32) -> u32 {
    var c = coords(pp, site);
    c[mu] = (c[mu] + pp.dims[mu] - 1u) % pp.dims[mu];
    return index(pp, c);
}

// ---- random numbers: a PCG hash per (seed, pass, mu, site) stream ----------

fn pcg(x: u32) -> u32 {
    let s = x * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rng_init(seed: u32, pass_id: u32, mu: u32, site: u32) -> u32 {
    return pcg(seed ^ pcg(pass_id ^ pcg(mu ^ pcg(site + 0x9e3779b9u))));
}

// Uniform in (0, 1).
fn rnd(state: ptr<function, u32>) -> f32 {
    *state = pcg(*state);
    return (f32(*state >> 8u) + 0.5) / 16777216.0;
}
