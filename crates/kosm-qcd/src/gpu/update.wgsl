// Cabibbo-Marinari heatbath / overrelaxation on one direction and one
// checkerboard parity: every link in the set has a staple built only from
// links outside it, so all of them update at once.

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> links: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> sites: array<u32>;

fn store(mu: u32, site: u32, m: M3) {
    let b = (mu * p.n + site) * 9u;
    for (var k = 0u; k < 9u; k++) {
        links[b + k] = m.e[k];
    }
}

// quaternion a0 + i a·σ  ↔  [[a0 + i a3, a2 + i a1], [−a2 + i a1, a0 − i a3]]
fn q_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
        a.x * b.y + b.x * a.y - (a.z * b.w - a.w * b.z),
        a.x * b.z + b.x * a.z - (a.w * b.y - a.y * b.w),
        a.x * b.w + b.x * a.w - (a.y * b.z - a.z * b.y),
    );
}

fn q_conj(a: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.x, -a.y, -a.z, -a.w);
}

fn unit3(st: ptr<function, u32>) -> vec3<f32> {
    let c = 2.0 * rnd(st) - 1.0;
    let s = sqrt(max(1.0 - c * c, 0.0));
    let phi = 6.2831853 * rnd(st);
    return vec3<f32>(s * cos(phi), s * sin(phi), c);
}

// Kennedy-Pendleton: SU(2) element with density ∝ exp(α y0).
fn kp(alpha: f32, st: ptr<function, u32>) -> vec4<f32> {
    var l2 = 1.0;
    for (var tries = 0u; tries < 64u; tries++) {
        let r1 = rnd(st);
        let r2 = rnd(st);
        let r3 = rnd(st);
        let c = cos(6.2831853 * r2);
        l2 = -(log(r1) + c * c * log(r3)) / (2.0 * alpha);
        let r4 = rnd(st);
        if (r4 * r4 <= 1.0 - l2) {
            break;
        }
    }
    let y0 = clamp(1.0 - 2.0 * l2, -1.0, 1.0);
    return vec4<f32>(y0, unit3(st) * sqrt(max(1.0 - y0 * y0, 0.0)));
}

fn left_mul(m: M3, i: u32, j: u32, x: vec4<f32>) -> M3 {
    let x00 = vec2<f32>(x.x, x.w);
    let x01 = vec2<f32>(x.z, x.y);
    let x10 = vec2<f32>(-x.z, x.y);
    let x11 = vec2<f32>(x.x, -x.w);
    var r = m;
    for (var c = 0u; c < 3u; c++) {
        let mi = m.e[3u * i + c];
        let mj = m.e[3u * j + c];
        r.e[3u * i + c] = cmul(x00, mi) + cmul(x01, mj);
        r.e[3u * j + c] = cmul(x10, mi) + cmul(x11, mj);
    }
    return r;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = p.n / 2u;
    if (gid.x >= half) {
        return;
    }
    let site = sites[p.parity * half + gid.x];
    let mu = p.mu;
    var u = load(mu, site);
    var w = m_mul(u, staple(site, mu));
    var st = rng_init(p.seed, p.pass_id, mu, site);
    for (var sg = 0u; sg < 3u; sg++) {
        var i = 0u;
        var j = 1u;
        if (sg == 1u) { j = 2u; }
        if (sg == 2u) { i = 1u; j = 2u; }
        let wii = w.e[3u * i + i];
        let wij = w.e[3u * i + j];
        let wji = w.e[3u * j + i];
        let wjj = w.e[3u * j + j];
        let v = 0.5 * vec4<f32>(wii.x + wjj.x, wij.y + wji.y, wij.x - wji.x, wii.y - wjj.y);
        let k = length(v);
        var x: vec4<f32>;
        if (k < 1e-6) {
            x = normalize(vec4<f32>(rnd(&st) - 0.5, rnd(&st) - 0.5, rnd(&st) - 0.5, rnd(&st) - 0.5));
        } else {
            let vh = v / k;
            if (p.kind == 0u) {
                x = q_mul(kp(2.0 * p.beta * k / 3.0, &st), q_conj(vh));
            } else {
                x = q_mul(q_conj(vh), q_conj(vh));
            }
        }
        u = left_mul(u, i, j, x);
        w = left_mul(w, i, j, x);
    }
    store(mu, site, m_reunit(u));
}
