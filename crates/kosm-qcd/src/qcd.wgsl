// kosm-qcd: gluon field surfaces and quark markers, lit two-sided on black.

struct Uniforms {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    light: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) col: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) wpos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) col: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var o: VsOut;
    o.clip = u.view_proj * vec4<f32>(in.pos, 1.0);
    o.wpos = in.pos;
    o.nrm = in.nrm;
    o.col = in.col;
    return o;
}

// instanced markers: a unit mesh placed by a per-instance matrix and tinted
struct InstIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) col: vec4<f32>,
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
    @location(8) tint: vec4<f32>,
};

@vertex
fn vs_inst(in: InstIn) -> VsOut {
    let m = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let wp = m * vec4<f32>(in.pos, 1.0);
    var o: VsOut;
    o.clip = u.view_proj * wp;
    o.wpos = wp.xyz;
    o.nrm = normalize((m * vec4<f32>(in.nrm, 0.0)).xyz);
    o.col = in.col * in.tint;
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let v = normalize(u.eye.xyz - in.wpos);
    var n = normalize(in.nrm);
    if (dot(n, v) < 0.0) {
        n = -n;
    }
    let l = normalize(u.light.xyz);
    let h = normalize(l + v);
    let diff = max(dot(n, l), 0.0);
    let spec = pow(max(dot(n, h), 0.0), 48.0);
    let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    let base = in.col.rgb;
    let c = base * (0.28 + 0.72 * diff) + vec3<f32>(0.35) * spec + base * 0.25 * rim;
    return vec4<f32>(c, 1.0);
}

struct BlitOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var blit_tex: texture_2d<f32>;
@group(0) @binding(1) var blit_smp: sampler;

@vertex
fn vs_blit(@builtin(vertex_index) i: u32) -> BlitOut {
    var o: BlitOut;
    let x = f32(i & 1u) * 2.0;
    let y = f32(i >> 1u) * 2.0;
    o.clip = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}

@fragment
fn fs_blit(in: BlitOut) -> @location(0) vec4<f32> {
    return textureSample(blit_tex, blit_smp, in.uv);
}
