// The ride tier: plain instanced meshes, lambert against one sun plus a sky
// ambient. No refraction, no caustics — a recording of a rollout only has to
// be legible.

struct Uniforms {
    view_proj: mat4x4<f32>,
    sun: vec4<f32>,   // xyz direction to the sun, w sun strength
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) nrm: vec3<f32>,
    @location(1) colour: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) m0: vec4<f32>,
    @location(3) m1: vec4<f32>,
    @location(4) m2: vec4<f32>,
    @location(5) m3: vec4<f32>,
    @location(6) colour: vec4<f32>,
) -> VsOut {
    let m = mat4x4<f32>(m0, m1, m2, m3);
    let world = m * vec4<f32>(pos, 1.0);
    var o: VsOut;
    o.clip = u.view_proj * world;
    // the model matrix is rigid, so the rotation part transforms normals
    o.nrm = normalize((mat3x3<f32>(m0.xyz, m1.xyz, m2.xyz) * nrm));
    o.colour = colour.rgb;
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // STLs are inconsistently wound, so light both faces the same way
    let n = normalize(in.nrm);
    let d = abs(dot(n, normalize(u.sun.xyz)));
    // sky above, ground bounce below, so the underside of a ramp is not black
    let sky = 0.5 + 0.5 * n.z;
    let amb = mix(vec3<f32>(0.18, 0.17, 0.16), vec3<f32>(0.34, 0.38, 0.46), sky);
    let lit = in.colour * (amb + u.sun.w * d);
    return vec4<f32>(pow(lit, vec3<f32>(1.0 / 2.2)), 1.0);
}

// ---- blit: the offscreen colour target onto the egui pass ----

struct BlitOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var blit_tex: texture_2d<f32>;
@group(0) @binding(1) var blit_smp: sampler;

@vertex
fn vs_blit(@builtin(vertex_index) i: u32) -> BlitOut {
    var o: BlitOut;
    let x = f32(i & 1u);
    let y = f32(i >> 1u);
    o.clip = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}

@fragment
fn fs_blit(in: BlitOut) -> @location(0) vec4<f32> {
    return textureSample(blit_tex, blit_smp, in.uv);
}
