// The shadow pass: the same vertex and instance buffers as the scene, the
// same swell for the sea's lattice (which never reaches here — the sea does
// not cast), and no fragment stage at all. Depth is the whole output.

struct Shadow {
    sun_view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> s: Shadow;

@vertex
fn vs_shadow(
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) m0: vec4<f32>,
    @location(3) m1: vec4<f32>,
    @location(4) m2: vec4<f32>,
    @location(5) m3: vec4<f32>,
    @location(6) ids: vec4<u32>,
    @location(7) glow: vec4<f32>,
) -> @builtin(position) vec4<f32> {
    let m = mat4x4<f32>(m0, m1, m2, m3);
    let world = (m * vec4<f32>(pos, 1.0)).xyz;
    // An instance the caller marked as casting nothing is collapsed to a
    // degenerate point behind the frustum rather than drawn — one branch, and
    // no second instance buffer to keep in step with the first.
    if ids.z == 0u {
        return vec4<f32>(0.0, 0.0, -2.0, 1.0);
    }
    return s.sun_view_proj * vec4<f32>(world, 1.0);
}
