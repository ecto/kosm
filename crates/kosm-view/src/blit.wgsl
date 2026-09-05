// The whole viewport: one oversized triangle carrying the scene's image.
// Vertex 0 is (0,0), 1 is (2,0), 2 is (0,2) in uv, which clips to the unit
// square — no vertex buffer, no index buffer, no geometry of our own.

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: Out;
    out.uv = uv;
    // uv (0,0) is the top left, so y flips on the way to clip space.
    out.pos = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSample(image, image_sampler, in.uv).rgb, 1.0);
}
