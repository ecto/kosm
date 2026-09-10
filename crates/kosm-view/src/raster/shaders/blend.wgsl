// The settle blend, on the device.
//
// `settle::present` is this in bytes; this is the same mix without the
// readback. Both textures are bound as **non-sRGB** views of already-sRGB
// bytes, so the interpolation happens in code space exactly as the CPU
// version's does — both frames have already been through the level's own film,
// which `kosm_render::post::Post` states once and both tiers apply, so a pixel
// that agrees agrees at every blend, and mixing in linear and re-encoding would
// be one more place for them to disagree.
//
// The reference is smaller than the raster (the tracer runs at the budget's
// size and the raster at the window's), and the sampler is linear with
// clamp-to-edge, which is the bilinear upscale on pixel centres `present`
// writes out by hand.

struct Knobs {
    // x is the blend, 0 for the raster alone and 1 for the reference alone.
    blend: vec4<f32>,
};

@group(0) @binding(0) var raster_tex: texture_2d<f32>;
@group(0) @binding(1) var ref_tex: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var<uniform> knobs: Knobs;

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    var o: Out;
    let x = f32(i & 1u);
    let y = f32(i >> 1u);
    o.clip = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    let a = textureSample(raster_tex, smp, in.uv).rgb;
    let b = textureSample(ref_tex, smp, in.uv).rgb;
    return vec4<f32>(mix(a, b, clamp(knobs.blend.x, 0.0, 1.0)), 1.0);
}
