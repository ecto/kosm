// The film, on the device: exposure, the lens's `cos⁴`, bloom, ACES, sRGB.
//
// `kosm_render::post::Post` is this file in Rust, and the tracer applies it to
// its own resolved film. **That is the whole point of it being a pass rather
// than four lines at the end of the shading.** The settle blend fades the
// raster frame into the traced one in code space, so every operation between
// the radiance and the byte has to be the same operation on both tiers, or a
// standing player watches the picture change its contrast as it converges.
//
//   c = hdr · exposure · cos⁴(off axis)
//     + strength · blur(max(0, c − threshold))
//   out = sRGB(ACES(c))
//
// The bloom runs at a quarter of the frame's resolution, which is where the
// Rust runs it too: a bloom is a low frequency by construction and blurring at
// full size costs sixteen times as much for the same picture.

struct Knobs {
    // x exposure, y the vignette's amount, z tan(fov/2) across, w up
    lens: vec4<f32>,
    // x the bloom threshold, y its strength, z its σ in quarter-res texels,
    // w unused
    bloom: vec4<f32>,
    // x, y the source size in pixels; z, w the blur's step in texels
    size: vec4<f32>,
};

@group(0) @binding(0) var<uniform> k: Knobs;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
@group(0) @binding(3) var smp: sampler;

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

// `kosm_render::post::vignette`, character for character: the ray at screen
// `(sx, sy)` is `forward + right·u + up·v` over an orthonormal basis, so
// `cos θ = 1/|ray|` and `cos⁴θ` is one over the square of `1 + u² + v²`.
fn vignette(sx: f32, sy: f32) -> f32 {
    if k.lens.y <= 0.0 {
        return 1.0;
    }
    let u = sx * k.lens.z;
    let v = sy * k.lens.w;
    let q = 1.0 + u * u + v * v;
    return 1.0 + (1.0 / (q * q) - 1.0) * clamp(k.lens.y, 0.0, 1.0);
}

/// The exposed, vignetted radiance at a full-resolution pixel.
fn exposed(px: vec2<i32>) -> vec3<f32> {
    let c = textureLoad(src, px, 0).rgb;
    let sx = (f32(px.x) + 0.5) / k.size.x * 2.0 - 1.0;
    let sy = 1.0 - (f32(px.y) + 0.5) / k.size.y * 2.0;
    return c * k.lens.x * vignette(sx, sy);
}

// ---- 1. what is over the threshold, at a quarter of the size ---------------
//
// A hard threshold on luminance, scaling the colour: a soft knee would be
// prettier and would also be a second formula to hold in step with the Rust.
@fragment
fn fs_bright(in: Out) -> @location(0) vec4<f32> {
    let base = vec2<i32>(i32(in.clip.x) * 4, i32(in.clip.y) * 4);
    var acc = vec3<f32>(0.0);
    var count = 0.0;
    for (var dy = 0; dy < 4; dy = dy + 1) {
        let y = base.y + dy;
        if y >= i32(k.size.y) { continue; }
        for (var dx = 0; dx < 4; dx = dx + 1) {
            let x = base.x + dx;
            if x >= i32(k.size.x) { continue; }
            let c = exposed(vec2<i32>(x, y));
            let l = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
            let over = max(l - k.bloom.x, 0.0);
            var s = 0.0;
            if l > 1e-6 { s = over / l; }
            acc = acc + c * s;
            count = count + 1.0;
        }
    }
    return vec4<f32>(acc / max(count, 1.0), 1.0);
}

// ---- 2. a nine-tap separable Gaussian, twice ------------------------------
@fragment
fn fs_blur(in: Out) -> @location(0) vec4<f32> {
    let px = vec2<i32>(i32(in.clip.x), i32(in.clip.y));
    let step = vec2<i32>(i32(k.size.z), i32(k.size.w));
    let s = max(k.bloom.z, 1e-3);
    var acc = vec3<f32>(0.0);
    var weight = 0.0;
    for (var t = -4; t <= 4; t = t + 1) {
        let w = exp(-0.5 * f32(t * t) / (s * s));
        let q = clamp(px + step * t, vec2<i32>(0, 0), vec2<i32>(i32(k.size.x) - 1, i32(k.size.y) - 1));
        acc = acc + textureLoad(bloom_tex, q, 0).rgb * w;
        weight = weight + w;
    }
    return vec4<f32>(acc / max(weight, 1e-6), 1.0);
}

// ---- 3. the film ----------------------------------------------------------

// `kosm_render::cpu::film::tonemap_aces` — the Narkowicz fit, unchanged.
fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb1(x: f32) -> f32 {
    if x <= 0.0031308 {
        return 12.92 * x;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

@fragment
fn fs_resolve(in: Out) -> @location(0) vec4<f32> {
    var c = exposed(vec2<i32>(i32(in.clip.x), i32(in.clip.y)));
    if k.bloom.y > 0.0 {
        // bilinear back up from the quarter-res buffer, which is the same
        // upscale `post::add_bloom` writes out by hand
        c = c + textureSampleLevel(bloom_tex, smp, in.uv, 0.0).rgb * k.bloom.y;
    }
    let t = tonemap_aces(c);
    return vec4<f32>(
        linear_to_srgb1(t.r),
        linear_to_srgb1(t.g),
        linear_to_srgb1(t.b),
        1.0,
    );
}
