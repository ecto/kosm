// Ambient occlusion, from the depth prepass, at half resolution.
//
// **Why the tier needs it at all.** The indirect half of every pixel is nine
// spherical harmonics read off a lattice half a metre across. That is a
// beautiful description of the *low* frequencies of the light and it has
// nothing whatever to say about the centimetre under a rock, the crease where
// the cliff meets the sand, or the contact between a hero's foot and the
// beach. The path tracer resolves all three exactly, by tracing the
// hemisphere; the parity number on shaded stone was five per cent low and
// that is where most of it lived. So the geometry the probes cannot see is
// recovered in screen space and multiplied into the probe term — and into
// nothing else, because the sun already has a shadow map.
//
// **Why half resolution.** Occlusion is a low frequency by construction: it
// is an integral over a hemisphere, and the thing it varies with is the
// geometry a metre away. A quarter of the pixels, bilaterally blurred back
// up, is indistinguishable and is the difference between a pass that costs
// two milliseconds and one that costs half.
//
// The cosine an occluder has to clear above the receiver's own tangent plane
// before it counts. Six degrees; see `fs_ao` for what it is protecting against.
const ANGLE_BIAS: f32 = 0.1;

// The estimator is the plain hemisphere kind — sample points in a ball of
// `radius_m` around the pixel, project each back to the screen, and count the
// ones the depth buffer says are in front — with the usual range check so a
// silhouette against a distant background does not read as a wall of
// occlusion.

struct Ao {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // x, y the half-resolution size; z, w the full-resolution size
    size: vec4<f32>,
    // x the radius in metres, y the depth bias in metres, z the blur's step in
    // texels along x, w along y
    knobs: vec4<f32>,
    eye: vec4<f32>,
};

@group(0) @binding(0) var<uniform> a: Ao;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var ao_in: texture_2d<f32>;

// The same spiral the shadow's PCSS uses, and for the same reason: no clumps,
// and no per-pixel rotation, so the picture does not shimmer while it stands.
fn disc16(i: u32) -> vec2<f32> {
    let f = (f32(i) + 0.5) / 16.0;
    let r = sqrt(f);
    let ang = f32(i) * 2.39996323;
    return vec2<f32>(r * cos(ang), r * sin(ang));
}

// A third coordinate for the ball, so the taps are a hemisphere and not a
// disc: the same index, spread over the unit interval by the golden ratio.
fn depth_of(i: u32) -> f32 {
    return fract(f32(i) * 0.61803398875 + 0.13) * 0.9 + 0.1;
}

/// The world point behind a full-resolution depth texel, or `w = 0` when the
/// pixel is the far plane and there is no surface there.
fn world_at(px: vec2<i32>) -> vec4<f32> {
    let d = textureLoad(depth_tex, px, 0);
    if d >= 1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let ndc = vec2<f32>(
        (f32(px.x) + 0.5) / a.size.z * 2.0 - 1.0,
        1.0 - (f32(px.y) + 0.5) / a.size.w * 2.0,
    );
    let h = a.inv_view_proj * vec4<f32>(ndc, d, 1.0);
    return vec4<f32>(h.xyz / h.w, 1.0);
}

/// The screen position of a world point, in full-resolution pixels, with `z`
/// its clip depth. `w = 0` when it is behind the eye.
fn project(p: vec3<f32>) -> vec4<f32> {
    let c = a.view_proj * vec4<f32>(p, 1.0);
    if c.w <= 1e-6 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let ndc = c.xyz / c.w;
    return vec4<f32>(
        (ndc.x * 0.5 + 0.5) * a.size.z,
        (0.5 - ndc.y * 0.5) * a.size.w,
        ndc.z,
        1.0,
    );
}

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
fn fs_ao(in: Out) -> @location(0) vec4<f32> {
    let half_px = vec2<i32>(i32(in.clip.x), i32(in.clip.y));
    let full = half_px * 2;
    let here = world_at(full);
    if here.w <= 0.0 {
        return vec4<f32>(1.0);      // the sky occludes nothing
    }
    // The normal, from the depth buffer's own gradient. Each axis takes the
    // *nearer* of the two one-sided differences, which is what keeps a
    // silhouette from bending its normal out across the background.
    let px = world_at(full + vec2<i32>(2, 0));
    let mx = world_at(full - vec2<i32>(2, 0));
    let py = world_at(full + vec2<i32>(0, 2));
    let my = world_at(full - vec2<i32>(0, 2));
    var dx = px.xyz - here.xyz;
    if mx.w > 0.0 && (px.w <= 0.0 || length(here.xyz - mx.xyz) < length(dx)) {
        dx = here.xyz - mx.xyz;
    }
    var dy = py.xyz - here.xyz;
    if my.w > 0.0 && (py.w <= 0.0 || length(here.xyz - my.xyz) < length(dy)) {
        dy = here.xyz - my.xyz;
    }
    var n = cross(dx, dy);
    let nl = length(n);
    if nl < 1e-9 {
        return vec4<f32>(1.0);
    }
    n = n / nl;
    let to_eye = normalize(a.eye.xyz - here.xyz);
    if dot(n, to_eye) < 0.0 {
        n = -n;
    }
    // a tangent frame that does not degenerate wherever the normal points
    var t = cross(n, vec3<f32>(0.0, 0.0, 1.0));
    if length(t) < 1e-4 {
        t = cross(n, vec3<f32>(1.0, 0.0, 0.0));
    }
    t = normalize(t);
    let b = cross(n, t);

    let radius = max(a.knobs.x, 1e-3);
    let bias = a.knobs.y;
    var occluded = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let d2 = disc16(i);
        let up = depth_of(i);
        // a point in the hemisphere, pulled toward the surface so the taps
        // cluster where the occlusion actually is
        let dir = t * d2.x + b * d2.y + n * up;
        let sample = here.xyz + normalize(dir) * radius * depth_of(i + 7u);
        let s = project(sample);
        if s.w <= 0.0 {
            continue;
        }
        let sp = vec2<i32>(i32(s.x), i32(s.y));
        if sp.x < 0 || sp.y < 0 || sp.x >= i32(a.size.z) || sp.y >= i32(a.size.w) {
            continue;
        }
        let seen = world_at(sp);
        if seen.w <= 0.0 {
            continue;
        }
        // Is the surface the depth buffer holds *in front of* the sample?
        // Then the sample is inside something and this direction is blocked.
        let gap = length(sample - a.eye.xyz) - length(seen.xyz - a.eye.xyz);
        if gap <= bias {
            continue;
        }
        let toward = seen.xyz - here.xyz;
        let d = length(toward);
        // **The angle bias, and it is not optional.** A blocker has to be
        // genuinely *above* this pixel's own tangent plane. Without the test a
        // flat wall occludes itself: the normal comes from a depth gradient,
        // the neighbouring texels of a surface seen at a slant differ by more
        // than the depth bias, and half the taps come back "blocked" in a
        // pattern that follows the sampling spiral — which drew a set of faint
        // diagonal stripes down the cove's cliff, on a face with nothing in
        // front of it at all.
        if dot(n, toward / max(d, 1e-6)) <= ANGLE_BIAS {
            continue;
        }
        // **The range check.** A blocker further away than the radius is not
        // this pixel's occluder — it is the cliff behind the hero — and
        // without this a silhouette against a distant background darkens
        // everything it overlaps. What it must *not* be is a weight that falls
        // off as the blocker gets *nearer*, which is the mistake that made a
        // wall's own foot lighter than the floor a metre out from it.
        occluded = occluded + clamp(radius / max(d, 1e-4), 0.0, 1.0);
    }
    return vec4<f32>(clamp(1.0 - occluded / 16.0, 0.0, 1.0));
}

// One separable, depth-aware blur pass. `knobs.zw` is the step in half-res
// texels, so the same entry point does the horizontal and the vertical.
//
// Bilateral and not plain: an occlusion blurred across a silhouette leaks the
// hero's own contact darkening onto the cliff behind it, and the leak is
// exactly at the edge the eye is looking at.
@fragment
fn fs_blur(in: Out) -> @location(0) vec4<f32> {
    let px = vec2<i32>(i32(in.clip.x), i32(in.clip.y));
    let step = vec2<i32>(i32(a.knobs.z), i32(a.knobs.w));
    let here = world_at(px * 2);
    var sum = 0.0;
    var weight = 0.0;
    for (var k = -3; k <= 3; k = k + 1) {
        let q = px + step * k;
        if q.x < 0 || q.y < 0 || q.x >= i32(a.size.x) || q.y >= i32(a.size.y) {
            continue;
        }
        var w = exp(-0.5 * f32(k * k) / 4.0);
        if here.w > 0.0 {
            let there = world_at(q * 2);
            if there.w <= 0.0 {
                continue;
            }
            // a tenth of the AO radius of depth disagreement halves the weight
            let gap = abs(length(there.xyz - a.eye.xyz) - length(here.xyz - a.eye.xyz));
            w = w * exp(-gap / max(0.15 * a.knobs.x, 1e-3));
        }
        sum = sum + textureLoad(ao_in, q, 0).r * w;
        weight = weight + w;
    }
    if weight <= 0.0 {
        return vec4<f32>(1.0);
    }
    return vec4<f32>(sum / weight);
}
