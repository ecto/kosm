//! The sun's shadow map: an orthographic frustum along the sun, over the
//! whole playable volume.
//!
//! A directional light has no position, so the frustum is not a perspective
//! one and its extent is a *choice*: it has to cover everything that can cast
//! into the frame, and every metre of it beyond that is texels spent on
//! nothing. The level's own playable volume — the cove square from the
//! seabed's floor to the top of the cliff — is the honest box, and it is what
//! [`Frustum::over`] is given.
//!
//! One map, 2048², for the whole level. No cascades: the cove is forty metres
//! across and a single 2048² map over it is a two-centimetre texel, which is
//! finer than the sand's own tessellation. A level that is a kilometre across
//! wants cascades and this is where they would go.

/// A world → sun-clip transform, and what it was fitted to.
#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    /// World → clip, column-major, wgpu's `z ∈ [0, 1]`.
    pub view_proj: [f32; 16],
    /// The world-space extent of one texel, metres — what the slope-scaled
    /// bias is measured in.
    pub texel_m: f32,
    /// The depth range the frustum spans, metres.
    pub depth_m: f32,
}

impl Frustum {
    /// The frustum that covers an axis-aligned box, looking along `-sun`.
    ///
    /// `sun` points **toward** the sun, so the light travels along `-sun` and
    /// that is the frustum's forward. The box's eight corners are projected
    /// into the light's own basis and the extent is taken there, which is
    /// what makes the fit tight for a low sun — a frustum sized from the
    /// box's world extent instead would be `√3` too big at every elevation.
    pub fn over(sun: [f64; 3], lo: [f64; 3], hi: [f64; 3], resolution: u32) -> Self {
        let f = unit([-sun[0], -sun[1], -sun[2]]);
        // A basis that does not degenerate at a sun straight overhead.
        let hint = if f[2].abs() > 0.95 { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 1.0] };
        let r = unit(cross(f, hint));
        let u = cross(r, f);

        let (mut min, mut max) = ([f64::MAX; 3], [f64::MIN; 3]);
        for k in 0..8 {
            let c = [
                if k & 1 == 0 { lo[0] } else { hi[0] },
                if k & 2 == 0 { lo[1] } else { hi[1] },
                if k & 4 == 0 { lo[2] } else { hi[2] },
            ];
            let p = [dot(c, r), dot(c, u), dot(c, f)];
            for a in 0..3 {
                min[a] = min[a].min(p[a]);
                max[a] = max[a].max(p[a]);
            }
        }
        // A hand's breadth of margin so a caster exactly on the boundary is
        // still in the map, and so the PCF kernel never reaches off the edge.
        let pad = 0.5;
        for a in 0..3 {
            min[a] -= pad;
            max[a] += pad;
        }
        let (hw, hh) = (0.5 * (max[0] - min[0]), 0.5 * (max[1] - min[1]));
        let (cx, cy) = (0.5 * (max[0] + min[0]), 0.5 * (max[1] + min[1]));
        let (near, far) = (min[2], max[2]);

        // view: rows r, u, f — the light looks *along* +f in its own frame,
        // so unlike a camera there is no negation here and the depth grows
        // away from the sun.
        let eye = [cx * r[0] + cy * u[0], cx * r[1] + cy * u[1], cx * r[2] + cy * u[2]];
        let view = [
            [r[0], u[0], f[0], 0.0],
            [r[1], u[1], f[1], 0.0],
            [r[2], u[2], f[2], 0.0],
            [-dot(eye, r), -dot(eye, u), -near, 1.0],
        ];
        let depth = (far - near).max(1e-6);
        let proj = [
            [1.0 / hw.max(1e-6), 0.0, 0.0, 0.0],
            [0.0, 1.0 / hh.max(1e-6), 0.0, 0.0],
            [0.0, 0.0, 1.0 / depth, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut view_proj = [0.0f32; 16];
        for c in 0..4 {
            for rr in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += proj[k][rr] * view[c][k];
                }
                view_proj[c * 4 + rr] = s as f32;
            }
        }
        Self {
            view_proj,
            texel_m: (2.0 * hw.max(hh) / resolution.max(1) as f64) as f32,
            depth_m: depth as f32,
        }
    }
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-12 { [v[0] / n, v[1] / n, v[2] / n] } else { [0.0, 0.0, 1.0] }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(m: &[f32; 16], p: [f64; 3]) -> [f64; 3] {
        let mut o = [0.0f64; 4];
        for r in 0..4 {
            o[r] = (0..3).map(|c| m[c * 4 + r] as f64 * p[c]).sum::<f64>() + m[12 + r] as f64;
        }
        [o[0] / o[3], o[1] / o[3], o[2] / o[3]]
    }

    /// Everything in the box lands in the unit clip volume, and the depth
    /// grows away from the sun — a point nearer the sun has the smaller `z`.
    #[test]
    fn the_frustum_covers_the_box_and_orders_the_depth() {
        let sun = unit([-0.35, -0.45, 0.82]);
        let (lo, hi) = ([-20.0, -25.0, -3.0], [20.0, 20.0, 12.0]);
        let fr = Frustum::over(sun, lo, hi, 2048);
        for k in 0..8 {
            let c = [
                if k & 1 == 0 { lo[0] } else { hi[0] },
                if k & 2 == 0 { lo[1] } else { hi[1] },
                if k & 4 == 0 { lo[2] } else { hi[2] },
            ];
            let q = apply(&fr.view_proj, c);
            assert!(q[0].abs() <= 1.0 && q[1].abs() <= 1.0, "corner {c:?} → {q:?} is off the map");
            assert!((0.0..=1.0).contains(&q[2]), "corner {c:?} → depth {}", q[2]);
        }
        // a point stepped toward the sun is nearer the light
        let p = [0.0, 0.0, 0.0];
        let toward = [sun[0], sun[1], sun[2]];
        assert!(
            apply(&fr.view_proj, toward)[2] < apply(&fr.view_proj, p)[2],
            "the depth does not grow away from the sun"
        );
    }

    /// The texel is the map's own resolution over the fitted extent, which is
    /// what the slope-scaled bias is stated in.
    #[test]
    fn the_texel_is_the_extent_over_the_resolution() {
        let fr = Frustum::over([0.0, 0.0, 1.0], [-20.0, -20.0, 0.0], [20.0, 20.0, 10.0], 2048);
        // 40 m plus two half-metre pads, over 2048
        assert!((fr.texel_m - 41.0 / 2048.0).abs() < 1e-4, "texel {}", fr.texel_m);
    }
}
