//! The spectral SH probe volume, as the shader reads it.
//!
//! [`ProbeVolume`] *is* `kosm::light::probes::ProbeVolume` — the baker's own
//! type, re-exported so there is one volume and not two. What lives here is
//! the raster tier's half of the contract:
//!
//! - [`sh_cosine`], the nine basis functions already scaled by the clamped
//!   cosine lobe's `Â_l`. `shaders/scene.wgsl` has this function too, and
//!   [`the_cosine_convolution_is_the_integral`](self) holds both to a numeric
//!   `∫ L(ω) max(0, ω·n) dω`. If those nine numbers are wrong then every
//!   pixel is wrong by the same factor and nothing else would notice.
//! - [`uniform`] and [`from_radiance`], the synthetic volumes the tier's own
//!   tests are written against, so a GPU test does not have to run a bake.
//!
//! ```
//! use kosm_view::raster::probes::{uniform, BANDS};
//! // A white furnace: every probe sees irradiance π, whatever the normal.
//! let v = uniform([0.0; 3], 1.0, [2, 2, 2], 1.0);
//! let e = v.sample(0.0, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0]);
//! for b in 0..BANDS {
//!     assert!((e[b] as f64 - std::f64::consts::PI).abs() < 1e-3);
//! }
//! ```

pub use kosm::light::probes::{BANDS, ProbeVolume, SH, VolumeSpec, sh_basis};

/// The band centres, nanometres.
pub const BANDS_NM: [f64; BANDS] = kosm::material::BANDS_NM;

/// The clamped-cosine lobe's SH coefficients, per degree `l`: `Â₀ = π`,
/// `Â₁ = 2π/3`, `Â₂ = π/4`.
///
/// These three numbers are the whole lighting model. A Lambert surface's
/// irradiance is `∫ L(ω) max(0, ω·n) dω`, the clamped cosine's own expansion
/// has only `l = 0, 1, 2` above a per cent, and so the irradiance is the
/// baked radiance's coefficients scaled by these and evaluated at `n`.
pub const COSINE_LOBE: [f32; 3] = [
    core::f32::consts::PI,
    2.0 * core::f32::consts::PI / 3.0,
    core::f32::consts::PI / 4.0,
];

/// The nine basis functions at `n`, already multiplied by [`COSINE_LOBE`].
///
/// The kernel a radiance expansion is dotted with to get irradiance. The
/// shader's `sh_cosine` is this function, line for line.
pub fn sh_cosine(n: [f64; 3]) -> [f32; SH] {
    let y = sh_basis(unit(n));
    let mut out = [0.0f32; SH];
    out[0] = (y[0] * COSINE_LOBE[0] as f64) as f32;
    for i in 1..4 {
        out[i] = (y[i] * COSINE_LOBE[1] as f64) as f32;
    }
    for i in 4..9 {
        out[i] = (y[i] * COSINE_LOBE[2] as f64) as f32;
    }
    out
}

/// A white furnace: a uniform radiance `l` from every direction and nothing
/// else. Every probe reads irradiance `π · l` at every normal, which is the
/// closed form a probe volume has to reproduce before it is worth baking one.
pub fn uniform(origin: [f64; 3], spacing: f64, dims: [u32; 3], l: f32) -> ProbeVolume {
    let spec = VolumeSpec { origin, spacing, dims, scene_per_metre: 1.0 };
    let mut v = ProbeVolume::zeros(&spec, vec![[0.0, 0.0, 1.0]]);
    // A constant radiance projects onto `Y₀₀` alone, at `l / Y₀₀`.
    let c0 = (l as f64 / sh_basis([0.0, 0.0, 1.0])[0]) as f32;
    for probe in 0..v.count() {
        for b in 0..BANDS {
            v.data[probe * SH * BANDS + b] = c0;
        }
    }
    v
}

/// A volume whose every probe is the SH projection of one radiance function,
/// integrated over a Fibonacci sphere.
///
/// The bake does this with paths; this does it with a closure, which is the
/// same projection with the transport taken out — enough to give a GPU test
/// a light field with structure in it without running a bake.
pub fn from_radiance(
    origin: [f64; 3],
    spacing: f64,
    dims: [u32; 3],
    rays: usize,
    radiance: impl Fn([f64; 3], [f64; 3]) -> [f32; BANDS],
) -> ProbeVolume {
    let spec = VolumeSpec { origin, spacing, dims, scene_per_metre: 1.0 };
    let mut v = ProbeVolume::zeros(&spec, vec![[0.0, 0.0, 1.0]]);
    let w = (4.0 * std::f64::consts::PI / rays as f64) as f32;
    for iz in 0..dims[2] {
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                let p = v.point(ix, iy, iz);
                let at = v.index(0, ix, iy, iz);
                for k in 0..rays {
                    let d = fibonacci(k, rays);
                    let l = radiance(p, d);
                    let y = sh_basis(d);
                    for c in 0..SH {
                        for b in 0..BANDS {
                            v.data[at + c * BANDS + b] += y[c] as f32 * l[b] * w;
                        }
                    }
                }
            }
        }
    }
    v
}

/// The `k`-th of `n` directions on a Fibonacci sphere: an even cover with no
/// clumping at the poles, which is what a projection integral wants.
pub fn fibonacci(k: usize, n: usize) -> [f64; 3] {
    let ga = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    let z = 1.0 - 2.0 * (k as f64 + 0.5) / n as f64;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let th = ga * k as f64;
    [r * th.cos(), r * th.sin(), z]
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-12 { [v[0] / n, v[1] / n, v[2] / n] } else { [0.0, 0.0, 1.0] }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The convolution is the integral.** A random L2 radiance expansion,
    /// convolved with the cosine lobe at a normal, must agree with a numeric
    /// `∫ L(ω) max(0, ω·n) dω` over that same expansion — both through
    /// [`sh_cosine`], which is the shader's own kernel, and through
    /// `ProbeVolume::sample`, which is the baker's.
    ///
    /// This is the one claim the whole lighting model rests on, and the only
    /// one that can be checked without a renderer.
    #[test]
    fn the_cosine_convolution_is_the_integral() {
        let mut coeffs = [[0.0f32; BANDS]; SH];
        let mut seed = 0x9E37_79B9u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        for c in 0..SH {
            for b in 0..BANDS {
                // a positive DC term, so the reconstructed radiance is mostly
                // positive and the comparison is of a real light field
                coeffs[c][b] = if c == 0 { 3.0 + next().abs() } else { 0.4 * next() };
            }
        }
        let mut v = uniform([0.0; 3], 1.0, [1, 1, 1], 0.0);
        for c in 0..SH {
            for b in 0..BANDS {
                v.data[c * BANDS + b] = coeffs[c][b];
            }
        }

        let rays = 200_000;
        let dw = 4.0 * std::f64::consts::PI / rays as f64;
        for n in [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], unit([0.3, -0.7, 0.5])] {
            // the numeric integral of the same expansion
            let mut want = [0.0f64; BANDS];
            for k in 0..rays {
                let d = fibonacci(k, rays);
                let cos = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
                if cos <= 0.0 {
                    continue;
                }
                let y = sh_basis(d);
                for b in 0..BANDS {
                    let l: f64 = (0..SH).map(|c| y[c] * coeffs[c][b] as f64).sum();
                    want[b] += l * cos * dw;
                }
            }
            // the shader's kernel
            let k = sh_cosine(n);
            let mut got = [0.0f64; BANDS];
            for c in 0..SH {
                for b in 0..BANDS {
                    got[b] += k[c] as f64 * coeffs[c][b] as f64;
                }
            }
            // and the baker's own `sample`
            let sampled = v.sample(0.0, [0.0, 0.0, 0.0], n);
            for b in 0..BANDS {
                let e = (got[b] - want[b]).abs();
                assert!(
                    e < 2e-3 * want[b].abs().max(1.0),
                    "band {b} at {n:?}: the shader's kernel gives {} against the integral {want:?} ({e:.2e})",
                    got[b]
                );
                assert!(
                    (sampled[b] as f64 - want[b]).abs() < 2e-3 * want[b].abs().max(1.0),
                    "band {b} at {n:?}: `sample` gives {} against the integral {}",
                    sampled[b],
                    want[b]
                );
            }
        }
    }

    /// A white furnace gives π per band at every probe and every normal.
    #[test]
    fn a_white_furnace_is_pi_everywhere() {
        let v = uniform([-1.0, -1.0, -1.0], 0.5, [4, 4, 4], 1.0);
        for p in [[0.0, 0.0, 0.0], [-0.9, 0.2, 0.4], [10.0, 10.0, 10.0]] {
            for n in [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], unit([1.0, 1.0, 1.0])] {
                let e = v.sample(0.0, p, n);
                for b in 0..BANDS {
                    assert!(
                        (e[b] as f64 - std::f64::consts::PI).abs() < 1e-3,
                        "the furnace read {} at {p:?}/{n:?}",
                        e[b]
                    );
                }
            }
        }
    }

    /// `sample` is continuous across a cell boundary: no seam where the
    /// trilinear weights change which eight probes they are over.
    #[test]
    fn the_sample_is_continuous_across_a_cell() {
        let mut v = uniform([0.0; 3], 1.0, [3, 1, 1], 0.0);
        let bright = (10.0 / sh_basis([0.0, 0.0, 1.0])[0]) as f32;
        for b in 0..BANDS {
            v.data[SH * BANDS + b] = bright;
        }
        let read = |x: f64| v.sample(0.0, [x, 0.0, 0.0], [0.0, 0.0, 1.0])[0] as f64;
        let h = 1e-4;
        assert!((read(1.0 - h) - read(1.0 + h)).abs() < 1e-3, "a seam at the cell boundary");
        assert!((read(1.0) - 10.0 * std::f64::consts::PI).abs() < 1e-2);
        assert!(read(0.0).abs() < 1e-4 && read(2.0).abs() < 1e-4);
        assert!((read(0.5) - 5.0 * std::f64::consts::PI).abs() < 1e-2);
    }

    /// A synthetic light field with structure: a sky that is bright above and
    /// dark below reads brighter on an up-facing normal than on a down-facing
    /// one, by the ratio the integral says.
    #[test]
    fn a_split_sky_reads_the_way_up_it_is() {
        let v = from_radiance([0.0; 3], 1.0, [2, 2, 2], 4096, |_, d| {
            if d[2] > 0.0 { [1.0; BANDS] } else { [0.0; BANDS] }
        });
        let up = v.sample(0.0, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0])[0] as f64;
        let down = v.sample(0.0, [0.5, 0.5, 0.5], [0.0, 0.0, -1.0])[0] as f64;
        // a hemisphere of unit radiance over an up-facing surface is exactly π
        assert!((up - std::f64::consts::PI).abs() < 0.05, "up read {up}");
        assert!(down < 0.1, "a surface facing the dark half read {down}");
    }
}
