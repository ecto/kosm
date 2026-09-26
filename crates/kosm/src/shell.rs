//! Thin shells of revolution, heard: the modes of a wine glass.
//!
//! `audio.rs` knows bars and spheres. A goblet's voice is neither: it is the
//! wall of a thin shell of revolution bending into `cos nθ` lobes, and the
//! note you hear when you tap a glass is the n = 2 mode — the rim going oval.
//!
//! **Method.** The meridian (the profile curve, base to rim) is cut into
//! straight conical frustums. On each one the displacement is
//! `u(s) cos nθ` along the meridian, `v(s) sin nθ` around it and `w(s) cos nθ`
//! along the normal; `u` and `v` are linear in `s`, `w` is a cubic Hermite so
//! the bending energy is conforming. Strains and curvatures are Love's for a
//! cone, the twist in the rotation form. Nodes carry `(U_r, U_z, V, ψ)` —
//! displacement in the global (r, z) frame, circumferential displacement, and
//! the meridional slope of `w` — so the frustums meet at kinks without
//! tearing. Stiffness and mass are assembled per circumferential wavenumber
//! `n` and handed to a dense generalized eigensolver (Cholesky, then cyclic
//! Jacobi): a goblet is a few hundred degrees of freedom.
//!
//! Checked against Rayleigh's inextensional ring formula on a free-free
//! cylinder and against rigid-body modes, which must cost no energy.
//!
//! Units: SI (metres, kg, Pa, seconds).

use crate::audio::Material;
use std::f64::consts::PI;

/// Degrees of freedom per node: `U_r, U_z, V, ψ`.
const DOF: usize = 4;

/// A meridian, base to rim, as `(r, z)` points in metres. Consecutive points
/// are one conical frustum each.
#[derive(Clone, Debug)]
pub struct Profile {
    pub points: Vec<[f64; 2]>,
    /// Clamp every degree of freedom at the first point (the stem).
    pub clamp_base: bool,
}

impl Profile {
    /// A goblet's bowl: radius `base_r` where it meets the stem, opening to
    /// `rim_r` at `height`, round-bottomed like an ellipse's quarter.
    /// `segments` frustums, graded finer at the rim where the n = 2 mode lives.
    pub fn goblet(rim_r: f64, height: f64, base_r: f64, segments: usize) -> Self {
        let points = (0..=segments)
            .map(|i| {
                let t = i as f64 / segments as f64;
                let a = t * PI / 2.0; // the parameter runs round a quarter ellipse
                let z = height * (1.0 - a.cos());
                let r = base_r + (rim_r - base_r) * a.sin();
                [r, z]
            })
            .collect();
        Self { points, clamp_base: true }
    }

    /// A straight cylinder of radius `r` and length `len`, free at both ends.
    pub fn cylinder(r: f64, len: f64, segments: usize) -> Self {
        let points = (0..=segments).map(|i| [r, len * i as f64 / segments as f64]).collect();
        Self { points, clamp_base: false }
    }

    pub fn nodes(&self) -> usize {
        self.points.len()
    }

    /// Wall area (both n-independent), for radiation.
    pub fn area(&self) -> f64 {
        self.points
            .windows(2)
            .map(|p| {
                let l = ((p[1][0] - p[0][0]).powi(2) + (p[1][1] - p[0][1]).powi(2)).sqrt();
                PI * (p[0][0] + p[1][0]) * l
            })
            .sum()
    }
}

/// One mode of the shell.
#[derive(Clone, Debug)]
pub struct ShellMode {
    /// Circumferential wavenumber: 2 is the oval, the note.
    pub n: u32,
    /// Meridional order within that `n`, 0 first.
    pub m: u32,
    pub hz: f64,
    /// Amplitude decay rate (1/s) from the material's loss factor: π f η.
    pub decay: f64,
    /// Mass-normalized nodal shape, `(U_r, U_z, V, ψ)` per profile point.
    pub shape: Vec<[f64; 4]>,
}

impl ShellMode {
    /// Radial displacement at a profile point (a tap on the rim pushes along r).
    pub fn radial(&self, node: usize) -> f64 {
        self.shape[node][0]
    }
}

/// Rayleigh's inextensional ring: the n-th mode of a free cylinder's wall,
/// constant along its length. The check the solver has to pass.
pub fn ring_hz(radius: f64, thickness: f64, mat: Material, n: u32) -> f64 {
    let n = n as f64;
    let d_over_rho = mat.e / (12.0 * mat.rho * (1.0 - mat.nu * mat.nu));
    thickness / (radius * radius) * d_over_rho.sqrt() * n * (n * n - 1.0) / (n * n + 1.0).sqrt() / (2.0 * PI)
}

/// A liquid standing in the bowl to `level` metres above the base.
///
/// It loads the wall as added mass: the potential flow inside a circle of
/// radius r whose wall moves as `w cos nθ` pushes back with a pressure
/// `ρ r / n` per unit of normal acceleration, so each wetted strip of the
/// meridian carries `ρ r / n` of extra mass on its normal motion. At the free
/// surface the pressure is zero, so a strip at depth δ carries only
/// `tanh(n δ / r)` of that: the lobes' field reaches r/n into the liquid.
/// Local, one-way (the liquid's own sloshing is ignored), and it is what
/// makes the note fall as the glass fills: slowly at first, where the oval
/// mode barely moves, then fast near the rim.
#[derive(Clone, Copy, Debug)]
pub struct Liquid {
    pub density: f64,
    pub level: f64,
}

// four-point Gauss on [0, 1]
const G: [(f64, f64); 4] = [
    (0.069_431_844_202_973_71, 0.173_927_422_568_726_9),
    (0.330_009_478_207_571_9, 0.326_072_577_431_273_1),
    (0.669_990_521_792_428_1, 0.326_072_577_431_273_1),
    (0.930_568_155_797_026_3, 0.173_927_422_568_726_9),
];

/// The normal-displacement row over an element's eight dofs at `xi`.
fn normal_row(len: f64, s: f64, c: f64, xi: f64) -> [f64; 8] {
    let hm = [1.0 - 3.0 * xi * xi + 2.0 * xi.powi(3), len * (xi - 2.0 * xi * xi + xi.powi(3)), 3.0 * xi * xi - 2.0 * xi.powi(3), len * (-xi * xi + xi.powi(3))];
    let mut w = [0.0; 8];
    for node in 0..2 {
        let o = node * DOF;
        w[o] = c * hm[2 * node];
        w[o + 1] = -s * hm[2 * node];
        w[o + 3] = hm[2 * node + 1];
    }
    w
}

/// Stiffness and mass for wavenumber `n`, dense, all nodes.
fn assemble(profile: &Profile, h: f64, mat: Material, n: u32) -> (Vec<f64>, Vec<f64>, usize) {
    assemble_with(profile, h, mat, n, None)
}

fn assemble_with(profile: &Profile, h: f64, mat: Material, n: u32, liquid: Option<Liquid>) -> (Vec<f64>, Vec<f64>, usize) {
    let dim = profile.nodes() * DOF;
    let mut k = vec![0.0; dim * dim];
    let mut m = vec![0.0; dim * dim];
    let nf = n as f64;
    let a = mat.e * h / (1.0 - mat.nu * mat.nu);
    let d = mat.e * h.powi(3) / (12.0 * (1.0 - mat.nu * mat.nu));
    let nu = mat.nu;

    for (e, p) in profile.points.windows(2).enumerate() {
        let (dr, dz) = (p[1][0] - p[0][0], p[1][1] - p[0][1]);
        let len = (dr * dr + dz * dz).sqrt();
        let (s, c) = (dr / len, dz / len);
        let mut ke = [[0.0; 8]; 8];
        let mut me = [[0.0; 8]; 8];

        for &(xi, wg) in &G {
            let r = p[0][0] + dr * xi;
            let (n0, n1) = (1.0 - xi, xi);
            let (dn0, dn1) = (-1.0 / len, 1.0 / len);
            let hm = [1.0 - 3.0 * xi * xi + 2.0 * xi.powi(3), len * (xi - 2.0 * xi * xi + xi.powi(3)), 3.0 * xi * xi - 2.0 * xi.powi(3), len * (-xi * xi + xi.powi(3))];
            let hd = [(-6.0 * xi + 6.0 * xi * xi) / len, 1.0 - 4.0 * xi + 3.0 * xi * xi, (6.0 * xi - 6.0 * xi * xi) / len, -2.0 * xi + 3.0 * xi * xi];
            let hdd = [(-6.0 + 12.0 * xi) / (len * len), (-4.0 + 6.0 * xi) / len, (6.0 - 12.0 * xi) / (len * len), (-2.0 + 6.0 * xi) / len];

            // rows over the element's eight dofs [Ur0 Uz0 V0 ψ0 Ur1 Uz1 V1 ψ1]
            let mut u = [0.0; 8];
            let mut du = [0.0; 8];
            let mut v = [0.0; 8];
            let mut dv = [0.0; 8];
            let mut w = [0.0; 8];
            let mut dw = [0.0; 8];
            let mut ddw = [0.0; 8];
            for (node, (nn, dnn)) in [(n0, dn0), (n1, dn1)].into_iter().enumerate() {
                let o = node * DOF;
                u[o] = s * nn;
                u[o + 1] = c * nn;
                du[o] = s * dnn;
                du[o + 1] = c * dnn;
                v[o + 2] = nn;
                dv[o + 2] = dnn;
                let (hv, hs) = (2 * node, 2 * node + 1); // Hermite value / slope index
                w[o] = c * hm[hv];
                w[o + 1] = -s * hm[hv];
                w[o + 3] = hm[hs];
                dw[o] = c * hd[hv];
                dw[o + 1] = -s * hd[hv];
                dw[o + 3] = hd[hs];
                ddw[o] = c * hdd[hv];
                ddw[o + 1] = -s * hdd[hv];
                ddw[o + 3] = hdd[hs];
            }

            let mut es = [0.0; 8];
            let mut et = [0.0; 8];
            let mut g = [0.0; 8];
            let mut ks = [0.0; 8];
            let mut kt = [0.0; 8];
            let mut tw = [0.0; 8];
            for i in 0..8 {
                es[i] = du[i];
                et[i] = (nf * v[i] + s * u[i] + c * w[i]) / r;
                g[i] = dv[i] - s * v[i] / r - nf * u[i] / r;
                ks[i] = -ddw[i];
                let b = (c * v[i] + nf * w[i]) / r;
                let db = (c * dv[i] + nf * dw[i]) / r - b * s / r;
                kt[i] = nf * b / r - s * dw[i] / r;
                tw[i] = nf * dw[i] / r + db - b * s / r;
            }

            let f = wg * len * r;
            for i in 0..8 {
                for j in 0..8 {
                    let memb = es[i] * es[j] + et[i] * et[j] + nu * (es[i] * et[j] + et[i] * es[j]) + 0.5 * (1.0 - nu) * g[i] * g[j];
                    let bend = ks[i] * ks[j] + kt[i] * kt[j] + nu * (ks[i] * kt[j] + kt[i] * ks[j]) + 0.5 * (1.0 - nu) * tw[i] * tw[j];
                    ke[i][j] += f * (a * memb + d * bend);
                    me[i][j] += f * mat.rho * h * (u[i] * u[j] + v[i] * v[j] + w[i] * w[j]);
                }
            }
        }

        // the wetted part of this frustum, integrated exactly up to the surface
        if let Some(liq) = liquid.filter(|_| n > 0) {
            let wet = if dz.abs() < 1e-15 {
                if p[0][1] < liq.level { 1.0 } else { 0.0 }
            } else {
                ((liq.level - p[0][1]) / dz).clamp(0.0, 1.0)
            };
            for &(t, wg) in G.iter().filter(|_| wet > 0.0) {
                let xi = t * wet;
                let r = p[0][0] + dr * xi;
                let depth = liq.level - (p[0][1] + dz * xi);
                let w = normal_row(len, s, c, xi);
                let f = wg * wet * len * r * liq.density * r / nf * (nf * depth / r).tanh();
                for i in 0..8 {
                    for j in 0..8 {
                        me[i][j] += f * w[i] * w[j];
                    }
                }
            }
        }

        let base = e * DOF;
        for i in 0..8 {
            for j in 0..8 {
                k[(base + i) * dim + base + j] += ke[i][j];
                m[(base + i) * dim + base + j] += me[i][j];
            }
        }
    }
    (k, m, dim)
}

/// The `count` lowest modes of wavenumber `n`, lowest first.
pub fn modes(profile: &Profile, thickness: f64, mat: Material, n: u32, count: usize) -> Vec<ShellMode> {
    modes_filled(profile, thickness, mat, n, count, None)
}

/// [`modes`] with a liquid in the bowl.
pub fn modes_filled(profile: &Profile, thickness: f64, mat: Material, n: u32, count: usize, liquid: Option<Liquid>) -> Vec<ShellMode> {
    let (k, m, dim) = assemble_with(profile, thickness, mat, n, liquid);
    let free: Vec<usize> = (0..dim).filter(|&i| !(profile.clamp_base && i < DOF)).collect();
    let f = free.len();
    let pick = |a: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; f * f];
        for (i, &gi) in free.iter().enumerate() {
            for (j, &gj) in free.iter().enumerate() {
                out[i * f + j] = a[gi * dim + gj];
            }
        }
        out
    };
    let (kf, mut mf) = (pick(&k), pick(&m));

    // ψ carries no mass (no rotary inertia); a whisker keeps M positive definite
    // without moving any audible mode.
    let scale = (0..f).map(|i| mf[i * f + i]).fold(0.0, f64::max);
    for i in 0..f {
        mf[i * f + i] += scale * 1e-9;
    }

    let (lam, vecs) = generalized_eigen(&kf, &mf, f);
    let mut order: Vec<usize> = (0..f).collect();
    order.sort_by(|&a, &b| lam[a].total_cmp(&lam[b]));

    order
        .into_iter()
        .filter(|&i| lam[i].is_finite() && lam[i] > 0.0)
        .map(|i| (i, lam[i]))
        .filter(|&(_, l)| l.sqrt() / (2.0 * PI) > 1.0) // rigid-body modes sit at zero
        .take(count)
        .enumerate()
        .map(|(mi, (i, l))| {
            let hz = l.sqrt() / (2.0 * PI);
            let mut shape = vec![[0.0; 4]; profile.nodes()];
            for (row, &gi) in free.iter().enumerate() {
                shape[gi / DOF][gi % DOF] = vecs[row * f + i];
            }
            ShellMode { n, m: mi as u32, hz, decay: PI * hz * mat.loss, shape }
        })
        .collect()
}

/// d hz / d level for one mode of a filled bowl: the first-order path.
///
/// The eigenproblem's own adjoint. For a mass-normalized φ, dλ = −λ φᵀ dM φ.
/// Raising the surface deepens every wetted strip, and the strip's added mass
/// `ρ r/n · tanh(n δ/r)` grows by `ρ sech²(n δ/r)` per metre, so dM/dL is
/// that integrated over the wet wall (the new strip at the waterline adds
/// nothing: tanh(0) = 0). No second solve, no finite step.
pub fn d_hz_d_level(profile: &Profile, mode: &ShellMode, liquid: Liquid) -> f64 {
    if mode.n == 0 {
        return 0.0;
    }
    let nf = mode.n as f64;
    let mut dm = 0.0;
    for (e, p) in profile.points.windows(2).enumerate() {
        let (dr, dz) = (p[1][0] - p[0][0], p[1][1] - p[0][1]);
        let len = (dr * dr + dz * dz).sqrt();
        let wet = if dz.abs() < 1e-15 {
            if p[0][1] < liquid.level { 1.0 } else { 0.0 }
        } else {
            ((liquid.level - p[0][1]) / dz).clamp(0.0, 1.0)
        };
        if wet == 0.0 {
            continue;
        }
        let q: Vec<f64> = (0..2).flat_map(|k| mode.shape[e + k]).collect();
        for &(t, wg) in &G {
            let xi = t * wet;
            let r = p[0][0] + dr * xi;
            let depth = liquid.level - (p[0][1] + dz * xi);
            let row = normal_row(len, dr / len, dz / len, xi);
            let w: f64 = row.iter().zip(&q).map(|(a, b)| a * b).sum();
            let sech = 1.0 / (nf * depth / r).cosh();
            dm += wg * wet * len * r * liquid.density * sech * sech * w * w;
        }
    }
    let lambda = (2.0 * PI * mode.hz).powi(2);
    -lambda * dm / (8.0 * PI * PI * mode.hz)
}

/// French's empirical fill law for a wine glass (Am. J. Phys. 51, 688, 1983):
/// `(f₀/f)² = 1 + α (ρ_l R / 5 ρ_g t) (level/H)⁴`, α ≈ 1.25. The published
/// check the added-mass model is held to.
pub fn french_ratio(rim_r: f64, height: f64, thickness: f64, rho_glass: f64, rho_liquid: f64, level: f64) -> f64 {
    let x = 1.0 + 1.25 * rho_liquid * rim_r / (5.0 * rho_glass * thickness) * (level / height).powi(4);
    1.0 / x.sqrt()
}

/// `K x = λ M x` for symmetric K and positive-definite M, dense row-major.
/// Eigenvectors come back M-normalized, column `i` for eigenvalue `i`.
fn generalized_eigen(k: &[f64], m: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    // M = L Lᵀ
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = m[i * n + j];
            for p in 0..j {
                sum -= l[i * n + p] * l[j * n + p];
            }
            if i == j {
                l[i * n + i] = sum.max(1e-300).sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    // C = L⁻¹ K L⁻ᵀ: solve L Y = K, then L Cᵀ = Yᵀ
    let forward = |b: &mut [f64]| {
        // b is n×n row-major; overwrite with L⁻¹ b, column by column
        for col in 0..n {
            for i in 0..n {
                let mut sum = b[i * n + col];
                for p in 0..i {
                    sum -= l[i * n + p] * b[p * n + col];
                }
                b[i * n + col] = sum / l[i * n + i];
            }
        }
    };
    let mut y = k.to_vec();
    forward(&mut y);
    let mut yt = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            yt[i * n + j] = y[j * n + i];
        }
    }
    forward(&mut yt);
    let mut a = yt; // symmetric up to round-off
    for i in 0..n {
        for j in 0..i {
            let s = 0.5 * (a[i * n + j] + a[j * n + i]);
            a[i * n + j] = s;
            a[j * n + i] = s;
        }
    }

    let (lam, z) = jacobi(a, n);

    // x = L⁻ᵀ z
    let mut x = z;
    for col in 0..n {
        for i in (0..n).rev() {
            let mut sum = x[i * n + col];
            for p in i + 1..n {
                sum -= l[p * n + i] * x[p * n + col];
            }
            x[i * n + col] = sum / l[i * n + i];
        }
    }
    (lam, x)
}

/// Cyclic Jacobi on a dense symmetric matrix. Eigenvalues, and orthonormal
/// eigenvectors as columns.
fn jacobi(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let norm: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    for _sweep in 0..60 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i * n + j].powi(2)).sum::<f64>().sqrt();
        if off <= 1e-14 * norm {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for kk in 0..n {
                    let akp = a[kk * n + p];
                    let akq = a[kk * n + q];
                    a[kk * n + p] = c * akp - s * akq;
                    a[kk * n + q] = s * akp + c * akq;
                }
                for kk in 0..n {
                    let apk = a[p * n + kk];
                    let aqk = a[q * n + kk];
                    a[p * n + kk] = c * apk - s * aqk;
                    a[q * n + kk] = s * apk + c * aqk;
                }
                for kk in 0..n {
                    let vkp = v[kk * n + p];
                    let vkq = v[kk * n + q];
                    v[kk * n + p] = c * vkp - s * vkq;
                    v[kk * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

/// A tap: a half-sine radial force of `impulse` (N·s) over `contact_s` at
/// profile point `node`, θ = 0. The pressure a listener hears, mono, as a sum
/// of the modes' damped sinusoids, peak-normalized to 0.8.
pub fn strike(modes: &[ShellMode], node: usize, impulse: f64, contact_s: f64, duration: f64, sr: f64, radiation: impl Fn(f64) -> f64) -> Vec<f32> {
    let len = (duration * sr) as usize;
    let mut out = vec![0.0f64; len];
    for mode in modes {
        if !(20.0..18_000.0).contains(&mode.hz) {
            continue;
        }
        let omega = 2.0 * PI * mode.hz;
        // a half-sine pulse's spectrum, normalized to 1 at DC
        let x = omega * contact_s / PI;
        let pulse = if (x - 1.0).abs() < 1e-9 { PI / 4.0 } else { ((PI * x / 2.0).cos() / (1.0 - x * x)).abs() };
        // modal velocity amplitude after an impulse: P φ(node) for a mass-normalized φ;
        // radiated pressure goes as surface acceleration, ω v, through √σ
        let amp = impulse * pulse * mode.radial(node).abs() * omega * radiation(mode.hz).sqrt();
        let (dw, decay) = (omega / sr, mode.decay / sr);
        for (i, o) in out.iter_mut().enumerate() {
            let t = i as f64;
            *o += amp * (-decay * t).exp() * (dw * t).sin();
        }
    }
    let peak = out.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let gain = if peak > 0.0 { 0.8 / peak } else { 0.0 };
    // a millisecond fade-in so the attack does not click
    let fade = (1e-3 * sr) as usize;
    out.iter()
        .enumerate()
        .map(|(i, &s)| (s * gain * (i.min(fade) as f64 / fade.max(1) as f64)) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::GLASS;

    #[test]
    fn a_free_cylinder_rings_like_rayleighs_ring() {
        let (r, h) = (0.04, 1.2e-3);
        let cyl = Profile::cylinder(r, 0.12, 40);
        for n in 2..=4 {
            let got = modes(&cyl, h, GLASS, n, 1)[0].hz;
            let want = ring_hz(r, h, GLASS, n);
            let err = (got - want).abs() / want;
            assert!(err < 0.02, "n={n}: {got:.1} Hz vs Rayleigh {want:.1} Hz ({:.2}%)", err * 100.0);
        }
    }

    #[test]
    fn rigid_motion_costs_nothing() {
        let bowl = Profile { clamp_base: false, ..Profile::goblet(0.04, 0.09, 0.005, 24) };
        let nodes = bowl.nodes();
        // n = 1 lateral translation: U_r = 1, V = −1; n = 0 axial: U_z = 1
        for (n, q) in [(1u32, [1.0, 0.0, -1.0, 0.0]), (0u32, [0.0, 1.0, 0.0, 0.0])] {
            let (k, _, dim) = assemble(&bowl, 1.2e-3, GLASS, n);
            let x: Vec<f64> = (0..nodes).flat_map(|_| q).collect();
            let energy: f64 = (0..dim).map(|i| x[i] * (0..dim).map(|j| k[i * dim + j] * x[j]).sum::<f64>()).sum();
            let scale: f64 = (0..dim).map(|i| k[i * dim + i]).sum();
            assert!(energy.abs() < 1e-9 * scale, "n={n}: rigid energy {energy:e} vs trace {scale:e}");
        }
    }

    fn filled(level: f64) -> ShellMode {
        let bowl = Profile::goblet(0.04, 0.09, 0.005, 30);
        let wine = Liquid { density: 992.0, level };
        modes_filled(&bowl, 1.2e-3, GLASS, 2, 1, Some(wine)).remove(0)
    }

    #[test]
    fn the_note_falls_as_the_glass_fills() {
        let empty = filled(0.0).hz;
        let hz: Vec<f64> = (0..=8).map(|i| filled(0.01 * i as f64).hz).collect();
        assert!(hz.windows(2).all(|w| w[1] <= w[0] + 1e-9), "monotone: {hz:?}");
        eprintln!("fill law: {:?}", hz.iter().map(|h| h / empty).collect::<Vec<_>>());
        // the bottom half moves the oval least; each centimetre near the rim more
        assert!(hz[4] / empty > 0.9, "half full: {:.3}", hz[4] / empty);
        assert!(hz[7] - hz[8] > hz[3] - hz[4], "steeper near the rim: {hz:?}");
        for (i, h) in hz.iter().enumerate() {
            let want = french_ratio(0.04, 0.09, 1.2e-3, GLASS.rho, 992.0, 0.01 * i as f64);
            assert!((h / empty - want).abs() < 0.08, "{} cm: {:.3} vs French {want:.3}", i, h / empty);
        }
    }

    #[test]
    fn the_adjoint_agrees_with_central_differences() {
        let bowl = Profile::goblet(0.04, 0.09, 0.005, 30);
        for level in [0.045, 0.062, 0.078] {
            let wine = Liquid { density: 992.0, level };
            let adj = d_hz_d_level(&bowl, &filled(level), wine);
            let h = 1e-5;
            let fd = (filled(level + h).hz - filled(level - h).hz) / (2.0 * h);
            assert!((adj - fd).abs() / fd.abs() < 0.01, "level {level}: adjoint {adj:.2} vs fd {fd:.2} Hz/m");
        }
    }

    #[test]
    fn a_goblet_sings_in_the_audible_band_and_converges() {
        let coarse = modes(&Profile::goblet(0.04, 0.09, 0.005, 30), 1.2e-3, GLASS, 2, 1)[0].hz;
        let fine = modes(&Profile::goblet(0.04, 0.09, 0.005, 60), 1.2e-3, GLASS, 2, 1)[0].hz;
        assert!((300.0..1500.0).contains(&fine), "the (2,0) note is {fine:.0} Hz");
        assert!((coarse - fine).abs() / fine < 0.01, "mesh: {coarse:.1} → {fine:.1} Hz");
    }
}
