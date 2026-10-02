//! Turning streamed lattice measurements into physics numbers with errors.
//!
//! Every quantity keeps its per-configuration values so errors come from a
//! binned jackknife: consecutive configurations in a Markov chain are
//! correlated, and binning absorbs that as long as bins outlast the
//! autocorrelation time.

/// ħc in MeV fm, to turn lattice units into MeV.
pub const HBARC: f64 = 197.327;

/// Literature values the page compares against.
pub mod reference {
    /// Wilson plaquette at β = 6.0, large volume.
    pub const PLAQ_BETA6: f64 = 0.5937;
    /// Phenomenological string tension √σ, MeV.
    pub const SQRT_SIGMA: f64 = 440.0;
    /// Quenched topological susceptibility χ^(1/4), MeV (Del Debbio, Giusti,
    /// Pica 2005: 191 ± 5).
    pub const CHI_QUARTER: (f64, f64) = (191.0, 5.0);
    /// Mean instanton radius in the instanton liquid model, fm (Shuryak).
    pub const INSTANTON_RHO: f64 = 0.33;
    /// Critical β of pure SU(3) for N_t = 4, 6, 8 (Boyd et al. 1996).
    pub const BETA_C: [(usize, f64); 3] = [(4, 5.6925), (6, 5.8941), (8, 6.0625)];
    /// T_c / √σ for pure SU(3) (Lucini, Teper, Wenger 2004).
    pub const TC_OVER_SQRT_SIGMA: f64 = 0.646;
}

/// Wilson-action lattice spacing from the Necco-Sommer r₀ fit, r₀ = 0.5 fm.
pub fn lattice_spacing_fm(beta: f64) -> f64 {
    let x = beta - 6.0;
    0.5 * (-1.6804 - 1.7331 * x + 0.7849 * x * x - 0.4428 * x * x * x).exp()
}

/// Mean and jackknife error of `f` over `bins` contiguous bins of `data`.
pub fn jackknife<T>(
    data: &[T],
    bins: usize,
    f: impl Fn(&[&T]) -> Option<f64>,
) -> Option<(f64, f64)> {
    let n = data.len();
    let bins = bins.min(n);
    if bins < 2 {
        return None;
    }
    let all: Vec<&T> = data.iter().collect();
    let mean = f(&all)?;
    let size = n / bins;
    let mut est = Vec::with_capacity(bins);
    for b in 0..bins {
        let (lo, hi) = (b * size, if b + 1 == bins { n } else { (b + 1) * size });
        let rest: Vec<&T> = data[..lo].iter().chain(&data[hi..]).collect();
        est.push(f(&rest)?);
    }
    let m = est.iter().sum::<f64>() / bins as f64;
    let var = est.iter().map(|e| (e - m).powi(2)).sum::<f64>() * (bins as f64 - 1.0) / bins as f64;
    Some((mean, var.sqrt()))
}

/// Static potential a·V(R) = ln[W(R, T₀) / W(R, T₀ + 1)] from mean Wilson loops
/// laid out `w[(R − 1) tmax + T − 1]`.
pub fn potential(w: &[f64], rmax: usize, tmax: usize, t0: usize) -> Vec<Option<f64>> {
    (1..=rmax)
        .map(|r| {
            let a = w[(r - 1) * tmax + t0 - 1];
            let b = w[(r - 1) * tmax + t0];
            (a > 0.0 && b > 0.0).then(|| (a / b).ln())
        })
        .collect()
}

/// Cornell fit V(R) = A − B/R + σR by least squares over R ≥ `r_min`;
/// returns (A, B, σ) in lattice units.
pub fn cornell(v: &[Option<f64>], r_min: usize) -> Option<(f64, f64, f64)> {
    let pts: Vec<(f64, f64)> = v
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.map(|v| ((i + 1) as f64, v)))
        .filter(|(r, _)| *r >= r_min as f64)
        .collect();
    if pts.len() < 3 {
        return None;
    }
    // normal equations for basis (1, −1/R, R)
    let mut m = [[0.0; 3]; 3];
    let mut y = [0.0; 3];
    for (r, v) in &pts {
        let b = [1.0, -1.0 / r, *r];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += b[i] * b[j];
            }
            y[i] += b[i] * v;
        }
    }
    solve3(m, y).map(|x| (x[0], x[1], x[2]))
}

#[allow(clippy::needless_range_loop)]
fn solve3(mut m: [[f64; 3]; 3], mut y: [f64; 3]) -> Option<[f64; 3]> {
    for c in 0..3 {
        let p = (c..3).max_by(|&a, &b| m[a][c].abs().total_cmp(&m[b][c].abs()))?;
        if m[p][c].abs() < 1e-12 {
            return None;
        }
        m.swap(c, p);
        y.swap(c, p);
        for r in 0..3 {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in 0..3 {
                    m[r][k] -= f * m[c][k];
                }
                y[r] -= f * y[c];
            }
        }
    }
    Some([y[0] / m[0][0], y[1] / m[1][1], y[2] / m[2][2]])
}

/// Instanton candidates in a 4D topological charge density: local extrema of
/// |q| over the 8 nearest neighbours, above `min_q`. Each peak gives a radius
/// from the BPST profile q(0) = 6 / (π² ρ⁴), in lattice units.
pub fn instanton_radii(q: &[f32], dims: [usize; 4], min_q: f32) -> Vec<f64> {
    let [nt, nx, ny, _] = dims;
    let idx = |c: [usize; 4]| c[0] + nt * (c[1] + nx * (c[2] + ny * c[3]));
    let mut out = Vec::new();
    for site in 0..q.len() {
        let v = q[site];
        if v.abs() < min_q {
            continue;
        }
        let c = [
            site % nt,
            (site / nt) % nx,
            (site / (nt * nx)) % ny,
            site / (nt * nx * ny),
        ];
        let peak = (0..4).all(|mu| {
            [1, dims[mu] - 1].iter().all(|&d| {
                let mut n = c;
                n[mu] = (c[mu] + d) % dims[mu];
                let w = q[idx(n)];
                w.signum() != v.signum() || w.abs() < v.abs()
            })
        });
        if peak {
            out.push((6.0 / (std::f64::consts::PI.powi(2) * v.abs() as f64)).powf(0.25));
        }
    }
    out
}

/// Every stream of measurements the engine produces, with derived results.
#[derive(Default)]
pub struct Stats {
    pub configs: usize,
    pub plaq: Vec<f64>,
    /// Topological charge per configuration.
    pub q: Vec<f64>,
    /// Instanton radii (lattice units), all configurations pooled.
    pub radii: Vec<f64>,
    /// Mean Wilson loops per configuration, `(R − 1) tmax + T − 1`.
    pub wilson: Vec<Vec<f64>>,
    pub rmax: usize,
    pub tmax: usize,
    /// Volume-averaged Polyakov loop per configuration.
    pub poly: Vec<(f64, f64)>,
}

const KEEP: usize = 4000;

impl Stats {
    pub fn push_bounded<T>(v: &mut Vec<T>, x: T) {
        if v.len() >= KEEP {
            v.remove(0);
        }
        v.push(x);
    }

    /// Mean W(R, T) and the static potential with jackknife errors.
    pub fn potential(&self, t0: usize) -> Vec<Option<(f64, f64)>> {
        if self.wilson.len() < 4 || self.tmax < t0 + 1 {
            return vec![];
        }
        (1..=self.rmax)
            .map(|r| {
                jackknife(&self.wilson, 20, |ws| {
                    let n = ws.len() as f64;
                    let mean: Vec<f64> = (0..self.rmax * self.tmax)
                        .map(|k| ws.iter().map(|w| w[k]).sum::<f64>() / n)
                        .collect();
                    potential(&mean, self.rmax, self.tmax, t0)[r - 1]
                })
            })
            .collect()
    }

    /// Cornell fit parameters (A, B, σ) with the jackknife error on σ.
    pub fn string_tension(&self, t0: usize, r_min: usize) -> Option<((f64, f64, f64), f64)> {
        if self.wilson.len() < 4 {
            return None;
        }
        let fit = |ws: &[&Vec<f64>]| {
            let n = ws.len() as f64;
            let mean: Vec<f64> = (0..self.rmax * self.tmax)
                .map(|k| ws.iter().map(|w| w[k]).sum::<f64>() / n)
                .collect();
            cornell(&potential(&mean, self.rmax, self.tmax, t0), r_min)
        };
        let all: Vec<&Vec<f64>> = self.wilson.iter().collect();
        let best = fit(&all)?;
        let (_, err) = jackknife(&self.wilson, 20, |ws| fit(ws).map(|f| f.2))?;
        Some((best, err))
    }

    /// χ^(1/4) in MeV with error, for a lattice of `volume` sites at spacing `a_fm`.
    pub fn chi_quarter(&self, volume: usize, a_fm: f64) -> Option<(f64, f64)> {
        let to_mev = |chi: f64| chi.max(0.0).powf(0.25) / a_fm * HBARC;
        jackknife(&self.q, 20, |qs| {
            let n = qs.len() as f64;
            let m = qs.iter().map(|q| **q).sum::<f64>() / n;
            let q2 = qs.iter().map(|q| **q * **q).sum::<f64>() / n;
            Some(to_mev((q2 - m * m) / volume as f64))
        })
    }
}

/// Running sums for the baryon–density correlation C(r) of the flux tube.
pub struct FluxSums {
    /// Σ over configurations of Σ_x0 W(x0) S_k(x0 + r), k = action, electric.
    pub ws: [Vec<f64>; 2],
    pub w: f64,
    pub s: [f64; 2],
    /// Junction sites summed over.
    pub samples: f64,
    pub configs: usize,
}

impl FluxSums {
    pub fn new(ns: usize) -> Self {
        Self {
            ws: [vec![0.0; ns], vec![0.0; ns]],
            w: 0.0,
            s: [0.0; 2],
            samples: 0.0,
            configs: 0,
        }
    }

    /// C_k(r) = ⟨W S_k(x0 + r)⟩ / (⟨W⟩ ⟨S_k⟩); ≈ 1 far away, < 1 in the tube.
    pub fn correlation(&self, k: usize) -> Option<Vec<f64>> {
        let (w, s) = (self.w / self.samples, self.s[k] / self.samples);
        (self.configs > 0 && w.abs() > 1e-12 && s.abs() > 1e-12).then(|| {
            self.ws[k]
                .iter()
                .map(|v| v / self.samples / (w * s))
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cornell_recovers_parameters() {
        let (a, b, s) = (0.6, 0.28, 0.045);
        let v: Vec<Option<f64>> = (1..=8)
            .map(|r| Some(a - b / r as f64 + s * r as f64))
            .collect();
        let (fa, fb, fs) = cornell(&v, 1).unwrap();
        assert!((fa - a).abs() < 1e-9 && (fb - b).abs() < 1e-9 && (fs - s).abs() < 1e-9);
    }

    #[test]
    fn jackknife_of_mean_is_standard_error() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.7).sin()).collect();
        let (m, e) = jackknife(&data, 100, |d| {
            Some(d.iter().map(|x| **x).sum::<f64>() / d.len() as f64)
        })
        .unwrap();
        let mean = data.iter().sum::<f64>() / 100.0;
        let sd = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 99.0).sqrt();
        assert!((m - mean).abs() < 1e-12);
        assert!((e - sd / 10.0).abs() < 1e-9, "{e} vs {}", sd / 10.0);
    }

    #[test]
    fn bpst_peak_gives_its_radius() {
        // a lone BPST density with ρ = 3 at the origin of a 12⁴ lattice
        let dims = [12; 4];
        let rho: f64 = 3.0;
        let q: Vec<f32> = (0..12usize.pow(4))
            .map(|s| {
                let c = [s % 12, (s / 12) % 12, (s / 144) % 12, s / 1728];
                let r2: f64 = c.iter().map(|&x| (x.min(12 - x) as f64).powi(2)).sum();
                (6.0 / std::f64::consts::PI.powi(2) * rho.powi(4) / (r2 + rho * rho).powi(4)) as f32
            })
            .collect();
        let radii = instanton_radii(&q, dims, 1e-4);
        assert_eq!(radii.len(), 1);
        assert!((radii[0] - rho).abs() < 1e-3, "{radii:?}");
    }
}
