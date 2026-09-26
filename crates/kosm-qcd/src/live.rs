//! Live mode: run the phyz-qft SU(3) Monte Carlo inside the viewer.
//!
//! The sampler generates configurations, and each one is smoothed with stout
//! steps. Every step's fields are published; the viewer shows either only the
//! finished ones (morphing from one configuration to the next while playing
//! through Euclidean time) or each step, to watch the UV noise flow away into
//! the instanton lumps underneath. Smoothing is destructive, so the chain
//! carries on from the rough configuration, not the smoothed copy.
//!
//! Work runs in small units under a per-frame time budget: in a browser there
//! are no threads, and a whole 12⁴ smearing step would stall a frame for
//! hundreds of milliseconds. Heatbath work goes by half-sweep (one direction,
//! one checkerboard parity); smearing and measurement go by chunks of sites.

use phyz_qft::su3::{FieldStrength, Su3, Su3Lattice};
use web_time::{Duration, Instant};

use crate::field::FieldFile;

#[derive(Clone, Copy, PartialEq)]
pub struct LiveParams {
    /// Lattice extent in all four directions.
    pub size: usize,
    pub beta: f64,
    /// Compound updates (1 heatbath + `n_or` overrelaxation) before the first
    /// measurement, and between measurements.
    pub therm: usize,
    pub sep: usize,
    pub n_or: usize,
    /// Stout steps per configuration, each of size `rho`.
    pub smear: usize,
    pub rho: f64,
    pub seed: u64,
}

impl Default for LiveParams {
    fn default() -> Self {
        Self {
            size: 10,
            beta: 6.0,
            therm: 15,
            sep: 4,
            n_or: 2,
            smear: 14,
            rho: 0.1,
            seed: 1,
        }
    }
}

enum Phase {
    Thermalize {
        left: usize,
    },
    Evolve {
        left: usize,
    },
    /// One stout step in progress: `next[d]` holds the new links for
    /// direction `d`, complete for `d < mu` and partly filled for `mu`.
    Smooth {
        lat: Box<Su3Lattice>,
        step: usize,
        mu: usize,
        next: [Vec<Su3>; 4],
    },
    /// Field strength after `step` stout steps, filled up to its length.
    Measure {
        lat: Box<Su3Lattice>,
        step: usize,
        f: Vec<[Su3; 6]>,
    },
}

pub struct LiveSim {
    pub params: LiveParams,
    lat: Su3Lattice,
    phase: Phase,
    /// Half-sweep index within the current compound update.
    half: usize,
    pub plaquette: f64,
    /// Configurations fully smoothed so far.
    pub configs: usize,
    /// Stout step of the latest published snapshot.
    pub flow_step: usize,
    /// Wall time of the last measured work unit, for the status line.
    pub last_unit: Duration,
}

/// Sites per work unit for smearing and measurement.
const CHUNK: usize = 256;

impl LiveSim {
    pub fn new(params: LiveParams) -> Self {
        let l = params.size;
        let lat = Su3Lattice::hot([l, l, l, l], params.beta, params.seed);
        Self {
            params,
            plaquette: lat.average_plaquette(),
            lat,
            phase: Phase::Thermalize { left: params.therm },
            half: 0,
            configs: 0,
            flow_step: 0,
            last_unit: Duration::ZERO,
        }
    }

    pub fn status(&self) -> String {
        match &self.phase {
            Phase::Thermalize { left } => format!(
                "thermalizing: {} of {} updates",
                self.params.therm - left,
                self.params.therm
            ),
            Phase::Evolve { left } => format!(
                "next configuration: {} of {} updates",
                self.params.sep - left,
                self.params.sep
            ),
            Phase::Smooth { step, .. } | Phase::Measure { step, .. } => {
                format!(
                    "smoothing: stout step {} of {}",
                    step + 1,
                    self.params.smear
                )
            }
        }
    }

    /// Progress through the current phase, 0..1.
    pub fn progress(&self) -> f32 {
        let n = self.lat.n_sites() as f32;
        match &self.phase {
            Phase::Thermalize { left } => 1.0 - *left as f32 / self.params.therm.max(1) as f32,
            Phase::Evolve { left } => 1.0 - *left as f32 / self.params.sep.max(1) as f32,
            Phase::Smooth { step, mu, next, .. } => {
                (*step as f32 + (*mu as f32 + next[*mu].len() as f32 / n) / 4.0)
                    / self.params.smear as f32
            }
            Phase::Measure { step, .. } => (*step + 1) as f32 / self.params.smear as f32,
        }
    }

    /// Run work units until `budget` is spent. Returns a new field snapshot
    /// when a smoothing step's measurement completes, flagged `true` when it
    /// is the configuration's final, fully smoothed one.
    pub fn step(&mut self, budget: Duration) -> Option<(FieldFile, bool)> {
        let start = Instant::now();
        while start.elapsed() < budget {
            let t = Instant::now();
            let out = self.unit();
            self.last_unit = t.elapsed();
            if out.is_some() {
                return out;
            }
        }
        None
    }

    fn unit(&mut self) -> Option<(FieldFile, bool)> {
        let n = self.lat.n_sites();
        let phase = std::mem::replace(&mut self.phase, Phase::Evolve { left: 0 });
        let (next, out) = match phase {
            Phase::Thermalize { left } => (
                self.update_unit(left, |left| Phase::Thermalize { left }),
                None,
            ),
            Phase::Evolve { left } => (self.update_unit(left, |left| Phase::Evolve { left }), None),
            Phase::Smooth {
                mut lat,
                step,
                mut mu,
                mut next,
            } => {
                let buf = &mut next[mu];
                let end = (buf.len() + CHUNK).min(n);
                for site in buf.len()..end {
                    buf.push(lat.stout_link(site, mu, self.params.rho));
                }
                if end == n {
                    mu += 1;
                }
                if mu == 4 {
                    lat.links = next;
                    (
                        Phase::Measure {
                            lat,
                            step,
                            f: Vec::with_capacity(n),
                        },
                        None,
                    )
                } else {
                    (
                        Phase::Smooth {
                            lat,
                            step,
                            mu,
                            next,
                        },
                        None,
                    )
                }
            }
            Phase::Measure { lat, step, mut f } => {
                let end = (f.len() + CHUNK).min(n);
                for site in f.len()..end {
                    f.push(lat.field_strength_at(site));
                }
                if end < n {
                    (Phase::Measure { lat, step, f }, None)
                } else {
                    let snap = snapshot(self.lat.dims, self.params.beta, &FieldStrength { f });
                    self.flow_step = step + 1;
                    let last = step + 1 >= self.params.smear;
                    let next = if !last {
                        Phase::Smooth {
                            lat,
                            step: step + 1,
                            mu: 0,
                            next: Default::default(),
                        }
                    } else {
                        self.configs += 1;
                        Phase::Evolve {
                            left: self.params.sep,
                        }
                    };
                    (next, Some((snap, last)))
                }
            }
        };
        self.phase = next;
        out
    }

    /// One half-sweep of a thermalize/evolve phase with `left` compound
    /// updates to go; hands over to smoothing when they run out.
    fn update_unit(&mut self, left: usize, make: impl Fn(usize) -> Phase) -> Phase {
        self.half_sweep();
        if self.half != 0 {
            return make(left);
        }
        self.plaquette = self.lat.average_plaquette();
        if left > 1 {
            return make(left - 1);
        }
        Phase::Smooth {
            lat: Box::new(self.lat.clone()),
            step: 0,
            mu: 0,
            next: Default::default(),
        }
    }

    fn half_sweep(&mut self) {
        // compound update: 8 heatbath half-sweeps, then 8 per overrelaxation
        let (mu, parity) = ((self.half % 8) / 2, self.half % 2);
        if self.half < 8 {
            self.lat.half_sweep_heatbath(mu, parity);
        } else {
            self.lat.half_sweep_overrelax(mu, parity);
        }
        self.half = (self.half + 1) % (8 * (1 + self.params.n_or));
    }

    /// An empty field of the right shape, shown until the first snapshot.
    pub fn placeholder(&self) -> FieldFile {
        let n = self.lat.n_sites();
        FieldFile {
            dims: self.lat.dims,
            a_fm: lattice_spacing_fm(self.params.beta),
            names: vec!["topo".into(), "action".into()],
            data: vec![vec![0.0; n], vec![0.0; n]],
            quarks: vec![],
        }
    }
}

fn snapshot(dims: [usize; 4], beta: f64, fs: &FieldStrength) -> FieldFile {
    let to32 = |v: Vec<f64>| v.into_iter().map(|x| x as f32).collect();
    FieldFile {
        dims,
        a_fm: lattice_spacing_fm(beta),
        names: vec!["topo".into(), "action".into()],
        data: vec![
            to32(fs.topological_charge_density()),
            to32(fs.action_density()),
        ],
        quarks: vec![],
    }
}

/// Wilson-action lattice spacing from the Necco-Sommer r₀ fit, r₀ = 0.5 fm.
pub fn lattice_spacing_fm(beta: f64) -> f64 {
    let x = beta - 6.0;
    0.5 * (-1.6804 - 1.7331 * x + 0.7849 * x * x - 0.4428 * x * x * x).exp()
}
