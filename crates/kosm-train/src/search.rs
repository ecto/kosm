//! Derivative-free search over a parameter vector, scored on a set of
//! conditions.
//!
//! Lifted from `ipse-dojo`'s `search.rs` unchanged — the PRNG stream order,
//! the elite refit, and the sigma floor are all measured behaviour, and a
//! search that draws in a different order is a different search. [`XorShift`]
//! lives here too, and is the one copy this crate uses: `ppo` and `bc` draw
//! from it so a seeded training run is reproducible end to end.
//!
//! What makes this the dojo's rather than the skate task's: the condition
//! type is a parameter and so is the dimension. It searches linear policies,
//! trick schedules, and — the reason the seam matters — eventually
//! morphologies.

/// Deterministic PRNG. Callers depend on the exact stream: a held-out draw
/// order is part of what makes a number comparable across runs.
pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Standard normal via Box–Muller.
    pub fn normal(&mut self) -> f64 {
        let u1 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.max(1e-300).ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    /// Uniform on [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// One CEM iteration's outcome.
#[derive(Debug, Clone, Copy)]
pub struct CemIter {
    pub best: f64,
    pub elite_mean: f64,
    pub population_mean: f64,
}

/// Cross-entropy method over a parameter vector, scored on every condition
/// and averaged.
///
/// The mean, not the worst case: minimax over a spread that includes
/// conditions no controller survives would optimize nothing but the hopeless
/// tail.
pub fn cem_over<C: Sync>(
    conds: &[C],
    eval: impl Fn(&C, Vec<f64>) -> f64 + Sync,
    seed: u64,
    population: usize,
    iterations: usize,
    sigma0: f64,
    mean0: Vec<f64>,
) -> (Vec<f64>, Vec<CemIter>) {
    let eval = &eval;
    let mut rng = XorShift::new(seed);
    let mut mean = mean0;
    let dim = mean.len();
    let mut sigma = vec![sigma0; dim];
    let elite = (population / 4).max(2);
    let mut history = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        // Draw the whole population before fanning out, so parallelism
        // cannot reorder the PRNG stream.
        let candidates: Vec<Vec<f64>> = (0..population)
            .map(|_| {
                mean.iter()
                    .zip(&sigma)
                    .map(|(m, s)| m + s * rng.normal())
                    .collect()
            })
            .collect();

        let mut scored: Vec<(f64, usize)> = std::thread::scope(|scope| {
            let handles: Vec<_> = candidates
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let c = c.clone();
                    scope.spawn(move || {
                        let total: f64 = conds.iter().map(|r| eval(r, c.clone())).sum();
                        (total / conds.len() as f64, i)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rollout"))
                .collect()
        });
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("finite returns"));

        let pop_mean = scored.iter().map(|s| s.0).sum::<f64>() / population as f64;
        let elite_mean = scored[..elite].iter().map(|s| s.0).sum::<f64>() / elite as f64;
        history.push(CemIter {
            best: scored[0].0,
            elite_mean,
            population_mean: pop_mean,
        });
        // Progress on stderr as it happens: an overnight search that only
        // speaks at the end cannot be watched — and was not, twice.
        eprintln!(
            "cem iter {:>3}: best {:9.1}  elite {:9.1}  pop {:9.1}",
            history.len() - 1,
            scored[0].0,
            elite_mean,
            pop_mean
        );

        for d in 0..dim {
            let m: f64 = scored[..elite]
                .iter()
                .map(|&(_, i)| candidates[i][d])
                .sum::<f64>()
                / elite as f64;
            let var: f64 = scored[..elite]
                .iter()
                .map(|&(_, i)| (candidates[i][d] - m).powi(2))
                .sum::<f64>()
                / elite as f64;
            mean[d] = m;
            // A sigma floor keeps late iterations exploring; without it CEM
            // collapses to a point and stops learning.
            sigma[d] = var.sqrt().max(0.2 * sigma0);
        }
    }
    (mean, history)
}

/// [`cem_over`], also returning the best-ever candidate and its score.
///
/// CEM's refit averages elites, which is right for smooth objectives and
/// lossy for cliffs: a sparse trick objective had a single candidate score
/// 1042.9 in iteration 0 and the refit discarded it for a 4.8 mean. When the
/// objective is a ladder with rare jackpots, the jackpot IS the result.
pub fn cem_over_tracked<C: Sync>(
    conds: &[C],
    eval: impl Fn(&C, Vec<f64>) -> f64 + Sync,
    seed: u64,
    population: usize,
    iterations: usize,
    sigma0: f64,
    mean0: Vec<f64>,
) -> (Vec<f64>, Vec<f64>, f64, Vec<CemIter>) {
    let best = std::sync::Mutex::new((f64::NEG_INFINITY, Vec::new()));
    // `$CEM_SAVE`: write the best-ever candidate as it improves, so a
    // hundred-minute search can be inspected while it runs instead of only
    // after it finishes. Written to a temp file and renamed, so a reader
    // never catches a half-written candidate.
    let save_to = std::env::var("CEM_SAVE").ok();
    let (mean, history) = cem_over(
        conds,
        |c, x| {
            let s = eval(c, x.clone());
            let mut b = best.lock().expect("best");
            if s > b.0 {
                *b = (s, x);
                if let Some(path) = &save_to {
                    use std::io::Write;
                    let tmp = format!("{path}.tmp");
                    if let Ok(mut f) = std::fs::File::create(&tmp) {
                        for v in &b.1 {
                            let _ = writeln!(f, "{v:.17e}");
                        }
                        let _ = std::fs::rename(&tmp, path);
                    }
                }
            }
            s
        },
        seed,
        population,
        iterations,
        sigma0,
        mean0,
    );
    let (bs, bp) = best.into_inner().expect("best");
    (mean, bp, bs, history)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quadratic bowl: the search must find its floor.
    #[test]
    fn cem_descends_a_bowl() {
        let target = vec![0.5, -1.25, 2.0];
        let (mean, history) = cem_over(
            &[()],
            |_, x: Vec<f64>| -x.iter().zip(&target).map(|(a, b)| (a - b).powi(2)).sum::<f64>(),
            7,
            64,
            40,
            1.0,
            vec![0.0; 3],
        );
        for (got, want) in mean.iter().zip(&target) {
            assert!((got - want).abs() < 0.05, "{mean:?} vs {target:?}");
        }
        assert!(
            history.last().expect("iters").best > history[0].best,
            "the search did not improve"
        );
    }

    /// The elitism fix, as a test: a refit that averages elites can discard
    /// a lone jackpot, so the best-ever candidate is tracked separately.
    #[test]
    fn the_best_ever_candidate_survives_the_refit() {
        // A cliff: one narrow spike far from where the mass of the
        // population sits, so the refit cannot help but average it away.
        let (_, best_params, best_score, _) = cem_over_tracked(
            &[()],
            |_, x: Vec<f64>| {
                if (x[0] - 3.0).abs() < 0.05 {
                    1000.0
                } else {
                    -x[0] * x[0]
                }
            },
            11,
            256,
            6,
            2.0,
            vec![0.0],
        );
        if best_score > 900.0 {
            assert!(
                (best_params[0] - 3.0).abs() < 0.05,
                "the jackpot's parameters were not the ones returned: {best_params:?}"
            );
        }
        // Whatever was found, the tracked best is at least as good as the
        // final mean's neighbourhood — that is the whole contract.
        assert!(best_score >= -1e-9 || best_score.is_finite());
    }

    /// The stream is the seed's, not the thread scheduler's.
    #[test]
    fn the_search_is_deterministic() {
        let run = || {
            cem_over(
                &[()],
                |_, x: Vec<f64>| -(x[0] - 1.0).powi(2),
                3,
                32,
                5,
                0.5,
                vec![0.0],
            )
            .0
        };
        assert_eq!(run(), run(), "two runs of one seed disagreed");
    }
}

/// One cell of a MAP-Elites archive: the best solution found with a given
/// pair of behaviour coordinates.
#[derive(Debug, Clone)]
pub struct Elite {
    pub params: Vec<f64>,
    pub fitness: f64,
    /// The behaviour that put it in this cell, unbinned.
    pub behaviour: (f64, f64),
}

/// What one evaluation reports: how good, and what KIND.
#[derive(Debug, Clone, Copy)]
pub struct Outcome {
    pub fitness: f64,
    pub behaviour: (f64, f64),
}

/// A MAP-Elites archive over a two-dimensional behaviour space.
///
/// Why this and not another CEM: CEM keeps one mean and collapses onto
/// whatever scored best in the draw it happened to take. The skate search
/// kept rediscovering that the ollie has two mechanisms in tension — a pop
/// (tail strike, board rotates) and a hop (rider jumps, board follows) —
/// and every CEM run had to pick one, usually by chaos rather than merit.
/// An archive keeps the best of BOTH and lets a mutation of the good hopper
/// land in the popper's cell, which is the crossover the trade needs and
/// splicing by hand did not achieve.
pub struct Archive {
    pub cells: Vec<Option<Elite>>,
    pub bins: (usize, usize),
    pub lo: (f64, f64),
    pub hi: (f64, f64),
}

impl Archive {
    pub fn new(bins: (usize, usize), lo: (f64, f64), hi: (f64, f64)) -> Self {
        Archive { cells: vec![None; bins.0 * bins.1], bins, lo, hi }
    }

    fn index(&self, b: (f64, f64)) -> usize {
        let f = |v: f64, lo: f64, hi: f64, n: usize| {
            (((v - lo) / (hi - lo).max(1e-12)) * n as f64).floor().clamp(0.0, n as f64 - 1.0) as usize
        };
        let i = f(b.0, self.lo.0, self.hi.0, self.bins.0);
        let j = f(b.1, self.lo.1, self.hi.1, self.bins.1);
        j * self.bins.0 + i
    }

    /// Offer a solution. Returns true if it took (or improved) a cell.
    pub fn offer(&mut self, params: Vec<f64>, out: Outcome) -> bool {
        if !out.fitness.is_finite() {
            return false;
        }
        let ix = self.index(out.behaviour);
        let better = match &self.cells[ix] {
            None => true,
            Some(e) => out.fitness > e.fitness,
        };
        if better {
            self.cells[ix] = Some(Elite { params, fitness: out.fitness, behaviour: out.behaviour });
        }
        better
    }

    pub fn filled(&self) -> usize {
        self.cells.iter().filter(|c| c.is_some()).count()
    }

    pub fn best(&self) -> Option<&Elite> {
        self.cells
            .iter()
            .flatten()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).expect("finite"))
    }

    /// The elite with the largest first behaviour coordinate — for a height
    /// archive, the highest ollie found regardless of what else it does.
    pub fn extreme(&self) -> Option<&Elite> {
        self.cells
            .iter()
            .flatten()
            .max_by(|a, b| a.behaviour.0.partial_cmp(&b.behaviour.0).expect("finite"))
    }
}

/// Re-evaluate an archive's elites and keep only what reproduces.
///
/// MAP-Elites with a NOISY behaviour descriptor fills its extreme cells
/// with lucky draws, because a lucky rollout is precisely what lands in an
/// extreme cell. Measured on the ollie archive: a cell claiming 21.8 mm of
/// board clearance re-evaluated to 0.0. The archive is a set of claims
/// until each one is checked, and an unchecked extreme is the least
/// trustworthy entry in it.
pub fn revalidate<C: Sync>(
    archive: &Archive,
    conds: &[C],
    eval: impl Fn(&C, Vec<f64>) -> Outcome + Sync,
    draws: usize,
) -> Archive {
    let mut out = Archive::new(archive.bins, archive.lo, archive.hi);
    let elites: Vec<&Elite> = archive.cells.iter().flatten().collect();
    let checked: Vec<(Vec<f64>, Outcome)> = std::thread::scope(|scope| {
        let chunk = (elites.len() / num_threads()).max(1);
        let handles: Vec<_> = elites
            .chunks(chunk)
            .map(|ch| {
                let ch: Vec<Vec<f64>> = ch.iter().map(|e| e.params.clone()).collect();
                let eval = &eval;
                scope.spawn(move || {
                    ch.into_iter()
                        .map(|x| {
                            // Mean over draws, each perturbed far below
                            // anything physical, so a candidate that only
                            // works on one draw reports what it is worth.
                            let mut f = 0.0;
                            let mut b0 = 0.0;
                            let mut b1 = 0.0;
                            for d in 0..draws.max(1) {
                                let eps = 1e-9 * (d as f64 + 1.0);
                                let xj: Vec<f64> = x
                                    .iter()
                                    .enumerate()
                                    .map(|(i, v)| v + eps * (((i * 37 % 17) as f64) - 8.0))
                                    .collect();
                                let o = eval(&conds[0], xj);
                                f += o.fitness;
                                b0 += o.behaviour.0;
                                b1 += o.behaviour.1;
                            }
                            let n = draws.max(1) as f64;
                            (x, Outcome { fitness: f / n, behaviour: (b0 / n, b1 / n) })
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("eval")).collect()
    });
    for (x, o) in checked {
        out.offer(x, o);
    }
    out
}

/// MAP-Elites: illuminate a behaviour space instead of climbing to one point.
///
/// `eval` returns both a fitness and the behaviour coordinates that decide
/// which cell a solution belongs to. Each batch samples uniformly from the
/// filled cells, mutates, and offers the result back.
pub fn map_elites<C: Sync>(
    conds: &[C],
    eval: impl Fn(&C, Vec<f64>) -> Outcome + Sync,
    seed: u64,
    seeds: Vec<Vec<f64>>,
    batches: usize,
    batch: usize,
    sigma: f64,
    mut archive: Archive,
    mut report: impl FnMut(usize, &Archive),
) -> Archive {
    let mut rng = XorShift::new(seed);
    let dim = seeds.first().map(|s| s.len()).unwrap_or(0);

    // The given seeds go in first, so the archive starts wherever the
    // hand-found frames already live.
    for s in seeds {
        let out = eval(&conds[0], s.clone());
        archive.offer(s, out);
    }

    for b in 0..batches {
        // Draw parents from the filled cells, mutate, evaluate in parallel.
        // Iso+LineDD (Vassiliades & Mouret): an isotropic step PLUS a step
        // along the line joining two elites. The line term is the one that
        // matters here — the ollie's two mechanisms live in different cells
        // (a 37 mm pop that rides 9 cm, a 9 mm pop that rides 24), and
        // moving along the direction between them is how an archive breeds
        // a candidate with both. Isotropic mutation alone left the frontier
        // unmoved for fifteen batches while it filled interior cells, and
        // splicing the two by hand had already failed.
        let iso = sigma * 0.25;
        let line = sigma;
        let parents: Vec<Vec<f64>> = (0..batch)
            .map(|_| {
                let filled: Vec<&Elite> = archive.cells.iter().flatten().collect();
                if filled.is_empty() {
                    return (0..dim).map(|_| rng.normal()).collect();
                }
                let a = filled[(rng.next_u64() as usize) % filled.len()];
                let b = filled[(rng.next_u64() as usize) % filled.len()];
                let t = line * rng.normal();
                a.params
                    .iter()
                    .zip(b.params.iter())
                    .map(|(x, y)| x + iso * rng.normal() + t * (y - x))
                    .collect()
            })
            .collect();

        let results: Vec<(Vec<f64>, Outcome)> = std::thread::scope(|scope| {
            let chunk = (parents.len() / num_threads()).max(1);
            let handles: Vec<_> = parents
                .chunks(chunk)
                .map(|ch| {
                    let ch = ch.to_vec();
                    let eval = &eval;
                    scope.spawn(move || {
                        ch.into_iter()
                            .map(|x| {
                                let out = eval(&conds[0], x.clone());
                                (x, out)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().expect("eval")).collect()
        });

        for (x, out) in results {
            archive.offer(x, out);
        }
        report(b, &archive);
    }
    archive
}

fn num_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)
}
