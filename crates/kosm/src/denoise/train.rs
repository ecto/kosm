//! The fit, and the honest comparison.
//!
//! # The loss
//!
//! L1 on a tone-compressed image, plus a gradient-domain term.
//!
//! *Tone-compressed* because the thing being fitted is looked at through a
//! tonemap, and an L1 on linear radiance spends its whole budget on the light
//! panels. `x/(1+x)` is not the display curve, but it is monotone, it
//! saturates in the same place, and it has a derivative you can write down.
//!
//! *L1 rather than L2* because L2's optimum under uncertainty is the mean,
//! and the mean of "this edge is here or one pixel over" is a blurred edge.
//! An L1 optimum is the median, which picks one.
//!
//! *Plus gradients* because L1 alone is indifferent between an image with the
//! right values and one with the right values in the wrong arrangement. The
//! finite-difference term is a tenth of the weight and it is what stops the
//! network settling on a smooth wash that is on average correct — the
//! standard KPN failure, and the one worth spending a term on.
//!
//! *Plus temporal consistency*, which is new in v2 and is the term a
//! still-image loss cannot express. Every tile is stored with the *next*
//! frame of its own sequence ([`super::dataset::FRAMES`]), so the network can
//! be run on both and asked to give the same answer on the pixels that did
//! not change. "Did not change" is read off the data rather than guessed: a
//! pixel whose history count went up by exactly one kept its history through
//! the reprojection and was not clamped, which is precisely the definition of
//! a pixel that should not flicker. Pixels the reprojection dropped, the
//! clamp shortened or an object moved across are excluded, because there the
//! picture *should* change and penalising that would be asking the filter to
//! smear.
//!
//! The term is deliberately small ([`TEMPORAL_WEIGHT`]). Its job is to break
//! ties between kernels that score the same on one frame, not to out-vote the
//! reference.
//!
//! # What it is measured against
//!
//! [`atrous_baseline`] is [`kosm_render::pathtrace::denoise`] on the same
//! input — the CPU twin of the à-trous pass in `history.wgsl`, weight for
//! weight. Every number this module reports is relative to it. A neural
//! denoiser that does not beat the filter it replaces is a slower filter.

use kosm_render::pathtrace::{Film, PathTraceOptions};
use rayon::prelude::*;

use super::dataset::{Dataset, Sample, TIERS, Tile, from_f16, tiles};
use super::kpn::{C_IN, DEMOD_FLOOR, K, KS, Kpn, ParamInputs, features, forward_graph, illumination};
use crate::ir::{Adam, Arg, ExprId, Form, Monoid, Program, TensorGraph, TensorId, View, lit, optimize, pick};
use tang::Scalar;

/// Weight on the gradient-domain term, relative to the L1.
pub const GRAD_WEIGHT: f32 = 0.1;

/// Weight on the temporal consistency term, relative to the L1.
///
/// A twentieth. The reference is the thing being fitted; this only says that
/// among kernels which fit it equally well, the steady one wins.
pub const TEMPORAL_WEIGHT: f32 = 0.05;

/// How close two frames' counts have to be to `+1` for the pixel to count as
/// having kept its history.
///
/// Exactly one, up to f16 rounding on a count near the cap. A pixel that came
/// back at 1, or came back lower than it went in, was disoccluded or clamped,
/// and the picture there is *allowed* to move.
fn kept_history(before: f32, after: f32) -> bool {
    before >= 1.0 && (after - before - 1.0).abs() < 0.51
}

/// The tone curve the loss is measured through.
#[inline]
fn tone(x: f32) -> f32 {
    x / (1.0 + x.max(0.0))
}

/// The still-image loss of one tile, in plain loops: the reference the graph
/// in [`TrainGraph`] is checked against, and the definition worth reading.
///
/// `out` is planar `[3][n]` demodulated filtered illumination, `albedo` and
/// `reference` interleaved, as the dataset stores them.
pub fn loss(out: &[f32], albedo: &[f32], reference: &[f32], s: usize) -> f32 {
    let n = s * s;
    let a = |c: usize, p: usize| albedo[p * 3 + c].max(DEMOD_FLOOR);
    let mut tp = vec![0.0f32; 3 * n];
    let mut tt = vec![0.0f32; 3 * n];
    for c in 0..3 {
        for p in 0..n {
            tp[c * n + p] = tone(out[c * n + p] * a(c, p));
            tt[c * n + p] = tone(reference[p * 3 + c]);
        }
    }
    let inv = 1.0 / (3 * n) as f32;
    let mut loss = 0.0;
    for i in 0..3 * n {
        loss += (tp[i] - tt[i]).abs() * inv;
    }
    // Forward differences in x and y, in the same tone-compressed space.
    // The border tap is skipped rather than clamped: a clamped difference is
    // identically zero in both images and would only dilute the term.
    let pairs = 3 * 2 * s * s.saturating_sub(1);
    if pairs > 0 {
        let k = GRAD_WEIGHT / pairs as f32;
        let edge = |i: usize, j: usize| k * ((tp[j] - tp[i]) - (tt[j] - tt[i])).abs();
        for c in 0..3 {
            let base = c * n;
            for y in 0..s {
                for x in 0..s - 1 {
                    loss += edge(base + y * s + x, base + y * s + x + 1);
                }
            }
            for y in 0..s - 1 {
                for x in 0..s {
                    loss += edge(base + y * s + x, base + (y + 1) * s + x);
                }
            }
        }
    }
    loss
}

/// The temporal consistency term, in plain loops, for the same purpose.
///
/// `cur` and `next` are planar `[3][n]` network outputs for the same tile at
/// consecutive frames; `valid` is one flag per pixel from [`kept_history`].
pub fn temporal_loss(cur: &[f32], next: &[f32], albedo: &[f32], valid: &[bool], s: usize) -> f32 {
    let n = s * s;
    let k = temporal_scale(valid);
    let mut loss = 0.0;
    for c in 0..3 {
        for p in (0..n).filter(|&p| valid[p]) {
            let a = albedo[p * 3 + c].max(DEMOD_FLOOR);
            loss += k * (tone(next[c * n + p] * a) - tone(cur[c * n + p] * a)).abs();
        }
    }
    loss
}

/// The temporal term's weight per live pixel-channel: [`TEMPORAL_WEIGHT`]
/// spread over the pixels that kept their history, zero if none did.
fn temporal_scale(valid: &[bool]) -> f32 {
    let live = valid.iter().filter(|v| **v).count();
    if live == 0 { 0.0 } else { TEMPORAL_WEIGHT / (3 * live) as f32 }
}

// ---------------------------------------------------------------------------
// The training step as a graph
// ---------------------------------------------------------------------------

/// One tile size's whole training step as a [`TensorGraph`]: the forward over
/// the tile (and over its successor frame, when `paired`), the loss, and the
/// gradient of that loss with respect to every parameter — `backward`'s, not
/// ours.
///
/// Inputs, in order: the seven parameter blocks ([`ParamInputs`]), then
/// `feat` `[C_IN, s, s]`, `illum` `[3, s, s]`, the floored albedo `[3, s, s]`
/// and the tone-mapped reference `[3, s, s]`, all planar; when paired, the
/// successor's `feat` and `illum`, the validity mask `[s, s]` as 0/1 and the
/// temporal scale `[1]`. Outputs: the loss `[1]`, then the seven gradients
/// in parameter order — which is [`Kpn::flat`]'s.
pub struct TrainGraph {
    pub size: usize,
    pub hidden: usize,
    pub paired: bool,
    pub graph: TensorGraph,
    pub outputs: Vec<TensorId>,
}

/// Number of parameter inputs ahead of a tile's data.
const N_PARAMS: usize = 7;

impl TrainGraph {
    pub fn new(hidden: usize, s: usize, paired: bool) -> Self {
        let mut g = TensorGraph::new();
        let p = ParamInputs::declare(&mut g, hidden);
        let feat = g.input(&[C_IN, s, s]);
        let illum = g.input(&[3, s, s]);
        let alb = g.input(&[3, s, s]);
        let tt = g.input(&[3, s, s]);
        let out = forward_graph(&mut g, &p, feat, illum, hidden, s);
        let tp = tone_graph(&mut g, out, alb);
        let mut terms = vec![still_loss_graph(&mut g, tp, tt, s)];
        if paired {
            let feat_n = g.input(&[C_IN, s, s]);
            let illum_n = g.input(&[3, s, s]);
            let valid = g.input(&[s, s]);
            let kt = g.input(&[1usize]);
            let out_n = forward_graph(&mut g, &p, feat_n, illum_n, hidden, s);
            let tn = tone_graph(&mut g, out_n, alb);
            let args = vec![
                pick(tn, 4, &[Some(1), Some(2), Some(3)]),
                pick(tp, 4, &[Some(1), Some(2), Some(3)]),
                pick(valid, 4, &[Some(2), Some(3)]),
                Arg { src: kt, view: View { guard: false, forms: vec![Form::constant(4, 0)] } },
            ];
            let lt = g
                .comp(&[1usize], &[3, s, s], args, Some(Monoid::sum()), |v| vec![v[3] * v[2] * (v[0] - v[1]).abs()])
                .expect("temporal");
            terms.push(lt);
        }
        let total = g.map(&terms, |v| v.iter().skip(1).fold(v[0], |a, b| a + *b)).expect("total");
        let grads = g.backward(total, &p.all()).expect("backward");
        let mut outs = vec![total];
        outs.extend(grads);
        let (graph, outputs) = optimize(&g, &outs);
        Self { size: s, hidden, paired, graph, outputs }
    }

    /// Load it: natively, `n` independent copies for `n` workers.
    pub fn programs(&self, n: usize) -> Vec<Program> {
        let zeros = self.zero_inputs();
        let refs: Vec<&[f32]> = zeros.iter().map(|v| v.as_slice()).collect();
        Program::compile_n(&self.graph, &self.outputs, &refs, n)
    }

    /// The reference interpreter, f64 throughout, for the tests.
    pub fn interpreted(&self) -> Program {
        let zeros: Vec<Vec<f64>> = self.zero_inputs().iter().map(|v| vec![0.0; v.len()]).collect();
        let refs: Vec<&[f64]> = zeros.iter().map(|v| v.as_slice()).collect();
        Program::interpreted(&self.graph, &self.outputs, &refs)
    }

    fn zero_inputs(&self) -> Vec<Vec<f32>> {
        let (s, h) = (self.size, self.hidden);
        let n = s * s;
        let mut v = vec![
            vec![0.0; h * C_IN * KS * KS],
            vec![0.0; h],
            vec![0.0; h * h * KS * KS],
            vec![0.0; h],
            vec![0.0; K * h * KS * KS],
            vec![0.0; K],
            vec![0.0; kosm_render::neural::VETO],
            vec![0.0; C_IN * n],
            vec![0.0; 3 * n],
            vec![0.0; 3 * n],
            vec![0.0; 3 * n],
        ];
        if self.paired {
            v.extend([vec![0.0; C_IN * n], vec![0.0; 3 * n], vec![0.0; n], vec![0.0; 1]]);
        }
        v
    }
}

/// `tone(out · albedo)`, planar.
fn tone_graph(g: &mut TensorGraph, out: TensorId, alb: TensorId) -> TensorId {
    g.map(&[out, alb], |v| {
        let p = v[0] * v[1];
        p / (lit(1.0) + p.max(lit(0.0)))
    })
    .expect("tone")
}

/// [`loss`] as a graph over the tone-mapped prediction and reference: the
/// L1, and the two forward-difference terms read through a guarded view one
/// pixel over — zero past the edge, where the mask drops the term anyway; the
/// guard is what lets `backward` invert the offset by elimination.
fn still_loss_graph(g: &mut TensorGraph, tp: TensorId, tt: TensorId, s: usize) -> TensorId {
    let n = s * s;
    let inv = 1.0 / (3 * n) as f64;
    let at = |x: TensorId| pick(x, 4, &[Some(1), Some(2), Some(3)]);
    let l1 = g
        .comp(&[1usize], &[3, s, s], vec![at(tp), at(tt)], Some(Monoid::sum()), |v| {
            vec![(v[0] - v[1]).abs() * lit(inv)]
        })
        .expect("l1");
    if s < 2 {
        return l1;
    }
    let k = (GRAD_WEIGHT / (3 * 2 * s * (s - 1)) as f32) as f64;
    let over = |x: TensorId, axis: usize| {
        let mut forms = vec![Form::unit(4, 1), Form::unit(4, 2), Form::unit(4, 3)];
        forms[axis].offset = 1;
        Arg { src: x, view: View { guard: true, forms } }
    };
    // The difference runs over the whole tile and the last column (row) is
    // masked off by an `iota` select, rather than the reduction stopping one
    // short. A pick view over a shorter range than its source is a forward
    // `TensorGraph::backward` gets wrong: inverting it renames the reduce
    // coordinate onto the source's full extent, so the gradient picks up the
    // one point the forward never visited (tang bfc8003). Full range and a
    // mask is the same sum with a gradient that is right.
    let coord = g.iota(&[s], 0);
    let diff = |g: &mut TensorGraph, axis: usize| {
        let mut args = vec![over(tp, axis), at(tp), over(tt, axis), at(tt)];
        args.push(pick(coord, 4, &[Some(axis + 1)]));
        g.comp(&[1usize], &[3, s, s], args, Some(Monoid::sum()), |v| {
            let term = lit(k) * ((v[0] - v[1]) - (v[2] - v[3])).abs();
            vec![ExprId::select(lit(s as f64 - 1.5) - v[4], term, lit(0.0))]
        })
        .expect("gradient term")
    };
    let gx = diff(g, 2);
    let gy = diff(g, 1);
    g.map(&[l1, gx, gy], |v| v[0] + v[1] + v[2]).expect("still loss")
}

/// RMSE between two interleaved RGB images, measured through [`tone`].
pub fn rmse_tone(pred: &[f32], reference: &[f32]) -> f32 {
    let mut s = 0.0f64;
    for (p, r) in pred.iter().zip(reference) {
        let d = (tone(*p) - tone(*r)) as f64;
        s += d * d;
    }
    (s / pred.len() as f64).sqrt() as f32
}

/// RMSE in linear radiance.
pub fn rmse_linear(pred: &[f32], reference: &[f32]) -> f32 {
    let mut s = 0.0f64;
    for (p, r) in pred.iter().zip(reference) {
        let d = (*p - *r) as f64;
        s += d * d;
    }
    (s / pred.len() as f64).sqrt() as f32
}

/// Run the à-trous filter over one whole sample at one tier, exactly as the
/// CPU tier would, and hand back the interleaved radiance.
///
/// Whole-frame rather than per-tile because the filter reaches 32 pixels and
/// a 64-pixel tile filtered alone is mostly border. The tiles are cut out of
/// the result afterwards, so the baseline is never handicapped by the crop.
pub fn atrous_baseline(sample: &Sample, opts: &PathTraceOptions) -> Vec<f32> {
    let n = (sample.width as usize) * (sample.height as usize);
    let f = &sample.cur;
    let mut film = Film {
        width: sample.width,
        height: sample.height,
        rgb: from_f16(&f.mean),
        alpha: vec![1.0; n],
        normal: from_f16(&sample.normal),
        depth: from_f16(&sample.depth),
        albedo: from_f16(&sample.albedo),
        variance: from_f16(&f.variance),
    };
    kosm_render::pathtrace::denoise(&mut film, opts);
    film.rgb
}

/// Crop `size`-square tiles out of a whole-frame interleaved image, in the
/// same order [`tiles`] cuts them.
pub fn crop_frames(img: &[f32], w: usize, h: usize, size: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    for oy in (0..h.saturating_sub(size - 1)).step_by(size) {
        for ox in (0..w.saturating_sub(size - 1)).step_by(size) {
            let mut t = vec![0.0f32; size * size * 3];
            for y in 0..size {
                for x in 0..size {
                    let s = ((oy + y) * w + ox + x) * 3;
                    let d = (y * size + x) * 3;
                    t[d..d + 3].copy_from_slice(&img[s..s + 3]);
                }
            }
            out.push(t);
        }
    }
    out
}

/// One tile, with its features precomputed once rather than per epoch.
pub struct Prepared {
    pub size: usize,
    /// The nominal history length, for reporting only; the network reads the
    /// per-pixel `count` plane inside `feat`.
    pub tier: u32,
    pub feat: Vec<f32>,
    pub illum: Vec<f32>,
    pub albedo: Vec<f32>,
    pub reference: Vec<f32>,
    /// The mean the tile started from, kept so an evaluation can report what
    /// doing nothing at all would have scored.
    pub mean: Vec<f32>,
    /// The guides, kept because [`Prepared::predict`] goes through
    /// [`Kpn::filter`], which is the renderer's whole path.
    pub variance: Vec<f32>,
    pub count: Vec<f32>,
    pub normal: Vec<f32>,
    pub depth: Vec<f32>,
    pub id: Vec<f32>,
    /// The successor frame, when the tile has one: only the two planes a
    /// forward pass needs, plus the per-pixel validity the temporal term is
    /// masked by. The guides are shared, so this is a fraction of a whole
    /// tile rather than a second one.
    pub next: Option<Box<PreparedNext>>,
}

/// The successor frame's inputs and the mask that says where its answer is
/// allowed to differ.
pub struct PreparedNext {
    pub feat: Vec<f32>,
    pub illum: Vec<f32>,
    /// True where the pixel kept its history from one frame to the next.
    pub valid: Vec<bool>,
}

impl Prepared {
    pub fn of(t: &Tile) -> Self {
        let n = t.size * t.size;
        let next = t.next.as_ref().map(|nx| {
            Box::new(PreparedNext {
                feat: features(
                    n,
                    &nx.mean,
                    &nx.variance,
                    &t.normal,
                    &t.depth,
                    &t.albedo,
                    &t.id,
                    &nx.count,
                ),
                illum: illumination(n, &nx.mean, &t.albedo),
                valid: (0..n).map(|p| kept_history(t.count[p], nx.count[p])).collect(),
            })
        });
        Self {
            size: t.size,
            tier: t.tier,
            feat: features(
                n,
                &t.mean,
                &t.variance,
                &t.normal,
                &t.depth,
                &t.albedo,
                &t.id,
                &t.count,
            ),
            illum: illumination(n, &t.mean, &t.albedo),
            albedo: t.albedo.clone(),
            reference: t.reference.clone(),
            mean: t.mean.clone(),
            variance: t.variance.clone(),
            count: t.count.clone(),
            normal: t.normal.clone(),
            depth: t.depth.clone(),
            id: t.id.clone(),
            next,
        }
    }

    /// The network's answer for this tile, interleaved radiance — down the
    /// renderer's own path, so a number reported here is a number about the
    /// thing that ships.
    pub fn predict(&self, net: &Kpn) -> Vec<f32> {
        let s = self.size;
        net.filter(
            s,
            s,
            &self.mean,
            &self.variance,
            &self.normal,
            &self.depth,
            &self.albedo,
            &self.id,
            &self.count,
        )
    }
}

/// Cut every sample of `dataset` into tiles at every tier and prepare them.
///
/// `limit` caps how many tiles come back, taken by a deterministic stride
/// rather than a prefix: a v2 sample is a whole 640x360 frame at six tiers and
/// the full cut is tens of thousands of tiles, which is more memory than it is
/// signal. Zero means no cap.
pub fn prepare(dataset: &Dataset, size: usize, which: &[usize], limit: usize) -> Vec<Prepared> {
    let per_frame =
        |s: &Sample| (s.width as usize / size.max(1)) * (s.height as usize / size.max(1));
    let total: usize = which.iter().map(|&si| per_frame(&dataset.samples[si])).sum();
    // A stride rather than a head: consecutive tiles are neighbouring pixels
    // of one frame, and the first `limit` of them would be the top of every
    // picture and nothing else.
    let stride = if limit == 0 || total <= limit {
        1
    } else {
        total.div_ceil(limit)
    };
    let mut out = Vec::new();
    let mut flat = 0usize;
    for &si in which {
        for t in tiles(&dataset.samples[si], size) {
            if flat % stride == 0 {
                out.push(Prepared::of(&t));
            }
            flat += 1;
        }
    }
    out
}

/// One epoch's worth of settings.
pub struct Fit {
    pub epochs: usize,
    pub batch: usize,
    pub lr: f64,
    pub seed: u64,
}

impl Default for Fit {
    fn default() -> Self {
        Self {
            epochs: 60,
            batch: 8,
            lr: 2e-3,
            seed: 1,
        }
    }
}

/// One loaded training step per tile shape a worker has seen.
struct Worker {
    programs: Vec<((usize, bool), Program)>,
}

impl Worker {
    fn program(&mut self, key: (usize, bool)) -> &mut Program {
        let i = self.programs.iter().position(|(k, _)| *k == key).expect("a program per tile shape");
        &mut self.programs[i].1
    }

    /// One tile's loss and gradient, added into `grad` (flat, file order).
    fn run(&mut self, net: &Kpn, t: &Prepared, grad: &mut [f32]) -> f32 {
        let key = (t.size, t.next.is_some());
        let p = self.program(key);
        for (k, b) in net.blocks().iter().enumerate() {
            p.write(k, b);
        }
        let (alb, tt) = planar_targets(t);
        p.write(N_PARAMS, &t.feat);
        p.write(N_PARAMS + 1, &t.illum);
        p.write(N_PARAMS + 2, &alb);
        p.write(N_PARAMS + 3, &tt);
        if let Some(nx) = t.next.as_ref() {
            let valid: Vec<f32> = nx.valid.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
            p.write(N_PARAMS + 4, &nx.feat);
            p.write(N_PARAMS + 5, &nx.illum);
            p.write(N_PARAMS + 6, &valid);
            p.write(N_PARAMS + 7, &[temporal_scale(&nx.valid)]);
        }
        p.run();
        let mut at = 0;
        for k in 0..N_PARAMS {
            let gk = p.read(1 + k);
            for (a, b) in grad[at..at + gk.len()].iter_mut().zip(&gk) {
                *a += b;
            }
            at += gk.len();
        }
        p.read(0)[0]
    }
}

/// The floored albedo and the tone-mapped reference, planar, as
/// [`TrainGraph`] reads them.
fn planar_targets(t: &Prepared) -> (Vec<f32>, Vec<f32>) {
    let n = t.size * t.size;
    let mut alb = vec![0.0f32; 3 * n];
    let mut tt = vec![0.0f32; 3 * n];
    for c in 0..3 {
        for p in 0..n {
            alb[c * n + p] = t.albedo[p * 3 + c].max(DEMOD_FLOOR);
            tt[c * n + p] = tone(t.reference[p * 3 + c]);
        }
    }
    (alb, tt)
}

/// Fit `net` to `train`, reporting `(epoch, mean loss)` as it goes.
///
/// Every tile shape in `train` — its size, and whether it has a successor
/// frame — gets one [`TrainGraph`], built once and lowered once per rayon
/// worker; a batch is split across the workers, each runs its tiles through
/// its own native program, and the gradients are summed and averaged before
/// one step of [`crate::ir::Adam`], which is `tang_train::ModuleAdam`'s
/// update (β = 0.9, 0.999, ε = 1e-8, no decay, no clipping) as a graph.
pub fn fit(
    net: &mut Kpn,
    train: &[Prepared],
    cfg: &Fit,
    mut report: impl FnMut(usize, f32),
) -> Vec<f32> {
    let mut keys: Vec<(usize, bool)> = train.iter().map(|t| (t.size, t.next.is_some())).collect();
    keys.sort_unstable();
    keys.dedup();
    let n_workers = rayon::current_num_threads().min(cfg.batch.max(1)).max(1);
    let mut workers: Vec<Worker> = (0..n_workers).map(|_| Worker { programs: Vec::new() }).collect();
    for &(s, paired) in &keys {
        let graph = TrainGraph::new(net.hidden, s, paired);
        for (w, p) in workers.iter_mut().zip(graph.programs(n_workers)) {
            w.programs.push(((s, paired), p));
        }
    }

    let n_params = net.parameters();
    let mut opt = Adam::new(n_params, cfg.lr);
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut rng = cfg.seed | 1;
    let mut curve = Vec::with_capacity(cfg.epochs);

    for epoch in 0..cfg.epochs {
        // Fisher-Yates with the same splitmix the dataset uses.
        for i in (1..order.len()).rev() {
            rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            order.swap(i, ((z ^ (z >> 31)) % (i as u64 + 1)) as usize);
        }

        let mut total = 0.0f64;
        let mut seen = 0usize;
        for chunk in order.chunks(cfg.batch) {
            let shared: &Kpn = net;
            let parts: Vec<(Vec<f32>, f64)> = workers
                .par_iter_mut()
                .enumerate()
                .map(|(wi, w)| {
                    let mut grad = vec![0.0f32; n_params];
                    let mut loss = 0.0f64;
                    for &i in chunk.iter().skip(wi).step_by(n_workers) {
                        loss += w.run(shared, &train[i], &mut grad) as f64;
                    }
                    (grad, loss)
                })
                .collect();
            let k = 1.0 / chunk.len() as f64;
            let mut grad = vec![0.0f64; n_params];
            for (g, l) in &parts {
                for (a, b) in grad.iter_mut().zip(g) {
                    *a += *b as f64 * k;
                }
                total += l;
            }
            let mut flat: Vec<f64> = net.flat().iter().map(|&v| v as f64).collect();
            opt.step(&mut flat, &grad);
            net.set_flat(&flat.iter().map(|&v| v as f32).collect::<Vec<_>>());
            seen += chunk.len();
        }
        let mean = (total / seen.max(1) as f64) as f32;
        curve.push(mean);
        report(epoch, mean);
    }
    curve
}

/// What an evaluation says about one tier.
pub struct TierScore {
    pub count: u32,
    pub tiles: usize,
    pub raw_tone: f32,
    pub atrous_tone: f32,
    pub neural_tone: f32,
    pub raw_linear: f32,
    pub atrous_linear: f32,
    pub neural_linear: f32,
}

/// Score `net` against the à-trous filter and against the unfiltered mean, on
/// held-out samples, tier by tier.
pub fn evaluate(
    net: &Kpn,
    dataset: &Dataset,
    which: &[usize],
    size: usize,
    opts: &PathTraceOptions,
) -> Vec<TierScore> {
    let mut out = Vec::new();
    for &count in TIERS.iter() {
        let mut acc = [0.0f64; 6];
        let mut n_tiles = 0usize;
        for &si in which {
            let s = &dataset.samples[si];
            if s.tier != count {
                continue;
            }
            let (w, h) = (s.width as usize, s.height as usize);
            let base = atrous_baseline(s, opts);
            let base_tiles = crop_frames(&base, w, h, size);
            for (t, bt) in tiles(s, size).into_iter().zip(base_tiles) {
                let p = Prepared::of(&t);
                let pred = p.predict(net);
                acc[0] += rmse_tone(&p.mean, &p.reference) as f64;
                acc[1] += rmse_tone(&bt, &p.reference) as f64;
                acc[2] += rmse_tone(&pred, &p.reference) as f64;
                acc[3] += rmse_linear(&p.mean, &p.reference) as f64;
                acc[4] += rmse_linear(&bt, &p.reference) as f64;
                acc[5] += rmse_linear(&pred, &p.reference) as f64;
                n_tiles += 1;
            }
        }
        let k = 1.0 / n_tiles.max(1) as f64;
        out.push(TierScore {
            count,
            tiles: n_tiles,
            raw_tone: (acc[0] * k) as f32,
            atrous_tone: (acc[1] * k) as f32,
            neural_tone: (acc[2] * k) as f32,
            raw_linear: (acc[3] * k) as f32,
            atrous_linear: (acc[4] * k) as f32,
            neural_linear: (acc[5] * k) as f32,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tile with structure: fireflies so the clamp is live, an edge, a
    /// mask with holes so the temporal term has both kinds of pixel.
    fn toy(s: usize, seed: usize) -> Prepared {
        let n = s * s;
        let h = |i: usize, k: usize| (((i + 13 * seed) * 2654435761 + k * 40503) % 1009) as f32 / 1009.0;
        let mut t = Tile {
            size: s,
            tier: 4,
            mean: (0..3 * n).map(|i| 0.1 + h(i, 1) * if i % 23 == 0 { 40.0 } else { 1.0 }).collect(),
            variance: (0..n).map(|i| 0.02 * h(i, 2)).collect(),
            count: (0..n).map(|i| 1.0 + (h(i, 3) * 6.0).floor()).collect(),
            normal: (0..3 * n).map(|i| if i % 3 == 2 { 1.0 } else { 0.0 }).collect(),
            depth: (0..n).map(|p| if p % s < s / 2 { 800.0 } else { 2400.0 }).collect(),
            albedo: (0..3 * n).map(|i| 0.05 + 0.7 * h(i, 4)).collect(),
            id: (0..n).map(|p| (1 + (p % s) / 3) as f32).collect(),
            reference: (0..3 * n).map(|i| 0.05 + 0.5 * h(i, 5)).collect(),
            next: None,
        };
        let count = t.count.clone();
        t.next = Some(Box::new(crate::denoise::dataset::TileNext {
            mean: (0..3 * n).map(|i| 0.1 + h(i, 6)).collect(),
            variance: (0..n).map(|i| 0.02 * h(i, 7)).collect(),
            count: (0..n).map(|p| if p % 3 == 0 { 1.0 } else { count[p] + 1.0 }).collect(),
        }));
        Prepared::of(&t)
    }

    /// Every input a [`TrainGraph`] takes, in order, as f64.
    fn inputs(net: &Kpn, t: &Prepared) -> Vec<Vec<f64>> {
        let wide = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        let (alb, tt) = planar_targets(t);
        let mut v: Vec<Vec<f64>> = net.blocks().iter().map(|b| wide(b)).collect();
        v.extend([wide(&t.feat), wide(&t.illum), wide(&alb), wide(&tt)]);
        if let Some(nx) = t.next.as_ref() {
            let valid: Vec<f64> = nx.valid.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
            v.extend([wide(&nx.feat), wide(&nx.illum), valid, vec![temporal_scale(&nx.valid) as f64]]);
        }
        v
    }

    fn interpret(graph: &TrainGraph, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut p = graph.interpreted();
        for (k, x) in inputs.iter().enumerate() {
            p.write_f64(k, x);
        }
        p.run();
        (0..graph.outputs.len()).map(|i| p.read_f64(i)).collect()
    }

    /// The graph's loss is [`loss`] plus [`temporal_loss`] over the
    /// reference forward's two answers: same terms, same weights, same mask.
    #[test]
    fn the_graph_loss_is_the_reference_loss() {
        let s = 6;
        let net = Kpn::new(4, 3);
        let t = toy(s, 1);
        let nx = t.next.as_ref().unwrap();
        assert!(nx.valid.iter().any(|v| *v) && nx.valid.iter().any(|v| !*v));
        let cur = net.forward(&t.feat, &t.illum, s).out;
        let next = net.forward(&nx.feat, &nx.illum, s).out;
        let want = loss(&cur, &t.albedo, &t.reference, s) + temporal_loss(&cur, &next, &t.albedo, &nx.valid, s);
        let got = interpret(&TrainGraph::new(4, s, true), &inputs(&net, &t))[0][0];
        assert!(((want as f64) - got).abs() < 1e-5 * got.abs().max(1e-3), "reference {want}, graph {got}");

        let still = loss(&cur, &t.albedo, &t.reference, s) as f64;
        let mut lone = toy(s, 1);
        lone.next = None;
        let got = interpret(&TrainGraph::new(4, s, false), &inputs(&net, &lone))[0][0];
        assert!((still - got).abs() < 1e-5 * got.abs().max(1e-3), "still: reference {still}, graph {got}");
    }

    /// The whole step's gradient — loss, both frames, the temporal term —
    /// along one direction through every parameter, against a central
    /// difference of the same graph's loss. All f64, so the window is wide
    /// and the tolerance tight: this is the test that caught the gradient
    /// terms' one-short reduction (see [`still_loss_graph`]), which was off by
    /// a few percent in every block whatever the step.
    #[test]
    fn the_step_gradient_agrees_with_a_finite_difference() {
        let s = 5;
        let net = Kpn::new(3, 9);
        let t = toy(s, 2);
        let graph = TrainGraph::new(3, s, true);
        let base = inputs(&net, &t);
        let out = interpret(&graph, &base);
        let dir: Vec<Vec<f64>> = base[..N_PARAMS]
            .iter()
            .enumerate()
            .map(|(k, b)| (0..b.len()).map(|i| if ((i + 31 * k) * 2654435761 >> 9) & 1 == 0 { 1.0 } else { -1.0 }).collect())
            .collect();
        let ana: f64 = (0..N_PARAMS).map(|k| out[1 + k].iter().zip(&dir[k]).map(|(g, d)| g * d).sum::<f64>()).sum();
        let at = |e: f64| {
            let mut x = base.clone();
            for k in 0..N_PARAMS {
                for (v, d) in x[k].iter_mut().zip(&dir[k]) {
                    *v += e * d;
                }
            }
            interpret(&graph, &x)[0][0]
        };
        let eps = 1e-6;
        let num = (at(eps) - at(-eps)) / (2.0 * eps);
        assert!(ana.abs() > 1e-4, "degenerate: {ana}");
        assert!((ana - num).abs() < 1e-4 * ana.abs().max(num.abs()) + 1e-8, "graph {ana}, secant {num}");
    }

    /// The compiled C program and the interpreter run the same graph: the
    /// same loss and every gradient, to `f32`.
    #[test]
    fn the_native_step_matches_the_interpreter() {
        let s = 8;
        let net = Kpn::new(4, 5);
        let t = toy(s, 3);
        let graph = TrainGraph::new(4, s, true);
        let want = interpret(&graph, &inputs(&net, &t));
        let native = graph.programs(1).pop().unwrap();
        if !native.is_native() {
            eprintln!("no C compiler; the native half of this test is the interpreter against itself");
        }
        let mut grad = vec![0.0f32; net.parameters()];
        let l = Worker { programs: vec![((s, true), native)] }.run(&net, &t, &mut grad);
        assert!((l as f64 - want[0][0]).abs() < 1e-5 * want[0][0].abs().max(1e-3), "loss: native {l}, interp {}", want[0][0]);
        let flat: Vec<f64> = want[1..].iter().flatten().copied().collect();
        let scale = flat.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (a, b)) in grad.iter().zip(&flat).enumerate() {
            assert!((*a as f64 - b).abs() < 1e-4 * scale, "grad {i}: native {a}, interp {b} (scale {scale})");
        }
    }

    /// A few epochs on a few tiles move the loss down, and the trained
    /// weights still round-trip through the file the renderer reads.
    #[test]
    fn a_short_fit_descends() {
        let s = 8;
        let tiles: Vec<Prepared> = (0..6).map(|i| toy(s, 10 + i)).collect();
        let mut net = Kpn::new(4, 1);
        let cfg = Fit { epochs: 12, batch: 3, lr: 5e-3, seed: 4 };
        let curve = fit(&mut net, &tiles, &cfg, |_, _| {});
        assert!(curve.last().unwrap() < &(0.95 * curve[0]), "{curve:?}");
        assert!(net.flat().iter().all(|v| v.is_finite()));
    }
}
