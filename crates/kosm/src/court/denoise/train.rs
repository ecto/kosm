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
use tang_train::{ModuleAdam, Optimizer, Parameter};

use super::dataset::{Dataset, Sample, TIERS, Tile, from_f16, tiles};
use super::kpn::{DEMOD_FLOOR, Grads, Kpn, features, illumination};

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

/// Its derivative.
#[inline]
fn tone_d(x: f32) -> f32 {
    let d = 1.0 + x.max(0.0);
    1.0 / (d * d)
}

/// The loss and its gradient with respect to the *demodulated* filtered
/// illumination, which is what [`Kpn::backward`] wants.
///
/// `out` and `g` are planar `[3][n]`; `albedo` and `reference` are
/// interleaved, as the dataset stores them.
pub fn loss_and_grad(
    out: &[f32],
    albedo: &[f32],
    reference: &[f32],
    s: usize,
    g: &mut [f32],
) -> f32 {
    let n = s * s;
    g.fill(0.0);
    let a = |c: usize, p: usize| albedo[p * 3 + c].max(DEMOD_FLOOR);

    // The tone-compressed prediction and target, kept so the gradient term
    // can difference them without recomputing.
    let mut tp = vec![0.0f32; 3 * n];
    let mut tt = vec![0.0f32; 3 * n];
    let mut dtp = vec![0.0f32; 3 * n];
    for c in 0..3 {
        for p in 0..n {
            let pred = out[c * n + p] * a(c, p);
            tp[c * n + p] = tone(pred);
            tt[c * n + p] = tone(reference[p * 3 + c]);
            // d tone(pred) / d out
            dtp[c * n + p] = tone_d(pred) * a(c, p);
        }
    }

    let inv = 1.0 / (3 * n) as f32;
    let mut loss = 0.0;
    for i in 0..3 * n {
        let d = tp[i] - tt[i];
        loss += d.abs() * inv;
        g[i] += d.signum() * inv * dtp[i];
    }

    // Forward differences in x and y, in the same tone-compressed space.
    // The border tap is skipped rather than clamped: a clamped difference is
    // identically zero in both images and would only dilute the term.
    let pairs = 3 * 2 * s * s.saturating_sub(1);
    if pairs > 0 {
        let k = GRAD_WEIGHT / pairs as f32;
        let edge = |i: usize, j: usize, g: &mut [f32], loss: &mut f32| {
            let d = (tp[j] - tp[i]) - (tt[j] - tt[i]);
            *loss += k * d.abs();
            let sg = d.signum() * k;
            g[j] += sg * dtp[j];
            g[i] -= sg * dtp[i];
        };
        for c in 0..3 {
            let base = c * n;
            for y in 0..s {
                for x in 0..s.saturating_sub(1) {
                    edge(base + y * s + x, base + y * s + x + 1, g, &mut loss);
                }
            }
            for y in 0..s.saturating_sub(1) {
                for x in 0..s {
                    edge(base + y * s + x, base + (y + 1) * s + x, g, &mut loss);
                }
            }
        }
    }
    loss
}

/// The temporal consistency term and its gradient with respect to both
/// frames' demodulated filtered illumination.
///
/// `cur` and `next` are planar `[3][n]` network outputs for the same tile at
/// consecutive frames; `valid` is one flag per pixel from [`kept_history`].
/// Both gradients are *added* to, so a caller can fold this on top of
/// [`loss_and_grad`]'s.
pub fn temporal_loss_and_grad(
    cur: &[f32],
    next: &[f32],
    albedo: &[f32],
    valid: &[bool],
    s: usize,
    g_cur: &mut [f32],
    g_next: &mut [f32],
) -> f32 {
    let n = s * s;
    let live = valid.iter().filter(|v| **v).count();
    if live == 0 {
        return 0.0;
    }
    let k = TEMPORAL_WEIGHT / (3 * live) as f32;
    let mut loss = 0.0;
    for c in 0..3 {
        for p in 0..n {
            if !valid[p] {
                continue;
            }
            let a = albedo[p * 3 + c].max(DEMOD_FLOOR);
            let (pc, pn) = (cur[c * n + p] * a, next[c * n + p] * a);
            let d = tone(pn) - tone(pc);
            loss += k * d.abs();
            let sg = d.signum() * k;
            g_next[c * n + p] += sg * tone_d(pn) * a;
            g_cur[c * n + p] -= sg * tone_d(pc) * a;
        }
    }
    loss
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

/// Fit `net` to `train`, reporting `(epoch, mean loss)` as it goes.
///
/// The optimiser is `tang_train`'s `ModuleAdam`, stepped over the same
/// `Parameter`s the network holds; the gradients come from
/// [`Kpn::backward`] rather than from a `Module::backward`, because the
/// convolutions here are ours (see [`super::kpn`]).
pub fn fit(
    net: &mut Kpn,
    train: &[Prepared],
    cfg: &Fit,
    mut report: impl FnMut(usize, f32),
) -> Vec<f32> {
    let mut opt = ModuleAdam::new(cfg.lr);
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
            let zero = Grads::zeros(net);
            let (grads, loss) = chunk
                .par_iter()
                .map(|&i| {
                    let t = &train[i];
                    let n = t.size * t.size;
                    let act = net.forward(&t.feat, &t.illum, t.size);
                    let mut g = vec![0.0f32; 3 * n];
                    let mut l =
                        loss_and_grad(&act.out, &t.albedo, &t.reference, t.size, &mut g);
                    let mut gr = Grads::zeros(net);
                    // The temporal term needs the successor frame's forward
                    // too, and contributes a gradient to both. Its backward
                    // accumulates into the same `Grads`, which is what makes
                    // the two frames one training example rather than two.
                    if let Some(nx) = t.next.as_ref() {
                        let act_n = net.forward(&nx.feat, &nx.illum, t.size);
                        let mut gn = vec![0.0f32; 3 * n];
                        l += temporal_loss_and_grad(
                            &act.out,
                            &act_n.out,
                            &t.albedo,
                            &nx.valid,
                            t.size,
                            &mut g,
                            &mut gn,
                        );
                        net.backward(&nx.feat, &nx.illum, &act_n, &gn, &mut gr);
                    }
                    net.backward(&t.feat, &t.illum, &act, &g, &mut gr);
                    (gr, l as f64)
                })
                .reduce(
                    || (zero.clone(), 0.0f64),
                    |mut a, b| {
                        a.0.add(&b.0);
                        a.1 += b.1;
                        a
                    },
                );
            let mut grads = grads;
            grads.scale(1.0 / chunk.len() as f32);
            step(net, &grads, &mut opt);
            total += loss;
            seen += chunk.len();
        }
        let mean = (total / seen.max(1) as f64) as f32;
        curve.push(mean);
        report(epoch, mean);
    }
    curve
}

/// Hand the accumulated gradients to the optimiser and take one step.
fn step(net: &mut Kpn, g: &Grads, opt: &mut ModuleAdam) {
    let set = |p: &mut Parameter<f32>, v: &[f32]| {
        p.grad = Some(tang_tensor::Tensor::new(v.to_vec(), p.data.shape().clone()));
    };
    set(&mut net.l1.weight, &g.w1);
    set(&mut net.l1.bias, &g.b1);
    set(&mut net.l2.weight, &g.w2);
    set(&mut net.l2.bias, &g.b2);
    set(&mut net.l3.weight, &g.w3);
    set(&mut net.l3.bias, &g.b3);
    set(&mut net.veto, &g.veto);
    // The order is fixed for the life of the run; `ModuleAdam` sizes its
    // moment vectors on the first step and indexes them positionally.
    let mut params: Vec<&mut Parameter<f32>> = vec![
        &mut net.l1.weight,
        &mut net.l1.bias,
        &mut net.l2.weight,
        &mut net.l2.bias,
        &mut net.l3.weight,
        &mut net.l3.bias,
        &mut net.veto,
    ];
    opt.step(&mut params);
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

    /// The temporal term is a new gradient path and it flows into *two*
    /// forwards, which is exactly the kind of term that is easy to get
    /// backwards. Both halves are differenced.
    #[test]
    fn the_temporal_gradient_agrees_with_a_finite_difference() {
        let s = 4;
        let n = s * s;
        let mut cur = vec![0.0f32; 3 * n];
        let mut next = vec![0.0f32; 3 * n];
        let mut albedo = vec![0.0f32; 3 * n];
        for i in 0..3 * n {
            cur[i] = 0.2 + (i % 7) as f32 * 0.09;
            next[i] = 0.2 + (i % 5) as f32 * 0.13;
            albedo[i] = 0.3 + (i % 5) as f32 * 0.07;
        }
        // Some pixels held their history and some did not; a mask of all-true
        // would not test that the mask is read at all.
        let valid: Vec<bool> = (0..n).map(|p| p % 3 != 0).collect();
        let mut gc = vec![0.0f32; 3 * n];
        let mut gn = vec![0.0f32; 3 * n];
        temporal_loss_and_grad(&cur, &next, &albedo, &valid, s, &mut gc, &mut gn);

        let eps = 1e-3;
        let mut d0 = vec![0.0f32; 3 * n];
        let mut d1 = vec![0.0f32; 3 * n];
        let mut probe = |v: &mut Vec<f32>, i: usize, other: &[f32], first: bool| -> f32 {
            let o = v[i];
            v[i] = o + eps;
            let lp = if first {
                temporal_loss_and_grad(v, other, &albedo, &valid, s, &mut d0, &mut d1)
            } else {
                temporal_loss_and_grad(other, v, &albedo, &valid, s, &mut d0, &mut d1)
            };
            v[i] = o - eps;
            let lm = if first {
                temporal_loss_and_grad(v, other, &albedo, &valid, s, &mut d0, &mut d1)
            } else {
                temporal_loss_and_grad(other, v, &albedo, &valid, s, &mut d0, &mut d1)
            };
            v[i] = o;
            (lp - lm) / (2.0 * eps)
        };
        for i in (0..3 * n).step_by(5) {
            let snapshot = next.clone();
            let num = probe(&mut cur, i, &snapshot, true);
            let scale = num.abs().max(gc[i].abs()).max(1e-6);
            assert!(
                (num - gc[i]).abs() / scale < 5e-2,
                "cur {i}: analytic {}, numeric {num}",
                gc[i]
            );
            let snapshot = cur.clone();
            let num = probe(&mut next, i, &snapshot, false);
            let scale = num.abs().max(gn[i].abs()).max(1e-6);
            assert!(
                (num - gn[i]).abs() / scale < 5e-2,
                "next {i}: analytic {}, numeric {num}",
                gn[i]
            );
        }
    }

    #[test]
    fn the_loss_gradient_agrees_with_a_finite_difference() {
        let s = 4;
        let n = s * s;
        let mut out = vec![0.0f32; 3 * n];
        let mut albedo = vec![0.0f32; 3 * n];
        let mut reference = vec![0.0f32; 3 * n];
        for i in 0..3 * n {
            out[i] = 0.2 + (i % 7) as f32 * 0.11;
            albedo[i] = 0.3 + (i % 5) as f32 * 0.07;
            reference[i] = 0.25 + (i % 11) as f32 * 0.09;
        }
        let mut g = vec![0.0f32; 3 * n];
        loss_and_grad(&out, &albedo, &reference, s, &mut g);
        let eps = 1e-3;
        let mut scratch = vec![0.0f32; 3 * n];
        for i in (0..3 * n).step_by(5) {
            let o = out[i];
            out[i] = o + eps;
            let lp = loss_and_grad(&out, &albedo, &reference, s, &mut scratch);
            out[i] = o - eps;
            let lm = loss_and_grad(&out, &albedo, &reference, s, &mut scratch);
            out[i] = o;
            let num = (lp - lm) / (2.0 * eps);
            let scale = num.abs().max(g[i].abs()).max(1e-4);
            assert!(
                (num - g[i]).abs() / scale < 5e-2,
                "index {i}: analytic {}, numeric {num}",
                g[i]
            );
        }
    }
}
