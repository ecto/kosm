//! The network: three convolutions that predict a filter, not a picture.
//!
//! # Why a kernel, and not a colour
//!
//! A network that outputs radiance directly is free to invent it, and at one
//! sample per pixel it will: the cheapest way to make an L1 loss small on a
//! noisy input is to hallucinate a plausible smooth image, and what comes out
//! is a picture of the training set rather than of the frame. A
//! kernel-predicting network cannot do that. Its 25 outputs are put through a
//! softmax and used as the weights of a 5x5 average over the *input's own*
//! illumination, so every pixel it produces is a convex combination of light
//! this frame actually measured. It can decide what to average and it cannot
//! decide what colour to be — which is the same contract the à-trous filter
//! signs, with the weights learned instead of stipulated.
//!
//! It also makes the device pass cheap and the failure mode legible. 25
//! weights that sum to one are 25 numbers you can look at, and an
//! oversmoothed frame is a kernel that stayed wide where it should have
//! collapsed.
//!
//! # What is derived
//!
//! Only the forward pass is written down. The training step — forward over
//! both frames of a tile, the loss in [`super::train`], and the gradient of
//! that loss with respect to every weight, bias and the veto's three raws —
//! is one [`TensorGraph`] per tile size ([`super::train::TrainGraph`]), and
//! the gradient is [`TensorGraph::backward`]'s. There is no hand backward any more: no
//! `conv_backward`, no quotient rule through the renormalised gate, no
//! `min`'s active branch picked out by hand for `cap`. The IR differentiates
//! a convolution by renaming coordinates and the veto's sigmoid and clamp by
//! `tang_expr`'s symbolic `diff`, and the finite-difference check in this
//! module's tests holds it to the reference forward.
//!
//! The graph is lowered to C and run natively ([`crate::ir::Program`]), one
//! program per rayon worker. Speed was the hand-written `f32` loops' whole
//! reason to exist — about 80 million multiply-accumulates per 64x64 tile
//! forward, a few thousand tile-passes an epoch — and the generated C, whose
//! inner loops vectorise across a row instead of calling `clamp_i` per tap,
//! runs about four times the tile-passes per second they did on sixteen
//! threads (`examples/denoise_bench.rs`: 64x64 tiles, 32 hidden, ~33 → ~130),
//! with the same loss curve to six digits.
//!
//! Clamp-to-edge padding is the one thing a view cannot say, because
//! `clamp(y − 1)` is not affine. [`pad_clamp`] builds it from affine pieces
//! instead: per padded axis, a guarded read at the interior offset (zero
//! outside the source) plus, for each border position `j`, a guarded read
//! offset so that it lands on the edge row exactly at `j`, gated by an `iota`
//! select `[coord == j]`. Every piece is a `±1` offset under a guard, which
//! is the form whose gradient `backward` inverts by eliminating a coordinate
//! rather than scattering, so the padding costs the backward pass a few
//! adds per element. The test pins it to `clamp_i` bit for bit.
//!
//! [`Kpn::forward`] stays as plain loops: it is the reference the graph is
//! checked against and the body of [`Kpn::filter`].
//!
//! # The contract with the shader
//!
//! [`Kpn::filter`] is the reference implementation of what
//! `kosm_render::gpu::neural`'s WGSL does, feature for feature and tap for
//! tap. [`features`] is where the two agree on what a pixel looks like, and
//! it is the part worth being careful about: a training-time feature that is
//! a hair different from the inference-time one is a network that works in
//! Rust and produces mush on the GPU.

use std::io::{Read, Write};
use std::path::Path;

use crate::ir::{Arg, ExprId, Form, Monoid, TensorGraph, TensorId, View, lit, pick};
use kosm_render::neural::{VETO, VETO_EPS, VETO_FLOOR};
use tang::Scalar;

/// Input feature planes per pixel; see [`features`].
pub const C_IN: usize = 11;
/// The filter footprint: 5x5, so 25 predicted weights per pixel.
pub const TAPS: usize = 5;
/// Predicted weights per pixel.
pub const K: usize = TAPS * TAPS;
/// Convolution kernel size in each hidden layer.
pub const KS: usize = 3;

/// The albedo floor the demodulation divides by, matching
/// `kosm_render::pathtrace`'s `DEMOD_FLOOR` and `history.wgsl`'s.
pub const DEMOD_FLOOR: f32 = 0.01;

/// The scale that maps a first-hit distance in millimetres onto roughly
/// 0..1. The gym is a few thousand millimetres across; this is a soft
/// normalisation, not a near/far plane.
pub const DEPTH_SCALE: f32 = 3000.0;

const MAGIC: &[u8; 8] = b"KOSMKPN2";

/// Per-pixel input features, in the order both the Rust and the WGSL forward
/// read them.
///
/// | plane | what |
/// |-------|------|
/// | 0..3  | `log(1 + illum)`, the demodulated running mean |
/// | 3     | `1/sqrt(count)`, the noise the pixel still has |
/// | 4     | `log(1 + sqrt(var))`, demodulated, the pixel's own error bar |
/// | 5..8  | the world normal |
/// | 8     | `depth / (depth + DEPTH_SCALE)`, zero on background |
/// | 9     | the albedo's luminance |
/// | 10    | a hash of the biased hit id |
///
/// The log compression on the two radiance-like planes is what keeps a light
/// panel at a few hundred from dominating a floor at a tenth. The count plane
/// is what lets one network serve every history length: a pixel on its first
/// sample and one on its sixteenth want different kernels and the network has
/// to be told which it is looking at.
///
/// `count` is **per pixel**, not per tile. On the device it always was —
/// `stats[i].x` — and a frame carried across a camera move has pixels at one
/// sample sitting beside pixels at thirty. The v1 trainer passed one scalar
/// for a whole tile because its dataset was built from *k* independent CPU
/// passes, where that was true; on the reprojected, clamped history the
/// viewer actually runs it is not, and the network was reading a plane that
/// lied to it about exactly the pixels the filter matters most for.
///
/// The id plane is a hash and not the id, for the reason
/// `kosm_render::neural::id_feature` gives: it is a label, and the only thing
/// a convolution should be able to ask of it is whether two pixels match.
pub fn features(
    n: usize,
    mean: &[f32],
    variance: &[f32],
    normal: &[f32],
    depth: &[f32],
    albedo: &[f32],
    id: &[f32],
    count: &[f32],
) -> Vec<f32> {
    let mut f = vec![0.0f32; C_IN * n];
    for p in 0..n {
        let a = [
            albedo[p * 3].max(DEMOD_FLOOR),
            albedo[p * 3 + 1].max(DEMOD_FLOOR),
            albedo[p * 3 + 2].max(DEMOD_FLOOR),
        ];
        let la = luminance(a).max(DEMOD_FLOOR);
        for c in 0..3 {
            f[c * n + p] = (1.0 + mean[p * 3 + c] / a[c]).max(1e-8).ln();
        }
        f[3 * n + p] = 1.0 / count[p].max(1.0).sqrt();
        f[4 * n + p] = (1.0 + (variance[p].max(0.0) / (la * la)).sqrt()).ln();
        for c in 0..3 {
            f[(5 + c) * n + p] = normal[p * 3 + c];
        }
        let d = depth[p];
        f[8 * n + p] = if d > 0.0 { d / (d + DEPTH_SCALE) } else { 0.0 };
        f[9 * n + p] = la;
        f[10 * n + p] = kosm_render::neural::id_feature(id[p]);
    }
    f
}

pub fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// The demodulated illumination the predicted kernel averages, 3 planes.
pub fn illumination(n: usize, mean: &[f32], albedo: &[f32]) -> Vec<f32> {
    let mut v = vec![0.0f32; 3 * n];
    for p in 0..n {
        for c in 0..3 {
            v[c * n + p] = mean[p * 3 + c] / albedo[p * 3 + c].max(DEMOD_FLOOR);
        }
    }
    v
}

/// One convolution layer: `[out, in, KS, KS]` weights and `[out]` biases,
/// flat and row-major, in the order the weight file stores them.
pub struct Conv {
    pub c_in: usize,
    pub c_out: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Conv {
    /// Kaiming-uniform init: the variance a ReLU stack needs to neither
    /// vanish nor explode through three layers.
    pub fn new(c_in: usize, c_out: usize, seed: u64) -> Self {
        let fan_in = c_in * KS * KS;
        let bound = (6.0 / fan_in as f32).sqrt();
        let mut s = seed | 1;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64 / (1u64 << 53) as f64) as f32;
            (u * 2.0 - 1.0) * bound
        };
        let weight: Vec<f32> = (0..c_out * fan_in).map(|_| next()).collect();
        Self { c_in, c_out, weight, bias: vec![0.0f32; c_out] }
    }

    fn w(&self) -> &[f32] {
        &self.weight
    }
    fn b(&self) -> &[f32] {
        &self.bias
    }
}

/// Clamp-to-edge index, the border rule both forwards use.
#[inline]
fn clamp_i(v: i32, hi: usize) -> usize {
    v.clamp(0, hi as i32 - 1) as usize
}

/// `out[o][p] = b[o] + Σ_i Σ_taps W[o][i][t] · a[i][p+t]`, planar and
/// clamp-padded.
fn conv_forward(conv: &Conv, a: &[f32], s: usize, out: &mut [f32]) {
    let (ci, co) = (conv.c_in, conv.c_out);
    let (w, b) = (conv.w(), conv.b());
    let n = s * s;
    for o in 0..co {
        let dst = &mut out[o * n..(o + 1) * n];
        dst.fill(b[o]);
        for i in 0..ci {
            let src = &a[i * n..(i + 1) * n];
            let wk = &w[(o * ci + i) * KS * KS..(o * ci + i + 1) * KS * KS];
            for dy in 0..KS {
                for dx in 0..KS {
                    let k = wk[dy * KS + dx];
                    if k == 0.0 {
                        continue;
                    }
                    let (oy, ox) = (dy as i32 - 1, dx as i32 - 1);
                    for y in 0..s {
                        let sy = clamp_i(y as i32 + oy, s) * s;
                        let dy0 = y * s;
                        for x in 0..s {
                            dst[dy0 + x] += k * src[sy + clamp_i(x as i32 + ox, s)];
                        }
                    }
                }
            }
        }
    }
}

/// The kernel-predicting network.
pub struct Kpn {
    pub hidden: usize,
    pub l1: Conv,
    pub l2: Conv,
    pub l3: Conv,
    /// The firefly veto's three raw scalars, trained with everything else.
    ///
    /// See `kosm_render::neural::Veto` for what they mean and why the filter
    /// needs them: a softmax has only positive weights, so on the backboard's
    /// glass — where the albedo is at the demodulation floor and one stray
    /// path comes back a hundred times too bright — it can shrink a firefly's
    /// share but never refuse it. The gate is a term the softmax lacked and
    /// the clamp is a `min` the network cannot learn its way around.
    ///
    /// Three scalars against twenty thousand weights, and they are the whole
    /// difference on the glass and the net.
    pub veto: [f32; VETO],
}

/// What the reference forward hands back: the applied kernel, the clamp, the
/// answer. Nothing a backward pass would need, because there is none here.
pub struct Activations {
    pub s: usize,
    /// The kernel actually applied, `[K][n]`: softmax times gate, normalised.
    pub w: Vec<f32>,
    /// Each tap's firefly clamp, `[K][n]`.
    pub clip: Vec<f32>,
    /// The filtered illumination, `[3][n]`.
    pub out: Vec<f32>,
}

impl Kpn {
    pub fn new(hidden: usize, seed: u64) -> Self {
        Self {
            hidden,
            l1: Conv::new(C_IN, hidden, seed ^ 0x1111),
            l2: Conv::new(hidden, hidden, seed ^ 0x2222),
            l3: Conv::new(hidden, K, seed ^ 0x3333),
            veto: kosm_render::neural::Veto::initial_raw(),
        }
    }

    /// The veto as the filter reads it, through its activations.
    pub fn veto_now(&self) -> kosm_render::neural::Veto {
        kosm_render::neural::Veto::from_raw(self.veto)
    }

    /// Trainable scalars.
    pub fn parameters(&self) -> usize {
        [&self.l1, &self.l2, &self.l3].iter().map(|c| c.weight.len() + c.bias.len()).sum::<usize>()
            + VETO
    }

    /// Predict a kernel for every pixel and apply it to `illum`.
    ///
    /// `feat` is `[C_IN][n]` from [`features`], `illum` is `[3][n]` from
    /// [`illumination`], both planar; the result's `out` is the filtered
    /// illumination, still demodulated.
    pub fn forward(&self, feat: &[f32], illum: &[f32], s: usize) -> Activations {
        let n = s * s;
        let mut z1 = vec![0.0f32; self.hidden * n];
        conv_forward(&self.l1, feat, s, &mut z1);
        for v in z1.iter_mut() {
            *v = v.max(0.0);
        }
        let mut z2 = vec![0.0f32; self.hidden * n];
        conv_forward(&self.l2, &z1, s, &mut z2);
        for v in z2.iter_mut() {
            *v = v.max(0.0);
        }
        let mut z3 = vec![0.0f32; K * n];
        conv_forward(&self.l3, &z2, s, &mut z3);

        // The neighbourhood every kernel averages, and its luminances: the
        // veto has to see the whole 5x5 before it can judge any tap of it.
        let (mut lum, mut mu) = (vec![0.0f32; K * n], vec![0.0f32; K * n]);
        let r = (TAPS / 2) as i32;
        for ky in 0..TAPS {
            for kx in 0..TAPS {
                let k = ky * TAPS + kx;
                let (oy, ox) = (ky as i32 - r, kx as i32 - r);
                for y in 0..s {
                    let sy = clamp_i(y as i32 + oy, s) * s;
                    for x in 0..s {
                        let sp = sy + clamp_i(x as i32 + ox, s);
                        let l = luminance([illum[sp], illum[n + sp], illum[2 * n + sp]]);
                        lum[k * n + y * s + x] = l.max(0.0);
                    }
                }
            }
        }
        for p in 0..n {
            let mut total = 0.0;
            for k in 0..K {
                total += lum[k * n + p];
            }
            let inv = 1.0 / (K - 1) as f32;
            for k in 0..K {
                mu[k * n + p] = ((total - lum[k * n + p]) * inv).max(0.0);
            }
        }

        // softmax over the 25 taps, per pixel, times the veto's gate
        let veto = self.veto_now();
        let mut smax = vec![0.0f32; K * n];
        let mut clip = vec![0.0f32; K * n];
        let mut w = vec![0.0f32; K * n];
        for p in 0..n {
            let mut m = f32::NEG_INFINITY;
            for k in 0..K {
                m = m.max(z3[k * n + p]);
            }
            let mut sm = 0.0;
            for k in 0..K {
                let e = (z3[k * n + p] - m).exp();
                smax[k * n + p] = e;
                sm += e;
            }
            let inv = 1.0 / sm;
            let mut v = 0.0;
            for k in 0..K {
                let sk = smax[k * n + p] * inv;
                smax[k * n + p] = sk;
                let (g, c) = veto.tap(lum[k * n + p], mu[k * n + p]);
                clip[k * n + p] = c;
                w[k * n + p] = sk * g;
                v += sk * g;
            }
            let vinv = 1.0 / v.max(1e-20);
            for k in 0..K {
                w[k * n + p] *= vinv;
            }
        }

        let mut out = vec![0.0f32; 3 * n];
        apply_clamped(&w, &clip, illum, s, &mut out);
        Activations { s, w, clip, out }
    }

    /// The whole filter, end to end, from a frame's planes — the reference
    /// the WGSL pass is checked against.
    ///
    /// Interleaved in, interleaved out, radiance in and radiance out: the
    /// demodulation and remodulation are inside, exactly as the shader has
    /// them inside.
    #[allow(clippy::too_many_arguments)]
    pub fn filter(
        &self,
        w: usize,
        h: usize,
        mean: &[f32],
        variance: &[f32],
        normal: &[f32],
        depth: &[f32],
        albedo: &[f32],
        id: &[f32],
        count: &[f32],
    ) -> Vec<f32> {
        let n = w * h;
        assert_eq!(w, h, "the reference forward is square-tiled; see filter_rect");
        let feat = features(n, mean, variance, normal, depth, albedo, id, count);
        let illum = illumination(n, mean, albedo);
        let act = self.forward(&feat, &illum, w);
        let mut out = vec![0.0f32; n * 3];
        for p in 0..n {
            for c in 0..3 {
                out[p * 3 + c] = act.out[c * n + p] * albedo[p * 3 + c].max(DEMOD_FLOOR);
            }
        }
        out
    }

    /// Every parameter, flat, in the order [`save`](Kpn::save) writes them.
    pub fn flat(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.parameters());
        for c in [&self.l1, &self.l2, &self.l3] {
            v.extend_from_slice(c.w());
            v.extend_from_slice(c.b());
        }
        v.extend_from_slice(&self.veto);
        v
    }

    /// Write the weights in the layout `kosm_render::neural::Weights` reads:
    /// an eight-byte magic, four `u32` of shape, then every f32 of
    /// `[W1 b1 W2 b2 W3 b3 veto]`.
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(MAGIC)?;
        f.write_all(&(C_IN as u32).to_le_bytes())?;
        f.write_all(&(self.hidden as u32).to_le_bytes())?;
        f.write_all(&(K as u32).to_le_bytes())?;
        f.write_all(&(KS as u32).to_le_bytes())?;
        let mut buf = Vec::with_capacity(self.parameters() * 4);
        for x in self.flat() {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        f.write_all(&buf)?;
        f.flush()
    }

    /// The inverse of [`Kpn::save`].
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut head = [0u8; 24];
        f.read_exact(&mut head)?;
        if &head[0..8] != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a kosm KPN weight file",
            ));
        }
        let hidden = u32::from_le_bytes(head[12..16].try_into().unwrap()) as usize;
        let mut me = Self::new(hidden, 0);
        let mut rest = Vec::new();
        f.read_to_end(&mut rest)?;
        let vals: Vec<f32> = rest
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if vals.len() < me.parameters() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("KPN weight file holds {} values, hidden {hidden} needs {}", vals.len(), me.parameters()),
            ));
        }
        let n = me.parameters();
        me.set_flat(&vals[..n]);
        Ok(me)
    }

    /// The inverse of [`flat`](Kpn::flat): every parameter from one vector in
    /// the weight file's order.
    pub fn set_flat(&mut self, vals: &[f32]) {
        assert_eq!(vals.len(), self.parameters(), "Kpn::set_flat: length");
        let mut at = 0;
        for c in [&mut self.l1, &mut self.l2, &mut self.l3] {
            let (nw, nb) = (c.weight.len(), c.bias.len());
            c.weight.copy_from_slice(&vals[at..at + nw]);
            at += nw;
            c.bias.copy_from_slice(&vals[at..at + nb]);
            at += nb;
        }
        self.veto.copy_from_slice(&vals[at..at + VETO]);
    }
}

/// Apply a per-pixel 5x5 kernel to planar illumination.
pub fn apply(w: &[f32], illum: &[f32], s: usize, out: &mut [f32]) {
    let n = s * s;
    let r = (TAPS / 2) as i32;
    out.fill(0.0);
    for ky in 0..TAPS {
        for kx in 0..TAPS {
            let k = ky * TAPS + kx;
            let (oy, ox) = (ky as i32 - r, kx as i32 - r);
            let wk = &w[k * n..(k + 1) * n];
            for c in 0..3 {
                let src = &illum[c * n..(c + 1) * n];
                let dst = &mut out[c * n..(c + 1) * n];
                for y in 0..s {
                    let sy = clamp_i(y as i32 + oy, s) * s;
                    for x in 0..s {
                        dst[y * s + x] += wk[y * s + x] * src[sy + clamp_i(x as i32 + ox, s)];
                    }
                }
            }
        }
    }
}

/// [`apply`], with each tap scaled by the firefly veto's clamp first.
///
/// `clip` is `[K][n]` and multiplies the *tap*, not the weight — the weights
/// still sum to one, and what changes is how bright the light they average
/// was allowed to be. Nothing puts the clamped energy back: a firefly is a
/// sampling artefact and the converged reference has no such pixel.
pub fn apply_clamped(w: &[f32], clip: &[f32], illum: &[f32], s: usize, out: &mut [f32]) {
    let n = s * s;
    let r = (TAPS / 2) as i32;
    out.fill(0.0);
    for ky in 0..TAPS {
        for kx in 0..TAPS {
            let k = ky * TAPS + kx;
            let (oy, ox) = (ky as i32 - r, kx as i32 - r);
            let wk = &w[k * n..(k + 1) * n];
            let ck = &clip[k * n..(k + 1) * n];
            for c in 0..3 {
                let src = &illum[c * n..(c + 1) * n];
                let dst = &mut out[c * n..(c + 1) * n];
                for y in 0..s {
                    let sy = clamp_i(y as i32 + oy, s) * s;
                    for x in 0..s {
                        let p = y * s + x;
                        dst[p] += wk[p] * ck[p] * src[sy + clamp_i(x as i32 + ox, s)];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The network as a graph
// ---------------------------------------------------------------------------

/// `ln(1 + eˣ)`, written so that neither branch of it overflows: the same
/// function as `kosm_render::neural::softplus` to rounding, generic so the
/// graph's body and an `f32` test can both call it.
pub fn softplus<T: Scalar>(x: T) -> T {
    x.max(T::from_f64(0.0)) + (T::from_f64(1.0) + (-x.abs()).exp()).ln()
}

/// `1 / (1 + eᵘ)`, the gate's falling logistic, with the exponential taken
/// on whichever side of zero keeps it at most one.
///
/// Not a nicety. The gate's argument is `scale · ln(l / μ)`, and on a
/// firefly over a black background that ratio is a million: `eᵘ` overflows
/// `f32`, and while `1/(1+∞)` is a perfectly good zero, its derivative is
/// `∞/∞`. The hand backward dodged that by writing the derivative as
/// `σ(1−σ)`; a symbolic one differentiates what it is given, so it is given
/// a form with nothing infinite in the branch the `select` keeps.
pub fn falling_logistic<T: Scalar>(u: T) -> T {
    let one = T::from_f64(1.0);
    let e = (-u.abs()).exp();
    T::select(u, e / (one + e), one / (one + e))
}

/// [`kosm_render::neural::Veto::tap`] over any [`Scalar`], from the three raw
/// parameters rather than a decoded `Veto`, so that the graph's gradient
/// reaches the raws through `from_raw`'s softplus.
///
/// `kosm-render` is a leaf crate that knows `tang` and nothing of the IR, and
/// its `tap` is `f32`-only; this is the same function written once more, and
/// the test `the_graph_veto_is_the_renderer_veto` holds the two together.
pub fn veto_tap<T: Scalar>(raw: [T; VETO], l: T, mu: T) -> (T, T) {
    let c = T::from_f64;
    let scale = softplus(raw[0]);
    let thresh = raw[1];
    let cap = c(1.0) + softplus(raw[2]);
    let eps = c(VETO_EPS as f64);
    let t = ((l + eps) / (mu + eps)).ln();
    let s = falling_logistic(scale * (t - thresh));
    let gate = c(VETO_FLOOR as f64) + c(1.0 - VETO_FLOOR as f64) * s;
    let clamp = ((cap * mu + eps) / (l + eps)).min(c(1.0));
    (gate, clamp)
}

/// The parameters as graph inputs, declared in the weight file's order so
/// that input `k` of a program is block `k` of [`Kpn::flat`].
///
/// `w3` is declared `[TAPS, TAPS, hidden, KS, KS]` and `b3` `[TAPS, TAPS]`:
/// row-major, that is exactly `[K, hidden, KS, KS]` and `[K]`, with the tap
/// index split into the row and column a view needs to address the
/// neighbour each weight is for. `k = ky·5 + kx` is not affine in `k`; the
/// pair is.
#[derive(Clone, Copy, Debug)]
pub struct ParamInputs {
    pub w1: TensorId,
    pub b1: TensorId,
    pub w2: TensorId,
    pub b2: TensorId,
    pub w3: TensorId,
    pub b3: TensorId,
    pub veto: TensorId,
}

impl ParamInputs {
    /// Declare the seven parameter inputs, which must be a graph's first.
    pub fn declare(g: &mut TensorGraph, hidden: usize) -> Self {
        Self {
            w1: g.input(&[hidden, C_IN, KS, KS]),
            b1: g.input(&[hidden]),
            w2: g.input(&[hidden, hidden, KS, KS]),
            b2: g.input(&[hidden]),
            w3: g.input(&[TAPS, TAPS, hidden, KS, KS]),
            b3: g.input(&[TAPS, TAPS]),
            veto: g.input(&[VETO]),
        }
    }

    pub fn all(&self) -> [TensorId; 7] {
        [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.veto]
    }
}

impl Kpn {
    /// The seven parameter blocks, in [`ParamInputs`]' (and the file's) order.
    pub fn blocks(&self) -> [&[f32]; 7] {
        [self.l1.w(), self.l1.b(), self.l2.w(), self.l2.b(), self.l3.w(), self.l3.b(), &self.veto]
    }
}

fn dims(g: &TensorGraph, x: TensorId) -> Vec<usize> {
    g.shape(x).iter().map(|d| d.n()).collect()
}

/// Clamp-to-edge padding of one axis by `r` on each side, from affine pieces.
///
/// Padded coordinate `t` reads `x[clamp(t − r)]`. Split by where `t` is:
///
/// * the interior, `r ≤ t < s + r`, is `x[t − r]` — a guarded read, which
///   is zero everywhere else;
/// * each low border position `j < r` reads `x[0]`, which is `x[t − j]` at
///   `t = j`;
/// * each high one `j ≥ s + r` reads `x[s − 1]`, which is `x[t − (j − s + 1)]`
///   at `t = j`.
///
/// So the padded value is the interior read plus, for every border `j`,
/// `[t = j] · x[t − oⱼ]`, the indicator an `iota` select. Each piece's view
/// is a unit coefficient and an offset under a guard, and those `backward`
/// inverts by solving for the coordinate rather than by scattering.
fn pad_axis(g: &mut TensorGraph, x: TensorId, axis: usize, r: usize) -> TensorId {
    let src = dims(g, x);
    let s = src[axis];
    let mut out = src.clone();
    out[axis] = s + 2 * r;
    let rank = out.len();
    let t = g.iota(&out, axis);
    let read = |o: usize| Arg {
        src: x,
        view: View {
            guard: true,
            forms: (0..rank)
                .map(|d| {
                    let mut f = Form::unit(rank, d);
                    if d == axis {
                        f.offset = -(o as i64);
                    }
                    f
                })
                .collect(),
        },
    };
    let mut args = vec![Arg { src: t, view: View::identity(rank, rank) }, read(r)];
    let mut at = Vec::new();
    for j in (0..r).chain(s + r..s + 2 * r) {
        args.push(read(if j < r { j } else { j + 1 - s }));
        at.push(j as f64);
    }
    g.comp(&out, &[] as &[usize], args, None, |v| {
        let mut acc = v[1];
        for (k, &j) in at.iter().enumerate() {
            let hit = lit(0.5) - (v[0] - lit(j)).abs();
            acc = acc + ExprId::select(hit, v[2 + k], lit(0.0));
        }
        vec![acc]
    })
    .expect("pad_axis")
}

/// Clamp-to-edge padding of the last two axes of `x` by `r`: `[…, s, s]` in,
/// `[…, s + 2r, s + 2r]` out, and `out[…, y, x] = x[…, clamp_i(y − r),
/// clamp_i(x − r)]` exactly — the same reads `clamp_i` makes in the
/// reference forward. Columns first, then rows of the column-padded tensor.
pub fn pad_clamp(g: &mut TensorGraph, x: TensorId, r: usize) -> TensorId {
    let rank = g.shape(x).len();
    let cols = pad_axis(g, x, rank - 1, r);
    pad_axis(g, cols, rank - 2, r)
}

/// A 3x3 convolution of an already-padded `[c_in, s+2, s+2]` input, plus
/// bias, through a ReLU when asked. `oc` is the output's channel dims — one
/// for a hidden layer, `[TAPS, TAPS]` for the kernel head.
///
/// The padded input is read through a guarded view although every read is
/// in range: a guard is what lets `backward` invert `y + ky` by eliminating
/// `y`, and it keeps the padded tensor materialised rather than spliced into
/// nine reads per tap.
#[allow(clippy::too_many_arguments)]
fn conv(
    g: &mut TensorGraph,
    w: TensorId,
    b: TensorId,
    xp: TensorId,
    oc: &[usize],
    c_in: usize,
    s: usize,
    relu: bool,
) -> TensorId {
    let no = oc.len();
    let mut out = oc.to_vec();
    out.extend([s, s]);
    let rank = no + 5;
    let (y, x, i, ky, kx) = (no, no + 1, no + 2, no + 3, no + 4);
    let mut wp: Vec<Option<usize>> = (0..no).map(Some).collect();
    wp.extend([Some(i), Some(ky), Some(kx)]);
    let mut fy = Form::unit(rank, y);
    fy.coefs[ky] = 1;
    let mut fx = Form::unit(rank, x);
    fx.coefs[kx] = 1;
    let xv = View { guard: true, forms: vec![Form::unit(rank, i), fy, fx] };
    let z = g
        .comp(&out, &[c_in, KS, KS], vec![pick(w, rank, &wp), Arg { src: xp, view: xv }], Some(Monoid::sum()), |v| {
            vec![v[0] * v[1]]
        })
        .expect("conv");
    let r = no + 2;
    let bp: Vec<Option<usize>> = (0..no).map(Some).collect();
    g.comp(&out, &[] as &[usize], vec![crate::ir::id(z, r, r), pick(b, r, &bp)], None, |v| {
        let a = v[0] + v[1];
        vec![if relu { a.max(lit(0.0)) } else { a }]
    })
    .expect("bias")
}

/// A read of a `[…, s+2r, s+2r]` padded tensor at tap `(ky, kx)` of point
/// `(y, x)`: forms `y + ky` and `x + kx` on its last two axes, `lead` on the
/// rest.
fn tap_view(rank: usize, lead: Vec<Form>, (ky, kx): (usize, usize), (y, x): (usize, usize)) -> View {
    let mut fy = Form::unit(rank, y);
    fy.coefs[ky] = 1;
    let mut fx = Form::unit(rank, x);
    fx.coefs[kx] = 1;
    let mut forms = lead;
    forms.extend([fy, fx]);
    View { guard: false, forms }
}

/// [`Kpn::forward`] as a graph: `feat` `[C_IN, s, s]` and `illum` `[3, s, s]`
/// in, the filtered illumination `[3, s, s]` out, every step of it the
/// reference's step — the same clamp-padded convolutions, the same 5x5
/// luminance gather and leave-one-out mean, the same softmax, veto and
/// renormalisation, the same clamped apply.
pub fn forward_graph(
    g: &mut TensorGraph,
    p: &ParamInputs,
    feat: TensorId,
    illum: TensorId,
    hidden: usize,
    s: usize,
) -> TensorId {
    let fp = pad_clamp(g, feat, 1);
    let a1 = conv(g, p.w1, p.b1, fp, &[hidden], C_IN, s, true);
    let a1p = pad_clamp(g, a1, 1);
    let a2 = conv(g, p.w2, p.b2, a1p, &[hidden], hidden, s, true);
    let a2p = pad_clamp(g, a2, 1);
    // [ky, kx, y, x]
    let z3 = conv(g, p.w3, p.b3, a2p, &[TAPS, TAPS], hidden, s, false);

    // The neighbourhood, padded once by the filter's radius; its luminance
    // is taken on the padded grid, so a tap's luminance is a view of it.
    let r = TAPS / 2;
    let ip = pad_clamp(g, illum, r);
    let sp = s + 2 * r;
    let chan = |c: i64| Arg {
        src: ip,
        view: View { guard: false, forms: vec![Form::constant(2, c), Form::unit(2, 0), Form::unit(2, 1)] },
    };
    let lum = g
        .comp(&[sp, sp], &[] as &[usize], vec![chan(0), chan(1), chan(2)], None, |v| {
            vec![(lit(0.2126) * v[0] + lit(0.7152) * v[1] + lit(0.0722) * v[2]).max(lit(0.0))]
        })
        .expect("lum");

    // Σ over the 25 taps, per pixel: space (y, x, ky, kx).
    let total = g
        .comp(&[s, s], &[TAPS, TAPS], vec![Arg { src: lum, view: tap_view(4, vec![], (2, 3), (0, 1)) }], Some(Monoid::sum()), |v| {
            vec![v[0]]
        })
        .expect("total");

    // Per tap, space (ky, kx, y, x): the gate and the clamp.
    let veto_tap_comp = |g: &mut TensorGraph, which: usize| {
        let raw = |k: i64| Arg { src: p.veto, view: View { guard: false, forms: vec![Form::constant(4, k)] } };
        let args = vec![
            Arg { src: lum, view: tap_view(4, vec![], (0, 1), (2, 3)) },
            pick(total, 4, &[Some(2), Some(3)]),
            raw(0),
            raw(1),
            raw(2),
        ];
        g.comp(&[TAPS, TAPS, s, s], &[] as &[usize], args, None, |v| {
            let l = v[0];
            let mu = ((v[1] - l) * lit(1.0 / (K - 1) as f64)).max(lit(0.0));
            let (gate, clip) = veto_tap([v[2], v[3], v[4]], l, mu);
            vec![if which == 0 { gate } else { clip }]
        })
        .expect("veto")
    };
    let gate = veto_tap_comp(g, 0);
    let clip = veto_tap_comp(g, 1);

    // Softmax over the taps, per pixel, times the gate; space (y, x, ky, kx)
    // for the reductions.
    let per_pixel = |g: &mut TensorGraph, x: TensorId, m: Monoid| {
        g.comp(&[s, s], &[TAPS, TAPS], vec![pick(x, 4, &[Some(2), Some(3), Some(0), Some(1)])], Some(m), |v| vec![v[0]])
            .expect("per pixel")
    };
    let m = per_pixel(g, z3, Monoid::max());
    let e = g
        .comp(&[TAPS, TAPS, s, s], &[] as &[usize], vec![crate::ir::id(z3, 4, 4), pick(m, 4, &[Some(2), Some(3)])], None, |v| {
            vec![(v[0] - v[1]).exp()]
        })
        .expect("exp");
    let sum = per_pixel(g, e, Monoid::sum());
    let gated = g
        .comp(
            &[TAPS, TAPS, s, s],
            &[] as &[usize],
            vec![crate::ir::id(e, 4, 4), pick(sum, 4, &[Some(2), Some(3)]), crate::ir::id(gate, 4, 4)],
            None,
            |v| vec![v[0] * v[1].recip() * v[2]],
        )
        .expect("gated");
    let vsum = per_pixel(g, gated, Monoid::sum());

    // The clamped apply: space (c, y, x, ky, kx).
    let at = |x: TensorId| pick(x, 5, &[Some(3), Some(4), Some(1), Some(2)]);
    let args = vec![
        at(gated),
        pick(vsum, 5, &[Some(1), Some(2)]),
        at(clip),
        Arg { src: ip, view: tap_view(5, vec![Form::unit(5, 0)], (3, 4), (1, 2)) },
    ];
    g.comp(&[3, s, s], &[TAPS, TAPS], args, Some(Monoid::sum()), |v| {
        vec![v[0] * v[1].max(lit(1e-20)).recip() * v[2] * v[3]]
    })
    .expect("apply")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy(s: usize) -> (Kpn, Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = s * s;
        let net = Kpn::new(4, 7);
        let mut feat = vec![0.0f32; C_IN * n];
        let mut illum = vec![0.0f32; 3 * n];
        let mut tgt = vec![0.0f32; 3 * n];
        for (i, v) in feat.iter_mut().enumerate() {
            *v = ((i * 37 % 19) as f32 / 19.0) - 0.5;
        }
        for (i, v) in illum.iter_mut().enumerate() {
            *v = ((i * 53 % 23) as f32 / 23.0) + 0.1;
        }
        // A few fireflies, because a neighbourhood with no outlier in it
        // never takes the clamp's active branch and a gradient check that
        // never takes it proves nothing about `cap`.
        for p in (0..n).step_by(7) {
            for c in 0..3 {
                illum[c * n + p] *= 40.0;
            }
        }
        for (i, v) in tgt.iter_mut().enumerate() {
            *v = ((i * 11 % 17) as f32 / 17.0) + 0.1;
        }
        (net, feat, illum, tgt)
    }

    /// A random linear projection of the output.
    ///
    /// Linear on purpose. A sum of squares is a perfectly good loss and a
    /// terrible thing to finite-difference in `f32`: it is order 10 while the
    /// gradient of any one of four thousand weights is order 1e-3, so
    /// `(L(w+ε) − L(w−ε))` is two nearly equal numbers subtracted and the
    /// answer is rounding. A random projection has the same Jacobian coverage
    /// — it checks `vᵀ ∂out/∂w` for a `v` that touches every output — and
    /// stays the same size as its own derivative.
    fn projection(out: &[f32], tgt: &[f32]) -> f64 {
        // `tgt` doubles as the projection direction, recentred on zero
        out.iter().zip(tgt).map(|(o, t)| ((t - 0.5) * o) as f64).sum()
    }

    /// [`forward_graph`] under that projection, and its gradient with respect
    /// to every parameter block — `backward`'s — read off the interpreter.
    /// Inputs: the seven blocks, `feat`, `illum`, the projection; outputs:
    /// the projection, the seven gradients, the filtered illumination.
    fn projected_graph(hidden: usize, s: usize) -> (TensorGraph, Vec<TensorId>) {
        let mut g = TensorGraph::new();
        let p = ParamInputs::declare(&mut g, hidden);
        let feat = g.input(&[C_IN, s, s]);
        let illum = g.input(&[3, s, s]);
        let dir = g.input(&[3, s, s]);
        let out = forward_graph(&mut g, &p, feat, illum, hidden, s);
        let at = |x| pick(x, 4, &[Some(1), Some(2), Some(3)]);
        let y = g
            .comp(&[1usize], &[3, s, s], vec![at(out), at(dir)], Some(Monoid::sum()), |v| vec![v[0] * v[1]])
            .unwrap();
        let mut outs = vec![y];
        outs.extend(g.backward(y, &p.all()).unwrap());
        outs.push(out);
        crate::ir::optimize(&g, &outs)
    }

    fn run_projected(net: &Kpn, feat: &[f32], illum: &[f32], tgt: &[f32], s: usize) -> Vec<Vec<f64>> {
        let (g, outs) = projected_graph(net.hidden, s);
        let wide = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        let dir: Vec<f32> = tgt.iter().map(|t| t - 0.5).collect();
        let mut inputs: Vec<Vec<f64>> = net.blocks().iter().map(|b| wide(b)).collect();
        inputs.extend([wide(feat), wide(illum), wide(&dir)]);
        let refs: Vec<&[f64]> = inputs.iter().map(|v| v.as_slice()).collect();
        let mut p = crate::ir::Program::interpreted(&g, &outs, &refs);
        p.run();
        (0..outs.len()).map(|i| p.read_f64(i)).collect()
    }

    /// The graph's forward is the reference forward: same kernel, same veto,
    /// same clamp-padded reads, to `f32` rounding (the interpreter is f64).
    #[test]
    fn the_graph_forward_is_the_reference_forward() {
        let s = 7;
        let (net, feat, illum, tgt) = toy(s);
        let mine = net.forward(&feat, &illum, s).out;
        let r = run_projected(&net, &feat, &illum, &tgt, s);
        let theirs = r.last().unwrap();
        for (i, (a, b)) in mine.iter().zip(theirs).enumerate() {
            let scale = (*a as f64).abs().max(b.abs()).max(1e-3);
            assert!((*a as f64 - b).abs() / scale < 2e-5, "element {i}: reference {a}, graph {b}");
        }
    }

    #[test]
    fn the_graph_gradient_agrees_with_a_finite_difference() {
        let s = 6;
        let (mut net, feat, illum, tgt) = toy(s);
        // The gradient is the graph's, from the interpreter; the secant is
        // the reference forward's, in `f32`. Two implementations, so this
        // checks the derivative and that it is a derivative of the right
        // function.
        let grads = run_projected(&net, &feat, &illum, &tgt, s);

        // A *directional* derivative per layer rather than one weight at a
        // time. Any single weight of a 3x3 convolution moves this loss by
        // about 1e-4, which in `f32` is only a decade above what the
        // difference of two forward passes can resolve; the same check along
        // a random direction through all of a layer's weights adds the
        // signal of every one of them and leaves the noise where it was.
        //
        // `eps` is the middle of the window this check has: at 1e-2 the walk
        // flips enough ReLU signs that the secant stops being the tangent,
        // and at 1e-5 two `f32` forward passes no longer differ by anything
        // but rounding. Both ends were measured; 1e-3 agrees to under 2%.
        let eps = 3e-3f32;
        for li in 0..3 {
            let n_w = [&net.l1, &net.l2, &net.l3][li].weight.len();
            let dir: Vec<f32> = (0..n_w)
                .map(|i| if (i * 2654435761 >> 7) & 1 == 0 { 1.0 } else { -1.0 })
                .collect();
            // outputs: the projection, then w1 b1 w2 b2 w3 b3 veto
            let gw = &grads[1 + 2 * li];
            let ana: f64 = gw.iter().zip(&dir).map(|(g, d)| g * *d as f64).sum();

            fn walk(net: &mut Kpn, li: usize, dir: &[f32], k: f32) {
                let conv = match li {
                    0 => &mut net.l1,
                    1 => &mut net.l2,
                    _ => &mut net.l3,
                };
                for (w, d) in conv.weight.iter_mut().zip(dir) {
                    *w += k * d;
                }
            }
            walk(&mut net, li, &dir, eps);
            let lp = projection(&net.forward(&feat, &illum, s).out, &tgt);
            walk(&mut net, li, &dir, -2.0 * eps);
            let lm = projection(&net.forward(&feat, &illum, s).out, &tgt);
            walk(&mut net, li, &dir, eps);

            let num = (lp - lm) / (2.0 * eps as f64);
            agrees(&format!("layer {li}"), ana, num);
        }

        // The veto's three scalars, one at a time — there are only three of
        // them, so a direction through them would only hide which one is
        // wrong. `cap` is checked here and not merely present: `toy` plants
        // fireflies so the clamp's `min` has an active branch to be
        // differentiated on.
        for (vi, name) in ["scale", "thresh", "cap"].iter().enumerate() {
            let ana = grads[7][vi];
            net.veto[vi] += eps;
            let lp = projection(&net.forward(&feat, &illum, s).out, &tgt);
            net.veto[vi] -= 2.0 * eps;
            let lm = projection(&net.forward(&feat, &illum, s).out, &tgt);
            net.veto[vi] += eps;
            agrees(name, ana, (lp - lm) / (2.0 * eps as f64));
        }
    }

    /// A finite difference and an analytic gradient, to a relative tolerance
    /// with an absolute floor under it.
    ///
    /// The floor is the point. A secant of two `f32` forward passes of a loss
    /// of order 0.1 is good to about 1e-4 in absolute terms however small the
    /// derivative it is estimating, and the second layer's directional
    /// gradient here is 2e-3 — nearly cancelled, because the direction is a
    /// coin flip per weight and that layer's weights disagree. Judging that
    /// one by a purely relative tolerance measures the rounding, not the
    /// backward pass; the sweep across `eps` that says so is the giveaway,
    /// because its error *falls* as the step grows.
    fn agrees(what: &str, ana: f64, num: f64) {
        let scale = num.abs().max(ana.abs()).max(1e-6);
        assert!(
            (num - ana).abs() < 2e-2 * scale + 2e-4,
            "{what}: analytic {ana}, numeric {num}"
        );
        assert!(ana.abs() > 1e-5, "{what} gradient is degenerate: {ana}");
    }

    /// The biases too, which the per-layer walk above leaves alone: one
    /// direction through all three bias vectors at once.
    #[test]
    fn the_bias_gradients_agree_with_a_finite_difference() {
        let s = 6;
        let (mut net, feat, illum, tgt) = toy(s);
        let grads = run_projected(&net, &feat, &illum, &tgt, s);
        let sign = |i: usize| if (i * 40503 >> 5) & 1 == 0 { 1.0f32 } else { -1.0 };
        let ana: f64 = (0..3)
            .map(|li| grads[2 + 2 * li].iter().enumerate().map(|(i, g)| g * sign(i + 97 * li) as f64).sum::<f64>())
            .sum();
        let walk = |net: &mut Kpn, k: f32| {
            for (li, c) in [&mut net.l1, &mut net.l2, &mut net.l3].into_iter().enumerate() {
                for (i, b) in c.bias.iter_mut().enumerate() {
                    *b += k * sign(i + 97 * li);
                }
            }
        };
        let eps = 3e-3f32;
        walk(&mut net, eps);
        let lp = projection(&net.forward(&feat, &illum, s).out, &tgt);
        walk(&mut net, -2.0 * eps);
        let lm = projection(&net.forward(&feat, &illum, s).out, &tgt);
        agrees("biases", ana, (lp - lm) / (2.0 * eps as f64));
    }

    /// Clamp-to-edge built from affine pieces reads exactly what `clamp_i`
    /// reads, bit for bit, under the interpreter and natively — including a
    /// side shorter than twice the radius, where the borders meet.
    #[test]
    fn affine_clamp_padding_is_clamp_to_edge() {
        for (c, s, r) in [(2usize, 5usize, 1usize), (3, 6, 2), (1, 3, 2), (2, 1, 1)] {
            let mut g = TensorGraph::new();
            let x = g.input(&[c, s, s]);
            let y = pad_clamp(&mut g, x, r);
            let (g, outs) = crate::ir::optimize(&g, &[y]);
            let data: Vec<f32> = (0..c * s * s).map(|i| (i as f32 * 0.37).sin() * 3.0 + i as f32).collect();
            let sp = s + 2 * r;
            let mut want = vec![0.0f32; c * sp * sp];
            for ch in 0..c {
                for py in 0..sp {
                    for px in 0..sp {
                        let (sy, sx) = (clamp_i(py as i32 - r as i32, s), clamp_i(px as i32 - r as i32, s));
                        want[(ch * sp + py) * sp + px] = data[(ch * s + sy) * s + sx];
                    }
                }
            }
            let wide: Vec<f64> = data.iter().map(|&v| v as f64).collect();
            let mut interp = crate::ir::Program::interpreted(&g, &outs, &[&wide]);
            interp.run();
            let mut native = crate::ir::Program::compile(&g, &outs, &[&data]);
            native.run();
            for (i, w) in want.iter().enumerate() {
                assert_eq!(interp.read_f64(0)[i], *w as f64, "interp ({c},{s},{r}) at {i}");
                assert_eq!(native.read(0)[i], *w, "native ({c},{s},{r}) at {i}");
            }
        }
    }

    /// The veto the graph differentiates is the renderer's veto: the same
    /// gate and clamp, from the same raws, across ordinary pixels, fireflies
    /// over black, and taps darker than their neighbourhood.
    #[test]
    fn the_graph_veto_is_the_renderer_veto() {
        let raws = [kosm_render::neural::Veto::initial_raw(), [0.2, -0.4, -1.5], [9.0, 2.0, 6.0], [-3.0, 0.0, 0.0]];
        for raw in raws {
            let theirs = kosm_render::neural::Veto::from_raw(raw);
            for &(l, mu) in &[(0.5f32, 0.5f32), (40.0, 0.3), (1e4, 0.0), (0.0, 2.0), (0.01, 0.02), (3.0, 1.0), (1e6, 1e-3)] {
                let (g0, c0) = theirs.tap(l, mu);
                let (g1, c1) = veto_tap(raw, l, mu);
                assert!((g0 - g1).abs() <= 1e-6 + 1e-5 * g0.abs(), "gate {raw:?} ({l},{mu}): {g0} vs {g1}");
                assert!((c0 - c1).abs() <= 1e-6 + 1e-5 * c0.abs(), "clamp {raw:?} ({l},{mu}): {c0} vs {c1}");
            }
        }
    }

    #[test]
    fn the_predicted_kernel_is_a_convex_combination() {
        let s = 5;
        let (net, feat, illum, _) = toy(s);
        let act = net.forward(&feat, &illum, s);
        let n = s * s;
        for p in 0..n {
            let sum: f32 = (0..K).map(|k| act.w[k * n + p]).sum();
            assert!((sum - 1.0).abs() < 1e-4, "kernel at {p} sums to {sum}");
            for k in 0..K {
                assert!(act.w[k * n + p] >= 0.0);
            }
        }
        // and therefore the output is inside the neighbourhood's range
        for c in 0..3 {
            let lo = illum[c * n..(c + 1) * n]
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);
            let hi = illum[c * n..(c + 1) * n]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            for p in 0..n {
                let v = act.out[c * n + p];
                assert!(v >= lo - 1e-5 && v <= hi + 1e-5);
            }
        }
    }

    /// The trainer's forward and the renderer's are the same network.
    ///
    /// They are two implementations by necessity — this one is the
    /// reference the training graph is checked against, and `kosm_render`'s
    /// is a leaf crate that knows nothing about training — and a weight file is only
    /// worth anything if they agree. Anything trained against a forward the
    /// renderer does not share is a picture nobody will ever see.
    #[test]
    fn the_trained_network_is_the_one_the_renderer_runs() {
        let s = 12usize;
        let n = s * s;
        let net = Kpn::new(5, 4242);
        let path = std::env::temp_dir().join("kosm-kpn-cross.bin");
        net.save(&path).unwrap();
        let theirs = kosm_render::neural::Weights::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(theirs.hidden, net.hidden);

        // Interleaved planes with a depth step and a background band, so the
        // guides are doing something rather than being constant.
        let mut mean = vec![0.0f32; n * 3];
        let mut albedo = vec![0.0f32; n * 3];
        let mut normal = vec![0.0f32; n * 3];
        let mut depth = vec![0.0f32; n];
        let mut variance = vec![0.0f32; n];
        for p in 0..n {
            let (x, y) = (p % s, p / s);
            depth[p] = if y == s / 2 {
                0.0
            } else if x < s / 2 {
                800.0
            } else {
                5200.0
            };
            normal[p * 3 + 2] = 1.0;
            variance[p] = 0.001 + 0.04 * ((p * 5) % 9) as f32 / 9.0;
            for c in 0..3 {
                albedo[p * 3 + c] = 0.12 + 0.3 * ((p + c) % 6) as f32 / 6.0;
                mean[p * 3 + c] = 0.03 + 1.4 * ((p * 17 + c * 11) % 23) as f32 / 23.0;
            }
        }

        // A count plane that varies and an id plane with a seam in it, so the
        // two new per-pixel inputs are exercised rather than being constants
        // both implementations happen to agree about.
        let count: Vec<f32> = (0..n).map(|p| 1.0 + (p % 31) as f32).collect();
        let id: Vec<f32> = (0..n)
            .map(|p| if depth[p] > 0.0 { (1 + p % 5) as f32 } else { 0.0 })
            .collect();
        let mine = net.filter(s, s, &mean, &variance, &normal, &depth, &albedo, &id, &count);
        let ours = theirs.forward(
            s, s, &mean, &variance, &normal, &depth, &albedo, &id, &count,
        );
        for i in 0..n * 3 {
            let scale = mine[i].abs().max(ours[i].abs()).max(1e-4);
            assert!(
                (mine[i] - ours[i]).abs() / scale < 1e-4,
                "element {i}: trainer {}, renderer {}",
                mine[i],
                ours[i]
            );
        }
    }

    #[test]
    fn weights_survive_a_round_trip() {
        let net = Kpn::new(6, 11);
        let dir = std::env::temp_dir().join("kosm-kpn-roundtrip.bin");
        net.save(&dir).unwrap();
        let back = Kpn::load(&dir).unwrap();
        assert_eq!(back.hidden, net.hidden);
        assert_eq!(back.flat(), net.flat());
        let _ = std::fs::remove_file(&dir);
    }
}
