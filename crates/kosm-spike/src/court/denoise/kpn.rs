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
//! # Why the layers are ours
//!
//! `tang_train::Conv2d` exists and is correct — it is gradient-checked in
//! that crate's tests — but its forward is `Tensor::from_fn` over a
//! multi-dimensional `get`, which recomputes strides per tap. This network is
//! about 80 million multiply-accumulates per 64x64 tile in the forward pass
//! alone, and at a few thousand tile-passes an epoch that is the difference
//! between a training run and a weekend. So the convolutions here are flat
//! `f32` loops over planar buffers, and what comes from `tang_train` is what
//! it is good at and we have no business rewriting: [`tang_train::Parameter`]
//! holding the weights and their gradients, and [`tang_train::ModuleAdam`]
//! stepping them.
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

use tang_tensor::{Shape, Tensor};
use tang_train::Parameter;

/// Input feature planes per pixel; see [`features`].
pub const C_IN: usize = 10;
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

const MAGIC: &[u8; 8] = b"KOSMKPN1";

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
///
/// The log compression on the two radiance-like planes is what keeps a light
/// panel at a few hundred from dominating a floor at a tenth. The count plane
/// is what lets one network serve every history length: a pixel on its first
/// sample and one on its sixteenth want different kernels and the network has
/// to be told which it is looking at.
pub fn features(
    n: usize,
    mean: &[f32],
    variance: &[f32],
    normal: &[f32],
    depth: &[f32],
    albedo: &[f32],
    count: u32,
) -> Vec<f32> {
    let mut f = vec![0.0f32; C_IN * n];
    let inv_sqrt_n = 1.0 / (count.max(1) as f32).sqrt();
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
        f[3 * n + p] = inv_sqrt_n;
        f[4 * n + p] = (1.0 + (variance[p].max(0.0) / (la * la)).sqrt()).ln();
        for c in 0..3 {
            f[(5 + c) * n + p] = normal[p * 3 + c];
        }
        let d = depth[p];
        f[8 * n + p] = if d > 0.0 { d / (d + DEPTH_SCALE) } else { 0.0 };
        f[9 * n + p] = la;
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
/// held in `tang_train`'s `Parameter` so its optimiser can step them.
pub struct Conv {
    pub c_in: usize,
    pub c_out: usize,
    pub weight: Parameter<f32>,
    pub bias: Parameter<f32>,
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
        let w: Vec<f32> = (0..c_out * fan_in).map(|_| next()).collect();
        Self {
            c_in,
            c_out,
            weight: Parameter::new(Tensor::new(w, Shape::from_slice(&[c_out, c_in, KS, KS]))),
            bias: Parameter::new(Tensor::new(
                vec![0.0f32; c_out],
                Shape::from_slice(&[c_out]),
            )),
        }
    }

    fn w(&self) -> &[f32] {
        self.weight.data.data()
    }
    fn b(&self) -> &[f32] {
        self.bias.data.data()
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

/// Gradients of a [`conv_forward`]: into `gw`/`gb` (accumulated) and, when
/// asked, into `ga`.
fn conv_backward(
    conv: &Conv,
    a: &[f32],
    gz: &[f32],
    s: usize,
    gw: &mut [f32],
    gb: &mut [f32],
    mut ga: Option<&mut [f32]>,
) {
    let (ci, co) = (conv.c_in, conv.c_out);
    let w = conv.w();
    let n = s * s;
    if let Some(g) = ga.as_deref_mut() {
        g.fill(0.0);
    }
    for o in 0..co {
        let gzo = &gz[o * n..(o + 1) * n];
        gb[o] += gzo.iter().sum::<f32>();
        for i in 0..ci {
            let src = &a[i * n..(i + 1) * n];
            let base = (o * ci + i) * KS * KS;
            for dy in 0..KS {
                for dx in 0..KS {
                    let (oy, ox) = (dy as i32 - 1, dx as i32 - 1);
                    let mut acc = 0.0f32;
                    for y in 0..s {
                        let sy = clamp_i(y as i32 + oy, s) * s;
                        let dy0 = y * s;
                        for x in 0..s {
                            acc += gzo[dy0 + x] * src[sy + clamp_i(x as i32 + ox, s)];
                        }
                    }
                    gw[base + dy * KS + dx] += acc;
                }
            }
        }
    }
    let Some(g) = ga else { return };
    for i in 0..ci {
        let gi = &mut g[i * n..(i + 1) * n];
        for o in 0..co {
            let gzo = &gz[o * n..(o + 1) * n];
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
                            // the tap read a[sy + sx]; its gradient lands there
                            gi[sy + clamp_i(x as i32 + ox, s)] += k * gzo[dy0 + x];
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
}

/// Everything a forward pass leaves behind that the backward pass needs.
pub struct Activations {
    pub s: usize,
    pub a1: Vec<f32>,
    pub a2: Vec<f32>,
    /// The softmaxed kernel, `[K][n]`.
    pub w: Vec<f32>,
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
        }
    }

    /// Trainable scalars.
    pub fn parameters(&self) -> usize {
        self.l1.weight.data.numel()
            + self.l1.bias.data.numel()
            + self.l2.weight.data.numel()
            + self.l2.bias.data.numel()
            + self.l3.weight.data.numel()
            + self.l3.bias.data.numel()
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

        // softmax over the 25 taps, per pixel
        let mut w = vec![0.0f32; K * n];
        for p in 0..n {
            let mut m = f32::NEG_INFINITY;
            for k in 0..K {
                m = m.max(z3[k * n + p]);
            }
            let mut sum = 0.0;
            for k in 0..K {
                let e = (z3[k * n + p] - m).exp();
                w[k * n + p] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for k in 0..K {
                w[k * n + p] *= inv;
            }
        }

        let mut out = vec![0.0f32; 3 * n];
        apply(&w, illum, s, &mut out);
        Activations {
            s,
            a1: z1,
            a2: z2,
            w,
            out,
        }
    }

    /// Backpropagate `g_out` (`[3][n]`, the gradient of the loss with respect
    /// to `Activations::out`) into the parameter gradients.
    pub fn backward(
        &self,
        feat: &[f32],
        illum: &[f32],
        act: &Activations,
        g_out: &[f32],
        grads: &mut Grads,
    ) {
        let s = act.s;
        let n = s * s;
        let r = (TAPS / 2) as i32;

        // dL/dw[k][p] = Σ_c g_out[c][p] · illum[c][tap k of p]
        let mut gw = vec![0.0f32; K * n];
        for ky in 0..TAPS {
            for kx in 0..TAPS {
                let k = ky * TAPS + kx;
                let (oy, ox) = (ky as i32 - r, kx as i32 - r);
                let gwk = &mut gw[k * n..(k + 1) * n];
                for y in 0..s {
                    let sy = clamp_i(y as i32 + oy, s) * s;
                    for x in 0..s {
                        let sp = sy + clamp_i(x as i32 + ox, s);
                        let p = y * s + x;
                        let mut acc = 0.0;
                        for c in 0..3 {
                            acc += g_out[c * n + p] * illum[c * n + sp];
                        }
                        gwk[p] = acc;
                    }
                }
            }
        }

        // through the softmax: dz[k] = w[k]·(gw[k] − Σ_j w[j]·gw[j])
        let mut gz3 = vec![0.0f32; K * n];
        for p in 0..n {
            let mut dot = 0.0;
            for k in 0..K {
                dot += act.w[k * n + p] * gw[k * n + p];
            }
            for k in 0..K {
                gz3[k * n + p] = act.w[k * n + p] * (gw[k * n + p] - dot);
            }
        }

        let mut ga2 = vec![0.0f32; self.hidden * n];
        conv_backward(
            &self.l3,
            &act.a2,
            &gz3,
            s,
            &mut grads.w3,
            &mut grads.b3,
            Some(&mut ga2),
        );
        for (g, a) in ga2.iter_mut().zip(&act.a2) {
            if *a <= 0.0 {
                *g = 0.0;
            }
        }
        let mut ga1 = vec![0.0f32; self.hidden * n];
        conv_backward(
            &self.l2,
            &act.a1,
            &ga2,
            s,
            &mut grads.w2,
            &mut grads.b2,
            Some(&mut ga1),
        );
        for (g, a) in ga1.iter_mut().zip(&act.a1) {
            if *a <= 0.0 {
                *g = 0.0;
            }
        }
        conv_backward(
            &self.l1,
            feat,
            &ga1,
            s,
            &mut grads.w1,
            &mut grads.b1,
            None,
        );
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
        count: u32,
    ) -> Vec<f32> {
        let n = w * h;
        assert_eq!(w, h, "the reference forward is square-tiled; see filter_rect");
        let feat = features(n, mean, variance, normal, depth, albedo, count);
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
        v
    }

    /// Write the weights in the layout `kosm_render::gpu::neural::Weights`
    /// reads: an eight-byte magic, four `u32` of shape, then every f32 of
    /// `[W1 b1 W2 b2 W3 b3]`.
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
        let mut at = 0;
        for c in [&mut me.l1, &mut me.l2, &mut me.l3] {
            let nw = c.weight.data.numel();
            let nb = c.bias.data.numel();
            c.weight.data.data_mut().copy_from_slice(&vals[at..at + nw]);
            at += nw;
            c.bias.data.data_mut().copy_from_slice(&vals[at..at + nb]);
            at += nb;
        }
        Ok(me)
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

/// One accumulator for every parameter's gradient, so a batch can be summed
/// across threads and stepped once.
#[derive(Clone)]
pub struct Grads {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub w3: Vec<f32>,
    pub b3: Vec<f32>,
}

impl Grads {
    pub fn zeros(net: &Kpn) -> Self {
        Self {
            w1: vec![0.0; net.l1.weight.data.numel()],
            b1: vec![0.0; net.l1.bias.data.numel()],
            w2: vec![0.0; net.l2.weight.data.numel()],
            b2: vec![0.0; net.l2.bias.data.numel()],
            w3: vec![0.0; net.l3.weight.data.numel()],
            b3: vec![0.0; net.l3.bias.data.numel()],
        }
    }

    pub fn clear(&mut self) {
        for v in self.planes_mut() {
            v.fill(0.0);
        }
    }

    pub fn add(&mut self, other: &Grads) {
        let mut o = [
            other.w1.as_slice(),
            other.b1.as_slice(),
            other.w2.as_slice(),
            other.b2.as_slice(),
            other.w3.as_slice(),
            other.b3.as_slice(),
        ]
        .into_iter();
        for v in self.planes_mut() {
            let s = o.next().unwrap();
            for (a, b) in v.iter_mut().zip(s) {
                *a += b;
            }
        }
    }

    pub fn scale(&mut self, k: f32) {
        for v in self.planes_mut() {
            for a in v.iter_mut() {
                *a *= k;
            }
        }
    }

    fn planes_mut(&mut self) -> [&mut Vec<f32>; 6] {
        [
            &mut self.w1,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
            &mut self.w3,
            &mut self.b3,
        ]
    }
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
        for (i, v) in tgt.iter_mut().enumerate() {
            *v = ((i * 11 % 17) as f32 / 17.0) + 0.1;
        }
        (net, feat, illum, tgt)
    }

    /// A random linear projection of the output, and its gradient.
    ///
    /// Linear on purpose. A sum of squares is a perfectly good loss and a
    /// terrible thing to finite-difference in `f32`: it is order 10 while the
    /// gradient of any one of four thousand weights is order 1e-3, so
    /// `(L(w+ε) − L(w−ε))` is two nearly equal numbers subtracted and the
    /// answer is rounding. A random projection has the same Jacobian coverage
    /// — it checks `vᵀ ∂out/∂w` for a `v` that touches every output — and
    /// stays the same size as its own derivative.
    fn loss_and_grad(out: &[f32], tgt: &[f32]) -> (f64, Vec<f32>) {
        let mut l = 0.0f64;
        let mut g = vec![0.0f32; out.len()];
        for i in 0..out.len() {
            // `tgt` doubles as the projection direction, recentred on zero
            let v = tgt[i] - 0.5;
            l += (v * out[i]) as f64;
            g[i] = v;
        }
        (l, g)
    }

    #[test]
    fn the_backward_pass_agrees_with_a_finite_difference() {
        let s = 6;
        let (mut net, feat, illum, tgt) = toy(s);
        let act = net.forward(&feat, &illum, s);
        let (_, g_out) = loss_and_grad(&act.out, &tgt);
        let mut grads = Grads::zeros(&net);
        net.backward(&feat, &illum, &act, &g_out, &mut grads);

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
        let eps = 1e-3f32;
        for li in 0..3 {
            let n_w = match li {
                0 => net.l1.weight.data.numel(),
                1 => net.l2.weight.data.numel(),
                _ => net.l3.weight.data.numel(),
            };
            let dir: Vec<f32> = (0..n_w)
                .map(|i| if (i * 2654435761 >> 7) & 1 == 0 { 1.0 } else { -1.0 })
                .collect();
            let gw: &[f32] = match li {
                0 => &grads.w1,
                1 => &grads.w2,
                _ => &grads.w3,
            };
            let ana: f64 = gw.iter().zip(&dir).map(|(g, d)| (g * d) as f64).sum();

            fn walk(net: &mut Kpn, li: usize, dir: &[f32], k: f32) {
                let conv = match li {
                    0 => &mut net.l1,
                    1 => &mut net.l2,
                    _ => &mut net.l3,
                };
                for (w, d) in conv.weight.data.data_mut().iter_mut().zip(dir) {
                    *w += k * d;
                }
            }
            walk(&mut net, li, &dir, eps);
            let lp = loss_and_grad(&net.forward(&feat, &illum, s).out, &tgt).0;
            walk(&mut net, li, &dir, -2.0 * eps);
            let lm = loss_and_grad(&net.forward(&feat, &illum, s).out, &tgt).0;
            walk(&mut net, li, &dir, eps);

            let num = (lp - lm) / (2.0 * eps as f64);
            let scale = num.abs().max(ana.abs()).max(1e-6);
            assert!(
                (num - ana).abs() / scale < 2e-2,
                "layer {li}: analytic {ana}, numeric {num}"
            );
            assert!(ana.abs() > 1e-5, "layer {li} gradient is degenerate: {ana}");
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
    /// They are two implementations by necessity — this one has to produce
    /// activations a backward pass can walk, and `kosm_render`'s is a leaf
    /// crate that knows nothing about training — and a weight file is only
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

        let count = 4u32;
        let mine = net.filter(s, s, &mean, &variance, &normal, &depth, &albedo, count);
        let ours = theirs.forward(
            s,
            s,
            &mean,
            &variance,
            &normal,
            &depth,
            &albedo,
            count as f32,
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
