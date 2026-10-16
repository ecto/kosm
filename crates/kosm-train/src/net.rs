//! The MLP every loop in this crate trains, as tensor-IR graphs.
//!
//! The weights are a flat `Vec<f64>` in the artifact's documented order —
//! `l1.w, l1.b, l2.w, l2.b, l3.w, l3.b`, each weight `[out, in]` row-major,
//! which is how `tang_train::Linear` laid them out, so every actor saved
//! before this file existed loads and acts the same. What runs them is a
//! [`kosm::ir`] graph: the forward pass is [`mlp`], a loss is a few more
//! comprehensions on top of it, and the gradient is
//! [`TensorGraph::backward`]'s. Nothing in this crate is a hand-written
//! backprop any more.
//!
//! A forward on one thread compiles once and is reused: [`Mlp::forward`] and
//! [`Mlp::forward_rows`] keep one [`Program`] per network shape per thread,
//! and rewrite its weight inputs only when the network has changed since it
//! last ran there. A rayon worker rolling out a thousand control steps pays
//! for one C compile on its first episode and a memcpy per step after.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use kosm::ir::{Monoid, Program, TensorGraph, TensorId, id, pick};
use tang::Scalar;

/// A two-hidden-layer tanh MLP with flat import/export. The weights are
/// `f64` here and in every artifact; natively they run as `f32`.
///
/// The parameter order is a stable, documented fact of the artifact format:
/// `l1.w, l1.b, l2.w, l2.b, l3.w, l3.b`, each weight `[out, in]` row-major.
#[derive(Clone)]
pub struct Mlp {
    flat: Vec<f64>,
    pub dims: (usize, usize, usize),
    /// Changes whenever `flat` does, so a cached [`Program`] knows whether
    /// its weight inputs are stale. Never 0 (0 is "nothing loaded").
    stamp: u64,
}

static STAMP: AtomicU64 = AtomicU64::new(1);

fn fresh_stamp() -> u64 {
    STAMP.fetch_add(1, Ordering::Relaxed)
}

/// `tang_train`'s `Rng` and `Parameter::randn`, replicated so a seed draws
/// the weights it always drew: a 64-bit LCG (Knuth's MMIX constants), Box–
/// Muller, Xavier/Glorot scale `√(2 / (fan_in + fan_out))` over an `[out,
/// in]` weight.
fn xavier(out: &mut Vec<f64>, rows: usize, cols: usize, seed: u64) {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let scale = (2.0 / (cols as f64 + rows as f64)).sqrt();
    for _ in 0..rows * cols {
        let u1 = next().max(1e-15);
        let u2 = next();
        out.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * scale);
    }
}

impl Mlp {
    /// Xavier weights and zero biases, drawn exactly as `tang_train::Linear`
    /// drew them (layer seeds `seed ^ 0x11`, `^ 0x22`, `^ 0x33`), so a seed
    /// is the same network it was before the IR.
    pub fn new(input: usize, hidden: usize, output: usize, seed: u64) -> Self {
        let mut flat = Vec::with_capacity(Self::param_count(input, hidden, output));
        xavier(&mut flat, hidden, input, seed ^ 0x11);
        flat.resize(flat.len() + hidden, 0.0);
        xavier(&mut flat, hidden, hidden, seed ^ 0x22);
        flat.resize(flat.len() + hidden, 0.0);
        xavier(&mut flat, output, hidden, seed ^ 0x33);
        flat.resize(flat.len() + output, 0.0);
        Self { flat, dims: (input, hidden, output), stamp: fresh_stamp() }
    }

    /// A copy with the same weights. Kept for callers from when `forward`
    /// was `&mut self` and one net could not be shared across workers; it is
    /// `&self` now, so a shared reference does.
    pub fn clone_weights(&self) -> Self {
        self.clone()
    }

    pub fn to_flat(&self) -> Vec<f64> {
        self.flat.clone()
    }

    /// The flat parameters, borrowed.
    pub fn params(&self) -> &[f64] {
        &self.flat
    }

    pub fn from_flat(input: usize, hidden: usize, output: usize, flat: &[f64]) -> Option<Self> {
        (flat.len() == Self::param_count(input, hidden, output)).then(|| Self {
            flat: flat.to_vec(),
            dims: (input, hidden, output),
            stamp: fresh_stamp(),
        })
    }

    /// Overwrite every parameter (an optimiser step). Panics on a length
    /// mismatch: the layout is fixed by `dims`.
    pub fn set_flat(&mut self, flat: &[f64]) {
        self.flat.copy_from_slice(flat);
        self.stamp = fresh_stamp();
    }

    pub fn param_count(input: usize, hidden: usize, output: usize) -> usize {
        input * hidden + hidden + hidden * hidden + hidden + hidden * output + output
    }

    /// Element ranges of the six parameter blocks in the flat vector, in
    /// order, and their shapes (`[out, in]` for weights, `[out]` for biases).
    pub(crate) fn blocks(dims: (usize, usize, usize)) -> [(std::ops::Range<usize>, Vec<usize>); 6] {
        let (i, h, o) = dims;
        let shapes = [vec![h, i], vec![h], vec![h, h], vec![h], vec![o, h], vec![o]];
        let mut at = 0;
        shapes.map(|s| {
            let n: usize = s.iter().product();
            at += n;
            (at - n..at, s)
        })
    }

    /// Scale the output layer's initial weights. The initial policy is then
    /// near-zero mean — plain PD at whatever pose the env spawns in, a
    /// controller that already works — and exploration is the log-std rather
    /// than Xavier noise saturating the clamp. Measured in ipse: without
    /// this, 120 iterations sat at 0.56 s episodes because the policy was
    /// fighting its own initialization.
    pub fn scale_output(&mut self, k: f64) {
        let r = Self::blocks(self.dims)[4].0.clone();
        for v in &mut self.flat[r] {
            *v *= k;
        }
        self.stamp = fresh_stamp();
    }

    /// One row through the net. Shorter inputs are zero-padded and longer
    /// ones truncated to the input width.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut row = vec![0.0; self.dims.0];
        let n = x.len().min(self.dims.0);
        row[..n].copy_from_slice(&x[..n]);
        run_cached(self, 1, &row)
    }

    /// Many rows (`x.len()` a multiple of the input width) through the net,
    /// `ROWS` at a time through one cached program; the last chunk is
    /// zero-padded and its padding dropped.
    pub fn forward_rows(&self, x: &[f64]) -> Vec<f64> {
        let (i, _, o) = self.dims;
        assert_eq!(x.len() % i.max(1), 0, "forward_rows: ragged input");
        let n = x.len() / i.max(1);
        let mut out = Vec::with_capacity(n * o);
        for chunk in x.chunks(ROWS * i) {
            let rows = chunk.len() / i;
            let mut buf = vec![0.0; ROWS * i];
            buf[..chunk.len()].copy_from_slice(chunk);
            out.extend_from_slice(&run_cached(self, ROWS, &buf)[..rows * o]);
        }
        out
    }
}

/// The batch [`Mlp::forward_rows`] compiles for.
const ROWS: usize = 256;

/// An MLP's six parameter tensors as graph inputs, declared in flat order.
#[derive(Clone, Copy)]
pub(crate) struct MlpVars(pub [TensorId; 6]);

impl MlpVars {
    pub fn inputs(g: &mut TensorGraph, dims: (usize, usize, usize)) -> Self {
        Self(Mlp::blocks(dims).map(|(_, s)| g.input(&s)))
    }

    /// Write `net`'s weights into the six inputs starting at graph input `k`.
    pub fn write(p: &mut Program, k: usize, net: &Mlp) {
        for (j, (r, _)) in Mlp::blocks(net.dims).into_iter().enumerate() {
            p.write_f64(k + j, &net.flat[r]);
        }
    }
}

/// `act(x · Wᵀ + b)` over a `[rows, in]` batch: `W` is `[out, in]`.
fn linear(g: &mut TensorGraph, x: TensorId, w: TensorId, b: TensorId, rows: usize, din: usize, dout: usize, tanh: bool) -> TensorId {
    let z = g
        .comp(&[rows, dout], &[din], vec![pick(x, 3, &[Some(0), Some(2)]), pick(w, 3, &[Some(1), Some(2)])], Some(Monoid::sum()), |v| {
            vec![v[0] * v[1]]
        })
        .expect("linear: matmul");
    g.comp(&[rows, dout], &[] as &[usize], vec![id(z, 2, 2), pick(b, 2, &[Some(1)])], None, |v| {
        let s = v[0] + v[1];
        vec![if tanh { s.tanh() } else { s }]
    })
    .expect("linear: bias")
}

/// The MLP over a `[rows, in]` input: `[rows, out]`.
pub(crate) fn mlp(g: &mut TensorGraph, x: TensorId, p: MlpVars, rows: usize, dims: (usize, usize, usize)) -> TensorId {
    let (i, h, o) = dims;
    let [w1, b1, w2, b2, w3, b3] = p.0;
    let a1 = linear(g, x, w1, b1, rows, i, h, true);
    let a2 = linear(g, a1, w2, b2, rows, h, h, true);
    linear(g, a2, w3, b3, rows, h, o, false)
}

/// The forward graph: inputs `x [rows, in]` then the six parameters;
/// output `y [rows, out]`.
pub(crate) fn forward_graph(dims: (usize, usize, usize), rows: usize) -> (TensorGraph, Vec<TensorId>) {
    let mut g = TensorGraph::new();
    let x = g.input(&[rows, dims.0]);
    let p = MlpVars::inputs(&mut g, dims);
    let y = mlp(&mut g, x, p, rows, dims);
    kosm::ir::optimize(&g, &[y])
}

/// Zero inputs of the right lengths for a fresh [`Program::compile`].
pub(crate) fn zero_inputs(g: &TensorGraph) -> Vec<Vec<f32>> {
    kosm::ir::input_lens(g).into_iter().map(|n| vec![0.0; n]).collect()
}

/// Compile with every input zero; the caller writes the real ones.
pub(crate) fn compile_zeroed(g: &TensorGraph, outs: &[TensorId]) -> Program {
    let z = zero_inputs(g);
    let refs: Vec<&[f32]> = z.iter().map(|v| v.as_slice()).collect();
    Program::compile(g, outs, &refs)
}

thread_local! {
    /// One forward program per (dims, rows) per thread, and the stamp of the
    /// weights it last loaded.
    static FORWARD: RefCell<HashMap<((usize, usize, usize), usize), (Program, u64)>> = RefCell::new(HashMap::new());
}

fn run_cached(net: &Mlp, rows: usize, x: &[f64]) -> Vec<f64> {
    FORWARD.with(|cache| {
        let mut cache = cache.borrow_mut();
        let (p, loaded) = cache.entry((net.dims, rows)).or_insert_with(|| {
            let (g, outs) = forward_graph(net.dims, rows);
            (compile_zeroed(&g, &outs), 0)
        });
        if *loaded != net.stamp {
            MlpVars::write(p, 1, net);
            *loaded = net.stamp;
        }
        p.write_f64(0, x);
        p.run();
        p.read_f64(0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The f64 reference: `tanh(tanh(x W1ᵀ + b1) W2ᵀ + b2) W3ᵀ + b3` in plain
    /// loops over the flat layout. If the layout or the graph disagreed with
    /// the documented order, this is where it would show.
    fn reference(net: &Mlp, x: &[f64]) -> Vec<f64> {
        let (i, h, o) = net.dims;
        let b = Mlp::blocks(net.dims);
        let f = &net.flat;
        let layer = |x: &[f64], w: &[f64], bias: &[f64], din: usize, dout: usize, t: bool| -> Vec<f64> {
            (0..dout)
                .map(|r| {
                    let s = bias[r] + (0..din).map(|c| w[r * din + c] * x[c]).sum::<f64>();
                    if t { s.tanh() } else { s }
                })
                .collect()
        };
        let a1 = layer(x, &f[b[0].0.clone()], &f[b[1].0.clone()], i, h, true);
        let a2 = layer(&a1, &f[b[2].0.clone()], &f[b[3].0.clone()], h, h, true);
        layer(&a2, &f[b[4].0.clone()], &f[b[5].0.clone()], h, o, false)
    }

    /// `Mlp::new` draws what `tang_train::Linear::new` drew. The bits were
    /// printed from the tang_train implementation at 45e4509 (the last
    /// commit that had it): `Mlp::new(3, 4, 2, 7)`.
    #[test]
    fn init_matches_tang_train_bit_for_bit() {
        let got: Vec<u64> = Mlp::new(3, 4, 2, 7).to_flat().iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, GOLDEN_3_4_2_SEED7.to_vec());
    }

    /// to_flat / from_flat is the identity, and the flat order is the
    /// documented one: perturbing one entry of `l3.w` at `[out d, in c]`
    /// moves output `d` by `δ · a2[c]` and nothing else.
    #[test]
    fn flat_round_trip_and_layout() {
        let net = Mlp::new(3, 5, 2, 9);
        let back = Mlp::from_flat(3, 5, 2, &net.to_flat()).unwrap();
        assert_eq!(net.to_flat(), back.to_flat());
        assert!(Mlp::from_flat(3, 5, 2, &net.to_flat()[1..]).is_none());

        let x = [0.3, -0.7, 0.2];
        let mut flat = net.to_flat();
        // The six blocks at their documented offsets: l1.w [5,3], l1.b [5],
        // l2.w [5,5], l2.b [5], l3.w [2,5], l3.b [2].
        let b = Mlp::blocks(net.dims);
        let ranges: Vec<_> = b.iter().map(|(r, _)| r.clone()).collect();
        assert_eq!(ranges, [0..15, 15..20, 20..45, 45..50, 50..60, 60..62]);
        assert_eq!(b[4].1, [2, 5]);
        let (d, c, delta) = (1, 3, 1e-3);
        flat[b[4].0.start + d * 5 + c] += delta;
        let moved = Mlp::from_flat(3, 5, 2, &flat).unwrap();
        let (y0, y1) = (reference(&net, &x), reference(&moved, &x));
        assert_eq!(y0[0], y1[0]);
        assert!((y1[1] - y0[1]).abs() > 0.0);

        // And the IR forward agrees with the loops (f32 natively).
        let ir = net.forward(&x);
        for (a, b) in ir.iter().zip(&y0) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        let ir2 = moved.forward(&x);
        assert!((ir2[1] - y1[1]).abs() < 1e-5);
    }

    /// Native f32 against the f64 interpreter on the same graph.
    #[test]
    fn native_forward_matches_interpreter() {
        let net = Mlp::new(4, 6, 3, 2);
        let rows = 5;
        let (g, outs) = forward_graph(net.dims, rows);
        let x: Vec<f64> = (0..rows * 4).map(|k| ((k * 7) % 11) as f64 * 0.2 - 1.0).collect();
        let mut inputs = vec![x.clone()];
        for (r, _) in Mlp::blocks(net.dims) {
            inputs.push(net.flat[r].to_vec());
        }
        let refs: Vec<&[f64]> = inputs.iter().map(|v| v.as_slice()).collect();
        let mut interp = Program::interpreted(&g, &outs, &refs);
        interp.run();
        let want = interp.read_f64(0);
        let mut native = compile_zeroed(&g, &outs);
        assert!(native.is_native(), "no C compiler: this test needs one");
        native.write_f64(0, &x);
        MlpVars::write(&mut native, 1, &net);
        native.run();
        for (a, b) in native.read_f64(0).iter().zip(&want) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        for r in 0..rows {
            let y = reference(&net, &x[r * 4..(r + 1) * 4]);
            for d in 0..3 {
                assert!((want[r * 3 + d] - y[d]).abs() < 1e-12);
            }
        }
        // forward_rows pads and drops: same answers as the interpreter.
        let rows_out = net.forward_rows(&x);
        assert_eq!(rows_out.len(), rows * 3);
        for (a, b) in rows_out.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    const GOLDEN_3_4_2_SEED7: [u64; 46] = [
        0x3fdbafef0b441bff, 0x3fe2a4124a3e5342, 0x3fb26256b4684fe5, 0x3fe2e7fd1a70e757,
        0x3fe93571aeafe66d, 0x3feab3ef457bfb56, 0x3feb1446a9d5a1d2, 0xbfe1e836ff75d3d4,
        0x3fdb05601f96df9e, 0x3fe192c88018b63b, 0x3fc8c1c430dfa80a, 0xbfbaf5d5c076d342,
        0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
        0x3fbcdb6491ca48f0, 0xbfc40dfd2654ad06, 0xbfcd64ea848b7edb, 0xbfd78a1063b2e1c4,
        0xbfa574a4aab8d788, 0xbfc373051266fc92, 0x3fbdd5c3a42a447a, 0xbfe5a0838ef94973,
        0x3fe7f9d6ce19809d, 0xbf9d20f01e7589bc, 0xbfbc1fdf57f95cc1, 0x3fe2f6372064a1a8,
        0x3fc340c259873881, 0xbfdd5d2d3840d25f, 0x3fa821147c7e9c8d, 0x3f90b07ee57727d4,
        0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
        0xbfe0f3926dff5318, 0xbfd463a4e240bff9, 0xbfd5eed3fe508dbc, 0x3f80e59929548eee,
        0xbff64af3a811d8d7, 0x3fb15461fdc9a88b, 0xbfed857e42c3b7f5, 0x3ff0e3dac0d55091,
        0x0000000000000000, 0x0000000000000000,
    ];
}
