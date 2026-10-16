//! The tensor IR, as kosm runs it.
//!
//! Every network kosm trains — the PPO actor and critic, the behaviour-cloning
//! fit, the denoiser — is a [`TensorGraph`]: comprehensions (`out[i…] = ⊕_k
//! body(x[view(i,k)]…)`) over a scalar `tang_expr` body. The forward pass is
//! written once, the gradient is [`TensorGraph::backward`]'s, and nothing here
//! is a hand-written backprop. `tang/docs/tensor-ir.md` is the argument for
//! the design; this module is only the part kosm owns: one pass pipeline and
//! one way to run a graph.
//!
//! A [`Program`] is a graph lowered and loaded. Natively that is
//! [`TensorGraph::to_c`] built by the system C compiler and `dlopen`ed
//! ([`tang_gpu::cpu::CpuProgram`]): f32, static buffers, no allocation per
//! step. Where there is no C compiler — or with `KOSM_IR=interp` — it is the
//! IR's own reference interpreter, which is f64 and slow but needs nothing.
//! Both run the same graph, so a test can hold one against the other.
//!
//! A `Program`'s buffers belong to it alone, so it is `&mut self` to write or
//! run one; a batch of workers each loads their own.

use std::sync::atomic::{AtomicBool, Ordering};

pub use tang_compile::tensor::{Arg, Dim, Form, Monoid, ShapeError, TensorGraph, TensorId, View};
pub use tang_expr::ExprId;
use tang_gpu::cpu::CpuProgram;

/// The rewrites every kosm graph goes through before it is lowered: fuse
/// maps into their readers, drop common and dead sub-expressions, merge
/// sibling reductions, and distribute sums. None of them change the answer.
pub fn optimize(g: &TensorGraph, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
    let (g, o) = g.substitute_maps(outputs);
    let (g, o) = g.eliminate_common(&o);
    let (g, o) = g.hoist_invariant_factors(&o);
    let (g, o) = g.merge_siblings(&o);
    let (g, o) = g.merge_reduce_dims(&o);
    let (g, o) = g.split_reduce_dims(&o);
    g.eliminate_dead(&o)
}

/// A literal, for use inside a comprehension body. Like every `ExprId`, it
/// must be made inside the closure that builds the body.
pub fn lit(v: f64) -> ExprId {
    <ExprId as tang::Scalar>::from_f64(v)
}

/// An argument read through the identity view: `x[i…]` at the point's own
/// coordinates, in an iteration space of rank `space` (`x`'s dims first).
pub fn id(x: TensorId, rank: usize, space: usize) -> Arg {
    Arg { src: x, view: View::identity(space, rank) }
}

/// An argument read through a pick view: source dim `d` reads iteration
/// coordinate `picks[d]` (`None` broadcasts it).
pub fn pick(x: TensorId, space: usize, picks: &[Option<usize>]) -> Arg {
    Arg { src: x, view: View::pick(space, picks) }
}

enum Backend {
    Native(CpuProgram),
    Interp { g: TensorGraph, inputs: Vec<Vec<f64>>, out: Vec<Vec<f64>> },
}

/// A graph, lowered and loaded, with its inputs resident.
pub struct Program {
    backend: Backend,
    outputs: Vec<TensorId>,
    input_lens: Vec<usize>,
}

static WARNED: AtomicBool = AtomicBool::new(false);

fn force_interp() -> bool {
    std::env::var("KOSM_IR").is_ok_and(|v| v == "interp")
}

impl Program {
    /// Lower `outputs` of `g` and load them with `inputs` (graph input order).
    /// Native when a C compiler is available, the interpreter otherwise.
    pub fn compile(g: &TensorGraph, outputs: &[TensorId], inputs: &[&[f32]]) -> Self {
        let input_lens = input_lens(g);
        assert_eq!(inputs.len(), input_lens.len(), "Program::compile: input count");
        if !force_interp() {
            match CpuProgram::compile(g, outputs, inputs) {
                Ok(p) => return Self { backend: Backend::Native(p), outputs: outputs.to_vec(), input_lens },
                Err(e) => {
                    if !WARNED.swap(true, Ordering::Relaxed) {
                        eprintln!("kosm::ir: no native CPU program ({e}); using the interpreter");
                    }
                }
            }
        }
        let inputs = inputs.iter().map(|x| x.iter().map(|&v| v as f64).collect()).collect();
        Self::interp(g, outputs, inputs, input_lens)
    }

    /// `n` independent copies of one program, compiled side by side.
    ///
    /// A native program's buffers are the library's static arrays, so two
    /// workers cannot share one; each copy is its own build in its own
    /// directory. The compiles run in parallel, so `n` copies cost about
    /// what one does on a machine with `n` cores.
    pub fn compile_n(g: &TensorGraph, outputs: &[TensorId], inputs: &[&[f32]], n: usize) -> Vec<Self> {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(|_| Self::compile(g, outputs, inputs)).collect()
    }

    /// The reference interpreter, always: f64 throughout. For tests, and for
    /// anything whose numbers must not move when the compiler does.
    pub fn interpreted(g: &TensorGraph, outputs: &[TensorId], inputs: &[&[f64]]) -> Self {
        let input_lens = input_lens(g);
        assert_eq!(inputs.len(), input_lens.len(), "Program::interpreted: input count");
        Self::interp(g, outputs, inputs.iter().map(|x| x.to_vec()).collect(), input_lens)
    }

    fn interp(g: &TensorGraph, outputs: &[TensorId], inputs: Vec<Vec<f64>>, input_lens: Vec<usize>) -> Self {
        for (k, (x, &n)) in inputs.iter().zip(&input_lens).enumerate() {
            assert_eq!(x.len(), n, "input {k} length");
        }
        Self {
            backend: Backend::Interp { g: g.clone(), inputs, out: Vec::new() },
            outputs: outputs.to_vec(),
            input_lens,
        }
    }

    /// Whether this is the compiled C program rather than the interpreter.
    pub fn is_native(&self) -> bool {
        matches!(self.backend, Backend::Native(_))
    }

    /// Element count of graph input `k`.
    pub fn input_len(&self, k: usize) -> usize {
        self.input_lens[k]
    }

    /// Overwrite graph input `k`.
    pub fn write(&mut self, k: usize, data: &[f32]) {
        assert_eq!(data.len(), self.input_lens[k], "input {k} length");
        match &mut self.backend {
            Backend::Native(p) => p.write_input(k, data),
            Backend::Interp { inputs, .. } => {
                for (d, &s) in inputs[k].iter_mut().zip(data) {
                    *d = s as f64;
                }
            }
        }
    }

    /// Overwrite graph input `k` from f64 (rounded to f32 natively).
    pub fn write_f64(&mut self, k: usize, data: &[f64]) {
        match &mut self.backend {
            Backend::Native(p) => {
                let v: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                assert_eq!(v.len(), self.input_lens[k], "input {k} length");
                p.write_input(k, &v);
            }
            Backend::Interp { inputs, .. } => {
                assert_eq!(data.len(), self.input_lens[k], "input {k} length");
                inputs[k].copy_from_slice(data);
            }
        }
    }

    /// Run every kernel once.
    pub fn run(&mut self) {
        match &mut self.backend {
            Backend::Native(p) => p.run(),
            Backend::Interp { g, inputs, out } => *out = g.eval(inputs, &self.outputs),
        }
    }

    /// Output `i` (in the order `compile` was given) after the last `run`.
    pub fn read(&self, i: usize) -> Vec<f32> {
        match &self.backend {
            Backend::Native(p) => p.read_output(i),
            Backend::Interp { out, .. } => out[i].iter().map(|&v| v as f32).collect(),
        }
    }

    /// [`read`](Self::read), widened to f64 (exact under the interpreter).
    pub fn read_f64(&self, i: usize) -> Vec<f64> {
        match &self.backend {
            Backend::Native(p) => p.read_output(i).into_iter().map(|v| v as f64).collect(),
            Backend::Interp { out, .. } => out[i].clone(),
        }
    }
}

/// Element count of every graph input, in input order — what a fresh
/// [`Program::compile`] has to be handed.
pub fn input_lens(g: &TensorGraph) -> Vec<usize> {
    g.nodes()
        .iter()
        .filter(|n| matches!(n.op, tang_compile::tensor::Op::Input(_)))
        .map(|n| tang_compile::tensor::numel(&n.shape))
        .collect()
}

/// One Adam step over a flat parameter vector, as a graph.
///
/// Inputs, in order: `p`, `grad`, `m`, `v` (each `[n]`) and `bc` (`[2]`, the
/// bias corrections `1 − β₁ᵗ` and `1 − β₂ᵗ`). Outputs `[m', v', p']`. The
/// update is the textbook one (Kingma & Ba, eq. after Alg. 1):
/// `p' = p − lr · (m'/bc₁) / (√(v'/bc₂) + ε)`.
pub fn adam_step(n: usize, lr: f64, beta1: f64, beta2: f64, eps: f64) -> (TensorGraph, [TensorId; 3]) {
    let mut g = TensorGraph::new();
    let p = g.input(&[n]);
    let grad = g.input(&[n]);
    let m = g.input(&[n]);
    let v = g.input(&[n]);
    let bc = g.input(&[2]);
    let m2 = g.map(&[m, grad], |x| x[0] * lit(beta1) + x[1] * lit(1.0 - beta1)).unwrap();
    let v2 = g.map(&[v, grad], |x| x[0] * lit(beta2) + x[1] * x[1] * lit(1.0 - beta2)).unwrap();
    let bcv = |k: i64| Arg { src: bc, view: View { guard: false, forms: vec![Form { coefs: vec![0], offset: k }] } };
    let p2 = g
        .comp(&[n], &[] as &[usize], vec![id(p, 1, 1), id(m2, 1, 1), id(v2, 1, 1), bcv(0), bcv(1)], None, |x| {
            let m_hat = x[1] / x[3];
            let v_hat = x[2] / x[4];
            vec![x[0] - lit(lr) * m_hat / (<ExprId as tang::Scalar>::sqrt(v_hat) + lit(eps))]
        })
        .unwrap();
    (g, [m2, v2, p2])
}

/// Adam over a flat parameter vector, run as a [`Program`], with its moment
/// state resident. The caller owns the parameters; [`step`](Self::step)
/// updates them in place.
pub struct Adam {
    program: Program,
    beta1: f64,
    beta2: f64,
    t: i32,
}

impl Adam {
    pub fn new(n: usize, lr: f64) -> Self {
        Self::with(n, lr, 0.9, 0.999, 1e-8)
    }

    pub fn with(n: usize, lr: f64, beta1: f64, beta2: f64, eps: f64) -> Self {
        let (g, outs) = adam_step(n, lr, beta1, beta2, eps);
        let z = vec![0.0f32; n];
        let program = Program::compile(&g, &outs, &[&z, &z, &z, &z, &[1.0, 1.0]]);
        Self { program, beta1, beta2, t: 0 }
    }

    /// Steps taken so far.
    pub fn t(&self) -> i32 {
        self.t
    }

    /// One step: `params -= lr · m̂ / (√v̂ + ε)` from `grad`.
    pub fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        self.t += 1;
        let bc = [1.0 - self.beta1.powi(self.t), 1.0 - self.beta2.powi(self.t)];
        self.program.write_f64(0, params);
        self.program.write_f64(1, grad);
        self.program.write_f64(4, &bc);
        self.program.run();
        let m = self.program.read_f64(0);
        let v = self.program.read_f64(1);
        params.copy_from_slice(&self.program.read_f64(2));
        self.program.write_f64(2, &m);
        self.program.write_f64(3, &v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `y = Σ_j tanh(Σ_k x[i,k] W[k,j])²`, and its gradient through
    /// `backward`, native against the interpreter against finite differences.
    #[test]
    fn mlp_gradient_native_matches_interpreter_and_fd() {
        let (b, i, h) = (4usize, 3usize, 5usize);
        let mut g = TensorGraph::new();
        let x = g.input(&[b, i]);
        let w = g.input(&[i, h]);
        let z = g.matmul(x, w).unwrap();
        let a = g.map(&[z], |v| <ExprId as tang::Scalar>::tanh(v[0])).unwrap();
        let y = g
            .comp(&[1usize], &[b, h], vec![pick(a, 3, &[Some(1), Some(2)])], Some(Monoid::sum()), |v| vec![v[0] * v[0]])
            .unwrap();
        let dw = g.backward(y, &[w]).unwrap()[0];
        let (g, outs) = optimize(&g, &[y, dw]);

        let xs: Vec<f64> = (0..b * i).map(|k| ((k * 7) % 11) as f64 * 0.1 - 0.5).collect();
        let ws: Vec<f64> = (0..i * h).map(|k| ((k * 5) % 13) as f64 * 0.07 - 0.4).collect();
        let mut interp = Program::interpreted(&g, &outs, &[&xs, &ws]);
        interp.run();
        let (y0, dw0) = (interp.read_f64(0)[0], interp.read_f64(1));

        let eps = 1e-6;
        for k in 0..i * h {
            let mut lo = ws.clone();
            lo[k] -= eps;
            let mut hi = ws.clone();
            hi[k] += eps;
            let f = |w: &[f64]| {
                let mut p = Program::interpreted(&g, &outs, &[&xs, w]);
                p.run();
                p.read_f64(0)[0]
            };
            let fd = (f(&hi) - f(&lo)) / (2.0 * eps);
            assert!((fd - dw0[k]).abs() < 1e-6, "dW[{k}]: fd {fd} vs ir {}", dw0[k]);
        }

        let xf: Vec<f32> = xs.iter().map(|&v| v as f32).collect();
        let wf: Vec<f32> = ws.iter().map(|&v| v as f32).collect();
        let mut native = Program::compile(&g, &outs, &[&xf, &wf]);
        native.run();
        assert!((native.read_f64(0)[0] - y0).abs() < 1e-4);
        for (a, b) in native.read_f64(1).iter().zip(&dw0) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn adam_descends_a_bowl() {
        let mut p = vec![3.0, -2.0, 0.5];
        let mut opt = Adam::new(3, 0.1);
        for _ in 0..300 {
            let grad: Vec<f64> = p.iter().map(|x| 2.0 * x).collect();
            opt.step(&mut p, &grad);
        }
        assert!(p.iter().all(|x| x.abs() < 1e-2), "{p:?}");
    }
}
