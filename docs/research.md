# Research notes

Collected 2026-09-02, alongside the pool spike. Two passes: first the
frontier of simulation and rendering as it stands, then a second pass seeded
by the question "if you had to simulate the real universe, what would you
use?" — nine techniques, and what arXiv already has for each.

The through-line, stated once: every level of the universe is an effective
theory, exact for the questions asked at its scale. A simulator is a machine
for deciding which effective theory applies where, and for keeping the
theories consistent with each other. Everything below is a piece of that
machine, and every piece exists as a mature literature that does not cite the
others. The composition is the work.

## Part I — the frontier as it stands

### Differentiating through contact (exact and differentiable)

- Zeng et al., *Fast and Reliable Gradients for Deformables Across Frictional
  Contact Regimes*, [2603.16478](https://arxiv.org/abs/2603.16478). The
  sharpest statement of why contact gradients fail (non-Markovian position
  tricks, heuristic gradients); fixed with Markovian dynamics on a
  position–velocity manifold, a mass-aligned preconditioner, a soft
  Fischer–Burmeister complementarity operator. Fully GPU. The recipe
  phyz-diff should follow.
- DiffMJX, *Differentiable Simulation of Hard Contacts with Soft Gradients*,
  [2506.14186](https://arxiv.org/abs/2506.14186). Adaptive time integration +
  penalty contact so autodiff gradients stay correct at stiff settings.
- *End-to-End and Highly-Efficient Differentiable Simulation for Robotics*,
  [2409.07107](https://arxiv.org/pdf/2409.07107). Implicit differentiation of
  the frictional NCP without relaxation.
- *Few-Shot Neural Differentiable Simulator: Real-to-Sim Rigid-Contact
  Modeling*, [2603.06218](https://arxiv.org/abs/2603.06218).

### Learned closures inside classical solvers

- *Learning constitutive models and rheology from partial flow measurements*,
  [2510.24673](https://arxiv.org/abs/2510.24673v1). Differentiable fluid solver
  with a tensor-basis neural stress closure fit to velocimetry.
- *Non-Markovian closures via differentiable physics-neural models*,
  [2511.21369](https://arxiv.org/pdf/2511.21369).
- *Energy-conserving neural closure for long-time stable LES*,
  [2504.05868](https://arxiv.org/pdf/2504.05868). The constraint a distilled
  fast tier must satisfy.
- *Neural stress fields for reduced-order elastoplasticity and fracture*,
  [2310.17790](https://arxiv.org/pdf/2310.17790). MPM with a learned stress.

### Multi-scale water, done properly

- Wang, …, Zhu, *Hamiltonian Two-Way Coupling of Nonlinear Waves and 3D
  Flows*, [2608.25203](https://arxiv.org/abs/2608.25203) (Aug 2026). Zakharov
  surface waves (η, ψ) coupled two-way to NB-FLIP; the Hamiltonian structure
  makes the exchange energy-consistent. 1.7–5× lower wave error than
  shallow-water / Airy baselines, >1000× faster than BEM, "suppression of
  visible seam artifacts". Our box + far field with the two-way part we
  skipped. They lack differentiability and MPM.
- *Real-Time Interactive Hybrid Ocean: Spectrum-Consistent Wave Particle–FFT
  Coupling*, [2511.02852](https://arxiv.org/abs/2511.02852). Global FFT ocean
  + local wave-particle patches under one spectrum; no visible discontinuity.
- *Jet formation and collapse for axisymmetric surface gravity waves: coupled
  potential flow and SPH*, [2512.19924](https://arxiv.org/pdf/2512.19924). A
  validation case with our exact physics.

### Splats as the physics substrate

- MeGAS (thermomechanical dynamic 3DGS), [2606.23455](https://arxiv.org/abs/2606.23455);
  GaussianFluent (mixed materials), [2601.09265](https://arxiv.org/pdf/2601.09265);
  i-PhysGaussian (implicit sim), [2602.17117](https://arxiv.org/pdf/2602.17117);
  Gaussian-augmented sim + system ID with complex colliders,
  [2511.06846](https://arxiv.org/pdf/2511.06846); LumiMotion (relighting under
  dynamics), [2604.10994](https://arxiv.org/pdf/2604.10994). MPM over splats
  with relighting is where the field is converging.

### Caustics and real-time light

- Manifold Path Guiding, [2311.12818](https://arxiv.org/html/2311.12818v1);
  Specular Polynomials, [2405.13409](https://arxiv.org/pdf/2405.13409). The
  offline answers to water-onto-floor; the reference tier should adopt one.
- Neural path guiding with distribution factorization,
  [2506.00839](https://arxiv.org/html/2506.00839); real-time neural irradiance
  volume, [2602.12949](https://arxiv.org/html/2602.12949v1); OctaOctree neural
  radiosity for glossy real-time, [2606.08469](https://arxiv.org/pdf/2606.08469).
  The live tier's GI path is neural caches.

### Fast learned tiers that do not drift

- FluxNet, capacity-constrained local transport operators,
  [2602.01941](https://arxiv.org/abs/2602.01941). Machine-precision
  conservation and bounds in a neural PDE surrogate.
- Conserved-quantity correction for long rollouts,
  [2601.22541](https://arxiv.org/html/2601.22541); cellular sheaf neural
  operators, [2606.00937](https://arxiv.org/html/2606.00937); self-refining
  surrogates, [2603.17750](https://arxiv.org/pdf/2603.17750).

### MPM at scale (from the earlier pull)

- Fei et al., *Principles towards Real-Time Simulation of MPM on Modern
  GPUs*, [2111.00699](https://arxiv.org/abs/2111.00699). Block-sorted P2G,
  shared-memory tiles, multi-GPU. Landed in kosm-mpm as p2g_block.
- Zhao et al., *Unified sparse framework for large-scale MPM*,
  [2605.28525](https://arxiv.org/abs/2605.28525). Hash/scan sparse grids.
- Bird et al., *Implicit octree-based adaptive MPM*,
  [2606.09275](https://arxiv.org/abs/2606.09275). Resolution as a field.
- Wang et al., *Hierarchical Optimization Time Integration for CFL-rate MPM*,
  [1911.07913](https://arxiv.org/abs/1911.07913). Escaping the sound-speed dt.

## Part II — nine techniques for a universe, and the literature for each

### 1. Compute only what can be distinguished

Goal-oriented adaptivity: the adjoint of the quantity of interest says where
model error matters.

- Goal-oriented error estimation and adaptivity in MsFEM,
  [1908.00367](https://arxiv.org/abs/1908.00367): the estimator chooses coarse
  vs sophisticated model per region.
- Locally different models in a checkerboard with error control for multiple
  quantities of interest, [2405.18567](https://arxiv.org/pdf/2405.18567).
- Multigoal-oriented adaptive FEM with convergence rates,
  [2601.01965](https://arxiv.org/html/2601.01965v1).
- Graphics has only output metrics: visual error prediction for real-time,
  [2310.09125](https://arxiv.org/html/2310.09125); perceptual evaluation of
  liquid simulation, [2011.10257](https://arxiv.org/pdf/2011.10257); learned
  controllable adaptive simulation, [2305.01122](https://arxiv.org/pdf/2305.01122).

Missing on arXiv: the quantity of interest being a rendered image or an
agent's decision. We have the adjoint of the renderer and of the sim.

### 2. Hierarchical effective theories, reconstruction on demand

The reconstruction map is "backmapping" in chemistry and "super-resolution"
in fluids; the good ones are generative, conditioned on the coarse state, and
energy-guided.

- BackDiff, [2310.01768](https://arxiv.org/abs/2310.01768); FlowBack-Adjoint
  (energy-guided flow matching), [2508.03619](https://arxiv.org/pdf/2508.03619);
  constraint-decoupled latent diffusion, [2410.13264](https://arxiv.org/html/2410.13264);
  iterative backmapping, [2505.18082](https://arxiv.org/abs/2505.18082).
- ReMD, physics-consistent diffusion super-resolution via multiscale residual
  correction, [2603.00149](https://arxiv.org/abs/2603.00149); diffusion
  super-resolution inside a differentiable coarse solver,
  [2406.20047](https://arxiv.org/html/2406.20047v1); temporally consistent
  turbulence between sparse snapshots, [2512.24813](https://ar5iv.labs.arxiv.org/html/2512.24813).
- Theory: Machine-Learning Renormalization Group,
  [2306.11054](https://arxiv.org/pdf/2306.11054) (proposes effective theories
  and finds RG flows automatically); diffusion models as inverse RG flows,
  [2501.09064](https://arxiv.org/html/2501.09064v1) — reconstruction *is* the
  RG run backwards.

### 3. Conservation and symmetry as the invariants

Port-Hamiltonian systems: couple anything to anything with power conserved at
the port.

- Structure-preserving coupling and decoupling of pH systems,
  [2511.20150](https://arxiv.org/abs/2511.20150) (monolithic for stability,
  decoupled for distributed simulation).
- pH modelling across hydraulic and mechanical domains,
  [2008.07985](https://arxiv.org/pdf/2008.07985); rigid/flexible multibody,
  [2608.05143](https://arxiv.org/html/2608.05143); pH strings with
  energy-consistent time stepping, [2304.10957](https://arxiv.org/abs/2304.10957).
- Structure- and stability-preserving *learning* of pH systems,
  [2604.13297](https://arxiv.org/html/2604.13297v1) — for the learned levels.

### 4. Statistics where dynamics are unobservable, consistent with what was seen

Data assimilation, now generative.

- Score-based Data Assimilation, [2306.10574](https://arxiv.org/abs/2306.10574):
  a learned prior over trajectories, observations as a likelihood, zero-shot.
- Generative assimilation of sparse weather stations at km scale,
  [2406.16947](https://arxiv.org/abs/2406.16947); DAISI (stochastic
  interpolants), [2512.00252](https://arxiv.org/pdf/2512.00252); conditional
  diffusion for PDE simulation, [2410.16415](https://arxiv.org/html/2410.16415);
  SLAMS multimodal latent assimilation, [2404.06665](https://arxiv.org/abs/2404.06665).

Open: stability under re-observation (sampling the same region twice must
agree).

### 5. Time is local

- Variational multirate integrators, [2406.12991](https://arxiv.org/abs/2406.12991):
  symplectic, momentum-preserving on macro/micro time grids.
- Asynchronous variational integrators (Maxwell), [0803.2070](https://arxiv.org/pdf/0803.2070).
- Asynchronous discrete-event schemes for PDEs, [1610.05051](https://arxiv.org/pdf/1610.05051);
  asynchronous multirate Taylor with independent local clocks,
  [2606.21044](https://arxiv.org/pdf/2606.21044); high-order conservative local
  time-stepping, [1811.02499](https://arxiv.org/abs/1811.02499).
- Known trap: resonance at rational step ratios.

### 6. Learned closures, distilled, never trusted alone — see Part I.
### 7. Differentiable throughout — see Part I (contact).
### 8. Sparse, hashed, GPU-native — see Part I (MPM at scale).
### 9. Consistency as the correctness criterion

Not "is level n right" but "do levels n and n+1 agree where both apply, to
the observable tolerance". This is how the pool was built: GPU against CPU
(gpu_vs_cpu), far against fine (seam_profile), caustic against flat water.

## What the pool taught, as data for the synthesis

Every seam we fought this week was one of the nine done badly:

- The pool "drained" because integrated J drifted from the particle density —
  a coarse quantity (J) inconsistent with the fine state (positions). Fixed by
  relaxing the coarse toward the fine (technique 9 as a control loop).
- The wall and melon layers packed because the density estimate lost kernel
  support at boundaries — a reconstruction error at an interface (2).
- The caustic square was a 10 cm blend between two surfaces a millimetre
  apart — an energy-inconsistent handoff (3), fixed twice by widening and
  finally by giving the far field the right dispersion.
- The far field lagged the ring because it had the wrong effective theory
  (shallow water for a deep-water wave) — the wrong level for the scale (2).
- The seam's remaining halo is the fine surface's short waves stopping where
  the coarse model cannot carry them — the residual has nowhere to go (2, 3).
