# Synthesis — a working draft

*Written 2026-09-02 as a seed for a conversation, not a conclusion. Each
section ends with the questions we have to answer together. See
research.md for the literature every claim leans on.*

## The claim

A world is a stack of effective theories. The engine's job is not to run the
finest one everywhere; it is to run, at every point, the coarsest theory
that still answers the question being asked, and to keep every pair of
adjacent theories consistent where both apply. Everything else is
engineering in service of that.

Five consequences, each a design rule:

1. **The coarse state is the world.** The default representation of a region
   is its coarsest adequate model: a spectral wave field for a pool, rigid
   bodies for furniture, a radiance field for a hall. Fine physics is a
   *residual* that exists only where the coarse model is wrong for the
   current question, and retires when it stops being wrong. There is no box.
2. **Handoffs are ports.** Every exchange between levels is written as a
   power-conserving port (port-Hamiltonian), so switching representation
   never creates or destroys energy, momentum, mass, or charge. Seams that
   are ports do not show.
3. **Reconstruction is sampling.** Going from coarse to fine is generative:
   sample a fine state from the distribution conditioned on the coarse state
   and on everything already observed, energy-guided, then cache it so
   re-observation agrees. Going fine to coarse is the RG step; reconstruction
   is the RG run backwards.
4. **The question chooses the level.** An adjoint of the quantity of interest
   — a rendered image, an agent's decision, a measurement — says where the
   coarse model's error matters. Refinement follows that field, in space and
   in time (each region on its own clock).
5. **One autodiff runs through all of it.** The adaptivity estimator (4), the
   energy-guided reconstruction (3), the learned couplings (2), and the
   observation-consistent sampling (3) all consume gradients of the same
   system, which is what makes the composition possible at all. The exact
   tier is teacher and fallback for every learned piece.

## What we already have, mapped

| rule | in Kosm today | gap |
|---|---|---|
| coarse state is the world | spectral far field in (η, ψ) with H conserved to 1e-13, 2nd-order Zakharov terms conserving H₂+H₃ to 8e-5; rigid melon; height-field surface; block-sparse MPM grid (dense block table, indirect dispatch; 11% tax at a full box) | the region is a disc of particles on a grid that spans the pool (dbcea1f): radial sponge, radial no-outflow wall, zero-gradient density beyond it, integer circle test; still fixed in place and pre-filled — spawn/retire is the next step |
| ports | nudging in a blend band with its energy booked (Far::injected); sponge; reaction booked at grid incl. projection impulses; hydrostatic force 1.03× Archimedes at 2.5 cm | the nudge is not power-conserving; Water::energy exists but its EOS term is fictitious while J is not derived from positions (drift 1e-2 of potential per 0.4 s in every variant measured); melon neutral, not afloat (dynamic dissipation) |
| reconstruction | particles spawn from the far field with J=1; rest map | no generative reconstruction; no caching of samples |
| question chooses level | none (box is fixed) | need the adjoint of render/decision w.r.t. model choice |
| one autodiff | tang through phyz (marble adjoint), reference renderer differentiable in principle | MPM step has no adjoint yet; live tier not differentiable |

## The smallest system that exhibits all five

The pool, again, but restated:

- State: (η, ψ) over the whole surface (Zakharov), rigid bodies, and a
  *set of active fine regions* that is empty when the pool is calm.
- Residual spawn: a nonlinearity indicator (steepness, curvature, body
  proximity, and eventually the coarse model's own error estimate) allocates
  sparse MPM blocks around the event; particles are sampled from the coarse
  state (velocity from ψ, J=1 for now; learned later).
- Port: the exchange between MPM and (η, ψ) written so that
  d/dt(H_wave + H_mpm) = −dissipation exactly, as in Zhu's coupling.
- Retire: when the residual's energy relative to the coarse model drops below
  the observer's tolerance, its mass and momentum go back through the port
  as a mollified source, and the blocks free.
- Test: a flat pool stays flat to < 1 mm as the indicator is dragged across
  it; a ring crossing a spawn boundary conserves energy to < 1%; the melon
  makes a residual that is gone in seconds.

## Open questions (for us)

1. **What is the indicator?** Steepness and body proximity are obvious and
   wrong in general. The principled one is goal-oriented: the adjoint of the
   image (or the agent's value) with respect to the coarse model's residual.
   Do we start with the heuristic and replace it, or build the adjoint first
   and let it teach us the heuristic?
2. **What is the coarse state for things that are not water?** Rigid bodies
   are their own coarse model; deformables and granular media need one
   (modal? reduced-order? learned?). Is the answer "a learned pH system per
   object class, distilled from MPM"?
3. **Where does the sample live?** Reconstruction must be cached so that
   looking twice agrees. Is the cache the run file (deterministic seeds per
   region and time), and does that make the run file the world's memory?
4. **Time.** Multirate variational integration across levels is known; doing
   it with spawn and retire events in the middle is not. Do events happen
   only at macro steps?
5. **Rendering as a level.** Is the live raster tier just the coarsest
   effective theory of light, with path-traced residuals where the adjoint
   says the raster is wrong (caustics, speculars)? That would make the
   renderer obey the same five rules as the physics.
6. **The paper.** Is the first paper the pool (energy-consistent
   differentiable MPM–spectral coupling with residual spawn/retire), or the
   framework? The pool is provable and demoable this quarter.

## Order of work if we accept the claim

1. Hamiltonian far field (η, ψ) and an energy for the MPM state.
2. Residual spawn/retire on sparse blocks, heuristic indicator, flatness and
   energy tests.
3. The port: energy-consistent exchange; the halo test.
4. Adjoint of the MPM step; gradient checks.
5. Goal-oriented indicator from the renderer's adjoint.
6. Generative reconstruction distilled from the exact tier; cached samples.
7. The same five rules applied to light.
