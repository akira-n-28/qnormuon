# QNorMuon / Quotient Spectral Optimizer

Research code for gauge-invariant spectral optimization of coupled SwiGLU
parameters.

The current core direction is a **K=I coupled horizontal spectral optimizer**
defined on the quotient geometry induced by the positive diagonal rescaling

    U -> C U
    D -> C^{-1} D.

Rather than applying spectral normalization independently to the up/down
matrices, the current method solves a coupled spectral linear minimization
problem on the horizontal tangent space of the quotient.

## Current status

This is an active research project.

The current production-v0 candidate combines:

- balanced gauge-canonical coordinates;
- canonical paired momentum;
- a quotient spectral norm;
- a coupled horizontal spectral LMO;
- a nuclear-norm dual problem;
- warm-started row-whitened Newton-CG;
- primal-dual certification;
- explicit fallback when the smooth solver is unreliable.

The current production candidate uses `K = I`.

The following ideas are intentionally not part of production-v0:

- shared K weighting;
- leverage balancing;
- contribution balancing;
- neuron birth;
- custom approximate polar kernels;
- distributed optimizer logic.

## Mathematical core

In balanced canonical coordinates, define

    L(P)_i =
        <U_i, P_U[i]> - <D_i, P_D[i]>.

The horizontal tangent space is

    H = ker L.

The quotient spectral norm is

    ||P||_Q =
        max(||P_U||_2, ||P_D||_2).

The corresponding coupled LMO is

    maximize
        <A_U, P_U> + <A_D, P_D>

    subject to
        ||P_U||_2 <= 1
        ||P_D||_2 <= 1
        L(P) = 0.

Its dual is

    min_lambda
        ||A_U - diag(lambda) U||_*
        +
        ||A_D + diag(lambda) D||_*.

In the smooth full-column-rank regime, dual stationarity is exactly the
horizontal condition.

At deficient rank, primal recovery is a joint subgradient problem and separate
partial polars are not generally sufficient.

## Theory documents

Read the theory in this order:

1. `docs/QUOTIENT_SPECTRAL_GEOMETRY.md`
2. `docs/HORIZONTAL_SPECTRAL_LMO.md`
3. `docs/DUAL_SOLVER_STUDY.md`
4. `docs/ZERO_STRATUM_GEOMETRY.md`
5. `docs/RANK_DEFICIENT_THEORY.md`
6. `docs/THEORY_AUDIT.md`
7. `docs/QNORMUON_THEORY.md`

Later reports supersede older statements where the design evolved.

`docs/QNORMUON_THEORY.md` is the original research specification and should be
treated as historical context rather than the final current contract.

## Reference implementations

Research/reference implementations live under `experiments/`, including:

- `experiments/dual_solver.py`
- `experiments/horizontal_spectral.py`
- `experiments/quotient_spectral.py`
- `experiments/rank_deficient.py`
- `experiments/zero_stratum.py`

Production code belongs under `qnormuon/`.

## Tests

Run the full suite with:

    python -m pytest -q

The mathematical suite contains adversarial tests for:

- positive gauge equivariance;
- canonicalization;
- quotient geometry;
- rank-deficient residuals;
- zero momentum;
- horizontal spectral optimization;
- nuclear-norm duality;
- primal-dual certification;
- Newton-CG derivatives;
- solver fallbacks;
- finite-precision behavior.

Do not remove failing adversarial tests merely to make a redesign pass.

## Numerical solver

The leading research solver for production-v0 is a warm-started,
row-whitened Newton-CG method applied to the dual problem.

A small fixed iteration budget is only an initial budget.

A returned step must be accepted using numerical diagnostics such as:

- primal-dual gap;
- horizontal residual;
- spectral feasibility;
- residual conditioning.

Near rank loss, a very small objective gap does not necessarily imply an
accurate update direction.

## Zero rows

The exact quotient theory currently applies to the regular domain

    ||u_i|| > 0
    ||d_i|| > 0.

Exact zero-weight rows belong to a singular stratum.

Production-v0 must not pretend that an absolute epsilon clamp is an exact
extension of the theory.

Neuron birth is a separate research problem and is not part of the current
optimizer.

## Training plan

The next experimental stage is a small decoder-only SwiGLU Transformer.

Relevant comparisons include:

- AdamW;
- Muon with a conventional optimizer for unsupported parameters;
- historical separate-polar QNorMuon;
- coupled K=I Quotient Spectral Optimizer.

Each optimizer should receive its own learning-rate sweep.

Important measurements include optimization quality, wall-clock cost, solver
iterations, fallback frequency, primal-dual gap, horizontal residual and GPU
memory usage.

## Compute environment

GPU experiments are intended to run on the Lagrange cluster through SLURM.

Operational and infrastructure requirements are documented in `AGENTS.md`.

Large datasets, model weights and checkpoints must already be available on
cluster-visible storage before compute jobs are launched.

Training code must not rely on implicit downloads at runtime.

## Research discipline

Keep separate:

- mathematical theorem;
- numerical verification;
- toy experiment;
- tiny-model result;
- large-scale empirical result;
- conjecture.

Do not claim superiority over existing optimizers without experimental evidence.