# QNorMuon research instructions for Codex

The mathematical specification of this project is:

    docs/QNORMUON_THEORY.md

Read that document before modifying the optimizer.

The document is a research specification, not assumed to be infallible. If you find a mathematical inconsistency, do not silently modify the implementation to fit it.

Instead:
1. construct a counterexample or failing numerical test;
2. identify the exact theorem/equation involved;
3. explain the issue;
4. propose a corrected statement;
5. update theory and code together only after the issue is understood.

## Primary goal

Develop a gauge-equivariant spectral optimizer for coupled SwiGLU `up_proj` / `down_proj` weights.

## Mathematical invariants are part of the specification

Do not knowingly break:

1. Gauge invariance of canonical coordinates under
   `U -> C U`, `D -> C^{-1} D`, for positive diagonal `C`.
2. Gauge equivariance of lifted updates:
   `DeltaU' = C DeltaU`, `DeltaD' = C^{-1} DeltaD`.
3. Canonical balance:
   `||u_i|| = ||d_i||` after canonical gauge fixing.
4. Spectral constraint:
   `P^T P ≈ I` for the tall-matrix reference setting.
5. Paired leverage target:
   `||P_u[i]||^2 + ||P_d[i]||^2 -> 2n/m`.

## Development rules

- Run the full mathematical test suite before and after each substantial change.
- Do not replace the exact SVD polar reference with an approximate method until an explicit comparison test exists.
- Keep a correct exact/reference path even after fast kernels are added.
- Separate changes to optimizer theory from performance optimizations.
- Prefer invariant-preserving derivations over heuristics.
- Add tests for every claimed mathematical property.
- Report numerical errors quantitatively, including dtype and matrix shape.
- Treat rank-deficient matrices, zero/near-zero row norms, mixed precision, and distributed semantics as explicit research cases rather than silently masking them.
- `gate_proj` is not part of the current positive-diagonal gauge pair because SiLU is not positively homogeneous.

## First task

Read `docs/CODEX_FIRST_TASK.md` and execute that audit before redesigning the optimizer.
