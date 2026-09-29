# Smooth residual Gram/EVD backend study

## Exact-arithmetic contract and derivation

This study concerns the **smooth dual residual** only. Production currently uses
fp64 thin SVD for each full-column-rank residual `B` in `SmoothDual.evaluate`.
The cached factors supply its singular values, nuclear norm, residual rcond,
polar `Q`, dual gradient `-L(Q_U,Q_D)`, polar derivative, and matrix-free
Newton-CG Hessian products. The projected-primal radial Gram upper bound is a
separate, already adopted certificate backend and is unchanged here.

For `B` of shape `[m,n]`, `m>=n`, full column rank, set `C=B.T@B`. In exact
arithmetic, symmetric EVD gives `C=V diag(lambda) V.T`, with positive
`lambda_j`. Sort descending and set `sigma_j=sqrt(lambda_j)`,
`U=B V diag(1/sigma)`, and `Vt=V.T`. Then `U.T U=I`,
`B=U diag(sigma) Vt`, and the unique polar is `P=U Vt`.
This is precisely the thin-SVD information used by the current code. Nuclear
value is `sum(sigma)` and rcond is `min(sigma)/max(sigma)`. Repeated positive
singular values can rotate the columns of `U,V` together without changing
these invariant quantities.

Equivalently write `B=P H` with `H=V diag(sigma) V.T`. Differentiating gives
`E=(DP)H+P(DH)`. From `P.T P=I`, `Omega=P.T DP` is skew; the tangent equation
`H Omega+Omega H=P.T E-E.T P` uniquely determines it because every
`sigma_i+sigma_j>0`. The normal part is
`(I-P P.T) E H^{-1}`. Hence

```
DP[E] = P Omega + (I-P P.T) E H^{-1}.
```

In the `V` basis let `F=U.T E V` and `N=E V-U F`. Then
`(V.T Omega V)_{ij}=(F_ij-F_ji)/(sigma_i+sigma_j)` and the normal term is
`N diag(1/sigma) V.T`, exactly the current SVD-factor derivative. Applying
this derivative to `E=-L_eff*(v)` yields the existing matrix-free HVP
`-L_eff(DP[E])`. Gram and SVD therefore define the same smooth mathematical
objective and derivatives **in exact arithmetic**. Finite precision, especially
formation of `B.T@B` and division by small `sigma`, is the subject of the
remaining study.

## Finite-precision model and posterior screen

The diagnostic prototype is [smooth_gram.py](../experiments/smooth_gram.py).
It scales `B` by its largest absolute entry before forming a Gram matrix,
symmetrizes the matrix, uses fp64 `eigh`, orders eigenpairs, and reconstructs
left factors without truncating positive eigenvalues. Any negative computed
eigenvalue, even one classed as roundoff-sized, rejects the Gram attempt: a
usable inverse square root cannot be obtained by silently clamping it. The
reason code distinguishes a roundoff-sized negative from one beyond the Gram
error allowance. Failed attempts retain the existing full-SVD route in the
diagnostic adaptive replay. Production is never patched on disk.

Let `u=2^-53`, `gamma_k=ku/(1-ku)`, `s=max(abs(B))`, and `X=fl(B/s)`.
The standard dot-product model gives an operator-norm allowance for the true
scaled Gram matrix of approximately
`gamma_m ||X||_F^2 + u ||G||_F`, plus a bound for the divisions defining `X`.
These are deliberately inflated from measured Frobenius reductions. For
computed eigenvectors `V`, eigenvalues `Lambda`, `delta=||V.T V-I||_F`,
and `rho=||G V-V Lambda||_F`, the exact identity

```
G-V Lambda V.T = G(I-V V.T) + (G V-V Lambda)V.T
```

bounds its norm by `||G||_F delta+rho sqrt(1+delta)` when `delta<1`.
Weyl's inequality and the near-orthogonality of `V` then give conservative
**model-based** intervals for every positive squared singular value. Taking
square roots, allowing scaled-division error, and dividing the minimum lower
bound by the maximum upper bound gives a lower bound on residual rcond.
The diagnostic rejects Gram unless that lower bound is strictly greater than
the unchanged `1e-4` guard. It also records an upper rcond bound, singular
interval width and nuclear-value interval. These intervals are conditional on
the standard fp64 operation model, not formal interval arithmetic or a
vendor-specific guarantee about every GPU reduction.

The posterior additionally measures eigensystem orthogonality/residual,
reconstructed-left orthogonality, reconstruction residual, polar
orthogonality, and polar reconstruction residual. The reconstruction envelope
accounts for the two `n`-term products, the division/multiplication, the
measured eigenvector orthogonality defect, and norm-reduction rounding. It is
an operation-count screen, **not a proved HVP error bound**. This distinction
is decisive: passing the scalar/rcond posterior does not yet certify Newton
directions or line-search decisions over the entire accepted smooth domain.

## Corpus, independent oracle, and observed errors

The H100 driver [study_smooth_gram_h100.py](../cluster/study_smooth_gram_h100.py)
uses the existing offline SLURM environment. It compares against independent
fp64 thin SVD, never Gram against itself. The deterministic corpus includes
Gaussian and prescribed-spectrum `[24,8]`, `[96,24]`, and `[1024,384]`
matrices; repeated and clustered extremes; ratios from `1` through `5e-5`
with dense coverage just above/below `1e-4`; nearly rank-one and wide-dynamic
cases; extreme representable overall scales; random orthogonal orientations;
and all residual sides encountered in one saved warm Transformer step. The
final corpus contains 95 matrix cases. Perturbations include random,
tangent-like, normal-like, smallest- and largest-direction aligned, and
clustered-subspace aligned matrices. Full-SVD polar derivatives and centered
finite differences on small shapes are independent checks. Finite differences
are not informative when the true derivative is near zero: relative error can
explode although absolute error is tiny.

**H100 result (job 28959):** 80/95 Gram attempts passed the research
posterior. Nine rejected because the conservative rcond lower bound was not
above `1e-4`; six produced a negative eigenvalue classified as roundoff and
were rejected without truncation. All 36 actual warm-step residual matrices
passed. Among passing cases, 19 had independent SVD rcond at or below
`3e-4`; no rcond lower bound exceeded its SVD oracle and there was no false
above-guard classification. This finite corpus does not prove a
backend-independent one-sided guarantee.

Across accepted cases, the largest relative singular-value error was
`7.08e-9`, nuclear-value error `3.69e-13`, polar Frobenius error `2.60e-9`,
and absolute rcond error `7.66e-13`. The worst reconstructed-left and polar
orthogonality defects reached `1.47e-8`. On random perturbations the worst
relative polar-derivative error was `7.54e-9`; on tangent-like perturbations
it was `2.33e-6`, attained close to the rcond guard. Some aligned directions
are in or near the derivative's null space. Their reference derivative norms
are so small that relative errors become huge despite small absolute errors;
centered finite differences there are also dominated by subtraction noise.
The script preserves absolute and relative derivative errors. On the real
warm residuals, worst polar and random-perturbation derivative relative errors
were `2.86e-12` and `6.89e-12`. A scalar rcond screen alone is thus not a
posterior guarantee for all HVPs near the permitted boundary.

The [adversarial JSON](../cluster/smooth_gram_adversarial-28959.json) contains
every case, posterior measurement, oracle error, perturbation result, and
synchronized timing. An initial overly narrow reconstruction screen rejected
11 benign matrices in job 28954; its envelope was corrected by accounting
for both reconstruction products and measured eigenvector defect. The
correction followed operation counts, not a fit to the real pairs. Corrected
jobs 28957/28959 agree on 80/95 accepts. Production rules did not change.
The first full-suite attempt, job 28956, exposed the same over-restrictive
screen in three new diagnostic test cases (284 passed, 3 failed); the test
traceback was retained. The revised screen passed all tests in jobs 28958
and 28960 without relaxing any production certificate threshold.

## Newton decisions and locked shadow replay

The diagnostic adaptive replay reconstructs the same six step-25 warm QSO
pair inputs from the locked checkpoint, then solves each independently with
full SVD and with the experimental Gram-or-full-SVD decomposition. It compares
every evaluation coordinate, objective, gradient, rcond, trial count,
certificate decision, next multiplier, and final direction. CPU ADMM is
disabled. This comparison tests actual Newton and line-search branches but
does not prove they remain identical at every possible residual.

All **six** warm pairs certified after two Newton iterations with both
diagnostic paths. All 18 evaluated residual *pairs* passed the research
posterior, so no diagnostic decomposition fallback occurred. Evaluation and
line-trial counts were `[3,3]` for each pair; every certificate and
line-search branch agreed. Maximum matching-trial coordinate difference was
`3.31e-15`, dual-value difference `1.10e-12`, gradient-component difference
`3.74e-13`, and rcond difference `9.14e-16`. Final normalized-gap difference
was at most `8.61e-14`, and relative direction difference at most `2.01e-12`.
Both paths had zero CPU ADMM use. This covers a real warm state and all of
its trials, but not a near-guard Newton decision adversary.

A separate SVD-controlled shadow replay covers all 50 locked steps. It
evaluates Gram alongside each smooth residual, returning only the unchanged
full-SVD evaluation to the optimizer. The baseline data/batch hashes and losses
match the saved run, so the shadow did not affect parameters or optimizer
state. Job 28955 observed **931 smooth evaluations**, all 931 passing the
research posterior, with no decomposition rejection; the minimum model-based
rcond lower bound was `9.77833895e-4`. Maximum relative polar difference was
`3.21e-12` and maximum relative HVP difference along one deterministic row
perturbation was `4.85e-13`. The maximum gradient-relative difference was
`1.88e-7`, which occurred with a small denominator; gradient-absolute error
was not logged in that run. No shadow result controlled a line search or
certificate. The shadow used the earlier, stricter reconstruction envelope;
the final study only relaxed that envelope via the derived operation count and
added uncertainty reporting, so all 931 would still pass its gate. Raw
per-evaluation data: `cluster/smooth_gram_shadow-28955.json`.

## Synchronized H100 cost and production decision

On a warmed real two-matrix residual pair, current full thin SVD took
`43.36 ms` median (p95 `43.38 ms`); raw Gram GEMM/EVD took `7.73 ms`
(p95 `7.73 ms`); Gram with the current posterior took `10.59 ms`
(p95 `10.60 ms`). For one real matrix, measured median Gram formation and
symmetrization took `0.10 ms`, EVD `3.90 ms`, and factor
reconstruction/posterior checks `1.07 ms`. Component timers include
synchronization and need not sum exactly to independently warmed pair medians.
The shadow replay had zero diagnostic decomposition fallback, but a near-guard
or deficient workload could lose some speed to full-SVD fallback. Its warm
median smooth-SVD time was `0.782 s` (p95 `0.828 s`), while concurrent
research Gram work took `0.191 s` (p95 `0.202 s`) per step. This is a
potential roughly `4.1x` decomposition
speed difference at the measured acceptance rate, not a production speedup:
the optimizer still used SVD and paid for *both* paths in shadow mode.

**Decision: keep full fp64 SVD as the production smooth default.** The
posterior reliably screened the tested rcond boundary under its stated model
and matched all real replay decisions, but the current checks do not certify
polar-derivative or Newton-direction accuracy in the permitted near-guard
region. Adopting the backend based on real-state speed alone would violate
the numerical contract in `AGENTS.md`. There is no new `smooth_backend`
production switch, no smooth decomposition fallback in production, and no
locked 50-step candidate-backend validation. The separate projected-primal
`gram_upper` backend remains production default for radial feasibility only.
All existing certificate thresholds, CPU ADMM fallback semantics, and the
`1e-4` guard remain unchanged.

An eventual adaptive design would attempt Gram, require its conservative
rcond lower bound and posterior structural checks, then require a defensible
bound tied to the HVP/CG and line-search decision margins. Any failure would
use the current fp64 full SVD with an explicit decomposition reason. That
last derivative/decision bound is unresolved, so no universal safe region
above `1e-4` has been established for the **whole solver**, despite a
conditional rcond-safe region for the singular spectrum. Full SVD remains the
independent oracle and the dominant production cost, about `0.77 s` of a
`1.02 s` warm QSO step. No LR sweep or approximate decomposition was run.

## Verification and limits

The H100 suite includes new small diagnostic regressions for prescribed
spectra on both sides of the guard, repeated extremes, large and small
overall scales, negative computed eigenvalue rejection, independent SVD
polar/nuclear/HVP comparisons, and an informative centered difference.
The final complete suite, job 28961, collected **287 tests: 287 passed,
0 failed, 0 skipped** in 10.16 s of pytest execution. Production fp32- and
bf16-parameter smoke checks still used direct fp64 pair solving, fp32 momentum,
the unchanged `3e-5` gap target, and no fallback. The test job recorded no
Python outbound-network attempt. Raw output is
[run_production_tests_h100-28961.log](../cluster/run_production_tests_h100-28961.log).

This study does **not** establish a backend-specific rigorous HVP-error,
Newton-direction, or Armijo-margin bound over the whole smooth domain. The
spectral intervals depend on the standard fp64 reduction model and measured
posterior quantities; they are not interval arithmetic. The real shadow
states remain well above the rcond guard and cannot replace an adversarial
near-guard trajectory. The answer to production adoption is therefore no,
despite promising speed and exact-arithmetic equivalence. A useful next step
is a decision-relevant derivative bound or different posterior validation
that does not itself require full SVD on every accepted attempt.

## Answers to the production questions

1. **Exact arithmetic:** yes, Gram EVD reconstructs the same thin-SVD polar,
   spectrum, nuclear objective, gradient and derivative for full-column-rank
   residuals.
2. **Finite precision:** squaring the condition number, uncertainty in small
   eigenvalues, reconstructed-left nonorthogonality, and amplified derivative
   error matter near the retained guard.
3. **Posterior detection:** tested structural and rcond failures are detected,
   but current checks do not bound every HVP or Newton decision tightly enough
   for a production claim.
4. **Safe region:** a conditional model-based lower rcond bound screened the
   tested guard boundary. No whole-solver safe region above `1e-4` is yet
   demonstrated.
5. **Polar/HVP:** real residual errors are tiny; permitted near-guard cases
   show up to `2.60e-9` polar and `2.33e-6` informative tangent-like
   derivative relative error.
6. **Newton/line search:** all six saved warm-pair replays agreed on every
   observed branch, but near-guard branch stability remains unproved.
7. **Real acceptance:** 931/931 smooth evaluations passed the research
   posterior in the unchanged 50-step shadow; no decomposition fallback.
8. **Cost:** validated Gram took `10.59 ms` versus `43.36 ms` for a warmed
   residual pair, about `4.1x` faster in isolation; shadow mode saved no
   production time because SVD still controlled the solver.
9. **Default:** keep full fp64 SVD. No `gram_validated` production default or
   production switch was introduced.
10. **Remaining bottleneck:** smooth full fp64 SVD/polar, about `0.77 s` per
    warm six-pair QSO step.
