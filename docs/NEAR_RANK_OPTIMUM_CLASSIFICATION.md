# Stage-D fixed-fixture optimum classification

**Overall classification A: FULL-RANK-BELOW-GUARD, for all five saved fixtures.**
High-accuracy stationarity and independently matched primal/dual values support
full-column-rank optima, with positive singular tails far above fp64
decomposition uncertainty. Every fixture's down residual remains below the
unchanged production `1e-4` admission guard at the optimum. Low and middle
fixtures also have up residuals below it. The evidence does not require a
deficient joint subgradient completion.

This is a numerical classification of fixed stored problems, not a formal
interval existence proof or a statement about all training trajectories.
Production code, defaults, certificates, fallback, and guard are unchanged.
No model was instantiated or trained; no data, packages, or artifacts were
downloaded. The Stage-D quality comparison remains inconclusive.

## Inputs and isolated research contract

Only the five `failed_pair.pt` files documented in
[STAGE_D_QSO_FAILURE_FORENSICS.md](STAGE_D_QSO_FAILURE_FORENSICS.md) were used:

| Fixture | Saved capture/run | Original failure step / pair |
|---|---|---|
| Tuned confirmation | capture-29044/confirmation-qso-a0.0003-q0.00122474487139 | 87 / blocks.1.mlp |
| Low | capture-29045/coarse-qso-a0.0003-q0.0003 | 92 / blocks.0.mlp |
| Middle | capture-29045/coarse-qso-a0.0003-q0.001 | 87 / blocks.0.mlp |
| High | capture-29045/coarse-qso-a0.0003-q0.004 | 56 / blocks.0.mlp |
| Ablation | capture-29045/ablation-qso-a0.0002-q0.00122474487139 | 81 / blocks.0.mlp |

The root is `/home/prignano/qnormuon-runs/stage-d-forensics/`. Each problem
contains balanced canonical U/D, the updated fp32 EMA objective promoted to
fp64, and the original-coordinate warm lambda. Both sides are `[1024,384]`.
These exact stored tensor values define the classification problem; fp32
storage is not treated as a license to truncate their positive fp64 spectra.

[near_rank_optimum.py](../experiments/near_rank_optimum.py) creates a separate
research configuration with the smallest positive float as the admission
guard, `fallback=False`, `tolerance=1e-10`, and independent residual
certification. Thus every computed positive residual spectrum is admitted;
zero singular values are not. This is **not production-safe admission**. The
production Newton/HVP, damping, 24-trial globalization, full fp64 SVD, objective,
and certificate formulas are otherwise unchanged. The projected-primal backend
remains `gram_upper` during continuation.

Starts were the captured warm lambda, current vertical-centering beta, and
original-coordinate zero. The tuned case additionally used the saved partial
CPU ADMM multiplier, adding captured beta to its centered multiplier. No ADMM
was executed in this study. Complete smooth evaluations, trial spectra,
gradients, slopes, CG traces, line-search decisions, accepted histories, and
internal coordinates are retained outside source storage.

## High-accuracy stationarity and independent solve

Guard-free continuation sharply reduces the gap while preserving positive
tails. All 16 production continuations eventually report `line_search_failed`
at the much stricter research target. This is **not evidence of deficiency**:
some endpoints already have independent SVD-radial gaps around `1e-13`, while
others stall at approximately `1e-8`–`2e-7`. Rounded nuclear values limit
globalization, and conservative `gram_upper` overscaling has a roughly `2e-10`
gap floor here, above the attempted `1e-10` stopping target. Neither limitation
is a change to the production `3e-5` target.

A genuinely different optimization method was therefore used: safeguarded
limited-memory BFGS, started independently at beta for every fixture. It uses
complete fp64 SVD values and gradients, but **no Newton directions or HVPs**.
While predicted objective decrease is resolvable it uses value Armijo; at the
rounding plateau it performs gradient-norm root refinement without trusting
rounded nuclear-value differences. This is an isolated classification method,
not a proposed production globalization change. It stops at internal
row-whitened gradient norm below `2e-12`, with a bounded iteration/trial budget.
Every reported endpoint is subsequently checked in original coordinates.

For each candidate the independent check reconstructs
`B = A - L*(lambda)`, takes its complete thin SVD, forms both unique full-rank
polars, computes `g = -L(P)`, and independently performs horizontal projection
and SVD-radial recovery. No cached dual spectrum is used in this check.

| Fixture | U sigma_min / rcond | D sigma_min / rcond | Original gradient norm | Independent normalized gap | Classification |
|---|---|---|---:|---:|---|
| Tuned | 4.795752958e-5 / 1.295438484e-4 | 3.758054065e-5 / 8.678981341e-5 | 5.3694e-13 | 2.3553e-13 | A |
| Low | 4.270841326e-5 / 9.785839640e-5 | 4.486443989e-5 / 7.741479766e-5 | 5.0276e-13 | 3.7729e-13 | A |
| Middle | 7.522040659e-5 / 6.648727922e-5 | 6.448253003e-5 / 4.489413119e-5 | 2.8893e-13 | 2.3036e-13 | A |
| High | 1.172483816e-4 / 1.152466277e-4 | 1.110285800e-4 / 7.706061263e-5 | 3.3604e-13 | 2.0459e-13 | A |
| Ablation | 1.708873965e-4 / 1.207848648e-4 | 1.603509601e-4 / 8.761157251e-5 | 1.0784e-13 | 1.3729e-13 | A |

The corresponding dual values are 1.245347728531058, 2.072797155936896,
4.129359899737081, 4.488891269524510, and 6.288362664943877. Signed gaps are
positive, with normalized recovered horizontality at most `8.29e-17` and
spectral excess zero. Final independent CPU KKT checks corroborate stationarity;
their gradient norms are at most `5.39e-13`.

At the same solutions the **unchanged production gram_upper certificate** has
normalized gaps `1.993e-10`–`1.995e-10`. All gap, signed-gap, horizontal, and
spectral tests pass the production tolerances; rcond alone prevents production
admission. These are research solutions, not authorized optimizer updates.

## Tuned singular-tail trajectory

The following Newton rows use the unchanged production algorithm with research
admission. Gradient norms here are in its internal row-whitened coordinates;
the final KKT table above uses original coordinates. Gaps in Newton rows use
the production `gram_upper` recovery.

| Progress | Internal gradient norm | Normalized gap | sigma_min U | sigma_min D | rcond D |
|---|---:|---:|---:|---:|---:|
| Warm initial | 7.8721e-1 | 3.7354e-2 | 4.790246e-5 | 3.772471e-5 | 8.712293e-5 |
| Newton 1 | 6.7484e-2 | 7.6351e-3 | 4.796934e-5 | 3.761853e-5 | 8.687755e-5 |
| Newton 2 | 3.4917e-3 | 5.8394e-4 | 4.795627e-5 | 3.758107e-5 | 8.679104e-5 |
| Newton 3 | 1.0947e-4 | 1.2823e-5 | 4.795753e-5 | 3.758055e-5 | 8.678984e-5 |
| Newton 4 | 1.1113e-6 | 1.0554e-7 | 4.795753e-5 | 3.758054e-5 | 8.678981e-5 |
| Newton 6 / strict line-search exit | 5.5560e-7 | 5.2881e-8 | 4.795753e-5 | 3.758054e-5 | 8.678981e-5 |
| Independent L-BFGS 26 | 1.7609e-12 | 2.3553e-13, independent SVD recovery | 4.795753e-5 | 3.758054e-5 | 8.678981e-5 |

The down tail settles after a modest change; it does not collapse toward zero
as stationarity improves. Its final eight smallest values are approximately
`4.309255e-5, 4.285046e-5, 4.234008e-5, 4.175415e-5, 4.070273e-5,
4.018641e-5, 3.870743e-5, 3.758054e-5`. The up tail likewise stabilizes.
The full progress table is
[near_rank_tuned_progress.csv](../cluster/near_rank_tuned_progress.csv).
Trial evaluations, including rejected strict-target trials, are retained in
the continuation JSON; none uses rank truncation.
An additional independent original-coordinate audit covers **all 666 initial
and trial evaluations** across the 16 continuations. It records dual value,
gradient, feasible gap, both spectrum tails and rconds, multiplier displacement,
CG work, slope and line-trial data. Every audited singular tail remains positive;
the minimum tuned down sigma_min over all starts/trials is `3.758014333e-5`.
See [near_rank_all_evaluations.csv](../cluster/near_rank_all_evaluations.csv).
Slope, direction norm, and Armijo RHS are explicitly labeled internal-coordinate
quantities; objective, KKT residual, and multiplier displacement are original
coordinate quantities. This is an offline audit, not additional solver control.

## Rank margin and decomposition independence

The same original residuals were decomposed through:

1. GPU fp64 direct thin SVD;
2. explicit GPU reduced QR followed by square fp64 SVD;
3. independent CPU fp64 LAPACK/MKL SVD, with bundled backend configuration saved.

The two GPU routes produced bitwise-equal factors on these inputs. Thus they
are a required cross-check, **not independent evidence of a distinct CUDA
algorithm**. CPU LAPACK is the independently different decomposition here.
Against GPU, CPU sigma_min relative discrepancies are at most `7.78e-14`,
and polar relative differences at most `1.97e-13`.

Measured GPU relative reconstruction errors are at most `1.31e-13`, left
orthogonality defects at most `1.15e-12`, and right defects at most `2.92e-12`.
CPU reconstruction errors are at most `4.75e-15`, with factor orthogonality
defects around `7e-14` or less.

The rank-separation estimate uses a conservative standard fp64 model rather
than an arbitrary rank epsilon. Let `u = eps64/2` and
`gamma_k = k*u/(1-k*u)`. Inflate measured orthogonality defects using GEMM and
norm-reduction bounds. For a factor X with inflated defect delta below one,
its nearby orthonormal factor Xhat satisfies

```text
||X-Xhat|| <= delta / (1 + sqrt(1-delta)).
```

If `Bhat = Uhat Sigma Vhat.T`, bound `||B-Bhat||` by the measured factorization
residual plus factor-orthogonalization corrections and GEMM/reduction rounding.
The implementation conservatively uses Frobenius bounds to control operator
errors. Weyl then gives
`sigma_min(B) >= sigma_min(computed Sigma) - error_bound` under that model.
Original residual construction separately has the bound
`gamma_3 * || |A| + |L*(lambda)| ||_F` per side. This is negligible relative to
the decomposition bound; frozen tensor inputs are treated as exact stored data.

| Fixture | Max GPU model uncertainty, absolute | Minimum sigma_min / uncertainty across sides |
|---|---:|---:|
| Tuned | 1.429e-11 | 2.631e6 |
| Low | 1.917e-11 | 2.340e6 |
| Middle | 4.741e-11 | 1.360e6 |
| High | 4.756e-11 | 2.335e6 |
| Ablation | 6.032e-11 | 2.658e6 |

The estimate is intentionally loose. It assumes ordinary fp64 GEMM/reduction
rounding and is not outward-rounded interval arithmetic or a backend-specific
proof. Nevertheless, the six-order rank separation, decomposition agreement,
stationarity, and value equality jointly make numerical deficiency an
unsupported interpretation of these optima. A relative singular value of
`4.49e-5` is below the admission guard, not numerically zero.

## Multiple starts, uniqueness, and deficient faces

All 16 continuation endpoints were independently refined by the L-BFGS
stationarity method and compared with each fixture's separate beta-start
solution. Across all starts, absolute lambda discrepancies are at most
`6.85e-15`, recovered direction relative differences at most `2.88e-13`.
They agree in objective, residual rank, and tails. The tuned ADMM start reaches
the same solution. Unrefined strict-line-search endpoints have larger direction
differences, consistent with their remaining gradient errors; they are not
different optima.

**Dual multiplier uniqueness is not proved.** Multi-start agreement is evidence
of an isolated numerical solution, not a uniqueness theorem. A chunked
Gershgorin lower bound on the Hessian's normal-component coercivity was tested;
it was nonpositive in every fixture and therefore inconclusive. No dense
production-sized Hessian was formed. The multiplier may in general be nonunique.

In contrast, at an exact full-rank KKT point both nuclear-norm subgradients are
unique polars. Primal-dual equality forces every optimal primal into those
singleton exposed faces, so the coupled primary primal is unique. This uses
the **current coupled theory** in
[HORIZONTAL_SPECTRAL_LMO.md](HORIZONTAL_SPECTRAL_LMO.md), not the historical
single-matrix partial-polar rule. The recovered numerical solutions support
that smooth regime here. No deficient KKT point was found or required; no
positive singular value was set to zero to manufacture one.

Consequently joint null-block completion, deficient-face minimum-Frobenius
selection, and finite-epsilon lexicographic solves were not run. Their
conditional prerequisite—evidence for a deficient primary optimum—is absent.
The singleton full-rank primary solution already supplies its secondary
selection. This does not exclude a different-rank dual representation without
a dual uniqueness proof; it does show that deficient primal recovery is not
needed to solve these fixed problems.

## Direction sensitivity

Sensitivity checks keep lambda fixed unless explicitly perturbing it; they do
not re-optimize the perturbed problem or certify a uniform admission rule.
Random deterministic perturbations use relative norms eps64, eps32, and 1%.
Residual perturbations and additive objective perturbations at fixed lambda
are algebraically the same operation, not independent corroboration.

| Fixture | Random A/residual eps32: relative polar change | Smallest-singular normal eps32: worst relative polar change |
|---|---:|---:|
| Tuned | 1.7535e-5 | 4.9669e-5 |
| Low | 2.0767e-5 | 5.5771e-5 |
| Middle | 3.2928e-5 | 9.5929e-5 |
| High | 1.9395e-5 | 5.5921e-5 |
| Ablation | 1.7466e-5 | 4.9195e-5 |

At eps64, measured polar changes stay below `2.41e-13`. At eps32, random
relative lambda perturbations produce changes `7.13e-9`–`1.95e-8`.
Adversarial normal perturbations along the smallest right-singular direction
expose larger amplification than random perturbations. Shrinking only a
positive singular value changes the spectrum but barely changes the polar,
as expected while its sign/rank remain unchanged. All perturbation outputs
include objective changes, spectra, gaps, and recovered direction changes in
the structured summary.

One-percent random objective perturbations produce substantial relative polar
changes, `0.756`–`0.924`: these are sensitive problems compared with ordinary
well-conditioned random matrices, despite stable fp64 decomposition. A
scale-only objective jump gives a different response and is not a replacement
for the actual temporal orientation change.

To relate sensitivity to the observed adjacent-step objective changes, the
previous EMA was *estimated* from saved gradients and the documented recurrence
`M_new=.95*M_old+.05*G_c`. Inverting its fp32 multiply/add roundings cannot
recover the exact old state. The estimated change has norm about the entire
new objective, corroborating the earlier 17–94x magnitude-jump observation.
Perturbations along this estimated temporal direction, at eps32, 1%, and 100%
of the change, are recorded separately and explicitly labeled estimates.
They are not regenerated training trajectories. No production sensitivity
threshold is proposed from these diagnostics.

## CPU reference evidence

The saved tuned partial ADMM iterate from the prior bounded experiment has
dual value about `1.245347728538275`, only `7.22e-12` above the independent
high-accuracy smooth result, with D rcond about `8.67889e-5`. Using its properly
uncentered multiplier as a start converges to the same full-rank solution.
This is corroborating evidence, not authority for rank or secondary selection.
The unchanged reference previously failed to return within 240 seconds; it was
not rerun or redesigned. Its stopping delay does not establish a nonsmooth face.

## Validation and artifacts

All tensor work ran on H100 under SLURM, one GPU, four CPUs, 16 GiB per job.
Runtime: Python 3.10.20, torch 2.10.0+cu128/CUDA 12.8, NVIDIA H100 NVL.
TF32 is disabled, deterministic algorithms enabled. The complete suite passed
**407 tests, zero failed, zero skipped**, including four new classification
controls, in **15.990 seconds**. Controls cover full-rank stationary data below
the guard, unchanged production rejection, independent nonstationary-start
refinement, rank-margin evaluation, and refusal to substitute a full-rank
completion for a deficient coupled face. Network audit logs are empty.

Research outputs are under
`/home/prignano/qnormuon-runs/near-rank-optimum/study-29066/`:

- initial multi-start continuation JSON and coordinate/checkpoint tensors;
- `refinement-29068/`: independent solutions, decomposition/sensitivity checks;
- `consistency-29069/`: polished cross-start checks and temporal sensitivity;
- `refinement-29068/final_audit.json`: production-radial, CPU KKT, adverse checks.
- `all_evaluation_audit.json`: independent metrics at every continuation evaluation.

Raw logs: [continuation](../cluster/near_rank_optimum-29066.log),
[independent refinement and tests](../cluster/refine_near_rank_optimum-29068.log),
[cross-start consistency](../cluster/check_near_rank_consistency-29069.log),
[final audit](../cluster/final_near_rank_audit-29070.log),
[complete trial audit](../cluster/audit_near_rank_trials-29071.log).
Compact evidence is
[near_rank_optimum_summary.json](../cluster/near_rank_optimum_summary.json);
test results are [JUnit XML](../cluster/near-rank-tests-29068.xml).
The first exploratory refinement log is retained separately; its rounded-value
stall in the tuned case was resolved by the explicitly documented independent
gradient-root refinement, not hidden or classified as rank loss.

## Explicit answers and next research question

1. **Tuned optimum:** evidence supports a full-rank smooth optimum below guard,
   classification A. All five fixtures receive A; overall A, not mixed.
2. **Tuned best spectrum:** U sigma_min `4.795752958e-5`, rcond `1.295438484e-4`;
   D sigma_min `3.758054065e-5`, rcond `8.678981341e-5`.
3. **Tail behavior:** stabilizes at positive values as stationarity improves;
   does not systematically collapse to zero.
4. **Independent decompositions:** CPU LAPACK agrees with GPU tall/QR-SVD
   spectra and polars far below relevant rank margins. GPU routes coincide
   bitwise here and are not claimed as independent implementations.
5. **Starts:** warm, beta, zero, and tuned saved ADMM all recover the same primal
   after high-accuracy refinement, within `2.88e-13` relative difference.
6. **Dual uniqueness:** numerically consistent, but not certified; the tested
   coercivity lower bound is inconclusive. Primal uniqueness follows at a
   full-rank exact KKT point, independently of dual uniqueness.
7. **Deficient completion:** no evidence makes it necessary; separate partial
   polars were never used to solve the coupled problem.
8. **Secondary selection:** no deficient face is supported, so no null-block
   P-dagger computation is warranted. The smooth primary solution is unique.
9. **Sensitivity:** fp64 perturbations are small; eps32-scale objective changes
   amplify to roughly `1e-5`–`1e-4` relative polar changes. Larger temporal
   changes are materially different problems; no uniform safety claim follows.
10. **Present guard:** blocks a numerically rank-separated smooth solution on
    these fixtures, rather than demonstrating an actual deficient optimal face.
    It may still serve a direction-accuracy role that this study does not replace.
11. **Single next task:** derive and validate a defensible posterior numerical
    admission/direction-accuracy criterion for full-rank residuals below `1e-4`,
    based on actual SVD backward error, matrix-construction uncertainty, polar
    sensitivity, and required direction accuracy. Study captured fixtures first.

**Do not lower the production guard from these observations.** No nonsmooth
solver, production repair, training continuation, further LR sweep, or extra
seed is authorized by this report. This study stops at classification.
