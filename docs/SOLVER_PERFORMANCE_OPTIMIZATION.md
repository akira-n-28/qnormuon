# Exact solver performance audit, 29 September 2026

This study changes no quotient geometry, LMO, acceptance tolerance, residual
conditioning guard, or fallback mathematics. It removes one optional
post-cast observation from normal training and reuses the already computed
float64 smooth residual spectrum in the original-coordinate dual certificate.
The complete H100 suite passed **225 tests** (job `28917`), and the locked
50-step QSO smoke passed with **300/300 certified pair solves**, zero fallback,
and identical training and validation losses at every recorded step (job
`28922`). No learning-rate sweep was run.

The acceptance checks remained: normalized gap `<=3e-5`, signed normalized
gap `>=-1e-10`, normalized horizontal residual `<=1e-10`, spectral excess
`<=1e-12`, and smooth residual rcond `>1e-4`. Certificates are still
floating-point numerical certificates, not interval proofs.

## Environment and fixed experiment

All tensor benchmarks and training ran under SLURM on one NVIDIA H100 NVL,
using Python 3.10.20, torch 2.10.0+cu128, and NumPy 2.2.6. The locked smoke
used the 11,457,408-parameter, six-layer SwiGLU Transformer, six
`[1024,384]` pairs, the same seed 2026, FineWeb token prefixes and batches,
bf16 forward/backward with fp32 parameters and canonical momentum, direct fp64
pair solving, QSO learning rate 0.001, `3e-5` normalized-gap target, and
`1e-4` rcond guard. No assets were downloaded or networks contacted.

The baseline is [smoke job 28914](../cluster/run_tiny_smoke-28914.log) with
full cast diagnostics and independent original-residual certificate spectra.
The optimized run is [job 28922](../cluster/run_tiny_smoke-28922.log). Its
compact metrics, solve events, summary, checkpoint, and provenance are under
`/home/prignano/qnormuon-runs/tiny-transformer/smoke-28922-qso-seed2026/`.
The six-pair microbenchmarks replayed the baseline's saved warm step-25 state.

## Complete decomposition accounting

Each listed call is one PyTorch batched call over **two** `[1024,384]` matrices.
The times are synchronized H100 time for one saved warm six-pair optimizer
step, including Python dispatch and CUDA synchronization. The source
classification reconciles all calls; there were no fallback/reference or
unclassified decompositions. [Before accounting](../cluster/solver_svd_accounting-28915.json)
and [after accounting](../cluster/solver_svd_accounting-28921.json) retain each
individual call, shape, dtype, device, pair, and elapsed time.

| Source | Operator | Before: calls / matrices / total / median call | After: calls / matrices / total / median call |
| --- | --- | ---: | ---: |
| Initial Newton residual | `svd` | 6 / 12 / 0.257 s / 42.83 ms | 6 / 12 / 0.256 s / 42.68 ms |
| Line-search trial | `svd` | 12 / 24 / 0.516 s / 42.78 ms | 12 / 24 / 0.514 s / 42.65 ms |
| Primal feasible recovery | `svdvals` | 18 / 36 / 0.744 s / 41.48 ms | 18 / 36 / 0.742 s / 41.32 ms |
| Original-residual dual certificate | `svdvals` | 18 / 36 / 0.775 s / 42.96 ms | 0 |
| Post-cast research diagnostics | `svdvals` | 6 / 12 / 0.245 s / 41.19 ms | 0 |
| **Total** | | **60 / 120 / 2.537 s** | **36 / 72 / 1.512 s** |

Per pair, the baseline counts were: initial residual **1 call / 2 matrices**,
line search **2 / 4**, primal recovery **3 / 6**, residual dual certificate
**3 / 6**, and cast diagnostics **1 / 2**. After optimization the first
three categories retain those per-pair counts and the last two are zero.
Multiplying each per-pair count by six gives the six-pair call and matrix
counts in the table.

Before: `svd` totaled 0.773 s and all `svdvals` 1.764 s, of which 1.519 s
was mandatory solver work. After: `svd` totaled 0.770 s and `svdvals` 0.742 s.
The counts agree with `svd_evaluations` (18 matrices per pair before, 12
after, excluding two cast-only matrices). SVD/HVP behavior was unchanged;
these savings came from two redundant/observational `svdvals` sources.

## Independent change 1: cast diagnostics

`QuotientSpectralOptimizer(..., diagnostics=True)` still records mandatory
solver metrics. New `cast_diagnostics=False` (default) avoids building the
post-cast research pair and calling `svdvals` solely to measure its spectral
excess. `cast_diagnostics=True` restores relative cast-direction error,
post-cast horizontality, and spectral excess explicitly. Certification always
applies to the recovered fp64 primal *before* the model-dtype cast.

On alternating identical saved warm states, full and basic modes produced
exactly the same model hash and per-pair mandatory metrics. Full diagnostics
used 42 `svdvals` calls and median optimizer time 2.595 s; basic used 36
calls and 2.349 s: **1.105×** faster. The six eliminated calls account for
about 0.246 s. Raw runs are in
[job 28916](../cluster/cast_diagnostics_benchmark-28916.json).

## Independent change 2: cached dual spectrum

Let `centered=A-L*(beta)`, `coord=W^(-1/2)`, `s=||centered||_F`, and let the
whitened Newton coordinate be `z`. The implemented original multiplier is
`lambda=beta+s*coord*z`. By linearity of `L*` and the definition of the
effective, row-whitened weights,

```
A-L*(lambda)
  = centered-s*L_eff*(z)
  = s*(centered/s-L_eff*(z))
  = s*B_internal(z).
```

Thus in exact arithmetic the original dual objective is `s` times the sum of
the **same accepted evaluation's** cached singular values, and positive `s`
does not change its rcond. The production fp64 certificate now uses this
identity only when the multiplier and primal candidate belong to that exact
smooth evaluation. An identity mismatch raises an error. A float32 research
solve, cancellation-dominated objective, zero cotangent, or CPU reference
fallback still uses independent original-residual `svdvals`.
`SolverConfig(independent_certificate=True)` forces that old computation for
debugging. The primal projection and full spectral feasibility check are
unchanged.

Randomized fp64 tests covered three shapes, scales `1e-8`, `1`, and `1e8`,
plus a residual near the `1e-4` rcond guard. Cached and independent dual
objectives, rconds, and gaps agreed well below `3e-5`; a mismatched multiplier
was rejected. On all six *identical saved warm training inputs*, lambda,
primal direction, and final model update were identical. Maximum absolute
differences were `4.72e-16` in dual objective, `4.19e-15` in normalized gap,
and `3.09e-16` in rcond. Independent recomputation used 108 matrix
decompositions per six-pair step versus 72 with cache. Median optimizer time
was 2.352 s versus 1.578 s, a **1.49×** incremental speedup after cast
diagnostics were disabled. Horizontal residuals, normalized horizontality,
spectral norms, and spectral excess were exactly equal; the final model hash
was also identical, verifying the lifted parameter update. See the final
explicit comparison in
[job 28924](../cluster/cached_certificate_benchmark-28924.json).

## Remaining exact primal spectral norm

The full `svdvals` of each horizontally projected primal pair remains
mandatory in the current implementation. Six accepted real warm candidates
showed these synchronized median *totals* for six batched calls:

| Norm method | Time | Largest value error versus full `svdvals` | Decision |
| --- | ---: | ---: | --- |
| `svdvals(P)` | 0.2434 s | 0 | Keep as oracle/default |
| `matrix_norm(P, ord=2)` | 0.2435 s | 0 | No speed gain |
| `sqrt(lambda_max(P.T @ P))`, fp64 `eigvalsh` | 0.0441 s | `1.18e-13` | Further validation required |
| QR then `svdvals(R)` | 0.2441 s | 0 | No speed gain |

For the six real candidates, the Gram norm changed the computed normalized
gap by at most `9.77e-14` and the primal objective by at most `1.91e-14`;
the largest independently checked spectral norm after radial scaling was
`1+9.4e-14`. All methods reproduced deterministically. The real warm
residual rconds ranged down to about `1.47e-3`, above the production guard.
The Gram method is only a candidate optimization: squaring a matrix squares
its condition number, and this study did not replace the default based on six
real candidates alone. The separate synthetic near-guard results are reported
below. [Kernel study](../cluster/solver_kernel_study-28923.json).

Four additional deterministic `[1024,384]` residuals had rconds `1e-4`,
`1.2e-4`, `1e-3`, and `1e-2`; their polars were projected with a saved real
pair's row constraint. At the guard boundary, the Gram top norm differed from
full SVD by at most `1.15e-13` across these cases. Its largest normalized-gap
difference was `1.07e-13`, and the independently checked norm after scaling
was at most `1+8.6e-14`. The maximum *absolute* primal-objective difference
was `4.52e-11` on these synthetic scales. This supports investigating Gram
for the **largest** singular value, but does not validate Gram reconstruction
of all singular vectors in the smooth Newton/HVP path or rule out more
adversarially conditioned candidates. It remains a measured option, not the
production default.

## Certificate frequency

The saved warm step made six complete certificates at each of Newton
iterations 0, 1, and 2. Iterations 0 and 1 certified **none** and cost
0.257 s and 0.253 s respectively; iteration 2 certified all six and cost
0.247 s. Across the locked 50-step baseline, 269/300 pair solves certified
at iteration 2 and 31/300 at iteration 3 or later; none accepted at 0 or 1.
The first two certificates therefore consume about 0.51 s per representative
warm step. Polar horizontality and gradient-related quantities are cheaper,
but no one-sided bound was established that could safely reject an otherwise
acceptable candidate without a full certificate. Production still certifies
**every** iterate; it does not hard-code a two-iteration minimum.

## Tall residual backends and batching

For the full-rank polar, singular spectrum, left/right factors, rcond, and
polar-derivative HVP, six real residual pairs were compared with full batched
`svd` as an independent oracle. Median per-pair costs were 43.00 ms for full
SVD, 43.16 ms for QR followed by SVD of `R`, and 7.84 ms for fp64 Gram EVD
plus left-vector reconstruction. QR gave no speedup. Gram EVD matched the real
HVP to at most `1.89e-12` relative error, but at synthetic rcond `1e-4` to
`1.2e-4` its HVP error reached `5.42e-9` and left-vector orthogonality error
reached `1.08e-8`. It squares condition number and has not been validated
over enough near-guard adversarial cases or a full certified trajectory to
replace the default. No backend was changed.

On six real `[2,1024,384]` residual/candidate batches, combining all 12
matrices into one call yielded isolated speedups of only **1.003×** for SVD,
**1.003×** for `svdvals`, and **1.005×** for `eigvalsh` compared with six
existing two-matrix calls. A batched multi-pair Newton restructure is not
justified by this kernel result.

## Locked 50-step H100 result

The optimized run used the same model initialization, batches, learning rate,
schedule, precision, and solver target as job 28914. Batch hashes matched
for all 50 steps; every training and periodic validation loss matched
**exactly** in the compact logs. All 300 pair solves certified, with the
same Newton bins (0: 0, 1: 0, 2: 269, 3+: 31), zero fallback, maximum gap
`2.9958e-5`, and minimum rcond `9.778e-4`. Three-step checkpoint replay had
zero parameter, loss, and optimizer-state discrepancy. The final validation
loss was `5.9848536253` in both runs.

Warm timings exclude the first five steps but the cold step remains in both
raw logs and summaries. They include the benchmark's synchronized timing
hooks and are directly comparable:

| Metric | Baseline median / p95 | Optimized median / p95 | Median speedup |
| --- | ---: | ---: | ---: |
| Complete step | 2.665 / 2.799 s | 1.617 / 1.705 s | **1.65×** |
| Optimizer | 2.637 / 2.771 s | 1.592 / 1.681 s | **1.66×** |
| Six pair solvers | 2.372 / 2.507 s | 1.579 / 1.667 s | **1.50×** |
| `svd` | 0.776 / 0.822 s | 0.774 / 0.819 s | 1.00× |
| `svdvals` | 1.520 / 1.605 s | 0.745 / 0.785 s | **2.04×** |
| Recovery/certificate | 1.540 / 1.626 s | 0.757 / 0.797 s | **2.03×** |

Warm median throughput rose from 769 to 1267 tokens/s (1.65×). Total measured
run wall time fell from 147.5 s to 90.5 s. Peak allocated H100 memory changed
from 724.1 MB to 725.7 MB; early and late live allocations in the optimized
run were about 237 MB and 236 MB, with no growth trend. No external-network
attempt was observed. These are performance measurements on one H100 and this
small fixed model, not a claim about optimization quality against AdamW.

The cold first step was not excluded from the raw data: complete step time was
3.557 s before versus 2.615 s after; pair-solver time was 3.264 s versus
2.349 s. Its SVD and `svdvals` times were 1.143/1.936 s before and
1.198/0.954 s after. First-use initialization therefore remains visible and
should not be mistaken for steady-state cost.

**Practicality classification: B — materially improved, still too slow for an
LR sweep.** The optimized QSO step is still roughly 48× the prior 0.0334 s
AdamW warm step, and its optimizer occupies about 98% of full-step time.
The dominant remaining work is about 0.774 s in required smooth SVD/polar
and 0.745 s in required projected-primal `svdvals`; Newton-CG/HVP is only
about 0.019 s. CPU ADMM fallback remained unused and is still an expensive,
observable reference path. The next low-risk target is a stronger adversarial
and end-to-end validation of an exact deterministic **top singular value**
backend for primal radial feasibility, with independent full-SVD checks near
the rcond guard. Any further reduction in certificate frequency needs a
proved safe filter rather than a fixed minimum Newton count.
