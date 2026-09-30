# Reduced QR plus square SVD: exact-decomposition study

## Contract and scope

This is an isolated research backend. Production still uses direct tall fp64
thin SVD; its warm multiplier, Newton-CG, line search, projected-primal
`gram_upper` recovery, certificate thresholds and reference fallback are
unchanged. Neither Gram reconstruction nor QDWH is used here.

For full-column-rank `B` of shape `[m,n]`, `m >= n`, reduced QR gives
`B = Qr R`, with `Qr.T Qr = I` and square `R`. A complete square SVD gives
`R = Ur Sigma Vt`. Consequently

```
B = (Qr Ur) Sigma Vt,
(Qr Ur).T (Qr Ur) = I.
```

These are complete thin-SVD factors, not a rank approximation. All singular
values are retained. The nuclear value is `sum(sigma)`, residual rcond is
`sigma_min/sigma_max`, and polar is `(Qr Ur) Vt`. The existing dual gradient,
SVD-based polar derivative and matrix-free HVP therefore apply unchanged.
Repeated singular values allow different bases within a singular subspace;
the comparison concerns the polar, spectrum and derivative rather than
individual factor columns.

In floating point this is two stable factorizations and a factor-composition
matmul, rather than one direct SVD. Exact-arithmetic equivalence does not imply
bitwise factor equality or identical near-boundary decisions. The independent
direct tall fp64 SVD remains the numerical oracle. No new posterior-based
branch semantics are introduced.

## Reproducible implementation

- `experiments/smooth_qr_svd.py`: fp64-only `tall_svd` and
  `qr_reduced_svd`; scoped research adapter changes only `SmoothDual.evaluate`.
- `tests/test_smooth_qr_svd.py`: independent direct-SVD factor/derivative/solver
  regressions, extreme scaling, repeated spectra and scoped restoration.
- `cluster/study_smooth_qr_svd_h100.sbatch`: full tests and numerical/kernel
  screen, one H100, four CPUs, 16 GiB, 20 minutes, offline shared-home paths.
- `cluster/study_smooth_qr_svd_h100.py`: prescribed/real corpus, unchanged-solver
  replay, synchronized component/driver/batching timings.

The corpus and local checkpoint reader are reused from an older study script;
its Gram decomposition routines are not invoked. CUDA TF32 is disabled. Driver
experiments use only full-accuracy default, `gesvdj`, and `gesvd`, never `gesvda`.

## Environment, provenance and validation

H100 jobs `28997` (initial screen) and `29001` (confirmation) completed with
exit code zero. Python was `/home/prignano/modded-nanogpt/.venv/bin/python`,
version 3.10.20; torch 2.10.0+cu128, CUDA runtime 12.8, NVIDIA H100 NVL.
Both jobs used the existing offline environment and network guard. No packages
or assets were downloaded, and the base environment was not modified.

The confirmation full suite passed **378 tests, zero failed, zero skipped**,
in 15.740 s. This includes 28 new research regressions. Raw output and results:

- `cluster/study_smooth_qr_svd_h100-29001.log`
- `cluster/smooth_qr_svd_tests-29001.xml`
- `cluster/smooth_qr_svd_study-29001.json`

The JSON retains individual matrix/direction errors and all kernel samples.
The initial screen's corresponding log and JSON have suffix `28997`.

Real inputs were reconstructed from the already-local locked checkpoint:
`/home/prignano/qnormuon-runs/tiny-transformer/smoke-28936-qso-seed2026/checkpoint.pt`
(25 completed steps). One allocated reconstruction step captured six pair
problems and 18 residual evaluations; the locked minibatch hash and loss were
checked against the existing log. This is a checkpoint-based input
reconstruction, not a candidate-controlled training experiment.

## Numerical corpus and factorization

The final screen compared 100 matrices: 64 generated cases and 36 real residual
sides. Shapes were `[24,8]`, `[96,24]`, and `[1024,384]`. Cases included Gaussian
matrices, orthogonally rotated prescribed spectra, repeated top/bottom values,
top clusters separated by about 1e-12, the requested rcond points, and additional
points immediately around the guard. Overall scales 1e-150 and 1e150 were
tested, including production-shaped near-guard spectra. Additional rconds 1e-8
and 1e-10 deliberately tested outside the admitted smooth regime.

Error norms use Frobenius norms and scale-normalized reconstruction to avoid
overflow/underflow in the measurement. Computation of both decompositions sees
the original unscaled matrix. No singular values are truncated.

For cases clearly above the guard (excluding the exactly prescribed boundary):

| Quantity | Worst QR-SVD versus direct tall SVD |
|---|---:|
| Relative polar error | 5.01e-13 |
| Maximum relative singular-value error | 6.31e-13 |
| Relative nuclear-value error | 3.55e-15 |
| Absolute rcond error | 2.66e-15 |
| Informative derivative relative error | 2.19e-10 |

Across cases down to rcond 5e-5, worst relative reconstruction error was
1.68e-13, identical to the direct-oracle worst; left/right orthogonality defects
were at most 1.18e-12 and 2.92e-12. These production-shaped accuracy limits
already occur in the direct tall-SVD factors.

All 36 real residuals produced identical polar, spectrum and derivative outputs
on this installation. Their minimum residual rcond was about 1.47e-3. The two
production-shaped extreme-scale cases also had zero measured polar/spectrum
differences; reconstruction errors were about 1.2e-13. These observations do
not establish bitwise equivalence for other shapes or CUDA implementations.

At the **exactly prescribed** 1e-4 boundary, one `[24,8]` case classified on
opposite sides: direct SVD gave `0.00010000000000002375`, QR-SVD gave
`0.00009999999999996001`. This is a roundoff-scale boundary difference, with
QR-SVD on the rejecting side; it is not an unexplained well-separated guard
crossing. No threshold or classification semantics were changed. No such
difference occurred on the real replay or the explicitly above/below-guard
points. Exact-arithmetic equivalence alone cannot guarantee a bitwise guard
decision on a value at the floating-point threshold.

The extra outside-domain rcond 1e-10 cases showed larger differences: worst
relative polar error 6.47e-8, relative singular error 3.15e-8, and informative
derivative error 6.73e-7. This is not evidence for extending the admitted
domain; the existing 1e-4 guard remains unchanged.

## Existing polar derivative and unchanged solver

The candidate factors were fed directly into the current
`SmoothDual.polar_derivative`; no derivative algorithm was rederived or changed.
Directions included random, tangent-like, normal-like, smallest-singular normal,
and clustered-subspace skew perturbations. Perturbations of the form
`u_i v_i.T` have **zero exact polar derivative**; they were retained as
absolute-error diagnostics rather than divided by a roundoff-only denominator.
For example, the worst such absolute discrepancy down to rcond 5e-5 was
2.14e-8; informative derivatives in that corpus agreed to 8.94e-10 relatively.

Centered finite differences were also checked on the smaller shapes. The worst
admitted near-guard difference was about 7.33e-5 relative, dominated by the
finite-difference comparison: candidate versus direct analytic derivatives
differs by at most 2.2e-10 there. The focused CPU regression independently checks
an informative derivative against centered differences within 1e-5.

The six real warm problems were solved from identical original-coordinate
lambda using the exact existing production solver with fallback disabled.
All twelve runs certified; neither backend entered CPU ADMM. Every pair used
two Newton iterations, two line-search trials and three residual evaluations.
CG counts were `[3,5,5,5,5,5]` for both backends. Initial HVP/Newton direction,
trial coordinates/objectives/gradients, certificate rejection/acceptance
sequence, final lambda, recovered direction and gap were identical in these
replays. Thus there was no observed unsafe certificate acceptance or changed
fallback decision on the tested real problems.

This is evidence for six real warm solves, not a claim that all 931 locked
evaluations or an entire candidate-controlled trajectory were validated.

## Synchronized H100 kernel cost

Each component had three untimed warmups and 21 synchronization-bracketed
samples. Timings include dispatch/allocation and synchronized completion;
component medians need not add exactly to the total. Shape is `[2,1024,384]`.

| Operation | Median ms | p95 ms |
|---|---:|---:|
| Direct tall thin SVD | 43.321 | 43.403 |
| Reduced QR | 4.347 | 4.365 |
| Complete square-R SVD | 38.971 | 39.036 |
| `Qr @ Ur` | 0.060 | 0.062 |
| Complete QR-reduced SVD | 43.480 | 43.546 |

Speedup is **0.996x**: explicit QR-SVD is about 0.37% slower, rather than
materially faster. The initial screen measured 43.348 versus 43.506 ms. A
reversed-order confirmation measured 43.685 versus 43.851 ms, again 0.996x.
The square SVD consumes nearly all of the time remaining after QR; explicit
reduction does not expose a new performance advantage on this installation.
Identical production-shaped factors are consistent with redundant explicit
reduction, but this study does not assert an uninspected cuSOLVER internal
implementation.

Square driver audit:

| Driver | Square SVD median/p95 ms | Total QR-SVD median/p95 ms |
|---|---:|---:|
| Default | 38.994 / 39.267 | 43.493 / 43.562 |
| `gesvdj` | 39.047 / 39.096 | 43.505 / 43.579 |
| `gesvd` | 75.051 / 75.111 | 79.498 / 79.554 |

Default and explicit `gesvdj` spectra were identical. `gesvd` was substantially
slower; its maximum relative spectrum difference was about 1.27e-13. There is
no reason to select it for speed. Approximate `gesvda` was not used.

Six **distinct** accepted real residual pairs were benchmarked sequentially and
as the same twelve matrices in one batch, with three warmups and 11 samples:

| Backend | Six sequential median/p95 ms | Batched median/p95 ms |
|---|---:|---:|
| Direct tall SVD | 259.721 / 259.832 | 258.830 / 259.058 |
| QR-reduced SVD | 260.780 / 261.005 | 259.331 / 259.491 |

Cross-pair batching saves only about 0.3–0.6%, with no material kernel gain.
This is not evidence for restructuring independent Newton solves into a
batched solver. The six complete warm replay solves totaled 0.984 s for direct
SVD and 0.985 s for QR-SVD in one sequential measurement, consistent with the
kernel result; these are not end-to-end training timings.

## Stage gate and decision

**B: numerically correct on the tested domain, but no material speed advantage.**

1. QR-reduced SVD supplies the complete factor contract with measured accuracy
   comparable to direct SVD, including near rcond 1e-4.
2. Repeated/clustered spectra and representable extreme scales were handled
   without truncation; exact boundary classifications can differ by roundoff.
3. The current polar derivative and HVP work unchanged.
4. Solver branching and final outputs matched on all six real warm replays.
5. QR costs about 4.35 ms per residual pair.
6. Square-R SVD costs about 38.97 ms per residual pair.
7. Total kernel speedup is about 0.996x, not an improvement.
8. Cross-pair batching provides no meaningful advantage.
9. Production integration is **not justified**.

The material-kernel-speedup gate failed. Therefore the conditional 931-evaluation
shadow, short candidate-controlled run and locked 50-step run were not launched.
No end-to-end speedup, complete locked-trajectory branch equivalence or new
checkpoint-trajectory result is claimed. Production code/defaults, smooth full
fp64 SVD, prior lambda, primal `gram_upper`, all acceptance thresholds and both
fallback concepts remain unchanged. The expensive complete smooth SVD remains
the dominant bottleneck; this explicit reduction is not its solution.
