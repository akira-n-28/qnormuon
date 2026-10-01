# Research full-rank epsilon-LMO admission experiment

**Classification A: FULL-RANK EPSILON-LMO ADMISSION READY FOR CONTROLLED
TRAINING**, within the stated conditional fp64 model and primary-only scope.
All five saved Stage-D blockers certify at Newton iterations **3,3,3,4,4**;
all seven available ordinary controls certify at their original stopping
iterations. The deterministic spectral stress corpus admits its 54 numerically
separated cases and refuses its 12 deficient/rank-ambiguous cases. No hidden CPU
reference use or nonfinite action occurred in the successful real or synthetic
solves. Deliberately injected invalid actions fail closed.

This justifies a **separate controlled tuned 512-step research experiment**.
It does not approve production integration, blanket guard removal, deficient
face recovery, another sweep, or extra seeds. No training occurred here.
Production-v0, its `rcond_guard=1e-4`, full fp64 thin SVD, previous original
lambda, all certificate thresholds, and CPU ADMM behavior remain unchanged.

## Contract and selection scope

The research result is labeled `primary_epsilon_lmo_full_rank` and
`selection_certified=False`. For the represented canonical fp64 inputs it
certifies, under the previously documented normal finite fp64 gamma model:

1. Positive numerical rank lower bounds for both final residuals.
2. An exactly feasible mathematical proxy Pc in the coupled horizontal
   spectral ball and a bound on the returned rounded pair's distance to Pc.
3. A conservative normalized primary support-value gap at most `3e-5`.

For any feasible Pc, weak duality gives

```
0 <= v(A)-<A,Pc> <= phi(lambda)-<A,Pc> <= G_upper.
```

With positive `dual_lower <= phi`, require `G_upper/dual_lower <= 3e-5`.
For positive support values, this also bounds relative support regret by
`3e-5`, as established in
[QSO_NUMERICAL_OUTPUT_CONTRACT.md](QSO_NUMERICAL_OUTPUT_CONTRACT.md).
No observed reference direction or fitted direction budget enters acceptance.

This is **not** a prescribed distance to P*, P-dagger recovery, deficient-face
completion, exact finite-step function/loss equivalence, or instantaneous-loss
descent for EMA momentum. Full rank at a current multiplier does not establish
optimal-face uniqueness. An explicitly known nonsmooth face is excluded with
`known_nonsmooth_face=True`, returning `explicit_nonsmooth_face`. Undetected
nonuniqueness is not ruled out by the rank posterior; a primary-only result must
never be advertised as secondary selection.

The represented intrinsic-zero branch remains separate: it returns the exact
zero pair with `selection_semantics="exact_zero"`. It executes before rank
admission and even before an explicit-face exclusion, since zero has its known
minimum-norm selection. Nearly vertical cancellation retains an explicit failure.

## Isolated implementation and numerical safeguards

[experiments/full_rank_admission.py](../experiments/full_rank_admission.py)
contains the research solver; production functions are not patched or edited.
It uses production `SmoothDual.evaluate`, cached-SVD polar derivatives/HVPs,
and `certificate(..., primal_norm_backend="gram_upper")`. Its local loop and
instrumented CG retain production normalization, whitening, damping, CG target
and budget, slope test, 24-trial Armijo/rounded-decrease/gradient-improvement
tests, stagnation check, and Newton budget. The admission conditions change
only in this isolated research path.

Inputs are explicitly fp64 and autocast is disabled. Canonical tensor inputs
are the already saved fp32 representations promoted to fp64, not a claim to
recover an unavailable unrounded EMA. Initialization and returned lambda use
the original multiplier convention. No predictor or alternative decomposition
backend is enabled.

For every initial or trial evaluation, reconstruct the original residual and
reuse the **same evaluation's** factors, with singular values multiplied by its
positive intrinsic magnitude. The identity is
`B_original = magnitude * B_internal` in exact arithmetic. Original-residual
reconstruction also measures rounded centering/whitening/scaling discrepancy.
Using the existing machinery from
[FULL_RANK_DIRECTION_ADMISSION.md](FULL_RANK_DIRECTION_ADMISSION.md), measured
factor orthogonality and reconstruction defects plus GEMM/reduction allowances
bound the distance eta to a nearby exactly orthonormalized-factor matrix.
Residual-construction rounding is included. Then

```
alpha_lower_j = (sigma_min_listed_j - eta_j) * (1-gamma_2).
```

Both alpha lower bounds must be strictly positive **before** an evaluation may
control Newton or an eligible line-search trial. Orthogonality defects must
permit the mathematical orthogonalization bound (inflated defects below one).
Residuals, factors, gradient, spectrum, multiplier and objective must be finite;
nonpositive or subnormal singular values are refused. There is no replacement
constant such as `rcond > 1e-5`: `1e-4` remains a recorded v0 diagnostic only.

This is conditional numerical certification, **not directed interval arithmetic
or a CUDA-backend formal proof**. It inherits normal finite fp64 operation-model
assumptions and does not prove all underflow behavior from finite checks alone.
The stress amplitudes are chosen within normal representable arithmetic.
Positive alpha certifies current rank; it is not a forward-error theorem for
every HVP or a positive lower bound on the multiplier Hessian.

Every CG query records the two sigma minima/alpha bounds, vector and HVP norms,
damped/undamped curvature, damping, and CG residual. Newton actions record
gradient norm, direction norm, slope, CG termination, target and final residual.
Nonfinite HVP/curvature/scalars, unrepresentable work, or no completed usable CG
action fail closed. A finite truncated CG direction remains eligible under the
existing semantics only if the subsequent negative-slope test passes. No fitted
HVP-error threshold is introduced.

Rank-invalid trials are explicitly ineligible and backtracked. Rank-valid trials
use the unchanged objective/globalization conditions. Exhaustion of 24 trials,
Newton budget, stagnation or non-descent returns failure. On failure, `pair=None`:
an uncertified approximation is not exposed as a returned update. This research
path never calls CPU ADMM; failure is explicit and separate from production's
unchanged reference fallback.

At each candidate recovery, the existing
[value_posterior](../experiments/qso_output_contract.py) supplies dual upper/lower
bounds, primal lower bound, G_upper and feasibility distance from matching cached
factors/radial metadata. The primary feasible proxy is the mathematical exact
horizontal projection followed by conservative radial scaling. Returned fp64 p
is within the documented proxy distance; it is not bitwise exactly horizontal.
Final acceptance additionally requires finite values and the unchanged checks:

| Check | Requirement |
| --- | ---: |
| Numerical rank | both alpha lower bounds > 0 |
| Conservative normalized value gap | `G_upper/dual_lower <= 3e-5`, dual_lower > 0 |
| Signed raw normalized gap | >= `-1e-10` |
| Normalized horizontal residual | <= `1e-10` |
| Spectral excess | <= `1e-12` |

No extra tall SVD is used for either posterior. Pre-cast certification does not
automatically extend through raw lifting, model-dtype casting or parameter
addition. Those remain separate numerical operations.

## Five fixed Stage-D blockers

Inputs are exactly the five saved `failed_pair.pt` paths in
[STAGE_D_QSO_FAILURE_FORENSICS.md](STAGE_D_QSO_FAILURE_FORENSICS.md). Actual
saved original warm lambdas are used. No model/data trajectory was regenerated.

| Fixture | First accepted Newton | rcond U / D | alpha_lower U / D | Conservative normalized gap | CG / line trials |
| --- | ---: | --- | --- | ---: | --- |
| Tuned confirmation | 3 | `1.29544e-4 / 8.67898e-5` | `4.79575e-5 / 3.75805e-5` | `1.282376e-5` | 9 / 3 |
| Low | 3 | `9.78585e-5 / 7.74149e-5` | `4.27084e-5 / 4.48645e-5` | `1.204232e-5` | 10 / 3 |
| Middle | 3 | `6.64874e-5 / 4.48940e-5` | `7.52205e-5 / 6.44824e-5` | `2.656944e-5` | 11 / 3 |
| High | 4 | `1.15247e-4 / 7.70606e-5` | `1.17248e-4 / 1.11029e-4` | `4.610439e-7` | 17 / 4 |
| Unsupported-AdamW ablation | 4 | `1.20785e-4 / 8.76116e-5` | `1.70887e-4 / 1.60351e-4` | `1.944436e-7` | 13 / 4 |

The five solves have 22 smooth pair evaluations (44 individual matrix SVDs),
60 CG/HVP queries, 17 line trials and no reference calls. Every evaluation is
below the old guard on at least one side, with positive posterior rank bounds.
No step reduction is needed in these fixtures. Positive tails remain well above
the model uncertainty: across blockers and controls, sigma_min/eta is at least
`1.36e6`. Rcond and alpha are different quantities: rcond is a relative
conditioning ratio, while alpha_lower includes the actual scale and measured/
modeled uncertainty. A below-guard ratio does not force alpha_lower negative.

### Offline high-accuracy audit

Saved independent L-BFGS multipliers from
[NEAR_RANK_OPTIMUM_CLASSIFICATION.md](NEAR_RANK_OPTIMUM_CLASSIFICATION.md) are
loaded **after acceptance**, and independently rechecked by original-coordinate
full SVD/KKT recovery. They are high-accuracy numerical references, not formal
interval-exact P*. The measured support regret is therefore reported as an
offline reference comparison, not called an exactly known true regret.

| Fixture | G_upper | Observed support regret | Relative direction difference | Multiplier difference norm |
| --- | ---: | ---: | ---: | ---: |
| Tuned | `1.597004e-5` | `1.596888e-5` | `1.383039e-5` | `1.190792e-7` |
| Low | `2.496129e-5` | `2.495979e-5` | `1.392306e-5` | `1.658883e-7` |
| Middle | `1.097148e-4` | `1.097110e-4` | `2.917486e-5` | `5.498837e-7` |
| High | `2.069576e-6` | `2.066058e-6` | `4.912966e-7` | `1.115373e-8` |
| Ablation | `1.222732e-6` | `1.218021e-6` | `2.176514e-7` | `8.450533e-9` |

All measured regrets lie inside G_upper. No direction comparison affects
admission. Final sigma minima, full rank uncertainty, spectra extrema,
feasibility and multiplier comparisons are in the structured output.

## Ordinary controls and operational safety

All six saved locked step-0 problems and the saved real diagnostic warm pair
pass, with stopping iterations **3,3,2,3,3,3,2**. Their returned pairs and lambdas
are bitwise identical to production-v0 on these saved inputs; CG/line counts
are unchanged. This measured equivalence is not a promise for every near-boundary
case: conservative value inflation may legitimately demand extra work.

Across the five blockers and seven controls, all 36 Newton CG solves stop at
their residual target. Every used damped curvature is positive (minimum
`2.996e-8`), all HVPs are finite (maximum recorded norm about `5.914e3`), and
every proposed slope is negative (largest/closest to zero about `-6.441e-11`).
All eligible real trials accept at step length one. This is operational evidence,
not a newly asserted universal HVP conditioning theorem. Final normalized
horizontality is at most `8.80e-17`, spectral excess zero, and exact-proxy distance
at most `4.42e-11` across these accepted real problems.

These seven controls are all currently available saved ordinary pair inputs;
they are not a full saved long-training distribution. No unavailable model
trajectory was regenerated to expand this coverage.

## Adversarial scope, rank and fail-closed tests

The H100 spectrum corpus uses `[24,4]` and `[64,12]` matrices constructed with
deterministic QR singular-vector orientations. Prescribed tails are repeated;
the top singular value is one. Rconds are
`3e-4,1.5e-4,1.01e-4,9e-5,7e-5,5e-5,2e-5,1e-5,1e-6`, plus `1e-16` and zero.
Overall scales are `1e-20,1,1e20`. Equal U/D and equal objective sides give a
known stationary optimum at lambda=0; nonzero perturbed warm multipliers force
actual Newton solves. These are solver problems, not learning-rate sweeps.

All 54 separated cases certify, 27 after four Newton iterations and 27 after
five. Maximum line-trial totals are five and six in the two shapes. There is no
globalization failure, hidden reference or nonfinite action. The cases include
243 Newton actions, and 162 additional fixed HVP probes at the independently
known stationary states remain finite. Minimum final rcond is about `1e-6`,
without using that number as a cutoff. Maximum conservative normalized gap is
`1.81e-5`. Both scales and orientations affect uncertainty; the result does not
establish a universal lowest admitted rcond.

All 12 exactly deficient/uncertainty-scale cases reject before Newton with
`rank_ambiguous_or_deficient`. Their alpha lower bounds are nonpositive, at all
three absolute scales. A failure at the mathematical rank margin is not rescued
by a small primary gap. No singular value is truncated.

Permanent research regressions also cover:

- Repeated positive values and tails as small as `1e-10` in analytic diagonal
  examples with positive numerical rank margins.
- Unique primal/nonunique dual; no multiplier identity is required.
- The explicit nonunique-face counterexample from the output-contract report:
  feasible `diag(1,1)` pairs have tiny/zero primary regret and positive current
  alpha at `lambda=[0,1e-6,0]`, but remain distance approximately `sqrt(2)` from
  P-dagger `diag(1,0)`. The primary certificate never marks selection certified.
  Supplying the known-face flag explicitly excludes the solve.
- Exact zero and exact represented vertical cotangent return exact zero.
- Corrupt factors, nonfinite HVP, unusable curvature, exhausted Newton budget,
  deliberately invalid trial objectives, and rank-invalid trials fail closed.
  A line-search exhaustion records all 24 trials and returns no pair.
- Each final criterion is tested just beyond its unchanged boundary; ambiguity,
  nonpositive dual lower/rank bound, or nonfinite certificate values cannot pass.
- Frozen dtype/threshold configuration and no extra posterior decomposition or
  hidden CPU reference; ordinary Newton replay agrees with independent production
  control rather than using the research solver as its own oracle.

The injected NaN HVP is a deliberate negative test, not an observed instability
of the real fixtures. It produces `nonfinite_action` with no returned update.

## Synchronized H100 cost

Three warmed alternating baseline/research repetitions per saved input run in
the same allocation, with explicit CUDA synchronization. Per-evaluation rank
posterior time is about **1.77–1.84 ms**, and recovered-value posterior time is
about **2.16–2.20 ms**. The separate value audit reuses the previous posterior
implementation, so some factor-bound work is intentionally repeated; this task
does not optimize that cost.

| Sample | Baseline per-solve medians | Research per-solve medians | Incremental median / p95 |
| --- | --- | --- | --- |
| Seven ordinary controls, 21 timed repetitions | 162–217 ms | 175–236 ms | **17.62 / 19.53 ms** |
| Five blockers, 15 timed repetitions | 223–281 ms | 240–307 ms | **19.06 / 25.47 ms** |

Median relative overhead is **8.19%** on controls and **8.35%** on blockers.
The blocker timing baseline is explicitly the existing **guard-bypassed research
continuation at the 3e-5 target**, not production-v0 successfully solving below
its guard. Actual v0 rejects them and requests reference. Ordinary control
timings use unchanged production-v0 with reference disabled for observability.
No misleading speedup over the blocked production path is claimed.

All seven controls and all five blockers need no additional Newton or line work
relative to their respective baseline arithmetic. Extra tall SVD count for
posterior certification is zero. Research logging/synchronized posterior timing
and CG instrumentation are included in total overhead. Reported p95 values are
descriptive, linearly interpolated quantiles of small samples; these are not
timing confidence intervals or training throughput estimates.

## Validation, provenance and artifacts

Final job **29079**, `lagrange0`: one H100 NVL, four CPUs, 16 GiB, 20-minute
limit; **COMPLETED**, exit `0:0`, elapsed **70 s**. Runtime Python 3.10.20,
torch 2.10.0+cu128, CUDA build 12.8; deterministic algorithms enabled, TF32
disabled. Provenance lists all input/output paths and production configuration.
Network guard recorded no outbound attempt. No environment modification or
download occurred. The fixture runner initializes no model, reads no token data,
and performs no training; the full suite retains its existing model/optimizer
unit tests. No training trajectory was launched.

Complete unfiltered suite: **454 passed, zero failed/errors/skipped**, **18.520 s**
by JUnit; 427 existing cases plus 27 research cases. Existing tests were not
modified. An initial job **29078** completed the same numerical corpus but one
new test incorrectly required bitwise equality across two different optimal
multiplier representations. That assertion was corrected to fp64 numerical
agreement (`1e-14` absolute), consistent with the unique primal theorem;
production tolerances and existing tests were not weakened. Its raw failure log
is retained. Exact repeated-run pair/lambda equality at the same fixed research
inputs is also asserted during the final timing repetitions.

Peak PyTorch GPU allocation was **431,819,264 bytes** (about 412 MiB), excluding
CUDA context/driver and CPU memory. Small accepted direction tensors are saved
only under the explicit shared-home output root. No production checkpoint
format is changed. The prior AGENTS.md documentation edit is preserved.

Production source hashes match before and after:

```
a9239a1cc11463c0c0e4cc857c2297dd6628a00f08f44e9f521db5c42e2b754a  qnormuon/coupled_solver.py
04a00a4631e1e37b7dfdd29c887fabfd5006fc6df34907f6256fa73a300db720  qnormuon/optimizer.py
```

Artifacts:

- [Research solver](../experiments/full_rank_admission.py) and
  [independent tests](../tests/test_full_rank_admission.py).
- [Fixture/stress runner](../cluster/full_rank_admission.py) and
  [SLURM script](../cluster/full_rank_admission.sbatch).
- [Structured complete audit](../cluster/full_rank_admission_summary.json),
  [final raw log](../cluster/full_rank_admission-29079.log),
  [JUnit XML](../cluster/admission-tests-29079.xml), and
  [initial raw test failure](../cluster/full_rank_admission-29078.log).
- Shared-home output: `/home/prignano/qnormuon-runs/full-rank-admission/study-29079/`,
  including one accepted pair/lambda tensor per saved Stage-D fixture.

Reproduce with `sbatch --parsable cluster/full_rank_admission.sbatch` from the
project root. It validates required saved paths and runs no training.

## Explicit answers and gate

1. **Certified:** conditional numerical current full rank, exact feasible-proxy
   primary epsilon-LMO value, and bounded returned-tensor feasibility error.
2. **Not certified:** selected-direction distance, P-dagger on a nonunique face,
   deficient recovery, finite-step equivalence, or EMA instantaneous-loss descent.
3. **Blockers:** all five pass, with no reference use or nonfinite used action.
4. **Iterations:** recomputed 3,3,3,4,4 from actual saved warm lambdas.
5. **Controls:** all seven pass with the same stopping iterations and, on these
   inputs, identical directions/lambdas to v0.
6. **Alpha/rcond:** all blocker down rconds are below 1e-4 while positive alpha
   margins remain over a million times decomposition uncertainty. No new cutoff.
7. **HVP/CG:** operational on these below-guard fixtures; finite values,
   positive used curvature, residual-target termination and negative slopes.
8. **Line search:** stable on saved real inputs; synthetic separated cases
   need at most one additional trial. Exhaustion/rank failures remain explicit.
9. **Ambiguity/deficiency:** all tested such residuals refuse this branch;
   explicitly known nonsmooth faces are excluded separately.
10. **Zero:** exact zero preserved, independently of the rank test.
11. **P-dagger misreport:** not supported by any primary metadata; the permanent
    counterexample tests this distinction. Current rank does not prove uniqueness.
12. **Overhead:** approximately 18–19 ms median per saved pair solve, about 8%,
    no additional tall SVD; small-sample p95 and full breakdown above.
13. **Next stage:** classification A justifies a separately authorized controlled
    tuned 512-step research trajectory. This fixed-fixture experiment stops here;
    production-v0 remains unchanged and no trajectory is launched automatically.
