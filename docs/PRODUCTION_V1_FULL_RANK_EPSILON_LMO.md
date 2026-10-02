# Opt-in production-v1 full-rank epsilon-LMO admission

**Classification A: OPT-IN PRODUCTION-V1 ENGINEERING VALIDATED.**

The package exposes `SolverConfig(admission_policy="full_rank_epsilon_lmo")`
as an **explicit opt-in production-v1 candidate**. `SolverConfig()` continues
to select `admission_policy="v0_rcond"`. The coupled objective, quotient
geometry, canonical EMA, full fp64 thin SVD, original-coordinate lambda warm
start, projected-primal `gram_upper`, Newton-CG and globalization are unchanged.
This integration changes the numerical admission/output contract only for the
explicitly selected policy. No performance optimization or retuning is included.

## Contracts and public API

```python
import torch
from qnormuon import SolverConfig, QuotientSpectralOptimizer, SwiGLUPair

v0 = SolverConfig()  # admission_policy="v0_rcond", reference fallback retained
v1 = SolverConfig(admission_policy="full_rank_epsilon_lmo", fallback=False)
optimizer = QuotientSpectralOptimizer(
    [SwiGLUPair("mlp", mlp.up_proj.weight, mlp.down_proj.weight)],
    lr=0.0012247448713915891,
    solver=v1,
    momentum_dtype=torch.float32,
    diagnostics=True,
    cast_diagnostics=False,
)
```

The example assumes the model's `mlp` is already available.
Unsupported parameters still require their conventional optimizer; nothing in
this API automatically assigns them to QSO.

| Property | Default v0 | Explicit v1 |
| --- | --- | --- |
| Admission policy | `v0_rcond` | `full_rank_epsilon_lmo` |
| Smooth decomposition | Full fp64 thin SVD | Same |
| Warm start | Previous original-coordinate lambda | Same |
| Primal radial feasibility | `gram_upper` | Same |
| Conditioning admission | Residual rcond > 1e-4 | Both conservative alpha lower bounds > 0 |
| Primary value acceptance | Existing normalized raw gap <= 3e-5 | Conservative G_upper / dual_lower <= 3e-5 |
| Numerical failure | Existing configurable CPU reference behavior | Unsuccessful result, no reference rescue |
| Normal selection metadata | Existing primary/secondary status | `primary_epsilon_lmo_full_rank`, `selection_certified=False` |
| Exact intrinsic zero | Exact zero | Exact zero, `selection_certified=True` |

V1 configuration requires fp64 arithmetic, the existing `3e-5` tolerance,
`rcond_guard=1e-4` as retained v0 diagnostic, and `gram_upper`. `None` for the
legacy tolerance/guard fields retains those same defaults. No smaller rcond
constant is introduced, and policy is never inferred from a guard value.
The v1 policy never invokes CPU ADMM, **even if `fallback=True` is supplied**;
`fallback=False` makes that scope explicit at construction. V0 retains its
existing fallback flag semantics.

A normal v1 return certifies the **primary** represented coupled horizontal
spectral LMO under the validated conditional normal-finite fp64 operation model.
It does not certify a prescribed distance to an exact direction, minimum-
Frobenius P-dagger on a nonunique face, deficient-face completion, instantaneous
loss descent for EMA momentum, or exact finite-step loss equivalence.
Full rank at a current multiplier does not establish optimal-face uniqueness.
An explicitly known nonsmooth face can be excluded using
`solve_coupled(..., known_nonsmooth_face=True)` in v1. No automatic face detector
or deficient-face heuristic is claimed. Exact represented intrinsic zero is
handled before that exclusion and selects exactly zero.

The mathematical scope and conditional numerical assumptions are inherited
without new claims from [QSO_NUMERICAL_OUTPUT_CONTRACT.md](QSO_NUMERICAL_OUTPUT_CONTRACT.md),
[FULL_RANK_DIRECTION_ADMISSION.md](FULL_RANK_DIRECTION_ADMISSION.md), and
[FULL_RANK_EPSILON_LMO_ADMISSION_EXPERIMENT.md](FULL_RANK_EPSILON_LMO_ADMISSION_EXPERIMENT.md).
Certification precedes raw lifting, model-dtype casting and parameter addition.
Those operations retain the optimizer's existing representability checks.

## Package numerical machinery

Reusable formulas live in
[qnormuon/_numerical_certification.py](../qnormuon/_numerical_certification.py).
The isolated v1 loop and result boundary live in
[qnormuon/_full_rank_admission.py](../qnormuon/_full_rank_admission.py).
Neither imports `experiments`. The research implementations remain untouched as
independent equivalence oracles. V0's existing lazy CPU reference import remains
unchanged and is not reachable through the v1 path.

The formulas are arithmetic-identical to the validated research implementation:
fp64 unit roundoff is `u=2^-53`, with `gamma_k=ku/(1-ku)`. Measured factor
orthogonality/reconstruction defects are inflated by the same GEMM, reduction,
column-scaling, subtraction and residual-construction allowances. The posterior
also includes internal-to-original residual conversion discrepancy, using the
matching evaluation's identity `B_original = magnitude * B_internal`.
Singular values are neither truncated nor replaced by a Gram smooth solve.

For each side j, eta_j bounds the original residual's distance to a nearby
matrix with exactly orthonormalized factors and the listed singular values.
The numerical rank lower bound remains

```
alpha_lower_j = (sigma_min_listed_j - eta_j) * (1-gamma_2).
```

Both bounds must be positive before an initial or trial evaluation controls a
Newton or line-search action. Finite factors/state, bounded orthogonality defects,
positive normal singular values and reconstruction uncertainty are also required.
Positive alpha is a numerical rank certificate under the stated model; it is
not a lower Hessian eigenvalue bound or a universal HVP forward-error theorem.

The value posterior reuses the same factors and radial metadata. Nuclear-value
upper/lower allowances yield `dual_upper` and `dual_lower`. Exact mathematical
horizontal projection and conservative radial scaling define an exactly feasible
proxy Pc, with a bound on the rounded returned tensor's distance to Pc. Dot-
product rounding and that distance yield `primal_lower`. Then

```
G_upper = dual_upper - primal_lower
          + gamma_4 * (abs(dual_upper) + abs(primal_lower)).
0 <= v(A) - <A,Pc> <= G_upper.
```

No extra tall SVD is needed. The inherited direction-posterior calculations are
retained without simplification; no direction-distance budget enters admission.
These are conditional fp64 model bounds, **not directed interval arithmetic or
a backend-specific formal guarantee for every overflow/underflow regime**.

Final normal acceptance requires all of:

| Quantity | Requirement |
| --- | --- |
| alpha_lower U and D | Both > 0 |
| dual_lower | > 0 |
| G_upper | Finite and nonnegative |
| Conservative normalized primary gap | G_upper / dual_lower <= 3e-5 |
| Signed normalized raw gap | >= -1e-10 |
| Normalized horizontal residual | <= 1e-10 |
| Spectral excess | <= 1e-12 |
| Returned pair, lambda and numerical certificate | Finite |

## Iteration, failure and atomicity

The v1 loop preserves the validated research arithmetic: row whitening,
normalization, cached-SVD polar derivative, damping, truncated CG target/budget,
negative-slope requirement, 24 line trials, Armijo rounding allowance,
resolved-decrease/gradient-improvement checks, stagnation window and Newton budget.
There is no fitted HVP-error threshold. Every CG query checks finite HVP,
curvature and intermediate work, with usable completed CG work required before
proposing a direction.

An invalid trial is never eligible. As in the validated trajectory adapter, an
invalid intermediate evaluation cannot be erased by a later successful
backtrack: the package boundary refuses such a return. Failure reports expose
the initiating rank/action/globalization/budget/cancellation reason and return
`converged=False`, `pair=None`, `fallback=False`, `reference_used=False`.
Input/schema misuse still raises a validation error. No unsuccessful result is
converted into a successful primary update by CPU ADMM.

`QuotientSpectralOptimizer.step` retains its existing all-pairs-before-commit
transaction. A later-pair failure commits none of the earlier paired parameter,
EMA or lambda proposals. Diagnostics can describe the failed attempt without
mutating optimizer tensors. Unsupported-parameter AdamW remains separately
managed; the controlled harness calls paired QSO before that optimizer.

Every result exposes `admission_policy`, `selection_semantics` and
`selection_certified`. V1 diagnostics additionally expose matching alpha lower
bounds, eta, sigma minima, side rconds, sigma_min/eta, dual bounds, primal lower,
G_upper, conservative normalized gap and feasible-proxy distance. Existing
certificate, Newton/CG, line-trial, dtype/backend and fallback diagnostics remain.
Detailed rank/posterior, evaluation and action histories support failure audits.
Unavailable failed-evaluation quantities are explicitly absent/None rather than
fabricated successful certificates.

## Checkpoint contract

New QSO checkpoints use **format 3** and explicitly serialize `admission_policy`
in every pair's solver configuration. Model tensors, pair topology, fp32
canonical EMA, fp64 original-coordinate lambda and counters are unchanged.
Loading restores the serialized policy even when the receiving optimizer was
constructed with the other policy.

Historical format 1/2 checkpoints lacking the field map explicitly to
`v0_rcond`; they cannot silently become v1 by being loaded into a v1 constructor.
Historical format-1 momentum-dtype and legacy `None` tolerance/guard migration
remain unchanged. A purported historical checkpoint carrying a v1 policy is
rejected. Format 3 without an explicit policy is rejected. Named pair topology,
state shape/dtype/finite-value validation and explicit state tensor restoration
remain mandatory.

## Equivalence and v0 nonregression evidence

The unchanged research solver is used as an independent arithmetic oracle.
Package-level regressions cover prescribed/repeated spectra at multiple scales,
above/below-old-guard full rank, numerical ambiguity, exact deficiency,
nonunique duals, the primary/P-dagger counterexample, intrinsic zero,
cancellation, nonfinite HVP, unusable curvature, no descent, invalid trials,
line-search/Newton-budget exhaustion, absence of research imports/extra posterior
SVD, late-pair atomicity, new checkpoints and historical migration.

On the five saved Stage-D blockers, package and research return exactly the same
pairs and original-coordinate lambdas at Newton iterations **3,3,3,4,4**.
All seven ordinary saved controls retain their stopping iterations.
Newton/CG/line counts, full histories, rank/value posteriors and selection
metadata match exactly. No hidden reference is used.

Default-v0 solves on all twelve saved inputs are compared independently against
the original committed solver, including rejected below-guard cases. Returned
pair/lambda, success/failure, reason, counts, histories and every pre-existing
metric match exactly. Source checks also confirm that `_solve`, `SmoothDual`,
`_newton_direction`, `certificate` and the radial backend are unchanged.
The intentional additive diagnostics and format-3 checkpoint metadata do not
alter v0 numerical arithmetic or admission.

## Seed-2027 integration replay

The trajectory uses exactly the frozen seed-2027 configuration and local data
from [QSO_MULTISEED_512STEP.md](QSO_MULTISEED_512STEP.md): 512 updates, 2048 tokens
per update, 1,048,576 nonrepeated tokens, paired peak LR
`0.0012247448713915891`, unsupported AdamW `3e-4`, 51-step warmup, cosine decay
to 10%, bf16 forward/backward, fp32 storage/EMA, six registered pairs, and
validation at steps 0,32,...,512. No AdamW reference trajectory is rerun.

The integration runner verifies initialization, data prefix/permutation,
minibatch and validation hashes, configuration and unchanged historical
model/data/training/research source hashes. Only the intended package changes
and explicit policy/fallback metadata differ. The package API controls every
update. The research solver independently audits each captured pair problem
**after the timed training step**, without proposing model updates or controlling
branches. Its duplicate work is logged separately and excluded from package
step timing.

The replay completed **512/512 updates and 1,048,576 nonrepeated tokens**.
All **3,072/3,072** package pairs certified; there were no rank ambiguities,
invalid numerical actions, solver failures, CPU reference calls or optimizer
fallbacks. Every returned pair and lambda is bitwise identical to the untouched
research oracle on the identical captured problem. Full Newton/CG/action/trial
histories and posteriors also match, excluding timing fields only. All 512
minibatch hashes, losses, gradient/parameter/update diagnostics, and all 17
validation losses match the recorded historical seed-2027 trajectory exactly.
Final validation is **4.441222548484802**; the final-three mean is
**4.454021334648132**, both identical to the research run.

| Integration numerical result | Measured value |
| --- | ---: |
| Final solves below the old guard | 102 / 3072 (3.3203%) |
| Below-guard zero-based steps | 90–122 |
| Counts by block 0 / 1 / 2 / 3 | 20 / 33 / 30 / 19 |
| U-only / D-only / both | 2 / 20 / 80 |
| Minimum raw rcond | 3.921010644689864e-5 |
| Minimum alpha_lower | 2.3909714428911524e-6 |
| Minimum sigma_min / eta | 1,184,319.8188783869 |
| Maximum conservative normalized gap | 2.998690238653861e-5 |
| Maximum normalized horizontality | 1.2952590438790532e-16 |
| Maximum spectral excess | 0 |
| Maximum exact-proxy distance bound | 4.435118974279257e-11 |
| Newton bins 0 / 1 / 2 / 3+ | 0 / 0 / 396 / 2676 |
| CG median / p95 / max | 9 / 14 / 26 |
| Line trials median / p95 / max | 3 / 3 / 4 |
| Individual smooth matrix SVDs | 23,796 |

Every count and numerical extremum in this table matches the historical
research trajectory. No old-guard event was silently removed or treated as a
new scalar threshold.

At step 256, retained historical model and optimizer tensors compare exactly:
model parameters, canonical momenta, original-coordinate lambdas, counters,
unsupported AdamW state and scheduler. Only intentional schema metadata is
normalized: format 2 to 3, the explicit admission policy, and fallback True to
False. New format-3 disk reload independently reproduces steps **257,258,259**
exactly: batches, losses, admission decisions, model and optimizer state.
The new checkpoint occupies **118,734,979 bytes**.

The old run retained no final checkpoint, so a direct historical step-512 tensor
comparison is unavailable. All-step exact pair/direction/lambda oracle checks,
identical historical losses and update diagnostics, the step-256 state comparison,
and unchanged update/unsupported-AdamW arithmetic provide fixed-backend
continuation equivalence evidence; unavailable historical final tensors are not
claimed to have been directly compared.

## Integration cost

CUDA synchronization in the unchanged training harness bounds actual package
steps. Offline duplicate research solves run after those timers and cannot
influence updates. First five training steps are retained in the raw logs and
excluded only from warm descriptive statistics, matching the reference rule.
The comparison uses different H100 allocations, not a paired end-to-end timing
experiment, and does not establish a speedup.

| Warm metric | Package median / p95 (s) | Historical research median / p95 (s) |
| --- | --- | --- |
| Full step | 1.473478 / 1.522888 | 1.478663 / 1.526443 |
| Six paired solves | 1.433779 / 1.483072 | 1.437239 / 1.484909 |
| Total optimizer | 1.448220 / 1.497473 | 1.452524 / 1.500300 |
| Forward/backward | 0.011597 / 0.011933 | 0.012217 / 0.012469 |

Measured package step sum is **734.172 s**, optimizer sum **721.036 s**,
throughput **1428.24 tokens/s**, and optimizer fraction **98.21%**. The warm
step ratio to recorded research is **0.99649x**: no material regression is
observed. The entire audited job took **1457.254 s** inside the runner, including
**716.128 s** of duplicate offline research-oracle work. That total is not package
training throughput. Peak PyTorch allocation was **588,241,920 bytes**; the
measurement includes audit capture/oracle lifetimes, so it is not an isolated
uninstrumented deployment memory estimate.

Separately, warmed alternating package/research solves on the same saved inputs
in the same allocation measured median package/research time ratios **1.00325x**
for ordinary controls and **1.00281x** for Stage-D blockers. The entire sample
range was approximately **0.992–1.0075x**. This small-sample result supports
negligible integration overhead and is not a performance-optimization claim.

## Validation and reproducibility

The complete existing suite plus package regressions passed: **568 tests**.
No existing test or acceptance threshold was weakened. An initial new
counterexample assertion incorrectly required the solver's recovered polar to
be far from P-dagger; it was corrected to test the independent feasible primary
candidate from the authoritative counterexample, and its primary-only metadata.
The raw initial test failure remains available.

Validation used one H100 per SLURM job, four CPUs and 16 GiB, Python 3.10.20,
torch 2.10.0+cu128, CUDA build 12.8, deterministic algorithms and TF32 disabled.
Local assets and the existing offline environment were reused, with no download,
installation, AdamW rerun, extra seed, sweep or performance research. The final
source review corrected only the failed-result Optional type annotation and
module description after the trajectory; numerical functions and arithmetic
remain identical; the final suite/fixture gate passed on that source (568 tests
and all twelve saved inputs).
The pre-existing AGENTS.md edit was preserved.

Artifacts and reproduction:

- [Package regressions](../tests/test_production_v1.py).
- [Full-suite and fixture gate](../cluster/validate_production_v1.py),
  [SLURM entry](../cluster/validate_production_v1.sbatch).
- [One seed-2027 replay](../cluster/production_v1_512step.py),
  [SLURM entry](../cluster/production_v1_512step.sbatch).
- [Initial gate log](../cluster/production_v1_gates-29104.log),
  [expanded gate log](../cluster/production_v1_gates-29106.log),
  [final-source gate log](../cluster/production_v1_gates-29107.log),
  [integration log](../cluster/production_v1_replay-29105.log).
- Saved gate summaries under
  `/home/prignano/qnormuon-runs/production-v1/gates-29104/`, `gates-29106/`
  and `gates-29107/`.
- Full replay provenance, numeric logs, validation, historical checkpoint audit,
  new checkpoint, resume audit and summary under
  `/home/prignano/qnormuon-runs/production-v1/trajectory-29105/`.

From the repository root, `sbatch cluster/validate_production_v1.sbatch` runs
unit/fixture gates only. The explicitly requested integration replay is
`sbatch cluster/production_v1_512step.sbatch <gate-summary.json>`; its gate source
hashes must match the package. Neither entry submits further experiments.

**Decision:** opt-in integration is validated. A separate performance engineering
phase under the v1 contract is justified; no such work is started here.

## Limitations and next gate

Production-v0 remains the default. V1 is opt-in and scoped to primary epsilon-LMO
full-rank admission; no deficient-face selection, P-dagger claim, arbitrary
lower rcond, direction threshold or alternative smooth decomposition is added.
The existing tiny-model quality evidence is not a universal optimizer claim.
Integration does not address the high fp64-SVD implementation cost.
Only a completed classification-A integration justifies a separate performance
engineering task; this task does not begin that phase or larger training.
