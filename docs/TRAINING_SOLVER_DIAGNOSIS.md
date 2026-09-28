# Real Transformer pair precision diagnosis

Date: 2026-09-28. **Diagnosis only: production sources, defaults, thresholds,
and the smoke configuration are unchanged.** No LR sweep, installation, download,
or generic ADMM solve was performed in this study.

**Recommendation:** for the next controlled training validation, promote the
pair solver directly to fp64, retain the existing training gap target `3e-5`
and rcond guard `1e-4`, and retain fp32 canonical momentum. Do not spend time
on a failed fp32 attempt first. Do not put automatic CPU ADMM in the normal
training path. This recommendation is supported by all six real cold pairs and
an eight-step diagnostic, not by a completed 50-step benchmark.

An important qualification: simply switching to the fp64 **default target
`1e-8`** is not a reliably validated training policy. It solved every cold pair,
but a populated warm state exposed a separate line-search arithmetic limitation.
That failed run was stopped and preserved. The successful eight-step run used
`3e-5`, the original training threshold; no threshold above `3e-5` was tried.

## 1. Reproduction, corpus, and controls

The authoritative starting evidence remains
[TINY_TRANSFORMER_RESULTS.md](TINY_TRANSFORMER_RESULTS.md). The new driver is
[training_solver_diagnosis.py](../cluster/training_solver_diagnosis.py), launched
by [training_solver_diagnosis.sbatch](../cluster/training_solver_diagnosis.sbatch).
Every numerical operation ran in SLURM on one H100 NVL, with 4 CPUs, 16 GiB RAM,
and a 15-minute limit. Frontend work was file editing and small log/JSON inspection.

Runtime: `/home/prignano/modded-nanogpt/.venv/bin/python`, Python 3.10.20,
torch 2.10.0+cu128, CUDA build 12.8, NumPy 2.2.6. Tests use project-local
pytest 9.0.3. TF32 was disabled; deterministic algorithms and the existing
cuBLAS workspace setting were retained. The existing network guard was active;
no attempted-network log was produced for these jobs.

The unchanged smoke configuration initializes the 11,457,408-parameter,
six-layer SwiGLU Transformer with seed 2026. It reads the same bounded local
FineWeb SP1024 prefix and the same first minibatch. Forward/backward uses bf16
autocast with fp32 parameters and loss reduction. Reconstructed first loss:
**7.010502815246582**, exactly matching Stage C. No model parameter changes
while collecting the corpus; equality to pre-backward snapshots is asserted.

For each named pair, canonical weights are obtained using the existing
`regular_canonicalize(..., dtype=float32)`. Canonical raw gradients transform
once, are rounded to fp32, and are multiplied by `1-beta=.05`, exactly matching
the initial EMA update. Each retained problem contains U, D, stacked A_U/A_D,
name, objective RMS, `initial_lambda=None`, and its explicit original-coordinate
initial multiplier. `None` means internal z=0, hence original lambda equals the
vertical-centering multiplier, not an assumed literal zero. Every cold policy
uses this same initialization and the same represented fp32 inputs.

Corpus:
`/home/prignano/qnormuon-runs/tiny-transformer/precision-28899-cold/step0-pairs.pt`
(**37,805,715 bytes**). Its sibling `corpus.json` records configuration, data
prefix hashes, objective scales, and SHA256
`7eea5e05211483e5e9cdde70fa5cb35bb05c13e25caad226769e0d54ec86ccf3`.
Tensor artifacts remain outside the source tree. No model checkpoint was needed.

| Pair | Shape U and D | Initial objective RMS |
| --- | --- | ---: |
| blocks.0.mlp | 1024 x 384 | 3.24558394e-5 |
| blocks.1.mlp | 1024 x 384 | 2.62894465e-5 |
| blocks.2.mlp | 1024 x 384 | 2.51567331e-5 |
| blocks.3.mlp | 1024 x 384 | 2.25673763e-5 |
| blocks.4.mlp | 1024 x 384 | 2.17211814e-5 |
| blocks.5.mlp | 1024 x 384 | 1.95493386e-5 |

The solver algorithm, centering, whitening, CG, damping, recovery and
certification functions are the unchanged production implementation. Experimental
configuration and temporary instrumentation are confined to the diagnostic
process. Fallback is disabled throughout. Full-rank fp64 solves provide the
high-accuracy comparison; a CPU ADMM oracle was unnecessary.

## 2. Six-pair policy comparison

Job **28899**, completed in 2m32s. CUDA full-size SVD and certificate paths in
both precisions were warmed before headline measurements. A is the current
fp32 solve, with its `3e-5` target. B uses direct fp64 at the stricter default
`1e-8`; a separate B run matches `3e-5`. All runs explicitly keep `rcond_guard=1e-4`.
Thus fp64 success is not caused by lowering the conditioning guard.

C/D/E start from **A's returned original-coordinate lambda**, recompute in fp64,
and allow at most 1/2/4 Newton iterations. They stop earlier if the strict
`1e-8` certificate passes. F recomputes fp64 polars/recovery/certification at
A's multiplier with zero Newton iterations. C/D/E/F are independently measured,
not chained into one increasingly expensive continuation. Their total time
includes A's failed attempt. Both acceptance targets are shown explicitly.

| Policy | Pass existing 3e-5 | Pass requested strict 1e-8 | Six-pair seconds |
| --- | ---: | ---: | ---: |
| A: fp32 | 0/6 | Not requested | 13.883 |
| B: direct fp64, 1e-8 | 6/6 | 6/6 | 3.902 |
| B: direct fp64, matched 3e-5 | 6/6 | Not requested | **2.992** |
| F: A + fp64 recovery, no Newton | 4/6 | 0/6 | 14.809 |
| C: A + up to 1 fp64 Newton | 6/6 | 2/6 | 15.432 |
| D: A + up to 2 fp64 Newton | 6/6 | 5/6 | 16.156 |
| E: A + up to 4 fp64 Newton | 6/6 | 6/6 | 16.460 |

A capped run with a gap below `3e-5` but above its requested `1e-8` still has
`converged=False` in the raw record. The table's existing-target column is an
explicit alternative certificate check, including feasibility and conditioning;
it does not silently relabel strict-target failures.

Normalized gaps for every pair:

| Pair index | A fp32 | B fp64 strict | F zero refinement | C one | D two | E four |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.15922e-5 | 1.68834e-10 | 3.15436e-5 | 5.30888e-8 | 5.14295e-8 | 3.13891e-12 |
| 1 | 4.46410e-5 | 2.41505e-10 | 4.87245e-7 | 1.11360e-9 | 1.11360e-9 | 1.11360e-9 |
| 2 | 4.56869e-5 | 1.88409e-11 | 3.29257e-6 | 2.60712e-8 | 3.02867e-12 | 3.02867e-12 |
| 3 | 9.63812e-5 | 5.82684e-10 | 8.09799e-5 | 3.36590e-7 | 5.76332e-10 | 5.76332e-10 |
| 4 | 5.18953e-5 | 1.59796e-9 | 2.20276e-5 | 2.41322e-7 | 8.28561e-11 | 8.28561e-11 |
| 5 | 5.05438e-5 | 2.15817e-9 | 7.54316e-7 | 2.51634e-9 | 2.51634e-9 | 2.51634e-9 |

All A failures are `line_search_failed`. A's Newton counts are
`[5,8,14,2,6,10]`, HVP counts `[14,24,32,7,18,43]`, matrix SVD counts
`[96,140,234,66,152,154]`, and line-search evaluations `[35,51,86,26,61,54]`.
B strict needs four Newton steps and four line evaluations on every pair,
with HVP counts `[10,10,12,11,11,14]` and 30 matrix SVDs each. C/D/E incremental
Newton counts are respectively `[1,1,1,1,1,1]`, `[2,1,2,2,2,1]`, and
`[4,1,2,2,2,1]`. Matched-target B uses `[3,3,2,3,3,3]` Newton steps.

The [complete structured cold results](../cluster/training_solver_precision-28899.json)
record **every policy's** primal/dual objectives, signed/normalized gap,
horizontal residual, spectral excess/norms, rcond, Newton/HVP counts, SVD counts,
line-search evaluations, history, and timing. Hybrid total counts are also
provided; counts for the incremental refinement remain separate. A batched
SVD of two matrices counts as two matrix SVDs.

## 3. What fails in fp32

There are **two linked effects**: inaccurate residual SVD/polar factors limit
the quality of feasible recovery, and inaccurate nuclear-norm values obstruct
line search. The failure is not simply a badly optimized dual value or final
gap subtraction in fp32. Production recovery and certification already use fp64.

### Same multiplier, same candidate controls

For pair 0 at A's returned lambda:

- True fp64 dual: **0.5061762965265763**; recovered primal:
  **0.5061501817753614**. Their difference is **2.6114751215e-5**.
- The fp32-generated raw polar has spectral norm **1.0000919506**.
  Horizontal projection gives radial scale **1.0000920735**.
- Raw/projected primal values are **0.5061967847423 / 0.5061967847728**;
  radial feasibility scaling removes **4.6602997453e-5** of objective.
  The raw value exceeds the dual because the candidate is infeasible.
- Replacing only the polar by an fp64 SVD of the **same rounded internal
  residual** reduces the recovered gap from **5.15922e-5 to 3.15436e-5**.
  Using the exact fp64-formed original residual gives **3.15435990e-5**:
  residual formation/normalization rounding is secondary here.
- The residual polar's relative error against an fp64 decomposition of that
  same rounded matrix is **5.27426e-5**, and its spectral Gram defect is
  **1.83910e-4**. The corresponding relative polar errors across six pairs
  are **5.27e-5–5.40e-5**.
- This lambda's dual exceeds the high-accuracy fp64 solution's dual by only
  **6.81e-12**. Nevertheless, its remaining polar stationarity error requires
  radial recovery and leaves a `3.15e-5` gap even after accurate SVD. A small
  dual-value optimization error is not equivalent to an already good recovered
  primal; radial recovery can lose objective to first order in residual error.

For the same feasible candidate rounded to fp32, evaluating its inner product
in fp32 versus fp64 changes pair 0's primal by **4.30e-9**. Summing the same
fp32 singular values in fp32 versus fp64 changes the nuclear value by
**3.25e-8**. In contrast, recomputing the SVD in fp64 on the identical rounded
residual changes the nuclear value by **2.03e-5**. Thus singular-value accuracy,
not just the scalar summation accumulator, is a major issue.

Across all six pairs, nuclear SVD error in this control is **6.58e-6–2.03e-5**,
while primal reduction error is at most **1.69e-8**. Original residual formation
error is measured separately in the JSON. Final primal-dual subtraction is in
fp64 on values of order .2–.5; ordinary fp64 cancellation cannot explain the
observed absolute gaps of order 1e-5. No cancellation-dominated cotangent
trigger or rank-loss trigger occurred.

### Why line search rejects useful steps

The trace recomputes important scalars without changing any production decision.
In pair 0's final failed backtrack sequence, the full proposed step reduces the
fp32 gradient norm from **1.52638e-4 to 1.66905e-5**. The fp32 nuclear value
instead reports an increase of **4.57764e-5**, beyond its **4.19309e-6** rounding
allowance. An fp64 SVD on the same rounded residuals reports a change of
**-1.47828e-9**. The Armijo value comparison rejects the step despite the large
gradient improvement. Seventeen of its final 24 trials fail the value gate;
eventually z rounds back to the unchanged point and the gradient cannot improve.

All six final full trials fail the value gate. Their apparent fp32 increases
are **1.14e-5–4.58e-5**, whereas the corresponding fp64 changes on those rounded
residuals have magnitude at most **3.44e-9**. Some of those tiny fp64 changes
are positive: rounded residual formation also matters at that scale. This is
not evidence that every rejected trial should be accepted. It establishes that
the current fp32 value noise substantially exceeds the line-search allowance.

Changing only scalar reductions to fp64, loosening the line-search gate, or
performing more fp32 Newton iterations would not remove the independent
fp32 polar/recovery floor below. No such change was made.

## 4. Empirical fp32 floor and conditioning

At each accurate fp64 multiplier, reconstruct the current normalized fp32
problem, recompute its fp32 polar, and use unchanged fp64 feasible recovery and
certification against the original inputs. Repeat three times. Results are
identical in these deterministic repetitions. Separate complete fp32 solves
also reproduce their counts and final gaps across three repeats.

| Pair | Gap with fp32 polar at fp64 lambda | Gap with fp64 polar merely stored in fp32 | Optimal residual rcond |
| --- | ---: | ---: | ---: |
| 0 | 4.24733e-5 | 2.21007e-8 | .00758757 |
| 1 | 4.02872e-5 | 2.27169e-8 | .00788653 |
| 2 | 4.45720e-5 | 2.33058e-8 | .00590656 |
| 3 | 4.60715e-5 | 2.34419e-8 | .00496293 |
| 4 | 4.69549e-5 | 2.39583e-8 | .00315123 |
| 5 | 5.00948e-5 | 2.33235e-8 | .00212402 |

**`3e-5` is below the stable observed floor of this fp32 SVD/polar/recovery
path on all six problems.** This is an empirical statement about these inputs,
this backend, and this algorithm; it is not a theorem that no fp32 algorithm
or differently generated candidate could certify. The cast-only control
shows that fp32 storage itself can represent a much more accurate direction.
No alternative SVD driver or kernel was adopted or benchmarked here.

Machine epsilon is **1.19209e-7**. The floor is roughly **338–420 eps**;
`384*eps = 4.57764e-5`, of the observed order. This suggests accumulated
matrix-size-dependent error, not a universal error bound or proof of a vendor
implementation defect. Actual dual magnitudes are **.20356877–.50617630**;
one epsilon times the objective is only **2.43e-8–6.03e-8**. The final gap
is hundreds of such units. Recovered polar radial scales at the fp64 lambda
are approximately **1.0000832–1.0000847**.

The cold problems are numerically full-rank and smooth, with rcond
**.002124–.007887**, at least 21 times the retained guard. This does not assert
universal safety at future steps or prove a well-conditioned multiplier Hessian.
No positive singular value was truncated. The accepted distinction between
value gap and direction accuracy near rank loss remains unchanged.

## 5. Repeated direct-fp64 cost and device verification

Three additional uninstrumented direct-fp64 solves per pair, strict `1e-8`:

| Pair | Median seconds | Range seconds | Newton / HVP | Matrix SVDs |
| --- | ---: | ---: | --- | ---: |
| 0 | .64241 | .64216–.64263 | 4 / 10 | 30 |
| 1 | .64881 | .64836–.64950 | 4 / 10 | 30 |
| 2 | .64797 | .64797–.64876 | 4 / 12 | 30 |
| 3 | .65000 | .64984–.65091 | 4 / 11 | 30 |
| 4 | .64947 | .64943–.65022 | 4 / 11 | 30 |
| 5 | .64667 | .64632–.64685 | 4 / 14 | 30 |

Summing corresponding repetitions across pairs gives **3.88750, 3.88623,
3.88453 seconds**. These are six pair-solver costs, including mandatory
certification; they exclude canonicalization, update lifting, AdamW, forward/
backward and optional optimizer cast diagnostics. The matched-target cold
six-pair cost is **2.99163 seconds**; its run uses fewer Newton iterations.

A separate synchronized profile of six strict solves takes **3.89640 s**:

| Component | Seconds | Interpretation |
| --- | ---: | --- |
| Residual SVD calls | 1.28010 | 60 matrix decompositions |
| Certificate SVD-values calls | 2.49564 | 120 matrix decompositions |
| All SVD/SVD-values calls | **3.77574** | About 97% of this profiled solve time |
| Complete recovery/certification | 2.52473 | Includes the 2.49564 s above |
| CG including cached HVPs | .04586 | No SVD per HVP |
| Line-search value/gradient evaluations | 1.03459 | Overlaps residual SVD time |

The last three categories overlap the first two; **do not sum all table rows**.
Timers synchronize CUDA and include host dispatch/wait costs. They are not a
kernel-only hardware profiler or an SM-occupancy measurement. Instrumented
runs are kept separate from headline timings.

Instrumentation checks every SVD/SVD-values input is on **CUDA**, records dtype
and device, and would fail if a spectral call moved to CPU. Both fp32 and fp64
smooth paths use H100 tensor operations. All reference iteration counts are
zero. The earlier Stage-C CPU ADMM costs remain separate (about 40–50 seconds
per completed layer solve including its failed fp32 prefix); they are not
included in these GPU smooth-solver timings.

## 6. Warm-start training diagnostics

Only the pair-solver call is experimentally promoted to fp64. Canonical weights,
gradient pullback and EMA remain fp32; model storage remains fp32 and forward/
backward remains bf16 autocast. Stored original lambda remains fp64. State
inspection confirms these dtypes for all six pairs. Merely setting the existing
optimizer's entire `SolverConfig.dtype=float64` would also change canonical EMA;
that is not the controlled experiment performed here.

The unchanged LR/configuration is used: QSO peak LR .001, unsupported AdamW
peak LR .0003, beta .95, original five-step warmup and 50-step schedule. The
sequence stops after eight diagnostic steps, not a shortened/rescaled schedule.
Each actual warm solve is paired with a separate cold solve of the identical
current inputs; only the warm result is applied. Counterfactual solves do not
change parameters or optimizer state. No generic fallback is invoked.

### Strict-target negative result

Job **28900** completed steps 0 and 1, then stopped on step 2,
`blocks.0.mlp`: four Newton iterations, 14 HVPs, `line_search_failed`, gap
**3.63990682e-8** versus experimental target `1e-8`, rcond **.01017428**.
No step-2 update was committed. Its cold counterfactual certified in four
iterations with gap **2.06444e-9**. Step 1's warm counts `[4,4,5,4,4,5]`
did not improve on cold `[4,4,4,4,4,4]` at this strict target.

Job **28904** deterministically reproduced that failure and retained the
6,302,199-byte input `failed-warm-pair.pt` under
`/home/prignano/qnormuon-runs/tiny-transformer/precision-28904-mini/`.
The [failure trace](../cluster/training_solver_strict_failure-28904.json)
shows the final full trial reducing gradient norm from **2.14320e-7 to
4.69062e-11**, but a nuclear-value increase **3.90799e-14** exceeds the
**8.01099e-15** rounding allowance. Predicted decrease is only **9.15131e-16**.
This is another arithmetic-resolution problem in the existing value gate,
now in fp64 at a much stricter accuracy, not a rank-loss event. No line-search
rule was changed. The strict failed trajectory is distinct from the successful
matched-target trajectory below.

### Existing-target successful result

Job **28902**, target **3e-5**, completed eight steps, **48/48 accepted pair
solves, zero fallback**, without nonfinite loss, gradient, parameter or update.
It processes 16,384 training tokens. These are diagnostic training updates,
not an optimizer quality comparison.

| Step | Loss | Warm Newton counts, layers 0–5 | Warm solver seconds | Cold counterfactual seconds | Max accepted gap | Min rcond |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 7.010503 | 3,3,2,3,3,3 | 3.317 | 2.976 | 2.276e-5 | .002124 |
| 1 | 6.866500 | 3,3,3,3,3,3 | 3.252 | 3.114 | 7.103e-6 | .001970 |
| 2 | 6.708051 | 2,3,3,3,3,3 | 2.984 | 3.114 | 1.938e-5 | .001699 |
| 3 | 6.620705 | 2,3,2,2,2,2 | 2.473 | 3.116 | 2.834e-5 | .001610 |
| 4 | 6.541110 | 2,3,2,2,2,2 | 2.587 | 3.132 | 2.734e-5 | .001551 |
| 5 | 6.474590 | 2,2,2,2,2,2 | 2.343 | 3.120 | 2.996e-5 | .001526 |
| 6 | 6.455825 | 2,3,2,2,2,2 | 2.465 | 3.121 | 2.664e-5 | .001514 |
| 7 | 6.419187 | 2,3,2,2,2,2 | 2.472 | 3.119 | 8.923e-6 | .001507 |

Cold counterfactuals take three Newton iterations for every pair after step 0.
Overall warm bins **0/1/2/3+ = 0/0/28/20**. Over steps 3–7, warm solver time
averages **2.46815 s**, versus **3.12147 s** cold: about **21% less**, with a
mean 12.8 versus 18 Newton iterations across the six pairs. This supports a
modest warm-start benefit at the existing target, not an unconditional two-step
rule. One accepted gap is **2.99575e-5**, close to the specified stopping target
because that iterate is the first to certify, not an fp64 precision floor.

Maximum normalized horizontal residual is **4.92e-17**, spectral excess is
zero under the evaluated certificate, and residual rcond stays above .0015.
Maximum relative warm-versus-cold recovered direction difference is
**3.11378e-5** at this tolerance. These finite-tolerance directions are not
claimed identical.

Peak PyTorch CUDA allocation is **554,204,160 bytes (~528.5 MiB)**. End-step
live allocation stays within **339–348 MB**; this short run does not establish
long-term memory stability. Step 0 includes first-use overhead. Logged total
step/optimizer times include the deliberately added cold counterfactuals and
JSON diagnostics, so their raw throughput is **not normal training throughput**.
Subtracting separately timed cold solves gives an approximate optimizer cost
of 2.68–2.92 s in steps 3–7, still including instrumentation/I/O. The directly
measured pair-solver sums above are the cleaner precision-policy comparison.
No full-size checkpoint/resume or gauge-reset validation is claimed here.

The [compact warm results](../cluster/training_solver_warm-28902.json) include
per-pair objective scales, gaps, rcond, counts, solver dtype, cold comparisons,
per-step loss and memory. Full histories remain in the shared output directory.

## 7. Answers and proposed production change

1. **Why does fp32 fail?** Its SVD/polar accuracy creates spectral inflation
   and a recovery gap floor; nuclear-value error also rejects useful line-search
   steps. Residual multiplier error adds recovery loss for some pairs. Final
   fp64 primal/dual subtraction and rank loss are not the main causes.
2. **Is 3e-5 below the stable fp32 floor?** Yes for the observed current H100
   path on all six cold problems: 4.03e-5–5.01e-5 at accurate multipliers.
   This is not an impossibility result for all fp32 implementations.
3. **Are the real problems smooth/full-rank?** Numerically yes: cold rcond
   .002124–.007887, successful mini-sequence minimum .001507, all above 1e-4.
4. **Does direct fp64 solve all six reliably?** Yes, including three repeated
   strict-target solves each. This does not imply all subsequent warm strict
   `1e-8` solves succeed; the retained step-2 counterexample disproves that.
5. **What is the cost?** About 3.886 s per six cold pair solves at `1e-8`, or
   2.992 s at the original `3e-5`; certification is a major component. Full
   optimizer cost also includes canonicalization, lifting, cast diagnostics
   and unsupported AdamW updates.
6. **Does warm-starting reduce it?** At `3e-5`, after the first few steps:
   about 2.47 s warm versus 3.12 s cold on the same current problems. It did
   not consistently help at the strict target before that run aborted.
7. **Is hybrid faster?** No for the requested policy starting with a complete
   current fp32 attempt. One fp64 refinement meets 3e-5 for all six, but total
   cost is 15.43 s versus direct fp64's 2.99 s. Strict four-step-capped hybrid
   costs 16.46 s versus 3.90 s direct. An early-abandoned fp32 prefix is a
   different, unmeasured policy; no advantage is assumed.
8. **Should generic ADMM remain in the normal chain?** No automatic CPU ADMM
   for ordinary training. Keep it available explicitly for research/oracle
   diagnosis. A genuine unresolved solve should stop with diagnostics rather
   than silently consume tens of seconds per layer or commit an uncertified step.
9. **Exact recommended change, not implemented:** separate canonical momentum
   precision from spectral-solver precision; keep canonicalization/EMA fp32,
   promote represented U/D/A and Newton-CG/SVD work directly to fp64, keep
   original lambda/recovery/certification fp64, preserve target **3e-5**, guard
   **1e-4**, initial budget 2 and adaptive certification. Keep model/update
   storage unchanged. Bypass the fp32 trial and make CPU reference fallback
   opt-in for training. Preserve explicit failure, capture the offending pair,
   and retain all conditioning/feasibility checks. Do not accidentally inherit
   the existing fp64 default target `1e-8` when choosing the work dtype.

A more permissive line-search arithmetic allowance, a different SVD driver,
certificate factor reuse, or a new fallback restart policy would each be a
separate numerical change needing validation. None is part of this diagnosis.
The observed strict-target failure should inform that later work, not be hidden
by claiming fp64 universally fixes convergence. No mathematical contradiction
or reason to redesign the coupled LMO was found.

## 8. Tests, raw artifacts, and limits

Full suite job **28903**: **202 passed, 0 failed, 0 skipped**, pytest **14.44 s**,
allocation 18 s. All 201 previous cases are unchanged. One new small analytic
control in [test_training_solver_precision.py](../tests/test_training_solver_precision.py)
injects anisotropic polar error at an exact, well-conditioned dual optimum and
verifies that feasible radial recovery can fail the existing gap threshold.
Accurate polar recomputation at the same lambda then certifies. It isolates
the mechanism; it does not assert vendor-specific SVD error on every platform.

Initial test job 28901 had all 201 existing tests pass and the new control fail
on an inappropriate exact-equality assertion (`1.0000000000000002 == 1`). Only
that new assertion was corrected to a two-fp64-epsilon bound. Production
thresholds and existing tests were not weakened. Its raw failure is retained.

Raw logs:

- [28899 cold study](../cluster/training_solver_diagnosis-28899.log)
- [28900 strict mini-sequence failure](../cluster/training_solver_diagnosis-28900.log)
- [28902 successful eight-step diagnostic](../cluster/training_solver_diagnosis-28902.log)
- [28904 strict failure replay](../cluster/training_solver_diagnosis-28904.log)
- [28901 initial new-test assertion](../cluster/test_tiny_harness-28901.log)
- [28903 complete passing suite](../cluster/test_tiny_harness-28903.log)

Each study output is under
`/home/prignano/qnormuon-runs/tiny-transformer/precision-JOBID-PHASE/`, with
`environment.json`, the unchanged input configuration and production hashes.
The successful sequence retains compact metrics, not repeated model checkpoints.
Strict failure runs intentionally exit nonzero and preserve tracebacks.

Reproduce only inside SLURM:

```sh
sbatch --parsable cluster/training_solver_diagnosis.sbatch cold
sbatch --parsable cluster/training_solver_diagnosis.sbatch mini --tolerance 3e-5
# Expected failure reproducer, stops at the failed strict warm solve:
sbatch --parsable cluster/training_solver_diagnosis.sbatch mini --tolerance 1e-8
sbatch --parsable cluster/test_tiny_harness.sbatch
```

Verified unchanged production SHA256 values:

```
df6552136f7a5c5e121622e97abd1e30c70762a0ee8422d61a02df54bd603ec6  qnormuon/optimizer.py
8345e386f00d9d9937d18ca512198bc3e37367fbb4a45f826874411e63b868c2  qnormuon/coupled_solver.py
```

This is one initialization, six cold problems, and one eight-step successful
trajectory at an unswept LR. It does not establish general training robustness,
quality superiority, full-run performance, gauge trajectories, checkpoint
continuation, or safety at future rank loss. No production change, broad
optimizer redesign, or LR sweep follows automatically from this report.
