# QDWH inner solver with authoritative production-SVD output

## Changed numerical contract

This is research toward a distinct production-v1 **numerical policy**, not an
exact-v0 backend replacement. Production-v0 code and defaults remain unchanged.
The coupled dual/LMO, canonical state conventions and acceptance thresholds are
unchanged. Intermediate Newton-CG, objective, polar derivative and ordinary
scalar Armijo decisions use raw fp64 QDWH/H-EVD values. They are not claimed
branch-equivalent to full SVD or proven by the expensive decision posterior.

The existing direct QDWH structural screen remains: convergence, finite values,
orthogonality, factorization and symmetry defects, H SPD, and its numerical
rcond screen. No decision-bound derivative, curvature, CG or Armijo interval
validator runs on the inner path.

The candidate multiplier is expressed in the original convention. An inner
success is **not permission to return its direction**. Outside the temporary
QDWH adapter, production `SmoothDual` evaluates the original residual pair at
that exact multiplier using full fp64 SVD. The production recovery rebuilds
the primal from those SVD polars and uses `gram_upper` radial feasibility. Its
certificate uses that same authoritative full-SVD spectrum (magnitude one).
Only this evaluation, or an ordinary full-SVD production rescue, may authorize
the returned direction. There are zero QDWH-only returned updates.

All accepted nonzero directions must satisfy normalized gap `<=3e-5`, signed
normalized gap `>=-1e-10`, normalized horizontality `<=1e-10`, spectral excess
`<=1e-12`, and residual rcond `>1e-4`. Nonfinite diagnostics never authorize.
An unsuccessful rescue is visibly uncertified and cannot be applied by the
optimizer. The research acceptance helper deliberately requires the smooth
rcond guard even after reference rescue; it is not an extension to zero
cotangents or deficient residual faces. Production's exact-zero and deficient
reference contracts are untouched.

## Fixed policies and work limits

The study compares two and three QDWH Newton iterations, with a fixed global
cap of seven residual-pair QDWH evaluations. The existing 24-trial numerical
Armijo rule is retained, but the global cap prevents long runs of decompositions
before rescue. No budget is fit per layer or per step.

If final verification rejects, policy A continues the ordinary SVD solver from
the candidate multiplier; policy B restarts it from the original warm lambda.
On a completed but uncertified inner solve (budget, line-search failure or
no descent), A uses its best reported candidate and B uses the original warm
lambda. An interrupted structural/evaluation-budget failure has no completed
candidate, so both policies rescue from the original warm start. Every reason
is recorded. Rescue uses the existing production solver and its configured CPU
ADMM option; CPU reference fallback is counted separately from SVD rescue.

## Reproducibility and stage gates

- `experiments/smooth_qdwh_hybrid.py`: isolated raw-QDWH adapter, authoritative
  verifier and explicit rescue.
- `tests/test_smooth_qdwh_hybrid.py`: independent full-SVD authority checks,
  rejection/rescue policies, budget failure, unchanged thresholds and scoping.
- `cluster/study_qdwh_hybrid_h100.sbatch`: complete suite and locked-pair study,
  one H100, four CPUs, 16 GiB, offline shared-home environment.
- `cluster/study_qdwh_hybrid_h100.py`: four fixed policies on the same locked
  pair problems plus the existing near-guard synthetic corpus.

The direct-SVD trajectory controls replay model state; hybrid solves are
counterfactual on identical inputs. Independent full-SVD audits occur after
each candidate and are excluded from candidate timing. All inner, verifier and
rescue work is included. Controlled short training and then the locked 50-step
run require complete certified replay, low rescue frequency and meaningful net
speed improvement.

## Decision

**A: the tested hybrid provides no useful net benefit.** A three-Newton inner
budget proposed SVD-accepted multipliers for all 300 locked problems, with zero
rescue or CPU ADMM. However it was only about **1.009x** faster across the whole
replay, and **0.998x** as fast in the warm portion. Warm six-pair median solver
time was slightly worse than direct SVD. This does not meet the material-speed
gate. Consequently no hybrid-controlled short trajectory or locked 50-step
candidate training run was launched. Production-v0 was not modified.

The result is specific to the measured prototype: the current production
iterate recovery/certificate is reused for provisional inner stopping, followed
by an additional mandatory SVD evaluation and recovery for authoritative output.
No claim is made that every possible inexact-inner/exact-final policy must have
this cost. No alternate stopping heuristic or certificate-frequency policy was
introduced after observing the result.

## Environment, tests and provenance

SLURM jobs `29003` (full suite and three-step screen) and `29014` (complete
locked replay) completed successfully with exit `0:0`. Each requested one H100,
four CPUs, 16 GiB RAM and a 30-minute cap. Numerical/model execution was entirely
inside the allocation. The existing shared-home interpreter was
`/home/prignano/modded-nanogpt/.venv/bin/python`: Python 3.10.20,
torch 2.10.0+cu128, CUDA build 12.8, NVIDIA H100 NVL. The offline network guard
was active and recorded no outbound Python attempt. Nothing was installed or
downloaded. The complete suite passed **392 tests, zero failed, zero skipped**,
in **16.09 s**, including 14 new research regressions. The unchanged Python
code reused that validation for the full replay rather than repeating tests.

The replay reconstructed the six pair inputs at every locked step from the
same 11.46M model, seed 2026, FineWeb token prefixes, batch stream, schedule,
learning rates, bf16 autocast, fp32 model/momentum and original fp64 lambda.
Steps 0–24 began from seeded initialization; steps 25–49 began from the already
local production checkpoint. Data/batch hashes and losses matched the locked
baseline exactly because **production SVD controlled model state**.
That identity is not a hybrid-controlled trajectory result.

The source run was
`/home/prignano/qnormuon-runs/tiny-transformer/smoke-28936-qso-seed2026`.
Detailed configuration and provenance are in the job log and result JSON.
Raw artifacts:

- `cluster/study_qdwh_hybrid_h100-29003.log`
- `cluster/qdwh_hybrid_tests-29003.xml`
- `cluster/qdwh_hybrid_study-29003.json`
- `cluster/study_qdwh_hybrid_h100-29014.log`
- `cluster/qdwh_hybrid_replay-29014.jsonl`
- `cluster/qdwh_hybrid_study-29014.json`

Reproduce from the project root with
`sbatch cluster/study_qdwh_hybrid_h100.sbatch`. No production dtype/backend
configuration is migrated by this script.

## Complete locked-pair replay

The four policies each solved the **same 300 problems**; thus 1,200 hybrid
results were checked. A residual-pair decomposition evaluation means two
matrices. SVD counts include all such evaluations within a candidate path;
there were no guarded primal-SVD fallbacks on these real inputs.

| Quantity | Direct SVD | 2 Newton / A continue | 2 Newton / B restart | 3 Newton / A continue | 3 Newton / B restart |
|---|---:|---:|---:|---:|---:|
| Final certified pairs | 300 | 300 | 300 | 300 | 300 |
| QDWH pair evaluations | 0 | 900 | 900 | 931 | 931 |
| Explicit final-SVD verifications | — | 269 | 269 | 300 | 300 |
| Failed explicit final verifications | — | 0 | 0 | 0 | 0 |
| Full-SVD rescue solves | — | 31 | 31 | 0 | 0 |
| Total SVD pair evaluations | 931 | 331 | 393 | 300 | 300 |
| Total QDWH + SVD evaluations | 931 | 1,231 | 1,293 | 1,231 | 1,231 |
| Total solver time, s | 51.082 | 51.412 | 54.540 | 50.608 | 50.600 |
| CPU ADMM fallback | 0 | 0 | 0 | 0 | 0 |

With two inner Newton iterations, 269/300 problems (89.7%) passed the inner
stopping test and then the explicit final SVD verifier. The remaining 31
exhausted the inner Newton budget. Continue rescue used 62 SVD evaluations
(two per rescued problem); restart used 124 (four per problem). These rescues
include their own authoritative SVD recovery/certification; they do not require
another standalone verifier after a certified production solve.

With three inner iterations, **300/300 proposals passed the final SVD verifier**.
There were no structural rejections, evaluation-cap failures or rescue on the
real corpus. A versus B then differs only by unexercised rescue configuration;
their slight timing difference is measurement variation, not a policy gain.

Three-iteration policy removes **631/931 = 67.8%** of full-SVD evaluations, but
adds 931 QDWH evaluations. Total decomposition count increases by 32.2%. Two
iterations with continue saves 600 SVD evaluations; restarting saves 538.
Removing SVD calls alone is therefore an insufficient performance metric.

All real inner paths used ordinary numerical Armijo without decision intervals.
Three-iteration policies used 631 line-search trials, matching the direct count.
Two-iteration continue also totaled 631 trials across inner plus rescue;
restart totaled 693. No real backtracking storm occurred. Newton-work bins
(0 / 1 / 2 / 3+) were `0 / 0 / 269 / 31` for all policies and direct SVD.
For restart, the 31 rescued problems expended five Newton steps (two discarded
inner steps plus three restarted production steps); continue expended three.
These are work counts, not claims of SVD-identical intermediate decisions.

### Final authority and independent audit

Every returned usable pair came from the standalone full-SVD verifier or
the production-SVD rescue's accepted certificate. **There were zero QDWH-only
returns.** Every final result passed the unchanged five-condition check. Each
was additionally audited offline with independent original-residual full SVD
and full-SVD primal radial norm, outside measured hybrid time. All 1,200 audits
passed. The largest independently checked normalized gap was approximately
`2.99577e-5`; minimum residual rcond was approximately `9.77858e-4`.
Worst independent normalized horizontality was `4.48e-17`, and spectral excess
was zero. Signed-gap lower-bound checks were also retained, not omitted.

The greatest final-direction relative difference from the identical-input
production solve was `4.04e-14`; greatest multiplier Frobenius difference was
`7.43e-17`. These are measured agreements for this finite corpus. Raw inner
actions were not proved safe individually and are not advertised as such.
Safety of the returned update rests on authoritative SVD verification.

## Actual synchronized cost

Each candidate and direct solve was timed with CUDA synchronization around the
complete operation. Phase timers synchronize boundaries as well. Duplicate
offline audit work, data reconstruction, backward computation and input loading
are excluded from paired-solver time. Initial/cold work is retained in totals;
warm summaries use steps 5–49, matching the existing five-step exclusion.

| Warm steps 5–49 | Direct SVD | 2 / continue | 2 / restart | 3 / continue | 3 / restart |
|---|---:|---:|---:|---:|---:|
| Total paired-solver time, s | 44.921 | 45.296 | 46.645 | 44.996 | 44.992 |
| Six-pair median, s | 0.9834 | 0.9887 | 0.9897 | 0.9895 | 0.9897 |
| Six-pair p95, s | 1.0403 | 1.0475 | 1.1566 | 1.0287 | 1.0285 |

For three-iteration continue, full-replay phase totals were:

| Phase | Seconds |
|---|---:|
| QDWH decomposition + existing cheap structural screen | 23.267 |
| Remaining inner work, including provisional recovery/certificates and CG | 11.334 |
| Authoritative final SVD evaluation + recovery/certificate | 15.758 |
| SVD rescue | 0 |
| Remaining wrapper/validation/bookkeeping | about 0.249 |
| Complete measured hybrid | 50.608 |

The inner solve itself took 34.601 s. Adding the 300 authoritative verification
operations almost erased its advantage over the direct 51.082 s production
solver. The observed direct/QDWH decomposition advantage remains real, but
the hybrid cannot reuse a QDWH polar or spectrum to authorize the update.
It must pay for the final SVD and its recovery/certificate. In this prototype,
provisional recovery/certification is also performed at every inner iterate.
The resulting net saving was **0.474 s (0.93%)** overall; warm work was
**0.075 s (0.17%) slower**. This is not a materially useful H100 speedup.
Warm six-pair median was about 0.6% slower. The modest p95 change does not
establish improved training throughput or a general tail-latency benefit.

Two-iteration continue spent 33.555 s in inner work, 14.137 s in explicit
verification and 3.502 s in rescue. Restart spent 6.742 s in rescue and was
6.8% slower overall. Thus continue is the preferable recovery policy in the
tested budget-limited cases, but does not make the hybrid practically faster.
Naturally failed final verifications did not occur in the corpus; the explicit
rejection/recovery logic is exercised by focused tests, not claimed as a
measured real failure-recovery distribution.

## Near-guard stress and wasted-work limit

The existing deterministic `[48,16]` coupled solver corpus was reconstructed
at starting residual rconds `1.01e-4`, `1.02e-4`, `1.05e-4`, `1.1e-4`,
`1.2e-4`, `1.5e-4`, `2e-4` and `3e-4`, with a repeated-extremal-spectrum
case. The ordinary production rescue had a declared 12-Newton study limit,
CPU ADMM disabled. This deliberately bounded synthetic study is not training.

| Policy | SVD rescues / cases | QDWH evaluations | SVD evaluations | Final certified / cases |
|---|---:|---:|---:|---:|
| 2 / continue | 8 / 8 | 36 | 121 | 7 / 8 |
| 2 / restart | 8 / 8 | 36 | 141 | 7 / 8 |
| 3 / continue | 8 / 8 | 45 | 126 | 7 / 8 |
| 3 / restart | 8 / 8 | 45 | 141 | 7 / 8 |

Two-iteration policies triggered seven Newton-budget and one evaluation-cap
rescues. Three-iteration policies triggered five Newton-budget and three
evaluation-cap rescues. No structural rejection, failed standalone final
verification or CPU ADMM occurred here. **All near-guard problems needed SVD
rescue**, so this domain does not offer the hoped-for rescue-free hybrid path.
The hard seven-evaluation cap prevented arbitrarily many QDWH trials. On an
interrupted path the reported iteration count does not reconstruct every
partially attempted Newton action; QDWH evaluations and line-trial counts
remain the authoritative wasted-work measurements.

The `1.02e-4` case failed to certify within the declared production-rescue
budget; direct production SVD also failed within its same study limit. The
hybrid's `converged=False` remained explicit, and **no update was authorized**.
All seven accepted results per policy passed the independent SVD audit.
There was no unsafe final acceptance, including on that difficult case.

The existing cheap structural rcond screen is retained. An earlier
performance-only route to SVD for difficult near-guard problems could avoid
wasted inner work, but a current rcond estimate itself requires a decomposition;
no additional pre-decomposition gate or empirical cutoff was established by
this task. Such a gate would not rescue the absent real warm-time benefit.
The screen is not claimed to prove intermediate QDWH decisions throughout the
admitted domain.

## Controlled-trajectory gate and answers

The three-iteration policy met the real certification and low-rescue gates,
but **failed the material net-speed gate**. No hybrid-controlled short run,
locked candidate training run, candidate checkpoint-continuation test,
throughput, full-step timing or candidate loss trajectory is claimed. The
production checkpoint boundary was used only to reconstruct locked inputs.
Stationary research tests repeat deterministically, but they do not establish
real hybrid checkpoint continuation. No large checkpoint or new model artifact
was created by this replay.

1. QDWH proposed a final-SVD-accepted multiplier on 300/300 real problems with
   a three-Newton cap; with two, it did so on 269/300.
2. Rescue was needed on 0/300 (three) or 31/300 (two) real problems, and 8/8
   near-guard problems for every studied policy.
3. The best SVD-count reduction was 631/931 evaluations, 67.8%, while total
   decomposition count rose to 1,231.
4. All real usable returned directions were production-SVD-certified and
   independently audited; uncertified near-guard results authorized no update.
5. Best full-replay speedup was about 1.009x; warm speedup was about 0.998x.
   Neither is a material practical improvement.
6. Near the guard rescue frequency was 100%, although wasted QDWH work was
   capped and unsuccessful production recovery stayed explicit.
7. A controlled trajectory was not launched because the performance gate failed.
8. A production-v1 integration discussion is **not justified by this prototype**.

The final numerical/performance conclusion is distinct from the earlier
one-sided posterior study: this hybrid avoided its Armijo ambiguity storm and
needed no near-normal real rescue with three Newton iterations, but final
verification cost removed the kernel-level gain. Production-v0, its exact
coupled objective, prior-lambda warm start, fp64 SVD smooth backend, fp32
momentum, `gram_upper` recovery, all thresholds, checkpoint format and CPU
ADMM fallback remain unchanged. No LR sweep or long training was run.
