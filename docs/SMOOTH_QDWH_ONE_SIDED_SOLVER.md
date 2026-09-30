# One-sided QDWH-controlled smooth solver study

## Decision and scope

**Classification A: conservative one-sided control still stagnates at Armijo and is not practical.** The research solver preserved the final production certificate on the locked 50-step pair corpus, but conservative trial rejection exhausted the unchanged 24-trial line search in most pair solves. It then needed an explicit full-SVD smooth-solver rescue. The paired-solver counterfactual was substantially slower than the current direct-fp64-SVD solver. No production default or certificate threshold changed. The QDWH-controlled short training and locked 50-step training stages were therefore **not run**; their stated numerical *and performance* gate failed.

This study uses [the previous QDWH factor and decision bounds](SMOOTH_QDWH_DECISION_VALIDATION.md) and the research-only implementation in `experiments/smooth_qdwh_one_sided.py`. The accepted production implementation in `qnormuon/coupled_solver.py` continues to use fp64 thin SVD for the smooth residual. The full SVD is the independent oracle and explicit decomposition/recovery fallback. CPU ADMM is a different optimizer/reference fallback and was not invoked by this research replay.

## Formal one-sided decision contract

The mathematical problem, row whitening, dual objective, coupled horizontal LMO, Newton-CG damping, 24-trial line-search budget, and production acceptance thresholds are unchanged. The research path acts on QDWH information only when the established posterior bounds prove that action safe within their standard-fp64 operation model. These are conditional numerical bounds, not directed-rounding interval arithmetic or a backend-specific theorem about every CUDA implementation.

| Decision | QDWH action requiring proof | If proof is unavailable |
|---|---|---|
| Final certificate | Return a direction only if the conservative dual upper/primal lower bound proves normalized gap `<= 3e-5` and signed gap `>= -1e-10`, the measured feasible recovery meets normalized horizontality `<= 1e-10` and spectral excess `<= 1e-12`, and posterior residual-rcond lower bound exceeds `1e-4`. | Continue Newton. A distinct SVD candidate's possible earlier acceptance is irrelevant. On budget/stagnation, request explicit SVD recovery. |
| Armijo trial | Prove the whole trial objective interval lies below the permitted current-value/slope interval, prove the trial remains above the rcond guard, and retain the production resolved-decrease or gradient-improvement condition. | Backtrack to the next production trial. After 24 failures, request explicit full-SVD solver recovery. |
| CG curvature | `pᵀ Ĥp - ||p|| ε_HVP(p) > 0` for the candidate search vector. | Request SVD; using uncertain curvature could corrupt the CG direction. |
| CG stop | The true damped-system residual upper bound must meet the unchanged truncated-CG target, using the gradient lower bound. | Request SVD. Early CG termination without this test was not introduced. |
| Descent | `ĝᵀd + ε_g ||d|| < 0`. | Request SVD; do not take an uncertain direction. |
| Factor/rcond/nonfinite | Posterior structural checks pass and the conservative rcond lower bound is above guard. | Request SVD or fail explicitly. |

There is no mathematical need to reproduce the SVD solver's rejection branch. Conservative rejection can change Newton count, accepted multiplier, and finite-precision trajectory. Positive-curvature use, descent, Armijo acceptance, and final update acceptance are different: they authorize an action and therefore need one-sided proof. The one-sided policy does **not** permit an uncertified model update.

The research solver first tries direct fp64 QDWH plus H-EVD. It reuses the prior posterior for factor, nuclear-value, rcond, gradient, derivative, and HVP uncertainty. A failed factor/rcond posterior, uncertain CG curvature or stop, or uncertain descent requests full SVD at that evaluation with a reason code. An ambiguous *certificate rejection* does not request SVD. An ambiguous *Armijo trial* backtracks. Exhausted line search, Newton budget, or stagnation requests an explicit full-SVD smooth-solver retry. The retry uses the existing production Newton-CG with CPU ADMM disabled, so the two fallback concepts remain separate.

## Locked real-pair replay

The replay reconstructed the same six paired problems at each of the 50 locked tiny-Transformer steps, with the same seed, FineWeb token prefixes, batch hashes, checkpoint boundary, bf16 forward/backward, fp32 model/momentum, and direct-fp64 paired arithmetic. A stand-alone QDWH solver controlled each **counterfactual pair solve** from the same initial multiplier. The SVD production solve separately controlled the model trajectory so that every pair comparison started from an identical stored training problem. QDWH did not control subsequent model parameters in this experiment.

| Paired-solver quantity | One-sided QDWH counterfactual | Current SVD control |
|---|---:|---:|
| Starting pair problems | 300 | 300 |
| Final certified pairs | 300 | 300 |
| QDWH-only final certificates | 66 | — |
| Full-SVD smooth-solver rescues | 234 | — |
| CPU ADMM fallbacks | 0 | 0 |
| QDWH evaluations | 6,313 | — |
| Full-SVD evaluations inside candidate path | 469 | 931 |
| Line-search trials in candidate path | 6,013 | — |
| Total measured paired-solver time | 209.78 s | 50.84 s |
| Mean time for six pair solves per step | 4.20 s | 1.02 s |
| Median per-pair time | 0.853 s | 0.164 s |
| p95 per-pair time | 0.897 s | 0.218 s |

The candidate performed **4.13× more paired-solver time** (about `0.242×` the reference solver throughput), despite roughly halving full-SVD evaluation count. It performed 21.0 QDWH decompositions per pair solve on average. **234/300 solves (78%)** needed full-SVD recovery; the 469 SVD evaluations were only 6.9% of all candidate-path decompositions because QDWH trials proliferated. The 234 exhausted line searches caused 5,616 ambiguous QDWH trial rejections; there were 397 proven QDWH Armijo acceptances, 631 conservative QDWH certificate continuations, and just one QDWH CG-uncertainty decomposition fallback. All 66 solves certified directly by QDWH at Newton iteration 2. Before SVD rescue, 203 of the other solves had taken one QDWH Newton step and 31 had taken two. Their recorded pre-rescue counts are **not** final certification bins; no directly QDWH-certified solve finished in zero or one Newton iteration.

Measured candidate-path time comprised 157.72 s in QDWH decomposition, 10.89 s in factor posterior, 5.00 s in derivative posterior, 6.38 s in primal recovery/certificate, and 25.85 s in full-SVD solver rescue. The decomposition alone still averaged about 25 ms per evaluation, consistent with the earlier kernel advantage. This accounting excludes independent offline SVD oracle checks from candidate timing. It demonstrates why a kernel-level gain did not become a solver-level gain: conservative Armijo ambiguity multiplied the number of decompositions.

All 300 returned candidate directions were rechecked using an independent full-SVD residual objective and full-SVD projected-primal radial norm. No unsafe final acceptance was observed. The largest independently checked normalized gap was about `2.996e-5`; the minimum independently checked residual rcond was about `9.78e-4`. The greatest direction difference from the independent SVD-controlled solve on the same starting pair was about `1.10e-12` in Frobenius norm. Most returned directions came from SVD rescue; this close agreement is **not** evidence that unconstrained QDWH control yields an identical trajectory.

An offline, post-decision SVD audit independently checked the QDWH-controlled internal actions. It observed 2,067 HVP queries within their predicted error bounds, 1,436 used positive-curvature actions with positive oracle curvature, 630 certified-descent actions with negative oracle slope, and all 397 QDWH-involved accepted Armijo trials satisfying the oracle's production condition. One accepted trial with an SVD current evaluation and QDWH trial evaluation was covered by a deterministic targeted replay of its first three training steps; the remaining 396 were covered in the full audit. No unsafe accepted internal action was observed. The oracle calculations ran **after** candidate decisions and were excluded from candidate-solver timing. This is empirical coverage of the locked corpus, not a proof beyond the posterior's stated operation model.

## Near-guard and boundary stress

Eight constructed coupled problems started from residual spectra with prescribed smallest-to-largest ratios from `1.01e-4` through `3e-4`; one case had repeated extreme singular values. QDWH controlled its Newton and line-search trajectory, while full SVD solved each same starting problem independently. Seven of eight QDWH-controlled problems certified, and each of those passed independent full-SVD certificate verification. One `1.02e-4` case exhausted the 12-Newton study budget; the SVD-controlled counterpart also did not certify within that budget. It therefore produced **no update**, rather than an unsafe acceptance.

Five of the eight stress solves needed some SVD work: four encountered uncertain CG control and one encountered nine rcond-posterior ambiguities plus a full-SVD solver rescue. The constructed stress used 141 QDWH evaluations and 140 SVD evaluations, the latter dominated by the common nonconvergent case's retry. These data establish explicit conservative fallback behavior on this finite corpus, not whole-domain safety at every point just above `1e-4`.

Focused tests probe ambiguous/equality Armijo decisions, gap and signed-gap boundaries, the rcond boundary, feasibility rejection, zero/near-zero curvature and descent classification, explicit Newton-budget SVD rescue, explicit line-search exhaustion rescue, and a QDWH-certified initial point with a mocked SVD evaluator that would fail if used as a hidden branch oracle. The accepted thresholds were not weakened.

The complete H100 suite passed with **350 passed, 0 failed, 0 skipped** after the research-only additions. Raw logs and compact replay metrics are in `cluster/study_smooth_qdwh_one_sided_h100-28990.log`, `cluster/smooth_qdwh_one_sided-28990.json`, and the targeted audit replay `cluster/smooth_qdwh_one_sided-28992.json`. The final complete-suite log is `cluster/study_smooth_qdwh_decisions_h100-28993.log`.

## Why controlled training was not launched

The stage gate required complete pair replay certification, zero observed unsafe decisions, a low SVD fallback rate, and a materially better net solver time. The first two were met on the tested corpus. The latter two failed: 78% of pair solves needed SVD rescue and measured paired-solver time increased by 4.13×. Running a real QDWH-controlled trajectory or the locked 50-step candidate training run would not test a competitive policy. Thus no candidate-controlled checkpoint-continuation result or end-to-end training speedup is claimed.

**Production integration is not justified.** Full fp64 thin SVD remains the production smooth backend. The negative result is specific to this one-sided posterior and unchanged 24-trial globalization policy; it does not refute exact-arithmetic QDWH equivalence or the measured direct-QDWH kernel advantage. A future separate study would need to reduce objective/Armijo interval ambiguity without weakening acceptance, or provide an explicit early SVD recovery policy whose total cost beats direct SVD. Neither change is made here.
