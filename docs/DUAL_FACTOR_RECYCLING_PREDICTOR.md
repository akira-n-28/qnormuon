# Recycling accepted smooth-residual factors for dual warm starts

## Decision

**B — useful but not compelling for production integration.** A research-only,
two-CG-iteration predictor based on the previous accepted thin SVD reduced the
locked trajectory's warm smooth evaluations from 908 to 845 (63 paired residual
evaluations, or 126 individual matrix SVDs). It made 52 of 294 warm solves
certify after one Newton iteration, but added roughly 52 MB of fp64 cache for
six pairs and made four early warm solves require one extra evaluation. The
fair paired-solve time including factor capture and prediction fell by 2.70
seconds over the 50-step replay, about 5.4% of the direct baseline paired-solve
time. The inexpensive
mathematically sufficient validity gate rejected every real warm state. This
does not justify a production checkpoint/state change at present. The
production solver, its full-fp64-SVD backend, and its previous-original-lambda
warm start remain unchanged.

This is a counterfactual replay on a **fixed baseline training trajectory**,
not a predictor-controlled training run. Each candidate's certified solve is
measured against the same stored current problem; only the baseline result
advances the model and optimizer state. Therefore the timing is an H100 solver
estimate, not a measured end-to-end training speedup under altered updates.

## Exact predictor construction

At a smoothly certified solve, the research wrapper retains the factors from
the **same smooth evaluation and multiplier** that produced the returned
certified primal. For each side, the production centering/whitening identity is

\[
B_{\mathrm{old,original}}=A_{\mathrm{old}}-L_{\mathrm{old}}^*\lambda_{\mathrm{old}}
=s_{\mathrm{old}}B_{\mathrm{old,internal}},\qquad s_{\mathrm{old}}>0.
\]

Thus the original-residual factors are \(U_{\rm old}\),
\(s_{\rm old}\sigma_{\rm internal}\), and \(V_{\rm old}^{T}\). The wrapper
checks reconstruction against the original residual; the largest relative
error in the locked replay was 2.02e-13. Rejected line-search evaluations
cannot populate the cache. The factors and original-coordinate lambda are
research state only, not production optimizer state.

For the new canonical \(U,D,A\), form \(B_{\rm new}(\lambda_{\rm old})\)
without a decomposition and set \(\Delta B=B_{\rm new}(\lambda_{\rm old})-
B_{\rm old}\). With the old validated polar derivative,

\[
P_{\rm pred}=P_{\rm old}+D\operatorname{polar}_{B_{\rm old}}[\Delta B],
\qquad g_{\rm pred}=-L_{\rm new}(P_{\rm pred}),
\]

and the matrix-free positive Hessian action is

\[
H_{\rm pred}(v)=L_{\rm new}D\operatorname{polar}_{B_{\rm old}}
  [L_{\rm new}^{*}(v)].
\]

The sign follows from \(g(\lambda)=-L(P(A-L^*\lambda))\): differentiating
with respect to \(\lambda\) contributes two minus signs. If
\(L_{\rm new}=L_{\rm old}+\delta L\), then

\[
g_{\rm pred}=-L_{\rm old}(P_{\rm old})-\delta L(P_{\rm old})
-L_{\rm old}D\operatorname{polar}_{B_{\rm old}}[\Delta B]
-\delta L D\operatorname{polar}_{B_{\rm old}}[\Delta B].
\]

The first term vanishes at an exact old stationary point; in this experiment
the old multiplier is only certified to the production gap tolerance, so the
term is retained by the actual predictor. The last term is second order in
the problem change. This is the usual first-order stationarity expansion, but
uses current \(L\) directly. Row-whitened, damped matrix-free CG approximately
solves \(H_{\rm pred}\delta\lambda=-g_{\rm pred}\), then proposes
\(\lambda_{\rm old}+\delta\lambda\). The candidate is **only an initial
multiplier**. It never supplies a certificate or substitutes old factors in
the current full-SVD Newton solve. No current SVD/SVDVALS is called by the
predictor; a focused test blocks both functions to enforce this.
The research CG uses the current row-whitening coordinate, a fixed damping
\(10^{-6}/\sigma_{\min}(B_{\rm old})\), and a loose residual target capped at
one tenth of the initial residual. These are initialization choices, not
changes to the production Newton system or mathematical objective.

The factor derivative depends on invariant polar quantities, not on matching
individual singular vectors. A test rotates an exactly repeated singular
subspace and obtains the same derivative and prediction within fp64 error.
Canonical residuals and the predicted original-coordinate multiplier are
compatible with positive regular gauge transformations in the focused test.

## State and checkpoint cost

For one [1024,384] pair, both sides together require 8,656,896 bytes of fp64
thin-SVD factors (\(U,\sigma,V^T\)); the fp64 multiplier adds 8,192 bytes.
Six pairs require 51,941,376 factor bytes, or 51,990,528 bytes including the
six multipliers. A six-pair `torch.save` research payload measured 51,997,851
bytes. A research save/reload at the locked checkpoint boundary preserved every
factor and multiplier bitwise and reproduced the next prediction. The
production checkpoint format was not modified.

An fp32 predictor-only cache halves factor storage to 4,328,448 bytes per
pair, or 25,970,688 bytes for six. Original lambda remains fp64; with six
lambda vectors, total tensor state would be 26,019,840 bytes. Compressed
factors are promoted to fp64 **before** reconstruction, polar, derivative, and
predictor arithmetic. The final current solve always remains fp64. Its
empirical effect is reported below.

## Locked H100 replay

The replay used the exact 50-step 11.46M-parameter tiny Transformer smoke
configuration, seed, FineWeb token prefixes, minibatch hashes, bf16 autocast,
fp32 model storage and canonical EMA, six [1024,384] pairs, and direct-fp64
production solver. The baseline replay asserted exact stored batch hashes and
losses at every step. The first six cold solves used 23 paired smooth
evaluations; the remaining 294 warm solves are the comparison population.
Each paired evaluation performs two fp64 thin SVDs. All candidate solves used
unchanged 3e-5 gap, 1e-4 rcond guard, full-SVD smooth backend, and the
production fallback configuration. All reported warm candidates certified
without CPU ADMM fallback. Every paired evaluation count reconciled with two
actual thin-SVD matrix decompositions. For the budget-2 candidate, the largest
final normalized gap was 2.99914e-5 (below 3e-5), and the smallest residual
rcond was 9.78e-4 (above 1e-4).

| Initialization | Warm 0/1/2/3+ Newton bins | Paired smooth evaluations | Change from previous lambda | Predictor time | Certified solver time |
|---|---:|---:|---:|---:|---:|
| Previous lambda, factor capture | 0 / 0 / 268 / 26 | 908 | baseline | 0 | 50.305 s |
| Previous lambda, direct solve | 0 / 0 / 268 / 26 | 908 | baseline | 0 | 50.006 s |
| Recycled, CG budget 1 | 0 / 28 / 250 / 16 | 870 | -38 | 0.502 s | 47.943 s |
| Recycled, CG budget 2, factor capture | 0 / 52 / 227 / 15 | 845 | -63 | 0.610 s | 46.691 s |
| Recycled, CG budget 4 | 0 / 52 / 227 / 15 | 845 | -63 | 0.606 s | 46.568 s |
| Recycled, CG budget 8 | 0 / 52 / 227 / 15 | 845 | -63 | 0.605 s | 46.568 s |
| fp32 factor cache, CG budget 2 | 0 / 52 / 227 / 15 | 845 | -63 | 0.652 s | 46.574 s |
| Frobenius-gated budget 2 | 0 / 0 / 268 / 26 | 908 | 0 | 0.089 s | 50.305 s |

The strict-configuration replay timed the **direct** previous-lambda solver
separately from the research factor-capturing wrapper, and timed budget-2
prediction with its required accepted-factor capture. Direct baseline warm
solving took 50.006 s; budget-2 solving **including cache capture** took
46.691 s, plus 0.610 s for prediction. This is 47.301 s total, a net 2.704 s
saving and 1.057x paired-solver speedup. Other candidate rows use lightweight
counterfactual solve timing without cache capture and should not be read as
fair integration-cost comparisons.

Budget 2 was the smallest useful setting; CG stopped within two iterations in
all real cases. Its synchronized predictor median was 2.21 ms per pair, far
below the approximately 43 ms current paired smooth-SVD evaluation observed
in earlier H100 profiling. Complete prediction includes reconstructing old
\(B\), constructing new \(B\), forming \(\Delta B\), applying the old polar
derivative, computing the current predicted gradient, CG/HVPs, and forming
\(\hat\lambda\). It improved the evaluation count on 67 states, left 223
unchanged, and cost one extra evaluation on four states, all at steps 3–4 of
the warmup. The net is 63 evaluations saved. Applied as a rough estimate to
the prior ~1.02 s full QSO step, the fair replay saving is about 0.054 s per
step; that is **not** an end-to-end measurement.

The baseline median initial normalized gap was 2.62e-3, versus 8.92e-4 after
prediction; the median initial gradient norm was 1.63e-2 versus 4.61e-3.
The predicted initial gap had a worse maximum (5.01e-2 versus baseline
2.15e-2), showing that a lower median does not remove abrupt-change tails.
Against the actual first current Newton correction, the budget-2 predicted
shift had median cosine 0.963 and median relative error 0.288. Against final
accepted multiplier drift the corresponding values were 0.962 and 0.290.
The first-order predicted gradient versus the true gradient at the old lambda
had median cosine 0.963 and median relative error 0.275, with a poor tail
(p95 relative error 1.12). The first-order model is useful but imperfect.

Optional fp32 cache compression preserved **every** warm certification bin
and smooth-evaluation count of the fp64 cache in the locked replay; it also
had the same four one-evaluation regressions. The predicted multiplier shift
relative to fp64-cache prediction had median relative difference 5.26e-7 and
maximum 1.31e-6. Its measured predictor time was 0.652 s across all warm
states versus 0.610 s for fp64-cache prediction, reflecting conversion work.
This shows compression is viable on these fixtures, not a general guarantee
that fp32 factors are adequate near the smooth rcond guard. The replay derived
each fp32 copy from the baseline fp64 cache before starting the predictor
timer; therefore the displayed fp32 time does **not** include the cost of
compressing each new accepted factor set, and no compressed-cache checkpoint
continuation was measured.

## Validity gate and abrupt changes

The sufficient no-rank-crossing condition
\(\|\Delta B\|_2<\sigma_{\min}(B_{\rm old})\) was tested through its cheap
Frobenius upper bound \(\|\Delta B\|_F<\sigma_{\min}(B_{\rm old})\).
It rejected **294/294** ordinary warm states. The ratio
\(\|\Delta B\|_F/\sigma_{\min}\) ranged from 17.25 to 910.62, median
57.93. It is mathematically defensible but unusable as a selector here.
The implementation uses the Frobenius norm of both stacked residual sides
against the smaller of their trailing singular values; this is even more
conservative than checking the sides separately.
Rejecting a prediction always returns the production previous lambda; this
gate is about predictor work, never final solver correctness.

A deterministic real-pair stress replay included ordinary change, 10x
objective scale, sign flip, column rotation, and large additive objective
perturbation. Baseline and ungated budget-2 candidate both certified in all
five; prediction saved one evaluation for the 10x and sign-flip cases and did
not inflate evaluation count in these cases. The Frobenius gate rejected all
five, including the ordinary case. The locked run also covers the first warm
step after momentum became populated, the five-step LR warmup transition, and
the research cache save/reload boundary. Positive gauge compatibility was
checked on balanced canonical inputs from gauge-related raw pairs. This
stress set is finite and does not prove that all abrupt changes are harmless;
the four real warmup regressions are evidence against such a claim.

## Validation and limitations

The research module and focused tests are `experiments/dual_factor_recycling.py`
and `tests/test_dual_factor_recycling.py`. The full suite ran on an H100 through
SLURM before each replay. The test checks factor scaling and accepted-evaluation
identity, zero current decompositions, derivative and Hessian finite
differences, repeated-subspace invariance, positive gauge behavior, the gate,
serialization, and absence of current-optimum/certificate access. The locked
replay artifacts are `cluster/dual_factor_recycling-28975.json` and its SLURM
log for the strict-configuration/fair-timing run, and
`cluster/dual_factor_recycling-28974.json` for the independent compression
replay. The final H100 job passed the complete 308-test suite; its network
guard log was empty. No production optimizer file or default was changed.

This predictor is a local linearization. Its no-SVD validity gate is too
conservative for the stochastic objective changes observed here. The observed
6.9% reduction in paired smooth evaluations is real on the locked states, but
the cache is O(mn+n²), requires a larger checkpoint, and has four observed
evaluation regressions. There is no evidence yet that predictor-controlled
training gains the same amount or that a cheap robust gate can prevent
regressions without rejecting normal work. A production integration task is
therefore **not recommended** on this evidence. The efficient next work, if
performance remains the goal, should reconsider where certified smooth
evaluations can be avoided without carrying large factor history; it should
not silently alter the coupled LMO or acceptance contract.
