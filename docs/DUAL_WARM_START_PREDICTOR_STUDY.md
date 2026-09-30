# Temporal dual-multiplier warm-start study

## Decision

**A — no useful reduction.** Keep the production previous-original-coordinate-λ
warm start. The best fixed history-only candidate in this study saved **3 of
908 warm smooth residual SVD evaluations (0.33%)** over 294 pair solves, with
two individual regressions. No candidate made a real warm solve certify at
zero or one Newton iteration. This is too little benefit to add predictor state
or a new production policy. The production full-fp64-SVD smooth backend,
coupled LMO, `3e-5` normalized-gap target, `1e-4` rcond guard, primal
`gram_upper` feasibility backend, and fallback semantics are unchanged. There
was no LR sweep, download, production edit, or predictor-controlled training
run. The conditional short-run gate was not met.

## Locked replay and measurement

The research-only predictor functions are in
[`experiments/dual_warm_start_predictor.py`](../experiments/dual_warm_start_predictor.py).
The [H100 replay driver](../cluster/study_dual_warm_start_predictor_h100.py)
uses the locked `configs/tiny_transformer/smoke.json` and the original
`smoke-28936-qso-seed2026` metrics. It reconstructs all 50 steps from seed
2026 on one H100 NVL. All **50 minibatch hashes and losses matched exactly**.
For every pair, only the unchanged previous-λ solve advanced the optimizer;
candidate solves were read-only with CPU ADMM disabled. All 300 controlling
solves certified, with no fallback. Candidate solves also certified in every
real state. The common inputs were the actual production fp64 canonical
`U,D,A`, and each candidate used the unchanged Newton-CG, full smooth SVD,
Gram-upper primal norm, certificate, and guards.

The complete per-pair/per-step [replay record](../cluster/dual_warm_start_predictor-28970.json)
contains the initial normalized gap and dual-gradient norm, absolute and
relative distance to the baseline accepted λ, error in the current whitened
coordinate, first Newton-direction norm in original λ coordinates, Newton
count, smooth SVD evaluation count, line-search count, and synchronized solve
time. It also stores relative changes in each canonical weight, objective,
centering β, objective magnitude, and whitening vector. The baseline's
**931** full-trajectory smooth evaluations reconcile with the previous smooth
shadow study: **908 warm** plus **23 cold**. Its 300 Newton bins are 269 at
two iterations and 31 at three or more, again matching the locked run.

The complete H100 test suite ran first in the same SLURM job: **300 passed,
0 failed, 0 skipped**. The job completed `0:0` in 10m25s; the replay itself
took 595.6 s because it performed ten counterfactual solves per actual pair.
The [raw job log](../cluster/study_dual_warm_start_predictor_h100-28970.log)
and compact JSON remain under `cluster/`. The offline network guard recorded
no Python outbound attempt. A separate 15-second
[checkpoint-step follow-up](../cluster/dual_warm_start_followup-28971.json)
replayed step 25 from the locked checkpoint and measured predictor cost; it
also matched the saved batch hash and loss exactly.

## Coordinates and fixed predictors

Production persists λ in its **original multiplier convention**. At each new
problem, the solver computes `β=L(A)/w`, `s=||A-L*(β)||_F`, and
`coord=w^(-1/2)`, then maps an initial original λ to its *current* internal
coordinate `z=(λ-β)/(s*coord)`. `center` means λ=β, or `z=0`; it is not a
literal all-zero original multiplier. All prediction policies leave the
objective untouched and are fixed before seeing the current accepted λ.

The tested state-light policies were previous λ; center; original-coordinate
`λ_prev+α(λ_prev-λ_older)` for the predefined `α={.25,.5,.75,1}`; and
current-center/current-magnitude transport of
`q=(λ-β)/s` with `α={0,.5,1}`. One `α=.5` predictor extrapolated the old
internal `z=(λ-β)/(s*coord)` and mapped it through **current** β, s, and
coord. This last policy is a research comparison, not persistence of an old
whitened coordinate as if it were invariant. At the first populated momentum
step, no history exists and all policies use the current center. No alpha was
fitted by layer or training step; no λ-star oracle is available to `predict`.

Canonical `U,D,A` and original-coordinate λ are invariant under a positive
raw gauge reset in exact arithmetic; β, s, and coord are therefore invariant
too. A focused test checks this construction through actual raw up/down
canonicalization and covector transforms. As usual, finite-precision gauge
equality is measured within rounding tolerance, not claimed bitwise.

## Results: decomposition count matters more than λ distance

All counts below exclude the six cold step-0 solves, so each row covers the
same **294** real warm pair problems. Each smooth evaluation is one batched
full-SVD call on two residual matrices. Every candidate certified and used
zero CPU ADMM fallback.

| Fixed initialization | Newton 0 / 1 / 2 / 3+ | Smooth evaluations | Change vs previous λ | Line-search evaluations |
| --- | ---: | ---: | ---: | ---: |
| Previous λ, production | 0 / 0 / 268 / 26 | **908** | — | 614 |
| Current center β | 0 / 0 / 0 / 294 | 1176 | +268 | 882 |
| Raw extrapolation α=.25 | 0 / 0 / 265 / 29 | 911 | +3 | 617 |
| Raw extrapolation α=.5 | 0 / 0 / 264 / 30 | 912 | +4 | 618 |
| Raw extrapolation α=.75 | 0 / 0 / 262 / 32 | 914 | +6 | 620 |
| Raw extrapolation α=1 | 0 / 0 / 256 / 38 | 920 | +12 | 626 |
| Center/magnitude transfer α=0 | 0 / 0 / 271 / 23 | **905** | **−3** | 611 |
| Center/magnitude transfer α=.5 | 0 / 0 / 264 / 30 | 912 | +4 | 618 |
| Center/magnitude transfer α=1 | 0 / 0 / 259 / 35 | 917 | +9 | 623 |
| Whitened transfer α=.5 | 0 / 0 / 264 / 30 | 912 | +4 | 618 |

The α=0 center/magnitude transfer is
`β_t+(s_t/s_{t-1})(λ_{t-1}-β_{t-1})`. It saved one SVD at five states and
cost one extra at two; its mean λ error was `6.84e-5` versus `6.87e-5`
for previous λ. Full raw extrapolation saved one evaluation five times but
cost one seventeen times. Across all real states, the largest per-solve
inflation was one evaluation, but the average was worse for every extrapolator.
The available pre-SVD scale/change signals did not give a defensible selector:
even the seven α=0 changed-count states have overlapping objective changes
and mixed win/loss signs. Selecting the winning predictor from λ-star or its
actual certificate would pay for an extra decomposition and is disallowed.

The previous-λ initial normalized gap had median **2.62e-3** and maximum
**2.15e-2**, well above `3e-5`; median initial dual-gradient norm was
`5.46e-2`. Median initial-to-accepted original λ distance was `2.54e-5`,
while the median first Newton-direction norm was `2.53e-5`: Newton removes
most temporal drift, but the remaining first-step certificate still usually
fails. At the independently loaded step-25 checkpoint, initial gaps across
six pairs were `1.93e-3`–`3.97e-3`; after the first Newton step they were
`9.34e-5`–`4.07e-4`, **all above** the target. After the second they were
`8.05e-7`–`1.28e-5`, all certified. This directly explains why merely
predicting λ more closely with two-point history rarely removes a full step.

The median relative change per warm transition was about `0.169%` for each
canonical weight matrix, versus **11.9%** for the canonical EMA objective.
Median β change relative to its previous norm was `4.03%`, median objective
magnitude ratio was `0.958`, and median row-whitening-vector change was only
`0.011%`. The fast-changing stochastic objective, including its direction,
dominates the very small weight/whitening drift. Centering and magnitude
transport account for some predictable scale change, but not enough of the
new optimal multiplier to pass the certificate one iteration earlier.

## Abrupt changes, state, and checkpoint implications

The replay includes the first populated EMA step, the original five-step
warmup and its schedule transition, all ordinary adjacent transitions, and
the locked checkpoint boundary. A separate read-only step-25 pair stress
multiplied the objective by ten, flipped its sign, rotated its columns, or
added a large deterministic perturbation. Every tested initialization still
certified with the unchanged solver and no reference fallback. On this small
stress corpus, candidates took at most one more SVD than previous λ; a few
scaled/rotated cases took one fewer. This is evidence of local globalization
robustness, not a guarantee for all abrupt stochastic changes. Positive gauge
compatibility was tested in the focused unit test; no gauge-reset training
trajectory was run for a predictor that is not being proposed for production.

Raw extrapolation needs one additional `m`-vector of fp64 λ per pair beyond
production state: **8 KiB** for `m=1024`. α=0 center/magnitude transfer needs
the previous β vector and scalar s in addition to persisted λ, about **8 KiB**
per pair. Extrapolating centered or whitened history needs older λ and scale
metadata; the generic two-state research serialization stores six fp64
`m`-vectors plus two scalars, **48 KiB per pair** at `m=1024` (288 KiB for
six pairs), without SVD factors or residual matrices. The focused checkpoint
test and H100 follow-up roundtripped this predictor payload and reproduced
every fixed prediction exactly. Production checkpoints were not changed.
An actual predictor-controlled resumed training trajectory was intentionally
not run, because no candidate met the material-savings gate; exact
predictor-state roundtrip is a narrower claim than full trajectory replay.

## Cost and first-order alternative

Warmed H100 medians for a length-1024 prediction were `0.0157 ms` to clone
previous λ, `0.0220 ms` for raw extrapolation, `0.0531 ms` for the best
center/magnitude transfer, and `0.0664 ms` for whitened extrapolation.
The best transfer's incremental prediction cost is about `0.0374 ms` per
pair, or **0.011 s over 294 warm calls**. The synchronized counterfactual
solve totals were `50.065 s` for previous λ and `49.917 s` for the best
transfer, a noisy gross difference of **0.148 s over 50 steps**. Subtracting
its roughly `0.011 s` extra predictor cost suggests at most about **0.137 s
net over 50 steps**, or **2.7 ms per six-pair step**. The primary stable
measure is the 0.33% SVD-count reduction. Neither this order-sensitive
microtiming nor the count supports a meaningful end-to-end speedup against
the approximately 1.02 s warm QSO step.

For completeness, first-order stationarity gives a possible implicit
predictor. Write `g=-L_θ(P(B))`, `B=A-L_θ*(λ)`, and let `H_λ=L_θ DP_B L_θ*`.
At an old solved state, a change of objective/weights would give

```
H_old δλ ≈ δL(P_old)
             + L_old DP_Bold[δA - δL*(λ_old)].
```

Here `δL` and `δL*` contain the actual changes in `U,D`; changes to β,
magnitude, and row whitening follow by re-expressing this original-coordinate
δλ through the current production normalization. The formula assumes a
usable inverse/action of `H_old`, which is not guaranteed merely by a
full-rank residual. Applying `DP_Bold` needs the previous polar/SVD data and
matrix-size perturbations. Persisting thin factors for both `[1024,384]`
residual sides requires about **8.7 MB per pair** even before auxiliary
vectors, versus O(m) history state; recomputing them costs another full
smooth SVD. Evaluating the exact new gradient at old λ also requires the
first new SVD that the current Newton solver already performs. An implicit
predictor therefore cannot cheaply create a pre-SVD zero-step start under
the stated compact-state constraint. A one- or two-history secant surrogate
is already represented by the tested fixed extrapolations; more elaborate
secant control is not justified by their negative result. No hidden expensive
O(mn²) predictor was implemented.

## Answers and limits

1. **Why two Newton steps?** Previous λ starts with a gap roughly two orders
   above target; the first Newton direction corrects most λ drift but leaves
   a certificate gap above `3e-5`, as the independent checkpoint replay shows.
2. **Best cheap candidate?** Current β/magnitude transport with α=0, by only
   three warm evaluations out of 908; previous λ remains the better practical
   policy because it has no new state and its performance is essentially equal.
3. **SVD saving?** Three fewer batched full-SVD evaluations (six fewer matrix
   decompositions) across 294 warm pair solves; zero 0- or 1-iteration solves.
4. **Worse cases?** Center/magnitude transport increased work on two real
   states; full raw extrapolation increased work on seventeen. No tested real
   transition inflated work by more than one evaluation, but there is no
   universal overshoot bound.
5. **State?** Extra O(m) multiplier/scale vectors as detailed above; no full
   factors are needed for history-only policies.
6. **Gauge?** Predictors expressed in canonical/original coordinates are
   positive-gauge compatible under the regular-domain contract.
7. **Checkpoint?** Small predictor state roundtrips exactly, and the loaded
   original step-25 checkpoint reproduces its saved loss and batch. No
   predictor-controlled checkpoint training was needed or claimed.
8. **Implicit predictor?** Its derivative/Hessian information is not cheap
   under O(m) state; it would duplicate or precede the current full SVD.
9. **Net H100 benefit?** At most about 2.7 ms per six-pair step for the best
   candidate in this fixed replay, with measurement noise and extra state.
10. **Production task?** No. Classification **A**. A future task would need a
    qualitatively better certificate-aware predictor or a different way to
    remove smooth evaluations without paying an equivalent spectral cost.

This is one seed, one 50-step configuration, and a small constructed stress
set. Candidate solves used the same actual pair inputs but did not steer model
updates; their certified directions can differ at finite tolerance. The
study does not establish behavior across LR sweeps, longer training, or near
rank loss. It preserves the distinction between predictor initialization and
the unchanged certified mathematical solve.
