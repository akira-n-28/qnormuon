# Decision-relevant posterior for direct fp64 QDWH

## Decision

**B — the conditional posterior was defensible and passed the tested oracle
comparisons, but validation and fallback cost removed the practical kernel
advantage.** Production still uses its full fp64 thin SVD for the smooth
residual. The coupled LMO, `3e-5` normalized-gap target, `1e-4` rcond guard,
projected-primal `gram_upper` rule, and CPU ADMM reference fallback were not
changed. The research implementation is
[`smooth_qdwh_decisions.py`](../experiments/smooth_qdwh_decisions.py). Its
[`H100 replay`](../cluster/study_smooth_qdwh_decisions_h100.py) keeps full SVD
in control of every optimizer/model decision. A QDWH rejection means a
**decomposition fallback** to full SVD, never an optimizer fallback to ADMM.

This study did **not** reach the QDWH-controlled trajectory or locked
candidate-run gates. It therefore makes no end-to-end training speedup or
checkpoint-determinism claim for a QDWH-controlled run.

## Exact matrix argument and operation model

For the computed `Qt,Ht`, the posterior measures
`d=||QtᵀQt-I||_F`, `f=||B-Qt Ht||_F`, and the eigensystem residual of
the symmetric SPD `Ht`. If `d<1`, define `Q0=polar(Qt)` *only in the proof*.
The singular values of `Qt` lie in `[sqrt(1-d),sqrt(1+d)]`, so

```
||Q0-Qt||_F <= ||Qt||_F d / [sqrt(1-d)(1+sqrt(1-d))] = qcorr.
```

With `B0=Q0 Ht`, `Q0,Ht` are an exact polar pair. The measured product defect,
`qcorr ||Ht||_2`, and a standard fp64 operation-model allowance give
`e >= ||B-B0||_F`. Weyl then gives
`sigma_min(B) >= mu0-e` and `sigma_max(B) <= M0+e`, where `mu0,M0` enclose
the extrema of `Ht`. The path from `B0` to `B` stays full rank if `e<mu0`.
Integrating the full-rank polar derivative bound along it yields

```
||polar(B)-Qt||_F <= qcorr + 2e/(mu0-e) = eta_Q.
```

This is deliberately conservative. It depends on the *direct* polar
factorization defect, not on `BᵀB` or a Gram-square-root condition number.
Since the exact `H=polar(B)ᵀB`, a separate direct bound is

```
||H-Ht||_F <= eta_Q ||B||_2 + ||QtᵀB-Ht||_F = eta_H.
```

For the computed `Ht` eigenpairs `V,st`, the exact identity
`Ht-V diag(st)Vᵀ = Ht(I-VVᵀ)+(HtV-V diag(st))Vᵀ` supplies an eigensystem
residual bound. Its measured orthogonality defect encloses `mu0` and `M0`
without matching individual eigenvectors, including at repeated eigenvalues.
Weyl yields `rcond(B)` inside
`[(mu0-e)/(M0+e), (mu0_upper+e)/(M0_lower-e)]` when denominators are
positive. QDWH is never classified above the production guard when this
interval overlaps `1e-4`. Nuclear value follows from
`| ||B||_* - tr(Ht) | <= sqrt(n)e`, plus a scalar-reduction allowance.

These are exact real-matrix inequalities **conditional on valid upper bounds
for the measured residuals**. The implementation inflates GPU GEMM, EVD, and
reduction measurements using the standard fp64 `gamma_k=ku/(1-ku)` model,
`u=2^-53`; it is not directed-rounding interval arithmetic or a proof of
undocumented cuSOLVER kernels. Full-SVD oracle comparisons remain essential
evidence. A nonfinite defect, non-SPD `Ht`, failed QDWH convergence, or
ambiguous rcond asks for full SVD with an explicit reason.

## Derivative and solver decisions

For `B=QH` and `Y=Dpolar_B[E]`, let `Omega=skew(QᵀY)`. The three exact
equations are

```
QᵀY+YᵀQ=0,
H Omega+Omega H=QᵀE-EᵀQ,
(I-QQᵀ)(YH-E)=0.
```

The candidate `Yc` from QDWH's `H` eigensystem is checked against these
equations at `Qt,Ht`, without an `m×m` projector. Residuals are inflated by
`eta_Q`, `eta_H`, and operation-model terms. For `mu<=sigma_min(H)` the
skew Sylvester inverse has norm at most `1/(2mu)` and the normal inverse at
most `1/mu`; their tangent/normal decomposition gives the implemented
conditional upper bound `epsilon_Y` for `||Yc-Y||_F`. Repeated positive
singular values need no special basis selection. The bound can become loose
near the guard, especially for derivative-near-null directions.

The whitened constraint operator satisfies `||L_eff* v||_F²=||v||²` in exact
arithmetic. The posterior measures the rounded maximum row norm and inflates
it under the same fp64 model. It therefore bounds HVP error by that operator
upper bound times the two-side `epsilon_Y` norm. Each CG curvature is accepted
only when `vᵀ(Hc+delta I)v ± ||v|| epsilon_HVP(v)` has an unambiguous sign.
The candidate CG solution receives a direct true damped-residual upper bound:
computed residual plus gradient and one final HVP uncertainty. That upper
bound must meet the *unchanged* truncated-CG stopping target. Its slope is
accepted as descent only when the gradient-error interval lies strictly below
zero. Ambiguity requests full SVD.

Current/trial nuclear-value intervals and a gradient-slope interval enclose
the production Armijo inequality with its existing rounding allowance. A
trial is called accept or reject only if its whole interval lies on one side,
including the rcond and resolved-decrease/gradient-improvement conditions.
At the primal certificate, the QDWH polar is horizontally projected and
radially scaled by the existing conservative `gram_upper` routine. This gives
an actual feasible candidate under the current fp64 certificate contract;
the QDWH nuclear upper bound and conservative primal dot-product bound can
certify acceptance at the unchanged five production criteria. A failed
QDWH-candidate certificate **cannot** certify rejection of the different
SVD-derived candidate. In particular, `gram_upper` is a conservative radial
estimate, not an exact 1-Lipschitz spectral norm map; bounding the two polar
factors does not by itself bound the difference of their computed radial
scales. The replay therefore requests full SVD for every uncertain rejected
certificate. This restriction is central to the performance result. The
replay did not change production certification or call CPU ADMM for a QDWH
uncertainty.

The residual-to-direction estimate `||x-x_true|| <= residual_upper/delta`
is valid because the smooth Hessian is PSD and the Newton damping is positive,
but it is very loose in this problem. Even where CG stopping/descent decisions
were certified, that estimate does **not** establish a close future
QDWH-controlled trajectory. The Armijo replay evaluates QDWH at trial
multipliers generated by the **full-SVD-controlled** Newton direction. It
certifies those fixed-trial decisions, not the trials a distinct
QDWH-controlled direction would generate.

## H100 oracle validation

One H100 NVL SLURM job ran the unfiltered suite: **338 passed, 0 failed, 0
skipped**. The locked 50-step counterfactual then replayed the same seed,
checkpoint boundary, batches, and losses as the existing production trajectory.
All 300 pair solves remained production-SVD-controlled and certified; there
was no CPU ADMM fallback. The compact observations are in
[`smooth_qdwh_decisions-28985.json`](../cluster/smooth_qdwh_decisions-28985.json)
and the raw log in
[`study_smooth_qdwh_decisions_h100-28985.log`](../cluster/study_smooth_qdwh_decisions_h100-28985.log).

| Fixed-trajectory oracle comparison | Result |
| --- | ---: |
| Smooth residual evaluations; factor posteriors passed | 931; 931 |
| Nuclear-value / gradient / polar-bound violations | 0 / 0 / 0 |
| HVP queries; bound violations | 1,436; 0 |
| Curvature signs certified; wrong certified signs | 1,436; 0 |
| CG/descent decisions certified | 630 / 631 |
| Armijo trials certified; wrong certified branches | 397 / 631; 0 |
| Certificate accepts certified; rejected certificates requiring SVD | 300; 631 |
| Wrong certified certificate decisions | 0 |
| Evaluations with no recorded need for SVD | 66 / 931 (7.1%) |
| SVD decomposition requests | 865 / 931 (92.9%) |

The reason histogram is 631 `qdwh_certificate_ambiguous`, 234
`qdwh_armijo_ambiguous`, and one `qdwh_cg_uncertain`; one evaluation has
overlapping reasons. The minimum real conservative rcond lower bound was
about `9.78e-4`, materially above the `1e-4` guard. The maximum observed
candidate/SVD CG-direction relative difference was about `1.17e-10`, but
the generic damped-system direction bound reached about `174` in absolute
norm, so fixed-trajectory agreement is not a control theorem.

The near-guard solver corpus used prescribed starting residual rconds
`1.01e-4`, `1.02e-4`, `1.05e-4`, `1.1e-4`, `1.2e-4`, `1.5e-4`, `2e-4`, and
`3e-4`, with a repeated-extremal-spectrum case. Its 54 evaluations had
zero value, gradient, polar, HVP, curvature, Armijo, or certificate oracle
violations. All 382 tested curvatures and 46 Armijo branches were decided
consistently; 21 of 24 CG decisions certified. All 32 rejected certificates
remained ambiguous and requested full SVD, giving **32/54 = 59.3%**
near-guard decomposition requests. Three CG decisions were also uncertain,
overlapping those certificate requests. The smallest rcond lower bound was approximately
`1.0099999905e-4`. These short synthetic solves deliberately stop after
three Newton iterations and are not evidence of eventual certification or
optimizer-quality training. Synthetic exact-boundary unit tests cover
marginal curvature/descent, CG stopping, Armijo equality, and certificate
gap decisions; they are not a complete solver-level construction of every
marginal branch. Zero observed unsafe decisions is a tested fact, not a
universal CUDA-backend proof.

## Cost and stage gate

In the same synchronized H100 replay, the 931 production-controlled full-SVD
smooth evaluations took **40.57 s** (43.6 ms/evaluation). Direct QDWH with
its original structural posterior took **23.50 s** (25.2 ms/evaluation).
The added factor/rcond posterior took **1.63 s** and derivative/HVP posterior
work **8.33 s**. The candidate primal recovery/certificate replay took
**8.52 s**; this duplicates mandatory certificate work in shadow mode and
must not be added as wholly new work in a live adaptive solver. The CG
decision timer overlaps the derivative timer and likewise must not be added.
The present timer does not isolate every small QDWH-gradient and Armijo
interval operation, making the following an optimistic lower estimate.

Using the side-by-side mean SVD evaluation cost for the 865 requested
decomposition fallbacks gives approximately `37.70 s` of fallback SVD work.
Even if derivative checks cost **zero**, QDWH, factor posterior, and fallback
alone would total approximately `23.50 + 1.63 + 37.70 = 62.83 s`, about
**55% slower** than the existing `40.57 s` SVD evaluations. Including the
measured derivative posterior gives an optimistic `71.16 s`, about **75%**
slower, before the small unisolated operations. The separate mandatory
certificate work applies to both paths. This is a fixed-trajectory cost
estimate, not an end-to-end candidate measurement; different fallback states
could change actual SVD time. Near the guard, the 54 small-shape evaluations
spent far more time in derivative posterior checks than in QDWH itself; those
small-shape timings must not be extrapolated to production shape.

The requested QDWH-controlled short trajectory required a **meaningful net
predicted speedup** in replay. This gate failed, and the loose Newton-direction
forward bound also leaves trajectory control unresolved. Consequently no
QDWH-controlled short or locked 50-step run was launched. Checkpoint
continuation was already verified for the unchanged production-SVD run, but
has not been validated for QDWH control. No end-to-end candidate speedup,
throughput, or memory figure is claimed.

## Answers and next step

1. QDWH factor and spectral errors admit cheap conditional bounds from the
   direct factorization residual and orthogonality defect, without runtime SVD.
2. The derivative residual equations produce a usable *error enclosure* for
   polar derivatives and HVPs on the tested states; the bound can be loose.
3. Curvature, CG stopping, descent, Armijo, and certificate branches can be
   conservatively classified near `1e-4` on this corpus, with ambiguity sent
   to SVD. This is not a proof that every admitted near-guard branch certifies.
4. No unsafe decision was observed in either tested corpus.
5. On the fixed real trajectory, only 66/931 evaluations needed no recorded
   SVD request; 865/931 did.
6. Thirty-two of 54 near-guard evaluations requested SVD, primarily because
   all rejected QDWH candidate certificates remained ambiguous.
7. Added factor and derivative posterior work totaled 9.96 s across the real
   replay; candidate certificate replay cost 8.52 s but duplicates mandatory
   work in this shadow experiment.
8. A candidate-controlled H100 speedup was **not measured** because its stage
   gate failed; the optimistic cost estimate is slower than production SVD.
9. Candidate-controlled checkpoint determinism was **not tested**. The
   unchanged SVD-controlled locked replay retained exact source batch/loss
   checks.
10. Production integration is **not justified** for this posterior. A future
    task would need a substantially cheaper decision validator or a stronger
    trajectory-control mechanism before reopening custom QDWH. The current
    full fp64 thin SVD remains the production backend and oracle.

No packages or data were downloaded. No LR sweep or long training was run.
