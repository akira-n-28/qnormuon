# Decision-relevant smooth Gram posterior: research study

## Decision

**B — the conditional bounds are defensible under the stated fp64 operation
model, but the present validation cost and uncertain decisions do not justify
production integration.** The production `SmoothDual` still uses full fp64
thin SVD. No quotient geometry, rcond guard, certificate threshold, production
backend, or training configuration changed. The research selector in
[`experiments/smooth_gram_decisions.py`](../experiments/smooth_gram_decisions.py)
uses Gram only when a supplied decision-margin gate certifies the particular
decision; otherwise it recomputes full SVD with an explicit reason. The
[`H100 replay`](../cluster/study_smooth_gram_decisions_h100.py) keeps **full SVD
in control** so that all comparisons use identical stored states and batches.
It is a counterfactual decomposition replay, not a Gram-controlled optimizer
trajectory.

The full unfiltered suite ran successfully in SLURM job 28968 (exit `0:0`),
followed by a 50-step locked replay and near-guard stress. The replay kept
the original batch hashes and losses, certified 300/300 pair solves, and used
zero optimizer/reference fallback. The compact records are in
[`cluster/smooth_gram_decisions-28968.json`](../cluster/smooth_gram_decisions-28968.json)
and the raw log is
[`cluster/study_smooth_gram_decisions_h100-28968.log`](../cluster/study_smooth_gram_decisions_h100-28968.log).
No data or packages were downloaded.

## Polar derivative equations and exact residual bound

For full-column-rank `B=P H`, `PᵀP=I`, and symmetric positive-definite
`H=(BᵀB)^{1/2}`, let `Y=Dpolar_B[E]` and `Omega=PᵀY`. Differentiating
`B=P H` and orthogonality gives

```
PᵀY + YᵀP = 0,             Omegaᵀ = -Omega,
H Omega + Omega H = PᵀE - EᵀP,
(I-PPᵀ)Y H = (I-PPᵀ)E.
```

The last equation follows by applying the normal projector to
`E=Y H+P(DH)`. The three equations uniquely specify `Y`: the Sylvester
operator on skew matrices has inverse Frobenius norm at most `1/(2 mu)`
when `mu=sigma_min(H)>0`, and the normal equation has inverse norm at most
`1/mu`. Repeated or clustered **positive** singular values do not weaken
these statements. No individual `U` or `V` column is compared.

For any candidate `Yc`, evaluate the three residuals `T`, `S`, and `N` at
the *true* `P,H`. Decompose `Yc-Y` into tangent and normal components. The
following is an exact matrix inequality:

```
||Yc-Y||_F <= sqrt((||T||_F/2 + ||S||_F/(2 mu))^2
                    + (||N||_F/mu)^2).
```

The implementation never knows exact `P,H`. It obtains upper bounds on their
errors from the computed Gram `Pt,Ht` without another full SVD. If
`d=||PtᵀPt-I||_F<1`, let `Q0=polar(Pt)` *in the proof only*. Then

```
||Q0-Pt||_F <= ||Pt||_F d/[sqrt(1-d)(1+sqrt(1-d))].
```

`B0=Q0 Ht` has exact polar `Q0` when `Ht` is SPD. Set `e0` to the measured
`||B-Pt Ht||_F`, the preceding `Q0-Pt` contribution, and a standard-model
rounding allowance. The Gram eigensystem defect supplies a lower eigenvalue
bound `mu0` for `Ht`. If `e0>=mu0`, this posterior fails. Otherwise Weyl's
inequality keeps `B0+t(B-B0)` full rank with minimum singular value at least
`mu0-e0`. Integrating the polar derivative bound along that segment gives

```
eta_P = ||P-Pt||_F <= ||Q0-Pt||_F + 2 e0/(mu0-e0),
eta_H = ||H-Ht||_F <= eta_P ||B||_2 + e0,
mu >= mu0-e0.
```

The factor 2 is a conservative bound for the full-rank polar derivative.
`||B||_2` is bounded by the preceding Gram spectrum posterior, not computed
by `svdvals`. The residuals measured at `Pt,Ht,Yc` are inflated by triangle
inequalities using `eta_P`, `eta_H`, `||Yc||`, and `||E||`; their explicit
formulas are in `derivative_error_bound`. For example, the true tangent
residual is at most measured tangent residual plus `2 eta_P ||Yc||_F`;
the true normal residual adds a projector-error term no larger than
`eta_P(2+eta_P)||Yc Ht-E||_F` and `eta_H||Yc||_F`. Substitution in the exact
inequality gives the computable `epsilon_Y`. These bounds do not depend on
the gauge choice inside a repeated-singular-value eigenspace.

The algebraic inequalities above hold for exact operations on the measured
matrices. Inflation of measured GPU reductions and GEMMs uses the standard
fp64 `gamma_k=ku/(1-ku)` model, `u=2^-53`, with conservative operation counts.
This is **not** directed-rounding interval arithmetic or a backend-specific
proof about every CUDA kernel. The full-SVD oracle tests are therefore part
of the numerical claim. The posterior can be loose, particularly near the
guard; loose means explicit SVD work, never an assumed accurate derivative.

## From derivative uncertainty to decisions

With row-whitened effective weights,

```
||L_eff*(v)||_F^2
  = sum_i v_i^2 (||U_eff,i||²+||D_eff,i||²) = ||v||².
```

Thus `||L_eff||_op=||L_eff*||_op=1`. Combining the two side-specific
derivative bounds gives
`||HVP_gram(v)-HVP_true(v)||_2 <= sqrt(epsilon_Y,U²+epsilon_Y,D²)`.
Likewise the polar uncertainty bounds the dual-gradient error by the pair
Frobenius polar-error bound. For the stored fp64 effective rows, the prototype
actually measures the maximum combined row norm and inflates its square for
the `2n`-term reduction, using that upper estimate of `||L_eff||` in both
error bounds; it does not assume rounded rows have norm exactly one. Damping
is an exact `delta*v` term and does
not add decomposition error.

For curvature `c=vᵀ(HVP_gram(v)+delta*v)`, the true curvature is within
`c +/- ||v|| epsilon_HVP(v)`. The sign is certified only when that interval
excludes zero. This explicitly sends marginal or negative-curvature cases to
full SVD. For a candidate CG direction `x`, the implementation directly
checks one additional Gram HVP at `x` and bounds the true damped-system
residual by

```
||g_gram+(H_gram+delta I)x|| + epsilon_g + epsilon_HVP(x).
```

Equivalently, if `x=sum alpha_j p_j`, a weaker bound accumulates
`sum |alpha_j| epsilon_HVP(p_j)`. The true CG stopping target is bounded
below using `max(0,||g_gram||-epsilon_g)`. If the residual upper bound
exceeds that target, CG is uncertain. Since the true Hessian is PSD and
`delta>0`, the distance to the exact *damped Newton* direction is at most
the residual upper bound divided by `delta`; this often becomes very loose.
The implemented descent check avoids relying on that loose direction bound:
`g_trueᵀx` lies within `g_gramᵀx +/- epsilon_g||x||`. An interval touching
zero triggers `gram_descent_ambiguous`.

The Gram eigensystem supplies conditional lower/upper nuclear-value intervals
for current and trial objectives. For positive `alpha`, Armijo acceptance is
certified only if the **trial upper** value is below the **current lower**
value plus the **slope lower** term and the minimum permitted rounding
allowance. Rejection is certified by the opposite strict inequality using
trial lower/current upper/slope upper and maximum allowance. The trial's
rcond lower bound must exceed `1e-4`; the production resolved-decrease or
gradient-improvement condition must also be certified. Any overlap returns
`gram_armijo_ambiguous`. Certificate acceptance separately uses a dual upper,
primal lower, signed-gap lower, positive denominator lower, rcond lower, and
the unchanged production feasibility checks. There is no tolerance relaxation.

## Independent oracle and adversarial tests

The tests in
[`tests/test_smooth_gram_decisions.py`](../tests/test_smooth_gram_decisions.py)
compare polar and derivative upper bounds against independent full fp64 SVD
at prescribed rconds down to `1.01e-4`, repeated top/bottom singular values,
and a nearly derivative-null perturbation. Synthetic curvature, descent,
Armijo equality, and CG-residual intervals test that marginal cases are
classified uncertain. The selector test intercepts `torch.linalg.svd` to
verify that a certified Gram selection does not invoke it, while an uncertain
selection does and reports its reason. The original smooth-Gram adversarial
corpus already covers larger `[1024,384]` matrices, extreme scales, clustered
spectra, and values below the guard; its structural/rcond posterior is reused
unchanged.

The 50-step real H100 replay attempted Gram at all **931** smooth residual
evaluations. All 931 passed structural/rcond and the new polar-error posterior.
The conditional value intervals contained the full-SVD oracle for all 931;
the gradient bounds had zero observed violations. All **1,436** observed HVP
errors lay inside their bounds, and all 1,436 oracle curvature signs agreed
with the certified curvature signs. Of 631 Newton/CG decisions, **630** were
certified and one returned `gram_cg_uncertain`. The largest observed Gram/SVD
truncated-CG direction difference was about `1.23e-9` relative; the generic
`residual/delta` forward bound was much looser and cannot certify a close
trajectory by itself. Of 631 line-search trials, **505** accepted decisions
were certified and **126** returned `gram_armijo_ambiguous`; none made an
unsafe decision. All 931 certificate decisions matched the full-SVD path:
300 accepts and 631 rejects. The union of uncertain decision reasons is
**127/931 evaluations** (126 Armijo, one CG); **804/931**, or 86.4%, have no
recorded reason requiring a full smooth SVD on this *fixed* trajectory.

For the missing near-guard regime, five constructed valid coupled problems
used residual rconds `1.01e-4`, `1.05e-4`, `1.2e-4`, `2e-4`, and `3e-4` at
the initial multiplier. A row reflection preserves balanced row norms and
`L(A)=0` while ensuring a nonzero polar dual gradient, so each problem takes
three real Newton steps rather than trivially certifying at iteration zero.
Across **41** evaluations and **238** HVP oracle comparisons, all error
intervals contained the observed values. Only **125/238** curvature signs
were certified. **10/15** CG/descent decisions were certified; five marginal
curvatures triggered explicit SVD requests. All 36 Armijo decisions and 20
certificate rejections agreed with the oracle. The counterfactual
decomposition fallback rate was **5/41 = 12.2%**, with no unsafe decision.
This constructed trajectory is an optimizer-solver stress test, not language
model training, and its three-iteration cap deliberately leaves the problems
uncertified. It supplies evidence of safe *rejection* near the guard, not a
claim that every possible near-guard direction is accurate. Warmed kernel
timings for these `[2,48,16]` residuals are recorded separately in
[`cluster/smooth_gram_decisions-28969.json`](../cluster/smooth_gram_decisions-28969.json).

## Cost and production implication

The synchronized real replay measured **11.22 s** for 931 Gram attempts plus
base posterior checks, **8.77 s** for derivative/HVP posterior work, and
**8.55 s** for the additional Gram primal certificate replay. Its **5.77 s**
CG-decision timer includes derivative work already counted above and must not
be added to it. These are shadow costs across 50 steps, not a live adaptive
optimizer measurement. At the prior warmed H100 full-SVD figure of about
43.36 ms per two-matrix evaluation, 931 full decompositions would cost roughly
40.4 s; 127 decomposition fallbacks add an estimated 5.5 s. The rough
`11.22+8.77+8.55+5.5=34.0 s` accounting suggests only **~1.19x**
potential speedup for this smooth decision segment after validation, despite
the earlier ~4x isolated Gram decomposition advantage. It is not a measured
end-to-end step speedup: replay duplicates certificates and the currently
SVD-controlled trajectory is not the one Gram would generate. Near the guard,
the median warmed **full-SVD pair** took **1.52 ms**, while the Gram plus base
posterior pair took **3.68 ms**. Adding the observed `5/41` fallback fraction
would make that decomposition-only estimate about **3.87 ms/evaluation**,
roughly **2.55× slower** than full SVD at this small stress shape, before
derivative checks. The 41-evaluation solver stress measured an additional
**0.78 s** of derivative/HVP posterior work; its **0.31 s** CG-decision timer
overlaps those derivative measurements. Marginal curvature forced fallback on
every constructed sequence. The near-guard small-matrix result must not be
extrapolated directly to `[1024,384]`, but it is a concrete warning that
posterior overhead can erase all kernel savings.

Therefore the answer to the production question is **no for this prototype**.
The bounds do make curvature, descent, CG stopping, Armijo, and certificate
decisions testable without matching individual SVD factors, and observed
interval containment was complete. But the whole-solver trajectory under
Gram-controlled directions remains unvalidated, the damped Newton direction
bound is too loose to ensure nearby future states, and the present posterior
cost consumes most of the isolated kernel speedup. Full fp64 SVD remains the
production smooth backend and the independent oracle. Any later integration
needs a true adaptive trajectory validation, stronger or cheaper direction
control, explicit decomposition fallback reasons, and renewed locked H100
checkpoint/trajectory tests. No LR sweep or long training was run.
