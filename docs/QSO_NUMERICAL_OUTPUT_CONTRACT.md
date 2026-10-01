# QSO numerical output contract

**Classification B: EPSILON-LMO CONTRACT NEEDS ADDITIONAL CONDITION.**
Feasibility plus an independently conservative support-value certificate is a
coherent numerical contract for the **primary** coupled spectral LMO. It does
not require a separately fitted direction-error budget. However, primary value
accuracy, even together with full rank at the current multiplier, cannot certify
the repository's minimum-Frobenius selection at a nonunique optimal face.
That secondary selection must remain a separate, explicitly scoped contract;
exact zero intrinsic cotangent must still select exactly zero. A future
implementation must not advertise a primary-value certificate as a certificate
of P-dagger recovery. This is the additional mathematical condition/scope
restriction, not a new empirical conditioning threshold.

A separate production-v1 **admission experiment for the full-rank smooth
branch** is justified with that distinction, conservative value/feasibility
bounds, and fail-closed numerical iteration safeguards. An unrestricted
replacement of the selected-direction contract is not justified. Production-v0,
its guard and tolerances remain unchanged. No training, sweep, download, CPU
ADMM, or production integration occurred in this study.

## 1. Exact epsilon-LMO theorem

Use balanced canonical weights U,D and the current coupled operator
`L(P)_i=<U_i,P_U[i]>-<D_i,P_D[i]>`. Let

```
F = {P: ||P_U||_2<=1, ||P_D||_2<=1, L(P)=0},
v(A) = max_{P in F} <A,P>,
B = A-L*(lambda),
phi(lambda) = ||B_U||_*+||B_D||_*.
```

F is nonempty, compact, convex and centrally symmetric. Its maximum exists
and v(A)>=0. For every feasible R, horizontality gives `<A,R>=<B,R>` and the
spectral/nuclear support inequality gives `<B,R><=phi(lambda)`. In particular,
`<A,P><=v(A)<=phi(lambda)` for every feasible P. Therefore

```
0 <= v(A)-<A,P> <= phi(lambda)-<A,P>.
```

No residual rank, differentiability, stationarity, unique multiplier, or unique
primal direction is needed. Even strong duality is unnecessary for this upper
bound; it explains why minimizing phi can make the bound tight. The authoritative
primal/dual derivation is in [HORIZONTAL_SPECTRAL_LMO.md](HORIZONTAL_SPECTRAL_LMO.md).
An upper bound G_upper on this gap is an absolute epsilon-LMO certificate with
epsilon=G_upper. It is not a certificate of distance to an optimal direction.

The two quantities are different contracts:

| Contract | Quantity certified | What it does not certify |
|---|---|---|
| Direction | Gauge-invariant distance to P* or P-dagger | A production budget has not been established |
| Primary epsilon-LMO | Feasibility and `v(A)-<A,P><=epsilon` | Direction distance, finite-step function error, secondary selection |
| Secondary selection | Minimum Frobenius norm on the primary optimal face | Does not follow from primary value accuracy |

For direction accuracy, the established full-rank theorem gives
`||P-P*||_F<=2 sqrt(2G/alpha)` at an arbitrary multiplier with both residuals
full rank. The conditional fp64 posterior exists, but is very loose and has no
principled production budget; see
[FULL_RANK_DIRECTION_ADMISSION.md](FULL_RANK_DIRECTION_ADMISSION.md).
The theorem remains valid. A primary value contract does not refute or replace
it with a claim that directions are automatically accurate.

## 2. Intrinsic normalization and descent meaning

A positive raw gauge sends `U_raw->C U_raw`, `D_raw->C^-1 D_raw`,
`H->C^2 H`. Balanced U,D and correctly transformed canonical A are invariant.
Hence F, v(A), the objective, lambda's original canonical multiplier convention,
and the absolute support regret are invariant. No raw Euclidean direction norm
is used to decide value accuracy. A deterministic solver with invariant inputs
can choose the same canonical approximate output and lift it equivariantly;
the contract alone does not enforce an implementation's deterministic selection.

Vertical changes `A->A+L*mu`, `lambda->lambda+mu` preserve B and every feasible
objective. Thus the exact contract is intrinsic also to the cotangent class.
Rounding/error allowances can become looser when a large vertical representative
is subtracted; this is a numerical representation issue, not a different exact
support function. The existing cancellation exit must not be removed.

Under positive objective scaling `A->tA`, `lambda->t lambda`, support regret
and phi scale by t. An absolute epsilon must scale too to describe equivalent
accuracy. A relative support criterion is invariant to t, aside from explicitly
declared floating-point tiny/underflow limitations. The raw lift
`Delta_U=sqrt(H) P_U`, `Delta_D=P_D/sqrt(H)` changes coordinates, not the
canonical support value. Raw Euclidean errors and update/parameter ratios are
gauge dependent and are not appropriate intrinsic error budgets.

For an exactly feasible P, central symmetry implies
`|<A,P>|<=v(A)<=phi(lambda)`. Therefore the exact counterpart of production's
denominator `max(abs(primal),abs(dual),tiny64)` is `max(phi,tiny64)`.
For nonzero phi define `tau_gap=(phi-<A,P>)/phi`. If tau_gap<=tau<1, then

```
<A,P> >= (1-tau) phi >= (1-tau) v(A),
(v(A)-<A,P>)/v(A) <= tau  (when v(A)>0).
```

Thus `3e-5` normalized gap guarantees at most `3e-5` relative support
regret; the two measured ratios need not be equal. It also defines the absolute
tolerance `epsilon=3e-5*phi`. Near-zero normalization must not use a floor of one. The
zero intrinsic class needs its separate exact-zero branch; no relative formula
determines a minimum-norm output when v=0.

When A is the current loss differential, minimizing that differential over
the quotient spectral unit ball is equivalent to maximizing `<A,P>` and
updating in direction -P. An epsilon-LMO output loses at most epsilon of the
best **first-order** decrease per unit step. If epsilon<v(A), it has positive
support and -P is a descent direction for that differential. A finite step
also has higher-order loss terms; no learning-rate or convergence theorem is
proved here. The guarantee permits a direction inside the ball, not necessarily
on its boundary.

Production uses updated canonical EMA A, not the instantaneous differential.
The same statement concerns its **momentum linear surrogate**. It does not
prove instantaneous descent, training stability, or superiority to AdamW.

## 3. What production-v0 actually accepts

The audit is of [coupled_solver.py](../qnormuon/coupled_solver.py), not an
inferred stronger guarantee. `certificate` horizontally projects the residual
polar pair, radially scales it with the conservative Gram upper norms, and
optionally flips sign. `accepted` checks:

| Check | Current threshold / purpose |
|---|---|
| Normalized nonnegative gap | <=3e-5; primary value accuracy |
| Signed normalized gap | >=-1e-10; expose materially inconsistent rounded values |
| Normalized horizontal residual | <=1e-10; numerical equality feasibility |
| Spectral excess | <=1e-12; numerical contraction feasibility |
| Smooth-loop residual rcond | >1e-4 before acceptance or Newton; numerical admission |

The production gap is an exact-arithmetic inequality **evaluated in floating
point**. Its original code is not the independently inflated value interval
introduced in this research audit. Similarly, tolerance-based equality checks
are not bitwise exact feasibility.

The rcond gate appears in the smooth loop and in its line-search trial gate.
It executes before gap acceptance and before the polar derivative/CG. There
are important scope exceptions: the exact-zero branch returns before this
gate; `accepted` itself contains no rcond test; the existing CPU reference
path checks primary gap/feasibility and reference convergence, not this smooth
admission gate. No fallback behavior was changed or silently strengthened here.

The guard is most accurately described as a **conservative polar-derivative /
Newton-HVP stability and polar-sensitivity admission policy**, also used to
avoid accepting a small value gap as if it established direction accuracy.
The denominators sigma_j and sigma_i+sigma_j in the derivative explain its
conditioning motivation. It is not a theorem of rank ambiguity: the five
Stage-D tails are millions of times larger than the estimated decomposition
uncertainty. Nor is it a calibrated direction-error budget. Rcond alone lacks
the absolute objective/gap scale, and no fixed error norm or allowed magnitude
was specified. These qualifications follow from Sections 2, 5–6 and 11 of
[DUAL_SOLVER_STUDY.md](DUAL_SOLVER_STUDY.md), the production contract, and the
two recent fixed-fixture reports.

## 4. Conservative numerical value certificate

Frozen fp64 tensor inputs and multiplier define the represented exact real
problem. Retain the previous study's conditional normal finite fp64 model,
`u=2^-53`, `gamma_k=ku/(1-ku)`, and its reduction/GEMM allowances. This is
neither directed interval arithmetic nor a formally certified CUDA backend.
Overflow, unmodeled underflow, inconsistent bounds, or failed factor checks
must fail rather than produce a certificate.

For matching cached factors, mathematical orthogonalization defines Bhat with
exactly the listed singular values. Reconstruction, factor orthogonality,
GEMM/reduction rounding, original residual construction, and centering/scaling
conversion give `||B_j-Bhat_j||_F<=eta_j`. The same-evaluation restriction is
essential: original residual singular values are internal values times the
matching positive magnitude, never factors from a different lambda/trial.

Nuclear Lipschitz continuity gives

```
sum_j(sum(s_j)-sqrt(n)*eta_j) <= phi(lambda)
    <= sum_j(sum(s_j)+sqrt(n)*eta_j).
```

Inflate scalar reductions to obtain `dual_lower<=phi<=dual_upper`, clamping
only the lower value bound to zero. No singular value is truncated.
The rank diagnostic is separately `alpha_lower_j=sigma_min_listed_j-eta_j`.
Neither positive alpha nor its inverse is required by the value-gap theorem.

For rounded returned p, let h upper-bound `||L(p)/sqrt(w)||_2`, including
dot-product/w rounding. Its mathematical exact horizontal projection q obeys
`||p-q||_F<=h`. Matching Gram radial bounds and final division allowances give
`rho>=max_j ||p_j||_2`. Define

```
t=max(1,rho+h), Pc=q/t in F,
delta_feas=h+(1-1/t)(||p||_F+h) >= ||p-Pc||_F,
primal_lower=fl(<A,p>)-dot_round-||A||_F*delta_feas,
G_upper=dual_upper-primal_lower (with scalar rounding allowances).
```

Then, under this model,
`0<=v(A)-<A,Pc><=G_upper`. Rank and direction distance have disappeared from
the value proof. If dual_lower>0, `G_upper/dual_lower` upper-bounds the exact
gap/phi normalization; using dual_upper as the denominator would not give this
conservative normalized-gap comparison. The code also reports
`G_upper/primal_lower` when primal_lower>0 as another bound on regret/v.
For a certified normalized gap below one, the preceding exact inequality gives
the tighter relative-regret bound `G_upper/dual_lower` itself.

The represented output contract must state this finite-precision scope:
an exactly feasible proxy Pc with certified support regret, and a returned
fp64 p within delta_feas, retaining all existing feasibility checks. It must
not call an approximately horizontal rounded tensor exactly feasible. On this
replay delta_feas is below `4.42e-11` at first candidate acceptance. Final raw
lifting, model-dtype casting and parameter addition introduce separate
rounding; they do not inherit the unchanged pre-cast certificate automatically.

[qso_output_contract.py](../experiments/qso_output_contract.py) reuses the
existing direction-posterior machinery only to obtain its rank, feasible-proxy
and value allowances. It does not use its direction bound as an acceptance
test. There is no reference solution, direction budget, or fixture-fitted
constant in its inputs. No additional tall SVD, eigendecomposition, dense
multiplier Hessian, or CPU reference is needed by the posterior.

## 5. Numerical rank is separate from iteration reliability

`alpha_lower>0` certifies full column rank of the **current** residual under
the model. It is more informative about rank than the fixed relative guard.
It does not prove a full-rank optimum, multiplier uniqueness, or a
well-conditioned Hessian. Even at rcond=1, perturbations of form `polar(B) T`
with symmetric T can have zero curvature. The row-whitened Hessian upper
bound grows like `1/sigma_min`; no positive lower Hessian bound follows merely
from row whitening or positive sigma_min.

During iteration, reliable factor reconstruction, positive/representable
singular values, finite derivative/HVP arithmetic, usable curvature and
descent, and bounded globalization work remain necessary operational checks.
The ratios eta/sigma_min quantify decomposition/rank separation but are not
alone a forward HVP-error theorem. A future admission study must examine
below-guard numerical derivative behavior and fail closed on unusable actions.
No proposed fixed replacement threshold is introduced here.

At final acceptance, feasible-proxy/value bounds authorize the primary LMO
output, independently of whether intermediate Newton decisions were exactly
the ideal decisions. This is not authorization to change the current solver.
An inaccurate HVP can waste work or trigger failure even when a final value
certificate is meaningful. Conversely a final conservative certificate does
not require a small error relative to every exact internal Newton direction.

## 6. Five Stage-D fixtures: new H100 replay

The fixed stored U,D,A and original warm lambda are exactly those in
[STAGE_D_QSO_FAILURE_FORENSICS.md](STAGE_D_QSO_FAILURE_FORENSICS.md).
No model, token stream or training trajectory was regenerated. The separate
research configuration admits computed positive spectra, disables reference
fallback, and uses the earlier strict diagnostic stopping target to expose
later iterates. Full fp64 SVD, Newton/CG, damping, line search, and Gram primal
recovery are unchanged. The candidate audit uses the unchanged **3e-5**
value tolerance. Strict continuations eventually encounter the documented
strict-target line-search limitation; this does not invalidate earlier audited
value candidates or constitute production admission.

| Fixture | First candidate Newton | Conservative normalized gap | G_upper | alpha_lower U / D | Min rcond | CG / line trials |
|---|---:|---:|---:|---|---:|---|
| Tuned | 3 | 1.282376e-5 | 1.597004e-5 | 4.795752e-5 / 3.758054e-5 | 8.678984e-5 | 9 / 3 |
| Low | 3 | 1.204232e-5 | 2.496129e-5 | 4.270843e-5 / 4.486446e-5 | 7.741487e-5 | 10 / 3 |
| Middle | 3 | 2.656944e-5 | 1.097148e-4 | 7.522049e-5 / 6.448236e-5 | 4.489405e-5 | 11 / 3 |
| High | 4 | 4.610439e-7 | 2.069576e-6 | 1.172483e-4 / 1.110285e-4 | 7.706061e-5 | 17 / 4 |
| Ablation | 4 | 1.944436e-7 | 1.222732e-6 | 1.708874e-4 / 1.603509e-4 | 8.761157e-5 | 13 / 4 |

All five pass existing feasibility checks at these iterations: normalized
horizontality <=`8.81e-17`, spectral excess zero, positive signed gap.
Their earlier candidate normalized bounds are:

| Fixture | Newton 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Tuned | .0373543 | .00763515 | .000583945 | 1.282376e-5 | 1.064672e-7 |
| Low | .0766280 | .0107445 | .000794632 | 1.204232e-5 | 1.237143e-7 |
| Middle | .0768393 | .0145804 | .000354230 | 2.656944e-5 | 1.986364e-7 |
| High | .103536 | .0227643 | .000725299 | 6.441799e-5 | 4.610439e-7 |
| Ablation | .0688398 | .0160229 | .000907656 | 4.722792e-5 | 1.944436e-7 |

The tuned gap passes at Newton 3, as before, with conservative normalization
now including denominator uncertainty. Rank lower bounds stay positive. All
accepted-iterate spectra, gradient norms, feasibility, Newton/CG/trial work,
and allowances are in the structured summary. This audit changes no parameters.

### Offline reference comparison, not acceptance input

The independent high-accuracy solutions from
[NEAR_RANK_OPTIMUM_CLASSIFICATION.md](NEAR_RANK_OPTIMUM_CLASSIFICATION.md)
are rechecked in original coordinates. They are numerical references, not
formal interval-exact P*. Direction comparison is diagnostic only.

| Fixture | Observed relative direction error | Observed support regret | Lambda difference norm |
|---|---:|---:|---:|
| Tuned | 1.383039e-5 | 1.596888e-5 | 1.190792e-7 |
| Low | 1.392306e-5 | 2.495979e-5 | 1.658883e-7 |
| Middle | 2.917486e-5 | 1.097110e-4 | 5.498837e-7 |
| High | 4.912966e-7 | 2.066058e-6 | 1.115373e-8 |
| Ablation | 2.176514e-7 | 1.218021e-6 | 8.450533e-9 |

Every measured regret is below G_upper. Positive singular tails remain near
the independent optima, not near decomposition uncertainty. These observations
support useful primary value accuracy on these fixtures. They are not a
direction-threshold fit or proof of all future training stability.

## 7. Ordinary above-guard controls

The six saved locked step-0 pairs and one saved real warm diagnostic pair are
the same available controls as the previous study. Their original production
configuration passes, and the new value/rank posterior also passes at the
same stopping iterate:

| Control | Newton | Conservative normalized gap | Min rcond |
|---|---:|---:|---:|
| Step-0 block 0 | 3 | 2.280648e-7 | .00758757 |
| Step-0 block 1 | 3 | 2.770061e-7 | .00788653 |
| Step-0 block 2 | 2 | 2.275907e-5 | .00590656 |
| Step-0 block 3 | 3 | 3.362742e-7 | .00496293 |
| Step-0 block 4 | 3 | 9.808267e-7 | .00315123 |
| Step-0 block 5 | 3 | 7.595917e-7 | .00212402 |
| Saved warm block 0 | 2 | 1.941916e-5 | .0101743 |

No new constants separate failures from controls. Full saved locked warm-input
coverage is still unavailable without regenerating a model; these seven
controls are not a full trajectory distribution. Conservatively inflated
bounds can reject a raw production gap arbitrarily close to its tolerance.
That would be legitimate extra work, not grounds to fit or loosen allowances.

## 8. Counterexamples and scope of the decision

Use regular `U=D=[[1,0],[0,1],[1,0]]` throughout the diagonal examples, so equal
P sides are horizontal. Let `B(t)=[[1,0],[0,t],[0,0]]` and
`A_U=A_D=B(t)`.

For t>0 the unique exact primal optimum is both sides `diag(1,1)` padded with
a zero row. Candidate `diag(1,0)` is feasible, has regret 2t, and has pair
distance sqrt(2) from that optimum. Its normalized gap is t/(1+t), arbitrarily
small as t decreases. The posterior accepts the t=`1e-10` example with
positive numerical rank lower bounds; it cannot and does not claim direction
proximity. At a fixed unique problem epsilon->0 forces approximate maximizers
to the optimizer by compactness, but there is no uniform rate as t->0.
This example limits a **direction** interpretation, not the epsilon-LMO theorem.

At t=0 the primary face is nonunique and the accepted minimum-Frobenius
selection has second diagonal entries zero. The feasible pair `diag(1,1)` is
primary-exact yet distance sqrt(2) from P-dagger. More strongly, at the current
multiplier `lambda=[0,1e-6,0]`, both residuals are full rank, with singular
values 1 and `1e-6`. The candidate still has essentially zero true regret,
gap/phi about `1e-6`, and passes positive alpha_lower. Thus **primary value,
feasibility and current numerical rank cannot certify secondary selection**.
This is why unrestricted classification A would silently weaken an accepted
mathematical contract. At the actual deficient multiplier lambda=0, the value
posterior also certifies the primary value while its separate rank test rejects
smooth use. No separate partial polars are used as a coupled solution.

Other deterministic checks:

- Scaling A_D by `1e-6` creates highly imbalanced residual magnitudes. At
  t=`1e-5`, dropping the weak direction still loses about `1e-5` relative
  support; the down alpha is about `1e-11`. This is meaningful primary value
  accuracy, not permission to ignore the weak direction in an exact selection.
- Repeated positive singular values (t=1) have no special instability in the
  contract. Dropping half the support produces normalized gap .5 and rejects.
- With m=2,n=1 and U=D=A_U=A_D=ones, the unique primal is the shared normalized
  vector while a segment of multipliers is optimal. Both tested full-rank
  multipliers give certified outputs. Lambda proximity/uniqueness is unnecessary.
- At A=0 every feasible pair has perfect primary value; pure value semantics
  cannot select zero. Preserve the existing exact intrinsic-zero branch.
- Positive gauges recanonicalize to the same data; objective scaling over
  `1e-20`, 1 and `1e20` preserves the normalized outcome without a new threshold.

No counterexample falsifies the primary epsilon-LMO inequality. Counterexamples
do falsify identifying it with P-dagger recovery, exact finite-step function
equivalence, or a universal convergence guarantee. Nonunique optima can differ
in update coordinates and function trajectory despite identical support.

## 9. Tolerance provenance and recommended contract

The earlier solver study chose `3e-5` for fp32 numerical experiments and used
`1e-6` for fp64 reference work. Production inherited a normalized gap tolerance
of `3e-5`; the later precision study retained it while changing the default
smooth arithmetic to fp64 because the real fp32 floor exceeded that target.
The production document now specifies it explicitly independently of dtype.
It functions as an accepted **numerical primary LMO value tolerance**. There is
no repository derivation of it as a model-quality accuracy requirement, a
training convergence constant, or a gauge-invariant direction budget. This
study keeps it unchanged and makes its gap/phi meaning precise.

A future v1 contract should explicitly distinguish:

1. Represented-input primary accuracy: exact feasible proxy Pc, bounded
   returned-tensor feasibility distance, and conservative gap/phi <=`3e-5`.
   All existing signed-gap, horizontal and spectral checks remain.
2. Smooth numerical validity: reliable matching full fp64 SVD data and positive
   residual rank lower bounds, with bounded, observable failure/globalization
   behavior. Positive rank alone is not an HVP-stability guarantee.
3. Selection: exact zero stays zero; deficient/nonunique optimal-face
   minimum-Frobenius recovery remains a separate requirement wherever selected
   recovery is claimed. A primary-only result must be labeled as such, as the
   existing reference fallback already is. Full rank at a current lambda
   must never be used as a proof of optimal-face uniqueness.
4. Precision scope: certification before lifting/casting; subsequent error is
   separately measured/bounded. No claimed fp64 contract for bf16 stored updates.

The additional selection condition is structural, not a fitted norm threshold.
One may explicitly study an approximate-primary optimizer without certifying
selected-direction proximity, but that is an explicit output-contract choice,
not an unnoticed theorem about the accepted P-dagger. The five independently
classified full-rank optima permit a narrowly scoped admission experiment
without needing deficient-face recovery. General singular-face behavior is
still separate research. No conclusion authorizes simply reducing the fixed
guard or starting the failed confirmation here.

## 10. Cost, validation and artifacts

The final tensor audit ran in SLURM job **29077**, one H100 NVL, four CPUs,
16 GiB, 15-minute limit, on lagrange0. Job completed in 62 seconds, exit `0:0`.
Python 3.10.20, torch 2.10.0+cu128/CUDA 12.8; deterministic algorithms enabled,
TF32 disabled. Shared-home paths were checked before loading. The stable
environment was not changed. Existing production hashes were asserted before
and after the audit.

The synchronized posterior timings over 71 current-iterate audits were
**2.321 ms median**, approximately **2.409 ms p95** (empirical order statistic).
This is observed in-solve overhead including the reused direction-study
allowances, not a new standalone kernel benchmark or optimization claim.
Current factors and radial metadata suffice. Extra full SVDs for offline KKT
references are explicitly outside the candidate posterior.

The complete suite passed **427 tests, zero failures/errors/skips**, in
**25.824 s** by JUnit. Nine new parameterized research cases cover analytic
tiny/repeated/imbalanced spectra, normalization, no additional decomposition
or CPU reference, gauge canonicalization, nonunique faces with full-rank
current residuals, and unique primal/nonunique dual. Existing tests were
unchanged. The offline network guard recorded no outbound attempts.

Artifacts:

- [research posterior](../experiments/qso_output_contract.py),
  [independent tests](../tests/test_qso_output_contract.py);
- [fixed-fixture runner](../cluster/qso_output_contract.py),
  [SLURM script](../cluster/qso_output_contract.sbatch);
- [structured complete iterate audit](../cluster/qso_output_contract_summary.json);
- [final raw SLURM log](../cluster/qso_output_contract-29077.log),
  [final JUnit XML](../cluster/output-contract-tests-29077.xml).

The same compact summary is under
`/home/prignano/qnormuon-runs/output-contract/study-29077/`.
The initial audit also passed 427/427 tests; its
[log](../cluster/qso_output_contract-29076.log) and
[XML](../cluster/output-contract-tests-29076.xml) are retained. The final rerun
includes the sharper relative-regret explanation and independent tests of the
dual interval and normalized regret bound.
No new large tensor files were saved. The pre-existing AGENTS.md documentation
edit was preserved; this study did not edit it or production sources.

## Explicit answers

1. **Existing gap:** primary support regret is bounded by any valid dual upper
   bound minus a feasible primal value. The raw production calculation is not
   an outward-rounded interval; this research adds conditional allowances.
2. **Direction versus value:** approximate steepest-descent support semantics
   need value accuracy, not a universal prescribed direction distance.
   Selected-direction recovery is a stronger separate contract.
3. **Nonuniqueness:** epsilon-optimality remains valid but cannot select
   P-dagger; even exact primary optimality can leave order-one selection error.
4. **Gauge-invariant contract:** canonical feasible-proxy support regret,
   normalized by the original dual support bound, with explicit numerical
   repair/cast scope and separate selection semantics.
5. **Rank/conditioning:** rank posterior validates current smooth data;
   derivative/Newton reliability remains operationally distinct. Neither
   rank nor conditioning is needed for the exact primary value theorem.
6. **3e-5:** can serve as the unchanged normalized primary gap tolerance;
   it also upper-bounds relative support regret for a positive supported value.
   It is not a mathematically derived training-quality tolerance.
7. **Five Stage-D fixtures:** all pass the research value/rank checks;
   first iterations 3,3,3,4,4. Production still rejects them under its guard.
8. **Controls:** all seven available controls pass at their ordinary stopping
   iterations; no constant was fitted and full warm-corpus coverage is absent.
9. **Limits:** tiny singular values permit large direction error, nonunique
   faces defeat minimum-norm selection, and full rank at current lambda does
   not prove an optimum is smooth/unique. No loss-decrease claim follows for EMA.
10. **Next step:** a separate, explicitly scoped production-v1 full-rank
    admission experiment is justified by this precise **B** result. It must
    preserve selection scope, conservative final value/feasibility verification,
    and fail-closed iteration behavior. No production change or resumed
    training is performed or automatically approved by this report.
