# Practical dual solvers for the K=I horizontal spectral LMO

Date: 2026-09-24. Research prototypes and CPU measurements; **no production
integration, shared K, contribution balancing, or GPU kernel changes**.
The complete original specification, THEORY_AUDIT, RANK_DEFICIENT_THEORY,
ZERO_STRATUM_GEOMETRY and HORIZONTAL_SPECTRAL_LMO reports were read first.
The intrinsic interpretation is in [QUOTIENT_SPECTRAL_GEOMETRY.md](QUOTIENT_SPECTRAL_GEOMETRY.md).

The practical result is conditional but encouraging: warm-started matrix-free
Newton-CG reaches a normalized primal-dual gap of 1e-6 within **two iterations
on all eight measured transitions** of both a slowly varying EMA proxy and an
actual small fixed-batch SwiGLU trajectory. One iteration is insufficient in
float64 on both sequences. This is not a universal iteration bound or a
large-model training result. L-BFGS is competitive and avoids Hessian products;
plain gradient methods are substantially less reliable at small iteration budgets.

The difficult cases still matter. Ill-conditioned residuals trigger an explicit
generic ADMM fallback; even its high-accuracy objective certificate cannot
guarantee direction accuracy near rank loss or minimum-Frobenius selection on
a nonunique optimal face. The reference, approximation and exact-direction
contracts are kept separate throughout this study.

Files:

- [dual_solver.py](../experiments/dual_solver.py): four smooth methods, cached-SVD
  derivatives, recovery/certification, reference wrapper and error measurements.
- [dual_solver_study.py](../experiments/dual_solver_study.py): reproducible
  sequences, adversaries, timing and matched-tolerance runs.
- [dual_solver_results.json](../experiments/dual_solver_results.json): **979
  individual records**, including 914 main measurements and 65 supplemental
  matched-tolerance/analytic checks. This includes intentionally uncertified
  limited-budget results, not 979 successful solves.
- [test_dual_solver.py](../tests/test_dual_solver.py): 25 new parameterized
  geometry and solver cases. The complete suite has **150 passing cases**.

## 1. The objective is a sum, not a difference

Part B's displayed expression has a minus sign between its two nuclear norms.
That conflicts with Part A, the accepted LMO and its already-proved dual. The
implemented problem is

\[
\phi(\lambda)=\|B_U\|_*+\|B_D\|_*,\qquad
B_U=A_U-\operatorname{diag}(\lambda)U,\quad
B_D=A_D+\operatorname{diag}(\lambda)D.
\tag{1}
\]

The difference is not the support value or a convex norm minimization problem.
For U=D=A_U=A_D=(1,1)^T, the difference at lambda=0 is zero, while the
horizontal primal value is 2sqrt(2). Its infimum is therefore not that primal
value. A regression test preserves this counterexample. The plus in (1) is
an explicit interpretation of the inconsistent display, not a new optimizer
design or a silent change to the accepted variational problem.

The primal remains `max <A,P>` over L(P)=0 and both operator norms <=1.
The geometric quotient dual norm and support value agree at every rank, and
all regular-domain gauge qualifications from the preceding reports still apply.

## 2. Full-rank derivatives and matrix-free Hessian products

Let a residual have thin SVD `B=Q diag(sigma) V^T`, sigma_j>0. Its polar is
P=QV^T. For a perturbation E define

\[
F=Q^TEV,\quad N=EV-QF,\quad
\Omega_{ij}=\frac{F_{ij}-F_{ji}}{\sigma_i+\sigma_j}.
\]

Differentiating B=P S and P^T P=I gives

\[
\boxed{D\operatorname{polar}_B[E]
=\big(Q\Omega+N\operatorname{diag}(\sigma^{-1})\big)V^T.}
\tag{2}
\]

The skew part solves the polar Sylvester equation; the remaining part follows
by projecting E orthogonal to Q. No differences of singular values appear,
so repeated **positive** singular values are harmless. Small singular values
still make this derivative ill-conditioned. This is the full-column-rank
Fréchet derivative, consistent with
[Gawlik and Leok, Computing the Fréchet Derivative of the Polar Decomposition](https://arxiv.org/abs/1608.04491).

Using L(P)_i=<U_i,P_U[i]>-<D_i,P_D[i]> and its adjoint,

\[
g(\lambda)=-L(P),\qquad
\boxed{\nabla^2\phi(\lambda)v
=-L\left(D\operatorname{polar}_{B_U}[-\operatorname{diag}(v)U],
          D\operatorname{polar}_{B_D}[\operatorname{diag}(v)D]\right).}
\tag{3}
\]

Both residual SVDs are cached once per value/gradient evaluation. Each Hessian-
vector product uses those factors and matrix multiplications, with no new SVD
and no m x m Hessian. Work is O(m n^2) per product and storage O(m n+n^2), in
addition to O(m) CG vectors. A dense Hessian is constructed only in tiny tests
to compare with independent autograd, never in the practical solver.

The quadratic form is

\[
D^2\|B\|_*[E,E]=
\sum_j\frac{\|N_{:,j}\|^2}{\sigma_j}
+\sum_{i<j}\frac{(F_{ij}-F_{ji})^2}{\sigma_i+\sigma_j}\ge0.
\tag{4}
\]

Tests compare (3) with finite differences of g and autograd Hessian products
in float64 and float32, check symmetry and PSD, and include repeated positive
singular values. Instrumented tests verify that the products do not call SVD.

## 3. What the row preconditioner proves

Let `W=diag(w_i)`, `w_i=||U_i||^2+||D_i||^2`, so `LL*=W`. On regular rows W>0.
Changing multiplier coordinates to `z=W^{1/2}lambda` replaces L* by
`L* W^{-1/2}`, an isometry from Euclidean multiplier space into the product
Frobenius matrix space. Gradient descent in z is preconditioned descent
`lambda <- lambda - alpha W^{-1}g` in the original coordinates.

Let s_min and s_max bound the positive singular values of both residuals.
From (4),

\[
0\preceq\nabla^2\phi\preceq s_{\min}^{-1}W,\qquad
0\preceq W^{-1/2}\nabla^2\phi W^{-1/2}\preceq s_{\min}^{-1}I.
\tag{5}
\]

The row scaling is therefore theoretically justified as constraint-Jacobian
whitening and as removal of w_max from this upper curvature bound. It is not
only an empirical heuristic. It is also not an unconditional improvement in
the actual condition number.

For a precise lower-bound qualification, let T_B project perturbations onto
the polar-changing parts `Q skew(F)V^T + NV^T`, removing the zero-curvature
directions `P times symmetric`. Define

\[
\gamma=\inf_{\|v\|=1}\sum_{j=U,D}
\|T_{B_j}((L^*W^{-1/2}v)_j)\|_F^2.
\]

Then 0<=gamma<=1, and if gamma>0,

\[
\frac\gamma{s_{\max}}I\preceq W^{-1/2}\nabla^2\phi W^{-1/2}
\preceq\frac1{s_{\min}}I,\qquad
\kappa\le\frac{s_{\max}}{\gamma s_{\min}}.
\tag{6}
\]

The unscaled bound can include the extra factor w_max/w_min. But W supplies
no positive lower bound for gamma: alignment can leave arbitrarily small
curvature or a flat multiplier direction even at well-conditioned full-rank
residuals. Singular multiplier Hessians are allowed; a nonunique multiplier
does not imply a nonunique primal when both residuals are full rank.

Common scaling of both weight rows by positive rho_i leaves the horizontal
hyperplane unchanged. With the corresponding multiplier change, the Hessian
becomes `diag(rho) Hess(phi) diag(rho)`. Whitening removes this scaling exactly.
It does not follow that arbitrary diagonal congruence always improves eigenvalue
ratios: our explicit seeded 7x3 counterexample has Hessian condition number
**2.06684 before** row preconditioning and **3.04057 after** it. The test forms
this example by first scaling balanced rows by the inverse square root of the
unscaled Hessian diagonal and verifies the congruence identity independently.

In a different adversary with common balanced row amplitudes 1e-3..1e3,
unpreconditioned GD stagnates and falls back, while float64 PGD reaches the
1e-6 gap target in 13 iterations. Both facts are consistent with (5)-(6).
This canonical amplitude imbalance is distinct from inverse raw gauge imbalance,
which should disappear during canonicalization.

## 4. Normalization, four methods and warm state

All methods first remove the Euclidean vertical representative in float64:

\[
\beta=W^+L(A),\quad A_h=A-L^*\beta,\quad s=\|A_h\|_F.
\tag{7}
\]

On a nonzero cotangent, solve the equivalent normalized objective with A_h/s.
GD keeps the original row coordinates. PGD, L-BFGS and Newton use the whitened
constraint rows `U_i/sqrt(w_i), D_i/sqrt(w_i)`. This is multiplier preconditioning,
not shared K or reweighting of the primal objective. The corresponding original
multiplier is `lambda=beta+s W^{-1/2} z` (GD uses the identity in place of
W^{-1/2}). Inactive zero equality rows, if supplied to this algebraic solver,
have zero coordinate scale; that does not define an H metric on zero raw weights.

An exactly zero centered cotangent returns the exact zero pair. Near-zero or
nearly vertical cotangents are normalized, not silently discarded. Cancellation
at the scale of input rounding is separately reported and can trigger fallback.
This preprocessing improves relative stopping on small objectives, but cannot
restore information already lost by storing a nearly vertical objective in
float32.

| Method | Direction and numerical choices |
| --- | --- |
| GD | Negative ordinary gradient; backtracking, with the previous accepted step doubled as the next proposal |
| PGD | Same algorithm in row-whitened coordinates, equivalent to W^{-1} gradient scaling |
| L-BFGS | Two-loop limited-memory inverse-Hessian recursion, 10 pairs, row-whitened coordinates; skip curvature pairs unless s^T y is sufficiently positive |
| Newton-CG | Row-whitened, matrix-free CG on `(Hess + delta I)d=-g`; delta=1e-6 times the local upper-curvature scale, at most min(30,m) products, residual-dependent truncation, backtracking |

The damping changes the linear system used to propose a search direction,
not the nuclear-norm objective or primal LMO. It handles singular or poorly
conditioned multiplier curvature without forming a dense inverse. If CG does
not supply descent, a scaled gradient direction is used. There is no claim
that this is an undamped exact Newton step or a full trust-region method.
The limited-memory method follows the standard two-loop approach described by
[Nocedal's L-BFGS resources](https://users.iems.northwestern.edu/~nocedal/lbfgs.html);
the matrix-free truncated-CG approach is in the family studied by
[Steihaug](https://repository.rice.edu/items/3ffddc5f-871b-4aa1-9662-dcbbeb7e0d72).

Line search uses an Armijo decrease test with an explicit floating-point
allowance. When the predicted decrease is below that allowance, the gradient
norm must also improve. This is a numerical safeguard, not a proof of exact
Armijo inequalities in rounded arithmetic. Acceptance of a returned optimizer
step still requires the primal-dual certificate below. A failed search or
stagnating certified gap triggers fallback rather than declaring convergence.

Warm state is **lambda in the original multiplier convention**. At the next
step, after U,D,A and their normalization have changed, transform that previous
lambda into the new internal coordinates. Keeping the old scaled z unchanged
would be wrong. L-BFGS curvature history is reset on each new objective;
only lambda is warm-started. The reported short-budget rollouts carry the
previous approximate lambda, not a reference-oracle reset at every frame.

## 5. Feasible recovery and adaptive certification

A smooth dual iterate provides separate residual polars P, which need not
yet be horizontal. To obtain a feasible lower bound, form in float64

\[
Q=\Pi_{\mathcal H}P,\qquad
\widehat P=Q/\max(1,\|Q_U\|_2,\|Q_D\|_2).
\tag{8}
\]

This is a recovery step, not an assertion that post-polar projection is an
exact LMO. Its adequacy is measured by the gap. If the resulting objective is
negative, central symmetry allows a common sign flip to obtain a nonnegative
feasible lower bound. For the returned original-coordinate lambda compute

\[
\mathrm{LB}=\langle A,\widehat P\rangle,\quad
\mathrm{UB}=\phi(\lambda),\quad
g_{\rm rel}=\frac{\max(0,\mathrm{UB}-\mathrm{LB})}
{\max(|\mathrm{UB}|,|\mathrm{LB}|,\mathrm{tiny}_{64})}.
\tag{9}
\]

The zero/zero case has zero gap. There is no max(1,value) denominator that would
turn a small nonzero objective into an automatic success. Every result reports:

- maximum absolute horizontal residual and its version divided by sqrt(w_i);
- both spectral norms and spectral excess over one;
- primal and dual objectives, signed absolute gap and signed normalized gap;
- normalized nonnegative gap, convergence status, smooth iterations, fallback
  occurrence/reason, minimum accepted residual rcond, SVD/polar/HVP counts and
  wall time, plus the secondary-selection contract.

The smooth polar's horizontal residual before recovery is retained in the
per-iteration history. Output feasibility alone cannot measure stationarity
since (8) enforces it; the gap supplies the optimality test.

Stopping requires g_rel <= the requested tolerance, normalized horizontal
residual <=1e-10, spectral excess <=1e-12, and signed normalized gap >=-1e-10.
The singular-value diagnostic must also pass before accepting a smooth result.
Main experiments use 1e-6 for float64 smooth work and 3e-5 for float32 work.
Matched-tolerance supplemental runs use 1e-6 in both precisions. Iteration
limits are safety budgets; hitting one is not successful convergence.

This is an exact-arithmetic primal-dual certificate **evaluated in floating
point**, not an interval-arithmetic proof. Signed gaps and feasibility defects
are exposed to reveal rounding or failed assumptions; a materially negative
gap is not quietly called convergence. Feasible recovery, certificate SVDs,
returned pairs and fallback reference solves use float64 even when smooth
optimization uses float32. The labels below explicitly refer to the smooth
work dtype. They do not certify an all-float32 production implementation or
the result after casting the returned direction back to a low precision.

## 6. Smoothness diagnostics and explicit fallback

Each smooth residual evaluation records
`rcond=min_j sigma_min(B_j)/sigma_max(B_j)`. Operational guards are 1e-8 in
float64 and 1e-4 in float32. These are derivative-conditioning diagnostics,
not mathematical rank decisions: **all singular values remain in the nuclear
norm and polar**, and no singular vector is removed. The thresholds are
declared numerical research settings, not universal precision guarantees.

An unusable trial is rejected and backtracked; an unusable current iterate,
nonfinite gradient, absence of descent, line-search failure, gap stagnation,
cancellation-dominated cotangent, or exhausted budget invokes the reference
when fallback is enabled. `fallback=True` and a reason are returned. With
fallback disabled for budget experiments, the same event returns a feasible
approximation with an explicit unsuccessful status unless already certified.

The existing `experiments.horizontal_spectral.solve_lmo` is the reference,
unchanged. The wrapper supplies an equivalent centered cotangent in float64,
uses tolerance 1e-11 and rho=1/sqrt(n), then independently re-certifies the
returned pair against the original A and translated lambda. Its primary
convergence status is checked rather than assumed. The normalized A has
typical singular values of order 1/sqrt(n); scaling rho accordingly avoids
unnecessary ADMM iterations at larger widths. This tunes a solver penalty,
not the variational problem. Initial large CPU trials at rho=1 were stopped
and replaced by the documented penalty setting; the JSON contains the final
setting's runs. The runner now checkpoints completed cases to disk.

For a nonunique optimal face, generic ADMM does not prove the secondary
minimum-Frobenius selection. Fallback results are labeled
`primary_only_unless_uniqueness_established`. Full-rank smooth stationarity
gives a unique primal in the exact limit, so the distinction disappears there;
the finite-tolerance step is still approximate. Analytically known deficient
faces can use the earlier Dykstra reference, but no numerical rank guess is
introduced to manufacture that face in this solver.

### Why even a reference value oracle is not an exact direction oracle

For the regular 3x2 example with U=D, both residuals diag(1,t) padded by one
zero row, t=1e-12, the analytic LMO has two unit active singular values on
each side. The smooth solver detects poor conditioning and falls back. ADMM
returns a feasible pair with relative gap **1.00009e-12**, yet its relative
direction error versus the exact solution is **.707107**. The relative finite-
step delta-X error is also .707107 at eta=.01. Comparing only against the
ADMM direction would incorrectly report zero error here; the supplemental
record explicitly compares against the analytic solution.

This is a mathematical limitation of a value-only tolerance near a rank
boundary, not a hidden hard threshold in the solver. A practical contract
requiring exact direction recovery in this regime needs more than fallback
to a small-gap oracle. The prototype deliberately does not claim otherwise.

## 7. Experimental design and reproducibility

The stored run used Python 3.9.6, PyTorch 2.8.0, macOS arm64, and one CPU
thread. The repository declares Python >=3.10; these are source-tree tests
under the available interpreter, not package-installation validation on its
declared supported Python versions. There are no GPU measurements.

Two sequences each contain nine frames, hence eight warm-start transitions:

- **EMA proxy, 96x32:** seed 2001, initially unit balanced rows and Gaussian
  objectives; factor drift .001/sqrt(n) followed by balancing; objective-base
  drift .001 and EMA coefficient .95.
- **Fixed-batch SwiGLU, 48x16:** seed 2011, batch size 64, fixed random inputs,
  targets and SiLU gates; squared-error loss and actual factor gradients;
  canonical EMA coefficient .95, initialized with the first gradient. Each
  frame follows a reference factor step of .001 and rebalancing. This is a
  small controlled training trajectory, not a language-model training run.

Both precisions use the same underlying float64 frames, cast to the smooth
work dtype. The reference solves the represented inputs in float64. Warm
runs reuse the previous **returned** multiplier, not the current frame's
oracle. Budget runs also propagate their own approximate state; only their
initial frame uses an oracle multiplier. They do not reset to the oracle at
each transition. L-BFGS histories are reset between frames, so only lambda
is warm-started. This isolates the requested multiplier warm start.

The main float64 target is 1e-6; the main float32 target is 3e-5. Supplemental
float32 runs use the same 1e-6 target as float64. Shape studies use one more
slow transition at 256x96, 768x256 and 1536x512. These have SwiGLU-like tall
aspect ratios and reach a small block's dimensions; they are not evidence
for performance at modern large-language-model widths. Timings are individual
CPU observations, including feasible recovery and certification, with no
confidence intervals. First-frame initialization and data generation are
outside the per-solve timings.

Run from the repository root:

```bash
python3 -m pytest -q
python3 -m experiments.dual_solver_study --large
python3 -m experiments.dual_solver_study --supplement
```

The first study command writes the main JSON, checkpointing completed cases;
the second adds the matched-tolerance and analytic checks without replacing
the original timings. `--output` can select a separate artifact. Seeds, shapes,
dtypes, per-solve metrics, fallbacks, evaluation counts and timings are in the
[raw results](../experiments/dual_solver_results.json).

## 8. Cold versus warm iterations

Entries are mean accepted smooth iterations over eight transitions, written
**cold -> warm**. Fallback counts are out of eight. Reference iterations are
additional to the smooth counts and are included in total time/SVD counts.

| Sequence / smooth dtype / target | GD | Row-PGD | L-BFGS | Newton-CG |
|---|---:|---:|---:|---:|
| EMA / float64 / 1e-6 | 13 -> 6.12 | 13 -> 6.12 | 5 -> 2.75 | 3 -> 1.75 |
| SwiGLU / float64 / 1e-6 | 63.62 -> 92.38 | 64.88 -> 93.12 | 14 -> 7.12 | 5 -> 2 |
| EMA / float32 / 3e-5 | 9.75 -> 1 | 9.38 -> 1 | 4.12 -> 1 | 2 -> 1 |
| SwiGLU / float32 / 3e-5 | 35.12 -> 3 | 36 -> 3 | 12 -> 2.25 | 4 -> 1 |

The float64 SwiGLU warm GD and PGD runs each needed fallback on **7/8**
transitions, versus none when cold. Float32 SwiGLU cold GD and PGD needed
fallback on 4/8 and 6/8 transitions, respectively; their warm runs needed
none at the looser target. All other entries in this table used no fallback.
Thus warm-starting alone does not make a poorly scaled first-order line search
reliable. The simple gradient step-size rule can be particularly ineffective
near an already good solution; these observations concern the implemented
algorithms, not a theorem that every gradient method must behave this way.

Float64 warm mean wall times were 5.78, 5.95, 2.43 and 2.12 ms on the EMA
sequence, in table order. On the toy SwiGLU sequence they were 50.81, 49.35,
1.81 and 1.30 ms, including fallback. Mean reference times were 23.88 and
28.90 ms, respectively. Newton used a mean 4.25 cached-SVD HVPs per EMA
transition and 12 per toy transition. Its minimum accepted residual rcond
over these warm float64 runs was .2982 and .04696, respectively: these
successful short solves were safely inside the tested smooth regime.

### Matched precision targets

Float32 at target 1e-6 gives the following warm results. Compare these with
the float64 warm columns above, not with the looser float32 target.

| Sequence | GD mean iterations / fallbacks | Row-PGD | L-BFGS | Newton-CG |
|---|---:|---:|---:|---:|
| EMA | 7 / 0 | 7.5 / 0 | 3.5 / 0 | 2.5 / 1 |
| SwiGLU | 66.12 / 5 | 51.12 / 3 | 7.12 / 0 | 2 / 0 |

All final returned results meet the requested target, including fallback
results. Float32's lower iteration counts in the main table therefore cannot
be attributed to precision alone: its requested accuracy was also lower.
The matched EMA Newton run's one fallback shows that a tight target can reach
the numerical limits of this smooth float32 procedure even on benign data.

## 9. Are one, two, four or eight warm iterations sufficient?

The table counts certified successes out of eight transitions with fallback
**disabled**. Columns give budgets 1 / 2 / 4 / 8; the solver may stop earlier
when its gap passes, so these are caps rather than forced iteration counts.

| Sequence / smooth dtype / target | GD | Row-PGD | L-BFGS | Newton-CG |
|---|---|---|---|---|
| EMA / float64 / 1e-6 | 0 / 0 / 0 / 8 | 0 / 0 / 0 / 8 | 0 / 2 / 8 / 8 | 2 / 8 / 8 / 8 |
| SwiGLU / float64 / 1e-6 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 8 | 0 / 8 / 8 / 8 |
| EMA / float32 / 3e-5 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 |
| SwiGLU / float32 / 3e-5 | 0 / 1 / 8 / 8 | 0 / 1 / 7 / 8 | 0 / 6 / 8 / 8 | 8 / 8 / 8 / 8 |

For float64 Newton, the worst one-iteration normalized gaps were 1.7e-6
(EMA) and 3.7e-6 (SwiGLU); at two iterations they were 8.9e-7 and 1.1e-8.
For L-BFGS, four iterations sufficed on EMA and eight on SwiGLU. GD still
had a 5.9e-5 worst gap on SwiGLU at eight iterations; PGD had 3.4e-5.

Two warm Newton iterations are a useful **observed initial budget** for these
slow sequences at 1e-6. They are not a stopping rule: keep the certificate,
continue if it fails, and report reference fallback when needed. Abrupt
momentum changes, poor residual conditioning and nonunique multipliers
invalidate an unconditional fixed-budget claim. No experiment here establishes
how frequently these events occur in large-scale stochastic training.

## 10. Runtime and decomposition counts at larger shapes

The table gives float64 warm solves at target 1e-6. ADMM is the float64
reference at tolerance 1e-11, so the runtime comparison is against a more
accurate value oracle, not an equal-tolerance algorithmic speed ratio.
Counts include certificate SVDs and rejected line-search trials; a batched
decomposition of the two sides counts as **two matrix SVDs**.

| Shape | Solver | Iterations | Matrix SVDs | Seconds |
|---|---|---:|---:|---:|
| 256x96 | ADMM reference | 80 | 196 | .2622 |
| 256x96 | GD | 9 | 74 | .0779 |
| 256x96 | Row-PGD | 9 | 74 | .0778 |
| 256x96 | L-BFGS | 2 | 18 | .0174 |
| 256x96 | Newton-CG | 1 | 12 | .0123 |
| 768x256 | ADMM reference | 70 | 172 | 1.6656 |
| 768x256 | GD | 7 | 58 | .4389 |
| 768x256 | Row-PGD | 7 | 58 | .4391 |
| 768x256 | L-BFGS | 2 | 18 | .1277 |
| 768x256 | Newton-CG | 2 | 18 | .1375 |
| 1536x512 | ADMM reference | 70 | 172 | 7.2630 |
| 1536x512 | GD | 6 | 50 | 1.6330 |
| 1536x512 | Row-PGD | 6 | 50 | 1.6301 |
| 1536x512 | L-BFGS | 2 | 18 | .5511 |
| 1536x512 | Newton-CG | 1 | 12 | .3901 |

At 1536x512 float64, cold -> warm Newton took 3 -> 1 iterations,
24 -> 12 SVDs, 8 -> 4 polar evaluations, 4 -> 1 HVPs and .8141 -> .3901 s.
L-BFGS took 3 -> 2 iterations, 24 -> 18 SVDs, 8 -> 6 polars and
.7361 -> .5511 s. GD took 45 -> 6 iterations, 362 -> 50 SVDs,
178 -> 22 polars and 12.3580 -> 1.6330 s. PGD had the same counts and
12.2176 -> 1.6301 s. Newton's warm final gap was 6.26e-7; the reference's
gap was 3.32e-14. The tall random rows are nearly uniform, so row scaling
does little in these particular shape cases.

At 1536x512 with float32 smooth work and target 3e-5, the previous multiplier
already passed the next step's gap test (2.69e-5). All four methods therefore
used zero optimization iterations, but still **two polar evaluations and
six matrix SVDs** including certification, taking about .163 s. Cold Newton
took two iterations, 18 SVDs, six polars and two HVPs in .492 s; cold L-BFGS
took .474 s with the same SVD/polar counts. Cold GD/PGD needed 18 iterations,
146 SVDs and about 4.00 s. At smaller shapes warm float32 Newton took one
iteration: .0099 s at 256x96 and .0727 s at 768x256.

Certification is a substantial part of the cost: every checked iterate adds
four matrix SVDs beyond the residual pair's SVDs. These prototypes prioritize
an independent, observable certificate over minimizing decomposition count.
Sharing factors for a future implementation would need a separate validation
of the certificate and precision contract. No GPU kernel optimization or
production integration follows from these CPU timings.

## 11. Direction accuracy and finite neuron updates

For a feasible descent direction P and factor step eta,

\[
\Delta X_i(P)=-\eta(P_{D,i}U_i^T+D_iP_{U,i}^T)
                  +\eta^2 P_{D,i}P_{U,i}^T.
\tag{10}
\]

The last term is retained. Measuring only the tangent map would miss the
finite-step factorization error. The implementation uses inner products of
outer products to evaluate these errors without allocating m x n x n arrays;
a small test compares with explicitly constructed matrices. The following
are maximum **relative** errors against the reference over eight warm
transitions at eta=.01. Objective error is divided by the reference value;
direction and delta-X errors by the respective reference Frobenius norms.

| Sequence / smooth dtype | Solver | Objective loss | Direction error | Finite delta-X error |
|---|---|---:|---:|---:|
| EMA / float64 | GD | 9.44e-7 | 1.10e-6 | 1.10e-6 |
| EMA / float64 | Row-PGD | 9.68e-7 | 1.12e-6 | 1.12e-6 |
| EMA / float64 | L-BFGS | 9.78e-7 | 1.18e-6 | 1.18e-6 |
| EMA / float64 | Newton-CG | 8.87e-7 | 1.08e-6 | 1.08e-6 |
| SwiGLU / float64 | GD, including fallback | 9.94e-7 | 1.30e-6 | 1.29e-6 |
| SwiGLU / float64 | Row-PGD, including fallback | 9.98e-7 | 1.30e-6 | 1.30e-6 |
| SwiGLU / float64 | L-BFGS | 7.52e-7 | 1.25e-6 | 1.20e-6 |
| SwiGLU / float64 | Newton-CG | 1.06e-8 | 1.79e-8 | 1.68e-8 |
| EMA / float32 | L-BFGS | 2.15e-5 | 2.41e-5 | 2.41e-5 |
| EMA / float32 | Newton-CG | 1.73e-6 | 2.05e-6 | 2.05e-6 |
| SwiGLU / float32 | L-BFGS | 2.74e-5 | 3.90e-5 | 3.71e-5 |
| SwiGLU / float32 | Newton-CG | 3.89e-6 | 6.15e-6 | 5.80e-6 |

The omitted float32 GD/PGD entries and every individual residual are in the
JSON. Across all stored cases, the largest normalized output horizontal
residual was 2.37e-16; float64 Newton's largest absolute horizontal residual
was 1.11e-16 on EMA and 3.33e-16 on SwiGLU. These tiny feasibility errors
reflect **float64 recovery**, not the accuracy of raw float32 polar stationarity.
All stored signed gaps were nonnegative; spectral excess was zero under the
computed radial certificate. These are numerical observations, not exact
floating-point feasibility guarantees independent of SVD error.

### When a gap does bound direction error

Suppose an exact dual minimizer has both residuals full column rank, with
smallest singular values >=alpha>0. Its unique primal pair is P*. For every
feasible P,

\[
\|P-P^*\|_F^2\le
\frac{2\big(v^*-\langle A,P\rangle\big)}{\alpha}
\le\frac{2\,\text{certified absolute gap}}{\alpha}.
\tag{11}
\]

To prove this, write each residual as B=P*S with S>=alpha I. The contraction
P implies I-sym(P*^T P)>=0. Consequently
`<B,P*-P> >= alpha (n-<P*,P>) >= alpha ||P*-P||_F^2/2`, since
`||P||_F^2<=n` and `||P*||_F^2=n`. Sum both sides and use horizontality to
cancel the dual multiplier. Any dual upper bound is >=v*, giving (11).
The assumption concerns the **optimal** residual, not merely a well-conditioned
current iterate. There is no uniform such bound as alpha tends to zero, in
agreement with the analytic example in Section 6.

If every balanced row has norm <=rho and P,R are feasible spectral pairs,
their row norms are <=1, and expansion of (10) gives

\[
\|\Delta X(P)-\Delta X(R)\|_F
\le\sqrt{2}(\eta\rho+\eta^2)\|P-R\|_F.
\tag{12}
\]

Here the left norm stacks all neuron matrices. Bound the two linear outer
products using rho and write the quadratic difference as
`(P_D-R_D)P_U^T + R_D(P_U-R_U)^T`; then use the row contraction bounds and
Cauchy-Schwarz on the pair. This controls **absolute** finite-step error.
There is no uniform relative bound if the reference delta-X vanishes or
nearly cancels. The JSON uses relative errors only where that comparison is
meaningful, and the analytic rank-boundary case exposes their possible size.

## 12. Adversarial outcomes and what they establish

| Case | Observed outcome | Consequence |
|---|---|---|
| Fractional 3x2 optimal face | Newton fell back after 13 float64 steps (gap stagnation), or 11 float32 steps (line-search failure); final gap about 5.68e-14 | Smooth polars cannot represent every deficient optimal face; fallback is observable. |
| 3x2 residual with singular values 1 and 1e-12 | Immediate conditioning fallback in both dtypes; gap about 1e-12 but analytic relative direction error .707107 | High value accuracy does not imply high direction accuracy. No positive singular value was truncated by the smooth solver. |
| Zero objective, 9x3 | Zero iterations, no fallback, exactly zero returned pair | The zero-cotangent branch avoids an arbitrary polar completion. |
| Nearly vertical, 9x3, A=L*lambda+1e-8 noise | Newton 5 float64 / 4 float32 iterations; gaps 4.03e-8 / 8.34e-7 | Centering removes a large irrelevant vertical component; tiny cotangent values are still tested relatively. |
| Nonunique lambda, 2x1 | Warm lambda=.8 times the all-ones vector accepted at iteration zero; same primal as other minimizing multipliers | Multiplier uniqueness is unnecessary for an accurate primal; the Hessian can be singular inside the smooth regime. |
| Balanced row amplitudes 1e-3 through 1e3, 9x3 | GD stagnated and fell back; PGD used 13 float64 / 9 float32 steps; Newton used four in both | Row whitening removes scale-induced difficulty here, but is not a universal condition-number theorem. |
| Rapidly replaced momentum, 9x3 | Newton needed five steps in both dtypes | The two-step slow-sequence observation does not survive arbitrary objective changes. |
| Extreme raw positive gauges | Separate tests use factors scaled by 1e-120 through 1e120 in float64 and 1e-12 through 1e12 in float32 | Stable regular canonicalization preserves the norm and solver input to numerical tolerance; this does not validate production clamps or zero rows. |

Near-vertical experiments in float32 are problems for the **rounded** input
covectors: rounding a large vertical component plus tiny noise can change
the small intrinsic objective. Float64 centering cannot recover information
already lost when the input was represented in float32. Cancellation detection
reports an unreliable smooth solve and delegates to the reference; it does
not turn that reference into a remedy for lost input information.

Neither a multiplier warm start nor row preconditioning resolves all deficient
faces. A multiplier may move within a nonunique minimizing set with no change
in the primal. Conversely, a small multiplier change can cross a residual
rank boundary and change a selected primal abruptly. A convergence monitor
based solely on lambda displacement would therefore be inappropriate.

## 13. Verification and decision supported by this study

The full mathematical suite passed before the changes (125 cases) and after
them (150 cases). New isolated tests cover:

- tangent/cotangent norm identities, the raw gauge pullback, the distinction
  from an infimum-over-lifts norm, and nonsmooth/non-strictly-convex examples;
- polar HVPs against independent derivatives, PSD/symmetry, repeated positive
  singular values, and the row-preconditioner counterexample;
- all four methods in both smooth dtypes, certificates, explicit fallback,
  exact zero, nearly vertical inputs and nonunique multipliers;
- warm-start work reduction, deliberately unsuccessful limited budgets,
  near-rank value/direction separation, finite delta-X measurements and exact
  accounting of SVD calls, including reference diagnostics;
- the incorrect difference-of-nuclear-norms expression in the prompt.

The evidence supports keeping **warm-started, row-scaled Newton-CG with a
gap certificate and explicit generic fallback** as the leading research
candidate, with L-BFGS a competitive simpler alternative. The row scaling
has a precise upper-curvature interpretation, not a universal promise of a
smaller condition number. Start with a small iteration budget if useful,
but let the measured gap and conditioning determine acceptance.

Remaining decisions before production review are the required direction
contract near rank loss, numerical implementation of the secondary minimum-
Frobenius optimal-face selection when uniqueness fails, precision after
casting returned directions, and behavior on longer stochastic training
trajectories and larger hardware. This study supplies neither a fixed two-step
guarantee nor an exact selected direction at every rank. The intrinsic core
norm and its exact variational problem do not depend on these numerical
solver choices. Production `qnormuon/` remains unchanged.
