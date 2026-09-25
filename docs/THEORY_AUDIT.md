# QNorMuon mathematical and implementation audit

Date: 2026-09-24. Scope: the complete `AGENTS.md`, `CODEX_FIRST_TASK.md`,
`QNORMUON_THEORY.md`, `qnormuon/core.py`, package exports, all five original tests,
and both existing experiments. The derivations below are independent checks of
the supplied specification, not an endorsement of its unqualified conclusions.

**The central construction is sound on its regular domain.** Positive, nonzero
paired row norms give a unique balanced representative; covariant gradients give
invariant state; a consistently chosen canonical update gives an equivariant lift.
Full-column-rank momenta give the stated log-determinant gradient and convexity.
Both momenta being full spark, with `0 < n < m`, suffices for a finite balancing
solution unique modulo global scale.

**The implementation does not enforce that domain.** Absolute norm clamping
breaks the exact gauge laws, deficient-rank SVD completions do not define the
log-determinant leverage problem, and a finite online balancing step does not
guarantee uniform leverage. Native bfloat16 SVD is unavailable on the audited CPU.
The connection between Theorem 4 and canonical momentum also needs explicit
notation to avoid applying the metric twice.

Only this report and `tests/test_theory_audit.py` were added. The optimizer,
original theory, exact SVD reference, and original tests were left unchanged.
Proposed corrected statements appear here for review before a coordinated theory
and implementation change. Counterexample tests assert the demonstrated failure;
they are not silently skipped or marked as expected failures.

## 1. Verdicts

The standing assumptions already explicitly stated before a theorem count as part
of its statement. In particular, Section 2 assumes positive row norms.

| Claim | Classification | Qualification or missing statement |
| --- | --- | --- |
| Lemma 1 | Correct as stated | For the displayed bias-free block, with the gate unchanged. The subsequent identification with *all* functional equivalence classes is too strong. |
| Theorem 1 | Correct as stated | On `r_i,s_i > 0`. “Positive” describes the diagonal gauge, not the entries of the representative. Clamped norms do not implement this theorem near zero. |
| Theorem 2 | Correct under additional assumptions | A differentiable gauge-invariant loss, matching canonical initial state, identical data/randomness and state transitions. The asserted gradient law is not true for arbitrary losses. |
| Theorem 3 | Correct under additional assumptions | Theorem 2's conditions, an unclamped covariant metric, a consistently chosen deterministic canonical map, and exact arithmetic. Finite precision requires conditioning qualifications. |
| Theorem 4 | Correct as stated | `H > 0`, `m >= n`, and full column rank as stated. Its identification with the EMA algorithm is **incomplete** until raw and canonical momentum are distinguished. |
| Theorem 5 | Correct as stated | Positive scalar norm-local metric on `r,s > 0`, with both functional identities required everywhere. Duality is a design axiom, not a symmetry of the entire gated architecture. |
| Theorem 6 | Correct under additional assumptions | Both momenta must have full column rank. At deficient rank, `log det` is not finite and its displayed derivative does not exist. |
| Theorem 7 | Correct under additional assumptions | Full column rank for the general convexity claim; the stated full-spark clause is sufficient for existence/uniqueness. “Interior” must mean relative interior in `sum ell = 2n`. The proof of existence/uniqueness is incomplete in the source; supplied below. |

None of the numbered results is refuted on the regular domain with these
conditions. Unrestricted extensions to zero rows, deficient rank, arbitrary
losses, or unconditional online convergence are false, with explicit examples
below. Section 15 therefore describes simultaneous properties under assumptions,
not unconditional properties of every optimizer step.

## 2. Independent derivations

### Lemma 1: dimensions and functional invariance

`U,D,G_U,G_D` have shape `[m,n]`; the stored down weight and its gradient have
shape `[n,m]`. For `a = SiLU(W_g z)`, diagonality gives

\[
(D')^T[a\odot U'z]
=D^TC^{-1}[a\odot CUz]=D^T[a\odot Uz].
\]

Thus `down -> down / c[None,:]`, while `grad_down -> grad_down * c[None,:]`.
The implementation's transposes agree throughout. No homogeneity of SiLU is
needed because `W_g` stays fixed. An up bias, if introduced, would also need its
matching transformation; such biases are outside this implementation.

This proof actually works for any invertible real diagonal `C`. Consequently the
positive-diagonal quotient in Section 1 removes this specified symmetry but is
not the full space of distinct functions. For example, simultaneously flipping
the signs of a nonzero `u_i,d_i` preserves the function and is not in its positive
orbit. Degenerate gates provide further identifications. Corrected wording:
“the parameter space modulo the positive diagonal gauge,” rather than “the real
space of models” or all functional equivalence classes. This distinction does not
invalidate the proposed positive-gauge optimizer.

### Theorem 1: balanced section

For `r,s > 0`, a balancing scale satisfies `c r = s/c`, hence uniquely
`c = sqrt(s/r)`. Both balanced norms equal `sqrt(rs)`. Since
`r' = cr`, `s' = s/c`, one has `H' = C²H` and

\[
(H')^{-1/2}U'=H^{-1/2}U,\qquad
(H')^{1/2}D'=H^{1/2}D.
\]

Functional equivalence follows directly from Lemma 1. With one norm zero and
the other nonzero, no finite positive scale balances them. With both zero, the
row pair is already fixed by every scale: the representative row pair is unique
as a point, but the balancing gauge is not, and `r/s` is undefined. These strata
must not be confused with the regular section.

### Theorem 2: covectors, canonical gradients, and state

If `L(CU,C^{-1}D)=L(U,D)`, differentiate with respect to arbitrary perturbations:

\[
\langle G_U',C\,dU\rangle+\langle G_D',C^{-1}dD\rangle
=\langle G_U,dU\rangle+\langle G_D,dD\rangle.
\]

It follows that `G_U'=C^{-1}G_U`, `G_D'=CG_D`. Thus
`sqrt(H') G_U' = sqrt(H) G_U` and similarly for the down side. The actual
SwiGLU autograd tests verify both laws, including the unchanged gate gradient.
Differentiation along each gauge direction also gives the stronger rowwise check

\[
\langle G_U[i],U[i]\rangle-\langle G_D[i],D[i]\rangle=0.
\]

State invariance follows by induction from equal initial canonical state and
equal invariant inputs to each transition. This includes `x` and balancing
momentum `q`, not just the two gradient EMAs. Raw momentum accumulated before a
reset cannot simply be called canonical momentum.

The canonical gradients are covectors expressed in the balanced frame; they are
not obtained by blindly differentiating `canonical_pair` while ignoring its
parameter dependence. For an invariant loss they also agree with Euclidean
gradients evaluated at the balanced representative. The frame map for update
vectors has the opposite scaling. The code correctly distinguishes these maps.

Counterexample to omitting the loss assumption: for
`L=(||U||_F²+||D||_F²)/2`, `G_U'=CU`, not `C^{-1}U`. The corresponding canonical
gradients differ. Raw parameter-dependent regularizers and clipping policies
therefore need their own analysis.

### Theorem 3: equivariant lift and approximate polar maps

Let `P_U,P_D` be any identical canonical outputs in the two runs, including
outputs using shared invariant `x`. Then

\[
\Delta U'=C H^{1/2}P_U=C\Delta U,\qquad
\Delta D'=C^{-1}H^{-1/2}P_D=C^{-1}\Delta D.
\]

Subtracting the same learning-rate multiple preserves the gauge relation of the
weights. If the optional balanced reset follows, both runs instead meet at the
same balanced representative. Their functions still agree; the original fixed
`C` need no longer relate the stored weights after this reset.

No orthogonality assumption enters this proof. A fixed number of Newton–Schulz
steps on identical canonical inputs is therefore sufficient for equivariance,
but not for exact Stiefel constraints or Theorem 6's leverage gradient. Stopping
criteria, warm starts, precision and randomness must also depend consistently
on canonical quantities. Numerically the inputs are only approximately equal;
ill conditioning, rank changes or different iteration branches can amplify the
difference. “Literally identical” in Section 4 is an exact-arithmetic statement.

The new test substitutes two Newton–Schulz iterations only inside a scoped test,
compares them with SVD, and exhibits gauge error `3.736e-16` alongside Stiefel
defect `9.926e-1`. It adds no approximate production backend.

### Theorem 4: weighted spectral optimization

Write `M_raw` for the covector in the original coordinates, and set
`A=H^{1/2}M_raw`. Substituting `Delta=H^{1/2}Y` gives

\[
\max_{Y^TY=I}\langle A,Y\rangle=\|A\|_*.
\]

If the thin SVD is `A=L Sigma R^T` with all singular values positive, each
`l_j^T Y r_j <= 1`. Achieving the nuclear-norm bound forces `Y r_j=l_j` for every
column, so the unique maximizer is `Y=LR^T`. Consequently
`Delta*=H^{1/2} polar(H^{1/2}M_raw)` and
`Delta*^T H^{-1} Delta*=I`. For the down side replace `H` by `H^{-1}`.

**Notation correction, not an implementation change:** the algorithm stores
`M_can`, already in the canonical frame. For QMuon (`K=I`) set

\[
M_{\mathrm{raw},U,t}=H_t^{-1/2}M_{\mathrm{can},U,t},\qquad
M_{\mathrm{raw},D,t}=H_t^{1/2}M_{\mathrm{can},D,t}.
\]

These are effective current covectors, not generally raw-gradient EMAs, since
`H_t` changes with time. The correct up update is
`H^{1/2}polar(M_can)`. Interpreting Theorem 4's `M` as `M_can` erroneously inserts
an additional `H^{1/2}` inside the polar. The numerical test gives strict
objective losses `0.5542` (up) and `0.8270` (down) for this misinterpretation.

For QNorMuon (`K != I`), the canonical spectral objective uses
`K^{1/2}M_can`, equivalently the effective raw covector `K^{1/2}M_raw`.
The spectral constraint is unchanged, but this is not in general the maximizer
for the unweighted raw covector. No extra `K` factor appears after polarizing.

At deficient rank the optimum value remains the nuclear norm, but the maximizer
is nonunique. Example: `A=[e_1,0]` in `R^{3x2}` has optimizers
`[e_1,e_2]` and `[e_1,-e_2]`. A thin SVD can select an orthonormal completion;
this differs from the uniquely defined rank-restricted polar partial isometry
that is zero on the kernel. The latter has `P^TP` a rank-`r` projector rather
than `I_n`. One cannot simultaneously claim this rank restriction, full Stiefel
columns, and uniqueness at deficient rank.

### Theorem 5: uniqueness within the specified metric class

Let `p=rs`. Gauge covariance at `c=sqrt(s/r)` gives

\[
a(r,s)=(r/s)\,a(\sqrt p,\sqrt p)=(r/s)\psi(p).
\]

Swap duality implies `psi(p)=1/psi(p)`. Positivity forces `psi(p)=1`, proving
`a(r,s)=r/s`. No differentiability or continuity assumption is needed. This
does not establish uniqueness among metrics depending on row directions, other
neurons, gradients, or history. Norm-locality and exact duality are substantive
restrictions. The tests check covariance, duality and value one at balance;
the functional uniqueness argument is analytical, not inferred from samples.

### Theorem 6: derivative and Hessian

For full-column-rank `M`, positive finite `K` makes `A=M^TKM` positive definite.
Differentiating its log determinant gives

\[
\partial_i\log\det A=e^{x_i}m_i^TA^{-1}m_i=\ell_i.
\]

Let `B=K^{1/2}M`, `P=B(B^TB)^{-1/2}`, and `Q=PP^T`. Differentiating again gives

\[
\nabla^2\log\det A=\operatorname{Diag}(\ell)-Q\odot Q.
\]

Thus `grad Phi = ell_U+ell_D-(2n/m)1`, and the two Hessian expressions add.
`Q²=Q` implies the Hessian annihilates `1`; the covariance proof below shows it
is positive semidefinite. Tests compare SVD leverage with independent autograd
and central finite differences of `slogdet`, then compare this analytic Hessian
with second-order autograd. The implementation uses `no_grad` helpers, so the
independent differentiable objective is defined in the test.

At rank deficiency `det A=0` for every finite `x`. `Phi=-infinity` is not a finite
differentiable objective. Leverages of an arbitrary SVD completion cannot equal
its derivative. A pseudo-determinant or a ridge would define a different problem
and requires a new derivation; neither has been silently substituted here.

### Theorem 7: convexity, existence and uniqueness

Let `B_M` be the size-`n` subsets with nonzero maximal minors. Cauchy–Binet gives

\[
f_M(x)=\log\sum_{S\in B_M}a_S e^{1_S^Tx},\qquad a_S=\det(M_S)^2>0.
\]

With full column rank this is a nonempty finite sum. Its gradient is the
expectation of `1_S` under strictly positive exponential weights, and its
Hessian is their covariance. This proves convexity without differentiating SVD.
Every support vector has sum `n`, hence `f_M(x+c1)=f_M(x)+nc` and `Phi` is
constant on global shifts.

For completeness define the finite support
`V={1_S+1_T : S in B_U, T in B_D}`, its convex hull `P`, and
`b=(2n/m)1`. Combining the two sums writes
`Phi=log sum_v a_v exp((v-b)^T x)` with positive combined coefficients.

* A finite stationary point exists **iff `b` is in the relative interior of
  `P`**. Necessity follows because a finite `x` gives a positive weighted mean
  of every support point. For sufficiency, restrict to the span of support
  differences: interiority makes the support function of `P-b` positive on
  every unit direction there, so `Phi` is coercive on that subspace and attains
  its minimum.
* The Hessian kernel consists of directions whose inner product with every
  support point is the same. Uniqueness modulo constants holds iff these
  directions are exactly `span(1)`.
* If `b` is on the relative boundary, no finite minimizer attains it; a limiting
  scaling may approach it. If `b` lies outside `P`, strict separation supplies
  a direction along which `Phi -> -infinity`.

If both matrices are full spark, **every** size-`n` subset occurs. The convex
hull for each side is `{z: 0<=z_i<=1, sum z_i=n}`, so
`P={z: 0<=z_i<=2, sum z_i=2n}`. For `0<n<m`, `b` is strictly inside relative to
this hyperplane. Exchanging a single member between size-`n` subsets shows that
a direction constant on all support points must have all components equal.
Existence and uniqueness on `sum x=0` follow. This fills the missing argument.

Full spark is sufficient, not necessary, and is much stronger than full column
rank. For `m=n`, both full-rank polar factors are square orthogonal matrices,
all paired leverages are already two, and every `x` minimizes `Phi`; the source
correctly excludes this case from its uniqueness clause.

## 3. Counterexamples and implementation consequences

| Finding | Exact source location or claim | Evidence and corrected scope |
| --- | --- | --- |
| Absolute epsilon breaks the gauge law | `core.py:56`, `core.py:84`; Theorems 1–3 as implemented | With a row pair `r=1e-16,s=1` and `c=1e4`, clamped `h=1e-12` but `h'=1e-8`, instead of `c²h=1e-4`. Balanced row norms differ by `9.999e-7`. State the guarantees only when clamping is inactive in both gauges; choose a boundary policy in a later design change. |
| One-sided zero cannot be balanced | Section 2 and `canonical_pair` docstring | `r=0,s=1` has no finite balancing gauge. Both-zero pairs have a stabilizer and undefined metric. Do not describe epsilon as an exact extension of the theorem. |
| Finite weights can yield nonfinite norms | `gauge_metric`, `canonicalize_pair_` | Constant `2x2` entries of `1e200` in float64 or `1e20` in float32 are finite, but sum-of-squares row norms overflow on this backend. Clamping does not fix overflow, ratio underflow, or exponential overflow. |
| Rank-deficient polar is not unique | `exact_polar` docstring at `core.py:34`; Sections 7–9 | The two optimizers `[e1,e2]` and `[e1,-e2]` above attain the same objective. Replace “well-defined” if it implies a unique intrinsic factor with “an SVD-selected Stiefel completion.” |
| Zero gradient causes motion | `core.py:329`, `core.py:330` | At zero momentum, `exact_polar(zeros(5,2))` has norm `sqrt(2)`. A fresh optimizer given zero gradients and `lr=.01` takes a canonical up step of norm `.014142`. This is an allowed but arbitrary Stiefel LMO choice; whether zero gradients should yield zero updates is an unresolved policy, not changed in this audit. |
| Full rank does not imply a feasible target | Theorem 7 cannot omit full spark without a replacement condition | `M_U=M_D=[[1,0],[0,1],[0,0]]` has paired leverage `(2,2,0)` for every `x`, never `(4/3,4/3,4/3)`. For `x=t(-1/2,-1/2,1)`, `Phi=-2t`. The reference solver returns residual `4/3` after 100 steps, with no infeasibility status. |
| A boundary target need not have a finite solution | Existence clause in Theorem 7 | In the tested `4x2` pair, every basis of `M_U` contains row 1, so `ell_U[1]=1`; `ell_D[1]>0` at every finite `x`. Target one requires `x_1 -> -infinity`. Residual falls to `2.517e-11` by `x_1=-24` but never vanishes at finite `x`. |
| Centering alone need not restore uniqueness | Uniqueness clause in Theorem 7 | For both momenta `[[1,0],[1,0],[0,1],[0,1]]`, all centered `x=(t,t,-t,-t)` give target `(1,1,1,1)` and identical `Phi`. Use the support-difference condition above. |
| Convexity is not an unconditional solver guarantee | Sections 10 and 14; `balance_shared_metric` | With full-spark `2x1` all-ones momenta and `x=(a,-a)`, the iteration is `a <- a-gamma*tanh(a)`. `gamma=4`, `a0=2` leaves residual `.9575` after 100 steps; `gamma=.5` converges. Finite iteration counts, smoothing and moving momenta need separate convergence/tracking statements. |

For a fixed full-rank problem, the Hessian above has nonpositive off-diagonal
entries and zero row sums. Its largest eigenvalue per matrix is at most
`2 max_i ell_i(1-ell_i) <= 1/2`, so `grad Phi` is globally 1-Lipschitz. Ordinary
gradient descent with `0<gamma<2` and an attained minimum has the usual fixed
objective convergence guarantee. This does not establish a rate uniform over
nearly degenerate matrices, or convergence of the smoothed, changing-momentum
online algorithm. The source's prediction that paired leverage CV “must converge
to zero” needs this distinction. Neuron activation and training-loss predictions
are empirical hypotheses, not consequences of uniform update leverage.

## 4. Implementation and research cases

The core implementation follows the intended regular-domain formula: canonical
gradients precede EMA; the same centered `x` reweights both sides before SVD;
the lift uses reciprocal factors; and the original down layout is restored.
Balancing updates `x` for the next step while the current step uses the already
computed polars. Optional weight canonicalization occurs after the update.

Additional limits established by inspection or tests:

* `balance_shared_metric` has a fixed iteration budget, no rank/feasibility
  check and no convergence status. It does not validate `x0` shape, finite
  inputs or an admissible learning rate. Positive `eps`, common pair dtype and
  device, nonempty matrix dimensions, and finite balancing state are not
  comprehensively validated by the optimizer either.
* SVD orthogonality alone cannot establish direction accuracy. The rank-boundary
  test changes a `3x2` input by `2e-12` and changes its exact polar by `2`. The
  float32 experiment below exhibits the same sensitivity through quantization.
  Low-rank minibatch gradients make this relevant beyond contrived zero inputs.
* Momentum and balancing state use parameter/gradient dtype. There is no float32
  master-state or promotion path. Native bfloat16 fails at CPU SVD. The explicit
  promote-to-float32 test is a measurement of quantization, not a new supported
  execution mode. Autocast, loss scaling, GPU/MPS and float16 were not validated.
* A warm checkpoint and mid-training gauge reset pass for a single registered
  pair, with weight canonicalization both enabled and disabled. State is keyed
  by the up parameter; reconstructing the same pair topology and ordering is
  required when loading. `_pairs` is not serialized as topology metadata.
* The implementation intentionally reads only `param_groups[0]` and iterates
  `_pairs`. Adding ordinary optimizer parameter groups would not register more
  pairs or apply their hyperparameters. Missing either gradient skips the whole
  pair. Multiple-pair checkpoint permutations, shared/duplicate parameters and
  distributed checkpoint formats remain untested API cases.
* `diagnostics()` recomputes polars with the updated `x`, so it describes the
  current hypothetical canonical map, not necessarily the last applied step.
  Its sample standard deviations are undefined for `m=1`; the intended theorem
  regime has `m>n>=1`, but the implementation also accepts square matrices.
* The original test module changes PyTorch's default dtype globally. Every new
  test explicitly chooses dtype and seeded local generators so its math does
  not depend on that collection side effect.

### Distributed semantics

There is no distributed implementation or distributed execution test. The
mathematics applies to global paired matrices, and local shard formulas are not
generally substitutes. A concrete column-shard test uses rows `(3,4)` and `(4,3)`:
the global norm ratio is one while local ratios are `3/4` and `4/3`.

For column sharding, row squared norms require reduction across the relevant
shards. For row sharding, the polar couples rows through the global Gram matrix;
local SVDs change the problem. Mean-centering `x`, the target `2n/m`, and paired
row ownership must use global dimensions and matching neuron identities. With
data parallelism, replicas must use consistent gauges and canonical states;
averaging raw gradients expressed in different gauges does not implement the
stated covector transformation. Collective rounding/reduction order introduces
additional numerical error. No distributed invariance claim is certified here.

### Scope of external-method claims

The source's descriptions of NorMuon, Aurora and DDC and its candidate novelty
claim in Section 13 were not independently verified as a literature review.
The raw-Muon experiment is a local diagnostic implementation, not a performance
or accuracy comparison against those projects. This audit establishes the
internal mathematical and implementation claims requested by the first task.

## 5. Theorem-to-test map

New test names below are in `tests/test_theory_audit.py`; names marked “original”
are in `tests/test_qnormuon.py`. Unless specified, inputs are CPU float64.
`rtol/atol` describe elementwise `assert_close`; scalar bounds are absolute unless
called relative. A finite numerical test supports a derivation, not a universal
proof. Parameterized seeds and precisions count separately in pytest totals.

| Claim | Assumptions | Implementation | Automated tests | Tolerance |
| --- | --- | --- | --- | --- |
| Lemma 1 | Fixed gate, positive diagonal scales; `12x4`, three seeds, `c=1e-6..1e6` | `canonical_pair`; actual SiLU forward in test | `test_actual_swiglu_function_gradients_and_canonical_coordinates` | Output and gradient `rtol=atol=2e-11`; rowwise gauge identity `<2e-10` |
| Theorem 1 | Nonzero unclamped paired norms | `gauge_metric`, `canonical_pair`, `canonicalize_pair_` | Original `test_canonical_pair_is_gauge_invariant_and_balanced`; `test_metric_covariance_duality_and_balanced_value`; reset test below | Original `1e-10`; metric and post-step norms `2e-12` |
| Theorem 2 | Invariant loss; matching canonical state | `canonical_gradients`, `QNorMuon.step` | Actual SwiGLU test; `test_optimizer_midtraining_gauge_reset_and_checkpoint` | Gradients `2e-11`; state/function `2e-10` over nine steps, reset after three |
| Theorem 2 domain | Loss must be invariant | `canonical_gradients` | `test_counterexample_raw_l2_loss_does_not_have_covariant_gradients` | Demonstrates relative discrepancy `>1` |
| Theorem 3 | Matching invariant `x`, deterministic map, regular domain | `quotient_polar_update`, `QNorMuon.step` | Original equivariance test; `test_large_dynamic_range_gauge_and_reference_precision`; warm reset test | Original `2e-9`; relative float64 `<2e-11`, float32 `<2e-5` |
| Theorem 3 approximate map | Fixed canonical-only approximation | Scoped test substitution of `core.exact_polar` | `test_fixed_approximate_polar_preserves_gauge_but_not_exact_stiefel` | Relative gauge `<2e-11`; demonstrates Stiefel defect `>0.1` and SVD discrepancy `>0.01` |
| Theorem 4 | Full rank, positive metric, raw covector used once | `quotient_polar_update`, `exact_polar` | `test_weighted_variational_optimum_uses_raw_covector_once`; original generalized condition test | Objective/Gram `2e-12`; original Gram `1e-10`; incorrect extra metric loses `>1e-3` |
| Theorem 5 | Positive norm-local metric; covariance and duality | `gauge_metric` | `test_metric_covariance_duality_and_balanced_value` | `rtol=atol=2e-12`; uniqueness proved analytically |
| Theorem 6 | Full column rank, finite `x`; `7x3`, three anisotropic seeds | `paired_leverage`, `exact_polar`; independent `phi` test helper | `test_phi_gradient_hessian_and_global_shift` | Analytic/autograd `2e-10`; central differences `2e-7`, step `1e-5`; Hessian agreement `2e-10` |
| Theorem 7 convexity | Full column rank | Independent `phi` and polar helpers | Same derivative test | Minimum Hessian eigenvalue `>-2e-10`; global null direction `<2e-10`; shift `2e-11` |
| Theorem 7 existence/uniqueness | Full spark, `m>n`; all `6x2` minors explicitly checked | `balance_shared_metric` | `test_full_spark_balancing_unique_centered_solution`; original balancing test | Residual `<2e-10`; centered solutions from three starts `2e-9`; Stiefel `2e-12`; original residual `<1e-9` |
| Theorem 7 excluded cases | Full rank without full spark | `balance_shared_metric`, independent `phi` | `test_counterexample_full_rank_does_not_imply_balancing_exists`; `test_counterexample_boundary_target_only_attained_at_infinity`; `test_counterexample_full_rank_can_have_extra_flat_balancing_directions` | Infeasible residual `4/3`; boundary residual `<1e-9` but positive; flat objective/gradient geometry `2e-12` |
| Sections 10/14 convergence | Learning-rate restriction needed even with full spark | `balance_shared_metric` | `test_counterexample_convexity_does_not_guarantee_arbitrary_solver_step_convergence` | `lr=4` residual `>0.9`; `lr=.5` residual `<2e-12` |
| Rank deficiency | Explicitly outside logdet domain | `exact_polar`, `QNorMuon.step` | `test_counterexample_rank_deficient_polar_is_nonunique_and_discontinuous`; `test_counterexample_zero_gradient_produces_nonzero_optimizer_step` | Input gap `<3e-12`, output gap `>1.9`; step norm `.01 sqrt(2)` within `2e-14` |
| Zero/near-zero norms | Clamped and singular cases | All gauge helpers | `test_counterexample_clamping_breaks_balance_and_gauge_covariance`; `test_both_zero_rows_have_no_unique_gauge_transformation`; `test_counterexample_finite_weights_can_overflow_norm_computation` | Balance failure `>9e-7`; metric relative comparison fails at `1e-3`; explicit nonfinite output for overflow |
| Reduced precision | Well-conditioned `24x8` and nearly deficient `8x3` | Exact SVD; explicit test-only input promotion | `test_large_dynamic_range_gauge_and_reference_precision`; `test_bfloat16_cpu_support_and_explicit_promoted_input_measurement`; `test_nearly_rank_deficient_float32_sensitivity` | Float32 relative/defect `<2e-5`; promoted bfloat16 reference `<.02`, gauge `<.03`; near-rank discrepancy `>1e-3` despite defect `<2e-5` |
| Distributed research case | Global versus column-shard norms | `gauge_metric` on a global matrix and its slices | `test_counterexample_column_shard_local_norms_are_not_global_metric` | Local/global metric relative error `>0.1`; no distributed run claimed |

## 6. Measured results and reproduction

Environment: macOS 26.6.2 arm64, CPU, Python **3.9.6**, PyTorch **2.8.0**, Apple
Accelerate BLAS/LAPACK. The available Python is below `pyproject.toml`'s declared
`>=3.10`; source-tree tests ran successfully, but package installation and a
supported-Python compatibility run were not performed. No dependency or
environment configuration was changed.

Commands run from the repository root:

```sh
python3 -m pytest
python3 -m pytest -s
python3 -m experiments.gauge_stress
python3 -m experiments.toy_trajectory
```

The full suite passed before changes (**5 passed**) and after the additions
(**34 passed**, no warnings). All five original tests remain unchanged; 29 new
parameterized cases add the coverage above. Both existing experiments also ran.

Metrics below are measured values, not portable exact constants. Relative
errors use Frobenius/Euclidean norm relative to the reference. Gauge errors for
the new precision tests undo `C` before comparison, avoiding domination by the
largest-scale rows. Stiefel defect is `||P^TP-I||_F`. Derivative discrepancies
and paired residuals are maximum absolute component errors.

| Experiment | Dtype and shape | Observed result |
| --- | --- | --- |
| Actual SwiGLU gauge, three seeds | float64, `12x4`, batch 20, scales `1e-6..1e6` | Maximum function error `2.446e-16`; canonical up-gradient error `4.207e-16` |
| Warm gauge reset, checkpoint, nine steps | float64, `9x3`, batch 32 | Maximum function error `2.955e-16` without weight reset; `2.392e-16` with reset |
| Analytic/autograd/finite difference | float64, `7x3`, three seeds | Maximum gradient discrepancy `1.554e-15`; finite-difference discrepancy `4.400e-10`; minimum sampled Hessian eigenvalue `-6.687e-17` |
| Full-spark solve, three initializations | float64, `6x2`, 1500 steps, lr `.5` | Maximum paired residual `3.331e-16` |
| Gauge and spectral precision | float64, `24x8`, scales `1e-6..1e6` | Gauge error `1.098e-15`; Stiefel defect `3.580e-15` |
| Gauge and spectral precision | float32, `24x8`, same data/scales | Gauge error `5.770e-7`; error versus float64 `4.638e-7`; Stiefel defect `2.132e-6` |
| Native low-precision path | bfloat16, CPU, `24x8` | Raises `linalg_svd_cpu not implemented for BFloat16` |
| Quantize then explicitly promote for SVD | bfloat16 inputs, float32 solve, `24x8` | Error versus float64 `2.793e-3`; gauge error `3.736e-3` |
| Near-rank precision stress | float32 versus float64, `8x3`, singular values `1,1e-4,1e-8` | Relative polar discrepancy `3.234e-1`, despite float32 Stiefel defect `2.703e-7` |
| Original gauge experiment | float64, `32x8`, sampled scales `1e-3.93..1e3.79` | Up/down equivariance errors `5.316e-16`, `7.645e-16` |
| Original trajectory experiment | float64, `18x6`, batch 64, 20 steps | QNorMuon maximum function mismatch `4.869e-16`; local raw-Muon baseline `3.056` |

The next design discussion should settle the zero-row and deficient-rank
policies first, then specify finite-precision and balancing convergence
contracts. Those decisions require coordinated theory/code changes; this audit
provides the counterexamples and reproducible checks without choosing a new
optimizer design.
