# Rank-deficient QNorMuon: partial-polar research and prototypes

Date: 2026-09-24. Status: mathematical proposal for review, **not an adopted
optimizer design**. The complete [theory audit](THEORY_AUDIT.md) and original
[specification](QNORMUON_THEORY.md) were read before this investigation.

All five partial-polar hypotheses are true in exact arithmetic, with the usual
canonical-input qualifications for gauge equivariance. The proposed map is the
**unique minimum-Frobenius-norm spectral-ball maximizer**. It removes arbitrary
null-space updates and is zero at zero momentum. It preserves an isometry on
the active input subspace, rather than full-column Stiefel orthogonality.

There is a principled convex balancing objective at deficient rank: for **fixed
momentum and positive finite row weights**, pseudo-logdet is exactly ordinary
logdet on an orthonormal basis of the active subspace. The paired target becomes
`(r_U+r_D)/m`. This equivalence does not make rank changes continuous, and rank
adaptation does not by itself guarantee balancing feasibility.

Fixed positive ridge is a different, smooth problem: it replaces unit nonzero
singular values by contractions and replaces integer leverage mass by effective
dimension. It cannot be combined with the exact rank target as an unconstrained
balancing objective with a finite stationary point, except for the all-zero
case. Neither ridge nor a rank threshold is selected as production policy here.

Research code is isolated in [experiments/rank_deficient.py](../experiments/rank_deficient.py),
with [tests/test_rank_deficient.py](../tests/test_rank_deficient.py). Nothing in
`qnormuon/`, the original specification, or earlier tests was changed.

## 1. Notation, domain and research sources

Let `M = M_can` be an `m x n` real canonical momentum matrix, usually `m>n`,
`K(x)=diag(exp(x))`, and

\[
A=K^{1/2}M=U_r\Sigma_rV_r^T,\qquad \sigma_j>0,\quad r=\operatorname{rank}(A).
\]

This is the compact SVD: `U_r` is `m x r`, `V_r` is `n x r`. Empty matrices are
allowed when `r=0`. Define

\[
P_0(A)=U_rV_r^T=A\big[(A^TA)^+\big]^{1/2}.
\tag{1}
\]

The subscript zero denotes no ridge, not a rank choice. Unless explicitly
described as numerical truncation, rank means mathematical rank. Positive finite
`K` is invertible, so **rank(A)=rank(M) and row(A)=row(M) for every finite x**.
This observation is what permits a fixed active subspace while balancing.

The canonical partial-isometry definition is consistent with Theorem 1.2 of
[Higham, Mehl and Tisseur, *The Canonical Generalized Polar Decomposition*](https://eprints.maths.manchester.ac.uk/1490/1/htm10.pdf).
For pseudo-determinants, [Holbrook, *Differentiating the pseudo determinant*](https://arxiv.org/pdf/1802.04878)
explicitly treats discontinuity between ranks and differentiation on restricted
domains. The effective-dimension terminology follows Definitions 5–6 of
[*PROMISE: Preconditioned Stochastic Optimization Methods*](https://www.jmlr.org/papers/volume25/23-1187/23-1187.pdf).
These are primary-source cross-checks; the optimizer-specific statements and
proofs below are derived here, including the balancing and continuity caveats.

This investigation does not repair zero **weight** norms: `H` must still be
positive, finite and covariant, with canonical gradients computed without an
active absolute clamp. Zero momentum and a zero weight row are different cases.

## 2. Spectral-ball maximization: proof of the minimum-norm claim

Consider

\[
\max_{\|Y\|_2\le1}\langle A,Y\rangle_F.
\tag{2}
\]

For each active singular triplet,
`u_j^T Y v_j <= ||Y v_j|| <= 1`, hence

\[
\langle A,Y\rangle=\sum_{j=1}^r\sigma_j u_j^TYv_j
\le\sum_{j=1}^r\sigma_j=\|A\|_*.
\]

`P_0` attains this bound. Equality forces `Yv_j=u_j` for every positive singular
value. Because `Y^T` is also a contraction, `Y^Tu_j=v_j`. Complete `U_r,V_r` to
orthogonal bases with `U_perp,V_perp`. Both off-diagonal blocks therefore vanish,
and **all maximizers**, including at rank zero, have the form

\[
Y=P_0+U_\perp ZV_\perp^T,\qquad \|Z\|_2\le1.
\tag{3}
\]

Conversely every such matrix is feasible and optimal. Orthogonal invariance gives

\[
\|Y\|_F^2=r+\|Z\|_F^2.
\]

The unique minimum is at `Z=0`. This proves the claim and uniqueness of the
tie-breaking rule; it does **not** claim uniqueness of all maximizers. For tall
full-column-rank matrices there is no remaining right block, so it agrees with
the original full polar. At `A=0`, every contraction is a maximizer but only
`Y=0` has minimum Frobenius norm.

Consequently:

| Hypothesis | Verdict | Exact statement |
| --- | --- | --- |
| Minimum-norm maximizer | Proven | Unique minimum-Frobenius-norm element of (2)'s maximizers |
| Zero input gives zero | Proven | `P_0(0)=0`; a zero current gradient alone does not imply zero accumulated momentum |
| Right Gram is row-space projector | Proven | `P_0^T P_0=V_r V_r^T=Pi_row(A)` |
| Nonzero singular values are one | Proven | Exactly `r` unit singular values; all others zero; empty positive spectrum at rank zero |
| Gauge-equivariant optimizer lift | Proven with assumptions | The partial polar is gauge invariant on identical canonical inputs; its **lift** is gauge equivariant |

Signs and orthogonal rotations of singular vectors within repeated positive
singular values cancel in `U_r V_r^T`. There is no arbitrary completion on the
kernel. This does not imply continuity across a rank boundary (Section 7).

## 3. Gauge law, lift and corrected generalized conditioning

Under the original positive neuron gauge, `H'=C²H` and correctly canonicalized
momentum and shared `x` are invariant. Thus `A'=A`, `P_0(A')=P_0(A)` and

\[
\Delta U=H^{1/2}P_{0,U},\quad
\Delta D=H^{-1/2}P_{0,D},\quad
\Delta U'=C\Delta U,\quad\Delta D'=C^{-1}\Delta D.
\tag{4}
\]

This proof applies at every rank, including zero, and also to canonical-only
ridge or threshold maps with identical settings. It does not certify numerical
rank decisions on approximately equal inputs near a cutoff, arbitrary
noninvariant regularization, or the existing clamp's boundary behavior.

The corrected Gram statements are

\[
\boxed{\Delta U^T H^{-1}\Delta U=\Pi_{\mathrm{row}(M_U)},\qquad
\Delta D^T H\Delta D=\Pi_{\mathrm{row}(M_D)}.}
\tag{5}
\]

In particular, `||H^{-1/2} DeltaU v||=||v||` for `v` in the active row space,
and it is zero on its orthogonal complement. The analogous down metric is `H`.
The metric-whitened update has `r` singular values one and `n-r` zeros:

* For `r=n`, the original generalized condition number is one.
* For `0<r<n`, the **restricted positive-spectrum condition number** is one;
  the full-column condition number is infinite, since an inverse on all input
  directions does not exist.
* For `r=0`, there is no positive spectrum. An active condition number is
  undefined, not one. Diagnostics should explicitly report zero rank.

The ordinary Stiefel defect is now `||P_0^T P_0-I_n||_F=sqrt(n-r)` in exact
arithmetic. At deficient rank this is intended, not solver error. Appropriate
diagnostics are projector defect, idempotence defect, active isometry error,
rank and smallest retained singular value. Generalized spectra of the lifted
update refer to the whitened metric, not raw Euclidean singular values.

The corresponding variational constraint must change from equality to the ball

\[
\Delta^T H^{-1}\Delta\preceq I
\quad\Longleftrightarrow\quad\|H^{-1/2}\Delta\|_2\le1.
\tag{6}
\]

For raw covector `B`, substitute `Y=H^{-1/2}Delta`; the canonical objective
matrix is `H^{1/2}B`. Its minimum-canonical-Frobenius-norm maximizer lifts (1).
As in the audit, stored `M_can` has already undergone this transformation. For
`K != I` the actual canonical objective is `K^{1/2}M_can`, not `M_can` and not a
second application of `H^{1/2}`. Partial polar is not feasible for the original
equality constraint at `r<n`; this is a real mathematical design change.

## 4. Partial leverage and the rank-adaptive paired target

Write `G=M^TKM=A^TA`. The left Gram is the orthogonal projector

\[
Q=P_0P_0^T=U_rU_r^T=AG^+A^T.
\]

Therefore the correct row leverage is

\[
\ell_i=\|P_0[i,:]\|^2=Q_{ii}
=e^{x_i}m_i^TG^+m_i,\qquad 0\le\ell_i\le1,
\]
\[
\boxed{\sum_i\ell_i=\operatorname{tr}Q=r.}
\tag{7}
\]

The equality is exact in mathematics and holds to rounding error in the SVD
prototype. It is mathematical rank for exact (1), and retained numerical rank
if positive singular values have been truncated. A zero row cannot acquire
leverage through row reweighting alone.

For the pair, let `R=r_U+r_D`. Uniform total leverage, if attainable, **must**
equal `R/m` since its sum is `R`. Ranks need not agree, and the two row spaces
need not agree. One zero momentum reduces the problem to the other side. If
both are zero, the target and both updates are zero, the balancing objective
is constant, and no unique balancing state is implied.

Rank adaptation does not remove support obstructions. For example, two `3x2`
rank-one matrices supported only on row one have paired leverage `(2,0,0)` for
all `x`, not `(2/3,2/3,2/3)`. A more subtle rank-(1,1) boundary example in the
tests has one matrix supported on the first two of four rows and another on all
four: the first side already contributes mass one to the first pair of rows,
while the second contributes a strictly positive amount at finite weights.
The uniform target needs the latter contribution to vanish only at infinity.

### A consequence for the existing centered dynamics

Let `J=I-11^T/m`. For any scalar target `t`, including `2n/m` and `R/m`,

\[
J(\ell_U+\ell_D-t1)=\ell_U+\ell_D-(R/m)1.
\tag{8}
\]

Thus changing the uniform target alone has **no effect** on exact, mean-centered
`x` dynamics for the same polars. This remains true with balancing EMA because
`Jq_{t+1}=beta Jq_t+(1-beta)Jell_t`. The uncentered EMA mean and reported
residual do change. The rank target is required for an honest gradient identity,
unconstrained objective and diagnostics; the partial-polar change is what alters
the centered vector field. The new test verifies this with populated balancing
momentum. Floating-point subtraction of a drifting mean can still matter.

## 5. Pseudo-determinant and active-subspace restriction

Fix `M` of rank `r`. Choose an **orthonormal right basis** `V` of `row(M)` and
let `B=MV`, so `M=BV^T` and `B` has full column rank. Since

\[
G=V(B^TKB)V^T,
\]

its nonzero eigenvalues are exactly those of the `r x r` positive-definite
matrix `B^TKB`. Consequently

\[
\boxed{f_0(x)=\log\operatorname{pdet}(M^TKM)
=\log\det(B^TKB).}
\tag{9}
\]

Use the empty-product convention `pdet(0)=det(empty)=1`, so `f_0=0` when `r=0`.
This is a convention for the rank-zero stratum, not a continuity statement.
The partial factor is equivalently
`P_0(K^{1/2}M)=polar(K^{1/2}B)V^T`. Rotating `V` by any orthogonal `r x r`
matrix changes neither the objective nor the reconstructed partial polar.

If instead an independent column subset `N` gives `M=NT` with `T` full row
rank, then
`pdet(M^TKM)=det(N^TKN) det(TT^T)`.
The second term is an `x`-independent constant; ignoring it preserves balancing
gradients but is not literal equality of objective values. A fixed orthonormal
basis makes (9) exact without this correction.

Since the kernel is fixed while `x` varies, ordinary differentiation of (9)
is legitimate and yields

\[
\partial_i f_0=e^{x_i}m_i^TG^+m_i=\ell_i,\qquad
\nabla^2 f_0=\operatorname{Diag}(\ell)-Q\odot Q.
\tag{10}
\]

This is a restricted-domain derivative, not an unrestricted smooth extension in
all entries of `M`. It agrees with the pseudo-determinant derivative domain
discussed by [Holbrook](https://arxiv.org/pdf/1802.04878); here the direct
restriction argument avoids differentiating an SVD basis or discrete rank.

For convexity, Cauchy–Binet on `B` gives

\[
\det(B^TKB)=\sum_{|S|=r}\det(B_S)^2 e^{1_S^Tx}.
\tag{11}
\]

Nonzero minors provide a nonempty log-sum-exp; for `r=0` the sole empty-set term
is one. Its Hessian is the covariance of subset indicator vectors, hence PSD.
`f_0(x+c1)=f_0(x)+rc`. The coupled objective is therefore

\[
\boxed{\Phi_0(x)=f_{0,U}(x)+f_{0,D}(x)-\frac Rm1^Tx,\quad
\nabla\Phi_0=\ell_U+\ell_D-\frac Rm1.}
\tag{12}
\]

It is convex, invariant to global `x` shifts, and has a zero-sum gradient.
The Hessian has the same upper spectral bound one as in the audit: each
projection contribution is a Laplacian with largest eigenvalue at most
`2 max_i ell_i(1-ell_i)<=1/2`. Fixed-momentum gradient descent with an attained
minimum and `0<gamma<2` has the usual convergence guarantee, with no uniform
rate near degeneracy and no automatic guarantee for online moving momenta.

### Existence and uniqueness with different ranks

Let `S_U` be the `r_U`-row subsets whose reduced matrices have nonzero determinant;
define `S_D` similarly. For rank zero use `{empty set}`. Let

\[
\mathcal P=\operatorname{conv}\{1_S+1_T:S\in S_U,T\in S_D\},\qquad b=(R/m)1.
\]

Exactly as in the audit, a finite minimizer exists iff `b` is in the **relative
interior** of `P`. The reason is that the gradient is a strictly positive mean
of the support points, and interiority gives coercivity on their difference
span. Uniqueness modulo constants holds iff the only vectors constant on this
support are multiples of `1`. Boundary targets require limiting scales; an
outside target makes the objective unbounded below in a separating direction.

A sufficient condition for the tall setting is that each nonzero reduced matrix
is full spark in its **own rank**: all `r_U`-row or `r_D`-row maximal minors are
nonzero. Each support hull is a rank-`r` hypersimplex, whose relative interior
contains `(r/m)1`. Their Minkowski sum contains `b` in its relative interior.
If at least one rank lies strictly between zero and `m`, the differences span
`1^perp`, giving uniqueness after centering. If both ranks are zero, all `x`
minimize the objective. The old test of nonzero `n x n` minors cannot be applied
to a deficient-rank matrix; all such minors vanish.

**Conclusion of this comparison:** pseudo-determinant and restriction to the
current complete active subspace are the same fixed-momentum mathematical
choice. Restriction is a transparent way to evaluate and differentiate it.
Freezing an obsolete or truncated subspace when momentum changes is a different
algorithm and need not maximize the current spectral objective.

## 6. Ridge logdet: derivation of a different smooth alternative

For fixed `lambda>0`, define

\[
f_\lambda(x)=\log\det(G+\lambda I_n),\qquad
P_\lambda(A)=A(G+\lambda I_n)^{-1/2}.
\tag{13}
\]

Its singular values are `sigma_j/sqrt(sigma_j²+lambda)`; zero singular values
stay zero. This matrix function is smooth in `A` for fixed positive `lambda`,
is zero at zero, and remains canonical-input gauge invariant. But it is neither
a partial isometry nor an exact maximizer of (2) for nonzero `A`.

Differentiation gives

\[
\partial_i f_\lambda=e^{x_i}m_i^T(G+\lambda I)^{-1}m_i
=\|P_\lambda[i,:]\|^2=\ell_{\lambda,i},
\]
\[
\boxed{d_\lambda=\sum_i\ell_{\lambda,i}
=\sum_{j=1}^r\frac{\sigma_j^2}{\sigma_j^2+\lambda}<r\quad(r>0).}
\tag{14}
\]

This is the effective dimension, consistent with the ridge-leverage identities
in [PROMISE, Definitions 5–6](https://www.jmlr.org/papers/volume25/23-1187/23-1187.pdf).
The shrinkage values rather than ones also determine generalized conditioning:
on the active space the positive condition number is the ratio of the largest
and smallest `sigma/sqrt(sigma²+lambda)`, generally greater than one. It can
diverge as a positive singular value tends to zero.

Convexity in `x` still holds. Expanding the ridge determinant gives

\[
\det(G+\lambda I)=
\sum_{S\subseteq[m],\,|S|\le r}
\lambda^{n-|S|}\det(M_SM_S^T)e^{1_S^Tx}.
\tag{15}
\]

All nonzero coefficients are positive and the empty term is `lambda^n`.
This is a log-sum-exp of supports of **different sizes**. Setting
`Q_lambda=P_lambda P_lambda^T` gives
`Hess f_lambda=Diag(ell_lambda)-Q_lambda ⊙ Q_lambda`, again PSD. Unlike (10),
it generally does not annihilate `1` because `Q_lambda` is not idempotent.

### Why the exact rank target cannot simply be reused

With nonzero paired rank `R`, consider
`Phi_lambda=f_lambda,U+f_lambda,D-(R/m)sum x` on unrestricted `R^m`. Then

\[
1^T\nabla\Phi_\lambda=d_{\lambda,U}+d_{\lambda,D}-R<0
\tag{16}
\]

at every finite `x`. Thus there is no stationary point: increasing every row
weight reduces the objective toward its unregularized limit. Replacing `R` by
`2n` makes the incompatibility at least as strong. Using partial-polar leverage
to drive a ridge objective is also incorrect: it is not its gradient.

There are derived alternatives, but each changes the problem:

* On the **constraint** `sum x=0`, minimize the ridge objective. Its projected
  gradient is `ell_lambda,U+ell_lambda,D - (d_lambda,U+d_lambda,D)1/m`.
  A stationary point equalizes ridge leverage with its **endogenous effective
  dimension** target. This is a constrained convex problem; it does not assert
  that such a point always exists for arbitrary row supports.
* Equivalently, define an objective on arbitrary `z` by evaluating ridge at
  `x=Jz`. Its gradient is the same projected expression and is invariant under
  `z -> z+c1` by construction. This fixes a scale relative to the ridge, rather
  than using an already irrelevant scale.
* A prescribed smaller total target would define another convex objective, with
  existence determined by the convex hull of the variable-cardinality supports
  in (15). It is not the rank-adaptive proposal.

Subtracting `d_lambda(x) sum x/m` and treating `d_lambda` as constant is **not**
an unconstrained derivation: differentiating it also introduces derivatives of
`d_lambda`. The projected-gradient argument above supplies the correct logic.

There are two separate symmetries: the original **neuron gauge** still holds for
fixed ridge on canonical inputs, but global **K scaling** does not. Taking
`lambda` proportional to a scale of `A²` can restore positive homogeneity of
the map; however it reintroduces discontinuity at zero and would change the
objective derivative when that scale depends on `x`.

### Relation to the pseudo-determinant limit

For fixed `M,x` and its true rank `r`, spectral factorization yields

\[
f_\lambda(x)=(n-r)\log\lambda+
\sum_{j=1}^r\log(\sigma_j^2+\lambda).
\]

Hence `f_lambda-(n-r)log lambda -> f_0`, `P_lambda -> P_0`, and ridge leverage
converges to partial leverage as `lambda -> 0`. The subtracted term is constant
in `x` only while rank is fixed. These limits are not uniform near a rank change.

| Choice | Gradient leverage | Total mass | Generalized Gram | Rank-change continuity | Global K scaling |
| --- | --- | --- | --- | --- | --- |
| True partial polar + pseudo-logdet | `diag(P_0 P_0^T)` | `r` | Active projector | Discontinuous | Invariant map; objective shifts by `rc` |
| Complete active restriction | Identical to previous row | `r` | Same projector | Same discontinuity when active space changes | Same |
| Fixed positive ridge and `P_lambda` | `diag(P_lambda P_lambda^T)` | `d_lambda<r` | Eigenvalues `sigma²/(sigma²+lambda)` | Smooth map and objective | Broken unless reformulated |
| Hard threshold | Retained projector leverage | Selected numerical rank | Projector onto retained space | Jumps at cutoff; possible ties | Absolute cutoff breaks it; pure relative cutoff preserves positive scaling away from decisions |

## 7. Continuity, counterexamples and numerical rank

Consider the `3x2` matrices

\[
A(t)=\begin{pmatrix}1&0\\0&t\\0&0\end{pmatrix}.
\tag{17}
\]

For `t != 0`, `P_0(A(t))=[e_1,sign(t)e_2]`; at zero it is `[e_1,0]`.
The distance to the rank-one value is **one for every nonzero t**. The two
one-sided limits differ by two. Leverage changes from `(1,1,0)` to `(1,0,0)`;
the right projector changes from `I_2` to `diag(1,0)`. Paired targets jump by
`1/m` when one momentum loses one rank. At the all-zero matrix, `P_0(tA)=P_0(A)`
for every `t>0`, so continuity there is impossible for any nonzero example.

This failure is not merely caused by the minimum-norm tie-break: for `t != 0`
in (17), (2)'s maximizer is unique and the opposite signs force incompatible
limits. **No everywhere-continuous exact spectral-ball maximizing selection
can agree on both paths.** A policy seeking global continuity must relax some
other property, such as exact linear maximization or unit active singular values.

On a fixed-rank stratum the canonical partial polar, row/column projectors and
pseudo-logdet are smooth locally, even when positive singular values repeat.
Sensitivity is not uniform as the smallest positive singular value goes to zero.
Individual SVD vectors may jump in sign or basis although the reconstructed map
does not. The tests include repeated active singular values and rotating
fixed-rank column spaces.

### Objective discontinuity and stale active spaces

At `x=0`, `log pdet(A(t)^T A(t))=2 log|t| -> -infinity` for nonzero `t`,
but its value at zero is `log(1)=0`. Recomputing the complete active restriction
has exactly the same behavior. Merely subtracting the value at `x=0` does not
repair it: for general `x`, the normalized objective is `x_1+x_2` at nonzero
`t`, but only `x_1` at zero. Its balancing gradient still jumps.

Freezing the old active space `span(e_1)` keeps a continuous value and update on
this example by ignoring the new second direction. For `t != 0` that loses
`|t|` in the linear objective and does not solve (2). Freezing an obsolete
larger subspace at a rank loss instead yields a singular restricted Gram. Rank,
subspace updates, and any temporal smoothing must be declared as design policy.

### Ridge limits

For (17), the second ridge singular factor is
`t/sqrt(t²+lambda)`, which tends to zero at fixed `lambda>0`. Its leverage is
`t²/(t²+lambda)` and its logdet increment is `log(1+t²/lambda)`; all are
continuous. But `lambda=t²` leaves a factor of magnitude `1/sqrt(2)` as
`t -> 0`. Taking ridge to zero first instead recovers the discontinuous partial
factor. Fixed positive ridge smoothness is therefore not a claim about a joint
zero-ridge, zero-singular-value limit.

### Hard thresholds and fixed numerical ranks

In floating point, a purported zero singular value of a constructed low-rank
matrix may be a small nonzero rounding artifact. The prototype intentionally
separates two contracts:

* `partial_polar(A, rank=r)` takes a caller-supplied mathematical rank. It is
  the exact partial-polar hypothesis when that assumption holds. It cannot
  certify omitted zeros; supplying a smaller rank computes the partial polar
  of a truncated approximation, not of the original input.
* `threshold_partial_polar(A, atol=..., rtol=...)` discards
  `sigma <= max(atol, rtol*sigma_max)` and reports the retained rank. There is no
  hidden default tolerance and no claim that its rank equals mathematical rank.

For `A=diag(1,.05)` and absolute cutoff `.1`, the retained-rank-one map loses
`.05` of the original LMO objective. An arbitrarily small perturbation across
the positive cutoff changes the polar by one. Finite row scaling can move the
second singular value across the cutoff while the **true rank stays two**;
absolute cutoff also breaks invariance under a global K scale. Fixing only a
rank while truncating within tied positive singular values can be ambiguous:
the leading rank-one polars of `diag(1+eps,1)` and `diag(1,1+eps)` differ by
`sqrt(2)` as `eps -> 0`. Keeping the entire repeated active space avoids that
particular ambiguity.

For fixed retained rank with a gap, the log product of the largest squared
singular values has the retained leverage as its derivative; this follows from
eigenvalue differentiation. However, a rank selected afresh by cutoff can jump,
and its value, linear rank target, and support are not the single fixed-rank
objective (12). Selecting a subspace before weighting and then holding it fixed
instead defines the objective of that truncated momentum. These alternatives
must not be silently conflated.

## 8. Prototype contracts and verification

The research module contains no `Optimizer` subclass and imports no production
optimizer code. It supplies the assumed-rank partial polar, an explicit cutoff
variant, fixed active bases, pseudo/active/ridge objective evaluators, a ridge
contraction, and a fixed-momentum active balancing solver. The solver returns
the target, residual and a convergence flag; lack of convergence is not an
infeasibility certificate. It holds the active spaces fixed during the solve.

Objectives use SVD singular values or QR determinants rather than forming a
nearly singular full Gram matrix. Active logdet uses QR of `K^(1/2) M V`;
ridge uses augmented QR of `[K^(1/2) M; sqrt(lambda) I]`. These are evaluation
identities, not approximations to the objective. Autograd checks differentiate
`x` with `M` and the selected basis fixed, not rank-selection logic. The reference
accepts explicit float32/float64 inputs; it does not silently promote bfloat16.

| Claims checked | Tests in `tests/test_rank_deficient.py` | Shapes/ranks and acceptance bounds |
| --- | --- | --- |
| LMO optimum, all-maximizer family, minimum norm, zero, projectors, leverage sum, unit active spectrum, full-rank reference agreement | `test_partial_polar_minimum_norm_maximizer_and_projectors` | float64 `7x4`, ranks 0/1/3/4; `rtol=atol=3e-11` |
| Canonical invariance, both lifts and metric projector constraints; ridge/cutoff canonical maps | `test_partial_gauge_lift_and_generalized_active_conditioning` | `9x4`, ranks (2,1), scales `1e-6..1e6`; float64 `3e-11`, float32 `3e-5` |
| Pseudo/active value and map equivalence, basis invariance, autograd/finite differences, Hessian PSD, global shift | `test_pseudodeterminant_active_restriction_gradients_hessian_and_basis_invariance` | float64 `7x4`, ranks (0,0)/(0,2)/(1,2)/(3,1); gradients/Hessian `3e-11`, FD `2e-8` with step `1e-5` |
| Rank target and convergence with unequal/zero ranks, multiple starts | `test_rank_adaptive_balancing_distinct_ranks_and_zero_cases` | float64 `6x4`, ranks (1,2)/(0,1)/(0,0); max residual `<1e-10`, centered solution differences `<2e-8` |
| Infeasible, boundary and nonunique support cases | `test_rank_adaptive_target_still_needs_support_feasibility_and_uniqueness`; `test_rank_adaptive_boundary_target_requires_infinite_scaling` | float64 rank-one examples; infeasible residual `4/3`, positive boundary residual `<1e-9` at large finite scale |
| Uniform target cancels from centered dynamics with EMA | `test_uniform_target_cancels_in_centered_dynamics_even_with_balance_momentum` | float64 `6x4`, ranks (1,2), 12 steps; centered states `3e-11` |
| Ridge gradient, effective dimension, Hessian and loss of exact LMO/projector | `test_ridge_objective_gradient_effective_rank_and_convexity` | float64 `6x4`, ranks 0/1/3; analytic/autograd `3e-11`, FD `2e-8` |
| Correct projected ridge target and fixed-rank zero-ridge limit | `test_ridge_centered_gradient_is_effective_leverage_residual`; `test_renormalized_ridge_limit_equals_pseudodeterminant_on_fixed_rank` | float64 `5x3` and `6x4`, rank 2; gradient `3e-11`, final corrected logdet error `<1e-6` |
| Exact map, projectors, mass and objective discontinuity; ridge continuity | `test_rank_boundary_discontinuity_partial_pseudodet_active_and_continuity_ridge` | float64 `3x2`, epsilon `1e-2,1e-6,1e-12,1e-30`; polar jumps exactly 1/2 within rounding; ridge checked against analytic formula |
| Zero limit, joint ridge limit, normalization cannot remove rank jump | `test_partial_zero_limit_and_noncommuting_ridge_limits`; `test_normalizing_pseudodeterminant_value_does_not_remove_rank_discontinuity` | float64 `3x2`; normalized logdet jump `.7`, relative ridge factor `1/sqrt(2)` |
| Constant-rank continuity and repeated active singular values | `test_continuity_along_fixed_rank_paths_and_repeated_active_singular_values` | float64 `7x4`, rank 2; polar error `<2e-6` at perturbation `1e-6` |
| Cutoff jump, lost LMO value, rank changes under K, positive ties | `test_hard_threshold_adds_discontinuity_and_does_not_solve_original_lmo`; `test_fixed_top_rank_is_ambiguous_at_positive_singular_value_ties` | float64 `3x2`; cutoff jump 1, LMO gap `.05`, tied top-rank jump `>1.4` |
| Explicit rank decision versus dtype/tolerance | `test_explicit_numerical_rank_differs_from_positive_mathematical_rank` | diagonal `5x3`, spectrum (1,1e-4,1e-8); selected rank 3 in float64 at rtol `1e-12`, rank 2 in float32 at rtol `1e-5` |

### Reproduction and measured errors

```sh
python3 -m pytest
python3 -m pytest -s tests/test_rank_deficient.py
python3 -m experiments.rank_deficient
```

The full suite passed before this work (**34 tests**) and after it (**64 tests**,
including 30 new parameterized research cases), with no warnings. The previous
34 cases, including tests demonstrating the unchanged production zero-gradient
behavior, still pass. Environment remains CPU macOS arm64, PyTorch 2.8.0,
Python 3.9.6. As in the audit, that Python is below the declared package minimum
3.10: these are source-tree validations, not supported-version installation
tests. No GPU, mixed-precision training or distributed execution was performed.

| Measurement | Dtype/shape | Observed error or value |
| --- | --- | --- |
| Lifted gauge error, ranks (2,1), scales `1e-6..1e6` | float64 `9x4` | Relative Frobenius error `4.935e-16` |
| Same gauge stress | float32 `9x4` | Relative Frobenius error `3.950e-7` |
| Active/pseudo leverage versus autograd, four rank pairs | float64 `7x4` | Maximum component error `3.886e-16`; central finite-difference error `6.380e-11` |
| Rank-adaptive solve, ranks (1,2), target `.5` | float64 `6x4` | Maximum residual `9.571e-11` at stopping tolerance `1e-10` |
| Ridge at lambda `.2`, true ranks 1 and 3 | float64 `6x4` | Effective dimensions `.758774`, `2.573472`; maximum gradient discrepancy `6.661e-16` |
| Renormalized ridge minus pseudo-logdet, rank 2 | float64 `6x4` | Errors `.024079`, `2.428e-4`, `2.428e-6`, `2.428e-8` at lambda `1e-2,1e-4,1e-6,1e-8` |

The deterministic sweep of (17) with ridge `lambda=1e-4` makes the continuity
tradeoff visible without a plot:

| t | True rank | `||P_0(A(t))-P_0(A(0))||_F` | Partial leverage mass | Ridge map distance | Pseudo-logdet |
| --- | --- | --- | --- | --- | --- |
| `1e-2` | 2 | 1 | 2 | `.707107` | `-9.210340` |
| `1e-6` | 2 | 1 | 2 | `1e-4` | `-27.631021` |
| `1e-12` | 2 | 1 | 2 | `1e-10` | `-55.262042` |
| `0` | 1 | 0 | 1 | 0 | 0 |
| `-1e-12` | 2 | 1 | 2 | `1e-10` | `-55.262042` |

## 9. Proposed specification changes awaiting review

The results justify the following **conditional** replacement statements, not
an automatic production migration:

1. Replace the deficient-rank completion by the minimum-norm spectral-ball
   maximizer (1), with the ball constraint (6) and a declared rank contract.
2. Replace full-column Stiefel/condition-number claims at deficient rank by (5)
   and active condition number one for positive rank, explicitly undefined at
   rank zero. Retain the full-rank reference behavior.
3. Replace leverage mass `n` by `r` and paired target `2n/m` by `R/m`; at fixed
   momentum, use the equivalent pseudo/active objective (12), subject to its
   support-feasibility conditions. This does not promise online convergence.
4. If continuity is required instead, evaluate fixed-ridge contraction and its
   constrained effective-dimension balancing as a different design, accepting
   the loss of exact LMO optimality and active unit singular values.

Review must settle mathematical versus numerical rank, thresholds or structural
rank knowledge, how active spaces change with momentum, rank-boundary behavior,
and which diagnostics/convergence contract is intended. Partial polar resolves
the arbitrary-null-update issue but **does not resolve tiny-positive-singular-value
amplification**. A numerical rank rule can suppress that amplification only by
changing the exact map and introducing another boundary. Distributed ranks must
refer to the global canonical matrix, and low-precision rank decisions need
separate validation. Those choices remain explicit research decisions; the main
optimizer is unchanged.
