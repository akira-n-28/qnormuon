# The regular quotient spectral geometry

Date: 2026-09-24. The K=I coupled horizontal LMO is provisionally accepted;
this report formalizes its geometry without changing production code. The
complete original specification and the preceding [audit](THEORY_AUDIT.md),
[rank report](RANK_DEFICIENT_THEORY.md), [zero-stratum report](ZERO_STRATUM_GEOMETRY.md)
and [horizontal LMO report](HORIZONTAL_SPECTRAL_LMO.md) were read.
Shared K and contribution balancing are outside this investigation.

The construction defines a **continuous, reversible norm on the tangent bundle
of the regular positive-gauge quotient**. In terminology that permits continuous
fiber norms it is a reversible C^0-Finsler structure. It is generally neither
smooth away from the zero section nor strongly convex. The horizontal connection
and balanced section are part of its definition; gauge invariance does not make
it the unique possible geometry of the function space.

## 1. Base manifold and tangent representation

Keep neuron identities and gates fixed. The regular factor space consists of
real pairs u,d in R^(m x n), with both factors of every row nonzero. Quotient
by the free positive diagonal action `(u,d)->(Cu,C^{-1}d)`. Its global balanced
section is

\[
\mathcal B=\{(U,D):\|U_i\|=\|D_i\|>0\text{ for all }i\}.
\tag{1}
\]

It is a smooth embedded manifold: the gradients of the m balance constraints
`(||U_i||^2-||D_i||^2)/2=0` have disjoint row supports and are nonzero. The
positive quotient is diffeomorphic to B, with dimension m(2n-1). In particular,

\[
T_{(U,D)}\mathcal B=\mathcal H_{U,D}=\ker L,\qquad
L(P)_i=\langle U_i,P_{U,i}\rangle-\langle D_i,P_{D,i}\rangle.
\tag{2}
\]

This is the horizontal representation from the H metric, not an assumption
that arbitrary factor velocities are already tangent to the balanced section.
The regular quotient by signed nonzero scalars further identifies simultaneous
row signs and is the product of nonzero rank-one matrix manifolds. The norms
below also descend through those discrete signs since left row-sign matrices
are orthogonal. The positive convention remains the default.

This quotient need not equal the entire network's function quotient: zero gates,
neuron permutations, restricted data and cancellations can identify more points.
No statement here extends the regular bundle over the singular zero-weight
stratum or chooses a neuron-birth transition.

## 2. A norm on each horizontal tangent space

On E=R^(m x n) x R^(m x n) let

\[
N(P)=\max(\|P_U\|_2,\|P_D\|_2),\qquad
\|P\|_Q=N(P)\quad(P\in\mathcal H).
\tag{3}
\]

Both operator norms are nonnegative, absolutely homogeneous and subadditive.
Their maximum vanishes exactly when both matrices vanish, and
`max_j ||P_j+R_j||_2 <= max_j ||P_j||_2 + max_j ||R_j||_2`.
Restriction to the linear space H preserves all these properties, proving
that (3) is a norm, not a seminorm. Its unit ball is exactly the accepted
horizontal primal feasible set. In the tall setting,

\[
\frac{\|P\|_F}{\sqrt{2n}}\le\|P\|_Q\le\|P\|_F,
\tag{4}
\]

where the Frobenius norm is on the pair. Thus the ball is compact, convex,
centrally symmetric and has nonempty interior relative to H.

The spectral norm couples neurons through whole matrices; this is not a sum
of independent neuron tangent norms. In matrix coordinates `X_i=d_i u_i^T`,
lift each tangent delta X_i by the balanced minimum-H-norm lift from the
zero-stratum report, stack those factor lifts, and apply (3). This describes
the same quotient norm without choosing a raw scale gauge. The simultaneous
sign ambiguity of balanced factors cancels from the norm.

### A different construction must not be conflated with (3)

One could instead put the ambient norm N on E and define the Banach quotient
norm `inf_{V in range L*} N(P+V)` on factor tangent classes. That is generally
**different** from taking the unique H-horizontal representative and using N.
Frobenius orthogonal projection onto H need not contract the operator norm.

The fractional 3x2 example in the horizontal report gives an explicit witness.
For its separate partial-polar pair R, N(R)=1 but
`N(Pi_H R)=1.068729304...`. Both represent the same quotient velocity since
R-Pi_H R is vertical. The infimum-over-lifts norm is at most 1, while the
chosen horizontal norm is greater than 1. This investigation adopts the
user-specified horizontal norm; it does not silently replace it by that infimum.

## 3. Cotangent space as a quotient of objective matrices

Use the Frobenius pairing to identify E* with matrix pairs. Restriction to H
is a surjective linear map E* -> H*: every functional on H has an extension,
for instance its Euclidean representative in H. Its kernel is the annihilator

\[
\mathcal H^\perp=\operatorname{range}(L^*),\qquad
L^*\lambda=(\operatorname{diag}(\lambda)U,-\operatorname{diag}(\lambda)D).
\tag{5}
\]

Indeed, range(L*) annihilates ker L by adjointness. On the regular section
`LL*=diag(w_i)` with `w_i=||U_i||^2+||D_i||^2>0`, so rank L=m and both spaces
have dimension m. Equivalently apply the fundamental theorem of linear algebra.
Therefore

\[
\boxed{T^*_{[u,d]}\mathcal Q\cong E^*/\operatorname{range}(L^*),\qquad
\langle[A],P\rangle=\langle A,P\rangle.}
\tag{6}
\]

The pairing is independent of the representative A. A nonzero vertical pair
represents the zero cotangent vector. The canonical Euclidean representative
is `A_h=Pi_H A`, but the minimum **nuclear-norm** representative need not be
A_h. Those two choices have different optimization objectives.

## 4. Dual norm, attained infimum, and support value

The ambient dual of N is the sum of nuclear norms:

\[
N^*(A)=\sup_{N(P)\le1}\langle A,P\rangle
=\|A_U\|_*+\|A_D\|_*.
\tag{7}
\]

This follows from independently maximizing over the two operator-norm balls;
each support value is its matrix's nuclear norm. By symmetry the absolute-value
definition of a dual norm has the same supremum. Restricting the primal norm
to H gives

\[
\begin{aligned}
\|[A]\|_{Q,*}
&=\max_{P\in\ker L,\ N(P)\le1}\langle A,P\rangle\\
&=\min_{\lambda\in\mathbb R^m}
\left(\|A_U-\operatorname{diag}(\lambda)U\|_*
+\|A_D+\operatorname{diag}(\lambda)D\|_*\right).
\end{aligned}
\tag{8}
\]

For a direct proof, introduce `-lambda^T L(P)` in the Lagrangian. Maximizing
over the product of balls gives (7) evaluated at A-L*lambda. P=0 strictly
satisfies both norm inequalities and exactly satisfies the affine equalities,
so Slater's condition yields equality and dual attainment. Alternatively,
the finite-dimensional norm-preserving extension theorem for a functional on
H yields an ambient extension with the same dual norm; all extensions differ
by (5), proving the same infimum identity. The concrete coercivity bound is

\[
N^*(A-L^*\lambda)\ge
\sqrt{\sum_i w_i\lambda_i^2}-\|A\|_F.
\tag{9}
\]

It proves that the infimum is attained on the regular section. No full-rank
assumption on A or the minimizing residuals enters these arguments.
The finite-dimensional convex duality qualification is the one used in
[Boyd and Vandenberghe, Section 5.2.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).

For completeness this is a genuine norm on cotangent classes. Homogeneity,
symmetry and the triangle inequality follow from the support representation.
If [A]!=0, then A_h!=0. Taking P=A_h/N(A_h) gives positive value
`||A_h||_F^2/N(A_h)`, so the dual norm is positive. Conversely [A]=0 annihilates
all P in H and has norm zero. Adding L*mu to A translates minimizing lambda
by mu and leaves the norm unchanged. Equation (8) is exactly the support
value of the accepted coupled horizontal spectral LMO.

## 5. Exact steepest descent and the selection convention

For a differentiable gauge-invariant loss ell, let [A] be its differential
at the current quotient point. Then

\[
\min_{P\in\mathcal H,\ \|P\|_Q\le1}d\ell[P]
=-\|[A]\|_{Q,*}.
\tag{10}
\]

If [A]!=0 every minimizer has Q norm one: an interior nonzero direction
with negative objective could be scaled to improve it. If [A]=0 every
direction in the unit ball is minimizing. Thus minus the accepted horizontal
LMO output is exact **unit-ball steepest descent**, and a radius-eta linear
step has value `-eta ||[A]||_{Q,*}`. This is a local linear statement; a finite
step still needs step-size control to decrease a nonlinear loss.

One alternative scaling solves
`min_V <A,V> + ||V||_Q^2/(2 eta)`. Its optimizers are
`V=-eta ||[A]||_{Q,*} P`, where P is a maximizing unit-ball direction,
and its optimum value is `-eta ||[A]||_{Q,*}^2/2`. This follows by optimizing
the scalar radius after the dual-norm inequality. It differs from the
unit-radius LMO step with magnitude eta; the norm does not force a learning-rate
or amplitude convention.

For canonical EMA momentum A, the same optimization is exact for the momentum
covector's linear surrogate. Momentum is not generally the current loss
differential, so a claim of exact steepest descent of the **current loss**
requires A to be that differential. The unchanged production optimizer still
uses separate polars and has not yet implemented this accepted coupled step.

Among maximizing unit-ball directions, the auxiliary rule

\[
P^\dagger=\arg\min_{P\in\operatorname{Opt}([A])}\|P\|_F^2
\tag{11}
\]

exists and is unique because the optimal face is compact convex and the
auxiliary objective strictly convex. It is intrinsic relative to the balanced
Euclidean/H convention and invariant under positive gauge and signed row
isometries. It is independent of which dual minimizer is used to describe
the face and is exactly zero when [A]=0. It changes neither (3) nor (8).
The spectral norm alone would permit other maximizing directions at a tie;
the Frobenius tie-break is extra structure, not part of the norm's definition.
Secondary uniqueness does not remove direction discontinuities at rank changes.

## 6. Raw-coordinate pullback and exact positive gauge invariance

Write raw weights u,d and `H=diag(||u_i||/||d_i||)>0`. Let
`U=H^{-1/2}u`, `D=H^{1/2}d`. At a fixed base point the frame map is

\[
J_H(v_u,v_d)=(H^{-1/2}v_u,H^{1/2}v_d).
\tag{12}
\]

On raw H-horizontal velocities the norm is

\[
\|v\|_{Q,\mathrm{raw}}=
\max(\|H^{-1/2}v_u\|_2,\|H^{1/2}v_d\|_2).
\tag{13}
\]

For an arbitrary raw velocity representing a quotient tangent, the derivative
of the balanced section is `Pi_H J_H v`, where Pi_H denotes projection onto
the **canonical** horizontal space (not multiplication by the metric H).
Thus the full pullback is `N(Pi_H J_H v)`. It is a seminorm on all raw factor
velocities, with kernel the vertical gauge directions, and a norm on tangent
classes. Merely freezing H without the projection in (12) would not give
the derivative of the weight-dependent balanced-section map.

Under a positive gauge `u'=Cu,d'=C^{-1}d`, velocities transform the same way
and H'=C^2 H. Consequently `J_H' v'=J_H v` exactly; U,D and their horizontal
projection are unchanged. Both (13) and its quotient-class version are
therefore exactly gauge invariant. The inverse horizontal lift is
`v_u=H^{1/2}P_U`, `v_d=H^{-1/2}P_D`, giving the usual equivariant raw updates.
For raw covectors G the paired objective is represented by
`A=(H^{1/2}G_u,H^{-1/2}G_d)`, which preserves the pairing on horizontal lifts.
This uses the metric once, not twice.

These statements require regular nonzero rows and exact covariant norms.
Clamping, finite-precision cancellation and overflow are numerical policies,
not extensions of the regular theorem to zero rows.

## 7. Precise geometric terminology

The spaces ker L form a smooth vector subbundle because L has constant rank m
on B. Its fiber norm is the restriction of the ambient continuous operator
norm, with bounds (4). In smooth local bundle charts it is continuous and
locally Lipschitz. It is positive on all nonzero tangent vectors and reversible.
Hence “continuous normed tangent bundle” is unambiguous. “Reversible C^0-Finsler
structure” is also correct under the explicit convention of a continuous
tangent-bundle function that restricts to a norm on every tangent space;
that convention is used, for example, in
[A large family of projectively equivalent C^0-Finsler manifolds](https://www.jstage.jst.go.jp/article/tmj/72/3/72_1601085624/_article/-char/en).

Unqualified smooth/strongly convex Finsler terminology would be misleading.
At regular U=D, pairs with identical sides `P_U=P_D=[[1,0],[0,t],[0,0]]`
are horizontal. Their norm is max(1,|t|), which has a kink at t=1. Distinct
points with |t|<1 and their midpoint all have norm one, so strict convexity
also fails. No positive-definite smooth fiber Hessian of half the squared
norm exists in general. This is not a Riemannian metric, and the regular
theory does not automatically give a unique smooth geodesic spray or a
single-valued smooth Legendre map.

It is not merely a sub-Finsler distribution on the quotient: after passing
to the quotient H represents the **entire** tangent space. On the unreduced
factor space it can instead be viewed as a norm on the chosen horizontal
distribution, but that is a different description of the base space.
The regular C^0 structure gives lengths of piecewise differentiable quotient
curves and a locally nondegenerate length distance by local norm equivalence.
It makes no completeness or smooth geodesic uniqueness assertion, and no
regular tangent-bundle extension across the zero stratum is supplied.

## 8. Verification

The isolated [norm helpers](../experiments/quotient_spectral.py) and
[tests](../tests/test_dual_solver.py) check the norm axioms on horizontal
vectors; annihilator equivalence; dual support and steepest direction; the
distinction from infimum-over-lifts; explicit nonsmoothness and non-strict
convexity; and raw gauge invariance with scales 1e-120 through 1e120.
Zero raw rows are rejected by the regular formulas. These checks use float64
and typical tolerance 2e-9, with exact algebraic examples where possible.
The companion [solver study](DUAL_SOLVER_STUDY.md) covers numerical dual norms,
matrix-free derivatives, float32 work and limitations of value certificates.
