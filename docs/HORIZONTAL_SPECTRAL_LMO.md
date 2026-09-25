# Coupled horizontal spectral LMO

Date: 2026-09-24. Research derivation and isolated prototypes, **not a production
optimizer change**. The complete [zero-stratum report](ZERO_STRATUM_GEOMETRY.md),
[rank-deficiency report](RANK_DEFICIENT_THEORY.md), and
[theory audit](THEORY_AUDIT.md) were read for this investigation.

The proposed nuclear-norm dual is correct, with attained strong duality at
every momentum rank. At full-column-rank **dual residuals**, its gradient is
minus the horizontal residual. At deficient rank, primal recovery is a coupled
subgradient feasibility problem: separate partial polars need not work.

Minimizing squared Frobenius norm over the optimal horizontal primal face gives
a unique selection at every rank and exactly zero at zero objective. It can
require fractional null-space contractions, so it need not be a partial
isometry. This is a consequence of imposing horizontality, not a failure of
the accepted single-matrix partial-polar theorem.

Shared positive row weighting can be composed with this LMO, but the old
logdet/leverage identity does not survive. Section 10 derives the actual
weighted value derivative, refutes the old identity, and derives a different
convex objective-contribution balancing candidate. No balancing policy is adopted.

Code: [experiments/horizontal_spectral.py](../experiments/horizontal_spectral.py).
Tests: [tests/test_horizontal_spectral.py](../tests/test_horizontal_spectral.py).
Neither module is imported by `qnormuon/`. Production code and the earlier
research files were left unchanged.

## 1. Domain and the horizontal condition

Let `U,D,A_U,A_D` be real `m x n` matrices, `m>=n>=1`; D uses the transposed
down-weight layout. U,D are balanced canonical **weights**, while A denotes
canonical objective matrices, possibly historical momentum. Weight balance,
momentum rank and dual-residual rank are different properties. Initially K=I.

Use the product Frobenius inner product and norm for pairs. Define

\[
L(P)_i=\langle U_i,P_{U,i}\rangle-\langle D_i,P_{D,i}\rangle,
\quad L^*\lambda=(\operatorname{diag}(\lambda)U,
                         -\operatorname{diag}(\lambda)D),
\tag{1}
\]
\[
\mathcal C=\{P:\|P_U\|_2\le1,\ \|P_D\|_2\le1\},\quad
\mathcal H=\ker L,\quad \mathcal F=\mathcal C\cap\mathcal H.
\tag{2}
\]

At a regular raw row pair `(u,d)`, put `h=||u||/||d||>0`. The H metric is

\[
g_H(v,w)=h^{-1}\langle v_u,w_u\rangle+h\langle v_d,w_d\rangle.
\]

The infinitesimal gauge vectors are `(t u,-t d)`, independently in each row.
Orthogonality to every such vector is exactly
`h^{-1}<u,v_u>-h<d,v_d>=0`. At balance h=1, this is (1)'s equality.
Equivalently substitute `u=sqrt(h)U`, `d=D/sqrt(h)` and the lifted vectors
`v_u=sqrt(h)P_U`, `v_d=P_D/sqrt(h)`. The same equality follows.
The regular tangent-map kernel is precisely the gauge space, as proved in
the zero-stratum report; hence this is exactly the minimum-H-norm horizontal
condition. H selects a horizontal distribution here; the spectral balls impose
an additional norm constraint. This is not the Frobenius-unit Riemannian LMO.

Write `w_i=||U_i||^2+||D_i||^2`. Since `LL*=diag(w)`, the horizontal projection is

\[
\Pi_{\mathcal H}P=P-L^*t,\qquad
t_i=L(P)_i/w_i\ (w_i>0),\qquad t_i=0\ (w_i=0).
\tag{3}
\]

It preserves each first-order delta X, but need not preserve either spectral
ball. Equal delta X does not imply equal finite Euler steps: those have an
additional quadratic factor-product term.

A balanced zero row U_i=D_i=0 has a vacuous equality. The convex problem still
exists, but this does not create a regular H metric or a neuron-birth policy.
A one-sided zero raw row has no finite balanced representative. The previous
zero-stratum restrictions remain in force.

## 2. Primal, dual and attained strong duality

The requested primal value is

\[
v(A)=\max_{P\in\mathcal F}\langle A,P\rangle.
\tag{4}
\]

F is nonempty (contains zero), compact, convex and centrally symmetric.
Thus the finite, nonnegative maximum is attained at every rank.
With sign convention `-lambda^T L(P)`, the Lagrangian is
`<A-L*lambda,P>`. The support function of a spectral ball is the nuclear norm,
so the proposed dual is correct:

\[
\boxed{\phi(\lambda)=\|B_U\|_*+\|B_D\|_*,\quad
B_U=A_U-\operatorname{diag}(\lambda)U,\quad
B_D=A_D+\operatorname{diag}(\lambda)D,}
\tag{5}
\]
\[
\boxed{v(A)=\min_{\lambda\in\mathbb R^m}\phi(\lambda).}
\tag{6}
\]

There is no sign constraint, centering constraint or leverage target on lambda.
It is an equality multiplier, not the old log-weight x.

**Constraint qualification.** In minimization form the objective is linear,
the inequalities `||P_j||_2-1<=0` are finite continuous convex functions, and
the equalities are affine. P=0 satisfies both inequalities strictly and all
equalities exactly. Slater's condition gives zero duality gap and dual
attainment. Norm differentiability, full rank of A or B, and independence of
all displayed equality rows are unnecessary. Equivalently,
`0 in int(C) intersect H` supplies the finite-dimensional relative-interior
qualification. Redundant zero equality rows can first be removed if a
full-row-rank equality formulation is used. See the general theorem in
[Boyd and Vandenberghe, Section 5.2.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).

There is a separate direct attainment check. On active multiplier coordinates
`I={i:w_i>0}`,

\[
\phi(\lambda)\ge\|A-L^*\lambda\|_F
\ge\sqrt{\sum_i w_i\lambda_i^2}-\|A\|_F.
\tag{7}
\]

Phi is continuous and coercive after setting inactive coordinates to zero;
it attains a minimum. Inactive coordinates are free. This proves attainment
independently, while Slater supplies equality of values.

For any feasible P and any lambda the certificate is

\[
\phi(\lambda)-\langle A,P\rangle
=\sum_{j=U,D}(\|B_j\|_*-\langle B_j,P_j\rangle)\ge0.
\tag{8}
\]

Every term must vanish at optimality. This also identifies the entire primal
optimal face without assuming a differentiable dual.

## 3. Full-column-rank residuals: gradient and curvature

If both B matrices have full column rank, their spectral-ball maximizers are
unique: `P_j=polar(B_j)=B_j(B_j^T B_j)^{-1/2}`. Differentiating (5) gives

\[
\boxed{\partial_i\phi=-\langle U_i,\operatorname{polar}(B_U)_i\rangle
+\langle D_i,\operatorname{polar}(B_D)_i\rangle=-L(P)_i.}
\tag{9}
\]

Dual stationarity is exactly horizontality and, by convexity, necessary and
sufficient for dual optimality. At such an optimum the polar pair is feasible,
has zero gap in (8), and is the **unique primal optimizer**. Full rank of A
alone does not establish these differentiability premises for B. Repeated
positive singular values do not destroy matrix-polar or nuclear-norm smoothness
on the full-column-rank domain.

A curvature formula supplies a precise multiplier-uniqueness test. For the
thin SVD `B=Q Sigma V^T` of a full-column-rank matrix and perturbation E, set
`F=Q^T E V`, `N=(I-QQ^T)EV`. Then

\[
D^2\|B\|_*[E,E]
=\sum_j\frac{\|N_{:,j}\|^2}{\sigma_j}
+\sum_{i<j}\frac{(F_{ij}-F_{ji})^2}{\sigma_i+\sigma_j}.
\tag{10}
\]

To derive it, differentiate B=P S and P^T P=I. The skew tangential derivative
in singular coordinates is `(F_ij-F_ji)/(sigma_i+sigma_j)` and the normal
derivative is `N Sigma^{-1}`. Pairing the polar derivative with E yields (10).
Zero curvature is equivalent to

\[
E=P T\quad\text{for a symmetric }T.
\tag{11}
\]

Apply (10) on the two sides with `E_U=-diag(v)U`, `E_D=diag(v)D` to obtain
`v^T Hess(phi) v`. The formula is PSD and valid also at repeated positive
singular values; individual singular-vector derivatives are unnecessary.

## 4. Primal and multiplier uniqueness

At a dual optimum with both residuals full column rank, these are equivalent
after removing inactive zero-row coordinates:

1. Lambda is unique on active coordinates.
2. Hess(phi) is positive definite there.
3. No nonzero active v makes both perturbations in (10) have form (11).

Positive definiteness implies a strict local minimum, hence a unique global
minimum by convexity. Conversely a Hessian null vector gives `B_j=P_j S_j`,
`E_j=P_j T_j` with S_j positive definite and T_j symmetric. For sufficiently
small positive and negative t, S_j+tT_j stays positive definite, so
`||B_j+tE_j||_*=tr(S_j+tT_j)` is exactly affine. The two slopes sum to zero
by stationarity. A nontrivial segment of multipliers therefore minimizes phi.
This converse uses full rank; zero Hessian curvature for a general convex
function would not by itself establish nonuniqueness.

**Counterexample:** with m=2,n=1 and `U=D=A_U=A_D=(1,1)^T`, every
`lambda=t(1,1)`, -1<=t<=1, minimizes phi with value 2sqrt(2). Both residuals
are full column rank when |t|<1. Nevertheless the primal is uniquely
`P_U=P_D=(1,1)^T/sqrt(2)`. At the endpoints a residual is zero but the same
primal is recovered. Residual ranks can depend on which dual optimizer is used.

At arbitrary rank, Section 5's intersection of exposed faces with H is exactly
the primal optimizer set. Primal uniqueness is equivalent to that intersection
being a singleton. Full rank of both residuals is sufficient, not necessary:
the fractional example below has a unique primal. A more general sufficient
test is injectivity of L on the product of free null-block spaces. It is not
necessary, since contraction-ball boundaries can also isolate a completion.

For a precise arbitrary-rank dual condition, fix any optimal primal P*. The
spectral-ball normal cone is

\[
N_{\mathcal C_j}(P_j^*)=
\{P_j^*T:T=T^T\succeq0,\ (I-P_j^{*T}P_j^*)T=0\}.
\tag{12}
\]

Indeed, a supporting B has active singular directions on which P* is an
isometry, so `B=P* (B^T B)^{1/2}` with that support condition. Conversely the
displayed conditions imply `B^T B=T^2` and `<B,P*>=tr(T)=||B||_*`. Therefore

\[
\operatorname{argmin}\phi
=\{\lambda:A_j-(L^*\lambda)_j\in N_{\mathcal C_j}(P_j^*)
\text{ on both sides}\}.
\tag{13}
\]

Dual uniqueness modulo inactive coordinates is exactly the singleton property
of this set. This characterization applies at nonsmooth minima, where a
smooth Hessian argument must not be substituted.

Neither primary uniqueness implies the other. The example above has a unique
primal and nonunique dual. With A=0 and all rows regular, (7) makes lambda=0
the unique dual minimizer, while every P in F is a primal optimizer.

## 5. Deficient residuals: coupled subgradient recovery

For a rank-r residual `B=Q_r Sigma_r V_r^T`, the complete subdifferential is

\[
\partial\|B\|_*=
\{P_0+W:P_0=Q_rV_r^T,\ Q_r^TW=0,\ WV_r=0,\ \|W\|_2\le1\}.
\tag{14}
\]

At rank zero this is the entire spectral ball; at full column rank W=0.
See [Recht, Fazel and Parrilo, equation (2.9)](https://www.mit.edu/~parrilo/pubs/files/RechtFazelParrilo-GuaranteedMinimumRankSolutionsOfLinearMatrixEquationsViaNuclearNormMinimization-SIAM.pdf).
It also follows directly from (8): equality in every active singular direction
forces `P v_k=q_k` and `P^T q_k=v_k`; the remaining orthogonal block is any
contraction. This supplies a direct proof in the present variational setting.

The exact convex sum and affine-composition rules apply since the nuclear
norms are finite continuous convex functions everywhere. Thus

\[
\partial\phi(\lambda)=
\{-L(P):P_U\in\partial\|B_U\|_*,\ P_D\in\partial\|B_D\|_*\}.
\tag{15}
\]

Zero in this set means there exists a **jointly horizontal choice** of
subgradients. It does not assert that the two separate minimum-norm choices
work. For every dual optimizer lambda*,

\[
\boxed{\operatorname{Opt}(A)=
\mathcal H\cap\big(\partial\|B_U^*\|_*\times\partial\|B_D^*\|_*\big).}
\tag{16}
\]

Necessity follows from the two nonnegative gaps in (8); sufficiency follows
by attaining both support values. This set is independent of the chosen dual
optimizer, even when its residual ranks differ.

### Exact fractional-completion counterexample

All weight rows below are balanced and regular, with m=3,n=2:

```text
U = [[1, 0], [sqrt(3)/2, 1/2], [0, 1]]
D = [[1, 0], [0,         1  ], [0, 1]]
A_U = [[1, 0], [0, 1], [0, 0]]
A_D = [[1, 0], [0, 0], [0, 0]]
lambda* = [0, 0, 0].
```

The dual value is 3. Separate partial polars are A_U,A_D, with horizontal
residual `(0,1/2,0)`. Any primal optimizer must have P_U=A_U since B_U is
full rank. Formula (14) forces `P_D=[[1,0],[0,t],[0,s]]`, `t^2+s^2<=1`.
Horizontality forces t=1/2 and s=0. This pair is feasible with objective 3,
proving primal and dual optimality and primal uniqueness explicitly.

The down singular values are **1 and 1/2**, and its Gram is diag(1,1/4),
not a projector. Pair squared norm is 3.25, whereas the residual ranks sum
to 3. These null components are determined by horizontality; they are not
arbitrary Stiefel completions.

For a both-deficient example take `U=D=[[1,0],[0,1],[1,0]]`,
`A_U=[[1,0],[0,0],[0,0]]`, `A_D=[[0,0],[0,1],[0,0]]`.
At lambda=0 both ranks are one and the partial-polar residual is `(1,-1,0)`.
The unique primal pair is `P_U=P_D=[[1,0],[0,1],[0,0]]`, objective 2.
Horizontality forces the missing diagonal entries to one, and the spectral
constraints force the remaining entries to zero. Both outputs are full rank.

## 6. Unique minimum-norm selection at every rank

Define the lexicographic selection

\[
\boxed{P^\dagger(A;U,D)=\operatorname{argmin}_{P\in\operatorname{Opt}(A)}
\tfrac12(\|P_U\|_F^2+\|P_D\|_F^2).}
\tag{17}
\]

Opt(A) is nonempty, compact and convex. The objective is strictly convex on
the product vector space, so a minimizer exists and is unique. This holds at
every momentum/residual rank, including redundant zero weight rows. It proves
uniqueness of the selection, not of the primary primal or dual separately.

In (14)'s coordinates, (17) minimizes `||W_U||_F^2+||W_D||_F^2`, subject to
the two null-block contraction constraints and

\[
L(W_U,W_D)=-L(P_{0,U},P_{0,D}).
\tag{18}
\]

The active norm contribution r_U+r_D is constant. This is a convex quadratic
completion problem. It returns separate partial polars exactly when that pair
is horizontal; otherwise the joint constraint requires nonzero completions.

The correct general Gram and mass statements are

\[
P_j^TP_j=V_{r_j}V_{r_j}^T+W_j^TW_j,\qquad
\|P_U\|_F^2+\|P_D\|_F^2=r_U+r_D+\|W_U\|_F^2+\|W_D\|_F^2\le2n.
\tag{19}
\]

Only residual-active singular directions have forced unit singular values;
free directions can have singular values throughout [0,1]. Full Stiefel
columns, projector Grams, active condition number one, and integer row-mass
targets are not universal properties of the coupled construction. Earlier
partial-polar claims remain valid for their original single-matrix problem.

If A=0, Opt(A)=F and zero is feasible, so **P-dagger=0 exactly**. More generally,
adding L*mu to A changes no horizontal objective, shifts dual minimizers by mu,
and leaves (17) unchanged. Thus a purely vertical nonzero objective also selects
zero. Historical canonical momentum need not be horizontal at current weights.

If both objective rows of a neuron are zero, its selected output rows are
zero: deleting them preserves horizontality and the objective, cannot increase
either spectral norm, and strictly decreases the secondary norm unless they
were already zero. A zero objective row on only one side can acquire an update,
as the fractional example shows.

There is also a useful limit characterization:

\[
P_\epsilon=\arg\max_{P\in\mathcal F}
\{\langle A,P\rangle-\tfrac\epsilon2\|P\|_F^2\},\qquad \epsilon>0.
\tag{20}
\]

Strict concavity makes P_epsilon unique. Comparing its value with P-dagger gives
`0<=v-<A,P_epsilon><=epsilon/2 (||P-dagger||^2-||P_epsilon||^2)<=epsilon n`,
and `||P_epsilon||<=||P-dagger||`. Every zero-epsilon cluster point is primary
optimal with minimum norm; compactness and uniqueness imply convergence to
P-dagger. Fixed nonzero epsilon solves a different problem. No uniform direction
bound near rank loss follows from this value-gap estimate, and the prototype
does not label a finite penalty as the exact lexicographic solution.

## 7. Gauge equivariance and optimizer state

On regular raw rows let H be the unclamped covariant metric. A positive
diagonal gauge gives H'=C^2 H and leaves balanced U,D and correctly transformed
canonical objective A invariant. Consequently F, Opt(A) and the secondary
objective are identical. Uniqueness of (17) gives identical canonical outputs
regardless of multiplier nonuniqueness. The lift obeys

\[
\Delta u=H^{1/2}P_U^\dagger,\quad\Delta d=H^{-1/2}P_D^\dagger,
\qquad\Delta u'=C\Delta u,\quad\Delta d'=C^{-1}\Delta d.
\tag{21}
\]

This proof covers every objective rank and presumes matching canonical state.
It does not extend a clamped ratio or undefined zero-row H into a gauge theorem.
Zero current gradient need not mean zero historical momentum.

For signed C=S|C| with diagonal signs S, canonical `(U,D,A_U,A_D)` transforms
to `(SU,SD,SA_U,SA_D)`. Simultaneously sending both P sides to SP is a bijective
isometry of feasible and optimal sets:

\[
L_{SU,SD}(SP)=L_{U,D}(P),\quad
\|SP_j\|_2=\|P_j\|_2,\quad\langle SA,SP\rangle=\langle A,P\rangle.
\]

The secondary norm is preserved, so `P-dagger'=S P-dagger`. The same lambda
remains dual optimal since B_j'=S B_j. Combining these signs with
`sqrt(H')=|C|sqrt(H)` proves the signed raw laws in (21). No singular-vector
sign choice is needed in the argument.

At a signed reset, canonical first moments must transform `M_U->S M_U`,
`M_D->S M_D`. Cached primal directions and other matrix-valued states must
follow their coordinate laws too. Scalar row states, elementwise squared
moments and invariant log-weights are unchanged; cached lambda may be unchanged.
Raw covector states, if used instead, transform by C^{-1} on up and C on down.
Keeping populated canonical first moments fixed after a sign flip fails, as
the tests show. Positive gauge remains the production convention; no signed
reset or optimizer-state migration code is introduced.

## 8. Continuity and numerical rank

Secondary uniqueness does not imply global continuity. Fix regular
`U=D=[[1,0],[0,1],[1,0]]` and

\[
A_U(t)=A_D(t)=\begin{pmatrix}1&0\\0&t\\0&0\end{pmatrix}.
\tag{22}
\]

Lambda=0 is optimal because the separate polar pair is horizontal. For t!=0
the unique optimum has both second diagonal entries sign(t). At zero the
minimum-norm solution sets them to zero. Pair distance to the zero-rank value
is sqrt(2) for every nonzero t; the one-sided limits differ by 2sqrt(2).
Yet discarding that component loses only 2|t| in objective. Tiny certified
objective gaps therefore cannot certify direction accuracy.

Also `P-dagger(tA)=P-dagger(A)` for t>0 and
`P-dagger(-A)=-P-dagger(A)`. A nontrivial exact selection is discontinuous at
zero objective despite returning exactly zero there. Thresholds or finite
penalties change the exact problem rather than prove its continuity.

For fixed U,D, compactness does give continuity at any A with a unique
**primary** optimizer: every cluster point of nearby selected optimizers
maximizes the limit objective. At a full-rank dual optimum with positive
definite active Hessian, the implicit-function theorem additionally gives
local smooth multiplier and primal dependence. Secondary uniqueness alone
does not permit differentiating through a nonsmooth primary optimizer face.

The separate zero-weight singularity remains: nonzero common scaling of
(U_i,D_i) leaves its horizontal hyperplane unchanged, but at an exactly zero
pair the equation disappears. Coupling momenta does not repair this limiting
feasible-set change or supply missing raw H/birth information.

The face prototype accepts explicit residual ranks, without a hidden cutoff.
An approximately optimized residual can have a tiny singular value that should
be mathematically zero; taking its full polar can cause O(1) horizontal error.
Truncation can instead discard genuinely positive small directions. The generic
primary solver below needs no rank decision, but its value certificate does
not imply a uniform direction or secondary-norm guarantee.

## 9. Prototype algorithms and quantitative comparison

### Generic primary solve

The torch-only reference applies ADMM to P=Z, with
`f(P)=-<A,P>+indicator_H(P)`, `g(Z)=indicator_C(Z)` and scaled multiplier Y:

```text
P <- project_H(Z - Y + A/rho)
Z <- project_C(P + Y)       # clip each matrix's singular values at 1
Y <- Y + P - Z.
```

These are exact proximal subproblems. The functions are proper closed convex
and have an attained saddle point, satisfying the usual two-block ADMM
convergence assumptions. See
[Boyd et al., ADMM, Section 3.2](https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf).
No finite iteration count is asserted exact or uniformly sufficient.

The code normalizes A by its pair Frobenius norm and later restores lambda
and gap scales. This preserves the optimizer set and improves stopping under
objective amplitude changes. It extracts `lambda=(LL*)^+ L(A-rho Y)` in the
normalized coordinates. Every lambda is dual feasible, so the upper bound
does not depend on convergence. The horizontal P is divided by
`max(1,||P_U||_2,||P_D||_2)` to make both spectral constraints feasible while
retaining horizontality. The result reports (8)'s gap, iteration count and a
convergence flag; stopping also checks primal and ADMM step residuals.

This certifies the **primary value**. Starting ADMM at zero does not prove
minimum-norm selection on a nonunique face. That is a separate contract.

### Minimum norm on a known optimal face

Given an exact dual optimizer and its actual ranks, write
`W_j=Q_perp Z_j V_perp^T`. Projection of a matrix T onto (14) is

\[
P_{0,j}+Q_\perp\operatorname{clip}_{\|\cdot\|_2\le1}
(Q_\perp^T T V_\perp)V_\perp^T.
\tag{23}
\]

Use this and (3) in Dykstra's algorithm, initialized at zero, with both
correction variables. It converges to the projection of zero onto (16),
which is precisely (17). Simple alternating projections without corrections
would not generally certify nearest-point selection. The projection/dual
interpretation is discussed in
[Tibshirani, Dykstra's Algorithm, ADMM, and Coordinate Descent](https://arxiv.org/abs/1705.04768).

Exact dual and rank inputs are explicit assumptions. Tests use analytically
certified faces; approximate generic dual outputs are not automatically passed
through a rank guess. Nonconvergence is reported, not treated as an infeasibility
certificate. The last iterate lies on the exposed faces and is approximately
horizontal; its finite-precision gap and residual must be read together.
No scalable automatic rank-face identification algorithm is claimed.

### K=I comparison

CPU float64, seeded balanced 7x3 pair, seed 901. Define horizontal residual
as `max_i |L(P)_i|`, spectral excess as `max(0,max_j ||P_j||_2-1)`, and
Stiefel defect as `max_j ||P_j^T P_j-I||_F`. Stiefel defect alone is not
infeasibility for a spectral-ball problem.

| Construction | Objective | Horizontal residual | Spectral excess | Stiefel defect |
| --- | ---: | ---: | ---: | ---: |
| Separate full polars | 14.725849693 | .951776 | 4.44e-16 | 1.64e-15 |
| Coupled horizontal optimum | 12.652262437 | 1.11e-16 | 0 | 2.96e-12 |
| Post-polar horizontal projection | 11.343174822 | 1.11e-16 | .0920519 | .742962 |
| Projection followed by common feasible rescaling | 10.387028817 | 1.25e-16 | 0 | .851398 |

The separate value exceeds the horizontal optimum by 2.073587255, but its
pair is not horizontal. The feasible rescaled projection loses 2.265233620
relative to the optimum. The raw projected value is not a feasible lower
bound: projection can increase spectral radius despite decreasing total
Frobenius norm. The coupled solve takes 150 iterations, gap 7.60e-12;
the smallest residual singular value is .684898. Both residuals are full rank,
certifying a unique primary pair as well as the minimum-norm selection.

Even when A itself is horizontal, as for current exact loss gradients,
separate polars need not be. Then post-projection preserves the separate
objective exactly: `<A,Pi_H P>=<A,P>`. It can have value **above** the horizontal
optimum because it violates spectral constraints. A separate test covers this
case; horizontal inputs do not turn projection into the exact LMO.

For Section 5's fractional 3x2 example:

| Construction | Objective | Horizontal residual | Spectral excess | Stiefel defect |
| --- | ---: | ---: | ---: | ---: |
| Separate partial polars | 3 | .5 | 0 | 1 |
| Coupled selected pair | 3 | 0 (exact) | 0 | .75 |
| Post-polar projection | 2.875 | 0 | .0687293 | .9375 |
| Projection and feasible rescaling | 2.690110572 | 2.78e-17 | 0 | .953441 |

The face solver reaches the analytic selected pair with Frobenius error
7.28e-12 in 37 iterations. The generic primary solve independently achieves
gap 1.08e-11 in 60 iterations. Its recovered dual has a numerical residual
singular value 1.23e-11; treating it as a genuinely active direction would
misidentify the limiting face.

## 10. Shared K: valid composition, invalid old derivative

Only now introduce `T(x)=K^{1/2}=diag(exp(x/2))`. The direct composition with
the existing convention is

\[
P^\dagger_x=P^\dagger(T(x)A_U,T(x)A_D;U,D).
\tag{24}
\]

H's weights and the two unit spectral bounds are unchanged. For finite
positive K this remains a coupled horizontal LMO and retains positive gauge
equivariance for canonical-invariant x. It also retains signed covariance
since signs commute with T. It optimizes the weighted objective; it need not
optimize the unweighted one. There is no extra T after the output: that would
instead change the spectral constraint to a weighted one, although common row
scaling of both output sides still preserves horizontality.

### Exact counterexample to reusing the old derivative

The old active/pseudo-logdet objective is

\[
\Phi_{\rm old}(x)=\sum_j\log\operatorname{pdet}(A_j^TK(x)A_j)
-\frac{\operatorname{rank}(A_U)+\operatorname{rank}(A_D)}m\,1^Tx.
\tag{25}
\]

Its derivative remains the **separate partial-polar** row mass minus that
target, by the rank report's fixed-active-subspace proof. That derivative
does not introduce horizontal constraints or lambda.

In the fractional example, the uncentered logdet derivative at x=0 is
(2,1,0), whereas the coupled selected row mass is (2,1.25,0). The difference
persists after centering. Thus substituting coupled outputs into the old
leverage gradient is false, also for the accepted active-subspace extension.
Formula (19) refutes substituting a residual-rank target too; that rank can
even depend on the chosen dual optimizer.

Putting optimized residuals B into a logdet does not rescue the old proof:
lambda depends on x, B need not be a fixed row-weighted matrix, its kernel
can change, and additional multiplier-response terms appear. Lambda minimizes
a nuclear-norm sum, not the proposed logdet; its stationarity cannot simply
cancel that different objective's chain-rule terms.

### Actual weighted support derivative

Define the coupled support value

\[
F(x)=\max_{P\in\mathcal F}\sum_i e^{x_i/2}
\big(\langle A_{U,i},P_{U,i}\rangle+\langle A_{D,i},P_{D,i}\rangle\big)
=\min_\lambda\sum_j\|T(x)A_j-(L^*\lambda)_j\|_*.
\tag{26}
\]

At a unique primary optimizer, differentiating the maximum over the fixed
compact feasible set gives the envelope identity

\[
\boxed{\partial_i F=\tfrac12 e^{x_i/2}
\big(\langle A_{U,i},P_{U,i}\rangle+\langle A_{D,i},P_{D,i}\rangle\big).}
\tag{27}
\]

At a nonunique optimum the directional derivative is the maximum of this
expression paired with the direction over all optimal P. Minimum-norm
tie-breaking alone does not ensure differentiability or a unique gradient.

In the smooth full-rank, unique-multiplier regime there is an explicit
coupled-row-mass response formula. Set
`g(x,lambda)=-L(polar(B_U),polar(B_D))` and
`ell(x,lambda)_i=sum_j ||polar(B_j)_i||^2`. Differentiating g=0 gives

\[
\frac{d\lambda}{dx}=-g_\lambda^{-1}g_x,\qquad
\frac{d\ell}{dx}=\ell_x-\ell_\lambda g_\lambda^{-1}g_x.
\tag{28}
\]

The response term is missing from the old projector/logdet calculation.
This Jacobian has no general symmetry guarantee. A float64 5x2 test, seed 902,
verifies full-rank residuals and positive multiplier Hessian, and compares (28)
with re-solved central differences. For `J=I-11^T/m`, it measures
`||J[(d ell/dx)-(d ell/dx)^T]J||_F=.0472614`. This is numerical evidence
against a local scalar potential for even the centered coupled-row-mass field
in that example. The analytic fractional counterexample already disproves
the specific old logdet identity without relying on this numerical observation.

### A different convex candidate, derived without adopting it

A potential does exist for **objective contributions**, distinct from leverage.
Simultaneously flipping both output rows at any neuron preserves spectral
norms and horizontality. Write
`c_i(P)=<A_U[i],P_U[i]>+<A_D[i],P_D[i]>`. Independent row signs imply

\[
F(x)=\max_{P\in\mathcal F}\sum_i |c_i(P)|e^{x_i/2}.
\tag{29}
\]

Each term of this supremum is convex in x. When positive, its logarithm is
a log-sum-exp with nonnegative coefficients and is convex too. Taking the
supremum proves convexity of F and log F on the positive-value domain.
If A is not purely vertical, F is positive for every finite x: T maps
range(L*) bijectively to itself, and F has positive support on any nonzero
horizontal functional. For purely vertical A, F is identically zero, its
logarithm is undefined, and the selected update is zero.

When F>0 define

\[
\boxed{\Psi(x)=2\log F(x)-\frac1m1^Tx.}
\tag{30}
\]

Since `F(x+c1)=exp(c/2) F(x)`, Psi is convex and invariant to global x shifts.
At a differentiable point its gradient is

\[
\partial_i\Psi=\frac{e^{x_i/2}c_i(P^*)}{F(x)}-\frac1m.
\tag{31}
\]

Optimal row contributions are nonnegative: a negative one could be sign-flipped
to increase the objective. They sum to F. Thus this candidate equalizes
**normalized objective contributions**, total mass one, rather than squared
row norms. At nonsmooth points the subdifferential is the convex hull of these
expressions over optimal P. The minimum-norm selection supplies a valid
subgradient, not necessarily a gradient. No rank target enters this derivation.

For an existence check let gamma_i be the best objective over feasible pairs
supported only on row i. It is positive exactly when that row's objective
has a nonzero restriction to its horizontal subspace. If all gamma_i>0,
`F(x)>=max_i gamma_i exp(x_i/2)`, so
`Psi(x)>=max_i x_i-mean(x)+2 min_i log(gamma_i)`. This is coercive on
sum(x)=0 and proves existence of a finite minimizer. It does not establish
uniqueness. If some but not all gamma_i vanish, F is independent of those
inactive coordinates: sending every active x down and compensating on inactive
x to keep the mean zero sends Psi to minus infinity. Equal contribution to
every row is impossible. If all vanish, F=0 everywhere.

This is a derived alternative research objective with its own target and
existence conditions. Tests check (27), (30)'s gradient, its shift identity
and sampled convexity. No balancing iteration or production integration is
introduced, and no equivalence to leverage balance is asserted.

## 11. Tests, reproduction and limits

The full mathematical suite passed before this investigation (**97 cases**)
and after it (**125 cases**, including 28 new parameterized research cases),
without warnings. Earlier tests, including counterexamples for unchanged
production behavior, continue to pass.

| Claim / case | Tests in `test_horizontal_spectral.py` | Checks |
| --- | --- | --- |
| H-horizontal condition | `test_horizontal_condition_is_metric_orthogonality_and_minimum_lift` | 7x3, h=1e-8..1e8; metric equality, vertical orthogonality and delta-X preservation, 1e-12 |
| Strong duality, stationarity and uniqueness | `test_full_rank_strong_duality_stationarity_and_unique_solutions`, three seeds | 7x3; certified gaps, full-rank residuals, polar reconstruction, positive Hessian |
| Gradient and curvature | `test_dual_gradient_and_curvature_formula_against_independent_derivatives` | 6x2; autograd/curvature agreement 1e-11, central gradient difference 2e-8 |
| Nonunique multiplier at full rank | `test_full_column_rank_does_not_imply_unique_multiplier` | 2x1; flat segment, unique primal, Hessian null vector |
| Same selection at different dual ranks | `test_same_selected_primal_at_full_rank_and_deficient_dual_optimizers` | 2x1; ranks (1,1) versus (0,1) |
| One deficient side | `test_one_side_deficient_requires_fractional_completion_and_new_leverage_mass` | 3x2; analytically certified optimum, fractional singular value, mass 3.25 |
| Both sides deficient | `test_both_sides_deficient_separate_partial_polars_are_not_stationary_subgradient` | 3x2; separate subgradient not stationary, jointly completed solution certified |
| Unique secondary selection on nonunique face | `test_nonunique_optimal_face_has_unique_minimum_norm_at_deficient_rank` | 3x2; explicit free completions have squared-norm penalty 2t^2 |
| Zero momentum and vertical objective | `test_zero_momentum_and_pure_vertical_objectives_select_exact_zero`, two cases | 7x3; exact zero output, optional redundant zero weight row and free multiplier |
| Rank-boundary discontinuity | `test_near_singular_exact_selection_is_discontinuous_and_gap_is_not_direction_error`, five t values | 3x2; t down to 1e-30, negative side, constant sqrt(2) jump, 2|t| objective gap |
| Objective homogeneity and covector equivalence | `test_objective_scaling_discontinuity_at_zero_and_vertical_shift_invariance` | 7x3; +/-1e-12 objective scales, unchanged selection under L* shifts |
| Positive gauges and lift | `test_extreme_positive_gauges_canonical_invariance_and_lifted_equivariance` | 7x3; float64 scales 1e-120..1e120, tolerance 3e-8; float32 scales 1e-12..1e12, tolerance 3e-4 |
| Signed state transformation | `test_signed_state_covariance_and_failure_without_momentum_transform` | 7x3 and deficient 3x2; covariance succeeds, frozen-state error >1 |
| Objective and spectral comparison | `test_separate_coupled_and_postproject_comparison`; `test_horizontal_objective_still_needs_coupled_spectral_constraints` | 7x3; arbitrary and already-horizontal A; projected output can violate spectral bounds |
| Old K derivative refuted | `test_shared_K_composes_but_old_logdet_gradient_is_not_coupled_row_mass` | Exact 3x2 counterexample, centered mismatch, finite differences |
| Weighted value derivative | `test_shared_K_value_derivative_is_envelope_not_leverage` | 5x2; finite differences through re-solved optima versus (27), tolerance 2e-7 |
| Coupled row-mass response | `test_coupled_row_mass_jacobian_need_not_be_symmetric_even_after_centering` | 5x2; full-rank residuals, positive Hessian, (28) versus finite differences |
| Derived contribution objective | `test_derived_log_support_balancing_candidate_convexity_shift_and_gradient` | 5x2; normalized contribution mass, sampled Jensen inequality, shift identity, gradient tolerance 2e-7 |
| Numerical limits explicit | `test_solver_limits_and_bad_inputs_are_not_silently_certified` | Iteration-limit nonconvergence and unbalanced inputs rejected |

Float64 solution comparisons ordinarily use relative/absolute tolerance 3e-9.
The generic feasibility/gap acceptance bound is 3e-8, with the gap scaled by
max(1,||A||_F). These are numerical validation tolerances, not rank thresholds
or uniform mathematical direction-error bounds. Finite tests support the
proofs; they do not establish universal properties on their own.

Additional measurements:

| Measurement | Dtype / shape | Observed result |
| --- | --- | --- |
| Primary dual gaps, seeds 901/902/903 | float64 7x3 | 7.60e-12, 4.36e-11, 5.81e-12 |
| Minimum multiplier Hessian eigenvalues, same seeds | float64 7x3 | .585996, .331510, .335230 |
| Lifted error after undoing C | float64 7x3, 240-decade gauge range | Relative Frobenius 1.09e-15 |
| Same gauge test | float32 7x3, 24-decade gauge range | Relative Frobenius 4.50e-7 |
| Weighted support-gradient finite-difference error | float64 5x2 | Maximum component 7.83e-10 |
| Centered row-mass Jacobian antisymmetry | float64 5x2 | Frobenius .0472614 |

Reproduce from the repository root:

```sh
python3 -m pytest
python3 -m pytest -s tests/test_horizontal_spectral.py
python3 -m experiments.horizontal_spectral
```

Environment: CPU macOS arm64, Python 3.9.6, PyTorch 2.8.0. As recorded in the
earlier reports, this Python is below the package's declared 3.10 minimum:
these are source-tree runs, not a supported-version installation validation.
No dependencies were added. Extreme-gauge tests use the earlier scaled-norm
research canonicalization, not the production clamp. The solver then uses
ordinary float32/64 canonical constraint arithmetic; no universal overflow or
underflow guarantee for arbitrarily small/large canonical rows is asserted.
No bfloat16 training, GPU, large-model throughput, distributed rank decisions,
or distributed optimizer-state semantics are certified.

## 12. Consequences for mathematical review

K=I now has a complete variational specification: primal (4), dual (5), joint
recovery (16) and unique secondary selection (17). It is horizontal, spectrally
feasible, gauge equivariant on the regular canonical domain, and zero at zero
objective. It does not generally preserve partial-isometry spectra, integer
leverage mass, or continuity across rank loss. These consequences belong in
any reviewed specification rather than being hidden in a numerical solver.

Production integration still needs a numerical primal-recovery and secondary-
selection contract near deficient dual residuals, accuracy budgets, and the
previously unresolved zero-weight policy. Shared K can be composed as (24),
but reusing the old paired-leverage update would lose its gradient justification.
The contribution potential (30) is derived for comparison, not selected as a
replacement. Production optimizer code remains unchanged.
