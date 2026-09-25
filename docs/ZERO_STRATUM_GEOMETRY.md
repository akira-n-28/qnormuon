# Zero weight rows: quotient geometry, horizontal lifts and neuron birth

Date: 2026-09-24. Research status: the partial-polar extension is provisionally
accepted, as instructed, but no production change is made here. The complete
[rank-deficient report](RANK_DEFICIENT_THEORY.md) and
[theory audit](THEORY_AUDIT.md) were read before this investigation.

**The regular signed quotient is exactly the nonzero rank-one manifold.** The
positive-scaling quotient is a two-sheeted cover of it. At zero, however, the
factor-orbit quotient and the space of neuron functions differ: many distinct
orbits represent the same zero matrix and zero contribution to the network.

**The balanced horizontal lift recovers the `H`-weighted factor metric, but not
an automatic horizontal interpretation of QNorMuon's spectral updates.** Its
inverse scale diverges as `1/sqrt(sigma)`. Separately polarizing the up/down
momentum need not give a horizontal pair; projecting it can change the spectral
constraints and finite-step behavior.

**A rank-one birth direction exists intrinsically in matrix space**, using a
top singular pair of the loss gradient. It is unique as a matrix when the
positive top singular value is simple. It cannot be implemented as a finite
first-order factor velocity at the both-zero pair. Continuous factor-equivariant
update maps cannot birth from any point of the zero fiber. For `n>=2`, a
continuous matrix vector field tangent on the regular rank-one manifold must
also vanish at zero. These are precise obstructions; they do not rule out
discrete matrix-space birth steps followed by a new factorization.

Isolated code: [experiments/zero_stratum.py](../experiments/zero_stratum.py).
Tests: [tests/test_zero_stratum.py](../tests/test_zero_stratum.py).
The main optimizer, earlier reports and existing tests remain unchanged.

## 1. Setting and meaning of function-space equivalence

For one neuron use column vectors `u,d in R^n`, and let

\[
\pi(u,d)=X=du^T,\qquad
g_c(u,d)=(cu,c^{-1}d).
\tag{1}
\]

Unless noted otherwise, `n>=2`, as in the intended SwiGLU setting. Write
`r=||u||`, `s=||d||`, and on the regular set

\[
\sigma=rs>0,\quad a=d/s,\quad b=u/r,\quad X=\sigma ab^T.
\]

The regular factor space is `E_*=(R^n\{0}) x (R^n\{0})` and its matrix image is
`M_1={X: rank X=1}`. Its closure is `V={X: rank X<=1}=M_1 union {0}`.

With a fixed gate row `w_g`, this neuron's output is

\[
f_i(z)=\operatorname{SiLU}(w_g^Tz)\,X_i z.
\tag{2}
\]

Thus equal `X_i` always means equal neuron functions. If `w_g != 0` and equality
is required for every input in `R^n`, the converse also holds: the gate is nonzero
on an open set, so `(X-Y)z=0` on that open set, forcing `X=Y`. For a zero gate,
restricted data support, or multiple interchangeable/canceling neurons, function
identifiability can fail further. This report fixes the gate and neuron identity;
it does not identify the entire network's function quotient with a product of
rank-one varieties.

Primary-source checks: [Mishra et al., *Fixed-rank matrix factorizations and
Riemannian low-rank optimization*](https://arxiv.org/pdf/1209.0430) describes
metric-dependent horizontal representations on fixed-rank quotients.
[Schneider and Uschmajew, Theorems 3.1–3.2](https://arxiv.org/pdf/1402.5284)
distinguishes regular tangent spaces from tangent cones of the closed low-rank
variety. The rank-one formulas, metric comparison and impossibility proofs below
are derived explicitly here rather than assuming a particular quotient metric.

## 2. Exact relationship between fibers and gauge orbits

**Proposition 1.** If `du^T=d'u'^T != 0`, there is a unique `c in R\{0}` such
that `u'=cu`, `d'=c^{-1}d`.

**Proof.** The common nonzero matrix has row space `span(u)=span(u')`, so
`u'=cu` for a unique nonzero scalar. Substituting and multiplying by `u` gives
`c d' ||u||^2=d ||u||^2`, hence `d'=d/c`. The converse follows from (1).

Every nonzero rank-one matrix has such a factorization, so the full nonzero-real
gauge identifies precisely the fibers on `E_*`. The action is free there.
Positive scaling identifies only factorizations with `c>0`; the two positive
orbits over a fixed `X` are represented by `(u,d)` and `(-u,-d)`.

### Smooth quotient, not just a set bijection

Balancing provides explicit coordinates. Let `rho=sqrt(rs)` and define

\[
\bar u=\sqrt{s/r}\,u=\rho b,\qquad
\bar d=\sqrt{r/s}\,d=\rho a.
\tag{3}
\]

The positive quotient is smoothly parameterized by
`(sigma,a,b) in (0,infinity) x S^{n-1} x S^{n-1}`. The signed quotient additionally
identifies `(a,b)` with `(-a,-b)`:

\[
E_*/\mathbb R^*\ \cong\
(0,\infty)\times (S^{n-1}\times S^{n-1})/\{\pm1\}
\ \cong\ \mathcal M_1.
\tag{4}
\]

These are diffeomorphisms. For a concrete smooth local inverse, choose a nonzero
matrix entry `X_ij`, set `d=X_:j`, and set `u=X_i:/X_ij`. The rank-one identities
give `du^T=X`; the construction is smooth on this chart. The free finite sign
action on balanced factors gives the same charts on the quotient. Dimension is
`2n-1`, matching the rank-one manifold.

The signed quotient has no globally continuous choice of balanced **factors**
when `n>=2`. For example, take
`a(t)=b(t)=cos(t)e1+sin(t)e2`, `0<=t<=pi`. The matrices `a(t)a(t)^T` form a closed
loop. Any continuous balanced lift must follow one of the two signs, so it ends
at the negative of its starting factors. A global section would have to end at
the starting factors, a contradiction. Local sign choices are enough for the
smooth quotient; a global factor convention necessarily has a seam.

### Balanced SVD factorization

The sole positive singular value of `X` is `sigma=rs`, with left/right singular
vectors `a=d/s`, `b=u/r`. Equation (3) is precisely

\[
\boxed{X=\sigma ab^T,\quad\bar u=\sqrt\sigma\,b,
\quad\bar d=\sqrt\sigma\,a.}
\tag{5}
\]

It is unique in each positive orbit. On a signed orbit it is unique only up to
the simultaneous sign. This makes the distinction between a unique quotient
point and a unique globally chosen factor pair explicit.

## 3. Zero fiber: different orbits, identical current neuron function

Over the reals,

\[
du^T=0\ \Longleftrightarrow\ u=0\text{ or }d=0.
\]

The zero fiber has three types:

| Factors | Orbit information | Stabilizer | Function through (2) |
| --- | --- | --- | --- |
| `(0,0)` | A single fixed orbit | Entire scaling group | Zero |
| `(0,d)`, `d!=0` | Direction/ray of `d` for positive scaling; line of `d` for signed scaling | Trivial | Zero |
| `(u,0)`, `u!=0` | Direction/ray of `u` for positive scaling; line of `u` for signed scaling | Trivial | Zero |

No finite nonzero gauge maps a one-sided-zero pair to `(0,0)`, changes which
factor is zero, or changes the direction of the surviving factor. These are
genuine distinctions of the **factor orbit space**, not of the current fixed-gate
neuron function. A surviving direction affects which nearby perturbations its
factor chart can express; this is a property of the parameterization and its
optimizer state, not a difference in the current function.

The full orbit space including zero is not Hausdorff: every one-sided-zero orbit
has `(0,0)` in its closure, e.g. `(0,d/c)->(0,0)` as `c->infinity`. An orbit which
is not closed maps to a nonclosed singleton in the quotient. Thus simply allowing
signed gauges does **not** make the full orbit quotient equal to `V` at zero.
The matrix representation additionally collapses the whole zero fiber. Every
continuous gauge-invariant observable into a Hausdorff space takes the same
value on these orbit-closure points.

One can instead complete the balanced section by adjoining `(0,0)` and quotient
by simultaneous sign. Its matrix image is homeomorphic to `V`: as `X->0`,
both balanced factor norms equal `sqrt(||X||_F)->0`, regardless of their directions.
This completion collapses every one-sided-zero factorization to the vertex;
it is a function-preserving identification, **not a finite gauge transform**.
Extending the balanced-factor map to zero in this way is continuous on factor
space, but not differentiable in general and not a repair by an epsilon ratio.

Changing the gate while `X=0` also leaves the current contribution zero, but can
change its future birth gradient. That is another reason to keep the scope
“fixed gate” explicit rather than discarding all dormant-neuron state.

## 4. Tangent map and its kernel

For differentiable factor curves, the product rule gives

\[
D\pi_{(u,d)}(\delta u,\delta d)
=\delta X=\delta d\,u^T+d\,\delta u^T.
\tag{6}
\]

At a regular point its kernel is exactly

\[
\ker D\pi=\{(t u,-t d):t\in\mathbb R\}.
\tag{7}
\]

To prove this, project (6) onto `d^perp` on the left. Since `u!=0`, zero image
forces `delta d` parallel to `d`. Similarly `delta u` is parallel to `u`.
Their coefficients must sum to zero. Equation (7) is the infinitesimal orbit
direction obtained by differentiating `c(t)=exp(t)`. Signed gauges have the
same infinitesimal direction; the extra sign is discrete.

For `X=sigma ab^T`, the image is

\[
T_X\mathcal M_1=\{Z:(I-aa^T)Z(I-bb^T)=0\}
=\{\alpha ab^T+p b^T+a q^T:p\perp a,q\perp b\}.
\tag{8}
\]

The three components are Frobenius-orthogonal and uniquely given by
`alpha=a^TZb`, `p=Zb-alpha a`, `q=Z^Ta-alpha b`. Its dimension is `2n-1`.
The ambient Frobenius tangent projection is

\[
\Pi_T W=aa^TW+Wbb^T-aa^TWbb^T.
\tag{9}
\]

At the zero fiber the kernel statement changes:

* At `(0,d!=0)`, image is `{d v^T:v in R^n}`, rank `n`, and kernel is
  `{(0,w):w in R^n}`. The gauge direction is only one of these kernel directions.
* At `(u!=0,0)`, image is `{w u^T:w in R^n}`, rank `n`, with kernel
  `{(v,0):v in R^n}`.
* At `(0,0)`, the differential is zero: its kernel is all `R^{2n}`, whereas the
  gauge orbit is a point and its tangent is zero.

Thus identifying the kernel with gauge directions is a **regular-stratum
statement**. At zero, factor redundancy and loss of differential information
are larger than the infinitesimal scaling symmetry.

## 5. Minimum-norm horizontal lift at a balanced point

Specify the norm: on balanced factors `u=rho b`, `d=rho a`, `rho=sqrt(sigma)`,
minimize `||delta u||^2+||delta d||^2` subject to (6) for a tangent `Z`.
Using (8), write `delta u=beta b+q/rho`, `delta d=gamma a+p/rho`. The radial
constraint is `rho(beta+gamma)=alpha`. Minimizing `beta^2+gamma^2` gives
`beta=gamma=alpha/(2rho)`. Therefore

\[
\boxed{\delta u_{\rm hor}=\frac{q+(\alpha/2)b}{\sqrt\sigma},\qquad
\delta d_{\rm hor}=\frac{p+(\alpha/2)a}{\sqrt\sigma}.}
\tag{10}
\]

It is unique and satisfies
`u^T delta u=d^T delta d`, precisely orthogonality to the vertical vector `(u,-d)`.
Every other lift adds `t(u,-d)` and increases the squared factor norm by
`2 sigma t^2`. This proves minimality without choosing a pseudoinverse cutoff;
the numerical tests independently compare against the Jacobian pseudoinverse.

The induced quotient metric is

\[
\boxed{g_X(Z,Z)=\frac{\|p\|^2+\|q\|^2+\alpha^2/2}{\sigma}.}
\tag{11}
\]

It differs from the ambient Frobenius metric, which is
`||p||^2+||q||^2+alpha^2`. The nonzero singular values of the balanced Jacobian
are `sqrt(2sigma)` once (radial) and `sqrt(sigma)` with multiplicity `2n-2`
(angular). Its horizontal condition number is `sqrt(2)`, but its inverse
amplification diverges as `sigma->0`:

\[
\frac{\|Z\|_F}{\sqrt{2\sigma}}
\le\|\operatorname{lift}_X Z\|
\le\frac{\|Z\|_F}{\sqrt\sigma}.
\tag{12}
\]

In SVD coordinates, (11) becomes

\[
ds^2=\frac{d\sigma^2}{2\sigma}
+\sigma(\|da\|^2+\|db\|^2)
=2d\rho^2+\rho^2(\|da\|^2+\|db\|^2).
\tag{13}
\]

The vertex is at finite radial distance `sqrt(2sigma)`. Angular directions
collapse there, and an ambient unit matrix velocity costs diverging factor
norm. The singularity is geometric, not just a division-by-zero implementation
detail. No finite positive-definite value of the regular metric at the vertex
extends all these formulas.

## 6. Relationship to QNorMuon's H

At arbitrary nonzero factors, let `h=r/s`. The gauge-invariant extension of
balanced Euclidean factor norm is

\[
g^H_{(u,d)}((\delta u,\delta d),(\delta u,\delta d))
=h^{-1}\|\delta u\|^2+h\|\delta d\|^2.
\tag{14}
\]

Under any nonzero signed `c`, `h'=c^2 h`, `delta u'=c delta u`,
`delta d'=delta d/c`, so (14) is unchanged. It is Euclidean at balance.
Transporting (10) back gives the minimum-(14)-norm lift

\[
\boxed{\delta u=\frac{q+(\alpha/2)b}{s},\qquad
\delta d=\frac{p+(\alpha/2)a}{r}.}
\tag{15}
\]

Equivalently it applies the QNorMuon lift factors `sqrt(h),1/sqrt(h)` to the
balanced-frame vectors. Its horizontal condition is
`h^{-1}u^T delta u-h d^T delta d=0`. The cotangent frame scales oppositely,
giving precisely `sqrt(h) G_u` and `G_d/sqrt(h)` as in the original theory.

This recovers **H as the regular invariant extension of the specified balanced
norm**. The quotient alone does not choose a metric. For example, multiplying
(14) by any positive function of `sigma` retains gauge invariance and the same
minimum-norm lift for a fixed `Z`, while changing its norm and steepest-descent
scale. More general metrics are also possible. H's normalization is justified
only once Euclidean norm at every balanced point is part of the premise.
The extra common multiplier is outside the original theory's normalized
reciprocal-coefficient class, so this observation does not contradict its
norm-local metric uniqueness theorem.

### Horizontal projection versus frozen frame and polar updates

For an arbitrary factor vector `(v_u,v_d)`, define

\[
t=\tfrac12\left(\frac{u^Tv_u}{r^2}-\frac{d^Tv_d}{s^2}\right),\qquad
(v_u,v_d)_{\rm hor}=(v_u-tu,\ v_d+td).
\tag{16}
\]

This preserves its `delta X` and minimizes (14) on the fiber of the tangent map.
At balance the horizontal space is exactly the tangent of the balanced section.
Differentiating (3) gives the horizontal projection of the **frozen** frame
`(v_u/sqrt(h),sqrt(h)v_d)`. Merely freezing H while changing coordinates is not
the full derivative of the parameter-dependent balanced-section map.

For a loss written in X, with `G=grad_X L`, factor gradients are
`G_u=G^T d`, `G_d=G u`. They are horizontal at balance. But separate polar maps
of the entire up/down matrices, or historical momentum at a new point, need not
obey the rowwise horizontal condition. In the `7x3` experiment, initially
horizontal loss gradients produce polar updates with maximum horizontal
residual `.591411`. Removing the vertical part gives up Stiefel defect `.134067`.
Thus this derivation does **not** establish that existing QNorMuon updates are
minimum-norm horizontal lifts, nor authorize projecting them without a new
spectral derivation.

Also, equal `delta X` does not mean equal finite updates:

\[
(d+\eta\delta d)(u+\eta\delta u)^T
=X+\eta\delta X+\eta^2\delta d\delta u^T.
\tag{17}
\]

Even a pure vertical Euler step changes X by `-eta^2 X`; the exact gauge curve
does not. Retraction, post-step balancing and tangent projection are distinct
operations. A post-step gauge reset cannot undo a different X from (17).

### Signed gauges require a state convention

Positive gauges leave the balanced factors and canonical momenta unchanged.
For signed per-neuron scales `C`, let `S=diag(sign(c_i))`. Then canonical factors
and gradients transform by `S`, not by identity. A signed reset requires
`M_U -> S M_U`, `M_D -> S M_D`; shared leverage state is unchanged.
Partial polar respects `P(SA)=S P(A)` because S is orthogonal, restoring the
signed lifted law. Keeping a populated canonical momentum unchanged after a
sign flip generally fails. Passing to the X quotient therefore requires either
this sign-covariant state or state defined directly on that quotient; it is not
already guaranteed by the positive-gauge reset theorem.

## 7. Tangent cone at zero and a principled birth direction

The **Bouligand tangent cone** of `V` at zero is

\[
\boxed{T_0\mathcal V=\{Z:\operatorname{rank}Z\le1\}=\mathcal V.}
\tag{18}
\]

Proof: a limit of `X_k/t_k` with `X_k in V`, `t_k>0` remains rank at most one
because that set is closed. Conversely any such `Z` is realized by `X(t)=tZ`.
This cone is not a vector space for `n>=2`: `e1e1^T` and `e2e2^T` belong, but
their sum has rank two. It is distinct from the tangent of the zero **stratum**,
which is `{0}`, and from the Zariski tangent space at zero, which is the whole
matrix space since all defining 2x2 minors have zero derivative there.

Let `G=grad_X L(0)` exist. A norm must be specified to make “steepest” meaningful.
Use the ambient Frobenius unit cone (for rank one, Frobenius, spectral and nuclear
norms coincide):

\[
\min_{\operatorname{rank}Z\le1,\ \|Z\|_F\le1}\langle G,Z\rangle.
\tag{19}
\]

Write `Z=t ab^T` with unit `a,b` and `0<=t<=1`, absorbing signs into a factor.
The maximum possible `|a^TG b|` is the top singular value `sigma_1(G)`, hence
the minimum in (19) is `-sigma_1(G)`. If `G != 0`, a solution is

\[
\boxed{Z_*=-a_1 b_1^T.}
\tag{20}
\]

When `sigma_1>sigma_2`, this **matrix** is unique despite the simultaneous sign
ambiguity of singular vectors. At a positive repeated top singular value there
are multiple optimal rank-one directions. For `G=0`, choose zero as the unique
minimum-Frobenius-norm member of the optimizer set. A coordinate-based SVD tie
choice is a representative, not a proof of intrinsic uniqueness.

There is a sharper tie obstruction: for `G=s I`, `s>0`, any rule equivariant
under orthogonal changes of input/output coordinates must return a matrix fixed
by `Z -> R Z R^T` for every orthogonal R, hence a scalar multiple of I. For
`n>=2` this cannot be a nonzero rank-one optimum. Gauge invariance alone does not
forbid choosing a coordinate direction; coordinate-free uniqueness does.

Thus (20) obtains a birth in X without choosing a scale gauge. It neither uses
nor remembers the direction of a one-sided nonzero factor in the zero fiber.
Balancing a finite birth `X_new=eta Z_*` can be done with
`d_new=sqrt(eta) a_1`, `u_new=-sqrt(eta) b_1`, up to simultaneous sign. These
are **new factors**, not a finite horizontal velocity at `(0,0)`.

### First-order factor gradients lose birth information

For all factorizations,

\[
G_u=G^Td,\qquad G_d=Gu.
\tag{21}
\]

Both vanish at `(0,0)` even if `G != 0`. At a one-sided zero they provide only
G's action on the surviving vector. They do not determine its top singular
direction: at `u=0,d=e1`, `G1=diag(1,0)` and `G2=diag(1,2)` give the same two
factor gradients but different birth optimizers.

For a minibatch with input `z_b`, upstream output gradient `g_b`, and gate value
`a_i(z_b)`, the needed matrix gradient is

\[
G_i=\sum_b a_i(z_b)\,g_b z_b^T,
\tag{22}
\]

with averaging included in `g_b`. The toy SwiGLU test checks this formula against
autograd. It requires information beyond the current optimizer's `.grad` tensors
at zero; explicitly materializing all `G_i` costs much more than the existing
factor state. No such integration or approximate singular solver is added.

Partial polar also preserves any zero input row: `P=A[(A^TA)^+]^(1/2)` has a
zero row wherever A does. Thus, if the two canonical momentum rows of a zero
neuron are zero, the accepted partial-polar extension supplies no spontaneous
birth there. Existing nonzero historical state is a separate case. This is
consistent with eliminating arbitrary null-space completions, but it does not
recover the missing G information in (22).

At the both-zero pair, the factor Hessian (ordering u,d) is

\[
\nabla^2_{(u,d)}L\big|_0=
\begin{pmatrix}0&G^T\\G&0\end{pmatrix}.
\tag{23}
\]

Its nonzero eigenvalues are `+/-sigma_j(G)`. The ordinary factor critical point
is a strict saddle when G is nonzero: birth information appears in mixed second
derivatives despite vanishing first derivatives. For example a unit factor
direction `(-b_1,a_1)/sqrt(2)` has Hessian curvature `-sigma_1`.

## 8. Exact impossibility statements and their limits

The four requested adjectives are insufficient for a universal yes/no theorem
without specifying the map's output, norm, homogeneity and continuity domain.
The following statements isolate the actual obstructions.

### A. No finite first-order factor lift of a nonzero birth at the origin

By (6), `Dpi_(0,0)=0`. Any differentiable factor path through `(0,0)` has
`u(t)=O(t)`, `d(t)=O(t)`, hence `X(t)=O(t^2)` and `X'(0)=0`.
To realize `X(t)=tZ`, `Z!=0`, its factors necessarily satisfy

\[
\|u(t)\|^2+\|d(t)\|^2\ge2\|d(t)u(t)^T\|_F=2t\|Z\|_F.
\tag{24}
\]

Balanced square-root factors attain this lower bound. They are not differentiable
at zero. This obstruction needs neither an optimizer nor a gauge assumption.

### B. Continuous factor-equivariant update maps cannot leave the zero fiber

Fix auxiliary data that is unchanged by gauge, such as G in X coordinates.
Let a factor-output map `F=(F_u,F_d)` satisfy

\[
F(cu,d/c)=(cF_u(u,d),F_d(u,d)/c),\qquad c>0,
\tag{25}
\]

and be continuous at `(0,0)`. The stabilizer forces `F(0,0)=0`. At `(0,d)`, take
`c->infinity`: the input tends to `(0,0)`, whereas the first output is
`c F_u(0,d)`. Continuity forces `F_u(0,d)=0`. At `(u,0)`, take `c->0` to obtain
`F_d(u,0)=0`.

This proves that any such **finite factor update map** sends the zero fiber
back into the zero fiber. For a velocity map it proves only that its instantaneous
`delta X` is zero there. Euler steps cannot birth; unique locally Lipschitz flows
preserve these invariant sets. With merely continuous non-Lipschitz fields,
uniqueness of ODE solutions must not be assumed.

The theorem already uses only positive gauges and does not need homogeneity under
common amplitude scaling or exact LMO assumptions. An auxiliary state that itself
changes under gauge can invalidate the “fixed data” premise; this must be stated,
not hidden. Returning a newly balanced factorization of a matrix-space birth is
also not a counterexample: its raw output factors do not satisfy (25).

An illustrative equivariant but singular one-sided rule is
`delta u=-G^Td/||d||^2` at `u=0,d!=0`. It gives
`delta X=-Pi_d G` and scales correctly, but its norm can diverge as `d->0`.
It also depends on the surviving direction, so it is not a function of X alone.

### C. Every continuous regular-tangent matrix field vanishes at zero, n>=2

Let `V(X)` be continuous at zero and satisfy `V(X) in T_X M_1` for every
nonzero rank-one X. For every pair of unit a,b, the tangent space along the ray
`X=t ab^T` is independent of t and closed, hence

\[
V(0)\in\bigcap_{a,b}T_{ab^T}\mathcal M_1=\{0\}.
\tag{26}
\]

To prove the intersection identity, for arbitrary vectors p,q choose unit
`a perpendicular p`, `b perpendicular q`, possible when `n>=2`. Equation (8)
implies `p^T V(0) q=0`. Since p,q were arbitrary, `V(0)=0`.

This is stronger than a normalization argument: **no continuous tangent vector
field on the regular manifold can extend to a nonzero first-order cone birth**,
regardless of gauge or homogeneity. Since (19) requires a nonzero direction when
`G(0)!=0`, no field can simultaneously be continuous at zero, tangent on every
regular point, and exact for the noncollapsed cone LMO at zero.

For a concrete linear loss `G=e1e1^T`, the regular Frobenius unit-tangent LMO
along `X=t e1e1^T` is `-e1e1^T`. Along `X=t e2e2^T`, this matrix is not even
tangent. No tie choice along the second ray can give the first ray's limit.

### D. Homogeneity and exact LMOs give additional obstructions

Three scalings must be distinguished:

| Scaling | What it changes | Appropriate law |
| --- | --- | --- |
| `(u,d)->(cu,d/c)` | Same X | Invariance of X outputs; equivariance of raw factor outputs |
| `(u,d)->(s u,s d)`, `s>0` | `X->s^2 X` | An amplitude homogeneity assumption with a specified degree |
| `G->tG`, `t>0` | Gradient magnitude, same linear preference | Exact LMO optimizer sets for a fixed feasible set are unchanged |

If a direction rule satisfies `V(tX)=V(X)` for all `t>0` and is continuous at
zero, then `V(X)=V(0)` on every ray. If it is also regular-tangent, (26) makes
it identically zero. A nontrivial degree-zero amplitude-homogeneous tangent rule
therefore has no continuous extension. Nonzero homogeneity degrees do **not**
obey this conclusion.

Separately, no selection of an exact LMO on a fixed nontrivial symmetric compact
feasible set can be continuous at zero **gradient** in every direction. For a
linear functional with positive support value, optimizer faces at `tG` and
`-tG` lie on opposite, disjoint supporting hyperplanes. If both selections had
the same zero-gradient limit, that limit would lie in both faces, a contradiction.
For (19), the support value is `||G||_2`; the statement is unaffected by the
nonconvex rank-one constraint. Spectral ties cause additional discontinuities at
nonzero G. This zero-gradient theorem is distinct from continuity at zero X.

### E. Why a blanket impossibility for every optimizer would be false

The metric (11) supplies an instructive counterexample to an overly broad claim.
Decompose the ambient G as `g0=a^TG b`, `p_g=(I-aa^T)G b`,
`q_g=(I-bb^T)G^Ta`. Its quotient Riemannian gradient is

\[
\operatorname{grad}_g L
=\sigma\left(2g_0ab^T+p_gb^T+a q_g^T\right)
=\sigma(aa^TG+Gbb^T).
\tag{27}
\]

Its norm squared in (11) is
`sigma(2g0^2+||p_g||^2+||q_g||^2)`. The exact regular **g-unit-ball** LMO is
minus (27) divided by this norm, choosing zero if the gradient is stationary.
For every G its ambient norm obeys

\[
\|V_g(X,G)\|_F\le\sqrt{2\sigma}.
\tag{28}
\]

It is nontrivial, gauge invariant in X, homogeneous of degree `1/2` in X
(degree one under common positive factor scaling), and extends continuously to
`V_g(0,G)=0`. It remains an exact regular-metric LMO. Its feasible unit tangent
balls collapse in ambient coordinates at zero, so it is **not** the exact LMO
over the ambient cone unit ball (19) there. It cannot give a first-order birth.
If the vertex's admissible set is defined instead as the limiting set `{0}`,
the zero direction is an exact LMO there too; this is a different vertex contract.
It also need not be continuous at regular stationary gradients; no such claim
is made. The unnormalized quotient gradient (27) similarly vanishes at zero,
with degree one in X for fixed G.

Zero velocity need not mean uniqueness of an ODE solution: along a descent ray,
the normalized metric field can give `dot sigma=sqrt(2sigma)`, allowing both the
zero solution and delayed departures `sigma=(t-t0)^2/2`. A numerical policy for
such departures is extra information, not supplied by the zero velocity itself.

Finally, a **finite matrix update** can birth continuously without being a tangent
field. For fixed eta, the best rank-one approximation of `X-eta G` solves a
quadratically regularized step over V. It is continuous near `X=0` when the
leading singular value of `-eta G` is simple, and generally nonzero there.
The displacement need not lie in `T_X M_1` for regular X, so (26) does not apply.
It is a proximal step, not an exact unit-ball LMO or a finite factor-horizontal
velocity. The tests include this counterexample to overclaiming impossibility.

## 9. The three domains must remain distinct

| Domain | Correct geometry | Valid claims and limits |
| --- | --- | --- |
| Regular `sigma>0` | Smooth rank-one manifold; signed quotient (4), positive double cover | Unique horizontal lift for a specified metric; H valid; inverse scale `1/sqrt(sigma)` |
| Exact zero `sigma=0` | Singular vertex of the matrix variety for `n>=2`; many original factor orbits | Cone (18), no regular H or nonzero differential lift at `(0,0)`; birth needs X-gradient information and a new transition policy |
| Numerical small nonzero sigma | Still a regular mathematical point | Cutoffs/regularizers are modeling and numerical choices, not a proof that the point lies in the zero stratum |

An invariant amplitude statistic is `sigma=||u||||d||=||X||_F`. Thresholding it
is gauge invariant in exact arithmetic, unlike thresholds on individual factor
norms. But declaring `sigma<tau` to be zero changes a nonzero function and hard
thresholding introduces a jump. The discarded output satisfies
`||f_i(z)|| <= |gate_i(z)| sigma ||z||`; this is a quantitative error bound,
not exact functional invariance. Smooth cutoffs can change continuity properties
but also change the exact LMO and metric. No cutoff is adopted here.

SVD sign seams and repeated leading singular values, finite-precision rank
choices, overflow/underflow of products, and mixed-precision gradients are separate
numerical issues. The research helper uses scaled norms and rejects exact zero
inputs to regular formulas. It does not replace them by epsilon ratios. Its
tangent-residual tolerance checks rounding in an asserted tangent vector; it is
not a zero-neuron policy. Tests cover float64 and float32, not low-precision
training or distributed execution.

For `n=1`, `V=R` is not a singular determinantal variety at zero; the loop and
intersection obstruction (26) do not apply. The factor map still has zero
differential at the both-zero pair, and factor-equivariance obstruction (25)
still applies. These geometric and parameterization singularities must not be
conflated even in this elementary exceptional case.

## 10. Experiments, tests and measurements

The isolated module supplies balanced factors, the tangent map/projection,
minimum-H-norm lifts, quotient metric/gradient, its regular unit LMO, and a
matrix-space birth LMO with a reported singular-value gap. At ties it reports
the absence of a separated leading value instead of claiming a unique direction.
`balanced_birth` returns a finite new factorization; it does not claim an
equivariant factor update or a globally continuous sign section.

| Question/claim | Automated tests in `tests/test_zero_stratum.py` | Main checks/tolerances |
| --- | --- | --- |
| Orbits, signed quotient, balanced SVD | `test_orbit_fiber_balanced_svd_and_positive_double_cover` | n=4; positive/negative c from `1e-6` to `1e6`; float64 `rtol=atol=3e-11` |
| No global balanced sign section | `test_balanced_sign_cover_has_loop_monodromy` | n=2 closed matrix loop, opposite factor endpoints |
| Three zero types; chain rule and missing gradient information | `test_zero_factor_types_have_same_swiglu_function_but_different_gradients`; `test_factor_gradients_do_not_determine_function_space_birth_gradient` | Actual SiLU, n=3, batch 17; ambient/factor autograd and (22) within `3e-11` |
| Tangent map, regular kernel, zero rank changes | `test_regular_jacobian_kernel_and_singular_zero_images` | n=3 Jacobian ranks 5/3/3/0; gauge null direction `3e-11` |
| Minimum-norm horizontal lift and Jacobian spectrum | `test_balanced_horizontal_lift_is_pseudoinverse_and_minimum_norm` | n=4; float64 `3e-11`, float32 `3e-5`; compare independent pseudoinverse and vertical alternatives |
| H invariance/equivariance; full section derivative | `test_H_metric_lift_and_horizontal_projection_are_gauge_equivariant`; `test_balanced_section_derivative_is_horizontal_projection_not_frozen_frame` | c=`1e-6,1e6,-7`; autograd JVP matches horizontal projection, `3e-11` |
| Quotient metric and exact regular LMO | `test_quotient_metric_gradient_and_exact_unit_ball_LMO` | n=3 Riesz identity, unit metric norm, optimum and sampled candidates, `3e-11` |
| Spectral map need not be horizontal | `test_separate_polar_updates_need_not_be_horizontal_and_projection_changes_spectrum` | float64 `7x3`; demonstrate residual and post-projection Stiefel defect `>0.01` |
| Signed reset needs covariant state | `test_signed_gauge_requires_covariant_canonical_state` | float64 `7x3`; row-sign polar covariance `3e-11`, frozen-state mismatch `>1` |
| Partial polar preserves zero momentum rows | `test_partial_polar_preserves_zero_momentum_rows_instead_of_inventing_birth` | float64 `5x3`, rank 2; zero output row within `3e-11` despite possible nonzero X-gradient |
| First-order versus finite update | `test_vertical_changes_leave_first_order_X_fixed_but_change_finite_step` | Pure vertical Euler step gives `(1-eta^2)X`, checked at eta `.1` |
| Near-zero lift divergence and metric-unit contraction | `test_horizontal_lift_blowup_and_vanishing_quotient_unit_direction` | n=2, sigma `1,1e-4,1e-8,1e-12`; compare closed-form norms |
| Cone not a space; continuous tangent intersection | `test_zero_cone_not_vector_space_and_regular_tangent_intersection_zero` | n=2 rank-one sum has rank 2; stacked normal constraints have full rank 4 |
| Birth LMO and square-root factors | `test_birth_cone_LMO_and_square_root_factors` | float64 `4x4`, three seeds; objective `-sigma1`, unit norm, factor product/eta within `3e-11` |
| Ties, coordinate invariance and zero-gradient continuity | `test_birth_ties_and_zero_gradient_preclude_global_continuity` | n=2; top-singular tie perturbations `1e-10` yield direction jump `>1.4`; scale/sign and zero cases |
| Origin critical point still has birth curvature | `test_origin_factor_hessian_contains_birth_information_despite_zero_gradient` | n=3 autograd Hessian and eigenvalues `+/-sigma_j` within `3e-11` |
| One-sided equivariant birth has a singular hidden-direction dependency | `test_equivariant_one_sided_birth_is_singular_and_depends_on_hidden_factor` | n=2; factor scale down to `1e-6`, inverse update scale up to `1e6` |
| No continuous ambient tangent-LMO extension; limits of theorem | `test_no_continuous_regular_ambient_LMO_extension_can_match_birth`; `test_continuous_discrete_rank_one_proximal_birth_is_not_a_tangent_vector_field` | Two regular rays have incompatible tangent limits; proximal step is continuous and nonzero but has normal displacement |
| Numerical policies kept separate | `test_invariant_amplitude_threshold_differs_from_individual_row_threshold`; `test_horizontal_lift_rejects_nontangent_direction_instead_of_silent_projection` | Invariant amplitude vs factor threshold; zero and nontangent regular inputs rejected |

Commands:

```sh
python3 -m pytest
python3 -m pytest -s tests/test_zero_stratum.py
python3 -m experiments.zero_stratum
```

The full suite passed before this change (**64 cases**) and after it (**97
cases**, including 33 new parameterized research cases), with no warnings.
The earlier audit and rank-deficiency tests remain unchanged. Measurements use
CPU macOS arm64, PyTorch 2.8.0, Python 3.9.6; as previously documented, this
Python is below the package's declared 3.10 minimum. These are source-tree runs,
not supported-version installation, GPU or distributed validations.

| Experiment | Dtype / shape | Observed value |
| --- | --- | --- |
| Horizontal reconstruction `||Dpi(lift Z)-Z||_F` | float64, n=4, sigma=2.3 | `3.544e-16` |
| Same reconstruction | float32, n=4, sigma=2.3 | `3.514e-7` |
| Separate polar horizontal residual | float64, up/down `7x3` | Maximum row residual `.591411` |
| Stiefel defect after horizontal projection | float64, projected up `7x3` | `.134067` |
| Birth LMO, seeds 610/611/612 | float64, gradient `4x4` | Objectives `-3.538331`, `-3.188651`, `-3.327439`; top gaps `1.291523`, `1.014872`, `1.393247` |

For `X=sigma e1e1^T` and radial unit tangent `Z=e1e1^T`, the deterministic sweep
reports the following. The birth column uses finite matrix step `eta=sigma`:

| sigma | Minimum factor lift norm | Regular g-unit LMO matrix norm | Balanced birth factor norm |
| --- | --- | --- | --- |
| `1` | `.707107` | `1.414214` | `1.414214` |
| `1e-4` | `70.710678` | `.01414214` | `.01414214` |
| `1e-8` | `7071.067812` | `1.414214e-4` | `1.414214e-4` |
| `1e-12` | `707106.781187` | `1.414214e-6` | `1.414214e-6` |

## 11. Consequences for the next design review

The provisionally accepted partial polar resolves rank-deficient **momentum**
semantics; it does not supply a regular weight metric on the zero fiber. The
matrix representation cleanly separates these issues, but adopting it requires
explicit choices that this investigation does not make:

1. Decide whether the model state identifies the whole zero fiber and signed
   factor sheets, and specify how canonical momentum/state transforms with signs.
2. Specify whether zero neurons remain first-order inactive, or use an explicit
   matrix-space birth transition with its required G information and tie handling.
3. If using horizontal geometry, derive its compatibility with the coupled
   matrix spectral constraints rather than applying a post-polar projection.
4. State whether descent is measured by the regular factor quotient metric or
   an ambient cone norm. They impose different scales and different behavior at
   zero; “exact steepest descent” without this choice is underspecified.
5. Treat small nonzero rows numerically, with quantified approximation errors,
   rather than declaring an epsilon clamp to be an exact singular-stratum theory.

A unique gap-separated X-space birth can avoid an arbitrary **scale gauge**.
It cannot simultaneously be a continuous raw-factor-equivariant update at the
zero fiber or a continuous extension of every regular tangent direction field.
Those limitations are proved above, and no main-optimizer redesign is hidden in
the research prototypes.
