# Smooth polar alternatives to the tall fp64 SVD

This is a **research-only** study. Production `SmoothDual.evaluate` still uses
the fp64 thin SVD. Neither the coupled LMO nor the `3e-5` gap target, `1e-4`
residual-rcond guard, certificate checks, or CPU ADMM fallback changed.

The prototype is [smooth_polar_alternative.py](../experiments/smooth_polar_alternative.py).
The H100 study, shadow replay, and component profiling are reproducible through
[study_smooth_polar_h100.sbatch](../cluster/study_smooth_polar_h100.sbatch),
[shadow_smooth_polar_h100.sbatch](../cluster/shadow_smooth_polar_h100.sbatch), and
[profile_smooth_polar_h100.sbatch](../cluster/profile_smooth_polar_h100.sbatch).
The independent oracle is the existing production full SVD.

## Mathematical information required

For each full-column-rank residual \(B\in\mathbb R^{m\times n}\), \(m\ge n\),
the smooth dual needs the polar \(Q\), the *complete* singular spectrum,
nuclear value, residual rcond, dual gradient, and reusable polar-derivative
action for matrix-free Newton-CG. With \(B=QH\), \(Q^TQ=I\), and \(H\) symmetric
positive definite, \(H=(B^TB)^{1/2}\) in exact arithmetic. Consequently,
\(\sigma(B)=\lambda(H)\), \(\|B\|_*=\operatorname{tr}H\), and
\(\operatorname{rcond}(B)=\lambda_{\min}(H)/\lambda_{\max}(H)\).
The dual gradient remains \(-L(Q)\).

For \(E=dB\) and \(Y=D\operatorname{polar}_B[E]\), set \(\Omega=Q^TY\).
Differentiating \(Q^TQ=I\) makes \(\Omega\) skew. Differentiating \(B=QH\)
and eliminating the symmetric \(dH\) gives

\[
H\Omega+\Omega H=Q^TE-E^TQ,\qquad
Y=Q\Omega+(E-Q(Q^TE))H^{-1}.
\]

The implementation diagonalizes the already-required small SPD \(H\), solves
the Sylvester equation by division by \(\sigma_i+\sigma_j\), and applies
\(H^{-1}\) through that eigenbasis. It neither forms an \(m\times m\) projector
nor decomposes again for each HVP. This is algebraically equivalent to the
existing SVD derivative, including repeated positive singular values. Unlike
the previously studied Gram route, the spectral matrix \(H\) has condition
number \(\kappa(B)\), not \(\kappa(B)^2\).

## Candidate and posterior

The two fp64 deterministic research routes are direct tall QDWH and tall QR
followed by QDWH on the square \(R\). They use the dynamically weighted
Halley coefficients and the stable stacked-QR or Cholesky/solve updates of
[Nakatsukasa, Bai, and Gygi (2010)](https://www.cs.ucdavis.edu/~bai/publications/nakatsukasabaigygi10.pdf).
For \(X_0=B/\|B\|_F\), with lower singular bound \(\ell\), the rational step is

\[
X_+=X(aI+bX^TX)(I+cX^TX)^{-1},\quad
d=[4(1-\ell^2)/\ell^4]^{1/3},\quad
a=\sqrt{1+d}+\tfrac12\sqrt{8-4d+8(2-\ell^2)/(\ell^2\sqrt{1+d})},\quad
b=(a-1)^2/4,\quad c=a+b-1.
\]

The lower-bound recurrence is
\(\ell_+=\ell(a+b\ell^2)/(1+c\ell^2)\). A computed value a few ulps over
one is capped at one. The starting \(\ell=10^{-4}/(2\sqrt n)\) is a valid
lower bound when the true residual rcond is at least \(5\cdot10^{-5}\), since
\(\sigma_{\min}(B)/\|B\|_F\ge\operatorname{rcond}(B)/\sqrt n\). Below that
range the prototype must reject or fall back, not treat its starting bound
as certified. No explicit matrix inverse or positive-singular-value
truncation is used. The maximum is ten iterations.

After QDWH, the candidate forms \(H=\operatorname{sym}(Q^TB)\), never
\(\sqrt{B^TB}\), and diagonalizes \(H\). It rejects nonconvergence, nonfinite
values, non-SPD \(H\), excessive polar orthogonality/reconstruction/symmetry
defects, or rcond too close to the guard. Rejection is an explicit
**decomposition** fallback to the existing smooth full SVD in the research
solver adapter; it is unrelated to CPU ADMM. The defect envelope uses fp64
operation-count factors (`64*(gamma_m+gamma_n)`) and measured residuals.
It is a conservative *experimental screen*, **not** a rigorous bound on
singular-value/rcond/HVP error or on Newton and Armijo decisions for the CUDA
backend. Thus its passing status is deliberately called
`posterior_passed_research_only`.

## H100 numerical evidence

The adversarial study covered 95 distinct matrices per route: Gaussian and
prescribed spectra at 24×8, 96×24, and 1024×384; repeated/clustered top and
bottom values; rcond from 1 down through 1.01e-4, 1e-4, 0.99e-4, and 5e-5;
near-rank-one and wide-dynamic spectra; extreme representable scaling; and
saved real Transformer residuals. Every comparison used independent fp64
thin SVD. Both routes passed the tested posterior on 80/95 cases. The 15
rejections per route were at/below the guard or extremely ill-conditioned:
ten `rcond_uncertain` and five `qdwh_iteration_budget`. There was no observed
unsafe rcond acceptance.

Across accepted cases, the worst polar relative difference was 6.07e-13,
worst individual singular-value relative difference 2.57e-12, worst nuclear
relative difference 6.38e-14, and worst rcond absolute difference 5.66e-14.
At rcond 1.01e-4 on 1024×384 matrices, singular-value relative errors were
about 2.22e-13 (tall route) and 2.17e-12 (QR-square). Repeated spectra were
compared through invariant Q/H and spectral values, not individual eigenvector
columns. The research derivative also matched centered SVD finite differences
on small shapes within the focused test tolerance.

The largest *relative* derivative disagreement, about 3.60e3, occurred for a
smallest-singular-direction perturbation whose exact derivative is essentially
zero; its absolute discrepancy was 5.11e-9. A similar near-null real residual
had relative error 3.06 but absolute error 1.10e-12. For random perturbations,
the worst adversarial relative derivative difference was 2.39e-12. These
near-null directions show why a small polar error alone cannot certify an HVP
or a marginal CG/line-search branch near the rcond guard. This study did not
derive a decision-relevant posterior bound of the kind required for production.

At the identical warm step-25 starting multipliers, counterfactual candidate
solves for all six real pairs matched the SVD oracle's two Newton iterations,
two line-search trials, certificate decisions, and final convergence. The
largest final-direction relative difference was 3.01e-14. This is one real
decision replay, not a proof over the admitted numerical domain.

The locked 50-step **shadow** replay kept full SVD in control of model and
optimizer state and evaluated direct tall QDWH alongside all 931 smooth
residual evaluations. All 931 passed the tested posterior. The worst polar
relative difference was 6.73e-14 and worst sampled HVP relative difference
1.02e-13. The real residuals were materially better conditioned than the
1e-4 guard (sampled step-25 minimum about 1.47e-3), so this shadow evidence
does not close the near-guard decision-safety question. Data/batch hashes and
losses matched the locked baseline exactly because only full SVD controlled
the trajectory. No candidate-controlled 50-step training run was made.

## H100 cost and driver audit

On warmed H100 NVL real [2,1024,384] residuals, synchronized paired medians
were 43.33 ms for the current full thin SVD, 24.95 ms for direct tall QDWH
including H-EVD and posterior, and 26.69 ms for QR-square QDWH. The direct
candidate is about **1.74× faster as a decomposition evaluation**, not as a
whole optimizer. In the 50-step shadow, full SVD totaled 40.58 s and direct
QDWH totaled 23.46 s over the same 931 evaluations (paired medians 43.25 and
25.08 ms; p95 44.20 and 25.16 ms). This suggests a potential 17.12 s
decomposition saving across that replay if production safety and trajectory
control were later established; the shadow itself ran both paths.

Component medians for one real pair, with synchronization in the profiling
path, were: direct QDWH 25.56 ms total, of which two stacked-QR iterations
used 10.43 ms, three Cholesky/solve iterations 3.80 ms, H formation 0.24 ms,
and H-EVD 7.81 ms. QR-square used 27.64 ms total, including 4.40 ms tall QR,
7.98 ms QDWH QR, 3.77 ms Cholesky/solve, 0.23 ms H formation, and 7.82 ms
H-EVD. A paired polar-derivative action (before horizontal contraction) took
about 0.80 ms. Remaining
time includes scaling, posterior reductions, and synchronization. The current
CUDA default full SVD matched explicit `gesvdj` at roughly 22 ms per single
real side; explicit `gesvd` took roughly 40 ms, with similar spectrum/polar.
No better deterministic full-accuracy SVD driver was found; approximate
`gesvda` was not evaluated.

## Decision

QDWH and QR-square QDWH recovered Q, H, spectrum, nuclear value, and HVPs
accurately on this corpus, including cases just above the rcond guard.
The direct route retained a meaningful kernel advantage after the *tested*
posterior. However, that posterior does not bound errors in curvature, CG,
descent, and Armijo decisions near the admitted guard. The large relative
discrepancies in near-null derivative directions are especially relevant to
such a bound. No unsafe branch was observed on the one real counterfactual
replay, but the real shadow states did not probe the critical region.

**Production integration is not justified yet.** The appropriate classification
is *not demonstrated safe across the full production domain*, rather than
an observed failure or a proven safe-and-faster backend. Full fp64 thin SVD
remains the production smooth default and oracle. A separate task would need
a decision-relevant QDWH/H-EVD error posterior and near-guard solver-level
branch replay before considering an adaptive integration with explicit full-SVD
fallback reasons. This conclusion does not change the mathematical LMO.

Full H100 regression after the research additions: **325 passed, 0 failed, 0
skipped**; no outbound Python network attempt was observed. Focused tests are
in [test_smooth_polar_alternative.py](../tests/test_smooth_polar_alternative.py).
