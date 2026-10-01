# Full-rank posterior direction admission

**Classification C: NO PRINCIPLED DIRECTION BUDGET.** A support-gap direction
theorem and a conditional fp64 posterior exist. They contain every measured
offline reference error, including the five below-guard Stage-D fixtures.
However, no production accuracy budget follows from the available observations,
and the bounds are substantially looser than observed errors. This study does
**not** justify production-v1 admission or resumed training.

Production sources, the `1e-4` guard, all five certificate thresholds, and
fallback semantics are unchanged. No model was instantiated or trained; only
saved tensor problems were solved under SLURM. No downloads or ADMM runs occurred.

## Exact support-gap theorem

All norms below are Frobenius except explicitly indicated operator norms.
Let a real `m x n` matrix B, m >= n, have full column rank, thin SVD
`B=U Sigma V^T`, polar `Q=U V^T`, and `alpha=sigma_min(B)>0`.
For any P with `||P||_2<=1`, put `c_i=u_i^T P v_i`. Each c_i<=1.
Consequently

```
||B||_* - <B,P> = sum_i sigma_i (1-c_i)
                 >= alpha (n-<Q,P>)
                 >= alpha/2 ||Q-P||_F^2.
```

The last inequality follows from `||Q||_F^2=n` and `||P||_F^2<=n`.
This is a support-function argument, not strong convexity of the nuclear norm.
Repeated positive singular values are allowed. A positive alpha is essential.

For a feasible horizontal pair P and any multiplier lambda whose two residuals
are full column rank, horizontality gives `<A,P>=<B,P>`. Write
`g_j=||B_j||_*-<B_j,P_j>`, `G=g_U+g_D`, and `alpha_j=sigma_min(B_j)`.
Then

```
sum_j alpha_j ||P_j-Q_j||_F^2 <= 2G,
||P-Q||_F^2 <= 2 sum_j g_j/alpha_j <= 2G/min_j alpha_j.
```

Let P* be **any** exact primal optimum. Weak duality implies
`0<=G*=dual(lambda)-primal(P*)<=G`. The same argument applies to P*.
The triangle inequality therefore proves, without assuming multiplier
uniqueness or that the current lambda is optimal,

```
||P-P*||_F <= 2 sqrt(2G/min_j alpha_j).
```

A sharper computable side-specific version is
`sqrt(2 sum_j g_j/alpha_j)+sqrt(2G/min_j alpha_j)`; take the smaller bound.
The weighted version is `sum_j alpha_j ||P_j-P*_j||_F^2<=8G`.
These bounds hold for every optimum, not just distance to the optimal set.
At an exact full-rank dual optimum, Q=P* and the sharper bound
`sqrt(2G/alpha)` applies. That special-case constant must not be used at an
arbitrary current multiplier.

## Numerical model and feasible proxy

Frozen tensor inputs and the floating-point multiplier define the exact real
problem. Assume normal, finite IEEE fp64 operations with unit roundoff
`u=2^-53`, `gamma_k=ku/(1-ku)`, standard GEMM/dot-product error bounds, and
no unmodeled underflow/overflow. These are conditional operation-model bounds,
not directed interval arithmetic or a backend-specific formal proof.

Orthogonalize the computed thin-SVD factors mathematically to define a nearby
matrix with exactly their listed singular values. Measured reconstruction and
orthogonality defects, inflated for GEMM/reduction rounding, bound its distance
from the exact residual by eta_j. Residual construction adds
`gamma_3 || |A_j|+|L*(lambda)_j| ||_F`. Weyl gives
`alpha_lower_j=sigma_min_listed_j-eta_j`. The nuclear upper bound is
`sum_j sum_i sigma_ji + sqrt(n) sum_j eta_j`, plus scalar-reduction rounding;
the sqrt(n) follows from the nuclear/Frobenius norm inequality. Cached internal
factors are usable only at their matching evaluation; reconstruct against the
original residual to include centering/whitening conversion error.

The returned floating-point pair p is not assumed exactly feasible. Let
`h>=||L(p)/sqrt(w)||_2`, including dot-product and w rounding. Its exact
horizontal projection q differs by at most h. Reuse the existing conservative
Gram norm bounds and radial scale, allowing final division rounding, to obtain
`rho>=max_j ||p_j||_2`. Put `t=max(1,rho+h)` and `Pc=q/t`. This defines an
exact feasible proxy without claiming that a rounded projection is exact.

```
delta_feas = h + (1-1/t)(||p||_F+h) >= ||p-Pc||_F,
primal_lower = fl(<A,p>) - dot_round - ||A||_F delta_feas,
G_upper = dual_upper-primal_lower,
direction_error_upper = delta_feas + 2 sqrt(2G_upper/min alpha_lower).
```

If either alpha_lower is nonpositive, no full-rank direction certificate is
issued. Per-side support bounds may sharpen the formula above, with their
own construction/dot/feasibility allowances. The certificate does not contain
an offline reference solution or a fitted safety multiplier.

## Coordinate interpretation

Squared canonical Frobenius error equals squared raw error in the regular H metric:
`||H^-1/2 delta_up_error||_F^2+||H^1/2 delta_down_error||_F^2`.
It also bounds the horizontal quotient spectral error because each operator
norm is at most its Frobenius norm. Raw Euclidean error has an additional
factor `max_i(sqrt(h_i),1/sqrt(h_i))`; it is gauge dependent. Final model-dtype
casting adds its own measured or bounded canonical error. Certification before
casting cannot certify an unchanged low-precision direction.

Canonical error, optionally normalized by `sqrt(2n)`, is the appropriate
gauge-invariant admission quantity. Neither this theorem nor model casting
determines an application accuracy budget automatically.

The quotient-norm statement concerns the exactly horizontal proxy Pc and P*.
The returned floating-point p, and especially its model-dtype cast, can contain
a small vertical component; do not call a product spectral norm on that
nonhorizontal pair the defined horizontal quotient norm. Add delta_feas when
bounding the returned tensor itself. A learning rate multiplies the direction
error, and the regular raw lift adds its own arithmetic/casting error.

## Auditable numerical allowances

Code is isolated in [full_rank_direction.py](../experiments/full_rank_direction.py).
For a computed thin decomposition `(F,s,Vt)`, the factors are not assumed
orthonormal bitwise. Let du,dv bound their exact Gram defects after adding
`gamma_m ||F||_F^2`, `gamma_n ||V||_F^2`, identity-subtraction and norm-reduction
rounding. For du,dv<1 their mathematical polar orthogonalizations satisfy

```
||F-Fhat||_2 <= eu = du/(1+sqrt(1-du)),
||V-Vhat||_2 <= ev = dv/(1+sqrt(1-dv)).
```

The exactly orthonormalized-factor matrix `Bhat=Fhat diag(s) Vhat.T` has exactly
the listed singular values. If rho bounds `||Bcomputed-F diag(s) Vt||_F`, then

```
||Bexact-Bhat||_F
 <= eta = construction_round + rho
          + ||s||_2 (eu sqrt(1+dv)+ev).
```

Here `||s||_2` is the Euclidean norm of the spectrum. Rho includes rounding in
column scaling, reconstruction GEMM, subtraction, and norm reduction. The short
positive scalar formulas receive gamma allowances for their operation counts;
there is no empirical safety inflation fitted to these fixtures. Weyl uses
this Frobenius upper bound also as an operator upper bound. Nuclear Lipschitz
continuity gives `||Bexact||_*<=sum(s)+sqrt(n) eta` per side.

For primal repair, the computed row residual has the componentwise allowance
`gamma_(2n+3) sum_k(|U_ik p_Uik|+|D_ik p_Dik|)`, with positive sums inflated
for their own rounding. A lower bound for w gives h. The radial bound is the
**same matching** `gram_upper` result divided by the same scale, augmented for
scalar and componentwise division rounding. It is not an unguarded SVD estimate.
The primal dot uses `gamma_(2mn+2) sum|A_ijk p_ijk|`; its inputs are exact frozen
data. Norms/reductions used in the allowances are themselves inflated. Negative
inconsistent gap/support bounds or unrepresentable quantities raise, rather
than declaring convergence. Nonpositive alpha_lower gives an infinite
direction bound. No positive singular value is truncated.

For the sharper two-side bound the code computes upper bounds on each support
gap at Pc, including residual-construction error, dot rounding, and
`||B_j||_F delta_feas`. It combines those with the global G_upper and takes the
minimum of the two valid formulas. These are bounds for the represented
canonical problem, not for an unavailable unrounded EMA or infinitely precise
canonicalization of the raw model. Those are separate input perturbations.

## Fixed fixtures and solver progress

The five inputs and high-accuracy reference multipliers are exactly those in
[NEAR_RANK_OPTIMUM_CLASSIFICATION.md](NEAR_RANK_OPTIMUM_CLASSIFICATION.md).
The prior independent L-BFGS endpoints are checked again in original coordinates
using complete fp64 SVD, gradient/horizontality, and independently recovered
primal/dual values. They are **numerical high-accuracy references**, not formal
interval-certified exact optima; observed errors below are comparisons with
those references. Their residual gradients are approximately `1e-13` and the
prior independent CPU decompositions agree. None enters the posterior itself.

Each continuation starts at the captured original warm lambda. The research
configuration alone admits every positive computed spectrum, disables fallback,
and uses the prior strict `1e-10` diagnostic target to expose later iterates.
Production full-SVD Newton/HVP, damping, 24-trial line search, objective, and
recovery are unchanged. All five strict continuations end in the already-known
`line_search_failed` limitation, **not** rank collapse. The production target
would stop earlier if its admission guard were bypassed. There is no proposed
strict-target solver change here.

Bounds normalized below use the fixed canonical scale `sqrt(2n)=sqrt(768)`;
the reference direction norms agree with that scale to reference accuracy.

| Fixture | First 3e-5 gap pass: Newton | Gap there | Direction upper / sqrt(768) | Observed relative error | High-accuracy endpoint bound / sqrt(768) |
|---|---:|---:|---:|---:|---:|
| Tuned confirmation | 3 | 1.28228e-5 | 6.46774e-2 | 1.38304e-5 | 5.73458e-4 |
| Low | 3 | 1.20416e-5 | 7.76409e-2 | 1.39231e-5 | 6.41841e-4 |
| Middle | 3 | 2.65685e-5 | 1.30495e-1 | 2.91749e-5 | 7.95888e-4 |
| High | 4 | 4.60260e-7 | 1.38333e-2 | 4.91297e-7 | 6.05822e-4 |
| Unsupported-AdamW ablation | 4 | 1.93695e-7 | 8.83050e-3 | 2.17651e-7 | 5.84704e-4 |

At the independent endpoints, alpha_lower remains positive on both sides;
residual uncertainty is approximately `1.22e-11`–`6.03e-11`, compared with
smallest singular values `3.76e-5`–`1.71e-4`. Current sigma_min, not a fixed
rcond constant, is the theorem's conditioning input. Down rconds still lie
below production guard in every fixture. Endpoint G_upper ranges from
`1.40e-9` to `5.96e-9`, including conservative feasibility/scalar allowances.
The `gram_upper` radial overscaling contributes a roughly `2e-10` normalized
value-gap floor, separately from these additional numerical allowances.

### Tuned confirmation: complete accepted-iterate sequence

| Newton | Gap | G_upper (absolute) | alpha_lower U / D | Bound / sqrt(768) | Observed relative error | CG queries / line trials so far |
|---|---:|---:|---|---:|---:|---|
| 0 | 3.73543e-2 | 4.65220e-2 | 4.79025e-5 / 3.77247e-5 | 3.48223 | 5.08118e-2 | 0 / 0 |
| 1 | 7.63515e-3 | 9.50842e-3 | 4.79693e-5 / 3.76185e-5 | 1.57756 | 8.41574e-3 | 2 / 1 |
| 2 | 5.83944e-4 | 7.27215e-4 | 4.79563e-5 / 3.75811e-5 | 4.36103e-1 | 6.12732e-4 | 5 / 2 |
| 3 | 1.28228e-5 | 1.59700e-5 | 4.79575e-5 / 3.75805e-5 | 6.46774e-2 | 1.38304e-5 | 9 / 3 |
| 4 | 1.05543e-7 | 1.32589e-7 | 4.79575e-5 / 3.75805e-5 | 5.88237e-3 | 1.17914e-7 | 14 / 4 |
| 5 | 5.28871e-8 | 6.70142e-8 | 4.79575e-5 / 3.75805e-5 | 4.17988e-3 | 5.90605e-8 | 21 / 6 |
| 6 | 5.28807e-8 | 6.70069e-8 | 4.79575e-5 / 3.75805e-5 | 4.17966e-3 | 5.90533e-8 | 28 / 20 |
| Independent endpoint | 1.99417e-10 | 1.39877e-9 | 4.79575e-5 / 3.75805e-5 | 5.73458e-4 | 1.99e-10 | Different classification solve |

The last row uses production Gram recovery at the saved independent multiplier;
its small observed difference is mostly conservative radial overscaling.
At Newton 6 another CG proposal and all 24 line trials fail the strict-target
search, giving total 35 HVPs and 44 line trials. No CPU reference is invoked.
Bounds exceeding the feasible-pair diameter are valid but uninformative;
the elementary diameter bound can cap them. This does not help
the near-converged accuracy question.

All five complete sequences, with both smallest singular values, alpha lower bounds,
rconds, absolute G_upper, direction error, observed error, and Newton/CG/trial
work, are in [full_rank_direction_progress.csv](../cluster/full_rank_direction_progress.csv)
and [full_rank_direction_summary.json](../cluster/full_rank_direction_summary.json).
The 45 accepted-iterate certificate evaluations reuse their matching smooth
factors. Rejected line trials remain controlled by the unchanged solver; this
posterior is an output audit, not a replacement HVP/globalization policy.

## Above-guard controls and coverage limit

The saved six step-0 problems from the locked smoke initialization/input stream
were used, without regenerating a model. Their original production settings
certify and are compared with independent high-accuracy L-BFGS references.
There is no serialized full warm-input corpus for the locked 50-step run:
the old research replays regenerated models from checkpoints. That regeneration
is forbidden here. Therefore these six controls sample the initial trajectory,
**not** its full warm distribution. A separately saved real warm problem from
the earlier precision diagnostic supplies one supplemental warm control; it
is explicitly not a newly reconstructed locked-trajectory sample.

| Control | rcond | Production gap | Bound / sqrt(768) | Observed relative error |
|---|---:|---:|---:|---:|
| Locked step-0 block 0 | 7.58757e-3 | 2.27907e-7 | 3.64865e-3 | 2.43851e-7 |
| Locked step-0 block 1 | 7.88653e-3 | 2.76845e-7 | 3.59916e-3 | 3.03177e-7 |
| Locked step-0 block 2 | 5.90656e-3 | 2.27589e-5 | 3.30917e-2 | 2.50682e-5 |
| Locked step-0 block 3 | 4.96293e-3 | 3.36086e-7 | 4.25167e-3 | 3.80683e-7 |
| Locked step-0 block 4 | 3.15123e-3 | 9.80622e-7 | 8.02816e-3 | 1.02792e-6 |
| Locked step-0 block 5 | 2.12402e-3 | 7.59382e-7 | 8.47387e-3 | 8.08399e-7 |
| Saved diagnostic warm step 2, block 0 | 1.01743e-2 | 1.94190e-5 | 3.00589e-2 | 2.06857e-5 |

The supplemental warm problem previously failed only the stricter `1e-8`
diagnostic search. At the unchanged `3e-5` target it certifies in two Newton
iterations without fallback. The control bounds range from `3.60e-3` to `3.31e-2`,
overlapping the Stage-D first-gap-pass range `8.83e-3`–`1.30e-1`; the tuned
fixture's bound exceeds every sampled control. Both groups have much smaller
actual errors than their own bounds. At their high-accuracy endpoints, the six
cold bounds normalize to `1.25e-4`–`1.89e-4`, still well above their casting floor.

### Empirical tightness

Observed error divided by the posterior upper bound, using linear-interpolated
quantiles:

| Sample | Count | Median | p95 | Max |
|---|---:|---:|---:|---:|
| All Stage-D accepted iterates | 45 | 2.00454e-5 | 1.70004e-2 | 2.27704e-2 |
| First Stage-D value-gap passes | 5 | 1.79326e-4 | 2.21623e-4 | 2.23570e-4 |
| Seven production-certified controls | 7 | 9.53990e-5 | 7.36727e-4 | 7.57536e-4 |

Every observed error is below its bound. These comparisons support the model;
finite samples do not prove an IEEE/backend guarantee. The poor tightness is
structural: the support-gap theorem converts value error into a square-root
distance bound using the smallest singular value, whereas radial recovery can
lose objective to first order even when the actual direction is much closer.
Conservative Frobenius backward/scalar bounds and Gram overscaling also leave
a nonzero numerical floor. Extra Newton iterations cannot remove those terms.

## Accuracy-budget investigation

No new admission threshold was selected. The tested reference scales disagree:

- Normal certified observed relative errors span `2.44e-7`–`2.51e-5`, with
  median `8.08e-7` over seven controls. This is a small empirical sample, not
  a guaranteed acceptable optimizer error or a model-quality tolerance.
- Final fp32 canonical direction casting is about `2.53e-8` relative. For the
  five raw lifted updates, pulling the measured fp32 cast error back to
  canonical coordinates gives absolute errors about `7.01e-7`–`7.02e-7`.
  Casting error is a representation floor, not permission for an arbitrarily
  larger solver error. Existing production directions themselves can differ
  from high-accuracy references by more than that floor.
- Re-optimized fp32-scale objective probes change the exact-problem numerical
  optimum by roughly `1e-5`–`1e-4` relative, depending on orientation. Input
  sensitivity is not an allowed optimization-error budget for a frozen input.
- Scaling the bound by the **fixed actual LR** and raw lift converts it into an
  update error. For tuned Newton 3/4, the raw paired update-error/paired-weight
  upper bounds are `3.26e-4` / `2.97e-5`. They quantify scale but provide no
  approved functional/loss accuracy tolerance. Raw Euclidean ratios are gauge
  dependent; the canonical/H metric is preferable for an admission contract.

For illustration only, tuned Newton 4 falls below the **median posterior bound**
of the seven already-accepted controls (`8.03e-3` normalized); no production
meaning follows from choosing that sample statistic. Its high-accuracy endpoint
still does not reach the normal controls' maximum **observed** error, the maximum
tested fp32-scale optimum perturbation, or the fp32 cast floor. This separates
an existing loose certificate scale from an actual direction accuracy budget.
There is consequently **no defined iteration of direction admission**. Newton 3
is the first value-gap pass; the direction-bound sequence above is the answer
conditional on any future separately justified budget.

For a prospective absolute budget b, the global formula would require
`G_upper <= alpha_lower (b-delta_feas)^2/8`, provided b>delta_feas. At a normalized
budget equal merely to the largest observed normal-control error, the tuned
allowable gap would be approximately `2.27e-12`; its independent endpoint
G_upper is `1.40e-9`. At the cast floor the discrepancy is far larger. This is
evidence of looseness, not a reason to fit a larger threshold or remove error
terms. There is no validated scalar budget and hence no counterfactual admission
policy or resumed training experiment in this study.

## Perturbation cross-check

Two distinct experiments were kept separate: fixed-lambda polar/recovered-pair
sensitivity, and independent re-optimization of the perturbed frozen problem.
The latter avoids calling a fixed-lambda polar change an optimum-direction
change. None of the perturbed reference solutions controls the posterior.

| Fixture | Re-optimized random eps32 change | Smallest-singular normal eps32 change | One-fp32-ulp objective proxy change |
|---|---:|---:|---:|
| Tuned | 1.75039e-5 | 5.84823e-5 | 8.82203e-6 |
| Low | 2.07356e-5 | 7.21047e-5 | 1.10155e-5 |
| Middle | 3.27703e-5 | 1.13368e-4 | 1.55848e-5 |
| High | 1.92506e-5 | 6.66899e-5 | 1.02782e-5 |
| Ablation | 1.74074e-5 | 6.03479e-5 | 9.54125e-6 |

At eps64, re-optimized changes stay below `2.75e-13`. The one-ulp proxy has
relative objective norm about `8.74e-8`–`8.78e-8`. The exact preceding EMA is not
saved, so this is **not** a reconstruction of the actual unrounded EMA update.
It tests representational sensitivity only. The previous report's 1% probes
remain larger-problem-change diagnostics and were not used as a rounding model.

For all 25 re-optimized perturbations, the within-problem recovered direction
error remains below the posterior bound. The posteriority claim is for each
represented input separately; it is not a uniform robustness claim across
objective perturbations. The observed amplification near small singular values
is compatible with, rather than hidden by, the spectrum-dependent bound.

## Cost, validation, and artifacts

One H100 NVL per job, four CPUs, 16 GiB, under SLURM. Runtime is Python 3.10.20,
torch 2.10.0+cu128, CUDA 12.8; deterministic algorithms enabled and TF32 disabled.
The validated base environment was not changed. Warmed, synchronized incremental
posterior time on the tuned pair is **2.086 ms median / 2.112 ms p95** (20 repeats),
with cached factors and matching radial diagnostics. The current cached-spectrum
Gram recovery/certificate is **9.153 ms median / 9.186 ms p95** in the same
allocation. The posterior itself performs no SVD/SVDVALS, new eigendecomposition,
dense multiplier Hessian, or reference call. An accepted-output-only audit would
cost roughly 12.5 ms for six pairs on this measurement; every-iterate auditing
costs more. Observer timings of the strict continuations include extra research
certification and must not be confused with incremental posterior cost.

The complete suite passed **418 tests, zero failures, zero skips**, including
11 new parameterized research cases, in **21.683 s** in the study allocation.
The final consistency-check revision independently passed **418/418** with zero
failures or skips in **17.185 s**.
Tests cover prescribed/repeated spectra, analytic coupled optima below guard,
arbitrary current multipliers, every-optimum semantics, approximate feasibility,
scale variation, rank ambiguity, fp64-only inputs, and no new decomposition or
hidden reference oracle. Existing mathematical and production tests are unchanged.
No network-guard audit entries were recorded; production source hashes match
the preceding study.

Files:

- [research posterior](../experiments/full_rank_direction.py),
  [tests](../tests/test_full_rank_direction.py);
- [fixed-fixture runner](../cluster/full_rank_direction_admission.py),
  [SLURM script](../cluster/full_rank_direction_admission.sbatch);
- [supplemental runner](../cluster/check_direction_scales.py),
  [SLURM script](../cluster/check_direction_scales.sbatch);
- [main raw log](../cluster/full_rank_direction-29072.log),
  [supplemental raw log](../cluster/direction_scales-29073.log);
- [final full-suite log](../cluster/direction_final_tests-29075.log),
  [JUnit XML](../cluster/direction-final-tests-29075.xml),
  [test SLURM script](../cluster/run_direction_tests.sbatch);
- [main JSON](../cluster/full_rank_direction_summary.json),
  [scale/perturbation JSON](../cluster/full_rank_direction_scales.json),
  [complete progress CSV](../cluster/full_rank_direction_progress.csv),
  [stdlib-only aggregator](../cluster/summarize_direction_admission.py).

Shared-home outputs are
`/home/prignano/qnormuon-runs/full-rank-direction/study-29072/` and
`/home/prignano/qnormuon-runs/full-rank-direction/scales-29073/`. No new large
tensor dumps or checkpoints were created. The pre-existing AGENTS.md edit was
preserved; production files were not modified.
An initial final-test submission used SLURM's `sh` wrap with Bash-only setup
and failed before tests started; its [startup log](../cluster/direction_final_tests-29074.log)
is retained. The explicit Bash script above corrected submission plumbing.

## Explicit answers and stage gate

1. **Support-gap theorem:** for a full-column-rank tall residual and any
   contraction, support loss is at least `sigma_min/2 ||P-polar(B)||_F^2`.
   The direct SVD/contraction proof is above.
2. **Coupled bound:** support losses add because horizontality cancels lambda;
   weighted squared error to the current polar pair is at most 2G.
3. **Returned-direction posterior:**
   `delta_feas+2 sqrt(2G_upper/min alpha_lower)`, optionally sharpened sidewise,
   bounds distance to every optimum under the stated operation model.
4. **Conditioning input:** actual current sigma_min less measured/model
   uncertainty, not a fixed rcond. It does not prove optimal multiplier
   uniqueness or certify all Newton/HVP arithmetic.
5. **Normal tightness:** accepted-control observed/bound ratios are about
   `1e-4` median; bounds are very pessimistic. Full locked warm-state coverage
   is unavailable without forbidden model regeneration.
6. **Stage-D tightness:** first-gap-pass ratios are about `1.8e-4` median;
   high-accuracy endpoint relative bounds remain `5.7e-4`–`8.0e-4` despite
   observed recovered-pair differences around `2e-10`.
7. **Tuned admission iteration:** not defined without a justified budget;
   the value criterion passes at Newton 3. Bounds and reference-scale
   crossings are explicitly reported rather than turned into a new threshold.
8. **Defensible budget:** none established. Casting, EMA sensitivity, normal
   empirical errors, and LR-scaled update errors are different quantities;
   none automatically supplies a permitted optimizer error.
9. **Overhead:** about 2.1 ms per pair posterior with reuse; no extra tall SVD.
10. **Stage-D removal:** not demonstrated. Positive rank separation makes
    finite bounds possible, while rank-ambiguous alpha lower bounds refuse
    certification. Neither condition alone licenses admission below guard.
11. **Production-v1 experiment:** not justified, classification **C**. The
    theorem is defensible conditionally, but the required application budget
    and practically informative bound are absent. Resolving an explicit
    gauge-invariant output accuracy contract and tightening its posterior
    would be separate research; do not lower the guard, resume confirmation,
    or implement that next work automatically.
