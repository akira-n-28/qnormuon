# Projected-primal spectral norm: conservative Gram backend

## Decision and scope

Production-v0 now defaults to `SolverConfig(primal_norm_backend="gram_upper")`
for **projected-primal radial feasibility only**. The independent
`primal_norm_backend="svd"` remains available. The smooth residual SVD/polar,
coupled objective, `3e-5` normalized-gap target, `1e-4` residual-rcond guard,
`1e-10` horizontal tolerance, `1e-12` spectral-excess tolerance, and
`-1e-10` signed normalized-gap floor are unchanged. No approximate polar or
rank truncation was introduced.

After horizontal projection, the certificate needs only
`sigma_max(P_U)` and `sigma_max(P_D)`. It sets
`scale=max(1,sigma_max(P_U),sigma_max(P_D))` and divides both matrices by
that common scale. It does not use the other projected-primal singular values
or singular vectors. In exact arithmetic,
`sigma_max(P)^2=lambda_max(P.T @ P)`. Replacing full `svdvals(P)` by a Gram
eigensystem therefore preserves the mathematical certificate *provided* its
finite-precision norm estimate does not underestimate the true top singular
value. The solver reports conservative upper estimates as `spectral_norms` in
Gram mode; `primal_spectral_backend` identifies the path.

## Numerical bound and its limits

The implementation first scales each finite matrix by its largest absolute
entry `s=max(abs(P))`, computes `X=fl(P/s)`, and forms the symmetric fp64 Gram
matrix `G=fl(X.T @ X)`. Scaling avoids Gram overflow/underflow in the tested
range from `1e-160` to `1e160`. It computes a **full** symmetric eigensystem
`(Q,D)`, then measures

```
delta = ||Q.T @ Q - I||_F
rho   = ||G @ Q - Q @ D||_F.
```

These measurements avoid requiring a cuSOLVER-specific eigenvalue forward
error assertion. In exact arithmetic, with `R=GQ-QD`,

```
G - Q D Q.T = G(I-Q Q.T) + R Q.T.
```

For square `Q` and `delta<1`, `||I-QQ.T||_2 <= delta` and
`||Q||_2 <= sqrt(1+delta)`. Thus

```
lambda_max(G) <= max(d_max,0)*(1+delta)
                 + ||G||_F*delta + rho*sqrt(1+delta).
```

With fp64 unit roundoff `u=2^-53` and `gamma_k=k*u/(1-k*u)`, the standard
dot-product model gives
`||fl(X.T X)-X.T X||_2 <= gamma_m*||X||_F^2` for `m` rows; symmetric
averaging adds an `u*||G||_F` allowance. The implementation inflates measured
Frobenius reductions with `gamma_(n*n)` and `gamma_(m*n)`, bounds rounding in
the `n`-term products that form `Q.T Q` and `GQ`, and adds the componentwise
division allowance `u/(1-u)*||X||_F` before multiplying by `s`. A final
`gamma_16` factor covers the short positive scalar expression. The full
formula is auditable in `primal_top_singular_upper` in
`qnormuon/coupled_solver.py`. A nonfinite estimate, orthogonality bound
`delta>=1`, or negative computed top eigenvalue beyond the Gram-rounding
allowance uses the full-SVD radial reference. Only a tiny negative top value
within that allowance is clamped to zero.

The algebraic eigensystem inequality is exact. The one-sided *floating-point*
claim is conditional on the conventional IEEE fp64 operation/reduction error
model, finite values, and normal division behavior; it is **not** interval
arithmetic or a formal guarantee for every undocumented GPU-kernel reduction.
The independent full-SVD path and H100 oracle tests remain necessary. Gram
matrices square the condition number of the whole spectrum, but this backend
uses only the **largest** eigenvalue, checked with full residual and
orthogonality bounds. It never reconstructs a polar from small Gram
eigenvalues. The smooth residual rcond guard remains separately enforced.

## H100 validation

The deterministic adversarial study (`cluster/primal_norm_study-28927.json`)
tested 41 cases across `[12,5]`, `[64,16]`, and `[1024,384]`: Gaussian,
nearly rank-one, clustered/repeated top values, trailing singular values
`1e-4`, `1e-8`, and `1e-12`, norms at `1 +/- 1e-12` and `1 +/- 1e-8`,
extreme scales, and a row-gauge-scaled real projected candidate. Full fp64
SVD/SVDVALS was the independent oracle. All conservative estimates exceeded
that oracle; relative positive slack ranged from `9.77e-15` to `1.99e-10`.
The unscaled raw Gram method reached `7.36%` relative error on an underflow
stress case, which is why production uses the scaled, bounded construction.
Repeated evaluations were deterministic on the allocated H100.

Two independent replays of the locked 50-step smoke covered **931** actual
certificate evaluations: 471 in steps 0–24 and 460 in steps 25–49, including
all **300 accepted pairs**. Every Gram/SVD acceptance decision agreed; no
candidate rejected by full SVD was accepted by Gram. Maximum normalized-gap
difference was `1.993e-10`. Among the 460 saved-checkpoint evaluations,
the smallest positive norm-upper margin was `1.982e-10`. The closest accepted
reference gap, at step 5,
was `2.9957532369e-5`, leaving `4.247e-8` to the unchanged target; Gram
reported `2.9957731569e-5` and also accepted it. Replays are recorded in
`cluster/primal_norm_early_replay-28929.json` and
`cluster/primal_norm_study-28927.json`.

Synthetic certificates straddling both `sigma_max(P)=1` and the `3e-5` gap
boundary produced **zero unsafe acceptances**. Three cases deliberately set
at the exact gap threshold were accepted by SVD and conservatively rejected
by Gram, with a gap shift about `5.3e-12`; this can cause an extra Newton
iteration at an exact boundary. False rejection is preferable to false
feasibility. The full-SVD debug path remains available for such investigations.

On the identical saved step-25 checkpoint, six production pair solves using
each backend had exactly the same inputs, Newton counts, and lambda. Largest
direction difference was `3.74e-11`; largest normalized-gap difference was
`1.993e-10`; dual objective and residual rcond matched exactly. The largest
float32 model/lifted-update difference after one step was `1.86e-9`.
See `cluster/primal_backend_comparison-28932.json`.

## H100 cost

Synchronized, warmed six-real-pair kernel timings from job 28927:

| Radial norm kernel | Per-pair median range | Six-pair sum of medians | Six-pair sum of p95s |
| --- | ---: | ---: | ---: |
| Full `svdvals` | 39.50–41.18 ms | 243.40 ms | 243.53 ms |
| Raw Gram `eigvalsh` (study only) | 7.31–7.34 ms | 43.95 ms | 44.00 ms |
| Conservative Gram `eigh` + bound | 8.38–8.41 ms | 50.36 ms | 50.57 ms |

The conservative kernel is **4.83×** faster than full `svdvals` here. For a
representative two-matrix pair, its fp64 Gram GEMM took about `0.18 ms`, EVD
`7.67 ms`, and bound evaluation `0.53 ms`. These are synchronized kernel-study
measurements, not a claim of general hardware speedup.

The final production-v0 H100 suite passed: **274 passed, 0 failed, 0 skipped**
(`cluster/run_production_tests_h100-28935.log`). The final locked 50-step QSO
run (`cluster/run_tiny_smoke-28936.log`) certified **300/300** pairs, used **zero**
fallbacks, and preserved the Newton bins (269 at two iterations, 31 at three).
Its three-step checkpoint continuation was exact for parameters, loss, and
optimizer state. The cold first step took `1.909 s` (`1.718 s` optimizer),
the full 50-step job took `58.11 s`, and peak allocated GPU memory remained
`725,709,824` bytes. Warm median throughput was about `2,003 tokens/s`.

| Warm metric (exclude first five steps) | Previous full-SVD radial | Conservative Gram | Speedup |
| --- | ---: | ---: | ---: |
| Full step median | 1.617 s | 1.023 s | **1.58×** |
| Full step p95 | 1.705 s | 1.080 s | 1.58× |
| Six-pair solver median | 1.579 s | 0.985 s | **1.60×** |
| Certificate median | 0.757 s | 0.164 s | 4.63× |
| Smooth residual SVD median | 0.774 s | 0.774 s | 1.00× |
| Projected-primal `svdvals` median | 0.745 s | 0 | removed |

All minibatch hashes matched. The training trajectory is **not bitwise
identical** after final float32 update casts: losses first differ at step 18;
maximum absolute training-loss difference over 50 steps is `1.01e-4` (median
`1.10e-5`), and maximum validation-loss difference is `4.97e-5`. The selected
Newton iterations and certification status stayed identical. The run's maximum
gap was `2.995773e-5`, and minimum residual rcond `9.779e-4`.

The remaining dominant cost is the untouched smooth residual SVD/polar,
approximately `0.774 s` of the `1.023 s` warm full step. QSO remains about
30× slower per step than the locked AdamW timing reference (`~0.0334 s`). This
is a material exact-solver improvement, but performance is still a serious
barrier to broad LR sweeps. The next separate study should target the **exact**
smooth residual SVD/polar work and certificate frequency, preserving the same
feasibility and conditioning rules. No LR sweep was started.
