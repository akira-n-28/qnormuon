# K=I Coupled Quotient Spectral Optimizer — production-v0

The new API is `QuotientSpectralOptimizer`, `SwiGLUPair`, and `SolverConfig`.
The historical `QNorMuon` API and its tests remain unchanged. This is an
initial auditable implementation, not a throughput claim or a training result.

## Mathematical contract

Each explicitly named pair holds actual PyTorch `up_proj.weight` of shape
`[m,n]` and `down_proj.weight` of shape `[n,m]`, with `m >= n >= 1`.
The mathematical down variable is `D = down.T`. Both row norms must be strictly
positive. Exact zero and one-sided zero rows raise an error; no epsilon clamp,
skip policy, or neuron birth is implemented. Nonfinite or unrepresentable
canonical factors also fail explicitly.

The metric is `h_i = ||u_i|| / ||d_i||`. Implementation evaluates `sqrt(h)` as
`sqrt(r)/sqrt(s)` using scaled float64 row norms, so it need not materialize
an overflowing squared ratio. Canonical weights are `U_bar = U/sqrt(h)` and
`D_bar = D*sqrt(h)`. This is the regular positive-gauge construction; extreme
inputs still need representable canonical arithmetic and float64 row metrics.

The raw covector transforms **once**:

```
G_Uc = sqrt(h) * grad_up
G_Dc = grad_down.T / sqrt(h)
M_U <- beta * M_U + (1-beta) * G_Uc
M_D <- beta * M_D + (1-beta) * G_Dc
```

EMA starts at zero. The current step solves with the **updated** momentum.
There is no bias correction, Nesterov term, implicit weight decay, or
post-update weight rebalance. With EMA, this is steepest descent for the
momentum linear surrogate, not necessarily the current loss differential.

For `A=(M_U,M_D)`, the primal maximizes `<A,P>` subject to
`||P_U||_2 <= 1`, `||P_D||_2 <= 1`, and
`L(P)_i = <U_bar[i],P_U[i]> - <D_bar[i],P_D[i]> = 0`.
The dual minimizes

```
||A_U - diag(lambda) U_bar||_* + ||A_D + diag(lambda) D_bar||_*.
```

The plus sign is essential. This is the spectral norm restricted to horizontal
tangents, not an infimum over vertical spectral lifts. Coupling occurs in the
dual solve; horizontal projection of separate input polars is not the optimizer.

Updates subtract `lr * Delta`, where
`Delta_U=sqrt(h)*P_U`, `Delta_down=(P_D/sqrt(h)).T`.
On the regular domain, positive gauge changes and matching canonical state
leave the canonical problem invariant and give equivariant raw updates in
exact arithmetic. Measured finite-precision behavior, including update casting,
is a separate contract. Tests include actual SwiGLU trajectories, a populated
momentum gauge reset, and finite-step neuron matrix changes including the
quadratic term.

An exactly zero represented intrinsic cotangent returns exactly zero. Tiny
nonzero objectives are normalized, not thresholded to zero. Floating-point
cancellation when subtracting a large vertical component is not proof of an
exact zero class and is handled as a reported numerical limitation.

## Solver and certificates

`qnormuon/coupled_solver.py` ports the validated Newton-CG equations from
`experiments/dual_solver.py`, independently of the historical optimizer.
Float64 preprocessing computes

```
w = rowsum(U_bar**2 + D_bar**2)
beta_vertical = L(A) / w
A_h = A - L*beta_vertical
s = ||A_h||_F
U_eff = U_bar / sqrt(w); D_eff = D_bar / sqrt(w)
lambda = beta_vertical + s * z / sqrt(w)
```

Warm state is **original-coordinate lambda**. Each new solve transforms it as
`z = (lambda_previous-beta_vertical)*sqrt(w)/s`; whitened `z` is never persisted.
The initial Newton budget is two iterations. Certification is checked at every
iterate; solving continues past two up to `max_iterations=100`. Two steps are
not a convergence guarantee. `extended_initial_budget` records continuation.

Thin SVD factors are cached within an evaluation. Hessian products use the
validated polar derivative with denominators `sigma_i+sigma_j` and `sigma_j`;
CG performs no new SVD and never constructs an `m x m` Hessian. Repeated
positive singular values are supported. Damping is `1e-6` times the local
curvature bound, applied only to the Newton system, never to the nuclear-norm
objective. CG is limited to `min(30,m)` iterations by default. Armijo
backtracking includes the documented floating-point decrease/gradient checks.
Row whitening removes row-amplitude factors from an upper curvature bound;
it is not asserted to universally improve the actual condition number.

Float64 recovery projects the solved residual polar pair onto `ker L`, then
radially scales by `max(1, ||Q_U||_2, ||Q_D||_2)`. A sign flip supplies a
nonnegative primal lower bound. Certification evaluates the original objective
and multiplier, not just the internally centered/scaled problem.

Defaults:

| Criterion | float64 smooth work | float32 smooth work |
| --- | ---: | ---: |
| Normalized nonnegative primal-dual gap | `1e-8` | `3e-5` |
| Smooth residual rcond guard | `1e-8` | `1e-4` |
| Normalized horizontal residual | `1e-10` | `1e-10` |
| Spectral excess | `1e-12` | `1e-12` |
| Signed normalized gap lower bound | `-1e-10` | `-1e-10` |

The gap denominator is `max(abs(primal),abs(dual),tiny_float64)`; it is not
floored at one. Signed gap, its absolute magnitude, and normalized gap are
reported separately. Conditioning is checked before smooth acceptance.
These are numerical certificates, not interval-arithmetic proofs. A value gap
alone never certifies direction accuracy near rank loss.

## Precision and diagnostics

`SolverConfig(dtype=torch.float32)` is the default, including for bf16/float16
model tensors. Select `torch.float64` explicitly for high-accuracy solver work.
Canonical gradients and EMA have solver dtype. Norm preprocessing, objective
centering, feasible recovery, certificates, and stored lambda use float64.
Smooth residual SVD/polar and CG use solver dtype. Autocast is disabled inside
the production numerical path. No bf16 SVD is called.

The certified direction is float64. The raw lifted direction is cast to the
original parameter dtype before the learning-rate update. Certification applies
to the recovered direction **before casting**. Casting and parameter addition
can introduce horizontal/spectral defects and rounding; there is no promise of
float64 gauge invariance for bf16 updates.

`optimizer.last_diagnostics` is keyed by stable pair name and includes:
objectives, signed/absolute/normalized gap, horizontal/normalized horizontal
residual, both spectral norms/excess, final residual rcond, minimum encountered
smooth rcond, Newton and CG counts, SVD counts, reference iterations, fallback
status/reason, convergence, selection semantics, and solver/certificate/update
dtypes. Optional cast diagnostics measure relative direction error, horizontal
residual, and spectral excess after lifting/casting and pulling back. Their two
additional SVDs are separately counted. `diagnostics=False` avoids these cast
checks and retaining optimizer diagnostics; solver certificates remain mandatory.

## Fallback and rank-loss limitation

Fallback is an explicit lazy call to the existing CPU float64 generic ADMM
reference solver. GPU fallback transfers inputs to CPU, then returns the pair
and original-coordinate lambda to the input device and rechecks the certificate.
The repository's `experiments/dual_solver.py` and `horizontal_spectral.py` must
remain importable; use this checkout on `PYTHONPATH`. A standalone distribution
omitting these modules cannot provide the configured fallback.

Triggers include poor residual conditioning, nonfinite evaluation, failed
line search, inability to obtain descent, certified-gap stagnation, exhausted
iteration budget, cancellation-dominated cotangent, and an SVD convergence error.
Invalid/nonfinite inputs fail validation directly. The fallback's own iteration
limit and convergence are checked. The optimizer raises on an uncertified
result, and validates all active pairs before committing parameter/state changes.
With `fallback=False`, an uncertified low-level result remains visibly failed.

Generic ADMM certifies the **primary value**, not the exact minimum-Frobenius
selection on every nonunique deficient optimal face. Diagnostics say
`primary_only_unless_uniqueness_established`. Smooth solutions report uniqueness
only in the exact full-rank limit. No positive singular values are hard-truncated.
The test suite retains the near-rank counterexample where a tiny value gap
coexists with an order-one direction error. Production-v0 exposes this accepted
fallback limitation rather than claiming a new singular-face solver.

## Registration, checkpoints, and usage

```python
import torch
from qnormuon import QuotientSpectralOptimizer, SwiGLUPair, SolverConfig

pairs = [
    SwiGLUPair(f"blocks.{i}.mlp", block.mlp.up_proj.weight,
               block.mlp.down_proj.weight)
    for i, block in enumerate(model.blocks)
]
qso = QuotientSpectralOptimizer(pairs, lr=1e-3, beta=0.95,
                               solver=SolverConfig(dtype=torch.float32))
paired_ids = {id(p) for pair in pairs for p in (pair.up, pair.down)}
other = [p for p in model.parameters() if id(p) not in paired_ids]
adam = torch.optim.AdamW(other, lr=1e-3)

qso.zero_grad(set_to_none=True)
adam.zero_grad(set_to_none=True)
loss.backward()
qso.step()
adam.step()
```

Names and up/down roles are mandatory. Registration is sorted by name;
parameter traversal order does not define pairing. Duplicate names/parameters
are rejected and topology is fixed after construction. Gate projections,
embeddings, norms, biases, heads, and unsupported matrices are not implicitly
owned. Both missing gradients skip a pair; a single missing or sparse gradient
is an error. Closures and ordinary optimizer parameter groups are supported;
each group is a fixed named pair, with learning rate/beta/solver controls.

Save model weights and `qso.state_dict()` together. The checkpoint stores named
pair topology, shapes, solver configuration, EMA tensors, original lambda, and
step count. Loading validates names/shapes/state precision, maps sorted names
deterministically, and explicitly preserves float32/64 state instead of casting
it to bf16 parameter storage. Different pair topology is rejected. A checkpoint
with matching names assumes the caller bound those names to the correct model
weights. Tests use a real `torch.save`/`torch.load(weights_only=True)` roundtrip
and reproduce continued trajectories exactly for the tested dtypes.

## Historical boundary and remaining research

`qnormuon/core.py` and `QNorMuon` remain historical: independent polars,
epsilon-clamped metric, shared-K state, paired-leverage target `2n/m`, old logdet
balancing, and thin-SVD completion at zero momentum. Their adversarial tests
remain meaningful. The new API imports none of those update routines.

Shared K, leverage/contribution balancing, signed gauges, neuron birth,
minimum-Frobenius recovery on every deficient face, approximate polar kernels,
custom CUDA, distributed optimization, and performance/quality claims from
Transformer training are outside this implementation. No benchmark is started.

## Validation record

Validation runs on Lagrange inside SLURM using Python 3.10.20,
torch 2.10.0+cu128 (CUDA build 12.8), NumPy 2.2.6, pytest 9.0.3,
and one NVIDIA H100 NVL. No installation or download is needed.
The script is `cluster/run_production_tests_h100.sbatch`; it requests one GPU,
four CPUs, 16 GiB RAM and 30 minutes. It preserves raw output with the job ID,
checks the existing spectral path, runs the unfiltered suite, and measures
production float32 and bf16/fp32 smoke steps and peak allocated GPU memory.
Most historical reference tests intentionally run CPU math inside the allocation;
new production solver tests exercise both CPU and CUDA.

Initial job 28893: 179 passed, three new-test assertion failures. All 150
historical tests passed. Diagnostic job 28894 reproduced the same behavior in
the unchanged research solver: native-fp32 random seed 901 has an approximately
`5.43e-11` smallest optimal residual singular value, rather than the assumed
full-rank smooth optimum. The fractional example reaches `gap_stagnation`
before the rcond guard. These were fixture/solver-contract assumptions, not
weakened numerical tolerances or theory regressions. Both cases remain covered
as explicit fallback tests. Raw failures and comparison output are retained in
`cluster/run_production_tests_h100-28893.log` and
`cluster/diagnose_production_v0-28894.log`.

Final job **28895**, node `lagrange0`: **189 collected, 189 passed, 0 failed,
0 skipped** (150 unchanged historical tests plus 39 new production tests).
Pytest took **7.95 seconds**; tests plus production smoke took **8.82 seconds**;
the complete allocation elapsed **34 seconds**, SLURM `COMPLETED`, exit `0:0`.
Peak PyTorch CUDA allocation was **33,602,560 bytes (32.05 MiB)** and peak
reserved memory was **35,651,584 bytes (34 MiB)**. These exclude CUDA context,
driver allocations, and CPU memory; SLURM did not report MaxRSS.

Raw output: [run_production_tests_h100-28895.log](../cluster/run_production_tests_h100-28895.log).
Reproduce from the project root with
`sbatch --parsable cluster/run_production_tests_h100.sbatch`.

| H100 production smoke | fp32 parameters | bf16 parameters |
| --- | ---: | ---: |
| Solver dtype | float32 | float32 |
| Certificate dtype | float64 | float64 |
| Newton / CG iterations | 3 / 6 | 3 / 6 |
| Normalized gap | `2.02575e-5` | `2.02779e-5` |
| Normalized horizontal residual before casting | `3.21272e-17` | `5.88340e-17` |
| Spectral excess before casting | 0 | 0 |
| Final residual rcond | `0.170183` | `0.170101` |
| Fallback | no | no |
| Relative direction error after update casting | `2.62845e-8` | `0.00163484` |
| Horizontal residual after update casting | `3.30815e-8` | `0.00264730` |
| Spectral excess after update casting | `5.85631e-9` | `0.00204037` |

All smoke tensors were small `[16,8]` mathematical pairs. Both paths extended
the initial two-iteration budget to obtain certification. Native CUDA bf16 SVD
again raised the expected `NotImplementedError`; production tests intercepted
SVD calls under autocast and verified only float32/float64 inputs.

Production test telemetry recorded **77 solves**, **6 explicit fallbacks**:
2 line-search failures on the known native-fp32 rank-boundary case (CPU/CUDA),
1 fractional-case gap stagnation, 1 ill-conditioned residual, and 2 deliberately
exhausted Newton budgets. Two deliberately uncertified results were expected:
fallback disabled with zero Newton budget, and an exhausted one-iteration
reference budget. Tests verified that these do not become accepted optimizer
steps. Both dedicated production smoke solves used no fallback.

The network guard observed no Python outbound attempt; static inspection found
no network/download calls in `tests/`, `experiments/`, or `qnormuon/`.
No packages were installed and no assets downloaded. Historical implementations
and all existing mathematical tests remain unchanged. `git diff --check` and
lightweight Python AST syntax checks passed. Numerical execution occurred only
in SLURM allocations.

Validated source SHA-256 values:

```
8345e386f00d9d9937d18ca512198bc3e37367fbb4a45f826874411e63b868c2  qnormuon/coupled_solver.py
df6552136f7a5c5e121622e97abd1e30c70762a0ee8422d61a02df54bd603ec6  qnormuon/optimizer.py
271d631a4b7ae185e54ad06513384e4191e432aedd1723df5a652ffd706404b6  tests/test_production_v0.py
ede583762a7f5368ef4bd97d228ad2346fa0e3cdb6fad18e712063236d4448b7  cluster/run_production_tests_h100.sbatch
1371fadc81670005edf552e91835088a28e5e78d8a1b7ddf3b4de58c619d0055  cluster/run_production_validation.py
```

The current mathematical/reference suite and the production-v0 implementation
are validated on the Lagrange H100 environment within the contracts and
rank-loss limitations above. No tiny Transformer benchmark or training run has
been started.
