# Stage-D QSO failure forensics

## Result

**Classification A: genuine residual conditioning / rank-boundary blocker.**
The tuned 512-step confirmation requests reference fallback with the exact
production reason `ill_conditioned_residual`, at zero-based training step **87**
(87 completed updates; attempted update 88), pair **`blocks.1.mlp`**. The initial
warm-start down residual has fp64 SVD rcond **8.712293147996454e-5**, below the
unchanged `1e-4` guard. This happens at Newton iteration **0**, with **zero CG
queries and zero line-search trials**. It is not a failed Armijo search, a
Newton-budget exit, or an inference from the CPU timeout.

All five representative first transitions reproduced the same reason and
iteration-zero exit. The residuals have positive singular values on the fp64
oracle; this establishes entry into the guarded ill-conditioned regime, **not
exact rank deficiency**, nor a theorem that every optimum is deficient. Raw
weight rows remain regular and nonzero. A separate bounded CPU reference
experiment achieved a small value gap but did not return within 240 seconds;
its down residual remained below the smooth guard.

Production optimizer mathematics, sources, dtype policy, thresholds,
globalization, warm starts, and CPU fallback are unchanged. No new LR sweep or
seed was run. This report does not revise the inconclusive quality result in
[QSO_LR_SWEEP.md](QSO_LR_SWEEP.md).

## Frozen reproduction and observability

The original Stage-D manifest and per-run provenance were used directly from
`/home/prignano/qnormuon-runs/lr-stage-d/study-29024`. The tuned paired peak LR
was **0.0012247448713915891**, preserving the actual stored value rather than
rounding the run label; unsupported AdamW peak LR was **0.0003**. The confirmation
used 512 steps, 51 warmup steps, and the existing cosine schedule to 10% of peak.
The representative pilots used their original 256-step / 26-warmup schedules.

The frozen model has 11,457,408 parameters, six explicitly registered
`[1024,384]` SwiGLU pairs, width 384, six layers, context 128, and batch 16.
Seed is 2026; storage and canonical EMA are fp32, forward/backward uses bf16
autocast, and solver arithmetic and thin SVD are fp64. TF32 is disabled,
weight decay is zero, and clipping is **none**, matching Stage D. The primal
backend remains `gram_upper`, diagnostics enabled, post-cast diagnostics disabled.
All five acceptance thresholds are unchanged.

The already-local FineWeb train/validation shard and tokenizer paths are those
in the original provenance. Train prefix SHA256 is
`2db88d5f4b0da3484f766946b8090b0839e527a0088d55d156f851dcc5e63a59`;
the single-pass permutation hash is
`b396bbb0fd4021e6c6427df4f396787a1d6ab916120696083c58108f6dccac97`.
Initialization hash is
`0607adc4c9cffa7b902c0956fa7477ea274078fdc31406d84d527b79ca6a2401`.
Frozen production, benchmark, runner, and config source hashes were asserted
against the original manifest before replay. Every completed replay minibatch
hash and training loss matched the original run exactly. The failing step's
model hash and all paired optimizer step counts remained unchanged: **no failing
training update was committed**. The existing harness checks finite loss and
all model gradients before entering the optimizer.

The observability gap is precise: `_solve` determines `reason`, then calls
`_reference`, and only afterwards builds `SolverResult`. Outer `solve_coupled`
and the Stage-D runner expose fallback metrics only after that result returns.
The runner logs a pair `begin` immediately, but its `end`/reason arrives after
the entire call. A 180-second alarm inside CPU ADMM therefore preserves the
active pair and traceback but loses the initiating smooth reason.

Research-only [stage_d_forensics.py](../experiments/stage_d_forensics.py) scopes
observers around the **existing** evaluation, certificate, CG, and reference
functions. At reference entry it reads the live production frame, records the
reason, normalization, current and initial original-coordinate lambda, current
and best gaps, current/minimum rcond, gradient, counts, and complete available
history, then raises before CPU ADMM executes. The runner immediately writes
`smooth_exit.json` and `failed_pair.pt`. Current lambda is reconstructed from
`beta + magnitude * coord * z`; it is not confused with the `best` lambda that
production assigns before fallback. Saved fixtures also contain canonical
U/D/A, raw weights and gradients, and all evaluated internal coordinates.
No copied or altered numerical solver controls these replays.

## First transitions across the fixed representative set

All step numbers below are zero-based, equal to the number of completed updates
when the next update fails. “Gap” is the normalized value gap, not a percentage.
Current and best gaps and current/minimum smooth rconds coincide because each
failing solve evaluated only its initial state.

| Case | Paired / unsupported peak LR | Step | Pair | Exact reason | Gap | Min side rcond | Newton / CG / line trials |
|---|---|---:|---|---|---:|---:|---|
| Tuned 512 confirmation | 0.0012247448713915891 / 0.0003 | 87 | blocks.1.mlp | ill_conditioned_residual | 0.0373543 | 8.71229e-5 | 0 / 0 / 0 |
| Low 256 pilot | 0.0003 / 0.0003 | 92 | blocks.0.mlp | ill_conditioned_residual | 0.0766280 | 7.72783e-5 | 0 / 0 / 0 |
| Middle 256 pilot | 0.001 / 0.0003 | 87 | blocks.0.mlp | ill_conditioned_residual | 0.0768393 | 4.52030e-5 | 0 / 0 / 0 |
| High 256 pilot | 0.004 / 0.0003 | 56 | blocks.0.mlp | ill_conditioned_residual | 0.103536 | 7.93289e-5 | 0 / 0 / 0 |
| Unsupported-AdamW ablation | 0.0012247448713915891 / 0.0002 | 81 | blocks.0.mlp | ill_conditioned_residual | 0.0688398 | 8.76021e-5 | 0 / 0 / 0 |

The first failing pair is block 0 in four cases and its neighbor block 1 in the
confirmation. This localizes the observed first transitions to early blocks;
it does not prove later blocks were safe at the failing step, because execution
stops at the first request. The other four original failed candidates were not
replayed, so their initiating reasons remain unverified.

## Residual spectrum and certificate

Each saved failing solve has **one** smooth evaluation and no trial evaluations.
Independent original-coordinate residual SVD corroborates the cached normalized
spectrum: relative spectrum discrepancy is at most **6.10e-14** across these
fixtures. There is no SVD exception or evidence of a normalization-induced
rcond artifact.

| Case | U sigma_max / sigma_min / rcond | D sigma_max / sigma_min / rcond |
|---|---|---|
| Tuned confirmation | 0.370204 / 4.79025e-5 / 1.29395e-4 | 0.433005 / 3.77247e-5 / 8.71229e-5 |
| Low | 0.436435 / 4.10710e-5 / 9.41055e-5 | 0.579529 / 4.47850e-5 / 7.72783e-5 |
| Middle | 1.13137 / 7.31968e-5 / 6.46973e-5 | 1.43630 / 6.49252e-5 / 4.52030e-5 |
| High | 1.01744 / 1.10529e-4 / 1.08634e-4 | 1.44074 / 1.14292e-4 / 7.93289e-5 |
| Ablation | 1.41485 / 1.68509e-4 / 1.19101e-4 | 1.83021 / 1.60330e-4 / 8.76021e-5 |

Only the D side is below guard in the tuned, high, and ablation cases; both sides
are below in low and middle. The tuned original nuclear values are
**0.6278414674323044** (U) and **0.6175841451974328** (D).
Its eight smallest singular values, in descending order as returned by SVD,
are:

```text
U: 5.385813e-5 5.362686e-5 5.242327e-5 5.181456e-5
   5.080254e-5 5.022326e-5 4.911422e-5 4.790246e-5
D: 4.337910e-5 4.291417e-5 4.239606e-5 4.162851e-5
   4.069782e-5 4.029112e-5 3.863494e-5 3.772471e-5
```

These are smoothly clustered positive tails, not an isolated zero singular
value. The equivalent eight-value tails for every fixture are retained in the
structured summary and `spectrum_forensics.json`. There is an abrupt **temporal
rcond deterioration** relative to the previous accepted same-pair solve, but
no within-solve Newton-induced collapse: Newton never runs. Absolute previous
singular tails were not retained by Stage D, so these data do not establish
whether absolute sigma_min collapsed between training batches.

| Case | Previous accepted step / rcond | Previous intrinsic magnitude | Failing intrinsic magnitude | Magnitude ratio |
|---|---|---:|---:|---:|
| Tuned | 86 / 1.55538e-3 | 0.0329010 | 0.571430 | 17.37x |
| Low | 91 / 4.09233e-3 | 0.0108336 | 0.736889 | 68.02x |
| Middle | 86 / 2.45391e-3 | 0.0237060 | 1.83487 | 77.40x |
| High | 55 / 4.56228e-3 | 0.0189106 | 1.77570 | 93.90x |
| Ablation | 80 / 4.02415e-3 | 0.0284900 | 2.33087 | 81.81x |

The preceding same-pair solves all certified after three Newton iterations,
with 7–9 CG queries and three line trials. In the tuned case, the previous gap
was **2.79484e-6** and normalized dual gradient norm **2.85740e-5**; the new initial
state has gap **0.0373543** and gradient norm **0.787208**. Its normalized internal
nuclear objective is **2.1794891350091246**. The objective-scale jump is shared
by all five failures and associates changing EMA objectives with the conditioning
transition; it does not by itself prove a causal data-outlier mechanism.

For the tuned failing candidate the independently recomputed certificate is:

| Quantity | Value | Interpretation |
|---|---:|---|
| Primal objective | 1.198903588919219 | Feasible lower bound |
| Dual objective | 1.2454256126297372 | Independent original residual spectrum |
| Signed gap | 0.04652202371051817 | Positive, far from convergence |
| Normalized gap | 0.03735431746283596 | Fails 3e-5 |
| Normalized horizontality | 4.90818e-17 | Passes 1e-10 |
| Spectral upper bounds | 1.0 / 0.998466048624928 | Radially feasible |
| Spectral excess | 0 | Passes 1e-12 |
| Residual rcond | 8.71229e-5 | Fails strict >1e-4 |

The failing initial gaps range from 0.0374 to 0.1035, not a tiny-gap stopping
floor. All five initial candidates are horizontal and spectrally feasible,
with positive signed gap. The conditioning gate executes before gap acceptance,
CG, damping, and the 24-trial line search. Consequently there is **no failed
Newton direction, curvature decision, Armijo RHS, or rejected trial to reconstruct
for these transitions**. Their saved Newton/trial lists are empty, which is the
exact decision history rather than missing instrumentation.

As an isolated diagnostic, replacing only the initial multiplier by the
existing vertical-centering initialization also exits at iteration zero in
every fixture. Rconds are respectively **8.71820e-5, 7.72490e-5, 4.51961e-5,
7.93238e-5, 8.77500e-5** (tuned, low, middle, high, ablation). This is not a
proposed warm-start change; it shows the captured condition is not cured by
that simple initialization. It still does not prove absence of an admissible
smooth optimum.

## Successful controls and layer trajectories

The two originally completed 256-step pilots were reproduced through 101
completed updates. All completed losses and batch hashes matched. Research
observations cover steps 50–100, 306 solves per control. Across both controls
and the observed successful prefixes of failures, 1,531 pair solves were recorded
with normalization, gradient, and work metrics.

| Control paired LR (unsupported 3e-4) | Min accepted rcond | Min smooth rcond | Max gap | Median Newton / CG / line trials | Max Newton / CG / line trials |
|---|---:|---:|---:|---|---|
| 0.0015 | 1.98993e-4 | 1.97810e-4 | 2.98586e-5 | 3 / 9 / 3 | 3 / 13 / 3 |
| 0.0012247448713915891 | 2.14619e-4 | 2.14514e-4 | 2.88903e-5 | 3 / 9 / 3 | 4 / 16 / 4 |

Both controls also experience worsening conditioning and increased solve work.
They avoid the operational transition because their evaluated smooth states
remain above the guard, not because their early two-Newton behavior persists.
At step 87 the tuned 256 pilot's block-1 rcond is **1.22974e-3**, intrinsic
magnitude **0.0271330**, gap **2.40977e-6**, gradient norm **1.80163e-5**, with
three Newton iterations, ten CG queries, and three trials. The failing 512
block-1 objective at the same batch position is much larger and below guard.

| Zero-based step | Tuned 512 block-1 accepted rcond | Tuned 256 block-1 accepted rcond | 0.0015 pilot block-1 accepted rcond |
|---:|---:|---:|---:|
| 80 | 1.98032e-3 | 1.81635e-3 | 2.13663e-3 |
| 86 | 1.55538e-3 | 1.24104e-3 | 2.88964e-4 |
| 87 | Initial state fails: 8.71229e-5 | 1.22974e-3 | 3.02038e-4 |
| 92 | Not reached | 1.19802e-3 | 2.10460e-4 |
| 100 | Not reached | 3.28099e-4 | 2.76460e-4 |

All six pair-specific series, including objective magnitude, final normalized
gradient, gap, Newton/CG counts, and line trials, are in
[stage_d_forensic_pair_region.csv](../cluster/stage_d_forensic_pair_region.csv).
The controls support an empirical distinction between trajectories; they do not
guarantee future guard avoidance or optimizer-quality superiority.

## Schedule association, not retuning

For zero-based step t the existing schedule is `(t+1)/warmup_steps` during
warmup, then

```text
0.1 + 0.45 * (1 + cos(pi * (t - warmup_steps) / (total_steps - warmup_steps - 1)))
```

The actual paired LR is peak times this factor. Unsupported AdamW uses the same
factor, so **both parameter groups' schedules differ** between pilot and
confirmation.

| Step | Tuned paired LR, 256 / 26 | Tuned paired LR, 512 / 51 | 512 / 256 ratio |
|---:|---:|---:|---:|
| 25 | 0.00122474487 | 0.000624379738 | 0.5098 |
| 50 | 0.00119514073 | 0.00122474487 | 1.0248 |
| 64 | 0.00115153558 | 0.00122257410 | 1.0617 |
| 80 | 0.00108030371 | 0.00121397060 | 1.1237 |
| 86 | 0.00104834483 | 0.00120907449 | 1.1533 |
| 87 | 0.00104276554 | 0.00120817083 | 1.1586 |
| 92 | 0.00101384842 | 0.00120327940 | 1.1868 |
| 100 | 0.000964311969 | 0.00119417120 | 1.2384 |

At failure step 87 unsupported AdamW LR is **0.00025542434919639416** under the pilot
and **0.00029594020568380556** under confirmation. The longer schedule is initially
slower to warm up, then materially larger in the failure region. Because step
87 gradients and canonical EMA are already computed before its parameter update,
that step's LR cannot directly cause its initial residual failure; previous
scheduled updates change the model and hence the objective trajectory. The
measured schedule difference is associated with the divergence in conditioning.
No causal fraction can be assigned without another intervention, which this
study did not perform. The complete step-0–100 paired and unsupported LR series
is in [stage_d_forensic_schedule.csv](../cluster/stage_d_forensic_schedule.csv).

## Raw quotient state and normalization

| Case | Min raw up / down row norm | Min balanced row norm | Gauge root range | Canonical EMA RMS |
|---|---|---:|---|---:|
| Tuned | 0.353372 / 0.103052 | 0.195024 | 1.70425–1.99607 | 6.44366e-4 |
| Low | 0.349949 / 0.0972166 | 0.190271 | 1.71658–2.00148 | 8.30944e-4 |
| Middle | 0.354896 / 0.0976071 | 0.191030 | 1.71660–2.00149 | 2.06907e-3 |
| High | 0.379188 / 0.104127 | 0.203746 | 1.71667–2.00155 | 2.00234e-3 |
| Ablation | 0.356374 / 0.0975878 | 0.190990 | 1.71659–2.00149 | 2.62837e-3 |

The tuned raw up row norms range 0.353372–0.443469 and down row norms
0.103052–0.127222; balanced norms range 0.195024–0.233598. Weight RMS is
**0.0202995 / 0.00586682**, and raw gradient RMS **0.00654833 / 0.0262324**.
Recanonicalization from saved weights exactly reproduces the passed fp32
canonical U/D in all five fixtures. The regular weight domain is intact.

Intrinsic magnitude divided by the input objective Frobenius norm is at least
**0.9999999809** across all failures. There is no cancellation-dominated
cotangent. Tuned beta norm is **3.59150e-4**, whitening coordinates span
**3.02702–3.62573**, and row metric w spans **0.0760691–0.109136**.
Other fixtures' whitening coordinates remain finite in 2.62347–3.71632.
All detailed min/max row norms, RMS, beta and whitening statistics are retained
in the structured summary. Normalization does not explain the oracle rcond
disagreement, because there is effectively no disagreement.

The captured canonical objective is the updated fp32 EMA passed to the solver,
so its RMS and canonical EMA RMS are the same here; it is not the instantaneous
raw loss gradient. The raw gradients are stored separately in each fixture.

The last committed tuned step has model-wide parameter RMS **0.0271578**, update
RMS **6.65149e-5**, and update/parameter ratio **0.00244920**. Last committed
ratios in low, middle, high, and ablation are respectively **0.00147423,
0.00213314, 0.00413628, 0.00191084**. No failing-step update RMS exists because
that update was intentionally not applied. Large finite objective/gradient
changes must not be described as observed NaN or loss divergence.

## Separate bounded CPU reference

The unchanged CPU float64 reference was run on the tuned fixture alone, with
four CPUs, 8 GiB RAM, a six-minute SLURM allocation and an explicit **240-second
parent-process limit**. It was terminated at **240.169 seconds** without a
returned reference result. The latest completed convergence check was iteration
**2350**, at **234.022 seconds**:

- pair-Frobenius-normalized value gap: **6.97796e-10**;
- objective-normalized gap: **3.20185e-10**;
- ADMM residual: **1.69732e-7**;
- U residual rcond: **1.29543e-4**;
- D residual rcond: **8.67890e-5**.

The ADMM reference requires **both** its value gap and residual at `1e-11`.
It therefore did not satisfy its stopping rule. The explicit final `_reference`
post-check also was not reached. This bounded result establishes that the
unchanged reference is not a practical return path at the existing training
timeout for this fixture; it does not establish eventual nonconvergence.

An independently checked partial iterate saved at iteration **2300** has primal
value **1.2453477280846306**, dual **1.2453477285382746**, normalized gap
**3.64271e-10**, normalized horizontality **4.69829e-17**, and spectral excess
zero, but rcond **8.67889e-5**. It is a **partial research iterate**, not a
returned optimizer update. Its small value gap does not bypass the smooth guard
or establish minimum-Frobenius selection on a potentially nonunique face.
No direction/secondary-selection result was certified by the interrupted
reference. Its finite positive residual tail also does not prove exact rank loss.

## Validation, artifacts, and next question

All numerical work ran under SLURM on `lagrange0`; GPU work requested one H100,
four CPUs, 16 GiB. Runtime was Python 3.10.20, torch 2.10.0+cu128, CUDA 12.8,
NumPy 2.2.6. The complete suite, including three observer regression tests,
passed **403/403**, with no failures or skips, in **20.33 s** and independently
**17.80 s** in the representative allocation. Tests cover preserved solver
trajectory/counts, reason capture before reference execution, and hook restoration.
Network-guard checks passed. No package, data, environment, or production file
was changed. The pre-existing AGENTS.md edit was left intact.

Reproduction logs and scripts:

- [tuned capture log](../cluster/stage_d_forensics-29044.log),
  [representative/control log](../cluster/stage_d_forensics-29045.log);
- [tuned independent spectrum log](../cluster/analyze_stage_d_fixture-29046.log),
  [representative spectrum log](../cluster/analyze_stage_d_fixture-29048.log);
- [bounded CPU reference log](../cluster/stage_d_reference-29047.log);
- [capture runner](../cluster/stage_d_forensics.py) and
  [SLURM script](../cluster/stage_d_forensics.sbatch);
- [fixture analysis](../cluster/analyze_stage_d_fixture.py) and
  [bounded reference runner](../cluster/stage_d_reference.py);
- [compact JSON summary](../cluster/stage_d_forensic_summary.json),
  [JSON-only aggregator](../cluster/summarize_stage_d_forensics.py).

Tensor fixtures remain outside source storage:

```text
/home/prignano/qnormuon-runs/stage-d-forensics/capture-29044/
    confirmation-qso-a0.0003-q0.00122474487139/failed_pair.pt
/home/prignano/qnormuon-runs/stage-d-forensics/capture-29045/
    coarse-qso-a0.0003-q0.0003/failed_pair.pt
    coarse-qso-a0.0003-q0.001/failed_pair.pt
    coarse-qso-a0.0003-q0.004/failed_pair.pt
    ablation-qso-a0.0002-q0.00122474487139/failed_pair.pt
```

Each directory contains the immediate smooth event, independent spectral
analysis, provenance, and regional metrics. The tuned directory additionally
contains bounded-reference progress and a single overwritten partial fixture.
Successful controls are in the same capture-29045 root. Example replay commands
are `sbatch cluster/stage_d_forensics.sbatch --case tuned` and
`sbatch cluster/stage_d_forensics.sbatch --case representatives`; they stop at
the first reference request rather than completing a new optimizer-quality run.

**The single justified next research question is correct, efficient coupled
production behavior in the near-rank/ill-conditioned residual regime represented
by these fixed fixtures**, including whether an appropriate GPU path can preserve
the accepted primary objective and minimum-Frobenius optimal-face selection
where needed. First distinguish genuinely deficient faces from full-rank states
outside the current smooth admission guard; the present evidence does not settle
that distinction at the exact optimum. This is a separate study, not permission
to lower the guard, relax certificates, replace CPU ADMM, alter learning rates,
or introduce a rank heuristic. No further quality sweep or multi-seed comparison
is justified before this blocker is addressed.
