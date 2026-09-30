# Tiny Transformer: direct-fp64 Stage-C smoke and historical fp32 attempt

## Current Stage-C result: direct-fp64 paired solve

**Practicality classification: B — numerically stable but computationally
impractical for an LR sweep in the current implementation.** Both AdamW and
QSO completed the same 50-step, 102,400-token H100 smoke. QSO certified all
300 paired solves at the unchanged `3e-5` gap target with **zero fallbacks**,
finite losses/updates, and exact full-size checkpoint continuation. Its warm
median step took **2.665 s**, about **80 times** AdamW's **0.0334 s**;
the paired optimizer occupied about **99%** of measured QSO step time. This
resolves the earlier fp32 certification failure but does not establish a
practical training cost. No LR sweep, multi-seed comparison, gauge trajectory,
or long run was started.

Job **28914** completed on `lagrange0`, one NVIDIA H100 NVL, 4 CPUs, 16 GiB,
SLURM elapsed **2m45s**, exit `0:0`. Runtime was Python 3.10.20,
torch 2.10.0+cu128, NumPy 2.2.6; model storage fp32, forward/backward bf16
autocast, canonical momentum fp32, paired solver/SVD/recovery/certificate
fp64, TF32 disabled. Raw log:
[run_tiny_smoke-28914.log](../cluster/run_tiny_smoke-28914.log). The two
structured run directories are
`/home/prignano/qnormuon-runs/tiny-transformer/smoke-28914-adamw-seed2026/`
and
`/home/prignano/qnormuon-runs/tiny-transformer/smoke-28914-qso-seed2026/`.
They contain provenance, per-step JSONL, per-solve JSONL, summaries and one
checkpoint each. The local network guard was active; no asset download was
requested or observed.

The model remained **11,457,408 parameters**, with six `[1024,384]` coupled
pairs. Both methods used initialization SHA256
`0607adc4c9cffa7b902c0956fa7477ea274078fdc31406d84d527b79ca6a2401`
and identical batch hashes at all 50 steps. FineWeb SP1024 token-prefix hashes,
batch size 16, context 128, seed 2026, five-step warmup, cosine schedule,
evaluation batches, AdamW peak LR `3e-4`, QSO paired peak LR `1e-3`, and
unsupported-parameter AdamW were unchanged from the first attempt. The only
intended optimizer policy change was **direct fp64 paired solving** at the
existing `3e-5` target and `1e-4` rcond guard; no fp32-first retry occurs.

| Measurement, 50 completed steps | AdamW | QSO + AdamW |
| --- | ---: | ---: |
| First / last training loss | 7.010503 / 6.022851 | 7.010503 / 6.025261 |
| Initial / final validation loss | 6.998700 / 5.982066 | 6.998700 / 5.984854 |
| Measured training-step sum | 1.942 s | 137.579 s |
| Measured optimizer sum | 0.149 s | 136.178 s |
| Optimizer fraction of step sum | 7.65% | 98.98% |
| Overall measured-step throughput | 52,731 tokens/s | 744 tokens/s |
| Peak PyTorch GPU allocation | 745,862,144 B (711 MiB) | 724,071,424 B (691 MiB) |
| Three-step checkpoint replay | exact | exact |

The validation losses at steps 10/20/30/40/50 were AdamW
`6.223100/6.030417/5.999305/5.988474/5.982066` and QSO
`6.229504/6.032339/6.001319/5.990697/5.984854`. Both decreased on this
single short run; the unswept learning rates do not support an optimizer-quality
ranking. All gradient, parameter and update finiteness checks passed. QSO's
first and last update/parameter RMS ratios were approximately `0.001687` and
`0.000145`. Its measured live GPU allocation averaged 242.0 MB over steps
5–9 and 240.5 MB over the last five steps, with no observed growth. PyTorch
allocation excludes CUDA context, driver memory and CPU memory.

### QSO certificates, warm starts, and fallbacks

All **300/300** smooth pair solves converged. The normalized gap had median
`2.27e-6`, p95 `1.89e-5`, maximum `2.99575e-5` against the unchanged
`3e-5` limit. The maximum raw horizontal residual was `2.08e-17`; spectral
excess was zero. The smallest residual rcond was `9.78e-4`, above the
`1e-4` guard. The gap maximum is close to the threshold; this is 50-step
empirical certification, not a margin guarantee for other trajectories.

Newton certification bins were **0/0/269/31** for 0/1/2/3+ iterations; the
maximum was 3. The cold first step contributed 0/0/1/5, and the remaining
49 steps contributed 0/0/268/26. CG iterations had mean `4.79`, median 5,
p95 6, maximum 9. Warm states therefore usually certified at the initial
two-iteration budget, though warm starting did not reduce solves to zero or
one iteration in this trajectory. **Fallback count and CPU ADMM time were zero**;
there are no fallback reasons to report. The generic CPU reference solver was
not part of the measured normal path.

### Synchronized timing and practicality

Timings below use steps 5–49, excluding five identified first-use/warmup
steps. Values are per full optimizer step across six pairs. Instrumentation
synchronizes CUDA around nested solver functions and is observational; it may
add overhead. Nested categories **overlap**: residual SVD/polar includes SVD,
certificate includes `svdvals`, and CG includes HVP. They must not be summed.

| Timed category | Median | p95 |
| --- | ---: | ---: |
| AdamW full step | 0.03337 s | 0.03942 s |
| QSO full step | 2.66489 s | 2.79889 s |
| QSO forward/backward | 0.01279 s | 0.01366 s |
| QSO complete optimizer | 2.63728 s | 2.77061 s |
| QSO six paired solver calls | 2.37244 s | 2.50700 s |
| Unsupported-parameter AdamW within QSO | 0.00193 s | 0.00210 s |
| Residual SVD/polar evaluations | 0.78513 s | 0.83163 s |
| All `torch.linalg.svd` calls | 0.77623 s | 0.82233 s |
| Recovery/certificate, including `svdvals` | 1.53964 s | 1.62561 s |
| All `torch.linalg.svdvals` calls | 1.52049 s | 1.60539 s |
| Newton CG, including HVP | 0.02214 s | 0.02438 s |
| HVP alone | 0.01429 s | 0.01577 s |
| CPU reference ADMM fallback | 0 s | 0 s |

The **cold first QSO step** took 3.557 s, including 3.528 s in the optimizer
and 3.264 s in six pair solves; it is retained in the raw output and overall
totals. QSO's warm median throughput was 769 tokens/s versus AdamW's
61,368 tokens/s. The dominant measured work is GPU SVD and certificate
spectral-value evaluation; CG/HVP is small. The current 50-step result therefore
passes numerical stability but fails the present practicality gate for an LR
sweep. Performance optimization is a separate task and was not attempted here.

### Checkpoint and remaining scope

Both methods saved after step 25 and replayed the next three minibatches from
disk. Model parameters, losses and optimizer state all matched the uninterrupted
run with maximum recorded error **zero**. The QSO checkpoint was 118,734,531
bytes and includes canonical momentum, original lambda, step count, unsupported
AdamW and scheduler/RNG state; AdamW's was 137,563,275 bytes. No memory-growth
or nonfinite issue appeared over 50 steps. One seed and one LR per method leave
quality, LR sensitivity, longer-run reliability and gauge-reset trajectories
unresolved. The earlier `1e-8` warm-start line-search failure remains a separate
known limitation and was not invoked by this `3e-5` production run.

## Historical record: first fp32 Stage-C attempt (job 28897)

**Stage C did not pass for QSO.** AdamW completed 50 steps on the H100. QSO
encountered repeated costly reference fallbacks during its first step and hit
the declared 180-second step limit before committing an update. No LR sweep,
multi-seed comparison, optimizer redesign, or follow-up training run was started.
This is a negative practicality result for the tested production-v0 fp32
configuration, not a comparison of optimization quality.

## A. Infrastructure and smoke validation

### Harness and local assets

- Harness: [benchmarks/tiny_transformer.py](../benchmarks/tiny_transformer.py).
- Configuration: [configs/tiny_transformer/smoke.json](../configs/tiny_transformer/smoke.json).
- Stage B: [test_tiny_harness.sbatch](../cluster/test_tiny_harness.sbatch).
- Stage C: [run_tiny_smoke.sbatch](../cluster/run_tiny_smoke.sbatch) and
  [driver](../cluster/run_tiny_smoke.py).
- Full bounded asset inventory: [TINY_TRANSFORMER_ASSETS.md](TINY_TRANSFORMER_ASSETS.md).

Training uses existing FineWeb SP1024 binary token shards in
`/home/prignano/parameter-golf/data/datasets/fineweb10B_sp1024/`.
Only the first **1,048,576 training tokens** and **65,536 validation tokens**
are read, inside the allocation. The local SentencePiece model exists and its
manifest identifies the vocabulary; no tokenizer library is needed to train
on existing IDs. Header, file size, token range and prefix hashes are checked.
A deterministic synthetic stream is also implemented for tests or future
explicit synthetic configurations. Missing real assets fail; no online fallback
exists.

A step deterministically selects its sequences from a stateless `(data seed,
batch index)` generator. Training and validation have distinct source files
and batch seeds. Both methods share initialization, minibatches, validation
batches, loss, schedule, precision and step budget. Initialization SHA256 was
identical: `0607adc4c9cffa7b902c0956fa7477ea274078fdc31406d84d527b79ca6a2401`.
Per-step batch hashes are recorded for completed steps. The diagnostic replay
reproduced QSO's first loss and objective scale exactly.

### Architecture and settings

| Setting | Value |
| --- | --- |
| Parameters | **11,457,408** |
| Layers / width / heads | 6 / 384 / 6 |
| SwiGLU hidden width | 1,024 |
| QSO mathematical pair shape | `[1024,384]`, six pairs |
| QSO-owned parameter count | 4,718,592 |
| Unsupported parameter count | 6,738,816, handled by AdamW |
| Vocabulary / context | 1,024 / 128 |
| Batch / accumulation | 16 / 1; 2,048 tokens per optimizer step |
| Seed | 2026 for initialization and common data configuration |
| Parameter/optimizer storage | fp32 parameters; QSO canonical EMA fp32, lambda fp64 |
| Forward/backward | bf16 autocast; cross entropy and RMSNorm reductions fp32 |
| QSO spectral arithmetic | fp32; recovery/certification fp64 |
| TF32 | Explicitly disabled |
| Determinism | PyTorch deterministic algorithms; cuBLAS workspace configured; no dropout |
| Schedule | Five-step linear warmup, then cosine to 10% of peak LR |
| Weight decay / clipping | 0 / none |
| Evaluation | Four fixed validation batches initially and every ten steps |

The model is a pre-norm causal decoder: token and learned absolute positional
embeddings, causal scaled-dot-product attention, RMSNorm (`epsilon=1e-6`),
SwiGLU, residual additions, final RMSNorm and an untied output projection.
Linear layers have no biases. Weights use normal initialization with standard
deviation .02; attention output and MLP down matrices are additionally divided
by `sqrt(2*layers)`. Down projections are nonzero at initialization. RMSNorm's
epsilon is unrelated to the quotient metric; the optimizer is unchanged.

Only `up_proj.weight` and `down_proj.weight` are registered with QSO, by stable
layer names and explicit transpose-aware pairs. Gates, attention, embeddings,
norms, positional embeddings and output weights belong to AdamW. Tests verify
complete, disjoint ownership with no silently unoptimized parameter.

### Validation and execution

Stage B job **28896**: **201 passed, 0 failed, 0 skipped**, 15.52 seconds of
pytest, 19 seconds allocation time. This comprises all existing 189 cases and
12 small harness checks for causality, data boundaries and determinism,
parameter ownership, schedule, finite delta-X algebra, checkpoint continuation,
and gauge-reset function/state conventions. Numerical work ran in SLURM.
Raw output: [test_tiny_harness-28896.log](../cluster/test_tiny_harness-28896.log).

Stage C job **28897**, `lagrange0`: one NVIDIA H100 NVL, 4 CPUs, 16 GiB RAM,
30-minute allocation limit. Runtime: Python 3.10.20, torch 2.10.0+cu128,
CUDA build 12.8, NumPy 2.2.6; pytest 9.0.3 for Stage B. Job elapsed 3m17s,
exit `1:0` due to QSO's explicit timeout. Both methods requested 50 steps.
Raw output: [run_tiny_smoke-28897.log](../cluster/run_tiny_smoke-28897.log).

| Smoke measurement | AdamW | Production QSO + AdamW |
| --- | ---: | ---: |
| Completed optimization steps | **50** | **0** |
| Training tokens in committed steps | **102,400** | **0** |
| Initial validation loss | 6.998700 | 6.998700 |
| Final validation loss | **5.982066** | Not measured after an update |
| First training loss | 7.010503 | 7.010503, reproduced by diagnostic |
| Last completed-step training loss | 6.022851 | None |
| Sum of measured training-step time | 1.66806 s | No completed step |
| Sum of optimizer time | 0.14824 s | First three layer solves alone: 144.28 s |
| Tokens/sec during measured steps | **61,389** | Not established |
| Optimizer fraction of measured step time | **8.89%** | Not established for a completed step |
| Peak PyTorch CUDA allocation | 745,862,144 bytes (~711 MiB) | 473,767,936 bytes (~452 MiB) before abort |
| Checkpoint replay | Three steps, exact match | Full-size checkpoint not reached |

AdamW's end-of-step live CUDA allocation was exactly **253,310,976 bytes** in
both steps 5–9 and the last five steps: no measured memory growth over this
short run. QSO cannot be assessed for memory growth from a single incomplete
step. CUDA allocation excludes the driver/context and CPU memory; the peak
includes evaluation and, for AdamW, the temporary resumed model. No NaN/Inf
was observed: loss/gradients were checked before optimizer entry, and every
completed update was checked afterward. QSO returned three finite certified
fallback solutions, but never committed a paired or unsupported-parameter
update because the paired optimizer validates all pairs first.

Timings synchronize CUDA. `step_seconds` includes batch preparation, forward/
backward, optimizer execution and compact diagnostics, including parameter
snapshots; `optimizer_seconds` excludes forward/backward but includes mandatory
QSO certification and enabled cast diagnostics. Evaluation and checkpoint/
resume replay are outside per-step throughput. AdamW's total run wall time,
including evaluation and checkpoint checks, was 3.27110 s. Short runs and
first-use overhead make these smoke timings unsuitable as mature throughput
claims. Per-step compute-only optimizer fractions are also in JSONL.

## B. Learning-rate sweep

**Not started.** The staging rule requires review after Stage C, and QSO failed
the smoke practicality check. No LR combinations were submitted automatically.
The interface accepts method-specific rates through configuration; it does not
force the paired optimizer to share AdamW's rate.

Early-scale measurements for AdamW (all parameters):

| Metric | Step 0 | Step 49 |
| --- | ---: | ---: |
| Gradient RMS | 0.00166239 | 0.000175902 |
| Parameter RMS before update | 0.0270490 | 0.0270978 |
| Actual update RMS | 0.0000594464 | 0.00000376322 |
| Update / parameter RMS | 0.00219773 | 0.000138875 |

The first five AdamW update/parameter ratios have mean 0.00375098 and maximum
0.00490757. QSO has no accepted model-update scale yet. A measured LR range
for QSO therefore cannot be chosen from this run. The solver problem depends
on momentum before the learning-rate lift; simply sweeping smaller first-step
LRs would not repair the observed inner-solve issue.

## C. Hyperparameters and baseline fidelity

These are **initial smoke settings, not selected hyperparameters**:

- AdamW: peak LR `3e-4`, betas `(.9,.95)`, epsilon PyTorch default `1e-8`,
  weight decay zero; foreach/fused disabled for an explicit common interface.
- QSO: peak paired LR `1e-3`, canonical EMA beta `.95`; unsupported parameters
  use the same AdamW settings above. Solver defaults remain fp32 gap `3e-5`,
  rcond guard `1e-4`, initial budget 2, maximum Newton iterations 100, CG 30,
  reference fallback enabled. No acceptance threshold was changed.
- Historical separate-polar `QNorMuon` is exposed through the unchanged local
  implementation with its documented defaults (`balance_lr=.1`,
  `balance_beta=.9`, `eps=1e-12`, no weight canonicalization). It is explicitly
  historical, including its existing shared-K behavior; no new balancing logic
  is introduced or used by production QSO. Ownership is tested, but this
  baseline was not trained in Stage C.
- Muon is omitted. Existing neighboring research projects contain specialized
  training code, but fidelity of a reusable matching baseline was not verified.
  No substitute Muon implementation was invented.

## D. Multi-seed comparison

**Not run.** Only the common seed 2026 was used for the attempted smoke.
Three-seed comparisons require successful smoke, reviewed LR ranges, and later
optimizer-specific sweep selection. No superiority claim is supported here.

## E. QSO solver behavior and failure analysis

The raw compact per-solve log preserves partial-step work, including the pair
identity, step, canonical momentum/objective RMS, dtype, elapsed time, Newton/
CG counts, rcond, gap and fallback reason. Three of three **completed pair
solves** used fallback; a fourth was interrupted inside CPU ADMM.

| Pair, all at step 0 | Newton / CG | Total solve time | Final fallback gap | Final rcond | Objective RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `blocks.0.mlp` | 5 / 14 | 52.13 s | `1.4761e-13` | .00758757 | `3.2456e-5` |
| `blocks.1.mlp` | 8 / 24 | 42.61 s | `1.2460e-13` | .00788653 | `2.6289e-5` |
| `blocks.2.mlp` | 14 / 32 | 49.55 s | `1.3933e-13` | .00590656 | `2.5157e-5` |
| `blocks.3.mlp` | Incomplete | Interrupted at 180 s total step time | Unknown | Unknown | `2.2567e-5` |

All three completed solves reported `line_search_failed` before invoking the
unchanged generic reference fallback. Newton counts have mean 9, median 8,
p95 13.4 and max 14; CG counts mean 23.33, median 24, p95 31.2, max 32.
The 0/1/2/3+ Newton bins are 0/0/0/3, but these are **failed smooth attempts
followed by certified fallback**, not counts of successful smooth certification.
No warm training transition was reached. The slow-trajectory two-iteration
observation from DUAL_SOLVER_STUDY is neither reproduced nor refuted for warm
stochastic training by this cold first-step failure.

A declared alarm interrupted the fourth solve while executing CPU
`torch.linalg.svd` in `experiments.horizontal_spectral.ball_project`, called by
reference ADMM. The exact traceback is retained in the raw SLURM log and the
QSO run's `failure.json`. The initial failure summary counted only completed
**training steps**, so its nested solver total is zero; that is not zero solver
activity. The authoritative partial-step evidence is `solves.jsonl` and the
[consolidated analysis](../cluster/tiny_smoke_analysis-28897.json). The driver
has since been corrected to distinguish completed-step statistics from observed
partial-step solves. This reporting-only correction and removal of a scalar
conversion warning do not change the measured training or optimizer algorithms.

### Bounded one-pair diagnosis

Job **28898** recreated the identical first minibatch and first pair
`[1024,384]`, with no model update. Fallback was disabled only in this diagnostic
to inspect the failed smooth result. The same represented inputs were then
promoted to fp64 for comparison. No configuration was adopted for another
training run.

| First pair diagnostic | fp32 | fp64 |
| --- | ---: | ---: |
| Result | Line-search failure, uncertified | Certified |
| Normalized gap | **5.15922e-5** | **1.68834e-10** |
| Requested default gap target | `3e-5` | `1e-8` |
| Newton / CG | 5 / 14 | 4 / 10 |
| SVD matrix evaluations | 96 | 30 |
| Measured solve time | 2.00695 s | 0.64839 s |
| Final residual rcond | .00758758 | .00758757 |
| Normalized horizontal residual | `2.44e-17` | `2.87e-17` |
| Spectral excess | 0 | 0 |

This is consistent with a numerical limitation in the current fp32 smooth
SVD/line-search/certification path at these training inputs. It is not a rank
truncation issue or evidence that the coupled variational theory is incorrect:
rcond is above the configured guard, and the higher-precision solve certifies.
The exact numerical source within that path has not been isolated. The
reference fallback successfully certifies values but is impractically costly
for the attempted smoke. The single-pair fp64 timing is not a full-training
performance estimate, and fp64 has not been validated for 50 training steps.
Raw diagnostic history: [diagnose_tiny_solver-28898.log](../cluster/diagnose_tiny_solver-28898.log).

The optimizer sources are unchanged, with the same validated SHA256 values:
`df6552136f7a5c5e121622e97abd1e30c70762a0ee8422d61a02df54bd603ec6`
(`optimizer.py`) and
`8345e386f00d9d9937d18ca512198bc3e37367fbb4a45f826874411e63b868c2`
(`coupled_solver.py`). No tolerance weakening or algorithm redesign occurred.

## F. Gauge-reset experiment and finite-step monitoring

The harness provides a seeded positive gauge reset with factors in `[1/4,4]`.
QSO canonical momentum and original-coordinate lambda remain unchanged. For
the AdamW control, raw first moments transform as covectors and second moments
by squared scale; this does not make AdamW equivariant. Tiny float64 Transformer
tests verify the immediate network function and QSO state convention after
populated-state resets. This is a correctness check, **not** a measured full-size
post-reset training trajectory. The planned matched continuation experiment is
deferred because Stage C failed.

Every completed smoke step measures the first pair's actual finite invariant
matrix change using
`deltaX = delta_d u^T + d delta_u^T + delta_d delta_u^T`.
Per-row three-term Gram products give its Frobenius norm using O(m*n) storage;
no `[m,n,n]` tensor is allocated. The test independently checks explicit small
outer products. AdamW's first/last relative delta-X norms were .0108322 and
.000732057. QSO produced no committed finite factor step to measure.

## G. Checkpoint/resume validation

The harness saves model parameters, paired canonical momentum, original lambda,
step counts, AdamW state, explicit schedule/base rates/next step, configuration,
dataset prefix provenance, and Python/NumPy/torch CPU/CUDA RNG states. Loading
rejects mismatched configuration/data. Stateless batch indices reproduce future
minibatches without a serialized iterator. Checkpoints are local trusted files.

AdamW saved after 25 updates and replayed updates 26–28 from disk against the
uninterrupted trajectory. Maximum parameter, loss, and optimizer-state errors
were all **exactly zero**. The checkpoint was **137,563,211 bytes**. Small
Transformer tests likewise validated real training save/load continuation for
QSO, but full-size QSO never reached checkpoint step 25. Therefore full-size
QSO checkpoint/resume remains **unvalidated**, not inferred from AdamW or the
unit tests.

Artifacts are under:

- `/home/prignano/qnormuon-runs/tiny-transformer/smoke-28897-adamw-seed2026/`
- `/home/prignano/qnormuon-runs/tiny-transformer/smoke-28897-qso-seed2026/`

Each has provenance and compact metrics or failure/solve logs. Only AdamW has
a checkpoint and successful summary. No large tensor debug dump is produced.
Reproduction commands, to run only when another smoke is authorized:

```
sbatch --parsable cluster/test_tiny_harness.sbatch
sbatch --parsable cluster/run_tiny_smoke.sbatch
```

The second command now reproduces the direct-fp64 Stage-C configuration reported
above; it is not a sweep launcher. This historical section describes the earlier
fp32 run and should not be read as the current solver policy.

## H. Limitations and decision

- QSO's 50-step smoke requirement was **not achieved**. Stability over many
  updates, warm solver behavior, steady-state memory, throughput, LR sensitivity,
  full-size checkpoint continuation and gauge-reset trajectory are unresolved.
- AdamW's decreasing short-run loss validates infrastructure, not superiority.
  LRs are unswept and only one seed was run.
- Data is a small fixed prefix of locally available real shards. Full-corpus
  provenance, semantic train/validation overlap and tokenizer quality are not
  newly audited. No decoding or generation-quality evaluation is performed.
- Personal `/home` quota is unknown. The actual small checkpoint write succeeded;
  aggregate free space is not proof of capacity for later sweeps.
- No optimizer source changed, no package was installed, and no dataset,
  tokenizer, checkpoint, model weight or large artifact was downloaded.
  The Python network guard was active and produced no attempted-network log.
- All numerical tests, training, checkpoint I/O and one-pair diagnosis ran in
  SLURM; frontend work was limited to source/config/report edits and lightweight
  metadata/log inspection.

**Stop at Stage C. Do not start the LR sweep until the failed QSO smoke and its
numerical diagnosis have been reviewed.**
