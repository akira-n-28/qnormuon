# Stage D: single-seed optimizer-specific learning-rate study

**B: tuned QSO and tuned AdamW are inconclusive / mixed on this single-seed study.**

## A. Frozen setup and provenance

All candidates ran on one H100 allocation, job `29024`. Production sources and defaults were unchanged; source SHA256 values, full configs, token-prefix hashes and all expected batch hashes are recorded in `/home/prignano/qnormuon-runs/lr-stage-d/study-29024/manifest.json` and each run provenance.

The complete suite passed **400 tests, zero failed, zero skipped**, in 16.66 s before the sweep. Raw validation: `cluster/lr-tests-29024.xml` and `cluster/run_lr_study-29024.log`. The study completed all 20 declared candidate runs: 11 passed and 9 QSO timeout failures, retained as data. Report/plot analysis used separate CPU-only SLURM allocations; no additional training was performed.

The common model is the 11,457,408-parameter six-layer SwiGLU Transformer, six `[1024,384]` pairs, seed 2026. Initialization SHA256 was asserted to equal `0607adc4c9cffa7b902c0956fa7477ea274078fdc31406d84d527b79ca6a2401` for every candidate. Every completed training minibatch hash was independently precomputed and asserted.

The production policy is K=I, explicit pairing, fp32 canonical EMA, direct fp64 full thin SVD, previous original-coordinate lambda, projected-primal `gram_upper`, diagnostics on and cast diagnostics off. Gap/rcond/horizontality/spectral/signed-gap thresholds remain `3e-5 / >1e-4 / 1e-10 / 1e-12 / >=-1e-10`. No research decomposition/predictor was enabled. Unsupported parameters use AdamW; QSO and AdamW have their own peak learning rates.

Model storage fp32, bf16 autocast, TF32 off, deterministic algorithms, no dropout, weight decay zero and no clipping. AdamW betas are `(0.9,0.95)`; QSO EMA beta is `0.95`. Batch 16, context 128, accumulation one: 2,048 tokens per step. Validation uses the same four fixed smoke batches. All local paths are in the saved configs. No network fallback or download occurred.

**Stage-D data clarification:** the old smoke samples windows with replacement. To satisfy the requested nonrepeated confirmation, Stage D uses one seed-2026 permutation of disjoint 128-input-token blocks. Every candidate shares the same order, and 256-step runs use its first half. The existing local shard supplies 1,048,576 distinct input positions plus one lookahead label token; header, size, range and prefix were verified in the allocation before training. This intentionally differs from the Stage-C sampled-window stream, not between Stage-D methods. Validation is unchanged. Repeated token *IDs* are natural; input *positions* do not repeat.

Pilot: 256 steps / 524,288 tokens, validation at step 0 then every 32 steps through 256, with 26 warmup steps. Confirmation: 512 steps / 1,048,576 tokens, validation at step 0 then every 32 steps through 512, with 51 warmup steps. Both decay by the same cosine to 10% of peak. The final three checkpoints are 192/224/256 or 448/480/512.

## B. Predeclared grids and selection

AdamW: `1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1.2e-3`. QSO paired: `3e-4, 6e-4, 1e-3, 1.5e-3, 2.5e-3, 4e-3`, unsupported AdamW fixed `3e-4`.

Selection is lexicographic: last-three validation mean, final validation, all post-initial validation mean, then lower maximum update/parameter RMS. Failed candidates are ineligible but retained. Training loss and wall time never enter selection. Refinement uses the declared geometric neighbors/boundary extension. Failure limits were fixed before execution: loss/validation >20, nonfinite values, uncertified pair,180-second step timeout, or reference fallback fraction >20% after12 observed pairs. Solver settings are never rescued by retuning.

## C. Complete coarse sweep

| Method | AdamW LR | Paired QSO LR | Status / completed steps | Last-three mean | Final | Post-initial mean | Max update ratio |
|---|---:|---:|---|---:|---:|---:|---:|
| adamw | 0.0001 | — | passed / 256 | 5.294002 | 5.255392 | 5.618990 | 0.001135 |
| adamw | 0.0002 | — | passed / 256 | 5.025711 | 4.980426 | 5.424090 | 0.002333 |
| adamw | 0.0003 | — | passed / 256 | 5.265753 | 5.188641 | 5.585298 | 0.004217 |
| adamw | 0.0005 | — | passed / 256 | 5.122214 | 5.029074 | 5.514803 | 0.007078 |
| adamw | 0.0008 | — | passed / 256 | 5.070175 | 5.011187 | 5.396337 | 0.009025 |
| adamw | 0.0012 | — | passed / 256 | 5.278760 | 5.238416 | 5.547681 | 0.015966 |
| qso | 0.0003 | 0.0003 | failed / 92 | — | — | — | — |
| qso | 0.0003 | 0.0006 | failed / 89 | — | — | — | — |
| qso | 0.0003 | 0.001 | failed / 87 | — | — | — | — |
| qso | 0.0003 | 0.0015 | passed / 256 | 4.949966 | 4.895971 | 5.425742 | 0.002968 |
| qso | 0.0003 | 0.0025 | failed / 89 | — | — | — | — |
| qso | 0.0003 | 0.004 | failed / 56 | — | — | — | — |

## D. Deterministic refinement

| Method | AdamW LR | Paired QSO LR | Status / completed steps | Last-three mean | Final | Post-initial mean | Max update ratio |
|---|---:|---:|---|---:|---:|---:|---:|
| adamw | 0.00014142136 | — | passed / 256 | 5.105138 | 5.066355 | 5.474810 | 0.001524 |
| adamw | 0.00024494897 | — | passed / 256 | 5.174663 | 5.108950 | 5.548525 | 0.003401 |
| qso | 0.0003 | 0.0012247449 | passed / 256 | 4.934507 | 4.881274 | 5.402132 | 0.002581 |
| qso | 0.0003 | 0.0019364917 | failed / 73 | — | — | — | — |

Coarse winners: AdamW `0.0002`, QSO `0.0015`. Combined coarse/refinement AdamW winner `0.0002`. The paired QSO winner was fixed before unsupported-parameter ablation.

## E. Unsupported-parameter AdamW ablation

| Method | AdamW LR | Paired QSO LR | Status / completed steps | Last-three mean | Final | Post-initial mean | Max update ratio |
|---|---:|---:|---|---:|---:|---:|---:|
| qso | 0.0003 | 0.0012247449 | passed / 256 | 4.934507 | 4.881274 | 5.402132 | 0.002581 |
| qso | 0.0002 | 0.0012247449 | failed / 81 | — | — | — | — |
| qso | 0.0005 | 0.0012247449 | failed / 83 | — | — | — | — |

The already-tested `3e-4` run is reused because its entire configuration matches; no extra training is needed. The paired LR is fixed across all three ablation entries.

Other recorded quality summaries for complete runs (failed horizons remain unavailable):

| Candidate | Minimum validation | Mean post-initial | Last-three mean | Normalized validation AUC |
|---|---:|---:|---:|---:|
| coarse-adamw-a0.0001-q0.001 | 5.255392 | 5.618990 | 5.294002 | 5.727947 |
| coarse-adamw-a0.0002-q0.001 | 4.980426 | 5.424090 | 5.025711 | 5.550232 |
| coarse-adamw-a0.0003-q0.001 | 5.188641 | 5.585298 | 5.265753 | 5.698427 |
| coarse-adamw-a0.0005-q0.001 | 5.029074 | 5.514803 | 5.122214 | 5.637904 |
| coarse-adamw-a0.0008-q0.001 | 5.011187 | 5.396337 | 5.070175 | 5.520557 |
| coarse-adamw-a0.0012-q0.001 | 5.238416 | 5.547681 | 5.278760 | 5.657699 |
| coarse-qso-a0.0003-q0.0015 | 4.895971 | 5.425742 | 4.949966 | 5.557163 |
| refinement-adamw-a0.000141421356237-q0.001 | 5.066355 | 5.474810 | 5.105138 | 5.595581 |
| refinement-adamw-a0.000244948974278-q0.001 | 5.108950 | 5.548525 | 5.174663 | 5.666635 |
| refinement-qso-a0.0003-q0.00122474487139 | 4.881274 | 5.402132 | 4.934507 | 5.534471 |
| confirmation-adamw-a0.0002-q0.001 | 4.512301 | 4.982804 | 4.522282 | 5.060504 |

## F. Tuned 512-step equal-token confirmation

| Method | AdamW LR | Paired QSO LR | Status / completed steps | Last-three mean | Final | Post-initial mean | Max update ratio |
|---|---:|---:|---|---:|---:|---:|---:|
| adamw | 0.0002 | — | passed / 512 | 4.522282 | 4.512301 | 4.982804 | 0.002912 |
| qso | 0.0003 | 0.0012247449 | failed / 87 | — | — | — | — |

| Completed step | Tokens | AdamW validation | QSO validation | QSO − AdamW |
|---:|---:|---:|---:|---:|
| 0 | 0 | 6.998700 | 6.998700 | +0.000000 |
| 32 | 65536 | 6.219527 | 6.114852 | -0.104675 |
| 64 | 131072 | 5.951776 | 5.973776 | +0.022000 |
| 96 | 196608 | 5.773079 | unavailable | unavailable |
| 128 | 262144 | 5.424902 | unavailable | unavailable |
| 160 | 327680 | 5.158140 | unavailable | unavailable |
| 192 | 393216 | 4.973089 | unavailable | unavailable |
| 224 | 458752 | 4.839091 | unavailable | unavailable |
| 256 | 524288 | 4.761232 | unavailable | unavailable |
| 288 | 589824 | 4.691720 | unavailable | unavailable |
| 320 | 655360 | 4.638597 | unavailable | unavailable |
| 352 | 720896 | 4.602895 | unavailable | unavailable |
| 384 | 786432 | 4.574163 | unavailable | unavailable |
| 416 | 851968 | 4.549817 | unavailable | unavailable |
| 448 | 917504 | 4.532738 | unavailable | unavailable |
| 480 | 983040 | 4.521806 | unavailable | unavailable |
| 512 | 1048576 | 4.512301 | unavailable | unavailable |

Signed summary differences (negative favors QSO):

| Metric | QSO − AdamW |
|---|---:|
| Final validation loss | unavailable |
| Minimum validation loss | unavailable |
| Last-three validation mean | unavailable |
| Post-initial validation mean | unavailable |
| Normalized validation AUC | unavailable |

**The tuned QSO 512-step confirmation failed.** The successful pilot does not establish a useful certified 512-step trajectory. Principal final/best/last-three/mean/AUC differences are unavailable; early common-checkpoint differences are diagnostics only. No partial loss substitutes for the missing target horizon. No extra seeds are justified.

AUC is trapezoidal validation-loss area divided by the common token horizon, including step0; the separately reported post-initial mean excludes step0. No metric disagreement is resolved by choosing a favorable checkpoint.

## G. QSO numerical validity

| Candidate | Pairs | Newton 0/1/2/3+ | CG mean/median/p95 | Max/p95 gap | Min rcond | Max normalized horizontal | Max spectral excess | ADMM | Line trials |
|---|---:|---|---|---|---:|---:|---:|---:|---:|
| coarse-qso-a0.0003-q0.0015 | 1536 | 0/0/364/1172 | 9.23/10/14 | 2.99e-05/1.69e-05 | 0.000199 | 1.29e-16 | 0 | 0 | 4244 |
| refinement-qso-a0.0003-q0.00122474487139 | 1536 | 0/0/378/1158 | 8.41/9/12 | 2.89e-05/1.62e-05 | 0.000215 | 9.68e-17 | 0 | 0 | 4232 |

Every observed completed QSO result was checked before the optimizer could commit it. Signed normalized gap and finiteness were also checked. Full per-pair metrics/reasons and partial-step solve-begin events are preserved in `solves.jsonl`.

Failures:

- `coarse-qso-a0.0003-q0.0003`: TimeoutError('unchanged 180-second per-step budget exceeded'); completed 92 steps; exact traceback in run `failure.json`.
- `coarse-qso-a0.0003-q0.0006`: TimeoutError('unchanged 180-second per-step budget exceeded'); completed 89 steps; exact traceback in run `failure.json`.
- `coarse-qso-a0.0003-q0.001`: TimeoutError('unchanged 180-second per-step budget exceeded'); completed 87 steps; exact traceback in run `failure.json`.
- `coarse-qso-a0.0003-q0.0025`: TimeoutError('unchanged 180-second per-step budget exceeded'); completed 89 steps; exact traceback in run `failure.json`.
- `coarse-qso-a0.0003-q0.004`: TimeoutError('unchanged 180-second per-step budget exceeded'); completed 56 steps; exact traceback in run `failure.json`.
- `refinement-qso-a0.0003-q0.0019364916731`: TimeoutError('unchanged 180-second per-step budget exceeded'); completed 73 steps; exact traceback in run `failure.json`.
- `ablation-qso-a0.0002-q0.00122474487139`: TimeoutError('unchanged 180-second per-step budget exceeded'); completed 81 steps; exact traceback in run `failure.json`.
- `ablation-qso-a0.0005-q0.00122474487139`: TimeoutError('unchanged 180-second per-step budget exceeded'); completed 83 steps; exact traceback in run `failure.json`.
- `confirmation-qso-a0.0003-q0.00122474487139`: TimeoutError('unchanged 180-second per-step budget exceeded'); completed 87 steps; exact traceback in run `failure.json`.

Failed-candidate partial numerical evidence (not quality-eligible):

| Candidate | Completed pairs / begun | Newton 0/1/2/3+ | CG mean/median/p95 | Max/p95 gap | Min rcond | Max normalized horizontal | Spectral excess | Completed ADMM / interrupted reference |
|---|---|---|---|---|---:|---:|---:|---|
| coarse-qso-a0.0003-q0.0003 | 552/553 | 0/0/448/104 | 5.32/5/9 | 2.98e-05/2.34e-05 | 0.000322 | 6.89e-17 | 0 | 0 / True |
| coarse-qso-a0.0003-q0.0006 | 534/535 | 0/0/425/109 | 5.38/5/9 | 2.99e-05/2.22e-05 | 0.000331 | 9.09e-17 | 0 | 0 / True |
| coarse-qso-a0.0003-q0.001 | 522/523 | 0/0/394/128 | 5.56/5/9 | 2.99e-05/2.26e-05 | 0.000343 | 9.37e-17 | 0 | 0 / True |
| coarse-qso-a0.0003-q0.0025 | 535/536 | 0/0/320/215 | 6.45/5/10 | 3e-05/2.25e-05 | 0.000129 | 1.62e-16 | 0 | 0 / True |
| coarse-qso-a0.0003-q0.004 | 336/337 | 0/0/267/69 | 5.25/5/9 | 2.91e-05/2.35e-05 | 0.0011 | 2.27e-16 | 0 | 0 / True |
| refinement-qso-a0.0003-q0.0019364916731 | 438/439 | 0/0/336/102 | 5.42/5/9 | 2.99e-05/2.28e-05 | 0.000581 | 1.31e-16 | 0 | 0 / True |
| ablation-qso-a0.0002-q0.00122474487139 | 486/487 | 0/0/342/144 | 5.59/5/9 | 2.96e-05/2.37e-05 | 0.000489 | 9.42e-17 | 0 | 0 / True |
| ablation-qso-a0.0005-q0.00122474487139 | 498/499 | 0/0/403/95 | 5.32/5/9 | 3e-05/2.25e-05 | 0.000385 | 1.09e-16 | 0 | 0 / True |
| confirmation-qso-a0.0003-q0.00122474487139 | 523/524 | 0/0/350/173 | 5.73/5/9 | 2.98e-05/2.24e-05 | 0.000143 | 9.2e-17 | 0 | 0 / True |

The timeout tracebacks enter production `_reference` and CPU ADMM spectral-ball projection. A zero count of completed ADMM returns does not mean reference fallback was never entered. The initiating smooth-failure reason was not exposed before interruption and is not guessed. The paired optimizer commits no partial uncertified step.

## H. Gradient and update scales

| Winner | Gradient RMS first/last | Update RMS first/last | Parameter RMS first/last | Update/parameter first/last/max |
|---|---|---|---|---|
| adamw | 0.0017145/0.00043102 | 3.8862e-06/4.696e-06 | 0.027049/0.027527 | 0.00014367/0.00017059/0.0029117 |
| qso | 0.0017145/0.00043018 | 4.4916e-06/6.6515e-05 | 0.027049/0.027158 | 0.00016605/0.0024492/0.0027775 |

QSO last-step scale is from its last completed step (87), whereas AdamW reaches512. These unequal horizons describe operational trajectories, not a matched final-quality comparison. Completed parameters, gradients and updates remained finite; the observed failures were solver/reference timeouts, not observed NaN divergence.

## I. Cost at equal tokens (not a selection criterion)

| Winner | Total wall s | Step sum s | Warm step median/p95 s | Six-pair median/p95 s | Fwd/back median/p95 s | Tokens/s | Optimizer fraction | Peak GPU MiB |
|---|---:|---:|---|---|---|---:|---:|---:|
| adamw | 15.173 | 13.731 | 0.02678/0.02698 | 0.00000/0.00000 | 0.01122/0.01138 | 76368.2 | 7.49% | 542.7 |
| qso | 279.011 | 98.772 | 1.07959/1.37110 | 1.04001/1.33172 | 0.01144/0.01157 | 1803.9 | 97.79% | 529.5 |

No tuned equal-token cost ratio exists: QSO confirmation terminated early. Its cost row summarizes completed steps only; wall time includes interrupted reference work. Warm metrics exclude steps0–4; cold work stays in totals. Total wall includes validation, checkpoint I/O and independent continuation replay; step sums exclude them. Pair timers synchronize only around production solves; no decomposition/decision policy is changed. GPU peak excludes driver/context and independently replayed model allocations. Candidates shared one allocation; sequential order and device-clock variation limit fine timing interpretations.

Per-run cost summaries (never used for tuning):

| Candidate | Wall s | Step sum s | Warm step median/p95 s | Six-pair median/p95 s | Fwd/back median/p95 s | Tokens/s | Optimizer % | Peak MiB |
|---|---:|---:|---|---|---|---:|---:|---:|
| coarse-adamw-a0.0001-q0.001 | 7.989 | 7.212 | 0.02677/0.02711 | 0.00000/0.00000 | 0.01123/0.01154 | 72699.6 | 7.65 | 539.8 |
| coarse-adamw-a0.0002-q0.001 | 7.298 | 6.871 | 0.02677/0.02702 | 0.00000/0.00000 | 0.01124/0.01142 | 76300.1 | 7.49 | 539.8 |
| coarse-adamw-a0.0003-q0.001 | 7.291 | 6.864 | 0.02678/0.02698 | 0.00000/0.00000 | 0.01124/0.01140 | 76377.6 | 7.50 | 539.8 |
| coarse-adamw-a0.0005-q0.001 | 7.292 | 6.868 | 0.02679/0.02704 | 0.00000/0.00000 | 0.01125/0.01140 | 76333.9 | 7.50 | 539.8 |
| coarse-adamw-a0.0008-q0.001 | 7.312 | 6.908 | 0.02693/0.02734 | 0.00000/0.00000 | 0.01133/0.01155 | 75896.0 | 7.48 | 539.8 |
| coarse-adamw-a0.0012-q0.001 | 7.307 | 6.884 | 0.02684/0.02714 | 0.00000/0.00000 | 0.01127/0.01151 | 76155.9 | 7.49 | 539.8 |
| coarse-qso-a0.0003-q0.0003 FAILED | 280.937 | unavailable (see partial metrics) | — | — | — | — | — | — |
| coarse-qso-a0.0003-q0.0006 FAILED | 277.556 | unavailable (see partial metrics) | — | — | — | — | — | — |
| coarse-qso-a0.0003-q0.001 FAILED | 276.676 | unavailable (see partial metrics) | — | — | — | — | — | — |
| coarse-qso-a0.0003-q0.0015 | 332.168 | 331.668 | 1.37048/1.40842 | 1.32847/1.36779 | 0.01148/0.01167 | 1580.8 | 98.06 | 537.4 |
| coarse-qso-a0.0003-q0.0025 FAILED | 283.779 | unavailable (see partial metrics) | — | — | — | — | — | — |
| coarse-qso-a0.0003-q0.004 FAILED | 241.267 | unavailable (see partial metrics) | — | — | — | — | — | — |
| refinement-adamw-a0.000141421356237-q0.001 | 7.260 | 6.845 | 0.02670/0.02692 | 0.00000/0.00000 | 0.01123/0.01140 | 76599.3 | 7.48 | 545.2 |
| refinement-adamw-a0.000244948974278-q0.001 | 7.265 | 6.851 | 0.02672/0.02687 | 0.00000/0.00000 | 0.01122/0.01132 | 76522.9 | 7.51 | 545.6 |
| refinement-qso-a0.0003-q0.00122474487139 | 329.725 | 329.235 | 1.36597/1.39134 | 1.32424/1.35085 | 0.01148/0.01160 | 1592.4 | 98.05 | 529.3 |
| refinement-qso-a0.0003-q0.0019364916731 FAILED | 260.620 | unavailable (see partial metrics) | — | — | — | — | — | — |
| ablation-qso-a0.0002-q0.00122474487139 FAILED | 271.155 | unavailable (see partial metrics) | — | — | — | — | — | — |
| ablation-qso-a0.0005-q0.00122474487139 FAILED | 270.453 | unavailable (see partial metrics) | — | — | — | — | — | — |
| confirmation-adamw-a0.0002-q0.001 | 15.173 | 13.731 | 0.02678/0.02698 | 0.00000/0.00000 | 0.01122/0.01138 | 76368.2 | 7.49 | 542.7 |
| confirmation-qso-a0.0003-q0.00122474487139 FAILED | 279.011 | unavailable (see partial metrics) | — | — | — | — | — | — |

The two successful tuned 256-step pilots have a measured-step cost ratio of **47.91×** (QSO/AdamW); this is separate from the 512-step principal comparison.

Failed-run wall time includes interrupted reference work. Completed-step throughput from their partial logs must not be substituted for end-to-end throughput. Complete partial-run cost and scale summaries are saved in `/home/prignano/qnormuon-runs/lr-stage-d/study-29024/partial_run_cost_scale.json`.

## J. Checkpoint continuation

- adamw: saved at step256, independently replayed steps257–259; exact minibatch hashes, losses, model and complete optimizer/scheduler state: `{'passed': True, 'exact': True, 'steps': 3, 'checkpoint_step': 256, 'checkpoint_bytes': 137563531, 'qso_state_checked': []}`.
- qso: confirmation terminated before completing the prescribed checkpoint-continuation audit; no Stage-D exact continuation claim.

The QSO audit, when reached, includes fp32 canonical up/down EMA, fp64 original lambda, pair counters and unsupported AdamW state. Historical smoke evidence does not substitute for an unreached Stage-D audit. Checkpoint format was not changed. No coarse/refinement checkpoints were saved.

## K. Limitations and plots

One seed, small local train/validation prefixes, four fixed validation batches and a short horizon do not provide a significance estimate or universal superiority claim. The data representation/tokenizer was reused, not downloaded or semantically re-audited. Personal quota remains unknown; ample filesystem free space is not a user-quota guarantee. Checkpoints are retained only for confirmations that reached step256; here only AdamW did so. This study did not reopen solver-performance research or use additional exploratory training.

Figures are dependency-free SVG vector artifacts with every measured point and unsmoothed connecting lines. All quality figures use tokens on the horizontal axis; cost is reported separately.

- [coarse_adamw](figures/qso_lr_sweep/coarse_adamw.svg)
- [coarse_qso](figures/qso_lr_sweep/coarse_qso.svg)
- [tuned_validation](figures/qso_lr_sweep/tuned_validation.svg)
- [tuned_training](figures/qso_lr_sweep/tuned_training.svg)
- [update_ratio](figures/qso_lr_sweep/update_ratio.svg)
- [gradient_rms](figures/qso_lr_sweep/gradient_rms.svg)
- [qso_newton](figures/qso_lr_sweep/qso_newton.svg)

## L. Answers and next-step decision

1. Best AdamW peak LR: **0.0002**.
2. Best paired QSO peak LR: **0.0012247449**.
3. Best unsupported-parameter AdamW peak LR with QSO: **0.0003**.
4. LR sensitivity is shown by the complete coarse/refinement validation tables and curves; boundary extensions are explicitly included, not silently substituted into the coarse grid.
5. Equal-token final validation difference QSO−AdamW: **unavailable**.
6. Pilot last-three difference: **-0.091204**; confirmation: **unavailable**. A promising pilot without a completed confirmation is inconclusive.
7. Actual update/parameter and gradient scales are tabulated and plotted above; differences reflect method-specific update geometry and learning rates.
8. Tuned QSO confirmation: 522 completed-step certified paired solves, 0 completed CPU ADMM returns; status **failed**. Interrupted reference work is separately documented.
9. Measured training-step cost ratio: **unavailable at equal target tokens**.
10. Separate multi-seed recommendation under the predeclared gate: **no**. No extra seeds were run.

**Final classification: B: tuned QSO and tuned AdamW are inconclusive / mixed on this single-seed study.** Production mathematics/defaults remain frozen. Stop after this report.
