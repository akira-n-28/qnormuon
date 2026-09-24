# QNorMuon experiment plan

## Stage 0 — Mathematical correctness

Do not benchmark language-model loss until the invariant tests are trustworthy.

Required checks:

- functional gauge invariance;
- canonical-representative invariance;
- canonical-gradient invariance;
- lifted-update gauge equivariance;
- generalized Stiefel constraint;
- paired-leverage convergence;
- analytic/autograd/finite-difference agreement for the balancing objective.

## Stage 1 — Numerical robustness

Measure errors for:

- float64;
- float32;
- bfloat16 where supported;
- gauge scales from roughly `1e-6` to `1e6`;
- full-rank, nearly rank-deficient and rank-deficient momentum matrices;
- zero and near-zero neuron norms.

Keep exact thin-SVD polar as the reference.

## Stage 2 — Fast polar backend

Add a separate approximate backend instead of replacing the reference implementation.
Compare:

- `||P_fast - P_svd||`;
- `||P_fast^T P_fast - I||`;
- gauge-equivariance error;
- wall-clock time and memory.

## Stage 3 — Optimizer API and real SwiGLU block

- clean `torch.optim.Optimizer` integration;
- explicit pair registration for `up_proj.weight` / `down_proj.weight`;
- separate optimizer policy for embeddings, norms, biases, gate projection and head;
- checkpoint/state-dict tests;
- multiple layers / multiple parameter pairs.

## Stage 4 — Intrinsic regularization

Derive and test weight decay / regularization on the quotient rather than assuming ordinary independent Euclidean weight decay is appropriate.

## Stage 5 — Controlled training experiments

Ablations:

1. AdamW;
2. Muon;
3. NorMuon;
4. QMuon with `K=I`;
5. QNorMuon with shared paired-leverage state;
6. QNorMuon with separate leverage states as a negative/diagnostic control;
7. shared leverage state without canonical gauge handling.

Track at least:

- training/validation loss vs tokens;
- wall-clock throughput;
- polar defect;
- paired-leverage CV;
- activation RMS distribution;
- update norms;
- gauge stress trajectory divergence.
