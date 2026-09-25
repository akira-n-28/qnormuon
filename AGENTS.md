# AGENTS.md

# QNorMuon / Quotient Spectral Optimizer

This repository is a research project on gauge-invariant spectral optimization
for coupled SwiGLU parameters.

The current core research direction is no longer "apply Muon separately to two
matrices and add corrections".

The accepted provisional core is:

    symmetry
        -> quotient geometry
        -> horizontal tangent space
        -> quotient spectral norm
        -> coupled horizontal spectral LMO
        -> certified numerical solve

The K=I coupled horizontal spectral optimizer is the current production-v0
candidate.

This is a research codebase. Mathematical claims must be treated as claims to
verify, not as axioms to preserve at all costs.


# 1. Read this before making changes

Before modifying optimizer theory or core implementation, read the relevant
documents.

For work on the current optimizer core, read in this order:

1. `docs/QUOTIENT_SPECTRAL_GEOMETRY.md`
2. `docs/HORIZONTAL_SPECTRAL_LMO.md`
3. `docs/DUAL_SOLVER_STUDY.md`
4. `docs/ZERO_STRATUM_GEOMETRY.md`
5. `docs/RANK_DEFICIENT_THEORY.md`
6. `docs/THEORY_AUDIT.md`
7. `docs/QNORMUON_THEORY.md`

Later reports override older statements when the theory has evolved.

`docs/QNORMUON_THEORY.md` is the historical/original specification. Do not
assume every claim in it is still the current design.

Important reference implementations and tests include:

- `experiments/dual_solver.py`
- `experiments/horizontal_spectral.py`
- `experiments/quotient_spectral.py`
- `experiments/rank_deficient.py`
- `experiments/zero_stratum.py`

and the corresponding files under `tests/`.

Research code under `experiments/` is a mathematical/reference implementation,
not automatically production code.


# 2. Current mathematical core

For each coupled SwiGLU up/down pair, use the transposed down layout so that

    U, D in R^{m x n}.

On the regular domain every paired row satisfies

    ||u_i|| > 0
    ||d_i|| > 0.

The positive diagonal gauge is

    U -> C U
    D -> C^{-1} D

for positive diagonal C.

The regular canonical metric is

    H_i = ||u_i|| / ||d_i||.

Balanced canonical weights are

    U_bar = H^{-1/2} U
    D_bar = H^{1/2} D.

Do not introduce an absolute epsilon clamp and then claim that the exact gauge
theorems still hold. An absolute clamp breaks exact covariance near zero.


# 3. Horizontal quotient geometry

In balanced canonical coordinates define

    L(P)_i =
        <U_i, P_U[i]> - <D_i, P_D[i]>.

The horizontal tangent space is

    H_space = ker L.

The current quotient spectral norm is

    ||P||_Q =
        max(
            ||P_U||_2,
            ||P_D||_2
        )

restricted to the horizontal tangent space.

Here `||.||_2` is the matrix operator/spectral norm.

The corresponding unit ball is

    ||P_U||_2 <= 1
    ||P_D||_2 <= 1
    L(P) = 0.

This is the feasible set of the current coupled spectral LMO.

Do not replace this norm with an infimum-over-vertical-lifts quotient norm.
Those are different constructions.


# 4. Cotangent classes and dual norm

Objective matrices are defined modulo vertical covectors:

    A ~ A + L* lambda

where

    L* lambda =
        (
            diag(lambda) U,
           -diag(lambda) D
        ).

The dual norm is

    ||[A]||_{Q,*}
      =
      min_lambda
        ||A_U - diag(lambda) U||_*
        +
        ||A_D + diag(lambda) D||_*.

The sign between the two nuclear norms is PLUS.

A difference of nuclear norms is not the accepted convex dual problem.

The coupled horizontal LMO is

    maximize
        <A_U, P_U> + <A_D, P_D>

    subject to
        ||P_U||_2 <= 1
        ||P_D||_2 <= 1
        L(P) = 0.

This is exact unit-ball steepest descent for the represented cotangent when A
is the current loss differential.

If A is EMA momentum, it is exact steepest descent for the momentum linear
surrogate, not necessarily for the instantaneous loss.


# 5. Dual residuals

For multiplier lambda define

    B_U =
        A_U - diag(lambda) U

    B_D =
        A_D + diag(lambda) D.

If both residuals have full column rank, the primal direction is

    P_U = polar(B_U)
    P_D = polar(B_D)

at dual stationarity.

The dual gradient is

    grad_i phi =
        - <U_i, P_U[i]>
        + <D_i, P_D[i]>.

Therefore dual stationarity is exactly the horizontal condition.

Repeated positive singular values are not themselves a problem.

Small singular values make the polar derivative ill-conditioned and must be
treated as a numerical conditioning issue.


# 6. Rank-deficient residuals

Do not assume that separate partial polars solve the coupled problem.

At deficient residual rank, primal recovery is a JOINT subgradient feasibility
problem.

Separate minimum-norm partial polars can fail the horizontal constraint.

The accepted canonical tie-break is:

    among all optimal horizontal primal solutions,
    minimize

        ||P_U||_F^2 + ||P_D||_F^2.

This minimum-Frobenius optimal-face selection is unique.

Important consequences:

- the selected coupled solution need not be a partial isometry;
- free singular values may lie anywhere in [0, 1];
- `P^T P = I` is NOT a universal invariant;
- projector Gram matrices are NOT universal;
- integer leverage mass is NOT universal;
- a zero objective selects exactly zero update.

Do not reintroduce full-Stiefel completion at zero momentum.


# 7. Zero weight rows

Momentum rank deficiency and zero WEIGHT rows are different problems.

The regular metric requires

    ||u_i|| > 0
    ||d_i|| > 0.

At exact zero weight rows the regular quotient geometry is singular.

For production-v0:

- do not invent an epsilon-based "exact extension";
- do not silently create a neuron-birth rule;
- use an explicit policy such as error or skip-pair;
- report the event clearly.

A one-sided zero row also has no finite balanced representative.

Neuron birth is a separate matrix-space transition problem and is not part of
production-v0.


# 8. Gauge equivariance

On the regular domain, with matching canonical optimizer state, the desired raw
update law is

    Delta U' = C Delta U
    Delta D' = C^{-1} Delta D.

The canonical problem must therefore be identical after positive gauge
transformation.

All claimed gauge invariants must have automated tests.

Do not claim exact gauge invariance for:

- clamped metrics;
- undefined zero rows;
- arbitrary mixed-precision code;
- numerically inconsistent rank decisions.

Finite-precision error must be measured, not hidden.


# 9. Signed gauges

Positive gauge is the default production convention.

If signed gauges are investigated, canonical matrix-valued first moments must
transform with the row-sign matrix.

Do not apply a sign reset while keeping populated canonical momentum unchanged.

Signed gauge support is research-only unless explicitly requested by the task.


# 10. Production-v0 scope

The first production candidate is intentionally conservative.

IN SCOPE:

- K = I;
- regular nonzero paired rows;
- canonical paired momentum;
- coupled horizontal spectral LMO;
- warm-started dual solve;
- row-whitened Newton-CG;
- certified primal recovery;
- explicit fallback;
- PyTorch optimizer integration;
- checkpointing;
- tiny Transformer experiments.

OUT OF SCOPE unless explicitly requested:

- shared K;
- leverage balancing;
- objective-contribution balancing;
- neuron birth;
- distributed tensor/model parallelism;
- custom CUDA polar kernels;
- approximate Newton-Schulz polar;
- new optimizer theory;
- automatic singular-stratum regularization.

Do not add one of these because it merely seems useful during production-v0.


# 11. Current numerical solver candidate

The leading smooth solver is:

    warm-started
    row-whitened
    Newton-CG

for

    phi(lambda)
      =
      ||B_U||_*
      +
      ||B_D||_*.

Use the previous step's multiplier lambda as warm state.

The previous lambda is stored in the ORIGINAL multiplier convention. When
weights and normalization change, transform it into the new internal whitened
coordinates.

Do not carry the previous whitened internal variable blindly between steps.


# 12. Row whitening

Define

    w_i =
        ||U_i||^2 + ||D_i||^2.

Use

    W = diag(w_i).

Whitening by W^{-1/2} has a mathematical interpretation as constraint-Jacobian
whitening.

It removes row-amplitude factors from an upper curvature bound.

It is NOT a theorem that it always improves the actual Hessian condition
number.

Do not describe it as universally optimal preconditioning.


# 13. Newton-CG

Use matrix-free Hessian-vector products.

Do not form an m x m dense Hessian for production-sized matrices.

Reuse cached SVD/polar factors whenever mathematically valid.

The research study found that two warm Newton iterations were enough for the
tested slow trajectories at the chosen tolerance, but:

    TWO ITERATIONS ARE NOT A UNIVERSAL GUARANTEE.

Use two iterations only as an initial budget.

Acceptance must depend on certificates and conditioning.

Continue solving or fall back if the certificate fails.


# 14. Required solver diagnostics

Every production solve should expose at least:

- primal objective;
- dual objective;
- absolute primal-dual gap;
- normalized primal-dual gap;
- horizontal residual;
- normalized horizontal residual;
- spectral norm of each primal side;
- spectral excess;
- residual conditioning / rcond;
- number of Newton iterations;
- number of CG/HVP iterations;
- number of SVD/polar evaluations if available;
- whether fallback occurred;
- fallback reason;
- solver dtype;
- returned-direction dtype.

Do not silently turn an unsuccessful solve into success.


# 15. Gap versus direction accuracy

A small primal-dual value gap does NOT universally imply an accurate direction.

Near rank loss it is possible to have an extremely small value gap and O(1)
direction error.

Therefore:

- always monitor residual conditioning;
- do not use gap alone near singular residuals;
- do not silently truncate small positive singular values;
- do not claim exact direction recovery from a value certificate alone.

In a well-conditioned full-rank regime a gap can provide a direction bound, but
the relevant minimum singular value must be controlled.


# 16. Fallback

Keep the generic high-accuracy reference solver available during production-v0
development.

Fallback must be explicit and observable.

Typical reasons include:

- residual conditioning below the configured research guard;
- nonfinite values;
- failed line search;
- failure to obtain a descent direction;
- stagnating certified gap;
- exhausted smooth-solver budget;
- cancellation-dominated intrinsic objective.

A fallback value solve does not automatically guarantee the unique
minimum-Frobenius direction on a nonunique deficient optimal face.

Keep the distinction between:

- primary-value certificate;
- primal direction;
- canonical minimum-Frobenius selection.


# 17. Shared K and balancing

Shared row weighting K can mathematically compose with the coupled LMO, but the
old QNorMuon / NorMuon-style logdet leverage identity DOES NOT survive.

Do not reuse

    leverage - target

as the gradient of the old logdet objective after replacing separate polars by
the coupled horizontal solve.

The currently derived alternative contribution objective is research-only.

Do not integrate shared K or contribution balancing into production-v0 unless a
task explicitly requests a separate research study.


# 18. PyTorch API expectations

Production code should live under `qnormuon/`.

A production optimizer should be compatible with `torch.optim.Optimizer` where
practical.

Paired parameters must be registered explicitly.

Do not infer pair identity solely from arbitrary parameter iteration order.

The intended supported pair is:

    up_proj.weight
    down_proj.weight

with correct transpose handling for the down projection.

Unsupported parameters should be handled by a conventional optimizer such as
AdamW, including typically:

- gate projection;
- embeddings;
- normalization parameters;
- biases;
- output heads;
- other unsupported matrices.

Provide a clear mechanism to use:

    Quotient Spectral Optimizer on supported pairs
    AdamW on all remaining parameters.


# 19. Optimizer state

At minimum, paired optimizer state may include:

- canonical up momentum;
- canonical down momentum;
- previous dual multiplier lambda;
- step count;
- solver metadata if needed.

State must survive

    state_dict()
    load_state_dict()

without changing the subsequent trajectory beyond expected floating-point
rounding.

Pair topology must not depend silently on dictionary or parameter ordering.

Checkpoint tests are required.


# 20. Theory/code consistency rule

If implementation contradicts the current theory:

DO NOT silently patch around it.

Instead:

1. identify the exact theorem/equation/assumption;
2. construct a counterexample or failing test if possible;
3. determine whether:
   - the code is wrong,
   - the theorem is wrong,
   - an assumption is missing,
   - the production approximation intentionally changes the mathematical problem;
4. document the distinction;
5. update theory and code together only after the design decision is explicit.

Every claimed invariant should have an automated test.


# 21. Testing requirements

Before accepting a core optimizer change, run the full existing mathematical
test suite.

Do not delete or weaken an adversarial test merely to make a redesign pass.

Add tests for every new production invariant.

Important production tests include:

- positive diagonal gauge stress;
- actual SwiGLU functional equivalence;
- canonical gradient/state invariance;
- horizontal residual;
- spectral feasibility;
- primal-dual gap;
- zero cotangent -> zero update;
- full-rank smooth solve vs reference;
- deficient-rank behavior;
- checkpoint save/load;
- mid-training positive gauge reset;
- float64 reference behavior;
- float32 numerical behavior;
- final-direction casting effects;
- finite-step delta-X comparisons.

Use float64 for mathematical/reference tests unless the test is explicitly about
reduced precision.


# 22. Precision

Do not assume native bfloat16 SVD support on every backend.

For H100 work, validate actual CUDA/H100 behavior rather than extrapolating from
CPU results.

Keep distinctions clear between:

- storage dtype;
- momentum dtype;
- solver dtype;
- SVD/polar dtype;
- certification dtype;
- returned update dtype.

Do not claim float64-level invariance after casting a direction to bfloat16.

Measure it.


# 23. Experimental methodology

When comparing optimizers, do not force the same learning rate if the methods
have different natural scales.

Perform optimizer-specific learning-rate sweeps.

Comparisons should use identical:

- model initialization;
- architecture;
- data order;
- tokenizer/data representation;
- number of tokens;
- schedule;
- seed;
- evaluation procedure.

After pilot hyperparameter selection, use multiple seeds.

Report both quality and cost.


# 24. Required training metrics

For meaningful optimizer experiments record at least:

- training loss vs step;
- training loss vs tokens;
- validation loss;
- training loss vs wall time;
- tokens/sec;
- optimizer wall-time fraction;
- gradient RMS;
- update RMS;
- Newton iterations;
- CG/HVP count if practical;
- fallback frequency;
- primal-dual gap;
- horizontal residual;
- residual rcond;
- GPU memory usage.

Gauge-reset experiments should also compare the subsequent functional trajectory.


# 25. Baselines

Relevant baselines may include:

- AdamW;
- Muon + AdamW fallback for unsupported parameters;
- historical separate-polar QNorMuon;
- coupled K=I Quotient Spectral Optimizer.

Do not label a local diagnostic implementation as a faithful baseline for an
external published optimizer unless that equivalence has been verified.


# 26. Claims

Do not claim that the optimizer is better than Muon, NorMuon, AdamW or another
method without experimental evidence.

Separate clearly:

- theorem;
- numerical verification;
- toy experiment;
- tiny-model result;
- full training result;
- conjecture.

Avoid turning observations from a few random seeds into universal statements.


# 27. Codex execution environment

Codex operates directly on a Linux machine through SSH.

The project files are already present in the current workspace.

Do not assume any external source-control service, synchronization service,
cloud filesystem, or external development machine.

Codex should work only with the files available in the current Linux workspace
and with explicitly configured cluster storage paths.

Do not attempt to solve missing-project-file problems by fetching code or
artifacts from external services.

Small source files, configuration files, reports and logs may be modified
directly in the project workspace.

Large datasets, checkpoints, caches and experiment artifacts must use explicitly
configured cluster-visible storage locations.


# 28. Lagrange cluster

The primary GPU environment is the Lagrange cluster at the Dipartimento di
Matematica Guido Castelnuovo, Sapienza Università di Roma.

The public cluster frontend is:

    lagrangectl.mat.uniroma1.it

The frontend is used to inspect the environment and submit work through SLURM.

Compute-heavy work must run inside an appropriate SLURM allocation on the
compute server.

For this project, default to ONE NVIDIA H100 per job unless the actual cluster
configuration and allocation policy have explicitly been verified otherwise.


# 29. Critical login-node constraint

The frontend/login environment has very limited usable memory for this workflow,
approximately 3.8 GB.

Treat the login node as a LIGHTWEIGHT CONTROL PLANE ONLY.

Allowed operations include:

- editing small source and configuration files;
- reading small logs;
- inspecting directories and file metadata;
- checking environment variables;
- `sinfo`;
- `squeue`;
- `scancel`;
- submitting jobs with `sbatch`;
- starting appropriate allocations with `srun`;
- lightweight Python or shell environment inspection.

DO NOT run directly on the login node:

- model training;
- optimizer benchmarks;
- dataset preprocessing;
- dataset tokenization;
- large tensor creation;
- large archive extraction;
- large checkpoint loading;
- memory-heavy Python programs;
- heavy compilation;
- large numerical experiments;
- substantial model initialization.

If an operation may consume significant RAM, CPU time, GPU resources or disk
bandwidth, run it inside a SLURM compute allocation instead.


# 30. No large downloads on the login node

Do not download or materialize large assets on the login node.

This includes:

- datasets;
- tokenizer corpora;
- pretrained model weights;
- checkpoints;
- model snapshots;
- large archives;
- large package caches;
- container images;
- large experiment artifacts.

Training and preprocessing code must not silently download missing assets.

Avoid runtime behavior that automatically fetches large remote files.

All large assets required for an experiment must already be present on a
cluster-visible filesystem before the compute job starts.

If an expected asset is absent:

    FAIL CLEARLY.

Do not silently fetch it from the Internet.

Before launching a job, validate required paths explicitly.


# 31. Cluster storage

Do not assume undocumented paths such as:

    /scratch
    /data
    /datasets
    /checkpoints

until they have been verified on the actual Linux cluster environment.

Use explicit configuration variables or command-line arguments, for example:

    QSO_DATA_ROOT
    QSO_OUTPUT_ROOT
    QSO_CACHE_ROOT
    QSO_CHECKPOINT_ROOT

Before large experiments, inspect the available filesystems with lightweight
commands such as:

    df -h
    mount
    pwd

and use the appropriate cluster-visible storage location.

Do not store large datasets, model weights, repeated checkpoints or large caches
inside the project source directory.


# 32. Offline experiment policy

Once the environment and data are prepared, training jobs should be able to run
without external network access.

Every experiment should receive explicit local paths for:

- dataset;
- tokenizer;
- checkpoints if required;
- cache directory;
- output directory.

Network access must not be a hidden requirement of the training loop.


# 33. SLURM execution

All compute-heavy work must be submitted with `sbatch` or run inside an
appropriate `srun` allocation.

Do not first execute a heavy command on the frontend and only later move it to
SLURM.

Request only the resources required by the experiment.

Before fixing CPU, RAM, GPU or time limits in scripts, inspect the actual
cluster configuration and current allocation policy.

Do not assume that all GPUs physically installed in the server are available to
one job.


# 34. Experiment provenance

At the beginning of every important compute job, record at least:

    hostname
    date
    SLURM_JOB_ID
    SLURM_JOB_NODELIST
    CUDA_VISIBLE_DEVICES
    nvidia-smi
    Python version
    PyTorch version
    torch.version.cuda
    GPU model
    allocated CPU count
    allocated memory if available
    project working directory
    dataset path
    tokenizer path
    checkpoint/input path if applicable
    output path
    random seed
    model configuration
    optimizer configuration
    solver configuration

This information should be written to the job log or structured experiment
metadata.


# 35. Cluster-safe experiment design

Dataset preprocessing that needs substantial memory or CPU must itself run in a
compute allocation.

Large experiment outputs should not be accumulated blindly.

Checkpoint cadence should be configurable.

Avoid producing many redundant multi-GB checkpoints during hyperparameter
sweeps.

Prefer compact metric logs plus selected checkpoints.

Failed jobs should leave enough metadata to diagnose the failure without
requiring large debug dumps.


# 36. Tiny Transformer phase

Before attempting a large language-model training run, validate the optimizer on
a small decoder-only SwiGLU Transformer.

The purpose is to establish:

1. numerical stability;
2. realistic Newton iteration counts;
3. fallback frequency;
4. optimizer overhead;
5. gauge-reset behavior;
6. whether optimization quality is competitive enough to justify scaling.

Do not jump directly from mathematical unit tests to an expensive large-model
run.


# 37. Documentation rule

A research result that changes the accepted mathematical contract should be
documented before or together with production integration.

A benchmark result belongs in a dedicated report, not buried in source-code
comments.

Keep `AGENTS.md` focused on rules and current contracts.

Do not copy entire mathematical derivations into this file; point to the
authoritative report instead.


# 38. Default decision rule

When uncertain:

1. preserve the current mathematical contract;
2. construct a test;
3. prefer an explicit failure over a silent heuristic;
4. keep research prototypes isolated from production;
5. measure numerical error;
6. report assumptions;
7. do not hide fallbacks;
8. do not download large assets on the Lagrange login node.
