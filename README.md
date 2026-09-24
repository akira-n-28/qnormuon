# QNorMuon — Codex research starter

This repository is a self-contained research starter for **QNorMuon: Gauge-Canonical Spectral Optimization**.

It contains:

- the current mathematical specification;
- a minimal PyTorch reference implementation;
- invariant/property tests;
- gauge and trajectory stress experiments;
- Codex project instructions;
- the first theory-audit task;
- a staged experiment plan.

## Start here

1. Read `AGENTS.md`.
2. Read `docs/QNORMUON_THEORY.md`.
3. Run the baseline tests.
4. Give Codex the task in `docs/CODEX_FIRST_TASK.md`.
5. Do not optimize performance before the theory audit is complete.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell

python -m pip install -U pip
pip install -e ".[dev]"
```

## Verify the baseline

```bash
pytest
python -m experiments.gauge_stress
python -m experiments.toy_trajectory
```

The current reference implementation intentionally uses an **exact thin-SVD polar factor**. It is slow by design: this isolates the mathematical construction from errors introduced by an approximate polar solver.

## Expected baseline behavior

The mathematical tests should pass in float64. The gauge stress test should report very small equivariance/canonical-polar errors. The trajectory experiment should show QNorMuon preserving functional equivalence under a diagonal gauge reset much better than a raw spectral Muon baseline.

Exact numbers depend on PyTorch / BLAS / platform and should not be hard-coded as scientific claims.

## Repository layout

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── docs/
│   ├── QNORMUON_THEORY.md
│   ├── CODEX_FIRST_TASK.md
│   └── EXPERIMENT_PLAN.md
├── qnormuon/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_qnormuon.py
└── experiments/
    ├── gauge_stress.py
    └── toy_trajectory.py
```

## Current scope and caveats

The reference implementation currently focuses on paired SwiGLU matrices

- `up_proj.weight`: `[m, n]`
- `down_proj.weight`: `[n, m]`

with `m >= n`.

Known research items include:

- zero/near-zero neuron norms;
- rank-deficient momentum;
- exact conditions behind leverage-balancing existence/uniqueness;
- approximate polar solvers;
- mixed precision;
- intrinsic/gauge-compatible regularization and weight decay;
- checkpoint/state semantics;
- distributed training;
- full LLM benchmarks.

`gate_proj` is deliberately excluded from the current gauge pair because SiLU is not positively homogeneous.
