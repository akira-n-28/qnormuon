# Lagrange H100 mathematical/reference suite validation

**Result (2026-09-28): 150 collected, 150 passed, 0 failed, 0 skipped.**

The current mathematical/reference suite is validated on the Lagrange H100 environment.

This validates the current research suite in the selected environment and
clears the environment for **production-v0 development**. It does not validate
an as-yet-unimplemented production optimizer or establish accuracy for every
GPU dtype and conditioning regime.

## Environment and provenance

The validated base environment is
`/home/prignano/modded-nanogpt/.venv/bin/python`, **Python 3.10.20**.
Only pytest-related packages were placed in the project-local target
`.qso-tools/pytest-site`; the base environment was not used as an installation
target. Its `pyvenv.cfg` and `torch/version.py` retained their pre-job size and
modification timestamps. The previously verified wheelhouse was used offline
with `--no-index`, `--find-links`, `--no-deps`, and `--target`.

| Component | Imported version | Source |
| --- | --- | --- |
| torch | **2.10.0+cu128**; `torch.version.cuda` **12.8** | Validated base environment |
| NumPy | **2.2.6** | Validated base environment |
| pytest | **9.0.3** | Project-local target |
| pluggy | **1.6.0** | Project-local target |
| iniconfig | **2.3.0** | Project-local target |

Pytest's other relevant dependencies were already in the base environment:
packaging 26.0, Pygments 2.19.2, exceptiongroup 1.3.1, and tomli 2.4.0.
The target was 1.4 MiB after installation. The earlier unexecuted environment
plan proposed torch 2.10.0+cu126 and pytest 9.0.2; this measured validation
uses the existing cu128 torch build and the subsequently prepared pytest 9.0.3
wheelhouse. Those are explicit environment choices for this result.

- Exact job script: [run_math_tests_h100.sbatch](../cluster/run_math_tests_h100.sbatch).
- Raw combined SLURM output: [run_math_tests_h100-28891.log](../cluster/run_math_tests_h100-28891.log).
- Python socket guard: [sitecustomize.py](../.qso-tools/network-guard/sitecustomize.py).
- Script SHA-256: `00697389f7d416f36f0763b0a8c4da098261c1f7d12fbfc0a3101b981ce893b4`.
- Guard SHA-256: `9d4edfad4db826e93ebfcae1462924e105d5e6c892f34713224610c8584e1bb9`.
- Submission: `sbatch --parsable cluster/run_math_tests_h100.sbatch`;
  job **28891**, partition `mat`, node `lagrange0`.
- Requested: **1 H100 GPU, 4 CPUs, 16 GiB RAM, 30-minute limit**.
  SLURM accounting: `COMPLETED`, exit `0:0`, allocated 4 CPUs and 1 GPU,
  elapsed **35 seconds**. `MaxRSS` was not reported.
- Log start/end: `2026-09-28T13:38:10+02:00` /
  `2026-09-28T13:38:44+02:00`.
- `CUDA_VISIBLE_DEVICES=0`; PyTorch identified **NVIDIA H100 NVL**,
  compute capability **9.0**, with CUDA available. `nvidia-smi` reported
  driver **570.211.01** and 95,830 MiB GPU memory.

The job recorded hostname, date, SLURM identifiers, GPU inventory, exact
interpreter and package versions, working directory, and `PYTHONPATH` before
the suite. Dataset, tokenizer, model, checkpoint, seed, and training optimizer
settings were recorded as not applicable; the suite's solver settings are
defined by individual tests.

## Complete suite result

The job first ran collection, then ran the unfiltered
`/home/prignano/modded-nanogpt/.venv/bin/python -m pytest -q` inside the
allocation. Pytest's repository configuration also requests quiet output,
so its final success summary was suppressed. The collection counts and all
150 progress markers in the raw log establish the totals below.

| Test module | Collected |
| --- | ---: |
| `test_dual_solver.py` | 25 |
| `test_horizontal_spectral.py` | 28 |
| `test_qnormuon.py` | 5 |
| `test_rank_deficient.py` | 30 |
| `test_theory_audit.py` | 29 |
| `test_zero_stratum.py` | 33 |
| **Total** | **150** |

Collection exit code was **0**. The full run produced **150 pass markers**, no
failure or skip markers, and exit code **0**. Collection took **8 seconds**;
the full pytest command took **16 seconds** by the job's wall clock. No tests
were skipped, filtered, or given looser tolerances. There are no failure
tracebacks to classify in this run.

The suite is mainly the existing mathematical/reference tests and follows
their own device choices. The dedicated check below exercised the H100 CUDA
spectral path explicitly. Passing the suite in a GPU allocation should not be
misread as every reference test running its math on the GPU.

## Dedicated H100 spectral check

A deterministic, full-column-rank 16 × 8 matrix used the reference
`torch.linalg.svd(x, full_matrices=False)` followed by `q @ vt` polar path.
No test tolerance was changed.

| CUDA input dtype | Observation |
| --- | --- |
| float64 | Passed; relative reconstruction error **1.91618504e-15**, maximum polar Gram error **5.55111512e-15** |
| float32 | Passed; relative reconstruction error **1.24045133e-06**, maximum polar Gram error **1.93782005e-06** |
| bfloat16 | Expected `NotImplementedError`: `"svd_cuda_gesvdjBatched" not implemented for 'BFloat16'` |

The bfloat16 result confirms that a future production solver receiving bf16
model tensors must promote its spectral/SVD work to **at least float32** before
calling this path. The selected SVD dtype, certification dtype, and returned
direction dtype still need separate implementation decisions and tests.

## Network and failure checks

A pre-run static scan of `tests/`, `experiments/`, and `qnormuon/` found no
references to URL/download APIs, `wget`, `curl`, or subprocess calls. The job
set offline variables for pip, uv, Hugging Face, Transformers, and W&B;
disabled third-party pytest plugin auto-loading; and loaded a socket guard
through `PYTHONPATH`. The guard fails and records Python external DNS or
connection attempts. The job reported
`NETWORK_AUDIT=NO_PYTHON_OUTBOUND_ATTEMPT_OBSERVED`; no attempt log was
created. This is evidence for the executed Python path, not a system-wide
network isolation guarantee for native libraries or external processes.

The job made no package installation or download attempt. Only the project
local pytest target and small test logs/cache are associated with this task.
No optimizer implementation was changed. The numerical checks here establish
environment suitability for development; production-v0 correctness still
requires its own implementation and invariant tests.
