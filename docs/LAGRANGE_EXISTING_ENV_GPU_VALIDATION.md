# H100 validation of existing Python environments

Validated 2026-09-25 on Lagrange. This is a small environment and numerical
smoke test, not the QNorMuon mathematical suite or optimizer validation.
`AGENTS.md`, `LAGRANGE_ENVIRONMENT.md`, `LAGRANGE_EXISTING_ENVS.md`, and
`LAGRANGE_PYTHON_ENV.md` were read completely before submission.

## Outcome

**Both existing environments can run the tested QNorMuon SVD/polar path on the
H100 in float64 and float32.** Both import NumPy 2.2.6 and PyTorch
2.10.0+cu128, see the allocated H100, allocate a CUDA tensor, and perform
matrix multiplication. Native CUDA bfloat16 SVD fails in both; the solver must
promote before SVD if its inputs are bfloat16. Neither environment has pytest.

There is **no measured technical reason to prefer one environment over the
other**: the versions, CUDA results, dependency inventory relevant to pytest,
and recorded numerical results were identical. Either can be used for a later
read-only QNorMuon GPU check. For ongoing project work, using an environment
owned by a different project has dependency-change risk; this task did not
establish its maintenance policy or modify it.

**A fresh cu126 PyTorch installation is not technically necessary for the
H100 operations tested here.** It remains necessary only if the project chooses
to enforce the earlier *exact* cu126 pin or later work exposes a cu128-specific
problem. The existing cu128 build should be documented as an explicit change
to the environment plan before treating it as the project's fixed baseline.
pytest and its missing dependencies still need a separate authorized offline
provisioning step before running the suite.

## Job provenance and isolation

- Submitted script: [validate_existing_envs_h100.sbatch](../cluster/validate_existing_envs_h100.sbatch).
- Raw output: [validate_existing_envs_h100-22045.log](../cluster/validate_existing_envs_h100-22045.log).
- Script SHA-256: `22e824506f9372f414df22d74d2582ff961a78a5c96ec23a7d56961cc7274d24`.
- Submission: `sbatch --parsable cluster/validate_existing_envs_h100.sbatch`,
  from `/home/prignano/qnormuon`; job ID **22045**.
- Partition `mat`, compute node `lagrange0`; 1 GPU, 2 CPUs, 12 GiB requested
  memory, eight-minute limit. SLURM accounting: `COMPLETED`, exit `0:0`,
  elapsed **19 seconds**. `MaxRSS` was blank in accounting.
- Log interval: `2026-09-25T18:12:46+02:00` to
  `2026-09-25T18:13:05+02:00`.
- `CUDA_VISIBLE_DEVICES=0`; `SLURM_MEM_PER_NODE=12288` MiB.
  `nvidia-smi` reported **NVIDIA H100 NVL**, **95,830 MiB**,
  driver **570.211.01**. PyTorch reported compute capability **9.0** and
  device count **1**.
- The job forced `PATH=/usr/local/bin:/usr/bin:/bin`, unset
  `LD_LIBRARY_PATH`, `CUDA_HOME`, `CUDA_PATH`, `PYTHONPATH`, and venv/conda
  activation variables. Thus no CUDA toolkit directory or system CUDA
  library directory was added for these tests.
- Both Python processes used `-I -B`; bytecode writes and the CUDA kernel
  cache were disabled. No compilation command, package manager, installer,
  or network access was used. The environments were not changed. The job
  created only its small log in the project workspace.

The earlier CUDA 12.6 toolkit detection concerns a separate system toolkit.
This test exercised the installed cu128 PyTorch wheels and NVIDIA driver with
the sanitized environment above. Success does not require matching the local
toolkit version to `torch.version.cuda` for these prebuilt operations.

## Runtime versions and GPU checks

| Check | `/home/prignano/modded-nanogpt/.venv` | `/home/prignano/pg` |
| --- | --- | --- |
| Python executable | `/home/prignano/modded-nanogpt/.venv/bin/python` | `/home/prignano/pg/bin/python` |
| Python version | 3.10.20 | 3.10.20 |
| `torch.__version__` | 2.10.0+cu128 | 2.10.0+cu128 |
| `torch.version.cuda` | 12.8 | 12.8 |
| `numpy.__version__` | 2.2.6 | 2.2.6 |
| `torch.cuda.is_available()` | `True` | `True` |
| `torch.cuda.get_device_name(0)` | NVIDIA H100 NVL | NVIDIA H100 NVL |
| `torch.cuda.get_device_properties(0)` | H100 NVL, major 9, minor 0, 95,329 MB reported by PyTorch, 132 multiprocessors | Same |
| CUDA tensor allocation/value check | Pass, `cuda:0`, 8 elements | Pass |
| CUDA float32 matrix multiply | Pass, identity product error 0 | Pass |

The `nvidia-smi` and PyTorch memory totals use different reported units/rounding;
they both identify the allocated H100. Full device properties, including UUID,
are preserved in the raw log.

## SVD and polar measurements

The same deterministic, full-column-rank **16 × 8** matrix was used in each
environment. The reference path was exactly
`q, sigma, vt = torch.linalg.svd(b, full_matrices=False)` followed by
`p = q @ vt`, as in `experiments/dual_solver.py:SmoothDual.evaluate`.
The matrix's CPU float64 singular values ranged from **1.0143810513** to
**2.2031439512**. It was deliberately well-conditioned; these results do not
test near-singular behavior or the coupled LMO.

All numerical values below were **identical in the two environments**.
Relative errors use Frobenius/Euclidean norms. Reconstruction compares
`U diag(S) Vh` to the input in its original selected dtype, with comparison
arithmetic in float64. Singular-value and polar errors compare GPU output to
the CPU float64 result; comparing the polar factor avoids nonunique signs of
individual SVD vectors.

| Path | Result | Relative reconstruction error | Relative singular-value error vs CPU float64 | Relative polar error vs CPU float64 | Relative polar orthogonality error |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU float64 SVD/polar | Pass | 1.1242e-15 | Reference | Reference | 9.9418e-16 |
| GPU float64 SVD/polar | Pass | 1.9162e-15 | 1.1336e-15 | 1.5926e-15 | 2.4467e-15 |
| GPU float32 SVD/polar | Pass | 1.2405e-06 | 5.5122e-07 | 5.8799e-07 | 1.1624e-06 |
| GPU bfloat16 SVD/polar | **Fail** | — | — | — | — |

The exact bfloat16 failure for **each** environment was:

```text
NotImplementedError: "svd_cuda_gesvdjBatched" not implemented for 'BFloat16'
```

The job's `CANDIDATE_RESULT: PASS` means all *required* checks passed; it
records bfloat16 SVD separately as a failed, nonrequired capability. Do not
interpret that summary line as bfloat16 SVD support. In a future optimizer,
storage/momentum dtype must be distinguished from the SVD, solver,
certification, and returned-direction dtypes. The float32 result here is a
small smoke measurement, not a bound for production-sized or ill-conditioned
matrices.

## pytest dependency metadata

Installed distribution metadata was checked without importing pytest or
installing packages. The results match in both environments:

| Distribution | Status in both environments |
| --- | --- |
| pytest | Absent |
| pluggy | Absent |
| packaging | Present, 26.0 |
| iniconfig | Absent |
| pygments | Present, 2.19.2 |

These are examples of pytest's dependency ecosystem, not a resolved lock for
pytest 9.0.2. A later offline package plan must use that version's actual
metadata. The full mathematical suite was not run.

## Limits and next decision

The tested matrix was small and well-conditioned. The job did not test
autograd, Newton-CG, rank-deficient recovery, the coupled horizontal
constraint, gauge invariance, checkpointing, or production optimizer behavior.
No assertion is made that cu128 and cu126 produce identical trajectories.

The present evidence supports reusing either environment for the next GPU
research step, provided the project accepts cu128 and keeps the other
project's environment unchanged. The exact fresh cu126 wheelhouse is no longer
needed merely to make these H100 CUDA/SVD operations work. It remains relevant
for an exact cu126 baseline or an isolated project environment. pytest is still
missing, so full-suite readiness remains unresolved.
