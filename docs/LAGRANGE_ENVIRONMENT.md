# Lagrange compute-node environment validation

Validated on **2026-09-25**, following a complete reread of `AGENTS.md`.
This report describes a single lightweight allocation, not a numerical or
performance validation. No optimizer code was changed, packages installed,
environments created, assets downloaded, or mathematical tests run.

## Evidence and reproduction

- Exact submitted script: [validate_lagrange.sbatch](../cluster/validate_lagrange.sbatch).
- Raw combined job output: [validate_lagrange-22022.log](../cluster/validate_lagrange-22022.log).
- Frontend-created marker: [frontend_visibility_probe.txt](../cluster/frontend_visibility_probe.txt).
- Submission command, run from `/home/prignano/qnormuon`:
  `sbatch --parsable cluster/validate_lagrange.sbatch`.
- Script SHA-256:
  `4f0f74a003d957035abd25ed78a50dcfffe88e6a8ac251c7c0f38517d24b1f18`.

The script records individual command exit statuses and tolerates absent optional
tools so that all checks can finish. Job completion does not mean every optional
tool exists. Python discovery is bounded to PATH and common local environment
locations; it is not an exhaustive search of cluster storage.

## Verified facts

### Allocation and identity

| Item | Observed value |
|---|---|
| Submission frontend | `lagrangectl` |
| Compute hostname | `lagrange0` |
| Job ID / node list | `22022` / `lagrange0` |
| Partition | `mat` |
| Account / QoS | `castelnuovo` / `castelnuovo_qos` |
| First recorded date | `2026-09-25T16:44:58+02:00` |
| Last recorded date | `2026-09-25T16:45:00+02:00` |
| Requested resources | 1 node, 1 task, 1 CPU per task, 1 GPU, 2 GiB RAM |
| Time limit | 5 minutes |
| Accounting outcome | `COMPLETED`, exit `0:0`, elapsed `00:00:03` |
| Allocated resources | 2 logical CPUs, 1 GPU, 2 GiB RAM |
| Working directory | `/home/prignano/qnormuon` |

Post-completion accounting was queried with:

```text
sacct -j 22022 --format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,AllocTRES,MaxRSS -P
JobID|State|ExitCode|Elapsed|AllocCPUS|ReqMem|AllocTRES|MaxRSS
22022|COMPLETED|0:0|00:00:03|2|2G|billing=2,cpu=2,gres/gpu=1,mem=2G,node=1|
22022.batch|COMPLETED|0:0|00:00:03|2||cpu=2,gres/gpu=1,mem=2G,node=1|
```

### GPU and CUDA

- `CUDA_VISIBLE_DEVICES=0`; `SLURM_JOB_GPUS=0`.
- `nvidia-smi` and its inventory query exposed **one NVIDIA H100 NVL**,
  **95,830 MiB** total GPU memory, with no running GPU processes at observation.
- GPU UUID: `GPU-23fa2f27-82fd-0d5c-540f-29ab2bef077a`.
- NVIDIA driver: **570.211.01**. `nvidia-smi` displayed **CUDA Version 12.8**;
  this is the driver's reported CUDA compatibility, not the installed toolkit
  version or a PyTorch runtime measurement.
- Bare `nvcc --version` failed because `nvcc` was absent from PATH.
- Both `/usr/local/cuda/bin/nvcc --version` and
  `/usr/local/cuda-12.6/bin/nvcc --version` succeeded and reported
  **CUDA compilation tools 12.6, V12.6.77**.
- `ldconfig -p` listed `libcuda.so`, `libnvidia-ml.so`, and `libcudart.so.12`;
  the latter resides under `/usr/local/cuda-12.6/targets/x86_64-linux/lib/`.
- `torch.version.cuda` was not queried because torch was not installed in any
  discovered interpreter. No CUDA kernels, tensors, or compiler workloads ran.

### CPU and RAM visibility

- `SLURM_CPUS_PER_TASK=1`, `SLURM_CPUS_ON_NODE=2`.
- Process CPU affinity and effective cgroup cpuset were **`24,72`**.
- `nproc` reported **1**, with `OMP_NUM_THREADS=1` deliberately set by the script;
  this output must not be interpreted as the complete affinity mask.
- `nproc --all` reported **96** host logical CPUs.
- `free -h` reported **755 GiB host RAM**, approximately **732 GiB available**,
  and **15 GiB swap**. These are host totals, not job entitlements.
- `SLURM_MEM_PER_NODE=2048`. The job and step-user cgroup ancestors each had
  `memory.max=2147483648` and `memory.high=2147483648`: an effective **2 GiB**
  memory limit. The leaf's `memory.max=max` does not remove ancestor limits.
- `MaxRSS` was blank in the accounting response; peak memory use is not known.

### Shared project and frontend marker

The compute job successfully read the project directory, `AGENTS.md`, and
`pyproject.toml`. It read the small marker created on `lagrangectl` at
`2026-09-25T14:44:37.051698+00:00`, including the token
`fd580300-6d0d-4645-8e52-3de9107eee64`.

The marker SHA-256 matched on frontend and compute node:

```text
3826488af34796e6fdf3a62dc77eeb0073469593887c55a6aac73b63faf0c0a8
```

The compute-generated log was subsequently readable from the frontend.

### Filesystems and quota visibility

Values below are snapshots from `df -hT`, rounded by that utility.

| Mount | Source / type | Size | Available |
|---|---|---:|---:|
| `/home` | `10.50.19.60:/lagrange_vol0`, GlusterFS | 14 TB | 12 TB |
| `/` | `/dev/sda4`, XFS | 200 GB | 107 GB |
| `/lagrangeGFS` | `/dev/mapper/vg--data-lv--data`, XFS | 14 TB | 13 TB |
| `/dev/shm` | tmpfs | 378 GB | 378 GB |

- `/home` is mounted read/write on the same GlusterFS source observed on the
  frontend. Root is a different, compute-local filesystem.
- Inode use was 1% on `/home` and 2% on `/`.
- Root mount options explicitly include `noquota`.
- The `quota` command was unavailable. No personal or project quota for `/home`
  or `/lagrangeGFS` was established. Aggregate free space is not a user quota.
- `/scratch`, `/data`, `/datasets`, and `/checkpoints` were absent.
- `QSO_DATA_ROOT`, `QSO_OUTPUT_ROOT`, `QSO_CACHE_ROOT`, and
  `QSO_CHECKPOINT_ROOT` were all unset.
- Large tmpfs capacity shown by `df` does not increase the job memory limit.

### Environment systems and Python

| Tool | Result in this batch environment |
|---|---|
| `module` / `module avail` | Unavailable, including checked standard initialization paths; no module catalog could be listed |
| `conda` | Not found on PATH |
| `micromamba` | Not found on PATH |
| `uv` | `/home/prignano/.local/bin/uv`, version **0.10.11** |
| `apptainer` | Not found on PATH |
| `singularity` | Not found on PATH |
| `python` | Not found on PATH |
| `python3` | `/usr/bin/python3` (also reachable through `/bin/python3`) |

The following existing interpreters were executed only for version and package
discovery. **torch, numpy, and pytest were absent in all three**:

| Executable | Python version |
|---|---|
| `/usr/bin/python3` | 3.12.3 |
| `/home/prignano/.venv/bin/python` | 3.10.20 |
| `/home/prignano/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3` | 3.10.20 |

The discovery script also checked common home venv/conda locations, direct
environments under `/opt`, PATH, and any conda environment registry. No additional
interpreter was found by those checks. Missing tools may exist elsewhere or
require site-specific initialization; the results do not establish cluster-wide
absence.

## Inferred facts

- `/home/prignano` is a suitable candidate for future shared environment and
  artifact directories outside the source tree: project reads, marker identity,
  and compute-log visibility demonstrate sharing for this job. Capacity policy,
  quota, and suitability for large I/O still need confirmation.
- Allocation of two logical CPUs for a one-CPU request is consistent with the
  previously observed `CR_CORE_MEMORY` scheduler selection and two threads per
  core. The exact physical-core mapping was not queried in this job.
- Existing CUDA tools and working NVIDIA management access suggest a usable
  starting point for a future CUDA-enabled Python environment. They do not prove
  PyTorch compatibility or numerical behavior.

## Unresolved questions and prerequisites for later work

- Which storage paths are approved for datasets, caches, environments, and
  checkpoints, and what user/project quotas and retention rules apply?
- Is `/lagrangeGFS` intended for user artifacts, writable by this user, or shared
  beyond `lagrange0`? Its mount presence alone does not answer these questions.
- Are there administrator-provided Python environments or module initialization
  paths outside the bounded search locations?
- What dependency versions and offline package source should be used for a
  reproducible environment? None was selected or installed here.
- CUDA execution, PyTorch CUDA compatibility, H100 SVD/dtype support, and the
  mathematical suite remain untested. These require a later authorized compute
  task after an appropriate environment exists.
- Dataset/tokenizer locations and all `QSO_*_ROOT` choices remain unconfigured.
  No training assets were searched for or downloaded.
- This accepted one-GPU job establishes that this resource request works for the
  current account; it does not establish all scheduler limits or future capacity.

Validation is complete. No subsequent setup or implementation was started.
