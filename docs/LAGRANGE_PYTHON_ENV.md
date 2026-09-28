# Proposed Python environment for Lagrange

Status: **plan only; nothing installed**, 2026-09-25. `AGENTS.md` and
`LAGRANGE_ENVIRONMENT.md` were read completely before this task.

Reuse the existing, empty Python **3.10.20** environment at
`/home/prignano/.venv`, with **uv 0.10.11** and a dedicated cache on shared
`/home`. Select **torch 2.10.0+cu126**, **numpy 2.2.6**, and **pytest 9.0.2**.
Installation is not ready to start: personal quota is unknown, and package-host
DNS resolution failed from the compute node. No environment was created, no
package payload downloaded, and no mathematical test or optimizer code changed.

## Verified local facts

### Frontend quota assessment

Only lightweight command discovery, mount inspection, metadata queries, and uv
help were used on the frontend.

| Tool/query | Result |
|---|---|
| `quota`, `repquota` | Not found on PATH or standard `/usr/{s,}bin`, `/{s,}bin` locations |
| `gluster` | Not found in those locations |
| `lfs`, `mmlsquota` | Not found; `/home` is not Lustre or GPFS |
| `xfs_quota` | `/usr/sbin/xfs_quota`; not applicable to GlusterFS `/home` |
| `getfattr` | `/usr/bin/getfattr` |
| `getfattr -d -m quota /home/prignano` | Exit 0, no matching attributes reported |
| `findmnt -T /home/prignano` | Read/write GlusterFS, `10.50.19.60:/lagrange_vol0` |

In the compute job, explicit queries for
`trusted.glusterfs.quota.limit-set` and `trusted.glusterfs.quota.size` on
`/home/prignano` both returned `ENODATA`. Absence of these client-visible
attributes does not establish the absence of a server-side or ancestor-directory
quota. No administrative quota changes or privileged queries were attempted.

**User/project quota and remaining personal allowance could not be established.**
The free space reported below is filesystem-wide and must not be treated as this
user's allowance. A quota/allowance answer from the cluster administrators remains
necessary to establish safe capacity; no administrator was contacted by this task.

### Compute allocation and storage checks

- Script: [plan_python_environment.sbatch](../cluster/plan_python_environment.sbatch).
- Raw evidence: [plan_python_environment-22028.log](../cluster/plan_python_environment-22028.log).
- Submitted from `/home/prignano/qnormuon` with
  `sbatch --parsable cluster/plan_python_environment.sbatch`.
- Job **22028**, partition `mat`, node `lagrange0`, **1 GPU, 2 CPUs, 16 GiB RAM**,
  five-minute limit. Accounting: **COMPLETED**, exit **0:0**, elapsed **2 seconds**.
- Recorded start: `2026-09-25T16:55:09+02:00`.
- `CUDA_VISIBLE_DEVICES=0`; GPU: H100 NVL, 95,830 MiB; driver 570.211.01.
- A tiny temporary file under `/home/prignano` was written, flushed, and read
  back successfully. A hard link succeeded. The temporary directory was removed.
- `/home/prignano`, `/home/prignano/.venv`, and `/home/prignano/.cache` exist and
  are writable by this user.
- `/home`: 14 TB total, approximately **12 TB available**, 1% inode use,
  GlusterFS. `/`: 200 GB total, 107 GB available; it is not the proposed storage.
- `/usr/local/cuda` resolves to `/usr/local/cuda-12.6`;
  `/usr/local/cuda-12.6/bin/nvcc --version` reports **12.6, V12.6.77**.
- `/home/prignano/.venv/bin/python` reports **3.10.20**, x86_64,
  glibc **2.39**, OpenSSL **3.5.5**. Its distribution inventory is empty.
- Base interpreter prefix:
  `/home/prignano/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu`.
- uv is `/home/prignano/.local/bin/uv`, version **0.10.11**.

### Compute-node network result

The job attempted only small HTTPS metadata GETs, with certificate verification,
timeouts, a 256 KiB per-response limit, and a 1 MiB total body budget. It would use
HEAD only for wheel files; no wheel GET is allowed by the probe.

All three initial metadata requests failed before HTTPS could be established:

| Request | Result |
|---|---|
| `https://download.pytorch.org/whl/cu126/torch/` | Temporary failure in name resolution (`Errno -3`) |
| `https://pypi.org/pypi/numpy/2.2.6/json` | Same DNS failure |
| `https://pypi.org/pypi/pytest/9.0.2/json` | Same DNS failure |

**Zero response-body bytes were downloaded.** Subsequent wheel HEAD requests were
not reached. Working outbound HTTPS to package hosts is therefore **not verified
and was unavailable through name resolution during this job**. This does not
prove a permanent firewall rule: DNS configuration, transient failure, or a
required site proxy remain possible explanations. No DNS, proxy, certificate, or
firewall settings were changed.

Public documentation was checked separately through the assistant's web research
tool. That research is not evidence of network connectivity from Lagrange.

## Selected environment and compatibility rationale

| Component | Proposed exact selection |
|---|---|
| Interpreter | `/home/prignano/.venv/bin/python` — CPython 3.10.20 |
| Environment | `/home/prignano/.venv` (existing; reuse, no recreation) |
| Package manager | `/home/prignano/.local/bin/uv` — 0.10.11 |
| torch | `2.10.0+cu126`, official CUDA 12.6 index |
| NumPy | `2.2.6`, PyPI binary wheel |
| pytest | `9.0.2`, PyPI binary wheel |

The current environment is empty, accessible from compute, and meets the project
Python requirement. Reusing it follows the requested preference and avoids a
second Python installation. Its home-wide name means other future projects could
also use it; it should be reserved for this dependency set. Recheck that it is
still empty before the first installation. If ownership/use changes, choose a
dedicated project environment in a later task rather than overwrite its contents.

The official [PyTorch version instructions](https://pytorch.org/get-started/previous-versions/)
publish torch 2.10.0 with the `cu126` index. The
[official wheel catalog](https://download.pytorch.org/whl/cu126/torch/)
lists the exact target artifact:

```text
torch-2.10.0+cu126-cp310-cp310-manylinux_2_28_x86_64.whl
```

Its CPython 3.10/x86_64 platform matches the existing interpreter, and the
compute-node glibc 2.39 exceeds the wheel's 2.28 floor. This is a specific published
release choice, not a claim that it is the newest version or that the research
suite has already passed on it.

The H100 is a Hopper GPU with compute capability 9.0 according to
[NVIDIA's GPU table](https://developer.nvidia.com/cuda/gpus). The selected CUDA
12.6 build is the proposed H100 backend. Driver 570.211.01 is newer than the
CUDA 12.6 GA driver 560.28.03 listed in
[NVIDIA's release notes](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html).
NVIDIA documents backward compatibility with newer drivers. This supports the
compatibility plan; actual torch CUDA initialization, compiled architecture
coverage, SVD behavior, and dtype support still require a later compute check.

Use the CUDA-enabled wheel and its declared binary runtime dependencies; do not
skip dependencies because `/usr/local/cuda-12.6` exists. The toolkit does not
replace all wheel dependencies. No system CUDA or driver installation is planned,
and no custom extension compilation is needed. Do not prepend the toolkit's
library directory to `LD_LIBRARY_PATH` just to force library selection.

[NumPy 2.2.6](https://pypi.org/project/numpy/2.2.6/) and
[pytest 9.0.2](https://pypi.org/project/pytest/9.0.2/) both support Python >=3.10.
Their published wheels avoid source compilation. torchvision, torchaudio,
containers, model weights, and datasets are not needed for this environment.

## Explicit shared-storage paths

All proposed paths are on the already verified `/home` filesystem, outside the
source repository. They are plans, not directories populated by this task.

| Purpose | Path |
|---|---|
| Existing environment | `/home/prignano/.venv` |
| Existing base Python | `/home/prignano/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu` |
| uv package cache | `/home/prignano/.cache/qnormuon/uv` |
| Temporary extraction/staging | `/home/prignano/.cache/qnormuon/tmp` |
| Other cache root | `/home/prignano/.cache/qnormuon` |
| Optional offline wheelhouse | `/home/prignano/qso-wheelhouse/py310-cu126` |

Explicit cache and temporary paths prevent reliance on `/dev/shm` or frontend
temporary storage. Cache and environment share a filesystem; the successful
small hard-link probe supports trying `--link-mode hardlink` to avoid unnecessary
copies. Actual uv linking across its final directories remains untested. uv
[documents why cache and environment should share a filesystem](https://docs.astral.sh/uv/concepts/cache/).

## Exact planned commands — NOT executed

Run only inside a later authorized SLURM compute allocation, after capacity and
package-source access are established. These commands reuse the existing venv;
they do not create one or install the project itself.

Common path setup for either installation route:

```bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a SLURM compute allocation}"

export QSO_PYTHON=/home/prignano/.venv/bin/python
export XDG_CACHE_HOME=/home/prignano/.cache/qnormuon
export UV_CACHE_DIR=/home/prignano/.cache/qnormuon/uv
export TMPDIR=/home/prignano/.cache/qnormuon/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PIP_CACHE_DIR=/home/prignano/.cache/qnormuon/pip
export CUDA_CACHE_PATH=/home/prignano/.cache/qnormuon/cuda
export TORCH_HOME=/home/prignano/.cache/qnormuon/torch
export TRITON_CACHE_DIR=/home/prignano/.cache/qnormuon/triton
export UV_PYTHON_DOWNLOADS=never
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_INSTALLS=2

mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$PIP_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$TORCH_HOME" "$TRITON_CACHE_DIR"
test -x "$QSO_PYTHON"
findmnt -T "$UV_CACHE_DIR"
findmnt -T "$TMPDIR"
findmnt -T /home/prignano/.venv
df -h /home
```

**Online route, only if compute-node DNS/HTTPS access is restored and verified:**

```bash
/home/prignano/.local/bin/uv pip install \
  --python "$QSO_PYTHON" \
  --cache-dir "$UV_CACHE_DIR" \
  --no-config --no-python-downloads \
  --only-binary :all: --link-mode hardlink \
  --default-index https://pypi.org/simple \
  --torch-backend cu126 \
  'torch==2.10.0+cu126' 'numpy==2.2.6' 'pytest==9.0.2'
```

Explicit `cu126` selection avoids automatic CPU/backend selection. uv documents
this interface in its [PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/);
the installed uv help also exposes the required options. torch comes from the
official CUDA 12.6 index; the remaining packages resolve against PyPI. The exact
torch local-version pin prevents substituting a CPU or different-CUDA wheel.
No source builds or automatic Python downloads are allowed. This command has not
been resolved or executed on Lagrange. It pins the three top-level packages,
**not the entire transitive dependency graph**. Capture a complete resolved
manifest and hashes for reproducibility before treating the environment as fixed.

**Offline route, preferred if compute egress remains unavailable:** have a
complete compatible wheelhouse staged by an approved cluster process after quota
assessment. This task neither assumes an external machine nor initiates any
transfer. The wheelhouse must include the exact torch artifact above, NumPy,
pytest, and every transitive dependency for CPython 3.10/Linux x86_64, together
with a fully pinned, hashed `requirements.lock` containing those exact top-level
versions. The lock must contain package pins and hashes, not remote URLs.

```bash
export QSO_WHEELHOUSE=/home/prignano/qso-wheelhouse/py310-cu126
test -d "$QSO_WHEELHOUSE"
test -s "$QSO_WHEELHOUSE/requirements.lock"
/home/prignano/.local/bin/uv pip install \
  --python "$QSO_PYTHON" \
  --cache-dir "$UV_CACHE_DIR" \
  --no-config --no-python-downloads \
  --offline --no-index --find-links "$QSO_WHEELHOUSE" \
  --only-binary :all: --link-mode hardlink \
  --require-hashes -r "$QSO_WHEELHOUSE/requirements.lock"
```

The wheelhouse and complete lock **do not exist as products of this task**. An
incomplete wheelhouse must fail explicitly; do not fetch missing wheels from the
frontend. The offline command cannot select an unpinned wheel when the complete
hashed lock is supplied. Neither route is authorized for execution in this task.

## Disk footprint and remaining risks

A reliable total footprint could not be determined: compute-node DNS failure
prevented even the torch wheel HEAD/metadata requests, and the dependency graph
was deliberately not downloaded or resolved. Published PyPI listings give
**16.8 MB** for the selected CPython 3.10/Linux NumPy wheel and **374.8 kB** for
pytest's wheel; these small figures exclude torch and its CUDA dependencies and
must not be used as an installation-size estimate.

Capacity accounting for the later task must include:

1. All compressed wheels (including the offline wheelhouse if retained).
2. Unpacked cache contents, which may share blocks with the environment through
   hard links; do not count such savings as guaranteed until checked.
3. Temporary extraction space and possible duplicate copies if linking fails.
4. Inode usage and any retained package/runtime caches.

The toolkit already installed under `/usr/local` does not eliminate those package
costs. Obtain remaining quota or an administrator-confirmed allowance, then
compare it with a complete wheel/expanded-size inventory plus temporary headroom.
No arbitrary quota value or claim that aggregate 12 TB free makes installation
safe is made here. Avoid repeated caches/environments while quota is unresolved.

Other unresolved items are whether compute DNS failure is transient or policy,
whether an approved package mirror/proxy or offline staging process exists,
the exact transitive lock and hashes, and actual H100 behavior with the selected
build. The Python 3.10 choice preserves the existing interpreter but constrains
future dependency upgrades; reassess version support before changing pins.

The only new executable artifact is the small validation script. Installation,
GPU numerical checks, the mathematical suite, and optimizer implementation remain
for separately authorized work.
