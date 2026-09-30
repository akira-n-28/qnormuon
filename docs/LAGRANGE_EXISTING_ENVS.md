# Existing Python environments on Lagrange

Inspection date: 2026-09-25. Scope: lightweight, read-only inspection from the
frontend under `/home/prignano`. `AGENTS.md` was read completely first.

## Result

Four existing CPython 3.10.20 environments contain CUDA-enabled PyTorch and
NumPy. The closest matches to the planned environment are
`/home/prignano/modded-nanogpt/.venv` and `/home/prignano/pg`: both contain
PyTorch **2.10.0+cu128** and NumPy **2.2.6**. Neither contains pytest.
These are candidates for subsequent read-only validation inside an H100 SLURM
allocation, not environments whose imports or GPU operation have been verified.

The documented plan selects **torch==2.10.0+cu126**, **numpy==2.2.6**, and
**pytest==9.0.2**, using `https://download.pytorch.org/whl/cu126` for PyTorch.
No complete environment matching those pins was found in the inspected scope.
The CUDA 12.8 builds found here differ from that plan.

## Verified facts: discovery and inspection method

- A bounded file search used `rg --files --hidden --no-ignore --maxdepth 5`
  under `/home/prignano`, selecting `pyvenv.cfg`, `activate`, and `python`.
  It excluded `.git`, `node_modules`, `.cache`, `.codex`, `.claude`, `.nvm`,
  `.npm`, `data`, `datasets`, `models`, `checkpoints`, `logs`, `runs`, `wandb`,
  `site-packages`, `lib`, `lib64`, `include`, and `share` subtrees.
- Known cache structures were then inspected separately at bounded depth.
  One explicit uv environment-cache symlink was followed to its target.
  This is not an exhaustive search of every directory or symlink under home.
- Each candidate interpreter was queried with `-I -S` and only `import sys`,
  disabling site initialization and avoiding package imports and `.pth` hooks.
  Every candidate version probe completed successfully.
- Package versions came from installed `*.dist-info/METADATA`; wheel ABI and
  platform tags came from `WHEEL`. CUDA build values were read statically from
  `torch/version.py`, without importing or executing it.
- No torch, NumPy, or pytest import, GPU initialization, numerical test,
  installation, download, environment modification, or SLURM submission was
  performed. The only new project artifact is this report.

## Verified facts: environment inventory

All environment paths in the first column are relative to `/home/prignano`.
For **every row**, the queried Python executable is exactly
`/home/prignano/<environment>/bin/python`.
“Absent” means no matching installed distribution metadata was found in that
environment's site-packages. All candidates disable system site-packages.

| Environment | Python | torch build, read statically | CUDA build value | NumPy | pytest | Classification |
| --- | --- | --- | --- | --- | --- | --- |
| `.venv` | 3.10.20 | Absent | N/A | Absent | Absent | Previously inspected; incomplete for this project |
| `modded-nanogpt/.venv` | 3.10.20 | 2.10.0+cu128 | 12.8 | 2.2.6 | Absent | Strong reuse candidate; different CUDA build, missing pytest |
| `pg` | 3.10.20 | 2.10.0+cu128 | 12.8 | 2.2.6 | Absent | Strong reuse candidate; different CUDA build, missing pytest |
| `autoresearch/.venv` | 3.10.20 | 2.9.1+cu128 | 12.8 | 2.2.6 | Absent | Older torch alternative; missing pytest |
| `TWA/twa_env` | 3.10.20 | 2.5.1+cu121 | 12.1 | 2.2.6 | Absent | Older torch alternative; missing pytest |
| `codex-cluster-deploy/.venv` | 3.12.3 | Absent | N/A | Absent | Absent | Different Python ABI and incomplete |
| `parameter-golf/myenv` | 3.12.3 | Absent | N/A | Absent | Absent | Different Python ABI and incomplete |
| `parameter-golf/.venv` | 3.12.3 | 2.10.0+cu128 | 12.8 | 2.4.3 | Absent | CPython 3.12 package ABI; different NumPy pin |
| `tesi/.venv-qwen36` | 3.12.3 | 2.10.0+cu128 | 12.8 | 2.2.6 | Absent | CPython 3.12 package ABI |
| `tesi/.venv-vllm` | 3.12.3 | 2.8.0+cu128 | 12.8 | 2.2.6 | Absent | CPython 3.12 package ABI; older torch |
| `tesi/.venv` | 3.12.3 | 2.6.0+cu124 | 12.4 | 2.4.6 | Absent | CPython 3.12 package ABI; different pins |
| `.cache/uv/archive-v0/Bpeh0574DVkNcp7gj4R1k` | 3.12.3 | Absent | N/A | Absent | Absent | Cached auxiliary environment; incomplete |

The final row was discovered through
`/home/prignano/.cache/uv/environments-v2/a18ede1e7dd8e4d9/e553cb359a57101f`.

For torch 2.10.0 entries, distribution metadata reports `2.10.0`, while
`torch/version.py` reports `2.10.0+cu128`. The CUDA column is the static value
assigned to `cuda` in that file, **not** a runtime measurement of
`torch.version.cuda` or proof that CUDA initializes successfully.

The CPython 3.10.20 executables resolve to:

```text
/home/prignano/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10
```

The CPython 3.12.3 executables resolve to `/usr/bin/python3.12`.
The four Python 3.10 environments containing torch have cp310 extension
filenames; inspected torch `_C` ELF headers identify x86-64. Their NumPy
installations also have cp310 wheel tags. Inspected Python 3.12 torch/NumPy
wheels carry cp312 tags and cannot be reused as binary packages in Python 3.10.
This does not imply those Python 3.12 environments are broken.

## Verified facts: limited dependency checks

Static checks compared active direct torch requirements with installed
distribution metadata. Optional extras and requirements inapplicable to the
interpreter were excluded. Required distribution presence and exact `==`
versions were checked; this was **not** recursive resolution or `pip check`,
and lower-bound constraints were not comprehensively evaluated.

| Environment | Exact direct dependency pins checked | Missing direct requirements / exact-pin mismatches |
| --- | ---: | --- |
| `modded-nanogpt/.venv` | 17 | None found |
| `pg` | 17 | None found |
| `autoresearch/.venv` | 16 | None found |
| `TWA/twa_env` | 13 | None found |
| `tesi/.venv-vllm` | 15 | None found |

Both torch 2.10.0 Python 3.10 candidates include triton 3.6.0 and the CUDA
dependencies requested by their torch metadata, including CUDA runtime
12.8.90, cuBLAS 12.8.4.1, cuDNN 9.10.2.21, and NCCL 2.27.5.
`autoresearch/.venv` contains triton 3.5.1; `TWA/twa_env` contains triton 3.1.0.
Some environments contain additional NVIDIA packages from other CUDA
generations. Their mere presence does not change torch's compiled CUDA build
and does not prove that all libraries can load together.

## Verified facts: caches and reusable artifacts

`VIRTUAL_ENV`, `CONDA_PREFIX`, `PIP_CACHE_DIR`, `UV_CACHE_DIR`, and
`XDG_CACHE_HOME` were unset in the inspection shell. Checking 36 explicit
`.cache/pip` and `.cache/uv` paths under the eleven non-cache environment
prefixes and their parents found only these shared user caches:

```text
/home/prignano/.cache/uv
/home/prignano/.cache/pip
```

### uv extracted package cache

The following entries under `/home/prignano/.cache/uv/wheels-v6` point to
existing extracted package directories in `archive-v0`. They are **not wheel
ZIP files**, despite the cache directory name. Static torch version files were
used to distinguish CUDA builds where the entry name omits a CUDA suffix.

| Cache location below `wheels-v6` | Relevant contents |
| --- | --- |
| `pypi/torch` | 2.10.0+cu128 for cp310 and cp312; 2.11.0+cu130 for cp310 and cp312 |
| `pypi/numpy` | 2.2.6 for cp310 and cp312; 2.3.5, 2.4.3, 2.4.6 for cp312 |
| `index/d2bd0b84f216183d/torch` | 2.9.1+cu128 cp310; 2.8.0+cu128 and 2.11.0+cu128 cp312 |
| `index/e1d141a6ca947dff/torch` | 2.10.0+cpu cp310; not a CUDA torch replacement |
| `index/0683d010c15737d1/torch` | 2.1.2+cu118 and 2.7.1+cu118 cp310 |
| `index/105f2b6141aa1e84/torch` | 2.5.1+cu121 cp310 |
| `index/d53304252ed7ec20/torch` | 2.6.0+cu124 cp312 |

Useful explicit cp310 cache links include:

```text
/home/prignano/.cache/uv/wheels-v6/pypi/torch/2.10.0-3-cp310-cp310-manylinux_2_28_x86_64
/home/prignano/.cache/uv/archive-v0/kaU6007u3fQRhubRLC6QL

/home/prignano/.cache/uv/wheels-v6/pypi/numpy/2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64
/home/prignano/.cache/uv/archive-v0/UvyFSC5aF32x1MVL5u5wv
```

The first pair is the torch cache link and its target; the second pair is
NumPy. NVIDIA cache inspection found 43 NVIDIA package directories with 67
non-sidecar entries under `pypi`, plus 11 package directories in the cu118
index cache and 12 in the cu121 index cache. These include extracted CUDA
dependency packages. No pytest cache directory or torch 2.10.0+cu126 entry
was found in the inspected PyPI and five custom-index cache locations.

The extracted caches are potential inputs to a future explicitly offline uv
operation, subject to dependency and cache-integrity checks. They cannot be
passed directly to `pip install --no-index --find-links` as wheel archives.
No cache material was copied, repacked, or modified. `.msgpack` and `.http`
sidecars are metadata, not standalone dependency wheels.

### pip wheel and HTTP caches

A bounded walk of `.cache/pip` (maximum depth seven, directory cap 1,000)
visited 27 directories without reaching the cap. It found one named wheel:

```text
/home/prignano/.cache/pip/wheels/a9/38/ad/428c088ad555122efa564c1f8bfde77548a8e7b21086b6f7c0/causal_conv1d-1.6.1-cp310-cp310-linux_x86_64.whl
```

Its file size is 110,126,872 bytes. It is not required by the planned minimal
torch/NumPy/pytest environment.

Four HTTP cache bodies were checked by reading ZIP directory/package metadata
only, without extraction or full-file hashing. One contains triton 3.1.0
(209,460,013 bytes):

```text
/home/prignano/.cache/pip/http-v2/b/c/d/d/2/bcdd219ff96033a0c3fce09efc739e18e912e09ffd1ed02c116ceb1c.body
```

Another contains pypdf 6.13.2; two are not ZIP archives. The triton body is a
potential recoverable artifact, but is not currently a correctly named wheel
in a verified wheelhouse and does not match torch 2.10.0's triton 3.6.0 pin.
No torch, NumPy, pytest, or NVIDIA dependency `.whl` archives were found in
this pip-cache inspection.

## Inferences and proposed later validation

The frontend mount inspection identifies `/home` as GlusterFS
`10.50.19.60:/lagrange_vol0`. Earlier compute jobs documented shared `/home`
visibility in `LAGRANGE_ENVIRONMENT.md`. The two strongest candidates and
their managed Python base are all on `/home`, so they should be accessible
from the compute node; these exact environment paths still need verification
inside the allocation.

The earlier H100 validation recorded an H100 NVL, driver 570.211.01 and
`nvidia-smi` CUDA capability 12.8. This makes the existing cu128 builds
plausible candidates for validation. The separately installed CUDA 12.6
toolkit does not make a cu128 torch wheel a cu126 build. Runtime library
loading, driver compatibility in practice, and GPU operation remain untested.

A later short, read-only SLURM job with one H100, two CPUs and 16 GB RAM could
invoke either of these executables directly:

```text
/home/prignano/modded-nanogpt/.venv/bin/python
/home/prignano/pg/bin/python
```

That job should disable bytecode writes (`PYTHONDONTWRITEBYTECODE=1`), avoid
package installation or network access, verify imports and runtime versions,
then test CUDA visibility, a tiny allocation/matmul, separate float64,
float32 and bfloat16 SVD attempts, and the reference polar/SVD path. Successful
matmul would not establish bfloat16 SVD support. No such job was submitted in
this inspection.

The existing `cluster/smoke_h100.sbatch` targets the documented `.venv`, cu126
pins and successful-install marker. It cannot validate these different
environments unchanged. A separate read-only validation job would be needed.

## Unresolved questions and wheelhouse decision

- No candidate was import-tested; native library integrity, complete
  transitive dependency consistency and GPU behavior remain unresolved.
- No inspected environment contains pytest. The four torch-bearing Python
  3.10 environments are therefore incomplete for the requested toolchain,
  even if they work for their original projects.
- The bounded search cannot exclude additional deeper environments, cache
  locations configured elsewhere, or artifacts in excluded directories.
- User quota remains unknown, as documented previously. No recursive disk
  usage scan or space-consuming recovery operation was performed here.
- Reusing an existing environment would share another project's dependencies;
  no permission to alter those environments is implied by this inspection.

**For the exact documented cu126 pins, an offline wheelhouse is still needed:**
this inspection found neither that complete installed environment nor a
complete set of matching wheel archives.

**A large torch download might be avoidable** if a later H100 validation passes
and the environment plan explicitly accepts an existing cu128 build. That
would still leave pytest and its dependencies to supply through an authorized
offline route. The current evidence does not establish that a complete
wheelhouse-free setup is ready.
