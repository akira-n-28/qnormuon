# Native PyTorch polar capability on Lagrange

Inspection date: 2026-09-30. This was a read-only, CPU-only API inspection on
the login node. No environment was modified, no package was installed or
downloaded, and no training or GPU benchmark was run.

## Decision: A — no suitable native polar API is available

None of the inspected, already-installed PyTorch environments exposes
`torch.linalg.polar`. Direct `hasattr(torch.linalg, "polar")` checks returned
`False` in every torch-bearing environment below; searching each runtime's
public `torch.linalg` names for `polar` also returned an empty list. In the
validated environment, `torch.polar` exists but its installed docstring says
`polar(abs, angle, *, out=None) -> Tensor`: it constructs a **complex tensor**
from magnitude and angle, not the matrix polar decomposition `B = Q H`.
The only `polar` name visible under `torch.ops.aten` has schema
`aten::polar(Tensor abs, Tensor angle) -> Tensor`, likewise the
complex-number operation. No equivalent supported public PyTorch matrix-polar
API was found in the inspected installations.

The current validated project interpreter is
`/home/prignano/modded-nanogpt/.venv/bin/python` (Python 3.10.20), with
`torch.__version__ == 2.10.0+cu128` and `torch.version.cuda == "12.8"`.
Its `hasattr(torch.linalg, "polar")` result is `False`. The earlier
[H100 environment validation](LAGRANGE_EXISTING_ENV_GPU_VALIDATION.md)
established this environment's CUDA/SVD behavior; this inspection did not
initialize CUDA.

## Installed environment inventory

Each row with torch was queried by invoking that environment's own interpreter
with `-I -B` and importing `torch` **without CUDA initialization**. Thus Python,
`torch.__version__`, `torch.version.cuda`, and API presence are runtime values.
The three no-torch rows were checked with distribution metadata instead of a
torch import. The search for additional obvious environments was limited to
known paths and shallow `/home/prignano` virtual-environment patterns; it did
not recurse through datasets or caches.

| Environment under `/home/prignano` | Python | torch | `torch.version.cuda` | `torch.linalg.polar` |
| --- | --- | --- | --- | --- |
| `modded-nanogpt/.venv` (validated QSO runtime) | 3.10.20 | 2.10.0+cu128 | 12.8 | Absent |
| `pg` | 3.10.20 | 2.10.0+cu128 | 12.8 | Absent |
| `autoresearch/.venv` | 3.10.20 | 2.9.1+cu128 | 12.8 | Absent |
| `TWA/twa_env` | 3.10.20 | 2.5.1+cu121 | 12.1 | Absent |
| `parameter-golf/.venv` | 3.12.3 | 2.10.0+cu128 | 12.8 | Absent |
| `tesi/.venv-qwen36` | 3.12.3 | 2.10.0+cu128 | 12.8 | Absent |
| `tesi/.venv-vllm` | 3.12.3 | 2.8.0+cu128 | 12.8 | Absent |
| `tesi/.venv` | 3.12.3 | 2.6.0+cu124 | 12.4 | Absent |
| `.venv` | 3.10.20 | Absent | N/A | N/A |
| `codex-cluster-deploy/.venv` | 3.12.3 | Absent | N/A | N/A |
| `parameter-golf/myenv` | 3.12.3 | Absent | N/A | N/A |

The previously documented uv cache includes extracted PyTorch artifacts, but
an extracted cache entry is not an installed environment. It was not used to
infer API availability, and nothing was installed from it.

## Consequence for the requested kernel study

There is no callable native `torch.linalg.polar` in the available installations,
so accepted arguments, returned `Q,H` objects, supported CUDA dtypes/devices,
and CUDA backend provenance cannot be tested. In particular, neither a
QDWH-like kernel nor an SVD-backed implementation can be attributed to a
nonexistent API. The planned production-shape numerical check, near-guard spot
check, and synchronized H100 comparison have no native candidate; submitting
a GPU job would not answer the capability question. Per the study's stop rule,
no SLURM job was submitted.

The prior [smooth polar alternative study](SMOOTH_POLAR_ALTERNATIVE_STUDY.md)
remains the evidence for the custom fp64 QDWH route. Production continues to
use its full fp64 thin-SVD smooth backend. This capability result makes no
claim about future PyTorch releases or uninspected external libraries.
