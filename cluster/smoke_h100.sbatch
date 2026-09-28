#!/bin/bash
# Submit after the offline installer succeeds; not a mathematical-suite run.
#SBATCH --job-name=qso-h100-smoke
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=cluster/smoke_h100-%j.log
#SBATCH --error=cluster/smoke_h100-%j.log

set -euo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail 'Submit this script with sbatch.'
grep -Eq "/job_${SLURM_JOB_ID}(/|$)" /proc/self/cgroup || fail 'Not inside the SLURM job cgroup; refusing frontend execution.'
[[ $(hostname -s) != lagrangectl ]] || fail 'GPU smoke work on the frontend is forbidden.'
export QSO_PYTHON=/home/prignano/.venv/bin/python
[[ -x "$QSO_PYTHON" ]] || fail 'Documented Python environment is missing.'
[[ -f /home/prignano/.venv/qnormuon-offline-install.json ]] || fail 'Successful offline installation marker is absent; run the installer first.'
export XDG_CACHE_HOME=/home/prignano/.cache/qnormuon
export TMPDIR=/home/prignano/.cache/qnormuon/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR"
export CUDA_CACHE_PATH=/home/prignano/.cache/qnormuon/cuda
export TORCH_HOME=/home/prignano/.cache/qnormuon/torch
export TRITON_CACHE_DIR=/home/prignano/.cache/qnormuon/triton
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for path in /home/prignano/.venv "$TMPDIR" "$CUDA_CACHE_PATH" "$TORCH_HOME" "$TRITON_CACHE_DIR"; do
    [[ $(realpath -m -- "$path") == /home/* ]] || fail "Path resolves outside /home: $path"
done
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$TORCH_HOME" "$TRITON_CACHE_DIR"
for path in /home/prignano/.venv "$TMPDIR" "$CUDA_CACHE_PATH" "$TORCH_HOME" "$TRITON_CACHE_DIR"; do
    [[ $(findmnt -n -o TARGET -T "$path") == /home ]] || fail "Path is not on shared /home: $path"
done
exec 9>"$XDG_CACHE_HOME/environment.lock"
flock -n -s 9 || fail 'Environment installation is in progress.'
hostname
date --iso-8601=seconds
printf 'SLURM_JOB_ID=%s SLURM_JOB_NODELIST=%s CUDA_VISIBLE_DEVICES=%s CPUS=%s MEMORY_MB=%s\n' "$SLURM_JOB_ID" "${SLURM_JOB_NODELIST:-}" "${CUDA_VISIBLE_DEVICES:-}" "${SLURM_CPUS_PER_TASK:-}" "${SLURM_MEM_PER_NODE:-}"
pwd
nvidia-smi
df -h /home

"$QSO_PYTHON" -I - <<'PY'
import importlib.metadata, json, pathlib, re, sys

if sys.version_info[:3] != (3, 10, 20) or sys.prefix != '/home/prignano/.venv':
    raise SystemExit('ERROR: expected the documented Python 3.10.20 environment')
marker = json.loads((pathlib.Path(sys.prefix) / 'qnormuon-offline-install.json').read_text())
versions = {re.sub(r'[-_.]+', '-', d.metadata['Name']).lower(): d.version for d in importlib.metadata.distributions()}
if marker.get('environment') != sys.prefix or marker.get('python') != '3.10.20' or marker.get('versions') != versions:
    raise SystemExit('ERROR: environment differs from successful installation record')
expected = {'torch': '2.10.0+cu126', 'numpy': '2.2.6', 'pytest': '9.0.2', 'pip': '25.3'}
if any(versions.get(k) != v for k, v in expected.items()):
    raise SystemExit('ERROR: required package versions differ from the documented pins')
print('Python:', sys.executable, sys.version)
print('Installation provenance:', json.dumps(marker, sort_keys=True))

import torch
print('torch.__version__:', torch.__version__)
print('torch.version.cuda:', torch.version.cuda)
print('torch.cuda.is_available():', torch.cuda.is_available())
if str(torch.__version__) != expected['torch'] or torch.version.cuda != '12.6':
    raise SystemExit('ERROR: imported torch does not match the selected cu126 build')
if not torch.cuda.is_available(): raise SystemExit('ERROR: CUDA unavailable')
print('torch.cuda.device_count():', torch.cuda.device_count())
print('torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0))
print('torch.cuda.get_device_properties(0):', torch.cuda.get_device_properties(0))
print('Compiled CUDA architectures:', torch.cuda.get_arch_list())
if 'H100' not in torch.cuda.get_device_name(0): raise SystemExit('ERROR: allocated GPU is not an H100')
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
device = torch.device('cuda:0')
print('Smoke configuration: deterministic tensors, pair shape=(2,16,8), seed not applicable; no model/data/optimizer')
failures, results = [], {}

def probe(label, operation, required=True):
    try:
        details = operation()
        torch.cuda.synchronize(device)
        results[label] = {'status': 'PASS', 'details': details}
    except Exception as e:
        results[label] = {'status': 'FAIL', 'error': f'{type(e).__name__}: {e}'}
        if required: failures.append(label)
    print(label + ': ' + json.dumps(results[label], sort_keys=True), flush=True)

def allocation():
    a = torch.ones(16, device=device)
    if a.sum().item() != 16: raise RuntimeError('CUDA allocation/value check failed')
    return {'device': str(a.device), 'elements': a.numel()}
probe('cuda_allocation', allocation)

# Full-column-rank batched matrices, including two sides as in SmoothDual.
# All casts below choose input dtype explicitly; native BF16 SVD is never
# promoted silently. Diagnostics alone are accumulated in float64.
base = torch.zeros((2, 16, 8), dtype=torch.float64, device=device)
base[:, :8, :] = torch.diag(torch.arange(1, 9, dtype=torch.float64, device=device))
base[:, 8:, :] = torch.arange(64, dtype=torch.float64, device=device).reshape(8, 8) / 128
eye = torch.eye(8, dtype=torch.float64, device=device)

def matmul(dtype):
    a = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4).to(dtype)
    product = a @ torch.eye(4, device=device, dtype=dtype)
    if not torch.equal(product, a): raise RuntimeError('Identity matrix product differs from input')
    return {'input_dtype': str(dtype), 'output_dtype': str(product.dtype)}

def svd(dtype):
    a = base.to(dtype)
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    if not all(torch.isfinite(x).all().item() for x in (u, s, vh)):
        raise RuntimeError('Nonfinite SVD outputs')
    reconstructed = (u.double() * s.double().unsqueeze(-2)) @ vh.double()
    error = (torch.linalg.vector_norm(reconstructed - a.double()) / torch.linalg.vector_norm(a.double())).item()
    tolerance = {torch.float64: 1e-10, torch.float32: 2e-5, torch.bfloat16: 5e-2}[dtype]
    if error > tolerance: raise RuntimeError(f'SVD reconstruction error {error} > {tolerance}')
    return {'input_dtype': str(dtype), 'factor_dtype': str(u.dtype), 'relative_reconstruction_error': error}

def polar(dtype):
    a = base.to(dtype)
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    p = u @ vh  # Same exact SVD/polar path as experiments/dual_solver.py.
    gram_error = (p.double().transpose(-2, -1) @ p.double() - eye).abs().max().item()
    support_error = (((a.double() * p.double()).sum((-2, -1)) - s.double().sum(-1)).abs() / s.double().sum(-1)).max().item()
    tolerance = 1e-10 if dtype == torch.float64 else 3e-5
    if not torch.isfinite(p).all().item() or gram_error > tolerance or support_error > tolerance:
        raise RuntimeError(f'Polar error: Gram={gram_error}, support={support_error}')
    return {'input_dtype': str(dtype), 'returned_dtype': str(p.dtype), 'max_gram_error': gram_error, 'relative_support_error': support_error}

for dtype in (torch.float64, torch.float32, torch.bfloat16):
    probe('matmul_' + str(dtype), lambda d=dtype: matmul(d))
    probe('native_svd_' + str(dtype), lambda d=dtype: svd(d), required=(dtype != torch.bfloat16))
for dtype in (torch.float64, torch.float32):
    probe('full_rank_polar_' + str(dtype), lambda d=dtype: polar(d))

print('RESULTS:', json.dumps(results, sort_keys=True, indent=2))
if failures: raise SystemExit('SMOKE_FAILED: ' + ', '.join(failures))
if results['native_svd_torch.bfloat16']['status'] == 'FAIL':
    print('SMOKE_BASELINE_PASS; BFLOAT16_NATIVE_SVD_FAILED (see exact error above). No BF16 SVD support is claimed.')
else:
    print('SMOKE_BASELINE_PASS; BFLOAT16_NATIVE_SVD_PASSED for this small input only.')
print('Full-rank polar checks do not establish deficient-rank selection, quotient certificates, or optimizer correctness.')
PY
date --iso-8601=seconds
