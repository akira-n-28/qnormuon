#!/bin/bash
# Read-only GPU validation of two existing shared-home Python environments.
#SBATCH --job-name=qso-existing-h100
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --output=cluster/validate_existing_envs_h100-%j.log
#SBATCH --error=cluster/validate_existing_envs_h100-%j.log

set -euo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail 'Submit with sbatch.'
[[ $(hostname -s) != lagrangectl ]] || fail 'Refusing to run on the login node.'
[[ ${CUDA_VISIBLE_DEVICES:-} != '' ]] || fail 'No allocated GPU is visible.'
[[ -d /home/prignano/qnormuon ]] || fail 'Project directory is not visible.'

# No toolkit or local package path is injected. Torch must use the installed
# wheels and the host driver as found in this allocation.
export PATH=/usr/local/bin:/usr/bin:/bin
unset LD_LIBRARY_PATH CUDA_HOME CUDA_PATH PYTHONPATH VIRTUAL_ENV CONDA_PREFIX
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

printf 'hostname=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'SLURM_JOB_ID=%s\nSLURM_JOB_NODELIST=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$SLURM_JOB_ID" "${SLURM_JOB_NODELIST:-}" "${CUDA_VISIBLE_DEVICES:-}"
printf 'SLURM_CPUS_PER_TASK=%s\nSLURM_MEM_PER_NODE=%s\n' \
  "${SLURM_CPUS_PER_TASK:-}" "${SLURM_MEM_PER_NODE:-}"
printf 'PATH=%s\nLD_LIBRARY_PATH=%s\n' "$PATH" "${LD_LIBRARY_PATH:-<unset>}"
printf 'pwd=%s\n' "$PWD"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

status=0
for candidate in /home/prignano/modded-nanogpt/.venv /home/prignano/pg; do
    printf '\n=== CANDIDATE %s ===\n' "$candidate"
    if [[ ! -x "$candidate/bin/python" ]]; then
        printf 'CANDIDATE_RESULT: FAIL: Python executable missing\n'
        status=1
        continue
    fi
    if ! QSO_CANDIDATE="$candidate" "$candidate/bin/python" -I -B - <<'PY'
import importlib.metadata as metadata
import json
import os
import sys

name = os.environ['QSO_CANDIDATE']
results = {}
failures = []

def emit(label, status, **details):
    results[label] = {'status': status, **details}
    print('CHECK:', json.dumps({'candidate': name, 'check': label, **results[label]}, sort_keys=True), flush=True)

def probe(label, function, required=True):
    try:
        emit(label, 'PASS', **function())
    except Exception as exc:
        emit(label, 'FAIL', error=f'{type(exc).__name__}: {exc}')
        if required:
            failures.append(label)

emit('python', 'PASS', executable=sys.executable, version=sys.version,
     prefix=sys.prefix)
if sys.version_info[:3] != (3, 10, 20) or sys.prefix != name:
    failures.append('python_expected_environment')

for package in ('pytest', 'pluggy', 'packaging', 'iniconfig', 'pygments'):
    try:
        emit('metadata_' + package, 'PRESENT', version=metadata.version(package))
    except metadata.PackageNotFoundError:
        emit('metadata_' + package, 'ABSENT')

try:
    import numpy as np
    emit('numpy_import', 'PASS', version=np.__version__)
except Exception as exc:
    emit('numpy_import', 'FAIL', error=f'{type(exc).__name__}: {exc}')
    failures.append('numpy_import')

try:
    import torch
    emit('torch_import', 'PASS', version=torch.__version__, cuda_build=torch.version.cuda)
except Exception as exc:
    emit('torch_import', 'FAIL', error=f'{type(exc).__name__}: {exc}')
    print('CANDIDATE_RESULT: FAIL: torch import failed', flush=True)
    raise SystemExit(1)

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
device = torch.device('cuda:0')

def gpu_info():
    available = torch.cuda.is_available()
    details = {'available': available, 'device_count': torch.cuda.device_count()}
    if not available:
        raise RuntimeError(f'torch CUDA unavailable: {details}')
    details['name'] = torch.cuda.get_device_name(0)
    details['properties'] = str(torch.cuda.get_device_properties(0))
    details['capability'] = list(torch.cuda.get_device_capability(0))
    if 'H100' not in details['name']:
        raise RuntimeError(f'Allocated GPU is not H100: {details}')
    return details
probe('gpu_info', gpu_info)

def allocation():
    x = torch.ones(8, device=device)
    torch.cuda.synchronize()
    if x.sum().item() != 8:
        raise RuntimeError('GPU allocation produced incorrect values')
    return {'device': str(x.device), 'elements': x.numel()}
probe('cuda_allocation', allocation)

def matmul():
    a = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    out = a @ torch.eye(4, dtype=torch.float32, device=device)
    torch.cuda.synchronize()
    if not torch.equal(out, a):
        raise RuntimeError('CUDA matrix product differs from identity product')
    return {'dtype': str(out.dtype), 'max_error': float((out - a).abs().max())}
probe('cuda_matmul', matmul)

# One deterministic, full-column-rank tall matrix, used for every dtype and
# both devices. Full rank avoids comparing nonunique polar completions.
base = torch.arange(128, dtype=torch.float64).reshape(16, 8) / 1000.0
base[:8, :] += torch.diag(torch.linspace(1.0, 2.0, 8, dtype=torch.float64))

def svd_polar(x):
    q, sigma, vt = torch.linalg.svd(x, full_matrices=False)
    polar = q @ vt  # Exactly the reference SmoothDual.evaluate path.
    return q, sigma, vt, polar

def measured_svd(x, reference=None):
    q, sigma, vt, polar = svd_polar(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    reconstruction = (q.double() * sigma.double().unsqueeze(-2)) @ vt.double()
    source = x.double()
    rel_reconstruction = (torch.linalg.vector_norm(reconstruction - source) /
                          torch.linalg.vector_norm(source)).item()
    gram = polar.double().T @ polar.double()
    orthogonality = (torch.linalg.vector_norm(gram - torch.eye(8, dtype=torch.float64,
                                                            device=x.device)) /
                     torch.linalg.vector_norm(torch.eye(8, dtype=torch.float64,
                                                        device=x.device))).item()
    details = {'input_dtype': str(x.dtype), 'factor_dtype': str(q.dtype),
               'shape': list(x.shape), 'relative_reconstruction_error': rel_reconstruction,
               'relative_polar_orthogonality_error': orthogonality,
               'singular_min': float(sigma[-1]), 'singular_max': float(sigma[0])}
    if reference is not None:
        ref_sigma, ref_polar = reference
        details['relative_singular_value_error_vs_cpu_float64'] = (
            torch.linalg.vector_norm(sigma.double().cpu() - ref_sigma) /
            torch.linalg.vector_norm(ref_sigma)).item()
        details['relative_polar_error_vs_cpu_float64'] = (
            torch.linalg.vector_norm(polar.double().cpu() - ref_polar) /
            torch.linalg.vector_norm(ref_polar)).item()
    return details

reference = None
try:
    _, cpu_sigma, _, cpu_polar = svd_polar(base)
    reference = (cpu_sigma, cpu_polar)
    emit('cpu_float64_svd_polar', 'PASS', **measured_svd(base))
except Exception as exc:
    emit('cpu_float64_svd_polar', 'FAIL', error=f'{type(exc).__name__}: {exc}')
    failures.append('cpu_float64_svd_polar')

for dtype, label in ((torch.float64, 'float64'),
                     (torch.float32, 'float32'),
                     (torch.bfloat16, 'bfloat16')):
    def run(dtype=dtype):
        return measured_svd(base.to(device=device, dtype=dtype), reference)
    probe('cuda_' + label + '_svd_polar', run, required=(dtype != torch.bfloat16))

print('CANDIDATE_RESULT:', 'PASS' if not failures else 'FAIL',
      json.dumps({'candidate': name, 'required_failures': failures}, sort_keys=True), flush=True)
raise SystemExit(0 if not failures else 1)
PY
    then
        status=1
    fi
done

printf '\nJOB_RESULT: %s\n' "$([[ $status -eq 0 ]] && echo PASS || echo FAIL)"
date --iso-8601=seconds
exit "$status"
