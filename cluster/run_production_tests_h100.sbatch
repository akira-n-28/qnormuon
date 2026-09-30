#!/bin/bash
# Full current mathematical/reference suite with a small H100 spectral check.
#SBATCH --job-name=qso-prod-h100
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/run_production_tests_h100-%j.log
#SBATCH --error=cluster/run_production_tests_h100-%j.log

set -euo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail 'Submit with sbatch.'
[[ $(hostname -s) != lagrangectl ]] || fail 'Refusing to run on the login node.'
[[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || fail 'No allocated GPU is visible.'

QSO_PROJECT=/home/prignano/qnormuon
QSO_PYTHON=/home/prignano/modded-nanogpt/.venv/bin/python
QSO_PYTEST_SITE="$QSO_PROJECT/.qso-tools/pytest-site"
QSO_NETWORK_GUARD="$QSO_PROJECT/.qso-tools/network-guard"
QSO_TMP="$QSO_PROJECT/.qso-tools/tmp"
QSO_NETWORK_AUDIT_LOG="$QSO_PROJECT/cluster/run_production_tests_h100_network-${SLURM_JOB_ID}.log"

[[ -x "$QSO_PYTHON" ]] || fail 'Validated base Python is missing.'
[[ -f "$QSO_PYTEST_SITE/pytest/__init__.py" ]] || fail 'Project-local pytest target is missing.'
[[ -f "$QSO_NETWORK_GUARD/sitecustomize.py" ]] || fail 'Network guard is missing.'
[[ -d "$QSO_TMP" ]] || fail 'Project-local temporary directory is missing.'
cd "$QSO_PROJECT"

export PYTHONPATH="$QSO_NETWORK_GUARD:$QSO_PYTEST_SITE:$QSO_PROJECT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export TMPDIR="$QSO_TMP" TMP="$QSO_TMP" TEMP="$QSO_TMP"
export XDG_CACHE_HOME="$QSO_TMP" TORCH_HOME="$QSO_TMP"
export CUDA_CACHE_DISABLE=1
export PIP_NO_INDEX=1 UV_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export QSO_NETWORK_AUDIT_LOG
unset PYTEST_ADDOPTS

printf 'HOSTNAME=%s\n' "$(hostname)"
date --iso-8601=seconds
printf 'SLURM_JOB_ID=%s\nSLURM_JOB_NODELIST=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$SLURM_JOB_ID" "${SLURM_JOB_NODELIST:-}" "$CUDA_VISIBLE_DEVICES"
printf 'SLURM_CPUS_PER_TASK=%s\nSLURM_MEM_PER_NODE=%s\n' \
  "${SLURM_CPUS_PER_TASK:-}" "${SLURM_MEM_PER_NODE:-}"
printf 'PYTHONPATH=%s\npwd=%s\n' "$PYTHONPATH" "$PWD"
printf 'DATASET_PATH=not applicable; TOKENIZER_PATH=not applicable; CHECKPOINT_PATH=not applicable\n'
printf 'OUTPUT_PATH=%s\nRANDOM_SEED=not applicable; MODEL_CONFIG=not applicable; OPTIMIZER_CONFIG=reference suite; SOLVER_CONFIG=test-defined\n' \
  "$QSO_PROJECT/cluster/run_production_tests_h100-${SLURM_JOB_ID}.log"
printf 'NETWORK_AUDIT_LOG=%s\n' "$QSO_NETWORK_AUDIT_LOG"
df -h /home
nvidia-smi

"$QSO_PYTHON" - <<'PY'
import sys
import sitecustomize
import numpy
import pytest
import torch

assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
print('NETWORK_GUARD:', sitecustomize.__file__, flush=True)
print('PYTHON_EXECUTABLE:', sys.executable, flush=True)
print('PYTHON_VERSION:', sys.version, flush=True)
print('TORCH_VERSION:', torch.__version__, flush=True)
print('TORCH_CUDA_BUILD:', torch.version.cuda, flush=True)
print('NUMPY_VERSION:', numpy.__version__, flush=True)
print('PYTEST_VERSION:', pytest.__version__, flush=True)
print('PYTEST_PATH:', pytest.__file__, flush=True)
print('TORCH_PATH:', torch.__file__, flush=True)
print('NUMPY_PATH:', numpy.__file__, flush=True)
print('CUDA_AVAILABLE:', torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available inside the H100 allocation')
print('GPU_NAME:', torch.cuda.get_device_name(0), flush=True)
print('GPU_PROPERTIES:', torch.cuda.get_device_properties(0), flush=True)
assert sys.version_info[:3] == (3, 10, 20)
assert torch.__version__ == '2.10.0+cu128' and torch.version.cuda == '12.8'
assert numpy.__version__ == '2.2.6' and pytest.__version__ == '9.0.3'
assert pytest.__file__.startswith('/home/prignano/qnormuon/.qso-tools/pytest-site/')
assert torch.__file__.startswith('/home/prignano/modded-nanogpt/.venv/')
assert numpy.__file__.startswith('/home/prignano/modded-nanogpt/.venv/')
assert 'H100' in torch.cuda.get_device_name(0)

base = torch.arange(128, dtype=torch.float64).reshape(16, 8) / 1000
base[:8, :] += torch.diag(torch.linspace(1.0, 2.0, 8, dtype=torch.float64))
eye = torch.eye(8, dtype=torch.float64)
for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 1e-4)):
    x = base.to(device='cuda', dtype=dtype)
    q, s, vt = torch.linalg.svd(x, full_matrices=False)
    p = q @ vt
    torch.cuda.synchronize()
    reconstructed = (q.double() * s.double().unsqueeze(-2)) @ vt.double()
    error = (torch.linalg.vector_norm(reconstructed - x.double()) /
             torch.linalg.vector_norm(x.double())).item()
    gram = (p.double().T @ p.double() - eye.cuda()).abs().max().item()
    print(f'SPECTRAL_SANITY {dtype} PASS reconstruction={error:.9g} polar_gram_max={gram:.9g}', flush=True)
    if error > tolerance or gram > tolerance:
        raise AssertionError(f'{dtype} spectral sanity exceeded {tolerance}')

try:
    torch.linalg.svd(base.to(device='cuda', dtype=torch.bfloat16), full_matrices=False)
    torch.cuda.synchronize()
except NotImplementedError as exc:
    print(f'SPECTRAL_SANITY torch.bfloat16 EXPECTED_FAILURE {type(exc).__name__}: {exc}', flush=True)
else:
    print('SPECTRAL_SANITY torch.bfloat16 SUPPORTED_UNEXPECTEDLY; reassess promotion policy', flush=True)
print('BF16_POLICY: promote SVD/polar work to at least float32 before calling torch.linalg.svd', flush=True)
PY

printf '\nCOLLECTION_BEGIN\n'
collection_start=$(date +%s)
set +e
"$QSO_PYTHON" -m pytest --collect-only -q
collection_rc=$?
set -e
collection_end=$(date +%s)
printf 'COLLECTION_EXIT_CODE=%s COLLECTION_WALL_SECONDS=%s\n' \
  "$collection_rc" "$((collection_end - collection_start))"

printf '\nFULL_PYTEST_BEGIN: python -m pytest -q\n'
test_start=$(date +%s)
set +e
"$QSO_PYTHON" cluster/run_production_validation.py
test_rc=$?
set -e
test_end=$(date +%s)
printf 'PYTEST_EXIT_CODE=%s PYTEST_WALL_SECONDS=%s\n' \
  "$test_rc" "$((test_end - test_start))"

network_rc=0
if [[ -s "$QSO_NETWORK_AUDIT_LOG" ]]; then
    printf 'NETWORK_AUDIT=ATTEMPT_DETECTED\n'
    cat "$QSO_NETWORK_AUDIT_LOG"
    network_rc=1
else
    printf 'NETWORK_AUDIT=NO_PYTHON_OUTBOUND_ATTEMPT_OBSERVED\n'
fi
date --iso-8601=seconds
if (( collection_rc != 0 || test_rc != 0 || network_rc != 0 )); then
    printf 'JOB_RESULT=FAIL collection=%s tests=%s network=%s\n' \
      "$collection_rc" "$test_rc" "$network_rc"
    exit 1
fi
printf 'JOB_RESULT=PASS\n'
