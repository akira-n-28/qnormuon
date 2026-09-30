# Source only from the reusable SLURM scripts.
set -euo pipefail
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || { echo 'SLURM required'; exit 1; }
[[ $(hostname -s) != lagrangectl ]] || { echo 'Compute node required'; exit 1; }
cd /home/prignano/qnormuon
export PYTHONPATH="$PWD/.qso-tools/network-guard:$PWD/.qso-tools/pytest-site:$PWD"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export TMPDIR="$PWD/.qso-tools/tmp" TMP="$PWD/.qso-tools/tmp" TEMP="$PWD/.qso-tools/tmp"
export XDG_CACHE_HOME="$TMPDIR" TORCH_HOME="$TMPDIR" CUDA_CACHE_DISABLE=1
export PIP_NO_INDEX=1 UV_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export QSO_NETWORK_AUDIT_LOG="$PWD/cluster/tiny-network-${SLURM_JOB_ID}.log"
export QSO_PYTHON=/home/prignano/modded-nanogpt/.venv/bin/python
unset PYTEST_ADDOPTS
hostname
date --iso-8601=seconds
printf 'JOB=%s NODE=%s CUDA_VISIBLE_DEVICES=%s CPUS=%s MEM=%s PWD=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "${CUDA_VISIBLE_DEVICES:-}" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM_PER_NODE" "$PWD"
df -h /home
nvidia-smi
