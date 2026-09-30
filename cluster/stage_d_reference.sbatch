#!/bin/bash
#SBATCH --job-name=qso-cpu-ref
#SBATCH --partition=mat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:06:00
#SBATCH --output=cluster/stage_d_reference-%j.log
#SBATCH --error=cluster/stage_d_reference-%j.log
set -euo pipefail
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 1
cd /home/prignano/qnormuon
export PYTHONPATH="$PWD/.qso-tools/network-guard:$PWD/.qso-tools/pytest-site:$PWD"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$PWD/.qso-tools/tmp" XDG_CACHE_HOME="$PWD/.qso-tools/tmp"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1 UV_OFFLINE=1
export QSO_NETWORK_AUDIT_LOG="$PWD/cluster/reference-network-${SLURM_JOB_ID}.log"
hostname
date --iso-8601=seconds
printf 'JOB=%s NODE=%s CPUS=%s MEM=%s PWD=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM_PER_NODE" "$PWD"
/home/prignano/modded-nanogpt/.venv/bin/python -u cluster/stage_d_reference.py "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
