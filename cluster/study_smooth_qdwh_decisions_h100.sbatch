#!/bin/bash
# Research-only: full SVD controls all optimizer state and decisions.
#SBATCH --job-name=qso-qdwh-decision
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=cluster/study_smooth_qdwh_decisions_h100-%j.log
#SBATCH --error=cluster/study_smooth_qdwh_decisions_h100-%j.log
set -euo pipefail
source /home/prignano/qnormuon/cluster/tiny_environment.sh
printf 'HOST=%s DATE=%s JOB=%s GPU=%s PYTHON=%s\n' \
  "$(hostname)" "$(date -Is)" "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES" "$QSO_PYTHON"
nvidia-smi
"$QSO_PYTHON" -m pytest -o addopts='' -q
if [[ "${1:-}" != "--tests-only" ]]; then
  "$QSO_PYTHON" -u cluster/study_smooth_qdwh_decisions_h100.py
fi
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
