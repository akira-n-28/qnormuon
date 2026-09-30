#!/bin/bash
# Research-only: production solver always uses full SVD for decisions.
#SBATCH --job-name=qso-gram-decisions
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/study_smooth_gram_decisions_h100-%j.log
#SBATCH --error=cluster/study_smooth_gram_decisions_h100-%j.log
set -euo pipefail
source /home/prignano/qnormuon/cluster/tiny_environment.sh
printf 'HOST=%s DATE=%s JOB=%s GPU=%s PYTHON=%s\n' \
  "$(hostname)" "$(date -Is)" "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES" "$QSO_PYTHON"
nvidia-smi
if [[ "${1:-}" != "--near-guard-only" ]]; then
  "$QSO_PYTHON" -m pytest -q
fi
"$QSO_PYTHON" -u cluster/study_smooth_gram_decisions_h100.py "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
