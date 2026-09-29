#!/bin/bash
# Research-only full-SVD counterfactual replay. Production code is unchanged.
#SBATCH --job-name=qso-warm-predictor
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/study_dual_warm_start_predictor_h100-%j.log
#SBATCH --error=cluster/study_dual_warm_start_predictor_h100-%j.log
set -euo pipefail
source /home/prignano/qnormuon/cluster/tiny_environment.sh
printf 'HOST=%s DATE=%s JOB=%s PYTHON=%s\n' "$(hostname)" "$(date -Is)" "$SLURM_JOB_ID" "$QSO_PYTHON"
"$QSO_PYTHON" -m pytest -q
"$QSO_PYTHON" -u cluster/study_dual_warm_start_predictor_h100.py
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
