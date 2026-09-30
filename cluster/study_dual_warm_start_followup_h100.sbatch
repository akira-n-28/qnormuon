#!/bin/bash
# Research-only checkpoint replay and predictor-cost measurement.
#SBATCH --job-name=qso-warm-followup
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=cluster/study_dual_warm_start_followup_h100-%j.log
#SBATCH --error=cluster/study_dual_warm_start_followup_h100-%j.log
set -euo pipefail
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/study_dual_warm_start_followup_h100.py
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
