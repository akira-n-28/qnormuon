#!/bin/bash
# Focused research-only exact-decomposition screen, never changes defaults.
#SBATCH --job-name=qso-qr-svd-study
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/study_smooth_qr_svd_h100-%j.log
#SBATCH --error=cluster/study_smooth_qr_svd_h100-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -m pytest -q --junitxml="cluster/smooth_qr_svd_tests-${SLURM_JOB_ID}.xml"
"$QSO_PYTHON" -u cluster/study_smooth_qr_svd_h100.py
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
