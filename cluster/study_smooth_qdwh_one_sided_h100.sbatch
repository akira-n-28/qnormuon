#!/bin/bash
# Research-only QDWH-controlled pair replay; the locked model remains SVD-controlled.
#SBATCH --job-name=qso-qdwh-one-sided
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/study_smooth_qdwh_one_sided_h100-%j.log
#SBATCH --error=cluster/study_smooth_qdwh_one_sided_h100-%j.log
set -euo pipefail
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -m pytest -o addopts='' -q tests/test_smooth_qdwh_one_sided.py
"$QSO_PYTHON" -u cluster/study_smooth_qdwh_one_sided_h100.py "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
