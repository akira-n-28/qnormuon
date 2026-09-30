#!/bin/bash
# Research only: inexact inner decisions, authoritative full-SVD outputs.
#SBATCH --job-name=qso-qdwh-hybrid
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/study_qdwh_hybrid_h100-%j.log
#SBATCH --error=cluster/study_qdwh_hybrid_h100-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
# A job-specific basetemp avoids pytest retention deleting tracked old fixtures.
if [[ "${1:-}" == "--skip-tests" ]]; then
  shift  # reuse the completed full-suite validation for unchanged Python code
else
  "$QSO_PYTHON" -m pytest -o addopts='' -q \
    --basetemp="$TMPDIR/qdwh-hybrid-tests-${SLURM_JOB_ID}" \
    --junitxml="cluster/qdwh_hybrid_tests-${SLURM_JOB_ID}.xml"
fi
"$QSO_PYTHON" -u cluster/study_qdwh_hybrid_h100.py "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
