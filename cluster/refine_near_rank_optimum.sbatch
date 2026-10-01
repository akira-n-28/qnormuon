#!/bin/bash
#SBATCH --job-name=qso-rank-check
#SBATCH --partition=mat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/refine_near_rank_optimum-%j.log
#SBATCH --error=cluster/refine_near_rank_optimum-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/refine_near_rank_optimum.py "$@"
"$QSO_PYTHON" -m pytest -q --junitxml="cluster/near-rank-tests-${SLURM_JOB_ID}.xml"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
