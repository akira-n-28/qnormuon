#!/bin/bash
#SBATCH --job-name=qso-rank-consist
#SBATCH --partition=mat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/check_near_rank_consistency-%j.log
#SBATCH --error=cluster/check_near_rank_consistency-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/check_near_rank_consistency.py "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
