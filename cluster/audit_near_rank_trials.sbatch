#!/bin/bash
#SBATCH --job-name=qso-rank-trials
#SBATCH --partition=mat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/audit_near_rank_trials-%j.log
#SBATCH --error=cluster/audit_near_rank_trials-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/audit_near_rank_trials.py "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
