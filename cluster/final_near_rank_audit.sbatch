#!/bin/bash
#SBATCH --job-name=qso-rank-audit
#SBATCH --partition=mat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=cluster/final_near_rank_audit-%j.log
#SBATCH --error=cluster/final_near_rank_audit-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/final_near_rank_audit.py "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
