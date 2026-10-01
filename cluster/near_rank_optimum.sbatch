#!/bin/bash
#SBATCH --job-name=qso-rank-opt
#SBATCH --partition=mat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/near_rank_optimum-%j.log
#SBATCH --error=cluster/near_rank_optimum-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/near_rank_optimum.py "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
