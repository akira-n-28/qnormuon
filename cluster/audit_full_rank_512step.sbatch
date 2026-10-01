#!/bin/bash
#SBATCH --job-name=qso-512-audit
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/audit_full_rank_512step-%j.log
#SBATCH --error=cluster/audit_full_rank_512step-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/audit_full_rank_512step.py "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
