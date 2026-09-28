#!/bin/bash
#SBATCH --job-name=qso-tiny-smoke
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/run_tiny_smoke-%j.log
#SBATCH --error=cluster/run_tiny_smoke-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/run_tiny_smoke.py "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
