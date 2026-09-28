#!/bin/bash
#SBATCH --job-name=qso-precision
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/training_solver_diagnosis-%j.log
#SBATCH --error=cluster/training_solver_diagnosis-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
phase="${1:-cold}"
if (( $# )); then shift; fi
"$QSO_PYTHON" -u cluster/training_solver_diagnosis.py "$phase" "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
