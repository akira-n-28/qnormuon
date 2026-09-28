#!/bin/bash
#SBATCH --job-name=qso-tiny-diag
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/diagnose_tiny_solver-%j.log
#SBATCH --error=cluster/diagnose_tiny_solver-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/diagnose_tiny_solver.py
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
