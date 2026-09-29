#!/bin/bash
#SBATCH --job-name=qso-primal-compare
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/compare_primal_backends_h100-%j.log
#SBATCH --error=cluster/compare_primal_backends_h100-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/compare_primal_backends_h100.py
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
