#!/bin/bash
#SBATCH --job-name=qso-svd-account
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/profile_solver_svd_accounting-%j.log
#SBATCH --error=cluster/profile_solver_svd_accounting-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/profile_solver_svd_accounting.py
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
