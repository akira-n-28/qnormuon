#!/bin/bash
#SBATCH --job-name=qso-kernels
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/study_solver_kernels-%j.log
#SBATCH --error=cluster/study_solver_kernels-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/study_solver_kernels.py
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
