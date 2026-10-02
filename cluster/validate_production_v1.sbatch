#!/bin/bash
#SBATCH --job-name=qso-v1-gates
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=cluster/production_v1_gates-%j.log
#SBATCH --error=cluster/production_v1_gates-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -m pytest --basetemp "/home/prignano/qnormuon-runs/production-v1/pytest-${SLURM_JOB_ID}"
"$QSO_PYTHON" -u cluster/validate_production_v1.py
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
