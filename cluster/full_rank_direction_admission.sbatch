#!/bin/bash
#SBATCH --job-name=qso-direction
#SBATCH --partition=mat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/full_rank_direction-%j.log
#SBATCH --error=cluster/full_rank_direction-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/full_rank_direction_admission.py
"$QSO_PYTHON" -m pytest -q --basetemp="/home/prignano/qnormuon-runs/direction-tests-${SLURM_JOB_ID}" --junitxml="cluster/direction-tests-${SLURM_JOB_ID}.xml"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
