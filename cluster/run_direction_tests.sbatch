#!/bin/bash
#SBATCH --job-name=qso-direction-tests
#SBATCH --partition=mat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=cluster/direction_final_tests-%j.log
#SBATCH --error=cluster/direction_final_tests-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -m pytest -q --basetemp="/home/prignano/qnormuon-runs/direction-final-tests-${SLURM_JOB_ID}" --junitxml="cluster/direction-final-tests-${SLURM_JOB_ID}.xml"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
