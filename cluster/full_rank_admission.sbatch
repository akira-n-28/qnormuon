#!/bin/bash
#SBATCH --job-name=qso-full-rank-admission
#SBATCH --partition=mat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/full_rank_admission-%j.log
#SBATCH --error=cluster/full_rank_admission-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/full_rank_admission.py
"$QSO_PYTHON" -m pytest -q --basetemp="/home/prignano/qnormuon-runs/admission-tests-${SLURM_JOB_ID}" --junitxml="cluster/admission-tests-${SLURM_JOB_ID}.xml"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
