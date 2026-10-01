#!/bin/bash
#SBATCH --job-name=qso-stage-d
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/run_lr_study-%j.log
#SBATCH --error=cluster/run_lr_study-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -m pytest -o addopts= -q --basetemp="$TMPDIR/lr-tests-${SLURM_JOB_ID}" --junitxml="cluster/lr-tests-${SLURM_JOB_ID}.xml"
"$QSO_PYTHON" -u cluster/run_lr_study.py
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
