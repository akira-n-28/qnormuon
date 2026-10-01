#!/bin/bash
#SBATCH --job-name=qso-full-rank-512
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=cluster/full_rank_512step-%j.log
#SBATCH --error=cluster/full_rank_512step-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -m pytest -o addopts= -q --basetemp="$TMPDIR/full-rank-512-tests-${SLURM_JOB_ID}" --junitxml="cluster/full-rank-512-tests-${SLURM_JOB_ID}.xml"
"$QSO_PYTHON" -u cluster/full_rank_512step.py
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
