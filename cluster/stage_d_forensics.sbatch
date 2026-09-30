#!/bin/bash
#SBATCH --job-name=qso-forensics
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/stage_d_forensics-%j.log
#SBATCH --error=cluster/stage_d_forensics-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -m pytest -o addopts= -q --basetemp="$TMPDIR/forensics-tests-${SLURM_JOB_ID}" --junitxml="cluster/forensics-tests-${SLURM_JOB_ID}.xml"
"$QSO_PYTHON" -u cluster/stage_d_forensics.py "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
