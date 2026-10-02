#!/bin/bash
#SBATCH --job-name=qso-seeds-check
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/qso_multiseed_preflight-%j.log
#SBATCH --error=cluster/qso_multiseed_preflight-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -m pytest -o addopts= -q --basetemp="$TMPDIR/multiseed-tests-${SLURM_JOB_ID}" --junitxml="cluster/multiseed-tests-${SLURM_JOB_ID}.xml"
"$QSO_PYTHON" -u cluster/run_qso_multiseed.py --preflight
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
