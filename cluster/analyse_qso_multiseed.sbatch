#!/bin/bash
#SBATCH --job-name=qso-seed-analysis
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/qso_multiseed_analysis-%j.log
#SBATCH --error=cluster/qso_multiseed_analysis-%j.log
set -euo pipefail
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 1
[[ $(hostname -s) != lagrangectl ]] || exit 1
cd /home/prignano/qnormuon
export PYTHONPATH="$PWD/.qso-tools/network-guard:$PWD"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/.qso-tools/tmp/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export QSO_NETWORK_AUDIT_LOG="$PWD/cluster/tiny-network-${SLURM_JOB_ID}.log"
hostname
date --iso-8601=seconds
/home/prignano/modded-nanogpt/.venv/bin/python cluster/analyse_qso_multiseed.py "$1"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
