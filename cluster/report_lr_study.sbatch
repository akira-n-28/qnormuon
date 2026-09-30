#!/bin/bash
#SBATCH --job-name=qso-lr-report
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=cluster/report_lr_study-%j.log
#SBATCH --error=cluster/report_lr_study-%j.log
set -euo pipefail
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 1
cd /home/prignano/qnormuon
export PYTHONPATH="$PWD/.qso-tools/network-guard:$PWD"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/.qso-tools/tmp"
export PIP_NO_INDEX=1 UV_OFFLINE=1
export QSO_NETWORK_AUDIT_LOG="$PWD/cluster/lr-report-network-${SLURM_JOB_ID}.log"
hostname
date --iso-8601=seconds
"/home/prignano/modded-nanogpt/.venv/bin/python" -u cluster/report_lr_study.py "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
