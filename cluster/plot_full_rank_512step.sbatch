#!/bin/bash
#SBATCH --job-name=qso-512-figures
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=cluster/full_rank_512step_figures-%j.log
#SBATCH --error=cluster/full_rank_512step_figures-%j.log
set -euo pipefail
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || { echo 'SLURM required'; exit 1; }
[[ $(hostname -s) != lagrangectl ]] || { echo 'Compute node required'; exit 1; }
cd /home/prignano/qnormuon
export PYTHONPATH="$PWD/.qso-tools/network-guard:$PWD"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/.qso-tools/tmp/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export QSO_NETWORK_AUDIT_LOG="$PWD/cluster/tiny-network-${SLURM_JOB_ID}.log"
hostname
date --iso-8601=seconds
printf 'CPU-only figure job=%s CPUs=%s memory=%s MB PWD=%s\n' "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM_PER_NODE" "$PWD"
/home/prignano/modded-nanogpt/.venv/bin/python cluster/plot_full_rank_512step.py "$@"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
