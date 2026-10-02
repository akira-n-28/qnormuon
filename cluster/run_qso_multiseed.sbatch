#!/bin/bash
#SBATCH --job-name=qso-paired-seeds
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-3%1
#SBATCH --output=cluster/qso_multiseed-%A_%a.log
#SBATCH --error=cluster/qso_multiseed-%A_%a.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/run_qso_multiseed.py --root "$1" --seed "$((2027 + SLURM_ARRAY_TASK_ID))"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
