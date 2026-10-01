#!/bin/bash
#SBATCH --job-name=qso-dir-scale
#SBATCH --partition=mat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/direction_scales-%j.log
#SBATCH --error=cluster/direction_scales-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/check_direction_scales.py
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
