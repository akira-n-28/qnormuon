#!/bin/bash
#SBATCH --job-name=qso-v1-replay
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=cluster/production_v1_replay-%j.log
#SBATCH --error=cluster/production_v1_replay-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -u cluster/production_v1_512step.py --gates "$1"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
