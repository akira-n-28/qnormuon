#!/bin/bash
#SBATCH --job-name=qso-tiny-check
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/test_tiny_harness-%j.log
#SBATCH --error=cluster/test_tiny_harness-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" -m pytest -o addopts= -q
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
