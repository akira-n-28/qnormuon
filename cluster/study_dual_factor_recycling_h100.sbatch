#!/bin/bash
# Research-only predictor replay; unchanged production full-SVD solver controls.
#SBATCH --job-name=qso-factor-predict
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/study_dual_factor_recycling_h100-%j.log
#SBATCH --error=cluster/study_dual_factor_recycling_h100-%j.log
set -euo pipefail
source /home/prignano/qnormuon/cluster/tiny_environment.sh
printf 'PYTHON=%s CUDA_BUILD=' "$QSO_PYTHON"
"$QSO_PYTHON" -c 'import torch,sys; print(sys.version.split()[0], torch.__version__, torch.version.cuda)'
"$QSO_PYTHON" -m pytest -o addopts= -q
"$QSO_PYTHON" -u cluster/study_dual_factor_recycling_h100.py
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
