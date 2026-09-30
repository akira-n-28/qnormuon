#!/bin/bash
# Research-only shadow; production full SVD controls the trajectory.
#SBATCH --job-name=qso-polar-shadow
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=cluster/shadow_smooth_polar_h100-%j.log
#SBATCH --error=cluster/shadow_smooth_polar_h100-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" - <<'PY'
import sys, torch, numpy
print('PYTHON=',sys.executable,sys.version.split()[0],flush=True)
print('TORCH=',torch.__version__,'CUDA=',torch.version.cuda,'NUMPY=',numpy.__version__,flush=True)
print('GPU=',torch.cuda.get_device_name(0),flush=True)
PY
"$QSO_PYTHON" -u cluster/shadow_smooth_polar_h100.py "${1:-tall}"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
