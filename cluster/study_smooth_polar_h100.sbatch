#!/bin/bash
# Research-only; never changes production SmoothDual decisions.
#SBATCH --job-name=qso-polar-study
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/study_smooth_polar_h100-%j.log
#SBATCH --error=cluster/study_smooth_polar_h100-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" - <<'PY'
import json, sys, torch, numpy
from pathlib import Path
config=json.loads(Path('configs/tiny_transformer/smoke.json').read_text())
print('PYTHON=',sys.executable,sys.version.split()[0],flush=True)
print('TORCH=',torch.__version__,'CUDA=',torch.version.cuda,'NUMPY=',numpy.__version__,flush=True)
print('GPU=',torch.cuda.get_device_name(0),'SEED=',config['seed'],flush=True)
print('MODEL=',config['model'],'SOLVER=',config['solver'],flush=True)
print('DATA=',config['data'],'OUTPUT_ROOT=',config['output_root'],flush=True)
PY
"$QSO_PYTHON" -u cluster/study_smooth_polar_h100.py
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
