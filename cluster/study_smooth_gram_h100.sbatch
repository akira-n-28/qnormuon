#!/bin/bash
# Diagnostic only: production SmoothDual remains full SVD.
#SBATCH --job-name=qso-smooth-gram
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/study_smooth_gram_h100-%j.log
#SBATCH --error=cluster/study_smooth_gram_h100-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
printf 'PHASE=%s\nOUTPUT_PATH=%s\nINPUT_CONFIG=%s\n' \
  "${1:-adversarial}" "$PWD/cluster/smooth_gram_*-${SLURM_JOB_ID}.json" \
  "$PWD/configs/tiny_transformer/smoke.json"
"$QSO_PYTHON" - <<'PY'
import json, sys, torch, numpy
from pathlib import Path
config=json.loads(Path('configs/tiny_transformer/smoke.json').read_text())
print('PYTHON_EXECUTABLE=',sys.executable,'PYTHON_VERSION=',sys.version.split()[0],flush=True)
print('TORCH_VERSION=',torch.__version__,'TORCH_CUDA=',torch.version.cuda,
      'NUMPY_VERSION=',numpy.__version__,flush=True)
print('GPU=',torch.cuda.get_device_name(0),'SEED=',config['seed'],flush=True)
print('MODEL_CONFIG=',config['model'],'OPTIMIZER_CONFIG=',
      {'qso_lr':config['qso_lr'],'beta':config['beta']},'SOLVER_CONFIG=',config['solver'],flush=True)
print('DATASET_PATH=',config['data']['train_path'],'VALIDATION_PATH=',
      config['data']['validation_path'],'TOKENIZER_PATH=',config['data']['tokenizer_path'],flush=True)
print('INPUT_CHECKPOINT=',str(Path(config['output_root'])/f"smoke-28936-qso-seed{config['seed']}"/'checkpoint.pt'),flush=True)
PY
"$QSO_PYTHON" -u cluster/study_smooth_gram_h100.py "${1:-adversarial}"
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
