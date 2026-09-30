#!/bin/bash
# Research component timing on one saved real paired residual.
#SBATCH --job-name=qso-polar-prof
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --output=cluster/profile_smooth_polar_h100-%j.log
#SBATCH --error=cluster/profile_smooth_polar_h100-%j.log
source /home/prignano/qnormuon/cluster/tiny_environment.sh
"$QSO_PYTHON" - <<'PY'
import json,statistics,time
import torch
from cluster.study_smooth_gram_h100 import real_inputs
from experiments.smooth_polar_alternative import diagnose,polar_derivative
rows,residuals,provenance=real_inputs()
b=residuals[-1]
output={}
for route in ('tall','qr_square'):
    samples=[]
    for _ in range(2):
        [diagnose(x,route=route) for x in b]
    for _ in range(8):
        torch.cuda.synchronize();start=time.perf_counter()
        result=[diagnose(x,route=route,profile=True) for x in b]
        torch.cuda.synchronize()
        samples.append(dict(total_seconds=time.perf_counter()-start,
            **{key:sum(z.measurements.get(key,0.) for z in result)
               for key in ('reduction_qr_seconds','qr_seconds','cholesky_solve_seconds',
                           'h_form_seconds','h_evd_seconds')}))
    output[route]={key:statistics.median(x[key] for x in samples) for key in samples[0]}
    output[route]['all_accepted']=all(z.accepted for z in result)
    output[route]['iterations']=[z.iterations for z in result]
    e=torch.randn_like(b)
    for _ in range(3):
        [polar_derivative(z,v) for z,v in zip(result,e)]
    hvp=[]
    for _ in range(10):
        torch.cuda.synchronize();start=time.perf_counter()
        [polar_derivative(z,v) for z,v in zip(result,e)]
        torch.cuda.synchronize();hvp.append(time.perf_counter()-start)
    output[route]['derivative_pair_median_seconds']=statistics.median(hvp)
print('PROFILE_RESULT',json.dumps(dict(provenance=provenance,results=output)),flush=True)
PY
[[ ! -s "$QSO_NETWORK_AUDIT_LOG" ]]
