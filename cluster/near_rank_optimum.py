"""Saved-fixture study, compute allocations only; no model/data imports."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('SLURM required')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import torch
import qnormuon.coupled_solver as cs
from experiments.near_rank_optimum import research_config, kkt
from experiments.stage_d_forensics import Observer

ROOT=Path('/home/prignano/qnormuon-runs/stage-d-forensics')
FIXTURES={
    'tuned':ROOT/'capture-29044/confirmation-qso-a0.0003-q0.00122474487139/failed_pair.pt',
    'low':ROOT/'capture-29045/coarse-qso-a0.0003-q0.0003/failed_pair.pt',
    'middle':ROOT/'capture-29045/coarse-qso-a0.0003-q0.001/failed_pair.pt',
    'high':ROOT/'capture-29045/coarse-qso-a0.0003-q0.004/failed_pair.pt',
    'ablation':ROOT/'capture-29045/ablation-qso-a0.0002-q0.00122474487139/failed_pair.pt',
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='all');args=p.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    output=Path('/home/prignano/qnormuon-runs/near-rank-optimum')/('study-'+os.environ['SLURM_JOB_ID'])
    output.mkdir(parents=True,exist_ok=True)
    provenance=dict(job=os.environ['SLURM_JOB_ID'],python=os.sys.version,torch=torch.__version__,
                    cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),output=str(output),
                    production_hash=hashlib.sha256(Path('qnormuon/coupled_solver.py').read_bytes()).hexdigest())
    print(json.dumps(provenance),flush=True)
    all_results={}
    for name,path in FIXTURES.items():
        if args.case!='all' and args.case!=name:continue
        f=torch.load(path,map_location='cuda',weights_only=False)
        u,d,a=(f[k].double() for k in ('u','d','a'))
        starts={'warm':f['initial_lambda'],'beta':f['beta'],'zero':torch.zeros_like(f['beta'])}
        partial_path=path.parent/'bounded_reference/partial_reference.pt'
        if partial_path.exists():
            partial=torch.load(partial_path,map_location='cuda',weights_only=False)
            starts['admm']=f['beta']+partial['lam']
        case={}
        for label,lam in starts.items():
            observer=Observer(u,d,a,lam,dict(f['metadata'],research_start=label))
            cfg=asdict(research_config());cfg['_device']=u.device
            torch.cuda.synchronize();started=time.perf_counter()
            result=observer.solve(cfg)
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            independent,polar,pair=kkt(u,d,a,result.lam)
            row=dict(reason=result.reason,converged=result.converged,iterations=result.iterations,
                     counts=asdict(result.counts),seconds=seconds,independent=independent,
                     history=result.history,evaluations=observer.evaluations,newton=observer.newton,
                     multiplier_displacement=float((result.lam-lam).norm()))
            case[label]=row
            torch.save(dict(lam=result.lam.cpu(),polar=polar.cpu(),pair=pair.cpu(),
                            initial_lambda=lam.cpu(),internal_coordinates=observer.coordinates),output/f'{name}-{label}.pt')
            (output/f'{name}.json').write_text(json.dumps(case,indent=2)+'\n')
            print(name,label,result.reason,result.iterations,independent,flush=True)
        all_results[name]=case
    (output/'summary.json').write_text(json.dumps(dict(provenance=provenance,cases=all_results),indent=2)+'\n')


if __name__=='__main__':main()
