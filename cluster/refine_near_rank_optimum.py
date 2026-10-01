"""Independent optimization, decomposition, and sensitivity checks on fixtures."""
import os
if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('SLURM required')
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
import qnormuon.coupled_solver as cs
from experiments.near_rank_optimum import independent_lbfgs,kkt,decomposition_check
from cluster.near_rank_optimum import FIXTURES


def rel(a,b):return float((a-b).norm()/b.norm())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('continuation_root');args=parser.parse_args()
    root=Path(args.continuation_root);output=root/('refinement-'+os.environ['SLURM_JOB_ID']);output.mkdir()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    results={}
    for name,path in FIXTURES.items():
        f=torch.load(path,map_location='cuda',weights_only=False)
        u,d,a=(f[k].double() for k in ('u','d','a'))
        torch.cuda.synchronize();started=time.perf_counter()
        lam,history,counts=independent_lbfgs(u,d,a,f['beta'])
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        independent,polar,pair=kkt(u,d,a,lam)
        row=dict(independent_method='safeguarded L-BFGS; no Newton/HVP',seconds=seconds,
                 history=history,counts=asdict(counts),kkt=independent,multiple_starts={})
        # Use every existing continuation endpoint; never train or reconstruct a model.
        for candidate in sorted(root.glob(name+'-*.pt')):
            prior=torch.load(candidate,map_location='cuda',weights_only=False)
            row['multiple_starts'][candidate.stem]=dict(lambda_distance=float((prior['lam']-lam).norm()),
                relative_lambda_distance=rel(prior['lam'],lam),polar_relative=rel(prior['polar'],polar),
                direction_relative=rel(prior['pair'],pair))
        # Three independent factorization routes on the SAME original residual.
        b=a-cs.adjoint(u,d,lam);ql,s,vt=torch.linalg.svd(b,full_matrices=False)
        qr,r=torch.linalg.qr(b,mode='reduced');ur,sr,vtr=torch.linalg.svd(r,full_matrices=False);qrleft=qr@ur
        cpu_b=b.cpu();cl,sc,ct=torch.linalg.svd(cpu_b,full_matrices=False)
        row['decompositions']={
            'gpu_tall':decomposition_check(b,ql,s,vt),
            'gpu_qr_square':dict(**decomposition_check(b,qrleft,sr,vtr),
                sigma_min_relative=((sr[:,-1]-s[:,-1]).abs()/s[:,-1]).tolist(),polar_relative=rel(qrleft@vtr,ql@vt)),
            'cpu_lapack':dict(**decomposition_check(cpu_b,cl,sc,ct),
                sigma_min_relative=((sc[:,-1].cuda()-s[:,-1]).abs()/s[:,-1]).tolist(),
                polar_relative=rel((cl@ct).cuda(),ql@vt)),
        }
        row['cpu_lapack_config']=torch.__config__.show()
        # Matrix construction uncertainty in the standard fp64 operation model.
        unit=torch.finfo(torch.float64).eps/2;gamma3=3*unit/(1-3*unit)
        row['construction_error_bound']=(gamma3*(a.abs()+cs.adjoint(u,d,lam).abs()).norm(dim=(-2,-1))).tolist()
        generator=torch.Generator(device='cuda').manual_seed(421)
        direction_lam=torch.randn(lam.shape,device='cuda',dtype=torch.float64,generator=generator);direction_lam/=direction_lam.norm()
        direction_a=torch.randn(a.shape,device='cuda',dtype=torch.float64,generator=generator);direction_a/=direction_a.norm()
        row['sensitivity']=[]
        for scale in (torch.finfo(torch.float64).eps,torch.finfo(torch.float32).eps,1e-2):
            for kind in ('lambda','objective','residual'):
                newlam=lam+scale*lam.norm()*direction_lam if kind=='lambda' else lam
                newa=a+scale*a.norm()*direction_a if kind!='lambda' else a
                check,newpolar,newpair=kkt(u,d,newa,newlam)
                row['sensitivity'].append(dict(kind=kind,relative_input_scale=scale,kkt=check,
                    polar_relative=rel(newpolar,polar),direction_relative=rel(newpair,pair),
                    objective_change=check['metrics']['dual_objective']-independent['metrics']['dual_objective']))
        # The last scale is explicitly diagnostic; saved temporal magnitude jumps
        # are much larger, and a norm jump alone does not specify their orientation.
        for scale in (17.37,):
            check,newpolar,newpair=kkt(u,d,a*(1+scale),lam)
            row['sensitivity'].append(dict(kind='objective_scale_only_observed_jump',relative_input_scale=scale,
                kkt=check,polar_relative=rel(newpolar,polar),direction_relative=rel(newpair,pair)))
        torch.save(dict(lam=lam.cpu(),polar=polar.cpu(),pair=pair.cpu()),output/f'{name}-lbfgs.pt')
        results[name]=row
        (output/f'{name}.json').write_text(json.dumps(row,indent=2)+'\n')
        print(name,'LBFGS',len(history)-1,'seconds',seconds,'KKT',independent,flush=True)
    (output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
