"""Supplemental saved warm control and re-optimized uncertainty probes."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('SLURM required')
from dataclasses import asdict
import json
import math
from pathlib import Path
import torch
from experiments.near_rank_optimum import independent_lbfgs,kkt
from cluster.full_rank_direction_admission import OLD,audit_solve,endpoint
from cluster.near_rank_optimum import FIXTURES
import qnormuon.coupled_solver as cs

torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
root=Path('/home/prignano/qnormuon-runs/full-rank-direction')/('scales-'+os.environ['SLURM_JOB_ID'])
root.mkdir(parents=True,exist_ok=True)
results=dict(job=os.environ['SLURM_JOB_ID'],warm={},perturbations={},update_scales={})
path=Path('/home/prignano/qnormuon-runs/tiny-transformer/precision-28904-mini/failed-warm-pair.pt')
f=torch.load(path,map_location='cuda',weights_only=False)
u,d,a=(f[k].double() for k in ('u','d','a'))
lam,history,counts=independent_lbfgs(u,d,a,cs.multipliers(u,d,a))
check,_,reference=kkt(u,d,a,lam)
assert check['normalized_gradient_norm']<1e-9
audit,result=audit_solve(u,d,a,f['initial_lambda'],reference,cs.SolverConfig(fallback=False))
assert result.converged
results['warm']=dict(path=str(path),name=f['name'],step=f['step'],reference=check,**audit)
print('warm',f['step'],f['name'],audit['rows'][-1],flush=True)
base=json.loads(Path('cluster/full_rank_direction_summary.json').read_text())
for name,path in FIXTURES.items():
    f=torch.load(path,map_location='cuda',weights_only=False)
    u,d,a=(f[k].double() for k in ('u','d','a'))
    oracle=torch.load(OLD/f'{name}-lbfgs.pt',map_location='cuda',weights_only=False)
    oldlam=oracle['lam'];_,_,reference=kkt(u,d,a,oldlam)
    generator=torch.Generator(device='cuda').manual_seed(973)
    noise=torch.randn(a.shape,generator=generator,device='cuda',dtype=torch.float64);noise/=noise.norm()
    _,_,_,(l,s,r)=endpoint(u,d,a,oldlam)
    normal=torch.randn(l.shape,generator=generator,device='cuda',dtype=torch.float64)
    normal-=l@(l.mT@normal);normal=normal[:,:,-1];normal/=normal.norm(dim=1,keepdim=True)
    aligned=normal[:,:,None]*r[:,-1,:][:,None,:];aligned/=aligned.norm()
    trials=[]
    for scale in (torch.finfo(torch.float64).eps,torch.finfo(torch.float32).eps):
        for label,vector in [('random',noise),('smallest_normal',aligned)]:
            trials.append((label,scale,a+scale*a.norm()*vector))
    # The captured objective is already rounded fp32. A one-ulp proxy tests
    # representational sensitivity; it is NOT the unavailable unrounded EMA.
    dest=a.float()+noise.sign().float()
    oneulp=torch.nextafter(a.float(),dest).double()
    trials.append(('one_fp32_ulp_proxy',None,oneulp))
    rows=[]
    for label,scale,newa in trials:
        lam,h,c=independent_lbfgs(u,d,newa,oldlam)
        check,polar,newref=kkt(u,d,newa,lam)
        assert check['normalized_gradient_norm']<1e-9
        p,metrics,posterior,_=endpoint(u,d,newa,lam)
        row=dict(kind=label,scale=scale,relative_objective_change=float((newa-a).norm()/a.norm()),
            relative_optimum_direction_change=float((newref-reference).norm()/reference.norm()),
            within_problem_observed_error=float((p-newref).norm()),posterior=posterior,
            within_problem_observed_over_bound=float((p-newref).norm())/posterior['direction_error_upper'],
            kkt=check,iterations=len(h)-1,counts=asdict(c))
        rows.append(row)
    results['perturbations'][name]=rows
    raw=f['raw'];up,down=raw['up'].double(),raw['down'].double().T
    root_h=(up.norm(dim=1)/down.norm(dim=1)).sqrt()
    raw_weight_norm=float(torch.stack((up,down)).norm())
    step=f['metadata']['training_step'] if 'training_step' in f['metadata'] else {'tuned':87,'low':92,'middle':87,'high':56,'ablation':81}[name]
    peak={'tuned':.0012247448713915891,'low':.0003,'middle':.001,'high':.004,'ablation':.0012247448713915891}[name]
    total,warmup=(512,51) if name=='tuned' else (256,26)
    factor=(step+1)/warmup if step<warmup else .1+.45*(1+math.cos(math.pi*(step-warmup)/(total-warmup-1)))
    lr=peak*factor
    results['update_scales'][name]=dict(step=step,actual_paired_lr=lr,raw_pair_weight_norm=raw_weight_norm,
        lift_norm_upper=float(torch.maximum(root_h,1/root_h).max()),
        rows=[dict(iteration=q['iteration'],raw_update_error_over_weight_upper=lr*float(root_h.max())*q['posterior']['direction_error_upper']/raw_weight_norm)
            for q in base['cases'][name]['rows']])
    print(name,'reoptimized perturbations',[(q['kind'],q['scale'],q['relative_optimum_direction_change']) for q in rows],flush=True)
    (root/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
Path('cluster/full_rank_direction_scales.json').write_text(json.dumps(results,indent=2)+'\n')
