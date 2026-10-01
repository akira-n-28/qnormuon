"""Fixed tensor inputs only. Never initializes a model or loads token data."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('SLURM compute allocation required')
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import time
from unittest.mock import patch
import torch
import qnormuon.coupled_solver as cs
from experiments.full_rank_direction import direction_posterior
from experiments.near_rank_optimum import research_config, independent_lbfgs, kkt
from cluster.near_rank_optimum import FIXTURES

OLD=Path('/home/prignano/qnormuon-runs/near-rank-optimum/study-29066/refinement-29068')
CONTROLS=Path('/home/prignano/qnormuon-runs/tiny-transformer/precision-28899-cold/step0-pairs.pt')


def timed(fn):
    torch.cuda.synchronize();t=time.perf_counter();value=fn();torch.cuda.synchronize()
    return value,time.perf_counter()-t


def statistics(values):
    x=torch.tensor(values,dtype=torch.float64)
    return dict(median=float(x.median()),p95=float(x.quantile(.95)),min=float(x.min()),max=float(x.max()))


def audit_solve(u,d,a,initial,reference,cfg):
    rows=[];original=cs.certificate
    def audit(*args,**kwargs):
        pair,metrics=original(*args,**kwargs)
        frame=inspect.currentframe().f_back
        loc=frame.f_locals
        assert frame.f_code is cs._solve.__wrapped__.__code__
        ev=loc['ev'];s=loc['magnitude']*ev.singular
        posterior,seconds=timed(lambda:direction_posterior(u,d,a,args[3],pair,ev.left,s,ev.right,metrics))
        observed=float((pair-reference).norm())
        cast=float((pair.float().double()-pair).norm())
        rows.append(dict(iteration=loc['iteration'],**metrics,posterior=posterior,
            observed_error=observed,relative_observed_error=observed/float(reference.norm()),
            observed_over_bound=observed/posterior['direction_error_upper'],
            fp32_cast_error=cast,relative_fp32_cast_error=cast/float(pair.norm()),
            cg=loc['counts'].hvp,line_trials=loc['counts'].line_trials,
            gradient_norm=float(ev.gradient.norm()),posterior_seconds=seconds))
        return pair,metrics
    with patch.object(cs,'certificate',audit):
        result,seconds=timed(lambda:cs.solve_coupled(u,d,a,config=cfg,initial_lambda=initial))
    return dict(reason=result.reason,iterations=result.iterations,counts=asdict(result.counts),
                seconds_with_observer=seconds,rows=rows),result


def endpoint(u,d,a,lam):
    b=a-cs.adjoint(u,d,lam);left,s,right=torch.linalg.svd(b,full_matrices=False)
    p,metrics=cs.certificate(u,d,a,lam,left@right,cs.Counts(),primal_norm_backend='gram_upper')
    post=direction_posterior(u,d,a,lam,p,left,s,right,metrics)
    return p,metrics,post,(left,s,right)


def sensitivity(u,d,a,lam,reference):
    generator=torch.Generator(device='cuda').manual_seed(971)
    noise=torch.randn(a.shape,generator=generator,device='cuda',dtype=torch.float64)
    noise/=noise.norm()
    baseline=a-cs.adjoint(u,d,lam);l,s,r=torch.linalg.svd(baseline,full_matrices=False)
    normal=torch.randn(l.shape,generator=generator,device='cuda',dtype=torch.float64)
    normal-=l@(l.mT@normal)
    normal=normal[:,:,-1];normal/=normal.norm(dim=1,keepdim=True)
    aligned=normal[:,:,None]*r[:,-1,:][:,None,:];aligned/=aligned.norm()
    output=[]
    for scale in (torch.finfo(torch.float64).eps,torch.finfo(torch.float32).eps):
        for kind,vector in [('random',noise),('smallest_normal',aligned)]:
            newa=a+scale*a.norm()*vector
            p,metrics,post,_=endpoint(u,d,newa,lam)
            # Offline observed sensitivity, not used by the posterior itself.
            output.append(dict(scale=scale,kind=kind,relative_direction_change=float((p-reference).norm()/reference.norm()),
                posterior=post,metrics=metrics))
    return output


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path('/home/prignano/qnormuon-runs/full-rank-direction')/('study-'+os.environ['SLURM_JOB_ID'])
    root.mkdir(parents=True,exist_ok=True)
    provenance=dict(job=os.environ['SLURM_JOB_ID'],python=os.sys.version,torch=torch.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),output=str(root),
        controls=str(CONTROLS),production_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
           for name in ('qnormuon/coupled_solver.py','qnormuon/optimizer.py')})
    print(json.dumps(provenance),flush=True)
    # Warm up GPU decompositions without model or dataset access.
    warm=torch.eye(8,device='cuda',dtype=torch.float64)
    torch.linalg.svd(warm);torch.linalg.eigh(warm);torch.cuda.synchronize()
    cases={};timing_input=None
    for name,path in FIXTURES.items():
        f=torch.load(path,map_location='cuda',weights_only=False)
        u,d,a=(f[k].double() for k in ('u','d','a'))
        oracle=torch.load(OLD/f'{name}-lbfgs.pt',map_location='cuda',weights_only=False)
        check,polar,reference=kkt(u,d,a,oracle['lam'])
        assert check['gradient_norm']<1e-10
        audit,result=audit_solve(u,d,a,f['initial_lambda'],reference,research_config())
        best,metrics,post,factors=endpoint(u,d,a,oracle['lam'])
        audit.update(oracle=check,endpoint=dict(metrics=metrics,posterior=post,
            observed_error=float((best-reference).norm()),fp32_cast_error=float((best.float().double()-best).norm())),
            sensitivity=sensitivity(u,d,a,oracle['lam'],best),metadata=f['metadata'])
        # Raw casting is measured separately; no raw update is applied.
        raw=f.get('raw',{})
        if 'up' in raw and 'down' in raw:
            up,down=raw['up'].double(),raw['down'].double().T
            root_h=(up.norm(dim=1)/down.norm(dim=1)).sqrt()
            lifted=torch.stack((root_h[:,None]*best[0],best[1]/root_h[:,None]))
            cast=lifted.float().double()-lifted
            pulled=torch.stack((cast[0]/root_h[:,None],cast[1]*root_h[:,None]))
            audit['cast']=dict(canonical_error=float(pulled.norm()),raw_euclidean_error=float(cast.norm()),
                max_gauge_lift=float(torch.maximum(root_h,1/root_h).max()))
        cases[name]=audit
        (root/f'{name}.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(name,'first gap pass',next((r['iteration'] for r in audit['rows'] if r['normalized_gap']<=3e-5),None),
              'endpoint bound',post['direction_error_upper'],flush=True)
        if name=='tuned':timing_input=(u,d,a,oracle['lam'],best,*factors,metrics)
    control_rows=[]
    inputs=torch.load(CONTROLS,map_location='cuda',weights_only=False)
    for f in inputs:
        u,d,a=(f[k].double() for k in ('u','d','a'))
        lam,history,counts=independent_lbfgs(u,d,a,cs.multipliers(u,d,a))
        check,polar,reference=kkt(u,d,a,lam)
        assert check['normalized_gradient_norm']<1e-9
        cfg=cs.SolverConfig(fallback=False)
        audit,result=audit_solve(u,d,a,f['initial_lambda'],reference,cfg)
        assert result.converged and not result.fallback
        p,metrics,post,_=endpoint(u,d,a,lam)
        row=dict(name=f['name'],reference=check,**audit,endpoint_posterior=post)
        control_rows.append(row)
        (root/'controls.json').write_text(json.dumps(control_rows,indent=2)+'\n')
        print('control',f['name'],'gap',result.metrics['normalized_gap'],'bound',audit['rows'][-1]['posterior']['direction_error_upper'],flush=True)
    # All factors and radial diagnostics are already available. No SVD is timed
    # in the incremental posterior measurement.
    u,d,a,lam,p,l,s,r,metrics=timing_input
    post_times=[];certificate_times=[]
    for _ in range(3):direction_posterior(u,d,a,lam,p,l,s,r,metrics)
    for _ in range(20):
        _,seconds=timed(lambda:direction_posterior(u,d,a,lam,p,l,s,r,metrics));post_times.append(seconds)
        cached=cs.CachedDualSpectrum(lam,l@r,s,1.)
        _,seconds=timed(lambda:cs.certificate(u,d,a,lam,cached.candidate,cs.Counts(),cached_dual=cached,primal_norm_backend='gram_upper'))
        certificate_times.append(seconds)
    result=dict(provenance=provenance,cases=cases,controls=control_rows,
                timing=dict(posterior=statistics(post_times),current_certificate=statistics(certificate_times)))
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    Path('cluster/full_rank_direction_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SUMMARY',json.dumps(result['timing']),flush=True)


if __name__=='__main__':main()
