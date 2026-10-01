"""Fixed-fixture numerical output-contract audit. No model or token imports."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('SLURM required')
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import time
from unittest.mock import patch
import torch
import qnormuon.coupled_solver as cs
from experiments.qso_output_contract import value_posterior
from experiments.near_rank_optimum import research_config, independent_lbfgs, kkt
from cluster.near_rank_optimum import FIXTURES
from cluster.full_rank_direction_admission import OLD, CONTROLS


def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();result=fn();torch.cuda.synchronize()
    return result,time.perf_counter()-start


def audit(u,d,a,initial,ref_lam,config):
    check,_,reference=kkt(u,d,a,ref_lam)
    assert check['normalized_gradient_norm']<1e-9
    reference_value=float((a*reference).sum())
    original=cs.certificate;rows=[];first=None
    def observer(*args,**kwargs):
        nonlocal first
        p,metrics=original(*args,**kwargs)
        frame=inspect.currentframe().f_back
        assert frame.f_code is cs._solve.__wrapped__.__code__
        loc=frame.f_locals;ev=loc['ev'];lam=args[3]
        post,seconds=timed(lambda:value_posterior(u,d,a,lam,p,ev.left,
            loc['magnitude']*ev.singular,ev.right,metrics))
        row=dict(iteration=loc['iteration'],**metrics,posterior=post,
            gradient_norm=float(ev.gradient.norm()),cg=loc['counts'].hvp,
            line_trials=loc['counts'].line_trials,posterior_seconds=seconds,
            offline_direction_error=float((p-reference).norm()),
            offline_relative_direction_error=float((p-reference).norm()/reference.norm()),
            offline_reference_value=reference_value,
            offline_support_regret=reference_value-float((a*p).sum()),
            offline_multiplier_difference=float((lam-ref_lam).norm()))
        rows.append(row)
        if first is None and post['research_value_rank_pass']:
            first=row
        return p,metrics
    with patch.object(cs,'certificate',observer):
        result,seconds=timed(lambda:cs.solve_coupled(u,d,a,config=config,initial_lambda=initial))
    return dict(rows=rows,first_candidate=first,reason=result.reason,
        iterations=result.iterations,counts=asdict(result.counts),
        seconds_with_observer=seconds,reference=check),result


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path('/home/prignano/qnormuon-runs/output-contract')/('study-'+os.environ['SLURM_JOB_ID'])
    root.mkdir(parents=True,exist_ok=True)
    paths=list(FIXTURES.values())+[CONTROLS,OLD/'tuned-lbfgs.pt',
        Path('/home/prignano/qnormuon-runs/tiny-transformer/precision-28904-mini/failed-warm-pair.pt')]
    for path in paths:
        if not path.is_file():raise FileNotFoundError(path)
    provenance=dict(job=os.environ['SLURM_JOB_ID'],python=os.sys.version,torch=torch.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),output=str(root),
        seed='saved fixtures; no model/data regeneration',tf32=False,
        deterministic=True,production_configuration=repr(cs.SolverConfig()),
        production_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ('qnormuon/coupled_solver.py','qnormuon/optimizer.py')},
        inputs=[str(path) for path in paths])
    print(json.dumps(provenance),flush=True)
    torch.linalg.svd(torch.eye(8,device='cuda',dtype=torch.float64));torch.cuda.synchronize()
    result=dict(provenance=provenance,cases={},controls=[])
    for name,path in FIXTURES.items():
        f=torch.load(path,map_location='cuda',weights_only=False)
        u,d,a=(f[k].double() for k in ('u','d','a'))
        oracle=torch.load(OLD/f'{name}-lbfgs.pt',map_location='cuda',weights_only=False)
        row,solve=audit(u,d,a,f['initial_lambda'],oracle['lam'],research_config())
        assert row['first_candidate'] is not None and not solve.fallback
        row['fixture']=str(path);result['cases'][name]=row
        print(name,'first',json.dumps(row['first_candidate']),flush=True)
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    controls=torch.load(CONTROLS,map_location='cuda',weights_only=False)
    warm=torch.load(paths[-1],map_location='cuda',weights_only=False)
    controls.append(dict(warm,name='saved warm '+warm['name']))
    for f in controls:
        u,d,a=(f[k].double() for k in ('u','d','a'))
        ref_lam,_,_=independent_lbfgs(u,d,a,cs.multipliers(u,d,a))
        row,solve=audit(u,d,a,f['initial_lambda'],ref_lam,cs.SolverConfig(fallback=False))
        assert solve.converged and row['first_candidate'] is not None
        row['name']=f['name'];result['controls'].append(row)
        print('control',f['name'],'first',row['first_candidate']['iteration'],
              'normalized upper',row['first_candidate']['posterior']['normalized_gap_upper'],flush=True)
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    # Independent analytic stress cases do not use reference solutions to
    # control acceptance; reference values here are known in closed form.
    adversarial=[]
    u=torch.tensor([[1.,0.],[0.,1.],[1.,0.]],device='cuda',dtype=torch.float64)
    for tail,down_scale,lam2 in [(1e-10,1.,0.),(1.,1.,0.),(1e-5,1e-6,0.),(0.,1.,0.),(0.,1.,1e-6)]:
        b=torch.tensor([[1.,0.],[0.,tail],[0.,0.]],device='cuda',dtype=torch.float64)
        a=torch.stack((b,down_scale*b));lam=torch.tensor([0.,lam2,0.],device='cuda',dtype=torch.float64)
        candidate=torch.zeros_like(a);candidate[:,0,0]=1
        if tail==0:candidate[:,1,1]=1
        left,s,right=torch.linalg.svd(a-cs.adjoint(u,u,lam),full_matrices=False)
        p,metrics=cs.certificate(u,u,a,lam,candidate,cs.Counts(),primal_norm_backend='gram_upper')
        post=value_posterior(u,u,a,lam,p,left,s,right,metrics)
        regret=(1+down_scale)*(1+tail)-float((a*p).sum())
        assert regret<=post['gap_upper']+1e-14
        adversarial.append(dict(tail=tail,down_scale=down_scale,lambda_row2=lam2,
            analytic_support_regret=regret,posterior=post,metrics=metrics))
    result['adversarial']=adversarial
    assert result['provenance']['production_hashes']=={
        name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in result['provenance']['production_hashes']}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    Path('cluster/qso_output_contract_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',len(result['cases']),len(result['controls']),flush=True)


if __name__=='__main__':main()
