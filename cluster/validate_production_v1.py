"""Package/research and unchanged-v0 oracle gates; fixed fixtures only."""
import os
if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('SLURM required')
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types
from dataclasses import asdict
from unittest.mock import patch
import torch
import qnormuon.coupled_solver as cs
from experiments.full_rank_admission import solve_full_rank
from experiments.full_rank_training import compact_metrics
from cluster.near_rank_optimum import FIXTURES
from cluster.full_rank_direction_admission import CONTROLS


def baseline():
    source=subprocess.check_output(['git','show','HEAD:qnormuon/coupled_solver.py'],text=True)
    module=types.ModuleType('v0_original_solver');sys.modules[module.__name__]=module
    exec(compile(source,'HEAD:qnormuon/coupled_solver.py','exec'),module.__dict__)
    # Compare exactly the untouched production loop and decomposition/CG code.
    import ast
    current=Path('qnormuon/coupled_solver.py').read_text()
    for name in ('_solve','_newton_direction','SmoothDual','certificate','primal_top_singular_upper'):
        def extract(s):return next(ast.get_source_segment(s,node) for node in ast.parse(s).body if getattr(node,'name',None)==name)
        assert extract(source)==extract(current),'v0 arithmetic edited: '+name
    return module


def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();out=fn();torch.cuda.synchronize()
    return out,time.perf_counter()-start


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    original=baseline();cfg=cs.SolverConfig(admission_policy='full_rank_epsilon_lmo',fallback=False)
    controls=torch.load(CONTROLS,map_location='cuda',weights_only=False)
    warm=torch.load('/home/prignano/qnormuon-runs/tiny-transformer/precision-28904-mini/failed-warm-pair.pt',map_location='cuda',weights_only=False)
    controls.append(dict(warm,name='saved warm '+warm['name']))
    inputs=[(name,torch.load(path,map_location='cuda',weights_only=False),True) for name,path in FIXTURES.items()]
    inputs += [(f['name'],f,False) for f in controls]
    rows=[]
    with patch.object(cs,'_reference',side_effect=AssertionError('hidden reference')):
        for name,f,below in inputs:
            u,d,a=(f[k].double() for k in ('u','d','a'));lam=f['initial_lambda']
            ref,_=timed(lambda:solve_full_rank(u,d,a,config=cfg,initial_lambda=lam,profile=True))
            new,_=timed(lambda:cs.solve_coupled(u,d,a,config=cfg,initial_lambda=lam))
            assert ref.converged and new.converged
            assert torch.equal(ref.pair,new.pair) and torch.equal(ref.lam,new.lam)
            assert ref.iterations==new.iterations and ref.counts==new.counts
            assert ref.history==new.history and ref.actions==new.actions and ref.evaluations==new.evaluations
            for key,value in compact_metrics(ref).items():
                if key not in ('rank_seconds','value_seconds'):assert new.metrics[key]==value,(name,key)
            v0=cs.solve_coupled(u,d,a,config=cs.SolverConfig(fallback=False),initial_lambda=lam)
            old=original.solve_coupled(u,d,a,config=original.SolverConfig(fallback=False),initial_lambda=lam)
            assert torch.equal(v0.pair,old.pair) and torch.equal(v0.lam,old.lam)
            assert (v0.converged,v0.reason,v0.iterations,asdict(v0.counts))==(old.converged,old.reason,old.iterations,asdict(old.counts))
            assert v0.history==old.history
            for key,value in old.metrics.items():assert v0.metrics[key]==value,(name,key)
            # Alternate the same already-warmed solve; do not optimize it.
            package_times=[];research_times=[]
            for _ in range(3):
                r,t=timed(lambda:solve_full_rank(u,d,a,config=cfg,initial_lambda=lam,profile=False));research_times.append(t)
                n,t=timed(lambda:cs.solve_coupled(u,d,a,config=cfg,initial_lambda=lam));package_times.append(t)
                assert torch.equal(n.pair,r.pair) and torch.equal(n.lam,r.lam)
            rows.append(dict(name=name,below_guard=below,iterations=new.iterations,counts=asdict(new.counts),
                alpha_lower=new.metrics['alpha_lower'],gap=new.metrics['conservative_normalized_gap'],
                research_seconds=research_times,package_seconds=package_times,v0_reason=v0.reason))
            print('EQUIVALENT',name,new.iterations,new.metrics['conservative_normalized_gap'],flush=True)
    root=Path('/home/prignano/qnormuon-runs/production-v1')/('gates-'+os.environ['SLURM_JOB_ID']);root.mkdir(parents=True)
    out=dict(passed=True,fixtures=5,controls=7,rows=rows,v0_original_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ('qnormuon/coupled_solver.py','qnormuon/optimizer.py','qnormuon/_full_rank_admission.py','qnormuon/_numerical_certification.py')})
    (root/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print('GATES_PASSED',root,flush=True)

if __name__=='__main__':main()
