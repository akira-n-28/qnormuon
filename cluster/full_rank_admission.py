"""Saved fixtures and deterministic synthetic problems only; no model imports."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('SLURM allocation required')
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
import torch
import qnormuon.coupled_solver as cs
from experiments.full_rank_admission import solve_full_rank, rank_posterior
from experiments.near_rank_optimum import research_config, kkt
from cluster.near_rank_optimum import FIXTURES
from cluster.full_rank_direction_admission import OLD, CONTROLS


def timed(fn):
    torch.cuda.synchronize(); start=time.perf_counter()
    out=fn(); torch.cuda.synchronize()
    return out,time.perf_counter()-start


def stats(values):
    x=torch.tensor(values,dtype=torch.float64)
    return dict(median=float(x.quantile(.5)),p95=float(x.quantile(.95)),
                min=float(x.min()),max=float(x.max()),count=len(values))


def summary(r):
    return dict(converged=r.converged,reason=r.reason,iterations=r.iterations,
        counts=asdict(r.counts),metrics=r.metrics,seconds=r.seconds,
        history=r.history,evaluations=r.evaluations,actions=r.actions,
        selection_semantics=r.selection_semantics,selection_certified=r.selection_certified,
        reference_used=r.reference_used)


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path('/home/prignano/qnormuon-runs/full-rank-admission')/('study-'+os.environ['SLURM_JOB_ID'])
    root.mkdir(parents=True,exist_ok=True)
    warm_path=Path('/home/prignano/qnormuon-runs/tiny-transformer/precision-28904-mini/failed-warm-pair.pt')
    paths=list(FIXTURES.values())+[OLD/f'{name}-lbfgs.pt' for name in FIXTURES]+[CONTROLS,warm_path]
    for path in paths:
        if not path.is_file():raise FileNotFoundError(path)
    hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in ('qnormuon/coupled_solver.py','qnormuon/optimizer.py')}
    provenance=dict(job=os.environ['SLURM_JOB_ID'],python=os.sys.version,torch=torch.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),output=str(root),
        cpus=os.environ.get('SLURM_CPUS_PER_TASK'),memory=os.environ.get('SLURM_MEM_PER_NODE'),
        tf32=False,deterministic=True,seed=2091,inputs=[str(p) for p in paths],
        production_configuration=repr(cs.SolverConfig()),production_hashes=hashes,
        research_contract='primary_epsilon_lmo_full_rank; conditional normal finite fp64 model')
    print(json.dumps(provenance),flush=True)
    out=dict(provenance=provenance,cases={},controls=[],stress=[],timing=[])
    def save():
        (root/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
        Path('cluster/full_rank_admission_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    torch.linalg.svd(torch.eye(8,device='cuda',dtype=torch.float64));torch.cuda.synchronize()
    timings=[]
    # Force any accidental reference use to become a visible study failure.
    with patch.object(cs,'_reference',side_effect=AssertionError('hidden CPU reference use')):
        for name,path in FIXTURES.items():
            f=torch.load(path,map_location='cuda',weights_only=False)
            u,d,a=(f[k].double() for k in ('u','d','a'))
            r,seconds=timed(lambda:solve_full_rank(u,d,a,initial_lambda=f['initial_lambda'],profile=True))
            assert r.converged and not r.reference_used
            oracle=torch.load(OLD/f'{name}-lbfgs.pt',map_location='cuda',weights_only=False)
            check,_,reference=kkt(u,d,a,oracle['lam'])
            # Offline-only comparison after research acceptance.
            regret=float((a*(reference-r.pair)).sum())
            assert regret<=r.metrics['posterior']['gap_upper']+1e-12
            row=summary(r);row.update(fixture=str(path),measured_seconds=seconds,
                offline_reference=check,offline_support_regret=regret,
                offline_direction_difference=float((r.pair-reference).norm()),
                offline_relative_direction_difference=float((r.pair-reference).norm()/reference.norm()),
                offline_multiplier_difference=float((r.lam-oracle['lam']).norm()))
            out['cases'][name]=row
            # One compact direction per fixture, outside source storage.
            torch.save(dict(pair=r.pair.cpu(),lam=r.lam.cpu(),selection_semantics=r.selection_semantics),root/f'{name}-accepted.pt')
            print('FIXTURE',name,r.reason,r.iterations,r.metrics['posterior']['normalized_gap_upper'],flush=True)
            timings.append((name,u,d,a,f['initial_lambda'],research_config(),r))
            save()
        inputs=torch.load(CONTROLS,map_location='cuda',weights_only=False)
        warm=torch.load(warm_path,map_location='cuda',weights_only=False)
        inputs.append(dict(warm,name='saved warm '+warm['name']))
        for f in inputs:
            u,d,a=(f[k].double() for k in ('u','d','a'))
            base,_=timed(lambda:cs.solve_coupled(u,d,a,initial_lambda=f['initial_lambda'],config=cs.SolverConfig(fallback=False)))
            r,seconds=timed(lambda:solve_full_rank(u,d,a,initial_lambda=f['initial_lambda'],profile=True))
            assert base.converged and r.converged
            row=summary(r);row.update(name=f['name'],measured_seconds=seconds,
                production_iterations=base.iterations,production_counts=asdict(base.counts),
                production_direction_difference=float((r.pair-base.pair).norm()),
                production_multiplier_difference=float((r.lam-base.lam).norm()))
            out['controls'].append(row)
            timings.append((f['name'],u,d,a,f['initial_lambda'],cs.SolverConfig(fallback=False),r))
            print('CONTROL',f['name'],r.reason,r.iterations,'v0',base.iterations,flush=True)
            save()
        # Paired side-by-side timings after every input has already been solved.
        # Below-guard baseline is explicitly guard-bypassed diagnostic arithmetic,
        # not an actual production-v0 success. Ordinary baseline is unchanged v0.
        for name,u,d,a,initial,base_config,previous in timings:
            base_times=[];research_times=[]
            for _ in range(3):
                _,seconds=timed(lambda:cs.solve_coupled(u,d,a,initial_lambda=initial,
                    config=cs.SolverConfig(fallback=False) if name not in FIXTURES else
                    type(base_config)(rcond_guard=base_config.rcond_guard,fallback=False)))
                base_times.append(seconds)
                r,seconds=timed(lambda:solve_full_rank(u,d,a,initial_lambda=initial,profile=True))
                assert r.converged and torch.equal(r.pair,previous.pair) and torch.equal(r.lam,previous.lam)
                research_times.append(seconds)
            out['timing'].append(dict(name=name,baseline=stats(base_times),research=stats(research_times),
                baseline_samples=base_times,research_samples=research_times,
                incremental_samples=[r-b for r,b in zip(research_times,base_times)],
                rank_seconds=r.metrics['rank_seconds'],value_seconds=r.metrics['value_seconds'],
                smooth_evaluations=len(r.evaluations),newton=r.iterations,
                label='guard-bypassed arithmetic comparison' if name in FIXTURES else 'production-v0 comparison'))
            save()
        # Prescribed spectra, including clustered tails, with independently known
        # stationary solution at lambda=0. Starts perturb that solution without
        # changing the problem. No constant is fitted to these results.
        gen=torch.Generator(device='cuda').manual_seed(2091)
        for shape in ((24,4),(64,12)):
            m,n=shape
            q=torch.linalg.qr(torch.randn(m,n,device='cuda',dtype=torch.float64,generator=gen),mode='reduced').Q
            v=torch.linalg.qr(torch.randn(n,n,device='cuda',dtype=torch.float64,generator=gen)).Q
            u=torch.randn(m,n,device='cuda',dtype=torch.float64,generator=gen)
            u=u/u.norm(dim=1,keepdim=True)
            initial_vector=torch.randn(m,device='cuda',dtype=torch.float64,generator=gen)
            for tail in (3e-4,1.5e-4,1.01e-4,9e-5,7e-5,5e-5,2e-5,1e-5,1e-6,1e-16,0.):
                for scale in (1e-20,1.,1e20):
                    s=torch.full((n,),tail,device='cuda',dtype=torch.float64);s[0]=1.
                    b=scale*(q*s)@v.T;a=torch.stack((b,b))
                    initial=scale*tail*.5*initial_vector
                    r,_=timed(lambda:solve_full_rank(u,u,a,initial_lambda=initial,profile=True))
                    stationary=solve_full_rank(u,u,a)
                    if tail<=1e-16:
                        assert not stationary.converged
                    else:
                        assert stationary.converged
                    # Fixed HVP probes exercise the small-singular regime even
                    # when value tolerance legitimately accepts at iteration 0.
                    probes=[]
                    if stationary.converged:
                        ev=cs.SmoothDual(u,u,a).evaluate(torch.zeros(m,device='cuda',dtype=torch.float64))
                        rank=rank_posterior(u,u,a,ev.coordinate,ev,1.)
                        for _ in range(3):
                            vector=torch.randn(m,device='cuda',dtype=torch.float64,generator=gen)
                            hv=cs.SmoothDual(u,u,a).hvp(ev,vector)
                            assert torch.isfinite(hv).all()
                            probes.append(dict(hvp_norm=float(hv.norm()),curvature=float(vector@hv),rank=rank))
                    row=summary(r);row.update(shape=shape,tail=tail,scale=scale,
                        stationary_converged=stationary.converged,stationary_reason=stationary.reason,hvp_probes=probes)
                    out['stress'].append(row);save()
        # Deterministic nonfinite-HVP injection is a safeguard test, not a real
        # numerical instability. The failed result must never contain an update.
        u=torch.randn(12,3,device='cuda',dtype=torch.float64,generator=gen)
        a=torch.randn(2,12,3,device='cuda',dtype=torch.float64,generator=gen)
        def bad(self,ev,v):return torch.full_like(v,float('nan'))
        with patch.object(cs.SmoothDual,'hvp',bad):
            failed=solve_full_rank(u,u,a)
        assert not failed.converged and failed.pair is None
        out['injected_failure']=summary(failed)
        # Selection scope: a known nonunique optimal face is explicitly excluded,
        # even if a nonoptimal multiplier makes its current residual full rank.
        u=torch.tensor([[1.,0.],[0.,1.],[1.,0.]],device='cuda',dtype=torch.float64)
        b=torch.tensor([[1.,0.],[0.,0.],[0.,0.]],device='cuda',dtype=torch.float64)
        a=torch.stack((b,b));lam=torch.tensor([0.,1e-6,0.],device='cuda',dtype=torch.float64)
        scoped=solve_full_rank(u,u,a,initial_lambda=lam)
        excluded=solve_full_rank(u,u,a,initial_lambda=lam,known_nonsmooth_face=True)
        assert scoped.converged and not scoped.selection_certified
        assert not excluded.converged and excluded.reason=='explicit_nonsmooth_face'
        zero=solve_full_rank(u,u,torch.zeros_like(a),known_nonsmooth_face=True)
        assert zero.converged and torch.count_nonzero(zero.pair)==0
        out['selection_scope']=dict(primary=summary(scoped),excluded=summary(excluded),zero=summary(zero))
    assert hashes=={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in hashes}
    out['peak_gpu_allocation']=torch.cuda.max_memory_allocated()
    save();print('COMPLETE',len(out['cases']),len(out['controls']),len(out['stress']),flush=True)


if __name__=='__main__':main()
