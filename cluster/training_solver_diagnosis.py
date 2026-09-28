"""Isolated H100 precision study; production code and defaults are not changed.

Run only through training_solver_diagnosis.sbatch. No ADMM, network, or sweep.
Timing runs are separate from instrumented arithmetic-isolation runs.
"""
import os
if __name__ == '__main__' and not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('SLURM allocation required')
import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import signal
import time
from unittest.mock import patch

import torch
from torch.nn import functional as F
import qnormuon.coupled_solver as cs
from qnormuon import regular_canonicalize
from benchmarks.tiny_transformer import (ModelConfig, TinyTransformer, datasets,
    seed_everything, autocast, Optimizers, train_step)


def sync():
    torch.cuda.synchronize()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def record(result, seconds):
    return dict(certified=result.converged, certified_3e5=cs.accepted(result.metrics, 3e-5)
                and result.metrics['residual_rcond'] > 1e-4,
                reason=result.reason, seconds=seconds, iterations=result.iterations,
                counts=asdict(result.counts), metrics=result.metrics, history=result.history)


def solve(u, d, a, dtype, lam=None, budget=100, tol=None):
    sync(); start=time.perf_counter()
    result=cs.solve_coupled(u,d,a, initial_lambda=lam,
        config=cs.SolverConfig(dtype=dtype, tolerance=tol, rcond_guard=1e-4,
                               max_iterations=budget, fallback=False))
    sync()
    return result, record(result,time.perf_counter()-start)


class Instrument:
    """Synchronized nested timings; categories overlap, never add blindly.

    Retains only the best candidate and last line-search base, not all tensors.
    Scalar trace recomputations are intentionally outside headline timings.
    """
    def __init__(self, trace=False):
        self.trace=trace; self.times=defaultdict(float); self.devices=defaultdict(int)
        self.best=None; self.lines=[]; self.base=None; self.stack=ExitStack()

    def __enter__(self):
        for name in ('svd','svdvals'):
            original=getattr(torch.linalg,name)
            def wrapped(x,*args,_fn=original,_name=name,**kwargs):
                self.devices[f'{_name}:{x.device}:{x.dtype}']+=1
                assert x.is_cuda, 'study spectral operation unexpectedly moved to CPU'
                sync(); start=time.perf_counter(); out=_fn(x,*args,**kwargs);sync()
                self.times[_name]+=time.perf_counter()-start
                return out
            self.stack.enter_context(patch.object(torch.linalg,name,wrapped))
        original_cert=cs.certificate
        def cert(u,d,a,lam,candidate,counts):
            sync();start=time.perf_counter();out=original_cert(u,d,a,lam,candidate,counts);sync()
            self.times['certificate']+=time.perf_counter()-start
            if self.best is None or out[1]['normalized_gap'] < self.best['gap']:
                self.best=dict(gap=out[1]['normalized_gap'],candidate=candidate.detach().clone(),lam=lam.clone())
            return out
        self.stack.enter_context(patch.object(cs,'certificate',cert))
        original_direction=cs._newton_direction
        def direction(problem,ev,*args,**kwargs):
            sync();start=time.perf_counter();out=original_direction(problem,ev,*args,**kwargs);sync()
            self.times['cg']+=time.perf_counter()-start
            self.base=(self.last_z.clone(),ev,out.clone())
            return out
        self.stack.enter_context(patch.object(cs,'_newton_direction',direction))
        original_eval=cs.SmoothDual.evaluate
        def evaluate(problem,z):
            sync();start=time.perf_counter();out=original_eval(problem,z);sync()
            elapsed=time.perf_counter()-start
            self.times['evaluate']+=elapsed
            if self.base is not None:self.times['line_search_evaluations']+=elapsed
            self.last_z=z.clone()
            if self.trace and self.base is not None:
                bz,ev,direction=self.base
                step=float(((z-bz).double()*direction.double()).sum()/direction.double().square().sum())
                b=problem.a-cs.adjoint(problem.u,problem.d,z)
                b0=problem.a-cs.adjoint(problem.u,problem.d,bz)
                q,s,v=torch.linalg.svd(b.double(),full_matrices=False)
                truegrad=-cs.horizontal_residual(problem.u.double(),problem.d.double(),q@v)
                trueval=float(s.sum())
                baseval=float(torch.linalg.svdvals(b0.double()).sum())
                slope=float(ev.gradient@direction)
                rounding=2*torch.finfo(z.dtype).eps*max(1.,abs(ev.value))
                self.lines.append(dict(step=step,slope=slope,rounding=rounding,
                    predicted_decrease=step*slope,value=out.value,base_value=ev.value,
                    delta_value=out.value-ev.value,delta_value64_same_rounded_residual=trueval-baseval,
                    gradient_norm=float(out.gradient.norm()),base_gradient_norm=float(ev.gradient.norm()),
                    gradient_norm64_same_rounded_residual=float(truegrad.norm()),
                    resolved_decrease=abs(step*slope)>rounding,
                    gradient_improves=float(out.gradient.norm())<float(ev.gradient.norm()),
                    armijo_pass=out.value <= ev.value+1e-4*step*slope+rounding))
            return out
        self.stack.enter_context(patch.object(cs.SmoothDual,'evaluate',evaluate))
        return self

    def __exit__(self,*args):
        return self.stack.__exit__(*args)


def effective(u,d,a,lam,dtype):
    u,d,a=(x.double() for x in (u,d,a))
    beta=cs.multipliers(u,d,a); centered=a-cs.adjoint(u,d,beta)
    mag=cs._stable_norm(centered);coord=(u.square()+d.square()).sum(1).rsqrt()
    problem=cs.SmoothDual((u*coord[:,None]).to(dtype),(d*coord[:,None]).to(dtype),(centered/mag).to(dtype))
    z=((lam-beta)/(mag*coord)).to(dtype)
    return problem,z


@torch.no_grad()
def inspect_candidate(u,d,a,lam,candidate):
    u,d,a,lam,candidate=(x.double() for x in (u,d,a,lam,candidate))
    p,m=cs.certificate(u,d,a,lam,candidate,cs.Counts())
    projected=cs.horizontal_project(u,d,candidate)
    scale=max(1.,float(torch.linalg.svdvals(projected).max()))
    b=a-cs.adjoint(u,d,lam)
    s64=torch.linalg.svdvals(b)
    b32=a.float()-cs.adjoint(u.float(),d.float(),lam.float())
    s32=torch.linalg.svdvals(b32)
    dual32=float(s32.sum());dual32sum64=float(s32.double().sum())
    primal32=float((a.float()*p.float()).sum())
    primal_same_rounded64=float((a.float().double()*p.float().double()).sum())
    return dict(metrics=m,radial_scale=scale,
        raw_primal=float((a*candidate).sum()),projected_primal=float((a*projected).sum()),
        radial_primal_loss=float((a*projected).sum())-m['primal_objective'],
        raw_horizontal=float(cs.horizontal_residual(u,d,candidate).abs().max()),
        raw_spectral_max=float(torch.linalg.svdvals(candidate).max()),
        primal32=primal32,primal64_same_rounded_candidate=primal_same_rounded64,
        dual32=dual32,dual32_values_summed64=dual32sum64,
        dual64_same_rounded_residual=float(torch.linalg.svdvals(b32.double()).sum()),
        dual64=m['dual_objective'],gap32=(dual32-primal32)/dual32,
        epsilon32_times_dual=torch.finfo(torch.float32).eps*m['dual_objective'],
        smallest_singular_value=float(s64.min()))


@torch.no_grad()
def isolation(u,d,a,r32,r64,candidate):
    initial=inspect_candidate(u,d,a,r32.lam,candidate)
    # Separate SVD/polar error from residual formation at the SAME rounded residual.
    problem,z=effective(u,d,a,r32.lam,torch.float32)
    b=problem.a-cs.adjoint(problem.u,problem.d,z)
    q32,s32,v32=torch.linalg.svd(b,full_matrices=False)
    q64,s64,v64=torch.linalg.svd(b.double(),full_matrices=False)
    p32=q32@v32;p64=q64@v64
    same_rounded=inspect_candidate(u,d,a,r32.lam,p64)
    exact_b=a.double()-cs.adjoint(u.double(),d.double(),r32.lam)
    qe,se,ve=torch.linalg.svd(exact_b,full_matrices=False)
    exact_residual=inspect_candidate(u,d,a,r32.lam,qe@ve)
    repeated_floor=[]
    for _ in range(3):
        prob,zz=effective(u,d,a,r64.lam,torch.float32)
        ev=prob.evaluate(zz)
        repeated_floor.append(inspect_candidate(u,d,a,r64.lam,ev.pair))
    # Polar computed in fp64 and ONLY its output rounded: storage error control.
    prob64,z64=effective(u,d,a,r64.lam,torch.float64)
    pstar=prob64.evaluate(z64).pair
    cast_only=inspect_candidate(u,d,a,r64.lam,pstar.float())
    eye=torch.eye(p32.shape[-1],device='cuda',dtype=torch.float64)
    return dict(actual_fp32_candidate=initial,
        fp64_polar_same_rounded_internal_residual=same_rounded,
        fp64_polar_exact_original_residual=exact_residual,
        fp32_floor_at_fp64_lambda=repeated_floor,fp64_polar_cast_to_fp32=cast_only,
        polar_relative_error_same_rounded_residual=float((p32.double()-p64).norm()/p64.norm()),
        polar_gram_defect_spectral=float(torch.linalg.matrix_norm(p32.double().mT@p32.double()-eye,ord=2).max()),
        normalized_internal_nuclear32=float(s32.sum()),normalized_internal_nuclear64=float(s64.sum()),
        normalized_internal_nuclear32_sum64=float(s32.double().sum()),
        lambda_dual_excess=initial['dual64']-r64.metrics['dual_objective'])


def build(c):
    seed_everything(c['seed'],c['tf32']);torch.set_default_dtype(torch.float32)
    model=TinyTransformer(ModelConfig(**c['model'])).float().cuda()
    train,val=datasets(c)
    return model,train,val


def corpus(c,out):
    model,train,val=build(c)
    before=[p.detach().clone() for p in model.parameters()]
    x,y=train.batch(0,c['batch_size'],c['model']['sequence_length'],'cuda')
    with autocast(model,c['precision']):logits=model(x)
    loss=F.cross_entropy(logits.float().flatten(0,1),y.flatten());loss.backward()
    pairs=[]
    with torch.no_grad():
        for pair in model.pairs():
            u,d,root=regular_canonicalize(pair.up,pair.down,dtype=torch.float32)
            a=torch.stack(((root[:,None]*pair.up.grad.double()).float(),
                           (pair.down.grad.T.double()/root[:,None]).float()))*(1-c['beta'])
            pairs.append(dict(name=pair.name,u=u.detach(),d=d.detach(),a=a.detach(),
                              initial_lambda=None,initial_original_lambda=cs.multipliers(u.double(),d.double(),a.double()),
                              objective_rms=float(a.double().square().mean().sqrt())))
    assert all(torch.equal(p,b) for p,b in zip(model.parameters(),before))
    serial=[{k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in p.items()} for p in pairs]
    torch.save(serial,out/'step0-pairs.pt')
    manifest=dict(loss=float(loss.detach()),config=c,data=[train.metadata,val.metadata],
        model_parameters=sum(p.numel() for p in model.parameters()),
        initial_lambda_convention='None; internal z=0, original lambda=vertical beta; identical for both dtypes',
        pairs=[dict(name=p['name'],shape=list(p['u'].shape),objective_rms=p['objective_rms']) for p in pairs],
        corpus_sha256=hashlib.sha256((out/'step0-pairs.pt').read_bytes()).hexdigest())
    write(out/'corpus.json',manifest)
    return pairs


def cold(c,out):
    pairs=corpus(c,out)
    # Warm both full-size CUDA SVD paths and certification kernels. Excluded from timing.
    p=pairs[0]
    for dtype in (torch.float32,torch.float64):solve(p['u'],p['d'],p['a'],dtype,budget=1)
    rows=[]
    for p in pairs:
        u,d,a=p['u'],p['d'],p['a'];row=dict(name=p['name'],policies={},repeats={})
        r32,a32=solve(u,d,a,torch.float32);row['policies']['A_fp32']=a32
        r64,b64=solve(u,d,a,torch.float64);row['policies']['B_fp64']=b64
        assert r64.converged, 'fp64 cold failure: stop before training'
        for budget,label in ((0,'F_recovery64'),(1,'C_refine1'),(2,'D_refine2'),(4,'E_refine4')):
            r,rec=solve(u,d,a,torch.float64,lam=r32.lam,budget=budget)
            rec['total_hybrid_seconds']=a32['seconds']+rec['seconds']
            rec['total_hybrid_counts']={k:a32['counts'][k]+v for k,v in rec['counts'].items()}
            rec['total_hybrid_newton']=a32['iterations']+rec['iterations']
            row['policies'][label]=rec
        _,rec=solve(u,d,a,torch.float64,tol=3e-5)
        row['policies']['B_fp64_matched_3e5']=rec
        for dtype in (torch.float32,torch.float64):
            row['repeats'][str(dtype)]=[solve(u,d,a,dtype)[1] for _ in range(3)]
        # Separate instrumented runs; no profiler or extra SVDs in headline timings.
        with Instrument(trace=True) as instrument:
            traced,_=solve(u,d,a,torch.float32)
        assert torch.equal(traced.lam,r32.lam), 'instrumentation changed solution'
        row['fp32_line_trace']=instrument.lines
        row['isolation']=isolation(u,d,a,r32,r64,instrument.best['candidate'])
        with Instrument() as profile:
            _,profile_rec=solve(u,d,a,torch.float64)
        row['fp64_profile']=dict(times=dict(profile.times),devices=dict(profile.devices),record=profile_rec)
        rows.append(row);write(out/'cold.json',rows)
        print('PAIR',p['name'],json.dumps({k:(v['certified'],v['metrics']['normalized_gap'],v['seconds']) for k,v in row['policies'].items()}),flush=True)
    assert all(x['policies']['B_fp64']['certified'] for x in rows)


def mini(c,out,tolerance):
    model,train,val=build(c);opts=Optimizers(model,'qso',c)
    # Experimental injection ONLY at the pair-solver boundary. EMA/canonicalization
    # remain production fp32, params fp32, forward bf16; no production file edits.
    pair_names=[p.name for p in model.pairs()];rows=[];step=0;idx=0
    original=cs.solve_coupled
    def solver(u,d,a,*,config,initial_lambda=None):
        nonlocal idx
        assert u.dtype==d.dtype==a.dtype==torch.float32
        sc=replace(config,dtype=torch.float64,tolerance=tolerance,rcond_guard=1e-4,fallback=False)
        sync();start=time.perf_counter()
        r=original(u,d,a,config=sc,initial_lambda=initial_lambda);sync()
        rec=record(r,time.perf_counter()-start)
        rec.update(step=step,pair=pair_names[idx],warm_lambda=initial_lambda is not None,
                   objective_rms=float(a.double().square().mean().sqrt()))
        # Counterfactual same current inputs, cold lambda, NO update from this solve.
        cold_r,cold_rec=solve(u,d,a,torch.float64,tol=tolerance)
        rec['cold_counterfactual']=cold_rec
        rec['warm_cold_direction_relative']=float((r.pair-cold_r.pair).norm()/cold_r.pair.norm())
        rows.append(rec);write(out/'mini-solves.json',rows);idx+=1
        if not r.converged:
            torch.save(dict(u=u.cpu(),d=d.cpu(),a=a.cpu(),initial_lambda=initial_lambda.cpu(),
                            name=pair_names[idx-1],step=step),out/'failed-warm-pair.pt')
            with Instrument(trace=True) as traced:
                replay=original(u,d,a,config=sc,initial_lambda=initial_lambda)
            write(out/'failed-warm-trace.json',dict(record=record(replay,replay.seconds),
                                                   lines=traced.lines,devices=dict(traced.devices)))
            raise RuntimeError('fp64 diagnostic failed: '+r.reason)
        return r
    steps=[]
    with patch('qnormuon.optimizer.solve_coupled',solver):
        for step in range(8):
            idx=0;signal.alarm(180)
            stats=train_step(model,opts,train,c,step);signal.alarm(0)
            assert idx==6
            stats['measured_step_includes_cold_counterfactuals']=True
            stats['solver_seconds']=sum(r['seconds'] for r in rows if r['step']==step)
            steps.append(stats);write(out/'mini-steps.json',steps)
            print('STEP',step,'loss',stats['loss'],'solve_seconds',stats['solver_seconds'],
                  'newton',[r['iterations'] for r in rows if r['step']==step],flush=True)
    # Verify only the solver boundary used fp64, not the canonical first moments.
    states=[s for s in opts.paired.state.values() if s]
    write(out/'mini-summary.json',dict(completed_steps=len(steps),fallbacks=sum(r['metrics']['fallback_used'] for r in rows),
          state_dtypes=[{k:str(v.dtype) for k,v in s.items() if isinstance(v,torch.Tensor)} for s in states],
          peak_gpu_bytes=torch.cuda.max_memory_allocated(),config=c,
          experimental_override=f'pair solve only: fp64, tolerance {tolerance}, rcond guard 1e-4, fallback disabled'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['cold','mini']);parser.add_argument('--tolerance',type=float,default=1e-8);args=parser.parse_args()
    if args.tolerance not in (1e-8,3e-5):parser.error('study permits only strict 1e-8 or existing 3e-5 target')
    signal.signal(signal.SIGALRM,lambda *args: (_ for _ in ()).throw(TimeoutError('diagnostic budget')))
    c=json.loads(Path('configs/tiny_transformer/smoke.json').read_text())
    out=Path(c['output_root'])/f"precision-{os.environ['SLURM_JOB_ID']}-{args.phase}"
    out.mkdir(parents=True,exist_ok=False)
    write(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),job=os.environ['SLURM_JOB_ID'],hostname=platform.node(),
        output=str(out),phase=args.phase,mini_tolerance=args.tolerance,tf32=False,config=c,
        production_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           map(Path,['qnormuon/optimizer.py','qnormuon/coupled_solver.py'])}))
    print('OUTPUT',out,flush=True)
    if args.phase=='cold':cold(c,out)
    else:mini(c,out,args.tolerance)


if __name__=='__main__':main()
