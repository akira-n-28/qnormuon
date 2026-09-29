"""H100 diagnostic-only Gram/SVD study; never changes production decisions."""
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM allocation required")

import sitecustomize
import torch
import qnormuon.coupled_solver as solver
import qnormuon.optimizer as optimizer_module
from experiments.smooth_gram import diagnose
from benchmarks.tiny_transformer import (ModelConfig, Optimizers, TinyTransformer,
    datasets, load_checkpoint, seed_everything, train_step)


def timed(fn, repeats=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    values=[]
    for _ in range(repeats):
        torch.cuda.synchronize(); start=time.perf_counter()
        fn(); torch.cuda.synchronize()
        values.append(time.perf_counter()-start)
    return dict(median=statistics.median(values), p95=sorted(values)[math.ceil(.95*len(values))-1], values=values)


def make_cases():
    cases=[]
    gen=torch.Generator(device="cuda").manual_seed(60831)
    ratios=(1.,1e-1,1e-2,1e-3,3e-4,2e-4,1.5e-4,1.2e-4,
            1.05e-4,1.01e-4,1e-4,.99e-4,5e-5)
    for m,n in ((24,8),(96,24),(1024,384)):
        q,_=torch.linalg.qr(torch.randn(m,n,device="cuda",dtype=torch.float64,generator=gen))
        v,_=torch.linalg.qr(torch.randn(n,n,device="cuda",dtype=torch.float64,generator=gen))
        for ratio in ratios:
            spectrum=torch.linspace(1.,.1,n,device="cuda",dtype=torch.float64)
            spectrum[-1]=ratio
            if ratio==1.:
                spectrum.fill_(1.)
            cases.append((f"{m}x{n}/rcond={ratio:g}", (q*spectrum)@v.T))
        spectra={
            "repeated_top":torch.cat((torch.ones(3,device="cuda",dtype=torch.float64),
                                  torch.linspace(.8,.01,n-3,device="cuda",dtype=torch.float64))),
            "clustered_top":torch.linspace(1.,1.-1e-12,n,device="cuda",dtype=torch.float64),
            "repeated_bottom":torch.cat((torch.linspace(1.,.1,n-3,device="cuda",dtype=torch.float64),
                                     torch.full((3,),1.1e-4,device="cuda",dtype=torch.float64))),
            "nearly_rank_one":torch.cat((torch.ones(1,device="cuda",dtype=torch.float64),
                                    torch.full((n-1,),1e-8,device="cuda",dtype=torch.float64))),
            "wide_dynamic":torch.logspace(0,-10,n,device="cuda",dtype=torch.float64),
        }
        for name,spectrum in spectra.items():
            cases.append((f"{m}x{n}/{name}",(q*spectrum)@v.T))
        if m==24:
            base=(q*torch.linspace(1.,.01,n,device="cuda",dtype=torch.float64))@v.T
            for scale in (1e-150,1e150):
                cases.append((f"24x8/scale={scale:g}",base*scale))
        cases.append((f"{m}x{n}/gaussian",torch.randn(m,n,device="cuda",dtype=torch.float64,generator=gen)))
    return cases


def perturbations(b, ref):
    gen=torch.Generator(device="cuda").manual_seed(776)
    u,s,vt=ref
    random=torch.randn(b.shape,device="cuda",dtype=torch.float64,generator=gen)
    small=torch.outer(u[:,-1],vt[-1])
    large=torch.outer(u[:,0],vt[0])
    tangent=u@torch.randn((s.numel(),s.numel()),device="cuda",dtype=torch.float64,generator=gen)@vt
    normal=random-u@(u.T@random)
    clustered=torch.outer(u[:,0],vt[1])+torch.outer(u[:,1],vt[0])
    return dict(random=random,tangent=tangent,normal=normal,smallest=small,
                largest=large,clustered=clustered)


def compare(name,b):
    # Normalize only for relative comparisons; the Gram diagnostic still sees
    # the original scale, including extreme representable amplitudes.
    result=diagnose(b)
    amplitude=float(b.abs().max())
    x=b/amplitude
    ref=torch.linalg.svd(x,full_matrices=False)
    u,s,vt=ref
    p=u@vt
    rec=dict(name=name,shape=list(b.shape),posterior=result.measurements,
             gram_accepted=result.accepted,reason=result.reason,
             svd_rcond=float(s[-1]/s[0]))
    if not result.accepted:
        return rec
    gs=result.singular/amplitude
    gp=result.polar
    rec.update(singular_max_relative=float(((gs-s).abs()/s.clamp_min(torch.finfo(s.dtype).tiny)).max()),
        nuclear_relative=abs(float(gs.sum()-s.sum()))/float(s.sum()),
        rcond_error=abs(result.measurements["rcond_estimate"]-rec["svd_rcond"]),
        polar_relative=float((gp-p).norm()/p.norm()),
        rcond_safe=(result.measurements["rcond_lower"] <= rec["svd_rcond"]+1e-15),
        guard_safe=not (result.measurements["rcond_lower"]>1e-4 and rec["svd_rcond"]<=1e-4))
    deriv=solver.SmoothDual(None,None,None)
    gr=SimpleNamespace(left=result.left,singular=result.singular/amplitude,right=result.right)
    sv=SimpleNamespace(left=u,singular=s,right=vt)
    hvps={}
    for label,e in perturbations(x,ref).items():
        de=deriv.polar_derivative(gr,e)
        oracle=deriv.polar_derivative(sv,e)
        scale=max(float(oracle.norm()),torch.finfo(torch.float64).tiny)
        eps=1e-6*max(float(s[-1]),1e-8)
        fd=(torch.linalg.svd(x+eps*e,full_matrices=False),
            torch.linalg.svd(x-eps*e,full_matrices=False)) if b.shape[0]<=96 else None
        fd_error=None
        if fd is not None:
            pplus=fd[0][0]@fd[0][2];pminus=fd[1][0]@fd[1][2]
            fd_error=float(((pplus-pminus)/(2*eps)-oracle).norm())/scale
        hvps[label]=dict(relative=float((de-oracle).norm())/scale,
                         absolute=float((de-oracle).norm()),fd_reference_relative=fd_error)
    rec["hvp"]=hvps
    return rec


def real_inputs():
    config=json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
    source=Path(config["output_root"])/f"smoke-28936-qso-seed{config['seed']}"
    checkpoint=source/"checkpoint.pt"
    assert checkpoint.is_file(),checkpoint
    seed_everything(config["seed"],config["tf32"])
    train,val=datasets(config)
    metadata=dict(train=train.metadata,validation=val.metadata)
    model=TinyTransformer(ModelConfig(**config["model"])).float().cuda()
    opts=Optimizers(model,"qso",config)
    assert load_checkpoint(checkpoint,model,opts,config,metadata)==25
    names=[p.name for p in opts.paired.pairs]
    rows=[]; residuals=[]
    original=optimizer_module.solve_coupled
    old_eval=solver.SmoothDual.evaluate
    def capture(u,d,a,**kwargs):
        rows.append(dict(name=names[len(rows)],u=u.detach().clone(),d=d.detach().clone(),
                         a=a.detach().clone(),initial_lambda=None if kwargs.get("initial_lambda") is None
                         else kwargs["initial_lambda"].detach().clone()))
        return original(u,d,a,**kwargs)
    def observe(self,lam):
        residuals.append((self.a-solver.adjoint(self.u,self.d,lam)).detach().clone())
        return old_eval(self,lam)
    with patch.object(optimizer_module,"solve_coupled",capture),patch.object(solver.SmoothDual,"evaluate",observe):
        record=train_step(model,opts,train,config,25)
    baseline=json.loads((source/"metrics.jsonl").read_text().splitlines()[25])
    assert record["batch_sha256"]==baseline["batch_sha256"] and len(rows)==6
    assert record["loss"]==baseline["loss"]
    return rows,residuals,dict(checkpoint=str(checkpoint),batch_sha256=record["batch_sha256"])


def adaptive_replay(rows):
    base=[];candidate=[];events=[];base_traces=[];gram_traces=[]
    original=solver.SmoothDual
    active_trace=[None]
    class Observed(original):
        def evaluate(self,lam):
            ev=super().evaluate(lam)
            active_trace[0].append((lam.detach().clone(),ev.value,
                                    ev.gradient.detach().clone(),ev.rcond))
            return ev
    class Adaptive(original):
        def evaluate(self,lam):
            b=self.a-solver.adjoint(self.u,self.d,lam)
            attempts=[diagnose(x) for x in b]
            if not all(x.accepted for x in attempts):
                events.append(dict(backend="svd_fallback",reasons=[x.reason for x in attempts]))
                ev=super().evaluate(lam)
                active_trace[0].append((lam.detach().clone(),ev.value,
                                        ev.gradient.detach().clone(),ev.rcond))
                return ev
            left=torch.stack([x.left for x in attempts])
            singular=torch.stack([x.singular for x in attempts])
            right=torch.stack([x.right for x in attempts])
            polar=torch.stack([x.polar for x in attempts])
            ratio=singular[:,-1]/singular[:,0]
            self.counts.polar_matrices+=2
            events.append(dict(backend="gram",reasons=[]))
            ev=solver.Evaluation(float(singular.sum()),
                -solver.horizontal_residual(self.u,self.d,polar),polar,left,singular,right,
                float(ratio.min()),lam)
            active_trace[0].append((lam.detach().clone(),ev.value,
                                    ev.gradient.detach().clone(),ev.rcond))
            return ev
    for row in rows:
        config=solver.SolverConfig(fallback=False)
        base_trace=[];active_trace[0]=base_trace
        with patch.object(solver,"SmoothDual",Observed):
            x=solver.solve_coupled(row["u"],row["d"],row["a"],config=config,
                                   initial_lambda=row["initial_lambda"])
        gram_trace=[];active_trace[0]=gram_trace
        with patch.object(solver,"SmoothDual",Adaptive):
            y=solver.solve_coupled(row["u"],row["d"],row["a"],config=config,
                                   initial_lambda=row["initial_lambda"])
        base.append(x);candidate.append(y);base_traces.append(base_trace);gram_traces.append(gram_trace)
    comparisons=[]
    for row,x,y,bt,gt in zip(rows,base,candidate,base_traces,gram_traces):
        same_trace_length=len(bt)==len(gt)
        matched=list(zip(bt,gt))
        comparisons.append(dict(pair=row["name"],svd_converged=x.converged,
            gram_converged=y.converged,svd_iterations=x.iterations,gram_iterations=y.iterations,
            svd_reason=x.reason,gram_reason=y.reason,
            evaluation_counts=[len(bt),len(gt)],same_trace_length=same_trace_length,
            trial_coordinate_max_abs=max(float((a[0]-b[0]).abs().max()) for a,b in matched),
            trial_value_max_abs=max(abs(a[1]-b[1]) for a,b in matched),
            trial_gradient_max_abs=max(float((a[2]-b[2]).abs().max()) for a,b in matched),
            trial_rcond_max_abs=max(abs(a[3]-b[3]) for a,b in matched),
            lambda_max_abs=float((x.lam-y.lam).abs().max()),
            direction_relative=float((x.pair-y.pair).norm()/x.pair.norm()),
            objective_abs=abs(x.metrics["dual_objective"]-y.metrics["dual_objective"]),
            gap_abs=abs(x.metrics["normalized_gap"]-y.metrics["normalized_gap"]),
            rcond_abs=abs(x.metrics["residual_rcond"]-y.metrics["residual_rcond"]),
            certificate_decisions=[solver.accepted(h,3e-5) for h in x.history],
            gram_certificate_decisions=[solver.accepted(h,3e-5) for h in y.history],
            line_trials_svd=x.counts.line_trials,line_trials_gram=y.counts.line_trials))
    return dict(comparisons=comparisons,events=events)


def adversarial():
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert "H100" in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.randn(8,8,device="cuda",dtype=torch.float64).T.contiguous()
    torch.linalg.svd(torch.randn(24,8,device="cuda",dtype=torch.float64),full_matrices=False)
    generated=make_cases()
    rows=[]
    for name,b in generated:
        rows.append(compare(name,b))
    real, residuals, provenance=real_inputs()
    for index,b in enumerate(residuals):
        for side in range(2):
            rows.append(compare(f"real_step25_eval{index}_side{side}",b[side]))
    replay=adaptive_replay(real)
    representative=residuals[-1]
    perf=dict(svd=timed(lambda:torch.linalg.svd(representative,full_matrices=False)),
              raw_gram=timed(lambda:torch.linalg.eigh(representative.transpose(-2,-1)@representative)),
              validated=timed(lambda:[diagnose(x) for x in representative]))
    output=dict(job=os.environ["SLURM_JOB_ID"],torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        provenance=provenance,cases=rows,replay=replay,performance=perf)
    target=Path("cluster")/f"smooth_gram_adversarial-{os.environ['SLURM_JOB_ID']}.json"
    target.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    reasons={r:sum(item["reason"]==r for item in rows) for r in sorted({item["reason"] for item in rows})}
    print("RESULT",json.dumps(dict(path=str(target),cases=len(rows),accepted=sum(x["gram_accepted"] for x in rows),
        reasons=reasons,guard_violations=sum(not x.get("guard_safe",True) for x in rows),
        newton_mismatches=sum(x["svd_iterations"]!=x["gram_iterations"] for x in replay["comparisons"]),
        performance=perf),allow_nan=False),flush=True)


def shadow():
    """Replay 50 locked steps with SVD decisions; Gram only observes."""
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert "H100" in torch.cuda.get_device_name(0)
    config=json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
    source=Path(config["output_root"])/f"smoke-28936-qso-seed{config['seed']}"
    baseline=[json.loads(line) for line in (source/"metrics.jsonl").read_text().splitlines()]
    assert len(baseline)==50 and (source/"checkpoint.pt").is_file()
    seed_everything(config["seed"],config["tf32"])
    train,val=datasets(config)
    metadata=dict(train=train.metadata,validation=val.metadata)
    all_rows=[]; step_now=[0]
    old_evaluate=solver.SmoothDual.evaluate
    def observe(self,lam):
        torch.cuda.synchronize(); start=time.perf_counter()
        ev=old_evaluate(self,lam)
        torch.cuda.synchronize(); svd_seconds=time.perf_counter()-start
        b=self.a-solver.adjoint(self.u,self.d,lam)
        torch.cuda.synchronize(); start=time.perf_counter()
        gs=[diagnose(x) for x in b]
        torch.cuda.synchronize(); gram_seconds=time.perf_counter()-start
        row=dict(step=step_now[0],svd_seconds=svd_seconds,gram_seconds=gram_seconds,
                 accepted=all(x.accepted for x in gs),reasons=[x.reason for x in gs],
                 rcond_svd=ev.rcond,
                 rcond_lower=min(x.measurements.get("rcond_lower",0.) for x in gs))
        if row["accepted"]:
            p=torch.stack([x.polar for x in gs])
            singular=torch.stack([x.singular for x in gs])
            row.update(value_abs=abs(float(singular.sum())-ev.value),
                polar_relative=float((p-ev.pair).norm()/ev.pair.norm()),
                gradient_relative=float((-solver.horizontal_residual(self.u,self.d,p)-ev.gradient).norm()
                    /ev.gradient.norm().clamp_min(torch.finfo(torch.float64).tiny)),
                rcond_abs=abs(float((singular[:,-1]/singular[:,0]).min())-ev.rcond))
            v=torch.linspace(-1.,1.,self.u.shape[0],dtype=lam.dtype,device=lam.device)
            ge=solver.Evaluation(float(singular.sum()),-solver.horizontal_residual(self.u,self.d,p),p,
                torch.stack([x.left for x in gs]),singular,torch.stack([x.right for x in gs]),
                float((singular[:,-1]/singular[:,0]).min()),lam)
            perturbation=-solver.adjoint(self.u,self.d,v)
            hv_g=-solver.horizontal_residual(self.u,self.d,self.polar_derivative(ge,perturbation))
            hv_s=-solver.horizontal_residual(self.u,self.d,self.polar_derivative(ev,perturbation))
            row["hvp_relative"]=float((hv_g-hv_s).norm()/hv_s.norm().clamp_min(torch.finfo(torch.float64).tiny))
        all_rows.append(row)
        return ev
    for segment in (0,25):
        seed_everything(config["seed"],config["tf32"])
        model=TinyTransformer(ModelConfig(**config["model"])).float().cuda()
        opts=Optimizers(model,"qso",config)
        if segment==25:
            assert load_checkpoint(source/"checkpoint.pt",model,opts,config,metadata)==25
        with patch.object(solver.SmoothDual,"evaluate",observe):
            for step in range(segment,segment+25):
                step_now[0]=step
                record=train_step(model,opts,train,config,step)
                assert record["batch_sha256"]==baseline[step]["batch_sha256"]
                assert record["loss"]==baseline[step]["loss"]
    path=Path("cluster")/f"smooth_gram_shadow-{os.environ['SLURM_JOB_ID']}.json"
    output=dict(job=os.environ["SLURM_JOB_ID"],source=str(source),
                gpu=torch.cuda.get_device_name(0),rows=all_rows)
    path.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    accepted=[x for x in all_rows if x["accepted"]]
    reasons={r:sum(r in x["reasons"] for x in all_rows) for r in sorted({r for x in all_rows for r in x["reasons"] if r!="posterior_passed_research_only"})}
    print("SHADOW_RESULT",json.dumps(dict(path=str(path),evaluations=len(all_rows),
        accepted=len(accepted),fallback_reasons=reasons,
        min_rcond_lower=min(x["rcond_lower"] for x in accepted) if accepted else None,
        max_polar_relative=max(x["polar_relative"] for x in accepted) if accepted else None,
        max_gradient_relative=max(x["gradient_relative"] for x in accepted) if accepted else None,
        max_hvp_relative=max(x["hvp_relative"] for x in accepted) if accepted else None,
        total_svd_seconds=sum(x["svd_seconds"] for x in all_rows),
        total_gram_seconds=sum(x["gram_seconds"] for x in all_rows))),flush=True)


if __name__=="__main__":
    if sys.argv[1:]==["adversarial"]:
        adversarial()
    elif sys.argv[1:]==["shadow"]:
        shadow()
    else:
        raise SystemExit("Usage: study_smooth_gram_h100.py adversarial|shadow")
