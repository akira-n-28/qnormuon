"""SLURM-only controlled-pair QDWH replay on the locked SVD trajectory."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM allocation required")

import sitecustomize
import torch
import qnormuon.coupled_solver as cs
import qnormuon.optimizer as optimizer_module
from benchmarks.tiny_transformer import (ModelConfig, Optimizers, TinyTransformer,
    datasets, load_checkpoint, seed_everything, train_step)
from experiments.smooth_qdwh_one_sided import solve_one_sided


def audit_candidate(u,d,a,result):
    if not result.converged:
        return dict(unsafe=False,certified=False)
    pair,metrics=cs.certificate(u.double(),d.double(),a.double(),result.lam,
        result.pair,cs.Counts(),primal_norm_backend="svd")
    ok=(cs.accepted(metrics,3e-5) and metrics["residual_rcond"]>1e-4)
    return dict(unsafe=not ok,certified=ok,independent_gap=metrics["normalized_gap"],
        independent_rcond=metrics["residual_rcond"],
        independent_horizontal=metrics["normalized_horizontal_residual"],
        independent_spectral_excess=metrics["spectral_excess"],
        direction_recovery_difference=float((pair-result.pair).norm()))


def audit_actions(u,d,a,events):
    """Offline SVD oracle after the QDWH solver has already returned."""
    if not any(events.values()):return {}
    beta=cs.multipliers(u,d,a)
    centered=a-cs.adjoint(u,d,beta)
    magnitude=cs._stable_norm(centered)
    coord=(u.square()+d.square()).sum(1).rsqrt()
    oracle=cs.SmoothDual(u*coord[:,None],d*coord[:,None],centered/magnitude)
    evaluations={}
    def at(z):
        key=id(z)
        if key not in evaluations:evaluations[key]=oracle.evaluate(z)
        return evaluations[key]
    results=Counter()
    for z,v,h,bound in events["hvp"]:
        exact=oracle.hvp(at(z),v)
        results["hvp_queries"]+=1
        if float((exact-h).norm())>bound+1e-12:results["unsafe_hvp_bound"]+=1
    for z,v,_,_ in events["curvature"]:
        exact=oracle.hvp(at(z),v)
        results["curvature_actions"]+=1
        if float(v@exact)<=0:results["unsafe_curvature"]+=1
    for z,p in events["descent"]:
        results["descent_actions"]+=1
        if float(at(z).gradient@p)>=0:results["unsafe_descent"]+=1
    for z,tz,p,alpha in events["armijo"]:
        current,trial=at(z),at(tz)
        slope=float(current.gradient@p)
        rounding=2*torch.finfo(torch.float64).eps*max(1.,abs(current.value))
        resolved=abs(alpha*slope)>rounding
        improved=float(trial.gradient.norm())<float(current.gradient.norm())
        okay=(trial.rcond>1e-4 and
              trial.value<=current.value+1e-4*alpha*slope+rounding and
              (resolved or improved))
        results["armijo_actions"]+=1
        if not okay:results["unsafe_armijo"]+=1
    results["oracle_svd_evaluations"] = len(evaluations)
    return dict(results)


def real_replay(steps,profile,oracle_actions):
    config=json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
    source=Path(config["output_root"])/f"smoke-28936-qso-seed{config['seed']}"
    baseline=[json.loads(line) for line in (source/"metrics.jsonl").read_text().splitlines()]
    assert len(baseline)==50
    seed_everything(config["seed"],config["tf32"])
    train,val=datasets(config)
    metadata=dict(train=train.metadata,validation=val.metadata)
    rows=[];step_now=[0]
    original=optimizer_module.solve_coupled
    def capture(u,d,a,**kwargs):
        torch.cuda.synchronize();start=time.perf_counter()
        cfg=kwargs.get("config") or cs.SolverConfig()
        candidate=solve_one_sided(u,d,a,config=cfg,
            initial_lambda=kwargs.get("initial_lambda"),profile=profile,
            oracle_audit=oracle_actions)
        torch.cuda.synchronize();candidate_seconds=time.perf_counter()-start
        audit=audit_candidate(u,d,a,candidate)
        action_audit=audit_actions(u.double(),d.double(),a.double(),
            candidate.audit_events) if oracle_actions else {}
        torch.cuda.synchronize();start=time.perf_counter()
        oracle=original(u,d,a,**kwargs)
        torch.cuda.synchronize();oracle_seconds=time.perf_counter()-start
        rows.append(dict(step=step_now[0],pair=f"pair{len(rows)%6}",
            qdwh_certified=candidate.converged,reason=candidate.reason,
            newton=candidate.iterations,qdwh_evaluations=candidate.qdwh_evaluations,
            svd_evaluations=candidate.svd_evaluations,line_trials=candidate.line_trials,
            hvps=candidate.hvps,fallbacks=candidate.fallbacks,
            actions=candidate.actions,
            action_audit=action_audit,
            qdwh_seconds=candidate_seconds,oracle_seconds=oracle_seconds,
            timing=candidate.timing,history=candidate.history,
            normalized_gap=candidate.metrics.get("normalized_gap") if candidate.metrics else None,
            rcond=candidate.metrics.get("residual_rcond") if candidate.metrics else None,
            oracle_newton=oracle.iterations,oracle_svd=oracle.counts.svd_matrices//2,
            lambda_difference=float((candidate.lam-oracle.lam).norm()) if candidate.lam is not None else None,
            direction_difference=float((candidate.pair-oracle.pair).norm()) if candidate.pair is not None else None,
            **audit))
        if len(rows)%6==0:
            print("STEP",step_now[0],"qdwh_certified",sum(r["qdwh_certified"] for r in rows[-6:]),
                "qdwh_seconds",round(sum(r["qdwh_seconds"] for r in rows[-6:]),3),
                "svd_fallbacks",sum(r["svd_evaluations"] for r in rows[-6:]),flush=True)
        return oracle
    for segment in (0,25):
        if segment>=steps:break
        seed_everything(config["seed"],config["tf32"])
        model=TinyTransformer(ModelConfig(**config["model"])).float().cuda()
        opts=Optimizers(model,"qso",config)
        if segment==25:
            assert load_checkpoint(source/"checkpoint.pt",model,opts,config,metadata)==25
        with patch.object(optimizer_module,"solve_coupled",capture):
            for step in range(segment,min(segment+25,steps)):
                step_now[0]=step
                rec=train_step(model,opts,train,config,step)
                assert rec["batch_sha256"]==baseline[step]["batch_sha256"]
                assert rec["loss"]==baseline[step]["loss"]
    return rows


def near_guard():
    gen=torch.Generator(device="cuda").manual_seed(9151)
    rows=[]
    for ratio in (1.01e-4,1.02e-4,1.05e-4,1.1e-4,1.2e-4,1.5e-4,2e-4,3e-4):
        m,n=48,16
        q,_=torch.linalg.qr(torch.randn(m,n,device="cuda",dtype=torch.float64,generator=gen))
        v,_=torch.linalg.qr(torch.randn(n,n,device="cuda",dtype=torch.float64,generator=gen))
        s=torch.linspace(1.,.2,n,device="cuda",dtype=torch.float64);s[-1]=ratio
        if ratio==1.5e-4:s[:2]=1.;s[-2:]=ratio
        b=(q*s)@v.T
        u=torch.randn(m,n,device="cuda",dtype=torch.float64,generator=gen)
        u=u/u.norm(dim=1)[:,None]
        projection=(u*b).sum(1)/b.square().sum(1)
        d=2*projection[:,None]*b-u
        a=torch.stack((b,b.clone()))
        cfg=cs.SolverConfig(fallback=False,max_iterations=12)
        result=solve_one_sided(u,d,a,config=cfg,initial_lambda=torch.zeros(m,device="cuda",dtype=torch.float64))
        audit=audit_candidate(u,d,a,result)
        oracle=cs.solve_coupled(u,d,a,config=cfg,initial_lambda=torch.zeros(m,device="cuda",dtype=torch.float64))
        rows.append(dict(ratio=ratio,converged=result.converged,reason=result.reason,
            qdwh_evaluations=result.qdwh_evaluations,svd_evaluations=result.svd_evaluations,
            newton=result.iterations,line_trials=result.line_trials,
            fallbacks=result.fallbacks,oracle_converged=oracle.converged,
            oracle_newton=oracle.iterations,**audit))
    return rows


def summarize(rows):
    reasons=Counter();actions=Counter();action_audit=Counter();times=defaultdict(float)
    for row in rows:
        reasons.update(row["fallbacks"])
        actions.update(row["actions"])
        action_audit.update(row["action_audit"])
        for k,v in row["timing"].items():times[k]+=v
    return dict(solves=len(rows),certified=sum(r["qdwh_certified"] for r in rows),
        unsafe=sum(r["unsafe"] for r in rows),
        qdwh_evaluations=sum(r["qdwh_evaluations"] for r in rows),
        svd_evaluations=sum(r["svd_evaluations"] for r in rows),
        oracle_svd=sum(r["oracle_svd"] for r in rows),
        line_trials=sum(r["line_trials"] for r in rows),
        qdwh_seconds=sum(r["qdwh_seconds"] for r in rows),
        oracle_seconds=sum(r["oracle_seconds"] for r in rows),
        fallbacks=dict(reasons),actions=dict(actions),
        action_audit=dict(action_audit),timing=dict(times),
        newton_bins={str(k):sum(r["newton"]==k for r in rows) for k in (0,1,2)}|
                    {"3+":sum(r["newton"]>=3 for r in rows)})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=50)
    parser.add_argument("--profile",action="store_true")
    parser.add_argument("--oracle-actions",action="store_true")
    args=parser.parse_args()
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert "H100" in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.linalg.svd(torch.randn(24,8,device="cuda",dtype=torch.float64),full_matrices=False)
    started=time.perf_counter()
    real=real_replay(args.steps,args.profile,args.oracle_actions)
    stress=near_guard()
    result=dict(job=os.environ["SLURM_JOB_ID"],torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),steps=args.steps,real=real,near_guard=stress,
        summary=summarize(real),elapsed_seconds=time.perf_counter()-started)
    path=Path("cluster")/f"smooth_qdwh_one_sided-{os.environ['SLURM_JOB_ID']}.json"
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("RESULT",json.dumps(dict(path=str(path),summary=result["summary"],
        near_guard=stress,elapsed_seconds=result["elapsed_seconds"])),flush=True)
