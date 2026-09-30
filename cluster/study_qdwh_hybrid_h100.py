"""SLURM-only locked-pair study of QDWH inner/SVD-authoritative output."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM allocation required")

import sitecustomize
import torch
from benchmarks.tiny_transformer import (ModelConfig, Optimizers, TinyTransformer,
    datasets, load_checkpoint, seed_everything, train_step)
import qnormuon.optimizer as optimizer_module
from qnormuon import coupled_solver as cs
from experiments.smooth_qdwh_hybrid import solve_hybrid, safe
from cluster.study_smooth_qdwh_one_sided_h100 import audit_candidate


POLICIES = [(n, policy) for n in (2, 3) for policy in ("continue", "restart")]


def provenance():
    config=json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
    source=Path(config["output_root"])/f"smoke-28936-qso-seed{config['seed']}"
    for path in [source/"metrics.jsonl",source/"checkpoint.pt",
                 *[Path(config["data"][k]) for k in ("train_path","validation_path","tokenizer_path")]]:
        assert path.is_file(),path
    print("PROVENANCE",json.dumps(dict(python=sys.executable,version=sys.version,
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        config=config,source=str(source),job=os.environ["SLURM_JOB_ID"],
        policies=POLICIES,inner_evaluation_budget=7)),flush=True)
    return config,source


def summary(rows):
    stats={}
    for budget,policy in POLICIES:
        group=[r for r in rows if r["budget"]==budget and r["policy"]==policy]
        if not group:
            continue
        reasons=Counter(r["stats"]["fallback_reason"] for r in group if r["stats"]["fallback_reason"])
        totals={key:sum(r["stats"][key] for r in group) for key in
            ("qdwh_evaluations","final_svd_verifications","failed_final_verifications",
             "svd_rescue_solves","structural_rejects","total_svd_evaluations",
             "total_decomposition_evaluations","inner_line_trials","rescue_line_trials",
             "inner_seconds","verification_seconds","rescue_seconds","qdwh_decomposition_seconds")}
        totals.update(solves=len(group),certified=sum(r["converged"] for r in group),
            unsafe=sum(r["audit"]["unsafe"] for r in group),
            cpu_admm=sum(r["stats"]["cpu_admm_fallback"] for r in group),
            seconds=sum(r["seconds"] for r in group),
            direct_seconds=sum(r["direct_seconds"] for r in group),
            direct_svd=sum(r["direct_svd"] for r in group),
            fallbacks=dict(reasons),
            newton_bins={str(k):sum(r["newton"]==k for r in group) for k in (0,1,2)}|
                         {"3+":sum(r["newton"]>=3 for r in group)},
            max_gap=max(r["gap"] for r in group),min_rcond=min(r["rcond"] for r in group),
            max_direction_relative=max(r["direction_relative"] for r in group),
            warm_seconds=sum(r["seconds"] for r in group if r["step"]>=5),
            warm_direct_seconds=sum(r["direct_seconds"] for r in group if r["step"]>=5))
        stats[f"n{budget}_{policy}"]=totals
    return stats


def real_replay(steps):
    config,source=provenance()
    baseline=[json.loads(line) for line in (source/"metrics.jsonl").read_text().splitlines()]
    assert len(baseline)==50
    seed_everything(config["seed"],config["tf32"])
    train,val=datasets(config)
    metadata=dict(train=train.metadata,validation=val.metadata)
    rows=[];step_now=[0]
    original=optimizer_module.solve_coupled
    target=Path("cluster")/f"qdwh_hybrid_replay-{os.environ['SLURM_JOB_ID']}.jsonl"
    with target.open("w",buffering=1) as stream:
        def capture(u,d,a,**kwargs):
            torch.cuda.synchronize();start=time.perf_counter()
            direct=original(u,d,a,**kwargs)
            torch.cuda.synchronize();direct_seconds=time.perf_counter()-start
            assert direct.converged and not direct.fallback
            name=f"blocks.{(len(rows)//len(POLICIES))%6}.mlp"
            for budget,policy in POLICIES:
                torch.cuda.synchronize();start=time.perf_counter()
                got=solve_hybrid(u,d,a,inner_iterations=budget,rescue_policy=policy,**kwargs)
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                result=got.result
                audit=audit_candidate(u,d,a,result)
                row=dict(step=step_now[0],pair=name,budget=budget,policy=policy,
                    stats=got.stats,converged=result.converged,reason=result.reason,
                    newton=result.iterations,gap=result.metrics["normalized_gap"],
                    rcond=result.metrics["residual_rcond"],seconds=seconds,
                    direct_seconds=direct_seconds,direct_svd=direct.counts.svd_matrices//2,
                    direct_newton=direct.iterations,direct_line_trials=direct.counts.line_trials,
                    direction_relative=float((result.pair-direct.pair).norm()/direct.pair.norm()),
                    lambda_difference=float((result.lam-direct.lam).norm()),audit=audit)
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+"\n")
                assert result.converged and safe(result.metrics,kwargs["config"]),row
                assert not audit["unsafe"] and not result.fallback,row
            return direct
        for segment in (0,25):
            if segment>=steps:
                break
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
                    print("STEP",step,json.dumps({f"n{n}_{p}":round(sum(r["seconds"] for r in rows[-24:]
                        if r["budget"]==n and r["policy"]==p),4) for n,p in POLICIES}),flush=True)
    return rows,dict(source=str(source),checkpoint=str(source/"checkpoint.pt"),config=config)


def near_guard():
    gen=torch.Generator(device="cuda").manual_seed(9151)
    rows=[]
    for ratio in (1.01e-4,1.02e-4,1.05e-4,1.1e-4,1.2e-4,1.5e-4,2e-4,3e-4):
        m,n=48,16
        q,_=torch.linalg.qr(torch.randn(m,n,device="cuda",dtype=torch.float64,generator=gen))
        v,_=torch.linalg.qr(torch.randn(n,n,device="cuda",dtype=torch.float64,generator=gen))
        s=torch.linspace(1.,.2,n,device="cuda",dtype=torch.float64);s[-1]=ratio
        if ratio==1.5e-4:
            s[:2]=1.;s[-2:]=ratio
        b=(q*s)@v.T
        u=torch.randn(m,n,device="cuda",dtype=torch.float64,generator=gen)
        u=u/u.norm(dim=1)[:,None]
        d=2*((u*b).sum(1)/b.square().sum(1))[:,None]*b-u
        a=torch.stack((b,b.clone()))
        cfg=cs.SolverConfig(fallback=False,max_iterations=12)
        warm=torch.zeros(m,device="cuda",dtype=torch.float64)
        direct=cs.solve_coupled(u,d,a,config=cfg,initial_lambda=warm)
        for budget,policy in POLICIES:
            got=solve_hybrid(u,d,a,config=cfg,initial_lambda=warm,
                inner_iterations=budget,rescue_policy=policy)
            audit=audit_candidate(u,d,a,got.result)
            assert not audit["unsafe"]
            rows.append(dict(ratio=ratio,budget=budget,policy=policy,
                converged=got.result.converged,direct_converged=direct.converged,
                direct_newton=direct.iterations,stats=got.stats,reason=got.result.reason,audit=audit))
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=50)
    args=parser.parse_args()
    assert 1<=args.steps<=50
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert "H100" in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.linalg.svd(torch.randn(24,8,device="cuda",dtype=torch.float64),full_matrices=False)
    real,provenance_data=real_replay(args.steps)
    stress=near_guard()
    output=dict(job=os.environ["SLURM_JOB_ID"],steps=args.steps,real=real,
                near_guard=stress,provenance=provenance_data,summary=summary(real))
    path=Path("cluster")/f"qdwh_hybrid_study-{os.environ['SLURM_JOB_ID']}.json"
    path.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print("RESULT",json.dumps(dict(path=str(path),summary=output["summary"],near_guard=stress)),flush=True)
