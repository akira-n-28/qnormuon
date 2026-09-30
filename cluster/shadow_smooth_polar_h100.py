"""Locked 50-step shadow: production full SVD controls every decision."""
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
import qnormuon.coupled_solver as solver
from benchmarks.tiny_transformer import (ModelConfig, Optimizers, TinyTransformer,
    datasets, load_checkpoint, seed_everything, train_step)
from experiments.smooth_polar_alternative import diagnose, polar_derivative


def main(route):
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert "H100" in torch.cuda.get_device_name(0)
    config=json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
    source=Path(config["output_root"])/f"smoke-28936-qso-seed{config['seed']}"
    baseline=[json.loads(line) for line in (source/"metrics.jsonl").read_text().splitlines()]
    assert len(baseline)==50
    seed_everything(config["seed"],config["tf32"])
    train,val=datasets(config)
    metadata=dict(train=train.metadata,validation=val.metadata)
    rows=[];step_now=[0]
    original=solver.SmoothDual.evaluate
    def observe(self,lam):
        torch.cuda.synchronize();start=time.perf_counter()
        ev=original(self,lam)
        torch.cuda.synchronize();svd_seconds=time.perf_counter()-start
        b=self.a-solver.adjoint(self.u,self.d,lam)
        torch.cuda.synchronize();start=time.perf_counter()
        parts=[diagnose(side,route=route) for side in b]
        torch.cuda.synchronize();candidate_seconds=time.perf_counter()-start
        row=dict(step=step_now[0],svd_seconds=svd_seconds,
                 candidate_seconds=candidate_seconds,
                 accepted=all(z.accepted for z in parts),
                 reasons=[z.reason for z in parts],
                 iterations=[z.iterations for z in parts])
        if row["accepted"]:
            p=torch.stack([z.polar for z in parts])
            s=torch.stack([z.singular for z in parts])
            row.update(polar_relative=float((p-ev.pair).norm()/ev.pair.norm()),
                gradient_relative=float((-solver.horizontal_residual(self.u,self.d,p)-ev.gradient).norm()
                    /ev.gradient.norm().clamp_min(1e-300)),
                value_relative=abs(float(s.sum())-ev.value)/max(abs(ev.value),1e-300),
                rcond_abs=abs(float((s[:,-1]/s[:,0]).min())-ev.rcond))
            v=torch.linspace(-1.,1.,self.u.shape[0],dtype=lam.dtype,device=lam.device)
            e=-solver.adjoint(self.u,self.d,v)
            got=torch.stack([polar_derivative(z,side) for z,side in zip(parts,e)])
            ref=self.polar_derivative(ev,e)
            row["hvp_relative"]=float((got-ref).norm()/ref.norm().clamp_min(1e-300))
        rows.append(row)
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
                rec=train_step(model,opts,train,config,step)
                assert rec["batch_sha256"]==baseline[step]["batch_sha256"]
                assert rec["loss"]==baseline[step]["loss"]
    target=Path("cluster")/f"smooth_polar_shadow-{os.environ['SLURM_JOB_ID']}.json"
    target.write_text(json.dumps(dict(job=os.environ["SLURM_JOB_ID"],route=route,
                                      source=str(source),rows=rows),indent=2,allow_nan=False)+"\n")
    accepted=[r for r in rows if r["accepted"]]
    print("SHADOW_RESULT",json.dumps(dict(path=str(target),evaluations=len(rows),
        accepted=len(accepted),max_polar=max((r["polar_relative"] for r in accepted),default=None),
        max_gradient=max((r["gradient_relative"] for r in accepted),default=None),
        max_hvp=max((r["hvp_relative"] for r in accepted),default=None),
        svd_seconds=sum(r["svd_seconds"] for r in rows),
        candidate_seconds=sum(r["candidate_seconds"] for r in rows))),flush=True)


if __name__=="__main__":
    if sys.argv[1:] not in (["tall"],["qr_square"]):
        raise SystemExit("Usage: shadow_smooth_polar_h100.py tall|qr_square")
    main(sys.argv[1])
