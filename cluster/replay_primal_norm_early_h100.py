"""Replay locked smoke steps 0-24 and compare every SVD/Gram certificate."""
import json
import os
from pathlib import Path
from unittest.mock import patch

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM allocation required")

import sitecustomize
import torch
import qnormuon.coupled_solver as solver
import qnormuon.optimizer as optimizer_module
from benchmarks.tiny_transformer import ModelConfig, Optimizers, TinyTransformer, datasets, seed_everything, train_step
from cluster.study_primal_norm_h100 import certificate_variant, gram_upper


def main():
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert "H100" in torch.cuda.get_device_name(0)
    config=json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
    source=Path(config["output_root"])/f"smoke-28922-qso-seed{config['seed']}"
    baseline=[json.loads(line) for line in (source/"metrics.jsonl").read_text().splitlines()]
    seed_everything(config["seed"],config["tf32"])
    model=TinyTransformer(ModelConfig(**config["model"])).float().cuda()
    opts=Optimizers(model,"qso",config)
    train,_=datasets(config)
    names=[p.name for p in opts.paired.pairs]
    active=[None];pair_count=[0];step_now=[0];rows=[]
    original_solve=optimizer_module.solve_coupled
    original_cert=solver.certificate

    def observed_solve(*args,**kwargs):
        active[0]=names[pair_count[0]%6]
        pair_count[0]+=1
        return original_solve(*args,**kwargs)

    def observed_cert(u,d,a,lam,candidate,counts,**kwargs):
        p0,m0=original_cert(u,d,a,lam,candidate,counts,**kwargs)
        p1,m1,scale=certificate_variant(u,d,a,lam,candidate,cached_dual=kwargs.get("cached_dual"),norm=gram_upper)
        rows.append(dict(step=step_now[0],pair=active[0],
                         accepted_svd=solver.accepted(m0,3e-5),
                         accepted_gram=solver.accepted(m1,3e-5),
                         svd_gap=m0["normalized_gap"],gram_gap=m1["normalized_gap"],
                         scale=scale, direction_error=float((p0-p1).abs().max()),
                         dual_error=abs(m0["dual_objective"]-m1["dual_objective"]),
                         primal_error=abs(m0["primal_objective"]-m1["primal_objective"])))
        return p0,m0

    with patch.object(optimizer_module,"solve_coupled",observed_solve),patch.object(solver,"certificate",observed_cert):
        for step in range(25):
            step_now[0]=step
            record=train_step(model,opts,train,config,step)
            assert record["batch_sha256"]==baseline[step]["batch_sha256"]
            assert record["loss"]==baseline[step]["loss"]
    assert pair_count[0]==150 and len(rows)>=450
    accepted_rows=[r for r in rows if r["accepted_svd"]]
    mismatches=[r for r in rows if r["accepted_svd"]!=r["accepted_gram"]]
    assert not mismatches, mismatches[:4]
    assert len(accepted_rows)==150
    path=Path("cluster")/f"primal_norm_early_replay-{os.environ['SLURM_JOB_ID']}.json"
    result=dict(job_id=os.environ["SLURM_JOB_ID"],torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                certificate_evaluations=len(rows),accepted_pairs=len(accepted_rows),
                max_accepted_svd_gap=max(r["svd_gap"] for r in accepted_rows),
                min_accepted_margin=min(3e-5-r["svd_gap"] for r in accepted_rows),
                max_gap_delta=max(abs(r["svd_gap"]-r["gram_gap"]) for r in rows),
                max_direction_error=max(r["direction_error"] for r in rows),
                max_primal_error=max(r["primal_error"] for r in rows),
                max_dual_error=max(r["dual_error"] for r in rows),
                near_threshold=sorted(accepted_rows,key=lambda r:r["svd_gap"],reverse=True)[:8],
                mismatches=mismatches)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("EARLY_REPLAY",json.dumps(result),flush=True)


if __name__=="__main__":
    main()
