"""Independent full-SVD versus production Gram on identical saved warm inputs."""
import json
import os
from pathlib import Path
from unittest.mock import patch

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM allocation required")

import sitecustomize
import torch
import qnormuon.optimizer as optimizer_module
from benchmarks.tiny_transformer import ModelConfig, Optimizers, TinyTransformer, datasets, load_checkpoint, seed_everything, train_step


def main():
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert "H100" in torch.cuda.get_device_name(0)
    config=json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
    source=Path(config["output_root"])/f"smoke-28922-qso-seed{config['seed']}"
    checkpoint=source/"checkpoint.pt"
    baseline=json.loads((source/"metrics.jsonl").read_text().splitlines()[25])
    seed_everything(config["seed"],config["tf32"])
    train,val=datasets(config)
    metadata=dict(train=train.metadata,validation=val.metadata)
    rows=[]
    original=optimizer_module.solve_coupled
    for backend in ("svd","gram_upper"):
        model=TinyTransformer(ModelConfig(**config["model"])).float().cuda()
        opts=Optimizers(model,"qso",config)
        assert load_checkpoint(checkpoint,model,opts,config,metadata)==25
        for group in opts.paired.param_groups:
            group["solver"]["primal_norm_backend"]=backend
        before={name:t.detach().clone() for name,t in model.named_parameters()}
        names=[p.name for p in opts.paired.pairs]
        saved=[]
        def observed(u,d,a,**kwargs):
            result=original(u,d,a,**kwargs)
            saved.append(dict(name=names[len(saved)],u=u.clone(),d=d.clone(),a=a.clone(),
                              initial_lambda=None if kwargs.get("initial_lambda") is None else kwargs["initial_lambda"].clone(),
                              result=result))
            return result
        with patch.object(optimizer_module,"solve_coupled",observed):
            record=train_step(model,opts,train,config,25)
        assert len(saved)==6 and record["batch_sha256"]==baseline["batch_sha256"]
        assert record["loss"]==baseline["loss"]
        rows.append(dict(backend=backend,model=model,opts=opts,before=before,saved=saved,record=record))
    old,new=rows
    def maximum_tensor_difference(x,y):
        return max(float((a-b).abs().max()) for a,b in zip(x,y))
    input_error=max(maximum_tensor_difference((a[k],),(b[k],)) for a,b in zip(old["saved"],new["saved"])
                    for k in ("u","d","a","initial_lambda"))
    assert input_error==0
    comparison=[]
    for a,b in zip(old["saved"],new["saved"]):
        x,y=a["result"],b["result"]
        assert a["name"]==b["name"] and x.converged and y.converged and not x.fallback and not y.fallback
        comparison.append(dict(pair=a["name"],svd_iterations=x.iterations,gram_iterations=y.iterations,
                               lambda_max_abs_error=float((x.lam-y.lam).abs().max()),
                               primal_direction_max_abs_error=float((x.pair-y.pair).abs().max()),
                               primal_objective_abs_error=abs(x.metrics["primal_objective"]-y.metrics["primal_objective"]),
                               dual_objective_abs_error=abs(x.metrics["dual_objective"]-y.metrics["dual_objective"]),
                               normalized_gap_abs_error=abs(x.metrics["normalized_gap"]-y.metrics["normalized_gap"]),
                               rcond_abs_error=abs(x.metrics["residual_rcond"]-y.metrics["residual_rcond"]),
                               horizontal_abs_error=abs(x.metrics["horizontal_residual"]-y.metrics["horizontal_residual"]),
                               spectral_excess_abs_error=abs(x.metrics["spectral_excess"]-y.metrics["spectral_excess"])))
    model_max=max(float((p.detach()-q.detach()).abs().max()) for p,q in zip(old["model"].parameters(),new["model"].parameters()))
    lifted_max=max(float(((old["before"][name]-p.detach())-(new["before"][name]-q.detach())).abs().max())
                   for (name,p),(_,q) in zip(old["model"].named_parameters(),new["model"].named_parameters()))
    result=dict(job_id=os.environ["SLURM_JOB_ID"],checkpoint=str(checkpoint),
                same_batch=True,input_error=input_error,
                loss_abs_error=abs(old["record"]["loss"]-new["record"]["loss"]),
                model_parameter_max_abs_error=model_max,lifted_update_max_abs_error=lifted_max,
                comparison=comparison)
    assert max(x["normalized_gap_abs_error"] for x in comparison)<1e-9
    assert max(x["primal_direction_max_abs_error"] for x in comparison)<1e-9
    path=Path("cluster")/f"primal_backend_comparison-{os.environ['SLURM_JOB_ID']}.json"
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("BACKEND_COMPARISON",json.dumps(result),flush=True)


if __name__=="__main__":
    main()
