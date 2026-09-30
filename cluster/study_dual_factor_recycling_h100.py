"""Locked 50-step, baseline-controlled research replay of recycled SVD factors."""
import io
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
import qnormuon.coupled_solver as solver
import qnormuon.optimizer as optimizer_module
from benchmarks.tiny_transformer import (ModelConfig, Optimizers, TinyTransformer,
    datasets, seed_everything, train_step)
from experiments.dual_factor_recycling import (pack_factors, predict,
    solve_and_capture, unpack_factors)
from experiments.dual_warm_start_predictor import problem_scale

BUDGETS = (1, 2, 4, 8)


def sync():
    torch.cuda.synchronize()


def angle_metrics(x, target):
    xn, tn = float(x.norm()), float(target.norm())
    return dict(cosine=float((x @ target) / (xn * tn)) if xn and tn else None,
                norm_ratio=xn/tn if tn else None,
                relative_error=float((x-target).norm())/tn if tn else None)


def solve_record(u, d, a, *, config, initial_lambda):
    first = {}
    original = solver.SmoothDual.evaluate
    def evaluate(problem, z):
        ev = original(problem, z)
        if not first:
            first["initial_gradient_norm"] = float(
                solver.horizontal_residual(u,d,ev.pair).norm())
        return ev
    sync(); start=time.perf_counter()
    try:
        with patch.object(solver.SmoothDual,"evaluate",evaluate):
            result=solver.solve_coupled(u,d,a,config=config,initial_lambda=initial_lambda)
        sync(); elapsed=time.perf_counter()-start
        return result, dict(certified=result.converged, fallback=result.fallback,
            reason=result.reason, newton_iterations=result.iterations,
            smooth_evaluations=1+result.counts.line_trials,
            line_search_evaluations=result.counts.line_trials,
            svd_matrices=result.counts.svd_matrices,
            initial_normalized_gap=result.history[0]["normalized_gap"] if result.history else None,
            final_normalized_gap=result.metrics["normalized_gap"],
            residual_rcond=result.metrics["residual_rcond"],
            solve_seconds=elapsed, **first)
    except Exception as error:
        sync()
        return None, dict(certified=False,fallback=False,
            reason=type(error).__name__+": "+str(error),solve_seconds=time.perf_counter()-start,**first)


def stress(sample, config):
    cache,u,d,a=sample["cache"],sample["u"],sample["d"],sample["a"]
    gen=torch.Generator(device=u.device).manual_seed(221)
    cases=dict(ordinary=a,objective_x10=10*a,objective_sign_flip=-a,
               column_rotation=torch.roll(a,83,dims=-1),
               additive_perturbation=a+5*torch.randn(a.shape,device=a.device,
                   dtype=a.dtype,generator=gen)*a.square().mean().sqrt())
    output={}
    for name,objective in cases.items():
        cfg=config
        baseline,br=solve_record(u,d,objective,config=cfg,initial_lambda=cache.lam)
        row=dict(baseline=br,candidates={})
        for label,budget,gate in (("budget2",2,False),("budget8",8,False),("gated2",2,True)):
            sync();started=time.perf_counter();prediction=predict(cache,u,d,objective,budget=budget,gate=gate)
            sync();pt=time.perf_counter()-started
            if prediction.reason=="frobenius_gate_rejected":
                cr=dict(br)
            else:
                _,cr=solve_record(u,d,objective,config=cfg,initial_lambda=prediction.lam)
            row["candidates"][label]=dict(predictor_seconds=pt,predictor_reason=prediction.reason,
                gate_ratio=prediction.indicator["delta_over_sigma_min"],**cr)
        output[name]=row
    return output


def summarize(rows):
    output={}
    for label in ("baseline","baseline_direct","budget1","budget2","budget4","budget8","gated2","fp32_budget2"):
        records=[r[label] for r in rows if r["step"]>0]
        bins={str(k):sum(rec.get("certified") and
            (rec.get("newton_iterations",99)==k if k<3 else rec.get("newton_iterations",0)>=3)
            for rec in records) for k in (0,1,2,3)}
        output[label]=dict(solves=len(records),certified=sum(r.get("certified",False) for r in records),
            failures=[dict(step=rows[i+6]["step"],pair=rows[i+6]["pair"],reason=r["reason"])
                for i,r in enumerate(records) if not r.get("certified")],
            newton_bins=bins,
            smooth_evaluations=sum(r.get("smooth_evaluations",0) for r in records),
            line_search_evaluations=sum(r.get("line_search_evaluations",0) for r in records),
            predictor_seconds=sum(r.get("predictor_seconds",0.) for r in records),
            solver_seconds=sum(r.get("solve_seconds",0.) for r in records),
            gate_rejections=sum(r.get("predictor_reason")=="frobenius_gate_rejected" for r in records))
    return output


def main():
    if not sitecustomize.QSO_NETWORK_GUARD_ACTIVE or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("offline H100 allocation required")
    torch.backends.cuda.matmul.allow_tf32=False
    config=json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
    source=Path(config["output_root"])/f"smoke-28936-qso-seed{config['seed']}"
    baseline=[json.loads(line) for line in (source/"metrics.jsonl").read_text().splitlines()]
    seed_everything(config["seed"],config["tf32"])
    train,val=datasets(config)
    model=TinyTransformer(ModelConfig(**config["model"])).float().cuda()
    opts=Optimizers(model,"qso",config)
    names=[p.name for p in model.pairs()]
    caches={};rows=[];training=[];sample={};checkpoint={}
    current_step=[0];call=[0]
    def capture(u,d,a,**kwargs):
        name=names[call[0]%6];call[0]+=1;step=current_step[0]
        previous=caches.get(name)
        if previous is not None and not torch.equal(previous.lam, kwargs["initial_lambda"]):
            raise RuntimeError(f"cached accepted lambda disagrees with production warm state: {name}")
        scale=problem_scale(u,d,a)
        candidates={}
        if previous is not None:
            for budget in BUDGETS:
                sync();start=time.perf_counter()
                trial=predict(previous,u,d,a,budget=budget)
                sync();seconds=time.perf_counter()-start
                candidates[f"budget{budget}"]=(trial,seconds)
            compressed=previous.predictor_copy(torch.float32)
            sync();start=time.perf_counter()
            compressed_trial=predict(compressed,u,d,a,budget=2)
            sync();seconds=time.perf_counter()-start
            candidates["fp32_budget2"]=(compressed_trial,seconds)
            sync();start=time.perf_counter()
            gated=predict(previous,u,d,a,budget=2,gate=True)
            sync();seconds=time.perf_counter()-start
            candidates["gated2"]=(gated,seconds)
        first={}
        def observe(kind,problem,ev,direction=None):
            if kind=="evaluation" and "initial_gradient_norm" not in first:
                first["initial_gradient"]=-solver.horizontal_residual(u,d,ev.pair).detach().clone()
                first["initial_gradient_norm"]=float(first["initial_gradient"].norm())
            if kind=="newton" and "first_newton_delta" not in first:
                first["first_newton_delta"]=scale.magnitude*scale.coord*direction.detach().clone()
        sync();start=time.perf_counter()
        result,current=solve_and_capture(u,d,a,config=kwargs["config"],
                                         initial_lambda=kwargs["initial_lambda"],observer=observe)
        sync();elapsed=time.perf_counter()-start
        if not result.converged or current is None or result.fallback:
            raise RuntimeError(f"baseline failed {step} {name}: {result.reason}")
        old_lambda=previous.lam if previous is not None else scale.beta
        br=dict(certified=True,fallback=False,reason=result.reason,
                newton_iterations=result.iterations,smooth_evaluations=1+result.counts.line_trials,
                line_search_evaluations=result.counts.line_trials,
                initial_normalized_gap=result.history[0]["normalized_gap"],
                final_normalized_gap=result.metrics["normalized_gap"],
                residual_rcond=result.metrics["residual_rcond"],
                solve_seconds=elapsed,initial_gradient_norm=first["initial_gradient_norm"],
                accepted_factor_reconstruction_error=current.reconstruction_error)
        row=dict(step=step,pair=name,baseline=br,cache_factor_bytes=current.factor_bytes(),
                 cache_total_bytes=current.tensor_bytes())
        if previous is not None:
            direct_result,direct_record=solve_record(u,d,a,config=kwargs["config"],
                                                     initial_lambda=previous.lam)
            if direct_result is None or not torch.equal(direct_result.lam,result.lam):
                raise RuntimeError("direct previous-lambda solve diverged from captured baseline")
            row["baseline_direct"]=direct_record
            actual_first=first.get("first_newton_delta")
            target=result.lam-old_lambda
            for label,(trial,pt) in candidates.items():
                if trial.reason=="frobenius_gate_rejected":
                    cr=dict(br)
                elif label=="budget2":
                    observed={}
                    def candidate_observer(kind,problem,ev,direction=None):
                        if kind=="evaluation" and "initial_gradient_norm" not in observed:
                            observed["initial_gradient_norm"]=float(
                                solver.horizontal_residual(u,d,ev.pair).norm())
                    sync();candidate_start=time.perf_counter()
                    candidate_result,candidate_cache=solve_and_capture(
                        u,d,a,config=kwargs["config"],initial_lambda=trial.lam,
                        observer=candidate_observer)
                    sync();candidate_seconds=time.perf_counter()-candidate_start
                    if not candidate_result.converged or candidate_cache is None:
                        raise RuntimeError(f"predicted solve failed at {step} {name}")
                    cr=dict(certified=True,fallback=candidate_result.fallback,
                        reason=candidate_result.reason,
                        newton_iterations=candidate_result.iterations,
                        smooth_evaluations=1+candidate_result.counts.line_trials,
                        line_search_evaluations=candidate_result.counts.line_trials,
                        svd_matrices=candidate_result.counts.svd_matrices,
                        initial_normalized_gap=candidate_result.history[0]["normalized_gap"],
                        final_normalized_gap=candidate_result.metrics["normalized_gap"],
                        residual_rcond=candidate_result.metrics["residual_rcond"],
                        solve_seconds=candidate_seconds,**observed)
                else:
                    _,cr=solve_record(u,d,a,config=kwargs["config"],
                                      initial_lambda=trial.lam)
                cr.update(predictor_seconds=pt,predictor_reason=trial.reason,
                    predictor_cg_iterations=trial.cg_iterations,
                    gate_ratio=trial.indicator["delta_over_sigma_min"],
                    gate_pass=trial.indicator["frobenius_full_rank_gate"],
                    delta_over_old_fro=trial.indicator["delta_over_old_fro"],
                    predicted_gradient_norm=float(trial.gradient.norm()))
                cr["predicted_gradient_vs_oracle"]=angle_metrics(
                    trial.gradient,first["initial_gradient"])
                if actual_first is not None:
                    cr["versus_first_newton"]=angle_metrics(trial.delta_lambda,actual_first)
                cr["versus_final_drift"]=angle_metrics(trial.delta_lambda,target)
                if label=="fp32_budget2":
                    cr["versus_fp64_prediction"]=angle_metrics(
                        trial.delta_lambda,candidates["budget2"][0].delta_lambda)
                row[label]=cr
            if step==25 and name==names[0]:
                sample.update(cache=previous,u=u.detach().clone(),d=d.detach().clone(),a=a.detach().clone())
        rows.append(row)
        caches[name]=current
        return result
    with patch.object(optimizer_module,"solve_coupled",capture):
        for step in range(50):
            current_step[0]=step
            rec=train_step(model,opts,train,config,step)
            if rec["batch_sha256"]!=baseline[step]["batch_sha256"] or rec["loss"]!=baseline[step]["loss"]:
                raise RuntimeError(f"locked data/loss mismatch at step {step}")
            training.append(dict(step=step,loss=rec["loss"],batch_sha256=rec["batch_sha256"]))
            if step==24:
                buffer=io.BytesIO()
                torch.save({name:pack_factors(value) for name,value in caches.items()},buffer)
                checkpoint["serialized_bytes"]=buffer.tell()
                buffer.seek(0)
                reloaded={name:unpack_factors(value) for name,value in
                          torch.load(buffer,weights_only=True).items()}
                checkpoint["factor_state_exact"]=all(torch.equal(getattr(caches[name],key),
                    getattr(reloaded[name],key)) for name in caches for key in
                    ("lam","left","singular","right"))
                if not checkpoint["factor_state_exact"]:
                    raise RuntimeError("factor cache checkpoint roundtrip changed tensors")
                caches.clear();caches.update(reloaded)
            if step%5==4:
                print(f"replay step {step+1}/50, pair solves {len(rows)}",flush=True)
    if len(rows)!=300:
        raise RuntimeError("expected six pairs per step")
    cfg=opts.paired.param_groups[0]["solver"]
    output=dict(job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,solver=repr(cfg),config=config,training=training,
        cache_bytes_per_pair=rows[-1]["cache_factor_bytes"],
        cache_total_bytes_per_pair=rows[-1]["cache_total_bytes"],
        fp32_cache_factor_bytes_per_pair=next(iter(caches.values())).predictor_copy(torch.float32).factor_bytes(),
        checkpoint=checkpoint,summary=summarize(rows),stress=stress(sample,kwargs_config(cfg)),rows=rows)
    path=Path("cluster")/f"dual_factor_recycling-{os.environ['SLURM_JOB_ID']}.json"
    path.write_text(json.dumps(output,separators=(",",":"),allow_nan=False)+"\n")
    print("RESULT",json.dumps(dict(path=str(path),summary=output["summary"],
                                  checkpoint=checkpoint,stress=output["stress"])),flush=True)


def kwargs_config(group):
    return solver.SolverConfig(**group)


if __name__=="__main__":
    main()
