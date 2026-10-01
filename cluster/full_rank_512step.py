"""One frozen tuned research trajectory, never a sweep; SLURM/H100 only."""
import os
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM required")
import copy
import gc
import hashlib
import json
from pathlib import Path
import signal
import socket
import sys
import time
import traceback
from collections import Counter
from unittest.mock import patch
import numpy as np
import torch
import sitecustomize
from benchmarks import tiny_transformer as tt
from benchmarks.lr_study import SinglePassStream, INITIAL_HASH, batch_digest, exact_tree, quality
from experiments.full_rank_training import (POLICY, AdmissionFailure, research_boundary,
    compact_metrics, online_gate)
from cluster.run_lr_study import fingerprint

SOURCE = Path("/home/prignano/qnormuon-runs/lr-stage-d/study-29024")
QSO_LABEL = "confirmation-qso-a0.0003-q0.00122474487139"
ADAM_LABEL = "confirmation-adamw-a0.0002-q0.001"
BLOCKER = Path("/home/prignano/qnormuon-runs/stage-d-forensics/capture-29044") / QSO_LABEL / "failed_pair.pt"
ROOT = Path("/home/prignano/qnormuon-runs/full-rank-512step")


def clean(value):
    if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return str(value)
    return value


def write(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n")


def cpu(value):
    return value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value


def fixture(u,d,a,lam,result=None):
    return dict(u=cpu(u),d=cpu(d),a=cpu(a),initial_lambda=cpu(lam),
        pair=cpu(result.pair) if result else None,lam=cpu(result.lam) if result else None,
        metrics=result.metrics if result else {},history=result.history if result else [],
        actions=result.actions if result else [],evaluations=result.evaluations if result else [])


def timeout(*args):
    raise TimeoutError("unchanged 180-second per-step budget exceeded")


def provenance_check(manifest, q, adam):
    for p,h in manifest["source_sha256"].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h, f"frozen source changed: {p}"
    config=q["config"]
    assert config["seed"]==2026 and config["qso_lr"]==0.0012247448713915891
    assert config["adam_lr"]==0.0003 and config["steps"]==512 and config["warmup_steps"]==51
    assert config["batch_size"]*config["model"]["sequence_length"]*config["gradient_accumulation"]==2048
    assert config["precision"]=="bf16_autocast_fp32_storage" and not config["tf32"]
    assert config["evaluation_interval"]==32 and config["weight_decay"]==0
    assert config["solver"]==dict(dtype="float64",tolerance=3e-5,rcond_guard=1e-4,
        initial_iterations=2,max_iterations=100,max_cg=30,fallback=True,primal_norm_backend="gram_upper")
    assert q["initialization_sha256"]==adam["initialization_sha256"]==INITIAL_HASH
    assert q["dataset"]==adam["dataset"]==manifest["data"]
    assert q["expected_batch_hashes"]==adam["expected_batch_hashes"]==manifest["batch_hashes"]
    assert q["clipping"]==adam["clipping"]=="none"
    expected=copy.deepcopy(config);expected.update(adam_lr=0.0002,qso_lr=0.001)
    assert adam["config"]==expected, "incomparable AdamW reference"
    assert json.loads((SOURCE/ADAM_LABEL/"summary.json").read_text())["status"]=="passed"
    return config


def statistics(rows, records):
    normal=[r for r in rows if r["selection_semantics"]==POLICY["name"]]
    below=[r for r in normal if r["below_old_guard"]]
    bins={str(n):sum(r["newton_iterations"]==n for r in rows) for n in (0,1,2)}
    bins["3+"]=sum(r["newton_iterations"]>=3 for r in rows)
    def dist(key): return tt.summarize([r[key] for r in normal])
    def sides(r): return "both" if max(r["rcond_sides"])<=1e-4 else ("U" if r["rcond_sides"][0]<=1e-4 else "D")
    return dict(pair_solves=len(rows),below_old_guard=len(below),
        below_fraction=len(below)/len(rows) if rows else 0.,
        first_below_step=min((r["step"] for r in below),default=None),
        last_below_step=max((r["step"] for r in below),default=None),
        below_by_pair=dict(Counter(r["pair"] for r in below)),
        below_by_side=dict(Counter(sides(r) for r in below)),
        any_evaluation_below_old_guard=sum(r["any_evaluation_below_old_guard"] for r in rows),
        minimum_rcond=min((min(r["rcond_sides"]) for r in normal),default=None),
        minimum_alpha_lower=min((min(r["alpha_lower"]) for r in normal),default=None),
        minimum_rank_margin=min((min(r["sigma_min_over_eta"]) for r in normal),default=None),
        newton_bins=bins,cg=tt.summarize([r["cg_iterations"] for r in rows]),
        line_trials=tt.summarize([r["line_trials"] for r in rows]),
        conservative_gap=dist("conservative_normalized_gap"),
        normalized_horizontal=dist("normalized_horizontal_residual"),
        spectral_excess=dist("spectral_excess"),proxy_distance=dist("feasible_proxy_distance"),
        rank_posterior_seconds=dist("rank_seconds"),value_posterior_seconds=dist("value_seconds"),
        total_rank_posterior_seconds=sum(r["rank_seconds"] for r in rows),
        total_value_posterior_seconds=sum(r["value_seconds"] for r in rows),
        cpu_reference_calls=0,all_returns_certified=all(r["converged"] for r in rows),
        svd_matrices=sum(r["svd_evaluations"] for r in rows))


def main():
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0)
    signal.signal(signal.SIGALRM,timeout)
    manifest=json.loads((SOURCE/"manifest.json").read_text())
    original=json.loads((SOURCE/QSO_LABEL/"provenance.json").read_text())
    adam=json.loads((SOURCE/ADAM_LABEL/"provenance.json").read_text())
    config=provenance_check(manifest,original,adam)
    output=ROOT/f"trajectory-{os.environ['SLURM_JOB_ID']}";output.mkdir(parents=True,exist_ok=False)
    historical=[json.loads(s) for s in (SOURCE/QSO_LABEL/"metrics.jsonl").read_text().splitlines()]
    assert len(historical)==87
    original_pairs={(r["step"],r["pair"]):r for r in map(json.loads,(SOURCE/QSO_LABEL/"solves.jsonl").read_text().splitlines()) if r["event"]=="end"}
    blocker=torch.load(BLOCKER,map_location="cpu",weights_only=False)
    tt.seed_everything(2026,False);torch.set_default_dtype(torch.float32)
    train_source,val=tt.datasets(config);train=SinglePassStream(train_source,128,2026)
    data=dict(train=train.metadata,validation=val.metadata)
    assert data==manifest["data"] and train.metadata["usable_input_tokens"]==1048576
    hashes=[batch_digest(train,i,config) for i in range(512)]
    assert hashes==manifest["batch_hashes"]
    validation_hashes=[]
    for i in range(config["evaluation_batches"]):
        x,y=val.batch(i,config["batch_size"],128,"cpu")
        validation_hashes.append(hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest())
    # The original immutable code, metadata, seed and validation convention match.
    # Explicit validation hashes are additionally recorded for this trajectory.
    sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        ["experiments/full_rank_training.py","experiments/full_rank_admission.py",
         "experiments/qso_output_contract.py","experiments/full_rank_direction.py",
         "cluster/full_rank_512step.py","cluster/full_rank_512step.sbatch"]}
    checkpoint_metadata=dict(data, research_admission=dict(POLICY,source_sha256=sources))
    model=tt.TinyTransformer(tt.ModelConfig(**config["model"])).float().cuda()
    opts=tt.Optimizers(model,"qso",config)
    assert fingerprint(model)==INITIAL_HASH
    assert opts.paired.record_diagnostics and not opts.paired.record_cast_diagnostics
    assert len(opts.paired.pairs)==6
    write(output/"provenance.json",dict(original_stage_d=original,adamw_reference=str(SOURCE/ADAM_LABEL),
        policy=POLICY,config=config,data=data,initialization_sha256=INITIAL_HASH,
        validation_batch_hashes=validation_hashes,expected_batch_hashes=hashes,
        source_sha256=sources,job_id=os.environ["SLURM_JOB_ID"],hostname=socket.gethostname(),
        python=sys.version,python_executable=sys.executable,torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),cpu_count=os.environ.get("SLURM_CPUS_PER_TASK"),
        memory_mb=os.environ.get("SLURM_MEM_PER_NODE"),output_path=str(output),
        parameter_count=sum(p.numel() for p in model.parameters()),clipping="none",
        prefix_evidence="87 losses/batches; completed pair numerical records; exact saved step-87 pair inputs/raw weights/gradients/lambda"))
    records=[];rows=[];evaluations={};samples={};resume=None;gate=None
    active={"step":0,"pair":None};last_inputs=None;pair_seconds=0.;prefix_pair_matches=0
    initial_state_hash=None;first_crossing=None;peak=0;wall=time.perf_counter()
    def before(pair,u,d,a,lam):
        nonlocal last_inputs
        active["pair"]=pair.name;last_inputs=(u,d,a,lam)
        if active["step"]==87 and pair.name=="blocks.1.mlp":
            for key,value in (("u",u),("d",d),("a",a),("initial_lambda",lam)):
                assert torch.equal(value.cpu(),blocker[key]), f"historical blocker {key} differs"
            for key,value in (("up",pair.up),("down",pair.down),("grad_up",pair.up.grad),("grad_down",pair.down.grad)):
                assert torch.equal(value.cpu(),blocker["raw"][key]), f"historical raw {key} differs"
            assert all(opts.paired.state[p.up]["step"]==87 for p in opts.paired.pairs)
            write(output/"prefix_gate.json",dict(passed=True,completed_steps=87,matched_losses=87,
                matched_batches=87,matched_completed_pair_records=prefix_pair_matches,
                exact_blocker_canonical_inputs=True,exact_raw_weights_gradients=True,
                exact_original_lambda=True,pair_counts=87,model_hash_at_87=fingerprint(model),
                historical_full_model_hash_available=False,historical_full_optimizer_checkpoint_available=False))
    def callback(pair,u,d,a,lam,result):
        nonlocal pair_seconds,prefix_pair_matches,first_crossing
        row=dict(compact_metrics(result),step=active["step"],pair=pair.name,seconds=result.seconds)
        if not result.converged:
            failed=fixture(u,d,a,lam,result);failed.update(metadata=dict(active),raw=dict(
                up=cpu(pair.up),down=cpu(pair.down),grad_up=cpu(pair.up.grad),grad_down=cpu(pair.down.grad)))
            torch.save(failed,output/"failed_pair.pt")
            write(output/"smooth_failure.json",dict(row,history=result.history,actions=result.actions,evaluations=result.evaluations))
        if active["step"]<87 and result.converged:
            old=original_pairs[(active["step"],pair.name)]
            for key in ("normalized_gap","residual_rcond","newton_iterations","cg_iterations"):
                assert row[key]==old[key], f"prefix pair {key} differs"
            assert result.counts.line_trials==old["line_search_trials"]
            prefix_pair_matches+=1
        if result.converged:
            rows.append(row);pair_seconds+=result.seconds
            solve_log.write(json.dumps(clean(row),allow_nan=False)+"\n")
            if row["below_old_guard"]:
                if first_crossing is None:
                    first_crossing=row;write(output/"first_old_guard_crossing.json",row)
                    print("OLD_GUARD_CROSSED",row,flush=True)
                criteria={"first":not samples,"minimum_rcond":("minimum_rcond" not in samples or min(row["rcond_sides"])<min(samples["minimum_rcond"]["metadata"]["rcond_sides"])),
                          "minimum_rank_margin":("minimum_rank_margin" not in samples or min(row["sigma_min_over_eta"])<min(samples["minimum_rank_margin"]["metadata"]["sigma_min_over_eta"]))}
                if "distinct_block" not in samples and first_crossing["pair"]!=pair.name:
                    criteria["distinct_block"]=True
                if any(criteria.values()):
                    sample=fixture(u,d,a,lam,result)
                    # Accepted fixtures retain only final tensors/metrics, not huge query histories.
                    for key in ("history","actions","evaluations"): sample.pop(key)
                    sample["metadata"]=row
                    for name,keep in criteria.items():
                        if keep:samples[name]=sample
    try:
        with (output/"solves.jsonl").open("w",buffering=1) as solve_log, (output/"metrics.jsonl").open("w",buffering=1) as log:
            initial=tt.evaluate(model,val,config)
            assert initial==json.loads((SOURCE/QSO_LABEL/"validation.json").read_text())["0"]
            evaluations[0]=initial;write(output/"validation.json",evaluations)
            torch.cuda.reset_peak_memory_stats()
            for step in range(512):
                active.update(step=step,pair=None);pair_seconds=0.;last_inputs=None
                # Hash state before an attempted step only in memory-efficient digests.
                before_model=fingerprint(model)
                pair_steps=[opts.paired.state.get(p.up,{}).get("step",0) for p in opts.paired.pairs]
                signal.alarm(config["step_timeout_seconds"])
                try:
                    with research_boundary(opts.paired,callback,before,profile=True):
                        record=tt.train_step(model,opts,train,config,step)
                except Exception:
                    assert fingerprint(model)==before_model, "failed step partially committed model"
                    assert pair_steps==[opts.paired.state.get(p.up,{}).get("step",0) for p in opts.paired.pairs]
                    raise
                finally: signal.alarm(0)
                assert record["batch_sha256"]==hashes[step]
                if step<87:assert record["loss"]==historical[step]["loss"], "historical loss prefix differs"
                if record["loss"]>config["divergence_loss_limit"]:raise AdmissionFailure("predeclared_loss_divergence")
                record["pair_solver_seconds"]=pair_seconds
                peak=max(peak,torch.cuda.max_memory_allocated())
                if (step+1)%32==0:
                    signal.alarm(config["step_timeout_seconds"])
                    loss=tt.evaluate(model,val,config);signal.alarm(0)
                    if not np.isfinite(loss) or loss>config["divergence_loss_limit"]:raise AdmissionFailure("validation_nonfinite_or_divergent")
                    evaluations[step+1]=loss;record["validation_loss"]=loss;write(output/"validation.json",evaluations)
                    print("VALIDATION",step+1,loss,flush=True)
                records.append(record);log.write(json.dumps(clean(record),allow_nan=False)+"\n")
                if step+1==128:
                    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
                    gate=online_gate(rows);write(output/"online_gate_128.json",gate);print("GATE_128",gate,flush=True)
                if step+1==256:
                    tt.save_checkpoint(output/"checkpoint.pt",model,opts,config,checkpoint_metadata)
                if step+1==259:
                    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
                    other=tt.TinyTransformer(tt.ModelConfig(**config["model"])).float().cuda()
                    other_opts=tt.Optimizers(other,"qso",config)
                    assert tt.load_checkpoint(output/"checkpoint.pt",other,other_opts,config,checkpoint_metadata)==256
                    replay_rows=[]
                    def replay_callback(pair,u,d,a,lam,result):
                        replay_rows.append(dict(compact_metrics(result),pair=pair.name))
                    for j in range(256,259):
                        signal.alarm(config["step_timeout_seconds"])
                        with research_boundary(other_opts.paired,replay_callback,profile=False):
                            replay=tt.train_step(other,other_opts,train,config,j)
                        signal.alarm(0)
                        assert replay["loss"]==records[j]["loss"] and replay["batch_sha256"]==hashes[j]
                        # Timings are observational; all admission decisions/rank/value quantities match exactly.
                        for x,y in zip(replay_rows[-6:],rows[j*6:(j+1)*6]):
                            for key in ("pair","selection_semantics","selection_certified","newton_iterations","cg_iterations","line_trials",
                                        "alpha_lower","conservative_normalized_gap","rcond_sides","spectral_excess","normalized_horizontal_residual"):
                                assert x[key]==y[key], f"checkpoint admission differs: {key}"
                    exact_tree(model.state_dict(),other.state_dict());exact_tree(opts.state_dict(),other_opts.state_dict())
                    resume=dict(passed=True,exact=True,checkpoint_step=256,replayed_completed_steps=[257,258,259],
                        checkpoint_bytes=(output/"checkpoint.pt").stat().st_size,
                        checked=["model","canonical_EMA","original_lambda","pair_steps","unsupported_AdamW","scheduler","data_hashes","losses","admission"])
                    write(output/"resume.json",resume)
                    del other,other_opts;gc.collect();torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
                    torch.cuda.reset_peak_memory_stats()
            assert gate and resume and len(rows)==3072 and sorted(evaluations)==list(range(0,513,32))
            summary=dict(status="passed",completed_steps=512,tokens=1048576,policy=POLICY,
                **quality(evaluations,records),total_wall_seconds=time.perf_counter()-wall,
                measured_step_seconds=sum(r["step_seconds"] for r in records),
                optimizer_seconds=sum(r["optimizer_seconds"] for r in records),peak_cuda_bytes=peak,
                solver=statistics(rows,records),prefix_gate=True,resume=resume,online_gate=gate,
                warm_excluded_steps=5,warm={k:tt.summarize([r[k] for r in records[5:]]) for k in
                    ("step_seconds","pair_solver_seconds","optimizer_seconds","forward_backward_seconds")})
            summary["tokens_per_second"]=summary["tokens"]/summary["measured_step_seconds"]
            summary["optimizer_fraction"]=summary["optimizer_seconds"]/summary["measured_step_seconds"]
            write(output/"summary.json",summary)
    except Exception as error:
        signal.alarm(0)
        if last_inputs is not None and not (output/"failed_pair.pt").exists():
            torch.save(dict(fixture(*last_inputs),metadata=dict(active)),output/"failed_pair.pt")
        failure=dict(status="failed",completed_steps=len(records),tokens=len(records)*2048,active=active,
            exception=repr(error),traceback=traceback.format_exc(),solver=statistics(rows,records),
            total_wall_seconds=time.perf_counter()-wall,peak_cuda_bytes=peak,policy=POLICY)
        write(output/"failure.json",failure);print("RESEARCH_FAILURE",json.dumps(clean(failure)),flush=True)
        raise
    finally:
        for name,sample in samples.items():torch.save(sample,output/f"audit_{name}.pt")
        for p,h in manifest["source_sha256"].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        print("OUTPUT",output,flush=True)


if __name__=="__main__":main()
