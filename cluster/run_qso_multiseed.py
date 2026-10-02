"""Frozen paired quality study. One H100 allocation per seed; no retuning."""
import os
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM required")
import argparse
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
import numpy as np
import torch
import sitecustomize
from benchmarks import tiny_transformer as tt
from benchmarks.lr_study import SinglePassStream, batch_digest, exact_tree, quality
from benchmarks.multiseed import SEEDS, configuration, verify_pair
from experiments.full_rank_training import POLICY, AdmissionFailure, research_boundary, compact_metrics, online_gate
from cluster.full_rank_512step import statistics, clean, write, cpu, fixture, provenance_check, SOURCE, QSO_LABEL, ADAM_LABEL
from cluster.run_lr_study import fingerprint, timeout

BASELINE = Path("/home/prignano/qnormuon-runs/full-rank-512step/trajectory-29082")
FILES = ["benchmarks/multiseed.py", "cluster/run_qso_multiseed.py",
         "cluster/run_qso_multiseed.sbatch", "cluster/preflight_qso_multiseed.sbatch",
         "tests/test_multiseed.py"]


def read(path):
    return json.loads(Path(path).read_text())


def check_sources(manifest):
    for path, digest in manifest["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("frozen source changed: " + path)


def data_contract(config):
    source, val = tt.datasets(config)
    train = SinglePassStream(source,128,config["seed"])
    if train.metadata["usable_input_tokens"] != 1048576 or len(train.order) != len(set(train.order.tolist())):
        raise RuntimeError("required nonrepeated token contract missing")
    hashes = [batch_digest(train,i,config) for i in range(512)]
    vh = [batch_digest(val,i,config) for i in range(4)]
    return train,val,dict(train=train.metadata,validation=val.metadata),hashes,vh


def preflight():
    original_manifest=read(SOURCE/"manifest.json")
    historical_q=read(SOURCE/QSO_LABEL/"provenance.json")
    historical_a=read(SOURCE/ADAM_LABEL/"provenance.json")
    provenance_check(original_manifest,historical_q,historical_a)
    old=read(BASELINE/"provenance.json")
    # Validate all numerical/model/training sources used by the completed run.
    for path,digest in old["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, path
    assert old["config"]==historical_q["config"] and old["original_stage_d"]==historical_q
    assert old["policy"]==POLICY and read(BASELINE/"summary.json")["status"]=="passed"
    assert read(BASELINE/"resume.json")["exact"]
    base=read("configs/tiny_transformer/lr_study.json")
    assert configuration(base,2026,"qso")==old["config"]
    assert configuration(base,2026,"adamw")==historical_a["config"]
    paths=set(original_manifest["source_sha256"])|set(old["source_sha256"])|set(FILES)
    sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(paths)}
    root=Path("/home/prignano/qnormuon-runs/paired-multiseed")/f"study-{os.environ['SLURM_JOB_ID']}"
    root.mkdir(parents=True,exist_ok=False)
    seeds={}
    for seed in SEEDS:
        config=configuration(base,seed,"qso")
        tt.seed_everything(seed,False);torch.set_default_dtype(torch.float32)
        train,val,data,hashes,vh=data_contract(config)
        assert data["train"]["prefix_sha256"]==old["data"]["train"]["prefix_sha256"]
        assert data["validation"]==old["data"]["validation"] and vh==old["validation_batch_hashes"]
        model=tt.TinyTransformer(tt.ModelConfig(**config["model"])).float().cuda()
        initial=fingerprint(model)
        initial_val=tt.evaluate(model,val,config)
        seeds[str(seed)]=dict(initialization_sha256=initial,data=data,expected_batch_hashes=hashes,
            validation_batch_hashes=vh,initial_validation=initial_val)
        if seed==2026:
            assert initial==old["initialization_sha256"]==historical_a["initialization_sha256"]
            assert data==old["data"]==historical_a["dataset"]
            assert hashes==old["expected_batch_hashes"]==historical_a["expected_batch_hashes"]
            assert initial_val==read(BASELINE/"validation.json")["0"]==read(SOURCE/ADAM_LABEL/"validation.json")["0"]
        del model;gc.collect();torch.cuda.empty_cache()
    # Audit retained complete metrics, not just report prose, before seed reuse.
    qrows=[json.loads(line) for line in (BASELINE/"solves.jsonl").read_text().splitlines()]
    assert len(qrows)==3072
    for row in qrows:
        assert row["converged"] and not row["reference_used"] and not row["fallback_used"]
        assert row["selection_semantics"]==POLICY["name"] and not row["selection_certified"]
        assert min(row["alpha_lower"])>0 and row["dual_lower"]>0
        assert row["conservative_normalized_gap"]<=3e-5 and row["signed_normalized_gap"]>=-1e-10
        assert row["normalized_horizontal_residual"]<=1e-10 and row["spectral_excess"]<=1e-12
    for directory in (BASELINE,SOURCE/ADAM_LABEL):
        records=[json.loads(line) for line in (directory/"metrics.jsonl").read_text().splitlines()]
        assert len(records)==512 and [r["batch_sha256"] for r in records]==seeds["2026"]["expected_batch_hashes"]
        assert sorted(map(int,read(directory/"validation.json")))==list(range(0,513,32))
    manifest=dict(seeds=list(SEEDS),new_seeds=list(SEEDS[1:]),base=base,seed_contracts=seeds,
        source_sha256=sources,policy=POLICY,reused_seed2026=dict(adamw=str(SOURCE/ADAM_LABEL),qso=str(BASELINE)),
        reuse_verified=True,preflight_job=os.environ["SLURM_JOB_ID"],
        analysis=dict(unit="seed",primary="Delta_last3 at steps 448,480,512",secondary="Delta_final",
            confidence="paired Student-t, n=5 df=4, two-sided 95%",missing="classification D; no complete-case inference",
            gate="both means negative; >=4/5 last3 negative; last3 CI upper<0; zero QSO failures"),
        run_policy=dict(new_seeds=[2027,2028,2029,2030],retuning=False,checkpoint_step=256,
            continuation_steps=3,checkpoint_each_new_qso_seed=True,warm_excluded_steps=5,
            one_gpu_per_job=True,no_reference=True))
    write(root/"manifest.json",manifest)
    print("PREFLIGHT_PASSED",root,flush=True)


def run(root,manifest,seed,method):
    check_sources(manifest)
    config=configuration(manifest["base"],seed,method)
    output=root/f"seed-{seed}"/method;output.mkdir(parents=True,exist_ok=False)
    tt.seed_everything(seed,False);torch.set_default_dtype(torch.float32)
    train,val,data,hashes,vh=data_contract(config)
    contract=manifest["seed_contracts"][str(seed)]
    assert data==contract["data"] and hashes==contract["expected_batch_hashes"] and vh==contract["validation_batch_hashes"]
    model=tt.TinyTransformer(tt.ModelConfig(**config["model"])).float().cuda()
    opts=tt.Optimizers(model,method,config)
    initial=fingerprint(model);assert initial==contract["initialization_sha256"]
    if method=="qso":
        assert len(opts.paired.pairs)==6 and opts.paired.record_diagnostics and not opts.paired.record_cast_diagnostics
        assert all(g["momentum_dtype"]==torch.float32 for g in opts.paired.param_groups)
    provenance=dict(seed=seed,method=method,config=config,data=data,initialization_sha256=initial,
        expected_batch_hashes=hashes,validation_batch_hashes=vh,clipping="none",
        parameter_count=sum(p.numel() for p in model.parameters()),policy=POLICY if method=="qso" else None,
        source_sha256=manifest["source_sha256"],hostname=socket.gethostname(),job_id=os.environ["SLURM_JOB_ID"],
        python=sys.version,python_executable=sys.executable,torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),allocated_cpus=os.environ.get("SLURM_CPUS_PER_TASK"),
        allocated_memory_mb=os.environ.get("SLURM_MEM_PER_NODE"),output_path=str(output),
        diagnostics=True,cast_diagnostics=False)
    write(output/"provenance.json",provenance)
    metadata=dict(data,research_admission=dict(POLICY,source_sha256=manifest["source_sha256"]))
    records=[];rows=[];evaluations={};resume=None;active=dict(step=0,pair=None)
    last_inputs=None;pair_seconds=0.;peak=0;wall=time.perf_counter()
    def before(pair,u,d,a,lam):
        nonlocal last_inputs
        active["pair"]=pair.name;last_inputs=(u,d,a,lam)
    def callback(pair,u,d,a,lam,result):
        nonlocal pair_seconds
        row=dict(compact_metrics(result),step=active["step"],pair=pair.name,seconds=result.seconds)
        if not result.converged:
            failed=fixture(u,d,a,lam,result);failed.update(metadata=dict(active,seed=seed))
            torch.save(failed,output/"failed_pair.pt")
            write(output/"smooth_failure.json",dict(row,reason=result.reason,history=result.history,
                actions=result.actions,evaluations=result.evaluations))
        else:
            rows.append(row);pair_seconds+=result.seconds
            solve_log.write(json.dumps(clean(row),allow_nan=False)+"\n")
    try:
        with (output/"solves.jsonl").open("w",buffering=1) as solve_log,(output/"metrics.jsonl").open("w",buffering=1) as log:
            signal.alarm(config["step_timeout_seconds"])
            evaluations[0]=tt.evaluate(model,val,config);signal.alarm(0)
            assert evaluations[0]==contract["initial_validation"]
            write(output/"validation.json",evaluations)
            torch.cuda.reset_peak_memory_stats()
            for step in range(512):
                active.update(step=step,pair=None);last_inputs=None;pair_seconds=0.
                before_model=fingerprint(model)
                counters=[opts.paired.state.get(p.up,{}).get("step",0) for p in opts.paired.pairs] if opts.paired else []
                signal.alarm(config["step_timeout_seconds"])
                try:
                    if method=="qso":
                        with research_boundary(opts.paired,callback,before,profile=True):
                            record=tt.train_step(model,opts,train,config,step)
                    else:record=tt.train_step(model,opts,train,config,step)
                except Exception:
                    if method=="qso":
                        assert fingerprint(model)==before_model,"failed step partially committed model"
                        assert counters==[opts.paired.state.get(p.up,{}).get("step",0) for p in opts.paired.pairs]
                    raise
                finally:signal.alarm(0)
                assert record["batch_sha256"]==hashes[step]
                if not np.isfinite(record["loss"]) or record["loss"]>config["divergence_loss_limit"]:
                    raise AdmissionFailure("predeclared_loss_divergence")
                record["pair_solver_seconds"]=pair_seconds
                peak=max(peak,torch.cuda.max_memory_allocated())
                if (step+1)%32==0:
                    signal.alarm(config["step_timeout_seconds"])
                    loss=tt.evaluate(model,val,config);signal.alarm(0)
                    if not np.isfinite(loss) or loss>config["divergence_loss_limit"]:
                        raise AdmissionFailure("validation_nonfinite_or_divergent")
                    evaluations[step+1]=loss;record["validation_loss"]=loss
                    write(output/"validation.json",evaluations)
                    print("VALIDATION",seed,method,step+1,loss,flush=True)
                records.append(record);log.write(json.dumps(clean(record),allow_nan=False)+"\n")
                if method=="qso" and step+1==128:
                    gate=online_gate(rows);write(output/"online_gate_128.json",gate)
                if method=="qso" and step+1==256:
                    tt.save_checkpoint(output/"checkpoint.pt",model,opts,config,metadata)
                if method=="qso" and step+1==259:
                    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
                    other=tt.TinyTransformer(tt.ModelConfig(**config["model"])).float().cuda()
                    other_opts=tt.Optimizers(other,method,config)
                    assert tt.load_checkpoint(output/"checkpoint.pt",other,other_opts,config,metadata)==256
                    replay_rows=[]
                    def replay_callback(pair,u,d,a,lam,result):
                        replay_rows.append(dict(compact_metrics(result),pair=pair.name))
                    for j in range(256,259):
                        signal.alarm(config["step_timeout_seconds"])
                        with research_boundary(other_opts.paired,replay_callback,profile=False):
                            replay=tt.train_step(other,other_opts,train,config,j)
                        signal.alarm(0)
                        assert replay["loss"]==records[j]["loss"] and replay["batch_sha256"]==hashes[j]
                        for x,y in zip(replay_rows[-6:],rows[j*6:(j+1)*6]):
                            for key in ("pair","selection_semantics","selection_certified","newton_iterations",
                                        "cg_iterations","line_trials","alpha_lower","conservative_normalized_gap",
                                        "rcond_sides","spectral_excess","normalized_horizontal_residual"):
                                assert x[key]==y[key],"checkpoint admission differs: "+key
                    exact_tree(model.state_dict(),other.state_dict());exact_tree(opts.state_dict(),other_opts.state_dict())
                    resume=dict(passed=True,exact=True,checkpoint_step=256,replayed_completed_steps=[257,258,259],
                                checkpoint_bytes=(output/"checkpoint.pt").stat().st_size)
                    write(output/"resume.json",resume)
                    del other,other_opts;gc.collect();torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
                    torch.cuda.reset_peak_memory_stats()
            assert sorted(evaluations)==list(range(0,513,32))
            if method=="qso":assert len(rows)==3072 and resume and resume["exact"]
            summary=dict(status="passed",seed=seed,method=method,completed_steps=512,tokens=1048576,
                **quality(evaluations,records),total_wall_seconds=time.perf_counter()-wall,
                measured_step_seconds=sum(r["step_seconds"] for r in records),
                optimizer_seconds=sum(r["optimizer_seconds"] for r in records),peak_cuda_bytes=peak,
                solver=statistics(rows,records) if method=="qso" else {},resume=resume,warm_excluded_steps=5,
                warm={k:tt.summarize([r[k] for r in records[5:]]) for k in
                    ("step_seconds","pair_solver_seconds","optimizer_seconds","forward_backward_seconds")})
            summary["tokens_per_second"]=summary["tokens"]/summary["measured_step_seconds"]
            summary["optimizer_fraction"]=summary["optimizer_seconds"]/summary["measured_step_seconds"]
            write(output/"summary.json",summary)
    except Exception as error:
        signal.alarm(0)
        if last_inputs is not None and not (output/"failed_pair.pt").exists():
            torch.save(dict(fixture(*last_inputs),metadata=dict(active,seed=seed)),output/"failed_pair.pt")
        summary=dict(status="failed",seed=seed,method=method,completed_steps=len(records),tokens=len(records)*2048,
            active=active,exception=repr(error),traceback=traceback.format_exc(),solver=statistics(rows,records),
            total_wall_seconds=time.perf_counter()-wall,peak_cuda_bytes=peak)
        write(output/"failure.json",summary);print("FROZEN_SEED_FAILURE",clean(summary),flush=True)
    finally:
        signal.alarm(0);check_sources(manifest)
        del model,opts;gc.collect();torch.cuda.empty_cache()
    print("RUN_END",seed,method,summary["status"],flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--root",type=Path);parser.add_argument("--seed",type=int)
    args=parser.parse_args()
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE and torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0)
    signal.signal(signal.SIGALRM,timeout)
    if args.preflight:preflight();return
    if args.seed not in SEEDS[1:]:raise SystemExit("only the four declared NEW seeds may train")
    manifest=read(args.root/"manifest.json");assert manifest["reuse_verified"]
    results={m:run(args.root,manifest,args.seed,m) for m in ("adamw","qso")}
    path=args.root/f"seed-{args.seed}"
    verify_pair(read(path/"adamw/provenance.json"),read(path/"qso/provenance.json"))
    write(path/"paired_gate.json",dict(passed=True,seed=args.seed,results={m:r["status"] for m,r in results.items()}))


if __name__=="__main__":main()
