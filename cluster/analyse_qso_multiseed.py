"""Seed-level analysis and standalone figures, after all declared runs finish."""
import os
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM required for experiment-log aggregation and plotting")
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from benchmarks.multiseed import SEEDS, analyse, validation_metrics, verify_pair


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
def dist(values):
    if not values:return {}
    return dict(mean=statistics.mean(values),median=statistics.median(values),
                p95=float(np.percentile(values,95)),max=max(values),min=min(values))


def numerical(rows):
    normal=[r for r in rows if r["selection_semantics"]=="primary_epsilon_lmo_full_rank"]
    below=[r for r in normal if r["below_old_guard"]]
    for r in normal:
        assert r["converged"] and not r["reference_used"] and not r["fallback_used"]
        assert not r["selection_certified"] and min(r["alpha_lower"])>0 and r["dual_lower"]>0
        assert r["conservative_normalized_gap"]<=3e-5 and r["signed_normalized_gap"]>=-1e-10
        assert r["normalized_horizontal_residual"]<=1e-10 and r["spectral_excess"]<=1e-12
    return dict(pair_solves=len(rows),below_old_guard_count=len(below),
        below_fraction=len(below)/len(rows) if rows else 0,
        below_by_pair=dict(Counter(r["pair"] for r in below)),
        below_steps_by_pair={p:sorted(set(r["step"] for r in below if r["pair"]==p)) for p in sorted(set(r["pair"] for r in below))},
        below_u=sum(r["rcond_sides"][0]<=1e-4 for r in normal),
        below_d=sum(r["rcond_sides"][1]<=1e-4 for r in normal),
        below_both=sum(max(r["rcond_sides"])<=1e-4 for r in normal),
        any_evaluation_below=sum(r["any_evaluation_below_old_guard"] for r in rows),
        min_rcond=min((min(r["rcond_sides"]) for r in normal),default=None),
        min_evaluated_rcond=min((r["minimum_evaluated_rcond"] for r in normal),default=None),
        min_alpha_lower=min((min(r["alpha_lower"]) for r in normal),default=None),
        min_sigma_eta=min((min(r["sigma_min_over_eta"]) for r in normal),default=None),
        max_conservative_gap=max((r["conservative_normalized_gap"] for r in normal),default=None),
        gap=dist([r["conservative_normalized_gap"] for r in normal]),
        max_horizontal=max((r["normalized_horizontal_residual"] for r in rows),default=None),
        max_spectral_excess=max((r["spectral_excess"] for r in rows),default=None),
        max_proxy_distance=max((r["feasible_proxy_distance"] for r in normal),default=None),
        newton_histogram=dict(Counter(r["newton_iterations"] for r in rows)),
        newton_bins={str(i):sum(r["newton_iterations"]==i for r in rows) for i in (0,1,2)}|
                    {"3+":sum(r["newton_iterations"]>=3 for r in rows)},
        cg=dist([r["cg_iterations"] for r in rows]),line_trials=dist([r["line_trials"] for r in rows]),
        cg_termination=dict(Counter(x for r in rows for x in r["cg_termination"])),
        max_cg_per_action=max((r["max_cg_queries_per_action"] for r in rows),default=None),
        minimum_curvature=min((r["minimum_used_curvature"] for r in rows if r["minimum_used_curvature"] is not None),default=None),
        maximum_slope=max((r["maximum_slope"] for r in rows if r["maximum_slope"] is not None),default=None),
        smooth_evaluations=sum(r["smooth_evaluations"] for r in rows),
        svd_matrices=sum(r["svd_evaluations"] for r in rows),
        rank_posterior_seconds=sum(r["rank_seconds"] for r in rows),
        value_posterior_seconds=sum(r["value_seconds"] for r in rows),
        work_regions=[dict(first_step=lo,last_step=hi-1,
            cg=dist([r["cg_iterations"] for r in rows if lo<=r["step"]<hi]),
            newton=dist([r["newton_iterations"] for r in rows if lo<=r["step"]<hi]),
            line=dist([r["line_trials"] for r in rows if lo<=r["step"]<hi]))
            for lo,hi in ((0,64),(64,128),(128,256),(256,512))],
        cpu_reference_calls=sum(r["reference_used"] for r in rows),
        optimizer_fallbacks=sum(r["fallback_used"] for r in rows),
        all_returned_pairs_certified=True)


def figures(pairs,output):
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "savefig.bbox":"tight","pdf.fonttype":42,"svg.fonttype":"none"})
    def save(fig,name):
        fig.tight_layout();fig.savefig(output/(name+".svg"));fig.savefig(output/(name+".pdf"))
        fig.savefig(output/(name+".png"),dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
    for ax,p in zip(axes.flat,pairs):
        for m,color in (("adamw","#3366aa"),("qso","#bf5b23")):
            vals=p["validation"][m];steps=sorted(map(int,vals))
            ax.plot([s*2048 for s in steps],[vals[str(s)] for s in steps],"o-",markersize=3,color=color,
                    label="AdamW" if m=="adamw" else "Research QSO")
        ax.set_title(f"Seed {p['seed']}"+(" (failed)" if p["qso_status"]!="passed" else ""));ax.grid(alpha=.2)
        ax.set_xlabel("Training input tokens");ax.set_ylabel("Validation loss")
    axes.flat[0].legend();axes.flat[-1].axis("off");save(fig,"validation_by_seed")
    fig,ax=plt.subplots(figsize=(8,4))
    for p in pairs:
        a,q=p["validation"]["adamw"],p["validation"]["qso"]
        steps=sorted(set(map(int,a))&set(map(int,q)))
        ax.plot([s*2048 for s in steps],[q[str(s)]-a[str(s)] for s in steps],"o-",markersize=3,label=str(p["seed"]))
    ax.axhline(0,color="black",lw=.8);ax.grid(alpha=.2);ax.legend(title="Paired seed")
    ax.set(xlabel="Training input tokens",ylabel="QSO − AdamW validation loss",title="Paired differences at equal tokens")
    save(fig,"paired_curves")
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,key,title in zip(axes,("final","last3"),("Final checkpoint","Mean of final three checkpoints")):
        for i,p in enumerate(pairs):
            if p.get("differences") is not None:
                d=p["differences"][key];ax.bar(i,d,color="#bf5b23" if d<0 else "#3366aa")
            else:ax.text(i,0,"Unavailable",rotation=90,ha="center",va="bottom")
        ax.axhline(0,color="black",lw=.8);ax.set_xticks(range(5),[str(s) for s in SEEDS]);ax.set_title(title)
        ax.set_xlabel("Seed");ax.set_ylabel("QSO − AdamW validation loss");ax.grid(axis="y",alpha=.2)
    save(fig,"paired_endpoints")
    common=set(range(0,513,32))
    for p in pairs:common &= set(map(int,p["validation"]["qso"]))&set(map(int,p["validation"]["adamw"]))
    steps=sorted(common)
    matrix=np.array([[p["validation"]["qso"][str(s)]-p["validation"]["adamw"][str(s)] for s in steps] for p in pairs])
    mean=matrix.mean(0);sd=matrix.std(0,ddof=1)
    fig,ax=plt.subplots(figsize=(8,4));tokens=np.array(steps)*2048
    ax.plot(tokens,mean,"o-",markersize=4,color="#804b96",label="Mean paired difference (five seeds)")
    ax.fill_between(tokens,mean-sd,mean+sd,color="#804b96",alpha=.18,label="±1 across-seed sample SD (descriptive)")
    ax.axhline(0,color="black",lw=.8);ax.grid(alpha=.2);ax.legend()
    ax.set(xlabel="Training input tokens",ylabel="QSO − AdamW validation loss",title="Across-seed descriptive uncertainty; checkpoints are not independent")
    save(fig,"mean_paired_curve")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("root",type=Path);args=parser.parse_args()
    manifest=read(args.root/"manifest.json")
    for path,digest in manifest["source_sha256"].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    pairs=[];pooled=[];ratios=[]
    for seed in SEEDS:
        locations=manifest["reused_seed2026"] if seed==2026 else {m:str(args.root/f"seed-{seed}"/m) for m in ("adamw","qso")}
        summaries={};validations={};provs={};costs={}
        for method,path in locations.items():
            directory=Path(path)
            # Missing runs must not be silently treated as observed failures.
            sp=directory/"summary.json";fp=directory/"failure.json"
            if not sp.exists() and not fp.exists():raise RuntimeError(f"declared run not finished: {seed}/{method}")
            summaries[method]=read(sp if sp.exists() else fp)
            validations[method]=read(directory/"validation.json")
            prov=read(directory/"provenance.json")
            if seed==2026:
                prov=dict(prov,seed=seed,data=prov.get("data",prov.get("dataset")),
                    validation_batch_hashes=manifest["seed_contracts"][str(seed)]["validation_batch_hashes"])
            provs[method]=prov
            records=lines(directory/"metrics.jsonl")
            expected=manifest["seed_contracts"][str(seed)]["expected_batch_hashes"]
            assert [r["batch_sha256"] for r in records]==expected[:len(records)]
            s=summaries[method]
            costs[method]=dict(total_wall_seconds=s["total_wall_seconds"],
                measured_step_seconds=sum(r["step_seconds"] for r in records),
                tokens_per_second=len(records)*2048/sum(r["step_seconds"] for r in records) if records else None,
                optimizer_fraction=sum(r["optimizer_seconds"] for r in records)/sum(r["step_seconds"] for r in records) if records else None,
                peak_cuda_bytes=s["peak_cuda_bytes"],warm={k:dist([r[k] for r in records[5:]]) for k in
                    ("step_seconds","optimizer_seconds","pair_solver_seconds","forward_backward_seconds")},
                cold_first_step=records[0]["step_seconds"] if records else None,
                max_update_parameter_ratio=max((r["update_parameter_ratio"] for r in records),default=None),
                final_scales={k:records[-1][k] for k in ("gradient_rms","update_rms","parameter_rms","update_parameter_ratio")} if records else None)
        verify_pair(provs["adamw"],provs["qso"])
        # Installed numerical environment must match the reused validated seed.
        baseline=read(Path(manifest["reused_seed2026"]["qso"])/"provenance.json")
        for p in provs.values():
            for k in ("python_executable","torch","numpy","cuda","gpu"):
                assert p[k]==baseline[k],f"environment provenance mismatch: {seed}/{k}"
        rows=lines(Path(locations["qso"])/"solves.jsonl")
        num=numerical(rows);pooled.extend(rows)
        failure=summaries["qso"] if summaries["qso"]["status"]!="passed" else None
        num.update(numerical_failure_count=int(failure is not None),
            rank_ambiguity_count=int(bool(failure and "rank_ambiguous" in failure.get("exception",""))))
        p=dict(seed=seed,locations=locations,adamw_status=summaries["adamw"]["status"],
            qso_status=summaries["qso"]["status"],validation=validations,cost=costs,numerical=num,
            checkpoint=summaries["qso"].get("resume"),failure=failure,
            completed_steps={m:s.get("completed_steps",s.get("steps")) for m,s in summaries.items()},
            initialization_sha256=provs["qso"]["initialization_sha256"],
            permutation_sha256=provs["qso"]["data"]["train"]["order_sha256"],
            job_ids={m:provs[m]["job_id"] for m in provs})
        if p["qso_status"]==p["adamw_status"]=="passed":
            qm,am=validation_metrics(validations["qso"]),validation_metrics(validations["adamw"])
            p.update(quality=dict(qso=qm,adamw=am),differences={k:qm[k]-am[k] for k in qm})
            ratio=dict(seed=seed,wall=costs["qso"]["total_wall_seconds"]/costs["adamw"]["total_wall_seconds"],
                measured_step=costs["qso"]["measured_step_seconds"]/costs["adamw"]["measured_step_seconds"],
                warm_median=costs["qso"]["warm"]["step_seconds"]["median"]/costs["adamw"]["warm"]["step_seconds"]["median"])
            p["cost_ratios"]=ratio;ratios.append(ratio)
        else:p["differences"]=None
        pairs.append(p)
    result=dict(manifest=str(args.root/"manifest.json"),pairs=pairs,analysis=analyse(pairs),
        pooled_numerical=numerical(pooled),cost_ratio_distribution={k:dist([r[k] for r in ratios]) for k in ("wall","measured_step","warm_median")})
    result["pooled_numerical"].update(numerical_failure_count=sum(p["numerical"]["numerical_failure_count"] for p in pairs),
        rank_ambiguity_count=sum(p["numerical"]["rank_ambiguity_count"] for p in pairs))
    write(args.root/"analysis.json",result);write("cluster/qso_multiseed_summary.json",result)
    figures(pairs,Path("docs/figures/qso_multiseed_512"))
    print("ANALYSIS",json.dumps(result["analysis"],indent=2),flush=True)
    print("FIGURES docs/figures/qso_multiseed_512",flush=True)


if __name__=="__main__":main()
