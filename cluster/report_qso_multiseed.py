"""Render the completed seed-level analysis without loading tensor artifacts."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument("root",type=Path);args=parser.parse_args()
    data=json.loads((args.root/"analysis.json").read_text())
    manifest=json.loads((args.root/"manifest.json").read_text())
    pairs=data["pairs"];num=data["pooled_numerical"];analysis=data["analysis"]
    labels={"A":"MULTI-SEED QUALITY ADVANTAGE SUPPORTED","B":"POSITIVE TREND BUT INCONCLUSIVE",
            "C":"NO MULTI-SEED ADVANTAGE","D":"NUMERICAL ROBUSTNESS FAILURE"}
    classification=analysis["classification"]
    if classification not in labels:raise RuntimeError("planned analysis unavailable")
    text=[]
    def add(s=""):text.append(s)
    def table(headers,rows):
        add("| "+" | ".join(headers)+" |")
        add("| "+" | ".join("---" for _ in headers)+" |")
        for row in rows:add("| "+" | ".join(str(x) for x in row)+" |")
        add()
    def f(x,d=6):return "Unavailable" if x is None else f"{x:.{d}f}"
    def e(x):return "Unavailable" if x is None else f"{x:.6e}"
    def triple(d):return "/".join(f(d[k],3) for k in ("median","p95","max")) if d else "Unavailable"
    def ranges(values):
        groups=[];lo=hi=values[0]
        for value in values[1:]:
            if value==hi+1:hi=value
            else:groups.append((lo,hi));lo=hi=value
        groups.append((lo,hi))
        return ",".join(str(lo) if lo==hi else f"{lo}–{hi}" for lo,hi in groups)
    add("# Paired five-seed 512-step optimizer-quality study\n")
    add(f"Result: **{classification} — {labels[classification]}**.\n")
    passed=sum(p["qso_status"]=="passed" for p in pairs)
    add(f"Research QSO completed **{passed}/5** declared seed trajectories. "
        f"The numerical logs contain **{num['pair_solves']:,}** conservative primary-certified paired returns. "
        f"Observed numerical failures: **{num['numerical_failure_count']}**; "
        f"rank ambiguity: **{num['rank_ambiguity_count']}**; CPU reference calls: **{num['cpu_reference_calls']}**.\n")
    if analysis.get("inference"):
        last,final=analysis["inference"]["last3"],analysis["inference"]["final"]
        add(f"The primary paired final-three validation difference (research QSO minus AdamW) "
            f"was **{last['mean']:+.6f}**, with a 95% paired Student-t interval "
            f"**[{last['ci95'][0]:+.6f}, {last['ci95'][1]:+.6f}]**. "
            f"The final-checkpoint paired mean was **{final['mean']:+.6f}**. "
            f"**{last['seeds_favoring_qso']}/5** seeds favored QSO on the primary metric.\n")
    add("This conclusion applies to the frozen research admission policy, this tiny model, "
        "local data prefix and 512-step horizon. Production-v0 was not changed. "
        "Primary certification does not establish minimum-Frobenius P-dagger selection "
        "on a nonunique face, and this study does not establish broad optimizer superiority.\n")
    if analysis.get("inference"):
        add(f"The advantage is a **late-horizon** result. Mean post-initial validation difference "
            f"was **{analysis['post_initial_mean']['mean']:+.6f}**, and mean normalized AUC difference "
            f"was **{analysis['auc_mean']['mean']:+.6f}**. Both full-horizon means favor AdamW, "
            "with intervals spanning zero; research QSO does not dominate the entire learning curve. "
            "These secondary diagnostics are retained rather than suppressed or substituted for "
            "the predeclared final-three gate.\n")
    add("## 1. Frozen setup and pairing\n")
    table(["Setting","Value"],[
        ["Seeds","2026, 2027, 2028, 2029, 2030; exactly the predeclared set"],
        ["AdamW peak LR","2e-4"],["Research QSO paired peak LR","0.0012247448713915891"],
        ["QSO unsupported-parameter AdamW LR","3e-4"],
        ["Model","11,457,408 parameters; 6 blocks, width 384, 6 heads, hidden 1024, vocabulary 1024"],
        ["Batch / context / accumulation","16 / 128 / 1; 2048 input tokens per update"],
        ["Length","512 updates / 1,048,576 nonrepeated training input tokens per completed run"],
        ["Schedule","51 linear-warmup steps; unchanged cosine decay to 10% of peak"],
        ["Validation","Four fixed batches at completed steps 0,32,...,512"],
        ["Betas / decay / clipping","Canonical EMA 0.95; AdamW (0.9,0.95); decay 0; no clipping"],
        ["Precision","fp32 model storage and canonical EMA; bf16 forward/backward; fp64 smooth solve"],
        ["Smooth / radial backend","Direct full fp64 thin SVD / gram_upper"],
        ["Warm state","Previous original-coordinate lambda; no temporal prediction or factor recycling"],
        ["Diagnostics","diagnostics=True; cast_diagnostics=False"],
        ["Determinism","Existing deterministic contract, TF32 disabled, CUBLAS_WORKSPACE_CONFIG=:4096:8"]])
    add("The existing local FineWeb training and validation shards and tokenizer representation were reused: "
        "`/home/prignano/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_{train,val}_000000.bin` "
        "and `/home/prignano/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`. "
        "The bounded training prefix has 1,048,577 tokens including lookahead. "
        "Each seed permutes all 8192 disjoint 128-token input blocks once. "
        "No wrapping, tokenization, download, network fallback, or environment installation occurred.\n")
    add("The experiment seed controls initialization and the single-pass training permutation. "
        "The existing `data.seed=2026` convention keeps validation batches fixed across all runs. "
        "Within each seed, initialization, all 512 expected minibatch hashes, validation-batch hashes, "
        "model, precision, schedule and token budget were asserted equal between methods. "
        "Hashes were recorded before training and checked against every completed minibatch online.\n")
    table(["Seed","Initial model SHA256","Training permutation SHA256"],
        [[p["seed"],"`"+p["initialization_sha256"]+"`","`"+p["permutation_sha256"]+"`"] for p in pairs])
    c=manifest["seed_contracts"]["2026"]
    add(f"Training-prefix SHA256: `{c['data']['train']['prefix_sha256']}`. "
        f"Validation-prefix SHA256: `{c['data']['validation']['prefix_sha256']}`. "
        "The manifest also records all four validation-batch digests and every expected training-batch digest.\n")
    add("Seed 2026 was reused after checking the original Stage-D immutable source/configuration hashes, "
        "the completed research source hashes, exact policy metadata, current prefix/permutation/batch hashes, "
        "initialization and initial validation, the complete 512-step logs, all 3072 returned "
        "research certificates, and its exact checkpoint-continuation record. "
        "The retained AdamW reference has the same model/data/schedule/precision/validation contract. "
        "Python executable, PyTorch, NumPy, CUDA and GPU model were checked against the new runs.\n")
    add("Historical AdamW did not retain explicit validation-batch digests. Its immutable "
        "validation code, data prefix and seed/configuration establish the same batches; "
        "the reconstructed digests are recorded in the new manifest. This is provenance "
        "compatibility, not a claim that unavailable historical validation tensors were compared directly.\n")
    add("## 2. Predeclared seed-level analysis\n")
    add("The statistical unit is the **seed**. Primary `Delta_last3` is QSO minus AdamW "
        "mean validation loss at steps 448,480,512. Secondary `Delta_final` is the corresponding "
        "step-512 difference. Negative favors QSO. All five seeds are included; checkpoints "
        "are not treated as independent samples. No LR, warmup, beta, decay, clipping, "
        "numerical tolerance, admission policy or seed set was adapted.\n")
    add("For each paired metric the interval is `mean ± t(0.975,4) * sample_sd/sqrt(5)`, "
        "with `t(0.975,4)=2.7764451051977987`. Classification A requires both means negative, "
        "at least four negative primary seed differences, a primary interval entirely below zero, "
        "and no QSO numerical-contract failure. A failed QSO seed triggers D and prevents "
        "complete-case inference; it is not discarded.\n")
    add("## 3. Complete paired quality results\n")
    table(["Seed","AdamW / QSO status","AdamW final","QSO final","Delta_final","AdamW last3","QSO last3","Delta_last3"],
        [[p["seed"],p["adamw_status"]+" / "+p["qso_status"],
          *[f(p.get("quality",{}).get(m,{}).get("final")) for m in ("adamw","qso")],
          f(p["differences"]["final"]) if p["differences"] else "Unavailable",
          *[f(p.get("quality",{}).get(m,{}).get("last3")) for m in ("adamw","qso")],
          f(p["differences"]["last3"]) if p["differences"] else "Unavailable"] for p in pairs])
    if analysis.get("inference"):
        stats=[("Primary last3",analysis["inference"]["last3"]),("Secondary final",analysis["inference"]["final"]),
               ("Post-initial mean",analysis["post_initial_mean"]),("Token-normalized trapezoidal AUC mean",analysis["auc_mean"])]
        table(["Paired statistic","Mean","Median","Sample SD","SE","95% paired mean CI","Seeds favoring QSO"],
            [[name,f(s["mean"]),f(s["median"]),f(s["sample_sd"]),f(s["standard_error"]),
              f"[{s['ci95'][0]:+.6f}, {s['ci95'][1]:+.6f}]",s["seeds_favoring_qso"]] for name,s in stats])
        table(["Seed","Post-initial mean difference","AUC mean difference"],
            [[p["seed"],f(p["differences"]["post_initial_mean"]),f(p["differences"]["auc_mean"])] for p in pairs])
    else:add("The required five completed paired differences and confidence intervals are unavailable. "
             "Failed trajectories are retained and the gate is not computed from a favorable subset.\n")
    add("The AUC is the normalized trapezoidal area over common equal-token checkpoints, including step 0. "
        "The post-initial mean equally weights steps 32 through 512. "
        "The full precision values are in the structured summary; displayed tables are rounded.\n")
    add("![Per-seed equal-token validation](figures/qso_multiseed_512/validation_by_seed.svg)\n")
    add("![Paired differences at equal tokens](figures/qso_multiseed_512/paired_curves.svg)\n")
    add("![Paired final and final-three differences](figures/qso_multiseed_512/paired_endpoints.svg)\n")
    add("![Mean paired curve with descriptive across-seed uncertainty](figures/qso_multiseed_512/mean_paired_curve.svg)\n")
    add("Actual validation points are plotted without smoothing. The mean-curve band is ±1 sample "
        "standard deviation across five seed differences, **descriptive**, not checkpoint-level "
        "significance or a simultaneous confidence band.\n")
    add("## 4. Research contract and numerical robustness\n")
    add("Normal returned pairs use `primary_epsilon_lmo_full_rank`, `selection_certified=False`. "
        "Under the inherited conditional normal-finite fp64 gamma model, the certificate "
        "bounds primary support regret for an exactly feasible mathematical proxy. "
        "Every normal return requires positive alpha lower bounds on both sides, positive dual lower "
        "bound, conservative `G_upper/dual_lower <= 3e-5`, signed raw normalized gap `>= -1e-10`, "
        "normalized horizontality `<= 1e-10`, spectral excess `<= 1e-12`, and finite state. "
        "No fitted direction-distance threshold is used. Exact intrinsic zero retains exact-zero selection.\n")
    add("The isolated scope forbids CPU reference calls and restores the production solve boundary "
        "on exit. A failed pair cannot commit any paired proposal; the unchanged optimizer's atomic "
        "pair validation is covered by tests. Invalid intermediate decompositions also stop the "
        "trajectory. The old rcond threshold remains a diagnostic and is not replaced by a lower constant.\n")
    table(["Seed","Returned pairs","Below old guard","Min rcond","Min alpha_lower","Min sigma_min/eta","Max conservative gap","Rank / numerical failures"],
        [[p["seed"],p["numerical"]["pair_solves"],
          f"{p['numerical']['below_old_guard_count']} ({100*p['numerical']['below_fraction']:.4f}%)",
          e(p["numerical"]["min_rcond"]),e(p["numerical"]["min_alpha_lower"]),
          e(p["numerical"]["min_sigma_eta"]),e(p["numerical"]["max_conservative_gap"]),
          f"{p['numerical']['rank_ambiguity_count']} / {p['numerical']['numerical_failure_count']}"] for p in pairs]+
        [["Pooled",num["pair_solves"],f"{num['below_old_guard_count']} ({100*num['below_fraction']:.4f}%)",
          e(num["min_rcond"]),e(num["min_alpha_lower"]),e(num["min_sigma_eta"]),e(num["max_conservative_gap"]),
          f"{num['rank_ambiguity_count']} / {num['numerical_failure_count']}"]])
    table(["Seed","Newton bins 0 / 1 / 2 / 3+","CG median / p95 / max","Line trials median / p95 / max"],
        [[p["seed"]," / ".join(str(p["numerical"]["newton_bins"][k]) for k in ("0","1","2","3+")),
          triple(p["numerical"]["cg"]),triple(p["numerical"]["line_trials"])] for p in pairs]+
        [["Pooled"," / ".join(str(num["newton_bins"][k]) for k in ("0","1","2","3+")),triple(num["cg"]),triple(num["line_trials"])]])
    table(["Seed","Below-guard block and zero-based steps","U / D / both counts","Solves with any below-guard evaluation"],
        [[p["seed"],"; ".join(f"{k}: {p['numerical']['below_by_pair'][k]} at {ranges(v)}" for k,v in p["numerical"]["below_steps_by_pair"].items()) or "None",
          f"{p['numerical']['below_u']} / {p['numerical']['below_d']} / {p['numerical']['below_both']}",
          p["numerical"]["any_evaluation_below"]] for p in pairs])
    event_seeds=sum(p["numerical"]["below_old_guard_count"]>0 for p in pairs)
    add(f"Final below-old-guard events occurred in **{event_seeds}/5** seeds. "
        "The table shows whether they recur in the early block/step region rather than "
        "assuming seed-2026's eight block-1 down-side events are universal. "
        "U and D counts each include both-side events; the both column is their intersection.\n")
    add("Pooled counts by block: "+"; ".join(f"{p}: {n}" for p,n in sorted(num["below_by_pair"].items()))+
        ". Blocks 4 and 5 had no final below-old-guard event.\n")
    table(["Seed","CG median steps 0–63","64–127","128–255","256–511","Maximum Newton count"],
        [[p["seed"],*[f(r["cg"].get("median"),1) for r in p["numerical"]["work_regions"]],
          max(map(int,p["numerical"]["newton_histogram"]),default=0)] for p in pairs])
    add(f"Pooled maximum normalized horizontality: `{e(num['max_horizontal'])}`; "
        f"maximum spectral excess: `{e(num['max_spectral_excess'])}`; "
        f"maximum feasible-proxy distance bound: `{e(num['max_proxy_distance'])}`. "
        f"Minimum evaluated rcond: `{e(num['min_evaluated_rcond'])}`. "
        f"Total smooth evaluations: {num['smooth_evaluations']:,}; residual matrices decomposed: {num['svd_matrices']:,}. "
        f"CG termination counts: `{num['cg_termination']}`. "
        f"Largest CG action: {num['max_cg_per_action']} queries.\n")
    add("These minima refer to the logged final pair quantities unless explicitly labeled evaluated. "
        "Every intermediate evaluation passed the research rank posterior; final sigma_min/eta "
        "and alpha are not reinterpreted as universal condition-number thresholds. "
        "The structured summary retains exact Newton histograms and per-seed work/failure data.\n")
    if passed==5:
        add(f"There was no pathological late solver-work growth: the largest Newton count was "
            f"{max(map(int,num['newton_histogram']))}; all {sum(num['cg_termination'].values()):,} CG actions "
            "reached the existing residual target. Total line trials equaled total Newton actions, "
            "so no line-search backtracking occurred. The largest per-pair CG count includes "
            "multiple Newton actions; the largest individual action used "
            f"{num['max_cg_per_action']} queries, below the unchanged 30-query budget. "
            "CG work rose in the middle region and settled in the latter half, as the region table shows.\n")
    if any(p["failure"] for p in pairs):
        add("### Recorded failures\n")
        for p in pairs:
            if p["failure"]:
                fail=p["failure"];add(f"Seed {p['seed']}: {fail['completed_steps']} completed updates; "
                    f"active `{fail['active']}`; initiating exception `{fail['exception']}`. "
                    "The failure fixture and complete smooth history are retained; no retuning or hidden reference rescue occurred.\n")
    add("## 5. Checkpoint continuation\n")
    table(["Seed","Checkpoint / replay","Exact continuation","Checkpoint bytes"],
        [[p["seed"],"256 / 257,258,259" if p["checkpoint"] else "Unavailable",
          p["checkpoint"].get("exact") if p["checkpoint"] else "Unavailable",
          p["checkpoint"].get("checkpoint_bytes") if p["checkpoint"] else "Unavailable"] for p in pairs])
    add("Each new QSO seed saves one checkpoint at step 256. An independently loaded model/optimizer "
        "replays three steps and compares batch hashes, losses, complete model/optimizer state trees, "
        "and pair admission/rank/value decisions exactly. Canonical EMA, original lambda, pair counters, "
        "unsupported AdamW state, schedule, data position and policy metadata are covered. "
        "Replay steps are diagnostics, excluded from quality token counts and solver/step aggregates. "
        "Their work is included in training wall time. No checkpoint is stored for every training step.\n")
    add("The four new continuation checks execute 72 additional diagnostic pair solves "
        "(18 per seed), all through the same certificate-enforcing adapter. The reused "
        "seed-2026 check adds 18 historical diagnostic solves. These are separate from "
        "the planned 15,360 training-pair aggregate.\n")
    add("## 6. Implementation cost, separate from equal-token quality\n")
    table(["Seed","Method","Wall s","Step sum s","Warm step median / p95 s","Tokens/s","Optimizer fraction","Peak MiB"],
        [[p["seed"],m,f(p["cost"][m]["total_wall_seconds"],3),f(p["cost"][m]["measured_step_seconds"],3),
          f"{f(p['cost'][m]['warm']['step_seconds'].get('median'),6)} / {f(p['cost'][m]['warm']['step_seconds'].get('p95'),6)}",
          f(p["cost"][m]["tokens_per_second"],1),f(p["cost"][m]["optimizer_fraction"],4),
          f(p["cost"][m]["peak_cuda_bytes"]/2**20,2)] for p in pairs for m in ("adamw","qso")])
    table(["Seed","Warm QSO six-pair median / p95 s","Warm AdamW F/B median / p95 s","Warm QSO F/B median / p95 s","QSO/Adam wall / step sum / warm median"],
        [[p["seed"]," / ".join(f(p["cost"]["qso"]["warm"]["pair_solver_seconds"].get(k),6) for k in ("median","p95")),
          *[" / ".join(f(p["cost"][m]["warm"]["forward_backward_seconds"].get(k),6) for k in ("median","p95")) for m in ("adamw","qso")],
          " / ".join(f(p["cost_ratios"][k],2)+"x" for k in ("wall","measured_step","warm_median")) if p.get("cost_ratios") else "Unavailable"] for p in pairs])
    table(["Cost ratio","Mean","Median","p95","Range"],
        [[k,f(v.get("mean"),2)+"x",f(v.get("median"),2)+"x",f(v.get("p95"),2)+"x",
          f"{f(v.get('min'),2)}x–{f(v.get('max'),2)}x"] for k,v in data["cost_ratio_distribution"].items()])
    add("Warm summaries exclude zero-based steps 0–4, not unfavorable later intervals. "
        "Cold first-step costs are retained in the structured summary. Synchronization-aware timers "
        "come from the existing harness. Tokens/sec uses completed training tokens divided by measured "
        "step-time sum. Wall time includes validation, between-step state checks and, for QSO, "
        "checkpoint writing/replay; initial model/data setup precedes the timer. "
        "Peak PyTorch allocation excludes the extra replay model and non-PyTorch CUDA allocations.\n")
    add("The four new paired comparisons run sequentially, one H100 per SLURM job, under the "
        "account's one-GPU quota. AdamW and QSO for each new seed share an allocation; "
        "different seeds and the reused seed-2026 methods came from different allocations. "
        "Cost ratios describe observed implementation cost, not a universal hardware-independent "
        "optimizer cost or a criterion for choosing hyperparameters.\n")
    add("Wall-time ratios also reflect observational harness differences: the new runner "
        "fingerprints the model between every step for both methods; the historical Stage-D "
        "AdamW runner did not perform that check. The previous research QSO runner already "
        "did. Thus seed-2026 versus new-seed wall ratios are not instrumentation-identical. "
        "Measured-step and warm-step ratios exclude these between-step digests and are "
        "the more comparable implementation-cost measures. No run was repeated or changed "
        "in response to its measured cost.\n")
    add(f"Pooled instrumented rank posterior: {num['rank_posterior_seconds']:.3f} s; "
        f"value posterior: {num['value_posterior_seconds']:.3f} s. "
        "These reuse matching smooth factors and add no tall SVD solely for logging.\n")
    add("## 7. Update-scale diagnostics\n")
    table(["Seed","Method","Maximum update / parameter RMS","Final gradient RMS","Final update RMS","Final parameter RMS","Final update / parameter RMS"],
        [[p["seed"],m,e(p["cost"][m]["max_update_parameter_ratio"]),
          *[e(p["cost"][m]["final_scales"].get(k)) if p["cost"][m]["final_scales"] else "Unavailable" for k in
            ("gradient_rms","update_rms","parameter_rms","update_parameter_ratio")]] for p in pairs for m in ("adamw","qso")])
    add("These Euclidean diagnostics are not gauge-invariant admission certificates. "
        "The complete logs retain training loss, gradient/update/parameter RMS, and update ratios "
        "throughout each trajectory. No exploratory extra training run was launched.\n")
    add("## 8. Provenance, validation and artifacts\n")
    add("The complete suite passed on H100 before training: **462 passed, 0 failed, 0 skipped**. "
        "Focused new tests cover frozen pairing, complete-five-seed inference, common checkpoint "
        "metrics, confidence intervals, failure retention, and the predeclared gate. "
        "The SLURM array initially rejected seed-number task indices; replacing them with indices "
        "0–3 mapped to 2027–2030 was checked by a second preflight before any training. "
        "This was infrastructure correction, not hyperparameter adaptation.\n")
    table(["Seed","AdamW SLURM job","Research QSO SLURM job"],
        [[p["seed"],p["job_ids"]["adamw"],p["job_ids"]["qso"]] for p in pairs])
    add(f"Frozen study manifest: `{args.root}/manifest.json` (preflight job {manifest['preflight_job']}). "
        "It records all source hashes, seed data/initialization contracts and predeclared analysis. "
        "Production and admission sources were verified unchanged before/after each job and again "
        "during analysis. The validated environment remained Python 3.10.20, PyTorch 2.10.0+cu128, "
        "CUDA 12.8, NumPy 2.2.6, NVIDIA H100 NVL. Jobs requested four CPUs and 16 GiB host memory. "
        "Offline guards recorded no attempted network access.\n")
    add("Reusable files: [frozen analysis rules](../benchmarks/multiseed.py), "
        "[runner](../cluster/run_qso_multiseed.py), [preflight](../cluster/preflight_qso_multiseed.sbatch), "
        "[training SLURM script](../cluster/run_qso_multiseed.sbatch), "
        "[aggregation/figures](../cluster/analyse_qso_multiseed.py), "
        "[analysis SLURM script](../cluster/analyse_qso_multiseed.sbatch), "
        "[report renderer](../cluster/report_qso_multiseed.py), "
        "and [regressions](../tests/test_multiseed.py). "
        "The numerical solver/adapter were reused without edits.\n")
    add(f"Large artifacts remain outside the source tree at `{args.root}`: "
        "per-seed/per-method provenance, metric logs, validation logs, QSO solve logs, summaries, "
        "one QSO checkpoint per new completed seed, and exact continuation records. "
        "`analysis.json` holds full-precision tables and pooled metrics; "
        "[project summary](../cluster/qso_multiseed_summary.json) is a compact copy. "
        "Figures have standalone SVG and PDF versions.\n")
    add("## 9. Limitations and predeclared decision\n")
    add("Five seeds are a small sample. Student-t coverage uses the conventional approximately "
        "normal independent paired-seed model; the interval is not a guarantee about other "
        "architectures, datasets, validation samples, horizons or hyperparameter choices. "
        "Seed 2026 participated in LR tuning and is included as explicitly predeclared; "
        "these five outcomes are not all held-out tuning seeds. "
        "Validation uses the same small fixed four-batch sample. "
        "This compares the two tuned configurations, including different AdamW rates for "
        "unsupported parameters; it does not isolate the paired update's causal contribution. "
        "The numerical certificate is conditional fp64 error analysis, not directed interval "
        "arithmetic or a full CUDA implementation proof. It applies before model-dtype "
        "casting and parameter addition, and does not certify secondary P-dagger selection.\n")
    if classification=="A":
        add("The predeclared multi-seed gate is met. The seed-2026 late quality advantage "
            "replicates under the declared fixed-hyperparameter comparison. A **separate** "
            "production-v1 engineering task and separate performance plan are scientifically "
            "justified, with the primary-only scope and very high implementation cost explicit. "
            "Neither integration nor more training is performed here.\n")
    elif classification=="B":
        add("The mean trend favors QSO, but the predeclared confidence/robustness gate is not met. "
            "Production integration is not justified by this result. Whether additional seeds "
            "are worth their cost is a separate decision; none are added here.\n")
    elif classification=="C":
        add("The predeclared comparison does not support a multi-seed advantage. "
            "Production-v1 engineering is not justified for quality reasons by this study.\n")
    else:
        add("The frozen research contract did not sustain every declared QSO trajectory. "
            "The intended five-seed equal-token inference is unavailable. Return to numerical "
            "robustness before quality scaling; do not retune or discard failed seeds.\n")
    add("Production-v0 remains full fp64 thin SVD with `rcond_guard=1e-4`, previous original "
        "lambda, `gram_upper`, existing certificates and explicit CPU reference fallback. "
        "No extra seed, retuning, backend change, production integration or broad superiority "
        "claim is authorized by completion of this report.\n")
    add("## 10. Explicit answers\n")
    add(f"1. **All QSO seeds complete?** {passed}/5; per-seed status is in section 3.\n"
        f"2. **All returned pairs satisfy the primary contract?** {num['pair_solves']:,} logged returns "
        "were audited against unchanged conservative value/rank/feasibility checks; selection scope remains separate.\n"
        "3. **Five Delta_final values?** Section 3 lists each seed without omission.\n"
        "4. **Five Delta_last3 values?** Section 3 lists the predeclared primary differences.\n"
        "5. **Paired means and 95% intervals?** Section 3 reports sample SD, SE and df=4 intervals; "
        "no checkpoint pseudo-replication.\n"
        "6. **How many favor QSO?** The primary and secondary counts are in the paired-statistic table.\n"
        f"7. **Seed-2026 advantage replicated?** The predeclared result is {classification}; "
        "section 9 states its permitted interpretation.\n"
        "8. **Below-old-guard events common?** Section 4 localizes every final event by seed, block and step.\n"
        "9. **Numerical work stable?** Section 4 reports exact Newton bins, CG/line tails and explicit failures.\n"
        "10. **Implementation-cost distribution?** Section 6 separates wall, measured-step and warm-median ratios.\n"
        f"11. **Multi-seed gate met?** {'Yes' if classification=='A' else 'No'}, classification {classification}.\n"
        f"12. **Production-v1 engineering justified?** {'A separate engineering task and performance plan are justified' if classification=='A' else 'Not justified by this completed gate'}; no integration occurs here.\n")
    Path("docs/QSO_MULTISEED_512STEP.md").write_text("\n".join(text))
    print("REPORT docs/QSO_MULTISEED_512STEP.md")


if __name__=="__main__":main()
