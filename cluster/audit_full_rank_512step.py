"""Bounded, independent offline audit of at most four accepted saved pairs."""
import os
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM required")
import argparse
import json
from pathlib import Path
import signal
import time
import torch
import qnormuon.coupled_solver as cs
from experiments.near_rank_optimum import independent_lbfgs, kkt


def timeout(*args):
    raise TimeoutError("180-second independent fixed-pair audit budget")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("trajectory");args=parser.parse_args()
    root=Path(args.trajectory)
    assert root.parent==Path("/home/prignano/qnormuon-runs/full-rank-512step")
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    signal.signal(signal.SIGALRM,timeout)
    rows={}
    for path in sorted(root.glob("audit_*.pt")):
        f=torch.load(path,map_location="cuda",weights_only=False)
        u,d,a=(f[k].double() for k in ("u","d","a"))
        torch.cuda.synchronize();started=time.perf_counter()
        try:
            signal.alarm(180)
            # Independent two-loop L-BFGS, started at centering beta. No Newton/HVP.
            lam,history,counts=independent_lbfgs(u,d,a,cs.multipliers(u,d,a))
            check,polar,p=kkt(u,d,a,lam)
            observed=float((a*(p-f["pair"])).sum())
            row=dict(status="audited",source=path.name,training_step=f["metadata"]["step"],
                pair_name=f["metadata"]["pair"],G_upper=f["metadata"]["G_upper"],
                observed_support_regret=observed,
                observed_regret_inside_bound=observed<=f["metadata"]["G_upper"],
                direction_difference=float((p-f["pair"]).norm()),
                relative_direction_difference=float((p-f["pair"]).norm()/p.norm()),
                multiplier_difference=float((lam-f["lam"]).norm()),
                iterations=len(history)-1,internal_gradient_norm=history[-1]["gradient_norm"],
                high_accuracy_target_reached=history[-1]["gradient_norm"]<2e-12,
                independent_method="safeguarded full-SVD L-BFGS, beta start; no Newton/HVP",
                kkt=check,svd_matrices=counts.svd_matrices,line_trials=counts.line_trials)
        except Exception as error:
            row=dict(status="bounded_audit_incomplete",source=path.name,exception=repr(error))
        finally:signal.alarm(0)
        torch.cuda.synchronize();row["seconds"]=time.perf_counter()-started
        rows[path.name]=row
        (root/"offline_audit.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")
        print("OFFLINE_AUDIT",json.dumps(row),flush=True)


if __name__=="__main__":main()
