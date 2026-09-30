"""Compare two compact 50-step QSO smoke logs without loading checkpoints."""
import argparse
import json
from pathlib import Path


def load(path):
    root = Path(path)
    summary = json.loads((root/"summary.json").read_text())
    records = [json.loads(line) for line in (root/"metrics.jsonl").read_text().splitlines()]
    solves = [json.loads(line) for line in (root/"solves.jsonl").read_text().splitlines()]
    return summary, records, solves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("optimized")
    args = parser.parse_args()
    old, a, old_solves = load(args.baseline)
    new, b, new_solves = load(args.optimized)
    assert len(a) == len(b) == 50
    assert old["status"] == new["status"] == "passed"
    assert all(x["batch_sha256"] == y["batch_sha256"] for x, y in zip(a, b))
    assert all(x["step"] == y["step"] for x, y in zip(a, b))
    old_end = [x for x in old_solves if x["event"] == "solve_end"]
    new_end = [x for x in new_solves if x["event"] == "solve_end"]
    assert len(old_end) == len(new_end) == 300
    assert [(x["step"], x["pair"]) for x in old_end] == [(x["step"], x["pair"]) for x in new_end]
    loss_delta = [abs(x["loss"]-y["loss"]) for x, y in zip(a, b)]
    val_delta = [abs(x["validation_loss"]-y["validation_loss"])
                 for x, y in zip(a, b) if "validation_loss" in x and "validation_loss" in y]
    def bins(events):
        return {key: sum(e["newton"] == int(key) if key != "3+" else e["newton"] >= 3
                         for e in events)
                for key in ("0", "1", "2", "3+")}
    report = dict(baseline_path=args.baseline, optimized_path=args.optimized,
                  same_batches=True, steps=50,
                  max_train_loss_abs_difference=max(loss_delta),
                  max_validation_loss_abs_difference=max(val_delta),
                  first_train_loss_difference=loss_delta[0],
                  last_train_loss_difference=loss_delta[-1],
                  baseline=dict(solves=len(old_end), certified=sum(e["converged"] for e in old_end),
                                fallback=sum(e["fallback"] for e in old_end), bins=bins(old_end),
                                warm_timing=old["warm_timing"], peak_cuda_bytes=old["peak_cuda_bytes"],
                                final_validation_loss=old["final_validation_loss"],
                                total_wall_seconds=old["total_wall_seconds"]),
                  optimized_metrics=dict(solves=len(new_end), certified=sum(e["converged"] for e in new_end),
                                         fallback=sum(e["fallback"] for e in new_end), bins=bins(new_end),
                                         warm_timing=new["warm_timing"], peak_cuda_bytes=new["peak_cuda_bytes"],
                                         final_validation_loss=new["final_validation_loss"],
                                         total_wall_seconds=new["total_wall_seconds"]),
                  max_pair_gap=max(e["normalized_gap"] for e in new_end),
                  min_pair_rcond=min(e["rcond"] for e in new_end),
                  checkpoint_replay=new["resume"])
    for key in ("step_seconds", "optimizer_seconds", "pair_solver_seconds", "svd_seconds",
                "svdvals_seconds", "certificate_seconds"):
        lhs, rhs = old["warm_timing"][key]["median"], new["warm_timing"][key]["median"]
        report.setdefault("warm_median_speedups", {})[key] = lhs/rhs if rhs else None
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
