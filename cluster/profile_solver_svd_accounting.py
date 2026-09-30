"""Count every decomposition in one saved warm QSO training step on H100."""
import os
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM allocation required")

from collections import defaultdict
from contextlib import ExitStack
import json
import linecache
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import sitecustomize
import torch
import qnormuon.optimizer as optimizer_module
from benchmarks.tiny_transformer import (
    ModelConfig, Optimizers, TinyTransformer, datasets, load_checkpoint,
    seed_everything, train_step,
)


def classify():
    caller = sys._getframe(2)
    name = caller.f_code.co_name
    if name == "evaluate" and caller.f_globals.get("__name__") == "qnormuon.coupled_solver":
        parent = caller.f_back
        line = linecache.getline(parent.f_code.co_filename, parent.f_lineno).strip()
        if parent.f_code.co_name == "_solve" and "trial = problem.evaluate" in line:
            return "line_search_trial"
        if parent.f_code.co_name == "_solve" and "ev = problem.evaluate" in line:
            return "newton_residual_evaluation"
    if name == "certificate" and caller.f_globals.get("__name__") == "qnormuon.coupled_solver":
        line = linecache.getline(caller.f_code.co_filename, caller.f_lineno).strip()
        if "residual_spectra =" in line:
            return "residual_dual_certificate"
        if "spectra =" in line:
            return "primal_feasible_recovery"
    if name == "step" and caller.f_globals.get("__name__") == "qnormuon.optimizer":
        return "cast_diagnostics"
    if any(frame.f_code.co_name == "_reference" for frame in (caller, caller.f_back, caller.f_back.f_back)):
        return "fallback_reference"
    return "other"


def summarize(calls):
    grouped = defaultdict(list)
    for call in calls:
        grouped[(call["operator"], call["category"])].append(call)
    return [dict(operator=op, category=category, calls=len(rows),
                 matrices=sum(row["matrices"] for row in rows),
                 total_seconds=sum(row["seconds"] for row in rows),
                 median_seconds=statistics.median(row["seconds"] for row in rows))
            for (op, category), rows in sorted(grouped.items())]


def main():
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0)
    config = json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
    source = Path(config["output_root"]) / f"smoke-28914-qso-seed{config['seed']}"
    checkpoint = source / "checkpoint.pt"
    baseline = [json.loads(line) for line in (source / "metrics.jsonl").read_text().splitlines()]
    if not checkpoint.is_file() or len(baseline) != 50:
        raise FileNotFoundError("validated 50-step baseline checkpoint/metrics are required")
    seed_everything(config["seed"], config["tf32"])
    model = TinyTransformer(ModelConfig(**config["model"])).float().cuda()
    opts = Optimizers(model, "qso", config)
    train, validation = datasets(config)
    metadata = dict(train=train.metadata, validation=validation.metadata)
    step = load_checkpoint(checkpoint, model, opts, config, metadata)
    assert step == 25
    # Warm the same CUDA operations before measuring. This replica is discarded.
    warm_model = TinyTransformer(ModelConfig(**config["model"])).float().cuda()
    warm_opts = Optimizers(warm_model, "qso", config)
    assert load_checkpoint(checkpoint, warm_model, warm_opts, config, metadata) == step
    train_step(warm_model, warm_opts, train, config, step)
    del warm_model, warm_opts
    torch.cuda.synchronize()

    calls = []
    active_pair = [None]
    pair_index = [0]
    names = [pair.name for pair in opts.paired.pairs]
    original_solve = optimizer_module.solve_coupled

    def observed_solve(*args, **kwargs):
        active_pair[0] = names[pair_index[0]]
        pair_index[0] += 1
        return original_solve(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(optimizer_module, "solve_coupled", observed_solve))
        for operator in ("svd", "svdvals"):
            original = getattr(torch.linalg, operator)

            def timed(*args, _operator=operator, _original=original, **kwargs):
                category = classify()
                tensor = args[0]
                matrices = tensor.shape[0] if tensor.ndim == 3 else 1
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = _original(*args, **kwargs)
                torch.cuda.synchronize()
                calls.append(dict(operator=_operator, category=category, pair=active_pair[0],
                                  shape=list(tensor.shape), matrices=matrices,
                                  seconds=time.perf_counter() - started,
                                  device=str(tensor.device), dtype=str(tensor.dtype)))
                return result

            stack.enter_context(patch.object(torch.linalg, operator, timed))
        record = train_step(model, opts, train, config, step)

    expected = baseline[step]
    assert record["batch_sha256"] == expected["batch_sha256"]
    assert record["loss"] == expected["loss"]
    assert pair_index[0] == 6
    assert all(call["category"] != "other" for call in calls), calls
    assert all(call["device"].startswith("cuda") for call in calls)
    rows = summarize(calls)
    by_pair = {name: summarize([call for call in calls if call["pair"] == name]) for name in names}
    for name in names:
        solver_matrices = sum(call["matrices"] for call in calls
                              if call["pair"] == name and call["category"] != "cast_diagnostics")
        assert solver_matrices == record["qso"][name]["svd_evaluations"]
    report = dict(job_id=os.environ["SLURM_JOB_ID"], step=step, checkpoint=str(checkpoint),
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda_build=torch.version.cuda,
                  loss=record["loss"], optimizer_seconds=record["optimizer_seconds"],
                  step_seconds=record["step_seconds"], baseline_loss=expected["loss"],
                  rows=rows, per_pair=by_pair, calls=calls,
                  solver_metrics=record["qso"],
                  total_svd_seconds=sum(call["seconds"] for call in calls if call["operator"] == "svd"),
                  total_svdvals_seconds=sum(call["seconds"] for call in calls if call["operator"] == "svdvals"))
    output = Path("cluster") / f"solver_svd_accounting-{os.environ['SLURM_JOB_ID']}.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("ACCOUNTING", json.dumps(dict(job_id=report["job_id"], step=step, rows=rows,
                                        total_svd_seconds=report["total_svd_seconds"],
                                        total_svdvals_seconds=report["total_svdvals_seconds"])), flush=True)
    print("OUTPUT", output, flush=True)


if __name__ == "__main__":
    main()
