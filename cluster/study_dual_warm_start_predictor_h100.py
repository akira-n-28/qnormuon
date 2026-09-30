"""Counterfactual full-SVD predictor replay of the locked QSO training path.

Only the saved baseline warm-start solve controls model and optimizer state.
Candidate solves are read-only and have CPU ADMM fallback disabled.
"""
import json
import math
import os
from pathlib import Path
import statistics
import time
from dataclasses import replace
from unittest.mock import patch

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM allocation required")

import sitecustomize
import torch
import qnormuon.coupled_solver as solver
import qnormuon.optimizer as optimizer_module
from benchmarks.tiny_transformer import ModelConfig, Optimizers, TinyTransformer, datasets, seed_everything, train_step
from experiments.dual_warm_start_predictor import clone_state, predict, problem_scale

POLICIES = ("previous", "center", "raw:0.25", "raw:0.5", "raw:0.75", "raw:1.0",
            "centered:0", "centered:0.5", "centered:1.0", "whitened:0.5")
BASELINE = "baseline"


def sync():
    torch.cuda.synchronize()


def relative_change(now, old):
    return float((now - old).norm() / old.norm().clamp_min(torch.finfo(torch.float64).tiny))


def observed_solve(u, d, a, *, config, initial_lambda, scale):
    first = {}
    original_eval = solver.SmoothDual.evaluate
    original_newton = solver._newton_direction
    def evaluate(problem, z):
        value = original_eval(problem, z)
        if "gradient_norm" not in first:
            first["gradient_norm"] = float(value.gradient.norm())
            first["initial_rcond"] = value.rcond
        return value
    def newton(problem, ev, curvature_scale, max_cg=30):
        direction = original_newton(problem, ev, curvature_scale, max_cg)
        if "first_newton_step_norm" not in first:
            first["first_newton_step_norm"] = float((scale.magnitude * scale.coord * direction).norm())
        return direction
    sync()
    start = time.perf_counter()
    try:
        with patch.object(solver.SmoothDual, "evaluate", evaluate), patch.object(solver, "_newton_direction", newton):
            result = solver.solve_coupled(u, d, a, config=config, initial_lambda=initial_lambda)
        sync()
        elapsed = time.perf_counter() - start
        record = dict(converged=result.converged, reason=result.reason, fallback=result.fallback,
                      iterations=result.iterations, smooth_evaluations=1 + result.counts.line_trials,
                      svd_matrices=result.counts.svd_matrices,
                      line_search_evaluations=result.counts.line_trials,
                      hvps=result.counts.hvp, seconds=elapsed,
                      initial_normalized_gap=result.history[0]["normalized_gap"] if result.history else None,
                      final_normalized_gap=result.metrics["normalized_gap"],
                      final_rcond=result.metrics["residual_rcond"], **first)
        return result, record
    except Exception as error:
        sync()
        return None, dict(converged=False, reason=type(error).__name__ + ": " + str(error),
                          fallback=False, seconds=time.perf_counter() - start, **first)


def add_target_error(record, guess, target, scale):
    delta = guess - target
    record["lambda_error"] = float(delta.norm())
    record["lambda_relative_error"] = float(delta.norm() / target.norm().clamp_min(torch.finfo(torch.float64).tiny))
    if scale.magnitude > 0:
        record["whitened_error"] = float((delta / (scale.magnitude * scale.coord)).norm())


def analyze():
    config = json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
    source = Path(config["output_root"]) / f"smoke-28936-qso-seed{config['seed']}"
    baseline = [json.loads(line) for line in (source / "metrics.jsonl").read_text().splitlines()]
    if len(baseline) != 50:
        raise RuntimeError("expected locked 50-step baseline")
    seed_everything(config["seed"], config["tf32"])
    train, validation = datasets(config)
    model = TinyTransformer(ModelConfig(**config["model"])).float().cuda()
    opts = Optimizers(model, "qso", config)
    names = [p.name for p in model.pairs()]
    previous = {}
    older = {}
    prior_problem = {}
    rows = []
    training = []
    stress_input = {}
    call = [0]
    step_now = [0]
    production = optimizer_module.solve_coupled
    def capture(u, d, a, **kwargs):
        name = names[call[0] % len(names)]
        call[0] += 1
        step = step_now[0]
        scale = problem_scale(u, d, a)
        prev, old = previous.get(name), older.get(name)
        baseline_guess = kwargs["initial_lambda"] if kwargs["initial_lambda"] is not None else scale.beta
        cfg = replace(kwargs["config"], fallback=False)
        # The unmodified production call alone controls the actual optimizer update.
        base, base_record = observed_solve(u, d, a, config=cfg,
                                           initial_lambda=kwargs["initial_lambda"], scale=scale)
        if base is None or not base.converged or base.fallback:
            raise RuntimeError(f"locked baseline failed at {step} {name}: {base_record}")
        add_target_error(base_record, baseline_guess, base.lam, scale)
        transitions = {}
        if name in prior_problem:
            pu, pd, pa, ps = prior_problem[name]
            transitions = dict(weight_u_change=relative_change(u, pu),
                weight_d_change=relative_change(d, pd), objective_change=relative_change(a, pa),
                beta_change=relative_change(scale.beta, ps.beta),
                magnitude_ratio=scale.magnitude / ps.magnitude if ps.magnitude else None,
                whitening_change=relative_change(scale.coord, ps.coord))
        row = dict(step=step, pair=name, objective_scale=scale.magnitude,
                   baseline=base_record, transitions=transitions, candidates={})
        for policy in POLICIES:
            guess = predict(policy, scale, prev, old)
            candidate, record = observed_solve(u, d, a, config=cfg,
                                               initial_lambda=guess, scale=scale)
            add_target_error(record, guess, base.lam, scale)
            if candidate is not None and candidate.converged:
                record["direction_relative_difference"] = float(
                    (candidate.pair - base.pair).norm() / base.pair.norm().clamp_min(1e-300))
                record["lambda_solution_relative_difference"] = float(
                    (candidate.lam - base.lam).norm() / base.lam.norm().clamp_min(1e-300))
            row["candidates"][policy] = record
        rows.append(row)
        older[name] = prev
        previous[name] = clone_state(base.lam, scale)
        prior_problem[name] = (u.detach().clone(), d.detach().clone(), a.detach().clone(), scale)
        if step == 25 and name == names[0]:
            stress_input.update(dict(u=u.detach().clone(), d=d.detach().clone(),
                                     a=a.detach().clone(), previous=prev, older=old))
        return base
    with patch.object(optimizer_module, "solve_coupled", capture):
        for step in range(50):
            step_now[0] = step
            result = train_step(model, opts, train, config, step)
            if result["batch_sha256"] != baseline[step]["batch_sha256"]:
                raise RuntimeError(f"batch mismatch at step {step}")
            # The baseline path should remain deterministic despite counterfactual solves.
            if result["loss"] != baseline[step]["loss"]:
                raise RuntimeError(f"loss mismatch at step {step}: {result['loss']} vs {baseline[step]['loss']}")
            training.append(dict(step=step, loss=result["loss"], batch_sha256=result["batch_sha256"]))
            if step % 5 == 4:
                print(f"replay step {step+1}/50, pair solves {len(rows)}", flush=True)
    if len(rows) != 300:
        raise RuntimeError("expected 300 real pair solves")
    return rows, training, stress_input, config


def stress_cases(sample, config):
    u, d, a = sample["u"], sample["d"], sample["a"]
    prior, older = sample["previous"], sample["older"]
    generator = torch.Generator(device=u.device).manual_seed(221)
    options = dict(ordinary=a, scale_10=10*a, sign_flip=-a,
                   column_rotation=torch.roll(a, shifts=83, dims=-1),
                   large_rotation=a + 5*torch.randn(a.shape, dtype=a.dtype, device=a.device, generator=generator)*a.square().mean().sqrt())
    cfg = solver.SolverConfig(dtype=torch.float64, tolerance=3e-5, rcond_guard=1e-4,
                              fallback=False, max_iterations=30)
    output = {}
    for label, target in options.items():
        scale = problem_scale(u, d, target)
        base, br = observed_solve(u, d, target, config=cfg,
                                  initial_lambda=prior.lam, scale=scale)
        trials = {}
        for policy in ("center", "raw:0.5", "raw:1.0", "centered:0.5", "whitened:0.5"):
            guess = predict(policy, scale, prior, older)
            _, record = observed_solve(u, d, target, config=cfg, initial_lambda=guess, scale=scale)
            if base is not None:
                add_target_error(record, guess, base.lam, scale)
            trials[policy] = record
        output[label] = dict(baseline=br, candidates=trials)
    return output


def summaries(rows):
    output = {}
    for policy in (BASELINE,) + POLICIES:
        records = [r["baseline"] if policy == BASELINE else r["candidates"][policy] for r in rows]
        warm = records[6:]
        values = [r["smooth_evaluations"] for r in warm if r.get("converged")]
        times = [r["seconds"] for r in warm]
        bins = {key: sum(r.get("converged") and (r.get("iterations", 99) == key if key < 3
                  else r.get("iterations", 0) >= 3) for r in warm) for key in (0,1,2,3)}
        output[policy] = dict(solves=len(records), warm_solves=len(warm),
            warm_certified=sum(r.get("converged", False) for r in warm),
            failures=[dict(step=rows[i+6]["step"], pair=rows[i+6]["pair"], reason=r["reason"])
                      for i,r in enumerate(warm) if not r.get("converged")],
            newton_bins=bins, smooth_evaluations=sum(values),
            mean_smooth_evaluations=statistics.mean(values) if values else None,
            line_search_evaluations=sum(r.get("line_search_evaluations", 0) for r in warm),
            seconds=sum(times), median_pair_seconds=statistics.median(times),
            mean_lambda_error=statistics.mean(r["lambda_error"] for r in warm))
    return output


if __name__ == "__main__":
    if not sitecustomize.QSO_NETWORK_GUARD_ACTIVE or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("offline H100 allocation required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.linalg.svd(torch.randn(2,24,8,device="cuda",dtype=torch.float64),full_matrices=False)
    sync()
    start = time.perf_counter()
    rows, training, sample, config = analyze()
    stress = stress_cases(sample, config)
    report = dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(0),
                  torch=torch.__version__, config=config, policies=POLICIES,
                  training=training, summary=summaries(rows), rows=rows,
                  stress=stress, elapsed_seconds=time.perf_counter()-start)
    output = Path("cluster") / f"dual_warm_start_predictor-{os.environ['SLURM_JOB_ID']}.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("RESULT", json.dumps(dict(output=str(output), elapsed_seconds=report["elapsed_seconds"],
                                 summary=report["summary"])), flush=True)
