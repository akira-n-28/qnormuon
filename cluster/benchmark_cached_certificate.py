"""Compare cached and independently recomputed dual certificates on H100."""
import os
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM allocation required")

import hashlib
import json
from pathlib import Path
import statistics
from unittest.mock import patch

import sitecustomize
import torch
import qnormuon.optimizer as optimizer_module
from benchmarks.tiny_transformer import (
    ModelConfig, Optimizers, TinyTransformer, datasets, load_checkpoint,
    seed_everything, train_step,
)


def digest_tensors(tensors):
    h = hashlib.sha256()
    for value in tensors:
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def compare(first, second):
    """Return maximum discrepancies for identical stored pair inputs."""
    assert len(first) == len(second) == 6
    errors = {}
    for left, right in zip(first, second):
        assert left["name"] == right["name"]
        assert left["input_sha256"] == right["input_sha256"]
        a, b = left["result"], right["result"]
        assert a.converged and b.converged and not a.fallback and not b.fallback
        assert a.iterations == b.iterations and a.counts.hvp == b.counts.hvp
        for field, x, y in (("lambda", a.lam, b.lam), ("primal_direction", a.pair, b.pair)):
            errors[field] = max(errors.get(field, 0.), float((x-y).abs().max()))
        for field in ("primal_objective", "dual_objective", "normalized_gap", "residual_rcond",
                      "horizontal_residual", "normalized_horizontal_residual", "spectral_excess"):
            errors[field] = max(errors.get(field, 0.), abs(a.metrics[field]-b.metrics[field]))
        errors["spectral_norms"] = max(errors.get("spectral_norms", 0.),
                                        max(abs(x-y) for x, y in zip(a.metrics["spectral_norms"],
                                                                      b.metrics["spectral_norms"])))
        assert a.metrics["dual_spectrum_source"] == "independent_original_residual"
        assert b.metrics["dual_spectrum_source"] == "cached_smooth_residual"
        assert a.counts.svd_matrices-b.counts.svd_matrices == 2*(a.iterations+1)
    assert errors["lambda"] < 1e-10
    assert errors["primal_direction"] < 1e-10
    assert errors["dual_objective"] < 1e-10
    assert errors["normalized_gap"] < 1e-10
    assert errors["residual_rcond"] < 1e-10
    assert errors["horizontal_residual"] == 0.
    assert errors["normalized_horizontal_residual"] == 0.
    assert errors["spectral_excess"] == 0.
    assert errors["spectral_norms"] == 0.
    return errors


def main():
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert "H100" in torch.cuda.get_device_name(0)
    config = json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
    source = Path(config["output_root"]) / f"smoke-28914-qso-seed{config['seed']}"
    checkpoint = source / "checkpoint.pt"
    baseline = [json.loads(line) for line in (source / "metrics.jsonl").read_text().splitlines()][25]
    assert checkpoint.is_file()
    seed_everything(config["seed"], config["tf32"])
    train, val = datasets(config)
    metadata = dict(train=train.metadata, validation=val.metadata)
    rows = []
    for independent in (True, False, False, True, False, True, True, False):
        model = TinyTransformer(ModelConfig(**config["model"])).float().cuda()
        opts = Optimizers(model, "qso", config)
        assert load_checkpoint(checkpoint, model, opts, config, metadata) == 25
        for group in opts.paired.param_groups:
            group["solver"]["independent_certificate"] = independent
        names = [p.name for p in opts.paired.pairs]
        saved = []
        original = optimizer_module.solve_coupled

        def observed(u, d, a, **kwargs):
            result = original(u, d, a, **kwargs)
            saved.append(dict(name=names[len(saved)], inputs=(u, d, a, kwargs.get("initial_lambda")),
                              result=result))
            return result

        with patch.object(optimizer_module, "solve_coupled", observed):
            record = train_step(model, opts, train, config, 25)
        assert record["batch_sha256"] == baseline["batch_sha256"]
        assert record["loss"] == baseline["loss"]
        assert len(saved) == 6
        pair_rows = []
        for item in saved:
            input_values = [x for x in item["inputs"] if x is not None]
            pair_rows.append(dict(name=item["name"], input_sha256=digest_tensors(input_values),
                                  result=item["result"]))
        model_sha = digest_tensors(model.state_dict().values())
        rows.append(dict(independent=independent, optimizer_seconds=record["optimizer_seconds"],
                         step_seconds=record["step_seconds"], loss=record["loss"],
                         model_sha256=model_sha, pair_rows=pair_rows,
                         svd_evaluations=sum(item["result"].counts.svd_matrices for item in pair_rows)))
        print("CERT_RUN", len(rows), independent, record["optimizer_seconds"],
              rows[-1]["svd_evaluations"], flush=True)
        del model, opts
    reference = next(row for row in rows if row["independent"])
    cached = next(row for row in rows if not row["independent"])
    errors = compare(reference["pair_rows"], cached["pair_rows"])
    assert reference["model_sha256"] == cached["model_sha256"]
    for row in rows:
        assert row["model_sha256"] == reference["model_sha256"]
    summary = {str(flag): dict(optimizer_median=statistics.median(
        row["optimizer_seconds"] for row in rows[2:] if row["independent"] == flag),
        optimizer_times=[row["optimizer_seconds"] for row in rows[2:] if row["independent"] == flag],
        svd_evaluations=sorted({row["svd_evaluations"] for row in rows if row["independent"] == flag}))
        for flag in (False, True)}
    output = dict(job_id=os.environ["SLURM_JOB_ID"], checkpoint=str(checkpoint), step=25,
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  errors=errors, summary=summary, model_sha256=reference["model_sha256"],
                  rows=[{key: value for key, value in row.items() if key != "pair_rows"}
                        for row in rows],
                  pair_metrics=[dict(name=item["name"], independent={key: value for key, value in item["result"].metrics.items()
                                                                    if key in ("primal_objective", "dual_objective", "normalized_gap", "residual_rcond")},
                                     cached={key: value for key, value in match["result"].metrics.items()
                                             if key in ("primal_objective", "dual_objective", "normalized_gap", "residual_rcond")})
                                for item, match in zip(reference["pair_rows"], cached["pair_rows"])])
    path = Path("cluster") / f"cached_certificate_benchmark-{os.environ['SLURM_JOB_ID']}.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("CERT_SUMMARY", json.dumps(dict(errors=errors, summary=summary)), flush=True)


if __name__ == "__main__":
    main()
