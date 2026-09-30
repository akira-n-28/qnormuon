"""Small H100 follow-up: Newton gap history and O(m) predictor cost."""
import io
import json
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM allocation required")

import sitecustomize
import torch
import qnormuon.optimizer as optimizer_module
from benchmarks.tiny_transformer import (ModelConfig, Optimizers, TinyTransformer,
    datasets, load_checkpoint, seed_everything, train_step)
from experiments.dual_warm_start_predictor import (clone_state, pack_state,
    predict, problem_scale, unpack_state)

if not sitecustomize.QSO_NETWORK_GUARD_ACTIVE or "H100" not in torch.cuda.get_device_name(0):
    raise RuntimeError("offline H100 allocation required")

config = json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
source = Path(config["output_root"]) / f"smoke-28936-qso-seed{config['seed']}"
baseline = [json.loads(line) for line in (source / "metrics.jsonl").read_text().splitlines()]
seed_everything(config["seed"], config["tf32"])
train, val = datasets(config)
metadata = dict(train=train.metadata, validation=val.metadata)
model = TinyTransformer(ModelConfig(**config["model"])).float().cuda()
opts = Optimizers(model, "qso", config)
assert load_checkpoint(source / "checkpoint.pt", model, opts, config, metadata) == 25
original = optimizer_module.solve_coupled
rows = []
sample = {}
names = [p.name for p in model.pairs()]
def capture(u, d, a, **kwargs):
    result = original(u, d, a, **kwargs)
    rows.append(dict(name=names[len(rows)], history=[dict(iteration=h["iteration"],
        gap=h["normalized_gap"], rcond=h["rcond"]) for h in result.history],
        evaluations=1 + result.counts.line_trials, certified=result.converged))
    if len(rows) == 1:
        sample.update(scale=problem_scale(u, d, a), previous=clone_state(kwargs["initial_lambda"],
            problem_scale(u, d, a)))
    return result
with patch.object(optimizer_module, "solve_coupled", capture):
    step = train_step(model, opts, train, config, 25)
assert step["batch_sha256"] == baseline[25]["batch_sha256"]
assert step["loss"] == baseline[25]["loss"]
assert len(rows) == 6 and all(row["certified"] for row in rows)

current = sample["scale"]
previous = sample["previous"]
older = clone_state(previous.lam * .99, previous.scale)
policies = ("previous", "center", "raw:0.5", "raw:1.0", "centered:0",
            "centered:0.5", "centered:1.0", "whitened:0.5")
timing = {}
for policy in policies:
    for _ in range(20):
        predict(policy, current, previous, older)
    torch.cuda.synchronize()
    values = []
    for _ in range(200):
        torch.cuda.synchronize()
        start = time.perf_counter()
        predict(policy, current, previous, older)
        torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    timing[policy] = dict(median_seconds=statistics.median(values),
                          p95_seconds=sorted(values)[189])

buffer = io.BytesIO()
torch.save(dict(previous=pack_state(previous), older=pack_state(older)), buffer)
buffer.seek(0)
stored = torch.load(buffer, weights_only=True)
checkpoint_equal = {policy: torch.equal(predict(policy, current, previous, older),
    predict(policy, current, unpack_state(stored["previous"]),
            unpack_state(stored["older"]))) for policy in policies}
report = dict(job=os.environ["SLURM_JOB_ID"], step=25, batch_sha256=step["batch_sha256"],
              loss=step["loss"], rows=rows, timing=timing,
              checkpoint_predictor_state_exact=checkpoint_equal,
              packed_state_bytes=sum(t.numel()*t.element_size() for state in stored.values()
                  for t in state.values() if isinstance(t, torch.Tensor)))
path = Path("cluster") / f"dual_warm_start_followup-{os.environ['SLURM_JOB_ID']}.json"
path.write_text(json.dumps(report, indent=2) + "\n")
print("RESULT", json.dumps(dict(path=str(path), rows=rows, timing=timing,
    checkpoint_predictor_state_exact=checkpoint_equal,
    packed_state_bytes=report["packed_state_bytes"])), flush=True)
