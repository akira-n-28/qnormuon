"""Exact H100 kernel, certificate-frequency, and batch studies on saved pair data."""
import os
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM allocation required")

from collections import defaultdict
import json
from pathlib import Path
import statistics
from types import SimpleNamespace
import time
from unittest.mock import patch

import sitecustomize
import torch
import qnormuon.coupled_solver as solver
import qnormuon.optimizer as optimizer_module
from benchmarks.tiny_transformer import (
    ModelConfig, Optimizers, TinyTransformer, datasets, load_checkpoint,
    seed_everything, train_step,
)


def timed(fn, repeats=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
    return dict(median_seconds=statistics.median(times), times=times)


def norm_svd(p):
    return torch.linalg.svdvals(p)[:, 0]


def norm_matrix(p):
    return torch.linalg.matrix_norm(p, ord=2)


def norm_gram(p):
    return torch.linalg.eigvalsh(p.transpose(-2, -1) @ p)[:, -1].clamp_min(0).sqrt()


def norm_qr(p):
    _, r = torch.linalg.qr(p, mode="reduced")
    return torch.linalg.svdvals(r)[:, 0]


def decompose_svd(b):
    return torch.linalg.svd(b, full_matrices=False)


def decompose_qr(b):
    q, r = torch.linalg.qr(b, mode="reduced")
    qr, s, vt = torch.linalg.svd(r, full_matrices=False)
    return q @ qr, s, vt


def decompose_gram(b):
    eigenvalues, v = torch.linalg.eigh(b.transpose(-2, -1) @ b)
    s = eigenvalues.flip(-1).clamp_min(0).sqrt()
    v = v.flip(-1)
    q = (b @ v) / s.unsqueeze(-2)
    return q, s, v.transpose(-2, -1)


def max_relative(a, b):
    return float((a-b).abs().max() / a.abs().max().clamp_min(torch.finfo(a.dtype).tiny))


def main():
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert "H100" in torch.cuda.get_device_name(0)
    config = json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
    source = Path(config["output_root"]) / f"smoke-28914-qso-seed{config['seed']}"
    checkpoint = source / "checkpoint.pt"
    assert checkpoint.is_file()
    seed_everything(config["seed"], config["tf32"])
    train, val = datasets(config)
    metadata = dict(train=train.metadata, validation=val.metadata)
    model = TinyTransformer(ModelConfig(**config["model"])).float().cuda()
    opts = Optimizers(model, "qso", config)
    assert load_checkpoint(checkpoint, model, opts, config, metadata) == 25
    names = [p.name for p in opts.paired.pairs]
    active = [None]
    pair_index = [0]
    ordinal = defaultdict(int)
    certs, residuals = [], []
    original_solve = optimizer_module.solve_coupled
    original_cert = solver.certificate
    original_svd = torch.linalg.svd

    def observed_solve(*args, **kwargs):
        active[0] = names[pair_index[0]]
        pair_index[0] += 1
        return original_solve(*args, **kwargs)

    def observed_certificate(u, d, a, lam, candidate, counts, **kwargs):
        index = ordinal[active[0]]
        ordinal[active[0]] += 1
        torch.cuda.synchronize()
        started = time.perf_counter()
        p, metrics = original_cert(u, d, a, lam, candidate, counts, **kwargs)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        # Keep exact projected candidates for later isolated kernel comparisons.
        projected = solver.horizontal_project(u.double(), d.double(), candidate.double())
        certs.append(dict(pair=active[0], ordinal=index, u=u, d=d, a=a,
                          projected=projected, dual=metrics["dual_objective"],
                          seconds=seconds, accepted=solver.accepted(metrics, 3e-5),
                          metrics=metrics,
                          polar_horizontal_max=float(solver.horizontal_residual(u, d, candidate).abs().max())))
        return p, metrics

    def observed_svd(b, *args, **kwargs):
        if b.ndim == 3 and b.shape[0] == 2 and b.shape[-2:] == (1024, 384):
            residuals.append(dict(pair=active[0], b=b))
        return original_svd(b, *args, **kwargs)

    with patch.object(optimizer_module, "solve_coupled", observed_solve), \
         patch.object(solver, "certificate", observed_certificate), \
         patch.object(torch.linalg, "svd", observed_svd):
        record = train_step(model, opts, train, config, 25)
    baseline = [json.loads(line) for line in (source / "metrics.jsonl").read_text().splitlines()][25]
    assert record["batch_sha256"] == baseline["batch_sha256"] and record["loss"] == baseline["loss"]
    assert pair_index[0] == 6 and len(certs) == 18 and len(residuals) == 18
    assert all(c["projected"].dtype == torch.float64 for c in certs)
    assert all(item["b"].device.type == "cuda" for item in residuals)
    selected_p = [next(c for c in reversed(certs) if c["pair"] == name) for name in names]
    selected_b = [next(c["b"] for c in reversed(residuals) if c["pair"] == name) for name in names]

    # Part 4: exact primal spectral norms, certification after radial scaling.
    primal = []
    methods = dict(svdvals=norm_svd, matrix_norm=norm_matrix, gram_eigh=norm_gram,
                   qr_svdvals=norm_qr)
    for case in selected_p:
        p, a, dual = case["projected"], case["a"], case["dual"]
        oracle = norm_svd(p)
        for method, fn in methods.items():
            value = fn(p)
            scale = max(1., float(value.max()))
            feasible = p/scale
            exact_norms = norm_svd(feasible)
            objective = float((a*feasible).sum())
            gap = (dual-objective)/max(abs(dual), abs(objective))
            replicated = fn(p)
            primal.append(dict(pair=case["pair"], method=method,
                               seconds=timed(lambda: fn(p))["median_seconds"],
                               max_abs_error=float((oracle-value).abs().max()),
                               deterministic=torch.equal(value, replicated),
                               max_exact_norm_after_scale=float(exact_norms.max()),
                               objective=objective, normalized_gap=gap,
                               rcond=case["metrics"]["residual_rcond"]))
    primal_summary = {}
    for method in methods:
        items = [row for row in primal if row["method"] == method]
        primal_summary[method] = dict(total_median_kernel_seconds=sum(r["seconds"] for r in items),
                                      max_abs_error=max(r["max_abs_error"] for r in items),
                                      max_exact_norm_after_scale=max(r["max_exact_norm_after_scale"] for r in items),
                                      deterministic=all(r["deterministic"] for r in items))

    # Part 5: full certificate at each Newton iterate, including early rejects.
    frequency = [dict(pair=c["pair"], iteration=c["ordinal"], seconds=c["seconds"],
                      normalized_gap=c["metrics"]["normalized_gap"],
                      polar_horizontal_max=c["polar_horizontal_max"], accepted=c["accepted"])
                 for c in certs]
    frequency_summary = {str(i): dict(calls=sum(c["ordinal"] == i for c in certs),
                                     total_seconds=sum(c["seconds"] for c in certs if c["ordinal"] == i),
                                     accepted=sum(c["accepted"] for c in certs if c["ordinal"] == i))
                         for i in sorted({c["ordinal"] for c in certs})}

    # Part 6: full polar/SVD data, including HVP derivative, against full SVD.
    generator = torch.Generator(device="cuda").manual_seed(4021)
    tall_cases = [(name, b) for name, b in zip(names, selected_b)]
    for rcond in (1e-4, 1.2e-4, 1e-3, 1e-2):
        raw = torch.randn(1024, 384, dtype=torch.float64, device="cuda", generator=generator)
        q, _ = torch.linalg.qr(raw, mode="reduced")
        right, _ = torch.linalg.qr(torch.randn(384, 384, dtype=torch.float64, device="cuda", generator=generator))
        s = torch.linspace(1., .1, 384, dtype=torch.float64, device="cuda")
        s[-1] = rcond
        b = (q * s) @ right.T
        tall_cases.append((f"synthetic_rcond_{rcond:g}", torch.stack((b, b))))
    backends = dict(full_svd=decompose_svd, qr_svd=decompose_qr, gram_eigh=decompose_gram)
    derivative = solver.SmoothDual(None, None, None)
    backend_rows = []
    for name, b in tall_cases:
        ref_q, ref_s, ref_vt = decompose_svd(b)
        ref_polar = ref_q @ ref_vt
        e = torch.randn(b.shape, dtype=b.dtype, device=b.device, generator=generator)
        e = e/e.norm()
        ref_hvp = derivative.polar_derivative(SimpleNamespace(left=ref_q, singular=ref_s, right=ref_vt), e)
        for backend, fn in backends.items():
            q, s, vt = fn(b)
            polar = q @ vt
            hvp = derivative.polar_derivative(SimpleNamespace(left=q, singular=s, right=vt), e)
            backend_rows.append(dict(case=name, backend=backend,
                                     seconds=timed(lambda: fn(b), repeats=3)["median_seconds"],
                                     rcond=float(ref_s[:, -1].div(ref_s[:, 0]).min()),
                                     singular_relative_error=max_relative(ref_s, s),
                                     smallest_singular_relative_error=float(((ref_s[:, -1]-s[:, -1]).abs()/ref_s[:, -1]).max()),
                                     polar_relative_error=float((polar-ref_polar).norm()/ref_polar.norm()),
                                     hvp_relative_error=float((hvp-ref_hvp).norm()/ref_hvp.norm()),
                                     left_orthogonality_max=float((q.transpose(-2,-1)@q-torch.eye(384, device=b.device, dtype=b.dtype)).abs().max())))

    # Near the production rcond boundary, check projected primal feasibility
    # independently rather than inferring safety from the real warm cases.
    near_guard_primal = []
    guard_u, guard_d = selected_p[0]["u"], selected_p[0]["d"]
    for name, b in tall_cases:
        if not name.startswith("synthetic_"):
            continue
        q, _, vt = decompose_svd(b)
        projected = solver.horizontal_project(guard_u, guard_d, q @ vt)
        b_spectrum = torch.linalg.svdvals(b)
        dual = float(b_spectrum.sum())
        residual_rcond = float(b_spectrum[:, -1].div(b_spectrum[:, 0]).min())
        oracle = norm_svd(projected)
        for method, fn in methods.items():
            value = fn(projected)
            scale = max(1., float(value.max()))
            feasible = projected/scale
            exact_norms = norm_svd(feasible)
            objective = float((b*feasible).sum())
            near_guard_primal.append(dict(case=name, method=method,
                                          rcond=residual_rcond,
                                          max_abs_error=float((oracle-value).abs().max()),
                                          max_exact_norm_after_scale=float(exact_norms.max()),
                                          objective=objective,
                                          normalized_gap=(dual-objective)/max(abs(dual),abs(objective)),
                                          seconds=timed(lambda: fn(projected), repeats=3)["median_seconds"]))

    # Part 7: compare six existing 2-matrix calls with one 12-matrix call.
    batch_b = torch.cat(selected_b)
    batch_p = torch.cat([c["projected"] for c in selected_p])
    batch_gram = batch_b.transpose(-2, -1) @ batch_b
    batching = {}
    for name, single, batched in (
        ("svd", lambda: [torch.linalg.svd(b, full_matrices=False) for b in selected_b],
         lambda: torch.linalg.svd(batch_b, full_matrices=False)),
        ("svdvals", lambda: [torch.linalg.svdvals(c["projected"]) for c in selected_p],
         lambda: torch.linalg.svdvals(batch_p)),
        ("eigvalsh", lambda: [torch.linalg.eigvalsh(g) for g in batch_gram.split(2)],
         lambda: torch.linalg.eigvalsh(batch_gram)),
    ):
        sequential = timed(single, repeats=3)
        grouped = timed(batched, repeats=3)
        batching[name] = dict(sequential_median_seconds=sequential["median_seconds"],
                              batch_median_seconds=grouped["median_seconds"],
                              speedup=sequential["median_seconds"]/grouped["median_seconds"])

    report = dict(job_id=os.environ["SLURM_JOB_ID"], checkpoint=str(checkpoint), step=25,
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  solver_dtype="torch.float64", tolerance=3e-5, rcond_guard=1e-4,
                  primal=primal, primal_summary=primal_summary,
                  certificate_frequency=frequency, frequency_summary=frequency_summary,
                  backend_rows=backend_rows, near_guard_primal=near_guard_primal,
                  batching=batching)
    output = Path("cluster")/f"solver_kernel_study-{os.environ['SLURM_JOB_ID']}.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("PRIMAL_SUMMARY", json.dumps(primal_summary), flush=True)
    print("CERT_FREQUENCY", json.dumps(frequency_summary), flush=True)
    print("BACKEND_ROWS", json.dumps(backend_rows), flush=True)
    print("NEAR_GUARD_PRIMAL", json.dumps(near_guard_primal), flush=True)
    print("BATCHING", json.dumps(batching), flush=True)


if __name__ == "__main__":
    main()
