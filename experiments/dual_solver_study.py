"""Reproducible CPU study; writes raw, per-solve measurements as JSON.

Run python3 -m experiments.dual_solver_study --output experiments/dual_solver_results.json
Use --large for additional 768x256 and 1536x512 timing cases. These are small
SwiGLU block dimensions, not a claim of LLM-scale production performance.
"""
import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path

import torch

from experiments.dual_solver import comparison, reference, solve_dual
from experiments.horizontal_spectral import random_example, fractional_example, adjoint


def balance(u, d):
    ru, rd = u.norm(dim=1), d.norm(dim=1)
    c = (rd / ru).sqrt()
    return c[:, None] * u, d / c[:, None]


def synthetic_sequence(m=96, n=32, steps=9, drift=.001, seed=2001):
    rng = torch.Generator().manual_seed(seed)
    u, d, a = random_example(seed, m=m, n=n)
    base = a.clone()
    frames = [(u.clone(), d.clone(), a.clone())]
    for _ in range(steps - 1):
        u = u + drift * torch.randn(m, n, generator=rng, dtype=torch.float64) / n ** .5
        d = d + drift * torch.randn(m, n, generator=rng, dtype=torch.float64) / n ** .5
        u, d = balance(u, d)
        base = base + drift * torch.randn(2, m, n, generator=rng, dtype=torch.float64)
        a = .95 * a + .05 * base
        frames.append((u.clone(), d.clone(), a.clone()))
    return frames


def toy_swiglu_sequence(steps=9, m=48, n=16, seed=2011):
    """Actual fixed-batch SwiGLU gradients, canonical EMA, reference factor steps."""
    rng = torch.Generator().manual_seed(seed)
    u, d, _ = random_example(seed, m=m, n=n)
    x = torch.randn(64, n, generator=rng, dtype=torch.float64)
    gate_w = torch.randn(m, n, generator=rng, dtype=torch.float64) / n ** .5
    target = torch.randn(64, n, generator=rng, dtype=torch.float64)
    gate = torch.nn.functional.silu(x @ gate_w.T)
    momentum, frames = None, []
    for _ in range(steps):
        u, d = u.detach().requires_grad_(), d.detach().requires_grad_()
        output = (gate * (x @ u.T)) @ d
        loss = (output - target).square().mean() / 2
        gu, gd = torch.autograd.grad(loss, (u, d))
        gradient = torch.stack((gu, gd))
        momentum = gradient if momentum is None else .95 * momentum + .05 * gradient
        u, d, momentum = u.detach(), d.detach(), momentum.detach()
        frames.append((u.clone(), d.clone(), momentum.clone()))
        direction = reference(u, d, momentum)
        u, d = balance(u - .001 * direction.pair[0], d - .001 * direction.pair[1])
    return frames


def record(r, ref, u, d, a, **labels):
    return {**labels, "shape": list(u.shape), "dtype": str(a.dtype),
            "method": r.method, "iterations": r.iterations, "converged": r.converged,
            "fallback": r.fallback, "reason": r.reason, "seconds": r.seconds,
            "counts": asdict(r.counts), "metrics": r.metrics,
            "min_accepted_rcond": r.min_accepted_rcond,
            "secondary_selection": r.secondary_selection,
            **comparison(u, d, a, r.pair, ref.pair)}


def run_sequence(frames, name, dtype, records):
    frames = [tuple(v.to(dtype) for v in f) for f in frames]
    refs = [reference(*f) for f in frames]
    for t, (f, ref) in enumerate(zip(frames, refs)):
        records.append(record(ref, ref, *f, case=name, mode="reference", step=t))
    tol = 1e-6 if dtype == torch.float64 else 3e-5
    for method in ("gd", "pgd", "lbfgs", "newton"):
        previous = refs[0].lam
        for t, f in enumerate(frames[1:], 1):
            cold = solve_dual(*f, method=method, tolerance=tol)
            warm = solve_dual(*f, method=method, initial_lambda=previous, tolerance=tol)
            previous = warm.lam
            for mode, result in (("cold", cold), ("warm", warm)):
                records.append(record(result, refs[t], *f, case=name, mode=mode, step=t))
        for budget in (1, 2, 4, 8):
            previous = refs[0].lam
            for t, f in enumerate(frames[1:], 1):
                r = solve_dual(*f, method=method, initial_lambda=previous, tolerance=tol,
                               max_iterations=budget, fallback=False)
                previous = r.lam  # closed-loop approximate warm state, not oracle resets
                records.append(record(r, refs[t], *f, case=name, mode="budget", step=t, budget=budget))
    print(name, str(dtype), "completed", flush=True)


def adversaries(records):
    u, d, a = random_example(2020, m=9, n=3)
    cases = []
    uf, df, af, _ = fractional_example()
    cases.append(("fractional", uf, df, af, None))
    uz = dz = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], dtype=torch.float64)
    az = torch.zeros(2, 3, 2, dtype=torch.float64)
    az[:, 0, 0], az[:, 1, 1] = 1., 1e-12
    cases.append(("near_rank", uz, dz, az, torch.zeros(3, dtype=torch.float64)))
    cases.append(("zero", u, d, torch.zeros_like(a), None))
    lam = torch.linspace(-1, 1, 9, dtype=torch.float64)
    cases.append(("nearly_vertical", u, d, adjoint(u, d, lam) + 1e-8 * a, lam))
    un = dn = torch.ones(2, 1, dtype=torch.float64)
    cases.append(("nonunique_multiplier", un, dn, torch.stack((un, dn)), torch.full((2,), .8, dtype=torch.float64)))
    amp = torch.logspace(-3, 3, 9, dtype=torch.float64)
    cases.append(("canonical_row_amplitude", u * amp[:, None], d * amp[:, None], a, None))
    _, _, previous_a = random_example(2021, m=9, n=3)
    old = reference(u, d, previous_a).lam
    cases.append(("rapid_momentum", u, d, a, old))
    for name, u, d, a, warm in cases:
        for dtype in (torch.float64, torch.float32):
            uu, dd, aa = (v.to(dtype) for v in (u, d, a))
            ref = reference(uu, dd, aa)
            for method in ("gd", "pgd", "lbfgs", "newton"):
                r = solve_dual(uu, dd, aa, method=method, initial_lambda=warm,
                               tolerance=1e-6 if dtype == torch.float64 else 3e-5)
                records.append(record(r, ref, uu, dd, aa, case=name, mode="adversary", step=0))
        print("adversary", name, "completed", flush=True)


def supplement(records):
    """Matched float32/float64 target and analytic near-rank direction check."""
    for name, frames in (("ema_proxy", synthetic_sequence()),
                         ("swiglu_fixed_batch", toy_swiglu_sequence())):
        frames = [tuple(v.float() for v in f) for f in frames]
        refs = [reference(*f) for f in frames]
        for method in ("gd", "pgd", "lbfgs", "newton"):
            previous = refs[0].lam
            for t, f in enumerate(frames[1:], 1):
                r = solve_dual(*f, method=method, initial_lambda=previous, tolerance=1e-6)
                previous = r.lam
                records.append(record(r, refs[t], *f, case=name, mode="matched_warm", step=t,
                                      requested_normalized_gap=1e-6))
    u = d = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], dtype=torch.float64)
    a = torch.zeros(2, 3, 2, dtype=torch.float64)
    a[:, 0, 0], a[:, 1, 1] = 1., 1e-12
    exact = a.clone()
    exact[:, 1, 1] = 1.
    r = solve_dual(u, d, a)
    records.append({"case": "analytic_rank_boundary", "shape": [3, 2], "dtype": "torch.float64",
                    "mode": "analytic_comparison", "method": "newton", "fallback": r.fallback,
                    "metrics": r.metrics, **comparison(u, d, a, r.pair, exact)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/dual_solver_results.json")
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--supplement", action="store_true", help="append small matched-tolerance checks to existing results")
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = []
    def save():
        output = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
                   "platform": platform.platform(), "threads": torch.get_num_threads(),
                   "certificate_dtype": "float64", "finite_step_size": .01,
                   "reference_rho": "1/sqrt(n)"}, "records": records}
        Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    if args.supplement:
        records = json.loads(Path(args.output).read_text())["records"]
        # Make this command idempotent while leaving original timings intact.
        records = [r for r in records if r['mode'] not in ('matched_warm', 'analytic_comparison')]
        supplement(records)
        save()
        print("supplemented", args.output, len(records), "records", flush=True)
        return
    sequences = [("ema_proxy", synthetic_sequence()), ("swiglu_fixed_batch", toy_swiglu_sequence())]
    for name, frames in sequences:
        for dtype in (torch.float64, torch.float32):
            run_sequence(frames, name, dtype, records)
            save()
    adversaries(records)
    save()
    shapes = [(256, 96)] + ([(768, 256), (1536, 512)] if args.large else [])
    for m, n in shapes:
        frames = synthetic_sequence(m, n, steps=2)
        for dtype in (torch.float64, torch.float32):
            first, second = [tuple(v.to(dtype) for v in f) for f in frames]
            r0, ref = reference(*first), reference(*second)
            records.append(record(ref, ref, *second, case="size", mode="reference", step=1))
            for method in ("gd", "pgd", "lbfgs", "newton"):
                for mode, warm in (("cold", None), ("warm", r0.lam)):
                    r = solve_dual(*second, method=method, initial_lambda=warm,
                                   tolerance=1e-6 if dtype == torch.float64 else 3e-5)
                    records.append(record(r, ref, *second, case="size", mode=mode, step=1))
            print("size", m, n, str(dtype), "completed", flush=True)
            save()
    print("wrote", args.output, len(records), "records", flush=True)


if __name__ == "__main__":
    main()
