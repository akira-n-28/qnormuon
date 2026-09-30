"""Only independent production SVD may authorize research hybrid output."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from experiments import smooth_qdwh_hybrid as hybrid
from qnormuon import coupled_solver as cs


def stationary():
    gen = torch.Generator().manual_seed(735)
    u = torch.randn(18, 5, dtype=torch.float64, generator=gen)
    b = torch.randn(18, 5, dtype=torch.float64, generator=gen)
    return u, u.clone(), torch.stack((b, b.clone()))


@pytest.mark.parametrize("budget", [2, 3])
def test_return_is_rebuilt_from_full_svd_at_identical_lambda(budget):
    u, d, a = stationary()
    seen = []
    original = cs.SmoothDual.evaluate
    def observe(problem, lam):
        # The authoritative evaluator sees the original U,D,A at precisely
        # the returned multiplier, not QDWH factors or an old trial spectrum.
        seen.append((problem.u, problem.d, problem.a, lam))
        return original(problem, lam)
    with patch.object(cs.SmoothDual, "evaluate", observe):
        got = hybrid.solve_hybrid(u, d, a, inner_iterations=budget,
                                  config=cs.SolverConfig(fallback=False))
    assert got.result.converged and hybrid.safe(got.result.metrics, cs.SolverConfig())
    assert got.stats["inner_success"] and got.stats["final_svd_verifications"] == 1
    assert got.stats["svd_rescue_solves"] == 0
    assert got.stats["total_svd_evaluations"] == 1
    assert len(seen) == 1
    assert seen[0][3] is got.result.lam
    assert torch.equal(seen[0][0], u) and torch.equal(seen[0][2], a)
    ref = hybrid.verify(u, d, a, got.result.lam, cs.SolverConfig())
    assert torch.equal(ref.pair, got.result.pair)


@pytest.mark.parametrize("policy", ["continue", "restart"])
def test_failed_verification_is_explicit_svd_rescue(policy):
    u, d, a = stationary()
    warm = torch.full((u.shape[0],), .001, dtype=u.dtype)
    verified = []
    rescue_starts = []
    real_verify, real_solve = hybrid.verify, cs.solve_coupled
    def reject(*args):
        result = real_verify(*args)
        verified.append(result.lam)
        result.converged = False  # simulated false proposal/final-boundary reject
        return result
    def spy(*args, **kwargs):
        if kwargs["config"].max_iterations == 100:
            rescue_starts.append(kwargs["initial_lambda"])
        return real_solve(*args, **kwargs)
    with patch.object(hybrid, "verify", reject), patch.object(cs, "solve_coupled", spy):
        got = hybrid.solve_hybrid(u, d, a, initial_lambda=warm, rescue_policy=policy,
                                  config=cs.SolverConfig(fallback=False))
    assert got.result.converged
    assert got.stats["failed_final_verifications"] == 1
    assert got.stats["svd_rescue_solves"] == 1
    assert got.stats["fallback_reason"] == "final_svd_verification_rejected"
    assert torch.equal(rescue_starts[0], verified[0] if policy == "continue" else warm)
    assert not got.stats["cpu_admm_fallback"]


def test_structural_failure_does_not_return_qdwh_pair():
    u, d, a = stationary()
    with patch.object(hybrid, "diagnose", return_value=SimpleNamespace(
            accepted=False, reason="h_not_spd")):
        got = hybrid.solve_hybrid(u, d, a, config=cs.SolverConfig(fallback=False))
    assert got.result.converged
    assert got.stats["structural_rejects"] == 1
    assert got.stats["svd_rescue_solves"] == 1
    assert got.stats["final_svd_verifications"] == 0
    assert got.stats["fallback_reason"] == "qdwh_structural_h_not_spd"
    assert got.stats["output_authority"] == "production_svd_rescue"


def test_inner_budget_is_global_and_rescue_failure_is_not_success():
    u, d, a = stationary()
    a[1] += .2 * u
    warm = torch.ones(u.shape[0], dtype=u.dtype)
    got = hybrid.solve_hybrid(u, d, a, initial_lambda=warm, evaluation_budget=1,
                              config=cs.SolverConfig(fallback=False, max_iterations=0))
    assert got.stats["qdwh_evaluations"] == 1
    assert got.stats["fallback_reason"] == "qdwh_evaluation_budget"
    assert got.stats["svd_rescue_solves"] == 1
    assert not got.result.converged
    assert got.result.reason.startswith("hybrid_uncertified:")


@pytest.mark.parametrize("key,value", [
    ("normalized_gap", 3.00001e-5), ("signed_normalized_gap", -1.00001e-10),
    ("normalized_horizontal_residual", 1.00001e-10), ("spectral_excess", 1.00001e-12),
    ("residual_rcond", 1e-4), ("normalized_gap", float("nan")),
])
def test_all_five_final_conditions_remain_authoritative(key, value):
    metrics = dict(normalized_gap=1e-6, signed_normalized_gap=1e-6,
                   normalized_horizontal_residual=0., spectral_excess=0., residual_rcond=.1)
    assert hybrid.safe(metrics, cs.SolverConfig())
    metrics[key] = value
    assert not hybrid.safe(metrics, cs.SolverConfig())


def test_no_decision_posterior_or_inner_svd_and_scope_restoration():
    u, d, a = stationary()
    parent = cs.SmoothDual
    original = torch.linalg.svd
    calls = []
    def observed(b, **kwargs):
        calls.append(b.shape)
        return original(b, **kwargs)
    with patch.object(torch.linalg, "svd", observed):
        got = hybrid.solve_hybrid(u, d, a, config=cs.SolverConfig(fallback=False))
    assert got.result.converged
    assert calls == [torch.Size([2, 18, 5])]
    assert cs.SmoothDual is parent
    assert cs.SolverConfig().dtype == torch.float64


def test_fixed_backend_repeats_and_configuration_limits():
    u, d, a = stationary()
    cfg = cs.SolverConfig(fallback=False)
    x = hybrid.solve_hybrid(u, d, a, config=cfg)
    y = hybrid.solve_hybrid(u, d, a, config=cfg)
    assert torch.equal(x.result.lam, y.result.lam)
    assert torch.equal(x.result.pair, y.result.pair)
    for kwargs in (dict(inner_iterations=4), dict(rescue_policy="hidden"),
                   dict(config=replace(cfg, dtype=torch.float32))):
        with pytest.raises(ValueError):
            hybrid.solve_hybrid(u, d, a, **kwargs)
