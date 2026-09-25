"""Quotient geometry, matrix-free derivatives, certificates and fallback tests."""
import pytest
import torch

from experiments.dual_solver import SmoothDual, comparison, reference, solve_dual
from experiments.dual_solver_study import synthetic_sequence
from experiments.horizontal_spectral import (
    adjoint, fractional_example, horizontal_project, horizontal_residual, random_example,
)
from experiments.quotient_spectral import pair_spectral_norm, raw_class_norm, tangent_norm


def close(a, b, tol=2e-9):
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def test_tangent_norm_axioms_cotangent_quotient_and_steepest_step():
    u, d, a = random_example(901)
    p = horizontal_project(u, d, a)
    q = horizontal_project(u, d, a.flip(0))
    np, nq = tangent_norm(u, d, p), tangent_norm(u, d, q)
    assert float(np) > 0
    assert float(tangent_norm(u, d, p + q)) <= float(np + nq) + 1e-12
    close(tangent_norm(u, d, -3 * p), 3 * np)
    zero = torch.zeros_like(p)
    assert tangent_norm(u, d, zero) == 0
    shift = adjoint(u, d, torch.linspace(-3, 3, 7, dtype=a.dtype))
    close(((a + shift) * p).sum(), (a * p).sum())
    oracle = reference(u, d, a)
    shifted = reference(u, d, a + shift)
    close(oracle.pair, shifted.pair)
    close(torch.tensor(oracle.metrics['dual_objective']), torch.tensor(shifted.metrics['dual_objective']))
    assert abs(float(pair_spectral_norm(oracle.pair)) - 1) < 1e-9
    assert abs(float((a * -oracle.pair).sum()) + oracle.metrics['dual_objective']) < 1e-9


def test_horizontal_norm_is_not_infimum_over_vertical_spectral_representatives():
    u, d, a, _ = fractional_example()
    projected = horizontal_project(u, d, a)
    assert float(pair_spectral_norm(a)) == 1
    assert float(tangent_norm(u, d, projected)) > 1.06
    close(horizontal_project(u, d, a - projected), torch.zeros_like(a))


def test_norm_is_nonsmooth_not_strictly_convex_on_regular_tangent_fiber():
    u = d = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], dtype=torch.float64)
    p = torch.zeros(2, 3, 2, dtype=u.dtype)
    p[:, 0, 0] = 1
    q = p.clone()
    q[:, 1, 1] = .5
    for v in (p, q, .5 * (p + q)):
        close(tangent_norm(u, d, v), torch.tensor(1., dtype=u.dtype))
    tie = p.clone()
    tie[:, 1, 1] = 1
    e = torch.zeros_like(p)
    e[:, 1, 1] = 1
    h = 1e-6
    left = (tangent_norm(u, d, tie) - tangent_norm(u, d, tie - h * e)) / h
    right = (tangent_norm(u, d, tie + h * e) - tangent_norm(u, d, tie)) / h
    assert float(left) < 1e-8 and float(right) > .999


@pytest.mark.parametrize('dtype,extent,tol', [(torch.float64, 120, 2e-9), (torch.float32, 12, 3e-5)])
def test_raw_norm_gauge_invariance_vertical_kernel_and_extreme_canonicalization(dtype, extent, tol):
    u, d, velocity = random_example(902, dtype=dtype)
    c = torch.logspace(-extent, extent, 7, dtype=u.dtype)
    value = raw_class_norm(u, d, velocity)
    uc, dc = c[:, None] * u, d / c[:, None]
    vc = torch.stack((c[:, None] * velocity[0], velocity[1] / c[:, None]))
    close(raw_class_norm(uc, dc, vc), value, tol)
    vertical = adjoint(u, d, torch.linspace(-2, 2, 7, dtype=u.dtype))
    assert float(raw_class_norm(u, d, vertical)) < tol
    close(raw_class_norm(u, d, velocity + vertical), value, tol)
    # Solver inputs are rebalanced with stable norms, never a clamped raw ratio.
    from experiments.zero_stratum import balanced_factors
    pairs = [balanced_factors(x, y) for x, y in zip(uc, dc)]
    ub, db = torch.stack([x[0] for x in pairs]), torch.stack([x[1] for x in pairs])
    close(solve_dual(ub, db, velocity).pair, solve_dual(u, d, velocity).pair, tol)


@pytest.mark.parametrize('dtype,tol,step', [(torch.float64, 2e-8, 1e-5), (torch.float32, 5e-3, 1e-3)])
def test_cached_svd_hvp_matches_autograd_finite_differences_and_is_psd(dtype, tol, step):
    u, d, a = random_example(903, m=8, n=3, dtype=dtype)
    lam = torch.linspace(-.1, .2, 8, dtype=dtype)
    v = torch.linspace(-.7, .5, 8, dtype=dtype)
    problem = SmoothDual(u, d, a)
    ev = problem.evaluate(lam)
    analytic = problem.hvp(ev, v)
    fd = (problem.evaluate(lam + step * v).gradient - problem.evaluate(lam - step * v).gradient) / (2 * step)
    close(analytic, fd, tol)
    hess = torch.autograd.functional.hessian(lambda z: torch.linalg.svdvals(a - adjoint(u, d, z)).sum(), lam)
    close(analytic, hess @ v, tol)
    assert float(v @ analytic) >= -tol
    # The row metric supplies an upper curvature bound, not a lower one.
    w = (u.square() + d.square()).sum(1)
    upper = (w * v.square()).sum() / ev.singular.min()
    assert float(v @ analytic) <= float(upper) + tol
    other = v.flip(0)
    close(v @ problem.hvp(ev, other), other @ analytic, tol)
    assert problem.counts.svd_matrices == 6  # HVPs themselves require no SVD


def test_hvp_repeated_positive_singular_values_needs_no_singular_vector_derivative():
    u = d = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], dtype=torch.float64)
    a = torch.tensor([[[1., 0.], [0., 1.], [0., 0.]]] * 2, dtype=u.dtype)
    problem = SmoothDual(u, d, a)
    zero = torch.zeros(3, dtype=u.dtype)
    ev = problem.evaluate(zero)
    v = torch.tensor([.2, -.3, .4], dtype=u.dtype)
    hv = problem.hvp(ev, v)
    fd = (problem.evaluate(1e-5 * v).gradient - problem.evaluate(-1e-5 * v).gradient) / 2e-5
    close(hv, fd, 2e-8)


@pytest.mark.parametrize('method', ['gd', 'pgd', 'lbfgs', 'newton'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_all_methods_report_certified_feasible_steps_and_reference_errors(method, dtype):
    u, d, a = random_example(902, m=12, n=3, dtype=dtype)
    tol = 1e-6 if dtype == torch.float64 else 3e-5
    r = solve_dual(u, d, a, method=method, tolerance=tol)
    oracle = reference(u, d, a)
    assert r.converged and r.metrics['normalized_gap'] <= tol
    assert r.metrics['normalized_horizontal_residual'] < 1e-10
    assert max(r.metrics['spectral_norms']) <= 1 + 1e-12
    errors = comparison(u, d, a, r.pair, oracle.pair)
    assert errors['relative_objective_loss'] < 2 * tol
    assert errors['relative_direction_error'] < .02
    assert errors['relative_finite_delta_x_error'] < .02
    assert r.pair.dtype == torch.float64  # explicit feasibility/certification precision
    assert r.counts.svd_matrices > 0


def test_row_preconditioner_removes_scale_congruence_but_need_not_improve_condition_number():
    u, d, a = random_example(901)
    r = reference(u, d, a)
    ev_problem = SmoothDual(u, d, a)
    ev = ev_problem.evaluate(r.lam)
    eye = torch.eye(7, dtype=u.dtype)
    h = torch.stack([ev_problem.hvp(ev, v) for v in eye], 1)
    scales = h.diag().rsqrt()
    us, ds = scales[:, None] * u, scales[:, None] * d
    shifted = SmoothDual(us, ds, a)
    evs = shifted.evaluate(r.lam / scales)
    hs = torch.stack([shifted.hvp(evs, v) for v in eye], 1)
    close(hs, scales[:, None] * h * scales[None, :])
    w = (us.square() + ds.square()).sum(1)
    hp = hs / w.sqrt()[:, None] / w.sqrt()[None, :]
    close(hp, h / 2)
    assert float(torch.linalg.cond(hp)) > float(torch.linalg.cond(hs))


def test_warm_start_reduces_newton_work_on_slow_sequence_and_budget_reports_failure():
    frames = synthetic_sequence(m=24, n=8, steps=4)
    previous = reference(*frames[0]).lam
    cold_total, warm_total = 0, 0
    for f in frames[1:]:
        cold = solve_dual(*f, tolerance=1e-6)
        warm = solve_dual(*f, initial_lambda=previous, tolerance=1e-6)
        assert warm.converged and cold.converged
        cold_total += cold.iterations
        warm_total += warm.iterations
        previous = warm.lam
    assert warm_total < cold_total
    limited = solve_dual(*frames[-1], max_iterations=0, fallback=False)
    assert not limited.converged and limited.reason == 'iteration_budget'
    assert limited.metrics['normalized_gap'] > 1e-6


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_near_rank_fallback_is_explicit_without_truncating_positive_singular_values(dtype):
    u = d = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], dtype=dtype)
    a = torch.zeros(2, 3, 2, dtype=dtype)
    a[:, 0, 0], a[:, 1, 1] = 1., 1e-12
    r = solve_dual(u, d, a)
    assert r.fallback and r.reason == 'ill_conditioned_residual'
    assert r.counts.reference_iterations > 0
    assert r.min_accepted_rcond < 1e-8
    assert r.metrics['normalized_gap'] < 1e-8
    assert 'primary_only' in r.secondary_selection
    # The reference may miss the tiny active direction: a value certificate
    # does not turn it into an accurate exact-LMO direction at the boundary.
    exact = a.double().clone()
    exact[:, 1, 1] = 1
    assert float((r.pair - exact).norm()) > 1


def test_zero_nearly_vertical_and_nonunique_multiplier_cases_are_explicit():
    u, d, a = random_example(902)
    zero = solve_dual(u, d, torch.zeros_like(a))
    assert torch.equal(zero.pair, torch.zeros_like(a)) and zero.metrics['normalized_gap'] == 0
    lam = torch.linspace(-1, 1, 7, dtype=u.dtype)
    almost = adjoint(u, d, lam) + 1e-7 * a
    r = solve_dual(u, d, almost, initial_lambda=lam, tolerance=1e-5)
    assert r.converged and r.metrics['normalized_gap'] < 1e-5
    un = dn = torch.ones(2, 1, dtype=u.dtype)
    an = torch.stack((un, dn))
    for start in (0., .8):
        result = solve_dual(un, dn, an, initial_lambda=torch.full((2,), start, dtype=u.dtype))
        assert result.converged and result.iterations == 0
        close(result.pair, an / 2 ** .5)


def test_finite_step_error_matches_explicit_neuron_matrix_products():
    u, d, a = random_example(902, m=5, n=2)
    oracle = reference(u, d, a)
    approx = solve_dual(u, d, a, max_iterations=1, fallback=False)
    eta = .01
    original = torch.einsum('mi,mj->mij', d, u)
    def delta(p):
        return torch.einsum('mi,mj->mij', d - eta * p[1], u - eta * p[0]) - original
    expected = (delta(approx.pair) - delta(oracle.pair)).norm() / delta(oracle.pair).norm()
    close(torch.tensor(comparison(u, d, a, approx.pair, oracle.pair)['relative_finite_delta_x_error'], dtype=u.dtype), expected)
    rho = max(float(u.norm(dim=1).max()), float(d.norm(dim=1).max()))
    bound = 2 ** .5 * (eta * rho + eta ** 2) * (approx.pair - oracle.pair).norm()
    assert float((delta(approx.pair) - delta(oracle.pair)).norm()) <= float(bound) + 1e-12

    # An analytic full-rank dual minimizer: identical sides make lambda=0
    # stationary. Test the value-to-direction bound on a feasible competitor.
    same_a = torch.stack((a[0], a[0]))
    q, singular, vt = torch.linalg.svd(a[0], full_matrices=False)
    exact = torch.stack((q @ vt, q @ vt))
    competitor = horizontal_project(u, u, a)
    competitor /= max(1., float(pair_spectral_norm(competitor)))
    gap = (same_a * (exact - competitor)).sum()
    assert float((exact - competitor).square().sum()) <= float(2 * gap / singular.min()) + 1e-12


def test_minus_nuclear_norm_expression_is_not_the_dual_support_value():
    u = d = torch.ones(2, 1, dtype=torch.float64)
    a = torch.stack((u, d))
    minus = torch.linalg.svdvals(a[0]).sum() - torch.linalg.svdvals(a[1]).sum()
    assert minus == 0
    assert reference(u, d, a).metrics['primal_objective'] > 2.8


@pytest.mark.parametrize('force_fallback', [False, True])
def test_svd_accounting_includes_certification_line_search_and_reference(monkeypatch, force_fallback):
    calls = [0]
    original_svd, original_values = torch.linalg.svd, torch.linalg.svdvals
    def wrap(fn):
        def counted(matrix, *args, **kwargs):
            calls[0] += matrix.numel() // (matrix.shape[-1] * matrix.shape[-2])
            return fn(matrix, *args, **kwargs)
        return counted
    monkeypatch.setattr(torch.linalg, 'svd', wrap(original_svd))
    monkeypatch.setattr(torch.linalg, 'svdvals', wrap(original_values))
    u, d, a = random_example(902)
    r = solve_dual(u, d, a, max_iterations=0 if force_fallback else 100)
    assert r.fallback == force_fallback
    assert calls[0] == r.counts.svd_matrices
