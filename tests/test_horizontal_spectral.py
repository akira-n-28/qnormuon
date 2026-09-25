"""Adversarial checks for the isolated coupled horizontal spectral LMO."""

import pytest
import torch

from experiments.horizontal_spectral import (
    ExposedFace, adjoint, diagnostics, dual_value, fractional_example,
    horizontal_project, horizontal_residual, minimum_norm_on_dual_face,
    partial_pair, random_example, smooth_dual_gradient, solve_lmo,
)
from experiments.zero_stratum import balanced_factors, stable_norm


DTYPE = torch.float64


def close(x, y, atol=3e-9):
    torch.testing.assert_close(x, y, rtol=atol, atol=atol)


def certified(u, d, a, result, tol=3e-8):
    assert result.converged
    stat = diagnostics(u, d, a, result.pair)
    assert stat["horizontal"] < tol
    assert stat["spectral_excess"] < tol
    assert -tol < result.gap < tol * max(1., float(a.norm()))


def test_horizontal_condition_is_metric_orthogonality_and_minimum_lift():
    u, d, p = random_example()
    h = torch.logspace(-8, 8, u.shape[0], dtype=DTYPE)
    raw_u, raw_d = h.sqrt()[:, None] * u, d / h.sqrt()[:, None]
    lifted_u, lifted_d = h.sqrt()[:, None] * p[0], p[1] / h.sqrt()[:, None]
    metric_vertical = (raw_u * lifted_u).sum(1) / h - h * (raw_d * lifted_d).sum(1)
    close(metric_vertical, horizontal_residual(u, d, p), 1e-12)
    projected = horizontal_project(u, d, p)
    close(horizontal_residual(u, d, projected), torch.zeros_like(h), 1e-12)
    vertical = p - projected
    close((vertical * projected).sum(), torch.tensor(0., dtype=DTYPE), 1e-12)
    assert float(projected.square().sum()) < float(p.square().sum())
    for row in range(u.shape[0]):
        before = torch.outer(p[1, row], u[row]) + torch.outer(d[row], p[0, row])
        after = torch.outer(projected[1, row], u[row]) + torch.outer(d[row], projected[0, row])
        close(before, after, 1e-12)


@pytest.mark.parametrize("seed", [901, 902, 903])
def test_full_rank_strong_duality_stationarity_and_unique_solutions(seed):
    u, d, a = random_example(seed)
    result = solve_lmo(u, d, a)
    certified(u, d, a, result)
    b = a - adjoint(u, d, result.lam)
    assert float(torch.linalg.svdvals(b).min()) > .05
    polars = partial_pair(b, (3, 3))
    close(result.pair, polars)
    close(smooth_dual_gradient(u, d, a, result.lam), torch.zeros(7, dtype=DTYPE))
    hessian = torch.autograd.functional.hessian(lambda x: dual_value(u, d, a, x), result.lam)
    assert float(torch.linalg.eigvalsh(hessian).min()) > .05
    print("full-rank", seed, "gap", result.gap, "iterations", result.iterations,
          "min Hessian eigenvalue", float(torch.linalg.eigvalsh(hessian).min()))


def test_dual_gradient_and_curvature_formula_against_independent_derivatives():
    u, d, a = random_example(915, m=6, n=2)
    lam = torch.linspace(-.3, .2, 6, dtype=DTYPE)
    gradient = torch.autograd.functional.jacobian(lambda x: dual_value(u, d, a, x), lam)
    close(gradient, smooth_dual_gradient(u, d, a, lam), 1e-12)
    eye = torch.eye(6, dtype=DTYPE)
    fd = torch.stack([(dual_value(u, d, a, lam + 1e-5 * v) -
                       dual_value(u, d, a, lam - 1e-5 * v)) / 2e-5 for v in eye])
    close(gradient, fd, 2e-8)
    direction = torch.linspace(-.7, .8, 6, dtype=DTYPE)
    curvature = torch.tensor(0., dtype=DTYPE)
    for b, e in zip(a - adjoint(u, d, lam), -adjoint(u, d, direction)):
        left, sigma, right = torch.linalg.svd(b, full_matrices=False)
        f = left.T @ e @ right.T
        normal = e @ right.T - left @ f
        curvature += (normal.square().sum(0) / sigma).sum()
        for i in range(2):
            for j in range(i + 1, 2):
                curvature += (f[i, j] - f[j, i]).square() / (sigma[i] + sigma[j])
    hess = torch.autograd.functional.hessian(lambda x: dual_value(u, d, a, x), lam)
    close(curvature, direction @ hess @ direction, 1e-11)


def test_full_column_rank_does_not_imply_unique_multiplier():
    u = d = torch.ones(2, 1, dtype=DTYPE)
    a = torch.stack((u, d))
    expected = a / 2. ** .5
    for t in (-.8, 0., .8):
        lam = torch.full((2,), t, dtype=DTYPE)
        close(dual_value(u, d, a, lam), torch.tensor(2 * 2. ** .5, dtype=DTYPE))
        close(partial_pair(a - adjoint(u, d, lam), (1, 1)), expected)
    hess = torch.autograd.functional.hessian(lambda x: dual_value(u, d, a, x), torch.zeros(2, dtype=DTYPE))
    close(hess @ torch.ones(2, dtype=DTYPE), torch.zeros(2, dtype=DTYPE), 1e-12)


def test_same_selected_primal_at_full_rank_and_deficient_dual_optimizers():
    u = d = torch.ones(2, 1, dtype=DTYPE)
    a = torch.stack((u, d))
    interior = minimum_norm_on_dual_face(u, d, a, torch.zeros(2, dtype=DTYPE), ranks=(1, 1))
    endpoint = minimum_norm_on_dual_face(u, d, a, torch.ones(2, dtype=DTYPE), ranks=(0, 1))
    certified(u, d, a, interior)
    certified(u, d, a, endpoint)
    close(interior.pair, endpoint.pair)
    close(endpoint.pair, a / 2. ** .5)


def test_one_side_deficient_requires_fractional_completion_and_new_leverage_mass():
    u, d, a, expected = fractional_example()
    lam = torch.zeros(3, dtype=DTYPE)
    separate = partial_pair(a, (2, 1))
    assert float(horizontal_residual(u, d, separate).abs().max()) == .5
    result = minimum_norm_on_dual_face(u, d, a, lam, ranks=(2, 1))
    certified(u, d, a, result)
    close(result.pair, expected, 2e-11)
    close(torch.linalg.svdvals(result.pair[1]), torch.tensor([1., .5], dtype=DTYPE), 2e-11)
    assert abs(float(result.pair.square().sum()) - 3.25) < 2e-11
    assert abs(float(result.pair.square().sum()) - sum((2, 1))) > .24
    generic = solve_lmo(u, d, a)
    certified(u, d, a, generic)
    close(generic.pair, expected)


def test_both_sides_deficient_separate_partial_polars_are_not_stationary_subgradient():
    u = d = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], dtype=DTYPE)
    a = torch.zeros(2, 3, 2, dtype=DTYPE)
    a[0, 0, 0] = a[1, 1, 1] = 1.
    lam = torch.zeros(3, dtype=DTYPE)
    result = minimum_norm_on_dual_face(u, d, a, lam, ranks=(1, 1))
    expected = torch.tensor([[1., 0.], [0., 1.], [0., 0.]], dtype=DTYPE).expand(2, -1, -1)
    certified(u, d, a, result)
    close(result.pair, expected)
    assert float(horizontal_residual(u, d, partial_pair(a, (1, 1))).abs().max()) == 1.
    # Both pairs are subgradients, but only the jointly completed pair maps to 0.
    face = ExposedFace(a, (1, 1))
    close(face.project(result.pair), result.pair)
    assert float(result.pair.square().sum()) > 3.999999
    close(dual_value(u, d, a, lam), (a * result.pair).sum())


def test_nonunique_optimal_face_has_unique_minimum_norm_at_deficient_rank():
    u = d = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], dtype=DTYPE)
    a = torch.zeros(2, 3, 2, dtype=DTYPE)
    a[:, 0, 0] = 1
    result = minimum_norm_on_dual_face(u, d, a, torch.zeros(3, dtype=DTYPE), ranks=(1, 1))
    certified(u, d, a, result)
    close(result.pair, a)
    for t in (-1., -.4, .7, 1.):
        other = a.clone()
        other[:, 1, 1] = t
        close(horizontal_residual(u, d, other), torch.zeros(3, dtype=DTYPE))
        close((a * other).sum(), (a * result.pair).sum())
        close(other.square().sum() - result.pair.square().sum(), torch.tensor(2 * t * t, dtype=DTYPE))


@pytest.mark.parametrize("zero_weight_row", [False, True])
def test_zero_momentum_and_pure_vertical_objectives_select_exact_zero(zero_weight_row):
    u, d, a = random_example()
    if zero_weight_row:
        u[0] = d[0] = 0
    zero = torch.zeros_like(a)
    result = solve_lmo(u, d, zero)
    assert torch.equal(result.pair, zero)
    lam = torch.linspace(-.7, .8, 7, dtype=DTYPE)
    vertical = adjoint(u, d, lam)
    selected = minimum_norm_on_dual_face(u, d, vertical, lam, ranks=(0, 0))
    assert selected.converged and torch.equal(selected.pair, zero)
    if zero_weight_row:
        shifted = lam.clone()
        shifted[0] += 100
        close(dual_value(u, d, vertical, shifted), torch.tensor(0., dtype=DTYPE))


@pytest.mark.parametrize("t", [1e-2, 1e-8, 1e-16, 1e-30, -1e-12])
def test_near_singular_exact_selection_is_discontinuous_and_gap_is_not_direction_error(t):
    u = d = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], dtype=DTYPE)
    a = torch.zeros(2, 3, 2, dtype=DTYPE)
    a[:, 0, 0] = 1
    zero_rank = minimum_norm_on_dual_face(u, d, a, torch.zeros(3, dtype=DTYPE), ranks=(1, 1))
    a[:, 1, 1] = t
    result = minimum_norm_on_dual_face(u, d, a, torch.zeros(3, dtype=DTYPE), ranks=(2, 2))
    certified(u, d, a, result)
    close((result.pair - zero_rank.pair).norm(), torch.tensor(2. ** .5, dtype=DTYPE), 1e-12)
    # The missed objective is tiny while the direction discrepancy stays sqrt(2).
    close((a * (result.pair - zero_rank.pair)).sum(), torch.tensor(2 * abs(t), dtype=DTYPE), 1e-12)


def test_objective_scaling_discontinuity_at_zero_and_vertical_shift_invariance():
    u, d, a = random_example()
    base = solve_lmo(u, d, a)
    for t in (1e-12, -1e-12):
        result = solve_lmo(u, d, t * a)
        close(result.pair, base.pair * (1 if t > 0 else -1))
        assert float(result.pair.norm()) > 2
    shift = adjoint(u, d, torch.linspace(-2, 2, 7, dtype=DTYPE))
    close(solve_lmo(u, d, a + shift).pair, base.pair)


@pytest.mark.parametrize("dtype,extent,tol", [(torch.float64, 120, 3e-8), (torch.float32, 12, 3e-4)])
def test_extreme_positive_gauges_canonical_invariance_and_lifted_equivariance(dtype, extent, tol):
    u, d, a = random_example(dtype=dtype)
    scales = torch.logspace(-extent, extent, u.shape[0], dtype=dtype)
    raw_u, raw_d = scales[:, None] * u, d / scales[:, None]
    canonical = [balanced_factors(ui, di) for ui, di in zip(raw_u, raw_d)]
    uc, dc = torch.stack([x[0] for x in canonical]), torch.stack([x[1] for x in canonical])
    hsqrt = torch.stack([(stable_norm(ui).sqrt() / stable_norm(di).sqrt()) for ui, di in zip(raw_u, raw_d)])
    covariant_a = torch.stack((a[0] / scales[:, None], a[1] * scales[:, None]))
    ac = torch.stack((hsqrt[:, None] * covariant_a[0], covariant_a[1] / hsqrt[:, None]))
    tolerance = 1e-10 if dtype == DTYPE else 2e-6
    base = solve_lmo(u, d, a, tolerance=tolerance)
    transformed = solve_lmo(uc, dc, ac, tolerance=tolerance)
    assert base.converged and transformed.converged
    close(transformed.pair, base.pair, tol)
    unscaled_lift = torch.stack((transformed.pair[0] * (hsqrt / scales)[:, None],
                                 transformed.pair[1] * (scales / hsqrt)[:, None]))
    error = float((unscaled_lift - base.pair).norm() / base.pair.norm())
    assert error < tol
    print("gauge", str(dtype), "shape", list(u.shape), "extent", extent, "relative error", error)


def test_signed_state_covariance_and_failure_without_momentum_transform():
    u, d, a = random_example()
    signs = torch.tensor([-1., 1., -1., 1., -1., 1., -1.], dtype=DTYPE)
    base = solve_lmo(u, d, a)
    transformed = solve_lmo(signs[:, None] * u, signs[:, None] * d, signs[None, :, None] * a)
    close(transformed.pair, signs[None, :, None] * base.pair)
    close(transformed.lam, base.lam)
    wrong = solve_lmo(signs[:, None] * u, signs[:, None] * d, a)
    assert float((wrong.pair - signs[None, :, None] * base.pair).norm()) > 1
    uf, df, af, expected = fractional_example()
    s = signs[:3]
    selected = minimum_norm_on_dual_face(s[:, None] * uf, s[:, None] * df,
        s[None, :, None] * af, torch.zeros(3, dtype=DTYPE), ranks=(2, 1))
    close(selected.pair, s[None, :, None] * expected)


def test_separate_coupled_and_postproject_comparison():
    u, d, a = random_example()
    result = solve_lmo(u, d, a)
    separate = partial_pair(a, (3, 3))
    projected = horizontal_project(u, d, separate)
    repaired = projected / max(1., float(torch.linalg.svdvals(projected).max()))
    stats = {name: diagnostics(u, d, a, p) for name, p in
             (("separate", separate), ("coupled", result.pair), ("postproject", projected), ("repaired", repaired))}
    assert stats["separate"]["objective"] - stats["coupled"]["objective"] > 2.
    assert stats["postproject"]["spectral_excess"] > .09
    assert stats["coupled"]["objective"] - stats["repaired"]["objective"] > 2.2
    print("comparison float64 7x3", stats)


def test_horizontal_objective_still_needs_coupled_spectral_constraints():
    u, d, a = random_example()
    a = horizontal_project(u, d, a)
    separate = partial_pair(a, (3, 3))
    projected = horizontal_project(u, d, separate)
    close((a * separate).sum(), (a * projected).sum(), 1e-12)
    result = solve_lmo(u, d, a)
    certified(u, d, a, result)
    assert float((a * (projected - result.pair)).sum()) > .01
    assert float(torch.linalg.svdvals(projected).max()) > 1.001


def test_shared_K_composes_but_old_logdet_gradient_is_not_coupled_row_mass():
    u, d, a, expected = fractional_example()
    x = torch.zeros(3, dtype=DTYPE, requires_grad=True)
    # Exact active logdet for this diagonal rank-(2,1) example.
    def old_objective(z):
        b = torch.exp(z / 2)[None, :, None] * a
        return 2 * torch.log(torch.linalg.svdvals(b[0])).sum() + 2 * torch.log(torch.linalg.svdvals(b[1])[0])
    old_gradient = torch.autograd.grad(old_objective(x), x)[0]
    mass = expected.square().sum((0, 2))
    close(old_gradient, torch.tensor([2., 1., 0.], dtype=DTYPE), 1e-12)
    close(mass, torch.tensor([2., 1.25, 0.], dtype=DTYPE), 1e-12)
    assert float(((mass - mass.mean()) - (old_gradient - old_gradient.mean())).norm()) > .2
    for row in range(3):
        e = torch.eye(3, dtype=DTYPE)[row] * 1e-5
        close((old_objective(x + e) - old_objective(x - e)) / 2e-5, old_gradient[row], 2e-8)


def test_shared_K_value_derivative_is_envelope_not_leverage():
    u, d, a = random_example(902, m=5, n=2)
    x = torch.linspace(-.3, .2, 5, dtype=DTYPE)
    weighted = torch.exp(x / 2)[None, :, None] * a
    result = solve_lmo(u, d, weighted, tolerance=1e-12)
    certified(u, d, weighted, result)
    envelope = .5 * (weighted * result.pair).sum((0, 2))
    fd = []
    for e in torch.eye(5, dtype=DTYPE) * 1e-4:
        objectives = []
        for z in (x + e, x - e):
            b = torch.exp(z / 2)[None, :, None] * a
            solved = solve_lmo(u, d, b, tolerance=1e-12)
            assert solved.converged
            objectives.append((b * solved.pair).sum())
        fd.append((objectives[0] - objectives[1]) / 2e-4)
    close(torch.stack(fd), envelope, 2e-7)
    assert float((envelope - result.pair.square().sum((0, 2))).norm()) > .1
    print("K envelope float64 5x2 FD max error", float((torch.stack(fd) - envelope).abs().max()))


def test_coupled_row_mass_jacobian_need_not_be_symmetric_even_after_centering():
    u, d, a = random_example(902, m=5, n=2)
    x = torch.linspace(-.3, .2, 5, dtype=DTYPE)
    solved = solve_lmo(u, d, torch.exp(x / 2)[None, :, None] * a, tolerance=1e-12)
    assert float(torch.linalg.svdvals(torch.exp(x / 2)[None, :, None] * a - adjoint(u, d, solved.lam)).min()) > .1
    z = torch.cat((x, solved.lam))

    def pair(v):
        b = torch.exp(v[:5] / 2)[None, :, None] * a - adjoint(u, d, v[5:])
        return partial_pair(b, (2, 2))

    stationarity_jacobian = torch.autograd.functional.jacobian(
        lambda v: -horizontal_residual(u, d, pair(v)), z)
    hessian = stationarity_jacobian[:, 5:]
    assert float(torch.linalg.eigvalsh(hessian).min()) > .05
    dual_response = -torch.linalg.solve(hessian, stationarity_jacobian[:, :5])
    mass_jacobian = torch.autograd.functional.jacobian(lambda v: pair(v).square().sum((0, 2)), z)
    derived = mass_jacobian[:, :5] + mass_jacobian[:, 5:] @ dual_response
    finite_difference = []
    for e in torch.eye(5, dtype=DTYPE) * 1e-4:
        masses = []
        for v in (x + e, x - e):
            r = solve_lmo(u, d, torch.exp(v / 2)[None, :, None] * a, tolerance=1e-12)
            assert r.converged
            masses.append(r.pair.square().sum((0, 2)))
        finite_difference.append((masses[0] - masses[1]) / 2e-4)
    close(derived, torch.stack(finite_difference, dim=1), 2e-7)
    center = torch.eye(5, dtype=DTYPE) - torch.ones(5, 5, dtype=DTYPE) / 5
    skew = center @ (derived - derived.T) @ center
    assert float(skew.norm()) > .01
    print("K centered row-mass Jacobian asymmetry float64 5x2", float(skew.norm()))


def test_derived_log_support_balancing_candidate_convexity_shift_and_gradient():
    u, d, a = random_example(902, m=5, n=2)
    x = torch.linspace(-.3, .2, 5, dtype=DTYPE)
    y = torch.tensor([.4, -.5, .1, .7, -.2], dtype=DTYPE)

    def candidate(z):
        b = torch.exp(z / 2)[None, :, None] * a
        r = solve_lmo(u, d, b, tolerance=1e-12)
        assert r.converged
        value = (b * r.pair).sum()
        return 2 * value.log() - z.mean(), (b * r.pair).sum((0, 2)) / value

    fx, contributions = candidate(x)
    fy, _ = candidate(y)
    fm, _ = candidate(.4 * x + .6 * y)
    assert float(fm) <= float(.4 * fx + .6 * fy) + 1e-9
    close(candidate(x + 3)[0], fx, 1e-9)
    assert float(contributions.min()) >= -1e-10
    close(contributions.sum(), torch.tensor(1., dtype=DTYPE), 1e-12)
    fd = torch.stack([(candidate(x + e)[0] - candidate(x - e)[0]) / 2e-4
                      for e in torch.eye(5, dtype=DTYPE) * 1e-4])
    close(fd, contributions - .2, 2e-7)


def test_solver_limits_and_bad_inputs_are_not_silently_certified():
    u, d, a = random_example()
    assert not solve_lmo(u, d, a, max_iterations=1).converged
    uf, df, af, _ = fractional_example()
    assert not minimum_norm_on_dual_face(uf, df, af, torch.zeros(3, dtype=DTYPE),
                                        ranks=(2, 1), max_iterations=1).converged
    with pytest.raises(ValueError, match="balanced"):
        solve_lmo(2 * u, d, a)
