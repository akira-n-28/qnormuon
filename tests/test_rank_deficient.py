"""Claims, counterexamples and numerical limits for isolated research prototypes."""

import itertools
import math

import pytest
import torch

from experiments.rank_deficient import (
    active_basis, active_logdet, active_polar, balance_active_pair, leverage,
    partial_polar, pseudo_logdet, ridge_logdet, ridge_polar,
    threshold_partial_polar, weighted_matrix,
)
from qnormuon import canonical_gradients, exact_polar


DT = torch.float64


def rand(seed, *shape):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=DT)


def known_rank(seed, m, n, rank):
    left, _ = torch.linalg.qr(rand(seed, m, m))
    right, _ = torch.linalg.qr(rand(seed + 1, n, n))
    singular = torch.linspace(0.7, 1.5, rank, dtype=DT)
    a = (left[:, :rank] * singular) @ right[:, :rank].T
    return a, left, right, singular


def close(a, b, tol=3e-11):
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


@pytest.mark.parametrize("rank", [0, 1, 3, 4])
def test_partial_polar_minimum_norm_maximizer_and_projectors(rank):
    a, u, v, s = known_rank(300, 7, 4, rank)
    p = partial_polar(a, rank=rank)
    close(p, u[:, :rank] @ v[:, :rank].T)
    close(p.T @ p, v[:, :rank] @ v[:, :rank].T)
    close(p @ p.T, u[:, :rank] @ u[:, :rank].T)
    close((a * p).sum(), s.sum())
    close(p.square().sum(), a.new_tensor(float(rank)))
    close(leverage(p).sum(), a.new_tensor(float(rank)))
    assert (leverage(p) >= -1e-12).all() and (leverage(p) <= 1 + 1e-12).all()
    expected_sv = torch.zeros(4, dtype=DT)
    expected_sv[:rank] = 1
    close(torch.linalg.svdvals(p), expected_sv)
    # Sample the COMPLETE analytically derived family of maximizers P+Uperp Z Vperp^T.
    for seed in range(3):
        if rank < 4:
            z = rand(310 + seed, 7 - rank, 4 - rank)
            z /= 1.1 * torch.linalg.matrix_norm(z, ord=2)
            y = p + u[:, rank:] @ z @ v[:, rank:].T
            assert torch.linalg.matrix_norm(y, ord=2) <= 1 + 3e-12
            close((a * y).sum(), s.sum())
            close(y.square().sum(), p.square().sum() + z.square().sum())
            assert y.norm() > p.norm()
        candidate = rand(320 + seed, 7, 4)
        candidate /= torch.linalg.matrix_norm(candidate, ord=2)
        assert (a * candidate).sum() <= s.sum() + 3e-12
    if rank == 0:
        assert torch.count_nonzero(p) == 0
        assert exact_polar(a).norm() > 1  # Production remains unchanged.
    if rank == 4:
        close(p, exact_polar(a))


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 3e-11), (torch.float32, 3e-5)])
def test_partial_gauge_lift_and_generalized_active_conditioning(dtype, tol):
    up, down = rand(330, 9, 4).to(dtype), rand(331, 4, 9).to(dtype)
    mu = known_rank(332, 9, 4, 2)[0].to(dtype)
    md = known_rank(334, 9, 4, 1)[0].to(dtype)
    h = up.norm(dim=1) / down.norm(dim=0)
    gu, gd = mu / h.sqrt()[:, None], (md * h.sqrt()[:, None]).T
    x = rand(336, 9).to(dtype)
    c = torch.logspace(-6, 6, 9, dtype=dtype)

    def maps(u, d, g_u, g_d):
        mc_u, mc_d, metric = canonical_gradients(u, d, g_u, g_d)
        pu = partial_polar(weighted_matrix(mc_u, x), rank=2)
        pd = partial_polar(weighted_matrix(mc_d, x), rank=1)
        return metric.sqrt()[:, None] * pu, pd / metric.sqrt()[:, None], metric, pu, pd

    du, dd, h, pu, pd = maps(up, down, gu, gd)
    du2, dd2, _, _, _ = maps(up * c[:, None], down / c, gu / c[:, None], gd * c)
    close(du2 / c[:, None], du, tol)
    close(dd2 * c[:, None], dd, tol)
    for a, p, delta, metric in ((mu, pu, du, h), (md, pd, dd, 1 / h)):
        basis = active_basis(a, rank=2 if a is mu else 1)
        gram = delta.T @ (delta / metric[:, None])
        close(gram, basis @ basis.T, tol)
        close(basis.T @ gram @ basis, torch.eye(basis.shape[1], dtype=dtype), tol)
    error = max((du2 / c[:, None] - du).norm().item() / du.norm().item(),
                (dd2 * c[:, None] - dd).norm().item() / dd.norm().item())
    # Other canonical-only maps also preserve gauge, away from rank thresholds.
    ordinary = canonical_gradients(up, down, gu, gd)
    gauged = canonical_gradients(up * c[:, None], down / c, gu / c[:, None], gd * c)
    for mapping in (lambda a: ridge_polar(a, lam=0.1),
                    lambda a: threshold_partial_polar(a, atol=0., rtol=1e-3)[0]):
        for i in range(2):
            close(mapping(weighted_matrix(ordinary[i], x)),
                  mapping(weighted_matrix(gauged[i], x)), tol)
    print(f"partial gauge {dtype} 9x4 ranks=(2,1), c=1e-6..1e6: relative error={error:.3e}")


@pytest.mark.parametrize("ranks", [(0, 0), (0, 2), (1, 2), (3, 1)])
def test_pseudodeterminant_active_restriction_gradients_hessian_and_basis_invariance(ranks):
    m, n = 7, 4
    matrices = [known_rank(350 + 2 * j, m, n, rank)[0] for j, rank in enumerate(ranks)]
    bases = [active_basis(a, rank=r) for a, r in zip(matrices, ranks)]
    x = (0.3 * rand(355, m)).requires_grad_()
    target = sum(ranks) / m

    def objective(z):
        return sum(active_logdet(a, z, v) for a, v in zip(matrices, bases)) - target * z.sum()

    ps = [active_polar(a, x, v) for a, v in zip(matrices, bases)]
    for a, r, v, p in zip(matrices, ranks, bases, ps):
        close(active_logdet(a, x, v), pseudo_logdet(a, x, rank=r))
        close(p, partial_polar(weighted_matrix(a, x), rank=r))
        if r:
            rotation, _ = torch.linalg.qr(rand(356, r, r))
            close(active_logdet(a, x, v @ rotation), active_logdet(a, x, v))
            close(active_polar(a, x, v @ rotation), p)
    expected = sum(leverage(p) for p in ps) - target
    grad = torch.autograd.grad(objective(x), x)[0]
    close(grad, expected)
    # The SVD-value objective independently gives the same first derivative.
    pseudo = sum(pseudo_logdet(a, x, rank=r) for a, r in zip(matrices, ranks)) - target * x.sum()
    close(torch.autograd.grad(pseudo, x)[0], expected)
    step = 1e-5
    fd = torch.stack([(objective(x + step * e) - objective(x - step * e)) / (2 * step)
                      for e in torch.eye(m, dtype=DT)])
    close(fd, expected, 2e-8)
    hessian = torch.autograd.functional.hessian(objective, x)
    expected_hessian = torch.zeros(m, m, dtype=DT)
    for p in ps:
        q = p @ p.T
        expected_hessian += torch.diag(q.diag()) - q.square()
    close(hessian, expected_hessian)
    assert torch.linalg.eigvalsh(hessian).min() >= -3e-11
    close(hessian @ torch.ones(m, dtype=DT), torch.zeros(m, dtype=DT))
    close(objective(x + 4), objective(x))
    print(f"active/PDet float64 7x4 ranks={ranks}: grad max error="
          f"{(grad-expected).abs().max().item():.3e}, FD={(fd-expected).detach().abs().max().item():.3e}")


def test_rank_adaptive_balancing_distinct_ranks_and_zero_cases():
    m, n = 6, 4
    nodes = torch.linspace(-1, 1, m, dtype=DT)
    mu = torch.zeros(m, n, dtype=DT)
    md = torch.zeros_like(mu)
    mu[:, 0] = torch.linspace(-0.8, 0.8, m, dtype=DT).exp()
    md[:, :2] = torch.stack((torch.ones_like(nodes), nodes), dim=1)
    md *= torch.linspace(0.7, -0.7, m, dtype=DT).exp()[:, None]
    for rows in itertools.combinations(range(m), 2):
        assert torch.linalg.det(md[list(rows), :2]).abs() > 0.01
    solutions = [balance_active_pair(mu, md, ranks=(1, 2), x0=x0)
                 for x0 in (torch.zeros(m, dtype=DT), rand(360, m), rand(360, m) + 10)]
    for result in solutions:
        assert result.converged and result.max_residual < 1e-10
        assert result.target == 0.5
        close(result.x, solutions[0].x, 2e-8)
        close(leverage(result.up).sum(), mu.new_tensor(1.))
        close(leverage(result.down).sum(), mu.new_tensor(2.))
    zero = torch.zeros_like(mu)
    one_side = balance_active_pair(zero, mu, ranks=(0, 1))
    assert one_side.converged
    assert one_side.target == 1 / m
    both_zero = balance_active_pair(zero, zero, ranks=(0, 0), x0=rand(361, m))
    assert both_zero.converged and both_zero.target == 0
    assert both_zero.up.norm() == 0 and both_zero.down.norm() == 0
    print(f"active balancing float64 6x4 ranks=(1,2): residual={solutions[0].max_residual:.3e}")


def test_rank_adaptive_target_still_needs_support_feasibility_and_uniqueness():
    a = torch.tensor([[1., 0.], [0., 0.], [0., 0.]], dtype=DT)
    result = balance_active_pair(a, a, ranks=(1, 1), steps=50)
    assert not result.converged
    assert result.max_residual == pytest.approx(4 / 3)
    # A feasible but nonunique example: disjoint row supports.
    mu = torch.tensor([[1., 0.], [1., 0.], [0., 0.], [0., 0.]], dtype=DT)
    md = torch.tensor([[0., 0.], [0., 0.], [0., 1.], [0., 1.]], dtype=DT)
    x0 = torch.tensor([3., 3., -3., -3.], dtype=DT)
    result = balance_active_pair(mu, md, ranks=(1, 1), x0=x0)
    assert result.converged
    close(result.x, x0)
    close(leverage(result.up) + leverage(result.down), torch.full((4,), 0.5, dtype=DT))


def test_rank_adaptive_boundary_target_requires_infinite_scaling():
    # Rank (1,1), ambient n=3. Target=(.5,.5,.5,.5) lies on support boundary.
    mu = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 0., 0.], [0., 0., 0.]], dtype=DT)
    md = torch.tensor([[0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]], dtype=DT)
    for t in (0., 10., 24.):
        x = torch.tensor([-t, -t, 0., 0.], dtype=DT)
        ps = [partial_polar(weighted_matrix(a, x), rank=1) for a in (mu, md)]
        ell = leverage(ps[0]) + leverage(ps[1])
        assert ell[:2].sum() > 1  # Mu already contributes exactly one there.
    assert (ell - 0.5).abs().max() < 1e-9


def test_uniform_target_cancels_in_centered_dynamics_even_with_balance_momentum():
    matrices = [known_rank(365 + j, 6, 4, j + 1)[0] for j in range(2)]
    xs = [torch.zeros(6, dtype=DT) for _ in range(2)]
    qs = [torch.zeros(6, dtype=DT) for _ in range(2)]
    for _ in range(12):
        for x, q, target in zip(xs, qs, (8 / 6, 3 / 6)):
            ell = sum(leverage(partial_polar(weighted_matrix(a, x), rank=r))
                      for a, r in zip(matrices, (1, 2)))
            q.mul_(0.9).add_(ell - target, alpha=0.1)
            x.add_(q, alpha=-0.5)
            x.sub_(x.mean())
        close(xs[0], xs[1])
        close(qs[0] - qs[0].mean(), qs[1] - qs[1].mean())
    assert abs((qs[0].mean() - qs[1].mean()).item()) > 0.1


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_ridge_objective_gradient_effective_rank_and_convexity(rank):
    a = known_rank(370, 6, 4, rank)[0]
    x = (0.3 * rand(373, 6)).requires_grad_()
    lam = 0.2
    p = ridge_polar(weighted_matrix(a, x), lam=lam)
    f = lambda z: ridge_logdet(a, z, lam=lam)
    grad = torch.autograd.grad(f(x), x)[0]
    ell = leverage(p)
    close(grad, ell)
    s = torch.linalg.svdvals(weighted_matrix(a, x))
    effective_rank = (s.square() / (s.square() + lam)).sum()
    close(ell.sum(), effective_rank)
    q = p @ p.T
    hessian = torch.autograd.functional.hessian(f, x)
    close(hessian, torch.diag(ell) - q.square())
    assert torch.linalg.eigvalsh(hessian).min() >= -3e-11
    step = 1e-5
    fd = torch.stack([(f(x + step * e) - f(x - step * e)) / (2 * step)
                      for e in torch.eye(6, dtype=DT)])
    close(fd, ell, 2e-8)
    if rank:
        assert 0 < effective_rank < rank
        assert (hessian @ torch.ones(6, dtype=DT)).norm() > 1e-3
        # The original rank target has strictly negative derivative along +1.
        assert (grad - rank / 6).sum() < 0
        assert not torch.allclose(p.T @ p, active_basis(a, rank=rank) @ active_basis(a, rank=rank).T)
        assert (a * ridge_polar(a, lam=lam)).sum() < torch.linalg.svdvals(a).sum()
    else:
        assert p.norm() == 0 and ell.sum() == 0
    print(f"ridge float64 6x4 rank={rank} lambda=.2: effective rank={effective_rank.item():.6f}, "
          f"gradient error={(grad-ell).abs().max().item():.3e}")


def test_ridge_centered_gradient_is_effective_leverage_residual():
    a = known_rank(380, 5, 3, 2)[0]
    z = rand(382, 5).requires_grad_()
    x = z - z.mean()
    f = ridge_logdet(a, x, lam=0.1)
    grad = torch.autograd.grad(f, z)[0]
    ell = leverage(ridge_polar(weighted_matrix(a, x), lam=0.1))
    close(grad, ell - ell.mean())
    # Centering defines a constrained objective, rather than an irrelevant scale.
    uncentered = ridge_logdet(a, x + 2, lam=0.1) - (2 / 5) * (x + 2).sum()
    original = ridge_logdet(a, x, lam=0.1) - (2 / 5) * x.sum()
    assert abs((uncentered - original).item()) > 0.01


def test_renormalized_ridge_limit_equals_pseudodeterminant_on_fixed_rank():
    a = known_rank(390, 6, 4, 2)[0]
    x = rand(392, 6) * 0.2
    value = pseudo_logdet(a, x, rank=2)
    errors = []
    for lam in (1e-2, 1e-4, 1e-6, 1e-8):
        corrected = ridge_logdet(a, x, lam=lam) - (4 - 2) * math.log(lam)
        errors.append(abs((corrected - value).item()))
    assert all(a > b for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 1e-6
    print(f"renormalized ridge float64 6x4 rank=2: errors={errors}")


@pytest.mark.parametrize("epsilon", [1e-2, 1e-6, 1e-12, 1e-30])
def test_rank_boundary_discontinuity_partial_pseudodet_active_and_continuity_ridge(epsilon):
    a0 = torch.tensor([[1., 0.], [0., 0.], [0., 0.]], dtype=DT)
    plus, minus = a0.clone(), a0.clone()
    plus[1, 1], minus[1, 1] = epsilon, -epsilon
    x = torch.zeros(3, dtype=DT)
    p0 = partial_polar(a0, rank=1)
    pp, pm = partial_polar(plus, rank=2), partial_polar(minus, rank=2)
    assert (pp - p0).norm().item() == pytest.approx(1.)
    assert (pp - pm).norm().item() == pytest.approx(2.)
    assert leverage(pp).sum() == 2 and leverage(p0).sum() == 1
    assert (pp.T @ pp - p0.T @ p0).norm() == 1
    f0 = pseudo_logdet(a0, x, rank=1)
    fp = pseudo_logdet(plus, x, rank=2)
    close(f0, a0.new_tensor(0.))
    close(fp, a0.new_tensor(2 * math.log(epsilon)))
    close(active_logdet(plus, x, active_basis(plus, rank=2)), fp)
    close(active_polar(plus, x, active_basis(plus, rank=2)), pp)
    # Freezing the OLD one-dimensional subspace makes a continuous different map.
    old_basis = active_basis(a0, rank=1)
    close(active_polar(plus, x, old_basis), p0)
    close(active_logdet(plus, x, old_basis), f0)
    assert (plus * pp).sum() >= (plus * p0).sum()
    lam = 1e-4
    ridge0, ridge_plus = ridge_polar(a0, lam=lam), ridge_polar(plus, lam=lam)
    expected = epsilon / math.sqrt(epsilon**2 + lam)
    assert (ridge_plus - ridge0).norm().item() == pytest.approx(expected)
    assert (leverage(ridge_plus) - leverage(ridge0)).norm().item() == pytest.approx(expected**2)
    ridge_delta = ridge_logdet(plus, x, lam=lam) - ridge_logdet(a0, x, lam=lam)
    close(ridge_delta, a0.new_tensor(math.log1p(epsilon**2 / lam)))


def test_partial_zero_limit_and_noncommuting_ridge_limits():
    base = torch.tensor([[1., 0.], [0., 1.], [0., 0.]], dtype=DT)
    zero = torch.zeros_like(base)
    assert partial_polar(zero, rank=0).norm() == 0
    for t in (1e-2, 1e-6, 1e-12):
        close(partial_polar(t * base, rank=2), base)
        assert ridge_polar(t * base, lam=0.01).norm() <= 15 * t
        # lambda=t^2 leaves a nonzero limit; scale-relative ridge does not fix this.
        close(ridge_polar(t * base, lam=t*t), base / math.sqrt(2))


def test_normalizing_pseudodeterminant_value_does_not_remove_rank_discontinuity():
    a0 = torch.tensor([[1., 0.], [0., 0.], [0., 0.]], dtype=DT)
    a = a0.clone()
    a[1, 1] = 1e-12
    x = torch.tensor([0.2, 0.7, 0.], dtype=DT)
    zero = torch.zeros_like(x)
    f0 = pseudo_logdet(a0, x, rank=1) - pseudo_logdet(a0, zero, rank=1)
    f = pseudo_logdet(a, x, rank=2) - pseudo_logdet(a, zero, rank=2)
    close(f - f0, a.new_tensor(0.7))


def test_continuity_along_fixed_rank_paths_and_repeated_active_singular_values():
    a, u, v, _ = known_rank(400, 7, 4, 2)
    a = u[:, :2] @ v[:, :2].T  # Repeated nonzero singular values.
    base = partial_polar(a, rank=2)
    errors = []
    for t in (1e-2, 1e-4, 1e-6):
        left = u[:, :2] + t * u[:, 2:4]
        perturbed = left @ v[:, :2].T
        errors.append((partial_polar(perturbed, rank=2) - base).norm().item())
        close(active_polar(perturbed, torch.zeros(7, dtype=DT), v[:, :2]),
              partial_polar(perturbed, rank=2))
        x = torch.zeros(7, dtype=DT)
        close(pseudo_logdet(perturbed, x, rank=2),
              active_logdet(perturbed, x, v[:, :2]))
        assert abs(pseudo_logdet(perturbed, x, rank=2).item()) < 3 * t * t + 1e-12
    assert errors[-1] < 2e-6 and errors[0] > 1000 * errors[-1]


def test_hard_threshold_adds_discontinuity_and_does_not_solve_original_lmo():
    tau = 0.1
    def matrix(t):
        return torch.tensor([[1., 0.], [0., t], [0., 0.]], dtype=DT)
    p_low, r_low = threshold_partial_polar(matrix(tau - 1e-12), atol=tau, rtol=0)
    p_high, r_high = threshold_partial_polar(matrix(tau + 1e-12), atol=tau, rtol=0)
    assert (r_low, r_high) == (1, 2)
    assert (p_low - p_high).norm().item() == pytest.approx(1.)
    a = matrix(0.05)
    p, rank = threshold_partial_polar(a, atol=tau, rtol=0)
    assert rank == 1
    assert (torch.linalg.svdvals(a).sum() - (a * p).sum()).item() == pytest.approx(.05)
    # True rank stays 2 under finite K; computed threshold-rank does not.
    x = torch.tensor([0., 2 * math.log(4), 0.], dtype=DT)
    _, changed_rank = threshold_partial_polar(weighted_matrix(a, x), atol=tau, rtol=0)
    assert changed_rank == 2
    # An absolute threshold also loses the global K-scale invariance.
    _, scaled_rank = threshold_partial_polar(4 * a, atol=tau, rtol=0)
    assert scaled_rank == 2


def test_fixed_top_rank_is_ambiguous_at_positive_singular_value_ties():
    eps = 1e-10
    a = torch.tensor([[1 + eps, 0.], [0., 1.], [0., 0.]], dtype=DT)
    b = torch.tensor([[1., 0.], [0., 1 + eps], [0., 0.]], dtype=DT)
    assert (partial_polar(a, rank=1) - partial_polar(b, rank=1)).norm() > 1.4
    # Keeping the whole repeated active space is basis-independent and continuous.
    close(partial_polar(a, rank=2), partial_polar(b, rank=2))


@pytest.mark.parametrize("dtype,rtol", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_explicit_numerical_rank_differs_from_positive_mathematical_rank(dtype, rtol):
    a = torch.zeros(5, 3, dtype=dtype)
    a[0, 0], a[1, 1], a[2, 2] = 1., 1e-4, 1e-8
    p, selected = threshold_partial_polar(a, atol=0., rtol=rtol)
    assert selected == (3 if dtype == torch.float64 else 2)
    assert leverage(p).sum() == selected
    print(f"threshold {dtype} 5x3 spectrum=(1,1e-4,1e-8), rtol={rtol}: selected rank={selected}")
