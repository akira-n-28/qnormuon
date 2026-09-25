"""Theorem checks and explicit counterexamples; no optimizer policy changes.

Tests named ``counterexample`` assert the observed limitation rather than
silently blessing it as a mathematical guarantee. See docs/THEORY_AUDIT.md.
Run with ``python3 -m pytest -s tests/test_theory_audit.py`` for error measurements.
"""

import copy
import itertools

import pytest
import torch
import torch.nn.functional as F

import qnormuon.core as core
from qnormuon import (
    QNorMuon,
    balance_shared_metric,
    canonical_gradients,
    canonical_pair,
    canonicalize_pair_,
    exact_polar,
    gauge_metric,
    paired_leverage,
    quotient_polar_update,
)


DTYPE = torch.float64


def random_tensors(seed, *shapes):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=generator, dtype=DTYPE) for shape in shapes]


def relative_error(actual, expected):
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-300)).item()


def swiglu(up, down, gate, inputs):
    return (F.silu(inputs @ gate.T) * (inputs @ up.T)) @ down.T


def phi(x, mu, md):
    """Independent differentiable objective; deliberately no ridge or clamp."""
    k = x.exp()
    return sum(torch.linalg.slogdet(a.T @ (k[:, None] * a))[1] for a in (mu, md)) - (
        2.0 * mu.shape[1] / mu.shape[0]
    ) * x.sum()


def polars_at(x, mu, md):
    return tuple(exact_polar((0.5 * x).exp()[:, None] * a) for a in (mu, md))


@pytest.mark.parametrize("seed", [101, 102, 103])
def test_actual_swiglu_function_gradients_and_canonical_coordinates(seed):
    m, n = 12, 4
    up, down, gate, inputs, target = random_tensors(
        seed, (m, n), (n, m), (m, n), (20, n), (20, n)
    )
    c = torch.logspace(-6, 6, m, dtype=DTYPE)
    up.requires_grad_()
    down.requires_grad_()
    gate.requires_grad_()
    up2 = (c[:, None] * up).detach().requires_grad_()
    down2 = (down / c).detach().requires_grad_()
    gate2 = gate.detach().clone().requires_grad_()
    y = swiglu(up, down, gate, inputs)
    y2 = swiglu(up2, down2, gate2, inputs)
    grads = torch.autograd.grad(F.mse_loss(y, target), (up, down, gate))
    grads2 = torch.autograd.grad(F.mse_loss(y2, target), (up2, down2, gate2))
    torch.testing.assert_close(y2, y, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(grads2[0] * c[:, None], grads[0], rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(grads2[1] / c, grads[1], rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(grads2[2], grads[2], rtol=2e-11, atol=2e-11)
    gc = canonical_gradients(up, down, *grads[:2])
    gc2 = canonical_gradients(up2, down2, *grads2[:2])
    for a, b in zip(gc[:2], gc2[:2]):
        torch.testing.assert_close(a, b, rtol=2e-11, atol=2e-11)
    # Gauge orthogonality holds per neuron, not just after summing neurons.
    identity = (grads[0] * up).sum(1) - (grads[1] * down).sum(0)
    assert identity.abs().max().item() < 2e-10
    uc, dc = canonical_pair(up2, down2)
    torch.testing.assert_close(swiglu(uc, dc, gate, inputs), y, rtol=2e-11, atol=2e-11)
    print(f"SwiGLU float64 {m}x{n} seed={seed}: function={relative_error(y2, y):.3e}, "
          f"canonical gradient={relative_error(gc2[0], gc[0]):.3e}")


@pytest.mark.parametrize("canonicalize", [False, True])
def test_optimizer_midtraining_gauge_reset_and_checkpoint(canonicalize):
    up, down, gate, inputs, target = random_tensors(110, (9, 3), (3, 9), (9, 3), (32, 3), (32, 3))
    pairs = [(torch.nn.Parameter(up.clone()), torch.nn.Parameter(down.clone())) for _ in range(2)]
    opts = [QNorMuon([pair], lr=1e-3, canonicalize_weights=canonicalize) for pair in pairs]
    c = torch.logspace(-4, 4, 9, dtype=DTYPE)
    worst = 0.0
    for step in range(9):
        if step == 3:
            # A real checkpoint roundtrip with nonzero momentum and balancing state.
            opts[1] = QNorMuon([pairs[1]])
            opts[1].load_state_dict(copy.deepcopy(opts[0].state_dict()))
            with torch.no_grad():
                pairs[1][0].mul_(c[:, None])
                pairs[1][1].div_(c)
        for (u, d), opt in zip(pairs, opts):
            opt.zero_grad(set_to_none=True)
            F.mse_loss(swiglu(u, d, gate, inputs), target).backward()
            opt.step()
        y0, y1 = [swiglu(u, d, gate, inputs) for u, d in pairs]
        worst = max(worst, relative_error(y1, y0))
        torch.testing.assert_close(y1, y0, rtol=2e-10, atol=2e-10)
        for key in ("momentum_up", "momentum_down_t", "x", "balance_momentum"):
            torch.testing.assert_close(opts[0].state[pairs[0][0]][key],
                                       opts[1].state[pairs[1][0]][key], rtol=2e-10, atol=2e-10)
        if canonicalize:
            for u, d in pairs:
                torch.testing.assert_close(u.norm(dim=1), d.norm(dim=0), rtol=2e-12, atol=2e-12)
    print(f"trajectory float64 9x3 canonicalize={canonicalize}: max function error={worst:.3e}")


def test_weighted_variational_optimum_uses_raw_covector_once():
    up, down, gu, gd = random_tensors(120, (11, 4), (4, 11), (11, 4), (4, 11))
    up *= torch.logspace(-2, 2, 11, dtype=DTYPE)[:, None]
    du, dd, pu, pd, h = quotient_polar_update(up, down, gu, gd)
    gaps = []
    for raw, delta, polar, metric in ((gu, du, pu, h), (gd.T, dd.T, pd, 1 / h)):
        canonical = metric.sqrt()[:, None] * raw
        optimum = torch.linalg.svdvals(canonical).sum()
        torch.testing.assert_close((raw * delta).sum(), optimum, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(delta.T @ (delta / metric[:, None]),
                                   torch.eye(4, dtype=DTYPE), rtol=2e-12, atol=2e-12)
        for seed in range(5):
            candidate = exact_polar(random_tensors(seed, (11, 4))[0])
            assert (canonical * candidate).sum() <= optimum + 2e-12
        # Misreading theorem 4's M as canonical momentum applies H twice.
        wrong = exact_polar(metric.sqrt()[:, None] * canonical)
        gaps.append((optimum - (canonical * wrong).sum()).item())
        assert gaps[-1] > 1e-3
    print(f"variational float64 11x4: double-canonicalization objective gaps={gaps}")


def test_metric_covariance_duality_and_balanced_value():
    up, down = random_tensors(130, (10, 3), (3, 10))
    c = torch.logspace(-6, 6, 10, dtype=DTYPE)
    h = gauge_metric(up, down)
    torch.testing.assert_close(gauge_metric(c[:, None] * up, down / c) / c.square(),
                               h, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(gauge_metric(down.T, up.T), 1 / h, rtol=2e-12, atol=2e-12)
    uc, dc = canonical_pair(up, down)
    torch.testing.assert_close(gauge_metric(uc, dc), torch.ones_like(h), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("seed", [140, 141, 142])
def test_phi_gradient_hessian_and_global_shift(seed):
    m, n = 7, 3
    mu, md, x = random_tensors(seed, (m, n), (m, n), (m,))
    scales = torch.linspace(-2, 2, m, dtype=DTYPE).exp()
    mu *= scales[:, None]
    md /= scales[:, None]
    x *= 0.4
    x.requires_grad_()
    pu, pd = polars_at(x, mu, md)
    analytic = paired_leverage(pu, pd) - 2.0 * n / m
    autodiff = torch.autograd.grad(phi(x, mu, md), x)[0]
    eye = torch.eye(m, dtype=DTYPE)
    step = 1e-5
    finite_diff = torch.stack([(phi(x + step * e, mu, md) - phi(x - step * e, mu, md))
                               / (2 * step) for e in eye])
    torch.testing.assert_close(analytic, autodiff, rtol=2e-10, atol=2e-10)
    torch.testing.assert_close(analytic, finite_diff, rtol=2e-7, atol=2e-7)
    hessian = torch.autograd.functional.hessian(lambda v: phi(v, mu, md), x)
    formula = torch.zeros_like(hessian)
    for p in (pu, pd):
        projection = p @ p.T
        formula += torch.diag(projection.diag()) - projection.square()
    torch.testing.assert_close(hessian, formula, rtol=2e-10, atol=2e-10)
    eigs = torch.linalg.eigvalsh(hessian)
    assert eigs[0] > -2e-10
    assert eigs[1] > 1e-5  # Only the global scaling direction is flat here.
    assert (hessian @ torch.ones(m, dtype=DTYPE)).abs().max() < 2e-10
    torch.testing.assert_close(phi(x + 5, mu, md), phi(x, mu, md), rtol=2e-11, atol=2e-11)
    print(f"Phi float64 {m}x{n} seed={seed}: grad={float((analytic-autodiff).abs().max()):.3e}, "
          f"FD={(analytic-finite_diff).detach().abs().max().item():.3e}, min Hessian eig={float(eigs[0]):.3e}")


def test_full_spark_balancing_unique_centered_solution():
    m = 6
    nodes = torch.linspace(-1, 1, m, dtype=DTYPE)
    mu = torch.stack((torch.ones_like(nodes), nodes), dim=1)
    md = torch.stack((torch.ones_like(nodes), nodes.pow(3)), dim=1)
    mu *= torch.linspace(-1, 1, m, dtype=DTYPE).exp()[:, None]
    md *= torch.linspace(0.7, -0.7, m, dtype=DTYPE).exp()[:, None]
    # Explicitly verify every maximal minor, rather than equating full rank with full spark.
    for a in (mu, md):
        for rows in itertools.combinations(range(m), 2):
            assert torch.linalg.det(a[list(rows)]).abs() > 1e-4
    x0 = random_tensors(150, (m,))[0]
    solutions = [balance_shared_metric(mu, md, steps=1500, lr=0.5, x0=start)
                 for start in (torch.zeros_like(x0), x0, x0 + 9)]
    worst = 0.0
    for x, pu, pd in solutions:
        error = (paired_leverage(pu, pd) - 4.0 / m).abs().max().item()
        worst = max(worst, error)
        assert error < 2e-10
        torch.testing.assert_close(x, solutions[0][0], rtol=2e-9, atol=2e-9)
        for p in (pu, pd):
            torch.testing.assert_close(p.T @ p, torch.eye(2, dtype=DTYPE), rtol=2e-12, atol=2e-12)
    print(f"full spark float64 6x2: maximum paired residual={worst:.3e}")


def test_counterexample_full_rank_does_not_imply_balancing_exists():
    a = torch.tensor([[1., 0.], [0., 1.], [0., 0.]], dtype=DTYPE)
    assert torch.linalg.matrix_rank(a) == 2
    x, pu, pd = balance_shared_metric(a, a, steps=100, lr=0.2)
    residual = (paired_leverage(pu, pd) - 4.0 / 3).abs().max().item()
    assert residual == pytest.approx(4.0 / 3)
    assert x.norm() > 20
    direction = torch.tensor([-0.5, -0.5, 1.], dtype=DTYPE)
    torch.testing.assert_close(phi(10 * direction, a, a), torch.tensor(-20., dtype=DTYPE))
    print(f"infeasible float64 3x2: paired residual={residual:.6f}, ||x||={x.norm().item():.3f}")


def test_counterexample_boundary_target_only_attained_at_infinity():
    mu = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [0., 1.]], dtype=DTYPE)
    md = torch.tensor([[1., 0.], [1., 0.], [-0.5, 3**0.5 / 2], [-0.5, -3**0.5 / 2]], dtype=DTYPE)
    errors = []
    for t in (0., 12., 24.):
        x = torch.tensor([-t, 0., 0., 0.], dtype=DTYPE)
        pu, pd = polars_at(x, mu, md)
        ell = paired_leverage(pu, pd)
        # Mu has a mandatory first row in every basis; Md contributes > 0 there.
        assert ell[0] > 1
        errors.append((ell - 1).abs().max().item())
        assert torch.isfinite(phi(x, mu, md))
    assert errors[0] > 0.1 and errors[-1] < 1e-9
    print(f"boundary float64 4x2: paired residuals at t=0,12,24: {errors}")


def test_counterexample_full_rank_can_have_extra_flat_balancing_directions():
    a = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], dtype=DTYPE)
    z = torch.zeros(4, dtype=DTYPE)
    v = torch.tensor([3., 3., -3., -3.], dtype=DTYPE)
    for x in (z, v):
        pu, pd = polars_at(x, a, a)
        torch.testing.assert_close(paired_leverage(pu, pd), torch.ones(4, dtype=DTYPE), rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(phi(z, a, a), phi(v, a, a), rtol=2e-12, atol=2e-12)
    hessian = torch.autograd.functional.hessian(lambda x: phi(x, a, a), z)
    assert (hessian @ v).norm() < 2e-12


def test_counterexample_convexity_does_not_guarantee_arbitrary_solver_step_convergence():
    a = torch.ones(2, 1, dtype=DTYPE)  # Full spark: every one-row minor is nonzero.
    x0 = torch.tensor([2., -2.], dtype=DTYPE)
    _, pu, pd = balance_shared_metric(a, a, x0=x0, steps=100, lr=4.)
    residual = (paired_leverage(pu, pd) - 1).abs().max().item()
    assert residual > 0.9
    _, pu, pd = balance_shared_metric(a, a, x0=x0, steps=100, lr=0.5)
    assert (paired_leverage(pu, pd) - 1).abs().max() < 2e-12
    print(f"large balancing step float64 2x1 lr=4: residual after 100 steps={residual:.3e}")


def test_counterexample_rank_deficient_polar_is_nonunique_and_discontinuous():
    a = torch.tensor([[1., 0.], [0., 0.], [0., 0.]], dtype=DTYPE)
    p1 = torch.tensor([[1., 0.], [0., 1.], [0., 0.]], dtype=DTYPE)
    p2 = torch.tensor([[1., 0.], [0., -1.], [0., 0.]], dtype=DTYPE)
    for p in (p1, p2, exact_polar(a)):
        torch.testing.assert_close(p.T @ p, torch.eye(2, dtype=DTYPE))
        assert (a * p).sum() == 1
    epsilon = 1e-12
    plus, minus = a.clone(), a.clone()
    plus[1, 1], minus[1, 1] = epsilon, -epsilon
    input_gap = (plus - minus).norm().item()
    output_gap = (exact_polar(plus) - exact_polar(minus)).norm().item()
    assert input_gap < 3e-12 and output_gap > 1.9
    assert torch.isneginf(phi(torch.zeros(3, dtype=DTYPE), a, a))
    print(f"rank boundary float64 3x2: input gap={input_gap:.3e}, polar gap={output_gap:.3e}")


def test_counterexample_zero_gradient_produces_nonzero_optimizer_step():
    up, down = random_tensors(160, (5, 2), (2, 5))
    up, down = torch.nn.Parameter(up), torch.nn.Parameter(down)
    before = up.detach().clone()
    h = gauge_metric(up, down)
    up.grad, down.grad = torch.zeros_like(up), torch.zeros_like(down)
    opt = QNorMuon([(up, down)], lr=0.01)
    opt.step()
    canonical_step = (before - up) / h.sqrt()[:, None]
    assert canonical_step.norm().item() == pytest.approx(0.01 * 2**0.5, abs=2e-14)
    # The returned completion has n columns even for zero input (rank zero).
    p = exact_polar(torch.zeros(5, 2, dtype=DTYPE))
    assert p.norm().item() == pytest.approx(2**0.5)
    print(f"zero gradient float64 5x2: canonical step norm={canonical_step.norm().item():.6f}")


@pytest.mark.parametrize("tiny", [0., 1e-16])
def test_counterexample_clamping_breaks_balance_and_gauge_covariance(tiny):
    up = torch.tensor([[tiny], [1.]], dtype=DTYPE)
    down = torch.ones(1, 2, dtype=DTYPE)
    c = torch.tensor([1e4, 1.], dtype=DTYPE)
    h = gauge_metric(up, down)
    h2 = gauge_metric(up * c[:, None], down / c)
    assert not torch.allclose(h2 / c.square(), h, rtol=1e-3, atol=0)
    uc, dc = canonical_pair(up, down)
    uc2, dc2 = canonical_pair(up * c[:, None], down / c)
    assert relative_error(dc2, dc) > 1e-8
    imbalance = (uc.norm(dim=1) - dc.norm(dim=0)).abs().max().item()
    assert imbalance > 9e-7
    ui, di = up.clone(), down.clone()
    canonicalize_pair_(ui, di)
    torch.testing.assert_close(ui, uc)
    torch.testing.assert_close(di, dc)
    print(f"clamp float64 2x1 tiny={tiny}: balance error={imbalance:.3e}, "
          f"first metric covariance ratio={(h2 / (c.square() * h))[0].item():.3e}")


def test_both_zero_rows_have_no_unique_gauge_transformation():
    up = torch.tensor([[0.], [1.]], dtype=DTYPE)
    down = torch.tensor([[0., 1.]], dtype=DTYPE)
    uc, dc = canonical_pair(up, down)
    assert torch.isfinite(uc).all() and torch.isfinite(dc).all()
    torch.testing.assert_close(uc, up)
    torch.testing.assert_close(dc, down)
    # Every positive c fixes the zero pair; H's covariance is undefined in theory.
    c = torch.tensor([10., 1.], dtype=DTYPE)
    torch.testing.assert_close(up * c[:, None], up)
    torch.testing.assert_close(down / c, down)
    assert gauge_metric(up, down)[0] == 1


@pytest.mark.parametrize("dtype,magnitude", [(torch.float64, 1e200), (torch.float32, 1e20)])
def test_counterexample_finite_weights_can_overflow_norm_computation(dtype, magnitude):
    up = torch.full((2, 1), magnitude, dtype=dtype)
    down = torch.ones((1, 2), dtype=dtype)
    assert torch.isfinite(up).all()
    # Use two columns so the backend computes a sum of squares, not abs(x).
    up = up.expand(2, 2).clone()
    down = down.expand(2, 2).clone()
    uc, dc = canonical_pair(up, down)
    assert not (torch.isfinite(uc).all() and torch.isfinite(dc).all())


def test_counterexample_raw_l2_loss_does_not_have_covariant_gradients():
    up, down = random_tensors(165, (6, 2), (2, 6))
    c = torch.logspace(-2, 2, 6, dtype=DTYPE)
    # Euclidean gradients of (||U||^2 + ||down||^2)/2 equal the parameters.
    ordinary = canonical_gradients(up, down, up, down)
    gauged = canonical_gradients(up * c[:, None], down / c, up * c[:, None], down / c)
    assert relative_error(gauged[0], ordinary[0]) > 1


def test_counterexample_column_shard_local_norms_are_not_global_metric():
    up = torch.tensor([[3., 4.], [5., 12.], [8., 15.]], dtype=DTYPE)
    dt = torch.tensor([[4., 3.], [12., 5.], [15., 8.]], dtype=DTYPE)
    global_metric = gauge_metric(up, dt.T)
    torch.testing.assert_close(global_metric, torch.ones(3, dtype=DTYPE))
    for column in range(2):
        local_metric = gauge_metric(up[:, column:column+1], dt[:, column:column+1].T)
        assert relative_error(local_metric, global_metric) > 0.1


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 2e-11), (torch.float32, 2e-5)])
def test_large_dynamic_range_gauge_and_reference_precision(dtype, tolerance):
    values = random_tensors(170, (24, 8), (8, 24), (24, 8), (8, 24), (24,))
    reference = quotient_polar_update(*values[:4], x=values[4])
    up, down, gu, gd, x = [value.to(dtype) for value in values]
    c = torch.logspace(-6, 6, 24, dtype=dtype)
    ordinary = quotient_polar_update(up, down, gu, gd, x=x)
    gauged = quotient_polar_update(up * c[:, None], down / c, gu / c[:, None], gd * c, x=x)
    errors = [relative_error(gauged[0] / c[:, None], ordinary[0]),
              relative_error(gauged[1] * c, ordinary[1])]
    precision = max(relative_error(ordinary[i].double(), reference[i]) for i in range(4))
    defect = max((p.T @ p - torch.eye(8, dtype=dtype)).norm().item() for p in ordinary[2:4])
    assert max(errors) < tolerance and precision < tolerance and defect < tolerance
    print(f"precision {dtype} 24x8 c=1e-6..1e6: gauge={max(errors):.3e}, "
          f"vs float64={precision:.3e}, Stiefel={defect:.3e}")


def test_bfloat16_cpu_support_and_explicit_promoted_input_measurement():
    values = random_tensors(180, (24, 8), (8, 24), (24, 8), (8, 24), (24,))
    bf = [a.to(torch.bfloat16) for a in values]
    # No promotion exists in production. Check capability without pretending this
    # diagnostic float32 promotion is an implemented mixed-precision optimizer.
    try:
        native = quotient_polar_update(*bf[:4], x=bf[4])
    except (RuntimeError, NotImplementedError) as error:
        message = str(error).lower()
        assert "bfloat16" in message and ("not implemented" in message or "not support" in message)
        print(f"native bfloat16 CPU 24x8: unsupported ({error})")
    else:
        assert all(torch.isfinite(t).all() for t in native)
        print("native bfloat16 CPU 24x8: supported on this backend")
    reference = quotient_polar_update(*values[:4], x=values[4])
    promoted = [a.float() for a in bf]
    result = quotient_polar_update(*promoted[:4], x=promoted[4])
    error = max(relative_error(result[i].double(), reference[i]) for i in range(4))
    assert error < 0.02
    c = torch.logspace(-6, 6, 24, dtype=DTYPE)
    gauged_values = [values[0] * c[:, None], values[1] / c,
                     values[2] / c[:, None], values[3] * c, values[4]]
    promoted_gauged = [a.to(torch.bfloat16).float() for a in gauged_values]
    gauged_result = quotient_polar_update(*promoted_gauged[:4], x=promoted_gauged[4])
    gauge_error = max(relative_error(gauged_result[0].double() / c[:, None], result[0].double()),
                      relative_error(gauged_result[1].double() * c, result[1].double()))
    assert gauge_error < 0.03
    print(f"bfloat16 quantization + explicit float32 solve 24x8: "
          f"vs float64={error:.3e}, gauge={gauge_error:.3e}")


def test_nearly_rank_deficient_float32_sensitivity():
    left, right = random_tensors(190, (8, 3), (3, 3))
    q, _ = torch.linalg.qr(left)
    v, _ = torch.linalg.qr(right)
    a = (q * torch.tensor([1., 1e-4, 1e-8], dtype=DTYPE)) @ v.T
    p64, p32 = exact_polar(a), exact_polar(a.float()).double()
    error = relative_error(p32, p64)
    defect = (p32.T @ p32 - torch.eye(3, dtype=DTYPE)).norm().item()
    assert torch.isfinite(p32).all() and defect < 2e-5
    # A small Stiefel defect alone cannot certify accurate directions.
    assert error > 1e-3
    print(f"near rank deficiency 8x3 singular values=1,1e-4,1e-8: "
          f"float32 vs float64={error:.3e}, float32 Stiefel={defect:.3e}")


def test_fixed_approximate_polar_preserves_gauge_but_not_exact_stiefel(monkeypatch):
    values = random_tensors(200, (12, 4), (4, 12), (12, 4), (4, 12), (12,))
    reference = quotient_polar_update(*values[:4], x=values[4])

    def short_newton_schulz(a):
        p = a / a.norm()
        for _ in range(2):
            p = 1.5 * p - 0.5 * p @ (p.T @ p)
        return p

    # Test-only substitution: the exact production backend is unchanged.
    monkeypatch.setattr(core, "exact_polar", short_newton_schulz)
    up, down, gu, gd, x = values
    c = torch.logspace(-6, 6, 12, dtype=DTYPE)
    ordinary = quotient_polar_update(up, down, gu, gd, x=x)
    gauged = quotient_polar_update(up * c[:, None], down / c, gu / c[:, None], gd * c, x=x)
    gauge_error = max(relative_error(gauged[0] / c[:, None], ordinary[0]),
                      relative_error(gauged[1] * c, ordinary[1]))
    assert gauge_error < 2e-11
    defect = (ordinary[2].T @ ordinary[2] - torch.eye(4, dtype=DTYPE)).norm().item()
    mismatch = relative_error(ordinary[2], reference[2])
    assert defect > 0.1 and mismatch > 0.01
    print(f"two NS iterations float64 12x4: gauge={gauge_error:.3e}, "
          f"Stiefel={defect:.3e}, vs SVD={mismatch:.3e}")
