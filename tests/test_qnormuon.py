import math

import torch

from qnormuon import (
    balance_shared_metric,
    canonical_pair,
    exact_polar,
    gauge_metric,
    paired_leverage,
    quotient_polar_update,
)


torch.set_default_dtype(torch.float64)


def rand_pair(m=24, n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    up = torch.randn(m, n, generator=g)
    down = torch.randn(n, m, generator=g)
    return up, down


def apply_gauge(up, down, log10_span=3.0, seed=1):
    g = torch.Generator().manual_seed(seed)
    z = (2 * torch.rand(up.shape[0], generator=g) - 1) * log10_span
    c = 10.0 ** z
    up2 = c[:, None] * up
    down2 = down * (1.0 / c)[None, :]
    return up2, down2, c


def test_canonical_pair_is_gauge_invariant_and_balanced():
    up, down = rand_pair()
    up2, down2, _ = apply_gauge(up, down, log10_span=4.0)

    u1, d1 = canonical_pair(up, down)
    u2, d2 = canonical_pair(up2, down2)

    torch.testing.assert_close(u1, u2, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(d1, d2, rtol=1e-10, atol=1e-10)

    ru = torch.linalg.vector_norm(u1, dim=1)
    rd = torch.linalg.vector_norm(d1.T, dim=1)
    torch.testing.assert_close(ru, rd, rtol=1e-10, atol=1e-10)


def test_quotient_polar_update_is_exactly_gauge_equivariant():
    up, down = rand_pair(seed=10)
    g = torch.Generator().manual_seed(11)
    gu = torch.randn(up.shape, generator=g)
    gd = torch.randn(down.shape, generator=g)

    # Shared intrinsic state: identical across gauge representatives.
    x = torch.randn(up.shape[0], generator=g) * 0.7

    du, dd, _, _, _ = quotient_polar_update(up, down, gu, gd, x=x)

    up2, down2, c = apply_gauge(up, down, log10_span=3.0, seed=12)
    # Euclidean gradients transform contravariantly.
    gu2 = (1.0 / c)[:, None] * gu
    gd2 = gd * c[None, :]

    du2, dd2, _, _, _ = quotient_polar_update(up2, down2, gu2, gd2, x=x)

    torch.testing.assert_close(du2, c[:, None] * du, rtol=2e-9, atol=2e-9)
    torch.testing.assert_close(dd2, dd * (1.0 / c)[None, :], rtol=2e-9, atol=2e-9)


def test_generalized_condition_number_is_one():
    up, down = rand_pair(m=30, n=7, seed=20)
    g = torch.Generator().manual_seed(21)
    gu = torch.randn(up.shape, generator=g)
    gd = torch.randn(down.shape, generator=g)
    x = torch.randn(up.shape[0], generator=g)

    du, dd, _, _, h = quotient_polar_update(up, down, gu, gd, x=x)
    Hinv = torch.diag(1.0 / h)
    H = torch.diag(h)
    I = torch.eye(up.shape[1])

    lhs_u = du.T @ Hinv @ du
    dd_t = dd.T
    lhs_d = dd_t.T @ H @ dd_t

    torch.testing.assert_close(lhs_u, I, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(lhs_d, I, rtol=1e-10, atol=1e-10)


def test_shared_metric_balances_paired_leverage_without_breaking_polarity():
    m, n = 24, 8
    g = torch.Generator().manual_seed(30)
    mu = torch.randn(m, n, generator=g)
    md = torch.randn(m, n, generator=g)

    x, pu, pd = balance_shared_metric(mu, md, steps=400, lr=0.2)
    ell = paired_leverage(pu, pd)
    target = 2.0 * n / m

    assert (ell - target).abs().max().item() < 1e-9
    assert abs(x.mean().item()) < 1e-12

    I = torch.eye(n)
    torch.testing.assert_close(pu.T @ pu, I, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(pd.T @ pd, I, rtol=1e-10, atol=1e-10)


def test_gradient_identity_for_gauge_direction_on_bilinear_surrogate():
    # Simplified branch y = down @ (a * (up @ x)), where a is a fixed gate.
    # The exact gauge symmetry implies <G_u,U> - <G_D,D> = 0.
    m, n = 12, 5
    up, down = rand_pair(m=m, n=n, seed=40)
    up = up.clone().requires_grad_(True)
    down = down.clone().requires_grad_(True)

    g = torch.Generator().manual_seed(41)
    x = torch.randn(n, generator=g)
    gate = torch.randn(m, generator=g)
    target = torch.randn(n, generator=g)

    y = down @ (gate * (up @ x))
    loss = 0.5 * (y - target).square().sum()
    loss.backward()

    G_u = up.grad
    G_D = down.grad.T
    D = down.detach().T

    identity = (G_u * up.detach()).sum() - (G_D * D).sum()
    assert abs(identity.item()) < 1e-9
