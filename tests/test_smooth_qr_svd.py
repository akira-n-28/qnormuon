"""Independent direct-SVD oracles for the isolated QR-SVD research adapter."""
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import qnormuon.coupled_solver as solver
from experiments.smooth_qr_svd import decompose, research_backend


def prescribed(ratio=1e-4, scale=1., repeated=False):
    gen = torch.Generator().manual_seed(674)
    q, _ = torch.linalg.qr(torch.randn(24, 8, generator=gen, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(8, 8, generator=gen, dtype=torch.float64))
    s = torch.logspace(0, math.log10(ratio), 8, dtype=torch.float64)
    if repeated:
        s[:3] = 1.
        s[-3:] = ratio
    return ((q * s) @ v.T) * scale


@pytest.mark.parametrize("ratio", [1., .1, .01, .001, 3e-4, 1.5e-4, 1.1e-4,
                                  1.01e-4, 1e-4, .99e-4, 5e-5])
@pytest.mark.parametrize("repeated", [False, True])
def test_prescribed_spectrum_and_invariant_factors(ratio, repeated):
    b = prescribed(ratio, repeated=repeated)
    u, s, vt = decompose(b)
    ru, rs, rvt = torch.linalg.svd(b, full_matrices=False)
    assert torch.linalg.norm((u * s) @ vt - b) / b.norm() < 1e-13
    assert (u.T @ u - torch.eye(8, dtype=b.dtype)).norm() < 1e-13
    assert (vt @ vt.T - torch.eye(8, dtype=b.dtype)).norm() < 1e-13
    assert ((s - rs).abs() / rs).max() < 1e-10
    assert ((u @ vt) - (ru @ rvt)).norm() < 1e-10
    assert abs(float(s.sum() / rs.sum()) - 1.) < 1e-13
    assert abs(float(s[-1] / s[0] - rs[-1] / rs[0])) < 1e-13


@pytest.mark.parametrize("scale", [1e-150, 1e150])
def test_extreme_scale(scale):
    b = prescribed(scale=scale)
    u, s, vt = decompose(b)
    ru, rs, rvt = torch.linalg.svd(b, full_matrices=False)
    assert (((u * (s / scale)) @ vt) - b / scale).norm() < 1e-13
    assert ((s - rs).abs() / rs).max() < 1e-10
    assert (u @ vt - ru @ rvt).norm() < 1e-10


def test_existing_derivative_and_finite_difference():
    b = prescribed(1.1e-4)
    gen = torch.Generator().manual_seed(678)
    e = torch.randn(b.shape, generator=gen, dtype=b.dtype)
    problem = solver.SmoothDual(None, None, None)
    def derivative(factors):
        u, s, vt = factors
        return problem.polar_derivative(SimpleNamespace(left=u, singular=s, right=vt), e)
    direct = derivative(torch.linalg.svd(b, full_matrices=False))
    candidate = derivative(decompose(b))
    assert (candidate - direct).norm() / direct.norm() < 1e-9
    eps = 1e-9
    plus = torch.linalg.svd(b + eps * e, full_matrices=False)
    minus = torch.linalg.svd(b - eps * e, full_matrices=False)
    fd = (plus[0] @ plus[2] - minus[0] @ minus[2]) / (2 * eps)
    assert (candidate - fd).norm() / direct.norm() < 1e-5


def test_clustered_spectrum_and_informative_smallest_direction():
    b = prescribed()
    ru, rs, rvt = torch.linalg.svd(b, full_matrices=False)
    rs[:3] = torch.tensor([1., 1.-1e-12, 1.-2e-12], dtype=b.dtype)
    b = (ru * rs) @ rvt
    ru, rs, rvt = torch.linalg.svd(b, full_matrices=False)
    u, s, vt = decompose(b)
    assert (u @ vt - ru @ rvt).norm() < 1e-10
    # A normal perturbation into the smallest right-singular direction has
    # derivative of size 1/s_min, unlike u_min v_min.T (exact derivative zero).
    normal = torch.arange(24, dtype=b.dtype)
    normal = normal - ru @ (ru.T @ normal)
    normal = normal / normal.norm()
    e = torch.outer(normal, rvt[-1])
    problem = solver.SmoothDual(None, None, None)
    got = problem.polar_derivative(SimpleNamespace(left=u, singular=s, right=vt), e)
    ref = problem.polar_derivative(SimpleNamespace(left=ru, singular=rs, right=rvt), e)
    assert (got - ref).norm() / ref.norm() < 1e-9


def test_unchanged_solver_replay_and_scoped_restore():
    gen = torch.Generator().manual_seed(680)
    u = torch.randn(16, 4, generator=gen, dtype=torch.float64)
    d = torch.randn(16, 4, generator=gen, dtype=torch.float64)
    d = d * (u.norm(dim=1) / d.norm(dim=1))[:, None]
    a = torch.randn(2, 16, 4, generator=gen, dtype=torch.float64)
    config = solver.SolverConfig(fallback=False)
    parent = solver.SmoothDual
    ref = solver.solve_coupled(u, d, a, config=config)
    with research_backend():
        got = solver.solve_coupled(u, d, a, config=config)
        repeated = solver.solve_coupled(u, d, a, config=config)
    assert solver.SmoothDual is parent
    assert ref.converged and got.converged
    assert ref.iterations == got.iterations
    assert ref.counts.hvp == got.counts.hvp
    assert ref.counts.line_trials == got.counts.line_trials
    assert torch.allclose(ref.lam, got.lam, rtol=1e-10, atol=1e-12)
    assert torch.allclose(ref.pair, got.pair, rtol=1e-10, atol=1e-12)
    assert torch.equal(got.lam, repeated.lam)
    assert torch.equal(got.pair, repeated.pair)


def test_only_square_full_svd_and_no_gram():
    b = prescribed()
    original = torch.linalg.svd
    shapes = []
    def observed(x, **kwargs):
        shapes.append(x.shape)
        assert kwargs == {"full_matrices": False}
        return original(x, **kwargs)
    with patch.object(torch.linalg, "svd", observed), patch.object(
            torch.linalg, "eigh", side_effect=AssertionError("no Gram/EVD")):
        decompose(b)
    assert shapes == [(8, 8)]
    with pytest.raises(ValueError):
        decompose(b.float())
    with pytest.raises(ValueError):
        decompose(b, driver="gesvda")
