"""Cached smooth spectra must reproduce the independent original-residual certificate."""
import pytest
import torch

from experiments.horizontal_spectral import random_example
from qnormuon import QuotientSpectralOptimizer as QSO, SolverConfig, SwiGLUPair
from qnormuon.coupled_solver import (
    CachedDualSpectrum, Counts, SmoothDual, adjoint, certificate, multipliers,
    solve_coupled,
)


@pytest.mark.parametrize('shape', [(7, 3), (12, 5), (32, 8)])
@pytest.mark.parametrize('scale', [1e-8, 1., 1e8])
def test_cached_dual_spectrum_matches_independent_fp64_certificate(shape, scale):
    m, n = shape
    generator = torch.Generator().manual_seed(1000 + m + n)
    u = torch.randn(m, n, dtype=torch.float64, generator=generator)
    d = torch.randn(m, n, dtype=torch.float64, generator=generator)
    d = d / d.norm(dim=1, keepdim=True) * u.norm(dim=1, keepdim=True)
    a = scale * torch.randn(2, m, n, dtype=torch.float64, generator=generator)
    beta = multipliers(u, d, a)
    centered = a - adjoint(u, d, beta)
    magnitude = float(centered.norm())
    coord = (u.square() + d.square()).sum(1).rsqrt()
    z = torch.randn(m, dtype=torch.float64, generator=generator) * .1
    problem = SmoothDual(u * coord[:, None], d * coord[:, None], centered / magnitude)
    ev = problem.evaluate(z)
    lam = beta + magnitude * coord * z
    cached = CachedDualSpectrum(lam, ev.pair, ev.singular, magnitude)
    p_cached, fast = certificate(u, d, a, lam, ev.pair, Counts(), cached_dual=cached)
    p_independent, slow = certificate(u, d, a, lam, ev.pair, Counts())
    assert torch.equal(p_cached, p_independent)
    assert fast['dual_spectrum_source'] == 'cached_smooth_residual'
    assert slow['dual_spectrum_source'] == 'independent_original_residual'
    relative = abs(fast['dual_objective']-slow['dual_objective'])/slow['dual_objective']
    assert relative < 1e-10  # over five orders below the production 3e-5 target
    assert abs(fast['residual_rcond']-slow['residual_rcond']) < 1e-10
    assert abs(fast['normalized_gap']-slow['normalized_gap']) < 1e-10
    with pytest.raises(ValueError, match='same multiplier'):
        certificate(u, d, a, lam.clone(), ev.pair, Counts(), cached_dual=cached)


def test_cached_spectrum_stays_accurate_near_smooth_rcond_guard():
    generator = torch.Generator().manual_seed(412)
    u = torch.randn(8, 3, dtype=torch.float64, generator=generator)
    d = u.clone()
    q, _ = torch.linalg.qr(torch.randn(8, 3, dtype=torch.float64, generator=generator))
    b = q @ torch.diag(torch.tensor([1., .02, 1.2e-4], dtype=torch.float64))
    a = torch.stack((b, b))
    beta = multipliers(u, d, a)
    centered = a - adjoint(u, d, beta)
    magnitude = float(centered.norm())
    coord = (u.square() + d.square()).sum(1).rsqrt()
    z = torch.zeros(8, dtype=torch.float64)
    ev = SmoothDual(u * coord[:, None], d * coord[:, None], centered / magnitude).evaluate(z)
    lam = beta + magnitude * coord * z
    _, fast = certificate(u, d, a, lam, ev.pair, Counts(),
                          cached_dual=CachedDualSpectrum(lam, ev.pair, ev.singular, magnitude))
    _, slow = certificate(u, d, a, lam, ev.pair, Counts())
    assert 1e-4 < slow['residual_rcond'] < 2e-4
    assert abs(fast['dual_objective']-slow['dual_objective'])/slow['dual_objective'] < 1e-10
    assert abs(fast['residual_rcond']-slow['residual_rcond']) < 1e-10


def test_debug_recomputation_and_cached_warm_solve_match():
    u, d, a = random_example(902)
    fast = solve_coupled(u, d, a, config=SolverConfig(fallback=False))
    slow = solve_coupled(u, d, a, config=SolverConfig(fallback=False,
                                                     independent_certificate=True))
    assert fast.converged and slow.converged
    assert fast.metrics['dual_spectrum_source'] == 'cached_smooth_residual'
    assert slow.metrics['dual_spectrum_source'] == 'independent_original_residual'
    torch.testing.assert_close(fast.lam, slow.lam, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(fast.pair, slow.pair, rtol=1e-11, atol=1e-11)
    for key in ('primal_objective', 'dual_objective', 'normalized_gap', 'residual_rcond'):
        assert abs(fast.metrics[key]-slow.metrics[key]) < 1e-10
    assert slow.counts.svd_matrices-fast.counts.svd_matrices == 2*(fast.iterations+1)


def test_cached_and_debug_checkpoint_continuations_match():
    u, d, a = random_example(902)
    pairs = []
    optimizers = []
    for independent in (False, True):
        pair = SwiGLUPair('p', torch.nn.Parameter(u.clone()), torch.nn.Parameter(d.T.clone()))
        pair.up.grad = a[0].clone()
        pair.down.grad = a[1].T.clone()
        pairs.append(pair)
        optimizers.append(QSO([pair], lr=.01, solver=SolverConfig(independent_certificate=independent)))
    for _ in range(2):
        for optimizer in optimizers:
            optimizer.step()
        torch.testing.assert_close(pairs[0].up, pairs[1].up, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(pairs[0].down, pairs[1].down, rtol=1e-11, atol=1e-11)
        for key in ('momentum_up', 'momentum_down_t', 'lambda'):
            torch.testing.assert_close(optimizers[0].state[pairs[0].up][key],
                                       optimizers[1].state[pairs[1].up][key], rtol=1e-11, atol=1e-11)
