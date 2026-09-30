"""Small analytic control for the real-pair precision diagnosis.

This injects anisotropic polar error; it does not assert a vendor-specific SVD
error on every device. The actual H100 reproducer is the retained six-pair corpus
and cluster/training_solver_diagnosis.py.
"""
import torch
from qnormuon.coupled_solver import Counts, accepted, certificate


def test_polar_error_can_fail_gap_at_exact_multiplier_without_rank_loss():
    u = d = torch.eye(2, dtype=torch.float64)
    a = torch.stack((u, d))
    lam = torch.zeros(2, dtype=torch.float64)
    # Exact optimum: P_U=P_D=I, lambda=0, residual rcond=1.
    # An anisotropic error of the measured order is horizontal but not feasible.
    noisy = a.float().clone()
    noisy[:, 0, 0] += 1e-4
    p, metrics = certificate(u, d, a, lam, noisy, Counts())
    delta = float(noisy[0, 0, 0]) - 1
    expected_gap = delta / (2 * (1 + delta))
    assert abs(metrics['normalized_gap'] - expected_gap) < 1e-14
    assert metrics['residual_rcond'] == 1
    assert metrics['horizontal_residual'] == metrics['spectral_excess'] == 0
    assert not accepted(metrics, 3e-5)
    assert abs(float(p.abs().max()) - 1) <= 2 * torch.finfo(torch.float64).eps
    # Recompute the polar accurately at the SAME lambda, then store in fp32.
    q, _, vt = torch.linalg.svd(a, full_matrices=False)
    _, repaired = certificate(u, d, a, lam, (q @ vt).float(), Counts())
    assert accepted(repaired, 1e-8)
    assert repaired['normalized_gap'] == 0
