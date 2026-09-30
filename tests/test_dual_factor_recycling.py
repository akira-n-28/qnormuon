"""Research-only factor predictor checks; final production solver is untouched."""
import io
import math
from unittest.mock import patch

import pytest
import torch

from experiments.dual_factor_recycling import (AcceptedFactors, change_indicator,
    hessian_action, pack_factors, polar_derivative, predict, solve_and_capture,
    unpack_factors)
from experiments.horizontal_spectral import random_example
from qnormuon.coupled_solver import SolverConfig, adjoint, horizontal_residual
from qnormuon.optimizer import regular_canonicalize


def fixture():
    u, d, a = random_example(902, dtype=torch.float64)
    result, cache = solve_and_capture(u, d, a,
        config=SolverConfig(fallback=False, tolerance=3e-5))
    assert result.converged and cache is not None
    return u, d, a, result, cache


def test_capture_final_accepted_original_residual_factors():
    u, d, a, result, cache = fixture()
    actual = a - adjoint(u, d, result.lam)
    torch.testing.assert_close(cache.reconstruct(), actual, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(cache.singular, torch.linalg.svdvals(actual), rtol=1e-12, atol=1e-12)
    assert torch.equal(cache.lam, result.lam)
    assert cache.reconstruction_error < 1e-12
    # The captured factors must belong to the final certified evaluation,
    # not to an earlier iterate or a rejected line-search trial.
    assert result.reason == "certified_gap" and result.history[-1]["normalized_gap"] <= 3e-5


def test_predictor_uses_zero_current_svd_or_svdvals():
    u, d, a, _, cache = fixture()
    new_a = a + .01 * torch.flip(a, (-1,))
    with patch.object(torch.linalg, "svd", side_effect=AssertionError("current SVD called")), \
         patch.object(torch.linalg, "svdvals", side_effect=AssertionError("current SVDVALS called")):
        value = predict(cache, u, d, new_a, budget=4)
        assert torch.isfinite(value.lam).all()
        assert value.cg_iterations <= 4


def test_cached_polar_derivative_and_hessian_match_independent_differences():
    u, d, a, _, cache = fixture()
    b = cache.reconstruct()
    perturbation = torch.randn(b.shape, generator=torch.Generator().manual_seed(13), dtype=b.dtype)
    perturbation /= perturbation.norm()
    analytic = polar_derivative(cache, perturbation)
    h = 1e-5
    def polar(matrix):
        q, _, vt = torch.linalg.svd(matrix, full_matrices=False)
        return q @ vt
    finite = (polar(b + h*perturbation) - polar(b - h*perturbation)) / (2*h)
    torch.testing.assert_close(analytic, finite, rtol=1e-6, atol=1e-7)
    v = torch.randn(u.shape[0], generator=torch.Generator().manual_seed(19), dtype=u.dtype)
    analytic_h = hessian_action(cache, u, d, v)
    def gradient(lam):
        return -horizontal_residual(u, d, polar(a - adjoint(u, d, lam)))
    finite_h = (gradient(cache.lam+h*v)-gradient(cache.lam-h*v))/(2*h)
    torch.testing.assert_close(analytic_h, finite_h, rtol=1e-5, atol=1e-6)
    assert float(v @ analytic_h) >= -1e-10


def test_repeated_singular_subspace_rotation_does_not_change_prediction():
    gen = torch.Generator().manual_seed(55)
    m, n = 10, 4
    left, _ = torch.linalg.qr(torch.randn(m,n,generator=gen,dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(n,n,generator=gen,dtype=torch.float64))
    left = left.expand(2,-1,-1).clone()
    vt = right.T.expand(2,-1,-1).clone()
    sigma = torch.tensor([[3.,3.,1.,.3],[3.,3.,1.,.3]],dtype=torch.float64)
    cache = AcceptedFactors(torch.zeros(m,dtype=torch.float64),left,sigma,vt,0.)
    angle = .37
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]],dtype=torch.float64)
    rotated_left, rotated_vt = left.clone(), vt.clone()
    rotated_left[:,:,:2] = left[:,:,:2] @ rotation
    rotated_vt[:,:2,:] = rotation.T @ vt[:,:2,:]
    rotated = AcceptedFactors(cache.lam,rotated_left,sigma,rotated_vt,0.)
    u = torch.randn(m,n,generator=gen,dtype=torch.float64)
    d = torch.randn(m,n,generator=gen,dtype=torch.float64)
    a = cache.reconstruct() + .001*torch.randn(2,m,n,generator=gen,dtype=torch.float64)
    e = a-cache.reconstruct()
    torch.testing.assert_close(polar_derivative(cache,e),polar_derivative(rotated,e),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(predict(cache,u,d,a,budget=4).lam,
                               predict(rotated,u,d,a,budget=4).lam,rtol=1e-11,atol=1e-11)


def test_positive_gauge_compatible_canonical_predictor():
    u,d,a,_,cache=fixture()
    raw_up=u.clone();raw_down=d.T.clone()
    gauge=torch.exp(torch.linspace(-2.,2.,u.shape[0],dtype=u.dtype))
    uc,dc,root=regular_canonicalize(raw_up,raw_down,dtype=torch.float64)
    ug,dg,rootg=regular_canonicalize(gauge[:,None]*raw_up,raw_down/gauge[None,:],dtype=torch.float64)
    raw_gu=a[0]/root[:,None];raw_gd=(a[1]*root[:,None]).T
    ac=torch.stack((root[:,None]*raw_gu,raw_gd.T/root[:,None]))
    ag=torch.stack((rootg[:,None]*(raw_gu/gauge[:,None]),
                    (raw_gd*gauge[None,:]).T/rootg[:,None]))
    torch.testing.assert_close(uc,ug,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(dc,dg,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(ac,ag,rtol=1e-13,atol=1e-13)
    first=predict(cache,uc,dc,ac,budget=2)
    second=predict(cache,ug,dg,ag,budget=2)
    torch.testing.assert_close(first.lam,second.lam,rtol=1e-10,atol=1e-10)


def test_frobenius_validity_gate_and_checkpoint_roundtrip():
    u,d,a,_,cache=fixture()
    same=a-adjoint(u,d,cache.lam)+adjoint(u,d,cache.lam)
    _,quiet=change_indicator(cache,u,d,same)
    assert quiet["frobenius_full_rank_gate"]
    shifted=a+100*torch.flip(a,(-1,))
    _,large=change_indicator(cache,u,d,shifted)
    assert not large["frobenius_full_rank_gate"]
    rejected=predict(cache,u,d,shifted,budget=2,gate=True)
    assert rejected.reason=="frobenius_gate_rejected"
    assert torch.equal(rejected.lam,cache.lam)
    buffer=io.BytesIO();torch.save(pack_factors(cache),buffer);buffer.seek(0)
    restored=unpack_factors(torch.load(buffer,weights_only=True))
    assert torch.equal(predict(cache,u,d,a,budget=2).lam,
                       predict(restored,u,d,a,budget=2).lam)


def test_predictor_does_not_use_current_optimum_or_certificate():
    u,d,a,_,cache=fixture()
    changed=a+.01*torch.flip(a,(-1,))
    with patch("qnormuon.coupled_solver.certificate",side_effect=AssertionError("oracle used")):
        result=predict(cache,u,d,changed,budget=1)
    assert torch.isfinite(result.lam).all()
    with pytest.raises(ValueError,match="budget"):
        predict(cache,u,d,changed,budget=3)


def test_fp32_factor_cache_is_predictor_only_and_uses_fp64_arithmetic():
    u,d,a,_,cache=fixture()
    compressed=cache.predictor_copy(torch.float32)
    assert compressed.factor_bytes()*2==cache.factor_bytes()
    assert compressed.lam.dtype==torch.float64
    assert compressed.reconstruct().dtype==torch.float64
    assert compressed.polar().dtype==torch.float64
    changed=a+.01*torch.flip(a,(-1,))
    with patch.object(torch.linalg,"svd",side_effect=AssertionError("SVD called")), \
         patch.object(torch.linalg,"svdvals",side_effect=AssertionError("SVDVALS called")):
        compressed_prediction=predict(compressed,u,d,changed,budget=2)
    reference_prediction=predict(cache,u,d,changed,budget=2)
    torch.testing.assert_close(compressed_prediction.lam,reference_prediction.lam,
                               rtol=1e-5,atol=1e-6)
