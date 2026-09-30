"""Conservative Gram radial norm, independently checked against full fp64 SVD."""
import pytest
import torch

from experiments.horizontal_spectral import random_example
from qnormuon import QuotientSpectralOptimizer as QSO, SolverConfig, SwiGLUPair
from qnormuon.coupled_solver import (
    CachedDualSpectrum, Counts, SmoothDual, accepted, adjoint, certificate,
    multipliers, primal_top_singular_upper, solve_coupled,
)


def matrix_with_spectrum(m, spectrum, seed=90):
    n = len(spectrum)
    generator = torch.Generator().manual_seed(seed)
    left, _ = torch.linalg.qr(torch.randn(m, n, dtype=torch.float64, generator=generator))
    right, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, generator=generator))
    return (left * torch.tensor(spectrum, dtype=torch.float64)) @ right.T


@pytest.mark.parametrize("m,n", [(8, 3), (24, 7), (64, 16)])
@pytest.mark.parametrize("kind", ["gaussian", "rank1", "clustered", "repeated", "guard", "below_guard"])
def test_gram_upper_prescribed_spectra(m, n, kind):
    if kind == "gaussian":
        p = torch.randn(m, n, dtype=torch.float64, generator=torch.Generator().manual_seed(m+n))
    else:
        spectrum = torch.linspace(1., .2, n).tolist()
        if kind == "rank1":
            spectrum[1:] = [1e-9]*(n-1)
        elif kind == "clustered":
            spectrum[1] = 1-1e-12
        elif kind == "repeated":
            spectrum[:2] = [1., 1.]
        elif kind == "guard":
            spectrum[-1] = 1e-4
        else:
            spectrum[-1] = 1e-8
        p = matrix_with_spectrum(m, spectrum)
    pair = torch.stack((p, .7*p))
    oracle = torch.linalg.svdvals(pair)[:, 0]
    upper = primal_top_singular_upper(pair)
    assert bool((upper >= oracle - 64*torch.finfo(torch.float64).eps*oracle).all())
    assert float(((upper-oracle).abs()/oracle).max()) < 1e-9
    assert torch.equal(upper, primal_top_singular_upper(pair))


@pytest.mark.parametrize("scale", [1e-160, 1e-60, 1., 1e60, 1e160])
@pytest.mark.parametrize("offset", [-1e-8, -1e-12, 0., 1e-12, 1e-8])
def test_gram_upper_near_unit_and_extreme_scale(scale, offset):
    p = matrix_with_spectrum(12, [1.] * 5) * ((1+offset)*scale)
    oracle = torch.linalg.svdvals(p)[0]
    upper = primal_top_singular_upper(p.unsqueeze(0))[0]
    assert upper >= oracle - 64*torch.finfo(torch.float64).eps*oracle
    assert abs(float(upper/oracle)-1) < 1e-9
    if scale == 1.:
        actual_after = float(torch.linalg.svdvals(p/max(1.,float(upper)))[0])
        assert actual_after <= 1+1e-12


def test_gram_and_svd_certificates_agree_on_smooth_pairs():
    for seed in (13, 75, 902):
        u, d, a = random_example(seed)
        beta = multipliers(u, d, a)
        centered = a-adjoint(u, d, beta)
        magnitude = float(centered.norm())
        coordinate = (u.square()+d.square()).sum(1).rsqrt()
        z = torch.zeros(u.shape[0], dtype=torch.float64)
        ev = SmoothDual(u*coordinate[:, None], d*coordinate[:, None], centered/magnitude).evaluate(z)
        lam = beta+magnitude*coordinate*z
        cached = CachedDualSpectrum(lam, ev.pair, ev.singular, magnitude)
        old_pair, old = certificate(u,d,a,lam,ev.pair,Counts(),cached_dual=cached,
                                    primal_norm_backend="svd")
        new_pair, new = certificate(u,d,a,lam,ev.pair,Counts(),cached_dual=cached,
                                    primal_norm_backend="gram_upper")
        assert new["primal_spectral_backend"] == "gram_eigh_upper"
        assert old["primal_spectral_backend"] == "full_svd_reference"
        assert torch.linalg.svdvals(new_pair).max() <= 1+1e-12
        torch.testing.assert_close(old_pair,new_pair,rtol=1e-9,atol=1e-9)
        for key in ("primal_objective","dual_objective","normalized_gap",
                    "horizontal_residual","residual_rcond"):
            assert abs(old[key]-new[key]) < 1e-9
        assert not (accepted(new,3e-5) and not accepted(old,3e-5))


def test_gram_cached_certificate_does_not_call_svdvals(monkeypatch):
    u,d,a=random_example(24)
    beta=multipliers(u,d,a)
    centered=a-adjoint(u,d,beta)
    magnitude=float(centered.norm())
    coordinate=(u.square()+d.square()).sum(1).rsqrt()
    z=torch.zeros(u.shape[0],dtype=torch.float64)
    ev=SmoothDual(u*coordinate[:,None],d*coordinate[:,None],centered/magnitude).evaluate(z)
    lam=beta+magnitude*coordinate*z
    cached=CachedDualSpectrum(lam,ev.pair,ev.singular,magnitude)
    def forbidden(*args,**kwargs):
        raise AssertionError("full SVDVALS called in Gram certificate")
    monkeypatch.setattr(torch.linalg,"svdvals",forbidden)
    _,metrics=certificate(u,d,a,lam,ev.pair,Counts(),cached_dual=cached,
                          primal_norm_backend="gram_upper")
    assert metrics["primal_spectral_backend"] == "gram_eigh_upper"


def test_implausible_negative_gram_top_uses_full_svd_guard(monkeypatch):
    u=torch.eye(2,dtype=torch.float64);d=u.clone();a=torch.stack((u,u))
    lam=torch.zeros(2,dtype=torch.float64)
    original=torch.linalg.eigh
    def damaged(matrix,*args,**kwargs):
        values,vectors=original(matrix,*args,**kwargs)
        values=values.clone();values[:,-1]=-1.
        return values,vectors
    monkeypatch.setattr(torch.linalg,"eigh",damaged)
    _,metrics=certificate(u,d,a,lam,a,Counts(),primal_norm_backend="gram_upper")
    assert metrics["primal_spectral_backend"] == "full_svd_guarded"


def test_warm_solve_reference_backend_and_default_agree():
    u,d,a=random_example(902)
    fast=solve_coupled(u,d,a,config=SolverConfig(fallback=False))
    reference=solve_coupled(u,d,a,config=SolverConfig(fallback=False,primal_norm_backend="svd"))
    assert fast.converged and reference.converged
    assert fast.metrics["primal_spectral_backend"] == "gram_eigh_upper"
    assert reference.metrics["primal_spectral_backend"] == "full_svd_reference"
    torch.testing.assert_close(fast.lam,reference.lam,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(fast.pair,reference.pair,rtol=1e-9,atol=1e-9)
    for old,new in ((fast,reference),):
        assert abs(old.metrics["normalized_gap"]-new.metrics["normalized_gap"]) < 1e-9
    warm=solve_coupled(u,d,a,config=SolverConfig(fallback=False),initial_lambda=fast.lam)
    warm_reference=solve_coupled(u,d,a,config=SolverConfig(fallback=False,primal_norm_backend="svd"),
                                 initial_lambda=reference.lam)
    assert warm.converged and warm_reference.converged
    torch.testing.assert_close(warm.pair,warm_reference.pair,rtol=1e-9,atol=1e-9)


def test_checkpoint_continuation_uses_named_pair_and_same_gram_backend():
    u,d,a=random_example(902)
    def make():
        pair=SwiGLUPair("stable",torch.nn.Parameter(u.clone()),torch.nn.Parameter(d.T.clone()))
        optimizer=QSO([pair],lr=.01,solver=SolverConfig())
        return pair,optimizer
    pair,opt=make()
    for _ in range(2):
        pair.up.grad=a[0].clone();pair.down.grad=a[1].T.clone();opt.step()
    state=opt.state_dict()
    saved_up=pair.up.detach().clone();saved_down=pair.down.detach().clone()
    pair.up.grad=a[0].clone();pair.down.grad=a[1].T.clone();opt.step()
    resumed,resumed_opt=make()
    resumed.up.data.copy_(saved_up);resumed.down.data.copy_(saved_down)
    resumed_opt.load_state_dict(state)
    resumed.up.grad=a[0].clone();resumed.down.grad=a[1].T.clone();resumed_opt.step()
    assert torch.equal(pair.up,resumed.up) and torch.equal(pair.down,resumed.down)
    for key in ("momentum_up","momentum_down_t","lambda"):
        assert torch.equal(opt.state[pair.up][key],resumed_opt.state[resumed.up][key])


def test_boundary_conservatism_and_config_validation():
    u=torch.eye(2,dtype=torch.float64);d=u.clone();a=torch.stack((u,u))
    lam=torch.zeros(2,dtype=torch.float64)
    for radial in (1-1e-12,1.,1+1e-12):
        for target in (3e-5-1e-9,3e-5,3e-5+1e-9):
            second=2*radial*(1-target)-radial
            candidate=torch.stack((torch.diag(torch.tensor([radial,second],dtype=torch.float64)),)*2)
            _,old=certificate(u,d,a,lam,candidate,Counts(),primal_norm_backend="svd")
            direction,new=certificate(u,d,a,lam,candidate,Counts(),primal_norm_backend="gram_upper")
            assert not (accepted(new,3e-5) and not accepted(old,3e-5))
            assert float(torch.linalg.svdvals(direction).max()) <= 1+1e-12
    with pytest.raises(ValueError,match="backend"):
        SolverConfig(primal_norm_backend="approximate")
