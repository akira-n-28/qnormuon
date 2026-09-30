"""Independent SVD oracles for the research-only QDWH posterior."""
import math

import pytest
import torch

from experiments.smooth_polar_alternative import diagnose, polar_derivative
from experiments.smooth_qdwh_decisions import (
    adaptive_decompose, armijo_interval, certificate_interval_decision,
    cg_stop_certified, derivative_error_bound, factor_posterior,
    interval_sign, row_operator_upper,
)
from qnormuon.coupled_solver import SmoothDual


def prescribed(ratio, *, repeated=False, scale=1.):
    gen=torch.Generator().manual_seed(451)
    u,_=torch.linalg.qr(torch.randn(32,8,dtype=torch.float64,generator=gen))
    v,_=torch.linalg.qr(torch.randn(8,8,dtype=torch.float64,generator=gen))
    s=torch.linspace(1.,.2,8,dtype=torch.float64)
    if repeated:
        s[:2]=1.;s[-2:]=ratio
    else:
        s[-1]=ratio
    return scale*(u*s)@v.T


@pytest.mark.parametrize("ratio",[1.,1e-2,3e-4,1.01e-4])
@pytest.mark.parametrize("repeated",[False,True])
def test_factor_bounds_contain_independent_svd(ratio,repeated):
    b=prescribed(ratio,repeated=repeated)
    qdwh=diagnose(b)
    assert qdwh.accepted,qdwh.reason
    post=factor_posterior(b,qdwh)
    assert post.accepted,post.reason
    left,s,right=torch.linalg.svd(b,full_matrices=False)
    polar=left@right
    h=right.T@torch.diag(s)@right
    assert float((polar-qdwh.polar).norm())<=post.polar_error+1e-12
    assert float((h-qdwh.h).norm())<=post.h_error+1e-12
    assert post.sigma_min_lower<=float(s[-1])+1e-12
    assert float(s[0])<=post.sigma_max_upper+1e-12
    assert post.rcond_lower<=float(s[-1]/s[0])+1e-12
    assert float(s[-1]/s[0])<=post.rcond_upper+1e-12
    assert post.nuclear_lower<=float(s.sum())+1e-12
    assert float(s.sum())<=post.nuclear_upper+1e-12


@pytest.mark.parametrize("ratio",[1e-2,1.01e-4])
def test_derivative_bound_contains_independent_svd(ratio):
    b=prescribed(ratio,repeated=True)
    result=diagnose(b)
    post=factor_posterior(b,result)
    assert post.accepted
    gen=torch.Generator().manual_seed(987)
    left,s,right=torch.linalg.svd(b,full_matrices=False)
    for e in (torch.randn(b.shape,dtype=b.dtype,generator=gen),
              left[:,-1,None]@right[-1,None,:]):
        yc=polar_derivative(result,e)
        oracle=SmoothDual(None,None,None).polar_derivative(
            type("E",(),dict(left=left[None],singular=s[None],right=right[None])),e[None])[0]
        radius,_=derivative_error_bound(b,e,yc,result,post)
        assert math.isfinite(radius)
        assert float((yc-oracle).norm())<=radius+1e-12


def test_guard_and_explicit_decomposition_fallback():
    b=prescribed(.99e-4)
    result=diagnose(b)
    post=factor_posterior(b,result)
    assert not post.accepted
    backend,reason,_,_=adaptive_decompose(b,lambda *_:(True,"certified"))
    assert backend=="full_svd_decomposition_fallback"
    assert reason.startswith("qdwh_")
    good=prescribed(1e-2)
    backend,reason,_,_=adaptive_decompose(good,lambda *_:(False,"qdwh_cg_uncertain"))
    assert backend=="full_svd_decomposition_fallback"
    assert reason=="qdwh_cg_uncertain"


def test_decision_intervals_do_not_guess_at_boundaries():
    assert interval_sign(0.,1e-9)=="uncertain"
    assert interval_sign(2.,1.)=="positive"
    assert interval_sign(-2.,1.)=="negative"
    assert armijo_interval((1.,1.),(1.,1.),(-1.,-1.),alpha=0.,rounding=(0.,0.))=="accept"
    assert armijo_interval((1.,1.),(1.,1.),(-1.,-1.),alpha=1.,rounding=(0.,0.))=="reject"
    assert armijo_interval((1.,1.1),(1.,1.1),(-1.,1.),alpha=1.,rounding=(0.,0.))=="qdwh_armijo_ambiguous"
    assert not cg_stop_certified(.0101,.1,damping=1e-6)
    assert cg_stop_certified(.001,.1,damping=1e-6)
    assert not cg_stop_certified(0.,.1,damping=0.)
    assert certificate_interval_decision((1.00002,1.00002),(1.,1.),
        rcond_lower=2e-4,feasible=True)=="accept"
    assert certificate_interval_decision((1.0000300011,1.0000300011),(1.,1.),
        rcond_lower=2e-4,feasible=True)=="qdwh_certificate_ambiguous"
    assert certificate_interval_decision((1.,1.),(1.,1.),
        rcond_lower=1e-4,feasible=True)=="qdwh_certificate_ambiguous"
    assert certificate_interval_decision((1.,1.),(-1.,1.),
        rcond_lower=2e-4,feasible=True)=="qdwh_certificate_ambiguous"


def test_rounded_whitening_operator_bound():
    u=torch.full((5,3),.5,dtype=torch.float64)
    d=u.clone()
    upper=row_operator_upper(u,d)
    assert upper>=math.sqrt(1.5)
