"""Independent SVD oracles for research-only Gram decision inequalities."""
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from experiments.smooth_gram import diagnose
from experiments.smooth_gram_decisions import (
    adaptive_decompose, armijo_interval, cg_residual_upper, derivative_error_bound,
    gram_derivative, interval_sign, polar_posterior,
)
from qnormuon.coupled_solver import SmoothDual


def matrix(low, *, repeated=False, seed=9):
    generator=torch.Generator().manual_seed(seed)
    m,n=24,8
    u,_=torch.linalg.qr(torch.randn(m,n,dtype=torch.float64,generator=generator))
    v,_=torch.linalg.qr(torch.randn(n,n,dtype=torch.float64,generator=generator))
    s=torch.linspace(1.,.25,n,dtype=torch.float64)
    s[-1]=low
    if repeated:
        s[:2]=1.
        s[-2:]=low
    return (u*s)@v.T


@pytest.mark.parametrize("low,repeated",[(.1,False),(1e-2,True),(1e-3,False),
                                       (3e-4,False),(1.01e-4,True)])
def test_posterior_contains_independent_polar_and_derivative(low,repeated):
    b=matrix(low,repeated=repeated)
    gram=diagnose(b)
    assert gram.accepted,gram.reason
    posterior=polar_posterior(b,gram)
    assert posterior.accepted,posterior.reason
    u,s,vt=torch.linalg.svd(b,full_matrices=False)
    p=u@vt
    h=(vt.T*s)@vt
    assert float((p-gram.polar).norm()) <= posterior.polar_error
    approx_h=(gram.right.T*gram.singular)@gram.right
    assert float((h-approx_h).norm()) <= posterior.h_error
    assert float(s[-1]) >= posterior.sigma_lower
    e=torch.randn(b.shape,dtype=torch.float64,
                  generator=torch.Generator().manual_seed(500+int(low*1e7)))
    y=gram_derivative(b,e,gram)
    oracle=SmoothDual(None,None,None).polar_derivative(
        SimpleNamespace(left=u[None],singular=s[None],right=vt[None]),e[None])[0]
    upper,_=derivative_error_bound(b,e,y,gram,posterior)
    assert math.isfinite(upper)
    assert float((y-oracle).norm()) <= upper


def test_guard_rejection_and_curvature_intervals():
    b=matrix(.99e-4)
    gram=diagnose(b)
    assert not gram.accepted
    assert polar_posterior(b,gram).reason=="gram_structural_failure"
    assert interval_sign(2.,.1)=="positive"
    assert interval_sign(-2.,.1)=="negative"
    assert interval_sign(.01,.1)=="uncertain"


def test_armijo_boundary_and_cg_residual_budget():
    assert armijo_interval((2.,2.),(1.,1.),(-1.,-1.),alpha=1.)=="accept"
    assert armijo_interval((2.,2.),(3.,3.),(-1.,-1.),alpha=1.)=="reject"
    assert armijo_interval((2.,2.),(2.-1e-4,2.+1e-4),(-1.,-1.),
                           alpha=1.)=="gram_armijo_ambiguous"
    assert cg_residual_upper(.01,.02,[.03,.04])==pytest.approx(.1)


def test_nearly_null_derivative_and_marginal_decision_intervals():
    b=matrix(1.01e-4,repeated=True)
    gram=diagnose(b)
    posterior=polar_posterior(b,gram)
    assert gram.accepted and posterior.accepted
    u,s,vt=torch.linalg.svd(b,full_matrices=False)
    p=u@vt
    h=(vt.T*s)@vt
    # Positive symmetric polar-direction perturbation is an exact derivative
    # null direction; a tiny tangent component stresses relative error.
    tangent=torch.randn(b.shape,dtype=torch.float64,
                        generator=torch.Generator().manual_seed(1009))
    e=p@h+1e-9*tangent
    y=gram_derivative(b,e,gram)
    oracle=SmoothDual(None,None,None).polar_derivative(
        SimpleNamespace(left=u[None],singular=s[None],right=vt[None]),e[None])[0]
    upper,_=derivative_error_bound(b,e,y,gram,posterior)
    assert float((y-oracle).norm())<=upper
    assert interval_sign(1e-14,1e-13)=="uncertain"
    assert interval_sign(-1e-14,1e-13)=="uncertain"
    threshold=2.-1e-4
    assert armijo_interval((2.,2.),(threshold-1e-12,threshold+1e-12),
                           (-1.,-1.),alpha=1.,c=1e-4)=="gram_armijo_ambiguous"


def test_research_adaptive_selector_recomputes_svd_only_for_uncertain_decision():
    b=matrix(.01)
    with patch.object(torch.linalg,"svd",side_effect=AssertionError("unneeded SVD")):
        gram=adaptive_decompose(b,lambda _g,_p:(True,"certified"))
    assert gram.backend=="gram_validated_research"
    full=adaptive_decompose(b,lambda _g,_p:(False,"gram_armijo_ambiguous"))
    assert full.backend=="full_svd_decomposition_fallback"
    assert full.reason=="gram_armijo_ambiguous"
    assert float((full.polar-gram.polar).norm()/full.polar.norm())<1e-10
