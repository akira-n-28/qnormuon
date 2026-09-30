"""Diagnostic smooth Gram checks; independent full SVD remains the oracle."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from experiments.smooth_gram import diagnose
from qnormuon.coupled_solver import SmoothDual


def prescribed(m, n, low, *, repeated=False, scale=1.):
    gen=torch.Generator().manual_seed(5631+m+n)
    q,_=torch.linalg.qr(torch.randn(m,n,generator=gen,dtype=torch.float64))
    v,_=torch.linalg.qr(torch.randn(n,n,generator=gen,dtype=torch.float64))
    singular=torch.linspace(1.,.2,n,dtype=torch.float64)
    singular[-1]=low
    if repeated:
        singular[:2]=1.
        singular[-2:]=low
    return ((q*singular)@v.T)*scale


@pytest.mark.parametrize("low",[1e-2,3e-4,1.2e-4,1.01e-4,1e-4,.99e-4,5e-5])
def test_rcond_lower_bound_never_false_accepts_small_prescribed_spectra(low):
    b=prescribed(32,8,low)
    result=diagnose(b)
    spectrum=torch.linalg.svdvals(b)
    true_rcond=float(spectrum[-1]/spectrum[0])
    lower=result.measurements["rcond_lower"]
    assert lower <= true_rcond+1e-13
    assert true_rcond <= result.measurements["rcond_upper"]+1e-13
    assert not (result.accepted and true_rcond<=1e-4)


@pytest.mark.parametrize("repeated,scale",[(False,1.),(True,1.),(False,1e-100),(False,1e100)])
def test_polar_nuclear_and_derivative_against_independent_svd(repeated,scale):
    b=prescribed(40,10,1e-2,repeated=repeated,scale=scale)
    result=diagnose(b)
    assert result.accepted,result.reason
    left,singular,vt=torch.linalg.svd(b,full_matrices=False)
    oracle=left@vt
    assert float((result.polar-oracle).norm()/oracle.norm()) < 1e-10
    assert abs(float(result.singular.sum()-singular.sum()))/float(singular.sum()) < 1e-10
    assert result.measurements["rcond_lower"] <= float(singular[-1]/singular[0])+1e-13
    assert result.measurements["nuclear_lower"] <= float(singular.sum())
    assert float(singular.sum()) <= result.measurements["nuclear_upper"]
    direction=torch.randn(b.shape,generator=torch.Generator().manual_seed(73),dtype=torch.float64)*scale
    smooth=SmoothDual(None,None,None)
    derivative=smooth.polar_derivative(SimpleNamespace(left=result.left,singular=result.singular,right=result.right),direction)
    reference=smooth.polar_derivative(SimpleNamespace(left=left,singular=singular,right=vt),direction)
    assert float((derivative-reference).norm()/reference.norm()) < 1e-8


def test_negative_computed_eigenvalue_is_rejected_without_truncation():
    b=prescribed(16,4,1e-2)
    original=torch.linalg.eigh
    def corrupted(c):
        values,vectors=original(c)
        values=values.clone(); values[0]=-1e-2
        return values,vectors
    with patch.object(torch.linalg,"eigh",corrupted):
        result=diagnose(b)
    assert not result.accepted
    assert result.reason=="negative_eigenvalue_beyond_bound"
    assert result.singular is None


def test_small_polar_derivative_against_centered_difference():
    b=prescribed(24,7,1e-2,repeated=True)
    result=diagnose(b)
    assert result.accepted,result.reason
    e=torch.randn(b.shape,generator=torch.Generator().manual_seed(74),dtype=torch.float64)
    e=e/e.norm()
    smooth=SmoothDual(None,None,None)
    derivative=smooth.polar_derivative(SimpleNamespace(left=result.left,singular=result.singular,right=result.right),e)
    eps=1e-6
    def polar(x):
        u,_,vt=torch.linalg.svd(x,full_matrices=False)
        return u@vt
    fd=(polar(b+eps*e)-polar(b-eps*e))/(2*eps)
    assert float((derivative-fd).norm()/fd.norm()) < 1e-7
