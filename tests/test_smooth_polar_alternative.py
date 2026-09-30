"""Independent SVD checks for the research-only QDWH polar candidate."""
import pytest
import torch

from experiments.smooth_polar_alternative import diagnose, polar_derivative


def prescribed(m=20,n=6,ratio=2e-4,seed=91):
    gen=torch.Generator().manual_seed(seed)
    u,_=torch.linalg.qr(torch.randn(m,n,dtype=torch.float64,generator=gen))
    v,_=torch.linalg.qr(torch.randn(n,n,dtype=torch.float64,generator=gen))
    spectrum=torch.tensor([1.,1.,.7,.7,.2,ratio],dtype=torch.float64)
    return (u*spectrum)@v.T


@pytest.mark.parametrize("route",["tall","qr_square"])
@pytest.mark.parametrize("ratio",[1.,1e-2,3e-4,1.01e-4])
def test_qdwh_polar_spectrum_and_spd(route,ratio):
    b=prescribed(ratio=ratio)
    candidate=diagnose(b,route=route)
    assert candidate.accepted,candidate.reason
    u,s,vt=torch.linalg.svd(b,full_matrices=False)
    assert torch.allclose(candidate.polar,u@vt,rtol=0,atol=1e-10)
    assert torch.allclose(candidate.singular,s,rtol=1e-9,atol=1e-12)
    assert torch.allclose(candidate.polar@candidate.h,b,rtol=1e-10,atol=1e-10)
    assert float(torch.linalg.eigvalsh(candidate.h).min())>0
    assert abs(float(candidate.singular.sum())-float(s.sum()))<1e-9


@pytest.mark.parametrize("route",["tall","qr_square"])
@pytest.mark.parametrize("amplitude",[1e-150,1e150])
def test_qdwh_extreme_scale(route,amplitude):
    b=prescribed(ratio=3e-4)*amplitude
    candidate=diagnose(b,route=route)
    assert candidate.accepted,candidate.reason
    x=b/amplitude
    u,s,vt=torch.linalg.svd(x,full_matrices=False)
    assert torch.allclose(candidate.polar,u@vt,rtol=0,atol=1e-10)
    assert torch.allclose(candidate.singular/amplitude,s,rtol=1e-8,atol=1e-12)


@pytest.mark.parametrize("route",["tall","qr_square"])
def test_qdwh_guard_rejection(route):
    candidate=diagnose(prescribed(ratio=0.99e-4),route=route)
    assert not candidate.accepted
    assert candidate.reason in ("rcond_uncertain","qdwh_iteration_budget")


@pytest.mark.parametrize("route",["tall","qr_square"])
def test_sylvester_derivative_matches_svd_and_finite_difference(route):
    b=prescribed(ratio=2e-4)
    candidate=diagnose(b,route=route)
    assert candidate.accepted,candidate.reason
    gen=torch.Generator().manual_seed(23)
    e=torch.randn(b.shape,dtype=b.dtype,generator=gen)
    got=polar_derivative(candidate,e)
    u,s,vt=torch.linalg.svd(b,full_matrices=False)
    ev=e@vt.T
    f=u.T@ev
    omega=(f-f.T)/(s[:,None]+s[None,:])
    expected=(u@omega+(ev-u@f)/s[None,:])@vt
    assert float((got-expected).norm()/expected.norm())<1e-9
    step=1e-8/float(e.norm())
    plus=torch.linalg.svd(b+step*e,full_matrices=False)
    minus=torch.linalg.svd(b-step*e,full_matrices=False)
    finite=((plus[0]@plus[2])-(minus[0]@minus[2]))/(2*step)
    assert float((got-finite).norm()/got.norm())<1e-5


def test_qdwh_candidate_never_calls_svd(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("candidate invoked full SVD")
    monkeypatch.setattr(torch.linalg,"svd",forbidden)
    assert diagnose(prescribed(ratio=3e-4),route="tall").accepted
