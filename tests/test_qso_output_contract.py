"""Independent analytic counterexamples and value-posterior tests."""
from unittest.mock import patch
import pytest
import torch
import qnormuon.coupled_solver as cs
from experiments.qso_output_contract import value_posterior


def diagonal_problem(t=1e-5, down_scale=1.):
    u=torch.tensor([[1.,0.],[0.,1.],[1.,0.]],dtype=torch.float64)
    b=torch.tensor([[1.,0.],[0.,t],[0.,0.]],dtype=torch.float64)
    return u,u.clone(),torch.stack((b,down_scale*b))


def audit(u,d,a,lam,candidate):
    left,s,right=torch.linalg.svd(a-cs.adjoint(u,d,lam),full_matrices=False)
    p,metrics=cs.certificate(u,d,a,lam,candidate,cs.Counts(),primal_norm_backend='gram_upper')
    with patch.object(torch.linalg,'svd',side_effect=AssertionError('extra SVD')), \
         patch.object(torch.linalg,'svdvals',side_effect=AssertionError('extra SVDVALS')), \
         patch.object(cs,'_reference',side_effect=AssertionError('reference')):
        post=value_posterior(u,d,a,lam,p,left,s,right,metrics)
    return p,post


@pytest.mark.parametrize('tail,down_scale',[(1e-5,1.),(1e-10,1.),(1.,1.),(1e-5,1e-6)])
def test_analytic_support_value_not_direction_budget(tail,down_scale):
    u,d,a=diagonal_problem(tail,down_scale)
    candidate=torch.zeros_like(a);candidate[:,0,0]=1
    p,post=audit(u,d,a,torch.zeros(3,dtype=torch.float64),candidate)
    exact_value=(1+down_scale)*(1+tail)
    regret=exact_value-float((a*p).sum())
    assert 0 <= regret <= post['gap_upper']
    assert post['dual_lower'] <= exact_value <= post['dual_upper']
    assert regret/exact_value <= post['relative_support_regret_upper']
    if tail <= 1e-5:
        assert post['research_value_rank_pass']
        opt=torch.zeros_like(a);opt[:,0,0]=1;opt[:,1,1]=1
        assert float((p-opt).norm()) > 1.4


def test_nonunique_face_current_full_rank_does_not_certify_selection():
    u,d,a=diagonal_problem(0.)
    candidate=torch.zeros_like(a);candidate[:,0,0]=1;candidate[:,1,1]=1
    lam=torch.tensor([0.,1e-6,0.],dtype=torch.float64)
    p,post=audit(u,d,a,lam,candidate)
    selected=torch.zeros_like(a);selected[:,0,0]=1
    assert post['research_value_rank_pass']
    assert float((p-selected).norm())>1.4
    assert not post['selection_certified']
    # At an actual deficient optimal multiplier, the primary value still
    # certifies, but the separate smooth full-rank test correctly does not.
    _,deficient=audit(u,d,a,torch.zeros_like(lam),candidate)
    assert deficient['value_certified']
    assert not deficient['numerical_full_rank']


def test_unique_primal_nonunique_dual_and_gauge_canonical_contract():
    u=torch.ones(2,1,dtype=torch.float64);d=u.clone();a=torch.stack((u,u))
    exact=torch.stack((u,u))/2**.5
    outcomes=[]
    for t in (0.,.8):
        p,post=audit(u,d,a,torch.full((2,),t,dtype=torch.float64),exact)
        assert post['research_value_rank_pass'];outcomes.append(p)
    assert torch.allclose(*outcomes,atol=1e-14,rtol=0)
    c=torch.tensor([1e-4,1e4],dtype=torch.float64)
    rawu=c[:,None]*u;rawd=d/c[:,None];root=(rawu.norm(dim=1)/rawd.norm(dim=1)).sqrt()
    assert torch.equal(rawu/root[:,None],u)
    assert torch.equal(rawd*root[:,None],d)


@pytest.mark.parametrize('scale',[1e-20,1.,1e20])
def test_value_normalization_scales_without_direction_threshold(scale):
    u,d,a=diagonal_problem();a*=scale
    candidate=torch.zeros_like(a);candidate[:,0,0]=1
    p,post=audit(u,d,a,torch.zeros(3,dtype=torch.float64),candidate)
    assert post['research_value_rank_pass']
    assert post['normalized_gap_upper']==pytest.approx(1e-5/(1+1e-5),rel=1e-5)
    assert cs.SolverConfig().rcond_guard==1e-4
    assert cs.SolverConfig().tolerance==3e-5
