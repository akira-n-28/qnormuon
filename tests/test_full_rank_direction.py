"""Independent analytic/oracle checks for research posterior inequalities."""
import math
from unittest.mock import patch
import pytest
import torch
import qnormuon.coupled_solver as cs
from experiments.full_rank_direction import exact_gap_bound, direction_posterior


@pytest.mark.parametrize('spectrum',[(3.,2.,1.),(1.,1.,1e-5),(1.,1e-5,1e-5)])
def test_support_gap_for_prescribed_and_repeated_spectra(spectrum):
    generator=torch.Generator().manual_seed(917)
    left=torch.linalg.qr(torch.randn(9,3,generator=generator,dtype=torch.float64)).Q
    right=torch.linalg.qr(torch.randn(3,3,generator=generator,dtype=torch.float64)).Q
    b=(left*torch.tensor(spectrum,dtype=torch.float64))@right.T;q=left@right.T
    for _ in range(8):
        p=torch.randn(9,3,generator=generator,dtype=torch.float64)
        p/=torch.linalg.svdvals(p)[0]*1.001
        support=sum(spectrum)-float((b*p).sum())
        assert support>=min(spectrum)/2*float((q-p).square().sum())-1e-13


def data():
    u=torch.ones(6,2,dtype=torch.float64);d=u.clone()
    b=torch.zeros(6,2,dtype=torch.float64);b[0,0]=1;b[1,1]=1e-5
    return u,d,torch.stack((b,b)),torch.zeros(6,dtype=torch.float64)


def posterior_at(u,d,a,lam,candidate):
    b=a-cs.adjoint(u,d,lam);left,s,right=torch.linalg.svd(b,full_matrices=False)
    p,metrics=cs.certificate(u,d,a,lam,candidate,cs.Counts(),primal_norm_backend='gram_upper')
    return p,metrics,(left,s,right)


@pytest.mark.parametrize('amplitude',[1e-20,1.,1e20])
def test_full_rank_posterior_contains_analytic_optimum_below_guard(amplitude):
    u,d,a,lam=data();a*=amplitude
    left,s,right=torch.linalg.svd(a,full_matrices=False);opt=left@right
    # A nonstationary multiplier tests the arbitrary-current-residual theorem.
    lam[2]=amplitude*1e-7
    p,metrics,factors=posterior_at(u,d,a,lam,.97*opt)
    post=direction_posterior(u,d,a,lam,p,*factors,metrics)
    assert min(post['alpha_lower'])>0
    assert float((p-opt).norm())<=post['direction_error_upper']
    truegap=float(torch.linalg.svdvals(a-cs.adjoint(u,d,lam)).sum()-(a*p).sum())
    assert post['gap_upper']>=truegap
    assert max(post['alpha_lower'])<=float(torch.linalg.svdvals(a-cs.adjoint(u,d,lam))[:,-1].max())


def test_posterior_no_new_decomposition_or_reference_oracle():
    u,d,a,lam=data();p,metrics,factors=posterior_at(u,d,a,lam,a)
    with patch.object(torch.linalg,'svd',side_effect=AssertionError('new SVD')), \
         patch.object(torch.linalg,'svdvals',side_effect=AssertionError('new SVDVALS')), \
         patch.object(cs,'_reference',side_effect=AssertionError('oracle')):
        post=direction_posterior(u,d,a,lam,p,*factors,metrics)
    assert post['direction_error_upper']>0
    assert cs.SolverConfig().rcond_guard==1e-4


def test_approximate_feasibility_not_treated_as_exact():
    u,d,a,lam=data();p,metrics,factors=posterior_at(u,d,a,lam,a)
    p=p.clone();p[0,2,0]+=1e-14
    metrics=dict(metrics,spectral_norms=cs.primal_top_singular_upper(p).tolist())
    post=direction_posterior(u,d,a,lam,p,*factors,metrics)
    # Fresh independent conservative norm data for this altered candidate;
    # the test isolates horizontal repair rather than trusting old metadata.
    q=cs.horizontal_project(u,d,p)
    h=float((p-q).norm())
    assert post['horizontal_projection_distance_upper']>=h
    assert post['feasibility_distance_upper']>0


def test_rank_ambiguity_cannot_get_a_finite_direction_bound():
    u,d,a,lam=data();a[:,:,1]=0
    p,metrics,factors=posterior_at(u,d,a,lam,a)
    post=direction_posterior(u,d,a,lam,p,*factors,metrics)
    assert min(post['alpha_lower'])<=0
    assert math.isinf(post['direction_error_upper'])


def test_distance_bound_covers_every_optimum_without_multiplier_uniqueness():
    # Zero A: every feasible pair is optimal. Full-rank current residuals alone
    # cannot justify substituting their polar for the unique optimal direction.
    generator=torch.Generator().manual_seed(918)
    u=torch.randn(6,2,generator=generator,dtype=torch.float64);d=u.clone()
    lam=torch.ones(6,dtype=torch.float64);a=torch.zeros(2,6,2,dtype=torch.float64)
    b=a-cs.adjoint(u,d,lam);s=torch.linalg.svdvals(b)
    gap=float(s.sum());bound=exact_gap_bound(gap,float(s.min()))
    p=torch.stack((u,u));p/=torch.linalg.svdvals(u)[0]
    assert float((p-(-p)).norm())<=bound


def test_fp64_required_and_no_budget_chosen():
    u,d,a,lam=data();p,metrics,factors=posterior_at(u,d,a,lam,a)
    with pytest.raises(ValueError,match='fp64'):
        direction_posterior(u.float(),d,a,lam,p,*factors,metrics)
    with pytest.raises(ValueError):exact_gap_bound(1.,0.)
