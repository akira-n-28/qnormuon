"""Independent small full-rank/deficient controls for fixture classification."""
import torch
import pytest
import qnormuon.coupled_solver as cs
from experiments.near_rank_optimum import research_config,kkt,independent_lbfgs,decomposition_check


def below_guard():
    u=torch.ones(4,2,dtype=torch.float64);d=u.clone()
    b=torch.tensor([[1.,0.],[0.,1e-5],[0.,0.],[0.,0.]],dtype=torch.float64)
    return u,d,torch.stack((b,b))


def test_guard_is_not_mathematical_rank():
    u,d,a=below_guard()
    default=cs.SolverConfig(fallback=False)
    rejected=cs.solve_coupled(u,d,a,config=default)
    assert rejected.reason=='ill_conditioned_residual'
    solved=cs.solve_coupled(u,d,a,config=research_config())
    assert solved.converged and not solved.fallback
    check,_,_=kkt(u,d,a,solved.lam)
    assert check['gradient_norm']==0
    assert check['metrics']['normalized_gap']<1e-12
    assert check['sides'][0]['sigma_min']==1e-5
    assert cs.SolverConfig().rcond_guard==1e-4


def test_independent_lbfgs_kkt_from_nonstationary_start():
    generator=torch.Generator().manual_seed(610)
    u=torch.randn(8,3,generator=generator,dtype=torch.float64)
    d=torch.randn(8,3,generator=generator,dtype=torch.float64)
    u=u/u.norm(dim=1,keepdim=True);d=d/d.norm(dim=1,keepdim=True)
    a=torch.randn(2,8,3,generator=generator,dtype=torch.float64)
    lam,history,counts=independent_lbfgs(u,d,a,torch.zeros(8,dtype=torch.float64))
    check,_,_=kkt(u,d,a,lam)
    assert check['normalized_gradient_norm']<1e-9
    assert check['metrics']['normalized_gap']<1e-9
    assert counts.hvp==0
    assert len(history)>1


def test_rank_margin_vs_prescribed_positive_spectrum():
    u,d,a=below_guard();b=a
    left,s,right=torch.linalg.svd(b,full_matrices=False)
    checked=decomposition_check(b,left,s,right)
    assert min(checked['rank_margin'])>1e6
    assert max(checked['reconstruction_relative'])<1e-14


def test_full_rank_kkt_does_not_complete_a_deficient_face():
    u,d,a=below_guard();a[:,:,1]=0
    with pytest.raises(ValueError,match='joint completion'):
        kkt(u,d,a,torch.zeros(4,dtype=torch.float64))
