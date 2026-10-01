"""Research admission tests: analytic answers and independent full SVD oracle."""
from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
import qnormuon.coupled_solver as cs
from experiments.full_rank_admission import (solve_full_rank, final_admission,
    rank_posterior, PRIMARY_SCOPE)
from experiments.qso_output_contract import value_posterior


def diagonal(tail=9e-5, scale=1.):
    u=torch.tensor([[1.,0.],[0.,1.],[1.,0.]],dtype=torch.float64)
    b=scale*torch.tensor([[1.,0.],[0.,tail],[0.,0.]],dtype=torch.float64)
    return u,u.clone(),torch.stack((b,b))


@pytest.mark.parametrize('tail',[1.,3e-4,9e-5,1e-6,1e-10])
@pytest.mark.parametrize('scale',[1e-20,1.,1e20])
def test_primary_full_rank_without_lower_rcond_constant(tail,scale):
    u,d,a=diagonal(tail,scale)
    with patch.object(cs,'_reference',side_effect=AssertionError('hidden reference')):
        r=solve_full_rank(u,d,a)
    assert r.converged and r.reason=='primary_certified'
    assert min(r.metrics['posterior']['alpha_lower'])>0
    assert r.metrics['posterior']['normalized_gap_upper']<=3e-5
    assert r.selection_semantics==PRIMARY_SCOPE and not r.selection_certified
    exact=torch.zeros_like(a);exact[:,0,0]=1;exact[:,1,1]=1
    assert torch.allclose(r.pair,exact,atol=1e-12,rtol=0)
    if tail<1e-4:
        v0=cs.solve_coupled(u,d,a,config=cs.SolverConfig(fallback=False))
        assert not v0.converged and v0.reason=='ill_conditioned_residual'
    assert cs.SolverConfig().rcond_guard==1e-4


@pytest.mark.parametrize('tail',[0.,1e-16])
def test_deficient_or_ambiguous_rank_fails_before_any_newton(tail):
    u,d,a=diagonal(tail)
    with patch.object(cs.SmoothDual,'hvp',side_effect=AssertionError('unsafe HVP')):
        r=solve_full_rank(u,d,a)
    assert not r.converged and r.pair is None
    assert r.reason=='rank_ambiguous_or_deficient' and r.counts.hvp==0


def test_exact_intrinsic_zero_bypasses_rank_and_selects_zero():
    u,d,a=diagonal();lam=torch.tensor([1.,2.,3.],dtype=torch.float64)
    for objective in (torch.zeros_like(a),cs.adjoint(u,d,lam)):
        r=solve_full_rank(u,d,objective)
        assert r.converged and r.reason=='zero_cotangent'
        assert torch.count_nonzero(r.pair)==0
        assert r.selection_semantics=='exact_zero' and r.selection_certified


def test_unique_primal_does_not_require_unique_dual():
    u=torch.ones(2,1,dtype=torch.float64);a=torch.stack((u,u))
    results=[solve_full_rank(u,u,a,initial_lambda=torch.full((2,),t,dtype=torch.float64)) for t in (0.,.8)]
    assert all(r.converged for r in results)
    assert torch.allclose(results[0].pair,results[1].pair,atol=1e-14,rtol=0)
    assert not torch.equal(results[0].lam,results[1].lam)


def test_primary_rank_value_never_certifies_pdagger_on_nonunique_face():
    u,d,a=diagonal(0.)
    lam=torch.tensor([0.,1e-6,0.],dtype=torch.float64)
    candidate=torch.zeros_like(a);candidate[:,0,0]=1;candidate[:,1,1]=1
    left,s,right=torch.linalg.svd(a-cs.adjoint(u,d,lam),full_matrices=False)
    p,metrics=cs.certificate(u,d,a,lam,candidate,cs.Counts(),primal_norm_backend='gram_upper')
    post=value_posterior(u,d,a,lam,p,left,s,right,metrics)
    assert final_admission(post,metrics) and not post['selection_certified']
    selected=candidate.clone();selected[:,1,1]=0
    assert float((p-selected).norm())>1.4
    result=solve_full_rank(u,d,a,initial_lambda=lam)
    assert result.converged and result.selection_semantics==PRIMARY_SCOPE
    assert not result.selection_certified
    excluded=solve_full_rank(u,d,a,initial_lambda=lam,known_nonsmooth_face=True)
    assert not excluded.converged and excluded.pair is None
    assert excluded.reason=='explicit_nonsmooth_face'


def random_problem():
    generator=torch.Generator().manual_seed(2087)
    u=torch.randn(12,3,generator=generator,dtype=torch.float64);d=u.clone()
    a=torch.randn(2,12,3,generator=generator,dtype=torch.float64)
    return u,d,a


def test_newton_replay_equals_production_when_both_admit():
    u,d,a=random_problem()
    production=cs.solve_coupled(u,d,a,config=cs.SolverConfig(fallback=False))
    research=solve_full_rank(u,d,a)
    assert production.converged and research.converged
    assert production.iterations==research.iterations
    assert production.counts==research.counts
    assert torch.equal(production.lam,research.lam)
    assert torch.equal(production.pair,research.pair)
    assert all(q['curvature']>0 for act in research.actions for q in act['queries'])


def test_no_extra_rank_or_final_posterior_decomposition():
    u,d,a=diagonal()
    ev=cs.SmoothDual(u,d,a).evaluate(torch.zeros(3,dtype=torch.float64))
    with patch.object(torch.linalg,'svd',side_effect=AssertionError('extra SVD')), \
         patch.object(torch.linalg,'svdvals',side_effect=AssertionError('extra SVDVALS')):
        post=rank_posterior(u,d,a,ev.coordinate,ev,1.)
    assert min(post['alpha_lower'])>0


def test_nonfinite_hvp_and_unusable_curvature_fail_closed():
    u,d,a=random_problem()
    for output,reason in [(float('nan'),'nonfinite_action'),(0.,'unusable_cg_result')]:
        def bad_hvp(self,ev,v):
            return torch.full_like(v,output) if output!=0 else -1e20*v
        with patch.object(cs.SmoothDual,'hvp',bad_hvp):
            r=solve_full_rank(u,d,a)
        assert not r.converged and r.pair is None and r.reason==reason


def test_iteration_budget_cannot_return_uncertified_candidate():
    u,d,a=random_problem()
    r=solve_full_rank(u,d,a,config=replace(cs.SolverConfig(fallback=False),max_iterations=0))
    assert not r.converged and r.reason=='iteration_budget' and r.pair is None
    assert not r.reference_used


def test_frozen_thresholds_and_represented_dtype():
    u,d,a=diagonal()
    with pytest.raises(ValueError,match='frozen'):
        solve_full_rank(u,d,a,config=replace(cs.SolverConfig(),rcond_guard=1e-6))
    with pytest.raises(ValueError,match='fp64'):
        solve_full_rank(u.float(),d,a)


def test_line_search_exhaustion_and_rank_trial_rejection_fail_closed():
    u,d,a=random_problem();original=cs.SmoothDual.evaluate
    for kind in ('bad_objective','rank_ambiguous'):
        calls=0
        def failed_trial(self,z):
            nonlocal calls
            ev=original(self,z);calls+=1
            if calls>1:
                if kind=='bad_objective':ev.value+=100.
                else:ev.singular[:,-1]=1e-18
            return ev
        with patch.object(cs.SmoothDual,'evaluate',failed_trial):
            r=solve_full_rank(u,d,a)
        assert not r.converged and r.pair is None and r.reason=='line_search_failed'
        assert r.counts.line_trials==24
        if kind=='rank_ambiguous':
            assert all(not t['eligible'] for t in r.actions[0]['trials'])


def test_factor_failure_and_final_thresholds_are_not_silent_acceptance():
    u,d,a=diagonal()
    ev=cs.SmoothDual(u,d,a).evaluate(torch.zeros(3,dtype=torch.float64))
    ev.left*=3
    with pytest.raises(RuntimeError,match='factor_posterior_failed'):
        rank_posterior(u,d,a,ev.coordinate,ev,1.)
    r=solve_full_rank(u,d,a);post=r.metrics['posterior'];metrics=r.metrics
    for field,value in [('normalized_gap_upper',3e-5+1e-12),('dual_lower',0.),
                         ('alpha_lower',[0.,1.]),('gap_upper',float('nan'))]:
        assert not final_admission(dict(post,**{field:value}),metrics)
    for field,value in [('signed_normalized_gap',-1e-10-1e-12),
                        ('normalized_horizontal_residual',1e-10+1e-12),
                        ('spectral_excess',1e-12+1e-14)]:
        assert not final_admission(post,dict(metrics,**{field:value}))
