"""Independent analytic and unchanged research-oracle admission regressions."""
import copy
from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
import qnormuon.coupled_solver as cs
import qnormuon.optimizer as om
from qnormuon._full_rank_admission import final_admission
from experiments.full_rank_admission import solve_full_rank
from experiments.full_rank_training import compact_metrics

V1 = cs.SolverConfig(admission_policy='full_rank_epsilon_lmo', fallback=False)


def diagonal(tail=9e-5, scale=1.):
    u=torch.tensor([[1.,0.],[0.,1.],[1.,0.]],dtype=torch.float64)
    b=scale*torch.tensor([[1.,0.],[0.,tail],[0.,0.]],dtype=torch.float64)
    return u,u.clone(),torch.stack((b,b))


def random_problem():
    g=torch.Generator().manual_seed(2087)
    u=torch.randn(12,3,generator=g,dtype=torch.float64)
    return u,u.clone(),torch.randn(2,12,3,generator=g,dtype=torch.float64)


def equivalent(u,d,a,initial=None,config=V1):
    with patch.object(cs,'_reference',side_effect=AssertionError('hidden reference')):
        old=solve_full_rank(u,d,a,config=config,initial_lambda=initial)
        new=cs.solve_coupled(u,d,a,config=config,initial_lambda=initial)
    assert (new.converged,new.reason,new.iterations,new.counts)==(old.converged,old.reason,old.iterations,old.counts)
    assert torch.equal(new.lam,old.lam)
    if old.pair is None:assert new.pair is None
    else:assert torch.equal(new.pair,old.pair)
    assert new.history==old.history and new.actions==old.actions and new.evaluations==old.evaluations
    for key,value in compact_metrics(old).items():
        if key not in ('rank_seconds','value_seconds','returned_direction_dtype'):
            assert new.metrics[key]==value,key
    assert new.admission_policy=='full_rank_epsilon_lmo'
    assert not new.reference_used and not new.fallback
    return new


@pytest.mark.parametrize('tail',[1.,3e-4,1.01e-4,9e-5,1e-6,1e-10,0.,1e-16])
@pytest.mark.parametrize('scale',[1e-20,1.,1e20])
def test_prescribed_spectra_and_research_equivalence(tail,scale):
    r=equivalent(*diagonal(tail,scale))
    assert r.converged==(tail>1e-16)
    if r.converged:
        assert min(r.metrics['alpha_lower'])>0 and r.metrics['conservative_normalized_gap']<=3e-5
        assert r.selection_semantics=='primary_epsilon_lmo_full_rank' and not r.selection_certified
    else:assert r.reason=='rank_ambiguous_or_deficient' and r.counts.hvp==0


def test_default_v0_rejects_below_guard_and_does_not_infer_policy():
    assert cs.SolverConfig().admission_policy=='v0_rcond'
    r=cs.solve_coupled(*diagonal(),config=cs.SolverConfig(fallback=False))
    assert not r.converged and r.reason=='ill_conditioned_residual'
    assert cs.SolverConfig(rcond_guard=1e-6).admission_policy=='v0_rcond'
    for kwargs in [dict(admission_policy='unknown'),dict(dtype=torch.float32),
                   dict(rcond_guard=1e-6),dict(tolerance=1e-4),dict(primal_norm_backend='svd')]:
        with pytest.raises(ValueError):replace(V1,**kwargs)


def test_newton_history_and_cached_posteriors_match_research():
    r=equivalent(*random_problem())
    assert r.converged and r.actions and final_admission(r.metrics['posterior'],r.metrics)
    for key in ['alpha_lower','eta','sigma_min','rcond_sides','sigma_min_over_eta',
                'dual_lower','dual_upper','primal_lower','G_upper','feasible_proxy_distance']:
        assert r.metrics[key] is not None


@pytest.mark.parametrize('shape',[(24,4),(64,12)])
@pytest.mark.parametrize('tail',[3e-4,1.5e-4,1.01e-4,9e-5,7e-5,5e-5,2e-5,1e-5,1e-6,1e-16,0.])
@pytest.mark.parametrize('scale',[1e-20,1.,1e20])
def test_prescribed_clustered_warm_admission_corpus(shape,tail,scale):
    m,n=shape;g=torch.Generator().manual_seed(2091)
    q=torch.linalg.qr(torch.randn(m,n,generator=g,dtype=torch.float64),mode='reduced').Q
    v=torch.linalg.qr(torch.randn(n,n,generator=g,dtype=torch.float64)).Q
    u=torch.randn(m,n,generator=g,dtype=torch.float64);u=u/u.norm(dim=1,keepdim=True)
    s=torch.full((n,),tail,dtype=torch.float64);s[0]=1.
    b=scale*(q*s)@v.T;a=torch.stack((b,b))
    initial=scale*tail*.5*torch.randn(m,generator=g,dtype=torch.float64)
    r=equivalent(u,u,a,initial)
    assert r.converged==(tail>1e-16)
    if r.converged:assert all(q['curvature']>0 for action in r.actions for q in action['queries'])


def test_package_path_needs_no_research_import_or_extra_posterior_svd():
    import builtins
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith('experiments'):raise AssertionError('package v1 imports research')
        return original(name,*args,**kwargs)
    with patch.object(builtins,'__import__',guarded):
        r=cs.solve_coupled(*random_problem(),config=V1)
    assert r.converged
    from qnormuon._numerical_certification import value_posterior
    u,d,a=diagonal();lam=torch.zeros(3,dtype=u.dtype)
    left,s,right=torch.linalg.svd(a,full_matrices=False)
    p,raw=cs.certificate(u,d,a,lam,left@right,cs.Counts(),primal_norm_backend='gram_upper')
    with patch.object(torch.linalg,'svd',side_effect=AssertionError('extra posterior SVD')), \
         patch.object(torch.linalg,'svdvals',side_effect=AssertionError('extra posterior SVDVALS')):
        post=value_posterior(u,d,a,lam,p,left,s,right,raw)
    assert final_admission(post,raw)


def test_repeated_positive_spectrum_and_nonunique_multiplier():
    r=equivalent(*diagonal(1.));assert r.converged
    u=torch.ones(2,1,dtype=torch.float64);a=torch.stack((u,u))
    results=[equivalent(u,u,a,torch.full((2,),v,dtype=torch.float64)) for v in (0.,.8)]
    assert all(r.converged for r in results)
    assert not torch.equal(results[0].lam,results[1].lam)


def test_exact_zero_and_primary_face_scope():
    u,d,a=diagonal()
    for objective in (torch.zeros_like(a),cs.adjoint(u,d,torch.tensor([1.,2.,3.],dtype=u.dtype))):
        r=equivalent(u,d,objective)
        assert r.selection_semantics=='exact_zero' and r.selection_certified
        assert torch.count_nonzero(r.pair)==0
    u,d,a=diagonal(0.)
    initial=torch.tensor([0.,1e-6,0.],dtype=u.dtype)
    r=equivalent(u,d,a,initial)
    selected=torch.zeros_like(a);selected[:,0,0]=1
    candidate=selected.clone();candidate[:,1,1]=1
    left,s,right=torch.linalg.svd(a-cs.adjoint(u,d,initial),full_matrices=False)
    p,raw=cs.certificate(u,d,a,initial,candidate,cs.Counts(),primal_norm_backend='gram_upper')
    from qnormuon._numerical_certification import value_posterior
    post=value_posterior(u,d,a,initial,p,left,s,right,raw)
    assert final_admission(post,raw) and float((p-selected).norm())>1.4
    assert r.converged and not post['selection_certified']
    assert not r.selection_certified
    r=cs.solve_coupled(u,d,a,config=V1,initial_lambda=initial,known_nonsmooth_face=True)
    assert not r.converged and r.pair is None and r.reason=='explicit_nonsmooth_face'


def test_cancellation_dominated_fails_without_reference_even_when_enabled():
    u,d,a=random_problem();vertical=cs.adjoint(u,d,torch.ones(12,dtype=u.dtype))
    r=cs.solve_coupled(u,d,vertical+1e-16*a,config=replace(V1,fallback=True))
    assert not r.converged and r.reason=='cancellation_dominated_cotangent' and r.pair is None


@pytest.mark.parametrize('kind',['nan_hvp','bad_curvature','line_search','budget','svd_failure'])
def test_operational_failures_never_call_reference(kind):
    u,d,a=random_problem()
    config=replace(V1,fallback=True,max_iterations=0 if kind=='budget' else 100)
    original=cs.SmoothDual.evaluate;calls=0
    def evaluate(self,z):
        nonlocal calls
        if kind=='svd_failure':raise torch.linalg.LinAlgError('injected SVD failure')
        ev=original(self,z);calls+=1
        if calls>1:ev.value+=100
        return ev
    if kind in ('nan_hvp','bad_curvature'):
        context=patch.object(cs.SmoothDual,'hvp',lambda self,ev,v:torch.full_like(v,float('nan')) if kind=='nan_hvp' else -1e20*v)
    elif kind in ('line_search','svd_failure'):context=patch.object(cs.SmoothDual,'evaluate',evaluate)
    else:context=patch.object(cs,'_reference',side_effect=AssertionError('hidden reference'))
    with context,patch.object(cs,'_reference',side_effect=AssertionError('hidden reference')):
        r=cs.solve_coupled(u,d,a,config=config)
    assert not r.converged and r.pair is None and not r.fallback
    assert r.metrics['failure_reason']==r.reason
    if kind=='line_search':assert r.reason=='line_search_failed' and r.counts.line_trials==24


def test_non_descent_and_recovered_invalid_trial_fail_closed():
    from qnormuon import _full_rank_admission as internal
    u,d,a=random_problem()
    with patch.object(internal,'newton_direction',
                      lambda problem,ev,*args:(ev.gradient.clone(),dict(queries=[],cg_termination='injected'))):
        r=cs.solve_coupled(u,d,a,config=V1)
    assert not r.converged and r.pair is None and r.reason=='no_descent_direction'
    original=cs.SmoothDual.evaluate;calls=0
    def first_trial_invalid(self,z):
        nonlocal calls
        calls+=1;ev=original(self,z)
        if calls==2:ev.singular[:,-1]=1e-18
        return ev
    with patch.object(cs.SmoothDual,'evaluate',first_trial_invalid):
        r=cs.solve_coupled(u,d,a,config=V1)
    assert not r.converged and r.pair is None
    assert any(e.get('eligible') is False for e in r.evaluations)
    # A successful backtrack cannot erase the trajectory's invalid-evaluation stop.
    assert r.reason.startswith('trajectory_invalid_evaluation:') or r.reason=='line_search_failed'


def make_optimizer(count=1,solver=V1):
    g=torch.Generator().manual_seed(16)
    pairs=[om.SwiGLUPair(str(i),torch.nn.Parameter(torch.randn(12,3,generator=g)),
                         torch.nn.Parameter(torch.randn(3,12,generator=g))) for i in range(count)]
    return om.QuotientSpectralOptimizer(pairs,solver=solver)


def set_gradients(o,seed):
    g=torch.Generator().manual_seed(seed)
    for p in o.pairs:
        p.up.grad=torch.randn(p.up.shape,generator=g);p.down.grad=torch.randn(p.down.shape,generator=g)


def test_late_pair_failure_commits_no_parameters_momenta_or_lambdas():
    o=make_optimizer(2);set_gradients(o,21);o.step()
    params=[v.detach().clone() for p in o.pairs for v in (p.up,p.down)]
    state=copy.deepcopy(o.state_dict());set_gradients(o,22);original=om.solve_coupled;calls=0
    def solve(*args,**kwargs):
        nonlocal calls
        calls+=1
        r=original(*args,**kwargs)
        if calls==2:r.converged=False;r.pair=None;r.reason='injected_late_failure'
        return r
    with patch.object(om,'solve_coupled',solve),pytest.raises(RuntimeError,match='injected_late_failure'):o.step()
    for before,after in zip(params,[v for p in o.pairs for v in (p.up,p.down)]):assert torch.equal(before,after)
    for key,values in state['state'].items():
        for name,value in values.items():
            after=o.state_dict()['state'][key][name]
            assert torch.equal(value,after) if isinstance(value,torch.Tensor) else value==after


def test_new_checkpoint_and_exact_fixed_backend_continuation():
    o=make_optimizer();set_gradients(o,61);o.step();saved=copy.deepcopy(o.state_dict())
    assert saved['qso_format_version']==3
    other=make_optimizer(solver=cs.SolverConfig())
    for p,q in zip(o.pairs,other.pairs):q.up.data.copy_(p.up);q.down.data.copy_(p.down)
    other.load_state_dict(saved)
    assert other.param_groups[0]['solver']['admission_policy']=='full_rank_epsilon_lmo'
    for seed in (62,63,64):
        set_gradients(o,seed);set_gradients(other,seed);o.step();other.step()
        for p,q in zip(o.pairs,other.pairs):
            assert torch.equal(p.up,q.up) and torch.equal(p.down,q.down)
            for key in ('momentum_up','momentum_down_t','lambda'):assert torch.equal(o.state[p.up][key],other.state[q.up][key])
        assert o.state[o.pairs[0].up]['momentum_up'].dtype==torch.float32
        assert o.state[o.pairs[0].up]['lambda'].dtype==torch.float64


@pytest.mark.parametrize('version',[1,2])
def test_historical_missing_policy_restores_v0_not_constructor_v1(version):
    o=make_optimizer(solver=cs.SolverConfig(fallback=False));set_gradients(o,11);o.step()
    saved=copy.deepcopy(o.state_dict());saved['qso_format_version']=version
    saved['param_groups'][0]['solver'].pop('admission_policy')
    if version==1:
        for values in saved['state'].values():
            for key in ('momentum_up','momentum_down_t'):values[key]=values[key].double()
    other=make_optimizer();other.load_state_dict(saved)
    assert other.param_groups[0]['solver']['admission_policy']=='v0_rcond'
    assert other.state[other.pairs[0].up]['momentum_up'].dtype==(torch.float64 if version==1 else torch.float32)
    broken=copy.deepcopy(saved);broken['qso_format_version']=3
    with pytest.raises(ValueError,match='explicit admission_policy'):other.load_state_dict(broken)
