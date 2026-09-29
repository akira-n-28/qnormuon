"""Direct fp64 spectral work with independent fp32 canonical state."""
import copy
import pytest
import torch
import qnormuon.optimizer as om
from qnormuon import QuotientSpectralOptimizer as QSO, SolverConfig, SwiGLUPair, regular_canonicalize
from experiments.horizontal_spectral import random_example


def make(dtype=torch.float32,device='cpu'):
    u,d,a=random_example(902)
    p=SwiGLUPair('pair',torch.nn.Parameter(u.to(device=device,dtype=dtype)),
                torch.nn.Parameter(d.T.to(device=device,dtype=dtype)))
    p.up.grad=a[0].to(p.up);p.down.grad=a[1].T.to(p.down)
    return p


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_default_boundary_single_direct_solve_ema_and_update_cast(monkeypatch,device,dtype):
    p=make(dtype,device);o=QSO([p],lr=.05,beta=.7)
    calls=[];original=om.solve_coupled
    def observed(u,d,a,**kw):
        assert u.dtype==d.dtype==a.dtype==torch.float64
        assert kw['config'].dtype==torch.float64 and kw['config'].tolerance==3e-5
        result=original(u,d,a,**kw);calls.append((a.clone(),kw['initial_lambda'],result))
        return result
    monkeypatch.setattr(om,'solve_coupled',observed)
    for step in range(2):
        before_u,before_d=p.up.detach().clone(),p.down.detach().clone()
        _,_,root=regular_canonicalize(p.up,p.down)
        gu=(root[:,None]*p.up.grad.double()).float()
        gd=(p.down.grad.T.double()/root[:,None]).float()
        expected_u=.7*o.state.get(p.up,{}).get('momentum_up',torch.zeros_like(gu))+.3*gu
        expected_d=.7*o.state.get(p.up,{}).get('momentum_down_t',torch.zeros_like(gd))+.3*gd
        with torch.autocast(device_type=device,dtype=torch.bfloat16):o.step()
        assert len(calls)==step+1  # no fp32 attempt, refinement, or retry
        a,lam,r=calls[-1]
        torch.testing.assert_close(a,torch.stack((expected_u,expected_d)).double(),rtol=1e-7,atol=0)
        assert o.state[p.up]['momentum_up'].dtype==o.state[p.up]['momentum_down_t'].dtype==torch.float32
        assert o.state[p.up]['lambda'].dtype==torch.float64
        assert lam is None if step==0 else torch.equal(lam,calls[-2][2].lam)
        assert torch.equal(p.up,before_u-.05*(root[:,None]*r.pair[0]).to(dtype))
        assert torch.equal(p.down,before_d-.05*(r.pair[1]/root[:,None]).T.to(dtype))
        assert r.metrics['gap_tolerance']==3e-5


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_default_checkpoint_continuation_preserves_separated_precision(dtype):
    p=make(dtype);o=QSO([p]);o.step();saved=copy.deepcopy(o.state_dict())
    q=SwiGLUPair('pair',torch.nn.Parameter(p.up.detach().clone()),torch.nn.Parameter(p.down.detach().clone()))
    q.up.grad=p.up.grad.clone();q.down.grad=p.down.grad.clone()
    resumed=QSO([q]);resumed.load_state_dict(saved)
    for _ in range(2):
        o.step();resumed.step()
        assert torch.equal(p.up,q.up) and torch.equal(p.down,q.down)
        for key in ('momentum_up','momentum_down_t','lambda'):
            assert torch.equal(o.state[p.up][key],resumed.state[q.up][key])
    assert resumed.param_groups[0]['solver']['dtype']==torch.float64
    assert resumed.param_groups[0]['momentum_dtype']==torch.float32


def test_default_policy_zero_and_expensive_fallback_diagnostics():
    p=make();p.up.grad.zero_();p.down.grad.zero_();before=p.up.detach().clone();o=QSO([p]);o.step()
    assert torch.equal(before,p.up) and torch.count_nonzero(o.state[p.up]['momentum_up'])==0
    assert o.last_diagnostics['pair']['normalized_gap']==0
    p=make();o=QSO([p],solver=SolverConfig(max_iterations=0));o.step()
    m=o.last_diagnostics['pair']
    assert m['fallback_used'] and m['fallback_reason']=='iteration_budget'
    assert m['fallback_backend']=='cpu_float64_reference_admm'
    assert m['solver_dtype']=='torch.float64' and m['normalized_gap']<=3e-5
    assert m['residual_rcond']>0


def test_default_gauge_equivariance_with_fp32_state():
    p=make();q=make();c=2.**torch.linspace(-16,16,p.up.shape[0],dtype=p.up.dtype).round()
    with torch.no_grad():q.up.mul_(c[:,None]);q.down.div_(c)
    q.up.grad=p.up.grad/c[:,None];q.down.grad=p.down.grad*c
    a,b=QSO([p]),QSO([q])
    for _ in range(3):
        a.step();b.step()
        torch.testing.assert_close(q.up/c[:,None],p.up,rtol=3e-5,atol=3e-5)
        torch.testing.assert_close(q.down*c,p.down,rtol=3e-5,atol=3e-5)
        torch.testing.assert_close(a.state[p.up]['momentum_up'],b.state[q.up]['momentum_up'],rtol=3e-5,atol=3e-5)


def test_legacy_checkpoint_is_explicit_policy_not_silent_precision_migration():
    p=make(torch.float64)
    o=QSO([p],momentum_dtype=torch.float64,solver=SolverConfig(tolerance=1e-8,rcond_guard=1e-8));o.step()
    saved=copy.deepcopy(o.state_dict());saved['qso_format_version']=1
    for g in saved['param_groups']:
        del g['momentum_dtype'];g['solver']['tolerance']=None;g['solver']['rcond_guard']=None
    q=make(torch.float64)
    with torch.no_grad():q.up.copy_(p.up);q.down.copy_(p.down)
    resumed=QSO([q]);resumed.load_state_dict(saved)
    assert resumed.param_groups[0]['solver']['tolerance']==1e-8
    assert resumed.param_groups[0]['momentum_dtype']==torch.float64
    o.step();resumed.step()
    assert torch.equal(p.up,q.up) and torch.equal(p.down,q.down)


def test_default_target_is_independent_of_dtype_and_none():
    from qnormuon.coupled_solver import solve_coupled
    u,d,a=random_example(902)
    for config in (SolverConfig(),SolverConfig(tolerance=None),SolverConfig(dtype=torch.float32)):
        r=solve_coupled(u,d,a,config=config)
        assert r.metrics['gap_tolerance']==3e-5
        assert r.converged and r.metrics['normalized_gap']<=3e-5


def test_basic_and_full_cast_diagnostics_have_identical_trajectory():
    basic_pair, full_pair = make(), make()
    basic = QSO([basic_pair], lr=.05)
    full = QSO([full_pair], lr=.05, cast_diagnostics=True)
    for _ in range(2):
        basic.step(); full.step()
        assert torch.equal(basic_pair.up, full_pair.up)
        assert torch.equal(basic_pair.down, full_pair.down)
        for key in ('momentum_up', 'momentum_down_t', 'lambda'):
            assert torch.equal(basic.state[basic_pair.up][key], full.state[full_pair.up][key])
        left, right = basic.last_diagnostics['pair'], full.last_diagnostics['pair']
        for key in ('normalized_gap', 'horizontal_residual', 'residual_rcond',
                    'newton_iterations', 'cg_iterations', 'fallback_used', 'solver_dtype'):
            assert left[key] == right[key]
        assert 'cast_spectral_excess' not in left
        assert 'cast_spectral_excess' in right
        assert right['cast_diagnostic_svd_evaluations'] == 2
