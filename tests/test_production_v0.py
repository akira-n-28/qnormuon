"""Production contracts checked against independent mathematical oracles."""
import copy
import io
import pytest
import torch
import torch.nn.functional as F
from qnormuon import (QuotientSpectralOptimizer as QSO, SwiGLUPair, SolverConfig,
                      regular_canonicalize, solve_coupled)
from qnormuon.coupled_solver import SmoothDual
from experiments.dual_solver import reference, comparison
from experiments.horizontal_spectral import random_example, fractional_example, adjoint
from experiments.dual_solver_study import synthetic_sequence


def close(a, b, tol=2e-8):
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def pair(seed=902, dtype=torch.float64, device='cpu', name='block'):
    u, d, a = random_example(seed, dtype=dtype)
    p = SwiGLUPair(name, torch.nn.Parameter(u.to(device)), torch.nn.Parameter(d.T.to(device)))
    p.up.grad, p.down.grad = a[0].to(device), a[1].T.to(device)
    return p


@pytest.mark.parametrize('seed', [901, 902, 903])
@pytest.mark.parametrize('dtype,tol', [(torch.float64, 1e-8), (torch.float32, 3e-5)])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_full_rank_reference_certificates_and_finite_step(seed, dtype, tol, device):
    # Generate the same verified full-rank instances before dtype conversion.
    u, d, a = (v.to(dtype) for v in random_example(seed))
    oracle = reference(u, d, a)
    spectrum = torch.linalg.svdvals(a.double() - adjoint(u.double(), d.double(), oracle.lam))
    assert float((spectrum[:, -1] / spectrum[:, 0]).min()) > 1e-3
    r = solve_coupled(u.to(device), d.to(device), a.to(device), config=SolverConfig(dtype=dtype,tolerance=tol))
    assert r.converged and not r.fallback
    assert r.metrics['normalized_gap'] <= tol
    assert r.metrics['normalized_horizontal_residual'] <= 1e-10
    assert r.metrics['spectral_excess'] <= 1e-12
    errors = comparison(u, d, a, r.pair.cpu(), oracle.pair)
    assert errors['relative_direction_error'] < (3e-4 if dtype == torch.float64 else .01)
    assert errors['relative_finite_delta_x_error'] < (3e-4 if dtype == torch.float64 else .01)
    assert r.metrics['residual_rcond'] > (1e-8 if dtype == torch.float64 else 1e-4)
    assert r.pair.dtype == r.lam.dtype == torch.float64
    print('PRODUCTION_ORACLE', device, dtype, seed, r.metrics, errors)


@pytest.mark.parametrize('vertical', [False, True])
def test_exact_zero_intrinsic_cotangent_and_no_parameter_motion(vertical):
    u = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], dtype=torch.float64)
    a = adjoint(u, u, torch.tensor([2., -4., 8.], dtype=u.dtype)) if vertical else torch.zeros(2,3,2,dtype=u.dtype)
    r = solve_coupled(u, u, a, config=SolverConfig(dtype=u.dtype,tolerance=1e-8,rcond_guard=1e-8))
    assert r.converged and torch.count_nonzero(r.pair) == 0
    p = SwiGLUPair('zero', torch.nn.Parameter(u.clone()), torch.nn.Parameter(u.T.clone()))
    p.up.grad, p.down.grad = a[0], a[1].T
    opt = QSO([p], beta=0, solver=SolverConfig(dtype=u.dtype,tolerance=1e-8,rcond_guard=1e-8))
    opt.step()
    assert torch.equal(p.up, u) and torch.equal(p.down, u.T)


@pytest.mark.parametrize('dtype,extent,tol', [(torch.float64, 150, 2e-8), (torch.float32, 15, 3e-5)])
def test_extreme_positive_gauge_and_complete_momentum_lift(dtype, extent, tol):
    base = pair(dtype=dtype)
    c = torch.logspace(-extent, extent, 7, dtype=dtype)
    gauged = SwiGLUPair('block', torch.nn.Parameter(base.up.detach()*c[:,None]),
                       torch.nn.Parameter(base.down.detach()/c))
    opts = [QSO([p], lr=.01, beta=.7, momentum_dtype=dtype, solver=SolverConfig(dtype=dtype,tolerance=1e-8 if dtype==torch.float64 else 3e-5)) for p in (base,gauged)]
    for _ in range(3):
        gauged.up.grad = base.up.grad / c[:,None]
        gauged.down.grad = base.down.grad * c
        for opt in opts:
            opt.step()
        close(gauged.up/c[:,None], base.up, tol)
        close(gauged.down*c, base.down, tol)
        for key in ('momentum_up','momentum_down_t','lambda'):
            close(opts[0].state[base.up][key],opts[1].state[gauged.up][key],tol)


@pytest.mark.parametrize('side', ['up', 'down', 'both'])
def test_zero_rows_rejected_without_mutating_weights_or_state(side):
    p = pair()
    with torch.no_grad():
        if side in ('up','both'): p.up[0].zero_()
        if side in ('down','both'): p.down[:,0].zero_()
    before = p.up.clone(),p.down.clone()
    opt = QSO([p])
    with pytest.raises(ValueError, match='nonzero'):
        opt.step()
    assert not opt.state
    assert torch.equal(before[0],p.up) and torch.equal(before[1],p.down)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_promoted_storage_path_autocast_and_no_bf16_svd(monkeypatch, device, dtype):
    p = pair(dtype=dtype,device=device)
    seen=[]
    for name in ('svd','svdvals'):
        original=getattr(torch.linalg,name)
        def checked(matrix,*args,_original=original,**kwargs):
            seen.append(matrix.dtype)
            assert matrix.dtype in (torch.float32,torch.float64)
            return _original(matrix,*args,**kwargs)
        monkeypatch.setattr(torch.linalg,name,checked)
    opt=QSO([p],lr=.05)
    before=p.up.detach().clone()
    with torch.autocast(device_type=device,dtype=torch.bfloat16):
        opt.step()
    assert seen and set(seen) == {torch.float64}
    assert p.up.dtype == dtype and not torch.equal(before,p.up)
    assert opt.state[p.up]['momentum_up'].dtype == torch.float32
    assert opt.state[p.up]['lambda'].dtype == torch.float64
    assert opt.last_diagnostics['block']['solver_dtype']=='torch.float64'


def test_warm_lambda_original_coordinates_when_weights_and_objective_change(monkeypatch):
    frames=synthetic_sequence(m=24,n=8,steps=3)
    first=solve_coupled(*frames[0],config=SolverConfig(dtype=torch.float64,tolerance=1e-8,rcond_guard=1e-8))
    u,d,a=frames[1]
    # Nonuniform constraint scaling and objective scaling change internal z.
    scale=torch.logspace(-2,2,u.shape[0],dtype=u.dtype)
    warm=solve_coupled(u*scale[:,None],d*scale[:,None],3*a,
                      initial_lambda=3*first.lam/scale,config=SolverConfig(dtype=u.dtype,tolerance=1e-8,rcond_guard=1e-8))
    cold=solve_coupled(u*scale[:,None],d*scale[:,None],3*a,config=SolverConfig(dtype=u.dtype,tolerance=1e-8,rcond_guard=1e-8))
    assert warm.converged and cold.converged and warm.iterations < cold.iterations
    close(warm.pair,cold.pair,2e-4)
    import qnormuon.optimizer as module
    original=module.solve_coupled
    starts=[]
    def observed(*args,**kw):
        starts.append(kw['initial_lambda'])
        return original(*args,**kw)
    monkeypatch.setattr(module,'solve_coupled',observed)
    p=pair()
    opt=QSO([p],solver=SolverConfig(dtype=torch.float64,tolerance=1e-8,rcond_guard=1e-8))
    opt.step()
    previous=opt.state[p.up]['lambda'].clone()
    opt.step()
    assert starts[0] is None
    assert torch.equal(starts[1],previous)


@pytest.mark.parametrize('dtype', [torch.float64,torch.bfloat16])
def test_checkpoint_roundtrip_named_order_independent_and_precision_preserved(dtype):
    pairs=[pair(901,dtype,name='z'),pair(902,dtype,name='a')]
    config=SolverConfig(dtype=torch.float64 if dtype==torch.float64 else torch.float32)
    opt=QSO(pairs,lr=.02,solver=config)
    opt.step()
    buffer=io.BytesIO()
    torch.save(opt.state_dict(),buffer)
    buffer.seek(0)
    saved=torch.load(buffer,weights_only=True)
    copies=[SwiGLUPair(p.name,torch.nn.Parameter(p.up.detach().clone()),
                      torch.nn.Parameter(p.down.detach().clone())) for p in reversed(pairs)]
    resumed=QSO(copies)
    resumed.load_state_dict(saved)
    for _ in range(2):
        for p in pairs:
            q=next(q for q in copies if q.name==p.name)
            q.up.grad=p.up.grad.clone(); q.down.grad=p.down.grad.clone()
        opt.step(); resumed.step()
        for p in pairs:
            q=next(q for q in copies if q.name==p.name)
            assert torch.equal(p.up,q.up) and torch.equal(p.down,q.down)
            for key in ('momentum_up','momentum_down_t','lambda'):
                assert torch.equal(opt.state[p.up][key],resumed.state[q.up][key])
    bad=copy.deepcopy(saved);bad['qso_pairs'][0]['name']='wrong'
    with pytest.raises(ValueError,match='topology'):resumed.load_state_dict(bad)


def test_fallback_rank_loss_and_fractional_direction_are_observable():
    u,d,a,expected=fractional_example()
    r=solve_coupled(u,d,a,config=SolverConfig(dtype=torch.float64,tolerance=1e-8,rcond_guard=1e-8))
    assert r.converged and r.fallback
    assert r.reason=='gap_stagnation'  # independently reproduced by the research solver
    assert 'primary_only' in r.secondary_selection
    close(r.pair,expected,3e-8)
    a=torch.zeros_like(a);a[:,0,0]=1.;a[:,1,1]=1e-12
    r=solve_coupled(u,u,a,config=SolverConfig(dtype=torch.float64,tolerance=1e-8,rcond_guard=1e-8))
    assert r.fallback and r.converged
    assert r.reason == 'ill_conditioned_residual'
    assert r.metrics['normalized_gap']<1e-8
    exact=a.clone();exact[:,1,1]=1
    assert (r.pair-exact).norm()>1  # small gap is not direction certification


def test_exhausted_budget_explicit_fallback_or_failure_and_no_partial_update():
    p=pair()
    config=SolverConfig(dtype=torch.float64,max_iterations=0)
    opt=QSO([p],solver=config)
    opt.step()
    assert opt.last_diagnostics['block']['fallback_used']
    assert opt.last_diagnostics['block']['fallback_reason']=='iteration_budget'
    p=pair();before=p.up.clone()
    opt=QSO([p],solver=SolverConfig(dtype=torch.float64,max_iterations=0,fallback=False))
    with pytest.raises(RuntimeError,match='not certified'):opt.step()
    assert torch.equal(p.up,before) and not opt.state
    assert not opt.last_diagnostics['block']['converged']


def test_transpose_raw_covector_once_and_finite_delta_x_against_reference():
    p=pair()
    with torch.no_grad():
        p.up.mul_(torch.logspace(-2,2,7,dtype=p.up.dtype)[:,None])
    u,d,root=regular_canonicalize(p.up,p.down,dtype=torch.float64)
    a=torch.stack((root[:,None]*p.up.grad,p.down.grad.T/root[:,None]))
    oracle=reference(u,d,a)
    before_u,before_d=p.up.detach().clone(),p.down.detach().T.clone()
    opt=QSO([p],beta=0,lr=.01,momentum_dtype=torch.float64,solver=SolverConfig(dtype=torch.float64,tolerance=1e-11))
    opt.step()
    close(opt.state[p.up]['momentum_up'],a[0],1e-13)
    close(opt.state[p.up]['momentum_down_t'],a[1],1e-13)
    eu=before_u-.01*root[:,None]*oracle.pair[0]
    ed=before_d-.01*oracle.pair[1]/root[:,None]
    close(p.up,eu,2e-7);close(p.down.T,ed,2e-7)
    outer=lambda d,u:torch.einsum('mi,mj->mij',d,u)
    measured=outer(p.down.detach().T,p.up.detach())-outer(before_d,before_u)
    expected=outer(ed,eu)-outer(before_d,before_u)
    assert float((measured-expected).norm()/expected.norm())<1e-5


def test_actual_swiglu_midtraining_gauge_reset():
    p=pair();q=pair()
    rng=torch.Generator().manual_seed(1001)
    x=torch.randn(16,3,generator=rng,dtype=torch.float64)
    gate=torch.randn(7,3,generator=rng,dtype=torch.float64)
    target=torch.randn(16,3,generator=rng,dtype=torch.float64)
    forward=lambda p:(F.silu(x@gate.T)*(x@p.up.T))@p.down.T
    opts=[QSO([v],momentum_dtype=torch.float64,solver=SolverConfig(dtype=torch.float64,tolerance=1e-11)) for v in (p,q)]
    c=torch.logspace(-5,5,7,dtype=torch.float64)
    for step in range(4):
        if step==2:
            with torch.no_grad():q.up.mul_(c[:,None]);q.down.div_(c)
        for v,opt in zip((p,q),opts):
            opt.zero_grad();F.mse_loss(forward(v),target).backward();opt.step()
        close(forward(p),forward(q),1e-8)
        close(opts[0].state[p.up]['momentum_up'],opts[1].state[q.up]['momentum_up'],1e-8)


def test_hvp_cached_factors_repeated_spectrum_and_independent_finite_difference(monkeypatch):
    u=torch.tensor([[1.,0.],[0.,1.],[1.,0.]],dtype=torch.float64)
    a=torch.tensor([[[1.,0.],[0.,1.],[0.,0.]]]*2,dtype=u.dtype)
    problem=SmoothDual(u,u,a)
    lam=torch.zeros(3,dtype=u.dtype);v=torch.tensor([.2,-.3,.4],dtype=u.dtype)
    ev=problem.evaluate(lam)
    original=torch.linalg.svd
    def forbidden(*args,**kw):raise AssertionError('HVP must reuse SVD')
    monkeypatch.setattr(torch.linalg,'svd',forbidden)
    hv=problem.hvp(ev,v)
    monkeypatch.setattr(torch.linalg,'svd',original)
    fd=(problem.evaluate(lam+1e-5*v).gradient-problem.evaluate(lam-1e-5*v).gradient)/2e-5
    close(hv,fd)


def test_pair_validation_and_unsupported_parameters_remain_separate():
    p=pair()
    with pytest.raises(ValueError,match='named'):QSO([(p.up,p.down)])
    with pytest.raises(ValueError,match='duplicate'):QSO([p,p])
    with pytest.raises(ValueError,match='only one'):QSO([p,SwiGLUPair('other',p.up,p.down)])
    gate=torch.nn.Parameter(torch.ones(3,dtype=torch.float64))
    opt=QSO([p]); adam=torch.optim.AdamW([gate],lr=.01)
    before=gate.clone();opt.step()
    assert torch.equal(before,gate)
    gate.grad=torch.ones_like(gate);adam.step()
    assert not torch.equal(before,gate)
    with pytest.raises(ValueError,match='topology'):opt.add_param_group(dict(params=[gate]))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_native_fp32_random_case_requires_observable_rank_boundary_fallback(device):
    # fp32 random draws differ from fp64 draws: this case has a deficient
    # optimal residual, as independently established in job 28894.
    u,d,a=random_example(901,dtype=torch.float32)
    oracle=reference(u,d,a)
    spectrum=torch.linalg.svdvals(a.double()-adjoint(u.double(),d.double(),oracle.lam))
    assert float(spectrum.min()) < 1e-8
    r=solve_coupled(u.to(device),d.to(device),a.to(device))
    assert r.converged and r.fallback
    assert r.reason in ('line_search_failed','ill_conditioned_residual','gap_stagnation')
    assert 'primary_only' in r.secondary_selection
    assert r.metrics['normalized_gap'] < 3e-5
    close(r.pair.cpu(),oracle.pair,3e-8)


@pytest.mark.parametrize('scale', [1e-12, 1e-200])
def test_tiny_nonzero_intrinsic_objective_is_not_truncated(scale):
    u,d,a=random_example(902)
    r=solve_coupled(u,d,scale*a,config=SolverConfig(dtype=torch.float64,tolerance=1e-8,rcond_guard=1e-8))
    oracle=reference(u,d,a)
    assert r.converged and r.pair.norm()>1
    close(r.pair,oracle.pair,2e-4)


def test_update_cast_errors_visible_for_bf16_and_no_fp64_invariance_claim():
    p=pair(dtype=torch.bfloat16)
    opt=QSO([p],lr=.05)
    opt.step()
    metrics=opt.last_diagnostics['block']
    assert 1e-5 < metrics['cast_direction_relative_error'] < .01
    assert metrics['cast_horizontal_residual'] > 1e-8
    assert metrics['returned_update_dtype']=='torch.bfloat16'


def test_invalid_inputs_and_missing_gradient_reject_without_state_changes():
    p=pair(); opt=QSO([p]);p.down.grad=None
    with pytest.raises(ValueError,match='both paired'):opt.step()
    assert not opt.state
    p.down.grad=torch.zeros_like(p.down);p.up.grad[0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):opt.step()
    assert not opt.state
    with pytest.raises(ValueError,match='dtype'):SolverConfig(dtype=torch.bfloat16)


def test_reference_budget_exhaustion_cannot_be_accepted():
    u,d,a=random_example(902)
    r=solve_coupled(u,d,a,config=SolverConfig(dtype=torch.float64,max_iterations=0,reference_max_iterations=1))
    assert r.fallback and not r.converged
    assert r.metrics['normalized_gap'] > 1e-8
