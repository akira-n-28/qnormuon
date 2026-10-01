from dataclasses import asdict
import torch
from unittest.mock import patch
import pytest
import qnormuon.coupled_solver as cs
from experiments.stage_d_forensics import Observer, ReferenceRequested


def problem():
    generator=torch.Generator().manual_seed(121)
    u=torch.randn(12,4,generator=generator,dtype=torch.float64)
    d=torch.randn(12,4,generator=generator,dtype=torch.float64)
    d=d/d.norm(dim=1)[:,None]*u.norm(dim=1)[:,None]
    a=torch.randn(2,12,4,generator=generator,dtype=torch.float64)
    return u,d,a


def test_observer_preserves_solver_trajectory():
    u,d,a=problem();config=cs.SolverConfig()
    expected=cs.solve_coupled(u,d,a,config=config)
    observer=Observer(u,d,a,None,dict(pair='test',training_step=2))
    actual=observer.solve(asdict(config))
    assert torch.equal(actual.lam,expected.lam)
    assert torch.equal(actual.pair,expected.pair)
    assert actual.history==expected.history
    assert actual.counts==expected.counts
    assert observer.newton and observer.newton[0]['cg']
    assert observer.normalization['intrinsic_magnitude']>0


def test_reference_reason_captured_before_reference_execution():
    u,d,a=problem();config=cs.SolverConfig(max_iterations=0)
    observer=Observer(u,d,a,None,dict(pair='test',training_step=7))
    with patch('experiments.dual_solver.reference',side_effect=AssertionError('must not execute')):
        with pytest.raises(ReferenceRequested,match='iteration_budget'):
            observer.solve(asdict(config))
    event=observer.event
    assert event['reason']=='iteration_budget' and event['pair']=='test'
    assert event['training_step']==7 and len(event['history'])==1
    assert event['current_rcond']>0
    assert len(event['current_lambda'])==12
    assert observer.inputs['current_lambda'].dtype==torch.float64


def test_observer_hooks_restore_on_capture():
    u,d,a=problem();original=cs._reference
    with pytest.raises(ReferenceRequested):
        Observer(u,d,a,None,{}).solve(asdict(cs.SolverConfig(max_iterations=0)))
    assert cs._reference is original
