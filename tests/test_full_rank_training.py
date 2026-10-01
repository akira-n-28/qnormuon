"""Research training plumbing: production stays frozen and failures are atomic."""
import copy
from unittest.mock import patch
import pytest
import torch
import qnormuon.optimizer as om
import qnormuon.coupled_solver as cs
from qnormuon import QuotientSpectralOptimizer, SwiGLUPair
from experiments.full_rank_training import (research_boundary, AdmissionFailure,
    compact_metrics, POLICY, online_gate)
from experiments.full_rank_admission import solve_full_rank
from benchmarks.lr_study import exact_tree


def setup():
    torch.manual_seed(6)
    pairs=[]
    for i in range(2):
        u=torch.nn.Parameter(torch.randn(5,2))
        d=torch.nn.Parameter(torch.randn(2,5))
        u.grad=torch.randn_like(u);d.grad=torch.randn_like(d)
        pairs.append(SwiGLUPair(str(i),u,d))
    return pairs,QuotientSpectralOptimizer(pairs)


def test_research_boundary_checkpoint_and_restore():
    pairs,opt=setup();original=om.solve_coupled
    with research_boundary(opt,profile=False):opt.step()
    assert om.solve_coupled is original
    assert all(r['selection_semantics']==POLICY['name'] and not r['selection_certified'] for r in opt.last_diagnostics.values())
    state=copy.deepcopy(opt.state_dict())
    other_pairs=[SwiGLUPair(p.name,torch.nn.Parameter(p.up.detach().clone()),torch.nn.Parameter(p.down.detach().clone())) for p in pairs]
    other=QuotientSpectralOptimizer(other_pairs);other.load_state_dict(state)
    for p,q in zip(pairs,other_pairs):
        q.up.grad=p.up.grad.clone();q.down.grad=p.down.grad.clone()
    with research_boundary(opt,profile=False):opt.step()
    with research_boundary(other,profile=False):other.step()
    for p,q in zip(pairs,other_pairs):
        assert torch.equal(p.up,q.up) and torch.equal(p.down,q.down)
    exact_tree(opt.state_dict(),other.state_dict())


def test_research_pair_failure_commits_nothing():
    pairs,opt=setup();weights=[p.detach().clone() for pair in pairs for p in (pair.up,pair.down)]
    calls=0
    def fake(*args,**kwargs):
        nonlocal calls
        calls+=1
        result=solve_full_rank(*args,**kwargs)
        if calls==2:result.converged=False;result.pair=None;result.reason='injected_rank_ambiguity'
        return result
    with patch('experiments.full_rank_training.solve_full_rank',fake),research_boundary(opt,profile=False):
        with pytest.raises(AdmissionFailure,match='injected_rank_ambiguity'):opt.step()
    assert not opt.state
    for value,old in zip((p for pair in pairs for p in (pair.up,pair.down)),weights):assert torch.equal(value,old)


def test_online_gate_does_not_use_losses_or_quality():
    row=dict(step=127,converged=True,reference_used=False,newton_iterations=4,
             cg_iterations=15,line_trials=4,max_cg_queries_per_action=5)
    assert online_gate([row])['passed']
    with pytest.raises(AdmissionFailure,match='work_explosion'):online_gate([dict(row,line_trials=24)])
    with pytest.raises(AdmissionFailure):online_gate([dict(row,reference_used=True)])


def test_rank_invalid_trial_stops_even_after_recovery():
    pairs,opt=setup()
    def fake(*args,**kw):
        r=solve_full_rank(*args,**kw)
        r.evaluations.append(dict(eligible=False,reason='rank_ambiguous_or_deficient'))
        return r
    with patch('experiments.full_rank_training.solve_full_rank',fake),research_boundary(opt,profile=False):
        with pytest.raises(AdmissionFailure,match='trajectory_invalid_evaluation'):opt.step()
    assert not opt.state
