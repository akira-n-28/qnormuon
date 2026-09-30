"""Research-only one-sided actions; full-SVD decisions are never a branch oracle."""
import pytest
import torch
from unittest.mock import patch

from experiments.smooth_qdwh_one_sided import (
    ControlledSolver, armijo_action, certificate_action, solve_one_sided,
)
from experiments.smooth_qdwh_decisions import (
    armijo_interval, certificate_interval_decision, cg_stop_certified,
    interval_sign,
)


@pytest.mark.parametrize("decision", ["qdwh_certificate_ambiguous", "reject"])
def test_ambiguous_certificate_continues_without_branch_oracle(decision):
    assert certificate_action(decision) == "continue"


def test_proved_certificate_only_accepts():
    assert certificate_action("accept") == "accept"
    assert certificate_action(certificate_interval_decision(
        (1.000030001, 1.000030002), (1., 1.),
        rcond_lower=2e-4, feasible=True)) == "continue"
    assert certificate_action(certificate_interval_decision(
        (1.00002, 1.00002), (1., 1.),
        rcond_lower=2e-4, feasible=True)) == "accept"


def test_ambiguous_armijo_backtracks_without_branch_oracle():
    decision = armijo_interval((1., 1.0000001), (0.9999, 1.0001),
        (-1., -1.), alpha=1.)
    assert decision == "qdwh_armijo_ambiguous"
    assert armijo_action(decision) == "backtrack"
    assert armijo_action("reject") == "backtrack"
    assert armijo_action("accept") == "accept"


@pytest.mark.parametrize("value,radius", [(0.,0.), (1e-12,1e-12),(-1e-12,1e-12)])
def test_marginal_curvature_or_descent_never_authorizes_action(value,radius):
    assert interval_sign(value,radius) == "uncertain"


def test_guard_and_gap_boundaries_are_not_accepted():
    for guard in (1e-4, 0.999e-4):
        assert certificate_action(certificate_interval_decision(
            (1.,1.),(1.,1.),rcond_lower=guard,feasible=True)) == "continue"
    assert certificate_action(certificate_interval_decision(
        (1.00003001,1.00003001),(1.,1.),rcond_lower=2e-4,
        feasible=True)) == "continue"
    assert certificate_action(certificate_interval_decision(
        (.9999999998,1.),(1.,1.),rcond_lower=2e-4,
        feasible=True)) == "continue"
    assert certificate_action(certificate_interval_decision(
        (1.,1.),(1.,1.),rcond_lower=2e-4,
        feasible=False)) == "continue"


def test_armijo_equality_requires_a_one_sided_proof():
    assert armijo_action(armijo_interval((1.,1.),(.9999,.9999),
        (-1.,-1.),alpha=1.,c=1e-4)) == "accept"
    assert armijo_action(armijo_interval((1.,1.0000001),
        (.9998999,.9999001),(-1.,-1.),alpha=1.,c=1e-4)) == "backtrack"
    assert not cg_stop_certified(.11,.1,damping=1e-6)


def test_qdwh_controlled_certified_initial_point_never_uses_svd():
    from qnormuon.coupled_solver import SmoothDual, SolverConfig
    gen=torch.Generator().manual_seed(215)
    u,_=torch.linalg.qr(torch.randn(16,4,dtype=torch.float64,generator=gen))
    a=torch.stack((u,u))
    with patch.object(SmoothDual,"evaluate",side_effect=AssertionError("hidden SVD branch oracle")):
        result=solve_one_sided(u,u,a,config=SolverConfig(fallback=False))
    assert result.converged, result.reason
    assert result.qdwh_evaluations == 1
    assert result.svd_evaluations == 0
    assert result.actions["certificate_qdwh_accept"] == 1


def test_newton_budget_requests_explicit_svd_rescue():
    from qnormuon.coupled_solver import SolverConfig
    gen=torch.Generator().manual_seed(215)
    u,_=torch.linalg.qr(torch.randn(16,4,dtype=torch.float64,generator=gen))
    a=torch.stack((u,u))
    original=ControlledSolver.certificate
    def reject_once(self,*args,**kwargs):
        p,m,_,_=original(self,*args,**kwargs)
        return p,m,False,"qdwh_certificate_ambiguous"
    with patch.object(ControlledSolver,"certificate",reject_once):
        result=solve_one_sided(u,u,a,config=SolverConfig(
            fallback=False,max_iterations=0))
    assert result.reason == "certified_svd_rescue"
    assert result.fallbacks["full_svd_solver_rescue"] == 1
    assert result.svd_evaluations > 0


def test_line_search_exhaustion_requests_explicit_svd_rescue():
    from qnormuon.coupled_solver import SolverConfig
    gen=torch.Generator().manual_seed(690)
    u=torch.randn(16,4,dtype=torch.float64,generator=gen)
    d=torch.randn(16,4,dtype=torch.float64,generator=gen)
    d=d*(u.norm(dim=1)/d.norm(dim=1))[:,None]
    a=torch.randn(2,16,4,dtype=torch.float64,generator=gen)
    with patch.object(ControlledSolver,"armijo",return_value="backtrack"):
        result=solve_one_sided(u,d,a,config=SolverConfig(
            fallback=False,max_iterations=3))
    assert result.line_trials == 24
    assert result.fallbacks["qdwh_line_search_exhausted"] == 1
    assert result.fallbacks["full_svd_solver_rescue"] == 1
