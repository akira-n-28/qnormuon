"""Scoped research admission at the unchanged production optimizer boundary.

No production source/default/state format is modified. All pairs still validate
before the existing optimizer commits any parameters or canonical EMA state.
The context is single-threaded research plumbing, never a production backend.
"""
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch
import torch
import qnormuon.optimizer as om
import qnormuon.coupled_solver as cs
from experiments.full_rank_admission import solve_full_rank, final_admission

POLICY = dict(name="primary_epsilon_lmo_full_rank", version=1,
              selection_certified=False, numerical_model="conditional_normal_fp64_gamma",
              production_v0_unchanged=True)


class AdmissionFailure(RuntimeError):
    pass


def compact_metrics(result):
    """Reuse matching factors/posteriors; no decompositions just for logging."""
    metrics = dict(result.metrics)
    rank = metrics.get("rank", {})
    post = metrics.get("posterior", {})
    evaluations = [e for e in result.evaluations if e.get("rank") and e.get("eligible", True)]
    metrics.update(
        sigma_min=rank.get("sigma_min"), alpha_lower=rank.get("alpha_lower"),
        eta=rank.get("uncertainty"), rcond_sides=rank.get("rcond"),
        sigma_min_over_eta=[s/e if e else None for s,e in zip(
            rank.get("sigma_min", []), rank.get("uncertainty", []))],
        initial_rcond_sides=evaluations[0]["rank"]["rcond"] if evaluations else None,
        initial_rank=evaluations[0]["rank"] if evaluations else None,
        minimum_evaluated_rcond=min((min(e["rank"]["rcond"]) for e in evaluations), default=None),
        below_old_guard=bool(rank and min(rank["rcond"]) <= 1e-4),
        any_evaluation_below_old_guard=any(min(e["rank"]["rcond"]) <= 1e-4 for e in evaluations),
        conservative_normalized_gap=post.get("normalized_gap_upper"),
        G_upper=post.get("gap_upper"), dual_lower=post.get("dual_lower"),
        feasible_proxy_distance=post.get("feasibility_distance_upper"),
        cg_termination=[a["cg_termination"] for a in result.actions],
        max_cg_queries_per_action=max((len(a["queries"]) for a in result.actions), default=0),
        minimum_used_curvature=min((q["curvature"] for a in result.actions for q in a["queries"]), default=None),
        maximum_hvp_norm=max((q["hvp_norm"] for a in result.actions for q in a["queries"]), default=None),
        maximum_slope=max((a["slope"] for a in result.actions), default=None),
        smooth_evaluations=len(result.evaluations),
        solver_dtype="torch.float64", certification_dtype="torch.float64",
        returned_direction_dtype="torch.float64", fallback_used=False,
        fallback_reason=None, fallback_backend=None, reference_iterations=0,
        gap_tolerance=3e-5, converged=result.converged,
        svd_evaluations=result.counts.svd_matrices,
        selection_semantics=result.selection_semantics,
        selection_certified=result.selection_certified,
        reference_used=result.reference_used)
    # Keep compact top-level quantities; detailed history is saved only on failure.
    metrics.pop("posterior", None)
    metrics.pop("rank", None)
    return metrics


def require_return(result):
    if not result.converged or result.pair is None or result.reference_used:
        raise AdmissionFailure(result.reason)
    if not bool(torch.isfinite(result.pair).all()) or not bool(torch.isfinite(result.lam).all()):
        raise AdmissionFailure("nonfinite_research_return")
    if result.selection_semantics == "exact_zero":
        if bool(torch.count_nonzero(result.pair)) or not result.selection_certified:
            raise AdmissionFailure("invalid_exact_zero_selection")
    elif (result.selection_semantics != POLICY["name"] or result.selection_certified
          or not final_admission(result.metrics["posterior"], result.metrics)):
        raise AdmissionFailure("final_research_certificate_failed")


@contextmanager
def research_boundary(optimizer, callback=None, before_solve=None, profile=True):
    """Replace only this explicitly scoped call boundary; restore on all exits.

    callback receives the complete result even when it is unsuccessful, allowing
    a compact fixed-pair fixture to be written before the optimizer raises.
    """
    index = 0
    def forbidden(*args, **kwargs):
        raise AdmissionFailure("hidden_cpu_reference_forbidden")
    def solve(u, d, a, *, config, initial_lambda=None):
        nonlocal index
        if index >= len(optimizer.pairs):
            raise AdmissionFailure("unexpected_pair_traversal")
        pair = optimizer.pairs[index]
        index += 1
        if before_solve:
            before_solve(pair, u, d, a, initial_lambda)
        result = solve_full_rank(u, d, a, initial_lambda=initial_lambda,
                                config=replace(config, fallback=False), profile=profile)
        # The trajectory's stricter fail-closed rule stops even if an invalid
        # trial could otherwise be backtracked by the isolated fixture solver.
        invalid = next((e for e in result.evaluations if e.get("eligible") is False), None)
        if invalid is not None:
            result.converged = False
            result.pair = None
            result.reason = "trajectory_invalid_evaluation: " + invalid["reason"]
        if callback:
            callback(pair, u, d, a, initial_lambda, result)
        require_return(result)
        metrics = compact_metrics(result)
        return cs.SolverResult(result.pair, result.lam, "research_full_rank", result.iterations,
            True, False, result.reason, metrics, result.counts, result.seconds,
            metrics.get("minimum_evaluated_rcond") or 0., result.selection_semantics)
    with patch.object(om, "solve_coupled", solve), patch.object(cs, "_reference", forbidden):
        yield


def online_gate(rows):
    """Predeclared operational budget gate, not fitted to trajectory quality.

    Use existing Newton/CG/24-trial budgets and the existing 12-iterate stagnation
    window. Ordinary multi-iteration work is allowed; sustained last-32-step
    median >=12 Newton actions or >=24 total line trials stops this experiment.
    No certificate or mathematical admission threshold is changed.
    """
    import statistics
    if not rows or any(not r["converged"] or r["reference_used"] for r in rows):
        raise AdmissionFailure("online_gate_uncertified_or_reference")
    recent = [r for r in rows if r["step"] >= 96]
    if not recent:
        raise AdmissionFailure("online_gate_missing_recent_pairs")
    stats = dict(newton_median=statistics.median(r["newton_iterations"] for r in recent),
                 cg_median=statistics.median(r["cg_iterations"] for r in recent),
                 line_median=statistics.median(r["line_trials"] for r in recent))
    if stats["newton_median"] >= 12 or stats["line_median"] >= 24:
        raise AdmissionFailure("online_gate_sustained_solver_work_explosion")
    if any(r["max_cg_queries_per_action"] > 30 for r in rows):
        raise AdmissionFailure("online_gate_cg_budget_violation")
    return dict(passed=True, **stats)
