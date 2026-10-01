"""Research-only inexact QDWH inner solve with authoritative SVD output.

The dual/LMO is unchanged, but intermediate floating-point decisions are raw
QDWH decisions, not posterior-certified SVD-equivalent decisions. No QDWH
pair can be returned: a successful proposal is rebuilt at the SAME original
lambda by production SmoothDual (full fp64 SVD), production feasible recovery
and production certificate. Rejected proposals enter explicit production-SVD
rescue. Production configuration, checkpoints and code are not modified.
"""
from dataclasses import dataclass, replace
import math
import time
from unittest.mock import patch

import torch
from experiments.smooth_polar_alternative import diagnose, polar_derivative
from qnormuon import coupled_solver as cs


class InnerFailure(RuntimeError):
    pass


@dataclass
class HybridResult:
    result: cs.SolverResult
    stats: dict


def safe(metrics, config):
    """The five unchanged conditions; nonfinite diagnostics cannot authorize."""
    keys = ("normalized_gap", "signed_normalized_gap",
            "normalized_horizontal_residual", "spectral_excess", "residual_rcond")
    return (all(math.isfinite(metrics[k]) for k in keys)
            and cs.accepted(metrics, min(config.tolerance or 3e-5, 3e-5))
            and metrics["residual_rcond"] > max(config.rcond_guard or 1e-4, 1e-4))


def verify(u, d, a, lam, config):
    """Rebuild both residual polars at exactly this ORIGINAL multiplier.

    Magnitude=1 here: production SmoothDual sees the original uncentered
    residual itself, so its spectrum is directly the original spectrum.
    No QDWH polar/spectrum enters the authoritative recovery or certificate.
    """
    started = time.perf_counter()
    counts = cs.Counts()
    ev = cs.SmoothDual(u, d, a, counts).evaluate(lam)
    cached = cs.CachedDualSpectrum(lam, ev.pair, ev.singular, 1.)
    pair, metrics = cs.certificate(u, d, a, lam, ev.pair, counts,
        cached_dual=cached, primal_norm_backend=config.primal_norm_backend)
    ok = safe(metrics, config) and ev.rcond > (config.rcond_guard or 1e-4)
    return cs.SolverResult(pair, lam, "newton", 0, ok, False,
        "certified_gap" if ok else "final_svd_verification_rejected", metrics,
        counts, time.perf_counter()-started, ev.rcond,
        "smooth_primary_unique_in_exact_limit" if ok else "uncertified_primary")


@torch.no_grad()
def solve_hybrid(u, d, a, *, config=None, initial_lambda=None,
                 inner_iterations=2, rescue_policy="continue", evaluation_budget=7,
                 profile=True):
    config = config or cs.SolverConfig()
    if config.dtype != torch.float64 or inner_iterations not in (2, 3):
        raise ValueError("fp64 and a fixed two/three-Newton inner budget required")
    if rescue_policy not in ("continue", "restart") or evaluation_budget < 1:
        raise ValueError("explicit rescue policy and positive evaluation budget required")
    if config.primal_norm_backend != "gram_upper":
        raise ValueError("this study preserves production gram_upper recovery")
    with torch.autocast(device_type=u.device.type, enabled=False):
        u, d, a = (v.double() for v in (u, d, a))
        cs._validate(u, d, a)
        def clock():
            if profile and u.is_cuda:
                torch.cuda.synchronize(u.device)
            return time.perf_counter()
        started = clock()
        stats = dict(qdwh_evaluations=0, final_svd_verifications=0,
            failed_final_verifications=0, svd_rescue_solves=0,
            structural_rejects=0, inner_success=False, inner_iterations=0,
            inner_line_trials=0, inner_hvps=0, inner_seconds=0.,
            verification_seconds=0., rescue_seconds=0., rescue_iterations=0,
            rescue_line_trials=0, rescue_svd_evaluations=0,
            inner_svd_matrices=0, fallback_reason=None, rescue_policy=rescue_policy,
            inner_newton_budget=inner_iterations, evaluation_budget=evaluation_budget)
        parent = cs.SmoothDual
        kernel_seconds = [0.]
        inner_counts = cs.Counts()
        class RawQDWH(parent):
            def evaluate(self, z):
                if stats["qdwh_evaluations"] >= evaluation_budget:
                    raise InnerFailure("qdwh_evaluation_budget")
                stats["qdwh_evaluations"] += 1
                b = self.a - cs.adjoint(self.u, self.d, z)
                start = clock()
                factors = [diagnose(side, guard=config.rcond_guard or 1e-4) for side in b]
                kernel_seconds[0] += clock()-start
                bad = next((r.reason for r in factors if not r.accepted), None)
                if bad:
                    stats["structural_rejects"] += 1
                    raise InnerFailure("qdwh_structural_" + bad)
                self.counts.polar_matrices += 2
                pair = torch.stack([r.polar for r in factors])
                singular = torch.stack([r.singular for r in factors])
                ev = cs.Evaluation(float(singular.sum()),
                    -cs.horizontal_residual(self.u, self.d, pair), pair, None,
                    singular, None, float((singular[:,-1]/singular[:,0]).min()), z)
                # Current HVPs use only the current accepted evaluation; retain
                # factors by evaluation identity across rejected trial calls.
                if not hasattr(self, "factors"):
                    self.factors = {}
                self.factors[id(ev)] = factors
                return ev

            def polar_derivative(self, ev, e):
                return torch.stack([polar_derivative(r, side)
                                    for r, side in zip(self.factors[id(ev)], e)])

            def __init__(self, eu, ed, objective, counts=None):
                super().__init__(eu, ed, objective, counts)
                nonlocal inner_counts
                inner_counts = self.counts

        inner = None
        start = clock()
        # Only the numerical evaluator/derivative changes. Production damping,
        # truncated CG and ordinary scalar Armijo semantics are reused intact.
        try:
            with patch.object(cs, "SmoothDual", RawQDWH):
                inner = cs.solve_coupled(u, d, a,
                    config=replace(config, max_iterations=inner_iterations, fallback=False),
                    initial_lambda=initial_lambda)
        except InnerFailure as error:
            stats["fallback_reason"] = str(error)
        except (torch.linalg.LinAlgError, OverflowError) as error:
            stats["structural_rejects"] += 1
            stats["fallback_reason"] = "qdwh_arithmetic_failure: " + str(error)
        stats["inner_seconds"] = clock()-start
        stats["qdwh_decomposition_seconds"] = kernel_seconds[0]
        stats["inner_line_trials"] = inner_counts.line_trials
        stats["inner_hvps"] = inner_counts.hvp
        stats["inner_svd_matrices"] = inner_counts.svd_matrices
        answer = None
        if inner is not None:
            stats["inner_iterations"] = inner.iterations
            stats["inner_success"] = inner.converged
            if inner.converged:
                # Outside the patch scope: this necessarily uses full SVD.
                start = clock()
                answer = verify(u, d, a, inner.lam, config)
                stats["verification_seconds"] = clock()-start
                stats["final_svd_verifications"] = 1
                stats["failed_final_verifications"] = int(not answer.converged)
                if not answer.converged:
                    stats["fallback_reason"] = "final_svd_verification_rejected"
            else:
                stats["fallback_reason"] = "qdwh_inner_" + inner.reason
        if answer is None or not answer.converged:
            # A: resume from the candidate; B: restart from the original warm
            # lambda. On an interrupted structural path no valid completed
            # candidate exists, so both policies use the original warm lambda.
            rescue_lambda = (inner.lam if inner is not None and rescue_policy == "continue"
                             else initial_lambda)
            start = clock()
            rescue = cs.solve_coupled(u, d, a, config=config, initial_lambda=rescue_lambda)
            stats["rescue_seconds"] = clock()-start
            stats["svd_rescue_solves"] = 1
            stats["rescue_iterations"] = rescue.iterations
            stats["rescue_line_trials"] = rescue.counts.line_trials
            stats["rescue_svd_evaluations"] = rescue.counts.svd_matrices // 2
            if answer is not None:
                # Retain the rejected verifier's decomposition in work counts.
                rescue.counts.svd_matrices += answer.counts.svd_matrices
                rescue.counts.polar_matrices += answer.counts.polar_matrices
            answer = rescue
        # All outputs come from full-SVD verification or ordinary production
        # SVD rescue. Reference fallback is reported separately. A failed rescue
        # cannot become a usable optimizer direction, even with a small gap.
        answer.converged = answer.converged and safe(answer.metrics, config)
        if not answer.converged:
            answer.reason = "hybrid_uncertified: " + answer.reason
        answer.iterations += stats["inner_iterations"]
        answer.counts.hvp += inner_counts.hvp
        answer.counts.line_trials += inner_counts.line_trials
        answer.counts.polar_matrices += inner_counts.polar_matrices
        answer.counts.svd_matrices += inner_counts.svd_matrices
        stats["total_svd_evaluations"] = answer.counts.svd_matrices // 2
        stats["total_decomposition_evaluations"] = stats["qdwh_evaluations"] + stats["total_svd_evaluations"]
        stats["cpu_admm_fallback"] = answer.fallback
        stats["output_authority"] = "production_svd_rescue" if stats["svd_rescue_solves"] else "production_svd_verifier"
        answer.metrics.update(newton_iterations=answer.iterations,
            cg_iterations=answer.counts.hvp, svd_evaluations=answer.counts.svd_matrices,
            fallback_used=answer.fallback, fallback_reason=answer.reason if answer.fallback else None,
            solver_dtype="torch.float64", certification_dtype="torch.float64",
            returned_direction_dtype=str(answer.pair.dtype), gap_tolerance=config.tolerance or 3e-5,
            primal_dual_gap=answer.metrics["signed_gap"], absolute_gap=abs(answer.metrics["signed_gap"]),
            minimum_smooth_rcond=answer.min_accepted_rcond, converged=answer.converged,
            hybrid=dict(stats))
        answer.seconds = clock()-started
        stats["seconds"] = answer.seconds
        return HybridResult(answer, stats)
