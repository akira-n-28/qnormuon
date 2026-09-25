"""Warm-started K=I dual research solvers; no production integration.

The objective is the SUM of two nuclear norms. Smooth algebra uses the input
dtype. Feasible recovery/certification and explicit ADMM fallback use float64.
SVD counts count individual matrices, including certification and line search.
Fallback certifies primary optimality, not the minimum-norm optimal face.
"""
from dataclasses import dataclass, field
import math
import time

import torch

from experiments.horizontal_spectral import (
    _validate, adjoint, horizontal_project, horizontal_residual, multipliers,
    solve_lmo,
)


@dataclass
class Counts:
    svd_matrices: int = 0
    polar_matrices: int = 0
    hvp: int = 0
    line_trials: int = 0
    reference_iterations: int = 0


@dataclass
class Evaluation:
    value: float
    gradient: torch.Tensor
    pair: torch.Tensor
    left: torch.Tensor
    singular: torch.Tensor
    right: torch.Tensor
    rcond: float


class SmoothDual:
    """Full-rank values/derivatives; no singular values are discarded."""
    def __init__(self, u, d, a, counts=None):
        self.u, self.d, self.a = u, d, a
        self.counts = counts if counts is not None else Counts()

    def evaluate(self, lam):
        b = self.a - adjoint(self.u, self.d, lam)
        q, sigma, vt = torch.linalg.svd(b, full_matrices=False)
        self.counts.svd_matrices += 2
        self.counts.polar_matrices += 2
        p = q @ vt
        ratio = sigma[:, -1] / sigma[:, 0].clamp_min(torch.finfo(sigma.dtype).tiny)
        return Evaluation(float(sigma.sum()), -horizontal_residual(self.u, self.d, p),
                          p, q, sigma, vt, float(ratio.min()))

    def polar_derivative(self, evaluation, e):
        q, s, vt = evaluation.left, evaluation.singular, evaluation.right
        if float(s.min()) <= 0:
            raise ValueError("polar derivative requires strictly positive singular values")
        ev = e @ vt.transpose(-2, -1)
        f = q.transpose(-2, -1) @ ev
        omega = (f - f.transpose(-2, -1)) / (s.unsqueeze(-1) + s.unsqueeze(-2))
        normal = ev - q @ f
        return (q @ omega + normal / s.unsqueeze(-2)) @ vt

    def hvp(self, evaluation, v):
        self.counts.hvp += 1
        e = -adjoint(self.u, self.d, v)
        return -horizontal_residual(self.u, self.d, self.polar_derivative(evaluation, e))


@dataclass
class SolverResult:
    pair: torch.Tensor
    lam: torch.Tensor
    method: str
    iterations: int
    converged: bool
    fallback: bool
    reason: str
    metrics: dict
    counts: Counts
    seconds: float
    min_accepted_rcond: float
    secondary_selection: str
    history: list = field(default_factory=list)


def certificate(u, d, a, lam, candidate, counts):
    """Float64 projection/radial feasible recovery and primal-dual diagnostics.

    This is an exact-arithmetic certificate evaluated in floating point, not
    interval arithmetic. The signed gap and equality residual remain visible.
    """
    u, d, a, lam, candidate = (v.double() for v in (u, d, a, lam, candidate))
    p = horizontal_project(u, d, candidate)
    spectra = torch.linalg.svdvals(p)
    counts.svd_matrices += 2
    scale = max(1., float(spectra.max()))
    p = p / scale
    if float((a * p).sum()) < 0:
        p = -p  # central symmetry supplies a nonnegative lower bound
    dual = float(torch.linalg.svdvals(a - adjoint(u, d, lam)).sum())
    counts.svd_matrices += 2
    primal = float((a * p).sum())
    gap = dual - primal
    denom = max(abs(primal), abs(dual), torch.finfo(torch.float64).tiny)
    residual = horizontal_residual(u, d, p)
    weights = (u.square() + d.square()).sum(1).sqrt()
    relative_h = residual.abs() / weights.clamp_min(torch.finfo(torch.float64).tiny)
    return p, {
        "horizontal_residual": float(residual.abs().max()),
        "normalized_horizontal_residual": float(relative_h.max()),
        "spectral_norms": (spectra[:, 0] / scale).tolist(),
        "spectral_excess": max(0., float(spectra.max()) / scale - 1.),
        "primal_objective": primal, "dual_objective": dual,
        "signed_gap": gap, "normalized_gap": max(0., gap) / denom,
        "signed_normalized_gap": gap / denom,
    }


def accepted(metrics, tolerance):
    return (metrics["normalized_gap"] <= tolerance
            and metrics["signed_normalized_gap"] >= -1e-10
            and metrics["normalized_horizontal_residual"] <= 1e-10
            and metrics["spectral_excess"] <= 1e-12)


def reference(u, d, a, *, tolerance=1e-11, max_iterations=20000):
    """Existing ADMM oracle on an equivalent, centered cotangent representative.

    Float64 is explicit even for float32 inputs. Counts follow the existing
    oracle's exact control flow (two SVDs/iteration and four/diagnostic check).
    """
    started = time.perf_counter()
    u, d, a = u.double(), d.double(), a.double()
    beta = multipliers(u, d, a)
    centered = a - adjoint(u, d, beta)
    # For pair-Frobenius-normalized objectives, typical singular values scale
    # as n^(-1/2). Match rho to that scale; this changes convergence, not the LMO.
    r = solve_lmo(u, d, centered, tolerance=tolerance, max_iterations=max_iterations,
                  rho=1. / math.sqrt(a.shape[-1]))
    checks = r.iterations // 10 + int(r.iterations % 10 != 0)
    counts = Counts(svd_matrices=2 * r.iterations + 4 * checks,
                    reference_iterations=r.iterations)
    lam = beta + r.lam
    p, metrics = certificate(u, d, a, lam, r.pair, counts)
    return SolverResult(p, lam, "admm", r.iterations,
        r.converged and accepted(metrics, max(10 * tolerance, 1e-10)), False,
        "reference", metrics, counts, time.perf_counter() - started, 0.,
        "primary_only_unless_uniqueness_established")


def _lbfgs_direction(g, memory, initial):
    q, alphas = g.clone(), []
    for s, y in reversed(memory):
        alpha = (s @ q) / (s @ y)
        alphas.append(alpha)
        q = q - alpha * y
    if memory:
        s, y = memory[-1]
        initial = float((s @ y) / (y @ y))
    r = initial * q
    for (s, y), alpha in zip(memory, reversed(alphas)):
        r = r + s * (alpha - (y @ r) / (s @ y))
    return -r


def _newton_direction(problem, ev, curvature_scale, max_cg=30):
    """Damped truncated CG; only cached-SVD HVPs, never an m x m Hessian."""
    g = ev.gradient
    damping = 1e-6 * curvature_scale
    x = torch.zeros_like(g)
    residual = -g.clone()
    direction = residual.clone()
    squared = float(residual @ residual)
    target = min(.1, math.sqrt(max(float(g.norm()), 1e-16))) * math.sqrt(squared)
    for _ in range(min(max_cg, g.numel())):
        hv = problem.hvp(ev, direction) + damping * direction
        curvature = float(direction @ hv)
        if curvature <= 0 or not math.isfinite(curvature):
            break
        alpha = squared / curvature
        x = x + alpha * direction
        residual = residual - alpha * hv
        new_squared = float(residual @ residual)
        if math.sqrt(new_squared) <= target:
            break
        direction = residual + (new_squared / squared) * direction
        squared = new_squared
    return x if float(g @ x) < 0 else -g / curvature_scale


@torch.no_grad()
def solve_dual(u, d, a, *, method="newton", initial_lambda=None, tolerance=None,
               max_iterations=100, fallback=True, rcond_guard=None, max_cg=30):
    """GD, row-preconditioned GD, L-BFGS or matrix-free damped Newton-CG.

    Warm state is the original-coordinate multiplier, not internally scaled
    coordinates. L-BFGS memory is reset at each new objective. With fallback
    disabled, iteration-budget results are explicitly uncertified if needed.
    Returned pairs and certificates are float64; smooth work uses input dtype.
    """
    started = time.perf_counter()
    _validate(u, d, a)
    if method not in ("gd", "pgd", "lbfgs", "newton") or max_iterations < 0:
        raise ValueError("unknown method or negative iteration budget")
    dtype = a.dtype
    eps = torch.finfo(dtype).eps
    tolerance = (1e-8 if dtype == torch.float64 else 3e-5) if tolerance is None else tolerance
    guard = (1e-8 if dtype == torch.float64 else 1e-4) if rcond_guard is None else rcond_guard
    if tolerance <= 0 or not 0 < guard < 1:
        raise ValueError("positive tolerance and rcond guard in (0,1) required")
    counts = Counts()
    ud, dd, ad = u.double(), d.double(), a.double()
    beta = multipliers(ud, dd, ad)
    centered = ad - adjoint(ud, dd, beta)
    magnitude = float(centered.norm())
    if magnitude == 0:
        p, metrics = certificate(ud, dd, ad, beta, torch.zeros_like(ad), counts)
        return SolverResult(p, beta, method, 0, accepted(metrics, tolerance), False,
            "zero_cotangent", metrics, counts, time.perf_counter() - started, 0., "exact_zero")
    w = (ud.square() + dd.square()).sum(1)
    coord = torch.ones_like(w) if method == "gd" else torch.where(w > 0, w.rsqrt(), 0.)
    effective_u, effective_d = (ud * coord[:, None]).to(dtype), (dd * coord[:, None]).to(dtype)
    problem = SmoothDual(effective_u, effective_d, (centered / magnitude).to(dtype), counts)
    z = torch.zeros(u.shape[0], dtype=dtype)
    if initial_lambda is not None:
        if initial_lambda.shape != beta.shape or not torch.isfinite(initial_lambda).all():
            raise ValueError("finite multiplier of shape (m,) required")
        nonzero = coord != 0
        z[nonzero] = ((initial_lambda.double()[nonzero] - beta[nonzero]) /
                      (magnitude * coord[nonzero])).to(dtype)
    ev = problem.evaluate(z)
    min_rcond = ev.rcond
    memory, history = [], []
    step_seed = float(ev.singular.median()) / max(float((effective_u.square() + effective_d.square()).sum(1).max()), 1e-30)
    reason, iteration = "iteration_budget", 0
    best = None
    for iteration in range(max_iterations + 1):
        lam = beta + magnitude * coord * z.double()
        p, metrics = certificate(ud, dd, ad, lam, ev.pair, counts)
        history.append({"iteration": iteration, "rcond": ev.rcond,
                        "polar_horizontal_residual": float(horizontal_residual(u, d, ev.pair).abs().max()),
                        **metrics})
        if best is None or metrics["normalized_gap"] < best[2]["normalized_gap"]:
            best = (p, lam, metrics)
        # Check conditioning BEFORE accepting a tiny value gap: it does not
        # certify direction accuracy at a near-rank boundary.
        if ev.rcond <= guard or not torch.isfinite(ev.gradient).all():
            reason = "ill_conditioned_residual"
            break
        if magnitude <= 32 * torch.finfo(torch.float64).eps * float(ad.norm()):
            reason = "cancellation_dominated_cotangent"
            break
        if accepted(metrics, tolerance):
            reason = "certified_gap"
            break
        if iteration == max_iterations:
            break
        if len(history) > 12 and history[-1]["normalized_gap"] > .99 * history[-12]["normalized_gap"]:
            reason = "gap_stagnation"
            break
        g = ev.gradient
        curvature_scale = max(float((effective_u.square() + effective_d.square()).sum(1).max()), 1e-30) / float(ev.singular.min())
        if method in ("gd", "pgd"):
            direction, step = -g, step_seed
        elif method == "lbfgs":
            direction, step = _lbfgs_direction(g, memory, step_seed), 1.
        else:
            direction, step = _newton_direction(problem, ev, curvature_scale, max_cg), 1.
        slope = float(g @ direction)
        if not math.isfinite(slope) or slope >= 0:
            reason = "no_descent_direction"
            break
        found = False
        for _ in range(24):
            counts.line_trials += 1
            trial_z = z + step * direction
            trial = problem.evaluate(trial_z)
            # A rejected trial does not trigger truncation: backtrack into the
            # smooth region, or explicitly fall back if no trial is usable.
            rounding = 2 * eps * max(1., abs(ev.value))
            resolved_decrease = abs(step * slope) > rounding
            gradient_improves = float(trial.gradient.norm()) < float(g.norm())
            if (trial.rcond > guard and trial.value <= ev.value + 1e-4 * step * slope + rounding
                    and (resolved_decrease or gradient_improves)):
                found = True
                break
            step *= .5
        if not found:
            reason = "line_search_failed"
            break
        s, y = trial_z - z, trial.gradient - ev.gradient
        if float(s @ y) > 1e-10 * float(s.norm() * y.norm()):
            memory.append((s, y))
            memory = memory[-10:]
        z, ev = trial_z, trial
        min_rcond = min(min_rcond, ev.rcond)
        if method in ("gd", "pgd"):
            step_seed = min(2 * step, 1e12)
    p, lam, metrics = best
    did_fallback = reason != "certified_gap" and fallback
    selection = "smooth_primary_unique_in_exact_limit" if reason == "certified_gap" else "uncertified_primary"
    if did_fallback:
        oracle = reference(ud, dd, ad)
        p, lam, metrics = oracle.pair, oracle.lam, oracle.metrics
        counts.svd_matrices += oracle.counts.svd_matrices
        counts.reference_iterations += oracle.counts.reference_iterations
        selection = oracle.secondary_selection
        converged = oracle.converged and accepted(metrics, tolerance)
    else:
        converged = reason == "certified_gap" and accepted(metrics, tolerance)
    return SolverResult(p, lam, method, iteration, converged, did_fallback, reason,
        metrics, counts, time.perf_counter() - started, min_rcond, selection, history)


def comparison(u, d, a, pair, reference_pair, *, step=.01):
    """Errors in the actual finite neuron matrix step, without m*n*n storage."""
    u, d, a, pair, reference_pair = (v.double() for v in (u, d, a, pair, reference_pair))
    error = pair - reference_pair
    squared_error, squared_reference = 0., 0.
    # Frobenius inner products of outer products: <ab^T,cd^T>=<a,c><b,d>.
    def outer_sum_norm(left, right):
        gl = torch.einsum("kin,lin->ikl", left, left)
        gr = torch.einsum("kin,lin->ikl", right, right)
        return float((gl * gr).sum().clamp_min(0))
    left = torch.stack((-step * error[1], -step * d,
                        step * step * error[1], step * step * reference_pair[1]))
    right = torch.stack((u, error[0], pair[0], error[0]))
    squared_error = outer_sum_norm(left, right)
    rl = torch.stack((-step * reference_pair[1], -step * d, step * step * reference_pair[1]))
    rr = torch.stack((u, reference_pair[0], reference_pair[0]))
    squared_reference = outer_sum_norm(rl, rr)
    value = float((a * reference_pair).sum())
    return {"relative_objective_loss": float((a * (reference_pair - pair)).sum()) / max(abs(value), 1e-300),
            "relative_direction_error": float(error.norm()) / max(float(reference_pair.norm()), 1e-300),
            "relative_finite_delta_x_error": math.sqrt(squared_error / max(squared_reference, 1e-300))}
