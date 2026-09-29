"""K=I coupled horizontal LMO, ported from the validated dual solver study.

Smooth work is device-local float32/64. Recovery and certification are float64.
The only research dependency is the explicit, lazy CPU reference fallback.
"""
from dataclasses import dataclass, field
import math
import time
import torch


def _stable_norm(value):
    scale = float(value.abs().max())
    return scale * float((value / scale).norm()) if scale else 0.


def horizontal_residual(u, d, p):
    return (u * p[0]).sum(1) - (d * p[1]).sum(1)


def adjoint(u, d, lam):
    return torch.stack((lam[:, None] * u, -lam[:, None] * d))


def multipliers(u, d, p):
    return horizontal_residual(u, d, p) / (u.square() + d.square()).sum(1)


def horizontal_project(u, d, p):
    return p - adjoint(u, d, multipliers(u, d, p))


def _validate(u, d, a):
    if u.ndim != 2 or d.shape != u.shape or a.shape != (2,) + u.shape:
        raise ValueError("expected m x n weights and a 2 x m x n objective")
    if not 1 <= u.shape[1] <= u.shape[0]:
        raise ValueError("require m >= n >= 1")
    for v in (u, d, a):
        if v.device != u.device or v.dtype != u.dtype or v.dtype not in (torch.float32, torch.float64):
            raise ValueError("common device and float32/float64 solver dtype required")
        if not torch.isfinite(v).all():
            raise ValueError("finite solver inputs required")
    w = (u.double().square() + d.double().square()).sum(1)
    if not torch.isfinite(w).all() or bool((w <= 0).any()):
        raise ValueError("regular row metric is not representable in float64")
    if bool((u.abs().amax(1) == 0).any()) or bool((d.abs().amax(1) == 0).any()):
        raise ValueError("regular nonzero rows required")
    if not torch.allclose(u.double().norm(dim=1), d.double().norm(dim=1), rtol=1e-5, atol=0):
        raise ValueError("balanced canonical weights required")


@dataclass(frozen=True)
class SolverConfig:
    dtype: torch.dtype = torch.float64
    tolerance: float | None = 3e-5
    rcond_guard: float | None = 1e-4
    initial_iterations: int = 2
    max_iterations: int = 100
    max_cg: int = 30
    fallback: bool = True
    reference_max_iterations: int = 20000

    def __post_init__(self):
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("solver dtype must be float32 or float64")
        if self.tolerance is not None and (not math.isfinite(self.tolerance) or self.tolerance <= 0):
            raise ValueError("finite positive tolerance required")
        if self.rcond_guard is not None and not 0 < self.rcond_guard < 1:
            raise ValueError("rcond guard must lie in (0,1)")
        if self.initial_iterations < 0 or self.max_iterations < 0 or self.max_cg < 1 or self.reference_max_iterations < 1:
            raise ValueError("invalid iteration budget")

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
    residual_spectra = torch.linalg.svdvals(a - adjoint(u, d, lam))
    dual = float(residual_spectra.sum())
    rcond = residual_spectra[:, -1] / residual_spectra[:, 0].clamp_min(torch.finfo(torch.float64).tiny)
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
        "residual_rcond": float(rcond.min()),
        "signed_gap": gap, "normalized_gap": max(0., gap) / denom,
        "signed_normalized_gap": gap / denom,
    }


def accepted(metrics, tolerance):
    return (metrics["normalized_gap"] <= tolerance
            and metrics["signed_normalized_gap"] >= -1e-10
            and metrics["normalized_horizontal_residual"] <= 1e-10
            and metrics["spectral_excess"] <= 1e-12)

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
    return x


@torch.no_grad()
def _solve(u, d, a, *, config, initial_lambda=None):
    """Whitened Newton-CG; an initial budget is extended until certified."""
    method = "newton"
    tolerance, rcond_guard = config.tolerance, config.rcond_guard
    max_iterations, max_cg, fallback = config.max_iterations, config.max_cg, config.fallback
    started = time.perf_counter()
    _validate(u, d, a)
    dtype = a.dtype
    eps = torch.finfo(dtype).eps
    tolerance = 3e-5 if tolerance is None else tolerance
    guard = 1e-4 if rcond_guard is None else rcond_guard
    if tolerance <= 0 or not 0 < guard < 1:
        raise ValueError("positive tolerance and rcond guard in (0,1) required")
    counts = Counts()
    ud, dd, ad = u.double(), d.double(), a.double()
    beta = multipliers(ud, dd, ad)
    centered = ad - adjoint(ud, dd, beta)
    magnitude = _stable_norm(centered)
    if not math.isfinite(magnitude) or not torch.isfinite(beta).all():
        raise ValueError("intrinsic objective normalization is not representable in float64")
    if magnitude == 0:
        p, metrics = certificate(ud, dd, ad, beta, torch.zeros_like(ad), counts)
        return SolverResult(p, beta, method, 0, accepted(metrics, tolerance), False,
            "zero_cotangent", metrics, counts, time.perf_counter() - started, 0., "exact_zero")
    w = (ud.square() + dd.square()).sum(1)
    coord = w.rsqrt()
    effective_u, effective_d = (ud * coord[:, None]).to(dtype), (dd * coord[:, None]).to(dtype)
    problem = SmoothDual(effective_u, effective_d, (centered / magnitude).to(dtype), counts)
    z = torch.zeros(u.shape[0], dtype=dtype, device=u.device)
    if initial_lambda is not None:
        if initial_lambda.shape != beta.shape or not torch.isfinite(initial_lambda).all():
            raise ValueError("finite multiplier of shape (m,) required")
        nonzero = coord != 0
        z[nonzero] = ((initial_lambda.to(device=u.device, dtype=torch.float64)[nonzero] - beta[nonzero]) /
                      (magnitude * coord[nonzero])).to(dtype)
    ev = problem.evaluate(z)
    min_rcond = ev.rcond
    history = []
    reason, iteration = "iteration_budget", 0
    best = None
    for iteration in range(max_iterations + 1):
        lam = beta + magnitude * coord * z.double()
        p, metrics = certificate(ud, dd, ad, lam, ev.pair, counts)
        history.append({"iteration": iteration, "rcond": ev.rcond,
                        "extended_initial_budget": iteration > config.initial_iterations,
                        "polar_horizontal_residual": float(horizontal_residual(u, d, ev.pair).abs().max()),
                        **metrics})
        if best is None or metrics["normalized_gap"] < best[2]["normalized_gap"]:
            best = (p, lam, metrics)
        # Check conditioning BEFORE accepting a tiny value gap: it does not
        # certify direction accuracy at a near-rank boundary.
        if not torch.isfinite(ev.gradient).all() or not math.isfinite(ev.value):
            reason = "nonfinite_evaluation"
            break
        if ev.rcond <= guard or metrics["residual_rcond"] <= guard:
            reason = "ill_conditioned_residual"
            break
        if magnitude <= 32 * torch.finfo(torch.float64).eps * _stable_norm(ad):
            reason = "cancellation_dominated_cotangent"
            break
        if accepted(metrics, tolerance):
            best = (p, lam, metrics)
            reason = "certified_gap"
            break
        if iteration == max_iterations:
            break
        if len(history) > 12 and history[-1]["normalized_gap"] > .99 * history[-12]["normalized_gap"]:
            reason = "gap_stagnation"
            break
        g = ev.gradient
        curvature_scale = max(float((effective_u.square() + effective_d.square()).sum(1).max()), 1e-30) / float(ev.singular.min())
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
        z, ev = trial_z, trial
        min_rcond = min(min_rcond, ev.rcond)
    p, lam, metrics = best
    did_fallback = reason != "certified_gap" and fallback
    selection = "smooth_primary_unique_in_exact_limit" if reason == "certified_gap" else "uncertified_primary"
    if did_fallback:
        oracle = _reference(ud, dd, ad, config.reference_max_iterations)
        p, lam, metrics = oracle.pair, oracle.lam, oracle.metrics
        counts.svd_matrices += oracle.counts.svd_matrices
        counts.reference_iterations += oracle.counts.reference_iterations
        selection = oracle.secondary_selection
        converged = oracle.converged and accepted(metrics, tolerance)
    else:
        converged = reason == "certified_gap" and accepted(metrics, tolerance)
    return SolverResult(p, lam, method, iteration, converged, did_fallback, reason,
        metrics, counts, time.perf_counter() - started, min_rcond, selection, history)


def _reference(u, d, a, max_iterations):
    # Deliberately retain the independent validated implementation. No rank
    # selection is inferred, and GPU inputs incur an explicit CPU transfer.
    from experiments.dual_solver import reference
    result = reference(u.cpu(), d.cpu(), a.cpu(), max_iterations=max_iterations)
    # Re-evaluate on the originating device, including residual conditioning.
    result.pair = result.pair.to(u.device)
    result.lam = result.lam.to(u.device)
    result.pair, result.metrics = certificate(u, d, a, result.lam, result.pair, result.counts)
    result.converged = result.converged and accepted(result.metrics, 1e-10)
    return result


@torch.no_grad()
def solve_coupled(u, d, a, *, config=None, initial_lambda=None):
    """Return a certified float64 direction, or an explicitly failed result.

    Inputs are balanced weights and canonical covectors. This function disables
    autocast: model storage precision must never choose the spectral dtype.
    Optimizer.step refuses any result whose ``converged`` flag is false.
    """
    config = config or SolverConfig()
    with torch.autocast(device_type=u.device.type, enabled=False):
        u, d, a = (v.to(config.dtype) for v in (u, d, a))
        _validate(u, d, a)
        try:
            result = _solve(u, d, a, config=config, initial_lambda=initial_lambda)
        except torch.linalg.LinAlgError as error:
            if not config.fallback:
                raise RuntimeError("spectral evaluation failed; fallback disabled") from error
            result = _reference(u.double(), d.double(), a.double(), config.reference_max_iterations)
            result.fallback = True
            result.reason = "svd_failure: " + str(error)
        tolerance = config.tolerance if config.tolerance is not None else 3e-5
        result.converged = result.converged and accepted(result.metrics, tolerance)
        result.metrics.update({
            "newton_iterations": result.iterations if result.method == "newton" else 0,
            "cg_iterations": result.counts.hvp,
            "svd_evaluations": result.counts.svd_matrices,
            "reference_iterations": result.counts.reference_iterations,
            "primal_dual_gap": result.metrics["signed_gap"],
            "absolute_gap": abs(result.metrics["signed_gap"]),
            "minimum_smooth_rcond": result.min_accepted_rcond,
            "fallback_used": result.fallback,
            "fallback_reason": result.reason if result.fallback else None,
            "solver_dtype": str(config.dtype),
            "gap_tolerance": tolerance,
            "fallback_backend": "cpu_float64_reference_admm" if result.fallback else None,
            "certification_dtype": "torch.float64",
            "returned_direction_dtype": str(result.pair.dtype),
            "secondary_selection": result.secondary_selection,
            "converged": result.converged,
            "extended_initial_budget": result.iterations > config.initial_iterations,
        })
        return result
