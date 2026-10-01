"""Research-only full-rank PRIMARY epsilon-LMO admission.

Production-v0 and its rcond guard are untouched. See the experiment report for
the conditional fp64 operation model. No reference solver or offline optimum
is used here. Full rank at a current multiplier never proves P-dagger selection.
"""
from dataclasses import dataclass, field
import math
import time
import torch
import qnormuon.coupled_solver as cs
from experiments.full_rank_direction import gamma, norm_upper, residual_uncertainty
from experiments.qso_output_contract import value_posterior

PRIMARY_SCOPE = "primary_epsilon_lmo_full_rank"


class NumericalFailure(RuntimeError):
    def __init__(self, reason, diagnostic=None):
        super().__init__(reason)
        self.diagnostic = diagnostic


def finite(*values):
    for value in values:
        ok = bool(torch.isfinite(value).all()) if isinstance(value, torch.Tensor) else math.isfinite(value)
        if not ok:
            raise NumericalFailure("nonfinite_action")


@torch.no_grad()
def rank_posterior(u, d, a, lam, ev, magnitude):
    """Matching original-residual factors; no new decomposition or rcond cutoff.

    B_original = magnitude * B_internal in exact arithmetic. Reconstruction
    against the original residual includes rounded centering/whitening conversion.
    Orthogonalization is a mathematical bound, not another numerical factorization.
    du,dv<1 and eta<sigma_min certify rank under the existing gamma model.
    """
    singular = magnitude * ev.singular
    term = cs.adjoint(u, d, lam)
    b = a - term
    finite(b, ev.left, singular, ev.right, ev.pair, ev.gradient, ev.value, lam)
    if bool((singular <= 0).any()):
        raise NumericalFailure("rank_ambiguous_or_deficient")
    # The inherited model is normal finite arithmetic, not an underflow model.
    if bool((singular < torch.finfo(torch.float64).tiny).any()):
        raise NumericalFailure("unrepresentable_spectrum")
    try:
        eta, details = residual_uncertainty(b, ev.left, singular, ev.right)
    except ValueError as error:
        raise NumericalFailure("factor_posterior_failed") from error
    construction = gamma(3) * norm_upper(a.abs() + term.abs(), (-2, -1)) * (1 + gamma(16))
    eta = eta + construction
    alpha = (singular[:, -1] - eta) * (1 - gamma(2))
    finite(eta, alpha)
    result = dict(alpha_lower=alpha.tolist(), uncertainty=eta.tolist(),
                  sigma_min=singular[:, -1].tolist(),
                  sigma_max=singular[:, 0].tolist(),
                  rcond=(singular[:, -1] / singular[:, 0]).tolist(), **details)
    if not bool((alpha > 0).all()):
        raise NumericalFailure("rank_ambiguous_or_deficient", result)
    return result


@dataclass
class AdmissionResult:
    pair: torch.Tensor | None
    lam: torch.Tensor
    converged: bool
    reason: str
    iterations: int
    counts: cs.Counts
    metrics: dict
    seconds: float
    history: list = field(default_factory=list)
    evaluations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    selection_semantics: str = PRIMARY_SCOPE
    selection_certified: bool = False
    reference_used: bool = False


def final_admission(post, metrics):
    """Conservative primary value plus the unchanged numerical feasibility tests.

    This predicate is not a selected-direction or optimal-face-uniqueness test.
    """
    scalars = [post["dual_lower"], post["dual_upper"], post["primal_lower"],
               post["gap_upper"], post["normalized_gap_upper"],
               *post["alpha_lower"], metrics["signed_normalized_gap"],
               metrics["normalized_horizontal_residual"], metrics["spectral_excess"]]
    return (all(math.isfinite(v) for v in scalars)
            and min(post["alpha_lower"]) > 0 and post["dual_lower"] > 0
            and post["gap_upper"] >= 0 and post["normalized_gap_upper"] <= 3e-5
            and metrics["signed_normalized_gap"] >= -1e-10
            and metrics["normalized_horizontal_residual"] <= 1e-10
            and metrics["spectral_excess"] <= 1e-12)


def newton_direction(problem, ev, curvature_scale, max_cg, rank):
    """Production damped truncated-CG arithmetic, with fail-closed observability.

    A finite partial direction at an ordinary nonpositive-curvature/budget exit
    remains eligible only if it has completed positive-curvature work and the
    subsequent production descent check passes. Nonfinite actions never survive.
    No direction-error or HVP-error threshold is fitted here.
    """
    g = ev.gradient
    damping = 1e-6 * curvature_scale
    x = torch.zeros_like(g)
    residual = -g.clone()
    direction = residual.clone()
    squared = float(residual @ residual)
    target = min(.1, math.sqrt(max(float(g.norm()), 1e-16))) * math.sqrt(squared)
    finite(damping, squared, target)
    if damping <= 0 or squared <= 0:
        raise NumericalFailure("unusable_cg_result")
    queries = []
    termination = "cg_budget"
    completed = 0
    for _ in range(min(max_cg, g.numel())):
        hvp = problem.hvp(ev, direction)
        finite(hvp)
        hv = hvp + damping * direction
        curvature = float(direction @ hv)
        vector_norm, hvp_norm = float(direction.norm()), float(hvp.norm())
        undamped_curvature = float(direction @ hvp)
        finite(hv, curvature, vector_norm, hvp_norm, undamped_curvature)
        queries.append(dict(vector_norm=vector_norm, hvp_norm=hvp_norm,
            curvature=curvature, undamped_curvature=undamped_curvature,
            cg_residual=math.sqrt(squared), damping=damping, **rank))
        if curvature <= 0:
            termination = "nonpositive_curvature"
            break
        alpha = squared / curvature
        finite(alpha)
        x = x + alpha * direction
        residual = residual - alpha * hv
        new_squared = float(residual @ residual)
        finite(x, residual, new_squared)
        completed += 1
        if math.sqrt(new_squared) <= target:
            termination = "residual_target"
            break
        direction = residual + (new_squared / squared) * direction
        finite(direction)
        squared = new_squared
    if completed == 0:
        raise NumericalFailure("unusable_cg_result")
    finite(x)
    return x, dict(queries=queries, cg_termination=termination,
        cg_completed=completed, cg_final_residual=float(residual.norm()),
        cg_target=target, damping=damping)


@torch.no_grad()
def _solve_full_rank(u, d, a, *, initial_lambda=None, config=None, profile=False,
                     known_nonsmooth_face=False):
    """No update on failure: pair=None. No CPU reference call on any path.

    Only iteration budgets are configurable; all contract thresholds stay fixed.
    The retained SolverConfig.rcond_guard is the v0 diagnostic, never a new cutoff.
    """
    config = config or cs.SolverConfig(fallback=False)
    if (config.dtype != torch.float64 or config.tolerance not in (None, 3e-5)
            or config.rcond_guard not in (None, 1e-4)
            or config.primal_norm_backend != "gram_upper"):
        raise ValueError("frozen fp64/3e-5/v0-diagnostic/gram_upper policy required")
    if any(v.dtype != torch.float64 for v in (u, d, a)):
        raise ValueError("explicit fp64 represented inputs required")
    timings = dict(rank_seconds=0., value_seconds=0.)
    def timed(key, fn):
        if profile and u.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        if profile and u.is_cuda:
            torch.cuda.synchronize()
        timings[key] += time.perf_counter() - start
        return out
    started = time.perf_counter()
    cs._validate(u, d, a)
    counts = cs.Counts()
    history, evaluations, actions = [], [], []
    metrics, iteration = {}, 0
    beta = cs.multipliers(u, d, a)
    centered = a - cs.adjoint(u, d, beta)
    magnitude = cs._stable_norm(centered)
    finite(beta, magnitude)
    lam = beta.clone()
    def finish(pair, reason):
        passed = pair is not None
        metrics.update(timings, selection_semantics="exact_zero" if reason == "zero_cotangent" else PRIMARY_SCOPE,
            selection_certified=reason == "zero_cotangent", reference_used=False,
            production_rcond_guard=1e-4, smooth_backend="full_fp64_thin_svd",
            newton_iterations=iteration, cg_iterations=counts.hvp, line_trials=counts.line_trials)
        return AdmissionResult(pair, lam.clone(), passed, reason, iteration, counts, metrics,
            time.perf_counter()-started, history, evaluations, actions,
            metrics["selection_semantics"], metrics["selection_certified"])
    if magnitude == 0:
        # The represented exact-zero branch is deliberately separate from rank.
        p, zero_metrics = cs.certificate(u, d, a, beta, torch.zeros_like(a), counts,
                                        primal_norm_backend="gram_upper")
        metrics.update(zero_metrics)
        return finish(p if cs.accepted(metrics, 3e-5) else None, "zero_cotangent")
    if known_nonsmooth_face:
        return finish(None, "explicit_nonsmooth_face")
    if magnitude <= 32 * torch.finfo(torch.float64).eps * cs._stable_norm(a):
        return finish(None, "cancellation_dominated_cotangent")
    w = (u.square()+d.square()).sum(1)
    coord = w.rsqrt()
    effective_u, effective_d = u * coord[:, None], d * coord[:, None]
    problem = cs.SmoothDual(effective_u, effective_d, centered/magnitude, counts)
    z = torch.zeros(u.shape[0], device=u.device, dtype=torch.float64)
    if initial_lambda is not None:
        if initial_lambda.shape != beta.shape:
            raise ValueError("finite original-coordinate multiplier of shape (m,) required")
        finite(initial_lambda)
        z = (initial_lambda.to(u.device)-beta)/(magnitude*coord)
    finite(z, coord, effective_u, effective_d, problem.a)
    def evaluate(coordinate, kind, step=0.):
        finite(coordinate)
        ev = problem.evaluate(coordinate)
        original_lambda = beta + magnitude * coord * coordinate
        if ev.coordinate is not coordinate:
            raise NumericalFailure("mismatched_evaluation")
        try:
            rank = timed("rank_seconds", lambda:rank_posterior(u,d,a,original_lambda,ev,magnitude))
        except NumericalFailure as error:
            evaluations.append(dict(iteration=iteration,kind=kind,step=step,
                reason=str(error),rank=error.diagnostic,eligible=False))
            raise
        evaluations.append(dict(iteration=iteration, kind=kind, step=step,
            value=ev.value, gradient_norm=float(ev.gradient.norm()), rank=rank,
            below_production_guard=ev.rcond <= 1e-4))
        return ev, rank
    try:
        ev, rank = evaluate(z, "initial")
        for iteration in range(config.max_iterations+1):
            lam = beta + magnitude * coord * z
            if ev.coordinate is not z:
                raise NumericalFailure("mismatched_evaluation")
            cached = cs.CachedDualSpectrum(lam, ev.pair, ev.singular, magnitude)
            p, raw = cs.certificate(u,d,a,lam,ev.pair,counts,cached_dual=cached,
                                   primal_norm_backend="gram_upper")
            finite(p)
            post = timed("value_seconds", lambda:value_posterior(
                u,d,a,lam,p,ev.left,magnitude*ev.singular,ev.right,raw))
            metrics.update(raw, posterior=post, rank=rank)
            history.append(dict(iteration=iteration, **raw, posterior=post,
                rank=rank, gradient_norm=float(ev.gradient.norm()),
                cg_count=counts.hvp, line_trials=counts.line_trials))
            if final_admission(post, raw):
                return finish(p, "primary_certified")
            if iteration == config.max_iterations:
                return finish(None, "iteration_budget")
            if len(history)>12 and history[-1]["normalized_gap"]>.99*history[-12]["normalized_gap"]:
                return finish(None, "gap_stagnation")
            curvature_scale = max(float((effective_u.square()+effective_d.square()).sum(1).max()),1e-30)/float(ev.singular.min())
            direction, action = newton_direction(problem,ev,curvature_scale,config.max_cg,rank)
            slope = float(ev.gradient @ direction)
            finite(direction, slope, float(direction.norm()))
            action.update(iteration=iteration, gradient_norm=float(ev.gradient.norm()),
                direction_norm=float(direction.norm()), slope=slope,
                below_production_guard=ev.rcond<=1e-4, trials=[])
            actions.append(action)
            if slope >= 0:
                return finish(None, "no_descent_direction")
            step, found = 1., False
            for _ in range(24):
                counts.line_trials += 1
                trial_z = z + step * direction
                if torch.equal(trial_z,z):
                    return finish(None, "unrepresentable_multiplier_step")
                try:
                    trial, trial_rank = evaluate(trial_z,"line_trial",step)
                except NumericalFailure as error:
                    action["trials"].append(dict(step=step, eligible=False, reason=str(error),rank=error.diagnostic))
                    step *= .5
                    continue
                rounding = 2*torch.finfo(torch.float64).eps*max(1.,abs(ev.value))
                resolved = abs(step*slope)>rounding
                improves = float(trial.gradient.norm())<float(ev.gradient.norm())
                rhs = ev.value + 1e-4*step*slope + rounding
                armijo = trial.value<=rhs
                action["trials"].append(dict(step=step,eligible=True,value=trial.value,
                    armijo_rhs=rhs,armijo=armijo,resolved_decrease=resolved,
                    gradient_improves=improves,rank=trial_rank))
                if armijo and (resolved or improves):
                    found=True
                    break
                step *= .5
            if not found:
                return finish(None,"line_search_failed")
            z, ev, rank = trial_z, trial, trial_rank
    except (NumericalFailure, ValueError, torch.linalg.LinAlgError) as error:
        if isinstance(error,NumericalFailure):
            metrics["failure_diagnostic"] = error.diagnostic
        return finish(None, str(error) if isinstance(error,NumericalFailure) else
                      "numerical_posterior_or_svd_failure: " + str(error))


def solve_full_rank(u, d, a, *, initial_lambda=None, config=None, profile=False,
                    known_nonsmooth_face=False):
    with torch.autocast(device_type=u.device.type, enabled=False):
        return _solve_full_rank(u,d,a,initial_lambda=initial_lambda,config=config,profile=profile,
                               known_nonsmooth_face=known_nonsmooth_face)
