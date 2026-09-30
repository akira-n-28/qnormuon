"""Research-only, one-sided QDWH control of the coupled smooth solver.

The production solver is untouched. A QDWH action requires a posterior proof;
an unproved certificate or Armijo acceptance only causes more work. Full SVD
is requested for unsafe derivative decisions or exhausted globalization.
These operation-model bounds are not directed-rounding interval arithmetic.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import time

import torch

from experiments.smooth_polar_alternative import diagnose
from experiments.smooth_qdwh_decisions import (
    armijo_interval, certificate_interval_decision, cg_stop_certified,
    factor_posterior, gamma, hvp_with_error, interval_sign, row_operator_upper,
)
from qnormuon import coupled_solver as cs


def certificate_action(decision):
    """Only a proved acceptance authorizes a returned update."""
    return "accept" if decision == "accept" else "continue"


def armijo_action(decision):
    """Only a proved acceptance authorizes a trial step."""
    return "accept" if decision == "accept" else "backtrack"


@dataclass
class Evaluation:
    z: torch.Tensor
    value: float
    gradient: torch.Tensor
    pair: torch.Tensor
    singular: torch.Tensor
    rcond: float
    backend: str
    value_interval: tuple
    gradient_error: float
    rcond_lower: float
    factors: list | None = None
    posteriors: list | None = None
    residual: torch.Tensor | None = None
    svd: object | None = None


@dataclass
class Result:
    pair: torch.Tensor | None
    lam: torch.Tensor | None
    metrics: dict
    converged: bool
    reason: str
    iterations: int
    qdwh_evaluations: int
    svd_evaluations: int
    line_trials: int
    hvps: int
    fallbacks: dict
    seconds: float
    timing: dict
    actions: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    audit_events: dict = field(default_factory=dict)


class ControlledSolver:
    def __init__(self, u, d, a, config, *, profile=False, oracle_audit=False):
        self.u, self.d, self.a = u, d, a
        self.config = config
        self.profile = profile
        self.oracle_audit = oracle_audit
        self.qdwh_evaluations = self.svd_evaluations = self.line_trials = self.hvps = 0
        self.fallbacks = Counter()
        self.actions = Counter()
        self.audit_events = dict(hvp=[], curvature=[], descent=[], armijo=[])
        self.timing = defaultdict(float)
        self.svd_problem = cs.SmoothDual(u, d, a)

    def clock(self):
        if self.profile and self.u.is_cuda:
            torch.cuda.synchronize(self.u.device)
        return time.perf_counter()

    def svd_evaluate(self, z, reason):
        self.fallbacks[reason] += 1
        start = self.clock()
        ev = self.svd_problem.evaluate(z)
        self.timing["full_svd_fallback"] += self.clock() - start
        self.svd_evaluations += 1
        return Evaluation(z, ev.value, ev.gradient, ev.pair, ev.singular,
                          ev.rcond, "svd", (ev.value, ev.value), 0., ev.rcond,
                          svd=ev)

    def evaluate(self, z):
        residual = self.a - cs.adjoint(self.u, self.d, z)
        start = self.clock()
        factors = [diagnose(b, guard=self.config.rcond_guard) for b in residual]
        self.timing["qdwh_decomposition"] += self.clock() - start
        self.qdwh_evaluations += 1
        start = self.clock()
        posts = [factor_posterior(b, r, guard=self.config.rcond_guard)
                 for b, r in zip(residual, factors)]
        self.timing["factor_posterior"] += self.clock() - start
        bad = next((p.reason for p in posts if not p.accepted), None)
        if bad:
            return self.svd_evaluate(z, bad)
        pair = torch.stack([r.polar for r in factors])
        singular = torch.stack([r.singular for r in factors])
        gradient = -cs.horizontal_residual(self.u, self.d, pair)
        error = row_operator_upper(self.u, self.d) * math.hypot(
            *(p.polar_error for p in posts))
        value = float(singular.sum())
        rounding = gamma(2 * self.u.shape[1] + 1) * sum(
            p.nuclear_upper for p in posts)
        interval = (sum(p.nuclear_lower for p in posts) - rounding,
                    sum(p.nuclear_upper for p in posts) + rounding)
        return Evaluation(z, value, gradient, pair, singular,
                          min(float(s[-1] / s[0]) for s in singular),
                          "qdwh", interval, error,
                          min(p.rcond_lower for p in posts), factors, posts,
                          residual)

    def hvp(self, ev, vector):
        self.hvps += 1
        if ev.backend == "svd":
            return self.svd_problem.hvp(ev.svd, vector), 0.
        start = self.clock()
        e = -cs.adjoint(self.u, self.d, vector)
        value, radius = hvp_with_error(ev.residual, e, ev.factors,
                                       ev.posteriors, self.u, self.d)
        if self.oracle_audit:
            self.audit_events["hvp"].append((ev.z, vector.clone(), value.clone(), radius))
        self.timing["derivative_posterior"] += self.clock() - start
        return value, radius

    def direction(self, ev, curvature_scale):
        g = ev.gradient
        damping = 1e-6 * curvature_scale
        x = torch.zeros_like(g)
        residual = -g.clone()
        p = residual.clone()
        rr = float(residual @ residual)
        target = min(.1, math.sqrt(max(float(g.norm()), 1e-16))) * math.sqrt(rr)
        if rr == 0:
            return None, "qdwh_descent_ambiguous"
        for _ in range(min(self.config.max_cg, g.numel())):
            h, err = self.hvp(ev, p)
            h = h + damping * p
            curvature = float(p @ h)
            radius = err * float(p.norm())
            if ev.backend == "qdwh" and interval_sign(curvature, radius) != "positive":
                return None, "qdwh_curvature_ambiguous"
            if self.oracle_audit and ev.backend == "qdwh":
                self.audit_events["curvature"].append((ev.z,p.clone(),curvature,radius))
            if curvature <= 0 or not math.isfinite(curvature):
                return None, "svd_nonpositive_curvature"
            alpha = rr / curvature
            x = x + alpha * p
            residual = residual - alpha * h
            new_rr = float(residual @ residual)
            if math.sqrt(new_rr) <= target:
                break
            p = residual + (new_rr / rr) * p
            rr = new_rr
        if ev.backend == "qdwh":
            hx, err = self.hvp(ev, x)
            upper = float((g + hx + damping*x).norm()) + ev.gradient_error + err
            lower = max(0., float(g.norm()) - ev.gradient_error)
            if not cg_stop_certified(upper, lower, damping=damping):
                return None, "qdwh_cg_uncertain"
            slope = float(g @ x)
            if interval_sign(slope, ev.gradient_error * float(x.norm())) != "negative":
                return None, "qdwh_descent_ambiguous"
            if self.oracle_audit:
                self.audit_events["descent"].append((ev.z,x.clone()))
        elif float(g @ x) >= 0:
            return None, "svd_no_descent"
        return x, "certified_direction"

    def certificate(self, ev, ud, dd, ad, lam, magnitude, tolerance):
        start = self.clock()
        cached = cs.CachedDualSpectrum(lam, ev.pair, ev.singular, magnitude)
        p, metrics = cs.certificate(ud, dd, ad, lam, ev.pair, cs.Counts(),
            cached_dual=cached, primal_norm_backend=self.config.primal_norm_backend)
        self.timing["certificate"] += self.clock() - start
        if ev.backend == "svd":
            return p, metrics, ev.rcond > self.config.rcond_guard and cs.accepted(metrics, tolerance), "svd_certificate"
        dual = (magnitude*ev.value_interval[0], magnitude*ev.value_interval[1])
        primal = metrics["primal_objective"]
        rounding = gamma(2*p.numel()+1)*float((ad*p).abs().sum())
        feasible = (metrics["normalized_horizontal_residual"] <= 1e-10 and
                    metrics["spectral_excess"] <= 1e-12 and
                    ev.rcond_lower > self.config.rcond_guard)
        decision = certificate_interval_decision(dual,
            (primal-rounding, primal+rounding), rcond_lower=ev.rcond_lower,
            feasible=feasible, gap_tolerance=tolerance)
        return p, metrics, certificate_action(decision) == "accept", decision

    def armijo(self, current, trial, direction, alpha):
        if trial.backend == "svd" and trial.rcond <= self.config.rcond_guard:
            self.actions["armijo_guard_backtrack"] += 1
            return "backtrack"
        if trial.backend == "qdwh" and trial.rcond_lower <= self.config.rcond_guard:
            self.actions["armijo_guard_backtrack"] += 1
            return "backtrack"
        slope = float(current.gradient @ direction)
        rounding = 2*torch.finfo(torch.float64).eps*max(1.,abs(current.value))
        if current.backend == "svd" and trial.backend == "svd":
            improvement = float(trial.gradient.norm()) < float(current.gradient.norm())
            accepted = (trial.value <= current.value+1e-4*alpha*slope+rounding and
                        (abs(alpha*slope)>rounding or improvement))
            self.actions["armijo_svd_accept" if accepted else "armijo_svd_backtrack"] += 1
            return "accept" if accepted else "backtrack"
        radius = current.gradient_error*float(direction.norm())
        slope_interval = (slope-radius, slope+radius)
        cl,cu = current.value_interval
        round_low = 2*torch.finfo(torch.float64).eps*max(
            1., 0. if cl <= 0 <= cu else min(abs(cl),abs(cu)))
        round_hi = 2*torch.finfo(torch.float64).eps*max(1.,abs(cl),abs(cu))
        decision = armijo_interval(current.value_interval,trial.value_interval,
            slope_interval,alpha=alpha,rounding=(round_low,round_hi))
        resolved = (slope_interval[1] < 0 and
                    alpha*min(abs(slope_interval[0]),abs(slope_interval[1])) > round_hi)
        improved = (float(trial.gradient.norm())+trial.gradient_error <
                    max(0.,float(current.gradient.norm())-current.gradient_error))
        if decision == "accept" and (resolved or improved):
            self.actions["armijo_qdwh_accept"] += 1
            return "accept"
        self.actions["armijo_qdwh_ambiguous" if decision != "reject" else "armijo_qdwh_proved_reject"] += 1
        return armijo_action(decision if decision != "accept" else "ambiguous")


@torch.no_grad()
def solve_one_sided(u, d, a, *, config=None, initial_lambda=None,
                    profile=False, oracle_audit=False):
    config = config or cs.SolverConfig(fallback=False)
    if config.dtype != torch.float64:
        raise ValueError("research QDWH solver requires fp64")
    with torch.autocast(device_type=u.device.type, enabled=False):
        u,d,a=(v.double() for v in (u,d,a))
        cs._validate(u,d,a)
        started=time.perf_counter()
        beta=cs.multipliers(u,d,a)
        centered=a-cs.adjoint(u,d,beta)
        magnitude=cs._stable_norm(centered)
        if magnitude == 0:
            oracle=cs.solve_coupled(u,d,a,config=config,initial_lambda=initial_lambda)
            return Result(oracle.pair,oracle.lam,oracle.metrics,oracle.converged,
                "zero_cotangent",0,0,0,0,0,{},time.perf_counter()-started,{}, {}, [])
        coord=(u.square()+d.square()).sum(1).rsqrt()
        eu,ed=u*coord[:,None],d*coord[:,None]
        work=ControlledSolver(eu,ed,centered/magnitude,config,profile=profile,
                              oracle_audit=oracle_audit)
        z=torch.zeros_like(beta)
        if initial_lambda is not None:
            z=(initial_lambda.double()-beta)/(magnitude*coord)
        ev=work.evaluate(z)
        history=[]
        reason="iteration_budget"
        pair=lam=metrics=None
        max_iterations=config.max_iterations
        for iteration in range(max_iterations+1):
            lam=beta+magnitude*coord*z
            pair,metrics,certified,decision=work.certificate(ev,u,d,a,lam,magnitude,
                config.tolerance or 3e-5)
            work.actions["certificate_qdwh_accept" if certified and ev.backend=="qdwh" else
                "certificate_svd_accept" if certified else
                "certificate_qdwh_continue" if ev.backend=="qdwh" else
                "certificate_svd_continue"] += 1
            history.append(dict(iteration=iteration,backend=ev.backend,
                certificate=decision,gap=metrics["normalized_gap"],
                rcond=ev.rcond,qdwh_evaluations=work.qdwh_evaluations,
                svd_evaluations=work.svd_evaluations))
            if certified:
                reason="certified_gap"; break
            if not torch.isfinite(ev.gradient).all() or not math.isfinite(ev.value):
                reason="nonfinite_evaluation"; break
            if ev.backend == "svd" and ev.rcond <= config.rcond_guard:
                reason="ill_conditioned_residual"; break
            if magnitude <= 32*torch.finfo(torch.float64).eps*cs._stable_norm(a):
                reason="cancellation_dominated_cotangent"; break
            if iteration == max_iterations:
                reason="newton_budget_exhausted"; break
            if len(history)>12 and history[-1]["gap"]>.99*history[-12]["gap"]:
                reason="gap_stagnation"; break
            scale=max(float((eu.square()+ed.square()).sum(1).max()),1e-30)/float(ev.singular.min())
            direction,dreason=work.direction(ev,scale)
            if direction is None and ev.backend=="qdwh":
                ev=work.svd_evaluate(z,dreason)
                scale=max(float((eu.square()+ed.square()).sum(1).max()),1e-30)/float(ev.singular.min())
                direction,dreason=work.direction(ev,scale)
            if direction is None:
                reason=dreason; break
            found=False; alpha=1.
            for _ in range(24):
                work.line_trials+=1
                trial_z=z+alpha*direction
                trial=work.evaluate(trial_z)
                if work.armijo(ev,trial,direction,alpha)=="accept":
                    if oracle_audit and (ev.backend=="qdwh" or trial.backend=="qdwh"):
                        work.audit_events["armijo"].append((z,trial_z,
                            direction.clone(),alpha))
                    z,ev=trial_z,trial
                    found=True
                    break
                alpha*=.5
            if not found:
                # Explicit recovery after the unchanged line-search budget.
                # This is a full-SVD solver retry, distinct from CPU ADMM.
                work.fallbacks["qdwh_line_search_exhausted"]+=1
                reason="qdwh_line_search_exhausted"
                break
        else:
            iteration=max_iterations
        if reason != "certified_gap" and reason in (
                "qdwh_line_search_exhausted","newton_budget_exhausted",
                "gap_stagnation","svd_no_descent","svd_nonpositive_curvature"):
            start=work.clock()
            rescue=cs.solve_coupled(u,d,a,config=cs.SolverConfig(
                dtype=torch.float64,tolerance=config.tolerance,
                rcond_guard=config.rcond_guard,max_iterations=config.max_iterations,
                max_cg=config.max_cg,fallback=False,
                primal_norm_backend=config.primal_norm_backend),
                initial_lambda=beta+magnitude*coord*z)
            work.timing["full_svd_solver_rescue"]+=work.clock()-start
            work.svd_evaluations+=rescue.counts.svd_matrices//2
            work.fallbacks["full_svd_solver_rescue"]+=1
            if rescue.converged:
                pair,lam,metrics=rescue.pair,rescue.lam,rescue.metrics
                reason="certified_svd_rescue"
        converged=reason in ("certified_gap","certified_svd_rescue")
        return Result(pair,lam,metrics,converged,reason,iteration,
            work.qdwh_evaluations,work.svd_evaluations,work.line_trials,
            work.hvps,dict(work.fallbacks),time.perf_counter()-started,
            dict(work.timing),dict(work.actions),history,work.audit_events)
