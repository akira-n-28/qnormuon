"""Research-only first-order multiplier predictor from an accepted residual SVD.

No predictor value is a certificate. The unchanged full-SVD production solver
must evaluate and certify the current problem after this initialization.
"""
from dataclasses import dataclass
import math
from unittest.mock import patch

import torch

import qnormuon.coupled_solver as coupled


@dataclass(frozen=True)
class AcceptedFactors:
    """Original-residual thin SVD for the exact accepted original lambda."""
    lam: torch.Tensor
    left: torch.Tensor       # [2,m,n]
    singular: torch.Tensor   # [2,n], in ORIGINAL residual scale
    right: torch.Tensor      # [2,n,n], V^T
    reconstruction_error: float

    def tensor_bytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.lam, self.left, self.singular, self.right))

    def factor_bytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.left, self.singular, self.right))

    def reconstruct(self):
        # A compressed cache stores factors in fp32, but prediction arithmetic
        # remains fp64. Casting after the product would compound cache rounding.
        left, singular, right = (value.double() for value in
                                 (self.left, self.singular, self.right))
        return (left * singular[:, None, :]) @ right

    def polar(self):
        return self.left.double() @ self.right.double()

    def predictor_copy(self, dtype):
        """Optional predictor-only compression; lambda remains original fp64."""
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("factor cache dtype must be fp32 or fp64")
        return AcceptedFactors(self.lam.clone(), self.left.to(dtype),
                               self.singular.to(dtype), self.right.to(dtype),
                               self.reconstruction_error)


def pack_factors(cache):
    return dict(lam=cache.lam.detach().clone(), left=cache.left.detach().clone(),
                singular=cache.singular.detach().clone(), right=cache.right.detach().clone(),
                reconstruction_error=cache.reconstruction_error)


def unpack_factors(payload):
    return AcceptedFactors(**payload)


@torch.no_grad()
def solve_and_capture(u, d, a, *, config=None, initial_lambda=None, observer=None):
    """Capture ONLY factors tied to the final smoothly certified result.

    The production solver is unmodified. A rejected trial or earlier certified
    candidate cannot be selected: final lambda AND returned recovered primal
    must be the same objects that passed the accepted certificate call.
    """
    original_evaluate = coupled.SmoothDual.evaluate
    original_certificate = coupled.certificate
    original_newton = coupled._newton_direction
    evaluations = {}
    certificates = []

    def evaluate(problem, z):
        ev = original_evaluate(problem, z)
        evaluations[id(ev.pair)] = ev
        if observer is not None:
            observer("evaluation", problem, ev)
        return ev

    def newton(problem, ev, curvature_scale, max_cg=30):
        direction = original_newton(problem, ev, curvature_scale, max_cg)
        if observer is not None:
            observer("newton", problem, ev, direction)
        return direction

    def certificate(u0, d0, a0, lam, candidate, counts, **kwargs):
        primal, metrics = original_certificate(u0, d0, a0, lam, candidate, counts, **kwargs)
        ev = evaluations.get(id(candidate))
        if ev is not None:
            certificates.append((lam, primal, ev, kwargs.get("cached_dual")))
        return primal, metrics

    with patch.object(coupled.SmoothDual, "evaluate", evaluate), \
         patch.object(coupled, "certificate", certificate), \
         patch.object(coupled, "_newton_direction", newton):
        result = coupled.solve_coupled(u, d, a, config=config, initial_lambda=initial_lambda)
    if not result.converged or result.fallback or result.reason != "certified_gap":
        return result, None
    matches = [(ev, cached) for lam, primal, ev, cached in certificates
               if lam is result.lam and primal is result.pair]
    if len(matches) != 1 or matches[0][1] is None:
        raise RuntimeError("accepted residual factors do not match final certified lambda")
    ev, cached = matches[0]
    if cached.lam is not result.lam or cached.candidate is not ev.pair:
        raise RuntimeError("accepted spectrum identity mismatch")
    if u.dtype != torch.float64 or d.dtype != torch.float64 or a.dtype != torch.float64:
        raise ValueError("factor recycling research path requires fp64 inputs")
    # A-L*(lambda)=s*B_internal, so U and Vt are unchanged by s>0 and
    # sigma_original=s*sigma_internal. Never scale U/Vt or use a trial SVD.
    cache = AcceptedFactors(result.lam.detach().clone(), ev.left.detach().clone(),
                            (cached.magnitude * ev.singular).detach().clone(),
                            ev.right.detach().clone(), 0.)
    actual = a - coupled.adjoint(u, d, result.lam)
    error = float((cache.reconstruct() - actual).norm() /
                  actual.norm().clamp_min(torch.finfo(torch.float64).tiny))
    if not math.isfinite(error) or error > 1e-10:
        raise RuntimeError(f"accepted residual reconstruction mismatch: {error}")
    return result, AcceptedFactors(cache.lam, cache.left, cache.singular, cache.right, error)


def polar_derivative(cache, perturbation):
    """The validated thin-SVD Fréchet derivative, using cached old factors."""
    # Predictor-only fp32 cache is promoted before derivative arithmetic.
    left, singular, right = (v.double() for v in
                             (cache.left, cache.singular, cache.right))
    if bool((singular <= 0).any()):
        raise ValueError("full-rank positive accepted singular spectrum required")
    ev = coupled.Evaluation(0., None, None, left, singular, right, 0., None)
    return coupled.SmoothDual(None, None, None).polar_derivative(ev, perturbation.double())


def hessian_action(cache, u, d, v):
    """H_pred(v)=L_new Dpolar_old[L_new*(v)] (positive semidefinite)."""
    return coupled.horizontal_residual(u, d,
        polar_derivative(cache, coupled.adjoint(u, d, v)))


def change_indicator(cache, u, d, a):
    current = a - coupled.adjoint(u, d, cache.lam)
    old = cache.reconstruct()
    delta = current - old
    sigma_min = float(cache.singular.double().min())
    delta_fro = float(delta.norm())
    old_fro = float(old.norm())
    return delta, dict(delta_fro=delta_fro, old_fro=old_fro,
                       sigma_min=sigma_min,
                       delta_over_sigma_min=delta_fro / sigma_min if sigma_min > 0 else math.inf,
                       delta_over_old_fro=delta_fro / old_fro if old_fro > 0 else math.inf,
                       frobenius_full_rank_gate=delta_fro < sigma_min)


@dataclass(frozen=True)
class Prediction:
    lam: torch.Tensor
    delta_lambda: torch.Tensor
    gradient: torch.Tensor
    indicator: dict
    cg_iterations: int
    reason: str


@torch.no_grad()
def predict(cache, u, d, a, *, budget=2, gate=False):
    """No SVD/SVDVALS: linearized gradient plus row-whitened matrix-free CG.

    `gate` applies the sufficient, perhaps conservative condition
    ||Delta_B||_F < sigma_min(B_old) to both residual sides. Rejection returns
    old lambda and never changes the downstream solver or its certificate.
    """
    if budget not in (1, 2, 4, 8):
        raise ValueError("research CG budget must be 1, 2, 4, or 8")
    if u.dtype != torch.float64 or d.dtype != torch.float64 or a.dtype != torch.float64:
        raise ValueError("fp64 current solver inputs required")
    delta, indicator = change_indicator(cache, u, d, a)
    if gate and not indicator["frobenius_full_rank_gate"]:
        return Prediction(cache.lam.clone(), torch.zeros_like(cache.lam),
                          torch.zeros_like(cache.lam), indicator, 0, "frobenius_gate_rejected")
    predicted_polar = cache.polar().double() + polar_derivative(cache, delta)
    gradient = -coupled.horizontal_residual(u, d, predicted_polar)
    coord = (u.square() + d.square()).sum(1).rsqrt()
    gz = coord * gradient
    damping = 1e-6 / indicator["sigma_min"]
    x = torch.zeros_like(gz)
    r = -gz.clone()
    direction = r.clone()
    squared = float(r @ r)
    reason = "budget"
    count = 0
    target = min(.1, math.sqrt(max(float(gz.norm()), 1e-16))) * math.sqrt(squared)
    for _ in range(budget):
        hv = coord * hessian_action(cache, u, d, coord * direction) + damping * direction
        curvature = float(direction @ hv)
        if curvature <= 0 or not math.isfinite(curvature):
            reason = "nonpositive_curvature"
            break
        alpha = squared / curvature
        x = x + alpha * direction
        r = r - alpha * hv
        count += 1
        new_squared = float(r @ r)
        if math.sqrt(new_squared) <= target:
            reason = "cg_residual"
            break
        direction = r + (new_squared / squared) * direction
        squared = new_squared
    shift = coord * x
    candidate = cache.lam + shift
    if not bool(torch.isfinite(candidate).all()):
        return Prediction(cache.lam.clone(), torch.zeros_like(cache.lam), gradient,
                          indicator, count, "nonfinite_prediction")
    return Prediction(candidate, shift, gradient, indicator, count, reason)
