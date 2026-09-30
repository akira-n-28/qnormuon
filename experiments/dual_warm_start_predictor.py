"""Research-only O(m) temporal predictors for original-coordinate dual lambda.

The returned multiplier only initializes the unchanged certified smooth solver.
No current or future optimum is an input to these functions.
"""
from dataclasses import dataclass
import math

import torch

from qnormuon.coupled_solver import adjoint, multipliers, _stable_norm


@dataclass(frozen=True)
class ProblemScale:
    beta: torch.Tensor
    magnitude: float
    coord: torch.Tensor


@dataclass(frozen=True)
class SolvedState:
    lam: torch.Tensor
    scale: ProblemScale


def problem_scale(u: torch.Tensor, d: torch.Tensor, a: torch.Tensor) -> ProblemScale:
    """Match production centering and row whitening using fp64 inputs."""
    u, d, a = u.double(), d.double(), a.double()
    beta = multipliers(u, d, a)
    magnitude = _stable_norm(a - adjoint(u, d, beta))
    coord = (u.square() + d.square()).sum(1).rsqrt()
    if not math.isfinite(magnitude) or not torch.isfinite(beta).all():
        raise ValueError("nonfinite dual centering")
    return ProblemScale(beta, magnitude, coord)


def predict(policy: str, current: ProblemScale, previous: SolvedState | None,
            older: SolvedState | None = None) -> torch.Tensor:
    """Predict in a declared coordinate; never consult the current optimum.

    `raw_*` predicts original lambda. `centered_*` predicts
    (lambda-beta)/magnitude. `whitened_*` predicts production's internal z
    and maps it through the CURRENT beta, magnitude, and row whitening.
    Whitened state is only a research candidate, never persisted as lambda.
    """
    if policy == "center":
        return current.beta.clone()
    if previous is None:
        return current.beta.clone()
    if policy == "previous":
        return previous.lam.clone()
    name, _, coefficient = policy.partition(":")
    if name not in ("raw", "centered", "whitened"):
        raise ValueError(f"unknown predictor {policy}")
    alpha = float(coefficient)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("finite nonnegative extrapolation coefficient required")
    if name == "raw":
        now = previous.lam
        before = older.lam if older is not None else now
        return now + alpha * (now - before)
    if current.magnitude <= 0 or previous.scale.magnitude <= 0:
        return current.beta.clone()
    def coordinate(state):
        residual = (state.lam - state.scale.beta) / state.scale.magnitude
        if name == "whitened":
            residual = residual / state.scale.coord
        return residual
    now = coordinate(previous)
    before = coordinate(older) if older is not None and older.scale.magnitude > 0 else now
    estimate = now + alpha * (now - before)
    if name == "whitened":
        estimate = estimate * current.coord
    return current.beta + current.magnitude * estimate


def clone_state(lam: torch.Tensor, scale: ProblemScale) -> SolvedState:
    return SolvedState(lam.detach().clone(), ProblemScale(scale.beta.detach().clone(),
                        scale.magnitude, scale.coord.detach().clone()))


def pack_state(state: SolvedState | None):
    """Small checkpoint payload, O(m), independent of SVD factors."""
    if state is None:
        return None
    return dict(lam=state.lam.detach().clone(), beta=state.scale.beta.detach().clone(),
                magnitude=state.scale.magnitude, coord=state.scale.coord.detach().clone())


def unpack_state(payload):
    if payload is None:
        return None
    return SolvedState(payload["lam"], ProblemScale(payload["beta"],
                       payload["magnitude"], payload["coord"]))
