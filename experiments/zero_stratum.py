"""Isolated quotient/zero-stratum geometry; no optimizer integration.

Regular formulas reject zero factors instead of inserting an epsilon metric.
Birth is an X-space cone LMO, not a finite horizontal factor velocity at zero.
See docs/ZERO_STRATUM_GEOMETRY.md for the distinction and impossibility results.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


def stable_norm(v: Tensor) -> Tensor:
    """Scaled Euclidean/Frobenius norm for reference experiments."""
    scale = v.abs().max()
    if scale == 0:
        return scale
    return scale * (v / scale).square().sum().sqrt()


def regular_data(u: Tensor, d: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if u.ndim != 1 or d.ndim != 1 or not u.numel() or not d.numel():
        raise ValueError("factors must be nonempty vectors")
    if u.dtype not in (torch.float32, torch.float64) or d.dtype != u.dtype or d.device != u.device:
        raise ValueError("use matching explicit float32/float64 factors")
    if not torch.isfinite(u).all() or not torch.isfinite(d).all():
        raise ValueError("factors must be finite")
    r, s = stable_norm(u), stable_norm(d)
    if r == 0 or s == 0:
        raise ValueError("regular geometry requires both factors nonzero")
    return r, s, d / s, u / r


def balanced_factors(u: Tensor, d: Tensor) -> tuple[Tensor, Tensor]:
    """Positive-gauge balanced factors; preserves the supplied common sign."""
    r, s, a, b = regular_data(u, d)
    rho = r.sqrt() * s.sqrt()
    return rho * b, rho * a


def tangent_map(u: Tensor, d: Tensor, du: Tensor, dd: Tensor) -> Tensor:
    return torch.outer(dd, u) + torch.outer(d, du)


def tangent_projection(a: Tensor, b: Tensor, z: Tensor) -> Tensor:
    """Ambient Frobenius projection onto T_(sigma a b^T) M_1; a,b unit."""
    return torch.outer(a, a @ z) + torch.outer(z @ b, b) - (a @ z @ b) * torch.outer(a, b)


def horizontal_lift(u: Tensor, d: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
    """Minimum g_H-norm lift. At balance, g_H is factor Euclidean norm.

    Reject genuinely nontangent z. The tolerance only checks floating-point
    tangent residual; it never classifies a nonzero weight norm as zero.
    """
    r, s, a, b = regular_data(u, d)
    if z.shape != (d.numel(), u.numel()):
        raise ValueError("tangent matrix has incompatible shape")
    residual = z - tangent_projection(a, b, z)
    tolerance = 100 * torch.finfo(z.dtype).eps * max(z.shape)
    if stable_norm(residual) > tolerance * max(1., stable_norm(z).item()):
        raise ValueError("matrix is not in the regular tangent space")
    alpha = a @ z @ b
    p, q = z @ b - alpha * a, z.T @ a - alpha * b
    return (q + 0.5 * alpha * b) / s, (p + 0.5 * alpha * a) / r


def factor_metric(u: Tensor, d: Tensor, du: Tensor, dd: Tensor) -> Tensor:
    r, s, _, _ = regular_data(u, d)
    h = r / s
    return du.square().sum() / h + h * dd.square().sum()


def horizontal_projection(u: Tensor, d: Tensor, du: Tensor, dd: Tensor) -> tuple[Tensor, Tensor]:
    r, s, _, _ = regular_data(u, d)
    coefficient = 0.5 * ((u @ du) / r.square() - (d @ dd) / s.square())
    return du - coefficient * u, dd + coefficient * d


def quotient_metric(u: Tensor, d: Tensor, z: Tensor) -> Tensor:
    du, dd = horizontal_lift(u, d, z)
    return factor_metric(u, d, du, dd)


def quotient_gradient(u: Tensor, d: Tensor, g: Tensor) -> Tensor:
    """g_Q gradient in X coordinates, not the ambient Frobenius projection."""
    r, s, a, b = regular_data(u, d)
    return (r * s) * (torch.outer(a, a @ g) + torch.outer(g @ b, b))


def quotient_unit_descent(u: Tensor, d: Tensor, g: Tensor) -> Tensor:
    """Exact regular g_Q-unit-ball LMO; its X-space norm vanishes at zero.

    At a stationary point choose the minimum-norm optimizer zero. This does
    not make the map continuous at every stationary gradient on the regular set.
    """
    r, s, a, b = regular_data(u, d)
    alpha = a @ g @ b
    p, q = g @ b - alpha * a, g.T @ a - alpha * b
    denominator = (2 * alpha.square() + p.square().sum() + q.square().sum()).sqrt()
    if denominator == 0:
        return torch.zeros_like(g)
    numerator = 2 * alpha * torch.outer(a, b) + torch.outer(p, b) + torch.outer(a, q)
    return -(r.sqrt() * s.sqrt()) * numerator / denominator


@dataclass
class BirthDirection:
    direction: Tensor
    largest_singular_value: float
    gap: float
    separated_at_tolerance: bool


@torch.no_grad()
def birth_lmo(g: Tensor, *, gap_tolerance: float) -> BirthDirection:
    """Minimize <G,Z> over rank(Z)<=1, ||Z||_F<=1, reporting possible ties.

    An SVD-selected representative at a tie is not an intrinsic unique choice.
    gap_tolerance is a diagnostic and does not truncate the computed direction.
    """
    if g.ndim != 2 or min(g.shape) == 0 or not torch.isfinite(g).all():
        raise ValueError("gradient must be a nonempty finite matrix")
    if g.dtype not in (torch.float32, torch.float64):
        raise ValueError("use explicit float32/float64 gradient")
    if not math.isfinite(gap_tolerance) or gap_tolerance < 0:
        raise ValueError("gap tolerance must be finite and nonnegative")
    left, singular, right_t = torch.linalg.svd(g, full_matrices=False)
    top = singular[0].item()
    gap = top - (singular[1].item() if singular.numel() > 1 else 0.)
    z = torch.zeros_like(g) if top == 0 else -torch.outer(left[:, 0], right_t[0])
    return BirthDirection(z, top, gap, top > 0 and gap > gap_tolerance)


@torch.no_grad()
def balanced_birth(z: Tensor, *, step: float) -> tuple[Tensor, Tensor]:
    """Factor step*Z through square roots; Z must be rank <= 1.

    These are new factors, not derivatives at (0,0). SVD fixes a representative
    only for this call; no globally continuous sign convention is promised.
    """
    if step < 0 or not math.isfinite(step):
        raise ValueError("step must be finite and nonnegative")
    left, singular, right_t = torch.linalg.svd(z, full_matrices=False)
    tolerance = 100 * torch.finfo(z.dtype).eps * max(z.shape)
    if singular.numel() > 1 and singular[1] > tolerance * singular[0]:
        raise ValueError("birth matrix must have rank at most one")
    amplitude = (step * singular[0]).sqrt()
    return amplitude * right_t[0], amplitude * left[:, 0]


def main() -> None:
    e = torch.eye(2, dtype=torch.float64)
    z = torch.outer(e[0], e[0])
    print("float64 n=2: radial unit tangent Z=e1 e1^T at X=sigma e1 e1^T")
    print("sigma       factor lift norm    g_Q-unit LMO X norm    birth factor norm")
    for sigma in (1., 1e-4, 1e-8, 1e-12):
        u = d = math.sqrt(sigma) * e[0]
        du, dd = horizontal_lift(u, d, z)
        velocity = quotient_unit_descent(u, d, z)
        ub, db = balanced_birth(-z, step=sigma)
        print(f"{sigma:.1e}     {torch.cat((du,dd)).norm().item():.6e}       "
              f"{velocity.norm().item():.6e}          {torch.cat((ub,db)).norm().item():.6e}")


if __name__ == "__main__":
    main()
