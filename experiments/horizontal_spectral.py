"""Isolated K=I horizontal spectral LMO research; never imported by qnormuon.

Two deliberately separate numerical contracts:
* solve_lmo: ADMM with a primal/dual objective certificate, no rank decisions
  and no claim to select the minimum-norm member of a nonunique optimum.
* minimum_norm_on_dual_face: Dykstra projection of zero onto the optimal face,
  given a dual optimizer and its explicitly supplied residual ranks.

Finite tolerances are research solver tolerances, not a production rank policy.
All matrices are CPU float32/float64; a pair has shape (2, m, n).
"""

from dataclasses import dataclass
import json

import torch


def _validate(u, d, a):
    if u.ndim != 2 or d.shape != u.shape or a.shape != (2,) + u.shape:
        raise ValueError("expected matching m x n weights and a 2 x m x n pair")
    if min(u.shape) < 1 or u.shape[0] < u.shape[1]:
        raise ValueError("reference expects m >= n >= 1")
    for z in (u, d, a):
        if z.dtype != u.dtype or z.device.type != "cpu":
            raise ValueError("common CPU dtype required")
        if z.dtype not in (torch.float32, torch.float64) or not torch.isfinite(z).all():
            raise ValueError("finite float32/float64 inputs required")
    if not torch.allclose(u.norm(dim=1), d.norm(dim=1), rtol=1e-5, atol=0):
        raise ValueError("weights must already be balanced")


def horizontal_residual(u, d, p):
    return (u * p[0]).sum(1) - (d * p[1]).sum(1)


def adjoint(u, d, lam):
    return torch.stack((lam[:, None] * u, -lam[:, None] * d))


def multipliers(u, d, p):
    denom = (u.square() + d.square()).sum(1)
    nonzero = denom > 0
    out = torch.zeros_like(denom)
    out[nonzero] = horizontal_residual(u, d, p)[nonzero] / denom[nonzero]
    return out


def horizontal_project(u, d, p):
    return p - adjoint(u, d, multipliers(u, d, p))


def ball_project(p):
    left, values, right = torch.linalg.svd(p, full_matrices=False)
    return (left * values.clamp(max=1).unsqueeze(-2)) @ right


def partial_pair(a, ranks):
    result = []
    for b, rank in zip(a, ranks):
        if not 0 <= rank <= b.shape[1]:
            raise ValueError("rank outside matrix dimensions")
        left, _, right = torch.linalg.svd(b, full_matrices=False)
        result.append(left[:, :rank] @ right[:rank])
    return torch.stack(result)


def dual_value(u, d, a, lam):
    return torch.linalg.svdvals(a - adjoint(u, d, lam)).sum()


def smooth_dual_gradient(u, d, a, lam):
    """Caller must establish both residuals have full column rank."""
    b = a - adjoint(u, d, lam)
    return -horizontal_residual(u, d, partial_pair(b, (a.shape[-1],) * 2))


def diagnostics(u, d, a, p):
    sv = torch.linalg.svdvals(p)
    eye = torch.eye(p.shape[-1], dtype=p.dtype)
    return {
        "objective": float((a * p).sum()),
        "horizontal": float(horizontal_residual(u, d, p).abs().max()),
        "spectral_excess": max(0.0, float(sv.max()) - 1.0),
        "stiefel_defect": float(torch.linalg.matrix_norm(p.transpose(-2, -1) @ p - eye).max()),
        "squared_norm": float(p.square().sum()),
    }


@dataclass
class LMOResult:
    pair: torch.Tensor
    lam: torch.Tensor
    gap: float
    iterations: int
    converged: bool


@torch.no_grad()
def solve_lmo(u, d, a, *, tolerance=1e-10, max_iterations=20000, rho=1.0):
    """General primary solve; output is horizontal and radially made feasible.

    A small objective gap does NOT certify direction accuracy near rank loss,
    nor minimum Frobenius norm on a nontrivial exposed face.
    """
    _validate(u, d, a)
    if tolerance <= 0 or rho <= 0 or max_iterations < 1:
        raise ValueError("positive solver controls required")
    # Scaling the objective improves absolute stopping semantics. It does not
    # introduce a rank cutoff and is undone in lambda and the reported gap.
    scale = float(a.norm())
    if scale == 0:
        return LMOResult(torch.zeros_like(a), torch.zeros(u.shape[0], dtype=u.dtype), 0., 0, True)
    objective = a / scale
    z, y = torch.zeros_like(a), torch.zeros_like(a)
    converged = False
    for iteration in range(1, max_iterations + 1):
        p = horizontal_project(u, d, z - y + objective / rho)
        previous = z
        z = ball_project(p + y)
        y = y + p - z
        if iteration % 10 == 0 or iteration == max_iterations:
            lam = multipliers(u, d, objective - rho * y)
            feasible = p / max(1.0, float(torch.linalg.svdvals(p).max()))
            gap_normalized = float(dual_value(u, d, objective, lam) - (objective * feasible).sum())
            residual = max(float((p - z).norm()), float(rho * (z - previous).norm()))
            if gap_normalized <= tolerance and residual <= tolerance:
                converged = True
                break
    return LMOResult(feasible, scale * lam, scale * gap_normalized, iteration, converged)


class ExposedFace:
    """Product of nuclear-norm subdifferentials at explicit residual ranks."""

    def __init__(self, b, ranks):
        if len(ranks) != 2:
            raise ValueError("two ranks required")
        self.parts = []
        for matrix, rank in zip(b, ranks):
            if not 0 <= rank <= matrix.shape[1]:
                raise ValueError("rank outside matrix dimensions")
            left, values, right = torch.linalg.svd(matrix, full_matrices=True)
            # Rank is supplied, not inferred: positive omitted singular values
            # would change the exact problem. Caller owns that assumption.
            self.parts.append((left[:, :rank] @ right[:rank], left[:, rank:], right[rank:].T))

    def project(self, pair):
        result = []
        for value, (base, left, right) in zip(pair, self.parts):
            if right.shape[1] == 0:
                result.append(base)
            else:
                block = left.T @ value @ right
                result.append(base + left @ ball_project(block) @ right.T)
        return torch.stack(result)


@torch.no_grad()
def minimum_norm_on_dual_face(u, d, a, lam, *, ranks, tolerance=1e-11, max_iterations=30000):
    """Project zero onto H intersect (partial-polar + null contractions).

    Assumes lambda is an exact dual optimizer and ranks are the exact residual
    ranks. Reports nonconvergence rather than certifying an infeasible face.
    Dykstra's corrections are essential for the minimum-norm contract.
    """
    _validate(u, d, a)
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("positive solver controls required")
    face = ExposedFace(a - adjoint(u, d, lam), ranks)
    p, correction_h, correction_f = (torch.zeros_like(a) for _ in range(3))
    converged = False
    for iteration in range(1, max_iterations + 1):
        previous = p
        q = horizontal_project(u, d, p + correction_h)
        correction_h = p + correction_h - q
        p = face.project(q + correction_f)
        correction_f = q + correction_f - p
        residual = float(horizontal_residual(u, d, p).abs().max())
        if residual <= tolerance and float((p - previous).norm()) <= tolerance:
            converged = True
            break
    gap = float(dual_value(u, d, a, lam) - (a * p).sum())
    return LMOResult(p, lam.clone(), gap, iteration, converged)


def fractional_example(dtype=torch.float64):
    """Optimal residual ranks (2,1), unique selected down spectrum (1,.5)."""
    u = torch.tensor([[1., 0.], [3. ** .5 / 2, .5], [0., 1.]], dtype=dtype)
    d = torch.tensor([[1., 0.], [0., 1.], [0., 1.]], dtype=dtype)
    a = torch.tensor([[[1., 0.], [0., 1.], [0., 0.]],
                      [[1., 0.], [0., 0.], [0., 0.]]], dtype=dtype)
    expected = a.clone()
    expected[1, 1, 1] = .5
    return u, d, a, expected


def random_example(seed=901, m=7, n=3, dtype=torch.float64):
    rng = torch.Generator().manual_seed(seed)
    u = torch.randn(m, n, generator=rng, dtype=dtype)
    d = torch.randn(m, n, generator=rng, dtype=dtype)
    u, d = u / u.norm(dim=1, keepdim=True), d / d.norm(dim=1, keepdim=True)
    a = torch.randn(2, m, n, generator=rng, dtype=dtype)
    return u, d, a


def main():
    for name, data in (("random", random_example()), ("fractional", fractional_example()[:3])):
        u, d, a = data
        solved = solve_lmo(u, d, a)
        separate = partial_pair(a, (a.shape[-1],) * 2 if name == "random" else (2, 1))
        projected = horizontal_project(u, d, separate)
        repaired = projected / max(1., float(torch.linalg.svdvals(projected).max()))
        results = {key: diagnostics(u, d, a, value) for key, value in
                   (("separate", separate), ("coupled", solved.pair),
                    ("post_project", projected), ("project_and_rescale", repaired))}
        results["solver"] = {"gap": solved.gap, "iterations": solved.iterations,
                             "converged": solved.converged,
                             "residual_singular_values": torch.linalg.svdvals(a - adjoint(u, d, solved.lam)).tolist()}
        print(json.dumps({"case": name, "dtype": str(a.dtype), "shape": list(a.shape[1:]), "results": results}))


if __name__ == "__main__":
    main()
