"""Isolated rank-deficiency research prototypes, not an optimizer backend.

See docs/RANK_DEFICIENT_THEORY.md. A supplied rank is a caller's mathematical
assumption, not an inferred numerical fact. Thresholding is a separate operation
with explicit tolerances. None of these functions is imported by qnormuon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def _matrix(a: Tensor) -> None:
    if a.ndim != 2 or min(a.shape) == 0:
        raise ValueError("expected a nonempty matrix")
    if a.dtype not in (torch.float32, torch.float64):
        raise ValueError("research reference requires explicit float32 or float64 inputs")
    if not torch.isfinite(a).all():
        raise ValueError("matrix must be finite")


def _rank(a: Tensor, rank: int) -> None:
    _matrix(a)
    if not isinstance(rank, int) or not 0 <= rank <= min(a.shape):
        raise ValueError("rank must be an integer between zero and min(shape)")


def weighted_matrix(m: Tensor, x: Tensor) -> Tensor:
    """K^(1/2) M without centering: global shifts matter for fixed ridge."""
    _matrix(m)
    if x.shape != (m.shape[0],) or x.dtype != m.dtype or x.device != m.device:
        raise ValueError("x must match matrix rows, dtype and device")
    a = (0.5 * x).exp()[:, None] * m
    if not torch.isfinite(a).all():
        raise ValueError("nonfinite weighted matrix")
    return a


@torch.no_grad()
def partial_polar(a: Tensor, *, rank: int) -> Tensor:
    """Leading-rank partial polar; equals the exact map if rank(A) is supplied.

    No rank tolerance is hidden here. If positive singular values are omitted,
    this is instead the partial polar of a truncated approximation to A. Floating
    point cannot certify that the omitted singular values are mathematically zero.
    """
    _rank(a, rank)
    if rank == 0:
        return torch.zeros_like(a)
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    if s[rank - 1] <= 0:
        raise ValueError("requested rank includes a zero computed singular value")
    return u[:, :rank] @ vh[:rank]


@torch.no_grad()
def threshold_partial_polar(a: Tensor, *, atol: float, rtol: float) -> tuple[Tensor, int]:
    """Discard sigma <= max(atol, rtol*sigma_max); returns the chosen rank."""
    _matrix(a)
    if not all(math.isfinite(t) and t >= 0 for t in (atol, rtol)):
        raise ValueError("explicit tolerances must be finite and nonnegative")
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    keep = s > max(atol, rtol * s[0].item())
    rank = int(keep.sum().item())
    return u[:, keep] @ vh[keep], rank


@torch.no_grad()
def active_basis(m: Tensor, *, rank: int) -> Tensor:
    """Orthonormal right basis, to be fixed while x varies for fixed M."""
    _rank(m, rank)
    if rank == 0:
        return m.new_zeros((m.shape[1], 0))
    _, s, vh = torch.linalg.svd(m, full_matrices=False)
    if s[rank - 1] <= 0:
        raise ValueError("requested active rank includes a zero singular value")
    return vh[:rank].T.contiguous()


def active_logdet(m: Tensor, x: Tensor, basis: Tensor) -> Tensor:
    """log det((M V)^T K (M V)); V must be an orthonormal active basis.

    For rank zero use det(empty)=1, logdet(empty)=0. Autograd is for x with
    M and the supplied basis fixed, not for discrete rank/basis selection.
    """
    a = weighted_matrix(m, x)
    if basis.ndim != 2 or basis.shape[0] != m.shape[1]:
        raise ValueError("basis must have shape [n,r]")
    if basis.shape[1] == 0:
        return 0.0 * x.sum()
    b = a @ basis
    # QR avoids explicitly squaring the condition number via a Gram matrix.
    _, triangular = torch.linalg.qr(b, mode="reduced")
    diagonal = triangular.diag().abs()
    if (diagonal == 0).any():
        raise ValueError("active restriction is singular")
    return 2 * diagonal.log().sum()


def pseudo_logdet(m: Tensor, x: Tensor, *, rank: int) -> Tensor:
    """log pdet(M^T K M) via the assumed positive singular values of K^1/2 M."""
    _rank(m, rank)
    a = weighted_matrix(m, x)
    if rank == 0:
        return 0.0 * x.sum()
    s = torch.linalg.svdvals(a)[:rank]
    if (s <= 0).any():
        raise ValueError("assumed positive spectrum contains a zero")
    return 2 * s.log().sum()


@torch.no_grad()
def active_polar(m: Tensor, x: Tensor, basis: Tensor) -> Tensor:
    """polar(K^1/2 M V) V^T, identical to partial polar when V spans row(M)."""
    a = weighted_matrix(m, x)
    if basis.shape[1] == 0:
        return torch.zeros_like(m)
    return partial_polar(a @ basis, rank=basis.shape[1]) @ basis.T


def _ridge(lam: float) -> None:
    if not math.isfinite(lam) or lam <= 0:
        raise ValueError("ridge must be finite and strictly positive")


def ridge_logdet(m: Tensor, x: Tensor, *, lam: float) -> Tensor:
    """log det(M^T K M + lam I), including the null-space constant."""
    _ridge(lam)
    a = weighted_matrix(m, x)
    n = m.shape[1]
    # Augmented QR is a determinant identity, not artificial-rank selection.
    augmented = torch.cat((a, math.sqrt(lam) * torch.eye(n, dtype=m.dtype, device=m.device)))
    _, triangular = torch.linalg.qr(augmented, mode="reduced")
    return 2 * triangular.diag().abs().log().sum()


@torch.no_grad()
def ridge_polar(a: Tensor, *, lam: float) -> Tensor:
    """Smooth contraction A(A^T A + lam I)^(-1/2), not a partial isometry."""
    _matrix(a)
    _ridge(lam)
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    factors = s / torch.hypot(s, torch.full_like(s, math.sqrt(lam)))
    return (u * factors) @ vh


def leverage(p: Tensor) -> Tensor:
    return p.square().sum(dim=1)


@dataclass
class BalanceResult:
    x: Tensor
    up: Tensor
    down: Tensor
    target: float
    max_residual: float
    converged: bool


@torch.no_grad()
def balance_active_pair(
    mu: Tensor, md: Tensor, *, ranks: tuple[int, int],
    steps: int = 1000, lr: float = 0.5, tolerance: float = 1e-10,
    x0: Tensor | None = None,
) -> BalanceResult:
    """Fixed-M research solve with fixed active spaces and reported residual.

    No feasibility oracle and no online momentum semantics. Nonconvergence is
    reported, not interpreted as proof of infeasibility.
    """
    if mu.shape != md.shape or mu.dtype != md.dtype or mu.device != md.device:
        raise ValueError("paired momenta must match shape, dtype and device")
    if steps < 0 or not 0 < lr < 2 or tolerance < 0:
        raise ValueError("require steps >= 0, 0 < lr < 2, tolerance >= 0")
    vu, vd = (active_basis(a, rank=r) for a, r in zip((mu, md), ranks))
    x = mu.new_zeros(mu.shape[0]) if x0 is None else x0.clone()
    target = sum(ranks) / mu.shape[0]
    for iteration in range(steps + 1):
        x -= x.mean()
        pu, pd = active_polar(mu, x, vu), active_polar(md, x, vd)
        residual = leverage(pu) + leverage(pd) - target
        error = residual.abs().max().item()
        if error <= tolerance or iteration == steps:
            return BalanceResult(x, pu, pd, target, error, error <= tolerance)
        x -= lr * residual
    raise AssertionError("unreachable")


def main() -> None:
    """A compact deterministic rank-boundary sweep; all matrices float64 3x2."""
    print("float64 3x2: A(t)=diag(1,t) with a zero third row; ridge=1e-4")
    print("t          rank  ||P-P(0)||  trace(PP^T)  ||P_ridge-P_ridge(0)||  logpdet")
    zero = torch.tensor([[1., 0.], [0., 0.], [0., 0.]], dtype=torch.float64)
    p0, ridge0 = partial_polar(zero, rank=1), ridge_polar(zero, lam=1e-4)
    for t in (1., 1e-2, 1e-6, 1e-12, 0., -1e-12):
        a = zero.clone()
        a[1, 1] = t
        rank = 1 + int(t != 0)
        p, ridge = partial_polar(a, rank=rank), ridge_polar(a, lam=1e-4)
        f = pseudo_logdet(a, a.new_zeros(3), rank=rank)
        print(f"{t: .1e}  {rank}     {(p-p0).norm().item():.3e}    "
              f"{leverage(p).sum().item():.6f}       {(ridge-ridge0).norm().item():.3e}          {f.item():.6f}")


if __name__ == "__main__":
    main()
