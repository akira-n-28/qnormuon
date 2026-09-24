"""QMuon / QNorMuon prototype for paired SwiGLU up/down projections.

Core idea
---------
For up_proj U in R^{m x n} and down_proj Wd in R^{n x m}, write
D = Wd^T in R^{m x n}.  The SwiGLU branch has the exact positive diagonal
'gauge' symmetry

    U -> C U,     D -> C^{-1} D,

for diagonal C > 0.  QMuon performs spectral normalization in gauge-canonical
coordinates rather than directly in the raw parameterization.

This file intentionally uses an exact SVD polar factor by default.  That keeps
the mathematical tests independent of any particular fast polar solver.  A
faster solver can be substituted later without changing the quotient geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer


@torch.no_grad()
def exact_polar(a: Tensor) -> Tensor:
    """Return the rectangular polar factor U V^T using a thin SVD.

    For a tall full-column-rank matrix a in R^{m x n}, m >= n, the result P
    satisfies P^T P = I.  SVD also gives a well-defined closest partial
    isometry when the input is rank deficient.
    """
    if a.ndim != 2:
        raise ValueError(f"exact_polar expects a matrix, got shape {tuple(a.shape)}")
    u, _, vh = torch.linalg.svd(a, full_matrices=False)
    return u @ vh


@torch.no_grad()
def gauge_metric(up: Tensor, down: Tensor, eps: float = 1e-12) -> Tensor:
    """Return h_i = ||up_i|| / ||down_col_i|| for a SwiGLU pair.

    Args:
        up:   [m, n] up-projection weight.
        down: [n, m] down-projection weight.

    Returns:
        h: [m] positive diagonal metric coefficients.
    """
    _check_pair_shapes(up, down)
    d = down.transpose(0, 1)
    ru = torch.linalg.vector_norm(up, dim=1).clamp_min(eps)
    rd = torch.linalg.vector_norm(d, dim=1).clamp_min(eps)
    return ru / rd


@torch.no_grad()
def canonical_pair(up: Tensor, down: Tensor, eps: float = 1e-12) -> Tuple[Tensor, Tensor]:
    """Return the unique positive balanced representative of the gauge orbit.

    The returned matrices satisfy row_norm(U_bar) == row_norm(D_bar), where
    D_bar = down_bar.T, up to floating-point error.
    """
    h = gauge_metric(up, down, eps=eps)
    hs = torch.sqrt(h)
    up_bar = up / hs[:, None]
    d_bar = down.transpose(0, 1) * hs[:, None]
    return up_bar, d_bar.transpose(0, 1)


@torch.no_grad()
def canonicalize_pair_(up: Tensor, down: Tensor, eps: float = 1e-12) -> None:
    """Gauge-transform a SwiGLU pair in-place to equal per-neuron norms.

    This leaves the represented SwiGLU branch exactly unchanged in exact
    arithmetic because U -> C U and down -> down C^{-1}.
    """
    _check_pair_shapes(up, down)
    d = down.transpose(0, 1)
    ru = torch.linalg.vector_norm(up, dim=1).clamp_min(eps)
    rd = torch.linalg.vector_norm(d, dim=1).clamp_min(eps)
    c = torch.sqrt(rd / ru)
    up.mul_(c[:, None])
    down.mul_((1.0 / c)[None, :])


@torch.no_grad()
def canonical_gradients(
    up: Tensor,
    down: Tensor,
    grad_up: Tensor,
    grad_down: Tensor,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Map Euclidean gradients into gauge-invariant canonical coordinates.

    Returns (G_u_c, G_d_c, h), with D = down.T and G_d referring to D.

        G_u_c = H^{1/2} G_u
        G_d_c = H^{-1/2} G_D

    Under U -> C U, D -> C^{-1}D, the Euclidean gradients transform as
    G_u -> C^{-1}G_u and G_D -> C G_D, so both canonical gradients are
    invariant.
    """
    _check_pair_shapes(up, down)
    if grad_up.shape != up.shape or grad_down.shape != down.shape:
        raise ValueError("gradient shapes must match parameter shapes")

    h = gauge_metric(up, down, eps=eps)
    hs = torch.sqrt(h)
    gu = hs[:, None] * grad_up
    gd_t = (1.0 / hs)[:, None] * grad_down.transpose(0, 1)
    return gu, gd_t, h


@torch.no_grad()
def quotient_polar_update(
    up: Tensor,
    down: Tensor,
    grad_up: Tensor,
    grad_down: Tensor,
    *,
    x: Tensor | None = None,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pure one-step QMuon/QNorMuon update map with no momentum.

    This function is useful for mathematical tests.  If x is provided, it is a
    shared intrinsic row reweighting used *before* the two polar factorizations.

    Returns:
        delta_up       [m, n]
        delta_down     [n, m]
        P_up           [m, n]
        P_down_t       [m, n]
        h              [m]
    """
    gu, gd, h = canonical_gradients(up, down, grad_up, grad_down, eps=eps)
    m, n = gu.shape
    if m < n:
        raise ValueError("prototype currently assumes tall matrices m >= n")

    if x is None:
        khalf = torch.ones(m, dtype=gu.dtype, device=gu.device)
    else:
        if x.shape != (m,):
            raise ValueError(f"x must have shape {(m,)}, got {tuple(x.shape)}")
        # Mean-centering fixes the irrelevant global scaling of K.
        xc = x - x.mean()
        khalf = torch.exp(0.5 * xc).to(dtype=gu.dtype, device=gu.device)

    pu = exact_polar(khalf[:, None] * gu)
    pd = exact_polar(khalf[:, None] * gd)

    hs = torch.sqrt(h)
    delta_up = hs[:, None] * pu
    delta_d_t = (1.0 / hs)[:, None] * pd
    delta_down = delta_d_t.transpose(0, 1)
    return delta_up, delta_down, pu, pd, h


@torch.no_grad()
def paired_leverage(pu: Tensor, pd: Tensor) -> Tensor:
    """Return row-wise paired leverage ||P_u[i]||^2 + ||P_d[i]||^2."""
    if pu.shape != pd.shape or pu.ndim != 2:
        raise ValueError("pu and pd must be same-shaped matrices")
    return pu.square().sum(dim=1) + pd.square().sum(dim=1)


@torch.no_grad()
def balance_shared_metric(
    mu: Tensor,
    md: Tensor,
    *,
    steps: int = 200,
    lr: float = 0.2,
    x0: Tensor | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Solve the shared paired-leverage balancing problem by gradient descent.

    This is primarily a reference solver / test utility for the convex problem

        min_x logdet(Mu^T K Mu) + logdet(Md^T K Md)
              - (2 n / m) 1^T x,
        K = diag(exp(x)).

    The gradient equals paired_leverage - 2n/m.

    Returns:
        x, P_u(x), P_d(x)
    """
    if mu.ndim != 2 or md.shape != mu.shape:
        raise ValueError("mu and md must be same-shaped matrices")
    m, n = mu.shape
    if m < n:
        raise ValueError("prototype currently assumes m >= n")

    if x0 is None:
        x = torch.zeros(m, dtype=mu.dtype, device=mu.device)
    else:
        x = x0.clone().to(dtype=mu.dtype, device=mu.device)

    target = 2.0 * n / m
    pu = pd = None
    for _ in range(steps):
        x = x - x.mean()
        kh = torch.exp(0.5 * x)
        pu = exact_polar(kh[:, None] * mu)
        pd = exact_polar(kh[:, None] * md)
        err = paired_leverage(pu, pd) - target
        x = x - lr * err

    x = x - x.mean()
    kh = torch.exp(0.5 * x)
    pu = exact_polar(kh[:, None] * mu)
    pd = exact_polar(kh[:, None] * md)
    return x, pu, pd


@dataclass
class PairDiagnostics:
    paired_leverage_cv: float
    paired_leverage_max_error: float
    polar_defect_up: float
    polar_defect_down: float
    x_std: float


class QNorMuon(Optimizer):
    """Prototype optimizer for paired SwiGLU up/down matrices.

    Parameters are supplied as pairs ``(up_proj.weight, down_proj.weight)``.
    The expected shapes are [m,n] and [n,m].

    This optimizer intentionally handles only the paired matrices.  Biases,
    norms, embeddings, gate_proj, lm_head, etc. should be optimized separately
    (e.g. with AdamW) in a full training setup.

    Important: the default SVD polar is a research/reference implementation,
    not a throughput-optimized kernel.
    """

    def __init__(
        self,
        pairs: Sequence[Tuple[Tensor, Tensor]],
        *,
        lr: float = 1e-3,
        beta1: float = 0.95,
        balance_lr: float = 0.1,
        balance_beta: float = 0.9,
        eps: float = 1e-12,
        canonicalize_weights: bool = False,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must lie in [0,1)")
        if not 0.0 <= balance_beta < 1.0:
            raise ValueError("balance_beta must lie in [0,1)")
        if balance_lr < 0:
            raise ValueError("balance_lr must be nonnegative")

        flat: List[Tensor] = []
        self._pairs: List[Tuple[Tensor, Tensor]] = []
        for up, down in pairs:
            _check_pair_shapes(up, down)
            flat.extend([up, down])
            self._pairs.append((up, down))

        defaults = dict(
            lr=lr,
            beta1=beta1,
            balance_lr=balance_lr,
            balance_beta=balance_beta,
            eps=eps,
            canonicalize_weights=canonicalize_weights,
        )
        super().__init__(flat, defaults)
        # One param group only in the prototype, so pair semantics stay explicit.

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        beta1 = group["beta1"]
        bal_lr = group["balance_lr"]
        bal_beta = group["balance_beta"]
        eps = group["eps"]
        do_canon = group["canonicalize_weights"]

        for up, down in self._pairs:
            if up.grad is None or down.grad is None:
                continue
            _check_pair_shapes(up, down)

            gu, gd, h = canonical_gradients(up, down, up.grad, down.grad, eps=eps)
            m, n = gu.shape
            if m < n:
                raise ValueError("QNorMuon prototype assumes m >= n")

            state = self.state[up]
            if len(state) == 0:
                state["momentum_up"] = torch.zeros_like(gu)
                state["momentum_down_t"] = torch.zeros_like(gd)
                state["x"] = torch.zeros(m, dtype=gu.dtype, device=gu.device)
                state["balance_momentum"] = torch.zeros(m, dtype=gu.dtype, device=gu.device)
                state["step"] = 0

            mu = state["momentum_up"]
            md = state["momentum_down_t"]
            x = state["x"]
            q = state["balance_momentum"]

            mu.mul_(beta1).add_(gu, alpha=1.0 - beta1)
            md.mul_(beta1).add_(gd, alpha=1.0 - beta1)

            x.sub_(x.mean())
            kh = torch.exp(0.5 * x)
            pu = exact_polar(kh[:, None] * mu)
            pd = exact_polar(kh[:, None] * md)

            ell = paired_leverage(pu, pd)
            target = 2.0 * n / m
            err = ell - target
            q.mul_(bal_beta).add_(err, alpha=1.0 - bal_beta)
            x.add_(q, alpha=-bal_lr)
            x.sub_(x.mean())

            hs = torch.sqrt(h)
            delta_up = hs[:, None] * pu
            delta_d_t = (1.0 / hs)[:, None] * pd

            up.add_(delta_up, alpha=-lr)
            down.add_(delta_d_t.transpose(0, 1), alpha=-lr)

            if do_canon:
                canonicalize_pair_(up, down, eps=eps)

            state["step"] += 1

        return loss

    @torch.no_grad()
    def diagnostics(self) -> List[PairDiagnostics]:
        """Return current geometry diagnostics for each initialized pair."""
        out: List[PairDiagnostics] = []
        for up, down in self._pairs:
            state = self.state.get(up, {})
            if not state:
                continue
            mu = state["momentum_up"]
            md = state["momentum_down_t"]
            x = state["x"] - state["x"].mean()
            kh = torch.exp(0.5 * x)
            pu = exact_polar(kh[:, None] * mu)
            pd = exact_polar(kh[:, None] * md)
            ell = paired_leverage(pu, pd)
            m, n = pu.shape
            target = 2.0 * n / m
            eye = torch.eye(n, dtype=pu.dtype, device=pu.device)
            defect_u = torch.linalg.matrix_norm(pu.T @ pu - eye).item()
            defect_d = torch.linalg.matrix_norm(pd.T @ pd - eye).item()
            out.append(
                PairDiagnostics(
                    paired_leverage_cv=(ell.std() / ell.mean()).item(),
                    paired_leverage_max_error=(ell - target).abs().max().item(),
                    polar_defect_up=defect_u,
                    polar_defect_down=defect_d,
                    x_std=x.std().item(),
                )
            )
        return out


def _check_pair_shapes(up: Tensor, down: Tensor) -> None:
    if up.ndim != 2 or down.ndim != 2:
        raise ValueError("up and down must both be matrices")
    m, n = up.shape
    if down.shape != (n, m):
        raise ValueError(
            f"expected down shape {(n, m)} for up shape {(m, n)}, "
            f"got {tuple(down.shape)}"
        )
