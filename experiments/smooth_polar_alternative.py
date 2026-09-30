"""Research-only fp64 QDWH polar paths for the smooth coupled dual.

This module does not change the production SmoothDual backend.  The weighted
Halley coefficients and QR update follow Nakatsukasa, Bai & Gygi (2010),
doi:10.1137/090774999.  A posterior failure requests the independent thin SVD.
The posterior is an operation-model screen, not a proof of identical Newton
or line-search branches on every CUDA implementation.
"""
from dataclasses import dataclass
import math
import time

import torch


@dataclass
class PolarResult:
    accepted: bool
    reason: str
    polar: torch.Tensor | None
    h: torch.Tensor | None
    singular: torch.Tensor | None
    eigenvectors: torch.Tensor | None
    iterations: int
    qr_iterations: int
    cholesky_iterations: int
    measurements: dict


def _coefficients(ell):
    # Stable real-valued form of the dynamically weighted Halley coefficients.
    d = (4.0 * (1.0 - ell * ell) / ell**4) ** (1.0 / 3.0)
    root = math.sqrt(1.0 + d)
    a = root + 0.5 * math.sqrt(8.0 - 4.0 * d +
                                8.0 * (2.0 - ell * ell) / (ell * ell * root))
    b = (a - 1.0) ** 2 / 4.0
    c = a + b - 1.0
    return a, b, c


def _qdwh_unit(x, ell, *, max_iterations=10, qr_switch=100.0, profile=False):
    """Return the polar of a scaled tall/square X; no current SVD is used."""
    m, n = x.shape
    eye = torch.eye(n, dtype=x.dtype, device=x.device)
    qr_count = chol_count = 0
    timing = {"qr_seconds": 0.0, "cholesky_solve_seconds": 0.0}
    def clock():
        if profile and x.is_cuda:
            torch.cuda.synchronize(x.device)
        return time.perf_counter() if profile else 0.0
    for iteration in range(max_iterations):
        a, b, c = _coefficients(ell)
        if c > qr_switch:
            start = clock()
            stack = torch.cat((math.sqrt(c) * x, eye), dim=0)
            q, _ = torch.linalg.qr(stack, mode="reduced")
            q1, q2 = q[:m], q[m:]
            nxt = (b / c) * x + ((a - b / c) / math.sqrt(c)) * (q1 @ q2.T)
            timing["qr_seconds"] += clock() - start
            qr_count += 1
        else:
            start = clock()
            gram = eye + c * (x.T @ x)
            gram = (gram + gram.T) * 0.5
            factor = torch.linalg.cholesky(gram)
            solved = torch.cholesky_solve(x.T, factor).T
            nxt = (b / c) * x + (a - b / c) * solved
            timing["cholesky_solve_seconds"] += clock() - start
            chol_count += 1
        # The exact recurrence approaches one from below; roundoff can put
        # its computed lower estimate a few ulps above one.
        ell = min(1.0, ell * (a + b * ell * ell) / (1.0 + c * ell * ell))
        x = nxt
        if not bool(torch.isfinite(x).all()):
            return None, iteration + 1, qr_count, chol_count, "nonfinite_iteration", timing
        # The final posterior, not this stop condition, decides whether Q is usable.
        defect = float(torch.linalg.matrix_norm(x.T @ x - eye, ord="fro"))
        if defect <= 10.0 * n * torch.finfo(x.dtype).eps:
            return x, iteration + 1, qr_count, chol_count, "converged", timing
    return x, max_iterations, qr_count, chol_count, "iteration_budget", timing


@torch.no_grad()
def diagnose(b, *, route="tall", guard=1e-4, max_iterations=10, profile=False):
    """Try QDWH, construct H without a Gram square root, then screen defects.

    `route='qr_square'` first reduces the tall B by QR and applies QDWH to R.
    The initial lower singular bound is guard/(2 sqrt(n)) for X=B/||B||F.
    It is valid for residuals whose true rcond is at least guard/2, including
    the entire production-admitted domain.  Below that domain the posterior
    may reject; it must not be used as evidence of a safe residual.
    """
    if b.ndim != 2 or b.dtype != torch.float64 or b.shape[0] < b.shape[1]:
        raise ValueError("expected a tall fp64 matrix")
    if route not in ("tall", "qr_square"):
        raise ValueError("route must be tall or qr_square")
    m, n = b.shape
    data = {"route": route}

    def reject(reason, iterations=0, qr_count=0, chol_count=0):
        return PolarResult(False, reason, None, None, None, None,
                           iterations, qr_count, chol_count, data)

    amplitude = float(b.abs().max())
    if amplitude == 0.0 or not math.isfinite(amplitude):
        return reject("zero_or_nonfinite_residual")
    scaled = b / amplitude
    alpha = float(torch.linalg.matrix_norm(scaled, ord="fro"))
    if not math.isfinite(alpha) or alpha == 0.0:
        return reject("invalid_scale")
    x = scaled / alpha
    ell = guard / (2.0 * math.sqrt(n))
    def clock():
        if profile and b.is_cuda:
            torch.cuda.synchronize(b.device)
        return time.perf_counter() if profile else 0.0
    try:
        if route == "qr_square":
            reduction_start = clock()
            tall_q, r = torch.linalg.qr(x, mode="reduced")
            data["reduction_qr_seconds"] = clock() - reduction_start
            square_p, it, qr_count, chol_count, status, timing = _qdwh_unit(
                r, ell, max_iterations=max_iterations, profile=profile)
            q = None if square_p is None else tall_q @ square_p
        else:
            q, it, qr_count, chol_count, status, timing = _qdwh_unit(
                x, ell, max_iterations=max_iterations, profile=profile)
    except (torch.linalg.LinAlgError, OverflowError, ValueError) as exc:
        data["exception"] = str(exc)
        return reject("qdwh_linear_algebra_failure")
    data.update(status=status, iterations=it, qr_iterations=qr_count,
                cholesky_iterations=chol_count, scale=amplitude * alpha)
    if profile:
        data.update(timing)
    if q is None or status != "converged":
        return reject("qdwh_" + status, it, qr_count, chol_count)
    # Form H from the polar relation, not sqrt(B.T B).  Work at scaled
    # amplitude so extreme representable overall scales do not overflow.
    h_start = clock()
    qt_b = q.T @ scaled
    h_scaled = (qt_b + qt_b.T) * 0.5
    symmetry_defect = float(torch.linalg.matrix_norm(qt_b - qt_b.T, ord="fro"))
    if profile:
        data["h_form_seconds"] = clock() - h_start
    try:
        evd_start = clock()
        sigma_scaled, v = torch.linalg.eigh(h_scaled)
        if profile:
            data["h_evd_seconds"] = clock() - evd_start
    except torch.linalg.LinAlgError:
        return reject("h_eigh_failure", it, qr_count, chol_count)
    sigma_scaled, v = sigma_scaled.flip(0), v.flip(1)
    orth = float(torch.linalg.matrix_norm(q.T @ q - torch.eye(n, dtype=b.dtype,
                                          device=b.device), ord="fro"))
    reconstruction = float(torch.linalg.matrix_norm(scaled - q @ h_scaled,
                                                    ord="fro")) / float(scaled.norm())
    smallest = float(sigma_scaled[-1])
    largest = float(sigma_scaled[0])
    rcond = smallest / largest if largest > 0 else 0.0
    unit = torch.finfo(torch.float64).eps / 2.0
    gamma = lambda k: (k * unit) / (1.0 - k * unit)
    # Screening envelope from QR/iteration, GEMM, and EVD rounding. It is
    # intentionally conservative, but not an interval-arithmetic guarantee.
    envelope = 64.0 * (gamma(m) + gamma(n))
    data.update(orthogonality=orth, reconstruction=reconstruction,
                symmetry_defect=symmetry_defect / float(scaled.norm()),
                smallest_scaled=smallest, largest_scaled=largest,
                rcond=rcond, posterior_envelope=envelope,
                nuclear_scaled=float(torch.trace(h_scaled)))
    if not all(math.isfinite(z) for z in (orth, reconstruction, symmetry_defect,
                                          smallest, largest, rcond)):
        return reject("nonfinite_posterior", it, qr_count, chol_count)
    if smallest <= 0:
        return reject("h_not_spd", it, qr_count, chol_count)
    if orth > envelope:
        return reject("polar_orthogonality", it, qr_count, chol_count)
    if reconstruction > envelope:
        return reject("polar_reconstruction", it, qr_count, chol_count)
    if data["symmetry_defect"] > envelope:
        return reject("h_symmetry", it, qr_count, chol_count)
    # Any rcond close enough to the guard to be ambiguous under this measured
    # envelope is sent to the independent full SVD.
    if rcond - envelope <= guard:
        return reject("rcond_uncertain", it, qr_count, chol_count)
    singular = sigma_scaled * amplitude
    h = h_scaled * amplitude
    if not bool(torch.isfinite(singular).all()) or not bool(torch.isfinite(h).all()):
        return reject("unrepresentable_h", it, qr_count, chol_count)
    return PolarResult(True, "posterior_passed_research_only", q, h, singular,
                       v, it, qr_count, chol_count, data)


@torch.no_grad()
def polar_derivative(result, e):
    """Derivative from P,H via an SPD Sylvester solve; no SVD or m² projector."""
    if not result.accepted:
        raise ValueError("cannot differentiate a rejected polar result")
    q, v, sigma = result.polar, result.eigenvectors, result.singular
    qte = q.T @ e
    skew = qte - qte.T
    rhs = v.T @ skew @ v
    omega = v @ (rhs / (sigma[:, None] + sigma[None, :])) @ v.T
    normal = e - q @ qte
    normal_hinv = ((normal @ v) / sigma) @ v.T
    return q @ omega + normal_hinv
