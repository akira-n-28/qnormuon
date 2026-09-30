"""Diagnostic fp64 Gram factors for the smooth residual; not a production backend.

The reported one-sided rcond interval uses the standard fp64 dot-product model,
not interval arithmetic or a guarantee about undocumented GPU reductions.
No positive eigenvalue is truncated. Rejection never changes the production SVD.
"""
from dataclasses import dataclass
import math
import time

import torch


def gamma(k):
    unit = torch.finfo(torch.float64).eps / 2
    if k * unit >= 1:
        raise ValueError("floating-point bound is not representable")
    return k * unit / (1 - k * unit)


@dataclass
class GramDiagnostic:
    accepted: bool
    reason: str
    left: torch.Tensor | None
    singular: torch.Tensor | None
    right: torch.Tensor | None
    polar: torch.Tensor | None
    measurements: dict


@torch.no_grad()
def diagnose(b, *, guard=1e-4):
    """Attempt scaled Gram/EVD and return factors only after posterior checks.

    For X=fl(B/s), the standard dot-product bound is
    ||fl(X.T X)-X.T X||_2 <= gamma_m ||X||_F^2 plus symmetry rounding.
    Division contributes at most u/(1-u)||X||_F to ||B/s-X||_2.
    If V is computed EVD, delta=||V.T V-I||_F and rho=||CV-VL||_F,
    then ||C-VLV.T||_2 <= ||C||_F delta+rho sqrt(1+delta).
    Weyl and congruence give lower/upper singular-value bounds. These
    inequalities justify a conservative rcond lower bound, conditional on
    the standard operation model. Other posterior thresholds below are
    deliberately diagnostic envelopes; passing them is not by itself a
    proof of HVP/line-search equivalence.
    """
    if b.ndim != 2 or b.dtype != torch.float64 or b.shape[0] < b.shape[1]:
        raise ValueError("expected a tall fp64 matrix")
    if not 0 < guard < 1:
        raise ValueError("guard must be in (0,1)")
    m, n = b.shape
    start = time.perf_counter()
    amplitude = float(b.abs().max())
    if not math.isfinite(amplitude) or amplitude == 0:
        return GramDiagnostic(False, "nonfinite_or_zero_residual", None, None, None, None, {})
    x = b / amplitude
    if not bool(torch.isfinite(x).all()):
        return GramDiagnostic(False, "scaled_residual_nonfinite", None, None, None, None, {})
    torch.cuda.synchronize() if b.is_cuda else None
    t0 = time.perf_counter()
    c = x.T @ x
    c = (c + c.T) * .5
    torch.cuda.synchronize() if b.is_cuda else None
    t1 = time.perf_counter()
    try:
        eigen, v = torch.linalg.eigh(c)
    except torch.linalg.LinAlgError:
        return GramDiagnostic(False, "eigh_failure", None, None, None, None, {})
    torch.cuda.synchronize() if b.is_cuda else None
    t2 = time.perf_counter()
    eigen, v = eigen.flip(0), v.flip(1)
    eye = torch.eye(n, dtype=b.dtype, device=b.device)
    delta = float(torch.linalg.matrix_norm(v.T @ v - eye, ord="fro"))
    rho = float(torch.linalg.matrix_norm(c @ v - v * eigen, ord="fro"))
    c_fro = float(torch.linalg.matrix_norm(c, ord="fro"))
    x_fro = float(torch.linalg.matrix_norm(x, ord="fro"))
    unit = torch.finfo(torch.float64).eps / 2
    # Inflate measured reductions; the second term covers the scaled division.
    x_fro_upper = x_fro / (1 - gamma(m*n))
    c_fro_upper = c_fro / (1 - gamma(n*n))
    division_error = unit/(1-unit)*x_fro_upper
    gram_round = (gamma(m)*x_fro_upper*x_fro_upper
                  + unit*c_fro_upper + 2*x_fro_upper*division_error
                  + division_error*division_error)
    evd_error = (c_fro_upper*delta + rho*math.sqrt(1+delta)) if delta < 1 else math.inf
    total_error = gram_round + evd_error
    lam_min, lam_max = float(eigen[-1]), float(eigen[0])
    lower_sq = lam_min*(1-delta) - total_error if lam_min > 0 and delta < 1 else -math.inf
    upper_sq = max(lam_max, 0)*(1+delta) + total_error
    sigma_min_lower = max(0., math.sqrt(lower_sq)-division_error) if lower_sq > 0 else 0.
    sigma_max_upper = math.sqrt(upper_sq)+division_error if upper_sq >= 0 else math.inf
    rcond_lower = sigma_min_lower/sigma_max_upper if sigma_max_upper > 0 else 0.
    max_lower_sq = lam_max*(1-delta)-total_error
    min_upper_sq = max(lam_min,0)*(1+delta)+total_error
    sigma_max_lower = max(0.,math.sqrt(max_lower_sq)-division_error) if max_lower_sq>0 else 0.
    sigma_min_upper = math.sqrt(min_upper_sq)+division_error if min_upper_sq>=0 else math.inf
    rcond_upper = min(1.,sigma_min_upper/sigma_max_lower) if sigma_max_lower>0 else 1.
    measurements = dict(shape=[m,n], amplitude=amplitude, eigen_min=lam_min,
        eigen_max=lam_max, eigen_orthogonality=delta, eigen_residual=rho,
        gram_round_bound=gram_round, eigensystem_error_bound=evd_error,
        singular_min_lower=amplitude*sigma_min_lower,
        singular_max_upper=amplitude*sigma_max_upper,
        rcond_lower=rcond_lower,rcond_upper=rcond_upper,
        gram_seconds=t1-t0, eigh_seconds=t2-t1)
    def reject(reason):
        measurements["total_seconds"] = time.perf_counter()-start
        return GramDiagnostic(False, reason, None, None, None, None, measurements)
    if not all(math.isfinite(vv) for vv in (delta,rho,gram_round,evd_error,upper_sq)):
        return reject("nonfinite_posterior")
    if delta >= 1:
        return reject("eigenvectors_not_invertible")
    if lam_min < 0:
        # Even a roundoff-sized negative value cannot be inverted without
        # changing the represented positive spectrum: reject, do not clamp.
        return reject("negative_eigenvalue_roundoff" if lam_min >= -gram_round
                      else "negative_eigenvalue_beyond_bound")
    if lam_min == 0:
        return reject("zero_eigenvalue")
    if lower_sq <= 0 or rcond_lower <= guard:
        return reject("rcond_lower_not_above_guard")
    sigma_scaled = eigen.sqrt()
    # For PSD Lambda and ||V.T V-I||<=delta, the ordered eigenvalues of
    # V Lambda V.T lie between (1-delta)*lambda_i and (1+delta)*lambda_i.
    # Weyl plus the measured Gram/EVD allowance gives per-sigma intervals.
    singular_lower = ((eigen*(1-delta)-total_error).clamp_min(0).sqrt()
                      - division_error).clamp_min(0)
    singular_upper = (eigen*(1+delta)+total_error).clamp_min(0).sqrt()+division_error
    left = (x @ v) / sigma_scaled
    right = v.T
    polar = left @ right
    h = (v * sigma_scaled) @ right
    left_orth = float(torch.linalg.matrix_norm(left.T @ left-eye, ord="fro"))
    polar_orth = float(torch.linalg.matrix_norm(polar.T @ polar-eye, ord="fro"))
    recon = float(torch.linalg.matrix_norm(x-(left*sigma_scaled)@right,ord="fro"))/x_fro
    polar_recon = float(torch.linalg.matrix_norm(x-polar@h,ord="fro"))/x_fro
    measurements.update(left_orthogonality=left_orth, polar_orthogonality=polar_orth,
                        reconstruction_residual=recon, polar_reconstruction_residual=polar_recon,
                        nuclear_value=amplitude*float(sigma_scaled.sum()),
                        nuclear_lower=amplitude*float(singular_lower.sum()),
                        nuclear_upper=amplitude*float(singular_upper.sum()),
                        rcond_estimate=float(sigma_scaled[-1]/sigma_scaled[0]),
                        singular_max_interval_width=amplitude*float((singular_upper-singular_lower).max()))
    torch.cuda.synchronize() if b.is_cuda else None
    measurements["reconstruction_seconds"] = time.perf_counter()-t2
    measurements["total_seconds"] = time.perf_counter()-start
    if not all(math.isfinite(vv) for vv in (left_orth,polar_orth,recon,polar_recon)):
        return reject("nonfinite_reconstruction")
    # Compare measured defects with operation-model envelopes. This is a
    # diagnostic screen, not a complete forward-error proof for Dpolar.
    orth_envelope = (gamma(m+n)*x_fro_upper*x_fro_upper + total_error) / lam_min
    # Reconstruction uses X@V, division/multiplication by sigma, and a
    # second n-term product. The eigensystem's measured orthogonality defect
    # also contributes to X V V.T-X. gamma_(mn) inflates the measured norms;
    # this is an operation-count bound, not a fitted tolerance.
    reconstruction_envelope = (delta + 2*gamma(n)*math.sqrt(1+delta)
                               + 4*unit + gamma(m*n))
    measurements.update(orthogonality_envelope=orth_envelope,
                        reconstruction_envelope=reconstruction_envelope)
    if left_orth > orth_envelope or polar_orth > orth_envelope:
        return reject("orthogonality_posterior_failed")
    if recon > reconstruction_envelope or polar_recon > reconstruction_envelope:
        return reject("reconstruction_posterior_failed")
    singular = sigma_scaled * amplitude
    if not bool(torch.isfinite(singular).all()) or not bool(torch.isfinite(polar).all()):
        return reject("unrepresentable_factors")
    return GramDiagnostic(True, "posterior_passed_research_only", left, singular,
                          right, polar, measurements)
