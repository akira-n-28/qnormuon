"""Research-only polar derivative/decision bounds for diagnostic Gram factors.

The bounds are exact for the measured real matrices if the supplied residual
norms are exact. ``rounding_allowance`` states the additional conventional
fp64 operation-model allowance; this is not interval arithmetic or a proof
about undocumented CUDA reduction order. Uncertain decisions require SVD.
"""
from dataclasses import dataclass
import math

import torch

from experiments.smooth_gram import diagnose, gamma
from qnormuon.coupled_solver import SmoothDual


def _fro(x):
    return float(torch.linalg.vector_norm(x))


def _round(norm_scale, count):
    return gamma(count) * norm_scale


@dataclass
class PolarPosterior:
    accepted: bool
    reason: str
    polar_error: float = math.inf
    h_error: float = math.inf
    sigma_lower: float = 0.
    base_error: float = math.inf
    measurements: dict | None = None


@dataclass
class AdaptiveDecomposition:
    backend: str
    reason: str
    left: torch.Tensor
    singular: torch.Tensor
    right: torch.Tensor
    polar: torch.Tensor
    posterior: PolarPosterior | None


def adaptive_decompose(b, decision, *, guard=1e-4):
    """Research-only single-evaluation Gram/SVD selector.

    ``decision`` consumes (GramDiagnostic, PolarPosterior) and returns
    ``(safe, reason)`` after checking the actual curvature/CG/Armijo or
    certificate margin. An undecided decision recomputes a full SVD.
    The caller must not reuse a Gram result for a different multiplier.
    """
    gram=diagnose(b,guard=guard)
    posterior=polar_posterior(b,gram)
    if posterior.accepted:
        safe,reason=decision(gram,posterior)
        if safe:
            return AdaptiveDecomposition("gram_validated_research", "decision_certified",
                gram.left,gram.singular,gram.right,gram.polar,posterior)
    else:
        reason=posterior.reason
    left,singular,right=torch.linalg.svd(b,full_matrices=False)
    return AdaptiveDecomposition("full_svd_decomposition_fallback",reason,
                                 left,singular,right,left@right,posterior)


def polar_posterior(b, gram):
    """Bound true polar/H errors via a nearby *exact* polar factorization.

    Let Pt,Ht be computed factors and Q0=polar(Pt). If ||Pt'Pt-I||<=d<1,
    ||Q0-Pt||_F <= ||Pt||_F d/[sqrt(1-d)(1+sqrt(1-d))].
    B0=Q0 Ht has polar Q0 provided Ht is symmetric positive definite.
    Along B0+t(B-B0), sigma_min >= mu0-||B-B0||_2. Integrating the
    full-rank polar derivative bound ||Dpolar[E]||_F<=2||E||_F/mu yields
    ||polar(B)-Q0||_F <= 2 e0/(mu0-e0). No individual eigenvector matching.
    """
    if not gram.accepted:
        return PolarPosterior(False, "gram_structural_failure")
    p = gram.polar
    v = gram.right.T
    h = (v * gram.singular) @ gram.right
    h = .5 * (h + h.T)
    m, n = b.shape
    eye = torch.eye(n, dtype=b.dtype, device=b.device)
    defect = _fro(p.T @ p - eye)
    if not math.isfinite(defect) or defect >= 1:
        return PolarPosterior(False, "gram_structural_failure")
    # Lower eigenvalue of V diag(s) V' from V'V defect. Use Frobenius
    # orthogonality residual as a conservative operator-norm bound.
    v_defect = _fro(v.T @ v-eye)
    mu0 = float(gram.singular.min()) * max(0., 1-v_defect)
    h_upper = float(gram.singular.max())*(1+v_defect)
    b_upper = gram.measurements["singular_max_upper"]
    q_correction = _fro(p)*defect/(math.sqrt(1-defect)*(1+math.sqrt(1-defect)))
    recon = _fro(b-p@h)
    # Conventional GEMM/reduction allowance for the measured reconstruction.
    rounding = _round(_fro(b)+_fro(p)*h_upper, max(m,n)*n+1)
    e0 = recon + q_correction*h_upper + rounding
    if not math.isfinite(e0) or e0 >= mu0 or mu0 <= 0:
        return PolarPosterior(False, "gram_derivative_bound_failed",
                              base_error=e0, sigma_lower=0.)
    eta_p = q_correction + 2*e0/(mu0-e0)
    # H=P'B, Ht=Q0'B0, hence this is an exact triangle inequality.
    eta_h = eta_p*b_upper + e0
    mu = max(0., mu0-e0)
    return PolarPosterior(True, "posterior_bound_available", eta_p, eta_h,
                          mu, e0, dict(orthogonality=defect,
                                       reconstruction=recon, rounding=rounding,
                                       nearby_sigma_lower=mu0,
                                       nearby_residual_upper=e0,
                                       h_upper=h_upper, b_upper=b_upper))


def derivative_error_bound(b, e, y, gram, posterior):
    """Return ||Y_candidate-Dpolar_B[E]||_F upper bound.

    Exact derivative Y* is characterized uniquely by tangent symmetry,
    H Omega+Omega H=P'E-E'P, and (I-PP')Y*H=(I-PP')E.
    We measure those residuals at Pt,Ht,Y and inflate for certified
    ||P-Pt||<=eta_p and ||H-Ht||<=eta_h. For the true equations, the
    skew Sylvester inverse has norm <=1/(2 sigma_min(H)); the normal
    equation inverse has norm <=1/sigma_min(H). These components are
    orthogonal. Rounding allowances are conventional-model assumptions.
    """
    if not posterior.accepted:
        return math.inf, {"reason": posterior.reason}
    p = gram.polar
    v = gram.right.T
    h = .5*((v*gram.singular)@gram.right + gram.right.T @ (gram.singular[:,None]*gram.right))
    m, n = b.shape
    omega = .5*(p.T@y-y.T@p)
    tangent = _fro(p.T@y+y.T@p)
    syl = _fro(h@omega+omega@h-(p.T@e-e.T@p))
    yh_minus_e = y@h-e
    normal = _fro(yh_minus_e-p@(p.T@yh_minus_e))
    eta_p, eta_h, mu = posterior.polar_error, posterior.h_error, posterior.sigma_lower
    yn, en, hn = _fro(y), _fro(e), posterior.measurements["h_upper"]
    # Inflate measured residuals for rounded products/reductions.
    round_scale = en + yn*(hn+1)
    rounding = _round(round_scale, max(m,n)*n+1)
    tangent_true = tangent + 2*eta_p*yn + rounding
    syl_true = (syl + 2*eta_h*_fro(omega)
                + 2*(hn+eta_h)*eta_p*yn + 2*eta_p*en + rounding)
    projector_error = eta_p*(2+eta_p)
    normal_true = (normal + projector_error*_fro(yh_minus_e)
                   + eta_h*yn + rounding)
    tangential = tangent_true/2 + syl_true/(2*mu)
    normal_component = normal_true/mu
    upper = math.hypot(tangential, normal_component)
    return upper, dict(tangent=tangent, sylvester=syl, normal=normal,
                       tangent_true_upper=tangent_true,
                       sylvester_true_upper=syl_true,
                       normal_true_upper=normal_true,
                       rounding=rounding, tangential_error=tangential,
                       normal_error=normal_component)


def gram_derivative(b, e, gram):
    ev = type("Factors", (), dict(left=gram.left[None],
                                  singular=gram.singular[None],
                                  right=gram.right[None]))
    return SmoothDual(None,None,None).polar_derivative(ev,e[None])[0]


def interval_sign(value, radius):
    if not math.isfinite(value) or not math.isfinite(radius) or radius < 0:
        return "uncertain"
    if value-radius > 0:
        return "positive"
    if value+radius < 0:
        return "negative"
    return "uncertain"


def armijo_interval(current, trial, slope, *, alpha, c=1e-4,
                    rounding_allowance=0.):
    """Classify the current production Armijo inequality by intervals.

    Intervals are inclusive and must bound exact current/trial nuclear values
    and the exact directional slope. The production rounding allowance is
    retained; an overlapping interval is never called an accept or reject.
    """
    cl,cu=current; tl,tu=trial; sl,su=slope
    rl,ru=(rounding_allowance if isinstance(rounding_allowance,tuple)
           else (rounding_allowance,rounding_allowance))
    if not all(math.isfinite(x) for x in (cl,cu,tl,tu,sl,su,rl,ru)):
        return "gram_armijo_ambiguous"
    if not (cl<=cu and tl<=tu and sl<=su and 0<=rl<=ru and alpha>=0 and c>=0):
        raise ValueError("invalid Armijo intervals")
    if tu <= cl+c*alpha*sl+rl:
        return "accept"
    if tl > cu+c*alpha*su+ru:
        return "reject"
    return "gram_armijo_ambiguous"


def cg_residual_upper(approx_residual, rhs_error, alpha_errors):
    """Bound true CG residual for x=sum(alpha_i*p_i), A~=A+errors.

    ``alpha_errors`` is the list |alpha_i|*||HVP_error(p_i)||. It is
    additive even though CG recurrence may be nonorthogonal under errors.
    """
    return approx_residual + rhs_error + sum(alpha_errors)
