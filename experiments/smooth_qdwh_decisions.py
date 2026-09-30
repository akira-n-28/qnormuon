"""Research-only QDWH decision posterior. Production always uses full SVD.

All bounds are exact inequalities for the represented matrices when residual
norms are evaluated exactly. ``gamma`` allowances assume conventional fp64
operation error, not directed rounding or a cuSOLVER-specific error theorem.
An undecided branch requests independent full SVD.
"""
from dataclasses import dataclass
import math

import torch

from experiments.smooth_polar_alternative import diagnose, polar_derivative


def gamma(k):
    unit = 2.0 ** -53
    x = k * unit
    return x / (1.0 - x) if x < 1.0 else math.inf


def fro(x):
    return float(torch.linalg.vector_norm(x))


@dataclass
class QDWHPosterior:
    accepted: bool
    reason: str
    polar_error: float = math.inf
    h_error: float = math.inf
    sigma_min_lower: float = 0.0
    sigma_max_upper: float = math.inf
    rcond_lower: float = 0.0
    rcond_upper: float = math.inf
    nuclear_lower: float = 0.0
    nuclear_upper: float = math.inf
    nearby_error: float = math.inf
    measurements: dict | None = None


def factor_posterior(b, result, *, guard=1e-4):
    """Bound the exact polar factors by a nearby Q0 Ht factorization.

    With d=||Qt'Qt-I||F<1, Q0=polar(Qt) exists (proof only), and
    ||Q0-Qt||F <= ||Qt||F*d/[sqrt(1-d)*(1+sqrt(1-d))].
    B0=Q0 Ht is exactly polarized if Ht is SPD.  Weyl and the full-rank
    polar derivative bound give ||Q(B)-Qt||F <= qcorr+2e/(mu0-e), where
    e>=||B-B0||F and mu0<=sigma_min(Ht). No runtime SVD is used.
    """
    if not result.accepted:
        reason={"qdwh_iteration_budget":"qdwh_nonconverged",
                "polar_reconstruction":"qdwh_factorization_failed",
                "polar_orthogonality":"qdwh_orthogonality_failed",
                "h_not_spd":"qdwh_h_not_spd",
                "rcond_uncertain":"qdwh_rcond_uncertain"}.get(
                    result.reason,"qdwh_structural_failure")
        return QDWHPosterior(False, reason)
    q, h, s, v = result.polar, result.h, result.singular, result.eigenvectors
    m, n = b.shape
    eye = torch.eye(n, dtype=b.dtype, device=b.device)
    d = fro(q.T @ q - eye)
    vd = fro(v.T @ v - eye)
    eig_res = fro(h @ v - v * s[None, :])
    if not all(map(math.isfinite, (d, vd, eig_res))) or d >= 1 or vd >= 1:
        return QDWHPosterior(False, "qdwh_orthogonality_failed")
    # Exact identity H(I-VV')+(HV-V diag(s))V' bounds H-V diag(s)V'.
    # Weyl and the full-rank congruence bound give eigenvalue enclosures.
    eig_err = fro(h) * vd + eig_res * math.sqrt(1 + vd)
    eig_round = gamma(n*n+1) * (fro(h) + fro(v) * fro(s))
    eig_err += eig_round
    mu0 = float(s.min()) * (1 - vd) - eig_err
    mu0_upper = float(s.min()) * (1 + vd) + eig_err
    hmax = float(s.max()) * (1 + vd) + eig_err
    hmax_lower = float(s.max()) * (1 - vd) - eig_err
    if mu0 <= 0:
        return QDWHPosterior(False, "qdwh_h_not_spd")
    qcorr = fro(q) * d / (math.sqrt(1-d) * (1+math.sqrt(1-d)))
    fact = fro(b-q@h)
    # Standard-model allowance for the measured tall product and reduction.
    round_fact = gamma(m*n+1) * (fro(b) + fro(q)*hmax)
    e = fact + qcorr*hmax + round_fact
    if not math.isfinite(e) or e >= mu0:
        return QDWHPosterior(False, "qdwh_factorization_failed")
    sigma_min_lower = mu0-e
    sigma_max_upper = hmax+e
    sigma_max_lower = hmax_lower-e
    if sigma_max_lower <= 0:
        return QDWHPosterior(False, "qdwh_rcond_uncertain")
    rlo = sigma_min_lower/sigma_max_upper
    rhi = (mu0_upper+e)/sigma_max_lower
    if rlo <= guard:
        return QDWHPosterior(False, "qdwh_rcond_uncertain",
                             rcond_lower=rlo, rcond_upper=rhi,
                             sigma_min_lower=sigma_min_lower,
                             sigma_max_upper=sigma_max_upper,
                             nearby_error=e)
    eta_q = qcorr + 2*e/sigma_min_lower
    # H=Q'B. The second term is measured directly, exploiting QDWH's
    # Q'B construction instead of propagating a Gram-square-root error.
    eta_h = eta_q*sigma_max_upper + fro(q.T@b-h)
    nuclear = float(torch.trace(h))
    nuclear_radius = math.sqrt(n)*e + gamma(n+1)*float(h.diag().abs().sum())
    return QDWHPosterior(True, "qdwh_posterior_available", eta_q, eta_h,
        sigma_min_lower, sigma_max_upper, rlo, rhi,
        max(0., nuclear-nuclear_radius), nuclear+nuclear_radius, e,
        dict(orthogonality=d, eigen_orthogonality=vd,
             eigen_residual=eig_res, eigen_error=eig_err,
             factorization=fact, factor_rounding=round_fact,
             orthogonalization_correction=qcorr,
             nuclear_radius=nuclear_radius))


def derivative_error_bound(b, e, y, result, posterior):
    """Conditional Frobenius bound from tangent/Sylvester/normal residuals.

    For exact Q,H, Y*=Dpolar_B[E] is uniquely specified by the three
    residual equations. The inverse skew Sylvester norm is <=1/(2*mu),
    and the normal inverse norm is <=1/mu. Measured residuals are inflated
    using posterior ||Q-Qt||F and ||H-Ht||F bounds.
    """
    if not posterior.accepted:
        return math.inf, {"reason": posterior.reason}
    q,h=result.polar,result.h
    m,n=b.shape
    omega=.5*(q.T@y-y.T@q)
    tangent=fro(q.T@y+y.T@q)
    syl=fro(h@omega+omega@h-(q.T@e-e.T@q))
    yh_e=y@h-e
    normal=fro(yh_e-q@(q.T@yh_e))
    eq,eh,mu=posterior.polar_error,posterior.h_error,posterior.sigma_min_lower
    yn,en=fro(y),fro(e)
    roundoff=gamma(m*n+1)*(en+yn*(posterior.sigma_max_upper+1))
    t=tangent+2*eq*yn+roundoff
    s=syl+2*eh*fro(omega)+2*(posterior.sigma_max_upper+eh)*eq*yn+2*eq*en+roundoff
    nn=normal+eq*(2+eq)*fro(yh_e)+eh*yn+roundoff
    bound=math.hypot(t/2+s/(2*mu),nn/mu)
    return bound,dict(tangent=tangent,sylvester=syl,normal=normal,
                      tangent_upper=t,sylvester_upper=s,normal_upper=nn,
                      roundoff=roundoff)


def row_operator_upper(u,d):
    """Account for rounded row whitening rather than assuming ||L||=1."""
    n=u.shape[1]
    row=(u.square()+d.square()).sum(1)
    return math.sqrt(float(row.max())/(1-gamma(2*n+1)))


def hvp_with_error(b,e,results,posteriors,u,d):
    ys=[polar_derivative(r,side_e) for r,side_e in zip(results,e)]
    errs=[derivative_error_bound(side_b,side_e,y,r,p)[0]
          for side_b,side_e,y,r,p in zip(b,e,ys,results,posteriors)]
    from qnormuon.coupled_solver import horizontal_residual
    return -horizontal_residual(u,d,torch.stack(ys)),row_operator_upper(u,d)*math.hypot(*errs)


def interval_sign(value,radius):
    if not math.isfinite(value) or not math.isfinite(radius) or radius<0:
        return "uncertain"
    if value-radius>0:return "positive"
    if value+radius<0:return "negative"
    return "uncertain"


def cg_stop_certified(residual_upper, gradient_lower, *, damping):
    """True damped residual must meet the unchanged truncated-CG target."""
    if damping<=0 or gradient_lower<0 or not all(map(math.isfinite,(residual_upper,gradient_lower))):
        return False
    target=min(.1,math.sqrt(max(gradient_lower,1e-16)))*gradient_lower
    return residual_upper<=target


def certificate_interval_decision(dual,primal,*,rcond_lower,feasible,
                                  gap_tolerance=3e-5):
    """Conservative acceptance for a *feasible QDWH-derived* primal pair.

    An uncertain rejection is deliberately not inferred from this helper:
    rejecting the QDWH-derived pair does not prove the SVD-derived pair fails.
    """
    dl,du=dual;pl,pu=primal
    if not feasible or rcond_lower<=1e-4 or not all(map(math.isfinite,(dl,du,pl,pu))):
        return "qdwh_certificate_ambiguous"
    if dl>du or pl>pu:
        raise ValueError("invalid objective intervals")
    if pl<=0 or dl<=0:
        return "qdwh_certificate_ambiguous"
    denominator_lower=max(pl,dl,torch.finfo(torch.float64).tiny)
    denominator_upper=max(pu,du,torch.finfo(torch.float64).tiny)
    gap_upper=max(0.,du-pl)/denominator_lower
    signed_lower=(dl-pu)/(denominator_lower if dl-pu<0 else denominator_upper)
    if gap_upper<=gap_tolerance and signed_lower>=-1e-10:
        return "accept"
    return "qdwh_certificate_ambiguous"


def armijo_interval(current,trial,slope,*,alpha,c=1e-4,rounding=(0.,0.)):
    cl,cu=current;tl,tu=trial;sl,su=slope;rl,ru=rounding
    if not all(math.isfinite(z) for z in (cl,cu,tl,tu,sl,su,rl,ru)):
        return "qdwh_armijo_ambiguous"
    if tu<=cl+c*alpha*sl+rl:return "accept"
    if tl>cu+c*alpha*su+ru:return "reject"
    return "qdwh_armijo_ambiguous"


def adaptive_decompose(b, decision, *, guard=1e-4):
    """Research-only selector; decision must certify its actual branch."""
    candidate=diagnose(b,guard=guard)
    post=factor_posterior(b,candidate,guard=guard)
    reason=post.reason
    if post.accepted:
        safe,reason=decision(candidate,post)
        if safe:
            return "qdwh_decision_certified",reason,candidate,post
    u,s,vt=torch.linalg.svd(b,full_matrices=False)
    return "full_svd_decomposition_fallback",reason,(u,s,vt),post
