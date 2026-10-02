"""Conditional fp64 certification for the opt-in full-rank primary LMO.

Bounds are conditional on normal finite fp64 gamma-model arithmetic; these are
not directed intervals. Frozen input tensors define the exact real problem.
See docs/FULL_RANK_DIRECTION_ADMISSION.md for the proof and feasible proxy.
No reference optimum or direction budget is an input to this module.
"""
import math
import torch
from . import coupled_solver as cs

UNIT = 2.**-53


def gamma(k):
    if k*UNIT >= 1:
        raise ValueError('reduction too long for the fp64 model')
    return k*UNIT/(1-k*UNIT)


def norm_upper(x, dims=None):
    length = x.numel() if dims is None else math.prod(x.shape[d] for d in dims)
    return x.norm()/(1-gamma(length+3)) if dims is None else x.norm(dim=dims)/(1-gamma(length+3))


def exact_gap_bound(gap, alpha):
    """Distance to every optimum at an arbitrary full-rank dual iterate."""
    if gap < 0 or alpha <= 0:
        raise ValueError('nonnegative gap and positive sigma_min required')
    return 2*math.sqrt(2*gap/alpha)


@torch.no_grad()
def residual_uncertainty(b, left, singular, right):
    """Distance to orthonormalized-factor matrix, with listed singular values.

    Frobenius bounds control both Frobenius and operator uncertainty. No new
    decomposition; all factors must belong to this same residual evaluation.
    """
    m,n=b.shape[-2:]
    eye=torch.eye(n,device=b.device,dtype=b.dtype)
    uf=norm_upper(left,(-2,-1)); vf=norm_upper(right,(-2,-1))
    sf=norm_upper(singular,(-1,))
    du=(norm_upper(left.mT@left-eye,(-2,-1))+gamma(m)*uf.square()
        +gamma(2)*math.sqrt(n))*(1+gamma(16))
    dv=(norm_upper(right@right.mT-eye,(-2,-1))+gamma(n)*vf.square()
        +gamma(2)*math.sqrt(n))*(1+gamma(16))
    if bool((du>=1).any()) or bool((dv>=1).any()):
        raise ValueError('orthogonality cannot be bounded')
    eu=du/(1+(1-du).sqrt()); ev=dv/(1+(1-dv).sqrt())
    reconstructed=(left*singular.unsqueeze(-2))@right
    # GEMM, the column scaling, and subtraction used to measure the residual.
    rho=(norm_upper(b-reconstructed,(-2,-1))
         +gamma(n+2)*sf*vf*(1+du).sqrt()
         +gamma(2)*(norm_upper(b,(-2,-1))+norm_upper(reconstructed,(-2,-1))))
    eta=(rho+sf*(eu*(1+dv).sqrt()+ev))*(1+gamma(128))
    return eta,dict(left_defect_upper=du.tolist(),right_defect_upper=dv.tolist(),
                    reconstruction_relative=(norm_upper(b-reconstructed,(-2,-1))/norm_upper(b,(-2,-1))).tolist())


@torch.no_grad()
def direction_posterior(u,d,a,lam,p,left,singular,right,radial_metrics):
    """Bound error of the SAME recovered p and SAME original residual factors.

    radial_metrics is the matching gram_upper certificate, before model casts.
    Its spectral_norms precede final componentwise division rounding. Internal
    factors may be passed with singular values rescaled to original units;
    reconstruction against B below also bounds that conversion discrepancy.
    """
    if any(v.dtype!=torch.float64 for v in (u,d,a,lam,p,left,singular,right)):
        raise ValueError('posterior requires fp64 data')
    if radial_metrics['primal_spectral_backend']!='gram_eigh_upper':
        raise ValueError('model-conservative matching Gram radial bounds required')
    m,n=u.shape
    term=cs.adjoint(u,d,lam); b=a-term
    if not all(bool(torch.isfinite(x).all()) for x in (b,p,left,singular,right)):
        raise ValueError('finite normal fp64 data required')
    eta,detail=residual_uncertainty(b,left,singular,right)
    construction=gamma(3)*norm_upper(a.abs()+term.abs(),(-2,-1))*(1+gamma(16))
    eta=eta+construction
    alpha=(singular[:,-1]-eta)*(1-gamma(2))
    dual_side=singular.sum(1)/(1-gamma(n+1))+math.sqrt(n)*eta
    dual_upper=float(dual_side.sum()*(1+gamma(16)))
    # Exact horizontal projection distance: ||L(p)/sqrt(w)||_2.
    weights=(u.square()+d.square()).sum(1)
    wlower=weights/(1+gamma(2*n+3))
    row_abs=(u.abs()*p[0].abs()+d.abs()*p[1].abs()).sum(1)/(1-gamma(2*n+3))
    hrows=(cs.horizontal_residual(u,d,p).abs()+gamma(2*n+3)*row_abs)/wlower.sqrt()
    h=float(norm_upper(hrows)*(1+gamma(16)))
    pf=float(norm_upper(p)); af=float(norm_upper(a))
    division=UNIT/(1-UNIT)*float(norm_upper(p,(-2,-1)).max())
    rho=max(radial_metrics['spectral_norms'])*(1+gamma(4))+division
    scale=max(1.,rho+h)*(1+gamma(4))
    # Avoid cancellation in (1-1/scale) near unit scale.
    delta=(h+(scale-1)/scale*(pf+h))*(1+gamma(32))
    dot=float((a*p).sum())
    dot_round=gamma(2*m*n+2)*float((a.abs()*p.abs()).sum())/(1-gamma(2*m*n+2))
    primal_lower=dot-dot_round-af*delta
    gap=(dual_upper-primal_lower)+gamma(4)*(abs(dual_upper)+abs(primal_lower))
    # A materially negative interval is an inconsistency, never success.
    if gap < 0:
        raise ValueError('inconsistent negative gap upper bound')
    amin=float(alpha.min())
    if amin<=0:
        bound=sharper=math.inf
    else:
        bound=delta+exact_gap_bound(gap,amin)
        # Each support gap is separately nonnegative at the exact feasible Pc.
        bdots=(b*p).sum((-2,-1))
        bdot_round=gamma(m*n+2)*(b.abs()*p.abs()).sum((-2,-1))/(1-gamma(m*n+2))
        support_upper=(dual_side-bdots+bdot_round
                       +construction*norm_upper(p,(-2,-1))
                       +(norm_upper(b,(-2,-1))+construction)*delta)
        if bool((support_upper<0).any()):
            raise ValueError('inconsistent negative side support bound')
        support_upper=support_upper*(1+gamma(32))
        sharper=delta+math.sqrt(2*float((support_upper/alpha).sum()))+math.sqrt(2*gap/amin)
    return dict(alpha_lower=alpha.tolist(),sigma_min=singular[:,-1].tolist(),
        rcond=(singular[:,-1]/singular[:,0]).tolist(),uncertainty=eta.tolist(),
        construction_uncertainty=construction.tolist(),dual_upper=dual_upper,
        primal_lower=primal_lower,gap_upper=gap,feasibility_distance_upper=delta,
        horizontal_projection_distance_upper=h,radial_norm_upper=rho,
        global_direction_error_upper=bound,
        direction_error_upper=min(bound,sharper),two_side_direction_error_upper=sharper,
        normalized_direction_upper=min(bound,sharper)/math.sqrt(2*n),
        model='conditional fp64 gamma model; no directed interval arithmetic',**detail)


@torch.no_grad()
def value_posterior(u, d, a, lam, p, left, singular, right, radial_metrics,
                    tolerance=3e-5):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('finite positive value tolerance required')
    post = direction_posterior(u, d, a, lam, p, left, singular, right,
                               radial_metrics)
    # Reverse nuclear Lipschitz inequality for the same nearby orthonormal
    # factor matrix. eta already includes original residual construction and
    # internal/original conversion uncertainty. No additional decomposition.
    n = u.shape[1]
    spectrum_lower = float(singular.sum()) / (1 + gamma(singular.numel()+1))
    dual_lower = max(0., spectrum_lower - math.sqrt(n)*sum(post['uncertainty']))
    dual_lower = max(0., dual_lower - gamma(16)*(
        spectrum_lower + math.sqrt(n)*sum(post['uncertainty'])))
    # For the exact feasible proxy Pc, weak duality and central symmetry give
    # phi >= |<A,Pc>|. Thus phi is the exact production normalization. Using
    # its LOWER bound certifies this normalized gap without denominator bias.
    relative_upper = post['gap_upper']/dual_lower if dual_lower > 0 else math.inf
    feasible = (radial_metrics['normalized_horizontal_residual'] <= 1e-10
                and radial_metrics['spectral_excess'] <= 1e-12
                and radial_metrics['signed_normalized_gap'] >= -1e-10)
    value_ok = feasible and relative_upper <= tolerance
    full_rank = min(post['alpha_lower']) > 0
    # For positive primal value, regret/v <= gap/phi because v<=phi.
    # gap/primal_lower is also valid but generally looser. A normalized gap
    # below one proves positivity; outside that regime do not use this shortcut.
    regret_over_value = (post['gap_upper']/post['primal_lower']
                         if post['primal_lower'] > 0 else math.inf)
    if relative_upper < 1 and post['primal_lower'] > 0:
        regret_over_value = min(regret_over_value, relative_upper)
    return dict(dual_upper=post['dual_upper'], dual_lower=dual_lower,
        primal_lower=post['primal_lower'], gap_upper=post['gap_upper'],
        normalized_gap_upper=relative_upper,
        relative_support_regret_upper=regret_over_value,
        alpha_lower=post['alpha_lower'], uncertainty=post['uncertainty'],
        sigma_min=post['sigma_min'], rcond=post['rcond'],
        feasibility_distance_upper=post['feasibility_distance_upper'],
        horizontal_projection_distance_upper=post['horizontal_projection_distance_upper'],
        radial_norm_upper=post['radial_norm_upper'],
        value_certified=value_ok, numerical_full_rank=full_rank,
        research_value_rank_pass=value_ok and full_rank,
        tolerance=tolerance, model=post['model'],
        selection_certified=False)
