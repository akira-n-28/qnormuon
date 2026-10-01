"""Research-only epsilon-LMO audit; no production admission or direction budget.

Reuse the conditional gamma-model feasible proxy and residual uncertainty from
the direction study. Primary value certification needs neither positive rank
nor proximity to a reference direction. Numerical rank is a SEPARATE result.
All factors and radial metadata must match the same recovered candidate.
"""
import math
import torch
from experiments.full_rank_direction import direction_posterior, gamma


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
