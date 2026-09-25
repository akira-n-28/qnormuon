"""Regular quotient spectral norm, independent of any direction selection."""
import torch

from experiments.horizontal_spectral import horizontal_project, horizontal_residual
from experiments.zero_stratum import balanced_factors, stable_norm


def pair_spectral_norm(pair):
    return torch.linalg.svdvals(pair)[:, 0].max()


def tangent_norm(u, d, pair):
    if bool((u.norm(dim=1) == 0).any()) or bool((d.norm(dim=1) == 0).any()):
        raise ValueError("regular rows required")
    tolerance = 100 * torch.finfo(pair.dtype).eps * max(1., float(pair.norm()))
    w = (u.square() + d.square()).sum(1).sqrt()
    if float((horizontal_residual(u, d, pair).abs() / w).max()) > tolerance:
        raise ValueError("supply a horizontal tangent, not an arbitrary factor velocity")
    return pair_spectral_norm(pair)


def raw_class_norm(u, d, velocity):
    """Pullback seminorm on raw velocities; norm after quotienting the gauge.

    Uses the derivative of the balanced section, including horizontal projection.
    It is not infimum spectral norm over arbitrary vertical representatives.
    """
    balanced = [balanced_factors(ui, di) for ui, di in zip(u, d)]
    uc, dc = torch.stack([v[0] for v in balanced]), torch.stack([v[1] for v in balanced])
    root_h = torch.stack([stable_norm(ui).sqrt() / stable_norm(di).sqrt() for ui, di in zip(u, d)])
    frame = torch.stack((velocity[0] / root_h[:, None], velocity[1] * root_h[:, None]))
    return tangent_norm(uc, dc, horizontal_project(uc, dc, frame))
