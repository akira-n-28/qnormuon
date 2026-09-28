"""Coupled production-v0 and preserved historical QNorMuon prototype."""

from .coupled_solver import SolverConfig, solve_coupled
from .optimizer import QuotientSpectralOptimizer, SwiGLUPair, regular_canonicalize

from .core import (
    PairDiagnostics,
    QNorMuon,
    balance_shared_metric,
    canonical_gradients,
    canonical_pair,
    canonicalize_pair_,
    exact_polar,
    gauge_metric,
    paired_leverage,
    quotient_polar_update,
)

__all__ = [
    "SolverConfig", "solve_coupled", "QuotientSpectralOptimizer",
    "SwiGLUPair", "regular_canonicalize",
    "PairDiagnostics",
    "QNorMuon",
    "balance_shared_metric",
    "canonical_gradients",
    "canonical_pair",
    "canonicalize_pair_",
    "exact_polar",
    "gauge_metric",
    "paired_leverage",
    "quotient_polar_update",
]
