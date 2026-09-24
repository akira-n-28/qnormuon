"""QNorMuon research prototype."""

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
