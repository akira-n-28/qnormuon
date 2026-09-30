"""Research-only, complete fp64 thin SVD by reduced QR and square SVD.

In exact arithmetic B=Qr R and R=Ur diag(s) Vt imply
B=(Qr Ur) diag(s) Vt. Qr and Ur have orthonormal columns, so the
returned factors have exactly the existing SmoothDual thin-SVD contract.
Nuclear value, rcond, polar, gradient and the existing polar derivative/HVP
need no algorithm changes. Floating-point factors are compared independently
with direct tall SVD; repeated-subspace bases themselves are not compared.

No Gram formation, truncation, new posterior or solver branch policy occurs.
The scoped adapter changes only decomposition and restores production on exit.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch
import qnormuon.coupled_solver as solver


def decompose(b, backend="qr_reduced_svd", *, driver=None):
    if b.dtype != torch.float64 or b.ndim < 2 or b.shape[-2] < b.shape[-1]:
        raise ValueError("tall fp64 residuals required")
    if driver not in (None, "gesvdj", "gesvd"):
        raise ValueError("only full-accuracy SVD drivers permitted")
    kwargs = {} if driver is None else {"driver": driver}
    if backend == "tall_svd":
        return torch.linalg.svd(b, full_matrices=False, **kwargs)
    if backend != "qr_reduced_svd":
        raise ValueError("unknown research decomposition backend")
    qr, r = torch.linalg.qr(b, mode="reduced")
    ur, singular, vt = torch.linalg.svd(r, full_matrices=False, **kwargs)
    return qr @ ur, singular, vt


def evaluate(problem, lam, *, backend="qr_reduced_svd", driver=None):
    b = problem.a - solver.adjoint(problem.u, problem.d, lam)
    left, singular, right = decompose(b, backend, driver=driver)
    problem.counts.svd_matrices += 2
    problem.counts.polar_matrices += 2
    pair = left @ right
    ratio = singular[:, -1] / singular[:, 0].clamp_min(torch.finfo(singular.dtype).tiny)
    return solver.Evaluation(float(singular.sum()),
        -solver.horizontal_residual(problem.u, problem.d, pair),
        pair, left, singular, right, float(ratio.min()), lam)


@contextmanager
def research_backend(backend="qr_reduced_svd", *, driver=None, trace=None):
    """Single-process study scope; never changes persisted production config."""
    parent = solver.SmoothDual

    class ResearchDual(parent):
        def evaluate(self, lam):
            ev = evaluate(self, lam, backend=backend, driver=driver)
            if trace is not None:
                trace.append(ev)
            return ev

    with patch.object(solver, "SmoothDual", ResearchDual):
        yield
