"""Synchronized observational timings, scoped to one benchmark pair solve.

Nested categories overlap: certificate includes certificate SVDs; CG includes
HVPs; evaluate includes residual SVD/polar. This never changes solver decisions.
"""
from collections import defaultdict
from contextlib import ExitStack
import time
from unittest.mock import patch
import torch
import qnormuon.coupled_solver as cs


class SolverTiming:
    def __init__(self):
        self.seconds=defaultdict(float);self.stack=ExitStack()

    def __enter__(self):
        def install(owner,name,label):
            original=getattr(owner,name)
            def timed(*args,**kw):
                torch.cuda.synchronize();start=time.perf_counter()
                try:return original(*args,**kw)
                finally:
                    torch.cuda.synchronize()
                    self.seconds[label]+=time.perf_counter()-start
            self.stack.enter_context(patch.object(owner,name,timed))
        install(torch.linalg,'svd','svd_seconds')
        install(torch.linalg,'svdvals','svdvals_seconds')
        install(cs.SmoothDual,'evaluate','residual_svd_polar_seconds')
        install(cs.SmoothDual,'hvp','hvp_seconds')
        install(cs,'_newton_direction','cg_seconds')
        install(cs,'certificate','certificate_seconds')
        install(cs,'_reference','cpu_reference_fallback_seconds')
        return self

    def __exit__(self,*args):return self.stack.__exit__(*args)
