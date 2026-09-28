"""Small H100 production smoke and unfiltered suite; run only via SLURM."""
import collections
import json
import os
import sys
import time

if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('Run inside the SLURM compute allocation')
import pytest
import torch
import qnormuon.coupled_solver as solver
from qnormuon import QuotientSpectralOptimizer, SwiGLUPair

assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name(0)
torch.cuda.reset_peak_memory_stats()
counts=collections.Counter()
original=solver._solve

def observed(*args,**kw):
    result=original(*args,**kw)
    counts['solves']+=1
    counts['fallbacks']+=int(result.fallback)
    counts['uncertified_results']+=int(not result.converged)
    counts['solver_dtype:'+str(args[2].dtype)]+=1
    if result.fallback:counts['fallback_reason:'+result.reason]+=1
    return result
solver._solve=observed
started=time.perf_counter()
rc=pytest.main(['-o','addopts=','-q',*sys.argv[1:]])
print('PRODUCTION_TEST_TELEMETRY',json.dumps(dict(counts)),flush=True)
for dtype in (torch.float32,torch.bfloat16):
    generator=torch.Generator(device='cuda').manual_seed(1701)
    up=torch.nn.Parameter(torch.randn(16,8,generator=generator,device='cuda',dtype=dtype))
    down=torch.nn.Parameter(torch.randn(8,16,generator=generator,device='cuda',dtype=dtype))
    up.grad=torch.randn(16,8,generator=generator,device='cuda',dtype=dtype)
    down.grad=torch.randn(8,16,generator=generator,device='cuda',dtype=dtype)
    opt=QuotientSpectralOptimizer([SwiGLUPair('smoke',up,down)],lr=.05)
    before=up.detach().clone()
    opt.step()
    assert not torch.equal(before,up)
    assert torch.isfinite(up).all() and torch.isfinite(down).all()
    assert opt.last_diagnostics['smoke']['converged']
    assert opt.state[up]['momentum_up'].dtype==torch.float32
    print('PRODUCTION_H100_SMOKE',str(dtype),json.dumps(opt.last_diagnostics),flush=True)
torch.cuda.synchronize()
print('PRODUCTION_TOTAL_SECONDS',time.perf_counter()-started,flush=True)
print('PEAK_CUDA_ALLOCATED_BYTES',torch.cuda.max_memory_allocated(),flush=True)
print('PEAK_CUDA_RESERVED_BYTES',torch.cuda.max_memory_reserved(),flush=True)
raise SystemExit(rc)
