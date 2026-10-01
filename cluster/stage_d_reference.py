"""Separately bounded unchanged CPU ADMM on one captured research fixture."""
import os
if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('SLURM required')
import argparse
import inspect
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time


def child(fixture,output):
    import torch
    from unittest.mock import patch
    import qnormuon.coupled_solver as cs
    import experiments.horizontal_spectral as hs
    torch.set_num_threads(4)
    f=torch.load(fixture,map_location='cpu',weights_only=False)
    begin=time.perf_counter();spectra={}
    original=torch.linalg.svdvals
    def svdvals(a,*args,**kwargs):
        s=original(a,*args,**kwargs)
        if inspect.currentframe().f_back.f_code is hs.dual_value.__code__:
            spectra['sides']=[dict(sigma_min=float(v[-1]),sigma_max=float(v[0]),rcond=float(v[-1]/v[0])) for v in s]
        return s
    code=hs.solve_lmo.__wrapped__.__code__
    line=next(i for i,s in enumerate(inspect.getsourcelines(hs.solve_lmo.__wrapped__)[0],
        inspect.getsourcelines(hs.solve_lmo.__wrapped__)[1]) if 'if gap_normalized <=' in s)
    def trace(frame,event,arg):
        if frame.f_code is not code:return None
        if event=='line' and frame.f_lineno==line:
            loc=frame.f_locals
            lower=float((loc['objective']*loc['feasible']).sum())
            upper=lower+loc['gap_normalized']
            row=dict(iteration=loc['iteration'],seconds=time.perf_counter()-begin,
                gap_pair_fro_normalized=loc['gap_normalized'],
                normalized_gap=loc['gap_normalized']/max(abs(lower),abs(upper)),
                admm_residual=loc['residual'],residual_spectra=spectra.get('sides'),
                converged_check=loc['gap_normalized']<=loc['tolerance'] and loc['residual']<=loc['tolerance'])
            with (output/'progress.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
            (output/'latest_progress.json').write_text(json.dumps(row,indent=2)+'\n')
            if loc['iteration']%100==0:
                torch.save(dict(pair=loc['feasible'],lam=loc['scale']*loc['lam'],
                    iteration=loc['iteration']),output/'partial_reference.pt')
        return trace
    sys.settrace(trace)
    try:
        with patch.object(torch.linalg,'svdvals',svdvals):
            result=cs._reference(f['u'],f['d'],f['a'],20000)
        row=dict(status='returned',seconds=time.perf_counter()-begin,converged=result.converged,
            iterations=result.iterations,metrics=result.metrics,secondary_selection=result.secondary_selection)
        (output/'result.json').write_text(json.dumps(row,indent=2)+'\n')
    finally:sys.settrace(None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('fixture');parser.add_argument('--seconds',type=int,default=240)
    args=parser.parse_args();output=Path(args.fixture).parent/'bounded_reference';output.mkdir(exist_ok=False)
    p=mp.get_context('spawn').Process(target=child,args=(args.fixture,output))
    begin=time.perf_counter();p.start();p.join(args.seconds)
    if p.is_alive():
        p.terminate();p.join(10)
        if p.is_alive():p.kill();p.join()
        status=dict(status='bounded_timeout',limit_seconds=args.seconds,elapsed=time.perf_counter()-begin,
            selection_status='not established; primary reference did not return')
    else:status=dict(status='child_returned',exitcode=p.exitcode,elapsed=time.perf_counter()-begin)
    (output/'bounded_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
