"""Independent fp64 residual/certificate replay of captured solves; SLURM only."""
import os
if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('SLURM required')
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from experiments.stage_d_forensics import stats, Observer, ReferenceRequested
import qnormuon.coupled_solver as cs
from qnormuon.optimizer import regular_canonicalize


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('capture_root');args=parser.parse_args()
    root=Path(args.capture_root)
    for path in sorted(root.glob('*/failed_pair.pt')):
        f=torch.load(path,map_location='cuda',weights_only=False)
        u,d,a=(f[k] for k in ('u','d','a'))
        beta,coord,magnitude=f['beta'],f['coord'],f['magnitude']
        problem=cs.SmoothDual(u*coord[:,None],d*coord[:,None],
            (a-cs.adjoint(u,d,beta))/magnitude)
        rows=[]
        for i,z in enumerate(f['internal_coordinates']):
            ev=problem.evaluate(z);lam=beta+magnitude*coord*z
            # Full SVD of the original-coordinate residual is an independent
            # calculation versus the normalized cached spectrum used in training.
            original_sigma=torch.linalg.svdvals(a-cs.adjoint(u,d,lam))
            p,metrics=cs.certificate(u,d,a,lam,ev.pair,cs.Counts(),primal_norm_backend='gram_upper')
            normalized_s=ev.singular*magnitude
            rows.append(dict(index=i,**metrics,value=ev.value,gradient_norm=float(ev.gradient.norm()),
                sides=[dict(sigma_max=float(s[0]),sigma_min=float(s[-1]),
                    rcond=float(s[-1]/s[0]),tail=s[-8:].tolist(),nuclear=float(s.sum())) for s in original_sigma],
                original_vs_internal_relative_spectrum_error=float((original_sigma-normalized_s).norm()/original_sigma.norm()),
                coordinate_rcond=ev.rcond,kind=f['evaluations'][i]['kind'],
                newton_iteration=f['evaluations'][i]['iteration']))
        raw=f['raw'];uc,dc,root_scale=regular_canonicalize(raw['up'],raw['down'],dtype=torch.float32)
        raw_state=dict(raw_up_row_norms=stats(raw['up'].double().norm(dim=1)),
            raw_down_row_norms=stats(raw['down'].T.double().norm(dim=1)),
            canonical_up_row_norms=stats(u.norm(dim=1)),canonical_down_row_norms=stats(d.norm(dim=1)),
            gauge_root=stats(root_scale),up_weight_rms=stats(raw['up'])['rms'],
            down_weight_rms=stats(raw['down'])['rms'],canonical_ema_rms=stats(a)['rms'],
            gradient_up_rms=stats(raw['grad_up'])['rms'],gradient_down_rms=stats(raw['grad_down'])['rms'],
            regular_domain=bool((uc.norm(dim=1)>0).all() and (dc.norm(dim=1)>0).all()),
            canonical_up_exact=torch.equal(uc.double(),u),canonical_down_exact=torch.equal(dc.double(),d))
        # Replay the frozen pair from precisely the saved warm multiplier.
        observer=Observer(u,d,a,f['initial_lambda'],f['metadata'],raw={})
        try:observer.solve(dict(asdict(cs.SolverConfig()),_device=u.device))
        except ReferenceRequested:pass
        assert observer.event['reason']==f['event']['reason']
        assert observer.event['history']==f['event']['history'], 'isolated replay history differs'
        result=dict(metadata=f['metadata'],event=f['event'],raw_state=raw_state,
                    oracle_evaluations=rows,isolated_replay_exact=True)
        # Diagnostic ONLY: distinguish a guard failure at transported warm
        # lambda from evidence that no admitted smooth solution exists. The
        # objective, thresholds, budget, and globalization are unchanged.
        cold=Observer(u,d,a,None,dict(f['metadata'],diagnostic='vertical_centering_initialization'))
        try:
            cr=cold.solve(dict(asdict(cs.SolverConfig()),_device=u.device))
            cold_summary=dict(status='certified' if cr.converged else 'uncertified',
                iterations=cr.iterations,reason=cr.reason,metrics=cr.metrics,
                counts=asdict(cr.counts),history=cr.history)
        except ReferenceRequested:
            cold_summary=dict(status='reference_requested_before_execution',event=cold.event)
        result['diagnostic_cold_initialization']=cold_summary
        partial_path=path.parent/'bounded_reference'/'partial_reference.pt'
        if partial_path.exists():
            partial=torch.load(partial_path,map_location='cuda',weights_only=False)
            _,pm=cs.certificate(u,d,a,beta+partial['lam'],partial['pair'],cs.Counts(),primal_norm_backend='svd')
            result['partial_cpu_reference_independent_certificate']=dict(
                iteration=partial['iteration'],metrics=pm,
                status='partial only; reference did not return a converged result',
                secondary_selection='not established on a possibly nonunique face')
        write(path.parent/'spectrum_forensics.json',result)
        print('ANALYZED',path.parent.name,f['event']['reason'],raw_state['regular_domain'],flush=True)


if __name__=='__main__':main()
