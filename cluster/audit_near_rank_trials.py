"""Independent certificate/spectrum at every recorded continuation evaluation."""
import os
if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('SLURM required')
import argparse
import csv
import json
from pathlib import Path
import torch
import qnormuon.coupled_solver as cs
from cluster.near_rank_optimum import FIXTURES


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args()
    root=Path(args.root);results={};csv_rows=[]
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    for name,path in FIXTURES.items():
        f=torch.load(path,map_location='cuda',weights_only=False);u,d,a=(f[k].double() for k in ('u','d','a'))
        recorded=json.loads((root/f'{name}.json').read_text());problem=cs.SmoothDual(u,d,a)
        case={}
        for label,run in recorded.items():
            state=torch.load(root/f'{name}-{label}.pt',map_location='cuda',weights_only=False)
            assert len(state['internal_coordinates'])==len(run['evaluations'])
            rows=[]
            for z,entry in zip(state['internal_coordinates'],run['evaluations']):
                lam=f['beta']+f['magnitude']*f['coord']*z
                ev=problem.evaluate(lam)
                cached=cs.CachedDualSpectrum(lam,ev.pair,ev.singular,1.)
                _,metrics=cs.certificate(u,d,a,lam,ev.pair,cs.Counts(),cached_dual=cached,primal_norm_backend='gram_upper')
                before=[x for x in run['newton'] if x['evaluation_index']<entry['index']]
                direction=before[-1] if before else None
                row=dict(fixture=name,start=label,index=entry['index'],iteration=entry['iteration'],
                    kind=entry['kind'],accepted_trial=entry.get('accepted'),dual_value=ev.value,
                    gradient_norm=float(ev.gradient.norm()),multiplier_displacement=float((lam-state['initial_lambda']).norm()),
                    sigma_min_U=float(ev.singular[0,-1]),sigma_max_U=float(ev.singular[0,0]),
                    sigma_min_D=float(ev.singular[1,-1]),sigma_max_D=float(ev.singular[1,0]),
                    rcond_U=float(ev.singular[0,-1]/ev.singular[0,0]),rcond_D=float(ev.singular[1,-1]/ev.singular[1,0]),
                    normalized_gap=metrics['normalized_gap'],signed_normalized_gap=metrics['signed_normalized_gap'],
                    horizontal_residual=metrics['normalized_horizontal_residual'],spectral_excess=metrics['spectral_excess'],
                    cg_work=sum(sum(t['event']=='curvature' for t in x['cg']) for x in before),
                    slope_internal=entry.get('slope'),step_length=entry.get('step_length'),armijo_rhs_internal=entry.get('armijo_rhs'),
                    direction_norm_internal=direction['direction_norm'] if direction else None)
                rows.append(dict(**row,spectrum_tail=ev.singular[:,-8:].tolist()))
                csv_rows.append(row)
            case[label]=rows
            print(name,label,'audited',len(rows),'evaluations',flush=True)
        results[name]=case
    (root/'all_evaluation_audit.json').write_text(json.dumps(results,indent=2)+'\n')
    with Path('cluster/near_rank_all_evaluations.csv').open('w') as output:
        writer=csv.DictWriter(output,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)


if __name__=='__main__':main()
