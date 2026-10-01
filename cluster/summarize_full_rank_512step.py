"""Small JSON-only report preparation; never loads tensors or trains."""
import argparse
import json
from collections import Counter
from pathlib import Path
import statistics


def dist(x):
    x=sorted(x)
    if not x:return {}
    index=(len(x)-1)*.95;lo=int(index);hi=min(lo+1,len(x)-1)
    return dict(mean=statistics.mean(x),median=statistics.median(x),p95=x[lo]+(x[hi]-x[lo])*(index-lo),max=x[-1])


def main():
    p=argparse.ArgumentParser();p.add_argument('trajectory');args=p.parse_args()
    root=Path(args.trajectory)
    records=[json.loads(s) for s in (root/'metrics.jsonl').read_text().splitlines()]
    rows=[json.loads(s) for s in (root/'solves.jsonl').read_text().splitlines()]
    result=json.loads((root/('summary.json' if (root/'summary.json').exists() else 'failure.json')).read_text())
    source=Path('/home/prignano/qnormuon-runs/lr-stage-d/study-29024')
    av={int(k):v for k,v in json.loads((source/'confirmation-adamw-a0.0002-q0.001/validation.json').read_text()).items()}
    qv={int(k):v for k,v in json.loads((root/'validation.json').read_text()).items()}
    comparison=[dict(step=s,tokens=s*2048,qso=qv[s],adamw=av[s],difference=qv[s]-av[s]) for s in sorted(qv)]
    normal=[x for x in rows if x['selection_semantics']=='primary_epsilon_lmo_full_rank']
    below=[x for x in normal if x['below_old_guard']]
    report=dict(result=result,validation=comparison,completed_steps=len(records),tokens=len(records)*2048,
        pair_returns=len(rows),committed_pair_solves=len(records)*6,below_count=len(below),
        below_fraction=len(below)/len(rows),below_by_pair=dict(Counter(x['pair'] for x in below)),
        below_U=sum(x['rcond_sides'][0]<=1e-4 for x in normal),
        below_D=sum(x['rcond_sides'][1]<=1e-4 for x in normal),
        below_both=sum(max(x['rcond_sides'])<=1e-4 for x in normal),
        any_evaluation_below_count=sum(x['any_evaluation_below_old_guard'] for x in rows),
        min_rcond=min(min(x['rcond_sides']) for x in normal),
        min_evaluated_rcond=min(x['minimum_evaluated_rcond'] for x in normal),
        min_alpha=min(min(x['alpha_lower']) for x in normal),
        min_sigma_eta=min(min(x['sigma_min_over_eta']) for x in normal),
        first_below=min((x['step'] for x in below),default=None),last_below=max((x['step'] for x in below),default=None),
        newton_bins={str(n):sum(x['newton_iterations']==n for x in rows) for n in (0,1,2)} |
                    {'3+':sum(x['newton_iterations']>=3 for x in rows)},
        newton_distribution=dict(Counter(x['newton_iterations'] for x in rows)),
        cg=dist([x['cg_iterations'] for x in rows]),line=dist([x['line_trials'] for x in rows]),
        cg_termination=dict(Counter(y for x in rows for y in x['cg_termination'])),
        max_conservative_gap=max(x['conservative_normalized_gap'] for x in normal),
        conservative_gap=dist([x['conservative_normalized_gap'] for x in normal]),
        max_horizontal=max(x['normalized_horizontal_residual'] for x in normal),
        max_spectral_excess=max(x['spectral_excess'] for x in normal),
        max_proxy_distance=max(x['feasible_proxy_distance'] for x in normal),
        minimum_used_curvature=min(x['minimum_used_curvature'] for x in normal if x['minimum_used_curvature'] is not None),
        max_hvp_norm=max(x['maximum_hvp_norm'] for x in normal if x['maximum_hvp_norm'] is not None),
        minimum_signed_gap=min(x['signed_normalized_gap'] for x in normal),
        maximum_newton_slope=max(x['maximum_slope'] for x in normal if x['maximum_slope'] is not None),
        max_cg_per_action=max(x['max_cg_queries_per_action'] for x in rows),
        measured_step_seconds=sum(x['step_seconds'] for x in records),
        warm={k:dist([x[k] for x in records[5:]]) for k in
              ['step_seconds','pair_solver_seconds','optimizer_seconds','forward_backward_seconds']},
        update_ratio=dist([x['update_parameter_ratio'] for x in records]),
        first_scales={k:records[0][k] for k in ['gradient_rms','update_rms','parameter_rms','update_parameter_ratio']},
        last_scales={k:records[-1][k] for k in ['gradient_rms','update_rms','parameter_rms','update_parameter_ratio']},
        solver_work_by_32_step_interval=[dict(first_step=s,last_step=min(s+31,len(records)-1),
            newton=dist([x['newton_iterations'] for x in rows if s<=x['step']<s+32]),
            cg=dist([x['cg_iterations'] for x in rows if s<=x['step']<s+32]),
            line=dist([x['line_trials'] for x in rows if s<=x['step']<s+32])) for s in range(0,len(records),32)])
    if len(records)==512:
        def metrics(v):
            post=[v[s] for s in range(32,513,32)]
            return dict(final=v[512],minimum=min(v.values()),last_three=statistics.mean(post[-3:]),
                post_initial_mean=statistics.mean(post),auc=(v[0]/2+sum(post[:-1])+post[-1]/2)/len(post))
        am,qm=metrics(av),metrics(qv)
        report['quality_summary']=dict(adamw=am,qso=qm,differences={k:qm[k]-am[k] for k in qm})
        report['multi_seed_gate']=qm['final']<am['final'] and qm['last_three']<am['last_three']
    output=Path('cluster/full_rank_512step_summary.json');output.write_text(json.dumps(report,indent=2)+'\n')
    print('SUMMARY',output)
    print(json.dumps({k:v for k,v in report.items() if k not in ('result','validation','solver_work_by_32_step_interval')},indent=2))


if __name__=='__main__':main()
