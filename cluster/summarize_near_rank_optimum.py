"""Small JSON/XML-only report aggregation; no torch or fixture loading."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path('/home/prignano/qnormuon-runs/near-rank-optimum/study-29066')
REFINED=ROOT/'refinement-29068'
def read(path):return json.loads(path.read_text())
def main():
    continuation=read(ROOT/'summary.json')['cases']
    refined=read(REFINED/'summary.json')
    consistency=read(ROOT/'consistency-29069/summary.json')
    final=read(REFINED/'final_audit.json')
    results={}
    for name,r in refined.items():
        c=consistency[name]
        assert r['kkt']['gradient_norm']<1e-10
        assert r['kkt']['metrics']['normalized_gap']<1e-10
        assert r['kkt']['metrics']['residual_rcond']<1e-4
        assert min(r['decompositions']['gpu_tall']['rank_margin'])>1e6
        results[name]=dict(classification='A: FULL-RANK-BELOW-GUARD',kkt=r['kkt'],
            lbfgs_iterations=len(r['history'])-1,lbfgs_counts=r['counts'],
            decompositions=r['decompositions'],construction_error_bound=r['construction_error_bound'],
            initial_continuations={start:dict(reason=x['reason'],iterations=x['iterations'],
                counts=x['counts'],kkt=x['independent']) for start,x in continuation[name].items()},
            polished_multiple_starts=c['starts'],normal_hessian_gershgorin_lower=c['normal_hessian_gershgorin_lower'],
            temporal_objective_estimate=c['temporal_objective_estimate'],temporal_sensitivity=c['temporal_sensitivity'],
            random_sensitivity=r['sensitivity'],final_audit=final[name])
    xml=ET.parse('cluster/near-rank-tests-29068.xml').getroot()
    tests=[dict(s.attrib) for s in xml.iter('testsuite')]
    payload=dict(classification='A: FULL-RANK-BELOW-GUARD',tests=tests,cases=results)
    Path('cluster/near_rank_optimum_summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    with Path('cluster/near_rank_tuned_progress.csv').open('w') as stream:
        fields=['phase','iteration','dual_value','gradient_norm_internal','normalized_gap',
                'sigma_min_U','sigma_min_D','rcond_U','rcond_D']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        r=continuation['tuned']['warm']
        # Initial evaluation followed by each explicitly accepted line trial.
        accepted_evaluations=[r['evaluations'][0]]+[x for x in r['evaluations'][1:] if x.get('accepted')]
        assert len(accepted_evaluations)==len(r['history'])
        for h in r['history']:
            ev=accepted_evaluations[h['iteration']]
            s=ev['singular_values'];scale=ev['original_singular_scale']
            writer.writerow(dict(phase='production Newton research admission',iteration=h['iteration'],
                dual_value=h['dual_objective'],gradient_norm_internal=ev['gradient_norm'],
                normalized_gap=h['normalized_gap'],sigma_min_U=scale*s[0][-1],sigma_min_D=scale*s[1][-1],
                rcond_U=s[0][-1]/s[0][0],rcond_D=s[1][-1]/s[1][0]))
        for h in refined['tuned']['history']:
            if h['iteration'] not in (0,5,10,15,20,25,len(refined['tuned']['history'])-1):continue
            s=h['singular_values']
            writer.writerow(dict(phase='independent LBFGS',iteration=h['iteration'],dual_value=h['value']*r['evaluations'][0]['original_singular_scale'],
                gradient_norm_internal=h['gradient_norm'],normalized_gap='',sigma_min_U=s[0][-1],sigma_min_D=s[1][-1],
                rcond_U=s[0][-1]/s[0][0],rcond_D=s[1][-1]/s[1][0]))
    for name,r in results.items():
        print(name,'KKT',r['kkt'])
        print('DECOMPOSITIONS',r['decompositions'])
        print('MULTISTART MAX',max(x['lambda_distance'] for x in r['polished_multiple_starts'].values()),
            max(x['direction_relative'] for x in r['polished_multiple_starts'].values()),
            'COERCIVITY',r['normal_hessian_gershgorin_lower'])
        print('SENSITIVITY',[(x['kind'],x['relative_input_scale'],x['polar_relative']) for x in r['random_sensitivity']])
        print('ADVERSE',[(x['kind'],x['relative_scale'],x['polar_relative']) for x in r['final_audit']['adverse_sensitivity']])
    print('TESTS',tests)


if __name__=='__main__':main()
