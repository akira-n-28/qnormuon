"""JSON/CSV only; safe on the frontend, no tensor libraries or artifacts."""
import csv
import json
from pathlib import Path

base=json.loads(Path('cluster/full_rank_direction_summary.json').read_text())
scales=json.loads(Path('cluster/full_rank_direction_scales.json').read_text())
rows=[]
for group,cases in [('stage',list(base['cases'].items())),
                    ('cold_controls',[(c['name'],c) for c in base['controls']]),
                    ('warm_control',[(scales['warm']['name'],scales['warm'])])]:
    for name,c in cases:
        for r in c['rows']:
            p=r['posterior']
            rows.append(dict(group=group,fixture=name,iteration=r['iteration'],
                normalized_gap=r['normalized_gap'],certified_gap_upper=p['gap_upper'],
                sigma_min_up=p['sigma_min'][0],sigma_min_down=p['sigma_min'][1],
                alpha_lower_up=p['alpha_lower'][0],alpha_lower_down=p['alpha_lower'][1],
                rcond_up=p['rcond'][0],rcond_down=p['rcond'][1],
                direction_error_upper=p['direction_error_upper'],
                normalized_direction_upper=p['normalized_direction_upper'],
                observed_error=r['observed_error'],relative_observed_error=r['relative_observed_error'],
                observed_over_bound=r['observed_over_bound'],fp32_cast_error=r['fp32_cast_error'],
                cg=r['cg'],line_trials=r['line_trials'],gradient_norm=r['gradient_norm']))
with Path('cluster/full_rank_direction_progress.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('wrote',len(rows),'fixed-fixture/control progress rows')
