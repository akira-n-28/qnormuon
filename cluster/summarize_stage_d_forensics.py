"""Lightweight JSON-only forensic aggregation; never imports torch or loads .pt."""
import csv
import json
import math
from pathlib import Path
import statistics


ROOT = Path('/home/prignano/qnormuon-runs/stage-d-forensics')
OUTPUT = Path('cluster')


def load(path):
    return json.loads(path.read_text())


def factor(step, steps, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return .1 + .45 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup - 1)))


def main():
    cases = []
    pair_rows = []
    for directory in sorted(ROOT.glob('capture-*/*')):
        if not (directory / 'status.json').exists():
            continue
        provenance = load(directory / 'provenance.json')
        config = provenance['original']['config']
        rows = load(directory / 'region_pairs.json')
        steps = [json.loads(line) for line in (directory / 'steps.jsonl').read_text().splitlines()]
        case = dict(run_id=directory.name, path=str(directory), status=load(directory / 'status.json'),
                    paired_peak_lr=config['qso_lr'], adam_peak_lr=config['adam_lr'],
                    schedule_steps=config['steps'], warmup=config['warmup_steps'],
                    observed_pairs=len(rows), last_completed_step={k: steps[-1][k] for k in
                    ('step', 'loss', 'gradient_rms', 'parameter_rms', 'update_rms', 'update_parameter_ratio')})
        for row in rows:
            pair_rows.append(dict(run_id=directory.name, **{k: row[k] for k in
                ('step', 'pair', 'residual_rcond', 'minimum_smooth_rcond', 'normalized_gap',
                 'newton_iterations', 'cg_iterations', 'line_search_count', 'intrinsic_magnitude',
                 'final_gradient_norm', 'objective_rms')}))
        if (directory / 'spectrum_forensics.json').exists():
            forensic = load(directory / 'spectrum_forensics.json')
            event = forensic['event']
            previous = [r for r in rows if r['pair'] == event['pair']][-1]
            case.update(event={k: v for k, v in event.items() if k not in ('current_lambda', 'initial_lambda')},
                oracle=forensic['oracle_evaluations'], raw_state=forensic['raw_state'],
                cold={k: v for k, v in forensic['diagnostic_cold_initialization'].items() if k != 'event'},
                cold_event={k: v for k, v in forensic['diagnostic_cold_initialization'].get('event', {}).items()
                            if k not in ('current_lambda', 'initial_lambda')},
                previous_same_pair={k: previous[k] for k in
                    ('step', 'residual_rcond', 'normalized_gap', 'newton_iterations', 'cg_iterations',
                     'line_search_count', 'intrinsic_magnitude', 'final_gradient_norm')},
                intrinsic_magnitude_jump=event['intrinsic_magnitude'] / previous['intrinsic_magnitude'],
                partial_reference=forensic.get('partial_cpu_reference_independent_certificate'))
            reference = directory / 'bounded_reference'
            if reference.exists():
                case['reference_bound'] = load(reference / 'bounded_status.json')
                case['reference_latest'] = load(reference / 'latest_progress.json')
        else:
            case['control_region'] = {k: dict(min=min(r[k] for r in rows),
                median=statistics.median(r[k] for r in rows), max=max(r[k] for r in rows)) for k in
                ('residual_rcond', 'minimum_smooth_rcond', 'normalized_gap', 'newton_iterations',
                 'cg_iterations', 'line_search_count', 'intrinsic_magnitude', 'final_gradient_norm')}
            case['by_pair'] = {name: dict(min_rcond=min(r['residual_rcond'] for r in rows if r['pair'] == name),
                max_gap=max(r['normalized_gap'] for r in rows if r['pair'] == name))
                for name in sorted({r['pair'] for r in rows})}
        cases.append(case)
    (OUTPUT / 'stage_d_forensic_summary.json').write_text(json.dumps(cases, indent=2, allow_nan=False) + '\n')
    with (OUTPUT / 'stage_d_forensic_pair_region.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader(); writer.writerows(pair_rows)
    peak = next(c['paired_peak_lr'] for c in cases if c['schedule_steps'] == 512)
    with (OUTPUT / 'stage_d_forensic_schedule.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['zero_based_step', 'paired_lr_256', 'paired_lr_512', 'adam_lr_256',
                         'adam_lr_512', '512_to_256_ratio'])
        for step in range(101):
            short, long = factor(step, 256, 26), factor(step, 512, 51)
            writer.writerow([step, peak * short, peak * long, .0003 * short, .0003 * long, long / short])
    print(f'Aggregated {len(cases)} fixed cases and {len(pair_rows)} observed successful pair solves')


if __name__ == '__main__':
    main()
