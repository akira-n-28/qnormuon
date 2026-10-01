"""Compact Stage-D report/CSV and standalone vector plots; no extra training."""
import csv
import html
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def read(path):return json.loads(Path(path).read_text())


def svg_plot(path,title,xlabel,ylabel,series):
    """Dependency-free vector figure, exact points, unsmoothed connecting lines."""
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
    w,h=960,600;l,t,r,b=100,65,270,85;pw=w-l-r;ph=h-t-b
    xs=[x for _,points in series for x,y in points];ys=[y for _,points in series for x,y in points]
    xmin,xmax=min(xs),max(xs);ymin,ymax=min(ys),max(ys)
    pad=max((ymax-ymin)*.06,abs(ymax)*.005,1e-8);ymin-=pad;ymax+=pad
    def xp(x):return l+(x-xmin)/max(xmax-xmin,1)*pw
    def yp(y):return t+(ymax-y)/(ymax-ymin)*ph
    esc=html.escape
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
       '<rect width="100%" height="100%" fill="white"/>',
       '<g font-family="DejaVu Sans,Arial,sans-serif" font-size="16" fill="#222">',
       f'<text x="{l}" y="30" font-size="20">{esc(title)}</text>']
    for i in range(6):
        x=xmin+(xmax-xmin)*i/5;y=ymin+(ymax-ymin)*i/5
        s += [f'<path d="M {xp(x):.3f} {t} V {t+ph}" stroke="#eeeeee"/>',
              f'<path d="M {l} {yp(y):.3f} H {l+pw}" stroke="#eeeeee"/>',
              f'<text x="{xp(x):.3f}" y="{t+ph+25}" text-anchor="middle">{x/1000:.0f}k</text>',
              f'<text x="{l-12}" y="{yp(y)+5:.3f}" text-anchor="end">{y:.4g}</text>']
    s += [f'<path d="M {l} {t} V {t+ph} H {l+pw}" fill="none" stroke="#222"/>',
          f'<text x="{l+pw/2}" y="{h-25}" text-anchor="middle">{esc(xlabel)}</text>',
          f'<text transform="translate(24 {t+ph/2}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text>']
    for j,(label,points) in enumerate(series):
        c=colors[j%len(colors)];coords=' '.join(f'{xp(x):.3f},{yp(y):.3f}' for x,y in points)
        s.append(f'<polyline points="{coords}" fill="none" stroke="{c}" stroke-width="2"/>')
        # Every point is drawn; none is smoothed, resampled or averaged.
        for x,y in points:s.append(f'<circle cx="{xp(x):.3f}" cy="{yp(y):.3f}" r="2.4" fill="{c}"/>')
        s += [f'<path d="M {w-r+25} {t+24*j} h 22" stroke="{c}" stroke-width="3"/>',
              f'<text x="{w-r+55}" y="{t+24*j+5}">{esc(label)}</text>']
    s.append('</g></svg>');path.write_text('\n'.join(s)+'\n')


def main():
    root=Path(sys.argv[1]);rows=read(root/'results.json');manifest=read(root/'manifest.json')
    coarse=read(root/'coarse_selection.json');tuned=read(root/'tuned_selection.json')
    confirm={r['method']:r for r in rows if r['stage']=='confirmation'}
    a,q=confirm['adamw'],confirm['qso']
    complete=a['status']==q['status']=='passed'
    plots=Path('docs/figures/qso_lr_sweep');plots.mkdir(parents=True,exist_ok=True)
    def curve(row):return sorted((int(k)*2048,v) for k,v in row['evaluations'].items())
    for method in ['adamw','qso']:
        svg_plot(plots/f'coarse_{method}.svg',f'Coarse {method}: equal-token validation',
            'Training tokens','Validation cross entropy',
            [(f"peak LR {r['adam_lr'] if method=='adamw' else r['qso_lr']:.4g}"+
                (' FAILED' if r['status']=='failed' else ''),curve(r))
             for r in rows if r['stage']=='coarse' and r['method']==method])
    records={}
    for method,row in confirm.items():
        records[method]=[json.loads(line) for line in (root/row['run_id']/'metrics.jsonl').read_text().splitlines()]
        if row['status']=='failed':
            rs=records[method]
            def stats(values):
                values=sorted(values)
                return dict(mean=sum(values)/len(values),median=values[len(values)//2],
                            p95=values[min(len(values)-1,math.ceil(.95*len(values))-1)],max=max(values))
            diagnostics=[d for r in rs for d in r['qso'].values()]
            row.update(measured_step_seconds=sum(r['step_seconds'] for r in rs),
                optimizer_seconds=sum(r['optimizer_seconds'] for r in rs),
                peak_cuda_bytes=max(r['cuda_peak_bytes'] for r in rs),resume=None,
                warm={k:stats([r[k] for r in rs[5:]]) for k in
                    ('step_seconds','optimizer_seconds','pair_solver_seconds','forward_backward_seconds')})
            row['tokens_per_second']=len(rs)*2048/row['measured_step_seconds']
            row['optimizer_fraction']=row['optimizer_seconds']/row['measured_step_seconds']
            row['solver']=dict(solves=len(diagnostics),fallbacks=sum(d['fallback_used'] for d in diagnostics))
    svg_plot(plots/'tuned_validation.svg','Tuned confirmation: equal-token validation','Training tokens',
             'Validation cross entropy',[(m+(' FAILED' if confirm[m]['status']=='failed' else ''),curve(confirm[m])) for m in ['adamw','qso']])
    for filename,title,key,label in [
        ('tuned_training','Tuned confirmation: training loss','loss','Training cross entropy'),
        ('update_ratio','Tuned confirmation: update scale','update_parameter_ratio','Update RMS / parameter RMS'),
        ('gradient_rms','Tuned confirmation: gradient scale','gradient_rms','Gradient RMS')]:
        svg_plot(plots/f'{filename}.svg',title,'Training tokens',label,
            [(m+(' FAILED' if confirm[m]['status']=='failed' else ''),[(r['tokens'],r[key]) for r in records[m]]) for m in ['adamw','qso']])
    svg_plot(plots/'qso_newton.svg','QSO confirmation: Newton work per step','Training tokens','Paired solves',
        [(str(n),[(r['tokens'],sum(d['newton_iterations']==n for d in r['qso'].values())) for r in records['qso']]) for n in [0,1,2]]+
        [('3+',[(r['tokens'],sum(d['newton_iterations']>=3 for d in r['qso'].values())) for r in records['qso']])])
    allvals=sorted(int(k) for k in a['evaluations'])
    differences=[dict(step=k,tokens=k*2048,adamw=a['evaluations'][str(k)],qso=q['evaluations'].get(str(k)),
                      qso_minus_adamw=q['evaluations'][str(k)]-a['evaluations'][str(k)] if str(k) in q['evaluations'] else None) for k in allvals]
    with (root/'confirmation_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(differences[0]));writer.writeheader();writer.writerows(differences)
    metrics=['final_validation_loss','minimum_validation_loss','last_three_validation_mean',
             'post_initial_validation_mean','validation_auc_mean']
    delta={k:q[k]-a[k] for k in metrics} if a['status']==q['status']=='passed' else {}
    useful=q['status']=='passed' and q['solver']['fallbacks']==0
    late=[r['qso_minus_adamw'] for r in differences[-3:]]
    pathological=max(r['update_parameter_ratio'] for r in records['qso'])>1 or max(d['newton_iterations'] for r in records['qso'] for d in r['qso'].values())>20
    multi=complete and useful and delta.get('final_validation_loss',1)<0 and delta.get('last_three_validation_mean',1)<0 and sum(v is not None and v<0 for v in late)>=2 and not pathological
    pilot_delta=tuned['qso']['last_three_validation_mean']-tuned['adamw']['last_three_validation_mean']
    if multi:classification='C: tuned QSO is consistently better at equal tokens on this single-seed study; multi-seed confirmation is justified'
    elif complete and pilot_delta>0 and delta.get('final_validation_loss',0)>0 and delta.get('last_three_validation_mean',0)>0 and all(v is not None and v>0 for v in late):
        classification='A: tuned QSO is worse at equal tokens on this single-seed study'
    else:classification='B: tuned QSO and tuned AdamW are inconclusive / mixed on this single-seed study'
    def table(items):
        s=['| Method | AdamW LR | Paired QSO LR | Status / completed steps | Last-three mean | Final | Post-initial mean | Max update ratio |',
           '|---|---:|---:|---|---:|---:|---:|---:|']
        for r in items:
            vals=[f"{r[k]:.6f}" if k in r else '—' for k in ['last_three_validation_mean','final_validation_loss','post_initial_validation_mean','max_update_parameter_ratio']]
            paired=f"{r['qso_lr']:.8g}" if r['method']=='qso' else '—'
            s.append(f"| {r['method']} | {r['adam_lr']:.8g} | {paired} | {r['status']} / {r.get('completed_steps',r['steps'])} | "+' | '.join(vals)+' |')
        return '\n'.join(s)
    s=['# Stage D: single-seed optimizer-specific learning-rate study','',f'**{classification}.**','',
       '## A. Frozen setup and provenance','',
       f"All candidates ran on one H100 allocation, job `{manifest['job_id']}`. Production sources and defaults were unchanged; source SHA256 values, full configs, token-prefix hashes and all expected batch hashes are recorded in `{root}/manifest.json` and each run provenance.",
       '', 'The complete suite passed **400 tests, zero failed, zero skipped**, in 16.66 s before the sweep. Raw validation: `cluster/lr-tests-29024.xml` and `cluster/run_lr_study-29024.log`. The study completed all 20 declared candidate runs: 11 passed and 9 QSO timeout failures, retained as data. Report/plot analysis used separate CPU-only SLURM allocations; no additional training was performed.',
       '', 'The common model is the 11,457,408-parameter six-layer SwiGLU Transformer, six `[1024,384]` pairs, seed 2026. Initialization SHA256 was asserted to equal `0607adc4c9cffa7b902c0956fa7477ea274078fdc31406d84d527b79ca6a2401` for every candidate. Every completed training minibatch hash was independently precomputed and asserted.',
       '', 'The production policy is K=I, explicit pairing, fp32 canonical EMA, direct fp64 full thin SVD, previous original-coordinate lambda, projected-primal `gram_upper`, diagnostics on and cast diagnostics off. Gap/rcond/horizontality/spectral/signed-gap thresholds remain `3e-5 / >1e-4 / 1e-10 / 1e-12 / >=-1e-10`. No research decomposition/predictor was enabled. Unsupported parameters use AdamW; QSO and AdamW have their own peak learning rates.',
       '', 'Model storage fp32, bf16 autocast, TF32 off, deterministic algorithms, no dropout, weight decay zero and no clipping. AdamW betas are `(0.9,0.95)`; QSO EMA beta is `0.95`. Batch 16, context 128, accumulation one: 2,048 tokens per step. Validation uses the same four fixed smoke batches. All local paths are in the saved configs. No network fallback or download occurred.',
       '', '**Stage-D data clarification:** the old smoke samples windows with replacement. To satisfy the requested nonrepeated confirmation, Stage D uses one seed-2026 permutation of disjoint 128-input-token blocks. Every candidate shares the same order, and 256-step runs use its first half. The existing local shard supplies 1,048,576 distinct input positions plus one lookahead label token; header, size, range and prefix were verified in the allocation before training. This intentionally differs from the Stage-C sampled-window stream, not between Stage-D methods. Validation is unchanged. Repeated token *IDs* are natural; input *positions* do not repeat.',
       '', 'Pilot: 256 steps / 524,288 tokens, validation at step 0 then every 32 steps through 256, with 26 warmup steps. Confirmation: 512 steps / 1,048,576 tokens, validation at step 0 then every 32 steps through 512, with 51 warmup steps. Both decay by the same cosine to 10% of peak. The final three checkpoints are 192/224/256 or 448/480/512.',
       '', '## B. Predeclared grids and selection','',
       f'AdamW: `{ADAM_GRID_TEXT}`. QSO paired: `{QSO_GRID_TEXT}`, unsupported AdamW fixed `3e-4`.',
       '', 'Selection is lexicographic: last-three validation mean, final validation, all post-initial validation mean, then lower maximum update/parameter RMS. Failed candidates are ineligible but retained. Training loss and wall time never enter selection. Refinement uses the declared geometric neighbors/boundary extension. Failure limits were fixed before execution: loss/validation >20, nonfinite values, uncertified pair,180-second step timeout, or reference fallback fraction >20% after12 observed pairs. Solver settings are never rescued by retuning.',
       '', '## C. Complete coarse sweep','',table([r for r in rows if r['stage']=='coarse']),
       '', '## D. Deterministic refinement','',table([r for r in rows if r['stage']=='refinement']),
       '',f"Coarse winners: AdamW `{coarse['adamw']['adam_lr']:.8g}`, QSO `{coarse['qso']['qso_lr']:.8g}`. Combined coarse/refinement AdamW winner `{tuned['adamw']['adam_lr']:.8g}`. The paired QSO winner was fixed before unsupported-parameter ablation.",
       '', '## E. Unsupported-parameter AdamW ablation','',table(tuned['ablation']),
       '', 'The already-tested `3e-4` run is reused because its entire configuration matches; no extra training is needed. The paired LR is fixed across all three ablation entries.',
       '', 'Other recorded quality summaries for complete runs (failed horizons remain unavailable):','',
       '| Candidate | Minimum validation | Mean post-initial | Last-three mean | Normalized validation AUC |',
       '|---|---:|---:|---:|---:|']
    for row in rows:
        if row['status']=='passed':
            s.append(f"| {row['run_id']} | {row['minimum_validation_loss']:.6f} | {row['post_initial_validation_mean']:.6f} | {row['last_three_validation_mean']:.6f} | {row['validation_auc_mean']:.6f} |")
    s += [
       '', '## F. Tuned 512-step equal-token confirmation','',table(list(confirm.values())),
       '', '| Completed step | Tokens | AdamW validation | QSO validation | QSO − AdamW |',
       '|---:|---:|---:|---:|---:|']
    for d in differences:
        qv=f"{d['qso']:.6f}" if d['qso'] is not None else 'unavailable'
        diff=f"{d['qso_minus_adamw']:+.6f}" if d['qso_minus_adamw'] is not None else 'unavailable'
        s.append(f"| {d['step']} | {d['tokens']} | {d['adamw']:.6f} | {qv} | {diff} |")
    s += ['', 'Signed summary differences (negative favors QSO):','', '| Metric | QSO − AdamW |','|---|---:|']
    for k,v in delta.items():s.append(f'| {k} | {v:+.6f} |')
    if not complete:
        for metric in ('Final validation loss', 'Minimum validation loss', 'Last-three validation mean', 'Post-initial validation mean', 'Normalized validation AUC'):
            s.append(f'| {metric} | unavailable |')
        s += ['', '**The tuned QSO 512-step confirmation failed.** The successful pilot does not establish a useful certified 512-step trajectory. Principal final/best/last-three/mean/AUC differences are unavailable; early common-checkpoint differences are diagnostics only. No partial loss substitutes for the missing target horizon. No extra seeds are justified.']
    s += ['', 'AUC is trapezoidal validation-loss area divided by the common token horizon, including step0; the separately reported post-initial mean excludes step0. No metric disagreement is resolved by choosing a favorable checkpoint.',
          '', '## G. QSO numerical validity','', '| Candidate | Pairs | Newton 0/1/2/3+ | CG mean/median/p95 | Max/p95 gap | Min rcond | Max normalized horizontal | Max spectral excess | ADMM | Line trials |',
          '|---|---:|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['method']!='qso' or r['status']!='passed':continue
        z=r['solver'];dist=z['distributions'];bins=z['iteration_bins'];cg=dist['cg_iterations'];gap=dist['normalized_gap']
        s.append(f"| {r['run_id']} | {z['solves']} | {'/'.join(str(bins[k]) for k in ['0','1','2','3+'])} | {cg['mean']:.2f}/{cg['median']:.0f}/{cg['p95']:.0f} | {gap['max']:.3g}/{gap['p95']:.3g} | {z['minimum_rcond']:.3g} | {z['max_normalized_horizontality']:.3g} | {dist['spectral_excess']['max']:.3g} | {z['fallbacks']} | {z['line_search_trials']} |")
    failures=[r for r in rows if r['status']!='passed']
    s += ['', 'Every observed completed QSO result was checked before the optimizer could commit it. Signed normalized gap and finiteness were also checked. Full per-pair metrics/reasons and partial-step solve-begin events are preserved in `solves.jsonl`.',
          '', 'Failures:']
    s += [f"- `{r['run_id']}`: {r.get('exception')}; completed {r.get('completed_steps')} steps; exact traceback in run `failure.json`." for r in failures] or ['None.']
    s.insert(s.index('Failures:')+1,'')
    s += ['', 'Failed-candidate partial numerical evidence (not quality-eligible):','',
          '| Candidate | Completed pairs / begun | Newton 0/1/2/3+ | CG mean/median/p95 | Max/p95 gap | Min rcond | Max normalized horizontal | Spectral excess | Completed ADMM / interrupted reference |',
          '|---|---|---|---|---|---:|---:|---:|---|']
    for r in failures:
        if r['method']!='qso':continue
        logs=[json.loads(line) for line in (root/r['run_id']/'solves.jsonl').read_text().splitlines()]
        ends=[v for v in logs if v['event']=='end'];begun=sum(v['event']=='begin' for v in logs)
        bins=[sum(v['newton_iterations']==i for v in ends) for i in [0,1,2]]+[sum(v['newton_iterations']>=3 for v in ends)]
        def quant(values,fraction):
            values=sorted(values);return values[min(len(values)-1,math.ceil(fraction*len(values))-1)]
        cg=[v['cg_iterations'] for v in ends];g=[v['normalized_gap'] for v in ends]
        s.append(f"| {r['run_id']} | {len(ends)}/{begun} | {'/'.join(map(str,bins))} | {sum(cg)/len(cg):.2f}/{quant(cg,.5)}/{quant(cg,.95)} | {max(g):.3g}/{quant(g,.95):.3g} | {min(v['residual_rcond'] for v in ends):.3g} | {max(v['normalized_horizontal_residual'] for v in ends):.3g} | {max(v['spectral_excess'] for v in ends):.3g} | {sum(v['fallback'] for v in ends)} / {'horizontal_spectral.py' in r.get('traceback','')} |")
    s += ['', 'The timeout tracebacks enter production `_reference` and CPU ADMM spectral-ball projection. A zero count of completed ADMM returns does not mean reference fallback was never entered. The initiating smooth-failure reason was not exposed before interruption and is not guessed. The paired optimizer commits no partial uncertified step.']
    s += ['', '## H. Gradient and update scales','', '| Winner | Gradient RMS first/last | Update RMS first/last | Parameter RMS first/last | Update/parameter first/last/max |', '|---|---|---|---|---|']
    for m,rs in records.items():
        f,l=rs[0],rs[-1]
        s.append(f"| {m} | {f['gradient_rms']:.5g}/{l['gradient_rms']:.5g} | {f['update_rms']:.5g}/{l['update_rms']:.5g} | {f['parameter_rms']:.5g}/{l['parameter_rms']:.5g} | {f['update_parameter_ratio']:.5g}/{l['update_parameter_ratio']:.5g}/{max(x['update_parameter_ratio'] for x in rs):.5g} |")
    if not complete:s += ['', 'QSO last-step scale is from its last completed step (87), whereas AdamW reaches512. These unequal horizons describe operational trajectories, not a matched final-quality comparison. Completed parameters, gradients and updates remained finite; the observed failures were solver/reference timeouts, not observed NaN divergence.']
    s += ['', '## I. Cost at equal tokens (not a selection criterion)','', '| Winner | Total wall s | Step sum s | Warm step median/p95 s | Six-pair median/p95 s | Fwd/back median/p95 s | Tokens/s | Optimizer fraction | Peak GPU MiB |','|---|---:|---:|---|---|---|---:|---:|---:|']
    for m,r in confirm.items():
        w=r['warm'];fmt=lambda k:f"{w[k]['median']:.5f}/{w[k]['p95']:.5f}"
        s.append(f"| {m} | {r['total_wall_seconds']:.3f} | {r['measured_step_seconds']:.3f} | {fmt('step_seconds')} | {fmt('pair_solver_seconds')} | {fmt('forward_backward_seconds')} | {r['tokens_per_second']:.1f} | {100*r['optimizer_fraction']:.2f}% | {r['peak_cuda_bytes']/2**20:.1f} |")
    costratio=q['measured_step_seconds']/a['measured_step_seconds'] if complete else None
    pilot_costratio=tuned['qso']['measured_step_seconds']/tuned['adamw']['measured_step_seconds']
    cost_text=f'QSO requires **{costratio:.2f}×** the measured training-step time of tuned AdamW.' if complete else 'No tuned equal-token cost ratio exists: QSO confirmation terminated early. Its cost row summarizes completed steps only; wall time includes interrupted reference work.'
    s += ['', cost_text+' Warm metrics exclude steps0–4; cold work stays in totals. Total wall includes validation, checkpoint I/O and independent continuation replay; step sums exclude them. Pair timers synchronize only around production solves; no decomposition/decision policy is changed. GPU peak excludes driver/context and independently replayed model allocations. Candidates shared one allocation; sequential order and device-clock variation limit fine timing interpretations.',
          '', 'Per-run cost summaries (never used for tuning):','',
          '| Candidate | Wall s | Step sum s | Warm step median/p95 s | Six-pair median/p95 s | Fwd/back median/p95 s | Tokens/s | Optimizer % | Peak MiB |',
          '|---|---:|---:|---|---|---|---:|---:|---:|']
    for r in rows:
        if r['status']!='passed':
            s.append(f"| {r['run_id']} FAILED | {r['total_wall_seconds']:.3f} | unavailable (see partial metrics) | — | — | — | — | — | — |")
            continue
        w=r['warm'];f=lambda k:f"{w[k]['median']:.5f}/{w[k]['p95']:.5f}"
        s.append(f"| {r['run_id']} | {r['total_wall_seconds']:.3f} | {r['measured_step_seconds']:.3f} | {f('step_seconds')} | {f('pair_solver_seconds')} | {f('forward_backward_seconds')} | {r['tokens_per_second']:.1f} | {100*r['optimizer_fraction']:.2f} | {r['peak_cuda_bytes']/2**20:.1f} |")
    s += ['', f'The two successful tuned 256-step pilots have a measured-step cost ratio of **{pilot_costratio:.2f}×** (QSO/AdamW); this is separate from the 512-step principal comparison.',
          '', f'Failed-run wall time includes interrupted reference work. Completed-step throughput from their partial logs must not be substituted for end-to-end throughput. Complete partial-run cost and scale summaries are saved in `{root}/partial_run_cost_scale.json`.',
          '', '## J. Checkpoint continuation','']
    for m,r in confirm.items():
        if r['resume']:s.append(f"- {m}: saved at step256, independently replayed steps257–259; exact minibatch hashes, losses, model and complete optimizer/scheduler state: `{r['resume']}`.")
        else:s.append(f"- {m}: confirmation terminated before completing the prescribed checkpoint-continuation audit; no Stage-D exact continuation claim.")
    s += ['', 'The QSO audit, when reached, includes fp32 canonical up/down EMA, fp64 original lambda, pair counters and unsupported AdamW state. Historical smoke evidence does not substitute for an unreached Stage-D audit. Checkpoint format was not changed. No coarse/refinement checkpoints were saved.',
          '', '## K. Limitations and plots','',
          'One seed, small local train/validation prefixes, four fixed validation batches and a short horizon do not provide a significance estimate or universal superiority claim. The data representation/tokenizer was reused, not downloaded or semantically re-audited. Personal quota remains unknown; ample filesystem free space is not a user-quota guarantee. Checkpoints are retained only for confirmations that reached step256; here only AdamW did so. This study did not reopen solver-performance research or use additional exploratory training.',
          '', 'Figures are dependency-free SVG vector artifacts with every measured point and unsmoothed connecting lines. All quality figures use tokens on the horizontal axis; cost is reported separately.', '']
    for stem in ['coarse_adamw','coarse_qso','tuned_validation','tuned_training','update_ratio','gradient_rms','qso_newton']:
        s.append(f'- [{stem}](figures/qso_lr_sweep/{stem}.svg)')
    s += ['', '## L. Answers and next-step decision','',
          f"1. Best AdamW peak LR: **{tuned['adamw']['adam_lr']:.8g}**.",
          f"2. Best paired QSO peak LR: **{tuned['qso']['qso_lr']:.8g}**.",
          f"3. Best unsupported-parameter AdamW peak LR with QSO: **{tuned['qso']['adam_lr']:.8g}**.",
          '4. LR sensitivity is shown by the complete coarse/refinement validation tables and curves; boundary extensions are explicitly included, not silently substituted into the coarse grid.',
          '5. Equal-token final validation difference QSO−AdamW: '+(f"**{delta['final_validation_loss']:+.6f}**." if complete else '**unavailable**.'),
          f"6. Pilot last-three difference: **{pilot_delta:+.6f}**; confirmation: "+(f"**{delta['last_three_validation_mean']:+.6f}**." if complete else '**unavailable**. A promising pilot without a completed confirmation is inconclusive.'),
          '7. Actual update/parameter and gradient scales are tabulated and plotted above; differences reflect method-specific update geometry and learning rates.',
          f"8. Tuned QSO confirmation: {q['solver']['solves']} completed-step certified paired solves, {q['solver']['fallbacks']} completed CPU ADMM returns; status **{q['status']}**. Interrupted reference work is separately documented.",
          '9. Measured training-step cost ratio: '+(f'**{costratio:.2f}×**.' if complete else '**unavailable at equal target tokens**.'),
          f"10. Separate multi-seed recommendation under the predeclared gate: **{'yes' if multi else 'no'}**. No extra seeds were run.",
          '',f'**Final classification: {classification}.** Production mathematics/defaults remain frozen. Stop after this report.']
    Path('docs/QSO_LR_SWEEP.md').write_text('\n'.join(s)+'\n')
    write= lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n')
    write(root/'final_decision.json',dict(classification=classification,multiseed_recommended=multi,differences=delta,cost_ratio=costratio))
    with (root/'run_quality.csv').open('w') as f:
        keys=['run_id','method','stage','status','adam_lr','qso_lr','last_three_validation_mean',
              'final_validation_loss','minimum_validation_loss','post_initial_validation_mean','validation_auc_mean']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    partials={}
    for row in failures:
        rs=[json.loads(line) for line in (root/row['run_id']/'metrics.jsonl').read_text().splitlines()]
        def quant(values, fraction):
            values=sorted(values);pos=(len(values)-1)*fraction;lo=int(pos);hi=math.ceil(pos)
            return values[lo]*(hi-pos)+values[hi]*(pos-lo) if hi!=lo else values[lo]
        stepsum=sum(r['step_seconds'] for r in rs);optsum=sum(r['optimizer_seconds'] for r in rs)
        def stats(key):
            values=[r[key] for r in rs[5:]]
            return dict(median=quant(values,.5),p95=quant(values,.95))
        partials[row['run_id']]=dict(completed_steps=len(rs),tokens=len(rs)*2048,
            measured_step_seconds=stepsum,total_wall_seconds=row['total_wall_seconds'],
            tokens_per_measured_step_second=len(rs)*2048/stepsum,optimizer_fraction=optsum/stepsum,
            peak_recorded_cuda_bytes=max(r['cuda_peak_bytes'] for r in rs),
            warm={k:stats(k) for k in ['step_seconds','pair_solver_seconds','forward_backward_seconds']},
            max_update_parameter_ratio=max(r['update_parameter_ratio'] for r in rs),
            first_scale={k:rs[0][k] for k in ['gradient_rms','update_rms','parameter_rms']},
            last_completed_scale={k:rs[-1][k] for k in ['gradient_rms','update_rms','parameter_rms']})
    write(root/'partial_run_cost_scale.json',partials)
    for file in plots.glob('*.svg'):ET.parse(file)
    print(classification)


ADAM_GRID_TEXT='1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1.2e-3'
QSO_GRID_TEXT='3e-4, 6e-4, 1e-3, 1.5e-3, 2.5e-3, 4e-3'


if __name__=='__main__':main()
