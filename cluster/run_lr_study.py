"""Frozen Stage-D single-seed sweep; only execute inside one H100 allocation."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('Stage D requires SLURM')
import copy
import gc
import hashlib
import json
from pathlib import Path
import signal
import socket
import sys
import time
import traceback
from collections import Counter
from unittest.mock import patch

import numpy as np
import torch
import sitecustomize
from benchmarks import tiny_transformer as tt
from benchmarks.lr_study import (ADAM_GRID,QSO_GRID,INITIAL_HASH,SinglePassStream,
    batch_digest,selection_key,select,refinement,quality,exact_tree)
import qnormuon.optimizer as om
from qnormuon.coupled_solver import accepted


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def fingerprint(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def timeout(*args):
    raise TimeoutError('unchanged 180-second per-step budget exceeded')


def run(root,stage,method,adam_lr,qso_lr,steps,base,train,val,metadata,hashes,source_hashes):
    config=copy.deepcopy(base)
    config.update(steps=steps,warmup_steps=26 if steps==256 else 51,adam_lr=adam_lr,qso_lr=qso_lr)
    label=f'{stage}-{method}-a{adam_lr:.12g}-q{qso_lr:.12g}'
    output=root/label;output.mkdir(exist_ok=False)
    for p,digest in source_hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest, 'frozen source changed'
    tt.seed_everything(2026,False);torch.set_default_dtype(torch.float32)
    model=tt.TinyTransformer(tt.ModelConfig(**config['model'])).float().cuda()
    opts=tt.Optimizers(model,method,config)
    assert fingerprint(model)==INITIAL_HASH, 'initialization hash differs'
    if method=='qso':
        assert opts.paired.record_diagnostics and not opts.paired.record_cast_diagnostics
        assert len(opts.paired.pairs)==6
        for group in opts.paired.param_groups:
            assert group['momentum_dtype']==torch.float32
            assert group['solver']==dict(config['solver'],dtype=torch.float64,
                reference_max_iterations=20000,independent_certificate=False)
    provenance=dict(run_id=label,stage=stage,method=method,config=config,dataset=metadata,
        initialization_sha256=INITIAL_HASH,parameter_count=sum(p.numel() for p in model.parameters()),
        expected_batch_hashes=hashes[:steps],python=sys.version,python_executable=sys.executable,
        torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        hostname=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
        source_sha256=source_hashes,clipping='none',momentum_dtype='float32',smooth_backend='full_fp64_thin_svd',
        diagnostics=True,cast_diagnostics=False,output_path=str(output))
    write(output/'provenance.json',provenance)
    print('RUN_BEGIN',label,flush=True)
    records=[];solves=[];evaluations={};resume=None
    active={'step':None,'pair':None};pair_index=0;pair_seconds=0.
    original=om.solve_coupled
    def observed(u,d,a,**kw):
        nonlocal pair_index,pair_seconds
        name=opts.paired.pairs[pair_index].name;pair_index+=1;active['pair']=name
        begin=dict(event='begin',step=active['step'],pair=name)
        solve_log.write(json.dumps(begin)+'\n')
        torch.cuda.synchronize();start=time.perf_counter()
        result=original(u,d,a,**kw)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;pair_seconds+=elapsed
        row=dict(result.metrics,event='end',step=active['step'],pair=name,seconds=elapsed,
                 converged=result.converged,fallback=result.fallback,reason=result.reason,
                 line_search_trials=result.counts.line_trials)
        solves.append(row);solve_log.write(json.dumps(row,allow_nan=False)+'\n')
        if not result.converged or not accepted(result.metrics,3e-5) or result.metrics['residual_rcond']<=1e-4:
            raise RuntimeError('QSO pair did not satisfy frozen Stage-D certificate')
        if not all(np.isfinite(result.metrics[k]) for k in ('normalized_gap','signed_normalized_gap',
            'normalized_horizontal_residual','spectral_excess','residual_rcond')):
            raise FloatingPointError('nonfinite QSO certificate')
        count=sum(r['fallback'] for r in solves)
        if len(solves)>=config['fallback_stop_minimum_solves'] and count/len(solves)>config['fallback_stop_fraction']:
            raise RuntimeError('uncontrolled reference fallback frequency')
        return result
    wall=time.perf_counter();training_peak=0
    with (output/'solves.jsonl').open('w',buffering=1) as solve_log, (output/'metrics.jsonl').open('w',buffering=1) as log:
        try:
            signal.alarm(config['step_timeout_seconds'])
            initial=tt.evaluate(model,val,config);signal.alarm(0)
            if not np.isfinite(initial):raise FloatingPointError('nonfinite initial validation')
            evaluations[0]=initial;write(output/'validation.json',evaluations)
            torch.cuda.reset_peak_memory_stats()
            for step in range(steps):
                active.update(step=step,pair=None);pair_index=0;pair_seconds=0.
                signal.alarm(config['step_timeout_seconds'])
                with patch.object(om,'solve_coupled',observed):
                    record=tt.train_step(model,opts,train,config,step)
                signal.alarm(0)
                assert record['batch_sha256']==hashes[step], 'batch stream changed'
                if record['loss']>config['divergence_loss_limit']:
                    raise FloatingPointError('predeclared loss>20 divergence limit')
                record['pair_solver_seconds']=pair_seconds
                training_peak=max(training_peak,torch.cuda.max_memory_allocated())
                if (step+1)%32==0:
                    signal.alarm(config['step_timeout_seconds'])
                    loss=tt.evaluate(model,val,config);signal.alarm(0)
                    if not np.isfinite(loss) or loss>config['divergence_loss_limit']:
                        raise FloatingPointError('validation nonfinite/divergent')
                    evaluations[step+1]=loss;record['validation_loss']=loss
                    write(output/'validation.json',evaluations)
                    print('VALIDATION',label,step+1,loss,flush=True)
                records.append(record);log.write(json.dumps(record,allow_nan=False)+'\n')
                if steps==512 and step+1==256:
                    tt.save_checkpoint(output/'checkpoint.pt',model,opts,config,metadata)
                if steps==512 and step+1==259:
                    # Compare a genuinely independent disk reload with the uninterrupted state.
                    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
                    other=tt.TinyTransformer(tt.ModelConfig(**config['model'])).float().cuda()
                    other_opts=tt.Optimizers(other,method,config)
                    assert tt.load_checkpoint(output/'checkpoint.pt',other,other_opts,config,metadata)==256
                    for j in range(256,259):
                        signal.alarm(config['step_timeout_seconds'])
                        replay=tt.train_step(other,other_opts,train,config,j);signal.alarm(0)
                        assert replay['loss']==records[j]['loss'] and replay['batch_sha256']==hashes[j]
                    exact_tree(model.state_dict(),other.state_dict())
                    exact_tree(opts.state_dict(),other_opts.state_dict())
                    resume=dict(passed=True,exact=True,steps=3,checkpoint_step=256,
                        checkpoint_bytes=(output/'checkpoint.pt').stat().st_size,
                        qso_state_checked=['momentum_up','momentum_down_t','lambda','step'] if method=='qso' else [])
                    write(output/'resume.json',resume)
                    del other,other_opts;gc.collect()
                    torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
                    torch.cuda.reset_peak_memory_stats()
            assert sorted(evaluations)==list(range(0,steps+1,32))
            if steps==512:assert resume and resume['exact']
            duration=time.perf_counter()-wall
            summary=dict(status='passed',run_id=label,stage=stage,method=method,steps=steps,
                adam_lr=adam_lr,qso_lr=qso_lr,tokens=steps*2048,evaluations=evaluations,
                **quality(evaluations,records),total_wall_seconds=duration,
                measured_step_seconds=sum(r['step_seconds'] for r in records),
                optimizer_seconds=sum(r['optimizer_seconds'] for r in records),
                peak_cuda_bytes=training_peak,resume=resume,solver=tt.solver_summary(records))
            summary['tokens_per_second']=summary['tokens']/summary['measured_step_seconds']
            summary['optimizer_fraction']=summary['optimizer_seconds']/summary['measured_step_seconds']
            summary['warm_excluded_steps']=5
            summary['warm']={k:tt.summarize([r[k] for r in records[5:]]) for k in
                ('step_seconds','optimizer_seconds','pair_solver_seconds','forward_backward_seconds')}
            summary['solver']['fallback_reasons']=dict(Counter(r['reason'] for r in solves if r['fallback']))
            summary['solver']['minimum_rcond']=min((r['residual_rcond'] for r in solves),default=None)
            summary['solver']['max_normalized_horizontality']=max((r['normalized_horizontal_residual'] for r in solves),default=None)
            summary['solver']['line_search_trials']=sum(r.get('line_search_trials',0) for r in solves)
            summary['solver']['observed_pair_count']=len(solves)
            write(output/'summary.json',summary)
        except Exception as exc:
            signal.alarm(0)
            summary=dict(status='failed',run_id=label,stage=stage,method=method,steps=steps,
                adam_lr=adam_lr,qso_lr=qso_lr,completed_steps=len(records),exception=repr(exc),
                traceback=traceback.format_exc(),active=active,evaluations=evaluations,
                observed_pairs=len(solves),fallback_count=sum(r['fallback'] for r in solves),
                total_wall_seconds=time.perf_counter()-wall)
            write(output/'failure.json',summary)
            print('RUN_FAILURE',json.dumps(summary),flush=True)
        finally:
            signal.alarm(0)
            del opts,model;gc.collect();torch.cuda.empty_cache()
    print('RUN_END',label,summary['status'],summary.get('last_three_validation_mean'),flush=True)
    return summary


def main():
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name(0)
    signal.signal(signal.SIGALRM,timeout)
    base=json.loads(Path('configs/tiny_transformer/lr_study.json').read_text())
    assert base['seed']==2026 and base['weight_decay']==0 and base['tf32'] is False
    train_source,val=tt.datasets(base)
    train=SinglePassStream(train_source,128,2026)
    assert train.metadata['usable_input_tokens']>=1048576, 'not enough nonrepeated tokens for confirmation'
    metadata=dict(train=train.metadata,validation=val.metadata)
    hashes=[batch_digest(train,i,base) for i in range(512)]
    root=Path(base['output_root'])/f"study-{os.environ['SLURM_JOB_ID']}";root.mkdir(parents=True,exist_ok=False)
    files=['benchmarks/tiny_transformer.py','benchmarks/lr_study.py','cluster/run_lr_study.py',
           'cluster/run_lr_study.sbatch','configs/tiny_transformer/lr_study.json',
           'qnormuon/optimizer.py','qnormuon/coupled_solver.py']
    files=[p for p in files if Path(p).exists()]
    sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    write(root/'manifest.json',dict(config=base,source_sha256=sources,data=metadata,batch_hashes=hashes,
          adam_grid=ADAM_GRID,qso_grid=QSO_GRID,selection='last three, final, all post-initial, max update ratio',
          refinement='geometric neighbors or boundary extension',divergence_loss_limit=20,
          job_id=os.environ['SLURM_JOB_ID']))
    rows=[]
    def candidate(stage,method,a,q,steps=256):
        r=run(root,stage,method,a,q,steps,base,train,val,metadata,hashes,sources)
        rows.append(r);write(root/'results.json',rows);return r
    for rate in ADAM_GRID:candidate('coarse','adamw',rate,1e-3)
    for rate in QSO_GRID:candidate('coarse','qso',3e-4,rate)
    a_coarse=select([r for r in rows if r['method']=='adamw'])
    q_coarse=select([r for r in rows if r['method']=='qso'])
    write(root/'coarse_selection.json',dict(adamw=a_coarse,qso=q_coarse))
    for rate in refinement(ADAM_GRID,a_coarse['adam_lr']):candidate('refinement','adamw',rate,1e-3)
    for rate in refinement(QSO_GRID,q_coarse['qso_lr']):candidate('refinement','qso',3e-4,rate)
    a_best=select([r for r in rows if r['method']=='adamw'])
    q_best=select([r for r in rows if r['method']=='qso'])
    write(root/'paired_selection.json',dict(adamw=a_best,qso=q_best))
    ablation=[q_best]  # 3e-4 already tested; exact same 256-step configuration.
    for rate in [2e-4,5e-4]:ablation.append(candidate('ablation','qso',rate,q_best['qso_lr']))
    q_final=select(ablation)
    write(root/'tuned_selection.json',dict(adamw=a_best,qso=q_final,ablation=ablation))
    candidate('confirmation','adamw',a_best['adam_lr'],1e-3,512)
    candidate('confirmation','qso',q_final['adam_lr'],q_final['qso_lr'],512)
    print('STUDY_COMPLETE',root,flush=True)


if __name__=='__main__':main()
