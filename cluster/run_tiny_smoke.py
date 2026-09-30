"""Stage C only: 50-step AdamW/QSO smoke, checkpoint replay, no LR sweep."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('Training requires a SLURM compute allocation')
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import platform
import signal
import socket
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch
import sitecustomize
from benchmarks.tiny_transformer import (ModelConfig,TinyTransformer,Optimizers,datasets,seed_everything,
    evaluate,train_step,save_checkpoint,load_checkpoint,solver_summary,summarize)
import qnormuon.optimizer as optimizer_module
from benchmarks.solver_timing import SolverTiming


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def fingerprint(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def timeout_handler(signum,frame):
    raise TimeoutError('smoke step exceeded the explicit wall-time safety budget; no solver settings changed')


def resume_check(checkpoint,model,opts,data,config,metadata,records):
    # Replay exactly the three already-completed continuation batches. This
    # diagnostic is outside the measured training steps and restores RNG after.
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    resumed=TinyTransformer(ModelConfig(**config['model'])).float().cuda()
    resumed_opts=Optimizers(resumed,opts.method,config)
    first=load_checkpoint(checkpoint,resumed,resumed_opts,config,metadata)
    replay=[]
    for step in range(first,first+config['resume_check_steps']):
        signal.alarm(config['step_timeout_seconds'])
        replay.append(train_step(resumed,resumed_opts,data,config,step))
        signal.alarm(0)
    max_parameter_error=max(float((p.detach()-q.detach()).abs().max()) for p,q in zip(model.parameters(),resumed.parameters()))
    loss_error=max(abs(r['loss']-records[r['step']]['loss']) for r in replay)
    if max_parameter_error>1e-6 or loss_error>1e-6:
        raise AssertionError(f'checkpoint continuation mismatch: parameter={max_parameter_error}, loss={loss_error}')
    state_error=0.
    def compare(a,b):
        nonlocal state_error
        if isinstance(a,torch.Tensor):
            if a.dtype!=b.dtype or a.shape!=b.shape:raise AssertionError('restored state dtype/shape mismatch')
            state_error=max(state_error,float((a-b).abs().max()) if a.numel() else 0.)
        elif isinstance(a,dict):
            if a.keys()!=b.keys():raise AssertionError('state keys mismatch')
            for k in a:compare(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            if len(a)!=len(b):raise AssertionError('state length mismatch')
            for x,y in zip(a,b):compare(x,y)
        elif a!=b:raise AssertionError('state scalar mismatch')
    compare(opts.state_dict(),resumed_opts.state_dict())
    if state_error>1e-6:raise AssertionError(f'optimizer state mismatch: {state_error}')
    del resumed,resumed_opts,replay
    gc.collect()
    torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
    return dict(passed=True,steps=config['resume_check_steps'],max_parameter_error=max_parameter_error,
                max_loss_error=loss_error,max_optimizer_state_error=state_error,
                checkpoint_bytes=checkpoint.stat().st_size)


def run(method,config,output):
    output.mkdir(parents=True,exist_ok=False)
    seed_everything(config['seed'],config['tf32'])
    torch.set_default_dtype(torch.float32)
    model=TinyTransformer(ModelConfig(**config['model'])).float().cuda()
    opts=Optimizers(model,method,config)
    train,validation=datasets(config)
    metadata=dict(train=train.metadata,validation=validation.metadata)
    provenance=dict(run_id=output.name,method=method,config=config,seed=config['seed'],
        python=sys.version,python_executable=sys.executable,torch=torch.__version__,numpy=np.__version__,
        cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(0),hostname=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'],node_list=os.environ.get('SLURM_JOB_NODELIST'),
        cpus=os.environ.get('SLURM_CPUS_PER_TASK'),memory=os.environ.get('SLURM_MEM_PER_NODE'),
        platform=platform.platform(),parameter_count=sum(p.numel() for p in model.parameters()),
        initialization_sha256=fingerprint(model),dataset=metadata,
        model_storage='float32',forward_backward='bfloat16 autocast',solver_dtype=config['solver']['dtype'],momentum_dtype='float32',
        certificate_dtype='float64',tf32=config['tf32'],deterministic_algorithms=True,
        checkpoint_path=str(output/'checkpoint.pt'),output_path=str(output),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                       ['benchmarks/tiny_transformer.py','qnormuon/optimizer.py','qnormuon/coupled_solver.py',
                        'cluster/run_tiny_smoke.py']})
    write_json(output/'provenance.json',provenance)
    print('RUN_BEGIN',json.dumps(provenance),flush=True)
    records=[];resume=None;active=dict(step=None,pair=None)
    total_start=time.perf_counter();initial_val=evaluate(model,validation,config)
    torch.cuda.reset_peak_memory_stats()
    solve_file=(output/'solves.jsonl').open('w',buffering=1)
    pairs=list(opts.paired.pairs) if method=="qso" else model.pairs();solve_index=0
    completed_solves=[]
    step_timing={}
    original=optimizer_module.solve_coupled
    def observed(u,d,a,**kwargs):
        nonlocal solve_index
        name=pairs[solve_index].name;solve_index+=1;active['pair']=name
        event=dict(step=active['step'],pair=name,objective_rms=float(a.double().square().mean().sqrt()),
                   solver_dtype=str(kwargs['config'].dtype),event='solve_begin')
        solve_file.write(json.dumps(event)+'\n')
        started=time.perf_counter()
        with SolverTiming() as timing:
            result=original(u,d,a,**kwargs)
        torch.cuda.synchronize()
        event.update(event='solve_end',seconds=time.perf_counter()-started,converged=result.converged,
                     fallback=result.fallback,reason=result.reason,rcond=result.metrics['residual_rcond'],
                     normalized_gap=result.metrics['normalized_gap'],newton=result.metrics['newton_iterations'],
                     cg=result.metrics['cg_iterations'])
        event['timing']=dict(timing.seconds)
        for key,value in timing.seconds.items():step_timing[key]=step_timing.get(key,0.)+value
        step_timing['pair_solver_seconds']=step_timing.get('pair_solver_seconds',0.)+event['seconds']
        solve_file.write(json.dumps(event)+'\n')
        completed_solves.append(dict(result.metrics))
        if result.fallback:print('FALLBACK',json.dumps(event),flush=True)
        return result
    original_adam_step=opts.adam.step
    def timed_adam(*args,**kwargs):
        torch.cuda.synchronize();started=time.perf_counter()
        result=original_adam_step(*args,**kwargs);torch.cuda.synchronize()
        step_timing['adam_seconds']=time.perf_counter()-started
        return result
    try:
        with (output/'metrics.jsonl').open('w',buffering=1) as metrics:
            for step in range(config['steps']):
                active.update(step=step,pair=None);solve_index=0;step_timing.clear()
                print('STEP_BEGIN',method,step,flush=True)
                signal.alarm(config['step_timeout_seconds'])
                with patch.object(optimizer_module,'solve_coupled',observed), patch.object(opts.adam,'step',timed_adam):
                    record=train_step(model,opts,train,config,step)
                signal.alarm(0)
                record['timing']=dict(step_timing)
                record['solver_aggregate']=solver_summary([record])
                record['fallback_reasons']={name:d['fallback_reason'] for name,d in record['qso'].items() if d['fallback_used']}
                record['cuda_live_after_step_bytes']=torch.cuda.memory_allocated()
                record['wall_seconds']=time.perf_counter()-total_start
                if (step+1)%config['evaluation_interval']==0 or step+1==config['steps']:
                    record['validation_loss']=evaluate(model,validation,config)
                records.append(record)
                metrics.write(json.dumps(record,allow_nan=False)+'\n')
                print('STEP_END',method,step,record['loss'],record['optimizer_seconds'],flush=True)
                stats=solver_summary(records)
                if stats['solves']>=config['fallback_stop_minimum_solves'] and stats['fallback_percent']/100>config['fallback_stop_fraction']:
                    raise RuntimeError('substantial fallback frequency: stopping smoke before tuning or sweeping')
                if step+1==config['checkpoint_step']:
                    save_checkpoint(output/'checkpoint.pt',model,opts,config,metadata)
                if step+1==config['checkpoint_step']+config['resume_check_steps']:
                    resume=resume_check(output/'checkpoint.pt',model,opts,train,config,metadata,records)
                    write_json(output/'resume.json',resume)
                    print('RESUME_CHECK',method,json.dumps(resume),flush=True)
        tokens=records[-1]['tokens'];step_seconds=sum(r['step_seconds'] for r in records)
        opt_seconds=sum(r['optimizer_seconds'] for r in records)
        summary=dict(status='passed',method=method,parameter_count=provenance['parameter_count'],steps=len(records),
                     tokens=tokens,initial_validation_loss=initial_val,final_validation_loss=records[-1]['validation_loss'],
                     first_training_loss=records[0]['loss'],last_training_loss=records[-1]['loss'],
                     training_step_seconds=step_seconds,optimizer_seconds=opt_seconds,
                     optimizer_fraction=opt_seconds/step_seconds,tokens_per_second=tokens/step_seconds,
                     total_wall_seconds=time.perf_counter()-total_start,
                     peak_cuda_bytes=torch.cuda.max_memory_allocated(),reserved_cuda_bytes=torch.cuda.max_memory_reserved(),
                     early_live_bytes=summarize([r['cuda_live_after_step_bytes'] for r in records[5:10]]),
                     late_live_bytes=summarize([r['cuda_live_after_step_bytes'] for r in records[-5:]]),
                     solver=solver_summary(records),resume=resume,
                     early_update_parameter_ratios=summarize([r['update_parameter_ratio'] for r in records[:5]]))
        summary['timing_warmup_excluded_steps']=5
        summary['warm_timing']={key:summarize([r.get('timing',{}).get(key,r.get(key,0.)) for r in records[5:]]) for key in
            ['step_seconds','optimizer_seconds','forward_backward_seconds','tokens_per_second','adam_seconds',
             'pair_solver_seconds','svd_seconds','svdvals_seconds','residual_svd_polar_seconds',
             'cg_seconds','hvp_seconds','certificate_seconds','cpu_reference_fallback_seconds']}
        summary['cold_step']=records[0]
        if not resume or not resume['passed']:raise AssertionError('checkpoint replay was not completed')
        write_json(output/'summary.json',summary)
        print('RUN_SUMMARY',json.dumps(summary),flush=True)
        return summary
    except BaseException as exc:
        signal.alarm(0)
        failure=dict(status='failed',method=method,active=active,completed_steps=len(records),
                     exception=repr(exc),traceback=traceback.format_exc(),
                     completed_step_solver=solver_summary(records),
                     observed_solver=solver_summary([dict(qso={str(i):v for i,v in enumerate(completed_solves)})]),
                     incomplete_solve_pair=active['pair'],
                     initial_validation_loss=initial_val,parameter_count=provenance['parameter_count'],
                     peak_cuda_bytes=torch.cuda.max_memory_allocated())
        write_json(output/'failure.json',failure)
        print('RUN_FAILURE',json.dumps(failure),flush=True)
        raise
    finally:
        solve_file.close()
        del model,opts
        gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/tiny_transformer/smoke.json')
    parser.add_argument('--method',choices=['adamw','qso','both'],default='both')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    if not 50<=config['steps']<=100:raise ValueError('Stage C requires only 50–100 steps; no sweep execution')
    if not 0<config['checkpoint_step']<config['checkpoint_step']+config['resume_check_steps']<=config['steps']:
        raise ValueError('checkpoint replay must fit the smoke budget')
    if not sitecustomize.QSO_NETWORK_GUARD_ACTIVE:raise RuntimeError('offline network guard missing')
    assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name(0)
    root=Path(config['output_root']).resolve()
    if not str(root).startswith('/home/prignano/') or str(root).startswith('/home/prignano/qnormuon/'):
        raise ValueError('artifacts must use an explicit shared-home path outside the source tree')
    root.mkdir(parents=True,exist_ok=True)
    stat=os.statvfs(root)
    if stat.f_bavail*stat.f_frsize<2*1024**3:raise RuntimeError('less than 2 GiB shared-home filesystem space')
    signal.signal(signal.SIGALRM,timeout_handler)
    for method in (['adamw','qso'] if args.method=='both' else [args.method]):
        run(method,config,root/f"smoke-{os.environ['SLURM_JOB_ID']}-{method}-seed{config['seed']}")


if __name__=='__main__':main()
