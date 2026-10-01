"""Replay fixed Stage-D failures, stopping BEFORE CPU reference; SLURM only."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('SLURM required')
import argparse
import gc
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback
from dataclasses import asdict
from unittest.mock import patch
import torch
import sitecustomize
from benchmarks import tiny_transformer as tt
from benchmarks.lr_study import INITIAL_HASH, SinglePassStream, batch_digest
import qnormuon.optimizer as om
from experiments.stage_d_forensics import Observer, ReferenceRequested, cpu, stats

SOURCE = Path('/home/prignano/qnormuon-runs/lr-stage-d/study-29024')


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def fingerprint(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def timeout(*args):
    raise TimeoutError('unchanged 180-second step limit')


def run(label, root, manifest):
    origin=SOURCE/label
    provenance=json.loads((origin/'provenance.json').read_text())
    config=provenance['config']
    source_metrics=[json.loads(s) for s in (origin/'metrics.jsonl').read_text().splitlines()]
    original_failure=json.loads((origin/'failure.json').read_text()) if (origin/'failure.json').exists() else None
    limit=original_failure['completed_steps']+1 if original_failure else 101
    output=root/label;output.mkdir()
    tt.seed_everything(2026,False);torch.set_default_dtype(torch.float32)
    train_source,val=tt.datasets(config);train=SinglePassStream(train_source,128,2026)
    assert train.metadata==manifest['data']['train'] and val.metadata==manifest['data']['validation']
    model=tt.TinyTransformer(tt.ModelConfig(**config['model'])).float().cuda()
    opts=tt.Optimizers(model,'qso',config)
    assert fingerprint(model)==INITIAL_HASH
    write(output/'provenance.json',dict(original=provenance,job_id=os.environ['SLURM_JOB_ID'],
        python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),limit=limit,source_sha256=manifest['source_sha256']))
    index=0;step=0;observed=[];initial=tt.evaluate(model,val,config)
    original=om.solve_coupled
    def solve(u,d,a,**kwargs):
        nonlocal index
        pair=opts.paired.pairs[index];index+=1
        if step < 50:return original(u,d,a,**kwargs)
        raw=dict(up=cpu(pair.up),down=cpu(pair.down),grad_up=cpu(pair.up.grad),grad_down=cpu(pair.down.grad))
        observer=Observer(u,d,a,kwargs.get('initial_lambda'),
            dict(pair=pair.name,training_step=step,completed_steps=step,run_id=label),raw=raw)
        solver_config=asdict(kwargs['config']);solver_config['_device']=u.device
        try:
            result=observer.solve(solver_config)
        except ReferenceRequested:
            # This write is reached immediately upon reference entry, before any ADMM.
            write(output/'smooth_exit.json',observer.event)
            torch.save(observer.fixture(),output/'failed_pair.pt')
            print('FIRST_REFERENCE_REQUEST',label,step,pair.name,observer.event['reason'],
                  observer.event['current_rcond'],observer.event['current_normalized_gap'],flush=True)
            raise
        row=dict(step=step,pair=pair.name,**result.metrics,**observer.normalization,
            final_gradient_norm=observer.evaluations[-1]['gradient_norm'],
            line_search_count=result.counts.line_trials,
            raw_up_norms=stats(pair.up.double().norm(dim=1)),
            raw_down_norms=stats(pair.down.T.double().norm(dim=1)),
            canonical_norms=stats(u.norm(dim=1)),objective_rms=stats(a)['rms'])
        observed.append(row)
        return result
    print('CASE_BEGIN',label,'initial_validation',initial,flush=True)
    with (output/'steps.jsonl').open('w',buffering=1) as log:
        try:
            for step in range(limit):
                index=0
                before=fingerprint(model) if step==limit-1 and original_failure else None
                signal.alarm(180)
                with patch.object(om,'solve_coupled',solve):
                    record=tt.train_step(model,opts,train,config,step)
                signal.alarm(0)
                assert record['batch_sha256']==manifest['batch_hashes'][step]
                assert record['loss']==source_metrics[step]['loss'], 'trajectory did not reproduce'
                if (step+1)%32==0:record['validation_loss']=tt.evaluate(model,val,config)
                log.write(json.dumps(record,allow_nan=False)+'\n')
                if step%10==0:print('REPLAY',label,step,record['loss'],flush=True)
            assert not original_failure, 'expected reference transition not reproduced'
            status=dict(status='control_reproduced',completed_steps=limit)
        except ReferenceRequested as exc:
            signal.alarm(0)
            assert original_failure and step==original_failure['completed_steps']
            assert index and opts.paired.pairs[index-1].name==original_failure['active']['pair']
            assert fingerprint(model)==before, 'failing step committed parameters'
            assert all(opts.paired.state[p.up]['step']==step for p in opts.paired.pairs)
            status=dict(status='captured',reason=str(exc),step=step,pair=opts.paired.pairs[index-1].name,
                        failing_step_uncommitted=True,matched_original_losses=True)
        except Exception:
            signal.alarm(0)
            write(output/'unexpected_failure.json',dict(traceback=traceback.format_exc(),step=step))
            raise
    write(output/'region_pairs.json',observed);write(output/'status.json',status)
    print('CASE_END',label,status,flush=True)
    del model,opts;gc.collect();torch.cuda.empty_cache()


def main():
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    signal.signal(signal.SIGALRM,timeout)
    parser=argparse.ArgumentParser();parser.add_argument('--case',default='tuned',choices=['tuned','representatives'])
    args=parser.parse_args()
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    for path,digest in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, 'frozen Stage-D source changed'
    root=Path('/home/prignano/qnormuon-runs/stage-d-forensics')/f"capture-{os.environ['SLURM_JOB_ID']}"
    root.mkdir(parents=True,exist_ok=False)
    labels=['confirmation-qso-a0.0003-q0.00122474487139'] if args.case=='tuned' else [
        'coarse-qso-a0.0003-q0.0003','coarse-qso-a0.0003-q0.001','coarse-qso-a0.0003-q0.004',
        'ablation-qso-a0.0002-q0.00122474487139',
        'coarse-qso-a0.0003-q0.0015','refinement-qso-a0.0003-q0.00122474487139']
    for label in labels:run(label,root,manifest)
    print('CAPTURE_COMPLETE',root,flush=True)


if __name__=='__main__':main()
