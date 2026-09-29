"""Full-vs-basic cast diagnostics on the identical saved warm H100 state."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('SLURM allocation required')

import hashlib
import json
from pathlib import Path
import statistics
from unittest.mock import patch

import sitecustomize
import torch
from benchmarks.tiny_transformer import (
    ModelConfig, Optimizers, TinyTransformer, datasets, load_checkpoint,
    seed_everything, train_step,
)


def digest(model):
    h=hashlib.sha256()
    for value in model.state_dict().values():
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert 'H100' in torch.cuda.get_device_name(0)
    config=json.loads(Path('configs/tiny_transformer/smoke.json').read_text())
    checkpoint=Path(config['output_root'])/f"smoke-28914-qso-seed{config['seed']}/checkpoint.pt"
    assert checkpoint.is_file()
    seed_everything(config['seed'],config['tf32'])
    train,val=datasets(config)
    metadata=dict(train=train.metadata,validation=val.metadata)
    baseline=[json.loads(line) for line in (checkpoint.parent/'metrics.jsonl').read_text().splitlines()][25]
    rows=[]
    for full in (True,False,False,True,False,True,True,False):
        model=TinyTransformer(ModelConfig(**config['model'])).float().cuda()
        opts=Optimizers(model,'qso',config)
        opts.paired.record_cast_diagnostics=full
        assert load_checkpoint(checkpoint,model,opts,config,metadata)==25
        count=[0]
        original=torch.linalg.svdvals
        def observed(*args,**kwargs):
            count[0]+=1
            return original(*args,**kwargs)
        with patch.object(torch.linalg,'svdvals',observed):
            record=train_step(model,opts,train,config,25)
        assert record['loss']==baseline['loss'] and record['batch_sha256']==baseline['batch_sha256']
        assert all(row['converged'] and not row['fallback_used'] for row in record['qso'].values())
        assert all(('cast_spectral_excess' in row)==full for row in record['qso'].values())
        rows.append(dict(full_cast_diagnostics=full,step_seconds=record['step_seconds'],
                         optimizer_seconds=record['optimizer_seconds'],svdvals_calls=count[0],
                         model_sha256=digest(model),loss=record['loss'],
                         pair_metrics={name:{key:value for key,value in value.items()
                                              if not key.startswith('cast_')}
                                       for name,value in record['qso'].items()}))
        print('CAST_RUN',len(rows),full,record['optimizer_seconds'],count[0],flush=True)
        del model,opts
    assert all(row['model_sha256']==rows[0]['model_sha256'] for row in rows)
    assert all(row['pair_metrics']==rows[0]['pair_metrics'] for row in rows)
    summary={str(flag):dict(optimizer_median=statistics.median(r['optimizer_seconds'] for r in rows[2:] if r['full_cast_diagnostics']==flag),
                            optimizer_times=[r['optimizer_seconds'] for r in rows[2:] if r['full_cast_diagnostics']==flag],
                            svdvals_calls=sorted({r['svdvals_calls'] for r in rows if r['full_cast_diagnostics']==flag}))
             for flag in (False,True)}
    report=dict(job=os.environ['SLURM_JOB_ID'],checkpoint=str(checkpoint),step=25,
                rows=rows,summary=summary,trajectory_equal=True)
    output=Path('cluster')/f"cast_diagnostics_benchmark-{os.environ['SLURM_JOB_ID']}.json"
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('CAST_SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
