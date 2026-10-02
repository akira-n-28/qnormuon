"""One frozen seed-2027 package integration replay; research oracle is offline."""
import os
if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('SLURM required')
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
from unittest.mock import patch
import torch
import sitecustomize
import qnormuon.coupled_solver as cs
import qnormuon.optimizer as om
from benchmarks import tiny_transformer as tt
from benchmarks.lr_study import exact_tree, quality
from benchmarks.multiseed import configuration
from cluster.run_qso_multiseed import data_contract, read
from cluster.run_lr_study import fingerprint, timeout
from cluster.full_rank_512step import clean, write, statistics
from experiments.full_rank_admission import solve_full_rank
from experiments.full_rank_training import compact_metrics, online_gate

SOURCE=Path('/home/prignano/qnormuon-runs/paired-multiseed/study-29093')
NUMERICAL_FILES=('qnormuon/coupled_solver.py','qnormuon/optimizer.py',
                 'qnormuon/_full_rank_admission.py','qnormuon/_numerical_certification.py')


def normalize_old_state(state):
    """Only intentional schema/policy differences; every state tensor is compared."""
    state=copy.deepcopy(state)
    paired=state['optimizers'][0]
    paired['qso_format_version']=3
    for group in paired['param_groups']:
        group['solver'].update(admission_policy='full_rank_epsilon_lmo',fallback=False)
    return state


def compare_result(new,old):
    assert (new.converged,new.reason,new.iterations,new.counts)==(old.converged,old.reason,old.iterations,old.counts)
    assert torch.equal(new.pair,old.pair) and torch.equal(new.lam,old.lam)
    assert new.history==old.history and new.evaluations==old.evaluations and new.actions==old.actions
    for key,value in compact_metrics(old).items():
        if key not in ('rank_seconds','value_seconds'):assert new.metrics[key]==value,key


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gates',type=Path,required=True);args=parser.parse_args()
    gate=read(args.gates);assert gate['passed']
    for p,h in gate['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    manifest=read(SOURCE/'manifest.json');historical=SOURCE/'seed-2027/qso'
    oldprovenance=read(historical/'provenance.json');oldsummary=read(historical/'summary.json')
    assert oldsummary['status']=='passed' and read(historical/'resume.json')['exact']
    # Every original model/data/runner/research source remains unchanged. The two
    # edited package files are checked against their original committed source.
    for p,h in manifest['source_sha256'].items():
        data=(subprocess.check_output(['git','show','HEAD:'+p]) if p in NUMERICAL_FILES else Path(p).read_bytes())
        assert hashlib.sha256(data).hexdigest()==h,'frozen source changed: '+p
    config=configuration(manifest['base'],2027,'qso');assert config==oldprovenance['config']
    config['solver'].update(admission_policy='full_rank_epsilon_lmo',fallback=False)
    assert config['steps']==512 and config['warmup_steps']==51 and config['seed']==2027
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE and 'H100' in torch.cuda.get_device_name(0)
    tt.seed_everything(2027,False);torch.set_default_dtype(torch.float32)
    train,val,data,hashes,vh=data_contract(config);contract=manifest['seed_contracts']['2027']
    assert data==contract['data'] and hashes==contract['expected_batch_hashes'] and vh==contract['validation_batch_hashes']
    model=tt.TinyTransformer(tt.ModelConfig(**config['model'])).float().cuda()
    opts=tt.Optimizers(model,'qso',config);assert fingerprint(model)==contract['initialization_sha256']
    assert all(g['solver']['admission_policy']=='full_rank_epsilon_lmo' for g in opts.paired.param_groups)
    oldrecords=[json.loads(line) for line in (historical/'metrics.jsonl').read_text().splitlines()]
    oldrows=[json.loads(line) for line in (historical/'solves.jsonl').read_text().splitlines()]
    oldvalidation=read(historical/'validation.json');assert len(oldrecords)==512 and len(oldrows)==3072
    root=Path('/home/prignano/qnormuon-runs/production-v1')/('trajectory-'+os.environ['SLURM_JOB_ID']);root.mkdir(parents=True,exist_ok=False)
    provenance=dict(config=config,data=data,expected_batch_hashes=hashes,validation_batch_hashes=vh,
        initialization_sha256=fingerprint(model),historical=str(historical),gates=str(args.gates),
        policy='full_rank_epsilon_lmo',format_version=3,selection_semantics='primary_epsilon_lmo_full_rank',selection_certified=False,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in NUMERICAL_FILES},
        job=os.environ['SLURM_JOB_ID'],hostname=os.uname().nodename,python=os.sys.version,
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        cpus=os.environ.get('SLURM_CPUS_PER_TASK'),memory=os.environ.get('SLURM_MEM_PER_NODE'),output=str(root))
    write(root/'provenance.json',provenance);metadata=dict(data,admission_policy=provenance['policy'],format_version=3)
    records=[];rows=[];validation={};audit_seconds=0.;resume=None;checkpoint_comparison=False;captured=[]
    original=om.solve_coupled
    def observe(u,d,a,*,config,initial_lambda=None):
        result=original(u,d,a,config=config,initial_lambda=initial_lambda)
        captured.append((u,d,a,initial_lambda,config,result))
        return result
    def forbidden(*args,**kwargs):raise AssertionError('hidden CPU reference')
    signal.signal(signal.SIGALRM,timeout);start=time.perf_counter();peak=0
    try:
        validation[0]=tt.evaluate(model,val,config);assert validation[0]==oldvalidation['0']
        torch.cuda.reset_peak_memory_stats()
        with patch.object(cs,'_reference',forbidden), (root/'solves.jsonl').open('w',buffering=1) as solve_log, (root/'metrics.jsonl').open('w',buffering=1) as metric_log:
            for step in range(512):
                captured.clear();signal.alarm(config['step_timeout_seconds'])
                with patch.object(om,'solve_coupled',observe):record=tt.train_step(model,opts,train,config,step)
                signal.alarm(0)
                assert len(captured)==6
                # Losses, batch hashes and all non-timing update diagnostics must
                # equal the retained historical trajectory at EVERY step.
                for key in ('loss','batch_sha256','gradient_rms','parameter_rms','update_rms',
                            'update_parameter_ratio','lr_factor','tokens','finite_delta_x_norm','finite_delta_x_relative'):
                    assert record[key]==oldrecords[step][key],('first historical difference',step,key,record[key],oldrecords[step][key])
                assert record['batch_sha256']==hashes[step]
                record['pair_solver_seconds']=sum(item[-1].seconds for item in captured)
                peak=max(peak,torch.cuda.max_memory_allocated())
                # Oracle does not propose updates, choose branches, or enter the
                # timed training step. Check exact directions and persisted lambdas.
                torch.cuda.synchronize();audit_start=time.perf_counter()
                for index,(pair,item) in enumerate(zip(opts.paired.pairs,captured)):
                    u,d,a,lam,cfg,result=item
                    ref=solve_full_rank(u,d,a,config=cfg,initial_lambda=lam,profile=False)
                    compare_result(result,ref)
                    row=dict(result.metrics,step=step,pair=pair.name,seconds=result.seconds)
                    for key,value in oldrows[step*6+index].items():
                        if key not in ('seconds','rank_seconds','value_seconds'):
                            assert row[key]==value,('historical pair difference',step,pair.name,key)
                    assert torch.equal(opts.paired.state[pair.up]['lambda'],ref.lam)
                    assert row['selection_semantics']=='primary_epsilon_lmo_full_rank' and not row['selection_certified']
                    row.pop('posterior',None);row.pop('rank',None)
                    rows.append(row);solve_log.write(json.dumps(clean(row),allow_nan=False)+'\n')
                torch.cuda.synchronize();audit_seconds+=time.perf_counter()-audit_start
                if (step+1)%32==0:
                    validation[step+1]=tt.evaluate(model,val,config)
                    assert validation[step+1]==oldvalidation[str(step+1)],('validation difference',step+1)
                    record['validation_loss']=validation[step+1]
                    write(root/'validation.json',validation)
                    print('EXACT_VALIDATION',step+1,validation[step+1],flush=True)
                # Compact numeric logs, no huge repeated factor/checkpoint state.
                for metrics in record['qso'].values():metrics.pop('posterior',None);metrics.pop('rank',None)
                records.append(record);metric_log.write(json.dumps(clean(record),allow_nan=False)+'\n')
                if step+1==128:write(root/'online_gate_128.json',online_gate(rows))
                if step+1==256:
                    tt.save_checkpoint(root/'checkpoint.pt',model,opts,config,metadata)
                    previous=torch.load(historical/'checkpoint.pt',map_location='cuda',weights_only=False)
                    exact_tree(model.state_dict(),previous['model'])
                    exact_tree(opts.state_dict(),normalize_old_state(previous['optimizers']))
                    checkpoint_comparison=True;del previous;gc.collect()
                    write(root/'historical_checkpoint_comparison.json',dict(exact=True,step=256,
                        normalized_metadata=['qso_format_version:2->3','explicit admission_policy','fallback:True->False'],
                        model=True,canonical_momenta=True,lambdas=True,counters=True,unsupported_adamw=True,scheduler=True))
                if step+1==259:
                    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
                    other=tt.TinyTransformer(tt.ModelConfig(**config['model'])).float().cuda()
                    other_opts=tt.Optimizers(other,'qso',config)
                    assert tt.load_checkpoint(root/'checkpoint.pt',other,other_opts,config,metadata)==256
                    for j in range(256,259):
                        replay=tt.train_step(other,other_opts,train,config,j)
                        assert replay['loss']==records[j]['loss'] and replay['batch_sha256']==hashes[j]
                        for pair in other_opts.paired.pairs:
                            for key in ('alpha_lower','conservative_normalized_gap','rcond_sides','newton_iterations','cg_iterations','line_trials','selection_semantics','selection_certified'):
                                assert other_opts.paired.last_diagnostics[pair.name][key]==records[j]['qso'][pair.name][key]
                    exact_tree(model.state_dict(),other.state_dict());exact_tree(opts.state_dict(),other_opts.state_dict())
                    resume=dict(exact=True,completed_steps=[257,258,259],checkpoint_step=256,checkpoint_bytes=(root/'checkpoint.pt').stat().st_size)
                    write(root/'resume.json',resume)
                    del other,other_opts;gc.collect();torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
                    torch.cuda.reset_peak_memory_stats()
            assert resume and checkpoint_comparison and len(rows)==3072
            summary=dict(status='passed',classification='A',steps=512,tokens=1048576,seed=2027,
                exact_historical_losses=True,exact_historical_pair_metrics=True,exact_research_results=3072,
                exact_historical_checkpoint=checkpoint_comparison,resume=resume,
                **quality(validation,records),solver=statistics(rows,records),peak_cuda_bytes=peak,
                measured_step_seconds=sum(r['step_seconds'] for r in records),
                optimizer_seconds=sum(r['optimizer_seconds'] for r in records),
                offline_oracle_seconds=audit_seconds,total_job_seconds=time.perf_counter()-start,
                warm_excluded_steps=5,warm={key:tt.summarize([r[key] for r in records[5:]]) for key in
                    ('step_seconds','pair_solver_seconds','optimizer_seconds','forward_backward_seconds')},
                historical_warm=oldsummary['warm'])
            summary['tokens_per_second']=1048576/summary['measured_step_seconds']
            summary['optimizer_fraction']=summary['optimizer_seconds']/summary['measured_step_seconds']
            summary['warm_step_ratio_to_research']=summary['warm']['step_seconds']['median']/oldsummary['warm']['step_seconds']['median']
            write(root/'summary.json',summary)
            print('INTEGRATION_PASSED',root,flush=True)
    except Exception as error:
        signal.alarm(0)
        write(root/'failure.json',dict(completed_steps=len(records),exception=repr(error)))
        if captured:
            u,d,a,lam,cfg,result=captured[-1]
            torch.save(dict(u=u.cpu(),d=d.cpu(),a=a.cpu(),initial_lambda=lam.cpu() if lam is not None else None,
                reason=result.reason,metrics=result.metrics),root/'failed_pair.pt')
        raise
    finally:
        signal.alarm(0)
        for p,h in provenance['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p

if __name__=='__main__':main()
