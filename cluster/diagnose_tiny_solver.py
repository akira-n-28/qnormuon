"""One first-batch pair: diagnostic only, no model update or LR search."""
import os
if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('SLURM required')
import json
from dataclasses import replace
from pathlib import Path
import signal
import time
import torch
from torch.nn import functional as F
from benchmarks.tiny_transformer import ModelConfig,TinyTransformer,datasets,seed_everything,autocast
from qnormuon import SolverConfig,regular_canonicalize,solve_coupled

signal.signal(signal.SIGALRM,lambda *args: (_ for _ in ()).throw(TimeoutError('diagnostic budget')))
c=json.loads(Path('configs/tiny_transformer/smoke.json').read_text())
seed_everything(c['seed'],c['tf32']);torch.set_default_dtype(torch.float32)
m=TinyTransformer(ModelConfig(**c['model'])).float().cuda()
train,_=datasets(c)
x,y=train.batch(0,c['batch_size'],c['model']['sequence_length'],'cuda')
with autocast(m,c['precision']):logits=m(x)
loss=F.cross_entropy(logits.float().flatten(0,1),y.flatten());loss.backward()
p=m.pairs()[0]
u,d,root=regular_canonicalize(p.up,p.down,dtype=torch.float32)
a=torch.stack(((root[:,None]*p.up.grad.double()).float(),(p.down.grad.T.double()/root[:,None]).float()))*(1-c['beta'])
print('CAPTURE',json.dumps(dict(pair=p.name,shape=list(u.shape),loss=float(loss.detach()),objective_rms=float(a.double().square().mean().sqrt()))),flush=True)
for dtype in (torch.float32,torch.float64):
    signal.alarm(150);start=time.perf_counter()
    r=solve_coupled(u,d,a,config=SolverConfig(dtype=dtype,fallback=False))
    signal.alarm(0)
    print('DIAGNOSTIC',json.dumps(dict(dtype=str(dtype),seconds=time.perf_counter()-start,reason=r.reason,metrics=r.metrics,history=r.history)),flush=True)
