"""Auditable single-device causal SwiGLU benchmark with no network loaders.

Training is launched only from cluster/run_tiny_smoke.py in a SLURM allocation.
Float32 parameters, bf16 autocast forward/backward, explicit QSO fp32 solves.
"""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass, asdict
import hashlib
import math
from pathlib import Path
import random
import struct
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from qnormuon import QuotientSpectralOptimizer, SolverConfig, SwiGLUPair, QNorMuon


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 1024
    width: int = 384
    layers: int = 6
    heads: int = 6
    hidden: int = 1024
    sequence_length: int = 128

    def __post_init__(self):
        if min(asdict(self).values()) <= 0 or self.width % self.heads or self.hidden < self.width:
            raise ValueError('positive dimensions, divisible attention heads, tall SwiGLU pairs required')


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        # Float32 reduction under autocast, float64 retained in reference tests.
        work = x if x.dtype == torch.float64 else x.float()
        return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.qkv = nn.Linear(c.width, 3*c.width, bias=False)
        self.out_proj = nn.Linear(c.width, c.width, bias=False)

    def forward(self, x):
        b,t,d = x.shape
        q,k,v = self.qkv(x).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        z = F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0)
        return self.out_proj(z.transpose(1,2).reshape(b,t,d))


class SwiGLU(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.gate_proj = nn.Linear(c.width,c.hidden,bias=False)
        self.up_proj = nn.Linear(c.width,c.hidden,bias=False)
        self.down_proj = nn.Linear(c.hidden,c.width,bias=False)

    def forward(self,x):
        return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))


class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.attention_norm = RMSNorm(c.width)
        self.attention = Attention(c)
        self.mlp_norm = RMSNorm(c.width)
        self.mlp = SwiGLU(c)

    def forward(self,x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class TinyTransformer(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.config=c
        self.token_embedding=nn.Embedding(c.vocab_size,c.width)
        self.position_embedding=nn.Embedding(c.sequence_length,c.width)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.final_norm=RMSNorm(c.width)
        self.output=nn.Linear(c.width,c.vocab_size,bias=False)
        # Nonzero down projections: no singular regular-domain initialization.
        for module in self.modules():
            if isinstance(module,(nn.Linear,nn.Embedding)):
                nn.init.normal_(module.weight,mean=0.,std=.02)
        for block in self.blocks:
            with torch.no_grad():
                block.attention.out_proj.weight.div_(math.sqrt(2*c.layers))
                block.mlp.down_proj.weight.div_(math.sqrt(2*c.layers))

    def forward(self,tokens):
        if tokens.ndim != 2 or tokens.shape[1]>self.config.sequence_length:
            raise ValueError('expected [batch,sequence] within configured context')
        x=self.token_embedding(tokens)+self.position_embedding(torch.arange(tokens.shape[1],device=tokens.device))
        for block in self.blocks:x=block(x)
        return self.output(self.final_norm(x))

    def pairs(self):
        return [SwiGLUPair(f'blocks.{i}.mlp',b.mlp.up_proj.weight,b.mlp.down_proj.weight)
                for i,b in enumerate(self.blocks)]


def seed_everything(seed,tf32=False):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=tf32
    torch.backends.cudnn.allow_tf32=tf32
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)


class TokenStream:
    """Small immutable prefix, never a full-shard loader; stateless batch IDs."""
    def __init__(self,tokens,vocab_size,seed,metadata):
        self.tokens=np.asarray(tokens,dtype=np.int64)
        if self.tokens.ndim!=1 or len(self.tokens)<2 or self.tokens.min()<0 or self.tokens.max()>=vocab_size:
            raise ValueError('invalid token IDs')
        self.seed=seed
        self.metadata=dict(metadata,loaded_tokens=len(tokens),
                           prefix_sha256=hashlib.sha256(self.tokens.astype('<u2').tobytes()).hexdigest())

    @classmethod
    def shard(cls,path,count,vocab_size,seed):
        path=Path(path)
        if count<2 or count>2**22:raise ValueError('bounded prefix must contain 2..4194304 tokens')
        with path.open('rb') as f:
            header=f.read(1024)
            if len(header)!=1024:raise ValueError('short shard header')
            magic,version,total=struct.unpack('<3i',header[:12])
            if magic!=20240520 or version!=1 or total<count or path.stat().st_size!=1024+2*total:
                raise ValueError('invalid token shard header/size')
            tokens=np.frombuffer(f.read(count*2),dtype='<u2').copy()
        if len(tokens)!=count:raise ValueError('short shard prefix')
        return cls(tokens,vocab_size,seed,dict(kind='token_shard_prefix',path=str(path.resolve()),
                                             file_bytes=path.stat().st_size,header_tokens=total))

    @classmethod
    def synthetic(cls,count,vocab_size,seed):
        # Independent streams with learnable periodic/noisy transitions.
        rng=np.random.default_rng(seed)
        t=np.arange(count,dtype=np.int64)
        tokens=(t%17*7+(t//17)%13)%vocab_size
        mask=rng.random(count)<.1
        tokens[mask]=rng.integers(0,vocab_size,mask.sum())
        return cls(tokens,vocab_size,seed,dict(kind='synthetic_periodic_v1',seed=seed))

    def batch(self,index,batch_size,sequence_length,device):
        if index<0 or len(self.tokens)<=sequence_length:raise ValueError('invalid batch index/context')
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,index]))
        starts=rng.integers(0,len(self.tokens)-sequence_length,size=batch_size)
        rows=self.tokens[starts[:,None]+np.arange(sequence_length+1)[None,:]]
        data=torch.from_numpy(rows).to(device=device,dtype=torch.long)
        return data[:,:-1],data[:,1:]


def datasets(config):
    d=config['data'];v=config['model']['vocab_size']
    if d['kind']=='token_shards':
        if not Path(d['tokenizer_path']).is_file():raise FileNotFoundError(d['tokenizer_path'])
        return (TokenStream.shard(d['train_path'],d['train_tokens'],v,d['seed']),
                TokenStream.shard(d['validation_path'],d['validation_tokens'],v,d['seed']+1))
    if d['kind']=='synthetic':
        return (TokenStream.synthetic(d['train_tokens'],v,d['seed']),
                TokenStream.synthetic(d['validation_tokens'],v,d['seed']+1))
    raise ValueError('explicit local token_shards or synthetic data required')


class Optimizers:
    def __init__(self,model,method,config):
        self.method=method;self.paired=None
        pairs=model.pairs();ids={id(t) for p in pairs for t in (p.up,p.down)}
        adam_params=list(model.parameters()) if method=='adamw' else [p for p in model.parameters() if id(p) not in ids]
        self.adam=torch.optim.AdamW(adam_params,lr=config['adam_lr'],betas=tuple(config['adam_betas']),
                                   weight_decay=config['weight_decay'],foreach=False,fused=False)
        if method=='qso':
            sc=dict(config['solver']);sc['dtype']=getattr(torch,sc['dtype'])
            self.paired=QuotientSpectralOptimizer(pairs,lr=config['qso_lr'],beta=config['beta'],solver=SolverConfig(**sc))
        elif method=='historical_qnormuon':
            # Faithful unchanged local implementation, explicitly historical.
            # No new shared-K logic is implemented in this harness.
            self.paired=QNorMuon([(p.up,p.down) for p in pairs],lr=config['historical_lr'],beta1=config['beta'])
        elif method!='adamw':raise ValueError('Muon omitted: no verified baseline integration')
        self.all=([self.paired] if self.paired else [])+[self.adam]
        self.base_lrs=[[g['lr'] for g in o.param_groups] for o in self.all]
        self.schedule=dict(steps=config['steps'],warmup=config['warmup_steps'],minimum=config['minimum_lr_fraction'])
        self.next_step=0

    def set_step(self,step):
        self.next_step=step
        for optimizer,rates in zip(self.all,self.base_lrs):
            factor=lr_factor(step,**self.schedule)
            for group,base in zip(optimizer.param_groups,rates):group['lr']=base*factor

    def zero_grad(self):
        for o in self.all:o.zero_grad(set_to_none=True)

    def step(self):
        for o in self.all:o.step()
        self.next_step+=1

    def state_dict(self):
        return dict(method=self.method,optimizers=[o.state_dict() for o in self.all],
                    base_lrs=self.base_lrs,schedule=self.schedule,next_step=self.next_step)

    def load_state_dict(self,state):
        if state['method']!=self.method or len(state['optimizers'])!=len(self.all):raise ValueError('optimizer topology mismatch')
        for o,s in zip(self.all,state['optimizers']):o.load_state_dict(s)
        self.base_lrs=state['base_lrs'];self.schedule=state['schedule'];self.next_step=state['next_step']


def lr_factor(step,steps,warmup,minimum):
    if not 0<=step<steps or not 0<=warmup<steps:raise ValueError('invalid schedule step')
    if step<warmup:return (step+1)/max(1,warmup)
    phase=(step-warmup)/max(1,steps-warmup-1)
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*phase))


def autocast(model,precision):
    device=next(model.parameters()).device.type
    if precision=='bf16_autocast_fp32_storage':return torch.autocast(device_type=device,dtype=torch.bfloat16)
    if precision=='float32':return nullcontext()
    raise ValueError('unknown precision')


def synchronize(model):
    if next(model.parameters()).is_cuda:torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model,data,config):
    model.eval();losses=[]
    for i in range(config['evaluation_batches']):
        x,y=data.batch(i,config['batch_size'],config['model']['sequence_length'],next(model.parameters()).device)
        with autocast(model,config['precision']):logits=model(x)
        losses.append(float(F.cross_entropy(logits.float().flatten(0,1),y.flatten())))
    model.train()
    return sum(losses)/len(losses)


def rms(values):
    total=sum(v.numel() for v in values)
    return math.sqrt(sum(float(v.detach().double().square().sum()) for v in values)/total) if total else 0.


def finite_delta_x(u,d,new_u,new_d):
    """Exact finite outer-product difference norm using three rank-one terms.

    deltaX = delta_d u^T + d delta_u^T + delta_d delta_u^T.
    Storage O(m*n), with per-row 3x3 Gram products, never m*n*n.
    """
    u,d,new_u,new_d=(v.detach().double() for v in (u,d,new_u,new_d))
    du,dd=new_u-u,new_d-d
    left=torch.stack((dd,d,dd));right=torch.stack((u,du,du))
    gl=torch.einsum('kin,lin->ikl',left,left)
    gr=torch.einsum('kin,lin->ikl',right,right)
    squared=float((gl*gr).sum().clamp_min(0))
    base=float((u.square().sum(1)*d.square().sum(1)).sum())
    return dict(finite_delta_x_norm=math.sqrt(squared),finite_delta_x_relative=math.sqrt(squared/base) if base else None)


def train_step(model,opts,data,config,step):
    synchronize(model);started=time.perf_counter()
    opts.set_step(step);opts.zero_grad()
    params=list(model.parameters())
    before=[p.detach().clone() for p in params]
    first=model.pairs()[0];old_u=first.up.detach().clone();old_d=first.down.detach().T.clone()
    losses=[];batch_hash=hashlib.sha256()
    compute_start=time.perf_counter()
    for micro in range(config['gradient_accumulation']):
        x,y=data.batch(step*config['gradient_accumulation']+micro,config['batch_size'],
                       config['model']['sequence_length'],params[0].device)
        batch_hash.update(x.cpu().numpy().tobytes());batch_hash.update(y.cpu().numpy().tobytes())
        with autocast(model,config['precision']):logits=model(x)
        loss=F.cross_entropy(logits.float().flatten(0,1),y.flatten())
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        losses.append(float(loss.detach()));(loss/config['gradient_accumulation']).backward()
    synchronize(model);compute_seconds=time.perf_counter()-compute_start
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise FloatingPointError('missing/nonfinite gradient')
    grad_rms=rms([p.grad for p in params]);parameter_rms=rms(before)
    synchronize(model);optimizer_start=time.perf_counter();opts.step();synchronize(model)
    optimizer_seconds=time.perf_counter()-optimizer_start
    deltas=[p.detach()-b for p,b in zip(params,before)]
    if any(not torch.isfinite(p).all() for p in [*params,*deltas]):raise FloatingPointError('nonfinite parameter/update')
    update_rms=rms(deltas)
    stats=dict(step=step,loss=sum(losses)/len(losses),batch_sha256=batch_hash.hexdigest(),
               gradient_rms=grad_rms,parameter_rms=parameter_rms,update_rms=update_rms,
               update_parameter_ratio=update_rms/parameter_rms,optimizer_seconds=optimizer_seconds,
               forward_backward_seconds=compute_seconds,lr_factor=lr_factor(step,**opts.schedule))
    stats.update(finite_delta_x(old_u,old_d,first.up,first.down.T))
    synchronize(model);stats['step_seconds']=time.perf_counter()-started
    stats['optimizer_fraction']=optimizer_seconds/stats['step_seconds']
    stats['compute_optimizer_fraction']=optimizer_seconds/(optimizer_seconds+compute_seconds)
    count=config['batch_size']*config['model']['sequence_length']*config['gradient_accumulation']
    stats['tokens']=(step+1)*count;stats['tokens_per_second']=count/stats['step_seconds']
    if params[0].is_cuda:
        stats['cuda_allocated_bytes']=torch.cuda.memory_allocated()
        stats['cuda_peak_bytes']=torch.cuda.max_memory_allocated()
    stats['qso']=copy_diagnostics(opts)
    return stats


def copy_diagnostics(opts):
    if opts.method!='qso':return {}
    return {name:dict(value) for name,value in opts.paired.last_diagnostics.items()}


def save_checkpoint(path,model,opts,config,data_metadata):
    state=dict(model=model.state_dict(),optimizers=opts.state_dict(),config=config,
               data_metadata=data_metadata,torch_rng=torch.get_rng_state(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               python_rng=random.getstate(),numpy_rng=np.random.get_state())
    temp=Path(str(path)+'.tmp');torch.save(state,temp);temp.replace(path)


def load_checkpoint(path,model,opts,config,data_metadata):
    # Only load this harness's own local checkpoints, never untrusted payloads.
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['config']!=config or state['data_metadata']!=data_metadata:raise ValueError('checkpoint configuration/data mismatch')
    model.load_state_dict(state['model']);opts.load_state_dict(state['optimizers'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    return opts.next_step


@torch.no_grad()
def positive_gauge_reset(model,opts,seed=902,log2_span=2.):
    """QSO canonical state unchanged; Adam raw moments transform covariantly.

    Adam exp_avg is a covector; exp_avg_sq follows its squared scaling.
    Adam remains a non-equivariant control even with these transformations.
    """
    rng=torch.Generator(device='cpu').manual_seed(seed)
    scales={}
    for p in model.pairs():
        c=2**((torch.rand(p.up.shape[0],generator=rng,dtype=torch.float64)*2-1)*log2_span)
        c=c.to(device=p.up.device,dtype=p.up.dtype)
        p.up.mul_(c[:,None]);p.down.div_(c[None,:]);scales[p.name]=c
        if opts.method=='adamw':
            for weight,factor in ((p.up,1/c[:,None]),(p.down,c[None,:])):
                state=opts.adam.state.get(weight,{})
                if 'exp_avg' in state:state['exp_avg'].mul_(factor)
                if 'exp_avg_sq' in state:state['exp_avg_sq'].mul_(factor.square())
        elif opts.method!='qso':raise ValueError('gauge experiment supports QSO and AdamW')
    return scales


def summarize(values):
    values=np.asarray(values,dtype=np.float64)
    if not len(values):return {}
    return dict(mean=float(values.mean()),median=float(np.median(values)),p95=float(np.percentile(values,95)),max=float(values.max()))


def solver_summary(records):
    rows=[d for r in records for d in r.get('qso',{}).values()]
    keys=['newton_iterations','cg_iterations','normalized_gap','horizontal_residual','spectral_excess','residual_rcond']
    return dict(solves=len(rows),fallbacks=sum(r['fallback_used'] for r in rows),
                fallback_percent=100*sum(r['fallback_used'] for r in rows)/len(rows) if rows else 0,
                iteration_bins={str(n):sum(r['newton_iterations']==n for r in rows) for n in (0,1,2)} |
                               {'3+':sum(r['newton_iterations']>=3 for r in rows)},
                distributions={k:summarize([r[k] for r in rows]) for k in keys})
