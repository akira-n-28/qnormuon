"""Small harness correctness checks; run in a compute allocation on Lagrange."""
import copy
import struct
import numpy as np
import pytest
import torch
from benchmarks.tiny_transformer import (ModelConfig,TinyTransformer,TokenStream,Optimizers,
    train_step,finite_delta_x,positive_gauge_reset,save_checkpoint,load_checkpoint,lr_factor)


def config():
    return dict(model=dict(vocab_size=32,width=8,layers=1,heads=2,hidden=16,sequence_length=8),
                steps=6,warmup_steps=1,minimum_lr_fraction=.1,batch_size=4,gradient_accumulation=1,
                precision='float32',adam_lr=3e-4,qso_lr=1e-3,historical_lr=1e-3,beta=.95,
                adam_betas=[.9,.95],weight_decay=0.,solver=dict(dtype='float64'),evaluation_batches=1)


def test_causal_mask_and_nonzero_explicit_pairs():
    torch.manual_seed(10);model=TinyTransformer(ModelConfig(**config()['model'])).double()
    a=torch.arange(8).reshape(1,8);b=a.clone();b[:,4:]=(b[:,4:]+3)%32
    torch.testing.assert_close(model(a)[:,:4],model(b)[:,:4],rtol=0,atol=0)
    for p in model.pairs():
        assert p.down.shape==p.up.T.shape
        assert (p.up.norm(dim=1)>0).all() and (p.down.norm(dim=0)>0).all()


def test_real_shard_prefix_and_missing_assets_fail(tmp_path):
    path=tmp_path/'tokens.bin';header=np.zeros(256,dtype='<i4');header[:3]=[20240520,1,50]
    path.write_bytes(header.tobytes()+np.arange(50,dtype='<u2').tobytes())
    data=TokenStream.shard(path,20,64,1)
    assert len(data.tokens)==20 and data.metadata['file_bytes']==1124
    with pytest.raises(FileNotFoundError):TokenStream.shard(tmp_path/'missing',20,64,1)
    path.write_bytes(b'bad')
    with pytest.raises(ValueError):TokenStream.shard(path,20,64,1)


def test_stateless_identical_batches_and_validation_independence():
    a=TokenStream.synthetic(1000,32,10);b=TokenStream.synthetic(1000,32,11)
    a0=a.batch(0,4,8,'cpu');a.batch(11,4,8,'cpu')
    assert torch.equal(a0[0],a.batch(0,4,8,'cpu')[0])
    assert not torch.equal(a0[0],b.batch(0,4,8,'cpu')[0])
    assert torch.equal(a0[0][:,1:],a0[1][:,:-1])


@pytest.mark.parametrize('method',['adamw','qso','historical_qnormuon'])
def test_optimizer_partition_exact_and_no_gate_pairing(method):
    c=config();m=TinyTransformer(ModelConfig(**c['model'])).float();o=Optimizers(m,method,c)
    ids=[id(p) for opt in o.all for group in opt.param_groups for p in group['params']]
    assert len(ids)==len(set(ids))==len(list(m.parameters()))
    if o.paired:
        paired={id(p) for g in o.paired.param_groups for p in g['params']}
        expected={id(t) for p in m.pairs() for t in (p.up,p.down)}
        assert paired==expected
        assert id(m.blocks[0].mlp.gate_proj.weight) not in paired


def test_finite_delta_x_matches_explicit_outer_products():
    g=torch.Generator().manual_seed(100)
    u,d,du,dd=[torch.randn(5,3,generator=g,dtype=torch.float64) for _ in range(4)]
    new_u,new_d=u+.01*du,d+.01*dd
    outer=lambda d,u:torch.einsum('mi,mj->mij',d,u)
    measured=finite_delta_x(u,d,new_u,new_d)
    exact=outer(new_d,new_u)-outer(d,u)
    assert measured['finite_delta_x_norm']==pytest.approx(float(exact.norm()),rel=1e-10)


@pytest.mark.parametrize('method',['adamw','qso'])
def test_real_training_checkpoint_restores_model_optimizer_schedule_rng(tmp_path,method):
    c=config();torch.manual_seed(100);m=TinyTransformer(ModelConfig(**c['model'])).float()
    o=Optimizers(m,method,c);data=TokenStream.synthetic(2000,32,9)
    train_step(m,o,data,c,0)
    path=tmp_path/'own.pt';save_checkpoint(path,m,o,c,data.metadata)
    expected=[train_step(m,o,data,c,i) for i in (1,2)]
    n=TinyTransformer(ModelConfig(**c['model'])).float();p=Optimizers(n,method,c)
    assert load_checkpoint(path,n,p,c,data.metadata)==1
    actual=[train_step(n,p,data,c,i) for i in (1,2)]
    for x,y in zip(m.parameters(),n.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert [r['loss'] for r in expected]==[r['loss'] for r in actual]
    assert p.next_step==o.next_step==3


@pytest.mark.parametrize('method',['adamw','qso'])
def test_gauge_reset_function_and_state_contract(method):
    c=config();torch.manual_seed(21);m=TinyTransformer(ModelConfig(**c['model'])).double()
    o=Optimizers(m,method,c);data=TokenStream.synthetic(2000,32,2)
    train_step(m,o,data,c,0)
    a,b=data.batch(2,4,8,'cpu');before=m(a).detach()
    old=copy.deepcopy(o.state_dict())
    scales=positive_gauge_reset(m,o,seed=13,log2_span=2)
    torch.testing.assert_close(m(a),before,rtol=1e-11,atol=1e-11)
    if method=='qso':
        for pid,state in old['optimizers'][0]['state'].items():
            for key in ('momentum_up','momentum_down_t','lambda'):
                assert torch.equal(state[key],o.state_dict()['optimizers'][0]['state'][pid][key])
    else:
        assert scales and all(torch.isfinite(v).all() for s in o.adam.state.values() for v in s.values() if isinstance(v,torch.Tensor))


def test_warmup_and_cosine_endpoints():
    assert lr_factor(0,50,5,.1)==.2
    assert lr_factor(4,50,5,.1)==1
    assert lr_factor(5,50,5,.1)==1
    assert lr_factor(49,50,5,.1)==pytest.approx(.1)
