"""Independent Stage-D data and predeclared quality-selection regressions."""
import numpy as np
import pytest
import torch
from benchmarks.tiny_transformer import TokenStream,lr_factor
from benchmarks.lr_study import SinglePassStream,refinement,select,quality,exact_tree


def test_single_pass_uses_each_input_once_no_wrap_and_next_token_labels():
    source=TokenStream(np.arange(129),256,1,{})
    data=SinglePassStream(source,8,2026)
    batches=[data.batch(i,4,8,'cpu') for i in range(4)]
    inputs=torch.cat([x.flatten() for x,_ in batches])
    assert sorted(inputs.tolist())==list(range(128))
    for x,y in batches:assert torch.equal(y,x+1)
    with pytest.raises(ValueError,match='exhausted'):data.batch(4,4,8,'cpu')
    assert np.array_equal(data.order,SinglePassStream(source,8,2026).order)


@pytest.mark.parametrize('winner,expected',[(1,[1/np.sqrt(2),np.sqrt(2)]),
    (2,[np.sqrt(2),np.sqrt(8)]),(4,[np.sqrt(8),4*np.sqrt(2)])])
def test_predeclared_geometric_refinement(winner,expected):
    assert refinement([1,2,4],winner)==pytest.approx(expected)


def test_selection_never_uses_training_loss_or_wall_time():
    a=dict(status='passed',last_three_validation_mean=4.,final_validation_loss=3.,
           post_initial_validation_mean=5.,max_update_parameter_ratio=.1,loss=10,time=100)
    b=dict(a,last_three_validation_mean=4.1,loss=1,time=.01)
    assert select([b,a]) is a
    failed=dict(a,status='failed',last_three_validation_mean=0.)
    assert select([failed,b,a]) is a
    tie=dict(a,max_update_parameter_ratio=.05)
    assert select([a,tie]) is tie


def test_quality_checkpoints_and_exact_checkpoint_comparison():
    values={0:8.,32:7.,64:6.,96:5.}
    result=quality(values,[{'update_parameter_ratio':.1}])
    assert result['last_three_validation_mean']==6.
    assert result['validation_auc_mean']==6.5
    assert result['post_initial_validation_mean']==6.
    exact_tree({'lambda':torch.ones(2,dtype=torch.float64)}, {'lambda':torch.ones(2,dtype=torch.float64)})
    with pytest.raises(AssertionError):exact_tree(torch.ones(2),torch.ones(2)+1e-6)


@pytest.mark.parametrize('steps,warmup',[(256,26),(512,51)])
def test_fixed_stage_d_schedule(steps,warmup):
    assert lr_factor(0,steps,warmup,.1)==1/warmup
    assert lr_factor(warmup-1,steps,warmup,.1)==1
    assert lr_factor(steps-1,steps,warmup,.1)==pytest.approx(.1)
