"""Stage-D rules: fixed grids, single-pass inputs, and quality-only selection."""
import hashlib
import math
import numpy as np
import torch

ADAM_GRID = [1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1.2e-3]
QSO_GRID = [3e-4, 6e-4, 1e-3, 1.5e-3, 2.5e-3, 4e-3]
INITIAL_HASH = '0607adc4c9cffa7b902c0956fa7477ea274078fdc31406d84d527b79ca6a2401'


class SinglePassStream:
    """Permutation of disjoint input blocks; one extra token supplies lookahead.

    No input position repeats. Target overlap with adjacent inputs is inherent
    in next-token prediction, not another training pass. No wrapping permitted.
    Validation continues to use the unchanged smoke TokenStream.batch.
    """
    def __init__(self, source, sequence_length, seed):
        self.tokens = source.tokens
        self.length = sequence_length
        self.blocks = (len(self.tokens)-1)//sequence_length
        self.order = np.random.default_rng(seed).permutation(self.blocks)
        self.metadata = dict(source.metadata, sampling='single_pass_blocks_v1',
                             usable_input_tokens=self.blocks*sequence_length,
                             order_sha256=hashlib.sha256(self.order.astype('<i8').tobytes()).hexdigest())

    def starts(self, index, batch_size):
        lo, hi = index*batch_size, (index+1)*batch_size
        if index < 0 or hi > self.blocks:
            raise ValueError('single-pass training prefix exhausted; no wrapping')
        return self.order[lo:hi]*self.length

    def batch(self, index, batch_size, sequence_length, device):
        if sequence_length != self.length:
            raise ValueError('context differs from locked block order')
        rows = self.tokens[self.starts(index, batch_size)[:, None] + np.arange(self.length+1)]
        data = torch.from_numpy(rows).to(device=device, dtype=torch.long)
        return data[:, :-1], data[:, 1:]


def batch_digest(data, index, config):
    h = hashlib.sha256()
    for micro in range(config['gradient_accumulation']):
        x, y = data.batch(index*config['gradient_accumulation']+micro,
                          config['batch_size'], config['model']['sequence_length'], 'cpu')
        h.update(x.numpy().tobytes()); h.update(y.numpy().tobytes())
    return h.hexdigest()


def selection_key(summary):
    if summary['status'] != 'passed':
        return (math.inf,)*4
    return (summary['last_three_validation_mean'], summary['final_validation_loss'],
            summary['post_initial_validation_mean'], summary['max_update_parameter_ratio'])


def select(rows):
    eligible = [r for r in rows if r['status'] == 'passed']
    if not eligible:
        raise RuntimeError('no completed quality-eligible candidate')
    return min(eligible, key=selection_key)


def refinement(grid, winner):
    i = grid.index(winner)
    if i == 0:
        return [winner/math.sqrt(grid[1]/winner), math.sqrt(winner*grid[1])]
    if i == len(grid)-1:
        return [math.sqrt(grid[-2]*winner), winner*math.sqrt(winner/grid[-2])]
    return [math.sqrt(grid[i-1]*winner), math.sqrt(winner*grid[i+1])]


def quality(evaluations, records):
    points = sorted(evaluations.items())
    if not points or points[0][0] != 0 or len(points) < 4:
        raise ValueError('initial and common validation checkpoints required')
    post = [v for k, v in points if k]
    # Equal spacing; normalized trapezoidal AUC includes the initial checkpoint.
    auc_mean = (points[0][1]/2 + sum(post[:-1]) + post[-1]/2)/(len(points)-1)
    return dict(final_validation_loss=post[-1], minimum_validation_loss=min(v for _, v in points),
                post_initial_validation_mean=float(np.mean(post)),
                last_three_validation_mean=float(np.mean(post[-3:])),
                validation_auc_mean=auc_mean,
                max_update_parameter_ratio=max(r['update_parameter_ratio'] for r in records))


def exact_tree(a, b):
    """Checkpoint audit: require exact dtype, shape, topology and tensor values."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.cpu(), b.cpu()), 'checkpoint tensor differs'
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: exact_tree(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) == type(b) and len(a) == len(b)
        for x, y in zip(a, b): exact_tree(x, y)
    else:
        assert a == b
