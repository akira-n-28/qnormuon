"""Research-only predictor invariants; independent of the certified solver."""
import io

import pytest
import torch

from experiments.dual_warm_start_predictor import (
    ProblemScale, clone_state, pack_state, predict, problem_scale, unpack_state,
)
from qnormuon.optimizer import regular_canonicalize


def scale(beta, magnitude=2., coord=None):
    beta = torch.tensor(beta, dtype=torch.float64)
    return ProblemScale(beta, magnitude, torch.ones_like(beta) if coord is None else
                        torch.tensor(coord, dtype=torch.float64))


def test_original_and_centered_coordinate_conventions():
    old = clone_state(torch.tensor([5., 9.], dtype=torch.float64), scale([1., 3.], 2., [.5, 1.]))
    older = clone_state(torch.tensor([3., 5.], dtype=torch.float64), scale([1., 1.], 4., [.25, 2.]))
    current = scale([2., 4.], 3., [2., .5])
    assert torch.equal(predict("center", current, old, older), current.beta)
    assert torch.equal(predict("previous", current, old, older), old.lam)
    assert torch.equal(predict("raw:0.5", current, old, older), torch.tensor([6., 11.], dtype=torch.float64))
    q_old = (old.lam-old.scale.beta)/old.scale.magnitude
    q_older = (older.lam-older.scale.beta)/older.scale.magnitude
    assert torch.allclose(predict("centered:0.5", current, old, older),
                          current.beta+current.magnitude*(q_old+.5*(q_old-q_older)))
    z_old = q_old/old.scale.coord
    z_older = q_older/older.scale.coord
    assert torch.allclose(predict("whitened:0.5", current, old, older),
                          current.beta+current.magnitude*current.coord*(z_old+.5*(z_old-z_older)))


def test_first_momentum_step_and_abrupt_change_do_not_use_future_solution():
    current = scale([11., -4.], .05)
    for policy in ("center", "previous", "raw:1", "centered:1", "whitened:1"):
        assert torch.equal(predict(policy, current, None), current.beta)
    previous = clone_state(torch.tensor([1., 2.], dtype=torch.float64), scale([3., 4.], .1))
    for policy in ("previous", "raw:1", "centered:1", "whitened:1"):
        assert torch.isfinite(predict(policy, current, previous)).all()
    with pytest.raises(ValueError, match="unknown predictor"):
        predict("oracle", current, previous)


def test_compact_state_checkpoint_reproduces_fixed_predictor():
    older = clone_state(torch.tensor([1., -3.], dtype=torch.float64), scale([2., 1.], .7))
    previous = clone_state(torch.tensor([4., 2.], dtype=torch.float64), scale([3., 0.], 1.2))
    current = scale([5., 1.], 2.)
    buffer = io.BytesIO()
    torch.save(dict(older=pack_state(older), previous=pack_state(previous)), buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    for policy in ("raw:0.5", "centered:0.5", "whitened:0.5"):
        assert torch.equal(predict(policy, current, previous, older),
                           predict(policy, current, unpack_state(loaded["previous"]),
                                   unpack_state(loaded["older"])))
    assert sum(v.numel() for state in loaded.values() for v in state.values()
               if isinstance(v, torch.Tensor)) == 12  # two states, three length-2 vectors each


def test_positive_gauge_preserves_canonical_predictor_inputs():
    gen = torch.Generator().manual_seed(913)
    up = torch.randn(9, 4, generator=gen, dtype=torch.float64)
    down = torch.randn(4, 9, generator=gen, dtype=torch.float64)
    gu = torch.randn_like(up)
    gd = torch.randn_like(down)
    gauge = torch.exp(torch.linspace(-2., 2., 9, dtype=torch.float64))
    uc, dc, root = regular_canonicalize(up, down, dtype=torch.float64)
    u2, d2, root2 = regular_canonicalize(gauge[:, None]*up,
                                         down/gauge[None, :], dtype=torch.float64)
    a = torch.stack((root[:,None]*gu, gd.T/root[:,None]))
    a2 = torch.stack((root2[:,None]*(gu/gauge[:,None]),
                      (gd*gauge[None,:]).T/root2[:,None]))
    first = problem_scale(uc, dc, a)
    second = problem_scale(u2, d2, a2)
    assert torch.allclose(uc, u2, atol=1e-14, rtol=1e-14)
    assert torch.allclose(dc, d2, atol=1e-14, rtol=1e-14)
    assert torch.allclose(first.beta, second.beta, atol=1e-14, rtol=1e-14)
    assert first.magnitude == pytest.approx(second.magnitude, rel=1e-14)
    history = clone_state(first.beta+.1, first)
    assert torch.allclose(predict("centered:0.5", first, history),
                          predict("centered:0.5", second, history), atol=1e-14, rtol=1e-14)
