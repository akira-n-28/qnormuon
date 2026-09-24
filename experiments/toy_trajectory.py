"""Compare functional trajectories of QNorMuon vs raw spectral Muon under a gauge reset."""

import copy
import torch
import torch.nn.functional as F

from qnormuon import QNorMuon, exact_polar


torch.set_default_dtype(torch.float64)


class PairModel(torch.nn.Module):
    def __init__(self, n=6, m=18, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.up = torch.nn.Parameter(torch.randn(m, n, generator=g) / n**0.5)
        self.down = torch.nn.Parameter(torch.randn(n, m, generator=g) / m**0.5)
        self.register_buffer("gate", torch.randn(m, n, generator=g) / n**0.5)

    def forward(self, x):
        gate = F.silu(x @ self.gate.T)
        return (gate * (x @ self.up.T)) @ self.down.T


class RawMuon:
    """Tiny diagnostic baseline: momentum + exact polar in raw coordinates."""
    def __init__(self, model, lr=3e-3, beta=0.9):
        self.model = model
        self.lr = lr
        self.beta = beta
        self.mu = torch.zeros_like(model.up)
        self.md = torch.zeros_like(model.down.T)

    @torch.no_grad()
    def step(self):
        self.mu.mul_(self.beta).add_(self.model.up.grad, alpha=1-self.beta)
        self.md.mul_(self.beta).add_(self.model.down.grad.T, alpha=1-self.beta)
        pu = exact_polar(self.mu)
        pd = exact_polar(self.md)
        self.model.up.add_(pu, alpha=-self.lr)
        self.model.down.add_(pd.T, alpha=-self.lr)


def gauged_copy(model, span=3.0, seed=7):
    other = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    z = (2 * torch.rand(model.up.shape[0], generator=g) - 1) * span
    c = 10.0 ** z
    with torch.no_grad():
        other.up.mul_(c[:, None])
        other.down.mul_((1.0/c)[None, :])
    return other, c


def train_pair(kind="q", steps=20):
    base = PairModel(seed=1)
    gauged, _ = gauged_copy(base)

    if kind == "q":
        oa = QNorMuon([(base.up, base.down)], lr=3e-3, beta1=0.9,
                      balance_lr=0.05, balance_beta=0.9)
        ob = QNorMuon([(gauged.up, gauged.down)], lr=3e-3, beta1=0.9,
                      balance_lr=0.05, balance_beta=0.9)
    else:
        oa = RawMuon(base)
        ob = RawMuon(gauged)

    g = torch.Generator().manual_seed(3)
    x = torch.randn(64, base.up.shape[1], generator=g)
    target = torch.randn(64, base.down.shape[0], generator=g)

    initial = (base(x) - gauged(x)).norm() / base(x).norm()
    max_mismatch = initial.item()
    for _ in range(steps):
        for model, opt in [(base, oa), (gauged, ob)]:
            model.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), target)
            loss.backward()
            opt.step()
        mismatch = (base(x) - gauged(x)).norm() / base(x).norm()
        max_mismatch = max(max_mismatch, mismatch.item())

    final = (base(x) - gauged(x)).norm() / base(x).norm()
    return initial.item(), final.item(), max_mismatch


for kind, label in [("q", "QNorMuon"), ("raw", "raw Muon")]:
    initial, final, maximum = train_pair(kind)
    print(f"{label:10s} | initial={initial:.3e} final={final:.3e} max={maximum:.3e}")
