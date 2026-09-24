"""Run a human-readable gauge stress test for QMuon."""

import torch

from qnormuon import quotient_polar_update


torch.set_default_dtype(torch.float64)

g = torch.Generator().manual_seed(123)
m, n = 32, 8
up = torch.randn(m, n, generator=g)
down = torch.randn(n, m, generator=g)
gu = torch.randn(m, n, generator=g)
gd = torch.randn(n, m, generator=g)
x = torch.randn(m, generator=g)

# Extreme but still numerically reasonable positive diagonal gauge.
z = (2 * torch.rand(m, generator=g) - 1) * 4.0
c = 10.0 ** z

up2 = c[:, None] * up
down2 = down * (1.0 / c)[None, :]
gu2 = (1.0 / c)[:, None] * gu
gd2 = gd * c[None, :]

du1, dd1, pu1, pd1, _ = quotient_polar_update(up, down, gu, gd, x=x)
du2, dd2, pu2, pd2, _ = quotient_polar_update(up2, down2, gu2, gd2, x=x)

rel_up = torch.linalg.vector_norm(du2 - c[:, None] * du1) / torch.linalg.vector_norm(du2)
rel_down = torch.linalg.vector_norm(dd2 - dd1 * (1.0 / c)[None, :]) / torch.linalg.vector_norm(dd2)
canon_pu = torch.linalg.vector_norm(pu2 - pu1) / torch.linalg.vector_norm(pu1)
canon_pd = torch.linalg.vector_norm(pd2 - pd1) / torch.linalg.vector_norm(pd1)

print(f"gauge dynamic range: 1e{z.min().item():.2f} .. 1e{z.max().item():.2f}")
print(f"relative equivariance error, up:   {rel_up.item():.3e}")
print(f"relative equivariance error, down: {rel_down.item():.3e}")
print(f"canonical polar mismatch, up:      {canon_pu.item():.3e}")
print(f"canonical polar mismatch, down:    {canon_pd.item():.3e}")
