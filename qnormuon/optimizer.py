"""Explicit paired K=I optimizer; historical QNorMuon remains separate."""
from __future__ import annotations
import copy
import math
from dataclasses import asdict, dataclass
import torch
from torch import Tensor
from torch.optim import Optimizer
from .coupled_solver import SolverConfig, solve_coupled, horizontal_residual


@dataclass(frozen=True)
class SwiGLUPair:
    """Stable checkpoint name and actual up [m,n], down [n,m] weights."""
    name: str
    up: Tensor
    down: Tensor

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("pair requires a nonempty stable name")
        if self.up.ndim != 2 or self.down.shape != self.up.T.shape:
            raise ValueError(f"{self.name}: expected up [m,n], down [n,m]")
        if not 1 <= self.up.shape[1] <= self.up.shape[0]:
            raise ValueError("require m >= n >= 1")
        if self.up.device != self.down.device or self.up.dtype != self.down.dtype:
            raise ValueError("paired device and dtype must match")
        if self.up.dtype not in (torch.float64, torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("floating model parameters required")


def _row_norm(matrix):
    scale = matrix.abs().amax(1)
    if bool((scale == 0).any()):
        rows = (scale == 0).nonzero().flatten().tolist()
        raise ValueError(f"regular domain requires nonzero paired rows; zero rows: {rows}")
    return scale * (matrix / scale[:, None]).square().sum(1).sqrt()


@torch.no_grad()
def regular_canonicalize(up, down, *, dtype=torch.float32):
    """Return U_bar, D_bar, sqrt(h), without materializing h or clamping.

    Scaled float64 norms and sqrt(r)/sqrt(s) avoid raw-square overflow.
    Canonical weights have the requested representation dtype; the lift is float64.
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("canonical arithmetic requires float32 or float64")
    if up.ndim != 2 or down.shape != up.T.shape:
        raise ValueError("expected up [m,n], down [n,m]")
    u, d = up.double(), down.T.double()
    if not torch.isfinite(u).all() or not torch.isfinite(d).all():
        raise ValueError("nonfinite paired weights")
    r, s = _row_norm(u), _row_norm(d)
    root = r.sqrt() / s.sqrt()
    uc, dc = (u / root[:, None]).to(dtype), (d * root[:, None]).to(dtype)
    if not all(bool(torch.isfinite(v).all()) for v in (root, uc, dc)):
        raise ValueError("canonical factors or lift are not representable")
    if bool((root == 0).any()) or bool((uc.abs().amax(1) == 0).any()) or bool((dc.abs().amax(1) == 0).any()):
        raise ValueError("canonical factors underflowed; choose float64 arithmetic")
    return uc, dc, root


class QuotientSpectralOptimizer(Optimizer):
    """EMA M <- beta*M + (1-beta)*G_c, then coupled LMO on updated M.

    No bias correction, Nesterov, weight decay or balancing is implicit.
    Missing both gradients skips a pair; one missing gradient is an error.
    """
    def __init__(self, pairs, *, lr=1e-3, beta=0.95, solver=None, momentum_dtype=torch.float32,
                 diagnostics=True, cast_diagnostics=False):
        if not math.isfinite(lr) or lr < 0 or not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError("require finite lr >= 0 and beta in [0,1)")
        if momentum_dtype not in (torch.float32, torch.float64):
            raise ValueError("momentum dtype must be float32 or float64")
        if cast_diagnostics and not diagnostics:
            raise ValueError("cast diagnostics require diagnostics=True")
        pairs = list(pairs)
        if not pairs or any(not isinstance(p, SwiGLUPair) for p in pairs):
            raise ValueError("register a nonempty sequence of named SwiGLUPair objects")
        self.pairs = tuple(sorted(pairs, key=lambda p: p.name))
        if len({p.name for p in pairs}) != len(pairs):
            raise ValueError("duplicate pair names")
        ids = [id(t) for p in pairs for t in (p.up, p.down)]
        if len(set(ids)) != len(ids):
            raise ValueError("a parameter may belong to only one pair")
        self.record_diagnostics = diagnostics
        self.record_cast_diagnostics = cast_diagnostics
        self.last_diagnostics = {}
        self._registration_closed = False
        groups = [dict(params=[p.up, p.down], pair_name=p.name) for p in self.pairs]
        super().__init__(groups, dict(lr=lr, beta=beta, momentum_dtype=momentum_dtype, solver=asdict(solver or SolverConfig())))
        self._registration_closed = True

    def add_param_group(self, param_group):
        if getattr(self, "_registration_closed", False):
            raise ValueError("pair topology is fixed; construct a new optimizer for new pairs")
        super().add_param_group(param_group)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        pending, diagnostics = [], {}
        for pair, group in zip(self.pairs, self.param_groups):
            up, down = pair.up, pair.down
            if up.grad is None and down.grad is None:
                continue
            if up.grad is None or down.grad is None:
                raise ValueError(f"{pair.name}: both paired gradients are required")
            if up.grad.is_sparse or down.grad.is_sparse:
                raise ValueError("sparse gradients are unsupported")
            config = SolverConfig(**group["solver"])
            momentum_dtype = group["momentum_dtype"]
            if momentum_dtype not in (torch.float32, torch.float64):
                raise ValueError("momentum dtype must be float32 or float64")
            beta, lr = group["beta"], group["lr"]
            if not math.isfinite(lr) or lr < 0 or not 0 <= beta < 1:
                raise ValueError("invalid optimizer group controls")
            with torch.autocast(device_type=up.device.type, enabled=False):
                u, d, root = regular_canonicalize(up, down, dtype=momentum_dtype)
                # Exactly one raw-to-canonical covector transformation.
                gu = (root[:, None] * up.grad.double()).to(momentum_dtype)
                gd = (down.grad.T.double() / root[:, None]).to(momentum_dtype)
                old = self.state.get(up, {})
                mu = beta * old.get("momentum_up", torch.zeros_like(gu)) + (1 - beta) * gu
                md = beta * old.get("momentum_down_t", torch.zeros_like(gd)) + (1 - beta) * gd
                # Preserve canonical representation/EMA; promote BEFORE any dual
                # residual, SVD, or Newton work. There is no fp32 trial/retry.
                su, sd = u.to(config.dtype), d.to(config.dtype)
                objective = torch.stack((mu, md)).to(config.dtype)
                result = solve_coupled(su, sd, objective, config=config,
                                       initial_lambda=old.get("lambda"))
                metrics = dict(result.metrics, model_dtype=str(up.dtype),
                               momentum_dtype=str(momentum_dtype), canonical_dtype=str(momentum_dtype),
                               returned_update_dtype=str(up.dtype), pair_name=pair.name)
                diagnostics[pair.name] = metrics
                if not result.converged:
                    self.last_diagnostics = diagnostics
                    raise RuntimeError(f"{pair.name}: coupled solve not certified: {result.reason}; {metrics}")
                delta_up = (root[:, None] * result.pair[0]).to(up.dtype)
                delta_down = (result.pair[1] / root[:, None]).T.to(down.dtype)
                new_up, new_down = up - lr * delta_up, down - lr * delta_down
                if not all(bool(torch.isfinite(v).all()) for v in (delta_up, delta_down, new_up, new_down)):
                    raise RuntimeError(f"{pair.name}: raw update not representable in parameter dtype")
                if self.record_cast_diagnostics:
                    cast_pair = torch.stack((delta_up.double() / root[:, None],
                                             delta_down.T.double() * root[:, None]))
                    magnitude = float(result.pair.norm())
                    metrics["cast_direction_relative_error"] = (float((cast_pair - result.pair).norm()) / magnitude
                                                                 if magnitude else 0.)
                    metrics["cast_horizontal_residual"] = float(horizontal_residual(u.double(), d.double(), cast_pair).abs().max())
                    metrics["cast_spectral_excess"] = max(0., float(torch.linalg.svdvals(cast_pair).max()) - 1.)
                    metrics["cast_diagnostic_svd_evaluations"] = 2
                state = dict(momentum_up=mu, momentum_down_t=md,
                             **{"lambda": result.lam.clone()}, step=old.get("step", 0) + 1)
                pending.append((pair, new_up, new_down, state))
        # All pairs validate before any parameter or momentum is mutated.
        for pair, new_up, new_down, state in pending:
            pair.up.copy_(new_up)
            pair.down.copy_(new_down)
            self.state[pair.up] = state
        self.last_diagnostics = diagnostics if self.record_diagnostics else {}
        return loss

    def _topology(self):
        return [dict(name=p.name, up_shape=tuple(p.up.shape), down_shape=tuple(p.down.shape))
                for p in self.pairs]

    def state_dict(self):
        result = super().state_dict()
        result["qso_format_version"] = 2
        result["qso_pairs"] = self._topology()
        return result

    def load_state_dict(self, state_dict):
        saved = copy.deepcopy(state_dict)
        version = saved.pop("qso_format_version", None)
        if version not in (1, 2) or saved.pop("qso_pairs", None) != self._topology():
            raise ValueError("checkpoint named pair topology does not match")
        groups = saved["param_groups"]
        if len(groups) != len(self.pairs):
            raise ValueError("checkpoint pair groups do not match")
        restored = {}
        for pair, group in zip(self.pairs, groups):
            if group["pair_name"] != pair.name or len(group["params"]) != 2:
                raise ValueError("checkpoint pair identity/order mismatch")
            if version == 1:
                # Legacy checkpoints coupled EMA to solver dtype. Preserve their
                # explicit old policy; loading is not an implicit policy migration.
                dtype = group["solver"]["dtype"]
                group["momentum_dtype"] = dtype
                if group["solver"].get("tolerance") is None:
                    group["solver"]["tolerance"] = 1e-8 if dtype == torch.float64 else 3e-5
                if group["solver"].get("rcond_guard") is None:
                    group["solver"]["rcond_guard"] = 1e-8 if dtype == torch.float64 else 1e-4
            config = SolverConfig(**group["solver"])
            momentum_dtype = group["momentum_dtype"]
            if momentum_dtype not in (torch.float32, torch.float64):
                raise ValueError("invalid checkpoint momentum dtype")
            if saved["state"].get(group["params"][1]):
                raise ValueError("unexpected down-parameter state")
            source = saved["state"].get(group["params"][0], {})
            if source:
                target = {}
                for key, shape, dtype in (("momentum_up", pair.up.shape, momentum_dtype),
                                          ("momentum_down_t", pair.up.shape, momentum_dtype),
                                          ("lambda", (pair.up.shape[0],), torch.float64)):
                    value = source[key]
                    if value.shape != shape or value.dtype != dtype or not torch.isfinite(value).all():
                        raise ValueError(f"invalid checkpoint {key} shape/dtype/value")
                    target[key] = value.to(device=pair.up.device).clone()
                if not isinstance(source["step"], int) or source["step"] < 0:
                    raise ValueError("invalid checkpoint step")
                target["step"] = source["step"]
                restored[pair.up] = target
        # Default Optimizer loading casts state to parameter dtype, destroying
        # fp32 momentum/fp64 lambda for bf16 parameters. Restore explicitly.
        saved["state"] = {}
        super().load_state_dict(saved)
        self.state.update(restored)
        self.last_diagnostics = {}
