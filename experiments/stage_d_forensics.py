"""Research observers of the unchanged production solver; never a new solver.

The reference hook reads its calling frame before the reference executes and
raises ReferenceRequested. Hooks are scoped to one diagnostic solve. No tensor
used by production is modified, and no production source is copied or edited.
"""
from dataclasses import asdict
import inspect
import math
import sys
from unittest.mock import patch
import torch
import qnormuon.coupled_solver as cs


class ReferenceRequested(RuntimeError):
    pass


def stats(x):
    x = x.detach().double()
    return dict(min=float(x.min()), max=float(x.max()), norm=float(x.norm()),
                rms=float(x.square().mean().sqrt()))


def cpu(x):
    return x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x


class Observer:
    def __init__(self, u, d, a, initial_lambda, metadata, raw=None, detailed=True):
        self.inputs = dict(u=cpu(u), d=cpu(d), a=cpu(a), initial_lambda=cpu(initial_lambda))
        self.metadata, self.raw, self.detailed = metadata, raw or {}, detailed
        self.evaluations, self.certificates, self.newton, self.coordinates = [], [], [], []
        self.normalization, self.event = {}, None

    def evaluate(self, original, problem, z):
        ev = original(problem, z)
        frame = inspect.currentframe().f_back
        # The wrapper lambda adds a frame; find the live production solve.
        while frame and frame.f_code is not cs._solve.__wrapped__.__code__:
            frame = frame.f_back
        loc = frame.f_locals if frame else {}
        if not self.normalization and 'magnitude' in loc:
            self.normalization = dict(intrinsic_magnitude=loc['magnitude'],
                objective_fro=cs._stable_norm(loc['ad']), beta_stats=stats(loc['beta']),
                whitening_stats=stats(loc['coord']), row_metric_stats=stats(loc['w']))
        entry = dict(index=len(self.evaluations), iteration=loc.get('iteration', 0),
                     value=ev.value, gradient_norm=float(ev.gradient.norm()), rcond=ev.rcond,
                     singular_values=ev.singular.detach().cpu().tolist(),
                     original_singular_scale=loc.get('magnitude', 1.))
        if 'trial_z' in loc and z is loc['trial_z']:
            current = loc['ev']; step = loc['step']; slope = loc['slope']
            rounding = 2 * loc['eps'] * max(1., abs(current.value))
            rhs = current.value + 1e-4 * step * slope + rounding
            resolved = abs(step * slope) > rounding
            improves = float(ev.gradient.norm()) < float(current.gradient.norm())
            entry.update(kind='line_trial', step_length=step, slope=slope,
                armijo_rhs=rhs, rounding_allowance=rounding, resolved_decrease=resolved,
                gradient_improves=improves, rcond_pass=ev.rcond > loc['guard'],
                value_pass=ev.value <= rhs,
                accepted=(ev.rcond > loc['guard'] and ev.value <= rhs and (resolved or improves)))
        else:
            entry['kind'] = 'initial'
        self.evaluations.append(entry)
        if self.detailed: self.coordinates.append(cpu(z))
        return ev

    def certificate(self, original, *args, **kwargs):
        p, metrics = original(*args, **kwargs)
        self.certificates.append(dict(evaluation_index=len(self.evaluations)-1, **metrics))
        return p, metrics

    def direction(self, original, problem, ev, scale, max_cg):
        rows = []
        code = original.__code__
        lines = inspect.getsourcelines(original)[0]
        start = original.__code__.co_firstlineno
        curvature_line = next(start+i for i,s in enumerate(lines) if 'if curvature <=' in s)
        residual_line = next(start+i for i,s in enumerate(lines) if 'if math.sqrt(new_squared)' in s)
        previous = sys.gettrace()
        def trace(frame, event, arg):
            if frame.f_code is not code: return None
            if event == 'line' and frame.f_lineno in (curvature_line, residual_line):
                loc = frame.f_locals
                row = dict(event='curvature' if frame.f_lineno == curvature_line else 'residual',
                           iteration=loc['_'], target=loc['target'], damping=loc['damping'])
                if row['event'] == 'curvature':
                    row.update(curvature=loc['curvature'], squared_residual=loc['squared'])
                else:
                    row.update(residual_norm=math.sqrt(loc['new_squared']))
                rows.append(row)
            return trace
        try:
            sys.settrace(trace)
            p = original(problem, ev, scale, max_cg)
        finally:
            sys.settrace(previous)
        self.newton.append(dict(evaluation_index=len(self.evaluations)-1,
            gradient_norm=float(ev.gradient.norm()), direction_norm=float(p.norm()),
            slope=float(ev.gradient @ p), damping=1e-6*scale, cg=rows))
        return p

    def reference(self, *args):
        frame = inspect.currentframe().f_back
        loc = frame.f_locals
        if 'reason' in loc and 'ev' in loc:
            ev = loc['ev']
            current_lambda = loc['beta'] + loc['magnitude'] * loc['coord'] * loc['z'].double()
            current = loc['history'][-1]
            self.event = dict(event='smooth_exit_before_reference', **self.metadata,
                reason=loc['reason'], newton_iteration=loc['iteration'],
                current_normalized_gap=current['normalized_gap'],
                best_normalized_gap=loc['best'][2]['normalized_gap'],
                current_rcond=ev.rcond, minimum_smooth_rcond=loc['min_rcond'],
                minimum_evaluated_rcond=min(r['rcond'] for r in self.evaluations),
                gradient_norm=float(ev.gradient.norm()), objective_value=ev.value,
                **self.normalization, cg_count=loc['counts'].hvp,
                line_search_count=loc['counts'].line_trials, counts=asdict(loc['counts']),
                history=loc['history'], current_lambda=current_lambda.detach().cpu().tolist(),
                initial_lambda=(self.inputs['initial_lambda'].tolist()
                    if self.inputs['initial_lambda'] is not None else None))
            self.inputs.update(current_lambda=cpu(current_lambda), beta=cpu(loc['beta']),
                coord=cpu(loc['coord']), magnitude=loc['magnitude'],
                internal_coordinates=self.coordinates)
        else:
            self.event = dict(event='smooth_exit_before_reference', **self.metadata,
                              reason='svd_failure: '+str(loc.get('error', 'unknown')),
                              history=[], **self.normalization)
        # Always stop before importing/transferring to the CPU reference.
        raise ReferenceRequested(self.event['reason'])

    def solve(self, config):
        oe, oc, on = cs.SmoothDual.evaluate, cs.certificate, cs._newton_direction
        with patch.object(cs.SmoothDual, 'evaluate', lambda problem,z:self.evaluate(oe,problem,z)), \
             patch.object(cs, 'certificate', lambda *a,**k:self.certificate(oc,*a,**k)), \
             patch.object(cs, '_newton_direction', lambda p,e,s,n:self.direction(on,p,e,s,n)), \
             patch.object(cs, '_reference', self.reference):
            device = config.pop('_device', None)
            u,d,a=(self.inputs[k].to(device) for k in ('u','d','a'))
            lam=self.inputs['initial_lambda']
            if lam is not None:lam=lam.to(device)
            return cs.solve_coupled(u,d,a,config=cs.SolverConfig(**config),initial_lambda=lam)

    def fixture(self):
        return dict(**self.inputs, metadata=self.metadata, raw=self.raw, event=self.event,
                    evaluations=self.evaluations, certificates=self.certificates, newton=self.newton)
