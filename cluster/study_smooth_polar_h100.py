"""Research-only H100 polar study; production SmoothDual remains full SVD."""
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM allocation required")

import sitecustomize
import torch
import qnormuon.coupled_solver as solver
from cluster.study_smooth_gram_h100 import make_cases, perturbations, real_inputs
from experiments.smooth_polar_alternative import diagnose, polar_derivative


def timed(fn, repeats=7):
    for _ in range(2):
        fn()
    samples=[]
    for _ in range(repeats):
        torch.cuda.synchronize(); start=time.perf_counter()
        fn(); torch.cuda.synchronize()
        samples.append(time.perf_counter()-start)
    return dict(median=statistics.median(samples),
                p95=sorted(samples)[math.ceil(.95*len(samples))-1])


def compare(name,b,route):
    result=diagnose(b,route=route)
    amp=float(b.abs().max())
    x=b/amp
    u,s,vt=torch.linalg.svd(x,full_matrices=False)
    p=u@vt
    row=dict(name=name,route=route,shape=list(b.shape),accepted=result.accepted,
             reason=result.reason,iterations=result.iterations,
             qr_iterations=result.qr_iterations,
             cholesky_iterations=result.cholesky_iterations,
             posterior=result.measurements,svd_rcond=float(s[-1]/s[0]))
    if not result.accepted:
        return row
    gs=result.singular/amp
    row.update(polar_relative=float((result.polar-p).norm()/p.norm()),
        singular_max_relative=float(((gs-s).abs()/s).max()),
        nuclear_relative=abs(float(gs.sum()-s.sum()))/float(s.sum()),
        rcond_abs=abs(result.measurements["rcond"]-float(s[-1]/s[0])),
        unsafe_guard=bool(result.measurements["rcond"]>1e-4 and float(s[-1]/s[0])<=1e-4))
    deriv=solver.SmoothDual(None,None,None)
    ref=SimpleNamespace(left=u,singular=s,right=vt)
    errors={}
    for kind,e in perturbations(x,(u,s,vt)).items():
        got=polar_derivative(result,e*amp)
        expected=deriv.polar_derivative(ref,e)
        absolute=float((got-expected).norm())
        errors[kind]=dict(absolute=absolute,
            relative=absolute/max(float(expected.norm()),torch.finfo(torch.float64).tiny))
        if b.shape[0]<=96 and kind in ("random","normal","smallest"):
            step=1e-6*max(float(s[-1]),1e-8)/max(float(e.norm()),1.)
            plus=torch.linalg.svd(x+step*e,full_matrices=False)
            minus=torch.linalg.svd(x-step*e,full_matrices=False)
            fd=((plus[0]@plus[2])-(minus[0]@minus[2]))/(2*step)
            errors[kind]["fd_oracle_relative"]=float((fd-expected).norm())/max(float(expected.norm()),1e-300)
    row["derivative"]=errors
    return row


def replay(rows,route):
    events=[];comparisons=[]
    parent=solver.SmoothDual
    class Alternative(parent):
        def evaluate(self,lam):
            b=self.a-solver.adjoint(self.u,self.d,lam)
            parts=[diagnose(side,route=route) for side in b]
            if not all(z.accepted for z in parts):
                events.append(dict(backend="svd_fallback",reasons=[z.reason for z in parts]))
                return super().evaluate(lam)
            p=torch.stack([z.polar for z in parts])
            s=torch.stack([z.singular for z in parts])
            ev=solver.Evaluation(float(s.sum()),-solver.horizontal_residual(self.u,self.d,p),
                p,None,s,None,float((s[:,-1]/s[:,0]).min()),lam)
            self._parts=parts;self._ev=ev
            self.counts.polar_matrices+=2
            events.append(dict(backend=route,reasons=[]))
            return ev
        def polar_derivative(self,ev,e):
            if ev is getattr(self,"_ev",None):
                return torch.stack([polar_derivative(z,v) for z,v in zip(self._parts,e)])
            return super().polar_derivative(ev,e)
    for row in rows:
        config=solver.SolverConfig(fallback=False)
        x=solver.solve_coupled(row["u"],row["d"],row["a"],config=config,
                               initial_lambda=row["initial_lambda"])
        with patch.object(solver,"SmoothDual",Alternative):
            y=solver.solve_coupled(row["u"],row["d"],row["a"],config=config,
                                   initial_lambda=row["initial_lambda"])
        comparisons.append(dict(pair=row["name"],converged=[x.converged,y.converged],
            reasons=[x.reason,y.reason],newton=[x.iterations,y.iterations],
            line_trials=[x.counts.line_trials,y.counts.line_trials],
            certificate_decisions_equal=[solver.accepted(h,3e-5) for h in x.history]
                ==[solver.accepted(h,3e-5) for h in y.history],
            lambda_max_abs=float((x.lam-y.lam).abs().max()),
            direction_relative=float((x.pair-y.pair).norm()/x.pair.norm()),
            gap_abs=abs(x.metrics["normalized_gap"]-y.metrics["normalized_gap"])))
    return dict(comparisons=comparisons,events=events)


def driver_audit(b):
    results={}
    ref=None
    for driver in (None,"gesvd","gesvdj"):
        name="default" if driver is None else driver
        try:
            fn=lambda:torch.linalg.svd(b,full_matrices=False,driver=driver)
            stats=timed(fn)
            u,s,vt=fn()
            if ref is None:
                ref=s,u@vt
            results[name]=dict(**stats,
                singular_max_abs=float((s-ref[0]).abs().max()),
                polar_relative=float(((u@vt)-ref[1]).norm()/ref[1].norm()))
        except Exception as exc:
            results[name]=dict(error=str(exc))
    return results


def main():
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert "H100" in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for name,b in make_cases():
        for route in ("tall","qr_square"):
            rows.append(compare(name,b,route))
    real,residuals,provenance=real_inputs()
    for i,b in enumerate(residuals):
        for j,side in enumerate(b):
            for route in ("tall","qr_square"):
                rows.append(compare(f"real_step25_eval{i}_side{j}",side,route))
    decision={route:replay(real,route) for route in ("tall","qr_square")}
    sample=residuals[-1]
    perf={"full_svd":timed(lambda:torch.linalg.svd(sample,full_matrices=False))}
    for route in ("tall","qr_square"):
        perf[route]=timed(lambda:[diagnose(side,route=route) for side in sample])
    drivers=driver_audit(sample[0])
    target=Path("cluster")/f"smooth_polar_study-{os.environ['SLURM_JOB_ID']}.json"
    target.write_text(json.dumps(dict(job=os.environ["SLURM_JOB_ID"],cases=rows,
        provenance=provenance,decision=decision,performance=perf,drivers=drivers),
        indent=2,allow_nan=False)+"\n")
    summary={route:dict(accepted=sum(x["accepted"] for x in rows if x["route"]==route),
        count=sum(x["route"]==route for x in rows),
        max_polar=max((x.get("polar_relative",0) for x in rows if x["route"]==route),default=None),
        max_hvp=max((v["relative"] for x in rows if x["route"]==route
                     for v in x.get("derivative",{}).values()),default=None),
        decision_mismatches=sum(a["newton"][0]!=a["newton"][1] or
            a["line_trials"][0]!=a["line_trials"][1]
            for a in decision[route]["comparisons"])) for route in ("tall","qr_square")}
    print("STUDY_RESULT",json.dumps(dict(path=str(target),summary=summary,
          performance=perf,drivers=drivers)),flush=True)


if __name__=="__main__":
    main()
