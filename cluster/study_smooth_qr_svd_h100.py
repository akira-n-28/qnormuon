"""Focused exact-decomposition screen. No alternative optimizer semantics."""
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM allocation required")

import sitecustomize
import torch
import qnormuon.coupled_solver as solver
from experiments.smooth_qr_svd import decompose, research_backend
# Reuse only the deterministic matrix generator, perturbations and local
# checkpoint reader. No Gram/QDWH routines are called by this study.
from cluster.study_smooth_gram_h100 import make_cases, perturbations, real_inputs


def timed(fn, repeats=21):
    for _ in range(3):
        fn()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return dict(median=statistics.median(samples),
                p95=sorted(samples)[math.ceil(.95 * len(samples))-1], samples=samples)


def compare(name, b):
    factors = decompose(b)
    ref = torch.linalg.svd(b, full_matrices=False)
    u, s, vt = factors
    ru, rs, rvt = ref
    amplitude = float(b.abs().max())
    x = b / amplitude
    p, rp = u @ vt, ru @ rvt
    record = dict(name=name, shape=list(b.shape),
        reconstruction=float(((u * (s/amplitude)) @ vt - x).norm()/x.norm()),
        oracle_reconstruction=float(((ru * (rs/amplitude)) @ rvt - x).norm()/x.norm()),
        left_orthogonality=float((u.T @ u - torch.eye(u.shape[1],device=b.device,dtype=b.dtype)).norm()),
        right_orthogonality=float((vt @ vt.T-torch.eye(s.numel(),device=b.device,dtype=b.dtype)).norm()),
        polar_relative=float((p-rp).norm()/rp.norm()),
        singular_max_relative=float(((s-rs).abs()/rs).max()),
        nuclear_relative=abs(float(s.sum()/rs.sum())-1.),
        rcond_abs=abs(float(s[-1]/s[0]-rs[-1]/rs[0])),
        oracle_rcond=float(rs[-1]/rs[0]),
        candidate_rcond=float(s[-1]/s[0]))
    problem = solver.SmoothDual(None,None,None)
    ce = SimpleNamespace(left=u,singular=s/amplitude,right=vt)
    re = SimpleNamespace(left=ru,singular=rs/amplitude,right=rvt)
    directions = {}
    perturb = perturbations(x,(ru,rs/amplitude,rvt))
    normal = perturb["normal"][:,0]
    normal = normal / normal.norm()
    perturb["smallest_normal"] = torch.outer(normal,rvt[-1])
    perturb["clustered_skew"] = torch.outer(ru[:,0],rvt[1])-torch.outer(ru[:,1],rvt[0])
    for label,e in perturb.items():
        dy = problem.polar_derivative(ce,e)
        refdy = problem.polar_derivative(re,e)
        absolute = float((dy-refdy).norm())
        item = dict(absolute=absolute,oracle_norm=float(refdy.norm()),
                    relative=absolute/max(float(refdy.norm()),1e-300))
        # u_i v_i.T and symmetric in-subspace perturbations have zero exact
        # polar derivative. Their roundoff-only denominator is not evidence of
        # poor accuracy on an informative derivative. Retain absolute errors.
        item["informative"] = label not in ("smallest","largest","clustered")
        if b.shape[0]<=96 and label in ("random","normal","tangent"):
            step = 1e-4 * float((rs/amplitude)[-1])/max(float(e.norm()),1.)
            plus = torch.linalg.svd(x+step*e,full_matrices=False)
            minus = torch.linalg.svd(x-step*e,full_matrices=False)
            fd = (plus[0]@plus[2]-minus[0]@minus[2])/(2*step)
            item["finite_difference_relative"] = float((dy-fd).norm())/max(float(refdy.norm()),1e-300)
        directions[label] = item
    record["derivative"] = directions
    return record


def replay(rows):
    results = []
    for row in rows:
        config=solver.SolverConfig(fallback=False)
        traces=[];answers=[];timings=[]
        for backend in ("tall_svd","qr_reduced_svd"):
            trace=[]
            torch.cuda.synchronize();start=time.perf_counter()
            with research_backend(backend,trace=trace):
                result=solver.solve_coupled(row["u"],row["d"],row["a"],
                    config=config,initial_lambda=row["initial_lambda"])
            torch.cuda.synchronize();timings.append(time.perf_counter()-start)
            traces.append(trace);answers.append(result)
        x,y=answers
        # Initial coordinates are exactly the same; derivative/HVP/Newton are
        # the original implementation supplied with either factor set.
        beta=solver.multipliers(row["u"],row["d"],row["a"])
        magnitude=solver._stable_norm(row["a"]-solver.adjoint(row["u"],row["d"],beta))
        coord=(row["u"].square()+row["d"].square()).sum(1).rsqrt()
        problem=solver.SmoothDual(row["u"]*coord[:,None],row["d"]*coord[:,None],row["a"]/magnitude)
        v=torch.linspace(-1.,1.,row["u"].shape[0],device=row["u"].device,dtype=torch.float64)
        hx,hy=(problem.hvp(t[0],v) for t in traces)
        dx,dy=(solver._newton_direction(problem,t[0],1./float(t[0].singular.min())) for t in traces)
        results.append(dict(pair=row["name"],converged=[x.converged,y.converged],
            reasons=[x.reason,y.reason],newton=[x.iterations,y.iterations],
            cg=[x.counts.hvp,y.counts.hvp],line_trials=[x.counts.line_trials,y.counts.line_trials],
            evaluations=[len(t) for t in traces],seconds=timings,
            initial_hvp_relative=float((hx-hy).norm()/hx.norm()),
            initial_direction_relative=float((dx-dy).norm()/dx.norm()),
            trace_value_max_abs=max(abs(a.value-b.value) for a,b in zip(*traces)),
            trace_gradient_max_abs=max(float((a.gradient-b.gradient).abs().max()) for a,b in zip(*traces)),
            trace_coordinate_max_abs=max(float((a.coordinate-b.coordinate).abs().max()) for a,b in zip(*traces)),
            certificate_decisions=[[solver.accepted(h,3e-5) for h in r.history] for r in answers],
            lambda_max_abs=float((x.lam-y.lam).abs().max()),
            direction_relative=float((x.pair-y.pair).norm()/x.pair.norm()),
            gap_abs=abs(x.metrics["normalized_gap"]-y.metrics["normalized_gap"])))
    return results


def kernels(sample, real_batches):
    qr,r=torch.linalg.qr(sample,mode="reduced")
    ur,s,vt=torch.linalg.svd(r,full_matrices=False)
    result={"tall_svd":timed(lambda:decompose(sample,"tall_svd")),
            "qr":timed(lambda:torch.linalg.qr(sample,mode="reduced")),
            "square_svd":timed(lambda:torch.linalg.svd(r,full_matrices=False)),
            "reconstruction":timed(lambda:qr@ur),
            "qr_reduced_svd":timed(lambda:decompose(sample))}
    drivers={}
    for driver in (None,"gesvdj","gesvd"):
        name=driver or "default"
        try:
            u,ds,dv=decompose(sample,driver=driver)
            drivers[name]=dict(square=timed(lambda:torch.linalg.svd(r,full_matrices=False,driver=driver)),
                total=timed(lambda:decompose(sample,driver=driver)),
                spectrum_relative=float(((ds-s).abs()/s).max()))
        except Exception as error:
            drivers[name]=dict(error=str(error))
    many=torch.cat(real_batches,dim=0)
    batching={}
    for backend in ("tall_svd","qr_reduced_svd"):
        batching[backend]=dict(
            sequential=timed(lambda:[decompose(many[i:i+2],backend) for i in range(0,12,2)],repeats=11),
            batched=timed(lambda:decompose(many,backend),repeats=11))
    # Repeat in reversed order to expose order/drift effects in a sub-percent
    # apparent difference. No optimization choice is made from a cold timing.
    confirmation={backend:timed(lambda:decompose(sample,backend))
                  for backend in ("qr_reduced_svd","tall_svd")}
    return dict(components=result,drivers=drivers,batching=batching,confirmation=confirmation)


def main():
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert "H100" in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32=False
    print("PROVENANCE",json.dumps(dict(python=sys.executable,version=sys.version,
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        seed=60831,solver=str(solver.SolverConfig(fallback=False)))),flush=True)
    cases=make_cases()
    # Include the exact additional requested guard-side point in every shape.
    for m,n in ((24,8),(96,24),(1024,384)):
        b=next(b for name,b in cases if name==f"{m}x{n}/rcond=0.00012")
        u,s,vt=torch.linalg.svd(b,full_matrices=False)
        s[-1]=1.1e-4*s[0]
        cases.append((f"{m}x{n}/rcond=0.00011",(u*s)@vt))
    production=next(b for name,b in cases if name=="1024x384/rcond=0.000101")
    for scale in (1e-150,1e150):
        cases.append((f"1024x384/near_guard_scale={scale:g}",production*scale))
    rows=[compare(name,b) for name,b in cases]
    real,residuals,provenance=real_inputs()
    for i,b in enumerate(residuals):
        for side in range(2):
            rows.append(compare(f"real_step25_eval{i}_side{side}",b[side]))
    decisions=replay(real)
    assert len(residuals)==18  # six independently solved pairs, three evaluations each
    perf=kernels(residuals[-1],[residuals[3*i+2] for i in range(6)])
    target=Path("cluster")/f"smooth_qr_svd_study-{os.environ['SLURM_JOB_ID']}.json"
    output=dict(job=os.environ["SLURM_JOB_ID"],cases=rows,replay=decisions,
                performance=perf,provenance=provenance)
    target.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print("RESULT",json.dumps(dict(path=str(target),cases=len(rows),
        max_polar=max(r["polar_relative"] for r in rows),
        max_singular_relative=max(r["singular_max_relative"] for r in rows),
        max_informative_derivative=max(v["relative"] for r in rows for v in r["derivative"].values() if v["informative"]),
        replay=decisions,performance=perf)),flush=True)


if __name__=="__main__":
    main()
