"""Research-only SVD-controlled QDWH decision replay on one SLURM H100."""
import collections
import json
import math
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM allocation required")

import sitecustomize
import torch
import qnormuon.coupled_solver as solver
import qnormuon.optimizer as optimizer_module
from benchmarks.tiny_transformer import (ModelConfig, Optimizers, TinyTransformer,
    datasets, load_checkpoint, seed_everything, train_step)
from experiments.smooth_polar_alternative import diagnose
from experiments.smooth_qdwh_decisions import (
    armijo_interval, certificate_interval_decision, cg_stop_certified,
    factor_posterior, gamma, hvp_with_error, interval_sign,
    row_operator_upper)


def sync():
    torch.cuda.synchronize()


class Shadow:
    def __init__(self):
        self.records=[];self.by_ev={};self.certificate_records=[]
        self.hvp_records=[];self.cg_records=[];self.cert_records=[]
        self.times=collections.defaultdict(float)
        self.original_eval=solver.SmoothDual.evaluate
        self.original_hvp=solver.SmoothDual.hvp
        self.original_newton=solver._newton_direction
        self.original_cert=solver.certificate

    def evaluate(self,problem,z):
        sync();t=time.perf_counter()
        ev=self.original_eval(problem,z)
        sync();self.times["svd_controlled_evaluation"]+=time.perf_counter()-t
        b=problem.a-solver.adjoint(problem.u,problem.d,z)
        sync();t=time.perf_counter()
        factors=[diagnose(side) for side in b]
        sync();self.times["qdwh_kernel"]+=time.perf_counter()-t
        t=time.perf_counter()
        posts=[factor_posterior(side,r) for side,r in zip(b,factors)]
        sync();self.times["factor_posterior"]+=time.perf_counter()-t
        ok=all(p.accepted for p in posts)
        r=dict(ev=ev,b=b,factors=factors,posts=posts,ok=ok,
               reason=next((p.reason for p in posts if not p.accepted),"qdwh_posterior_available"))
        if ok:
            pair=torch.stack([f.polar for f in factors])
            r["pair"]=pair
            r["singular"]=torch.stack([f.singular for f in factors])
            r["gradient"]=-solver.horizontal_residual(problem.u,problem.d,pair)
            r["grad_error"]=row_operator_upper(problem.u,problem.d)*math.hypot(*(p.polar_error for p in posts))
            value_round=gamma(2*problem.u.shape[1]+1)*sum(p.nuclear_upper for p in posts)
            r["value_interval"]=(sum(p.nuclear_lower for p in posts)-value_round,
                                 sum(p.nuclear_upper for p in posts)+value_round)
            r["value_contained"]=r["value_interval"][0]-1e-12<=ev.value<=r["value_interval"][1]+1e-12
            r["gradient_contained"]=float((r["gradient"]-ev.gradient).norm())<=r["grad_error"]+1e-12
            r["polar_observed"]=float((pair-ev.pair).norm())
            r["polar_bound"]=math.hypot(*(p.polar_error for p in posts))
        self.records.append(r);self.by_ev[id(ev)]=r
        return ev

    def hvp_bound(self,problem,r,v):
        if not r["ok"]:return None,math.inf
        e=-solver.adjoint(problem.u,problem.d,v)
        sync();t=time.perf_counter()
        h,bound=hvp_with_error(r["b"],e,r["factors"],r["posts"],problem.u,problem.d)
        sync();self.times["derivative_posterior"]+=time.perf_counter()-t
        return h,bound

    def hvp(self,problem,ev,v):
        oracle=self.original_hvp(problem,ev,v)
        r=self.by_ev[id(ev)]
        approx,error=self.hvp_bound(problem,r,v)
        if approx is None:
            self.hvp_records.append(dict(contained=None,curvature="uncertain",reason=r["reason"]))
        else:
            observed=float((approx-oracle).norm())
            c=float(v@approx)
            self.hvp_records.append(dict(contained=observed<=error+1e-12,
                curvature=interval_sign(c,error*float(v.norm())),
                oracle_curvature=float(v@oracle),observed=observed,bound=error))
        return oracle

    def newton(self,problem,ev,curvature_scale,max_cg=30):
        oracle=self.original_newton(problem,ev,curvature_scale,max_cg)
        r=self.by_ev[id(ev)]
        if not r["ok"]:
            r["cg_reason"]=r["reason"]
            self.cg_records.append(dict(reason=r["reason"],direction_difference=None))
            return oracle
        sync();t=time.perf_counter()
        g=r["gradient"]; ge=r["grad_error"]; damping=1e-6*curvature_scale
        x=torch.zeros_like(g);res=-g.clone();p=res.clone();rr=float(res@res)
        target=min(.1,math.sqrt(max(float(g.norm()),1e-16)))*math.sqrt(rr)
        reason="qdwh_cg_uncertain";iterations=0;direction_bound=math.inf
        for _ in range(min(max_cg,g.numel())):
            h,he=self.hvp_bound(problem,r,p)
            if h is None or not math.isfinite(he):
                reason="qdwh_derivative_uncertain";break
            h=h+damping*p
            curvature=float(p@h)
            if interval_sign(curvature,he*float(p.norm()))!="positive":
                reason="qdwh_curvature_ambiguous";break
            alpha=rr/curvature
            x=x+alpha*p;res=res-alpha*h;iterations+=1
            next_rr=float(res@res)
            if math.sqrt(next_rr)<=target:
                reason="candidate_cg_stop";break
            p=res+(next_rr/rr)*p;rr=next_rr
        if reason=="candidate_cg_stop":
            hx,he=self.hvp_bound(problem,r,x)
            residual_upper=float((g+hx+damping*x).norm())+ge+he
            direction_bound=residual_upper/damping
            g_lower=max(0.,float(g.norm())-ge)
            if not cg_stop_certified(residual_upper,g_lower,damping=damping):
                reason="qdwh_cg_uncertain"
            elif interval_sign(float(g@x),ge*float(x.norm()))!="negative":
                reason="qdwh_descent_ambiguous"
            else:
                reason="qdwh_cg_descent_certified"
        sync();self.times["cg_decision_including_derivative"]+=time.perf_counter()-t
        r["cg_reason"]=reason
        self.cg_records.append(dict(reason=reason,iterations=iterations,
             direction_difference=float((x-oracle).norm()),direction_bound=direction_bound,
             direction_relative=float((x-oracle).norm())/max(float(oracle.norm()),1e-300)))
        return oracle

    def certificate(self,u,d,a,lam,candidate,counts,**kwargs):
        r=next((rec for rec in reversed(self.records) if rec["ev"].pair is candidate),None)
        if r is not None:self.certificate_records.append(r)
        oracle=self.original_cert(u,d,a,lam,candidate,counts,**kwargs)
        oracle_accept=(r is not None and r["ev"].rcond>1e-4 and solver.accepted(oracle[1],3e-5))
        if r is None or not r["ok"] or kwargs.get("cached_dual") is None:
            reason=r["reason"] if r is not None else "qdwh_certificate_ambiguous"
            if r is not None:r["certificate_reason"]=reason
            self.cert_records.append(dict(reason=reason,oracle_accept=oracle_accept,safe=True))
            return oracle
        sync();t=time.perf_counter()
        cache=solver.CachedDualSpectrum(lam,r["pair"],r["singular"],kwargs["cached_dual"].magnitude)
        recovered,metrics=self.original_cert(u,d,a,lam,r["pair"],solver.Counts(),
                             cached_dual=cache,primal_norm_backend="gram_upper")
        magnitude=cache.magnitude
        dl=magnitude*r["value_interval"][0]
        du=magnitude*r["value_interval"][1]
        primal=metrics["primal_objective"]
        pround=gamma(2*recovered.numel()+1)*float((a*recovered).abs().sum())
        pl=primal-pround;pu=primal+pround
        denom_lower=max(abs(pl),abs(dl),torch.finfo(torch.float64).tiny)
        denom_upper=max(abs(pu),abs(du),torch.finfo(torch.float64).tiny)
        gap_upper=max(0.,du-pl)/denom_lower
        signed_lower=(dl-pu)/(denom_lower if dl-pu<0 else denom_upper)
        rlo=min(p.rcond_lower for p in r["posts"])
        feasible=(metrics["normalized_horizontal_residual"]<=1e-10 and
                  metrics["spectral_excess"]<=1e-12 and rlo>1e-4)
        reason=certificate_interval_decision((dl,du),(pl,pu),
            rcond_lower=rlo,feasible=feasible)
        # A QDWH candidate's failed gap does not prove that the distinct
        # SVD-derived candidate would fail. In particular, gram_upper's
        # conservative radial scale is not an exact 1-Lipschitz norm map.
        # Any such rejection must be recomputed by full SVD.
        safe=(reason=="qdwh_certificate_ambiguous" or (reason=="accept")==oracle_accept)
        r["certificate_reason"]=reason
        self.cert_records.append(dict(reason=reason,oracle_accept=oracle_accept,safe=safe,
            gap_upper=gap_upper,gap_oracle=oracle[1]["normalized_gap"],
            rcond_lower=rlo))
        sync();self.times["certificate_posterior"]+=time.perf_counter()-t
        return oracle

    def armijo(self):
        outcomes=[]
        positions=[self.records.index(r) for r in self.certificate_records]
        for j,idx in enumerate(positions):
            end=positions[j+1] if j+1<len(positions) else len(self.records)
            trials=self.records[idx+1:end+1] if j+1<len(positions) else self.records[idx+1:]
            if not trials:continue
            current=self.records[idx]
            direction=trials[0]["ev"].coordinate-current["ev"].coordinate
            for k,trial in enumerate(trials):
                is_accepted=j+1<len(positions) and k==len(trials)-1
                if not current["ok"] or not trial["ok"]:
                    reason=trial["reason"] if not trial["ok"] else current["reason"]
                else:
                    alpha=.5**k
                    slope=float(current["gradient"]@direction)
                    radius=current["grad_error"]*float(direction.norm())
                    slope_interval=(slope-radius,slope+radius)
                    cl,cu=current["value_interval"]
                    round_low=2*torch.finfo(torch.float64).eps*max(1.,0. if cl<=0<=cu else min(abs(cl),abs(cu)))
                    round_hi=2*torch.finfo(torch.float64).eps*max(1.,abs(cl),abs(cu))
                    arm=armijo_interval(current["value_interval"],trial["value_interval"],
                        slope_interval,alpha=alpha,rounding=(round_low,round_hi))
                    rlo=min(p.rcond_lower for p in trial["posts"])
                    rhi=min(p.rcond_upper for p in trial["posts"])
                    resolved=(slope_interval[1]<0 and
                              alpha*min(abs(slope_interval[0]),abs(slope_interval[1]))>round_hi)
                    improvement=(float(trial["gradient"].norm())+trial["grad_error"]<
                                 max(0.,float(current["gradient"].norm())-current["grad_error"]))
                    if rhi<=1e-4 or arm=="reject":reason="reject"
                    elif arm=="accept" and rlo>1e-4 and (resolved or improvement):reason="accept"
                    else:reason="qdwh_armijo_ambiguous"
                trial["armijo_reason"]=reason
                outcomes.append(dict(reason=reason,safe=(reason not in ("accept","reject") or
                                                  (reason=="accept")==is_accepted),
                                     oracle_accept=is_accepted))
        return outcomes

    def summary(self):
        arms=self.armijo();hist=collections.Counter();fallback=0
        for r in self.records:
            reasons=[]
            if not r["ok"]:reasons.append(r["reason"])
            if r.get("certificate_reason") not in ("accept","reject",None):
                reasons.append(r["certificate_reason"])
            if r.get("cg_reason") not in (None,"qdwh_cg_descent_certified"):
                reasons.append(r["cg_reason"])
            if r.get("armijo_reason") not in (None,"accept","reject"):
                reasons.append(r["armijo_reason"])
            if reasons:fallback+=1;hist.update(set(reasons))
        def reasons(rows):return dict(collections.Counter(r["reason"] for r in rows))
        return dict(evaluations=len(self.records),factor_pass=sum(r["ok"] for r in self.records),
            value_violations=sum(r.get("value_contained") is False for r in self.records),
            gradient_violations=sum(r.get("gradient_contained") is False for r in self.records),
            polar_bound_violations=sum(r.get("polar_observed",0)>r.get("polar_bound",math.inf)+1e-12 for r in self.records),
            hvps=len(self.hvp_records),hvp_violations=sum(r.get("contained") is False for r in self.hvp_records),
            curvature_certified=sum(r["curvature"]!="uncertain" for r in self.hvp_records),
            curvature_unsafe=sum(r["curvature"]!="uncertain" and
                (r["curvature"]=="positive")!=(r["oracle_curvature"]>0)
                for r in self.hvp_records if "oracle_curvature" in r),
            cg=len(self.cg_records),cg_certified=sum(r["reason"]=="qdwh_cg_descent_certified" for r in self.cg_records),
            cg_reasons=reasons(self.cg_records),
            max_direction_difference=max((r.get("direction_relative",0) or 0 for r in self.cg_records),default=0),
            max_direction_bound=max((r.get("direction_bound",0) or 0 for r in self.cg_records if math.isfinite(r.get("direction_bound",0) or 0)),default=0),
            armijo=len(arms),armijo_certified=sum(r["reason"] in ("accept","reject") for r in arms),
            armijo_unsafe=sum(not r["safe"] for r in arms),armijo_reasons=reasons(arms),
            certificates=len(self.cert_records),certificate_certified=sum(r["reason"] in ("accept","reject") for r in self.cert_records),
            certificate_unsafe=sum(not r["safe"] for r in self.cert_records),certificate_reasons=reasons(self.cert_records),
            svd_fallbacks=fallback,qdwh_controlled=len(self.records)-fallback,
            fallback_reasons=dict(hist),times=dict(self.times),
            min_rcond_lower=min((min(p.rcond_lower for p in r["posts"]) for r in self.records if r["ok"]),default=None))


def observe_solve(u,d,a,**kwargs):
    shadow=Shadow()
    with patch.object(solver.SmoothDual,"evaluate",lambda self,z:shadow.evaluate(self,z)),\
         patch.object(solver.SmoothDual,"hvp",lambda self,ev,v:shadow.hvp(self,ev,v)),\
         patch.object(solver,"_newton_direction",lambda problem,ev,scale,max_cg=30:
             shadow.newton(problem,ev,scale,max_cg)),\
         patch.object(solver,"certificate",shadow.certificate):
        result=solver.solve_coupled(u,d,a,**kwargs)
    return result,shadow.summary()


def real_replay():
    config=json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
    source=Path(config["output_root"])/f"smoke-28936-qso-seed{config['seed']}"
    baseline=[json.loads(line) for line in (source/"metrics.jsonl").read_text().splitlines()]
    assert len(baseline)==50
    seed_everything(config["seed"],config["tf32"])
    train,val=datasets(config)
    metadata=dict(train=train.metadata,validation=val.metadata)
    rows=[];step_now=[0]
    def capture(u,d,a,**kwargs):
        result,summary=observe_solve(u,d,a,**kwargs)
        rows.append(dict(step=step_now[0],pair=f"pair{len(rows)%6}",summary=summary,
                         certified=result.converged,fallback=result.fallback,
                         newton_iterations=result.iterations))
        return result
    for segment in (0,25):
        seed_everything(config["seed"],config["tf32"])
        model=TinyTransformer(ModelConfig(**config["model"])).float().cuda()
        opts=Optimizers(model,"qso",config)
        if segment==25:assert load_checkpoint(source/"checkpoint.pt",model,opts,config,metadata)==25
        with patch.object(optimizer_module,"solve_coupled",capture):
            for step in range(segment,segment+25):
                step_now[0]=step
                rec=train_step(model,opts,train,config,step)
                assert rec["batch_sha256"]==baseline[step]["batch_sha256"]
                assert rec["loss"]==baseline[step]["loss"]
    return rows


def near_guard():
    gen=torch.Generator(device="cuda").manual_seed(9151)
    rows=[]
    for ratio in (1.01e-4,1.02e-4,1.05e-4,1.1e-4,1.2e-4,1.5e-4,2e-4,3e-4):
        m,n=48,16
        q,_=torch.linalg.qr(torch.randn(m,n,device="cuda",dtype=torch.float64,generator=gen))
        v,_=torch.linalg.qr(torch.randn(n,n,device="cuda",dtype=torch.float64,generator=gen))
        s=torch.linspace(1.,.2,n,device="cuda",dtype=torch.float64);s[-1]=ratio
        if ratio==1.5e-4:s[:2]=1.;s[-2:]=ratio
        b=(q*s)@v.T
        u=torch.randn(m,n,device="cuda",dtype=torch.float64,generator=gen)
        u=u/u.norm(dim=1)[:,None]
        projection=(u*b).sum(1)/b.square().sum(1)
        d=2*projection[:,None]*b-u
        a=torch.stack((b,b.clone()))
        result,summary=observe_solve(u,d,a,config=solver.SolverConfig(
            fallback=False,max_iterations=3),initial_lambda=torch.zeros(m,device="cuda",dtype=torch.float64))
        rows.append(dict(ratio=ratio,summary=summary,converged=result.converged,
                         reason=result.reason,newton_iterations=result.iterations))
    return rows


if __name__=="__main__":
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert "H100" in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.linalg.svd(torch.randn(24,8,device="cuda",dtype=torch.float64),full_matrices=False)
    start=time.perf_counter()
    real=real_replay()
    stress=near_guard()
    output=dict(job=os.environ["SLURM_JOB_ID"],torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0),real=real,near_guard=stress,
                elapsed_seconds=time.perf_counter()-start)
    target=Path("cluster")/f"smooth_qdwh_decisions-{os.environ['SLURM_JOB_ID']}.json"
    target.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    keys=("evaluations","factor_pass","value_violations","gradient_violations",
          "polar_bound_violations","hvps","hvp_violations","curvature_certified",
          "curvature_unsafe","cg","cg_certified","armijo","armijo_certified",
          "armijo_unsafe","certificates","certificate_certified","certificate_unsafe",
          "svd_fallbacks","qdwh_controlled")
    totals={k:sum(r["summary"][k] for r in real) for k in keys}
    times={k:sum(r["summary"]["times"].get(k,0) for r in real)
           for k in ("qdwh_kernel","factor_posterior","derivative_posterior",
                     "certificate_posterior","cg_decision_including_derivative")}
    times["svd_controlled_evaluation"]=sum(r["summary"]["times"].get("svd_controlled_evaluation",0)
                                            for r in real)
    print("RESULT",json.dumps(dict(path=str(target),totals=totals,times=times,
        near_guard=[r["summary"] for r in stress],elapsed_seconds=output["elapsed_seconds"])),flush=True)
