"""SVD-controlled, research-only decision-bound replay on one SLURM H100."""
import json
import math
import os
from pathlib import Path
import statistics
import sys
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
from experiments.smooth_gram import diagnose, gamma
from experiments.smooth_gram_decisions import (armijo_interval,
    derivative_error_bound, gram_derivative, interval_sign, polar_posterior)


def sync():
    torch.cuda.synchronize()


class Shadow:
    def __init__(self):
        self.evaluations=[]
        self.certified=[]
        self.hvps=[]
        self.cg=[]
        self.certificates=[]
        self.seconds=dict(gram=0.,derivative=0.,decision=0.,certificate=0.)
        self.orig_eval=solver.SmoothDual.evaluate
        self.orig_hvp=solver.SmoothDual.hvp
        self.orig_cert=solver.certificate
        self.orig_newton=solver._newton_direction

    def evaluate(self, problem, z):
        ev=self.orig_eval(problem,z)
        b=problem.a-solver.adjoint(problem.u,problem.d,z)
        sync(); start=time.perf_counter()
        gr=[diagnose(side) for side in b]
        posts=[polar_posterior(side,g) for side,g in zip(b,gr)]
        sync(); self.seconds["gram"]+=time.perf_counter()-start
        structure=all(g.accepted for g in gr)
        bounded=all(p.accepted for p in posts)
        row_squared=(problem.u.square()+problem.d.square()).sum(1)
        row_operator_upper=math.sqrt(float(row_squared.max())/(1-gamma(2*problem.u.shape[1]+1)))
        record=dict(ev=ev,b=b,gram=gr,posts=posts,structure=structure,
                    row_operator_upper=row_operator_upper,
                    bounded=bounded,reason="posterior_bound_available" if bounded else
                    next((p.reason for p in posts if not p.accepted),"gram_structural_failure"))
        if bounded:
            record["gradient"]=-solver.horizontal_residual(problem.u,problem.d,
                                               torch.stack([g.polar for g in gr]))
            record["grad_error"]=row_operator_upper*math.hypot(*(p.polar_error for p in posts))
            scalar_round=gamma(2*problem.u.shape[1]+1)*sum(g.measurements["nuclear_upper"] for g in gr)
            record["value_interval"]=(sum(g.measurements["nuclear_lower"] for g in gr)-scalar_round,
                                      sum(g.measurements["nuclear_upper"] for g in gr)+scalar_round)
            record["value_contains_oracle"]=(record["value_interval"][0]-1e-12<=ev.value<=
                                               record["value_interval"][1]+1e-12)
            record["gradient_contains_oracle"]=(float((record["gradient"]-ev.gradient).norm())<=
                                                  record["grad_error"]+1e-12)
        self.evaluations.append(record)
        return ev

    def hvp_bound(self,problem,record,v):
        if not record["bounded"]:
            return None,math.inf
        e=-solver.adjoint(problem.u,problem.d,v)
        start=time.perf_counter()
        ys=[gram_derivative(b,side_e,g) for b,side_e,g in zip(record["b"],e,record["gram"])]
        bounds=[derivative_error_bound(b,side_e,y,g,p)[0]
                for b,side_e,y,g,p in zip(record["b"],e,ys,record["gram"],record["posts"])]
        sync(); self.seconds["derivative"]+=time.perf_counter()-start
        hv=-solver.horizontal_residual(problem.u,problem.d,torch.stack(ys))
        # L_eff L_eff*=I since each row's combined whitened weight is one.
        return hv,record["row_operator_upper"]*math.hypot(*bounds)

    def hvp(self,problem,ev,v):
        oracle=self.orig_hvp(problem,ev,v)
        record=next((r for r in reversed(self.evaluations) if r["ev"] is ev),None)
        if record is None:
            raise RuntimeError("HVP has no matching evaluation")
        approximate,error=self.hvp_bound(problem,record,v)
        if approximate is None:
            self.hvps.append(dict(reason=record["reason"],contained=None,
                                  curvature="uncertain"))
        else:
            observed=float((oracle-approximate).norm())
            radius=error*float(v.norm())
            curvature=float(v@approximate)
            self.hvps.append(dict(reason="bounded",contained=observed<=error+1e-12,
                                  error_bound=error,observed_error=observed,
                                  curvature=interval_sign(curvature,radius),
                                  curvature_oracle=float(v@oracle)))
        return oracle

    def newton(self,problem,ev,curvature_scale,max_cg=30):
        oracle=self.orig_newton(problem,ev,curvature_scale,max_cg)
        record=next((r for r in reversed(self.evaluations) if r["ev"] is ev),None)
        if record is None or not record["bounded"]:
            self.cg.append(dict(reason="gram_structural_failure",direction_certified=False))
            if record is not None:
                record["cg_reason"]="gram_structural_failure"
            return oracle
        sync(); start=time.perf_counter()
        g=record["gradient"]
        ge=record["grad_error"]
        damping=1e-6*curvature_scale
        x=torch.zeros_like(g)
        r=-g.clone(); p=r.clone(); rr=float(r@r)
        target=min(.1,math.sqrt(max(float(g.norm()),1e-16)))*math.sqrt(rr)
        reason="gram_cg_uncertain"
        iterations=0
        direction_error_bound=None
        for _ in range(min(max_cg,g.numel())):
            h,he=self.hvp_bound(problem,record,p)
            if h is None:
                reason="gram_derivative_bound_failed";break
            h=h+damping*p
            c=float(p@h)
            radius=he*float(p.norm())
            if interval_sign(c,radius)!="positive":
                reason="gram_curvature_ambiguous";break
            alpha=rr/c
            x=x+alpha*p
            r=r-alpha*h
            iterations+=1
            new_rr=float(r@r)
            if math.sqrt(new_rr)<=target:
                reason="candidate_cg_stop";break
            p=r+(new_rr/rr)*p
            rr=new_rr
        if reason=="candidate_cg_stop":
            hx,he=self.hvp_bound(problem,record,x)
            true_residual_upper=float((g+hx+damping*x).norm())+ge+he
            direction_error_bound=true_residual_upper/damping
            g_lower=max(0.,float(g.norm())-ge)
            true_target_lower=min(.1,math.sqrt(max(g_lower,1e-16)))*g_lower
            if true_residual_upper> true_target_lower:
                reason="gram_cg_uncertain"
            else:
                slope=float(g@x)
                slope_radius=ge*float(x.norm())
                if interval_sign(slope,slope_radius)!="negative":
                    reason="gram_descent_ambiguous"
                else:
                    reason="gram_cg_and_descent_certified"
        sync();self.seconds["decision"]+=time.perf_counter()-start
        self.cg.append(dict(reason=reason,direction_certified=reason=="gram_cg_and_descent_certified",
                            iterations=iterations,
                            direction_error=float((x-oracle).norm()),
                            direction_error_bound=direction_error_bound,
                            direction_relative=float((x-oracle).norm())/
                                max(float(oracle.norm()),torch.finfo(torch.float64).tiny)))
        record["cg_reason"]=reason
        return oracle

    def certificate(self,u,d,a,lam,candidate,counts,**kwargs):
        record=next((r for r in reversed(self.evaluations) if r["ev"].pair is candidate),None)
        if record is not None:
            self.certified.append(record)
        result=self.orig_cert(u,d,a,lam,candidate,counts,**kwargs)
        oracle_accept=(solver.accepted(result[1],3e-5) and record is not None
                       and record["ev"].rcond>1e-4)
        if record is None or not record["bounded"] or kwargs.get("cached_dual") is None:
            self.certificates.append(dict(reason="gram_structural_failure",
                                          oracle_accept=oracle_accept))
            if record is not None:
                record["certificate_reason"]="gram_structural_failure"
            return result
        gp=torch.stack([g.polar for g in record["gram"]])
        cache=solver.CachedDualSpectrum(lam,gp,
            torch.stack([g.singular for g in record["gram"]]),
            kwargs["cached_dual"].magnitude)
        sync(); start=time.perf_counter()
        recovered,metrics=self.orig_cert(u,d,a,lam,gp,solver.Counts(),cached_dual=cache,
                                 primal_norm_backend="gram_upper")
        mag=cache.magnitude
        dual_upper=mag*record["value_interval"][1]
        dual_lower=mag*record["value_interval"][0]
        primal=metrics["primal_objective"]
        primal_round=gamma(2*recovered.numel()+1)*float((a*recovered).abs().sum())
        primal_lower=primal-primal_round
        primal_upper=primal+primal_round
        if primal_lower<=0 or dual_lower<=0:
            self.certificates.append(dict(reason="gram_certificate_ambiguous",
                                          oracle_accept=oracle_accept,safe=True))
            record["certificate_reason"]="gram_certificate_ambiguous"
            return result
        denom_lower=max(primal_lower,dual_lower,torch.finfo(torch.float64).tiny)
        denom_upper=max(primal_upper,dual_upper,torch.finfo(torch.float64).tiny)
        gap_upper=max(0.,dual_upper-primal_lower)/denom_lower
        numerator_lower=dual_lower-primal_upper
        gap_lower=numerator_lower/(denom_lower if numerator_lower<0 else denom_upper)
        rcond_lower=min(g.measurements["rcond_lower"] for g in record["gram"])
        feasible=(metrics["normalized_horizontal_residual"]<=1e-10 and
                  metrics["spectral_excess"]<=1e-12 and rcond_lower>1e-4)
        if feasible and gap_upper<=3e-5 and gap_lower>=-1e-10:
            reason="accept"
        elif gap_lower>3e-5 or rcond_lower<=1e-4:
            reason="reject"
        else:
            reason="gram_certificate_ambiguous"
        self.certificates.append(dict(reason=reason,oracle_accept=oracle_accept,
             safe=reason=="gram_certificate_ambiguous" or (reason=="accept")==oracle_accept,
             gap_upper=gap_upper,gap_lower=gap_lower,oracle_gap=result[1]["normalized_gap"],
             rcond_lower=rcond_lower))
        record["certificate_reason"]=reason
        sync(); self.seconds["certificate"]+=time.perf_counter()-start
        return result

    def armijo(self):
        results=[]
        indices=[self.evaluations.index(r) for r in self.certified]
        for j,current_index in enumerate(indices):
            next_index=indices[j+1] if j+1<len(indices) else len(self.evaluations)
            trials=self.evaluations[current_index+1:next_index+1] if j+1<len(indices) else self.evaluations[current_index+1:]
            if not trials:
                continue
            current=self.evaluations[current_index]
            direction=trials[0]["ev"].coordinate-current["ev"].coordinate
            for k,trial in enumerate(trials):
                alpha=.5**k
                is_accepted=(j+1<len(indices) and k==len(trials)-1)
                if not current["bounded"] or not trial["bounded"]:
                    trial["armijo_reason"]="gram_structural_failure"
                    results.append(dict(reason="gram_structural_failure",oracle_accept=is_accepted));continue
                slope=float(current["gradient"]@direction)
                slope_err=current["grad_error"]*float(direction.norm())
                slope_int=(slope-slope_err,slope+slope_err)
                cl,cu=current["value_interval"]
                minabs=0. if cl<=0<=cu else min(abs(cl),abs(cu))
                rounding_lower=2*torch.finfo(torch.float64).eps*max(1.,minabs)
                rounding_upper=2*torch.finfo(torch.float64).eps*max(1.,abs(cl),abs(cu))
                arm=armijo_interval(current["value_interval"],trial["value_interval"],
                                    slope_int,alpha=alpha,
                                    rounding_allowance=(rounding_lower,rounding_upper))
                rcond_lower=min(g.measurements["rcond_lower"] for g in trial["gram"])
                rcond_upper=min(g.measurements["rcond_upper"] for g in trial["gram"])
                resolved=(alpha*min(abs(slope_int[0]),abs(slope_int[1]))>
                          rounding_upper and slope_int[1]<0)
                gtrial_upper=float(trial["gradient"].norm())+trial["grad_error"]
                gcur_lower=max(0.,float(current["gradient"].norm())-current["grad_error"])
                grad_improves=gtrial_upper<gcur_lower
                if rcond_upper<=1e-4:
                    reason="reject"
                elif arm=="reject":
                    reason="reject"
                elif arm=="accept" and rcond_lower>1e-4 and (resolved or grad_improves):
                    reason="accept"
                else:
                    reason="gram_armijo_ambiguous"
                results.append(dict(reason=reason,oracle_accept=is_accepted,
                                    safe=reason=="gram_armijo_ambiguous" or (reason=="accept")==is_accepted,
                                    armijo=arm,rcond_lower=rcond_lower))
                trial["armijo_reason"]=reason
        return results

    def summary(self):
        arms=self.armijo()
        fallback_reasons=[]
        for record in self.evaluations:
            reasons=[]
            if not record["bounded"]:
                reasons.append(record["reason"])
            if record.get("certificate_reason")=="gram_certificate_ambiguous":
                reasons.append("gram_certificate_ambiguous")
            if record.get("cg_reason") not in (None,"gram_cg_and_descent_certified"):
                reasons.append(record["cg_reason"])
            if record.get("armijo_reason") in ("gram_armijo_ambiguous","gram_structural_failure"):
                reasons.append(record["armijo_reason"])
            fallback_reasons.append(reasons)
        return dict(evaluations=len(self.evaluations),structure_pass=sum(r["structure"] for r in self.evaluations),
                    derivative_bound_pass=sum(r["bounded"] for r in self.evaluations),
                    value_interval_violations=sum(r.get("value_contains_oracle") is False for r in self.evaluations),
                    gradient_bound_violations=sum(r.get("gradient_contains_oracle") is False for r in self.evaluations),
                    hvps=len(self.hvps),hvp_bound_violations=sum(r.get("contained") is False for r in self.hvps),
                    curvature_certified=sum(r["curvature"]!="uncertain" for r in self.hvps),
                    curvature_sign_violations=sum(r["curvature"] in ("positive","negative") and
                        (r["curvature"]=="positive")!=(r["curvature_oracle"]>0) for r in self.hvps if "curvature_oracle" in r),
                    cg=len(self.cg),cg_certified=sum(r["direction_certified"] for r in self.cg),
                    cg_max_observed_direction_relative=max((r.get("direction_relative",0.) for r in self.cg),default=0.),
                    cg_max_observed_direction_absolute=max((r.get("direction_error",0.) for r in self.cg),default=0.),
                    cg_max_bound=max((r.get("direction_error_bound") or 0. for r in self.cg),default=0.),
                    cg_reasons={key:sum(r["reason"]==key for r in self.cg) for key in sorted({r["reason"] for r in self.cg})},
                    armijo=len(arms),armijo_certified=sum(r["reason"]!="gram_armijo_ambiguous" for r in arms),
                    armijo_unsafe=sum(r.get("safe") is False for r in arms),
                    certificates=len(self.certificates),
                    certificate_certified=sum(r["reason"]!="gram_certificate_ambiguous" for r in self.certificates),
                    certificate_unsafe=sum(r.get("safe") is False for r in self.certificates),
                    certificate_reasons={key:sum(r["reason"]==key for r in self.certificates)
                                         for key in sorted({r["reason"] for r in self.certificates})},
                    adaptive_gram_evaluations=sum(not reasons for reasons in fallback_reasons),
                    adaptive_svd_evaluations=sum(bool(reasons) for reasons in fallback_reasons),
                    decomposition_fallback_reasons={key:sum(key in reasons for reasons in fallback_reasons)
                                                    for key in sorted({key for reasons in fallback_reasons for key in reasons})},
                    armijo_reasons={key:sum(r["reason"]==key for r in arms) for key in sorted({r["reason"] for r in arms})},
                    seconds=self.seconds,
                    min_rcond_lower=min((min(g.measurements["rcond_lower"] for g in r["gram"])
                                         for r in self.evaluations if r["structure"]),default=None))


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
    seed_everything(config["seed"],config["tf32"])
    train,val=datasets(config)
    metadata=dict(train=train.metadata,validation=val.metadata)
    rows=[];step_now=[0]
    original=optimizer_module.solve_coupled
    def capture(u,d,a,**kwargs):
        result,summary=observe_solve(u,d,a,**kwargs)
        rows.append(dict(step=step_now[0],name=f"pair{len(rows)%6}",summary=summary,
                         certified=result.converged,fallback=result.fallback,
                         iterations=result.iterations))
        return result
    for segment in (0,25):
        seed_everything(config["seed"],config["tf32"])
        model=TinyTransformer(ModelConfig(**config["model"])).float().cuda()
        opts=Optimizers(model,"qso",config)
        if segment==25:
            assert load_checkpoint(source/"checkpoint.pt",model,opts,config,metadata)==25
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
    def kernel_time(fn):
        for _ in range(3):fn()
        sync();values=[]
        for _ in range(20):
            sync();start=time.perf_counter();fn();sync()
            values.append(time.perf_counter()-start)
        return statistics.median(values)
    for ratio in (1.01e-4,1.05e-4,1.2e-4,2e-4,3e-4):
        m,n=48,16
        q,_=torch.linalg.qr(torch.randn(m,n,device="cuda",dtype=torch.float64,generator=gen))
        v,_=torch.linalg.qr(torch.randn(n,n,device="cuda",dtype=torch.float64,generator=gen))
        s=torch.linspace(1.,.2,n,device="cuda",dtype=torch.float64);s[-1]=ratio
        b=(q*s)@v.T
        pair=torch.stack((b,b))
        svd_pair_seconds=kernel_time(lambda:torch.linalg.svd(pair,full_matrices=False))
        gram_posterior_pair_seconds=kernel_time(lambda:[polar_posterior(side,diagnose(side))
                                                for side in pair])
        u=torch.randn(m,n,device="cuda",dtype=torch.float64,generator=gen)
        u=u/u.norm(dim=1)[:,None]
        projection=(u*b).sum(1)/b.square().sum(1)
        d=2*projection[:,None]*b-u  # same row norms, same <row,B>, different polar row action
        # Both residuals initially have the prescribed near-guard spectrum.
        a=torch.stack((b,b.clone()))
        result,summary=observe_solve(u,d,a,config=solver.SolverConfig(
            fallback=False,max_iterations=3),initial_lambda=torch.zeros(m,device="cuda",dtype=torch.float64))
        rows.append(dict(ratio=ratio,summary=summary,iterations=result.iterations,
                         result_reason=result.reason,converged=result.converged,
                         svd_pair_seconds=svd_pair_seconds,
                         gram_posterior_pair_seconds=gram_posterior_pair_seconds))
    return rows


if __name__=="__main__":
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert "H100" in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.linalg.svd(torch.randn(24,8,device="cuda",dtype=torch.float64),full_matrices=False)
    start=time.perf_counter()
    real=real_replay() if "--near-guard-only" not in sys.argv else []
    stress=near_guard()
    output=dict(job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__,real=real,near_guard=stress,
                elapsed_seconds=time.perf_counter()-start)
    target=Path("cluster")/f"smooth_gram_decisions-{os.environ['SLURM_JOB_ID']}.json"
    target.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    totals={key:sum(r["summary"][key] for r in real) for key in
            ("evaluations","structure_pass","derivative_bound_pass",
             "value_interval_violations","gradient_bound_violations","hvps",
             "hvp_bound_violations","curvature_certified","cg","cg_certified",
             "armijo","armijo_certified","armijo_unsafe","certificates",
             "certificate_certified","certificate_unsafe")}
    print("RESULT",json.dumps(dict(path=str(target),totals=totals,
          near_guard=[r["summary"] for r in stress],
          elapsed_seconds=output["elapsed_seconds"])),flush=True)
