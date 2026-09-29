"""Research-only H100 study of projected-primal radial norm backends."""
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
from benchmarks.tiny_transformer import (
    ModelConfig, Optimizers, TinyTransformer, datasets, load_checkpoint,
    seed_everything, train_step,
)


def gamma(k):
    unit = torch.finfo(torch.float64).eps / 2
    return k * unit / (1 - k * unit)


def gram_upper(p, *, profile=False):
    """Standard-rounding-model upper estimate from all EVD residuals.

    Exact-arithmetic bound: if R=GQ-QD and delta=||Q'Q-I||_F<1, then
    lambda_max(G) <= max(D,0)*(1+delta) + ||G||_F*delta
                     + ||R||_F*sqrt(1+delta).
    The computed EVD need not satisfy a backend-specific forward error model:
    its measured full residual and orthogonality defect supply a posteriori
    bounds. Standard gamma_k allowances cover GEMM/reductions and final scalars.
    This remains a conditional IEEE floating-point model, not interval proof.
    """
    assert p.dtype == torch.float64 and p.ndim == 3
    m, n = p.shape[-2:]
    t = time.perf_counter()
    amplitude = p.abs().amax((-2,-1))
    safe_amplitude = torch.where(amplitude > 0, amplitude, 1.)
    x = p / safe_amplitude[:,None,None]
    g = x.transpose(-2, -1) @ x
    g = .5 * (g + g.transpose(-2, -1))
    torch.cuda.synchronize()
    gemm = time.perf_counter() - t
    t = time.perf_counter()
    values, q = torch.linalg.eigh(g)
    torch.cuda.synchronize()
    eigh = time.perf_counter() - t
    t = time.perf_counter()
    eye = torch.eye(n, dtype=p.dtype, device=p.device)
    defect = torch.linalg.matrix_norm(q.transpose(-2, -1) @ q - eye, ord="fro")
    residual = torch.linalg.matrix_norm(g @ q - q * values.unsqueeze(-2), ord="fro")
    norm_g = torch.linalg.matrix_norm(g, ord="fro")
    norm_q = torch.linalg.matrix_norm(q, ord="fro")
    norm_x2 = x.square().sum((-2,-1))
    unit = torch.finfo(torch.float64).eps / 2
    # Frobenius reductions can touch n^2 (or mn) terms. Inflate their
    # observed values before using them in the exact algebraic bound.
    g_bound = norm_g / (1-gamma(n*n))
    q_bound = norm_q / (1-gamma(n*n))
    x2_bound = norm_x2 / (1-gamma(m*n))
    delta = (defect / (1-gamma(n*n)) + gamma(n)*q_bound.square()
             + unit*math.sqrt(n))
    rho = (residual / (1-gamma(n*n)) + gamma(n)*g_bound*q_bound
           + 2*unit*q_bound*(g_bound+values.abs().amax(-1)))
    gram_round = gamma(m)*x2_bound + unit*g_bound
    upper_lambda = (values[:,-1].clamp_min(0)*(1+delta) + g_bound*delta
                    + rho*(1+delta).sqrt() + gram_round)
    division_round = (unit/(1-unit))*x2_bound.sqrt()
    result = safe_amplitude*(upper_lambda.clamp_min(0).sqrt()*(1+gamma(16))
                             + division_round)*(1+gamma(16))
    torch.cuda.synchronize()
    overhead = time.perf_counter() - t
    if profile:
        return result, dict(gemm=gemm, eigh=eigh, bound=overhead,
                            defect=defect.tolist(), residual=residual.tolist(),
                            delta_bound=delta.tolist(), rho_bound=rho.tolist(),
                            gram_round=gram_round.tolist(),
                            division_round=division_round.tolist())
    return result


def gram_raw(p):
    return torch.linalg.eigvalsh(p.transpose(-2, -1) @ p)[:, -1].clamp_min(0).sqrt()


def measured(fn, repeats=7):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - t)
    return dict(median=statistics.median(values), p95=sorted(values)[math.ceil(.95 * len(values)) - 1])


def prescribed(m, n, spectrum, generator):
    left, _ = torch.linalg.qr(torch.randn(m, n, device="cuda", dtype=torch.float64, generator=generator), mode="reduced")
    right, _ = torch.linalg.qr(torch.randn(n, n, device="cuda", dtype=torch.float64, generator=generator))
    return (left * spectrum) @ right.T


def corpus(generator):
    cases = []
    for m, n in ((12, 5), (64, 16), (1024, 384)):
        base = torch.randn(m, n, dtype=torch.float64, device="cuda", generator=generator)
        cases.append((f"gaussian_{m}_{n}", base))
        linear = torch.linspace(1, .2, n, device="cuda", dtype=torch.float64)
        spectra = {
            "near_rank1": torch.cat((torch.ones(1, device="cuda", dtype=torch.float64),
                                      torch.full((n-1,), 1e-9, device="cuda", dtype=torch.float64))),
            "clustered_top": linear.clone(),
            "repeated_top": linear.clone(),
            "guard_1e-4": linear.clone(),
            "below_guard_1e-8": linear.clone(),
            "dynamic_1e-12": torch.logspace(0, -12, n, device="cuda", dtype=torch.float64),
        }
        spectra["clustered_top"][:2] = torch.tensor([1, 1-1e-12], device="cuda", dtype=torch.float64)
        spectra["repeated_top"][:min(3, n)] = 1
        spectra["guard_1e-4"][-1] = 1e-4
        spectra["below_guard_1e-8"][-1] = 1e-8
        for name, s in spectra.items():
            cases.append((f"{name}_{m}_{n}", prescribed(m, n, s, generator)))
        unit = prescribed(m, n, torch.ones(n, device="cuda", dtype=torch.float64), generator)
        for eps in (-1e-12, 0, 1e-12, -1e-8, 1e-8):
            cases.append((f"unit_{eps:+g}_{m}_{n}", unit * (1 + eps)))
    for scale in (1e-160, 1e-60, 1e60, 1e160):
            cases.append((f"scaled_{scale:g}_{m}_{n}", unit * scale))
    return cases


def compare_norm(name, p):
    batch = p.unsqueeze(0) if p.ndim == 2 else p
    oracle = torch.linalg.svdvals(batch)[:, 0]
    try:
        raw = gram_raw(batch)
        if not bool(torch.isfinite(raw).all()):
            raise torch.linalg.LinAlgError("raw unscaled Gram overflow/underflow")
        raw_values = raw.tolist()
        raw_absolute = (raw-oracle).abs().tolist()
        raw_relative = ((raw-oracle).abs()/oracle).tolist()
    except torch.linalg.LinAlgError:
        raw_values = raw_absolute = raw_relative = None
    upper, components = gram_upper(batch, profile=True)
    slack = upper - oracle
    return dict(name=name, shape=list(batch.shape), sigma_svd=oracle.tolist(),
                sigma_gram=raw_values, sigma_upper=upper.tolist(),
                absolute_error=raw_absolute, relative_error=raw_relative,
                upper_slack=slack.tolist(), components=components,
                deterministic=torch.equal(upper, gram_upper(batch)))


def certificate_variant(u, d, a, lam, candidate, *, cached_dual, norm):
    # Independent study implementation: same projection/scaling and all
    # acceptance fields as production, changing only radial norm evaluation.
    p = solver.horizontal_project(u.double(), d.double(), candidate.double())
    radii = norm(p)
    scale = max(1., float(radii.max()))
    p = p / scale
    if float((a*p).sum()) < 0:
        p = -p
    if cached_dual is None:
        spectra = torch.linalg.svdvals(a-solver.adjoint(u,d,lam))
        dual = float(spectra.sum())
        rcond = float((spectra[:, -1]/spectra[:, 0]).min())
    else:
        dual = cached_dual.magnitude * float(cached_dual.singular.sum())
        rcond = float((cached_dual.singular[:, -1]/cached_dual.singular[:, 0]).min())
    primal = float((a*p).sum())
    gap = dual-primal
    denom = max(abs(primal),abs(dual),torch.finfo(torch.float64).tiny)
    residual = solver.horizontal_residual(u,d,p)
    weight = (u.square()+d.square()).sum(1).sqrt()
    metrics = dict(primal_objective=primal,dual_objective=dual,normalized_gap=max(0.,gap)/denom,
                   signed_normalized_gap=gap/denom,residual_rcond=rcond,
                   spectral_norms=(radii/scale).tolist(),
                   spectral_excess=max(0.,float(radii.max())/scale-1.),
                   horizontal_residual=float(residual.abs().max()),
                   normalized_horizontal_residual=float((residual.abs()/weight).max()))
    return p, metrics, scale


def main():
    assert sitecustomize.QSO_NETWORK_GUARD_ACTIVE
    assert "H100" in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    generator = torch.Generator(device="cuda").manual_seed(8217)
    # Warm CUDA before all timings.
    warm = torch.randn(1024,384,device="cuda",dtype=torch.float64,generator=generator)
    torch.linalg.svdvals(warm)
    gram_upper(warm.unsqueeze(0))
    adversarial = [compare_norm(name,p) for name,p in corpus(generator)]
    config = json.loads(Path("configs/tiny_transformer/smoke.json").read_text())
    source = Path(config["output_root"])/f"smoke-28922-qso-seed{config['seed']}"
    if not (source/"checkpoint.pt").is_file():
        raise FileNotFoundError(source/"checkpoint.pt")
    seed_everything(config["seed"],config["tf32"])
    train,val = datasets(config)
    metadata = dict(train=train.metadata,validation=val.metadata)
    model = TinyTransformer(ModelConfig(**config["model"])).float().cuda()
    opts = Optimizers(model,"qso",config)
    assert load_checkpoint(source/"checkpoint.pt",model,opts,config,metadata)==25
    names=[p.name for p in opts.paired.pairs]
    real=[]
    active=[None]
    original_solve=optimizer_module.solve_coupled
    original_cert=solver.certificate
    call=[0]
    def observed_solve(*args,**kwargs):
        active[0]=names[call[0]%6]
        call[0]+=1
        return original_solve(*args,**kwargs)
    def observed_cert(u,d,a,lam,candidate,counts,**kwargs):
        p0,m0=original_cert(u,d,a,lam,candidate,counts,**kwargs)
        p1,m1,s1=certificate_variant(u,d,a,lam,candidate,cached_dual=kwargs.get("cached_dual"),norm=gram_upper)
        oracle_norm=torch.linalg.svdvals(p1)[:,0]
        real.append(dict(step=current_step[0],pair=active[0],ordinal=sum(r["step"]==current_step[0] and r["pair"]==active[0] for r in real),
                         svd_scale=max(1.,float(torch.linalg.svdvals(solver.horizontal_project(u,d,candidate)).max())),
                         gram_scale=s1, max_direction_error=float((p0-p1).abs().max()),
                         max_actual_norm_after_scale=float(oracle_norm.max()),
                         accepted_svd=solver.accepted(m0,3e-5),accepted_gram=solver.accepted(m1,3e-5),
                         fields={key:dict(svd=m0[key],gram=m1[key]) for key in
                                 ("primal_objective","dual_objective","normalized_gap","spectral_excess",
                                  "horizontal_residual","normalized_horizontal_residual","residual_rcond")},
                         margin=float(gram_upper(solver.horizontal_project(u,d,candidate)) .max()-torch.linalg.svdvals(solver.horizontal_project(u,d,candidate)).max())))
        return p0,m0
    current_step=[25]
    with patch.object(optimizer_module,"solve_coupled",observed_solve),patch.object(solver,"certificate",observed_cert):
        for step in range(25,50):
            current_step[0]=step
            record=train_step(model,opts,train,config,step)
            baseline=[json.loads(line) for line in (source/"metrics.jsonl").read_text().splitlines()][step]
            assert record["batch_sha256"]==baseline["batch_sha256"] and record["loss"]==baseline["loss"]
    assert call[0]==25*6 and len(real)>=450
    # Certificate boundary: horizontal m=2,n=2 pair with diagonal objective.
    # Change candidate's second diagonal to straddle the 3e-5 gap at norms
    # 1 +/- tiny perturbations; SVD remains the independent oracle.
    edge=[]
    for radial in (1-1e-12,1,1+1e-12,1-1e-8,1+1e-8):
        for target in (3e-5-1e-9,3e-5,3e-5+1e-9):
            dtype=torch.float64;device="cuda"
            u=torch.eye(2,dtype=dtype,device=device);d=u.clone()
            a=torch.stack((u,u));lam=torch.zeros(2,dtype=dtype,device=device)
            second=2*radial*(1-target)-radial
            candidate=torch.stack((torch.diag(torch.tensor([radial,second],dtype=dtype,device=device)),)*2)
            po,mo=solver.certificate(u,d,a,lam,candidate,solver.Counts())
            pg,mg,scale=certificate_variant(u,d,a,lam,candidate,cached_dual=None,norm=gram_upper)
            exact_before=float(torch.linalg.svdvals(candidate).max())
            exact_after=float(torch.linalg.svdvals(pg).max())
            edge.append(dict(radial=radial,target=target,exact_before=exact_before,
                             exact_after=exact_after,svd_gap=mo["normalized_gap"],
                             gram_gap=mg["normalized_gap"],svd_accept=solver.accepted(mo,3e-5),
                             gram_accept=solver.accepted(mg,3e-5),scale=scale))
    # Kernel medians on six real accepted projected candidates.
    selected=[]
    # Reconstruct candidates at the same checkpoint in a separate replay.
    model2=TinyTransformer(ModelConfig(**config["model"])).float().cuda()
    opts2=Optimizers(model2,"qso",config)
    assert load_checkpoint(source/"checkpoint.pt",model2,opts2,config,metadata)==25
    def collect_cert(u,d,a,lam,candidate,counts,**kwargs):
        p=solver.horizontal_project(u,d,candidate)
        selected.append(p)
        return original_cert(u,d,a,lam,candidate,counts,**kwargs)
    with patch.object(solver,"certificate",collect_cert):
        train_step(model2,opts2,train,config,25)
    selected=[selected[i] for i in (2,5,8,11,14,17)]
    # Row-gauge stress of an actual production-shaped projected candidate.
    # The norm kernel itself has no gauge assumption; this intentionally
    # creates anisotropic side matrices while retaining real tensor geometry.
    row_scale=torch.exp2(torch.linspace(-8,8,selected[0].shape[1],
                                        device="cuda",dtype=torch.float64))
    gauged=torch.stack((selected[0][0]*row_scale[:,None],
                         selected[0][1]/row_scale[:,None]))
    gauged=gauged/torch.linalg.svdvals(gauged)[:,0].max()
    adversarial.append(compare_norm("gauge_scaled_real_1024_384",gauged))
    timings={}
    for name,fn in (("svdvals",lambda p:torch.linalg.svdvals(p)[:,0]),
                    ("gram_raw",gram_raw),("gram_upper",gram_upper)):
        per=[measured(lambda p=p:fn(p)) for p in selected]
        timings[name]=dict(per_pair=per,six_pair_median_sum=sum(x["median"] for x in per),
                           six_pair_p95_sum=sum(x["p95"] for x in per))
    _,profile=gram_upper(selected[0],profile=True)
    result=dict(job_id=os.environ["SLURM_JOB_ID"],torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                adversarial=adversarial,real=real,edge=edge,timings=timings,gram_components=profile,
                counts=dict(adversarial=len(adversarial),real=len(real),edge=len(edge)))
    path=Path("cluster")/f"primal_norm_study-{os.environ['SLURM_JOB_ID']}.json"
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("RESULT_PATH",path,flush=True)
    print("COUNTS",result["counts"],flush=True)
    print("ADVERSARIAL_MIN_SLACK",min(min(r["upper_slack"]) for r in adversarial),flush=True)
    print("REAL_MAX_GAP_DELTA",max(abs(r["fields"]["normalized_gap"]["svd"]-r["fields"]["normalized_gap"]["gram"]) for r in real),flush=True)
    print("REAL_DECISION_MISMATCH",sum(r["accepted_svd"]!=r["accepted_gram"] for r in real),flush=True)
    print("EDGE_DECISION_MISMATCH",sum(r["svd_accept"]!=r["gram_accept"] for r in edge),flush=True)
    print("TIMINGS",json.dumps(timings),flush=True)


if __name__=="__main__":
    main()
