"""Fixture-only classification tools; never imported by production.

The smallest positive float guard admits every computed positive residual
singular value. This is NOT a production admission or direction guarantee.
No positive singular value is discarded. Independent KKT checks always rebuild
the original-coordinate residual and use the complete fp64 thin SVD.
"""
import math
import torch
import qnormuon.coupled_solver as cs


def research_config():
    return cs.SolverConfig(rcond_guard=math.nextafter(0., 1.), tolerance=1e-10,
                           fallback=False, independent_certificate=True)


@torch.no_grad()
def kkt(u, d, a, lam):
    b = a - cs.adjoint(u, d, lam)
    left, s, right = torch.linalg.svd(b, full_matrices=False)
    if bool((s <= 0).any()):
        raise ValueError('full-rank KKT check requires positive singular values; deficient faces need joint completion')
    polar = left @ right
    gradient = -cs.horizontal_residual(u, d, polar)
    recovered, metrics = cs.certificate(u, d, a, lam, polar, cs.Counts(),
                                       primal_norm_backend='svd')
    return dict(metrics=metrics, gradient_norm=float(gradient.norm()),
                gradient_max=float(gradient.abs().max()),
                normalized_gradient_norm=float((gradient / (u.square()+d.square()).sum(1).sqrt()).norm()),
                sides=[dict(sigma_max=float(x[0]), sigma_min=float(x[-1]),
                            rcond=float(x[-1]/x[0]), tail=x[-8:].tolist()) for x in s]), polar, recovered


@torch.no_grad()
def independent_lbfgs(u, d, a, initial_lambda, max_iterations=400):
    """Independent quasi-Newton classification solve, no Newton/HVP calls.

    Nuclear-value Armijo is used while the decrease is resolvable. At its
    rounding plateau, gradient-norm root refinement is used instead (without
    trusting rounded nuclear-value differences). This
    research root refinement does not redefine production globalization.
    """
    beta=cs.multipliers(u,d,a); centered=a-cs.adjoint(u,d,beta)
    magnitude=cs._stable_norm(centered); coord=(u.square()+d.square()).sum(1).rsqrt()
    problem=cs.SmoothDual(u*coord[:,None],d*coord[:,None],centered/magnitude)
    z=(initial_lambda-beta)/(magnitude*coord)
    ev=problem.evaluate(z); memory=[]; history=[]
    # Independent two-loop quasi-Newton recursion; no cached Newton direction.
    for iteration in range(max_iterations+1):
        history.append(dict(iteration=iteration,value=ev.value,gradient_norm=float(ev.gradient.norm()),
                            rcond=ev.rcond,singular_values=(magnitude*ev.singular).tolist()))
        if float(ev.gradient.norm()) < 2e-12 or iteration==max_iterations:break
        g=ev.gradient; q=g.clone(); coefficients=[]
        for s,y in reversed(memory):
            alpha=float(s@q)/float(s@y);coefficients.append(alpha);q=q-alpha*y
        scale=float((memory[-1][0]@memory[-1][1])/(memory[-1][1]@memory[-1][1])) if memory else float(ev.singular.median())
        direction=scale*q
        for (s,y),alpha in zip(memory,reversed(coefficients)):
            direction=direction+s*(alpha-float(y@direction)/float(s@y))
        direction=-direction
        if float(g@direction)>=0:direction=-scale*g
        found=False
        for restart in range(2):
            if restart:direction=-scale*g
            slope=float(g@direction); step=1.
            for trial_index in range(30):
                trial_z=z+step*direction;trial=problem.evaluate(trial_z)
                problem.counts.line_trials+=1
                rounding=32*torch.finfo(torch.float64).eps*max(1.,abs(ev.value))
                resolved=abs(step*slope)>rounding
                passes=(trial.value<=ev.value+1e-4*step*slope+rounding if resolved
                        else float(trial.gradient.norm())<float(g.norm()))
                if trial.rcond>0 and passes:
                    found=True;break
                step*=.5
            if found:break
        if not found:break
        s=trial_z-z;y=trial.gradient-g
        if float(s@y)>1e-12*float(s.norm()*y.norm()):
            memory.append((s,y));memory=memory[-20:]
        z,ev=trial_z,trial
    return beta+magnitude*coord*z,history,problem.counts


@torch.no_grad()
def decomposition_check(b, left, singular, right):
    """Conservative standard-model rank-separation estimate, not interval proof.

    Orthogonalizing the computed factors defines a nearby matrix with exactly
    the listed singular values. Bound that matrix's distance from B using
    measured factorization/orthogonality residuals plus GEMM/reduction rounding.
    Weyl then separates positive sigma_min from the model uncertainty.
    """
    m,n=b.shape[-2:];unit=torch.finfo(torch.float64).eps/2
    def gamma(k):return k*unit/(1-k*unit)
    eye=torch.eye(n,device=b.device,dtype=b.dtype)
    ul_def=torch.linalg.matrix_norm(left.transpose(-2,-1)@left-eye,ord='fro')
    vr_def=torch.linalg.matrix_norm(right@right.transpose(-2,-1)-eye,ord='fro')
    recon=(left*singular.unsqueeze(-2))@right
    residual=torch.linalg.matrix_norm(b-recon,ord='fro')
    bnorm=torch.linalg.matrix_norm(b,ord='fro');sfro=singular.norm(dim=-1)
    ufro=torch.linalg.matrix_norm(left,ord='fro');vfro=torch.linalg.matrix_norm(right,ord='fro')
    du=ul_def/(1-gamma(n*n))+gamma(m)*ufro.square()+unit*math.sqrt(n)
    dv=vr_def/(1-gamma(n*n))+gamma(n)*vfro.square()+unit*math.sqrt(n)
    if bool((du>=1).any()) or bool((dv>=1).any()):
        raise ValueError('orthogonality too poor for the backward model')
    # delta_F bounds delta_2; ||U-Uhat|| <= delta/(1+sqrt(1-delta)).
    eu=du/(1+(1-du).sqrt());ev=dv/(1+(1-dv).sqrt())
    residual_upper=(residual/(1-gamma(m*n))+gamma(n)*sfro*vfro*(1+du).sqrt()
                    +gamma(3)*(bnorm+torch.linalg.matrix_norm(recon,ord='fro')))
    uncertainty=residual_upper+sfro*(eu*(1+dv).sqrt()+ev)
    return dict(reconstruction_relative=(residual/bnorm).tolist(),left_orthogonality=ul_def.tolist(),
                right_orthogonality=vr_def.tolist(),backward_model_uncertainty=uncertainty.tolist(),
                rank_margin=(singular[:,-1]/uncertainty).tolist())
