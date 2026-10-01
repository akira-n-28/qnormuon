"""Final independent CPU KKT and worst-direction perturbation checks."""
import os
if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('SLURM required')
import argparse
import json
from pathlib import Path
import torch
import qnormuon.coupled_solver as cs
from experiments.near_rank_optimum import kkt
from cluster.near_rank_optimum import FIXTURES


def main():
    parser=argparse.ArgumentParser();parser.add_argument('refined');args=parser.parse_args()
    root=Path(args.refined);results={}
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    for name,path in FIXTURES.items():
        f=torch.load(path,map_location='cuda',weights_only=False);u,d,a=(f[k].double() for k in ('u','d','a'))
        candidate=torch.load(root/f'{name}-lbfgs.pt',map_location='cuda',weights_only=False);lam=candidate['lam']
        base,polar,pair=kkt(u,d,a,lam)
        _,prod=cs.certificate(u,d,a,lam,polar,cs.Counts(),primal_norm_backend='gram_upper')
        cpu_check,cpu_polar,cpu_pair=kkt(u.cpu(),d.cpu(),a.cpu(),lam.cpu())
        row=dict(production_radial_certificate=prod,cpu_kkt=cpu_check,
            cpu_polar_relative=float((cpu_polar.cuda()-polar).norm()/polar.norm()),adverse_sensitivity=[])
        b=a-cs.adjoint(u,d,lam);left,s,right=torch.linalg.svd(b,full_matrices=False)
        generator=torch.Generator(device='cuda').manual_seed(701)
        for side in range(2):
            normal=torch.randn(u.shape[0],dtype=u.dtype,device=u.device,generator=generator)
            normal-=left[side]@(left[side].T@normal);normal/=normal.norm()
            for scale in (torch.finfo(torch.float64).eps,torch.finfo(torch.float32).eps):
                for kind in ('smallest_normal','smallest_singular_shrink'):
                    perturb=torch.zeros_like(a)
                    vector=normal if kind=='smallest_normal' else -left[side,:,-1]
                    perturb[side]=scale*b[side].norm()*vector[:,None]*right[side,-1,:][None,:]
                    check,np,nr=kkt(u,d,a+perturb,lam)
                    row['adverse_sensitivity'].append(dict(side=side,kind=kind,relative_scale=scale,
                        input_absolute=float(perturb.norm()),kkt=check,
                        polar_relative=float((np-polar).norm()/polar.norm()),
                        direction_relative=float((nr-pair).norm()/pair.norm())))
        results[name]=row
        print(name,'gram gap',prod['normalized_gap'],'cpu gradient',cpu_check['gradient_norm'],flush=True)
    (root/'final_audit.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
