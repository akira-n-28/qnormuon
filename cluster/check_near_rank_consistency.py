"""High-accuracy endpoint consistency and temporal sensitivity; fixtures only."""
import os
if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('SLURM required')
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
import qnormuon.coupled_solver as cs
from qnormuon.optimizer import regular_canonicalize
from experiments.near_rank_optimum import independent_lbfgs,kkt
from cluster.near_rank_optimum import FIXTURES


@torch.no_grad()
def normal_coercivity(u,d,polar,singular):
    """Chunked Gershgorin lower bound for the normal part of the dual Hessian.

    H >= sum_j [diag(||W_i||^2)-(Q Q.T) o (W W.T)]/sigma_max_j.
    No dense Hessian or full m x m array is formed. A positive lower bound
    suffices for local strict convexity; a nonpositive bound proves nothing.
    """
    minima=[]
    for start in range(0,u.shape[0],64):
        stop=min(start+64,u.shape[0]);rows=torch.arange(start,stop,device=u.device)
        block=torch.zeros(stop-start,u.shape[0],dtype=u.dtype,device=u.device)
        for weight,q,s in zip((u,d),polar,singular):
            block-=(q[start:stop]@q.T)*(weight[start:stop]@weight.T)/s[0]
            block[torch.arange(stop-start,device=u.device),rows]+=weight[start:stop].square().sum(1)/s[0]
        diag=block[torch.arange(stop-start,device=u.device),rows]
        minima.append(float((2*diag-block.abs().sum(1)).min()))
    return min(minima)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('refined');args=parser.parse_args()
    root,refined=Path(args.root),Path(args.refined)
    output=root/('consistency-'+os.environ['SLURM_JOB_ID']);output.mkdir()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    results={}
    for name,path in FIXTURES.items():
        f=torch.load(path,map_location='cuda',weights_only=False);u,d,a=(f[k].double() for k in ('u','d','a'))
        target=torch.load(refined/f'{name}-lbfgs.pt',map_location='cuda',weights_only=False)
        target_check,_,_=kkt(u,d,a,target['lam']);row={'starts':{}}
        for endpoint in sorted(root.glob(name+'-*.pt')):
            state=torch.load(endpoint,map_location='cuda',weights_only=False)
            lam,history,counts=independent_lbfgs(u,d,a,state['lam'],max_iterations=100)
            check,polar,pair=kkt(u,d,a,lam)
            row['starts'][endpoint.stem]=dict(kkt=check,counts=asdict(counts),iterations=len(history)-1,
                lambda_distance=float((lam-target['lam']).norm()),
                polar_relative=float((polar-target['polar']).norm()/target['polar'].norm()),
                direction_relative=float((pair-target['pair']).norm()/target['pair'].norm()),
                objective_difference=check['metrics']['dual_objective']-target_check['metrics']['dual_objective'])
            torch.save(dict(lam=lam.cpu(),pair=pair.cpu()),output/f'{endpoint.stem}-refined.pt')
        raw=f['raw'];_,_,scale=regular_canonicalize(raw['up'],raw['down'],dtype=torch.float32)
        gu=(scale[:,None]*raw['grad_up'].double()).float()
        gd=(raw['grad_down'].T.double()/scale[:,None]).float()
        gc=torch.stack((gu,gd)).double()
        # Inverse of the documented fp32 EMA recurrence; this reconstructs only
        # an estimate, since the original multiply/add roundings are not invertible.
        previous=(a-.05*gc)/.95
        delta=a-previous
        row['temporal_objective_estimate']=dict(relative_delta=float(delta.norm()/a.norm()),
            previous_objective_norm=float(previous.norm()),current_objective_norm=float(a.norm()),
            status='inverse EMA estimate; not an exact historical state')
        row['temporal_sensitivity']=[]
        for fraction in (torch.finfo(torch.float32).eps,1e-2,1.):
            newa=a+fraction*delta
            check,polar,pair=kkt(u,d,newa,target['lam'])
            row['temporal_sensitivity'].append(dict(fraction_of_estimated_change=fraction,kkt=check,
                polar_relative=float((polar-target['polar']).norm()/target['polar'].norm()),
                direction_relative=float((pair-target['pair']).norm()/target['pair'].norm())))
        singular=torch.linalg.svdvals(a-cs.adjoint(u,d,target['lam']))
        row['normal_hessian_gershgorin_lower']=normal_coercivity(u,d,target['polar'],singular)
        print(name,json.dumps(row),flush=True)
        results[name]=row
    (output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
