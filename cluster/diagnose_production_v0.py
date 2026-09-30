import torch
from experiments.horizontal_spectral import random_example, fractional_example, adjoint
from experiments.dual_solver import solve_dual, reference
from qnormuon import solve_coupled, SolverConfig
for name,data in [('fp32-native-901',random_example(901,dtype=torch.float32)),('fractional',fractional_example()[:3])]:
    u,d,a=data
    oracle=reference(u,d,a)
    print(name,'oracle residual spectrum',torch.linalg.svdvals(a.double()-adjoint(u.double(),d.double(),oracle.lam)).tolist(),flush=True)
    for label,result in [('research',solve_dual(u,d,a)),('production',solve_coupled(u,d,a,config=SolverConfig(dtype=a.dtype)))]:
        print(name,label,'reason',result.reason,'fallback',result.fallback,'min_rcond',result.min_accepted_rcond,'gap',result.metrics['normalized_gap'],flush=True)
        print('last history',result.history[-3:],flush=True)
