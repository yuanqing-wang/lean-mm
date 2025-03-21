
import torch
import numpy as np

def loop_integrate(
    n: int,
    force,
    system,
    context,
    integrator,
    steps,
    num_samples,
    T: float=1.0,
):
    collect = np.random.choice(steps, num_samples, replace=False)
    _x, _t, _a = [], [], []
    for idx in range(n):
        x, t, a = force.integrate(
            system=system,
            context=context,
            integrator=integrator,
            steps=steps,
            collect=collect,
            T=T,
        )
        _x.append(x)
        _t.append(t)
        _a.append(a)
    _x, _t, _a = torch.stack(_x), torch.stack(_t), torch.stack(_a)
    return _x, _t, _a