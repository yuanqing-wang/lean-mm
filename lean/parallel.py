
import torch

def loop_integrate(
    n: int,
    force,
    system,
    context,
    integrator,
    steps,
    num_samples,
):
    _x, _t, _a = [], [], []
    for idx in range(n):
        x, t, a = force.integrate(
            system=system,
            context=context,
            integrator=integrator,
            steps=steps,
            num_samples=num_samples,
        )
        _x.append(x)
        _t.append(t)
        _a.append(a)
    for x in _x:
        print(x.shape)
    _x, _t, _a = torch.stack(_x), torch.stack(_t), torch.stack(_a)
    return _x, _t, _a