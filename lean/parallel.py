
import torch
import numpy as np

def loop_integrate(
    force,
    system,
    context,
    integrator,
    steps,
    num_samples,
    T: float=1.0,
    n: int=8,
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

# =============================================================================
# MPI Operations
# =============================================================================
def get_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD

def get_rank():
    return get_comm().Get_rank()

def broadcast_objects(objects):
    if get_rank() != 0:
        objects = None
    objects = get_comm().bcast(objects, root=0)
    return objects

def mpi_integrate(
    force,
    system,
    context,
    integrator,
    steps,
    num_samples,
    T: float=1.0,
):
    if get_rank() == 0:
        collect = np.random.choice(steps, num_samples, replace=False)
    else:
        collect = None
        force = None
    
    # broadcast objects
    collect, force = broadcast_objects([collect, force])
    
    # integrate
    x, t, a = force.integrate(
        system=system,
        context=context,
        integrator=integrator,
        steps=steps,
        collect=collect,
        T=T,
    )
    
    # gather results
    x = get_comm().gather(x, root=0)
    t = get_comm().gather(t, root=0)
    a = get_comm().gather(a, root=0)
    
    # stack results
    if get_rank() == 0:
        x = torch.stack(x)
        t = torch.stack(t)
        a = torch.stack(a)
        return x, t, a
    
    return None
    
    

    