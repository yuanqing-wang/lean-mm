import math
from typing import defaultdict
import openmm as mm
import numpy as np
import torch
import re
from argparse import Namespace
GROUPS = Namespace(
    SYSTEM=0,
    GAUSSIAN=1,
    UNBIASING=2,
)

from .integrators import integrate
from .unit import LENGTH, ENERGY



class GaussianForce(object):
    def __init__(
        self,
        k=0.01*ENERGY/LENGTH**2,
    ):
        self.k = k
        
    def add_force(self, system):
        force = mm.CustomExternalForce("k*(x^2+y^2+z^2)")
        force.addGlobalParameter("k", self.k)
        for i in range(system.getNumParticles()):
            force.addParticle(i, [])
        force.setForceGroup(GROUPS.GAUSSIAN)
        system.addForce(force)
        return system

class InductiveRadialUnbiasingForce(torch.nn.Module):
    def __init__(
        self,
        n: int,
        num_time: int=32,
    ):
        super().__init__()
        self.n = n
        self.k = torch.nn.Parameter(
            torch.randn(n, n, num_time),
        )
        self.b = torch.nn.Parameter(
            torch.randn(n, n, num_time),
        )
        self.gamma = torch.nn.Parameter(
            torch.randn(n, n, num_time),
        )
        self.num_time = num_time
        
    def forward(
        self,
        x: torch.Tensor,
        t: float,
        scale: float = 1.0,
    ):
        # (N, N)
        x = torch.norm(x.unsqueeze(-2) - x.unsqueeze(-3), dim=-1)
        
        if isinstance(t, float):
            t = torch.tensor(t)
        
        # (T, )
        t = torch.stack(
            [t ** idx for idx in range(self.num_time)],
        )
        
        # (N, N, T)
        x = x.unsqueeze(-1)
        
        # (N, N, T)
        u = self.k * (-((x - self.b)/self.gamma) ** 2).exp()
        
        # (N, N, T)
        u = u * t
        
        # no self interaction
        u[..., torch.arange(self.n), torch.arange(self.n), :] = 0.0
        
        # optionally, set scale
        u = u * scale
        
        # (,)
        return u.sum(dim=[-1, -2, -3])
    
    def forward_openmm(
        self,
        x: torch.Tensor,
        t: float,
        system: mm.System,
        context: mm.Context,
    ):
        self.parametrize(system, context)
        x = x.detach().numpy() * LENGTH
        context.setPositions(x)
        context.setParameter("t", t)
        energy = (
            context
            .getState(getEnergy=True, groups=set([2]))
            .getPotentialEnergy()
            .value_in_unit(ENERGY)
        )
        return energy
        
    def add_force(self, system):
        if any(force.getForceGroup() == 2 for force in system.getForces()):
            return system
        
        mapping = defaultdict(dict)
        for idx_t in range(self.num_time):
            force = mm.CustomBondForce(f"t^{idx_t}*k*exp(-((r-b)/gamma)^2)".replace("t^0*", ""))
            # print(force.getEnergyFunction())
            force.addGlobalParameter("t", 1.0)
            force.addPerBondParameter("b")
            force.addPerBondParameter("k")
            force.addPerBondParameter("gamma")
            force.setForceGroup(GROUPS.UNBIASING)
            for idx0 in range(self.n):
                for idx1 in range(self.n):
                    if idx0 != idx1:
                        k = self.k[idx0, idx1, idx_t].item() * ENERGY
                        b = self.b[idx0, idx1, idx_t].item() * LENGTH
                        gamma = self.gamma[idx0, idx1, idx_t].item() * LENGTH
                        idx_bond = force.addBond(idx0, idx1, [b, k, gamma])
                        mapping[idx_t][idx_bond] = (idx0, idx1, idx_t)
        system.addForce(force)
        self.mapping = mapping
        return system
    
    def parametrize(self, system, context):
        for idx in range(system.getNumForces()):
            force = system.getForce(idx)
            if force.getForceGroup() == 2:
                expression = force.getEnergyFunction()
                idx_t = re.search(r"t\^(\d+)", expression)
                if idx_t is None:
                    idx_t = 0
                else:
                    idx_t = int(idx_t.group(1))
                
                _mapping = self.mapping[idx_t]
                for idx_bond in _mapping:
                    idx0, idx1, idx_t = _mapping[idx_bond]
                    k = self.k[idx0, idx1, idx_t].item()
                    b = self.b[idx0, idx1, idx_t].item()
                    gamma = self.gamma[idx0, idx1, idx_t].item()
                    k = k * ENERGY
                    b = b * LENGTH
                    gamma = gamma * LENGTH
                    force.setBondParameters(idx_bond, idx0, idx1, [b, k, gamma])
                force.updateParametersInContext(context)
        return self
            
            
    def integrate(self, *args, **kwargs):
        return integrate(self, *args, **kwargs)
        
                    

    
    

        
        
        
        
        
        
        
        
        
        
        


