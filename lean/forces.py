import math
from typing import defaultdict
import openmm as mm
from openmm import unit
import numpy as np
import torch
import re
from .integrators import integrate

def gaussian_force_factory(system, k=1.00*unit.kilocalories_per_mole/unit.angstrom**2):
    force = mm.CustomExternalForce("k*(x^2+y^2+z^2)")
    force.addGlobalParameter("k", k)
    for i in range(system.getNumParticles()):
        force.addParticle(i, [])
    force.setForceGroup(1)
    system.addForce(force)
    return force

class InductiveRadialBiasingForce(torch.nn.Module):
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
    ):
        # (N, N)
        x = torch.norm(x.unsqueeze(-2) - x.unsqueeze(-3), dim=-1)
        
        # (T, )
        t = torch.cat(
            [torch.sin(2*np.pi*idx*t) for idx in range(self.num_time)],
        )
        
        # (N, N, B, T)
        x = x.unsqueeze(-1).unsqueeze(-1)
        
        # (N, N, B, T)
        u = self.k * (x - self.b) ** 2
        
        # (N, N, B, T)
        u = u * t
        
        # (,)
        return u.sum(dim=[-1, -2, -3, -4])
            
    def add_force(self, system):
        mapping = defaultdict(dict)
        for idx_t in range(self.num_time):
            force = mm.CustomBondForce(f"t^{idx_t}*k*exp(-((r-b)/gamma)^2)")
            force.addGlobalParameter("t", 1.0)
            force.addPerBondParameter("b")
            force.addPerBondParameter("k")
            force.addPerBondParameter("gamma")
            force.setForceGroup(2)
            for idx0 in range(self.n):
                for idx1 in range(self.n):
                    if idx0 != idx1:
                        k = self.k[idx0, idx1, idx_t].item() * unit.kilocalories_per_mole
                        b = self.b[idx0, idx1, idx_t].pow(2).item() * unit.angstrom
                        gamma = self.gamma[idx0, idx1, idx_t].item() * unit.angstrom
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
                idx_t = re.search(r"t\^(\d+)", expression).group(1)
                idx_t = int(idx_t)
                _mapping = self.mapping[idx_t]
                for idx_bond in _mapping:
                    idx0, idx1, idx_t = _mapping[idx_bond]
                    k = self.k[idx0, idx1, idx_t].item()
                    b = self.b[idx0, idx1, idx_t].pow(2).item()
                    gamma = self.gamma[idx0, idx1, idx_t].item()
                    k = k * unit.kilocalories_per_mole
                    b = b * unit.angstrom
                    gamma = gamma * unit.angstrom
                    force.setBondParameters(idx_bond, idx0, idx1, [b, k, gamma])
                force.updateParametersInContext(context)
        return self
            
            
    def integrate(self, *args, **kwargs):
        return integrate(self, *args, **kwargs)
        
                    

    
    

        
        
        
        
        
        
        
        
        
        
        


