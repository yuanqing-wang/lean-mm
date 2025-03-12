from math import gamma
import openmm as mm
from openmm import unit
import numpy as np
import torch

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
        num_offsets: int=8,
        max_distance: float=10.0,
        num_orders: int=8,
        gamma: float=1.0,
    ):
        super().__init__()
        self.n = n
        self.offset = torch.nn.Parameter(
            torch.distributions.Uniform(0, max_distance).sample(
                (n, num_offsets, num_orders),
            )
        )
        
        self.gamma = torch.nn.Parameter(
            gamma * (1.0 + torch.normal(
                mean=0.0,
                std=1.0,
                size=(n, num_offsets, num_orders),
            ))
        )
        
        self.coefficient = torch.nn.Parameter(
            torch.normal(
                mean=0.0,
                std=1.0,
                size=(n, num_offsets, num_orders),
            )
        )
        
        self.num_offsets = num_offsets
        self.num_orders = num_orders
        
    def forward(
        self,
        x: torch.Tensor,
        t: float,
    ):
        # expand t
        # (orders, )
        t = torch.stack(
            [
                torch.sin(t * 2 * torch.pi * idx)
                for idx in range(self.num_orders)
            ]
        )
        
        # (N, N, 1)
        x = torch.norm(x, dim=-1, keepdim=True).unsqueeze(-1)
        
        # (N, N, num_offsets, num_orders)
        offset = self.offset.unsqueeze(0) + self.offset.unsqueeze(1)
        x = x - offset
        
        # (N, N, num_offsets, num_orders)
        gamma = self.gamma.unsqueeze(0) * self.gamma.unsqueeze(1)
        x = torch.exp(-x**2 * gamma**2)
        
        # (N, N, num_offsets, num_orders)
        x = x * t
        
        # (N, N, num_offsets, num_orders)
        coefficient = self.coefficient.unsqueeze(0) + self.coefficient.unsqueeze(1)
        x = coefficient * x
        
        return x.sum(-1).sum(-1).sum(-1).sum(-1)
        
    def force(self):
        expression = ""
        for idx_offset in range(self.num_offsets):
            for idx_order in range(self.num_orders):
                k = f"(k_{idx_offset}_{idx_order}_1 + k_{idx_offset}_{idx_order}_2)"
                b = f"(b_{idx_offset}_{idx_order}_1 + b_{idx_offset}_{idx_order}_2)"
                gamma = f"(gamma_{idx_offset}_{idx_order}_1 * gamma_{idx_offset}_{idx_order}_2)"
                t = f"sin(2*pi*{idx_order}*t)"
                expression += f"{k} * exp(-(r - {b})^2 * {gamma}^2) * {t} + "
        expression = expression[:-3]
                
        force = mm.CustomNonbondedForce(expression)
        force.addGlobalParameter("t", 1.0)
        force.addGlobalParameter("pi", np.pi)
        force.setCutoffDistance(5.0 * unit.angstrom)
        for idx_offset in range(self.num_offsets):
            for idx_order in range(self.num_orders):
                force.addPerParticleParameter(f"k_{idx_offset}_{idx_order}_")
                force.addPerParticleParameter(f"b_{idx_offset}_{idx_order}_")
                force.addPerParticleParameter(f"gamma_{idx_offset}_{idx_order}_")
        for idx_particle in range(self.n):
            force.addParticle([])
        force.setForceGroup(2)
        return force
    
    def parametrize(self, force):
        for idx_particle in range(self.n):
            parameters = []
            for idx_offset in range(self.num_offsets):
                for idx_order in range(self.num_orders):
                    parameters.append(self.coefficient[idx_particle, idx_offset, idx_order].item())
                    parameters.append(self.offset[idx_particle, idx_offset, idx_order].item())
                    parameters.append(self.gamma[idx_particle, idx_offset, idx_order].item())
            force.setParticleParameters(idx_particle, parameters)
        return force
                    
    
    def parametrize_system(self, system):
        force = self.force()
        force = self.parametrize(force)
        system.addForce(force)
        return force

    
    

        
        
        
        
        
        
        
        
        
        
        


