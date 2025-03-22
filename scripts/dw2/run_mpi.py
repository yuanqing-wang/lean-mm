import sys
import torch
import openmm as mm
from lean.forces import InductiveRadialUnbiasingForce, GaussianForce
from lean.integrators import NonEquilibriumAnnealedImportanceSamplingOverdampedLangevinIntegrator as Integrator
from lean.parallel import mpi_integrate, get_rank, broadcast_objects
from lean.loss import action_matching_loss
from lean import unit

def dw_force(k2=-2.0, k4=0.45, d0=4.0):
    force = mm.CustomBondForce("k2*(r-d0)^2+k4*(r-d0)^4")
    force.addPerBondParameter("k2")
    force.addPerBondParameter("k4")
    force.addPerBondParameter("d0")
    return force

def dw(n=2):
    system = mm.System()
    force = dw_force()
    k2 = -2.0 * unit.ENERGY / unit.LENGTH**2
    k4 = 0.45 * unit.ENERGY / unit.LENGTH**4
    d0 = 4.0 * unit.LENGTH
    for _ in range(n):
        system.addParticle(1.0)
    
    for idx0 in range(n):
        for idx1 in range(n):
            if idx0 < idx1:
                force.addBond(idx0, idx1, [k2, k4, d0])
    system.addForce(force)
    return system

def ess(a):
    return ((a.softmax(0) ** 2).sum()).pow(-1)
    
def run():
    if get_rank() == 0:
        system = dw(2)
        force = InductiveRadialUnbiasingForce(2)
        force.add_force(system)
        gaussian_force = GaussianForce()
        gaussian_force.add_force(system)
        
        friction = 10.0 / mm.unit.picoseconds
        stepsize= 1.0 * mm.unit.femtoseconds
        temperature = 300 * mm.unit.kelvin
        steps = 10000
        
        integrator = Integrator(
            temperature=temperature,
            friction=friction,
            stepsize=stepsize,
        )
        
        mass = [system.getParticleMass(i) for i in range(system.getNumParticles())]
        epsilon = [1 / (mass[i] * friction) for i in range(system.getNumParticles())]
        epsilon = [_epsilon.value_in_unit(unit.TIME/unit.MASS) for _epsilon in epsilon]
        epsilon = torch.tensor(epsilon)
        time_scale = stepsize * steps
        time_scale = time_scale.value_in_unit(unit.TIME)
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    
        optimizer = torch.optim.Adam(force.parameters(), lr=1e-3)
        
    system, context, integrator, steps, epsilon, time_scale \
        = broadcast_objects(
            [system, context, integrator, steps, epsilon, time_scale]
    )
        
    for _ in range(100):
        x, t, a = mpi_integrate(
            n=16,
            force=force,
            system=system,
            context=context,
            integrator=integrator,
            steps=steps,
            num_samples=100,
            T=1,
        )
        
        if get_rank() == 0:
            optimizer.zero_grad()
            
            _loss = action_matching_loss(
                unbiasing_force=force,
                samples=x,
                times=t,
                weights=a,
                epsilon=epsilon,
                time_scale=time_scale,
            )

            _loss.backward()
            optimizer.step()
        
if __name__ == "__main__":
    run()