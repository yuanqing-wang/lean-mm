import sys
import torch
import openmm as mm
from lean.forces import InductiveRadialBiasingForce
from lean.integrators import NonEquilibriumAnnealedImportanceSamplingOverdampedLangevinIntegrator as Integrator
from lean.parallel import loop_integrate

def dw_force(k2=-2.0, k4=0.45, d0=4.0):
    force = mm.CustomBondForce("k2*(r-d0)^2+k4*(r-d0)^4")
    force.addPerBondParameter("k2")
    force.addPerBondParameter("k4")
    force.addPerBondParameter("d0")
    return force

def dw2():
    system = mm.System()
    force = dw_force()
    system.addParticle(1.0)
    system.addParticle(1.0)
    force.addBond(0, 1, [-2.0, 0.45, 4.0])
    system.addForce(force)
    return system

def run():
    system = dw2()
    force = InductiveRadialBiasingForce(2)
    force.add_force(system)
    integrator = Integrator(
        temperature=300 * mm.unit.kelvin,
        friction=1.0 / mm.unit.picoseconds,
        stepsize=0.01 * mm.unit.femtoseconds,
    )
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    
    x, t, a = loop_integrate(
        n=4,
        force=force,
        system=system,
        context=context,
        integrator=integrator,
        steps=10000,
        num_samples=100,
    )
    
    
    

    
    
if __name__ == "__main__":
    run()