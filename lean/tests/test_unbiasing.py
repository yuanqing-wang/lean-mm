from turtle import position
from lean.forces import InductiveRadialBiasingForce
from lean.integrators import (
    AnnealedImportanceSamplingOverdampedLangevinIntegrator,
    NonEquilibriumAnnealedImportanceSamplingOverdampedLangevinIntegrator,
)
import openmm as mm
from openmm import unit
from openmmtools.testsystems import AlanineDipeptideVacuum

def test_unbiasing():
    adlp = AlanineDipeptideVacuum()
    system, positions = adlp.system, adlp.positions
    force = InductiveRadialBiasingForce(n=system.getNumParticles())    
    integrator = NonEquilibriumAnnealedImportanceSamplingOverdampedLangevinIntegrator(
        temperature=300 * unit.kelvin,
        friction=1.0 / unit.picoseconds,
        stepsize=0.01 * unit.femtoseconds,
    )
    
    force.add_force(system)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions(positions)
    force.parametrize(system, context)
    
    xs = []
    import time
    start = time.time()
    for t in range(10000):
        t = float(t) / 10000.0
        context.setParameter('t', t)
        integrator.step(1)
        # x = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        # xs.append(x)
        
    end = time.time()
    print(f"Time taken: {end - start}")
    print(integrator.A)
    
    
    
if __name__ == "__main__":
    test_unbiasing()
    