from turtle import position
from lean.forces import InductiveRadialBiasingForce
from lean.integrators import OverdampedLangevinIntegrator
import openmm as mm
from openmm import unit
from openmmtools.testsystems import AlanineDipeptideVacuum

def test_unbiasing():
    adlp = AlanineDipeptideVacuum()
    system, positions = adlp.system, adlp.positions
    force = InductiveRadialBiasingForce(n=system.getNumParticles())
    # force.parametrize_system(system)
    
    integrator = OverdampedLangevinIntegrator(
        temperature=300 * unit.kelvin,
        friction=1.0 / unit.picoseconds,
        stepsize=0.01 * unit.femtoseconds,
    )
    
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions(positions)
    import time
    start = time.time()
    integrator.step(1000)
    end = time.time()
    print(end - start)
    
    
if __name__ == "__main__":
    test_unbiasing()
    