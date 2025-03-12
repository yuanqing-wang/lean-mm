import openmm as mm
from openmm import unit
from openmmtools.testsystems import HarmonicOscillator
from openmmtools.states import ThermodynamicState
from lean.integrators import OverdampedLangevinIntegrator

def test_overdamped():
    harmonic_oscillator = HarmonicOscillator()
    system, positions = harmonic_oscillator.system, harmonic_oscillator.positions
    integrator = OverdampedLangevinIntegrator(
        temperature=300 * unit.kelvin,
        friction=1.0 / unit.picoseconds,
        stepsize=0.01 * unit.femtoseconds,
    )
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('CPU'))
    context.setPositions(positions)
    integrator.step(1000)
    
    # get the mean potential 
    potentials = []
    for _ in range(1000):
        integrator.step(10)
        state = context.getState(getEnergy=True)
        potentials.append(state.getPotentialEnergy())
                
    mean_hat = (sum([x._value for x in potentials]) / len(potentials)) * potentials[0].unit
    std_hat = (sum([(x._value - mean_hat._value)**2 for x in potentials]) / len(potentials))**0.5 * potentials[0].unit
    
    thermodynamic_state = ThermodynamicState(temperature=300 * unit.kelvin, system=system)
    mean = harmonic_oscillator.get_potential_expectation(thermodynamic_state)
    std = harmonic_oscillator.get_potential_standard_deviation(thermodynamic_state)
    
    print(mean_hat, mean)
    print(std_hat, std)
    
if __name__ == "__main__":
    test_overdamped()