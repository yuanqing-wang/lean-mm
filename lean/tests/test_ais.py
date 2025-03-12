import openmm as mm
from openmm import unit
from openmmtools.testsystems import HarmonicOscillator
from openmmtools.integrators import DummyIntegrator
from openmmtools.states import ThermodynamicState
from lean.processes import annealed_importance_sampling as ais

def test_ais():    
    def get_ais_energy():
        harmonic_oscillator = HarmonicOscillator()
        system, positions = harmonic_oscillator.system, harmonic_oscillator.positions
        positions = ais(system)
        # get the potential
        integrator = DummyIntegrator()
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
        context.setPositions(positions)
        state = context.getState(getEnergy=True)
        potential = state.getPotentialEnergy()
        return potential

    # get the mean potential
    harmonic_oscillator = HarmonicOscillator()
    system, positions = harmonic_oscillator.system, harmonic_oscillator.positions
    state = ThermodynamicState(temperature=300 * unit.kelvin, system=system)
    energies = [get_ais_energy() for _ in range(100)]
    print(energies)
    mean_hat = sum([x._value for x in energies]) / len(energies) * energies[0].unit
    mean = harmonic_oscillator.get_potential_expectation(state)
    print(mean_hat, mean)
    
    
if __name__ == "__main__":
    test_ais()