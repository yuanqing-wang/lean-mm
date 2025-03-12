import openmm as mm
from openmm import unit
from openmmtools.integrators import ThermostatedIntegrator
from openmmtools.testsystems import HarmonicOscillator

class OverdampedLangevinIntegrator(ThermostatedIntegrator):
    """Overdamped Langevin. """
    def __init__(
        self,
        temperature, 
        friction, 
        stepsize,
        use_force_group=False,
    ):
        super().__init__(temperature, stepsize)
        # variable definitions
        self.addGlobalVariable("gamma", friction)
        self.addPerDofVariable("w", 0)
        self.addPerDofVariable("epsilon", 0)
        self.addPerDofVariable("x_old", 0)

        # propagation
        self.addUpdateContextState()
        self.addComputeTemperatureDependentConstants({"epsilon": "dt/gamma/m"})
        self.addComputePerDof("w", "gaussian")

        if use_force_group:
            # position update
            self.addComputePerDof(
                "x", "x+epsilon*f0 + sqrt(2*epsilon*kT)*w"
            )
        else:
            # position update
            self.addComputePerDof(
                "x", "x+epsilon*f + sqrt(2*epsilon*kT)*w"
            )
        
        self.addComputePerDof("v", "(x - x_old) / dt")
        self.addConstrainVelocities()
        self.addConstrainPositions()
        
def run():
    harmonic_oscillator = HarmonicOscillator()
    system, positions = harmonic_oscillator.system, harmonic_oscillator.positions
    for force in system.getForces():
        force.setForceGroup(0)
    
    integrator = OverdampedLangevinIntegrator(
        temperature=300 * unit.kelvin,
        friction=1.0 / unit.picoseconds,
        stepsize=0.01 * unit.femtoseconds,
        use_force_group=True,
    )
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions(positions)
    integrator.step(10000)
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    print(energy)
    
    
if __name__ == "__main__":
    run()