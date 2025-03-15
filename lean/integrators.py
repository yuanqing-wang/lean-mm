from openmm import unit
from openmmtools.integrators import ThermostatedIntegrator
import torch
import numpy as np

class BaseOverdampedLangevinIntegrator(ThermostatedIntegrator):
    """Overdamped Langevin. """
    def __init__(
        self,
        temperature, 
        friction, 
        stepsize,
    ):
        super().__init__(temperature, stepsize)
        self.addGlobalVariable("gamma", friction)
        
    def _forward(self):        
        self.addPerDofVariable("w", 0)
        self.addPerDofVariable("epsilon", 0)
        self.addPerDofVariable("F", 0)

        # propagation
        self.addUpdateContextState()
        self.addComputeTemperatureDependentConstants({"epsilon": "dt/gamma/m"})
        self.addComputePerDof("w", "gaussian")
        
        # compute total force
        self._F()

        # position update
        self.addComputePerDof(
            "x", "x+epsilon*F + sqrt(2*epsilon*kT)*w"
        )
        
        self.addConstrainPositions()
        
    def _F(self):
        # for the simple case, the total force is just the force
        self.addComputePerDof(
            "F", 
            "f"
        )
                
class OverdampedLangevinIntegrator(BaseOverdampedLangevinIntegrator):
    def __init__(
        self,
        temperature, 
        friction, 
        stepsize,
    ):
        super().__init__(temperature, friction, stepsize)
        self._forward()
        
class AnnealedImportanceSamplingOverdampedLangevinIntegrator(BaseOverdampedLangevinIntegrator):
    """Annealed Importance Sampling Langevin. """
    def __init__(
        self,
        temperature, 
        friction, 
        stepsize,
    ):
        super().__init__(temperature, friction, stepsize)
        self.addGlobalVariable("_t", 1)
        self._forward()

    def _F(self):
        self.addComputePerDof(
            "F", 
            "f0*_t"
        )
        
        self.addComputePerDof(
            "F", 
            "F+f1*(1-_t)"
        )
                
    @property
    def t(self):
        return self.getGlobalVariableByName("_t")
    
    @t.setter
    def t(self, value):
        self.setGlobalVariableByName("_t", value)
        
class NonEquilibriumAnnealedImportanceSamplingOverdampedLangevinIntegrator(BaseOverdampedLangevinIntegrator):
    """NonEquilibrium Annealed Importance Sampling Langevin. """
    def __init__(
        self,
        temperature, 
        friction, 
        stepsize,
    ):
        super().__init__(temperature, friction, stepsize)
        self._forward()
    
    def _F(self):    
        self.addPerDofVariable("_F", 0)
            
        self.addComputePerDof(
            "_F", 
            "f0*_t"
        )
        
        self.addComputePerDof(
            "_F", 
            "F+f1*(1-_t)"
        )
        
        self.addComputePerDof(
            "F", 
            "_F+f2"
        )
        
    def _forward(self):
        self.addGlobalVariable("_t", 1)
        self.addGlobalVariable("A", 0)
        
        self.addPerDofVariable("x_old", 0)
        self.addComputePerDof("x_old", "x")
        
        self.addGlobalVariable("energy_old", 0)
        self.addComputeGlobal("energy_old", "energy0")
        self.addComputeGlobal("energy_old", "energy_old+energy1")
        
        super()._forward()
        
        self.addGlobalVariable("energy_new", 0)
        self.addComputeGlobal("energy_new", "energy0")
        self.addComputeGlobal("energy_new", "energy_new+energy1")
        
        scale = "1/(4*kT*epsilon)"
        delta_x = "x-x_old"
        original_force = "dt*epsilon*_F"
        new_force = "dt*epsilon*f2"
        r_plus = f"{scale}*({delta_x}+{original_force}+{new_force})^2"
        r_minus = f"{scale}*({delta_x}+{original_force}-{new_force})^2"
        delta_a = f"(energy_old/kT)-(energy_new/kT)+{r_plus}-{r_minus}"
        self.addComputeGlobal("A", f"A+{delta_a}")
        
    @property
    def A(self):
        return self.getGlobalVariableByName("A")

    @A.setter
    def A(self, value):
        self.setGlobalVariableByName("A", value)
        
def integrate(
    force,
    system,
    context,
    integrator,
    steps,
    num_samples,
):
    # set positions
    positions = np.random.randn(system.getNumParticles(), 3) * unit.angstrom
    context.setPositions(positions)
    
    # parametrize the system
    force.parametrize(system, context)
    
    # sample the positions to collect
    collect = np.random.choice(
        range(steps),
        num_samples,
        replace=False,
    )
    
    # run the simulation
    xs = []
    ts = []
    weights = []
    
    integrator.A = 0.0
    for t in range(steps):
        _t = float(t) / steps
        context.setParameter('t', _t)
        integrator.step(1)
        if t in collect:
            x = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometers)
            xs.append(x)
            ts.append(_t)
            weights.append(integrator.A)
    
    return torch.tensor(np.array(xs)), torch.tensor(np.array(ts)), torch.tensor(np.array(weights))
        