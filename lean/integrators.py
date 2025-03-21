from openmm import unit
from openmmtools.integrators import ThermostatedIntegrator
import torch
import numpy as np
from .unit import LENGTH
from .forces import GROUPS

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
        self.addComputeTemperatureDependentConstants({"epsilon": "1/(gamma*m)"})
        self.addComputePerDof("w", "gaussian")
        
        # compute total force
        self._F()

        # position update
        self.addComputePerDof(
            "x", "x+epsilon*F*dt + sqrt(2*epsilon*kT*dt)*w"
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
            f"f{GROUPS.SYSTEM}*_t"
        )
        
        self.addComputePerDof(
            "F", 
            f"F+f{GROUPS.GAUSSIAN}*(1-_t)"
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
        self.addComputePerDof(
            "_F", 
            f"f{GROUPS.SYSTEM}*_t"
        )
        
        self.addComputePerDof(
            "_F", 
            f"_F+f{GROUPS.GAUSSIAN}*(1-_t)"
        )
        
        self.addComputePerDof(
            "F",
            f"_F+f{GROUPS.UNBIASING}"
        )
        
    def _forward(self):
        self.addGlobalVariable("A", 0)
        self.addGlobalVariable("_t", 1)
        self.addPerDofVariable("_F", 0)
        
        self.addPerDofVariable("x_old", 0)
        self.addComputePerDof("x_old", "x")
        
        self.addGlobalVariable("energy_old", 0)
        self.addComputeGlobal("energy_old", f"energy{GROUPS.SYSTEM}*_t")
        self.addComputeGlobal("energy_old", f"energy_old+energy{GROUPS.GAUSSIAN}*(1-_t)")
        
        super()._forward()
        
        self.addGlobalVariable("energy_new", 0)
        self.addComputeGlobal("energy_new", f"energy{GROUPS.SYSTEM}*_t")
        self.addComputeGlobal("energy_new", f"energy_new+energy{GROUPS.GAUSSIAN}*(1-_t)")
        
        scale = "1/(4*epsilon*kT*dt)"
        delta_x = "(x-x_old)"
        original_drift = "dt*epsilon*_F"
        new_drift = f"dt*epsilon*f{GROUPS.UNBIASING}"
        r_plus = f"{scale}*({delta_x}-{original_drift}-{new_drift})^2"
        r_minus = f"{scale}*(-{delta_x}-{original_drift}+{new_drift})^2"
        
        self.addComputeSum("r_plus", r_plus)
        self.addComputeSum("r_minus", r_minus)
        
        # delta_a = "(energy_old/kT)-(energy_new/kT)+r_plus-r_minus"
        delta_a = "r_plus-r_minus"
        self.addComputeGlobal("A", f"{delta_a}")
        
    @property
    def A(self):
        return self.getGlobalVariableByName("A")

    @A.setter
    def A(self, value):
        self.setGlobalVariableByName("A", value)
        
    @property
    def t(self):
        return self.getGlobalVariableByName("_t")
    
    @t.setter
    def t(self, value):
        self.setGlobalVariableByName("_t", value)
        
def integrate(
    force,
    system,
    context,
    integrator,
    steps,
    collect,
    T: float=1.0,
):
    # set positions
    positions = np.random.randn(system.getNumParticles(), 3) * LENGTH
    context.setPositions(positions)
    
    # parametrize the system
    force.parametrize(system, context)
    
    # run the simulation
    xs = []
    ts = []
    weights = []
    
    integrator.A = 0.0
    for t in range(steps):
        _t = T * float(t) / steps
        context.setParameter("t", _t)
        integrator.t = _t
        integrator.step(1)
        if t in collect:
            x = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(LENGTH)
            xs.append(x)
            ts.append(_t)
            weights.append(integrator.A)
    return torch.tensor(np.array(xs)), torch.tensor(np.array(ts)), torch.tensor(np.array(weights))
        