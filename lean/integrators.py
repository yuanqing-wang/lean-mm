from openmmtools.integrators import ThermostatedIntegrator

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
        self.addGlobalVariable("t", 1)
        self._forward()

    def _F(self):
        self.addComputePerDof(
            "F", 
            "f0*t"
        )
        
        self.addComputePerDof(
            "F", 
            "F+f1*(1-t)"
        )
                
    @property
    def t(self):
        return self.getGlobalVariableByName("t")
    
    @t.setter
    def t(self, value):
        self.setGlobalVariableByName("t", value)
        
class NonEquilibriumAnnealedImportanceSamplingOverdampedLangevinIntegrator(ThermostatedIntegrator):
    """NonEquilibrium Annealed Importance Sampling Langevin. """
    
    def _F(self):
        self.addComputePerDof(
            "F", 
            "f0*t"
        )
        
        self.addComputePerDof(
            "F", 
            "F+f1*(1-t)"
        )
        
        self.addComputePerDof(
            "F", 
            "F+f2"
        )
    
    