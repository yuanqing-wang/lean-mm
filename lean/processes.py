import openmm as mm
from openmm import unit
import numpy as np
from .forces import gaussian_force_factory
from .integrators import AnnealedImportanceSamplingOverdampedLangevinIntegrator, OverdampedLangevinIntegrator

def annealed_importance_sampling(
        system,
        temperature=300 * unit.kelvin,
        friction=1.0 / unit.picoseconds,
        stepsize=1.0 * unit.femtoseconds,
        steps=1000,
        sigma=1.0 * unit.angstrom,
    ):
    # set the force group
    for force in system.getForces():
            force.setForceGroup(0)
    
    # add the force
    gaussian_force_factory(system)
    
    # construct the integrator
    integrator = AnnealedImportanceSamplingOverdampedLangevinIntegrator(
        temperature=temperature,
        friction=friction,
        stepsize=stepsize,
    )
    
    # create the context
    context = mm.Context(
        system, 
        integrator, 
        mm.Platform.getPlatformByName('Reference')
    )
    
    # sample gaussian position
    # positions = np.random.normal(size=(system.getNumParticles(), 3)) * sigma
    positions = np.zeros((system.getNumParticles(), 3)) * sigma
    
    # set the positions
    context.setPositions(positions)
    
    # run the simulation
    for idx in range(steps):
        t = float(idx) / steps
        integrator.t = t
        integrator.step(1)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        print(energy)
        
    # get the positions
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True)
    return positions
    
    
    
    
    
    
    
    
    