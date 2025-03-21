import torch
import openmm as mm
from openmm import unit
from openmmtools.integrators import DummyIntegrator

def test_single():
    from lean.forces import InductiveRadialUnbiasingForce
    force = InductiveRadialUnbiasingForce(2)
    x = torch.randn(2, 3)
    t = torch.tensor(1.0)
    f = force(x, t)
    assert f.shape == torch.Size([])
    
def test_batch():
    from lean.forces import InductiveRadialUnbiasingForce
    force = InductiveRadialUnbiasingForce(2)
    x = torch.randn(4, 2, 3)
    t = torch.ones(4)
    f = torch.vmap(force)(x, t)
    assert f.shape == torch.Size([4])
    
def test_consistenty():
    from lean.forces import InductiveRadialUnbiasingForce
    NUM_PARTICLES = 10
    force = InductiveRadialUnbiasingForce(NUM_PARTICLES, num_time=1)
    
    system = mm.System()
    for _ in range(NUM_PARTICLES):
        system.addParticle(1.0 * unit.dalton)
    for force in system.getForces():
        force.setForceGroup(0)
    
    x = torch.randn(NUM_PARTICLES, 3)
    t = torch.randn(size=())
    
    u_torch = force(x, t)
    force.add_force(system)
    context = mm.Context(system, DummyIntegrator(), mm.Platform.getPlatformByName('Reference'))
    u_openmm = force.forward_openmm(x, t, system, context)
    
    assert torch.isclose(u_torch, torch.tensor(u_openmm), atol=1e-5)

def test_objective_unit():
    from lean import unit
    df_dt = 1.0 * unit.ENERGY / unit.TIME
    df_dx = 1.0 * unit.ENERGY / unit.LENGTH
    epsilon = 1.0 * unit.TIME / unit.MASS
    df_dt = epsilon * df_dt
    df_dx = epsilon * df_dx
    loss = df_dt + df_dx ** 2
    
def test_drift_unit():
    from lean import unit
    x = 1.0 * unit.LENGTH
    epsilon = 1.0 * unit.TIME / unit.MASS
    dt = 1.0 * unit.TIME
    r = x ** 2 / (epsilon * dt)
    print(r)
    
if __name__ == "__main__":
    test_drift_unit()