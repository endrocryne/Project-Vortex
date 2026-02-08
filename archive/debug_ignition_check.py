import json
import numpy as np
from simulation import SuicideBurnSimulation
from pathlib import Path

cfg_path = Path('results/single_run_20260128_163835/config.json')
cfg = json.loads(cfg_path.read_text())
rocket = cfg['rocket']
env = cfg['environment']
simc = cfg['simulation']

sim = SuicideBurnSimulation(rocket, env, simc)

initial_altitude = 1000.0
initial_velocity = -50.0
initial_state = np.array([0,0,initial_altitude, 0,0,initial_velocity, 1,0,0,0, 0,0,0, rocket['dry_mass'] + rocket['propellant_mass']])

# Align with run_simulation's pre-processing
initial_mass = initial_state[13]
initial_cg_body = sim.calculate_dynamic_cg(initial_mass)
z_cg_offset_local = initial_cg_body - sim.fuel_tank_bottom
current_sim_state = initial_state.copy()
current_sim_state[0:3] = initial_state[0:3] + np.array([0,0,z_cg_offset_local])
current_alt = current_sim_state[2]
current_vel = current_sim_state[5]

print('initial_mass', initial_mass)
print('initial_cg_body', initial_cg_body)
print('z_cg_offset_local', z_cg_offset_local)
print('current_alt (CG)', current_alt)
print('current_vel', current_vel)

ign = sim.calculate_ignition_altitude(current_vel, current_alt)
print('calculate_ignition_altitude returned (nozzle altitude m)', ign)

# Recompute internal values used in function
v_e = sim.motor.total_impulse / sim.motor.propellant_mass
t_burn = sim.motor.burn_time
g = sim.physics.g
m0 = sim.initial_mass
mf = sim.initial_mass - sim.motor.propellant_mass
rho = sim.physics.get_air_density(current_alt/2)
v_term = (2 * m0 * g / (rho * sim.physics.Cd * sim.physics.A_ref))**0.5
v_impact_sq = v_term**2 * (1 - np.exp(-2 * g * (ign) / v_term**2)) + current_vel**2
v_impact = np.sqrt(max(0, v_impact_sq))

print('v_e', v_e, 't_burn', t_burn, 'g', g)
print('m0', m0, 'mf', mf)
print('v_term', v_term)
print('v_impact (est from ignition point)', v_impact)
print('dv_max', (v_e * np.log(m0/mf)) - (g * t_burn))

# Also print ignition_altitude_sensed variability
import numpy as np
sensed = [ign * (1.0 + np.random.uniform(-sim.altimeter_error, sim.altimeter_error)) for _ in range(5)]
print('sample ignition_altitude_sensed', sensed)

# If ignition altitude is near zero, maybe function returned nozzle altitude instead of cg-based

# Evaluate ignition_event for a sample state near expected ignition height
from physics_engine import PhysicsEngine
R_mat = sim.physics.quaternion_to_rotation_matrix(current_sim_state[6:10])
off_local = sim.calculate_dynamic_cg(current_sim_state[13]) - sim.fuel_tank_bottom
pos_n = current_sim_state[0:3] - R_mat @ np.array([0,0,off_local])
print('initial nozzle position', pos_n)

# Compute time to hit ignition altitude assuming constant g (rough)
if ign > 0:
    # Compute time to fall from pos_n[2] to ign under gravity ignoring drag and initial vel
    v0 = current_sim_state[5]
    h = pos_n[2] - ign
    # Solve 0.5*g*t^2 + v0*t - h = 0
    a = 0.5 * g
    b = v0
    c = -h
    disc = b*b - 4*a*c
    if disc >= 0:
        t_to_ign = (-b + np.sqrt(disc)) / (2*a)
        t_to_ign2 = (-b - np.sqrt(disc)) / (2*a)
        print('approx time to reach ignition altitude (s):', t_to_ign, t_to_ign2)

print('done')
