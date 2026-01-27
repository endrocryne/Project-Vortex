
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation import SuicideBurnSimulation
from physics_engine import PhysicsEngine

def test_ascent_simulation():
    print("\n--- Testing Ascent Simulation ---")
    
    # Config
    rocket_config = {
        'dry_mass': 50.0,
        'propellant_mass': 10.0,
        'ascent_motor_casing_mass': 2.0,
        'length': 5.0,
        'diameter': 0.3,
        'use_dynamic_inertia': False,
        'thrust_curve': [[0, 0], [0.1, 2500], [3.0, 2500], [3.1, 0]],
        'burn_time': 3.0,
    }
    
    env_config = {
        'gravity': 9.81,
        'air_density': 1.225,
        'temperature': 288.15,
        'drag_coefficient': 0.5,
        'reference_area': 0.1,
    }
    
    sim_config = {
        'simulate_ascent': True,
        'ascent_initial_pitch': 5.0, # Launch with slight pitch
        'num_monte_carlo': 1,
    }
    
    sim = SuicideBurnSimulation(rocket_config, env_config, sim_config)
    
    initial_state = np.array([
        0, 0, 0.0,      # x, y, z (Ground)
        0, 0, 0.0,      # vx, vy, vz
        1, 0, 0, 0,     # q (will be overwritten if simulate_ascent is True)
        0, 0, 0,        # omega
        0.0             # mass (overwritten)
    ])
    
    print("Running simulation with Ascent...")
    success, final_state, history = sim.run_simulation(initial_state)
    
    max_alt = np.max(history['z'])
    print(f"Max Altitude: {max_alt:.2f} m")
    
    # Check if we went up
    if max_alt > 100.0:
        print("PASS: Rocket ascended significantly.")
    else:
        print("FAIL: Rocket did not ascend.")
        return False

    # Check mass drop
    # Mass should be high during ascent, then drop at apogee
    mass_start = history['mass'][0]
    mass_end = history['mass'][-1]
    print(f"Start Mass: {mass_start:.2f} kg")
    print(f"End Mass: {mass_end:.2f} kg")
    
    # Expected Start Mass: 50 + 10 + 2 + 10 = 72
    # Expected End Mass: 50
    if mass_start > 70.0 and mass_end <= 50.0:
        print("PASS: Mass modeled correctly (Ascent Stack -> Descent Vehicle).")
    else:
        print("FAIL: Mass modeling incorrect.")
        return False
        
    return True

def test_descent_orientation():
    print("\n--- Testing Descent Orientation Override ---")
    
    # Config
    rocket_config = {
        'dry_mass': 50.0,
        'propellant_mass': 10.0,
        'length': 5.0,
        'diameter': 0.3,
        'use_dynamic_inertia': False,
        'thrust_curve': [[0, 0], [0.1, 1000], [3.0, 1000], [3.1, 0]],
        'burn_time': 3.0,
    }
    
    env_config = {
        'gravity': 9.81,
        'air_density': 1.225,
        'temperature': 288.15,
        'drag_coefficient': 0.5,
        'reference_area': 0.1,
    }
    
    sim_config = {
        'simulate_ascent': False,
        'descent_initial_pitch': 45.0, # Start descent at 45 degrees
        'num_monte_carlo': 1,
    }
    
    sim = SuicideBurnSimulation(rocket_config, env_config, sim_config)
    
    initial_state = np.array([
        0, 0, 1000.0,      # x, y, z
        0, 0, -50.0,       # vx, vy, vz
        1, 0, 0, 0,        # q (default Identity)
        0, 0, 0,           # omega
        60.0               # mass
    ])
    
    print("Running simulation (Descent only, 45 deg pitch)...")
    success, final_state, history = sim.run_simulation(initial_state)
    
    # Check first quaternion
    q0 = np.array([history['qw'][0], history['qx'][0], history['qy'][0], history['qz'][0]])
    
    # Convert back to Euler or check against expected
    pe = PhysicsEngine(env_config)
    q_expected = pe.euler_to_quaternion(0, np.radians(45), 0)
    
    diff = np.linalg.norm(q0 - q_expected)
    print(f"Quaternion Diff: {diff:.6f}")
    
    if diff < 1e-5:
        print("PASS: Orientation applied correctly.")
    else:
        print("FAIL: Orientation mismatch.")
        return False
        
    return True

if __name__ == "__main__":
    p1 = test_ascent_simulation()
    p2 = test_descent_orientation()
    
    if p1 and p2:
        print("\nALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\nTESTS FAILED")
        sys.exit(1)
