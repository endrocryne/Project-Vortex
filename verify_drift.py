import numpy as np
import os
from simulation import SuicideBurnSimulation

def test_drift_correction():
    # Base rocket config
    rocket_config = {
        'dry_mass': 10.0,
        'propellant_mass': 1.0,
        'length': 5.0,
        'diameter': 0.3,
        'thrust_curve': [[0.0, 0.0], [0.1, 500.0], [3.0, 500.0], [3.1, 0.0]],
        'burn_time': 3.0,
        'tvc_max_angle': 10.0,
        'tvc_response_time': 0.1,
        'tvc_kp_pitch': 0.8,
        'tvc_ki_pitch': 0.1,
        'tvc_kd_pitch': 0.2,
        'tvc_kp_yaw': 0.8,
        'tvc_ki_yaw': 0.1,
        'tvc_kd_yaw': 0.2,
        'tvc_mode': 'orientation',
        'tvc_drift_gain': 0.0
    }
    
    env_config = {
        'gravity': 9.81,
        'air_density': 1.225,
        'temperature': 288.15,
        'drag_coefficient': 0.5,
        'reference_area': 0.07,
        'wind_model': 'constant',
        'wind_speed': 8.0, # Strong North wind (+Y)
        'wind_direction': 0.0 # North
    }
    
    sim_config = {
        'num_monte_carlo': 1,
        'show_plots': False
    }

    # Initial state (200m alt, -30m/s velocity)
    initial_state = np.array([0, 0, 200.0, 0, 0, -30.0, 1, 0, 0, 0, 0, 0, 0, 11.0])
    
    print("Running with Orientation Hold (Traditional)...")
    sim1 = SuicideBurnSimulation(rocket_config.copy(), env_config, sim_config)
    # Fast optimization
    print("Finding optimal altitude (coarse sweep)...")
    best_alt, _, _ = sim1.optimize_ignition_altitude(initial_state, num_monte_carlo=1, altitude_search_range=2.0, altitude_step=0.5)
    
    history1, success1 = sim1.run_simulation(initial_state, best_alt)
    
    final_v1 = history1[-1, 3:6]
    final_speed1 = np.linalg.norm(final_v1)
    print(f"Final Velocity: {final_v1}")
    print(f"Final Total Speed: {final_speed1:.2f} m/s")
    print(f"Landing Status: {'SUCCESS' if success1 else 'FAIL'}")
    
    print("\nRunning with Velocity Hold (Drift Correction)...")
    rc2 = rocket_config.copy()
    rc2['tvc_mode'] = 'velocity'
    # Gain calculation: 0.05 rad (2.8 deg) per 1 m/s of horizontal speed
    rc2['tvc_drift_gain'] = 0.05 
    
    sim2 = SuicideBurnSimulation(rc2, env_config, sim_config)
    history2, success2 = sim2.run_simulation(initial_state, best_alt)
    
    final_v2 = history2[-1, 3:6]
    final_speed2 = np.linalg.norm(final_v2)
    print(f"Final Velocity: {final_v2}")
    print(f"Final Total Speed: {final_speed2:.2f} m/s")
    print(f"Landing Status: {'SUCCESS' if success2 else 'FAIL'}")
    
    if final_speed2 < final_speed1:
        reduction = (final_speed1 - final_speed2) / final_speed1 * 100
        print(f"\nSUCCESS: Drift correction reduced final speed by {reduction:.1f}%!")
        if success2 and final_speed2 < 3.0:
            print("The landing is now SUCCESSFUL thanks to drift correction!")
    else:
        print("\nFAILURE: Drift correction did not reduce final speed.")

if __name__ == "__main__":
    test_drift_correction()
