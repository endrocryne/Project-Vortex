
import numpy as np
from simulation import SuicideBurnSimulation
import os

print("Minimal Feature Test")

rocket_config = {
    'dry_mass': 50.0,
    'propellant_mass': 10.0,
    'length': 5.0,
    'diameter': 0.3,
    'thrust_curve': [[0, 0], [0.1, 1000], [3.0, 1000], [3.1, 0]],
    'burn_time': 3.0,
    'tvc_max_angle': 5.0,
    'tvc_response_time': 0.1,
    'tvc_kp_pitch': 0.5,
    'tvc_ki_pitch': 0.05,
    'tvc_kd_pitch': 0.1,
    'tvc_kp_yaw': 0.5,
    'tvc_ki_yaw': 0.05,
    'tvc_kd_yaw': 0.1,
}

env_config = {
    'gravity': 9.81,
    'air_density': 1.225,
    'temperature': 288.15,
    'drag_coefficient': 0.5,
    'reference_area': 0.07,
    'wind_model': 'constant',
    'wind_speed': 0.0,
    'wind_direction': 0.0
}

sim_config = {'altimeter_error': 0.0, 'velocity_sensor_error': 0.0}

sim = SuicideBurnSimulation(rocket_config, env_config, sim_config)
initial_state = np.array([0, 0, 500.0, 0, 0, -30.0, 1, 0, 0, 0, 0, 0, 0, 60.0])
ignition_alt = sim.calculate_ignition_altitude(-30.0, 500.0)
print(f"Ignition altitude: {ignition_alt:.2f} m")

success, final_state, history = sim.run_simulation(initial_state, ignition_alt)
print(f"Simulation completed. Success: {success}, Final Alt: {history['final_altitude']:.2f}")
