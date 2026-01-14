#!/usr/bin/env python3
"""
Passive Test Case 4: Flight in strong crosswind.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from tvc_simulation.rocket import Rocket
from tvc_simulation.environment import Environment
from tvc_simulation.gnc import GNC
from tvc_simulation.simulation import Simulation
from tvc_simulation.visualization import plot_results, print_summary
from tvc_simulation.utils import euler_to_quaternion

def main():
    print("\\n" + "="*60)
    print("PASSIVE TEST 4: STRONG CROSSWIND")
    print("="*60 + "\\n")

    # Same as passive_flight.py for direct comparison
    thrust_time = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1])
    thrust_values = np.array([0.0, 25.0, 22.0, 18.0, 15.0, 14.0, 13.5, 13.0, 12.5, 12.0, 10.0, 7.0, 3.0, 0.0])

    rocket_config = {
        'mass_dry': 0.150,
        'mass_fuel': 0.025,
        'length': 0.5,
        'diameter': 0.04,
        'cd': 0.5,
        'cp_position': 0.4,
        'motor_position': 0.4,
        'motor_length': 0.07,
        'grain_outer_radius': 0.013,
        'grain_inner_radius_initial': 0.004,
        'grain_height': 0.07,
        'thrust_curve_time': thrust_time.tolist(),
        'thrust_curve_thrust': thrust_values.tolist(),
        'isp': 120,
        'gimbal_position': 0.45,
        'max_gimbal_angle_deg': 1.0, # Not used
    }

    environment_config = {
        'wind_reference_speed': 15.0, # Strong 15 m/s wind
        'wind_direction': np.radians(90), # From the East
    }

    # Disable TVC
    gnc_config = { 'kp_pitch': 0.0, 'ki_pitch': 0.0, 'kd_pitch': 0.0, 'kp_yaw': 0.0, 'ki_yaw': 0.0, 'kd_yaw': 0.0, 'control_start_time': 999 }

    rocket = Rocket(rocket_config)
    environment = Environment(environment_config)
    gnc = GNC(gnc_config)
    simulation = Simulation(rocket, environment, gnc)

    initial_quaternion = euler_to_quaternion(0.0, np.radians(-90.0), 0.0) # Vertical launch

    simulation.set_initial_state([0,0,0], [0,0,0], initial_quaternion, [0,0,0])

    results = simulation.run(t_span=(0.0, 30.0))
    history = simulation.get_history()
    print_summary(history)

if __name__ == '__main__':
    sys.exit(main())
