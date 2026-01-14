#!/usr/bin/env python3
"""
Passive Test Case 2: Lightweight rocket with a high-thrust motor.
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
    print("PASSIVE TEST 2: LIGHTWEIGHT ROCKET, HIGH THRUST")
    print("="*60 + "\\n")

    # High thrust motor (F-class)
    thrust_time = np.array([0.0, 0.1, 0.5, 1.5, 2.5, 2.6])
    thrust_values = np.array([0.0, 60.0, 55.0, 40.0, 10.0, 0.0])

    rocket_config = {
        'mass_dry': 0.200,        # 200g dry mass
        'mass_fuel': 0.050,       # 50g propellant
        'length': 0.6,
        'diameter': 0.04,
        'cd': 0.45,
        'cp_position': 0.48,
        'motor_position': 0.48,
        'motor_length': 0.09,
        'grain_outer_radius': 0.015,
        'grain_inner_radius_initial': 0.005,
        'grain_height': 0.09,
        'thrust_curve_time': thrust_time.tolist(),
        'thrust_curve_thrust': thrust_values.tolist(),
        'isp': 150,
        'gimbal_position': 0.52,
        'max_gimbal_angle_deg': 1.0, # Not used
    }

    environment_config = { 'wind_reference_speed': 2.0 } # Light wind

    # Disable TVC
    gnc_config = { 'kp_pitch': 0.0, 'ki_pitch': 0.0, 'kd_pitch': 0.0, 'kp_yaw': 0.0, 'ki_yaw': 0.0, 'kd_yaw': 0.0, 'control_start_time': 999 }

    rocket = Rocket(rocket_config)
    environment = Environment(environment_config)
    gnc = GNC(gnc_config)
    simulation = Simulation(rocket, environment, gnc)

    initial_quaternion = euler_to_quaternion(0.0, np.radians(-89.0), 0.0) # 1 deg tilt

    simulation.set_initial_state([0,0,0], [0,0,0], initial_quaternion, [0,0,0])

    results = simulation.run(t_span=(0.0, 40.0))
    history = simulation.get_history()
    print_summary(history)

if __name__ == '__main__':
    sys.exit(main())
