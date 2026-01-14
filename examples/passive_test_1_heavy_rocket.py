#!/usr/bin/env python3
"""
Passive Test Case 1: Heavy rocket with a low-power motor.
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
    print("PASSIVE TEST 1: HEAVY ROCKET, LOW POWER")
    print("="*60 + "\\n")

    # Low power motor (A-class)
    thrust_time = np.array([0.0, 0.1, 0.5, 0.8, 0.9])
    thrust_values = np.array([0.0, 5.0, 4.5, 2.0, 0.0])

    rocket_config = {
        'mass_dry': 0.500,        # 500g dry mass
        'mass_fuel': 0.010,       # 10g propellant
        'length': 0.7,
        'diameter': 0.06,
        'cd': 0.6,
        'cp_position': 0.5,
        'motor_position': 0.5,
        'motor_length': 0.05,
        'grain_outer_radius': 0.01,
        'grain_inner_radius_initial': 0.003,
        'grain_height': 0.05,
        'thrust_curve_time': thrust_time.tolist(),
        'thrust_curve_thrust': thrust_values.tolist(),
        'isp': 100,
        'gimbal_position': 0.55,
        'max_gimbal_angle_deg': 1.0, # Not used
    }

    environment_config = { 'wind_reference_speed': 1.0 } # Minimal wind

    # Disable TVC
    gnc_config = { 'kp_pitch': 0.0, 'ki_pitch': 0.0, 'kd_pitch': 0.0, 'kp_yaw': 0.0, 'ki_yaw': 0.0, 'kd_yaw': 0.0, 'control_start_time': 999 }

    rocket = Rocket(rocket_config)
    environment = Environment(environment_config)
    gnc = GNC(gnc_config)
    simulation = Simulation(rocket, environment, gnc)

    initial_quaternion = euler_to_quaternion(0.0, np.radians(-88.0), 0.0) # 2 deg tilt

    simulation.set_initial_state([0,0,0], [0,0,0], initial_quaternion, [0,0,0])

    results = simulation.run(t_span=(0.0, 30.0))
    history = simulation.get_history()
    print_summary(history)

if __name__ == '__main__':
    sys.exit(main())
