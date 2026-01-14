#!/usr/bin/env python3
"""
Example: Custom rocket flight.

This example uses a custom-defined rocket to test the simulation's robustness.
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

def create_custom_thrust_curve():
    """
    Create a custom motor thrust curve (e.g., a small E-class motor).
    """
    time = np.array([0.0, 0.05, 0.2, 0.5, 1.0, 1.5, 1.8, 2.0, 2.1])
    thrust = np.array([0.0, 40.0, 38.0, 35.0, 30.0, 25.0, 15.0, 5.0, 0.0])
    return time, thrust

def main():
    print("\\n" + "="*60)
    print("CUSTOM ROCKET SIMULATION")
    print("="*60 + "\\n")

    thrust_time, thrust_values = create_custom_thrust_curve()

    rocket_config = {
        'mass_dry': 0.250,        # 250g dry mass
        'mass_fuel': 0.040,       # 40g propellant
        'length': 0.6,            # 60cm rocket
        'diameter': 0.045,        # 45mm diameter
        'cd': 0.48,
        'cp_position': 0.45,
        'motor_position': 0.45,
        'motor_length': 0.08,
        'grain_outer_radius': 0.015,
        'grain_inner_radius_initial': 0.005,
        'grain_height': 0.08,
        'thrust_curve_time': thrust_time.tolist(),
        'thrust_curve_thrust': thrust_values.tolist(),
        'isp': 130,
        'gimbal_position': 0.50,
        'max_gimbal_angle_deg': 5.0,
    }

    environment_config = {
        'gravity': 9.80665,
        'rho_0': 1.225,
        'wind_reference_speed': 2.0,  # Light wind
        'wind_reference_altitude': 10.0,
        'wind_roughness_length': 0.03,
        'wind_direction': np.radians(180), # South wind
        'enable_turbulence': False,
    }

    gnc_config = {
        'kp_pitch': 2.0,  # Moderate gains
        'ki_pitch': 0.1,
        'kd_pitch': 1.0,
        'kp_yaw': 2.0,
        'ki_yaw': 0.1,
        'kd_yaw': 1.0,
        'max_gimbal_angle_deg': 5.0,
        'control_start_time': 0.1,
    }

    rocket = Rocket(rocket_config)
    environment = Environment(environment_config)
    gnc = GNC(gnc_config)
    simulation = Simulation(rocket, environment, gnc)

    # Launch with a small tilt
    initial_position = [0.0, 0.0, 0.0]
    initial_velocity = [0.0, 0.0, 0.0]
    initial_quaternion = euler_to_quaternion(0.0, np.radians(-90.0 + 2.0), 0.0)
    initial_angular_velocity = [0.0, 0.0, 0.0]

    simulation.set_initial_state(
        initial_position, initial_velocity,
        initial_quaternion, initial_angular_velocity
    )

    print("Running custom rocket simulation...")
    results = simulation.run(t_span=(0.0, 30.0), max_step=0.01)

    if results['success']:
        print("Simulation completed!")
        history = simulation.get_history()
        print_summary(history)
        plot_results(history, save_path='custom_flight_results.png')
        print("Results saved to 'custom_flight_results.png'")
    else:
        print(f"Simulation failed: {results['message']}")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
