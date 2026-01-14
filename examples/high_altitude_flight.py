#!/usr/bin/env python3
"""
Example: High-altitude flight with stronger motor.

This example demonstrates a high-power rocket flight simulation.
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


def create_high_power_thrust_curve():
    """
    Create a high-power motor thrust curve (similar to G-class motor).
    Average thrust: ~80 N, Burn time: ~1.5s
    """
    time = np.array([0.0, 0.05, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.6])
    thrust = np.array([0.0, 120.0, 110.0, 90.0, 80.0, 75.0, 60.0, 30.0, 0.0])
    return time, thrust


def main():
    print("\n" + "="*60)
    print("HIGH-ALTITUDE ROCKET SIMULATION")
    print("="*60 + "\n")
    
    thrust_time, thrust_values = create_high_power_thrust_curve()
    
    rocket_config = {
        'mass_dry': 0.400,  # 400g dry mass (larger rocket)
        'mass_fuel': 0.060,  # 60g propellant
        'length': 0.8,  # 80cm rocket
        'diameter': 0.05,  # 50mm diameter
        'cd': 0.45,
        'cp_position': 0.65,
        'motor_position': 0.65,
        'motor_length': 0.10,
        'grain_outer_radius': 0.018,
        'grain_inner_radius_initial': 0.006,
        'grain_height': 0.10,
        'thrust_curve_time': thrust_time.tolist(),
        'thrust_curve_thrust': thrust_values.tolist(),
        'isp': 140,
        'gimbal_position': 0.70,
        'max_gimbal_angle_deg': 5.0,
    }
    
    environment_config = {
        'gravity': 9.80665,
        'rho_0': 1.225,
        'wind_reference_speed': 2.0,  # Light wind
        'wind_reference_altitude': 10.0,
        'wind_roughness_length': 0.03,
        'wind_direction': np.radians(0),
        'enable_turbulence': False,
    }
    
    gnc_config = {
        'kp_pitch': 2.5,
        'ki_pitch': 0.15,
        'kd_pitch': 1.2,
        'kp_yaw': 2.5,
        'ki_yaw': 0.15,
        'kd_yaw': 1.2,
        'max_gimbal_angle_deg': 5.0,
        'control_start_time': 0.1,
    }
    
    rocket = Rocket(rocket_config)
    environment = Environment(environment_config)
    gnc = GNC(gnc_config)
    simulation = Simulation(rocket, environment, gnc)
    
    # Launch vertically
    initial_position = [0.0, 0.0, 0.0]
    initial_velocity = [0.0, 0.0, 0.0]
    initial_quaternion = euler_to_quaternion(0.0, np.radians(-90.0 + 1.0), 0.0)
    initial_angular_velocity = [0.0, 0.0, 0.0]
    
    simulation.set_initial_state(
        initial_position, initial_velocity,
        initial_quaternion, initial_angular_velocity
    )
    
    print("Running high-altitude simulation...")
    results = simulation.run(t_span=(0.0, 30.0), max_step=0.01)
    
    if results['success']:
        print("Simulation completed!")
        history = simulation.get_history()
        print_summary(history)
        plot_results(history, save_path='high_altitude_results.png')
        print("Results saved to 'high_altitude_results.png'")
    else:
        print(f"Simulation failed: {results['message']}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
