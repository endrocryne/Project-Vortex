#!/usr/bin/env python3
"""
Example: Windy conditions test.

This example demonstrates rocket behavior in strong wind conditions.
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
    print("\n" + "="*60)
    print("WINDY CONDITIONS SIMULATION")
    print("="*60 + "\n")
    
    # Use standard F15 motor
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
        'max_gimbal_angle_deg': 5.0,
    }
    
    # Strong wind conditions
    environment_config = {
        'gravity': 9.80665,
        'rho_0': 1.225,
        'wind_reference_speed': 10.0,  # Strong 10 m/s wind
        'wind_reference_altitude': 10.0,
        'wind_roughness_length': 0.03,
        'wind_direction': np.radians(90),  # East wind
        'enable_turbulence': True,
        'turbulence_intensity': 0.25,  # 25% turbulence
        'random_seed': 123,
    }
    
    gnc_config = {
        'kp_pitch': 4.0,  # Higher gains for wind
        'ki_pitch': 0.3,
        'kd_pitch': 1.5,
        'kp_yaw': 4.0,
        'ki_yaw': 0.3,
        'kd_yaw': 1.5,
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
    initial_quaternion = euler_to_quaternion(0.0, np.radians(-90.0), 0.0)
    initial_angular_velocity = [0.0, 0.0, 0.0]
    
    simulation.set_initial_state(
        initial_position, initial_velocity,
        initial_quaternion, initial_angular_velocity
    )
    
    print("Running simulation with strong wind...")
    print("Wind speed: 10 m/s from East")
    print("Turbulence: 25%\n")
    
    results = simulation.run(t_span=(0.0, 20.0), max_step=0.01)
    
    if results['success']:
        print("Simulation completed!")
        history = simulation.get_history()
        print_summary(history)
        plot_results(history, save_path='windy_conditions_results.png')
        print("Results saved to 'windy_conditions_results.png'")
        
        # Additional wind drift analysis
        final_pos = history['position'][-1]
        print(f"\nWind Drift Analysis:")
        print(f"  Horizontal displacement: {np.sqrt(final_pos[0]**2 + final_pos[1]**2):.2f} m")
        print(f"  East drift: {final_pos[1]:.2f} m")
    else:
        print(f"Simulation failed: {results['message']}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
