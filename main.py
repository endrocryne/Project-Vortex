#!/usr/bin/env python3
"""
Main script to run 6-DOF TVC rocket simulation.

Simulates a model rocket with thrust vector control using an Estes F15-like motor.
"""

import numpy as np
import sys
from tvc_simulation.rocket import Rocket
from tvc_simulation.environment import Environment
from tvc_simulation.gnc import GNC
from tvc_simulation.simulation import Simulation
from tvc_simulation.visualization import plot_results, print_summary
from tvc_simulation.utils import euler_to_quaternion


def create_estes_f15_thrust_curve():
    """
    Create a mock Estes F15 thrust curve.
    
    The Estes F15 motor characteristics:
    - Average thrust: ~15 N
    - Peak thrust: ~25 N
    - Burn time: ~2 seconds
    - Total impulse: ~30 N·s
    
    Returns:
        Tuple of (time_array, thrust_array)
    """
    # Time points (seconds)
    time = np.array([
        0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1
    ])
    
    # Thrust values (Newtons)
    thrust = np.array([
        0.0, 25.0, 22.0, 18.0, 15.0, 14.0, 13.5, 13.0, 12.5, 12.0, 10.0, 7.0, 3.0, 0.0
    ])
    
    return time, thrust


def main():
    """Main function to set up and run the simulation."""
    
    print("\n" + "="*60)
    print("6-DOF TVC ROCKET SIMULATION")
    print("="*60 + "\n")
    
    # ========== ROCKET CONFIGURATION ==========
    thrust_time, thrust_values = create_estes_f15_thrust_curve()
    
    rocket_config = {
        # Mass properties
        'mass_dry': 0.150,  # kg (150g dry mass)
        'mass_fuel': 0.025,  # kg (25g propellant)
        
        # Geometry
        'length': 0.5,  # m (50cm rocket)
        'diameter': 0.04,  # m (40mm diameter)
        
        # Aerodynamics
        'cd': 0.5,  # Drag coefficient
        'cp_position': 0.4,  # Center of pressure at 40cm from nose
        
        # Motor properties
        'motor_position': 0.4,  # Motor at 40cm from nose
        'motor_length': 0.07,  # 70mm motor length
        
        # Fuel grain (hollow cylinder)
        'grain_outer_radius': 0.013,  # 13mm outer radius
        'grain_inner_radius_initial': 0.004,  # 4mm initial bore
        'grain_height': 0.07,  # 70mm grain height
        
        # Thrust curve
        'thrust_curve_time': thrust_time.tolist(),
        'thrust_curve_thrust': thrust_values.tolist(),
        'isp': 120,  # Specific impulse (seconds)
        
        # TVC
        'gimbal_position': 0.45,  # Gimbal pivot at 45cm from nose
        'max_gimbal_angle_deg': 5.0,  # ±5° gimbal limit
    }
    
    # ========== ENVIRONMENT CONFIGURATION ==========
    environment_config = {
        'gravity': 9.80665,  # m/s²
        'rho_0': 1.225,  # kg/m³ (sea level air density)
        'wind_reference_speed': 3.0,  # m/s (light wind)
        'wind_reference_altitude': 10.0,  # m
        'wind_roughness_length': 0.03,  # m (short grass)
        'wind_direction': np.radians(45),  # 45° from North
        'enable_turbulence': True,
        'turbulence_intensity': 0.15,  # 15% turbulence
        'random_seed': 42,
    }
    
    # ========== GNC CONFIGURATION ==========
    gnc_config = {
        # PID gains (tuned for small model rocket)
        'kp_pitch': 3.0,
        'ki_pitch': 0.2,
        'kd_pitch': 1.0,
        'kp_yaw': 3.0,
        'ki_yaw': 0.2,
        'kd_yaw': 1.0,
        
        'max_gimbal_angle_deg': 5.0,
        'control_start_time': 0.1,  # Start control after 0.1s (allow liftoff)
    }
    
    # ========== CREATE SIMULATION OBJECTS ==========
    print("Initializing simulation components...")
    rocket = Rocket(rocket_config)
    environment = Environment(environment_config)
    gnc = GNC(gnc_config)
    simulation = Simulation(rocket, environment, gnc)
    
    # ========== SET INITIAL CONDITIONS ==========
    # Launch from ground level
    initial_position = [0.0, 0.0, 0.0]  # NED frame (z=0 is ground level)
    initial_velocity = [0.0, 0.0, 0.0]  # m/s
    
    # Initial attitude: rocket standing vertically, pointing up
    # In NED frame: "up" is negative Z
    # Rocket body frame: X points forward (out the nose)
    # So we need to rotate the rocket 90° in pitch to point the nose up
    # Then add small perturbation for testing control
    initial_roll = 0.0
    initial_pitch = np.radians(-90.0 + 2.0)  # -90° to point up, +2° perturbation
    initial_yaw = np.radians(1.0)             # 1° yaw perturbation
    initial_quaternion = euler_to_quaternion(initial_roll, initial_pitch, initial_yaw)
    
    initial_angular_velocity = [0.0, 0.0, 0.0]  # rad/s
    
    simulation.set_initial_state(
        initial_position,
        initial_velocity,
        initial_quaternion,
        initial_angular_velocity
    )
    
    print("Initial conditions set:")
    print(f"  Position: {initial_position}")
    print(f"  Attitude: Roll={np.degrees(initial_roll):.1f}°, "
          f"Pitch={np.degrees(initial_pitch):.1f}°, Yaw={np.degrees(initial_yaw):.1f}°")
    
    # ========== RUN SIMULATION ==========
    print("\nRunning simulation...")
    t_start = 0.0
    t_end = 20.0  # Simulate for 20 seconds (or until ground impact)
    
    results = simulation.run(
        t_span=(t_start, t_end),
        max_step=0.01,  # 10ms time steps
        method='RK45'
    )
    
    if results['success']:
        print(f"Simulation completed successfully!")
        print(f"Message: {results['message']}")
    else:
        print(f"Simulation failed: {results['message']}")
        return 1
    
    # ========== GET RESULTS ==========
    history = simulation.get_history()
    
    # ========== PRINT SUMMARY ==========
    print_summary(history)
    
    # ========== GENERATE PLOTS ==========
    print("Generating plots...")
    fig = plot_results(history, save_path='tvc_simulation_results.png')
    
    print("Plot saved as 'tvc_simulation_results.png'")
    print("\nSimulation complete!")
    
    # Show plot interactively if possible
    try:
        import matplotlib.pyplot as plt
        plt.show()
    except Exception as e:
        print(f"Could not display plot interactively: {e}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
