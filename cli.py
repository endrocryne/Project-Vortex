#!/usr/bin/env python3
"""
Command-line interface for the suicide burn simulation
Use this if GUI is not available
"""

import numpy as np
import argparse
import json
import os
from datetime import datetime

from simulation import SuicideBurnSimulation


def run_single_simulation(config_file=None):
    """Run a single simulation with parameters"""
    
    # Default configuration
    rocket_config = {
        'dry_mass': 50.0,
        'propellant_mass': 10.0,
        'length': 5.0,
        'diameter': 0.3,
        'thrust_curve': [[0, 0], [0.1, 1000], [3.0, 1000], [3.1, 0]],
        'burn_time': 3.0,
        'tvc_max_angle': 5.0,
        'tvc_response_time': 0.1,
        'tvc_kp': 0.5,
        'tvc_kd': 0.1,
        'thrust_variation': 0.0,
        'tvc_response_variation': 0.0,
        'mass_variation': 0.0
    }
    
    env_config = {
        'gravity': 9.81,
        'air_density': 1.225,
        'temperature': 288.15,
        'drag_coefficient': 0.5,
        'reference_area': np.pi * (0.3 / 2) ** 2,
        'wind_model': 'constant',
        'wind_speed': 5.0,
        'wind_direction': 0.0,
        'drag_variation': 0.0,
        'air_density_variation': 0.0
    }
    
    sim_config = {
        'altimeter_error': 0.0,
        'velocity_sensor_error': 0.0
    }
    
    # Load config if provided
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            rocket_config.update(config.get('rocket', {}))
            env_config.update(config.get('environment', {}))
            sim_config.update(config.get('simulation', {}))
    
    print("=" * 60)
    print("SUICIDE BURN SIMULATION - SINGLE RUN")
    print("=" * 60)
    
    # Create simulation
    sim = SuicideBurnSimulation(rocket_config, env_config, sim_config)
    
    # Initial conditions
    initial_altitude = 1000.0
    initial_velocity = -50.0
    
    initial_state = np.array([
        0, 0, initial_altitude,
        0, 0, initial_velocity,
        1, 0, 0, 0,
        0, 0, 0,
        rocket_config['dry_mass'] + rocket_config['propellant_mass']
    ])
    
    # Calculate ignition altitude
    print(f"\nInitial altitude: {initial_altitude:.2f} m")
    print(f"Initial velocity: {initial_velocity:.2f} m/s")
    
    ignition_altitude = sim.calculate_ignition_altitude(initial_velocity, initial_altitude)
    print(f"Calculated ignition altitude: {ignition_altitude:.2f} m")
    
    # Run simulation
    print("\nRunning simulation...")
    success, final_state, history = sim.run_simulation(initial_state, ignition_altitude)
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success: {success}")
    print(f"Final altitude: {history['final_altitude']:.3f} m")
    print(f"Final velocity: {history['final_velocity']:.3f} m/s")
    print(f"Final vertical velocity: {history['vz'][-1]:.3f} m/s")
    print(f"Simulation time: {history['t'][-1]:.3f} s")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    import csv
    filename = f'results/single_run_{timestamp}.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'Mass'])
        for i in range(len(history['t'])):
            writer.writerow([
                history['t'][i],
                history['x'][i],
                history['y'][i],
                history['z'][i],
                history['vx'][i],
                history['vy'][i],
                history['vz'][i],
                history['mass'][i]
            ])
    
    print(f"\nTrajectory saved to: {filename}")
    
    # Generate plots
    print("\nGenerating plots...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    t = history['t']
    
    # 2D plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(t, history['z'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Altitude (m)')
    axes[0, 0].set_title('Altitude vs Time')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(t, history['vz'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Vertical Velocity (m/s)')
    axes[0, 1].set_title('Vertical Velocity vs Time')
    axes[0, 1].grid(True)
    
    speed = np.sqrt(history['vx']**2 + history['vy']**2 + history['vz']**2)
    axes[1, 0].plot(t, speed, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Speed (m/s)')
    axes[1, 0].set_title('Total Speed vs Time')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(t, history['mass'], 'k-', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Mass (kg)')
    axes[1, 1].set_title('Mass vs Time')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_file = f'results/trajectory_2d_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"2D plots saved to: {plot_file}")
    plt.close()
    
    # 3D plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(history['x'], history['y'], history['z'], 'b-', linewidth=2)
    ax.scatter([history['x'][0]], [history['y'][0]], [history['z'][0]], 
              c='g', s=100, marker='o', label='Start')
    ax.scatter([history['x'][-1]], [history['y'][-1]], [history['z'][-1]], 
              c='r', s=100, marker='x', label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('3D Trajectory')
    ax.legend()
    
    plot_file = f'results/trajectory_3d_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"3D plot saved to: {plot_file}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


def run_optimization(config_file=None):
    """Run Monte Carlo optimization"""
    
    # Default configuration
    rocket_config = {
        'dry_mass': 50.0,
        'propellant_mass': 10.0,
        'length': 5.0,
        'diameter': 0.3,
        'thrust_curve': [[0, 0], [0.1, 1000], [3.0, 1000], [3.1, 0]],
        'burn_time': 3.0,
        'tvc_max_angle': 5.0,
        'tvc_response_time': 0.1,
        'tvc_kp': 0.5,
        'tvc_kd': 0.1,
        'thrust_variation': 0.05,
        'tvc_response_variation': 0.1,
        'mass_variation': 0.02
    }
    
    env_config = {
        'gravity': 9.81,
        'air_density': 1.225,
        'temperature': 288.15,
        'drag_coefficient': 0.5,
        'reference_area': np.pi * (0.3 / 2) ** 2,
        'wind_model': 'altitude_varying',
        'wind_speed': 5.0,
        'wind_direction': 0.0,
        'drag_variation': 0.1,
        'air_density_variation': 0.05
    }
    
    sim_config = {
        'altimeter_error': 0.01,
        'velocity_sensor_error': 0.01
    }
    
    # Load config if provided
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            rocket_config.update(config.get('rocket', {}))
            env_config.update(config.get('environment', {}))
            sim_config.update(config.get('simulation', {}))
    
    print("=" * 60)
    print("SUICIDE BURN SIMULATION - MONTE CARLO OPTIMIZATION")
    print("=" * 60)
    
    # Create simulation
    sim = SuicideBurnSimulation(rocket_config, env_config, sim_config)
    
    # Initial conditions
    initial_altitude = 1000.0
    initial_velocity = -50.0
    
    initial_state = np.array([
        0, 0, initial_altitude,
        0, 0, initial_velocity,
        1, 0, 0, 0,
        0, 0, 0,
        rocket_config['dry_mass'] + rocket_config['propellant_mass']
    ])
    
    print(f"\nInitial altitude: {initial_altitude:.2f} m")
    print(f"Initial velocity: {initial_velocity:.2f} m/s")
    print(f"Monte Carlo runs per altitude: 100")
    print(f"Search range: ±10 m")
    print(f"Step size: 0.1 m")
    print("\nRunning optimization (this may take a few minutes)...\n")
    
    # Run optimization
    optimal_altitude, success_rates, best_history = sim.optimize_ignition_altitude(
        initial_state,
        num_monte_carlo=100,
        altitude_search_range=10.0,
        altitude_step=0.1
    )
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Optimal ignition altitude: {optimal_altitude:.2f} m")
    print(f"Success rate at optimal altitude: {success_rates[optimal_altitude]*100:.1f}%")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    import csv
    filename = f'results/optimization_{timestamp}.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ignition Altitude (m)', 'Success Rate'])
        for altitude, rate in sorted(success_rates.items()):
            writer.writerow([altitude, rate])
    
    print(f"\nOptimization results saved to: {filename}")
    
    # Generate plots
    print("\nGenerating plots...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    altitudes = sorted(success_rates.keys())
    rates = [success_rates[a] for a in altitudes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(altitudes, rates, 'b-', linewidth=2)
    plt.axvline(optimal_altitude, color='r', linestyle='--', label=f'Optimal: {optimal_altitude:.2f} m')
    plt.xlabel('Ignition Altitude (m)')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs Ignition Altitude')
    plt.grid(True)
    plt.legend()
    
    plot_file = f'results/success_rate_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Success rate plot saved to: {plot_file}")
    plt.close()
    
    # Generate trajectory plots if we have a successful run
    if best_history:
        t = best_history['t']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(t, best_history['z'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Altitude (m)')
        axes[0, 0].set_title('Best Run: Altitude vs Time')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(t, best_history['vz'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Vertical Velocity (m/s)')
        axes[0, 1].set_title('Best Run: Vertical Velocity vs Time')
        axes[0, 1].grid(True)
        
        speed = np.sqrt(best_history['vx']**2 + best_history['vy']**2 + best_history['vz']**2)
        axes[1, 0].plot(t, speed, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Speed (m/s)')
        axes[1, 0].set_title('Best Run: Total Speed vs Time')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(t, best_history['mass'], 'k-', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Mass (kg)')
        axes[1, 1].set_title('Best Run: Mass vs Time')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_file = f'results/best_trajectory_2d_{timestamp}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Best trajectory 2D plots saved to: {plot_file}")
        plt.close()
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Suicide Burn Flight Dynamics Simulation')
    parser.add_argument('--mode', choices=['single', 'optimize'], default='single',
                        help='Run mode: single simulation or optimization')
    parser.add_argument('--config', type=str, help='Configuration file (JSON)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        run_single_simulation(args.config)
    elif args.mode == 'optimize':
        run_optimization(args.config)


if __name__ == '__main__':
    main()
