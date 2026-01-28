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


def print_banner():
    banner = r"""
 _____ _______   _______ _   _______ _   _  _____ _____ _____ _____ 
|_   _|  ___\ \ / /_   _| | / /_   _| \ | ||  ___|_   _|_   _/  __ \
  | | | |__  \ V /  | | | |/ /  | | |  \| || |__   | |   | | | /  \/
  | | |  __| /   \  | | |    \  | | | . ` ||  __|  | |   | | | |    
  | | | |___/ /^\ \ | | | |\  \_| |_| |\  || |___  | |  _| |_| \__/\
  \_/ \____/\/   \/ \_/ \_| \_/\___/\_| \_/\____/  \_/  \___/ \____/
                                                                    
                                                           
                                                                                        
         !
         !
         ^
        / \
       /___\
      |=   =|
      |  T  |
      |  E  |
      |  X  |
      |  T  |
      |     |
      |  K  |
      |  I  |
      |  N  |
      |  E  |
      |  T  |
      |  I  |
      |  C  |
      |_____|
     /|##!##|\
    / |##!##| \
   /  |##!##|  \
  | /  /   \  \ |
  |/  /     \  \|
      -------
     /   X   \
    /    X    \
   /     X     \
"""
    # Use cyan for banner, yellow for header
    print("\033[96m" + banner + "\033[0m")
    print("\033[93m" + "  >>> VORTEX DESKTOP SUITE - TEXTKINETIC v0.4.3 <<<" + "\033[0m")
    print("\033[92m" + "  >>> INITIALIZING 6DOF FLIGHT PARAMETERS...      <<<" + "\033[0m")
    print("-" * 60 + "\n")


def run_single_simulation(config_file=None, sim_overrides=None):
    """Run a single simulation with parameters"""
    
    # Default configuration
    rocket_config = {
        'dry_mass': 50.0,
        'propellant_mass': 10.0,
        'length': 5.0,
        'diameter': 0.3,
        'use_dynamic_inertia': False,
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
    
    # Apply overrides
    if sim_overrides:
        sim_config.update(sim_overrides)
    
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
    initial_altitude = env_config.get('initial_altitude', 1000.0)
    initial_velocity = env_config.get('initial_velocity', -50.0)
    
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
    
    # Feasibility Check
    print("-" * 40)
    print("FEASIBILITY CHECK")
    is_possible, r = sim.check_feasibility(initial_velocity, initial_altitude)
    print(f"Status: {'POSSIBLE' if is_possible else 'IMPOSSIBLE'}")
    print(f"Impact Speed (No Burn): {r['v_impact_unpowered']:.1f} m/s")
    print(f"Delta-V Capacity:       {r['dv_capacity']:.1f} m/s (Gross: {r['dv_gross']:.1f}, Gravity Loss: {r['dv_gravity_loss']:.1f})")
    print(f"Margin:                 {r['margin']:.1f} m/s")
    print(f"Max TWR:                {r['max_twr']:.2f}")
    
    if not is_possible:
        print("\nWARNING: Landing appears physically impossible with current configuration.")
    print("-" * 40)
    
    # Run simulation
    print("\nRunning simulation...")
    # Passing ignition_altitude=None allows the simulation to calculate it 
    # dynamically based on apogee if simulate_ascent is True.
    success, final_state, history = sim.run_simulation(initial_state, None)
    
    # Ignition altitude used (can be retrieved from history)
    actual_ignition_altitude = history.get('ignition_altitude', 0.0)
    print(f"Calculated ignition altitude: {actual_ignition_altitude:.2f} m")
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success: {success}")
    print(f"Final altitude: {history['final_altitude']:.3f} m")
    print(f"Final velocity: {history['final_velocity']:.3f} m/s")
    print(f"Final vertical velocity: {history['vz'][-1]:.3f} m/s")
    print(f"Simulation time: {history['t'][-1]:.3f} s")
    
    # Results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'single_run_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            "rocket": rocket_config,
            "environment": env_config,
            "simulation": sim_config
        }, f, indent=2)
    
    # Save CSV
    import csv
    filename = os.path.join(results_dir, 'single_run.csv')
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'QW', 'QX', 'QY', 'QZ', 'Mass'])
        for i in range(len(history['t'])):
            writer.writerow([
                history['t'][i],
                history['x'][i],
                history['y'][i],
                history['z'][i],
                history['vx'][i],
                history['vy'][i],
                history['vz'][i],
                history['qw'][i],
                history['qx'][i],
                history['qy'][i],
                history['qz'][i],
                history['mass'][i]
            ])
    
    print(f"\nResults saved to folder: {results_dir}")
    print(f"Trajectory saved to: {filename}")
    print(f"Configuration saved to: {config_path}")
    
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
    plot_file = os.path.join(results_dir, 'trajectory_2d.png')
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
    
    plot_file = os.path.join(results_dir, 'trajectory_3d.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"3D plot saved to: {plot_file}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


def run_optimization(config_file=None, sim_overrides=None):
    """Run Monte Carlo optimization"""
    
    # Default configuration
    rocket_config = {
        'dry_mass': 50.0,
        'propellant_mass': 10.0,
        'length': 5.0,
        'diameter': 0.3,
        'use_dynamic_inertia': False,
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
    
    # Apply overrides
    if sim_overrides:
        sim_config.update(sim_overrides)
    
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
    initial_altitude = env_config.get('initial_altitude', 1000.0)
    initial_velocity = env_config.get('initial_velocity', -50.0)
    
    initial_state = np.array([
        0, 0, initial_altitude,
        0, 0, initial_velocity,
        1, 0, 0, 0,
        0, 0, 0,
        rocket_config['dry_mass'] + rocket_config['propellant_mass']
    ])
    
    print(f"\nInitial altitude: {initial_altitude:.2f} m")
    print(f"Initial velocity: {initial_velocity:.2f} m/s")
    
    # Get parameters from config or use defaults
    num_mc = sim_config.get('num_monte_carlo', 100)
    search_range = sim_config.get('altitude_search_range', 10.0)
    alt_step = sim_config.get('altitude_step', 0.1)
    
    # Calculate number of altitudes to check
    num_altitudes = int(2 * search_range / alt_step) + 1
    
    print(f"Monte Carlo runs per altitude: {num_mc}")
    print(f"Search range: ±{search_range} m")
    print(f"Step size: {alt_step} m")
    print(f"Total altitudes to check: {num_altitudes}")
    print(f"Total simulations: {num_mc * num_altitudes}")
    print("\nRunning optimization (this may take a few minutes)...\n")
    
    # Run optimization
    optimal_altitude, success_rates, best_history = sim.optimize_ignition_altitude(
        initial_state,
        num_monte_carlo=num_mc,
        altitude_search_range=search_range,
        altitude_step=alt_step
    )
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Optimal ignition altitude: {optimal_altitude:.2f} m")
    print(f"Success rate at optimal altitude: {success_rates[optimal_altitude]*100:.1f}%")
    
    # Results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'optimization_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            "rocket": rocket_config,
            "environment": env_config,
            "simulation": sim_config
        }, f, indent=2)

    # Save results CSV
    import csv
    filename = os.path.join(results_dir, 'optimization.csv')
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ignition Altitude (m)', 'Success Rate'])
        for altitude, rate in sorted(success_rates.items()):
            writer.writerow([altitude, rate])
    
    print(f"\nResults saved to folder: {results_dir}")
    print(f"Optimization results saved to: {filename}")
    print(f"Configuration saved to: {config_path}")
    
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
    
    plot_file = os.path.join(results_dir, 'success_rate.png')
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
        plot_file = os.path.join(results_dir, 'best_trajectory_2d.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Best trajectory 2D plots saved to: {plot_file}")
        plt.close()
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)


def ensure_configs_dir():
    from pathlib import Path
    Path('configs').mkdir(exist_ok=True)


def _input(prompt_text, default=None):
    try:
        if default is not None:
            val = input(f"{prompt_text} [{default}]: ")
            return val if val != "" else default
        return input(f"{prompt_text}: ")
    except EOFError:
        return default


def _parse_value(input_str, current_value):
    if input_str is None:
        return current_value
    # Keep type of current value when possible
    try:
        if isinstance(current_value, bool):
            lowered = str(input_str).strip().lower()
            return lowered in ('1', 'true', 'y', 'yes')
        if isinstance(current_value, int):
            return int(input_str)
        if isinstance(current_value, float):
            return float(input_str)
        if isinstance(current_value, (list, dict)):
            import json
            return json.loads(input_str)
    except Exception:
        pass
    return input_str


def load_config_file(path):
    import json
    if not path:
        return None
    if not os.path.exists(path):
        print(f"Config file not found: {path}")
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_config_file(path, config):
    import json
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {path}")


def show_config_summary(config):
    print("\nCurrent Configuration:\n")
    for section in ('rocket', 'environment', 'simulation'):
        print(f"[{section}]")
        for k, v in sorted(config.get(section, {}).items()):
            print(f"  {k}: {v}")
        print("")


def edit_section_interactive(config, section_name):
    section = config.setdefault(section_name, {})
    keys = sorted(section.keys())
    while True:
        print(f"\nEditing section: {section_name}")
        for i, k in enumerate(keys, 1):
            print(f"{i}) {k} = {section[k]}")
        print("a) Add new key")
        print("b) Back")
        choice = _input("Choose an entry to edit", "b")
        if choice == 'b':
            break
        if choice == 'a':
            new_key = _input("Enter new key name")
            if not new_key:
                continue
            new_val = _input("Enter value (as JSON for lists/dicts) or raw string")
            section[new_key] = _parse_value(new_val, new_val)
            keys = sorted(section.keys())
            continue
        try:
            idx = int(choice) - 1
            key = keys[idx]
        except Exception:
            print("Invalid choice")
            continue
        cur = section[key]
        new_val = _input(f"New value for {key}", str(cur))
        section[key] = _parse_value(new_val, cur)


def interactive_menu(initial_config=None, initial_overrides=None):
    ensure_configs_dir()
    config = initial_config or {
        'rocket': {},
        'environment': {},
        'simulation': {}
    }
    overrides = initial_overrides or {}

    while True:
        print('\n' + '=' * 60)
        print('VORTEX CLI - Interactive Mode')
        print('1) Show current configuration')
        print('2) Edit rocket parameters')
        print('3) Edit environment parameters')
        print('4) Edit simulation parameters')
        print('5) Load config from file')
        print('6) Save config to file')
        print('7) Run single simulation')
        print('8) Run optimization')
        print('9) Set overrides (ascent/initial attitude / MC params)')
        print('0) Exit')
        choice = _input('Select an option', '0')

        if choice == '1':
            show_config_summary(config)
        elif choice == '2':
            edit_section_interactive(config, 'rocket')
        elif choice == '3':
            edit_section_interactive(config, 'environment')
        elif choice == '4':
            edit_section_interactive(config, 'simulation')
        elif choice == '5':
            path = _input('Config file path', '')
            data = load_config_file(path)
            if data:
                config.setdefault('rocket', {}).update(data.get('rocket', {}))
                config.setdefault('environment', {}).update(data.get('environment', {}))
                config.setdefault('simulation', {}).update(data.get('simulation', {}))
                print('Loaded config and merged into current configuration')
        elif choice == '6':
            name = _input('Filename to save (will be placed in configs/)', 'my_config.json')
            path = os.path.join('configs', name)
            save_config_file(path, config)
        elif choice == '7':
            # write temp config and run single
            import tempfile, json
            tmp_path = os.path.join('configs', f'temp_run_single_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            save_config_file(tmp_path, config)
            print('\nStarting single simulation... (use Ctrl+C to cancel)')
            try:
                run_single_simulation(tmp_path, overrides)
            except KeyboardInterrupt:
                print('\nSimulation interrupted by user')
        elif choice == '8':
            tmp_path = os.path.join('configs', f'temp_run_opt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            save_config_file(tmp_path, config)
            print('\nStarting optimization... (this may take a while; use Ctrl+C to cancel)')
            try:
                run_optimization(tmp_path, overrides)
            except KeyboardInterrupt:
                print('\nOptimization interrupted by user')
        elif choice == '9':
            print('\nCurrent overrides:')
            for k, v in overrides.items():
                print(f"  {k}: {v}")
            k = _input('Override key to set (e.g. num_monte_carlo, altitude_search_range) or blank to go back', '')
            if k:
                v = _input('Value')
                overrides[k] = _parse_value(v, overrides.get(k))
                print('Override set')
        elif choice == '0':
            print('Exiting interactive CLI')
            break
        else:
            print('Invalid selection')


def main():
    print_banner()
    parser = argparse.ArgumentParser(description='Suicide Burn Flight Dynamics Simulation')
    parser.add_argument('--mode', choices=['single', 'optimize'], default='single',
                        help='Run mode: single simulation or optimization')
    parser.add_argument('--config', type=str, help='Configuration file (JSON)')
    parser.add_argument('--mc-runs', type=int, help='Monte Carlo runs per altitude (default: 100)')
    parser.add_argument('--search-range', type=float, help='Altitude search range in meters (default: 10.0)')
    parser.add_argument('--altitude-step', type=float, help='Altitude step size in meters (default: 0.1)')
    parser.add_argument('--auto', action='store_true', help='Run immediately without interactive menu (backwards compatible)')

    # New arguments
    parser.add_argument('--ascent', action='store_true', help='Simulate full ascent phase')
    parser.add_argument('--pitch', type=float, default=0.0, help='Initial Pitch (deg). If ascent, applies to launch. If descent only, applies to start.')
    parser.add_argument('--yaw', type=float, default=0.0, help='Initial Yaw (deg)')
    parser.add_argument('--roll', type=float, default=0.0, help='Initial Roll (deg)')
    parser.add_argument('--ignition-percent-offset', type=float, default=0.0, help='Ignition altitude percent offset (percent)')
    parser.add_argument('--ignition-hard-offset', type=float, default=0.0, help='Ignition altitude hard offset (m)')

    args = parser.parse_args()

    # Prepare config overrides
    sim_overrides = {}
    if args.mc_runs is not None:
        sim_overrides['num_monte_carlo'] = args.mc_runs
    if args.search_range is not None:
        sim_overrides['altitude_search_range'] = args.search_range
    if args.altitude_step is not None:
        sim_overrides['altitude_step'] = args.altitude_step

    sim_overrides['ignition_percent_offset'] = args.ignition_percent_offset / 100.0
    sim_overrides['ignition_hard_offset'] = args.ignition_hard_offset

    # Handle Ascent/Orientation Logic
    if args.ascent:
        sim_overrides['simulate_ascent'] = True
        sim_overrides['ascent_initial_pitch'] = args.pitch
        sim_overrides['ascent_initial_yaw'] = args.yaw
        sim_overrides['ascent_initial_roll'] = args.roll
    elif args.pitch != 0.0 or args.yaw != 0.0 or args.roll != 0.0:
        # If any orientation flag is provided but no --ascent, assume descent mode
        sim_overrides['simulate_ascent'] = False
        sim_overrides['descent_initial_pitch'] = args.pitch
        sim_overrides['descent_initial_yaw'] = args.yaw
        sim_overrides['descent_initial_roll'] = args.roll

    # If --auto is provided, preserve previous behavior and run immediately
    if args.auto:
        if args.mode == 'single':
            run_single_simulation(args.config, sim_overrides)
        elif args.mode == 'optimize':
            run_optimization(args.config, sim_overrides)
        return

    # Otherwise launch interactive CLI
    initial_config = None
    if args.config:
        initial_config = load_config_file(args.config)
    interactive_menu(initial_config or {}, sim_overrides)


if __name__ == '__main__':
    main()
