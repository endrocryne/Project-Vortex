
import numpy as np
import json
import os
import csv
import sys
from simulation import SuicideBurnSimulation

def run_verify():
    rocket_config = {
        'dry_mass': 50.0,
        'propellant_mass': 20.0,
        'length': 5.0,
        'diameter': 0.3,
        'use_dynamic_inertia': True,
        'thrust_curve': [[0, 0], [0.1, 2000], [4.0, 2000], [4.1, 0]],
        'burn_time': 4.0,
        'tvc_max_angle': 10.0,
        'tvc_response_time': 0.05,
        'tvc_kp_pitch': 2.0,
        'tvc_ki_pitch': 0.5,
        'tvc_kd_pitch': 1.0,
        'tvc_kp_yaw': 2.0,
        'tvc_ki_yaw': 0.5,
        'tvc_kd_yaw': 1.0,
        'rcs_thrust': 200.0, # Reduced
        'rcs_arm': 2.3,
        'cp_z': -1.2 # Less extreme
    }

    env_config = {
        'gravity': 9.81,
        'air_density': 1.225,
        'temperature': 288.15,
        'drag_coefficient': 0.5,
        'reference_area': np.pi * (0.3 / 2) ** 2,
        'wind_model': 'constant',
        'wind_speed': 0.0,
        'wind_direction': 0.0
    }

    sim_config = {
        'altimeter_error': 0.0,
        'velocity_sensor_error': 0.0,
        'simulate_ascent': False,
        'descent_initial_pitch': 5.0,
        'descent_initial_yaw': 0.0,
        'descent_initial_roll': 0.0
    }

    sim = SuicideBurnSimulation(rocket_config, env_config, sim_config)

    # Start at 800m
    initial_state = np.array([
        0, 0, 800.0,
        0, 0, 0,
        1, 0, 0, 0,
        0, 0, 0,
        rocket_config['dry_mass'] + rocket_config['propellant_mass']
    ])

    print("Starting simulation...")
    success, final_state, history = sim.run_simulation(initial_state, ignition_altitude=150.0, max_time=30.0)
    print("Simulation finished.")

    print(f"Success: {success}")
    print(f"Final Alt: {history['final_altitude']:.2f}")
    print(f"Final Vel: {history['final_velocity']:.2f}")

    filename = 'visual_verify.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'QW', 'QX', 'QY', 'QZ', 'Mass', 'MLC', 'FT', 'FM', 'WX', 'WY', 'PH', 'RCS_PP', 'RCS_PN', 'RCS_YP', 'RCS_YN'])

        for i in range(len(history['t'])):
            thrust = history['thrust'][i]
            rcs_active = (history['rcs_p_pos'][i] > 0 or history['rcs_p_neg'][i] > 0 or
                          history['rcs_y_pos'][i] > 0 or history['rcs_y_neg'][i] > 0)

            phase = 1 # Freefall
            if thrust > 10:
                phase = 3 # Descent
            elif rcs_active:
                phase = 2 # Flip

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
                history['mass'][i],
                0.0, 0, 0.0,
                history['wx'][i],
                history['wy'][i],
                phase,
                history['rcs_p_pos'][i],
                history['rcs_p_neg'][i],
                history['rcs_y_pos'][i],
                history['rcs_y_neg'][i]
            ])
    print(f"Saved {filename}")

if __name__ == '__main__':
    run_verify()
