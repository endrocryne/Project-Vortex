import numpy as np
import matplotlib.pyplot as plt
from rocketpy import Environment, SolidMotor, Rocket, Flight
from rocketpy.control.controller import _Controller
import time
import sys
import tkinter as tk
import csv
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- Monte Carlo Parameters ---
N_RUNS = 1

# The base thrust curve is the mean. STD is the per-point variation.
THRUST_CURVE_STD = 0.05 # 5% Variation

PROPELLANT_MASS_MEAN = 0.0125 # 12.5g for a C-class motor, matching thrust curve
PROPELLANT_MASS_STD_PERCENT = 0.05 # 5% Variation

DRAG_COEFFICIENT_MEAN = 0.6
DRAG_COEFFICIENT_STD_PERCENT = 0.10 # 10% Variation

DRY_MASS_MEAN = 0.060  # 60 grams
DRY_MASS_STD_PERCENT = 0.025 # 2.5% Variation

AIR_DENSITY_MEAN = 1.225
AIR_DENSITY_STD_PERCENT = 0.02 # 2% Variation for atmospheric changes

# Wind (m/s)
GROUND_WIND_SPEED_MEAN = 4.0
GROUND_WIND_SPEED_STD_PERCENT = 0.375 # 37.5% Variation (i.e., 1.5 m/s STD) to represent gusting
# Sensor Noise (rad for gyro, m for barometer)
GYRO_NOISE_STD = 0.005 # rad, approx 0.28 degrees. This is an absolute value.
BAROMETER_NOISE_STD = 0.2 # m. This is an absolute value.
# Actuator Characteristics
ACTUATOR_RESPONSE_TAU_MEAN = 0.075 # Time constant in seconds
ACTUATOR_RESPONSE_TAU_STD_PERCENT = 0.25 # 25% Variation
ACTUATOR_PRECISION_NOISE_STD = 0.001 # rad

# --- Advanced Aerodynamics Variations ---
C_N_ALPHA_MEAN = 2.5 # Normal Force Coefficient derivative (per radian)
C_N_ALPHA_STD_PERCENT = 0.10 # 10% Variation

CP_Z_OFFSET_MEAN = -0.2 # Center of Pressure Z-offset from CoM. MUST be negative for stability.
CP_Z_OFFSET_STD_PERCENT = 0.05 # 5% Variation

# --- Control System & Initial State Variations ---
YAW_INIT_MEAN = 0.1 # Initial yaw disturbance
YAW_INIT_STD_PERCENT = 0.20 # 20% Variation
PITCH_INIT_MEAN = 0.1 # Initial pitch disturbance
PITCH_INIT_STD_PERCENT = 0.20 # 20% Variation

# --- Constants ---
G = 9.80665

# Base thrust curve, to be scaled by random parameter
TIME_POINTS = np.array([
    0.0, 0.01248, 0.014, 0.026, 0.067, 0.099, 0.15, 0.183, 0.207, 0.219,
    0.262, 0.333, 0.349, 0.392, 0.475, 0.653, 0.913, 1.366, 1.607, 1.745,
    1.978, 2.023, 2.024
])
THRUST_POINTS_BASE = np.array([
    0.0, 0.0227, 0.633, 1.533, 2.726, 5.136, 9.103, 11.465, 11.635, 11.391,
    6.377, 5.014, 5.209, 4.722, 4.771, 4.746, 4.673, 4.625, 4.625, 4.868,
    4.795, 0.828, 0.0
])

ENGINE_BURN_TIME = TIME_POINTS[-1]
INERTIA_TENSOR = np.diag([0.007, 0.007, 0.0001])  # Inertia tensor matching rocketpy-sim values
ROCKET_DIAMETER = 0.034  # 34mm diameter (same as rocketpy-sim)
# --- Advanced Aerodynamics ---
GIMBAL_VECTOR = np.array([0, 0, -1.0]) # Location of gimbal pivot relative to CoM
KP, KI, KD = 0.0, 0.0, 0.0 # PID gains are now constant
MAX_GIMBAL_ANGLE = np.deg2rad(10) # Limit gimbal to 10 degrees

# --- PID Controller ---
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self._integral, self._prev_error = 0.0, 0.0
        self.last_output = 0.0

    def update(self, error, dt):
        if dt <= 0: return 0.0
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt
        self._prev_error = error
        self.last_output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        return self.last_output

# --- Grain Geometry (for workaround) ---
# These are fixed values to calculate a volume, so we can derive density from mass.
GRAIN_NUMBER = 1
GRAIN_OUTER_RADIUS = 0.015 # 15mm
GRAIN_INITIAL_INNER_RADIUS = 0.005 # 5mm
GRAIN_HEIGHT = 0.05 # 5cm

def run_simulation_rocketpy(params: dict, full_logs=False) -> dict:
    """
    Runs a single rocket flight simulation using the RocketPy engine.
    """
    # 1. Set up Environment
    env = Environment(
        latitude=32.990254, longitude=-106.974998, elevation=1400,
        date=(2023, 1, 1, 12) # Arbitrary date
    )
    R_SPECIFIC_AIR = 287.058
    STANDARD_TEMP = 288.15
    pressure_pa = params['air_density'] * R_SPECIFIC_AIR * STANDARD_TEMP
    env.set_atmospheric_model(
        type='custom_atmosphere', pressure=pressure_pa, temperature=STANDARD_TEMP,
        wind_u=params['ground_wind_speed'], wind_v=0
    )

    # 2. Set up Motor
    thrust_curve = np.column_stack((TIME_POINTS, params['thrust_curve']))
    grain_volume = GRAIN_NUMBER * np.pi * (GRAIN_OUTER_RADIUS**2 - GRAIN_INITIAL_INNER_RADIUS**2) * GRAIN_HEIGHT
    grain_density = params['propellant_mass'] / grain_volume
    grains_com_position = -GRAIN_HEIGHT / 2
    motor = SolidMotor(
        thrust_source=thrust_curve, dry_mass=0, dry_inertia=(0, 0, 0),
        grain_number=GRAIN_NUMBER, grain_density=grain_density,
        grain_outer_radius=GRAIN_OUTER_RADIUS, grain_initial_inner_radius=GRAIN_INITIAL_INNER_RADIUS,
        grain_initial_height=GRAIN_HEIGHT, grain_separation=0.005,
        grains_center_of_mass_position=grains_com_position, center_of_dry_mass_position=grains_com_position,
        burn_time=ENGINE_BURN_TIME, nozzle_radius=ROCKET_DIAMETER / 4,
        throat_radius=ROCKET_DIAMETER / 8, coordinate_system_orientation='nozzle_to_combustion_chamber'
    )

    # 3. Set up Rocket
    rocket = Rocket(
        radius=ROCKET_DIAMETER / 2, mass=params['dry_mass'], inertia=tuple(INERTIA_TENSOR.flatten()),
        power_off_drag=params['drag_coefficient'], power_on_drag=params['drag_coefficient'],
        center_of_mass_without_motor=0, coordinate_system_orientation='tail_to_nose'
    )
    rocket.add_motor(motor, position=0)
    fins = rocket.add_trapezoidal_fins(
        n=4, root_chord=0.1, tip_chord=0.05, span=0.05,
        position=params['cp_z_offset'], cant_angle=0, airfoil=None,
    )
    fins.cpz_alpha = params['C_N_alpha']

    # 4. Set up TVC Controller
    pid_pitch = PIDController(KP, KI, KD)
    pid_yaw = PIDController(KP, KI, KD)
    actual_gimbal_angles = [0.0, 0.0]

    def tvc_controller(time, sampling_rate, state_vector, state_history, observed_variables, interactive_objects, sensors):
        rocket_obj = interactive_objects
        dt = 1/sampling_rate
        e0, e1, e2, e3 = state_vector[6:10]
        
        # Simplified rotation matrix to get Z-axis components in world frame
        z_axis_world_y = 2 * (e2 * e3 - e0 * e1)
        z_axis_world_x = 2 * (e1 * e3 + e0 * e2)
        
        true_error_pitch = z_axis_world_y
        true_error_yaw = -z_axis_world_x
        
        sensed_error_pitch = true_error_pitch + np.random.normal(0, params['gyro_noise_std'])
        sensed_error_yaw = true_error_yaw + np.random.normal(0, params['gyro_noise_std'])

        gimbal_pitch_cmd = np.clip(-pid_pitch.update(sensed_error_pitch, dt), -MAX_GIMBAL_ANGLE, MAX_GIMBAL_ANGLE)
        gimbal_yaw_cmd = np.clip(-pid_yaw.update(sensed_error_yaw, dt), -MAX_GIMBAL_ANGLE, MAX_GIMBAL_ANGLE)

        tau = params['actuator_response_tau']
        actual_gimbal_angles[0] += (dt / tau) * (gimbal_pitch_cmd - actual_gimbal_angles[0])
        actual_gimbal_angles[1] += (dt / tau) * (gimbal_yaw_cmd - actual_gimbal_angles[1])

        final_pitch = actual_gimbal_angles[0] + np.random.normal(0, params['actuator_precision_noise_std'])
        final_yaw = actual_gimbal_angles[1] + np.random.normal(0, params['actuator_precision_noise_std'])

        rocket_obj.thrust_eccentricity_y = np.tan(final_pitch) * abs(GIMBAL_VECTOR[2])
        rocket_obj.thrust_eccentricity_x = np.tan(final_yaw) * abs(GIMBAL_VECTOR[2])

        # Log everything needed for plots
        return (time, true_error_pitch, true_error_yaw, sensed_error_pitch, sensed_error_yaw, gimbal_pitch_cmd, gimbal_yaw_cmd, actual_gimbal_angles[0], actual_gimbal_angles[1])

    controller = _Controller(
        interactive_objects=rocket, controller_function=tvc_controller, sampling_rate=100,
        initial_observed_variables=(0,0,0,0,0,0,0,0,0)
    )
    rocket._add_controllers(controller)
    
    # 5. Run Flight Simulation
    initial_solution = [0, 0, 0, env.elevation, 0, 0, 0, 1, 0, 0, 0, params['pitch_init'], params['yaw_init'], 0]
    flight = Flight(
        rocket=rocket, environment=env, rail_length=5,
        inclination=90, heading=0, initial_solution=initial_solution
    )

    # 6. Extract Results and Logs
    apogee = flight.apogee - env.elevation
    landing_distance = np.sqrt(flight.x_impact**2 + flight.y_impact**2)
    stable = rocket.static_margin(flight.apogee_time) > 1.5

    logs = {}
    if full_logs:
        controller_data = np.array(controller.observed_variables[1:])
        logs = {
            'time': flight.time,
            'state': flight.solution_array[:, 1:],
            'controller_time': controller_data[:, 0],
            'wind_speed': flight.wind_velocity_y.y_array,
            'error_true': controller_data[:, 1:3],
            'sensed_error': controller_data[:, 3:5],
            'gimbal_cmd': np.rad2deg(controller_data[:, 5:7]),
            'gimbal_actual': np.rad2deg(controller_data[:, 7:9]),
        }

    # Reconstruct the original detailed results dictionary for CSV export and stats
    final_state = flight.impact_state
    final_quat = final_state[6:10]
    z_axis_at_landing = rotate_by_quaternion(np.array([0, 0, 1]), final_quat)
    final_tilt_angle_rad = np.arccos(np.clip(np.dot(z_axis_at_landing, np.array([0, 0, 1])), -1.0, 1.0))

    results = {
        'apogee': apogee,
        'flight_time': flight.t_final,
        'stable': stable,
        'landing_x': flight.x_impact,
        'landing_y': flight.y_impact,
        'landing_distance': landing_distance,
        'landing_vel_x': final_state[3],
        'landing_vel_y': final_state[4],
        'landing_vel_z': final_state[5],
        'landing_tilt_deg': np.rad2deg(final_tilt_angle_rad),
        'logs': logs
    }

    if flight.apogee_time > 0:
        apogee_state = flight.apogee_state
        apogee_vel = apogee_state[3:6]
        apogee_tilt_quat = apogee_state[6:10]
        z_axis_at_apogee = rotate_by_quaternion(np.array([0, 0, 1]), apogee_tilt_quat)
        apogee_tilt_angle = np.arccos(np.clip(np.dot(z_axis_at_apogee, np.array([0, 0, 1])), -1.0, 1.0))
        results.update({
            'apogee_vel_x': apogee_vel[0],
            'apogee_vel_y': apogee_vel[1],
            'apogee_vel_z': apogee_vel[2],
            'apogee_tilt_deg': np.rad2deg(apogee_tilt_angle)
        })

    return results

# Helper Functions for plotting
def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

def rotate_by_quaternion(vector, q):
    return quaternion_to_rotation_matrix(q) @ vector


# --- Main Monte Carlo Loop ---
if __name__ == "__main__":
    all_results = []
    print(f"Starting Monte Carlo simulation with {N_RUNS} runs...")
    start_mc_time = time.time()
    for i in range(N_RUNS):
        params = {
            'thrust_curve': THRUST_POINTS_BASE * np.random.normal(1.0, THRUST_CURVE_STD, len(THRUST_POINTS_BASE)),
            'propellant_mass': np.random.normal(PROPELLANT_MASS_MEAN, PROPELLANT_MASS_MEAN * PROPELLANT_MASS_STD_PERCENT),
            'drag_coefficient': np.random.normal(DRAG_COEFFICIENT_MEAN, DRAG_COEFFICIENT_MEAN * DRAG_COEFFICIENT_STD_PERCENT),
            'dry_mass': np.random.normal(DRY_MASS_MEAN, DRY_MASS_MEAN * DRY_MASS_STD_PERCENT),
            'air_density': np.random.normal(AIR_DENSITY_MEAN, AIR_DENSITY_MEAN * AIR_DENSITY_STD_PERCENT),
            'C_N_alpha': np.random.normal(C_N_ALPHA_MEAN, C_N_ALPHA_MEAN * C_N_ALPHA_STD_PERCENT),
            'cp_z_offset': np.random.normal(CP_Z_OFFSET_MEAN, abs(CP_Z_OFFSET_MEAN) * CP_Z_OFFSET_STD_PERCENT),
            'yaw_init': np.random.normal(YAW_INIT_MEAN, abs(YAW_INIT_MEAN) * YAW_INIT_STD_PERCENT),
            'pitch_init': np.random.normal(PITCH_INIT_MEAN, abs(PITCH_INIT_MEAN) * PITCH_INIT_STD_PERCENT),
            'ground_wind_speed': np.random.normal(GROUND_WIND_SPEED_MEAN, GROUND_WIND_SPEED_MEAN * GROUND_WIND_SPEED_STD_PERCENT),
            'wind_direction': np.random.uniform(0, 360),
            'gyro_noise_std': GYRO_NOISE_STD, 'barometer_noise_std': BAROMETER_NOISE_STD,
            'actuator_response_tau': np.random.normal(ACTUATOR_RESPONSE_TAU_MEAN, ACTUATOR_RESPONSE_TAU_MEAN * ACTUATOR_RESPONSE_TAU_STD_PERCENT),
            'actuator_precision_noise_std': ACTUATOR_PRECISION_NOISE_STD,
        }
        params['thrust_curve'][0] = 0.0; params['thrust_curve'][-1] = 0.0
        all_results.append(run_simulation_rocketpy(params, full_logs=True))
        sys.stdout.write(f'\rRunning simulation {i+1}/{N_RUNS}...'); sys.stdout.flush()

    end_mc_time = time.time()
    if N_RUNS > 0: print(f"\nMonte Carlo finished in {end_mc_time - start_mc_time:.2f}s (Avg: {(end_mc_time - start_mc_time)/N_RUNS:.3f}s/run)")

    if all_results:
        successful_runs = [r for r in all_results if r.get('stable', False)]
        print(f"\n--- Monte Carlo Simulation Results ---\nSuccess Rate: {len(successful_runs)/N_RUNS:.2%}")
        apogees = [r.get('apogee', 0) for r in all_results]
        landing_distances = [r.get('landing_distance', 0) for r in all_results]
        print("\n--- Apogee (m) ---\n" + f"  Mean: {np.mean(apogees):.2f}, Std Dev: {np.std(apogees):.2f}, Min: {np.min(apogees):.2f}, Max: {np.max(apogees):.2f}")
        print("\n--- Landing Distance (m) ---\n" + f"  Mean: {np.mean(landing_distances):.2f}, Std Dev: {np.std(landing_distances):.2f}, Min: {np.min(landing_distances):.2f}, Max: {np.max(landing_distances):.2f}")
        print("------------------------------------")

        # --- Export raw data to CSV ---
        csv_filename = "phase4_monte_carlo_results.csv"
        print(f"\nExporting raw simulation data to {csv_filename}...")
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['run_number', 'stable', 'apogee', 'flight_time', 'landing_distance', 'landing_x', 'landing_y',
                              'landing_vel_x', 'landing_vel_y', 'landing_vel_z', 'landing_tilt_deg', 'apogee_vel_x', 'apogee_vel_y',
                              'apogee_vel_z', 'apogee_tilt_deg']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for i, res in enumerate(all_results):
                    row = res.copy()
                    row['run_number'] = i + 1
                    writer.writerow(row)
            print("Saved raw data successfully.")
        except IOError as e:
            print(f"Error writing to CSV file: {e}")

        # --- Averaging and Plotting ---
        print("\nAveraging all simulation runs for plotting...")
        max_time = max((res['logs']['time'][-1] for res in all_results if res['logs']['time'].size > 0), default=0)
        common_time = np.linspace(0, max_time, 1000)
        
        def interpolate_log(log_key, time_key, num_cols):
            interpolated = []
            for res in all_results:
                log = res['logs']
                if log[time_key].size > 1:
                    data = log[log_key]
                    if len(data.shape) == 1: data = data.reshape(-1, 1)
                    interp_data = np.array([np.interp(common_time, log[time_key], data[:, i]) for i in range(num_cols)]).T
                    interpolated.append(interp_data)
            return np.mean(interpolated, axis=0) if interpolated else np.zeros((len(common_time), num_cols))

        logs = { 'time': common_time, 'state': interpolate_log('state', 'time', 13),
                 'gimbal_cmd': interpolate_log('gimbal_cmd', 'controller_time', 2),
                 'gimbal_actual': interpolate_log('gimbal_actual', 'controller_time', 2),
                 'wind_speed': interpolate_log('wind_speed', 'time', 1),
                 'error_true': interpolate_log('error_true', 'controller_time', 2),
                 'sensed_error': interpolate_log('sensed_error', 'controller_time', 2)}

        # --- GUI and Plotting ---
        root = tk.Tk()
        root.title("Vortex Rocket Simulation v0.5 (RocketPy Engine)")
        notebook = ttk.Notebook(root); notebook.pack(pady=10, padx=10, expand=True, fill="both")
        def create_plot_tab(tab_name, plot_function):
            frame = ttk.Frame(notebook, width=800, height=600)
            notebook.add(frame, text=tab_name)
            fig = Figure(figsize=(10, 7), dpi=100); fig.set_tight_layout(True)
            plot_function(fig)
            canvas = FigureCanvasTkAgg(fig, master=frame); canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        def plot_position(fig):
            ax = fig.add_subplot(111); ax.plot(logs['time'], logs['state'][:,2], label='Altitude')
            ax.set_title("Position vs. Time (Averaged)"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (m)"); ax.legend(); ax.grid(True)
        def plot_velocity(fig):
            ax = fig.add_subplot(111); ax.plot(logs['time'], logs['state'][:,5], label='Vz')
            ax.set_title("Velocity vs. Time (Averaged)"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Velocity (m/s)"); ax.legend(); ax.grid(True)
        def plot_actuator_response(fig):
            ax = fig.add_subplot(111)
            ax.plot(logs['time'], logs['gimbal_cmd'][:,0], label='Commanded Pitch', linestyle='--'); ax.plot(logs['time'], logs['gimbal_actual'][:,0], label='Actual Pitch')
            ax.set_title('Actuator Response (Averaged)'); ax.set_xlabel('Time (s)'); ax.set_ylabel('Gimbal Angle (degrees)'); ax.legend(); ax.grid(True)
        def plot_gyro_noise_effect(fig):
            ax = fig.add_subplot(111)
            ax.plot(logs['time'], np.rad2deg(logs['error_true'][:,0]), label='True Pitch Error', linestyle='--'); ax.plot(logs['time'], np.rad2deg(logs['sensed_error'][:,0]), label='Sensed Pitch Error', alpha=0.7)
            ax.set_title('Gyro Noise Effect on Pitch Error (Averaged)'); ax.set_xlabel('Time (s)'); ax.set_ylabel('Error (degrees)'); ax.legend(); ax.grid(True)
        def plot_wind_profile(fig):
            ax = fig.add_subplot(111); ax.plot(logs['wind_speed'], logs['state'][:, 2])
            ax.set_title('Wind Speed Profile vs. Altitude'); ax.set_xlabel('Wind Speed (m/s)'); ax.set_ylabel('Altitude (m)'); ax.grid(True)
        def plot_3d_trajectory(fig):
            ax = fig.add_subplot(111, projection='3d'); ax.plot(logs['state'][:,0], logs['state'][:,1], logs['state'][:,2], label='Average Trajectory')
            ax.set_title("3D Trajectory (Averaged)"); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)'); ax.legend(); ax.axis('equal')

        create_plot_tab("Position", plot_position)
        create_plot_tab("Velocity", plot_velocity)
        create_plot_tab("Actuator Response", plot_actuator_response)
        create_plot_tab("Gyro Noise", plot_gyro_noise_effect)
        create_plot_tab("Wind Profile", plot_wind_profile)
        create_plot_tab("3D Trajectory", plot_3d_trajectory)
        root.mainloop()
    else:
        print("\nNo simulation runs were completed.")