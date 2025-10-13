import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt
import time
import sys
import tkinter as tk
import csv
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- Monte Carlo Parameters ---
N_RUNS = 4

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
INERTIA_TENSOR = np.diag([0.01, 0.01, 0.001])
INERTIA_TENSOR_INV = np.linalg.inv(INERTIA_TENSOR)
ROCKET_DIAMETER = 0.034 # 34mm diameter
ROCKET_AREA = np.pi * (ROCKET_DIAMETER / 2)**2
# --- Advanced Aerodynamics ---
GIMBAL_VECTOR = np.array([0, 0, -1.0]) # Location of gimbal pivot relative to CoM
KP, KI, KD = 0.05, 0.1, 0.01 # PID gains are now constant
MAX_GIMBAL_ANGLE = np.deg2rad(10) # Limit gimbal to 10 degrees

# --- Helper Functions ---
def q_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

def rotate_by_quaternion(vector, q):
    return quaternion_to_rotation_matrix(q) @ vector

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

# --- Equations of Motion ---
def rocket_dynamics(t, state, params, pid_pitch, pid_yaw, dt):
    # Unpack parameters needed for this step
    C_A, propellant_mass, total_impulse, thrust_points = params['drag_coefficient'], params['propellant_mass'], params['total_impulse'], params['thrust_curve']
    air_density, C_N_alpha, cp_vector = params['air_density'], params['C_N_alpha'], params['cp_vector']
    pos, vel, quat, ang_vel, mass = state[0:3], state[3:6], state[6:10], state[10:13], state[13]
    quat /= np.linalg.norm(quat)

    # Get current thrust from the thrust curve
    current_thrust = np.interp(t, TIME_POINTS, thrust_points)

    # Calculate mass flow rate
    mass_flow_rate = 0.0
    if t < ENGINE_BURN_TIME and current_thrust > 0:
        mass_flow_rate = (propellant_mass / total_impulse) * current_thrust

    F_g = np.array([0, 0, -mass * G])

    v_rel = -vel
    v_mag = np.linalg.norm(v_rel)
    F_aero_world, τ_aero = np.zeros(3), np.zeros(3)

    if v_mag > 1e-6:
        # --- Advanced Aerodynamic Force Calculation ---
        # 1. Find Angle of Attack (AoA)
        z_axis_body_world = rotate_by_quaternion(np.array([0, 0, 1]), quat)
        vel_dir_world = vel / v_mag
        cos_aoa = np.dot(z_axis_body_world, vel_dir_world)
        # Clamp to prevent floating point errors with acos
        cos_aoa = np.clip(cos_aoa, -1.0, 1.0)
        aoa = np.arccos(cos_aoa)

        # 2. Calculate Force Coefficients based on AoA
        C_N = C_N_alpha * np.sin(aoa) # Normal force coefficient

        # 3. Calculate forces in the aerodynamic frame and rotate to body frame
        q_dyn = 0.5 * air_density * v_mag**2 * ROCKET_AREA # Dynamic Pressure q = 1/2 * rho * v^2 * A

        # Calculate forces in the body frame.
        # Axial force acts along the -Z axis of the rocket.
        F_axial_body = np.array([0, 0, -q_dyn * C_A])

        # Normal force acts perpendicular to the rocket's body, opposing the component of relative wind that is not aligned with the body.
        v_rel_body = rotate_by_quaternion(v_rel, q_conjugate(quat))
        v_rel_body_dir = v_rel_body / np.linalg.norm(v_rel_body)
        # The normal force vector in the body frame is proportional to the X and Y components of the relative wind direction.
        F_normal_body = q_dyn * C_N * np.array([v_rel_body_dir[0], v_rel_body_dir[1], 0])

        # Total aerodynamic force in the body frame
        F_aero_body = F_axial_body + F_normal_body

        # 4. Calculate aerodynamic torque
        τ_aero = np.cross(cp_vector, F_aero_body)
        F_aero_world = rotate_by_quaternion(F_aero_body, quat)

    F_thrust_body, τ_tvc = np.zeros(3), np.zeros(3)
    if t < ENGINE_BURN_TIME:
        z_axis_world = rotate_by_quaternion(np.array([0, 0, 1]), quat)
        error_pitch = z_axis_world[1]
        error_yaw = -z_axis_world[0]

        gimbal_pitch_cmd = -pid_pitch.update(error_pitch, dt)
        gimbal_yaw_cmd = -pid_yaw.update(error_yaw, dt)

        # --- Apply Gimbal Limits ---
        gimbal_pitch_cmd = np.clip(gimbal_pitch_cmd, -MAX_GIMBAL_ANGLE, MAX_GIMBAL_ANGLE)
        gimbal_yaw_cmd = np.clip(gimbal_yaw_cmd, -MAX_GIMBAL_ANGLE, MAX_GIMBAL_ANGLE)

        F_thrust_body = current_thrust * np.array([
            np.sin(gimbal_yaw_cmd),
            np.sin(gimbal_pitch_cmd),
            np.cos(gimbal_pitch_cmd)*np.cos(gimbal_yaw_cmd)
        ])
        τ_tvc = np.cross(GIMBAL_VECTOR, F_thrust_body)

    F_thrust_world = rotate_by_quaternion(F_thrust_body, quat)
    F_total = F_g + F_thrust_world + F_aero_world
    τ_total = τ_aero + τ_tvc

    pos_dot = vel
    vel_dot = F_total / mass
    omega_dot = INERTIA_TENSOR_INV @ τ_total
    
    omega_x, omega_y, omega_z = ang_vel
    omega_matrix = np.array([[0,-omega_x,-omega_y,-omega_z],[omega_x,0,omega_z,-omega_y],[omega_y,-omega_z,0,omega_x],[omega_z,omega_y,-omega_x,0]])
    quat_dot = 0.5 * omega_matrix @ quat

    dm_dt = -mass_flow_rate

    return np.concatenate((pos_dot, vel_dot, quat_dot, omega_dot, [dm_dt]))

def run_simulation(params: dict, full_logs=False) -> dict:
    """
    Runs a single rocket flight simulation with varied parameters.
    """
    # Calculate derived parameters and add them to the dictionary
    params['total_impulse'] = np.trapezoid(params['thrust_curve'], TIME_POINTS)
    params['cp_vector'] = np.array([0, 0, params['cp_z_offset']])

    # Initial state
    initial_state = np.zeros(14)
    initial_state[6] = 1.0  # qw = 1
    initial_state[13] = params['dry_mass'] + params['propellant_mass']
    initial_state[11] = params['yaw_init'] # Angular velocity around y-axis
    initial_state[10] = params['pitch_init'] # Angular velocity around x-axis

    t_start, t_end, dt = 0.0, 30.0, 0.02
    pid_pitch = PIDController(KP, KI, KD)
    pid_yaw = PIDController(KP, KI, KD)

    # --- Metrics tracking for this run ---
    apogee = 0.0
    state_at_apogee = None
    stable = True
    time_unstable = 0.0

    logs = {'time': [], 'state': [], 'gimbal': [], 'error': []}
    current_t, current_state = t_start, initial_state

    # Launch Clamp Logic
    initial_weight = (params['dry_mass'] + params['propellant_mass']) * G
    liftoff_thrust_indices = np.where(params['thrust_curve'] > initial_weight)[0]
    liftoff_time = TIME_POINTS[liftoff_thrust_indices[0]] if len(liftoff_thrust_indices) > 0 else t_end
    has_lifted_off = False

    while current_t < t_end:
        if has_lifted_off and current_state[2] <= 0:
            break

        if full_logs:
            logs['time'].append(current_t)
            logs['state'].append(current_state.copy())
            z_axis = rotate_by_quaternion(np.array([0,0,1]), current_state[6:10])
            logs['error'].append([z_axis[1], -z_axis[0]])
            logs['gimbal'].append([np.rad2deg(pid_pitch.last_output), np.rad2deg(pid_yaw.last_output)])

        # Always track apogee and stability, regardless of full_logs
        time_step = 0.0
        tilt_angle = 0.0 # Initialize here to prevent UnboundLocalError
        if has_lifted_off:
            apogee = max(apogee, current_state[2])
            if current_state[2] >= apogee and apogee > 0:
                state_at_apogee = current_state.copy()
            z_axis_body = rotate_by_quaternion(np.array([0, 0, 1]), current_state[6:10]) # This is the world vector
            tilt_angle = np.arccos(np.clip(np.dot(z_axis_body, np.array([0, 0, 1])), -1.0, 1.0))

            if np.rad2deg(tilt_angle) > 20:
                # This will be updated after the physics step to get the correct duration
                pass
            else:
                time_unstable = 0.0 # Reset the timer if the rocket is stable

        if current_t >= liftoff_time:
            prev_t = current_t
            solver = RK45(
                lambda t, y: rocket_dynamics(t, y, params, pid_pitch, pid_yaw, dt),
                current_t, current_state, t_end, max_step=dt
            )
            solver.step()
            has_lifted_off = True
            current_t, current_state = solver.t, solver.y
            time_step = current_t - prev_t
        else:
            prev_t = current_t
            current_thrust = np.interp(current_t, TIME_POINTS, params['thrust_curve'])
            if current_thrust > 0 and params['total_impulse'] > 0:
                mass_flow_rate = (params['propellant_mass'] / params['total_impulse']) * current_thrust
                current_state[13] -= mass_flow_rate * dt
            current_t += dt
            time_step = current_t - prev_t

        # Update stability timer after the time step is known
        if has_lifted_off and np.rad2deg(tilt_angle) > 20:
            time_unstable += time_step
            if time_unstable > 1.2:
                stable = False
                # The 'break' was removed as requested to allow failed simulations to complete.

    if full_logs:
        for key in logs:
            logs[key] = np.array(logs[key])
        return logs

    flight_time = current_t
    landing_pos = current_state[0:2]

    # --- Prepare detailed results ---
    results = {
        'apogee': apogee,
        'flight_time': flight_time,
        'stable': stable,
        'landing_x': landing_pos[0],
        'landing_y': landing_pos[1],
        'landing_distance': np.linalg.norm(landing_pos),
        'landing_vel_x': current_state[3],
        'landing_vel_y': current_state[4],
        'landing_vel_z': current_state[5],
    }

    if state_at_apogee is not None:
        apogee_vel = state_at_apogee[3:6]
        apogee_tilt_quat = state_at_apogee[6:10]
        z_axis_at_apogee = rotate_by_quaternion(np.array([0, 0, 1]), apogee_tilt_quat)
        apogee_tilt_angle = np.arccos(np.clip(np.dot(z_axis_at_apogee, np.array([0, 0, 1])), -1.0, 1.0))

        results.update({
            'apogee_vel_x': apogee_vel[0],
            'apogee_vel_y': apogee_vel[1],
            'apogee_vel_z': apogee_vel[2],
            'apogee_tilt_deg': np.rad2deg(apogee_tilt_angle)
        })

    return results


# --- Main Monte Carlo Loop ---
if __name__ == "__main__":
    all_results = []
    print(f"Starting Monte Carlo simulation with {N_RUNS} runs...")

    for i in range(N_RUNS):
        # --- Generate Randomized Thrust Curve ---
        # Create random multipliers for each point, normally distributed around 1.0
        random_factors = np.random.normal(loc=1.0, scale=THRUST_CURVE_STD, size=len(THRUST_POINTS_BASE))
        # Apply the random factors to the base curve
        randomized_thrust_curve = THRUST_POINTS_BASE * random_factors
        # Ensure the start and end of the thrust curve remain zero
        randomized_thrust_curve[0] = 0.0
        randomized_thrust_curve[-1] = 0.0
        
        # --- Generate Other Random Parameters ---
        params = {
            'thrust_curve': randomized_thrust_curve,
            'propellant_mass': np.random.normal(PROPELLANT_MASS_MEAN, PROPELLANT_MASS_MEAN * PROPELLANT_MASS_STD_PERCENT),
            'drag_coefficient': np.random.normal(DRAG_COEFFICIENT_MEAN, DRAG_COEFFICIENT_MEAN * DRAG_COEFFICIENT_STD_PERCENT),
            'dry_mass': np.random.normal(DRY_MASS_MEAN, DRY_MASS_MEAN * DRY_MASS_STD_PERCENT),
            'air_density': np.random.normal(AIR_DENSITY_MEAN, AIR_DENSITY_MEAN * AIR_DENSITY_STD_PERCENT),
            'C_N_alpha': np.random.normal(C_N_ALPHA_MEAN, C_N_ALPHA_MEAN * C_N_ALPHA_STD_PERCENT),
            'cp_z_offset': np.random.normal(CP_Z_OFFSET_MEAN, abs(CP_Z_OFFSET_MEAN) * CP_Z_OFFSET_STD_PERCENT), # Use abs for STD calculation
            'yaw_init': np.random.normal(YAW_INIT_MEAN, abs(YAW_INIT_MEAN) * YAW_INIT_STD_PERCENT),
            'pitch_init': np.random.normal(PITCH_INIT_MEAN, abs(PITCH_INIT_MEAN) * PITCH_INIT_STD_PERCENT),
        }

        result = run_simulation(params)
        all_results.append(result)
        sys.stdout.write(f'\rRunning simulation {i+1}/{N_RUNS}...')
        sys.stdout.flush()

    sys.stdout.write('\n')
    print("Monte Carlo simulation finished.")

    # --- Post-Processing and Output ---
    successful_runs = [r for r in all_results if r['stable']]
    success_rate = (len(successful_runs) / N_RUNS) * 100 if N_RUNS > 0 else 0

    if all_results:
        print("\n--- Monte Carlo Simulation Results ---")
        print(f"Success Rate: {success_rate:.2f}% ({len(successful_runs)}/{N_RUNS} stable flights)")

        # Collect data from ALL runs, not just successful ones
        apogees = [r.get('apogee', 0) for r in all_results]
        landing_distances = [r.get('landing_distance', 0) for r in all_results]

        print("\n--- Apogee (m) ---")
        if apogees:
            print(f"  Mean: {np.mean(apogees):.2f}")
            print(f"  Std Dev: {np.std(apogees):.2f}")
            print(f"  Min: {np.min(apogees):.2f}")
            print(f"  Max: {np.max(apogees):.2f}")
        
        print("\n--- Landing Distance (m) ---")
        if landing_distances:
            print(f"  Mean: {np.mean(landing_distances):.2f}")
            print(f"  Std Dev: {np.std(landing_distances):.2f}")
            print(f"  Min: {np.min(landing_distances):.2f}")
            print(f"  Max: {np.max(landing_distances):.2f}")
        print("------------------------------------")

        # --- Export raw data to CSV ---
        print("\nExporting raw simulation data to CSV...")
        csv_filename = "phase3_monte_carlo_results.csv"
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                if all_results:
                    # Define a fixed order for columns for consistency
                    fieldnames = ['run_number', 'stable', 'apogee', 'flight_time', 'landing_distance', 'landing_x', 'landing_y',
                                  'landing_vel_x', 'landing_vel_y', 'landing_vel_z', 'apogee_vel_x', 'apogee_vel_y',
                                  'apogee_vel_z', 'apogee_tilt_deg']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for i, result in enumerate(all_results):
                        row = {'run_number': i + 1, **result}
                        writer.writerow(row)
            print(f"Saved raw data to {csv_filename}")
        except IOError as e:
            print(f"Error writing to CSV file: {e}")
        
        # --- Generate Aggregate Plots ---
        print("\nGenerating and saving aggregate plots...")
        
        # Plot Apogee Distribution
        fig_apogee, ax_apogee = plt.subplots(figsize=(10, 6))
        ax_apogee.hist(apogees, bins=30, color='skyblue', edgecolor='black')
        ax_apogee.set_title('Apogee Distribution (All Flights)')
        ax_apogee.set_xlabel('Apogee (m)')
        ax_apogee.set_ylabel('Frequency')
        ax_apogee.grid(True)
        fig_apogee.tight_layout()
        apogee_filename = "phase3_apogee_distribution.png"
        fig_apogee.savefig(apogee_filename)
        print(f"Saved apogee distribution plot to {apogee_filename}")        
        
        # Plot Landing Distance Distribution
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        ax_dist.hist(landing_distances, bins=30, color='lightgreen', edgecolor='black')
        ax_dist.set_title('Landing Distance Distribution (All Flights)')
        ax_dist.set_xlabel('Landing Distance (m)')
        ax_dist.set_ylabel('Frequency')
        ax_dist.grid(True)
        fig_dist.tight_layout()
        distance_filename = "phase3_landing_distance_distribution.png"
        fig_dist.savefig(distance_filename)
        print(f"Saved landing distance distribution plot to {distance_filename}")

        # Plot Stability Distribution (Pie Chart)
        fig_stability, ax_stability = plt.subplots(figsize=(8, 8))
        unsuccessful_runs_count = N_RUNS - len(successful_runs)
        labels = 'Stable', 'Unstable'
        sizes = [len(successful_runs), unsuccessful_runs_count]
        colors = ['limegreen', 'orangered']
        explode = (0.1, 0) if sizes[0] > 0 and sizes[1] > 0 else (0, 0)

        ax_stability.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax_stability.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax_stability.set_title('Flight Stability Distribution')
        stability_filename = "phase3_stability_distribution.png"
        fig_stability.savefig(stability_filename)
        print(f"Saved stability distribution plot to {stability_filename}")

        # --- Single Run with Mean Parameters for Plotting ---
        print("\nGenerating plots for a single run with mean parameters...")
        mean_params = {
            'thrust_curve': THRUST_POINTS_BASE,
            'propellant_mass': PROPELLANT_MASS_MEAN,
            'drag_coefficient': DRAG_COEFFICIENT_MEAN,
            'dry_mass': DRY_MASS_MEAN,
            'air_density': AIR_DENSITY_MEAN,
            'C_N_alpha': C_N_ALPHA_MEAN,
            'cp_z_offset': CP_Z_OFFSET_MEAN,
            'yaw_init': YAW_INIT_MEAN, 'pitch_init': PITCH_INIT_MEAN,
        }
        logs = run_simulation(mean_params, full_logs=True)

        # --- Create Tabbed Plot Window ---
        root = tk.Tk()
        root.title("Vortex Rocket Simulation v0.3 Beta")
        notebook = ttk.Notebook(root)
        notebook.pack(pady=10, padx=10, expand=True, fill="both")

        def create_plot_tab(tab_name, plot_function):
            frame = ttk.Frame(notebook, width=800, height=600)
            notebook.add(frame, text=tab_name)
            
            fig = Figure(figsize=(10, 7), dpi=100)
            fig.set_tight_layout(True)
            
            plot_function(fig)
            
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Define Plotting Functions for each Tab ---
        def plot_position(fig):
            ax = fig.add_subplot(111)
            ax.plot(logs['time'], logs['state'][:,0], label='X Position')
            ax.plot(logs['time'], logs['state'][:,1], label='Y Position')
            ax.plot(logs['time'], logs['state'][:,2], label='Z Position (Altitude)')
            ax.set_title("Position vs. Time")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (m)")
            ax.legend(); ax.grid(True)

        def plot_velocity(fig):
            ax = fig.add_subplot(111)
            ax.plot(logs['time'], logs['state'][:,3], label='Vx'); ax.plot(logs['time'], logs['state'][:,4], label='Vy'); ax.plot(logs['time'], logs['state'][:,5], label='Vz')
            ax.set_title("Velocity Components vs. Time"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Velocity (m/s)")
            ax.legend(); ax.grid(True)

        def plot_acceleration(fig):
            ax = fig.add_subplot(111)
            accel = np.gradient(logs['state'][:, 3:6], logs['time'], axis=0)
            ax.plot(logs['time'], accel[:,0], label='Ax'); ax.plot(logs['time'], accel[:,1], label='Ay'); ax.plot(logs['time'], accel[:,2], label='Az')
            ax.set_title("Acceleration Components vs. Time"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Acceleration (m/s^2)")
            ax.legend(); ax.grid(True)

        def plot_tilt_error(fig):
            ax = fig.add_subplot(111)
            ax.plot(logs['time'], np.rad2deg(logs['error'][:,0]), label='Pitch Error'); ax.plot(logs['time'], np.rad2deg(logs['error'][:,1]), label='Yaw Error')
            ax.set_title("Tilt Error vs. Time"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (degrees)")
            ax.legend(); ax.grid(True)

        def plot_gimbal_cmd(fig):
            ax = fig.add_subplot(111)
            ax.plot(logs['time'], logs['gimbal'][:,0], label='Pitch Command'); ax.plot(logs['time'], logs['gimbal'][:,1], label='Yaw Command')
            ax.set_title("Gimbal Command vs. Time"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (degrees)")
            ax.legend(); ax.grid(True)

        def plot_3d_trajectory(fig):
            ax = fig.add_subplot(111, projection='3d')
            position = logs['state'][:, 0:3].T
            ax.plot(position[0], position[1], position[2], label='Trajectory')
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
            ax.set_title("3D Trajectory"); ax.legend(); ax.axis('equal')

        # --- Create all tabs ---
        create_plot_tab("Position", plot_position)
        create_plot_tab("Velocity", plot_velocity)
        create_plot_tab("Acceleration", plot_acceleration)
        create_plot_tab("Tilt Error", plot_tilt_error)
        create_plot_tab("Gimbal Command", plot_gimbal_cmd)
        create_plot_tab("3D Trajectory", plot_3d_trajectory)

        root.mainloop()

    else:
        print("\n--- Monte Carlo Simulation Results ---")
        print("No simulation runs were completed.")
        print("------------------------------------")
