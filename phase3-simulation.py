import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt
import time
import sys

# --- Monte Carlo Parameters ---
N_RUNS = 500

# Base values from Phase 2 are now the means
THRUST_FORCE_MEAN = 60.0
THRUST_FORCE_STD = 0.05 # 5% Variation

PROPELLANT_MASS_MEAN = 0.8
PROPELLANT_MASS_STD = 0.1 # 10% Variation

DRAG_COEFFICIENT_MEAN = 0.6
DRAG_COEFFICIENT_STD = 0.1 # 10% variation

# --- Constants ---
G = 9.80665
DRY_MASS = 0.060  # 60 grams - typical for a mid-size model rocket

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
AIR_DENSITY = 1.225
ROCKET_DIAMETER = 0.034 # 34mm diameter
ROCKET_AREA = np.pi * (ROCKET_DIAMETER / 2)**2
# --- Advanced Aerodynamics ---
C_N_alpha = 2.5 # Normal Force Coefficient derivative (per radian)
CP_VECTOR = np.array([0, 0, -0.2]) # Center of Pressure MUST be behind CoM for stability
GIMBAL_VECTOR = np.array([0, 0, -1.0]) # Location of gimbal pivot relative to CoM
KP, KI, KD = 0.05, 0.1, 0.01 # Re-tuned PID gains for a lighter rocket
YAW_INIT = 0.1 # Initial yaw disturbance
PITCH_INIT = 0.1 # Initial pitch disturbance
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
def rocket_dynamics(t, state, pid_pitch, pid_yaw, dt, C_A, propellant_mass, total_impulse, thrust_points):
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
        q_dyn = 0.5 * AIR_DENSITY * v_mag**2 * ROCKET_AREA # Dynamic Pressure q = 1/2 * rho * v^2 * A

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
        τ_aero = np.cross(CP_VECTOR, F_aero_body)
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

def run_simulation(params: dict) -> dict:
    """
    Runs a single rocket flight simulation with varied parameters.
    """
    # Extract parameters
    thrust_force_multiplier = params['thrust_force']
    propellant_mass = params['propellant_mass']
    C_A = params['drag_coefficient']

    # Scale the thrust curve
    thrust_points = THRUST_POINTS_BASE * thrust_force_multiplier
    total_impulse = np.trapz(thrust_points, TIME_POINTS)

    # Initial state
    initial_state = np.zeros(14)
    initial_state[6] = 1.0  # qw = 1
    initial_state[13] = DRY_MASS + propellant_mass
    initial_state[11] = YAW_INIT
    initial_state[10] = PITCH_INIT

    t_start, t_end, dt = 0.0, 30.0, 0.02
    pid_pitch = PIDController(KP, KI, KD)
    pid_yaw = PIDController(KP, KI, KD)

    logs = {'time': [], 'state': []}
    current_t, current_state = t_start, initial_state

    # Launch Clamp Logic
    initial_weight = (DRY_MASS + propellant_mass) * G
    liftoff_thrust_indices = np.where(thrust_points > initial_weight)[0]
    liftoff_time = TIME_POINTS[liftoff_thrust_indices[0]] if len(liftoff_thrust_indices) > 0 else t_end
    has_lifted_off = False

    while current_t < t_end:
        if has_lifted_off and current_state[2] <= 0:
            break

        logs['time'].append(current_t)
        logs['state'].append(current_state.copy())

        if current_t >= liftoff_time:
            solver = RK45(
                lambda t, y: rocket_dynamics(t, y, pid_pitch, pid_yaw, dt, C_A, propellant_mass, total_impulse, thrust_points),
                current_t, current_state, t_end, max_step=dt
            )
            solver.step()
            has_lifted_off = True
            current_t, current_state = solver.t, solver.y
        else:
            current_thrust = np.interp(current_t, TIME_POINTS, thrust_points)
            if current_thrust > 0 and total_impulse > 0:
                mass_flow_rate = (propellant_mass / total_impulse) * current_thrust
                current_state[13] -= mass_flow_rate * dt
            current_t += dt

    for key in logs:
        logs[key] = np.array(logs[key])

    # Post-simulation analysis
    if len(logs['time']) > 1:
        apogee = np.max(logs['state'][:, 2])
        flight_time = logs['time'][-1]
        landing_pos = logs['state'][-1, 0:2]
        landing_distance = np.linalg.norm(landing_pos)

        # Stability check
        stable = True
        max_tilt_angle = 0
        for q_state in logs['state'][:, 6:10]:
            z_axis_body = rotate_by_quaternion(np.array([0, 0, 1]), q_state)
            # Tilt is the angle between rocket's Z and world's Z
            tilt_angle = np.arccos(np.dot(z_axis_body, np.array([0, 0, 1])))
            if np.rad2deg(tilt_angle) > 20:
                stable = False
                break
            max_tilt_angle = max(max_tilt_angle, tilt_angle)
    else:
        # Handle cases where simulation ends prematurely
        apogee, flight_time, landing_distance, stable = 0, 0, 0, False

    return {
        'apogee': apogee,
        'landing_distance': landing_distance,
        'flight_time': flight_time,
        'stable': stable
    }


# --- Main Monte Carlo Loop ---
if __name__ == "__main__":
    all_results = []
    print(f"Starting Monte Carlo simulation with {N_RUNS} runs...")

    # Calculate the thrust multiplier needed to achieve the mean thrust
    peak_thrust_base = np.max(THRUST_POINTS_BASE)
    if peak_thrust_base > 0:
        thrust_multiplier_mean = THRUST_FORCE_MEAN / peak_thrust_base
    else:
        thrust_multiplier_mean = 1.0
    
    thrust_multiplier_std_dev = thrust_multiplier_mean * THRUST_FORCE_STD

    for i in range(N_RUNS):
        # Generate Random Parameters
        thrust_multiplier = np.random.normal(thrust_multiplier_mean, thrust_multiplier_std_dev)
        
        params = {
            'thrust_force': thrust_multiplier,
            'propellant_mass': np.random.normal(PROPELLANT_MASS_MEAN, PROPELLANT_MASS_STD * PROPELLANT_MASS_MEAN),
            'drag_coefficient': np.random.normal(DRAG_COEFFICIENT_MEAN, DRAG_COEFFICIENT_STD * DRAG_COEFFICIENT_MEAN),
        }

        result = run_simulation(params)
        all_results.append(result)
        sys.stdout.write(f'\rRunning simulation {i+1}/{N_RUNS}...')
        sys.stdout.flush()

    sys.stdout.write('\n')
    print("Monte Carlo simulation finished.")

    # --- Post-Processing and Output ---
    successful_runs = [r for r in all_results if r['stable']]
    success_rate = (len(successful_runs) / N_RUNS) * 100

    if successful_runs:
        apogees = [r['apogee'] for r in successful_runs]
        landing_distances = [r['landing_distance'] for r in successful_runs]

        print("\n--- Monte Carlo Simulation Results ---")
        print(f"Success Rate: {success_rate:.2f}% ({len(successful_runs)}/{N_RUNS} stable flights)")
        print("\n--- Apogee (m) ---")
        print(f"  Mean: {np.mean(apogees):.2f}")
        print(f"  Std Dev: {np.std(apogees):.2f}")
        print(f"  Min: {np.min(apogees):.2f}")
        print(f"  Max: {np.max(apogees):.2f}")

        print("\n--- Landing Distance (m) ---")
        print(f"  Mean: {np.mean(landing_distances):.2f}")
        print(f"  Std Dev: {np.std(landing_distances):.2f}")
        print(f"  Min: {np.min(landing_distances):.2f}")
        print(f"  Max: {np.max(landing_distances):.2f}")
        print("------------------------------------")

        # --- Generate Aggregate Plots ---
        print("\nGenerating and saving aggregate plots...")
        
        # Plot Apogee Distribution
        fig_apogee, ax_apogee = plt.subplots(figsize=(10, 6))
        ax_apogee.hist(apogees, bins=30, color='skyblue', edgecolor='black')
        ax_apogee.set_title('Apogee Distribution for Successful Flights')
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
        ax_dist.set_title('Landing Distance Distribution for Successful Flights')
        ax_dist.set_xlabel('Landing Distance (m)')
        ax_dist.set_ylabel('Frequency')
        ax_dist.grid(True)
        fig_dist.tight_layout()
        distance_filename = "phase3_landing_distance_distribution.png"
        fig_dist.savefig(distance_filename)
        print(f"Saved landing distance distribution plot to {distance_filename}")

    else:
        print("\n--- Monte Carlo Simulation Results ---")
        print("No successful flights were recorded.")
        print("------------------------------------")
