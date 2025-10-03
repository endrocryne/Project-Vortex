import numpy as np
from scipy.integrate import RK45

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
GIMBAL_VECTOR = np.array([0, 0, -1.0]) # Location of gimbal pivot relative to CoM
MAX_GIMBAL_ANGLE = np.deg2rad(10) # Limit gimbal to 10 degrees

# --- Default Monte Carlo Parameters ---
# These will be used by the GUI as default values
DEFAULT_PARAMS = {
    'thrust_mean': 1.0, 'thrust_std': 0.05,
    'propellant_mass_mean': 0.0125, 'propellant_mass_std': 0.05,
    'drag_coefficient_mean': 0.6, 'drag_coefficient_std': 0.10,
    'dry_mass_mean': 0.060, 'dry_mass_std': 0.025,
    'air_density_mean': 1.225, 'air_density_std': 0.02,
    'c_n_alpha_mean': 2.5, 'c_n_alpha_std': 0.10,
    'cp_z_offset_mean': -0.2, 'cp_z_offset_std': 0.05,
    'yaw_init_mean': 0.1, 'yaw_init_std': 0.20,
    'pitch_init_mean': 0.1, 'pitch_init_std': 0.20,
    'gyro_noise_std': 0.0, # New parameter for GUI
    'actuator_tau': 0.0, # New parameter for GUI
}

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
        z_axis_body_world = rotate_by_quaternion(np.array([0, 0, 1]), quat)
        vel_dir_world = vel / v_mag
        cos_aoa = np.dot(z_axis_body_world, vel_dir_world)
        cos_aoa = np.clip(cos_aoa, -1.0, 1.0)
        aoa = np.arccos(cos_aoa)
        C_N = C_N_alpha * np.sin(aoa)
        q_dyn = 0.5 * air_density * v_mag**2 * ROCKET_AREA
        F_axial_body = np.array([0, 0, -q_dyn * C_A])
        v_rel_body = rotate_by_quaternion(v_rel, q_conjugate(quat))
        v_rel_body_dir = v_rel_body / np.linalg.norm(v_rel_body)
        F_normal_body = q_dyn * C_N * np.array([v_rel_body_dir[0], v_rel_body_dir[1], 0])
        F_aero_body = F_axial_body + F_normal_body
        τ_aero = np.cross(cp_vector, F_aero_body)
        F_aero_world = rotate_by_quaternion(F_aero_body, quat)

    F_thrust_body, τ_tvc = np.zeros(3), np.zeros(3)
    if t < ENGINE_BURN_TIME:
        z_axis_world = rotate_by_quaternion(np.array([0, 0, 1]), quat)
        
        # Sensor Noise
        gyro_noise = np.random.normal(0, params.get('gyro_noise_std', 0.0))
        
        error_pitch = z_axis_world[1] + gyro_noise
        error_yaw = -z_axis_world[0] + gyro_noise

        gimbal_pitch_cmd = -pid_pitch.update(error_pitch, dt)
        gimbal_yaw_cmd = -pid_yaw.update(error_yaw, dt)

        # Actuator Delay (simple first-order filter)
        tau = params.get('actuator_tau', 0.0)
        # This part is tricky in RK45. A proper stateful actuator model would be better.
        # For now, we assume the command is executed, as RK45 doesn't let us easily model this.
        
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

    # Return the full state derivative, including gimbal angles for logging
    return np.concatenate((pos_dot, vel_dot, quat_dot, omega_dot, [dm_dt]))

def run_simulation(params: dict, Kp, Ki, Kd, full_logs=False, progress_callback=None) -> dict:
    """
    Runs a single rocket flight simulation with varied parameters.
    """
    params['total_impulse'] = np.trapezoid(params['thrust_curve'], TIME_POINTS)
    params['cp_vector'] = np.array([0, 0, params['cp_z_offset']])

    initial_state = np.zeros(14)
    initial_state[6] = 1.0
    initial_state[13] = params['dry_mass'] + params['propellant_mass']
    initial_state[11] = params.get('yaw_init', 0.0)
    initial_state[10] = params.get('pitch_init', 0.0)

    t_start, t_end, dt = 0.0, 30.0, 0.02
    pid_pitch = PIDController(Kp, Ki, Kd)
    pid_yaw = PIDController(Kp, Ki, Kd)

    apogee = 0.0
    max_tilt_angle = 0.0
    stable = True
    
    logs = {'time': [], 'state': [], 'gimbal_pitch': [], 'gimbal_yaw': []}
    current_t, current_state = t_start, initial_state

    initial_weight = (params['dry_mass'] + params['propellant_mass']) * G
    liftoff_thrust_indices = np.where(params['thrust_curve'] > initial_weight)[0]
    liftoff_time = TIME_POINTS[liftoff_thrust_indices[0]] if len(liftoff_thrust_indices) > 0 else t_end
    has_lifted_off = False

    solver = RK45(
        lambda t, y: rocket_dynamics(t, y, params, pid_pitch, pid_yaw, dt),
        current_t, current_state, t_end, max_step=dt
    )

    while solver.status == 'running':
        if has_lifted_off and current_state[2] <= 0:
            break
            
        if current_t >= liftoff_time:
            has_lifted_off = True
            solver.step()
            current_t, current_state = solver.t, solver.y
        else: # On launchpad, pre-liftoff
            current_t += dt
            current_thrust = np.interp(current_t, TIME_POINTS, params['thrust_curve'])
            if current_thrust > 0 and params['total_impulse'] > 0:
                mass_flow_rate = (params['propellant_mass'] / params['total_impulse']) * current_thrust
                current_state[13] -= mass_flow_rate * dt
            solver.t = current_t
            solver.y = current_state
            
        if has_lifted_off:
            apogee = max(apogee, current_state[2])
            z_axis_body = rotate_by_quaternion(np.array([0, 0, 1]), current_state[6:10])
            tilt_angle = np.arccos(np.clip(np.dot(z_axis_body, np.array([0, 0, 1])), -1.0, 1.0))
            max_tilt_angle = max(max_tilt_angle, np.rad2deg(tilt_angle))

            if np.rad2deg(tilt_angle) > 20:
                stable = False

        if full_logs:
            logs['time'].append(current_t)
            logs['state'].append(current_state.copy())
            logs['gimbal_pitch'].append(np.rad2deg(pid_pitch.last_output))
            logs['gimbal_yaw'].append(np.rad2deg(pid_yaw.last_output))
            if progress_callback:
                progress_callback(current_t, current_state)

    landing_pos = current_state[0:2]
    results = {
        'apogee': apogee,
        'flight_time': current_t,
        'stable': stable,
        'landing_distance': np.linalg.norm(landing_pos),
        'max_tilt_angle': max_tilt_angle
    }
    
    if full_logs:
        results['logs'] = logs

    return results