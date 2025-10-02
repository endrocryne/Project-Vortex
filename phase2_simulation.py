import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt

# --- Constants ---
G = 9.80665
DRY_MASS = 1.5
PROPELLANT_MASS = 0.8
ENGINE_BURN_TIME = 4.0
THRUST_FORCE = 60.0
INERTIA_TENSOR = np.diag([0.01, 0.01, 0.001])
INERTIA_TENSOR_INV = np.linalg.inv(INERTIA_TENSOR)
AIR_DENSITY = 1.225
ROCKET_DIAMETER = 0.05
ROCKET_AREA = np.pi * (ROCKET_DIAMETER / 2)**2
DRAG_COEFFICIENT = 0.6
CP_VECTOR = np.array([0, 0, 0.5])
GIMBAL_VECTOR = np.array([0, 0, -1.0])
KP, KI, KD = 0.08, 0.02, 0.06

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
def rocket_dynamics(t, state, pid_pitch, pid_yaw, dt):
    pos, vel, quat, ang_vel = state[0:3], state[3:6], state[6:10], state[10:13]
    quat /= np.linalg.norm(quat)

    mass = DRY_MASS + (PROPELLANT_MASS * (1 - t / ENGINE_BURN_TIME) if t < ENGINE_BURN_TIME else 0)
    F_g = np.array([0, 0, -mass * G])

    v_rel = -vel
    v_mag = np.linalg.norm(v_rel)
    F_drag, τ_aero = np.zeros(3), np.zeros(3)
    if v_mag > 1e-6:
        F_drag = 0.5 * AIR_DENSITY * v_mag**2 * ROCKET_AREA * DRAG_COEFFICIENT * (v_rel / v_mag)
        F_drag_body = rotate_by_quaternion(F_drag, q_conjugate(quat))
        τ_aero = np.cross(CP_VECTOR, F_drag_body)

    F_thrust_body, τ_tvc = np.zeros(3), np.zeros(3)
    if t < ENGINE_BURN_TIME:
        z_axis_world = rotate_by_quaternion(np.array([0, 0, 1]), quat)
        error_pitch = z_axis_world[1]
        error_yaw = -z_axis_world[0]

        # Definitive Corrected Control Logic for Negative Feedback
        gimbal_pitch_cmd = -pid_pitch.update(error_pitch, dt)
        gimbal_yaw_cmd = -pid_yaw.update(error_yaw, dt)

        F_thrust_body = THRUST_FORCE * np.array([
            np.sin(gimbal_yaw_cmd),
            np.sin(gimbal_pitch_cmd),
            np.cos(gimbal_pitch_cmd)*np.cos(gimbal_yaw_cmd)
        ])
        τ_tvc = np.cross(GIMBAL_VECTOR, F_thrust_body)

    F_thrust_world = rotate_by_quaternion(F_thrust_body, quat)
    F_total = F_g + F_thrust_world + F_drag
    τ_total = τ_aero + τ_tvc

    pos_dot = vel
    vel_dot = F_total / mass
    omega_dot = INERTIA_TENSOR_INV @ τ_total

    omega_x, omega_y, omega_z = ang_vel
    omega_matrix = np.array([[0,-omega_x,-omega_y,-omega_z],[omega_x,0,omega_z,-omega_y],[omega_y,-omega_z,0,omega_x],[omega_z,omega_y,-omega_x,0]])
    quat_dot = 0.5 * omega_matrix @ quat

    return np.concatenate((pos_dot, vel_dot, quat_dot, omega_dot))

# --- Main Simulation Loop ---
if __name__ == "__main__":
    initial_state = np.zeros(13)
    initial_state[6] = 1.0 # qw = 1
    # As per prompt, disturbance is on state[11]
    initial_state[11] = 0.1

    t_start, t_end, dt = 0.0, 15.0, 0.02

    pid_pitch = PIDController(KP, KI, KD)
    pid_yaw = PIDController(KP, KI, KD)

    logs = {'time': [], 'state': [], 'gimbal': [], 'error': []}
    current_t, current_state = t_start, initial_state

    print("Starting simulation...")
    while current_t < t_end:
        if current_t > 0.1 and current_state[2] <= 0:
            print(f"Ground impact detected at t={current_t:.2f}s. Stopping simulation.")
            break

        logs['time'].append(current_t)
        logs['state'].append(current_state)
        z_axis = rotate_by_quaternion(np.array([0,0,1]), current_state[6:10])
        logs['error'].append([z_axis[1], -z_axis[0]])
        logs['gimbal'].append([np.rad2deg(pid_pitch.last_output), np.rad2deg(pid_yaw.last_output)])

        solver = RK45(lambda t, y: rocket_dynamics(t, y, pid_pitch, pid_yaw, dt), current_t, current_state, t_end, max_step=dt)
        solver.step()
        current_t, current_state = solver.t, solver.y
    print("Simulation finished.")

    for key in logs: logs[key] = np.array(logs[key])

    print("Generating and saving plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10,6)); plt.plot(logs['time'], logs['state'][:,2]); plt.title("Altitude vs. Time"); plt.xlabel("Time (s)"); plt.ylabel("Altitude (m)"); plt.savefig("phase2_altitude.png")
    plt.figure(figsize=(10,6)); plt.plot(logs['time'], np.rad2deg(logs['error'][:,0]), label='Pitch Error'); plt.plot(logs['time'], np.rad2deg(logs['error'][:,1]), label='Yaw Error'); plt.title("Tilt Error vs. Time"); plt.xlabel("Time (s)"); plt.ylabel("Error (degrees)"); plt.legend(); plt.savefig("phase2_tilt.png")
    plt.figure(figsize=(10,6)); plt.plot(logs['time'], logs['gimbal'][:,0], label='Pitch Command'); plt.plot(logs['time'], logs['gimbal'][:,1], label='Yaw Command'); plt.title("Gimbal Command vs. Time"); plt.xlabel("Time (s)"); plt.ylabel("Angle (degrees)"); plt.legend(); plt.savefig("phase2_gimbal.png")
    print("Plots saved to phase2_altitude.png, phase2_tilt.png, and phase2_gimbal.png.")
    plt.show()