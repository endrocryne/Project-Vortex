import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import sys

# --- Constants ---
G = 9.80665
DRY_MASS = 0.5 # Dry mass is the mass of the rocket without propellant
PROPELLANT_MASS = 0.8 # Mass of the propellant
ENGINE_BURN_TIME = 1.0 # seconds # Duration of engine burn
THRUST_FORCE = 450.0 # Newtons, constant thrust (TODO: Implement thrust curve)
INERTIA_TENSOR = np.diag([0.01, 0.01, 0.001]) # ELI5: Diagonal inertia tensor for simplicity
INERTIA_TENSOR_INV = np.linalg.inv(INERTIA_TENSOR) # ELI5: Inverse of the inertia tensor
AIR_DENSITY = 5.225 # kg/m^3, density of air at sea level
ROCKET_DIAMETER = 0.05 # meters
ROCKET_AREA = np.pi * (ROCKET_DIAMETER / 2)**2 # Cross-sectional area
DRAG_COEFFICIENT = 0.9 # Coefficient of drag
CP_VECTOR = np.array([0, 0, 0.5]) # Location of center of pressure relative to CoM (Center of Mass)
GIMBAL_VECTOR = np.array([0, 0, -1.0]) # Location of gimbal pivot relative to CoM (Center of Mass)
KP, KI, KD = 0.1454, 2.389, 0.06 # Tuned PID gains
YAW_INIT = 0.1 # Initial yaw disturbance
PITCH_INIT = 0.1 # Initial pitch disturbance

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
    initial_state[11] = YAW_INIT

     # Disturbance on ang_vel_x (state[10]) to induce pitch and Y-axis motion
    initial_state[10] = PITCH_INIT
    
    t_start, t_end, dt = 0.0, 40.0, 0.02
    
    pid_pitch = PIDController(KP, KI, KD)
    pid_yaw = PIDController(KP, KI, KD)
    
    logs = {'time': [], 'state': [], 'gimbal': [], 'error': []}
    current_t, current_state = t_start, initial_state
    
    print("Starting simulation...")
    start_real_time = time.time()
    last_update_time = start_real_time

    while current_t < t_end:
        if current_t > 0.1 and current_state[2] <= 0:
            print(f"Ground impact detected at t={current_t:.2f}s. Stopping simulation.")
            break
        
        logs['time'].append(current_t)
        logs['state'].append(current_state)
        z_axis = rotate_by_quaternion(np.array([0,0,1]), current_state[6:10])
        logs['error'].append([z_axis[1], -z_axis[0]])
        logs['gimbal'].append([np.rad2deg(pid_pitch.last_output), np.rad2deg(pid_yaw.last_output)])

        # Re-initialize the solver to take a single step
        solver = RK45(lambda t, y: rocket_dynamics(t, y, pid_pitch, pid_yaw, dt), current_t, current_state, t_end, max_step=dt)
        solver.step() # This advances the solver by one internal step
        
        # Update state for the next iteration
        current_t, current_state = solver.t, solver.y

        # --- Progress Bar Update ---
        now = time.time()
        if now - last_update_time > 0.1:  # Update display every 0.1 seconds
            last_update_time = now
            progress = current_t / t_end
            elapsed_time = now - start_real_time

            if progress > 1e-6: # Avoid division by zero at the start
                total_time_estimated = elapsed_time / progress
                remaining_time = total_time_estimated - elapsed_time
                etr_str = f"ETR: {remaining_time:.1f}s"
            else:
                etr_str = "ETR: calculating..."

            percent = int(progress * 100)
            bar = '█' * int(percent / 2) + '-' * (50 - int(percent / 2))
            sys.stdout.write(f'\rProgress: |{bar}| {percent}% Complete | {etr_str}   ')
            sys.stdout.flush()

    sys.stdout.write('\n') # Move to the next line after the progress bar
    print("Simulation finished.")

    for key in logs: logs[key] = np.array(logs[key])

    print("\n--- Raw Simulation Data ---")
    print(logs)
    print("---------------------------\n")

    print("Generating interactive plot window...")

    # --- Create Tabbed Plot Window ---
    root = tk.Tk()
    root.title("Rocket Simulation Analysis")
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