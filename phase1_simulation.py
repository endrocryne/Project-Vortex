import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Constants ---
G = 9.80665  # m/s^2, gravitational acceleration
DRY_MASS = 1.5  # kg, mass of rocket without propellant
PROPELLANT_MASS = 0.8  # kg, mass of propellant
ENGINE_BURN_TIME = 4.0  # seconds
THRUST_FORCE = 60.0  # Newtons, constant thrust

# Rotational inertia tensor [I_xx, I_yy, I_zz] (kg*m^2)
INERTIA_TENSOR = np.diag([0.01, 0.01, 0.001])
INERTIA_TENSOR_INV = np.linalg.inv(INERTIA_TENSOR)

# --- Helper Functions ---

def quaternion_to_rotation_matrix(q):
    """Converts a quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

def rotate_by_quaternion(vector, q):
    """Rotates a 3D vector by a quaternion."""
    rotation_matrix = quaternion_to_rotation_matrix(q)
    return rotation_matrix @ vector

# --- Equations of Motion ---

def rocket_dynamics(t, state):
    """
    Calculates the time derivative of the state vector for the rocket.
    State vector: [pos(3), vel(3), quaternion(4), angular_vel(3)]
    """
    # 1. Unpack State
    position = state[0:3]
    velocity = state[3:6]
    quaternion = state[6:10]
    angular_velocity = state[10:13]

    # Normalize the quaternion to prevent drift
    quaternion /= np.linalg.norm(quaternion)

    # 2. Calculate Current Mass
    if t < ENGINE_BURN_TIME:
        mass_flow_rate = PROPELLANT_MASS / ENGINE_BURN_TIME
        current_mass = DRY_MASS + PROPELLANT_MASS - mass_flow_rate * t
    else:
        current_mass = DRY_MASS

    # 3. Calculate Forces
    # Gravity (World Frame)
    F_g = np.array([0, 0, -current_mass * G])

    # Thrust (Body Frame)
    if t < ENGINE_BURN_TIME:
        F_thrust_body = np.array([0, 0, THRUST_FORCE])
    else:
        F_thrust_body = np.array([0, 0, 0])

    # 4. Rotate Thrust to World Frame
    F_thrust_world = rotate_by_quaternion(F_thrust_body, quaternion)

    # 5. Total Force
    F_total = F_g + F_thrust_world

    # 6. Calculate State Derivatives
    # Position derivative (velocity)
    pos_dot = velocity

    # Velocity derivative (acceleration)
    vel_dot = F_total / current_mass

    # Angular velocity derivative (no torques in this phase)
    omega_dot = np.zeros(3)

    # Quaternion derivative
    omega_x, omega_y, omega_z = angular_velocity
    omega_matrix = np.array([
        [0, -omega_x, -omega_y, -omega_z],
        [omega_x, 0, omega_z, -omega_y],
        [omega_y, -omega_z, 0, omega_x],
        [omega_z, omega_y, -omega_x, 0]
    ])
    quat_dot = 0.5 * omega_matrix @ quaternion

    # 7. Return Derivatives
    derivatives = np.concatenate((pos_dot, vel_dot, quat_dot, omega_dot))
    return derivatives

# --- Main Simulation Logic ---
if __name__ == "__main__":
    # Initial state: [pos, vel, quat, ang_vel]
    # pos = [0,0,0], vel = [0,0,0], quat = [1,0,0,0], ang_vel = [0,0,0]
    initial_state = np.zeros(13)
    initial_state[6] = 1.0  # Set qw to 1 for no initial rotation

    # Simulation time
    t_span = [0, 30]
    t_eval = np.linspace(t_span[0], t_span[1], 1001)

    # Run the simulation
    result = solve_ivp(
        rocket_dynamics,
        t_span,
        initial_state,
        t_eval=t_eval,
        dense_output=True,
        method='RK45'
    )

    # --- Output and Verification ---
    # Extract results
    time = result.t
    altitude = result.y[2]  # z-position

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(time, altitude)
    plt.title("Altitude vs. Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.grid(True)
    plt.savefig("phase1_trajectory.png")
    print("Simulation complete. Trajectory plot saved as 'phase1_trajectory.png'")
    # plt.show() # Uncomment to display the plot directly