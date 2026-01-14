"""
Visualization module for simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_results(history, save_path=None):
    """
    Generate comprehensive plots of simulation results.
    
    Creates:
    - 3D trajectory plot
    - Altitude vs time
    - Velocity vs time
    - Euler angles vs time
    - Gimbal angles vs time
    - Additional plots (mass, thrust)
    
    Args:
        history: Dictionary with simulation history arrays
        save_path: Optional path to save figure
    """
    # Convert to arrays if needed
    time = np.array(history['time'])
    position = np.array(history['position'])
    velocity = np.array(history['velocity'])
    euler_angles = np.array(history['euler_angles'])
    gimbal_angles = np.array(history['gimbal_angles'])
    thrust = np.array(history['thrust'])
    mass = np.array(history['mass'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # ========== 3D TRAJECTORY PLOT ==========
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    
    # Extract coordinates (NED convention: x=North, y=East, z=Down)
    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]
    
    # Plot trajectory
    ax1.plot(x, y, -z, 'b-', linewidth=2, label='Trajectory')  # Negative z for altitude up
    ax1.scatter(x[0], y[0], -z[0], c='green', s=100, marker='o', label='Launch')
    ax1.scatter(x[-1], y[-1], -z[-1], c='red', s=100, marker='x', label='Landing')
    
    ax1.set_xlabel('North (m)')
    ax1.set_ylabel('East (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # ========== ALTITUDE VS TIME ==========
    ax2 = fig.add_subplot(3, 3, 2)
    altitude = -z  # Altitude is negative z in NED
    ax2.plot(time, altitude, 'b-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude vs Time')
    ax2.grid(True)
    
    # ========== VELOCITY MAGNITUDE VS TIME ==========
    ax3 = fig.add_subplot(3, 3, 3)
    velocity_magnitude = np.linalg.norm(velocity, axis=1)
    ax3.plot(time, velocity_magnitude, 'r-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity Magnitude vs Time')
    ax3.grid(True)
    
    # ========== VELOCITY COMPONENTS ==========
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(time, velocity[:, 0], 'r-', label='Vx (North)', linewidth=1.5)
    ax4.plot(time, velocity[:, 1], 'g-', label='Vy (East)', linewidth=1.5)
    ax4.plot(time, velocity[:, 2], 'b-', label='Vz (Down)', linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Velocity Components')
    ax4.legend()
    ax4.grid(True)
    
    # ========== EULER ANGLES VS TIME ==========
    ax5 = fig.add_subplot(3, 3, 5)
    roll = np.degrees(euler_angles[:, 0])
    pitch = np.degrees(euler_angles[:, 1])
    yaw = np.degrees(euler_angles[:, 2])
    
    ax5.plot(time, roll, 'r-', label='Roll', linewidth=1.5)
    ax5.plot(time, pitch, 'g-', label='Pitch', linewidth=1.5)
    ax5.plot(time, yaw, 'b-', label='Yaw', linewidth=1.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angle (degrees)')
    ax5.set_title('Euler Angles vs Time')
    ax5.legend()
    ax5.grid(True)
    
    # ========== GIMBAL ANGLES VS TIME ==========
    ax6 = fig.add_subplot(3, 3, 6)
    gimbal_pitch = np.degrees(gimbal_angles[:, 0])
    gimbal_yaw = np.degrees(gimbal_angles[:, 1])
    
    ax6.plot(time, gimbal_pitch, 'r-', label='Gimbal Pitch', linewidth=1.5)
    ax6.plot(time, gimbal_yaw, 'b-', label='Gimbal Yaw', linewidth=1.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Gimbal Angle (degrees)')
    ax6.set_title('TVC Gimbal Actuation vs Time')
    ax6.legend()
    ax6.grid(True)
    
    # Check for saturation
    max_gimbal = 5.0  # degrees (typical limit)
    if np.max(np.abs(gimbal_pitch)) > max_gimbal * 0.95:
        ax6.axhline(y=max_gimbal, color='k', linestyle='--', label='Limit')
        ax6.axhline(y=-max_gimbal, color='k', linestyle='--')
    
    # ========== THRUST VS TIME ==========
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(time, thrust, 'orange', linewidth=2)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Thrust (N)')
    ax7.set_title('Thrust vs Time')
    ax7.grid(True)
    
    # ========== MASS VS TIME ==========
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.plot(time, mass, 'purple', linewidth=2)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Mass (kg)')
    ax8.set_title('Rocket Mass vs Time')
    ax8.grid(True)
    
    # ========== TRAJECTORY TOP VIEW ==========
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.plot(x, y, 'b-', linewidth=2)
    ax9.scatter(x[0], y[0], c='green', s=100, marker='o', label='Launch')
    ax9.scatter(x[-1], y[-1], c='red', s=100, marker='x', label='Landing')
    ax9.set_xlabel('North (m)')
    ax9.set_ylabel('East (m)')
    ax9.set_title('Trajectory Top View')
    ax9.legend()
    ax9.grid(True)
    ax9.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def print_summary(history):
    """
    Print summary statistics of the flight.
    
    Args:
        history: Dictionary with simulation history arrays
    """
    time = np.array(history['time'])
    position = np.array(history['position'])
    velocity = np.array(history['velocity'])
    
    # Calculate key metrics
    altitude = -position[:, 2]  # Negative z is altitude
    max_altitude = np.max(altitude)
    max_altitude_time = time[np.argmax(altitude)]
    
    velocity_magnitude = np.linalg.norm(velocity, axis=1)
    max_velocity = np.max(velocity_magnitude)
    max_velocity_time = time[np.argmax(velocity_magnitude)]
    
    flight_time = time[-1]
    
    # Downrange distance (horizontal)
    downrange = np.sqrt(position[-1, 0]**2 + position[-1, 1]**2)
    
    print("\n" + "="*60)
    print("FLIGHT SUMMARY")
    print("="*60)
    print(f"Flight Time:          {flight_time:.2f} s")
    print(f"Maximum Altitude:     {max_altitude:.2f} m (at t={max_altitude_time:.2f} s)")
    print(f"Maximum Velocity:     {max_velocity:.2f} m/s (at t={max_velocity_time:.2f} s)")
    print(f"Downrange Distance:   {downrange:.2f} m")
    print(f"Final Position:       N={position[-1, 0]:.2f} m, E={position[-1, 1]:.2f} m")
    
    # Landing velocity
    landing_velocity = velocity_magnitude[-1]
    print(f"Landing Velocity:     {landing_velocity:.2f} m/s")
    
    print("="*60 + "\n")
