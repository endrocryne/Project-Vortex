"""
GNC (Guidance, Navigation, Control) class for attitude control.
"""

import numpy as np
from .utils import quaternion_to_euler, quaternion_conjugate, quaternion_multiply


class GNC:
    """
    Guidance, Navigation, and Control system for TVC rocket.
    
    Implements PID controller for attitude stabilization using quaternion feedback.
    """
    
    def __init__(self, config):
        """
        Initialize GNC system with PID gains.
        
        Args:
            config: Dictionary containing PID gains and control parameters
        """
        # PID gains for pitch control
        self.kp_pitch = config.get('kp_pitch', 2.0)
        self.ki_pitch = config.get('ki_pitch', 0.1)
        self.kd_pitch = config.get('kd_pitch', 0.5)
        
        # PID gains for yaw control
        self.kp_yaw = config.get('kp_yaw', 2.0)
        self.ki_yaw = config.get('ki_yaw', 0.1)
        self.kd_yaw = config.get('kd_yaw', 0.5)
        
        # Integral error accumulators
        self.integral_pitch = 0.0
        self.integral_yaw = 0.0
        
        # Previous error for derivative
        self.prev_error_pitch = 0.0
        self.prev_error_yaw = 0.0
        
        # Control limits
        self.max_gimbal_angle = np.radians(config.get('max_gimbal_angle_deg', 5.0))
        
        # Target attitude (vertical: roll=0, pitch=0, yaw=0)
        self.target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        # Enable/disable control
        self.control_enabled = True
        self.control_start_time = config.get('control_start_time', 0.0)
    
    def set_target_attitude(self, target_quaternion):
        """
        Set target attitude quaternion.
        
        Args:
            target_quaternion: Desired quaternion [w, x, y, z]
        """
        # Ensure quaternion has non-zero norm before normalization
        norm = np.linalg.norm(target_quaternion)
        if norm < 1e-10:
            # Default to identity quaternion if invalid
            self.target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            self.target_quaternion = target_quaternion / norm
    
    def compute_control(self, current_quaternion, angular_velocity, dt, time):
        """
        Compute gimbal angles using PID control.
        
        Args:
            current_quaternion: Current attitude quaternion [w, x, y, z]
            angular_velocity: Current angular velocity in body frame [ωx, ωy, ωz]
            dt: Time step in seconds
            time: Current simulation time
        
        Returns:
            Tuple (gimbal_pitch, gimbal_yaw) in radians
        """
        # Check if control should be enabled
        if not self.control_enabled or time < self.control_start_time:
            return 0.0, 0.0
        
        # Compute attitude error quaternion: q_error = q_target^* ⊗ q_current
        q_target_conj = quaternion_conjugate(self.target_quaternion)
        q_error = quaternion_multiply(q_target_conj, current_quaternion)
        
        # Convert error quaternion to Euler angles for control
        # Small angle approximation: error_angles ≈ 2 * [q1, q2, q3] for small errors
        # For larger errors, use full conversion
        euler_error = quaternion_to_euler(q_error)
        roll_error, pitch_error, yaw_error = euler_error
        
        # PID control for pitch
        self.integral_pitch += pitch_error * dt
        # Anti-windup
        self.integral_pitch = np.clip(self.integral_pitch, -1.0, 1.0)
        
        derivative_pitch = (pitch_error - self.prev_error_pitch) / dt if dt > 0 else 0.0
        
        gimbal_pitch = -(
            self.kp_pitch * pitch_error +
            self.ki_pitch * self.integral_pitch +
            self.kd_pitch * derivative_pitch
        )
        
        # PID control for yaw
        self.integral_yaw += yaw_error * dt
        # Anti-windup
        self.integral_yaw = np.clip(self.integral_yaw, -1.0, 1.0)
        
        derivative_yaw = (yaw_error - self.prev_error_yaw) / dt if dt > 0 else 0.0
        
        gimbal_yaw = -(
            self.kp_yaw * yaw_error +
            self.ki_yaw * self.integral_yaw +
            self.kd_yaw * derivative_yaw
        )
        
        # Add damping term using angular velocity (rate feedback)
        # This helps reduce oscillations
        gimbal_pitch -= 0.1 * angular_velocity[1]  # ωy affects pitch
        gimbal_yaw -= 0.1 * angular_velocity[2]    # ωz affects yaw
        
        # Clamp to gimbal limits
        gimbal_pitch = np.clip(gimbal_pitch, -self.max_gimbal_angle, self.max_gimbal_angle)
        gimbal_yaw = np.clip(gimbal_yaw, -self.max_gimbal_angle, self.max_gimbal_angle)
        
        # Update previous errors
        self.prev_error_pitch = pitch_error
        self.prev_error_yaw = yaw_error
        
        return gimbal_pitch, gimbal_yaw
    
    def reset(self):
        """Reset integral and derivative terms."""
        self.integral_pitch = 0.0
        self.integral_yaw = 0.0
        self.prev_error_pitch = 0.0
        self.prev_error_yaw = 0.0
